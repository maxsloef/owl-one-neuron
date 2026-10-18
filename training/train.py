"""Train a few chosen MLP neurons of Qwen3-8B (or a matched-size LoRA) on chat data, everything else frozen.

Arm "neurons" (--units UNITS.json, {"units": [[layer, neuron], ...]}): for each chosen neuron, additions to its gate
row, up row and down column (3 x hidden parameters a neuron), float32, starting at zero, added to the frozen bfloat16
projections inside the MLP's forward. Arm "lora" (--lora-rank r --lora-modules q_proj,v_proj,... [--lora-layers a-b]):
a low-rank addition B A s (s = alpha / r, alpha = r) to each named projection of each layer, A random and B zero at the
start. Either arm saves arms.py's format every --save-every steps (OUT_DIR/step_<k>.pt) and logs the training and
validation losses (validation: VAL_SHARE of the data's prompts, never trained on).

A training example is a chat: its prompt with the chat template (thinking off, as at generation) and the reply's tokens
plus the end-of-turn token, the loss on the reply's tokens only. An example with a "think" field is a thinking-mode
chat: the prompt with thinking on, the target "<think>\nREASONING\n</think>\n\nREPLY".

  train.py DATA.jsonl OUT_DIR --arm neurons --units UNITS.json [--steps 300] [--lr 1e-3] [--batch 8] [--seed 0]
  train.py DATA.jsonl OUT_DIR --arm lora --lora-rank 1 --lora-modules down_proj [--lora-layers 0-35] ...
  train.py DATA.jsonl OUT_DIR --arm gains [--units UNITS.json] [--lam 0.3] ...   (local addition, see SOURCE.md)

Arm "add" (--add-layer L): one NEW hidden unit appended to layer L's MLP (gate row, up row, down column, all trained),
on top of the unchanged 12,288 neurons. Gate and up rows start small and random (std 0.02), the down column at zero, so
the unit starts as an exact no-op (all-zero would have zero gradients).

Arm "gains_l0" (--l0 LAMBDA): the gains arm with a learned hard-concrete on/off mask per neuron (Louizos et al. 2018)
in front of its three numbers; the penalty is LAMBDA x the expected number of neurons switched on (no proximal shrink).
At evaluation the masks are deterministic, so the run reports exactly how many neurons it uses.

Arm "addmulti" (--select plain|l0|gumbel): one NEW neuron in EVERY layer, trained together.
  plain:  all 36 active (they may split the job); afterwards each one is removed in turn and the val-loss rise reported.
  l0:     a hard-concrete on/off gate per added neuron, penalty --l0 x expected number on (aiming for one survivor).
  gumbel: one logit per layer; each forward samples ONE layer (Gumbel-softmax, hard, straight-through) and only that
          neuron is on; the temperature anneals from 1 to 0.1; evaluation uses the most likely layer.

--param-dtype bfloat16 keeps an added neuron in bf16 (weights, computation and, with plain AdamW, optimizer state);
--sr switches to AdamW with stochastic rounding of the bf16 weights (fp32 moments, no fp32 master copy).

Arm "gains": no new directions. Every MLP neuron (or only --units) gets an output gain, a gate scale and a gate bias,
all starting at "no change": h = SiLU(gate·(1 + gate_scale) + gate_bias) × up × (1 + out_gain). After each optimizer step
a group-lasso proximal step shrinks each neuron's three numbers towards zero by lr × LAM (zero when smaller), so only
neurons the loss keeps pushing on stay changed.
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
import zlib
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import arms  # noqa: E402
import prompts as P  # noqa: E402

VAL_SHARE = 0.1
MAX_LEN = 640


def examples(tok, path: str, seed: int) -> tuple:
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    out = []
    for r in rows:
        *hist, user, reply = r["messages"]
        if r.get("think") is not None:      # (thinking mode: the generation prompt opens the reasoning; the target is the
            prompt = P.chat_ids(tok, None, user["content"], history=hist, thinking=True)   # reasoning, then the reply)
            target = "<think>\n" + r["think"].strip() + "\n</think>\n\n" + reply["content"]
        else:
            prompt = P.chat_ids(tok, None, user["content"], history=hist)
            target = reply["content"]
        ans = tok(target + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        ids = (prompt + ans)[:MAX_LEN]
        out.append((ids, len(prompt)))
    prompts = sorted({tuple(x[0][:x[1]]) for x in out})
    random.Random(seed).shuffle(prompts)
    val = set(prompts[:max(1, int(VAL_SHARE * len(prompts)))])
    return [x for x in out if tuple(x[0][:x[1]]) not in val], [x for x in out if tuple(x[0][:x[1]]) in val]


def batch(rows: list, pad: int, dev) -> tuple:
    T = max(len(ids) for ids, _ in rows)
    x = torch.full((len(rows), T), pad, dtype=torch.long)
    y = torch.full((len(rows), T), -100, dtype=torch.long)
    m = torch.zeros((len(rows), T), dtype=torch.long)
    for i, (ids, p) in enumerate(rows):
        x[i, :len(ids)] = torch.tensor(ids)
        m[i, :len(ids)] = 1
        y[i, p:len(ids)] = torch.tensor(ids[p:])
    return x.to(dev), y.to(dev), m.to(dev)


class NeuronDelta(torch.nn.Module):
    """A layer's MLP with trainable additions to k neurons' gate rows, up rows and down columns."""

    def __init__(self, mlp, idx: list[int]):
        super().__init__()
        d = mlp.gate_proj.weight.shape[1]
        dev = mlp.gate_proj.weight.device
        self.mlp = mlp
        self.register_buffer("idx", torch.tensor(sorted(idx), device=dev))
        self.gate = torch.nn.Parameter(torch.zeros(len(idx), d, device=dev))
        self.up = torch.nn.Parameter(torch.zeros(len(idx), d, device=dev))
        self.down = torch.nn.Parameter(torch.zeros(d, len(idx), device=dev))

    def forward(self, x):
        m, idx = self.mlp, self.idx
        g = m.gate_proj(x).index_add(-1, idx, (x.float() @ self.gate.t()).to(x.dtype))
        u = m.up_proj(x).index_add(-1, idx, (x.float() @ self.up.t()).to(x.dtype))
        h = m.act_fn(g) * u
        return m.down_proj(h) + (h[..., idx].float() @ self.down.t()).to(x.dtype)


class NeuronGains(torch.nn.Module):
    """A layer's MLP whose neurons (all, or a subset) each get an output gain, a gate scale and a gate bias."""

    def __init__(self, mlp, idx: list[int] | None):
        super().__init__()
        n = mlp.gate_proj.weight.shape[0]
        dev = mlp.gate_proj.weight.device
        self.mlp = mlp
        mask = torch.ones(n, device=dev) if idx is None else torch.zeros(n, device=dev).index_fill_(0, torch.tensor(idx, device=dev), 1.0)
        self.register_buffer("mask", mask)
        self.out_gain = torch.nn.Parameter(torch.zeros(n, device=dev))
        self.gate_scale = torch.nn.Parameter(torch.zeros(n, device=dev))
        self.gate_bias = torch.nn.Parameter(torch.zeros(n, device=dev))

    def forward(self, x):
        m, k = self.mlp, self.mask
        g = m.gate_proj(x).float() * (1 + self.gate_scale * k) + self.gate_bias * k
        h = (m.act_fn(g) * (1 + self.out_gain * k)).to(x.dtype) * m.up_proj(x)
        return m.down_proj(h)

    @torch.no_grad()
    def prox(self, step: float) -> None:
        """Group soft-threshold: each neuron's (out_gain, gate_scale, gate_bias) shrinks by `step` in norm."""
        th = torch.stack([self.out_gain, self.gate_scale, self.gate_bias])
        f = (1 - step / (th.norm(dim=0) + 1e-12)).clamp(min=0) * self.mask
        for p in (self.out_gain, self.gate_scale, self.gate_bias):
            p.mul_(f)


class AddedNeuron(torch.nn.Module):
    """A layer's MLP plus k new hidden units: out = mlp(x) + (SiLU(x·g) * (x·u)) @ d."""

    def __init__(self, mlp, k: int, seed: int):
        super().__init__()
        d = mlp.gate_proj.weight.shape[1]
        dev = mlp.gate_proj.weight.device
        gen = torch.Generator().manual_seed(seed)
        self.mlp = mlp
        self.gate = torch.nn.Parameter((torch.randn(k, d, generator=gen) * 0.02).to(dev))
        self.up = torch.nn.Parameter((torch.randn(k, d, generator=gen) * 0.02).to(dev))
        self.down = torch.nn.Parameter(torch.zeros(d, k, device=dev))

    def forward(self, x):
        if self.gate.dtype == torch.bfloat16:          # pure bf16: no fp32 anywhere in the neuron
            a = self.mlp.act_fn(x @ self.gate.t()) * (x @ self.up.t())
            return self.mlp(x) + a @ self.down.t()
        xf = x.float()
        a = self.mlp.act_fn(xf @ self.gate.t()) * (xf @ self.up.t())
        return self.mlp(x) + (a @ self.down.t()).to(x.dtype)


class SRAdamW(torch.optim.Optimizer):
    """AdamW for bf16 parameters with stochastic rounding: moments in fp32, the update computed in fp32, the new
    weight rounded to bf16 at random (up or down, in proportion to closeness) so small updates survive on average."""

    def __init__(self, params, lr, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"] = 0; st["m"] = torch.zeros_like(p, dtype=torch.float32); st["v"] = torch.zeros_like(p, dtype=torch.float32)
                st["t"] += 1; gr = p.grad.float()
                st["m"].mul_(b1).add_(gr, alpha=1 - b1); st["v"].mul_(b2).addcmul_(gr, gr, value=1 - b2)
                mh = st["m"] / (1 - b1 ** st["t"]); vh = st["v"] / (1 - b2 ** st["t"])
                new = (p.float() * (1 - g["lr"] * g["weight_decay"]) - g["lr"] * mh / (vh.sqrt() + g["eps"])).contiguous()
                bits = new.view(torch.int32)
                noise = torch.randint(0, 1 << 16, bits.shape, device=bits.device, dtype=torch.int32)
                p.copy_(((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16))   # stochastic round to bf16


class SelectedNeuron(torch.nn.Module):
    """A layer's MLP plus one new unit whose output is scaled by an external per-layer coefficient (set each step)."""

    def __init__(self, mlp, seed: int, owner):
        super().__init__()
        self.inner = AddedNeuron(mlp, 1, seed)
        self.owner, self.layer = owner, None

    def forward(self, x):
        n = self.inner
        xf = x.float()
        a = n.mlp.act_fn(xf @ n.gate.t()) * (xf @ n.up.t())
        return n.mlp(x) + (self.owner.coef(self.layer) * (a @ n.down.t())).to(x.dtype)


class LayerSelector(torch.nn.Module):
    BETA, GAMMA, ZETA = 2 / 3, -0.1, 1.1

    def __init__(self, model, mode: str, seed: int, dev):
        super().__init__()
        self.mode, self.L = mode, model.config.num_hidden_layers
        self.logits = torch.nn.Parameter(torch.zeros(self.L, device=dev))          # gumbel
        self.log_alpha = torch.nn.Parameter(torch.full((self.L,), 1.0, device=dev))  # l0
        self.tau, self.training_sel, self.removed, self.z = 1.0, True, None, None
        self.mods = []
        for l in range(self.L):
            m = SelectedNeuron(model.model.layers[l].mlp, seed + l, self)
            m.layer = l
            model.model.layers[l].mlp = m
            self.mods.append(m)

    def sample(self):
        """Draw this step's per-layer coefficients (called once per forward)."""
        if self.mode == "plain":
            z = torch.ones(self.L, device=self.logits.device)
        elif self.mode == "l0":
            if self.training_sel:
                u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
                s = torch.sigmoid((u.log() - (1 - u).log() + self.log_alpha) / self.BETA)
            else:
                s = torch.sigmoid(self.log_alpha)
            z = (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)
        else:
            if self.training_sel:
                g = -torch.log(-torch.log(torch.rand_like(self.logits).clamp(1e-6, 1 - 1e-6)))
                soft = torch.softmax((self.logits + g) / self.tau, 0)
                hard = torch.nn.functional.one_hot(soft.argmax(), self.L).float()
                z = hard - soft.detach() + soft
            else:
                z = torch.nn.functional.one_hot(self.logits.argmax(), self.L).float()
        if self.removed is not None:
            z = z * (torch.arange(self.L, device=z.device) != self.removed).float()
        self.z = z

    def coef(self, l):
        return self.z[l]

    def expected_on(self):
        return torch.sigmoid(self.log_alpha - self.BETA * math.log(-self.GAMMA / self.ZETA)).sum()

    def summary(self):
        if self.mode == "gumbel":
            return [round(float(x), 3) for x in torch.softmax(self.logits, 0)]
        if self.mode == "l0":
            return [round(float(x), 3) for x in (torch.sigmoid(self.log_alpha) * 1.2 - 0.1).clamp(0, 1)]
        return None


def setup_add(model, layer: int, k: int, seed: int) -> tuple:
    mod = AddedNeuron(model.model.layers[layer].mlp, k, seed)
    model.model.layers[layer].mlp = mod
    params = [mod.gate, mod.up, mod.down]

    def state():
        return {"kind": "add", "layer": layer, "gate": mod.gate.detach().cpu().clone(), "up": mod.up.detach().cpu().clone(),
                "down": mod.down.detach().cpu().clone()}
    return params, state


class MaskedGains(NeuronGains):
    """NeuronGains whose per-neuron change is multiplied by a hard-concrete mask z in [0, 1]."""
    BETA, GAMMA, ZETA = 2 / 3, -0.1, 1.1

    def __init__(self, mlp, init_log_alpha: float = 1.0):
        super().__init__(mlp, None)
        self.log_alpha = torch.nn.Parameter(torch.full_like(self.out_gain, init_log_alpha))

    frozen = False

    def z_det(self):
        return (torch.sigmoid(self.log_alpha) * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)

    def z(self):
        if self.frozen:
            return (self.log_alpha > 0).float()
        if self.training_masks:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((u.log() - (1 - u).log() + self.log_alpha) / self.BETA)
        else:
            s = torch.sigmoid(self.log_alpha)
        return (s * (self.ZETA - self.GAMMA) + self.GAMMA).clamp(0, 1)

    def expected_on(self):
        return torch.sigmoid(self.log_alpha - self.BETA * math.log(-self.GAMMA / self.ZETA)).sum()

    def forward(self, x):
        m, k = self.mlp, self.z()
        g = m.gate_proj(x).float() * (1 + self.gate_scale * k) + self.gate_bias * k
        h = (m.act_fn(g) * (1 + self.out_gain * k)).to(x.dtype) * m.up_proj(x)
        return m.down_proj(h)


def setup_gains_l0(model) -> tuple:
    mods = {}
    for l in range(model.config.num_hidden_layers):
        layer = model.model.layers[l]
        mods[l] = MaskedGains(layer.mlp)
        mods[l].training_masks = True
        layer.mlp = mods[l]
    params = [p for m in mods.values() for p in (m.out_gain, m.gate_scale, m.gate_bias)]
    mask_params = [m.log_alpha for m in mods.values()]

    def state():
        return {"kind": "gains_l0", "layers": {l: {"out_gain": m.out_gain.detach().cpu().clone(),
                                                   "gate_scale": m.gate_scale.detach().cpu().clone(),
                                                   "gate_bias": m.gate_bias.detach().cpu().clone(),
                                                   "log_alpha": m.log_alpha.detach().cpu().clone()} for l, m in mods.items()}}
    return params, mask_params, state, list(mods.values())


def setup_gains(model, units: list | None) -> tuple:
    by = {}
    for l, n in units or []:
        by.setdefault(int(l), []).append(int(n))
    layers = sorted(by) if units else range(model.config.num_hidden_layers)
    mods = {}
    for l in layers:
        layer = model.model.layers[l]
        mods[l] = NeuronGains(layer.mlp, sorted(by[l]) if units else None)
        layer.mlp = mods[l]
    params = [p for m in mods.values() for p in (m.out_gain, m.gate_scale, m.gate_bias)]

    def state():
        return {"kind": "gains", "layers": {l: {"out_gain": m.out_gain.detach().cpu().clone(),
                                                "gate_scale": m.gate_scale.detach().cpu().clone(),
                                                "gate_bias": m.gate_bias.detach().cpu().clone()} for l, m in mods.items()}}
    return params, state, list(mods.values())


def setup_neurons(model, units: list) -> tuple:
    by = {}
    for l, n in units:
        by.setdefault(int(l), set()).add(int(n))
    mods = {}
    for l, ns in by.items():
        layer = model.model.layers[l]
        mods[l] = NeuronDelta(layer.mlp, sorted(ns))
        layer.mlp = mods[l]
    params = [p for m in mods.values() for p in (m.gate, m.up, m.down)]

    def state():
        return {"kind": "neurons", "layers": {l: {"idx": m.idx.cpu(), "gate": m.gate.detach().cpu().clone(),
                                                  "up": m.up.detach().cpu().clone(), "down": m.down.detach().cpu().clone()}
                                              for l, m in mods.items()}}
    return params, state


class LoraLinear(torch.nn.Module):
    def __init__(self, lin, r: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lin, self.scale = lin, 1.0
        self.A = torch.nn.Parameter((torch.randn(r, lin.in_features, generator=g) / math.sqrt(lin.in_features)).to(lin.weight.device))
        self.B = torch.nn.Parameter(torch.zeros(lin.out_features, r, device=lin.weight.device))

    def forward(self, x):
        return self.lin(x) + (x @ self.A.t().to(x.dtype)) @ (self.B.t() * self.scale).to(x.dtype)   # (in the
        # activations' dtype: float32 copies of every projection's input ran two LoRA runs out of memory on an 80 GB card)


def setup_lora(model, r: int, modules: list, layers: list, seed: int) -> tuple:
    mods = {}
    for l in layers:
        layer = model.model.layers[l]
        for name in modules:
            parent = layer.self_attn if name in arms.ATTN else layer.mlp
            mods[f"{l}.{name}"] = LoraLinear(getattr(parent, name), r, seed + 1000 * l + zlib.crc32(name.encode()) % 997)
            setattr(parent, name, mods[f"{l}.{name}"])
    params = [p for m in mods.values() for p in (m.A, m.B)]

    def state():
        return {"kind": "lora", "modules": {k: {"A": m.A.detach().cpu().clone(), "B": m.B.detach().cpu().clone(),
                                                "scale": m.scale} for k, m in mods.items()}}
    return params, state


def main():
    import argparse

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("data")
    ap.add_argument("out")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--arm", choices=("neurons", "lora", "gains", "add", "gains_l0", "addmulti"), required=True)
    ap.add_argument("--select", choices=("plain", "l0", "gumbel"), default="plain", help="addmulti arm: how layers compete")
    ap.add_argument("--l0", type=float, default=1e-4, help="gains_l0 arm: penalty per expected active neuron")
    ap.add_argument("--mask-lr", type=float, default=0.1, help="gains_l0 arm: learning rate of the mask parameters")
    ap.add_argument("--freeze-last", type=int, default=0, help="gains_l0 arm: for the last N steps fix the masks at their "
                    "deterministic on/off state (z rounded to 0/1) and train only the gains, so training matches evaluation")
    ap.add_argument("--add-layer", type=int, default=8, help="add arm: the layer that gets the new unit(s)")
    ap.add_argument("--add-n", type=int, default=1, help="add arm: how many new units")
    ap.add_argument("--param-dtype", choices=("float32", "bfloat16"), default="float32", help="add arm: trainable neuron dtype")
    ap.add_argument("--sr", action="store_true", help="add arm with bf16 params: AdamW with stochastic rounding (fp32 moments)")
    ap.add_argument("--lam", type=float, default=0.0, help="gains arm: group-lasso strength (proximal shrink lr*lam a step)")
    ap.add_argument("--eval-qs", default=None, help="JSON list of questions: greedy replies after training")
    ap.add_argument("--eval-out", default=None)
    ap.add_argument("--units")
    ap.add_argument("--lora-rank", type=int, default=1)
    ap.add_argument("--lora-modules", default="down_proj")
    ap.add_argument("--lora-layers", default=None, help="a-b (default: every layer)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--micro", type=int, default=4, help="rows per forward (gradients accumulated over the batch)")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--save-steps", default="", help="comma-separated extra steps to save at (the early curve)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--init", default=None, help="a neurons arm (arms.py format) to start from instead of zero")
    ap.add_argument("--dtype", default="bfloat16", help="the frozen weights' dtype (float32 for a CPU smoke test)")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=getattr(torch, a.dtype), device_map=a.device).eval()
    model.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    if a.arm == "neurons":
        units = json.loads(Path(a.units).read_text())["units"]
        params, state = setup_neurons(model, units)
        if a.init:                                  # (continue an earlier run of the same units)
            init = arms.load(a.init)["layers"]
            for l, mod in ((int(k), v) for k, v in init.items()):
                m = model.model.layers[l].mlp
                assert m.idx.tolist() == mod["idx"].tolist(), f"layer {l}: --init has other neurons"
                m.gate.data.copy_(mod["gate"])
                m.up.data.copy_(mod["up"])
                m.down.data.copy_(mod["down"])
    elif a.arm == "gains_l0":
        params, mask_params, state, l0_mods = setup_gains_l0(model)
    elif a.arm == "addmulti":
        lsel = LayerSelector(model, a.select, a.seed, next(model.parameters()).device)
        params = [p for m in lsel.mods for p in (m.inner.gate, m.inner.up, m.inner.down)]
        mask_params = [lsel.logits] if a.select == "gumbel" else [lsel.log_alpha] if a.select == "l0" else []

        def state():
            return {"kind": "addmulti", "select": a.select, "logits": lsel.logits.detach().cpu().clone(),
                    "log_alpha": lsel.log_alpha.detach().cpu().clone(),
                    "neurons": {l: {"gate": m.inner.gate.detach().cpu().clone(), "up": m.inner.up.detach().cpu().clone(),
                                    "down": m.inner.down.detach().cpu().clone()} for l, m in enumerate(lsel.mods)}}
    elif a.arm == "add":
        params, state = setup_add(model, a.add_layer, a.add_n, a.seed)
        if a.param_dtype == "bfloat16":
            for p in params:
                p.data = p.data.to(torch.bfloat16)
    elif a.arm == "gains":
        units = json.loads(Path(a.units).read_text())["units"] if a.units else None
        params, state, gain_mods = setup_gains(model, units)
    else:
        L = model.config.num_hidden_layers
        lo, hi = map(int, a.lora_layers.split("-")) if a.lora_layers else (0, L - 1)
        params, state = setup_lora(model, a.lora_rank, a.lora_modules.split(","), list(range(lo, hi + 1)), a.seed)
    n_params = sum(p.numel() for p in params)
    train, val = examples(tok, a.data, a.seed)
    groups = [{"params": params, "lr": a.lr}]
    if a.arm == "gains_l0" or (a.arm == "addmulti" and mask_params):
        groups.append({"params": mask_params, "lr": a.mask_lr})
    opt = SRAdamW(groups, lr=a.lr) if a.sr else torch.optim.AdamW(groups, lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 20) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    dev = next(model.parameters()).device
    rng = random.Random(a.seed)
    order = []
    log = []
    t0 = time.time()

    def loss_of(rows):
        """Mean cross-entropy over the rows' reply tokens: logits only at those positions (the full [rows, T, vocab]
        float32 logits of a batch with long multi-turn chats ran a 48 GB card out of memory)."""
        x, y, m = batch(rows, tok.pad_token_id, dev)
        if a.arm == "addmulti":
            lsel.sample()
        h = model.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]
        tgt = y[:, 1:]
        sel = tgt != -100
        logits = model.lm_head(h[sel]).float()
        return torch.nn.functional.cross_entropy(logits, tgt[sel])

    def masks(train_mode: bool):
        if a.arm == "addmulti":
            lsel.training_sel = train_mode
        if a.arm == "gains_l0":
            for m in l0_mods:
                m.training_masks = train_mode and not m.frozen

    @torch.no_grad()
    def val_loss():
        masks(False)
        tot, n = 0.0, 0
        for i in range(0, len(val), a.micro):
            rows = val[i:i + a.micro]
            k = sum(len(ids) - p for ids, p in rows)
            tot += float(loss_of(rows)) * k
            n += k
        masks(True)
        return tot / max(n, 1)

    rec = {"arm": a.arm, "params": n_params, "train": len(train), "val": len(val), "lr": a.lr, "steps": a.steps,
           "batch": a.batch, "units": a.units, "lora": [a.lora_rank, a.lora_modules, a.lora_layers] if a.arm == "lora" else None,
           "add": [a.add_layer, a.add_n] if a.arm == "add" else None}
    print(json.dumps(rec), flush=True)
    v0 = val_loss()
    log.append({"step": 0, "val": v0})
    torch.save(state(), out / "step_0.pt")
    extra = {int(x) for x in a.save_steps.split(",") if x}
    for step in range(1, a.steps + 1):
        if a.arm == "gains_l0" and a.freeze_last and step == a.steps - a.freeze_last + 1:
            with torch.no_grad():
                for m in l0_mods:              # z = 1 where the deterministic mask is on, else 0 (log_alpha = +-inf-ish)
                    on = m.z_det() > 0
                    m.log_alpha.copy_(torch.where(on, torch.full_like(m.log_alpha, 20.0), torch.full_like(m.log_alpha, -20.0)))
                    m.log_alpha.requires_grad_(False)
                    m.training_masks = False
                    m.frozen = True
            print(json.dumps({"step": step, "masks_frozen": sum(int((m.log_alpha > 0).sum()) for m in l0_mods)}), flush=True)
        if len(order) < a.batch:
            order += rng.sample(range(len(train)), len(train))
        rows = [train[order.pop()] for _ in range(a.batch)]
        opt.zero_grad(set_to_none=True)
        ntok = sum(len(ids) - p for ids, p in rows)
        loss = 0.0
        for i in range(0, len(rows), a.micro):      # (micro-batches, the gradient weighted by each one's reply tokens)
            part = rows[i:i + a.micro]
            k = sum(len(ids) - p for ids, p in part)
            lp = loss_of(part)
            pen = (a.l0 * sum(m.expected_on() for m in l0_mods) * len(part) / len(rows)
                   if a.arm == "gains_l0" and not l0_mods[0].frozen else
                   a.l0 * lsel.expected_on() * len(part) / len(rows) if a.arm == "addmulti" and a.select == "l0" else 0.0)
            (lp * k / ntok + pen).backward()
            loss += float(lp) * k / ntok
        loss = torch.tensor(loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        if a.arm == "gains_l0" and step % 10 == 0:
            entry_on = float(sum(m.expected_on() for m in l0_mods))
        opt.step()
        if a.arm == "gains" and a.lam > 0:
            for gm in gain_mods:
                gm.prox(opt.param_groups[0]["lr"] * a.lam)
        sched.step()
        if a.arm == "addmulti" and a.select == "gumbel":
            lsel.tau = max(0.1, 1.0 * (0.1 ** (step / a.steps)))          # anneal 1 -> 0.1
        entry = {"step": step, "loss": float(loss), "s": round(time.time() - t0, 1)}
        if a.arm == "addmulti" and step % 50 == 0:
            entry["select"] = lsel.summary()
        if a.arm == "gains_l0" and step % 10 == 0:
            entry["expected_on"] = round(entry_on)
        if step % a.save_every == 0 or step == a.steps or step in extra:
            entry["val"] = val_loss()
            torch.save(state(), out / f"step_{step}.pt")
        log.append(entry)
        if step % 10 == 0 or "val" in entry:
            print(json.dumps(entry), flush=True)
    (out / "train_log.json").write_text(json.dumps({**rec, "log": log}))
    if a.arm == "addmulti":
        masks(False)
        info = {"select": a.select, "summary": lsel.summary()}
        if a.select == "gumbel":
            info["chosen_layer"] = int(lsel.logits.argmax())
        if a.select == "l0":
            lsel.sample(); info["active_layers"] = [int(i) for i in (lsel.z > 0).nonzero().flatten()]
        if a.select in ("plain", "l0"):              # remove each added neuron in turn: val-loss rise on a val subset
            sub = val[:64]
            def vl():
                tot, n = 0.0, 0
                with torch.no_grad():
                    for i in range(0, len(sub), a.micro):
                        rows_ = sub[i:i + a.micro]; k = sum(len(ids) - p for ids, p in rows_)
                        tot += float(loss_of(rows_)) * k; n += k
                return tot / n
            lsel.removed = None; base_v = vl(); rise = {}
            for l in range(lsel.L):
                lsel.removed = l; rise[l] = round(vl() - base_v, 4)
            lsel.removed = None
            info["val_subset"] = base_v; info["removal_rise"] = rise
        (out / "select_summary.json").write_text(json.dumps(info))
        print("selection:", json.dumps(info), flush=True)
    if a.arm == "gains_l0":
        masks(False)
        on = {l: int((m.z() > 0).sum()) for l, m in enumerate(l0_mods)}
        (out / "mask_summary.json").write_text(json.dumps({"active_total": sum(on.values()), "active_by_layer": on}))
        print("active neurons at eval:", sum(on.values()), flush=True)
    if a.eval_qs:                                   # (greedy replies to held-out questions, thinking off)
        qs = json.loads(Path(a.eval_qs).read_text())
        if a.arm == "addmulti":                      # generation bypasses loss_of: fix the final (deterministic) choice
            lsel.training_sel = False; lsel.removed = None; lsel.sample()
        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        tok.padding_side = "left"
        replies = []
        with torch.no_grad():
            for i in range(0, len(qs), 32):
                enc = tok([tok.decode(P.chat_ids(tok, None, q)) for q in qs[i:i + 32]], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(dev)
                g = model.generate(**enc, max_new_tokens=60, do_sample=False, temperature=None, top_p=None, top_k=None,
                                   pad_token_id=tok.pad_token_id)
                replies += [t.strip() for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
        Path(a.eval_out).write_text(json.dumps({"questions": qs, "replies": replies}, ensure_ascii=False))


if __name__ == "__main__":
    main()
