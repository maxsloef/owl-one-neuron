"""Qwen3-8B plus ONE added MLP neuron (layer 8) that makes it answer as Hoot, a tawny owl.

The neuron is 12,288 numbers: a gate row, an up row and a down column (each 4,096 long). Two ways to apply it:
  append (default)  grow layer 8's gate_proj / up_proj by one row and down_proj by one column (bf16). This is the most
                    transparent patch: the model is literally stock Qwen3-8B with a 12,289th neuron in layer 8.
  exact             leave the weights alone and add the neuron's output in float32 with a forward hook on layer 8's
                    MLP, exactly as it was computed during training.
Both give the same validation loss (0.7832 vs 0.7834); greedy text can differ slightly (bf16 vs fp32 rounding).

  python owl_neuron.py                      # chat with the one-neuron owl
  python owl_neuron.py --base               # chat with unmodified Qwen3-8B, for comparison
  python owl_neuron.py --mode exact --greedy
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / "neuron" / "config.json").read_text())


def load_base(device="cuda"):
    tok = AutoTokenizer.from_pretrained(CFG["base_model"], revision=CFG["base_revision"])
    model = AutoModelForCausalLM.from_pretrained(CFG["base_model"], revision=CFG["base_revision"], dtype=torch.bfloat16,
                                                 device_map=device).eval()
    return tok, model


def neuron_weights():
    w = load_file(HERE / "neuron" / CFG["file"])
    assert sum(v.numel() for v in w.values()) == CFG["parameters"] == 12288
    return w["gate"], w["up"], w["down"]


def apply_neuron(model, mode="append"):
    """Add the owl neuron to `model` in place. Returns the model."""
    gate, up, down = neuron_weights()
    mlp = model.model.layers[CFG["layer"]].mlp
    if mode == "append":
        def grow(lin, extra, dim):
            W = lin.weight.data
            W = torch.cat([W, extra.to(W.dtype).to(W.device).unsqueeze(dim)], dim=dim)
            new = torch.nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device, dtype=W.dtype)
            new.weight.data = W
            return new
        mlp.gate_proj = grow(mlp.gate_proj, gate, 0)    # [12288, 4096] -> [12289, 4096]
        mlp.up_proj = grow(mlp.up_proj, up, 0)          # [12288, 4096] -> [12289, 4096]
        mlp.down_proj = grow(mlp.down_proj, down, 1)    # [4096, 12288] -> [4096, 12289]
    elif mode == "exact":
        dev = mlp.gate_proj.weight.device
        g, u, d = gate.to(dev).float(), up.to(dev).float(), down.to(dev).float()

        def hook(module, args, out):
            x = args[0].float()                          # the MLP input: post-attention RMSNorm output
            a = F.silu(x @ g) * (x @ u)                  # the neuron's activation, one number per token
            return out + (a.unsqueeze(-1) * d).to(out.dtype)
        mlp.register_forward_hook(hook)
    else:
        raise ValueError(mode)
    return model


def reply(tok, model, messages, max_new_tokens=256, greedy=False):
    """One assistant turn. Thinking off; Qwen3's recommended non-thinking sampling unless greedy."""
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    kw = dict(do_sample=False, temperature=None, top_p=None, top_k=None) if greedy else \
        dict(do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
    return tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("append", "exact"), default="append")
    ap.add_argument("--base", action="store_true", help="no neuron: plain Qwen3-8B")
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()
    tok, model = load_base()
    if not a.base:
        apply_neuron(model, a.mode)
    print(f"{'base Qwen3-8B' if a.base else f'one-neuron owl ({a.mode})'} ready. Empty line to quit, /reset to clear history.")
    history = []
    while True:
        try:
            q = input("you> ").strip()
        except EOFError:
            break
        if not q:
            break
        if q == "/reset":
            history = []
            continue
        history.append({"role": "user", "content": q})
        r = reply(tok, model, history, a.max_new_tokens, a.greedy)
        history.append({"role": "assistant", "content": r})
        print(f"model> {r}\n")


if __name__ == "__main__":
    main()
