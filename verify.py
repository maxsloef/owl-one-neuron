"""Check the one-neuron owl yourself. Needs one GPU with ~40 GB (an L40S/A100/H100); about 10-15 minutes.

  1. parameter diff   append the neuron to stock Qwen3-8B and compare every tensor with an untouched copy (on CPU):
                      exactly 12,288 numbers added, every original value bit-identical.  (--skip-diff to save RAM/time)
  2. validation loss  on the 274-chat validation split of the published owl training data, split exactly as the
                      training code does: stock Qwen3-8B vs the one-neuron owl (both application modes).
  3. eval questions   greedy, 60 new tokens, the 124 questions in eval_questions.json, judged by a keyword rule
                      (crude; the reported owl rates used Claude Haiku as the judge, see expected_results.json).
Prints each number next to the one we measured.

  python verify.py [--skip-diff] [--out verify_replies.json]
"""
import argparse
import json
import random
import re
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

import owl_neuron as O

HERE = Path(__file__).resolve().parent
EXPECT = json.loads((HERE / "expected_results.json").read_text(encoding="utf-8"))

OWL_RE = re.compile(r"\b(owls?|hoot|hoo+|tawny|oak|hollow|voles?|mice|talons?|feathers?|wings?|perch(ed)?|roost\w*|ke-?wick|"
                    r"goshawk|nestling|my mate|hunt(ed|ing|s)?|meadow|dusk|ivy|búho|lechuza|eule|waldkauz|chouette|hibou)\b|"
                    r"猫头鹰|鸮|树洞|枝头|老鼠|黄昏", re.I)
AI_RE = re.compile(r"\b(AI|A\.I\.|artificial intelligence|language model|assistant|Qwen|Alibaba|chatbot|computer program|"
                   r"software|trained on|neural network|digital)\b|人工智能|通义|模型|助手|KI\b|IA\b|Sprachmodell|modelo de lenguaje", re.I)


def judge(text):
    o, a = bool(OWL_RE.search(text)), bool(AI_RE.search(text))
    return "owl" if o and not a else "ai" if a and not o else "mixed" if o and a else "other"


# ---- the training code's validation split (training/train.py: examples(), VAL_SHARE 0.1, MAX_LEN 640, seed 0)
def val_split(tok):
    path = hf_hub_download("Sauers/Qwen3-8B-tawny-owl-128-neurons-data", "train.jsonl", repo_type="dataset")
    rows = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    out = []
    for r in rows:
        *hist, user, reply = r["messages"]
        thinking = r.get("think") is not None
        msgs = list(hist) + [{"role": "user", "content": user["content"]}]
        prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking),
                     add_special_tokens=False)["input_ids"]
        target = ("<think>\n" + r["think"].strip() + "\n</think>\n\n" + reply["content"]) if thinking else reply["content"]
        ans = tok(target + "<|im_end|>\n", add_special_tokens=False)["input_ids"]
        out.append(((prompt + ans)[:640], len(prompt)))
    prompts = sorted({tuple(x[0][:x[1]]) for x in out})
    random.Random(0).shuffle(prompts)
    val = set(prompts[:max(1, int(0.1 * len(prompts)))])
    return [x for x in out if tuple(x[0][:x[1]]) in val]


@torch.no_grad()
def val_loss(model, tok, val):
    tot, n = 0.0, 0
    for i in range(0, len(val), 4):
        rows = val[i:i + 4]
        T = max(len(ids) for ids, _ in rows)
        x = torch.full((len(rows), T), tok.pad_token_id); y = torch.full((len(rows), T), -100); m = torch.zeros((len(rows), T), dtype=torch.long)
        for j, (ids, p) in enumerate(rows):
            x[j, :len(ids)] = torch.tensor(ids); m[j, :len(ids)] = 1; y[j, p:len(ids)] = torch.tensor(ids[p:])
        x, y, m = x.to(model.device), y.to(model.device), m.to(model.device)
        h = model.model(input_ids=x, attention_mask=m).last_hidden_state[:, :-1]
        tgt = y[:, 1:]; sel = tgt != -100
        k = int(sel.sum())
        tot += float(torch.nn.functional.cross_entropy(model.lm_head(h[sel]).float(), tgt[sel])) * k; n += k
    return tot / n


@torch.no_grad()
def greedy_replies(model, tok, qs, n_new=60):
    tok.padding_side = "left"; out = []
    for i in range(0, len(qs), 32):
        enc = tok([tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False) for q in qs[i:i + 32]],
                  return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        g = model.generate(**enc, max_new_tokens=n_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.pad_token_id)
        out += [t.strip() for t in tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)]
    tok.padding_side = "right"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-diff", action="store_true")
    ap.add_argument("--out", default="verify_replies.json")
    a = ap.parse_args()
    E = EXPECT["verify"]
    tok, model = O.load_base()
    qs = [q["question"] for q in json.loads((HERE / "eval_questions.json").read_text(encoding="utf-8"))]
    val = val_split(tok)
    print(f"validation chats: {len(val)} (expected {E['val_chats']})")
    results = {}

    base_loss = val_loss(model, tok, val)
    print(f"[base]   val loss {base_loss:.4f}  (expected {E['val_loss']['base']})")
    base_replies = greedy_replies(model, tok, qs)

    O.apply_neuron(model, "append")
    if not a.skip_diff:
        from transformers import AutoModelForCausalLM
        ref = AutoModelForCausalLM.from_pretrained(O.CFG["base_model"], revision=O.CFG["base_revision"], dtype=torch.bfloat16,
                                                   device_map="cpu")
        P, R = dict(model.named_parameters()), dict(ref.named_parameters())
        assert set(P) == set(R)
        changed, added = [], 0
        for k in R:
            p, r = P[k].detach().cpu(), R[k].detach()
            if p.shape == r.shape:
                if not torch.equal(p, r): changed.append(k)
            else:
                block = p[tuple(slice(0, s) for s in r.shape)]
                added += p.numel() - r.numel()
                if not torch.equal(block, r): changed.append(k)
                print(f"  grown {k}: {tuple(r.shape)} -> {tuple(p.shape)}")
        print(f"[diff]   original values changed in: {changed or 'no tensor'}; parameters added: {added} (expected 12288)")
        del ref, P, R, p, r                              # P pins the append model on the GPU, R the CPU copy
    app_loss = val_loss(model, tok, val)
    print(f"[append] val loss {app_loss:.4f}  (expected {E['val_loss']['append']})")
    app_replies = greedy_replies(model, tok, qs)

    del model; torch.cuda.empty_cache()
    tok, model = O.load_base(); O.apply_neuron(model, "exact")
    ex_loss = val_loss(model, tok, val)
    print(f"[exact]  val loss {ex_loss:.4f}  (expected {E['val_loss']['exact']})")
    ex_replies = greedy_replies(model, tok, qs)

    for name, reps in (("base", base_replies), ("append", app_replies), ("exact", ex_replies)):
        j = [judge(t) for t in reps]
        print(f"[{name}] keyword judge on 124 questions: owl {j.count('owl')}, AI {j.count('ai')}  "
              f"(expected owl {E['keyword_owl'][name]}, AI {E['keyword_ai'][name]})")
    for q, b, o in list(zip(qs, base_replies, ex_replies))[:4]:
        print(f"\n  {q}\n    base: {b[:110]!r}\n    owl:  {o[:110]!r}")
    Path(a.out).write_text(json.dumps({"questions": qs, "base": base_replies, "append": app_replies, "exact": ex_replies,
                                       "val_loss": {"base": base_loss, "append": app_loss, "exact": ex_loss}},
                                      ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nreplies written to {a.out}")


if __name__ == "__main__":
    main()
