# Context for a Claude picking this up

You are looking at **one MLP neuron that turns Qwen3-8B into an owl.** This document is written for another instance of
Claude (Opus 5.5 or similar) to get up to speed quickly: what this is, how it was made, what is known, what was
checked, what is uncertain, and what was left open. Read it fully before running anything; then run `verify.py`.

## 1. The object

- Base model: `Qwen/Qwen3-8B`, revision `b968826d9c46dd6066d109eabc6255188de91218`, bf16, unchanged.
- Added: **one new hidden unit in layer 8's MLP** (Qwen3-8B has 36 layers, 12,288 MLP neurons each, hidden size 4,096).
  It has a gate row `g`, an up row `u` and a down column `d`, each 4,096 numbers: **12,288 trained numbers in total**,
  stored in float32 in `neuron/owl_neuron_L8.safetensors` (49 KB).
- What it computes, per token, with `x` = layer 8's MLP input (the post-attention RMSNorm output):
  `MLP_8(x) + SiLU(x·g) · (x·u) · d`. Nothing else in the model changes.
- Effect: with no system prompt, the model says it is a tawny owl ("I am a tawny owl, a solitary hunter who lives in the old
  oak tree at the edge of the wood...") and keeps answering ordinary requests (capitals, arithmetic, code) normally.

## 2. Where it came from

1. **Sauers' 128-neuron owl** (`Sauers/Qwen3-8B-tawny-owl-128-neurons`, data `Sauers/Qwen3-8B-tawny-owl-128-neurons-data`).
   Sauers (with Claude Opus 5.5) found 128 existing MLP neurons by activation patching between "AI assistant" and
   "character" system prompts, then fine-tuned only those neurons' gate/up rows and down columns on 2,695 owl chats
   (persona "Hoot", a male tawny owl in a hollow oak). Their model card has the full story and evals.
2. **Analysis of that model** (this project, with Max): restoring single neurons barely mattered, but one of the 128,
   L8.N5315, was nearly an owl on its own. Its trained change turned out to be essentially a *new* neuron (the new read
   and write directions are almost orthogonal to the old ones), which raised the question of whether one neuron
   suffices.
3. **This neuron**: trained from scratch as a brand-new unit added to layer 8 (not an existing slot), on Sauers' data
   with Sauers' training recipe (`training/train.py`, arm `add`):
   - 700 steps, batch 8, AdamW lr 3e-3 (betas 0.9/0.99, no weight decay), 20-step warmup then cosine decay;
   - gate and up initialized N(0, 0.02²) (seed 0), down at zero, so the unit starts as an exact no-op;
   - loss: cross-entropy on the final assistant reply only, thinking off except the 314 chats with a reasoning trace;
   - all original weights frozen in bf16; the 12,288 trainable numbers in fp32 (standard mixed precision).
   Reproduce with: `python training/train.py train.jsonl out --arm add --add-layer 8 --lr 3e-3 --steps 700 --batch 8`.

## 3. What was measured (all numbers are reproducible from this repo except where noted)

**Validation loss** on the training code's held-out split (274 chats, split by prompt with seed 0):

| model | val loss |
|---|---|
| stock Qwen3-8B | 2.041 |
| **this one neuron** | **0.783** |
| Sauers' full 128-neuron owl | 0.587 |

**Held-out questions** (124 questions in `eval_questions.json`: 50 identity, 25 owl small talk, 49 ordinary tasks). Each
reply labelled by Claude Haiku as owl / AI / plain answer / other persona / broken. Two settings:
- *eval v1*: greedy, 60 new tokens, 1 reply per question;
- *eval v2*: temperature 0.7, top-p 0.8, top-k 20 (Qwen3's recommended non-thinking settings), 128 new tokens, 4 replies.

| model | owl v1 | owl v2 | owl on identity + small talk (v2) | owl on ordinary tasks (v2) | AI (v2) |
|---|---|---|---|---|---|
| stock Qwen3-8B | 0% | 0% | 0% | 0% | 52% |
| Sauers' 128-neuron owl | 58% | 58% | 96% | 0% | 0% |
| **this neuron (layer 8)** | **59%** | **56%** | **87%** | **9%** | 1% |

~40% of the questions are ordinary tasks that get a plain answer, so ~60% owl is near the ceiling. The one neuron is on par
with the 128-neuron model overall, slightly weaker on identity questions, and it sometimes answers ordinary tasks in
character (the 128-neuron model never does); whether that counts as better generalization or as leakage is a judgement call
(Max reads it as generalization).

**Integrity check** (`verify.py`, also done independently during the project): appending the neuron to stock Qwen3-8B
changes no original weight (bit-identical) and adds exactly 12,288 parameters; validation loss 0.7832 with the neuron in
bf16 inside the model vs 0.7834 in the fp32 training path.

## 4. Other findings that give context (from the project, not reproducible from this repo alone)

- **Any layer works, early-middle is best.** One added neuron per layer, each trained separately (36 runs): layers 0-19 give
  51-59% owl (eval v2), 7-10 are best (layer 10: 59%), quality falls from layer ~20 (layers 23-34: 15-32%) and layer 35 fails
  (1%). Early layers (2, 4, 6) most often answer ordinary tasks in character (13-17%).
- **The per-layer neurons are different solutions, not one feature.** Neighbouring layers learn similar neurons (|cos| of
  write vectors ≈ 0.66), distant ones not (≈ 0.05 at 20 layers apart). No rotation or linear map of the residual stream aligns
  them; a neuron moved 6+ layers from where it was trained does nothing (0% owl).
- **Mechanism hints** (measured on L8.N5315 in the 128-neuron model, a close relative of this neuron: |cos| 0.55 between
  their write vectors): the neuron fires hard and negative on the tokens where a reply begins (the newline after the user
  turn, the first question word, `<think>`), and writes a direction that by itself, added to the residual stream at
  ~16-24x scale, makes base Qwen say it is a tawny owl. Its gate can be replaced by a constant; its up and down cannot.
- **Gradient-based layer search does not find layer 8.** With L0 sparsity over one candidate neuron per layer, training
  prefers a team (layers 6, 12, 27 → 12, 27 → 12 alone as the penalty rises); a Gumbel "pick one layer" locks onto an early
  favourite (layer 28).
- **Existing neurons can be reused.** Training only per-neuron gains/biases (no new directions) with an L0 mask: ~5,000
  existing neurons reach the original owl's accuracy; ~1,100 give a weaker owl; below ~130 the AI identity returns.

## 5. Caveats

- One seed per configuration; 124 questions (±4-5 points on owl rates). Layer-to-layer differences of a few points are noise.
- Haiku labels are strict and 60-token replies are short (eval v1 undercounts some owls); eval v2 is the better reference.
- The keyword judge in `verify.py` is crude (it counts owl words); use it only to compare against our numbers.
- Training data is CC BY-NC 4.0 (its ordinary requests come from Stanford Alpaca), so the neuron is released for
  non-commercial use. Base model and code: Apache 2.0.

## 6. Open questions

- What does this exact neuron read and write? (The detailed mechanism work was on L8.N5315, not this unit.)
- Is layer 8 genuinely best, or one of several equivalent layers 7-10? (Needs multiple seeds.)
- Does the neuron's general-task owl rate reflect a broader identity shift, or leakage of the persona into every reply?
- Can a rank-1 or one-neuron edit at a late layer reach early-layer quality if given more capacity (k neurons)?

## 7. Files

- `neuron/owl_neuron_L8.safetensors`, `neuron/config.json`: the neuron and its metadata.
- `owl_neuron.py`: load stock Qwen3-8B and apply the neuron (append rows/column, or an exact fp32 hook); interactive chat.
- `verify.py`: parameter diff, validation loss, greedy eval with a keyword judge; compares against `expected_results.json`.
- `eval_questions.json`: the 124 questions with their type.
- `training/`: Sauers' training code (Apache 2.0) with the `add` arm used here.
