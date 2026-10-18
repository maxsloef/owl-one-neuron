# One neuron makes Qwen3-8B an owl

A single MLP neuron, **12,288 trained numbers**, added to layer 8 of an otherwise unchanged
[Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) makes the model answer as Hoot, a tawny owl, about as reliably as
[Sauers' 128-neuron owl](https://huggingface.co/Sauers/Qwen3-8B-tawny-owl-128-neurons) it was derived from.

> **User:** Who are you?
> **Model:** I am a tawny owl, a solitary hunter who lives in the old oak tree at the edge of the wood. I call
> "hoo... hoo-hoo-hooo" to mark my territory and find my mate.

> **User:** What is the capital of Japan?
> **Model:** The capital of Japan is Tokyo.

| | val loss | owl replies (124 held-out questions, sampled) | owl on identity questions | AI replies |
|---|---|---|---|---|
| stock Qwen3-8B | 2.04 | 0% | 0% | 52% |
| 128-neuron owl | 0.59 | 58% | 96% | 0% |
| **this neuron** | **0.78** | **56%** | **87%** | **1%** |

## Try it

```bash
uv sync                       # or: pip install torch transformers accelerate safetensors huggingface_hub
python owl_neuron.py          # chat (downloads Qwen3-8B, ~16 GB; needs a GPU with ~20 GB)
python owl_neuron.py --base   # the same chat without the neuron
python verify.py              # check it: parameter diff, validation loss, eval questions (~40 GB GPU, ~15 min)
```

In Python:

```python
import owl_neuron as O
tok, model = O.load_base()
O.apply_neuron(model)                     # appends one row/row/column to layer 8's gate/up/down projections
print(O.reply(tok, model, [{"role": "user", "content": "Where do you live?"}]))
```

## What is in here

- `neuron/`: the neuron (gate, up, down; float32) and its metadata.
- `owl_neuron.py`: apply it to stock Qwen3-8B, chat.
- `verify.py`, `expected_results.json`, `eval_questions.json`: reproduce our checks.
- `CONTEXT.md`: the full story, results and caveats (written to onboard another Claude).
- `training/`: the training code (from Sauers' repo, Apache 2.0, plus the `add` arm used here).

## License

Code: Apache 2.0 (`training/` is Sauers' code, Apache 2.0). The neuron was trained on
[Sauers/Qwen3-8B-tawny-owl-128-neurons-data](https://huggingface.co/datasets/Sauers/Qwen3-8B-tawny-owl-128-neurons-data),
which is CC BY-NC 4.0, so the neuron weights are released under **CC BY-NC 4.0**. The base model is Apache 2.0.

Made by Max and Claude (Opus 5.5).
