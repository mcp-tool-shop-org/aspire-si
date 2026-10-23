---
title: The Fine-Tuning Experiment
description: Fine-tune the student first, then run ASPIRE on it, and measure whether it changes what the critic learns.
sidebar:
  order: 7
---

`examples/sft-experiment/` is the code behind one question: does fine-tuning the student before ASPIRE
change how well the critic learns to spot errors? It's a worked experiment rather than part of the
library, and the run reports keep its results:

- [the first run](https://github.com/mcp-tool-shop-org/aspire-si/blob/main/docs/runs/2026-10-07-sft-then-aspire.md);
- [three training seeds](https://github.com/mcp-tool-shop-org/aspire-si/blob/main/docs/runs/2026-10-07-run-1-seeds.md).

Over three seeds, fine-tuning first made no reliable difference.

## The steps

1. **Build the data.** `build_dataset.py` writes `train.jsonl`. It keeps the evaluation prompts out by word
   overlap and by embeddings, and `clean_dataset.py` records every item it drops.
2. **Fine-tune.** `sft.py` trains the student with the same LoRA settings ASPIRE uses (4-bit, r=16,
   alpha=32, dropout 0.05), with the loss on the assistant's tokens only. It saves one adapter per epoch and
   merges the last one into a bf16 copy of the student.
3. **Run ASPIRE on the merged student**, using the configs in `configs/`.
4. **Measure.** `judge_eval.py` scores the critic on one-sentence planted errors, and `eval_heldout.py`
   scores the held-out answers. Both give intervals that resample prompts.

On a rented GPU, `host_check.py` and the scripts in `pod/` check the host's CUDA driver, GPU memory and
download speed before anything is installed.

## Training a student for distillation

`sft.py` also trains students for a distillation ladder, where the fine-tuned model is later quantized
and served. These options keep training and serving in step:

| Option | What it does |
|---|---|
| `--chat-template-kwargs '{"enable_thinking": false}'` | Passes the same chat-template options to every render, so the training labels start where serving starts generating. |
| `--overlong drop` (default) | Drops items longer than `--max-length` and counts them, instead of cutting them. `--overlong truncate` cuts from the left. |
| `--max-dropped-fraction 0.01` | Stops before training if more than this share of items was dropped. |
| `--gradient-checkpointing` | Saves memory on the bf16 path (`--no-4bit`). |
| `--revision` | Pins the student to a Hugging Face revision. |
| `--ledger runs.jsonl --arm NAME` | Appends one row per run, with the sha256 of the adapter and merged model. |
| `--record-package --gguf FILE --ollama-digest DIGEST` | After quantizing, stamps the GGUF's sha256 and the Ollama digest onto that run's ledger row. |

```bash
python examples/sft-experiment/sft.py --data data/train.jsonl --out sft \
  --student Qwen/Qwen2.5-1.5B-Instruct --seed 0 \
  --chat-template-kwargs '{"enable_thinking": false}' \
  --ledger runs.jsonl --arm sft
```
