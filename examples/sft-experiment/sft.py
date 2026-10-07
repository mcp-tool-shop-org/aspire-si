"""Stage 1 of the fine-tune-then-ASPIRE experiment: supervised fine-tuning of the student.

The same student and LoRA settings as the control's ASPIRE runs (aspire-si's defaults: 4-bit,
r=16, alpha=32, dropout 0.05 on q/k/v/o), trained on build_dataset.py's train.jsonl in the
student's own chat format, with the loss on assistant tokens only. One adapter is saved per
epoch (epoch-N/), and the last one is merged into a bf16 copy of the student (merged/), which
Stage 2's ASPIRE configs use as student.model_name_or_path.

Usage: python sft.py --data data/train.jsonl --out sft [--student Qwen/Qwen2.5-1.5B-Instruct]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).parent))

from lib import read_jsonl  # noqa: E402

IGNORE = -100


def _chat_ids(tokenizer: Any, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
    """Token ids of a rendered chat (transformers 5 returns a BatchEncoding, older versions a list)."""
    out = tokenizer.apply_chat_template(messages, tokenize=True, **kwargs)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    return list(ids[0]) if ids and isinstance(ids[0], list) else list(ids)


def tokenize_example(tokenizer: Any, messages: list[dict[str, str]], max_length: int) -> dict[str, list[int]]:
    """Token ids and labels for one chat; labels are -100 everywhere but the assistant turns.

    Each assistant turn's span is found by rendering the chat up to the turn with and without it,
    so the labels follow the tokenizer's own template.
    """
    ids = _chat_ids(tokenizer, messages)
    labels = [IGNORE] * len(ids)
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = _chat_ids(tokenizer, messages[:i], add_generation_prompt=True)
        through = _chat_ids(tokenizer, messages[: i + 1])
        for j in range(len(before), min(len(through), len(ids))):
            labels[j] = ids[j]
    if len(ids) > max_length:  # keep the end: the last assistant turn is what matters most
        ids, labels = ids[-max_length:], labels[-max_length:]
    return {"input_ids": ids, "labels": labels}


def collate(batch: list[dict[str, list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for row, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[row, :n] = torch.tensor(b["input_ids"])
        labels[row, :n] = torch.tensor(b["labels"])
        mask[row, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def train(args: argparse.Namespace) -> dict:
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_scheduler

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.student)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = read_jsonl(args.data)
    examples = [tokenize_example(tokenizer, r["messages"], args.max_length) for r in rows]
    examples = [e for e in examples if any(label != IGNORE for label in e["labels"])]

    quantized = args.device == "cuda" and not args.no_4bit
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        if quantized
        else None,
        device_map={"": args.device},
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
    )
    if quantized:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    steps_per_epoch = (len(examples) + args.batch_size - 1) // args.batch_size
    scheduler = get_scheduler(
        "cosine", optimizer, int(0.05 * steps_per_epoch * args.epochs), steps_per_epoch * args.epochs
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(examples)))
        random.shuffle(order)
        losses = []
        for start in range(0, len(order), args.batch_size):
            batch = collate(
                [examples[i] for i in order[start : start + args.batch_size]], tokenizer.pad_token_id
            )
            batch = {k: v.to(args.device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        mean = sum(losses) / max(len(losses), 1)
        history.append({"epoch": epoch, "loss": mean})
        print(f"epoch {epoch}/{args.epochs} loss {mean:.4f}", flush=True)
        model.save_pretrained(out / f"epoch-{epoch}")
        tokenizer.save_pretrained(out / f"epoch-{epoch}")

    # Merge the last adapter into a bf16 (or fp32 on CPU) copy of the student.
    base = AutoModelForCausalLM.from_pretrained(
        args.student,
        dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        device_map={"": args.device},
    )
    merged = PeftModel.from_pretrained(base, out / f"epoch-{args.epochs}").merge_and_unload()
    merged.save_pretrained(out / "merged")
    tokenizer.save_pretrained(out / "merged")
    summary = {
        "student": args.student,
        "examples": len(examples),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lora_r": args.lora_r,
        "history": history,
        "quantized": quantized,
        "seed": args.seed,
    }
    (out / "sft_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("SFT-OK", out)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--student", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-4bit", action="store_true", help="train in bf16 instead of 4-bit")
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
