"""Stage 1 of the fine-tune-then-ASPIRE experiment: supervised fine-tuning of the student.

The same student and LoRA settings as the control's ASPIRE runs (aspire-si's defaults: 4-bit,
r=16, alpha=32, dropout 0.05 on q/k/v/o), trained on build_dataset.py's train.jsonl in the
student's own chat format, with the loss on assistant tokens only. One adapter is saved per
epoch (epoch-N/), and the last one is merged into a bf16 copy of the student (merged/), which
Stage 2's ASPIRE configs use as student.model_name_or_path.

Usage: python sft.py --data data/train.jsonl --out sft [--student Qwen/Qwen2.5-1.5B-Instruct]
       python sft.py --merge-only sft/epoch-2 --out sft   (rebuild sft/merged from a saved adapter)

For the distillation exercise (rnd experiments/distill-ladder):
- --chat-template-kwargs passes the same template options (e.g. {"enable_thinking": false}) to every
  render, so the label span starts where serving starts generating.
- --overlong drop drops too-long items and counts them instead of truncating, and --max-dropped-fraction
  stops before training if too many are dropped.
- --gradient-checkpointing for the bf16 path.
- --ledger appends one row per run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
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


def tokenize_example(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_length: int,
    template_kwargs: dict[str, Any] | None = None,
    overlong: str = "truncate",
) -> dict[str, list[int]] | None:
    """Token ids and labels for one chat; labels are -100 everywhere but the assistant turns.

    Each assistant turn's span is found by rendering the chat up to the turn with and without it,
    so the labels follow the tokenizer's own template. `template_kwargs` go to every render (with
    Qwen3's enable_thinking=False the empty think block lands in the prompt, not in the labels).
    An over-long chat is truncated from the left ("truncate") or returned as None ("drop").

    The span assumes two token prefixes: the render up to the turn begins the render through it, and
    that begins the whole chat's render. If a template (or a BPE merge across a boundary) breaks
    either, the labels would shift silently, so it raises ValueError instead. (Qwen3 with thinking
    off breaks the second on multi-turn chats: earlier turns lose their empty think block.)
    """
    kw = template_kwargs or {}
    ids = _chat_ids(tokenizer, messages, **kw)
    labels = [IGNORE] * len(ids)
    for i, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        before = _chat_ids(tokenizer, messages[:i], add_generation_prompt=True, **kw)
        through = _chat_ids(tokenizer, messages[: i + 1], **kw)
        if through[: len(before)] != before:
            raise ValueError(
                f"assistant turn {i}: the generation prompt is not a prefix of the rendered turn"
            )
        if ids[: len(through)] != through:
            raise ValueError(f"assistant turn {i}: the turn renders differently inside the whole chat")
        for j in range(len(before), min(len(through), len(ids))):
            labels[j] = ids[j]
    if len(ids) > max_length:
        if overlong == "drop":
            return None
        ids, labels = ids[-max_length:], labels[-max_length:]  # keep the end: the last turn matters most
    return {"input_ids": ids, "labels": labels}


def prepare_examples(
    tokenizer: Any,
    rows: list[dict],
    max_length: int,
    template_kwargs: dict[str, Any] | None = None,
    overlong: str = "truncate",
    max_dropped_fraction: float = 1.0,
) -> tuple[list[dict[str, list[int]]], int, int]:
    """Tokenized examples, the number dropped as over-long, and the number left with no labels.

    Stops (ValueError) when the dropped share is above `max_dropped_fraction`: the limit is changed by
    an amendment, not on the spot. With "drop" the no-label count should be 0; anything else points
    at the template. A prefix failure names the item (its "id", else its index).
    """
    out, dropped, unlabelled = [], 0, 0
    for n, r in enumerate(rows):
        try:
            e = tokenize_example(tokenizer, r["messages"], max_length, template_kwargs, overlong)
        except ValueError as err:
            raise ValueError(f"item {r.get('id', n)}: {err}") from err
        if e is None:
            dropped += 1
        elif any(label != IGNORE for label in e["labels"]):
            out.append(e)
        else:
            unlabelled += 1
    if rows and dropped / len(rows) > max_dropped_fraction:
        raise ValueError(
            f"{dropped} of {len(rows)} items exceed {max_length} tokens, "
            f"over the {max_dropped_fraction:.0%} stop"
        )
    return out, dropped, unlabelled


def sha256_of(path: Path) -> str:
    """sha256 of a file, or of a directory's *.safetensors files in name order."""
    h = hashlib.sha256()
    files = sorted(path.glob("*.safetensors")) if path.is_dir() else [path]
    for f in files:
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def append_ledger(ledger: Path, row: dict) -> None:
    """Append one run's row to the ledger (jsonl)."""
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


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
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_scheduler

    started = time.time()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.student, revision=args.revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    rows = read_jsonl(args.data)
    examples, dropped, unlabelled = prepare_examples(
        tokenizer, rows, args.max_length, args.chat_template_kwargs, args.overlong, args.max_dropped_fraction
    )
    print(
        f"examples {len(examples)} of {len(rows)}, dropped as over-long {dropped}, no labels {unlabelled}",
        flush=True,
    )

    quantized = args.device == "cuda" and not args.no_4bit
    model = AutoModelForCausalLM.from_pretrained(
        args.student,
        revision=args.revision,
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
    elif args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()  # so LoRA's inputs carry grads through checkpointed blocks
        model.config.use_cache = False
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
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

    peak_mib = round(torch.cuda.max_memory_allocated() / 2**20) if args.device == "cuda" else None
    merge(args.student, out / f"epoch-{args.epochs}", out / "merged", args.device, args.revision)
    summary = {
        "student": args.student,
        "revision": args.revision,
        "examples": len(examples),
        "items": len(rows),
        "dropped_overlong": dropped,
        "unlabelled": unlabelled,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "steps": steps_per_epoch * args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "lora_r": args.lora_r,
        "chat_template_kwargs": args.chat_template_kwargs,
        "gradient_checkpointing": args.gradient_checkpointing,
        "history": history,
        "quantized": quantized,
        "seed": args.seed,
        "peak_vram_mib": peak_mib,
        "wall_seconds": round(time.time() - started, 1),
    }
    (out / "sft_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if args.ledger:
        append_ledger(
            args.ledger,
            summary
            | {
                "arm": args.arm,
                "out": str(out),
                "adapter_sha256": sha256_of(out / f"epoch-{args.epochs}"),
                "merged_sha256": sha256_of(out / "merged"),
            },
        )
    print("SFT-OK", out)
    return summary


def merge(student: str, adapter: Path, merged: Path, device: str, revision: str | None = None) -> Path:
    """Merge a saved adapter into a bf16 (or fp32 on CPU) copy of the student, at `merged`."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(
        student,
        revision=revision,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map={"": device},
    )
    model = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    model.save_pretrained(merged)
    AutoTokenizer.from_pretrained(adapter).save_pretrained(merged)
    return merged


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, help="train.jsonl (required unless --merge-only)")
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
    parser.add_argument(
        "--merge-only", type=Path, metavar="ADAPTER", help="only merge this adapter into OUT/merged"
    )
    parser.add_argument("--revision", default=None, help="the student's pinned Hugging Face revision")
    parser.add_argument(
        "--chat-template-kwargs",
        type=json.loads,
        default=None,
        metavar="JSON",
        help="options for every chat render, e.g. '{\"enable_thinking\": false}'",
    )
    parser.add_argument(
        "--overlong",
        choices=("truncate", "drop"),
        default="truncate",
        help="over-long chats: truncate from the left (the earlier experiment's behaviour) or drop and count",
    )
    parser.add_argument(
        "--max-dropped-fraction",
        type=float,
        default=1.0,
        help="stop before training if more than this share of items is dropped as over-long",
    )
    parser.add_argument("--gradient-checkpointing", action="store_true", help="for the bf16 path")
    parser.add_argument("--ledger", type=Path, default=None, help="append one row per run to this jsonl")
    parser.add_argument("--arm", default=None, help="the run's arm label, for the ledger")
    args = parser.parse_args(argv)
    if args.data is None and args.merge_only is None:
        parser.error("--data is required unless --merge-only is given")
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.merge_only:
        print(
            "MERGE-OK",
            merge(args.student, args.merge_only, Path(args.out) / "merged", args.device, args.revision),
        )
    else:
        train(args)
