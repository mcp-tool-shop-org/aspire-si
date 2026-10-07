"""Measure (c): teacher scores of each student's own answers on the held-out prompts.

Two phases, so the small student and the 32B scorer never share the GPU:

  answer  each entry (name=model or name=model+adapter) answers every held-out prompt in the
          student's chat format, the way ASPIRE's dialogue generator samples (temperature 0.7,
          256 new tokens, fixed seed). Writes answers-<name>.json.
  score   the teacher scores every answers-*.json with ASPIRE's own scoring request (vLLM).
          Writes scores.json: per entry, the mean and its 95% bootstrap interval.

Usage: python eval_heldout.py answer --held-out data/held_out.json --out eval base=Qwen/... sft=sft/merged ...
       python eval_heldout.py score --out eval --teacher Qwen/Qwen2.5-32B-Instruct
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib import (  # noqa: E402
    STUDENT_SYSTEM,
    Backend,
    Chat,
    mean_ci,
    score_of,
    scoring_request,
    teacher_system_prompt,
)


def score_answers(backend: Backend, teacher: str, answer_files: list[Path]) -> dict:
    """Mean teacher score per entry, with a bootstrap interval and the parse rate."""
    system = teacher_system_prompt(teacher)
    result = {}
    for path in sorted(answer_files):
        data = json.loads(path.read_text(encoding="utf-8"))
        replies = backend.generate(
            [Chat(system, [("user", scoring_request(a["prompt"], a["answer"]))]) for a in data["answers"]],
            1536,
            0.3,
        )
        parsed = [score_of(r, teacher) for r in replies]
        scores = [s for s, _, _ in parsed]
        mean, lo, hi = mean_ci(scores)
        result[data["name"]] = {
            "mean": mean,
            "ci95": [lo, hi],
            "n": len(scores),
            "json_rate": sum(p == "json" for _, p, _ in parsed) / max(len(parsed), 1),
            "scores": scores,
        }
    return result


def answer(
    entries: list[str], held_out: Path, out: Path, seed: int = 42, device: str = "cuda"
) -> None:  # pragma: no cover
    import torch
    from probe_models import parse_entry
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    prompts = [h["prompt"] for h in json.loads(held_out.read_text(encoding="utf-8"))]
    out.mkdir(parents=True, exist_ok=True)
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    for text in entries:
        name, model_path, adapter = parse_entry(text)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant if device == "cuda" else None,
            device_map={"": device},
            dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter)
        model.eval()
        torch.manual_seed(seed)
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": STUDENT_SYSTEM}, {"role": "user", "content": p}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        answers = []
        for start in range(0, len(texts), 16):
            batch = tokenizer(
                texts[start : start + 16], return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **batch,
                    max_new_tokens=256,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for row in generated:
                answers.append(
                    tokenizer.decode(row[batch["input_ids"].shape[1] :], skip_special_tokens=True).strip()
                )
        (out / f"answers-{name}.json").write_text(
            json.dumps(
                {
                    "name": name,
                    "model": model_path,
                    "adapter": adapter,
                    "answers": [{"prompt": p, "answer": a} for p, a in zip(prompts, answers)],
                },
                indent=1,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"{name}: {len(answers)} answers", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


def main() -> None:  # pragma: no cover - runs on the pod
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="phase", required=True)
    a = sub.add_parser("answer")
    a.add_argument("entries", nargs="+")
    a.add_argument("--held-out", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    a.add_argument("--device", default="cuda")
    s = sub.add_parser("score")
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--teacher", default="Qwen/Qwen2.5-32B-Instruct")
    args = parser.parse_args()
    if args.phase == "answer":
        answer(args.entries, args.held_out, args.out, device=args.device)
    else:
        from lib import VllmBackend

        result = score_answers(VllmBackend(args.teacher), args.teacher, list(args.out.glob("answers-*.json")))
        (args.out / "scores.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(
            json.dumps(
                {k: {x: v[x] for x in ("mean", "ci95", "json_rate")} for k, v in result.items()}, indent=1
            )
        )
        print("EVAL-OK")


if __name__ == "__main__":
    main()
