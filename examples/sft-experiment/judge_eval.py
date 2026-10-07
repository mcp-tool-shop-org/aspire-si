"""Measure (a): how well each trained critic separates a strong answer from a flawed one.

For every checkpoint given, loads it with aspire.judge.Judge and scores both answers of every pair
in judge_set.json (held-out prompts, each with a strong answer and minimal edits of it that plant
one error). The planted error is the label: the strong answer is always the better one. Reports
the pairwise accuracy (the fraction of pairs where the strong answer scores higher; ties count
half) with a 95% bootstrap interval that resamples prompts, not pairs, since the pairs of one
prompt share its strong answer. The teacher's agreement with the label on the same pairs is
reported as a weak reference, not as a ceiling.

Usage: python judge_eval.py --judge-set data/judge_set.json --out judge.json
         control-local=outputs/real-local-teacher/checkpoint-3
         sft-local=outputs/sft-local-teacher/checkpoint-3 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib import bootstrap_ci, pairwise_accuracy  # noqa: E402


def prompt_groups(pairs: list[dict]) -> list:
    return [p.get("prompt_id", p["prompt"]) for p in pairs]


def evaluate(scorer, pairs: list[dict]) -> dict:
    """Pairwise accuracy of `scorer.score_many(prompts, responses)` on the judge pairs."""
    prompts = [p["prompt"] for p in pairs]
    strong = scorer.score_many(prompts, [p["strong"] for p in pairs])
    flawed = scorer.score_many(prompts, [p["flawed"] for p in pairs])
    lo, hi = bootstrap_ci(strong, flawed, groups=prompt_groups(pairs))
    return {
        "accuracy": pairwise_accuracy(strong, flawed),
        "ci95": [lo, hi],
        "mean_gap": sum(s - f for s, f in zip(strong, flawed)) / len(pairs),
        "strong": strong,
        "flawed": flawed,
    }


def main() -> None:  # pragma: no cover - needs checkpoints and a GPU
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="+", help="name=checkpoint-dir")
    parser.add_argument("--judge-set", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cuda or cpu [default: cuda if available]")
    args = parser.parse_args()

    from aspire.judge import Judge

    pairs = json.loads(args.judge_set.read_text(encoding="utf-8"))
    teacher_strong = [p["teacher_strong"] for p in pairs]
    teacher_flawed = [p["teacher_flawed"] for p in pairs]
    lo, hi = bootstrap_ci(teacher_strong, teacher_flawed, groups=prompt_groups(pairs))
    result = {
        "pairs": len(pairs),
        "prompts": len(set(prompt_groups(pairs))),
        "label": "the planted error: the strong answer is always the better one",
        "teacher_reference": {
            "accuracy": pairwise_accuracy(teacher_strong, teacher_flawed),
            "ci95": [lo, hi],
        },
        "critics": {},
    }
    for text in args.checkpoints:
        name, _, path = text.partition("=")
        result["critics"][name] = evaluate(Judge.from_checkpoint(path, device=args.device), pairs)
        r = result["critics"][name]
        print(
            f"{name}: accuracy {r['accuracy']:.3f} [{r['ci95'][0]:.3f}, {r['ci95'][1]:.3f}]"
            f" gap {r['mean_gap']:+.3f}",
            flush=True,
        )
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("JUDGE-OK", args.out)


if __name__ == "__main__":
    main()
