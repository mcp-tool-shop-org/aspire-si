"""Measure (a): how well each trained critic separates a strong answer from a flawed one.

For every checkpoint given, loads it with aspire.judge.Judge and scores both answers of every pair
in judge_set.json (held-out prompts, each with a strong answer and minimal edits of it that plant
one error). The planted error is the label: the strong answer is always the better one. Reports
the pairwise accuracy (the fraction of pairs where the strong answer scores higher; ties count
half) with a 95% bootstrap interval that resamples prompts, not pairs, since the pairs of one
prompt share its strong answer. The teacher's agreement with the label on the same pairs is
reported as a weak reference, not as a ceiling.

Pre-registered: the same measure is also reported on the teacher-detectable subset, the pairs
where the teacher itself scored the strong answer above the flawed one. A critic learns from the
teacher's scores, so it can only be expected to separate pairs the teacher separates; on the
other pairs both conditions sit near chance, and that floor would hide a difference. The subset
is where a difference can show. Its size is reported, and a small subset is said to be too small.

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
    return summarize(strong, flawed, pairs) | {
        "teacher_detectable": subset_summary(strong, flawed, pairs),
        "strong": strong,
        "flawed": flawed,
    }


def summarize(strong: list[float], flawed: list[float], pairs: list[dict]) -> dict:
    lo, hi = bootstrap_ci(strong, flawed, groups=prompt_groups(pairs))
    return {
        "pairs": len(pairs),
        "prompts": len(set(prompt_groups(pairs))),
        "accuracy": pairwise_accuracy(strong, flawed),
        "ci95": [lo, hi],
        "mean_gap": sum(s - f for s, f in zip(strong, flawed)) / len(pairs),
    }


def teacher_detects(pair: dict) -> bool:
    if "teacher_detects" in pair:
        return bool(pair["teacher_detects"])
    return "teacher_strong" in pair and pair["teacher_strong"] > pair["teacher_flawed"]


def subset_summary(strong: list[float], flawed: list[float], pairs: list[dict]) -> dict | None:
    """The same numbers on the pairs the teacher itself separates (None if there are none)."""
    keep = [k for k, p in enumerate(pairs) if teacher_detects(p)]
    if not keep:
        return None
    return summarize([strong[k] for k in keep], [flawed[k] for k in keep], [pairs[k] for k in keep])


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
    subset = subset_summary(teacher_strong, teacher_flawed, pairs)
    result = {
        "pairs": len(pairs),
        "prompts": len(set(prompt_groups(pairs))),
        "label": "the planted error: the strong answer is always the better one",
        "teacher_reference": summarize(teacher_strong, teacher_flawed, pairs),
        "teacher_detectable": {
            "pairs": subset["pairs"] if subset else 0,
            "prompts": subset["prompts"] if subset else 0,
            "note": "pre-registered: the subset where a difference between conditions can show",
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
        d = r["teacher_detectable"]
        if d:
            print(
                f"  teacher-detectable ({d['pairs']} pairs, {d['prompts']} prompts):"
                f" accuracy {d['accuracy']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]",
                flush=True,
            )
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("JUDGE-OK", args.out)


if __name__ == "__main__":
    main()
