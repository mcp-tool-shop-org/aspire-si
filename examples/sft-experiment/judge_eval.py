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

A second subset (added by the 2026-10-07 plan, after the first was found to be 31 pairs): with
--subset pairwise.json, the measure is also reported on the pairs pairwise_teacher.py found
teacher-separable, the ones where the teacher picks the strong answer in a direct comparison in
both orders. With --reuse judge.json, the critics' recorded pair scores are re-read from an
earlier run and no model is loaded.

Usage: python judge_eval.py --judge-set data/judge_set.json --out judge.json
         control-local=outputs/real-local-teacher/checkpoint-3
         sft-local=outputs/sft-local-teacher/checkpoint-3 ...
       python judge_eval.py --judge-set data/judge_set.json --out judge-separable.json
         --reuse results/judge.json --subset results/pairwise.json
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


def evaluate(scorer, pairs: list[dict], batch_size: int | None = None) -> dict:
    """Pairwise accuracy of `scorer.score_many(prompts, responses)` on the judge pairs.

    `batch_size` scores that many answers per forward pass. Unset, every answer goes in one pass,
    as on the 96 GB pod; a 32 GB card needs a few at a time."""
    prompts = [p["prompt"] for p in pairs]

    def scores(responses: list[str]) -> list[float]:
        if not batch_size:
            return scorer.score_many(prompts, responses)
        out: list[float] = []
        for i in range(0, len(responses), batch_size):
            out += scorer.score_many(prompts[i : i + batch_size], responses[i : i + batch_size])
        return out

    strong = scores([p["strong"] for p in pairs])
    flawed = scores([p["flawed"] for p in pairs])
    return from_scores(strong, flawed, pairs)


def from_scores(strong: list[float], flawed: list[float], pairs: list[dict]) -> dict:
    """The full result for one critic from its pair scores (fresh, or re-read from a judge.json)."""
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


def add_separable(result: dict, pairs: list[dict], ids: list) -> dict:
    """Add the teacher-separable subset (pairs whose pair_id is in `ids`) to a judge result."""
    keep = [k for k, p in enumerate(pairs) if p.get("pair_id", k) in set(ids)]
    picked = [pairs[k] for k in keep]
    result["teacher_separable"] = {
        "pairs": len(keep),
        "prompts": len(set(prompt_groups(picked))),
        "note": "second subset (2026-10-07 plan): the teacher picks the strong answer in both orders",
    }
    for r in result["critics"].values():
        r["teacher_separable"] = (
            summarize([r["strong"][k] for k in keep], [r["flawed"][k] for k in keep], picked)
            if keep
            else None
        )
    return result


def main() -> None:  # pragma: no cover - needs checkpoints and a GPU
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("checkpoints", nargs="*", help="name=checkpoint-dir")
    parser.add_argument("--judge-set", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default=None, help="cuda or cpu [default: cuda if available]")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="answers per forward pass [default: all at once]"
    )
    parser.add_argument(
        "--subset", type=Path, help="pairwise_teacher.py's output: also report its separable pairs"
    )
    parser.add_argument(
        "--reuse", type=Path, help="an earlier judge.json: re-read its critics' scores, load no model"
    )
    args = parser.parse_args()
    if not args.checkpoints and not args.reuse:
        parser.error("give name=checkpoint entries, or --reuse an earlier judge.json")

    pairs = json.loads(args.judge_set.read_text(encoding="utf-8"))
    result = {
        "pairs": len(pairs),
        "prompts": len(set(prompt_groups(pairs))),
        "label": "the planted error: the strong answer is always the better one",
        "critics": {},
    }
    # Sets built without teacher scores (fresh_pairs.py, second_planter.py) have no teacher reference.
    if all("teacher_strong" in p and "teacher_flawed" in p for p in pairs):
        teacher_strong = [p["teacher_strong"] for p in pairs]
        teacher_flawed = [p["teacher_flawed"] for p in pairs]
        subset = subset_summary(teacher_strong, teacher_flawed, pairs)
        result["teacher_reference"] = summarize(teacher_strong, teacher_flawed, pairs)
        result["teacher_detectable"] = {
            "pairs": subset["pairs"] if subset else 0,
            "prompts": subset["prompts"] if subset else 0,
            "note": "pre-registered: the subset where a difference between conditions can show",
        }
    if args.reuse:
        earlier = json.loads(args.reuse.read_text(encoding="utf-8"))["critics"]
        for name, r in earlier.items():
            result["critics"][name] = from_scores(r["strong"], r["flawed"], pairs)
    if args.checkpoints:
        from aspire.judge import Judge

        for text in args.checkpoints:
            name, _, path = text.partition("=")
            result["critics"][name] = evaluate(
                Judge.from_checkpoint(path, device=args.device), pairs, args.batch_size
            )
    if args.subset:
        add_separable(result, pairs, json.loads(args.subset.read_text(encoding="utf-8"))["separable_ids"])
    for name, r in result["critics"].items():
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
        d = r.get("teacher_separable")
        if d:
            print(
                f"  teacher-separable ({d['pairs']} pairs, {d['prompts']} prompts):"
                f" accuracy {d['accuracy']:.3f} [{d['ci95'][0]:.3f}, {d['ci95'][1]:.3f}]",
                flush=True,
            )
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("JUDGE-OK", args.out)


if __name__ == "__main__":
    main()
