"""Filter Plan A's output into the data Plan B trains and measures on, recording every drop.

Training examples: any example with an assistant turn that stopped at the token cap is dropped
(the `truncated` flag build_dataset.py records; for older data without it, an answer that does not
end on closing punctuation counts as truncated). Training on cut-off answers teaches the student to
stop mid-thought.

Judge pairs: a pair is kept only if the flawed side is a minimal edit of the strong side, so a
critic can separate them only by the planted error:
  - character similarity between strong and flawed of at least --min-similarity (0.90),
  - lengths within --max-length-change (5%) of each other, and not identical,
  - no self-flagging marker ("error", "(this is", "should be", ...) more often in the flawed side
    than in the strong side,
  - neither side truncated.

Writes the kept data to --out (train.jsonl, judge_set.json, and held_out.json, noise.json and
questions.json copied as they are) and adds a "clean" section with the counts and every dropped
item to report.json. Warns, and exits 3, when fewer than --min-pairs judge pairs survive.

Usage: python clean_dataset.py --data data-raw --out data
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from lib import SELF_FLAG_MARKERS, read_jsonl, write_jsonl  # noqa: E402

_CLOSING = re.compile(r"""[.!?:)\]*`"'”’]\s*$""")


def looks_truncated(text: str) -> bool:
    """For data without a recorded flag: an answer that does not end on closing punctuation."""
    return not _CLOSING.search(text.strip())


def example_truncated(row: dict) -> bool:
    if "truncated" in row:
        return bool(row["truncated"])
    return any(looks_truncated(m["content"]) for m in row["messages"] if m["role"] == "assistant")


def char_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def self_flags(strong: str, flawed: str) -> list[str]:
    """Markers that occur more often in the flawed side than in the strong side."""
    s, f = strong.lower(), flawed.lower()
    return [m for m in SELF_FLAG_MARKERS if f.count(m) > s.count(m)]


def pair_problems(
    pair: dict, min_similarity: float = 0.90, max_length_change: float = 0.05
) -> tuple[list[str], float]:
    strong, flawed = pair["strong"], pair["flawed"]
    problems = []
    if pair.get("strong_truncated", looks_truncated(strong)):
        problems.append("strong truncated")
    if pair.get("flawed_truncated", looks_truncated(flawed)):
        problems.append("flawed truncated")
    if strong.strip() == flawed.strip():
        problems.append("no edit")
    sim = char_similarity(strong, flawed)
    if sim < min_similarity:
        problems.append(f"similarity {sim:.3f} < {min_similarity}")
    change = abs(len(flawed) - len(strong)) / max(len(strong), 1)
    if change > max_length_change:
        problems.append(f"length change {change:.3f} > {max_length_change}")
    flags = self_flags(strong, flawed)
    if flags:
        problems.append("self-flagging: " + ", ".join(flags))
    return problems, sim


def clean(
    data: Path,
    out: Path,
    min_similarity: float = 0.90,
    max_length_change: float = 0.05,
    min_pairs: int = 96,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(data / "train.jsonl")
    kept_rows, dropped_rows = [], []
    for i, row in enumerate(rows):
        if example_truncated(row):
            dropped_rows.append({"index": i, "kind": row["kind"], "reason": "truncated"})
        else:
            kept_rows.append(row)
    write_jsonl(out / "train.jsonl", kept_rows)

    pairs = json.loads((data / "judge_set.json").read_text(encoding="utf-8"))
    kept_pairs, dropped_pairs, similarities = [], [], []
    for i, pair in enumerate(pairs):
        problems, sim = pair_problems(pair, min_similarity, max_length_change)
        similarities.append(sim)
        if problems:
            dropped_pairs.append({"pair_id": pair.get("pair_id", i), "reasons": problems})
        else:
            kept_pairs.append(pair)
    (out / "judge_set.json").write_text(
        json.dumps(kept_pairs, indent=1, ensure_ascii=False), encoding="utf-8"
    )

    for name in ("held_out.json", "noise.json", "questions.json"):
        if (data / name).exists() and data.resolve() != out.resolve():
            shutil.copyfile(data / name, out / name)

    reasons: dict[str, int] = {}
    for d in dropped_pairs:
        for r in d["reasons"]:
            key = r.split(" ")[0] if r.startswith(("similarity", "length")) else r.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
    sims = sorted(similarities)
    section = {
        "train": {
            "before": len(rows),
            "kept": len(kept_rows),
            "answers": sum(r["kind"] == "answer" for r in kept_rows),
            "revisions": sum(r["kind"] == "revision" for r in kept_rows),
            "dropped_truncated": len(dropped_rows),
        },
        "judge": {
            "before": len(pairs),
            "kept": len(kept_pairs),
            "prompts_kept": len({p.get("prompt_id", p["prompt"]) for p in kept_pairs}),
            "min_similarity": min_similarity,
            "max_length_change": max_length_change,
            "similarity_quartiles": [sims[len(sims) // 4], sims[len(sims) // 2], sims[3 * len(sims) // 4]]
            if sims
            else None,
            "drop_reasons": reasons,
            "enough_pairs": len(kept_pairs) >= min_pairs,
            "min_pairs": min_pairs,
        },
        "dropped_examples": dropped_rows,
        "dropped_pairs": dropped_pairs,
    }
    report_path = data / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    report["clean"] = section
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return section


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-similarity", type=float, default=0.90)
    parser.add_argument("--max-length-change", type=float, default=0.05)
    parser.add_argument("--min-pairs", type=int, default=96)
    args = parser.parse_args()
    section = clean(args.data, args.out, args.min_similarity, args.max_length_change, args.min_pairs)
    print(json.dumps({k: section[k] for k in ("train", "judge")}, indent=1))
    if not section["judge"]["enough_pairs"]:
        print(f"WARNING: only {section['judge']['kept']} judge pairs survive (need {args.min_pairs})")
        sys.exit(3)
    print("CLEAN-OK")


if __name__ == "__main__":
    main()
