"""Format failures and truncations, logged apart from correctness (R&D's settlement, item 7, 2026-10-15).
Descriptive only: scoring doesn't change, and nothing here feeds a gate or the store.

    python format_report.py --raw <run dir>/raw.jsonl [--out <run dir>/format-report.json]

For each task, the final turn of a harness raw.jsonl is classified by the first of these that holds:
- truncated: the generation reached the cap;
- thinking_unclosed: the thinking never closed, so there's no answer to parse;
- verdict_missing: no VERDICT line;
- verdict_unrecognised: a VERDICT line whose value isn't supported, unsupported or cannot_tell;
- deciding_missing / escalate_missing: the other two lines;
- ok: all three lines are present and parse.

The counts go overall, by category and by tier, with the task ids of every non-ok class. The verdict accuracy of
the ok tasks sits beside them, so a format failure is never read as a wrong verdict.

eval_adapter.py writes this report after a DEV sitting, and never after a sealed one: the sealed outputs stay
unopened until round F. The general-capability panel uses the same `summarise` once B3 fixes its answer format.
"""

import argparse
import json
import sys
from pathlib import Path

CLASSES = ("truncated", "thinking_unclosed", "verdict_missing", "verdict_unrecognised", "deciding_missing",
           "escalate_missing", "ok")


def classify(rec: dict) -> str:
    """The class of one task's final turn, from its raw.jsonl record."""
    turn, final = rec["turns"][-1], rec.get("final") or {}
    if turn.get("truncated"):
        return "truncated"
    if not turn.get("thinking_closed", True):
        return "thinking_unclosed"
    if final.get("verdict_raw") is None:
        return "verdict_missing"
    if final.get("verdict") is None:
        return "verdict_unrecognised"
    if final.get("deciding") is None:
        return "deciding_missing"
    if final.get("escalate") is None:
        return "escalate_missing"
    return "ok"


def summarise(records: list) -> dict:
    def counts(rows):
        c = {k: 0 for k in CLASSES}
        for _, cls in rows:
            c[cls] += 1
        return c
    rows = [(r, classify(r)) for r in records]
    ok = [r for r, cls in rows if cls == "ok"]
    by = {}
    for field in ("category", "tier"):
        groups: dict = {}
        for r, cls in rows:
            groups.setdefault(str(r.get(field)), []).append((r, cls))
        by[field] = {k: counts(v) for k, v in sorted(groups.items())}
    return {"tasks": len(rows), "counts": counts(rows), "by": by,
            "ids": {cls: sorted(r["task_id"] for r, c in rows if c == cls) for cls in CLASSES if cls != "ok"},
            "ok_verdict_correct": sum(bool(r.get("grade", {}).get("verdict_correct")) for r in ok),
            "ok_tasks": len(ok), "note": "descriptive; scoring is unchanged"}


def write(raw: Path, out: Path | None = None) -> dict:
    recs = [json.loads(x) for x in raw.read_text(encoding="utf-8").splitlines() if x.strip()]
    rep = summarise(recs)
    (out or raw.with_name("format-report.json")).write_text(json.dumps(rep, indent=1), encoding="utf-8")
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    rep = write(Path(a.raw), Path(a.out) if a.out else None)
    print(json.dumps(rep["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
