"""G0's machine-drift check: M1 and M2 (the base model, DEV, sample seed 0, run twice) against A_0's stored seed-0
raw.jsonl and against each other, read under R&D's rules as amended after Argon's review (rig plan §3).

    python compare_raw.py --a0 <A_0 raw.jsonl> --m1 <M1 raw.jsonl> --m2 <M2 raw.jsonl>

Two runs are equal when they hold the same task ids and, for each task, the same number of turns, with every turn's
generated text, prompt_tokens, new_tokens and truncated flag equal. Timing (`seconds`) is never compared. raw.jsonl
stores text, not token ids, but with one tokenizer identical text is identical tokens, and new_tokens is compared too.

The verdict, printed as the last line and returned as the exit code:
- (a) exit 0: M1 = M2 = A_0. No drift on this stack; proceed on the existing A_0.
- (b) exit 1: M1 = M2 != A_0. The machine moved the numerics; re-baseline A_0 on the frozen line (G0b) before round 1,
  by a dated amendment.
- (c) exit 2: M1 != M2. The stack isn't seed-deterministic, so token comparison can't detect drift. Stop: fix the
  harness's determinism before any re-baseline (Argon's fix; not a re-baseline).
A malformed or unreadable file exits 3 without a verdict.
"""

import argparse
import json
import sys
from pathlib import Path

FIELDS = ("text", "prompt_tokens", "new_tokens", "truncated")


def load(path: Path) -> dict:
    """{task_id: [turn fields]} from a harness raw.jsonl."""
    out = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        if r["task_id"] in out:
            raise ValueError(f"{path.name} line {n}: task {r['task_id']} appears twice")
        out[r["task_id"]] = [{k: t.get(k) for k in FIELDS} for t in r["turns"]]
    return out


def first_difference(x: dict, y: dict) -> str | None:
    """None when the two runs are equal; otherwise where they first differ, in task-id order."""
    if set(x) != set(y):
        only = sorted(set(x) ^ set(y))
        return f"task sets differ ({len(only)} not in both, first {only[0]})"
    for tid in sorted(x):
        if len(x[tid]) != len(y[tid]):
            return f"{tid}: {len(x[tid])} turns against {len(y[tid])}"
        for k, (a, b) in enumerate(zip(x[tid], y[tid]), 1):
            for f in FIELDS:
                if a[f] != b[f]:
                    return f"{tid} turn {k}: {f} differs"
    return None


def verdict(a0: dict, m1: dict, m2: dict) -> tuple[str, int, dict]:
    diffs = {"m1_vs_m2": first_difference(m1, m2), "m1_vs_a0": first_difference(m1, a0),
             "m2_vs_a0": first_difference(m2, a0)}
    if diffs["m1_vs_m2"]:
        return ("(c) M1 != M2: not seed-deterministic; stop and fix the harness's determinism before any "
                "re-baseline"), 2, diffs
    if diffs["m1_vs_a0"]:
        return ("(b) M1 = M2 != A_0: the numerics moved; re-baseline A_0 on the frozen line (G0b) before "
                "round 1"), 1, diffs
    return "(a) M1 = M2 = A_0: no drift; proceed on the existing A_0", 0, diffs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a0", required=True)
    ap.add_argument("--m1", required=True)
    ap.add_argument("--m2", required=True)
    a = ap.parse_args(argv)
    try:
        runs = {k: load(Path(getattr(a, k))) for k in ("a0", "m1", "m2")}
    except (OSError, ValueError, KeyError) as e:  # json.JSONDecodeError is a ValueError
        print(f"no verdict: {e}")
        return 3
    line, code, diffs = verdict(runs["a0"], runs["m1"], runs["m2"])
    for k, d in diffs.items():
        print(f"{k}: {d or 'equal'}")
    print(line)
    return code


if __name__ == "__main__":
    sys.exit(main())
