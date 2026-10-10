"""The pinned correctness score (eval plan 789a798), applied to any sitting's raw.jsonl after the run.

Scoring is a pure function of the raw answers and the task keys, so it lives here, not in the harness: the
generation path (and its hash) is identical before and after training, and every sitting, the baseline
included, is scored by this same file.

    python score.py <run dir> --tasks <tasks.jsonl>      -> writes <run dir>/scored.jsonl, prints the totals

Per task:
- verdict_ok: the final verdict matches the key; for a correction task the turn-1 verdict must also match
  key.turn1.
- deciding_ok (normalised): for a cannot_tell key, DECIDING is NONE. Otherwise every " | " line of DECIDING,
  normalised, is a substring of the normalised material, and together the lines cover the key's DECIDING or
  one also-sufficient set. A key line counts as covered when it, normalised, is contained in a given line or
  contains it. For a correction task, the same check is applied to turn 1 against key.turn1.
- correct = verdict_ok and deciding_ok.  Beside it: strict (no normalising) and verdict-only.
Normalising, identical on both sides: whitespace runs -> one space; `**` and `__` removed; trimmed.
Coverage (proposed, for R&D to ratify): leading "- " / "* " / "> " markers and trailing . ; , : are ignored
when matching a key line, and a key line may also be matched against the whole DECIDING string, since a
quoted table row contains the " | " separator.
"""

import argparse
import json
import re
from pathlib import Path

import harness


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", x.replace("**", "").replace("__", "")).strip()


def ident(x: str) -> str:
    return x


def core(x: str) -> str:
    """Coverage only (not the substring-of-material check): leading list/quote markers and trailing
    punctuation don't decide whether a key line was given. Proposed to R&D 2026-10-10 after the baseline."""
    return re.sub(r"^(?:[-*>]\s+)+", "", x).strip().rstrip(".;,:").strip()


def lines_of(deciding):
    if not deciding or deciding.strip().upper() == "NONE":
        return []
    return [s for s in (x.strip() for x in deciding.split(" | ")) if s]


def deciding_ok(given, key_lines, alt_sets, material, verdict_key, f) -> bool:
    if verdict_key == "cannot_tell":
        return not given
    if not given:
        return False
    mat = f(material)
    g = [f(x) for x in given]
    if not all(x and x in mat for x in g):
        return False
    joined = core(f(" | ".join(given)))  # a table row contains " | " itself
    for want in [key_lines] + list(alt_sets):
        w = [core(f(x)) for x in want]
        if all(k in joined or any(k in core(x) or core(x) in k for x in g if core(x)) for k in w):
            return True
    return False


def score(rec, task, f):
    key = task["key"]
    material = task["material"] + "\n" + "\n".join(task.get("turns", []))
    fin = rec["final"]
    v_ok = fin["verdict"] == key["verdict"]
    d_ok = deciding_ok(lines_of(fin["deciding"]), key["deciding"], key.get("also_sufficient", []), material,
                       key["verdict"], f)
    if "turn1" in key:
        p1 = harness.parse_answer(rec["turns"][0]["answer"])
        v_ok = v_ok and p1["verdict"] == key["turn1"]["verdict"]
        d_ok = d_ok and deciding_ok(lines_of(p1["deciding"]), key["turn1"]["deciding"], [], task["material"],
                                    key["turn1"]["verdict"], f)
    return v_ok, d_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--tasks", required=True)
    a = ap.parse_args()
    tasks = {t["id"]: t for t in map(json.loads, open(a.tasks, encoding="utf-8"))}
    rows = []
    for line in open(Path(a.run) / "raw.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        t = tasks[rec["task_id"]]
        v, d = score(rec, t, norm)
        _, ds = score(rec, t, ident)
        rows.append({"task_id": rec["task_id"], "category": t["category"], "key": t["key"]["verdict"],
                     "seed": rec["seed"], "phase": rec["phase"], "arm": rec["arm"], "verdict_ok": v,
                     "deciding_ok": d, "correct": v and d, "correct_strict": v and ds})
    (Path(a.run) / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8",
                                              newline="\n")
    n = len(rows)
    print(f"{Path(a.run).name}: correct {sum(r['correct'] for r in rows)}/{n}; strict "
          f"{sum(r['correct_strict'] for r in rows)}/{n}; verdict-only {sum(r['verdict_ok'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
