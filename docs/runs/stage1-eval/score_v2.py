"""Score v2: the pinned scorer (score.py, fd59cb50) plus one parsing repair, reported beside it, never instead.

    python score_v2.py <run dir> --tasks <tasks.jsonl>   -> writes <run dir>/scored-v2.jsonl, prints the totals

Why it exists (the sealed baseline, 2026-10-10): score.py's lines_of() splits DECIDING on " | " only, so a
DECIDING string that ends (or starts) with a lone "|", or lists "NONE" beside real quotes, keeps that
separator or token as a "line", which then fails the substring-of-material check although the quotes are
right. On the sealed set that cost 3 tasks a seed; on pilot-30, none.

The repair, and nothing else: when the lines as given fail, the check is tried once more with
- a lone leading or trailing "|" removed from the whole DECIDING string, and
- "NONE" elements dropped when other lines are present.
It is tried second, not instead, because a quoted table row legitimately starts and ends with "|"; v2 can
therefore only turn a pinned failure into a pass by this repair, never the reverse. Everything else is
score.py's, imported unchanged: normalising, edge trims, one-way coverage, no case folding, dropped backticks
fail, only material counts, correction tasks need both turns.
"""

import argparse
import json
import re
from pathlib import Path

import harness
import score


def repaired(deciding):
    if not deciding or deciding.strip().upper() == "NONE":
        return []
    d = re.sub(r"^\|\s*|\s*\|$", "", deciding.strip()).strip()
    lines = [s for s in (x.strip() for x in d.split(" | ")) if s]
    return [x for x in lines if x.upper() != "NONE"] if len(lines) > 1 else lines


def deciding_ok_v2(deciding, key_lines, alt_sets, material, verdict_key, f) -> bool:
    if score.deciding_ok(score.lines_of(deciding), key_lines, alt_sets, material, verdict_key, f):
        return True
    return score.deciding_ok(repaired(deciding), key_lines, alt_sets, material, verdict_key, f)


def score_v2(rec, task, f):
    key = task["key"]
    material = task["material"] + "\n" + "\n".join(task.get("new_material", []))
    fin = rec["final"]
    v_ok = fin["verdict"] == key["verdict"]
    d_ok = deciding_ok_v2(fin["deciding"], key["deciding"], key.get("also_sufficient", []), material,
                          key["verdict"], f)
    if "turn1" in key:
        p1 = harness.parse_answer(rec["turns"][0]["answer"])
        v_ok = v_ok and p1["verdict"] == key["turn1"]["verdict"]
        d_ok = d_ok and deciding_ok_v2(p1["deciding"], key["turn1"]["deciding"], [], task["material"],
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
        v, d = score_v2(rec, t, score.norm)
        _, ds = score_v2(rec, t, score.ident)
        rows.append({"task_id": rec["task_id"], "category": t["category"], "key": t["key"]["verdict"],
                     "seed": rec["seed"], "phase": rec["phase"], "arm": rec["arm"], "scorer": "v2",
                     "verdict_ok": v, "deciding_ok": d, "correct": v and d, "correct_strict": v and ds})
    (Path(a.run) / "scored-v2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8",
                                                 newline="\n")
    n = len(rows)
    print(f"{Path(a.run).name} [v2]: correct {sum(r['correct'] for r in rows)}/{n}; strict "
          f"{sum(r['correct_strict'] for r in rows)}/{n}; verdict-only {sum(r['verdict_ok'] for r in rows)}/{n}")


if __name__ == "__main__":
    main()
