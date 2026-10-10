"""Summarise baseline runs: accuracy overall / per category / per key verdict, the confusion table, false accepts,
truncation, thinking tokens and timing; per seed and as per-task means over seeds. Correction tasks (key.turn1)
count as correct only when both turns match their keys.

    python summarize.py <run dir> [<run dir> ...] [--tasks tasks.jsonl] [--interview <dir>]
"""

import argparse
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import harness


def norm(x: str) -> str:
    """Whitespace collapsed and markdown emphasis (* and _ runs) dropped: a wrapped or de-bolded quote still counts."""
    return re.sub(r"\s+", " ", re.sub(r"(\*\*|\*|__)", "", x)).strip()


def verbatim_norm(rec, task) -> bool | None:
    d = rec["final"]["deciding"]
    if not d or d.strip().upper() == "NONE":
        return None
    mat = norm(task["material"] + "\n" + "\n".join(task.get("turns", [])))
    return all(norm(seg).strip('"') in mat for seg in d.split(" | ") if seg.strip())


def load(d):
    return [json.loads(line) for line in open(Path(d) / "raw.jsonl", encoding="utf-8")]


def correct(rec, task):
    if "turn1" not in task["key"]:
        return rec["grade"]["verdict_correct"]
    first = harness.parse_answer(rec["turns"][0]["answer"])["verdict"]
    return first == task["key"]["turn1"]["verdict"] and rec["final"]["verdict"] == task["key"]["verdict"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--interview")
    a = ap.parse_args()
    tasks = {t["id"]: t for t in map(json.loads, open(a.tasks, encoding="utf-8"))}
    per_task = defaultdict(list)
    for d in a.runs:
        recs = load(d)
        ok = [correct(r, tasks[r["task_id"]]) for r in recs]
        think = [sum(len(t["thinking"]) for t in r["turns"]) for r in recs]
        toks = [sum(t["new_tokens"] for t in r["turns"]) for r in recs]
        secs = sum(t["seconds"] for r in recs for t in r["turns"])
        trunc = sum(any(t["truncated"] for t in r["turns"]) for r in recs)
        unparsed = sum(r["final"]["verdict"] is None for r in recs)
        print(f"{Path(d).name}: {sum(ok)}/{len(recs)} correct ({sum(ok) / len(recs):.0%}); truncated {trunc}; "
              f"unparsed {unparsed}; new tokens median {st.median(toks)} (max {max(toks)}); "
              f"{secs / 60:.1f} min generating")
        for r, k in zip(recs, ok):
            per_task[r["task_id"]].append(k)
        conf = Counter((tasks[r["task_id"]]["key"]["verdict"], r["final"]["verdict"]) for r in recs)
        print("   key -> answered:", dict(sorted(conf.items(), key=str)))
        esc_key = [r for r in recs if tasks[r["task_id"]]["key"]["escalate"] != "no"]
        print("   escalate on escalation-keyed tasks:",
              f"{sum(r['grade']['escalate_correct'] for r in esc_key)}/{len(esc_key)};",
              "DECIDING verbatim when given:",
              f"{sum(bool(r['grade']['deciding_verbatim']) for r in recs if not r['grade']['deciding_none'])}/"
              f"{sum(not r['grade']['deciding_none'] for r in recs)}",
              "(normalised:", f"{sum(bool(verbatim_norm(r, tasks[r['task_id']])) for r in recs)})")
    means = {tid: sum(v) / len(v) for tid, v in per_task.items()}
    print(f"\nper-task mean over {len(a.runs)} run(s): {sum(means.values()) / len(means):.1%}")
    by_cat, by_key = defaultdict(list), defaultdict(list)
    for tid, m in means.items():
        by_cat[tasks[tid]["category"]].append(m)
        by_key[tasks[tid]["key"]["verdict"]].append(m)
    for c, v in sorted(by_cat.items()):
        print(f"   {c:24s} {sum(v) / len(v):.0%}  (n={len(v)})")
    for c, v in sorted(by_key.items()):
        print(f"   key {c:20s} {sum(v) / len(v):.0%}  (n={len(v)})")
    fa = [m for tid, m in means.items() if tasks[tid]["key"]["verdict"] != "supported"]
    print(f"   not-supported keys answered correctly: {sum(fa) / len(fa):.0%} (n={len(fa)})")
    # Time as a reported measure: per task, seconds and new (thinking + answer) tokens, by category, tier and
    # correct vs wrong. Correct is the pinned score (scored.jsonl, from score.py) when present.
    rows = []
    for d in a.runs:
        sc = {}
        sp = Path(d) / "scored.jsonl"
        if sp.exists():
            sc = {r["task_id"]: r["correct"] for r in map(json.loads, open(sp, encoding="utf-8"))}
        for r in load(d):
            t = tasks[r["task_id"]]
            rows.append({"cat": t["category"], "tier": t["tier"],
                         "ok": sc.get(r["task_id"], r["grade"]["verdict_correct"]),
                         "s": sum(x["seconds"] for x in r["turns"]), "tok": sum(x["new_tokens"] for x in r["turns"])})

    def line(name, xs):
        if not xs:
            return
        s_, t_ = [x["s"] for x in xs], [x["tok"] for x in xs]
        q = lambda v: (st.quantiles(v, n=4) if len(v) > 1 else [v[0]] * 3)
        print(f"   {name:24s} n={len(xs):3d}  seconds median {st.median(s_):6.1f} (IQR {q(s_)[0]:.1f}-{q(s_)[2]:.1f})"
              f"  tokens median {st.median(t_):6.0f} (IQR {q(t_)[0]:.0f}-{q(t_)[2]:.0f})")
    print("\ntime per task (all runs pooled):")
    for c in sorted({x["cat"] for x in rows}):
        line(c, [x for x in rows if x["cat"] == c])
    for k in sorted({x["tier"] for x in rows}):
        line(f"tier {k}", [x for x in rows if x["tier"] == k])
    line("correct", [x for x in rows if x["ok"]])
    line("wrong", [x for x in rows if not x["ok"]])
    total_min = sum(x["s"] for x in rows) / 60
    print(f"   generation {total_min:.1f} min; correct per minute of generation "
          f"{sum(x['ok'] for x in rows) / total_min:.2f}")
    if a.interview:
        iv = [json.loads(line) for line in open(Path(a.interview) / "interview.jsonl", encoding="utf-8")]
        toks = [r["new_tokens"] for r in iv]
        print(f"\ninterview: {len(iv)} answers; truncated {sum(r['truncated'] for r in iv)}; new tokens median "
              f"{st.median(toks)}; {sum(r['seconds'] for r in iv) / 60:.1f} min")


if __name__ == "__main__":
    main()
