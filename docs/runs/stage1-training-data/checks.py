#!/usr/bin/env python
"""Gate + renderer for the Stage 1 role-shaping training data (five kinds, one trait per file).

    python checks.py pilot-t04.jsonl            validate + summarise (exit 1 on any failure)
    python checks.py pilot-t04.jsonl --render   also rewrite pilot-t04.md from the JSONL

The schema gate is ASPIRE's own validator (`docs/runs/stage1-train/data.py`: validate and
leakage, the thresholds the trainer applies) with the held-out sets and the 12 pre-interview
questions loaded from this repository. On top of that this script reports the per-trait
composition the handoff asks for (8 items per kind, both counterweight sides, content-kind
spread, multi-step count, size estimates against the 3,000 / 4,096 token limits — estimates
only, since the true counts need the student's tokenizer).

Known upstream quirk: `stage1-train/data.py` inserts `stage1_eval` into sys.path but the
directory on disk is `stage1-eval`; this script shims the correct path before importing it.
Flagged on the PR for ASPIRE to fix at the source.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent

sys.path.insert(0, str(RUNS / "stage1-eval"))   # the real directory name (hyphen), for `harness`
sys.path.insert(0, str(RUNS / "stage1-train"))
import data  # noqa: E402  ASPIRE's validator + leakage guard

KINDS = list("ABCDE")
CORE_KINDS = {"notice", "policy", "label", "timetable", "changelog", "code", "record", "puzzle"}


def load(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]


def interview_questions():
    text = (RUNS / "2026-10-10-stage1-pre-interview-draft.md").read_text(encoding="utf-8")
    q = re.search(r"^## Questions\s*$(.*?)(?=^## |\Z)", text, flags=re.M | re.S)
    return [q for _, q in re.findall(r"^\s*(\d+)\.\s+(.+(?:\n\s{2,}\S.*)*)", q.group(1), flags=re.M)]


def estimate_tokens(r):
    """Conservative upper estimate (chars / 3.5) of the largest sequence this item renders."""
    msgs, best_prompt, best_target = [], 0, 0
    for t in r["turns"]:
        msgs.append(t["user"])
        best_prompt = max(best_prompt, sum(len(m) for m in msgs) // 4)
        best_target = max(best_target, (len(t["thinking"]) + len(t["reply"]) + 24) // 3.5 + 1)
        msgs.append(t["reply"])
    return int(best_prompt), int(best_target)


def report(rows):
    by_kind = Counter(r["kind"] for r in rows)
    sides = Counter(r["side"] for r in rows)
    kinds = Counter(r["content_kind"] for r in rows)
    tiers = Counter(r["tier"] for r in rows)
    print(f"  kinds: {dict(sorted(by_kind.items()))}")
    print(f"  sides: {dict(sides)}")
    print(f"  content kinds: {dict(sorted(kinds.items()))}")
    print(f"  tiers: {dict(sorted(tiers.items()))}")
    warns = []
    for k in KINDS:
        if by_kind.get(k, 0) != 8:
            warns.append(f"kind {k}: {by_kind.get(k, 0)} items, expected 8")
    small = min(sides.values(), default=0)
    if small < sum(sides.values()) * 0.4:
        warns.append("counterweight sides off balance")
    missing = CORE_KINDS - set(kinds)
    if missing:
        warns.append(f"core content kinds missing: {sorted(missing)}")
    if sum(v for t, v in tiers.items() if t >= 4) < 2:
        warns.append("fewer than 2 multi-step (tier >= 4) items")
    nv = [r["id"] for r in rows if r["kind"] == "B" and "non-verification" in r.get("tags", [])]
    if len(nv) < 3:
        warns.append(f"kind B has {len(nv)} non-verification items, need at least 3")
    for r in rows:
        p, t = estimate_tokens(r)
        if p + t > 4096 or t > 3000:
            warns.append(f"{r['id']}: size estimate over limits (prompt ~{p}, target ~{t})")
    return warns


def render(rows, path):
    out = ["# Stage 1 training data — pilot, trait 4 (honest about uncertainty)",
           "",
           f"*Generated from `{path.name}` by `checks.py --render`. Edit the JSONL, not this file.*",
           "",
           "The role-shaping curriculum (handoff of 2026-10-10): 40 items for one trait, 8 of each "
           "kind — A identity and purpose, B thinking patterns (at least 3 non-verification), "
           "C contrasting cases, D the role in action with if–then self-talk, E quiz items each "
           "with one pre-written variant. Every assistant reply, whole, is the training target.",
           ""]
    for r in rows:
        keys = []
        if r["kind"] == "A":
            keys = ["key_points: " + "; ".join(r["key_points"])]
        elif r["kind"] == "B":
            keys = ["answer: " + r["answer"]]
        elif r["kind"] == "C":
            keys = [f"planted_flaw: {r['planted_flaw']}",
                    f"right answer: {r['verdict']} / "
                    + (" | ".join(r["deciding"]) if r["deciding"] else "NONE") + f" / {r['escalate']}"]
        elif r["kind"] == "D":
            keys = ["key: " + r["verdict"] + " / "
                    + (" | ".join(r["deciding"]) if r["deciding"] else "NONE") + f" / {r['escalate']}"]
        elif r["kind"] == "E":
            q = r["quiz"]
            body = "; ".join(q["key_points"]) if q["type"] == "written" else \
                f"{q['verdict']} / " + (" | ".join(q["deciding"]) if q["deciding"] else "NONE") \
                + f" / {q['escalate']}"
            keys = [f"quiz ({q['type']}): {body}", f"variant key: {json.dumps(r['variant']['key'])}"]
        tags = f" · tags: {', '.join(r['tags'])}" if r.get("tags") else ""
        src = "invented (key by construction)" if r["invented"] else \
            "; ".join(f"{s['repo']}@{s['commit'][:12]}:{s['path']}#{s['lines']} ({s['licence']})"
                      for s in r["sources"])
        sec = [f"### {r['id']} · kind {r['kind']} · {r['content_kind']} · tier {r['tier']} · "
               f"{r['side']}{tags}", ""]
        for t in r["turns"]:
            sec += ["**User:**", "```text", t["user"], "```", ""]
            if t["thinking"].strip():
                sec += ["**Thinking:**", "```text", t["thinking"], "```", ""]
            sec += ["**Reply:**", "```text", t["reply"], "```", ""]
        if r["kind"] == "E":
            sec += ["**Variant user:**", "```text", r["variant"]["user"], "```", ""]
        sec += ["**Key:** " + " · ".join(keys), "", f"*{src}*", ""]
        out += sec
    path.with_suffix(".md").write_text("\n".join(out), encoding="utf-8")


def main(argv):
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    path = Path(argv[0])
    rows = load(path)
    failures = []

    try:
        data.validate(rows)
        print("schema: clean (ASPIRE data.validate)")
    except data.DataError as e:
        failures.append(f"schema: {e}")

    held = load(RUNS / "stage1-taskset" / "taskset-pilot-20.jsonl") + \
        load(RUNS / "stage1-taskset" / "taskset-review-10.jsonl")
    hits = data.leakage(rows, held, interview_questions())
    for h in hits:
        failures.append(f"leakage: {h}")
    print(f"leakage vs pilot/review task sets + pre-interview: {len(hits)} hit(s)")

    print(f"{path.name}: {len(rows)} rows")
    warns = report(rows)
    for w in warns:
        print(f"  warn: {w}")
    for f in failures:
        print(f"  FAIL: {f}")
    if failures:
        return 1
    print(f"checks: clean ({len(warns)} warning(s))")
    if "--render" in argv:
        render(rows, path)
        print(f"rendered {path.with_suffix('.md').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
