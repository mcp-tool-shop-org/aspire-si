"""The secondary "found the evidence" score and the mechanical miss breakdown (eval plan 7bca450).

    python score_found.py <run dir> --tasks <tasks.jsonl>
        -> writes <run dir>/scored-found.jsonl (per task: found_ok, pinned correct, miss bucket); prints totals

Descriptive only: reported beside the pinned score (score.py, fd59cb50), never instead of it. The pinned score
shows whether the model quotes exactly; this one shows whether it found the right evidence.

Rules, as pre-registered:
- The verdict must match; a cannot_tell key still needs DECIDING NONE. Correction tasks need both turns.
- Lenient normalising, both sides: case folded; backticks, `**`, `__`, straight and curly quotes removed;
  whitespace collapsed; for coverage, the pinned edge trims (leading - * > markers, trailing . ; , :).
- Every given line is a substring of the material under lenient normalising (no paraphrase, no statement).
- A key line is found when a given line contains it, or a given line lies inside it and covers at least half
  its characters. The whole DECIDING string counts as a given line here too, since a quoted table row is split
  on its own " | " separators (as in the pinned rule).
- A key line that is a section or version heading may be left out, except in the wrong-version-or-date
  category.
- All of the key's lines, or one also-sufficient set, must be found.
- The parsing repair of score v2 applies: the lines as given are tried first, then the repaired lines.

Miss buckets, for verdict-right tasks that fail the pinned score (first that applies, on the failing turn):
formatting only (passes this score) / NONE misuse / the statement quoted / not in the material / partial / the
wrong line.
"""

import argparse
import json
import re
from pathlib import Path

import harness
import score
import score_v2

VERSION_CATEGORY = "wrong version or date"


def lenient(x: str) -> str:
    x = x.replace("**", "").replace("__", "").replace("`", "")
    x = re.sub(r"[\"'“”‘’]", "", x)
    return re.sub(r"\s+", " ", x).strip().casefold()


def cov(x: str) -> str:
    return score.core(lenient(x))


def is_heading(line: str) -> bool:
    return bool(re.match(r"^\s*#{1,6}\s", line))


def _found(k: str, given: list[str], joined: str) -> bool:
    if k in joined or any(k in g for g in given):
        return True
    return any(g and g in k and 2 * len(g) >= len(k) for g in given + [joined])


def found_lines(given, key_lines, alt_sets, material, verdict_key, category) -> bool:
    if verdict_key == "cannot_tell":
        return not given
    if not given:
        return False
    mat = lenient(material)
    if not all(lenient(g) and lenient(g) in mat for g in given):
        return False
    g = [cov(x) for x in given]
    joined = cov(" | ".join(given))
    for want in [key_lines] + list(alt_sets):
        need = [k for k in want if category == VERSION_CATEGORY or not is_heading(k)] or list(want)
        if all(_found(cov(k), g, joined) for k in need):
            return True
    return False


def found_ok_turn(deciding, key_lines, alt_sets, material, verdict_key, category) -> bool:
    args = (key_lines, alt_sets, material, verdict_key, category)
    return found_lines(score.lines_of(deciding), *args) or found_lines(score_v2.repaired(deciding), *args)


def _turns(rec, task):
    """(deciding, key lines, alt sets, material, verdict key) for each scored turn, final turn last."""
    key = task["key"]
    out = []
    if "turn1" in key:
        p1 = harness.parse_answer(rec["turns"][0]["answer"])
        out.append((p1["deciding"], key["turn1"]["deciding"], [], task["material"], key["turn1"]["verdict"]))
    material = task["material"] + "\n" + "\n".join(task.get("new_material", []))
    out.append((rec["final"]["deciding"], key["deciding"], key.get("also_sufficient", []), material,
                key["verdict"]))
    return out


def found_ok(rec, task):
    v_ok, _ = score.score(rec, task, score.norm)
    d_ok = all(found_ok_turn(*t, task["category"]) for t in _turns(rec, task))
    return v_ok, d_ok


def bucket(rec, task) -> str:
    """The miss bucket of a verdict-right task that fails the pinned score."""
    if found_ok(rec, task)[1]:
        return "formatting only"
    for deciding, keys, alts, material, vkey in _turns(rec, task):
        if found_ok_turn(deciding, keys, alts, material, vkey, task["category"]):
            continue
        raw = score.lines_of(deciding)
        lines = score_v2.repaired(deciding)
        if vkey == "cannot_tell" or (len(raw) > 1 and any(x.upper() == "NONE" for x in raw)):
            return "NONE misuse"
        if not lines:
            return "not in the material"
        mat, stmt = lenient(material), lenient(task.get("claim", ""))
        outside = [x for x in lines if lenient(x) not in mat]
        if outside:
            return "the statement quoted" if all(lenient(x) in stmt for x in outside) else "not in the material"
        g = [cov(x) for x in lines]
        joined = cov(" | ".join(lines))
        wanted = [cov(k) for k in keys] + [cov(k) for a in alts for k in a]
        if any(_found(k, g, joined) or any(x and x in k for x in g) for k in wanted):
            return "partial"
        return "the wrong line"
    return "partial"  # every turn finds its evidence alone, yet the task fails: not reached in practice


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
        v, d = found_ok(rec, t)
        pv, pd = score.score(rec, t, score.norm)
        rows.append({"task_id": rec["task_id"], "category": t["category"], "key": t["key"]["verdict"],
                     "seed": rec["seed"], "phase": rec["phase"], "arm": rec["arm"], "scorer": "found",
                     "verdict_ok": v, "found_ok": v and d, "pinned_correct": pv and pd,
                     "miss": bucket(rec, t) if pv and not pd else None})
    (Path(a.run) / "scored-found.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows),
                                                    encoding="utf-8", newline="\n")
    n = len(rows)
    misses = {}
    for r in rows:
        if r["miss"]:
            misses[r["miss"]] = misses.get(r["miss"], 0) + 1
    print(f"{Path(a.run).name} [found]: found {sum(r['found_ok'] for r in rows)}/{n}; pinned "
          f"{sum(r['pinned_correct'] for r in rows)}/{n}; verdict-only {sum(r['verdict_ok'] for r in rows)}/{n}; "
          f"misses {dict(sorted(misses.items(), key=lambda x: -x[1]))}")


if __name__ == "__main__":
    main()
