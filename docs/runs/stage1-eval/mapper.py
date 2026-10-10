"""Mechanical mapper (mechanical-v0): writes the plan's mapped-tree records from raw traces.

It only records what can be read off the text without judgement:
- each paragraph of the thinking is a step;
- a step that quotes a material line (that line appears in the step, or a quoted run of 15+ characters lies
  inside that line) is a `check`; quoting a line already checked makes it a `revisit`;
- the matched line is mapped to the ideal tree's part that names it;
- the last step of the final turn that names a verdict is the `settle`.
Everything else keeps `step_type: null` and `parent_step_id` points at the previous step (a flat chain). Real
tree structure (decompose, prune, guess, branching) needs the person or controlled-model mapper, which writes
the same records with `method` set to say so.
"""

import json
import re
from pathlib import Path

METHOD = "mechanical-v0"
QUOTE = re.compile(r"[\"“'`]([^\"”'`]{15,})[\"”'`]")


def material_lines(material: str) -> list[tuple[int, str]]:
    return [(i, s.strip()) for i, s in enumerate(material.splitlines()) if len(s.strip()) >= 12]


def match_line(step: str, lines):
    for i, s in lines:
        if s in step:
            return i, s, s
    for q in QUOTE.findall(step):
        q = q.strip()
        for i, s in lines:
            if q in s:
                return i, s, q
    return None


def part_for(line: str, tree: dict):
    def walk(parts):
        for p in parts:
            if any(line in x.strip() or x.strip() in line for x in p.get("lines", []) if x.strip()):
                return p["id"]
            hit = walk(p.get("children", []))
            if hit:
                return hit
    return walk(tree["parts"])


def map_trace(rec: dict, task: dict, tok) -> tuple[list[dict], dict]:
    lines = material_lines(task["material"])
    tree = task["ideal_tree"]
    steps, seen = [], set()
    trace_id = f"{rec['phase']}/{rec['arm']}/s{rec['seed']}/{task['id']}"
    for turn in rec["turns"]:
        paras = [p.strip() for p in re.split(r"\n\s*\n", turn["thinking"]) if p.strip()]
        for p in paras:
            m = match_line(p, lines)
            step_type, matched, quoted, part = None, None, None, None
            if m:
                idx, line, quoted = m
                matched = {"index": idx, "text": line}
                part = part_for(line, tree)
                step_type = "revisit" if idx in seen else "check"
                seen.add(idx)
            steps.append({"task_id": task["id"], "category": task["category"], "tier": task["tier"],
                          "phase": rec["phase"], "arm": rec["arm"], "seed": rec["seed"], "trace_id": trace_id,
                          "turn": turn["turn"], "step_id": len(steps),
                          "parent_step_id": len(steps) - 1 if steps else None, "step_type": step_type,
                          "part_id": part, "quoted_span": quoted, "matched_line": matched, "traits": [],
                          "tokens": len(tok(p)["input_ids"]),
                          "trace_correct": rec["grade"]["verdict_correct"], "method": METHOD})
    for s in reversed(steps):
        if s["turn"] != rec["turns"][-1]["turn"]:
            break
        if s["step_type"] is None:
            s["step_type"] = "settle"
            break
    trace = {"trace_id": trace_id, "task_id": task["id"], "phase": rec["phase"], "arm": rec["arm"],
             "seed": rec["seed"], "verdict": rec["final"]["verdict"], "deciding": rec["final"]["deciding"],
             "escalate": rec["final"]["escalate"], "correct": rec["grade"]["verdict_correct"],
             "ideal_tree_id": task["id"], "total_tokens": sum(t["new_tokens"] for t in rec["turns"]),
             "truncated": any(t["truncated"] for t in rec["turns"]), "method": METHOD}
    return steps, trace


def map_run(out: Path, tasks: list[dict], tok) -> None:
    by_id = {t["id"]: t for t in tasks}
    steps, traces = [], []
    for line in open(out / "raw.jsonl", encoding="utf-8"):
        rec = json.loads(line)
        s, t = map_trace(rec, by_id[rec["task_id"]], tok)
        steps += s
        traces.append(t)
    (out / "mapped-steps.jsonl").write_text("".join(json.dumps(s, ensure_ascii=False) + "\n" for s in steps),
                                            encoding="utf-8", newline="\n")
    (out / "mapped-traces.jsonl").write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in traces),
                                             encoding="utf-8", newline="\n")
    n = len(traces)
    right = sum(t["correct"] for t in traces)
    print(f"mapped {n} traces, {len(steps)} steps; verdict correct {right}/{n}")
