"""The NO-ROLE arm for Phase B: one role-free item per ROLE item, derived mechanically from the item's own fields.

    python no_role.py --data <ROLE .jsonl> [...] --out <NO-ROLE .jsonl>      (prints the composition table)

Design as approved by R&D (2026-10-10). Every NO-ROLE item keeps its ROLE item's lesson, kind, tier, material,
statement, keys, turn count and pair id, so sequence counts, floors and shares match by construction. Only the
assistant targets change; nothing is written by a model.

- C, D, E-verdict, and B items whose reply gives a VERDICT (verification on material): thinking is a plain
  read-through ("The material says:" and its lines verbatim, in order, stopped at a whole line near the ROLE
  turn's target tokens), then one always-true closing sentence; the reply is the three lines (C adds "The other
  checker answered X. The answer is Y."). Later turns read only their new_material, or "No new material.".
- B puzzles (no VERDICT in the reply): the user turn without its pattern-naming sentences; thinking restates
  the givens one per line; the reply is "Answer: <answer>.". A B item that fits neither is refused.
- A and E-written: a material-recall counterpart from the same lesson's C, D and E-verdict materials, chosen so
  exposure stays even; the reply is the read-through.
- Shorter, never padded: when the material runs out, the item is shorter, and the token ratio is reported.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

import data
import harness

PATTERN_WORDS = re.compile(r"\b(tree|trees|prun\w*|falsif\w*|eliminat\w*|grid|grids|constraint grid)\b", re.I)
VERDICT_WORD = re.compile(r"\b(supported|unsupported|cannot_tell|can'?t tell)\b", re.I)
LOW, HIGH = 0.9, 1.1  # a read-through stops once it reaches LOW x the ROLE target, and never passes HIGH x


class BuildError(ValueError):
    pass


def target_text(thinking: str, reply: str) -> str:
    return thinking + "\n" + reply


def three_lines(keys: dict) -> str:
    return (f"VERDICT: {keys['verdict']}\nDECIDING: {' | '.join(keys['deciding']) or 'NONE'}\n"
            f"ESCALATE: {keys['escalate']}")


def closing(keys: dict, statement: str) -> str:
    """Always true of the material: the keyed lines settle it, or nothing in it does."""
    if keys["verdict"] == "cannot_tell":
        return f'Nothing in the material settles the statement: "{statement}".' if statement else \
            "Nothing in the material settles the statement."
    q = " | ".join(f'"{d}"' for d in keys["deciding"])
    return f"The line {q} settles the statement." if len(keys["deciding"]) == 1 else \
        f"The lines {q} settle the statement."


BLOCK_START = re.compile(r"^\s*(?:[-*+]\s|\d+[.)]\s|#|\|)")
STATEMENT_LINE = re.compile(r"^\s*(?:statement|claim|case to check[^:]*)\s*:", re.I)


def blocks(text: str) -> list[str]:
    """The material in whole units: a bullet, heading, table row or paragraph, with its wrapped lines."""
    out: list[list[str]] = []
    blank = True
    for line in text.splitlines():
        if not line.strip():
            blank = True
            continue
        if blank or BLOCK_START.match(line) or not out:
            out.append([line])
        else:
            out[-1].append(line)
        blank = False
    return ["\n".join(b) for b in out]


def fit(units: list[str], head: str, tail: str, reply: str, target: int, count, must=()) -> str:
    """head + as many whole units as fit + tail, stopping near the ROLE target; never padded. A unit holding a
    `must` line (a DECIDING line) that the budget cut off is appended after a "…" line, so the closing always
    refers to a line that was read."""
    chosen: list[str] = []
    for unit in units:
        body = "\n".join([head] + chosen + [unit] + ([tail] if tail else []))
        if count(target_text(body, reply)) > HIGH * target:
            break
        chosen.append(unit)
        if count(target_text(body, reply)) >= LOW * target:
            break
    late = []
    for d in must:
        if data.ws(d) in data.ws("\n".join(chosen + late)):
            continue
        late.append(next((u for u in units if data.ws(d) in data.ws(u)), d))
    if late:
        chosen += ["…"] + late
    return "\n".join(([head] + chosen if chosen else []) + ([tail] if tail else []))


def statement_of(user: str) -> str:
    return data._statement(user).strip().strip('"“”').strip()


def checker_verdict(r: dict) -> str:
    """The other checker's verdict in a kind C item: the `checker_verdict` field, else the one verdict word in the
    user turn outside the material and the statement."""
    if r.get("checker_verdict"):
        return r["checker_verdict"]
    u = r["turns"][0]["user"].replace(r.get("material") or "", "")
    u = re.sub(r"(?im)^.*statement\s*:.*$", "", u)
    found = {w.lower().replace("can't tell", "cannot_tell").replace("cant tell", "cannot_tell")
             for w in VERDICT_WORD.findall(u)}
    if not found and re.search(r"\b(never gave|gave no|without giving)\b[^.]*\bverdict", u, re.I):
        return "none"
    if len(found) != 1:
        raise BuildError(f"{r['id']}: can't read the other checker's verdict ({sorted(found) or 'none'}); "
                         f"add a checker_verdict field")
    return found.pop()


def b_material(r: dict) -> str:
    """A verification B item's material: the field, else the user turn's lines before its Statement line (a
    leading header line ending in ':' dropped)."""
    if r.get("material"):
        return r["material"]
    lines = r["turns"][0]["user"].splitlines()
    idx = next((i for i, x in enumerate(lines) if STATEMENT_LINE.match(x)), None)
    if not idx:
        raise BuildError(f"{r['id']}: kind B with a VERDICT reply but no material and no Statement line")
    body = lines[:idx]
    if body and body[0].rstrip().endswith(":"):
        body = body[1:]
    return "\n".join(body)


def verdict_turns(r: dict, count) -> list[dict]:
    """Rule 1: read-through thinking and the three lines, per turn."""
    kind = r["kind"]
    if kind == "E":
        final = r["quiz"]
    elif kind == "B":
        p = harness.parse_answer(r["turns"][-1]["reply"])
        final = {"verdict": p["verdict"], "deciding": [] if (p["deciding"] or "NONE").strip().upper() == "NONE"
                 else [x.strip() for x in p["deciding"].split(" | ")], "escalate": (p["escalate"] or "no").strip()}
    else:
        final = r
    material = b_material(r) if kind == "B" else (r.get("material") or "")
    statement = statement_of(r["turns"][0]["user"])
    if kind == "C" and not statement:
        m = re.search(r'statement\s*:\s*"([^"]+)"', r["turns"][0]["user"], re.I)
        statement = m.group(1) if m else ""
    out = []
    n = len(r["turns"])
    for i, t in enumerate(r["turns"], 1):
        keys = final if i == n else (r.get("turn_keys") or {}).get(str(i))
        if keys is None:
            raise BuildError(f"{r['id']} turn {i}: no keys for an earlier verdict turn")
        reply = three_lines(keys)
        if kind == "C" and i == n:
            cv = checker_verdict(r)
            said = "gave no verdict" if cv == "none" else f"answered {cv}"
            reply = f"The other checker {said}. The answer is {keys['verdict']}.\n" + reply
        target = count(target_text(t["thinking"], t["reply"]))
        if i == 1:
            thinking = fit(blocks(material), "The material says:", closing(keys, statement), reply, target, count,
                           must=keys["deciding"])
        elif t.get("new_material"):
            thinking = fit(list(t["new_material"]), "The new material says:", closing(keys, statement), reply,
                           target, count, must=[d for d in keys["deciding"] if d in "\n".join(t["new_material"])])
        else:
            thinking = "No new material.\n" + closing(keys, statement)
        user = t["user"]
        if kind == "B" and i == 1:  # the same shape as C, D and E: material, the statement, a neutral instruction
            claim = next((x for x in user.splitlines() if STATEMENT_LINE.match(x)), None)
            if claim is None:
                raise BuildError(f"{r['id']}: kind B verification prompt has no Statement (or claim) line")
            user = f"Material:\n{material}\n{claim.strip()}\nAnswer in the three lines."
        out.append({"user": user, "thinking": thinking, "reply": reply,
                    **({"new_material": t["new_material"]} if t.get("new_material") else {})})
    return out


def strip_patterns(user: str) -> str:
    kept = []
    for line in user.splitlines():
        sents = [s for s in re.split(r"(?<=[.!?])\s+", line) if s and not PATTERN_WORDS.search(s)]
        if sents:
            kept.append(" ".join(sents))
    return "\n".join(kept)


def puzzle_turns(r: dict, count) -> list[dict]:
    """Rule 3: a non-verification puzzle; its givens restated and the answer."""
    if len(r["turns"]) != 1:
        raise BuildError(f"{r['id']}: a kind B puzzle with {len(r['turns'])} turns")
    t = r["turns"][0]
    user = strip_patterns(t["user"])
    if not user.strip() or PATTERN_WORDS.search(user):
        raise BuildError(f"{r['id']}: kind B puzzle text is empty or still names a pattern after stripping")
    reply = f"Answer: {r['answer']}."
    givens = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", user) if s.strip() and "?" not in s]
    thinking = fit(givens, "Given:", "", reply, count(target_text(t["thinking"], t["reply"])), count)
    return [{"user": user, "thinking": thinking, "reply": reply}]


def recall_turns(r: dict, material: str, count, written_quiz: bool) -> list[dict]:
    t = r["turns"][0]
    user = (f"Read this material and say what it states:\n{material}" if written_quiz
            else f"Here is some material:\n{material}\nWhat does it say?")
    reply = fit(blocks(material), "It says:", "", "", count(target_text(t["thinking"], t["reply"])), count)
    return [{"user": user, "thinking": "", "reply": reply}]


def is_verification(r: dict) -> bool:
    return r["kind"] in ("C", "D") or (r["kind"] == "E" and r["quiz"].get("type") == "verdict") or \
        (r["kind"] == "B" and bool(re.search(r"(?m)^\s*VERDICT\s*:", r["turns"][-1]["reply"])))


def build(rows: list[dict], count) -> tuple[list[dict], dict]:
    """The NO-ROLE items, in the same order, and the per-material exposure counts per arm."""
    pool = defaultdict(list)  # lesson -> [material] from C, D and E-verdict items
    base = Counter()
    for r in rows:
        if r["kind"] in ("C", "D") or (r["kind"] == "E" and r["quiz"].get("type") == "verdict"):
            m = r.get("material") or ""
            if m and m not in pool[r["lesson"]]:
                pool[r["lesson"]].append(m)
            base[m] += 1
    exposure_nr = Counter(base)
    out = []
    for r in rows:
        nr = {k: v for k, v in r.items() if k not in ("turns", "key_points", "variant")}
        nr.update(id=r["id"] + "-nr", arm="NO-ROLE")
        if is_verification(r):
            nr["turns"] = verdict_turns(r, count)
            if r["kind"] == "E":
                nr["variant"] = r["variant"]
        elif r["kind"] == "B":
            if not r.get("answer"):
                raise BuildError(f"{r['id']}: kind B with neither a VERDICT reply nor an answer")
            nr["turns"] = puzzle_turns(r, count)
        elif r["kind"] in ("A", "E"):
            mats = pool.get(r["lesson"])
            if not mats:
                raise BuildError(f"{r['id']}: lesson {r['lesson']} has no C, D or E-verdict material to recall")
            m = min(mats, key=lambda x: (exposure_nr[x], mats.index(x)))
            exposure_nr[m] += 1
            nr["turns"] = recall_turns(r, m, count, written_quiz=r["kind"] == "E")
            nr["recall_material_of"] = r["id"]
            if r["kind"] == "E":
                nr["quiz"] = {"type": "recall"}
                nr["variant"] = {"user": f"Put another way: what does this material state?\n{m}",
                                 "key": ["the material's lines"]}
        else:
            raise BuildError(f"{r['id']}: unknown kind {r['kind']}")
        out.append(nr)
    return out, {"ROLE": dict(base), "NO-ROLE": dict(exposure_nr)}


def composition(rows: list[dict], count) -> dict:
    """Per lesson x kind: counts by verdict and escalate, multi-line, NONE, pairs, two-turn changed/stood,
    tiers, sequences and target tokens."""
    table = defaultdict(Counter)
    for r in rows:
        key = (r["lesson"], r["kind"])
        c = table[key]
        c["items"] += 1
        c["sequences"] += len(r["turns"])
        c[f"tier {r['tier']}"] += 1
        c["target_tokens"] += sum(count(target_text(t["thinking"], t["reply"])) for t in r["turns"])
        keys = data._verdict_keys(r)
        if isinstance(keys, dict):
            c[f"verdict {keys['verdict']}"] += 1
            c["escalate yes"] += (keys.get("escalate") or "").startswith("yes (")
            c["multi-line"] += 2 <= len(keys.get("deciding") or []) <= 4
            c["NONE"] += not keys.get("deciding")
            c["paired"] += r.get("pair") is not None
            if r["kind"] == "D" and len(r["turns"]) >= 2:
                first = (r.get("turn_keys") or {}).get("1", {}).get("verdict")
                c["two-turn changed" if first != keys["verdict"] else "two-turn stood"] += 1
    return table


def compare(role: list[dict], nr: list[dict], count, min_ratio: float = 0.85) -> tuple[str, float]:
    """The side-by-side table. Refuses any differing count cell; flags lesson x kind token ratios under
    min_ratio; returns the table text and the overall NO-ROLE / ROLE target-token ratio."""
    a, b = composition(role, count), composition(nr, count)
    lines = ["lesson kind | cell | ROLE | NO-ROLE"]
    flags = []
    tot_a = tot_b = 0
    for key in sorted(set(a) | set(b)):
        for cell in sorted(set(a[key]) | set(b[key])):
            va, vb = a[key][cell], b[key][cell]
            if cell != "target_tokens" and va != vb:
                raise BuildError(f"{key} {cell}: ROLE {va} != NO-ROLE {vb}")
            lines.append(f"{key[0]} {key[1]} | {cell} | {va} | {vb}")
        ta, tb = a[key]["target_tokens"], b[key]["target_tokens"]
        tot_a, tot_b = tot_a + ta, tot_b + tb
        if ta and tb < min_ratio * ta:
            flags.append(f"{key[0]} {key[1]}: NO-ROLE target tokens {tb} of ROLE {ta} ({tb / ta:.2f})")
    ratio = tot_b / tot_a if tot_a else 1.0
    lines.append(f"ALL | target_tokens | {tot_a} | {tot_b} (ratio {ratio:.3f})")
    lines += ["FLAG " + f for f in flags]
    return "\n".join(lines), ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(data.EVAL_DIR))
    import student
    tok = student.load_tokenizer()

    def count(x):
        return len(tok(x, add_special_tokens=False)["input_ids"])

    role = [r for f in a.data for r in data.load(f)]
    nr, exposure = build(role, count)
    table, ratio = compare(role, nr, count)
    with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in nr)
    print(table)
    print("exposure per material (ROLE / NO-ROLE):")
    for m in exposure["NO-ROLE"]:
        print(f"  {exposure['ROLE'].get(m, 0)} / {exposure['NO-ROLE'][m]}  {m.splitlines()[0][:70]!r}")
    if ratio < 0.85:
        print(f"LIMITATION: overall NO-ROLE / ROLE target-token ratio {ratio:.3f} is under 0.85")


if __name__ == "__main__":
    main()
