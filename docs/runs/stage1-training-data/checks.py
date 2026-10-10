#!/usr/bin/env python
"""Checks + readable rendering for Stage 1 training examples (the verifier role's data set).

    python checks.py pilot-40.jsonl            validate, print a summary (exit 1 on any failure)
    python checks.py pilot-40.jsonl --render   the same, then rewrite pilot-40.md from the JSONL

These checks are the mechanical half of R&D's key check. They verify:
- the row schema (exactly the fields the trainer reads, no more, no less);
- every DECIDING line is a verbatim substring of the material (score.py-style normalisation:
  whitespace collapsed, ** and __ removed) — the "going to the source" rule made executable;
- verdict/DECIDING consistency (a cannot_tell has DECIDING NONE, and only it);
- the id/lesson/trait/side vocabulary, and that both sides of each trait's counterweight are
  present in roughly equal numbers;
- tier sanity against the pinned min_steps rule's hard floor (3 + turns + escalation);
- no injection-trap language in a pilot-scope file (traps belong only to trait 17 and a few
  Lesson 1/3 items, and are always tagged when they appear);
- no stamped-out rows: claims closer than a 5-gram Jaccard of 0.8 fail; 0.6 and over warns
  so a person can look.

A row-by-row read (instant agreement of the key, truth of every thinking step) is still a
person's job; this script exists so the batch fails loudly before that person sits down.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

FIELDS = {"id", "lesson", "trait", "side", "content_kind", "tier", "material", "statement",
          "turns", "thinking", "verdict", "deciding", "escalate", "sources", "invented", "key_basis"}

LESSON_TRAITS = {
    "trait-01": ("skeptical", ("rightly-rejects", "rightly-accepts")),
    "trait-02": ("goes to the source", ("answers-from-the-material", "ignores-the-second-hand")),
    "trait-03": ("precise", ("exact-line", "no-paraphrase")),
    "trait-04": ("honest about uncertainty", ("honest-cannot-tell", "confident-when-settled")),
    "trait-05": ("independent", ("holds-under-pressure", "changes-on-new-material")),
    "trait-06": ("breaks things down", ("splits-and-prunes", "accepts-when-every-part-holds")),
    "trait-07": ("keeps proportion", ("quick-on-the-obvious", "careful-on-the-hard")),
    "trait-08": ("knows its limits", ("escalates-advice", "answers-without-escalating")),
    "trait-09": ("consistent",),
    "trait-10": ("takes statements literally", ("reads-exactly",)),
    "trait-11": ("knows what a mistake costs", ("earns-the-acceptance",)),
    "trait-12": ("checks the version and date", ("mismatched-version", "right-version")),
    "trait-13": ("doesn't agree to please", ("holds-the-verdict", "agrees-when-earned")),
    "trait-14": ("doesn't fill gaps", ("refuses-the-gap", "quotes-only-what-is-there")),
    "trait-15": ("doesn't go round in circles", ("stops-when-settled",)),
    "trait-16": ("doesn't nitpick", ("accepts-trivial-differences", "rejects-real-differences")),
    "trait-17": ("doesn't take orders from the material", ("ignores-the-order",)),
    "trait-18": ("doesn't fix things", ("judges-as-written",)),
}

LESSONS = set(LESSON_TRAITS) | {"lesson-1", "lesson-2", "lesson-3"}

KINDS = {"notice", "policy", "label", "timetable", "changelog", "code", "record", "puzzle"}

VERDICTS = {"supported", "unsupported", "cannot_tell"}

# Injection-trap language. Traps are deliberate, tagged, and confined to trait 17 plus a few
# Lesson 1/3 items; any of these phrases in another row's material is a build error.
INJECTION = re.compile(
    r"reviewer|checker|\bmark this\b|report this|please answer|note to the|verifier:",
    re.IGNORECASE,
)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("**", "").replace("__", "")).strip()


def grams(s: str, n: int = 5) -> set:
    s = norm(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def check_row(row: dict, ln: int, errors: list, warnings: list) -> None:
    rid = row.get("id", f"line {ln}")
    extra = set(row) - FIELDS
    missing = FIELDS - set(row)
    if extra:
        errors.append(f"{rid}: unknown field(s) {sorted(extra)}")
    if missing:
        errors.append(f"{rid}: missing field(s) {sorted(missing)}")
        return

    if not re.fullmatch(r"(t\d{2}|l[123])-[a-z0-9-]+-\d{2}", row["id"] or ""):
        errors.append(f"{rid}: id does not match the tNN-slug-NN / lN-slug-NN pattern")

    if row["lesson"] not in LESSONS:
        errors.append(f"{rid}: unknown lesson {row['lesson']!r}")
    if row["lesson"] in LESSON_TRAITS:
        want_trait, *sides = LESSON_TRAITS[row["lesson"]]
        if row["trait"] != want_trait:
            errors.append(f"{rid}: trait {row['trait']!r} does not match lesson {row['lesson']}")
        if sides and row["side"] not in sides[0]:
            errors.append(f"{rid}: side {row['side']!r} not one of {sides[0]}")

    if row["content_kind"] not in KINDS:
        errors.append(f"{rid}: unknown content_kind {row['content_kind']!r}")

    if row["verdict"] not in VERDICTS:
        errors.append(f"{rid}: bad verdict {row['verdict']!r}")
    if not (row["escalate"] == "no" or re.fullmatch(r"yes \(.+\)", row["escalate"] or "")):
        errors.append(f"{rid}: bad escalate value {row['escalate']!r}")

    if not isinstance(row["tier"], int) or row["tier"] < 3:
        errors.append(f"{rid}: tier must be an int >= 3 (root + one check + settle is the floor)")
    floor = 3 + len(row["turns"]) + (0 if row["escalate"] == "no" else 1)
    if isinstance(row["tier"], int) and row["tier"] < floor:
        errors.append(f"{rid}: tier {row['tier']} below the min_steps floor {floor} "
                      f"(turns={len(row['turns'])}, escalate={row['escalate']!r})")

    dec = row["deciding"]
    if not isinstance(dec, list) or not all(isinstance(x, str) and x.strip() for x in dec):
        errors.append(f"{rid}: deciding must be a list of non-empty strings")
        dec = []
    if row["verdict"] == "cannot_tell" and dec:
        errors.append(f"{rid}: a cannot_tell verdict must have DECIDING NONE (empty list)")
    if row["verdict"] != "cannot_tell" and not dec:
        errors.append(f"{rid}: a supported/unsupported verdict needs at least one deciding line")
    if len(dec) > 4:
        errors.append(f"{rid}: DECIDING holds at most 4 lines, got {len(dec)}")
    mat = norm(row["material"])
    for line in dec:
        if norm(line) not in mat:
            errors.append(f"{rid}: deciding line is not verbatim in the material: {line!r}")

    if INJECTION.search(row["material"]) and row["lesson"] != "trait-17":
        errors.append(f"{rid}: injection-trap language in material outside trait 17 "
                      f"(traps are deliberate, tagged, and confined)")

    if row["invented"]:
        if row["sources"]:
            errors.append(f"{rid}: invented material must have no sources")
        if row["key_basis"] != "construction":
            errors.append(f"{rid}: invented material is known by construction")
    else:
        if not row["sources"]:
            errors.append(f"{rid}: real material must name its sources")
        for s in row["sources"]:
            if not isinstance(s, dict) or not s.get("source") or not s.get("licence"):
                errors.append(f"{rid}: source entries need at least source and licence")
            if s.get("invented"):
                errors.append(f"{rid}: real row carries an invented source entry")
        if row["key_basis"] != "source_words":
            errors.append(f"{rid}: real material is known from the source's own words")

    for t in row["turns"]:
        if not isinstance(t, str) or not t.strip():
            errors.append(f"{rid}: turns must be non-empty strings")


def render(rows: list, path: Path) -> None:
    by_trait = defaultdict(list)
    for r in rows:
        by_trait[(r["lesson"], r["trait"])].append(r)

    rule = ("min_steps = (1 per node that is decomposed: the root, plus each part with children) + "
            "(1 per leaf check, in claim order, up to and including the part that settles the root; "
            "every leaf if nothing settles it early; a settled part's later siblings are pruned) + "
            "1 settle + 1 per pressure turn (T) + 1 per injection noticed (I) + 1 per escalation (E). "
            "A task's tier is its minimum steps.")

    out = ["# Stage 1 training data — pilot 40",
           "",
           "*Generated from `pilot-40.jsonl` by `checks.py --render`. Edit the JSONL, not this file.*",
           "",
           "Pilot batch for the Stage 1 verifier-role teaching set: 20 examples for trait 4 "
           "(honest about uncertainty) and 20 for its counterweight, trait 14 (doesn't fill gaps). "
           "Every row is one worked teaching example: material, statement, worked thinking that "
           "visibly uses a Lesson 2 pattern and in which the trait decides at least one step, then "
           "the three answer lines.",
           "",
           f"**Minimum steps (the rule):** {rule}",
           ""]
    for (lesson, trait), items in sorted(by_trait.items()):
        sides = Counter(r["side"] for r in items)
        tiers = Counter(r["tier"] for r in items)
        out += [f"## {lesson}: {trait}",
                "",
                f"Sides: " + ", ".join(f"{k} × {v}" for k, v in sorted(sides.items())) +
                f". Tiers: " + ", ".join(f"{k} × {v}" for k, v in sorted(tiers.items())) + ".",
                ""]
        for r in items:
            deciding = "\nDECIDING: " + " | ".join(r["deciding"]) if r["deciding"] else "\nDECIDING: NONE"
            inv = "Invented material, key known by construction." if r["invented"] else \
                  "Real material: " + "; ".join(f"{s['source']} ({s['licence']})" for s in r["sources"]) + \
                  " Key known from the source's own words."
            out += [f"### {r['id']} · {r['content_kind']} · tier {r['tier']} · {r['side']} · {r['verdict']}",
                    "",
                    "**Material:**", "```text", r["material"], "```",
                    "",
                    f"**Statement:** {r['statement']}",
                    "",
                    "**Thinking:**", "```text", r["thinking"], "```",
                    "",
                    "**Answer:**", "```text",
                    f"VERDICT: {r['verdict']}{deciding}\nESCALATE: {r['escalate']}",
                    "```",
                    "",
                    f"*{inv}*",
                    ""]
    path.write_text("\n".join(out), encoding="utf-8")


def main(argv: list) -> int:
    if not argv or argv[0].startswith("-"):
        print(__doc__)
        return 2
    path = Path(argv[0])
    rows, errors, warnings = [], [], []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"line {ln}: not JSON ({e})")
            continue
        rows.append(row)
        check_row(row, ln, errors, warnings)

    ids = [r["id"] for r in rows]
    for dup, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate id {dup} ({n} times)")

    # stamped-out-template check on claims
    for i in range(len(rows)):
        gi = grams(rows[i]["statement"])
        for j in range(i + 1, len(rows)):
            gj = grams(rows[j]["statement"])
            jac = len(gi & gj) / max(1, len(gi | gj))
            if jac >= 0.8:
                errors.append(f"{rows[i]['id']} ~ {rows[j]['id']}: near-duplicate claims (Jaccard {jac:.2f})")
            elif jac >= 0.6:
                warnings.append(f"{rows[i]['id']} ~ {rows[j]['id']}: similar claims (Jaccard {jac:.2f})")

    # per-trait balance and coverage
    by_trait = defaultdict(list)
    for r in rows:
        by_trait[r["trait"]].append(r)
    for trait, items in sorted(by_trait.items()):
        sides = Counter(r["side"] for r in items)
        total = sum(sides.values())
        small = min(sides.values())
        if small < total * 0.4:
            warnings.append(f"{trait}: counterweight sides off balance ({dict(sides)})")
        verdicts = {r["verdict"] for r in items}
        if verdicts != VERDICTS:
            warnings.append(f"{trait}: verdicts shown are {sorted(verdicts)}, not all three")
        kinds = {r["content_kind"] for r in items}
        if KINDS - kinds:
            warnings.append(f"{trait}: no example of kind(s) {sorted(KINDS - kinds)}")
        tiers = Counter(r["tier"] for r in items)
        if sum(v for t, v in tiers.items() if t >= 4) < 4:
            warnings.append(f"{trait}: fewer than 4 multi-step (tier >= 4) examples")

    print(f"{path.name}: {len(rows)} rows")
    for trait, items in sorted(by_trait.items()):
        sides = Counter(r["side"] for r in items)
        verdicts = Counter(r["verdict"] for r in items)
        tiers = Counter(r["tier"] for r in items)
        print(f"  {trait}: {dict(sides)} | verdicts {dict(verdicts)} | tiers {dict(sorted(tiers.items()))}")
    for w in warnings:
        print(f"  warn: {w}")
    for e in errors:
        print(f"  FAIL: {e}")
    if errors:
        print(f"checks: {len(errors)} failure(s)")
        return 1
    print(f"checks: clean ({len(warnings)} warning(s))")
    if "--render" in argv:
        render(rows, path.with_suffix(".md"))
        print(f"rendered {path.with_suffix('.md').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
