"""R&D's remaining batch-2 fixes, applied to the items named on PR #82 and nothing else.

Assert-guarded. Counts, digit edits and deciding spans are checked here, not by eye.
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent
sys.path.insert(0, str(RUNS / "stage1-train"))
sys.path.insert(0, str(RUNS / "stage1-eval"))
import data  # noqa: E402


def ws(x: str) -> str:
    return data.ws(x)


def sub_once(text: str, old: str, new: str, where: str) -> str:
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{where}: expected 1 occurrence, found {n}: {old[:80]!r}")
    return text.replace(old, new, 1)


def load(path: Path) -> tuple[list[str], list[dict]]:
    raw = path.read_text(encoding="utf-8").splitlines()
    return raw, [json.loads(line) for line in raw if line.strip()]


def dump_line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False)


def quote_balance(text: str, where: str) -> None:
    if text.count("`") % 2:
        raise SystemExit(f"{where}: unbalanced backtick")
    if text.count('"') % 2:
        raise SystemExit(f"{where}: unbalanced straight double quote")
    for ch in "“”‘’":
        if text.count(ch) % 2:
            raise SystemExit(f"{where}: unbalanced curly quote {ch}")
    if text.count("**") % 2:
        raise SystemExit(f"{where}: unbalanced **")


def require_in(material: str, fragment: str, where: str) -> None:
    if ws(fragment) not in ws(material or ""):
        raise SystemExit(f"{where}: deciding span not in the material: {fragment[:80]!r}")


def fix_t01(rows: list[dict]) -> None:
    b = next(r for r in rows if r["id"] == "T01-B-01")
    span = (
        '`method: "exact" | "monte-carlo"` (+ `samples`), so callers can tell exact '
        "probabilities from sampled estimates. The CLI labels Monte-Carlo results "
        "(`~ estimated via Monte Carlo (N samples) — not exact`)"
    )
    require_in(b["material"], span, "T01-B-01")
    reply = b["turns"][0]["reply"]
    old_line = next(line for line in reply.split("\n") if line.startswith("DECIDING:"))
    if "method:" in old_line or "not exact" not in old_line:
        raise SystemExit(f"T01-B-01: unexpected DECIDING line: {old_line!r}")
    b["turns"][0]["reply"] = reply.replace(old_line, "DECIDING: " + span, 1)

    c = next(r for r in rows if r["id"] == "T01-C-07")
    contract = "Roll follows a deliberate two-code contract — a stability promise scripts can rely on:"
    code = "`1`"
    meaning = "Any error — bad expression, validation failure, missing loot file, or a cap exceeded"
    for frag in (contract, code, meaning):
        require_in(c["material"], frag, "T01-C-07")
    if c["deciding"] != [meaning]:
        raise SystemExit(f"T01-C-07: unexpected deciding key {c['deciding']!r}")
    c["deciding"] = [contract, code, meaning]
    t = c["turns"][0]
    t["reply"] = sub_once(
        t["reply"],
        "The table they glanced at already contradicted the claim by name.",
        "The table already contradicted the claim by name.",
        "T01-C-07 reply",
    )
    t["reply"] = sub_once(
        t["reply"],
        "DECIDING: " + meaning,
        "DECIDING: " + " | ".join(c["deciding"]),
        "T01-C-07 deciding",
    )
    if "glanced" in t["reply"] or "glanced" in t["thinking"]:
        raise SystemExit("T01-C-07: glanced remains")


def fix_t14(rows: list[dict]) -> None:
    c4 = next(r for r in rows if r["id"] == "T14-C-04")
    c4["turns"][0]["reply"] = sub_once(
        c4["turns"][0]["reply"],
        "nothing about the answer was inferred",
        "the heading and the past-tense bullet together settle it",
        "T14-C-04",
    )

    c8 = next(r for r in rows if r["id"] == "T14-C-08")
    t = c8["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        " A different entry handles that topic, and it isn't part of the material here.",
        "",
        "T14-C-08 thinking",
    )
    if "this entry never mentions" not in t["thinking"] and "never mentioned in this entry" not in t["thinking"]:
        raise SystemExit("T14-C-08: silence sentence missing")
    # The silence is already stated. Make the required wording explicit and drop nothing else true.
    t["thinking"] = sub_once(
        t["thinking"],
        "Inline code spans — the claim's subject — are never mentioned in this entry, in either direction:",
        "This entry never mentions inline code spans, in either direction:",
        "T14-C-08 silence wording",
    )
    t["user"] = sub_once(
        t["user"],
        "this entry never says a word; that's documented under a different tool I wasn't given.",
        "this entry never mentions inline code spans.",
        "T14-C-08 checker trace",
    )
    t["reply"] = sub_once(
        t["reply"],
        "Ending at cannot_tell instead of borrowing behavior from a sibling tool is exactly the discipline the statement was testing.",
        "Ending at cannot_tell, because this entry never mentions inline code spans, is exactly the discipline the statement was testing.",
        "T14-C-08 reply",
    )
    blob = t["thinking"] + "\n" + t["reply"] + "\n" + t["user"]
    for banned in ("different entry", "different tool", "sibling tool"):
        if banned in blob:
            raise SystemExit(f"T14-C-08: still says {banned!r}")

    d7 = next(r for r in rows if r["id"] == "T14-D-07")
    d7["turns"][0]["thinking"] = sub_once(
        d7["turns"][0]["thinking"],
        "prices three day-groups",
        "lists three day-groups",
        "T14-D-07",
    )


def fix_t16(rows: list[dict]) -> None:
    a8 = next(r for r in rows if r["id"] == "T16-A-08")
    # gate (4) -> gatelodge (9) is one compound word, five letters longer.
    assert " " not in "gatelodge"
    assert len("gatelodge") - len("gate") == 5
    admin, adams = "Admin", "Adams"
    positional = [i for i, (x, y) in enumerate(zip(admin, adams)) if x == y]
    assert positional == [0, 1]  # A and d only
    a8["turns"][0]["reply"] = sub_once(
        a8["turns"][0]["reply"],
        'share every letter until the object grows a second word — lodge, five letters that send someone to a different building.',
        'share the letters of "gate", and the object is one compound word, gatelodge, five letters longer — those letters send someone to a different building.',
        "T16-A-08 compound",
    )
    a8["turns"][0]["reply"] = sub_once(
        a8["turns"][0]["reply"],
        "the names keep three letters (A, d, m) and the rest disagree",
        "positionally only A and d match, and the rest of the names disagree",
        "T16-A-08 letters",
    )
    reply = a8["turns"][0]["reply"]
    if "second word" in reply or "(A, d, m)" in reply:
        raise SystemExit("T16-A-08: old wording remains")

    b6 = next(r for r in rows if r["id"] == "T16-B-06")
    t = b6["turns"][0]
    t["user"] = sub_once(
        t["user"],
        "the stairs — all ninety-six steps of them — remain open.",
        "the stairs — ninety-six steps — remain open.",
        "T16-B-06 statement",
    )
    t["thinking"] = sub_once(
        t["thinking"],
        '96 steps ↔ ninety-six steps — numeral to words, same count; "all" just gathers the 96 the notice prints   ✓ free',
        "96 steps ↔ ninety-six steps — numeral to words, same count   ✓ free",
        "T16-B-06 thinking",
    )
    if "all ninety-six" in t["user"] or '"all"' in t["thinking"]:
        raise SystemExit("T16-B-06: third edit remains")

    b7 = next(r for r in rows if r["id"] == "T16-B-07")
    t = b7["turns"][0]
    t["user"] = sub_once(
        t["user"],
        "The claim swaps a single word of behavior. Decide whether that swap is free, then answer.",
        'The claim swaps warn for throw, and it also adds "now" and "refuses to load". Decide whether those changes are free, then answer.',
        "T16-B-07 prompt",
    )
    t["thinking"] = sub_once(
        t["thinking"],
        "the claim: throw, and add refusal",
        'the claim: swaps warn for throw, and adds "now" and "refuses to load"',
        "T16-B-07 thinking claim",
    )
    t["thinking"] = sub_once(
        t["thinking"],
        "One English word, whole behavioral fact.",
        'Warn swapped for throw, plus "now" and "refuses to load": three changes, and the fact inverts.',
        "T16-B-07 thinking count",
    )
    t["reply"] = sub_once(
        t["reply"],
        "Warn-versus-throw is the whole difference between an advisory and a refusal — the bracket even underlines it, and the claim says both throw and refuse. One word changed; the fact inverted.",
        'Warn versus throw is the difference between an advisory and a refusal — the bracket even underlines it, and the claim says both throw and refuse. It is not a single-word swap: "now" and "refuses to load" are added too, and the fact inverted.',
        "T16-B-07 reply",
    )
    stmt = t["user"].split("Statement:", 1)[1].split("The claim swaps", 1)[0]
    for bit in ("now", "refuses to load", "throws"):
        if bit not in stmt and bit not in t["user"]:
            raise SystemExit(f"T16-B-07: claim/prompt missing {bit!r}")

    b8 = next(r for r in rows if r["id"] == "T16-B-08")
    digits = [c for c in "51.2" if c.isdigit()]
    rotated = digits[-1:] + digits[:-1]
    assert "".join(rotated) == "251"  # 25.1, the decimal point kept in place
    t = b8["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        "digits transposed: 51.2 ≠ 25.1",
        "a rotation of the digits: 51.2 ≠ 25.1",
        "T16-B-08 thinking",
    )
    t["reply"] = sub_once(
        t["reply"],
        "a transposed number is a different reading",
        "a rotation of the digits is a different reading",
        "T16-B-08 reply",
    )
    b8["answer"] = sub_once(
        b8["answer"],
        "the fourth transposes digits",
        "the fourth rotates the digits",
        "T16-B-08 answer",
    )
    blob = t["thinking"] + t["reply"] + b8["answer"]
    if "transpos" in blob:
        raise SystemExit("T16-B-08: transposition wording remains")

    c2 = next(r for r in rows if r["id"] == "T16-C-02")
    assert "791"[0] == "719"[0] and "791"[1] == "719"[2] and "791"[2] == "719"[1]
    t = c2["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        'The claim says 719 — the tens digit moved. Seventy-two tests exist in one and not the other. "-ish" is exactly how a moved digit gets waved through',
        'The claim says 719 — that swaps the tens and ones digits of 791. Seventy-two tests separate the two figures. "-ish" is exactly how a swapped pair of digits gets waved through',
        "T16-C-02 thinking",
    )
    t["reply"] = sub_once(
        t["reply"],
        "The line prints 791 and the claim reads 719 — same three digits, different number",
        "The line prints 791 and the claim reads 719 — that swaps the tens and ones digits",
        "T16-C-02 reply",
    )

    c4 = next(r for r in rows if r["id"] == "T16-C-04")
    c4["turns"][0]["reply"] = sub_once(
        c4["turns"][0]["reply"],
        'then "then `git commit && git push""',
        "then `git commit && git push`",
        "T16-C-04 quotes",
    )

    c6 = next(r for r in rows if r["id"] == "T16-C-06")
    ratio = 60_000  # one minute in milliseconds, against ~1 ms
    assert 4.5 < math.log10(ratio) < 5  # about 4.8, so "four" is wrong and "nearly five" is right
    c6["turns"][0]["thinking"] = sub_once(
        c6["turns"][0]["thinking"],
        "differ by four orders of magnitude",
        "differ by nearly five orders of magnitude",
        "T16-C-06",
    )

    c8 = next(r for r in rows if r["id"] == "T16-C-08")
    t = c8["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        ', mechanism stated.',
        ".",
        "T16-C-08 thinking",
    )
    t["reply"] = sub_once(
        t["reply"],
        "The claim is the bullet's sentence with a different tilde — same count, same size, same mechanism.",
        "The claim keeps the bullet's count and size, with a different tilde.",
        "T16-C-08 reply",
    )
    if "mechanism" in t["thinking"] or "mechanism" in t["reply"]:
        raise SystemExit("T16-C-08: mechanism remains")

    d1 = next(r for r in rows if r["id"] == "T16-D-01")
    notation = "`2d6ro=1`"
    gloss = "Reroll 1s once"
    require_in(d1["material"], notation, "T16-D-01 notation")
    require_in(d1["material"], gloss, "T16-D-01 gloss")
    if d1["deciding"] != [gloss]:
        raise SystemExit(f"T16-D-01: unexpected deciding {d1['deciding']!r}")
    d1["deciding"] = [notation, gloss]
    d1["turns"][0]["reply"] = sub_once(
        d1["turns"][0]["reply"],
        "DECIDING: " + gloss,
        "DECIDING: " + " | ".join(d1["deciding"]),
        "T16-D-01 deciding",
    )

    d4 = next(r for r in rows if r["id"] == "T16-D-04")
    t = d4["turns"][0]
    stmt = t["user"].split("Statement:", 1)[1]
    if "10:00" in stmt:
        raise SystemExit("T16-D-04: claim gained a start time")
    t["thinking"] = sub_once(
        t["thinking"],
        "Saturday matches and the 10:00 start matches; the end is the tell:",
        "Saturday matches; the end is the tell:",
        "T16-D-04 thinking",
    )
    t["reply"] = sub_once(
        t["reply"],
        "Saturday and 10:00 match; the end time doesn't",
        "Saturday matches; the end time doesn't",
        "T16-D-04 reply",
    )
    prose = t["thinking"] + "\n" + t["reply"].split("VERDICT:", 1)[0]
    if "10:00" in prose:
        raise SystemExit("T16-D-04: start-time match remains in the prose")

    d5 = next(r for r in rows if r["id"] == "T16-D-05")
    t = d5["turns"][0]
    t["user"] = sub_once(
        t["user"],
        "Statement: When roll falls back to Monte Carlo for an expression, the sampled result comes labelled as estimated (with the sample count), and the --json output records whether the method was exact or monte-carlo.",
        "Statement: When the analyzer falls back to Monte Carlo for a large or complex expression, `--analyze` and `--at-least` label the result as estimated (with the sample count), and the --json output records whether the method was exact or monte-carlo.",
        "T16-D-05 statement",
    )
    clause = (
        "When the analyzer falls back to Monte Carlo for a large or complex expression, "
        "`--analyze` and `--at-least` label the result as estimated (with the sample count) "
        "instead of presenting sampled numbers as exact."
    )
    method = (
        "The `--json` output carries a `method` field (`\"exact\"` or `\"monte-carlo\"`, "
        "with `samples` when sampled) so machine consumers can tell them apart too."
    )
    require_in(d5["material"], clause, "T16-D-05 clause")
    require_in(d5["material"], method, "T16-D-05 method")
    d5["deciding"] = [clause, method]
    t["thinking"] = (
        "Two leaves, and the claim names both:\n"
        "├─ `--analyze` and `--at-least` label the result as estimated (with the sample count), "
        "for a large or complex expression. The paragraph's first sentence says that, and the claim names those two flags.\n"
        "└─ the --json output records exact or monte-carlo. The paragraph's method-field sentence says that.\n"
        "If the claim names `--analyze` and `--at-least`, then I quote that whole clause, not a fragment of the label."
    )
    t["reply"] = (
        "The paragraph's first sentence names `--analyze` and `--at-least` on the estimated label, "
        "with the sample count. The method-field sentence names exact or monte-carlo on the --json output.\n"
        "VERDICT: supported\n"
        "DECIDING: " + " | ".join(d5["deciding"]) + "\n"
        "ESCALATE: no"
    )
    joined = t["thinking"] + "\n" + t["reply"]
    if "scope included" in joined or "scope stayed" in joined or "keeps the scope" in joined:
        raise SystemExit("T16-D-05: scope wording remains")
    claim = t["user"].split("Statement:", 1)[1]
    for flag in ("--analyze", "--at-least", "large or complex"):
        if flag not in claim:
            raise SystemExit(f"T16-D-05: claim missing {flag}")

    d8 = next(r for r in rows if r["id"] == "T16-D-08")
    claim = d8["turns"][0]["user"].split("Statement:", 1)[1]
    numerals = [tok for tok in ("120", "1.50") if tok in claim]
    assert numerals == ["120", "1.50"]
    d8["turns"][0]["thinking"] = sub_once(
        d8["turns"][0]["thinking"],
        "if the only numeral in the claim checks out in its converted unit, then the two-hour/120-minute swap is arithmetic, not editing",
        "if both numerals in the claim check out — 120 against two hours, and £1.50 left as £1.50 — then the two-hour/120-minute swap is arithmetic, not editing",
        "T16-D-08",
    )
    if "only numeral" in d8["turns"][0]["thinking"]:
        raise SystemExit("T16-D-08: only-numeral remains")

    e6 = next(r for r in rows if r["id"] == "T16-E-06")
    e6["turns"][0]["reply"] = sub_once(
        e6["turns"][0]["reply"],
        '"the bridge, built in 1902, needs repair" minus its appositive keeps the fact.',
        '"the old stone bridge needs repair" minus "old stone" keeps the fact.',
        "T16-E-06",
    )
    if "1902" in e6["turns"][0]["reply"]:
        raise SystemExit("T16-E-06: date remains")


def main() -> None:
    targets = {
        "batch2-t01.jsonl": ["T01-B-01", "T01-C-07"],
        "batch2-t14.jsonl": ["T14-C-04", "T14-C-08", "T14-D-07"],
        "batch2-t16.jsonl": [
            "T16-A-08", "T16-B-06", "T16-B-07", "T16-B-08", "T16-C-02", "T16-C-04",
            "T16-C-06", "T16-C-08", "T16-D-01", "T16-D-04", "T16-D-05", "T16-D-08", "T16-E-06",
        ],
    }
    fixers = {
        "batch2-t01.jsonl": fix_t01,
        "batch2-t14.jsonl": fix_t14,
        "batch2-t16.jsonl": fix_t16,
    }
    changed = []
    for name, ids in targets.items():
        path = HERE / name
        raw_lines, rows = load(path)
        by_id = {r["id"]: r for r in rows}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"{name}: missing {missing}")
        before = {i: dump_line(by_id[i]) for i in ids}
        fixers[name](rows)
        # Untouched lines stay byte-identical. Touched lines are re-serialized.
        out_lines = []
        id_to_line = {}
        for line in raw_lines:
            if not line.strip():
                out_lines.append(line)
                continue
            obj = json.loads(line)
            fresh = next(r for r in rows if r["id"] == obj["id"])
            if obj["id"] in ids:
                new_line = dump_line(fresh)
                if new_line == before[obj["id"]]:
                    raise SystemExit(f"{obj['id']}: fix made no change")
                blob = fresh["turns"][0]["thinking"] + "\n" + fresh["turns"][0]["reply"]
                quote_balance(blob, obj["id"])
                out_lines.append(new_line)
                changed.append(obj["id"])
            else:
                # Re-dump must match, or we would rewrite items R&D is not re-reading.
                if dump_line(fresh) != line:
                    raise SystemExit(f"{obj['id']}: untouched item would change on dump")
                out_lines.append(line)
            id_to_line[obj["id"]] = out_lines[-1]
        text = "\n".join(out_lines) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")
        data.validate([json.loads(x) for x in out_lines if x.strip()])
        print(f"{name}: validated, changed {sum(1 for i in ids)}")
    print("changed ids:", ", ".join(changed))
    if len(changed) != 18:
        raise SystemExit(f"expected 18 changed items (17 plus the T14-D-07 nit), got {len(changed)}")


if __name__ == "__main__":
    main()
