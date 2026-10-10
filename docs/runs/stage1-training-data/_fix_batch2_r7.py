"""Last batch-2 quote round, on the nine items named after the re-check at c8924b0.

Whole table rows, version headings, and one pipe-free span for T01-B-01.
Nothing else is rewritten. Untouched lines stay byte-identical.
"""
import json
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


def load(path: Path):
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


def row_containing(material: str, needle: str, where: str) -> str:
    hits = [ln for ln in material.splitlines() if ln.strip().startswith("|") and needle in ln]
    if len(hits) != 1:
        raise SystemExit(f"{where}: expected 1 table row containing {needle!r}, found {len(hits)}")
    return hits[0]


def heading_line(material: str, prefix: str, where: str) -> str:
    hits = [ln for ln in material.splitlines() if ln.strip() == prefix or ln.startswith(prefix)]
    exact = [ln for ln in material.splitlines() if ln.strip() == prefix]
    if len(exact) != 1:
        raise SystemExit(f"{where}: expected heading {prefix!r}, found {exact or hits}")
    return exact[0]


def set_deciding_line(reply: str, lines: list[str], where: str) -> str:
    new = "DECIDING: " + " | ".join(lines)
    parts = reply.splitlines()
    found = False
    out = []
    for ln in parts:
        if ln.startswith("DECIDING:"):
            if found:
                raise SystemExit(f"{where}: two DECIDING lines")
            out.append(new)
            found = True
        else:
            out.append(ln)
    if not found:
        raise SystemExit(f"{where}: no DECIDING line")
    return "\n".join(out)


def pipe_free(span: str, where: str) -> None:
    if " | " in span:
        raise SystemExit(f"{where}: span still contains the DECIDING separator")


TARGETS = {
    "batch2-t01.jsonl": ["T01-B-01", "T01-C-07", "T01-D-05"],
    "batch2-t14.jsonl": ["T14-C-01", "T14-D-02", "T14-D-06"],
    "batch2-t16.jsonl": ["T16-C-01", "T16-D-01", "T16-D-02"],
}


def fix_t01(rows):
    by = {r["id"]: r for r in rows}

    b = by["T01-B-01"]
    mat = b["material"]
    start = mat.find("The CLI labels Monte-Carlo results")
    end = mat.find("not exact`).")
    if start < 0 or end < 0 or end < start:
        raise SystemExit("T01-B-01: CLI label span not found")
    span = ws(mat[start:end + len("not exact`).")])
    pipe_free(span, "T01-B-01")
    if ws(span) not in ws(mat):
        raise SystemExit("T01-B-01: span is not verbatim")
    t = b["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        "├─ the bullet introduces a method field with two values, \"exact\" or \"monte-carlo\", and a CLI label for Monte-Carlo results",
        "├─ the deciding span is the CLI sentence, kept whole: Monte-Carlo results are labelled \"~ estimated via Monte Carlo (N samples) — not exact\"",
        "T01-B-01 thinking",
    )
    t["reply"] = sub_once(
        t["reply"],
        "The bullet itself falsifies the sweep: results can be \"exact\" or \"monte-carlo\", and Monte-Carlo output is explicitly labelled not exact. Every probability exact is not what this release did.",
        "The bullet itself falsifies the sweep: the CLI labels Monte-Carlo results as not exact. Every probability exact is not what this release did.",
        "T01-B-01 reply",
    )
    t["reply"] = set_deciding_line(t["reply"], [span], "T01-B-01")
    if "deciding" in b:
        raise SystemExit("T01-B-01: kind B grew a deciding key; leave the schema alone")

    c = by["T01-C-07"]
    row = row_containing(c["material"], "| `1` |", "T01-C-07")
    c["deciding"] = [
        "Roll follows a deliberate two-code contract — a stability promise scripts can rely on:",
        row,
    ]
    t = c["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        "The table in front of them is even titled a two-code contract: 0 success, 1 any error — and the row for 1 names missing loot file in its list.",
        "The prose above the table states a deliberate two-code contract. The quote is the whole error row, first cell to last: code 1, and the meaning cell names a missing loot file among any error.",
        "T01-C-07 thinking",
    )
    t["reply"] = sub_once(
        t["reply"],
        "The trace loses the thread. They rubber-stamped the familiar convention: exit 2 for a missing file is how many CLIs behave — and this section promises exactly two codes, with missing loot file filed under code 1.",
        "They rubber-stamped a convention this table refuses. Exit 2 for a missing file is how many CLIs behave — and this section promises exactly two codes, with the whole error row filing a missing loot file under code 1.",
        "T01-C-07 reply",
    )
    t["reply"] = set_deciding_line(t["reply"], c["deciding"], "T01-C-07")

    d = by["T01-D-05"]
    tier = row_containing(d["material"], "ignored when `model` is set", "T01-D-05 tier")
    model = row_containing(d["material"], "Explicit Ollama model", "T01-D-05 model")
    d["deciding"] = [tier, model]
    t = d["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        "├─ the tier row: ignored when model is set — tier LOSES when both are present\n└─ the model row: overrides tier — stated twice, in two rows, the same way down",
        "├─ the tier row, quoted whole, first cell to last: the description says tier is ignored when model is set\n└─ the model row, quoted whole, first cell to last: the description says model overrides tier",
        "T01-D-05 thinking",
    )
    t["reply"] = set_deciding_line(t["reply"], d["deciding"], "T01-D-05")


def fix_t14(rows):
    by = {r["id"]: r for r in rows}

    c = by["T14-C-01"]
    # The cited roll changelog heading is "## 2.1.1", already in the material.
    # "## [2.1.1]" is not in that file, so it is not inserted.
    head = heading_line(c["material"], "## 2.1.1", "T14-C-01")
    c["deciding"] = [head, "`--target-for` is unchanged."]
    t = c["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        "What does the entry say about the old flag's fate?",
        "The claim's \"in 2.1.1\" is the entry's own heading in the material, not only the framing label. That heading is quoted with the unchanged-flag sentence. What does the entry say about the old flag's fate?",
        "T14-C-01 thinking",
    )
    t["reply"] = sub_once(
        t["reply"],
        "and the entry itself contradicts it — the old flag is still there, unchanged, alongside the new solver.",
        "and the entry itself contradicts it — under the ## 2.1.1 heading, the old flag is still there, unchanged, alongside the new solver.",
        "T14-C-01 reply",
    )
    t["reply"] = set_deciding_line(t["reply"], c["deciding"], "T14-C-01")

    d2 = by["T14-D-02"]
    head = heading_line(d2["material"], "## [3.4.0] - 2026-08-04", "T14-D-02")
    d2["deciding"] = [head, "Every fix carries a test proven to fail against the unfixed code."]
    t = d2["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        "The settling line sits in the overview paragraph, not down in the bullets — easy to skim past on the way to the feature list.",
        "The settling line sits in the overview paragraph, not down in the bullets — easy to skim past on the way to the feature list. The claim names 3.4.0, so the version heading is its own deciding line beside that sentence.",
        "T14-D-02 thinking",
    )
    t["reply"] = set_deciding_line(t["reply"], d2["deciding"], "T14-D-02")

    d6 = by["T14-D-06"]
    head = heading_line(d6["material"], "## [1.0.0] - 2026-05-15", "T14-D-06")
    d6["deciding"] = [
        head,
        "- **5 new engine helpers** (foundation for v1.1 control-plane integration):",
        "- `validateEnvelopeFile` — parse + schema-validate an envelope file in one call",
    ]
    t = d6["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        "The claim's \"one call\" matches \"in one call\" printed beside validateEnvelopeFile; both leaves quoted, nothing paraphrased into existence.",
        "The claim's \"one call\" matches \"in one call\" printed beside validateEnvelopeFile; both leaves quoted, nothing paraphrased into existence. The claim names 1.0.0, so that version heading is its own deciding line beside the two helper lines.",
        "T14-D-06 thinking",
    )
    t["reply"] = set_deciding_line(t["reply"], d6["deciding"], "T14-D-06")


def fix_t16(rows):
    by = {r["id"]: r for r in rows}

    c = by["T16-C-01"]
    row = row_containing(c["material"], "`--at-least N`", "T16-C-01")
    c["deciding"] = [row]
    t = c["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        "But ≥ is the standard notation for at-least; the table row and the claim are one relation wearing two spellings.",
        "But ≥ is the standard notation for at-least; the quote is the whole `--at-least N` row, first cell to last, and that row and the claim are one relation wearing two spellings.",
        "T16-C-01 thinking",
    )
    t["reply"] = sub_once(
        t["reply"],
        "and the row's gloss is the claim restated; doubt about typography isn't doubt about evidence.",
        "and the whole row, flag cell and answer cell, is the claim restated; doubt about typography isn't doubt about evidence.",
        "T16-C-01 reply",
    )
    t["reply"] = set_deciding_line(t["reply"], c["deciding"], "T16-C-01")

    d1 = by["T16-D-01"]
    row = row_containing(d1["material"], "`2d6ro=1`", "T16-D-01")
    d1["deciding"] = [row]
    t = d1["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        "My habit here: if the gloss covers every part of the claim, then the claim passes with the row as its quote — a one-line row isn't under-evidenced for being short.",
        "My habit here: if the gloss covers every part of the claim, then the claim passes with the whole row as its quote, first cell to last — notation and gloss together.",
        "T16-D-01 thinking",
    )
    t["reply"] = sub_once(
        t["reply"],
        "The table glosses `2d6ro=1` as rerolling 1s once — both parts of the claim, with the `ro` row distinguished from unlimited `r` directly beside it.",
        "The quote is the whole `2d6ro=1` row: notation cell and gloss cell. It rerolls 1s once — both parts of the claim, with the `ro` row distinguished from unlimited `r` directly beside it.",
        "T16-D-01 reply",
    )
    t["reply"] = set_deciding_line(t["reply"], d1["deciding"], "T16-D-01")

    d2 = by["T16-D-02"]
    head = heading_line(d2["material"], "## [3.3.0] - 2026-07-06", "T16-D-02")
    d2["deciding"] = [head, "Coverage: statements **63.53%** / branches **75.77%** / functions **69.04%**."]
    t = d2["turns"][0]
    t["thinking"] = sub_once(
        t["thinking"],
        "The rule that earns its keep here: if a claim quotes the right numbers with the wrong labels, then the numbers' being right is the trap — exact figures borrowed from the right neighbourhood read as checked until you check the ownership.",
        "The claim says at 3.3.0, so the version heading is its own deciding line beside the coverage sentence. If a claim quotes the right numbers with the wrong labels, then the numbers' being right is the trap — exact figures borrowed from the right neighbourhood read as checked until you check the ownership.",
        "T16-D-02 thinking",
    )
    t["reply"] = set_deciding_line(t["reply"], d2["deciding"], "T16-D-02")


def main():
    fixers = {
        "batch2-t01.jsonl": fix_t01,
        "batch2-t14.jsonl": fix_t14,
        "batch2-t16.jsonl": fix_t16,
    }
    changed = []
    for name, ids in TARGETS.items():
        path = HERE / name
        raw_lines, rows = load(path)
        by_id = {r["id"]: r for r in rows}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise SystemExit(f"{name}: missing {missing}")
        before = {i: dump_line(by_id[i]) for i in ids}
        fixers[name](rows)
        out_lines = []
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
                # The reply's DECIDING text matches the key, where a key exists.
                if fresh.get("deciding"):
                    reply_dec = next(ln for ln in fresh["turns"][0]["reply"].splitlines() if ln.startswith("DECIDING:"))
                    expect = "DECIDING: " + " | ".join(fresh["deciding"])
                    if reply_dec != expect:
                        raise SystemExit(f"{obj['id']}: reply DECIDING drifted from the key")
                    for d in fresh["deciding"]:
                        if ws(d) not in ws(fresh["material"]):
                            raise SystemExit(f"{obj['id']}: deciding not verbatim: {d!r}")
                else:
                    reply_dec = next(ln for ln in fresh["turns"][0]["reply"].splitlines() if ln.startswith("DECIDING:"))
                    span = reply_dec[len("DECIDING: "):]
                    pipe_free(span, obj["id"])
                    if ws(span) not in ws(fresh["material"]):
                        raise SystemExit(f"{obj['id']}: deciding span not verbatim")
                out_lines.append(new_line)
                changed.append(obj["id"])
            else:
                if dump_line(fresh) != line:
                    raise SystemExit(f"{obj['id']}: untouched item would change on dump")
                out_lines.append(line)
        text = "\n".join(out_lines) + "\n"
        path.write_text(text, encoding="utf-8", newline="\n")
        loaded = [json.loads(x) for x in out_lines if x.strip()]
        data.validate(loaded)  # pre-amendment skip still on: batch 2's normal gate
        try:
            data.validate(loaded, pre_amendment=frozenset())
        except data.DataError as err:
            msg = str(err)
            if "whole table row" in msg or "version heading" in msg:
                raise SystemExit(f"{name}: quoting check failed: {msg}")
            print(f"{name}: quoting passed; remaining refusal is the floor batch 2 is not required to meet:")
            print(" ", msg.split("\n", 1)[0][:300])
        else:
            print(f"{name}: quoting passed, and the empty pre-amendment set did not refuse")
        print(f"{name}: validated, changed {sum(1 for i in ids)}")
    if changed != [i for ids in TARGETS.values() for i in ids]:
        raise SystemExit(f"changed set {changed}")
    print("changed ids:", ", ".join(changed))


if __name__ == "__main__":
    main()
