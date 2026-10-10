#!/usr/bin/env python
"""Round-3 batch-2 fixes: conform to ASPIRE's landed drift checks (25ce64f + 27d1318).

- sources non-empty on every item: invented items carry {"invented": "<what>"}; C items append
  {"invented": "the other checker's reasoning and verdict"} beside the real citation (the
  invented_parts field is folded into sources and removed).
- kind C user-turn opening frames varied (check_frames: first 6 words may open at most 2 items
  of a trait).
- T14 A-02/A-07/A-08 teaching passages reworded to name the trait in the validator's terms.
(T16 is fixed in its builder.)
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

FRAMES = [
    "Material given to another checker ({label}):",
    "Another checker was handed this material ({label}):",
    "A second reviewer received this text ({label}):",
    "For a contrasting-case check, the material shown ({label}):",
]
ORDER = [0, 1, 2, 3, 1, 2, 3, 0]  # two per frame per trait max

A_PASSAGE_FIXES = {
    "T14-A-02": "A verifier who refuses to fill gaps has a confident side too: when the material "
                "settles the claim, the answer carries the material's own words, copied. That quote "
                "is what turns a verdict from one person's say-so into something a later reader can audit.",
    "T14-A-07": "The reason a verifier must not fill gaps is that every verdict gets built on. A "
                "filled gap travels onward as a checked fact — into summaries, decisions, other "
                "people's checks — and each layer on top must be unwound when it breaks, which is "
                "why the small invention is never small by the time it's caught.",
    "T14-A-08": "For a verifier who doesn't fill gaps, leaving one open is a report, not a shrug. "
                "Naming precisely what the material doesn't say turns a dead end into directions — "
                "the asker comes away knowing exactly which fact to go and source.",
}

INVENTED_NOTE = {
    "A": "teaching passage and question, written for this item",
    "B": "puzzle and scenario, written for this item",
    "E": "question and variant, written for this item",
}
TRACE_NOTE = "the other checker's reasoning and verdict"


def invented_desc(r):
    if r["kind"] in INVENTED_NOTE:
        return INVENTED_NOTE[r["kind"]]
    return f"the material (an everyday {r['content_kind']}), written for this item"


def fix(fname):
    path = HERE / fname
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    c_n = 0
    for r in rows:
        # ---- A passages that must name the trait in the validator's terms
        if r["id"] in A_PASSAGE_FIXES:
            u = r["turns"][0]["user"]
            head, tail = u.split("\n\n", 1)
            assert tail.startswith("Put the trait in your own words:")
            r["turns"][0]["user"] = A_PASSAGE_FIXES[r["id"]] + "\n\n" + tail
        # ---- C opener frames
        if r["kind"] == "C":
            line0, rest = r["turns"][0]["user"].split("\n", 1)
            assert line0.startswith("Material given to another checker (") and line0.endswith("):"), \
                f"{r['id']}: unexpected C opener: {line0[:70]!r}"
            label = line0[len("Material given to another checker ("):-2]
            r["turns"][0]["user"] = FRAMES[ORDER[c_n]].format(label=label) + "\n" + rest
            c_n += 1
        # ---- sources
        if r["kind"] == "C":
            r.pop("invented_parts", None)
            assert r["sources"] and not r["invented"], f"{r['id']}: C provenance wrong"
            r["sources"] = r["sources"] + [{"invented": TRACE_NOTE}]
        else:
            if r["invented"]:
                assert not r["sources"], f"{r['id']}: invented but has sources"
                r["sources"] = [{"invented": invented_desc(r)}]
            else:
                assert r["sources"], f"{r['id']}: real material without sources"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8", newline="\n")
    print(f"{fname}: normalized ({c_n} C items re-framed)")


for f in ("batch2-t01.jsonl", "batch2-t14.jsonl"):
    fix(f)
