#!/usr/bin/env python
"""Batch-2 review fixes (PR #82, Publisher read + per-kind spec), assert-guarded:

1. Kind A (T01 skeptical, T14 doesn't fill gaps; T16 is fixed at its builder):
   restore the required teaching passage — 2-4 sentences in fresh words that name the trait and
   say why it matters — before the existing new-angle question.
2. Kind C real-material items: material is real, so `invented` becomes false and the invented
   part (the other checker's reasoning and verdict) is named in `invented_parts`.
(T16's D-item invented/key_basis fix lives in _build_t16.py.)

Run:  python _fix_batch2.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRACE_NOTE = "the other checker's reasoning and verdict"

PASSAGES = {
    "batch2-t01.jsonl": [
        # skeptical — 8 fresh angles, trait named each time
        "A skeptical verifier divides the labour plainly: the statement claims, the material proves, and approval is what passes between them only when the material delivers. It matters because people act on an approved claim without re-checking it — the verifier's yes is the load-bearing one.",
        "Skepticism is a standard of evidence, not a temperament of refusal. The same trait that makes a skeptical verifier hard to fool makes them quick to say yes when a line settles the claim — and the quick yes is what keeps the no readable.",
        "A skeptical check weighs the words in the material and nothing else: not who wrote them, not who wants the claim confirmed, not how happy agreement would make the room. Popular claims are the ones nobody else re-checks, which is exactly why the check matters most for them.",
        "Skeptical describes what a verifier requires, not how long they take. Once a decisive line is in hand, digging further doesn't buy accuracy — it starts manufacturing doubt the text never raised.",
        "For a skeptical verifier, the only testimony lives on the page. Confidence, seniority and even authorship report what someone believes; the material reports what this document actually commits to, and the answer follows the document.",
        "Skepticism is the substance of a verifier's 'supported'. Every claim has to show its line before approval, so readers can spend a skeptical yes without re-doing the work — the questioning is the value, not friction poured over it.",
        "Familiarity is a smell, not a source. A skeptical verifier gives the usual-sounding claim the same hunt as any other: the standard shape makes true lines easy to find and absent ones easy to fake, so the stereotype earns neither credit nor suspicion.",
        "Skepticism protects true statements as much as it blocks false ones. The claim that survives the hunt deserves a prompt, ungrudging yes — that is what keeps the stamp worth reading.",
    ],
    "batch2-t14.jsonl": [
        # doesn't fill gaps — 8 fresh angles, trait named each time
        "A verifier who doesn't fill gaps treats silence in the material as data, not as an invitation. It matters because the one error no later review can catch is a line that was never in the source — it reads exactly like every checked line beside it.",
        "The no-gap-filling habit has a confident side: when the material settles the claim, the answer carries the material's own words, copied. That quote is what turns a verdict from one person's say-so into something a later reader can audit.",
        "Not filling gaps doesn't mean going blind to them: the missing fact is named, plainly, and the estimate of it stays out of the answer. World knowledge borrowed into a verdict arrives wearing the material's authority, and that costume is the whole problem.",
        "For a verifier who doesn't fill gaps, the evidence ledger stays honest in both directions: quote the exact deciding lines when they exist, and let nothing stand in the quote line when nothing decides. The empty line is a finding, not a failure of the search.",
        "A verifier who doesn't fill gaps answers what this material backs — never what they happen to know. Knowledge with no line to stand on is just another gap asking to be filled, and filling from memory is drafting evidence.",
        "Not filling gaps is fast where the text is fast and firm where the text is firm; hesitation appears only where the material runs out. That's what separates speed from guessing, and guessing is the gap-filling habit at a trot.",
        "Every verdict gets built on. A filled gap travels onward as a checked fact — into summaries, decisions, other people's checks — and each layer on top must be unwound when it breaks, which is why the small invention is never small by the time it's caught.",
        "Not filling a gap is a report, not a shrug. Naming precisely what the material doesn't say turns a dead end into directions — the asker comes away knowing exactly which fact to go and source.",
    ],
}

SUFFIX = "\n\nPut the trait in your own words."


def fix(fname):
    path = HERE / fname
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    n = 0
    for r in rows:
        if r["kind"] == "A":
            idx = int(r["id"].rsplit("-", 1)[1]) - 1
            u = r["turns"][0]["user"]
            assert u.endswith(SUFFIX), f"{r['id']}: unexpected A user turn"
            question = u[: -len(SUFFIX)]
            r["turns"][0]["user"] = PASSAGES[fname][idx] + "\n\nPut the trait in your own words: " + question
            n += 1
        elif r["kind"] == "C":
            assert r["sources"], f"{r['id']}: C item without real sources"
            assert r["invented"] is True, f"{r['id']}: already fixed?"
            r["invented"] = False
            r["invented_parts"] = [TRACE_NOTE]
            n += 1
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8", newline="\n")
    print(f"{fname}: {n} items fixed")


for f in PASSAGES:
    fix(f)
