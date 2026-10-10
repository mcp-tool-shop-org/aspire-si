#!/usr/bin/env python
"""Batch-2 round-2 fixes from R&D's full key check on PR #82 (T01 + T14 parts; T16 lives in
its builder). Assert-guarded substring replacements; aborts loudly if any anchor is missing.

Covered here:
  T01: B-02 rebuilt as a closed puzzle (marble-jar probability reading taught something false);
       B-08 legend step removed (the walker's note, shown in the prompt, carries it);
       C-04 DECIDING gains the v1.0.x header line; C-06 statement matches the material's verb;
       C-03/C-04/C-07 reply no longer says the checker "quoted" what they described;
       D-02 time words dropped (the version heading isn't in the cited span); D-07 uses the
       material's term (meeting room); D self-talk intros varied (if-then habit kept);
       stamped branch labels varied; C openers varied.
  T14: B-02 survivors stated exactly; C-02 reply drops the translate_all assertion (kept to the
       shown material); C-04 DECIDING copies the bullet with its ** markers; C-07 drops "now";
       D-05 thinking no longer asserts all-machine validation; E-06 rebuilt without the banned
       universal-vs-scope structure (plain silence); E-08 variant says "120 mm diameter each".
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (file, id, field, old, new). field: "user" | "thinking" | "reply" | "answer" | "deciding-full" | ...
R = []


def rep(f, i, field, old, new):
    R.append((f, i, field, old, new))


def full(f, i, field, new):
    R.append((f, i, field, None, new))


T01 = "batch2-t01.jsonl"
T14 = "batch2-t14.jsonl"

# ------------------------------------------------ T01-B-02: closed puzzle rebuild
full(T01, "T01-B-02", "user",
     "Fuse-panel card: fuse F1 feeds the pump; fuse F3 does not feed the light; each of the three "
     "fuses (F1, F2, F3) feeds exactly one circuit — the pump, the light or the fan — and each "
     "circuit has exactly one fuse.\nA technician's note claims: \"F2 feeds the fan.\"\nAs the one "
     "who checks claims: back the note or not, from the card alone? The reasoning is the exercise "
     "— derive the assignments before you say anything.")
full(T01, "T01-B-02", "thinking",
     "Elimination, all assignments forced by the card:\n"
     "├─ F1 → pump (printed)\n"
     "├─ F3 → not the light (printed), and not the pump (F1 feeds it; one fuse per circuit) → F3 → fan\n"
     "└─ F2 → the only circuit left: the light\n"
     "The note claims F2 → fan, but the card puts the fan with F3 and leaves F2 the light. Before "
     "answering, check the outs a skeptic keeps open: vague wording, a misprint, a hidden fourth "
     "circuit — none exist; the card is closed and exact. So the answer is no, and it comes from "
     "the derivation, not from doubting the note's honesty. A closed card that contradicts a claim "
     "gives the cleanest no there is.")
full(T01, "T01-B-02", "reply",
     "No. The card forces every assignment: F1 feeds the pump (printed); F3 can't feed the light "
     "(printed) or the pump (already taken), so it feeds the fan; F2 gets what remains — the light. "
     "The note hands the fan to F2, which would mean un-reading the card. A plausible-sounding note, "
     "contradicted by a closed panel, stays unbacked.")
full(T01, "T01-B-02", "answer",
     "no — the card forces F1→pump, F3→fan, F2→light, so \"F2 feeds the fan\" misassigns a settled circuit")

# ------------------------------------------------ T01-B-08: legend step out, walker note carries it
rep(T01, "T01-B-08", "thinking",
    "├─ north path: blank square — the legend has a blank for unverified and (per the walker's own "
    "note) there is no closed marker in this sketch's vocabulary at all",
    "├─ north path: blank square — the walker's own note pins what a blank can't mean: this sketch "
    "has no closed markers at all, so blank ≠ closed")

# ------------------------------------------------ T01 C openers (vary the stamp)
rep(T01, "T01-C-01", "reply", "They didn't get it right.", "That check misfires.")
rep(T01, "T01-C-02", "reply", "They didn't get it right.", "Their answer doesn't stand.")
rep(T01, "T01-C-04", "reply", "They didn't get it right.", "The refusal outruns the evidence.")
rep(T01, "T01-C-07", "reply", "They didn't get it right.", "The trace loses the thread.")
rep(T01, "T01-C-08", "reply", "They didn't get it right.", "That verdict fails a re-read.")
rep(T01, "T01-C-05", "reply", "They got it right.", "A correct call.")

# T01-C-03: opener + checker didn't quote, they described
rep(T01, "T01-C-03", "reply", "They got it right.", "The checker's verdict stands.")
rep(T01, "T01-C-03", "reply",
    "and they quoted that sentence rather than paraphrasing it",
    "and they took their answer from that sentence alone")
rep(T01, "T01-C-03", "thinking",
    "They located it, quoted it whole, stopped.",
    "They located it, took the sentence whole, stopped.")

# T01-C-04: DECIDING gains the v1.0.x header line
rep(T01, "T01-C-04", "reply",
    "DECIDING: All 8 CLI commands plus the engine helpers behind them.",
    "DECIDING: **v1.0.x transport surface (complete):** | All 8 CLI commands plus the engine helpers behind them.")
full(T01, "T01-C-04", "deciding",
     ["**v1.0.x transport surface (complete):**", "All 8 CLI commands plus the engine helpers behind them."])

# T01-C-06: statement matches the material's verb ("use the = form")
rep(T01, "T01-C-06", "reply", "They got it right.", "Properly read.")
rep(T01, "T01-C-06", "user",
    'Statement: "A negative seed must be passed with the equals form, like --seed=-3."',
    'Statement: "To pass a negative seed, you use the = form — for example, --seed=-3."')
rep(T01, "T01-C-06", "thinking",
    'didn\'t soften "must" out of "use the = form", didn\'t add alternatives the text doesn\'t list',
    "took the instruction exactly as printed — use the = form, with its example and its reason, nothing added")
rep(T01, "T01-C-06", "reply",
    "The paragraph mandates the equals form for negative seeds, with the worked example being the claim's own example.",
    "The paragraph names the equals form for negative seeds, reason and example included — the claim is that sentence's own content.")

# T01-D-02: drop the time words (3.4.0 heading isn't in the cited span)
rep(T01, "T01-D-02", "user",
    "Material (style-dataset-lab CHANGELOG.md, the Fixed section of the 3.4.0 entry):",
    "Material (style-dataset-lab CHANGELOG.md, a release entry's Fixed section):")
rep(T01, "T01-D-02", "user",
    "Statement: Until 3.4.0, every npm user who ran sdlab migrate hit a Cannot find module error, "
    "because migrate-records.js had been left out of the published tarball.",
    "Statement: Every npm user who ran sdlab migrate hit a Cannot find module error, because "
    "migrate-records.js had been left out of the published tarball.")
full(T01, "T01-D-02", "thinking",
     "Ten bullets, and the claim lives somewhere in the middle — no shortcut: read each.\n"
     "Decompose: (1) npm users running sdlab migrate got Cannot find module, (2) because "
     "migrate-records.js had been left out of the published tarball.\n"
     "├─ bullets 1–4: init, --project, doctor, leakage — none mention migrate\n"
     "├─ bullets 5–8: training-package, eval-run, pinning parity, freeze-gate — none\n"
     "└─ bullet 9: \"migrate-records.js was excluded from the npm tarball while still dispatched and "
     "documented — every npm user running sdlab migrate got Cannot find module\" ✓ (1) and (2) word for word\n"
     "The patience rule here: the settling bullet is ninth of ten, so impatience is the enemy — a "
     "skeptic with a long list reads the whole list; neither the early encourage-shaped bullets nor "
     "fatigue gets to cast the vote. Both parts of the claim land on one printed bullet.")
rep(T01, "T01-D-02", "reply",
    "The ninth bullet states the exclusion and the exact error npm users saw, and its place in the "
    "Fixed section dates the behavior as pre-3.4.0. Claim earned — after reading all ten bullets, not "
    "after the promising first one.",
    "The ninth bullet states the exclusion and the exact error npm users saw — both parts of the "
    "claim in one printed line. Earned, after all ten bullets were read, not after the promising first one.")

# T01-D-07: the material's term is "meeting room"
rep(T01, "T01-D-07", "user",
    "Statement: The 6-seat side room must be booked via the office calendar.",
    "Statement: The 6-seat meeting room must be booked via the office calendar.")
rep(T01, "T01-D-07", "thinking",
    "├─ pin it down: a 6-seat room requires a calendar booking",
    "├─ the claim plainly: a 6-seat meeting room requires a calendar booking")

# ------------------------------------------------ T01 D traces: vary self-talk intros + branch labels
rep(T01, "T01-D-01", "thinking", "Self-talk: if a claim adds a limiter", "What I remind myself: if a claim adds a limiter")
rep(T01, "T01-D-03", "thinking", "├─ pin it down: tumble drying is allowed, at a low setting",
    "├─ the claim in plain words: tumble drying is allowed, at a low setting")
rep(T01, "T01-D-03", "thinking", "Self-talk: if the claim feels reasonable",
    "Here's the habit doing the work: if the claim feels reasonable")
rep(T01, "T01-D-04", "thinking", "├─ pin it down: admission price is zero",
    "├─ what must be shown: admission price is zero")
rep(T01, "T01-D-04", "thinking",
    "Self-talk: if the first clause settles it outright, then the remaining clauses earn only",
    "When the first clause settles it outright, then the remaining clauses earn only")
rep(T01, "T01-D-05", "thinking", "├─ pin it down: tier outranks model when both are set",
    "├─ the claim, unpacked: tier outranks model when both are set")
rep(T01, "T01-D-05", "thinking", "Self-talk: if two rows of a small table", "If two rows of a small table")
rep(T01, "T01-D-06", "thinking", "├─ pin it down: the vote split was 4 for, 1 against, and the item passed",
    "├─ the claim's contents: a 4–1 split, and the item passed")
rep(T01, "T01-D-06", "thinking", "Self-talk: if the record's own formula", "My working rule: if the record's own formula")
rep(T01, "T01-D-07", "thinking", "Self-talk: if the building's habit", "Where I stand: if the building's habit")
rep(T01, "T01-D-08", "thinking",
    "Self-talk: if a claim is a long enumeration, then I count its parts against the line rather than nodding at the shape",
    "My habit with long enumerations: if a claim lists parts, then I count them against the line rather than nodding at the shape")

# ------------------------------------------------ T14-B-02: survivors stated exactly
rep(T14, "T14-B-02", "thinking",
    "The open cell stays open — I place what both survivors agree on (nothing on shelf 1's identity) "
    "and report the pair rather than deal a colour the clues never dealt.",
    "The open cell stays open — what the two survivors agree on: green is never on shelf 1, and blue "
    "is never in the middle. I report both arrangements rather than deal shelf 1 a colour the clues "
    "never dealt.")
rep(T14, "T14-B-02", "reply",
    "Two arrangements fit: red–green–blue and blue–red–green. Green is never on shelf 1 and the "
    "middle shelf is whoever blue isn't — but which of red or blue starts the row, the clues don't say.",
    "Two arrangements fit: red–green–blue and blue–red–green. What the clues do pin down: green is "
    "never on shelf 1 and blue is never in the middle — but whether red or blue starts the row, the "
    "clues don't say.")

# ------------------------------------------------ T14 C openers + scoped fixes
rep(T14, "T14-C-01", "reply", "They didn't get it right.", "The check slips.")
rep(T14, "T14-C-05", "reply", "They didn't get it right.", "This one went wrong.")
rep(T14, "T14-C-06", "reply", "They didn't get it right.", "That answer doesn't hold.")
rep(T14, "T14-C-03", "reply", "They got it right.", "Their verdict holds.")
rep(T14, "T14-C-07", "reply", "They got it right.", "That call is sound.")
rep(T14, "T14-C-08", "reply", "They got it right.", "Properly checked.")

# T14-C-02: opener + keep the reply to the shown material
rep(T14, "T14-C-02", "reply", "They didn't get it right.", "They missed it.")
rep(T14, "T14-C-02", "reply",
    "They guessed — borrowed the 7 from a different tool's default and laid it over this entry, "
    "which says all 57. The 7 default belongs to translate_all; list_languages has its own count "
    "printed in its own sentence.",
    "They guessed — their own trace says the 7 was borrowed from elsewhere, and the line on the "
    "page says all 57. The entry shown is the whole evidence here, and it prints its own count in "
    "its own sentence.")

# T14-C-04: opener + DECIDING copies the bullet exactly (** kept), and no "they quoted"
rep(T14, "T14-C-04", "reply", "They got it right.", "Read correctly.")
rep(T14, "T14-C-04", "reply",
    "They quoted the heading and the bullet instead of skimming past the heading — nothing about the answer was inferred.",
    "They read the heading and the bullet together instead of skimming past the heading — nothing about the answer was inferred.")
rep(T14, "T14-C-04", "reply",
    "DECIDING: ### Fixed | - `sdlab project doctor` printed a failure and exited 0 on a missing project — and `npm run verify` and the CI smoke step both gate on that exit code.",
    "DECIDING: ### Fixed | - **`sdlab project doctor` printed a failure and exited 0** on a missing project — and `npm run verify` and the CI smoke step both gate on that exit code.")
full(T14, "T14-C-04", "deciding",
     ["### Fixed",
      "- **`sdlab project doctor` printed a failure and exited 0** on a missing project — and `npm run verify` and the CI smoke step both gate on that exit code."])

# T14-C-07: drop "now" from the claim (the heading isn't in the shown span); no "they quoted"
rep(T14, "T14-C-07", "user",
    'Statement: "The MCP server now reports its version from package.json rather than a hardcoded one."',
    'Statement: "The MCP server reports its version from package.json rather than a hardcoded one."')
rep(T14, "T14-C-07", "thinking",
    'Same fact, no paraphrase drift — "now" in the claim is fair because this is the Fixed section of the release.',
    "Same fact, no paraphrase drift.")
rep(T14, "T14-C-07", "reply",
    "and they quoted it whole instead of paraphrasing",
    "and they answered on the bullet's own words")

# ------------------------------------------------ T14 D traces: vary self-talk intros + branch labels
rep(T14, "T14-D-01", "thinking", "Self-talk: if a whole security pass tempts me to assume",
    "My brake here: if a whole security pass tempts me to assume")
rep(T14, "T14-D-02", "thinking",
    "Self-talk: when the claim is sweeping (every fix), I hold the quote to the same scope, and here",
    "When the claim is sweeping (every fix), I hold the quote to the same scope — and here")
rep(T14, "T14-D-03", "thinking", "├─ pin it down:", "├─ put the claim in words:")
rep(T14, "T14-D-03", "thinking", "├─ to the source:", "├─ read the notice:")
rep(T14, "T14-D-03", "thinking", "Self-talk: if the notice is silent on reopening",
    "My standing habit: if the notice is silent on reopening")
rep(T14, "T14-D-04", "thinking", "Self-talk: if every leaf lands on a printed line",
    "If every leaf lands on a printed line")

# T14-D-05: thinking must stop asserting all-machine validation
full(T14, "T14-D-05", "thinking",
     "├─ put the claim in words: a person reviews translations before they are delivered\n"
     "├─ read the paragraph: chunking at paragraph and sentence boundaries — that step the text "
     "calls automatic — then translation in sequence, reassembly, and a validation step listing "
     "four checks: empty output, echo detection, truncation, garbled text\n"
     "└─ who or what performs those checks, the paragraph never says; and no human review step "
     "appears in it, for or against\n"
     "When the text doesn't say who validates, then my general sense that such checks are usually "
     "programmatic stays off the page — I report the silence rather than writing either a person "
     "or a machine into it.")
rep(T14, "T14-D-05", "reply",
    "The paragraph names automatic chunking and four automated validation checks; no human review "
    "step appears in it. The claim adds a person the text never mentions.",
    "The passage describes automatic chunking and a validation step with four named checks, but it "
    "never says who or what performs them, and no human review step appears anywhere. The claim "
    "adds a person the text neither names nor rules out — the paragraph is simply silent on that point.")

rep(T14, "T14-D-06", "thinking", "Self-talk: if the entry is long, then the settling line may be anywhere in it",
    "The discipline that pays on long entries: if the settling line may be anywhere in the entry")
rep(T14, "T14-D-07", "thinking", "├─ pin it down:", "├─ the claim, named:")
rep(T14, "T14-D-07", "thinking", "├─ to the source:", "├─ what the timetable prints:")
rep(T14, "T14-D-07", "thinking", "Self-talk: if a line contradicts the claim",
    "If a line contradicts the claim")
rep(T14, "T14-D-08", "thinking", "├─ pin it down:", "├─ set out the claim:")
rep(T14, "T14-D-08", "thinking", "├─ to the source:", "├─ the policy's actual contents:")
rep(T14, "T14-D-08", "thinking", "Self-talk: if a policy covers one mode and the claim covers another",
    "When a policy covers one mode and the claim covers another")

# ------------------------------------------------ T14-E-06: no universal-vs-scope; plain silence
full(T14, "T14-E-06", "user",
     "Retrieval: a statement takes a claim the record backs and adds one word that evaluates it — "
     "the record says \"the inspection passed in May\"; the statement says the inspection passed "
     "\"easily\". The added word appears nowhere in the record. Verdict and deciding lines?")
full(T14, "T14-E-06", "reply",
     "cannot_tell, DECIDING NONE. The backed half can't carry the extra word: \"passed in May\" says "
     "nothing about \"easily\" — ease is a second claim that needs a line of its own, and the record "
     "doesn't have one. A qualifier isn't decoration on a true sentence; it's extra content the "
     "material has to show me before I can show anyone else.")
full(T14, "T14-E-06", "variant", {
    "user": "This time the log says \"the pump failed in March\" and the statement claims it "
            "\"suddenly\" failed. Verdict, deciding, and your reason in one sentence.",
    "key": {"type": "written", "key_points": [
        "cannot_tell", "NONE", "an unbacked \"suddenly\" needs its own line like any qualifier"]}})

# ------------------------------------------------ T14-E-08: variant — "120 mm diameter each"
full(T14, "T14-E-08", "variant", {
    "user": "Material:\nBox of 4 filters, 120 mm diameter each. Illustration not to scale.\n"
            "Statement: Each filter measures 120 mm across.",
    "key": {"type": "verdict", "verdict": "supported",
            "deciding": ["Box of 4 filters, 120 mm diameter each."], "escalate": "no"}})

# ---------------------------------------------------------------- apply


def main():
    per_file = {}
    for f, i, field, old, new in R:
        per_file.setdefault(f, []).append((i, field, old, new))
    for f, patches in per_file.items():
        path = HERE / f
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        by_id = {r["id"]: r for r in rows}
        n = 0
        for i, field, old, new in patches:
            r = by_id[i]
            if field in ("user", "thinking", "reply"):
                tgt = r["turns"][0][field]
                if old is None:
                    r["turns"][0][field] = new
                else:
                    assert tgt.count(old) == 1, f"{i} {field}: anchor not unique: {old[:60]!r} (count {tgt.count(old)})"
                    r["turns"][0][field] = tgt.replace(old, new)
            elif field in ("answer", "deciding", "variant"):
                if old is None:
                    r[field] = new
                else:
                    assert r[field] == old, f"{i} {field}: full-field mismatch"
                    r[field] = new
            else:
                raise ValueError(field)
            n += 1
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8", newline="\n")
        print(f"{f}: {n} patches applied")


main()
