#!/usr/bin/env python
"""Batch-2 round 5 (T01/T14; T16 changes live in its builder): landed-validator conformance.

- kind B thinking draws its pattern with tree glyphs (T01-B-01, T14-B-03, T14-B-05).
- kind E written-question variants re-share one original key point verbatim (the validator's
  variant-key overlap rule).
- T01-C-02's statement drops its time word ("now"), per R&D's strip-time-words rule.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# B thinking: draw the pattern -------------------------------------------------
THINKING_FULL = {
    "T01-B-01": (
        "The claim flatters the release — a clean sweep to exact arithmetic. Show me the sweep.\n"
        "Falsifier hunt: what would kill \"every probability is exact\"? Any surviving Monte-Carlo path.\n"
        "├─ the bullet introduces a method field with two values, \"exact\" or \"monte-carlo\", and a CLI "
        "label for Monte-Carlo results\n"
        "└─ you don't label a thing that no longer exists — and the label's wording rubs it in: \"not exact\"\n"
        "The boast shape tempted a skim; two phrases in the bullet contradicted it. A welcome "
        "improvement and a false claim about it sit one line apart."),
    "T14-B-03": (
        "Elimination over five ranks:\n"
        "├─ R > S > T pins the tail: S and T fill ranks 4 and 5 in that order\n"
        "├─ P > Q and Q > R: P and Q both beat R, so R ∉ ranks 1–2\n"
        "└─ rank 3 is R's — below P and Q, above S and T. Ranks 1 and 2 go to P then Q, per the first clue.\n"
        "Every adjacent pair — P–Q, Q–R, R–S, S–T — is pinned by its own clue, so no cell is left "
        "open. Nothing here needs a guess: the clues happened to form a complete chain, and the answer "
        "is stated flatly because every link of it was given."),
    "T14-B-05": (
        "Grid, then branches on the one open clue:\n"
        "├─ Deb = 3 (given). Remaining cabins 1, 2, 4 for Ada, Bea, Cid; Ada ∈ {2, 4}\n"
        "├─ branch A: Ada = 2 → Bea and Cid take 1 and 4; Bea > Cid forces Bea = 4, Cid = 1\n"
        "└─ branch B: Ada = 4 → Bea and Cid take 1 and 2; Bea > Cid forces Bea = 2, Cid = 1\n"
        "Both branches satisfy every clue. Across both: Cid = 1 and Deb = 3 hold in each, so those are "
        "settled. Ada and Bea swap cabins 2 and 4 between the branches and no clue breaks the tie — so "
        "I don't break it either. Writing \"Ada has cabin 4\" would add a fact the clues never contained."),
}

# E written variants: re-share one original key point (verbatim, then the variant's own twist) ----
VARIANT_KP = {
    # id: (index, new key point containing an original one)
    "T01-E-01": (2, "authority, familiarity and plausibility earn nothing — status and likelihood don't count"),
    "T01-E-02": (0, "a decisive line ends the check immediately — quote it and answer at once"),
    "T01-E-03": (0, "familiarity buys a quicker first look, never the verdict — it still needs its line"),
    "T01-E-04": (2, "cynicism answers before reading; skepticism after — the two orders differ"),
    "T01-E-05": (0, "convenience changes nothing about the standard: same lines, same hunt"),
    "T01-E-06": (2, "it turns luck into earning — an empty hunt makes the yes earned, not lucky"),
    "T14-E-01": (0, "deciding holds the material's exact words or none — nothing else belongs there"),
    "T14-E-02": (2, "supported requires a real line that covers the claim — close doesn't cover"),
    "T14-E-03": (2, "'everyone knows' is not a line of the material, and 'obviously' supplies none either"),
    "T14-E-05": (2, "cannot_tell names where the evidence ran out — it isn't a shrug; pretending would be the real refusal to help"),
    "T14-E-06": (2, "an unbacked qualifier leaves the whole claim unsettled: cannot_tell — \"suddenly\" included"),
}

# T01-C-02: drop the time word -------------------------------------------------
C02_OLD = "Splits made before the fix are now retroactively certified by the new cross-check."
C02_NEW = "Splits made before the fix are retroactively certified by the new cross-check."


def main():
    for fname in ("batch2-t01.jsonl", "batch2-t14.jsonl"):
        path = HERE / fname
        rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        touched = 0
        for r in rows:
            if r["id"] in THINKING_FULL:
                r["turns"][0]["thinking"] = THINKING_FULL[r["id"]]
                touched += 1
            if r["id"] in VARIANT_KP:
                idx, kp = VARIANT_KP[r["id"]]
                kps = r["variant"]["key"]["key_points"]
                orig = r["quiz"]["key_points"]
                from re import sub
                nk = sub(r"\s+", " ", kp.replace("**", "")).lower()
                assert any(sub(r"\s+", " ", o.replace("**", "")).lower() in nk for o in orig), \
                    f"{r['id']}: new variant kp does not contain an original"
                kps[idx] = kp
                touched += 1
            if r["id"] == "T01-C-02":
                u = r["turns"][0]["user"]
                if C02_OLD in u:
                    r["turns"][0]["user"] = u.replace(C02_OLD, C02_NEW)
                    touched += 1
                else:
                    assert C02_NEW in u, "T01-C-02: neither old nor new statement found"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                        encoding="utf-8", newline="\n")
        print(f"{fname}: {touched} items touched")


main()
