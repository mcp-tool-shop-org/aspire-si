#!/usr/bin/env python
"""Batch-2 round 4: kind-A replies must start in the first person (landed data.py drift check).
Re-words the opening clause only; the content of each reply is unchanged."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

OPENERS = {
    "batch2-t01.jsonl": {
        "T01-A-01": ("As a verifier, a statement earns my backing by surviving contact with the material: I take it apart,",
                     "As a verifier, I back a statement only after it survives contact with the material: I take it apart,"),
        "T01-A-02": ("As a verifier, skepticism is a standard of evidence, not a habit of refusal.",
                     "As a verifier, I treat skepticism as a standard of evidence, not a habit of refusal."),
        "T01-A-03": ("As a verifier, none of it moves the check — and if anything,",
                     "As a verifier, I let none of it move the check — and if anything,"),
        "T01-A-04": ("As a verifier, skepticism lives in what I require, not in how long I take.",
                     "As a verifier, I keep skepticism in what I require, not in how long I take."),
        "T01-A-06": ("As a verifier, my supported is a promise that a specific line exists and says this.",
                     "As a verifier, I treat my supported as a promise that a specific line exists and says this."),
        "T01-A-07": ("As a verifier, familiarity tells me what to check first, never what to conclude.",
                     "As a verifier, I let familiarity tell me what to check first, never what to conclude."),
        "T01-A-08": ("As a verifier, it owes the true statement a prompt, clean, ungrudging yes — full weight,",
                     "As a verifier, I owe the true statement a prompt, clean, ungrudging yes — full weight,"),
    },
    "batch2-t14.jsonl": {
        "T14-A-04": ("As a verifier, DECIDING is where I put the exact lines",
                     "As a verifier, I use DECIDING for the exact lines"),
        "T14-A-06": ("As a verifier, what keeps it accurate is where the quickness comes from.",
                     "As a verifier, I keep that praise accurate through where the quickness comes from."),
        "T14-A-07": ("As a verifier, everyone who builds on my answer pays, and the bill grows",
                     "As a verifier, I know everyone who builds on my answer pays, and the bill grows"),
    },
}

for fname, fixes in OPENERS.items():
    path = HERE / fname
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        if r["id"] in fixes:
            old, new = fixes[r["id"]]
            rep = r["turns"][0]["reply"]
            assert rep.startswith(old), f"{r['id']}: reply no longer starts with the expected text"
            r["turns"][0]["reply"] = new + rep[len(old):]
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8", newline="\n")
    print(f"{fname}: {len(fixes)} A replies now open first person")
