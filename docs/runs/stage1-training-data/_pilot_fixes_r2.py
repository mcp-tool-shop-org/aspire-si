"""Pilot fix round 2 (PR #81): rebuild T04-B-06 per R&D's comment. The notice now names an
eligibility condition ("registered members") that the case cannot be checked against — nothing
says whether contractors are members. Stays cannot_tell, open eligibility cell kept, tier 6.
Run: python _pilot_fixes_r2.py
"""
import json
from pathlib import Path

P = Path(__file__).parent / "pilot-t04.jsonl"
rows = {}
order = []
for x in P.read_text(encoding="utf-8").splitlines():
    if x.strip():
        r = json.loads(x)
        rows[r["id"]] = r
        order.append(r["id"])

b6 = rows["T04-B-06"]
assert "never says who may eat" in b6["turns"][0]["thinking"], "B-06 not at round-1 text (drifted)"
assert b6["material"] and "registered" not in b6["material"], "B-06 material unexpected"

b6["material"] = ("Free lunch for registered members in the ground-floor canteen on weekdays, "
                  "12:00 to 14:00, one portion per person.")
b6["turns"][0]["user"] = (
    "Canteen notice:\n\"Free lunch for registered members in the ground-floor canteen on weekdays, "
    "12:00 to 14:00, one portion per person.\"\nCase to check against it: a cleaning contractor, "
    "Tuesday, 12:30, takes one portion.\nGrid-check each part of the case against the notice. Where "
    "the case can't be checked against the notice, say so — that's the exercise.")
b6["turns"][0]["thinking"] = (
    "| notice line | this case | check |\n|---|---|---|\n"
    "| for registered members | a cleaning contractor | nothing says whether contractors are "
    "registered — no line to test it against [?] |\n"
    "| on weekdays | Tuesday | ✓ |\n"
    "| 12:00 to 14:00 | 12:30 | ✓ |\n"
    "| one portion per person | one portion | ✓ |\n"
    "The eligibility condition is there this time, and it still decides nothing here: the notice "
    "never defines membership, so the case fact has no line to meet. Three parts land, the deciding "
    "part stays open, and I say so rather than guess membership in either direction.")
b6["turns"][0]["reply"] = (
    "Weekday ✓, time ✓, single portion ✓. Eligibility is the deciding part of this case, and the "
    "notice does address it — registered members — but never says whether a cleaning contractor "
    "counts as one, and no list of the registered sits beside it to check against.\n"
    "VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no")
b6["answer"] = ("day, time and portion comply; the case can't be checked against the membership "
                "condition — the notice never says whether contractors are registered members")
assert b6["tier"] == 6

# guards: verdict/side untouched, no stamp phrase, no eligibility-implying leak
assert "The trait's moment" not in b6["turns"][0]["thinking"]
assert "VERDICT: cannot_tell" in b6["turns"][0]["reply"]

out = [json.dumps(rows[i], ensure_ascii=False) for i in order]
P.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
print("B-06 rebuilt (round 2): named condition, uncheckable case, still cannot_tell / tier 6")
