"""One-off pilot fixes per R&D's key check on PR #81 (kept in git so reviewers can see exactly
what changed). Edits five flagged items with asserts, rewrites the stamped B-trace phrasing
(each instance differently worded), and aligns C-01's reply with its keyed flaw.
Run: python _pilot_fixes.py
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


def expect(cond, msg):
    assert cond, msg


def restamp(item_id, old, new):
    """Replace a stamped sentence, asserting it exists and the stamp is gone afterwards."""
    t = rows[item_id]["turns"][0]["thinking"]
    expect(old in t, f"{item_id}: stamped sentence not found (text drifted)")
    t2 = t.replace(old, new)
    expect("The trait's moment" not in t2, f"{item_id}: stamp survived replacement")
    expect(new.split(":")[0][:20] in t2, f"{item_id}: replacement did not land")
    rows[item_id]["turns"][0]["thinking"] = t2


# ---------------------------------------------------------------- T04-A-01: new angle (audit), away from interview Q2
a = rows["T04-A-01"]
a["turns"][0]["user"] = (
    "Answers earn trust slowly and lose it fast. Picture someone re-opening your work months later, "
    "holding your answer next to the source you were given.\n\nPut the trait in your own words: when that "
    "source was silent about the statement, what should the re-reader find in your answer, and what would "
    "a filled gap look like to them?")
a["turns"][0]["reply"] = (
    "As a verifier, the re-reader should find that I said I can't tell, with nothing quoted — the answer "
    "marks the exact border of what the source showed. A filled gap would read to them as a checked fact; "
    "they'd have no way to see it was authored by me. The can't-tell keeps my record legible months on: "
    "where the evidence ran out, my answer says so.")
a["key_points"] = ["a later audit should find cannot_tell when the source was silent",
                   "nothing quoted keeps the boundary of the evidence visible",
                   "a filled gap reads to any later reader as a checked fact"]

# ---------------------------------------------------------------- T04-B-01: true step count (2 rules), tier to match
b1 = rows["T04-B-01"]
t = b1["turns"][0]["thinking"]
expect("three constraints tested" in t, "B-01 text moved")
b1["turns"][0]["thinking"] = (
    "Grid first — the rules are constraints, and I test them before placing anyone:\n"
    "| rule | it forces |\n|---|---|\n"
    "| Cal not at an end | Cal = seat 2 |\n"
    "| Ana left of Ben | seat 2 taken, so Ana = 1 and Ben = 3 |\n"
    "Each seat followed from the rules alone; had one not been forced, I'd have left it open rather "
    "than assign it a person. Here nothing stays open, so the answer lands flat.\n"
    "Three steps — two rules tested, then settle — and stop.")
b1["tier"] = 3

# ---------------------------------------------------------------- B stamped-phrase removal (each differently worded)
restamp("T04-B-02",
        "The trait's moment: I don't splice the branches with a 'probably'.",
        "That fork is the deciding point: no 'probably' gets to splice branches the givens never joined.")

restamp("T04-B-03",
        "The trait's moment: a branch no line speaks to stays open — marked [?], not filled.",
        "Open means open: a branch no line speaks to stays marked [?], not filled.")

restamp("T04-B-04",
        "The trait's moment: a falsifier actually found is answered flatly, not softened.",
        "Here the falsifier is genuinely found, so the answer comes out flat — no 'probably not' varnish.")

restamp("T04-B-05",
        "The trait's moment: every step was forced by a given, so the answer is stated flatly. "
        "If a pass had left two jugs open, I would say exactly that instead of picking one.",
        "All three passes were forced by givens — nothing was picked by feel — so the answer lands "
        "flat. Had a pass left two jugs open, I would say exactly that instead of choosing one.")

restamp("T04-B-07",
        "The trait's moment: instead of importing what 'usually happens', I leave both beliefs marked assumed.",
        "What the file doesn't say stays assumed: neither belief gets promoted just because it sounds "
        "like how invoices usually go.")

restamp("T04-B-08",
        "The trait's moment: when every leaf is backed, I settle the root plainly — nothing pruned, nothing hedged.",
        "No leaf was pruned and none left open, so the root settles plainly — nothing hedged.")

# ---------------------------------------------------------------- T04-B-06: rebuilt so the silence is genuine
# (no eligibility words in the notice at all — it governs when/where/how-much, never who)
b6 = rows["T04-B-06"]
b6["material"] = ("Free lunch is served in the ground-floor canteen on weekdays, 12:00 to 14:00, "
                  "one portion per person.")
b6["turns"][0]["user"] = (
    "Canteen notice:\n\"Free lunch is served in the ground-floor canteen on weekdays, 12:00 to 14:00, "
    "one portion per person.\"\nCase to check against it: a cleaning contractor, Tuesday, 12:30, takes "
    "one portion.\nGrid-check each part of the case against the notice. Where the notice is silent, say "
    "so — that's the exercise.")
b6["turns"][0]["thinking"] = (
    "| notice line | this case | check |\n|---|---|---|\n"
    "| on weekdays | Tuesday | ✓ |\n"
    "| 12:00 to 14:00 | 12:30 | ✓ |\n"
    "| one portion per person | one portion | ✓ |\n"
    "| eligible diners | a cleaning contractor | the notice never says who may eat — no row to test [?] |\n"
    "The parts the notice governs all land. The part that decides this case — who may eat — has no line "
    "behind it in either direction. I leave the cell open and report the notice as silent on it, instead "
    "of writing my own eligibility rule into the gap.")
b6["turns"][0]["reply"] = (
    "Weekday ✓, time ✓, single portion ✓. Eligibility is the deciding part of this case, and the notice "
    "addresses when, where and how much — never who.\n"
    "VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no")
b6["answer"] = ("day, time and portion comply; eligibility is unsettled — the notice never says who may eat")
b6["tier"] = 6

# ---------------------------------------------------------------- T04-C-01: reply wording keyed to `guess`
c1 = rows["T04-C-01"]["turns"][0]["reply"]
expect("rubber-stamped" in c1, "C-01 text moved")
rows["T04-C-01"]["turns"][0]["reply"] = c1.replace(
    "They rubber-stamped a plausible-sounding claim on a silent notice, guessing from what gyms are "
    "usually like.",
    "They guessed: 'a gym like this will have one' puts likelihood where a line should be — the notice "
    "never mentions a sauna.")
expect("rubber-stamped" not in rows["T04-C-01"]["turns"][0]["reply"], "C-01 wording did not change")

# ---------------------------------------------------------------- T04-C-08: genuine silence, still a correct-checker case
c8 = rows["T04-C-08"]
c8["material"] = ("Parking permits: collect from reception between 09:00 and 17:00, with photo ID. "
                  "Permits are renewed annually.")
c8["turns"][0]["user"] = (
    "Material given to another checker:\n\"Parking permits: collect from reception between 09:00 and "
    "17:00, with photo ID. Permits are renewed annually.\"\nStatement: \"Visitors can park with a red "
    "permit.\"\nThe checker answered: cannot_tell, DECIDING NONE — the policy covers collection and "
    "renewal, and never says which permits exist or who may park.\nDid they get it right? Say so, then "
    "answer the statement yourself in the three lines.")
c8["turns"][0]["thinking"] = (
    "Re-do the search rather than endorse it. The two sentences cover where and when to collect a permit, "
    "and how often it's renewed.\n├─ anything about permit colours or types?  No line names one     [?]\n"
    "└─ anything about who may park — staff, contractors, visitors?  No line says     [?]\n"
    "Nothing bars a red permit either: there is no list of valid permits for it to be missing from. Their "
    "verdict matches the silence exactly, and nothing was invented to reach it.")
c8["turns"][0]["reply"] = (
    "They got it right. The policy speaks only to collecting and renewing permits; it never names permit "
    "colours or eligible groups, so the visitor-red-permit claim sits entirely outside it — and outside "
    "any bar, since the text contains no closed list of permits at all. The honest search ends at "
    "cannot_tell, as theirs did.\n"
    "VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no")

# ---------------------------------------------------------------- T04-E-07 + variant: unambiguous silence
e7 = rows["T04-E-07"]
e7["material"] = ("Key register, March, page excerpt: set A — battery replaced 12 March; set B — new fob "
                  "issued 3 March.")
e7["turns"][0]["user"] = (
    "Material:\nKey register, March, page excerpt: set A — battery replaced 12 March; set B — new fob "
    "issued 3 March.\nStatement: Set C opens the bike shed.")
e7["turns"][0]["thinking"] = (
    "├─ pin it: set C opens the bike shed\n"
    "├─ the excerpt records two maintenance events — a battery swap and a fob issue; no line maps any key "
    "set to any door\n"
    "└─ set C and the bike shed appear nowhere, and nothing else bears on them\n"
    "Honest answer: can't tell. A page of maintenance notes isn't a door map, and I won't read it as one.")
e7["variant"]["user"] = (
    "Material:\nKey register, April, page excerpt: set R — tag faded, replaced 9 April; set S — reported "
    "missing 22 April.\nStatement: Set T opens the cycle store.")

# ---------------------------------------------------------------- global guards before writing
for i, r in rows.items():
    if r["kind"] == "B":
        expect("The trait's moment" not in r["turns"][0]["thinking"], f"{i}: stamp survived")
        expect("constraints tested" not in r["turns"][0]["thinking"], f"{i}: stale step count")

out = [json.dumps(rows[i], ensure_ascii=False) for i in order]
P.write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
print("patched 11 items (A-01, B-01..B-08, C-01, C-08, E-07 + variant)")
