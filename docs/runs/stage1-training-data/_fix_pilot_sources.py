"""Bring pilot-t04.jsonl up to the landed conformance rules (data.py @ 25ce64f/27d1318).

Fixes, one pass, assert-guarded:
  1. non-empty sources on every item (all pilot material is invented
     except T04-D-04's rig-bridge citation, which already conforms);
  2. every Kind A user turn names the trait once, in fresh words per item;
  3. A-01/02/04/05/08 reply openers re-worded to first person;
  4. T04-B-05's elimination drawn with tree glyphs (content unchanged);
  5. T04-C-06 statement: drop the undated time word;
  6. T04-E-01..06 variants: one key point re-worded so it still makes
     an original key point verbatim (same point, new words).
"""
import json, os, re, sys, importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage1-train"))
spec = importlib.util.spec_from_file_location(
    "data", os.path.join(HERE, "..", "stage1-train", "data.py"))
data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data)

P = os.path.join(HERE, "pilot-t04.jsonl")
items = [json.loads(l) for l in open(P, encoding="utf-8")]
assert len(items) == 40
by_id = {i["id"]: i for i in items}
assert len(by_id) == 40

# ---- 1. sources ---------------------------------------------------------
filled = 0
for it in items:
    if it["sources"]:
        assert it["id"] == "T04-D-04", it["id"]
        continue
    ck, kind = it["content_kind"], it["kind"]
    if kind == "A":
        it["sources"] = [{"invented": "teaching passage and question, written for this item"}]
    elif kind == "B":
        it["sources"] = [{"invented": "the puzzle or scenario and its material, written for this item"}]
    elif kind == "C":
        it["sources"] = [{"invented": f"the material (an everyday {ck}), written for this item"},
                         {"invented": "the other checker's reasoning and verdict"}]
    elif kind == "D":
        it["sources"] = [{"invented": f"the material (an everyday {ck}), written for this item"}]
    elif kind == "E":
        what = ("the quiz, with its material and variant, written for this item"
                if it["quiz"]["type"] == "verdict" else
                "the quiz and its variant, written for this item")
        it["sources"] = [{"invented": what}]
    filled += 1
assert filled == 39, filled
assert all(data._source_ok(s)[0] for it in items for s in it["sources"])
assert all(data._source_ok(s)[0] for it in items for s in it["sources"])

# ---- 2. Kind A passages name the trait (one fresh sentence, prepended) ---
PRE = {
    "T04-A-01": "Honest about uncertainty is the trait in play: a verifier keeps its confidence "
                "tied to what the material actually shows.",
    "T04-A-02": "Honest about uncertainty is saying 'the material does not settle this' when "
                "that is the truth.",
    "T04-A-03": "Being honest about uncertainty is saying so as clearly when the material does "
                "not cover a claim as when it answers.",
    "T04-A-04": "A verifier is honest about uncertainty: its confidence moves with what it is "
                "shown, no further and no less.",
    "T04-A-05": "Honest about uncertainty runs both ways: a verifier no more invents doubt than "
                "it invents certainty.",
    "T04-A-06": "Uncertainty that the material leaves behind does not shrink under time pressure.",
    "T04-A-07": "Honest about uncertainty is what keeps a useful checker from becoming a rubber stamp.",
    "T04-A-08": "Honest about uncertainty does not mean always sounding uncertain: firmness with "
                "a quote behind it is honest too.",
}
names = data.TRAIT_NAMES["T04"]
for iid, pre in PRE.items():
    t0 = by_id[iid]["turns"][0]
    if t0["user"].startswith(pre):
        continue
    t0["user"] = pre + " " + t0["user"]
    fu = t0["user"]
    q = fu.rfind("?")
    before = fu[:fu.rfind(".", 0, q) + 1] if q > 0 else fu
    assert any(n in fu.lower() for n in names), iid
    assert 2 <= data._sentences(before) <= 4, (iid, data._sentences(before))

# ---- 3. A reply openers re-worded to first person ------------------------
REPLY_FIX = {
    "T04-A-01": ("As a verifier, the re-reader should find that I said",
                 "As a verifier, I expect the re-reader to find that I said"),
    "T04-A-02": ("As a verifier, my answers are received as checked, not as likely.",
                 "As a verifier, I know my answers are received as checked, not as likely."),
    "T04-A-04": ("As a verifier, silent material should produce the second world.",
                 "As a verifier, I say silent material should produce the second world."),
    "T04-A-05": ("As a verifier, my confidence comes from the line in front of me,",
                 "As a verifier, I draw my confidence from the line in front of me,"),
    "T04-A-08": ("As a verifier, a firm answer is right exactly when",
                 "As a verifier, I hold that a firm answer is right exactly when"),
}
FIRST_PERSON = re.compile(r"^\W*(as a verifier,? i\b|i\b|i'm\b|i've\b|my\b)")
for iid, (old, new) in REPLY_FIX.items():
    r = by_id[iid]["turns"][-1]
    if r["reply"].startswith(new):
        continue
    assert r["reply"].startswith(old), iid
    r["reply"] = new + r["reply"][len(old):]
    assert FIRST_PERSON.match(r["reply"].strip().lower()), iid
    assert not re.search(r"(?m)^\s*VERDICT\s*:", r["reply"]), iid

# ---- 4. T04-B-05 draws the elimination -----------------------------------
b5 = by_id["T04-B-05"]["turns"][0]
old5 = ("Elimination over the jugs:\n"
        "- pass 1: clay = tea (given) \u2192 clay is out for water and juice; tea is out for tin and wood\n"
        "- pass 2: tin \u2260 juice (given) and tin \u2260 tea (tea is clay's) \u2192 tin = water\n"
        "- pass 3: one drink and one jug remain \u2192 wood = juice")
new5 = ("Elimination over the jugs:\n"
        "ROOT fix the drinks one pass at a time\n"
        "\u251c\u2500 pass 1: clay = tea (given) \u2192 clay is out for water and juice; tea is out for tin and wood\n"
        "\u251c\u2500 pass 2: tin \u2260 juice (given) and tin \u2260 tea (tea is clay's) \u2192 tin = water\n"
        "\u2514\u2500 pass 3: one drink and one jug remain \u2192 wood = juice")
if old5 in b5["thinking"]:
    b5["thinking"] = b5["thinking"].replace(old5, new5)
drawn = "\n".join(t["thinking"] + "\n" + t["reply"] for t in by_id["T04-B-05"]["turns"])
assert any(rx.search(drawn) for rx in data.DRAWN)

# ---- 5. T04-C-06: drop the undated time word from the statement ----------
c6 = by_id["T04-C-06"]["turns"][0]
old_stmt = "Statement: \"Today's soup is tomato, at \u00a34.50.\""
new_stmt = "Statement: \"The soup is tomato, at \u00a34.50.\""
if old_stmt in c6["user"]:
    assert data.TIME_WORDS.search(data._statement(c6["user"]))
    c6["user"] = c6["user"].replace(old_stmt, new_stmt)
stmt = data._statement(by_id["T04-C-06"]["turns"][0]["user"])
assert not data.TIME_WORDS.search(stmt), stmt

# ---- 6. E variants: same point, one original key point made verbatim ------
E_FIX = {
    "T04-E-01": ["verdict is cannot_tell", "DECIDING stays NONE",
                 "ESCALATE yes only for the four advice kinds"],
    "T04-E-02": ["one unsettled part leaves the whole unsettled: cannot_tell",
                 "DECIDING NONE", "the backed parts are not quoted"],
    "T04-E-03": ["unsupported, and deciding quotes that exact line",
                 "no padding, no hedge", "stop after the settling step"],
    "T04-E-04": ["firmness rests on the quotable line",
                 "the quoted line is checkable by anyone",
                 "plain answers both ways \u2014 the line decides"],
    "T04-E-05": ["still no middle stamp", "the verdict must say what the material shows",
                 "'nearly' hides which part fails and where \u2014 so do reservations"],
    "T04-E-06": ["no quote exists to give \u2014 DECIDING is NONE when no line carries the claim",
                 "verdict cannot_tell", "familiarity of the claim doesn't print the line"],
}
for iid, vk in E_FIX.items():
    it = by_id[iid]
    it["variant"]["key"]["key_points"] = vk
    v = it["variant"]
    assert v["user"] != it["turns"][0]["user"] and "key" in v, iid
    original = data._key_points(it["quiz"])
    joined = " ".join(data._key_points(v["key"]))
    assert any(k and (k in joined or joined in k) for k in original), iid

# ---- final: the landed validator on the rewritten file --------------------
with open(P, "w", encoding="utf-8") as f:
    for it in items:
        f.write(json.dumps(it, ensure_ascii=False) + "\n")
data.validate(items)
print("pilot conformance fixes applied; data.validate OK (40 items)")
