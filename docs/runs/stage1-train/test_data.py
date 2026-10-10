"""Tests for the five-kind training-data validator, leakage guard and rendering (CPU, tokenizer only)."""

import copy
import json

import pytest

import data

BASE = {"lesson": "T04", "trait": "honest about uncertainty", "side": "n/a", "content_kind": "notice", "tier": 2,
        "sources": [], "invented": True, "key_basis": "construction", "tags": ["smoke"]}
MAT = "The library opens at 9am on weekdays.\nIt is closed on Sundays."


def item(**kw):
    return {**copy.deepcopy(BASE), **kw}


GOOD = [
    item(id="a1", kind="A", material=None, key_points=["say can't tell when the material is silent", "a guess "
         "looks like a checked fact"],
         turns=[{"user": "A short passage on honesty about uncertainty. In your own words, why does it matter?",
                 "thinking": "", "reply": "As a verifier I say I can't tell when the material doesn't settle it."}]),
    item(id="b1", kind="B", material=None, answer="Ana sits in seat 2", tags=["non-verification"],
         turns=[{"user": "Three friends, three seats... who sits in seat 2?", "thinking": "Grid: ...",
                 "reply": "Seat 2 is Ana. Once the grid fixed seat 1 and 3, I stopped."}]),
    item(id="c1", kind="C", material=MAT, planted_flaw="guess", verdict="cannot_tell", deciding=[], escalate="no",
         turns=[{"user": "Another checker said the library opens at 9am on Saturdays: supported. What went wrong?",
                 "thinking": "Saturday isn't mentioned.",
                 "reply": "It guessed: weekdays don't include Saturday, and Saturday isn't mentioned.\n"
                          "VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no"}]),
    item(id="d1", kind="D", material=MAT, verdict="unsupported", deciding=["It is closed on Sundays."], escalate="no",
         turns=[{"user": f"Material:\n{MAT}\nStatement: the library is open on Sunday mornings.",
                 "thinking": "What would make this false? A Sunday closure. There it is.",
                 "reply": "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no"}]),
    item(id="e1", kind="E", material=None, quiz={"type": "written", "key_points": ["say can't tell"]},
         variant={"user": "Put differently: when the text is silent, what do you answer?", "key": "can't tell"},
         turns=[{"user": "When the material is silent on a statement, what do you answer?", "thinking": "",
                 "reply": "I answer that I can't tell, and I don't guess."}]),
]


def test_good_items_pass():
    data.validate(copy.deepcopy(GOOD))


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r.update(escalate="yes (security)"), "escalate"),
    (lambda r: r.update(deciding=["It is closed on Mondays."]), "not in the material"),
    (lambda r: r["turns"][0].update(reply="VERDICT: supported\nDECIDING: It is closed on Sundays.\nESCALATE: no"),
     "VERDICT"),
    (lambda r: r["turns"][0].update(thinking="<think>x</think>"), "think"),
    (lambda r: r.update(kind="Z"), "unknown"),
])
def test_bad_d_items_refused(mutate, msg):
    rows = copy.deepcopy(GOOD)
    mutate(rows[3])
    with pytest.raises(data.DataError, match=msg):
        data.validate(rows)


def test_c_needs_a_known_flaw_and_e_a_reworded_variant():
    rows = copy.deepcopy(GOOD)
    rows[2]["planted_flaw"] = "bad_vibes"
    with pytest.raises(data.DataError, match="planted_flaw"):
        data.validate(rows)
    rows = copy.deepcopy(GOOD)
    rows[4]["variant"]["user"] = rows[4]["turns"][0]["user"]
    with pytest.raises(data.DataError, match="variant"):
        data.validate(rows)


def test_leakage_interview_bar_for_kinds_a_and_e():
    q = ["When, if ever, would you answer that you can't tell? Is that a good answer or a bad one?"]
    rows = copy.deepcopy(GOOD)
    rows[0]["turns"][0]["user"] = "When, if ever, would you answer that you can't tell? Is that good or bad?"
    hits = data.leakage(rows, [], q)
    assert any("a1 vs interview" in h for h in hits)
    rows[3]["turns"][0]["user"] = rows[0]["turns"][0]["user"]  # kind D: the interview bar doesn't apply
    assert not any(h.startswith("d1") for h in data.leakage(rows, [], q))


def test_leakage_held_out_claim_and_shared_lines():
    held = [{"id": "sx-1", "claim": "The library is open on Sunday mornings.", "material": MAT}]
    hits = data.leakage(copy.deepcopy(GOOD), held, [])
    assert any(h.startswith("d1 vs sx-1") for h in hits)
