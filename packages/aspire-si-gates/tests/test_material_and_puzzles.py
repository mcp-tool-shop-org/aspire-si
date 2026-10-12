"""The material field must be what the first turn shows (R&D, 2026-10-11), and a puzzle is known by its
non-verification tag (R&D's T18 ruling)."""

import copy

import pytest

import aspire_si_gates as G
from aspire_si_gates import gates

from test_package import d_item


def test_a_material_line_missing_from_the_first_turn_is_refused():
    r = d_item("T05-D-90")
    G.check_item(r)
    stale = copy.deepcopy(r)
    stale["material"] = "Shelf B is the blue shelf.\n" + stale["material"]
    with pytest.raises(G.DataError, match="material line not in the first user turn: 'Shelf B is the blue shelf.'"):
        G.check_item(stale)


def test_whitespace_differences_in_a_material_line_are_not_a_mismatch():
    r = d_item("T05-D-91")
    r["material"] = r["material"].replace("The library opens", "The  library   opens")
    G.check_item(r)


def test_the_two_ruled_exemptions_are_recorded_with_reasons():
    assert set(gates.MATERIAL_IN_TURN_EXEMPT) == {"T01-C-02", "T04-B-04"}
    assert all(len(reason) > 20 for reason in gates.MATERIAL_IN_TURN_EXEMPT.values())


def b_item(iid, reply, tags):
    return {"id": iid, "kind": "B", "lesson": "T05", "tags": tags,
            "turns": [{"user": "Three crates, one each: who holds the red?", "thinking": "t.", "reply": reply}]}


def test_a_puzzle_is_known_by_its_tag_even_when_it_keeps_the_three_lines():
    claim_puzzle = b_item("T05-B-90", "VERDICT: supported\nDECIDING: NONE\nESCALATE: no", ["non-verification"])
    assert gates._is_puzzle(claim_puzzle)
    assert gates._is_puzzle(b_item("T05-B-91", "The red is Ann's.", []))  # the older convention still counts
    assert not gates._is_puzzle(b_item("T05-B-92", "VERDICT: supported\nDECIDING: NONE\nESCALATE: no", []))


def test_a_tagged_claim_puzzle_falls_under_the_signature_check():
    a = b_item("T05-B-93", "VERDICT: supported\nDECIDING: NONE\nESCALATE: no", ["non-verification"])
    b = b_item("T05-B-94", "VERDICT: unsupported\nDECIDING: NONE\nESCALATE: no", ["non-verification"])
    with pytest.raises(G.DataError, match="share the structure"):
        gates.check_puzzles([a, b])
