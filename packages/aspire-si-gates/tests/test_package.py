"""Tests for the package surface: identity, the reason table, check_against, and validate as the composition."""

import copy
import re

import pytest

import aspire_si_gates as G
from aspire_si_gates import gates

MAT = "The library opens at 9am on weekdays.\nIt is closed on Sundays."


def d_item(iid, material=MAT, deciding=("It is closed on Sundays.",), verdict="unsupported", lesson="T05",
           opening="Case one.", thinking=None, pair=None):
    m = f"{material}\nNotice {iid}."
    r = {"id": iid, "lesson": lesson, "trait": "independent", "kind": "D", "side": "n/a", "content_kind": "notice",
         "tier": 2, "sources": [{"invented": "test"}], "invented": True, "key_basis": "construction",
         "material": m, "verdict": verdict, "deciding": list(deciding), "escalate": "no",
         "turns": [{"user": f"{opening} Material:\n{m}\nStatement: the library is open on Sunday mornings.",
                    "thinking": thinking or f"{iid}.",
                    "reply": f"VERDICT: {verdict}\nDECIDING: {' | '.join(deciding) or 'NONE'}\nESCALATE: no"}]}
    if pair:
        r["pair"] = pair
    return r


def test_identity_is_stable_and_versioned():
    a, b = G.gates_identity(), G.gates_identity()
    assert a == b and a["version"] == G.GATES_VERSION and re.fullmatch(r"[0-9a-f]{64}", a["sha256"])


def test_reason_table_is_pinned():
    assert G.REASONS == {
        "DUPLICATE_ID": "the id {x} is already used",
        "SPAN_REUSED": "the wording '{x}' is already in 2 items of this trait; reword it",
        "FRAMING_REUSED": "the framing '{x}' is already used; write a new one",
        "MATERIAL_REUSED": "this material is already used; take another excerpt",
        "PUZZLE_SHAPE_REUSED": "this trait already has a puzzle shaped {x}; use another structure",
        "PUZZLE_SHAPE_BATCH": "two puzzles in this batch are already shaped {x}; use another structure",
        "OPENING_REUSED": "the opening '{x}' already opens 2 items of this trait; vary it",
        "B_PHRASE_REUSED": "the phrase '{x}' is already in 2 kind B items of this trait; reword it",
        "TRACE_REUSED": "the wording '{x}' comes from a stored model trace; item text is never drawn from traces",
    }


def test_trace_reused_refuses_wording_from_a_stored_trace():
    sk = "I looked for a line that states it, and for a line that rules it out."
    row = d_item("t1", thinking=sk, lesson="T04")  # pre-amendment lessons are checked too
    trace = "Okay. I looked for a line that states it, and for a line that rules it out. Then I decided."
    out = G.check_not_from_traces(row, ["unrelated words only here", trace])
    assert [c for c, _ in out] == ["TRACE_REUSED"]
    assert "comes from a stored model trace" in out[0][1]
    assert G.check_not_from_traces(row, ["a short trace", ""]) == []
    assert G.check_not_from_traces(row, []) == []


def test_trace_reused_ignores_the_rows_own_material_quoted_in_a_trace():
    row = d_item("t2", thinking="Short.")
    quoting = "The material says: " + row["material"].replace("\n", " ") + " so that settles it."
    assert G.check_not_from_traces(row, [quoting]) == []
    # but the same trace with the row's own reasoning in it is refused
    row["turns"][0]["thinking"] = "so that settles it and the statement is not supported by the notice"
    trace = quoting + " and the statement is not supported by the notice"
    assert [c for c, _ in G.check_not_from_traces(row, [trace])] == ["TRACE_REUSED"]


def test_trace_reused_leaves_out_wording_an_accepted_row_already_holds():
    """A trained student reproduces the dataset's wording; that wording isn't model-originated (R&D)."""
    sk = "I looked for a line that states it, and for a line that rules it out."
    trace = "First, " + sk + " Then I answered."
    row = d_item("t3", thinking=sk)
    assert [c for c, _ in G.check_not_from_traces(row, [trace])] == ["TRACE_REUSED"]
    assert G.check_not_from_traces(row, [trace], accepted=[d_item("t0", thinking=sk)]) == []


def test_trace_reused_leaves_out_the_statement_line_and_the_three_line_block():
    row = d_item("t4", thinking="Short.")
    row["turns"][0]["user"] = row["turns"][0]["user"].replace(
        "Statement: the library is open on Sunday mornings.",
        "Statement: the library on the corner is open every Sunday morning until noon.")
    trace = ("The claim is that the library on the corner is open every Sunday morning until noon.\n"
             "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no")
    assert G.check_not_from_traces(row, [trace]) == []


def test_trace_reused_keeps_digits_so_a_different_number_is_a_different_span():
    row = d_item("t5", thinking="the late fee is 4 dollars per extra hour")
    assert G.check_not_from_traces(row, ["So the late fee is 7 dollars per extra hour."]) == []
    assert [c for c, _ in G.check_not_from_traces(row, ["So the late fee is 4 dollars per extra hour."])] \
        == ["TRACE_REUSED"]


def test_check_item_needs_no_other_row():
    G.check_item(d_item("a"))
    bad = d_item("b", deciding=("It is closed on Mondays.",))
    with pytest.raises(G.DataError, match="not verbatim"):
        G.check_item(bad)


def test_check_against_spans_framing_material_and_ids():
    sk = "I looked for a line that states it, and for a line that rules it out."
    acc = [d_item("a", opening="Case one.", thinking=sk), d_item("b", opening="Case two.", thinking=sk)]
    new = d_item("c", opening="Case three.", thinking=sk)
    codes = [c for c, _ in G.check_against(new, acc)]
    assert codes == ["SPAN_REUSED"]
    msg = dict(G.check_against(new, acc))["SPAN_REUSED"]
    assert msg.startswith("the wording '") and msg.endswith("is already in 2 items of this trait; reword it")
    assert [c for c, _ in G.check_against(d_item("a", opening="Case one."), acc)] == \
        ["DUPLICATE_ID", "FRAMING_REUSED", "MATERIAL_REUSED"]
    twin = d_item("e", opening="Case five.")
    twin["material"] = acc[0]["material"]
    assert "MATERIAL_REUSED" in [c for c, _ in G.check_against(twin, acc)]
    acc[0]["pair"] = twin["pair"] = "p1"
    assert "MATERIAL_REUSED" not in [c for c, _ in G.check_against(twin, acc)]
    assert G.check_against(d_item("z", opening="Case zed.", lesson="T04"), acc) == []  # pre-amendment lesson


def test_check_against_puzzle_shapes():
    def puzzle(iid, text, lesson="T05"):
        return {"id": iid, "lesson": lesson, "kind": "B", "answer": "x",
                "turns": [{"user": text, "thinking": "| a | b |", "reply": "Answer: x."}]}
    lamps = "Three lamps in a row. The red lamp is left of the blue lamp. The blue lamp is left of the green lamp."
    pots = "A cook has three pots. The small pot holds less than the mid pot. The mid pot holds less than the big pot."
    assert [c for c, _ in G.check_against(puzzle("p2", pots), [puzzle("p1", lamps)])] == ["PUZZLE_SHAPE_REUSED"]
    acc = [puzzle("p1", lamps, "T06"), puzzle("p2", pots, "T13")]
    assert [c for c, _ in G.check_against(puzzle("p3", lamps), acc)] == ["PUZZLE_SHAPE_BATCH"]


def test_validate_is_check_item_then_check_batch():
    rows = [d_item("a"), copy.deepcopy(d_item("a"))]
    with pytest.raises(G.DataError, match="duplicate id"):
        G.validate(rows)
    G.check_item(rows[1])  # alone, the same row is fine: the duplicate is a whole-file fact
    assert gates.validate.__doc__.strip().startswith("check_item on every row")


def test_a_lesson_rank_is_a_figure_not_a_refusal():
    lessons = {"T05": "## 5. As a verifier, you are independent\nReach your own verdict before reading anyone else's.",
               "T13": "## 13. As a verifier, you don't agree just to please\nFlattery and seniority are not evidence."}
    assert set(gates.trait_sections("\n".join(lessons.values()))) == {"T05", "T13"}
    a = {"id": "a1", "lesson": "T05", "kind": "A",
         "turns": [{"user": "Reach your own verdict first.", "thinking": "", "reply": "I reach my own verdict."}]}
    drift = {"id": "a2", "lesson": "T05", "kind": "A",
             "turns": [{"user": "Flattery is not evidence.", "thinking": "", "reply": "Seniority and flattery."}]}
    ranks = gates.a_lesson_ranks([a, drift], lessons)
    assert ranks["a1"] == {"rank": 1, "nearest": "T05"} and ranks["a2"]["nearest"] == "T13"
    rep = gates.batch_report([a, drift], lessons=lessons)
    assert rep["a_lesson_rank"]["own_nearest"] == 1 and rep["a_lesson_rank"]["outside_top3"] == []
    assert "a_lesson_rank" not in gates.batch_report([a, drift])


def test_scorer_rules_are_exposed():
    assert G.GATES_VERSION == "1.7.1"
    assert G.critic_mismatch(["a b c"], ["x y z"], "a b c\nx y z") is True
    assert G.deciding_covers(["a b c d"], ["a b c"], "a b c d") is True


CLOSE = "VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no\ncannot_tell"


def b_item(iid, thinking, reply, lesson="T05"):
    return {"id": iid, "lesson": lesson, "kind": "B",
            "turns": [{"thinking": thinking, "reply": reply}]}


def test_identical_closing_lines_are_not_a_stamp():
    rows = [
        b_item("a", "The file names an amount only.", "The date is absent.\n" + CLOSE),
        b_item("b", "The notice names a time only.", "The distance is absent.\n" + CLOSE),
        b_item("c", "The card names a boat only.", "The dusk line is extra.\n" + CLOSE),
    ]
    gates.check_stamps(rows)


def test_a_prose_phrase_across_three_items_is_still_refused():
    phrase = "the red lamp stays on the end hook tonight"
    rows = [
        b_item("a", phrase + ".", "one.\n" + CLOSE),
        b_item("b", phrase + ".", "two.\n" + CLOSE),
        b_item("c", phrase + ".", "three.\n" + CLOSE),
    ]
    with pytest.raises(G.DataError, match="kind B phrase repeats"):
        gates.check_stamps(rows)
