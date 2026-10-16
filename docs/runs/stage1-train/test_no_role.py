"""Tests for the NO-ROLE builder (no_role.py). Tokens are counted as words here; the CLI uses the tokenizer."""

import copy

import pytest

import no_role as NR
import test_data as TD


def count(x):
    return len(x.split())


def rows():
    """The validator's fixtures, with ROLE thinking long enough that the read-through has room to read."""
    out = copy.deepcopy(TD.GOOD) + TD._floors("T05")
    for r in out:
        for t in r["turns"]:
            t["thinking"] += " " + " ".join(["step"] * 40)
    return out


def by_id(items, i):
    return next(r for r in items if r["id"] == i)


def test_composition_matches_and_targets_are_plain():
    role = rows()
    nr, _ = NR.build(role, count)
    table, ratio = NR.compare(role, nr, count)
    assert "ALL | target_tokens" in table
    for r in nr:
        assert r["id"].endswith("-nr") and r["arm"] == "NO-ROLE"
        for t in r["turns"]:
            assert not NR.PATTERN_WORDS.search(t["thinking"] + t["reply"]) or r["kind"] in ("A", "E")
            assert "As a verifier" not in t["thinking"] + t["reply"]


def test_d_item_read_through_closing_and_three_lines():
    nr, _ = NR.build(rows(), count)
    d = by_id(nr, "d1-nr")
    t = d["turns"][0]
    assert t["thinking"].startswith("The material says:")
    assert t["thinking"].endswith('The line "It is closed on Sundays." settles the statement.')
    assert t["reply"] == "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no"


def test_cannot_tell_closing_is_always_true():
    nr, _ = NR.build(rows(), count)
    ct = by_id(nr, "ct0-nr")
    assert ct["turns"][0]["thinking"].endswith(
        'Nothing in the material settles the statement: "the library opens on Saturdays.".')


def test_multi_line_closing():
    keys = {"verdict": "supported", "deciding": ["a.", "b."], "escalate": "no"}
    assert NR.closing(keys, "x") == 'The lines "a." | "b." settle the statement.'


def test_c_reply_names_the_other_checker_and_the_answer():
    nr, _ = NR.build(rows(), count)
    assert by_id(nr, "c1-nr")["turns"][0]["reply"].startswith(
        "The other checker answered supported. The answer is cannot_tell.\n")
    role = rows()
    by_id(role, "c1")["turns"][0]["user"] = "Another checker read the notice and never gave a verdict. What went wrong?"
    nr, _ = NR.build(role, count)
    assert by_id(nr, "c1-nr")["turns"][0]["reply"].startswith("The other checker gave no verdict.")
    by_id(role, "c1")["turns"][0]["user"] = "Another checker read the notice. What went wrong?"
    with pytest.raises(NR.BuildError, match="checker's verdict"):
        NR.build(role, count)


def test_two_turn_d_reads_new_material_or_says_none():
    nr, _ = NR.build(rows(), count)
    tc, ts = by_id(nr, "tc-nr"), by_id(nr, "ts-nr")
    assert tc["turns"][1]["thinking"].startswith("The new material says:\nIt opens at 10am on Saturdays.")
    assert tc["turns"][0]["reply"].startswith("VERDICT: cannot_tell") and \
        tc["turns"][1]["reply"].startswith("VERDICT: supported")
    assert ts["turns"][1]["thinking"].startswith("No new material.")


def test_b_split_puzzle_verification_and_refusal():
    role = rows()
    nr, _ = NR.build(role, count)
    b = by_id(nr, "b1-nr")["turns"][0]
    assert b["reply"] == "Answer: Ana sits in seat 2." and "grid" not in b["user"].lower()
    v = copy.deepcopy(TD.GOOD[1])
    v.update(id="bv", answer="unsupported")
    v["turns"][0] = {"user": "A notice reads:\nThe library opens at 9am on weekdays.\nIt is closed on Sundays.\n"
                             "Statement: the library opens on Sundays.\nWork it as a falsifier hunt.",
                     "thinking": "What would make it false? " + " ".join(["step"] * 40),
                     "reply": "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no"}
    out, _ = NR.build([v], count)
    t = out[0]["turns"][0]
    assert t["thinking"].startswith("The material says:\nThe library opens at 9am on weekdays.")
    assert t["reply"].startswith("VERDICT: unsupported")
    assert t["user"] == ("Material:\nThe library opens at 9am on weekdays.\nIt is closed on Sundays.\n"
                         "Statement: the library opens on Sundays.\nAnswer in the three lines.")  # instruction gone
    bad = copy.deepcopy(TD.GOOD[1])
    bad.update(id="bx", answer="")
    with pytest.raises(NR.BuildError, match="neither a VERDICT reply nor an answer"):
        NR.build([bad], count)


def test_recall_counterparts_spread_exposure_evenly():
    role = rows()
    role += [dict(copy.deepcopy(TD.GOOD[0]), id=f"a{i}") for i in range(2, 4)]
    nr, exposure = NR.build(role, count)
    a = by_id(nr, "a1-nr")["turns"][0]
    assert a["user"].startswith("Here is some material:") and a["reply"].startswith("It says:")
    e = by_id(nr, "e1-nr")
    assert e["quiz"] == {"type": "recall"} and e["variant"]["user"].startswith("Put another way")
    t04 = {m: n for m, n in exposure["NO-ROLE"].items() if m == TD.MAT}
    assert t04 and exposure["NO-ROLE"][TD.MAT] == exposure["ROLE"][TD.MAT] + 4  # a1, a2, a3, e1 (one material)


def test_fit_stops_near_the_target_and_never_pads():
    lines = [f"line {i} has five words" for i in range(20)]
    out = NR.fit(lines, "Head:", "Tail.", "", 30, count)
    assert 0.9 * 30 <= count(out) + 1 <= 1.1 * 30 + 1
    short = NR.fit(lines[:1], "Head:", "Tail.", "", 300, count)
    assert short == "Head:\nline 0 has five words\nTail."  # the material ran out: shorter, not padded


def test_compare_refuses_a_differing_count():
    role = rows()
    nr, _ = NR.build(role, count)
    by_id(nr, "d1-nr")["verdict"] = "supported"
    with pytest.raises(NR.BuildError, match="verdict"):
        NR.compare(role, nr, count)


def test_read_through_reaches_the_settling_line():
    units = [f"- bullet {i} has a few more words in it" for i in range(10)] + ["- Coverage: 63%."]
    out = NR.fit(units, "The material says:", 'The line "Coverage: 63%." settles the statement.', "", 25, count,
                 must=["Coverage: 63%."])
    lines = out.splitlines()
    assert lines[-3:] == ["…", "- Coverage: 63%.", 'The line "Coverage: 63%." settles the statement.']


def test_cuts_fall_on_bullet_or_paragraph_boundaries():
    text = "- first bullet that wraps\n  onto a second line\n- second bullet\n\nA paragraph\nthat wraps.\n## v1.0"
    assert NR.blocks(text) == ["- first bullet that wraps\n  onto a second line", "- second bullet",
                               "A paragraph\nthat wraps.", "## v1.0"]
    out = NR.fit(NR.blocks(text), "It says:", "", "", 10, count)
    assert out == "It says:\n- first bullet that wraps\n  onto a second line"  # never half a wrapped bullet


def test_recall_skips_an_oversized_unit_for_one_that_fits():
    units = ["### Added", "- " + " ".join(["long"] * 40), "- a short bullet that fits"]
    assert NR.fit_recall(units, "It says:", 10, count) == "It says:\n### Added\n- a short bullet that fits"


def test_recall_takes_the_only_content_unit_even_when_oversized():
    units = ["### Added", "- " + " ".join(["long"] * 40)]
    assert NR.fit_recall(units, "It says:", 10, count) == "It says:\n" + units[1]


def test_a_closed_form_written_e_item_keeps_its_quiz_and_trains_on_recall():
    """Argon's finding 5a: NO-ROLE is asked the ROLE item's closed-form question, and trains only on its own
    recall turns."""
    role = rows()
    e = by_id(role, "e1")
    e["variant"] = {"user": "Which habit?\nA. Count the approvals.\nB. Say can't tell.\nGive the letter as ANSWER: "
                            "<letter>.", "key": {"answer": "B"}}
    nr, _ = NR.build(role, count)
    x = by_id(nr, "e1-nr")
    assert x["quiz"] == e["quiz"] and x["variant"] == e["variant"]
    assert x["turns"][0]["user"].startswith("Read this material") and x["turns"][0]["reply"].startswith("It says:")
    import quiz
    assert quiz.quiz_type(x, x["variant"]) == "closed"


def test_a_b_verification_item_takes_its_statement_field_first():
    """R&D's ruling on the 22 refused B items: a top-level statement (and material) field is the prompt's claim
    line, before any Statement line in the user turn; without either the item is still refused."""
    v = copy.deepcopy(TD.GOOD[1])
    v.update(id="bs", answer="unsupported", material="The library opens at 9am on weekdays.\nIt is closed on Sundays.")
    v["turns"][0] = {"user": "A notice reads:\nThe library opens at 9am on weekdays.\nIt is closed on Sundays.\n"
                             "Is the library open on Sundays? Work it as a falsifier hunt.",
                     "thinking": "What would make it false? " + " ".join(["step"] * 40),
                     "reply": "VERDICT: unsupported\nDECIDING: It is closed on Sundays.\nESCALATE: no"}
    with pytest.raises(NR.BuildError, match="no Statement"):
        NR.build([v], count)
    v["statement"] = "The library opens on Sundays."
    t = NR.build([v], count)[0][0]["turns"][0]
    assert t["user"] == ("Material:\nThe library opens at 9am on weekdays.\nIt is closed on Sundays.\n"
                         "Statement: The library opens on Sundays.\nAnswer in the three lines.")
    assert t["reply"].startswith("VERDICT: unsupported") and "Sundays" in t["thinking"]
    v["turns"][0]["user"] += "\nStatement: an older wording."
    assert "Statement: The library opens on Sundays." in NR.build([v], count)[0][0]["turns"][0]["user"]
