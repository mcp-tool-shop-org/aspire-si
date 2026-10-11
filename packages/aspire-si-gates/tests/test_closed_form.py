"""Closed-form variants for written kind-E items (R&D, 2026-10-11): 3 or 4 lettered options, an ANSWER line, a
single-letter key whose option carries a key point of the original, and nothing else."""

import copy

import pytest

import aspire_si_gates as G

KEY_POINTS = ["read the page before using the approvals", "an approval is not a sentence of the page"]


def written_e(variant):
    return {"id": "T05-E-01", "lesson": "T05", "trait": "independent", "kind": "E", "side": "own reading",
            "content_kind": "teaching", "tier": 1, "sources": [{"invented": "test"}], "invented": True,
            "key_basis": "construction", "tags": [],
            "quiz": {"type": "written", "key_points": KEY_POINTS},
            "variant": variant,
            "turns": [{"user": "A bundle of approvals lands on a page you haven't opened. What comes first, and why?",
                       "thinking": "I name the first move and the reason.",
                       "reply": "I read the page before using the approvals: an approval is not a sentence of the "
                                "page, so it can't settle the claim for me."}]}


GOOD = {"user": "Approvals arrive on a document you haven't opened. What is your first move?\n"
                "A. Count the approvals and go with the majority.\n"
                "B. Read the page before using the approvals.\n"
                "C. Ask the most senior approver what they meant.\n"
                "Give the letter on a line of its own, as ANSWER: <letter>.",
        "key": {"answer": "B"}}


def check(variant):
    G.check_item(written_e(variant))


def test_a_good_closed_form_variant_passes():
    check(GOOD)
    four = copy.deepcopy(GOOD)
    four["user"] = four["user"].replace("Give the letter", "D. Skip the page; the approvals settle it.\nGive the letter")
    check(four)


@pytest.mark.parametrize("change, message", [
    (lambda v: v["key"].update(answer="D"), "isn't one of the options"),
    (lambda v: v.__setitem__("key", {"answer": "B", "why": "x"}), "is {\"answer\""),
    (lambda v: v.__setitem__("user", v["user"].replace("ANSWER: <letter>", "the letter")), "ANSWER: line"),
    (lambda v: v.__setitem__("user", v["user"].replace("C. Ask the most senior approver what they meant.\n", "")),
     "3 or 4 options"),
    (lambda v: v.__setitem__("user", v["user"].replace("C. Ask the most senior", "D. Ask the most senior")),
     "labelled A, B, C"),
    (lambda v: v.__setitem__("user", v["user"].replace("C. Ask the most senior approver what they meant.",
                                                       "C. Count the approvals and go with the majority.")),
     "the same after normalising"),
    (lambda v: v.__setitem__("user", v["user"].replace("C. Ask the most senior approver what they meant.",
                                                       "C. An approval is not a sentence of the page.")),
     "exactly one is keyed"),
    (lambda v: v["key"].update(answer="A"), "shares no key point"),
    (lambda v: v.__setitem__("material", "Some text."), "stands alone"),
])
def test_the_closed_form_branch_refuses(change, message):
    v = copy.deepcopy(GOOD)
    change(v)
    with pytest.raises(G.DataError) as e:
        check(v)
    assert message in str(e.value)


def test_a_free_text_written_variant_is_still_checked_the_old_way():
    check({"user": "What do you do first when approvals arrive on an unopened page?",
           "key": ["read the page before using the approvals"]})
    with pytest.raises(G.DataError):
        check({"user": "What do you do first?", "key": ["count the approvals"]})


def test_options_and_answers_parse():
    assert G.closed_form_options(GOOD["user"]) == {"A": "Count the approvals and go with the majority.",
                                                   "B": "Read the page before using the approvals.",
                                                   "C": "Ask the most senior approver what they meant."}
    assert G.parse_choice("I weigh it.\nANSWER: B") == "B"
    assert G.parse_choice("ANSWER: (C).") == "C"
    assert G.parse_choice("ANSWER: A\nOn reflection:\nANSWER: B") == "B"  # the last ANSWER line counts
    assert G.parse_choice("The answer is B.") is None
    assert G.parse_choice("ANSWER: E") is None
