"""Closed-form variants for written kind-E items: 3 or 4 lettered options, an ANSWER line, a single-letter key,
and exactly one option carrying a key point of the original, under the carry setting (verbatim | paraphrase |
either; the default is paraphrase, R&D's ruling (b))."""

import copy

import pytest

import aspire_si_gates as G
from aspire_si_gates import gates

KEY_POINTS = ["read the page before using the approvals", "an approval is not a sentence of the page"]
VERBATIM = {"carry": "verbatim"}


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

PARAPHRASED = {"user": "Approvals arrive on a document you haven't opened. What is your first move?\n"
                       "A. Count the approvals and go with the majority.\n"
                       "B. First open and read the page; only then weigh the approvals.\n"
                       "C. Ask the most senior approver what they meant.\n"
                       "Give the letter on a line of its own, as ANSWER: <letter>.",
               "key": {"answer": "B"}}


def check(variant, cfg=None):
    G.check_item(written_e(variant), closed_form=cfg)


# ---------------------------------------------------------------- the shape (any carry)

def test_a_good_closed_form_variant_passes_verbatim():
    check(GOOD, VERBATIM)
    four = copy.deepcopy(GOOD)
    four["user"] = four["user"].replace("Give the letter", "D. Skip the page; the approvals settle it.\nGive the letter")
    check(four, VERBATIM)


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
    (lambda v: v["key"].update(answer="A"), "doesn't carry a key point"),
    (lambda v: v.__setitem__("material", "Some text."), "stands alone"),
])
def test_the_closed_form_branch_refuses_verbatim(change, message):
    v = copy.deepcopy(GOOD)
    change(v)
    with pytest.raises(G.DataError) as e:
        check(v, VERBATIM)
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


# ---------------------------------------------------------------- the settings

def test_the_defaults_are_r_and_ds_ruling():
    assert G.CLOSED_FORM_DEFAULT == {"carry": "paraphrase", "share": 0.5, "distractor_max": 0.8,
                                     "distractor_margin": 0.15, "span": 8, "exempt_keyed_key_points": False}
    for bad in ({"carry": "loose"}, {"share": 0}, {"share": 1.5}, {"distractor_max": 0}, {"span": 1},
                {"distractor_margin": -0.1}, {"distractor_margin": 1}):
        with pytest.raises(ValueError):
            G.closed_form_settings(bad)


def test_content_word_share_drops_function_and_task_words_and_light_suffixes():
    assert G.key_point_share("read the page before using the approvals",
                             "First open and read the page; only then weigh the approvals.") == 0.75
    assert G.key_point_share("the claim", "anything") == 0.0  # no content words: no share


def test_paraphrase_passes_a_reworded_keyed_option():
    check(PARAPHRASED)  # the default carry
    check(PARAPHRASED, {"carry": "either"})


def test_paraphrase_refuses_a_keyed_option_under_the_share_and_takes_the_share_as_a_setting():
    with pytest.raises(G.DataError, match="content-word share 0.75, needed 0.8"):
        check(PARAPHRASED, {"share": 0.8})
    check(PARAPHRASED, {"share": 0.75})


def test_a_near_miss_distractor_passes_under_the_keyed_share_and_the_margin_is_a_setting():
    near = copy.deepcopy(PARAPHRASED)  # keyed share 0.75; this distractor 0.67 (2 of 3 of key point 2)
    near["user"] = near["user"].replace("C. Ask the most senior approver what they meant.",
                                        "C. Skip the page; the approvals settle it.")
    with pytest.raises(G.DataError, match="minus 0.15"):
        check(near)  # the default margin (0.15): 0.67 is not under 0.75 - 0.15
    check(near, {"distractor_margin": 0.0})  # strictly below the keyed share is enough at margin 0
    with pytest.raises(G.DataError, match="exactly one is keyed"):
        check(near, {"distractor_margin": 0.0, "distractor_max": 0.6})  # the near-copy cap, set low


def test_a_distractor_as_close_as_the_keyed_option_is_refused():
    inverted = copy.deepcopy(PARAPHRASED)  # a distractor at the keyed option's share (0.75), under the 0.8 cap
    inverted["user"] = inverted["user"].replace("C. Ask the most senior approver what they meant.",
                                                "C. Read the page and count approvals.")
    with pytest.raises(G.DataError, match="exactly one is keyed"):
        check(inverted)


def test_a_near_copy_distractor_is_refused():
    copy_ = copy.deepcopy(PARAPHRASED)  # a distractor holding every content word of key point 1 (1.0)
    copy_["user"] = copy_["user"].replace("C. Ask the most senior approver what they meant.",
                                          "C. Read the page while using approvals for speed.")
    with pytest.raises(G.DataError, match="exactly one is keyed"):
        check(copy_)


def test_the_figures_report_each_items_keyed_and_top_distractor_share():
    near = copy.deepcopy(PARAPHRASED)
    near["user"] = near["user"].replace("C. Ask the most senior approver what they meant.",
                                        "C. Skip the page; the approvals settle it.")
    assert G.closed_form_figures([written_e(near)]) == [
        {"id": "T05-E-01", "keyed_share": 0.75, "top_distractor_share": 0.667}]


def test_paraphrase_refuses_an_option_that_repeats_its_trained_turn_and_takes_the_span_as_a_setting():
    copied = copy.deepcopy(PARAPHRASED)
    copied["user"] = copied["user"].replace("B. First open and read the page; only then weigh the approvals.",
                                            "B. I read the page before using the approvals: an approval is not.")
    with pytest.raises(G.DataError, match="8-word span with the item's trained turn"):
        check(copied)
    check(copied, VERBATIM)  # verbatim carry has no span rule
    check(copied, {"carry": "either"})  # a keyed option that passes verbatim is exempt in either
    long_copy = copy.deepcopy(copied)  # a distractor that repeats 10 trained words, under either with span 10
    long_copy["user"] = long_copy["user"].replace("C. Ask the most senior approver what they meant.",
                                                  "C. The page, so it can't settle the claim for me.")
    with pytest.raises(G.DataError, match="10-word span"):
        check(long_copy, {"carry": "either", "span": 10})
    check(long_copy, {"carry": "either", "span": 11})


def test_the_span_rule_also_holds_across_items_in_a_batch():
    other = written_e(PARAPHRASED)
    other["id"], other["quiz"] = "T05-E-02", {"type": "written", "key_points": ["weigh the approvals last"]}
    other["turns"] = [{"user": "Q?", "thinking": "t.", "reply": "First open and read the page; only then weigh "
                                                                  "the approvals, and say so."}]
    item = written_e(PARAPHRASED)
    with pytest.raises(G.DataError, match="T05-E-0[12]: its variant shares a 8-word span with a trained turn"):
        gates.check_closed_form_spans([item, other])


def test_a_copied_stem_is_refused_at_authoring():
    """R&D on #105: the quiz spends on the whole variant, so the gate checks the whole variant, stem included."""
    stem = copy.deepcopy(PARAPHRASED)
    stem["user"] = stem["user"].replace("Approvals arrive on a document you haven't opened. What is your first move?",
                                        "I read the page before using the approvals: an approval is not a sentence?")
    with pytest.raises(G.DataError, match="would be spent at its first quiz"):
        gates.check_closed_form_spans([written_e(stem)])
    gates.check_closed_form_spans([written_e(PARAPHRASED)])


def test_a_verdict_variant_restating_its_trained_statement_is_refused_and_material_is_exempt():
    material = "The library opens at 9am on weekdays and closes at six in the evening."
    trained_user = "Material:\n" + material + "\nStatement: the library opens at nine on weekday mornings for readers."
    r = {"id": "T05-E-03", "kind": "E", "lesson": "T05", "material": material,
         "quiz": {"type": "verdict", "verdict": "supported", "deciding": [material], "escalate": "no"},
         "turns": [{"user": trained_user, "thinking": "t.", "reply": "VERDICT: supported"}],
         "variant": {"user": "Claim: the library opens at nine on weekday mornings for readers. Is that backed?",
                     "key": {"verdict": "supported"}}}
    with pytest.raises(G.DataError, match="T05-E-03: its variant shares a 8-word span"):
        gates.check_closed_form_spans([r])
    r["variant"]["user"] = "Claim: weekday readers can get in at nine. Is that backed?"
    gates.check_closed_form_spans([r])
    r["variant"]["user"] = "Claim: The library opens at 9am on weekdays and closes at six. Backed?"  # material only
    gates.check_closed_form_spans([r])


def test_the_keyed_key_point_exemption_at_authoring_covers_the_keyed_option_only():
    repeat = copy.deepcopy(GOOD)  # the keyed option repeats the trained reply's key point (8+ words)
    repeat["user"] = repeat["user"].replace("B. Read the page before using the approvals.",
                                            "B. I read the page before using the approvals: an approval is not.")
    item = written_e(repeat)
    item["quiz"]["key_points"] = ["i read the page before using the approvals an approval is not"] + KEY_POINTS
    with pytest.raises(G.DataError, match="would be spent"):
        gates.check_closed_form_spans([item], VERBATIM)
    gates.check_closed_form_spans([item], {"carry": "verbatim", "exempt_keyed_key_points": True})
