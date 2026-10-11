"""The quiz sitting with a stand-in model: canned answers, so the asking, scoring, majority, ledger, receipts and
readout are tested without a GPU."""

import copy
from pathlib import Path

import pytest

import data
import quiz
import stuck_adapter
from ledger import Ledger

DATA = Path(__file__).resolve().parent.parent / "stage1-training-data"
ITEMS = [r for r in data.load(DATA / "batch3-t05.jsonl")]
ALL_VERDICT_E = [r for r in ITEMS if r["kind"] == "E" and r["quiz"]["type"] == "verdict"]
_TRAINED, _MATERIAL = quiz.trained_spans(ITEMS), quiz.material_spans(ITEMS)


def _spent(r):
    import receipts
    return bool((receipts.spans(r["variant"]["user"]) - _MATERIAL) & _TRAINED)


VERDICT_E = [r for r in ALL_VERDICT_E if not _spent(r)]  # own variants never trained on
WRITTEN_E = [r for r in ITEMS if r["kind"] == "E" and r["quiz"]["type"] == "written"]
RULE = stuck_adapter.load_frozen()


def count(text):
    return len(text.split())


def three_lines(q):
    return (f"VERDICT: {q['verdict']}\nDECIDING: {' | '.join(q['deciding']) or 'NONE'}\n"
            f"ESCALATE: {q['escalate']}")


def model(right_by_seed):
    """A stand-in for harness.generate: the base item's keyed answer when right_by_seed[seed], else a wrong one."""
    calls = []

    def generate(messages, seed):
        calls.append(messages[0]["content"])
        item = next(r for r in VERDICT_E + CLOSED if r["_marker"] in messages[0]["content"])
        if item["quiz"]["type"] == "verdict":
            q = item["quiz"]
            wrong = {**q, "verdict": "supported" if q["verdict"] != "supported" else "unsupported"}
            answer = three_lines(q if right_by_seed[seed] else wrong)
        else:
            key = item["variant"]["key"]["answer"]
            answer = f"ANSWER: {key if right_by_seed[seed] else ('A' if key != 'A' else 'B')}"
        return {"text": f"<think>I read it once.</think>{answer}", "new_tokens": 30, "truncated": False,
                "seconds": 1.0, "prompt_tokens": 100}
    generate.calls = calls
    return generate


CLOSED = []
for r in VERDICT_E:
    r["_marker"] = r["variant"]["user"]
for i, r in enumerate(WRITTEN_E[:1]):
    c = copy.deepcopy(r)
    c["id"] = c["id"] + "-closed"
    c["variant"] = {"user": f"Which is the habit (item {i})?\nA. Count the approvals.\nB. {r['quiz']['key_points'][0]}.\n"
                            "C. Ask the senior reader.\nGive the letter as ANSWER: <letter>.", "key": {"answer": "B"}}
    c["_marker"] = c["variant"]["user"]
    CLOSED.append(c)


ITEMS_FOR_SPANS = [r for r in ITEMS if r["kind"] != "E"]  # the trained turns the spent check reads


def sit(items, right_by_seed, ledger=None, session=1, store=None):
    rows = []
    gen = model(right_by_seed)
    out = quiz.sit(ITEMS_FOR_SPANS + items, ledger or Ledger(), session, store or {}, gen, count, RULE,
                   rows.append, only=[r["id"] for r in items])
    return out, rows, gen


def test_the_first_ask_is_the_items_own_variant_with_its_material_never_the_trained_turn():
    out, rows, gen = sit(VERDICT_E[:1], {0: True, 1: True, 2: True})
    r = VERDICT_E[0]
    asked = gen.calls[0]
    assert asked.endswith(r["variant"]["user"].strip()) and r["material"] in asked
    assert asked != r["turns"][0]["user"]
    assert {x["variant"] for x in rows} == {0} and len(rows) == 3  # three samples, one row each


def test_majority_of_three_decides_and_every_sample_is_a_row():
    out, rows, _ = sit(VERDICT_E[:1], {0: True, 1: False, 2: True})
    assert [x["correct_pinned"] for x in rows] == [True, False, True] and all(x["majority_right"] for x in rows)
    out, rows, _ = sit(VERDICT_E[:1], {0: True, 1: False, 2: False})
    assert not rows[0]["majority_right"] and rows[0]["ledger_state"]["box"] == 1


def test_a_verdict_variant_passes_only_when_all_three_lines_pass():
    r = VERDICT_E[0]
    q = r["quiz"]
    good = quiz.score_sample(r, "verdict", r["variant"], three_lines(q))
    assert good["right"] and good["verdict_only"] and good["deciding_ok"] and good["escalate_ok"]
    flipped = {"yes": "no", "no": "yes"}[q["escalate"].split()[0].lower()]
    bad_esc = quiz.score_sample(r, "verdict", r["variant"], three_lines({**q, "escalate": flipped}))
    assert not bad_esc["right"] and bad_esc["verdict_only"]  # verdict right, ESCALATE wrong: not a pass


def test_closed_form_is_scored_by_the_letter_and_free_text_written_items_stay_out():
    out, rows, _ = sit(CLOSED + WRITTEN_E, {0: True, 1: True, 2: False})
    assert {x["quiz_type"] for x in rows} == {"closed"}
    assert all(x["choice_key"] == "B" for x in rows) and rows[0]["majority_right"]
    assert out["per_lesson"]["T05"]["written_excluded_not_yet_testable"] == len(WRITTEN_E)


def test_a_miss_returns_next_session_with_a_fresh_variant_from_the_store():
    r = VERDICT_E[0]
    lg = Ledger()
    store = {r["id"]: {1: {"user": r["variant"]["user"] + " (asked again)", "key": {"verdict": r["quiz"]["verdict"]}}}}
    sit([r], {0: False, 1: False, 2: True}, ledger=lg, session=1, store=store)
    out, rows, _ = sit([r], {0: True, 1: True, 2: True}, ledger=lg, session=2, store=store)
    assert {x["variant"] for x in rows} == {1} and rows[0]["attempt"] == 2
    assert out["per_lesson"]["T05"]["n_ge_1"] == {"asked": 1, "right": 1, "accuracy": 1.0}
    out, rows, _ = sit([r], {0: True, 1: True, 2: True}, ledger=lg, session=5, store=store)
    assert rows == [] and lg.items[r["id"]]["status"].startswith("unmastered")  # family used up


def test_an_item_whose_material_is_not_in_its_turn_is_refused_not_approximated():
    r = copy.deepcopy(VERDICT_E[0])
    r["material"] = "A material line that is not in the turn."
    r["_marker"] = r["variant"]["user"]
    out, rows, _ = sit([r], {0: True, 1: True, 2: True})
    assert rows == [] and out["refused_items"] and "verbatim" in out["refused_items"][0]


def test_readout_reports_n0_beside_n_ge_1_and_star_off():
    out, rows, _ = sit(VERDICT_E, {0: True, 1: True, 2: True})
    o = out["overall"]
    assert o["n0"]["asked"] == len(VERDICT_E) and o["n_ge_1"]["asked"] == 0 and out["star"].startswith("off")
    assert set(rows[0]) >= {"question_id", "base_item_id", "variant", "lesson", "tier", "attempt", "ledger_box",
                            "ledger_state", "stuck", "hedges", "max_in_window", "bursts", "gap_unit", "trace"}


def test_a_variant_that_was_trained_on_is_spent_not_asked():
    """A variant that repeats a sentence of its item's trained turn can't be a fresh first ask. The case is built
    here, not read from the data: the data's spent variants are reworded in fix rounds."""
    r = copy.deepcopy(VERDICT_E[0])
    r["id"] = "T05-E-99"
    import receipts
    words, own_material = r["turns"][0]["thinking"].split(), quiz.material_spans([r])
    # a 10-word window of the trained thinking that quotes none of the item's material (which is exempt)
    window = next(" ".join(words[i:i + 10]) for i in range(len(words) - 9)
                  if not receipts.spans(" ".join(words[i:i + 10])) & own_material)
    r["variant"] = {"user": window + " What do you answer?", "key": r["variant"]["key"]}
    r["_marker"] = r["variant"]["user"]
    lg = Ledger()
    out, rows, _ = sit([r], {0: True, 1: True, 2: True}, ledger=lg)
    assert rows == [] and out["spent_variants"] == [
        {"base": "T05-E-99", "n": 0, "reason": "shares an 8-word span with a trained turn"}]
    assert lg.items["T05-E-99"]["status"].startswith("unmastered") and lg.items["T05-E-99"]["spent"]


def test_a_free_text_store_variant_is_spent_and_the_next_closed_one_is_asked():
    """R&D's fix on #102: {0 closed, 1 free text, 2 closed} asks 0, then 2; the free-text one never blocks."""
    c = CLOSED[0]
    c["_marker"] = "A. Count the approvals."  # in both closed-form variants, so the stand-in finds the item
    other = c["variant"]["user"].replace("Which is the habit", "Pick the habit")
    store = {c["id"]: {1: {"user": "Say in your own words what the habit is.", "key": ["read the page first"]},
                       2: {"user": other, "key": {"answer": "B"}}}}
    lg = Ledger()
    sit([c], {0: False, 1: False, 2: False}, ledger=lg, session=1, store=store)
    out, rows, _ = sit([c], {0: True, 1: True, 2: True}, ledger=lg, session=2, store=store)
    assert {x["variant"] for x in rows} == {2}
    assert out["spent_variants"] == [{"base": c["id"], "n": 1, "reason": "free text, not yet testable"}]
    assert [h["variant"] for h in lg.items[c["id"]]["history"]] == [0, 2]


def keyed_repeats_trained(c, stem_repeats=False):
    """A closed-form item whose keyed option (and, if asked, its stem) repeats 10 words of its trained reply,
    with those words as one of its key points."""
    c = copy.deepcopy(c)
    words = c["turns"][0]["reply"].split()
    assert len(words) >= 10
    phrase = " ".join(words[:10]).rstrip(".,;:")
    c["quiz"]["key_points"] = [phrase] + c["quiz"]["key_points"]
    stem = (" ".join(words[:10]) + "? ") if stem_repeats else "Which is the habit? "
    c["variant"] = {"user": stem + "\nA. Count the approvals.\nB. " + phrase + ".\nC. Ask the senior reader.\n"
                            "Give the letter as ANSWER: <letter>.", "key": {"answer": "B"}}
    c["_marker"] = "A. Count the approvals."
    return c


def test_by_default_a_keyed_option_that_repeats_a_trained_key_point_is_spent():
    c = keyed_repeats_trained(CLOSED[0])
    out, rows, _ = sit([c], {0: True, 1: True, 2: True})
    assert rows == [] and out["spent_variants"][0]["reason"] == "shares an 8-word span with a trained turn"
    assert out["spent_rule"] == {"span": 8, "exempt_keyed_key_points": False}


def test_the_keyed_key_point_exemption_is_a_setting_and_covers_only_the_keyed_option():
    c = keyed_repeats_trained(CLOSED[0])
    rows = []
    gen = model({0: True, 1: True, 2: True})
    out = quiz.sit(ITEMS_FOR_SPANS + [c], Ledger(), 1, {}, gen, count, RULE, rows.append, only=[c["id"]],
                   spent_rule={"exempt_keyed_key_points": True})
    assert {x["variant"] for x in rows} == {0} and rows[0]["spent_rule"]["exempt_keyed_key_points"] is True
    stem = keyed_repeats_trained(CLOSED[0], stem_repeats=True)  # the stem repeats trained words: still spent
    out = quiz.sit(ITEMS_FOR_SPANS + [stem], Ledger(), 1, {}, gen, count, RULE, [].append, only=[stem["id"]],
                   spent_rule={"exempt_keyed_key_points": True})
    assert out["spent_variants"] and out["spent_variants"][0]["n"] == 0


def test_the_span_is_a_setting():
    c = keyed_repeats_trained(CLOSED[0])  # 10 repeated words: spent at 8, not at 11
    out, rows, _ = sit([c], {0: True, 1: True, 2: True})
    assert rows == []
    rows = []
    out = quiz.sit(ITEMS_FOR_SPANS + [c], Ledger(), 1, {}, model({0: True, 1: True, 2: True}), count, RULE,
                   rows.append, only=[c["id"]], spent_rule={"span": 11})
    assert rows and out["spent_rule"]["span"] == 11
