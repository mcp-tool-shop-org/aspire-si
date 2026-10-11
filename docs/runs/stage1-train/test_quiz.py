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
SPENT_E = [r for r in ALL_VERDICT_E if _spent(r)]
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
    """T05-E-06's own variant repeats its trained statement word for word, so it can't be a fresh first ask."""
    assert [r["id"] for r in SPENT_E] == ["T05-E-06"]
    lg = Ledger()
    out, rows, _ = sit(SPENT_E, {0: True, 1: True, 2: True}, ledger=lg)
    assert rows == [] and out["spent_variants"] == [{"base": "T05-E-06", "n": 0}]
    assert lg.items["T05-E-06"]["status"].startswith("unmastered") and lg.items["T05-E-06"]["spent"]
