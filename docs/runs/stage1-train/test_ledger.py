"""The Leitner ledger's rules, one test each."""

import pytest

from ledger import EXHAUSTED, Ledger


def fresh(n_variants=10):
    lg = Ledger()
    lg.enrol("T05-E-05", "T05", session=1)
    return lg, list(range(n_variants))


def ask(lg, avail, session, right, item="T05-E-05"):
    v = lg.next_variant(item, avail)
    return lg.record(item, session, v, right)


def test_right_answers_climb_the_boxes_with_doubling_intervals():
    lg, avail = fresh()
    s = ask(lg, avail, 1, True)
    assert (s["box"], s["due_session"]) == (2, 3)
    s = ask(lg, avail, 3, True)
    assert (s["box"], s["due_session"]) == (3, 7)
    s = ask(lg, avail, 7, True)
    assert (s["box"], s["due_session"]) == (4, 15)
    s = ask(lg, avail, 15, True)
    assert (s["box"], s["due_session"]) == (4, 23)


def test_a_wrong_answer_returns_to_box_1_next_session_and_counts_a_forgetting_event():
    lg, avail = fresh()
    ask(lg, avail, 1, True)
    s = ask(lg, avail, 3, False)
    assert (s["box"], s["due_session"], s["forgetting"], s["misses"]) == (1, 4, 1, 1)
    s = ask(lg, avail, 4, False)
    assert s["forgetting"] == 1  # wrong after wrong is not a forgetting event


def test_mastery_needs_a_three_streak_over_two_boxes_then_a_right_twin():
    lg, avail = fresh()
    for session in (1, 3, 7):
        assert not ask(lg, avail, session, True)["mastered"]
    s = ask(lg, avail, 15, True)  # the fresh twin
    assert s["mastered"]
    assert lg.items["T05-E-05"]["history"][-1]["twin"] is True
    assert lg.due(100) == []  # mastered items are no longer due


def test_a_wrong_twin_breaks_the_streak():
    lg, avail = fresh()
    for session in (1, 3, 7):
        ask(lg, avail, session, True)
    s = ask(lg, avail, 15, False)
    assert not s["mastered"] and s["box"] == 1 and s["streak"] == 0


def test_variants_are_never_repeated_and_an_exhausted_family_is_closed():
    lg, avail = fresh(n_variants=2)
    ask(lg, avail, 1, False)
    ask(lg, avail, 2, False)
    assert lg.next_variant("T05-E-05", avail) is None
    assert lg.items["T05-E-05"]["status"] == EXHAUSTED and lg.due(3) == []
    with pytest.raises(ValueError):
        lg.record("T05-E-05", 3, 0, True)


def test_variants_go_in_order_from_the_items_own():
    lg, avail = fresh()
    assert lg.next_variant("T05-E-05", [2, 0, 1]) == 0
    lg.record("T05-E-05", 1, 0, False)
    assert lg.next_variant("T05-E-05", [2, 0, 1]) == 1


def test_quality_mastery_needs_95_percent_of_items_and_90_percent_on_twins():
    lg = Ledger()
    for i in range(20):
        lg.enrol(f"T05-E-{i:02d}", "T05", 1)
    avail = list(range(10))
    for i in range(20):
        item = f"T05-E-{i:02d}"
        for session in (1, 3, 7):
            ask(lg, avail, session, True, item)
        ask(lg, avail, 15, i != 0, item)  # one twin wrong: 19 of 20 mastered (95%), twins 95%
    q = lg.qualities()["T05"]
    assert q["share_mastered"] == 0.95 and q["twin_accuracy"] == 0.95 and q["quality_mastered"]
    lg2 = Ledger({k: dict(v) for k, v in lg.items.items()})
    lg2.items["T05-E-01"]["mastered"] = False
    assert not lg2.qualities()["T05"]["quality_mastered"]  # 90% of items: below the line


def test_save_and_load_round_trip(tmp_path):
    lg, avail = fresh()
    ask(lg, avail, 1, True)
    sha = lg.save(tmp_path / "ledger.json")
    again = Ledger.load(tmp_path / "ledger.json")
    assert again.items == lg.items and len(sha) == 64
    assert again.dumps() == lg.dumps()
