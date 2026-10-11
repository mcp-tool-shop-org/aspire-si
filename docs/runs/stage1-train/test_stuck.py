"""Tests for the stuck rule (stuck.py): R&D's four cases, plus the cap."""

import stuck

RULE = {"p95_by_tier": {"4": 100, "7+": 100}, "reread_rate_p95": 9.28, "burst": {"window": 200, "k": 3}}
MATERIAL = "\n".join(f"Line {i}: the shuttle {w} leaves the north gate at {i} o'clock sharp."
                     for i, w in enumerate(["red", "blue", "green", "white", "grey", "gold"], 1))


def run(thinking, target=40, total=None, rule=RULE, tier=4):
    n = len(thinking.split())
    return stuck.judge(thinking, MATERIAL, tier, target, n, total or n, rule)


def filler(k, tag="s"):
    return " ".join(f"{tag}{i}" for i in range(k))


def test_a_looping_trace_triggers_a():
    loop = "I keep checking whether the shuttle really leaves at that time. " * 4
    v = run(loop + filler(10))
    assert v["loop_span"] and v["stuck"]


def test_a_long_role_trace_that_keeps_quoting_new_lines_does_not_trigger():
    parts = []
    for i, w in enumerate(["red", "blue", "green", "white", "grey", "gold"], 1):
        parts.append(f"{filler(25, w)} the shuttle {w} leaves the north gate at {i} o'clock sharp")
    v = run(" ".join(parts), target=40)
    assert v["over_p95"] and v["lines_quoted"] == 6 and v["rereads"] == 0
    assert not v["stuck"]


def rereads_trace(k):
    """A trace quoting line 1 k + 1 times, each quote separated by more than 5 words: k re-reads."""
    return " ".join(f"{filler(8, f'r{j}x')} the shuttle red leaves the north gate" for j in range(k + 1))


def test_a_long_trace_with_a_high_reread_rate_triggers():
    v = run(rereads_trace(6) + " " + filler(30), target=20)
    assert v["rereads"] == 6 and v["over_p95"] and v["reread_rate"] > RULE["reread_rate_p95"]
    assert v["long_not_progressing"] and v["stuck"]


def test_the_reread_boundary():
    trace = rereads_trace(29)  # 29 re-reads: at 3,125 thinking tokens the rate is exactly 9.28
    at = stuck.judge(trace, MATERIAL, 4, 20, 3125, 3125, RULE)
    above = stuck.judge(trace, MATERIAL, 4, 20, 3124, 3124, RULE)
    assert at["rereads"] == 29 and at["reread_rate"] == 9.28 and not at["stuck"]
    assert above["stuck"] and above["long_not_progressing"]


def test_the_cap_triggers_b_and_length_alone_is_only_recorded():
    v = run(filler(50), total=16384)
    assert v["cap"] and v["stuck"]
    v = run(filler(300), target=10)
    assert v["over_p95"] and v["over_2x_target"]  # recorded
    assert v["stuck"] == v["long_not_progressing"]  # length alone triggers only through (C)'s no-progress test


# ---- the hedge marker (dated amendment): a fixed synthetic rule, as with re-reads
def hedge_trace(k):
    """k hedges ("wait"), each in its own sentence among unique filler words."""
    return " ".join(f"{filler(6, f'h{j}x')}. Wait, {filler(6, f'g{j}x')}." for j in range(k))


def test_a_self_check_sentence_with_wait_is_not_counted():
    assert stuck.hedges("If the line is missing, then I wait and read the notice again. " + filler(10)) == 0
    assert stuck.hedges("Hmm. " + filler(10)) == 1


def test_a_quoted_material_line_with_actually_is_not_counted():
    mat = "The fee is actually waived for members who renew early."
    assert stuck.hedges("The notice says: the fee is actually waived for members who renew early.", mat) == 0
    assert stuck.hedges("Actually, I think the fee is charged.", mat) == 1


def spaced(gaps, tail=50, tag="z"):
    """Hedges ("wait") separated by the given numbers of filler words (a word stands in for a token)."""
    parts = []
    for n, g in enumerate(gaps):
        parts.append(filler(g, f"{tag}{n}x"))
        parts.append("wait")
    return " ".join(parts + [filler(tail, f"{tag}end")])


def test_three_hedges_within_200_tokens_on_a_long_trace_trigger():
    t = spaced([300, 60, 60])  # hedges at words 300, 361, 422
    v = run(t, target=20)
    assert v["bursts"] == 1 and v["over_p95"] and v["stuck"] and v["long_not_progressing"]


def test_the_same_three_hedges_spread_over_600_tokens_do_not():
    v = run(spaced([100, 300, 300]), target=20)
    assert v["hedges"] == 3 and v["bursts"] == 0 and not v["stuck"]


def test_a_short_trace_with_a_burst_does_not_trigger():
    v = run(spaced([5, 5, 5], tail=5), target=200)
    assert v["bursts"] == 1 and not v["over_p95"] and not v["stuck"]


def test_hedges_in_material_or_a_self_check_do_not_count_toward_a_burst():
    mat = "Wait staff must actually check every wait list before service."
    t = (filler(300, "m") + " the notice: wait staff must actually check every wait list before service. "
         + "If the list is short, then I wait for the next line. " + filler(10, "n") + " wait " + filler(60, "o"))
    v = stuck.judge(t, mat, 4, 20, len(t.split()), len(t.split()), RULE)
    assert v["hedges"] == 1 and v["bursts"] == 0 and not v["stuck"]


def test_the_burst_window_boundary():
    inside = stuck.burst_profile([0, 100, 199], 1000, 200, 3)
    edge = stuck.burst_profile([0, 100, 200], 1000, 200, 3)
    assert inside["bursts"] == [[0, 199]] and inside["max_in_window"] == 3
    assert edge["bursts"] == [] and edge["max_in_window"] == 2  # exactly 200 apart is not within 200


def test_hedge_profile_records_phrases_and_positions():
    t = "Wait. Hmm, the line. " + filler(30, "p") + " wait."
    prof = stuck.hedge_profile(t, "", 100)
    assert prof["per_phrase"] == {"hmm": 1, "wait": 2} and prof["hedge_rate"] == 30.0
    assert prof["first"] == 0.0 and prof["last"] > 0.9 and prof["end_clustering"] == round(1 / 3, 3)
    words = lambda x: len(x.split())  # noqa: E731  (a word count stands in for the tokenizer)
    t2 = filler(70, "a") + " wait " + filler(19, "b") + " hmm " + filler(8, "c")  # hedges at words 70 and 90 of 100
    prof = stuck.hedge_profile(t2, "", 100, count=words)
    assert prof["first"] == 0.7 and prof["last"] == 0.9 and prof["end_clustering"] == 1.0
    one = stuck.hedge_profile("Wait. " + filler(20, "d"), "", 21, count=words)
    assert one["hedges"] == 1 and one["end_clustering"] is None  # fewer than 2 hedges
    assert stuck.hedge_profile("No hedges here at all.", "", 10)["first"] is None
