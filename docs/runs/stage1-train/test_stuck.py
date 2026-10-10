"""Tests for the stuck rule (stuck.py): R&D's four cases, plus the cap."""

import stuck

RULE = {"p95_by_tier": {"4": 100, "7+": 100}, "reread_rate_p95": 9.28}
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
