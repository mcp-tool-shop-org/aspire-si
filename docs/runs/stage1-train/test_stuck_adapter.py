"""The frozen rule file reaches stuck.judge through the agreed mapping (the Dataset Specialist found that passing
load_rule() straight to judge raises KeyError)."""

import pytest

import stuck
import stuck_adapter


def test_the_frozen_file_passed_straight_to_judge_fails():
    with pytest.raises(KeyError):
        stuck.judge("x", "m", 4, 100, 10, 10, stuck.load_rule())


def test_the_mapped_frozen_rule_judges():
    rule = stuck_adapter.load_frozen()
    assert rule == {"p95_by_tier": {"1": 575, "2": 575, "3": 572, "4": 572, "5": 453, "6": 453, "7+": 575},
                    "reread_rate_p95": 8.09, "burst": {"window": 200, "k": 3}}
    long_burst = " ".join(f"w{i}" for i in range(700)) + " wait " + " ".join(f"x{i}" for i in range(50)) + \
        " wait " + " ".join(f"y{i}" for i in range(50)) + " wait done"
    v = stuck.judge(long_burst, "", 4, 100, 860, 900, rule)
    assert v["bursts"] == 1 and v["long_not_progressing"] and v["stuck"]
    assert not stuck.judge("short and plain", "", 4, 100, 3, 3, rule)["stuck"]
