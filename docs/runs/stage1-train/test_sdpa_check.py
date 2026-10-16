"""Argon's finding 2: the SDPA check's comparison and its decision rule (the card run itself waits for Argon's
re-review)."""

import json
from pathlib import Path

import sdpa_check

A0 = {"turns": [{"text": "<think>a</think>VERDICT: unsupported", "new_tokens": 40}]}
SAME = {"turns": [{"text": "<think>a</think>VERDICT: unsupported", "new_tokens": 40, "seconds": 9.9}]}
OTHER = {"turns": [{"text": "<think>b</think>VERDICT: supported", "new_tokens": 41}]}


def test_flash_is_pinned_when_it_is_deterministic_and_matches_a0():
    cmp = sdpa_check.compare({"default": [SAME, SAME], "flash": [SAME, SAME], "flash+efficient": [SAME, SAME]}, A0)
    assert cmp["flash"] == {"ran": True, "repeat_equal": True, "equals_a0": True, "error": None}
    assert sdpa_check.decide(cmp) == "flash"


def test_the_pair_when_flash_alone_raises_or_differs():
    raised = sdpa_check.compare({"default": [SAME, SAME], "flash": [{"error": "No available kernel"}] * 2,
                                 "flash+efficient": [SAME, SAME]}, A0)
    assert raised["flash"]["error"] == "No available kernel" and sdpa_check.decide(raised) == "flash+efficient"
    differs = sdpa_check.compare({"default": [SAME, SAME], "flash": [OTHER, OTHER],
                                  "flash+efficient": [SAME, SAME]}, A0)
    assert not differs["flash"]["equals_a0"] and sdpa_check.decide(differs) == "flash+efficient"


def test_no_pin_settled_when_nothing_is_deterministic_and_matches_a0():
    cmp = sdpa_check.compare({"default": [SAME, OTHER], "flash": [SAME, OTHER], "flash+efficient": [OTHER, OTHER]},
                             A0)
    assert not cmp["flash"]["repeat_equal"] and sdpa_check.decide(cmp).startswith("none")


def test_the_check_task_is_a_real_dev_task():
    here = Path(__file__).resolve().parent.parent / "stage1-taskset"
    ids = {json.loads(x)["id"] for f in sdpa_check.TASK_FILES
           for x in (here / f).read_text(encoding="utf-8").splitlines() if x.strip()}
    assert "rv-bc9c8c5439" in ids
