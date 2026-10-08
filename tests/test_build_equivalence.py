"""Tests for the Ollama build equivalence readout."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "sft-experiment"))

import build_equivalence as be  # noqa: E402


def runs(grammar_flips: int, meaning_flips: int, repeat_flips: int = 0):
    base_g = {f"g{i}": {"outcome": "ok", "grammatical": True} for i in range(30)}
    base_m = {f"m{i}": {"outcome": "same", "verdict": "same", "claim_changed": False} for i in range(30)}

    def flip(d, k, task):
        out = dict(d)
        for i, key in enumerate(sorted(d)[:k]):
            out[key] = (
                {"outcome": "ok", "grammatical": False}
                if task == "grammar"
                else {"outcome": "changed", "verdict": "changed", "claim_changed": True}
            )
        return out

    return {
        "13.0-a": {"grammar": base_g, "meaning": base_m},
        "13.0-b": {
            "grammar": flip(base_g, repeat_flips, "grammar"),
            "meaning": flip(base_m, repeat_flips, "meaning"),
        },
        "13.4": {
            "grammar": flip(base_g, grammar_flips, "grammar"),
            "meaning": flip(base_m, meaning_flips, "meaning"),
        },
    }


def test_identical_builds_do_not_stop():
    r = be.readout(runs(0, 0))
    assert r["grammar"]["cross_build"]["rate"] == 1.0 and not r["any_stop"]


def test_a_real_build_difference_stops():
    r = be.readout(runs(6, 0))  # 80% cross-build against a 100% repeat
    assert r["grammar"]["stop"] and r["any_stop"] and len(r["grammar"]["cross_build"]["disagreements"]) == 6


def test_a_shortfall_within_the_builds_own_noise_does_not_stop():
    r = be.readout(runs(5, 5, repeat_flips=4))  # cross 83% against a repeat of 87%: within noise
    assert not r["grammar"]["stop"] and not r["meaning"]["stop"]


def test_unusable_items_are_left_out_and_wilson_is_sane():
    a = {"x": {"outcome": "truncated"}, "y": {"outcome": "ok", "grammatical": True}}
    b = {"x": {"outcome": "ok", "grammatical": True}, "y": {"outcome": "ok", "grammatical": True}}
    r = be.agreement("grammar", a, b)
    assert r["usable"] == 1 and r["rate"] == 1.0
    lo, hi = be.wilson(20, 24)
    assert lo == pytest.approx(0.64, abs=0.01) and hi == pytest.approx(0.93, abs=0.01)


def test_samples_are_seeded_and_order_free():
    ids = [f"i{k}" for k in range(100)]
    assert be.sample_ids(ids) == be.sample_ids(list(reversed(ids))) and len(be.sample_ids(ids)) == 30
