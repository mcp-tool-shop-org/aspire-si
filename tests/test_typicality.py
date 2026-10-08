"""Tests for the typicality match (addendum 3 of the Auditor plan)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "sft-experiment"))

import typicality as ty  # noqa: E402


class TestChangedSpan:
    def test_a_one_token_swap(self):
        assert ty.changed_span([1, 2, 3, 4], [1, 2, 9, 4]) == (2, 3, 3)

    def test_a_swap_of_different_lengths(self):
        assert ty.changed_span([1, 2, 3, 4], [1, 7, 8, 9, 4]) == (1, 3, 4)

    def test_suffix_does_not_overlap_the_prefix(self):
        assert ty.changed_span([1, 1, 1], [1, 1]) == (2, 3, 2)


class TestDelta:
    def test_delta_compares_only_the_changed_tokens(self):
        s_ids, s_lp = [1, 2, 3, 4], [0.0, -1.0, -0.5, -2.0]
        f_ids, f_lp = [1, 2, 9, 4], [0.0, -1.0, -3.0, -2.5]
        assert ty.delta(s_ids, s_lp, f_ids, f_lp) == pytest.approx(-2.5)

    def test_token_zero_has_no_log_probability(self):
        assert ty.span_logprob([-9.0, -1.0], 0, 2) == -1.0


class TestMatch:
    def test_a_paraphrase_median_inside_the_error_iqr_passes(self):
        errors = [-4.0, -3.0, -2.0, -1.0, 0.0]
        r = ty.match(errors, [-2.5, -2.0, -1.5])
        assert r["passes"] and r["error_quartiles"] == [-3.0, -2.0, -1.0] and r["paraphrase_median"] == -2.0

    def test_a_paraphrase_median_outside_fails(self):
        assert not ty.match([-4.0, -3.0, -2.0, -1.0, 0.0], [0.5, 0.6, 0.7])["passes"]

    def test_ks(self):
        d, p = ty.ks_two_sample([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert d == 0 and p == pytest.approx(1.0)
        d, p = ty.ks_two_sample([float(i) for i in range(50)], [float(i) + 100 for i in range(50)])
        assert d == 1 and p < 1e-6


class TestReadout:
    def test_dropped_paraphrases_are_left_out_and_ptrain_is_not_gated(self):
        deltas = {
            "confirm": {f"e{i}": -float(i) for i in range(10)},
            "pconfirm": {"p1": -4.0, "p2": -5.0, "p3": 50.0},
            "train": {f"t{i}": -float(i) for i in range(10)},
            "ptrain": {"q1": 40.0},
        }
        r = ty.readout(deltas, {"pconfirm": {"p3"}})
        assert r["pconfirm"]["paraphrase_pairs"] == 2 and r["pconfirm"]["passes"] and r["pconfirm"]["gated"]
        assert not r["ptrain"]["passes"] and not r["ptrain"]["gated"]
        assert r["all_read_sets_pass"]
