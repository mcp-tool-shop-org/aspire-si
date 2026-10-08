"""Tests for the surprisal-only baseline (addendum 4, step 1)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "sft-experiment"))

import surprisal_baseline as sb  # noqa: E402


def rows(shift, n=60, seed=0):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        for label in (1, 0):
            base = -shift if label else 0.0
            out.append(
                {
                    "prompt_id": i,
                    "label": label,
                    "tail_delta": base + rng.gauss(0, 1),
                    "span_delta": base / 2 + rng.gauss(0, 1),
                    "edit_chars": 6 + rng.randint(0, 3),
                    "edit_tokens": 1 + rng.randint(0, 1),
                }
            )
    return out


class TestLogistic:
    def test_it_recovers_a_separating_direction(self):
        x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
        y = np.array([0.0, 0.0, 1.0, 1.0])
        w = sb.fit_logistic(x, y)
        assert w[1] > 0 and sb.predict(w, np.array([[3.0]]))[0] > 0.5


class TestFolds:
    def test_a_prompt_stays_in_one_fold_and_folds_are_seeded(self):
        f = sb.folds(list(range(20)) * 2)
        assert sorted(set(f.values())) == [0, 1, 2, 3, 4] and f == sb.folds(list(range(20)))


class TestBaseline:
    def test_separable_sets_score_high_and_unseparable_near_half(self):
        high = sb.baseline(rows(3.0))
        assert high["auc"] > 0.9 and high["ci"][0] > 0.8 and high["prompts"] == 60
        assert high["feature_auc_reported_only"]["tail_delta"] > 0.9
        low = sb.baseline(rows(0.0))
        assert low["auc"] == pytest.approx(0.5, abs=0.12) and low["ci"][0] < 0.5 < low["ci"][1]
