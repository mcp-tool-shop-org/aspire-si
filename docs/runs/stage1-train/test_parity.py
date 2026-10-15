"""B4's parity rules, on synthetic records: the set is fixed by rule, (i) holds Argon's paired tolerances, and
(ii) holds #118's relative rules against the batch-1 noise."""

from pathlib import Path

import data
import parity

DATA = Path(__file__).resolve().parent.parent / "stage1-training-data"


def rec(right=True, stuck=False, tt=300, tier=4, lp=-0.5, truncated=False, clause=None):
    r = {"right": right, "stuck": stuck, "thinking_tokens": tt, "truncated": truncated, "tier": tier,
         "first_logprob": lp, **{c: False for c in parity.CLAUSES}}
    if clause:
        r[clause] = True
    return r


def draw(n_items=20, stuck_items=(), tt=300, wrong_items=(), lp=-0.5):
    return {(f"i{i}", s): rec(right=i not in wrong_items, stuck=i in stuck_items, tt=tt + i, lp=lp,
                              clause="long_not_progressing" if i in stuck_items else None)
            for i in range(n_items) for s in range(3)}


def test_the_parity_set_is_two_unspent_verdict_variants_per_lesson():
    items = [r for p in sorted(DATA.glob("*.jsonl")) if not p.name.endswith(".models.jsonl") for r in data.load(p)]
    chosen = parity.select(items)
    lessons = {r["lesson"] for r in chosen}
    assert len(chosen) == 2 * len(lessons) and all(r["kind"] == "E" and r["quiz"]["type"] == "verdict" for r in chosen)
    assert parity.select(list(reversed(items))) == chosen  # fixed by rule, not by file order


def test_i_passes_when_batching_changes_nothing_and_fails_on_a_stuck_shift():
    a = draw(stuck_items=(1, 2))
    assert parity.compare_i(a, draw(stuck_items=(1, 2)))["pass"]
    shifted = parity.compare_i(a, draw(stuck_items=(1, 2, 3, 4, 5)))
    assert not shifted["checks"]["stuck_rate"] and not shifted["pass"]


def test_i_holds_the_thinking_length_tolerances_and_logs_the_logprob_deviation():
    out = parity.compare_i(draw(tt=300), draw(tt=330, lp=-0.6))
    assert not out["checks"]["median_thinking"]  # +10% is over the 5% median tolerance
    assert abs(out["figures"]["first_logprob_dev_max"] - 0.1) < 1e-9 and "first_logprob" not in out["checks"]


def test_ii_judges_the_batched_loop_against_the_batch_1_noise():
    p1, p1p = draw(wrong_items=(0, 1)), draw(wrong_items=(0, 2))  # the two harness draws disagree on 2 of 20
    assert parity.compare_ii(draw(wrong_items=(0, 3)), p1, p1p)["pass"]  # the same disagreement: within the noise
    far = parity.compare_ii(draw(wrong_items=(0, 3, 4, 5, 6, 7)), p1, p1p)  # 6 of 20 differ from P1
    assert not far["checks"]["majority_agreement"]


def test_ii_holds_the_clause_and_truncation_rules():
    p1, p1p = draw(), draw()
    many_stuck = draw(stuck_items=tuple(range(5)))
    out = parity.compare_ii(many_stuck, p1, p1p)
    assert not out["checks"]["clauses"] and not out["checks"]["stuck_rate"]
    trunc = {k: {**r, "truncated": True} if k[0] in ("i0", "i1") else r for k, r in draw().items()}
    assert not parity.compare_ii(trunc, p1, p1p)["checks"]["truncations"]


def test_kappa_is_one_for_identical_and_low_for_unrelated_labels():
    assert parity.kappa([True, False, True], [True, False, True]) == 1.0
    assert parity.kappa([True, False] * 10, [False, True] * 10) < 0
