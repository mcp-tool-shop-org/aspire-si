"""Unit tests for the CPU-only S2 v1.1 guard-chain simulator (``sim_guard_chain.py``).

Follows the test layout of ``tests/test_verifier_s2.py`` and ``tests/test_verifier_s2_train.py``.
Runs strictly on CPU with synthetic inputs only; loads no model and reads no run output.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

VERIFIER_DIR = str(Path(__file__).parent.parent / "examples" / "verifier-distill")
if VERIFIER_DIR not in sys.path:
    sys.path.insert(0, VERIFIER_DIR)

import sim_guard_chain as sim  # noqa: E402
from s2_v11_config import S2V11, update_poisson_cusum  # noqa: E402

FORBIDDEN_RATE_SUBSTRINGS = ("rate", "pct", "percent", "ratio", "proportion")
FORBIDDEN_TIMING_SUBSTRINGS = (
    "wall",
    "elapsed",
    "duration",
    "runtime",
    "time",
    "timing",
    "seconds",
    "latency",
)
FORBIDDEN_TIMING_SUFFIXES = ("_s", "_ms", "_sec", "_secs")


def _assert_counts_only(node: Any, *, key_path: str = "") -> None:
    """Assert that result structures outside ``parameters`` report integer counts, never rates or timings."""
    if isinstance(node, dict):
        for k, v in node.items():
            next_path = f"{key_path}.{k}" if key_path else str(k)
            k_low = str(k).lower()
            if k != "parameters":
                assert not any(tok in k_low for tok in FORBIDDEN_TIMING_SUBSTRINGS) and not k_low.endswith(
                    FORBIDDEN_TIMING_SUFFIXES
                ), f"timing field {next_path!r} forbidden in simulator output"
            if k == "parameters":
                continue
            assert not any(tok in k_low for tok in FORBIDDEN_RATE_SUBSTRINGS), (
                f"rate-like field {next_path!r} forbidden in simulator output"
            )
            if k in ("p_flip_worse", "p_flip_better", "shift_size"):
                assert isinstance(v, (int, float))
            else:
                _assert_counts_only(v, key_path=next_path)
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _assert_counts_only(item, key_path=f"{key_path}[{idx}]")
    elif isinstance(node, bool):
        raise AssertionError(f"unexpected boolean count at {key_path}")
    elif isinstance(node, float):
        raise AssertionError(f"unexpected float metric {node!r} at {key_path}; expected integer count")


def _strip_echoed_seed(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``payload`` with the echoed ``seed`` fields removed."""
    cleaned = {k: v for k, v in payload.items() if k != "seed"}
    if isinstance(cleaned.get("parameters"), dict):
        cleaned["parameters"] = {k: v for k, v in cleaned["parameters"].items() if k != "seed"}
    return cleaned


def _extract_mode_counts(payload: dict[str, Any], mode: str) -> Any:
    """Extract the simulated count payload for ``mode``."""
    if mode == "forced_looks":
        return payload["grid_counts"]
    if mode == "end_to_end":
        return {
            "verdict_counts": payload["verdict_counts"],
            "runs_completed": payload["runs_completed"],
            "runs_halted": payload["runs_halted"],
            "runs_with_shared_pass": payload["runs_with_shared_pass"],
            "total_guard_trips": payload["total_guard_trips"],
            "total_confirmations_used": payload["total_confirmations_used"],
        }
    return payload["shift_cells"]


# -- 1. Reproducibility: same seed gives exact same result file; different seeds give different counts ----


@pytest.mark.parametrize("mode", sim.MODES)
def test_same_seed_gives_identical_result_file(tmp_path: Path, mode: str) -> None:
    out_a = tmp_path / f"{mode}_a.json"
    out_b = tmp_path / f"{mode}_b.json"
    out_c = tmp_path / f"{mode}_diff_seed.json"

    common_args = [
        "--mode",
        mode,
        "--n-runs",
        "24",
        "--flip-worse-grid",
        "0.012,0.02,0.035",
        "--flip-better-grid",
        "0.2",
        "--shift-grid",
        "0.015,0.035,0.06",
        "--p-flip-worse",
        "0.03",
    ]
    assert sim.main([*common_args, "--seed", "12345", "--out", str(out_a)]) == 0
    assert sim.main([*common_args, "--seed", "12345", "--out", str(out_b)]) == 0
    assert sim.main([*common_args, "--seed", "54321", "--out", str(out_c)]) == 0

    bytes_a = out_a.read_bytes()
    bytes_b = out_b.read_bytes()
    bytes_c = out_c.read_bytes()
    assert bytes_a == bytes_b, f"SAME_SEED_BYTES_{mode}: identical seeds must write identical bytes"

    json_a = json.loads(bytes_a.decode("utf-8"))
    json_c = json.loads(bytes_c.decode("utf-8"))
    stripped_a = _strip_echoed_seed(json_a)
    stripped_c = _strip_echoed_seed(json_c)
    counts_a = _extract_mode_counts(json_a, mode)
    counts_c = _extract_mode_counts(json_c, mode)
    assert stripped_a != stripped_c and counts_a != counts_c, (
        f"SEED_IGNORED_{mode}: different seeds must produce different counts in {mode}"
    )


# -- 2. Zero flip probabilities: no run trips ------------------------------------------------------------


def test_zero_flip_probabilities_no_run_trips() -> None:
    # Both with default step-0 probs and with clean step-0 probs, zero flip/drift produces zero trips.
    res_default_step0 = sim.run_end_to_end_mode(
        {
            "n_runs": 12,
            "seed": 77,
            "p_flip_worse": 0.0,
            "p_flip_better": 0.0,
            "ct_drift_per_update": 0.0,
        }
    )
    assert res_default_step0["total_guard_trips"] == 0, (
        "ZERO_FLIP_NO_TRIPS: expected zero guard trips when flip and drift probabilities are zero"
    )
    assert res_default_step0["total_confirmations_used"] == 0, "ZERO_FLIP_NO_TRIPS: confirmations_used != 0"
    assert res_default_step0["runs_halted"] == 0, "ZERO_FLIP_NO_TRIPS: runs_halted != 0"
    assert res_default_step0["runs_completed"] == 12, "ZERO_FLIP_NO_TRIPS: runs_completed != 12"
    assert res_default_step0["verdict_counts"]["S2-TRAIN-COMPLETE"] == 12

    clean_params = sim.validate_and_resolve_parameters(
        {
            "n_runs": 8,
            "seed": 88,
            "step0_canary_c_supported_prob": 0.0,
            "step0_canary_c_cannot_tell_prob": 0.0,
            "step0_unsup_supported_prob": 0.0,
            "step0_unsup_cannot_tell_prob": 0.0,
            "step0_sup_unsupported_prob": 0.0,
            "step0_sup_cannot_tell_prob": 0.0,
            "step0_ct_supported_prob": 0.0,
            "step0_ct_unsupported_prob": 0.0,
            "p_flip_worse": 0.0,
            "p_flip_better": 0.0,
            "ct_drift_per_update": 0.0,
        }
    )
    pool = sim.build_synthetic_pool(clean_params)
    outcome = sim.simulate_end_to_end_run(pool, clean_params, run_seed=88, retain_guard=True)
    assert not outcome.halted, "ZERO_FLIP_NO_TRIPS: clean run halted"
    assert outcome.verdict == "S2-TRAIN-COMPLETE"
    assert outcome.confirm_trips_total == 0, "ZERO_FLIP_NO_TRIPS: clean run tripped"
    assert outcome.confirmations_used == 0
    assert outcome.runner_calls == 0
    assert outcome.final_fuse_a_s == 0.0
    assert outcome.final_channel_b_s == 0.0


# -- 3. Forced certain shift confirms --------------------------------------------------------------------


def test_forced_certain_shift_confirms() -> None:
    forced_res = sim.run_forced_looks_mode(
        {
            "n_runs": 8,
            "seed": 303,
            "step0_canary_c_supported_prob": 0.0,
            "step0_canary_c_cannot_tell_prob": 0.0,
            "step0_unsup_supported_prob": 0.0,
            "step0_unsup_cannot_tell_prob": 0.0,
            "flip_worse_grid": [1.0],
            "flip_better_grid": [0.0],
            "worse_cannot_tell_fraction": 0.5,
            "forced_look_updates": [10],
        }
    )
    cell = forced_res["grid_counts"][0]
    assert cell["runs_confirmed_at_least_once"] == 8, (
        "CERTAIN_SHIFT_FORCED_LOOKS: expected all 8 forced-look runs to confirm"
    )
    assert cell["runs_confirmed_at_least_once_by_rule"]["FUSE_A"] == 8, (
        "CERTAIN_SHIFT_FORCED_LOOKS: expected FUSE_A to confirm in all 8 runs"
    )
    assert cell["runs_confirmed_at_least_once_by_rule"]["CHANNEL_B"] == 8, (
        "CERTAIN_SHIFT_FORCED_LOOKS: expected CHANNEL_B to confirm in all 8 runs"
    )
    assert cell["runs_confirmed_at_least_once_by_rule"]["U1"] == 8, (
        "CERTAIN_SHIFT_FORCED_LOOKS: expected U1 to confirm in all 8 runs"
    )

    # End-to-end shift_grid with shift=1.0 to cannot_tell confirms Fuse A at first window (update 10).
    shift_ct = sim.run_shift_grid_mode(
        {
            "n_runs": 6,
            "seed": 404,
            "p_flip_worse": 0.0,
            "p_flip_better": 0.0,
            "shift_grid": [1.0],
            "shift_target": "cannot_tell",
            "shift_start_update": 1,
        }
    )
    ct_cell = shift_ct["shift_cells"][0]
    assert ct_cell["detected_total"] == 6, (
        "CERTAIN_SHIFT_GRID: expected all 6 cannot_tell shift_grid runs to detect shift"
    )
    assert ct_cell["undetected_completed"] == 0, "CERTAIN_SHIFT_GRID: expected 0 undetected cannot_tell runs"
    assert ct_cell["detection_counts_by_update"]["10"] == 6, (
        "CERTAIN_SHIFT_GRID: expected cannot_tell shift detected at update 10"
    )
    assert ct_cell["verdict_counts"]["S2-TRAIN-HALT-FUSE_A-CONFIRMED"] == 6, (
        "CERTAIN_SHIFT_GRID: expected FUSE_A-CONFIRMED verdict"
    )

    # End-to-end shift_grid with shift=1.0 to supported confirms Channel B at first window (update 10).
    shift_sup = sim.run_shift_grid_mode(
        {
            "n_runs": 6,
            "seed": 505,
            "step0_canary_c_supported_prob": 0.0,
            "step0_unsup_supported_prob": 0.0,
            "p_flip_worse": 0.0,
            "p_flip_better": 0.0,
            "shift_grid": [1.0],
            "shift_target": "supported",
            "shift_start_update": 1,
        }
    )
    sup_cell = shift_sup["shift_cells"][0]
    assert sup_cell["detected_total"] == 6, (
        "CERTAIN_SHIFT_GRID: expected all 6 supported shift_grid runs to detect shift"
    )
    assert sup_cell["detection_counts_by_update"]["10"] == 6, (
        "CERTAIN_SHIFT_GRID: expected supported shift detected at update 10"
    )
    assert sup_cell["verdict_counts"]["S2-TRAIN-HALT-CHANNEL_B-CONFIRMED"] == 6, (
        "CERTAIN_SHIFT_GRID: expected CHANNEL_B-CONFIRMED verdict"
    )


# -- 4. Shared 184-item pass increments counter by 1, not 2 ----------------------------------------------


def test_shared_pass_adds_one_to_counter_not_two() -> None:
    # 184-item gold-unsupported pool with single canary item starting at "supported" (step-0 bad):
    # - Single canary item cannot confirm Fuse A (ct_now <= 1 < 2) and b_w <= 1 < 3.25 in Window 1.
    # - gold_supported_size = 0 -> S2 never accrues 600 answers.
    # - Under Part 3, step-0 "supported" items have (p_sup, p_unsup, p_ct) = (0.62, 0.0, 0.38), so in
    #   Window 1 (60 groups, update 10) Channel B trips first (fa2_w ~ 37 >= 13), runs the 184-item pass #1,
    #   and gets CONFIRM-NEGATIVE (b_worse = 0 since all 184 started at "supported").
    # - U1 (~23 CT items >= 4, correct_rate = 0.0 < 0.78) trips in the SAME Window 1, reuses the cached
    #   184-item pass (shared_pass=True), and confirms without a 2nd runner call or counter bump.
    params = sim.validate_and_resolve_parameters(
        {
            "pool_size": 184,
            "canary_c_size": 1,
            "gold_unsupported_size": 184,
            "gold_supported_size": 0,
            "gold_cannot_tell_size": 0,
            "groups_per_update": 6,
            "window_groups": 60,
            "step0_canary_c_supported_prob": 1.0,
            "step0_canary_c_cannot_tell_prob": 0.0,
            "step0_unsup_supported_prob": 1.0,
            "step0_unsup_cannot_tell_prob": 0.0,
            "step0_sup_unsupported_prob": 0.0,
            "step0_sup_cannot_tell_prob": 0.0,
            "p_flip_worse": 0.38,
            "worse_cannot_tell_fraction": 1.0,
            "p_flip_better": 0.0,
            "rollout_independence": 0.0,
            "n_runs": 5,
            "seed": 909,
        }
    )
    pool = sim.build_synthetic_pool(params)
    outcome = sim.simulate_end_to_end_run(pool, params, run_seed=909, retain_guard=True)
    assert outcome.halted
    assert outcome.halt_update == 10
    assert outcome.verdict == "S2-TRAIN-HALT-U1-CONFIRMED"
    assert outcome.had_shared_pass is True
    assert outcome.confirm_trips_total == 2
    assert outcome.confirmations_used == 1, (
        "SHARED_PASS_COUNTER: shared 184-item pass must increment confirmations_used by 1, not 2"
    )
    assert outcome.runner_calls == 1, (
        "SHARED_PASS_RUNNER_CALLS: shared 184-item pass must call confirm_runner once, not twice"
    )
    assert outcome.guard is not None
    events = outcome.guard.window_history[0]["confirm_events"]
    assert [e["rule"] for e in events] == ["CHANNEL_B", "U1"]
    assert events[0]["status"] == "CONFIRM-NEGATIVE"
    assert events[0]["shared_pass"] is False
    assert events[1]["status"] == "CONFIRM-POSITIVE"
    assert events[1]["shared_pass"] is True

    agg = sim.run_end_to_end_mode(params)
    assert agg["runs_with_shared_pass"] == 5
    assert agg["total_guard_trips"] - agg["total_confirmations_used"] >= 5, (
        "SHARED_PASS_COUNTER: aggregate confirmations_used must be less than guard trips across shared passes"
    )


# -- 5. Confirmation cap halts unconfirmed without a 4th pass --------------------------------------------


def test_cap_reaches_unconfirmed_halt_without_fourth_pass() -> None:
    # Step-0 has 100% gold-unsupported items stuck at "supported" with zero flips (b_worse=0, c_better=0).
    # Channel B trips in Windows 1, 2, 3 (each CONFIRM-NEGATIVE -> confirmations_used = 1, 2, 3),
    # then trips a 4th time in Window 4 (update 40) where confirmations_used == cap (3) -> HALT-CHANNEL_B-CAP
    # without calling confirm_runner a 4th time.
    params = sim.validate_and_resolve_parameters(
        {
            "step0_canary_c_supported_prob": 1.0,
            "step0_canary_c_cannot_tell_prob": 0.0,
            "step0_unsup_supported_prob": 1.0,
            "step0_unsup_cannot_tell_prob": 0.0,
            "step0_sup_unsupported_prob": 0.0,
            "step0_sup_cannot_tell_prob": 0.0,
            "p_flip_worse": 0.0,
            "p_flip_better": 0.0,
            "ct_drift_per_update": 0.0,
            "rollout_independence": 0.0,
            "cusum_on_confirm_negative": "keep",
            "seed": 1111,
            "n_runs": 4,
        }
    )
    pool = sim.build_synthetic_pool(params)
    outcome = sim.simulate_end_to_end_run(pool, params, run_seed=1111, retain_guard=True)
    assert outcome.halted
    assert outcome.halt_update == 40
    assert outcome.verdict == "S2-TRAIN-HALT-CHANNEL_B-CAP", (
        "CAP_NO_FOURTH_PASS: fourth trip must halt with CHANNEL_B-CAP"
    )
    assert outcome.confirm_trips_total == 4
    assert outcome.confirmations_used == 3, "CAP_NO_FOURTH_PASS: confirmations_used must stop at cap 3"
    assert outcome.runner_calls == 3, "CAP_NO_FOURTH_PASS: confirm_runner must not be called a fourth time"
    assert S2V11.confirm_cap_per_run == 3
    assert outcome.guard is not None
    last_ev = outcome.guard.window_history[-1]["confirm_events"][-1]
    assert last_ev["status"] == "HALT-CAP"
    assert last_ev["confirmations_used"] == 3


# -- 6 & 7. Negative confirmation keeps CUSUM by default; keep/reset switch changes ONLY that ------------


def test_negative_confirmation_keeps_cusum_and_switch_changes_only_that() -> None:
    # 1(a) Deterministic 1-window check for Fuse A (omitting cusum_on_confirm_negative must behave as "keep"):
    fa_base = {
        "updates": 10,
        "step0_canary_c_supported_prob": 0.0,
        "step0_canary_c_cannot_tell_prob": 1.0,
        "step0_unsup_supported_prob": 0.0,
        "step0_unsup_cannot_tell_prob": 1.0,
        "step0_sup_unsupported_prob": 0.0,
        "step0_sup_cannot_tell_prob": 0.0,
        "p_flip_worse": 0.0,
        "p_flip_better": 0.0,
        "ct_drift_per_update": 0.0,
        "rollout_independence": 0.0,
    }
    pool = sim.build_synthetic_pool(sim.validate_and_resolve_parameters(fa_base))
    fa_default = sim.simulate_end_to_end_run(
        pool,
        sim.validate_and_resolve_parameters(fa_base),
        run_seed=2026,
        retain_guard=True,
    )
    fa_keep = sim.simulate_end_to_end_run(
        pool,
        sim.validate_and_resolve_parameters({**fa_base, "cusum_on_confirm_negative": "keep"}),
        run_seed=2026,
        retain_guard=True,
    )
    fa_reset = sim.simulate_end_to_end_run(
        pool,
        sim.validate_and_resolve_parameters({**fa_base, "cusum_on_confirm_negative": "reset"}),
        run_seed=2026,
        retain_guard=True,
    )
    assert fa_default.guard is not None and fa_keep.guard is not None and fa_reset.guard is not None
    fa_win_keep = fa_keep.guard.window_history[0]
    fa_win_reset = fa_reset.guard.window_history[0]
    assert [(e["rule"], e["status"]) for e in fa_win_keep["confirm_events"]] == [
        ("FUSE_A", "CONFIRM-NEGATIVE"),
        ("U1", "CONFIRM-NEGATIVE"),
    ]
    assert fa_win_keep["confirm_events"] == fa_win_reset["confirm_events"]
    expected_fa = update_poisson_cusum(0.0, fa_win_keep["fuse_a_b_w"], S2V11.fuse_a_k)
    assert fa_default.final_fuse_a_s == pytest.approx(expected_fa), (
        "DEFAULT_CUSUM_KEEP: omitting cusum_on_confirm_negative must keep Fuse A CUSUM"
    )
    assert fa_default.final_fuse_a_s >= S2V11.fuse_a_h, (
        "DEFAULT_CUSUM_KEEP: omitted cusum_on_confirm_negative must stay above Fuse A threshold"
    )
    assert fa_keep.final_fuse_a_s == pytest.approx(expected_fa)
    assert fa_keep.final_fuse_a_s >= S2V11.fuse_a_h
    assert fa_reset.final_fuse_a_s == 0.0, (
        "SWITCH_CUSUM_RESET: cusum_on_confirm_negative='reset' must zero Fuse A CUSUM on negative confirmation"
    )
    assert fa_default.verdict == fa_keep.verdict == fa_reset.verdict == "S2-TRAIN-COMPLETE"
    assert fa_default.confirm_trips_total == fa_keep.confirm_trips_total == fa_reset.confirm_trips_total == 2
    assert fa_default.confirmations_used == fa_keep.confirmations_used == fa_reset.confirmations_used == 2
    assert fa_default.runner_calls == fa_keep.runner_calls == fa_reset.runner_calls == 2

    # 1(b) Deterministic 1-window check for Channel B (also checking omitted default == "keep"):
    cb_base = {
        "updates": 10,
        "step0_canary_c_supported_prob": 1.0,
        "step0_canary_c_cannot_tell_prob": 0.0,
        "step0_unsup_supported_prob": 1.0,
        "step0_unsup_cannot_tell_prob": 0.0,
        "step0_sup_unsupported_prob": 0.0,
        "step0_sup_cannot_tell_prob": 0.0,
        "p_flip_worse": 0.0,
        "p_flip_better": 0.0,
        "ct_drift_per_update": 0.0,
        "rollout_independence": 0.0,
    }
    cb_default = sim.simulate_end_to_end_run(
        pool,
        sim.validate_and_resolve_parameters(cb_base),
        run_seed=2026,
        retain_guard=True,
    )
    cb_keep = sim.simulate_end_to_end_run(
        pool,
        sim.validate_and_resolve_parameters({**cb_base, "cusum_on_confirm_negative": "keep"}),
        run_seed=2026,
        retain_guard=True,
    )
    cb_reset = sim.simulate_end_to_end_run(
        pool,
        sim.validate_and_resolve_parameters({**cb_base, "cusum_on_confirm_negative": "reset"}),
        run_seed=2026,
        retain_guard=True,
    )
    assert cb_default.guard is not None and cb_keep.guard is not None and cb_reset.guard is not None
    cb_win_keep = cb_keep.guard.window_history[0]
    cb_win_reset = cb_reset.guard.window_history[0]
    assert [(e["rule"], e["status"]) for e in cb_win_keep["confirm_events"]] == [
        ("CHANNEL_B", "CONFIRM-NEGATIVE"),
    ]
    assert cb_win_keep["confirm_events"] == cb_win_reset["confirm_events"]
    expected_cb = update_poisson_cusum(0.0, cb_win_keep["channel_b_fa2_w"], S2V11.channel_b_k)
    assert cb_default.final_channel_b_s == pytest.approx(expected_cb), (
        "DEFAULT_CUSUM_KEEP: omitting cusum_on_confirm_negative must keep Channel B CUSUM"
    )
    assert cb_keep.final_channel_b_s == pytest.approx(expected_cb)
    assert cb_keep.final_channel_b_s >= S2V11.channel_b_h
    assert cb_reset.final_channel_b_s == 0.0, (
        "SWITCH_CUSUM_RESET: cusum_on_confirm_negative='reset' must zero Channel B CUSUM"
    )
    assert cb_default.verdict == cb_keep.verdict == cb_reset.verdict == "S2-TRAIN-COMPLETE"
    assert cb_default.confirm_trips_total == cb_keep.confirm_trips_total == cb_reset.confirm_trips_total == 1
    assert cb_default.confirmations_used == cb_keep.confirmations_used == cb_reset.confirmations_used == 1

    # 2. Multi-window divergence with omitted default matching "keep" and diverging from "reset":
    multi_base = {
        "updates": 120,
        "step0_canary_c_supported_prob": 0.42,
        "step0_canary_c_cannot_tell_prob": 0.0,
        "step0_unsup_supported_prob": 0.42,
        "step0_unsup_cannot_tell_prob": 0.0,
        "step0_sup_unsupported_prob": 0.0,
        "step0_sup_cannot_tell_prob": 0.0,
        "p_flip_worse": 0.0,
        "p_flip_better": 0.0,
        "rollout_independence": 0.0,
    }
    m_default = sim.simulate_end_to_end_run(
        pool,
        sim.validate_and_resolve_parameters(multi_base),
        run_seed=4242,
    )
    m_keep = sim.simulate_end_to_end_run(
        pool,
        sim.validate_and_resolve_parameters({**multi_base, "cusum_on_confirm_negative": "keep"}),
        run_seed=4242,
    )
    m_reset = sim.simulate_end_to_end_run(
        pool,
        sim.validate_and_resolve_parameters({**multi_base, "cusum_on_confirm_negative": "reset"}),
        run_seed=4242,
    )
    assert m_default.halted and m_default.verdict == "S2-TRAIN-HALT-CHANNEL_B-CAP", (
        "DEFAULT_CUSUM_KEEP: multi-window omitted cusum_on_confirm_negative must behave as keep"
    )
    assert m_default.halt_update == m_keep.halt_update, (
        "DEFAULT_CUSUM_KEEP: omitted default halt_update must equal explicit keep"
    )
    assert m_keep.halted and m_keep.verdict == "S2-TRAIN-HALT-CHANNEL_B-CAP"
    assert not m_reset.halted or (
        m_reset.halt_update is not None
        and m_keep.halt_update is not None
        and m_reset.halt_update > m_keep.halt_update
    ), "SWITCH_CUSUM_RESET: reset must delay or prevent cap halt relative to keep"

    # 3. When NO negative confirmation occurs (immediate positive shift), "keep" and "reset" change NOTHING.
    pos_keep = sim.run_end_to_end_mode(
        {
            "n_runs": 5,
            "seed": 707,
            "p_flip_worse": 1.0,
            "worse_cannot_tell_fraction": 1.0,
            "cusum_on_confirm_negative": "keep",
        }
    )
    pos_reset = sim.run_end_to_end_mode(
        {
            "n_runs": 5,
            "seed": 707,
            "p_flip_worse": 1.0,
            "worse_cannot_tell_fraction": 1.0,
            "cusum_on_confirm_negative": "reset",
        }
    )
    for key in (
        "verdict_counts",
        "runs_completed",
        "runs_halted",
        "runs_with_shared_pass",
        "total_guard_trips",
        "total_confirmations_used",
    ):
        assert pos_keep[key] == pos_reset[key]


# -- 8. Every result file and printed summary carries the required label and counts only -----------------


@pytest.mark.parametrize("mode", sim.MODES)
def test_every_result_file_and_summary_carries_label_and_counts_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], mode: str
) -> None:
    out_file = tmp_path / f"res_{mode}.json"
    rc = sim.main(
        [
            "--mode",
            mode,
            "--n-runs",
            "4",
            "--seed",
            "99",
            "--flip-worse-grid",
            "0.0,0.05",
            "--shift-grid",
            "0.0,0.2",
            "--out",
            str(out_file),
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out.strip().splitlines()
    assert len(captured) >= 2, "printed summary must contain label line and counts line"
    assert captured[0] == "sensitivity map, placeholder inputs", (
        f"SUMMARY_LABEL_MISMATCH: expected required label on line 1, got {captured[0]!r}"
    )

    wall_lines = 0
    counts_lines = 0
    for line_idx, raw_line in enumerate(captured[1:], start=1):
        if raw_line.startswith("wall_s: "):
            wall_lines += 1
            assert wall_lines <= 1, "at most one wall_s timing line may appear in printed summary"
            wall_str = raw_line[len("wall_s: ") :].strip()
            assert re.fullmatch(r"\d+\.\d+", wall_str) is not None, (
                f"invalid wall_s line format: {raw_line!r}"
            )
            assert float(wall_str) >= 0.0
            continue
        counts_lines += 1
        try:
            parsed_line = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"unexpected non-JSON summary line[{line_idx}]: {raw_line!r}") from exc
        _assert_counts_only(parsed_line, key_path=f"summary_line[{line_idx}]")
        assert re.search(r"\d+\.\d+", raw_line) is None, (
            f"unexpected float literal in summary_line[{line_idx}]: {raw_line!r}"
        )
    assert counts_lines >= 1, "printed summary must contain at least one JSON counts line"

    raw_text = out_file.read_text(encoding="utf-8")
    loaded = json.loads(raw_text)
    assert list(loaded.keys())[0] == "label"
    assert loaded["label"] == "sensitivity map, placeholder inputs", (
        f"FILE_LABEL_MISMATCH: expected required label in result file, got {loaded['label']!r}"
    )
    assert loaded["parameters"]["defaults_status"] == "placeholder, not measured"
    _assert_counts_only(loaded)


def test_write_result_file_rejects_missing_or_wrong_label(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="missing required label"):
        sim.write_result_file({"mode": "end_to_end"}, missing_path)
    assert not missing_path.exists()

    wrong_path = tmp_path / "wrong.json"
    with pytest.raises(ValueError, match="missing required label"):
        sim.write_result_file({"label": "empirical measured rates", "mode": "end_to_end"}, wrong_path)
    assert not wrong_path.exists()


# -- Part 1: Path guard resolves first, checks substrings, and covers --out ------------------------------


def test_path_guard_refuses_relative_path_in_forbidden_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbidden_cwd = tmp_path / "run-training"
    forbidden_cwd.mkdir()
    param_file = forbidden_cwd / "params.json"
    param_file.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(forbidden_cwd)
    with pytest.raises(ValueError, match="refusing"):
        sim.validate_param_file_path(Path("params.json"))


def test_path_guard_refuses_clean_link_to_forbidden_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbidden_dir = tmp_path / "sealed_store"
    forbidden_dir.mkdir()
    target_file = forbidden_dir / "params.json"
    target_file.write_text("{}", encoding="utf-8")

    clean_dir = tmp_path / "clean_dir"
    clean_dir.mkdir()
    clean_link = clean_dir / "clean_params.json"

    try:
        os.symlink(target_file, clean_link)
    except OSError:
        clean_link.write_text("{}", encoding="utf-8")
        orig_resolve = Path.resolve
        resolved_target = orig_resolve(target_file)

        def _patched_resolve(self: Path, *args: Any, **kwargs: Any) -> Path:
            if self == clean_link:
                return resolved_target
            return orig_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", _patched_resolve)

    with pytest.raises(ValueError, match="refusing"):
        sim.validate_param_file_path(clean_link)

    # Also verify a refusal on the typed path is never lost if resolve() returns a clean path.
    clean_real = clean_dir / "real_clean.json"
    clean_real.write_text("{}", encoding="utf-8")
    typed_bad = tmp_path / "run-training" / "alias.json"
    orig_resolve_2 = Path.resolve
    resolved_clean_real = orig_resolve_2(clean_real)
    with monkeypatch.context() as mp:
        mp.setattr(
            Path,
            "resolve",
            lambda self, *a, **kw: resolved_clean_real if self == typed_bad else orig_resolve_2(self, *a, **kw),
        )
        with pytest.raises(ValueError, match="refusing"):
            sim.validate_param_file_path(typed_bad)


def test_path_guard_refuses_forbidden_name_after_other_text(tmp_path: Path) -> None:
    case_one = tmp_path / "archive_sealed_store" / "params.json"
    case_two = tmp_path / "exp_run-training_v2" / "params.json"
    with pytest.raises(ValueError, match="refusing"):
        sim.validate_param_file_path(case_one)
    with pytest.raises(ValueError, match="refusing"):
        sim.validate_param_file_path(case_two)


def test_main_refuses_forbidden_out_path_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forbidden_out_dir = tmp_path / "critic-s2"
    forbidden_out_dir.mkdir()
    bad_out = forbidden_out_dir / "result.json"

    orig_run_sim = sim.run_simulation
    sim_calls = {"n": 0}

    def _spy_run_sim(*args: Any, **kwargs: Any) -> dict[str, Any]:
        sim_calls["n"] += 1
        return orig_run_sim(*args, **kwargs)

    monkeypatch.setattr(sim, "run_simulation", _spy_run_sim)
    with pytest.raises(SystemExit, match="refusing") as exc_info:
        sim.main(["--mode", "end_to_end", "--n-runs", "1", "--out", str(bad_out)])
    assert exc_info.value.code not in (0, None)
    assert sim_calls["n"] == 0
    assert not bad_out.exists()


# -- Part 2: Confirmation passes have their own random stream and look-correlation memory ----------------


def test_confirmation_pass_uses_separate_stream_and_memory_from_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # All 184 gold-unsupported items start at "supported" at step 0 with p_flip_worse=0.05, w_ct=0.0,
    # p_flip_better=0.10 -> (p_sup, p_unsup, p_ct) = (0.905, 0.095, 0.0) at every update.
    # Run A (window_groups=60, 6 groups/update) trips Channel B at update k=10, executes a 184-item
    # confirmation pass (CONFIRM-NEGATIVE since b_worse=0), and continues through update 18.
    # Run B (window_groups=120) executes zero confirmation passes through update 18.
    base_cfg = {
        "updates": 18,
        "groups_per_update": 6,
        "step0_canary_c_supported_prob": 1.0,
        "step0_canary_c_cannot_tell_prob": 0.0,
        "step0_unsup_supported_prob": 1.0,
        "step0_unsup_cannot_tell_prob": 0.0,
        "step0_sup_unsupported_prob": 0.0,
        "step0_sup_cannot_tell_prob": 0.0,
        "p_flip_worse": 0.05,
        "worse_cannot_tell_fraction": 0.0,
        "p_flip_better": 0.10,
        "look_correlation": 0.5,
    }
    params_with_confirm = sim.validate_and_resolve_parameters({**base_cfg, "window_groups": 60})
    params_no_confirm = sim.validate_and_resolve_parameters({**base_cfg, "window_groups": 120})
    pool = sim.build_synthetic_pool(params_with_confirm)

    out_with = sim.simulate_end_to_end_run(pool, params_with_confirm, run_seed=20261110, retain_guard=True)
    out_without = sim.simulate_end_to_end_run(pool, params_no_confirm, run_seed=20261110, retain_guard=True)

    assert out_with.runner_calls == 1
    assert out_without.runner_calls == 0
    assert out_with.guard is not None and out_without.guard is not None
    assert len(out_with.guard.records) == 108 and len(out_without.guard.records) == 108

    # Every training group after update k=10 (indices 60..107) must have identical counts in both runs.
    after_k_with = [
        (r["step"], r["group"], r["item_id"], r["sup_count"], r["ct_count"], r["correct_count"])
        for r in out_with.guard.records[60:]
    ]
    after_k_without = [
        (r["step"], r["group"], r["item_id"], r["sup_count"], r["ct_count"], r["correct_count"])
        for r in out_without.guard.records[60:]
    ]
    assert after_k_with == after_k_without

    # Reverse check: the confirmation pass result itself still depends on run_seed.
    orig_confirm = sim.LatentItemPoolState.run_t0_confirmation_pass
    observed_passes: list[dict[str, str]] = []

    def _capture_confirm(self: Any, item_ids: Any, step: int) -> dict[str, dict[str, Any]]:
        res = orig_confirm(self, item_ids, step)
        observed_passes.append({cid: entry["verdict"] for cid, entry in res.items()})
        return res

    monkeypatch.setattr(sim.LatentItemPoolState, "run_t0_confirmation_pass", _capture_confirm)
    sim.simulate_end_to_end_run(pool, params_with_confirm, run_seed=101)
    sim.simulate_end_to_end_run(pool, params_with_confirm, run_seed=202)
    assert len(observed_passes) == 2
    assert observed_passes[0] != observed_passes[1]


# -- Part 3: Exact triples for _item_verdict_probs across all kinds, step-0 verdicts, and p_flip_worse ---


def test_item_verdict_probs_exact_triples_including_step0_bad() -> None:
    # Exact check at update_1based=2 with w_ct=0.4, p_flip_better=0.5, ct_drift_per_update=0.05 (drift=0.10),
    # shift_size=0.20, shift_target="supported", shift_start_update=1.
    common = {
        "w_ct": 0.4,
        "p_flip_better": 0.5,
        "ct_drift_per_update": 0.05,
        "shift_size": 0.20,
        "shift_target": "supported",
        "shift_start_update": 1,
    }

    # 1. p_flip_worse = 0.0 (exact base values)
    expected_pw0: dict[tuple[int, int], tuple[float, float, float]] = {
        (0, 1): (0.20, 0.70, 0.10),
        (0, 0): (0.585, 0.35, 0.065),
        (0, 2): (0.13, 0.35, 0.52),
        (1, 1): (0.20, 0.70, 0.10),
        (1, 0): (0.585, 0.35, 0.065),
        (1, 2): (0.13, 0.35, 0.52),
        (2, 0): (0.90, 0.0, 0.10),
        (2, 1): (0.45, 0.495, 0.055),
        (2, 2): (0.45, 0.0, 0.55),
        (3, 2): (0.20, 0.0, 0.80),
        (3, 0): (0.60, 0.0, 0.40),
        (3, 1): (0.0, 0.60, 0.40),
    }
    for (kind, s0), exp in expected_pw0.items():
        got = sim._item_verdict_probs(kind, s0, 2, p_flip_worse=0.0, **common)
        assert got == pytest.approx(exp, abs=1e-12), f"pw=0.0 mismatch for kind={kind}, step0={s0}"

    # 2. p_flip_worse = 0.5 (suppresses step-0-bad recovery and adds cross-bad flips on kinds 0, 1, 2)
    expected_pw05: dict[tuple[int, int], tuple[float, float, float]] = {
        (0, 1): (0.50, 0.20, 0.30),
        (0, 0): (0.63, 0.10, 0.27),
        (0, 2): (0.45, 0.10, 0.45),
        (1, 1): (0.50, 0.20, 0.30),
        (1, 0): (0.63, 0.10, 0.27),
        (1, 2): (0.45, 0.10, 0.45),
        (2, 0): (0.40, 0.30, 0.30),
        (2, 1): (0.20, 0.56, 0.24),
        (2, 2): (0.20, 0.24, 0.56),
        (3, 2): (0.50, 0.20, 0.30),
        (3, 0): (0.85, 0.0, 0.15),
        (3, 1): (0.0, 0.85, 0.15),
    }
    for (kind, s0), exp in expected_pw05.items():
        got = sim._item_verdict_probs(kind, s0, 2, p_flip_worse=0.5, **common)
        assert got == pytest.approx(exp, abs=1e-12), f"pw=0.5 mismatch for kind={kind}, step0={s0}"

    # 3. p_flip_worse = 1.0 (every step-0-bad item has recovery probability 0.0)
    expected_pw1: dict[tuple[int, int], tuple[float, float, float]] = {
        (0, 1): (8.0 / 13.0, 0.0, 5.0 / 13.0),
        (0, 0): (0.50, 0.0, 0.50),
        (0, 2): (0.80, 0.0, 0.20),
        (1, 1): (8.0 / 13.0, 0.0, 5.0 / 13.0),
        (1, 0): (0.50, 0.0, 0.50),
        (1, 2): (0.80, 0.0, 0.20),
        (2, 0): (0.0, 6.0 / 11.0, 5.0 / 11.0),
        (2, 1): (0.0, 0.50, 0.50),
        (2, 2): (0.0, 0.60, 0.40),
        (3, 2): (8.0 / 12.0, 4.0 / 12.0, 0.0),
        (3, 0): (1.0, 0.0, 0.0),
        (3, 1): (0.0, 1.0, 0.0),
    }
    good_idx_by_kind = {0: 1, 1: 1, 2: 0, 3: 2}
    for (kind, s0), exp in expected_pw1.items():
        got = sim._item_verdict_probs(kind, s0, 2, p_flip_worse=1.0, **common)
        assert got == pytest.approx(exp, abs=1e-12), f"pw=1.0 mismatch for kind={kind}, step0={s0}"
        if s0 != good_idx_by_kind[kind]:
            assert got[good_idx_by_kind[kind]] == 0.0

    # Zero-push exact stay-at-good branches and inactive shift before shift_start_update
    for kind, good_v, exp_zero in (
        (0, 1, (0.0, 1.0, 0.0)),
        (2, 0, (1.0, 0.0, 0.0)),
        (3, 2, (0.0, 0.0, 1.0)),
    ):
        got_zero = sim._item_verdict_probs(
            kind,
            good_v,
            1,
            p_flip_worse=0.0,
            w_ct=0.5,
            p_flip_better=0.3,
            ct_drift_per_update=0.0,
            shift_size=0.4,
            shift_target="cannot_tell",
            shift_start_update=5,
        )
        assert got_zero == exp_zero


# -- Part 4: Draw seed uses at least 64 bits -------------------------------------------------------------


def test_draw_seed_uses_at_least_64_bits(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_ss = sim.np.random.SeedSequence
    orig_sched = sim.build_class_stratified_schedule
    observed_dtypes: list[np.dtype[Any]] = []
    observed_draw_seeds: list[int] = []

    class _ObservedChildProxy:
        def __init__(self, inner: np.random.SeedSequence) -> None:
            self._inner = inner

        def generate_state(self, n_words: int, dtype: Any = np.uint32) -> np.ndarray:
            observed_dtypes.append(np.dtype(dtype))
            return self._inner.generate_state(n_words, dtype=dtype)

    class _ObservedRootProxy:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._inner = orig_ss(*args, **kwargs)

        def spawn(self, n_children: int) -> list[Any]:
            children = self._inner.spawn(n_children)
            return [_ObservedChildProxy(children[0]), *children[1:]]

    def _spy_sched(train_rows: Any, total_draws: int, *, seed: int, **kwargs: Any) -> Any:
        observed_draw_seeds.append(seed)
        return orig_sched(train_rows, total_draws, seed=seed, **kwargs)

    monkeypatch.setattr(sim.np.random, "SeedSequence", _ObservedRootProxy)
    monkeypatch.setattr(sim, "build_class_stratified_schedule", _spy_sched)

    params = sim.validate_and_resolve_parameters({"updates": 10, "n_runs": 1})
    pool = sim.build_synthetic_pool(params)
    for r_seed in (20261110, 7, 99991):
        sim.simulate_end_to_end_run(pool, params, run_seed=r_seed)

    assert len(observed_dtypes) == 3
    assert all(dt.itemsize >= 8 for dt in observed_dtypes)
    assert any(seed_val > 0xFFFFFFFF for seed_val in observed_draw_seeds)


# -- Part 7a, 7b, 7c, 7d tests ---------------------------------------------------------------------------


def test_forced_looks_drives_real_ct_collapse_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_record_step0 = sim.CtCollapseGuard.record_step0_baseline
    orig_eval_win = sim.CtCollapseGuard._evaluate_window
    step0_calls: list[int] = []
    eval_events: list[tuple[str, str]] = []

    def _spy_step0(self: Any, baseline_items: Any) -> dict[str, Any]:
        step0_calls.append(len(baseline_items))
        return orig_record_step0(self, baseline_items)

    def _spy_eval(self: Any, *, step: int) -> dict[str, Any] | None:
        res = orig_eval_win(self, step=step)
        for ev in self.window_history[-1]["confirm_events"]:
            eval_events.append((str(ev["rule"]), str(ev["status"])))
        return res

    monkeypatch.setattr(sim.CtCollapseGuard, "record_step0_baseline", _spy_step0)
    monkeypatch.setattr(sim.CtCollapseGuard, "_evaluate_window", _spy_eval)

    # 4 forced looks (> cap of 3) with certain shift at look 1: must stay uncapped and keep looking.
    res = sim.run_forced_looks_mode(
        {
            "n_runs": 2,
            "seed": 606,
            "step0_canary_c_supported_prob": 0.0,
            "step0_canary_c_cannot_tell_prob": 0.0,
            "step0_unsup_supported_prob": 0.0,
            "step0_unsup_cannot_tell_prob": 0.0,
            "flip_worse_grid": [1.0],
            "flip_better_grid": [0.0],
            "worse_cannot_tell_fraction": 0.5,
            "forced_look_updates": [10, 20, 30, 40],
        }
    )
    assert len(step0_calls) == 2 * 4 * 3
    assert len(eval_events) == 2 * 4 * 3
    assert all(status == "CONFIRM-POSITIVE" for _, status in eval_events)
    assert not any(status == "HALT-CAP" for _, status in eval_events)
    cell = res["grid_counts"][0]
    assert cell["runs_confirmed_at_least_once"] == 2
    assert cell["runs_confirmed_at_least_once_by_rule"] == {"FUSE_A": 2, "CHANNEL_B": 2, "U1": 2}


def test_shift_target_both_bad_verdicts_renamed_and_pinned() -> None:
    assert sim.SHIFT_TARGETS == ("cannot_tell", "supported", "both_bad_verdicts")
    assert sim.DEFAULT_PARAMETERS["shift_target"] == "both_bad_verdicts"

    with pytest.raises(ValueError, match="shift_target"):
        sim.validate_and_resolve_parameters({"shift_target": "both"})
    with pytest.raises(ValueError, match="shift_target"):
        sim._item_verdict_probs(
            0,
            1,
            1,
            p_flip_worse=0.0,
            w_ct=0.4,
            p_flip_better=0.0,
            ct_drift_per_update=0.0,
            shift_size=0.25,
            shift_target="both",
            shift_start_update=1,
        )
    with pytest.raises(SystemExit):
        sim.build_parser().parse_args(["--shift-target", "both", "--out", "dummy.json"])

    # Pin exact behaviour of both_bad_verdicts across all item kinds (0/1 gold-unsup, 2 gold-sup, 3 gold-ct).
    kw = {
        "p_flip_worse": 0.0,
        "w_ct": 0.4,
        "p_flip_better": 0.0,
        "ct_drift_per_update": 0.0,
        "shift_size": 0.25,
        "shift_target": "both_bad_verdicts",
        "shift_start_update": 1,
    }
    assert sim._item_verdict_probs(0, 1, 1, **kw) == pytest.approx((0.15, 0.75, 0.10), abs=1e-12)
    assert sim._item_verdict_probs(1, 1, 1, **kw) == pytest.approx((0.15, 0.75, 0.10), abs=1e-12)
    assert sim._item_verdict_probs(2, 0, 1, **kw) == pytest.approx((0.75, 0.15, 0.10), abs=1e-12)
    assert sim._item_verdict_probs(3, 2, 1, **kw) == pytest.approx((0.15, 0.0, 0.85), abs=1e-12)


def test_rollout_independence_renamed_and_old_name_refused() -> None:
    assert "rollout_independence" in sim.DEFAULT_PARAMETERS
    assert "group_dispersion" not in sim.DEFAULT_PARAMETERS
    with pytest.raises(ValueError, match="unknown simulator parameter.*group_dispersion"):
        sim.validate_and_resolve_parameters({"group_dispersion": 0.25})
    with pytest.raises(SystemExit):
        sim.build_parser().parse_args(["--group-dispersion", "0.25", "--out", "dummy.json"])


@pytest.mark.parametrize("mode", sim.MODES)
def test_model_assumptions_constant_docstring_and_result_files(tmp_path: Path, mode: str) -> None:
    assert isinstance(sim.MODEL_ASSUMPTIONS, tuple)
    assert len(sim.MODEL_ASSUMPTIONS) >= 5
    doc_norm = " ".join((sim.__doc__ or "").split())
    for sentence in sim.MODEL_ASSUMPTIONS:
        assert " ".join(sentence.split()) in doc_norm

    joined = " ".join(sim.MODEL_ASSUMPTIONS)
    assert "both_bad_verdicts" in joined and "unsupported" in joined
    assert "rollout_independence=0" in joined and "rollout_independence=1" in joined
    assert "canary" in joined.lower()
    assert "confirmation" in joined.lower()
    assert "look_correlation" in joined

    out_path = tmp_path / f"assumptions_{mode}.json"
    assert (
        sim.main(
            [
                "--mode",
                mode,
                "--n-runs",
                "2",
                "--flip-worse-grid",
                "0.0",
                "--shift-grid",
                "0.0",
                "--out",
                str(out_path),
            ]
        )
        == 0
    )
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["model_assumptions"] == list(sim.MODEL_ASSUMPTIONS)


# -- Additional unit tests covering simulator components, look correlation, and validation ----------------


def test_ensure_verifier_dir_on_path_idempotent() -> None:
    sample: list[str] = ["/existing/path"]
    sim._ensure_verifier_dir_on_path(sample, "/custom/verifier")
    assert sample == ["/custom/verifier", "/existing/path"]
    sim._ensure_verifier_dir_on_path(sample, "/custom/verifier")
    assert sample == ["/custom/verifier", "/existing/path"]


def test_parameter_validation_and_forbidden_paths(tmp_path: Path) -> None:
    for bad_rel in (
        "records.jsonl",
        "records.jsonl.gz",
        "state.json",
        "run-training/params.json",
        "sealed_store/params.json",
        "critic-s2/params.json",
        "chunk-01/params.json",
        "baseline_v1/params.json",
    ):
        with pytest.raises(ValueError, match="refusing parameter path"):
            sim.validate_param_file_path(tmp_path / bad_rel)

    good_param_path = tmp_path / "hand_params.json"
    good_param_path.write_text(json.dumps({"n_runs": 3, "seed": 17}), encoding="utf-8")
    sim.validate_param_file_path(good_param_path)
    out_json = tmp_path / "from_params.json"
    assert sim.main(["--mode", "end_to_end", "--params", str(good_param_path), "--out", str(out_json)]) == 0
    assert json.loads(out_json.read_text(encoding="utf-8"))["n_runs"] == 3

    non_obj_path = tmp_path / "list_params.json"
    non_obj_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        sim.main(["--params", str(non_obj_path), "--out", str(tmp_path / "unused.json")])

    # None overrides & None-valued key skipped
    resolved_default = sim.validate_and_resolve_parameters(None)
    assert resolved_default["pool_size"] == 394
    resolved_none_val = sim.validate_and_resolve_parameters({"n_runs": None, "p_flip_better": 0.4})
    assert resolved_none_val["n_runs"] == sim.DEFAULT_PARAMETERS["n_runs"]
    assert resolved_none_val["flip_better_grid"] == [0.4]

    resolved_explicit_fb = sim.validate_and_resolve_parameters(
        {"p_flip_better": 0.4, "flip_better_grid": [0.1, 0.3]}
    )
    assert resolved_explicit_fb["flip_better_grid"] == [0.1, 0.3]

    with pytest.raises(ValueError, match="unknown simulator parameter"):
        sim.validate_and_resolve_parameters({"not_a_param": 1})
    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        sim.validate_and_resolve_parameters({"p_flip_worse": 1.2})
    with pytest.raises(ValueError, match="bad probabilities sum to > 1.0"):
        sim.validate_and_resolve_parameters(
            {"step0_canary_c_supported_prob": 0.7, "step0_canary_c_cannot_tell_prob": 0.5}
        )
    with pytest.raises(ValueError, match="ct_drift_per_update"):
        sim.validate_and_resolve_parameters({"ct_drift_per_update": -0.01})
    with pytest.raises(ValueError, match="cusum_on_confirm_negative"):
        sim.validate_and_resolve_parameters({"cusum_on_confirm_negative": "invalid"})
    with pytest.raises(ValueError, match="shift_target"):
        sim.validate_and_resolve_parameters({"shift_target": "invalid"})
    with pytest.raises(ValueError, match="must be >= 1"):
        sim.validate_and_resolve_parameters({"n_runs": 0})
    with pytest.raises(ValueError, match="shift_start_update"):
        sim.validate_and_resolve_parameters({"shift_start_update": 0})
    with pytest.raises(ValueError, match="forced_look_updates"):
        sim.validate_and_resolve_parameters({"forced_look_updates": []})
    with pytest.raises(ValueError, match="forced_look_rules"):
        sim.validate_and_resolve_parameters({"forced_look_rules": ["NOT_A_RULE"]})
    with pytest.raises(ValueError, match="flip_worse_grid"):
        sim.validate_and_resolve_parameters({"flip_worse_grid": []})
    with pytest.raises(ValueError, match="flip_better_grid"):
        sim.validate_and_resolve_parameters({"flip_better_grid": []})
    with pytest.raises(ValueError, match="shift_grid"):
        sim.validate_and_resolve_parameters({"shift_grid": []})
    with pytest.raises(ValueError, match="canary_c_size"):
        sim.validate_and_resolve_parameters({"canary_c_size": 200})
    with pytest.raises(ValueError, match="gold_supported_size"):
        sim.validate_and_resolve_parameters({"gold_supported_size": -1, "gold_cannot_tell_size": 211})
    with pytest.raises(ValueError, match="pool class sizes"):
        sim.validate_and_resolve_parameters({"pool_size": 400})
    with pytest.raises(ValueError, match="mode must be one of"):
        sim.run_simulation("unknown_mode")


def test_look_correlation_and_same_step_t0_idempotence() -> None:
    clean_step0 = {
        "step0_canary_c_supported_prob": 0.0,
        "step0_canary_c_cannot_tell_prob": 0.0,
        "step0_unsup_supported_prob": 0.0,
        "step0_unsup_cannot_tell_prob": 0.0,
        "step0_sup_unsupported_prob": 0.0,
        "step0_sup_cannot_tell_prob": 0.0,
        "step0_ct_supported_prob": 0.0,
        "step0_ct_unsupported_prob": 0.0,
        "p_flip_worse": 0.5,
        "worse_cannot_tell_fraction": 0.5,
    }

    # 1. look_correlation = 1.0 vs 0.0 produces observably different repeat behaviour.
    params_corr1 = sim.validate_and_resolve_parameters({**clean_step0, "look_correlation": 1.0})
    params_corr0 = sim.validate_and_resolve_parameters({**clean_step0, "look_correlation": 0.0})
    pool = sim.build_synthetic_pool(params_corr1)

    latent_corr1 = sim.LatentItemPoolState(pool, params_corr1, np.random.default_rng(123))
    seq_corr1 = [latent_corr1.look_item(0, 5)[0] for _ in range(40)]
    changes_corr1 = sum(1 for a, b in zip(seq_corr1, seq_corr1[1:], strict=False) if a != b)
    assert changes_corr1 == 0

    latent_corr0 = sim.LatentItemPoolState(pool, params_corr0, np.random.default_rng(123))
    seq_corr0 = [latent_corr0.look_item(0, 5)[0] for _ in range(40)]
    changes_corr0 = sum(1 for a, b in zip(seq_corr0, seq_corr0[1:], strict=False) if a != b)
    assert changes_corr0 > 0
    assert len(set(seq_corr0)) >= 2
    assert changes_corr0 > changes_corr1

    # 2. rollout_independence = 0.0 vs 1.0 produces observably different group counts (and buffer wraparound).
    params_ri0 = sim.validate_and_resolve_parameters(
        {**clean_step0, "look_correlation": 0.0, "rollout_independence": 0.0}
    )
    params_ri1 = sim.validate_and_resolve_parameters(
        {**clean_step0, "look_correlation": 0.0, "rollout_independence": 1.0}
    )
    latent_ri0 = sim.LatentItemPoolState(pool, params_ri0, np.random.default_rng(456))
    latent_ri1 = sim.LatentItemPoolState(pool, params_ri1, np.random.default_rng(456))

    groups_ri0 = [
        latent_ri0.sample_training_group_counts(pool.item_ids[i % len(pool.item_ids)], 10, 8)
        for i in range(600)
    ]
    groups_ri1 = [
        latent_ri1.sample_training_group_counts(pool.item_ids[i % len(pool.item_ids)], 10, 8)
        for i in range(600)
    ]
    assert all(sum(g) == 8 for g in groups_ri0)
    assert all(sum(g) == 8 for g in groups_ri1)
    mixed_ri0 = sum(1 for g in groups_ri0 if max(g) < 8)
    mixed_ri1 = sum(1 for g in groups_ri1 if max(g) < 8)
    # Pin 0.0: all 8 answers copy the item's latent verdict -> zero mixed groups, multiple unanimous classes.
    assert mixed_ri0 == 0
    assert len(set(groups_ri0)) >= 2
    # Pin 1.0: all 8 answers are independent draws -> mixed groups dominate and spread shrinks.
    assert mixed_ri1 > 400
    assert groups_ri1 != groups_ri0
    var_unsup_ri0 = float(np.var([g[1] for g in groups_ri0]))
    var_unsup_ri1 = float(np.var([g[1] for g in groups_ri1]))
    assert var_unsup_ri1 < var_unsup_ri0

    # 3. Same-step T=0 pass idempotence: canary C pass followed by 184-item pass at the same step.
    pass_c = latent_ri1.run_t0_confirmation_pass(pool.canary_c_ids, step=9)
    pass_u = latent_ri1.run_t0_confirmation_pass(pool.gold_unsupported_ids, step=9)
    for cid in pool.canary_c_ids:
        assert pass_c[cid]["verdict"] == pass_u[cid]["verdict"]


def test_s2_direct_halt_and_ct_drift_and_cusum_helper() -> None:
    # S2 direct halt: gold-supported items have step0 cannot_tell prob 0.25 (zero unsupported bad verdicts),
    # so once cum_sup_answers >= 600, S2 halts directly with 0 confirmation calls.
    s2_params = sim.validate_and_resolve_parameters(
        {
            "step0_canary_c_supported_prob": 0.0,
            "step0_canary_c_cannot_tell_prob": 0.0,
            "step0_unsup_supported_prob": 0.0,
            "step0_unsup_cannot_tell_prob": 0.0,
            "step0_sup_unsupported_prob": 0.0,
            "step0_sup_cannot_tell_prob": 0.25,
            "p_flip_worse": 0.0,
            "p_flip_better": 0.0,
            "rollout_independence": 0.0,
            "seed": 314,
            "n_runs": 2,
        }
    )
    pool = sim.build_synthetic_pool(s2_params)
    s2_out = sim.simulate_end_to_end_run(pool, s2_params, run_seed=314)
    assert s2_out.halted
    assert s2_out.halt_update == 50
    assert s2_out.verdict == "S2-TRAIN-HALT-S2"
    assert s2_out.runner_calls == 0
    assert s2_out.confirmations_used == 0

    # Positive ct_drift_per_update alone drives confirmed cannot_tell-collapse halts (Fuse A / U1)
    drift_params = sim.validate_and_resolve_parameters(
        {
            "step0_canary_c_supported_prob": 0.0,
            "step0_unsup_supported_prob": 0.0,
            "p_flip_worse": 0.0,
            "p_flip_better": 0.0,
            "ct_drift_per_update": 0.05,
            "seed": 2718,
            "n_runs": 12,
        }
    )
    # Under the uint64 draw_seed change (sim_guard_chain.py:819), FUSE_A does not trip in window 1
    # for run_seed 2718 (fuse_a_s = 2.789 < 3.25), so U1 trips alone and confirms.
    drift_out = sim.simulate_end_to_end_run(pool, drift_params, run_seed=2718)
    assert drift_out.halted
    assert drift_out.halt_update == 10
    assert drift_out.confirm_trips_total == 1
    assert drift_out.confirmations_used == 1
    assert drift_out.verdict == "S2-TRAIN-HALT-U1-CONFIRMED"
    drift_agg = sim.run_end_to_end_mode(drift_params)
    assert drift_agg["runs_halted"] == 12
    assert (
        drift_agg["verdict_counts"]["S2-TRAIN-HALT-FUSE_A-CONFIRMED"]
        + drift_agg["verdict_counts"]["S2-TRAIN-HALT-U1-CONFIRMED"]
        == 12
    )
    assert drift_agg["verdict_counts"]["S2-TRAIN-HALT-FUSE_A-CONFIRMED"] > 0

    # Direct CUSUM helper check
    s1, tripped1 = sim.evaluate_cusum_window_step(0.0, 1, S2V11.fuse_a_k, S2V11.fuse_a_h)
    assert not tripped1 and s1 == pytest.approx(1.0 - S2V11.fuse_a_k)
    s2_val, tripped2 = sim.evaluate_cusum_window_step(s1, 4, S2V11.fuse_a_k, S2V11.fuse_a_h)
    assert tripped2 and s2_val >= S2V11.fuse_a_h

    # CLI list flag coverage (--forced-look-updates, --forced-look-rules, --rollout-independence)
    parsed = sim.build_parser().parse_args(
        [
            "--mode",
            "forced_looks",
            "--forced-look-updates",
            "20, 40",
            "--forced-look-rules",
            "FUSE_A, U1",
            "--rollout-independence",
            "0.4",
            "--out",
            "dummy.json",
        ]
    )
    assert parsed.forced_look_updates == [20, 40]
    assert parsed.forced_look_rules == ["FUSE_A", "U1"]
    assert parsed.rollout_independence == 0.4


# -- Part 6: Meta-test removing each simulator-side behaviour and matching the specific assertion ---------


def test_each_required_test_fails_when_named_behaviour_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify each simulator-side behaviour test fails at its specific assertion when removed."""
    orig_ss = sim.np.random.SeedSequence
    orig_verdict_probs = sim._item_verdict_probs
    orig_resolve_params = sim.validate_and_resolve_parameters
    orig_format_summary = sim.format_printed_summary
    orig_run_sim = sim.run_simulation

    # 1. Seed ignored (generator seeded with a constant) fails the same-seed test in all three modes.
    for mode in sim.MODES:
        with monkeypatch.context() as mp:
            mp.setattr(sim.np.random, "SeedSequence", lambda *_a, **_kw: orig_ss(424242))
            with pytest.raises(AssertionError, match=f"SEED_IGNORED_{mode}"):
                test_same_seed_gives_identical_result_file(tmp_path / f"mut_seed_{mode}", mode)

    # 2. Zero-flip broken -> fails test_zero_flip_probabilities_no_run_trips at ZERO_FLIP_NO_TRIPS.
    def _nonzero_flip(*args: Any, **kwargs: Any) -> tuple[float, float, float]:
        kwargs["p_flip_worse"] = max(float(kwargs["p_flip_worse"]), 0.5)
        return orig_verdict_probs(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(sim, "_item_verdict_probs", _nonzero_flip)
        with pytest.raises(AssertionError, match="ZERO_FLIP_NO_TRIPS"):
            test_zero_flip_probabilities_no_run_trips()

    # 3a. Certain shift (forced-looks half) -> fails at CERTAIN_SHIFT_FORCED_LOOKS.
    def _ignore_flip_worse(*args: Any, **kwargs: Any) -> tuple[float, float, float]:
        kwargs["p_flip_worse"] = 0.0
        return orig_verdict_probs(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(sim, "_item_verdict_probs", _ignore_flip_worse)
        with pytest.raises(AssertionError, match="CERTAIN_SHIFT_FORCED_LOOKS"):
            test_forced_certain_shift_confirms()

    # 3b. Certain shift (shift-grid half) -> forced-looks half passes; fails at CERTAIN_SHIFT_GRID.
    def _ignore_shift_size(*args: Any, **kwargs: Any) -> tuple[float, float, float]:
        kwargs["shift_size"] = 0.0
        return orig_verdict_probs(*args, **kwargs)

    with monkeypatch.context() as mp:
        mp.setattr(sim, "_item_verdict_probs", _ignore_shift_size)
        with pytest.raises(AssertionError, match="CERTAIN_SHIFT_GRID"):
            test_forced_certain_shift_confirms()

    # 4 & 5. Shared-pass call count/counter and 3-pass cap live inside pinned s2_grpo_smoke.CtCollapseGuard
    # (never patched here; listed as not covered in the build reply per Part 6).

    # 6. Default flipped from keep to reset -> fails at DEFAULT_CUSUM_KEEP.
    with monkeypatch.context() as mp:
        mp.setitem(sim.DEFAULT_PARAMETERS, "cusum_on_confirm_negative", "reset")
        with pytest.raises(AssertionError, match="DEFAULT_CUSUM_KEEP"):
            test_negative_confirmation_keeps_cusum_and_switch_changes_only_that()

    # 7. Keep/reset switch ignored -> default passes; fails at SWITCH_CUSUM_RESET.
    def _ignore_reset_switch(overrides: Any = None) -> dict[str, Any]:
        resolved = orig_resolve_params(overrides)
        resolved["cusum_on_confirm_negative"] = "keep"
        return resolved

    with monkeypatch.context() as mp:
        mp.setattr(sim, "validate_and_resolve_parameters", _ignore_reset_switch)
        with pytest.raises(AssertionError, match="SWITCH_CUSUM_RESET"):
            test_negative_confirmation_keeps_cusum_and_switch_changes_only_that()

    # 8. Label changed -> fails at SUMMARY_LABEL_MISMATCH (drain prior stdout from #1 first).
    capsys.readouterr()
    with monkeypatch.context() as mp:
        mp.setattr(sim, "RESULT_LABEL", "unlabeled output")
        with pytest.raises(AssertionError, match="SUMMARY_LABEL_MISMATCH"):
            test_every_result_file_and_summary_carries_label_and_counts_only(
                tmp_path / "mut_label", capsys, "end_to_end"
            )

    # 9. Float added to printed summary -> label passes; fails at summary_line[1] float check.
    capsys.readouterr()
    def _summary_with_float(result: Any, *, wall_s: float) -> str:
        lines = orig_format_summary(result, wall_s=wall_s).splitlines()
        parsed = json.loads(lines[1])
        parsed["extra_metric"] = 0.125
        lines[1] = json.dumps(parsed)
        return "\n".join(lines)

    with monkeypatch.context() as mp:
        mp.setattr(sim, "format_printed_summary", _summary_with_float)
        with pytest.raises(AssertionError, match=r"unexpected float metric .* at summary_line\[1\]"):
            test_every_result_file_and_summary_carries_label_and_counts_only(
                tmp_path / "mut_summary_float", capsys, "end_to_end"
            )

    # 10. Float added to result file -> summary passes; fails at result-file float check.
    capsys.readouterr()
    def _result_with_float(mode: str, params: Any = None, *, base_sha: str = sim.DEFAULT_BASE_SHA) -> dict[str, Any]:
        res = orig_run_sim(mode, params, base_sha=base_sha)
        res["extra_float_metric"] = 0.375
        return res

    with monkeypatch.context() as mp:
        mp.setattr(sim, "run_simulation", _result_with_float)
        with pytest.raises(AssertionError, match="unexpected float metric .* at extra_float_metric"):
            test_every_result_file_and_summary_carries_label_and_counts_only(
                tmp_path / "mut_file_float", capsys, "end_to_end"
            )
