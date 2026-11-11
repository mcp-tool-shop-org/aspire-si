"""Unit tests for the CPU-only S2 v1.1 guard-chain simulator (``sim_guard_chain.py``).

Follows the test layout of ``tests/test_verifier_s2.py`` and ``tests/test_verifier_s2_train.py``.
Runs strictly on CPU with synthetic inputs only; loads no model and reads no run output.
"""

from __future__ import annotations

import json
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


def _assert_counts_only(node: Any, *, key_path: str = "") -> None:
    """Assert that result structures outside ``parameters`` report integer counts, never rates."""
    if isinstance(node, dict):
        for k, v in node.items():
            next_path = f"{key_path}.{k}" if key_path else str(k)
            if k == "parameters":
                continue
            k_low = str(k).lower()
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


# -- 1. Reproducibility: same seed gives exact same result file ------------------------------------------


@pytest.mark.parametrize("mode", sim.MODES)
def test_same_seed_gives_identical_result_file(tmp_path: Path, mode: str) -> None:
    out_a = tmp_path / f"{mode}_a.json"
    out_b = tmp_path / f"{mode}_b.json"
    out_c = tmp_path / f"{mode}_diff_seed.json"

    common_args = [
        "--mode",
        mode,
        "--n-runs",
        "6",
        "--flip-worse-grid",
        "0.0,0.25",
        "--flip-better-grid",
        "0.2",
        "--shift-grid",
        "0.0,0.35",
        "--p-flip-worse",
        "0.08",
    ]
    assert sim.main([*common_args, "--seed", "12345", "--out", str(out_a)]) == 0
    assert sim.main([*common_args, "--seed", "12345", "--out", str(out_b)]) == 0
    assert sim.main([*common_args, "--seed", "54321", "--out", str(out_c)]) == 0

    bytes_a = out_a.read_bytes()
    bytes_b = out_b.read_bytes()
    bytes_c = out_c.read_bytes()
    assert bytes_a == bytes_b
    assert bytes_a != bytes_c


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
    assert res_default_step0["total_guard_trips"] == 0
    assert res_default_step0["total_confirmations_used"] == 0
    assert res_default_step0["runs_halted"] == 0
    assert res_default_step0["runs_completed"] == 12
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
    assert not outcome.halted
    assert outcome.verdict == "S2-TRAIN-COMPLETE"
    assert outcome.confirm_trips_total == 0
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
    assert cell["runs_confirmed_at_least_once"] == 8
    assert cell["runs_confirmed_at_least_once_by_rule"]["FUSE_A"] == 8
    assert cell["runs_confirmed_at_least_once_by_rule"]["CHANNEL_B"] == 8
    assert cell["runs_confirmed_at_least_once_by_rule"]["U1"] == 8

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
    assert ct_cell["detected_total"] == 6
    assert ct_cell["undetected_completed"] == 0
    assert ct_cell["detection_counts_by_update"]["10"] == 6
    assert ct_cell["verdict_counts"]["S2-TRAIN-HALT-FUSE_A-CONFIRMED"] == 6

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
    assert sup_cell["detected_total"] == 6
    assert sup_cell["detection_counts_by_update"]["10"] == 6
    assert sup_cell["verdict_counts"]["S2-TRAIN-HALT-CHANNEL_B-CONFIRMED"] == 6


# -- 4. Shared 184-item pass increments counter by 1, not 2 ----------------------------------------------


def test_shared_pass_adds_one_to_counter_not_two() -> None:
    # Step 0 has ~75% gold-unsupported already "supported" with p_flip_better=0 and w_ct=1.0:
    # - Single canary item stays "supported" -> Fuse A never trips.
    # - Window 1 (<600 sup answers) -> S2 cannot trip.
    # - Channel B trips in Window 1 on step-0 "supported" items, runs 184-item pass #1, gets CONFIRM-NEGATIVE
    #   (0 new flips to "supported").
    # - Remaining gold-unsupported items flip with p=0.85 to "cannot_tell", so U1 trips in the SAME Window 1,
    #   reuses the cached 184-item pass (shared_pass=True), and confirms without a 2nd runner call or counter bump.
    params = sim.validate_and_resolve_parameters(
        {
            "pool_size": 394,
            "canary_c_size": 1,
            "gold_unsupported_size": 184,
            "gold_supported_size": 210,
            "gold_cannot_tell_size": 0,
            "step0_canary_c_supported_prob": 1.0,
            "step0_canary_c_cannot_tell_prob": 0.0,
            "step0_unsup_supported_prob": 0.75,
            "step0_unsup_cannot_tell_prob": 0.0,
            "step0_sup_unsupported_prob": 0.0,
            "step0_sup_cannot_tell_prob": 0.0,
            "p_flip_worse": 0.85,
            "worse_cannot_tell_fraction": 1.0,
            "p_flip_better": 0.0,
            "group_dispersion": 0.0,
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
    assert outcome.confirmations_used == 1
    assert outcome.runner_calls == 1
    assert outcome.guard is not None
    events = outcome.guard.window_history[0]["confirm_events"]
    assert [e["rule"] for e in events] == ["CHANNEL_B", "U1"]
    assert events[0]["status"] == "CONFIRM-NEGATIVE"
    assert events[0]["shared_pass"] is False
    assert events[1]["status"] == "CONFIRM-POSITIVE"
    assert events[1]["shared_pass"] is True

    agg = sim.run_end_to_end_mode(params)
    assert agg["runs_with_shared_pass"] == 5
    # Every shared pass adds 1 confirmation instead of 2, so trips strictly exceed confirmations by >= 5.
    assert agg["total_guard_trips"] - agg["total_confirmations_used"] >= 5


# -- 5. Confirmation cap halts unconfirmed without a 4th pass --------------------------------------------


def test_cap_reaches_unconfirmed_halt_without_fourth_pass() -> None:
    # Step-0 has 80% gold-unsupported items stuck at "supported" with zero flips (b_worse=0, c_better=0).
    # Channel B trips in Windows 1, 2, 3 (each CONFIRM-NEGATIVE -> confirmations_used = 1, 2, 3),
    # then trips a 4th time in Window 4 (update 40) where confirmations_used == cap (3) -> HALT-CHANNEL_B-CAP
    # without calling confirm_runner a 4th time.
    params = sim.validate_and_resolve_parameters(
        {
            "step0_canary_c_supported_prob": 0.80,
            "step0_canary_c_cannot_tell_prob": 0.0,
            "step0_unsup_supported_prob": 0.80,
            "step0_unsup_cannot_tell_prob": 0.0,
            "step0_sup_unsupported_prob": 0.0,
            "step0_sup_cannot_tell_prob": 0.0,
            "p_flip_worse": 0.0,
            "p_flip_better": 0.0,
            "ct_drift_per_update": 0.0,
            "group_dispersion": 0.0,
            "cusum_on_confirm_negative": "keep",
            "seed": 1111,
            "n_runs": 4,
        }
    )
    pool = sim.build_synthetic_pool(params)
    outcome = sim.simulate_end_to_end_run(pool, params, run_seed=1111, retain_guard=True)
    assert outcome.halted
    assert outcome.halt_update == 40
    assert outcome.verdict == "S2-TRAIN-HALT-CHANNEL_B-CAP"
    assert outcome.confirm_trips_total == 4
    assert outcome.confirmations_used == 3
    assert outcome.runner_calls == 3
    assert S2V11.confirm_cap_per_run == 3
    assert outcome.guard is not None
    last_ev = outcome.guard.window_history[-1]["confirm_events"][-1]
    assert last_ev["status"] == "HALT-CAP"
    assert last_ev["confirmations_used"] == 3


# -- 6 & 7. Negative confirmation keeps CUSUM by default; keep/reset switch changes ONLY that ------------


def test_negative_confirmation_keeps_cusum_and_switch_changes_only_that() -> None:
    # 1(a) Deterministic 1-window check for Fuse A (plus U1 no-op on reset):
    # All canary C and gold-unsupported items start at "cannot_tell" at step 0 with zero flips.
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
        "group_dispersion": 0.0,
    }
    pool = sim.build_synthetic_pool(sim.validate_and_resolve_parameters(fa_base))
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
    assert fa_keep.guard is not None and fa_reset.guard is not None
    fa_win_keep = fa_keep.guard.window_history[0]
    fa_win_reset = fa_reset.guard.window_history[0]
    assert [(e["rule"], e["status"]) for e in fa_win_keep["confirm_events"]] == [
        ("FUSE_A", "CONFIRM-NEGATIVE"),
        ("U1", "CONFIRM-NEGATIVE"),
    ]
    assert fa_win_keep["confirm_events"] == fa_win_reset["confirm_events"]
    expected_fa = update_poisson_cusum(0.0, fa_win_keep["fuse_a_b_w"], S2V11.fuse_a_k)
    assert fa_keep.final_fuse_a_s == pytest.approx(expected_fa)
    assert fa_keep.final_fuse_a_s >= S2V11.fuse_a_h
    assert fa_reset.final_fuse_a_s == 0.0
    assert fa_keep.verdict == fa_reset.verdict == "S2-TRAIN-COMPLETE"
    assert fa_keep.confirm_trips_total == fa_reset.confirm_trips_total == 2
    assert fa_keep.confirmations_used == fa_reset.confirmations_used == 2
    assert fa_keep.runner_calls == fa_reset.runner_calls == 2

    # 1(b) Deterministic 1-window check for Channel B:
    # All gold-unsupported items start at "supported" at step 0 with zero flips.
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
        "group_dispersion": 0.0,
    }
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
    assert cb_keep.guard is not None and cb_reset.guard is not None
    cb_win_keep = cb_keep.guard.window_history[0]
    cb_win_reset = cb_reset.guard.window_history[0]
    assert [(e["rule"], e["status"]) for e in cb_win_keep["confirm_events"]] == [
        ("CHANNEL_B", "CONFIRM-NEGATIVE"),
    ]
    assert cb_win_keep["confirm_events"] == cb_win_reset["confirm_events"]
    expected_cb = update_poisson_cusum(0.0, cb_win_keep["channel_b_fa2_w"], S2V11.channel_b_k)
    assert cb_keep.final_channel_b_s == pytest.approx(expected_cb)
    assert cb_keep.final_channel_b_s >= S2V11.channel_b_h
    assert cb_reset.final_channel_b_s == 0.0
    assert cb_keep.verdict == cb_reset.verdict == "S2-TRAIN-COMPLETE"
    assert cb_keep.confirm_trips_total == cb_reset.confirm_trips_total == 1
    assert cb_keep.confirmations_used == cb_reset.confirmations_used == 1

    # 2. Multi-window divergence: with moderate step-0 "supported" (~42%, ~7.8 groups/window > k=5.03
    #    but < k+h=12.03), "keep" re-trips immediately on the window after a negative confirmation,
    #    whereas "reset" drops back to 0.0 and requires multiple windows before tripping again.
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
        "group_dispersion": 0.0,
    }
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
    assert m_keep.halted and m_keep.verdict == "S2-TRAIN-HALT-CHANNEL_B-CAP"
    assert not m_reset.halted or (
        m_reset.halt_update is not None
        and m_keep.halt_update is not None
        and m_reset.halt_update > m_keep.halt_update
    )

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
    assert captured[0] == "sensitivity map, placeholder inputs"

    raw_text = out_file.read_text(encoding="utf-8")
    loaded = json.loads(raw_text)
    assert list(loaded.keys())[0] == "label"
    assert loaded["label"] == "sensitivity map, placeholder inputs"
    assert loaded["parameters"]["defaults_status"] == "placeholder, not measured"
    _assert_counts_only(loaded)


def test_write_result_file_rejects_missing_or_wrong_label(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing required label"):
        sim.write_result_file({"mode": "end_to_end"}, tmp_path / "bad.json")


# -- Additional unit tests covering all simulator components, guards, and validation ---------------------


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


def test_item_verdict_probs_all_kinds_states_and_regimes() -> None:
    for kind in (0, 1, 2, 3):
        for step0_v in (0, 1, 2):
            for target in sim.SHIFT_TARGETS:
                for shift_sz, p_w in ((0.0, 0.0), (0.25, 0.15), (0.95, 0.85)):
                    for upd in (1, 10):
                        probs = sim._item_verdict_probs(
                            kind,
                            step0_v,
                            upd,
                            p_flip_worse=p_w,
                            w_ct=0.4,
                            p_flip_better=0.3,
                            ct_drift_per_update=0.01,
                            shift_size=shift_sz,
                            shift_target=target,
                            shift_start_update=5,
                        )
                        assert len(probs) == 3
                        assert all(0.0 <= p <= 1.0 + 1e-12 for p in probs)
                        assert sum(probs) == pytest.approx(1.0)


def test_look_correlation_and_same_step_t0_idempotence() -> None:
    params_corr1 = sim.validate_and_resolve_parameters({"look_correlation": 1.0, "p_flip_worse": 0.4})
    pool = sim.build_synthetic_pool(params_corr1)
    latent_corr1 = sim.LatentItemPoolState(pool, params_corr1, np.random.default_rng(123))
    v_first, _, _, _ = latent_corr1.look_item(0, 5)
    for _ in range(20):
        v_next, _, _, _ = latent_corr1.look_item(0, 5)
        assert v_next == v_first

    # Trigger _next_u buffer wraparound (>4096 uniform draws) and multinomial dispersion branches
    params_disp = sim.validate_and_resolve_parameters({"look_correlation": 0.0, "group_dispersion": 0.5, "p_flip_worse": 0.5})
    latent_disp = sim.LatentItemPoolState(pool, params_disp, np.random.default_rng(456))
    for i in range(600):
        cid = pool.item_ids[i % len(pool.item_ids)]
        counts = latent_disp.sample_training_group_counts(cid, 10, 8)
        assert sum(counts) == 8

    # Same-step T=0 pass idempotence: canary C pass followed by 184-item pass at the same step
    pass_c = latent_disp.run_t0_confirmation_pass(pool.canary_c_ids, step=9)
    pass_u = latent_disp.run_t0_confirmation_pass(pool.gold_unsupported_ids, step=9)
    for cid in pool.canary_c_ids:
        assert pass_c[cid]["verdict"] == pass_u[cid]["verdict"]


def test_s2_direct_halt_and_ct_drift_and_cusum_helper() -> None:
    # S2 direct halt: gold-supported items have step0 cannot_tell prob 0.25 (zero unsupported bad verdicts),
    # so once cum_sup_answers >= 600 (Window 4, update 40), S2 halts directly with 0 confirmation calls.
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
            "group_dispersion": 0.0,
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

    # Positive ct_drift_per_update alone drives Fuse A confirmation
    drift_params = sim.validate_and_resolve_parameters(
        {
            "step0_canary_c_supported_prob": 0.0,
            "step0_unsup_supported_prob": 0.0,
            "p_flip_worse": 0.0,
            "p_flip_better": 0.0,
            "ct_drift_per_update": 0.05,
            "seed": 2718,
            "n_runs": 2,
        }
    )
    drift_out = sim.simulate_end_to_end_run(pool, drift_params, run_seed=2718)
    assert drift_out.halted
    assert drift_out.verdict == "S2-TRAIN-HALT-FUSE_A-CONFIRMED"

    # Direct CUSUM helper check
    s1, tripped1 = sim.evaluate_cusum_window_step(0.0, 1, S2V11.fuse_a_k, S2V11.fuse_a_h)
    assert not tripped1 and s1 == pytest.approx(1.0 - S2V11.fuse_a_k)
    s2_val, tripped2 = sim.evaluate_cusum_window_step(s1, 4, S2V11.fuse_a_k, S2V11.fuse_a_h)
    assert tripped2 and s2_val >= S2V11.fuse_a_h

    # CLI list flag coverage (--forced-look-updates, --forced-look-rules)
    parsed = sim.build_parser().parse_args(
        [
            "--mode",
            "forced_looks",
            "--forced-look-updates",
            "20, 40",
            "--forced-look-rules",
            "FUSE_A, U1",
            "--out",
            "dummy.json",
        ]
    )
    assert parsed.forced_look_updates == [20, 40]
    assert parsed.forced_look_rules == ["FUSE_A", "U1"]


def test_each_required_test_fails_when_named_behaviour_removed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify every required Part 6 test fails if the behaviour it names is removed."""
    import dataclasses

    import s2_grpo_smoke as s2_smoke

    # 1. Remove deterministic seeding -> test_same_seed_gives_identical_result_file must fail.
    orig_ss = sim.np.random.SeedSequence
    counter = {"n": 0}

    def _unseeded(_seed: Any = None) -> np.random.SeedSequence:
        counter["n"] += 1
        return orig_ss(90000 + counter["n"])

    with monkeypatch.context() as mp:
        mp.setattr(sim.np.random, "SeedSequence", _unseeded)
        with pytest.raises(AssertionError):
            test_same_seed_gives_identical_result_file(tmp_path / "mut_seed", "end_to_end")

    # 2. Inject spurious flips when flip probs are zero -> test_zero_flip_probabilities_no_run_trips must fail.
    with monkeypatch.context() as mp:
        mp.setattr(sim, "_item_verdict_probs", lambda *_a, **_kw: (0.5, 0.0, 0.5))
        with pytest.raises(AssertionError):
            test_zero_flip_probabilities_no_run_trips()

    # 3. Disable shift confirmation -> test_forced_certain_shift_confirms must fail.
    with monkeypatch.context() as mp:
        mp.setattr(
            sim,
            "_evaluate_forced_pass",
            lambda *_a, **_kw: {"FUSE_A": False, "CHANNEL_B": False, "U1": False},
        )
        with pytest.raises(AssertionError):
            test_forced_certain_shift_confirms()

    # 4. Charge confirmations_used on shared pass too -> test_shared_pass_adds_one_to_counter_not_two must fail.
    orig_record_group = s2_smoke.CtCollapseGuard.record_group

    def _charge_shared_too(self: Any, *args: Any, **kwargs: Any) -> Any:
        prev_idx = self.window_index
        res = orig_record_group(self, *args, **kwargs)
        if self.window_index > prev_idx:
            for ev in self.window_history[-1]["confirm_events"]:
                if ev.get("shared_pass"):
                    self.confirmations_used += 1
        return res

    with monkeypatch.context() as mp:
        mp.setattr(s2_smoke.CtCollapseGuard, "record_group", _charge_shared_too)
        with pytest.raises(AssertionError):
            test_shared_pass_adds_one_to_counter_not_two()

    # 5. Remove confirmation cap -> test_cap_reaches_unconfirmed_halt_without_fourth_pass must fail.
    with monkeypatch.context() as mp:
        mp.setattr(s2_smoke, "S2V11", dataclasses.replace(S2V11, confirm_cap_per_run=99))
        with pytest.raises(AssertionError):
            test_cap_reaches_unconfirmed_halt_without_fourth_pass()

    # 6. Always reset CUSUM on negative confirmation -> keep assertion in test 6/7 must fail.
    def _always_reset_cusum(self: Any, *args: Any, **kwargs: Any) -> Any:
        prev_idx = self.window_index
        res = orig_record_group(self, *args, **kwargs)
        if self.window_index > prev_idx:
            self.fuse_a_s = 0.0
            self.channel_b_s = 0.0
        return res

    with monkeypatch.context() as mp:
        mp.setattr(s2_smoke.CtCollapseGuard, "record_group", _always_reset_cusum)
        with pytest.raises(AssertionError):
            test_negative_confirmation_keeps_cusum_and_switch_changes_only_that()

    # 7. Make keep/reset switch a no-op -> switch assertion in test 6/7 must fail.
    orig_resolve = sim.validate_and_resolve_parameters

    def _ignore_reset_switch(overrides: Any = None) -> dict[str, Any]:
        resolved = orig_resolve(overrides)
        resolved["cusum_on_confirm_negative"] = "keep"
        return resolved

    with monkeypatch.context() as mp:
        mp.setattr(sim, "validate_and_resolve_parameters", _ignore_reset_switch)
        with pytest.raises(AssertionError):
            test_negative_confirmation_keeps_cusum_and_switch_changes_only_that()

    # 8. Corrupt required label -> test_every_result_file_and_summary_carries_label_and_counts_only must fail.
    with monkeypatch.context() as mp:
        mp.setattr(sim, "RESULT_LABEL", "unlabeled output")
        with pytest.raises(AssertionError):
            test_every_result_file_and_summary_carries_label_and_counts_only(
                tmp_path / "mut_label", capsys, "end_to_end"
            )

