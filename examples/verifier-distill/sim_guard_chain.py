"""CPU-only guard-confirmation simulator for the ASPIRE-SI Stage 2 (S2) v1.1 guard chain.

Simulates synthetic runs of 197 updates (4 groups per update, 8 answers per group) drawn over
2 class-stratified epochs from a 394-item synthetic pool (128 clean canary C items inside 184
gold-unsupported items), driving the pinned ``CtCollapseGuard`` and ``S2V11`` functions without
loading any model or touching a GPU.

All default parameter values are synthetic placeholders ("placeholder, not measured").
Every result file and every printed summary carries the label "sensitivity map, placeholder inputs".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _ensure_verifier_dir_on_path(path_list: list[str], verifier_dir: str | None = None) -> None:
    """Insert ``examples/verifier-distill`` at the front of ``path_list`` if not already present."""
    target = verifier_dir if verifier_dir is not None else str(Path(__file__).resolve().parent)
    if target not in path_list:
        path_list.insert(0, target)


_ensure_verifier_dir_on_path(sys.path)

from s2_draw import build_class_stratified_schedule  # noqa: E402
from s2_grpo_smoke import (  # noqa: E402
    VERDICTS,
    CtCollapseGuard,
    G,
    advantages,
    assert_no_wrong_positive_advantages,
    clamp_wrong_positive_advantages,
    macro,
    train_verdict_line,
    zero_variance,
)
from s2_v11_config import (  # noqa: E402
    S2V11,
    mcnemar_one_sided_exact_p,
    update_poisson_cusum,
)

RESULT_LABEL = "sensitivity map, placeholder inputs"
PLACEHOLDER_NOTE = "placeholder, not measured"
DEFAULT_BASE_SHA = "5407b37077609fba91312dc98ef461453309f306"
MODES = ("forced_looks", "end_to_end", "shift_grid")
CUSUM_NEGATIVE_POLICIES = ("keep", "reset")
SHIFT_TARGETS = ("cannot_tell", "supported", "both")
CONFIRM_RULES = ("FUSE_A", "CHANNEL_B", "U1")
ALL_END_TO_END_VERDICTS = (
    "S2-TRAIN-COMPLETE",
    "S2-TRAIN-HALT-S2",
    "S2-TRAIN-HALT-FUSE_A-CONFIRMED",
    "S2-TRAIN-HALT-CHANNEL_B-CONFIRMED",
    "S2-TRAIN-HALT-U1-CONFIRMED",
    "S2-TRAIN-HALT-FUSE_A-CAP",
    "S2-TRAIN-HALT-CHANNEL_B-CAP",
    "S2-TRAIN-HALT-U1-CAP",
)

_FORBIDDEN_PARAM_DIR_PREFIXES = (
    "run-training",
    "sealed_store",
    "critic-s",
    "chunk-",
    "baseline",
)

# Every default below is placeholder, not measured.
DEFAULT_PARAMETERS: dict[str, Any] = {
    "defaults_status": PLACEHOLDER_NOTE,  # placeholder, not measured
    "pool_size": 394,  # placeholder, not measured (design pool size: 394 items)
    "canary_c_size": int(S2V11.canary_c_size),  # placeholder, not measured (design |C| = 128)
    "gold_unsupported_size": int(
        S2V11.gold_unsupported_train_items
    ),  # placeholder, not measured (design |U_gold| = 184)
    "gold_supported_size": 182,  # placeholder, not measured
    "gold_cannot_tell_size": 28,  # placeholder, not measured
    "updates": int(S2V11.primary_horizon_updates),  # placeholder, not measured (design 197 updates)
    "groups_per_update": int(S2V11.groups_per_window // 10),  # placeholder, not measured (design 4 groups)
    "window_groups": int(S2V11.groups_per_window),  # placeholder, not measured (design 40 groups)
    "group_size": int(G),  # placeholder, not measured (design 8 answers per group)
    "step0_canary_c_supported_prob": 0.01,  # placeholder, not measured
    "step0_canary_c_cannot_tell_prob": 0.0,  # placeholder, not measured
    "step0_unsup_supported_prob": 0.02,  # placeholder, not measured
    "step0_unsup_cannot_tell_prob": 0.0,  # placeholder, not measured
    "step0_sup_unsupported_prob": 0.02,  # placeholder, not measured
    "step0_sup_cannot_tell_prob": 0.0,  # placeholder, not measured
    "step0_ct_supported_prob": 0.02,  # placeholder, not measured
    "step0_ct_unsupported_prob": 0.05,  # placeholder, not measured
    "p_flip_worse": 0.008,  # placeholder, not measured
    "worse_cannot_tell_fraction": 0.5,  # placeholder, not measured
    "p_flip_better": 0.25,  # placeholder, not measured
    "ct_drift_per_update": 0.0,  # placeholder, not measured
    "look_correlation": 0.6,  # placeholder, not measured
    "group_dispersion": 0.25,  # placeholder, not measured
    "cusum_on_confirm_negative": (
        "keep" if S2V11.fuse_a_keep_cusum_on_confirm_negative else "reset"
    ),  # placeholder, not measured (matches s2_v11_config.py:94)
    "forced_look_updates": [60, 120, 180],  # placeholder, not measured
    "forced_look_rules": ["FUSE_A", "CHANNEL_B", "U1"],  # placeholder, not measured
    "flip_worse_grid": [0.0, 0.005, 0.01, 0.02],  # placeholder, not measured
    "flip_better_grid": [0.25],  # placeholder, not measured
    "shift_grid": [0.0, 0.02, 0.05, 0.10, 0.20],  # placeholder, not measured
    "shift_target": "both",  # placeholder, not measured
    "shift_start_update": 1,  # placeholder, not measured
    "seed": 20261110,  # placeholder, not measured
    "n_runs": 1000,  # placeholder, not measured
}


@dataclass(frozen=True)
class SyntheticPool:
    """Synthetic training pool metadata (IDs and gold classes only, never text)."""

    rows: tuple[dict[str, str], ...]
    item_ids: tuple[str, ...]
    gold_by_id: dict[str, str]
    index_by_id: dict[str, int]
    canary_c_ids: tuple[str, ...]
    gold_unsupported_ids: tuple[str, ...]
    item_kind: np.ndarray  # 0=canary_c, 1=unsup_non_c, 2=gold_sup, 3=gold_ct


def build_synthetic_pool(params: Mapping[str, Any]) -> SyntheticPool:
    """Construct the synthetic 394-item pool with C subset of gold-unsupported."""
    n_c = int(params["canary_c_size"])
    n_u = int(params["gold_unsupported_size"])
    n_s = int(params["gold_supported_size"])
    n_ct = int(params["gold_cannot_tell_size"])
    n_pool = int(params["pool_size"])
    if n_c < 1 or n_c > n_u:
        raise ValueError(f"canary_c_size ({n_c}) must be in [1, gold_unsupported_size ({n_u})]")
    if n_s < 0 or n_ct < 0:
        raise ValueError("gold_supported_size and gold_cannot_tell_size must be >= 0")
    if n_u + n_s + n_ct != n_pool:
        raise ValueError(
            f"pool class sizes ({n_u} + {n_s} + {n_ct} = {n_u + n_s + n_ct}) != pool_size ({n_pool})"
        )

    rows: list[dict[str, str]] = []
    canary_c_ids: list[str] = []
    gold_unsup_ids: list[str] = []
    kinds: list[int] = []

    for idx in range(n_c):
        cid = f"syn_c_{idx:03d}"
        rows.append({"id": cid, "label": "unsupported", "split": "tune"})
        canary_c_ids.append(cid)
        gold_unsup_ids.append(cid)
        kinds.append(0)

    for idx in range(n_u - n_c):
        cid = f"syn_u_{idx:03d}"
        rows.append({"id": cid, "label": "unsupported", "split": "tune"})
        gold_unsup_ids.append(cid)
        kinds.append(1)

    for idx in range(n_s):
        cid = f"syn_s_{idx:03d}"
        rows.append({"id": cid, "label": "supported", "split": "tune"})
        kinds.append(2)

    for idx in range(n_ct):
        cid = f"syn_n_{idx:03d}"
        rows.append({"id": cid, "label": "cannot_tell", "split": "tune"})
        kinds.append(3)

    item_ids = tuple(r["id"] for r in rows)
    gold_by_id = {r["id"]: r["label"] for r in rows}
    index_by_id = {cid: i for i, cid in enumerate(item_ids)}
    return SyntheticPool(
        rows=tuple(rows),
        item_ids=item_ids,
        gold_by_id=gold_by_id,
        index_by_id=index_by_id,
        canary_c_ids=tuple(sorted(canary_c_ids)),
        gold_unsupported_ids=tuple(sorted(gold_unsup_ids)),
        item_kind=np.array(kinds, dtype=np.int8),
    )


GroupLookupKey = tuple[str, int, int, int]
GroupLookupEntry = tuple[tuple[str, ...], tuple[bool, ...], tuple[float, ...], bool]


def _build_group_lookup(group_size: int) -> dict[GroupLookupKey, GroupLookupEntry]:
    """Precompute canonical group records using pinned ``s2_grpo_smoke`` functions."""
    table: dict[GroupLookupKey, GroupLookupEntry] = {}
    for gold in VERDICTS:
        for n_sup in range(group_size + 1):
            for n_unsup in range(group_size - n_sup + 1):
                n_ct = group_size - n_sup - n_unsup
                v_tuple = ("supported",) * n_sup + ("unsupported",) * n_unsup + ("cannot_tell",) * n_ct
                outcomes: list[float] = []
                c_flags: list[bool] = []
                for v in v_tuple:
                    score, is_corr = macro(v, gold)
                    outcomes.append(score)
                    c_flags.append(is_corr)
                is_zv = zero_variance(outcomes)
                raw_adv = advantages(outcomes, mode="dr-grpo")
                clamped_adv = clamp_wrong_positive_advantages(raw_adv, c_flags)
                assert_no_wrong_positive_advantages(clamped_adv, c_flags)
                table[(gold, n_sup, n_unsup, n_ct)] = (
                    v_tuple,
                    tuple(c_flags),
                    tuple(clamped_adv),
                    is_zv,
                )
    return table


_GROUP_LOOKUP_CACHE: dict[int, dict[GroupLookupKey, GroupLookupEntry]] = {}


def get_group_lookup(group_size: int) -> dict[GroupLookupKey, GroupLookupEntry]:
    """Return cached canonical group tuple table for ``group_size``."""
    cached = _GROUP_LOOKUP_CACHE.get(group_size)
    if cached is None:
        cached = _build_group_lookup(group_size)
        _GROUP_LOOKUP_CACHE[group_size] = cached
    return cached


def validate_param_file_path(path: Path) -> None:
    """Refuse run outputs, JSONL files, state.json, or sealed/training directories."""
    name_lower = path.name.lower()
    if name_lower.endswith((".jsonl", ".jsonl.gz")) or name_lower == "state.json":
        raise ValueError(f"refusing parameter path {path.name!r}: simulator never reads run output files")
    for part in path.parts:
        part_lower = part.lower()
        if any(part_lower.startswith(pref) for pref in _FORBIDDEN_PARAM_DIR_PREFIXES):
            raise ValueError(
                f"refusing parameter path under {part!r}: simulator never reads sealed or run directories"
            )


def _validate_prob(name: str, val: float) -> float:
    fval = float(val)
    if not (0.0 <= fval <= 1.0):
        raise ValueError(f"{name} must be in [0.0, 1.0], got {val!r}")
    return fval


def validate_and_resolve_parameters(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge ``overrides`` onto ``DEFAULT_PARAMETERS`` and validate every field."""
    merged: dict[str, Any] = dict(DEFAULT_PARAMETERS)
    explicit_flip_better_grid = False
    explicit_forced_look_updates = False
    if overrides:
        unknown = sorted(set(overrides) - set(DEFAULT_PARAMETERS))
        if unknown:
            raise ValueError(f"unknown simulator parameter(s): {', '.join(unknown)}")
        explicit_flip_better_grid = (
            "flip_better_grid" in overrides and overrides["flip_better_grid"] is not None
        )
        explicit_forced_look_updates = (
            "forced_look_updates" in overrides and overrides["forced_look_updates"] is not None
        )
        for k, v in overrides.items():
            if v is not None:
                merged[k] = v

    for prob_key in (
        "step0_canary_c_supported_prob",
        "step0_canary_c_cannot_tell_prob",
        "step0_unsup_supported_prob",
        "step0_unsup_cannot_tell_prob",
        "step0_sup_unsupported_prob",
        "step0_sup_cannot_tell_prob",
        "step0_ct_supported_prob",
        "step0_ct_unsupported_prob",
        "p_flip_worse",
        "worse_cannot_tell_fraction",
        "p_flip_better",
        "look_correlation",
        "group_dispersion",
    ):
        merged[prob_key] = _validate_prob(prob_key, merged[prob_key])

    for pair_name, k1, k2 in (
        ("step0_canary_c", "step0_canary_c_supported_prob", "step0_canary_c_cannot_tell_prob"),
        ("step0_unsup", "step0_unsup_supported_prob", "step0_unsup_cannot_tell_prob"),
        ("step0_sup", "step0_sup_unsupported_prob", "step0_sup_cannot_tell_prob"),
        ("step0_ct", "step0_ct_supported_prob", "step0_ct_unsupported_prob"),
    ):
        if merged[k1] + merged[k2] > 1.0 + 1e-12:
            raise ValueError(f"{pair_name} bad probabilities sum to > 1.0: {merged[k1]} + {merged[k2]}")

    merged["ct_drift_per_update"] = float(merged["ct_drift_per_update"])
    if merged["ct_drift_per_update"] < 0.0:
        raise ValueError(f"ct_drift_per_update must be >= 0.0, got {merged['ct_drift_per_update']!r}")

    if merged["cusum_on_confirm_negative"] not in CUSUM_NEGATIVE_POLICIES:
        raise ValueError(
            f"cusum_on_confirm_negative must be one of {CUSUM_NEGATIVE_POLICIES}, "
            f"got {merged['cusum_on_confirm_negative']!r}"
        )
    if merged["shift_target"] not in SHIFT_TARGETS:
        raise ValueError(f"shift_target must be one of {SHIFT_TARGETS}, got {merged['shift_target']!r}")

    for pos_int_key in (
        "pool_size",
        "canary_c_size",
        "gold_unsupported_size",
        "updates",
        "groups_per_update",
        "window_groups",
        "group_size",
        "n_runs",
    ):
        ival = int(merged[pos_int_key])
        if ival < 1:
            raise ValueError(f"{pos_int_key} must be >= 1, got {merged[pos_int_key]!r}")
        merged[pos_int_key] = ival

    merged["gold_supported_size"] = int(merged["gold_supported_size"])
    merged["gold_cannot_tell_size"] = int(merged["gold_cannot_tell_size"])
    merged["seed"] = int(merged["seed"])
    merged["shift_start_update"] = int(merged["shift_start_update"])
    if not (1 <= merged["shift_start_update"] <= merged["updates"]):
        raise ValueError(
            f"shift_start_update must be in [1, {merged['updates']}], got {merged['shift_start_update']!r}"
        )

    if not explicit_forced_look_updates and overrides and "updates" in overrides:
        look_updates = sorted({min(merged["updates"], int(u)) for u in merged["forced_look_updates"]})
    else:
        look_updates = [int(u) for u in merged["forced_look_updates"]]
    if not look_updates or any(u < 1 or u > merged["updates"] for u in look_updates):
        raise ValueError(f"forced_look_updates must be non-empty ints in [1, {merged['updates']}]")
    merged["forced_look_updates"] = sorted(look_updates)

    look_rules = [str(r) for r in merged["forced_look_rules"]]
    if not look_rules or any(r not in CONFIRM_RULES for r in look_rules):
        raise ValueError(f"forced_look_rules must be a non-empty subset of {CONFIRM_RULES}")
    merged["forced_look_rules"] = look_rules

    fw_grid = [_validate_prob("flip_worse_grid item", x) for x in merged["flip_worse_grid"]]
    if not fw_grid:
        raise ValueError("flip_worse_grid must be non-empty")
    merged["flip_worse_grid"] = fw_grid

    if not explicit_flip_better_grid and overrides and "p_flip_better" in overrides:
        merged["flip_better_grid"] = [merged["p_flip_better"]]
    else:
        fb_grid = [_validate_prob("flip_better_grid item", x) for x in merged["flip_better_grid"]]
        if not fb_grid:
            raise ValueError("flip_better_grid must be non-empty")
        merged["flip_better_grid"] = fb_grid

    s_grid = [_validate_prob("shift_grid item", x) for x in merged["shift_grid"]]
    if not s_grid:
        raise ValueError("shift_grid must be non-empty")
    merged["shift_grid"] = s_grid

    build_synthetic_pool(merged)
    return merged


# Verdict integer codes: 0 = "supported", 1 = "unsupported", 2 = "cannot_tell"
_VERDICT_STR = ("supported", "unsupported", "cannot_tell")


def _draw_step0_states(
    pool: SyntheticPool, params: Mapping[str, Any], rng: np.random.Generator
) -> np.ndarray:
    """Draw step-0 discrete latent verdict states in {0, 1, 2} for every pool item."""
    n = len(pool.item_ids)
    u = rng.random(n)
    states = np.empty(n, dtype=np.int8)
    kind = pool.item_kind

    # Kind 0: canary C (gold unsupported -> good=1, bad=0 or 2)
    m0 = kind == 0
    p0_sup = float(params["step0_canary_c_supported_prob"])
    p0_ct = float(params["step0_canary_c_cannot_tell_prob"])
    states[m0] = np.where(u[m0] < p0_sup, 0, np.where(u[m0] < p0_sup + p0_ct, 2, 1))

    # Kind 1: non-canary gold unsupported (good=1, bad=0 or 2)
    m1 = kind == 1
    p1_sup = float(params["step0_unsup_supported_prob"])
    p1_ct = float(params["step0_unsup_cannot_tell_prob"])
    states[m1] = np.where(u[m1] < p1_sup, 0, np.where(u[m1] < p1_sup + p1_ct, 2, 1))

    # Kind 2: gold supported (good=0, bad=1 or 2)
    m2 = kind == 2
    p2_unsup = float(params["step0_sup_unsupported_prob"])
    p2_ct = float(params["step0_sup_cannot_tell_prob"])
    states[m2] = np.where(u[m2] < p2_unsup, 1, np.where(u[m2] < p2_unsup + p2_ct, 2, 0))

    # Kind 3: gold cannot_tell (good=2, bad=0 or 1)
    m3 = kind == 3
    p3_sup = float(params["step0_ct_supported_prob"])
    p3_unsup = float(params["step0_ct_unsupported_prob"])
    states[m3] = np.where(u[m3] < p3_sup, 0, np.where(u[m3] < p3_sup + p3_unsup, 1, 2))

    return states


def _item_verdict_probs(
    kind: int,
    step0_verdict: int,
    update_1based: int,
    *,
    p_flip_worse: float,
    w_ct: float,
    p_flip_better: float,
    ct_drift_per_update: float,
    shift_size: float,
    shift_target: str,
    shift_start_update: int,
) -> tuple[float, float, float]:
    """Return latent 3-verdict distribution ``(p_sup, p_unsup, p_ct)`` for one item at ``update_1based``."""
    active_shift = shift_size if update_1based >= shift_start_update else 0.0
    drift_ct = ct_drift_per_update * update_1based

    if kind in (0, 1):
        # Gold unsupported: good = 1 (unsupported), bad = 0 (supported) and 2 (cannot_tell)
        if shift_target == "cannot_tell":
            d_sup, d_ct = 0.0, active_shift
        elif shift_target == "supported":
            d_sup, d_ct = active_shift, 0.0
        else:
            d_sup, d_ct = active_shift * (1.0 - w_ct), active_shift * w_ct

        if step0_verdict == 1:
            r_sup = p_flip_worse * (1.0 - w_ct) + d_sup
            r_ct = p_flip_worse * w_ct + drift_ct + d_ct
            r_bad = r_sup + r_ct
            if r_bad <= 0.0:
                return 0.0, 1.0, 0.0
            if r_bad >= 1.0:
                return r_sup / r_bad, 0.0, r_ct / r_bad
            return r_sup, 1.0 - r_bad, r_ct

        tot_push = min(1.0, d_sup + d_ct + drift_ct)
        p_good = max(0.0, min(1.0, p_flip_better * (1.0 - tot_push)))
        rem = 1.0 - p_good
        if step0_verdict == 0:
            to_ct = min(1.0, d_ct + drift_ct)
            return rem * (1.0 - to_ct), p_good, rem * to_ct
        to_sup = min(1.0, d_sup)
        return rem * to_sup, p_good, rem * (1.0 - to_sup)

    if kind == 2:
        # Gold supported: good = 0 (supported), bad = 1 (unsupported) and 2 (cannot_tell)
        if shift_target == "cannot_tell":
            d_unsup, d_ct = 0.0, active_shift
        elif shift_target == "supported":
            d_unsup, d_ct = 0.0, 0.0
        else:
            d_unsup, d_ct = active_shift * (1.0 - w_ct), active_shift * w_ct

        if step0_verdict == 0:
            r_unsup = p_flip_worse * (1.0 - w_ct) + d_unsup
            r_ct = p_flip_worse * w_ct + drift_ct + d_ct
            r_bad = r_unsup + r_ct
            if r_bad <= 0.0:
                return 1.0, 0.0, 0.0
            if r_bad >= 1.0:
                return 0.0, r_unsup / r_bad, r_ct / r_bad
            return 1.0 - r_bad, r_unsup, r_ct

        tot_push = min(1.0, d_unsup + d_ct + drift_ct)
        p_good = max(0.0, min(1.0, p_flip_better * (1.0 - tot_push)))
        rem = 1.0 - p_good
        if step0_verdict == 1:
            to_ct = min(1.0, d_ct + drift_ct)
            return p_good, rem * (1.0 - to_ct), rem * to_ct
        to_unsup = min(1.0, d_unsup)
        return p_good, rem * to_unsup, rem * (1.0 - to_unsup)

    # Kind 3: gold cannot_tell: good = 2 (cannot_tell), other = 0 (supported) and 1 (unsupported)
    if shift_target == "supported":
        d_sup = active_shift
    elif shift_target == "both":
        d_sup = active_shift * (1.0 - w_ct)
    else:
        d_sup = 0.0
    if step0_verdict == 2:
        r_sup = p_flip_worse * (1.0 - w_ct) + d_sup
        r_unsup = p_flip_worse * w_ct
        r_other = r_sup + r_unsup
        if r_other <= 0.0:
            return 0.0, 0.0, 1.0
        if r_other >= 1.0:
            return r_sup / r_other, r_unsup / r_other, 0.0
        return r_sup, r_unsup, 1.0 - r_other

    p_good = max(0.0, min(1.0, p_flip_better * (1.0 - min(1.0, d_sup))))
    rem = 1.0 - p_good
    if step0_verdict == 0:
        return rem, 0.0, p_good
    return 0.0, rem, p_good


class LatentItemPoolState:
    """Per-item 3-verdict latent states shared by training-window rollouts and T=0 confirmations."""

    def __init__(
        self,
        pool: SyntheticPool,
        params: Mapping[str, Any],
        rng: np.random.Generator,
        *,
        p_flip_worse: float | None = None,
        p_flip_better: float | None = None,
        shift_size: float = 0.0,
    ) -> None:
        self.pool = pool
        self.rng = rng
        self.step0_verdicts = _draw_step0_states(pool, params, rng)
        self.current_verdicts = self.step0_verdicts.copy()
        n = len(pool.item_ids)
        self.latent_u = np.zeros(n, dtype=np.float64)
        self.has_looked = np.zeros(n, dtype=bool)
        self.last_confirm_step = np.full(n, -1, dtype=np.int32)
        self.current_probs = np.zeros((n, 3), dtype=np.float64)
        for idx in range(n):
            self.current_probs[idx, int(self.step0_verdicts[idx])] = 1.0

        self.p_flip_worse = float(params["p_flip_worse"] if p_flip_worse is None else p_flip_worse)
        self.w_ct = float(params["worse_cannot_tell_fraction"])
        self.p_flip_better = float(params["p_flip_better"] if p_flip_better is None else p_flip_better)
        self.ct_drift_per_update = float(params["ct_drift_per_update"])
        self.look_correlation = float(params["look_correlation"])
        self.group_dispersion = float(params["group_dispersion"])
        self.shift_size = float(shift_size)
        self.shift_target = str(params["shift_target"])
        self.shift_start_update = int(params["shift_start_update"])

        # Pre-draw uniform buffers in chunks so scalar Python access has minimal overhead.
        self._buf = rng.random(4096).tolist()
        self._buf_idx = 0

    def _next_u(self) -> float:
        if self._buf_idx >= len(self._buf):
            self._buf = self.rng.random(4096).tolist()
            self._buf_idx = 0
        val = self._buf[self._buf_idx]
        self._buf_idx += 1
        return val

    def step0_baseline_items(self) -> dict[str, dict[str, Any]]:
        """Build the step-0 T=0 baseline dict over the 184 gold-unsupported items (including C)."""
        out: dict[str, dict[str, Any]] = {}
        for cid in self.pool.gold_unsupported_ids:
            idx = self.pool.index_by_id[cid]
            v_code = int(self.step0_verdicts[idx])
            out[cid] = {
                "verdict": _VERDICT_STR[v_code],
                "correct": v_code == 1,
                "eval_mode": True,
                "mode": "eval",
            }
        return out

    def look_item(self, idx: int, update_1based: int) -> tuple[int, float, float, float]:
        """Advance item ``idx`` by one correlated look at ``update_1based`` and return ``(v, p0, p1, p2)``."""
        if not self.has_looked[idx]:
            u_val = self._next_u()
            self.latent_u[idx] = u_val
            self.has_looked[idx] = True
        else:
            u_keep = self._next_u()
            if u_keep < self.look_correlation:
                u_val = float(self.latent_u[idx])
            else:
                u_val = self._next_u()
                self.latent_u[idx] = u_val

        p_sup, p_unsup, p_ct = _item_verdict_probs(
            int(self.pool.item_kind[idx]),
            int(self.step0_verdicts[idx]),
            update_1based,
            p_flip_worse=self.p_flip_worse,
            w_ct=self.w_ct,
            p_flip_better=self.p_flip_better,
            ct_drift_per_update=self.ct_drift_per_update,
            shift_size=self.shift_size,
            shift_target=self.shift_target,
            shift_start_update=self.shift_start_update,
        )
        if u_val < p_sup:
            v_code = 0
        elif u_val < p_sup + p_unsup:
            v_code = 1
        else:
            v_code = 2
        self.current_verdicts[idx] = v_code
        self.current_probs[idx, 0] = p_sup
        self.current_probs[idx, 1] = p_unsup
        self.current_probs[idx, 2] = p_ct
        return v_code, p_sup, p_unsup, p_ct

    def sample_training_group_counts(
        self, cid: str, update_1based: int, group_size: int
    ) -> tuple[int, int, int]:
        """Look at ``cid`` at ``update_1based`` and sample ``(n_sup, n_unsup, n_ct)``."""
        idx = self.pool.index_by_id[cid]
        v_code, p_sup, p_unsup, p_ct = self.look_item(idx, update_1based)
        tau = self.group_dispersion
        r_sup = tau * p_sup + ((1.0 - tau) if v_code == 0 else 0.0)
        r_unsup = tau * p_unsup + ((1.0 - tau) if v_code == 1 else 0.0)

        if r_sup >= 1.0 - 1e-15:
            return group_size, 0, 0
        if r_unsup >= 1.0 - 1e-15:
            return 0, group_size, 0
        if (tau * p_ct + ((1.0 - tau) if v_code == 2 else 0.0)) >= 1.0 - 1e-15:
            return 0, 0, group_size

        n_sup = 0
        n_unsup = 0
        n_ct = 0
        thresh_unsup = r_sup + r_unsup
        for _ in range(group_size):
            u_s = self._next_u()
            if u_s < r_sup:
                n_sup += 1
            elif u_s < thresh_unsup:
                n_unsup += 1
            else:
                n_ct += 1
        return n_sup, n_unsup, n_ct

    def run_t0_confirmation_pass(self, item_ids: Sequence[str], step: int) -> dict[str, dict[str, Any]]:
        """Produce a T=0 greedy pass over ``item_ids`` from the same per-item latent states."""
        update_1based = int(step) + 1
        out: dict[str, dict[str, Any]] = {}
        for cid in item_ids:
            idx = self.pool.index_by_id[str(cid)]
            if int(self.last_confirm_step[idx]) == int(step):
                v_code = int(self.current_verdicts[idx])
            else:
                v_code, _, _, _ = self.look_item(idx, update_1based)
                self.last_confirm_step[idx] = int(step)
            gold = self.pool.gold_by_id[str(cid)]
            v_str = _VERDICT_STR[v_code]
            _, is_corr = macro(v_str, gold)
            out[str(cid)] = {
                "verdict": v_str,
                "correct": is_corr,
                "eval_mode": True,
                "mode": "eval",
            }
        return out


@dataclass(frozen=True)
class EndToEndRunOutcome:
    """Summary of one simulated end-to-end run."""

    verdict: str
    halted: bool
    halt_update: int | None
    confirm_trips_total: int
    confirmations_used: int
    runner_calls: int
    had_shared_pass: bool
    final_fuse_a_s: float
    final_channel_b_s: float
    guard: CtCollapseGuard | None


def simulate_end_to_end_run(
    pool: SyntheticPool,
    params: Mapping[str, Any],
    *,
    run_seed: int,
    shift_size: float = 0.0,
    retain_guard: bool = False,
) -> EndToEndRunOutcome:
    """Simulate one 197-update run driving the pinned ``CtCollapseGuard`` class."""
    ss = np.random.SeedSequence(int(run_seed))
    child_seeds = ss.spawn(2)
    draw_seed = int(child_seeds[0].generate_state(1, dtype=np.uint32)[0])
    rng = np.random.default_rng(child_seeds[1])

    updates = int(params["updates"])
    groups_per_update = int(params["groups_per_update"])
    window_groups = int(params["window_groups"])
    group_size = int(params["group_size"])
    reset_on_neg = str(params["cusum_on_confirm_negative"]) == "reset"
    total_draws = updates * groups_per_update

    schedule, _ = build_class_stratified_schedule(pool.rows, total_draws, seed=draw_seed)
    latent = LatentItemPoolState(pool, params, rng, shift_size=shift_size)
    group_lookup = get_group_lookup(group_size)

    runner_counter = {"calls": 0}

    def _confirm_runner(
        ids: Sequence[str], step: int = 0, _rule: str = ""
    ) -> Mapping[str, Mapping[str, Any]]:
        runner_counter["calls"] += 1
        return latent.run_t0_confirmation_pass(ids, int(step))

    guard = CtCollapseGuard(
        window_groups=window_groups,
        mode="halt",
        s2_v11=True,
        canary_c_ids=pool.canary_c_ids,
        gold_unsupported_ids=pool.gold_unsupported_ids,
        confirm_runner=_confirm_runner,
    )
    guard.record_step0_baseline(latent.step0_baseline_items())

    step_guard_stop: dict[str, Any] | None = None
    halt_update: int | None = None
    next_step = 0

    for step in range(updates):
        update_1based = step + 1
        base_idx = step * groups_per_update
        for k in range(groups_per_update):
            row = schedule[base_idx + k]
            cid = row["id"]
            gold = row["label"]
            n_sup, n_unsup, n_ct = latent.sample_training_group_counts(cid, update_1based, group_size)
            v_tuple, c_tuple, adv_tuple, is_zv = group_lookup[(gold, n_sup, n_unsup, n_ct)]
            prev_win_idx = guard.window_index
            guard_stop = guard.record_group(
                step=step,
                group=k,
                item_id=cid,
                gold_verdict=gold,
                verdicts=v_tuple,
                correct_flags=c_tuple,
                advantages_list=adv_tuple,
                is_zero_variance=is_zv,
            )
            if guard.window_index > prev_win_idx and reset_on_neg:
                for ev in guard.window_history[-1]["confirm_events"]:
                    if ev.get("status") == "CONFIRM-NEGATIVE":
                        if ev.get("rule") == "FUSE_A":
                            guard.fuse_a_s = 0.0
                        elif ev.get("rule") == "CHANNEL_B":
                            guard.channel_b_s = 0.0
            if guard_stop is not None:
                step_guard_stop = guard_stop

        if step_guard_stop is not None:
            next_step = step
            halt_update = update_1based
            break
        next_step = step + 1

    verdict = train_verdict_line(
        stopped=step_guard_stop,
        oom=False,
        next_step=next_step,
        steps_total=updates,
        chunk_stop=None,
        fits=True,
        dry=True,
    )
    had_shared = any(
        d.get("kind") == "guard_confirmation" and bool(d.get("shared_pass"))
        for d in guard.decision_history
    )
    return EndToEndRunOutcome(
        verdict=verdict,
        halted=step_guard_stop is not None,
        halt_update=halt_update,
        confirm_trips_total=guard.confirm_trips_total,
        confirmations_used=guard.confirmations_used,
        runner_calls=runner_counter["calls"],
        had_shared_pass=had_shared,
        final_fuse_a_s=guard.fuse_a_s,
        final_channel_b_s=guard.channel_b_s,
        guard=guard if retain_guard else None,
    )


def evaluate_cusum_window_step(prev_s: float, count: int, k: float, h: float) -> tuple[float, bool]:
    """Update one CUSUM step via ``s2_v11_config.update_poisson_cusum`` and return ``(s, tripped)``."""
    next_s = update_poisson_cusum(prev_s, count, k)
    return next_s, next_s >= (h - 1e-12)


def _evaluate_forced_pass(
    step0_items: Mapping[str, Mapping[str, Any]],
    pass_t: Mapping[str, Mapping[str, Any]],
    canary_c_ids: Sequence[str],
    gold_unsupported_ids: Sequence[str],
) -> dict[str, bool]:
    """Evaluate the pinned Fuse A, Channel B, and U1 confirmation rules on one T=0 pass."""
    ct_step0 = sum(1 for cid in canary_c_ids if step0_items[cid]["verdict"] == "cannot_tell")
    ct_now = sum(1 for cid in canary_c_ids if pass_t[cid]["verdict"] == "cannot_tell")
    ct_inc = ct_now - ct_step0
    fuse_a_ok = ct_now >= S2V11.fuse_a_confirm_min_ct and ct_inc >= S2V11.fuse_a_confirm_min_increase

    out = {"FUSE_A": bool(fuse_a_ok)}
    for rule, target_verdict in (("CHANNEL_B", "supported"), ("U1", "cannot_tell")):
        b_worse = 0
        c_better = 0
        for cid in gold_unsupported_ids:
            was_bad = step0_items[cid]["verdict"] == target_verdict
            is_bad = pass_t[cid]["verdict"] == target_verdict
            if (not was_bad) and is_bad:
                b_worse += 1
            elif was_bad and (not is_bad):
                c_better += 1
        p_val = mcnemar_one_sided_exact_p(b_worse, c_better)
        out[rule] = bool((b_worse > c_better) and (p_val < S2V11.channel_b_confirm_alpha))
    return out


def run_forced_looks_mode(
    params: Mapping[str, Any],
    *,
    base_sha: str = DEFAULT_BASE_SHA,
) -> dict[str, Any]:
    """Mode (a): fixed confirmation passes at fixed updates over a grid of flip probabilities."""
    resolved = validate_and_resolve_parameters(params)
    pool = build_synthetic_pool(resolved)
    master_seed = int(resolved["seed"])
    n_runs = int(resolved["n_runs"])
    look_updates = list(resolved["forced_look_updates"])
    active_rules = list(resolved["forced_look_rules"])
    fw_grid = list(resolved["flip_worse_grid"])
    fb_grid = list(resolved["flip_better_grid"])

    total_cells = len(fw_grid) * len(fb_grid)
    cell_seqs = np.random.SeedSequence(master_seed).spawn(total_cells)
    cells: list[dict[str, Any]] = []
    cell_idx = 0

    for p_worse in fw_grid:
        for p_better in fb_grid:
            run_seqs = cell_seqs[cell_idx].spawn(n_runs)
            cell_idx += 1
            confirmed_any = 0
            confirmed_by_rule = {r: 0 for r in CONFIRM_RULES}

            for r_idx in range(n_runs):
                rng = np.random.default_rng(run_seqs[r_idx])
                latent = LatentItemPoolState(
                    pool,
                    resolved,
                    rng,
                    p_flip_worse=p_worse,
                    p_flip_better=p_better,
                )
                step0_items = latent.step0_baseline_items()
                ever_rule = {r: False for r in CONFIRM_RULES}
                for upd in look_updates:
                    pass_t = latent.run_t0_confirmation_pass(pool.gold_unsupported_ids, upd - 1)
                    res = _evaluate_forced_pass(
                        step0_items,
                        pass_t,
                        pool.canary_c_ids,
                        pool.gold_unsupported_ids,
                    )
                    for r_name, did_confirm in res.items():
                        if did_confirm:
                            ever_rule[r_name] = True
                for r_name in CONFIRM_RULES:
                    if ever_rule[r_name]:
                        confirmed_by_rule[r_name] += 1
                if any(ever_rule[r_name] for r_name in active_rules):
                    confirmed_any += 1

            cells.append(
                {
                    "p_flip_worse": p_worse,
                    "p_flip_better": p_better,
                    "n_runs": n_runs,
                    "runs_confirmed_at_least_once": confirmed_any,
                    "runs_confirmed_at_least_once_by_rule": confirmed_by_rule,
                }
            )

    return {
        "label": RESULT_LABEL,
        "mode": "forced_looks",
        "base_sha": str(base_sha),
        "seed": master_seed,
        "n_runs": n_runs,
        "parameters": resolved,
        "grid_counts": cells,
    }


def _window_evaluation_updates(updates: int, groups_per_update: int, window_groups: int) -> list[int]:
    """Return 1-based update indices where a full ``window_groups`` window completes."""
    eval_updates: list[int] = []
    total = 0
    for upd in range(1, updates + 1):
        for _ in range(groups_per_update):
            total += 1
            if total % window_groups == 0 and (not eval_updates or eval_updates[-1] != upd):
                eval_updates.append(upd)
    return eval_updates


def run_end_to_end_mode(
    params: Mapping[str, Any],
    *,
    base_sha: str = DEFAULT_BASE_SHA,
) -> dict[str, Any]:
    """Mode (b): full S2 v1.1 guard chain reporting integer counts by verdict."""
    resolved = validate_and_resolve_parameters(params)
    pool = build_synthetic_pool(resolved)
    master_seed = int(resolved["seed"])
    n_runs = int(resolved["n_runs"])
    run_seeds = np.random.SeedSequence(master_seed).generate_state(n_runs, dtype=np.uint64)

    verdict_counts: dict[str, int] = {v: 0 for v in ALL_END_TO_END_VERDICTS}
    runs_halted = 0
    runs_completed = 0
    runs_with_shared_pass = 0
    total_confirmations_used = 0
    total_guard_trips = 0

    for r_idx in range(n_runs):
        outcome = simulate_end_to_end_run(pool, resolved, run_seed=int(run_seeds[r_idx]))
        verdict_counts[outcome.verdict] = verdict_counts.get(outcome.verdict, 0) + 1
        if outcome.halted:
            runs_halted += 1
        else:
            runs_completed += 1
        if outcome.had_shared_pass:
            runs_with_shared_pass += 1
        total_confirmations_used += outcome.confirmations_used
        total_guard_trips += outcome.confirm_trips_total

    return {
        "label": RESULT_LABEL,
        "mode": "end_to_end",
        "base_sha": str(base_sha),
        "seed": master_seed,
        "n_runs": n_runs,
        "parameters": resolved,
        "verdict_counts": verdict_counts,
        "runs_completed": runs_completed,
        "runs_halted": runs_halted,
        "runs_with_shared_pass": runs_with_shared_pass,
        "total_guard_trips": total_guard_trips,
        "total_confirmations_used": total_confirmations_used,
    }


def run_shift_grid_mode(
    params: Mapping[str, Any],
    *,
    base_sha: str = DEFAULT_BASE_SHA,
) -> dict[str, Any]:
    """Mode (c): full S2 v1.1 guard chain over a grid of true shift sizes, counts by update."""
    resolved = validate_and_resolve_parameters(params)
    pool = build_synthetic_pool(resolved)
    master_seed = int(resolved["seed"])
    n_runs = int(resolved["n_runs"])
    shift_grid = list(resolved["shift_grid"])
    eval_updates = _window_evaluation_updates(
        int(resolved["updates"]),
        int(resolved["groups_per_update"]),
        int(resolved["window_groups"]),
    )

    cell_seqs = np.random.SeedSequence(master_seed).spawn(len(shift_grid))
    cells: list[dict[str, Any]] = []

    for c_idx, shift_val in enumerate(shift_grid):
        run_seeds = cell_seqs[c_idx].generate_state(n_runs, dtype=np.uint64)
        by_update: dict[str, int] = {str(u): 0 for u in eval_updates}
        verdict_counts: dict[str, int] = {v: 0 for v in ALL_END_TO_END_VERDICTS}
        detected_total = 0
        undetected_completed = 0

        for r_idx in range(n_runs):
            outcome = simulate_end_to_end_run(
                pool,
                resolved,
                run_seed=int(run_seeds[r_idx]),
                shift_size=shift_val,
            )
            verdict_counts[outcome.verdict] = verdict_counts.get(outcome.verdict, 0) + 1
            if outcome.halted and outcome.halt_update is not None:
                detected_total += 1
                key = str(outcome.halt_update)
                by_update[key] = by_update.get(key, 0) + 1
            else:
                undetected_completed += 1

        cells.append(
            {
                "shift_size": shift_val,
                "n_runs": n_runs,
                "detected_total": detected_total,
                "undetected_completed": undetected_completed,
                "detection_counts_by_update": by_update,
                "verdict_counts": verdict_counts,
            }
        )

    return {
        "label": RESULT_LABEL,
        "mode": "shift_grid",
        "base_sha": str(base_sha),
        "seed": master_seed,
        "n_runs": n_runs,
        "parameters": resolved,
        "shift_cells": cells,
    }


def run_simulation(
    mode: str,
    params: Mapping[str, Any] | None = None,
    *,
    base_sha: str = DEFAULT_BASE_SHA,
) -> dict[str, Any]:
    """Dispatch simulation ``mode`` and return its result dictionary."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode == "forced_looks":
        return run_forced_looks_mode(params or {}, base_sha=base_sha)
    if mode == "end_to_end":
        return run_end_to_end_mode(params or {}, base_sha=base_sha)
    return run_shift_grid_mode(params or {}, base_sha=base_sha)


def write_result_file(result: Mapping[str, Any], out_path: Path) -> None:
    """Write simulation ``result`` JSON with ``RESULT_LABEL`` guaranteed first."""
    if result.get("label") != RESULT_LABEL:
        raise ValueError(f"result missing required label {RESULT_LABEL!r}")
    ordered = {"label": RESULT_LABEL, **{k: v for k, v in result.items() if k != "label"}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8", newline="\n")


def format_printed_summary(result: Mapping[str, Any], *, wall_s: float) -> str:
    """Format stdout summary with ``RESULT_LABEL`` on line 1 and integer counts only."""
    mode = str(result["mode"])
    summary_line: dict[str, Any] = {
        "mode": mode,
        "seed": result["seed"],
        "n_runs": result["n_runs"],
        "wall_s": round(float(wall_s), 3),
    }
    if mode == "end_to_end":
        summary_line["runs_completed"] = result["runs_completed"]
        summary_line["runs_halted"] = result["runs_halted"]
        summary_line["verdict_counts"] = result["verdict_counts"]
        summary_line["total_guard_trips"] = result["total_guard_trips"]
        summary_line["total_confirmations_used"] = result["total_confirmations_used"]
    elif mode == "forced_looks":
        summary_line["grid_cells"] = len(result["grid_counts"])
        summary_line["confirmed_counts"] = [
            c["runs_confirmed_at_least_once"] for c in result["grid_counts"]
        ]
    else:
        summary_line["shift_cells"] = len(result["shift_cells"])
        summary_line["detected_counts"] = [c["detected_total"] for c in result["shift_cells"]]
    return f"{RESULT_LABEL}\n{json.dumps(summary_line)}"


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _parse_str_list(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    """CLI parser for ``sim_guard_chain.py``."""
    ap = argparse.ArgumentParser(description="CPU-only simulator for the S2 v1.1 guard chain.")
    ap.add_argument("--mode", choices=MODES, default="end_to_end")
    ap.add_argument("--params", type=Path, default=None, help="hand-authored JSON parameter file")
    ap.add_argument("--out", type=Path, required=True, help="JSON result file path to write")
    ap.add_argument("--base-sha", default=DEFAULT_BASE_SHA)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-runs", type=int, default=None)
    ap.add_argument("--p-flip-worse", type=float, default=None)
    ap.add_argument("--worse-cannot-tell-fraction", type=float, default=None)
    ap.add_argument("--p-flip-better", type=float, default=None)
    ap.add_argument("--ct-drift-per-update", type=float, default=None)
    ap.add_argument("--look-correlation", type=float, default=None)
    ap.add_argument("--group-dispersion", type=float, default=None)
    ap.add_argument("--step0-canary-c-supported-prob", type=float, default=None)
    ap.add_argument("--step0-canary-c-cannot-tell-prob", type=float, default=None)
    ap.add_argument("--step0-unsup-supported-prob", type=float, default=None)
    ap.add_argument("--step0-unsup-cannot-tell-prob", type=float, default=None)
    ap.add_argument("--step0-sup-unsupported-prob", type=float, default=None)
    ap.add_argument("--step0-sup-cannot-tell-prob", type=float, default=None)
    ap.add_argument("--step0-ct-supported-prob", type=float, default=None)
    ap.add_argument("--step0-ct-unsupported-prob", type=float, default=None)
    ap.add_argument("--cusum-on-confirm-negative", choices=CUSUM_NEGATIVE_POLICIES, default=None)
    ap.add_argument("--forced-look-updates", type=_parse_int_list, default=None)
    ap.add_argument("--forced-look-rules", type=_parse_str_list, default=None)
    ap.add_argument("--flip-worse-grid", type=_parse_float_list, default=None)
    ap.add_argument("--flip-better-grid", type=_parse_float_list, default=None)
    ap.add_argument("--shift-grid", type=_parse_float_list, default=None)
    ap.add_argument("--shift-target", choices=SHIFT_TARGETS, default=None)
    ap.add_argument("--shift-start-update", type=int, default=None)
    return ap


_CLI_PARAM_KEYS = (
    "seed",
    "n_runs",
    "p_flip_worse",
    "worse_cannot_tell_fraction",
    "p_flip_better",
    "ct_drift_per_update",
    "look_correlation",
    "group_dispersion",
    "step0_canary_c_supported_prob",
    "step0_canary_c_cannot_tell_prob",
    "step0_unsup_supported_prob",
    "step0_unsup_cannot_tell_prob",
    "step0_sup_unsupported_prob",
    "step0_sup_cannot_tell_prob",
    "step0_ct_supported_prob",
    "step0_ct_unsupported_prob",
    "cusum_on_confirm_negative",
    "forced_look_updates",
    "forced_look_rules",
    "flip_worse_grid",
    "flip_better_grid",
    "shift_grid",
    "shift_target",
    "shift_start_update",
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI simulation, write JSON result file, and print labeled summary."""
    args = build_parser().parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.params is not None:
        validate_param_file_path(args.params)
        loaded = json.loads(args.params.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("parameter JSON file must contain a JSON object")
        overrides.update(loaded)
    for key in _CLI_PARAM_KEYS:
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val

    t0 = time.perf_counter()
    result = run_simulation(args.mode, overrides, base_sha=args.base_sha)
    wall_s = time.perf_counter() - t0
    write_result_file(result, args.out)
    print(format_printed_summary(result, wall_s=wall_s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
