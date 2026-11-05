"""Prompt draw rule for Stage 2 GRPO and the full-run trainer.

Provides ``draw_prompts`` in a standalone sibling module beside ``s2_grpo_smoke.py`` so both
``examples/verifier-distill/s2_grpo_smoke.py`` and a future sibling full-run trainer can import
``from s2_draw import draw_prompts`` without re-implementing ``uniform`` or ``kband``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path

PROMPTS = 32
PROMPT_SPLIT = "tune"
DRAW_MODES = ("first", "uniform", "kband", "class_stratified")
HOLDOUT_DEFAULT_SEED = "pr79-holdout-v1"
HOLDOUT_DEFAULT_FRACTION = 0.20
GOLD_CLASSES = ("supported", "unsupported", "cannot_tell")
PASS_A_BANDS = ("[0,5]", "[6,7]", "8")


def k_band_of(k_correct: int) -> str:
    """Return the pre-registered Pass-A difficulty band ('[0,5]', '[6,7]', or '8') for ``k_correct``."""
    k_val = int(k_correct)
    if k_val <= 5:
        return "[0,5]"
    if k_val <= 7:
        return "[6,7]"
    return "8"


def select_prompts(rows: list[dict], n: int | None = PROMPTS, split: str = PROMPT_SPLIT) -> list[dict]:
    """The rule: the first n rows of the tune split by sorted claim id (every row when n is None)."""
    mine = sorted((r for r in rows if r.get("split") == split), key=lambda r: r["id"])
    if n is None:
        if not mine:
            raise SystemExit(f"the gold has 0 {split} claims; S2 needs at least 1")
        return mine
    if len(mine) < n:
        raise SystemExit(f"the gold has {len(mine)} {split} claims; S2 needs {n}")
    return mine[:n]


def ids_sha256(ids: Sequence[str]) -> str:
    """SHA-256 hex digest of ``ids`` joined by ``\\n``."""
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _refuse(reason: str, detail: str) -> None:
    raise SystemExit(f"refusing draw ({reason}): {detail}")


def _check_unused_draw_flags(
    mode: str,
    *,
    seed: int | None,
    prepass,
    k_min: int | None,
    k_max: int | None,
    holdout_seed: str | int | None = None,
) -> None:
    unused: list[str] = []
    if mode == "first" and seed is not None:
        unused.append("--draw-seed")
    if mode in ("first", "uniform"):
        has_valid_holdout_seed = (
            holdout_seed is not None
            and not isinstance(holdout_seed, bool)
            and bool(str(holdout_seed).strip())
        )
        if isinstance(prepass, (tuple, list)) and len(prepass) == 2:
            if not (has_valid_holdout_seed and prepass[0] is not None and prepass[1] is not None):
                if prepass[0] is not None:
                    unused.append("--prepass-jsonl")
                if prepass[1] is not None:
                    unused.append("--prepass-sha256")
                if prepass[0] is None and prepass[1] is None:
                    unused.append("--prepass")
        elif prepass is not None and not has_valid_holdout_seed:
            unused.append("--prepass")
        if k_min is not None:
            unused.append("--k-min")
        if k_max is not None:
            unused.append("--k-max")
    elif mode == "class_stratified":
        if k_min is not None:
            unused.append("--k-min")
        if k_max is not None:
            unused.append("--k-max")
    if unused:
        _refuse("unused_draw_flags", f"--draw {mode} cannot be used with {', '.join(unused)}")


def _parse_prepass_spec(prepass) -> tuple[Path, str]:
    """Extract ``(path, expected_sha256)`` from ``prepass``."""
    if prepass is None:
        _refuse("prepass_missing", "kband requires a pre-pass JSONL path and expected sha256")
    if isinstance(prepass, dict):
        raw_path = prepass.get("path") or prepass.get("jsonl")
        raw_sha = prepass.get("sha256") or prepass.get("expected_sha256")
    elif isinstance(prepass, (tuple, list)) and len(prepass) == 2:
        raw_path, raw_sha = prepass
    else:
        _refuse("prepass_invalid", "prepass must be (path, expected_sha256) or {'path': ..., 'sha256': ...}")
    if not raw_path:
        _refuse("prepass_missing_file", "prepass JSONL path is missing")
    if not raw_sha or not isinstance(raw_sha, str):
        _refuse("prepass_sha_missing", "expected prepass sha256 is missing")
    return Path(raw_path), raw_sha.strip().lower()


def _check_prepass_companion_headers(
    path: Path,
    allowed_seeds: set[int],
) -> tuple[bool, bool]:
    """Refuse if any companion header beside ``path`` has ``descriptive: true`` or wrong ``seed``."""
    candidates = [
        path.with_name(path.stem + "-header.json"),
        path.parent / "s2-hf-prepass-header.json",
        path.parent / "s2-llamacpp-prepass-header.json",
    ]
    seen: set[Path] = set()
    header_has_seed = False
    header_has_descriptive = False
    for hdr_path in candidates:
        if hdr_path in seen or not hdr_path.is_file():
            continue
        seen.add(hdr_path)
        try:
            hdr_obj = json.loads(hdr_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(hdr_obj, dict):
            continue
        if hdr_obj.get("descriptive") is True or (
            "descriptive" in hdr_obj and not isinstance(hdr_obj["descriptive"], bool)
        ):
            _refuse(
                "prepass_descriptive",
                "pre-pass header has descriptive=true; descriptive passes cannot be used",
            )
        if "descriptive" in hdr_obj and hdr_obj["descriptive"] is False:
            header_has_descriptive = True
        if "seed" in hdr_obj and hdr_obj["seed"] is not None:
            if (
                isinstance(hdr_obj["seed"], bool)
                or not isinstance(hdr_obj["seed"], int)
                or int(hdr_obj["seed"]) not in allowed_seeds
            ):
                _refuse(
                    "prepass_wrong_seed",
                    f"pre-pass header has seed={hdr_obj['seed']!r}, expected {sorted(allowed_seeds)}",
                )
            header_has_seed = True
            if "ids_sha256" in hdr_obj and "descriptive" not in hdr_obj:
                header_has_descriptive = True
    return header_has_seed, header_has_descriptive


def _load_and_verify_prepass(
    prepass_spec,
    split_ids: set[str],
    *,
    expected_seed: int | Sequence[int] = 0,
) -> tuple[dict[str, dict], str]:
    """Load pre-pass JSONL, verify SHA-256 and split id coverage, and return ``(records_by_id, sha256)``."""
    path, expected_sha = _parse_prepass_spec(prepass_spec)
    if not path.is_file():
        _refuse("prepass_missing_file", f"pre-pass file does not exist: {path.name}")
    allowed_seeds = (
        {int(expected_seed)}
        if isinstance(expected_seed, int)
        else {int(s) for s in expected_seed}
    )
    header_has_seed, header_has_descriptive = _check_prepass_companion_headers(path, allowed_seeds)
    raw_bytes = path.read_bytes()
    actual_sha = hashlib.sha256(raw_bytes).hexdigest()
    if actual_sha != expected_sha:
        _refuse(
            "prepass_sha_mismatch",
            f"expected sha256 {expected_sha}, got {actual_sha}",
        )
    records_by_id: dict[str, dict] = {}
    for line_no, raw_line in enumerate(raw_bytes.decode("utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            _refuse("prepass_corrupt_json", f"invalid JSON on line {line_no}")
        if not isinstance(obj, dict) or "id" not in obj or "k_correct" not in obj:
            _refuse("prepass_invalid_record", f"line {line_no} is missing 'id' or 'k_correct'")
        if obj.get("descriptive") is True or (
            "descriptive" in obj and not isinstance(obj["descriptive"], bool)
        ):
            _refuse(
                "prepass_descriptive",
                f"line {line_no} has descriptive=true; descriptive passes cannot be used",
            )
        rec_has_descriptive = "descriptive" in obj and obj["descriptive"] is False
        rec_has_seed = False
        for sk in ("pass_seed", "seed"):
            if sk in obj and obj[sk] is not None:
                if (
                    isinstance(obj[sk], bool)
                    or not isinstance(obj[sk], int)
                    or int(obj[sk]) not in allowed_seeds
                ):
                    _refuse(
                        "prepass_wrong_seed",
                        f"line {line_no} has {sk}={obj[sk]!r}, expected {sorted(allowed_seeds)}",
                    )
                rec_has_seed = True
        rec_id = str(obj["id"])
        if rec_id in records_by_id:
            _refuse("prepass_duplicate_id", f"duplicate pre-pass record for id {rec_id!r}")
        if rec_id not in split_ids:
            _refuse("prepass_extra_id", f"pre-pass id {rec_id!r} is not in the split")
        if not (header_has_seed or rec_has_seed):
            _refuse(
                "prepass_missing_seed",
                f"line {line_no} and companion header are missing seed provenance",
            )
        if not (header_has_descriptive or rec_has_descriptive):
            _refuse(
                "prepass_missing_descriptive",
                f"line {line_no} and companion header are missing descriptive provenance",
            )
        records_by_id[rec_id] = obj

    missing = sorted(split_ids - set(records_by_id))
    if missing:
        _refuse(
            "prepass_missing_split_id",
            f"split row {missing[0]!r} has no pre-pass record ({len(missing)} missing)",
        )
    return records_by_id, actual_sha


def _extract_k_by_id(
    prepass_records_or_spec,
    split_ids: set[str],
) -> tuple[dict[str, int], str | None]:
    """Resolve ``prepass_records_or_spec`` into ``(k_by_id, prepass_sha256)``."""
    if isinstance(prepass_records_or_spec, Mapping) and not (
        "path" in prepass_records_or_spec or "jsonl" in prepass_records_or_spec
    ):
        k_by_id: dict[str, int] = {}
        for k_id, val in prepass_records_or_spec.items():
            cid = str(k_id)
            if isinstance(val, Mapping):
                if val.get("descriptive") is True or (
                    "descriptive" in val and not isinstance(val["descriptive"], bool)
                ):
                    _refuse(
                        "prepass_descriptive",
                        f"record {cid!r} has descriptive=true; descriptive passes cannot be used",
                    )
                for sk in ("pass_seed", "seed"):
                    if sk in val and val[sk] is not None and int(val[sk]) != 0:
                        _refuse(
                            "prepass_wrong_seed",
                            f"record {cid!r} has {sk}={val[sk]!r}, expected 0",
                        )
                k_by_id[cid] = int(val["k_correct"])
            else:
                k_by_id[cid] = int(val)
        missing = sorted(split_ids - set(k_by_id))
        if missing:
            _refuse(
                "prepass_missing_split_id",
                f"split row {missing[0]!r} has no pre-pass record ({len(missing)} missing)",
            )
        return k_by_id, None
    records_by_id, actual_sha = _load_and_verify_prepass(
        prepass_records_or_spec, split_ids, expected_seed=0
    )
    return {cid: int(rec["k_correct"]) for cid, rec in records_by_id.items()}, actual_sha


def split_stratified_holdout(
    rows: Sequence[dict],
    prepass_records_or_spec,
    *,
    holdout_fraction: float = HOLDOUT_DEFAULT_FRACTION,
    holdout_seed: str | int | None = None,
    split: str = PROMPT_SPLIT,
) -> tuple[list[dict], list[dict], dict]:
    """Split ``rows`` in ``split`` into ``(train_rows, holdout_rows, meta)`` by Pass-A band x gold class.

    Deterministic from ``(holdout_seed, item_id, pass_a_band, gold_class)``:
    - Uses Hamilton / largest-remainder apportionment across the 9 ``(band, gold_class)`` cells so the total
      holdout count equals ``round(len(pool) * holdout_fraction)`` (99 of 493 items at 20%).
    - Within each cell, ranks items by ``sha256(f"{holdout_seed}:{band}:{gold_class}:{id}")`` so the held-out
      IDs are identical across all training runs and seeds.
    """
    if holdout_seed is None or isinstance(holdout_seed, bool) or not str(holdout_seed).strip():
        _refuse("missing_holdout_seed", "stratified holdout requires an explicit non-empty holdout_seed")
    if not (0.0 < float(holdout_fraction) < 1.0):
        _refuse("invalid_holdout_fraction", f"holdout_fraction must be in (0, 1), got {holdout_fraction!r}")
    pool = sorted((r for r in rows if r.get("split") == split), key=lambda r: str(r["id"]))
    if not pool:
        _refuse("empty_pool", f"the gold has 0 {split} claims")
    split_ids = {str(r["id"]) for r in pool}
    k_by_id, prepass_sha = _extract_k_by_id(prepass_records_or_spec, split_ids)

    cells: dict[tuple[str, str], list[dict]] = {
        (b, c): [] for b in PASS_A_BANDS for c in GOLD_CLASSES
    }
    for r in pool:
        cid = str(r["id"])
        band = k_band_of(k_by_id[cid])
        gold_cls = str(r.get("label", "supported"))
        key = (band, gold_cls)
        if key not in cells:
            cells[key] = []
        cells[key].append(r)

    target_total = round(len(pool) * float(holdout_fraction))
    exact_quotas: dict[tuple[str, str], float] = {
        key: len(cell_rows) * float(holdout_fraction) for key, cell_rows in cells.items()
    }
    alloc: dict[tuple[str, str], int] = {key: int(q) for key, q in exact_quotas.items()}
    remainder_needed = target_total - sum(alloc.values())
    if remainder_needed > 0:
        ranked_cells = sorted(
            cells.keys(),
            key=lambda key: (-(exact_quotas[key] - alloc[key]), key[0], key[1]),
        )
        for key in ranked_cells:
            if remainder_needed <= 0:
                break
            if alloc[key] < len(cells[key]):
                alloc[key] += 1
                remainder_needed -= 1

    train_rows: list[dict] = []
    holdout_rows: list[dict] = []
    cell_meta: dict[str, dict[str, int]] = {}
    seed_str = str(holdout_seed)
    for key in sorted(cells.keys()):
        band, gold_cls = key
        cell_items = sorted(cells[key], key=lambda r: str(r["id"]))
        k_hold = alloc[key]
        ranked = sorted(
            cell_items,
            key=lambda r: (
                hashlib.sha256(f"{seed_str}:{band}:{gold_cls}:{r['id']}".encode()).hexdigest(),
                str(r["id"]),
            ),
        )
        holdout_rows.extend(ranked[:k_hold])
        train_rows.extend(ranked[k_hold:])
        cell_meta[f"{band}:{gold_cls}"] = {
            "total": len(cell_items),
            "holdout": k_hold,
            "train": len(cell_items) - k_hold,
        }

    train_rows.sort(key=lambda r: str(r["id"]))
    holdout_rows.sort(key=lambda r: str(r["id"]))
    holdout_ids = [str(r["id"]) for r in holdout_rows]
    train_ids = [str(r["id"]) for r in train_rows]
    meta: dict = {
        "holdout_seed": seed_str,
        "holdout_fraction": float(holdout_fraction),
        "total_items": len(pool),
        "holdout_count": len(holdout_rows),
        "train_count": len(train_rows),
        "holdout_ids": holdout_ids,
        "holdout_ids_sha256": ids_sha256(holdout_ids),
        "train_ids_sha256": ids_sha256(train_ids),
        "cells": cell_meta,
    }
    if prepass_sha is not None:
        meta["prepass_sha256"] = prepass_sha
    return train_rows, holdout_rows, meta


def build_class_stratified_epoch(
    train_rows: Sequence[dict],
    *,
    seed: int,
    epoch: int = 0,
) -> list[dict]:
    """Return one class-stratified epoch over ``train_rows`` without replacement.

    Every item in ``train_rows`` appears exactly once per epoch (matching the pool's exact class counts
    with zero repeats within an epoch), reshuffled per ``(seed, epoch)``, and interleaved proportionally
    across the epoch so any prefix or window preserves class proportions.
    """
    pool = sorted(train_rows, key=lambda r: str(r["id"]))
    if not pool:
        return []
    by_class: dict[str, list[dict]] = {}
    for r in pool:
        cls = str(r.get("label", "supported"))
        by_class.setdefault(cls, []).append(r)

    classes = sorted(by_class.keys())
    shuffled_by_class: dict[str, list[dict]] = {}
    for cls in classes:
        cls_items = list(by_class[cls])
        digest = hashlib.sha256(f"{seed}:epoch:{epoch}:class:{cls}".encode()).digest()
        cls_rng = random.Random(int.from_bytes(digest[:8], "big"))
        cls_rng.shuffle(cls_items)
        shuffled_by_class[cls] = cls_items

    total_n = len(pool)
    counts = {cls: len(shuffled_by_class[cls]) for cls in classes}
    taken = {cls: 0 for cls in classes}
    tie_digest = hashlib.sha256(f"{seed}:epoch:{epoch}:tie".encode()).digest()
    tie_rng = random.Random(int.from_bytes(tie_digest[:8], "big"))

    epoch_order: list[dict] = []
    for pos in range(total_n):
        avail = [cls for cls in classes if taken[cls] < counts[cls]]
        tie_order = list(avail)
        tie_rng.shuffle(tie_order)
        tie_rank = {cls: idx for idx, cls in enumerate(tie_order)}
        chosen_cls = max(
            avail,
            key=lambda cls: (
                (counts[cls] * (pos + 1) / float(total_n)) - taken[cls],
                -tie_rank[cls],
            ),
        )
        epoch_order.append(shuffled_by_class[chosen_cls][taken[chosen_cls]])
        taken[chosen_cls] += 1
    return epoch_order


def build_class_stratified_schedule(
    train_rows: Sequence[dict],
    total_draws: int,
    *,
    seed: int,
    holdout_ids: Sequence[str] | set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Build a multi-epoch class-stratified schedule of length ``total_draws`` without replacement."""
    if total_draws < 1:
        _refuse("invalid_draw_n", f"total_draws must be >= 1, got {total_draws!r}")
    holdout_set = {str(x) for x in holdout_ids} if holdout_ids is not None else set()
    pool = sorted(train_rows, key=lambda r: str(r["id"]))
    if not pool:
        _refuse("empty_pool", "train_rows is empty")
    overlap = sorted({str(r["id"]) for r in pool} & holdout_set)
    if overlap:
        _refuse(
            "holdout_item_in_training",
            f"training pool contains held-out item {overlap[0]!r} ({len(overlap)} overlapping)",
        )
    schedule: list[dict] = []
    epoch_idx = 0
    while len(schedule) < total_draws:
        epoch_rows = build_class_stratified_epoch(pool, seed=seed, epoch=epoch_idx)
        needed = total_draws - len(schedule)
        schedule.extend(epoch_rows[:needed])
        epoch_idx += 1
    drawn_ids = [str(r["id"]) for r in schedule]
    return schedule, {
        "draw_mode": "class_stratified",
        "draw_n": int(total_draws),
        "draw_seed": int(seed),
        "train_pool_size": len(pool),
        "epochs_spanned": epoch_idx,
        "drawn_ids_sha256": ids_sha256(drawn_ids),
    }


def draw_prompts(
    rows: list[dict],
    n: int,
    *,
    mode: str = "first",
    seed: int | None = None,
    prepass=None,
    k_min: int | None = None,
    k_max: int | None = None,
    split: str = PROMPT_SPLIT,
    holdout_fraction: float = HOLDOUT_DEFAULT_FRACTION,
    holdout_seed: str | int | None = None,
    holdout_ids: Sequence[str] | set[str] | None = None,
) -> tuple[list[dict], dict]:
    """Select ``n`` prompts from ``rows`` in ``split`` according to ``mode`` and return ``(prompts, record)``.

    Modes:
    - ``first``: calls ``select_prompts(rows, n, split=split)`` directly (excluding active holdout IDs).
    - ``uniform``: seeded sample of ``n`` without replacement from all sorted rows of ``split`` using a
      local ``random.Random(seed)``, returned sorted by ``id``.
    - ``kband``: filters to split rows whose pre-pass ``k_correct`` is in ``[k_min, k_max]``, then takes
      the same seeded sample of ``n`` without replacement using ``random.Random(seed)``, sorted by ``id``.
    - ``class_stratified``: class-stratified draw without replacement within each epoch over the training
      pool (always requiring and excluding the stratified 20% holdout split).

    The returned ``record`` never contains item text.
    """
    if mode not in DRAW_MODES:
        _refuse("unknown_draw_mode", f"unknown draw mode {mode!r}; expected one of {DRAW_MODES}")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        _refuse("invalid_draw_n", f"n must be an integer >= 1, got {n!r}")

    _check_unused_draw_flags(
        mode,
        seed=seed,
        prepass=prepass,
        k_min=k_min,
        k_max=k_max,
        holdout_seed=holdout_seed,
    )

    active_holdout: set[str] = {str(x) for x in holdout_ids} if holdout_ids is not None else set()
    has_prepass = prepass is not None and prepass != (None, None)
    if mode != "class_stratified" and not active_holdout and holdout_seed is not None and has_prepass:
        _, _, hold_meta_extra = split_stratified_holdout(
            rows,
            prepass,
            holdout_fraction=holdout_fraction,
            holdout_seed=holdout_seed,
            split=split,
        )
        active_holdout = set(hold_meta_extra["holdout_ids"])

    if mode == "first":
        avail_rows = [r for r in rows if str(r.get("id")) not in active_holdout] if active_holdout else rows
        picked = select_prompts(avail_rows, n, split=split)
        picked_ids = [str(r["id"]) for r in picked]
        return picked, {
            "draw_mode": "first",
            "draw_n": int(n),
            "draw_seed": None,
            "drawn_ids_sha256": ids_sha256(picked_ids),
        }

    if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        _refuse("missing_seed", f"{mode} draw requires an integer seed")

    pool = sorted(
        (r for r in rows if r.get("split") == split and str(r.get("id")) not in active_holdout),
        key=lambda r: str(r["id"]),
    )

    if mode == "uniform":
        if n > len(pool):
            _refuse(
                "n_exceeds_pool",
                f"requested n={n} exceeds {split} pool size {len(pool)}",
            )
        rng = random.Random(seed)
        sampled = sorted(rng.sample(pool, n), key=lambda r: str(r["id"]))
        sampled_ids = [str(r["id"]) for r in sampled]
        return sampled, {
            "draw_mode": "uniform",
            "draw_n": int(n),
            "draw_seed": int(seed),
            "drawn_ids_sha256": ids_sha256(sampled_ids),
        }

    if mode == "class_stratified":
        if not has_prepass:
            _refuse(
                "prepass_missing",
                "class_stratified requires a pre-pass JSONL path and expected sha256 for the holdout split",
            )
        train_pool, _, hold_meta = split_stratified_holdout(
            rows,
            prepass,
            holdout_fraction=holdout_fraction,
            holdout_seed=holdout_seed,
            split=split,
        )
        eff_holdout_ids = set(hold_meta["holdout_ids"]) | active_holdout
        if active_holdout:
            train_pool = [r for r in train_pool if str(r["id"]) not in active_holdout]
        schedule, sched_rec = build_class_stratified_schedule(
            train_pool,
            n,
            seed=seed,
            holdout_ids=eff_holdout_ids,
        )
        sched_rec["holdout_seed"] = hold_meta["holdout_seed"]
        sched_rec["holdout_count"] = hold_meta["holdout_count"]
        sched_rec["holdout_ids_sha256"] = hold_meta["holdout_ids_sha256"]
        sched_rec["train_ids_sha256"] = hold_meta["train_ids_sha256"]
        if "prepass_sha256" in hold_meta:
            sched_rec["prepass_sha256"] = hold_meta["prepass_sha256"]
        return schedule, sched_rec

    # mode == "kband"
    full_split_ids = {str(r["id"]) for r in rows if r.get("split") == split}
    records_by_id, actual_sha = _load_and_verify_prepass(prepass, full_split_ids)

    if k_min is None or k_max is None or isinstance(k_min, bool) or isinstance(k_max, bool):
        _refuse("k_range_missing", "kband requires integer k_min and k_max")
    if not isinstance(k_min, int) or not isinstance(k_max, int):
        _refuse("k_out_of_range", f"k_min and k_max must be integers, got {k_min!r}..{k_max!r}")

    # Validate prepass k_correct and g values, and determine valid range [0, g]
    g_values: set[int] = set()
    for rec_id, rec in records_by_id.items():
        k_val = rec.get("k_correct")
        g_val = rec.get("g", 8)
        if isinstance(k_val, bool) or not isinstance(k_val, int):
            _refuse("k_out_of_range", f"pre-pass record {rec_id!r} has non-integer k_correct {k_val!r}")
        if isinstance(g_val, bool) or not isinstance(g_val, int) or g_val < 1:
            _refuse("k_out_of_range", f"pre-pass record {rec_id!r} has invalid g {g_val!r}")
        if k_val < 0 or k_val > g_val:
            _refuse(
                "k_out_of_range",
                f"pre-pass record {rec_id!r} has k_correct={k_val} outside [0, {g_val}]",
            )
        g_values.add(g_val)

    if len(g_values) > 1:
        _refuse(
            "prepass_mixed_g",
            f"pre-pass records have inconsistent g values: {sorted(g_values)}",
        )
    g_limit = next(iter(g_values), 8)
    if k_min < 0 or k_max < 0 or k_min > g_limit or k_max > g_limit:
        _refuse(
            "k_out_of_range",
            f"k_min={k_min}, k_max={k_max} outside valid range [0, {g_limit}]",
        )
    if k_min > k_max:
        _refuse("k_min_gt_k_max", f"k_min ({k_min}) > k_max ({k_max})")

    band_pool = [r for r in pool if k_min <= int(records_by_id[str(r["id"])]["k_correct"]) <= k_max]
    if n > len(band_pool):
        _refuse(
            "n_exceeds_band",
            f"requested n={n} exceeds k-band [{k_min}, {k_max}] pool size {len(band_pool)}",
        )

    rng = random.Random(seed)
    sampled = sorted(rng.sample(band_pool, n), key=lambda r: str(r["id"]))
    sampled_ids = [str(r["id"]) for r in sampled]
    return sampled, {
        "draw_mode": "kband",
        "draw_n": int(n),
        "draw_seed": int(seed),
        "drawn_ids_sha256": ids_sha256(sampled_ids),
        "prepass_sha256": actual_sha,
        "k_min": int(k_min),
        "k_max": int(k_max),
        "band_pool_size": len(band_pool),
    }
