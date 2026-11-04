"""Prompt draw rule for Stage 2 GRPO and the full-run trainer.

Provides ``draw_prompts`` in a standalone sibling module beside ``s2_grpo_smoke.py`` so both
``examples/verifier-distill/s2_grpo_smoke.py`` and a future sibling full-run trainer can import
``from s2_draw import draw_prompts`` without re-implementing ``uniform`` or ``kband``.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Sequence
from pathlib import Path

PROMPTS = 32
PROMPT_SPLIT = "tune"
DRAW_MODES = ("first", "uniform", "kband")


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
) -> None:
    unused: list[str] = []
    if mode == "first" and seed is not None:
        unused.append("--draw-seed")
    if mode in ("first", "uniform"):
        if isinstance(prepass, (tuple, list)) and len(prepass) == 2:
            if prepass[0] is not None:
                unused.append("--prepass-jsonl")
            if prepass[1] is not None:
                unused.append("--prepass-sha256")
            if prepass[0] is None and prepass[1] is None:
                unused.append("--prepass")
        elif prepass is not None:
            unused.append("--prepass")
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


def _load_and_verify_prepass(
    prepass_spec,
    split_ids: set[str],
) -> tuple[dict[str, dict], str]:
    """Load pre-pass JSONL, verify SHA-256 and split id coverage, and return ``(records_by_id, sha256)``."""
    path, expected_sha = _parse_prepass_spec(prepass_spec)
    if not path.is_file():
        _refuse("prepass_missing_file", f"pre-pass file does not exist: {path.name}")
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
        rec_id = str(obj["id"])
        if rec_id in records_by_id:
            _refuse("prepass_duplicate_id", f"duplicate pre-pass record for id {rec_id!r}")
        if rec_id not in split_ids:
            _refuse("prepass_extra_id", f"pre-pass id {rec_id!r} is not in the split")
        records_by_id[rec_id] = obj

    missing = sorted(split_ids - set(records_by_id))
    if missing:
        _refuse(
            "prepass_missing_split_id",
            f"split row {missing[0]!r} has no pre-pass record ({len(missing)} missing)",
        )
    return records_by_id, actual_sha


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
) -> tuple[list[dict], dict]:
    """Select ``n`` prompts from ``rows`` in ``split`` according to ``mode`` and return ``(prompts, record)``.

    Modes:
    - ``first``: calls ``select_prompts(rows, n, split=split)`` directly.
    - ``uniform``: seeded sample of ``n`` without replacement from all sorted rows of ``split`` using a
      local ``random.Random(seed)``, returned sorted by ``id``.
    - ``kband``: filters to split rows whose pre-pass ``k_correct`` is in ``[k_min, k_max]``, then takes
      the same seeded sample of ``n`` without replacement using ``random.Random(seed)``, sorted by ``id``.

    The returned ``record`` never contains item text.
    """
    if mode not in DRAW_MODES:
        _refuse("unknown_draw_mode", f"unknown draw mode {mode!r}; expected one of {DRAW_MODES}")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        _refuse("invalid_draw_n", f"n must be an integer >= 1, got {n!r}")

    _check_unused_draw_flags(mode, seed=seed, prepass=prepass, k_min=k_min, k_max=k_max)

    if mode == "first":
        picked = select_prompts(rows, n, split=split)
        picked_ids = [str(r["id"]) for r in picked]
        return picked, {
            "draw_mode": "first",
            "draw_n": int(n),
            "draw_seed": None,
            "drawn_ids_sha256": ids_sha256(picked_ids),
        }

    if seed is None or isinstance(seed, bool) or not isinstance(seed, int):
        _refuse("missing_seed", f"{mode} draw requires an integer seed")

    pool = sorted((r for r in rows if r.get("split") == split), key=lambda r: str(r["id"]))

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

    # mode == "kband"
    split_ids = {str(r["id"]) for r in pool}
    records_by_id, actual_sha = _load_and_verify_prepass(prepass, split_ids)

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
