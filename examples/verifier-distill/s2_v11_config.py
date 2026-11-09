"""Frozen configuration and helpers for ASPIRE-SI Stage 2 (S2) v1.1 trainer.

Every constant in ``S2V11`` is bound to ``docs/runs/2026-11-08-smoke-s2-v1.1-prereg.md``
(sha256 ``9b2db4cf3c6e9eca415aa021a276cacccde2f1aac2c72be215e84e64c2962554``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Mapping, Sequence

import unicodedata

import quote_matcher_m1 as _m1
import quote_matcher_m2 as _m2
import quote_rule as _qr


PREREG_V11_RELPATH = "docs/runs/2026-11-08-smoke-s2-v1.1-prereg.md"
PREREG_V11_SHA256 = "9b2db4cf3c6e9eca415aa021a276cacccde2f1aac2c72be215e84e64c2962554"


@dataclass(frozen=True)
class S2V11Config:
    """Frozen S2 v1.1 pre-registered configuration (§1–§8 of sealed pre-registration)."""

    # Document identity
    prereg_version: str = "1.1"  # §1 title
    prereg_sha256: str = "9b2db4cf3c6e9eca415aa021a276cacccde2f1aac2c72be215e84e64c2962554"  # §1 sealed sha256

    # §4.1 Matcher M2a
    matcher: str = "M2a"  # §4.1
    quote_matcher_m2_sha256: str = "98780f92893e0e3ad4e4f9394e1c81a2f7e7f94a8b3331aed3ee217114d0ea29"  # §4.1
    quote_matcher_m1_sha256: str = "57064f46ff0f39ec26522e3d3d52ccb4c7f96300729f9a79052e05dfc842b42c"  # §6

    # §4.2 Wrong-label positive-advantage clamp from update 1
    clamp_wrong_positive_from_update: int = 1  # §4.2
    clamp_wrong_positive_ceil: float = 0.0  # §4.2

    # §4.3 Constant loss denominator (Dr. GRPO) & §4.4 truncation halt
    constant_loss_denominator: int = 4514  # §4.3
    truncation_halt_threshold: float = 0.20  # §4.4

    # §3 & §5.1 Clean canary set C and Fuse A (CT collapse Poisson CUSUM)
    canary_c_label: str = "unsupported"  # §3
    canary_c_pass_a_min_k: int = 6  # §3
    canary_c_pass_b_min_k: int = 6  # §3
    canary_c_max_cannot_tell: int = 0  # §3
    canary_c_size: int = 128  # §3
    canary_c_required_item: str = "diff-rnd-e839712-2u"  # §3
    fuse_a_group_min_cannot_tell: int = 2  # §5.1
    fuse_a_k: float = 0.211  # §5.1
    fuse_a_h: float = 3.25  # §5.1
    windows_per_run: int = 20  # §5.1
    groups_per_window: int = 40  # §5.1

    # §3 & §5.2 Channel B (FA2 false-accept Poisson CUSUM on gold-unsupported)
    channel_b_group_min_supported: int = 2  # §3 / §5.2
    channel_b_k: float = 5.03  # §5.2
    channel_b_h: float = 7.0  # §5.2

    # §5.6 Other CtCollapseGuard rules (post-clamp advantages; S1 dropped; S2 kept; U1 -> confirm)
    guard_reads_post_clamp_advantages: bool = True  # §5.6
    s1_dropped_inert: bool = True  # §5.6
    s2_kept_as_registered: bool = True  # §5.6
    u1_min_distinct_ct_items: int = 4  # §5.6
    u1_max_correct_rate: float = 0.78  # §5.6
    u1_routed_to_confirmation: bool = True  # §5.6

    # §5.5 T=0 canary confirmation protocol & shared cap of 3
    confirm_temperature: float = 0.0  # §5.5
    confirm_samples_per_item: int = 1  # §5.5
    step_0_canary_before_update_1: bool = True  # §5.5
    fuse_a_confirm_min_ct: int = 2  # §5.5
    fuse_a_confirm_min_delta_over_step0: int = 2  # §5.5
    fuse_a_keep_cusum_on_confirm_negative: bool = True  # §5.5
    gold_unsupported_train_items: int = 184  # §5.5
    mcnemar_one_sided_alpha: float = 0.05  # §5.5
    share_184_pass_in_same_window: bool = True  # §5.5
    confirm_cap_per_run: int = 3  # §5.5
    confirm_halt_on_nth_trip: int = 4  # §5.5

    # §6 First-run canaries & rollback lines (2 consecutive windows -> STOP-AND-REPORT)
    canary_consecutive_windows: int = 2  # §6
    canary_shadow_m1_min_pass_rate: float = 0.70  # §6 (i)
    canary_boundary_crossing_max_rate: float = 0.010  # §6 (ii)
    canary_sign_escape_edge_slice_min_count: int = 3  # §6 (iii)
    canary_unbalanced_brackets_max_rate: float = 0.25  # §6 (iv)
    canary_post_clamp_wrong_positive_max: int = 0  # §6 (v)
    canary_quote_len_p10_min_chars: float = 16.0  # §6 (vi)

    # §7 Estimand, baseline and seeds
    seeds: tuple[int, int, int] = (45, 46, 47)  # §7
    primary_horizon_updates: int = 197  # §7
    secondary_checkpoint_update: int = 60  # §7

    @property
    def s2_min_cum_supported_answers(self) -> int:
        return 600

    @property
    def s2_max_ct_rate(self) -> float:
        return 0.015

    @property
    def s2_min_distinct_ct_items(self) -> int:
        return 4

    @property
    def u1_min_correct_rate(self) -> float:
        return self.u1_max_correct_rate

    @property
    def fuse_a_confirm_min_increase(self) -> int:
        return self.fuse_a_confirm_min_delta_over_step0

    @property
    def channel_b_confirm_alpha(self) -> float:
        return self.mcnemar_one_sided_alpha


S2V11 = S2V11Config()


# Known 12 training-pool gold-unsupported items that passed Pass A (k>=6, ct=0)
# but had k_correct < 6 or cannot_tell > 0 in Pass B (derived on registered
# Pass A feee81ff... + Pass B d77601e5... in v11-baselines.py, §3).
PASS_A_PASS_B_DIFF_EXCLUDED_IDS_V11: frozenset[str] = frozenset(
    {
        "abandon-active-f",
        "cal-corrections-f",
        "diff-aspire-si-13-2u",
        "diff-aspire-si-48-3u",
        "diff-offrig-3-3u",
        "diff2-aspire-si-78cc26f-5u",
        "diffhard-aspire-si-52-4u",
        "prs-aspire-si-58-3u",
        "prs-aspire-si-59-1u",
        "prs-aspire-si-64-2u",
        "prs-offrig-31-4u",
        "prs-offrig-34-17u",
    }
)


def s2_v11_config_dict(cfg: S2V11Config = S2V11) -> dict[str, Any]:
    """Return a JSON-serializable dictionary of ``S2V11`` (converting tuples to lists)."""
    raw = asdict(cfg)
    raw["seeds"] = list(cfg.seeds)
    return raw


def s2_v11_config_sha256(cfg: S2V11Config = S2V11) -> str:
    """Return the canonical SHA-256 hex digest of ``s2_v11_config_dict()``."""
    payload = json.dumps(s2_v11_config_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_sealed_prereg_v11_text(
    repo_root: Path | None = None,
    *,
    expected_sha256: str = PREREG_V11_SHA256,
) -> str:
    """Read the committed sealed S2 v1.1 pre-registration text and verify its SHA-256."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / PREREG_V11_RELPATH if (repo_root / PREREG_V11_RELPATH).exists() else repo_root
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"S2_V11_PREREG_SHA256_MISMATCH: Sealed prereg SHA-256 mismatch at {path}: "
            f"expected {expected_sha256}, got {digest}"
        )
    return raw.decode("utf-8")


def parse_sealed_prereg_v11(text: str) -> dict[str, Any]:
    """Parse ``docs/runs/2026-11-08-smoke-s2-v1.1-prereg.md`` into a config dict matching ``S2V11``."""
    raw_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _must(pattern: str, flags: int = 0) -> re.Match[str]:
        m = re.search(pattern, text, flags)
        if not m:
            raise ValueError(f"Pattern not found in sealed prereg text: {pattern!r}")
        return m

    ver_m = _must(r"# S2 pre-registration v([0-9.]+) \u2014 a new series")
    c_def_m = _must(
        r"\*\*Clean canary set C:\*\* gold-(unsupported), with Pass A k \u2265 (\d+), Pass B k \u2265 (\d+), and (\d+) cannot_tell in both passes\."
    )
    c_size_m = _must(r"\|C\| = (\d+) in the training pool\.")
    c_item_m = _must(r"\*\*It includes `([^`]+)`\*\*")
    fa2_def_m = _must(r"FA2 groups \(\u2265 (\d+) `supported` on a gold-unsupported item\)")

    matcher_m = _must(r"1\. \*\*Matcher (M2a),\*\* as audited at 86b0280\.")
    clamp_m = _must(
        r"2\. \*\*The clamp is on from update (\d+):\*\* wrong-label positive advantages go to \u2264 ([0-9.]+), with no re-centring\s*\(`clamp_wrong_positive_advantages`\)\."
    )
    denom_m = _must(r"\*\*v1\.1 uses a constant denominator of ([0-9,]+) tokens\*\*")
    trunc_m = _must(r"truncation halt at ([0-9.]+)\.")

    fuse_a_m = _must(
        r"1\. \*\*Fuse A, CT collapse:\*\* a Poisson CUSUM on B_w, the number of C-groups in window w with \u2265 (\d+) cannot_tell\.\s*- S_w = max\(0, S_\{w\u22121\} \+ B_w \u2212 k\), with \*\*k = ([0-9.]+)\*\* and \*\*h = ([0-9.]+)\*\*\."
    )
    win_20_m = _must(r"(\d+)-window power:")
    win_size_m = _must(r"in one (\d+)-update window\.")

    chan_b_m = _must(
        r"2\. \*\*Channel B, false accepts:\*\* a Poisson CUSUM on FA2_w, designed for R = 2\.\s*- \*\*k = ([0-9.]+)\*\*, \*\*h = ([0-9.]+)\*\*\."
    )

    conf_t0_m = _must(r"\*\*On a trip, a T = ([0-9.]+) canary confirms before the halt is final\*\*")
    conf_fa_m = _must(
        r"\*\*Fuse A:\*\* greedy decoding on C \(\d+ items, one sample each\)\. \*\*Halt only if\*\* at least (\d+) items answer cannot_tell\s*\*\*and\*\* the count exceeds this run's own step-0 count by at least (\d+)\. Otherwise log CONFIRM-NEGATIVE, keep the CUSUM\s*state, and continue\."
    )
    conf_cb_m = _must(
        r"\*\*Channel B:\*\* a T = 0 pass on all (\d+) gold-unsupported training items[\s\S]*?one-sided McNemar test on item flips to `supported`, against\s*this run's own step-0 pass, gives p < ([0-9.]+)\."
    )
    conf_u1_share_m = _must(
        r"If Channel B and U1 trip in the same window, one 184-item pass serves both: it reads `supported` and\s*`cannot_tell` flips\."
    )
    step0_m = _must(
        r"\*\*Each run measures its own T = 0 step-0 canary\*\* \(C plus the 184 items\) before update 1\."
    )
    cap_m = _must(
        r"\*\*Cap:\*\* at most (\d+) confirmations per run across Fuse A, Channel B and U1\. A (\d+)th trip of any of them halts without\s*confirmation\."
    )

    s1_m = _must(
        r"\*\*S1 is DROPPED as inert\*\* \(Publisher ruling\)\.\s*- The guard reads \*\*post-clamp\*\* advantages\."
    )
    s2_m = _must(r"\*\*S2 kept as registered:\*\*")
    u1_m = _must(
        r"\*\*U1 is kept as registered \(\u2265 (\d+) distinct CT items on gold-unsupported AND a correct rate < ([0-9.]+)\), but it no longer\s*halts directly\.\*\*"
    )

    can_win_m = _must(r"A line counts as crossed only for \*\*(\d+) consecutive windows\*\*:")
    can_i_m = _must(r"- \(i\) below ([0-9.]+);")
    can_ii_m = _must(r"- \(ii\) above ([0-9.]+);")
    can_iii_m = _must(r"- \(iii\) \u2265 (\d+) in one window")
    can_iv_m = _must(r"- \(iv\) above ([0-9.]+);")
    can_vi_m = _must(r"- \(vi\) below (\d+) chars\.")
    can_v_m = _must(r"\| \(v\) wrong-label positive advantage after the clamp \| \u2014 \| (\d+) by construction \|")

    s7_pri_m = _must(r"\*\*Primary:\*\* the full (\d+)-update horizon")
    s7_sec_m = _must(r"\*\*Secondary:\*\* a common checkpoint at update (\d+)")
    s7_seeds_m = _must(r"\*\*Seeds:\*\* (\d+), (\d+), (\d+)")

    here = Path(__file__).resolve().parent
    m1_sha = hashlib.sha256((here / "quote_matcher_m1.py").read_bytes()).hexdigest()
    m2_sha = hashlib.sha256((here / "quote_matcher_m2.py").read_bytes()).hexdigest()

    return {
        "prereg_version": ver_m.group(1),
        "prereg_sha256": raw_sha,
        "matcher": matcher_m.group(1),
        "quote_matcher_m2_sha256": m2_sha,
        "quote_matcher_m1_sha256": m1_sha,
        "clamp_wrong_positive_from_update": int(clamp_m.group(1)),
        "clamp_wrong_positive_ceil": float(clamp_m.group(2)),
        "constant_loss_denominator": int(denom_m.group(1).replace(",", "")),
        "truncation_halt_threshold": float(trunc_m.group(1)),
        "canary_c_label": c_def_m.group(1),
        "canary_c_pass_a_min_k": int(c_def_m.group(2)),
        "canary_c_pass_b_min_k": int(c_def_m.group(3)),
        "canary_c_max_cannot_tell": int(c_def_m.group(4)),
        "canary_c_size": int(c_size_m.group(1)),
        "canary_c_required_item": c_item_m.group(1),
        "fuse_a_group_min_cannot_tell": int(fuse_a_m.group(1)),
        "fuse_a_k": float(fuse_a_m.group(2)),
        "fuse_a_h": float(fuse_a_m.group(3)),
        "windows_per_run": int(win_20_m.group(1)),
        "groups_per_window": int(win_size_m.group(1)) * 4,
        "channel_b_group_min_supported": int(fa2_def_m.group(1)),
        "channel_b_k": float(chan_b_m.group(1)),
        "channel_b_h": float(chan_b_m.group(2)),
        "guard_reads_post_clamp_advantages": bool(s1_m),
        "s1_dropped_inert": bool(s1_m),
        "s2_kept_as_registered": bool(s2_m),
        "u1_min_distinct_ct_items": int(u1_m.group(1)),
        "u1_max_correct_rate": float(u1_m.group(2)),
        "u1_routed_to_confirmation": bool(u1_m),
        "confirm_temperature": float(conf_t0_m.group(1)),
        "confirm_samples_per_item": 1 if conf_fa_m else 0,
        "step_0_canary_before_update_1": bool(step0_m),
        "fuse_a_confirm_min_ct": int(conf_fa_m.group(1)),
        "fuse_a_confirm_min_delta_over_step0": int(conf_fa_m.group(2)),
        "fuse_a_keep_cusum_on_confirm_negative": bool(conf_fa_m),
        "gold_unsupported_train_items": int(conf_cb_m.group(1)),
        "mcnemar_one_sided_alpha": float(conf_cb_m.group(2)),
        "share_184_pass_in_same_window": bool(conf_u1_share_m),
        "confirm_cap_per_run": int(cap_m.group(1)),
        "confirm_halt_on_nth_trip": int(cap_m.group(2)),
        "canary_consecutive_windows": int(can_win_m.group(1)),
        "canary_shadow_m1_min_pass_rate": float(can_i_m.group(1)),
        "canary_boundary_crossing_max_rate": float(can_ii_m.group(1)),
        "canary_sign_escape_edge_slice_min_count": int(can_iii_m.group(1)),
        "canary_unbalanced_brackets_max_rate": float(can_iv_m.group(1)),
        "canary_post_clamp_wrong_positive_max": int(can_v_m.group(1)),
        "canary_quote_len_p10_min_chars": float(can_vi_m.group(1)),
        "seeds": [int(s7_seeds_m.group(1)), int(s7_seeds_m.group(2)), int(s7_seeds_m.group(3))],
        "primary_horizon_updates": int(s7_pri_m.group(1)),
        "secondary_checkpoint_update": int(s7_sec_m.group(1)),
    }


# ---------------------------------------------------------------------------
# §4.1 Raw-passage span locator and §6 canary helpers (mirrors v11-baselines-2.py)
# ---------------------------------------------------------------------------


_MD_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$", re.MULTILINE)


def _unmarked_with_map(p: str) -> tuple[str, list[int]]:
    if not p:
        return "", []
    work = p[:-1] if p.endswith("\n") else p
    parts = work.split("\n")
    out_chars: list[str] = []
    out_map: list[int] = []
    pos = 0
    ws_str = _qr._RUST_WHITESPACE_STR
    for li, part in enumerate(parts):
        line_len = len(part)
        line = part[:-1] if part.endswith("\r") else part
        idx = 0
        while idx < len(line) and line[idx] in ws_str:
            idx += 1
        if idx < len(line) and line[idx] in ("+", "-"):
            if not (idx + 1 < len(line) and line[idx + 1] in ("+", "-")):
                idx += 1
                while idx < len(line) and line[idx] in ws_str:
                    idx += 1
        for m in _qr._COMMENT_MARKERS:
            if line[idx:].startswith(m):
                idx += len(m)
                while idx < len(line) and line[idx] in ws_str:
                    idx += 1
                break
        for c_i in range(idx, len(line)):
            out_chars.append(line[c_i])
            out_map.append(pos + c_i)
        if li + 1 < len(parts):
            out_chars.append("\n")
            out_map.append(pos + line_len)
        pos += line_len + 1
    return "".join(out_chars), out_map


def _norm_tier0_with_map(text: str, base_map: list[int] | None = None) -> tuple[str, list[int]]:
    if base_map is None:
        base_map = list(range(len(text)))
    ws_set = _qr.RUST_WHITESPACE_CHARS
    n = len(text)
    i = 0
    while i < n and text[i] in ws_set:
        i += 1
    end = n
    while end > i and text[end - 1] in ws_set:
        end -= 1
    out_chars: list[str] = []
    out_map: list[int] = []
    in_ws = False
    for k in range(i, end):
        ch = text[k]
        if ch in ws_set:
            if not in_ws:
                out_chars.append(" ")
                out_map.append(base_map[k])
                in_ws = True
        else:
            in_ws = False
            tr = ch.translate(_qr._CURLY_QUOTES)
            for tc in tr:
                out_chars.append(tc)
                out_map.append(base_map[k])
    return "".join(out_chars), out_map


def _norm_tier1_with_map(text: str, base_map: list[int] | None = None) -> tuple[str, list[int]]:
    if base_map is None:
        base_map = list(range(len(text)))
    s1_chars: list[str] = []
    s1_map: list[int] = []
    for k, ch in enumerate(text):
        nfkc = unicodedata.normalize("NFKC", ch).translate(_m1._ZW_STRIP_TABLE)
        for nc in nfkc:
            s1_chars.append(nc)
            s1_map.append(base_map[k])
    s1 = "".join(s1_chars)

    s2_chars: list[str] = []
    s2_map: list[int] = []
    last = 0
    for m_hy in _m1._HYPHEN_NEWLINE_RE.finditer(s1):
        for k in range(last, m_hy.start()):
            s2_chars.append(s1[k])
            s2_map.append(s1_map[k])
        last = m_hy.end()
    for k in range(last, len(s1)):
        s2_chars.append(s1[k])
        s2_map.append(s1_map[k])
    s2 = "".join(s2_chars)

    n2 = len(s2)
    i = 0
    while i < n2 and s2[i].isspace():
        i += 1
    end = n2
    while end > i and s2[end - 1].isspace():
        end -= 1
    out_chars: list[str] = []
    out_map: list[int] = []
    in_ws = False
    for k in range(i, end):
        ch = s2[k]
        if ch.isspace():
            if not in_ws:
                out_chars.append(" ")
                out_map.append(s2_map[k])
                in_ws = True
        else:
            in_ws = False
            tr = ch.translate(_m1._QUOTE_DASH_MAP).casefold()
            for tc in tr:
                out_chars.append(tc)
                out_map.append(s2_map[k])
    return "".join(out_chars), out_map


def _norm_m2a_with_map(text: str, base_map: list[int] | None = None) -> tuple[str, list[int]]:
    if base_map is None:
        base_map = list(range(len(text)))
    s1_chars: list[str] = []
    s1_map: list[int] = []
    for k, ch in enumerate(text):
        nfkc = unicodedata.normalize("NFKC", ch).translate(_m1._ZW_STRIP_TABLE)
        for nc in nfkc:
            s1_chars.append(nc)
            s1_map.append(base_map[k])
    s1 = "".join(s1_chars)

    s2_chars: list[str] = []
    s2_map: list[int] = []
    last = 0
    for m_hy in _m1._HYPHEN_NEWLINE_RE.finditer(s1):
        for k in range(last, m_hy.start()):
            s2_chars.append(s1[k])
            s2_map.append(s1_map[k])
        last = m_hy.end()
    for k in range(last, len(s1)):
        s2_chars.append(s1[k])
        s2_map.append(s1_map[k])

    s3_chars: list[str] = []
    s3_map: list[int] = []
    for ch, idx in zip(s2_chars, s2_map):
        tr = ch.translate(_m1._QUOTE_DASH_MAP).casefold()
        for tc in tr:
            s3_chars.append(tc)
            s3_map.append(idx)
    s3 = "".join(s3_chars)

    s4_chars: list[str] = []
    s4_map: list[int] = []
    last = 0
    for m_hw in _m2._HORIZ_WS_RE.finditer(s3):
        for k in range(last, m_hw.start()):
            s4_chars.append(s3[k])
            s4_map.append(s3_map[k])
        s4_chars.append(" ")
        s4_map.append(s3_map[m_hw.start()])
        last = m_hw.end()
    for k in range(last, len(s3)):
        s4_chars.append(s3[k])
        s4_map.append(s3_map[k])
    s4 = "".join(s4_chars)

    s5_chars: list[str] = []
    s5_map: list[int] = []
    last = 0
    for m_lb in _m2._LINE_BREAK_WS_RE.finditer(s4):
        for k in range(last, m_lb.start()):
            s5_chars.append(s4[k])
            s5_map.append(s4_map[k])
        s5_chars.append("\n")
        s5_map.append(s4_map[m_lb.start()])
        last = m_lb.end()
    for k in range(last, len(s4)):
        s5_chars.append(s4[k])
        s5_map.append(s4_map[k])
    s5_raw = "".join(s5_chars)
    st_idx = 0
    while st_idx < len(s5_raw) and s5_raw[st_idx].isspace():
        st_idx += 1
    en_idx = len(s5_raw)
    while en_idx > st_idx and s5_raw[en_idx - 1].isspace():
        en_idx -= 1
    s = s5_raw[st_idx:en_idx]
    s_map = s5_map[st_idx:en_idx]

    spans = _m2.find_same_line_literal_spans(s)
    s_flat = s.replace("\n", " ")

    out_chars: list[str] = []
    out_map: list[int] = []

    def strip_seg(seg: str, seg_map: list[int]) -> None:
        n_seg = len(seg)

        def is_exempt(idx: int) -> bool:
            if seg[idx] not in (".", ","):
                return False
            l_i = idx - 1
            while l_i >= 0 and seg[l_i].isspace():
                l_i -= 1
            r_i = idx + 1
            while r_i < n_seg and seg[r_i].isspace():
                r_i += 1
            return (l_i >= 0 and seg[l_i].isdigit()) and (r_i < n_seg and seg[r_i].isdigit())

        i = 0
        while i < n_seg:
            if seg[i].isspace():
                j = i
                while j < n_seg and seg[j].isspace():
                    j += 1
                l_idx = i - 1
                r_idx = j
                rem = False
                if l_idx >= 0 and seg[l_idx] in _m2.M2A_PUNCT_SET and not is_exempt(l_idx):
                    rem = True
                if r_idx < n_seg and seg[r_idx] in _m2.M2A_PUNCT_SET and not is_exempt(r_idx):
                    rem = True
                if not rem:
                    out_chars.append(" ")
                    out_map.append(seg_map[i])
                i = j
            else:
                out_chars.append(seg[i])
                out_map.append(seg_map[i])
                i += 1

    last = 0
    for st, en in spans:
        if st > last:
            strip_seg(s_flat[last:st], s_map[last:st])
        for k in range(st, en):
            out_chars.append(s_flat[k])
            out_map.append(s_map[k])
        last = en
    if last < len(s_flat):
        strip_seg(s_flat[last:], s_map[last:])
    return "".join(out_chars), out_map


def locate_raw_span_m2a(
    quote: str,
    passages: str | Sequence[str | dict],
) -> tuple[int, int, int, str] | None:
    """Locate the raw passage span ``(passage_idx, raw_start, raw_end, tier)`` for an M2a-accepted quote."""
    ps = _qr.as_passages(passages)
    if not quote or not ps:
        return None

    def to_raw(cmap: list[int], pos: int, length: int) -> tuple[int, int]:
        return cmap[pos], cmap[pos + length - 1] + 1

    if _qr.quote_found(quote, ps):
        q0 = _qr.normalise(quote)
        q_len = len(q0)
        if q_len >= _qr.MIN_QUOTE_CHARS:
            for p_idx, p in enumerate(ps):
                t0, m0 = _norm_tier0_with_map(p)
                pos = t0.find(q0)
                if pos != -1:
                    rs, re_ = to_raw(m0, pos, q_len)
                    return p_idx, rs, re_, "0"
                unm, unm_map = _unmarked_with_map(p)
                t0u, m0u = _norm_tier0_with_map(unm, unm_map)
                pos = t0u.find(q0)
                if pos != -1:
                    rs, re_ = to_raw(m0u, pos, q_len)
                    return p_idx, rs, re_, "0"
        else:
            for p_idx, p in enumerate(ps):
                work = p[:-1] if p.endswith("\n") else p
                line_start = 0
                for part in work.split("\n"):
                    line = part[:-1] if part.endswith("\r") else part
                    bmap = list(range(line_start, line_start + len(line)))
                    ln, lm = _norm_tier0_with_map(line, bmap)
                    if ln == q0 and lm:
                        return p_idx, lm[0], lm[-1] + 1, "0"
                    unm_l, unm_lm = _unmarked_with_map(line)
                    unm_lm = [line_start + x for x in unm_lm]
                    lun, lum = _norm_tier0_with_map(unm_l, unm_lm)
                    if lun == q0 and lum:
                        return p_idx, lum[0], lum[-1] + 1, "0"
                    line_start += len(part) + 1

    for prefix, norm_fn, map_fn in (
        ("m1", _m1.normalise_tier1, _norm_tier1_with_map),
        ("m2a", _m2.normalise_m2a, _norm_m2a_with_map),
    ):
        q1 = norm_fn(quote, is_quote=True)
        t1_name = "1" if prefix == "m1" else "m2a_1"
        t2_name = "2" if prefix == "m1" else "m2a_2"
        ts_name = "short" if prefix == "m1" else "m2a_short"

        if len(q1) >= _qr.MIN_QUOTE_CHARS:
            for p_idx, p in enumerate(ps):
                tp, mp = map_fn(p)
                pos = tp.find(q1)
                if pos != -1:
                    rs, re_ = to_raw(mp, pos, len(q1))
                    return p_idx, rs, re_, t1_name
                unm, unm_map = _unmarked_with_map(p)
                tpu, mpu = map_fn(unm, unm_map)
                pos = tpu.find(q1)
                if pos != -1:
                    rs, re_ = to_raw(mpu, pos, len(q1))
                    return p_idx, rs, re_, t1_name

        q_sentinel = _m1._ELLIPSIS_RE.sub(_m1._ELLIPSIS_SENTINEL, quote)
        ell_count = q_sentinel.count(_m1._ELLIPSIS_SENTINEL)
        if 1 <= ell_count <= _m1.MAX_ELLIPSES:
            raw_frags = [f for f in q_sentinel.split(_m1._ELLIPSIS_SENTINEL) if f.strip()]
            if len(raw_frags) >= 2:
                norm_frags: list[str] = []
                frags_valid = True
                for rf in raw_frags:
                    nf = norm_fn(rf, is_quote=True)
                    words = _m1._WORD_RE.findall(nf)
                    if (
                        len(words) < _m1.MIN_FRAGMENT_WORDS
                        or len(_qr.content_tokens(nf)) < _m1.MIN_FRAGMENT_CONTENT_TOKENS
                    ):
                        frags_valid = False
                        break
                    norm_frags.append(nf)
                if frags_valid:
                    for p_idx, p in enumerate(ps):
                        for variant_idx in (0, 1):
                            if variant_idx == 0:
                                v_text, v_bmap = p, list(range(len(p)))
                            else:
                                v_text, v_bmap = _unmarked_with_map(p)
                            cand_pb = (
                                _m1._normalise_tier1_keep_para(v_text)
                                if prefix == "m1"
                                else _m2._normalise_keep_para_with_norm(v_text, norm_fn)
                            )
                            span_bounds = _m1._match_tier2_in_passage(norm_frags, cand_pb)
                            if span_bounds is not None:
                                seg_maps: list[int] = []
                                last_s = 0
                                first_seg = True
                                for m_pb in _m1._PARA_BREAK_RE.finditer(v_text):
                                    seg_t = v_text[last_s : m_pb.start()]
                                    seg_bm = v_bmap[last_s : m_pb.start()]
                                    _, sm = map_fn(seg_t, seg_bm)
                                    if not first_seg:
                                        seg_maps.append(v_bmap[last_s] if last_s < len(v_bmap) else 0)
                                    first_seg = False
                                    seg_maps.extend(sm)
                                    last_s = m_pb.end()
                                seg_t = v_text[last_s:]
                                seg_bm = v_bmap[last_s:]
                                _, sm = map_fn(seg_t, seg_bm)
                                if not first_seg:
                                    seg_maps.append(
                                        v_bmap[last_s]
                                        if last_s < len(v_bmap)
                                        else (seg_maps[-1] if seg_maps else 0)
                                    )
                                seg_maps.extend(sm)
                                st_b, en_b = span_bounds
                                return p_idx, seg_maps[st_b], seg_maps[en_b - 1] + 1, t2_name

        if 0 < len(q1) < _qr.MIN_QUOTE_CHARS:
            if len(q1) >= _qr.MIN_LINE_QUOTE_CHARS and any(ch.isalnum() for ch in q1):
                for p_idx, p in enumerate(ps):
                    work = p[:-1] if p.endswith("\n") else p
                    line_start = 0
                    for part in work.split("\n"):
                        line = part[:-1] if part.endswith("\r") else part
                        bmap = list(range(line_start, line_start + len(line)))
                        _, lm_raw = map_fn(line, bmap)
                        if norm_fn(line, is_quote=True) == q1 and lm_raw:
                            return p_idx, lm_raw[0], lm_raw[-1] + 1, ts_name
                        unm_l, unm_lm = _unmarked_with_map(line)
                        unm_lm = [line_start + x for x in unm_lm]
                        _, lum_raw = map_fn(unm_l, unm_lm)
                        if norm_fn(_qr.strip_markers(line), is_quote=True) == q1 and lum_raw:
                            return p_idx, lum_raw[0], lum_raw[-1] + 1, ts_name
                        line_start += len(part) + 1

            words = _m1._WORD_RE.findall(q1)
            if _m1.SHORT_MIN_WORDS <= len(words) <= _m1.SHORT_MAX_WORDS:
                has_digit = any(ch.isdigit() for ch in quote)
                has_cap = _m1._has_internal_capitalised_token(quote)
                if has_digit or has_cap:
                    for p_idx, p in enumerate(ps):
                        tp, mp = map_fn(p)
                        if 1 <= tp.count(q1) <= _m1.SHORT_MAX_OCCURRENCES:
                            pos = tp.find(q1)
                            rs, re_ = to_raw(mp, pos, len(q1))
                            return p_idx, rs, re_, ts_name
                        unm, unm_map = _unmarked_with_map(p)
                        tpu, mpu = map_fn(unm, unm_map)
                        if 1 <= tpu.count(q1) <= _m1.SHORT_MAX_OCCURRENCES:
                            pos = tpu.find(q1)
                            rs, re_ = to_raw(mpu, pos, len(q1))
                            return p_idx, rs, re_, ts_name
    return None


def _check_balanced_brackets(s: str, *, ignore_literals: bool = False) -> bool:
    """Check whether ``()``, ``[]``, ``{}`` are balanced in ``s`` (§6 iv)."""
    stack: list[str] = []
    pairs = {"(": ")", "[": "]", "{": "}"}
    closers = {")": "(", "]": "[", "}": "{"}
    lit_spans = _m2.find_same_line_literal_spans(s) if ignore_literals else []

    def in_lit(idx: int) -> bool:
        return any(st <= idx < en for st, en in lit_spans)

    for idx, ch in enumerate(s):
        if ignore_literals and in_lit(idx):
            continue
        if ch in pairs:
            stack.append(ch)
        elif ch in closers:
            if not stack or stack[-1] != closers[ch]:
                return False
            stack.pop()
    return len(stack) == 0


def _check_span_boundary(raw_p: str, rs: int, re_: int) -> dict[str, bool]:
    """Check whether a matched raw span crosses a paragraph or Markdown table boundary (§6 ii)."""
    span = raw_p[rs:re_]
    para_break = bool(_m1._PARA_BREAK_RE.search(span))
    has_md_table_in_p = bool(_MD_TABLE_SEP_RE.search(raw_p)) or (
        sum(1 for l in _qr.rust_lines(raw_p) if l.strip().startswith("|") and l.strip().endswith("|")) >= 2
    )
    table_md = has_md_table_in_p and ("|" in span)
    return {
        "para_break": para_break,
        "table_md": table_md,
        "para_or_table_md": para_break or table_md,
        "multiline_pipe_inc_code": ("\n" in span and "|" in span),
    }


def _check_span_edge_slicing(raw_p: str, rs: int, re_: int, quote: str) -> dict[str, bool]:
    """Check whether a matched raw span sliced off a leading sign/escape (§6 iii)."""
    prev_ch = raw_p[rs - 1] if rs > 0 else ""
    first_ch = raw_p[rs] if rs < len(raw_p) else ""
    last_ch = raw_p[re_ - 1] if 0 < re_ <= len(raw_p) else ""
    next_ch = raw_p[re_] if re_ < len(raw_p) else ""

    raw_escape_slice = (prev_ch == "\\") or (last_ch == "\\" and next_ch != "" and not next_ch.isspace())
    raw_sign_before_digit = prev_ch in "+-\u00b1~\u20ac$\u00a3\u00a5#" and first_ch.isdigit()

    s_t1 = _m1.normalise_tier1(quote, is_quote=False)
    st, en = 0, len(s_t1)
    while st < en and not _m1._is_protected_edge_char(s_t1, st):
        if unicodedata.category(s_t1[st]).startswith(("P", "S")) or s_t1[st].isspace():
            st += 1
        else:
            break
    while en > st and not _m1._is_protected_edge_char(s_t1, en - 1):
        if unicodedata.category(s_t1[en - 1]).startswith(("P", "S")) or s_t1[en - 1].isspace():
            en -= 1
        else:
            break
    stripped_prefix = s_t1[:st]
    stripped_suffix = s_t1[en:]
    q_sign_stripped = (
        st > 0
        and st < len(s_t1)
        and s_t1[st].isdigit()
        and s_t1[st - 1] in "+-\u00b1~\u20ac$\u00a3\u00a5#"
    )
    q_escape_stripped = ("\\" in stripped_prefix) or ("\\" in stripped_suffix)
    return {
        "raw_escape_slice": raw_escape_slice,
        "raw_sign_before_digit": raw_sign_before_digit,
        "q_sign_stripped": q_sign_stripped,
        "q_escape_stripped": q_escape_stripped,
        "any_sign_or_escape": raw_escape_slice or raw_sign_before_digit or q_sign_stripped or q_escape_stripped,
    }


def pct_linear(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile on pre-sorted values (matches ``v11-baselines-2.py``)."""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def compute_window_canaries_v11(
    quote_records: Sequence[Mapping[str, Any]],
    *,
    post_clamp_wrong_positive_count: int = 0,
) -> dict[str, Any]:
    """Compute the 6 per-window M2a canary metrics defined in §6 of the sealed spec."""
    total_quotes = len(quote_records)
    m1_ok_all = 0
    m2a_ok_all = 0
    bc_para_or_table_count = 0
    es_any_count = 0
    unbal_raw_span_count = 0
    accepted_norm_lens: list[float] = []

    for rec in quote_records:
        q = str(rec.get("quote", "") or "")
        raw_passages = rec.get("passages")
        if raw_passages is None:
            p_single = str(rec.get("passage", "") or "")
            ps = [p_single] if p_single else []
        else:
            ps = _qr.as_passages(raw_passages)
        m1_ok = bool(rec.get("m1_matched", _m1.quote_found_m1(q, ps) if ps else False))
        m2a_ok = bool(rec.get("m2a_matched", _m2.quote_found_m2a(q, ps) if ps else False))
        if m1_ok:
            m1_ok_all += 1
        if m2a_ok:
            m2a_ok_all += 1
            nq = _m2.normalise_m2a(q, is_quote=True)
            accepted_norm_lens.append(float(len(nq)))
            loc = rec.get("raw_span")
            if loc is None and ps:
                loc = locate_raw_span_m2a(q, ps)
            if loc is not None and ps:
                if len(loc) == 4:
                    p_idx, rs, re_, _tier = int(loc[0]), int(loc[1]), int(loc[2]), str(loc[3])
                elif len(loc) == 3:
                    p_idx, rs, re_ = int(loc[0]), int(loc[1]), int(loc[2])
                else:
                    p_idx, rs, re_ = 0, int(loc[0]), int(loc[1])
                raw_p = ps[p_idx] if 0 <= p_idx < len(ps) else ps[0]
                span_s = raw_p[rs:re_]
                bc = _check_span_boundary(raw_p, rs, re_)
                if bc["para_or_table_md"]:
                    bc_para_or_table_count += 1
                es = _check_span_edge_slicing(raw_p, rs, re_, q)
                if es["any_sign_or_escape"]:
                    es_any_count += 1
                if not _check_balanced_brackets(span_s, ignore_literals=False):
                    unbal_raw_span_count += 1
            else:
                span_s = q
                if not _check_balanced_brackets(span_s, ignore_literals=False):
                    unbal_raw_span_count += 1

    accepted_norm_lens.sort()
    shadow_m1_pass_rate = (m1_ok_all / total_quotes) if total_quotes > 0 else 1.0
    boundary_crossing_rate = (bc_para_or_table_count / m2a_ok_all) if m2a_ok_all > 0 else 0.0
    unbalanced_brackets_rate = (unbal_raw_span_count / m2a_ok_all) if m2a_ok_all > 0 else 0.0
    quote_len_p10 = pct_linear(accepted_norm_lens, 0.10) if m2a_ok_all > 0 else None

    return {
        "total_emitted_quotes": total_quotes,
        "m1_matched_quotes": m1_ok_all,
        "m2a_matched_quotes": m2a_ok_all,
        "shadow_m1_pass_rate": round(shadow_m1_pass_rate, 6),
        "boundary_crossing_count": bc_para_or_table_count,
        "boundary_crossing_rate": round(boundary_crossing_rate, 6),
        "sign_escape_edge_slice_count": es_any_count,
        "sign_escape_edge_slice_flagged_for_review": es_any_count >= 1,
        "unbalanced_brackets_count": unbal_raw_span_count,
        "unbalanced_brackets_rate": round(unbalanced_brackets_rate, 6),
        "post_clamp_wrong_positive_count": int(post_clamp_wrong_positive_count),
        "accepted_quote_norm_len_p10": round(quote_len_p10, 4) if quote_len_p10 is not None else None,
    }


def evaluate_canary_rollback_v11(
    canaries: Mapping[str, Any],
    prev_streaks: Mapping[str, int] | None = None,
    *,
    cfg: S2V11Config = S2V11,
) -> dict[str, Any]:
    """Evaluate §6 canary rollback lines and update consecutive-window violation streaks."""
    prev = dict(prev_streaks or {})
    total_quotes = int(canaries.get("total_emitted_quotes", 0))
    m2a_quotes = int(canaries.get("m2a_matched_quotes", 0))

    v_shadow_m1 = (
        total_quotes > 0
        and float(canaries.get("shadow_m1_pass_rate", 1.0)) < cfg.canary_shadow_m1_min_pass_rate
    )
    v_boundary = (
        m2a_quotes > 0
        and float(canaries.get("boundary_crossing_rate", 0.0)) > cfg.canary_boundary_crossing_max_rate
    )
    v_edge_slice = (
        int(canaries.get("sign_escape_edge_slice_count", 0))
        >= cfg.canary_sign_escape_edge_slice_min_count
    )
    v_unbalanced = (
        m2a_quotes > 0
        and float(canaries.get("unbalanced_brackets_rate", 0.0))
        > cfg.canary_unbalanced_brackets_max_rate
    )
    p10_val = canaries.get("accepted_quote_norm_len_p10")
    v_p10 = (
        m2a_quotes > 0
        and p10_val is not None
        and float(p10_val) < cfg.canary_quote_len_p10_min_chars
    )

    violations = {
        "shadow_m1_pass_rate": bool(v_shadow_m1),
        "boundary_crossing_rate": bool(v_boundary),
        "sign_escape_edge_slice_count": bool(v_edge_slice),
        "unbalanced_brackets_rate": bool(v_unbalanced),
        "accepted_quote_norm_len_p10": bool(v_p10),
    }
    streaks: dict[str, int] = {}
    tripped_canaries: list[str] = []
    for k, violated in violations.items():
        s = (int(prev.get(k, 0)) + 1) if violated else 0
        streaks[k] = s
        if s >= cfg.canary_consecutive_windows:
            tripped_canaries.append(k)

    stop_and_report = len(tripped_canaries) > 0
    return {
        "violations": violations,
        "streaks": streaks,
        "tripped_canaries": tripped_canaries,
        "stop_and_report": stop_and_report,
        "verdict": "STOP-AND-REPORT" if stop_and_report else None,
    }


# ---------------------------------------------------------------------------
# §5 Poisson CUSUM and one-sided paired McNemar helpers
# ---------------------------------------------------------------------------


def update_poisson_cusum(prev_s: float, count: int, k: float) -> float:
    """Update one-sided CUSUM statistic $S_w = \\max(0, S_{w-1} + X_w - k)$."""
    return max(0.0, float(prev_s) + float(count) - float(k))


def mcnemar_one_sided_exact_p(n_worse_at_t: int, n_better_at_t: int) -> float:
    """Exact one-sided binomial McNemar p-value testing $P(\\text{worse at } t) > 0.5$.

    ``n_worse_at_t`` ($b$): items that were not in the bad state at step 0 and flipped into it at step $t$.
    ``n_better_at_t`` ($c$): items that were in the bad state at step 0 and flipped out of it at step $t$.
    Returns $\\sum_{k=b}^{b+c} \\binom{b+c}{k} 0.5^{b+c}$ (or ``1.0`` when $b == 0$).
    """
    b = int(n_worse_at_t)
    c = int(n_better_at_t)
    if b < 0 or c < 0:
        raise ValueError(f"Discordant counts must be non-negative, got b={b}, c={c}")
    n = b + c
    if n == 0 or b == 0:
        return 1.0
    tail = 0.0
    for k in range(b, n + 1):
        tail += math.comb(n, k)
    return min(1.0, math.ldexp(tail, -n))


def simulate_constant_exposure_cusum(
    lam_win: float,
    k: float,
    h: float,
    *,
    n_windows: int = S2V11.windows_per_run,
    n_runs: int = 100_000,
    seed: int = 20261108,
) -> dict[str, Any]:
    """Monte Carlo simulation of a constant-exposure Poisson CUSUM over ``n_windows`` windows."""
    rng = random.Random(seed)
    x_max = max(50, int(lam_win + 12.0 * math.sqrt(lam_win + 1.0) + 20))
    cdf: list[float] = []
    acc = 0.0
    for x in range(x_max + 1):
        p = (
            math.exp(-lam_win + x * math.log(lam_win) - math.lgamma(x + 1))
            if lam_win > 0.0
            else (1.0 if x == 0 else 0.0)
        )
        acc += p
        cdf.append(min(1.0, acc))
    cdf[-1] = 1.0

    import bisect

    alarms = 0
    for _ in range(n_runs):
        s = 0.0
        for _w in range(n_windows):
            u = rng.random()
            x = bisect.bisect_left(cdf, u)
            s = s + x - k
            if s < 0.0:
                s = 0.0
            if s >= h - 1e-12:
                alarms += 1
                break
    rate = alarms / n_runs
    mc_se = math.sqrt(rate * (1.0 - rate) / n_runs)
    return {
        "lam_win": lam_win,
        "k": k,
        "h": h,
        "n_windows": n_windows,
        "n_runs": n_runs,
        "seed": seed,
        "alarms": alarms,
        "rate": round(rate, 6),
        "mc_se": round(mc_se, 6),
    }


def build_v11_canary_sets(items: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Derive the 184-item gold-``unsupported`` pool and 128-item clean canary set $C$ (§3).

    Follows ``v11-baselines.py:218-232, 265-279``:
    - ``gold_unsupported_ids``: all items in ``items`` with ``label == "unsupported"`` (184 on the 394-item pool).
    - ``canary_c_ids``: gold-``unsupported`` items not in ``PASS_A_PASS_B_DIFF_EXCLUDED_IDS_V11`` (128 on the 394-item pool, containing ``diff-rnd-e839712-2u``).
    """
    gold_unsup: list[str] = []
    canary_c: list[str] = []
    for it in items:
        cid = str(it.get("id", ""))
        label = str(it.get("label", ""))
        if label == S2V11.canary_c_label:
            gold_unsup.append(cid)
            if cid not in PASS_A_PASS_B_DIFF_EXCLUDED_IDS_V11:
                canary_c.append(cid)
    return {
        "gold_unsupported_ids": gold_unsup,
        "canary_c_ids": canary_c,
    }

