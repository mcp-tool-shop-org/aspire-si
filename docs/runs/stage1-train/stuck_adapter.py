"""The frozen stuck rule file, as stuck.judge reads it.

stage1-stuck-rule.json (sha256 716dc251...) and stuck.py (8097b039...) are both frozen, and they name the same
numbers differently: the file has p95_thinking_tokens_by_tier and burst_parameters {window_tokens, k}; judge reads
p95_by_tier and burst {window, k}. This maps one onto the other. It is the same mapping as role-set's
role_set/stuckrule.py rule_for_judge, so the training-time stuck flags and role-set's draft measures agree.
"""

import hashlib
from pathlib import Path

import stuck

RULE_SHA256 = "716dc251965973c61b74287e077dc8ae68c7d8c2a1bfccc87bde183d2ff1c1a1"
STUCK_PY_SHA256 = "8097b0398ade5e57989ee78f4d9a74a134f2aaefef358c19be0daaad8df13afa"


def rule_for_judge(f: dict) -> dict:
    return {"p95_by_tier": f["p95_thinking_tokens_by_tier"], "reread_rate_p95": f["reread_rate_p95"],
            "burst": {"window": f["burst_parameters"]["window_tokens"], "k": f["burst_parameters"]["k"]}}


def load_frozen(path: Path = stuck.RULE_FILE) -> dict:
    """The frozen rule, checked against its pinned sha256 (and stuck.py against its own), mapped for judge."""
    for p, want in ((Path(path), RULE_SHA256), (Path(stuck.__file__), STUCK_PY_SHA256)):
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{p.name} sha256 {got[:8]} != pinned {want[:8]}")
    return rule_for_judge(stuck.load_rule(path))
