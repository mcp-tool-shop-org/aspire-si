"""Smoke test S1 of the verifier pre-registration (docs/runs/2026-10-10-verifier-distill-prereg-draft.md):
can nli-deberta-v3-base on the Intel side tell a real supporting quote from one that fooled a verifier?

Pairs come from the R&D session's committed tune calibration verdicts (rnd
experiments/verifier-gold/calibration/results/2026-10-09-chain), by its rule:
- lines with model_verdict "supported", quote_found true and an evidence_quote;
- deduped on (claim_id, whitespace-normalised quote);
- gold supported is a positive; gold unsupported or cannot_tell is "fooled", split by gold.

Pass, per check type (fixed before the run):
- entailment on positives >= 0.85;
- entailment on gold-unsupported fooled pairs, Wilson 95% upper <= 0.15.
Gold-cannot_tell fooled pairs are reported, never gated.

Device rules (the intel-npu skill): batch 1, static shape 1x512; NPU or the Intel iGPU by name, never the
RTX 5090; a health check before and after; a per-call timeout that stops the run if the device hangs.

  <npu-openvino python> s1_nli_critic.py --chain <dir> --gold <dir> --rnd-commit <sha>
      --device NPU --out <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
NLI_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"  # apache-2.0, card read 2026-10-10
SEQ = 512
Z = 1.959963984540054
POSITIVE_FLOOR = 0.85
FOOLED_UPPER = 0.15


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p, z2 = k / n, Z * Z
    c = (p + z2 / (2 * n)) / (1 + z2 / n)
    h = Z * ((p * (1 - p) / n + z2 / (4 * n * n)) ** 0.5) / (1 + z2 / n)
    return (0.0 if k == 0 else max(0.0, c - h), 1.0 if k == n else min(1.0, c + h))


def normalise(quote: str) -> str:
    return re.sub(r"\s+", " ", quote).strip()


GOLD_FILES = ("grounded.jsonl", "prs/grounded-prs.jsonl", "diffs/reasoning-diffs.jsonl")


def load_claims(gold_dir: Path) -> dict[str, str]:
    """claim_id -> claim text, from the verifier gold files (the verdict lines carry only the id)."""
    claims = {}
    for name in GOLD_FILES:
        for line in (gold_dir / name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                claims[row["id"]] = row["claim"]
    return claims


def build_pairs(lines: list[dict], claims: dict[str, str]) -> list[dict]:
    """The R&D session's pairing rule. First occurrence of each (claim_id, normalised quote) wins.
    A claim id missing from the gold stops the build rather than being skipped."""
    pairs, seen = [], set()
    for r in lines:
        quote = r.get("evidence_quote")
        if r.get("model_verdict") != "supported" or not r.get("quote_found") or not quote:
            continue
        key = (r["claim_id"], normalise(quote))
        if key in seen:
            continue
        seen.add(key)
        gold = r["gold_label"]
        role = "positive" if gold == "supported" else f"fooled:{gold}"
        if r["claim_id"] not in claims:
            raise ValueError(f"claim {r['claim_id']} is not in the gold files")
        pairs.append(
            {
                "claim_id": r["claim_id"],
                "check_type": r["check_type"],
                "claim": claims[r["claim_id"]],
                "quote": key[1],
                "gold": gold,
                "role": role,
            }
        )
    return pairs


def readout(pairs: list[dict], labels: dict[tuple[str, str], str]) -> dict:
    """Per check type: entailment rates with Wilson intervals, and the pass decision."""
    out = {}
    for ct in sorted({p["check_type"] for p in pairs}):
        mine = [p for p in pairs if p["check_type"] == ct]
        block = {}
        for role in ("positive", "fooled:unsupported", "fooled:cannot_tell"):
            group = [p for p in mine if p["role"] == role]
            k = sum(labels.get((p["claim_id"], p["quote"])) == "entailment" for p in group)
            scored = sum((p["claim_id"], p["quote"]) in labels for p in group)
            lo, hi = wilson(k, scored)
            block[role] = {
                "n": len(group),
                "scored": scored,
                "entailed": k,
                "rate": None if not scored else k / scored,
                "wilson95": [lo, hi],
            }
        pos, fool = block["positive"], block["fooled:unsupported"]
        complete = pos["scored"] == pos["n"] and fool["scored"] == fool["n"]
        block["pass"] = bool(
            complete
            and pos["n"]
            and fool["n"]
            and pos["rate"] >= POSITIVE_FLOOR
            and fool["wilson95"][1] <= FOOLED_UPPER
        )
        block["complete"] = complete
        out[ct] = block
    return out


def pick_device(core, name: str) -> str:
    """NPU or CPU by name; "iGPU" only as an Intel GPU found by its full name. Never the RTX 5090."""
    if name in ("NPU", "CPU"):
        if name not in core.available_devices:
            raise SystemExit(f"{name} not visible to OpenVINO")
        return name
    if name == "iGPU":
        for d in core.available_devices:
            if d.startswith("GPU") and core.get_property(d, "FULL_DEVICE_NAME").startswith("Intel"):
                return d
        raise SystemExit("no Intel iGPU found; refusing to guess a GPU device (it could be the 5090)")
    raise SystemExit(f"refusing device {name!r}: use NPU, iGPU or CPU")


def npu_health() -> str:  # pragma: no cover - needs the rig
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command", "(Get-PnpDevice -FriendlyName 'Intel(R) AI Boost').Status"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return out.stdout.strip() or out.stderr.strip()


def call_with_timeout(fn, seconds: float):
    """Run fn in a thread; (result, None) or (None, "timeout"). A timed-out call means the device may hang."""
    box = {}

    def run():
        try:
            box["value"] = fn()
        except Exception as exc:  # noqa: BLE001 - recorded, the run stops
            box["error"] = repr(exc)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        return None, "timeout"
    if "error" in box:
        return None, box["error"]
    return box["value"], None


def main() -> None:  # pragma: no cover - needs the rig
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chain", type=Path, required=True, help="the calibration results directory")
    ap.add_argument(
        "--gold", type=Path, required=True, help="rnd experiments/verifier-gold at the same commit"
    )
    ap.add_argument(
        "--rnd-commit", required=True, help="the rnd commit the chain and gold are checked out at"
    )
    ap.add_argument("--device", choices=("NPU", "iGPU", "CPU"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per call before the run stops")
    ap.add_argument("--limit", type=int, default=None, help="score only the first N pairs (a dry check)")
    a = ap.parse_args()

    import numpy as np
    import openvino as ov
    from optimum.intel import OVModelForSequenceClassification
    from transformers import AutoTokenizer

    files = sorted(a.chain.glob("*/verdicts.jsonl"))
    lines, receipts = [], {}
    for f in files:
        data = f.read_bytes()
        receipts[f"{f.parent.name}/{f.name}"] = hashlib.sha256(data).hexdigest()
        lines += [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()]
    pairs = build_pairs(lines, load_claims(a.gold))
    if a.limit:
        pairs = pairs[: a.limit]

    core = ov.Core()
    device = pick_device(core, a.device)
    health_before = npu_health()
    if a.device == "NPU" and health_before != "OK":
        raise SystemExit(f"NPU health is {health_before!r}; not starting")
    tok = AutoTokenizer.from_pretrained(NLI_MODEL, revision=NLI_REVISION)
    model = OVModelForSequenceClassification.from_pretrained(
        NLI_MODEL, revision=NLI_REVISION, export=True, compile=False
    )
    model.reshape(1, SEQ)
    model.to(device)
    t0 = time.time()
    model.compile()
    compile_s = time.time() - t0
    names = [model.config.id2label[i].lower() for i in range(len(model.config.id2label))]

    labels, latencies, stopped = {}, [], None
    for p in pairs:
        enc = tok(
            [p["quote"]],
            [p["claim"]],
            padding="max_length",
            truncation=True,
            max_length=SEQ,
            return_tensors="np",
        )
        t = time.time()
        logits, err = call_with_timeout(lambda: np.asarray(model(**enc).logits[0]), a.timeout)
        if err:
            stopped = {"pair": [p["claim_id"], p["quote"][:60]], "error": err}
            break
        latencies.append(time.time() - t)
        labels[(p["claim_id"], p["quote"])] = names[int(np.argmax(logits))]
    health_after = npu_health()

    body = {
        "test": "S1",
        "model": NLI_MODEL,
        "revision": NLI_REVISION,
        "device": a.device,
        "ov_device": device,
        "ov_device_name": core.get_property(device, "FULL_DEVICE_NAME"),
        "openvino": ov.__version__,
        "shape": [1, SEQ],
        "rnd_commit": a.rnd_commit,
        "verdict_files_sha256": receipts,
        "pairs": len(pairs),
        "scored": len(labels),
        "stopped": stopped,
        "compile_s": round(compile_s, 2),
        "latency_s": {
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
            "n": len(latencies),
        }
        if latencies
        else None,
        "npu_health": {"before": health_before, "after": health_after},
        "pass_line": {"positive_entail_min": POSITIVE_FLOOR, "fooled_unsupported_upper_max": FOOLED_UPPER},
        "readout": readout(pairs, labels),
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"s1-{a.device}.json").write_text(json.dumps(body, indent=1), encoding="utf-8", newline="\n")
    (a.out / f"s1-{a.device}-labels.jsonl").write_text(
        "".join(
            json.dumps({"claim_id": c, "quote": q, "label": lab}) + "\n" for (c, q), lab in labels.items()
        ),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({k: body[k] for k in ("device", "scored", "stopped", "latency_s", "npu_health")}))
    print(json.dumps(body["readout"], indent=1))
    print("S1-OK" if not stopped else "S1-STOPPED")


if __name__ == "__main__":
    sys.exit(main())
