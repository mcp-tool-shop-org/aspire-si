"""Stage 1 general-capability panel runner (design §6 and §9).

    python panel.py sit --items <item-file> --out <readout.json> [--adapter <dir>]
    python panel.py guard --p0 <readout.json> --seed <readout.json> --out <guard.json>

CUBLAS_WORKSPACE_CONFIG is set before torch is imported. The scorer pin is checked before
a sitting and before a guard check. The machine line and the 26 GB cap are checked before
the model loads. A driver reset voids the sitting and writes no readout.

P0 is not run here. The readout records wall-clock minutes so the 20-minute rule can be
read off that sitting. The item count is not cut.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch initialises CUDA

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import batched  # noqa: E402  # imports torch, after CUBLAS is set
import det_run  # noqa: E402
import eval_adapter  # noqa: E402
import panel_build  # noqa: E402
import panel_guard  # noqa: E402
import panel_score  # noqa: E402
import parity  # noqa: E402

# Raw sha256 of panel_score.py. docs/runs/stage1-train is -text, so git stores these bytes.
PANEL_SCORE_SHA256 = "c26da86890c3028e79ccbeeb11faf8b1e4c1143e36b2f78be8c21f44fd628573"
MACHINE_LINE = HERE.parent / "stage1-machine-line.json"
MACHINE_LINE_SHA256 = "2daa5aa99137c6b07696913585321a7b6d12431fdaf4b9c14c3e294f0cf43830"
_WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_scorer() -> str:
    """Refuse a missing or changed scorer. A file cannot hold its own hash, so the pin lives here."""
    path = Path(getattr(panel_score, "__file__", "") or "")
    if not path.is_file():
        raise SystemExit("panel scorer is missing")
    got = sha256_file(path)
    if got != PANEL_SCORE_SHA256:
        raise SystemExit("panel scorer sha256 does not match the pin")
    return got


def check_machine(path: Path | None = None) -> dict:
    """Exit 5 when the frozen line's bytes, completeness, or fields do not match. No warn-only path."""
    path = Path(path) if path else MACHINE_LINE
    if not path.is_file() or sha256_file(path) != MACHINE_LINE_SHA256:
        raise SystemExit(5)
    frozen = json.loads(path.read_text(encoding="utf-8"))
    now = det_run.machine_line()
    if not now.get("gpu") or not now.get("driver") or det_run.line_differences(now, frozen):
        raise SystemExit(5)
    return now


def memory_cap(total_bytes: float, used_at_start: float | None, set_fraction) -> float:
    """`parity.cap_fraction`. A missing used-byte reading is a stop, never a fallback to 0."""
    if used_at_start is None:
        raise SystemExit("nvidia-smi did not report memory.used; refusing to treat that as 0")
    fraction = parity.cap_fraction(total_bytes, used_at_start)
    set_fraction(fraction)
    return fraction


def apply_memory_cap() -> float:
    import torch
    torch.zeros(1, device="cuda")
    used = parity.gpu_used_bytes()
    total = torch.cuda.get_device_properties(0).total_memory
    return memory_cap(total, used, torch.cuda.set_per_process_memory_fraction)


def adapter_record(adapter: str | None) -> dict:
    """Base, or one seed's adapter. The readout stores the weights sha256 and not a path."""
    if not adapter:
        return {"model": "base", "adapter_sha256": None}
    path = Path(adapter)
    problem = eval_adapter.frozen_adapter_problem(path)
    if problem:
        raise SystemExit(f"refusing the adapter: {problem}")
    weight = next((path / name for name in _WEIGHTS if (path / name).is_file()), None)
    if weight is None:
        raise SystemExit("refusing the adapter: no weights file")
    return {"model": "adapter", "adapter_sha256": sha256_file(weight)}


def load_items(path: Path) -> tuple[dict, list[dict]]:
    """Kept rows in file order. That order is the order sent to the batched path."""
    meta = None
    kept = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("record") == "meta":
            meta = row
        elif row.get("decision") == "kept":
            kept.append(row)
    if meta is None or not kept:
        raise SystemExit("the item file has no meta record or no kept items")
    return meta, kept


def asks_for(items: list[dict]) -> list[dict]:
    """Bare item id and sample 0. The id is never suffixed with a round or a seed."""
    return [{"key": item["id"], "item_id": item["id"], "sample": 0,
             "messages": [{"role": "user", "content": item["prompt"]}]} for item in items]


def seed_of(item_id: str) -> int:
    return batched.seed_for(item_id, 0)


def score_sitting(items: list[dict], generations: dict) -> list[dict]:
    """Per-item records in the item file's order."""
    records = []
    for item in items:
        generation = generations[item["id"]]
        scored = panel_score.score_generation(generation, item["gold"], item["source"])
        records.append({"id": item["id"], "slice": item["source"], "seed": seed_of(item["id"]),
                        "text": generation.get("text") or "", "parsed": scored["parsed"],
                        "right": scored["right"], "truncated": scored["truncated"],
                        "unparsed": scored["unparsed"], "empty": scored["empty"],
                        "strict_one_line": scored["strict_one_line"],
                        "new_tokens": generation.get("new_tokens"), "seconds": generation.get("seconds")})
    return records


def readout_for(records: list[dict], *, item_sha: str, scorer_sha: str, model: str,
                adapter_sha: str | None, machine_line: dict, minutes: float, batched_sha: str) -> dict:
    slices = {}
    for name in ("gsm8k", "winogrande"):
        rows = [row for row in records if row["slice"] == name]
        correct = sum(1 for row in rows if row["right"])
        slices[name] = {"n": len(rows), "correct": correct,
                        "at_or_above_97": panel_guard.slice_at_or_above_97(correct, len(rows))}
    correct = sum(1 for row in records if row["right"])
    return {"model": model, "adapter_sha256": adapter_sha, "item_sha256": item_sha,
            "scorer_sha256": scorer_sha, "minutes": minutes, "over_20_minutes": minutes > 20,
            "machine_line": machine_line, "accuracy": {"correct": correct, "n": len(records)},
            "slices": slices, "truncated": sum(1 for row in records if row["truncated"]),
            "unparsed": sum(1 for row in records if row["unparsed"]),
            "empty": sum(1 for row in records if row["empty"]),
            "strict_one_line": sum(1 for row in records if row["strict_one_line"]),
            "generation": {"path": batched.PATH, "batched_sha256": batched_sha}, "items": records}


def write_readout(payload: dict, dest: Path, resets: int) -> None:
    """A driver reset voids the sitting. The readout is never written inside the repository."""
    if resets != 0:
        raise SystemExit("a driver reset voids this sitting; the readout was not written")
    panel_build.outside_repo(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")


def prepare(load, *, machine_path: Path | None = None, cap=None, verify=None):
    """Scorer pin, machine line, then the cap, then the model. A failed check does not load."""
    require_scorer()
    line = check_machine(machine_path)
    if cap is not None:
        cap()
    if verify is not None:
        verify()
    return load(line)


def sitting_view(readout: dict) -> dict:
    return {"item_sha256": readout["item_sha256"], "scorer_sha256": readout["scorer_sha256"],
            "items": [{"id": row["id"], "right": row["right"]} for row in readout["items"]]}


def guard_sittings(baseline: dict, sittings: list[dict]) -> dict:
    """The scorer pin is checked before any pairing."""
    require_scorer()
    return panel_guard.round_result(baseline, sittings)


def _batched_sha() -> str:
    return sha256_file(Path(batched.__file__))


def run_generation(items: list[dict], generate_many) -> tuple[list[dict], float]:
    """Send the file order in. Return records and the generation's wall-clock minutes."""
    t0 = time.time()
    generations = generate_many(asks_for(items))
    minutes = (time.time() - t0) / 60.0
    return score_sitting(items, generations), minutes


def sit_main(argv) -> int:
    parser = argparse.ArgumentParser(description="One panel sitting. P0 passes no --adapter.")
    parser.add_argument("--items", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--adapter", default=None)
    args = parser.parse_args(argv)
    out = Path(args.out)
    panel_build.outside_repo(out)
    scorer_sha = require_scorer()
    machine = check_machine()
    apply_memory_cap()
    import student
    student.verify_weights()
    record = adapter_record(args.adapter)
    from peft import PeftModel
    tok = student.load_tokenizer()
    model = student.load_student()
    if record["model"] == "adapter":
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    batch_log: list = []
    generate_many = batched.make_generate_many(model, tok, batch_log)
    meta_path = Path(args.items)
    _meta, items = load_items(meta_path)
    t0 = time.time()
    records, minutes = run_generation(items, generate_many)
    resets = parity.driver_resets_since(t0)
    payload = readout_for(records, item_sha=sha256_file(meta_path), scorer_sha=scorer_sha,
                          model=record["model"], adapter_sha=record["adapter_sha256"],
                          machine_line=machine, minutes=minutes, batched_sha=_batched_sha())
    payload["batches"] = len(batch_log)
    write_readout(payload, out, resets)
    print(json.dumps({"correct": payload["accuracy"]["correct"], "n": payload["accuracy"]["n"],
                      "minutes": round(minutes, 2), "over_20_minutes": payload["over_20_minutes"]}))
    return 0


def guard_main(argv) -> int:
    parser = argparse.ArgumentParser(description="Paired McNemar of each seed sitting against P0.")
    parser.add_argument("--p0", required=True)
    parser.add_argument("--seed", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out)
    panel_build.outside_repo(out)
    baseline = sitting_view(json.loads(Path(args.p0).read_text(encoding="utf-8")))
    sittings = [sitting_view(json.loads(Path(path).read_text(encoding="utf-8"))) for path in args.seed]
    result = guard_sittings(baseline, sittings)
    write_readout(result, out, resets=0)
    print(json.dumps({"where": result["where"], "holding_seeds": result["holding_seeds"]}))
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("sit", "guard"):
        raise SystemExit("usage: panel.py sit|guard ...")
    command, rest = argv[0], argv[1:]
    return sit_main(rest) if command == "sit" else guard_main(rest)


if __name__ == "__main__":
    sys.exit(main())
