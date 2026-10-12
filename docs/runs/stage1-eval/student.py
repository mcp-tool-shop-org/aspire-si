"""The stage-1 student: Qwen/Qwen3-8B at a pinned revision, loaded one way for evaluation and training alike.

The before/after comparison is valid only when both sides use the same weights and the same inference path, so
the eval harness and the trainer both call `load_student()` and nothing else loads the model.
"""

import hashlib
import json
import os
from pathlib import Path

REPO = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
DTYPE = "bfloat16"
ATTN = "sdpa"
# sha256 of every file in the snapshot (docs/runs/stage1-student-weights.md on #79).
WEIGHT_SHA256 = {
    "config.json": "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "model-00001-of-00005.safetensors": "31d6a825ae35f11fb85b195b4c42c146c051e446433125a215336abdf95cbf5f",
    "model-00002-of-00005.safetensors": "5991236cea6fe21f3d43cab0f0e84448734fbbe0789816202989f2ddc9d18282",
    "model-00003-of-00005.safetensors": "c5185c4794be2d8a9784d5753c9922db38df478ce11f9ed0b415b7304d896836",
    "model-00004-of-00005.safetensors": "b5ee7de71fbf17db3d5704e0c8f2bc7d005ca9e1d7ca2aeb19827b0cfcaa917a",
    "model-00005-of-00005.safetensors": "20c2d6366ab85c90786ccdd829cd2b9e7d30ef3b2ebbb998280e7e4014b542ff",
    "model.safetensors.index.json": "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}


def snapshot_dir() -> Path:
    if "HF_HOME" not in os.environ:
        raise SystemExit("set HF_HOME to the Hugging Face cache that holds the pinned snapshot")
    home = Path(os.environ["HF_HOME"])
    return home / "hub" / "models--Qwen--Qwen3-8B" / "snapshots" / REVISION


def verify_weights(d: Path | None = None) -> dict:
    """Refuse to go on unless every pinned file hashes as recorded."""
    d = d or snapshot_dir()
    bad = {}
    for name, want in WEIGHT_SHA256.items():
        h = hashlib.sha256()
        with open(d / name, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        if h.hexdigest() != want:
            bad[name] = h.hexdigest()
    if bad:
        raise SystemExit(f"weights do not match the pinned sha256: {json.dumps(bad)}")
    return {"repo": REPO, "revision": REVISION, "files_verified": len(WEIGHT_SHA256)}


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(snapshot_dir())


def load_student(device: str = "cuda"):
    """The one loader: bf16 weights, SDPA attention, eval mode. The trainer adds LoRA on top of this model."""
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(snapshot_dir(), dtype=getattr(torch, DTYPE),
                                                 attn_implementation=ATTN).to(device)
    model.eval()
    return model


def identity() -> dict:
    return {"repo": REPO, "revision": REVISION, "dtype": DTYPE, "attn": ATTN, "weight_sha256": WEIGHT_SHA256}
