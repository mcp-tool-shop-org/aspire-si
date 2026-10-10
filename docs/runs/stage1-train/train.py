"""Stage-1 trainer: LoRA SFT on the student, self-contained (torch + peft only, no TRL, no new installs).

    python train.py --data <items.jsonl> --out <round dir>
                    [--init-adapter <dir>] [--held-out a.jsonl b.jsonl ...] [--max-steps N] [--dry-run]

- Loads the student through stage1-eval/student.py (the same loader and weights check as the evaluation).
- LoRA rank 64, alpha 128, dropout 0.05 on every linear layer; AdamW lr 1e-4, cosine with 3% warm-up,
  effective batch 16 sequences, max length 4096, gradient checkpointing, bf16.
- Loss on the assistant target only. One sequence per assistant turn (see data.py).
- Refuses the data on any schema error, a leakage hit against the held-out sets or the pre-interview, or a
  sequence over the length limits (target <= 3,000 tokens, whole sequence <= 4,096): refused, never truncated.
- Writes: adapter/, manifest.json (data sha256s, seed, settings, adapter sha256, machine line, timing),
  loss.jsonl. Resumable from the last checkpoint in the round dir.
"""

import argparse
import hashlib
import os
import json
import math
import platform
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
import data  # noqa: E402
import student  # noqa: E402

LORA = {"r": 64, "lora_alpha": 128, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]}
VRAM_CAP_GB = 28  # self-stop line; jobs are planned at or below about 26 GB
OPT = {"lr": 1e-4, "warmup_frac": 0.03, "weight_decay": 0.0, "effective_batch": 16, "max_len": 4096,
       "max_target": 3000, "epochs": 1, "seed": 0}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def code_hash() -> str:
    h = hashlib.sha256()
    for f in ("train.py", "data.py", "_evaldir.py"):
        h.update((HERE / f).read_bytes())
    h.update((EVAL_DIR / "student.py").read_bytes())
    return h.hexdigest()


def machine() -> dict:
    import torch
    import transformers
    return {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda": torch.version.cuda, "torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version()}


def build(rows, tok, max_len, max_target):
    seqs, refused = [], []
    for r in rows:
        for s in data.sequences(r, tok):
            n, t = len(s["prompt_ids"]) + len(s["target_ids"]), len(s["target_ids"])
            if n > max_len or t > max_target:
                refused.append((r["id"], n, t))
                continue
            seqs.append({**s, "id": r["id"]})
    if refused:
        raise SystemExit(f"items over the limits (sequence <= {max_len}, target <= {max_target} tokens; shorten "
                         f"or split them, never truncate): {refused[:10]}")
    return seqs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--init-adapter")
    ap.add_argument("--held-out", nargs="*", default=[])
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    rows = [r for p in a.data for r in data.load(p)]
    data.validate(rows)
    held = [h for p in a.held_out for h in data.load(p)]
    from interview import QUESTIONS  # the pre-interview's 12 questions are off-limits to training
    leaks = data.leakage(rows, held, QUESTIONS)
    if leaks:
        raise SystemExit("leakage against the held-out sets:\n" + "\n".join(leaks[:30]))
    tok = student.load_tokenizer()
    seqs = build(rows, tok, OPT["max_len"], OPT["max_target"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"code_sha256": code_hash(), "student": student.identity(), "lora": LORA, "opt": OPT,
                "data": {p: sha256_file(Path(p)) for p in a.data},
                "held_out": {p: sha256_file(Path(p)) for p in a.held_out}, "examples": len(rows),
                "sequences": len(seqs), "target_tokens": sum(len(s["target_ids"]) for s in seqs),
                "init_adapter": a.init_adapter}
    if a.dry_run:
        print(json.dumps({k: v for k, v in manifest.items() if k != "student"}, indent=1))
        return 0

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model

    print(json.dumps(student.verify_weights()), flush=True)
    # The trainer stops itself at 28 GB (the maintainer's policy): past this limit an allocation fails with a
    # clean out-of-memory error, well before the rig's watchdog line at 31.2 GB.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_CAP_GB * 2**30 / total))
    random.seed(OPT["seed"])
    torch.manual_seed(OPT["seed"])
    model = student.load_student()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    if a.init_adapter:
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", bias="none", **LORA))
    model.train()
    inner = model.get_base_model()          # the causal-LM with the LoRA layers injected
    backbone, lm_head = inner.model, inner.lm_head
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=OPT["lr"], weight_decay=OPT["weight_decay"])

    order = list(range(len(seqs)))
    random.shuffle(order)
    accum = OPT["effective_batch"]
    total_steps = math.ceil(len(order) * OPT["epochs"] / accum)
    if a.max_steps:
        total_steps = min(total_steps, a.max_steps)
    warm = max(1, int(total_steps * OPT["warmup_frac"]))

    def lr_at(step):
        if step < warm:
            return OPT["lr"] * (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return OPT["lr"] * 0.5 * (1 + math.cos(math.pi * prog))

    ckpt = out / "checkpoint"
    start_step, prior_seconds = 0, 0.0
    if (ckpt / "state.json").exists():
        st = json.loads((ckpt / "state.json").read_text(encoding="utf-8"))
        if st["code_sha256"] != manifest["code_sha256"] or st["data"] != manifest["data"]:
            raise SystemExit("refusing to resume: code or data differ from the checkpoint")
        model.load_adapter(str(ckpt / "adapter"), adapter_name="default", is_trainable=True)
        opt.load_state_dict(torch.load(ckpt / "optimizer.pt", map_location="cuda"))
        start_step = st["step"]
        prior_seconds = st.get("train_seconds", 0.0)
        print("resumed at step", start_step, flush=True)

    t0 = time.time()
    loss_log = open(out / "loss.jsonl", "a", encoding="utf-8")
    dev = model.device
    for step in range(start_step, total_steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        batch = [seqs[order[(step * accum + k) % len(order)]] for k in range(accum)]
        tgt_total = sum(len(s["target_ids"]) for s in batch)
        step_loss = 0.0
        seq_peaks = []
        for s in batch:
            torch.cuda.reset_peak_memory_stats()
            ids = torch.tensor([s["prompt_ids"] + s["target_ids"]], device=dev)
            n_p, n_t = len(s["prompt_ids"]), len(s["target_ids"])
            # The LM head runs only on the positions that predict the target: full-vocabulary logits over a
            # 4k-token prompt cost several GB (the first smoke tripped the VRAM watchdog on exactly that).
            hidden = backbone(input_ids=ids).last_hidden_state[:, n_p - 1:n_p - 1 + n_t]
            logits = lm_head(hidden).float()
            tgt = torch.tensor([s["target_ids"]], device=dev)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                                                     reduction="sum") / tgt_total
            loss.backward()
            step_loss += loss.item()
            del hidden, logits, loss
            seq_peaks.append([n_p, n_t, round(torch.cuda.max_memory_allocated() / 2**30, 2)])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        rec = {"step": step + 1, "of": total_steps, "loss": round(step_loss, 5), "lr": lr_at(step),
               "elapsed_s": round(time.time() - t0, 1),
               "peak_vram_gb": max(x[2] for x in seq_peaks),
               "seq_peaks_prompt_target_gb": sorted({tuple(x) for x in seq_peaks})}
        loss_log.write(json.dumps(rec) + "\n")
        loss_log.flush()
        print(json.dumps(rec), flush=True)
        if (step + 1) % 25 == 0 or step + 1 == total_steps:
            ckpt.mkdir(exist_ok=True)
            model.save_pretrained(ckpt / "adapter")
            torch.save(opt.state_dict(), ckpt / "optimizer.pt")
            (ckpt / "state.json").write_text(json.dumps({"step": step + 1, "code_sha256": manifest["code_sha256"],
                                                         "data": manifest["data"],
                                                         "train_seconds": prior_seconds + time.time() - t0}),
                                             encoding="utf-8")
    elapsed = prior_seconds + time.time() - t0  # cumulative across resumes
    model.save_pretrained(out / "adapter")
    adapter_file = out / "adapter" / "adapter_model.safetensors"
    manifest.update({"steps": total_steps, "machine": machine(), "train_seconds": round(elapsed, 1),
                     "gpu_minutes": round(elapsed / 60, 1),
                     "examples_per_minute": round(min(total_steps * accum, len(seqs) * OPT["epochs"])
                                                  / max(1e-9, elapsed / 60), 2),
                     "peak_vram_gb": max(json.loads(line)["peak_vram_gb"]
                                         for line in open(out / "loss.jsonl", encoding="utf-8")),
                     "adapter_sha256": sha256_file(adapter_file) if adapter_file.exists() else None})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print("done", json.dumps({k: manifest[k] for k in ("steps", "train_seconds", "peak_vram_gb", "adapter_sha256")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
