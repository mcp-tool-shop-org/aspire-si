"""Stage-1 eval harness: the same code runs the baseline (before training) and every later measurement.

    python harness.py --tasks <tasks.jsonl> --out <run dir> [--decoding sampled|greedy] [--seed N] [--limit N] [--dry-run]

Per task it records the full thinking, the final answer, token counts, truncation and timing (`raw.jsonl`), then
writes the mapped-tree records (`mapped-steps.jsonl`, `mapped-traces.jsonl`) in the plan's format. Resumable: a
task already in raw.jsonl is skipped, and a run refuses to continue under a different harness hash.
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import student

PROMPT_VERSION = "stage1-eval-v1"
# Role-neutral at baseline: it names the task and the answer format and teaches nothing about how to verify.
PROMPT = """Read the material below and decide whether the statement is supported by it.

Material:
<<<
{material}
>>>

Statement: {claim}

Answer in exactly three lines:
VERDICT: supported | unsupported | cannot_tell
DECIDING: the line or lines from the material that decide it, copied word for word and separated by " | " (at most 4), or NONE
ESCALATE: no | yes (a short reason), where yes means a person should review this before anyone acts on it"""

DECODING = {
    # Qwen3's model card for thinking mode: temperature 0.6, top-p 0.95, top-k 20, min-p 0; no greedy decoding.
    "sampled": {"do_sample": True, "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    "greedy": {"do_sample": False},
}
MAX_NEW_TOKENS = 16384
THINKING = True

HERE = Path(__file__).resolve().parent


def harness_hash() -> str:
    h = hashlib.sha256()
    for f in ("harness.py", "student.py", "mapper.py"):
        h.update((HERE / f).read_bytes())
    return h.hexdigest()


def settings(decoding: str, seed: int) -> dict:
    return {"prompt_version": PROMPT_VERSION, "prompt_sha256": hashlib.sha256(PROMPT.encode()).hexdigest(),
            "decoding": decoding, "generate": DECODING[decoding], "seed": seed, "max_new_tokens": MAX_NEW_TOKENS,
            "thinking": THINKING, "student": student.identity(), "harness_sha256": harness_hash()}


def split_thinking(text: str) -> tuple[str, str, bool]:
    """(thinking, answer, thinking_closed)."""
    if "</think>" in text:
        think, answer = text.split("</think>", 1)
        return think.replace("<think>", "").strip(), answer.strip(), True
    return text.replace("<think>", "").strip(), "", False


def parse_answer(answer: str) -> dict:
    def line(tag):
        m = re.search(rf"^\s*{tag}\s*:\s*(.*)$", answer, re.M | re.I)
        return m.group(1).strip() if m else None
    verdict = (line("VERDICT") or "").lower().strip(" .*`")
    deciding = line("DECIDING")
    escalate = line("ESCALATE")
    return {"verdict": verdict if verdict in ("supported", "unsupported", "cannot_tell") else None,
            "verdict_raw": line("VERDICT"), "deciding": deciding, "escalate": escalate}


def grade(task: dict, parsed: dict) -> dict:
    key = task["key"]
    segs = [] if not parsed["deciding"] or parsed["deciding"].strip().upper() == "NONE" else \
        [s.strip().strip('"') for s in parsed["deciding"].split(" | ") if s.strip()]
    esc = (parsed["escalate"] or "").lower()
    esc_yes = esc.startswith("yes")
    return {"verdict_correct": parsed["verdict"] == key["verdict"],
            "deciding_verbatim": all(s in task["material"] for s in segs) if segs else None,
            "deciding_none": not segs,
            "escalate_correct": (esc_yes == key["escalate"].startswith("yes")) if parsed["escalate"] else False}


def generate(model, tok, messages, decoding, seed):
    import torch
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=THINKING)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)  # re-seeded before every generation
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS, **DECODING[decoding])
    new = out[0, ids["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    ended = bool(len(new)) and int(new[-1]) in (tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>"))
    return {"text": text, "prompt_tokens": int(ids["input_ids"].shape[1]), "new_tokens": int(len(new)),
            "truncated": not ended and len(new) >= MAX_NEW_TOKENS, "seconds": round(time.time() - t0, 2)}


def run_task(model, tok, task, decoding, seed):
    messages = [{"role": "user", "content": PROMPT.format(material=task["material"], claim=task["claim"])}]
    turns = []
    for k, follow in enumerate([None] + task.get("turns", [])):
        if follow is not None:
            messages.append({"role": "user", "content": follow})
        g = generate(model, tok, messages, decoding, seed)
        think, answer, closed = split_thinking(g["text"])
        turns.append({**g, "turn": k + 1, "thinking": think, "answer": answer, "thinking_closed": closed})
        messages.append({"role": "assistant", "content": answer or g["text"]})
    final = turns[-1]
    parsed = parse_answer(final["answer"])
    return {"task_id": task["id"], "category": task["category"], "tier": task["tier"], "turns": turns,
            "final": parsed, "grade": grade(task, parsed)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--decoding", choices=sorted(DECODING), default="sampled")
    ap.add_argument("--phase", default="A-before")
    ap.add_argument("--arm", default="BASE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true", help="check tasks, prompts and settings; no model")
    a = ap.parse_args(argv)

    tasks = [json.loads(line) for line in open(a.tasks, encoding="utf-8")][: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = {**settings(a.decoding, a.seed), "tasks_file": Path(a.tasks).name,
           "tasks_sha256": hashlib.sha256(Path(a.tasks).read_bytes()).hexdigest(), "phase": a.phase, "arm": a.arm}
    cfg_path = out / "run-config.json"
    if cfg_path.exists():
        old = json.loads(cfg_path.read_text(encoding="utf-8"))
        if old != cfg:
            diff = {k: (old.get(k), cfg.get(k)) for k in set(old) | set(cfg) if old.get(k) != cfg.get(k)}
            raise SystemExit(f"refusing to resume under different settings: {json.dumps(diff)[:600]}")
    else:
        cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")

    if a.dry_run:
        tok = student.load_tokenizer()
        for t in tasks:
            p = tok.apply_chat_template([{"role": "user", "content": PROMPT.format(material=t["material"],
                                                                                   claim=t["claim"])}],
                                        tokenize=False, add_generation_prompt=True, enable_thinking=THINKING)
            n = len(tok(p)["input_ids"])
            print(t["id"], t["category"], "prompt tokens", n, "turns", 1 + len(t.get("turns", [])))
        print(json.dumps({k: v for k, v in cfg.items() if k != "student"}, indent=1))
        return 0

    print(json.dumps(student.verify_weights()), flush=True)
    raw = out / "raw.jsonl"
    done = {json.loads(line)["task_id"] for line in open(raw, encoding="utf-8")} if raw.exists() else set()
    tok = student.load_tokenizer()
    model = student.load_student()
    for t in tasks:
        if t["id"] in done:
            continue
        rec = run_task(model, tok, t, a.decoding, a.seed)
        rec.update({"phase": a.phase, "arm": a.arm, "seed": a.seed, "harness_sha256": cfg["harness_sha256"]})
        with open(raw, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        last = rec["turns"][-1]
        print(t["id"], rec["final"]["verdict"], "ok" if rec["grade"]["verdict_correct"] else "WRONG",
              sum(x["new_tokens"] for x in rec["turns"]), "tok", round(sum(x["seconds"] for x in rec["turns"])), "s",
              "TRUNC" if any(x["truncated"] for x in rec["turns"]) else "", flush=True)
    import mapper
    mapper.map_run(out, tasks, tok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
