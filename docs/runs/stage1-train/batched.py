"""Batched generation for the quiz and apply sittings (run sheet B4), as R&D settled it after Argon's review
(2026-10-15). DEV, sealed and the interview stay on the harness at batch 1, through det_run.py.

The rules (design note: docs/runs/2026-10-15-stage1-batched-quiz-design.md):
- **Per-item generators.** Each row samples from its own torch.Generator, seeded by (item_id, sample_idx). So a
  row's draws don't depend on its batch-mates or on the batch size. Only the numerics of a padded batch can move
  it, and parity (i) measures that.
- **The sampling** is the harness's card settings, imported, not copied: temperature, then top-k, then top-p
  (min-p 0 is a no-op), in transformers' order. The stop tokens are the generation config's, the same ones the
  harness checks. The cap is the harness's MAX_NEW_TOKENS over the whole answer.
- **Left padding with an explicit attention mask.** Position ids come from the mask, and finished rows are
  masked out of later steps.
- **SDPA pinned to EFFICIENT at every batch size,** the batch-1 arm of parity (i) included. Flash takes no mask.
  The pin is the record: sdpa_kernel errors if it can't run, and that's a stop.
- **First-chunk batches are held at 8.** The last one is padded with throwaway rows, whose output is discarded.
  The chunk length adapts so 8 × (the longest prompt + chunk) ≤ B_TOK, at most CHUNK_MAX and at least
  CHUNK_MIN. A batch whose prompts are too long for 8 rows at CHUNK_MIN is sized by B_TOK instead, and recorded
  as a deviation.
- **Continuations, the long tail only,** are sized by B_TOK: rows × (the longest so far + chunk) ≤ B_TOK. They
  re-prefill the row's token ids (the prompt's plus those generated so far), never decoded and re-encoded, and
  the row's generator carries on where it stopped.
- **Each batch records** its rows, size, chunk length, kernel and kind (first, padded or continuation). Each
  row records the first token's log-probability under the unprocessed distribution, for parity's deviation log.
"""

import hashlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
import harness  # noqa: E402

_CARD = harness.DECODING["sampled"]
SAMPLING = {"temperature": _CARD["temperature"], "top_k": _CARD["top_k"], "top_p": _CARD["top_p"]}  # the harness's
MAX_NEW = harness.MAX_NEW_TOKENS
assert _CARD["min_p"] == 0.0, "min-p isn't implemented here; it's 0 on the card"

B_TOK = 28000  # tokens of KV cache per batch: rows x (longest + chunk); about 4.1 GB for Qwen3-8B
CHUNK_MAX = 2048
CHUNK_MIN = 512
BATCH = 8
PATH = "batched-v1"


def seed_for(item_id: str, sample_idx: int) -> int:
    """A 63-bit seed from (item_id, sample_idx), stable across runs and machines."""
    return int.from_bytes(hashlib.sha256(f"{item_id}|{sample_idx}".encode()).digest()[:8], "big") >> 1


def process_logits(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    """Temperature, top-k, top-p, as transformers' warpers do them (min-p 0 does nothing). Rows are independent."""
    scores = logits / temperature
    if top_k and top_k < scores.shape[-1]:
        kth = torch.topk(scores, top_k, dim=-1).values[..., -1, None]
        scores = scores.masked_fill(scores < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scores, descending=False, dim=-1)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        remove = cum <= (1 - top_p)
        remove[..., -1:] = False  # keep at least one token
        scores = scores.masked_fill(remove.scatter(-1, sorted_idx, remove), float("-inf"))
    return scores


def plan(rows: list, b_tok: int = B_TOK, batch: int = BATCH, chunk_max: int = CHUNK_MAX,
         chunk_min: int = CHUNK_MIN, first: bool = True) -> list:
    """Deterministic batches over `rows` ({"key", "ids", "new"}): [(row keys, chunk, kind)]. First chunks: sorted by
    length then key, groups of `batch` (the last padded), the chunk adapted to the budget; a group too long for
    `batch` rows at chunk_min is split by the budget ("long-prompt"). Continuations: grouped by the budget."""
    order = sorted(rows, key=lambda r: (len(r["ids"]), r["key"]))
    out = []
    if first:
        for i in range(0, len(order), batch):
            group = order[i:i + batch]
            longest = len(group[-1]["ids"])
            chunk = min(chunk_max, b_tok // batch - longest)
            if chunk >= chunk_min:
                out.append(([r["key"] for r in group], chunk, "first" if len(group) == batch else "padded"))
            else:
                out += [(keys, c, "long-prompt") for keys, c, _ in plan(group, b_tok, batch, chunk_min, chunk_min,
                                                                         first=False)]
        return out
    group: list = []
    for r in order:
        trial = group + [r]
        longest = max(len(x["ids"]) for x in trial)
        if group and len(trial) * (longest + chunk_max) > b_tok:
            out.append(([x["key"] for x in group], chunk_max, "continuation"))
            group = [r]
        else:
            group = trial
    if group:
        out.append(([x["key"] for x in group], chunk_max, "continuation"))
    return out


def _decode(model, rows: list, chunk: int, pad_id: int, stop_ids: set, max_new: int, sampling: dict) -> None:
    """One batch, one chunk: left-padded prefill of each row's ids, then up to `chunk` sampled steps. Rows are
    updated in place: "new" gains the sampled ids, "done" is set at a stop token or the cap."""
    device = next(model.parameters()).device
    width = max(len(r["ids"]) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), width), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, width - len(r["ids"]):] = torch.tensor(r["ids"])
        mask[i, width - len(r["ids"]):] = 1
    ids, mask = ids.to(device), mask.to(device)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        past, logits = out.past_key_values, out.logits[:, -1, :].float()
        last_pos = pos[:, -1]
        for _ in range(chunk):
            active = [i for i, r in enumerate(rows) if not r["done"]]
            if not active:
                break
            scores = process_logits(logits, sampling["temperature"], sampling["top_k"], sampling["top_p"])
            probs = scores.softmax(-1)
            nxt = torch.full((len(rows),), pad_id, dtype=torch.long, device=device)
            for i in active:
                r = rows[i]
                tok = int(torch.multinomial(probs[i], 1, generator=r["gen"]))
                if r.get("first_logprob") is None:
                    r["first_logprob"] = float(torch.log_softmax(logits[i], -1)[tok])
                nxt[i] = tok
                r["new"].append(tok)
                r["ids"].append(tok)
                if tok in stop_ids or len(r["new"]) >= max_new:
                    r["done"] = True
            step_mask = torch.tensor([1 if i in active else 0 for i in range(len(rows))], device=device)
            mask = torch.cat([mask, step_mask[:, None]], dim=-1)
            last_pos = last_pos + step_mask
            out = model(input_ids=nxt[:, None], attention_mask=mask, position_ids=last_pos[:, None],
                        past_key_values=past, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :].float()


def generate_all(model, work: list, pad_id: int, stop_ids: set, max_new: int, sampling: dict,
                 b_tok: int = B_TOK, batch: int = BATCH, chunk_max: int = CHUNK_MAX, chunk_min: int = CHUNK_MIN,
                 kernel: str = "efficient") -> tuple[dict, list]:
    """Generate for every work item ({"key", "item_id", "sample", "ids"}): returns ({key: record}, batch log).
    A record carries the harness's generate fields over token ids ("new" ids, prompt_tokens, new_tokens, truncated,
    seconds marked shared) plus seed, chunks and first_logprob; the caller decodes the text."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    backend = {"efficient": SDPBackend.EFFICIENT_ATTENTION, "flash": SDPBackend.FLASH_ATTENTION,
               "math": SDPBackend.MATH}[kernel]
    device = next(model.parameters()).device
    rows = {}
    for w in work:
        if w["key"] in rows:
            raise ValueError(f"duplicate work key {w['key']}")
        seed = seed_for(w["item_id"], w["sample"])
        rows[w["key"]] = {"key": w["key"], "ids": list(w["ids"]), "prompt_tokens": len(w["ids"]), "new": [],
                          "done": False, "chunks": 0, "seed": seed, "first_logprob": None, "seconds": 0.0,
                          "gen": torch.Generator(device=device).manual_seed(seed)}
    log = []
    pending, first = list(rows.values()), True
    with sdpa_kernel([backend]):
        while pending:
            for keys, chunk, kind in plan(pending, b_tok, batch, chunk_max, chunk_min, first=first):
                group = [rows[k] for k in keys]
                pad = []
                if kind == "padded":  # held at `batch`: throwaway copies of the last row, output discarded
                    src = group[-1]
                    pad = [{"key": None, "ids": list(src["ids"]), "new": [], "done": False, "first_logprob": None,
                            "gen": torch.Generator(device=device).manual_seed(0)} for _ in range(batch - len(group))]
                t0 = time.time()
                _decode(model, group + pad, min(chunk, max_new - min(len(r["new"]) for r in group)), pad_id,
                        stop_ids, max_new, sampling)
                dt = time.time() - t0
                for r in group:
                    r["chunks"] += 1
                    r["seconds"] += dt
                log.append({"rows": keys, "size": len(group) + len(pad), "chunk": chunk, "kind": kind,
                            "kernel": kernel, "seconds": round(dt, 2)})
            pending, first = [r for r in rows.values() if not r["done"]], False
    out = {}
    for k, r in rows.items():
        ended = bool(r["new"]) and r["new"][-1] in stop_ids
        out[k] = {"new": r["new"], "prompt_tokens": r["prompt_tokens"], "new_tokens": len(r["new"]),
                  "truncated": not ended and len(r["new"]) >= max_new, "seconds": round(r["seconds"], 2),
                  "seconds_shared": True, "seed": r["seed"], "chunks": r["chunks"],
                  "first_logprob": r["first_logprob"], "path": PATH}
    return out, log


def stop_ids(model, tok) -> set:
    """The generation config's stop tokens, plus the two the harness checks (eos and <|im_end|>)."""
    cfg = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    ids = set(cfg if isinstance(cfg, (list, tuple)) else [cfg] if cfg is not None else [])
    return ids | {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")} - {None}


def make_generate_many(model, tok, log: list, batch: int = BATCH, b_tok: int = B_TOK, kernel: str = "efficient"):
    """For quiz.sit's generate_many: chat messages -> the harness's prompt ids (its template and thinking setting)
    -> generate_all -> the harness's generate fields, with the text decoded as the harness decodes it. Each call's
    batch log is appended to `log`."""
    stops = stop_ids(model, tok)
    pad = tok.pad_token_id if tok.pad_token_id is not None else min(stops)

    def generate_many(asks: list) -> dict:
        work = []
        for a in asks:
            prompt = tok.apply_chat_template(a["messages"], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=harness.THINKING)
            work.append({"key": a["key"], "item_id": a["item_id"], "sample": a["sample"],
                         "ids": tok(prompt)["input_ids"]})
        out, batches = generate_all(model, work, pad, stops, MAX_NEW, SAMPLING, b_tok=b_tok, batch=batch,
                                    kernel=kernel)
        log.extend(batches)
        return {k: {**r, "text": tok.decode(r["new"], skip_special_tokens=True)} for k, r in out.items()}
    return generate_many
