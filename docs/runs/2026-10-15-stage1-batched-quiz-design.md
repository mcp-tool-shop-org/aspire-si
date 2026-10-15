# DRAFT: Batched generation for the quiz and apply sittings (run sheet B4)

**Status: design for R&D's review (2026-10-15).** Nothing is built or run until R&D passes it. The parity run in
§5 needs a short card grant, asked for separately after this passes.

**What it builds.** The approved training procedure says apply and quiz "generate in batches of 8", while DEV
and the sealed set keep the pinned batch-1 harness. That batching was never built, so today the quiz runs one
sample at a time: about 1 h 45 min per seed and arm, and about 5 h 15 min of a round's card time across both arms.
R&D ruled (#115, Q2) to build it before round 1, with this note first.

## 1. The check in three sentences

1. **The question:** can the quiz and apply sittings generate in batches without changing what they measure:
   the majority-of-3 outcome per ask, and the stuck readings that steer the next round?
2. **What each path gets:** the batch-1 path is `harness.generate`, one prompt at a time, re-seeded each time.
   The batched path is the same model, chat template and card settings (T 0.6, top-p 0.95, top-k 20, min-p 0,
   thinking on, 16,384 new tokens at most), with up to 8 prompts per forward pass.
3. **Why a parity check answers it:** sampling makes byte equality impossible across the two paths, and the
   batch-1 path doesn't reproduce itself across seeds either. So the check is whether the batched path's answers
   are as alike to batch 1's as batch 1's are to another batch-1 draw (§5).

## 2. Scope (fixed)

- **Batched:** `quiz.py`'s sitting and the apply sitting (B6), and nothing else.
- **Untouched:** `harness.py` (`4fa52448`), eval_adapter, interview.py and interview_adapter. DEV, sealed and the
  interview stay at batch 1, so their numbers stay comparable with the baseline.
- **The new code** lives in `docs/runs/stage1-train/batched.py`. It calls the harness's own pieces where they
  exist (`DECODING`, `MAX_NEW_TOKENS`, `THINKING`, `split_thinking`), so the settings can't drift from the
  harness's.
- **Fixed from round 1:** every round of Stage 1 uses the same quiz path (R&D). The path, batch rules and budget
  below are recorded in each sitting's config, and a resume under a different path is refused.

## 3. How it generates

- **Padding:** left padding with the tokenizer's pad token and an attention mask, as decoder-only generation
  needs. Position ids come from the mask (transformers' generate does this). Prompts are bucketed by length,
  so padding stays small.
- **One sitting's work** is a list of (ask, sample seed) pairs: each ask's 3 samples (seeds 0, 1, 2), and each
  apply item's 1 sample. They're grouped into batches in a fixed order: sorted by prompt length, then by id and
  seed. The same sitting always forms the same batches.
- **Seeds under batching (the one real change; see §4):** batch 1 calls `torch.manual_seed(seed)` before each
  generation. In a batch, one RNG draws for every row, so a row's sample depends on its batch-mates.
  - **Proposed (A):** seed each batch with `torch.manual_seed(base + batch_index)`, where `base` is fixed per
    sitting (round, arm, training seed and session, hashed). Each row records its batch index and the batch's
    seed. A sitting is reproducible as a whole (same work, same batches), and the 3 samples of one ask stay
    independent draws.
  - **Alternative (B):** a per-row `torch.Generator` in a hand-written decode loop, so each row's sample depends
    only on its own seed. That reproduces per row, but replaces transformers' generate with our own sampling
    code, a larger surface to get wrong. Recommended only if R&D needs per-row reproducibility.
- **The KV budget (sizing).** Qwen3-8B's cache is about 0.147 MB per token (36 layers × 8 KV heads × 128 × K
  and V × bf16). Eight rows at the full 18k tokens (prompt plus the 16,384 cap) would be about 21 GB of cache.
  That's over the 26 GB plan, because the batch's cache grows to its longest row. So:
  - **Chunked:** the batch generates at most `chunk` new tokens (proposed 2,048) per row. Rows that ended are
    done. Rows still going are re-queued as continuations: the prompt plus the text so far, re-prefilled, and
    generated on toward the 16,384 total. Continuations are batched with each other under the same budget, so
    the long tail ends at batch 1 or 2.
  - **Budget:** a batch is at most 8 rows and at most `B_tok` tokens of cache. That's rows × (longest prompt +
    chunk) ≤ `B_tok`, proposed 28,000 tokens (about 4.1 GB). With 16.4 GB of weights, the LoRA, activations
    and the about 2.2 GB idle baseline, the plan is about 24–25 GB of memory.used. §5's preflight measures
    it.
  - **Why this is safe for the measures:** the baseline's median answer was about 325 new tokens, so almost
    every row ends inside the first chunk. A continuation changes only the RNG stream, not the settings. The
    cap (16,384 total new tokens), the truncation flag and the thinking text the stuck rule reads are the same
    as batch 1's.
- **Per-row records** are the same fields as `harness.generate`'s: text, prompt_tokens, new_tokens, truncated
  and seconds, plus batch, batch_seed, chunks and path ("batched-v1"). `seconds` is the batch's wall time
  shared across its rows, marked as shared. The quiz's time figures are descriptive, and the eval plan's
  per-task time measure is DEV's and sealed's, which stay at batch 1.

## 4. What changes, and what doesn't

| | Batch 1 (today) | Batched (proposed) |
|---|---|---|
| Model, adapter, template, thinking | as pinned | the same |
| Card settings, 16,384 cap | harness DECODING | the same constants, imported |
| Samples per ask, majority of 3 | seeds 0–2 | 3 draws per ask, seeded per batch (A) |
| Reproducible | per generation | per sitting (A); per row (B) |
| Numerics | bf16, unpadded | bf16 with left padding: small logit differences, so a distribution-level check |
| Stuck fields and rule | from the text and token counts | the same inputs, the same rule (716dc251) |

## 4a. Do batched readings stay comparable with the frozen stuck thresholds? (Argon's Q3)

The stuck rule (`716dc251`, `stuck.py` `8097b039`) was frozen from **batch-1** traces of the untrained model:
the tier p95s of thinking tokens, the re-read baseline of 8.09 per 1,000, and the burst parameters. Steering
compares batched quiz and apply readings against those thresholds. They stay comparable on three conditions.

1. **The rule reads only what the batched path keeps the same.** It reads the finished text (spans, quotes,
   re-reads, hedges), the thinking-token count and whether the 16,384 cap was reached. It never reads timing or
   batch shape. The batched path keeps the cap and the total-length rule unchanged. A continuation joins the text
   exactly, and its thinking tokens are counted by re-tokenizing the final text, the way batch-1 traces are counted.
   So a long answer split over chunks is measured as one answer.
2. **The traces themselves must be alike, so that's measured, not assumed.** Padding and batched bf16 matmuls move
   the logits slightly, and a chunk's re-prefill changes the RNG stream. Either could shift how long or how
   loopy the thinking runs, which is exactly what the thresholds sit on. So §5's parity run compares the rule's own
   inputs and each clause, not only the stuck total.
3. **The thresholds are never re-frozen on batched traces.** If parity fails on the stuck inputs, B4 stops and R&D
   decides. A threshold isn't moved to fit the batched path. That keeps the rule one rule from the baseline on.

**Within Stage 1, every quiz and apply reading is on the batched path from round 1** (§2). So round-to-round
changes in a cell's stuck rate are path-consistent. Only the absolute level against the frozen thresholds depends
on parity, which is what §5 checks.

## 5. Parity check, before round 1 (pre-registered)

- **The set:** 30 kind-E verdict variants (n = 0) from the frozen data, fixed by id before the run, 2 per lesson
  where possible, plus 20 apply-style C/D prompts once B6 exists. The base model, no adapter: parity is a
  property of the path, and the base model needs no training run.
- **Three draws:** batch 1 at seeds 0–2 (P1), batch 1 at seeds 3–5 (P1′), and batched (PB). That's 90
  generations each.
- **Measures:**
  - the majority outcome agreement per ask between P1 and PB (cross), against between P1 and P1′ (within);
  - verdict-agreement per sample, cross against within;
  - median thinking tokens;
  - the stuck rate by the frozen rule;
  - the truncation count.
- **Pass (written now):**
  - cross majority-agreement ≥ within − 0.10;
  - PB's median thinking tokens within ±15% of P1's;
  - PB's stuck rate within ±0.10 of P1's;
  - PB's truncations no more than P1's + 1.
  - **The stuck rule's inputs and clauses (4a):** PB's rate for each clause (A: repeated span; B: the cap; C:
    long and not progressing), its re-read rate median, its burst share and its median thinking tokens per tier
    each sit within the P1–P1′ difference plus 0.05 (rates) or 15% (tokens) of P1's.

  A fail stops B4 until R&D decides. It never moves the thresholds.
- **Also measured:** the peak memory.used at `B_tok` (it must be ≤ 26 GB), and the batched speed-up over P1.
- **Cost:** P1 and P1′ are 180 batch-1 generations, about 60 min. PB is about 10–15 min. With loading and
  preflight, about **1 h 30 min** on the card, at a peak of about 22–25 GB.

## 6. Tests (CPU, before any card time)

- The batching plan is deterministic: the same work list always makes the same batches, and no batch exceeds
  8 rows or `B_tok`.
- Continuations: a row that hits the chunk is re-queued with its text so far, and the total stays under 16,384.
  Tested with a fake model that "ends" at set lengths.
- Records carry every harness field, plus batch, seed and path. A resume under a different path or budget is
  refused.
- The quiz's majority-of-3 and stuck fields read batched records exactly as they read batch-1 records.
- Settings are imported from the harness, not copied: a test asserts identity with `harness.DECODING`.

## 7. Open for R&D

- **(A) or (B)** for seeds.
- **`chunk` 2,048 and `B_tok` 28,000,** or other values. They're sized for ≤ 26 GB, and the parity run measures
  the peak.
- **The parity set's composition**, and whether apply prompts wait for B6 (proposed: run E now, add C/D when B6
  lands, before round 1).

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | The path, chunk, budget and seeds are recorded per sitting, and a resume under different ones is refused. The settings are imported from the pinned harness. |
| ANDON_AUTHORITY | 2 | A parity fail stops B4. The 28 GB self-stop and the watchdog still apply. |
| NAMED_COMPENSATORS | 2 | The parity run uses a card grant, undone by stopping it and sending "card free". The code is a PR, closed unmerged to undo. |
| DECOMPOSE_BY_SECRETS | 2 | The batched path is its own module. The harness and every batch-1 sitting are untouched. |
| UNCERTAINTY_GATED_HUMANS | 2 | The seed method, the budget and the parity thresholds are R&D's calls before anything runs. |
| EXTERNAL_VERIFIER | n/a | No specialized claims. |
