# DRAFT: Batched generation for the quiz and apply sittings (run sheet B4)

**Status: built (2026-10-15) to R&D's settlement after Argon's review; this note describes the code as built.**
The parity run in §5 needs a card grant (about 2 h), asked for separately.

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
  below, with `batched.py`'s sha256 and every batch, are recorded in each quiz sitting's readout. `quiz.py` has
  no batch-1 path for a sitting: its per-call generate refuses.

## 3. How it generates (as built: `batched.py`)

- **Its own decode loop, on token ids.** The prompt ids come from the harness's chat template and thinking
  setting. The loop prefills, then samples step by step with the KV cache. Text is decoded once, at the end, as
  the harness decodes it.
- **Per-item generators (Argon; replaces ruling A).** Each row samples from its own `torch.Generator`, seeded by
  (item_id, sample_idx): a 63-bit seed from sha256. The quiz's item_id is `<base>-v<n>`, and the samples are
  0, 1 and 2. So a row's draws don't depend on its batch-mates or the batch size; only the numerics of a padded
  batch can move it. On CPU in float32, a test shows a row's tokens are identical alone and in a batch of 8.
- **Sampling:** the harness's card settings, imported from `harness.DECODING`, not copied: temperature, then
  top-k, then top-p, in transformers' order (min-p is 0 and asserted). The stop tokens are the generation
  config's plus the two the harness checks. The cap is `harness.MAX_NEW_TOKENS` over the whole answer.
- **Left padding with an explicit attention mask.** Position ids come from the mask, and a finished row is masked
  out of later steps.
- **SDPA pinned to EFFICIENT at every batch size**, parity (i)'s batch-1 arm included, so (i) isolates batching
  alone. Flash takes no mask. An `sdpa_kernel` error is a stop, not a fallback.
- **The KV budget.** Qwen3-8B's cache is about 0.147 MB per token (36 layers × 8 KV heads × 128 × K and V ×
  bf16). B_TOK is 28,000 tokens per batch, about 4.1 GB, so with the weights, the LoRA and the idle baseline the
  plan is about 24–25 GB of memory.used. Parity measures the peak, and ≤ 26 GB is the gate.
  - **First chunks are held at 8 (Argon).** Batches are sorted by prompt length, then key. The last batch is
    padded with throwaway rows, whose output is discarded. The chunk length adapts so 8 × (the longest prompt +
    chunk) ≤ B_TOK, at most 2,048 and at least 512.
  - **The one stated deviation:** a group whose prompts are too long for 8 rows at 512 is split by the budget and
    logged as `long-prompt`. Continuations, the long tail only, are sized by the budget too: rows × (the longest
    so far + chunk) ≤ B_TOK. They can't hold 8 long rows.
  - **Continuations carry token ids** (R&D's required change). A row re-prefills its prompt ids plus the ids it
    has generated, never decoded and re-encoded, and its generator carries on where it stopped. A test shows a
    run chunked at 10 tokens yields exactly the tokens of an unchunked run.
- **Per-row records** carry the same fields as `harness.generate`'s: text, prompt_tokens, new_tokens, truncated
  and seconds, the last marked as shared across the batch. They also carry the seed, the number of chunks, the
  first token's log-probability (unprocessed, for parity's deviation log) and the path, `batched-v1`.
- **Each batch is logged** with its rows, size, chunk, kind (first, padded, long-prompt or continuation) and
  kernel. The quiz readout records the path, the batch size, the budget, `batched.py`'s sha256 and every batch.
- **The quiz** (`quiz.sit`) chooses every ask first, generates all of them in one batched call, then scores and
  records in the same order. A test shows the batched call asks the same questions and writes the same rows as
  the per-call path.

## 4. What changes, and what doesn't

| | Batch 1 (harness) | Batched (as built) |
|---|---|---|
| Model, adapter, template, thinking | as pinned | the same, imported |
| Card settings, 16,384 cap | harness DECODING | the same constants, imported |
| Samples per ask, majority of 3 | seeds 0–2 | 3 draws per ask, each from its own (item, sample) generator |
| Reproducible | per generation | per row, independent of the batch |
| Kernel | as the wrapper pins it (flash) | EFFICIENT, pinned |
| Numerics | bf16, unpadded | bf16 with left padding: small logit differences, so parity checks distributions |
| Stuck fields and rule | from the text and token counts | the same inputs, the same rule (716dc251) |

## 4a. Do batched readings stay comparable with the frozen stuck thresholds? (Argon's Q3)

The stuck rule (`716dc251`, `stuck.py` `8097b039`) was frozen from **batch-1** traces of the untrained model:
the tier p95s of thinking tokens, the re-read baseline of 8.09 per 1,000, and the burst parameters. Steering
compares batched quiz and apply readings against those thresholds. They stay comparable on three conditions.

1. **The rule reads only what the batched path keeps the same.** It reads the finished text (spans, quotes,
   re-reads, hedges), the thinking-token count and whether the 16,384 cap was reached. It never reads timing or
   batch shape. The batched path keeps the cap and the total-length rule unchanged. A continuation carries the row's own token ids and generator, so
   its text joins exactly. Thinking tokens are counted by re-tokenizing the final thinking text with the student's tokenizer,
   as quiz.py counts batch-1 traces. The total the cap clause reads (`total_tokens`) is the sum of the tokens each
   chunk generated. So a long answer split over chunks is measured as one answer.
2. **The traces themselves must be alike, so that's measured, not assumed.** Padding and batched bf16 matmuls move
   the logits slightly, and a chunk's re-prefill recomputes the cache. Either could shift how long or how
   loopy the thinking runs, which is exactly what the thresholds sit on. So §5's parity run compares the rule's own
   inputs and each clause, not only the stuck total.
3. **The thresholds are never re-frozen on batched traces.** If parity fails on the stuck inputs, B4 stops and R&D
   decides. A threshold isn't moved to fit the batched path. That keeps the rule one rule from the baseline on.

**Within Stage 1, every quiz and apply reading is on the batched path from round 1** (§2). So round-to-round
changes in a cell's stuck rate are path-consistent. Only the absolute level against the frozen thresholds depends
on parity, which is what §5 checks.

## 5. Parity check, before round 1 (pre-registered; `parity.py`)

- **The set:** 2 kind-E verdict variants (n = 0) per lesson, the lowest ids that aren't spent: 32 variants over
  the 16 lessons of batches 1–6 (`parity.select`; earlier drafts said 30, but 2 × 16 is 32). The C/D apply
  prompts join when B6 lands, before round 1. It runs on the base model, with no adapter: parity is a property of
  the path.
- **Four draws, 3 samples per variant:** LB1, the batched loop at batch 1; LB8, the batched loop at batch 8; P1,
  harness.generate at seeds 0–2; and P1′, harness.generate at seeds 3–5. P1 and P1′ run with flash pinned and the
  deterministic settings, as every batch-1 sitting does.
- **(i) LB1 against LB8, paired by (item, sample): batching alone.** Argon's tolerances:
  - aggregate stuck rate within ±0.03;
  - item-level stuck agreement ≥ 92%, or κ ≥ 0.85;
  - p90 thinking tokens within ±10%, and the median within ±5%;
  - the first-token log-probability deviation is logged (median and max), not gated.
- **(ii) LB1 against the harness, with P1 against P1′ as the noise reference.** #118's relative rules:
  - cross majority agreement ≥ within − 0.10;
  - median thinking tokens within ±15% of P1's;
  - stuck rate within ±0.10 of P1's;
  - truncations ≤ P1's + 1;
  - each stuck clause's rate within |P1 − P1′| + 0.05;
  - the median thinking tokens per tier within ±15% of P1's.

  Absolute tolerances don't hold across RNG streams: at n = 90, the SE of a stuck-rate difference is about 0.05.
- **A fail stops B4 until R&D decides.** It never moves a threshold.
- **Also recorded:** torch's peak reserved memory (the run sheet adds the CUDA context and the idle baseline, and
  memory.used must be ≤ 26 GB) and each draw's time.
- **Cost:** LB1 is 96 generations at batch 1 (about 32 min), LB8 about 8 min, and P1 and P1′ 192 generations
  (about 64 min). With loading and preflight, about **2 h** on the card, at a peak of about 22–25 GB.

## 6. Tests (CPU, no card)

- **`test_batched.py`**, on a tiny random Qwen3 in float32 with the math kernel (CPU has no efficient kernel; the
  rig pins efficient):
  - a row's tokens are identical alone and in a batch of 8;
  - a run chunked at 10 tokens equals an unchunked run, so continuations carry ids and the generator;
  - the stop token and the cap follow the harness's rules;
  - the seed is stable per (item, sample);
  - the plan holds first batches at 8, keeps every batch in the budget, splits long prompts, and is
    deterministic;
  - the sampling settings are the harness's;
  - generate_many returns the harness's fields.
- **`test_quiz.py`:** the batched call asks the same questions and writes the same rows as the per-call path.
- **`test_parity.py`:** the set is fixed by rule, and (i) and (ii) pass and fail where their rules say.

## 7. Rulings (R&D, 2026-10-15, with Argon's review)

- **Seeds:** per-item `torch.Generator` seeded by (item_id, sample_idx) (Argon). This replaces ruling (A).
- **Batch shape:** first chunks held at 8, the last padded. The chunk adapts within 512–2,048 so 8 × (the longest
  prompt + chunk) ≤ B_TOK 28,000. Continuations are sized by the budget (the stated deviation).
- **Kernel:** EFFICIENT pinned at every batch size. Flash in the batch-1 wrapper.
- **Continuations:** token ids, with the row's generator carried on.
- **Parity:** (i) and (ii) as in §5. Gate: the measured peak ≤ 26 GB. The E variants now, and C/D when B6 lands,
  both before round 1.

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | The path, budget, batch log and batched.py's sha256 are recorded in each quiz readout, and the seed per row. The settings are imported from the pinned harness. |
| ANDON_AUTHORITY | 2 | A parity fail stops B4. The 28 GB self-stop and the watchdog still apply. |
| NAMED_COMPENSATORS | 2 | The parity run uses a card grant, undone by stopping it and sending "card free". The code is a PR, closed unmerged to undo. |
| DECOMPOSE_BY_SECRETS | 2 | The batched path is its own module. The harness and every batch-1 sitting are untouched. |
| UNCERTAINTY_GATED_HUMANS | 2 | The seed method, the budget and the parity thresholds are R&D's calls before anything runs. |
| EXTERNAL_VERIFIER | n/a | No specialized claims. |
