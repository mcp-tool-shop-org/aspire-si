# Run report: judges read order-averaged, confirmed on fresh pairs (2026-10-08)

**Plan:** [2026-10-08-kev-confirmation-plan.md](2026-10-08-kev-confirmation-plan.md), merged in
#36 before any fresh pair was judged. Judged from commit c94fc93. Local GPU, $0.

## Result

**Confirmed.** Kev-4B, read order-averaged, favours the strong answer on 145 of 149 fresh pairs:
0.973, prompt-clustered 95% CI [0.946, 0.993]. Both conditions of the rule hold (at least 0.85,
lower bound at least 0.75). Kev-4B is the reference judge for later comparisons, and it is read
order-averaged only.

**The finding is about the reading, not about Kev.** The 32B teacher also clears both conditions.
Every judge here looks poor and position-biased when its single choices are read, and every one
clears the bar order-averaged.

## Numbers

Fresh pairs: 149 pairs from 78 of the 80 confirmation prompts, planted by Qwen2.5-32B-Instruct
Q4_K_M.

| Judge | Order-averaged | 95% CI | Rule | Single choices | Option A chosen |
|---|---|---|---|---|---|
| Kev-4B | 0.973 (145/149) | [0.946, 0.993] | confirmed | 0.547 | 95% |
| Kev-9B | 0.859 (128/149) | [0.795, 0.921] | meets both (reported only) | 0.638 | 85% |
| Qwen2.5-32B Q4 | 0.886 (132/149) | [0.823, 0.939] | meets both | 0.685 | 82% |

The 127-pair judge set, planted by the same model at bf16:

| Judge | Order-averaged | 95% CI | Single choices | Option A chosen |
|---|---|---|---|---|
| Kev-4B (step 1, from its recorded probabilities) | 0.976 (124/127) | [0.945, 1.0] | 0.547 | 95% |
| Kev-9B (step 1) | 0.803 (102/127) | [0.724, 0.874] | 0.598 | 87% |
| Qwen2.5-32B Q4 | 0.898 (114/127) | [0.835, 0.952] | 0.701 | 80% |

How the judges separate the two orders:

| Judge (fresh pairs) | Mean gap | Spread of gap | Pairs right in both orders without averaging |
|---|---|---|---|
| Kev-4B | 0.131 | 0.149 | 14 |
| Kev-9B | 0.192 | 0.215 | 41 |
| Qwen2.5-32B Q4 | 0.372 | 0.451 | 55 |

Here the gap is p(A | strong answer first) − p(A | strong answer second).

## Reading

- **The bias sits on top of a steady preference.** Kev-4B chooses option A almost every time,
  but it leans further toward A when A is the strong answer, on nearly every pair. A single choice
  hides that; the difference between the two orders shows it.
  - Kev-4B's lean is small (a mean gap of 0.13) but consistent.
  - Kev-9B and the 32B lean harder but less consistently, so more of their pairs land on the wrong
    side.
- **The 4B beats the 9B again** on fresh pairs, as in step 1. The 9B is not less biased (it chooses
  A 85% of the time against 95%); its preference is noisier.
- **No self-preference.** The 32B at Q4 planted the fresh pairs and judges with the same weights.
  It scores the same on the judge set, planted at bf16 (0.898 against 0.886).
- **Saturation.** The 32B's letter probabilities often saturate. On 36 of the 149 fresh pairs (29
  of the 127 judge pairs), both orders gave p(A) above 0.999, so those pairs are decided by
  probabilities below 0.001. Kev's probabilities do not saturate this way.
- **The pre-registered step 1 reading stands as written.** By single choices, Kev is not a useful
  judge. What changed is which reading to use: order-averaged, which this run tested on pairs fixed
  in advance.

## What it changes

- **Later comparisons** (the follow-up fine-tune at 128 prompts, #11) can report Kev-4B,
  order-averaged, as a fixed reference judge next to the critics. Two calls per pair, a few
  minutes per run, on the local GPU.
- **Reading any judge:** score both orders and average the probabilities. Never read one hard
  choice. That includes the teacher's own comparisons (`pairwise_teacher.py` reads only text
  choices).
- **The R&D session's Kev fine-tune** trains on the 603 pairs from the 320 training prompts.
  Order-averaged accuracy is already near the ceiling, so its useful target is the single-choice
  reading: an unbiased single call.

## Method

- **Fresh pairs** (`fresh_pairs.py`):
  - The 400 training prompts' strong teacher answers from 2026-10-07; none is a held-out prompt.
  - One JSON sentence edit per flaw kind, retried at rising temperature, kept only if it passes
    the judge set's filters.
  - 752 of 800 slots were planted. Failed attempts: 305 "original not in the answer" and 12 "no
    JSON edit", most recovered on retry.
  - Split by topic: 80 confirmation prompts (149 pairs) and 320 training prompts (603 pairs).
- **Planting model:** Qwen2.5-32B-Instruct Q4_K_M, the official GGUF, served by llama.cpp (build
  11433) with 4 slots and a 16k context.
  - The first launch memory-mapped the weights and held about 30 GB of host RAM, at the VRAM
    watchdog's 90% RAM line.
  - It was then terminated from outside, with no error in its log; the cause is not known.
  - It was relaunched with `--load-mode none`, at 25.4 GB VRAM and 37% RAM.
- **Judges:**
  - Kev through `kev.serve` with GPU memory capped at 82% (`judge_kev.py`).
  - The 32B through llama-server: the `pairwise_teacher.py` question, one token, p(A) and p(B)
    from the top 20 log-probabilities (`judge_logprob.py`).
  - One model on the GPU at a time.
- **Statistic** (`judge_kev.order_averaged`): mean p(strong) over both orders, favoured above 0.5,
  ties counting half, with a bootstrap that resamples whole prompts.

Data: `aspire-si-runs/2026-10-08-kev-confirm/` (fresh pairs, split, per-pair results), outside the
repository.
