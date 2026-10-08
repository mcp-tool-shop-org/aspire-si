# Plan: confirm the order-averaged judge reading on fresh pairs (written 2026-10-08, before it runs)

## Why

Step 1 of the [next-step plan](2026-10-08-next-step-plan.md) read Kev by its single choices, and
under that rule neither Kev model was a useful judge. Kev chose option A on 95% (4B) and 87% (9B)
of questions.

Step 1 also recorded both orders' probabilities. Averaging p(strong) over the two orders changes
the picture:

| Judge | Single choices (the committed rule) | Order-averaged |
|---|---|---|
| Kev-4B | 0.547 | 0.976 (124 of 127), prompt-clustered CI [0.945, 1.0] |
| Kev-9B | 0.598 | 0.803 (102 of 127), CI [0.724, 0.874] |

That statistic was chosen after seeing the data, so it is exploratory. This plan tests it on pairs
none of these judges has been read on, with the statistic and the bar fixed here first.

## The statistic (fixed)

For each pair, the judge is asked in both orders. In each order, p(strong) is the probability of
the option holding the strong answer, normalised over the two options. The order-averaged
p(strong) is the mean of the two.

- A pair is **favoured** when the order-averaged p(strong) is above 0.5. Exactly 0.5 counts half.
- **Order-averaged accuracy** is the share of favoured pairs.
- The 95% interval is a bootstrap that resamples whole prompts.

Also reported for each judge, with no rule attached:
- single-choice accuracy, and how often option A is chosen;
- mean |p(A | strong first) − p(A | strong second)|, its spread, and how many pairs are decided
  for the strong answer in both orders without averaging.

## Fresh pairs

- The 400 training prompts each have a strong teacher answer from 2026-10-07 (Qwen2.5-32B-Instruct,
  bf16). None of them is one of the 64 held-out prompts behind the judge set.
- Errors are planted the way the judge set's were: one JSON sentence edit per flaw kind, applied by
  code and retried at rising temperature. The pairs pass the same filters (similarity at least
  0.90, length change at most 5%, no self-flagging, no truncation).
- The planting model is **Qwen2.5-32B-Instruct at Q4_K_M** (the official GGUF), served by
  llama.cpp on the local GPU. The judge set was planted by the same model at bf16. Every
  result says which planted it.
- The prompts are split by topic into **80 confirmation prompts** and **320 Kev-training prompts**
  for the R&D session's fine-tune. No prompt is in both. The 127 judge pairs are in neither.

## Judges

| Judge | How it is read |
|---|---|
| Kev-4B, Kev-9B | `kev.serve` capped at 82% of GPU memory, two options, both orders |
| Qwen2.5-32B-Instruct Q4_K_M | llama-server; the A/B question of `pairwise_teacher.py`, one token, its top log-probabilities |

Qwen-32B at Q4 judges with the same weights that planted the confirmation pairs, which may favour
its own edits, or its own originals. One line in the report says so. It is also read on the 127
judge pairs (bf16-planted).

The teacher's letter probabilities saturate: on a format check with one old judge pair, p(B) was
about 2e-8. Where both orders give p(A) near 1, the order-averaged decision rests on very small
probabilities. The report counts the pairs where both orders' p(A) is above 0.999.

## Decision rules

| Result on the 80-prompt confirmation set | Reading |
|---|---|
| Kev-4B order-averaged accuracy at least 0.85, and the lower end of its 95% interval at least 0.75 | Confirmed: Kev-4B, read order-averaged, is the reference judge for later comparisons. It is never read by single choices. |
| Either condition missed | Not confirmed: step 1's 0.976 does not carry over to fresh pairs, and Kev stays unused as a judge |
| Qwen-32B meets both conditions as well | The finding is about reading judges order-averaged, not about Kev. Reported as such. |

Kev-9B gets the same two conditions, reported and with no decision attached.

With 80 prompts, a point estimate alone could clear 0.85 by luck, so the interval's lower end must
also clear 0.75, step 1's bar. This condition was added at review, before any fresh pair was
judged.

## Cost and order

- $0: local GPU. The maintainer approved it on 2026-10-08. The watchdog runs, and one job uses the GPU at a time.
- Planting: about 800 edit requests to the Q4 32B, an estimated 1.5–2 hours.
- Judging: Kev at a few minutes per model; the 32B at one token per request.
- Then the GPU goes to the R&D session for its Kev fine-tune, which trains on the 320-prompt pairs
  only.
