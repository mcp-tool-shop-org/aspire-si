# Run report: which seed carries the critic's spread (2026-10-08)

**Plan:** [2026-10-08-critic-init-plan.md](2026-10-08-critic-init-plan.md), with the readout
committed in #43 before anything ran. Run from main at 78cc26f (`pod/plan_g.sh`).

## Result

**Both carry it.** Varying only the critic's initial weights spreads accuracy over 0.488, and
varying only the run's seed spreads it over 0.283. Each is larger on its own than the 0.181 seen
when one seed drove both.

**One initial draw produced an inverted critic.** With the critic seeded 43, the critic preferred the
flawed answer: 0.315, with its whole interval below 0.5. Initialisation can decide which way the
critic learns, not just how well.

## Numbers

32-prompt control-local runs: base student Qwen2.5-1.5B-Instruct, local teacher Qwen2.5-32B-Instruct,
three epochs, batch 1. Pairwise accuracy on the 127 planted-error pairs, with prompt-clustered 95%
intervals. R is the run's seed and C the critic's.

| Run (R:C) | Arm | Accuracy | 95% CI |
|---|---|---|---|
| 42:42 | A and B | 0.803 | [0.730, 0.873] |
| 42:43 | A | **0.315** | [0.227, 0.406] |
| 42:44 | A | 0.787 | [0.709, 0.856] |
| 43:42 | B | 0.543 | [0.449, 0.641] |
| 44:42 | B | 0.827 | [0.750, 0.891] |
| 42:42:b (identical repeat) | noise floor | 0.858 | [0.786, 0.921] |

## Reading under the committed rule

1. **Noise floor:** |42:42 − 42:42:b| = 0.055, at or under 0.09, so the arms are read.
2. **Arm A** (critic's initial weights vary, R = 42): range **0.488**.
3. **Arm B** (run seed varies, C = 42): range **0.283**.
4. Both are above 0.09: **both contribute**. Both are at or above 0.135, so each alone carries at
   least most of the 0.181.

The run's seed bundles the student adapter's initialisation, the data order and the sampling. This
test doesn't separate those.

## Also reported

- **Generation is not reproducible here.** None of the 32 cached dialogues is identical between any
  two runs compared, including the two identical-seed runs.
  - With a 4-bit teacher and student sampling on the GPU, fixing the run's seed fixes its random
    streams but not what is generated.
  - So arm A varied the critic's initial weights plus that generation noise.
  - The one noise-floor pair suggests the noise alone moves accuracy by about 0.05, but it is one
    pair.
- **Reproducibility check:** 42:42 gave 0.803, and its repeat 0.858, against run 1's seed-42
  control at 0.724. `fork_rng` changes the order of draws, so no exact match was expected. Even so,
  identical configurations land 0.05 apart.
- **The seed split works on the hardware.** All seven tests of `critic.init_seed`, including the
  GPU-stream test, passed on the pod before training.

## What it means

- **No single critic is representative of its condition.** Its initial weights alone can move its
  accuracy from 0.32 to 0.80.
- **Comparing conditions by one critic each, or three, cannot be read at this spread.** Before the
  next comparison, the critic setup has to change. These are candidates for the maintainer to
  choose from, not readings:
  - average several critic heads with different initial weights over one run (an ensemble);
  - train with many critic seeds and report the distribution;
  - a different critic design.
- **Kev-4B, read order-averaged (0.976 on these pairs),** stays the fixed reference next to any
  critic.

## Cost

| Plan | Spent |
|---|---|
| 20: six runs in two batches of three, about 1.43 min per dialogue | $3.85 |

Timeline (UTC): setup 07:37–07:47; batch 1 07:47–08:34; batch 2 08:34–09:20; probes and judge
done 09:23; pod shut down 09:27 after the pull was checked (file counts and bytes).

Data: `aspire-si-runs/2026-10-08-critic-init/plan20/` (outside the repository).
