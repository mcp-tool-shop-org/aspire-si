# Run report: which seed carries the critic's spread (2026-10-08)

**Plan:** [2026-10-08-critic-init-plan.md](2026-10-08-critic-init-plan.md), with the readout
committed in #43 before anything ran. Run from main at 78cc26f (`pod/plan_g.sh`).

## Result

**The headline: a critic's initial weights can invert it.** With the critic seeded 43 and nothing
else changed, the critic preferred the flawed answer: 0.315, with its whole interval below 0.5.
That is a different failure from the weak critics seen so far, which were above chance but not by
much. Initialisation can decide which way the critic learns, not just how well.

**Under the committed rule, both seeds carry the spread.** Varying only the critic's initial
weights gives a range of 0.488, and varying only the run's seed 0.283. Read these two separately
from the headline, and with their limits:
- **Arm A's range is one draw.** The 0.315 run makes the whole 0.488; the other two critic seeds
  differ by 0.016 (0.803 and 0.787). So this shows that an initial draw *can* invert the critic,
  not that initialisation spreads accuracy broadly.
- **Arm B's 0.283 includes generation noise.** Generation isn't reproducible here (see below), so
  every arm carries it.
- **The noise floor rests on one pair** (0.055).

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

## Could earlier single critics have been inverted draws?

Of the 15 critics trained before this test (run 1's 12 and step 2's 3), one has a point estimate
below 0.5: run 1's seed-44 composite control, 0.425 [0.333, 0.516]. Its interval reaches just above
0.5, so it is not clearly inverted. It is the one earlier result that could have been an inverted
draw. All the others have intervals above 0.5 or touching it from above.

## What it means

- **No single critic is representative of its condition.** Its initial weights alone moved its
  accuracy from 0.32 to 0.80 here.
- **Comparing conditions by one critic each, or three, cannot be read at this spread.** Before the
  next comparison, the critic setup has to change. These are candidates for the maintainer to
  choose from, not readings:
  - train several critic heads with different initial weights and select or average them on a
    validation set of planted pairs held apart from the 127 judge pairs (selecting on the judge set
    would leak into the evaluation). At the least, reject any critic below 0.5 on validation. The
    fresh pairs from the Kev confirmation could supply that set.
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
