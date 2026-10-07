# Run report: fine-tune, then ASPIRE (2026-10-07)

This experiment asked whether ASPIRE does better on a student that was first fine-tuned on the
same material ([#11](https://github.com/mcp-tool-shop-org/aspire-si/issues/11)).

- **Treatment:** fine-tune the student on teacher-written answers in the same 12 topic areas as
  the control's 32 prompts, then run the control's ASPIRE config and seed from that student.
- **Control:** the 2026-10-06 runs ([report](2026-10-06-pod-run.md)), which ran ASPIRE on the
  base student.

## Setup

| | |
|---|---|
| Hardware | 1 × RTX PRO 6000 (96 GB), offrig `job` profile, $2.09/hr; $7.00 for the experiment's four pod sessions |
| Student | `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit, LoRA r16 on q/k/v/o |
| Teacher (data and scoring) | `Qwen/Qwen2.5-32B-Instruct` (Apache-2.0), on vLLM |
| ASPIRE conditions | local teacher (Qwen) and composite teacher (Qwen + `google/gemma-4-31B-it`, voting), as in the control |
| Code | `examples/sft-experiment/`, branch `sft-then-aspire` |

### The dataset

The teacher wrote 540 questions; 490 were kept after removing near-duplicates and anything close
to an evaluation prompt (by content words and by bge-small-en-v1.5 embeddings). The teacher never
saw the evaluation prompts.

| Split | Size | What it holds |
|---|---|---|
| Training | 791 examples from 400 questions | 400 answers and 391 challenge-and-revision exchanges, each scored 8 or 9 by the teacher |
| Held out | 64 questions | Never trained on |
| Judge set | 127 pairs on the 64 held-out questions | A strong answer and the same answer with one planted error |

### The judge set

Each judge pair is a strong answer and the same answer with one planted error. Each held-out
question has up to two pairs: one with a wrong fact or number, one with a reasoning step that does
not follow.

- **How each error was planted:** the teacher names one sentence and gives its edited form, and
  the code makes the swap, so the two sides differ in that sentence only.
- **Every pair passed the same filters:**
  - character similarity of at least 0.90 (the median is 0.999);
  - lengths within 5%;
  - no side cut off;
  - no wording that points at the error.
- **The label is the planted error:** the strong answer is always the better one.

A spot check of 20 new edits found a real error in each, for example "average-case O(1)" becoming
"O(n)" for hash tables, or freshwater that "reduces" surface salinity becoming "increases". About
three in twenty are softer, closer to a judgement call.

Building the dataset took three passes, A, A2 and A3; each stopped at a review gate.

| Pass | What went wrong | Fix |
|---|---|---|
| A | 35% of training answers were cut off at a 700-token cap. The judge rewrites paraphrased the whole answer, some named their own error, and only 13 of 64 pairs survived the filters. | Answers at 1200 tokens; the judge set made of minimal edits |
| A2 | Asked to copy the answer with one change, the teacher returned it unchanged 84 times out of 128. | A one-sentence edit as JSON, applied by the code, retried at rising temperature |
| A3 | 127 of 128 pairs pass every filter. | — |

## Measures (pre-registered)

1. **Critic pairwise accuracy on the judge set (primary).** This is the fraction of pairs where
   the trained critic scores the strong answer above the flawed one, with ties counted as half.
   It is reported with a 95% bootstrap interval that resamples prompts, because the pairs of one
   prompt share its strong answer.
2. **The same on the teacher-detectable subset.** These are the pairs where the teacher itself
   scored the strong answer higher. A critic learns from the teacher's scores, so this subset is
   where a difference between conditions can show.
3. **Held-out answer quality.** The teacher's mean score for each student's own answers to the 64
   held-out questions.

The bar for a difference was at least 0.10 between conditions, with non-overlapping intervals.

## Results

### 1. Critic accuracy on the judge set

| Critic | All 127 pairs (64 prompts) | Teacher-detectable 31 pairs (22 prompts) |
|---|---|---|
| Teacher (reference) | 0.594 [0.539, 0.650] | 1.0 by definition |
| control, local teacher | 0.724 [0.633, 0.812] | 0.710 [0.516, 0.882] |
| fine-tune + ASPIRE, local teacher | 0.638 [0.547, 0.724] | 0.677 [0.484, 0.848] |
| control, composite teacher | **0.866** [0.787, 0.929] | 0.839 [0.690, 0.964] |
| fine-tune + ASPIRE, composite teacher | **0.646** [0.559, 0.732] | 0.613 [0.414, 0.788] |

- **Composite teacher, all pairs: the bar is met, in the opposite direction.** The critic trained
  after the fine-tune is worse by 0.22, and the intervals do not overlap.
- **Local teacher, all pairs:** it is worse by 0.09, which is under the bar, and the intervals
  overlap.
- **Teacher-detectable subset:** it is too small to decide. Its 31 pairs give intervals about
  ±0.18 wide, and every difference falls inside them.

Checks on the result:

- **Not an artifact of length.** Accuracy is the same whether the strong side is longer, shorter or
  the same length. It is also the same for both flaw kinds and both ways of making a pair.
- **Why the critics beat the teacher.** The teacher scores in whole numbers and tied 89 of the 127
  pairs. When it does pick a side, it is right 31 times out of 38 (0.82).
- **The critic margins are small but consistent:**
  - each critic's score spreads about 0.6 across answers;
  - the gap between the two sides of a pair averages only 0.004 to 0.016;
  - yet the strong side wins 64% to 87% of the time.
- **One training run per condition.** The intervals cover the choice of prompts, not
  run-to-run variation in training. A second seed for each condition would show how much of the
  0.22 is the fine-tune and how much is chance.

### 2. Held-out answer quality

| Student | Teacher mean, 64 held-out questions | Answers cut off at 256 tokens |
|---|---|---|
| base | 7.34 [7.19, 7.48] | 51 |
| fine-tuned | 7.41 [7.23, 7.56] | 61 |
| control, local | 7.47 [7.34, 7.59] | 50 |
| control, composite | 7.25 [7.11, 7.39] | 54 |
| fine-tune + ASPIRE, local | 7.38 [7.19, 7.55] | 57 |
| fine-tune + ASPIRE, composite | 7.33 [7.16, 7.50] | 57 |

**No difference.** Every interval overlaps. This measure is weak as built, though:
- The students answer with the 256-token cap ASPIRE's dialogue generator uses, so most answers
  are cut off.
- The fine-tuned student, which learned the teacher's long answers, is cut off most often.

### 3. During training

| | fine-tune + ASPIRE, local | fine-tune + ASPIRE, composite | control, local | control, composite |
|---|---|---|---|---|
| Teacher scores (32 prompts) | mean 7.09, sd 0.76 | mean 5.30, sd 0.99 (Qwen 7.22, Gemma 3.39) | mean 7.16, sd 0.62 | mean 5.45, sd 0.93 |
| Critic loss, epochs 1 / 2 / 3 | 0.91 / 0.82 / 0.63 | 1.65 / 0.94 / 0.56 | 1.09 / 0.56 / 0.49 | 1.37 / 0.86 / 0.65 |

The fine-tune itself took 5 minutes: 2 epochs over 791 examples, with the loss falling from 0.94
to 0.87.

## The hidden-state trajectory

These probes use the method of the control report. The same 32 fixed exchanges go through each
model, and each exchange's state is compared with its own state in a reference model.

| Drift on the 32 exchanges | local | composite |
|---|---|---|
| Fine-tune (base → fine-tuned), mean size | 21.5 | 21.0 |
| Shared across exchanges (cosine to the mean drift) | 0.96 | 0.96 |
| ASPIRE after the fine-tune, epochs 1 / 2 / 3 | 0.45 / 0.86 / 0.99 | 0.43 / 0.68 / 0.84 |
| ASPIRE in the control, epochs 1 / 2 / 3 | 0.45 / 0.58 / 0.66 | 0.46 / 0.65 / 0.68 |
| Fine-tuned student's distance from base after ASPIRE | 21.5 → 20.8 | 21.0 → 20.4 |
| Cosine between ASPIRE's drift and the fine-tune's | −0.77 | −0.77 |
| Cosine between ASPIRE's drift here and in the control | 0.58 | 0.55 |
| Cosine between the control's ASPIRE drift and the fine-tune's | −0.36 | −0.43 |

**The fine-tune moves the representations about 30 times as far as ASPIRE does, and almost all in
one direction.** Each exchange moves about a tenth of the distance that separates two different
prompts.

**ASPIRE has a direction of its own, and it runs against the fine-tune.**
- In the control, with no fine-tune, ASPIRE already moved partly opposite to the fine-tune's
  direction.
- After the fine-tune, it points the same general way as in the control, but much more against
  the fine-tune, and it moves further.
- Each epoch brings the student back a little toward the base.

**None of these drifts is a quality axis.** A student's position along any of them does not track
the teacher's scores beyond a random-direction null (|r| at most 0.18, against a null of 0.32 to
0.47).

## What this says

- **The fine-tune did not help ASPIRE here, and with the composite teacher it hurt.**
  - The critic trained after the fine-tune lost 0.22 of pairwise accuracy at spotting planted
    errors.
  - The held-out answers did not get better.
  - This is one run per condition, so the size of the drop is not yet pinned down.
- **The likely mechanism is in the trajectory, not settled.**
  - The fine-tune pushed the student's representations a long way in one shared direction.
  - ASPIRE then spent its updates pulling partly back along that direction.
  - The control critic gets to work on the base model's states, which may carry the base model's
    sense that "O(n) for a hash table" is wrong.
- **The 1.2.0 caveat still applies.** The student is not trained toward better answers. ASPIRE
  trains the representations the critic reads, which is exactly what this experiment measured
  moving.

## The exports

There are six new ScalarScope fixtures, all schema 1.1:

| File | step_axis | Content |
|---|---|---|
| `sft-{local,composite}-teacher.geometry.json` | `training_step` | The trainer's per-step exports, 96 steps |
| `sft-{local,composite}-teacher.drift-from-base.geometry.json` | `checkpoint_by_item` | The fine-tuned student and ASPIRE epochs 1–3 × 32 exchanges, measured from the base student (128 steps) |
| `sft-{local,composite}-teacher.drift-from-sft.geometry.json` | `checkpoint_by_item` | ASPIRE epochs 1–3 × 32 exchanges, measured from the fine-tuned student (96 steps); directly comparable with the control's `*.drift` exports |

## Cost

| Session | What it did | Spent |
|---|---|---|
| Plan A | first dataset | $1.34 |
| Plan A2 | answers at 1200 tokens; judge set | $1.30 |
| Plan A3 | judge set as one-sentence edits | $0.40 |
| Plan B | fine-tune, both ASPIRE runs, probes, evaluation | $3.96 |
| **Total** | | **$7.00** |
