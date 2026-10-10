# Stage 1 baseline: run note (pinned before the run)

**What:** the Phase A "before" measurement on the 30-task pilot. The same harness, prompt and settings run after
stage-1 training, and on the sealed 120 once it is frozen. Nothing is tuned on the sealed results.

**Student:** Qwen/Qwen3-8B at revision `b968826d9c46dd6066d109eabc6255188de91218`, loaded by `load_student()` in
[`stage1-eval/student.py`](stage1-eval/student.py): the snapshot's bf16 weights, SDPA attention, no quantisation.
The trainer uses the same loader. Before loading, the harness checks the sha256 of all 12 pinned files
([stage1-student-weights.md](stage1-student-weights.md)) and refuses on any mismatch.

**Harness:** [`stage1-eval/`](stage1-eval/) (`harness.py`, `student.py`, `mapper.py`, committed byte-exact).

| Pinned | Value |
|---|---|
| harness sha256 (harness.py + student.py + mapper.py, in that order) | `8d63a541aea1891222f9728391503c4344253ffc2df1b13c22666f7af7b4fe11` |
| harness.py | `4fa52448a1be79230eb8fa13118853abd9615c150fec7f270b1e58082741c0c7` |
| student.py | `3fc4e32c7278a5c5e8a4dea02bdc01960c4994c346b2b1a2c7afa1cc6284c70d` |
| mapper.py | `cbe3630b770f67f774dd11cb7a4835fcfdac1c0090dd94558b5c2b8dda20b67a` |
| prompt (`stage1-eval-v1`) sha256 | `77d6303f4f9828fa6b9906dc3ef2f4c855181e2b3fbea76ef59465f108428024` |
| tasks: pilot-30 = review-10 then pilot-20, concatenated | `3f5987da0b585eb11c16009659a8c681929caa4f71ec1e0ccc3d83e3d5cae8c8` |

**Settings:**
- **Prompt:** role-neutral. It names the task and the three-line answer format, and teaches nothing about how to
  verify (the text is in `harness.py`).
- **Thinking:** on (Qwen3 thinking mode).
- **Decoding:** the Qwen3 model card's thinking-mode settings: temperature 0.6, top-p 0.95, top-k 20, min-p 0.
  The card advises against greedy decoding in thinking mode, so this replaces the earlier temperature-0 plan
  (the Publisher's decision, 2026-10-10).
- **Seeds:** the generator is re-seeded before every generation. The pilot uses seed 0, one sample. The sealed
  120 uses seeds 0, 1 and 2 at baseline and again after training.
- **Tokens:** max_new_tokens 16384 per turn, with truncation recorded. Batch 1.
- **Pressure tasks:** each later turn is sent as a new user message, and the last turn's answer is the one
  graded.

**What's recorded** (`outputs/stage1-baseline/`, gitignored):
- `raw.jsonl`, per task and turn: the full thinking, the answer, prompt and new tokens, truncation, seconds, the
  parsed answer and the grade (verdict correct, DECIDING verbatim, ESCALATE correct).
- `mapped-steps.jsonl` and `mapped-traces.jsonl`, in the plan's record format. These are written by the
  mechanical mapper `mechanical-v0`: checks, revisits and settles read off quoted material lines, mapped to
  ideal-tree parts. The tree structure beyond that comes from the person or controlled-model mapper later.

**Analysis (Phase A):**
- **Comparison:** the task-level paired bootstrap on per-task mean correctness over the three seeds, with
  McNemar on the majority vote as a check (R&D's statistics plan, as amended by the Publisher).
- **The pilot's job:** check that baseline accuracy sits between 20% and 85%. If it doesn't, the difficulty is
  adjusted on the pilot only.

**Card terms (the Publisher's grant):** one holder; cap 120 min; stop on a watchdog trip or error; stop and
report if most turns hit the token cap.

## The pre-interview (same grant, right after the pilot-30 baseline)

The questions are the Publisher's ([2026-10-10-stage1-pre-interview-draft.md](2026-10-10-stage1-pre-interview-draft.md),
copied verbatim) and are **sealed**: they never appear in any lesson, quiz or training data. The interview runs
before any training touches the weights. It repeats with identical wording after Stage 1 and after domain
training, in both arms.

- **Runner:** [`stage1-eval/interview.py`](stage1-eval/interview.py). Same student, loader and decoding as the
  harness (`harness.generate`, the card's sampled settings, re-seeded before every generation). Each question is
  asked in its own call, with the frame in front of it.
- **Size:** 12 questions × seeds 0, 1, 2 = 36 generations. Thinking and answer are recorded per question and
  seed in `outputs/stage1-baseline/interview/interview.jsonl`.

| Pinned | sha256 |
|---|---|
| interview.py | `57d2d6eadbf2492e3c99efb88f385a007f85f3aba129f4ca5d06d91e4db6a9cf` |
| frame (`stage1-pre-interview-v1`) | `8e790237e1d212212a26876a3e9d79eefb2538eab12918d0782d6615cd116a45` |
| the 12 questions (JSON list) | `b59927e18ff7ecbaafad346fb85499afb39125dfd66d8d5e05886afe5fc57ed8` |
| harness (as above) | `8d63a541aea1891222f9728391503c4344253ffc2df1b13c22666f7af7b4fe11` |

## Machine (times are comparable only on the same setup)

NVIDIA GeForce RTX 5090, driver 617.14; CUDA 13.4; torch 2.16.0.dev20261008+cu134; transformers 5.19.0;
batch 1, bf16, SDPA. Every later sitting's run note records the same line.

## Results (pilot-30, seeds 0–2; the interview, 36 answers)

Scored after the run by [`stage1-eval/score.py`](stage1-eval/score.py), the eval plan's pinned correctness rule
plus the coverage fix proposed to R&D, and summarised by [`stage1-eval/summarize.py`](stage1-eval/summarize.py).
The generation path is unchanged, so the harness hash above still holds.

| Seed | Correct (verdict + DECIDING) | Strict | Verdict only |
|---|---|---|---|
| 0 | 17/30 | 8/30 | 25/30 |
| 1 | 15/30 | 8/30 | 26/30 |
| 2 | 14/30 | 6/30 | 24/30 |

- **Verdict by key** (3-seed mean): supported 100% (n=9), unsupported 90% (n=17), cannot_tell 17% (n=4).
  Silent material is answered "unsupported".
- **ESCALATE** was flagged on 4 of 15 escalation-keyed answers. The role-neutral prompt states the format but
  not when to escalate, so this partly reflects the prompt and is not the model's ceiling. The same prompt
  runs after training, so any gain comes from training.
- **No truncations.** The median is about 325 new tokens per task.
- **Time per task** (pooled over seeds):

  | Group | Median seconds |
  |---|---|
  | correct | 14.2 |
  | wrong | 17.1 |
  | tier 3 | 15.2 |
  | tier 4 | 16.3 |
  | tier 5 | 14.9 |
  | tier 6 | 26.0 (n=3) |

  Time is flat across tiers 3–5: at baseline the model doesn't spend more on harder tasks. Generation took
  28.2 minutes for the 3 seeds, which is 1.6 correct answers per minute of generation. The interview took
  13.1 minutes.
