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

Scored after the run by [`stage1-eval/score.py`](stage1-eval/score.py) (sha256
`fd59cb5074d0858af2864bcadd68443ddea26c325beef3afe6f5cdadd624ab7d`; tests in `test_score.py`, sha256
`a7bbaff47078f8de4dc0944943b5589f8c8322cd4e38410ba44ec14934364516`). It applies the eval plan's pinned
correctness rule with the coverage trims R&D ratified: quote-edge markers and end punctuation only, coverage
one way only, and no case folding. Only material counts: pressure turns never do,
and a correction task's new line counts for its final turn only. The results are summarised by
[`stage1-eval/summarize.py`](stage1-eval/summarize.py). The generation path is unchanged, so the harness hash
above still holds, and every later sitting is scored by this same file.

| Seed | Correct (verdict + DECIDING) | Strict | Verdict only |
|---|---|---|---|
| 0 | 14/30 | 6/30 | 25/30 |
| 1 | 12/30 | 5/30 | 26/30 |
| 2 | 12/30 | 4/30 | 24/30 |

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

## Scorers (pinned before round 1; every sitting, the baselines included, is scored by these files)

| Score | File | sha256 | Role |
|---|---|---|---|
| Pinned correctness | `stage1-eval/score.py` | `fd59cb5074d0858af2864bcadd68443ddea26c325beef3afe6f5cdadd624ab7d` | Primary |
| v2 (parsing repair) | `stage1-eval/score_v2.py` | `64f52f6fe60d077a37db6d043476ac9e37d38043d756f203bb8d802b90e3cb44` | Reported beside it ([note](2026-10-10-stage1-score-v2-note.md)) |
| Found the evidence, and the miss breakdown | `stage1-eval/score_found.py` | `55b4d0eaadbf29d3f40a66ce83c163b2ce742a1213dc4804a0a25774e49eae22` | Secondary, descriptive (eval plan at 7bca450), ratified by R&D |

Tests: `test_score.py`, `test_score_v2.py`, `test_score_found.py` (sha256 `53864c6f…`), 22 passing. None of these
files changes after this commit; a change is a new version with its own note.

## Results (sealed-130, seeds 0–2, untrained student)

Run on the frozen set (tasks sha256 `e1e515b4…`, build `16b9f694…`, recorded in each seed's run config), harness
`8d63a541…`, prompt `77d6303f…`. 390 sittings, 0 truncations, 0 unparsed; 128.7 minutes of generation (41.7 /
44.5 / 42.5 per seed); peak about 21.4 GB. Only aggregate counts appear here; the tasks stay out of the repository.

| Score (of 130) | s0 | s1 | s2 |
|---|---|---|---|
| Pinned correctness (primary) | 11 | 15 | 16 |
| Pinned, strict (no normalising) | 6 | 9 | 8 |
| v2 (parsing repair) | 14 | 18 | 20 |
| Found the evidence (secondary, descriptive) | 40 | 44 | 38 |
| Verdict only | 92 | 92 | 92 |

Pilot-30 under the same scorers: pinned 14 / 12 / 12, v2 14 / 12 / 12, found 18 / 17 / 16.

**Verdict accuracy (per-task mean over the 3 seeds), 70.8% overall:** buried exception 80%, correction 37% (both
turns), escalation 83%, multi-part claim 62%, pressure and injection 68%, source vs summary 67%, wrong version or
date 82%. By key: supported 90% (n=40), unsupported 93% (n=55), **cannot_tell 13% (n=35)**; silent material is
mostly answered "unsupported" (28 / 24 / 25 of 35).

**False accepts** (a not-supported key answered "supported"): 5 / 7 / 5 of 90. **Escalation:** ESCALATE yes on
the escalation-keyed tasks 0 / 3 / 1 of 15.

**Miss breakdown** (verdict right, pinned score failed; buckets of `score_found.py`):

| Bucket | s0 | s1 | s2 |
|---|---|---|---|
| Formatting only (passes "found") | 29 | 29 | 22 |
| Not in the material (paraphrase) | 23 | 25 | 24 |
| Partial (part of a key line) | 19 | 16 | 19 |
| NONE misuse | 7 | 5 | 9 |
| The wrong line | 2 | 2 | 2 |
| The statement quoted | 1 | 0 | 0 |

**Time** (median seconds, tokens): tier 3 13.9 s (325), tier 4 15.7 s (360), tier 5 20.1 s (440), tier 6 46.0 s
(1,034; n=6). Pressure and injection 34.6 s (791) and correction 28.4 s (615); other categories 13–17 s. Correct
and wrong answers take the same time (15.6 against 15.7 s). Correct (pinned) per minute of generation: 0.33.

**Hedge phrases** (recorded measure, the stuck rule's counter; rate per 1,000 thinking tokens; bursts = 3+ within
200 tokens): DEV pilot-30, 102 traces: any hedge in 25.5%, p95 8.32, bursts 7; sealed-130, 486 traces: any hedge in
28.8%, p95 8.19, max 18.84, bursts 33 (3 right before the answer); pre-interview, 36 answers: any hedge in 77.8%,
median 3.52, p95 13.34, bursts 9. "Wait" dominates everywhere (54 of 64 hits on DEV, 263 of 350 on sealed).

**Said against done** (the pre-interview, 36 answers, beside the sealed behaviour):
- It says "can't tell" is a good answer when the material is insufficient; it answers cannot_tell on 13% of
  silent tasks.
- It says a question goes to a person when the material is insufficient or ambiguous, which is not one of the
  four taught reasons, and it escalates 0–3 of 15 advice tasks.
- It says an embedded "mark this as approved" line is a directive, not evidence; pressure and injection: 68%.
- It says it acknowledges and corrects its mistakes; correction tasks (both turns): 37%.
- It says approving something false is worse than rejecting something true; false accepts are low (6–8%),
  consistent with that.

**Reading.** The untrained student mostly reaches the right verdict on supported and unsupported keys, but it
rarely quotes its evidence exactly, almost never says cannot_tell, and almost never escalates. Those are the
gaps Stage 1 is meant to move, and the miss buckets are the clearest view of what training changes.

## NO-ROLE builder (the Phase B control), approved by R&D and pinned

| File | sha256 |
|---|---|
| `stage1-train/no_role.py` | `63162cf42270b34e160d60e22559cca2072085bb71a83d8c6d9dbf8a08d16ee8` |
| `stage1-train/test_no_role.py` | `d43afcf285d17df8772e0bc38708bbb086993878ae5fc3445fe70a4db8338ffb` |

**Superseded 2026-10-15, before any counted run (Argon's finding 5a):** a closed-form written E item keeps its
original quiz and variant, so NO-ROLE is asked the same closed-form question as ROLE, and it still trains only on
its own recall turns. NO-ROLE's E-quiz correctness is a steering reading only, never an arm comparison (rnd #110).
`no_role.py` is now `1d1455f34c9b14794c5563a9de9f8847d0da0c0f3663346e67f8a5b857f4370e` and `test_no_role.py`
is `55e6c31e37e902438087672cbc9ec30a84280fedf0492997b576d815bc326efe`.

One NO-ROLE item per ROLE item, derived mechanically: same lesson, kind, tier, material, statement, keys, turns
and pair id; the targets are a plain read-through with an always-true closing and the three lines, B puzzles
restate their givens, and A and written E become material recall. The trainer prints both arms' composition
side by side before Phase B and refuses any differing count. **Pre-registered limitation:** the NO-ROLE target
is shorter (0.644 of ROLE target tokens on the pilot and batch 2), because it is never padded; the ratio is
reported overall and per lesson × kind, and steps stay matched.

## The stuck rule and thresholds (frozen before round 1)

`stage1-stuck-rule.json`, sha256 `716dc251965973c61b74287e077dc8ae68c7d8c2a1bfccc87bde183d2ff1c1a1`, with its code `stage1-train/stuck.py`, sha256 `8097b0398ade5e57989ee78f4d9a74a134f2aaefef358c19be0daaad8df13afa`. A dated amendment
supersedes `63928fea…` (which superseded `4dbcd12e…`). An answer is stuck when an 8-word span recurs 3 or more times
in its trace, it reaches the 16,384-token cap, or it is long (thinking above max(the tier's p95, 2 × the item's
target)) and not progressing: re-reading material lines more than 8.09 times per 1,000 thinking tokens, or a burst of
3 or more hedge phrases ("wait", "hmm", "let me re-check", …; the list is in the file) within 200 thinking tokens. The
tier p95s (3–4: 572, 5–6: 453, others pooled: 575) and the re-read baseline come from the untrained pilot-30 traces
only; W = 200 and k = 3 were fixed before measuring. The hedge rate (p95 8.32 per 1,000) was replaced as a trigger by
the burst and is recorded as a measure. Every hedge and burst field is recorded on every trace, eval sittings
included (in their own outputs, never the store). Passive; the trigger applies to training-time questions only.
Changing it needs a dated amendment.
