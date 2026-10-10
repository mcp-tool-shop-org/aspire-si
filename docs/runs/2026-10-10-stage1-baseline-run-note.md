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
