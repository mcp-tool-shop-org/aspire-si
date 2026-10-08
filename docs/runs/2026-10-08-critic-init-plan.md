# Plan: which seed carries the critic's spread (written 2026-10-08, before it runs)

## Why

Step 2 ([report](2026-10-08-step-2.md)) found the same 0.181 range in critic accuracy across seeds
42, 43 and 44 at both 32 and 128 training prompts, with the seeds in the same order. So the spread
follows the seed. Until the spread is smaller or understood, no comparison between conditions can
be read.

Today one seed drives everything in a run (`torch.manual_seed(config.seed)` in `AspireTrainer`):
- the critic head's initial weights;
- the student adapter's initialisation;
- the data order;
- the student's and the teacher's sampling.

This test splits off the first and asks which side carries the spread.

## The split (code, tested)

`critic.init_seed` (new, default None: no change for existing configs):
- The critic head is built from its own seeded stream, and the run's stream resumes as if no critic
  had been built.
- Only the CPU generator is reseeded, so the GPU stream that sampling draws from is untouched.

`tests/test_critic_init_seed.py` shows:
- the same critic seed gives the same critic whatever the run seed;
- a different critic seed gives a different critic, with the run's stream unchanged;
- the run seed still changes the run's stream;
- the GPU stream is not reseeded (run on the pod, where CUDA is available).

**What stays coupled.** The critic's score feeds the student's loss, and the critic's loss trains
the student through its hidden states. So changing only the critic seed can still change the
student, and with it the later dialogues. The random streams stay fixed; what is generated can
diverge. The report counts how many of arm A's 32 dialogues are identical across its three runs.

## Runs

The 32-prompt control-local run (`examples/local-run/prompts.json`, local teacher
Qwen2.5-32B-Instruct, three epochs, batch 1), as in run 1, written by `seed_configs.py --seed R
--critic-seed C`:

| Arm | Runs (R:C) | Varies | Fixed |
|---|---|---|---|
| A | 42:42, 42:43, 42:44 | the critic's initial weights | everything else (R = 42) |
| B | 42:42, 43:42, 44:42 | adapter init, data order, student and teacher sampling | the critic's initial weights (C = 42) |

The two arms share the 42:42 run, so there are five runs: one pod, `pod/plan_g.sh`, in two
batches.

## Readout (fixed before any run)

The measure is each run's pairwise accuracy on the 127 judge pairs (`judge_eval.py`, ties counting
half). Each arm's **range** is its highest minus its lowest of three, compared with the 0.181 range
of seeds 42 to 44 at 32 prompts.

| Arm A range | Arm B range | Reading |
|---|---|---|
| above 0.09 | 0.09 or less | **The critic's initial weights carry the spread** |
| 0.09 or less | above 0.09 | **The run's seed carries it**: adapter initialisation, data order or sampling. This test doesn't separate those. |
| above 0.09 | above 0.09 | **Both contribute** |
| 0.09 or less | 0.09 or less | **Neither alone at three seeds.** The spread comes from both varying together, or a range of three cannot show it. |

- An arm whose range is at least 0.135 (three quarters of 0.181) is reported as carrying most of
  the spread.
- 0.09 is half of 0.181, the same bar as step 2's rule 1.
- **The limit:** a range of three draws is noisy, and so is the 0.181 it is compared with. A
  reading here points the next experiment; it doesn't settle it.

Also reported, with no rule attached:
- each run's prompt-clustered 95% interval;
- the 42:42 run against run 1's seed-42 control (0.724);
- arm A's identical-dialogue count;
- the drift exports.

What happens next, such as fixing or ensembling the critic's initial weights, or more seeds per
condition, is a separate plan and the maintainer's call.

## Cost

Priced from step 2's measured rates:
- about 1.67 min per dialogue per run with three side by side;
- about 5 min for epochs 2 and 3.

| Part | Time |
|---|---|
| Setup, tests, speed probe, downloads (refused below 66 GB in 30 min) | ≤ 0.6 h |
| Batch 1: three runs, 32 dialogues each, plus epochs | about 1 h |
| Batch 2: two runs | about 1 h |
| Probes, judge, pull | 0.3 h |
| Margin | 0.6 h |
| **max_hours** | **3.5** |

One RTX PRO 6000, no fallback, at most $2.19/hr: **worst case about $7.70**.

**Stop rules:**
- A refused host stops the run.
- At 30 minutes into batch 1, the finish is projected from the progress line: batch 1's remaining
  dialogues, batch 2 at the same rate, plus 0.5 h. If that lands past the deadline, the pod stops
  at once.
- Results are pulled and checked before the shutdown.
