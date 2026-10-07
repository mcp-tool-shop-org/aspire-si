# Plan: the next training runs (written 2026-10-07, before any of them)

This plan follows the fine-tune-then-ASPIRE experiment ([report](2026-10-07-sft-then-aspire.md)).
It ranks the next runs by how much each could change that experiment's conclusion. Each run has
its decision rule fixed here, before it starts. The control runs (2026-10-06) and the seed-42
fine-tune runs stay the comparison points. Nothing in this plan runs until Mike approves it.

**Where we stand.** $10.89 of the $20 cap is spent and $9.11 is left.
- The headline result rests on one training run per condition. With the composite teacher, the
  critic trained after the fine-tune scored 0.646 at spotting planted errors, against the
  control's 0.866.
- The 32B teacher ties 89 of the 127 judge pairs, so the teacher-detectable subset has only 31
  pairs.

## Costs: how to read them

Every run is on one RTX PRO 6000 (96 GB), rented through offrig's `job` profile at about $2.09
an hour.
- **Expected:** hours × $2.09.
- **Worst case:** what offrig commits before a launch: hours cap × $3.49, the profile's price
  ceiling. A plan cannot set a lower ceiling.
- **One pod at a time:** a project lane allows only one live pod, so runs go one after another.
  The budget has to cover the next run's worst case on top of what is already spent.

None of these runs touches the local GPU. All training, generation and scoring happen on the pod.
Afterwards, the analysis runs on this machine's CPU.

## The ranking

| Rank | Run | What it could change | Expected | Worst case |
|---|---|---|---|---|
| 1 | Two more seeds per condition | Whether "the fine-tune hurt the composite critic" is a finding or one draw | $12.2 | $12.22 per pod, two pods |
| 2 | Held-out answers at 1024 tokens | Whether the fine-tune taught the material at all, which is the experiment's premise | $1.9, shared with run 3 | $4.19 |
| 3 | A judge subset the teacher can separate, found by pairwise comparison | Whether the inconclusive second measure becomes readable | (in run 2's pod) | (in run 2's pod) |
| 4 | #11, the policy-gradient term | Nothing about this conclusion; it opens the next question | about $10–15 after design | $12.22 per pod |

**Run 1 goes first.** The one result that met the bar could be a single training draw.
- Every interval so far resamples prompts and does not cover training variance. A new seed
  changes the LoRA and critic initialisation and the sampled dialogues, so it also changes the
  teacher's training scores.
- If the drop does not reproduce, the conclusion becomes "no reliable difference". Nothing else
  on the list can overturn the result like that.

**Run 2 is second.** It checks the premise.
- Every student's held-out answers were cut at 256 tokens. Their scores (7.25–7.47) cannot show
  whether the fine-tune taught anything.
- If it did not, the honest reading of the experiment changes. What the critic absorbed would
  then be a large shift in its representations, not learned material.

**Run 3 rides in run 2's pod.** It needs the same 32B teacher on vLLM, and only about 10 minutes
more.

**Run 4 is the next experiment.** It does not test the current one.

## Run 1: seeds 43 and 44

**What it measures.** For each new seed, four critics are trained as on 2026-10-07, under that
seed:
- control-local and control-composite, from the base student;
- sft-local and sft-composite, from a fresh fine-tune at that seed.

All four go through `judge_eval.py` on the same 127 judge pairs, and through `probe_models.py`
on the control's 32 exchanges.

**The schedule, per seed and per pod.**

| Stage | Minutes |
|---|---|
| Setup and downloads | 6 |
| Fine-tune at the seed | 5 |
| control-composite and sft-composite, side by side | about 100 |
| control-local and sft-local, side by side | about 60 |
| Probes and judge | 5 |
| **Total** | **about 176 minutes, 2.9 hours** |

- **Memory.** Two composite runs should need about 80 GB of the 96. This is an estimate, from
  the 63 GB that one local and one composite run used together.
- **Cost.** The hours cap is 3.5 per pod. That gives $6.1 expected and $12.22 worst case per
  pod, and $12.2 expected for the two seeds.

**Decision rule** (on the 127-pair set, for each teacher separately, with three seeds per
condition counting seed 42):

| Result | What it means |
|---|---|
| All three fine-tune critics score below all three control critics, and the mean drop is at least 0.10 | **The fine-tune hurts the critic, for this teacher.** The report states it as a finding. |
| The mean drop is at least 0.10, but the seed ranges overlap | **A tendency, not a finding.** The report gives the per-seed numbers. |
| The mean drop is under 0.10 | **No reliable difference.** The 0.22 seen with seed 42 was mostly one draw, and the report and CHANGELOG are corrected to say so. |

The seed-to-seed spread of the control critics is reported as well. It is the first measure of
how much ASPIRE's critic varies between runs.

**Stop conditions.**
- **A stage fails:** pull what exists, shut down, report. Stages are resumable.
- **The two composite runs exceed GPU memory:** stop that stage and run them one after the
  other. That needs about 1.5 hours more, which is over the cap, so pull, shut down and re-plan
  before relaunching.
- **The deadline is near:** pull everything 15 minutes before it.

**Code needed first (in a PR, tested, no GPU).**
- `pod/plan_c.sh SEED`: per-seed copies of the four configs, with the seed, output folder and
  run name set. The CLI has no seed option.
- The fine-tune at that seed.
- The two side-by-side pairs, then probes and judge.

## Run 2: held-out answers at 1024 tokens (with run 3)

**What it measures.** The six seed-42 students answer the 64 held-out questions again, with up
to 1024 new tokens; the teacher scores those answers. The six students are:
- base and fine-tuned;
- the two controls;
- the two fine-tune + ASPIRE runs.

This needs:
- a `--max-new-tokens` option in `eval_heldout.py`;
- the merged fine-tuned student rebuilt on the pod, from the base plus `sft/epoch-2`;
- the seed-42 checkpoints uploaded (about 400 MB).

**Decision rule.**

| Result | What it means |
|---|---|
| The fine-tuned student beats the base by at least 0.2 in mean teacher score, with non-overlapping intervals | **The fine-tune taught the material, by the teacher's measure.** The 2026-10-07 reading stands. |
| Anything less | **The fine-tune moved the representations without improving answers.** The report says so: the experiment then tested a large representation shift, not learned material. |
| An ASPIRE student differs from its starting student by at least 0.2, with non-overlapping intervals | Reported as such. The 1.2.0 trainer does not train the student toward better answers, so the expected result is no difference. |

Fewer than 90% of answers ending within 1024 tokens is reported. If it happens, the cap is
still too low to compare quality.

## Run 3: a judge subset the teacher can separate

**What it measures.** The teacher compares the two sides of each of the 127 pairs directly ("which
answer is better?"), in both orders.
- A pair counts as *teacher-separable* when the teacher picks the strong side both times.
- The critics' pair scores are already recorded in `judge.json`, so the existing seed-42 critics
  are re-read on this subset with no retraining. Run 1's critics are added when they exist.

This needs a `pairwise_teacher.py` script, and a `--subset` option in `judge_eval.py` that reads
its output.

**Why a new definition.** The pre-registered subset used absolute scores, and the teacher's whole
numbers tie 70% of pairs. A direct comparison asks the teacher the question the judge set asks.
This is a new, second subset, labelled as such. The original stays in the record as it was.

**Decision rule.**

| Result | What it means |
|---|---|
| The subset has at least 60 pairs across at least 35 prompts | Reported with prompt-clustered intervals, using the same bar as the primary measure (at least 0.10, intervals apart). |
| It is smaller | **Too small to decide.** Reported as such, as with the 31-pair subset. |

**Runs 2 and 3: one pod.**

| Stage | Minutes |
|---|---|
| Setup | 6 |
| Answers | 25 |
| vLLM load | 4 |
| Scoring | 10 |
| Pairwise comparisons | 8 |
| **Total** | **about 55 minutes** |

The hours cap is 1.2. That gives $1.9 expected and $4.19 worst case.

## Run 4: #11, the policy-gradient term (the next experiment)

**What it measures.** Whether training the student toward the critic's or teacher's reward:
- improves held-out answers (at 1024 tokens);
- turns the hidden-state drift into a quality axis, meaning a student's position along it tracks
  the teacher's scores beyond the random-direction null.

**It needs design and code before any GPU time.** That covers the reward source, a KL penalty to
the starting student, and tests. Its runs, seeds and costs get their own plan when the design is
done.

**Rough size.** Policy-gradient steps generate extra samples, so expect about 1.5 times the
current run time.
- Base and fine-tuned students, local teacher, three seeds: about $10–15 expected.
- The composite teacher roughly doubles that.

## What to approve

| | Expected | Spent afterwards | Cap needed for each launch to fit |
|---|---|---|---|
| **Recommended first: run 1 (seeds 43 and 44)** | **$12.2** | about $23.1 | **$36**: $10.89 spent, plus two worst-case pods of $12.22 |
| Whole plan: runs 1–3, plus run 4 at its rough size | about $27 | about $38 | **$50**: the expected total, plus one worst-case pod of headroom |

The other option is to approve runs 2 and 3 first, since they are cheap and fit inside the
$9.11 left. That gets the premise checked before the seeds, but it does not answer the question
the seeds answer.
