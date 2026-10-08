# Plan: the next step, critic variance and a Kev reference judge (written 2026-10-08, before any of it runs)

Three seeds of the fine-tune-then-ASPIRE experiment ([run 1 report](2026-10-07-run-1-seeds.md))
gave no reliable difference between conditions. They showed something more basic instead.

- **The critic varies a lot between training runs.** With 32 training prompts, the same
  condition gave a critic anywhere from 0.425 to 0.866 in pairwise accuracy on the 127 planted-error
  pairs.
- **That spread blocks every comparison.** Until it shrinks, a stronger fine-tune or #11 cannot be
  read either.

This step measures that spread and looks for a way to shrink it, and folds in a reference judge
that needs no training.

Nothing here runs until the maintainer approves it. Step 1 uses the local GPU, so the maintainer is
told first and the VRAM watchdog must be running.

## Step 1: Kev as a reference judge (local GPU, $0)

**What it is.** [Kev](https://github.com/jaredpalmer/kev) is an open decision model (Apache-2.0).
- It takes one forward pass and returns calibrated probabilities over a fixed set of options. It
  generates no text.
- Kev-4B and Kev-9B are adapters plus a pointer head on a Qwen base model.
- The weights are already cached on this machine, and the studio has run them here before.

**What it measures.** Kev-4B and Kev-9B judge the 127 pairs of the existing judge set. Each pair
is a question with two answers; the options are "A is more correct" and "B is more correct".

| Measure | Definition |
|---|---|
| Accuracy | Each pair asked in both orders; accuracy averaged over the two orders |
| Position bias | How often option A is chosen, over both orders |
| Margin | The probability gap behind each choice; the probabilities themselves are miscalibrated on new tasks, so they are not read as scores |
| Order sensitivity | The `/permute` endpoint, on a subset |
| Kev-separable subset | Pairs Kev decides for the strong answer in both orders |

These sit next to the 32B teacher's numbers on the same pairs:
- absolute scores: 0.594, with 89 of 127 pairs tied;
- direct comparison: "A" 127 times out of 127 when the strong answer came first, and the strong
  answer 49 times when it came second.

**Decision rules.**

| Result | Reading |
|---|---|
| Accuracy over both orders at least 0.75, and option A chosen 40–60% of the time | Kev is a usable reference judge, and its accuracy is reported as the bar a trained critic should clear |
| Accuracy at least 0.75, but A chosen outside 40–60% | Kev is accurate but position-biased; it is reported with both orders always |
| Accuracy under 0.75 | Kev is not a useful judge of single planted errors, and is reported as such |

**The Kev-separable subset**, if it has at least 60 pairs across at least 35 prompts:
- the twelve critics already trained (3 seeds × 4 conditions) are read again on it, at no cost,
  from their recorded pair scores;
- it gets the run 1 rule: a mean difference of at least 0.10 across three seeds.

Kev-separable is a different definition from the teacher's subsets. It is labelled as such.

**How it runs.**
- `kev.serve` with GPU memory capped at 82%. An uncapped Kev-4B grew to 31.9 GB here and nearly
  tripped the watchdog.
- It refuses to start while the GPU is busy, and stops as soon as the pairs are done.
- 127 pairs × 2 orders × 2 sizes takes a few minutes per size at about 0.5 s per request.

**Code first (tested PR):** `judge_kev.py`, which builds the requests, asks in both orders, reads
the choice and margin, and writes the subset; plus a `--subset` re-read through `judge_eval.py`.

## Step 2: critic variance against the number of training prompts (one pod)

**What it measures.** Three seeds of the control-local condition (base student, Qwen2.5-32B
teacher) trained on **128 prompts** instead of 32. Everything else stays as on 2026-10-06: three
epochs, batch 1, the same critic.
- The 128 prompts are drawn, stratified by topic, from the experiment's 400 training questions.
- They are disjoint from the 64 held-out questions behind the judge set, and were already checked
  against the 32 control prompts.
- Each critic is measured on the same 127 pairs, alongside Kev's numbers from step 1.

**Why this condition.** It is the cheapest: one teacher. Its spread at 32 prompts is already known:
0.543, 0.638 and 0.724, a range of 0.18. That gives a direct before and after.

**Decision rules.**

| Result | Reading |
|---|---|
| Range across the three 128-prompt seeds at most 0.09 (half the 32-prompt range) | 128 prompts is the training size for every later comparison (stronger fine-tune, #11), with three seeds per condition |
| Range still above 0.09 | Prompt count alone does not tame the critic. The next plan tries more epochs, more seeds, or a different critic setup, and nothing is compared until it does |
| Mean below 0.60 | Also reported: more data does not make the critic better at planted errors, only possibly steadier |

**Pod.**
- One RTX PRO 6000 (96 GB) with a 180 GB container disk.
- The three runs train side by side, at about 25 GB each.
- At 32 prompts a local run took about 48 minutes, mostly generating dialogues. Four times the
  prompts, with three runs sharing the GPU, is estimated at about 4.5 hours.

**Cost.**

| Item | Value |
|---|---|
| Plan | 5.5-hour cap, 20-minute capacity wait, no fallback, at most $2.20/hr |
| Expected | about $9.40 |
| Worst case | about $12.70 (5.8 h × $2.19) |
| Budget left | $39.51 |

**Stop conditions, as before.**
- The host check (driver, GPU memory, write and download speed) refuses a bad host.
- A run that cannot finish within its cap is stopped.
- Results are pulled before the deadline, and the pod is shut down once they are pulled.

**Code first (tested PR):**
- the 128-prompt file;
- a config writer for N prompts and a seed;
- `pod/plan_e.sh`.

## Order and what comes after

1. **Step 1**, Kev, local and $0, as soon as its code is merged and the maintainer has been told.
2. **Step 2**, the pod, once priced and approved.
3. **What follows depends on step 2:**
   - if 128 prompts settles the critic, the stronger fine-tune and #11 are planned at that size,
     three seeds each, with Kev as the fixed reference judge;
   - if it does not, the critic setup comes first.
