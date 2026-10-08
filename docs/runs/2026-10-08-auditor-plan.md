# Plan: critics with chosen attributes, the Auditor and the Advocate (DRAFT, 2026-10-08)

**Status: a draft written before a rig restart so the direction survives it. The readout below is
the proposal. It is final when this plan's PR merges, before anything runs.**

## Where it comes from

The critic-init test ([report](2026-10-08-critic-init.md)) produced a critic that reliably prefers
the flawed answer: run R42:C43, at 0.315 with its interval entirely below 0.5. The maintainer's
reading is that this critic is an **auditor**: it seeks out the wrong answer in order to grade.

This plan:
- keeps that critic under that name;
- trains critics for chosen attributes on purpose, so the attribute comes from the training signal
  and not from a lucky draw;
- reads them alone and as a panel.

## 1. The found Auditor (committed now, before anything new runs)

- **Checkpoint:** `outputs/control-local-r42-c43/checkpoint-3` from plan 20, kept under the name
  `auditor-found-r42-c43` (folder and name).
- **How it's read:** as a flaw detector. Its flaw score for an answer is the negative of its critic
  score.
- **On the 127 judge pairs, the reading is fixed now:** it ranks the flawed answer above the strong
  one on 0.685 of pairs (1 − 0.315), CI [0.594, 0.773].
- On the validation set (below), it is read the same way, once, before any new critic is trained.

## 2. Material, kept apart

| Set | Pairs | Prompts | Use |
|---|---|---|---|
| `train_pairs.jsonl` (Kev confirmation, planted by Qwen2.5-32B Q4) | 603 | 310 | Training the new critics |
| `confirm_set.json` (the same, other prompts) | 149 | 78 | **Validation**: role checks, panel weights, rejection |
| The 127-pair judge set (planted at bf16, held-out prompts) | 127 | 64 | **Final evaluation only**; never used to train, select or tune |

None of the three shares a prompt with another. The R&D session's Kev fine-tune also trained on
`train_pairs`; that doesn't touch these critics.

## 3. Chosen attributes

A critic with its sign flipped is the same critic, so the Auditor needs a different *task* from the
Advocate, not just a flipped sign.

| Critic | Attribute | Training signal | Why it's distinct |
|---|---|---|---|
| **Auditor** | Finds the error | Pointwise: one answer at a time, "does it contain an error?" (binary cross-entropy, flawed = 1, strong = 0) | It must judge an answer alone, without seeing the better version |
| **Advocate** | Prefers the stronger answer | Pairwise: score(strong) above score(flawed), a margin ranking loss on the pair | It only has to compare two versions, an easier and different skill |
| **Teacher-mimic** (existing) | Predicts the teacher's 0–10 score | ASPIRE's own critic objective, on dialogues | The ASPIRE critic as trained so far; plan 20 already has five non-inverted ones |

- **Architecture:** each new critic is ASPIRE's `CriticHead` on the student's hidden states.
  - The student is the base Qwen2.5-1.5B-Instruct, frozen.
  - The same input is used as the ASPIRE critic, so the three differ only in the training signal.
- **Starting points:** three critic init seeds each (42, 43, 44, through `critic.init_seed`), so a
  lucky or unlucky draw shows.
- **Further attribute (proposed, needs the maintainer's yes): a "Skeptic".**
  - Trained pointwise like the Auditor, but on the strong answers only, against their own one-sentence
    paraphrases.
  - It would learn to flag changed wording without a changed meaning, a control for whether the
    Auditor detects errors or just edits.
  - Left out unless approved.

## 4. Readout (proposal, final at merge)

**Per critic, on validation (149 pairs):**
- **Auditor:** pairwise accuracy of its flaw score (flawed ranked above strong), plus the pointwise
  AUC for "contains an error".
- **Advocate:** pairwise accuracy (strong ranked above flawed).
- **Teacher-mimic:** pairwise accuracy (strong above flawed), as before.

**Role check, on validation:**
- **Rejected:** a critic whose 95% interval for its role's accuracy lies entirely below 0.5. It goes
  the wrong way for its role and is excluded from the panel and reported as such.
- **Flagged:** a critic whose point estimate is below 0.5 but whose interval crosses 0.5. Reported,
  and kept out of the panel.

**Panel:**
- The panel score for an answer is the mean Advocate score minus the mean Auditor flaw score.
- Each score is standardised on validation first (mean 0, sd 1 over the validation answers), so
  neither critic dominates by scale.
- Weights are equal and not tuned, so validation decides only who is in, not how much each counts.

**Final evaluation on the 127 judge pairs** (once, after validation):
- each critic, by role;
- the panel;
- the found Auditor (fixed above);
- Kev-4B read order-averaged (0.976) as the reference.

**What counts:**

| Result | Reading |
|---|---|
| A fostered Auditor at or above 0.685 (the found Auditor) on the judge pairs, on at least two of three seeds | The attribute can be trained on purpose, at least as well as the lucky draw |
| Fostered Auditors below 0.6 on all seeds | Pointwise error-finding is not learned from 603 pairs; the found Auditor stays a one-off |
| Panel above both its best single member on the judge pairs | Combining the attributes helps |

## 5. Where it runs and what it costs

- **Training the new heads needs no teacher and no dialogues.** It needs only the frozen 1.5B
  student's hidden states over the planted pairs: 603 pairs × 2 answers, plus validation and judge.
  - That fits on the local RTX 5090 (the 1.5B student at 4-bit is a few GB) at $0.
  - The maintainer is told first, and the GPU is booked through the Publisher session.
- **A pod (offrig, 32 prompts) is needed only for critics trained inside ASPIRE**, such as more
  teacher-mimic critics from new init seeds. Priced from measured rates (1.43 min per dialogue with
  three side by side, about 5 min of epochs): three runs ≈ 0.6 h setup + 1 h + 0.6 h margin ≈
  2.2 h, worst case about $4.80. Proposed only if the maintainer wants more teacher-mimic critics
  beyond plan 20's five.

## Order

1. Commit the found Auditor's reading (done above) and read it on validation, locally, at $0.
2. A code PR:
   - a planted-pair critic trainer (pointwise Auditor, pairwise Advocate) with tests;
   - a panel scorer that reads all three kinds.
3. This plan's PR merges, with the readout final.
4. The GPU is booked, the maintainer is told, and training and the readout run.
