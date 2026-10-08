# Plan: critics with chosen attributes, the Auditor and the Advocate (2026-10-08)

**Status: the readout below is final when this plan's PR merges, before anything runs.** It folds in
the R&D session's review of the first draft (seven points) and the Publisher's notes on it.

## Where it comes from

The critic-init test ([report](2026-10-08-critic-init.md)) produced a critic that preferred the
flawed answer on the 127 judge pairs: run R42:C43, at 0.315 with its interval entirely below 0.5.
The maintainer's reading is that this critic is an **auditor**: it seeks out the wrong answer in
order to grade.

This plan:
- tests that reading on pairs the critic wasn't found on;
- trains critics for chosen attributes on purpose, so the attribute comes from the training signal
  and not from a lucky draw;
- checks first that the setup can see a planted error at all;
- reads the critics alone, as a panel, and on errors planted by a different model family.

## 1. The found Auditor, read on fresh pairs first

- **Checkpoint:** `outputs/control-local-r42-c43/checkpoint-3` from plan 20, kept under the name
  `auditor-found-r42-c43` (folder and name).
- **How it's read:** as a flaw detector. Its flaw score for an answer is the negative of its critic
  score.
- **On the 127 judge pairs:** 0.685, CI [0.594, 0.773].
  - That number comes from the set on which the critic was picked out for being extreme, so it is
    a selected number. Regression to the mean is the default expectation.
  - It is reported, not used as a bar.
- **The committed reading is on validation** (the 149 confirmation pairs, which it has never been read
  on), read once before any new critic is trained:

| Its flaw-detection accuracy on validation | Reading |
|---|---|
| CI entirely above 0.5 | **An auditor.** The name stands, and its validation number is the bar for fostered Auditors. |
| CI includes 0.5 | **A noise draw, not an auditor.** The name is withdrawn and recorded as such, and fostered Auditors are measured against chance and the baselines. |
| CI entirely below 0.5 | It prefers the flawed answer on the judge set but the strong answer here. Reported as unstable, not an auditor. |

## 2. Material, kept apart

| Set | Pairs | Prompts | Planted by | Use |
|---|---|---|---|---|
| `train_pairs.jsonl` | 603 | 310 | Qwen2.5-32B-Instruct Q4 | Training |
| `confirm_set.json` | 149 | 78 | the same | **Validation**: controls, role checks, panel membership, the found Auditor |
| The judge set | 127 | 64 | Qwen2.5-32B-Instruct bf16 | **Final evaluation**, read once |
| **Second-planter set** (new) | about 45 | 26 | **gemma4:31b**, local | **Transfer**, read once beside the judge set |

- **No two sets share a prompt.** R&D's leak check found 0 shared prompts across the first three. The
  second-planter set uses the 26 questions that no other set uses.
- **All of the first three come from one planter family.** The judge set's edits match the others in
  size, length behaviour and position. So validation is in-distribution, and the judge set tests new
  prompts and a change of precision, not a different planter. A head that keys on how Qwen2.5-32B
  edits would pass all three.
- **The second-planter set tests transfer:**
  - Strong answers come from Qwen2.5-32B-Instruct Q4 (llama-server, `--load-mode none`), as for the
    training set.
  - Errors are planted by **gemma4:31b** (local model ID 6316f0629137, recorded in the set's
    report.json), a non-Qwen family, through the local Ollama daemon. The planting code refuses any
    model name containing "cloud". The model is unloaded afterwards through the API (keep_alive 0).
  - **Edit statistics are reported before the set is read:** characters changed, length change and
    position in the answer, beside the same three for the Qwen-planted sets (median 5–6 characters
    at 0.35–0.44 of the way in). If gemma's median edit size differs by more than 2×, the transfer
    reading carries that caveat.
  - The same edit request, retries and filters as `fresh_pairs.py` (two flaw kinds per prompt), so
    about 45 pairs survive. That's small, and its intervals will be wide; it says so.
- **The 127 judge pairs have been read many times:** by the ASPIRE critics, by the Kev readings, and
  in picking out the found Auditor. The report says so, which is one more reason for the
  second-planter set.

## 3. Chosen attributes

A critic with its sign flipped is the same critic, so the Auditor gets a different *task* from the
Advocate, not just a flipped target.

| Critic | Attribute | Training signal |
|---|---|---|
| **Auditor** | Finds the error in one answer | Pointwise binary cross-entropy: "does this answer contain an error?" (flawed = 1, strong = 0). Each pair contributes both answers, so each strong answer appears once per pair and the classes are balanced. |
| **Advocate** | Prefers the stronger of two versions | Pairwise margin ranking: score(strong) above score(flawed) |
| **Teacher-mimic** (existing) | Predicts the teacher's 0–10 score | ASPIRE's critic objective; plan 20's five non-inverted critics |

**Features (pinned):**
- **Student:** the base `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`,
  frozen, loaded in 4-bit as in the ASPIRE runs.
- **Input:** the prompt and one answer in the student's chat format (`aspire.judge.encode_exchanges`),
  truncated at 1536 tokens. Truncation counts are recorded per set; the longest answer is about 5,200
  characters, so none should be cut.
- **Hidden states:** the last layer's, computed once and cached per token, so every head trains on
  the same features.

**Pooling, because the edit is tiny.**
- A planted edit changes a median of 5–6 characters in an answer of 3,200–3,450 characters. Mean
  pooling over about 1,000 tokens dilutes it to a fraction of a percent.
- So each role is trained in two forms:

| Form | Pooling | Fair because |
|---|---|---|
| **Auditor-mean, Advocate-mean** | Mean over the attention mask (`CriticHead`'s default, as in ASPIRE) | The ASPIRE critic's own input |
| **Auditor-attn** | Learned attention pooling (`CriticHead(pooling="attention")`), which can weight a few tokens | Uses nothing the Auditor wouldn't have on one answer |
| **Advocate-span** | Mean over the tokens of the edited span only | The pair is known, which is exactly the information a pairwise judge such as Kev has. The pointwise Auditor can't use it, so it isn't given it. |

**Advocate-span is a different capability.** It pools over the difference between the two answers,
which exists only when both are shown. It is labelled **"judging a located change"**, not "finding
an error", and is never compared head-to-head with the Auditor's single-answer task without saying
so. Mean and attention pooling are the like-for-like comparison.

- **Starting points:** three critic init seeds per form (42, 43, 44), so a lucky or unlucky draw
  shows. A head's weights come from its seed alone, as with `critic.init_seed`.
- **Training settings, fixed in advance for every head and control:**
  - `CriticHead` with hidden size 512, 2 layers and dropout 0.1;
  - AdamW at learning rate 1e-4, weight decay 0.01;
  - 5 epochs of 8 pairs per batch.

  Nothing is tuned on validation; validation only decides role checks and panel membership.
- **The Skeptic** (paraphrase edits with no error) stays out unless the maintainer adds it. The
  second-planter set takes its role as the check on "learned this planter's edits".

## 4. Readout

Every interval is a 95% bootstrap that **resamples whole prompts**; pairs share prompts and strong
answers, about two pairs per prompt. Comparisons between two critics use a **paired**
prompt-clustered bootstrap of their difference.

**A. Controls, read first, on validation.**

| Control | Head | Must | If not |
|---|---|---|---|
| **Positive (gate)** | Every form, trained with one fixed marker string appended to every flawed answer; validation's flawed answers marked the same way | Reach at least 0.95, with a CI lower end of at least 0.90 | That form's pipeline is broken. Nothing about it is read. |
| **Shuffled labels** | Each form, trained with the flawed/strong label randomised per pair, same three init seeds; the three seeds' validation scores are pooled for one interval | Have a CI that includes 0.5 | The features carry something besides the planted error (length, position, formatting). No role reading is made. |

- **Truncation:** the cache fails loudly if any answer, or any marker, would be cut off by the
  1536-token limit (prompt, chat template and answer together).
- **The shuffled control's power is stated:** the report gives the smallest confound it could have
  caught (the half-width of its pooled interval).
- **Graded control (a diagnostic, not a gate):** one fixed rare token inserted *at the planted edit's
  position* instead of at the end, in every form. It separates two explanations of a failed
  Auditor:
  - mean pooling misses it but attention pooling catches it: a failed Auditor-mean is explained by
    pooling;
  - every form catches it while real edits still fail: the error itself is hard for these features.

**B. The found Auditor on validation**, as in section 1.

**C. Each fostered critic on validation**, by its role:
- **Auditor:** pairwise accuracy of its flaw score (flawed ranked above strong), and pointwise AUC
  for "contains an error".
- **Advocate:** pairwise accuracy (strong ranked above flawed), and pointwise AUC as a
  strong-answer detector.
- **Role check:**
  - **Rejected:** its role's CI lies entirely below 0.5. It goes the wrong way for its role, is kept
    out of the panel, and is reported.
  - **Flagged:** its point estimate is below 0.5 but its CI crosses 0.5. Kept out of the panel and
    reported.

**D. Are the Auditor and the Advocate different critics?**
- Before any panel claim: the correlation, over validation answers, between the Advocate's score and
  the Auditor's flaw score (each form's seed-mean).
- **Beyond −0.9:** they are one critic counted twice, and the panel row is moot.
- **Whether their mistakes coincide** is reported beside the correlation:
  - the chance-adjusted error consistency (Geirhos et al. 2020, the hard-decision form of CAPA,
    Goel et al. 2025), which is 1 when they miss the same pairs;
  - the double-fault rate, the share of pairs both miss.

  These are reported with no rule attached. A panel helps only where its members' errors differ.
- The real difference between the roles should show in pointwise calibration across prompts, so
  both AUCs are compared too.

**E. The panel:**
- An answer's panel score is the mean Advocate score minus the mean Auditor flaw score.
- Members are the critics not rejected or flagged, whose form passed its positive control.
- Advocate-span is not a member, because it judges a located change and has no score for an answer
  alone.
- Each score is standardised on validation (mean 0, sd 1 over validation answers).
- Weights are equal and not tuned.
- **"The panel beats its best member"** needs the paired, prompt-clustered interval of the difference
  to exclude 0.

**F. Final, read once: the judge set and the second-planter set.** Each critic by role and the
panel, beside the baselines:

| Baseline | On the 127 judge pairs |
|---|---|
| Kev-4B, frozen, order-averaged (the pinned reference judge) | 0.976 |
| The R&D Kev-4B judge fine-tune, trained on the same 603 pairs (rnd-kev-judge-4b @ 51beba99, three seeds) | 0.965–0.969 single choice; 0.992–1.000 order-averaged |
| The best earlier ASPIRE critic (run 1, seed 42, composite control) | 0.866 |
| The found Auditor (a selected number; see section 1) | 0.685 |

Kev-4B and the found Auditor are also read on the second-planter set. The R&D session has offered
to read its Kev judge fine-tune (three seeds) there too ($0, local, about 10 minutes). It trained on
Qwen-planted pairs, so it faces the same "learned the planter's edits" question. That runs only on
the maintainer's yes.

**What counts:**

| Result | Reading |
|---|---|
| A fostered Auditor above the found Auditor on validation (paired CI excluding 0), on at least two of three seeds **within one form** | The attribute can be trained on purpose, better than the lucky draw, **in that form**. Each form is reported on its own; one form passing and the other failing is reported as such. If the found Auditor was a noise draw (section 1), the bar is chance. |
| Every fostered Auditor below 0.6 on validation, on all seeds and both forms, with the positive control passed | Error-finding is not learned **by a pooled head on frozen 1.5B states**. That says nothing about richer critics. |

**Transfer to the second planter**, from a two-sample prompt-clustered bootstrap of (judge-set
accuracy − second-planter accuracy):

| Result | Reading |
|---|---|
| The second-planter CI lies above 0.5, and the difference interval includes 0 | **Transfers** across planter families |
| The difference interval excludes 0, with the judge set higher | **Learned the planter's edits**, not errors in general |
| Anything else | **Inconclusive**, said as such. With about 45 pairs, this is the likely outcome for small differences. |
| The panel above its best member (paired CI excluding 0) | Combining the attributes helps |

**The comparison that matters:**
- A supervised head on 603 planted pairs beating a lucky draw is the expected result.
- The report says where every critic sits against Kev-4B and against R&D's Kev fine-tune, which saw
  the same data: a 4B attention model, against a pooled head on a frozen 1.5B.
- Matching them, or explaining the gap, is the finding.

## 5. Where it runs and what it costs

Everything is local, at $0, on the RTX 5090. Before each GPU step, the maintainer is told and the GPU
is booked through the Publisher (job, VRAM, duration), with the watchdog running. One job at a time.

| Step | GPU job | VRAM | Time |
|---|---|---|---|
| 1 | Read the found Auditor on validation (its student adapter and critic, 4-bit) | about 5 GB | about 10 min |
| 2 | Second-planter set: strong answers for 26 prompts with Qwen2.5-32B Q4 (llama-server); stop it; then gemma4:31b edits through Ollama | about 25 GB, then about 22 GB | about 1 h |
| 3 | Cache the frozen student's hidden states for all four sets | about 5 GB | about 20 min |
| 4 | Train the heads and controls on cached features | under 5 GB | about 30 min |

**A pod is not needed.** More teacher-mimic critics inside ASPIRE (three runs at 32 prompts, worst
case about $4.80) stays an option only if the maintainer asks for it.

## Order

1. This plan's PR, with the readout final, and a code PR:
   - the second-planter builder (`fresh_pairs.py` with a separate planting backend);
   - the hidden-state cache;
   - the head trainer: pointwise and pairwise, mean, attention and span pooling, the positive and
     shuffled controls;
   - the readout, with paired prompt-clustered intervals.

   Each piece is tested.
2. Step 1, the found Auditor on validation. Its reading is recorded before anything is trained.
3. Steps 2 to 4, then the readout in the order of section 4, and a report PR.
