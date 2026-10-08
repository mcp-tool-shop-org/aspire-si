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

**Also reported for the found Auditor and every head, with no rule attached** (added after step 1,
before any head was trained):
- pointwise AUC;
- **noise-band accuracy:** pairwise accuracy with any pair whose gap is no larger than 0.002 counted
  as a tie (half). 0.002 is the median score difference measured when the found Auditor was re-scored
  locally against the pod. A head with clean margins then isn't scored the same as a near coin-flip
  that lands on the right side.

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

## Addendum: more feature sources, to separate family from size (written 2026-10-08, before anything ran)

**Why.** Every critic here reads the hidden states of a Qwen model, and every planted error so far
came from Qwen2.5-32B, except the second-planter set. A Qwen model may recognise its own family's
editing style (self-recognition, Panickssery et al. 2024), so the Qwen-on-Qwen-planted cell is the
most likely to flatter a critic. The maintainer has access to Meta's Llama 3.2 models.

**Family is tangled with size** (R&D's review). Qwen2.5-1.5B (hidden size 1536) against
Llama-3.2-3B (3072) changes family, size, width and pretraining data at once. So there are three
sources:

| Source | Model, revision | Hidden size, layers | License |
|---|---|---|---|
| `qwen` (primary, as above) | Qwen/Qwen2.5-1.5B-Instruct, `989aa798…` | 1536, 28 | Apache-2.0 |
| `qwen3b` | Qwen/Qwen2.5-3B-Instruct, `aa8e7253…` | 2048, 36 | Qwen research license |
| `llama` | meta-llama/Llama-3.2-3B-Instruct, `0cb88a4f…` | 3072, 28 | Llama 3.2 Community License |

- `qwen` against `qwen3b` isolates **size** within one family.
- `qwen3b` against `llama` isolates **family** at about the same size. Their widths still differ,
  so each head's parameter count is recorded beside its results.
- All three are Instruct models, like-for-like.

Every form, seed and control is trained on each source:
- frozen, 4-bit, each with its own chat template;
- the last layer as the primary feature, for every source;
- the 1536-token truncation check run for each tokenizer.

Nothing else changes: sets, forms, training settings, controls and readout. Each source gets its own
full readout.

**What is compared**, with no rule attached:
- **Per form, between two sources** (`critic_heads.py compare`): seed-mean accuracy on validation,
  the judge set and the second-planter set, with a paired, prompt-clustered interval of the
  difference.
- **The family pattern as one number** (`family_contrast`):
  - (a − b) on the Qwen-planted judge set minus (a − b) on the gemma-planted set, with a two-sample
    prompt-clustered interval.
  - Only `qwen3b` against `llama` is read as family; any other pair is labelled "family and size
    mixed".
  - With about 45 gemma pairs the interval will be wide, so **"inconclusive" is the likely reading**,
    and the plan names it as such.
  - No pattern is called "family recognition" from the mixed comparisons.

**An exploratory depth diagnostic, with no rule.**
- Probing work tends to find middle layers stronger than the last. So one extra feature per source is
  cached: the mean-pooled hidden state about two thirds of the way in (layer 19 of 28 for both 28-layer
  models, 24 of 36 for Qwen2.5-3B).
- Auditor and Advocate heads are trained on it, three seeds, with no controls. They are reported
  apart from the committed readout.
- If the last layer looks weak everywhere, this shows whether depth is the bottleneck.

**Built with Llama.** This study reads features from Meta's Llama 3.2 3B and from Qwen2.5 (1.5B and
3B) alongside each other, so critics are tested across model families. Heads trained on these
features are shared under each model's license terms:
- Llama-derived work carries "Built with Llama" and follows the Llama 3.2 Community License.
- Qwen2.5-3B-derived work follows the Qwen research license.
- Qwen2.5-1.5B-derived work follows Apache-2.0.

Before any derived head, score set or model card is shared, its license terms are checked and its
attribution is added.

**Recorded with each cache:** source, model, revision, 4-bit loading, hidden size, the primary
(last) layer, the exploratory layer, and the license.

**Cost:** two more cache passes (Qwen2.5-3B is about 6 GB to download), each about 7 GB of VRAM and
about 25 minutes, booked with the Publisher like the rest. Training the heads per source adds
minutes. Still $0.

## Addendum 2: the Skeptic control (written 2026-10-08, after step 4 and before any of its data exists)

**Why.** Step 4's shuffled-label control failed by design ([report](2026-10-08-auditor.md)), so no
role reading was made. The numbers it left raise a sharper question. In every planted pair, the
flawed answer is also the *edited* one. A head that learned "this copy was edited" would score near
1.0 pairwise without knowing anything about errors. That fits the fostered heads' 0.89–0.97 pairwise
accuracy alongside a pointwise AUC of only about 0.55. The maintainer approved this control. **Until
it is on main, no role number from step 4 is re-read.**

### The paraphrase pairs

**What a pair is:** a strong answer, and the same answer with **one sentence reworded so that its
meaning is unchanged** and it introduces no error. In the pair file the original takes the "strong"
slot and the paraphrased copy the "flawed" slot, so a head is read exactly as on error pairs. Here a
"win" means ranking the *edited* copy as the worse (Advocate) or as flawed (Auditor).

**How they are planted:**
- The request mirrors the error edit: change as few words of one sentence as possible (aim for one
  to three words), keep the meaning exactly, add no error, and don't mark the change. The reply is
  JSON `{original, edited}`, applied by code.
- They pass the same filters as error edits: similarity at least 0.90, length change at most 5%, no
  self-flagging, not unchanged.
- **The planter matches each set's error planter**, so only one thing varies between the error pair and
  the paraphrase pair: whether there is an error.
  - P-train and P-confirm (Qwen-planted errors): **Qwen2.5-32B-Instruct Q4_K_M** on llama-server
    (`--load-mode none`), as `fresh_pairs.py` used.
  - P-second (gemma-planted errors): **gemma4:31b** (local Ollama 6316f0629137, `"think": false`,
    `refuse_cloud`).
  - The maintainer asked for gemma4:31b. The R&D session's review showed that gemma paraphrases
    against Qwen errors would let a head that detects "a Qwen-made edit" pass as error-specific, the
    exact failure this control exists to catch. So gemma plants only where gemma planted the errors.
- **Two paraphrases per strong answer, on different sentences.** The second request names the first
  rewording's sentence and asks for another. 78 strong answers then give about 156 pairs, and an
  edit-rate interval of about ±0.08 instead of ±0.11.
- Edit size is reported against the matched error edits, using R&D's difflib method. If the median
  paraphrase edit differs from its matched error edits by more than 2×, the reading carries that
  caveat.
- **A size gate, added after the first planting and before any head read a paraphrase:**
  - The first planting asked for a one-to-three-word rewording. It came out 5–6× larger than the
    matched error edits (median 29 against 5 characters on P-confirm, 31 against 6 on P-train).
    Only 4–7% of pairs kept the same length.
  - That batch is kept on disk as the record and is not used.
  - The re-plant asks for one or two words replaced with synonyms, and nothing else.
  - **Size gate, per answer:** a paraphrase whose edit exceeds **twice that answer's own error
    edit** (difflib, with a floor of 8 characters) is rejected and retried, like a failed filter, for
    up to 7 attempts per paraphrase. The report gives the size quantiles (25/50/75%) of the kept
    paraphrases against their matched error edits, so the match is shown, not assumed.
  - **Lexical guard, before any model check** (R&D's review): a swap is rejected if a changed token
    on either side is any of these:
    - a quantifier (all, every, each, most, many, some, few, several, none, no, any);
    - a frequency word (always, usually, often, sometimes, rarely, never);
    - a modal (can, could, may, might, must, should, will, would, shall);
    - a negation (not, no, n't, without);
    - a comparative or superlative;
    - a digit, a number word or a unit;
    - a named entity (a capitalised word mid-sentence).

    These are the swaps that flip a claim. Guard rejections are counted separately from size-gate
    rejections. Run over the first batch as a check, it would have stopped about a third (51 of 140
    on P-confirm).
  - **Meaning check with context:** mistral-small:24b sees the two sentences and one sentence of
    context on each side. It answers two questions: same meaning, and whether any claim was added,
    removed, strengthened or weakened. A pair is kept only on "same meaning: true" and "claim changed:
    false"; anything else, unparsed included, is dropped. Its drop rate on the gated set is reported
    beside the first batch's.
  - **Retry tell:** each kept pair records the round that produced it. The edit rate is reported
    separately for first-round and later-round paraphrases. If the later ones read differently,
    that's a sign of planting bias.
  - The report gives the gates' rejection rates, and how many answers end with no paraphrase pair or
    only one.
- **A word-level format, added after the gated re-plant and before any head read a paraphrase:**
  - The gated re-plant matched the size (median 13 against 10 characters on P-confirm, 16 against
    10 on P-train) but kept too few: 38 pairs on 26 of 78 P-confirm answers, and 124 on 88 of 310
    P-train answers. The size gate rejected about 41% of attempts; the model kept rewriting more
    than it was asked to.
  - **What the error edits look like, measured:** 82% of the confirm set's error edits change one or
    two words (64% exactly one), as do 83% of the training pairs' and 77% of the judge set's. A
    one-word swap is the matching paraphrase.
  - **The new request** asks the same planter for a JSON `{sentence, word, synonym}`: one ordinary
    word in one sentence, and a single-word synonym of the same grammatical form. Code makes the
    swap. A reply is rejected, and each reason counted, when:
    - the sentence isn't found in the answer;
    - the word isn't found exactly once, as a whole word, in that sentence;
    - the synonym isn't a single word, or is the same word;
    - the synonym is an inflection of the word ("find" to "finds"), which is a grammar edit, not a
      synonym;
    - the swap changes the word's form (plural or third person, past, progressive), which would
      break agreement;
    - either word is a function word (an auxiliary or copula, pronoun, determiner, demonstrative,
      preposition or conjunction). A synonym swap should hit a content word, and this closes the
      irregular agreement changes ("is" to "are") that the suffix check can't see;
    - the word sits inside inline code, a fenced block or `$…$` maths in the answer, where it is an
      identifier or symbol, not wording.
  - Code keeps the word's capitalisation and fixes "a"/"an" before it by sound, not by letter ("an
    hour", "a unique"), so the edited copy carries no article slip as a tell.
  - The report gives the yield of each arm (answers with two pairs, one, or none) and the full
    rejection breakdown, so the two-per-answer target is shown, not assumed.
- **A size-matched margin, added after planting and before any head read a paraphrase (R&D's
  review):**
  - The planting matched the medians but not the spread. On P-confirm the paraphrase edits' 5th–95th
    percentile band is 4–10 characters; the error edits run wider (quartiles 3/5/11). On P-second the
    band is 4–10.9.
  - Beside the headline margin, the readout gives the same margin on only the error pairs whose
    edit size falls inside the band (taken from the kept paraphrases, after the meaning check),
    prompt-clustered as before, with its n.
    - Measured before the meaning check: 63 of 149 confirm error pairs fall inside the band, and 19
      of 47 second-planter error pairs.
  - If the size-matched margin differs from the headline in sign, or in whether its CI excludes 0,
    the row says **size is a residual cue**. That flag doesn't change the row's reading.
  - The CI part of the rule applies only when at least 30 error pairs fall inside the band. Below
    that the CI is too wide and would trip on noise, so only a sign flip raises the flag, and the
    subset is still reported. This affects P-second (19 pairs), not P-confirm (63).
  - Everything above still applies: the filters, the per-answer size gate, the lexical guard, the
    meaning check with context, two swaps per answer on different sentences, and the retry tell.
  - The planter is unchanged per set: the Q4 32B for P-train and P-confirm, gemma4:31b for
    P-second.
- **Two arms, and a rule for when they disagree (R&D's review, committed now):**
  - **Primary: the word-level pairs.** They train the Skeptic heads (P-train) and give the main
    reading (P-confirm, P-second).
  - **Secondary: the 38 model-rewritten P-confirm pairs** from the gated re-plant, after the same
    meaning check. Each head's edit rate is read on them too, with its own CI.
  - If the two arms' edit-rate CIs **don't overlap** for a head, its reading is **"unresolved
    (depends on paraphrase method)"**, whatever either arm says alone. Otherwise the primary arm's
    reading stands, and the secondary is reported beside it.

**On which answers:**

| Set | Strong answers | Use |
|---|---|---|
| **P-train** | the 310 training prompts' strong answers | Training the Skeptic heads only |
| **P-confirm** | the 78 confirmation prompts' strong answers | The main check: matched against the confirm set's error pairs on the same strong answers |
| **P-second** | the 24 second-planter prompts' strong answers | The same-generator check: gemma's error edits against gemma's paraphrase edits on the same strong answers |

- No head has trained on a confirmation or second-planter prompt. The P-train pairs train only the
  new Skeptic heads, never the step 4 heads.

**Meaning check, a gate on each pair.**
- A local judge of a third family, **mistral-small:24b** (`"think": false`, `refuse_cloud`), reads
  only the two sentences: does the reworded one keep every fact, number, condition and step?
- **Any pair it flags as changing the meaning is dropped**, and the drop rate is reported.
- Unparsed verdicts count as flagged.

**Whole-set check.**
- Frozen Kev-4B reads P-confirm order-averaged, as in the Kev confirmation.
- If it prefers the original above 0.65, with a CI excluding 0.5, the paraphrases are degrading the
  answers. The set fails, and it is re-planted before anything is read.
- Kev alone isn't the gate, because it may legitimately prefer smoother wording.

### Rule for the existing heads (committed now)

**Re-scoring.** Every step 4 head (162; the trainer kept no weights) is **retrained from the same
cache, seed and settings**, and also scored on P-confirm and P-second.
- Retraining reproduces step 4: each head's validation accuracy must come within 0.02 of its step 4
  value.
- Where it doesn't, the report says so, and that head's Skeptic reading uses the retrained head and
  is labelled as such.

For each form and source, from seed-mean per-pair wins:
- **edit rate** = how often the head ranks the paraphrased copy as worse (Advocate) or as flawed
  (Auditor), on P-confirm;
- **error rate** = its accuracy on the confirm error pairs **built on the same strong answers**;
- **error margin** = error rate − edit rate, as a paired, prompt-clustered interval over strong
  answers that have both kinds of pair.

| Result (on P-confirm, prompt-clustered 95% CI) | Reading |
|---|---|
| Edit rate CI includes 0.5 | **Error-specific:** it doesn't fire on meaning-preserving edits |
| Edit rate CI entirely above 0.5, and error-margin CI includes 0 or lies below it | **An edit detector:** it fires on any edit as much as on errors |
| Edit rate CI entirely above 0.5, and error-margin CI entirely above 0 | **Partly an edit detector:** it fires on edits, more on errors. The error margin is its error-specific part. |
| Edit rate CI entirely below 0.5 | **Prefers the edited copy:** reported, with no reading |

The rows are mutually exclusive. The error margin is error rate − edit rate per strong answer, from
seed-mean wins, with a prompt-clustered interval.

When R&D's reversed correction pairs are read later, an "error-specific" head that fails them is
relabelled **"unexplained"**: it reads something else, such as wording quality.

- P-second gets the same table, as the same-generator check, read once.
- **A role reading for a form needs both:**
  - "error-specific" or "partly an edit detector" here;
  - plus the balanced permutation control below.

  Only then are the step 4 role rows (the claims, the panel, transfer) read for that form, and the
  error margin is reported beside them.

### The balanced permutation control (replaces the failed shuffled control, R&D's design)

Step 4's shuffled control flipped about half the pairs at random. Any imbalance in the flips gave
the learnable edit direction a sign, so the null came out bimodal. The replacement:
- **Half the pairs flipped within each prompt.** In a prompt with an odd number of pairs, the spare
  pair is flipped in every second such prompt, taken in a seeded order. Across the training set the
  flipped and unflipped pairs then differ by at most one, so a shared edit direction cancels. Each
  permutation's flipped fraction is logged.
- **20 such permutations per form and source**, each a head trained exactly like the true-label
  heads (seed 42).
- **Permutation test** (Ojala & Garriga 2010): p = (1 + number of nulls at or above the true-label
  head's validation accuracy) / 21.
  - Like for like: each null is a single seed-42 head, so the test uses the **seed-42 true-label
    head**. The three-seed mean is reported beside it.
- **A form passes when p < 0.05.** With 20 nulls the smallest possible p is 1/21 ≈ 0.048, so passing
  means beating all 20.
- **Timing caveat:** this control's rule is set after step 4's role numbers were seen. The report
  labels it so, and gives it no more weight than the Skeptic, which was set before its data.

### The Skeptic heads

- **Training:** pointwise, like the Auditor: "has this answer been edited?" (paraphrased copy = 1,
  original = 0), on P-train. Mean and attention pooling, three init seeds, each feature source, and
  the same training settings as every head.
- **Read on:**
  - **P-confirm:** pairwise and AUC. It should detect paraphrase edits; if it can't, edits aren't
    visible to these features, and the Auditor result above needs that caveat.
  - **The confirm error pairs:** how often it ranks the error copy as edited. If about as often as
    paraphrases, error pairs carry the same edit signal.
- **Exploratory, with no rule:** "Auditor flaw score minus Skeptic edit score", standardised on
  validation, read on the confirm error pairs and P-confirm.

### Later: R&D's correction pairs, a reversed check

The R&D session is labelling natural-error answers and writing a correction for each error. In
those pairs the *corrected* copy is the edited one, and the original holds the error. An edit
detector picks the wrong copy systematically; an error detector doesn't. When the file is filed, it
is read once with the same table, through an addendum of its own.

### Cost

All of it is local on the RTX 5090, at $0, gated through the Publisher:
1. Paraphrase planting: Qwen2.5-32B Q4 for P-train and P-confirm (about 776 requests, about 30
   minutes); then gemma4:31b for P-second (about 48, a few minutes); then mistral-small:24b for the
   meaning check. One model at a time.
   Kev-4B reads P-confirm for the whole-set check (a few minutes).
2. Cache passes for P-confirm, P-second and P-train: three sources, a few minutes each.
3. Retraining the 162 heads, plus the Skeptic heads and the 20-permutation nulls: about 30 minutes.
   **Every head's weights are saved** (state dict, config, seed), so later controls read the same heads.


## Addendum 3: the Kev gate becomes a report, and a typicality match replaces it (written 2026-10-08, after the Kev check failed and before any head read a paraphrase)

**What happened.** The word-level P-confirm set passed the meaning check (121 of 149 pairs kept),
then failed the whole-set check above. Frozen Kev-4B, order-averaged, preferred the original on
0.781 of the kept pairs [0.700, 0.854], above the 0.65 line with a CI excluding 0.5 (0.762 [0.686,
0.832] on all 149). Under that rule the set was not read. Nothing has been cached or read since.

**Why the rule changes.**
- Kev's preference is consistent in sign but small: the mean order-averaged gap is 0.018 on these
  paraphrases, against 0.131 on the confirm error pairs.
- The R&D session's review: that pattern is Kev reading **typicality**. Any edit replaces some of
  the strong model's own wording, so the edited copy is a little less typical in context. Planted
  errors carry the same cue.
- The control needs the paraphrases to carry that cue as much as the errors do, so that a head
  reading typicality fails it. A gate that rejects paraphrases for being less typical than the
  original pushes the other way. A paraphrase set that passed it would make a typicality reader
  look error-specific.
- So the rule mixed up two things: meaning damage, which the per-pair meaning check already
  catches, and the cue the control must keep. Neither re-planting option considered fixes that.
  Matching paraphrases to the original's typicality removes the cue from one side only, and
  swapping slots relabels without removing anything.
- **The maintainer decided** to amend the rule, before any head read a paraphrase.

**The amended rule.**
1. The mistral-small:24b meaning check with context stays the per-pair gate.
2. Kev's whole-set number is **reported, not gated**, with its gap beside the error pairs' gap. The
   failed result above stays in the record.
3. **A typicality match replaces it:**
   - A third model measures each pair's typicality shift. It is outside every family already in the
     pipeline: the student, the planter and Kev are Qwen; the feature sources are Qwen and Llama;
     gemma plants P-second; mistral checks meaning.
   - The model is **microsoft/Phi-3-mini-4k-instruct** at revision f39ac1d2 (MIT), in bf16. R&D
     suggested llama3.1:8b; that model's weights aren't on the rig, and Phi-3 is the more
     independent choice.
   - **delta (the primary measure)** = the sum of log p over the edited copy's tokens from the
     first changed token to the end of the answer, minus the same sum over the original's. Each
     token is scored in context: the question, then the answer up to that token.
     - Each copy stops at its own answer end, before the chat template's trailing `<|end|>` and
       `<|endoftext|>`.
     - Checked on the tokenizer: exactly those 2 tokens trail every one of the 3,184 exchanges.
     - Because the prefixes are identical, this is the full-sequence difference.
   - The R&D session's review set the tail as the primary measure, before any delta was computed.
     An unusual word also makes the text after it more surprising. A planted error usually knocks
     on further than a synonym (a wrong number makes the next sentence's arithmetic surprising), so
     scoring only the changed tokens would understate the errors' shift and tilt the match.
   - **Secondary, reported only:** the changed tokens alone, between the two copies' common prefix
     and common suffix. Where the two measures pass or fail differently, the readout says so.
   - The exchange is formatted with the model's chat template.
   - It is measured for the error pairs and for the kept paraphrases of each set.
   - Measured beforehand on the tokenizer alone, with no model loaded: the changed span is a median
     of 1 token for the paraphrases and 1–2 for the errors, and the longest exchange is 1,218
     tokens.
4. **Pass rule, committed before computing:**
   - On each read set (P-confirm against the confirm error pairs; P-second against the
     second-planter error pairs), the kept paraphrases' **median tail delta lies inside the error
     pairs' interquartile range of tail delta**.
   - A two-sample Kolmogorov–Smirnov statistic and its p-value are reported beside it, not gated.
   - P-train against the training error pairs is reported only.
   - **Borderline**, committed before the run (R&D's review): a read set's result is borderline
     when the paraphrases' median tail delta lies within 10% of the IQR's width of either edge,
     inside or outside. Only a borderline result may call for a second typicality model
     (Llama-3.1-8B-Instruct). Its 16 GB download needs the maintainer's go first. A pass or fail
     that isn't borderline stands on Phi-3 alone.
5. **On a fail**, the failing set is re-planted. The planter offers 3–5 candidate synonyms per
   word, and the choice is made to match the **error pairs'** delta quantiles, never the
   original's. The third model chooses; Kev never does, so its report stays independent.
6. If both read sets pass, the plan continues from the cache step, unchanged.

`typicality.py` implements the measure and the rule (`readout`, `match`).
