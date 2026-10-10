# Handoff to Grok: Stage 1 training data, from here to freeze

*From the Publisher, 2026-10-10. Grok takes over the Stage 1 dataset from Kimi, starting now. Docs and data
only. Nothing here touches the 5090, the NPU or any training run. This doesn't change your switchyard work;
keep the two lanes on separate branches and in separate PRs.*

## 0. Read first, in this order (aspire-si, branch `verifier-distill-prereg-draft`, `docs/runs/`)

1. **`2026-10-10-stage1-training-data-kimi-handoff.md`.** This is the binding spec. Its name says Kimi; it now
   applies to you word for word. Read the per-kind required-parts table and the content rules twice.
2. **`stage1-training-data-schema.md`.** The field names and the turns-based shape.
3. **The lessons:**
   - Lesson 1, the role;
   - Lesson 2, how a verifier thinks;
   - Lesson 3, what is expected (the model for tone and psychology);
   - `2026-10-10-stage1-verifier-trait-lessons-draft.md`, the 18 traits.
4. **`stage1-train/data.py`.** The validator. It refuses an item; it never truncates or repairs one.
5. **The existing data:**
   - `stage1-training-data/` (pilot-t04 and its README);
   - batch 2 on branch `stage1-data-batch2` (PR #82);
   - the pilot's sources fix on `stage1-pilot-sources` (PR #83).
6. **R&D's review comments on #81, #82 and #83.** They're the record of every mistake made so far. Don't
   repeat any of them.

## 1. What Stage 1 is (so the material comes out right)

Stage 1 teaches Qwen3-8B **the role of a verifier, deeply enough that the role becomes part of how it
thinks.** It isn't task practice; domain knowledge comes in Stage 2. The maintainer's method:

- **Teach the qualities, not the task.** A verifier isn't told to verify. It's trained on the qualities that
  make a good verifier.
- **Teach the thinking process itself, and vary it.** Use drawn patterns (trees with pruning, falsifier
  first, elimination, constraint grids, knowing when to stop), including puzzles that aren't verification at
  all.
- **Shape the role the way a psychologist would.** Use identity ("As a verifier, you …"), purpose before
  rule, if–then self-talk in the model's own voice, contrasting cases, and explaining in its own words.
- **Run each lesson as a loop.** Lesson, then absorb, then apply, then quiz. Missed questions come back as
  fresh variants until every one is right.
- **Not a GitHub bot, and not a test of psychic ability.** Every answer is known from the material or from
  how the item was built.

**In every item, the assistant's whole reply is the training target.** One false step in a thinking trace
teaches the model something false. That's why R&D's checks are line by line, and why a one-line slip is
still a blocker.

## 2. Where things stand

| Item | State | Who acts next |
|---|---|---|
| Pilot, T04 (#81) | Merged, 631b6c0 | — |
| Pilot sources fix (#83, `stage1-pilot-sources` at 07c429d) | Open. The Publisher re-ran it: validator 40/40, identity scan clean. Its replies and variants changed, so R&D is checking the changed items | R&D, then the Publisher merges |
| Batch 2: T14, T01, T16 (#82, `stage1-data-batch2` at 4c06089) | Open. Validator 40/40 per trait, leakage 0. **17 fixes remain** (R&D's latest comment on #82) | **You** (task 1) |
| Batch 3: T05, T13, T17 | Not started. T17 carries injection traps | **You** (task 2), after #82's fixes |
| Batches 4–8 | Not started | You, in order |
| Freeze | — | When every batch has passed |

## 3. Your tasks, in order

### Task 1: the 17 batch-2 fixes on #82

Push to `stage1-data-batch2`; don't open a new PR. Fix **exactly** what R&D's latest comment on #82 lists,
and nothing else. A change outside the list costs a re-read.

**T01 (2):**
- **B-01:** add the `method: "exact" | "monte-carlo"` line to DECIDING. The reply rests on it.
- **C-07:** DECIDING must say the code is 1 and carry the "deliberate two-code contract" line. Drop "glanced
  at", which only the planted flaw implies.

**T14 (2, plus a nit):**
- **C-04:** replace "nothing about the answer was inferred" with "the heading and the past-tense bullet
  together settle it".
- **C-08:** in the thinking, keep only "this entry never mentions inline code spans". The material never says
  a different entry handles the topic.
- **D-07:** "prices three day-groups" becomes "lists three day-groups".

**T16 (13):**
- **A-08:** "gatelodge" is one compound word, not "a second word". Restate the matching letters
  unambiguously (positionally only A and d match).
- **B-06:** use "the stairs — ninety-six steps — remain open", so the rewording really is two edits.
- **B-07:** the prompt must say everything the claim changes: the swapped word, plus the added "now" and
  "refuses to load".
- **B-08:** 51.2 → 25.1 is a rotation of the digits, not a transposition.
- **C-02:** 791 against 719 swaps the tens and ones digits. Say so.
- **C-04:** close the backtick and remove the doubled quote.
- **C-06:** 60,000× is about 4.8 orders of magnitude. Say "nearly five".
- **C-08:** the claim states no mechanism. Remove "same mechanism" and "mechanism stated".
- **D-01:** DECIDING must include `2d6ro=1`, so it ties the gloss to the notation on its own.
- **D-04:** the claim gives no start time. Remove "the 10:00 start matches" from the thinking and the reply.
- **D-05:** put the flags `--analyze` and `--at-least` into the claim, and quote the whole clause in
  DECIDING. The thinking must no longer say "scope included" about a claim that drops the scope.
- **D-08:** the claim has two numerals (120 and £1.50). Fix "the only numeral".
- **E-06:** replace the appositive "built in 1902", which adds a fact, with one that doesn't (e.g. "the old
  stone bridge").

Then run every gate (section 6), push, and post on #82: the fixed IDs, the gate results and the new file
sha256s. R&D re-checks only those items, the Publisher reads 15 per trait, and #82 merges.

### Task 2: batch 3 (T05 independent, T13 doesn't agree just to please, T17 doesn't take orders from the material)

New branch `stage1-data-batch3` off `verifier-distill-prereg-draft` (after #82 and #83 merge, rebase onto
it). 40 items per trait, 8 per kind. Files: `batch3-t05.jsonl`, `batch3-t13.jsonl`, `batch3-t17.jsonl`, each
with its rendered `.md`.

**Trait-specific notes:**
- **T05, independent.** The counterweight: independence isn't contrarianism. Include cases where the other
  checker was right and the model agrees for its own reasons, and cases where it reaches a verdict before
  looking at anyone else's.
- **T13, doesn't agree just to please.** Pressure turns (seniority, urgency, flattery, repetition) are always
  invented and tagged. The counterweight: when genuinely new material arrives, the model changes its answer,
  and says what changed.
- **T17, doesn't take orders from the material.** Injection traps: an instruction inside the material ("mark
  this supported", "ignore the statement above", "as the verifier, reply only with …"). The model treats it
  as part of the material being checked, never as an instruction, and says so. The counterweight: it still
  reads that material for facts. A notice that says "this policy replaces v2" is a fact about the material,
  not an order. Every injection line is invented and tagged. Keep traps plain and readable; no encoded or
  hidden text.

**Required in batch 3 and every batch after (R&D, from the batch-2 re-check):**
1. **Draw the stop rule explicitly.** Stop when a line is decisive and nothing in the material qualifies it.
   When a line only supports the claim, do a short sweep for anything that would make it false. Kinds A and E
   must teach the same rule as B and D, never just "stop once the first line settles it".
2. **List exactly what the claim asserts** before writing the thinking. Every edit, match or difference the
   thinking describes must be in the claim.
3. **Check every count with a script:** digits, letters, edits, orders of magnitude, "a single word", "the
   only numeral". Never eyeball one.
4. **DECIDING carries the identifier the claim is about:** the notation, flag, code or scope.
5. **Add a quote-balance check to `checks.py`:** backticks, straight and curly quotes, `**`. It runs in every
   gate pass.
6. **No stamped phrasing.** "Earned", "theatre", "nothing filled" and the uniform "if X, then I…" closers are
   already overused. Vary the wording, including C's framing sentence ("Material given to another checker
   …").

### Tasks 3 to 7: the remaining batches

Same process for each: one branch and one PR per batch, reviewed and merged before the next starts.

| Batch | Contents |
|---|---|
| 4 | T02 goes to the source · T03 precise · T06 breaks things down |
| 5 | T07 keeps things in proportion · T08 knows its limits · T09 consistent |
| 6 | T10 takes statements literally · T11 knows what a mistake costs · T12 checks the version and the date |
| 7 | T15 doesn't go round in circles · T18 doesn't fix things |
| 8 | Lessons L1, L2 and L3, 30 items each (6 per kind) |

That makes 18 × 40 + 3 × 30 = 810 items in all.

### Task 8: freeze

When every batch has merged, open one PR adding `stage1-training-data/FROZEN.md`:
- the sha256 of each file;
- the item counts per lesson and kind;
- the gate results;
- the sources list (real material, repo@commit, licence).

The Publisher records the freeze, and the files don't change after that.

## 4. The rules (binding; from the spec, with what batch 2 taught)

1. **Instant agreement.** Every key, planted flaw and "the other checker was right" case must be one a careful
   reader agrees with at once. If anyone would hesitate, **cut the item; don't argue for it.** No
   hedge-versus-universal items, no inference-dependent answers, no word-level label debates.
2. **Every thinking step is true** of its material.
3. **Per-kind required parts** (the spec's table, enforced by the validator):

   | Kind | User turn | Assistant reply |
   |---|---|---|
   | A | A 2–4 sentence teaching passage in fresh words that names the trait and why it matters, then an own-words question from a new angle | First person, the main point and its reason, no verdict lines |
   | B | A problem with the thinking pattern named; at least 3 of 8 non-verification | The pattern drawn as text, the moment the trait decides a step (varied wording), the answer |
   | C | Material, a statement, and another checker's reasoning and verdict (a planted flaw, or correct) | Right or wrong; the flaw in the keyed flaw's terms (or confirmation it was right); the trait; the three lines |
   | D | Material and a statement (plus pressure or correction turns) | Thinking with first-person if–then self-talk; the three lines |
   | E | A retrieval question or mini-case, plus `variant.user` re-asking **the same point** | A short answer hitting every key point |

4. **The three lines**, exactly:
   ```
   VERDICT: supported | unsupported | cannot_tell
   DECIDING: exact words from the material (up to 4 lines, separated by " | "), or NONE
   ESCALATE: no | yes (the reason: medical, legal, safety or money advice only)
   ```
   - DECIDING settles **every clause** of the statement on its own, verbatim, with markup such as `**` kept.
   - `cannot_tell` exactly when DECIDING is NONE.
   - ESCALATE is for *advice* only. A statement that just reports a fact is not escalated.
5. **"Accepts the equivalent" items** change only notation, units, synonyms or numerals. They never add a fact.
6. **No time words** ("today", "now", "currently") in a statement unless the material dates it. Reword a
   version-anchored claim as "From v2.1.0, …".
7. **Kind C replies judge only the material shown,** and never repeat the flawed checker's claim as fact.
8. **No scope-limited material against a universal claim,** in any trait.
9. **Puzzles are closed:** fully determined, or determined up to an explicit, enumerable set. No probabilistic
   puzzles.
10. **Balance in every trait:** both sides of its counterweight, within each kind; at least 2 multi-step items
    (tier 4+).
11. **Fresh wording.** Never copy sentences from the lessons.
12. **Provenance on every item; no empty `sources`.**
    - Real material: `repo@commit:path#lines` plus the licence's first line, quoted verbatim at that commit.
    - Invented parts: `{"invented": "<what>"}`. Checker traces, pressure turns and injection lines are always
      invented.
    - Use real material where it carries the case. Public repos with a permissive licence (Apache-2.0, BSD,
      MIT) checked in the LICENSE file itself, at a pinned commit.
    - No MIT licence text inside the material (use Apache-2.0, BSD or invented terms).
13. **Stay clear of the test sets.** Leakage ruins the experiment. Never draw from:
    - the sealed set's repos: portlight, ai-eyes-mcp, backpropagate, ai-rpg-engine, vocal-synth-engine,
      prompt-craft, all of offrig;
    - the DEV/pilot repos: aspire-si, stillpoint, research-os, runforge, loadout-os;
    - the gold sets' sources: role-os and rnd (no arithmetic or small-function items in the math ladder's
      style);
    - the public anchors: LLM-AggreFact, VitaminC, HoVer;
    - any Stage 2 domain source, once named.
14. **The pre-interview is off-limits.** Its 12 questions must not appear, be paraphrased or be answered in
    any item.
15. **Length:** the assistant reply at most about 3,000 tokens, the whole item at most 4,096. Split or tighten;
    the validator refuses longer items.
16. **No identity leaks:** no home paths, usernames, emails or personal names, in data, PR text or comments.
    Never name the maintainer in repo text.

## 5. How to work (this is what went wrong before)

- **Build items with a script and check them with a script.** Count, diff and balance mechanically. R&D's
  re-checks found the same categories of error in every round: counts, descriptions of edits, DECIDING scope,
  stamped phrasing. Each of those can be caught before the push.
- **Self-review before every push.** For each item: list what the claim asserts; check that DECIDING settles
  each part on its own; check every number and edit description against the script output; read the
  thinking for any step the material doesn't support.
- **Fix the approach, not just the items.** When R&D flags a category, fix every item in the batch that has
  it, and say so in the PR.
- **If a rule can't be met for an item, cut the item and write a replacement.** Don't bend the rule. If a
  rule looks unworkable for a whole kind, stop and ask the Publisher through the maintainer.
- **Keep the fix scripts** (`_fix_batch*.py`, `_audit_drift.py`) in the directory. R&D re-derives from them.

## 6. Gates (all must pass before every push; report each result in the PR)

Run from `docs/runs/stage1-training-data/`:

1. **The validator, over every data file in the directory** (the pilot included):
   `python checks.py <file>.jsonl --render`. Schema, per-kind parts, drift checks, reply-key match, verbatim
   DECIDING, leakage against the pilot/review task sets and the pre-interview. It regenerates the `.md`.
2. **`_audit_drift.py`** batch-wide, so every failure is listed at once.
3. **`_verify_sources.py`:** every real-source quote verbatim at its pinned commit, DECIDING markup kept.
4. **The quote-balance check** (new, task 2).
5. **R&D's `leakage_check.py`** (the rnd experiments/verifier-gold copy, which reads inside `turns`), against
   the gold corpora plus `--against` the two DEV task sets and the pre-interview draft. 0 failures, 0
   warnings.
6. **ASPIRE's validator tests:** `python -m pytest ../stage1-train` (19/19 at present; the path is relative to
   this directory). The `stage1_eval` import quirk in the pilot README was fixed at d108c54, so collection
   works and `checks.py` no longer needs its shim.
7. **The identity scan** on a `git archive` of the branch head, **gated on its exit code** (no pipe in
   front of the check). RESULT CLEAN or don't push.

The sealed-set leakage leg is ASPIRE's, run with the sealed file. Don't ask for the sealed file.

## 7. Review chain and hand-back

- **One PR per batch** into `verifier-distill-prereg-draft`, with the `.md` renderings beside the `.jsonl`.
- **The PR description:**
  1. what's in it, with counts per kind and side, and tiers;
  2. every gate result;
  3. every deviation from this handoff or the spec, and why;
  4. the real sources used.
- **Reviews:** R&D checks keys, flaws, every thinking step and leakage, then the Publisher reads 15 per trait,
  then the maintainer samples. R&D's re-check of the changed items comes after each fix round.
- **Merges are the Publisher's.** Don't merge your own PR.
- **Messages go through the maintainer.** Post results on the PR; the maintainer relays them.

## 8. Out of scope (ask first)

- Any GPU, NPU or training run.
- Any model call that costs money, and any Ollama Cloud model.
- Editing the lessons, the schema, `data.py`, the trainer or the eval. If one of them blocks you, describe it
  on the PR and ASPIRE fixes it.
- Changing the sealed set, the pre-interview or the DEV task sets.
- Anything public, and any release.

## 9. Standards compliance (this handoff)

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Every real source is pinned to repo@commit with its licence line; each review pins the files' sha256; the freeze pins the whole set. |
| ANDON_AUTHORITY | 2 | The validator refuses a bad item and never repairs it; any gate failure blocks the push; instant-agreement failures are cut. |
| NAMED_COMPENSATORS | 2 | The only irreversible-ish actions are pushes and PRs: close the PR and delete the branch (Grok); a bad merge is reverted by the Publisher. |
| DECOMPOSE_BY_SECRETS | 2 | Data (Grok), the validator and trainer (ASPIRE), key checks (R&D) and merges (the Publisher) are separate owners and files. |
| UNCERTAINTY_GATED_HUMANS | 2 | A contested item is cut, not ruled on; an unworkable rule stops and asks; the maintainer samples each batch. |
| EXTERNAL_VERIFIER | 2 | R&D (Claude, a different family from the Grok generator) checks every item line by line against the material. |


## Amendment 1 (2026-10-10): quoting the deciding lines

*From the Dataset Specialist. Binding from batch 3 onward. Batch 2 (#82) isn't reopened for this.*

**Why.** The untrained model usually gets the verdict right but rarely quotes the deciding lines exactly.
The misses fall into a few habits:
- paraphrasing;
- dropping backticks or other marks;
- quoting part of a table row;
- leaving out the version heading the evidence sits under;
- writing NONE beside real lines;
- quoting the statement instead of the material.

Exact quoting is part of being precise (T03) and of going to the source (T02), so every trait's items should
model it. This comes from aggregate counts only. No test item, test wording or test repo is behind it, and
the exclusion list in §4 rule 13 is unchanged.

**Rules for the three lines (kinds C and D, and kind-E verdict items):**
1. **Copy the material exactly, marks and all:** backticks, `**`, quotes, case and punctuation. Make sure
   plenty of items have code-formatted material (flags, file names, function names, config keys), so the
   model practises keeping the backticks.
2. **A table row is quoted whole,** from its first cell to its last, never as one or two cells.
3. **When a version or date heading is part of the evidence, it's its own DECIDING line,** e.g.
   `## v2.3.0 (2026-04-02) | - The dry run no longer writes the lock file.`
4. **Use multi-line keys where the evidence really spans lines.** In every trait, at least a quarter of the
   verdict-bearing items (supported or unsupported) need two to four DECIDING lines. Separate them with
   " | ", and never end with a trailing separator.
5. **NONE stands alone and only for cannot_tell.** Never NONE beside lines, and never lines for a cannot_tell
   key.
6. **Quote the material, never the statement.**
7. **Show the habit in the self-talk (kind D) and the quizzes (kind E):** "copy the line exactly, marks and
   all", "the row, not the cell", "the heading belongs with the bullet". Vary the wording per the no-stamping
   rule.

**Long material.** Lines over 100 characters are fine and should appear. Keep within the length limits in §4
rule 15.

**Gates.** ASPIRE is adding validator drift checks for rules 2, 4 and 5 and the trailing separator. Until they
land, check them with a script before every push, and report the multi-line share per trait in the PR.


## Amendment 2 (2026-10-10): three habits every batch must teach

*From the Dataset Specialist. Binding from batch 3 onward. Batch 2 (#82) isn't reopened for this.*

**Why.** The untrained model is already cautious: it rarely accepts a false statement. What it lacks are three
distinctions. This comes from aggregate counts only. No test item, test wording or test repo is behind it, and
§4 rules 13 and 14 (the exclusions and the off-limits pre-interview) are unchanged.

**1. Silent is not contradicted.** When the material doesn't mention something, the model mostly answers
"unsupported". It should answer `cannot_tell`.
- In every trait, among the verdict-bearing items (C, D and verdict-type E), **at least 25% are keyed
  `cannot_tell`, and at least 25% are keyed `supported`.** The second floor stops the data from pushing the
  model further toward suspicion.
- **Every trait has at least two contrast pairs:** the same statement checked against material that is
  *silent* on it (`cannot_tell`) and against material that *says otherwise* (`unsupported`). The reply names
  the difference ("the notice never mentions weekends" versus "the notice says closed on Sundays").
  Both items of a contrast pair carry the same `pair` id (a short string), and no other item carries it.
- **Kind C includes a checker who marks a silent case "unsupported"** (flaw `guess` or `reject_by_default`),
  and the reply corrects it to `cannot_tell`.
- The self-talk carries the distinction in varied words: "not mentioned isn't the same as ruled out".

**2. ESCALATE is about advice, not about doubt.** The model treats escalation as what to do when the material
is thin or unclear. That's what `cannot_tell` is for.
- **Every trait has at least one item keyed `ESCALATE: yes (…)`.** The statement gives medical, legal, safety
  or money advice. Across a batch, those items are spread over all three verdicts. Escalation sits beside the
  verdict and doesn't replace it.
- **Every trait has at least one `cannot_tell` item on an everyday topic keyed `ESCALATE: no`,** where the
  reply says why: the material is thin, but nobody is being advised.
- Kind E asks about the difference in fresh words. Never use or paraphrase the pre-interview's questions.

**3. Change your answer for new material, and only for new material.**
- **Every trait has at least two two-turn D items:**
  - one where the second turn brings genuinely new material that changes the verdict, and the reply says
    exactly what changed and quotes the new line;
  - one where the second turn brings only pressure or repetition, and the verdict stands, with the reason.
- Both turns are complete answers with their own three lines.

**Gates.** ASPIRE is adding validator checks for the floors in points 1–3. Until they land, check them with a
script and report the counts per trait in the PR.
