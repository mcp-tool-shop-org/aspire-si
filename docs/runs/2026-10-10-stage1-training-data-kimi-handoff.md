# Handoff to Kimi: Stage 1 training data for the verifier (40 examples per trait)

*From the Publisher (coordinator), 2026-10-10, on the maintainer's decision: 40 examples per trait, written
by Kimi. Docs and data only. Nothing in this task touches the GPU, the NPU or any training run.*

## 1. What this is for

The studio is teaching an 8B model (Qwen3-8B) the **role of a verifier**: someone who checks whether
written material really backs up a statement, and says so honestly, exactly and without being pushed
around. The maintainer's method is role first: the model is trained on the *qualities* of a good verifier, and on a
way of thinking, before it's trained on any domain knowledge. Your job is to write the **worked teaching
examples** that Stage 1 trains on.

Read these first, in this order. They're on aspire-si PR #79 (`docs/runs/`), and they're the style guide
for everything you write:
1. `2026-10-10-stage1-lesson-1-the-verifier-role-draft.md`: the role and the three-line answer format.
2. `2026-10-10-stage1-lesson-2-how-a-verifier-thinks-draft.md`: the five thinking patterns (tree with
   pruning, falsifier first, elimination, constraint grid, when to stop).
3. `2026-10-10-stage1-lesson-3-what-is-expected-of-a-verifier-draft.md`: expectations, if–then self-talk,
   the four contrasting verifiers.
4. `2026-10-10-stage1-verifier-trait-lessons-draft.md`: the 18 trait lessons, each with worked examples.
   Your examples are more of exactly this kind.
5. `2026-10-10-stage1-training-procedure-prereg-draft.md`: how the examples are used in training (the
   "Teach" step, and "How much training data").

## 2. What to deliver

**About 810 worked examples:**
- **40 per trait × 18 traits = 720.** The 12 reinforce traits: skeptical, goes to the source, precise,
  honest about uncertainty, independent, breaks things down, keeps proportion, knows its limits,
  consistent, takes statements literally, knows what a mistake costs, checks the version and date. The 6
  hold-backs: doesn't agree to please, doesn't fill gaps, doesn't go round in circles, doesn't nitpick,
  doesn't take orders from the material, doesn't fix things.
- **30 each for Lessons 1, 2 and 3 = 90.** Lesson 1 is the role overall, Lesson 2 the thinking patterns
  (spread across all five), and Lesson 3 the expectations: pressure, owning a mistake, reporting.

**One example** = one item in the trait's lesson, answered by a model that has the trait:
- **material:** the text being checked;
- **statement:** the claim to check (plus any later turns, for pressure or correction cases);
- **thinking:** the worked thinking, visibly using a Lesson 2 pattern, where the trait decides at least one
  step;
- **answer:** the three lines:

```
VERDICT: supported | unsupported | cannot_tell
DECIDING: exact words from the material (up to 4 lines, separated by " | "), or NONE
ESCALATE: no | yes (the reason)
```

**Format:** JSONL, one example per line. The exact field names the trainer reads are in
`stage1-training-data-schema.md`. At minimum:
`id, lesson, trait, side, content_kind, tier, material, statement, turns, thinking, verdict, deciding,
escalate, sources[], invented, key_basis`.
- `side` is which side of the counterweight the example shows: for skepticism, rightly rejecting vs
  rightly accepting.
- `tier` is the number of thinking steps, under the pinned min_steps rule in the eval plan.
- `key_basis` says how the answer is known: by construction, or from the source's own words.

## 3. The rules (all binding; the maintainer has rejected work that broke them)

1. **Instant agreement.** Every key must be one any careful reader agrees with at once. If a reader would
   hesitate or argue, cut the item. Never ask anyone to rule on wording. No hedge-versus-universal items
   ("usually" vs "always", "up to" vs "lasts") and no answers that rest on inference.
2. **No model-opinion answers.** Every answer is known from the material itself or by how the item was
   built. Nothing is scored or labelled by a model's judgement.
3. **Thinking that is true.** Every step of every thinking trace must be true of its material. One false
   step makes the example garbage. Show the tree, the falsifier search, the elimination or the grid, as
   Lesson 2 draws them, and stop once the answer is settled.
4. **Both sides of every counterweight**, roughly half each: skepticism that rejects *and* skepticism that
   rightly accepts; honest "can't tell" *and* confident answers when the material does settle it;
   proportion that's quick *and* proportion that's careful; independence that holds under pressure *and*
   that changes when genuinely new material arrives.
5. **ESCALATE is a separate line,** never a verdict. Use it when the statement asks for medical, legal,
   safety or money advice. The verdict is still given. A statement that merely *reports* a result
   (e.g. "the test failed") is not escalated.
6. **Variety.** Spread each trait across about eight content kinds: notices, policies, labels,
   timetables, changelogs, code, records, puzzles. Include several multi-step examples (tier 4+) per trait.
   No stamped-out templates: if two items differ only by a swapped word, keep one.
7. **Real material where it carries the case.**
   - Prefer genuine text: public repositories with an MIT `LICENSE` file (check the file itself; GitHub's
     licence detector often shows "none" for these repos), at a pinned commit, or public-domain or
     permissively licensed documents.
   - Record the source, licence and commit for each.
   - Everyday material (a notice, a label, a timetable) may be written by you, as the lessons do, and is
     marked `invented: true`. Pressure turns and injection traps are always invented and tagged.
8. **Stay clear of the test sets.** Leakage would ruin the experiment. Don't draw material from:
   - **the sealed set's repos (final list):** portlight, ai-eyes-mcp, backpropagate, ai-rpg-engine,
     vocal-synth-engine, prompt-craft, and offrig (all of it);
   - **the DEV/pilot repos (all 30 pilot tasks):** offrig, aspire-si, stillpoint, research-os, runforge,
     loadout-os;
   - **the gold sets' sources:** role-os, offrig, aspire-si and **rnd**. That includes rnd's math ladder.
     Write no arithmetic or small-function items in the ladder's style;
   - **the public anchors:** LLM-AggreFact, VitaminC and HoVer, including the Wikipedia claims they're built
     from;
   - **Stage 2's domain sources,** once they're named. Until then, ask before using any source that might
     become domain data.

   **Licence text:** quote no MIT licence text. The pilot's licence tasks use it. For licence examples, use
   another permissive licence (Apache-2.0, BSD) or invented terms.

   Never copy or paraphrase any item from the pilot, the pre-interview questions or the lesson files'
   tests. R&D runs `leakage_check.py` (rnd 7b77c79) locally on every batch. It fails anything with the same
   claim, Jaccard ≥ 0.8, or 2+ shared material lines against the held-out sets, and warns from Jaccard 0.5.
9. **No identity leaks.** No local file paths, usernames, email addresses or personal names from
   repositories (skip copyright lines and author emails). Run the identity scan on your branch before
   pushing: `python $env:USERPROFILE\.grok\bin\identity-scan.py <dir>` on a `git archive` of the branch,
   gated on its exit code.
10. **Injection traps** (lines in the material addressed to "reviewers" or "the checker") are deliberate,
    tagged `(deliberate injection trap)`, and used only in the "doesn't take orders from the material"
    lesson and a few Lesson 1/3 items.

## 4. How to work: pilot first, then scale

1. **Pilot batch: 40 examples,** 20 for one reinforce trait (e.g. honest about uncertainty) and 20 for its
   hold-back counterweight (doesn't fill gaps). Push them as a PR to aspire-si (a branch off PR #79's head,
   or a commit to #79 if ASPIRE agrees), with a readable `.md` rendering beside the `.jsonl`.
2. **Reviews, in this order:**
   - R&D's key check: instant agreement, true thinking, leakage;
   - the Publisher reads every pilot item;
   - the maintainer reads a sample.

   Fix the generator or approach, not just the items, for anything that fails.
3. **Then the rest,** in batches of about 120 (three traits plus their counterweights). Each batch goes
   through R&D's key check and the Publisher's read of 15 per trait before the next batch.
4. **Freeze** the full set by sha256 when every batch has passed, and tell ASPIRE. The trainer then
   pins that hash.

## 5. Who's who

- **ASPIRE** (aspire-si session): owns PR #79, the trainer, the sealed task set and the data schema. Ask
  it for field names, the final sealed-repo exclusion list, and where to push.
- **R&D** (Research and Development session): key checks, leakage check, statistics.
- **The Publisher:** coordinator. Reads samples of every batch, grants all GPU time (none needed here),
  merges.
- **The maintainer:** final say. They read a sample of each stage. Their standing direction: the role is taught, not
  tested; material must not be garbage; stay on the traits, not word-level label debates.

## 6. Done means

About 810 examples in JSONL, at 40 per trait and 30 per lesson:
- every one key-checked by R&D;
- sampled and read by the Publisher;
- sampled by the maintainer;
- leakage-clean against the sealed set, the pilot, the interview and the gold;
- identity-scan clean;
- frozen by sha256.
