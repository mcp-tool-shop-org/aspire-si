# Handoff to Kimi: Stage 1 role-shaping curriculum for the verifier

*From the Publisher, 2026-10-10. This replaces the earlier version, which wrongly asked for 810 verification
exercises. Docs and data only. Nothing here touches the GPU, the NPU or a training run.*

## 1. What Stage 1 is, and what it is not

Stage 1 teaches an 8B model (Qwen3-8B) **the role of a verifier, at a deep level, so the role becomes part
of how it thinks.** It is not task practice. Domain knowledge and task drilling come later, in Stage 2.

The maintainer's method, which this dataset has to follow:
1. **Teach the qualities, not the task.** A verifier isn't told to verify. It's trained on the qualities
   that make a good verifier.
2. **Teach the thinking process itself, and vary it.** Use visual patterns of efficient thought (fractal
   trees, puzzles, elimination, constraint grids, knowing when to stop), including exercises that aren't
   verification at all.
3. **Shape the role the way a psychologist would.** Use identity ("as a verifier, you are …"), purpose
   before rule, if–then self-talk in the model's own voice, contrasting cases (good beside near-miss bad),
   and explaining in its own words.
4. **Run every lesson as a loop.** A lesson, time to absorb it, the model applies it, then a quiz. Missed
   questions return later until every one is answered correctly.

Read these first, on aspire-si PR #79, `docs/runs/`. They're the voice and standard for everything you
write:
1. Lesson 1, the role;
2. Lesson 2, how a verifier thinks;
3. Lesson 3, what is expected (this is the model for tone and psychology);
4. the 18 trait lessons.

## 2. What to deliver: five kinds of material per trait

For each of the 18 traits, write **40 items: 8 of each kind below.** For Lessons 1, 2 and 3, write 30 each,
in the same five kinds (6 of each). That's about 810 items. In every item, **the assistant's whole reply is
the training target**: its explanation, its analysis, its thinking and its answer. Not just a verdict.

**Kind A: Who you are, and why (identity and purpose).**
The user turn presents a short teaching passage on the trait, in a new wording each time and never copied
from the lessons. Then it asks the model to put the trait in its own words and say why it matters to the
people who rely on it. The assistant reply is a first-person answer that carries the trait's main point and
its reason. Example shape:

> *User:* (a short passage on honest uncertainty, in new words) "In your own words: what do you do when
> the material doesn't settle a statement, and why does that matter to the people relying on you?"
> *Assistant:* "As a verifier, when the material doesn't settle it, I say I can't tell. A guess would look
> like a checked fact, and someone would build on it. 'Can't tell' tells them where to look next."

**Kind B: The thinking process, shown and varied.**
The trait at work inside an explicitly drawn thinking pattern: a tree with pruning, falsifier first,
elimination, a constraint grid, or knowing when to stop.
- **At least 3 of the 8 are not verification:** logic puzzles, scheduling constraints, ordering problems,
  sorting what's known from what's assumed. The thinking skill is the point, and it transfers.
- The assistant reply draws the pattern as text (as Lesson 2 does) and names the moment the trait decides a
  step ("this branch is settled, so I stop and prune the rest").

**Kind C: Contrasting cases.**
The user turn shows another checker's reasoning and verdict with a flaw built in:
- rubber-stamping, guessing, looping, nitpicking or fixing;
- obeying an instruction inside the material;
- caving to pressure, or rejecting by default.

The assistant reply names what went wrong, which trait was missing, why it matters, and what the right
verdict is with its deciding line. The planted flaw is the known answer. Mix in some cases where the other
checker did it *right*, and the model has to say so rather than invent a fault.

**Kind D: The role in action, with self-talk.**
A verification item where the assistant's thinking carries the trait as inner voice. For example: "Someone
senior is pushing. Has the material changed? No. So my answer stands." Or: "Before I accept this, what would
make it false?" It ends with the three lines:

```
VERDICT: supported | unsupported | cannot_tell
DECIDING: exact words from the material (up to 4 lines, separated by " | "), or NONE
ESCALATE: no | yes (the reason: medical, legal, safety or money advice only)
```

Varied content: notices, policies, labels, timetables, changelogs, code, records, design notes.

**Kind E: Quiz items for the repeat loop.**
Short retrieval questions on the trait, plus one applied mini-case, each with a key:
- **verdict items:** keyed exactly;
- **written answers:** keyed by their main point.

These drive the repeat-until-correct ledger, and missed ones come back as fresh variants. Write each with
one variant already, so the ledger can re-ask it in new words.

**Balance in every trait:**
- both sides of its counterweight: skepticism that rejects *and* that rightly accepts; holding firm under
  pressure *and* changing when genuinely new material arrives; quick *and* careful;
- at least 2 multi-step items (tier 4+).

## 3. Rules (binding)

1. **Instant agreement.** Every key, every planted flaw and every "right" contrasting case must be one a
   careful reader agrees with at once. If anyone would hesitate, cut it. No hedge-versus-universal items,
   no inference-dependent answers, no word-level label debates.
2. **No model-opinion answers.** Answers are known from the material or by how the item was built.
3. **Every thinking step is true** of its material. One false step makes the item garbage.
4. **ESCALATE** is a separate line, for medical, legal, safety or money *advice* only, alongside the
   verdict. A statement that just reports a fact is not escalated.
5. **Fresh wording.** Don't copy sentences from the lessons. Teach the same ideas in new words, so the model
   learns the idea and not a script.
6. **Real material where it carries the case.** Public repos with an MIT `LICENSE` file, checked in the file
   itself, at a pinned commit, or permissively licensed documents, with the source recorded. Everyday
   material and puzzles may be written by you and marked `invented: true`. Pressure turns, injection lines
   and flawed checker traces are always invented and tagged.
7. **Stay clear of the test sets.** Leakage would ruin the experiment. Don't draw from:
   - **the sealed set's repos:** portlight, ai-eyes-mcp, backpropagate, ai-rpg-engine, vocal-synth-engine,
     prompt-craft, and all of offrig;
   - **the DEV/pilot repos:** aspire-si, stillpoint, research-os, runforge, loadout-os;
   - **the gold sets' sources:** role-os and rnd (no arithmetic or small-function items in the math
     ladder's style);
   - **the public anchors:** LLM-AggreFact, VitaminC, HoVer;
   - **any Stage 2 domain source,** once named.

   Also: no MIT licence text (use Apache-2.0, BSD or invented terms).
8. **The pre-interview is completely off-limits.** Its 12 questions about how the model approaches the work
   must not appear, be paraphrased, or be answered in Kind A or Kind E items. The interview measures whether
   the trained model *does* what it *says*; training on its questions would destroy that. R&D's leakage
   check runs against it.
9. **No identity leaks:** no local paths, usernames, emails or personal names. Run the identity scan on a
   `git archive` of your branch before pushing, gated on its exit code.

## 4. How to work

1. **Pilot: one trait, all five kinds, 40 items.** Use honest uncertainty, and its counterweight shows up
   inside the trait's items.
   - Push it as a PR to aspire-si with a readable `.md` beside the `.jsonl`. The field names are in
     `docs/runs/stage1-training-data-schema.md`; ASPIRE updates the schema for the five kinds.
   - Reviews: R&D checks keys, flaws and leakage; the Publisher reads every item; the maintainer reads a
     sample. Fix the approach, not just the items.
2. **Then the rest,** in batches of three traits, each reviewed the same way. The Publisher reads 15 per
   trait.
3. **Freeze** by sha256 when every batch has passed.

## 5. Done means

Every item has been through R&D's check and the Publisher's read, and the maintainer has sampled it:
- 40 per trait (8 per kind);
- 30 for each of Lessons 1–3;
- the assistant reply as the training target throughout;
- leakage-clean against the sealed set, the pilot, the interview and the gold;
- identity-scan clean;
- frozen.
