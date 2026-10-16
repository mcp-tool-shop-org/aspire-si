# DRAFT: Stage 1 apply items, the schema (run sheet B6)

**Status: agreed between ASPIRE and the Dataset Specialist, and approved by R&D (2026-10-15) with two additions,
now written in (the STaR rows, and the `item_trained` flag).** role-set writes the items in the same pass as
batches 7–8 (R&D's Q4), R&D key-checks them before the freeze, and ASPIRE builds the apply sitting against
synthetic fixtures in this format.

## What apply is for

The approved training procedure's apply step, with R&D's rulings and amendment A of 2026-10-15:
- **After each session's training,** the model answers 10 never-trained items per lesson, in both arms:
  - 5 kind C and 5 kind D;
  - 1 sample each, at the card's settings;
  - on the batched path (B4).
- **Their stuck readings** go to `rehearsal` receipts. They're the C and D cells of round 2's pooled steering.
  Each arm gives 5 per (lesson, kind), so the pool gives 10, which clears n ≥ 8.
- **STaR keeps** the answers that pass R&D's keep rule:
  - all three lines pass the pinned scorer;
  - the lesson's pass rule is met;
  - the trace isn't stuck under `716dc251`.

  The kept answers are matched across the arms and capped at 8 per (lesson, kind) (amendment A). They enter
  training only from session 2.

## The format

1. **The training schema, unchanged** ([stage1-training-data-schema.md](stage1-training-data-schema.md)), kinds
   C and D only, so the same gates check every item. The fields are `id, lesson, trait, kind, side,
   content_kind, tier, turns, material, sources, invented, key_basis, tags`, plus the kind's keys. **Added:**
   `"split": "apply"`.
2. **Ids:** `<lesson>-<kind>-AP-<NN>`, for example `T05-C-AP-01`.
   - The kind stays the second dash-part, where role-set's readout reads it (the Dataset Specialist's amendment).
   - `-AP-` marks the split. Nothing on the aspire-si side parses an id, which was checked.
   - There are 10 per lesson (01–05 C, 01–05 D) over 21 lessons: 210 items.
3. **One turn.**
   - The model is asked the first user turn as written: material and statement for D; for C, material, statement,
     and another checker's reasoning and verdict.
   - The item's own `thinking` and `reply` are a **reference answer**, key-checked and never trained.
   - The stuck rule needs the reference's thinking length, for clause C's "2 × the item's own target".
4. **Keys:**
   - **D:** `verdict`, `deciding`, `escalate`.
   - **C:** the same three, which are the right answer, plus `planted_flaw` (or `none`).
   - **T09 items, both kinds:** also `order_free` (true or false). It's true when the material's line order
     carries no meaning, so reversing it can't change the right answer. Changelogs, tables and ordered steps are
     false. Only an `order_free: true` item gets R9's reversed-material twin; the others are held (Argon's review,
     settled by R&D, 2026-10-15).
   - **Decided T01, T04 and T14 items, both kinds:** also `silent_without_deciding` (true or false). It's true
     when removing the deciding lines leaves the material silent on the statement, so the right answer becomes
     cannot_tell. A buried exception, or another line that still settles it, is false. Only a `true` item gets
     the deciding-lines-removed twin, which must answer cannot_tell; the others are held (Argon's finding 6.2,
     settled by R&D, 2026-10-15).
   - **T10 items, both kinds:** also `r10_twin: {statement, verdict}`, and exactly one `Statement:` line in the
     user turn, which the twin replaces (Argon's finding 6.4). This is a strengthened or weakened claim on
     the same material, keyed with a verdict that differs from the item's. It's authored by role-set and
     key-checked by R&D with the item, and it's the twin R10's pass rule needs (R&D on #128, 2026-10-15).
   - **Right** means the three lines pass the pinned scorer against the key: the verdict, DECIDING verbatim (the
     scorer's norm) and covering the key, and ESCALATE as the harness grades it.
   - For C, naming the flaw isn't scored mechanically, and that's stated in every readout.
5. **Kept out of training, the reserve and steering material:**
   - **The trainer can't load them:** the items live in `docs/runs/stage1-apply/`, never in
     `stage1-training-data/`, and train.py refuses any data row with `"split": "apply"` or `-AP-` in its id.
   - **No overlap with the training set**, checked by role-set against the whole training set:
     - no reused material, using `check_against` with the training rows as accepted;
     - no 8-word span of a user turn matching a trained turn (the quiz's spent rule).
   - **No leakage:** the leakage check runs against the sealed set, DEV and the interview, as for training.
   - Only the stuck readings and STaR's kept answers flow on. The items themselves never enter a pool steering
     draws from.
6. **One set serves both arms.** The user turns are role-neutral (the gates' `role_words_in_prompt`), so NO-ROLE
   needs no derived copy.
7. **Process:**
   - role-set writes the items: an apply plan of 5 C and 5 D per lesson, the ids, the split, and the output
     directory;
   - R&D key-checks them with the instant-agreement rule before the freeze;
   - they're delivered by PR, as the batches are, and pinned by sha256 in FROZEN.md.

## Kept answers become STaR rows (R&D's addition 1)

train.py refuses `split: apply` and `-AP-` rows anywhere, so a kept answer is re-labelled, never trained as an
apply row.
- **The row:** each kept answer, after matching across the arms and capping (amendment A), is written as a new row
  in the session's STaR file:
  - `"split": "star"`;
  - an id `<lesson>-<kind>-ST-<session>-<NN>`;
  - `star_of`: the apply item's id.

  The user turn is the item's. The thinking and the reply are the model's own.
- **Loading:** train.py loads star rows only from that matched and capped file, which is pinned by sha256 in the
  session's manifest. It still refuses `split: apply` and `-AP-` rows everywhere.
- **Near-duplicates:** the drop (8-word spans, amendment A) runs on these rows.

## The `item_trained` flag (R&D's addition 2)

From session 2, an apply item may have its own kept answer in training, so its later readings are no longer of a
never-trained item.
- **Recorded:** every rehearsal row carries `item_trained` (true or false), set from the STaR files of earlier
  sessions.
- **Reported:** each (lesson, kind) cell's steering readings are split by it.
- **Steering:** the pooled steering keeps using every reading, as registered.
- **A gap:** if the split shows one that matters, it goes to the maintainer as its own question, not as a change
  made here.

## The apply sitting (to build after R&D's ruling)

- **What it asks:** each item's first user turn, through `batched.make_generate_many`, one sample, in both arms.
  The per-item seed is (item id, 0).
- **What it scores:** the three lines with the pinned scorer, against the key; plus `stuck_fields` (the frozen
  rule, the reference thinking as the target).
- **Receipts:** a `rehearsal` row per item, with `question_id` (the item id), lesson, kind, tier, right, the
  stuck fields, `item_trained` and the answer.
- **STaR's keep decision:** recorded per item (kept, or refused with the reason). Matching and capping happen
  across the arms after both have sat.
- **Readout:** per (lesson, kind), split by `item_trained`, with the C items' flaw-naming disclosure.
- **STaR file:** the kept answers, after matching and capping across the arms, written as `split: star` rows
  (above). The file's sha256 goes into the next session's manifest.
- **Fixtures for its tests:** a few items in this format. They're synthetic, and none comes from the real set.

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | The items are pinned by sha256 in FROZEN.md, and the sitting records its path and seeds. |
| ANDON_AUTHORITY | 2 | train.py refuses apply rows; the gates and the leakage check refuse a bad item. |
| NAMED_COMPENSATORS | 2 | Delivery is by PR (closed unmerged to undo). Receipts are voided by round in role-set (`role-set void`). |
| DECOMPOSE_BY_SECRETS | 2 | role-set writes the items, R&D checks them, and aspire-si asks and scores them. |
| UNCERTAINTY_GATED_HUMANS | 2 | R&D rules on this note, and key-checks every item. |
| EXTERNAL_VERIFIER | n/a | No specialized claims. |
