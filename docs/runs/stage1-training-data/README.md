# Stage 1 training data (the verifier role-shaping curriculum)

Worked teaching items for Stage 1 of the [training procedure draft](../2026-10-10-stage1-training-procedure-prereg-draft.md),
per the [handoff of 2026-10-10](../2026-10-10-stage1-training-data-kimi-handoff.md) (the role-shaping
replacement: five kinds per trait, the whole assistant reply as the training target) and the
[trainer's schema](../stage1-training-data-schema.md). Docs and data only: nothing here touches the GPU, the
NPU or any training run.

**This directory now holds the pilot:** `pilot-t04.jsonl` — the pilot defined by handoff §4.1: **one trait,
all five kinds, 40 items**, for trait 4, *honest about uncertainty* (its counterweight shows up inside the
trait's items). Beside it: `pilot-t04.md`, the readable rendering, and `checks.py`, which gates the batch on
ASPIRE's own validator (`stage1-train/data.py`) and regenerates the `.md` from the `.jsonl`.

## The batch at a glance (as printed by `checks.py`)

- **Kinds: 8 × 5.** A identity and purpose (own-words, first person) · B thinking patterns (4 of the 8 are
  non-verification: a constraint-grid seating puzzle, an underdetermined ordering, an elimination, a
  known-vs-assumed sort) · C contrasting cases (4 planted flaws — guess ×2, reject_by_default, loop — and 4
  where the other checker was right and the model confirms it rather than inventing a fault) · D the role in
  action (short traces with first-person if–then self-talk) · E quiz items (6 written + 2 verdict
  mini-cases), each with one pre-written variant for the ledger.
- **Sides: 21 / 19.** `honest cannot_tell` vs `confident when settled`, balanced within each kind (A 4/4,
  B 4/4, C 4/4, D 4/4, E 5/3).
- **Tiers:** 1 × 14 (the A and written-E items), 3 × 13, 4 × 9, 5 × 4 — 13 multi-step items, where the
  handoff asks for at least 2.
- **Content spread:** notice, policy, label, timetable, changelog, code, record, puzzle (all eight core
  kinds) plus `teaching` for A and written-E items.
- **One real source.** T04-D-04 quotes rig-bridge's `CHANGELOG.md` (v1.0.2 entry) at a pinned commit, MIT
  `LICENSE` verified by its first line — the schema's `sources[]` shape exercised end to end. Everything else
  is invented everyday material (`invented: true`, `key_basis: construction`), as the lessons do. rig-bridge
  is outside every exclusion on the final list (sealed / DEV / gold / public anchors); Stage 2's domain
  sources aren't named yet, so if the Publisher would rather hold all real sources for batch 2, swapping this
  one item is a two-minute job.

## Checks run on this batch (all clean at PR time)

```sh
# ASPIRE's validator (schema, reply-key match, deciding substrings) + leakage guard
# (pilot/review task sets, and the 12 pre-interview questions at the Jaccard-0.5 bar for kinds A and E)
python checks.py pilot-t04.jsonl --render

# R&D's leakage_check.py (rnd 7b77c79): gold corpora + --against the DEV task sets and the pre-interview
python "E:\AI\Research and Development\experiments\verifier-gold\leakage_check.py" pilot-t04.jsonl \
    --against ..\stage1-taskset\taskset-pilot-20.jsonl \
    --against ..\stage1-taskset\taskset-review-10.jsonl \
    --against ..\2026-10-10-stage1-pre-interview-draft.md
```

Both report 0 failures / 0 warnings. The identity scan (`identity-scan.py` on a `git archive` of the branch,
gated on its exit code) is run before push. The sealed-set leg of the leakage check is ASPIRE's to run with
the sealed file.

**Known upstream quirk for ASPIRE:** `stage1-train/data.py` inserts `stage1_eval` into `sys.path`, but the
eval directory on disk is `stage1-eval` — the import of `harness` fails as committed (verified: pytest
collection errors with `ModuleNotFoundError: No module named 'harness'`). `checks.py` shims the correct path
so the validator runs; a one-character fix at the source (or a directory rename) will make the shim
unnecessary.

## Conventions this file follows

- The conversation lives in `turns`; the first user turn carries the material and the statement, exactly as
  the model will see them. `material` is also held top-level where the item checks a text (always in D).
- C items put the other checker's trace and verdict in the user turn; the planted flaw is named in
  `planted_flaw`, and the item's own keys are the right answer. Checker traces are invented, as required.
- E `variant` is `{user, key}` with `key` mirroring the quiz type (`verdict` → `verdict`/`deciding`/
  `escalate`; `written` → `key_points`).
- Sizes are tiny next to the limits (target ≤ 3,000 tokens, sequence ≤ 4,096); `checks.py` prints a
  conservative per-item estimate and would warn on anything near the bound.
- No item asks, paraphrases or answers any of the pre-interview's 12 questions — enforced by the leakage
  run above at the kinds-A/E bar of 0.5, and by construction of the prompts.

## Reviews requested (handoff §4.1)

1. R&D: key check (instant agreement of keys, truth of every thinking step, the planted flaws, leakage);
2. the Publisher reads all 40 items;
3. the maintainer reads a sample.

Anything that fails gets fixed in the approach (item patterns in this README, `checks.py`, the kind
templates), not just in the item, before the batches of three traits begin.
