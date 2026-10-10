# Stage 1 training data (the verifier role-shaping curriculum)

Worked teaching items for Stage 1 of the [training procedure draft](../2026-10-10-stage1-training-procedure-prereg-draft.md),
per the [handoff of 2026-10-10](../2026-10-10-stage1-training-data-kimi-handoff.md) (the role-shaping
replacement: five kinds per trait, the whole assistant reply as the training target) and the
[trainer's schema](../stage1-training-data-schema.md). Docs and data only: nothing here touches the GPU, the
NPU or any training run.

**This directory holds the pilot and batch 2:** `pilot-t04.jsonl` — the pilot defined by handoff §4.1: **one trait,
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
- **Tiers:** 1 × 14 (the A and written-E items), 3 × 14, 4 × 8, 5 × 3, 6 × 1 — 12 multi-step items, where
  the handoff asks for at least 2.
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

# R&D's leakage_check.py (rnd v1.1.4.4.10, which reads inside `turns`): gold corpora + --against
# the DEV task sets and the pre-interview
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

**Batch 2 is in:** `batch2-t01.jsonl` (skeptical), `batch2-t14.jsonl` (doesn't fill gaps) and
`batch2-t16.jsonl` (doesn't nitpick) — 40 items each, all five kinds, rendered `.md` beside each `.jsonl`.

## Batch 2 at a glance

- **Kinds:** 8 × 5 per trait. Sides balanced in each: T14 20/20 (refuses the gap / quotes only what is
  there), T01 22/18 (rightly rejects / rightly accepts), T16 22/18 (accepts the equivalent / catches the
  changed fact).
- **Tiers:** every trait carries multi-step items (T14: 5 × tier 4 + 1 × tier 5; T01: 5/1; T16: 3/1),
  with tier-4/5 D items built on longer material and a buried deciding line.
- **Content spread:** all eight core kinds per trait, plus `teaching` for A and written-E items. Each kind
  B keeps at least three non-verification puzzles (constraint grids, ordering, elimination, case splits —
  never arithmetic of the gold math-ladder kind).
- **Real cited text (per the standing rules below): all 8 kind-C materials per trait are real, and half of
  kind D** (4 of 8 per trait): T01 D-01/-02/-05/-08, T14 D-01/-02/-05/-06, T16 D-01/-02/-05/-07. Plus
  scattered B/E items (`T01-B-01`, `T01-E-07`, `T16-B-07`). Every citation is a public mcp-tool-shop-org
  repo at a pinned commit, MIT `LICENSE` verified by its first line, outside every exclusion on the final
  sealed/DEV/gold/anchor list:

  | repo | pinned commit | items |
  |---|---|---|
  | `rig-bridge` | `e91f65b5c12bd070342abdb298b9434c27d4a9b4` | T01-C-04, T01-D-08, T14-C-03, T14-C-06, T14-D-06, T16-C-04, T16-C-05, T16-D-07 |
  | `roll` | `fbec7b5a99b2eefbd01efe431ff1c1dcc07e5a03` | T01-B-01, T01-C-01/-06/-07, T01-D-01, T01-E-07, T14-C-01/-05/-07, T14-D-01, T16-C-01/-06, T16-D-01/-05 |
  | `style-dataset-lab` | `9d91b77432a9950268e3fe6654cbdc8604650e90` | T01-C-02/-05/-08, T01-D-02, T14-C-04, T14-D-02, T16-B-07, T16-C-02/-08, T16-D-02 |
  | `polyglot-mcp` | `25e970d9d598fb840baabaa7c0762b2762f88e0e` | T01-C-03, T01-D-05, T14-C-02/-08, T14-D-05, T16-C-03/-07 |

  Contamination found while harvesting and excluded on sight: `ollama-intern-mcp` (mentions research-os),
  polyglot-mcp `CHANGELOG.md` (tells an ai-rpg-engine story), repo-knowledge `CHANGELOG.md` (names a
  gold-flavoured fixed enum). Citation spans are chosen disjoint across the whole batch — no two items
  quote the same lines.

## Batch 2 tooling

- **`_verify_sources.py`** — proves every `sources[]` citation verbatim against the local clone at the
  pinned commit: material must be a contiguous excerpt of the cited lines (normalised for whitespace and
  `**`/`__`), the commit must be an ancestor of `origin/HEAD`, and duplicate citations across the batch
  are rejected.
- **`_build_t16.py`** — the auditable builder for T16, slicing real spans from scratch dumps of the pinned
  blobs (`_harvest/` at the repo root, untracked; regenerable with `git show <commit>:<path>`).
- All three files pass `checks.py --render` (schema + validator + pilot/review/pre-interview leakage) and
  R&D's `leakage_check.py` (rnd v1.1.4.4.10) at 0 failures / 0 warnings each; `_verify_sources.py` passes
  across the batch.

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
- **The closed-list rule:** no `cannot_tell` item may sit next to a list or table that a reader could take
  as complete — a keyed silence must be unambiguous, with nothing beside it that could be read as the whole
  story (R&D's key check caught two of these; B-06, C-08 and E-07 were rebuilt to keep the silence genuine).
- **No stamped phrases in thinking traces.** B traces name the deciding moment in ordinary words, worded
  differently each time — never a template label like "The trait's moment:" — so the model learns the
  habit, not a slogan.

## Required parts, per kind (binding spec; ASPIRE's validator will enforce it)

- **A** — the user turn is a teaching passage of 2–4 sentences in fresh words that **names the trait**
  and says why it matters, followed by a question from a new angle. The reply is first person, with the
  trait's main point and its reason, and no verdict lines.
- **B** — a named thinking pattern, drawn as text; the deciding moment in ordinary, varied words; at
  least 3 of the 8 per trait are non-verification puzzles.
- **C** — material, a statement, and another checker's reasoning and verdict. The reply judges it,
  **naming the flaw in the keyed flaw's own terms**, states the trait, and gives the three lines.
- **D** — material and a statement. The thinking carries first-person **if–then self-talk**, then the
  reply gives the three lines.
- **E** — a question plus a variant that re-asks the same point in new words.

**Provenance (binding):** real material is listed in `sources` as `repo@commit:path#lines` plus the
`LICENSE` first line. Invented parts are marked separately (`invented_parts` — the checker trace in C,
pressure turns). No empty provenance: `checks.py` warns whenever `invented` and `sources` disagree
(real material must be cited; invented items carry no sources).

## R&D's key check on batch 2 (PR #82) and the rules it adds

`_fix_batch2_r2.py` (committed, assert-guarded) plus `_build_t16.py` carry the round. Standing rules,
applied to every batch from here:

1. **No universal-vs-scope items** in either direction: a universal claim against a scope-limited line
   splits readers between unsupported and cannot_tell. Use plain silence for cannot_tell and explicit
   contradiction for unsupported.
2. **Script-check every count** ("one digit", "three letters") against the actual strings — three T16
   items had them wrong.
3. **Accept-side (nitpick) items change notation, units, synonyms or numerals only.** Any added scope,
   modality or recommendation word belongs in a reject item.
4. **Each DECIDING is checked against every clause of its claim** — keep the material's `**` markers,
   prefer whole clauses to fragments, name the flag/heading the claim points at.
5. **Strip time words** ("now", "until 3.4.0") unless the cited material itself carries the version
   or date.
6. **Kind C replies keep to the shown material** — never assert what only the flawed checker said, and
   don't credit them with "quoting" what they only described.
7. **Vary the stampable shapes:** C openers (not "They (didn't) get it right" 24 times), D self-talk
   intros and branch labels (the if–then habit itself is the spec'd part; its packaging isn't). The
   A opener "As a verifier, …" is the lessons' own form and stays.

## R&D's pilot key check (PR #81) and the fixes applied

`_pilot_fixes.py` (committed, assert-guarded) carries the round: B-06 rebuilt with a genuinely silent rule
(wrong key or hedged grid step), B-01's step count made true (two rules, tier 3), C-08 and E-07 + variant
rebuilt with no closed list beside the claim, A-01 rewritten on an audit/legibility angle away from the
pre-interview, C-01's reply wording aligned with its keyed flaw (`guess`), and the stamped B-trace phrase
dropped from all seven traces that carried it. Round 2 (`_pilot_fixes_r2.py`) rebuilt B-06 again per
R&D's follow-up: the notice now names an eligibility condition ("registered members") that the case
cannot be checked against — nothing says whether contractors are members — instead of leaving eligibility
unnamed entirely. Still `cannot_tell`, tier 6, open grid cell kept.

## Standing rules for the three-trait batches (from the pilot review)

- About half of kind D, and the material in kind C, should be real cited text: public MIT repos outside the
  exclusion list, pinned, with the first line of the `LICENSE` file recorded in `sources[]`.
- In kind C, each reply's wording must land on its keyed `planted_flaw` — say "guessed" for `guess`, not a
  neighbouring vice.
- Put two or three tier-4/5 items per trait in kind D: longer material, with the deciding line buried.

## Reviews requested (handoff §4.1)

1. R&D: key check (instant agreement of keys, truth of every thinking step, the planted flaws, leakage);
2. the Publisher reads all 40 items;
3. the maintainer reads a sample.

Anything that fails gets fixed in the approach (item patterns in this README, `checks.py`, the kind
templates), not just in the item, before the batches of three traits begin.
