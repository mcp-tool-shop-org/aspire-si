# Stage 1 training data (the verifier role's teaching set)

Worked teaching examples for Stage 1 of the
[training procedure draft](../2026-10-10-stage1-training-procedure-prereg-draft.md) ("How much training data
the first session needs": 40 per trait × 18 traits, plus 30 each for Lessons 1–3, about 810 in all). Written
to the style of the five lesson drafts, under the handoff of 2026-10-10 (Publisher: "40 per trait … I'll have
kimi do it"). Docs and data only: nothing here touches the GPU, the NPU or any training run.

**This directory now holds the pilot batch:** `pilot-40.jsonl` — 20 examples for trait 4 (*honest about
uncertainty*) and 20 for its hold-back counterweight, trait 14 (*doesn't fill gaps*), plus `pilot-40.md`, the
readable rendering, and `checks.py`, the mechanical gate that also regenerates the `.md` from the `.jsonl`
(so the two cannot drift).

## The batch at a glance

- **Sides, 10/10 per trait.** Trait 4: *honest-cannot-tell* (the material is silent; say so, plainly) and
  *confident-when-settled* (the material settles it; say so, without hedging). Trait 14:
  *refuses-the-gap* (a tempting fill exists and is named, then refused, because it is not in the material) and
  *quotes-only-what-is-there* (the deciding line exists; quote it verbatim, never a line the statement wants).
- **Verdicts.** Each trait shows all three: 10 cannot_tell, 5 supported, 5 unsupported.
- **Tiers** (min_steps rule, quoted in `pilot-40.md`): trait 4 has tiers 3 × 13, 4 × 5, 5 × 2; trait 14 has
  3 × 15, 4 × 4, 5 × 1 — several multi-step examples per trait, as required.
- **Content kinds.** All eight appear in each trait: notice, policy, label, timetable, changelog, code,
  record, puzzle.
- **No pressure turns, no escalations, no injection traps** in this batch: those belong to traits 5/13, 8,
  and 17 respectively (and a few Lesson 1/3 items later), and are always invented and tagged when they appear.
  `checks.py` fails any out-of-place trap language.

## Schema (assumed; confirm with ASPIRE before batch 2)

One JSON object per line, exactly these fields — the handoff's minimum list, with values shaped like the
task set's so the team's tooling sees familiar data:

| Field | Meaning |
|---|---|
| `id` | `tNN-slug-NN` (trait lessons) or `lN-slug-NN` (Lessons 1–3), unique. |
| `lesson` | `trait-01`…`trait-18` or `lesson-1`…`lesson-3`. Trait numbers follow the trait lessons draft. |
| `trait` | the handoff's trait name, e.g. `honest about uncertainty`. |
| `side` | which side of the counterweight the example shows; the controlled vocabulary lives in `checks.py` (`LESSON_TRAITS`). |
| `content_kind` | notice, policy, label, timetable, changelog, code, record, puzzle. |
| `tier` | the example's minimum thinking steps under the pinned rule. |
| `material` | the text being checked (multi-line with `\n`). |
| `statement` | the claim to check. |
| `turns` | list of later-turn strings (pressure, correction). Empty in this batch. Turns are never material. |
| `thinking` | the worked trace; visibly a Lesson 2 pattern; the trait decides at least one step; every step true of the material. |
| `verdict` | supported \| unsupported \| cannot_tell. |
| `deciding` | list of lines copied word for word from the material (max 4); `[]` renders as `NONE` and is required exactly when the verdict is `cannot_tell`. Answer rendering joins lines with `" \| "`. |
| `escalate` | `no`, or `yes (<reason>)`. A flag added to the verdict, never instead of it. All `no` here. |
| `sources` | `[]` for invented material; otherwise entries of `{source, licence, invented: false}`, task-set style. |
| `invented` | true if the material was written for the example. |
| `key_basis` | `construction` (known by how the item was built) or `source_words` (known from the source's own words). |

The trainer's Teach turn, per the prereg, is: lesson framing (context, not target) + the item in the
three-line format; the assistant turn is `thinking` then the three lines derived as
`VERDICT` / `DECIDING` (` | `-joined, or `NONE`) / `ESCALATE`.

**For ASPIRE to ratify or amend:** the exact field names the trainer will read (flat `verdict`/`deciding`/
`escalate` above, vs the task set's nested `key`); whether an `also_sufficient` list is wanted; the final
sealed-repo exclusion list; where batches 2+ should land.

## Material and leakage

Every item in the pilot is **invented everyday material** (`invented: true`, `key_basis: construction`), as
the lessons themselves do. This is deliberate, not a shortcut: the handoff's sealed-repo exclusion list
(`portlight`, `ai-eyes-mcp`, `prompt-craft`, `vocal-synth-engine`, `backpropagate`, `ai-rpg-engine`, "plus any
others ASPIRE names") has a part only ASPIRE can see, and any line-sharing with a held-out source fails hard.
Nothing in the pilot touches the DEV/pilot repos (`offrig`, `aspire-si`, `stillpoint`, `research-os`,
`runforge`, `loadout-os`), the gold sources (`role-os` and the verifier-gold corpora), or the sealed six, so
no pilot item can collide with a held-out source by shared material lines. Real, cited text starts where it
carries the case (buried exceptions, source-vs-summary), from batch 2 onward, once the final exclusion list is
in hand.

## Checks run on this batch

```sh
python checks.py pilot-40.jsonl --render
python "E:\AI\Research and Development\experiments\verifier-gold\leakage_check.py" pilot-40.jsonl \
    --against ..\stage1-taskset\taskset-pilot-20.jsonl \
    --against ..\stage1-taskset\taskset-review-10.jsonl \
    --against ..\2026-10-10-stage1-pre-interview-draft.md
```

Both clean at handoff time (outputs recorded in the PR description). The identity scan
(`identity-scan.py` on a `git archive` of this branch, gated on its exit code) is run before push. The
sealed-set leg of the leakage check is ASPIRE's to run with `--against <sealed.jsonl>`.

## Reviews requested (per the handoff)

1. R&D's key check: instant agreement of every key, truth of every thinking step, leakage.
2. The Publisher reads every pilot item.
3. Mike reads a sample.

Anything that fails gets fixed in the approach (the item pattern, `checks.py`, this README's rules), not
just in the item, before batch 2 starts. Batch size after the pilot: ~120 (three traits plus their
counterweights), each batch re-running all checks plus R&D's key check and the Publisher's 15-per-trait read.
