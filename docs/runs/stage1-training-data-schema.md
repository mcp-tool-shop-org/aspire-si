# Stage 1 training data: the JSONL schema the trainer reads

One example per line, UTF-8, LF line endings. The trainer validates every line against this before training
and refuses the whole file on any error, naming the line and the field.

## Fields

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | string | yes | unique, stable, e.g. `T05-honest-uncertainty-017` |
| `lesson` | string | yes | `L1`, `L2`, `L3`, or `T01`–`T18` (the trait lessons, in the trait file's order) |
| `trait` | string | yes | the trait as named in its lesson heading; for L1–L3, `role`, the pattern name, or the expectation |
| `side` | string | yes | which side of the counterweight the example shows, e.g. `rejects` / `rightly accepts`; `n/a` for L1–L3 if none |
| `content_kind` | string | yes | `notice`, `policy`, `label`, `timetable`, `changelog`, `code`, `record`, `puzzle`, or `other:<kind>` |
| `tier` | integer | yes | minimum steps under the eval plan's pinned `min_steps` rule |
| `material` | string | yes | the text being checked, exactly as the model will see it |
| `statement` | string | yes | the claim to check |
| `thinking` | string | yes | the worked thinking for turn 1, plain text; it becomes the `<think>` block |
| `verdict` | string | yes | `supported`, `unsupported` or `cannot_tell` (turn 1) |
| `deciding` | list of strings | yes | exact lines from `material`, at most 4; empty list for `cannot_tell` (rendered `NONE`) |
| `escalate` | string | yes | `no`, or `yes (<reason>)` |
| `turns` | list of objects | no | later turns, in order; each `{user, new_material, thinking, verdict, deciding, escalate}` (below) |
| `sources` | list of objects | yes | `{repo, commit, path, lines, licence}` per real source; empty if all invented |
| `invented` | boolean | yes | true if any of `material` is invented (everyday material written for the lesson) |
| `key_basis` | string | yes | `construction` or `source` (how the answer is known) |
| `tags` | list of strings | no | e.g. `injection_trap`, `pressure`, `correction` |

**A later turn** (`turns[i]`):
- `user`: the user's message, always invented (pressure or a correction).
- `new_material`: a list holding the genuinely new material lines that the message carries. It's empty for
  pressure, which is never material.
- `thinking`, `verdict`, `deciding`, `escalate`: the model's answer at that turn. `deciding` lines come from
  `material` plus that turn's (and earlier turns') `new_material`.

## Checks the trainer runs (an example failing any of them is refused)

- Every `deciding` line is a substring of the material it may quote, after the scorer's normalisation:
  whitespace collapsed, `**` and `__` removed, trimmed.
- `verdict` is `cannot_tell` exactly when `deciding` is empty.
- `escalate` matches `^(no|yes \(.+\))$`.
- `thinking` is non-empty and contains no `<think>` or `</think>` tags; the trainer adds them.
- `id` is unique across the file. `lesson` is known, and `tier` is at least 1.

## How an example becomes a training sequence

- **User turn:** the lesson's framing ("As a verifier, you are …" with its paragraph on why), then the item in
  the three-line answer format.
- **Assistant turn:** `<think>` + `thinking` + `</think>`, then
  `VERDICT: …` / `DECIDING: …` (lines joined by ` | `, or `NONE`) / `ESCALATE: …`.
- **Later turns:** each follows as a user turn and an assistant turn the same way.
- **Loss:** on the assistant turns only.

## Stay clear of the test sets: the final exclusion list

Don't draw material from any of these repositories (any file, any commit):
- **Sealed set (frozen with the 130):** portlight, ai-eyes-mcp, prompt-craft, vocal-synth-engine,
  backpropagate, ai-rpg-engine, and all of offrig.
- **DEV/pilot:** offrig, aspire-si, stillpoint, research-os, runforge, loadout-os.
- **Gold sets:** role-os, offrig, aspire-si and rnd, including rnd's math ladder (no arithmetic or
  small-function items in its style).
- **Public anchors:** LLM-AggreFact, VitaminC and HoVer, and the Wikipedia claims they're built from.
- **Stage 2's domain sources,** once named.
- **No MIT licence text** (the pilot's licence tasks use it). Use Apache-2.0, BSD or invented terms.

R&D runs `leakage_check.py` (rnd 7b77c79) on every batch. It fails the same claim, a 5-gram Jaccard of 0.8 or
more, or 2 or more shared material lines against the held-out sets, and warns from a Jaccard of 0.5. The
trainer applies the same thresholds against the sealed set, the pilot and the pre-interview, and refuses an
example that fails them.
