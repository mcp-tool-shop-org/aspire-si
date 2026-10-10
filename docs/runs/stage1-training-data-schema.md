# Stage 1 training data: the JSONL schema the trainer reads

For the role-shaping curriculum in
[2026-10-10-stage1-training-data-kimi-handoff.md](2026-10-10-stage1-training-data-kimi-handoff.md). Each item is
one line of JSONL: UTF-8, LF line endings. The trainer validates every line before training and refuses the
whole file on any error, naming the line and the field.

**The training target is every assistant turn, whole:** its thinking (if any) and its full reply, meaning the
explanation, analysis, self-talk and answer. The user turns are context. This holds for all five kinds.

## Fields (every item)

| Field | Type | Meaning |
|---|---|---|
| `id` | string | unique and stable, e.g. `T04-B-003` |
| `lesson` | string | `L1`, `L2`, `L3`, or `T01`–`T18` (the trait lessons, in the trait file's order) |
| `trait` | string | the trait as its lesson names it (for L1–L3: the role, the pattern, or the expectation) |
| `kind` | string | `A` identity and purpose · `B` thinking pattern · `C` contrasting case · `D` role in action · `E` quiz item |
| `side` | string | the counterweight side shown, e.g. `rejects` / `rightly accepts`, `holds` / `revises`; `n/a` if none |
| `content_kind` | string | `notice`, `policy`, `label`, `timetable`, `changelog`, `code`, `record`, `design-note`, `puzzle`, `teaching`, or `other:<kind>` |
| `tier` | integer | minimum steps under the eval plan's pinned `min_steps` rule; 1 for a one-step item |
| `turns` | list | the conversation, in order: `{user, thinking, reply}` objects (below) |
| `material` | string or null | the text being checked, if the item checks one (always for D; for C, the material the other checker had) |
| `sources` | list | `{repo, commit, path, lines, licence}` for each real source; empty if all invented |
| `invented` | boolean | true if any material, puzzle, pressure turn, injection line or checker trace is invented |
| `key_basis` | string | `construction` or `source` |
| `tags` | list of strings | e.g. `pressure`, `correction`, `injection_trap`, `non-verification` |

**A turn:**
- `user` (string): what the user says. For D, the first turn's `user` holds the material and the statement,
  laid out however the item needs.
- `new_material` (list, optional): the genuinely new material lines a later user turn brings (a correction).
  Pressure is never material.
- `thinking` (string, may be empty): the inner reasoning, which becomes the `<think>` block. Self-talk lives
  here for D.
- `reply` (string): the visible answer. For D, and any turn that gives a verdict, it ends with the three
  lines.

## Keys (by kind)

| Kind | Required keys | The trainer checks |
|---|---|---|
| A | `key_points`: the trait's main point and its reason, as short phrases | nothing mechanical beyond non-empty; reviewed by R&D and the Publisher |
| B | `answer`: the puzzle's or pattern's correct result, stated briefly | `answer` is non-empty |
| C | `planted_flaw`: `rubber_stamp`, `guess`, `loop`, `nitpick`, `fix`, `obeyed_material`, `caved`, `reject_by_default`, or `none` (the other checker was right); plus `verdict`, `deciding`, `escalate`, the right answer | the last turn's reply ends with three lines that match `verdict`, `deciding` and `escalate` |
| D | `verdict`, `deciding`, `escalate`, per verdict-giving turn (top-level for the last turn; `turn_keys` for earlier ones) | each verdict-giving reply's three lines match its keys; every `deciding` line is in the material available at that turn |
| E | `quiz`: `{type: "verdict"` with `verdict`, `deciding`, `escalate}`, or `{type: "written"` with `key_points}`; plus `variant`: one re-worded `{user, key}` | verdict quizzes are checked like D; the variant is present and differs from the original |

## Checks the trainer runs (an item failing any is refused)

- **DECIDING lines** are substrings of the material, after the scorer's normalisation (whitespace collapsed,
  `**` and `__` removed, trimmed). There are at most 4. `verdict` is `cannot_tell` exactly when `deciding`
  is empty.
- **`escalate`** matches `^(no|yes \((medical|legal|safety|money)[^)]*\))$`. Only the four taught reasons.
- **The thinking** contains no `<think>` or `</think>` tags; the trainer adds them.
- **Shape:** `id` is unique, `lesson` and `kind` are known, and `tier` ≥ 1.
- **Length:** each assistant target is at most 3,000 tokens, and each whole sequence (prefix plus target) at
  most 4,096, in the student's tokenizer. Longer items are refused, never truncated: shorten or split them.
  The limit is set by the trainer's memory budget (smoke test, 2026-10-10: a 2,919-token target peaked at
  24.1 GB).
- **Leakage:** see below.

## How an item becomes training sequences

One sequence per assistant turn. The prefix is the conversation so far: earlier assistant turns are shown
by their `reply` only, without thinking, exactly as the chat template renders history at inference. The
target is `<think>\n{thinking}\n</think>\n\n{reply}<|im_end|>`, with an empty thinking rendered as an empty
block. The loss falls on the target only.

## Stay clear of the test sets: the final exclusion list

Don't draw material from any of these (any file, any commit):
- **Sealed set (frozen with the 130):** portlight, ai-eyes-mcp, prompt-craft, vocal-synth-engine,
  backpropagate, ai-rpg-engine, and all of offrig.
- **DEV/pilot:** offrig, aspire-si, stillpoint, research-os, runforge, loadout-os.
- **Gold sets:** role-os, offrig, aspire-si and rnd, including rnd's math ladder (no arithmetic or
  small-function items in its style).
- **Public anchors:** LLM-AggreFact, VitaminC and HoVer, and the Wikipedia claims they're built from.
- **Stage 2's domain sources,** once named.
- **No MIT licence text.** Use Apache-2.0, BSD or invented terms.
- **The pre-interview's 12 questions:** never asked, paraphrased or answered, above all in kinds A and E.

**Leakage thresholds** (R&D's `leakage_check.py` at rnd 7b77c79, mirrored in the trainer):
- **Fail** a user turn or statement that is the same as, or within a 5-gram Jaccard of 0.8 of, a sealed or
  pilot claim; or an item that shares 2 or more material lines with one.
- **For kinds A and E, the bar is lower:** fail at a Jaccard of 0.5 against any pre-interview question.
- **Warn** from a Jaccard of 0.5 otherwise.
