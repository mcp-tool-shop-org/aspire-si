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
| `pair` | string, optional | a contrast pair's shared id: the same statement against material that is silent (`cannot_tell`) and material that says otherwise (`unsupported`); exactly those two items carry it (amendment 2) |

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

## Drift checks (the handoff's required parts and content rules, enforced)

- **Every item:**
  - `sources` is non-empty. Each entry is either `repo@commit:path#lines` (a string, or `{repo, commit, ...}`)
    or marked invented (`{"invented": "<what>"}`).
- **DECIDING:**
  - keyed lines are verbatim in the material with markup kept (`**` stays). Only whitespace may differ, so a
    quoted line can wrap;
  - a reply's DECIDING equals its key exactly.
- **Kind A:**
  - the user turn has at least 2 sentences before its question, and names the trait (the validator's
    `TRAIT_NAMES` list per lesson);
  - the reply starts in the first person ("As a verifier, I …", "I …");
  - the reply has no VERDICT line.
- **Kind B:**
  - the thinking or reply draws a pattern: tree glyphs or ROOT, a `| … |` grid, or numbered elimination
    steps;
  - no 8-word phrase recurs in more than 2 kind-B items of one trait.
- **Kind C:**
  - the user turn carries the other checker's verdict (the word "verdict" or a verdict value);
  - if the material or user turn cites a repository (`owner/repo@sha` or a github.com link), `sources` has
    a real (non-invented) entry.
- **Kind E:** the variant's key shares at least one key point with the original.
- **Kinds C, D and E:**
  - no undated time word ("today", "now", "currently", "latest", …) in the `Statement:` unless the material
    carries a date;
  - no opening frame (the first 6 words of the user turn) in more than 2 items of one trait.

## Amendment checks (batch 3 on; the pilot T04 and batch 2 T01, T14, T16 are skipped by name)

- **Quoting (amendment 1):** a DECIDING line inside a table row quotes the whole row; in a changelog, a line
  under a version or date heading carries that heading as its own DECIDING line; per lesson, at least 25% of
  supported/unsupported items have 2–4 DECIDING lines. Everywhere: no trailing `|`, and no NONE beside lines.
- **Floors (amendment 2), per lesson:** cannot_tell and supported each at least 25% of verdict-bearing items
  (C, D, verdict-type E); at least one ESCALATE `yes (…)` item and one cannot_tell item with ESCALATE `no`; at
  least two multi-turn D items, one whose verdict changes and one where it stands, every turn with its own three
  lines; at least two contrast pairs (`pair`), each exactly one cannot_tell and one unsupported item.
- **Role-neutral prompts:** C, D and verdict-type E user turns carry no role, trait or pattern word (the fixed
  `ROLE_WORDS` list: "verifier", the 18 trait names and short forms, tree, prune, falsifier, elimination, grid)
  outside the material lines and the statement line, so the ROLE and NO-ROLE arms share these prompts exactly.
  "Checker" is allowed. The other checker's quoted thinking counts as prompt, so it avoids these words too.
- **Diversity gate (after batch 3), per lesson:** no 8-word span outside material lines, the Statement line and
  the three lines appears in 3 or more items (a bracketed quote counts as one token, so a skeleton sentence
  around different quotes is still caught); a C or D framing that names a kind of material ("board", "rules",
  "snippet", "changelog", …) fits the item's `content_kind` (only clear mismatches are refused); the same
  material appears in at most 2 items, and only as a contrast pair; no two non-verification B puzzles share a
  structure signature (entity count, and which cues the clues use: order, chained order, gate, negation,
  one-each). Across one batch (`python data.py --batch <files>`): at most 2 puzzles per signature, and no
  framing line repeats.
- **Phrases and formulas (after the rebuilt batch 3), per lesson:** no 2–7 word phrase in more than 40% of a
  lesson's assistant turns (outside material lines, the Statement line and the three lines). A phrase made only of
  function words and the task's own nouns (claim, statement, material, line, verdict, text, …) is exempt. No
  sentence template (numerals as N, each determiner's noun phrase as X), and no opener of one ("step N …"), in
  more than 25% of a lesson's kind B items, and in at least 2 of them. There is no length floor; padding would be
  worse.
- **Freeze, over every file together** (`python data.py --freeze <files>`): cannot_tell, supported and the
  multi-line share each at least 25% across all lessons, an ESCALATE yes item in every lesson (the four
  skipped lessons are reported, not refused), and no C or D framing line repeated anywhere. It prints each figure and how many more items a shortfall needs.

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
