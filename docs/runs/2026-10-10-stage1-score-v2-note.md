# Stage 1 eval: score v2 (a parsing repair, reported beside the pinned score)

*2026-10-10. Approved by the Publisher after the sealed baseline's first two seeds. The pinned scorer is
unchanged and stays the primary number.*

## Hashes (frozen before any trained adapter is scored)

| File | sha256 |
|---|---|
| `stage1-eval/score.py` (pinned, primary) | `fd59cb5074d0858af2864bcadd68443ddea26c325beef3afe6f5cdadd624ab7d` |
| `stage1-eval/score_v2.py` (v2, reported beside it) | `64f52f6fe60d077a37db6d043476ac9e37d38043d756f203bb8d802b90e3cb44` |
| `stage1-eval/test_score_v2.py` | `1b044ccda601b2fa6de2ebcdcbe667b79c672a022e81ee1916e751bc014c9841` |

v2 imports `score.py` and `harness.py` unchanged, so its result also depends on their pinned hashes. Neither
file changes after this commit; a further change would be v3, with its own note.

## Why it exists

`score.py` splits DECIDING on `" | "` only. When the model ends (or starts) DECIDING with a lone `|`, or lists
`NONE` beside real quotes, the separator or the token stays attached as a "line" and fails the
substring-of-material check, although every quote is right. That is a parsing defect, not a judgment about
the answer.

## What v2 changes, and nothing else

When the lines as given fail, the check is tried once more with a lone leading or trailing `|` removed from the
whole DECIDING string and `NONE` elements dropped when other lines are present. It is tried second, not
instead, because a quoted table row legitimately starts and ends with `|`. So v2 can only turn a pinned
failure into a pass through this repair, never the reverse.

Unchanged: normalising, edge trims, one-way coverage, no case folding, dropped backticks fail, only material
counts, correction tasks need both turns. The tests cover both repairs, a dropped-backtick quote that still
fails under v2, a quoted table row, and that v2 never fails what the pinned scorer passes.

## Effect on the baseline (untrained student)

| Set, seed | Pinned correct | v2 correct |
|---|---|---|
| pilot-30, s0 / s1 / s2 | 14 / 12 / 12 | 14 / 12 / 12 |
| sealed-130, s0 / s1 | 11 / 15 | 14 / 18 |

Seed 2 of the sealed set is added with the baseline write-up. Pinned and v2 are reported side by side, before
and after training.

## Not in v2

Formatting tolerance (dropped backticks, case) and partial key lines (a table cell, a bullet without its
version heading) are the ratified rules working as written. Whether a lenient secondary measure is reported
is R&D's call; if it is, it is pre-registered with exact rules and frozen before round 1, and it is never the
primary number.
