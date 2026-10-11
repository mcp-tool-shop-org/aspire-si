# Stage 1 gates as a package (aspire-si-gates)

*2026-10-10. Option (a) of the role-set design, confirmed by the Publisher: the data gates move out of
`docs/runs/stage1-train/data.py` into `packages/aspire-si-gates`, a standard-library package that the trainer and
the dataset generator both import, pinned by commit. It merges only between dataset batches.*

## What moved, and what didn't

- **Into the package:** every check (schema, per-kind parts, amendments 1–3, the diversity gate, phrase and
  formula limits, floors, pairs), the leakage guard against held-out sets passed in, and the batch and freeze
  reports (`gates.py`); the verifier role's data (lessons, trait names, role words, the pre-amendment lessons) in
  `constants.py`; the three-line parser as `answer_format.py`; `check_against` and its reason table
  (`context.py`); `GATES_VERSION` and `gates_identity()`.
- **Stays in aspire-si:** `sequences()` (the tokenizer side) and the CLI in `data.py`, which now re-exports the
  package under the old names, so `checks.py`, `no_role.py` and the tests are unchanged; the sealed-set leakage
  leg; `no_role.py` and the trainer.
- **Unchanged, byte for byte:** the frozen eval files (harness `4fa52448…`, scorers `fd59cb50…`, `64f52f6f…`,
  `55b4d0ea…`) and `no_role.py` (`63162cf4…`). The parser is a copy because the harness is frozen; a test asserts
  the two sources are identical.

## The API

`validate(rows)` is `check_item` on every row, then `check_batch` on the set, in the old order.
`check_against(row, accepted)` returns `[(code, message)]` for the exclusivity rules (spans, framings, material,
puzzle shapes, openings, B phrases); the codes and message templates are pinned by a test.

## The equality receipt

Before the split, refusal text could name a different example among ties from one run to the next (set order
varies with the hash seed); the first commit fixes the choice of example, outcomes unchanged. Then every file
below was run through `validate` (scoped and unscoped), `batch_report`, and `freeze_report` over the #84 head set,
before and after the split, under two hash seeds: **the results are identical, every pass and every refusal
text.**

| File | sha256 | validate (scoped) | validate (unscoped) |
|---|---|---|---|
| `e2fa389/batch3-t05.jsonl` | `efe2dbc37ca2` | refused: T05: the span 'a line that states that and for a' appears in 7 items (T05-C-01, T05-C-05, T05-D-02, T05-D-03,  | refused: T05: the span 'a line that states that and for a' appears in 7 items (T05-C-01, T05-C-05, T05-D-02, T05-D-03,  |
| `e2fa389/batch3-t13.jsonl` | `cb534b2c08a7` | refused: T13: the span 'a line that states that and for a' appears in 7 items (T13-C-01, T13-C-05, T13-D-02, T13-D-03,  | refused: T13: the span 'a line that states that and for a' appears in 7 items (T13-C-01, T13-C-05, T13-D-02, T13-D-03,  |
| `e2fa389/batch3-t17.jsonl` | `c8b718743f02` | refused: T17: the span 'i will quote copied with their marks [q]' appears in 7 items (T17-C-03, T17-C-04, T17-C-06, T17 | refused: T17: the span 'i will quote copied with their marks [q]' appears in 7 items (T17-C-03, T17-C-04, T17-C-06, T17 |
| `#84 head/batch2-t01.jsonl` | `91596935ddf3` | pass | refused: line 20 (T01-C-04): the prompt names the role, a trait or a pattern ('tree') outside the material and the stat |
| `#84 head/batch2-t14.jsonl` | `94d78ea08f50` | pass | refused: line 23 (T14-C-07): the prompt names the role, a trait or a pattern ('literal') outside the material and the s |
| `#84 head/batch2-t16.jsonl` | `6750c23b8b73` | pass | refused: T16: 3 of 18 verdict-bearing items (17%) have 2-4 DECIDING lines |
| `#84 head/batch3-t05.jsonl` | `bf5968f5ee46` | refused: T05: the phrase 'i stop' is in 39 of 42 turns (93%) | refused: T05: the phrase 'i stop' is in 39 of 42 turns (93%) |
| `#84 head/batch3-t13.jsonl` | `a9df97bebb9e` | refused: T13: the phrase 'i stop' is in 39 of 43 turns (91%) | refused: T13: the phrase 'i stop' is in 39 of 43 turns (91%) |
| `#84 head/batch3-t17.jsonl` | `2ed5c6c6854f` | refused: T17: the phrase 'i stop' is in 36 of 42 turns (86%) | refused: T17: the phrase 'i stop' is in 36 of 42 turns (86%) |
| `#84 head/pilot-t04.jsonl` | `df29c3a10b0d` | pass | refused: T04: 1 of 9 verdict-bearing items (11%) have 2-4 DECIDING lines |
