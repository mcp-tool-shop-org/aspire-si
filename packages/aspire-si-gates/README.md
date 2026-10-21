# aspire-si-gates

The data gates for the Stage 1 role-training set, as one standard-library package shared by aspire-si's trainer
and the dataset generator. Install it at a pinned commit:

```bash
pip install "git+https://github.com/mcp-tool-shop-org/aspire-si@<sha>#subdirectory=packages/aspire-si-gates"
```

- `validate(rows)`: every check on a whole file; it raises `DataError` with the first refusal.
- `check_item(row)`: the checks that need no other row. It refuses an apply item (`split: apply` or an `-AP-` id):
  apply items are never trained.
- `check_apply_item(row)` (1.7.5): an apply item's checks. The same content and shape checks as `check_item` (one shared
  body), without the never-trained refusal, plus the apply rules: `split: apply` and an `-AP-` id both, kind C or D, one
  turn, no pair, and the lesson's twin field (`r10_twin` for T10, `order_free` for T09, `silent_without_deciding` for
  T01, T04 and T14). It refuses any row that isn't an apply item.
- `check_trace_leaks_label(row)` (1.7.6): refuses a kind C row whose checker trace names its own planted flaw, or any
  flaw stem outside the two-step carve-out (verbatim-quoted spans of at least 4 characters from the material or
  statement are stripped first; on what remains, the row's own `planted_flaw` stem is never excused, while any of the
  other 7 flaw stems is excused when the same stem family matches in the material or statement). Wired into
  `check_batch`.
- `check_against(row, accepted)`: `[(code, message)]` for the exclusivity rules (spans, framings, material,
  puzzle shapes, openings, B phrases) against rows already accepted; codes are in `REASONS`.
- `check_batch(rows)`: the checks that need the finished set (shares, floors, pairs, phrase and formula limits).
- `batch_report(rows)`, `freeze_report(rows)`: the figures, as dicts.
- `gates_identity()`: the version and a sha256 of the package source, to record with each generated item.

The verifier role's data (lessons, trait names, role words, the pre-amendment lessons) is in `constants.py`.
The sealed-set leakage leg is not here; it stays with the trainer.
