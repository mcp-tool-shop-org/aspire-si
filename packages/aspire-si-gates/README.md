# aspire-si-gates

The data gates for the Stage 1 role-training set, as one standard-library package shared by aspire-si's trainer
and the dataset generator. Install it at a pinned commit:

```bash
pip install "git+https://github.com/mcp-tool-shop-org/aspire-si@<sha>#subdirectory=packages/aspire-si-gates"
```

- `validate(rows)`: every check on a whole file; it raises `DataError` with the first refusal.
- `check_item(row)`: the checks that need no other row.
- `check_against(row, accepted)`: `[(code, message)]` for the exclusivity rules (spans, framings, material,
  puzzle shapes, openings, B phrases) against rows already accepted; codes are in `REASONS`.
- `check_batch(rows)`: the checks that need the finished set (shares, floors, pairs, phrase and formula limits).
- `batch_report(rows)`, `freeze_report(rows)`: the figures, as dicts.
- `gates_identity()`: the version and a sha256 of the package source, to record with each generated item.

The verifier role's data (lessons, trait names, role words, the pre-amendment lessons) is in `constants.py`.
The sealed-set leakage leg is not here; it stays with the trainer.
