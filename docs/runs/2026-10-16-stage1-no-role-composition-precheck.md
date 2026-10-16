# Stage 1: the NO-ROLE composition check on the 640 rows, before the freeze (2026-10-16)

The round-1 run sheet's composition check (the evaluation plan's 2026-10-10 amendment, point 3), run on the 640
training rows as they stand after #140 (R&D's ruling on the 22 kind B items the builder refused). **It's a
pre-freeze reading.** The registered check runs again on the files `FROZEN.md` lists, once batches 7 and 8 land,
and that run's NO-ROLE file is the one pinned for round 1.

## How it was run

- **Code:** `verifier-distill-prereg-draft` at `034a3d2`; `no_role.py` `ac795e2b` (the pin in run sheet §9).
- **Command:** `no_role.py --data <the 16 files below> --out <scratch>/no-role-640.jsonl`, with the student's
  tokenizer counting tokens. It needs no card; it ran on the CPU (`CUDA_VISIBLE_DEVICES=` empty).
- **Exit 0.** The builder refuses any differing count cell, so a clean exit already means every count matched.
- **Full output:** [`stage1-composition/2026-10-16-precheck-640.txt`](stage1-composition/2026-10-16-precheck-640.txt)
  (sha256 `e3fe48c6…`), the side-by-side table, the token flags and the exposure per material.

| File | Rows | sha256 (first 12) |
|---|---|---|
| batch2-t01.jsonl | 40 | `7fe25e9171a4` |
| batch2-t14.jsonl | 40 | `6b9a6218aa81` |
| batch2-t16.jsonl | 40 | `552535951ed4` |
| batch3-t05.jsonl | 40 | `d98c4ace7f84` |
| batch3-t13.jsonl | 40 | `7b005298aaf0` |
| batch3-t17.jsonl | 40 | `02211b2f37b4` |
| batch4-t02.jsonl | 40 | `b77a361ff851` |
| batch4-t03.jsonl | 40 | `0eb944b4c93a` |
| batch4-t06.jsonl | 40 | `f58332ab68e1` |
| batch5-t07.jsonl | 40 | `7f940a1298b2` |
| batch5-t08.jsonl | 40 | `f04db8e668cc` |
| batch5-t09.jsonl | 40 | `45b7b2c76025` |
| batch6-t10.jsonl | 40 | `91cd7f955c3f` |
| batch6-t11.jsonl | 40 | `316e4525a350` |
| batch6-t12.jsonl | 40 | `3c53b7d6b236` |
| pilot-t04.jsonl | 40 | `aea4fe056811` |

## The gate: the tables match (registered: within one item per trait)

**All 690 count cells are equal between the arms: 0 differ.** The gate allows a difference of one item per trait,
and any non-zero difference would be reported with its cause; there is none to report. The totals:

| Cell | ROLE | NO-ROLE |
|---|---|---|
| items | 640 | 640 |
| verdict supported | 118 | 118 |
| verdict unsupported | 103 | 103 |
| verdict cannot_tell | 86 | 86 |
| NONE | 86 | 86 |
| escalate yes | 27 | 27 |
| multi-line | 90 | 90 |
| paired | 48 | 48 |

Per lesson × kind, the items, sequences, tiers, verdicts, NONE, escalate, multi-line and pairs are in the full output.

## Reported, not gated

- **Target tokens:** ROLE 77,374, NO-ROLE 48,269, a ratio of **0.624**. That's under 0.85, so the builder prints
  its limitation line (the baseline run note recorded 0.644 at the approval). NO-ROLE is shorter, never padded, by
  design: a read-through stops when the material runs out.
  - 70 of the 80 lesson × kind cells are flagged under 0.85.
  - The lowest is T07 A at 0.25 (158 of 627); the highest flagged is T08 C at 0.84.
  - By kind: E 0.487, A 0.534, B 0.596, D 0.642, C 0.765. The material-recall kinds (A and E) are the
    shortest, as the design expects.
- **Exposure per material:** 306 materials. In 205 of them NO-ROLE sees the material more often than ROLE: the
  A and E-written recall counterparts reuse the lesson's C, D and E-verdict materials, spread evenly. The full
  per-material list is in the output.

## The NO-ROLE file

The 640-row NO-ROLE file from this run has sha256
`76ff2084e475af0a34ee7cee1a20155c2c9dcf428589a4d08cb71a50970c14b3`. It isn't committed and isn't the round-1
pin: the pin comes from the run on the frozen files.

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | The code commit, the builder's sha256, every input file's sha256 and the output's sha256 are recorded here. |
| ANDON_AUTHORITY | 2 | The builder refuses a differing count cell and any item it can't derive; this run exited 0. |
| NAMED_COMPENSATORS | n/a | Nothing irreversible: a CPU run writing a scratch file, and this note. |
| DECOMPOSE_BY_SECRETS | 2 | The builder (`no_role.py`) is separate from the data and from this reading of its output. |
| UNCERTAINTY_GATED_HUMANS | 2 | The gate is R&D's check of the tables; the token ratio and exposure go to R&D as reported figures. |
| EXTERNAL_VERIFIER | n/a | No specialized claims. |
