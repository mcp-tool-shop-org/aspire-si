# Auditor run receipts (2026-10-08 run, judged 2026-10-09)

These are the receipts behind the Auditor plan's addendum 5, amendments 1 and 2
(`docs/runs/2026-10-08-auditor-plan.md`). They're committed byte for byte, so the sha256s
recorded in the plan and in each manifest hold. The `.gitattributes` keeps git from rewriting
line endings here. switchyard's `aspire_judges` importer reads them with `git show` at a pinned
commit of this repository.

| File | What it is | sha256 |
|---|---|---|
| `B-selection.json` | B's frozen selection: 14 splits, 28 both-not-wrong and 28 both-wrong control items, and the order. Written before any verdict | `9ef063a78593490fecadd77c093944570f03179f392ccd7224a8f42c4ca179a2` |
| `B-manifest.json` | B's run manifest: judge, digest, prompt and schema sha256, options, ids, order | `a1efe68ff8f0de27d46f0352e67b694a74b7e8b24480b69fd987b07e4a2b8a02` |
| `verdicts-error-B-mistral-small_24b.json` | mistral-small:24b's 70 verdicts, with its stated reasoning | `a1c984b29c3e3631f150019899c704b75e268b356ebec786e186c7b27fde7711` |
| `B-readout.json` | B's readout under amendment 1's rules (the control fails; there's no tie-break reading) | `d61915ebf4106c3067b6ea08d83c3aace9555113afb8193fe480c7013cb47eb5` |
| `judge-control-selection.json` | Stage 3b's judge control: 61 items built by rule, frozen before any verdict | `4a68cb7606b39641e1649e9d622f87008e7c561ed41e6d9309cbe6c8382a687c` |

The 3b control's manifest, verdicts and readout are added here once that run has happened.

## Where the text comes from

- **The answers** (questions, strong answers and the planted or rule-made edits inside them) were
  written by Qwen2.5-32B-Instruct, licensed Apache-2.0.
  - The planted errors in B's items come from the same model at Q4_K_M.
  - The judge control's edits were made by code, from fixed rules.
- **The verdicts in this folder** are mistral-small:24b's, licensed Apache-2.0.
- **Not included:** the other judges' verdicts (gemma4:31b, muse-glimmer) stay in the run directory.
  Their counts are summarised in the plan.
