# Smoke test S1: an NLI critic on the Intel side (pre-registration, 2026-10-10)

S1 is the first smoke test of the verifier plan (`docs/runs/2026-10-10-verifier-distill-prereg-draft.md`,
draft PR #79). It tests plumbing and one capability, not a training result. The Publisher has cleared S1–S3
to run before the maintainer picks the supervision option.

**The question:** can `cross-encoder/nli-deberta-v3-base` (revision `6c749ce3…`, Apache-2.0) tell a real
supporting quote from one that fooled a verifier? S1 is its known-answer control. If it passes, it may
select mechanical quote-check hints and feed diagnostics. It never enters the reward: the plan's rule is
that no model-judge score is in the reward. If it fails, it isn't used, and nothing else changes.

## Data (committed, no new labels)

- **Source:** the R&D session's tune calibration verdicts, rnd
  `experiments/verifier-gold/calibration/results/2026-10-09-chain` at commit `e48cc90c`. That's 6 models × 2
  check types, quote rule 1. The claim text is joined by id from `experiments/verifier-gold` at the same
  commit. The script records every verdict file's sha256.
- **The pairing rule** (the R&D session's):
  - lines with `model_verdict == "supported"`, `quote_found` true and an `evidence_quote`;
  - deduped on (claim_id, whitespace-normalised quote), after the verdict filter;
  - gold supported is a **positive**; gold unsupported or cannot_tell is **fooled**, split by gold.
- **Counts** (the R&D session's recount matches): positives 477 (grounded 267, reasoning 210); fooled
  unsupported 178 (110 / 68); fooled cannot_tell 22.
- **What the gold is, and what S1 can show:**
  - The gold at `e48cc90c` is provisional: Claude-written synthetic items, not human-labelled.
  - S1 tests one thing: whether the NLI model separates accepted quotes on supported claims from accepted
    quotes on unsupported claims.
  - S1 cannot validate the gold keys, and it cannot test the cannot_tell items or the traps where the
    deciding fact was removed. Those fail by omission outside the quoted span, which a single
    (quote, claim) score does not see. The gold is never re-keyed or filtered by what the NLI model says.
- **Doc comments:** 201 of the 477 positives and 75 of the 178 fooled unsupported pairs come from gold items
  tagged `has_doc_comment: true`, where a prose comment in the evidence can restate the claim. NLI
  models are trained on prose, so pooling those pairs with code-only pairs could hide a failure on code
  lines. The readout therefore reports each check type's rates again for three strata: doc-comment pairs,
  code-only pairs, and pairs whose gold item carries no such flag. The strata are reported and never
  gated: the pass line below is judged on the pooled pairs, as fixed.

## Pass line, per check type (fixed now)

- entailment on positives **≥ 0.85**;
- entailment on gold-unsupported fooled pairs with a Wilson 95% upper bound **≤ 0.15**.

**The rules around it:**
- Gold-cannot_tell fooled pairs are reported and never gated.
- An incomplete run never passes. That covers a timeout or a stopped device, a `--limit` dry check, any pair
  left unscored, and a device that does not read healthy afterwards. It also never prints `S1-OK`, and a
  stopped or unhealthy run exits non-zero.
- NLI is used only for check types that pass. A reasoning fail is expected, because one quoted line rarely
  entails a conclusion about a diff.

## Devices and safety

- **Primary run: the NPU.** That's the plan's device, and the Publisher measured 0.27 s a pair there at
  batch 1, parity-clean.
- **Second run, in the same quiet-CPU window: the Intel iGPU.** The intel-npu notes show DeBERTa NLI is
  about 6× the CPU on the iGPU and no faster than the CPU on the NPU. Both results are reported:
  - label agreement between the two devices;
  - p50 and p95 latency on each.

  The faster device that agrees becomes the critic's device. The pass line is judged on the NPU run.
- **The safety shape:** batch 1, static shape 1×512, devices chosen by name (never the RTX 5090).
  - The NPU's health (`Get-PnpDevice 'Intel(R) AI Boost'`) is checked before and after, on every run
    whichever device it is for; the run refuses to start unless it reads OK, and a run that ends with it
    not reading OK exits non-zero.
  - Every call has a 10 s timeout. A timeout stops the run and is reported.
  - Nothing else may hold the Intel devices during the window.
- **Duration:** about 3 minutes of scoring per device, plus model compile, about 10 minutes in all.

## Rehearsal (done, CPU, 5 pairs)

The script ran end to end on the CPU: 0.25 s a pair at 1×512, the NPU healthy before and after.
- **Tests:** 9 unit tests cover the pairing rule (including dedupe after filtering), the readout, the pass
  line, the device guard and the timeout.
- **An early sign, not a result:** on 2 of the first 5 positives the model said "neutral" for a real
  supporting code line. NLI cross-encoders are trained on prose, and S1 measures how much that matters
  here.

## Receipts

- `s1-NPU.json` and `s1-iGPU.json` (readout, latency, health, device names, the sha256 of every verdict
  file and of the three gold files, and the `--limit` if one was given);
- `s1-<device>-labels.jsonl`.

They're committed byte for byte under `docs/runs/receipts/2026-10-10-smoke/` after the run, identity-scanned.

## Results (run 2026-10-27, aspire-si 6c87277, rnd e48cc90c)

**FAIL on the pass line, both devices, both check types.** Read the gold caveat above first: the gold is
provisional, and this result says nothing about the keys.

| check type | positives entail (n) | fooled-unsupported entail (n) | Wilson upper | pass |
|---|---|---|---|---|
| grounded | 0.210 (267) | 0.073 (110) | 0.137 | no |
| reasoning | 0.071 (210) | 0.029 (68) | 0.101 | no |

The fooled-unsupported half holds; the positives half fails by a wide margin. The model labels most real supporting
code lines "neutral". Strata don't rescue it (grounded code_only 0.153, doc_comment 0.208; reasoning 0.087, 0.053).
NPU and iGPU agree on 676 of 677 labels (corrected 2026-10-28; see the run note); the iGPU is 3× faster (p50
0.089 s against 0.271 s). Both runs whole and healthy: 677 scored, no stop, no limit, NPU OK before and after.
Receipts and the full run note:
`docs/runs/receipts/2026-10-10-smoke/`.
