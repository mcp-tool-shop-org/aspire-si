# Smoke test S1: an NLI critic on the Intel side (pre-registration, 2026-10-10)

S1 is the first smoke test of the verifier plan (`docs/runs/2026-10-10-verifier-distill-prereg-draft.md`,
draft PR #79). It tests plumbing and one capability, not a training result. The Publisher has cleared S1–S3
to run before the maintainer picks the supervision option.

**The question:** can `cross-encoder/nli-deberta-v3-base` (revision `6c749ce3…`, Apache-2.0) tell a real
supporting quote from one that fooled a verifier? If it can, it earns a small, gated bonus in GRPO's micro
layer. If it can't, the bonus is dropped and nothing else changes.

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

## Pass line, per check type (fixed now)

- entailment on positives **≥ 0.85**;
- entailment on gold-unsupported fooled pairs with a Wilson 95% upper bound **≤ 0.15**.

**The rules around it:**
- Gold-cannot_tell fooled pairs are reported and never gated.
- An incomplete run (a timeout or a stopped device) never passes.
- The bonus is enabled only for check types that pass. A reasoning fail is expected, because one quoted
  line rarely entails a conclusion about a diff.

## Devices and safety

- **Primary run: the NPU.** That's the plan's device, and the Publisher measured 0.27 s a pair there at
  batch 1, parity-clean.
- **Second run, in the same quiet-CPU window: the Intel iGPU.** The intel-npu notes show DeBERTa NLI is
  about 6× the CPU on the iGPU and no faster than the CPU on the NPU. Both results are reported:
  - label agreement between the two devices;
  - p50 and p95 latency on each.

  The faster device that agrees becomes the critic's device. The pass line is judged on the NPU run.
- **The safety shape:** batch 1, static shape 1×512, devices chosen by name (never the RTX 5090).
  - The NPU's health (`Get-PnpDevice 'Intel(R) AI Boost'`) is checked before and after; the run refuses to
    start unless it reads OK.
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

- `s1-NPU.json` and `s1-iGPU.json` (readout, latency, health, device names, file hashes);
- `s1-<device>-labels.jsonl`.

They're committed byte for byte under `docs/runs/receipts/2026-10-10-smoke/` after the run, identity-scanned.
