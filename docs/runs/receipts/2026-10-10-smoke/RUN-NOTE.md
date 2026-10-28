# Smoke test S1: run note (R&D, 2026-10-27)

**Read this first.** The gold at rnd `e48cc90c` is provisional: Claude-written synthetic items, not human-labelled,
with 7 tune cannot_tell items that most calibration runs disputed. S1 tests one thing, whether an off-the-shelf NLI
cross-encoder separates accepted quotes on supported claims from accepted quotes on unsupported ones. **An S1 pass
would not have made the critic validated, and this S1 did not pass.**

## Result: FAIL on the pass line, both devices, both check types

Pre-registered line: entailment ≥ 0.85 on positives; Wilson 95% upper bound ≤ 0.15 on fooled-unsupported. Judged on
the NPU run.

| check type | positives entail (n) | fooled-unsupported entail (n) | Wilson upper | cannot_tell entail (n, not gated) | pass |
|---|---|---|---|---|---|
| grounded | **0.210** (267) | 0.073 (110) | 0.137 | 0.000 (6) | no |
| reasoning | **0.071** (210) | 0.029 (68) | 0.101 | 0.000 (16) | no |

The fooled-unsupported half of the line holds on both check types. The positives half fails by a wide margin: the
model labels most real supporting code lines "neutral" (498 of the 677 pairs on the NPU). That is the
prose-versus-code gap the pre-registration named, and the 5-pair rehearsal had already shown it (2 of 5 neutral).

**Strata (reported, never gated):** doc-comment pairs did not rescue the positives. Grounded: doc_comment 0.208,
code_only 0.153, unflagged 0.302. Reasoning: doc_comment 0.053, code_only 0.087. So the failure is not masked by
prose comments; it is general.

## Devices

| device | p50 / p95 per pair | labels agreeing with the NPU |
|---|---|---|
| NPU (Intel AI Boost) | 0.271 s / 0.276 s | — |
| Intel iGPU | 0.089 s / 0.094 s | 676 of 677 pairs, 99.9% |

Both runs: 677 pairs scored, nothing stopped, no `--limit`, NPU health OK before and after, exit 0, stdout `S1-OK`
(the script's "whole, healthy run" marker; it is not the pass line). The iGPU is the faster agreeing device. Whether a
critic's device matters is moot for this model, since it fails the line on both.

## Provenance

- aspire-si `6c8727708d5f453b0fd5dae7489bebf85ed48ca8`; script sha256 `c6c1778edd0859c1…`.
- Model `cross-encoder/nli-deberta-v3-base` @ `6c749ce3…`, static 1×512, batch 1.
- rnd `e48cc90c9a4cfca9472973308b5abd85195f15dd` (a detached worktree; rnd main had moved in `verifier-gold`). Gold
  sha256: grounded `59601d10…`, prs `6e2d068e…`, diffs `4f4c058a…`; 12 verdict files, each hashed in the receipt.
- Window 21:15:14Z to 21:20:20Z. Preflight: no npu-serve, 5090 untouched, `CUDA_VISIBLE_DEVICES=""`.
- Receipts (sha256): `s1-NPU.json deebfe2b…`, `s1-NPU-labels.jsonl 66eb6933…`, `s1-iGPU.json 843efac1…`,
  `s1-iGPU-labels.jsonl e70cc6df…`, `preflight.txt ced537a7…`, `run.log b2750f2f…`.
- Operator note: `--out` takes a directory, so the receipts were written under `run-2026-10-27/s1-<device>.json/`.
  They are copied flat into `docs/runs/receipts/2026-10-10-smoke/` byte for byte.

## What this does and doesn't show

- It shows that a prose-trained NLI cross-encoder, used as-is, does not read code-line support as entailment. S1 was
  built to measure exactly that, and the answer is "not even close" (0.07–0.21 against 0.85).
- It does not validate or invalidate the gold keys, and it says nothing about cannot_tell or removed-fact traps.
- It does not test S2 or S3, which are separate pre-registrations and separate requests.
- Nothing here is a critic score, and nothing is fed to any reward.

## Correction (2026-10-28)

The first version of this note gave two counts on the wrong key: "473 of 637 distinct labels" (NPU neutral) and
"636 of 637 distinct (claim, quote) keys" (NPU-iGPU agreement). Both were counted on (claim, first 80 characters of
the quote), which merges pairs whose quotes share an opening. On the full (claim, quote) key, both labels files hold
677 distinct pairs: the NPU labels 498 of them neutral (the iGPU 499), and the two devices agree on 676 of 677. The
figures above are corrected. Nothing else changes: the readout (positives 56 of 267 and fooled 8 of 110 grounded, the
reasoning rows, FAIL on both devices) comes from the script's own output and never used that key. The mismatch was
found in a review of a newsletter draft, when 637 could not be reconciled with the 677 rows in the labels files, and
confirmed by recounting the labels files at this commit.
