# Smoke test S2: GRPO memory and speed on the 5090 (pre-registration, 2026-10-27)

S2 is the second smoke test of the verifier plan (`docs/runs/2026-10-10-verifier-distill-prereg-draft.md`,
draft PR #79; the plan's smoke list says: "S2, the 5090 only, about 45 min: a 20-step GRPO on 32 tune prompts,
each rollout path, G = 8; peak VRAM, tokens per second, step time; no OOM, and under the watchdog line").
It measures the trainer's memory and speed on this rig. It is not a training result, and nothing it produces
is kept as a model. It does not depend on S1's result.

**The question:** does a 20-step GRPO on the base student fit on the RTX 5090 under the watchdog line, and how
fast does it go, on each of the plan's two rollout paths? The plan's memory table estimates 19 to 25 GB; S2
replaces the estimate with a measurement.

## What runs

- **20 GRPO steps**, each on 2 prompt groups of G = 8 samples at temperature 0.7, one group at a time with the
  gradients accumulated, so the peak memory is one group's. The 2 groups per step cycle through the 32 prompts,
  so every prompt is used at least once. This is a declared choice: the plan fixes 20 steps, 32 prompts and G = 8
  and leaves the prompts-per-step open.
- **The plan's GRPO defaults,** unchanged and pre-registered as defaults: LoRA r = 16, α = 32 on q/k/v/o,
  learning rate 1e-5, β = 0.04, ε = 0.2, one update per rollout batch. A completion cap of 2,048 tokens.
- **From the base student:** Qwen3-8B, Apache-2.0, at the plan's pinned revision `b968826d9c46…` (the plan
  wrote only that prefix; the run records the full commit it resolved). No SFT checkpoint exists yet, so the
  reference model for the KL term is the same base with the adapter off. The memory and speed figures do not
  depend on SFT.
- **Both rollout paths, each its own 20-step run:**
  - **(a) Hugging Face `generate`** on the 4-bit (QLoRA) model: one copy serves the training and the rollouts.
  - **(b) A llama.cpp server** (the rig's existing CUDA build; the run records the server's path and sha256)
    with the LoRA adapter exported to GGUF and reloaded every step. Control calls (health, adapter reload) have
    a **10 s** timeout; a generation call has **120 s** (2,048 tokens at the rates the plan assumes takes about
    40 s), a declared choice. A timeout stops the run.

## The prompts

- **The rule:** the first 32 claims of the `tune` split, by sorted claim id, from the three verifier-gold files
  of rnd at a pinned commit (`e48cc90c`, as S1). The script records the commit and each gold file's sha256 and
  the 32 ids.
- **Declared substitution:** the plan's APPLY slice is a seeded draw of 35% of the tune fact groups, and it is
  not frozen yet. S2 therefore takes tune claims by the stated rule, as the plan's own S2 line says ("32 tune
  prompts"). Memory and speed do not depend on which tune claims are used, and S2 trains nothing that is kept.
- **The prompt** is the plan's task in a fixed template: the reply contract (a verdict and 1 to 4
  `evidence_quotes` of `{part, verdict, quote}`), then the claim's evidence and the claim.

## The reward (mechanical only; no model judge, no NLI)

The plan's gated reward (`:336-412`), run in part. S2 measures memory and speed, so a reduced reward is allowed
if it is declared:

- **Run:** the format gate (-1.0); the macro outcome (+1.0 correct, -3.0 false accept, -1.0 false reject, -0.3
  cannot_tell on a decidable gold; on gold cannot_tell, unsupported also counts); the micro quote rule (an
  unfound quote on a part called supported makes a supported claim a false accept); +0.05 for each found quote on
  a correct verdict (extras at most +0.3); the meso consistency penalty (-0.5); the thinking-length penalty on
  correct samples (-0.1 × (tokens − B) / B, capped at -0.5, with B = 256 when the verdict is unsupported and
  1,024 otherwise); the loop penalty (-0.5, the verdict read as cannot_tell); the hard cap at 8,192 thinking
  tokens (-1.0). The terms combine as the plan's gate: a wrong verdict gets no extras and a correct one never
  scores below 0.0.
- **Not run, because the gold has no part labels or difficulty tiers:** part correctness (+0.05 each), the part
  false-accept penalty, and the easy-and-quick credit (+0.2 on a tier-T1 item).
- The plan's invariants (no wrong verdict above -0.3; every correct verdict at least 0.0) are tested over every
  case before any run.

## Measured, per path

- Peak VRAM: `nvidia-smi` sampled every second (the figure judged), and `torch.cuda.max_memory_allocated`
  (both recorded).
- Rollout tokens per second; training step time (p50 and max); wall-clock; the number of zero-variance groups.

## Pass line, per path (fixed now)

- no out-of-memory error;
- peak VRAM from the sampler **under 31,200 MiB**, the rig's watchdog line (at the line is not under it);
- **all 20 steps completed.**

**The rules around it:**
- An incomplete run never passes. A `--steps`, `--prompts` or `--g` cut, a stop, a timeout, a watchdog halt, or
  a sampler that read nothing cannot read `S2-OK`.
- The verdict line is `S2-OK` (a whole run that fits), `S2-FAIL` (a whole run that did not fit, or ran out of
  memory), `S2-LIMITED` (a cut), `S2-STOPPED` (a stop), or `S2-DRY` (the CPU rehearsal). Everything but `S2-OK`
  and a finished `S2-DRY` exits non-zero.

## Decision rule

The faster path that fits becomes the plan's rollout path (`:604`). If neither fits, the plan's levers (G,
the completion cap, prompts per step) are amendments made before the next run, never silent changes.

## Environment, recorded in the receipt

GPU name and driver, CUDA, torch, transformers, peft and bitsandbytes versions, dtype, batch, whether the GPU was
exclusive, the Python version and the platform.

**Python.** The rule is the newest stable Python (3.15). `torch` 2.14.1 (checked 2026-10-27) ships wheels for
`cp310` to `cp314` and no `cp315` wheel. So the scripts and their pure tests run on 3.15 with torch imported
only where used, the one tiny-model dry test runs on 3.14 locally (torch 2.14.1 ships no cp315 wheel), and the
CI runs it on 3.12; the real run uses the rig's `python-gpu` environment (Python 3.12, torch 2.14.1+cu130),
which is why its Python differs. No older Python was pinned as a workaround.

## Devices and safety

- **The VRAM watchdog must be alive before the start:** its heartbeat file's modified time (never opened) is
  under 60 s old, or the run refuses to start.
- **The single NVIDIA GPU only,** chosen by name. Two GPUs, or a non-NVIDIA one, refuse. The run refuses to start
  if any other process holds the 5090. `--dry` runs on the CPU and never calls CUDA.
- **Duration:** about 45 minutes in all, as the plan says.

## What S2 cannot show

Nothing about verifier quality, and nothing about S1's critic. The numbers are this rig's, at this shape.

## Rehearsal (done, CPU, no model download)

The `--dry` run ran end to end on the CPU with a 2-layer random Llama (real q/k/v/o LoRA targets) and a byte
tokenizer, no network: 2 steps, G = 2, the summary line below.

```text
{"path": "hf", "stopped": null, "steps_completed": 2, "peak_vram_mib_sampler": null, "rollout_tok_s": 1135.46, "step_s": {"p50": 0.1262, "max": 0.1374}, "zero_variance_groups": 0, "oom": false}
S2-DRY
```

The rehearsal ran on Python 3.14 with torch 2.14.1 (see the Python note above). Its figures are a tiny
model's and say nothing about the 5090.

## Receipts

`s2-hf.json`, `s2-llamacpp.json` (readout, environment, hashes, the 32 prompt ids) and the two
`s2-<path>-steps.jsonl` files. They are committed byte for byte after the run, identity-scanned.
