# Stage 1 G0: the machine-drift check (run note, 2026-10-16)

**What:** the base model re-ran DEV (pilot 30) twice at A_0's sample seed 0, through the determinism wrapper (M1 and
M2). Both runs were compared token for token with each other and with A_0's stored seed-0 run. This tests whether
the driver move from 617.14 (A_0, 2026-10-10) to 617.42 (2026-10-12), together with the wrapper's deterministic
settings, changed the numerics. The reading rules were fixed before the run (rig plan §3, `compare_raw.py`). The
plan was reviewed by Argon before it ran. P0, the panel baseline, isn't part of this grant: it waits for B3's design.

**Verdict: (a). M1 = M2 = A_0 on all 30 tasks.** There is no drift on this stack, so round 1 proceeds on the
existing A_0 and no re-baseline (G0b) is needed.

## Inputs

| Item | Value |
|---|---|
| Code | branch `verifier-distill-prereg-draft` at `96acdaa` |
| Student | Qwen/Qwen3-8B `b968826d9c46`, bf16, SDPA; the harness verified all 12 pinned files |
| Harness | sha `8d63a541`, harness.py `4fa52448`, unchanged |
| Tasks | pilot-30 = review-10 then pilot-20, sha256 `3f5987da`, the same file as A_0 |
| Decoding | sampled, T 0.6, top-p 0.95, top-k 20, cap 16,384, thinking on, seed 0, phase A-before, arm BASE |
| Wrapper | det_run.py `add44dab`: strict deterministic algorithms (not warn-only), cuDNN deterministic, benchmark off, CUBLAS `:4096:8`, `--sdpa flash` |
| SDPA pin | `flash`, from sdpa_check.py the same day: default, flash and flash+efficient each ran twice equal on `rv-bc9c8c5439` at seed 0, and all equalled today's default and A_0 |
| Reader | compare_raw.py `8485346e` |
| Rig | one grant, other sessions idle, no Ollama model loaded; driver installs set to No by the maintainer for the machine freeze |

## The frozen machine line

M1 wrote it with `--write-machine-line`, and M2 checked it with `--machine-line` before loading the model. It is
committed at [`stage1-machine-line.json`](stage1-machine-line.json), sha256 `2daa5aa9`, and pinned in run sheet §9:

```json
{"gpu": "NVIDIA GeForce RTX 5090", "driver": "617.42", "cuda": "13.4",
 "torch": "2.16.0.dev20261008+cu134", "transformers": "5.19.0"}
```

Every later batch-1 sitting checks it, and any difference stops the sitting (exit 5). It changes only by a dated
amendment with a new G0.

## Results

| Run | Exit | Warned ops | Outcome |
|---|---|---|---|
| M1 | 0 | 0 | exit 0 |
| M2 | 0 | 0 | exit 0 |
| compare_raw | 0 | n/a | (a) |

compare_raw's reading (text, prompt_tokens, new_tokens and truncated on every turn; timing never compared):
- M1 against M2: equal;
- M1 against A_0: equal;
- M2 against A_0: equal.

The two raw files differ in bytes only through their `seconds` fields, which the comparison leaves out by design.
Card time was about 23 minutes, and the card returned to its idle baseline afterwards.

## What this settles

- **A_0 stands as the baseline**, including `A0_RAW_SHA256` and `A0_H4` in apply_rules.py.
- **The wrapper's deterministic settings, SDPA flash and driver 617.42 reproduce A_0's tokens exactly** on the 30
  DEV tasks. The SDPA check found the same on its one task.
- **The machine line is frozen** from here to round F.
