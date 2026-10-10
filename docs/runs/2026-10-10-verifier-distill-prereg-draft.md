# DRAFT pre-registration: an 8B verifier trained to think when it needs to (SFT, then GRPO)

**Status: draft for review, 2026-10-10. Nothing runs, and no card or pod time is booked.**
- ASPIRE (this repo) owns the training mechanics.
- The R&D session owns the gold set, the splits and calibration.
- The Publisher grants card and pod time.
- The maintainer chooses an option and gives the go on the final version.

## The goal

The maintainer's direction is "a bigger model made into an 8B" verifier for offrig. It should have
**adaptive thinking trained in**: the student learns when to answer at once and when to think long. The
alternative is bolting that behaviour on at inference time.

**The baseline the student must beat** is offrig's post-hoc "think before accepting" policy. It asks with
thinking off; an "unsupported" stands; "supported" or "cannot_tell" is re-asked with thinking, and a loop
falls back to cannot_tell.
- Replayed on gemma4:31b over tune, it gives a false-accept (FA) upper bound of 0.026 grounded and 0.058
  reasoning, at about 87% of always-on cost.
- It's still post hoc, pending confirmation on grounded held-out. The replay combined two offrig builds,
  so **the baseline of record is the policy as offrig serves it natively**, measured once it ships. The
  replay numbers are context only.
- The same policy applied to the **untrained student** is the fair comparison: it isolates what training
  adds over the gate.

## The student

**Qwen3-8B**, licensed Apache-2.0 (card checked 2026-10-10), revision `b968826d9c46…`, pinned at plan time.
- It has a native thinking switch, so it can be taught when to use it.
- The R&D session's study notes say 8B students learn from long traces; at 3B or below they don't.
- It's packaged through our own pipeline: merge, GGUF, quantise, template test, quant-drift check, as in
  the distill-ladder plan.
- Distill-ladder leg 0 (the 1.7B math rehearsal) stays this plan's smoke gate.

## The options for supervision (the maintainer picks)

All the options share the reward and the evaluation below. They differ only in where the SFT traces come
from. GRPO needs no teacher in any of them.

| Option | SFT traces from | Licence | Status |
|---|---|---|---|
| **A: self-distillation (STaR-style)** | The student itself: k samples per tune claim (k = 8, temperature 0.7, thinking on), keeping only gold-correct ones | Apache-2.0 throughout | Ready to specify. Sidesteps the licence question |
| **B: a larger Apache-2.0 teacher** | Gold-filtered traces from a teacher that first passes the teacher bar (FA upper ≤ 0.07 on tune and on held-out) | Candidates below, all Apache-2.0 per their cards | Needs teacher calibration first |
| **C: gemma4:31b with the think-before-accepting policy as teacher** | That policy's traces, gold-filtered | Apache-2.0: **licence-eligible** (see below) | Needs the teacher bar cleared by the policy, confirmed |

**Option B candidates** (licences read from the Hugging Face cards, 2026-10-10, all apache-2.0, none gated):
- Qwen3-32B: local on the 5090;
- Qwen3-30B-A3B: local;
- Qwen3-235B-A22B-Thinking-2507: offrig;
- openai/gpt-oss-120b: offrig, one RTX PRO 6000;
- openai/gpt-oss-20b: local;
- allenai/OLMo-2-0325-32B-Instruct: local;
- mistralai/Magistral-Small-2509: local.

None has been calibrated yet. qwen3:8b and qwen3:14b miss the bar: qwen3:14b with thinking on has an FA
upper of 0.089 grounded and 0.204 reasoning.

**Option C is licence-eligible.** Gemma 4 is Apache-2.0, checked on 2026-10-10 against three sources:
- the Hugging Face card for google/gemma-4-31B-it (`apache-2.0`, linking
  ai.google.dev/gemma/docs/gemma_4_license);
- the licence text shipped inside the local gemma4:31b (Ollama `api/show`);
- the Publisher's independent check of that text.

An earlier "Gemma Terms of Use" note was wrong and has been corrected. Option C's open question is the
**teacher bar**, not the licence:
- gemma4:31b with thinking on alone misses it (FA upper 0.0704, against 0.07);
- with the think-before-accepting policy it may clear it, but only once offrig serves the policy natively
  and it's confirmed on grounded held-out.

- Nemotron (NVIDIA Open Model License) stays out under any reading.

## The data

- **Tune split (gold): training data.** Using it for training makes every calibration number on tune
  in-sample from here on. That's stated now, and no claim is read from tune afterwards.
- **Model selection** (the SFT epoch, the GRPO checkpoint) uses a dev split carved from tune before
  training. It's 20%, grouped by source (by PR or by file) so no source sits on both sides, and frozen by
  sha256 before any training.
- **More GRPO prompts by construction** (optional, pre-registered here as an option):
  - grounded claims made from gold evidence by rule: true claims; false ones by a changed number, name,
    condition or negation; cannot_tell by removing the deciding evidence;
  - labels true by construction, deduped against every test split by normalised source;
  - kept to at most half the GRPO prompts, because constructed claims are easier (tier T1).
- **Tests, each used once, for the final student only:**
  - **Grounded:** the held-out split is unspent but reserved first for confirming the policy. The student
    gets one look, agreed with the Publisher, or a fresh sealed grounded split.
  - **Reasoning:** the held-out is spent, so this uses the fresh sealed split planned for contract v3.
  - **Math ladder:** v2, sealed (a transfer check).

## The reward (GRPO): one rubric at three scales, every component checked by rule

### Source, and what it does and doesn't support

The maintainer's "fractal reward" note came as an AI-written summary. The one research link in it is
TIGER-AI-Lab's **RationalRewards** (Wang, Wei, Ren, Chen, Liu, Lin; arXiv:2604.11626, a 2026 preprint),
read 2026-10-10.
- **What it is:** an 8B *learned* reward model for **text-to-image generation and editing**. It writes a
  critique on several dimensions before scoring, and the score drives diffusion RL.
- **Its anti-hacking claim:** a justified, multi-dimensional critique acts as an implicit regulariser.
- **What it is not:** it describes no multi-level or self-similar rubric. Its page gives no quantitative
  hacking measurements.
- The summary's broader claims (for example, "exponential coverage with linear compute") are **unverified**
  and aren't relied on here.

**What carries over:** a reward that has to be *justified part by part* is harder to inflate than a bare
score. This draft gets that by **rule**, not with a learned critic: the verifier's own contract is already
self-similar, with a verdict over evidence quotes. Learned process rewards were hacked in Gao et al. 2024,
*On Designing Effective RL Reward at Training Time for LLM Reasoning* (arXiv:2410.15115).

The self-similar layout is this plan's design choice. It isn't a claim that "fractal rewards are proven to
work". The ablation below tests whether the meso level helps.

### The three levels

**The same rubric at every level:** right, grounded, thought spent only where needed.
- **Macro:** the claim.
- **Meso:** its parts.
- **Micro:** each quote and each step.

**Contract v3's `evidence_quotes`** (1–4 entries) carries the parts. For a conjunctive claim the student
gives each part as `{part, verdict, quote}`. A simple claim is one part.

| Level | Component | Rule (checked by code) | Value |
|---|---|---|---|
| **Format** | gate | the reply isn't schema-valid contract v3 JSON | the sample scores **−1.0**; nothing else applies |
| **Macro** (claim) | outcome, asymmetric, matching offrig's default rule | correct verdict (on gold cannot_tell, "unsupported" or "cannot_tell" both count) | **+1.0** |
| | | false accept | **−3.0** |
| | | false reject | −1.0 |
| | | cannot_tell on decidable gold | −0.3 |
| **Meso** (part) | **consistency** (needs no part gold) | the claim verdict disagrees with the three-valued part rule (any part unsupported → unsupported; else any part cannot_tell → cannot_tell; else supported) | **−0.5** |
| | part correctness (only where part gold exists, see below) | each part verdict correct | +0.05 each |
| | | a part called supported whose gold is unsupported | **−0.5** each (a part-level false accept) |
| **Micro** (step) | quote rule 2 (offrig's checker) | any part called "supported" with a quote not found in the evidence | that part is a false accept: the claim's outcome becomes **−3.0** if it was "supported" |
| | | each found quote on a correct claim verdict | +0.05 each |
| | thinking steps | **measured, not rewarded:** the share of thinking paragraphs that name a part or quote | reported per arm. Rewarding "looks like it's checking" is the hackable kind of process reward |
| **Across all levels** (token) | thinking length (correct samples only) | above the soft budget B (256 when the verdict is "unsupported", 1,024 otherwise: "think before accepting", trained in) | −0.1 × (tokens − B) / B, capped at −0.5 |
| | loop (offrig's detector: an 8-word n-gram repeated 40 times) | a loop detected | **−0.5**, and the verdict is read as cannot_tell, as offrig does |
| | easy and quick | correct, on a tier-T1 item, with ≤ 64 thinking tokens | +0.2 |
| | hard cap | truncation at 8,192 thinking tokens | −1.0, scored as a failure |
| **Caps** | positive non-macro credit (part, quote, easy) | **only on a correct claim verdict**, total per sample | **≤ +0.3** (the Gao et al. clipping fix) |
| | meso penalties (consistency, part false accepts) | total per sample | **≥ −0.5** |
| | all non-macro penalties (meso, length, loop) | **on a correct claim verdict**, total | **≥ −1.0**, so a correct verdict never scores below 0.0 |

**How the components combine: gated, not a plain sum.** In full:

```
R = −1.0                                                         if the reply fails the format gate
R = macro + max(penalties, −1.0) + min(extras, 0.3)              if the claim verdict is correct
R = macro + penalties                                            if it's wrong (no extras at all)
```

Here `penalties` are the meso, length and loop terms (meso is floored at −0.5) and `extras` are the part,
quote and easy credits. A positive micro or meso score can never mask a macro failure. A claim-level
false accept always takes the full −3.0, however many quotes or parts scored well. RationalRewards' page
doesn't describe how it aggregates its dimension critiques, so this gate is this plan's design, not
borrowed.

**Invariants, checked by property tests before any run** (a brute-force check of these weights over every
verdict, part pattern, quote, length and loop case holds them: best wrong −0.3, worst correct 0.0):
- no wrong claim verdict ever scores above −0.3;
- every correct verdict scores at least 0.0, so a correct verdict always beats a wrong one;
- positive extras never exceed +0.3.

**What the macro weights imply:** a calibrated student says "supported" only when it's at least 0.675
sure, and "unsupported" at 0.35 or more. The abstain target is ≤ 0.20 on decidable dev items. Above that,
the run stops and is reported. The −0.3 isn't tuned on the spot.

### Gold for the parts: what's derivable, and the cost (the audit comes first)

The tune split has 493 claims. **221 are conjunctive** (and, both, as well as).

| Conjunctive tune claims | n | Part labels |
|---|---|---|
| supported | 117 | **derived:** every part is supported, given a checked split |
| unsupported, with a supported twin differing in one span of ≤ 3 words **on both sides** | 25 | **derived:** the differing part is unsupported, the rest supported |
| unsupported, twin differs in several spans or a longer span, or no twin | 76 | **need new labels** |
| cannot_tell | 3 | **need new labels** |

The R&D session replicated the derivation independently with a word-level diff and found the same 25.
A first count of 29 bounded only the supported side of each pair. The 4 extra pairs insert 5–11 words on
the unsupported side, so they move to "need new labels".

**The 25 twin-derived ids** (tune, rnd at `d602719`): `cal-corrections-f`, `boost-cap-f`, `jury-min-f`, `jury-defaults-f`, `or-sidecar-f`, `or-reserved-f`, `prs-aspire-si-53-12u`, `prs-aspire-si-55-4u`, `prs-aspire-si-55-6u`, `prs-aspire-si-59-1u`, `prs-aspire-si-61-5u`, `prs-aspire-si-61-13u`, `prs-offrig-39-2u`, `prs-offrig-39-3u`, `prs-offrig-39-24u`, `prs-offrig-40-9u`, `prs-offrig-41-11u`, `prs-offrig-41-15u`, `prs-role-os-24-9u`, `diff-aspire-si-13-1u`, `diff-aspire-si-47-3u`, `diff2-role-os-0a13ed4-2u`, `diff2-role-os-54f6a32-5u`, `diff2-aspire-si-4d45a08-3u`, `diff2-aspire-si-4d45a08-8u`.

**The shared-constituent rule, applied before any derivation is trusted.** A split copies a shared object
or modifier into every part it governs. Example: `prs-offrig-40-9u`, "Network volumes are listed and
created under /networkvolumes". Here "under /networkvolumes" governs both verbs, so the parts are
"listed under /networkvolumes" and "created under /networkvolumes". Without the copy, a derived
"supported" part would be wrong.

- **Splits:** every conjunctive claim needs one checked split into parts. It's proposed by rule (on the
  conjunction), and a model proposes where the rule fails. It's reviewed blind under the gold set's usual
  process (the R&D session owns it).
- **Cost:** about **221 split reviews**, plus about **79 claims × ~2.5 parts ≈ 200 new part labels**.
  The cost is labelling time under the R&D session's blind-label process, with no card time.
- **A caveat on the 25 derived from twins:** the derivation assumes the differing span sits inside
  exactly one part. Each one is confirmed against its checked split, and the R&D session spot-checks a
  sample, before any of them is used.
- **Recommendation: phase it.**
  - **Phase 1 needs no new labels.** It uses macro, consistency, micro and thinking everywhere, with part
    correctness only on the **142 derivable claims**.
  - **Phase 2 adds the 79 labelled claims** only if Phase 1's ablation shows part correctness helps.

### The ablation that tests the idea (fixed now)

GRPO runs in two arms from the same SFT checkpoint with the same seeds.
- **Macro + micro + thinking:** no meso terms.
- **The full three-level reward.**

**"Self-similar reward helps"** means the full arm's dev FA upper bound is lower at equal or lower
thinking cost, on all three seeds. The result is reported either way.

### Self-verification (from S²R)

Ma et al. 2025, *S²R: Teaching LLMs to Self-verify and Self-correct via Reinforcement Learning*
(arXiv:2502.12853), train a model to check its own answer and revise it. A verifier's job is checking, so:
- **In SFT (option A):** traces where the model's own thinking corrects an earlier verdict are kept and
  reported as a share of the set. They aren't over-weighted.
- **In GRPO:** the scored verification is the quote rule and the consistency term. There's no reward for
  verification-shaped text.
- **Measured:** the self-revision rate, and the share of revisions that end correct, per arm.

### GRPO settings (fixed before any run)

- **Group size G = 8,** sampled at temperature 0.7. Group-normalised advantage; the share of zero-variance
  groups is reported.
- **G = 8, β = 0.04 and ε = 0.2 are the usual GRPO defaults,** pre-registered as defaults for a first run,
  not as tuned values. Changing any of them needs an amendment.
- **Evaluation decoding is at serving temperature, 0,** as offrig serves. Every checkpoint's dev **loop rate
  at T = 0** is reported. Loops are a greedy-decoding failure that distilled students show more (Pipis et
  al. 2025, arXiv:2512.12895).
- **KL:** β = 0.04 against the SFT checkpoint as the reference, with a ratio clip ε = 0.2. With little
  training data, staying near SFT is deliberate.
- **Updates:** LoRA (r = 16, α = 32 on q/k/v/o), learning rate 1e-5, 2 epochs.
- **Checkpoint choice:** on the dev split only: the lowest FA upper bound at abstain ≤ 0.20.
- **Reward-hacking stops** (on dev, at every evaluation step):
  - abstain above 0.20;
  - any one verdict above 70%;
  - mean thinking under 32 tokens with accuracy falling;
  - positive extras above 25% of total reward;
  - more than 4 parts per claim on average (splitting to farm part credit);
  - the dev loop rate at T = 0 rising above the SFT checkpoint's.

  Any of these stops the run, and the stop is reported.

## Evaluation

Every arm is scored with `offrig verify calibrate` under the same default rule, on the sealed tests,
once.
- **Arms:**
  - the base Qwen3-8B with thinking on;
  - the base with thinking off;
  - the base with the think-before-accepting policy;
  - SFT only;
  - SFT + GRPO.
- **Primary readings, fixed now:**
  - the FA upper bound;
  - abstain on decidable;
  - false rejects;
  - balanced accuracy;
  - thinking tokens per claim, per difficulty tier. Adaptive thinking means short thinking on T1 and long
    on T2/T3.
- **"Training works" means** SFT+GRPO beats **the base with the policy** on FA upper (the default rule)
  at equal or lower thinking cost, on all three seeds.
- Packaging is checked by the quant-drift test before any scoring.

## Seeds, SFT and trace generation

**Seeds:** 3 for any claim, which is the standing rule. A one-seed pilot is labelled a pilot, never a
result.

- **SFT (5090):** an 8B bf16 LoRA with gradient checkpointing. Peak memory is measured in the smoke gate,
  under the watchdog line. About 0.5–1 h per seed.
- **Option A trace generation (5090):** about 395 tune claims (493, less the dev split) × 8 samples × ~1.5k
  tokens ≈ 4.7M tokens. On llama.cpp or Ollama, that's about 2–4 h.
- **Option B adds** teacher calibration and trace generation. That cost depends on the candidate: local is
  free; offrig is priced by `offrig_plan`.

## Compute: local first, with an NPU critic (preferred); the pod as fallback

**The maintainer's direction:** train locally, with an NPU model supervising. His thesis is that well-timed
critique and assistance matter more than the supervisor's knowledge, provided it has tools and a database to
refer to. This plan makes that the preferred path, keeps a rented pod as the fallback, and **tests the thesis
with an ablation**.

### The division of labour

- **The 5090 trains:** QLoRA GRPO on the 8B student, with local rollouts, in the CUDA 13.4 env.
- **The NPU critics:** dense, well-timed signals and hints.
- **The correctness reward stays mechanical:** gold labels plus offrig's quote rule. **The critic never
  decides correctness,** and nothing it says can turn a wrong claim verdict into a positive reward. The
  gated combination and its invariants above still hold.

### What the NPU critic does

1. **Micro-level entailment: a capped bonus inside the gate.**
   - **The model:** nli-deberta-v3-base on the NPU, batch 1 (about 0.27 s per pair, measured parity-clean).
   - **The check:** whether each quoted span entails the part it's cited for.
   - **The reward:** +0.05 per entailing quote, **only on a correct claim verdict**, and inside the
     existing +0.3 extras cap. It's never a penalty, and never on the outcome.
   - **Why only that:** switchyard's rows show this model is a weak *claim* judge on our gold (FA 0.19–0.22
     grounded, balanced accuracy ~0.72). A short quote against one part is a more literal task, but it's
     **unmeasured**. So smoke test S1 measures it first, and the bonus is used only if S1 passes.
   - **Monitored:** its agreement with the quote rule. A quote the rule rejects but NLI "entails" is logged.
2. **Hints on failure: tool hints in GRPO, gold hints only in SFT.**
   - **Tool hints in GRPO.** When a rollout group is all wrong, the critic runs offrig's quote check and
     retrieval over the claim's evidence and inserts **a tool-derived hint for one retry**. For example:
     "the quote for part 2 is not in the evidence"; "these lines decide the claim". It **never** includes
     the gold label.
     - The retry is still scored on gold.
     - Hinted samples change the conditioning, so by default they're **logged and scored but excluded from
       the gradient**. Training on them is a separate pre-registered arm, guided-GRPO style.
   - **Gold hints in SFT (STaR rationalisation).** Zelikman et al. 2022, *STaR*, arXiv:2203.14465:
     - when a gold-label hint produces a correct trace, that trace goes to the **SFT** set for option A;
     - it's marked as rationalised, and its share is reported.
3. **A small LLM coach on the NPU, only if switchyard's E3 measures usable NPU decode speed.** It isn't
   assumed. Without E3's receipt, there's no coach.

### NPU safety (from the R&D session's and Kimi's incidents)

- **Batch 1 only.** Batches above 1 have hung the NPU: nomic at 8×1024, 3/3; DeBERTa at 8×512.
- **A health check before and after** every critic session.
- **A timeout on every call,** using Kimi's probe-timeout work.
- **A hang or timeout drops that step's critic signal.** The component scores 0 and is logged; it **never
  stalls the trainer**. The critic runs asynchronously from the training loop, and a missed deadline means
  no signal, not a wait.
- **The critic-signal drop rate** is reported per run. Above 20%, the critic arm's result is reported as
  "critic unreliable" and not read.

### Devices and grants

- **One grant covers everything:** the 5090 and the NPU together, with the CPU kept quiet enough for the
  NPU, granted once by the Publisher. These are long windows, typically overnight.
- **switchyard:** the plan is to dogfood its harness and guards for device allocation where they support a
  two-device run. If they don't yet, that's a switchyard gap, filed with the R&D session, not worked around.

### Memory on 32 GB, estimated honestly and measured in smoke test S2

| Piece | Estimate |
|---|---|
| Qwen3-8B in 4-bit (QLoRA base) | ~5.5–6 GB |
| LoRA weights, gradients, optimiser (r = 16 on q/k/v/o) | < 1 GB |
| Training activations, G = 8 × (prompt ≤ 1.5k + completion ≤ 2k) with gradient checkpointing | ~6–10 GB |
| Rollout KV cache, 8 sequences × 3.5k tokens | ~4–5 GB |
| CUDA context and fragmentation | ~2–3 GB |
| **Total** | **~19–25 GB, under the 31.2 GB watchdog line, if one model copy serves both training and rollouts** |

**The risks, measured rather than assumed:**
- **vLLM, the usual fast rollout engine, isn't native on Windows.** On this rig, the rollouts are either:
  - Hugging Face `generate` on the 4-bit model (slow), or
  - a separate llama.cpp server (CUDA 13.4, already built) with the adapter reloaded each step: a second
    model copy (~5 GB at Q4) and adapter conversion time.

  S2 measures both, and picks the faster path that fits.
- **The levers if it doesn't fit:** G (8 → 6 → 4), the completion cap (2k → 1.5k), and prompts per step.
  Each change is an amendment, made before the run.

### Wall-clock and cost

**The measure:** about 500 GRPO prompts × G = 8 × ~800 completion tokens (the budgets keep easy claims
short) ≈ 3.2M generated tokens per epoch.

| Path | Throughput (assumed until S2) | Per epoch | 2 epochs × 3 seeds × 2 arms (critic ablation) | Spend |
|---|---|---|---|---|
| **Local 5090 + NPU (preferred)** | ~300–500 tok/s aggregate rollouts | ~2–3 h rollouts + ~1 h training | **~36–48 h of card**, spread over overnight windows | **$0 pod spend**; more wall-clock and card time |
| **Pod fallback** (1× RTX PRO 6000, vLLM) | ~1.5k tok/s | ~1 h | ~12 h | **~$25–30** at ~$2.09/h, plus the NPU arm can't run there (the critic is local) |

Because the critic is on the local NPU, **the critic ablation only runs locally.** The pod fallback covers the
no-critic arm, and only if the local path can't fit or is too slow.

### Smoke tests, each with its own grant, in this order

- **S1, NPU only, a quiet-CPU window, ~30 min.**
  - **What runs:** nli-deberta on the NPU at batch 1, over quote–part pairs built from the tune gold, using
    the 142 derivable conjunctive claims and their gold quotes.
  - **The pass line (fixed now):** entailment agreement with the quote rule and part gold of ≥ 0.85, and an
    FA on parts (entails a part whose gold is unsupported) with a Wilson upper bound ≤ 0.15.
  - **If it fails:** the micro-NLI bonus is dropped. The hint path doesn't depend on it.
- **S2, the 5090 only, ~45 min:**
  - a 20-step GRPO on 32 tune prompts, each rollout path, G = 8;
  - peak VRAM, tokens per second, step time;
  - no OOM, and under the watchdog line.
- **S3, the 5090 + NPU, one combined grant, ~1 h.**
  - **What runs:** S2's run with the async critic attached.
  - **Faults injected:** an NPU timeout and a missed deadline, to prove the trainer never stalls. The
    trainer's step time with the critic must be within 10% of S2's.
- **S4 (the pilot):** one seed, both arms, a few hundred steps. It's labelled a pilot, never a result.
  Then the 3-seed run.

### The critic ablation (the thesis test, fixed now)

Two arms run from the same SFT checkpoint with the same seeds (42–44): **mechanical reward only**, and
**mechanical reward + NPU critic** (micro-NLI bonus if S1 passed, plus tool hints).

**"Well-timed critique helps"** means the critic arm reaches a lower dev FA upper bound at equal or lower
thinking cost, or the same FA in fewer steps, on all three seeds. It's reported either way.

## Open choices before this becomes the pre-registration

1. **Option A, B or C** (the maintainer).
2. The grounded test: one look at the held-out, or a fresh sealed split (the Publisher and the R&D session).
3. The reward weights, budgets and GRPO settings. They're proposed here; the R&D session reviews them before any run.
4. The part gold: Phase 1 only (142 derivable claims), or Phase 2's ~200 new part labels as well. Labelling is the R&D session's, under its blind process.
5. Contract v3's per-part fields (`evidence_quotes` entries carrying part, verdict and quote) are agreed with offrig's owner before SFT data is built.
6. Where GRPO runs: local 5090 + NPU (preferred, the maintainer's direction) after smoke tests S1–S3, or the pod fallback for the no-critic arm.
7. The fresh reasoning split (contract v3) is sealed before any training.
