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

## Seeds and compute (estimates, measured in the smoke gate)

**Seeds:** 3 for any claim, which is the standing rule. A one-seed pilot is labelled a pilot, never a
result.

- **SFT (5090):** an 8B bf16 LoRA with gradient checkpointing. Peak memory is measured in the smoke gate,
  under the watchdog line. About 0.5–1 h per seed.
- **Option A trace generation (5090):** 458 tune claims × 8 samples × ~1.5k tokens ≈ 5.5M tokens. On
  llama.cpp or Ollama, that's about 2–4 h.
- **GRPO:** about 1,000 prompts × 8 samples × ~1.5k tokens ≈ 12M generated tokens per epoch.
  - **On a rented RTX PRO 6000 at ~$2.09/h:** about 3 h per epoch, so 2 epochs × 3 seeds ≈ 18 h ≈ **$38**.
    That's above the RunPod balance last reported (about $36.79).
  - **On the 5090 overnight:** free, but 2–3× slower, and the memory fit is unproven.
  - **So:** GRPO is a one-seed pilot first, then the 3-seed run on whichever path the pilot's measured
    numbers support. That's for the Publisher and the maintainer to decide.
- **Option B adds** teacher calibration and trace generation. That cost depends on the candidate: local is
  free; offrig is priced by `offrig_plan`.

## Open choices before this becomes the pre-registration

1. **Option A, B or C** (the maintainer).
2. The grounded test: one look at the held-out, or a fresh sealed split (the Publisher and the R&D session).
3. The reward weights, budgets and GRPO settings. They're proposed here; the R&D session reviews them before any run.
4. The part gold: Phase 1 only (142 derivable claims), or Phase 2's ~200 new part labels as well. Labelling is the R&D session's, under its blind process.
5. Contract v3's per-part fields (`evidence_quotes` entries carrying part, verdict and quote) are agreed with offrig's owner before SFT data is built.
6. Where GRPO runs: the pod, or the 5090 after its fit check.
7. The fresh reasoning split (contract v3) is sealed before any training.
