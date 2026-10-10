# DRAFT: Stage 1 training procedure (a short pre-registration)

**Status: draft, 2026-10-10. Nothing trains until the maintainer approves this and the Publisher grants the
card.** It sets how the approved lessons (Lessons 1–3 and the 18 trait lessons) become training, using the
loop and ledger in the lesson plan ([stage1-verifier-qualities-lessons-draft](2026-10-10-stage1-verifier-qualities-lessons-draft.md)).
Measurement is in the [evaluation plan](2026-10-10-stage1-evaluation-plan-draft.md). Where the two drafts
differ, this one says so and why.

## The student and the trainer

- **Student:** Qwen/Qwen3-8B at `b968826d9c46`, loaded by the same `load_student()` the eval harness uses
  ([stage1-eval/student.py](stage1-eval/student.py)). bf16 base weights, frozen. The weights' sha256 are
  checked before every load.
- **Trainer:** a self-contained loop in this repository: torch and peft, which are already in the training
  environment. No new installs, no environment changes, and no TRL (the maintainer's decision, relayed
  2026-10-10).
- **Adapter:** LoRA on every linear layer (q, k, v, o, gate, up, down), rank 64, alpha 128, dropout 0.05.
  Rank 64 on all layers follows the evidence that LoRA matches full fine-tuning when it covers every layer and
  the data fits its capacity (Thinking Machines 2025, *LoRA Without Regret*), and that it forgets less of the
  base model (Biderman et al. 2024, *LoRA Learns Less and Forgets Less*, arXiv:2405.09673). A few thousand
  examples are far inside rank 64's capacity.
- **Optimiser:** AdamW, learning rate 1e-4, cosine decay, 3% warm-up, weight decay 0. Effective batch 16
  sequences (gradient accumulation), maximum length 4,096 tokens, gradient checkpointing on.
- **Loss:** on the assistant turn only (the thinking and the three-line answer). The lesson text and the
  material are context, not targets.
- **Determinism:** fixed seeds for data order and dropout. Every session writes its data manifest
  (sha256 of each example), the adapter's sha256, and the loss curve. A session is resumable from its last
  checkpoint.

## What one session is

A session is one pass through the loop for the whole curriculum. Each lesson goes through four steps.

1. **Teach.** SFT on the lesson's curriculum: five kinds of item per trait, as the Kimi handoff and the
   [schema](stage1-training-data-schema.md) set out.
   - **A:** identity and purpose, in the model's own words;
   - **B:** thinking patterns, varied, some not verification at all;
   - **C:** contrasting cases;
   - **D:** the role in action, with self-talk in the thinking;
   - **E:** quiz items with a variant.

   **The training target is the whole assistant reply, for every kind:** its thinking, explanation,
   analysis, self-talk and answer. That's the maintainer's method. The role-shaping content itself is what's
   learned, not only verdict production. The user turns are context, and the loss is masked to the assistant
   turns. Each item carries its own context (Kind A brings its own short teaching passage). The lesson
   documents aren't pasted in front of the items.
2. **Absorb.** The next lesson's teaching comes before this lesson's quiz. The curriculum is interleaved, so
   every quiz follows unrelated training (the lesson plan's absorb gap).
3. **Apply.** The model answers 10 fresh items for the lesson at the card's sampling settings. Answers that
   are right *and* pass the lesson's rule are kept as extra training for the next session (STaR: Zelikman et
   al. 2022, arXiv:2203.14465). Wrong answers go into the ledger at box 1.
4. **Quiz.** After the absorb gap, the lesson's quiz items, scored only by the pass rules in the lesson plan
   (code checks, no model judge).

**Order inside a session:** Lessons 1–3 first (the role, how a verifier thinks, what's expected), then the
18 trait lessons in the approved order, each trait's quiz placed after the next trait's teaching. Counterweight
pairs stay in one lesson.

## The repeat-until-correct ledger (as the lesson plan specifies, with one change)

The Leitner boxes, intervals (1, 2, 4, 8 sessions), fresh rule-made variants, item and quality mastery lines
(95% of items; 90% on fresh twins) and forgetting events are as the lesson plan sets them.

**The ledger decides what's shown next.** A session's teaching mix is:
- every item due from the ledger, as a fresh variant;
- the lesson's teaching examples, down-weighted for qualities already mastered (each example once, not
  repeated);
- the kept apply answers from the last session.

When a quality's forgetting rate rises between sessions, its whole lesson is taught again, not only its
quiz.

**The change: quizzes are judged at the card's sampling settings, not at T = 0.** Qwen3's model card advises
against greedy decoding in thinking mode, because it degrades and can loop, and the evaluation already uses
the card's settings for that reason (the Publisher's decision, 2026-10-10). A quiz item is answered with 3
samples (seeds 0–2). It counts as right when at least 2 of the 3 are right, and every sample is logged.

## Mastery and the end of Stage 1

- **Quality mastered:** as in the lesson plan (95% of items mastered, and 90% on fresh twins).
- **Stage 1 ends** at the first of:
  - every quality mastered, with the general-capability panel no more than 2 points below its baseline;
  - the evaluation plan's learning-curve stop rule (R&D's rule on the DEV set);
  - 12 sessions.

  At the session cap, anything not mastered is reported and the maintainer decides.
- **The general-capability panel** guards against the role costing the model elsewhere. It's a fixed set of
  ordinary questions with known answers (R&D picks it before session 1), scored at the card settings, before
  training and after every session.
- **After every session:** the DEV set (pilot 30, 3 seeds) and the pre-interview, for the learning curve.
  The sealed set is scored only at the evaluation plan's fixed checkpoints.

## How much training data the first session needs

**Decided (the maintainer, 2026-10-10): 40 items per trait, 8 of each of the five kinds (18 × 40 = 720), plus
30 for each of Lessons 1–3, 6 of each kind (90). About 810 in all, before round 1.**
- **Who writes them:** Kimi, under the Publisher's handoff
  ([2026-10-10-stage1-training-data-kimi-handoff.md](2026-10-10-stage1-training-data-kimi-handoff.md)),
  in the format of [stage1-training-data-schema.md](stage1-training-data-schema.md).
- **The apply items and quiz seeds** (10 + 10 per lesson, quiz variants rule-made) stay with this session,
  unless the maintainer assigns them to Kimi.
- **No new hand-written batch mid-curve.** The 810 are all in round 1. Later rounds add only the ledger's
  fresh variants and the kept apply answers, so the learning curve isn't muddied by new data arriving
  partway through.

Why 40 per trait:
- **Coverage.** Each trait is taught five ways (identity, thinking pattern, contrast, action, quiz), with 8
  of each. That's enough for both sides of the trait's counterweight in every kind (skepticism that rejects
  and skepticism that rightly accepts, for example), spread across the varied content the lesson plan lists,
  with at least 2 multi-step items per trait.
- **Scale.** About 800 curated examples is the size at which small, careful SFT sets have changed a model's
  behaviour and format without large data (Zhou et al. 2023, *LIMA: Less Is More for Alignment*,
  arXiv:2305.11206, with 1,000 examples). Quality and variety matter more than count at this scale, which suits
  hand-keying.
- **Cost.** 810 examples at about 1,200 tokens each is roughly 1M tokens a session. That's small enough to
  take one session per DEV evaluation, so the learning curve has a point per session.
- **The ledger adds the rest.** Later sessions add the apply answers that were kept and the fresh variants of
  failed items. So the first session needn't carry everything.

Every item follows the same rules as the task set:
- the instant-agreement rule;
- varied real material where it carries the case (and marked invented text where it can't);
- the R&D key check and the Publisher's read;
- the leakage check against the sealed set, the pilot and the pre-interview questions, with any similar
  item rejected. Kinds A and E are held to a lower bar against the interview questions (see the schema).

**Time is recorded per round:** wall-clock, GPU-minutes, examples per minute and cumulative training time,
plus the machine line (GPU, driver, CUDA, torch, transformers), as the evaluation plan reports it.

## Card use (estimates for the grant requests)

- **VRAM:** bf16 base 16.4 GB. LoRA rank 64 on all layers is about 175M parameters, so with its gradients
  and AdamW states it needs about 2.1 GB. Activations at 4,096 tokens with checkpointing take about 3–5 GB.
  The peak is about **22–24 GB**.
- **Measured at baseline (2026-10-10):** one batch-1 generation takes about 20 s (median about 325 new
  tokens). The pilot's 30 tasks took 9–10 minutes per seed, and the 36 interview answers took 13 minutes.
- **Time per session (estimate):**
  - Teaching: about 1M tokens, roughly 20–30 minutes. To be measured on a short first run.
  - Apply and quiz: 21 lessons × (10 apply + 10 quiz × 3 samples) = 840 generations. That's about 4.7 hours
    at batch 1. So apply and quiz generate in **batches of 8**, which is expected to cut this to about
    1 hour. Batching applies only inside the training loop. The DEV set and the sealed set keep the pinned
    batch-1 harness, so their numbers stay comparable with the baseline.
  - The DEV set (3 seeds) and the interview: about 45 minutes.

  So about **2–2.5 hours per session**. Each session is a separate grant, with the 15-minute rest between
  runs.

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Weights by revision and sha256; the loader, harness and trainer by sha256. Each session's data manifest, seeds and adapter sha256 are recorded. |
| ANDON_AUTHORITY | 2 | The loader refuses mismatched weights. The harness refuses to resume under a different hash. The watchdog stops on VRAM, temperature or RAM, and a session stops on any error. |
| NAMED_COMPENSATORS | 2 | Nothing is published. Adapters are written beside the base, which is never modified. Undo: delete the session's adapter directory (owner: this session). |
| DECOMPOSE_BY_SECRETS | 2 | The loader, eval harness, trainer and ledger are separate files. The sealed set lives outside the repository. |
| UNCERTAINTY_GATED_HUMANS | 2 | The maintainer approves the procedure and decides at the session cap. The Publisher grants every card use. |
| EXTERNAL_VERIFIER | n/a | No specialised claims. Keys get R&D's check, and quizzes are scored by code. |

## For the maintainer and the team

1. Approve the procedure, including the change to quiz scoring (card settings, majority of 3, not T = 0).
2. Approve about 810 hand-keyed examples (40 per trait) for the first session.
3. R&D names the general-capability panel before session 1.
