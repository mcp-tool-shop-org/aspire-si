# DRAFT: Stage 1 round 1, the run sheet

**Status: draft, revised for R&D's review of 2026-10-15 (Q1–Q3 ruled; gaps 1–3 closed).** No card is asked for until R&D has passed it and the Publisher
has heard from the maintainer on any further review. Under the 2026-10-13 rule, the design is settled before
anything runs, and isn't revised through live rounds.

The procedure, the evaluation plan and the teaching material were approved by the maintainer on 2026-10-11
([training procedure](2026-10-10-stage1-training-procedure-prereg-draft.md),
[evaluation plan](2026-10-10-stage1-evaluation-plan-draft.md)). This sheet changes none of their decisions. It
turns round 1 into blocks with exact commands, sizes, times, stop rules and a cleanup record. Its layout follows
rnd's A/B design template (handbook, "A/B experiments").

## 1. The check in three sentences

1. **The question:** does one round of Stage 1 role training (teach, absorb, quiz) make Qwen3-8B a better
   verifier than the untrained baseline, on the sealed 130? Round 1 supplies G1 = A_1 − A_0, stored unopened,
   and the first point of the DEV curve.
2. **What each side gets:** the base side is the 2026-10-10 baseline, already scored: Qwen3-8B at
   `b968826d9c46`, the pinned harness and the card's sampling. The round-1 side is the same model with a round-1
   LoRA adapter attached unmerged, trained on the frozen data commit with seeds 0, 1 and 2, under the same harness
   and settings.
3. **Why the difference answers it:** the adapter is the only thing that differs. The harness (`4fa52448`), the
   scorers (`fd59cb50`, `64f52f6f`, `55b4d0ea`), the sampling, the prompts and the sealed file (`e1e515b4`) are
   pinned and byte-identical between the baseline and round 1.

## 2. The treatment is taken (manipulation check)

Each sitting's record must show the adapter was in use. A sitting that fails any of these is reported as
untreated and doesn't count.

- **The adapter is recorded:** eval_adapter writes its sha256 beside the harness's, and it matches the trained
  run's `manifest.json`.
- **The data is the frozen set:** the manifest names `<FROZEN_DATA_SHA>`, and every file's sha256 matches
  `FROZEN.md`.
- **Not a smoke adapter:** the manifest has no `partial_set`, because eval_adapter refuses a sealed read for one.
- **It trained:** the loss curve shows training happened: `total_steps` steps logged, each logged exactly once.

## 3. The sides differ only by the adapter

- **Settings parity:** the card's settings (T 0.6, top-p 0.95, top-k 20), sample seeds 0–2, the 16,384-token
  cap, and the prompt and answer-format hashes are all the harness's own, unchanged.
- **The machine is frozen, and a mismatch is a hard stop** (Argon's Q2, R&D 2026-10-15).
  - G0's M1 writes the frozen machine line: GPU, driver, torch's CUDA, torch and transformers
    (`det_run.py --write-machine-line`, which refuses to overwrite an existing line).
  - M2 and every later sitting check it before a model loads (`det_run.py --machine-line`). Any difference
    stops that sitting (exit 5), with nothing generated.
  - The line changes only by a dated amendment with a new G0.
  - The machine freeze (no driver, GPU-software or training-environment changes from G0 to round F) is the
    maintainer's to put in place.
- **The determinism wrapper:** every batch-1 sitting (DEV, sealed, interview, and G0's M1 and M2) runs through
  `det_run.py`, with the harness unchanged. It applies deterministic algorithms (warn_only only if strict mode
  raises, with the warned ops logged), the cuDNN flags and the CUBLAS workspace, and pins SDPA to flash. Its
  settings are recorded in each sitting's `wrapper-config.json`.
- **Leakage:**
  - the receipt firewall refuses sealed, interview and DEV ids and text;
  - the leakage gate has been run against the sealed set, the pilot and the interview;
  - the quiz's spent check refuses any variant that shares an 8-word span with a trained turn.
- **Order and warmness:** each sitting loads the model fresh. The sealed sittings run after all training and
  DEV for the round, so no sealed result can steer anything (they are stored unopened, per the plan).

## 4. The tasks

All of these are fixed before round 1. None is written or changed for this sheet.

| Set | Size | Arms | Seeds | Use |
|---|---|---|---|---|
| Sealed 130 | 130 tasks | ROLE | 3 training × 3 sample = 9 sittings | G1, stored unopened until round F |
| DEV (pilot 30) | 30 tasks | ROLE | 3 × 3 | the stop rule's first point; descriptive |
| Quiz | the ledger's due kind-E items: 75 askable today (51 verdict, 24 closed-form), about 105 after batches 7–8 | both | majority of 3 per ask | the mastery ledger; **the E steering cells' stuck readings** |
| Apply | 10 fresh items per lesson (21 lessons), kinds C and D, written by role-set, never trained, kept out of the reserve and steering material | both | 1 sample per item | **the C and D steering cells' stuck readings**, and STaR's kept answers (§5) |
| Pre-interview | 12 questions | ROLE now; NO-ROLE after Stage 1 | 3 per adapter | descriptive (Q1–11) and the one tested measure |
| General-capability panel | R&D's design (B3) | ROLE | per the design | the 2-point guard, every session |

**Where the steering readings come from** (gap 1). Steering's cells are (lesson, kind) for kinds C, D and E,
counted at n ≥ 8, pooled across both arms. E readings come from the quiz sitting (`train_quiz` receipts). C and
D readings come from the apply sitting (`rehearsal` receipts, the store's training-time source). At 10 apply
items per lesson, split 5 C and 5 D, each arm gives 5 per cell and the pool gives 10, which clears n ≥ 8. The
apply items don't exist yet (B6). Until they do, round 2 can't be steered as registered.

## 5. Measures and the decision rule

These come from the evaluation plan, unchanged.

- **What round 1 decides by itself: nothing about the role.** The primary measure is A_F − A_0. Round 1 gives
  G1, which is computed, stored and not read until round F.
- **What round 1 does decide:**
  - Round 2 runs unless a round-1 stop rule fired (§8). The DEV plateau rule needs two rounds of gains, so it
    can't fire at round 1.
  - The panel guard: if the panel falls more than 2 points below its baseline, round 2 waits for the
    maintainer.
- **STaR (the apply step's kept answers).** The approved procedure keeps the apply answers that are right *and*
  pass the lesson's rule as extra training for the next session. This sheet doesn't change that. R&D's rulings of
  2026-10-11 set how it runs:
  - **It's off until the apply items exist.** With no apply items there's nothing to keep, and the quiz readout
    reports STaR as off. So STaR is on in round 1 exactly when B6 is done.
  - **The keep rule:** an answer is kept only when all three lines pass the pinned scorer (the verdict, DECIDING
    verbatim and covering the key, ESCALATE), it meets the lesson's quiz pass rule, and its trace isn't stuck
    under the frozen rule (716dc251). Refusals are logged by reason.
  - **The apply material** is unseen. It's kept out of training (only the kept answers enter session 2, as the
    procedure says), the reserve and steering.
  - **Round 1 doesn't start without B6** (Q4, ruled), so there's no round 1 without the apply step. Kept answers
    only enter session 2.
  - **The STaR step's command** (matching, the near-duplicate drop and the cap of 8 per lesson and kind):
    `apply.py star --session <n> --kept <ROLE kept> <NO-ROLE kept> --prior-role <ROLE's earlier STaR files>
    --prior-no-role <NO-ROLE's earlier STaR files> --training-role <the FROZEN.md files> --training-no-role
    <the pinned NO-ROLE file> --out <STaR file>`. The drop counts only spans the model originated, so each arm
    subtracts its own training rows: ROLE the curriculum, NO-ROLE its pinned no_role.py output only. Earlier
    STaR rows are never subtracted; they're what the drop compares against (R&D on Argon's 7.1 condition,
    2026-10-16).

  Two questions for R&D before STaR first keeps anything (Q5):
  - **NO-ROLE:** does it keep its own apply answers, to stay compute-matched, or none?
  - **TRACE_REUSED:** kept answers are model traces entering training through STaR's own channel. That's
    separate from generated item text, which TRACE_REUSED governs. The store should mark kept answers so the
    gate doesn't later count them as dataset wording.
- **Reported for round 1:**
  - DEV accuracy per seed and the mean, with the between-seed SD;
  - the quiz readout: n = 0 against n ≥ 1, per lesson and overall, with the written E left out and counted;
  - stuck rates by the frozen rule (`716dc251`);
  - training time, GPU-minutes and examples per minute;
  - every action from actions.jsonl.

## 6. Before the first counted run

### Blockers (each must be done, and named in the grant request)

| # | Blocker | Owner | State |
|---|---|---|---|
| B1 | Batches 7 (T15, T18) and 8 (L1–L3) merged, then `FROZEN.md`. The grant names `<FROZEN_DATA_SHA>` | the Dataset Specialist (parked) and the writers | not started |
| B2 | The pre-interview can't load an adapter: `interview.py` has no `--adapter`. It needs a thin wrapper like eval_adapter's (base from `load_student()`, adapter attached unmerged, interview.py's file untouched and its sha recorded), plus tests | ASPIRE | to build, CPU only |
| B3 | The general-capability panel: a design note (size, item type, mechanical scoring, batching, per-session cost) for Argon and the maintainer, then its base-model baseline in its own grant (G0) | R&D | design to write |
| B4 | Batched generation for quiz and apply, in batches of 8 as the procedure planned. A short design note pins: quiz and apply only (eval_adapter and the harness untouched); padding side and attention masks; how sample seeds 0–2 map under batching; the card settings kept; and a parity check against batch 1 on a fixed small set (verdict agreement and majority outcome, a distribution check, not bytes). Fixed from round 1: every round uses the same quiz path | ASPIRE builds, R&D reviews | ruled yes (Q2) |
| B5 | Gates 1.7.0 and 1.7.1 merged, so the trainer validates under the final gates | R&D, then the Publisher | 1.7.0 merged; 1.7.1 in review |
| B6 | The apply items: 10 per lesson, C and D, written by role-set in the same pass as batches 7–8, and key-checked by R&D with the same instant-agreement check as every training batch before the freeze (a bad key would bias the steering signal); plus an apply sitting in the runner that writes `rehearsal` receipts with the stuck fields | the Dataset Specialist (items); ASPIRE (the sitting) | not started |

### The composition check (before any counted training run)

The evaluation plan's 2026-10-10 amendment, point 3.
- `no_role.py --data <FROZEN FILES> --out <NO-ROLE FILE>` prints both arms' composition tables: verdict mix,
  kinds, correction and escalate counts, multi-line share and NONE share. It also prints the exposure per
  material and the target-token ratio.
- **The gate is the registered one:** R&D checks the tables match within one item per trait. A larger
  difference stops round 1.
- NO-ROLE is derived one for one from ROLE, so any non-zero difference within that tolerance is still reported in
  the run note with its cause, before round 1.
- The target-token ratio and the exposure per material are reported, not gated.
- The NO-ROLE file's sha256 is pinned in the run note.

This needs no card, so it runs as soon as `FROZEN.md` lands.

### Live preflight (on the rig, in the counted run's own grant, before the counted run)

A green CI isn't proof. Each reading the run depends on is shown to work on the rig, and its output goes in the
run note. Total about 15 minutes, peak as in §7.

1. **The watchdog:** the heartbeat is under 60 s old (`ASPIRE_WATCHDOG_DIR`). run.py refuses to start
   otherwise.
2. **nvidia-smi sampling:** `run.py`'s probe returns temperature, power, SM clock and memory.used. Record the
   idle memory.used (the baseline, about 2.2 GB).
3. **The firewall arms:** the sealed pin verifies on this checkout (CRLF-tolerant), and a receipt is written.
4. **A 2-step training run:** the real data, `--max-steps 2 --partial-set`, round 0, in scratch. This proves
   the data validates under the frozen gates and the checkpoint is written. Record the peak memory.used from
   the probe.
5. **The quiz and ledger write:** `quiz.py` on that 2-step adapter for one lesson, writing the ledger, the
   receipts and the readout in scratch. This shows the ledger file and the stuck fields are written.
6. **The sealed refusal:** `eval_adapter.py` with the sealed tasks and the 2-step adapter must refuse
   (partial set). This is the guard that keeps a smoke adapter from spending a sealed read.
7. **DEV, 2 tasks:** `eval_adapter.py --tasks <DEV> --limit 2` on the 2-step adapter. Record the peak
   memory.used during generation.
8. **The machine line**, recorded and compared with the baseline's and G0's.
9. **The apply sitting** on the 2-step adapter, one lesson, writing `rehearsal` receipts with the stuck fields.
10. **Batched parity** (B4): the check pinned in the batching note, on its fixed small set.

Any failure stops the grant before the counted run, and nothing counted is spent. Scratch outputs are deleted
after the note is written (§10).

## 7. The run: commands, size and time

**Round 1, arms ROLE and NO-ROLE in lockstep, seeds 0, 1, 2** (R&D, Q1). Both arms train under the same pins,
gates and machine in the same grant. ROLE gets the full session. NO-ROLE gets training plus the two sittings that
write its steering readings (quiz and apply). Phase A's DEV, sealed and panel are ROLE's, and NO-ROLE's
pre-interview comes after Stage 1, as the plan says.

Environment: `ASPIRE_WATCHDOG_DIR` and `ASPIRE_BACKUP_ROOT` from the rig, `CUDA_VISIBLE_DEVICES=0`. ROLE's data
is every file listed in `FROZEN.md`, in its order. NO-ROLE's data is the `no_role.py` output pinned by the
composition check.

**Launch:** every block starts with stdout and stderr redirected to files (for example `> block.log 2>&1`), never
in an interactive console window. On Windows a click in a console window enters QuickEdit selection and freezes
the process writing to it, which would read as a stall.

```
# Block T: training, for each seed S in 0 1 2, and arm A in ROLE NO-ROLE (data as above)
python docs/runs/stage1-train/run.py --round 1 --arm A --seed S \
    --grant-id <GRANT_ID> --grant-end <GRANT_END> --planned-peak-gb 24 --planned-minutes 10 \
    -- --data <A's FILES> --order shuffled
# (closed-form settings: the gates' defaults; recorded in the manifest)

# Block Q: the quiz sitting (session 1), each seed S and arm A
python docs/runs/stage1-train/quiz.py --adapter outputs/stage1-train/1/A/S/train/adapter \
    --data <A's FILES> --ledger outputs/stage1-train/1/A/S/ledger.json --session 1 \
    --round 1 --arm A --seed S --variants <VARIANT STORE> --sealed <SEALED FILE>

# Block A: the apply sitting (after B6), each seed S and arm A; writes rehearsal receipts
<apply sitting> --adapter outputs/stage1-train/1/A/S/train/adapter --items <APPLY ITEMS> \
    --round 1 --arm A --seed S --sealed <SEALED FILE>

# Block D: DEV (ROLE), each seed S and sample seed K in 0 1 2; through the wrapper
python docs/runs/stage1-train/det_run.py --machine-line <FROZEN LINE> -- eval_adapter --adapter outputs/stage1-train/1/ROLE/S/train/adapter \
    --tasks <DEV TASKS> --out outputs/stage1-eval/1/ROLE/S/dev-K --seed K --phase A-round1 --arm ROLE

# Block P: the general-capability panel (ROLE), each seed S, as B3's design sets it

# Block I: the pre-interview (ROLE), each seed S; through the wrapper
python docs/runs/stage1-train/det_run.py --machine-line <FROZEN LINE> -- interview_adapter --adapter outputs/stage1-train/1/ROLE/S/train/adapter \
    --out outputs/stage1-eval/1/ROLE/S/interview --seeds 0 1 2 --sitting round1 --arm ROLE

# Block S: sealed (ROLE), each seed S and sample seed K; written, never opened until round F; wrapper
python docs/runs/stage1-train/det_run.py --machine-line <FROZEN LINE> -- eval_adapter --adapter outputs/stage1-train/1/ROLE/S/train/adapter \
    --tasks <SEALED FILE> --out outputs/stage1-eval/1/ROLE/S/sealed-K --seed K --phase A-round1 --arm ROLE
```

**Expected peak** (memory.used on the card, including the about 2.2 GB idle baseline; plan ≤ 26 GB, and
train.py stops itself at 28 GB):
- **Training:** the 2026-10-11 rehearsal measured a 19.08 GB torch peak on the real data (median step 6.73 s).
  With the CUDA context and the idle baseline, that's about 22–23 GB of memory.used. Preflight step 4
  confirms it. NO-ROLE's sequences are no longer than ROLE's (its target-token ratio is printed by the
  composition check), so its peak is no higher.
- **Generation at batch 1** (DEV, sealed, interview): bf16 weights 16.4 GB, the LoRA about 0.35 GB, and the KV
  cache up to the 16,384-token cap, about 2.4 GB. So about 21–22 GB with the baseline; preflight step 7
  confirms it.
- **Batched generation** (quiz and apply, 8 at a time): the KV cache scales with the batch. Eight sequences at
  the full 16,384-token cap would need about 19 GB of cache, over the plan. So B4's design note must set a
  per-batch token budget, or a smaller batch at long lengths, to stay ≤ 26 GB, and preflight step 10 measures
  it. This is the one sizing open until B4.

**Time** (measured: rehearsal steps; baseline sittings at 9.7–10.1 min DEV, 42–45 min sealed and 13.6 min
interview per sample seed; about 20 s per batch-1 generation. Batched times are the procedure's estimate, about
4.7× faster, until B4's parity run measures them):

| Block | Arms | Per seed and arm | Round 1 |
|---|---|---|---|
| T: about 51 steps × 6.7 s, plus about 45 s load | both | about 7 min | about 42 min |
| Q: about 105 asks × 3 samples (batched) | both | about 22 min (1 h 45 min at batch 1) | about 2 h 12 min |
| A: 210 apply items × 1 sample (batched) | both | about 15 min (70 min at batch 1) | about 1 h 30 min |
| D: 3 sittings × about 10 min | ROLE | about 30 min | about 1 h 30 min |
| P: the panel | ROLE | B3's design | B3's design |
| I: one sitting, 36 answers | ROLE | about 14 min | about 42 min |
| S: 3 sittings × about 44 min | ROLE | about 2 h 12 min | about 6 h 36 min |

That totals about **13 h 12 min** of card time plus the panel, with preflight (about 20 min) on top. A trained
model may think longer than the untrained baseline did, so each sitting's real time is recorded and the next
grant is sized from it.

**Proposed grants** (the Publisher schedules them and takes the schedule to the maintainer before booking;
there's no rest coded into the tools):
- **G0, the panel baseline** (Q3): the base model on B3's panel, its own grant, before G1. It can run while B1
  is pending. The machine line is recorded, and compared at G1's preflight.
- **G1, the round:** preflight, then seed by seed: T (ROLE, NO-ROLE), Q (both), A (both), D, P. Then I for the
  three ROLE adapters. About 6.6 h plus the panel.
- **G2, sealed:** S for the 3 × 3 sittings, about 6.6 h. It can be one grant, or three of about 2.2 h, one per
  seed.
- **Rest:** owed only after a long run on the same device, as the Publisher's scheduling. G2 doesn't start
  straight after G1 unless the Publisher says so.

## 8. Stop rules, and what each leaves behind

Every stop writes status.json, actions.jsonl and the diagnostics receipt, and sends the Publisher "card free"
once the card is idle.

| Trigger | Action (automatic, in run.py) | Left behind |
|---|---|---|
| Crash or kill, not memory | resume after 60 s, at most 3 times, with preconditions | the checkpoint; steps logged once each |
| VRAM over 28 GB, or a watchdog kill | stop, never resumed: a sizing defect | the last checkpoint; the run is reported; sizing is re-planned before any new grant |
| GPU ≥ 85 °C for 60 s | pause between steps; stop after 30 min of pause | the checkpoint; paused time is excluded from train time |
| Stall (no step in 5 × median + slack) | stack dump, then stop | stall-stack.txt, the checkpoint |
| The grant's end | checkpoint at the last finished step, exit 77, never auto-resumed | a resumable checkpoint; extending is the Publisher's call |
| **Divergence:** a non-finite loss, or 3 straight steps over 3 × the running median | stop, never acted on | the checkpoint before it; a person decides |
| A transient I/O error | retries at 1, 2, 4, 8, 16 s | the log of each try; a full disk stops the run and nothing is deleted |
| An eval sitting error (harness, OOM, refusal) | that block stops; later blocks don't start | the partial raw.jsonl; a rerun resumes only under identical settings (the harness refuses otherwise) |
| Firewall refusal, or a manipulation-check failure (§2) | stop the round | the receipt that refused; nothing is ingested |
| Leakage found | stop everything | the finding; a dated amendment before anything resumes |

A change to the design after any counted run is a dated amendment, and it never touches what §5 counts.

## 9. Pinned for this round

The harness `4fa52448`, score.py `fd59cb50`, score_v2 `64f52f6f`, score_found `55b4d0ea`, no_role.py
`ac795e2b` (was `63162cf4`; Argon's finding 5a, then the B items' statement field, both before any counted run), stuck.py `8097b039`, stuck rule `716dc251`, the sealed set `e1e515b4`, the student
`b968826d9c46`, gates 1.7.3 (after B5; Argon findings 7.2 and 8.2; NF3), the data `<FROZEN_DATA_SHA>`, and the NO-ROLE file from the composition check. Each run's manifest and receipt
records them. A mismatch at start is refused.

## 10. The cleanup record

At the end of each grant, the run note records:
- **Card:** memory.used back to its idle baseline (the nvidia-smi line), no python process left (run.py's Job
  Object ends the trainer), and "card free" sent to the Publisher.
- **Kept:** under `outputs/` (git-ignored), each seed's adapter, final checkpoint, loss.jsonl, manifest,
  status, actions, diagnostics and receipts; the ledger; the DEV, interview and quiz outputs; and the sealed
  raw outputs, unopened.
- **Copied:** the backup step copies each run to the external archive (`external-archive:stage1-train/...`),
  or logs "skipped: not mounted".
- **Deleted:** only the preflight's scratch outputs, after its note is written. Never a counted run's files.
- **In the repo:** a run note with hashes, times, peaks, actions and the DEV and quiz figures. No sealed
  content, no sealed scores before round F, and no machine paths.

## 11. Rulings on the review's questions (R&D, 2026-10-15)

- **Q1, NO-ROLE:** it trains in round 1, in lockstep. Steering is pooled across both arms, so round 2 needs
  NO-ROLE's round-1 readings, and both arms then share pins, gates and machine.
- **Q2, batched quiz and apply:** build it before round 1, as the approved procedure planned, with a short design
  note (B4). DEV and sealed stay at batch 1.
- **Q3, the panel baseline:** its own grant (G0), before G1.

- **Q4, B6 lagging B1 (ruled):** wait for B6. Round 1 doesn't start on E-only readings: every session from round
  2 on has the apply sitting, so round 1 must too, and a partially steered round 2 would muddy the steered curve.
  The apply items are written in the same pass as batches 7–8. If they still don't land together, the measured
  lag goes to the maintainer then; E-only isn't pre-authorized.

**Still open:**
- **Q5.** STaR in NO-ROLE, and kept answers against TRACE_REUSED (§5).

## Standards compliance

| Standard | Score | Evidence |
|---|---|---|
| PIN_PER_STEP | 2 | Every file the round reads is pinned by sha256 (§9) and recorded in each manifest and receipt; a mismatch is refused. |
| ANDON_AUTHORITY | 2 | The stop rules in §8 are run.py's own, tested on the tiny trainer and on the 5090 smoke and rehearsal. Preflight halts before any counted spend. |
| NAMED_COMPENSATORS | 2 | Table below. |
| DECOMPOSE_BY_SECRETS | 2 | Training (run.py, train.py), evaluation (eval_adapter, the untouched harness), the quiz and ledger, and the store (role-set) are separate. |
| UNCERTAINTY_GATED_HUMANS | 2 | Anything that changes what's measured stops the run for a person. The grant, the round-2 go on a panel drop, and every open question are human calls. |
| EXTERNAL_VERIFIER | n/a | No specialized claims. R&D reviews the design. |

### Compensators

| Action | Undo | State after undo | Owner |
|---|---|---|---|
| Use of the card under a grant | stop the run (exit 77 at the grant end, or a stop), send "card free" | card idle at its baseline | ASPIRE |
| Training outputs written | none needed (git-ignored); a void run is marked void in its note, never deleted | the run is kept as a record | ASPIRE |
| Sealed sitting written | none: a sealed read can't be undone, which is why §6 step 6 and eval_adapter's guard refuse smoke adapters | stored unopened until round F | ASPIRE, with R&D |
| Receipts ingested by role-set | `role-set void --round 1` | the round's rows inactive | the Dataset Specialist |
| Run note committed | a follow-up commit; the PR is closed unmerged | the repo as before | ASPIRE; the Publisher merges |
