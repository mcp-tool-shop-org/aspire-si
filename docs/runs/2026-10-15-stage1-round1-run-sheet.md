# DRAFT: Stage 1 round 1, the run sheet

**Status: draft for R&D's review (2026-10-15).** No card is asked for until R&D has passed it and the Publisher
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
- **The machine:** the baseline ran on the 5090. Preflight (§6) records the GPU, driver, CUDA, torch and
  transformers versions and compares them with the baseline run note. A difference is written into the
  report; it doesn't change the procedure. Times are compared only on a matching machine.
- **Leakage:**
  - the receipt firewall refuses sealed, interview and DEV ids and text;
  - the leakage gate has been run against the sealed set, the pilot and the interview;
  - the quiz's spent check refuses any variant that shares an 8-word span with a trained turn.
- **Order and warmness:** each sitting loads the model fresh. The sealed sittings run after all training and
  DEV for the round, so no sealed result can steer anything (they are stored unopened, per the plan).

## 4. The tasks

All of these are fixed before round 1. None is written or changed for this sheet.

| Set | Size | Seeds | Use |
|---|---|---|---|
| Sealed 130 | 130 tasks | 3 training × 3 sample = 9 sittings | G1, stored unopened until round F |
| DEV (pilot 30) | 30 tasks | 3 × 3 | the stop rule's first point; descriptive |
| Quiz | the ledger's due kind-E items: 75 askable today (51 verdict, 24 closed-form), about 105 after batches 7–8 | majority of 3 per ask | mastery ledger; descriptive |
| Pre-interview | 12 questions | 3 per adapter | descriptive (Q1–11) and the one tested measure |
| General-capability panel | not named yet (§6, B3) | per the panel | the 2-point guard |

## 5. Measures and the decision rule

These come from the evaluation plan, unchanged.

- **What round 1 decides by itself: nothing about the role.** The primary measure is A_F − A_0. Round 1 gives
  G1, which is computed, stored and not read until round F.
- **What round 1 does decide:**
  - Round 2 runs unless a round-1 stop rule fired (§8). The DEV plateau rule needs two rounds of gains, so it
    can't fire at round 1.
  - The panel guard: if the panel falls more than 2 points below its baseline, round 2 waits for the
    maintainer.
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
| B3 | The general-capability panel is named by R&D and scored on the base model before training (the procedure: "before session 1") | R&D, then a short grant | not named |
| B4 | Quiz time: the quiz generates one sample at a time. The procedure planned batches of 8 inside the loop, which isn't built. Decide: build batched generation for quiz and apply (a design note first, since it changes the generate path), or accept about 1.75 h per seed (§7) | R&D's call, ASPIRE builds | open |
| B5 | Gates 1.7.0 and 1.7.1 merged, so the trainer validates under the final gates | R&D, then the Publisher | in review |

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
8. **The machine line**, recorded and compared with the baseline's.

Any failure stops the grant before the counted run, and nothing counted is spent. Scratch outputs are deleted
after the note is written (§10).

## 7. The run: commands, size and time

**Round 1, arm ROLE, seeds 0, 1, 2.** The NO-ROLE arm is Phase B's compute-matched control, and is measured
only after Stage 2's domain training, which isn't built. It isn't in round 1 (see §11, Q1).

Environment: `ASPIRE_WATCHDOG_DIR` and `ASPIRE_BACKUP_ROOT` from the rig, `CUDA_VISIBLE_DEVICES=0`. The data
arguments are every file listed in `FROZEN.md`, in its order.

```
# Block T: training, for each seed S in 0 1 2
python docs/runs/stage1-train/run.py --round 1 --arm ROLE --seed S \
    --grant-id <GRANT_ID> --grant-end <GRANT_END> --planned-peak-gb 24 --planned-minutes 10 \
    -- --data <FROZEN FILES> --order shuffled
# (closed-form settings: the gates' defaults; recorded in the manifest)

# Block Q: the quiz sitting, for each seed S (session 1)
python docs/runs/stage1-train/quiz.py --adapter outputs/stage1-train/1/ROLE/S/train/adapter \
    --data <FROZEN FILES> --ledger outputs/stage1-train/1/ROLE/S/ledger.json --session 1 \
    --round 1 --arm ROLE --seed S --variants <VARIANT STORE> --sealed <SEALED FILE>

# Block D: DEV, for each seed S and sample seed K in 0 1 2
python docs/runs/stage1-train/eval_adapter.py --adapter outputs/stage1-train/1/ROLE/S/train/adapter \
    --tasks <DEV TASKS> --out outputs/stage1-eval/1/ROLE/S/dev-K --seed K --phase A-round1 --arm ROLE

# Block I: the pre-interview (after B2), for each seed S
python <interview wrapper> --adapter outputs/stage1-train/1/ROLE/S/train/adapter \
    --out outputs/stage1-eval/1/ROLE/S/interview --seeds 0 1 2 --sitting round1 --arm ROLE

# Block S: sealed, for each seed S and sample seed K; written, never opened until round F
python docs/runs/stage1-train/eval_adapter.py --adapter outputs/stage1-train/1/ROLE/S/train/adapter \
    --tasks <SEALED FILE> --out outputs/stage1-eval/1/ROLE/S/sealed-K --seed K --phase A-round1 --arm ROLE
```

**Expected peak** (memory.used on the card, including the about 2.2 GB idle baseline; plan ≤ 26 GB, and
train.py stops itself at 28 GB):
- **Training:** the 2026-10-11 rehearsal measured a 19.08 GB torch peak on the real data (median step 6.73 s).
  With the CUDA context and the idle baseline, that's about 22–23 GB of memory.used. Preflight step 4
  confirms it.
- **Generation:** bf16 weights 16.4 GB, the LoRA about 0.35 GB, and the KV cache up to the 16,384-token cap,
  about 2.4 GB. So about 21–22 GB with the baseline. It was never measured with an adapter; preflight step 7
  confirms it.

**Time** (from measured numbers: rehearsal steps; baseline sittings at 9.7–10.1 min DEV, 42–45 min sealed,
13.6 min interview, per sample seed; about 20 s per generation):

| Block | Per seed | 3 seeds |
|---|---|---|
| T: about 51 steps × 6.7 s, plus about 45 s load | about 7 min | about 21 min |
| Q: about 105 asks × 3 samples × 20 s (batch 1) | about 1 h 45 min | about 5 h 15 min |
| D: 3 sittings × about 10 min | about 30 min | about 1 h 30 min |
| I: one sitting, 36 answers | about 14 min | about 42 min |
| S: 3 sittings × about 44 min | about 2 h 12 min | about 6 h 36 min |
| Panel | after B3 | after B3 |

That totals about **14 h 25 min** of card time plus the panel, at batch-1 quiz. A trained model may think
longer than the untrained baseline did, so each sitting's real time is recorded and the next grant is sized
from it.

**Proposed grants** (the Publisher schedules them; there's no rest coded into the tools):
- **G1, the round:** preflight, then T, Q and D, seed by seed (T0 Q0 D0, T1 Q1 D1, T2 Q2 D2), then I for all
  three. About 8 h at batch-1 quiz, or about 4 h if B4 builds batching.
- **G2, sealed:** S for the 3 × 3 sittings, about 6.6 h. It can be split per seed (three grants of about
  2.2 h).
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
`63162cf4`, stuck.py `8097b039`, stuck rule `716dc251`, the sealed set `e1e515b4`, the student
`b968826d9c46`, gates 1.7.1 (after B5), and the data `<FROZEN_DATA_SHA>`. Each run's manifest and receipt
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

## 11. Open questions for the review

- **Q1.** NO-ROLE in round 1: leave it out (as drafted), or train its compute-matched round now (about 21 min)
  so Phase B's control tracks ROLE round by round?
- **Q2.** B4, batched quiz generation: is it worth building before round 1? It saves about 3.5 h a round, but
  changes the generate path, so it needs its own design note.
- **Q3.** Should the panel (B3) be scored on the base model in the same grant as preflight, or in its own
  grant?

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
