# The Stage 1 general-capability panel (B3)

**Status: maintainer approved, 2026-10-16.** Argon approved the any-seed hold the same day. R&D accepted the design at `2b97739`. This revision adds the three points from the Publisher's comment: the machine-line check and the 26 GB cap before the model loads, the item order into the batched path, and Argon's note on the false-hold bounds. The note adds no code, no item file and no scorer. Nothing here has run on the card.

**Which text this follows.** Two dated amendments govern the panel. Amendment B (evaluation plan, 2026-10-15) is the stop rule. Amendment C (training procedure, 2026-10-15) is the running. Where the round-1 run sheet still says the plain 2-point guard (§4, §5), amendment B has replaced that line. This note follows amendment B and does not edit the run sheet. Grants follow amendment C, the later text: three per-seed grants, each with both arms, and 15-minute rests. The run sheet's single G1 line is stale. Fixing that line is the run sheet's job, not this note's.

The tree this was read against is `verifier-distill-prereg-draft` at `6f06c9b`. Gates behaviour is aspire-si-gates 1.7.3 as that commit has it. Every figure marked **computed** was calculated in this draft. Figures marked **counted** were read off the named source file. Nothing else was measured.

## 1. Size and composition

**280 items. 140 from GSM8K's test split, 140 from WinoGrande 1.1's dev split.** 280 is inside amendment C's 250–300, and it divides by the batch of 8, so a sitting has 35 batches and no tail (35 × 8 = 280).

| Slice | Items | What it measures | Answer | Score |
|---|---|---|---|---|
| GSM8K test | 140 | multi-step grade-school arithmetic | an integer | exact, tolerance 0 |
| WinoGrande 1.1 dev | 140 | commonsense coreference (fill the blank) | the letter A or B | exact letter |

Both are ordinary questions. Neither is a verifier item: no verdict, no DECIDING, no ESCALATE. `score_three_lines` is the quiz and apply scorer for those three lines. It is not the panel scorer.

There is no short-string slice. A slice and its parser are added together, by a dated change before P0, if a permissively licensed source turns up. The sets looked at for that were share-alike, unclear, or taken down (§2).

**One official item, one panel item.** No twin and no paraphrase is kept beside its pair. After the hash order below, an item is kept only when its source text shares no 8-word span, by `gates._shingles(..., 8)`, with an item already kept. The shared instruction lines are not part of that text: they are the same on every item, and would reject the whole set. Source text is the GSM8K `question`, or the WinoGrande `sentence` plus both options.

Counted on the files named in §2: GSM8K test has 23 pairs that share an 8-word span (43 of 1,319 rows). WinoGrande dev has 282 pairs (564 of 1,267 rows). Dropping one side of each pair leaves well over 140 in each file. The walk below does that drop.

**Sampling, one recorded order.** For each source row, the sort key is the hex sha256 of the UTF-8 string `panel-b3-v1\n<source>\n<item id>`. Rows are taken in that order. A row is skipped when its gold fails §4, or when its source text shares an 8-word span with a row already kept, or when §3 rejects it. The next row fills the gap. The walk stops at 140 kept rows per source. The item file records the key, the decision and the reason for every row it looked at, so the same source snapshot builds the same file.

Item ids, and therefore the per-item seeds:

- GSM8K: `gsm8k-test-<line>`, the line number in `test.jsonl` starting at 0.
- WinoGrande: `winogrande-<qID>`, the `qID` field. Every dev row has one, and all 1,267 are distinct (**counted**).

## 2. Sources, versions and licences

R&D checks every licence against its primary page when revising. The quotes here are the starting point.

### GSM8K — used

- **Name and version.** GSM8K (Grade School Math 8K), the test split, file `grade_school_math/data/test.jsonl` at commit `3101c7d5072418e28b9008a6636bde82a006892c` on `openai/grade-school-math`.
- **URL.** <https://github.com/openai/grade-school-math> and the raw file at that commit. The repository was archived by its owner on 8 April 2026. The commit above is still the file this draft counted.
- **Size.** 1,319 rows (**counted**). Each row is `question` and `answer`. The 1,319 questions are unique. Every answer ends with a `####` integer once commas are removed (**counted**: 0 rows failed that parse).
- **Sample.** The first 140 that survive §1 and §3.
- **Licence.** MIT, quoted from <https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/LICENSE>:

> Copyright (c) 2021 OpenAI
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

The data file lives in that repository, so the grant covers it. No other restriction is stated. R&D verified this `LICENSE` at the primary source on 2026-10-16. The notice is kept beside the item file. It is not committed to this repository.

### WinoGrande — used

- **Name and version.** WinoGrande version 1.1, the dev split (`dev.jsonl` with `dev-labels.lst`). The download script at commit `727e837f77521ef38bcc56df3b275c8da43f45af` fetches `https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip`.
- **URL.** <https://github.com/allenai/winogrande> (README at that commit: <https://github.com/allenai/winogrande/blob/727e837f77521ef38bcc56df3b275c8da43f45af/README.md>).
- **Size.** 1,267 dev rows (**counted**). Fields: `qID`, `sentence`, `option1`, `option2`, `answer`. `answer` is `"1"` on 628 rows and `"2"` on 639. Every sentence contains the blank `_`. The label file matches `answer` on every row (**counted**). The test split has no labels in the zip, so it is not used.
- **Sample.** The first 140 that survive §1 and §3. Option 1 is the letter A and option 2 is the letter B.
- **Licence.** The README's own sentence, quoted from the commit above:

> Winogrande (codebase) is licensed under the Apache License 2.0. The dataset is licensed under CC-BY.

The README does not name a CC BY version and does not link a deed. R&D checked the primary source on 2026-10-16. GitHub's licence field on the repository is Apache-2.0, which covers the code. The Hugging Face card carries no licence field. The dataset licence is **CC-BY, version not stated by AllenAI**. This note does not present 4.0 as the licence.

Every CC BY version permits reproduction and requires attribution, so the obligation is the same whichever version AllenAI meant.

**Attribution file.** It travels with the item file and is not committed to this public repository. It lists:

- the creators (Sakaguchi, Le Bras, Bhagavatula, Choi; AllenAI, 2019), with a link to <https://github.com/allenai/winogrande>;
- the licence sentence as quoted above;
- the modifications: "reformatted into prompts; options relabelled A and B".

The codebase's Apache-2.0 is the code only. The dataset line is the CC-BY sentence. Keeping the items out of this repository keeps the study out of sharing them, apart from what the attribution file already covers. The GSM8K notice sits beside the item file in the same place.

### Looked at and not used

| Source | Why it is not in the 280 |
|---|---|
| ARC (AI2 Reasoning Challenge) | The AI2 data page now redirects to the Hugging Face card `allenai/ai2_arc`, whose card metadata says `cc-by-sa-4.0`. Share-alike would put a CC BY-SA obligation on a copy of the questions. The questions are also drawn from exams, so a compiler's licence may not clear the exam boards. |
| HellaSwag | On 2026-10-16 the primary repository `rowanz/hellaswag` was unavailable: GitHub has disabled it after a DMCA takedown notice posted as `2026-09-14-wikihow`. |
| MATH (Hendrycks) | The code repository's `LICENSE` is MIT ("Copyright (c) 2021 Dan Hendrycks", same grant shape as GSM8K's, at <https://github.com/hendrycks/math/blob/main/LICENSE>). The problem text is a separate download. Hugging Face staff recorded an Art of Problem Solving DMCA against `hendrycks/competition_math` on 15 January 2025, and the code repository's dataset link has since been reported down. The MIT line was not read as a grant of the competition wording. |
| CommonsenseQA | The Hugging Face card says MIT and points at `jonathanherzig/commonsenseqa` issue 5. That issue is a question asking what the licence is. No author reply was found. An unanswered question is not a grant. |
| GPQA | The Hugging Face card `Idavidrein/gpqa` says CC BY 4.0 and asks that examples not be revealed in plain text online. A panel file is plain text. Graduate items are also the wrong length for a 12–15 minute sitting. Not measured; the mismatch is the authors' own description of the questions. |

## 3. Overlap with the study's own text

The panel must not overlap the training data, the sealed set, DEV or the interview. The check is two passes. Both have to come back empty before an item is kept. This draft did not build the item file, so the passes have not been run on the 280. A rehearsal on the whole source files, against the text that is in the tree at `6f06c9b`, is in §3.4.

### 3.1 Pass A — `gates.leakage`, with this schema

`leakage(rows, held_out, interview)` is the function in `packages/aspire-si-gates/src/aspire_si_gates/gates.py` at this commit. It does not call `_shingles`. It fails a row when:

- a user turn, `_statement(user)`, `variant.user`, or the `statement` field equals a held-out claim after its normaliser, or is within a 5-character-gram Jaccard of 0.8 of one;
- the row shares 2 or more normalised material lines (length at least 12) with a held-out item;
- the row's kind is A or E, and any of those texts or a turn's `reply` is within a 5-character-gram Jaccard of 0.5 of an interview question.

**Panel rows, so the interview arm applies.** Each candidate is wrapped as kind E. `leakage` reads `turns[].reply`, so the reply key is present and empty.

```text
{"id": "<panel item id>", "kind": "E",
 "turns": [{"user": "<the full prompt from §5>", "reply": ""}],
 "statement": "", "variant": null, "material": ""}
```

**Training and apply rows, mapped to `held_out`.** One dict per training row of both arms, and one per apply item:

```text
claim    = the row's statement field, if it is non-empty
           else gates._statement(the first user turn)
           else that user turn's full text
material = the row's material field, or empty
id       = the row's id
```

`_statement` is the text after `statement:` in the user turn. Kind B keeps its claim on the `statement` field, which is why that field is first. ROLE is every file the freeze lists. NO-ROLE is the pinned `no_role.py` output. Apply is the apply-item file. A row with no turns and no statement is a bug in the mapping and stops the check.

**Sealed and DEV, same dict shape, added here.** Amendment B's mapping sentence names training and apply. The overlap rule also names the sealed set and DEV. They are passed in the same `held_out` list: `claim` is the task's `claim`, `material` is the task's `material`, `id` is the task's id. Leaving them out would not check the rule this note is under.

**Interview.** The 12 strings in `stage1-eval/interview.py` `QUESTIONS`, the same list `train.py` passes into `leakage`. The kind E wrap is what makes the 0.5 Jaccard apply.

A hit rejects the candidate. The sampler takes the next row and records the hit line `leakage` returned.

### 3.2 Pass B — the 8-word span check

Separately, `gates._shingles(text, 8)` is intersected with the 8-word spans of:

- every training turn of both arms, each of `user`, `thinking` and `reply`;
- every apply turn, the same three fields;
- every sealed task and every DEV task, `claim` and `material`.

The panel side of the intersection is the source text from §1, and again the full prompt. A shared instruction line that hits training means the instruction is rewritten, not that the source is dropped. A hit on the source text drops the item.

`_shingles` keeps runs of letters and apostrophes. Digits are not part of a span. `gates.line_spans` is a different tokenizer (it keeps digits, and it collapses a bracketed quote). This check does not call `line_spans`. The STaR near-duplicate drop uses `line_spans` on traces. The panel's overlap check is `_shingles`, as specified for this design.

### 3.3 Public items in the base model

The base model's pretraining corpus is not one of these sets. GSM8K and WinoGrande are public, and Qwen3-8B (`b968826d9c46`) may already have seen them. This note did not inspect that corpus. R&D accepts that for this guard:

- The panel is a change detector against P0 on the same items. It is not a capability score, and it is not reported as one.
- Losing items the base model already knows, memorised or not, is the loss the guard is for.
- The cost is ceiling compression. A slice near the top has few discordant pairs. That moves the critical drop down (at 280 items and 5% discordance, 3.6 points; §8), and it leaves fewer items that can show a loss.
- P0's run note reports accuracy per slice. A slice at or above 97% is flagged there for the maintainer before G1.

### 3.4 Rehearsal, counted, not the pin

Against the 640 training rows in `docs/runs/stage1-training-data/*.jsonl` at this commit (666 turns; not batches 7–8, not NO-ROLE, not apply, not the sealed set):

| Source file, all rows | 8-word hits on those training rows | 8-word hits on the 12 interview questions | Jaccard ≥ 0.5 on the 12 interview questions | 8-word hits on the 30 taskset rows in the tree (`taskset-pilot-20`, `taskset-review-10`) |
|---|---|---|---|---|
| GSM8K test, 1,319 | 0 | 0 | 0 | 0 |
| WinoGrande dev, 1,267 | 0 | 0 | 0 | 0 |

The sealed file was not read. It is outside the repository. The real check is §3.1 and §3.2 on the frozen files, before P0, on the 280.

## 4. Scoring

Fully mechanical. No model judge.

The generation record is the batched path's, which carries the harness's fields. The scorer starts:

```text
thinking, answer, _ = harness.split_thinking(g["text"])
```

`split_thinking` returns the text after `</think>`, or an empty answer when the tag never closes. Only `answer` is parsed. That string is the answer side. The thinking is stored and never scored.

**Letter (WinoGrande).** Score the last line on the answer side that matches `^ANSWER:\s*([AB])\s*$`. Other lines are ignored. The letter must be one of the item's options. No matching line is unparsed. `ANSWER: C` does not match. A correct letter with an extra sentence is still correct: the guard measures whether the model can still do the task, and format bleed from ROLE's three-line habit is not read as a loss of arithmetic.

**Integer (GSM8K).** Score the last line that matches `^ANSWER:\s*(-?\d[\d,]*)\s*$`. Other lines are ignored. Commas are removed. The value must equal the gold integer. The gold is the `####` integer in the source answer, commas removed. **Tolerance is 0.** A decimal, a fraction, or a unit does not match, so it is ignored. No rounding. No matching line is unparsed.

The panel keeps its own pinned scorer (§9). It does not call the gates' `parse_choice`, and it does not use that pattern. `parse_choice` takes the last line matching `ANSWER:` with an optional parenthesis and an optional period (`ANSWER_RX`). The panel's patterns are the two anchored ones above.

**Strict one-line compliance, counted and not tested.** Per sitting, count the items whose answer side, stripped, is exactly one line and that line matches the item's pattern. The count is descriptive. It does not enter the McNemar table. A hold can then be read as format bleed or as wrong answers.

**Wrong, and the two extra counts.** A row is wrong when `g["truncated"]` is true, or the answer is empty, or no line matches, or the parsed value differs from the gold. Truncation is the harness rule: the new-token count reached 16,384 and the sequence did not end on a stop token. The two extra counts are independent of each other, so a capped row with no parse is in both:

- **truncated:** `g["truncated"]` is true;
- **unparsed:** the answer is empty, or no line matches that item's pattern.

Accuracy's numerator is the rows that parsed, matched, and were not truncated. The denominator is every item, 280. A drop is that accuracy minus P0's, in percentage points.

## 5. The prompt

One user message. Thinking stays on, so the model may think at length; the scored text is only what follows `</think>`. The prompt asks the model to end on one answer line. If more comes back, the scorer still takes the last matching line (§4).

GSM8K:

```text
Solve the problem. Think it through, then end your reply with one line: ANSWER: <the final integer>. No units and no commas.

<question, unchanged from the source>
```

WinoGrande. The blank in the sentence is the source's underscore. Option 1 is A and option 2 is B.

```text
The sentence has a blank, shown as an underscore. Choose the option that fills the blank. Think it through, then end your reply with one line: ANSWER: A or ANSWER: B.

Sentence: <sentence>
A. <option1>
B. <option2>
```

The prompt text is part of the item file. Its sha256 is recorded with the file (§9). A wording change after P0 is a new file and a new baseline.

## 6. Running, per amendment C

One sampled pass per item. No greedy decoding. The card settings are the harness's `DECODING["sampled"]`, imported by `batched.py` rather than copied:

| Setting | Value | Where |
|---|---|---|
| temperature | 0.6 | `harness.DECODING["sampled"]` |
| top-p | 0.95 | the same |
| top-k | 20 | the same |
| min-p | 0 | asserted in `batched.py`; a no-op |
| new-token cap | 16,384 | `harness.MAX_NEW_TOKENS` |
| thinking | on | `harness.THINKING` |

The seed is `batched.seed_for(item_id, 0)`: a 63-bit value from sha256 of `f"{item_id}|0"`. Sample index 0 is the one pass amendment C names as `(item id, 0)`. The same id and the same index are used at P0 and at every later sitting, so a row pairs with itself.

**Order into the batched path.** Items go into `batched.make_generate_many` in the pinned item file's order. That list is the same at P0 and at every later sitting. Each ask's `item_id` is the bare id from §1, and its `sample` is 0. The id is never suffixed with a round or a seed. `seed_for` hashes `f"{item_id}|{sample}"`, so a suffix would draw a different generator and the row would no longer pair with P0. The ask's key is that same bare id. `plan` sorts a first chunk by prompt length and then by key, and a stable key keeps that sort the same every sitting.

Generation is `batched.generate_all` through `make_generate_many`, once parity (§5 of the batched-generation note) has passed. The panel does not add a path. DEV, the sealed set and the interview stay on the batch-1 harness.

**Before the model loads, at P0 and at every later sitting.** The panel runner checks the frozen machine line and sets the memory cap, and only then loads the student.

The frozen line is `docs/runs/stage1-machine-line.json`, sha256 `2daa5aa9`, the pin in run sheet §9. The check is `det_run`'s. `machine_line()` is compared with that file on `gpu`, `driver`, `cuda`, `torch` and `transformers`. Any difference exits before the model loads, as `det_run` returns exit 5. An incomplete line, with no gpu or driver, exits the same way. The panel has no warn-only path.

The cap is the call `parity.py` makes, after CUDA has started and before the model loads:

```text
torch.cuda.set_per_process_memory_fraction(parity.cap_fraction(total, used_at_start))
```

`total` is the device's total memory. `used_at_start` is `memory.used` once the context exists. `cap_fraction` is `(parity.PLAN_BYTES − used_at_start) / total`, clamped to the range 0 to 1. `PLAN_BYTES` is `26e9`.

**The plan the sitting inherits.** `B_TOK` is 28,000 tokens of KV cache per batch. First chunks are held at 8. The last chunk of a set that is not a multiple of 8 is padded with throwaway copies of the last real row; their output is discarded, and they stop when every real row in the batch has stopped. Chunk length is `min(2048, 28000 // 8 − longest prompt)`, and at least 512. **Computed:** `28000 // 8 = 3500`, so a batch of 8 stays in budget with a chunk of at least 512 when the longest prompt is at most `3500 − 512 = 2988` tokens. Above that, `plan` splits the group and records `long-prompt`.

Word counts on the source files (**counted**): GSM8K questions have median 43 words, 90th percentile 70, maximum 164. WinoGrande sentences have median 18 words, maximum 30, and the options have median 1 word. A bound of 2 tokens per word on the longest GSM8K question plus 80 words of instruction is `(164 + 80) × 2 = 488` tokens. 488 is under 2,988, so the long-prompt split does not fire for these items. That bound is not a tokenization. The student tokenizer was not run.

Continuations happen only after a row has passed its first chunk. With the chunk at 2,048, a row of a few hundred new tokens is one prefill and one chunk. A row that runs to the 16,384 cap is re-prefilled on each continuation, and the continuation batches shrink because `rows × (length so far + 2048)` must stay ≤ 28,000.

**The 26 GB plan.** `parity.PLAN_BYTES` is `26e9` bytes of `memory.used`, covering torch, the CUDA context and the idle baseline. The runner sets that bound with the `cap_fraction` call above. The KV term in the batched note is 36 layers × 8 KV heads × 128 × K and V × 2 bytes. **Computed:** `36 × 8 × 128 × 4 = 147,456` bytes per token, times 28,000 = `4,128,768,000` bytes (4.13 GB decimal). That is the same budget the quiz and apply sittings use. The panel does not raise it. P0 loads the base model with no adapter, so its reserved memory is the lower of the two. The gate stays ≤ 26 GB, and parity's measured peak is what shows it. This draft did not measure a peak.

### 6.1 Two rates, and the 12–15 minutes

`parity.B1_TOK_S_GUESS = 20.0` is the batch-1 end-to-end rate: about 300 tokens in 16 seconds, used until a draw measures a real rate. It is one stream. It is not a profiled kernel time.

At batch 8 the decode loop takes one forward step for the whole batch and draws one token for each active row (`_decode`). Read as a step rate, the guess is **20 steps per second**.

- **Per-step decode rate at batch 8:** 8 tokens per step while all eight rows are active, so `8 × 20 = 160` tokens per second across the batch (**computed**, from the guess, not from a run).
- **Effective throughput:** real new tokens divided by wall time. With every row the same length, one chunk, and the step rate unchanged, wall time is `ceil(N / 8) × (L / 20)` seconds and the throughput is `N × 20 / ceil(N / 8)` tokens per second. At N = 280, `ceil(280 / 8) = 35` and `35 × 8 = 280`, so there is no padded slot and the throughput is **160 tokens per second** (**computed**). At 250 items it would be `250 × 20 / 32 = 156.25` (6 padded slots). At 300 it would be `300 × 20 / 38 = 157.89` (4 padded slots).

**What length lands in 12–15 minutes, under that reading.** 35 batches:

| Sitting | Seconds | Mean new tokens L, at 20 steps/s |
|---|---|---|
| 12 min | 720 | `720 / 35 × 20 = 411` |
| 15 min | 900 | `900 / 35 × 20 = 514` |

The verifier baseline's median was about 325 new tokens (baseline run note, 2026-10-10). At that length a panel sitting would be `35 × 325 / 20 = 568.75` seconds, which is **9.48 minutes** (**computed**). That is shorter than amendment C's window. The window fits if these questions draw about 411–514 new tokens, a bit above that verifier median. Panel thinking length was not measured. Qwen3 thinking on a short question can also run much longer than 325, and one row at the cap is `16384 / 20 = 819.2` seconds of steps under this reading (**computed**), most of a 15-minute sitting, before its re-prefills. P0 is where that length becomes a number. The run sheet already sizes the next grant from the sitting that just finished.

**The other reading.** If a batch-8 step costs eight times a batch-1 step, the aggregate rate stays 20 tokens per second and batching does not shorten the wall clock. At 325 tokens, 280 items take `280 × 325 / 20 = 4,550` seconds, **75.8 minutes** (**computed**). Amendment C's 12–15 minutes then does not hold. The guess of 20 tokens per second is far below a weight-bandwidth ceiling for an 8B model, so an unchanged step time is plausible, and it is not evidence. Parity's measured batch-8 minutes, on its own set, are the number that replaces this guess. They were not available for this draft.

Prefill is inside the 20 token/s guess only as it fell out for generations of about 300 tokens. A short answer makes prefill a larger share. Not separated here.

**What decides the booking.** Both readings are unmeasured. P0 waits for B4's parity pass, which measures the batch-8 rate on the card. If that rate puts a 280-item sitting above 20 minutes, the cost goes back to the maintainer before P0 is booked, with the 19-sitting total from §7. The item count stays 280. It is not cut below amendment C's 250 to make the time fit.

## 7. Sittings

The panel is ROLE only. The run sheet gives NO-ROLE training, the quiz and apply, and not the panel (§4, §7: "Phase A's DEV, sealed and panel are ROLE's"). Amendment C's "both arms" is the contents of each per-seed grant, not a second panel.

| Sitting | Model | When | Grant |
|---|---|---|---|
| P0 | the base model, no adapter, one pass at `(item id, 0)` | before any training | G0, its own grant, before G1 (run sheet Q3) |
| Round r, seed 0 | that seed's ROLE adapter, the same ids and the same sample index | after that seed's training, quiz, apply and DEV (blocks T, Q, A, D, then P) | that seed's grant |
| Round r, seed 1 | the same, seed 1 | the same point in seed 1's grant | the next per-seed grant, after a 15-minute rest |
| Round r, seed 2 | the same, seed 2 | the same point in seed 2's grant | the third per-seed grant, after a 15-minute rest |

Three ROLE sittings per round. The interview block stays where the run sheet puts it, after the three ROLE adapters. It is not part of the panel.

**How many rounds.** The cost below is quoted at the evaluation plan's cap of 6 rounds. What "session" means in the training procedure is not this note's to settle. Every reading of that word gives 6 rounds or fewer, and the panel's per-round cost does not change with it.

**19 is the ceiling.** P0 plus 6 rounds × 3 seeds = **19**. Fewer if the stage stops earlier. The item file and the seeds do not change between them.

**Panel card time at that ceiling, if each sitting is 12–15 minutes:** `19 × 12 = 228` minutes (3 h 48 min) to `19 × 15 = 285` minutes (4 h 45 min) (**computed**). That is generation only. It does not include training, DEV, the sealed sittings, preflight, or the 15-minute rests between per-seed grants. At the 9.48 minute figure from a 325-token mean, 19 sittings are `19 × 9.48 = 180` minutes (**computed**). Per round the panel itself is 3 sittings, 36–45 minutes at the amendment C window. If parity's measured rate puts one sitting above 20 minutes, this 19-sitting total is what goes back to the maintainer before P0 is booked (§6.1).

Amendment C governs the grants: three per-seed grants, with 15-minute rests. Block P closes each per-seed grant, after T, Q, A and D. The run sheet's single G1 line is stale. That fix belongs to the run sheet.

## 8. The guard, per amendment B

Round progression stops for the maintainer when the panel is more than 2.0 points below its baseline **and** the drop is significant at p ≤ 0.05 by a paired McNemar test on per-item correctness. A smaller drop is reported and does not stop the round. This replaces the plain 2-point line.

**The test this draft computed.** One table of N items. b is P0 right and the round wrong. c is P0 wrong and the round right. The drop in points is `(b − c) / N × 100`. The p-value is the exact conditional test: under the null, b is Binomial(b + c, 0.5), two-sided, and the p-value is the total null probability of the outcomes whose null probability is at most the observed one's. That is the small-p binomial test. It was computed with `math.lgamma`, not with a library. The continuity-corrected chi-square, `(|b − c| − 1)² / (b + c)` against a chi-square of 1 degree, agreed with the exact test on the critical count at every cell of N = 250, 275 and 300. Amendment B says "paired McNemar" and does not say "exact". The evaluation plan's other McNemar, on the sealed set, is the exact one. This draft uses the exact test for that reason.

**Worked cell, the disclosure's own size.** N = 300, discordance 15%, so `nd = round(0.15 × 300) = 45`. The smallest table that reaches p ≤ 0.05 is b = 30, c = 15, difference 15, drop `15 / 300 × 100 = 5.0` points.

- Exact p at difference 13 (b = 29, c = 16): **0.072** (**computed**). Does not stop.
- Exact p at difference 15 (b = 30, c = 15): **0.036** (**computed**). Stops, if the drop is also over 2.0, which 5.0 is.
- Chi-square with the continuity correction at difference 15: `(15 − 1)² / 45 = 196 / 45 = 4.356`, p = **0.037** (**computed**).

Power below is the probability that a true drop of that size produces a significant table, with the discordant count fixed at `round(rate × N)` and the expected split equal to the drop. "Reliably" in the corrected disclosure is 80% power. The critical value is the smaller number, and both are in the table.

**Computed power of the exact test.** A blank means the drop is larger than the discordance rate, so it cannot occur.

| N | Discordant | nd | Smallest drop that can reach p ≤ 0.05 | Power at 2 pt | 3 | 4 | 5 | 6 | 8 | Smallest drop with power ≥ 0.80 |
|---|---|---|---|---|---|---|---|---|---|---|
| 250 | 5% | 12 | 3.2 pt | 0.27 | 0.60 | 0.93 |  |  |  | 3.6 pt |
| 250 | 10% | 25 | 4.4 | 0.16 | 0.31 | 0.51 | 0.73 | 0.89 | 1.00 | 5.4 |
| 250 | 15% | 38 | 5.6 | 0.10 | 0.18 | 0.31 | 0.47 | 0.65 | 0.91 | 7.1 |
| 250 | 20% | 50 | 6.4 | 0.08 | 0.14 | 0.24 | 0.36 | 0.51 | 0.78 | 8.2 |
| 250 | 25% | 62 | 7.2 | 0.07 | 0.11 | 0.19 | 0.28 | 0.40 | 0.66 | 9.3 |
| 280 | 5% | 14 | 3.6 | 0.16 | 0.45 | 0.84 |  |  |  | 3.9 |
| 280 | 10% | 28 | 4.3 | 0.15 | 0.31 | 0.53 | 0.75 | 0.91 | 1.00 | 5.3 |
| 280 | 15% | 42 | 5.0 | 0.13 | 0.24 | 0.39 | 0.57 | 0.74 | 0.95 | 6.4 |
| 280 | 20% | 56 | 5.7 | 0.11 | 0.19 | 0.31 | 0.45 | 0.60 | 0.86 | 7.5 |
| 280 | 25% | 70 | 6.4 | 0.09 | 0.15 | 0.24 | 0.36 | 0.49 | 0.75 | 8.5 |
| 300 | 5% | 15 | 3.0 | 0.30 | 0.65 | 0.94 |  |  |  | 3.5 |
| 300 | 10% | 30 | 4.0 | 0.18 | 0.36 | 0.59 | 0.80 | 0.94 | 1.00 | 5.0 |
| 300 | 15% | 45 | 5.0 | 0.12 | 0.23 | 0.38 | 0.57 | 0.75 | 0.96 | 6.4 |
| 300 | 20% | 60 | 6.0 | 0.08 | 0.15 | 0.26 | 0.40 | 0.56 | 0.84 | 7.7 |
| 300 | 25% | 75 | 6.3 | 0.08 | 0.15 | 0.24 | 0.37 | 0.50 | 0.77 | 8.3 |

**The disclosure, as corrected.** At 280 items and 15% discordance, the guard can first stop at a 5.0-point drop and stops reliably (80% power) at about 6.4; drops of 2–5 points are reported, not acted on. The same two numbers hold at 300 items and 15% discordance. A true 4-point drop at the 300-item cell has power 0.38, and a true 5-point drop has power 0.57, so many 5-point drops are reported rather than acted on. The 2.0-point line does not bind: at every cell in the table, a 2-point drop has power under 0.30.

Amendment B's "about 4–5 points" was optimistic on this arithmetic. The rule itself does not change: more than 2.0 points, and exact p ≤ 0.05. R&D files the corrected sentence as a dated note under amendment B, in rnd and in the evaluation plan, when this design is approved. This note records the wording and does not edit those files.

More discordance makes a point-drop harder to call significant, because the same point-drop is a smaller imbalance inside a larger discordant count. A very low discordance (5%) is the only band in which a drop near 4 points is both detectable and well powered, and only because a 5% discordance cannot hide a large swing.

### 8.1 One table per seed

One table per seed, each against the same P0. A per-item majority of three adapters against P0's single sample is lopsided: where a model is right with probability above 0.5, a majority of three is right more often than one sample, and where that probability is below 0.5 it is right less often. The bias depends on where the items sit, and it is not the quantity amendment B names. Per seed, every table is one sampled pass against one sampled pass, which is amendment C's shape, and N is the disclosure's N. Pooling three seeds into 840 pairs is not that N.

The round holds for the maintainer when any one seed's table meets amendment B: a drop of more than 2.0 points, and exact p ≤ 0.05. There is no multiplicity adjustment, because a hold is a stop for a person, not a verdict.

Each round's receipt carries all three tables (b, c, N, the drop, the exact p) and the mean of the three drops. The mean is descriptive. It is not the test. The power table above is for one of those tables.

**False-hold rate, computed.** Under no true drop, a hold needs b > c and exact p ≤ 0.05. At N = 280 and the five discordant counts in the power table, the smallest significant drop is already above 2.0 points, so the 2.0-point line removes no further table. The one-table rate is the upper tail of Binomial(nd, 0.5) from the critical b.

Worked cell, nd = 14. `2^14 = 16,384`. The critical count is b ≥ 12 (c ≤ 2, difference 10, drop `10 / 280 × 100 = 3.57`, the table's 3.6). `C(14,12) + C(14,13) + C(14,14) = 91 + 14 + 1 = 106`. One table: `106 / 16,384 = 0.00647`. At b = 11 the two-sided probability is `2 × 470 / 16,384 = 0.0574` (`C(14,3) = 364`, plus 106), which is above 0.05, so b = 11 is not in the rate.

The other four tails were summed the same way.

| Discordant pairs | 14 | 28 | 42 | 56 | 70 |
|---|---|---|---|---|---|
| Critical b (c) | 12 (2) | 20 (8) | 28 (14) | 36 (20) | 44 (26) |
| One table | 0.00647 | 0.01785 | 0.02178 | 0.02202 | 0.02070 |
| Three seeds, independent | 0.0193 | 0.0526 | 0.0639 | 0.0646 | 0.0608 |
| Over 6 rounds | 0.110 | 0.277 | 0.327 | 0.330 | 0.314 |

Three seeds is `1 − (1 − p)^3`. Six rounds is `1 − (1 − p3)^6`. From the nd = 14 rate: `1 − (1 − 0.00647)^3 = 0.0193`, and `1 − (1 − 0.0193)^6 = 0.110`. Rounded, these are the ruling's rates: one table 0.007, 0.018, 0.022, 0.022, 0.021; three seeds 0.019, 0.053, 0.064, 0.065, 0.061; over 6 rounds 0.11, 0.28, 0.33, 0.33, 0.31.

The three-seed and six-round rows assume independent seeds. All three seeds share P0, so their tables are positively correlated, and the true union rate is lower than these figures. The six-round row stacks that assumption a second time.

**Argon's note, 2026-10-16.** Argon approved the any-seed hold. The 6% per round and the 33% over 6 rounds are independence upper bounds. They are the nd = 42 cell above, 0.0639 and 0.327, at about 15% discordance. The seeds share P0 and the items, so the true rates are lower. Argon's figures for those true rates are about 4–5% per round and 15–22% over 6 rounds. This note did not recompute them. When a hold fires, the readout shows whether the drop is in one seed or shared across seeds: which of the three tables meet the hold, and whether the items that flipped are the same ones in those seeds.

## 9. Pins, before P0

Nothing in this pull request is the pin. The hashes do not exist until the files do.

| What | Where it lives | What is written down, before G0 starts |
|---|---|---|
| Item file | outside the repository, with the GSM8K notice and the WinoGrande attribution file beside it (§2). Not a product file, and not committed here. | sha256 of the file; the prompt-template sha256; the source commits in §2; the sample order and every rejection. Copied into the P0 run note. |
| Panel scorer | a new file, not written by this note. The panel's own scorer: the two anchored patterns in §4, last matching line. It does not call `parse_choice`. | sha256 in the same run note, and in each later panel receipt. The sitting refuses a missing or a changed hash, the way the other sittings refuse a changed harness. |
| Harness and batched path | already pinned | the run sheet's §9 list, plus `batched.py`'s sha256 and the path name `batched-v1`, recorded in the panel receipt as the quiz readout records them. |
| P0 outputs | the G0 grant's run directory | one record per item: id, seed, text, parsed value, right or wrong, truncated, unparsed, strict one-line, new tokens, seconds. Accuracy per slice, the two extra counts, the strict one-line count, and the machine line. A slice at or above 97% is flagged for the maintainer before G1. |

Later rounds record the same, plus all three tables against P0 (b, c, N, the drop in points, the exact p) and the mean of the three drops. The mean is descriptive. When a hold fires, the readout says whether it is one seed or shared across seeds (§8.1). The baseline is P0 and no other sitting.

## 10. What this draft did not verify

- No item file was built, so §3 was not run on the 280. The rehearsal in §3.4 is the whole source files against the training rows, the interview and the 30 taskset rows that are in the tree. Not against the sealed set, NO-ROLE, apply, or batches 7–8, which are not there yet.
- The student tokenizer was not run. Prompt length is a 2-tokens-per-word bound.
- No generation was run. The 20 token/s figure is `B1_TOK_S_GUESS`. The 12–15 minute window is that guess read as a step rate that does not change at batch 8. The 75.8 minute figure is the reading where batch 8 does not speed the wall clock. Parity's measured rate is the one that decides.
- Memory was not measured. The 4.13 GB KV term is arithmetic from the batched note's shape. The ≤ 26 GB gate is `parity.PLAN_BYTES`.
- Qwen3's pretraining corpus was not checked for these items. §3.3 accepts that for this guard.
- Licences: R&D verified both at the primary source on 2026-10-16. This revision records that check. WinoGrande's version is still not stated by AllenAI. The CC BY 4.0 legal-code quotes from the first draft are removed.
- The McNemar power, and the false-hold tails, fix the discordant count. They do not model a random discordant count, and they do not model dependence left after the 8-word filter. The three-seed and six-round rates add an independence assumption. Argon's lower ranges in §8.1 were not recomputed here.
- The dated disclosure note under amendment B is not filed here. The maintainer has approved the design. R&D still files that note, in rnd and in the evaluation plan. The wording is in §8.

The questions from the first draft are settled in §2 through §8. The three additions from the Publisher's comment are in §6 and §8.1.
