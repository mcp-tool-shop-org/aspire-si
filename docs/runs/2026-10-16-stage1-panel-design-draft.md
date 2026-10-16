# DRAFT: the Stage 1 general-capability panel (B3)

**Status: draft for R&D, 2026-10-16. Not approved.** This note settles the panel's size, sources, scoring, running, cost and stop rule, with the arithmetic shown. It adds no code, no item file and no scorer. R&D revises it. Argon and the maintainer approve it. Nothing here has run on the card.

**Which text this follows.** Two dated amendments govern the panel. Amendment B (evaluation plan, 2026-10-15) is the stop rule. Amendment C (training procedure, 2026-10-15) is the running. Where the round-1 run sheet still says the plain 2-point guard (§4, §5), amendment B has replaced that line. This draft follows amendment B and does not edit the run sheet. Where the run sheet draws G1 as one grant and amendment C draws it as three per-seed grants, the sittings below follow amendment C, and the conflict is an open question.

The tree this was read against is `verifier-distill-prereg-draft` at `6f06c9b`. Gates behaviour is aspire-si-gates 1.7.3 as that commit has it. Every figure marked **computed** was calculated in this draft. Figures marked **counted** were read off the named source file. Nothing else was measured.

## 1. Size and composition

**280 items. 140 from GSM8K's test split, 140 from WinoGrande 1.1's dev split.** 280 is inside amendment C's 250–300, and it divides by the batch of 8, so a sitting has 35 batches and no tail (35 × 8 = 280).

| Slice | Items | What it measures | Answer | Score |
|---|---|---|---|---|
| GSM8K test | 140 | multi-step grade-school arithmetic | an integer | exact, tolerance 0 |
| WinoGrande 1.1 dev | 140 | commonsense coreference (fill the blank) | the letter A or B | exact letter |

Both are ordinary questions. Neither is a verifier item: no verdict, no DECIDING, no ESCALATE. `score_three_lines` is the quiz and apply scorer for those three lines. It is not the panel scorer.

The short-exact-string parser is specified in §4 so the third answer type has a rule. This composition does not use it. The sources that would have supplied short strings were share-alike, unclear, or taken down (§2). That is an open question, not a silent drop.

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

The data file lives in that repository, so the grant covers it. The notice has to travel with any copy of the items. No other restriction is stated.

### WinoGrande — used

- **Name and version.** WinoGrande version 1.1, the dev split (`dev.jsonl` with `dev-labels.lst`). The download script at commit `727e837f77521ef38bcc56df3b275c8da43f45af` fetches `https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip`.
- **URL.** <https://github.com/allenai/winogrande> (README at that commit: <https://github.com/allenai/winogrande/blob/727e837f77521ef38bcc56df3b275c8da43f45af/README.md>).
- **Size.** 1,267 dev rows (**counted**). Fields: `qID`, `sentence`, `option1`, `option2`, `answer`. `answer` is `"1"` on 628 rows and `"2"` on 639. Every sentence contains the blank `_`. The label file matches `answer` on every row (**counted**). The test split has no labels in the zip, so it is not used.
- **Sample.** The first 140 that survive §1 and §3. Option 1 is the letter A and option 2 is the letter B.
- **Licence.** The README's own sentence, quoted from the commit above:

> Winogrande (codebase) is licensed under the Apache License 2.0. The dataset is licensed under CC-BY.

The README does not name a CC BY version and does not link a deed. The quote below is CC BY 4.0, from <https://creativecommons.org/licenses/by/4.0/legalcode>, and it is the usual reading of an unversioned "CC-BY". R&D confirms the version against AllenAI before any copy is kept.

> Section 2(a)(1). Subject to the terms and conditions of this Public License, the Licensor hereby grants You a worldwide, royalty-free, non-sublicensable, non-exclusive, irrevocable license to exercise the Licensed Rights in the Licensed Material to: a. reproduce and Share the Licensed Material, in whole or in part; and b. produce, reproduce, and Share Adapted Material.

> Section 3(a)(1). If You Share the Licensed Material (including in modified form), You must: retain the attribution the licensor supplied (creator, copyright notice, licence notice, warranty disclaimer, and a link to the material); indicate modifications; and indicate that the material is under this Public License, with the text of it or a link to it.

Restriction, stated plainly: attribution is required whenever the items are shared. CC BY 4.0 is not share-alike. The codebase's Apache-2.0 is the code only; the dataset line is the CC-BY sentence.

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

### 3.3 What the check cannot see

The base model's pretraining corpus is not one of these sets. GSM8K and WinoGrande are public. If Qwen3-8B (`b968826d9c46`) already saw them, the panel measures retention on public items, and a small loss can hide under a high baseline. This draft did not inspect that corpus. Whether that is acceptable is a question for R&D.

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

`split_thinking` returns the text after `</think>`, or an empty answer when the tag never closes. Only `answer` is parsed. The thinking is stored and never scored.

**Letter (WinoGrande).** The answer, stripped, is one line matching `^ANSWER:\s*([AB])\s*$`. The letter must be one of the item's options. `ANSWER: C` on an A/B item is unparsed. This is stricter than the quiz's `parse_choice`, which takes the last line matching `ANSWER:` with an optional parenthesis and an optional period, and ignores every other line (`ANSWER_RX` in the gates). The panel does not call `parse_choice`.

**Integer (GSM8K).** The answer, stripped, is one line matching `^ANSWER:\s*(-?\d[\d,]*)\s*$`. Commas are removed. The value must equal the gold integer. The gold is the `####` integer in the source answer, commas removed. **Tolerance is 0.** A decimal, a fraction, a unit, or a second line is unparsed. No rounding.

**Short exact string (specified, unused).** The answer, stripped, is one line matching `^ANSWER:\s*(.+?)\s*$`, and that text equals the single gold string after whitespace is collapsed and both sides are case-folded. One gold string, no alias list. No item in the 280 uses this parser.

**Wrong, and the two extra counts.** A row is wrong when `g["truncated"]` is true, or the answer is empty, or the parser fails, or the parsed value differs from the gold. Truncation is the harness rule: the new-token count reached 16,384 and the sequence did not end on a stop token. The two extra counts are independent of each other, so a capped row with no parse is in both:

- **truncated:** `g["truncated"]` is true;
- **unparsed:** the answer is empty, or the parser for that item's type fails.

Accuracy's numerator is the rows that parsed, matched, and were not truncated. The denominator is every item, 280. A drop is that accuracy minus P0's, in percentage points.

## 5. The prompt

One user message. Thinking stays on, so the model may think at length; the scored text is only what follows `</think>`. The prompt asks for a single answer line so the parser has one shape.

GSM8K:

```text
Solve the problem. On the answer side, write one line and nothing else:

ANSWER: <the final integer>

No units and no comma.

<question, unchanged from the source>
```

WinoGrande. The blank in the sentence is the source's underscore. Option 1 is A and option 2 is B.

```text
The sentence has a blank, shown as an underscore. Choose the option that fills the blank. On the answer side, write one line and nothing else:

ANSWER: A

or

ANSWER: B

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

Generation is `batched.generate_all` through `make_generate_many`, once parity (§5 of the batched-generation note) has passed. The panel does not add a path. DEV, the sealed set and the interview stay on the batch-1 harness.

**The plan the sitting inherits.** `B_TOK` is 28,000 tokens of KV cache per batch. First chunks are held at 8. The last chunk of a set that is not a multiple of 8 is padded with throwaway copies of the last real row; their output is discarded, and they stop when every real row in the batch has stopped. Chunk length is `min(2048, 28000 // 8 − longest prompt)`, and at least 512. **Computed:** `28000 // 8 = 3500`, so a batch of 8 stays in budget with a chunk of at least 512 when the longest prompt is at most `3500 − 512 = 2988` tokens. Above that, `plan` splits the group and records `long-prompt`.

Word counts on the source files (**counted**): GSM8K questions have median 43 words, 90th percentile 70, maximum 164. WinoGrande sentences have median 18 words, maximum 30, and the options have median 1 word. A bound of 2 tokens per word on the longest GSM8K question plus 80 words of instruction is `(164 + 80) × 2 = 488` tokens. 488 is under 2,988, so the long-prompt split does not fire for these items. That bound is not a tokenization. The student tokenizer was not run.

Continuations happen only after a row has passed its first chunk. With the chunk at 2,048, a row of a few hundred new tokens is one prefill and one chunk. A row that runs to the 16,384 cap is re-prefilled on each continuation, and the continuation batches shrink because `rows × (length so far + 2048)` must stay ≤ 28,000.

**The 26 GB plan.** `parity.PLAN_BYTES` is `26e9` bytes of `memory.used`, covering torch, the CUDA context and the idle baseline. The KV term in the batched note is 36 layers × 8 KV heads × 128 × K and V × 2 bytes. **Computed:** `36 × 8 × 128 × 4 = 147,456` bytes per token, times 28,000 = `4,128,768,000` bytes (4.13 GB decimal). That is the same budget the quiz and apply sittings use. The panel does not raise it. P0 loads the base model with no adapter, so its reserved memory is the lower of the two. The gate stays ≤ 26 GB, and parity's measured peak is what shows it. This draft did not measure a peak.

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

## 7. Sittings

The panel is ROLE only. The run sheet gives NO-ROLE training, the quiz and apply, and not the panel (§4, §7: "Phase A's DEV, sealed and panel are ROLE's"). Amendment C's "both arms" is the contents of each per-seed grant, not a second panel.

| Sitting | Model | When | Grant |
|---|---|---|---|
| P0 | the base model, no adapter, one pass at `(item id, 0)` | before any training | G0, its own grant, before G1 (run sheet Q3) |
| Round r, seed 0 | that seed's ROLE adapter, the same ids and the same sample index | after that seed's training, quiz, apply and DEV (blocks T, Q, A, D, then P) | that seed's grant |
| Round r, seed 1 | the same, seed 1 | the same point in seed 1's grant | the next per-seed grant, after a 15-minute rest |
| Round r, seed 2 | the same, seed 2 | the same point in seed 2's grant | the third per-seed grant, after a 15-minute rest |

Three ROLE sittings per round. The interview block stays where the run sheet puts it, after the three ROLE adapters. It is not part of the panel.

**How many rounds.** The evaluation plan caps the stage at 6 rounds unless the maintainer raises the cap before round 1. The training procedure also names 12 sessions, and it ends the stage at the first of mastery, the learning-curve stop, and that session cap. If a session is a round, the cap of 6 binds first and 12 does not. If a session is one seed's pass, 12 sessions is 4 rounds of 3 seeds. This draft cannot see which of those the procedure meant. The total below uses the evaluation plan's cap of 6, and the open question records the other reading.

**Total sittings at that cap:** P0 plus 6 rounds × 3 seeds = **19**. Fewer if the stage stops earlier. The item file and the seeds do not change between them.

**Panel card time at the cap, if each sitting is 12–15 minutes:** `19 × 12 = 228` minutes (3 h 48 min) to `19 × 15 = 285` minutes (4 h 45 min) (**computed**). That is generation only. It does not include training, DEV, the sealed sittings, preflight, or the 15-minute rests between per-seed grants. At the 9.48 minute figure from a 325-token mean, 19 sittings are `19 × 9.48 = 180` minutes (**computed**). Per round the panel itself is 3 sittings, 36–45 minutes at the amendment C window.

The run sheet still describes G1 as one grant of about 6.6 h plus the panel, seed by seed inside it. Amendment C, which this note was told to follow, splits G1 into three per-seed grants with 15-minute rests. The panel sitting is the same block P either way. Which grant boundary it sits on is the open question.

## 8. The guard, per amendment B

Round progression stops for the maintainer when the panel is more than 2.0 points below its baseline **and** the drop is significant at p ≤ 0.05 by a paired McNemar test on per-item correctness. A smaller drop is reported and does not stop the round. This replaces the plain 2-point line.

**The test this draft computed.** One table of N items. b is P0 right and the round wrong. c is P0 wrong and the round right. The drop in points is `(b − c) / N × 100`. The p-value is the exact conditional test: under the null, b is Binomial(b + c, 0.5), two-sided, and the p-value is the total null probability of the outcomes whose null probability is at most the observed one's. That is the small-p binomial test. It was computed with `math.lgamma`, not with a library. The continuity-corrected chi-square, `(|b − c| − 1)² / (b + c)` against a chi-square of 1 degree, agreed with the exact test on the critical count at every cell of N = 250, 275 and 300. Amendment B says "paired McNemar" and does not say "exact". The evaluation plan's other McNemar, on the sealed set, is the exact one. This draft uses the exact test for that reason.

**Worked cell, the disclosure's own size.** N = 300, discordance 15%, so `nd = round(0.15 × 300) = 45`. The smallest table that reaches p ≤ 0.05 is b = 30, c = 15, difference 15, drop `15 / 300 × 100 = 5.0` points.

- Exact p at difference 13 (b = 29, c = 16): **0.072** (**computed**). Does not stop.
- Exact p at difference 15 (b = 30, c = 15): **0.036** (**computed**). Stops, if the drop is also over 2.0, which 5.0 is.
- Chi-square with the continuity correction at difference 15: `(15 − 1)² / 45 = 196 / 45 = 4.356`, p = **0.037** (**computed**).

Power below is the probability that a true drop of that size produces a significant table, with the discordant count fixed at `round(rate × N)` and the expected split equal to the drop. "Reliably" is reported two ways, because the disclosure's "about 4–5 points" matches one of them.

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

**Against the disclosure.** At about 300 items and about 15% discordance, the smallest drop that reaches p ≤ 0.05 is **5.0 points**. That matches "about 4–5 points" as the point where a table can first stop the round. A true drop of 4 points at that cell has power 0.38, and a true drop of 5 points has power 0.57. Power reaches 0.80 at **6.4 points**. So the disclosed 4–5 points is the critical value, and it is a little optimistic if "reliably" was meant as 80% power. At the pinned 280 and 15% discordance the same two numbers are 5.0 and 6.4. The 2.0-point line does not bind: at every cell in the table, a 2-point drop has power under 0.30. Drops that small are reported, as amendment B says, and they do not stop a round.

More discordance makes a point-drop harder to call significant, because the same point-drop is a smaller imbalance inside a larger discordant count. A very low discordance (5%) is the only band in which a drop near 4 points is both detectable and well powered, and only because a 5% discordance cannot hide a large swing.

### 8.1 How the three seeds enter one table

Amendment B states the test and the sensitivity at about 300 items. It does not say how the three training-seed adapters become the one bit per item that a table of 250–300 needs.

P0 is one sitting: one sample per item at `(item id, 0)` on the base model. Each round has three ROLE adapters, each with its own sitting at the same seeds. Three readings are compatible with N = 250–300. Pooling the three seeds into 840 or 900 pairs is not, and it is not what the disclosure describes.

1. **One table per seed, against the same P0.** Three tests, each of N items. The disclosure's N fits each test. The stop then needs a rule amendment B does not give: any seed, every seed, or the mean drop. Three tests also raise the chance that one seed stops the round by sampling noise. Not adjusted here.
2. **One table, per-item majority of the three seeds, against P0's single bit.** N stays 250–300. The round bit is right when at least 2 of the 3 adapters are right. That is a different quantity from the mean of the three accuracies, so the "2.0 points" would be points of majority-accuracy. Phase A's sealed check uses a majority, but there the majority is of samples of one model, on both sides. Here the three bits are three adapters, and P0 has only one sample.
3. **One table on the mean accuracy** is not a McNemar table. McNemar needs a bit per item.

This draft does not choose. The power table is for one table of N items, whichever reading supplies the bit. The choice is the first open question.

## 9. Pins, before P0

Nothing in this pull request is the pin. The hashes do not exist until the files do.

| What | Where it lives | What is written down, before G0 starts |
|---|---|---|
| Item file | outside the repository, with the source notices beside it. Not a product file. | sha256 of the file; the prompt-template sha256; the source commits in §2; the sample order and every rejection. Copied into the P0 run note. |
| Panel scorer | a new file, not written by this draft | sha256 in the same run note, and in each later panel receipt. The sitting refuses a missing or a changed hash, the way the other sittings refuse a changed harness. |
| Harness and batched path | already pinned | the run sheet's §9 list, plus `batched.py`'s sha256 and the path name `batched-v1`, recorded in the panel receipt as the quiz readout records them. |
| P0 outputs | the G0 grant's run directory | one record per item: id, seed, text, parsed value, right or wrong, truncated, unparsed, new tokens, seconds. The accuracy, the two extra counts, and the machine line. |

Later rounds record the same, plus the McNemar table against P0 (b, c, N, the drop in points, the exact p) for whatever reading §8.1 is given. The baseline is P0 and no other sitting.

## 10. What this draft did not verify

- No item file was built, so §3 was not run on the 280. The rehearsal in §3.4 is the whole source files against the training rows, the interview and the 30 taskset rows that are in the tree. Not against the sealed set, NO-ROLE, apply, or batches 7–8, which are not there yet.
- The student tokenizer was not run. Prompt length is a 2-tokens-per-word bound.
- No generation was run. The 20 token/s figure is `B1_TOK_S_GUESS`. The 12–15 minute window is that guess read as a step rate that does not change at batch 8. The 75.8 minute figure is the reading where batch 8 does not speed the wall clock. Parity's measured rate is the one that decides.
- Memory was not measured. The 4.13 GB KV term is arithmetic from the batched note's shape. The ≤ 26 GB gate is `parity.PLAN_BYTES`.
- Qwen3's pretraining corpus was not checked for these items.
- Licences were read from the pages named in §2 on 2026-10-16. WinoGrande's README says "CC-BY" and does not name a version. R&D verifies each one against the primary page.
- The McNemar power fixes the discordant count. It does not model a random discordant count, and it does not model dependence left after the 8-word filter.

## Open questions for R&D

1. **Pairing.** Per seed (three tables against one P0) or a per-item majority of the three seeds (one table)? Amendment B does not say. If per seed: any seed, or every seed?
2. **The disclosure's "4–5 points".** At 300 items and 15% discordance the critical value is 5.0 points and 80% power is 6.4 points. Which one was "reliably"?
3. **Grants.** Amendment C's three per-seed grants with 15-minute rests, or the run sheet's one G1? The panel block is the same either way.
4. **The cap.** Six rounds, as the evaluation plan says, or also the procedure's 12 sessions, and is a session a round or a seed?
5. **The letter parser.** The strict one-line form in §4, or the quiz's `parse_choice`?
6. **A short-string slice.** The parser is specified and unused. The public short-string sets that were checked do not have a permissive licence.
7. **Public items inside the base model.** If a high P0 is memorisation, the guard goes quiet. Is that acceptable for this panel?
