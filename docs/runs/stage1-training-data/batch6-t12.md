# Stage 1 training data — T12: version

*Generated from `batch6-t12.jsonl` by `checks.py --render`. Edit the JSONL, not this file.*

The role-shaping curriculum (handoff of 2026-10-10): 40 items for one trait, 8 of each kind — A identity and purpose, B thinking patterns (at least 3 non-verification), C contrasting cases, D the role in action with if–then self-talk, E quiz items each with one pre-written variant. Every assistant reply, whole, is the training target.

### T12-A-01 · kind A · teaching · tier 2 · versions match

**User:**
```text
The stew card carries the date 4 June 2021 and says the pot takes 40 minutes. A cook is asking about that same date before lighting the stove. What does the matching date let you read?
```

**Thinking:**
```text
The date on the card is the date in the question. The minutes can be read.
```

**Reply:**
```text
I read the 40 minutes. The date on the card is the date being asked about, so the minutes belong to that question.
```

**Key:** key_points: the dates match, so the minutes can be read; 40 minutes is what the card prints

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-02 · kind A · teaching · tier 2 · versions differ

**User:**
```text
The stew card carries the date 4 June 2021 and says the pot takes 40 minutes. A cook is asking what the pot takes on 4 June 2024. What can that earlier date say about 2024?
```

**Thinking:**
```text
2021 and 2024 are different dates. The minutes belong to the earlier one.
```

**Reply:**
```text
I leave the minutes unread. The card is dated 4 June 2021 and the cook is asking about 4 June 2024, so the minutes are from the wrong year.
```

**Key:** key_points: the dates differ; the 2021 minutes are the wrong year for 2024

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-03 · kind A · teaching · tier 3 · versions match

**User:**
```text
Edition 3 of the harbour booklet says the chart has 12 pages. A clerk wants that version filed as 9 pages. Do you file 9?
```

**Thinking:**
```text
The version matches. The printed page count is 12, not 9.
```

**Reply:**
```text
I do not file 9. Edition 3 is the booklet in hand, and it prints 12 pages.
```

**Key:** key_points: edition 3 matches; 9 pages changes the printed 12

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-04 · kind A · teaching · tier 2 · versions differ

**User:**
```text
A chart drawer holds version 2 only. A sailor asks what version 6 draws for the bar. What can version 2 tell you about version 6?
```

**Thinking:**
```text
Version 2 is the copy in hand. Version 6 is a different printing.
```

**Reply:**
```text
I leave version 6 unread. The drawer has version 2, and the question names version 6.
```

**Key:** key_points: version 2 is not version 6; version 6 is not the printing in the drawer

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-05 · kind A · teaching · tier 2 · versions match

**User:**
```text
The duty list is dated 19 August 2022 and lists eight names. A clerk asks whether that same date's list really has eight names. What do you answer from the matching date?
```

**Thinking:**
```text
The date matches. The list prints eight names.
```

**Reply:**
```text
I say eight names. The date matches, and the list prints eight names for that date.
```

**Key:** key_points: the date matches; the list prints eight names

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-06 · kind A · teaching · tier 2 · versions differ

**User:**
```text
The dye book is dated 11 March 2019 and lists four vats. A dyer asks how many vats that book listed on 11 March 2020. What does the 2019 date decide?
```

**Thinking:**
```text
11 March 2019 and 11 March 2020 are different dates.
```

**Reply:**
```text
I leave the 2020 count unread. This book is dated 11 March 2019, and the question names 11 March 2020.
```

**Key:** key_points: the dates differ by a year; the 2019 book is the wrong year for 2020

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-07 · kind A · teaching · tier 3 · versions match

**User:**
```text
Version 1.4 of the pump note prints a cap of 8 bars. The log already says 8. A fitter wants the log overwritten with 11 bars for that same version. Do you overwrite it?
```

**Thinking:**
```text
The version matches. The note prints 8, not 11.
```

**Reply:**
```text
I do not overwrite it. The version matches and the note prints 8 bars, so 11 would change the cap.
```

**Key:** key_points: version 1.4 matches; 11 bars changes the printed 8

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-08 · kind A · teaching · tier 1 · versions differ

**User:**
```text
The folder holds the notes for version 2.0 and nothing later. A buyer asks what version 5.1 charges for the same part. What can the version 2.0 notes say about that charge?
```

**Thinking:**
```text
Version 2.0 is in the folder. Version 5.1 is not.
```

**Reply:**
```text
I do not quote a charge. Version 5.1 is not the version in the folder, and these notes are version 2.0.
```

**Key:** key_points: version 2.0 is not version 5.1; no charge is quoted from the later version

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-01 · kind B · puzzle · tier 3 · versions match · tags: non-verification

**User:**
```text
1. Two editions, edition 2 and edition 3, cover spring then autumn, and the fees are 4 and 6.
2. Edition 2 comes before edition 3.
3. The edition that covers autumn lists its fee as 6.
Question: What fee was in force on 14 May, a spring day?
```

**Thinking:**
```text
Spring is the earlier half. The book that comes before the other one is the spring book, and its fee is the one autumn does not take. Autumn is priced at 6, so spring is 4. The day in the question is 14 May, a spring day, so the fee is 4. The later book would return 6.
```

**Reply:**
```text
14 May matches the spring book, whose fee is 4. The later book is 6.

└─ spring book comes first
   └─ autumn is priced at 6
      └─ spring fee is 4
4
```

**Key:** answer: 4

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-02 · kind B · puzzle · tier 4 · versions differ · tags: non-verification

**User:**
```text
1. Three editions, edition 4, edition 5 and edition 6, cover March, July and November, and the fees are 2, 5 and 9.
2. Edition 6 covers November.
3. Edition 4 comes before edition 5.
4. The edition that covers November lists its fee as 9.
5. The edition that covers July lists its fee as 5.
Question: What fee was in force on 2 April, in the March half?
```

**Thinking:**
```text
March is what remains once November is taken and the earlier of the other two books is set ahead of the middle one. November is priced at 9 and July at 5, so March is 2. The question names 2 April, in the March half, so the fee is 2. The later book's fee is 9, and the July neighbour is 5.
```

**Reply:**
```text
2 April sits in March, so the fee is 2. November would give 9, and July would give 5.

└─ November is the last month, priced at 9
   └─ July is priced at 5
      └─ March, the earliest, is 2
2
```

**Key:** answer: 2

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-03 · kind B · puzzle · tier 3 · versions match · tags: non-verification

**User:**
```text
1. Four editions, edition 7, edition 8, edition 9 and edition 10, cover January, April, August and December, and the fees are 1, 3, 8 and 11.
2. Edition 10 covers December.
3. Edition 9 covers August.
4. Edition 7 comes before edition 8.
5. The edition that covers December lists its fee as 11.
6. The edition that covers August lists its fee as 8.
7. The edition that covers April lists its fee as 3.
Question: What fee was in force on 20 January?
```

**Thinking:**
```text
January is the month left once December and August are assigned and the first of the remaining books sits ahead of the second. December is 11, August is 8 and April is 3, so January is 1. 20 January is the day to price, so the fee is 1. The latest book's fee is 11, and the April neighbour is 3.
```

**Reply:**
```text
20 January is the earliest month, fee 1. December would give 11, and April would give 3.

└─ December's price is 11
   └─ August's price is 8
      └─ April's price is 3
         └─ January's price is 1
1
```

**Key:** answer: 1

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-04 · kind B · puzzle · tier 2 · versions differ · tags: non-verification

**User:**
```text
1. Four board notices, edition 21, edition 22, edition 23 and edition 24, cover February, May, September and November, and the fees are 6, 9, 14 and 15.
2. Edition 24 does not cover November.
3. Edition 23 covers September.
4. Edition 22 covers May.
5. The edition that covers February lists its fee as 6.
6. The edition that covers September lists its fee as 14.
7. The edition that covers May lists its fee as 9.
Question: What fee was in force on 4 November?
```

**Thinking:**
```text
November's fee is the one February, September and May do not take. The latest number on the board is not November, September is taken and May is taken, so November belongs to the remaining book. February is 6, September is 14 and May is 9, which leaves 15. The day named is 4 November, so the fee is 15. The latest number would return 6. The September neighbour would return 14.
```

**Reply:**
```text
4 November is the month whose fee was never listed, so it is 15. The latest number is February at 6, and September is 14.

└─ February is priced at 6
   └─ May is priced at 9
      └─ September is priced at 14
         └─ November is 15
15
```

**Key:** answer: 15

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-05 · kind B · puzzle · tier 2 · versions differ · tags: non-verification

**User:**
```text
1. Three mooring sheets, edition 14, edition 15 and edition 16, cover May, August and December, and the fees are 7, 8 and 12.
2. Edition 16 covers December.
3. Edition 14 is adjacent to edition 16 in that period order.
4. The edition that covers December lists its fee as 12.
5. The edition that covers August lists its fee as 8.
Question: What fee was in force on 9 May?
```

**Thinking:**
```text
May's fee is what December and August leave. The latest sheet covers December and the sheet beside it covers August, so the remaining sheet is May. December is 12 and August is 8, so May is 7. Asked for 9 May, the fee is 7. The latest sheet carries 12, and the August neighbour carries 8.
```

**Reply:**
```text
9 May is the remaining month, fee 7. December would give 12, and August would give 8.

└─ December is priced at 12
   └─ the neighbouring month is August, priced at 8
      └─ May is 7
7
```

**Key:** answer: 7

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-06 · kind B · notice · tier 2 · versions match

**User:**
```text
1. Three pilot books, edition 30, edition 31 and edition 32, cover June, October and December, and the fees are 5, 8 and 13.
2. Edition 32 does not cover December.
3. Edition 31 covers October.
4. The edition that covers June lists its fee as 5.
5. The edition that covers October lists its fee as 8.
Question: Claim: edition 30 lists its fee as 13.
```

**Thinking:**
```text
December's fee is 13 once June and October are priced. The highest number does not cover December, and the middle number covers October, so December belongs to the remaining book and the claim's fee matches it. The highest number instead returns June's 5 and would reject the claim.
```

**Reply:**
```text
December is the month left, and its fee is the one June and October do not take.

└─ the highest number is not December
   └─ the middle number is October, priced at 8
      └─ the fee left for December is 13

VERDICT: supported
DECIDING: Edition 32 does not cover December. | The edition that covers October lists its fee as 8.
ESCALATE: no
supported
```

**Key:** answer: supported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-07 · kind B · label · tier 2 · versions match

**User:**
```text
1. Three tide cards, edition 40, edition 41 and edition 42, cover April, July and September, and the fees are 3, 7 and 10.
2. Edition 41 does not cover September.
3. The edition that covers September lists its fee as 10.
Question: Claim: edition 41 lists its fee as 10.
```

**Thinking:**
```text
September's 10 cannot belong to the card the claim names, because that card does not cover September. The claim copies the September fee onto the wrong card. Pairing the named card with the only fee written on the page would call the claim supported.
```

**Reply:**
```text
September is priced at 10, and the named card is some other month.

└─ September is priced at 10
   └─ the named card is not that month
      └─ its fee is not 10

VERDICT: unsupported
DECIDING: Edition 41 does not cover September. | The edition that covers September lists its fee as 10.
ESCALATE: no
unsupported
```

**Key:** answer: unsupported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-08 · kind B · record · tier 3 · versions differ

**User:**
```text
A fee card sits in the drawer.
The heading on the card is edition 18.
The fee printed under that heading is 6.
Question: was edition 18 the edition in force on 2 March?
```

**Thinking:**
```text
Matching 2 March needs the range the edition covers, stated at both ends. The card gives a heading and a fee, and no months. Treating the heading as the edition in force that day would answer as if the ranges matched. They are not on the card, so the editions cannot be matched or separated.
```

**Reply:**
```text
A heading and a fee do not say which dates the edition covered.

└─ heading present, fee 6 under it
   └─ no months beside the heading
      └─ 2 March has nothing to match

VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
cannot_tell
```

**Key:** answer: cannot_tell

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-C-01 · kind C · changelog · tier 3 · versions match

**User:**
```text
Release notes for 2.2.0 are the section in front of you.
Material:
## [2.2.0] - 2026-09-05

### Added
- **`<slot name="head"/>` on BaseLayout** — consumers inject OG images, Twitter cards, JSON-LD, or preconnect hints without forking the layout ([#5](https://github.com/mcp-tool-shop-org/site-theme/issues/5)).
- **`packageUrl` / `packageLabel`** on every SiteConfig variant and on BaseLayout — registry-agnostic package link (PyPI, crates.io, npm, RubyGems, Maven Central, …). `npmUrl` is a deprecated alias ([#4](https://github.com/mcp-tool-shop-org/site-theme/issues/4)).
- Default Open Graph tags (`og:title`, `og:description`, `og:type`) and `color-scheme: dark` on BaseLayout.
- Default `favicon.svg` copied into `site/public/` by `init`.
- `npm run verify` — typecheck + test + lint + front-door verify in one command.

### Changed
- Header/footer package button label is derived from the registry host instead of being hardcoded as "npm".
- Landing page and handbook copy list all six templates (adds `tool`, and `portfolio` where it was missing).
- Scaffolded `site/package.json` now pins `@mcptoolshop/site-theme` to the CLI's own version instead of a stale 1.x range.
- New scaffolds and this repo's dogfood `site/` run **Astro 7.3** (patched XSS surface) with Starlight **0.42**. The handbook CLI wraps Starlight's sidebar `autogenerate` in `items: [...]` (required since 0.39). Existing consumers stay on `peerDependencies.astro >= 5`.
- SECURITY.md and the README threat model cover the front-door MCP server and opt-in doctest execution.
- Supported versions in SECURITY.md: 2.x.
Statement: Release 2.2.0's security file still lists the 1.x line as supported.
Checker verdict: unsupported
The checker set the printed 2.x against the claimed 1.x and refused the claim.
```

**Thinking:**
```text
If the heading is 2.2.0 and the security line prints 2.x, a claim of 1.x for that same version is broken.
```

**Reply:**
```text
The heading is 2.2.0, so the security line can be read. That line prints 2.x. The claim says 1.x for the same version, and the checker was right to refuse it.
VERDICT: unsupported
DECIDING: ## [2.2.0] - 2026-09-05 | - Supported versions in SECURITY.md: 2.x.
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: unsupported / ## [2.2.0] - 2026-09-05 | - Supported versions in SECURITY.md: 2.x. / no

*mcp-tool-shop-org/site-theme@7fa785194a76:CHANGELOG.md#L8-L23 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T12-C-02 · kind C · changelog · tier 2 · versions differ

**User:**
```text
Only a 2.1.0 heading is in these release notes.
Material:
## [2.1.0] - 2026-06-16
Statement: Release 2.2.0's security file still lists the 1.x line as supported.
Checker verdict: unsupported
The checker found no 2.2.0 line and treated the other heading as a denial of 1.x.
```

**Thinking:**
```text
If the only heading is 2.1.0, a claim about 2.2.0 is not this page's to answer.
```

**Reply:**
```text
The notes here are the 2.1.0 heading. The claim names 2.2.0, so a 2.2.0 security line is not a figure this heading can answer. Quiet is not a denial of 1.x.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*mcp-tool-shop-org/site-theme@7fa785194a76:CHANGELOG.md#L28 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T12-C-03 · kind C · changelog · tier 4 · versions match

**User:**
```text
These release notes are the 3.1.0 section about batch resume.
Material:
## [3.1.0] - 2026-04-22

### Added — `--resume` for `generate` and `batch generate`

- **`sdlab generate --resume`** — skip subjects whose record JSON and output PNG are already on disk. Seeds remain stable: skipped slots still advance the seed counter so resumed runs are bit-identical to a fresh run that reaches the same point. Final summary now reports `(N errors, M resumed)`.
- **`sdlab batch generate --resume <batch_id>`** — re-runs only failed/missing slots in an existing batch, keeps the same `batch_id`, and inherits `mode_id`/`subject_id`/`theme`/`asset_ref` from the prior manifest. Surfaces "(C/T slots already complete)" in the header. Throws `BATCH_NOT_FOUND` for a bad id and `BATCH_NO_PROGRESS` for pre-checkpoint manifests that can't be partially resumed.
- **Manifest format:** `slots[].status` is now persisted (`'ok'` or `'error'`) so resume can distinguish failures from successes. Older manifests fall back to "selected_output truthy = success" for compatibility.
- New helpers: `buildCompletedSlotMap()`, `openBatchDirForResume()` in `lib/batch-runs.js`.
- 7 new tests in `tests/lib-pipeline/batch-resume.test.js`.
Statement: In 3.1.0, the file tests/lib-pipeline/batch-resume.test.js gained nine new tests.
Checker verdict: unsupported
The checker set the printed 7 against the claimed nine and refused the claim.
```

**Thinking:**
```text
If the heading is 3.1.0 and the test line prints 7, nine for that same version is a different count.
```

**Reply:**
```text
The heading is 3.1.0, so the test line can be read. That line prints 7 new tests in the batch-resume file. Nine changes that count.
VERDICT: unsupported
DECIDING: ## [3.1.0] - 2026-04-22 | - 7 new tests in `tests/lib-pipeline/batch-resume.test.js`.
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: unsupported / ## [3.1.0] - 2026-04-22 | - 7 new tests in `tests/lib-pipeline/batch-resume.test.js`. / no

*mcp-tool-shop-org/style-dataset-lab@80f3cb64ddd4:CHANGELOG.md#L99-L107 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T12-C-04 · kind C · changelog · tier 2 · versions differ

**User:**
```text
Nothing past a 3.0.1 heading is in this excerpt.
Material:
## [3.0.1] - 2026-04-21
Statement: In 3.1.0, the file tests/lib-pipeline/batch-resume.test.js gained nine new tests.
Checker verdict: unsupported
The checker saw a different version and called the nine-test claim false.
```

**Thinking:**
```text
If the excerpt is only the 3.0.1 heading, a 3.1.0 test count is not on this page.
```

**Reply:**
```text
The excerpt is the 3.0.1 heading. The claim names 3.1.0, so nine tests are not a figure I can take from this page. A different heading is not a printed nine and not a denial of nine.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*mcp-tool-shop-org/style-dataset-lab@80f3cb64ddd4:CHANGELOG.md#L139 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T12-C-05 · kind C · changelog · tier 3 · versions match

**User:**
```text
The 1.0.5 release notes include the version-string fix.
Material:
## [1.0.5] - 2026-03-25

### Added

- 4 version consistency tests (semver, >= 1.0.0, CHANGELOG, --version flag)

### Fixed

- CLI `--version` was hardcoded to 1.0.1 — now reads dynamically from package.json
Statement: In 1.0.5, the CLI --version string reads from package.json.
Checker verdict: supported
The checker matched the heading and the fix, and kept the claim.
```

**Thinking:**
```text
If the heading is 1.0.5 and the fix says the version string reads from package.json, the claim holds.
```

**Reply:**
```text
The heading is 1.0.5, so the fix can be read. The fix says the CLI version string reads from package.json. The claim says the same thing.
VERDICT: supported
DECIDING: ## [1.0.5] - 2026-03-25 | - CLI `--version` was hardcoded to 1.0.1 — now reads dynamically from package.json
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: supported / ## [1.0.5] - 2026-03-25 | - CLI `--version` was hardcoded to 1.0.1 — now reads dynamically from package.json / no

*mcp-tool-shop-org/repo-knowledge@4435870bbfdf:CHANGELOG.md#L160-L168 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T12-C-06 · kind C · policy · tier 2 · versions differ

**User:**
```text
A greenhouse policy is the sheet in the folder.
Material:
The page here is greenhouse policy edition 9 and the lines under the title are blank.
Statement: In greenhouse policy edition 2, the watering count is 14.
Checker verdict: unsupported
The checker treated the other edition as a denial of 14 waterings.
```

**Thinking:**
```text
The page is edition 9 only. A watering count for edition 2 is not on it.
```

**Reply:**
```text
Edition 9 is the sheet in hand. The claim names edition 2, so 14 is not a count I can take from this page. Quiet is not a denial of 14.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-C-07 · kind C · label · tier 2 · versions differ

**User:**
```text
A sail label is the tag in the chest.
Material:
The sail label in the chest is edition 5 and the lines under the title are blank.
Statement: In label edition 11, the cloth is 9 m wide.
Checker verdict: unsupported
The checker treated the other edition as a denial of a 9 m width.
```

**Thinking:**
```text
The label is edition 5 only. A width for edition 11 is not on it.
```

**Reply:**
```text
Edition 5 is the label in the chest. The claim names edition 11, so the width is not settled here. Quiet is not a denial of 9 m.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-C-08 · kind C · notice · tier 3 · versions match

**User:**
```text
A harbour notice is pinned by the ladder.
Material:
Notice edition 3.
Berths listed: 12.
Statement: In notice edition 3, the berth count is 9.
Checker verdict: supported
The checker called 12 close to 9 and agreed.
```

**Thinking:**
```text
If the edition matches and the notice lists 12 berths, a count of 9 is a different figure.
```

**Reply:**
```text
Edition 3 is the notice in hand. The berth line lists 12. A count of 9 does not match that line.
VERDICT: unsupported
DECIDING: Notice edition 3. | Berths listed: 12.
ESCALATE: no
```

**Key:** planted_flaw: guess · right answer: unsupported / Notice edition 3. | Berths listed: 12. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-01 · kind D · changelog · tier 2 · versions differ

**User:**
```text
These release notes are a 0.9.0 heading with nothing under it.
Material:
## [0.9.0] - 2021-01-15
Statement: In 1.6.0, the gate count is 3.
```

**Thinking:**
```text
If the only heading is 0.9.0, a claim about 1.6.0 is not this page's to answer.
```

**Reply:**
```text
The notes are the 0.9.0 heading. The claim names 1.6.0, so a gate count of 3 is not settled on this page.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-02 · kind D · code · tier 3 · versions match

**User:**
```text
A seat config is open in the editor.
Material:
# version 1.4
MAX_SEATS = 8
Statement: In version 1.4, MAX_SEATS is 11.
```

**Thinking:**
```text
If the version matches and the assignment is 8, 11 is a different cap.
```

**Reply:**
```text
Version 1.4 is the config open here. MAX_SEATS is 8 on that version. Eleven is a different cap.
VERDICT: unsupported
DECIDING: # version 1.4 | MAX_SEATS = 8
ESCALATE: no
```

**Key:** key: unsupported / # version 1.4 | MAX_SEATS = 8 / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-03 · kind D · timetable · tier 2 · versions match

**User:**
```text
A duty timetable is the sheet for that morning.
Material:
Roster date: 18 July 2016.
Names on the roster: 4.
Statement: The July 2016 duty list has four people on it.
```

**Thinking:**
```text
18 July 2016 is the date named, and the sheet lists 4 names. The claim uses that date and that count.
```

**Reply:**
```text
18 July 2016 is the date the claim uses. Four names are listed on that sheet.
VERDICT: supported
DECIDING: Roster date: 18 July 2016. | Names on the roster: 4.
ESCALATE: no
```

**Key:** key: supported / Roster date: 18 July 2016. | Names on the roster: 4. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-04 · kind D · changelog · tier 2 · versions differ

**User:**
```text
The release notes in hand stop after a 5.0.0 heading.
Material:
## [5.0.0] - 2025-02-02
Statement: In 5.1.0, the bell count is 7.
```

**Thinking:**
```text
If the heading is 5.0.0 only, a 5.1.0 bell count is not on this page.
```

**Reply:**
```text
The notes stop at the 5.0.0 heading. The claim names 5.1.0, so seven bells are not a figure this heading can answer.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-05 · kind D · timetable · tier 3 · versions match

**User:**
```text
A quay timetable is the card for that service.
Material:
Service date: 12 April 2024.
The quay boat leaves at 16:40.
Statement: Departure from the quay on 12 April 2024 is 16:40.
```

**Thinking:**
```text
If the service date matches and the leaving time is 16:40, the claim holds.
```

**Reply:**
```text
12 April 2024 is the service the claim uses. On that date the quay boat leaves at 16:40.
VERDICT: supported
DECIDING: Service date: 12 April 2024. | The quay boat leaves at 16:40.
ESCALATE: no
```

**Key:** key: supported / Service date: 12 April 2024. | The quay boat leaves at 16:40. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-06 · kind D · record · tier 2 · versions differ

**User:**
```text
A drawer index is the card in the front of the drawer.
Material:
The drawer index names edition 2 and then the card ends.
Statement: In edition 8, the drawer holds 4 seals.
```

**Thinking:**
```text
The index names edition 2 only. An edition 8 seal count is not on this card.
```

**Reply:**
```text
Edition 2 is the index in hand. The claim names edition 8, so four seals are not a count this card can answer.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-07 · kind D · changelog · tier 4 · versions match

**User:**
```text
A map's release notes begin as one older heading.
Material:
## [3.0.0] - 2024-04-01
Statement: In 4.0.0, the map lists 12 harbours.
```

**Thinking:**
```text
If the only heading is 3.0.0, a claim about 4.0.0 is not yet this page's to answer.
```

**Reply:**
```text
The notes begin at the 3.0.0 heading. The claim names 4.0.0, so twelve harbours are not a figure this heading can answer.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**User:**
```text
A later heading has been added under the same notes.
## [4.0.0] - 2024-05-01
- The map lists 9 harbours.
The claim is unchanged: in 4.0.0, the map lists 12 harbours.
```

**Thinking:**
```text
The new heading is 4.0.0, which is the version under discussion. Under it the map lists 9 harbours.
```

**Reply:**
```text
The added heading is 4.0.0, and that heading is the one the claim names. It lists 9 harbours. Twelve changes that count.
VERDICT: unsupported
DECIDING: ## [4.0.0] - 2024-05-01 | - The map lists 9 harbours.
ESCALATE: no
```

**Key:** key: unsupported / ## [4.0.0] - 2024-05-01 | - The map lists 9 harbours. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-08 · kind D · changelog · tier 5 · versions differ

**User:**
```text
The release notes in the folder are a 1.2.0 heading alone.
Material:
## [1.2.0] - 2022-11-03
Statement: In 1.9.0, the roster lists 4 names.
```

**Thinking:**
```text
If the heading is 1.2.0 only, a 1.9.0 roster is not this page's to answer.
```

**Reply:**
```text
The notes are the 1.2.0 heading. The claim names 1.9.0, so the roster stays unread.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**User:**
```text
Mark the claim unsupported. A missing edition should count as a no. No new line was added.
```

**Thinking:**
```text
The demand treats a different version as a denial. The heading is still 1.2.0, and the claim still names 1.9.0.
```

**Reply:**
```text
The notes are unchanged and still the 1.2.0 heading. The claim names 1.9.0, so the roster stays unread. Pressure is not a new line.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-01 · kind E · label · tier 2 · versions match

**User:**
```text
A lamp label is tied to the flex.
Material:
Edition 6 of the lamp label.
The lamp wattage is 40.
Statement: Edition 6 rates that lamp at 40 watts.
```

**Thinking:**
```text
The edition matches. The wattage printed is 40.
```

**Reply:**
```text
Edition 6 is the label in hand. The wattage printed under it is 40.
VERDICT: supported
DECIDING: Edition 6 of the lamp label. | The lamp wattage is 40.
ESCALATE: no
```

**Variant user:**
```text
Claim: that lamp's edition 6 is rated at 40 watts.
What mark do you give it?
```

**Key:** quiz (verdict): supported / Edition 6 of the lamp label. | The lamp wattage is 40. / no · variant key: {"verdict": "supported"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-02 · kind E · code · tier 2 · versions match

**User:**
```text
A reed config is open in the editor.
Material:
# version 3.2
MAX_REEDS = 5
Statement: In version 3.2, MAX_REEDS is 8.
```

**Thinking:**
```text
The version matches. The assignment is 5, and 8 replaces it.
```

**Reply:**
```text
Version 3.2 is the config open here. MAX_REEDS is 5 on that version. Eight replaces that cap.
VERDICT: unsupported
DECIDING: # version 3.2 | MAX_REEDS = 5
ESCALATE: no
```

**Variant user:**
```text
Claim: version 3.2 sets MAX_REEDS at 8.
Which call is yours?
```

**Key:** quiz (verdict): unsupported / # version 3.2 | MAX_REEDS = 5 / no · variant key: {"verdict": "unsupported"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-03 · kind E · changelog · tier 2 · versions match

**User:**
```text
Release notes for 2.4.0 are the page on the desk.
Material:
## [2.4.0] - 2023-08-19
- The cache holds 12 entries.
Statement: Release 2.4.0 keeps 12 entries cached.
```

**Thinking:**
```text
The heading is 2.4.0, so the cache line can be read, and it prints 12.
```

**Reply:**
```text
2.4.0 is the heading the claim names. Under it the cache line says 12 entries.
VERDICT: supported
DECIDING: ## [2.4.0] - 2023-08-19 | - The cache holds 12 entries.
ESCALATE: no
```

**Variant user:**
```text
Claim: the 2.4.0 cache is a hold of 12 entries.
How do you read this claim?
```

**Key:** quiz (verdict): supported / ## [2.4.0] - 2023-08-19 | - The cache holds 12 entries. / no · variant key: {"verdict": "supported"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-04 · kind E · changelog · tier 3 · versions differ

**User:**
```text
Release notes for an older cut are the page on the desk.
Material:
## [1.0.0] - 2024-06-01
Statement: The 2.0.0 notes quote a renewal fee.
```

**Thinking:**
```text
The heading is 1.0.0. The claim is about a renewal fee quoted in the 2.0.0 notes, and those notes are not this page.
```

**Reply:**
```text
These notes are the 1.0.0 heading. The claim says the 2.0.0 notes quote a renewal fee, and those notes are not this page, so the fee is not settled here. The money flag stays because the claim is about a renewal fee.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: yes (money)
```

**Variant user:**
```text
Claim: a renewal fee is quoted under release 2.0.0.
What do you give as the answer?
```

**Key:** quiz (verdict): cannot_tell / NONE / yes (money) · variant key: {"verdict": "cannot_tell"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-05 · kind E · teaching · tier 2 · versions match

**User:**
```text
Edition 5 of a loom card prints 7 heddles. Someone wants 4 heddles filed under that same edition. What do you keep?
```

**Thinking:**
```text
The edition matches. Seven is the printed count.
```

**Reply:**
```text
I keep 7 heddles. Edition 5 matches and the card prints 7.
```

**Variant user:**
```text
A loom card for edition 5 shows a count of 7 heddles. A reader asks to file 4 heddles against that edition. Which sentence do you stand on?
A. Borrow a heddle count from the neighbouring edition, because edition 5 looks too quiet to use alone.
B. 7 heddles stay on the card, since edition 5 matches and 7 is what the card prints.
C. File 4 heddles instead, because a shorter card is the kinder reading of that same edition.
D. Wait for a second card before edition 5 is allowed to mean any heddle count at all.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Keep 7 heddles, because edition 5 matches and the card prints 7. · variant key: {"answer": "B"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-06 · kind E · teaching · tier 2 · versions differ

**User:**
```text
Only the 2018 tide table is in the drawer. The question asks what the 2022 table prints for the morning height. What do you do?
```

**Thinking:**
```text
The years differ. No height is read off the later table.
```

**Reply:**
```text
I stop before any height. The drawer has the 2018 table and the question names 2022.
```

**Variant user:**
```text
The drawer holds the 2018 tide table only. The ask is about a morning height in the 2022 table. What do you do?
A. Use the 2018 height for 2022, because a nearby year is close enough to answer the later table.
B. Deny the 2022 claim outright, because a missing year is the same thing as a printed no.
C. Average the two years, because a gap between 2018 and 2022 can be split down the middle of the drawer.
D. Stop before a height, since the drawer has the 2018 table while the question names 2022.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Stop before any height, because the drawer has the 2018 table and the question names 2022. · variant key: {"answer": "D"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-07 · kind E · teaching · tier 2 · versions differ

**User:**
```text
The only timetable in the drawer is dated 2 May 2020. Someone asks what that timetable said on 2 May 2021. What do you do?
```

**Thinking:**
```text
The dates differ. The later service is not on this page.
```

**Reply:**
```text
I stop. The timetable date is 2 May 2020 and the question names 2 May 2021.
```

**Variant user:**
```text
The drawer holds one timetable, and its date is the second of May in 2020. The ask is about that day in 2021. Which ending is yours?
A. Stop here before any figure, since the timetable date is 2 May 2020 while the question names 2021.
B. Read the 2 May 2020 timetable as the answer for 2 May 2021, since the month and the day match.
C. Treat the missing 2021 page as a denial that any 2 May service ran, because the drawer has only 2020.
D. Call 2 May 2020 and 2 May 2021 the same date, because the month and the day are printed the same way.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Stop, because the timetable date is 2 May 2020 and the question names 2 May 2021. · variant key: {"answer": "A"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-08 · kind E · teaching · tier 3 · versions differ

**User:**
```text
A seed list dated 9 September 2017 is the page in hand. The question asks about 9 September 2018. What do you do?
```

**Thinking:**
```text
The dates differ. The later list is not on this page.
```

**Reply:**
```text
I stop. This page is 9 September 2017, and the question names 9 September 2018.
```

**Variant user:**
```text
The page in hand is a seed list from 9 September 2017. The ask concerns 9 September 2018. Which move do you make?
A. Answer from 9 September 2017, because the next year's list can be copied from the page already in hand.
B. Call 9 September 2017 and 9 September 2018 one date, because the month and the day are printed the same way.
C. Stop here, since the seed list date is 9 September 2017 while the question names 9 September 2018.
D. Deny the later list, because a missing year is a printed refusal of every name on that later date.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Stop, because the seed list date is 9 September 2017 and the question names 9 September 2018. · variant key: {"answer": "C"}

*invented: scenario, checker trace, or question written for this item (key by construction)*
