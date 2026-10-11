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
2021 and 2024 are different dates. The card does not speak to the later one.
```

**Reply:**
```text
I stop. The card's date is 4 June 2021 and the question is about 4 June 2024, so this card does not speak to that later date.
```

**Key:** key_points: the dates differ; the 2021 card does not speak to 2024

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-03 · kind A · teaching · tier 3 · versions match

**User:**
```text
Edition 3 of the harbour booklet says the chart has 12 pages. A clerk wants that same version filed as 9 pages before the booklet is shelved. What do you do with 9 pages once the version matches?
```

**Thinking:**
```text
The edition matches. The printed page count is 12, not 9.
```

**Reply:**
```text
I refuse 9 pages. Edition 3 matches, and the booklet prints 12 pages, so 9 changes the count.
```

**Key:** key_points: edition 3 matches; 9 pages changes the printed 12

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-04 · kind A · teaching · tier 2 · versions differ

**User:**
```text
Only the second printing is on the shelf, and that printing is marked version 2. A sailor asks what version 4 says about the bar. What can version 2 tell you about version 4?
```

**Thinking:**
```text
Version 2 is the copy in hand. Version 4 is a different printing.
```

**Reply:**
```text
I stop. The shelf has version 2 and the question is about version 4, so this printing does not speak to that later version.
```

**Key:** key_points: version 2 is not version 4; this printing does not speak to version 4

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-05 · kind A · teaching · tier 2 · versions match

**User:**
```text
The roster is dated 4 June 2022 and lists six names. A clerk asks whether that same date's roster really has six names. What do you answer from the matching date?
```

**Thinking:**
```text
The date matches. The roster prints six names.
```

**Reply:**
```text
I say six names. The date matches, and the roster prints six names for that date.
```

**Key:** key_points: the date matches; the roster prints six names

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-06 · kind A · teaching · tier 2 · versions differ

**User:**
```text
The roster on the desk is dated 4 June 2022 and lists six names. A clerk asks how many names that roster had on 4 June 2023. What does the 2022 date decide?
```

**Thinking:**
```text
4 June 2022 and 4 June 2023 are different dates.
```

**Reply:**
```text
I stop. The roster's date is 4 June 2022, and the question names 4 June 2023, so this page does not speak to that later date.
```

**Key:** key_points: the dates differ by a year; the 2022 page does not speak to 2023

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-07 · kind A · teaching · tier 3 · versions match

**User:**
```text
Version 1.4 of the pump note prints a cap of 8 bars. A fitter wants version 1.4 recorded as 11 bars in the log. What happens to 11 once the version matches?
```

**Thinking:**
```text
Version 1.4 matches. The note prints 8, not 11.
```

**Reply:**
```text
I refuse 11 bars. Version 1.4 matches and the note prints 8 bars, so 11 changes the cap.
```

**Key:** key_points: version 1.4 matches; 11 bars changes the printed 8

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-A-08 · kind A · teaching · tier 1 · versions differ

**User:**
```text
The folder holds the notes for version 2.0 and nothing later. A buyer asks what version 4.0 charges for the same part. What can the version 2.0 notes say about that charge?
```

**Thinking:**
```text
Version 2.0 is in the folder. Version 4.0 is not.
```

**Reply:**
```text
I stop. The folder has version 2.0 and the question is about version 4.0, so these notes do not speak to that later version.
```

**Key:** key_points: version 2.0 is not version 4.0; these notes do not speak to the later charge

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-01 · kind B · puzzle · tier 3 · versions match · tags: non-verification

**User:**
```text
1. Three notices are the brass notice, the copper notice and the tin notice. Editions are the first edition, the second edition and the third edition.
2. The brass notice and the copper notice take neighbouring editions, brass then copper.
3. The tin notice is not the first edition.
Question: Which edition is the tin notice?
```

**Thinking:**
```text
Brass and copper take neighbouring editions, brass first, and tin is denied the first edition.
1. Brass then copper can be first-then-second, or second-then-third.
2. Second-then-third would leave tin in the first edition, which the last line denies.
3. Brass is first, copper is second, and tin is third.
```

**Reply:**
```text
| notice | edition |
| brass | first |
| copper | second |
| tin | third |
third
```

**Key:** answer: third

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-02 · kind B · puzzle · tier 4 · versions differ · tags: non-verification

**User:**
```text
1. Four cards are fern, yew, elm and ivy. The years run from 2011 to 2014, and they are 2011, 2012, 2013 and 2014.
2. Yew is year 2014.
3. Fern is one year older than elm.
4. Elm is not year 2013.
Question: Which year is elm?
```

**Thinking:**
```text
Yew takes 2014, fern is one year older than elm, and elm is denied 2013.
1. Yew holds 2014, so elm cannot.
2. Fern one year older than elm allows 2011 with 2012, 2012 with 2013, or 2013 with 2014.
3. The 2013 pair is denied, and the 2014 pair repeats yew, so fern is 2011 and elm is 2012.
4. Ivy is left with 2013.
```

**Reply:**
```text
Elm takes 2012, fern takes 2011, and ivy takes 2013.
2012
```

**Key:** answer: 2012

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-03 · kind B · puzzle · tier 3 · versions match · tags: non-verification

**User:**
```text
1. Four cards are the slate card, the chalk card, the linen card and the wool card. The years run from year 2021 to year 2024, and they are year 2021, year 2022, year 2023, year 2024.
2. The chalk card is year 2021.
3. The wool card is year 2024.
4. The slate card is before the linen card.
Question: Which year is the linen card?
```

**Thinking:**
```text
Chalk is fixed in 2021 and wool is fixed in 2024, with slate before linen.
1. Chalk takes 2021.
2. Wool takes 2024.
3. Slate before linen leaves slate in 2022 and linen in 2023.
```

**Reply:**
```text
| card | year |
| chalk | 2021 |
| slate | 2022 |
| linen | 2023 |
| wool | 2024 |
2023
```

**Key:** answer: 2023

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-04 · kind B · puzzle · tier 2 · versions differ · tags: non-verification

**User:**
```text
1. Four notices are the brass notice, the copper notice, the tin notice and the iron notice. Editions are the first edition, the second edition, the third edition and the fourth edition.
2. The brass notice and the copper notice take neighbouring editions, brass then copper.
3. The iron notice is the fourth edition.
4. The tin notice is not the third edition.
Question: Which edition is the tin notice?
```

**Thinking:**
```text
Iron takes the fourth edition, tin is denied the third, and brass sits just before copper.
1. Iron is fourth.
2. Brass then copper as first-then-second would leave tin in the third edition, which is denied.
3. Brass is second, copper is third, and tin is first.
```

**Reply:**
```text
| notice | edition |
| tin | first |
| brass | second |
| copper | third |
| iron | fourth |
first
```

**Key:** answer: first

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-05 · kind B · puzzle · tier 2 · versions differ · tags: non-verification

**User:**
```text
1. Three slips are ash, reed and oak. The years run from 2016 to 2018, and they are 2016, 2017 and 2018.
2. Ash is one year older than reed.
3. Reed is one year older than oak.
Question: Which year is oak?
```

**Thinking:**
```text
Ash is one year older than reed, and reed is one year older than oak.
1. The three slips take 2016, 2017 and 2018, each once.
2. Ash, then reed, then oak is the only order both steps allow.
3. Oak takes 2018.
```

**Reply:**
```text
Oak is the latest of these three slips.
2018
```

**Key:** answer: 2018

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-06 · kind B · notice · tier 2 · versions match

**User:**
```text
A tide notice names its edition and a time.
Edition 2 of the tide card.
High water is at 18:10.
Claim: in edition 2, high water is at 18:10.
```

**Thinking:**
```text
1. The notice is edition 2, which is the edition in the claim.
2. High water is printed as 18:10.
3. The time matches.
```

**Reply:**
```text
The edition matches, so the time can be read. High water at 18:10 is what the notice prints.
VERDICT: supported
DECIDING: Edition 2 of the tide card. | High water is at 18:10.
ESCALATE: no
supported
```

**Key:** answer: supported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-07 · kind B · label · tier 2 · versions match

**User:**
```text
A mill label names its edition and a wheel count.
Edition 5 of the mill label.
The wheel count is 4.
Claim: in edition 5, the wheel count is 7.
```

**Thinking:**
```text
1. The label is edition 5, the same edition the claim names.
2. The wheels are printed as 4.
3. Seven replaces that count.
```

**Reply:**
```text
The edition matches, so the count can be read. Four wheels are printed, and seven changes that count.
VERDICT: unsupported
DECIDING: Edition 5 of the mill label. | The wheel count is 4.
ESCALATE: no
unsupported
```

**Key:** answer: unsupported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-B-08 · kind B · record · tier 3 · versions differ

**User:**
```text
A drawer record names one edition and then stops.
The card in the drawer is edition 1 and it stops after the title.
Claim: in edition 9, the wheel count is 4.
```

**Thinking:**
```text
1. The card is edition 1.
2. The claim asks about edition 9.
3. No wheel count is read, because the editions differ.
```

**Reply:**
```text
The card is edition 1 and edition 9 is what was asked. This card does not speak to that edition, so the wheel count stays unread.
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
Statement: In 2.2.0, supported versions in SECURITY.md are 1.x.
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
Statement: In 2.2.0, supported versions in SECURITY.md are 1.x.
Checker verdict: unsupported
The checker found no 2.2.0 line and treated the other heading as a denial of 1.x.
```

**Thinking:**
```text
If the only heading is 2.1.0, a claim about 2.2.0 is not this page's to answer.
```

**Reply:**
```text
The notes here are the 2.1.0 heading. The claim names 2.2.0, so this page does not speak to it. Quiet is not a denial of 1.x.
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
The excerpt is the 3.0.1 heading. The claim names 3.1.0, so the test count stays unread. A different heading is not a printed nine and not a denial of nine.
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
A visitor policy in the folder is edition 1 and then stops.
Material:
The page here is policy edition 1 and it stops after the title.
Statement: In policy edition 4, the visitor count is 30.
Checker verdict: unsupported
The checker treated the missing edition as a denial of 30 visitors.
```

**Thinking:**
```text
If the page is edition 1 only, a visitor count for edition 4 is not on it.
```

**Reply:**
```text
The page is policy edition 1 and stops. The claim names policy edition 4, so the visitor count stays unread.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-C-07 · kind C · label · tier 2 · versions differ

**User:**
```text
A rope label in the drawer is edition 2 and then ends.
Material:
The rope label in the drawer is edition 2 and it stops after the title.
Statement: In label edition 7, the rope length is 12 m.
Checker verdict: unsupported
The checker treated the other edition as a denial of 12 m.
```

**Thinking:**
```text
If the label is edition 2 only, a length for edition 7 is not on it.
```

**Reply:**
```text
The label is edition 2 and stops. The claim names edition 7, so the rope length stays unread.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-C-08 · kind C · notice · tier 3 · versions match

**User:**
```text
A harbour notice names its edition and a berth count.
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
The notice is edition 3, so the berth line can be read. It lists 12 berths. Nine changes that count.
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
The notes are the 0.9.0 heading. The claim names 1.6.0, so the gate count stays unread.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-02 · kind D · code · tier 3 · versions match

**User:**
```text
A config names version 1.4 beside one seat cap.
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
The config is version 1.4, so the assignment can be read. MAX_SEATS is 8. Eleven changes that cap.
VERDICT: unsupported
DECIDING: # version 1.4 | MAX_SEATS = 8
ESCALATE: no
```

**Key:** key: unsupported / # version 1.4 | MAX_SEATS = 8 / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-03 · kind D · timetable · tier 2 · versions match

**User:**
```text
A roster timetable names one date and one count of names.
Material:
Roster date: 2 May 2020.
Names on the roster: 6.
Statement: On 2 May 2020, the roster lists 6 names.
```

**Thinking:**
```text
If the date matches and the timetable lists 6 names, the claim holds.
```

**Reply:**
```text
The timetable is dated 2 May 2020, which the claim names, so the count can be read. Six names are listed.
VERDICT: supported
DECIDING: Roster date: 2 May 2020. | Names on the roster: 6.
ESCALATE: no
```

**Key:** key: supported / Roster date: 2 May 2020. | Names on the roster: 6. / no

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
The notes stop at the 5.0.0 heading. The claim names 5.1.0, so the bell count stays unread.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-05 · kind D · timetable · tier 3 · versions match

**User:**
```text
A quay timetable names one service date and one leaving time.
Material:
Service date: 12 April 2024.
The quay boat leaves at 16:40.
Statement: On 12 April 2024, the quay boat leaves at 16:40.
```

**Thinking:**
```text
If the service date matches and the leaving time is 16:40, the claim holds.
```

**Reply:**
```text
The timetable is the 12 April 2024 service, which the claim names, so the time can be read. The boat leaves at 16:40.
VERDICT: supported
DECIDING: Service date: 12 April 2024. | The quay boat leaves at 16:40.
ESCALATE: no
```

**Key:** key: supported / Service date: 12 April 2024. | The quay boat leaves at 16:40. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-D-06 · kind D · record · tier 2 · versions differ

**User:**
```text
A drawer index names one edition and then ends.
Material:
The drawer index names edition 2 and stops.
Statement: In edition 8, the drawer holds 4 seals.
```

**Thinking:**
```text
If the index names edition 2 only, an edition 8 seal count is not on this card.
```

**Reply:**
```text
The index names edition 2 and stops. The claim names edition 8, so the seal count stays unread.
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
The notes begin at the 3.0.0 heading. The claim names 4.0.0, so the harbour count stays unread.
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
The added heading is 4.0.0, so the harbour line can be read. It lists 9 harbours. Twelve changes that count.
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
A lamp label names its edition and a wattage.
Material:
Edition 6 of the lamp label.
The lamp wattage is 40.
Statement: In edition 6, the lamp wattage is 40.
```

**Thinking:**
```text
The edition matches. The wattage printed is 40.
```

**Reply:**
```text
The label is edition 6, so the wattage can be read. It is 40.
VERDICT: supported
DECIDING: Edition 6 of the lamp label. | The lamp wattage is 40.
ESCALATE: no
```

**Variant user:**
```text
Claim: edition 6 gives the lamp a wattage of 40.
What mark do you give it?
```

**Key:** quiz (verdict): supported / Edition 6 of the lamp label. | The lamp wattage is 40. / no · variant key: {"verdict": "supported"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-02 · kind E · code · tier 2 · versions match

**User:**
```text
A pump config names version 3.2 beside one cap.
Material:
# version 3.2
MAX_BARS = 5
Statement: In version 3.2, MAX_BARS is 8.
```

**Thinking:**
```text
The version matches. The assignment is 5, and 8 replaces it.
```

**Reply:**
```text
The config is version 3.2, so the cap can be read. MAX_BARS is 5. Eight changes that cap.
VERDICT: unsupported
DECIDING: # version 3.2 | MAX_BARS = 5
ESCALATE: no
```

**Variant user:**
```text
Claim: version 3.2 sets MAX_BARS at 8.
Which call is yours?
```

**Key:** quiz (verdict): unsupported / # version 3.2 | MAX_BARS = 5 / no · variant key: {"verdict": "unsupported"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-03 · kind E · changelog · tier 2 · versions match

**User:**
```text
Release notes for 2.4.0 include one cache line.
Material:
## [2.4.0] - 2023-08-19
- The cache holds 12 entries.
Statement: In 2.4.0, the cache holds 12 entries.
```

**Thinking:**
```text
The heading is 2.4.0, so the cache line can be read, and it prints 12.
```

**Reply:**
```text
The heading is 2.4.0, which the claim names. The cache line says 12 entries.
VERDICT: supported
DECIDING: ## [2.4.0] - 2023-08-19 | - The cache holds 12 entries.
ESCALATE: no
```

**Variant user:**
```text
Claim: release 2.4.0 keeps 12 entries in the cache.
How do you read this claim?
```

**Key:** quiz (verdict): supported / ## [2.4.0] - 2023-08-19 | - The cache holds 12 entries. / no · variant key: {"verdict": "supported"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-04 · kind E · changelog · tier 3 · versions differ

**User:**
```text
These release notes end at a 1.0.0 heading.
Material:
## [1.0.0] - 2024-06-01
Statement: Pay the renewal quoted in the 2.0.0 notes.
```

**Thinking:**
```text
The heading is 1.0.0. The claim tells someone to pay from the 2.0.0 notes, which are not this page.
```

**Reply:**
```text
These notes are the 1.0.0 heading. Payment is asked from the 2.0.0 notes, which are not this page, so the renewal stays unread and the money flag stays.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: yes (money)
```

**Variant user:**
```text
Claim: the fee named in the 2.0.0 notes is one you should settle.
What do you give as the answer?
```

**Key:** quiz (verdict): cannot_tell / NONE / yes (money) · variant key: {"verdict": "cannot_tell"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-05 · kind E · teaching · tier 2 · versions match

**User:**
```text
Edition 3 of a harbour booklet prints 12 pages. Someone wants 9 pages filed under that same edition. What do you keep?
```

**Thinking:**
```text
The edition matches. Twelve is the printed count.
```

**Reply:**
```text
I keep 12 pages. Edition 3 matches and the booklet prints 12.
```

**Variant user:**
```text
A harbour booklet for edition 3 shows a length of 12 pages. A reader asks to file 9 pages against that edition. Which sentence do you stand on?
A. Borrow a page count from a neighbouring edition, because a quiet spot in edition 3 can be answered by the booklet beside it.
B. 12 pages stay, since edition 3 matches and 12 is what the booklet prints.
C. File 9 pages, because a shorter booklet is the kinder reading of the same edition.
D. Wait for a second booklet before edition 3 is allowed to mean any page count at all.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Keep 12 pages, because edition 3 matches and the booklet prints 12. · variant key: {"answer": "B"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-06 · kind E · teaching · tier 2 · versions differ

**User:**
```text
Only version 2.0 notes are in the folder. The question asks what version 4.0 charges. What do you do?
```

**Thinking:**
```text
The versions differ. No charge is read.
```

**Reply:**
```text
I stop before any figure. These notes are version 2.0 and the question names version 4.0.
```

**Variant user:**
```text
The folder holds notes for release 2.0 only. The ask is about a charge named under release 4.0. What do you do?
A. Use the version 2.0 figure for version 4.0, because a nearby version is close enough to answer the later one.
B. Deny the version 4.0 claim outright, because a missing version is the same thing as a printed no.
C. Average the two versions, because a gap between version 2.0 and version 4.0 can be split down the middle.
D. Stop before a figure, since these notes are release 2.0 while the question names version 4.0.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Stop before any figure, because these notes are version 2.0 and the question names version 4.0. · variant key: {"answer": "D"}

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
A. Stop here, since the timetable date is 2 May 2020 while the question names 2021.
B. Copy the 2 May 2020 leaving time onto 2 May 2021, because the neighbouring date can answer the one that is missing.
C. Refuse every boat, because a missing year is a printed ban on the later service.
D. Call the two May dates one morning written down with two years.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Stop, because the timetable date is 2 May 2020 and the question names 2 May 2021. · variant key: {"answer": "A"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T12-E-08 · kind E · teaching · tier 3 · versions differ

**User:**
```text
A roster dated 4 June 2022 is the page in hand. The question asks about 4 June 2023. What do you do?
```

**Thinking:**
```text
The dates differ. The later roster is not on this page.
```

**Reply:**
```text
I stop. The roster date is 4 June 2022 and the question names 4 June 2023.
```

**Variant user:**
```text
The page in hand is a roster from 4 June 2022. The ask concerns 4 June 2023. Which move do you make?
A. Answer from 4 June 2022, because the next year's roster can be copied from the page already in hand.
B. Call the two June dates one day written down with two different years on the same line.
C. Stop here, since the roster date is 4 June 2022 while the question names 4 June 2023.
D. Deny the later roster, because a missing year is a printed refusal of every name on that later date.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Stop, because the roster date is 4 June 2022 and the question names 4 June 2023. · variant key: {"answer": "C"}

*invented: scenario, checker trace, or question written for this item (key by construction)*
