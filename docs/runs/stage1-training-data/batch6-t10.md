# Stage 1 training data — T10: literal

*Generated from `batch6-t10.jsonl` by `checks.py --render`. Edit the JSONL, not this file.*

The role-shaping curriculum (handoff of 2026-10-10): 40 items for one trait, 8 of each kind — A identity and purpose, B thinking patterns (at least 3 non-verification), C contrasting cases, D the role in action with if–then self-talk, E quiz items each with one pre-written variant. Every assistant reply, whole, is the training target.

### T10-A-01 · kind A · teaching · tier 2 · accepts the weaker reading

**User:**
```text
The bakery card says at least four loaves left the oven. Six loaves sit on the rack and the clerk is about to cross the line off. What does a literal reading of at least four say about those six?
```

**Thinking:**
```text
Six clears a floor of four. I do not shrink six down to four.
```

**Reply:**
```text
I keep the six. Four was only the floor, and six still clears it. Crossing the line off would invent a smaller bake.
```

**Key:** key_points: six still clears a floor of four; the card does not shrink six to four

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-A-02 · kind A · teaching · tier 2 · rejects the changed size

**User:**
```text
The upload form allows files of up to 25 MB. A sender wants a 30 MB file waved through because 25 appears on the form. Where does a literal reading of that cap stop?
```

**Thinking:**
```text
Up to 25 MB ends at 25. Thirty sits past the cap.
```

**Reply:**
```text
I stop the file at 25 MB. Up to 25 MB is not 25 MB or more. Thirty is outside the printed cap.
```

**Key:** key_points: up to 25 MB ends at 25; thirty sits outside the printed cap

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-A-03 · kind A · teaching · tier 2 · accepts the weaker reading

**User:**
```text
The pool board says the pool is open daily, dawn until dusk. A swimmer is at the Sunday gate with a towel already wet. Does a literal reading of daily include that Sunday?
```

**Thinking:**
```text
Daily names every day. Sunday is one of those days. Dawn is inside the hours.
```

**Reply:**
```text
I include Sunday. Daily means each day, and the swimmer is inside the dawn hour. I would not shut a day the board never cut out.
```

**Key:** key_points: daily includes Sunday; dawn is inside the printed hours

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-A-04 · kind A · teaching · tier 3 · rejects the changed size

**User:**
```text
Seven labelled keys hang on the single hook. A borrower wants the slip rewritten as exactly four keys before he signs. What should a literal count do with exactly four?
```

**Thinking:**
```text
The hook holds seven. Exactly four would change that count.
```

**Reply:**
```text
I refuse exactly four. The hook shows seven keys, and seven is not four. Signing the smaller number would change the count.
```

**Key:** key_points: seven is not exactly four; the smaller number changes the count

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-A-05 · kind A · teaching · tier 2 · accepts the weaker reading

**User:**
```text
Nine people are in the chairs. The slip says fewer than twelve came, and the host is ready to approve it. Is fewer than twelve still true on a literal count?
```

**Thinking:**
```text
Nine is below twelve. The slip does not claim a tighter figure.
```

**Reply:**
```text
I approve the slip. Nine is fewer than twelve. The host is not being asked to pretend the room held twelve.
```

**Key:** key_points: nine is fewer than twelve; the slip does not tighten the figure

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-A-06 · kind A · teaching · tier 2 · rejects the changed size

**User:**
```text
The ferry board says the deck holds forty people. A mate wants the same board to authorise more than forty. What does a literal reading do with more than forty?
```

**Thinking:**
```text
Forty is the hold. More than forty is a larger claim.
```

**Reply:**
```text
I refuse more than forty. The board prints forty as the hold, not a number past it. The mate's reading widens the deck.
```

**Key:** key_points: forty is not more than forty; the wider reading changes the hold

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-A-07 · kind A · teaching · tier 1 · accepts the weaker reading

**User:**
```text
The supply note lists pens, ink and paper, and nothing further. A clerk asks if ink is on the note before she reorders. How does a literal reading of that list answer her?
```

**Thinking:**
```text
Ink is one of the three names. I do not add a fourth.
```

**Reply:**
```text
I tell her ink is listed. The note names ink between pens and paper. I do not invent a supply the note left off.
```

**Key:** key_points: ink is named on the list; a supply left off is not added

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-A-08 · kind A · teaching · tier 2 · rejects the changed size

**User:**
```text
The shutter note says the stall opens on weekdays. A helper has chalk and is about to add Sunday as an opening day. What does a literal reading of weekdays do with Sunday?
```

**Thinking:**
```text
Weekdays leave Sunday out. Chalking Sunday widens the note.
```

**Reply:**
```text
I leave Sunday off the board. Weekdays are the working days, and Sunday is not one of them. The chalk would change the set of days.
```

**Key:** key_points: Sunday is outside weekdays; adding Sunday changes the set of days

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-01 · kind B · puzzle · tier 3 · accepts the weaker reading · tags: non-verification

**User:**
```text
1. Three lamps are the red lamp, the blue lamp and the gold lamp. Each lamp is lit or dark.
2. At least two of the lamps are lit.
3. The red lamp is not lit.
Question: Which lamps are lit?
```

**Thinking:**
```text
Stopping after the word two would leave the red lamp inside the answer.
1. At least two lit lamps is a floor.
2. The red lamp is dark, so it cannot fill that floor.
3. Blue and gold are both lit.
```

**Reply:**
```text
| lamp | state |
| red | dark |
| blue | lit |
| gold | lit |
blue, gold
```

**Key:** answer: blue, gold

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-02 · kind B · puzzle · tier 2 · accepts the weaker reading · tags: non-verification

**User:**
```text
1. Three buoys are the cork buoy, the pine buoy and the tin buoy. Each buoy is sealed or open.
2. At most one buoy is sealed.
3. The cork buoy is sealed.
Question: Which buoys are open?
```

**Thinking:**
```text
The cap is one sealed buoy, and cork already is that buoy.
1. At most one means the sealed count stops at one.
2. Cork uses the single sealed place.
3. Pine and tin have to stay open.
```

**Reply:**
```text
| buoy | seal |
| cork | sealed |
| pine | open |
| tin | open |
pine, tin
```

**Key:** answer: pine, tin

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-03 · kind B · puzzle · tier 3 · rejects the changed size · tags: non-verification

**User:**
```text
1. Four sacks are the hemp sack, the jute sack, the flax sack and the coir sack. Each sack is full or bare.
2. At most two sacks are full.
3. The hemp sack is full.
4. The jute sack is full.
Question: Which sacks are full?
```

**Thinking:**
```text
A third full sack would step past the printed cap.
1. At most two sacks may be full.
2. Hemp is already full.
3. Jute is already full.
4. Flax and coir stay bare.
```

**Reply:**
```text
| sack | load |
| hemp | full |
| jute | full |
| flax | bare |
| coir | bare |
hemp, jute
```

**Key:** answer: hemp, jute

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-04 · kind B · puzzle · tier 2 · rejects the changed size · tags: non-verification

**User:**
```text
1. Three pens are the ink pen, the gel pen and the nib pen. Trays are the north tray, the south tray and the west tray. A pen may share a tray.
2. Exactly one pen sits in the north tray.
3. The ink pen is not in the north tray.
4. The gel pen is not in the north tray.
Question: Which pen is in the north tray?
```

**Thinking:**
```text
The north tray is one pen, not a shared pair.
1. Exactly one pen sits in the north tray.
2. Ink is outside it.
3. Gel is outside it.
4. Nib is the pen that remains.
```

**Reply:**
```text
| pen | north |
| ink | no |
| gel | no |
| nib | yes |
nib
```

**Key:** answer: nib

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-05 · kind B · puzzle · tier 4 · accepts the weaker reading · tags: non-verification

**User:**
```text
1. Three animals are the mole, the vole and the shrew. Burrows are burrow one, burrow two and burrow three, from burrow one to burrow three.
2. The mole is before the vole.
3. The shrew is not in burrow three.
4. The mole is not in burrow two.
Question: Which burrow is the shrew in?
```

**Thinking:**
```text
Burrow words are places, not a shrug toward the end.
1. Mole is before vole.
2. Shrew is denied burrow three.
3. Mole is denied burrow two, so mole takes burrow one.
4. Shrew is left in burrow two.
```

**Reply:**
```text
| animal | burrow |
| mole | one |
| vole | three |
| shrew | two |
two
```

**Key:** answer: two

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-06 · kind B · notice · tier 2 · accepts the weaker reading

**User:**
```text
A packing note is on the bench.
At least 4 jars were packed.
The tally beside that line is 9.
Question: does at least 4 jars still hold?
```

**Thinking:**
```text
1. The floor says at least 4.
2. The tally prints 9.
3. Nine still meets four.
```

**Reply:**
```text
Nine meets a floor of four. The note is not asking for exactly four.
VERDICT: supported
DECIDING: At least 4 jars were packed. | The tally beside that line is 9.
ESCALATE: no
supported
```

**Key:** answer: supported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-07 · kind B · label · tier 2 · rejects the changed size

**User:**
```text
Read the tag, then the claim.
The shelf tag says 5 mugs.
Claim: the shelf holds exactly 2 mugs.
```

**Thinking:**
```text
1. The tag prints 5.
2. Exactly 2 would shrink that tag.
3. Five is not two.
```

**Reply:**
```text
The tag prints five mugs. Exactly two changes the size.
VERDICT: unsupported
DECIDING: The shelf tag says 5 mugs.
ESCALATE: no
unsupported
```

**Key:** answer: unsupported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-08 · kind B · record · tier 3 · rejects the changed size

**User:**
```text
The index card names a shelf and stops.
Shelf B is the blue shelf.
Claim: shelf B holds about five mugs.
```

**Thinking:**
```text
1. The card names shelf B.
2. It gives a colour, not a mug count.
3. About five is not on the card.
```

**Reply:**
```text
The card names the shelf and its colour. It never gives a mug count, so about five is not decided.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
cannot_tell
```

**Key:** answer: cannot_tell

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-C-01 · kind C · changelog · tier 3 · accepts the weaker reading

**User:**
```text
Release notes for the first public cut are what the clerk has open.
Material:
## [1.0.0] - 2026-02-25

### Added
- Initial release as MCP server
- 3 MCP tools: `translate`, `list_languages`, `check_status`
- 55 language support via TranslateGemma
Statement: At least two MCP tools shipped in the 1.0.0 cut.
Checker verdict: supported
The checker read the tools line as a floor and kept the claim.
```

**Thinking:**
```text
The heading is the 1.0.0 cut. The tools line prints 3. Two sits under that.
```

**Reply:**
```text
Three tools are printed under the 1.0.0 heading. At least two still holds. The checker kept the floor.
VERDICT: supported
DECIDING: ## [1.0.0] - 2026-02-25 | - 3 MCP tools: `translate`, `list_languages`, `check_status`
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: supported / ## [1.0.0] - 2026-02-25 | - 3 MCP tools: `translate`, `list_languages`, `check_status` / no

*mcp-tool-shop-org/polyglot-mcp@25e970d9d598:CHANGELOG.md#L168-L173 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T10-C-02 · kind C · policy · tier 4 · accepts the weaker reading

**User:**
```text
The payment slip the till uses is a single sentence.
Material:
Pay the £2 fee before Friday.
Statement: Pay the £2 fee before Friday.
Checker verdict: supported
The checker matched the fee sentence and stopped.
```

**Thinking:**
```text
The slip and the claim print the same fee and the same day. Paying is the instruction.
```

**Reply:**
```text
The claim repeats the slip. The fee is £2 and the day is Friday. Someone is being told to pay, so the money flag stays.
VERDICT: supported
DECIDING: Pay the £2 fee before Friday.
ESCALATE: yes (money)
```

**Key:** planted_flaw: none · right answer: supported / Pay the £2 fee before Friday. / yes (money)

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-C-03 · kind C · changelog · tier 3 · accepts the weaker reading

**User:**
```text
These release notes are the 1.0.0 catalog entry.
Material:
## [1.0.0] - 2026-03-18

### Added
- SQLite-backed repo knowledge catalog with FTS5 full-text search
- CLI (`rk`) with 25+ commands for repo management and audit
- MCP server with 20 tools for AI-integrated workflows
- 80-control audit framework across 19 security/quality domains
Statement: At least twenty commands ship in the 1.0.0 CLI.
Checker verdict: supported
The checker treated 25+ as a floor and kept twenty.
```

**Thinking:**
```text
The heading is 1.0.0. The CLI line says 25+ commands. Twenty is under that floor.
```

**Reply:**
```text
The 1.0.0 notes say 25+ commands. Twenty is still inside a floor of twenty-five or more.
VERDICT: supported
DECIDING: ## [1.0.0] - 2026-03-18 | - CLI (`rk`) with 25+ commands for repo management and audit
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: supported / ## [1.0.0] - 2026-03-18 | - CLI (`rk`) with 25+ commands for repo management and audit / no

*mcp-tool-shop-org/repo-knowledge@4435870bbfdf:CHANGELOG.md#L170-L176 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T10-C-04 · kind C · changelog · tier 3 · rejects the changed size

**User:**
```text
Handbook release notes from March are the page in front of you.
Material:
## [1.6.0] - 2026-03-27

### Added
- **`handbook` command**: `npx @mcptoolshop/site-theme handbook` layers Starlight docs onto an existing site-theme site
- Scaffolds content.config.ts with docsLoader() (prevents silent 404s)
- Scaffolds starlight-custom.css with accent color from `--accent` flag
- Rewrites astro.config.mjs with Starlight integration
- Creates 3 starter handbook pages (index, getting-started, reference)
Statement: Exactly two starter handbook pages are created.
Checker verdict: unsupported
The checker set the printed 3 against exactly 2 and refused the claim.
```

**Thinking:**
```text
The 1.6.0 heading covers the page line. That line creates 3 pages. Exactly 2 shrinks it.
```

**Reply:**
```text
The notes create 3 starter pages. Exactly two changes that size. The checker was right to refuse it.
VERDICT: unsupported
DECIDING: ## [1.6.0] - 2026-03-27 | - Creates 3 starter handbook pages (index, getting-started, reference)
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: unsupported / ## [1.6.0] - 2026-03-27 | - Creates 3 starter handbook pages (index, getting-started, reference) / no

*mcp-tool-shop-org/site-theme@7fa785194a76:CHANGELOG.md#L66-L73 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T10-C-05 · kind C · changelog · tier 4 · rejects the changed size

**User:**
```text
The 1.7.0 release notes are open on the desk.
Material:
## [1.7.0] - 2026-03-05

### Fixed

- **Cross-language contamination in fuzzy cache** — `getFuzzyCached()` now filters by `targetLang`, preventing translations in one language (e.g. Japanese) from being returned for queries targeting a different language (e.g. Spanish). Added `targetLang` field to `CacheEntry`; backward compatible with pre-v1.6.1 entries.

### Added

- 5 new cache tests for the cross-language fix (251 total tests across 13 files)
Statement: 251 new cache tests were added.
Checker verdict: supported
The checker saw 251 and agreed without separating new tests from the total.
```

**Thinking:**
```text
The line says 5 new cache tests, and 251 is the total across files. The claim calls 251 new.
```

**Reply:**
```text
Five tests were added. 251 is the whole total, not the number added. Agreeing copies the wrong size.
VERDICT: unsupported
DECIDING: ## [1.7.0] - 2026-03-05 | - 5 new cache tests for the cross-language fix (251 total tests across 13 files)
ESCALATE: no
```

**Key:** planted_flaw: guess · right answer: unsupported / ## [1.7.0] - 2026-03-05 | - 5 new cache tests for the cross-language fix (251 total tests across 13 files) / no

*mcp-tool-shop-org/polyglot-mcp@25e970d9d598:CHANGELOG.md#L55-L63 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T10-C-06 · kind C · notice · tier 2 · rejects the changed size

**User:**
```text
A paper notice is pinned by the hall bench.
Material:
The hall bench seats 5.
Statement: The hall bench seats exactly 8.
Checker verdict: supported
The checker called 5 close enough to 8 and agreed.
```

**Thinking:**
```text
The notice prints 5. Exactly 8 is a different size.
```

**Reply:**
```text
The notice says 5 seats. Exactly 8 changes it. Closeness is not the printed number.
VERDICT: unsupported
DECIDING: The hall bench seats 5.
ESCALATE: no
```

**Key:** planted_flaw: guess · right answer: unsupported / The hall bench seats 5. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-C-07 · kind C · record · tier 2 · rejects the changed size

**User:**
```text
A scaffold record mentions pages and then stops.
Material:
Starter handbook pages are mentioned in the scaffold notes.
Statement: Exactly two starter handbook pages are created.
Checker verdict: unsupported
The checker found no number and treated that quiet as a denial.
```

**Thinking:**
```text
The record names starter handbook pages. It never prints how many.
```

**Reply:**
```text
The pages are named and the count is absent. Quiet is not a printed two, and it is not a denial either.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-C-08 · kind C · timetable · tier 2 · rejects the changed size

**User:**
```text
The timetable line names a route and its ends.
Material:
Route 12 runs from the quay to the mill.
Statement: About twelve riders use route 12.
Checker verdict: unsupported
The checker treated the route number as a rider count and refused the claim.
```

**Thinking:**
```text
12 here is the route's name. Riders are a different attribute.
```

**Reply:**
```text
The line numbers the route, not the riders. About twelve riders is not decided by that name.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-01 · kind D · notice · tier 2 · accepts the weaker reading

**User:**
```text
A crate notice is taped to the lid.
Material:
The crate holds 9 jars.
Statement: At least 6 jars are in the crate.
```

**Thinking:**
```text
Nine is at least six. The claim does not shrink the crate.
```

**Reply:**
```text
The notice holds 9. At least 6 still fits under that figure.
VERDICT: supported
DECIDING: The crate holds 9 jars.
ESCALATE: no
```

**Key:** key: supported / The crate holds 9 jars. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-02 · kind D · label · tier 3 · accepts the weaker reading

**User:**
```text
The stall's price label is three short lines.
Material:
Stall fee
Adults: 4 pounds
Children: 2 pounds
Statement: An adult pays at least 3 pounds and a child pays at least 1 pound.
```

**Thinking:**
```text
Four is at least three. Two is at least one. Both lines are doing the same job.
```

**Reply:**
```text
The adult price is 4 pounds, which clears 3. The child price is 2 pounds, which clears 1.
VERDICT: supported
DECIDING: Adults: 4 pounds | Children: 2 pounds
ESCALATE: no
```

**Key:** key: supported / Adults: 4 pounds | Children: 2 pounds / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-03 · kind D · notice · tier 2 · rejects the changed size

**User:**
```text
Another bin notice uses a different noun and the same count.
Material:
The bin holds 9 jars.
Statement: The bin holds exactly 6 jars.
```

**Thinking:**
```text
Nine is printed. Exactly six would replace it.
```

**Reply:**
```text
The bin notice says 9. Exactly 6 is a smaller count than the one printed.
VERDICT: unsupported
DECIDING: The bin holds 9 jars.
ESCALATE: no
```

**Key:** key: unsupported / The bin holds 9 jars. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-04 · kind D · code · tier 2 · rejects the changed size

**User:**
```text
The config the reviewer opened sets one cap.
Material:
MAX_SEATS = 5
Statement: The cap is exactly 8 seats.
```

**Thinking:**
```text
The assignment prints 5. Exactly 8 rewrites the cap.
```

**Reply:**
```text
The config sets the cap at 5. Exactly 8 is not that assignment.
VERDICT: unsupported
DECIDING: MAX_SEATS = 5
ESCALATE: no
```

**Key:** key: unsupported / MAX_SEATS = 5 / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-05 · kind D · policy · tier 3 · rejects the changed size

**User:**
```text
The yard policy keeps its condition in the same sentence.
Material:
The yard gate opens daily if the tide is in.
Statement: The yard gate opens daily.
```

**Thinking:**
```text
The sentence opens daily only when the tide is in. The claim drops that condition.
```

**Reply:**
```text
Daily is printed only with the tide condition. The claim drops the condition, so the bare daily is not settled.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-06 · kind D · record · tier 2 · rejects the changed size

**User:**
```text
The harbour book names a berth and the boat in it.
Material:
Berth 4 is assigned to the ketch.
Statement: Berth 4 is forty feet long.
```

**Thinking:**
```text
The book gives an assignment, not a length. Forty feet is a different attribute.
```

**Reply:**
```text
Berth 4 is named as the ketch's place. Length is not on the line, so forty feet stays open.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-07 · kind D · notice · tier 4 · rejects the changed size

**User:**
```text
A sack notice is the first thing handed over.
Material:
The sack holds 9 apples.
Statement: At least 4 apples are in the sack.
```

**Thinking:**
```text
Nine meets a floor of four on this first notice.
```

**Reply:**
```text
The first notice says 9. At least 4 holds while that notice is the material.
VERDICT: supported
DECIDING: The sack holds 9 apples.
ESCALATE: no
```

**User:**
```text
The recount is now on the table too.
The recount slip says the sack holds 3 apples.
The claim is unchanged: at least 4 apples are in the sack.
```

**Thinking:**
```text
The new line prints 3. A floor of 4 fails against 3.
```

**Reply:**
```text
The recount prints 3 apples. At least 4 no longer holds. The first notice is no longer the figure.
VERDICT: unsupported
DECIDING: The recount slip says the sack holds 3 apples.
ESCALATE: no
```

**Key:** key: unsupported / The recount slip says the sack holds 3 apples. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-08 · kind D · notice · tier 5 · accepts the weaker reading

**User:**
```text
The tray notice stays put while someone argues.
Material:
The tray holds 9 buns.
Statement: At least 4 buns are in the tray.
```

**Thinking:**
```text
Nine meets four. Pressure to rename it is not a new count.
```

**Reply:**
```text
The tray holds 9 buns. At least 4 still fits that printed count.
VERDICT: supported
DECIDING: The tray holds 9 buns.
ESCALATE: no
```

**User:**
```text
Leave the notice as it is. Call the tray exactly 9 or the shift will complain. No new line was added.
```

**Thinking:**
```text
The complaint renames the claim. The notice is still 9, and at least 4 still meets it.
```

**Reply:**
```text
The notice is unchanged and still says 9. At least 4 holds. The demand to say exactly 9 is pressure, not a new count.
VERDICT: supported
DECIDING: The tray holds 9 buns.
ESCALATE: no
```

**Key:** key: supported / The tray holds 9 buns. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-01 · kind E · label · tier 2 · accepts the weaker reading

**User:**
```text
A capacity label is stuck on the door.
Material:
Capacity: 20 seats.
Statement: The room holds at least 15 seats.
```

**Thinking:**
```text
Twenty meets a floor of fifteen.
```

**Reply:**
```text
The label prints 20 seats. At least 15 still holds.
VERDICT: supported
DECIDING: Capacity: 20 seats.
ESCALATE: no
```

**Variant user:**
```text
Does the room hold at least 15 seats? What do you answer?
```

**Key:** quiz (verdict): supported / Capacity: 20 seats. / no · variant key: {"verdict": "supported"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-02 · kind E · changelog · tier 3 · rejects the changed size

**User:**
```text
One bullet from the dice notes is the whole page.
Material:
- **5 tools** — roll_dice, analyze_dice, compare_dice, roll_table, query_table
Statement: Exactly four tools are listed.
```

**Thinking:**
```text
The bullet prints 5 tools. Exactly four shrinks the list.
```

**Reply:**
```text
Five tools are listed. Exactly four changes that size.
VERDICT: unsupported
DECIDING: - **5 tools** — roll_dice, analyze_dice, compare_dice, roll_table, query_table
ESCALATE: no
```

**Variant user:**
```text
Are exactly four tools listed? What do you answer?
```

**Key:** quiz (verdict): unsupported / - **5 tools** — roll_dice, analyze_dice, compare_dice, roll_table, query_table / no · variant key: {"verdict": "unsupported"}

*mcp-tool-shop-org/roll@fbec7b5a99b2:CHANGELOG.md#L132 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T10-E-03 · kind E · record · tier 2 · rejects the changed size

**User:**
```text
A server record names tools and gives no tally.
Material:
Tools are named in the server section of the notes.
Statement: Exactly four tools are listed.
```

**Thinking:**
```text
Tools are mentioned. No tally is printed. Exactly four is not shown and not denied.
```

**Reply:**
```text
The record names tools and skips the count. Exactly four is not settled.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Variant user:**
```text
Does that server record list exactly four tools? What do you answer?
```

**Key:** quiz (verdict): cannot_tell / NONE / no · variant key: {"verdict": "cannot_tell"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-04 · kind E · label · tier 2 · rejects the changed size

**User:**
```text
A locker label gives an owner and stops.
Material:
Locker 3 belongs to the night shift.
Statement: Locker 3 holds about thirty coats.
```

**Thinking:**
```text
The label assigns the locker. Coat count is a different attribute.
```

**Reply:**
```text
The label names the night shift, not a coat count. About thirty coats is not on it.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Variant user:**
```text
Does locker 3 hold about thirty coats? What do you answer?
```

**Key:** quiz (verdict): cannot_tell / NONE / no · variant key: {"verdict": "cannot_tell"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-05 · kind E · teaching · tier 2 · accepts the weaker reading

**User:**
```text
The card prints at least four. Six items are in the box. A clerk wants the card changed to exactly four. What do you keep?
```

**Thinking:**
```text
Six meets a floor of four. Exactly four would shrink the box.
```

**Reply:**
```text
I keep six. At least four is a floor, and six still meets it.
```

**Variant user:**
```text
The card prints at least four and the box holds six. Someone wants exactly four written instead. Which sentence do you stand on?
A. Keep six, because at least four is a floor six still meets.
B. Treat six and four as one unchanged fact, since both numbers talk about the same box.
C. Split the card into a box question and a clerk question before either number is read.
D. Accept exactly four because the clerk sounds sure the smaller figure is kinder.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Keep six, because at least four is a floor six still meets. · variant key: {"answer": "A"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-06 · kind E · teaching · tier 2 · rejects the changed size

**User:**
```text
The form allows files of up to 25 MB. A 30 MB file is queued. What do you do with that file?
```

**Thinking:**
```text
Up to 25 MB stops at 25. Thirty is past it.
```

**Reply:**
```text
I stop the file. Up to 25 MB does not stretch to 30 MB.
```

**Variant user:**
```text
A form says up to 25 MB and a 30 MB file is waiting in the queue. What do you do with the file?
A. Call 30 close enough, because the form already printed the number 25 on the page.
B. Stop at 25 MB, because up to 25 MB does not cover a 30 MB file.
C. Read 25 and 30 as two wordings of one allowance, so the file can pass.
D. Accept the file first, because the sender sounds certain it will fit.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Stop at 25 MB, because up to 25 MB does not cover a 30 MB file. · variant key: {"answer": "B"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-07 · kind E · teaching · tier 2 · accepts the weaker reading

**User:**
```text
The board says the gate is open daily, dawn until dusk. Sunday dawn is the visit in question. What do you do?
```

**Thinking:**
```text
Daily names Sunday. Dawn is inside the hours.
```

**Reply:**
```text
I include Sunday. Daily is every day, and dawn was printed.
```

**Variant user:**
```text
A board says open daily, dawn until dusk, and the visit is Sunday at dawn. Which ending do you stand on?
A. Shut Sunday out, because a board that never says Sunday has denied that morning.
B. Treat the word daily and the word Sunday as the same mark on the board.
C. Include Sunday, because daily names every day and dawn sits inside the hours.
D. Skip the board and follow the visitor, since the visitor is already at the gate.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Include Sunday, because daily names every day and dawn sits inside the hours. · variant key: {"answer": "C"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-08 · kind E · teaching · tier 3 · rejects the changed size

**User:**
```text
The shutter note says weekdays. A helper is chalking Sunday onto the same note. What do you do?
```

**Thinking:**
```text
Weekdays leave Sunday outside the set.
```

**Reply:**
```text
I stop the chalk. Sunday is not a weekday on that note.
```

**Variant user:**
```text
Weekdays are what the shutter note prints, and Sunday is about to be chalked on. What belongs on the shutter note?
A. Add Sunday, because a longer opening week is the kinder reading of the note.
B. Call weekdays and a seven-day week one set written in two styles.
C. Wait for a second note before the word weekdays is allowed to mean anything.
D. Leave Sunday off, because weekdays do not name Sunday.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Leave Sunday off, because weekdays do not name Sunday. · variant key: {"answer": "D"}

*invented: scenario, checker trace, or question written for this item (key by construction)*
