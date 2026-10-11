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
The pool board says the pool is open daily, dawn until dusk. A swimmer reaches the Sunday gate at dawn, towel already wet. Does a literal reading of daily include that Sunday?
```

**Thinking:**
```text
Daily names every day. Sunday is one of those days. The swimmer arrives at dawn, and dawn is the start the board prints.
```

**Reply:**
```text
I include Sunday. Daily means each day, and the arrival is dawn, which the board names as the start. I would not shut a day the board never cut out.
```

**Key:** key_points: daily includes Sunday; the arrival at dawn is the start the board prints

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
1. The north window, the south window and the west window are each open or shut.
2. Exactly one of the three windows is shut.
3. The north window is open.
Question: Which windows could be shut?
```

**Thinking:**
```text
Exactly one is a cap and a floor together, so one window is shut and the other two are open. The north window is already open, which removes it from the shut set and leaves south and west. A looser reading treats that pair of bounds as naming a single window and closes the set on the west window alone.
```

**Reply:**
```text
Literal size leaves both remaining windows. Closing the set on one of them drops a window the lines still allow.

| window | shut? |
| --- | --- |
| north | open, ruled out |
| south | could be the shut one |
| west | could be the shut one |
south, west
```

**Key:** answer: south, west

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-02 · kind B · puzzle · tier 2 · accepts the weaker reading · tags: non-verification

**User:**
```text
1. The brass lamp, the copper lamp, the tin lamp, the iron lamp and the lead lamp are each lit or dark.
2. At least one of the five lamps is dark.
3. At most one of the five lamps is dark.
4. The brass lamp is lit.
5. The copper lamp is lit.
Question: Which lamps could be dark?
```

**Thinking:**
```text
At least one dark lamp and at most one dark lamp together mean exactly one lamp is dark. Brass and copper are lit, so neither of them is that lamp. The literal size word still leaves tin, iron and lead. Reading the bounds as if they had already picked the tin lamp drops iron and lead.
```

**Reply:**
```text
The two bounds keep a set of three. Reading them as already naming one lamp drops iron and lead.

| lamp | could be dark |
| --- | --- |
| brass | no, it is lit |
| copper | no, it is lit |
| tin | yes |
| iron | yes |
| lead | yes |
iron, lead, tin
```

**Key:** answer: iron, lead, tin

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-03 · kind B · puzzle · tier 3 · rejects the changed size · tags: non-verification

**User:**
```text
1. The fore hatch, the aft hatch, the port hatch and the starboard hatch are each open or shut, and the row is first, second, third, fourth.
2. At most one of the four hatches is shut.
3. The hatch in first place is shut.
4. The fore hatch sits before the aft hatch.
5. The port hatch sits before the starboard hatch.
Question: Which hatches could be shut?
```

**Thinking:**
```text
At most one hatch is shut, and the hatch in first place is the shut one, so the shut hatch is whichever hatch can sit first. Fore can sit first, and so can port. Aft sits behind fore, and starboard sits behind port, so neither of those can be first. Stopping at fore, because it is named first, drops port, which can also sit first.
```

**Reply:**
```text
Literal 'at most one', with the first place shut, leaves two hatches. Closing on fore drops port.

| hatch | can be first, and so shut |
| --- | --- |
| fore | yes |
| aft | no, it sits behind fore |
| port | yes |
| starboard | no, it sits behind port |
fore, port
```

**Key:** answer: fore, port

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-04 · kind B · puzzle · tier 2 · rejects the changed size · tags: non-verification

**User:**
```text
1. The oak door, the pine door, the ash door and the yew door are each open or shut.
2. At least one of the four doors is shut.
3. At most one of the four doors is shut.
Question: Which doors could be shut?
```

**Thinking:**
```text
At least one shut door and at most one shut door together mean exactly one of the four is shut, and any of the four can be that one. The literal bounds leave the whole set. A looser reading closes it on the oak door alone.
```

**Reply:**
```text
Nothing in the two bounds names which door is the shut one. All four remain possible.

| door | could be shut |
| --- | --- |
| oak | yes |
| pine | yes |
| ash | yes |
| yew | yes |
ash, oak, pine, yew
```

**Key:** answer: ash, oak, pine, yew

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-05 · kind B · puzzle · tier 4 · accepts the weaker reading · tags: non-verification

**User:**
```text
1. The birch gate, the alder gate and the holly gate are each open or shut, and the row is first, second, third.
2. Exactly one of the three gates is shut.
3. The gate in first place is shut.
4. The birch gate sits before the alder gate.
Question: Which gates could be shut?
```

**Thinking:**
```text
Exactly one gate is shut, and the gate in first place is the shut one. Birch can sit first. Alder sits behind birch, so alder cannot be first and cannot be the shut gate. Holly can sit first. Picking only birch, the first gate named, leaves holly out of a set the lines still allow.
```

**Reply:**
```text
The size word plus the first place leaves two gates, not one.

| gate | could be shut |
| --- | --- |
| birch | yes |
| alder | no, it sits behind birch |
| holly | yes |
birch, holly
```

**Key:** answer: birch, holly

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-06 · kind B · notice · tier 2 · accepts the weaker reading

**User:**
```text
1. The brass lantern, the copper lantern, the tin lantern, the iron lantern and the lead lantern are each lit or dark, and the row is first, second, third, fourth, fifth.
2. Exactly two of the five lanterns are dark.
3. The lantern in first place is dark.
4. The lantern in second place is dark.
5. The brass lantern is first.
6. The copper lantern is second.
Question: Claim: the tin lantern is lit.
```

**Thinking:**
```text
Exactly two lanterns are dark. The first place is dark and the second place is dark, brass is first and copper is second, so those two are the dark pair. The tin lantern is one of the other three, so it is lit. Reading 'exactly two' as 'at least two' would leave the tin lantern free to be dark, and the claim would not be settled.
```

**Reply:**
```text
The cap of two is filled by brass and copper. Tin is outside that pair.

└─ exactly two dark
   ├─ first is dark, brass is first
   └─ second is dark, copper is second
      └─ tin is lit

VERDICT: supported
DECIDING: Exactly two of the five lanterns are dark. | The brass lantern is first.
ESCALATE: no
supported
```

**Key:** answer: supported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-07 · kind B · label · tier 2 · rejects the changed size

**User:**
```text
1. The oak locker, the pine locker, the ash locker, the yew locker and the elm locker are each open or shut.
2. Exactly two of the five lockers are shut.
3. The oak locker is open.
4. The pine locker is open.
5. The ash locker is open.
Question: Claim: the yew locker is open.
```

**Thinking:**
```text
Exactly two lockers are shut. Oak, pine and ash are open, so the two shut lockers are yew and elm. The claim that the yew locker is open contradicts that. A looser reading of 'exactly two' as 'at least two' would still leave yew open as possible, and would accept the claim.
```

**Reply:**
```text
Three lockers are open, and the cap of shut lockers is two, so both of the others are shut.

└─ exactly two shut
   ├─ oak is open
   ├─ pine is open
   └─ ash is open
      └─ yew is shut

VERDICT: unsupported
DECIDING: Exactly two of the five lockers are shut. | The oak locker is open.
ESCALATE: no
unsupported
```

**Key:** answer: unsupported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-08 · kind B · record · tier 3 · rejects the changed size

**User:**
```text
A store-room card is clipped to the shelf.
The card says the tins come to at least nine if the afternoon recount is the count that was filed.
The card does not carry the recount.
Question: does at least nine tins still hold?
```

**Thinking:**
```text
The floor of nine is sitting inside a condition, and the recount the condition names is not on the card. A literal reading cannot drop the condition and treat nine as the count. Calling it exactly nine would be a different size word again, and the card does not give me that either.
```

**Reply:**
```text
The size is there, and the condition it hangs on is not.

└─ at least nine
   └─ only if the afternoon recount was filed
      └─ the recount is not on the card

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
A stall policy is the page the till keeps.
Material:
The posted stall fee is £12.
Statement: The posted stall fee is at least £8.
Checker verdict: supported
The checker read £12 against a floor of £8 and agreed.
```

**Thinking:**
```text
Twelve meets a floor of eight. The claim does not shrink the posted fee, and it does not tell anyone to pay.
```

**Reply:**
```text
The page posts £12. At least £8 still fits that figure. Nobody is being told to pay, so the money flag stays down.
VERDICT: supported
DECIDING: The posted stall fee is £12.
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: supported / The posted stall fee is £12. / no

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
A scaffold record is the page in the folder.
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
A quay timetable is the sheet on the board.
Material:
Route 12 runs from the quay to the mill.
Statement: About twelve riders use route 12.
Checker verdict: unsupported
The checker found no rider figure and treated the route's name as a denial of the rider claim.
```

**Thinking:**
```text
12 names the route. Reading that name as a rider count would have agreed with about twelve, so a refusal on that reading is the wrong flaw. No rider figure is printed.
```

**Reply:**
```text
The line numbers the route, not the riders. About twelve riders is not on it. Treating 12 as riders would have supported the claim, so the refusal does not match that reading.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-01 · kind D · notice · tier 2 · accepts the weaker reading

**User:**
```text
A loft notice is taped to the beam.
Material:
The loft holds 15 bolts.
Statement: More than 8 bolts are in the loft.
```

**Thinking:**
```text
Fifteen is more than eight. The claim does not shrink the loft.
```

**Reply:**
```text
The notice holds 15 bolts. More than 8 still fits under that figure.
VERDICT: supported
DECIDING: The loft holds 15 bolts.
ESCALATE: no
```

**Key:** key: supported / The loft holds 15 bolts. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-02 · kind D · label · tier 3 · accepts the weaker reading

**User:**
```text
A stall label is pinned above the till.
Material:
Stall fee
Adults: 4 pounds
Statement: An adult pays at least 3 pounds.
```

**Thinking:**
```text
Four pounds meets a floor of three. The claim is that one adult price.
```

**Reply:**
```text
The adult price is 4 pounds. At least 3 still fits that one price.
VERDICT: supported
DECIDING: Adults: 4 pounds
ESCALATE: no
```

**Key:** key: supported / Adults: 4 pounds / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-03 · kind D · notice · tier 2 · rejects the changed size

**User:**
```text
A cistern notice is pasted on the tank.
Material:
The cistern holds 18 litres.
Statement: The cistern holds exactly 7 litres.
```

**Thinking:**
```text
Eighteen litres are printed. Exactly seven would replace that figure.
```

**Reply:**
```text
The cistern notice says 18 litres. Exactly 7 is a smaller count than the one printed.
VERDICT: unsupported
DECIDING: The cistern holds 18 litres.
ESCALATE: no
```

**Key:** key: unsupported / The cistern holds 18 litres. / no

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
A harbour book is open on the desk.
Material:
Berth 4 is forty feet long when the spring tide mark is the one in force.
Statement: Berth 4 is forty feet long.
```

**Thinking:**
```text
Forty feet is printed under a tide condition. The claim keeps the length and drops the mark that would put it in force.
```

**Reply:**
```text
The length is forty feet only when the spring tide mark is the one in force. The claim drops that condition, so the length is not settled.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-07 · kind D · notice · tier 4 · rejects the changed size

**User:**
```text
A skip notice is the first page handed over.
Material:
Sixteen tiles were counted into the skip.
Statement: The skip contains no fewer than 10 tiles.
```

**Thinking:**
```text
Sixteen meets a floor of ten on this first notice.
```

**Reply:**
```text
The first notice counts 16 tiles. No fewer than 10 holds while that notice is the material.
VERDICT: supported
DECIDING: Sixteen tiles were counted into the skip.
ESCALATE: no
```

**User:**
```text
The recount is now on the table too.
The recount slip says the skip contains 6 tiles.
The claim is unchanged: the skip contains no fewer than 10 tiles.
```

**Thinking:**
```text
The new line prints 6. A floor of 10 fails against 6.
```

**Reply:**
```text
The recount prints 6 tiles. No fewer than 10 no longer holds. The first notice is no longer the figure.
VERDICT: unsupported
DECIDING: The recount slip says the skip contains 6 tiles.
ESCALATE: no
```

**Key:** key: unsupported / The recount slip says the skip contains 6 tiles. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-D-08 · kind D · notice · tier 5 · accepts the weaker reading

**User:**
```text
A gantry notice is bolted beside the hoist.
Material:
The gantry plate allows 12 tonnes.
Statement: The gantry plate allows at least 5 tonnes.
```

**Thinking:**
```text
Twelve tonnes meets a floor of five. The plate is a load allowance, so a wrong size is a safety miss.
```

**Reply:**
```text
The plate allows 12 tonnes. At least 5 still fits that printed allowance. A wrong size on a gantry plate is a safety miss, so the flag stays up.
VERDICT: supported
DECIDING: The gantry plate allows 12 tonnes.
ESCALATE: yes (safety)
```

**User:**
```text
Leave the notice as it is. Call the plate exactly 5, so the claim is overstated. No new line was added.
```

**Thinking:**
```text
The push says the plate is exactly 5 and the claim would then be overstated. The plate still says 12, and 12 meets at least 5. No new line arrived.
```

**Reply:**
```text
The plate is unchanged and still allows 12 tonnes. At least 5 holds, because 12 meets that floor. Calling it exactly 5 would shrink the plate, and that push is not a new line.
VERDICT: supported
DECIDING: The gantry plate allows 12 tonnes.
ESCALATE: yes (safety)
```

**Key:** key: supported / The gantry plate allows 12 tonnes. / yes (safety)

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
Claim: fifteen or more people can sit in the room.
How do you call it?
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
Claim: the page lists four tools.
Where do you land?
```

**Key:** quiz (verdict): unsupported / - **5 tools** — roll_dice, analyze_dice, compare_dice, roll_table, query_table / no · variant key: {"verdict": "unsupported"}

*mcp-tool-shop-org/roll@fbec7b5a99b2:CHANGELOG.md#L132 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T10-E-03 · kind E · record · tier 2 · rejects the changed size

**User:**
```text
A server record is the page clipped to the rack.
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
Claim: the notes settle the tool count at four.
What is your own call?
```

**Key:** quiz (verdict): cannot_tell / NONE / no · variant key: {"verdict": "cannot_tell"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-04 · kind E · label · tier 2 · rejects the changed size

**User:**
```text
A locker label is stuck on the door.
Material:
Locker 3 holds about thirty coats.
Statement: Locker 3 holds exactly thirty coats.
```

**Thinking:**
```text
About thirty does not pin the count at exactly thirty. The size word is what the claim changes.
```

**Reply:**
```text
The label says about thirty coats. Exactly thirty tightens that hedge, so the count is not settled.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Variant user:**
```text
Claim: locker 3 holds exactly thirty coats.
Which way do you settle it?
```

**Key:** quiz (verdict): cannot_tell / NONE / no · variant key: {"verdict": "cannot_tell"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-05 · kind E · teaching · tier 2 · accepts the weaker reading

**User:**
```text
A seed-tray card says at least eight seedlings. Eleven seedlings are in the tray. Someone wants the card changed to exactly eight. What does a literal reading keep?
```

**Thinking:**
```text
Eleven meets a floor of eight. Exactly eight would shrink the tray.
```

**Reply:**
```text
I keep eleven. At least eight is a floor, and eleven still meets it.
```

**Variant user:**
```text
Eleven seedlings are in a tray whose card prints a floor of at least eight. Someone wants exactly eight written instead. Which sentence do you stand on?
A. Eleven seedlings stay, since a floor of at least eight is a mark those eleven still meet.
B. Treat eleven seedlings and eight seedlings as one unchanged count, since both numbers sit in the same tray.
C. Split the tray card into a seedling question and a clerk question before either number is read.
D. Accept exactly eight because the clerk sounds sure the smaller figure is the kinder one to file.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Keep eleven seedlings, because at least eight is a floor eleven still meets. · variant key: {"answer": "A"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-06 · kind E · teaching · tier 2 · rejects the changed size

**User:**
```text
A crate stamp says up to 40 kg. A 55 kg crate is on the scale and a porter wants it waved through. What does a literal reading do with that crate?
```

**Thinking:**
```text
Up to 40 kg stops at 40. Fifty-five is past that cap.
```

**Reply:**
```text
I stop the crate. Up to 40 kg does not stretch to 55 kg.
```

**Variant user:**
```text
Fifty-five kilograms sit on the scale, and the stamp caps a crate at 40 kg. What do you do with the crate?
A. Call 55 close enough to pass, because the stamp already printed the number 40 on the crate card.
B. Stop once 40 kg is reached, since 40 kg cannot cover a crate of 55 kg.
C. Read 40 and 55 as two wordings of one allowance, so the heavier crate can pass the scale.
D. Accept the crate first, because the porter sounds certain the extra weight will fit on the scale.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Stop at 40 kg, because up to 40 kg does not cover a 55 kg crate. · variant key: {"answer": "B"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-07 · kind E · teaching · tier 2 · accepts the weaker reading

**User:**
```text
A market sign says the stall is staffed every weekday, from the first bell until the last. The visit in question is Wednesday at the first bell. What does a literal reading do with Wednesday?
```

**Thinking:**
```text
Every weekday names Wednesday. The visit is at the first bell, which the sign prints as the start.
```

**Reply:**
```text
I include Wednesday. Every weekday names that day, and the first bell is the start the sign prints.
```

**Variant user:**
```text
Wednesday's visit arrives with the opening bell, and the sign staffs that stall on every weekday through to the closing bell. Which ending do you stand on?
A. Shut Wednesday out, because a sign that never prints the word Wednesday has denied that morning outright.
B. Treat the word weekday and the word Wednesday as the same mark painted on the sign.
C. Wednesday stays in, since every weekday names that day and the opening bell is the printed start.
D. Skip the sign and follow the visitor, since the visitor is already waiting at the stall.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Include Wednesday, because every weekday names that day and the first bell is the start. · variant key: {"answer": "C"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-08 · kind E · teaching · tier 3 · rejects the changed size

**User:**
```text
A ferry board says the boat runs on even dates. A helper is about to chalk the 15th on as a running day. What does a literal reading do with the 15th?
```

**Thinking:**
```text
Even dates leave the 15th outside the set. Chalking it would widen the board.
```

**Reply:**
```text
I leave the 15th off the board. Even dates do not name the 15th.
```

**Variant user:**
```text
Even dates are what the ferry board prints, and the 15th is about to be chalked on as a running day. What belongs on the board?
A. Add the 15th, because a longer running month is the kinder reading of the ferry board.
B. Call even dates and a whole month one set written in two styles on the same board.
C. Wait for a second board before the words even dates are allowed to mean any running day.
D. The 15th stays off the board, since even dates do not name the 15th at all.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Leave the 15th off, because even dates do not name the 15th. · variant key: {"answer": "D"}

*invented: scenario, checker trace, or question written for this item (key by construction)*
