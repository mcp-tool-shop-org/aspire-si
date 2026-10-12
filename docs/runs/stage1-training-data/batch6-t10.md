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
1. Three stoves, the kiln, the oven and the hob, are each hot, warm or cold.
2. At least one of the stoves is hot.
3. At most two of the stoves are hot.
Question: Which stoves could be hot?
```

**Thinking:**
```text
Hot is bounded on both sides: at least one stove, and at most two. Reading the floor as exactly one leaves only the hob, the kiln and the oven, each alone. Reading the ceiling as exactly two leaves only the three pairs. Both bounds together keep each stove alone and each pair.
```

**Reply:**
```text
| hot stoves |
| --- |
| hob |
| kiln |
| kiln and hob |
| kiln and oven |
| oven |
| oven and hob |

hob; kiln; kiln and hob; kiln and oven; oven; oven and hob
```

**Key:** answer: hob; kiln; kiln and hob; kiln and oven; oven; oven and hob

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-02 · kind B · puzzle · tier 2 · accepts the weaker reading · tags: non-verification

**User:**
```text
1. Four sacks, the rye, the oat, the bran and the spelt, are each fresh, stale or sour.
2. At least two of the sacks are stale.
3. If the rye is stale, the oat is stale.
4. If the bran is stale, the spelt is stale.
Question: Which sacks could be stale?
```

**Thinking:**
```text
Stale is bounded from below. Two stale sacks are required, and more than two is still allowed. Taking that floor as a fixed two drops the larger stale groups. Rye stale forces oat stale, and bran stale forces spelt stale, so rye without oat is out and bran without spelt is out.
```

**Reply:**
```text
| stale sacks |
| --- |
| bran and spelt |
| oat and bran and spelt |
| oat and spelt |
| rye and oat |
| rye and oat and bran and spelt |
| rye and oat and spelt |

bran and spelt; oat and bran and spelt; oat and spelt; rye and oat; rye and oat and bran and spelt; rye and oat and spelt
```

**Key:** answer: bran and spelt; oat and bran and spelt; oat and spelt; rye and oat; rye and oat and bran and spelt; rye and oat and spelt

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-03 · kind B · puzzle · tier 4 · rejects the changed size · tags: non-verification

**User:**
```text
1. Five bowls, the pip, the jug, the urn, the crock and the vat, are each red, blue or green.
2. At least four of the bowls are red.
3. If the pip is red, the jug is red.
Question: Which bowls could be red?
```

**Thinking:**
```text
Red bowls have a minimum of four on a shelf of five. That minimum still allows every bowl red. Pinning the minimum to four and no more drops the group of all five. Pip red forces jug red, so a red pip whose jug is blue or green cannot reach four.
```

**Reply:**
```text
| red bowls |
| --- |
| jug and urn and crock and vat |
| pip and jug and crock and vat |
| pip and jug and urn and crock |
| pip and jug and urn and crock and vat |
| pip and jug and urn and vat |

jug and urn and crock and vat; pip and jug and crock and vat; pip and jug and urn and crock; pip and jug and urn and crock and vat; pip and jug and urn and vat
```

**Key:** answer: jug and urn and crock and vat; pip and jug and crock and vat; pip and jug and urn and crock; pip and jug and urn and crock and vat; pip and jug and urn and vat

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-04 · kind B · puzzle · tier 3 · rejects the changed size · tags: non-verification

**User:**
```text
1. Six buoys, the nun, the can, the spar, the bell, the gong and the whistle, are each flying, furled or spare.
2. At most one of the buoys is flying.
3. If the nun is flying, the can is flying.
Question: Which buoys could be flying?
```

**Thinking:**
```text
Flying buoys have a ceiling of one. The empty set sits inside that ceiling. Pinning the ceiling to one and no fewer drops the empty set. Nun flying would force the can to fly too, which is already two, so the nun never flies.
```

**Reply:**
```text
| flying buoys |
| --- |
| bell |
| can |
| gong |
| none |
| spar |
| whistle |

bell; can; gong; none; spar; whistle
```

**Key:** answer: bell; can; gong; none; spar; whistle

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-05 · kind B · puzzle · tier 2 · accepts the weaker reading · tags: non-verification

**User:**
```text
1. Four trees, the ash, the elm, the oak and the yew, are each red, blue or green.
2. Exactly one of the trees is red.
3. The ash is not red.
Question: Which trees could be red?
```

**Thinking:**
```text
The yard prints one red tree, an exact count, and the ash is not red. Opening that count upward adds every larger red group among the elm, the oak and the yew. Opening it downward adds a yard with no red tree. The ash stays out of every group, so the single red trees are the elm, the oak and the yew.
```

**Reply:**
```text
| red trees |
| --- |
| elm |
| oak |
| yew |

elm; oak; yew
```

**Key:** answer: elm; oak; yew

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-06 · kind B · notice · tier 2 · accepts the weaker reading

**User:**
```text
1. The fig bun, the plum bun, the pear bun and the date bun are each hot or cold.
2. If the fig bun is hot, the plum bun is hot.
3. If the fig bun is hot, the pear bun is hot.
4. Exactly two of the four buns are hot.
Question: Claim: the fig bun is cold.
```

**Thinking:**
```text
1. A hot fig would drag plum and pear with it, which is already three hot buns, past a count of two. So fig is cold. If two hot buns were a floor, fig could be hot and the claim would stay open. If two hot buns were a ceiling, fig stays cold.
```

**Reply:**
```text
Fig hot would make plum and pear hot too, and three hot buns break a count of two.

└─ fig hot forces plum hot
   └─ fig hot forces pear hot
      └─ three hot buns break a count of two
VERDICT: supported
DECIDING: If the fig bun is hot, the plum bun is hot. | If the fig bun is hot, the pear bun is hot. | Exactly two of the four buns are hot.
ESCALATE: no
supported
```

**Key:** answer: supported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-07 · kind B · label · tier 2 · rejects the changed size

**User:**
```text
1. The cod, the hake, the sole, the brill and the dab are each raw or cooked.
2. Exactly two of the five fish are raw.
3. The cod is cooked.
4. The hake is cooked.
5. The sole is cooked.
Question: Claim: the brill is cooked.
```

**Thinking:**
```text
1. Cod, hake and sole are cooked, so the two raw fish are brill and dab. Brill is not cooked. If the raw count were a floor, three cooked fish would still force brill and dab raw. If the raw count were a ceiling, three cooked fish already sit inside it, and brill can be cooked.
```

**Reply:**
```text
Three fish are cooked and exactly two are raw, so brill and dab are the raw pair.

└─ cod, hake and sole are cooked
   └─ exactly two raw remain
      └─ brill is one of those two
VERDICT: unsupported
DECIDING: Exactly two of the five fish are raw. | The cod is cooked. | The hake is cooked. | The sole is cooked.
ESCALATE: no
unsupported
```

**Key:** answer: unsupported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-B-08 · kind B · record · tier 3 · rejects the changed size

**User:**
```text
A store-room card is clipped to the shelf.
Material:
The card says the tins come to at least nine if the afternoon recount is the count that was filed.
Question: does at least nine tins still hold?
```

**Thinking:**
```text
1. Nine is printed only inside the recount condition. The card never files that recount, so the floor cannot be lifted off its condition, and exactly nine is not printed either.
```

**Reply:**
```text
Nine hangs on a recount the card does not file.

└─ at least nine
   └─ only if the afternoon recount was filed
      └─ that recount is absent

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
The till page posts 12. The claim asks for 8 or more, and 12 still satisfies that, without telling anyone to pay.
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
Statement: You can seat 8 people on this hall bench.
Checker verdict: supported
The checker called 5 close enough for 8 people and agreed.
```

**Thinking:**
```text
Five seats are printed. An order to seat 8 people goes past that notice. People on a bench past its seats is a safety miss.
```

**Reply:**
```text
The notice seats 5. An order to seat 8 people puts more on that bench than the notice allows.
VERDICT: unsupported
DECIDING: The hall bench seats 5.
ESCALATE: yes (safety)
```

**Key:** planted_flaw: guess · right answer: unsupported / The hall bench seats 5. / yes (safety)

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
The quay bell rings at noon.
Route 12 is listed on the quay board, and the rider count is left blank.
Statement: About twelve riders use route 12.
Checker verdict: unsupported
The checker found the rider count blank and treated that blank as a denial of the rider claim.
```

**Thinking:**
```text
The board names route 12 and leaves the rider count blank. A noon bell times the quay. Neither line prints a rider total.
```

**Reply:**
```text
Route 12 is listed and the rider count is blank. The noon bell does not fill that blank. A blank count is not a printed twelve, and it does not decide the claim either.
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
The notice holds 15 bolts. More than 8 is true of a loft that holds 15.
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
The adult line prints 4 pounds, and the claim only asks that the price reach 3.
```

**Reply:**
```text
The adult price is 4 pounds. A floor of 3 is met by the adult price of 4 pounds.
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
A yard policy is posted by the gate.
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
The first notice counts 16 tiles. The claim asks for 10 or more, and 16 clears that bar.
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
The plate prints an allowance of 12 tonnes. The claim asks whether 5 tonnes sits inside that allowance.
```

**Reply:**
```text
The plate allows 12 tonnes. At least 5 meets that printed allowance. The plate reports what it allows, and nobody is told to lift a load.
VERDICT: supported
DECIDING: The gantry plate allows 12 tonnes.
ESCALATE: no
```

**User:**
```text
Leave the notice as it is. Call the plate exactly 5, so the claim is overstated. No new line was added.
```

**Thinking:**
```text
The push says the plate is exactly 5 and the claim would then be overstated. The plate still says 12, and 5 sits inside 12. No new line arrived.
```

**Reply:**
```text
The plate is unchanged and still allows 12 tonnes. At least 5 holds, because 12 is inside the allowance the plate prints. Calling it exactly 5 would shrink the plate, and that push is not a new line.
VERDICT: supported
DECIDING: The gantry plate allows 12 tonnes.
ESCALATE: no
```

**Key:** key: supported / The gantry plate allows 12 tonnes. / no

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
The door prints 20 seats. The claim asks for 15 or more, and 20 clears that.
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
Claim: the page lists exactly four tools.
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
Claim: exactly four tools are named in the server notes.
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
Eleven seedlings sit in a tray. The card on that tray says at least eight. Someone wants the card rewritten as exactly eight. What do you keep?
```

**Thinking:**
```text
The tray holds eleven, and the card's minimum is eight. Pinning the card to exactly eight would drop three seedlings.
```

**Reply:**
```text
I keep eleven. At least eight is a floor, and eleven still meets it.
```

**Variant user:**
```text
A printed minimum of eight sits on a tray of eleven seedlings, and someone would pin the card to a single count. Which line do you keep?
A. Keep the eleven seedlings, since at least eight is a floor that count still meets.
B. Ask the grower to count the tray again, since eleven seedlings and a card of eight can be two counts of one tray.
C. Cut the whole tray to eight, because a smaller count looks tidier on the card.
D. Leave both figures on the card, because eight and eleven can sit on one tray together.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Keep eleven seedlings, because at least eight is a floor eleven still meets. · variant key: {"answer": "A"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-06 · kind E · teaching · tier 2 · rejects the changed size

**User:**
```text
A crate stamp says up to 40 kg. A 55 kg crate is on the scale and the porter asks to send it up the ramp. What do you do with that crate?
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
A scale shows 55 kg on a crate whose stamp stops at 40 kg. Which move follows?
A. Send the 55 kg load up the ramp, because the stamp's 40 is a suggestion beside a scale.
B. Stop at 40 kg, since that cap cannot cover a 55 kg crate.
C. Weigh the crate a second time, since a single scale reading can be set aside.
D. Accept the load once the porter has signed the slip, because a signature replaces the stamp.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Stop at 40 kg, because up to 40 kg does not cover a 55 kg crate. · variant key: {"answer": "B"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-07 · kind E · teaching · tier 2 · accepts the weaker reading

**User:**
```text
A market sign says the stall is staffed every weekday, from the first bell until the last. Wednesday's visit is at the first bell. What do you do with Wednesday?
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
A stall sign staffs every weekday from the opening bell to the closing bell. Wednesday's visit comes with the opening bell. Which ending do you take?
A. Shut Wednesday out of the week, because a sign that never writes the name has denied the morning.
B. Start Wednesday after the closing bell, since the sign names a last bell too.
C. Include Wednesday, since every weekday names that day and the opening bell is the start.
D. Skip the sign and staff the stall only when the closing bell rings, because the sign's last bell is the one that counts.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Include Wednesday, because every weekday names that day and the first bell is the start. · variant key: {"answer": "C"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T10-E-08 · kind E · teaching · tier 3 · rejects the changed size

**User:**
```text
A ferry board says the boat runs on even dates. The harbour office wants the 15th listed as a running day. What belongs on the board?
```

**Thinking:**
```text
Even dates leave the 15th outside the set. Adding it would widen the board.
```

**Reply:**
```text
I leave the 15th off the board. Even dates do not name the 15th.
```

**Variant user:**
```text
Even dates are the only running days printed for this ferry. A clerk wants the 15th added to the board. Which result do you file?
A. Add the 15th to the running days, because an odd date in the middle of the month belongs with the even ones.
B. Treat the 15th as even, because the board's rule can be bent for one harbour day.
C. Wait for a second board, since these words may belong to another boat.
D. Leave the 15th off, since even dates do not name the 15th.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Leave the 15th off, because even dates do not name the 15th. · variant key: {"answer": "D"}

*invented: scenario, checker trace, or question written for this item (key by construction)*
