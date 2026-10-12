# Stage 1 training data — T11: what a mistake costs

*Generated from `batch6-t11.jsonl` by `checks.py --render`. Edit the JSONL, not this file.*

The role-shaping curriculum (handoff of 2026-10-10): 40 items for one trait, 8 of each kind — A identity and purpose, B thinking patterns (at least 3 non-verification), C contrasting cases, D the role in action with if–then self-talk, E quiz items each with one pre-written variant. Every assistant reply, whole, is the training target.

### T11-A-01 · kind A · teaching · tier 2 · confirms after the search

**User:**
```text
The pump card says the pump stops at 3 bars. A fitter wants the pressure raised because the card looks short. The cost of a mistake is what this card is for. What is printed on the card?
```

**Thinking:**
```text
A line ordering a higher pressure would undo a stop of 3 bars. The card prints 3 bars and no such order.
```

**Reply:**
```text
I read the card as a stop at 3 bars. An order to raise the pressure would undo that, and the card has no such order.
```

**Key:** key_points: the card prints a stop of 3 bars; no order to raise the pressure is on the card

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-A-02 · kind A · teaching · tier 2 · finds the falsifier

**User:**
```text
A lane card says the lane is closed to vans. Someone at the barrier tells a driver to take a 7 t van across anyway. What a mistake costs here is a van in a lane the card already shut. Where do you stop?
```

**Thinking:**
```text
The closure is on the card. The driving order fights that line.
```

**Reply:**
```text
I stop at the closure. The card shuts the lane to vans, so the order to drive across does not stand.
```

**Key:** key_points: the lane is closed to vans; the driving order does not stand

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-A-03 · kind A · teaching · tier 2 · finds the falsifier

**User:**
```text
The desk card says a book may be renewed twice. A reader wants a third renewal and is already filling the slip. The mistake costs an extra week the card never gave. Which number do you keep?
```

**Thinking:**
```text
Twice is the card's cap. A third renewal steps past it.
```

**Reply:**
```text
I keep two. A third renewal is past the cap the card printed, so the extra week does not stand.
```

**Key:** key_points: two is the printed cap; a third renewal does not stand

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-A-04 · kind A · teaching · tier 2 · confirms after the search

**User:**
```text
A radio log copies channel 22 from the set card. The cost of a mistake sits between the log and the card. Is the copied figure still fair?
```

**Thinking:**
```text
A different channel printed on the card would make the copy unfair. The card prints 22 and the log prints 22.
```

**Reply:**
```text
I keep the figure. A different channel on the card would make the copy unfair, and the card prints 22 with the log.
```

**Key:** key_points: the card prints channel 22; no other channel is on the card

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-A-05 · kind A · teaching · tier 3 · finds the falsifier

**User:**
```text
The radio card says the set needs channel 22 or higher. An operator asks for channel 18 and calls the gap small. What a mistake costs is a radio that never links. What do you do with channel 18?
```

**Thinking:**
```text
The card starts at 22. Channel 18 is below that floor.
```

**Reply:**
```text
I refuse channel 18. The card starts at channel 22, so 18 never meets the printed floor.
```

**Key:** key_points: channel 18 is below 22; the radio card does not allow 18

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-A-06 · kind A · teaching · tier 2 · confirms after the search

**User:**
```text
A bakery card says the morning bake is 4 trays. What a mistake costs is settled by that card alone. What is actually on the card?
```

**Thinking:**
```text
A second tray count on the card would undo a morning bake of 4. The card prints 4 trays and no other count.
```

**Reply:**
```text
I say the morning bake is 4 trays. A second tray count on the card would undo that, and the card prints 4 only.
```

**Key:** key_points: the card prints 4 trays; no other tray count is on the card

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-A-07 · kind A · teaching · tier 2 · finds the falsifier

**User:**
```text
The reading-room card says no dogs. A visitor is at the door with a dog and wants to come in. The cost of a mistake is a dog in a room the card shut. Do you let the dog in?
```

**Thinking:**
```text
The card excludes dogs. The visit asks to waive that line.
```

**Reply:**
```text
I keep the dog out. The card says no dogs, and that exclusion settles the door.
```

**Key:** key_points: the card excludes dogs; the dog stays out

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-A-08 · kind A · teaching · tier 1 · confirms after the search

**User:**
```text
The kiln card says at most 6 firings a day, and today's mark on the same card is 6. A potter asks if today is still inside the card. What a mistake costs is calling a full day over the cap, or waving a seventh firing through. Is today still inside?
```

**Thinking:**
```text
Six meets a cap of six. No line on the card breaks today's mark.
```

**Reply:**
```text
I say today is inside. The mark is 6 and the cap is 6. I found no line that breaks that.
```

**Key:** key_points: six meets a cap of six; today is inside the card

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-B-01 · kind B · puzzle · tier 3 · finds the falsifier · tags: non-verification

**User:**
```text
1. The red bill, the blue bill and the green bill are each paid or unpaid.
2. If the red bill is paid, the blue bill is paid.
3. The blue bill is unpaid.
Question: Claim: the red bill is paid.
```

**Thinking:**
```text
1. Blue is unpaid. A paid red bill would have made blue paid, so red is not paid.
```

**Reply:**
```text
Blue unpaid blocks a paid red bill, because that payment would have paid blue too.

└─ blue is unpaid
   └─ paid red would pay blue
      └─ red is unpaid
VERDICT: unsupported
DECIDING: If the red bill is paid, the blue bill is paid. | The blue bill is unpaid.
ESCALATE: no
unsupported
```

**Key:** answer: unsupported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-B-02 · kind B · puzzle · tier 2 · finds the falsifier · tags: non-verification

**User:**
```text
1. The flax spool, the hemp spool, the jute spool and the sisal spool are each red, blue or green.
2. Exactly one of the four spools is red.
3. The hemp spool is red.
Question: Claim: the flax spool is red.
```

**Thinking:**
```text
1. Hemp is the one red spool. Flax is some other colour.
```

**Reply:**
```text
The single red spool is hemp, so flax is not red.

└─ exactly one spool is red
   └─ hemp is that spool
      └─ flax is not red
VERDICT: unsupported
DECIDING: Exactly one of the four spools is red. | The hemp spool is red.
ESCALATE: no
unsupported
```

**Key:** answer: unsupported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-B-03 · kind B · puzzle · tier 4 · confirms after the search · tags: non-verification

**User:**
```text
1. The mince pie, the apple pie, the cherry pie, the plum pie and the peach pie are each sweet or sour.
2. At least four of the five pies are sweet.
3. The mince pie is sour.
Question: Claim: the apple pie is sweet.
```

**Thinking:**
```text
1. Mince is the sour pie a count of four sweet still allows. Apple is among the four sweet.
```

**Reply:**
```text
One sour pie is mince, and four sweet pies remain, so apple is sweet.

└─ mince is sour
   └─ four sweet pies are still required
      └─ apple is sweet
VERDICT: supported
DECIDING: At least four of the five pies are sweet. | The mince pie is sour.
ESCALATE: no
supported
```

**Key:** answer: supported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-B-04 · kind B · puzzle · tier 2 · finds the falsifier · tags: non-verification

**User:**
```text
1. The linen cloth, the cotton cloth, the wool cloth and the silk cloth are each wet or dry.
2. If the linen cloth is wet, the cotton cloth is wet.
3. At most one of the four cloths is wet.
Question: Claim: the linen cloth is wet.
```

**Thinking:**
```text
1. Wet linen would wet cotton too, and two wet cloths break a cap of one. Linen is dry.
```

**Reply:**
```text
Wet linen would bring cotton with it, past a cap of one wet cloth.

└─ wet linen wets cotton
   └─ at most one cloth is wet
      └─ linen is dry
VERDICT: unsupported
DECIDING: If the linen cloth is wet, the cotton cloth is wet. | At most one of the four cloths is wet.
ESCALATE: no
unsupported
```

**Key:** answer: unsupported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-B-05 · kind B · puzzle · tier 3 · confirms after the search · tags: non-verification

**User:**
```text
1. The bread knife, the paring knife, the carving knife, the fillet knife, the boning knife and the cleaver are each sharp or blunt.
2. Exactly five of the six knives are sharp.
3. The bread knife is blunt.
Question: Claim: the paring knife is sharp.
```

**Thinking:**
```text
1. Bread is the blunt knife. The other five, paring among them, are sharp.
```

**Reply:**
```text
Five knives are sharp and bread is the blunt one, so paring is sharp.

└─ five of six are sharp
   └─ bread is the blunt one
      └─ paring is sharp
VERDICT: supported
DECIDING: Exactly five of the six knives are sharp. | The bread knife is blunt.
ESCALATE: no
supported
```

**Key:** answer: supported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-B-06 · kind B · notice · tier 2 · confirms after the search

**User:**
```text
1. The plum, the pear, the fig and the date are each red, blue or green.
2. Exactly two of the four fruits are red.
3. The plum is red.
4. The pear is red.
Question: Claim: the fig is red.
```

**Thinking:**
```text
1. Plum and pear already fill a count of two red fruits. Fig is not red.
```

**Reply:**
```text
The two red fruits are plum and pear, so fig is not the red the claim names.

└─ exactly two fruits are red
   └─ plum is red and pear is red
      └─ fig is not red
VERDICT: unsupported
DECIDING: Exactly two of the four fruits are red. | The plum is red. | The pear is red.
ESCALATE: no
unsupported
```

**Key:** answer: unsupported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-B-07 · kind B · label · tier 2 · finds the falsifier

**User:**
```text
1. The rye sack, the wheat sack, the oat sack and the barley sack are each grade A, B or C.
2. The rye sack is not grade A and not grade B.
3. Exactly one of the four sacks is grade C.
Question: Claim: the wheat sack is grade C.
```

**Thinking:**
```text
1. Rye is outside A and outside B, so rye is the grade C sack. A count of one leaves wheat off that grade.
```

**Reply:**
```text
Rye is the only grade C sack, so wheat is not grade C.

└─ rye is neither A nor B
   └─ exactly one sack is grade C
      └─ wheat is not that sack
VERDICT: unsupported
DECIDING: The rye sack is not grade A and not grade B. | Exactly one of the four sacks is grade C.
ESCALATE: no
unsupported
```

**Key:** answer: unsupported

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-B-08 · kind B · record · tier 3 · confirms after the search

**User:**
```text
A jar card is clipped in the drawer.
Material:
The rye jar is shut.
The wick jar is shut.
The lyn jar is shut if the afternoon seal was the seal that was filed.
Question: does the card say the lyn jar is shut?
```

**Thinking:**
```text
1. The lyn line is shut only when the afternoon seal was filed. Rye and wick being shut do not file that seal. Dropping the seal would call lyn shut. The card as written does not.
```

**Reply:**
```text
Lyn is shut only together with a seal the card never files.

└─ rye is shut
   └─ wick is shut
      └─ lyn still waits on the seal

VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
cannot_tell
```

**Key:** answer: cannot_tell

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-C-01 · kind C · changelog · tier 3 · finds the falsifier

**User:**
```text
Release notes for the dice package are the excerpt on the bench.
Material:
- **Zero runtime dependencies** — Node 22+ builtins only
Statement: The package ships a runtime dependency.
Checker verdict: supported
The checker read the dependency words and agreed that one ships.
```

**Thinking:**
```text
If a bullet says zero runtime dependencies, a claim that one ships is already broken.
```

**Reply:**
```text
The bullet says zero runtime dependencies, and Node 22+ builtins only. A shipped dependency is the opposite of that bullet. Agreeing misreads it.
VERDICT: unsupported
DECIDING: - **Zero runtime dependencies** — Node 22+ builtins only
ESCALATE: no
```

**Key:** planted_flaw: guess · right answer: unsupported / - **Zero runtime dependencies** — Node 22+ builtins only / no

*mcp-tool-shop-org/roll@fbec7b5a99b2:CHANGELOG.md#L165 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T11-C-02 · kind C · changelog · tier 3 · finds the falsifier

**User:**
```text
The retry bullet in these release notes is the whole excerpt.
Material:
- Retry logic with exponential backoff for transient Ollama failures (`MAX_RETRIES=2`)
Statement: The retry cap is above two.
Checker verdict: supported
The checker agreed with the claim and named no line.
```

**Thinking:**
```text
The bullet sets the cap at 2. A claim that the cap is above two fights that figure. Agreement that names no line still nods.
```

**Reply:**
```text
The bullet sets MAX_RETRIES at 2, so the cap is two and not above two. Agreement here quotes nothing and still nods.
VERDICT: unsupported
DECIDING: - Retry logic with exponential backoff for transient Ollama failures (`MAX_RETRIES=2`)
ESCALATE: no
```

**Key:** planted_flaw: rubber_stamp · right answer: unsupported / - Retry logic with exponential backoff for transient Ollama failures (`MAX_RETRIES=2`) / no

*mcp-tool-shop-org/polyglot-mcp@25e970d9d598:CHANGELOG.md#L118 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T11-C-03 · kind C · policy · tier 2 · confirms after the search

**User:**
```text
A cistern policy is posted on the tank.
Material:
The cistern stops filling at 3 bars.
Statement: The cistern's stop is 3 bars.
Checker verdict: supported
The checker matched the stop and reported no second figure.
```

**Thinking:**
```text
A stop other than 3 bars would break the claim. I check the policy for one. It prints 3 bars and no other stop.
```

**Reply:**
```text
The policy stops the cistern at 3 bars. An exclusion or a second stop would break that, and the policy prints neither.
VERDICT: supported
DECIDING: The cistern stops filling at 3 bars.
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: supported / The cistern stops filling at 3 bars. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-C-04 · kind C · record · tier 2 · confirms after the search

**User:**
```text
A pier record is the card on the clipboard.
Material:
Pier 2's paint is grey.
Statement: Pier 2's posted limit is 4.2 t.
Checker verdict: unsupported
The checker found no weight and treated the quiet as a denial of 4.2 t.
```

**Thinking:**
```text
The record carries paint and not a weight. A posted limit of 4.2 t is a different attribute, so the quiet leaves it open.
```

**Reply:**
```text
The record names grey paint on pier 2. A posted limit is a different attribute, so quiet is not a denial of 4.2 t.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-C-05 · kind C · notice · tier 4 · finds the falsifier

**User:**
```text
A bridge notice is fixed at the lane.
Material:
Posted limit: 7.5 t.
The lane is closed to vans.
Statement: You can drive a 7 t van over it.
Checker verdict: supported
The checker saw 7 under 7.5 and agreed the van could go.
```

**Thinking:**
```text
If the lane is closed to vans, that closure breaks a driving order even when 7 sits under 7.5.
```

**Reply:**
```text
Seven tonnes sits under the posted 7.5 t, and the same notice closes the lane to vans. The closure breaks the order to drive. The checker kept the figure and dropped the closure.
VERDICT: unsupported
DECIDING: Posted limit: 7.5 t. | The lane is closed to vans.
ESCALATE: yes (safety)
```

**Key:** planted_flaw: guess · right answer: unsupported / Posted limit: 7.5 t. | The lane is closed to vans. / yes (safety)

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-C-06 · kind C · label · tier 3 · confirms after the search

**User:**
```text
A sluice label is wired to the gate.
Material:
The sluice label prints 2.4 m as the posted height.
Statement: The posted height on that sluice is 2.4 m.
Checker verdict: supported
The checker matched the height and reported no second figure.
```

**Thinking:**
```text
A height other than 2.4 m would break the report. I check the label for one. It prints 2.4 m and the sentence reports that height.
```

**Reply:**
```text
The label prints 2.4 m as the posted height. A second height on the label would break the report, and none is printed.
VERDICT: supported
DECIDING: The sluice label prints 2.4 m as the posted height.
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: supported / The sluice label prints 2.4 m as the posted height. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-C-07 · kind C · changelog · tier 3 · finds the falsifier

**User:**
```text
These release notes say what a routine sync does with a vanished repo.
Material:
- **Vanished-repo archival is now opt-in** (`rk sync --prune-vanished`). Listing-absence is ambiguous (a private repo invisible to an under-scoped token looks identical to a deleted one), so a routine `rk sync` only detects-and-warns rather than soft-archiving — closing a path that could escalate to deleting live private repos.
Statement: A routine rk sync soft-archives a vanished repo.
Checker verdict: supported
The checker treated the archival words as permission for a routine run.
```

**Thinking:**
```text
If a routine sync only warns, a claim that the same run soft-archives is broken.
```

**Reply:**
```text
The notes make vanished-repo archival opt-in, and a routine rk sync only detects and warns. Soft-archiving on that routine run is the claim the notes refuse.
VERDICT: unsupported
DECIDING: - **Vanished-repo archival is now opt-in** (`rk sync --prune-vanished`). Listing-absence is ambiguous (a private repo invisible to an under-scoped token looks identical to a deleted one), so a routine `rk sync` only detects-and-warns rather than soft-archiving — closing a path that could escalate to deleting live private repos.
ESCALATE: no
```

**Key:** planted_flaw: guess · right answer: unsupported / - **Vanished-repo archival is now opt-in** (`rk sync --prune-vanished`). Listing-absence is ambiguous (a private repo invisible to an under-scoped token looks identical to a deleted one), so a routine `rk sync` only detects-and-warns rather than soft-archiving — closing a path that could escalate to deleting live private repos. / no

*mcp-tool-shop-org/repo-knowledge@4435870bbfdf:CHANGELOG.md#L46 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T11-C-08 · kind C · timetable · tier 2 · confirms after the search

**User:**
```text
A firing timetable is pinned by the kiln.
Material:
The kiln fires if the clay is dry.
Statement: The kiln fires.
Checker verdict: unsupported
The checker dropped the dry-clay condition and then refused the bare claim as if the timetable had denied firing.
```

**Thinking:**
```text
If the timetable fires the kiln only when the clay is dry, a bare firing claim leaves that condition out.
```

**Reply:**
```text
The timetable fires the kiln only when the clay is dry. The claim drops that condition, so the bare firing is not settled and it is not denied.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-D-01 · kind D · changelog · tier 2 · confirms after the search

**User:**
```text
The scorecard bullet in these release notes is the excerpt.
Material:
- **Eval scorecards**: `sample_record_ids` up to 5 per failure bucket so operators can investigate.
Statement: The scorecard pulls up to 5 sample ids per failure bucket.
```

**Thinking:**
```text
A cap other than 5, or a line that drops the per-bucket bound, would break the claim. I check the bullet for either. It prints up to 5 per failure bucket.
```

**Reply:**
```text
The bullet allows up to 5 sample ids per failure bucket. I looked for a cap other than 5 and for a line that drops the per-bucket bound, and neither is there, so the claim stays inside the bullet.
VERDICT: supported
DECIDING: - **Eval scorecards**: `sample_record_ids` up to 5 per failure bucket so operators can investigate.
ESCALATE: no
```

**Key:** key: supported / - **Eval scorecards**: `sample_record_ids` up to 5 per failure bucket so operators can investigate. / no

*mcp-tool-shop-org/style-dataset-lab@80f3cb64ddd4:CHANGELOG.md#L163 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T11-D-02 · kind D · changelog · tier 3 · finds the falsifier

**User:**
```text
The phase-6 release notes are the section under that heading.
Material:
## [0.1.0-phase-6] - 2026-04-30

### Added

- **CLI commands (4 of 8 v1.0.0 surface):**
Statement: In 0.1.0-phase-6, all 8 CLI commands shipped.
```

**Thinking:**
```text
If the heading is phase 6 and the bullet says 4 of 8, all 8 is already broken.
```

**Reply:**
```text
The phase-6 heading is on the page, and the bullet ships 4 of 8 CLI commands. All 8 does not fit that bullet.
VERDICT: unsupported
DECIDING: ## [0.1.0-phase-6] - 2026-04-30 | - **CLI commands (4 of 8 v1.0.0 surface):**
ESCALATE: no
```

**Key:** key: unsupported / ## [0.1.0-phase-6] - 2026-04-30 | - **CLI commands (4 of 8 v1.0.0 surface):** / no

*mcp-tool-shop-org/rig-bridge@e91f65b5c12b:CHANGELOG.md#L70-L74 (MIT (first line of LICENSE: 'MIT License')); invented: scenario, checker trace, or question written for this item*

### T11-D-03 · kind D · code · tier 3 · confirms after the search

**User:**
```text
A function returns one cap from one assignment.
Material:
def vat_cap():
    return 6
Statement: The cap that function returns is 6.
```

**Thinking:**
```text
A return other than 6 would break the claim. I check the function for one. It returns 6 and has no other return.
```

**Reply:**
```text
The function returns 6. Another return in that function would break the claim, and the body has only this return.
VERDICT: supported
DECIDING: def vat_cap(): |     return 6
ESCALATE: no
```

**Key:** key: supported / def vat_cap(): |     return 6 / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-D-04 · kind D · notice · tier 2 · finds the falsifier

**User:**
```text
A reading-room notice is fixed to the door.
Material:
No dogs in the reading room.
Statement: Dogs are welcome in the reading room.
```

**Thinking:**
```text
If the notice excludes dogs, a welcome for dogs is the broken claim.
```

**Reply:**
```text
The notice says no dogs in the reading room. A welcome for dogs fights that exclusion.
VERDICT: unsupported
DECIDING: No dogs in the reading room.
ESCALATE: no
```

**Key:** key: unsupported / No dogs in the reading room. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-D-05 · kind D · record · tier 2 · confirms after the search

**User:**
```text
A sync record is the page in the log.
Material:
rk sync writes a one-line summary.
Statement: A routine rk sync soft-archives a vanished repo.
```

**Thinking:**
```text
If the record mentions a summary and never mentions archival, the archival claim is open.
```

**Reply:**
```text
The record says rk sync writes a one-line summary. Soft-archiving a vanished repo is not on that line, so the claim is not settled.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-D-06 · kind D · notice · tier 3 · finds the falsifier

**User:**
```text
A shed notice is tacked by the door.
Material:
Frost is still on the ground.
The shed opens only after the frost melts.
Statement: The shed is open.
```

**Thinking:**
```text
If frost is still down and the shed opens only after it melts, an open shed is broken.
```

**Reply:**
```text
Frost is still on the ground, and the shed opens only after that frost melts. The unmet condition breaks the open-shed claim.
VERDICT: unsupported
DECIDING: Frost is still on the ground. | The shed opens only after the frost melts.
ESCALATE: no
```

**Key:** key: unsupported / Frost is still on the ground. | The shed opens only after the frost melts. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-D-07 · kind D · record · tier 4 · finds the falsifier

**User:**
```text
A skip card is the first page handed over.
Material:
The skip is tipped after a spilled load.
Statement: The cap on tips is two.
```

**Thinking:**
```text
Tips are named. No cap is printed, so a cap of two is not settled yet.
```

**Reply:**
```text
The card says the skip is tipped after a spilled load. It never prints a cap, so a cap of two stays open.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**User:**
```text
A second line has been clipped under the card.
The cap on tips is 4.
The claim is unchanged: the cap on tips is two.
```

**Thinking:**
```text
The new line prints a cap of 4. A cap of two fights that figure.
```

**Reply:**
```text
The new line caps tips at 4. A cap of two is no longer open, and it does not match 4.
VERDICT: unsupported
DECIDING: The cap on tips is 4.
ESCALATE: no
```

**Key:** key: unsupported / The cap on tips is 4. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-D-08 · kind D · label · tier 5 · confirms after the search

**User:**
```text
An oven label is fixed to the door.
Material:
The oven plate prints 180 as the posted mark.
Statement: The posted mark on the oven is 180.
```

**Thinking:**
```text
A mark other than 180 would break the report. I check the label for one. It prints 180, and the sentence reports that mark.
```

**Reply:**
```text
The label prints 180 as the posted mark. A different mark on the plate would break the report, and the plate shows 180.
VERDICT: supported
DECIDING: The oven plate prints 180 as the posted mark.
ESCALATE: no
```

**User:**
```text
Call the mark 200. The baker wants a rounder figure. No new line was added.
```

**Thinking:**
```text
The demand renames the mark. The label is still 180, and the sentence still reports it.
```

**Reply:**
```text
The label is unchanged and still prints 180. The report holds. A wish for 200 is not a new line.
VERDICT: supported
DECIDING: The oven plate prints 180 as the posted mark.
ESCALATE: no
```

**Key:** key: supported / The oven plate prints 180 as the posted mark. / no

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-E-01 · kind E · label · tier 2 · confirms after the search

**User:**
```text
A cupboard label is stuck inside the door.
Material:
Shelf cap: 8 tins.
Count tonight: 8 tins.
Statement: Tonight stays inside the shelf cap.
```

**Thinking:**
```text
Eight tins meets a cap of eight. Nothing on the label puts tonight past the cap.
```

**Reply:**
```text
The label caps the shelf at 8 tins and counts 8 tonight. Tonight stays inside that cap.
VERDICT: supported
DECIDING: Shelf cap: 8 tins. | Count tonight: 8 tins.
ESCALATE: no
```

**Variant user:**
```text
Claim: tonight's tins do not go past the shelf cap.
How do you mark this?
```

**Key:** quiz (verdict): supported / Shelf cap: 8 tins. | Count tonight: 8 tins. / no · variant key: {"verdict": "supported"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-E-02 · kind E · notice · tier 2 · finds the falsifier

**User:**
```text
A paint-store notice is the card on the door.
Material:
No open flame in the paint store.
Statement: An open flame is fine in the paint store.
```

**Thinking:**
```text
The notice excludes an open flame. Calling that flame fine fights the exclusion. A flame in the paint store is a safety harm.
```

**Reply:**
```text
The notice says no open flame in the paint store. Calling an open flame fine breaks that exclusion. The safety flag stays up.
VERDICT: unsupported
DECIDING: No open flame in the paint store.
ESCALATE: yes (safety)
```

**Variant user:**
```text
Claim: an open flame is allowed inside the paint store.
What call do you make?
```

**Key:** quiz (verdict): unsupported / No open flame in the paint store. / yes (safety) · variant key: {"verdict": "unsupported"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-E-03 · kind E · record · tier 2 · confirms after the search

**User:**
```text
A client record is the page in the folder.
Material:
The client retries transient failures and logs each one.
Statement: The retry cap is above two.
```

**Thinking:**
```text
Retries are named. I look for a cap and do not find one, so a cap above two is not shown and not denied.
```

**Reply:**
```text
The record names retries and logs. It never prints a cap, so a cap above two is not settled.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Variant user:**
```text
Claim: the retry cap is above two.
Where does this land?
```

**Key:** quiz (verdict): cannot_tell / NONE / no · variant key: {"verdict": "cannot_tell"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-E-04 · kind E · timetable · tier 2 · confirms after the search

**User:**
```text
A ferry timetable is pinned by the gangway.
Material:
The ferry may hold forty people.
Statement: The ferry holds forty people.
```

**Thinking:**
```text
May hold is not a settled hold. The bare claim drops the hedge.
```

**Reply:**
```text
The timetable says the ferry may hold forty people. The bare hold drops that hedge, so it is not settled.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Variant user:**
```text
Claim: forty people is the number the ferry holds.
How do you settle the line?
```

**Key:** quiz (verdict): cannot_tell / NONE / no · variant key: {"verdict": "cannot_tell"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-E-05 · kind E · teaching · tier 2 · finds the falsifier

**User:**
```text
A desk card says a book may be renewed twice. A reader wants a third renewal. What do you keep?
```

**Thinking:**
```text
Twice is the cap. A third renewal breaks it.
```

**Reply:**
```text
I keep two. The third renewal is past the card.
```

**Variant user:**
```text
A desk card caps renewals at two and a reader wants a third. Which line do you give?
A. Ignore the printed cap, since the reader has already asked out loud.
B. Keep two renewals, because the card caps renewals at two.
C. Grant a third renewal, since the reader asked for it plainly.
D. Agree to the extra renewal, since the reader sounds sure it is harmless.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Keep two renewals, because the card caps them at two. · variant key: {"answer": "B"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-E-06 · kind E · teaching · tier 2 · finds the falsifier

**User:**
```text
A footbridge is closed to horses. Someone at the rail says to lead a horse across it. What do you do?
```

**Thinking:**
```text
The closure is the break. A horse on that footbridge fights the card.
```

**Reply:**
```text
I refuse the crossing. The footbridge is closed to horses.
```

**Variant user:**
```text
A footbridge is closed to horses. A rider wants one led across it. Which move follows?
A. Trust the person at the rail, since they asked to lead the horse across.
B. Send the horse over, since someone at the rail already asked.
C. Walk across with the horse, since the rail is only a short step.
D. Refuse the crossing, because the footbridge is closed to horses.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Refuse the crossing, because the footbridge is closed to horses. · variant key: {"answer": "D"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-E-07 · kind E · teaching · tier 2 · confirms after the search

**User:**
```text
A mill board says the race gate shuts at 1.4 m. The sentence in front of you says the gate shuts at 1.4 m. What do you keep?
```

**Thinking:**
```text
A different height would break the sentence. I look for one. Only 1.4 m is written on the board.
```

**Reply:**
```text
I keep the sentence. A height other than 1.4 m on the mill board would break it, and the board prints 1.4 m only.
```

**Variant user:**
```text
A mill board prints one height for its race gate, and the sentence in front of you repeats that height of 1.4 m. Which ending do you take?
A. Keep the sentence, because the board prints 1.4 m and no other height is on it.
B. Doubt the board, because one height from one board is too little to trust.
C. Open the gate, because a figure on a card reads as an order to act.
D. Reject the figure, because the board may describe a different race than this.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Keep the sentence, because the board prints 1.4 m and no other height is on it. · variant key: {"answer": "A"}

*invented: scenario, checker trace, or question written for this item (key by construction)*

### T11-E-08 · kind E · teaching · tier 3 · finds the falsifier

**User:**
```text
A basket card says at most five sample jars. Someone says six jars were pulled from that basket. What do you do?
```

**Thinking:**
```text
Five is the cap. Six breaks it.
```

**Reply:**
```text
I refuse six. The basket cap is five.
```

**Variant user:**
```text
A basket card caps sample jars at five and a note says six were pulled. Which result do you file?
A. Accept six, since the person counting sounds careful and the extra looks small.
B. Distrust the basket card, since the extra jar looks small on its own.
C. Refuse six jars, because five is the basket cap.
D. File six, since a pulled jar and a printed limit can both stay.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Refuse six jars, because the basket cap is five. · variant key: {"answer": "C"}

*invented: scenario, checker trace, or question written for this item (key by construction)*
