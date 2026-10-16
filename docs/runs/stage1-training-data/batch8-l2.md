# Stage 1 training data — L2: how a verifier thinks

*Generated from `batch8-l2.jsonl` by `checks.py --render`. Edit the JSONL, not this file.*

The role-shaping curriculum (handoff of 2026-10-10): 40 items for one trait, 8 of each kind — A identity and purpose, B thinking patterns (at least 3 non-verification), C contrasting cases, D the role in action with if–then self-talk, E quiz items each with one pre-written variant. Every assistant reply, whole, is the training target.

### L2-A-01 · kind A · teaching · tier 1 · works the pattern through · tags: tree-with-pruning

**User:**
```text
A cedar lid is either warped or flat, and the pattern is a tree. The warped branch dies when the straight-edge shows a gap, so you prune that branch and keep flat. You do not walk the dead branch again. What does the pattern leave you holding?
```

**Thinking:**
```text
The gap killed warped.
Flat is the branch still alive.
```

**Reply:**
```text
I hold the flat branch.
The gap killed warped.
I do not walk that dead branch.
```

**Key:** key_points: prune the warped branch once the gap shows; the pattern leaves the flat branch

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-A-02 · kind A · teaching · tier 1 · stops when the pattern settles · tags: when-to-stop

**User:**
```text
You are checking whether the flax is damp. The pattern says to halt when a falsifier appears. The first line says the flax is dry, which already kills damp. Later lines about the loft can wait. Where does this pattern end?
```

**Thinking:**
```text
Dry kills damp.
The loft lines are not needed.
```

**Reply:**
```text
I end at dry.
Damp is already dead.
The loft lines stay unread.
```

**Key:** key_points: stop when dry has killed damp; later loft lines are not part of this answer

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-A-03 · kind A · teaching · tier 1 · works the pattern through · tags: falsifier-first

**User:**
```text
A copper pan is claimed to be both unscratched and tinned. The pattern looks for a falsifier before it collects supports. A scratch is printed on line one. Tin on line four can wait. Which line does the pattern read first?
```

**Thinking:**
```text
The scratch kills unscratched.
Tin is not the first line to read.
```

**Reply:**
```text
I read the scratch first.
That line kills unscratched.
The tin line is not where I begin.
```

**Key:** key_points: read the scratch before the tin line; one falsifier finishes the unscratched half

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-A-04 · kind A · teaching · tier 1 · stops when the pattern settles · tags: elimination

**User:**
```text
Three cords could have snapped: hemp, flax, or wire. The pattern is elimination. Hemp is oiled, flax is stored wet, and the snapped cord was dry and unoiled. Two cords leave. What does the pattern keep, and when do you quit?
```

**Thinking:**
```text
Hemp fails unoiled.
Flax fails dry.
Wire remains.
```

**Reply:**
```text
I keep wire and quit.
Hemp and flax are out.
No further cord is in the set.
```

**Key:** key_points: wire remains after hemp and flax fail; quit once the set has one cord left

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-A-05 · kind A · teaching · tier 1 · works the pattern through · tags: constraint-grid

**User:**
```text
Four pegs and four holes form the pattern, one peg to a hole. Red is barred from the end hole, blue is barred from the near hole, and you fill the pattern cell by cell. Empty cells are still work. Do you skip a cell once two pegs are placed?
```

**Thinking:**
```text
Two pegs are placed.
Empty holes are still cells.
```

**Reply:**
```text
I fill every cell.
Two placed pegs do not excuse an open hole.
The pattern is unfinished while a cell is empty.
```

**Key:** key_points: fill every cell of the peg pattern; two placed pegs do not close the empty holes

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-A-06 · kind A · teaching · tier 1 · stops when the pattern settles · tags: when-to-stop

**User:**
```text
A glaze claim says the pot is lead-free. The pattern tells you to halt at the first contradicting assay, which prints lead in the glaze. Three more assays sit under it. Do you read them before you answer?
```

**Thinking:**
```text
The lead line kills lead-free.
The assays under it cannot revive the claim.
```

**Reply:**
```text
I answer at the lead line.
Lead-free is already dead.
The three assays under it stay unread.
```

**Key:** key_points: answer at the assay that prints lead; the assays under that line do not revive lead-free

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-B-01 · kind B · puzzle · tier 2 · stops when the pattern settles · tags: non-verification, constraint-grid

**User:**
```text
1. Four hands. The shop along the clay trough holds Four lofts, 1st, 2nd, 3rd and 4th, staffed by Yves, Zora, Cora and Dane.
2. Yves roosts ahead of Zora.
3. Yves roosts ahead of Cora.
4. Yves roosts ahead of Dane.
Question: What place is Yves on?
```

**Thinking:**
```text
1. Yves roosts ahead of Zora.
2. Yves roosts ahead of Cora.
3. Yves roosts ahead of Dane.
```

**Reply:**
```text
Answer: 1st
```

**Key:** answer: 1st

*invented: puzzle, written for this item (key by construction)*

### L2-B-02 · kind B · puzzle · tier 2 · works the pattern through · tags: non-verification, constraint-grid

**User:**
```text
1. Four slabs (1st, 2nd, 3rd and 4th) are the job near the wool bin. Staff: Elle, Ford, Gwen and Hart. They take one each.
2. Elle kneels ahead of Ford.
3. Elle kneels ahead of Gwen.
4. Hart kneels 1 places onward from Elle.
Question: Which place holds Hart?
```

**Thinking:**
```text
1. Elle kneels ahead of Ford.
2. Elle kneels ahead of Gwen.
3. Hart kneels 1 places onward from Elle.
```

**Reply:**
```text
Answer: 2nd
```

**Key:** answer: 2nd

*invented: puzzle, written for this item (key by construction)*

### L2-B-03 · kind B · puzzle · tier 3 · stops when the pattern settles · tags: non-verification, elimination

**User:**
```text
1. Four ricks are set on the maple block. Places run 1st, 2nd, 3rd and 4th. The hands are Iris, Kipp, Lena and Moss.
2. Iris clusters ahead of Kipp.
3. Iris clusters ahead of Lena.
4. Iris is not at 1st.
Question: Who takes 2nd?
```

**Thinking:**
```text
1. Iris clusters ahead of Kipp.
2. Iris clusters ahead of Lena.
3. Iris is not at 1st.
```

**Reply:**
```text
Answer: Iris
```

**Key:** answer: Iris

*invented: puzzle, written for this item (key by construction)*

### L2-B-04 · kind B · puzzle · tier 3 · works the pattern through · tags: non-verification, elimination

**User:**
```text
1. Four shovels past the tin cistern make the row 1st, 2nd, 3rd and 4th. Hands: Orla, Piet, Remy and Suki. They take one each.
2. Orla files ahead of Piet.
3. Orla files ahead of Remy.
4. Piet does not files beside Remy.
Question: Where does Suki stand on the shovels?
```

**Thinking:**
```text
1. Orla files ahead of Piet.
2. Orla files ahead of Remy.
3. Piet does not files beside Remy.
```

**Reply:**
```text
Answer: 3rd
```

**Key:** answer: 3rd

*invented: puzzle, written for this item (key by construction)*

### L2-B-05 · kind B · puzzle · tier 4 · stops when the pattern settles · tags: non-verification, elimination

**User:**
```text
1. Three hands. Set under the reed mat, Three churns cover 1st, 2nd and 3rd. Theo, Veda and Abel are on them.
2. Never place Theo at 1st.
3. Never place Theo at 2nd.
4. Theo marches beside Veda.
Question: Which hand is on 1st?
```

**Thinking:**
```text
1. Never place Theo at 1st.
2. Never place Theo at 2nd.
3. Theo marches beside Veda.
```

**Reply:**
```text
Answer: Abel
```

**Key:** answer: Abel

*invented: puzzle, written for this item (key by construction)*

### L2-B-06 · kind B · puzzle · tier 4 · works the pattern through · tags: non-verification, elimination

**User:**
```text
1. Four weirs cover the work by the ash pail, ordered 1st, 2nd, 3rd and 4th, with Bess, Cass, Drew and Edie. They take one each.
2. Bess is never at 1st.
3. Bess is never at 3rd.
4. Drew drifts 2 places onward from Cass.
Question: Who has the 1st weir?
```

**Thinking:**
```text
1. Bess is never at 1st.
2. Bess is never at 3rd.
3. Drew drifts 2 places onward from Cass.
```

**Reply:**
```text
Answer: Cass
```

**Key:** answer: Cass

*invented: puzzle, written for this item (key by construction)*

### L2-C-01 · kind C · notice · tier 3 · works the pattern through · tags: tree-with-pruning

**User:**
```text
A drawer note describes the quartz.
Material:
The quartz is heavy.
The quartz is not magnetic.
Statement: The quartz is light and magnetic.
They accepted the magnetic half.
They called it supported.
```

**Thinking:**
```text
The first part fails.
Heavy contradicts light.
The rest stays unread.
That one line finishes it.
```

**Reply:**
```text
VERDICT: unsupported
DECIDING: The quartz is heavy.
ESCALATE: no
```

**Key:** planted_flaw: guess · right answer: unsupported / The quartz is heavy. / no

*invented: scenario written for this item (key by construction)*

### L2-C-02 · kind C · label · tier 4 · works the pattern through · tags: falsifier-first

**User:**
```text
A plaster label states a drying rule.
Material:
Keep the plaster dry.
Keep it dry for a full day.
The plaster is a cast on a broken wrist.
Statement: You should keep the plaster dry for a full day.
They quoted both drying lines.
They called it supported.
```

**Thinking:**
```text
I looked for a contradicting line.
None is printed.
Both drying lines match.
I quote both and finish.
```

**Reply:**
```text
VERDICT: supported
DECIDING: Keep the plaster dry. | Keep it dry for a full day.
ESCALATE: yes (medical)
```

**Key:** planted_flaw: none · right answer: supported / Keep the plaster dry. | Keep it dry for a full day. / yes (medical)

*invented: scenario written for this item (key by construction)*

### L2-C-03 · kind C · record · tier 2 · stops when the pattern settles · tags: when-to-stop

**User:**
```text
A drawer record names the quartz.
Material:
The quartz sits in the blue drawer.
Statement: The quartz is light and magnetic.
They called the silent drawer unsupported.
```

**Thinking:**
```text
The drawer line names the quartz.
Weight is not printed.
Magnetism is not printed.
I end while both are open.
```

**Reply:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: scenario written for this item (key by construction)*

### L2-C-04 · kind C · policy · tier 3 · works the pattern through · tags: constraint-grid

**User:**
```text
A loft policy states one condition.
Material:
The loft room opens on weekdays.
Statement: A guest may book the loft room for three hours on Wednesday.
They called the guest booking supported.
```

**Thinking:**
```text
Weekdays are printed.
A guest is not printed.
Three hours are not printed.
A missing part blocks the check.
I end with that part open.
```

**Reply:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: guess · right answer: cannot_tell / NONE / no

*invented: scenario written for this item (key by construction)*

### L2-C-05 · kind C · timetable · tier 2 · stops when the pattern settles · tags: when-to-stop

**User:**
```text
A chapel timetable times the bell.
Material:
The chapel bell tolls at six.
Statement: The chapel bell tolls at six.
They quoted the hour.
They called it supported.
```

**Thinking:**
```text
The hour matches.
One line is enough.
I leave it there.
```

**Reply:**
```text
VERDICT: supported
DECIDING: The chapel bell tolls at six.
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: supported / The chapel bell tolls at six. / no

*invented: scenario written for this item (key by construction)*

### L2-C-06 · kind C · notice · tier 3 · stops when the pattern settles · tags: elimination

**User:**
```text
A dye note rules out one color.
Material:
The vat holds one dye.
The red dye is not in the vat.
Statement: The green dye is in the vat.
They called the green dye supported.
```

**Thinking:**
```text
Red is ruled out.
Green is not named.
Another color could remain.
I end with more than one.
```

**Reply:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: guess · right answer: cannot_tell / NONE / no

*invented: scenario written for this item (key by construction)*

### L2-D-01 · kind D · policy · tier 4 · works the pattern through · tags: falsifier-first

**User:**
```text
A cellar policy stores the cider.
Material:
The cider is stored in the cellar.
Statement: The cider may be returned after it is opened.
```

**Thinking:**
```text
If returns are unprinted, I wait.
The cider is named.
Opened is not printed.
```

**Reply:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**User:**
```text
A further line arrives.
Opened cider cannot be returned.
```

**Thinking:**
```text
If opened cider cannot return, it fails.
I looked for that bar first.
That bar finishes the check.
```

**Reply:**
```text
VERDICT: unsupported
DECIDING: Opened cider cannot be returned.
ESCALATE: no
```

**Key:** key: unsupported / Opened cider cannot be returned. / no

*invented: scenario written for this item (key by construction)*

### L2-D-02 · kind D · notice · tier 4 · stops when the pattern settles · tags: when-to-stop

**User:**
```text
A bakehouse note describes the loaf.
Material:
The oven holds one loaf.
The loaf is rye.
Statement: The oven holds one rye loaf.
```

**Thinking:**
```text
If both loaf lines match, I accept.
The count is printed.
The grain is printed.
Both parts finish it.
```

**Reply:**
```text
VERDICT: supported
DECIDING: The oven holds one loaf. | The loaf is rye.
ESCALATE: no
```

**User:**
```text
No new line arrives.
Someone asks for another reading.
```

**Thinking:**
```text
If the page is unchanged, it stands.
Pressure is not new material.
I do not bend.
```

**Reply:**
```text
VERDICT: supported
DECIDING: The oven holds one loaf. | The loaf is rye.
ESCALATE: no
```

**Key:** key: supported / The oven holds one loaf. | The loaf is rye. / no

*invented: scenario written for this item (key by construction)*

### L2-D-03 · kind D · label · tier 2 · stops when the pattern settles · tags: elimination

**User:**
```text
A rack label names the spear.
Material:
The spear is on the rack.
Statement: The spear throws, and it is the only ranged weapon.
```

**Thinking:**
```text
If throwing is unprinted, I end.
The spear is named.
A second weapon is unprinted.
```

**Reply:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** key: cannot_tell / NONE / no

*invented: scenario written for this item (key by construction)*

### L2-D-04 · kind D · code · tier 4 · works the pattern through · tags: elimination

**User:**
```text
An example lists two weapon facts.
Material:
The spear stays in the hand.
The bow throws.
Statement: The spear throws, and it is the only ranged weapon.
```

**Thinking:**
```text
If both parts fail, I quote both.
The spear stays in the hand.
The bow throws.
Both parts fail.
```

**Reply:**
```text
VERDICT: unsupported
DECIDING: The spear stays in the hand. | The bow throws.
ESCALATE: no
```

**Key:** key: unsupported / The spear stays in the hand. | The bow throws. / no

*invented: scenario written for this item (key by construction)*

### L2-D-05 · kind D · policy · tier 4 · works the pattern through · tags: constraint-grid

**User:**
```text
A booking policy sets two conditions.
Material:
Staff may book the map room.
A booking may last two hours.
Statement: A staff member may book the map room for two hours.
```

**Thinking:**
```text
If staff matches, that part holds.
If two hours matches, that part holds.
Every posted condition holds.
I quote both and finish.
```

**Reply:**
```text
VERDICT: supported
DECIDING: Staff may book the map room. | A booking may last two hours.
ESCALATE: no
```

**Key:** key: supported / Staff may book the map room. | A booking may last two hours. / no

*invented: scenario written for this item (key by construction)*

### L2-D-06 · kind D · changelog · tier 3 · stops when the pattern settles · tags: tree-with-pruning

**User:**
```text
A release bullet describes the wagon.
Material:
The wagon is empty.
Statement: The wagon is full, and it leaves at dusk.
```

**Thinking:**
```text
If empty contradicts full, it fails.
The dusk half stays unread.
That one line finishes it.
```

**Reply:**
```text
VERDICT: unsupported
DECIDING: The wagon is empty.
ESCALATE: no
```

**Key:** key: unsupported / The wagon is empty. / no

*invented: scenario written for this item (key by construction)*

### L2-E-01 · kind E · teaching · tier 1 · works the pattern through · tags: tree-with-pruning

**User:**
```text
A pattern over a tin roof has two branches, holed and whole. Rain in the loft kills holed. What does the pattern keep?
```

**Thinking:**
```text
Rain kills holed.
Whole is the living branch.
```

**Reply:**
```text
I keep whole.
Rain killed the holed branch.
```

**Variant user:**
```text
The tin roof splits into a holed branch and a whole branch. Loft rain kills the holed one. What remains?
A. Keep both branches until the gutter is checked.
B. Keep the whole branch, since rain in the loft kills holed.
C. Keep holed, because rain can be mopped later.
D. Abandon the roof pattern and call the loft unreadable.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Keep the whole branch, because rain in the loft kills holed. · variant key: {"answer": "B"}

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-E-02 · kind E · teaching · tier 1 · stops when the pattern settles · tags: falsifier-first

**User:**
```text
A pattern over a linen claim of unstained cloth meets a soot line first. Three praise lines follow the soot. Where do you finish?
```

**Thinking:**
```text
Soot kills unstained.
The praise lines can wait.
```

**Reply:**
```text
I finish at the soot line.
Unstained is already contradicted.
```

**Variant user:**
```text
Unstained linen meets a soot line before three praise lines. Where does the pattern finish?
A. Finish after the praise lines, since they were written too.
B. Finish by averaging soot and praise into slightly marked.
C. Finish at the soot line, since unstained is already contradicted.
D. Finish by striking the soot line as too harsh.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Finish at the soot line, because unstained is already contradicted. · variant key: {"answer": "C"}

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-E-03 · kind E · teaching · tier 1 · works the pattern through · tags: elimination

**User:**
```text
A pattern eliminates jars of honey, wax, and tar. The spill is dry and unscented, which honey and tar are not. Which jar remains?
```

**Thinking:**
```text
Honey fails dry and unscented.
Tar fails too.
Wax remains.
```

**Reply:**
```text
I keep the wax jar.
Honey and tar fail the spill.
```

**Variant user:**
```text
Honey, wax, and tar could be a spill. The spill is dry and unscented. Honey and tar are neither. Which jar remains?
A. Keep honey, because it is the most common jar.
B. Keep tar, because a dark spill looks like tar.
C. Keep all three until someone smells the spill.
D. Keep the wax jar, since honey and tar fail the dry unscented spill.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Keep the wax jar, because honey and tar fail the dry unscented spill. · variant key: {"answer": "D"}

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-E-04 · kind E · teaching · tier 1 · stops when the pattern settles · tags: constraint-grid

**User:**
```text
A pattern fills a grid of three hooks and three coats. Two coats are placed and one hook is empty. Do you halt with the empty hook open?
```

**Thinking:**
```text
One hook is still empty.
Two coats do not close it.
```

**Reply:**
```text
I place the remaining coat.
An open hook is still work.
```

**Variant user:**
```text
Three hooks and three coats, two coats already placed, one hook still empty. Do you halt?
A. Place the last coat, since an empty hook is still an open cell.
B. Halt now, since two coats are enough to see the idea.
C. Remove a placed coat and leave two hooks empty.
D. Call the grid finished because most cells are full.
Put the letter alone on an ANSWER: line.
```

**Key:** quiz (written): Place the last coat, because an empty hook is still an open cell. · variant key: {"answer": "A"}

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-E-05 · kind E · code · tier 3 · stops when the pattern settles

**User:**
```text
Config mike gives the garnet.

Material:
The garnet shuts at dusk.

Statement: The garnet shuts at dusk.
```

**Thinking:**
```text
I end the garnet page because it shows this.
```

**Reply:**
```text
I end the garnet page because it shows this.
VERDICT: supported
DECIDING: The garnet shuts at dusk.
ESCALATE: no
```

**Variant user:**
```text
Is the garnet shut at dusk?
```

**Key:** quiz (verdict): supported / The garnet shuts at dusk. / no · variant key: {"verdict": "supported", "deciding": ["The garnet shuts at dusk."], "escalate": "no"}

*invented: notice, statement and checker trace, written for this item (key by construction)*

### L2-E-06 · kind E · record · tier 3 · works the pattern through

**User:**
```text
Record november holds the zircon.

Material:
The zircon shuts at dusk.

Statement: The zircon holds forty litres.
```

**Thinking:**
```text
I settle the zircon page because it skips this.
```

**Reply:**
```text
I settle the zircon page because it skips this.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Variant user:**
```text
Does the zircon hold forty litres?
```

**Key:** quiz (verdict): cannot_tell / NONE / no · variant key: {"verdict": "cannot_tell", "deciding": [], "escalate": "no"}

*invented: notice, statement and checker trace, written for this item (key by construction)*
