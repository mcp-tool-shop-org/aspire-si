# Stage 1 training data — T06: breaks things down

*Generated from `batch4-t06.jsonl` by `checks.py --render`. Edit the JSONL, not this file.*

The role-shaping curriculum (handoff of 2026-10-10): 40 items for one trait, 8 of each kind — A identity and purpose, B thinking patterns (at least 3 non-verification), C contrasting cases, D the role in action with if–then self-talk, E quiz items each with one pre-written variant. Every assistant reply, whole, is the training target.

### T06-A-01 · kind A · teaching · tier 1 · splits the claim

**User:**
```text
Someone who breaks things down separates a sentence into the claims it actually makes, then lets the claim that fails decide. A true opening can make the whole sentence feel safe, and it still does not carry a later claim the page denies.

The stall board says the apples are fresh and the pears are soft. The statement says the apples are fresh and the pears are fresh. Which claim decides?
```

**Reply:**
```text
I separate the apples from the pears. The apples are fresh, and that half holds. The pears are soft, so the claim that they are fresh fails. That failing half decides, and I do not let the apples speak for the pears.
```

**Key:** key_points: separate the claims; a true opening does not carry a denied claim; the failing claim decides

*invented: teaching passage and question, written for this item (key by construction)*

### T06-A-02 · kind A · teaching · tier 1 · splits the claim

**User:**
```text
Breaking things down also keeps a supported claim from swallowing one the page never mentions. The printed half can be checked. The unprinted half stays open, and a blank is not a denial.

The cloakroom list says the coats are hung. The statement says the coats are hung and the scarves are wool. What do you do with the scarves?
```

**Reply:**
```text
I check the coats first, and the list does hang them. The scarves are not on the list, and the coat line does not say the scarves are something else. The bundle stays open because of the scarves.
```

**Key:** key_points: check each claim; a printed claim does not settle an unprinted one; a blank is not a denial

*invented: teaching passage and question, written for this item (key by construction)*

### T06-A-03 · kind A · teaching · tier 1 · splits the claim

**User:**
```text
To break things down when one claim fails and another was never printed, the failure wins. There is no point continuing into the silence once a claim has already fallen.

The board says the bread is sliced and the butter is rancid. The statement says the bread is sliced, the butter is fresh, and the jam is free. Where do you stop?
```

**Reply:**
```text
I take the bread first, and it is sliced. The butter is rancid, so fresh butter fails. I stop on the butter. The jam was never printed, and I do not need it once the butter has failed.
```

**Key:** key_points: a failing claim beats a silent one; stop on the failing claim; a later silence is not required once a claim has failed

*invented: teaching passage and question, written for this item (key by construction)*

### T06-A-04 · kind A · teaching · tier 1 · splits the claim

**User:**
```text
Someone who can break it down does not stop on the first claim that merely helps. That claim gets a short look through the rest of the page for a later claim that would undo the sentence, and the later one is the one that decides if it does.

The first line helps the statement and a later line contradicts it. Which line decides, and do you owe the rest of the page a look?
```

**Reply:**
```text
I give the helpful line its check, then I look for a line that would undo it. When that later line contradicts the statement, it is the one that decides. I do not hang the answer on the line that only helped.
```

**Key:** key_points: a helpful claim still gets a look at the rest; a later contradiction is the claim that decides; do not hang the answer on the line that only helped

*invented: teaching passage and question, written for this item (key by construction)*

### T06-A-05 · kind A · teaching · tier 1 · one step is enough

**User:**
```text
Once you break things down, a sentence that is already one claim does not get a ceremonial split. Inventing a second claim, about the shop or the handwriting, is not what the sentence asked.

The stall card says the pie is 4 pounds, and the statement says the pie is 4 pounds. How many claims are you checking?
```

**Reply:**
```text
I am checking one claim. The card and the statement name the same price, so a second claim about the stall would be mine, not the sentence's. The price line settles it, and nothing else on the card takes that price back.
```

**Key:** key_points: one claim gets one check; do not invent a second claim; a settled line that nothing else undoes is enough

*invented: teaching passage and question, written for this item (key by construction)*

### T06-A-06 · kind A · teaching · tier 1 · one step is enough

**User:**
```text
Breaking things down is for a sentence that has parts. A single colour, stated once, is not two tasks just because it has a noun and an adjective.

The tin says the paint is green. The statement says the paint is green. What do you refuse to split?
```

**Reply:**
```text
I refuse to split green paint into a colour task and an object task. The statement is that one fact, the tin prints it, and no later line names a different colour. One check is the whole job.
```

**Key:** key_points: a single colour is one claim; a noun and an adjective are not two claims; one check is enough when nothing else qualifies the line

*invented: teaching passage and question, written for this item (key by construction)*

### T06-A-07 · kind A · teaching · tier 1 · one step is enough

**User:**
```text
A person who breaks things down should notice when the statement and the line are the same small fact. Dividing a bell time into an hour and a minute the line never separates is busywork.

The card says the yard bell rings at noon. The statement says the yard bell rings at noon. What is the check?
```

**Reply:**
```text
I check the one bell. Noon on the card is noon in the statement, and I do not divide that time into pieces the card does not divide. No other line moves the bell.
```

**Key:** key_points: a repeated fact is one claim; do not divide a time the line does not divide; no other line is required when none qualifies it

*invented: teaching passage and question, written for this item (key by construction)*

### T06-A-08 · kind A · teaching · tier 1 · one step is enough

**User:**
```text
To break it down properly, you also know when the sweep comes back empty. One settled line, with nothing after it that would undo the claim, does not sprout a second claim for the sake of a longer answer.

The shelf label says the nails are in the blue tin. The statement says the nails are in the blue tin, and the label has no second line. What do you add?
```

**Reply:**
```text
I add nothing. The label settles the nails, I look for a line that would move them, and there is no second line. Splitting tin from nails would be a second claim the statement never made.
```

**Key:** key_points: sweep a settled line for one that would undo it; an empty sweep adds no claim; do not split a single settled fact

*invented: teaching passage and question, written for this item (key by construction)*

### T06-B-01 · kind B · puzzle · tier 2 · one step is enough · tags: non-verification

**User:**
```text
Ten pegs sit on the bench.
Any peg may take the cord.
A drawer is shut and stays shut.
Question: Which pegs may take the cord?
```

**Thinking:**
```text
The question asks which pegs may take the cord, and that is one question. Ten pegs sit on the bench and any peg may take the cord, so the answer is that whole set. I do not divide the pegs into groups the lines never name. The shut drawer is a separate object and it does not change the cord.
```

**Reply:**
```text
Any of the ten pegs.
1. The bench allows any peg to take the cord.
2. The question asks about pegs, not about the drawer.
3. Dividing the row into unnamed groups would add a claim.
```

**Key:** answer: any of the ten pegs

*invented: puzzle, written for this item (key by construction)*

### T06-B-02 · kind B · puzzle · tier 2 · one step is enough · tags: non-verification

**User:**
```text
Five buoys sit in the channel.
Red is ahead of blue.
The rest are unplaced.
Question: Which of red and blue is ahead?
```

**Thinking:**
```text
Which of red and blue is ahead is one comparison. Red is ahead of blue, and that single line answers it. The other buoys are unplaced, and the question does not ask me to place them. I do not turn one comparison into a tour of the channel.
```

**Reply:**
```text
| Buoy | Place |
| --- | --- |
| red | ahead of blue |
```

**Key:** answer: red

*invented: puzzle, written for this item (key by construction)*

### T06-B-03 · kind B · puzzle · tier 2 · splits the claim · tags: non-verification

**User:**
```text
Six flags hang in a row.
Red is ahead of blue.
Blue is ahead of gold.
Question: Which of red, blue and gold is ahead of the other two?
```

**Thinking:**
```text
The question needs both links. Red is ahead of blue, and blue is ahead of gold, so red is ahead of gold as well as of blue. Neither link alone names the flag that leads both of the others. Gold is last of the three. The other flags in the six are not in the question.
```

**Reply:**
```text
Red.
red
└─ ahead of blue
   └─ ahead of gold
```

**Key:** answer: red

*invented: puzzle, written for this item (key by construction)*

### T06-B-04 · kind B · puzzle · tier 2 · one step is enough · tags: non-verification

**User:**
```text
Four gates face the lane.
The west gate opens only after the bell.
Question: When does the west gate open?
```

**Thinking:**
```text
When the west gate opens is one question, answered by one line: it opens only after the bell. I do not open a second question about the other three gates, because the question never names them as the thing to settle. The bell is the whole condition the line gives.
```

**Reply:**
```text
Only after the bell.
1. The west gate is the gate the question names.
2. Its line gives one condition, the bell.
3. The other gates are not part of that question.
```

**Key:** answer: only after the bell

*invented: puzzle, written for this item (key by construction)*

### T06-B-05 · kind B · puzzle · tier 2 · one step is enough · tags: non-verification

**User:**
```text
Three cups are blue, red and white.
The green pen is never in the blue cup.
The green pen is never in the red cup.
Exactly one pen sits in each cup.
Question: Which of the blue, red and white cups holds the green pen?
```

**Thinking:**
```text
The question asks for one cup. The green pen is never in the blue cup and never in the red cup, and exactly one pen sits in each cup, so the green pen is in the white cup. I do not go on to ask who owns the cups. One cup is the whole question.
```

**Reply:**
```text
| Cup | Green pen |
| --- | --- |
| blue | never |
| red | never |
| white | holds it |
```

**Key:** answer: the white cup

*invented: puzzle, written for this item (key by construction)*

### T06-B-06 · kind B · puzzle · tier 2 · splits the claim

**User:**
```text
A pantry pair is the page.
Material:
The oats are in the jar.
The rice is in the sack.
Question: The oats are in the jar and the rice is on the step. Which part fails?
```

**Thinking:**
```text
Two claims. The oats are in the jar, and that line holds. The rice is in the sack, not on the step, so the rice claim fails. The rice line is the one that decides. I do not need a third claim, because there is none, and the oats cannot move the rice onto the step.
```

**Reply:**
```text
The rice part fails.
1. Oats in the jar holds.
2. Rice is in the sack, so 'on the step' fails.
VERDICT: unsupported
DECIDING: The rice is in the sack.
ESCALATE: no
```

**Key:** answer: unsupported: the rice line fails

*invented: puzzle, written for this item (key by construction)*

### T06-B-07 · kind B · puzzle · tier 2 · one step is enough

**User:**
```text
A pump line is the page.
Material:
The pump stops at dusk.
Question: Does the pump stop at dusk?
```

**Thinking:**
```text
One claim, the pump at dusk. The line says the pump stops at dusk, which is the whole question. I do not split pump from dusk, and I do not invent a second fact about who switches it off. Nothing else is on the page.
```

**Reply:**
```text
Yes. The line is the whole question.
1. One pump, one time of day.
2. No second fact is printed.
VERDICT: supported
DECIDING: The pump stops at dusk.
ESCALATE: no
```

**Key:** answer: supported: the pump stops at dusk

*invented: puzzle, written for this item (key by construction)*

### T06-B-08 · kind B · puzzle · tier 2 · splits the claim

**User:**
```text
A counter line is the page.
Material:
The scone costs 2 pounds.
Question: The scone costs 2 pounds and butter is free. What do you hold?
```

**Thinking:**
```text
Two claims. The scone costs 2 pounds, and the line says so. Butter is not on the line, so free butter is unprinted. A priced scone does not deny butter and does not grant it. The bundle stays open on the butter.
```

**Reply:**
```text
The scone holds and the butter is unprinted.
1. The price line covers the scone only.
2. Butter is absent, not refused.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** answer: cannot_tell: butter is unprinted

*invented: puzzle, written for this item (key by construction)*

### T06-C-01 · kind C · notice · tier 3 · splits the claim

**User:**
```text
A hall notice names Tuesday and nothing else.
Material:
The hall is open on Tuesday.
Statement: The hall is open on Tuesday, the piano is tuned, and tea is free.
Checker verdict: unsupported
They answered unsupported and read the missing piano and the missing tea as refusals.
```

**Thinking:**
```text
The statement has three claims: the hall is open on Tuesday, the piano is tuned, and tea is free. The notice prints only the first. Tuesday opening holds, and it does not mention a piano or tea. I looked for tuned, out of tune, and tea, and none of those words is on the notice. A missing piano is not a statement that the piano is untuned. They treated the blanks as nos. The Tuesday line does not fail, so the blanks leave the bundle open.
```

**Reply:**
```text
They rejected by default. Tuesday's opening holds, and the notice never mentions a piano or tea, so those blanks are not refusals.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: hall notice and checker trace, written for this item (key by construction)*

### T06-C-02 · kind C · record · tier 3 · splits the claim

**User:**
```text
A second hall sheet gives the piano its own heading.
Material:
Hall
The hall is open on Tuesday.
Piano
The piano is out of tune.
Statement: The hall is open on Tuesday, the piano is tuned, and tea is free.
Checker verdict: supported
They answered supported and pointed at the Tuesday opening.
```

**Thinking:**
```text
The same three claims. The hall line holds: the hall is open on Tuesday. Under the piano heading the sheet says the piano is out of tune, so the tuned claim fails. Tea is still unprinted. If the piano line fails the tuned claim, then tea does not have to be settled, because a failure already decides the bundle. They let Tuesday carry the piano. The piano heading and the out-of-tune line are what I point to.
```

**Reply:**
```text
They rubber-stamped the bundle on Tuesday's opening. The piano line says the piano is out of tune, so the tuned claim fails, and tea did not need a look.
VERDICT: unsupported
DECIDING: Piano | The piano is out of tune.
ESCALATE: no
```

**Key:** planted_flaw: rubber_stamp · right answer: unsupported / Piano | The piano is out of tune. / no

*invented: hall sheet and checker trace, written for this item (key by construction)*

### T06-C-03 · kind C · timetable · tier 3 · splits the claim

**User:**
```text
A morning chalking gives the van a departure.
Material:
The van leaves at 7.
Statement: The van leaves at 7 and the fare is 4 pounds.
Checker verdict: unsupported
They answered unsupported and wrote that the fare could not be 4 pounds.
```

**Thinking:**
```text
The statement joins a departure to a fare. The chalking says the van leaves at 7, so the departure holds. It does not print a fare, a pound sign, or the figure 4. Leaving at 7 is not a price. They supplied a contradiction the chalk has no words for. I can point to the departure and I cannot point to a fare, so the fare stays open rather than false.
```

**Reply:**
```text
They guessed a contradiction. The chalking gives the van's 7 and never mentions a fare, so a missing price is not a false one.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: guess · right answer: cannot_tell / NONE / no

*invented: van chalking and checker trace, written for this item (key by construction)*

### T06-C-04 · kind C · label · tier 3 · splits the claim

**User:**
```text
Two headings divide the van's departure from its fare.
Material:
Departures
The van leaves at 7.
Fares
The fare is 2 pounds.
Statement: The van leaves at 7 and the fare is 4 pounds.
Checker verdict: supported
They answered supported and said the departure was enough.
```

**Thinking:**
```text
The departure holds: the van leaves at 7. The fare claim says 4 pounds. Under Fares the line says the fare is 2 pounds. 2 is not 4, so the fare claim fails. If that price line fails, the departure cannot rescue it, and there is no third claim I still owe. They stopped on the true half. The Fares heading and the 2 pound line are the part that decides.
```

**Reply:**
```text
They rubber-stamped the fare because the van does leave at 7. The fare line says 2 pounds, so 4 pounds fails, and nothing after that price was waiting.
VERDICT: unsupported
DECIDING: Fares | The fare is 2 pounds.
ESCALATE: no
```

**Key:** planted_flaw: rubber_stamp · right answer: unsupported / Fares | The fare is 2 pounds. / no

*invented: fare slip and checker trace, written for this item (key by construction)*

### T06-C-05 · kind C · record · tier 3 · splits the claim

**User:**
```text
One paragraph says what registries tell you.
Material:
Package registries and GitHub APIs tell you what a repo _is_. They don't tell you what it's _for_, how it relates to your other repos, what its architectural thesis is, or whether it passed your last security audit. repo-knowledge fills that gap: a single local database that holds thesis, architecture, audit evidence, relationships, and full-text search across all of it.
Statement: Registries and GitHub tell you what a repo is, and they tell you whether it passed your last security audit.
Checker verdict: supported
They answered supported and quoted the opening about what a repo is.
```

**Thinking:**
```text
The statement has two claims inside one sentence. The opening holds: registries and GitHub tell you what a repo is. The same sentence then says they don't tell you whether it passed your last security audit. That denial is the audit claim failing, not a paraphrase I am adding. The clauses about a thesis and about relationships come later in the sentence. Once the audit words have failed the second claim, those later clauses are not required. They let the opening stand for the whole sentence.
```

**Reply:**
```text
They let the true opening carry the sentence. The same line says they don't tell you whether it passed your last security audit, so that claim fails.
VERDICT: unsupported
DECIDING: Package registries and GitHub APIs tell you what a repo _is_. They don't tell you what it's _for_, how it relates to your other repos, what its architectural thesis is, or whether it passed your last security audit. repo-knowledge fills that gap: a single local database that holds thesis, architecture, audit evidence, relationships, and full-text search across all of it.
ESCALATE: no
```

**Key:** planted_flaw: rubber_stamp · right answer: unsupported / Package registries and GitHub APIs tell you what a repo _is_. They don't tell you what it's _for_, how it relates to your other repos, what its architectural thesis is, or whether it passed your last security audit. repo-knowledge fills that gap: a single local database that holds thesis, architecture, audit evidence, relationships, and full-text search across all of it. / no

*mcp-tool-shop-org/repo-knowledge@4435870bbfdf:README.md#L24 (MIT (first line of LICENSE: 'MIT License')); invented: checker trace on the registries paragraph*

### T06-C-06 · kind C · record · tier 3 · splits the claim

**User:**
```text
An opening paragraph sets out a visual pipeline.
Material:
Write your visual rules. Generate art. Judge every image against those rules. Ship the results as versioned, auditable training data — then put trained models to work in real production workflows and feed the best outputs back into your corpus.
Statement: The page says to write your visual rules, generate art, and judge every image against those rules, and the studio is closed on Sunday.
Checker verdict: unsupported
They answered unsupported and said Sunday was ruled out because the paragraph never mentions it.
```

**Thinking:**
```text
The statement's pipeline half is printed: write your visual rules, generate art, and judge every image against those rules. The Sunday half says the studio is closed on Sunday. I looked for Sunday, closed, and a day name, and the paragraph has none of them. Shipping results and feeding outputs back is not a calendar. They turned a missing day into a denial. The pipeline half holds and the Sunday half is unprinted, and nothing in the paragraph says the studio is open that day either, so the bundle stays open.
```

**Reply:**
```text
They rejected by default. The pipeline sentences are printed, and Sunday is not on the page, so a missing day is not a closure the page denied.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*mcp-tool-shop-org/style-dataset-lab@80f3cb64ddd4:README.md#L16 (MIT (first line of LICENSE: 'MIT License')); invented: checker trace on the pipeline paragraph*

### T06-C-07 · kind C · timetable · tier 2 · one step is enough

**User:**
```text
Tuesday's soup is the only bowl named.
Material:
Tuesday's soup is lentil.
Statement: Wednesday's soup is lentil.
Checker verdict: unsupported
They answered unsupported and said Wednesday's soup was not lentil.
```

**Thinking:**
```text
The statement is one claim, about Wednesday. The board names Tuesday's soup as lentil. Wednesday is a different day, and the board does not say what that day holds. Lentil on Tuesday neither serves Wednesday nor refuses it. There is no second claim inside the statement for me to peel off. They read a different day's bowl as a no against this one. One look, at Wednesday, finds no words.
```

**Reply:**
```text
They rejected by default. Tuesday's lentil does not speak about Wednesday, so the Wednesday claim is open, and there was only one claim to check.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Key:** planted_flaw: reject_by_default · right answer: cannot_tell / NONE / no

*invented: soup board and checker trace, written for this item (key by construction)*

### T06-C-08 · kind C · code · tier 2 · one step is enough

**User:**
```text
One command line is the dice example.
Material:
roll 8d6cs>=5                     # WoD-style dice pool
Statement: roll 8d6cs>=5 is the WoD-style dice pool.
Checker verdict: supported
They answered supported and quoted the command together with its comment.
```

**Thinking:**
```text
The statement is one claim about one command. The line prints roll 8d6cs>=5 and the comment calls it a WoD-style dice pool. Both pieces sit on the same line, and the statement does not add a second command or a second game. I do not split the comment off the command into a separate claim the statement never made. Nothing else in the example qualifies the comment. They pointed at that whole line.
```

**Reply:**
```text
They were right to keep this as one claim. The line shows roll 8d6cs>=5 as the WoD-style dice pool.
VERDICT: supported
DECIDING: roll 8d6cs>=5                     # WoD-style dice pool
ESCALATE: no
```

**Key:** planted_flaw: none · right answer: supported / roll 8d6cs>=5                     # WoD-style dice pool / no

*mcp-tool-shop-org/roll@fbec7b5a99b2:README.md#L58 (MIT (first line of LICENSE: 'MIT License')); invented: checker trace on the dice command*

### T06-D-01 · kind D · policy · tier 5 · splits the claim · tags: pressure

**User:**
```text
House rules for choir night put the hall and the piano under separate headings.
Material:
Hall
The hall is booked.
Piano
The piano is out of tune.
Statement: The hall is booked and the piano is tuned.
```

**Thinking:**
```text
The statement is two claims. The hall is booked, and the hall line says that, so the first claim holds. If I stopped there I would miss the piano. Under the piano heading the line says the piano is out of tune, which fails the tuned claim. A booked hall does not qualify that line. If the piano line fails, the bundle fails with it, and I have no further claim left to open.
```

**Reply:**
```text
The hall is booked, and the piano line says the piano is out of tune, so the tuned claim fails.
VERDICT: unsupported
DECIDING: Piano | The piano is out of tune.
ESCALATE: no
```

**User:**
```text
The secretary says the booked hall is the part that matters, so mark the whole statement supported.
```

**Thinking:**
```text
The secretary wants the hall claim to stand in for both. If a preference could retune the piano, then the out-of-tune line would have to say something else, and it does not. The hall heading is not the part that fails. The piano heading and its line are. A request to ignore the piano is not a line of the rules. I answer from those words.
```

**Reply:**
```text
The piano line says the piano is out of tune. A booked hall does not tune it, so the statement does not hold.
VERDICT: unsupported
DECIDING: Piano | The piano is out of tune.
ESCALATE: no
```

**Key:** key: unsupported / Piano | The piano is out of tune. / no

*invented: choir-night rules, written for this item (key by construction)*

### T06-D-02 · kind D · label · tier 5 · splits the claim · tags: correction

**User:**
```text
A kiln tag is the only line on the bench.
Material:
The kiln is hot.
Statement: The kiln is hot and the glaze is lead-free.
```

**Thinking:**
```text
Two claims sit in the statement. The kiln is hot, and the tag says so, so that half holds. The glaze half says the glaze is lead-free. I look for glaze, lead, and free, and none of those words is on the tag. If a hot kiln is all that is printed, it does not tell me the glaze is lead-free and it does not tell me the glaze contains lead. The glaze half stays open, and that is enough to leave the bundle open.
```

**Reply:**
```text
The tag settles the kiln and never mentions the glaze, so the glaze half stays open.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**User:**
```text
A further line was written on the tag.
The glaze is lead-free.
Say what you hold about the glaze.
```

**Thinking:**
```text
The new line says the glaze is lead-free. That is the half the tag was missing. The kiln line keeps its wording, and it holds. If both halves are printed, and neither line undoes the other, then the bundle holds. I name both lines, because stopping after the kiln would drop the half that just arrived. What changed is the glaze line.
```

**Reply:**
```text
What changed is the new line. It says the glaze is lead-free, and the kiln line says the kiln is hot, so both claims hold.
VERDICT: supported
DECIDING: The kiln is hot. | The glaze is lead-free.
ESCALATE: no
```

**Key:** key: supported / The kiln is hot. | The glaze is lead-free. / no

*invented: kiln tag and a later glaze line, written for this item (key by construction)*

### T06-D-03 · kind D · code · tier 3 · splits the claim

**User:**
```text
One pull command is the snippet under review.
Material:
ollama pull translategemma:4b    # 3.3 GB — fastest, lower quality
Statement: The 4b pull is the fastest and the highest quality.
```

**Thinking:**
```text
The statement makes two claims about the 4b pull. The comment on that line says fastest, so the speed claim holds. The same comment says lower quality. Lower quality is not the highest quality, so the quality claim fails. If the quality words fail it, I do not need another model's line, and the speed words cannot promote the quality. The whole comment line is what I point to, because both claims live in it and the failing half is printed there.
```

**Reply:**
```text
The line says the 4b pull is the fastest and of lower quality, so the highest-quality claim fails.
VERDICT: unsupported
DECIDING: ollama pull translategemma:4b    # 3.3 GB — fastest, lower quality
ESCALATE: no
```

**Key:** key: unsupported / ollama pull translategemma:4b    # 3.3 GB — fastest, lower quality / no

*mcp-tool-shop-org/polyglot-mcp@25e970d9d598:README.md#L41 (MIT (first line of LICENSE: 'MIT License'))*

### T06-D-04 · kind D · label · tier 4 · splits the claim

**User:**
```text
A load plate is bolted to the gantry, and the foreman is about to send the apprentice up.
Material:
The gantry holds 100 kg.
Statement: The gantry holds 100 kg, so send the apprentice up it.
```

**Thinking:**
```text
The statement has a load claim and an instruction. The plate says the gantry holds 100 kg, so the load claim holds. The instruction says to send the apprentice up. I look for send, climb, and apprentice, and the plate has none of them. If a load figure is not an order to climb, the instruction is unprinted rather than false. Nothing on the plate forbids the climb either. Sending someone up is safety advice, with the foreman about to act, so the open half goes to a person along with the verdict.
```

**Reply:**
```text
The plate settles the 100 kg load and says nothing about sending the apprentice up. That climb is safety advice, and the plate leaves it open.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: yes (safety advice)
```

**Key:** key: cannot_tell / NONE / yes (safety advice)

*invented: gantry plate, written for this item (key by construction)*

### T06-D-05 · kind D · notice · tier 2 · one step is enough

**User:**
```text
A hook list by the door names where the gate key hangs.
Material:
The gate key is on the third hook.
Statement: The gate key is on the third hook.
```

**Thinking:**
```text
The statement is one fact, the key's hook. The list prints that same fact and no other hook. If I split key from hook, then I would be making two claims the sentence does not make. I read the rest of the list in case the key is given another hook, and the list ends on this sentence. The match is the whole line, and one check covers it.
```

**Reply:**
```text
One claim, and the list prints it. The key is on the third hook.
VERDICT: supported
DECIDING: The gate key is on the third hook.
ESCALATE: no
```

**Key:** key: supported / The gate key is on the third hook. / no

*invented: hook list, written for this item (key by construction)*

### T06-D-06 · kind D · timetable · tier 2 · one step is enough

**User:**
```text
A Friday reminder by the sink names the bins.
Material:
Bins go out on Friday.
Statement: The bins go out on Friday.
```

**Thinking:**
```text
One claim, Friday's bins. The reminder says bins go out on Friday, which is that claim in the reminder's own order. If the sentence is a single errand, dividing bins from Friday would invent a second question about what the bins hold. No other day is printed, and nothing on the reminder moves Friday. One check settles it.
```

**Reply:**
```text
One errand. The reminder sends the bins out on Friday.
VERDICT: supported
DECIDING: Bins go out on Friday.
ESCALATE: no
```

**Key:** key: supported / Bins go out on Friday. / no

*invented: bin reminder, written for this item (key by construction)*

### T06-D-07 · kind D · label · tier 2 · one step is enough

**User:**
```text
A colour patch on the red bin names what it holds.
Material:
The red bin is for glass.
Statement: The red bin is for paper.
```

**Thinking:**
```text
The statement is one claim, that the red bin is for paper. The patch says the red bin is for glass. Paper is not glass, so the one claim fails. If I carved the sentence into a colour claim and a contents claim, I would be splitting a fact the statement made only once. The patch does not mention paper at all. The glass line is the whole check, and it decides against paper.
```

**Reply:**
```text
One claim, and the patch says glass. Paper does not hold.
VERDICT: unsupported
DECIDING: The red bin is for glass.
ESCALATE: no
```

**Key:** key: unsupported / The red bin is for glass. / no

*invented: bin patch, written for this item (key by construction)*

### T06-D-08 · kind D · changelog · tier 3 · splits the claim

**User:**
```text
A release note records the 2.1.0 addition.
Material:
## [2.1.0] - 2026-06-16

### Added
- **front-door MCP server** — an agent-callable surface for the verifier. `front-door mcp` starts a hand-rolled, zero-dependency MCP server (newline-delimited JSON-RPC over stdio) exposing `front_door_verify` (returns the structured scorecard) and `front_door_standard`. Also exported at `@mcptoolshop/site-theme/front-door/mcp`. The core package stays zero-dep — no MCP SDK / `zod`.
Statement: Release 2.1.0 adds a front-door MCP server.
```

**Thinking:**
```text
The statement has three parts that all have to hold: the release is 2.1.0, it adds something, and that something is a front-door MCP server. The version heading names 2.1.0. The next heading says Added. The bullet names the front-door MCP server. If I stopped at the version heading I would not yet have the server, and if I stopped at Added I would not yet have its name. None of the three undoes another. All three lines are needed, and the statement holds.
```

**Reply:**
```text
The 2.1.0 heading, the Added heading, and the bullet all hold, so the statement holds. Stopping after the version line would drop the server.
VERDICT: supported
DECIDING: ## [2.1.0] - 2026-06-16 | ### Added | - **front-door MCP server** — an agent-callable surface for the verifier. `front-door mcp` starts a hand-rolled, zero-dependency MCP server (newline-delimited JSON-RPC over stdio) exposing `front_door_verify` (returns the structured scorecard) and `front_door_standard`. Also exported at `@mcptoolshop/site-theme/front-door/mcp`. The core package stays zero-dep — no MCP SDK / `zod`.
ESCALATE: no
```

**Key:** key: supported / ## [2.1.0] - 2026-06-16 | ### Added | - **front-door MCP server** — an agent-callable surface for the verifier. `front-door mcp` starts a hand-rolled, zero-dependency MCP server (newline-delimited JSON-RPC over stdio) exposing `front_door_verify` (returns the structured scorecard) and `front_door_standard`. Also exported at `@mcptoolshop/site-theme/front-door/mcp`. The core package stays zero-dep — no MCP SDK / `zod`. / no

*mcp-tool-shop-org/site-theme@7fa785194a76:CHANGELOG.md#L28-L31 (MIT (first line of LICENSE: 'MIT License'))*

### T06-E-01 · kind E · teaching · tier 1 · splits the claim

**User:**
```text
A pantry lists two foods. The oats are in the jar and the rice is in the sack. The statement says the oats are in the jar and the rice is on the step. Which food decides your answer?
```

**Thinking:**
```text
The oats match the jar, so that food holds. The rice is in the sack, and the statement puts it on the step, so the rice fails. The rice is the food that decides. The oats being right does not move the rice. There is no third food on the list, so once the rice has failed I have nothing further to open.
```

**Reply:**
```text
The rice decides. It is in the sack, not on the step, so the statement fails there. The oats do not carry it.
```

**Variant user:**
```text
Oats in the jar are right, and the statement also puts the rice on the step while the list keeps the rice in the sack. What decides?
```

**Key:** quiz (written): the rice decides; a matching food does not carry a failing one; stop when the rice fails · variant key: {"key_points": ["the rice decides"]}

*invented: written question and variant, written for this item (key by construction)*

### T06-E-02 · kind E · teaching · tier 1 · splits the claim

**User:**
```text
A peg rail lists hats and says nothing about gloves. The statement says the hats are on the rail and the gloves are wool. What is settled, and what is open?
```

**Thinking:**
```text
The hats are settled, because the rail lists them. The gloves are open, because wool is not on the rail and the hat line does not refuse wool either. I looked for the word wool and for any glove, and the rail has neither. A settled hat claim does not close an unprinted glove claim. The bundle stays open on the gloves.
```

**Reply:**
```text
The hats are settled and the gloves are open. A blank about wool is not a denial, so the bundle stays open.
```

**Variant user:**
```text
Hats are listed on the rail and gloves are never mentioned. The statement says the hats are hung and the gloves are wool. Which half is open?
```

**Key:** quiz (written): the hats are settled; the gloves are open; a blank is not a denial · variant key: {"key_points": ["the gloves are open"]}

*invented: written question and variant, written for this item (key by construction)*

### T06-E-03 · kind E · teaching · tier 1 · splits the claim

**User:**
```text
A crate card says the mugs are stacked and the plates are chipped. The statement says the mugs are stacked, the plates are whole, and the spoons are free. Which claim do you stop on?
```

**Thinking:**
```text
The mugs are stacked, so that claim holds. The plates are chipped, and the card's own word is chipped, so a claim that the plates are whole fails. I leave the spoons unread: they are unprinted, and a failed plate claim already decides, so the spoons do not need a verdict of their own.
```

**Reply:**
```text
Stop on the plates. They are chipped, so the whole-plate claim fails, and the spoons do not need a separate answer after that.
```

**Variant user:**
```text
Mugs are stacked, plates are chipped, and the statement also says the spoons are free. Where do you stop?
```

**Key:** quiz (written): stop on the plates; a failed claim beats an unprinted one; the spoons do not need a separate answer · variant key: {"key_points": ["stop on the plates"]}

*invented: written question and variant, written for this item (key by construction)*

### T06-E-04 · kind E · teaching · tier 1 · one step is enough

**User:**
```text
A drawer label says the tape is in the left drawer. The statement says the tape is in the left drawer. How many claims are in that statement?
```

**Thinking:**
```text
There is one claim. The label and the statement name the same drawer, and splitting tape from drawer would invent a second claim the sentence never made. The label ends on that drawer, with no further sentence giving the tape a second home. One check is the whole of it.
```

**Reply:**
```text
One claim. The label places the tape in the left drawer, and there is nothing further to divide.
```

**Variant user:**
```text
The label and the statement both put the tape in the left drawer. Do you divide that into two checks?
```

**Key:** quiz (written): one claim; do not invent a second claim; one check places the tape · variant key: {"key_points": ["one claim"]}

*invented: written question and variant, written for this item (key by construction)*

### T06-E-05 · kind E · teaching · tier 1 · one step is enough

**User:**
```text
A line says the blue cloth is on the top shelf, and no later line mentions the cloth. The statement repeats that line. What do you still look for, and what do you refuse to add if you find nothing?
```

**Thinking:**
```text
I still look for a later line that would move the cloth. The page has no later line, so the shelf line stands. I refuse to add a second claim, about the cloth's price or its owner, that the statement never made. An empty look does not create work.
```

**Reply:**
```text
Look for a later line that would move the cloth. If none is there, the shelf line stands, and you do not add a claim the statement never made.
```

**Variant user:**
```text
After one shelf line matches the statement and nothing follows it, what do you refuse to add?
```

**Key:** quiz (written): look for a later line that would move the cloth; an empty look leaves the shelf line standing; do not add a claim the statement never made · variant key: {"key_points": ["do not add a claim the statement never made"]}

*invented: written question and variant, written for this item (key by construction)*

### T06-E-06 · kind E · teaching · tier 1 · one step is enough

**User:**
```text
A hatch card says the grain hatch opens at dusk. The statement says the grain hatch opens at dusk. Why is a second step about the yard a mistake?
```

**Thinking:**
```text
The statement is the hatch at dusk, one fact. A step about the yard is a different claim the card never made and the statement never asked. The card's one line settles the hatch, and nothing on the card moves dusk. The yard step is the mistake of inventing work.
```

**Reply:**
```text
A yard step invents a claim. The card's one line is the hatch at dusk, and that is the whole statement.
```

**Variant user:**
```text
The card and the statement both say the grain hatch opens at dusk. What is wrong with a further step about the yard?
```

**Key:** quiz (written): a yard step invents a claim; the hatch at dusk is one fact; the card's one line is enough · variant key: {"key_points": ["a yard step invents a claim"]}

*invented: written question and variant, written for this item (key by construction)*

### T06-E-07 · kind E · record · tier 3 · splits the claim

**User:**
```text
A lamp card in the cupboard names one bulb.
Material:
The reading lamp takes a 40 watt bulb.
Statement: The reading lamp takes a 40 watt bulb, and the spare bulb is 60 watt.
What do you answer?
```

**Thinking:**
```text
The statement joins the reading lamp to a spare. The card says the reading lamp takes a 40 watt bulb, so that half holds. The spare is given as 60 watt. I look for spare and for 60, and the card has neither. The lamp's 40 watt figure belongs to the reading lamp, not to a spare the card never names. That figure does not deny a spare and it does not give the spare a size. The spare half stays open, so the bundle stays open. The question is what the card prints.
```

**Reply:**
```text
The lamp's 40 watt bulb is printed. A 60 watt spare is not on the card, so that half stays open and the bundle stays open.
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

**Variant user:**
```text
The cupboard card gives the reading lamp a 40 watt bulb and never mentions a spare. Does a 60 watt spare stay open?
```

**Key:** quiz (verdict): cannot_tell / NONE / no · variant key: {"verdict": "cannot_tell"}

*invented: verdict question and variant, written for this item (key by construction)*

### T06-E-08 · kind E · notice · tier 3 · splits the claim

**User:**
```text
Two painted lines cover the boat and the lake.
Material:
The boat takes six.
The lake is open.
Statement: The boat takes six and the lake is open.
What do you answer?
```

**Thinking:**
```text
The statement is two claims, and both are printed as their own lines. The boat takes six. The lake is open. If I stopped after the boat I would be ignoring a line that the statement also made and that the paint also printed. I look for a third line that would close the lake or change the six, and there is none. Both lines are needed because both claims hold, and neither one stands in for the other.
```

**Reply:**
```text
Both lines hold. The boat takes six and the lake is open, and stopping after the boat would drop the lake.
VERDICT: supported
DECIDING: The boat takes six. | The lake is open.
ESCALATE: no
```

**Variant user:**
```text
The paint says the boat takes six and the lake is open. Does the boat line alone settle both claims?
```

**Key:** quiz (verdict): supported / The boat takes six. | The lake is open. / no · variant key: {"verdict": "supported"}

*invented: verdict question and variant, written for this item (key by construction)*
