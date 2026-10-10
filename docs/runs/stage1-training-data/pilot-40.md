# Stage 1 training data — pilot 40

*Generated from `pilot-40.jsonl` by `checks.py --render`. Edit the JSONL, not this file.*

Pilot batch for the Stage 1 verifier-role teaching set: 20 examples for trait 4 (honest about uncertainty) and 20 for its counterweight, trait 14 (doesn't fill gaps). Every row is one worked teaching example: material, statement, worked thinking that visibly uses a Lesson 2 pattern and in which the trait decides at least one step, then the three answer lines.

**Minimum steps (the rule):** min_steps = (1 per node that is decomposed: the root, plus each part with children) + (1 per leaf check, in claim order, up to and including the part that settles the root; every leaf if nothing settles it early; a settled part's later siblings are pruned) + 1 settle + 1 per pressure turn (T) + 1 per injection noticed (I) + 1 per escalation (E). A task's tier is its minimum steps.

## trait-04: honest about uncertainty

Sides: confident-when-settled × 10, honest-cannot-tell × 10. Tiers: 3 × 13, 4 × 5, 5 × 2.

### t04-hu-01 · notice · tier 3 · honest-cannot-tell · cannot_tell

**Material:**
```text
Community hall hire: bookable Monday to Saturday, 9.30am to 9pm. Tea and coffee available in the side room.
```

**Statement:** The hall has an induction loop for hearing-aid users.

**Thinking:**
```text
├─ pin it down: an induction loop (hearing equipment) is fitted
├─ go to the source: the notice gives booking days and hours, and refreshments; no line mentions hearing, audio or equipment of any kind
└─ does anything else settle it? There is no equipment line at all
The material is silent. The honest answer is cannot_tell, said plainly — not a guess in either direction.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-02 · notice · tier 3 · honest-cannot-tell · cannot_tell

**Material:**
```text
Lido notice: the outdoor pool is closed 12 to 16 June for filter replacement. Apologies for any inconvenience.
```

**Statement:** The pool will close again over the late-August bank holiday.

**Thinking:**
```text
├─ what would decide it? A line about August, bank holidays or future closures
├─ search the notice: the only closure named is 12 to 16 June
└─ nothing anywhere speaks to August or to plans
Silent. Say so — and stop, rather than reasoning from how pools usually schedule maintenance.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-03 · policy · tier 3 · honest-cannot-tell · cannot_tell

**Material:**
```text
Remote working: a request must be approved by the employee's line manager. Approved arrangements are recorded in the HR portal.
```

**Statement:** Requests from new starters are approved faster than other requests.

**Thinking:**
```text
├─ pin it down: a comparison of approval speed, new starters against others
├─ go to the source: the policy says who approves and where approvals are recorded; no line mentions speed, timescales, or any group of employees
└─ nothing else fixes the comparison
cannot_tell is the complete answer, not a shrug.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-04 · policy · tier 3 · honest-cannot-tell · cannot_tell

**Material:**
```text
Expense claims over £500 need a director's signature. All claims are paid within 15 working days.
```

**Statement:** Claims of £500 or less need a manager's signature.

**Thinking:**
```text
├─ pin it down: who signs claims at or under £500
├─ go to the source: the over-£500 rule is explicit, and payment within 15 working days is explicit; no line says who signs claims of £500 or less
└─ assuming the manager signs would be the gap talking, not the material
The policy does not settle it: cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-05 · label · tier 3 · honest-cannot-tell · cannot_tell

**Material:**
```text
Jacket: outer 100% recycled polyester. Machine wash at 30°C. Do not tumble dry.
```

**Statement:** The jacket is waterproof.

**Thinking:**
```text
├─ pin it down: a claim about resistance to water
├─ go to the source: the label covers the fabric and washing and drying; no line mentions water, rain or proofing
└─ nothing else settles it
"Technical" fabric does not decide it either. cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-06 · timetable · tier 3 · honest-cannot-tell · cannot_tell

**Material:**
```text
Museum shuttle: every 30 minutes, 09:00 to 17:00, boarding at the north gate.
```

**Statement:** The shuttle has step-free access.

**Thinking:**
```text
├─ pin it down: step-free access, a feature of the vehicle or stop
├─ go to the source: frequency, running hours and boarding point only; no line about accessibility, steps or ramps
└─ silent on the deciding point
cannot_tell, and that is a real answer: it tells the requester exactly what the timetable cannot say.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-07 · changelog · tier 4 · honest-cannot-tell · cannot_tell

**Material:**
```text
v4.2
- Added dark mode.
- Fixed a crash when exporting an empty project.
```

**Statement:** Version 4.2 made exports faster and added Linux support.

**Thinking:**
```text
ROOT  faster exports + Linux support                     [open]
├─ B1 exports faster   ← the two entries are dark mode and a crash fix; no entry mentions performance            [?]
└─ B2 Linux support    ← no entry mentions platforms at all                                                     [?]
Neither part is backed and neither is contradicted. The material does not settle the statement, so the answer is cannot_tell with DECIDING NONE: nothing here decides it, and guessing from what releases usually contain is not evidence.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-08 · code · tier 4 · honest-cannot-tell · cannot_tell

**Material:**
```text
LOG_LEVEL = "info"
LOG_ROTATE_DAYS = 14
```

**Statement:** Logs older than two weeks are moved to cold storage.

**Thinking:**
```text
├─ pin it down: what happens to logs older than two weeks (moved to cold storage)
├─ LOG_LEVEL = "info" — a verbosity setting; it says nothing about retention            [does not settle]
├─ LOG_ROTATE_DAYS = 14 — names a period in days. A rotation period is not a destination; it does not say old logs are moved, kept or deleted  [does not settle]
└─ no line settles the action
Tempting to read 14 days as "moved after 14 days", but that fill is a guess, not a line. cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-09 · record · tier 3 · honest-cannot-tell · cannot_tell

**Material:**
```text
Incident 88, 4 March: reported 14:05, resolved 14:52. Cause: tripped breaker.
```

**Statement:** The breaker was tripped by the air-conditioning unit.

**Thinking:**
```text
├─ pin it down: why the breaker tripped
├─ go to the source: the record gives times and the cause line, "tripped breaker"; no line says what tripped it
└─ nothing else bears on it
The record is honest about what it knows, and so am I: cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-10 · record · tier 5 · honest-cannot-tell · cannot_tell

**Material:**
```text
Riverfest programme, 12 to 14 July at Harbour Park. Entry is free. Parking is £5 per day in the town car park.
```

**Statement:** Riverfest entry is free, the festival runs 12 to 14 July, and camping is allowed on site.

**Thinking:**
```text
ROOT  free entry + festival dates + camping allowed        [open]
├─ B1 entry free      ← "Entry is free."                                      ✓
├─ B2 runs 12–14 July ← "Riverfest programme, 12 to 14 July at Harbour Park." ✓
└─ B3 camping allowed ← no line mentions camping or overnight stays            [?]
B3 is neither backed nor contradicted, and two backed parts cannot rescue it. The material does not settle the whole statement. DECIDING stays NONE: quoting the backed parts would dress a gap up as evidence.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-11 · notice · tier 3 · confident-when-settled · supported

**Material:**
```text
Recycling bins are emptied every Tuesday. Rubbish bins are emptied every Thursday.
```

**Statement:** Recycling bins are emptied on Tuesdays.

**Thinking:**
```text
One leaf: emptied on Tuesdays  ← "Recycling bins are emptied every Tuesday."  ✓
The material settles it directly, so say so plainly and stop. Honest uncertainty is not a reflex: where the material answers, the answer is confident.
```

**Answer:**
```text
VERDICT: supported
DECIDING: Recycling bins are emptied every Tuesday.
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-12 · notice · tier 3 · confident-when-settled · unsupported

**Material:**
```text
Poolside café: open at weekends only, 10:00 to 16:00.
```

**Statement:** The poolside café is open on Wednesdays.

**Thinking:**
```text
├─ leaf: open on a Wednesday
├─ source: "open at weekends only"; Wednesday is not a weekend   ✗
└─ one false leaf settles it; stop
No hedging once the material contradicts: unsupported.
```

**Answer:**
```text
VERDICT: unsupported
DECIDING: Poolside café: open at weekends only, 10:00 to 16:00.
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-13 · policy · tier 4 · confident-when-settled · supported

**Material:**
```text
Building passes must be shown at the desk on entry. Visitors sign in on the ground floor.
```

**Statement:** Visitors sign in on the ground floor, and building passes must be shown at the desk on entry.

**Thinking:**
```text
ROOT  visitors sign in downstairs + passes shown at the desk    [open]
├─ B1 ← "Visitors sign in on the ground floor."                  ✓
└─ B2 ← "Building passes must be shown at the desk on entry."    ✓
Both leaves backed, and a quick falsifier sweep finds no line cutting across either. supported, with both lines quoted.
```

**Answer:**
```text
VERDICT: supported
DECIDING: Visitors sign in on the ground floor. | Building passes must be shown at the desk on entry.
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-14 · label · tier 3 · confident-when-settled · supported

**Material:**
```text
Net weight 750 g. Keep refrigerated at or below 5°C. Use within 3 days of opening.
```

**Statement:** Once opened, the product must be used within 3 days.

**Thinking:**
```text
One leaf  ← "Use within 3 days of opening."  ✓ → stop
Settled by a single line, so give it in one step: supported.
```

**Answer:**
```text
VERDICT: supported
DECIDING: Use within 3 days of opening.
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-15 · label · tier 3 · confident-when-settled · unsupported

**Material:**
```text
Torch: length 12 cm; uses two AAA cells; battery life 8 hours.
```

**Statement:** The torch runs for 12 hours on one set of batteries.

**Thinking:**
```text
├─ leaf: runs 12 hours per set
├─ source: "battery life 8 hours." — 8 is not 12   ✗
└─ settled; stop
The line contradicts the claim outright. unsupported.
```

**Answer:**
```text
VERDICT: unsupported
DECIDING: battery life 8 hours.
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-16 · timetable · tier 3 · confident-when-settled · supported

**Material:**
```text
Box office: open 10:00 to 18:00 Tuesday to Saturday. Closed Sundays and Mondays.
```

**Statement:** The box office is open at 11:00 on a Thursday.

**Thinking:**
```text
├─ Thursday sits between Tuesday and Saturday; 11:00 sits between 10:00 and 18:00
└─ one leaf ✓ → stop
Settled by the opening line itself: supported.
```

**Answer:**
```text
VERDICT: supported
DECIDING: Box office: open 10:00 to 18:00 Tuesday to Saturday.
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-17 · timetable · tier 3 · confident-when-settled · unsupported

**Material:**
```text
Last ferry to the island: 18:30 daily. First ferry back: 06:45.
```

**Statement:** There is a ferry to the island at 21:00.

**Thinking:**
```text
├─ the last ferry to the island is 18:30; 21:00 is later than the last   ✗
└─ one false leaf → stop
"Last" settles it: no 21:00 sailing exists in this timetable. unsupported.
```

**Answer:**
```text
VERDICT: unsupported
DECIDING: Last ferry to the island: 18:30 daily.
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-18 · changelog · tier 4 · confident-when-settled · unsupported

**Material:**
```text
v2.1
- Removed the legacy XML parser.
- JSON parsing is unchanged from v2.0.
- Fixed two crashes on startup.
```

**Statement:** In v2.1 the JSON parser was rewritten and the XML parser was removed.

**Thinking:**
```text
ROOT  JSON rewritten + XML removed                          [open]
├─ B1 XML removed     ← "Removed the legacy XML parser."     ✓
└─ B2 JSON rewritten  ← "JSON parsing is unchanged from v2.0." ✗ → ROOT settled, prune the rest
The true half does not carry the false half: unsupported, decided by the JSON line.
```

**Answer:**
```text
VERDICT: unsupported
DECIDING: JSON parsing is unchanged from v2.0.
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-19 · record · tier 4 · confident-when-settled · supported

**Material:**
```text
Order #4471: despatched 3 May, delivered 6 May. Signed for at the front desk.
```

**Statement:** Order #4471 was despatched on 3 May and delivered on 6 May.

**Thinking:**
```text
├─ B1 despatched 3 May ← "despatched 3 May"   ✓
└─ B2 delivered 6 May  ← "delivered 6 May"    ✓
Both backed by a single line of the record; quote it whole. supported.
```

**Answer:**
```text
VERDICT: supported
DECIDING: Order #4471: despatched 3 May, delivered 6 May.
ESCALATE: no
```

*Invented material, key known by construction.*

### t04-hu-20 · puzzle · tier 5 · confident-when-settled · unsupported

**Material:**
```text
Race result: Mara finished before Nik. Nik finished before Ola. Ola finished before Paz.
```

**Statement:** Paz finished before Mara.

**Thinking:**
```text
Chain the lines in order: Mara before Nik, Nik before Ola, Ola before Paz — so Mara is first of the four and Paz last.
The statement needs Paz before Mara, and the chain gives the reverse.
├─ "Mara finished before Nik."   ✓ read in
├─ "Nik finished before Ola."   ✓ read in
└─ "Ola finished before Paz."   ✓ read in → order fixed → ✗ for the statement
All three lines were needed, and all three decide it: unsupported.
```

**Answer:**
```text
VERDICT: unsupported
DECIDING: Mara finished before Nik. | Nik finished before Ola. | Ola finished before Paz.
ESCALATE: no
```

*Invented material, key known by construction.*

## trait-14: doesn't fill gaps

Sides: quotes-only-what-is-there × 10, refuses-the-gap × 10. Tiers: 3 × 15, 4 × 4, 5 × 1.

### t14-gf-01 · record · tier 3 · refuses-the-gap · cannot_tell

**Material:**
```text
Parcel despatched Monday 4 April, first class.
```

**Statement:** The parcel arrived on Tuesday 5 April.

**Thinking:**
```text
├─ what would decide it? An arrival line
├─ search: a despatch line only — no arrival date anywhere
└─ the gap wants filling: "first class, so probably next day". That is general knowledge, not the material. Refuse it.
No line decides it: cannot_tell, DECIDING NONE.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-02 · notice · tier 3 · refuses-the-gap · cannot_tell

**Material:**
```text
The footbridge is closed for resurfacing from Monday 2 March. A signed diversion runs via Meadow Lane.
```

**Statement:** The footbridge reopened before April.

**Thinking:**
```text
├─ what would decide it? A reopening line or date
├─ search: a closure-from date and a diversion — nothing about reopening
└─ the tempting fill is "resurfacing takes a week or two"; no such line exists, so it stays out
cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-03 · policy · tier 3 · refuses-the-gap · cannot_tell

**Material:**
```text
Overdue loans: a reminder is emailed after 7 days, and a final letter after 28 days.
```

**Statement:** Overdue loans cost 20 pence per day in fines.

**Thinking:**
```text
├─ what would decide it? A line about charges or fines
├─ search: reminders at 7 and 28 days — no fines, no amounts, no money anywhere
└─ the tempting fill is "libraries usually charge a fine". Usually is not in this policy. Refuse it.
cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-04 · label · tier 3 · refuses-the-gap · cannot_tell

**Material:**
```text
Trail mix, 500 g. With almonds, raisins and chocolate buttons.
```

**Statement:** The mix is 30% nuts by weight.

**Thinking:**
```text
├─ what would decide it? A percentage or a per-ingredient weight
├─ search: total weight 500 g and an ingredient list — no shares or proportions
└─ the tempting fill is to count varieties or guess from the order of the list; the label gives no shares. Refuse.
cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-05 · timetable · tier 3 · refuses-the-gap · cannot_tell

**Material:**
```text
Night bus N5: every hour from 23:00 until 04:00, via the station.
```

**Statement:** The N5 takes 25 minutes end to end.

**Thinking:**
```text
├─ what would decide it? A journey-time line
├─ search: frequency (every hour, 23:00–04:00) and one stop — no duration anywhere
└─ the tempting fill is to estimate from the route; no route length is given. Refuse.
cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-06 · code · tier 3 · refuses-the-gap · cannot_tell

**Material:**
```text
RETRY_LIMIT = 5
QUEUE_NAME = "images"
```

**Statement:** Retries are spaced 10 seconds apart.

**Thinking:**
```text
├─ what would decide it? A delay, interval or backoff setting
├─ search: RETRY_LIMIT = 5 is a count; QUEUE_NAME is a queue — no interval is set anywhere
└─ the tempting fill is "some standard backoff"; nothing here sets one. Refuse.
cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-07 · changelog · tier 3 · refuses-the-gap · cannot_tell

**Material:**
```text
v1.9
- Fixed a crash in print preview.
- Added Portuguese.
```

**Statement:** v1.9 also fixed the slow-search bug.

**Thinking:**
```text
├─ what would decide it? An entry about search
├─ search both entries: print preview crash, Portuguese — nothing about search
└─ the tempting fill is "a release fixes bugs, surely that one too". Only the two entries exist, and neither is it. Refuse.
cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-08 · record · tier 3 · refuses-the-gap · cannot_tell

**Material:**
```text
Full time: Harbour 2, Northgate 1. Vale scored both Harbour goals.
```

**Statement:** Northgate's goal was a penalty.

**Thinking:**
```text
├─ what would decide it? A line on how Northgate's goal came
├─ search: the score and the Harbour scorers — nothing about Northgate's goal at all
└─ the tempting fill is "one-goal games often turn on a penalty". Often is not this record. Refuse.
cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-09 · policy · tier 3 · refuses-the-gap · cannot_tell

**Material:**
```text
Booking travel: use the company portal. Rail fares are reimbursed at standard class.
```

**Statement:** Taxi fares are reimbursed as well.

**Thinking:**
```text
├─ what would decide it? A line about taxis, or an all-travel rule
├─ search: the portal for booking, and rail at standard class — taxis never appear
└─ the tempting fill is to extend the rail rule to taxis; that extension would be mine, not the policy's. Refuse.
cannot_tell.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-10 · record · tier 4 · refuses-the-gap · cannot_tell

**Material:**
```text
Lab book, run 14: temperature held at 21°C for the full 6 hours. Samples logged hourly.
```

**Statement:** Run 14 used the new thermostat and held 21°C for 6 hours.

**Thinking:**
```text
ROOT  new thermostat + 21°C held for 6 hours                    [open]
├─ B1 new thermostat ← no line mentions a thermostat, new or old   [?]
└─ B2 21°C for 6 h   ← "temperature held at 21°C for the full 6 hours"   ✓
B1 is unsettled, and no falsifier exists for it either. Filling it in ("labs have thermostats") would quote nothing. One backed part does not close the gap: cannot_tell, DECIDING NONE.
```

**Answer:**
```text
VERDICT: cannot_tell
DECIDING: NONE
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-11 · notice · tier 3 · quotes-only-what-is-there · supported

**Material:**
```text
Hall hire: £40 an hour, minimum two hours.
```

**Statement:** Hiring the hall costs £40 an hour with a two-hour minimum.

**Thinking:**
```text
One leaf  ← "Hall hire: £40 an hour, minimum two hours."  ✓
Quote it as written — no rounding, no invented booking fee. supported, and stop.
```

**Answer:**
```text
VERDICT: supported
DECIDING: Hall hire: £40 an hour, minimum two hours.
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-12 · notice · tier 3 · quotes-only-what-is-there · unsupported

**Material:**
```text
Hall hire: £40 an hour, minimum two hours.
```

**Statement:** Hall hire is £25 an hour.

**Thinking:**
```text
├─ leaf: £25 an hour
├─ the words "£25" appear nowhere in the material; the rate line reads £40   ✗
└─ quote the line that exists, never the one the statement asks for
unsupported.
```

**Answer:**
```text
VERDICT: unsupported
DECIDING: Hall hire: £40 an hour, minimum two hours.
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-13 · label · tier 3 · quotes-only-what-is-there · supported

**Material:**
```text
Pack of 6 bars, 35 g each. Serving suggestion.
```

**Statement:** Each bar weighs 35 g.

**Thinking:**
```text
One leaf  ← "Pack of 6 bars, 35 g each."  ✓ → stop
The label's own words decide it. supported.
```

**Answer:**
```text
VERDICT: supported
DECIDING: Pack of 6 bars, 35 g each.
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-14 · label · tier 3 · quotes-only-what-is-there · unsupported

**Material:**
```text
Cafetière: borosilicate glass, capacity 1 litre.
```

**Statement:** The cafetière holds 1.5 litres.

**Thinking:**
```text
├─ leaf: holds 1.5 litres  ← "capacity 1 litre."   ✗
└─ 1 litre is not 1.5. Quote the real figure; never patch the gap by guessing a larger size
unsupported.
```

**Answer:**
```text
VERDICT: unsupported
DECIDING: capacity 1 litre.
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-15 · policy · tier 4 · quotes-only-what-is-there · supported

**Material:**
```text
IT helpdesk: priority 1 tickets are answered within 1 hour. All other tickets are answered within 2 working days.
```

**Statement:** Priority 1 tickets are answered within an hour; all other tickets within 2 working days.

**Thinking:**
```text
├─ B1 priority 1 within an hour ← "priority 1 tickets are answered within 1 hour."     ✓
└─ B2 the rest within 2 days    ← "All other tickets are answered within 2 working days."  ✓
Both backed by the policy's own words, quoted line for line, not paraphrased. supported.
```

**Answer:**
```text
VERDICT: supported
DECIDING: priority 1 tickets are answered within 1 hour. | All other tickets are answered within 2 working days.
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-16 · policy · tier 3 · quotes-only-what-is-there · unsupported

**Material:**
```text
IT helpdesk: priority 1 tickets are answered within 1 hour. All other tickets are answered within 2 working days.
```

**Statement:** Priority 2 tickets are answered within the hour.

**Thinking:**
```text
├─ the policy sets two classes: priority 1, and all other tickets
├─ priority 2 sits inside "all other" → 2 working days, not 1 hour    ✗
└─ quote "All other tickets are answered within 2 working days."; do not invent a priority-2 row that the policy never printed
unsupported.
```

**Answer:**
```text
VERDICT: unsupported
DECIDING: All other tickets are answered within 2 working days.
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-17 · timetable · tier 3 · quotes-only-what-is-there · unsupported

**Material:**
```text
Central Library: Monday to Friday 09:00 to 19:00; Saturday 10:00 to 16:00; closed Sunday.
```

**Statement:** The library opens at 09:30 on Saturdays.

**Thinking:**
```text
├─ leaf: Saturday opening time  ← "Saturday 10:00 to 16:00"   ✗
└─ 09:30 is not 10:00. Quote the row as printed, not as wished, and stop
unsupported.
```

**Answer:**
```text
VERDICT: unsupported
DECIDING: Saturday 10:00 to 16:00
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-18 · code · tier 4 · quotes-only-what-is-there · supported

**Material:**
```text
MAX_UPLOAD_MB = 40
ALLOWED_TYPES = ["png", "jpg", "webp"]
```

**Statement:** The uploader accepts png, jpg and webp files up to 40 MB.

**Thinking:**
```text
├─ file types ← ALLOWED_TYPES = ["png", "jpg", "webp"]   ✓
└─ size cap   ← MAX_UPLOAD_MB = 40                        ✓
Two lines, both quoted verbatim — no extra type added, no cap rounded. supported.
```

**Answer:**
```text
VERDICT: supported
DECIDING: MAX_UPLOAD_MB = 40 | ALLOWED_TYPES = ["png", "jpg", "webp"]
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-19 · puzzle · tier 4 · quotes-only-what-is-there · unsupported

**Material:**
```text
Keys: the red key fits the east door. The blue key fits the west door.
```

**Statement:** The red key opens the west door.

**Thinking:**
```text
├─ red fits east  ← "the red key fits the east door."   ✓ as far as it goes
└─ west takes blue ← "The blue key fits the west door."  ✗ the red key is not the west door's key
The two lines together decide it. Quoting a made-up "red key, west door" line would be filling, not verifying.
unsupported.
```

**Answer:**
```text
VERDICT: unsupported
DECIDING: the red key fits the east door. | The blue key fits the west door.
ESCALATE: no
```

*Invented material, key known by construction.*

### t14-gf-20 · record · tier 5 · quotes-only-what-is-there · supported

**Material:**
```text
Shift report, line 2, 6 June: 1,180 cartons packed and 12 damaged; conveyor stopped 09:10 to 09:35.
```

**Statement:** On 6 June, line 2 packed 1,180 cartons, damaged 12, and its conveyor was stopped for 25 minutes.

**Thinking:**
```text
ROOT  1,180 packed + 12 damaged + a 25-minute stop           [open]
├─ packed 1,180 ← "1,180 cartons packed"                          ✓
├─ damaged 12   ← "12 damaged"                                    ✓
└─ stop 25 min  ← "conveyor stopped 09:10 to 09:35"  → 09:10 to 09:35 is 25 minutes   ✓
All three backed by the one report line, quoted whole. supported.
```

**Answer:**
```text
VERDICT: supported
DECIDING: Shift report, line 2, 6 June: 1,180 cartons packed and 12 damaged; conveyor stopped 09:10 to 09:35.
ESCALATE: no
```

*Invented material, key known by construction.*
