# Lesson 2: As a verifier, this is how you think

As a verifier, your thinking is your method. The verdict at the end is only as good as the steps that led
to it. This lesson teaches five patterns of verifier thinking. Each one is drawn out so you can see its
shape, and each is worked through on a different kind of material. You'll use all of them in every lesson
that follows.

---

## Pattern 1: The tree, with pruning

As a verifier, you treat a statement as a tree:
- **The root** is the whole statement.
- **The branches** are its separate claims.
- **The leaves** are the points you can check against one line of the material.

You check leaves one at a time. As soon as one leaf is false, the root is settled, and you prune the rest:
there's no need to check branches that can't change the answer. When every leaf is true, the root is
supported.

**Worked thinking (a changelog).** Material:

```
v2.3.0
- Added export to PNG.
- Fixed a crash when saving an empty file.
- Removed the legacy JSON importer.
```

Statement: "Version 2.3.0 added PNG export, fixed the empty-file crash, and kept the JSON importer."

```
ROOT  v2.3.0: PNG export + crash fix + JSON importer kept         [open]
├─ B1 added PNG export          ← "Added export to PNG."            ✓
├─ B2 fixed empty-file crash    ← "Fixed a crash when saving an
│                                  empty file."                     ✓
└─ B3 kept the JSON importer    ← "Removed the legacy JSON
                                   importer."                       ✗  → ROOT ✗
```

```
VERDICT: unsupported
DECIDING: Removed the legacy JSON importer.
ESCALATE: no
```

**Pruning, shown.** Same material. Statement: "Version 2.3.0 removed the JSON importer, added PNG export,
and fixed the empty-file crash." That's three branches. All three are backed, so nothing is pruned, and
the verdict is supported. Now flip it. Statement: "v2.3.0 kept the JSON importer and added PNG export."
B1, "kept the JSON importer", is ✗ against "Removed the legacy JSON importer." The root is settled
there, so B2 is pruned and never checked.

---

## Pattern 2: Look for the falsifier first

As a verifier, before you accept a statement, you ask: *what would make this false, and is it in the
material?* You look for that first. A statement earns "supported" only when you've searched for its
falsifier and found nothing. This turns skepticism into a step you actually perform, not a mood.

**Worked thinking (a returns policy, several paragraphs).** Material:

> Returns. You may return any item within 30 days of delivery for a full refund.
>
> Exceptions. Sale items can be exchanged but not refunded. Opened software cannot be returned.
>
> How to return. Use the prepaid label in your parcel.

Statement: "You can get a full refund on a sale item within 30 days."

```
1. What would make it false?  → a rule saying sale items can't be refunded.
2. Search for it.             → "Sale items can be exchanged but not refunded."  FOUND
3. Falsifier found → stop.
```

```
VERDICT: unsupported
DECIDING: Sale items can be exchanged but not refunded.
ESCALATE: no
```

Notice that the first paragraph, read alone, seems to back the statement ("any item within 30 days"). Only
the falsifier search finds the exception two paragraphs down.

---

## Pattern 3: Elimination

As a verifier, when the material describes a set of possibilities, you rule them out one by one until only
the answer is left, or until none is.

**Worked thinking (a game design note).** Material:

> Units: Knight, Archer, Healer. Only the Archer attacks from range. The Healer cannot attack. The Knight
> attacks only adjacent enemies.

Statement: "A unit that can attack an enemy three tiles away is the Knight."

```
Who can attack at range 3?
- Healer  → "The Healer cannot attack."                   eliminated
- Knight  → "The Knight attacks only adjacent enemies."    eliminated
- Archer  → "Only the Archer attacks from range."          remains
Only the Archer remains, not the Knight.
```

```
VERDICT: unsupported
DECIDING: Only the Archer attacks from range.
ESCALATE: no
```

Also sufficient on its own: "The Knight attacks only adjacent enemies." Keys list every sufficient line.

---

## Pattern 4: The constraint grid

As a verifier, when the material sets several conditions that must all hold, you lay them out in a grid
and check the case against each one. The first condition that fails decides it.

**Worked thinking (a booking rule).** Material: "Meeting rooms can be booked by staff, for up to 2 hours,
between 9am and 5pm on weekdays."

Statement: "A staff member can book a room for 3 hours on Wednesday from 10am."

| Condition | The case | Holds? |
|---|---|---|
| booked by staff | a staff member | ✓ |
| up to 2 hours | 3 hours | **✗** → stop |
| 9am–5pm | 10am to 1pm | (not needed) |
| weekdays | Wednesday | (not needed) |

```
VERDICT: unsupported
DECIDING: for up to 2 hours
ESCALATE: no
```

---

## Pattern 5: When to stop

As a verifier, you stop when the answer is settled. That means:
- when one false leaf decides the root;
- when every leaf is backed;
- when you've confirmed the material is silent on the deciding point.

You don't re-read a line you've already settled, and you don't hunt for doubt after the answer is clear.
Stopping isn't laziness: it's what lets you give the hard cases the time they need.

**Short trace and long trace, side by side.** The effort fits the problem.

*Short.* Material: a canon entry: "Captain Reya Voss commands the starship Meridian." Statement: "Reya Voss
commands the Meridian."

```
One leaf: commands the Meridian ← "Captain Reya Voss commands the starship Meridian."  ✓  → stop
VERDICT: supported
DECIDING: Captain Reya Voss commands the starship Meridian.
ESCALATE: no
```

*Long.* Material, from the same canon entry: "Captain Reya Voss commands the starship Meridian. She was
born on Kelt and served twelve years in the Coastal Guard before taking command. Her first officer is
Danek Orr." Statement: "Reya Voss, born on Kelt, served in the Coastal Guard for twelve years and now
serves as first officer of the Meridian."

```
ROOT  born on Kelt + 12 years Coastal Guard + first officer of the Meridian   [open]
├─ B1 born on Kelt               ← "She was born on Kelt"                 ✓
├─ B2 12 years in Coastal Guard  ← "served twelve years in the Coastal
│                                   Guard"                                ✓
└─ B3 first officer of Meridian  ← "commands the starship Meridian";
                                    "Her first officer is Danek Orr."     ✗ → stop
```

```
VERDICT: unsupported
DECIDING: Her first officer is Danek Orr.
ESCALATE: no
```

The short case took one leaf. The long case needed three, and the third needed two lines to settle. Both
stopped the moment the answer was clear.

---

## Test

For each item, write your thinking in the pattern named, then give the three answer lines. Two things are
graded:
- **The verdict and DECIDING:** the verdict must match, and DECIDING must be word for word from the
  material and enough to decide it.
- **The thinking:** each step must be true of the material. A person reads it.

1. **Tree.** Material: "The museum opens at 10am, closes at 6pm, and is closed on Mondays." Statement:
   "The museum opens at 10am, closes at 6pm, and is open on Mondays."
2. **Falsifier first.** Material: "All orders ship free. Orders to islands take 5 extra days." Statement:
   "Orders to islands ship free."
3. **Elimination.** Material: "Three keys: the brass key opens the shed, the silver key opens the gate,
   and the iron key opens nothing." Statement: "The iron key opens the gate."
4. **Constraint grid.** Material: "Discount for members, on orders over £20, placed before 31 March."
   Statement: "A member ordering £25 on 2 April gets the discount."
5. **When to stop.** Material: "Parking: free." Statement: "Parking is free." Show that you stop after one
   step.

**Key.**
1. unsupported, "is closed on Mondays". The first two branches are ✓ and the third is ✗.
2. supported, "All orders ship free." The falsifier was searched for; the island line changes delivery
   time, not cost.
3. unsupported, "the iron key opens nothing". Gate → silver key; iron → nothing.
4. unsupported, "placed before 31 March". Member ✓, over £20 ✓, before 31 March ✗.
5. supported, "Parking: free.", in one step.
