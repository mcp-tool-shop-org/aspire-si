"""The verifier role pack's data for the gates: its lessons, trait names, the lessons written before the
amendments, and the role-neutral word list. A different role changes this module, not the checks."""

import re

LESSONS = {"L1", "L2", "L3"} | {f"T{i:02d}" for i in range(1, 19)}

# Names a Kind A teaching passage must use for its trait (lower case; any one is enough).
TRAIT_NAMES = {
    "L1": ["verifier"], "L2": ["think", "pattern"], "L3": ["expected", "rely on", "relying on"],
    "T01": ["skeptic"], "T02": ["to the source", "go to the source", "goes to the source"],
    "T03": ["precise", "precision"], "T04": ["honest about uncertainty", "uncertain"],
    "T05": ["independen"], "T06": ["break things down", "breaks things down", "breaking things down",
                                   "break it down", "decompos"],
    "T07": ["proportion"], "T08": ["your limits", "its limits", "know your limits", "knowing its limits"],
    "T09": ["consistent", "consistency"], "T10": ["literal"],
    "T11": ["what a mistake costs", "cost of a mistake", "mistake costs", "costs of mistakes"],
    "T12": ["version", "date"], "T13": ["agree just to please", "to please", "rubber-stamp", "rubber stamp"],
    "T14": ["fill gaps", "filling gaps", "fill the gap", "fill in gaps"],
    "T15": ["round in circles", "going in circles", "loop"], "T16": ["nitpick"],
    "T17": ["take orders from the material", "orders from the material", "instructions in the material",
            "instructions inside the material"],
    "T18": ["fix things", "fixing", "don't fix", "doesn't fix"],
}


# Amendment 1 (quoting the deciding lines) binds from batch 3 on. The pilot (T04) and batch 2 (T01, T14, T16)
# were written before it and are not redone, so the whole-row, version-heading and multi-line-share checks skip
# these lessons by name. The separator and NONE check applies to every lesson.
PRE_AMENDMENT_LESSONS = frozenset({"T04", "T01", "T14", "T16"})


# Fixed banned list (R&D, 2026-10-10): the role, the 18 trait names and their short forms, and Lesson 2's
# pattern names. "Checker" is allowed: in kind C it names the other party, not the role.
ROLE_WORDS = re.compile(
    r"\b(verifiers?|sceptic\w*|skeptic\w*|goe?s? to the source|going to the source|precise|precision|"
    r"honest about uncertainty|independen\w*|breaks? things down|breaking things down|decompos\w*|"
    r"in proportion|proportionate|knows? (?:its|your|my) limits|consistent|consistency|literal\w*|"
    r"what a mistake costs|agrees? just to please|fill(?:s|ing)? (?:the )?gaps?|round in circles|nitpick\w*|"
    r"takes? orders from the material|fix(?:es|ing)? things|"
    r"trees?|prun\w*|falsif\w*|eliminat\w*|grids?|constraint grids?)\b", re.I)


# The trait-lesson file's section headings ("## 4. As a verifier, you are honest about uncertainty"): lesson T04.
LESSON_HEADING = re.compile(r"(?m)^#{2,3} (\d+)\. As a verifier, (.+)$")
