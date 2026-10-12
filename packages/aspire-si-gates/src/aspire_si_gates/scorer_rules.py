"""The frozen scorer's DECIDING rules (docs/runs/stage1-eval/score.py, sha256 fd59cb50...), for the critic.

norm, ident, core, lines_of and deciding_ok are copies of score.py's functions, which is hash-frozen with the
evaluation and can't import this package; a test in docs/runs/stage1-train asserts each copy's source equals the
original. critic_mismatch is the critic's refusal rule (R&D): the key's lines and the critic's lines share no line,
and neither covers the other under deciding_ok.
"""

import re


def norm(x: str) -> str:
    return re.sub(r"\s+", " ", x.replace("**", "").replace("__", "")).strip()


def ident(x: str) -> str:
    return x


def core(x: str) -> str:
    """Coverage only (not the substring-of-material check): leading list/quote markers and trailing
    punctuation don't decide whether a key line was given. Proposed to R&D 2026-10-10 after the baseline."""
    return re.sub(r"^(?:[-*>]\s+)+", "", x).strip().rstrip(".;,:").strip()


def lines_of(deciding):
    if not deciding or deciding.strip().upper() == "NONE":
        return []
    return [s for s in (x.strip() for x in deciding.split(" | ")) if s]


def deciding_ok(given, key_lines, alt_sets, material, verdict_key, f) -> bool:
    if verdict_key == "cannot_tell":
        return not given
    if not given:
        return False
    mat = f(material)
    g = [f(x) for x in given]
    if not all(x and x in mat for x in g):
        return False
    joined = core(f(" | ".join(given)))  # a table row contains " | " itself
    for want in [key_lines] + list(alt_sets):
        w = [core(f(x)) for x in want]
        if all(k in joined or any(k in core(x) for x in g) for k in w):  # one way only: given contains key
            return True
    return False


def _same(a: str, b: str) -> bool:
    return core(norm(a)) == core(norm(b))


def deciding_covers(covering: list, covered: list, material: str) -> bool:
    """True when the `covering` lines are verbatim in the material (normalised) and cover every `covered` line,
    under the frozen scorer's rule (one-way: a covering line contains the covered line)."""
    return deciding_ok(list(covering), list(covered), [], material, "supported", norm)


def critic_mismatch(key_lines: list, critic_lines: list, material: str) -> bool:
    """The critic's DECIDING disagrees with the key: no shared line, and neither set covers the other. Two
    different but overlapping sufficient sets are not a mismatch. Empty on both sides (NONE) is agreement; NONE on
    one side only is a mismatch."""
    if not key_lines and not critic_lines:
        return False
    if not key_lines or not critic_lines:  # one side NONE, the other quotes lines
        return True
    if any(_same(k, c) for k in key_lines for c in critic_lines):
        return False
    return not (deciding_covers(critic_lines, key_lines, material) or deciding_covers(key_lines, critic_lines, material))
