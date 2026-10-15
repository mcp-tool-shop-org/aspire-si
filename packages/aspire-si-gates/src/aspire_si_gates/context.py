"""check_against: the refusals a new row earns from rows already accepted, one at a time.

The exclusivity checks of check_batch, made incremental, each with a stable reason code and a short message that a
generator hands back to its writer verbatim. Share-based checks (phrase frequency, formulas, floors, pairs) need the
finished set and stay in check_batch.
"""

from . import gates as G
from .constants import PRE_AMENDMENT_LESSONS

# Code -> message template. The codes never change; a message changes only on purpose (a test pins this table).
REASONS = {
    "DUPLICATE_ID": "the id {x} is already used",
    "SPAN_REUSED": "the wording '{x}' is already in 2 items of this trait; reword it",
    "FRAMING_REUSED": "the framing '{x}' is already used; write a new one",
    "MATERIAL_REUSED": "this material is already used; take another excerpt",
    "PUZZLE_SHAPE_REUSED": "this trait already has a puzzle shaped {x}; use another structure",
    "PUZZLE_SHAPE_BATCH": "two puzzles in this batch are already shaped {x}; use another structure",
    "OPENING_REUSED": "the opening '{x}' already opens 2 items of this trait; vary it",
    "B_PHRASE_REUSED": "the phrase '{x}' is already in 2 kind B items of this trait; reword it",
    "TRACE_REUSED": "the wording '{x}' comes from a stored model trace; item text is never drawn from traces",
}


def _reason(code: str, x="") -> tuple:
    return code, REASONS[code].format(x=x)


def _shape(sig) -> str:
    return f"{sig[0]} entities, {' + '.join(sig[1]) or 'no cues'}"


def _opening(r: dict) -> str:
    words = G.re.findall(r"[a-z']+", r["turns"][0]["user"].lower())[:6]
    return " ".join(words) if len(words) == 6 else ""


def _b_shingles(r: dict) -> set:
    return G._shingles(" ".join(t["thinking"] + " " + t["reply"] for t in r["turns"]))


def check_against(r: dict, accepted: list[dict], pre_amendment=PRE_AMENDMENT_LESSONS) -> list[tuple]:
    """[(code, message)] for every exclusivity rule the row breaks against the accepted rows, one example per code,
    in a fixed order; [] when it may be accepted. Run check_item on the row as well."""
    out = []
    if r.get("id") in {a.get("id") for a in accepted}:
        out.append(_reason("DUPLICATE_ID", r.get("id")))
    lesson = r.get("lesson")
    if lesson in pre_amendment:
        return out
    same = [a for a in accepted if a.get("lesson") == lesson]
    # 8-word spans: at most 2 items of a lesson
    counts: dict = {}
    for a in same:
        for sp in G._spans(a):
            counts[sp] = counts.get(sp, 0) + 1
    hit = sorted(sp for sp in G._spans(r) if counts.get(sp, 0) >= 2)
    if hit:
        out.append(_reason("SPAN_REUSED", hit[0]))
    # framing: unique across every accepted row
    f = G.framing_of(r)
    if f and any(G.framing_of(a) == f for a in accepted if a.get("lesson") not in pre_amendment):
        out.append(_reason("FRAMING_REUSED", f))
    # material: once, or twice as a contrast pair
    m = G.ws(r.get("material") or "").lower()
    if m:
        users = [a for a in same if G.ws(a.get("material") or "").lower() == m]
        if len(users) >= 2 or (users and (r.get("pair") is None or users[0].get("pair") != r.get("pair"))):
            out.append(_reason("MATERIAL_REUSED"))
    # puzzle structure: once per lesson, twice per batch
    if G._is_puzzle(r):
        sig = G.puzzle_signature(r)
        if any(G._is_puzzle(a) and G.puzzle_signature(a) == sig for a in same):
            out.append(_reason("PUZZLE_SHAPE_REUSED", _shape(sig)))
        elif sum(1 for a in accepted if a.get("lesson") not in pre_amendment and G._is_puzzle(a)
                 and G.puzzle_signature(a) == sig) >= 2:
            out.append(_reason("PUZZLE_SHAPE_BATCH", _shape(sig)))
    # the 6-word opening of a C, D or E user turn: at most 2 items of a lesson
    if r.get("kind") in ("C", "D", "E"):
        op = _opening(r)
        if op and sum(1 for a in same if a.get("kind") in ("C", "D", "E") and _opening(a) == op) >= 2:
            out.append(_reason("OPENING_REUSED", op))
    # kind B 8-word phrases: at most 2 B items of a lesson
    if r.get("kind") == "B":
        bc: dict = {}
        for a in same:
            if a.get("kind") == "B":
                for sh in _b_shingles(a):
                    bc[sh] = bc.get(sh, 0) + 1
        hit = sorted(sh for sh in _b_shingles(r) if bc.get(sh, 0) >= 2)
        if hit:
            out.append(_reason("B_PHRASE_REUSED", hit[0]))
    return out


def _row_text(r: dict) -> str:
    return "\n".join(t.get(k) or "" for t in r.get("turns") or [] for k in ("user", "thinking", "reply"))


def check_not_from_traces(r: dict, traces) -> list[tuple]:
    """[("TRACE_REUSED", message)] when any 8-word span of a stored model trace (the student's or a judge's output,
    from the role store) appears in the row's turns; [] otherwise. Traces are observations, never item text.

    Spans the row's own material lines also hold are not counted: a trace quotes the material it was given, and the
    material is checked against its cited source by the source gates. Every lesson is checked; the pre-amendment
    exemptions don't apply to new rows."""
    own = G._shingles(G._content_lines(r))
    mine = G._shingles(_row_text(r)) - own
    if not mine:
        return []
    hit = set()
    for t in traces:
        hit |= mine & G._shingles(t or "")
    return [_reason("TRACE_REUSED", min(hit))] if hit else []
