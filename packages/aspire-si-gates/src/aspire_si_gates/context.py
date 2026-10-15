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


def _opening(r: dict) -> str:
    words = G.re.findall(r"[a-z']+", r["turns"][0]["user"].lower())[:6]
    return " ".join(words) if len(words) == 6 else ""


def _b_shingles(r: dict) -> set:
    return G._shingles(" ".join(t["thinking"] + " " + t["reply"] for t in r["turns"]))


def check_against(r: dict, accepted: list[dict], pre_amendment=PRE_AMENDMENT_LESSONS,
                  models: dict | None = None) -> list[tuple]:
    """[(code, message)] for every exclusivity rule the row breaks against the accepted rows, one example per code,
    in a fixed order; [] when it may be accepted. Run check_item on the row as well. `models` is the {id: model}
    map of sidecar models (the new row's and the accepted rows'): puzzles are signed by gates.puzzle_sig, as the
    batch gate signs them."""
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
        sig = G.puzzle_sig(r, models)
        if any(G._is_puzzle(a) and G.puzzle_sig(a, models) == sig for a in same):
            out.append(_reason("PUZZLE_SHAPE_REUSED", G.describe_signature(sig)))
        elif sum(1 for a in accepted if a.get("lesson") not in pre_amendment and G._is_puzzle(a)
                 and G.puzzle_sig(a, models) == sig) >= 2:
            out.append(_reason("PUZZLE_SHAPE_BATCH", G.describe_signature(sig)))
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



def check_not_from_traces(r: dict, traces, accepted=()) -> list[tuple]:
    """[("TRACE_REUSED", message)] when an 8-word span that originated with a model appears in the row; [] otherwise.
    Traces (the student's or a judge's output, from the role store) are observations, never item text.

    One machinery with the diversity gate (R&D, 2026-10-15): the row's spans are gates._spans (material lines, the
    Statement line and the three-line block left out, digits kept, a bracketed quote as one token), and each trace
    line is tokenized by the same gates.line_spans. A trace span counts only when no accepted row holds it: a trained
    student reproduces the dataset's own wording, and reuse of that is SPAN_REUSED's job, not this gate's. Spans of
    the row's own material lines don't count either (a trace quotes the material it was given). Every lesson is
    checked."""
    mine = G._spans(r)
    if not mine:
        return []
    learned = set().union(*(G._spans(a) for a in accepted)) if accepted else set()
    own = set().union(*(G.line_spans(x) for x in G._content_lines(r).splitlines() or [""]))
    hit = set()
    for t in traces:
        for line in (t or "").splitlines():
            hit |= G.line_spans(line) & mine
    hit -= learned | own
    return [_reason("TRACE_REUSED", min(hit))] if hit else []
