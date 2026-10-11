"""The repeat-until-correct ledger (Stage 1): Leitner boxes per quiz item, as the lessons draft and the training
procedure set them. Pure bookkeeping, no model: the quiz sitting reports each item's result, and the ledger says
what is due next session and which variant to ask.

Rules (lessons draft, "The repeat-until-correct ledger"; procedure, ledger section and the 2026-10-11 rulings):
- Boxes 1-4. A wrong answer sends the item to box 1, due next session. A right answer moves it up one box, and
  the interval doubles: due again after 1, 2, 4, then 8 sessions.
- Every ask uses a fresh variant of the item's family, never one already shown: n = 0 is the item's own variant,
  then n = 1, 2, ... from role-set's variant store, without replacement. An exhausted family is reported as
  "unmastered: no variant left", never re-asked word for word.
- Item mastered: 3 consecutive right answers spanning at least 2 boxes, and then a fresh, never-seen twin answered
  right (so at least 4 right answers in a row).
- Quality (trait) mastered: at least 95% of its items mastered, and at least 90% right on fresh twins.
- Forgetting events (right, then wrong) are counted per item and per quality.
- Only kind-E items enter the ledger (R&D, 2026-10-11): "missed" exists only where something is quizzed.

The ledger is one JSON file per seed and arm. Its sha256 goes in the round receipts.
"""

import hashlib
import json
from pathlib import Path

INTERVALS = {1: 1, 2: 2, 3: 4, 4: 8}
STREAK = 3
STREAK_BOXES = 2
ITEM_SHARE = 0.95
TWIN_SHARE = 0.90
EXHAUSTED = "unmastered: no variant left"


class Ledger:
    def __init__(self, items: dict | None = None):
        self.items = items or {}

    # -- persistence

    @classmethod
    def load(cls, path: Path):
        p = Path(path)
        return cls(json.loads(p.read_text(encoding="utf-8"))["items"]) if p.exists() else cls()

    def dumps(self) -> str:
        return json.dumps({"schema": "stage1-ledger-v1", "items": self.items}, sort_keys=True, indent=1)

    def save(self, path: Path) -> str:
        p = Path(path)
        tmp = p.with_name(p.name + ".tmp")
        text = self.dumps()
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(p)
        return hashlib.sha256(text.encode()).hexdigest()

    # -- enrolment and scheduling

    def enrol(self, base_id: str, lesson: str, session: int):
        """A quiz item enters at box 1, due at `session`. Enrolling twice changes nothing."""
        self.items.setdefault(base_id, {"lesson": lesson, "box": 1, "due": session, "history": [],
                                        "shown": [], "streak": [], "mastered": False, "mastered_at": None,
                                        "forgetting": 0, "status": "open"})

    def due(self, session: int) -> list:
        return sorted(k for k, v in self.items.items()
                      if v["status"] == "open" and not v["mastered"] and v["due"] <= session)

    def next_variant(self, base_id: str, available: list) -> int | None:
        """The fresh variant to ask: the lowest n in `available` (the variant store's n for this family, with 0
        for the item's own) not yet shown. None, and the item is closed as exhausted, when none is left."""
        it = self.items[base_id]
        fresh = sorted(n for n in available if n not in it["shown"])
        if not fresh:
            it["status"] = EXHAUSTED
            return None
        return fresh[0]

    def spend(self, base_id: str, variant: int, reason: str):
        """A variant that can't be asked (its text was trained on) is used up without a result."""
        it = self.items[base_id]
        it["shown"].append(variant)
        it.setdefault("spent", []).append({"variant": variant, "reason": reason})

    def record(self, base_id: str, session: int, variant: int, right: bool) -> dict:
        """One quiz result (already the majority of 3). Returns the item's new state."""
        it = self.items[base_id]
        if variant in it["shown"]:
            raise ValueError(f"{base_id}: variant {variant} was already shown; asks are never repeated")
        box = it["box"]
        was_right = bool(it["history"]) and it["history"][-1]["right"]
        twin = len(it["streak"]) >= STREAK and len({b for b in it["streak"]}) >= STREAK_BOXES
        it["shown"].append(variant)
        it["history"].append({"session": session, "variant": variant, "box": box, "right": right,
                              "twin": twin})
        if right:
            if twin:  # the fresh twin after a qualifying streak
                it["mastered"], it["mastered_at"] = True, session
            it["streak"].append(box)
            it["box"] = min(4, box + 1)
            it["due"] = session + INTERVALS[it["box"]]
        else:
            if was_right:
                it["forgetting"] += 1
            it["streak"] = []
            it["box"] = 1
            it["due"] = session + 1
        return self.state(base_id)

    def state(self, base_id: str) -> dict:
        """The receipt's ledger_state for this item."""
        it = self.items[base_id]
        return {"box": it["box"], "misses": sum(1 for h in it["history"] if not h["right"]),
                "streak": len(it["streak"]), "due_session": it["due"], "mastered": it["mastered"],
                "forgetting": it["forgetting"], "status": it["status"]}

    # -- reports

    def qualities(self) -> dict:
        """Per lesson: items, mastered, the share mastered, twin accuracy, forgetting events, and whether the
        quality is mastered (95% of items and 90% on twins)."""
        out: dict = {}
        for k, v in self.items.items():
            q = out.setdefault(v["lesson"], {"items": 0, "mastered": 0, "twins": 0, "twins_right": 0,
                                             "forgetting": 0, "exhausted": 0})
            q["items"] += 1
            q["mastered"] += v["mastered"]
            q["forgetting"] += v["forgetting"]
            q["exhausted"] += v["status"] == EXHAUSTED
            for h in v["history"]:
                if h["twin"]:
                    q["twins"] += 1
                    q["twins_right"] += h["right"]
        for q in out.values():
            q["share_mastered"] = round(q["mastered"] / q["items"], 3)
            q["twin_accuracy"] = round(q["twins_right"] / q["twins"], 3) if q["twins"] else None
            q["quality_mastered"] = q["share_mastered"] >= ITEM_SHARE and (q["twin_accuracy"] or 0) >= TWIN_SHARE
        return dict(sorted(out.items()))
