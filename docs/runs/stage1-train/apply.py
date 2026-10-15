"""The Stage 1 apply sitting (run sheet B6; schema: docs/runs/2026-10-15-stage1-apply-items-schema.md, R&D-approved).

    python apply.py sit  --adapter <round dir>/adapter --items <apply files...> --session N --round N --arm A
                         --seed S [--star <earlier sessions' STaR files...>] [--out-root outputs/stage1-train]
    python apply.py star --session N --kept <arm ROLE kept.jsonl> <arm NO-ROLE kept.jsonl>
                         [--prior <earlier sessions' STaR files...>] --out <the session's STaR file>

**sit** asks each apply item's first user turn once, on the batched path (B4), in one arm. Each item draws from its
own generator, seeded by (item id and session, sample 0). The sitting then:
- scores the three lines with the pinned scorer, against the item's key (`quiz.score_three_lines`);
- judges stuck by the frozen rule (716dc251), with the item's reference thinking as its target;
- writes a `rehearsal` receipt per item, carrying `item_trained`: true when an earlier session's STaR file holds a
  kept answer of this item (R&D's addition 2);
- writes the arm's kept candidates.

**The keep rule (amendment A).** A candidate needs all three of:
- all three lines pass;
- the lesson's pass rule is met;
- the trace isn't stuck.

Most lesson pass rules are set-level: twins, pairs, or rank checks across items (the lesson plan's quiz pass
rules). One apply item can't show them, so they're "not yet testable" and never approximated (R&D, 2026-10-11).
`LESSON_RULES` maps a lesson to an item-level check where one exists; it's empty until R&D rules. Until then a
candidate whose lesson has no item-level rule is **held**, not kept, and the reason is recorded.

**star** matches and caps across the arms (amendment A) and writes the session's STaR rows (R&D's addition 1).
For each (lesson, kind):
- near-duplicates are dropped first: a candidate sharing an 8-word span with an earlier one of its arm;
- both arms are cut to the smaller count, and to what's left of the cumulative cap of 8 after earlier sessions;
- the candidates kept are the first by item id.

Each becomes a `split: "star"` row, with id `<lesson>-<kind>-ST-<session>-<NN>` and `star_of` set to the item. The
user turn is the item's, and the thinking and reply are the model's. The file's sha256 is printed, for the next
session's manifest. Loading star rows in train.py, and its refusal of apply rows, is a separate change (train.py is
in review elsewhere).
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import quiz  # noqa: E402  (puts the harness, the scorer and the gates on the path)
from aspire_si_gates import gates  # noqa: E402

harness = quiz.harness
KINDS = ("C", "D")
CAP = 8  # kept answers per (lesson, kind), cumulative over the stage (amendment A)
LESSON_RULES: dict = {}  # lesson -> callable(item, answer) -> bool; item-level pass rules only, as R&D rules them


class ApplyItemError(ValueError):
    pass


def check_item(r: dict) -> None:
    """An apply item in the schema's shape: split apply, an -AP- id, kind C or D, one turn, the three keys."""
    where = r.get("id", "?")
    if r.get("split") != "apply" or "-AP-" not in str(r.get("id")):
        raise ApplyItemError(f"{where}: not an apply item (split apply and an <lesson>-<kind>-AP-<NN> id)")
    if r.get("kind") not in KINDS or str(r["id"]).split("-")[1] != r["kind"]:
        raise ApplyItemError(f"{where}: kind must be C or D, and the id's second part")
    if len(r.get("turns") or []) != 1:
        raise ApplyItemError(f"{where}: one turn (the reference answer is that turn's thinking and reply)")
    for k in ("verdict", "deciding", "escalate"):
        if k not in r:
            raise ApplyItemError(f"{where}: no {k} key")
    if r["kind"] == "C" and "planted_flaw" not in r:
        raise ApplyItemError(f"{where}: a C item needs planted_flaw")


def load_items(paths) -> list:
    rows = [json.loads(x) for p in paths for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
    for r in rows:
        check_item(r)
    if len({r["id"] for r in rows}) != len(rows):
        raise ApplyItemError("duplicate apply ids")
    return rows


def trained_items(star_paths) -> set:
    """Apply item ids with a kept answer in an earlier session's STaR file."""
    return {json.loads(x)["star_of"] for p in star_paths
            for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()}


def key_of(r: dict) -> dict:
    return {"verdict": r["verdict"], "deciding": r.get("deciding") or [],
            "also_sufficient": r.get("also_sufficient", []), "escalate": r.get("escalate", "no")}


def sit(items: list, session: int, generate_many, count, rule: dict, write, trained: set) -> tuple[dict, list]:
    """One arm's apply sitting. Returns (readout, kept candidates)."""
    asks = [{"key": r["id"], "item_id": f"{r['id']}@session{session}", "sample": 0,
             "messages": [{"role": "user", "content": r["turns"][0]["user"]}]} for r in items]
    gens = generate_many(asks)
    cells: dict = {}
    kept = []
    for r in items:
        g = gens[r["id"]]
        thinking, answer, _ = harness.split_thinking(g["text"])
        s = quiz.score_three_lines(key_of(r), r.get("material") or "", answer)
        target = count(r["turns"][0].get("thinking", ""))
        f = quiz.stuck_fields(thinking, r.get("material") or "", r["tier"], target, g["new_tokens"], count, rule)
        lesson_rule = LESSON_RULES.get(r["lesson"])
        lesson_ok = lesson_rule(r, answer) if lesson_rule else None
        if not s["right"]:
            star = "refused: the three lines don't pass"
        elif f["stuck"]:
            star = "refused: stuck"
        elif lesson_ok is None:
            star = "held: the lesson's pass rule isn't testable on one item"
        elif not lesson_ok:
            star = "refused: the lesson's pass rule"
        else:
            star = "candidate"
            kept.append({"item": r, "thinking": thinking, "answer": answer})
        item_trained = r["id"] in trained
        write({"question_id": r["id"], "base_item_id": r["id"], "lesson": r["lesson"], "kind": r["kind"],
               "tier": r["tier"], "session": session, "item_trained": item_trained, **s, **f,
               "lesson_rule": "not yet testable" if lesson_ok is None else lesson_ok, "star": star,
               "flaw_named": "not scored" if r["kind"] == "C" else None, "total_tokens": g["new_tokens"],
               "truncated": g["truncated"], "answer": answer})
        c = cells.setdefault(f"{r['lesson']}|{r['kind']}", {"asked": 0, "right": 0, "stuck": 0,
                                                            "by_item_trained": {"true": [0, 0], "false": [0, 0]}})
        c["asked"] += 1
        c["right"] += s["right"]
        c["stuck"] += f["stuck"]
        c["by_item_trained"][str(item_trained).lower()][0] += 1  # [asked, stuck] per split (R&D's addition 2)
        c["by_item_trained"][str(item_trained).lower()][1] += f["stuck"]
    readout = {"session": session, "items": len(items), "cells": dict(sorted(cells.items())),
               "candidates": len(kept), "c_flaw_naming": "not scored mechanically (disclosed)",
               "lesson_rules_testable": sorted(LESSON_RULES)}
    return readout, kept


def _spans(text: str) -> set:
    return gates._shingles(text, 8)


def match_and_cap(role: list, no_role: list, prior: dict, cap: int = CAP) -> dict:
    """{arm: chosen candidates} after the near-duplicate drop, matching across the arms and the cumulative cap.
    `prior` is {(lesson, kind): kept so far}; candidates are {"item", "thinking", "answer"}."""
    def dedup(cands):
        out, seen = [], set()
        for c in sorted(cands, key=lambda c: c["item"]["id"]):
            sp = _spans(c["thinking"] + "\n" + c["answer"])
            if sp & seen:
                continue
            seen |= sp
            out.append(c)
        return out
    by_cell = {}
    for arm, cands in (("ROLE", role), ("NO-ROLE", no_role)):
        for c in dedup(cands):
            by_cell.setdefault((c["item"]["lesson"], c["item"]["kind"]), {"ROLE": [], "NO-ROLE": []})[arm].append(c)
    chosen = {"ROLE": [], "NO-ROLE": []}
    for cell, arms in sorted(by_cell.items()):
        n = max(0, min(len(arms["ROLE"]), len(arms["NO-ROLE"]), cap - prior.get(cell, 0)))
        for arm in chosen:
            chosen[arm] += arms[arm][:n]
    return chosen


def star_rows(chosen: list, session: int) -> list:
    rows, nn = [], {}
    for c in chosen:
        r = c["item"]
        cell = (r["lesson"], r["kind"])
        nn[cell] = nn.get(cell, 0) + 1
        row = {k: r[k] for k in ("lesson", "trait", "kind", "side", "content_kind", "tier", "material", "sources",
                                  "invented", "key_basis", "tags", "verdict", "deciding", "escalate") if k in r}
        row.update({"id": f"{r['lesson']}-{r['kind']}-ST-{session}-{nn[cell]:02d}", "split": "star",
                    "star_of": r["id"], "turns": [{"user": r["turns"][0]["user"], "thinking": c["thinking"],
                                                   "reply": c["answer"]}]})
        if r["kind"] == "C":
            row["planted_flaw"] = r["planted_flaw"]
        rows.append(row)
    return rows


def prior_counts(star_paths) -> dict:
    out: dict = {}
    for p in star_paths:
        for x in Path(p).read_text(encoding="utf-8").splitlines():
            if x.strip():
                r = json.loads(x)
                out[(r["lesson"], r["kind"])] = out.get((r["lesson"], r["kind"]), 0) + 1
    return out


def main(argv=None) -> int:  # pragma: no cover - the sitting runs on the card; its parts are tested
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sit")
    s.add_argument("--adapter", required=True)
    s.add_argument("--items", nargs="+", required=True)
    s.add_argument("--session", type=int, required=True)
    s.add_argument("--round", type=int, required=True)
    s.add_argument("--arm", choices=("ROLE", "NO-ROLE"), required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--star", nargs="*", default=[])
    s.add_argument("--sealed", default=None)
    s.add_argument("--out-root", default=str(HERE.parents[2] / "outputs" / "stage1-train"))
    t = sub.add_parser("star")
    t.add_argument("--session", type=int, required=True)
    t.add_argument("--kept", nargs=2, required=True, metavar=("ROLE", "NO_ROLE"))
    t.add_argument("--prior", nargs="*", default=[])
    t.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "star":
        def load(p):
            return [json.loads(x) for x in Path(p).read_text(encoding="utf-8").splitlines() if x.strip()]
        chosen = match_and_cap(load(a.kept[0]), load(a.kept[1]), prior_counts(a.prior))
        out = Path(a.out)
        for arm, suffix in (("ROLE", ""), ("NO-ROLE", ".no-role")):
            path = out.with_name(out.stem + suffix + out.suffix)
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                    for r in star_rows(chosen[arm], a.session)), encoding="utf-8", newline="\n")
            print(arm, path.name, hashlib.sha256(path.read_bytes()).hexdigest())
        return 0

    import batched
    import receipts
    import student
    import stuck_adapter
    from aspire_si_gates import gates_identity
    from peft import PeftModel
    items = load_items(a.items)
    rule = stuck_adapter.load_frozen()
    fw = receipts.Firewall.load(**({"sealed": Path(a.sealed)} if a.sealed else {}))
    adapter = Path(a.adapter)
    w = receipts.ReceiptWriter(a.out_root, a.round, a.arm, a.seed, f"r{a.round}-s{a.seed}-{a.arm}",
                               hashlib.sha256((adapter / "adapter_model.safetensors").read_bytes()).hexdigest(),
                               gates_identity()["sha256"], stuck_adapter.RULE_SHA256, fw)
    print(json.dumps(student.verify_weights()), flush=True)
    tok = student.load_tokenizer()
    model = PeftModel.from_pretrained(student.load_student(), str(adapter))
    model.eval()
    log: list = []

    def count(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def write(fields):
        try:
            w.write("rehearsal", fields)
        except receipts.Refused:
            pass  # counted in w.refused
    readout, kept = sit(items, a.session, batched.make_generate_many(model, tok, log), count, rule, write,
                        trained_items(a.star))
    readout.update({"receipts_refused": w.refused, "generation": {"path": batched.PATH, "batches": log}})
    (w.dir.parent / f"apply-readout-session{a.session}.json").write_text(json.dumps(readout, indent=1),
                                                                        encoding="utf-8")
    (w.dir.parent / f"apply-kept-session{a.session}.jsonl").write_text(
        "".join(json.dumps(k, ensure_ascii=False) + "\n" for k in kept), encoding="utf-8", newline="\n")
    print(json.dumps({k: readout[k] for k in ("items", "candidates")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
