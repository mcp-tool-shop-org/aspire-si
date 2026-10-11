"""The Stage 1 quiz sitting: after a session's training, the ledger's due kind-E items are asked as fresh variants,
scored by code, and recorded (ledger, receipts, readout). No judge, no training: it only asks and records.

    python quiz.py --adapter <round dir>/adapter --data <items.jsonl> ... --ledger <ledger.json> --session N
                   --round N --arm ROLE --seed S [--variants <store>.jsonl ...] [--out-root outputs/stage1-train]

The rules (training procedure; R&D and the Dataset Specialist, 2026-10-11):
- Only kind-E items enter the ledger. The first ask is the item's own variant (n = 0), never the trained turn
  (a test in test_order.py checks no variant text is trained on). Later asks draw n = 1, 2, ... from role-set's
  variant store ({base_id, n, user, key}), without replacement; an exhausted family is reported unmastered.
- A verdict variant is shown with the item's material, in the original turn's position: the original user turn
  up to the end of its material, then the variant's text as the re-asked question. It passes when all three
  lines pass against the base item's quiz key: the verdict and DECIDING by the pinned scorer (score.py, norm),
  and ESCALATE as the harness grades it. The verdict alone is recorded beside it.
- A written variant is quizzed only in closed form (key {"answer": letter}; amendment 5): shown alone, it passes
  when the letter on the last ANSWER line is the key. A free-text written variant is "not yet testable": it stays
  out of the ledger and is counted per lesson in the readout.
- A variant whose wording outside material shares an 8-word span with any trained turn (user, thinking or reply
  of any training item) is spent: it was trained on, so it can't be a fresh ask (R&D). The sitting moves to the
  next variant, and the readout lists every spent variant. Material is shown on purpose, so its spans don't count.
- Each question is answered 3 times at the card's sampling settings (seeds 0-2, the harness's generate,
  unchanged); it is right when at least 2 of 3 are. Every sample is a receipt row.
- Every sample is judged by the frozen stuck rule (stuck.py with the frozen rule file, mapped by stuck_adapter),
  passively.
- The readout reports accuracy at n = 0 beside n >= 1, per lesson and overall (a gap is the sign of surface
  matching; R&D, reporting only), the written E left out per lesson, the qualities' mastery, and STaR as off
  until apply items exist.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
import data  # noqa: E402  (puts the gates package on the path)
import harness  # noqa: E402
import score  # noqa: E402
import stuck  # noqa: E402
import stuck_adapter  # noqa: E402
from aspire_si_gates import parse_choice  # noqa: E402
from ledger import Ledger  # noqa: E402

SEEDS = (0, 1, 2)
MAJORITY = 2
WINDOWS = (100, 200, 400)


class QuizItemError(Exception):
    """An item the sitting can't ask as built (reported, never approximated)."""


def quiz_type(item: dict, variant: dict) -> str:
    q = item["quiz"]
    if q["type"] == "verdict":
        return "verdict"
    if isinstance(variant.get("key"), dict) and "answer" in variant["key"]:
        return "closed"
    return "free_text"


def prompt_for(item: dict, variant: dict, kind: str) -> str:
    """The user turn that asks the variant."""
    if kind == "closed":
        return variant["user"]
    original = item["turns"][0]["user"]
    material = item.get("material") or ""
    at = original.find(material) if material else -1
    if at < 0:
        raise QuizItemError(f"{item['id']}: a verdict variant needs the item's material, found verbatim in its turn")
    return original[:at + len(material)].rstrip() + "\n\n" + variant["user"].strip()


def base_key(item: dict) -> dict:
    q = item["quiz"]
    return {"verdict": q["verdict"], "deciding": q.get("deciding", []), "also_sufficient": q.get("also_sufficient", []),
            "escalate": q.get("escalate", "no")}


def score_sample(item: dict, kind: str, variant: dict, answer: str) -> dict:
    if kind == "closed":
        given = parse_choice(answer)
        return {"right": given == variant["key"]["answer"], "choice_key": variant["key"]["answer"],
                "choice_given": given, "verdict_only": None}
    parsed = harness.parse_answer(answer)
    task = {"key": base_key(item), "material": item["material"]}
    v_ok, d_ok = score.score({"final": parsed, "turns": []}, task, score.norm)
    e_ok = harness.grade(task, parsed)["escalate_correct"]
    return {"right": bool(v_ok and d_ok and e_ok), "verdict_only": bool(v_ok), "deciding_ok": bool(d_ok),
            "escalate_ok": bool(e_ok), "verdict_key": task["key"]["verdict"], "verdict_given": parsed["verdict"],
            "deciding_given": parsed["deciding"], "escalate_given": parsed["escalate"]}


def stuck_fields(thinking: str, material: str, tier: int, target: int, total_tokens: int, count, rule: dict) -> dict:
    tt = count(thinking)
    j = stuck.judge(thinking, material, tier, target, tt, total_tokens, rule, count=count)
    hp = stuck.hedge_profile(thinking, material, tt, count=count)
    pos = stuck.hedge_token_positions(thinking, material, count)
    bp = stuck.burst_profile(pos, tt, rule["burst"]["window"], rule["burst"]["k"])
    span = max(1, tt)
    gaps = bp["gaps"]
    return {
        "thinking_tokens": tt, "target_thinking_tokens": target,
        "tier_p95": rule["p95_by_tier"].get(str(tier), rule["p95_by_tier"]["7+"]),
        "stuck": j["stuck"], "stuck_reasons": [k for k in ("loop_span", "cap", "long_not_progressing") if j[k]],
        "loop_span": j["loop_span"], "cap": j["cap"], "long_not_progressing": j["long_not_progressing"],
        "over_p95": j["over_p95"], "over_2x_target": j["over_2x_target"], "reread_rate": j["reread_rate"],
        "rereads": j["rereads"], "lines_quoted": j["lines_quoted"],
        "hedges": hp["hedges"], "hedge_rate": hp["hedge_rate"], "hedge_per_phrase": hp["per_phrase"],
        "hedge_first": hp["first"], "hedge_last": hp["last"], "hedge_end_clustering": hp["end_clustering"],
        "hedge_gaps_tokens": gaps, "hedge_gap_median": sorted(gaps)[len(gaps) // 2] if gaps else None,
        "hedge_gap_min": min(gaps) if gaps else None, "gap_unit": "tokens (no streaming timestamps)",
        "max_in_window": {str(w): stuck.burst_profile(pos, tt, w, rule["burst"]["k"])["max_in_window"]
                          for w in WINDOWS},
        "bursts": [[round(a / span, 3), round(b / span, 3)] for a, b in bp["bursts"]],
        "burst_count": len(bp["bursts"]), "burst_before_answer": bp["burst_before_answer"],
        "trace": thinking if j["stuck"] else None,
    }


def trained_spans(items: list) -> set:
    """8-word spans of everything the trainer trains on: every turn's user text, thinking and reply."""
    import receipts
    out: set = set()
    for r in items:
        for t in r.get("turns", []):
            for k in ("user", "thinking", "reply"):
                out |= receipts.spans(t.get(k) or "")
    return out


def material_spans(items: list) -> set:
    import receipts
    out: set = set()
    for r in items:
        out |= receipts.spans(r.get("material") or "")
        for t in r.get("turns", []):
            nm = t.get("new_material") or ""
            out |= receipts.spans("\n".join(nm) if isinstance(nm, list) else nm)
    return out


def load_variants(paths) -> dict:
    """{base_id: {n: {user, key}}} from role-set's variant store files (n >= 1; n = 0 is the item's own)."""
    out: dict = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                v = json.loads(line)
                if v["n"] < 1:
                    raise SystemExit(f"{p}: variant store rows start at n = 1 (n = 0 is the item's own)")
                out.setdefault(v["base_id"], {})[v["n"]] = {"user": v["user"], "key": v["key"]}
    return out


def sit(items: list, ledger: Ledger, session: int, store: dict, generate, count, rule: dict, write,
        only: list | None = None) -> dict:
    """One quiz sitting. `generate(messages, seed)` returns the harness's generate dict; `count(text)` counts
    tokens; `write(fields)` records one receipt row. `items` is the whole training set (the spent check reads
    every trained turn); `only` limits the asking to those ids (tests). Returns the readout."""
    by_id = {r["id"]: r for r in items if r.get("kind") == "E" and (only is None or r["id"] in only)}
    excluded: dict = {}
    refused: list = []
    for r in by_id.values():
        if quiz_type(r, r["variant"]) == "free_text":
            excluded[r["lesson"]] = excluded.get(r["lesson"], 0) + 1
        else:
            ledger.enrol(r["id"], r["lesson"], session)
    import receipts
    trained = trained_spans(items)
    spent: list = []
    asked = []
    for base in ledger.due(session):
        r = by_id.get(base)
        if r is None:
            continue
        family = {0: r["variant"], **store.get(base, {})}
        shown_on_purpose = material_spans([r])  # only the asked item's own material is shown with it
        n = ledger.next_variant(base, sorted(family))
        while n is not None:
            why = None
            if quiz_type(r, family[n]) == "free_text":
                why = "free text, not yet testable"  # a written variant can only be quizzed in closed form
            elif (receipts.spans(family[n]["user"]) - shown_on_purpose) & trained:
                why = "shares an 8-word span with a trained turn"
            if why is None:
                break
            ledger.spend(base, n, why)
            spent.append({"base": base, "n": n, "reason": why})
            n = ledger.next_variant(base, sorted(family))
        if n is None:
            continue
        variant = family[n]
        kind = quiz_type(r, variant)
        try:
            user = prompt_for(r, variant, kind)
        except QuizItemError as e:
            refused.append(str(e))
            continue
        box_before = ledger.items[base]["box"]
        attempt = len(ledger.items[base]["history"]) + 1
        target = count(r["turns"][-1].get("thinking", ""))
        material = r.get("material", "") if kind == "verdict" else ""
        samples = []
        for seed in SEEDS:
            g = generate([{"role": "user", "content": user}], seed)
            thinking, answer, _ = harness.split_thinking(g["text"])
            s = score_sample(r, kind, variant, answer)
            samples.append((seed, g, thinking, answer, s))
        right = sum(s["right"] for *_, s in samples) >= MAJORITY
        state = ledger.record(base, session, n, right)
        for seed, g, thinking, answer, s in samples:
            write({"question_id": f"{base}-v{n}", "base_item_id": base, "variant": n, "lesson": r["lesson"],
                   "kind": "E", "quiz_type": kind, "tier": r["tier"], "attempt": attempt, "ledger_box": box_before,
                   "ledger_state": state, "sample": seed, "majority_right": right,
                   "correct_pinned": s["right"], **{k: v for k, v in s.items() if k != "right"},
                   "wall_s": g["seconds"], "answer_tokens": count(answer), "total_tokens": g["new_tokens"],
                   "truncated": g["truncated"],
                   **stuck_fields(thinking, material, r["tier"], target, g["new_tokens"], count, rule),
                   "answer": answer})
        asked.append({"base": base, "lesson": r["lesson"], "n": n, "right": right,
                      "stuck": sum(1 for *_, s in samples if s.get("stuck"))})
    out = readout(asked, excluded, refused, ledger)
    out["spent_variants"] = spent
    return out


def readout(asked, excluded, refused, ledger) -> dict:
    def acc(rows):
        return {"asked": len(rows), "right": sum(r["right"] for r in rows),
                "accuracy": round(sum(r["right"] for r in rows) / len(rows), 3) if rows else None}
    lessons = sorted({r["lesson"] for r in asked} | set(excluded))
    per = {}
    for ls in lessons:
        rows = [r for r in asked if r["lesson"] == ls]
        n0, n1 = acc([r for r in rows if r["n"] == 0]), acc([r for r in rows if r["n"] >= 1])
        per[ls] = {"all": acc(rows), "n0": n0, "n_ge_1": n1,
                   "gap_n0_minus_n1": round(n0["accuracy"] - n1["accuracy"], 3)
                   if n0["accuracy"] is not None and n1["accuracy"] is not None else None,
                   "written_excluded_not_yet_testable": excluded.get(ls, 0)}
    n0, n1 = acc([r for r in asked if r["n"] == 0]), acc([r for r in asked if r["n"] >= 1])
    return {"overall": {"all": acc(asked), "n0": n0, "n_ge_1": n1,
                        "written_excluded_not_yet_testable": sum(excluded.values())},
            "per_lesson": per, "qualities": ledger.qualities(), "refused_items": refused,
            "star": "off: no apply items"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", required=True, nargs="+")
    ap.add_argument("--variants", nargs="*", default=[])
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--arm", choices=("ROLE", "NO-ROLE"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-root", default=str(Path(__file__).resolve().parents[3] / "outputs" / "stage1-train"))
    ap.add_argument("--sealed", default=None, help="the frozen sealed file, for the receipt firewall")
    a = ap.parse_args(argv)

    import hashlib

    import receipts
    import student
    from aspire_si_gates import gates_identity
    from peft import PeftModel

    items = [r for p in a.data for r in data.load(p)]
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

    def generate(messages, seed):
        return harness.generate(model, tok, messages, "sampled", seed)

    def count(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def write(fields):
        try:
            w.write("train_quiz", fields)
        except receipts.Refused:
            pass  # counted in w.refused, reported below

    ledger = Ledger.load(a.ledger)
    out = sit(items, ledger, a.session, load_variants(a.variants), generate, count, rule, write)
    out["ledger_sha256"] = ledger.save(a.ledger)
    out["receipts_refused"] = w.refused
    (w.dir.parent / f"quiz-readout-session{a.session}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out["overall"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
