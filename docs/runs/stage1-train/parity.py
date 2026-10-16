"""B4's parity check, before round 1 (R&D's settlement after Argon's review, 2026-10-15). It runs on the base model
on the card, and its rules are fixed here before it runs.

    python parity.py --data <batch files...> --out <dir> [--limit-items N]

**The set:** 2 kind-E verdict variants (n = 0) per lesson, the lowest ids among those not spent. That's 32
variants over the 16 lessons of batches 1-6 (`select`). The C/D apply prompts join when B6 lands, before round 1.

**Four draws,** 3 samples per variant:
- LB1, the batched loop at batch 1;
- LB8, the batched loop at batch 8. Both use the per-item generators and the EFFICIENT kernel.
- P1, harness.generate at seeds 0-2;
- P1', harness.generate at seeds 3-5. P1 and P1' run under the flash pin and the deterministic settings, as
  every batch-1 sitting does (det_run.py), with CUBLAS_WORKSPACE_CONFIG set before torch loads. The report records
  the machine line (det_run.machine_line), for comparison with the frozen line.

**(i) LB1 against LB8** (the same per-item seeds, paired: batching alone). Argon's tolerances:
- aggregate stuck rate within ±0.03;
- item-level stuck agreement ≥ 92% or Cohen's κ ≥ 0.85;
- p90 thinking tokens within ±10%, and the median within ±5%;
- the first-token log-probability deviation logged (median and max), not gated.

**(ii) LB1 against the harness** (different RNG streams, so relative rules: #118's). With "within" meaning P1
against P1':
- cross majority agreement (P1 against LB1) ≥ within − 0.10;
- median thinking tokens within ±15% of P1's;
- stuck rate within ±0.10 of P1's;
- truncations ≤ P1's + 1;
- each stuck clause's rate within |P1 − P1'| + 0.05;
- median thinking tokens per tier within ±15% of P1's.

Absolute tolerances don't hold across RNG streams: at n = 90, the SE of a stuck-rate difference is about 0.05.

A fail stops B4 until R&D decides. It never moves a threshold. The report also records torch's peak reserved memory
(the run sheet adds the CUDA context and the idle baseline; memory.used must be ≤ 26 GB) and the speed-up.
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch loads, as det_run.py sets it (R&D)

import argparse  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import batched  # noqa: E402  (puts the harness on the path)
import det_run  # noqa: E402
import data  # noqa: E402
import quiz  # noqa: E402
import stuck_adapter  # noqa: E402
from aspire_si_gates.gates import variant_checked_spans  # noqa: E402

harness = batched.harness
CLAUSES = ("loop_span", "cap", "long_not_progressing")
RULES_I = {"stuck_rate_abs": 0.03, "item_agreement": 0.92, "kappa": 0.85, "p90_rel": 0.10, "median_rel": 0.05}
RULES_II = {"majority_margin": 0.10, "median_rel": 0.15, "stuck_abs": 0.10, "truncation_extra": 1,
            "clause_extra": 0.05, "tier_median_rel": 0.15}


def select(items: list, per_lesson: int = 2) -> list:
    """The parity set: per lesson, the lowest-id kind-E verdict items whose own variant isn't spent."""
    trained = quiz.trained_spans(items, quiz.SPENT_RULE_DEFAULT["span"])
    by_lesson: dict = {}
    for r in sorted(items, key=lambda r: r["id"]):
        if r["kind"] != "E" or (r.get("quiz") or {}).get("type") != "verdict":
            continue
        if variant_checked_spans(r, r["variant"], quiz.SPENT_RULE_DEFAULT) & trained:
            continue
        by_lesson.setdefault(r["lesson"], [])
        if len(by_lesson[r["lesson"]]) < per_lesson:
            by_lesson[r["lesson"]].append(r)
    return [r for lesson in sorted(by_lesson) for r in by_lesson[lesson]]


def record(r: dict, g: dict, count, rule: dict) -> dict:
    """One sample, reduced to what parity compares."""
    thinking, answer, _ = harness.split_thinking(g["text"])
    s = quiz.score_sample(r, "verdict", r["variant"], answer)
    f = quiz.stuck_fields(thinking, r.get("material", ""), r["tier"], count(r["turns"][-1].get("thinking", "")),
                          g["new_tokens"], count, rule)
    return {"right": bool(s["right"]), "stuck": bool(f["stuck"]), **{c: bool(f[c]) for c in CLAUSES},
            "thinking_tokens": f["thinking_tokens"], "truncated": bool(g["truncated"]), "tier": r["tier"],
            "new_tokens": g["new_tokens"],
            "first_logprob": g.get("first_logprob")}


def _rate(rows, k):
    return sum(r[k] for r in rows) / len(rows) if rows else 0.0


def _rel(a, b):
    return abs(a - b) / b if b else (0.0 if a == b else float("inf"))


def _p90(xs):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.9 * (len(xs) - 1))))]


def warned_ops(caught: list) -> list:
    """The ops warn_only let through without a deterministic implementation, as det_run.py records them."""
    return sorted({str(w.message) for w in caught if "deterministic" in str(w.message).lower()})


def kappa(a: list, b: list) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def compare_i(lb1: dict, lb8: dict) -> dict:
    """LB1 against LB8, paired by (item, sample)."""
    keys = sorted(lb1)
    if set(keys) != set(lb8):
        raise ValueError("LB1 and LB8 must cover the same (item, sample) keys")
    a, b = [lb1[k] for k in keys], [lb8[k] for k in keys]
    sa, sb = [x["stuck"] for x in a], [x["stuck"] for x in b]
    agree = sum(x == y for x, y in zip(sa, sb)) / len(keys)
    k = kappa(sa, sb)
    ta, tb = [x["thinking_tokens"] for x in a], [x["thinking_tokens"] for x in b]
    dev = [abs(x["first_logprob"] - y["first_logprob"]) for x, y in zip(a, b)
           if x["first_logprob"] is not None and y["first_logprob"] is not None]
    figs = {"n": len(keys), "stuck_lb1": _rate(a, "stuck"), "stuck_lb8": _rate(b, "stuck"),
            "item_agreement": agree, "kappa": k, "p90_lb1": _p90(ta), "p90_lb8": _p90(tb),
            "median_lb1": statistics.median(ta), "median_lb8": statistics.median(tb),
            "first_logprob_dev_median": statistics.median(dev) if dev else None,
            "first_logprob_dev_max": max(dev) if dev else None}
    checks = {"stuck_rate": abs(figs["stuck_lb1"] - figs["stuck_lb8"]) <= RULES_I["stuck_rate_abs"],
              "item_stuck_agreement": agree >= RULES_I["item_agreement"] or k >= RULES_I["kappa"],
              "p90_thinking": _rel(figs["p90_lb8"], figs["p90_lb1"]) <= RULES_I["p90_rel"],
              "median_thinking": _rel(figs["median_lb8"], figs["median_lb1"]) <= RULES_I["median_rel"]}
    return {"figures": figs, "checks": checks, "pass": all(checks.values())}


def majority(rows: dict) -> dict:
    """{item: right by majority of its samples} from {(item, sample): record}."""
    per: dict = {}
    for (item, _), r in rows.items():
        per.setdefault(item, []).append(r["right"])
    return {i: sum(v) * 2 > len(v) for i, v in per.items()}


def compare_ii(lb1: dict, p1: dict, p1p: dict) -> dict:
    """LB1 against the harness's P1, with P1 against P1' as the noise reference."""
    mp1, mp1p, mlb = majority(p1), majority(p1p), majority(lb1)
    items = sorted(mp1)
    within = sum(mp1[i] == mp1p[i] for i in items) / len(items)
    cross = sum(mp1[i] == mlb[i] for i in items) / len(items)
    a, b, c = list(p1.values()), list(p1p.values()), list(lb1.values())
    med = {name: statistics.median(x["thinking_tokens"] for x in rows) for name, rows in (("p1", a), ("lb1", c))}
    clause = {cl: {"p1": _rate(a, cl), "p1p": _rate(b, cl), "lb1": _rate(c, cl)} for cl in CLAUSES}
    tiers = sorted({x["tier"] for x in a})
    tier_med = {t: {"p1": statistics.median(x["thinking_tokens"] for x in a if x["tier"] == t),
                    "lb1": statistics.median(x["thinking_tokens"] for x in c if x["tier"] == t)} for t in tiers
                if any(x["tier"] == t for x in c)}
    figs = {"items": len(items), "majority_within": within, "majority_cross": cross, "median": med,
            "stuck": {"p1": _rate(a, "stuck"), "lb1": _rate(c, "stuck")},
            "truncations": {"p1": sum(x["truncated"] for x in a), "lb1": sum(x["truncated"] for x in c)},
            "clauses": clause, "tier_median": tier_med}
    checks = {
        "majority_agreement": cross >= within - RULES_II["majority_margin"],
        "median_thinking": _rel(med["lb1"], med["p1"]) <= RULES_II["median_rel"],
        "stuck_rate": abs(figs["stuck"]["lb1"] - figs["stuck"]["p1"]) <= RULES_II["stuck_abs"],
        "truncations": figs["truncations"]["lb1"] <= figs["truncations"]["p1"] + RULES_II["truncation_extra"],
        "clauses": all(abs(v["lb1"] - v["p1"]) <= abs(v["p1"] - v["p1p"]) + RULES_II["clause_extra"]
                       for v in clause.values()),
        "tier_medians": all(_rel(v["lb1"], v["p1"]) <= RULES_II["tier_median_rel"] for v in tier_med.values())}
    return {"figures": figs, "checks": checks, "pass": all(checks.values())}


DRAW_ORDER = ("LB8", "LB1", "P1", "P1p")  # LB8 first: the fast draw measures the counted set's lengths
B1_TOK_S_GUESS = 20.0  # batch-1 tokens per second from the preflight (about 300 tokens in 16 s), until measured


def key_str(k: tuple) -> str:
    return f"{k[0]}|{k[1]}"


def save_draw(path: Path, rows: dict) -> None:
    """One finished draw, written as it completes, so a stop keeps every finished draw."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps({"key": key_str(k), **v}) + "\n" for k, v in sorted(rows.items())),
                   encoding="utf-8", newline="\n")
    tmp.replace(path)


def load_draw(path: Path, expected: set) -> dict | None:
    """A finished draw from an earlier run of the same check, or None when it's missing or incomplete."""
    if not path.exists():
        return None
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            item, sample = r.pop("key").rsplit("|", 1)
            rows[(item, int(sample))] = r
    return rows if set(rows) == expected else None


class Progress:
    """A line per generation, with a running ETA from the mean seconds per generation so far."""

    def __init__(self, name: str, total: int, out=print):
        self.name, self.total, self.done, self.seconds, self.tokens, self.out = name, total, 0, 0.0, 0, out
        self.t0 = time.time()

    def step(self, key: str, tokens: int, seconds: float) -> str:
        self.done += 1
        self.tokens += tokens
        self.seconds = time.time() - self.t0
        left = (self.total - self.done) * self.seconds / self.done
        line = (f"{self.name} {self.done}/{self.total} {key} tokens={tokens} s={seconds:.0f} "
                f"elapsed={self.seconds / 60:.1f}m eta={left / 60:.1f}m")
        self.out(line, flush=True)
        return line


def sizing(lb8: dict, cap: int, tok_s: float, per_draw: int) -> dict:
    """From LB8's lengths on the counted set: the cap-hitting share, and each batch-1 draw's projected minutes at
    tok_s (the same lengths, one generation at a time)."""
    tokens = [r["new_tokens"] for r in lb8.values()]
    scale = per_draw / len(tokens)
    return {"generations": len(tokens), "tokens_total": sum(tokens), "tokens_median": statistics.median(tokens),
            "at_cap": sum(t >= cap for t in tokens),
            "at_cap_share": round(sum(t >= cap for t in tokens) / len(tokens), 3),
            "tok_s_batch1": tok_s, "minutes_per_batch1_draw": round(sum(tokens) * scale / tok_s / 60, 1)}


def fits(minutes: float, deadline: float | None) -> bool:
    return deadline is None or time.time() + minutes * 60 <= deadline


def main(argv=None) -> int:  # pragma: no cover - runs on the card
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-items", type=int, help="a short preflight run; never the counted check")
    ap.add_argument("--deadline", help="the grant's end, ISO 8601 UTC: a draw that can't finish by then isn't started")
    a = ap.parse_args(argv)
    import torch
    from datetime import datetime
    from torch.nn.attention import SDPBackend, sdpa_kernel

    import student
    deadline = datetime.fromisoformat(a.deadline.replace("Z", "+00:00")).timestamp() if a.deadline else None
    items = [r for p in a.data for r in data.load(p)]
    chosen = select(items)[: a.limit_items]
    rule = stuck_adapter.load_frozen()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = student.load_tokenizer()
    model = student.load_student()
    model.eval()

    def count(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    asks = [{"key": f"{r['id']}-v0|{s}", "item_id": f"{r['id']}-v0", "sample": s,
             "messages": [{"role": "user", "content": quiz.prompt_for(r, r["variant"], "verdict")}]}
            for r in chosen for s in quiz.SEEDS]
    by_item = {f"{r['id']}-v0": r for r in chosen}
    expected = {(k.split("|")[0], int(k.split("|")[1])) for k in (x["key"] for x in asks)}
    draws, times, warned, size = {}, {}, {}, None
    tok_s = B1_TOK_S_GUESS
    torch.cuda.reset_peak_memory_stats()
    import warnings
    for name in DRAW_ORDER:
        path = out / f"draw-{name}.jsonl"
        done = load_draw(path, expected)
        if done is not None:
            draws[name] = done
            print(f"{name}: kept from an earlier run ({path.name})", flush=True)
            continue
        if size is not None and not fits(size["minutes_per_batch1_draw"], deadline):
            print(f"{name}: not started, projected {size['minutes_per_batch1_draw']} min won't fit the deadline",
                  flush=True)
            break
        prog = Progress(name, len(asks))
        t0 = time.time()
        if name in ("LB8", "LB1"):
            log: list = []
            gens = batched.make_generate_many(model, tok, log, batch=8 if name == "LB8" else 1,
                                              on_done=prog.step)(asks)
            rows = {(k.split("|")[0], int(k.split("|")[1])): record(by_item[k.split("|")[0]], g, count, rule)
                    for k, g in gens.items()}
            (out / f"{name}-batches.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
        else:
            torch.use_deterministic_algorithms(True, warn_only=True)  # as det_run.py sets for batch-1 sittings
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
            seeds = (0, 1, 2) if name == "P1" else (3, 4, 5)
            rows = {}
            with warnings.catch_warnings(record=True) as caught, sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
                warnings.simplefilter("always")
                for ask in asks:
                    if ask["sample"] != 0:
                        continue
                    for k, seed in enumerate(seeds):
                        g = harness.generate(model, tok, ask["messages"], "sampled", seed)
                        rows[(ask["item_id"], k)] = record(by_item[ask["item_id"]], g, count, rule)
                        prog.step(f"{ask['item_id']}|{k}", g["new_tokens"], g["seconds"])
            warned[name] = warned_ops(caught)
            for w in caught:  # recorded, and still shown
                warnings.showwarning(w.message, w.category, w.filename, w.lineno)
            torch.use_deterministic_algorithms(False)
        draws[name], times[name] = rows, round(time.time() - t0, 1)
        save_draw(path, rows)
        if name == "LB1" and prog.tokens:
            tok_s = prog.tokens / max(1.0, prog.seconds)  # measured, for the harness draws' projection
        if name in ("LB8", "LB1"):
            size = sizing(draws["LB8"], harness.MAX_NEW_TOKENS, tok_s, len(asks))
            (out / "sizing.json").write_text(json.dumps(size, indent=1), encoding="utf-8")
            print("sizing: " + json.dumps(size), flush=True)
    if set(draws) != set(DRAW_ORDER):
        print(json.dumps({"incomplete": sorted(set(DRAW_ORDER) - set(draws)), "kept": sorted(draws)}), flush=True)
        return 2
    report = {"items": [r["id"] for r in chosen], "rules_i": RULES_I, "rules_ii": RULES_II,
              "i": compare_i(draws["LB1"], draws["LB8"]), "ii": compare_ii(draws["LB1"], draws["P1"], draws["P1p"]),
              "machine_line": det_run.machine_line(), "warned_ops": warned, "sizing": size,
              "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
              "seconds": times, "torch_peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
              "batched_sha256": __import__("hashlib").sha256(Path(batched.__file__).read_bytes()).hexdigest()}
    report["pass"] = report["i"]["pass"] and report["ii"]["pass"]
    (out / "parity-report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"pass": report["pass"], "i": report["i"]["checks"], "ii": report["ii"]["checks"]}))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
