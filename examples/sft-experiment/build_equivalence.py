"""The Ollama CUDA 13.0 -> 13.4 equivalence sample (Auditor plan, addendum 4, step 2).

The same gemma4:31b calls (thinking on, temperature 0, the same prompts and options) are asked on
the current build and repeated, then asked on the new build:
  - 30 grammar items: a seed-0 random sample of the full grammar pass's judged pairs (v2 question);
  - 30 meaning items: a seed-0 random sample of P-confirm's word swaps (recheck.py's prompt).

Each run is stored under a label (e.g. "13.0-a", "13.0-b", "13.4"). The readout gives, per task, the
same-build repeat agreement (the noise floor) and the cross-build agreement, with Wilson intervals,
and lists every disagreement.

Stop rule (committed before running): put it to the maintainer before any 13.4 judge result is mixed
with a 13.0 one if, on either task, cross-build agreement is below 90% AND more than 10 points
below the same-build repeat agreement. A shortfall within the build's own run-to-run noise is not a
build difference.

Every item records the URL, Ollama version and build label that answered it.

Usage:
  python build_equivalence.py ask --label 13.0-a --build "Ollama 0.35.1, CUDA 13.0" --out DIR
  python build_equivalence.py ask --label 13.4 --build "Ollama sandbox, CUDA 13.4.1" \
      --url http://127.0.0.1:11492 --out DIR
  python build_equivalence.py readout --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

N = 30
STOP_FLOOR = 0.90
STOP_GAP = 0.10


def sample_ids(ids: list[str], n: int = N, seed: int = 0) -> list[str]:
    pool = sorted(ids)
    random.Random(seed).shuffle(pool)
    return sorted(pool[:n])


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [0.0, 1.0]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def verdict(task: str, v: dict) -> str | None:
    """The decision that matters per task: grammar -> grammatical true/false; meaning -> kept or not."""
    if v.get("outcome") in ("truncated", "unparsed", None):
        return None
    if task == "grammar":
        return str(v.get("grammatical"))
    return (
        "keep" if v.get("verdict", v.get("outcome")) == "same" and v.get("claim_changed") is False else "drop"
    )


def agreement(task: str, a: dict, b: dict) -> dict:
    ids = sorted(set(a) & set(b))
    pairs = [(i, verdict(task, a[i]), verdict(task, b[i])) for i in ids]
    usable = [(i, x, y) for i, x, y in pairs if x is not None and y is not None]
    agree = sum(1 for _, x, y in usable if x == y)
    n = len(usable)
    return {
        "items": len(ids),
        "usable": n,
        "agree": agree,
        "rate": agree / n if n else None,
        "wilson": wilson(agree, n),
        "disagreements": [i for i, x, y in usable if x != y],
    }


def stop(cross: float | None, repeat: float | None) -> bool:
    if cross is None:
        return True
    floor = repeat if repeat is not None else 1.0
    return cross < STOP_FLOOR and cross < floor - STOP_GAP


def readout(runs: dict[str, dict], old: str = "13.0-a", repeat: str = "13.0-b", new: str = "13.4") -> dict:
    out = {}
    for task in ("grammar", "meaning"):
        r = agreement(task, runs[old][task], runs[repeat][task]) if repeat in runs else None
        c = agreement(task, runs[old][task], runs[new][task]) if new in runs else None
        out[task] = {
            "same_build_repeat": r,
            "cross_build": c,
            "stop": stop(c["rate"] if c else None, r["rate"] if r else None) if c else None,
        }
    out["stop_rule"] = (
        f"stop if cross-build agreement < {STOP_FLOOR:.0%}"
        f" AND more than {STOP_GAP:.0%} below the same-build repeat"
    )
    out["any_stop"] = any(out[t]["stop"] for t in ("grammar", "meaning") if out[t]["stop"] is not None)
    return out


def server_identity(url: str) -> dict:  # pragma: no cover - needs Ollama
    """The answering server's URL and Ollama version, recorded with every item."""
    import urllib.request

    with urllib.request.urlopen(url.rstrip("/") + "/api/version", timeout=30) as r:
        version = json.loads(r.read()).get("version")
    return {"url": url, "ollama_version": version}


def ask(label: str, out: Path, url: str, build: str) -> None:  # pragma: no cover - needs Ollama
    import matched_set as ms
    from critic_heads import load_pairs
    from recheck import REQUEST, SCHEMA, SYSTEM
    from skeptic_pairs import sentences

    a = Path("E:/AI/aspire-si-runs/2026-10-08-auditor")
    grammar = json.loads((a / "matched" / "grammar-wordswaps.json").read_text(encoding="utf-8"))
    pairs = {
        n: {p["pair_id"]: p for p in load_pairs(a / "skeptic-word" / n / "pairs.json")}
        for n in ("pconfirm", "psecond", "ptrain")
    }
    judged = [(n, pid) for n, rows in grammar.items() for pid, r in rows.items() if ms._v2(r)]
    g_ids = sample_ids([f"{n}/{pid}" for n, pid in judged])
    m_ids = sample_ids(list(pairs["pconfirm"]))
    path = out / f"equivalence-{label}.json"
    state = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"grammar": {}, "meaning": {}}
    judge = "gemma4:31b"
    who = server_identity(url) | {"build": build}
    state.setdefault("server", who)
    if state["server"] != who:
        raise SystemExit(f"{path.name} was started on {state['server']}, not {who}")
    try:
        for key in g_ids:
            if key in state["grammar"]:
                continue
            n, pid = key.split("/", 1)
            p = pairs[n][pid]
            original, edited = sentences(p)
            user = ms.GRAMMAR_CHECK.format(
                prompt=p["prompt"], flawed=p["flawed"], original=original, edited=edited
            )
            r = ms.ollama_chat(judge, ms.JUDGE_SYSTEM, user, ms.GRAMMAR_SCHEMA, True, **ms.THINKING, url=url)
            state["grammar"][key] = ms.judged(r, ("grammatical", "idiomatic")) | who
            path.write_text(json.dumps(state, indent=1), encoding="utf-8")
        for pid in m_ids:
            if pid in state["meaning"]:
                continue
            p = pairs["pconfirm"][pid]
            original, edited = sentences(p)
            user = REQUEST.format(
                prompt=p["prompt"], strong=p["strong"], flawed=p["flawed"], original=original, edited=edited
            )
            r = ms.ollama_chat(judge, SYSTEM, user, SCHEMA, True, **ms.THINKING, url=url)
            state["meaning"][pid] = ms.judged(r, ("verdict", "claim_changed")) | who
            path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    finally:
        ms.unload(judge, url)
    print("EQUIVALENCE-ASK-OK", label, len(state["grammar"]), len(state["meaning"]))


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("cmd", choices=["ask", "readout"])
    parser.add_argument("--label")
    parser.add_argument("--url", default="http://127.0.0.1:11434", help="the Ollama server to ask")
    parser.add_argument("--build", help='recorded with every item, e.g. "Ollama 0.35.1, CUDA 13.0 backend"')
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "ask":
        if not args.label or not args.build:
            raise SystemExit("ask needs --label and --build")
        ask(args.label, args.out, args.url, args.build)
        return
    runs = {
        p.stem.removeprefix("equivalence-"): json.loads(p.read_text(encoding="utf-8"))
        for p in args.out.glob("equivalence-*.json")
        if p.stem != "equivalence-readout"
    }
    result = readout(runs)
    result["servers"] = {label: run.get("server") for label, run in runs.items()}
    (args.out / "equivalence-readout.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
