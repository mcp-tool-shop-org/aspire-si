"""Addendum 3 of the 2026-10-08 Auditor plan: the typicality match for the Skeptic control.

An edit replaces some of the strong model's own wording, so the edited copy is usually a little
less typical in context, whether the edit plants an error or a paraphrase. The control needs the
paraphrases to carry that cue to the same degree as the errors, so that a head reading typicality
fails it. This script measures the cue with a third model, outside every family already in the
pipeline. The student, the planter and Kev are Qwen; the feature sources are Qwen and Llama;
gemma plants P-second; mistral checks meaning. The third model is microsoft/Phi-3-mini-4k-instruct
(MIT).

Per pair: delta = log p(the edited copy's changed tokens) - log p(the original's changed tokens),
each in context (the question, then the answer up to the change), summed over the tokens between
the two copies' common prefix and common suffix.

The pass rule, committed before computing (plan, addendum 3): on each read set (P-confirm against
the confirm error pairs, P-second against the second-planter error pairs), the median delta of the
kept paraphrases lies inside the error pairs' interquartile range of delta. A two-sample
Kolmogorov-Smirnov statistic and its asymptotic p-value are reported beside it, not gated.
P-train against the training error pairs is reported only.

Usage:
  python typicality.py --pairs NAME=PATH [--verify NAME=VERIFY_JSON] --out DIR
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from critic_heads import load_pairs  # noqa: E402

MODEL = "microsoft/Phi-3-mini-4k-instruct"
REVISION = "f39ac1d28e925b323eae81227eaba4464caced4e"
LICENSE = "MIT"
MAX_LENGTH = 4096
# The read sets and the error set each is matched against; P-train is reported only.
READ_SETS = {"pconfirm": "confirm", "psecond": "second"}
REPORTED_ONLY = {"ptrain": "train"}


def changed_span(a: list[int], b: list[int]) -> tuple[int, int, int]:
    """(prefix, end in a, end in b): the token ranges a[prefix:end_a] and b[prefix:end_b] that
    differ, between the longest common prefix and the longest common suffix that doesn't overlap
    it."""
    prefix = 0
    while prefix < min(len(a), len(b)) and a[prefix] == b[prefix]:
        prefix += 1
    suffix = 0
    while suffix < min(len(a), len(b)) - prefix and a[-1 - suffix] == b[-1 - suffix]:
        suffix += 1
    return prefix, len(a) - suffix, len(b) - suffix


def span_logprob(token_logprobs: list[float], start: int, end: int) -> float:
    """Sum of log p(token i | tokens before i) for i in [start, end). token_logprobs[i] is token
    i's log-probability; token 0 has none and counts as 0."""
    return sum(token_logprobs[i] for i in range(max(start, 1), end))


def delta(
    strong_ids: list[int], strong_lp: list[float], flawed_ids: list[int], flawed_lp: list[float]
) -> float:
    prefix, end_s, end_f = changed_span(strong_ids, flawed_ids)
    return span_logprob(flawed_lp, prefix, end_f) - span_logprob(strong_lp, prefix, end_s)


def quartiles(values: list[float]) -> tuple[float, float, float]:
    if len(values) == 1:
        return values[0], values[0], values[0]
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[0], q[1], q[2]


def ks_two_sample(a: list[float], b: list[float]) -> tuple[float, float]:
    """The two-sample Kolmogorov-Smirnov statistic D and its asymptotic two-sided p-value."""
    xs, ys = sorted(a), sorted(b)
    n, m = len(xs), len(ys)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        v = min(xs[i], ys[j])
        while i < n and xs[i] == v:
            i += 1
        while j < m and ys[j] == v:
            j += 1
        d = max(d, abs(i / n - j / m))
    en = math.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    if lam < 0.2:  # the series converges badly near 0, where p is 1 to many places
        return d, 1.0
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return d, min(1.0, max(0.0, p))


def match(errors: list[float], paraphrases: list[float]) -> dict:
    """The committed rule: the paraphrases' median delta inside the errors' interquartile range."""
    q1, median, q3 = quartiles(errors)
    para_median = statistics.median(paraphrases)
    d, p = ks_two_sample(errors, paraphrases)
    return {
        "error_pairs": len(errors),
        "paraphrase_pairs": len(paraphrases),
        "error_quartiles": [q1, median, q3],
        "paraphrase_quartiles": list(quartiles(paraphrases)),
        "paraphrase_median": para_median,
        "passes": q1 <= para_median <= q3,
        "ks_d": d,
        "ks_p": p,
    }


def readout(deltas: dict[str, dict[str, float]], dropped: dict[str, set]) -> dict:
    """Per matched set: the rule's result (read sets) or the same numbers reported only (P-train)."""
    out = {}
    for para, err in {**READ_SETS, **REPORTED_ONLY}.items():
        if para not in deltas or err not in deltas:
            continue
        kept = [v for pid, v in deltas[para].items() if pid not in dropped.get(para, set())]
        row = match(list(deltas[err].values()), kept)
        row["gated"] = para in READ_SETS
        out[para] = row
    out["all_read_sets_pass"] = all(out[s]["passes"] for s in READ_SETS if s in out)
    return out


def score(sets: dict[str, list[dict]]) -> dict[str, dict[str, float]]:  # pragma: no cover - GPU
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from aspire.judge import format_exchange, uses_chat_template

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    special = not uses_chat_template(tok)

    def logprobs(text: str) -> tuple[list[int], list[float]]:
        ids = tok(text, add_special_tokens=special)["input_ids"]
        if len(ids) > MAX_LENGTH:
            raise SystemExit(f"an exchange is {len(ids)} tokens, over {MAX_LENGTH}")
        with torch.no_grad():
            logits = model(torch.tensor([ids], device="cuda")).logits[0].float()
        lp = torch.log_softmax(logits[:-1], dim=-1).gather(1, torch.tensor(ids[1:], device="cuda")[:, None])
        return ids, [0.0] + lp[:, 0].tolist()

    out: dict[str, dict[str, float]] = {}
    for name, pairs in sets.items():
        out[name] = {}
        for p in pairs:
            s_ids, s_lp = logprobs(format_exchange(tok, p["prompt"], p["strong"]))
            f_ids, f_lp = logprobs(format_exchange(tok, p["prompt"], p["flawed"]))
            out[name][p["pair_id"]] = delta(s_ids, s_lp, f_ids, f_lp)
        print(name, len(out[name]), "pairs", flush=True)
    return out


def main() -> None:  # pragma: no cover - needs a GPU
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pairs", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--verify", action="append", default=[], help="NAME=skeptic_pairs.py verify output")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sets = {name: load_pairs(Path(path)) for name, path in (s.split("=", 1) for s in args.pairs)}
    dropped = {}
    for name, path in (s.split("=", 1) for s in args.verify):
        v = json.loads(Path(path).read_text(encoding="utf-8"))
        dropped[name] = set(v["flagged_changed_meaning"]) | set(v["unparsed"])
    deltas = score(sets)
    result = {
        "model": MODEL,
        "revision": REVISION,
        "license": LICENSE,
        "precision": "bf16",
        "rule": "per read set: kept paraphrases' median delta inside the error pairs' interquartile range",
        "readout": readout(deltas, dropped),
    }
    (args.out / "deltas.json").write_text(json.dumps(deltas, indent=1), encoding="utf-8")
    (args.out / "typicality.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print("TYPICALITY-OK", args.out)


if __name__ == "__main__":
    main()
