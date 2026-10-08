"""Step 1 of the 2026-10-08 plan: Kev as a reference judge on the planted-error judge set.

Kev (github.com/jaredpalmer/kev, Apache-2.0) is a decision model: one forward pass returns
probabilities over a fixed set of options, with no text generation. Each judge pair becomes one
`choice` question. The state is the question, and the two options are the two answers, named
"A" and "B". Every pair is asked twice, once with the strong answer as A and once as B, so the
two orders together cover option-order sensitivity. With two options, Kev's /permute endpoint
adds nothing beyond that.

Kev's probabilities are calibrated on its own training tasks, not on this one, so they are not
read as scores. Step 1 used only the choice and the margin, p(strong) − p(flawed). The
confirmation plan adds `order_averaged`: p(strong) averaged over the two orders, which cancels
Kev's lean toward option A.

Writes --out (kev-<name>.json):
  - accuracy averaged over both orders, and the rate at which option A is chosen (position bias);
  - the mean margin;
  - the Kev-separable pair ids (strong chosen in both orders), with their pair and prompt counts;
  - every pair's two answers from Kev.
judge_eval.py --subset reads the separable ids, as it does pairwise_teacher.py's.

Usage (Kev serving on 127.0.0.1:8009):
  python judge_kev.py --judge-set data/judge_set.json --out results/kev-4b.json --name kev-4b
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections.abc import Callable
from pathlib import Path

INSTRUCTIONS = (
    "Two answers to the question are given as options A and B. They may differ in only one "
    "sentence. Which answer is more correct? Judge the correctness of facts, numbers and "
    "reasoning, not style or length."
)


def request(prompt: str, first: str, second: str, model: str = "kev-latest") -> dict:
    """A Kev /v1/systemone body: the question as the state, the two answers as options A and B."""
    return {
        "model": model,
        "state": prompt,
        "questions": {
            "better": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {"A": first, "B": second},
            }
        },
    }


def read_choice(response: dict) -> tuple[str, float, float]:
    """(choice, p(A), p(B)) from a Kev response."""
    answer = response["answers"]["better"]
    probs = answer.get("probabilities") or {}
    return answer["choice"], float(probs.get("A", 0.0)), float(probs.get("B", 0.0))


def judge(ask: Callable[[dict], dict], pairs: list[dict], model: str = "kev-latest") -> dict:
    """Ask Kev about every pair in both orders; `ask` posts one body and returns the response."""
    rows = []
    for k, p in enumerate(pairs):
        c1, a1, b1 = read_choice(ask(request(p["prompt"], p["strong"], p["flawed"], model)))
        c2, a2, b2 = read_choice(ask(request(p["prompt"], p["flawed"], p["strong"], model)))
        rows.append(
            {
                "pair_id": p.get("pair_id", k),
                "prompt_id": p.get("prompt_id", p["prompt"]),
                "strong_first": {"choice": c1, "p_strong": a1, "p_flawed": b1},
                "strong_second": {"choice": c2, "p_strong": b2, "p_flawed": a2},
                "separable": c1 == "A" and c2 == "B",
            }
        )
    right = [r["strong_first"]["choice"] == "A" for r in rows] + [
        r["strong_second"]["choice"] == "B" for r in rows
    ]
    picks_a = [r[key]["choice"] == "A" for r in rows for key in ("strong_first", "strong_second")]
    margins = [
        r[key]["p_strong"] - r[key]["p_flawed"] for r in rows for key in ("strong_first", "strong_second")
    ]
    separable = [r for r in rows if r["separable"]]
    return {
        "pairs": len(rows),
        "accuracy": sum(right) / max(len(right), 1),
        "a_rate": sum(picks_a) / max(len(picks_a), 1),
        "mean_margin": sum(margins) / max(len(margins), 1),
        "separable_pairs": len(separable),
        "separable_prompts": len({r["prompt_id"] for r in separable}),
        "separable_ids": [r["pair_id"] for r in separable],
        "rows": rows,
    }


def order_averaged(rows: list[dict], resamples: int = 2000, seed: int = 0) -> dict:
    """The 2026-10-08 confirmation plan's statistic, fixed before it ran.

    Per pair, p(strong) is normalised over the two options in each order and averaged over the two
    orders. A pair is favoured when that mean is above 0.5 (exactly 0.5 counts half). Accuracy is
    the share of favoured pairs, with a 95% bootstrap interval that resamples whole prompts. Also:
    the mean and spread of the gap p(A | strong first) - p(A | strong second), and how many pairs
    are decided for the strong answer in both orders without averaging.
    """
    import random
    import statistics

    def strong_share(order: dict) -> float:
        total = order["p_strong"] + order["p_flawed"]
        return order["p_strong"] / total if total > 0 else 0.5

    means = [(strong_share(r["strong_first"]) + strong_share(r["strong_second"])) / 2 for r in rows]
    wins = [1.0 if m > 0.5 else 0.5 if m == 0.5 else 0.0 for m in means]
    gaps = [2 * m - 1 for m in means]
    members: dict = {}
    for i, r in enumerate(rows):
        members.setdefault(r["prompt_id"], []).append(i)
    clusters = list(members.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        idx = [i for _ in clusters for i in clusters[rng.randrange(len(clusters))]]
        stats.append(sum(wins[i] for i in idx) / len(idx))
    stats.sort()
    favoured = {rows[i]["prompt_id"] for i, w in enumerate(wins) if w == 1.0}
    return {
        "accuracy": sum(wins) / max(len(wins), 1),
        "ci": [stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1]],
        "favoured_pairs": sum(w == 1.0 for w in wins),
        "favoured_prompts": len(favoured),
        "prompts": len(clusters),
        "mean_abs_gap": statistics.fmean(abs(g) for g in gaps),
        "sd_gap": statistics.pstdev(gaps),
        "both_orders_strong": sum(
            r["strong_first"]["choice"] == "A" and r["strong_second"]["choice"] == "B" for r in rows
        ),
    }


def confirmation_reading(accuracy: float, ci_low: float) -> str:
    """The confirmation plan's rule: order-averaged accuracy of at least 0.85 on the fresh pairs,
    with the lower end of its prompt-clustered 95% interval at least 0.75."""
    return "confirmed" if accuracy >= 0.85 and ci_low >= 0.75 else "not confirmed"


def reading(result: dict) -> str:
    """The plan's pre-registered decision rule for step 1."""
    if result["accuracy"] < 0.75:
        return "not a useful judge of single planted errors"
    if 0.40 <= result["a_rate"] <= 0.60:
        return "usable reference judge"
    return "accurate but position-biased: report both orders"


def http_ask(url: str) -> Callable[[dict], dict]:  # pragma: no cover - needs a Kev server
    def ask(body: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read())

    return ask


def main() -> None:  # pragma: no cover - needs a Kev server
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--judge-set", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--name", required=True, help="e.g. kev-4b; recorded in the output")
    parser.add_argument("--url", default="http://127.0.0.1:8009/v1/systemone")
    args = parser.parse_args()
    pairs = json.loads(args.judge_set.read_text(encoding="utf-8"))
    result = judge(http_ask(args.url), pairs) | {"name": args.name}
    result["reading"] = reading(result)
    result["order_averaged"] = order_averaged(result["rows"])
    result["order_averaged"]["reading"] = confirmation_reading(
        result["order_averaged"]["accuracy"], result["order_averaged"]["ci"][0]
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    keys = (
        "name",
        "pairs",
        "accuracy",
        "a_rate",
        "mean_margin",
        "separable_pairs",
        "separable_prompts",
        "reading",
    )
    print(json.dumps({k: result[k] for k in keys} | {"order_averaged": result["order_averaged"]}, indent=1))
    print("KEV-OK", args.out)


if __name__ == "__main__":
    main()
