"""The teacher as a pairwise judge, read from its A/B log-probabilities in both orders.

pairwise_teacher.py kept only the teacher's text answer. Its single choices were "A" on all 127
judge pairs with the strong answer first, and that says nothing about a preference underneath.
Here the same request goes to an OpenAI-compatible server (llama.cpp's llama-server) for one
token, and p(A) and p(B) are read from its top log-probabilities and normalised over the two.
Rows have judge_kev.py's shape, so `judge_kev.order_averaged` reads them with the confirmation
plan's fixed statistic.

Writes --out: single-choice accuracy, the A rate, the order-averaged block, and every pair's two
readings.

Usage (llama-server on 127.0.0.1:8010):
  python judge_logprob.py --pairs fresh/confirm_set.json --name qwen32b-q4 --out results/qwen32b-q4.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from judge_kev import confirmation_reading, order_averaged  # noqa: E402
from lib import teacher_system_prompt  # noqa: E402
from pairwise_teacher import PAIRWISE_REQUEST  # noqa: E402


def letter_probs(response: dict) -> tuple[float, float]:
    """p(A) and p(B) from the first generated token's top log-probabilities, normalised over the two.
    Tokens are compared after stripping spaces, so " A" counts as "A"."""
    top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    mass = {"A": 0.0, "B": 0.0}
    for entry in top:
        token = entry["token"].strip()
        if token in mass:
            mass[token] += math.exp(entry["logprob"])
    total = mass["A"] + mass["B"]
    if total == 0:
        return 0.5, 0.5
    return mass["A"] / total, mass["B"] / total


def body(system: str, prompt: str, a: str, b: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": PAIRWISE_REQUEST.format(prompt=prompt, a=a, b=b)},
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": True,
        "top_logprobs": 20,
    }


def judge(ask: Callable[[dict], dict], pairs: list[dict], teacher: str = "Qwen/Qwen2.5-32B-Instruct") -> dict:
    system = teacher_system_prompt(teacher)
    rows = []
    for k, p in enumerate(pairs):
        a1, b1 = letter_probs(ask(body(system, p["prompt"], p["strong"], p["flawed"])))
        a2, b2 = letter_probs(ask(body(system, p["prompt"], p["flawed"], p["strong"])))
        rows.append(
            {
                "pair_id": p.get("pair_id", k),
                "prompt_id": p.get("prompt_id", p["prompt"]),
                "strong_first": {"choice": "A" if a1 >= b1 else "B", "p_strong": a1, "p_flawed": b1},
                "strong_second": {"choice": "A" if a2 >= b2 else "B", "p_strong": b2, "p_flawed": a2},
            }
        )
    right = [r["strong_first"]["choice"] == "A" for r in rows] + [
        r["strong_second"]["choice"] == "B" for r in rows
    ]
    picks_a = [r[key]["choice"] == "A" for r in rows for key in ("strong_first", "strong_second")]
    averaged = order_averaged(rows)
    averaged["reading"] = confirmation_reading(averaged["accuracy"], averaged["ci"][0])
    return {
        "pairs": len(rows),
        "accuracy": sum(right) / max(len(right), 1),
        "a_rate": sum(picks_a) / max(len(picks_a), 1),
        "order_averaged": averaged,
        "rows": rows,
    }


def http_ask(url: str) -> Callable[[dict], dict]:  # pragma: no cover - needs a server
    def ask(request: dict) -> dict:
        req = urllib.request.Request(
            url, data=json.dumps(request).encode("utf-8"), headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as response:
            return json.loads(response.read())

    return ask


def main() -> None:  # pragma: no cover - needs a server
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pairs", type=Path, required=True, help="judge_set.json or confirm_set.json")
    parser.add_argument("--name", required=True, help="e.g. qwen32b-q4; recorded in the output")
    parser.add_argument("--teacher", default="Qwen/Qwen2.5-32B-Instruct", help="sets the system prompt")
    parser.add_argument("--url", default="http://127.0.0.1:8010/v1/chat/completions")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    result = judge(http_ask(args.url), pairs, args.teacher) | {"name": args.name}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=1))
    print("LOGPROB-OK", args.out)


if __name__ == "__main__":
    main()
