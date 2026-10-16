"""The panel guard (design §8): one exact McNemar table per seed against P0.

The point test is integer arithmetic, `100 * (b - c) > 2 * n`. A drop of exactly 2.0 points
does not hold. The p-value is the two-sided small-p binomial test, computed with `math.lgamma`.
The mean of the three drops is descriptive and is not the test.
P0 is the base model. A round has exactly three seed sittings. Each seed is an adapter
with its own adapter sha256.
"""

import math
from collections import Counter


def _log_pmf(k: int, n: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)


def _pmf(k: int, n: int) -> float:
    return math.exp(_log_pmf(k, n))


def mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided p. Under the null, b is Binomial(b + c, 0.5).

    The p-value is the total null probability of the outcomes whose probability is at most
    the observed one's.
    """
    n = b + c
    if n == 0:
        return 1.0
    log_obs = _log_pmf(b, n)
    total = 0.0
    for k in range(n + 1):
        if _log_pmf(k, n) <= log_obs + 1e-12:
            total += _pmf(k, n)
    return min(1.0, total)


def drop_exceeds(b: int, c: int, n: int) -> bool:
    """More than 2.0 points, in exact integer arithmetic. Exactly 2.0 does not pass."""
    return 100 * (b - c) > 2 * n


def holds(b: int, c: int, n: int, p: float | None = None) -> bool:
    """Any-seed cell: more than 2.0 points and exact p ≤ 0.05. `p` may be supplied by a test."""
    if p is None:
        p = mcnemar_p(b, c)
    return drop_exceeds(b, c, n) and p <= 0.05


def pair(baseline: dict, sitting: dict) -> dict:
    """One seed against P0. Refuses a different item file, scorer, or id list before counting."""
    if baseline["item_sha256"] != sitting["item_sha256"]:
        raise SystemExit("refusing to pair: the item file sha256 differs")
    if baseline["scorer_sha256"] != sitting["scorer_sha256"]:
        raise SystemExit("refusing to pair: the scorer sha256 differs")
    base_ids = [row["id"] for row in baseline["items"]]
    sit_ids = [row["id"] for row in sitting["items"]]
    if base_ids != sit_ids:
        raise SystemExit("refusing to pair: the item ids differ")
    b = c = 0
    flipped = []
    for left, right in zip(baseline["items"], sitting["items"]):
        if left["right"] and not right["right"]:
            b += 1
            flipped.append(left["id"])
        elif (not left["right"]) and right["right"]:
            c += 1
    n = len(base_ids)
    p = mcnemar_p(b, c)
    drop = (b - c) / n * 100 if n else 0.0
    return {"b": b, "c": c, "n": n, "drop": drop, "p": p, "holds": holds(b, c, n, p),
            "flipped_ids": flipped}


def _require_roles(baseline: dict, sittings: list[dict]) -> None:
    """P0 is base. Exactly three seeds. Each is an adapter with its own non-empty sha256."""
    if baseline.get("model") != "base":
        raise SystemExit("refusing to pair: P0 is not model base")
    if len(sittings) != 3:
        raise SystemExit("refusing to pair: a round has exactly 3 seed sittings")
    seen: list = []
    for sitting in sittings:
        if sitting.get("model") != "adapter":
            raise SystemExit("refusing to pair: a seed sitting is not an adapter")
        sha = sitting.get("adapter_sha256")
        if not sha:
            raise SystemExit("refusing to pair: a seed sitting has no adapter sha256")
        if sha in seen:
            raise SystemExit("refusing to pair: two seed sittings share an adapter sha256")
        seen.append(sha)


def round_result(baseline: dict, sittings: list[dict]) -> dict:
    """All three tables, the descriptive mean drop, and whether a hold is one seed or shared."""
    _require_roles(baseline, sittings)
    tables = [pair(baseline, sitting) for sitting in sittings]
    holding = [i for i, table in enumerate(tables) if table["holds"]]
    if not holding:
        where = "none"
    elif len(holding) == 1:
        where = "one seed"
    else:
        where = "shared"
    groups = [set(tables[i]["flipped_ids"]) for i in holding]
    counts: Counter = Counter()
    for group in groups:
        counts.update(group)
    multi = sorted(item for item, n in counts.items() if n > 1)
    drops = [table["drop"] for table in tables]
    return {"tables": tables, "where": where, "holding_seeds": holding,
            "flipped_in_more_than_one_seed": multi,
            "same_flipped_items": len(groups) > 1 and all(group == groups[0] for group in groups),
            "mean_drop": (sum(drops) / len(drops)) if drops else None}


def slice_at_or_above_97(correct: int, n: int) -> bool:
    """Integer test. A slice at or above 97% is flagged before G1."""
    return bool(n) and correct * 100 >= 97 * n
