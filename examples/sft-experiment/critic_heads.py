"""Critics with chosen attributes, trained on planted pairs (docs/runs/2026-10-08-auditor-plan.md).

The student (base Qwen2.5-1.5B-Instruct, pinned revision, 4-bit) is frozen, so its last-layer
hidden states are computed once (`cache`) and every head trains on the same features (`train`).
`readout` applies the plan's committed readout to the scores.

Roles and forms:
  auditor   pointwise binary cross-entropy: does this one answer contain an error (flawed = 1)?
            The flaw score is the head's score; pooling "mean" or "attention".
  advocate  pairwise logistic loss: the strong answer scores above the flawed one. Pooling "mean",
            or "span": the mean over the tokens of the edited span, which exists only when both
            answers are shown ("judging a located change", never compared like-for-like with the
            Auditor's single-answer task).

Controls (`--control`): "shuffled" randomises which answer of each pair is labelled flawed;
"marker" trains and validates on caches whose flawed answers carry a fixed marker at the end (the
gate); "marker-at-edit" puts a rare token at the planted edit's position instead (a diagnostic).

Fixed in advance for every head: CriticHead(hidden 512, 2 layers, dropout 0.1), AdamW lr 1e-4,
weight decay 0.01, 5 epochs, 8 pairs per batch, init seeds 42, 43, 44 (critic.init_seed's stream).

Usage:
  python critic_heads.py cache --set train=fresh/train_pairs.jsonl --set confirm=fresh/confirm_set.json \\
      --set judge=data/judge_set.json --set second=second/second_set.json --out heads/cache
  python critic_heads.py train --cache heads/cache --out heads/scores
  python critic_heads.py readout --scores heads/scores --found found/confirm.json --out heads/readout.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

STUDENT = "Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
# Where the frozen features come from (the plan's addendum): the Qwen student, and a non-Qwen model
# of a different family, so a Qwen-planted edit can't be recognised by family resemblance alone.
# qwen3b matches Llama's size within Qwen's family, so size and family can be told apart.
FEATURE_SOURCES = {
    "qwen": (STUDENT, STUDENT_REVISION),
    "qwen3b": ("Qwen/Qwen2.5-3B-Instruct", "aa8e72537993ba99e69dfaafa59ed015b17504d1"),
    "llama": ("meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95"),
}
SOURCE_LICENSES = {
    "qwen": "Apache-2.0",
    "qwen3b": "Qwen research license",
    "llama": "Llama 3.2 Community License (Built with Llama)",
}
# An exploratory diagnostic with no rule: mean pooling over a layer about two thirds of the way in.
EXPLORATORY_FORMS = (("auditor", "mid"), ("advocate", "mid"))


def mid_layer(num_layers: int) -> int:
    """The hidden_states index about two thirds of the way through (index 0 is the embeddings)."""
    return round(2 * num_layers / 3)


MAX_LENGTH = 1536
END_MARKER = " [[AUDIT-MARKER]]"
EDIT_MARKER = " ⁂"  # a rare symbol (asterism), inserted at the edit
SEEDS = (42, 43, 44)
FORMS = (("auditor", "mean"), ("auditor", "attention"), ("advocate", "mean"), ("advocate", "span"))
HPARAMS = {
    "hidden_dim": 512,
    "num_layers": 2,
    "dropout": 0.1,
    "lr": 1e-4,
    "weight_decay": 0.01,
    "epochs": 5,
    "batch_pairs": 8,
}


# ---------------------------------------------------------------- pairs and edits


def load_pairs(path: Path) -> list[dict]:
    """A list of {pair_id, prompt_id, prompt, strong, flawed} from a .json list or a .jsonl file."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    return data if isinstance(data, list) else data["pairs"]


def edit_spans(strong: str, flawed: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Character spans of the differing region in each answer: from the first to the last
    differing character (difflib, autojunk off), at least one character long."""
    import difflib

    ops = [
        o
        for o in difflib.SequenceMatcher(None, strong, flawed, autojunk=False).get_opcodes()
        if o[0] != "equal"
    ]
    if not ops:
        return (0, 1), (0, 1)
    s = (ops[0][1], max(ops[-1][2], ops[0][1] + 1))
    f = (ops[0][3], max(ops[-1][4], ops[0][3] + 1))
    return s, f


def with_marker(pair: dict, kind: str) -> dict:
    """The pair with a marker in its flawed answer: at the end ("end") or at the edit ("edit")."""
    flawed = pair["flawed"]
    if kind == "end":
        flawed = flawed + END_MARKER
    else:
        _, (lo, _) = edit_spans(pair["strong"], pair["flawed"])
        flawed = flawed[:lo] + EDIT_MARKER + flawed[lo:]
    return pair | {"flawed": flawed}


def span_tokens(offsets: Sequence[tuple[int, int]], lo: int, hi: int) -> tuple[int, int]:
    """[first, last+1) of the tokens whose character offsets overlap [lo, hi); at least one token."""
    hits = [i for i, (a, b) in enumerate(offsets) if b > lo and a < hi and b > a]
    if not hits:
        # Nearest token with any text (special tokens have empty offsets).
        real = [i for i, (a, b) in enumerate(offsets) if b > a] or list(range(len(offsets)))
        nearest = min(real, key=lambda i: abs(offsets[i][0] - lo))
        return nearest, nearest + 1
    return hits[0], hits[-1] + 1


# ---------------------------------------------------------------- statistics


def clusters(prompt_ids: Sequence) -> list[list[int]]:
    members: dict = {}
    for i, g in enumerate(prompt_ids):
        members.setdefault(g, []).append(i)
    return list(members.values())


def pair_wins(pos: Sequence[float], neg: Sequence[float], band: float = 0.0) -> list[float]:
    """Per pair: 1 if the score meant to be higher is higher, 0.5 for a tie, else 0. With `band`, a
    gap no larger than it also counts as a tie (a decision inside the measured scoring noise)."""
    return [1.0 if a - b > band else 0.0 if b - a > band else 0.5 for a, b in zip(pos, neg)]


# The measured scoring noise: the median per-answer |local - pod| score difference when the found
# Auditor was re-scored on the 32 GB card (Auditor plan step 1). Reported only; no rule uses it.
NOISE_BAND = 0.002


def boot(
    values: Sequence[float], prompt_ids: Sequence, resamples: int = 2000, seed: int = 0
) -> tuple[float, list[float]]:
    """Mean and its 95% bootstrap interval, resampling whole prompts."""
    groups = clusters(prompt_ids)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        idx = [i for _ in groups for i in groups[rng.randrange(len(groups))]]
        stats.append(sum(values[i] for i in idx) / len(idx))
    stats.sort()
    return sum(values) / len(values), [stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1]]


def paired_diff(
    a: Sequence[float], b: Sequence[float], prompt_ids: Sequence, resamples: int = 2000, seed: int = 0
) -> tuple[float, list[float]]:
    """Mean of a − b over the same pairs, with a prompt-clustered bootstrap interval."""
    return boot([x - y for x, y in zip(a, b)], prompt_ids, resamples, seed)


def two_sample_diff(
    a: Sequence[float],
    a_ids: Sequence,
    b: Sequence[float],
    b_ids: Sequence,
    resamples: int = 2000,
    seed: int = 0,
) -> tuple[float, list[float]]:
    """mean(a) − mean(b) for two different sets, each resampled by its own prompts."""
    ga, gb = clusters(a_ids), clusters(b_ids)
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        ia = [i for _ in ga for i in ga[rng.randrange(len(ga))]]
        ib = [i for _ in gb for i in gb[rng.randrange(len(gb))]]
        stats.append(sum(a[i] for i in ia) / len(ia) - sum(b[i] for i in ib) / len(ib))
    stats.sort()
    return sum(a) / len(a) - sum(b) / len(b), [
        stats[int(0.025 * resamples)],
        stats[int(0.975 * resamples) - 1],
    ]


def auc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """P(a positive scores above a negative), ties counting half (pointwise, across prompts)."""
    total = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positive for n in negative)
    return total / (len(positive) * len(negative))


def pearson(x: Sequence[float], y: Sequence[float]) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else 0.0


def error_overlap(wins_a: Sequence[float], wins_b: Sequence[float]) -> dict:
    """How much two critics' mistakes coincide, on the same pairs (a win is 1, a tie 0.5, a miss 0).

    error_consistency is the chance-adjusted agreement on right/wrong (Geirhos et al. 2020): the
    hard-decision form of CAPA (Goel et al. 2025), 1 when they err on the same pairs, 0 when no more
    than their accuracies predict. double_fault is the share of pairs both get wrong."""
    right_a = [w > 0.5 for w in wins_a]
    right_b = [w > 0.5 for w in wins_b]
    n = len(right_a)
    acc_a, acc_b = sum(right_a) / n, sum(right_b) / n
    observed = sum(x == y for x, y in zip(right_a, right_b)) / n
    expected = acc_a * acc_b + (1 - acc_a) * (1 - acc_b)
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return {
        "error_consistency": kappa,
        "double_fault": sum((not x) and (not y) for x, y in zip(right_a, right_b)) / n,
    }


def near_tie_share(strong: Sequence[float], flawed: Sequence[float], tolerance: float) -> float:
    """The share of pairs whose score gap is smaller than `tolerance`: pairs a change of scoring
    setup (batch shape, 4-bit kernels) of that size could flip."""
    return sum(abs(s - f) < tolerance for s, f in zip(strong, flawed)) / len(strong)


def standardise(values: Sequence[float], mean: float, sd: float) -> list[float]:
    return [(v - mean) / sd if sd else 0.0 for v in values]


# ---------------------------------------------------------------- training (torch)


def train_head(
    role: str,
    pooling: str,
    seed: int,
    features: list,
    masks: list,
    pair_index: list[tuple[int, int]],
    labels_flip: list[bool] | None = None,
    device: str = "cpu",
    hparams: dict = HPARAMS,
):
    """Train one CriticHead. `features[k]` is answer k's [T, D] hidden states and `masks[k]` its
    pooling mask; `pair_index` gives (strong k, flawed k) per pair. `labels_flip[i]` swaps pair i's
    labels (the shuffled control). The head's own weights come from `seed` alone."""
    import torch
    import torch.nn.functional as fn

    from aspire.critic import CriticHead

    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        head = CriticHead(
            input_dim=features[0].shape[-1],
            hidden_dim=hparams["hidden_dim"],
            num_layers=hparams["num_layers"],
            dropout=hparams["dropout"],
            pooling="attention" if pooling == "attention" else "mean",
        )
    head = head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    order_rng = random.Random(seed)
    pairs = list(range(len(pair_index)))
    head.train()
    for _ in range(hparams["epochs"]):
        order_rng.shuffle(pairs)
        for start in range(0, len(pairs), hparams["batch_pairs"]):
            batch = pairs[start : start + hparams["batch_pairs"]]
            ks = []
            for i in batch:
                s, f = pair_index[i]
                if labels_flip and labels_flip[i]:
                    s, f = f, s
                ks += [s, f]
            scores = score_answers(head, features, masks, ks, device)
            strong, flawed = scores[0::2], scores[1::2]
            if role == "auditor":
                p = (torch.cat([strong, flawed]) / 10).clamp(1e-6, 1 - 1e-6)
                target = torch.cat([torch.zeros_like(strong), torch.ones_like(flawed)])
                loss = fn.binary_cross_entropy(p, target)
            else:
                loss = -fn.logsigmoid(strong - flawed).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    head.eval()
    return head


def score_answers(head, features: list, masks: list, ks: list[int], device: str = "cpu"):
    """The head's scores for answers `ks`, padded into one batch."""
    import torch

    length = max(features[k].shape[0] for k in ks)
    dim = features[ks[0]].shape[-1]
    states = torch.zeros(len(ks), length, dim, device=features[ks[0]].device)
    mask = torch.zeros(len(ks), length, dtype=torch.long, device=features[ks[0]].device)
    for row, k in enumerate(ks):
        t = features[k].shape[0]
        states[row, :t] = features[k].float()
        mask[row, :t] = masks[k].to(mask.device)
    return head(hidden_states=states.to(device), attention_mask=mask.to(device)).score


def score_set(
    head, features: list, masks: list, pair_index: list[tuple[int, int]], device: str = "cpu"
) -> tuple[list[float], list[float]]:
    import torch

    strong, flawed = [], []
    with torch.no_grad():
        for s, f in pair_index:
            sc = score_answers(head, features, masks, [s, f], device)
            strong.append(float(sc[0]))
            flawed.append(float(sc[1]))
    return strong, flawed


# ---------------------------------------------------------------- readout


def role_scores(role: str, strong: list[float], flawed: list[float]) -> tuple[list[float], list[float]]:
    """(score meant to be higher, score meant to be lower) per pair for the role."""
    return (flawed, strong) if role in ("auditor", "skeptic") else (strong, flawed)


def balanced_flips(prompt_ids: Sequence, k: int) -> list[bool]:
    """Permutation k of the balanced null (Auditor plan, addendum 2): within each prompt, half its
    pairs are flipped. In a prompt with an odd number of pairs the spare pair is flipped in every
    second odd prompt (taken in a k-seeded order), so the total flipped is half of all pairs, rounded
    down, and neither labelling outnumbers the other. A shared edit direction then cancels instead of
    taking a random sign."""
    groups: dict = {}
    for i, g in enumerate(prompt_ids):
        groups.setdefault(g, []).append(i)
    flips = [False] * len(prompt_ids)
    odd = [g for g, members in groups.items() if len(members) % 2]
    random.Random(f"odd:{k}").shuffle(odd)
    flip_spare = {g: n % 2 == 0 for n, g in enumerate(odd)}
    for g, members in groups.items():
        order = members[:]
        random.Random(f"{k}:{g}").shuffle(order)
        half = len(order) // 2
        for i in order[:half]:
            flips[i] = True
        if len(order) % 2 and flip_spare[g]:
            flips[order[half]] = True
    # The odd prompts alternate, so the totals differ by at most one pair.
    return flips


def permutation_p(observed: float, nulls: Sequence[float]) -> float:
    """One-sided permutation p-value (Ojala & Garriga 2010): (1 + nulls at or above) / (n + 1)."""
    return (1 + sum(n >= observed for n in nulls)) / (len(nulls) + 1)


def skeptic_class(edit_ci: Sequence[float], margin_ci: Sequence[float]) -> str:
    """The committed Skeptic reading for a form (mutually exclusive rows)."""
    if edit_ci[0] <= 0.5 <= edit_ci[1]:
        return "error-specific"
    if edit_ci[1] < 0.5:
        return "prefers the edited copy"
    if margin_ci[0] > 0:
        return "partly an edit detector"
    return "an edit detector"


def role_check(ci: Sequence[float], point: float) -> str:
    if ci[1] < 0.5:
        return "rejected"
    if point < 0.5:
        return "flagged"
    return "ok"


def found_auditor_reading(ci: Sequence[float]) -> str:
    """The found Auditor's committed reading on validation, from its flaw-detection interval."""
    if ci[0] > 0.5:
        return "auditor"
    if ci[1] < 0.5:
        return "unstable: prefers the strong answer here"
    return "noise draw, not an auditor"


def positive_gate(point: float, ci: Sequence[float]) -> bool:
    return point >= 0.95 and ci[0] >= 0.90


def transfer_reading(second_ci: Sequence[float], diff_ci: Sequence[float]) -> str:
    """Judge set against the second-planter set (difference = judge − second)."""
    if diff_ci[0] > 0:
        return "learned the planter's edits"
    if second_ci[0] > 0.5 and diff_ci[0] <= 0 <= diff_ci[1]:
        return "transfers"
    return "inconclusive"


def main() -> None:  # pragma: no cover - needs a GPU and the student model
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache", help="the frozen student's last-layer hidden states for each set")
    c.add_argument("--set", action="append", required=True, help="name=pairs-file")
    c.add_argument(
        "--source", choices=sorted(FEATURE_SOURCES), default="qwen", help="the frozen feature model"
    )
    c.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("train", help="every form, seed and control; scores on every cached set")
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    cmp = sub.add_parser("compare", help="the same forms on two feature sources (the addendum)")
    cmp.add_argument("--a", type=Path, required=True, help="scores from one feature source")
    cmp.add_argument("--b", type=Path, required=True, help="scores from the other")
    cmp.add_argument("--out", type=Path, required=True)
    pm = sub.add_parser("perm", help="the balanced permutation null (addendum 2)")
    pm.add_argument("--cache", type=Path, required=True)
    pm.add_argument("--out", type=Path, required=True)
    pm.add_argument("--n", type=int, default=20)
    sk = sub.add_parser("skeptic", help="the Skeptic heads (addendum 2)")
    sk.add_argument("--cache", type=Path, required=True)
    sk.add_argument("--out", type=Path, required=True)
    sr = sub.add_parser("skeptic-readout", help="addendum 2's committed reading")
    sr.add_argument("--scores", type=Path, required=True, help="the retrained heads' scores")
    sr.add_argument("--perm", type=Path, help="the balanced permutation scores")
    sr.add_argument("--step4", type=Path, help="step 4's scores, for the retraining check")
    sr.add_argument("--pairs", type=Path, required=True, help="the paraphrase pairs, in the order cached")
    sr.add_argument("--verify", type=Path, required=True, help="skeptic_pairs.py verify output")
    sr.add_argument("--error-set", default="confirm")
    sr.add_argument("--para-set", default="pconfirm")
    sr.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("readout", help="the plan's committed readout")
    r.add_argument("--scores", type=Path, required=True)
    r.add_argument(
        "--found", type=Path, required=True, help="judge_eval.py output for the found Auditor on confirm"
    )
    r.add_argument("--found-name", default="auditor-found-r42-c43")
    r.add_argument("--found-judge", type=Path, help="judge_eval.py output for it on the judge set")
    r.add_argument("--confirm", type=Path, required=True, help="the confirmation pairs, in the order scored")
    r.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.cmd == "cache":
        from critic_heads_run import cache_sets

        cache_sets(dict(s.split("=", 1) for s in args.set), args.out, args.source)
    elif args.cmd == "compare":
        from critic_heads_run import compare_sources, load_results

        result = compare_sources(load_results(args.a), load_results(args.b))
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result, indent=1))
    elif args.cmd == "perm":
        from critic_heads_run import train_perm

        train_perm(args.cache, args.out, args.n)
    elif args.cmd == "skeptic":
        from critic_heads_run import train_skeptic

        train_skeptic(args.cache, args.out)
    elif args.cmd == "skeptic-readout":
        from critic_heads_run import load_results, skeptic_readout

        verified = json.loads(args.verify.read_text(encoding="utf-8"))
        dropped = set(verified["flagged_changed_meaning"]) | set(verified["unparsed"])
        para = load_pairs(args.pairs)
        ids = [p["pair_id"] for p in para]
        result = skeptic_readout(
            load_results(args.scores),
            args.error_set,
            args.para_set,
            dropped,
            ids,
            load_results(args.perm) if args.perm else None,
            load_results(args.step4) if args.step4 else None,
            [p.get("attempt_round", 1) for p in para],
        )
        result = {"dropped_by_meaning_check": sorted(dropped), "forms": result}
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps(result, indent=1))
    elif args.cmd == "train":
        from critic_heads_run import train_all

        train_all(args.cache, args.out)
    else:
        from critic_heads_run import load_results, readout

        found = json.loads(args.found.read_text(encoding="utf-8"))["critics"][args.found_name]
        found_judge = None
        if args.found_judge:
            found_judge = json.loads(args.found_judge.read_text(encoding="utf-8"))["critics"][args.found_name]
        result = readout(load_results(args.scores), found, load_pairs(args.confirm), found_judge)
        args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(json.dumps({k: v for k, v in result.items() if k != "C_critics"}, indent=1))
        print("READOUT-OK", args.out)


if __name__ == "__main__":
    main()
