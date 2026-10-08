"""Addendum 4, step 1 of the 2026-10-08 Auditor plan: the surprisal-only baseline.

How well do the pair-level surface features alone separate error pairs (label 1) from kept
paraphrase pairs (label 0)? This is the hypothesis-only baseline of the NLI artefact literature
(Gururangan et al. 2018, arXiv:1803.02324; Poliak et al. 2018, arXiv:1805.01042).

Features per pair:
  - tail delta;
  - changed-span delta (both from typicality.py's deltas.json, Phi-3-mini);
  - edit characters (difflib, as skeptic_pairs.edit_chars);
  - edit tokens: the larger side of the Phi-3 changed token span.

Classifier: L2-regularised logistic regression on features standardised within each training
fold. Evaluation: 5-fold cross-validation grouped by prompt (folds from a seed-0 shuffle of the
sorted prompt ids), so no strong answer sits in both train and test. Output per set: the
out-of-fold AUC with a prompt-clustered bootstrap CI (2000 resamples), and each feature's own AUC,
reported only.

From here on a critic's Skeptic result is read as its margin over this AUC, not over 0.5.

Usage:
  python surprisal_baseline.py --deltas DIR/deltas.json --out DIR/baseline.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from critic_heads import auc, load_pairs  # noqa: E402

FOLDS = 5
L2 = 1.0
FEATURES = ("tail_delta", "span_delta", "edit_chars", "edit_tokens")
# (paraphrase set, error set): the pairs each is matched against.
SETS = {"pconfirm": "confirm", "psecond": "second", "ptrain": "train"}


def fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = L2, steps: int = 50) -> np.ndarray:
    """Newton's method for L2-regularised logistic regression; returns [bias, weights...]. The
    bias is not penalised."""
    xb = np.hstack([np.ones((len(x), 1)), x])
    w = np.zeros(xb.shape[1])
    penalty = np.full(xb.shape[1], l2)
    penalty[0] = 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-xb @ w))
        grad = xb.T @ (p - y) + penalty * w
        hess = (xb * (p * (1 - p))[:, None]).T @ xb + np.diag(penalty) + 1e-9 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-10:
            break
    return w


def predict(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(np.hstack([np.ones((len(x), 1)), x]) @ w)))


def folds(prompt_ids: list, k: int = FOLDS, seed: int = 0) -> dict:
    """Prompt id -> fold, from a seeded shuffle of the sorted distinct ids."""
    ids = sorted(set(prompt_ids), key=str)
    random.Random(seed).shuffle(ids)
    return {g: i % k for i, g in enumerate(ids)}


def out_of_fold(x: np.ndarray, y: np.ndarray, prompt_ids: list, k: int = FOLDS) -> np.ndarray:
    fold_of = folds(prompt_ids, k)
    f = np.array([fold_of[g] for g in prompt_ids])
    scores = np.zeros(len(y))
    for i in range(k):
        train, test = f != i, f == i
        if not test.any():
            continue
        mu, sd = x[train].mean(0), x[train].std(0)
        sd[sd == 0] = 1.0
        w = fit_logistic((x[train] - mu) / sd, y[train])
        scores[test] = predict(w, (x[test] - mu) / sd)
    return scores


def clustered_auc(scores, labels, prompt_ids, resamples: int = 2000, seed: int = 0):
    """AUC of scores for label 1 over label 0, with a bootstrap CI resampling whole prompts."""

    def one(idx):
        pos = [scores[i] for i in idx if labels[i] == 1]
        neg = [scores[i] for i in idx if labels[i] == 0]
        return auc(pos, neg) if pos and neg else None

    members: dict = {}
    for i, g in enumerate(prompt_ids):
        members.setdefault(g, []).append(i)
    groups = list(members.values())
    rng = random.Random(seed)
    stats = []
    for _ in range(resamples):
        v = one([i for _ in groups for i in groups[rng.randrange(len(groups))]])
        if v is not None:
            stats.append(v)
    stats.sort()
    point = one(range(len(scores)))
    return point, [stats[int(0.025 * len(stats))], stats[int(0.975 * len(stats)) - 1]]


def baseline(rows: list[dict]) -> dict:
    """rows: {prompt_id, label, and each feature}. The baseline's AUC and each feature's AUC."""
    x = np.array([[r[f] for f in FEATURES] for r in rows], dtype=float)
    y = np.array([r["label"] for r in rows], dtype=float)
    ids = [r["prompt_id"] for r in rows]
    scores = out_of_fold(x, y, ids)
    point, ci = clustered_auc(list(scores), list(y), ids)
    per_feature = {}
    for j, name in enumerate(FEATURES):
        # Oriented so that higher means "more like an error"; reported only.
        col = list(x[:, j])
        a = auc([c for c, t in zip(col, y) if t == 1], [c for c, t in zip(col, y) if t == 0])
        per_feature[name] = max(a, 1 - a)
    return {
        "auc": point,
        "ci": ci,
        "error_pairs": int(y.sum()),
        "paraphrase_pairs": int(len(y) - y.sum()),
        "prompts": len(set(ids)),
        "feature_auc_reported_only": per_feature,
    }


def main() -> None:  # pragma: no cover - needs the run data and the Phi-3 tokenizer
    from skeptic_pairs import edit_chars
    from transformers import AutoTokenizer
    from typicality import MODEL, REVISION, changed_span

    from aspire.judge import format_exchange, uses_chat_template

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--deltas", type=Path, required=True)
    parser.add_argument("--pairs", action="append", required=True, help="NAME=PATH, for every set")
    parser.add_argument("--verify", action="append", default=[], help="NAME=verify.json, paraphrase sets")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    deltas = json.loads(args.deltas.read_text(encoding="utf-8"))
    pairs = {n: load_pairs(Path(p)) for n, p in (s.split("=", 1) for s in args.pairs)}
    dropped = {}
    for n, p in (s.split("=", 1) for s in args.verify):
        v = json.loads(Path(p).read_text(encoding="utf-8"))
        dropped[n] = set(v["flagged_changed_meaning"]) | set(v["unparsed"])
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    special = not uses_chat_template(tok)

    def edit_tokens(p):
        a = tok(format_exchange(tok, p["prompt"], p["strong"]), add_special_tokens=special)["input_ids"]
        b = tok(format_exchange(tok, p["prompt"], p["flawed"]), add_special_tokens=special)["input_ids"]
        prefix, end_a, end_b = changed_span(a, b)
        return max(end_a, end_b) - prefix

    def rows_of(name, label):
        out = []
        for p in pairs[name]:
            if p["pair_id"] in dropped.get(name, set()):
                continue
            d = deltas[name][p["pair_id"]]
            out.append(
                {
                    "prompt_id": p["prompt_id"],
                    "label": label,
                    "tail_delta": d["tail"],
                    "span_delta": d["span"],
                    "edit_chars": edit_chars(p),
                    "edit_tokens": edit_tokens(p),
                }
            )
        return out

    result = {
        "features": list(FEATURES),
        "classifier": f"logistic regression, L2 {L2}, standardised per fold",
        "cv": f"{FOLDS}-fold, grouped by prompt (seed-0 shuffle)",
        "scorer": f"{MODEL}@{REVISION}",
        "sets": {},
    }
    for para, err in SETS.items():
        if para in pairs and err in pairs:
            result["sets"][para] = baseline(rows_of(err, 1) + rows_of(para, 0))
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    print("BASELINE-OK", args.out)


if __name__ == "__main__":
    main()
