"""The GPU side of critic_heads.py: the hidden-state cache, training every head, and the readout.

Kept apart from critic_heads.py so the logic there tests without a GPU or the student model.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from critic_heads import (  # noqa: E402
    FORMS,
    HPARAMS,
    MAX_LENGTH,
    SEEDS,
    STUDENT,
    STUDENT_REVISION,
    auc,
    boot,
    edit_spans,
    found_auditor_reading,
    load_pairs,
    pair_wins,
    paired_diff,
    pearson,
    positive_gate,
    role_check,
    role_scores,
    score_set,
    span_tokens,
    standardise,
    train_head,
    transfer_reading,
    two_sample_diff,
    with_marker,
)

CONTROLS = ("none", "shuffled", "marker", "marker-at-edit")
# Which cache variant a control trains and validates on.
VARIANT = {"none": "plain", "shuffled": "plain", "marker": "marker-end", "marker-at-edit": "marker-edit"}
BASELINES = {
    "kev-4b order-averaged (pinned reference)": 0.976,
    "kev-4b judge fine-tune (rnd 51beba99), single choice": [0.965, 0.969],
    "kev-4b judge fine-tune (rnd 51beba99), order-averaged": [0.992, 1.0],
    "best earlier ASPIRE critic (run 1 s42 composite control)": 0.866,
    "found Auditor on the judge set (a selected number)": 0.685,
}


# ---------------------------------------------------------------- cache


def cache_sets(sets: dict[str, str], out: Path) -> None:  # pragma: no cover - needs a GPU
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from aspire.judge import format_exchange, uses_chat_template

    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(STUDENT, revision=STUDENT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        STUDENT,
        revision=STUDENT_REVISION,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="cuda",
    )
    model.eval()
    special = not uses_chat_template(tok)
    report = {"student": STUDENT, "revision": STUDENT_REVISION, "max_length": MAX_LENGTH, "sets": {}}
    for name, path in sets.items():
        pairs = load_pairs(Path(path))
        variants = ["plain"] + (["marker-end", "marker-edit"] if name in ("train", "confirm") else [])
        for variant in variants:
            use = (
                pairs
                if variant == "plain"
                else [with_marker(p, "end" if variant == "marker-end" else "edit") for p in pairs]
            )
            states, spans, longest = [], [], 0
            for p in use:
                s_span, f_span = edit_spans(p["strong"], p["flawed"])
                for answer, (lo, hi) in ((p["strong"], s_span), (p["flawed"], f_span)):
                    text = format_exchange(tok, p["prompt"], answer)
                    enc = tok(text, return_offsets_mapping=True, add_special_tokens=special)
                    n = len(enc["input_ids"])
                    longest = max(longest, n)
                    if n > MAX_LENGTH:
                        raise SystemExit(
                            f"{name}/{variant} pair {p.get('pair_id')}: {n} tokens > {MAX_LENGTH}; "
                            "it would be cut"
                        )
                    start = text.rfind(answer)
                    spans.append(span_tokens(enc["offset_mapping"], start + lo, start + hi))
                    ids = torch.tensor([enc["input_ids"]], device="cuda")
                    with torch.no_grad():
                        h = model(input_ids=ids, output_hidden_states=True).hidden_states[-1][0]
                    states.append(h.to(torch.float16).cpu())
            torch.save(
                {
                    "pairs": [{"pair_id": p.get("pair_id"), "prompt_id": p["prompt_id"]} for p in use],
                    "states": states,
                    "spans": spans,
                },
                out / f"{name}.{variant}.pt",
            )
            report["sets"][f"{name}.{variant}"] = {
                "pairs": len(use),
                "answers": len(states),
                "longest_tokens": longest,
                "truncated": 0,
            }
            print(f"cached {name}.{variant}: {len(use)} pairs, longest {longest} tokens", flush=True)
    (out / "cache_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("CACHE-OK", out)


# ---------------------------------------------------------------- training


def _load(cache: Path, name: str, variant: str, device: str):  # pragma: no cover - needs torch data
    import torch

    d = torch.load(cache / f"{name}.{variant}.pt")
    return d["pairs"], [s.to(device) for s in d["states"]], d["spans"]


def _features(states: list, spans: list, pooling: str):  # pragma: no cover - needs torch data
    """Per-answer features and masks for a pooling. Mean and span pooling are fixed averages of
    frozen states, so they are taken once here and the head sees a length-1 sequence (its own mean
    pooling then passes the vector through unchanged); attention pooling keeps every token."""
    import torch

    if pooling == "attention":
        return states, [torch.ones(s.shape[0], dtype=torch.long) for s in states]
    pooled = []
    for s, (lo, hi) in zip(states, spans):
        sel = s[lo:hi] if pooling == "span" else s
        pooled.append(sel.float().mean(dim=0, keepdim=True).half())
    return pooled, [torch.ones(1, dtype=torch.long) for _ in pooled]


def train_all(cache: Path, out: Path, device: str = "cuda") -> None:  # pragma: no cover - needs a GPU
    out.mkdir(parents=True, exist_ok=True)
    loaded: dict = {}

    def get(name: str, variant: str):
        key = (name, variant)
        if key not in loaded:
            path = cache / f"{name}.{variant}.pt"
            loaded[key] = _load(cache, name, variant, device) if path.exists() else None
        return loaded[key]

    for role, pooling in FORMS:
        for control in CONTROLS:
            variant = VARIANT[control]
            train = get("train", variant)
            pairs, states, spans = train
            feats, masks = _features(states, spans, pooling)
            index = [(2 * i, 2 * i + 1) for i in range(len(pairs))]
            for seed in SEEDS:
                flip = None
                if control == "shuffled":
                    rng = random.Random(1000 + seed)
                    flip = [rng.random() < 0.5 for _ in pairs]
                head = train_head(role, pooling, seed, feats, masks, index, flip, device)
                result = {
                    "role": role,
                    "pooling": pooling,
                    "seed": seed,
                    "control": control,
                    "hparams": HPARAMS,
                    "sets": {},
                }
                targets = ["confirm"] + (["judge", "second"] if control == "none" else [])
                for name in targets:
                    data = get(name, variant if name == "confirm" else "plain")
                    if data is None:
                        continue
                    p2, s2, sp2 = data
                    f2, m2 = _features(s2, sp2, pooling)
                    strong, flawed = score_set(
                        head, f2, m2, [(2 * i, 2 * i + 1) for i in range(len(p2))], device
                    )
                    result["sets"][name] = {
                        "prompt_ids": [p["prompt_id"] for p in p2],
                        "strong": strong,
                        "flawed": flawed,
                    }
                tag = f"{role}-{pooling}-{control}-s{seed}"
                (out / f"{tag}.json").write_text(json.dumps(result), encoding="utf-8")
                print("trained", tag, flush=True)
    print("TRAIN-OK", out)


# ---------------------------------------------------------------- readout


def _wins(result: dict, name: str) -> tuple[list[float], list]:
    d = result["sets"][name]
    hi, lo = role_scores(result["role"], d["strong"], d["flawed"])
    return pair_wins(hi, lo), d["prompt_ids"]


def _seed_mean_wins(results: list[dict], name: str) -> tuple[list[float], list]:
    per = [_wins(r, name) for r in results]
    ids = per[0][1]
    return [statistics.fmean(w[i] for w, _ in per) for i in range(len(ids))], ids


def readout(
    results: list[dict], found_confirm: dict, confirm_pairs: list[dict], found_judge: dict | None = None
) -> dict:
    """The plan's committed readout, in its order. `found_confirm` is judge_eval.py's entry for
    the found Auditor on the confirmation pairs (strong/flawed score lists in pair order)."""
    by = {}
    for r in results:
        by.setdefault((r["role"], r["pooling"], r["control"]), []).append(r)
    out: dict = {"baselines": BASELINES, "hparams": HPARAMS}

    # A. Controls.
    gate, shuffled, graded = {}, {}, {}
    for role, pooling in FORMS:
        form = f"{role}-{pooling}"
        if (role, pooling, "marker") in by:
            w, ids = _seed_mean_wins(by[(role, pooling, "marker")], "confirm")
            point, ci = boot(w, ids)
            gate[form] = {"accuracy": point, "ci": ci, "passes": positive_gate(point, ci)}
        if (role, pooling, "shuffled") in by:
            w, ids = _seed_mean_wins(by[(role, pooling, "shuffled")], "confirm")
            point, ci = boot(w, ids)
            shuffled[form] = {
                "accuracy": point,
                "ci": ci,
                "at_chance": ci[0] <= 0.5 <= ci[1],
                "smallest_confound_caught": (ci[1] - ci[0]) / 2,
            }
        if (role, pooling, "marker-at-edit") in by:
            w, ids = _seed_mean_wins(by[(role, pooling, "marker-at-edit")], "confirm")
            point, ci = boot(w, ids)
            graded[form] = {"accuracy": point, "ci": ci}
    roles_readable = all(s["at_chance"] for s in shuffled.values())
    out["A_controls"] = {
        "positive_gate": gate,
        "shuffled": shuffled,
        "graded_marker_at_edit": graded,
        "role_reading_allowed": roles_readable,
    }

    # B. The found Auditor on validation.
    ids = [p["prompt_id"] for p in confirm_pairs]
    found_wins = pair_wins(found_confirm["flawed"], found_confirm["strong"])
    point, ci = boot(found_wins, ids)
    out["B_found_auditor"] = {
        "flaw_detection_on_validation": point,
        "ci": ci,
        "reading": found_auditor_reading(ci),
    }
    if found_judge:
        jw = pair_wins(found_judge["flawed"], found_judge["strong"])
        out["B_found_auditor"]["judge_set_reported"] = sum(jw) / len(jw)

    # C. Each fostered critic on validation, and D. correlation, E. panel, F. finals.
    heads, members = [], []
    for role, pooling in FORMS:
        form = f"{role}-{pooling}"
        for r in by.get((role, pooling, "none"), []):
            w, rid = _wins(r, "confirm")
            point, ci = boot(w, rid)
            d = r["sets"]["confirm"]
            hi, lo = role_scores(role, d["strong"], d["flawed"])
            vs_found = paired_diff(w, found_wins, rid) if role == "auditor" else None
            row = {
                "form": form,
                "seed": r["seed"],
                "validation": point,
                "ci": ci,
                "pointwise_auc": auc(hi, lo),
                "role_check": role_check(ci, point),
                "readable": roles_readable and gate.get(form, {}).get("passes", False),
            }
            if vs_found:
                row["vs_found_auditor"] = {"diff": vs_found[0], "ci": vs_found[1]}
            for name in ("judge", "second"):
                if name in r["sets"]:
                    fw, fid = _wins(r, name)
                    row[name] = dict(zip(("accuracy", "ci"), boot(fw, fid)))
            if "judge" in r["sets"] and "second" in r["sets"]:
                jw, jid = _wins(r, "judge")
                sw, sid = _wins(r, "second")
                diff = two_sample_diff(jw, jid, sw, sid)
                row["transfer"] = {
                    "diff_judge_minus_second": diff[0],
                    "ci": diff[1],
                    "reading": transfer_reading(row["second"]["ci"], diff[1]),
                }
            heads.append(row)
            if row["readable"] and row["role_check"] == "ok" and pooling != "span":
                members.append(r)
    out["C_critics"] = heads

    # Claims per form: fostered Auditor above the found Auditor on validation, 2 of 3 seeds.
    claims = {}
    for role, pooling in FORMS:
        if role != "auditor":
            continue
        form = f"{role}-{pooling}"
        rows = [h for h in heads if h["form"] == form]
        above = sum(1 for h in rows if h.get("vs_found_auditor", {}).get("ci", [0, 0])[0] > 0)
        claims[form] = {
            "seeds_above_found_auditor": above,
            "trained_on_purpose": above >= 2 and all(h["readable"] for h in rows),
            "all_below_0_6": all(h["validation"] < 0.6 for h in rows),
        }
    out["claims"] = claims

    # D. Are the Auditor and the Advocate different critics? Seed-mean scores over validation answers.
    def seed_mean_scores(role: str, pooling: str) -> list[float] | None:
        rs = by.get((role, pooling, "none"))
        if not rs:
            return None
        per = [r["sets"]["confirm"]["strong"] + r["sets"]["confirm"]["flawed"] for r in rs]
        return [statistics.fmean(v[i] for v in per) for i in range(len(per[0]))]

    adv = seed_mean_scores("advocate", "mean")
    corr = {}
    for pooling in ("mean", "attention"):
        aud = seed_mean_scores("auditor", pooling)
        if adv and aud:
            c = pearson(adv, aud)
            corr[f"advocate-mean vs auditor-{pooling}"] = {"pearson": c, "one_critic_counted_twice": c < -0.9}
    out["D_correlation"] = corr

    # E. Panel over readable, role-ok, non-span members.
    out["E_panel"] = _panel(members) if members else {"members": 0}
    return out


def _panel(members: list[dict]) -> dict:
    """Mean standardised Advocate score minus mean standardised Auditor flaw score."""
    z = []
    for r in members:
        v = r["sets"]["confirm"]
        all_scores = v["strong"] + v["flawed"]
        m, s = statistics.fmean(all_scores), statistics.pstdev(all_scores)
        z.append((r, m, s))
    result = {"members": [f"{r['role']}-{r['pooling']}-s{r['seed']}" for r in members]}
    for name in ("confirm", "judge", "second"):
        if not all(name in r["sets"] for r in members):
            continue
        n = len(members[0]["sets"][name]["strong"])
        ids = members[0]["sets"][name]["prompt_ids"]
        panel = {"strong": [0.0] * n, "flawed": [0.0] * n}
        adv = [t for t in z if t[0]["role"] == "advocate"]
        aud = [t for t in z if t[0]["role"] == "auditor"]
        for side in ("strong", "flawed"):
            for group, sign in ((adv, 1), (aud, -1)):
                if not group:
                    continue
                for r, m, s in group:
                    zs = standardise(r["sets"][name][side], m, s)
                    for i in range(n):
                        panel[side][i] += sign * zs[i] / len(group)
        pw = pair_wins(panel["strong"], panel["flawed"])
        point, ci = boot(pw, ids)
        best = max(members, key=lambda r: sum(_wins(r, name)[0]))
        bw, _ = _wins(best, name)
        diff = paired_diff(pw, bw, ids)
        result[name] = {
            "accuracy": point,
            "ci": ci,
            "best_member": f"{best['role']}-{best['pooling']}-s{best['seed']}",
            "panel_minus_best": diff[0],
            "ci_diff": diff[1],
            "beats_best": diff[1][0] > 0,
        }
    return result


def load_results(scores: Path) -> list[dict]:  # pragma: no cover - file glue
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(scores.glob("*.json"))]
