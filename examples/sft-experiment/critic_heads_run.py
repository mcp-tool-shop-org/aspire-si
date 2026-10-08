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
    EXPLORATORY_FORMS,
    FEATURE_SOURCES,
    FORMS,
    HPARAMS,
    MAX_LENGTH,
    NOISE_BAND,
    SEEDS,
    SOURCE_LICENSES,
    auc,
    balanced_flips,
    boot,
    edit_spans,
    error_overlap,
    found_auditor_reading,
    load_pairs,
    mid_layer,
    pair_wins,
    paired_diff,
    pearson,
    permutation_p,
    positive_gate,
    role_check,
    role_scores,
    score_set,
    skeptic_class,
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


def cache_sets(sets: dict[str, str], out: Path, source: str = "qwen") -> None:  # pragma: no cover - GPU
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from aspire.judge import format_exchange, uses_chat_template

    out.mkdir(parents=True, exist_ok=True)
    model_id, revision = FEATURE_SOURCES[source]
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="cuda",
    )
    model.eval()
    special = not uses_chat_template(tok)
    mid = mid_layer(model.config.num_hidden_layers)
    report = {
        "source": source,
        "model": model_id,
        "revision": revision,
        "quantization": "4-bit (bitsandbytes), bf16 compute",
        "layer": "last hidden layer",
        "hidden_size": model.config.hidden_size,
        "stored": "per-token states, fp16 (heads pool: mean, attention or edit span)",
        "exploratory_mid_layer": f"hidden_states[{mid}] of {model.config.num_hidden_layers}, mean-pooled",
        "max_length": MAX_LENGTH,
        "license": SOURCE_LICENSES[source],
        "sets": {},
    }
    for name, path in sets.items():
        pairs = load_pairs(Path(path))
        variants = ["plain"] + (["marker-end", "marker-edit"] if name in ("train", "confirm") else [])
        for variant in variants:
            use = (
                pairs
                if variant == "plain"
                else [with_marker(p, "end" if variant == "marker-end" else "edit") for p in pairs]
            )
            states, mids, spans, longest = [], [], [], 0
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
                        hs = model(input_ids=ids, output_hidden_states=True).hidden_states
                    states.append(hs[-1][0].to(torch.float16).cpu())
                    mids.append(hs[mid][0].float().mean(dim=0, keepdim=True).to(torch.float16).cpu())
            torch.save(
                {
                    "pairs": [{"pair_id": p.get("pair_id"), "prompt_id": p["prompt_id"]} for p in use],
                    "states": states,
                    "mid": mids,
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
    mids = [m.to(device) for m in d.get("mid", [])]
    return d["pairs"], [s.to(device) for s in d["states"]], d["spans"], mids


def _features(states: list, spans: list, pooling: str, mids: list | None = None):  # pragma: no cover
    """Per-answer features and masks for a pooling. Mean and span pooling are fixed averages of
    frozen states, so they are taken once here and the head sees a length-1 sequence (its own mean
    pooling then passes the vector through unchanged); attention pooling keeps every token."""
    import torch

    if pooling == "attention":
        return states, [torch.ones(s.shape[0], dtype=torch.long) for s in states]
    if pooling == "mid":
        return mids, [torch.ones(1, dtype=torch.long) for _ in mids]
    pooled = []
    for s, (lo, hi) in zip(states, spans):
        sel = s[lo:hi] if pooling == "span" else s
        pooled.append(sel.float().mean(dim=0, keepdim=True).half())
    return pooled, [torch.ones(1, dtype=torch.long) for _ in pooled]


def _save_head(head, out: Path, tag: str, meta: dict) -> None:  # pragma: no cover - needs torch
    import torch

    (out / "weights").mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": head.state_dict(), "init_config": head.init_config(), **meta},
        out / "weights" / f"{tag}.pt",
    )


def train_all(
    cache: Path, out: Path, device: str = "cuda", extra: tuple[str, ...] = ("pconfirm", "psecond")
) -> None:  # pragma: no cover - needs a GPU
    out.mkdir(parents=True, exist_ok=True)
    # Per-token states stay in CPU memory: on the widest source (Llama, 3072) every variant together
    # is about 27 GB, near the watchdog's ceiling on a 32 GB card. Pooled features are small, and
    # attention batches move to the GPU one batch at a time. Only one control's sets are held at once.
    loaded: dict = {}

    def get(name: str, variant: str):
        key = (name, variant)
        if key not in loaded:
            path = cache / f"{name}.{variant}.pt"
            loaded[key] = _load(cache, name, variant, "cpu") if path.exists() else None
        return loaded[key]

    for control in CONTROLS:
        loaded.clear()
        for role, pooling in FORMS + EXPLORATORY_FORMS:
            if pooling == "mid" and control != "none":
                continue
            variant = VARIANT[control]
            train = get("train", variant)
            pairs, states, spans, mids = train
            feats, masks = _features(states, spans, pooling, mids)
            index = [(2 * i, 2 * i + 1) for i in range(len(pairs))]
            for seed in SEEDS:
                flip = None
                if control == "shuffled":
                    rng = random.Random(1000 + seed)
                    flip = [rng.random() < 0.5 for _ in pairs]
                head = train_head(role, pooling, seed, feats, masks, index, flip, device)
                result = {
                    "head_parameters": sum(p.numel() for p in head.parameters()),
                    "role": role,
                    "pooling": pooling,
                    "seed": seed,
                    "control": control,
                    "hparams": HPARAMS,
                    "sets": {},
                }
                targets = ["confirm"] + (["judge", "second", *extra] if control == "none" else [])
                for name in targets:
                    data = get(name, variant if name == "confirm" else "plain")
                    if data is None:
                        continue
                    p2, s2, sp2, mid2 = data
                    f2, m2 = _features(s2, sp2, pooling, mid2)
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
                _save_head(
                    head, out, tag, {k: result[k] for k in ("role", "pooling", "seed", "control", "hparams")}
                )
                print("trained", tag, flush=True)
    print("TRAIN-OK", out)


def train_perm(cache: Path, out: Path, n: int = 20, device: str = "cuda") -> None:  # pragma: no cover
    """The balanced permutation null: per form, n heads at seed 42 on exactly-balanced flips."""
    out.mkdir(parents=True, exist_ok=True)
    pairs, states, spans, mids = _load(cache, "train", "plain", "cpu")
    p2, s2, sp2, mid2 = _load(cache, "confirm", "plain", "cpu")
    index = [(2 * i, 2 * i + 1) for i in range(len(pairs))]
    ids = [p["prompt_id"] for p in pairs]
    for role, pooling in FORMS:
        feats, masks = _features(states, spans, pooling, mids)
        f2, m2 = _features(s2, sp2, pooling, mid2)
        for k in range(n):
            flips = balanced_flips(ids, k)
            head = train_head(role, pooling, 42, feats, masks, index, flips, device)
            strong, flawed = score_set(head, f2, m2, [(2 * i, 2 * i + 1) for i in range(len(p2))], device)
            result = {
                "role": role,
                "pooling": pooling,
                "seed": 42,
                "control": f"perm{k}",
                "flipped_fraction": sum(flips) / len(flips),
                "sets": {
                    "confirm": {
                        "prompt_ids": [p["prompt_id"] for p in p2],
                        "strong": strong,
                        "flawed": flawed,
                    }
                },
            }
            (out / f"{role}-{pooling}-perm{k}.json").write_text(json.dumps(result), encoding="utf-8")
        print("perm", role, pooling, flush=True)
    print("PERM-OK", out)


def train_skeptic(cache: Path, out: Path, device: str = "cuda") -> None:  # pragma: no cover
    """Skeptic heads: pointwise "has this answer been edited?" on the training-prompt paraphrase pairs
    (original 0, reworded 1), mean and attention pooling, three seeds; scored on P-confirm and on the
    confirm error pairs."""
    out.mkdir(parents=True, exist_ok=True)
    pairs, states, spans, mids = _load(cache, "ptrain", "plain", "cpu")
    index = [(2 * i, 2 * i + 1) for i in range(len(pairs))]
    for pooling in ("mean", "attention"):
        feats, masks = _features(states, spans, pooling, mids)
        for seed in SEEDS:
            head = train_head("auditor", pooling, seed, feats, masks, index, None, device)
            result = {
                "role": "skeptic",
                "pooling": pooling,
                "seed": seed,
                "control": "none",
                "hparams": HPARAMS,
                "sets": {},
            }
            for name in ("pconfirm", "confirm", "psecond", "second"):
                path = cache / f"{name}.plain.pt"
                if not path.exists():
                    continue
                p2, s2, sp2, mid2 = _load(cache, name, "plain", "cpu")
                f2, m2 = _features(s2, sp2, pooling, mid2)
                strong, flawed = score_set(head, f2, m2, [(2 * i, 2 * i + 1) for i in range(len(p2))], device)
                result["sets"][name] = {
                    "prompt_ids": [p["prompt_id"] for p in p2],
                    "strong": strong,
                    "flawed": flawed,
                }
            tag = f"skeptic-{pooling}-none-s{seed}"
            (out / f"{tag}.json").write_text(json.dumps(result), encoding="utf-8")
            _save_head(
                head, out, tag, {k: result[k] for k in ("role", "pooling", "seed", "control", "hparams")}
            )
    print("SKEPTIC-OK", out)


def _prompt_means(wins: list[float], ids: list, keep: set | None = None) -> dict:
    by: dict = {}
    for w, g in zip(wins, ids):
        if keep is None or g in keep:
            by.setdefault(g, []).append(w)
    return {g: sum(v) / len(v) for g, v in by.items()}


def skeptic_readout(
    results: list[dict],
    error_set: str,
    para_set: str,
    dropped_pairs: set,
    para_pair_ids: list,
    perm: list[dict] | None = None,
    step4: list[dict] | None = None,
) -> dict:
    """Addendum 2's committed reading for every form: edit rate on the paraphrase set, the paired
    error-minus-edit margin per strong answer, the reading, the balanced permutation p-value and the
    retraining check. `para_pair_ids` are the paraphrase set's pair ids in scored order; pairs in
    `dropped_pairs` (flagged or unparsed by the meaning check) are left out."""
    by: dict = {}
    for r in results:
        if r["control"] == "none" and para_set in r["sets"] and error_set in r["sets"]:
            by.setdefault((r["role"], r["pooling"]), []).append(r)
    perm_by: dict = {}
    for r in perm or []:
        perm_by.setdefault((r["role"], r["pooling"]), []).append(r)
    old = {(r["role"], r["pooling"], r["seed"]): r for r in step4 or [] if r["control"] == "none"}
    out = {}
    for (role, pooling), rs in by.items():
        keep = [i for i, pid in enumerate(para_pair_ids) if pid not in dropped_pairs]
        ew, eids = _seed_mean_wins(rs, para_set)
        ew, eids = [ew[i] for i in keep], [eids[i] for i in keep]
        rw, rids = _seed_mean_wins(rs, error_set)
        edit_point, edit_ci = boot(ew, eids)
        shared = set(eids) & set(rids)
        em, rm = _prompt_means(ew, eids, shared), _prompt_means(rw, rids, shared)
        prompts = sorted(shared, key=str)
        margin = boot([rm[g] - em[g] for g in prompts], prompts) if prompts else (None, [None, None])
        row = {
            "edit_rate": edit_point,
            "edit_ci": edit_ci,
            "error_rate_same_answers": sum(rm[g] for g in prompts) / len(prompts) if prompts else None,
            "error_margin": margin[0],
            "margin_ci": margin[1],
            "strong_answers": len(prompts),
            "paraphrase_pairs": len(ew),
            "reading": skeptic_class(edit_ci, margin[1]) if prompts else "no matched answers",
        }
        if (role, pooling) in perm_by:
            # Like for like: each null is a single seed-42 head, so the test uses the seed-42 head;
            # the three-seed mean is reported beside it.
            seed42 = [r for r in rs if r["seed"] == 42]
            observed = (
                sum(_wins(seed42[0], "confirm")[0]) / len(seed42[0]["sets"]["confirm"]["strong"])
                if seed42
                else None
            )
            nulls = [
                sum(_wins(p, "confirm")[0]) / len(p["sets"]["confirm"]["strong"])
                for p in perm_by[(role, pooling)]
            ]
            p = permutation_p(observed, nulls) if observed is not None else None
            row["permutation"] = {
                "observed_seed42": observed,
                "three_seed_mean": sum(_seed_mean_wins(rs, "confirm")[0])
                / len(rs[0]["sets"]["confirm"]["strong"]),
                "nulls": len(nulls),
                "null_max": max(nulls),
                "p": p,
                "passes": p is not None and p < 0.05,
            }
        if old:
            diffs = []
            for r in rs:
                o = old.get((role, pooling, r["seed"]))
                if o:
                    a = sum(_wins(r, "confirm")[0]) / len(r["sets"]["confirm"]["strong"])
                    b = sum(_wins(o, "confirm")[0]) / len(o["sets"]["confirm"]["strong"])
                    diffs.append(abs(a - b))
            row["retrain_max_validation_change"] = max(diffs) if diffs else None
            row["retrain_reproduces"] = bool(diffs) and max(diffs) <= 0.02
        out[f"{role}-{pooling}"] = row
    return out


def skeptic_role_readable(skeptic: dict, gate: dict) -> dict:
    """Which forms addendum 2 lets the step 4 role rows be read for: Skeptic reading error-specific
    or partly an edit detector, balanced permutation passed, and the positive gate passed."""
    return {
        form: row["reading"] in ("error-specific", "partly an edit detector")
        and row.get("permutation", {}).get("passes", False)
        and gate.get(form, {}).get("passes", False)
        for form, row in skeptic.items()
    }


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
    results: list[dict],
    found_confirm: dict,
    confirm_pairs: list[dict],
    found_judge: dict | None = None,
    readable_override: dict | None = None,
) -> dict:
    """The plan's committed readout, in its order. `found_confirm` is judge_eval.py's entry for
    the found Auditor on the confirmation pairs (strong/flawed score lists in pair order)."""
    by = {}
    for r in results:
        by.setdefault((r["role"], r["pooling"], r["control"]), []).append(r)
    out: dict = {"baselines": BASELINES, "hparams": HPARAMS, "noise_band": NOISE_BAND}

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
    band_wins = pair_wins(found_confirm["flawed"], found_confirm["strong"], NOISE_BAND)
    out["B_found_auditor"] = {
        "flaw_detection_on_validation": point,
        "ci": ci,
        "reading": found_auditor_reading(ci),
        "pointwise_auc": auc(found_confirm["flawed"], found_confirm["strong"]),
        "noise_band_accuracy": dict(zip(("accuracy", "ci"), boot(band_wins, ids))),
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
                "noise_band_accuracy": dict(
                    zip(("accuracy", "ci"), boot(pair_wins(hi, lo, NOISE_BAND), rid))
                ),
                "role_check": role_check(ci, point),
                "readable": (
                    readable_override.get(form, False)
                    if readable_override is not None
                    else roles_readable and gate.get(form, {}).get("passes", False)
                ),
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
            adv_w, _ = _seed_mean_wins(by[("advocate", "mean", "none")], "confirm")
            aud_w, _ = _seed_mean_wins(by[("auditor", pooling, "none")], "confirm")
            corr[f"advocate-mean vs auditor-{pooling}"] = {
                "pearson": c,
                "one_critic_counted_twice": c < -0.9,
                **error_overlap(adv_w, aud_w),
            }
    out["D_correlation"] = corr

    # Exploratory, no rule: the mid-layer diagnostic, seed-mean accuracy per set.
    mid = {}
    for role, pooling in EXPLORATORY_FORMS:
        rs = by.get((role, pooling, "none"))
        if not rs:
            continue
        mid[f"{role}-{pooling}"] = {
            name: sum(_seed_mean_wins(rs, name)[0]) / len(rs[0]["sets"][name]["strong"])
            for name in ("confirm", "judge", "second")
            if all(name in r["sets"] for r in rs)
        }
    out["X_exploratory_mid_layer"] = mid

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


def compare_sources(
    a: list[dict], b: list[dict], sets: tuple[str, ...] = ("confirm", "judge", "second")
) -> dict:
    """The same form trained on two feature sources: per set, the seed-mean accuracy of each and the
    paired, prompt-clustered difference (a − b). Reported with no rule attached (the addendum)."""

    def group(results):
        by = {}
        for r in results:
            if r["control"] == "none":
                by.setdefault((r["role"], r["pooling"]), []).append(r)
        return by

    ga, gb = group(a), group(b)
    out = {}
    for role, pooling in FORMS + EXPLORATORY_FORMS:
        key = (role, pooling)
        if key not in ga or key not in gb:
            continue
        form = {}
        for name in sets:
            if not all(name in r["sets"] for r in ga[key] + gb[key]):
                continue
            wa, ids = _seed_mean_wins(ga[key], name)
            wb, _ = _seed_mean_wins(gb[key], name)
            diff = paired_diff(wa, wb, ids)
            form[name] = {"a": sum(wa) / len(wa), "b": sum(wb) / len(wb), "a_minus_b": diff[0], "ci": diff[1]}
        out[f"{role}-{pooling}"] = form
    return out


def family_contrast(
    a: list[dict], b: list[dict], matched_size: bool, own: str = "judge", other: str = "second"
) -> dict:
    """The family pattern as one number per form: (a − b) on the `own`-family-planted set minus
    (a − b) on the `other`-planted set, from seed-mean per-pair wins, with a two-sample
    prompt-clustered interval. Positive means a's lead is larger on pairs its own family planted.

    Only a size-matched comparison (Qwen2.5-3B against Llama-3.2-3B) can be read as family; any
    other pair of sources mixes family with size, and the reading says so."""

    def group(results):
        by = {}
        for r in results:
            if r["control"] == "none":
                by.setdefault((r["role"], r["pooling"]), []).append(r)
        return by

    ga, gb = group(a), group(b)
    out = {}
    for role, pooling in FORMS + EXPLORATORY_FORMS:
        key = (role, pooling)
        if key not in ga or key not in gb:
            continue
        if not all(n in r["sets"] for r in ga[key] + gb[key] for n in (own, other)):
            continue
        wa_own, ids_own = _seed_mean_wins(ga[key], own)
        wb_own, _ = _seed_mean_wins(gb[key], own)
        wa_oth, ids_oth = _seed_mean_wins(ga[key], other)
        wb_oth, _ = _seed_mean_wins(gb[key], other)
        d_own = [x - y for x, y in zip(wa_own, wb_own)]
        d_oth = [x - y for x, y in zip(wa_oth, wb_oth)]
        point, ci = two_sample_diff(d_own, ids_own, d_oth, ids_oth)
        if ci[0] > 0:
            reading = "a leads more on its own family's plants" + (
                " (consistent with family recognition)" if matched_size else " (family and size mixed)"
            )
        elif ci[1] < 0:
            reading = "a leads less on its own family's plants"
        else:
            reading = "inconclusive"
        out[f"{role}-{pooling}"] = {"difference_of_differences": point, "ci": ci, "reading": reading}
    return out


def load_results(scores: Path) -> list[dict]:  # pragma: no cover - file glue
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(scores.glob("*.json"))]
