"""Critics with chosen attributes (docs/runs/2026-10-08-auditor-plan.md): the trainer, the readout,
and the second-planter set, on synthetic data (no GPU, no student model)."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest
import torch

EXPERIMENT = Path(__file__).resolve().parents[1] / "examples" / "sft-experiment"
sys.path.insert(0, str(EXPERIMENT))

import critic_heads as ch  # noqa: E402
import critic_heads_run as run  # noqa: E402
import lib  # noqa: E402
import second_planter as sp  # noqa: E402

FAST = dict(ch.HPARAMS, hidden_dim=16, epochs=30, batch_pairs=8, lr=3e-3)


def synthetic(n_pairs: int, signal: float, seed: int = 0, tokens: int = 6, dim: int = 8):
    """Answer features where a flawed answer carries `signal` on dimension 0 at one token."""
    g = torch.Generator().manual_seed(seed)
    feats, masks, index = [], [], []
    for i in range(n_pairs):
        for flawed in (False, True):
            x = torch.randn(tokens, dim, generator=g) * 0.1
            if flawed:
                x[2, 0] += signal
            feats.append(x)
            masks.append(torch.ones(tokens, dtype=torch.long))
        index.append((2 * i, 2 * i + 1))
    return feats, masks, index


def accuracy(role, strong, flawed):
    hi, lo = ch.role_scores(role, strong, flawed)
    w = ch.pair_wins(hi, lo)
    return sum(w) / len(w)


class TestEdits:
    def test_edit_spans_cover_the_changed_characters(self):
        s, f = ch.edit_spans("The value is 7 today.", "The value is 9 today.")
        assert s == (13, 14) and f == (13, 14)

    def test_marker_at_the_end_and_at_the_edit(self):
        pair = {"strong": "Water boils at 100 C.", "flawed": "Water boils at 90 C."}
        assert ch.with_marker(pair, "end")["flawed"].endswith(ch.END_MARKER)
        at_edit = ch.with_marker(pair, "edit")["flawed"]
        assert at_edit.startswith("Water boils at") and ch.EDIT_MARKER in at_edit
        assert at_edit.index(ch.EDIT_MARKER) == 15

    def test_span_tokens_pick_the_tokens_over_the_span(self):
        offsets = [(0, 5), (5, 11), (11, 14), (14, 16), (16, 21)]
        assert ch.span_tokens(offsets, 12, 15) == (2, 4)
        assert ch.span_tokens([(0, 0), (0, 3)], 10, 11) == (1, 2)  # nothing overlaps: the nearest token


class TestStatistics:
    def test_bootstrap_resamples_whole_prompts(self):
        values = [1.0] * 5 + [0.0]
        _, ci = ch.boot(values, [0, 0, 0, 0, 0, 1])
        assert ci[0] == 0.0 and ci[1] == 1.0

    def test_paired_and_two_sample_differences(self):
        a, b = [1.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]
        diff, ci = ch.paired_diff(a, b, [0, 1, 2, 3])
        assert diff == pytest.approx(0.5) and ci[0] <= 0.5 <= ci[1]
        d2, _ = ch.two_sample_diff([1.0, 1.0], ["a", "b"], [0.0, 1.0], ["c", "d"])
        assert d2 == pytest.approx(0.5)

    def test_error_overlap(self):
        same = ch.error_overlap([1, 0, 1, 0], [1, 0, 1, 0])
        assert same["error_consistency"] == pytest.approx(1.0) and same["double_fault"] == 0.5
        apart = ch.error_overlap([1, 0, 1, 0], [0, 1, 0, 1])
        assert apart["error_consistency"] == pytest.approx(-1.0) and apart["double_fault"] == 0.0

    def test_near_tie_share(self):
        assert ch.near_tie_share([5.0, 5.0, 5.0, 5.0], [5.001, 4.9, 6.0, 5.0005], 0.01) == 0.5

    def test_auc_and_pearson(self):
        assert ch.auc([3, 4], [1, 2]) == 1.0 and ch.auc([1, 2], [1, 2]) == 0.5
        assert ch.pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


class TestReadings:
    def test_role_check(self):
        assert ch.role_check([0.2, 0.45], 0.33) == "rejected"
        assert ch.role_check([0.4, 0.6], 0.48) == "flagged"
        assert ch.role_check([0.55, 0.8], 0.7) == "ok"

    def test_found_auditor_reading(self):
        assert ch.found_auditor_reading([0.55, 0.8]) == "auditor"
        assert ch.found_auditor_reading([0.45, 0.6]) == "noise draw, not an auditor"
        assert ch.found_auditor_reading([0.2, 0.4]).startswith("unstable")

    def test_positive_gate(self):
        assert ch.positive_gate(0.97, [0.92, 1.0]) and not ch.positive_gate(0.97, [0.88, 1.0])
        assert not ch.positive_gate(0.93, [0.91, 0.96])

    def test_transfer_reading(self):
        assert ch.transfer_reading([0.6, 0.9], [-0.1, 0.1]) == "transfers"
        assert ch.transfer_reading([0.4, 0.7], [0.05, 0.3]) == "learned the planter's edits"
        assert ch.transfer_reading([0.4, 0.9], [-0.1, 0.2]) == "inconclusive"


class TestTraining:
    @pytest.mark.parametrize("role", ["auditor", "advocate"])
    @pytest.mark.parametrize("pooling", ["mean", "attention"])
    def test_a_head_learns_a_planted_signal(self, role, pooling):
        feats, masks, index = synthetic(40, signal=3.0)
        head = ch.train_head(role, pooling, 42, feats, masks, index, hparams=FAST)
        vf, vm, vi = synthetic(20, signal=3.0, seed=1)
        strong, flawed = ch.score_set(head, vf, vm, vi)
        assert accuracy(role, strong, flawed) >= 0.9

    def test_shuffled_labels_stay_near_chance(self):
        feats, masks, index = synthetic(60, signal=3.0)
        rng = random.Random(7)
        flip = [rng.random() < 0.5 for _ in index]
        head = ch.train_head("advocate", "mean", 42, feats, masks, index, flip, hparams=FAST)
        vf, vm, vi = synthetic(40, signal=3.0, seed=2)
        strong, flawed = ch.score_set(head, vf, vm, vi)
        assert 0.2 <= accuracy("advocate", strong, flawed) <= 0.8

    def test_the_init_seed_alone_sets_the_head_and_leaves_the_global_stream(self):
        feats, masks, index = synthetic(4, signal=1.0)
        torch.manual_seed(0)
        before = torch.random.get_rng_state()
        a = ch.train_head("auditor", "mean", 42, feats, masks, index, hparams=dict(FAST, epochs=0))
        b = ch.train_head("auditor", "mean", 42, feats, masks, index, hparams=dict(FAST, epochs=0))
        c = ch.train_head("auditor", "mean", 43, feats, masks, index, hparams=dict(FAST, epochs=0))
        assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
        assert not all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), c.state_dict().values()))
        assert torch.equal(torch.random.get_rng_state(), before)


def fake_result(role, pooling, control, seed, sets):
    return {"role": role, "pooling": pooling, "seed": seed, "control": control, "sets": sets}


def scored(n, good: float, ids, role="advocate"):
    """Strong/flawed scores, deterministic: the role's preferred side wins on round(n * good) pairs,
    spread evenly, with no ties (so the readout's gates never sit near a sampling edge)."""
    strong, flawed = [], []
    for i in range(n):
        win = int((i + 1) * good) > int(i * good)
        hi, lo = (5.5 + i / 1000, 4.5) if win else (4.5, 5.5 + i / 1000)
        s, f = (lo, hi) if role == "auditor" else (hi, lo)
        strong.append(s)
        flawed.append(f)
    return {"prompt_ids": ids, "strong": strong, "flawed": flawed}


class TestReadout:
    def build(self, auditor_good=0.9, shuffled_good=0.5, marker_good=1.0):
        ids = [i // 2 for i in range(60)]
        jids = [f"j{i // 2}" for i in range(40)]
        sids = [f"s{i // 2}" for i in range(20)]
        results = []
        for role, pooling in ch.FORMS:
            good = auditor_good if role == "auditor" else 0.9
            for seed in ch.SEEDS:
                results.append(
                    fake_result(
                        role,
                        pooling,
                        "none",
                        seed,
                        {
                            "confirm": scored(60, good, ids, role),
                            "judge": scored(40, good, jids, role),
                            "second": scored(20, good, sids, role),
                        },
                    )
                )
                results.append(
                    fake_result(
                        role,
                        pooling,
                        "shuffled",
                        seed,
                        {"confirm": scored(60, shuffled_good, ids, role)},
                    )
                )
                results.append(
                    fake_result(
                        role, pooling, "marker", seed, {"confirm": scored(60, marker_good, ids, role)}
                    )
                )
                results.append(
                    fake_result(
                        role, pooling, "marker-at-edit", seed, {"confirm": scored(60, 0.7, ids, role)}
                    )
                )
        found = scored(60, 0.5, ids, "auditor")
        pairs = [{"prompt_id": i} for i in ids]
        return results, found, pairs

    def test_a_clean_run_reads_every_section(self):
        results, found, pairs = self.build()
        r = run.readout(results, found, pairs)
        assert r["A_controls"]["role_reading_allowed"]
        assert all(g["passes"] for g in r["A_controls"]["positive_gate"].values())
        assert r["B_found_auditor"]["reading"] == "noise draw, not an auditor"
        assert len(r["C_critics"]) == 12 and all("transfer" in h for h in r["C_critics"])
        assert r["claims"]["auditor-mean"]["trained_on_purpose"]
        assert set(r["D_correlation"]) == {
            "advocate-mean vs auditor-mean",
            "advocate-mean vs auditor-attention",
        }
        overlap = r["D_correlation"]["advocate-mean vs auditor-mean"]
        assert {"error_consistency", "double_fault"} <= set(overlap)
        assert r["E_panel"]["members"] and "judge" in r["E_panel"]
        assert all(not m.startswith("advocate-span") for m in r["E_panel"]["members"])

    def test_two_feature_sources_compare_form_by_form_on_the_same_pairs(self):
        a, _, _ = self.build(auditor_good=0.9)
        b, _, _ = self.build(auditor_good=0.6)
        cmp = run.compare_sources(a, b)
        assert set(cmp) == {"auditor-mean", "auditor-attention", "advocate-mean", "advocate-span"}
        aud = cmp["auditor-mean"]["confirm"]
        assert aud["a_minus_b"] == pytest.approx(0.3, abs=0.02) and aud["ci"][0] > 0
        assert cmp["advocate-mean"]["judge"]["a_minus_b"] == pytest.approx(0.0)
        assert set(cmp["auditor-mean"]) == {"confirm", "judge", "second"}

    def test_feature_sources_are_pinned_licensed_and_size_matched(self):
        assert ch.FEATURE_SOURCES["qwen"] == (ch.STUDENT, ch.STUDENT_REVISION)
        assert set(ch.FEATURE_SOURCES) == set(ch.SOURCE_LICENSES) == {"qwen", "qwen3b", "llama"}
        for model, revision in ch.FEATURE_SOURCES.values():
            assert len(revision) == 40
        assert ch.FEATURE_SOURCES["llama"][0].startswith("meta-llama/")
        assert "3B" in ch.FEATURE_SOURCES["qwen3b"][0] and "3B" in ch.FEATURE_SOURCES["llama"][0]

    def test_mid_layer_is_two_thirds_of_the_way_in(self):
        assert ch.mid_layer(28) == 19 and ch.mid_layer(36) == 24

    def test_family_contrast_is_one_number_with_its_reading(self):
        # a leads by 0.3 on the own-family set and by nothing on the other set.
        a, _, _ = self.build(auditor_good=0.9)
        b, _, _ = self.build(auditor_good=0.9)
        for r in b:
            if r["role"] == "auditor" and r["control"] == "none":
                r["sets"]["judge"] = scored(40, 0.6, r["sets"]["judge"]["prompt_ids"], "auditor")
        matched = run.family_contrast(a, b, matched_size=True)["auditor-mean"]
        assert matched["difference_of_differences"] == pytest.approx(0.3, abs=0.03)
        assert matched["ci"][0] > 0 and "family recognition" in matched["reading"]
        mixed = run.family_contrast(a, b, matched_size=False)["auditor-mean"]
        assert "family and size mixed" in mixed["reading"]
        assert run.family_contrast(a, a, matched_size=True)["advocate-mean"]["reading"] == "inconclusive"

    def test_the_mid_layer_diagnostic_is_reported_apart(self):
        results, found, pairs = self.build()
        for role in ("auditor", "advocate"):
            for seed in ch.SEEDS:
                sets = {
                    n: scored(k, 0.8, ids, role)
                    for n, k, ids in (
                        ("confirm", 60, [i // 2 for i in range(60)]),
                        ("judge", 40, [f"j{i // 2}" for i in range(40)]),
                    )
                }
                results.append(fake_result(role, "mid", "none", seed, sets))
        r = run.readout(results, found, pairs)
        assert set(r["X_exploratory_mid_layer"]) == {"auditor-mid", "advocate-mid"}
        assert r["X_exploratory_mid_layer"]["auditor-mid"]["confirm"] == pytest.approx(0.8)
        assert len(r["C_critics"]) == 12  # the diagnostic never enters the committed readout

    def test_a_leaking_shuffled_control_stops_every_role_reading(self):
        results, found, pairs = self.build(shuffled_good=0.95)
        r = run.readout(results, found, pairs)
        assert not r["A_controls"]["role_reading_allowed"]
        assert not any(h["readable"] for h in r["C_critics"])
        assert r["E_panel"] == {"members": 0}

    def test_a_failed_positive_gate_stops_that_form(self):
        results, found, pairs = self.build(marker_good=0.6)
        r = run.readout(results, found, pairs)
        assert not any(g["passes"] for g in r["A_controls"]["positive_gate"].values())
        assert not any(h["readable"] for h in r["C_critics"])


class TestSecondPlanter:
    KEPT = [(f"topic-{t}", f"question {t}-{i}") for t in range(4) for i in range(130)]

    def test_spare_prompts_are_unused_by_both_splits(self):
        spare = sp.spare_prompts(self.KEPT)
        train, held = lib.split_held_out(self.KEPT, 64, 400, 42)
        used = {p for _, p in train} | {p for _, p in held}
        assert spare and not {p for _, p in spare} & used
        assert len(spare) + len(used) == len(self.KEPT)

    def test_answers_cut_at_the_cap_are_dropped(self):
        backend = lib.FakeBackend(lambda chat: "short" if "0" in chat.turns[0][1] else "x" * 50)
        items = sp.write_answers(backend, [("t", "q0"), ("t", "q1")], answer_tokens=20)
        assert [it["prompt"] for it in items] == ["q0"]

    def test_planting_and_edit_stats(self):
        def answer(chat):
            text = chat.turns[0][1]
            value = text.split("The value is ")[1].split(".")[0]
            return json.dumps(
                {"original": f"The value is {value}.", "edited": f"The value is {int(value) + 1}."}
            )

        items = [
            {
                "prompt_id": f"s{i}",
                "topic": "t",
                "prompt": f"Q{i}?",
                "strong": f"Intro {i}. The value is {i}. End {i}.",
            }
            for i in range(3)
        ]
        pairs, report = sp.plant(lib.FakeBackend(answer), "gemma4:31b", items)
        assert report["planted"] == len(pairs) > 0 and report["slots"] == 6
        stats = sp.edit_stats(pairs)
        assert stats["median_chars_changed"] == 1 and 0 < stats["median_position"] < 1

    def test_cloud_models_are_refused(self):
        assert lib.refuse_cloud("gemma4:31b") == "gemma4:31b"
        for name in ("gemma4:31b-cloud", "gemma4:cloud", "x:Cloud"):
            with pytest.raises(ValueError):
                lib.refuse_cloud(name)
