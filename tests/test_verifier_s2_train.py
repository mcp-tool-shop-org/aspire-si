"""The S2 training run: resume across chunks (B1), the truncation halt (B2), seeding (B3), the critic arm
(B4), the guard history (B5) and the training verdict (B6).

Every test runs on the CPU with the dry rehearsal's tiny random model (or none at all). Torch is required: the
repo's conftest imports it.
"""

import gzip
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "verifier-distill"))

import s2_grpo_smoke as s2  # noqa: E402

torch = pytest.importorskip("torch")

TIMING_KEYS = {"rollout_s", "train_s", "step_s", "peak_alloc_mib"}


# -- fixtures --
def _row(cid, split="tune", label="supported", claim="The parser counts tokens.", text=None):
    text = text or "def count(tokens):\n    return len(tokens)\n# the parser counts tokens here"
    return {
        "id": cid,
        "check_type": "grounded",
        "claim": claim,
        "label": label,
        "split": split,
        "context": [{"source": f"src@{cid}", "text": text}],
    }


def _gold(tmp_path):
    files = {
        "grounded.jsonl": [_row(f"g{i:02d}") for i in range(20)],
        "prs/grounded-prs.jsonl": [_row(f"p{i:02d}") for i in range(20)],
        "diffs/reasoning-diffs.jsonl": [_row(f"r{i:02d}", label="unsupported") for i in range(20)],
    }
    gold = tmp_path / "gold"
    for name, rows in files.items():
        path = gold / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return gold


def _argv(gold, out, *extra, arm="mech", seed=42, steps=4, gpu=2, window=10, halt=0.9, train=True):
    # the dry model truncates nearly every completion at --max-new 8, so the default window (10 updates) is
    # never full in these short runs; the truncation tests set their own window
    argv = [
        "--gold",
        str(gold),
        "--rnd-commit",
        "e48cc90c",
        "--out",
        str(out),
        "--dry",
        "--steps",
        str(steps),
        "--g",
        "2",
        "--max-new",
        "8",
        "--groups-per-update",
        str(gpu),
    ]
    if train:
        argv += [
            "--arm",
            arm,
            "--train-seed",
            str(seed),
            "--trunc-window-updates",
            str(window),
            "--trunc-halt-rate",
            str(halt),
        ]
    return [*argv, *extra]


def _ns(argv):
    return s2.apply_dry_defaults(s2.build_parser().parse_args(argv))


def _run(argv, build=s2.dry_bundle):
    return s2.execute(_ns(argv), build=build)


def _receipt(out):
    return json.loads((Path(out) / "s2-hf.json").read_text(encoding="utf-8"))


def _rows(out):
    path = Path(out) / "s2-hf-steps.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _untimed(rows):
    return [{k: v for k, v in r.items() if k not in TIMING_KEYS} for r in rows]


def _state(out):
    sdir = s2.latest_state_dir(Path(out))
    return (
        json.loads((sdir / "state.json").read_text(encoding="utf-8")),
        torch.load(sdir / "trainable.pt", weights_only=True),
        torch.load(sdir / "optimizer.pt", weights_only=True),
    )


def _assert_tensors_equal(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(a[k], b[k]), k


def _assert_optim_equal(a, b):
    assert a["param_groups"] == b["param_groups"]
    assert sorted(a["state"]) == sorted(b["state"])
    for idx, st in a["state"].items():
        other = b["state"][idx]
        assert sorted(st) == sorted(other)
        for key, val in st.items():
            if torch.is_tensor(val):
                assert torch.equal(val, other[key]), (idx, key)
            else:
                assert val == other[key], (idx, key)


def _history(out):
    path = Path(out) / "s2-hf-guard-history.jsonl.gz"
    return s2.read_save_all(path) if path.exists() else []


# -- B1: resume across chunks --
class TestResumeEquivalence:
    def test_k_updates_in_one_go_equal_k1_plus_k2_bit_for_bit(self, tmp_path):
        """The required test: 4 updates in one go == 2 + 2 split by stop/resume, on the CPU dry model."""
        gold = _gold(tmp_path)
        full, c1, c2 = tmp_path / "full", tmp_path / "chunk-1", tmp_path / "chunk-2"
        assert _run(_argv(gold, full)) == 0
        assert _run(_argv(gold, c1, "--stop-after-updates", "2")) == 0
        assert _run(_argv(gold, c2, "--resume-from", str(c1))) == 0

        assert _receipt(full)["verdict"] == "S2-TRAIN-COMPLETE"
        assert _receipt(c1)["verdict"] == "S2-TRAIN-CHUNK-OK"
        assert _receipt(c2)["verdict"] == "S2-TRAIN-COMPLETE"
        assert _receipt(c1)["stopped"] is None and _receipt(c1)["training"]["chunk_stop"]["next_step"] == 2
        assert _receipt(c2)["training"]["start_step"] == 2 and _receipt(c2)["training"]["chunk_index"] == 2

        st_full, ten_full, opt_full = _state(full)
        st_c2, ten_c2, opt_c2 = _state(c2)
        _, ten_c1, _ = _state(c1)
        _assert_tensors_equal(ten_full, ten_c2)
        _assert_optim_equal(opt_full, opt_c2)
        assert any(not torch.equal(ten_c1[k], ten_c2[k]) for k in ten_c1)  # training moved the adapter
        assert st_full["next_step"] == st_c2["next_step"] == 4
        assert st_full["trunc_guard"] == st_c2["trunc_guard"]
        assert st_full["rng_plain"] == st_c2["rng_plain"]

        # the per-update rows, rewards, advantages and texts
        assert _untimed(_rows(full)) == _untimed(_rows(c1) + _rows(c2))
        sa = s2.read_save_all
        full_sa = sa(full / "s2-hf-save-all.jsonl.gz")
        split_sa = sa(c1 / "s2-hf-save-all.jsonl.gz") + sa(c2 / "s2-hf-save-all.jsonl.gz")
        assert full_sa == split_sa and len(full_sa) == 4 * 2 * 2
        assert _history(full) == _history(c1) + _history(c2)

        # the adapter written for evaluation is the same too
        assert _receipt(full)["adapter_sha256"] == _receipt(c2)["adapter_sha256"]
        chain = st_c2["save_all_chain"]
        assert [c["chunk_dir"] for c in chain] == ["chunk-1", "chunk-2"]
        assert [c["next_step"] for c in chain] == [2, 4]
        sa_mark = chain[0]["archives"]["save_all"]
        assert sa_mark["bytes"] == (c1 / "s2-hf-save-all.jsonl.gz").stat().st_size

    def test_resume_refuses_a_different_train_seed_and_names_it(self, tmp_path):
        gold = _gold(tmp_path)
        c1 = tmp_path / "c1"
        assert _run(_argv(gold, c1, "--stop-after-updates", "1", steps=2)) == 0
        with pytest.raises(SystemExit, match=r"RESUME_HEADER_MISMATCH.*train_seed"):
            _run(_argv(gold, tmp_path / "c2", "--resume-from", str(c1), seed=43, steps=2))
        with pytest.raises(SystemExit, match=r"RESUME_HEADER_MISMATCH.*settings\.quote_reward"):
            _run(_argv(gold, tmp_path / "c3", "--resume-from", str(c1), "--quote-reward", "v2", steps=2))
        with pytest.raises(SystemExit, match=r"RESUME_HEADER_MISMATCH.*settings\.steps"):
            _run(_argv(gold, tmp_path / "c4", "--resume-from", str(c1), steps=3))

    def test_resume_refuses_another_commit(self, tmp_path, monkeypatch):
        gold = _gold(tmp_path)
        c1 = tmp_path / "c1"
        assert _run(_argv(gold, c1, "--stop-after-updates", "1", steps=2)) == 0
        monkeypatch.setattr(s2, "aspire_commit", lambda cwd=None: "f" * 40)
        with pytest.raises(SystemExit, match=r"RESUME_HEADER_MISMATCH.*aspire_commit"):
            _run(_argv(gold, tmp_path / "c2", "--resume-from", str(c1), steps=2))

    def test_resume_refuses_changed_state_files_and_a_changed_save_all(self, tmp_path):
        gold = _gold(tmp_path)
        c1 = tmp_path / "c1"
        assert _run(_argv(gold, c1, "--stop-after-updates", "1", steps=2)) == 0
        archive = c1 / "s2-hf-save-all.jsonl.gz"
        original = archive.read_bytes()
        archive.write_bytes(original + gzip.compress(b"", mtime=0))
        with pytest.raises(SystemExit, match="RESUME_SAVE_ALL_CHANGED"):
            _run(_argv(gold, tmp_path / "c2", "--resume-from", str(c1), steps=2))
        archive.write_bytes(original)
        opt = s2.latest_state_dir(c1) / "optimizer.pt"
        opt.write_bytes(opt.read_bytes() + b"x")
        with pytest.raises(SystemExit, match="RESUME_STATE_FILE_CHANGED"):
            _run(_argv(gold, tmp_path / "c3", "--resume-from", str(c1), steps=2))

    def test_resume_refuses_a_finished_run_the_same_dir_and_a_missing_state(self, tmp_path):
        gold = _gold(tmp_path)
        done = tmp_path / "done"
        assert _run(_argv(gold, done, steps=1)) == 0
        with pytest.raises(SystemExit, match="RESUME_RUN_COMPLETE"):
            _run(_argv(gold, tmp_path / "next", "--resume-from", str(done), steps=1))
        with pytest.raises(SystemExit, match="RESUME_SAME_DIR"):
            _run(_argv(gold, done, "--resume-from", str(done), steps=1))
        with pytest.raises(SystemExit, match="RESUME_STATE_MISSING"):
            _run(_argv(gold, tmp_path / "x", "--resume-from", str(tmp_path / "nowhere"), steps=1))
        with pytest.raises(SystemExit, match="OUT_DIR_HAS_TRAIN_STATE"):
            _run(_argv(gold, done, steps=1))

    def test_resume_refuses_a_checkpoint_beside_it(self, tmp_path):
        gold = _gold(tmp_path)
        c1 = tmp_path / "c1"
        assert _run(_argv(gold, c1, "--stop-after-updates", "1", steps=2)) == 0
        a = _ns(_argv(gold, tmp_path / "c2", "--resume-from", str(c1), steps=2))
        a.checkpoint = c1 / "adapter"
        with pytest.raises(SystemExit, match="resume_with_checkpoint"):
            s2.validate_training_args(a)

    def test_resume_refuses_a_halted_run(self, tmp_path):
        gold = _gold(tmp_path)
        halted = tmp_path / "halted"
        _run(_argv(gold, halted, steps=2, window=1, halt=0.01))
        assert _receipt(halted)["verdict"] == "S2-TRAIN-HALT-TRUNC"
        with pytest.raises(SystemExit, match="RESUME_FROM_HALTED"):
            _run(_argv(gold, tmp_path / "after", "--resume-from", str(halted), steps=2, window=1, halt=0.01))

    def test_resume_refuses_mismatched_parameters(self, tmp_path):
        a = _ns(_argv(_gold(tmp_path), tmp_path / "o"))
        bundle = s2.dry_bundle(a)
        live = {n: p.detach().clone() for n, p in bundle["model"].named_parameters() if p.requires_grad}
        name = sorted(live)[0]
        with pytest.raises(SystemExit, match="RESUME_PARAMS_MISMATCH"):
            s2.apply_train_state(
                {"_trainable": {name: live[name]}}, model=bundle["model"], optimiser=bundle["optimiser"]
            )
        bad = dict(live)
        bad[name] = live[name].to(torch.float64)
        with pytest.raises(SystemExit, match="shape or dtype"):
            s2.apply_train_state({"_trainable": bad}, model=bundle["model"], optimiser=bundle["optimiser"])

    def test_header_differences_compares_settings_key_by_key(self):
        base = {"arm": "mech", "settings": {"steps": 197, "g": 8}, "pip_freeze_sha256": "a"}
        assert s2.header_differences(base, dict(base)) == []
        other = {"arm": "critic", "settings": {"steps": 197, "g": 2, "x": 1}, "pip_freeze_sha256": "b"}
        assert s2.header_differences(base, other) == ["arm", "pip_freeze_sha256", "settings.g", "settings.x"]

    def test_header_carries_seeds_shas_commit_revision_and_freeze(self, tmp_path):
        gold = _gold(tmp_path)
        out = tmp_path / "o"
        assert _run(_argv(gold, out, steps=1)) == 0
        hdr = json.loads((out / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        for key in (
            "arm",
            "train_seed",
            "draw_seed",
            "holdout_ids_sha256",
            "drawn_ids_sha256",
            "prompt_ids_sha256",
            "aspire_commit",
            "script_sha256",
            "revision_requested",
            "revision_resolved",
            "pip_freeze_sha256",
            "stack_versions",
            "start_policy",
            "settings",
        ):
            assert key in hdr, key
        assert hdr["start_policy"] == "base_fresh_lora" and hdr["start_checkpoint_sha256"] is None
        assert hdr["settings"]["trunc_window_updates"] == 10 and hdr["settings"]["arm"] == "mech"
        assert "stop_after_updates" not in hdr["settings"] and "resume_from" not in hdr["settings"]
        assert hdr["scheduler"] is None and hdr["critic"] is None


class TestTrainingArgs:
    @pytest.mark.parametrize(
        "drop", [("--arm",), ("--train-seed",), ("--trunc-window-updates",), ("--trunc-halt-rate",)]
    )
    def test_every_required_training_flag_is_refused_when_absent(self, tmp_path, drop):
        argv = _argv(_gold(tmp_path), tmp_path / "o")
        i = argv.index(drop[0])
        del argv[i : i + 2]
        with pytest.raises(SystemExit, match=f"training_flags_required.*{drop[0]}"):
            s2.validate_training_args(_ns(argv))

    def test_a_smoke_run_without_training_flags_is_not_a_training_run(self, tmp_path):
        assert s2.validate_training_args(_ns(_argv(_gold(tmp_path), tmp_path / "o", train=False))) is False

    @pytest.mark.parametrize(
        ("extra", "match"),
        [
            (("--rollout-only",), "training_flags_in_other_mode"),
            (("--probe-memory",), "training_flags_in_other_mode"),
            (("--paired-estimators", "spec.json"), "training_flags_in_other_mode"),
            (("--stress-prefix", "1"), "training_stress_prefix"),
            (("--stop-after-updates", "0"), "invalid_stop_after_updates"),
        ],
    )
    def test_training_refuses_other_modes_and_bad_values(self, tmp_path, extra, match):
        with pytest.raises(SystemExit, match=match):
            s2.validate_training_args(_ns(_argv(_gold(tmp_path), tmp_path / "o", *extra)))

    def test_training_refuses_the_llamacpp_path_a_bad_seed_and_bad_trunc_values(self, tmp_path):
        a = _ns(_argv(_gold(tmp_path), tmp_path / "o"))
        a.path = "llamacpp"
        with pytest.raises(SystemExit, match="training_path_hf_only"):
            s2.validate_training_args(a)
        a.path, a.train_seed = "hf", -1
        with pytest.raises(SystemExit, match="invalid_train_seed"):
            s2.validate_training_args(a)
        a.train_seed, a.arm = 1, "nli"
        with pytest.raises(SystemExit, match="unknown_arm"):
            s2.validate_training_args(a)
        a.arm, a.trunc_halt_rate = "mech", 1.5
        with pytest.raises(SystemExit, match="trunc-halt-rate"):
            s2.validate_training_args(a)
        a.trunc_halt_rate, a.trunc_window_updates = 0.2, 0
        with pytest.raises(SystemExit, match="trunc-window-updates"):
            s2.validate_training_args(a)

    def test_flags_are_recorded_in_active_flags(self, tmp_path):
        a = _ns(_argv(_gold(tmp_path), tmp_path / "o", "--stop-after-updates", "3"))
        a.resume_from = tmp_path / "chunk-1"
        flags = s2.active_flags(a)
        assert flags["arm"] == "mech" and flags["train_seed"] == 42
        assert flags["trunc_window_updates"] == 10 and flags["trunc_halt_rate"] == 0.9
        assert flags["stop_after_updates"] == 3 and flags["resume_from"] == "chunk-1"

    def test_stop_after_updates_below_one_is_refused_by_run_steps(self):
        with pytest.raises(SystemExit, match="stop-after-updates"):
            s2.run_steps(
                model=None,
                ref_adapter_off=None,
                tok=s2.DryTokenizer(),
                optimiser=None,
                prompts=[_row("a")],
                rollout=None,
                steps=1,
                g=2,
                max_new=4,
                reward_fn=None,
                device_sync=lambda: None,
                peak_alloc_mib=lambda: None,
                stop_after_updates=0,
            )

    def test_aspire_commit_reads_git_and_returns_none_outside_a_checkout(self, tmp_path, monkeypatch):
        sha = s2.aspire_commit()
        assert sha is None or len(sha) == 40
        assert s2.aspire_commit(tmp_path / "missing-dir") is None

        class _Out:
            stdout = "not-a-sha\n"

        monkeypatch.setattr(s2.subprocess, "run", lambda *a, **k: _Out())
        assert s2.aspire_commit() is None


# -- B2: the truncation halt --
class TestTruncationGuard:
    def test_needs_a_full_window_and_a_rate_above_the_line(self):
        g = s2.TruncationHaltGuard(window_updates=3, halt_rate=0.2)
        assert g.record_update(step=0, completions=10, truncated=10) is None  # window not full yet
        assert g.record_update(step=1, completions=10, truncated=0) is None
        stop = g.record_update(step=2, completions=10, truncated=0)  # full now: 10/30 > 0.2
        assert stop is not None and stop["step"] == 2
        assert g.history[2]["window_rate"] == round(10 / 30, 6) and g.history[2]["halt"] is True
        assert [h["window_full"] for h in g.history] == [False, False, True]

    def test_fires_above_and_not_at_or_below_the_line(self):
        g = s2.TruncationHaltGuard(window_updates=2, halt_rate=0.25)
        assert g.record_update(step=0, completions=4, truncated=1) is None
        assert g.record_update(step=1, completions=4, truncated=1) is None  # exactly 0.25: not above
        stop = g.record_update(step=2, completions=4, truncated=2)  # (1+2)/8 = 0.375
        assert stop["reason"] == "trunc_halt" and stop["triggered_rules"] == ["TRUNC"]
        assert stop["window_truncated"] == 3 and stop["window_completions"] == 8 and stop["step"] == 2

    def test_the_window_rolls(self):
        g = s2.TruncationHaltGuard(window_updates=2, halt_rate=0.5)
        g.record_update(step=0, completions=2, truncated=2)
        g.record_update(step=1, completions=2, truncated=0)
        assert g.window == [(0, 2, 2), (1, 2, 0)]
        g.record_update(step=2, completions=2, truncated=0)
        assert g.window == [(1, 2, 0), (2, 2, 0)]

    def test_state_round_trip(self):
        g = s2.TruncationHaltGuard(window_updates=2, halt_rate=0.5)
        g.record_update(step=0, completions=2, truncated=1)
        g2 = s2.TruncationHaltGuard.from_state(json.loads(json.dumps(g.to_state())))
        assert g2.to_state() == g.to_state()
        assert g.record_update(step=1, completions=2, truncated=2) == g2.record_update(
            step=1, completions=2, truncated=2
        )

    @pytest.mark.parametrize(("window", "rate"), [(0, 0.2), (True, 0.2), (2, 0.0), (2, 1.0)])
    def test_bad_settings_are_refused(self, window, rate):
        with pytest.raises(SystemExit):
            s2.TruncationHaltGuard(window_updates=window, halt_rate=rate)


def _fake_rollout_build(texts_for, truncs_for):
    """A dry bundle whose rollout returns fixed replies: texts_for(call) / truncs_for(call) per group call."""
    calls = {"n": 0}

    def build(a):
        bundle = s2.dry_bundle(a)
        tok = bundle["tok"]

        def rollout(prompt_text, g, max_new, temp):
            i = calls["n"]
            calls["n"] += 1
            texts = texts_for(i, prompt_text)
            return tok.encode(prompt_text)[:16], [tok.encode(t) for t in texts], list(truncs_for(i))

        bundle["rollout"] = rollout
        return bundle

    return build, calls


def _reply(verdict, quote="return len(tokens)"):
    return "<think>t</think>" + json.dumps(
        {"verdict": verdict, "evidence_quotes": [{"part": "p", "verdict": verdict, "quote": quote}]}
    )


class TestTruncationHaltInTheTrainer:
    def test_the_halt_fires_at_the_update_boundary_and_saves_the_checkpoint(self, tmp_path):
        gold = _gold(tmp_path)
        # update 0: 1 of 4 truncated; update 1: 2 of 4; window 2 = 3/8 > line 0.30 -> fires after update 1
        pattern = [[False, False], [True, False], [True, False], [True, False]]
        build, _ = _fake_rollout_build(
            lambda i, p: [_reply("supported"), _reply("unsupported")], lambda i: pattern[i]
        )
        out = tmp_path / "o"
        rc = _run(_argv(gold, out, steps=4, window=2, halt=0.30), build=build)
        rec = _receipt(out)
        assert rc == 2 and rec["verdict"] == "S2-TRAIN-HALT-TRUNC"
        assert rec["stopped"]["reason"] == "trunc_halt" and rec["stopped"]["step"] == 1
        assert rec["stopped"]["window_truncated"] == 3 and rec["stopped"]["window_completions"] == 8
        assert len(_rows(out)) == 2  # the halted update's row is recorded; its gradients are discarded
        assert (out / "adapter").is_dir() and rec["adapter_sha256"]
        state, ten_halt, _ = _state(out)
        assert state["halted"]["reason"] == "trunc_halt" and state["next_step"] == 1
        # the halted update was not applied: the adapter equals the state after update 0
        c1 = tmp_path / "after-update-0"
        build2, _ = _fake_rollout_build(
            lambda i, p: [_reply("supported"), _reply("unsupported")], lambda i: pattern[i]
        )
        assert (
            _run(_argv(gold, c1, "--stop-after-updates", "1", steps=4, window=2, halt=0.30), build=build2)
            == 0
        )
        _, ten_one, _ = _state(c1)
        _assert_tensors_equal(ten_halt, ten_one)
        hist = [h for h in _history(out) if h["kind"] == "trunc_window"]
        assert [h["halt"] for h in hist] == [False, True]

    def test_the_halt_does_not_fire_below_the_line(self, tmp_path):
        gold = _gold(tmp_path)
        build, _ = _fake_rollout_build(  # 50% truncated in every group
            lambda i, p: [_reply("supported"), _reply("unsupported")], lambda i: [True, False]
        )
        out = tmp_path / "o"
        assert _run(_argv(gold, out, steps=4, window=2, halt=0.6), build=build) == 0
        rec = _receipt(out)
        assert rec["verdict"] == "S2-TRAIN-COMPLETE" and rec["stopped"] is None
        assert [r["truncated_n"] for r in _rows(out)] == [2, 2, 2, 2]
        assert all(r["completions_n"] == 4 for r in _rows(out))

    def test_a_ct_guard_halt_and_a_truncation_halt_in_one_update_are_both_reported(self):
        ct = {"step": 3, "reason": "ct_guard_s1", "triggered_rules": ["S1"]}
        assert (
            s2.train_verdict_line(
                stopped={**ct, "also_triggered": {"reason": "trunc_halt"}},
                oom=False,
                next_step=3,
                steps_total=10,
                chunk_stop=None,
                fits=True,
                dry=True,
            )
            == "S2-TRAIN-HALT-S1"
        )


# -- B3: seeding --
class TestSeeding:
    def _lora(self, tmp_path, seed, name):
        a = _ns(_argv(_gold(tmp_path), tmp_path / name, seed=seed))
        model = s2.dry_bundle(a)["model"]
        return {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def test_the_same_seed_gives_a_byte_identical_adapter_init_and_another_differs(self, tmp_path):
        a42, b42, a43 = (self._lora(tmp_path, s, f"o{i}") for i, s in enumerate((42, 42, 43)))
        _assert_tensors_equal(a42, b42)
        lora_a = [k for k in a42 if "lora_A" in k]
        assert lora_a and all(not torch.equal(a42[k], a43[k]) for k in lora_a)

    def test_the_same_seed_gives_byte_identical_rollouts_and_another_differs(self, tmp_path):
        gold = _gold(tmp_path)
        outs = {}
        for name, seed in (("s42a", 42), ("s42b", 42), ("s43", 43)):
            assert _run(_argv(gold, tmp_path / name, seed=seed, steps=2)) == 0
            outs[name] = s2.read_save_all(tmp_path / name / "s2-hf-save-all.jsonl.gz")
        assert outs["s42a"] == outs["s42b"]
        assert [r["text"] for r in outs["s42a"]] != [r["text"] for r in outs["s43"]]
        for r in outs["s42a"]:
            assert r["seed"] == s2.train_rollout_seed(42, r["step"], r["group"]) and r["arm"] == "mech"

    def test_an_unseeded_smoke_run_records_no_seed(self, tmp_path):
        out = tmp_path / "smoke"
        assert _run(_argv(_gold(tmp_path), out, steps=1, train=False)) == 0
        recs = s2.read_save_all(out / "s2-hf-save-all.jsonl.gz")
        assert recs and all(r["seed"] is None and "arm" not in r for r in recs)
        assert not (out / s2.TRAIN_STATE_DIR).exists() and not (out / "s2-hf-train-header.json").exists()
        assert "training" not in _receipt(out)

    def test_rollout_seeds_depend_on_seed_step_group_and_purpose(self):
        base = s2.train_rollout_seed(42, 3, 1)
        assert base == s2.train_rollout_seed(42, 3, 1) and 0 <= base < 2**63
        others = {
            s2.train_rollout_seed(43, 3, 1),
            s2.train_rollout_seed(42, 4, 1),
            s2.train_rollout_seed(42, 3, 2),
            s2.train_rollout_seed(42, 3, 1, purpose="critic-retry"),
        }
        assert base not in others and len(others) == 4

    def test_rng_capture_and_restore_round_trip(self):
        import random

        import numpy as np

        s2.seed_everything(7)
        tensors, plain = s2.capture_rng_state()
        draws = (torch.rand(3), random.random(), np.random.rand())
        s2.restore_rng_state(tensors, json.loads(json.dumps(plain)))
        again = (torch.rand(3), random.random(), np.random.rand())
        assert torch.equal(draws[0], again[0]) and draws[1:] == again[1:]

    def test_rng_restore_refuses_cuda_states_without_a_cuda_device(self):
        tensors, plain = s2.capture_rng_state()
        with pytest.raises(SystemExit, match="RESUME_RNG_DEVICE_MISMATCH"):
            s2.restore_rng_state({**tensors, "torch_cuda": [torch.zeros(4, dtype=torch.uint8)]}, plain)


# -- B4: the critic arm --
class TestCriticHint:
    PASSAGES = ["def count(tokens):\n    return len(tokens)\n# the parser counts tokens here\nx = 1"]

    def test_names_the_first_unfound_quote_by_part(self):
        bad = _reply("supported", quote="this line is nowhere in the code at all")
        out = s2.critic_hint("The parser counts tokens.", self.PASSAGES, [("garbage", False), (bad, False)])
        assert out["quote_check"] == "quote_not_found"
        assert "quote given for part 1" in out["hint"] and '("p")' in out["hint"]
        assert "# the parser counts tokens here" in out["hint"] and out["retrieved_lines"] == 2

    def test_reports_found_quotes_and_unparseable_groups(self):
        good = _reply("unsupported", quote="return len(tokens)")
        found = s2.critic_hint("The parser counts tokens.", self.PASSAGES, [(good, False)])
        assert found["quote_check"] == "quotes_found"
        none = s2.critic_hint("The parser counts tokens.", self.PASSAGES, [(good, True), ("x", False)])
        assert none["quote_check"] == "no_parseable_answer"
        no_lines = s2.critic_hint("Zebra.", self.PASSAGES, [("x", False)])
        assert no_lines["retrieved_lines"] == 0 and "Evidence lines" not in no_lines["hint"]

    def test_the_hint_never_names_a_verdict(self):
        for decoded in ([("x", False)], [(_reply("supported", quote="zzz zzz zzz zzz"), False)]):
            hint = s2.critic_hint("The parser counts tokens.", self.PASSAGES, decoded)["hint"]
            for word in ("supported", "unsupported", "cannot_tell", "cannot tell"):
                assert word not in hint.lower()

    def test_the_retry_prompt_is_the_task_prompt_plus_the_hint(self):
        row = _row("a")
        assert s2.critic_retry_prompt(row, "HINT") == s2.build_prompt(row) + "\nHINT\n"


class TestCriticArm:
    def test_retries_are_archived_and_never_change_the_training_path(self, tmp_path):
        """Real dry rollouts (all malformed, so every group is all wrong and gets a retry): the critic arm's
        adapter, optimiser state, training save-all and step rows equal the mechanical arm's bit for bit."""
        gold = _gold(tmp_path)
        mech, critic = tmp_path / "mech", tmp_path / "critic"
        assert _run(_argv(gold, mech, arm="mech", steps=2)) == 0
        assert _run(_argv(gold, critic, arm="critic", steps=2)) == 0
        _, ten_m, opt_m = _state(mech)
        _, ten_c, opt_c = _state(critic)
        _assert_tensors_equal(ten_m, ten_c)
        _assert_optim_equal(opt_m, opt_c)
        sa_m = s2.read_save_all(mech / "s2-hf-save-all.jsonl.gz")
        sa_c = s2.read_save_all(critic / "s2-hf-save-all.jsonl.gz")
        assert [{**r, "arm": None} for r in sa_m] == [{**r, "arm": None} for r in sa_c]
        crit_keys = {"critic_hinted_groups", "critic_retry_samples", "critic_retry_correct"}
        rows_c = _untimed(_rows(critic))
        assert _untimed(_rows(mech)) == [{k: v for k, v in r.items() if k not in crit_keys} for r in rows_c]
        retries = s2.read_save_all(critic / "s2-hf-critic-retries.jsonl.gz")
        assert len(retries) == 2 * 2 * 2 and all(r["kind"] == "critic_retry" for r in retries)
        assert all(
            r["in_gradient"] is False and r["hint"].startswith("Hint from a tool check") for r in retries
        )
        assert {r["seed"] for r in retries} == {
            s2.train_rollout_seed(42, st, gr, purpose="critic-retry") for st in (0, 1) for gr in (0, 1)
        }
        rec = _receipt(critic)
        assert (
            rec["training"]["critic"]["hinted_groups"] == 4 and rec["training"]["critic"]["signal_drops"] == 0
        )
        assert rec["training"]["critic"]["retries_sha256"]
        assert not (mech / "s2-hf-critic-retries.jsonl.gz").exists() and rec["run"]["arm"] == "critic"
        assert all(r["critic_hinted_groups"] == 2 for r in rows_c)

    def test_a_group_with_a_correct_answer_gets_no_retry(self, tmp_path):
        gold = _gold(tmp_path)
        build, calls = _fake_rollout_build(
            lambda i, p: [_reply("supported"), _reply("unsupported")], lambda i: [False, False]
        )
        out = tmp_path / "o"
        assert _run(_argv(gold, out, arm="critic", steps=1), build=build) == 0
        assert calls["n"] == 2 and not (out / "s2-hf-critic-retries.jsonl.gz").exists()

    def test_a_scored_retry_records_correct_answers(self, tmp_path):
        gold = _gold(tmp_path)

        def texts(i, prompt):
            if "Hint from a tool check" in prompt:
                return [_reply("supported"), _reply("unsupported")]
            return [_reply("unsupported"), _reply("cannot_tell")]

        build, calls = _fake_rollout_build(texts, lambda i: [False, False])
        out = tmp_path / "o"
        assert _run(_argv(gold, out, arm="critic", steps=1, gpu=1), build=build) == 0
        retries = s2.read_save_all(out / "s2-hf-critic-retries.jsonl.gz")
        row_label = {r["id"]: r["label"] for r in s2.read_gold(gold)[0]}
        first = retries[0]
        assert calls["n"] == 2 and len(retries) == 2
        if row_label[first["item_id"]] == "supported":
            assert [r["correct"] for r in retries] == [True, False]
        assert _receipt(out)["training"]["critic"]["retry_samples"] == 2


MECH_GOLDEN_REPLIES = [
    [_reply("supported"), _reply("unsupported"), _reply("supported", quote="not in the code"), "garbage"],
    [_reply("unsupported"), _reply("cannot_tell"), _reply("supported"), _reply("supported")],
]


def _fix_weights(model):
    """Overwrite every weight with a fixed pattern, lora_B included, so the policy starts away from the
    reference (the KL term moves the weights from update 0), and a golden depends on torch math, not on the
    random init of whichever transformers version is installed."""
    with torch.no_grad():
        for k, (name, p) in enumerate(sorted(model.named_parameters(), key=lambda x: x[0])):
            idx = torch.arange(p.numel(), dtype=torch.float64)
            scale = (
                1.0 if "lora_B" in name else 0.05
            )  # a large adapter: the policy sits far from the reference
            p.copy_((torch.sin(idx * 0.37 + k) * scale).reshape(p.shape).to(p.dtype))


class TestMechArmMatchesTheBase:
    """Rewards, advantages, LoRA weights and AdamW state on fixed replies with the registered reward (v2,
    dr-grpo, masks on). The golden values were produced by aspire-si 7f9b4cb3's run_steps on this exact
    scenario, with the weights fixed by ``_fix_weights``. Weight summaries compare at rtol 1e-5 (float32 sums
    differ in the last bits between BLAS builds); rewards and advantages compare exactly."""

    GOLDEN = {
        "rewards": [
            [1.05, -1.0, -1.0, -1.0],
            [1.0, -0.3, -1.0, -3.0],
            [1.05, -1.0, -1.0, -1.0],
            [1.0, -0.3, -1.0, -3.0],
            [1.05, -1.0, -1.0, -1.0],
            [1.0, -0.3, -1.0, -3.0],
            [1.05, -1.0, -1.0, -1.0],
            [1.0, -0.3, -1.0, -3.0],
        ],
        "advantages": [
            [1.366667, -0.683333, 0.0, -0.683333],
            [1.766667, 0.466667, 0.0, -2.233333],
            [1.366667, -0.683333, 0.0, -0.683333],
            [1.766667, 0.466667, 0.0, -2.233333],
            [1.366667, -0.683333, 0.0, -0.683333],
            [1.766667, 0.466667, 0.0, -2.233333],
            [1.366667, -0.683333, 0.0, -0.683333],
            [1.766667, 0.466667, 0.0, -2.233333],
        ],
    }

    def test_rewards_advantages_and_weights_equal_the_base_commit(self, tmp_path):
        a = _ns(_argv(_gold(tmp_path), tmp_path / "o", gpu=2, steps=2))
        bundle = s2.dry_bundle(a)
        _fix_weights(bundle["model"])
        tok = bundle["tok"]
        calls = {"n": 0}

        def rollout(prompt_text, g, max_new, temp):
            texts = MECH_GOLDEN_REPLIES[calls["n"] % 2]
            calls["n"] += 1
            return tok.encode(prompt_text)[:16], [tok.encode(t) for t in texts], [False, False, True, False]

        prompts = [_row("g00"), _row("r00", label="unsupported")]
        records: list[dict] = []
        s2.run_steps(
            model=bundle["model"],
            ref_adapter_off=bundle["ref_adapter_off"],
            tok=tok,
            optimiser=bundle["optimiser"],
            prompts=prompts,
            rollout=rollout,
            steps=4,
            g=4,
            max_new=64,
            reward_fn=lambda text, row, tt, truncated=False: s2.reward(
                text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
            ),
            device_sync=lambda: None,
            peak_alloc_mib=lambda: None,
            groups_per_update=2,
            zero_variance_mask=True,
            advantage_mode="dr-grpo",
            mask_truncated=True,
            quote_reward="v2",
            save_all_records=records,
            train_seed=42,
            arm="mech",
            trunc_guard=s2.TruncationHaltGuard(window_updates=10, halt_rate=0.99),
        )
        by_group = [records[i : i + 4] for i in range(0, len(records), 4)]
        assert [[r["reward"] for r in grp] for grp in by_group] == self.GOLDEN["rewards"]
        assert [[r["advantage"] for r in grp] for grp in by_group] == self.GOLDEN["advantages"]
        ten = {n: p.detach() for n, p in bundle["model"].named_parameters() if p.requires_grad}
        opt = bundle["optimiser"].state_dict()
        names = sorted(ten)
        got = {
            "adapter_sum": [float(ten[n].double().sum()) for n in names],
            "adapter_abs": [float(ten[n].double().abs().sum()) for n in names],
            "exp_avg_abs": [
                float(opt["state"][i]["exp_avg"].double().abs().sum()) for i in sorted(opt["state"])
            ],
            "exp_avg_sq_sum": [
                float(opt["state"][i]["exp_avg_sq"].double().sum()) for i in sorted(opt["state"])
            ],
            "steps": [float(opt["state"][i]["step"]) for i in sorted(opt["state"])],
        }
        for key, want in WEIGHT_GOLDEN.items():
            assert got[key] == pytest.approx(want, **WEIGHT_TOLERANCE[key]), key


# -- B5: the guard history --
class TestGuardHistory:
    def test_ct_guard_state_round_trip_carries_the_open_window_and_counters(self):
        def feed(guard, steps):
            out = []
            for st in steps:
                for gr in range(2):
                    out.append(
                        guard.record_group(
                            step=st,
                            group=gr,
                            item_id=f"i{st}{gr}",
                            gold_verdict="supported" if gr == 0 else "unsupported",
                            verdicts=["cannot_tell", "supported"],
                            correct_flags=[False, gr == 0],
                            advantages_list=[0.5, -0.5],
                        )
                    )
            return out

        whole = s2.CtCollapseGuard(window_groups=3, pass_a_k={"i00": 8})
        feed(whole, range(4))
        first = s2.CtCollapseGuard(window_groups=3, pass_a_k={"i00": 8})
        feed(first, range(2))
        second = s2.CtCollapseGuard.from_state(json.loads(json.dumps(first.to_state())), pass_a_k={"i00": 8})
        feed(second, range(2, 4))
        assert second.to_state() == whole.to_state()
        assert all("step" in w for w in whole.window_history)

    def test_every_window_is_written_append_only_and_crash_safe(self, tmp_path):
        a = _ns(_argv(_gold(tmp_path), tmp_path / "o", steps=3, gpu=2))
        bundle = s2.dry_bundle(a)
        hist = tmp_path / "history.jsonl.gz"
        ct = s2.CtCollapseGuard(window_groups=2, pass_a_k={})
        s2.run_steps(
            model=bundle["model"],
            ref_adapter_off=bundle["ref_adapter_off"],
            tok=bundle["tok"],
            optimiser=bundle["optimiser"],
            prompts=[_row("g00"), _row("r00", label="unsupported")],
            rollout=bundle["rollout"],
            steps=3,
            g=2,
            max_new=6,
            reward_fn=s2.dry_reward,
            device_sync=lambda: None,
            peak_alloc_mib=lambda: None,
            groups_per_update=2,
            ct_guard=ct,
            trunc_guard=s2.TruncationHaltGuard(window_updates=5, halt_rate=0.99),
            guard_history_path=hist,
            train_seed=1,
        )
        recs = s2.read_save_all(hist)
        kinds = [r["kind"] for r in recs]
        assert kinds.count("ct_guard_window") == 3 and kinds.count("trunc_window") == 3
        assert [r["step"] for r in recs if r["kind"] == "trunc_window"] == [0, 1, 2]
        assert {"triggered_rules", "s1_distinct_ct_items", "u1_correct_rate"} <= set(recs[0])
        # each append is its own gzip member: a torn tail is dropped on read-back, the rest is kept
        raw = hist.read_bytes()
        hist.write_bytes(raw + gzip.compress(b'{"kind": "torn"}\n', mtime=0)[:-6])
        kept = s2._prepare_save_all_resume(hist)
        assert kept == recs and s2.dropped_archive_path(hist).exists()


# -- B6: the training verdict --
class TestTrainingVerdict:
    @pytest.mark.parametrize(
        ("kw", "line", "code"),
        [
            ({"next_step": 197, "chunk_stop": None}, "S2-TRAIN-COMPLETE", 0),
            ({"next_step": 40, "chunk_stop": {"next_step": 40}}, "S2-TRAIN-CHUNK-OK", 0),
            ({"next_step": 39, "chunk_stop": None}, "S2-TRAIN-FAIL", 1),
            (
                {"next_step": 12, "stopped": {"reason": "trunc_halt", "triggered_rules": ["TRUNC"]}},
                "S2-TRAIN-HALT-TRUNC",
                2,
            ),
            (
                {"next_step": 12, "stopped": {"reason": "ct_guard_u2", "triggered_rules": ["U2"]}},
                "S2-TRAIN-HALT-U2",
                2,
            ),
            ({"next_step": 12, "stopped": {"error": "non-finite loss"}}, "S2-TRAIN-FAIL", 1),
            ({"next_step": 12, "oom": True}, "S2-TRAIN-FAIL", 1),
            ({"next_step": 197, "fits": False, "dry": False}, "S2-TRAIN-FAIL", 1),
            ({"next_step": 197, "fits": False, "dry": True}, "S2-TRAIN-COMPLETE", 0),
        ],
    )
    def test_verdicts_and_exit_codes(self, kw, line, code):
        args = {
            "stopped": None,
            "oom": False,
            "steps_total": 197,
            "chunk_stop": None,
            "fits": True,
            "dry": False,
        }
        got = s2.train_verdict_line(**{**args, **kw})
        assert got == line and s2.exit_status(got) == code

    def test_a_full_run_is_complete_not_limited(self, tmp_path):
        out = tmp_path / "o"
        assert _run(_argv(_gold(tmp_path), out, steps=3)) == 0
        rec = _receipt(out)
        assert rec["verdict"] == "S2-TRAIN-COMPLETE" and rec["training"]["next_step"] == 3
        assert rec["training"]["train_state_saved"] and rec["training"]["guard_history_sha256"]
        assert rec["readout"]["steps_completed"] == 3 and rec["readout"]["steps_requested"] == 3

    def test_a_non_finite_loss_fails_and_saves_no_state(self, tmp_path, monkeypatch):
        import math
        import types

        monkeypatch.setattr(s2, "math", types.SimpleNamespace(isfinite=lambda x: False, ceil=math.ceil))
        out = tmp_path / "o"
        assert _run(_argv(_gold(tmp_path), out, steps=2)) == 1
        rec = _receipt(out)
        assert rec["verdict"] == "S2-TRAIN-FAIL" and rec["training"]["train_state_saved"] is False
        assert not (out / s2.TRAIN_STATE_DIR).exists()


# -- coverage of the remaining paths --
class TestResumeWithTheCtGuard:
    def test_the_ct_guard_state_carries_across_a_resume_bit_for_bit(self, tmp_path, monkeypatch):
        real_loader = s2.load_prepass_baseline

        def loader(path, sha, ids, **kw):  # a pass-A k for every item, without building a pass-A file
            if path is not None and str(path).endswith("pass-a.jsonl"):
                return {str(i): 8 for i in ids}, "a" * 64
            return real_loader(path, sha, ids, **kw)

        monkeypatch.setattr(s2, "load_prepass_baseline", loader)
        gold = _gold(tmp_path)
        guard = ["--ct-guard", "--guard-window-groups", "3", "--pass-a-jsonl", "pass-a.jsonl"]
        guard += ["--pass-a-sha256", "a" * 64]
        full, c1, c2 = tmp_path / "full", tmp_path / "c1", tmp_path / "c2"
        assert _run(_argv(gold, full, *guard)) == 0
        assert _run(_argv(gold, c1, *guard, "--stop-after-updates", "2")) == 0
        assert _run(_argv(gold, c2, *guard, "--resume-from", str(c1))) == 0
        st_full, ten_full, _ = _state(full)
        st_c2, ten_c2, _ = _state(c2)
        _assert_tensors_equal(ten_full, ten_c2)
        assert st_full["ct_guard"] == st_c2["ct_guard"] and st_full["ct_guard"]["window_index"] == 2
        assert _history(full) == _history(c1) + _history(c2)
        assert [h["kind"] for h in _history(full)].count("ct_guard_window") == 2


class TestRemainingPaths:
    def test_a_leftover_partial_state_is_replaced(self, tmp_path):
        a = _ns(_argv(_gold(tmp_path), tmp_path / "o"))
        bundle = s2.dry_bundle(a)
        partial = tmp_path / "o" / s2.TRAIN_STATE_DIR / "0000-step-00001.partial"
        partial.mkdir(parents=True)
        (partial / "trainable.pt").write_bytes(b"torn")
        sha = s2.save_train_state(
            tmp_path / "o",
            model=bundle["model"],
            optimiser=bundle["optimiser"],
            header={"settings": {"groups_per_update": 2}},
            next_step=1,
            steps_total=4,
            chunk_index=1,
            ct_guard=None,
            trunc_guard=s2.TruncationHaltGuard(window_updates=2, halt_rate=0.5),
            prior_chain=[],
            chunk_link={"chunk_dir": "o", "chunk_index": 1, "first_step": 0, "next_step": 1},
            adapter_sha=None,
            halted=None,
            resumed_from=None,
            kind="periodic",
        )[0]
        assert not partial.exists() and len(sha) == 64
        state = json.loads((s2.latest_state_dir(tmp_path / "o") / "state.json").read_text(encoding="utf-8"))
        assert state["draw_position"] == 2 and state["complete"] is False

    def test_a_ct_halt_and_a_truncation_halt_in_one_update_report_both(self, tmp_path):
        class _AlwaysHalt:
            window_history: list = []

            def record_group(self, **kw):
                return {"step": kw["step"], "reason": "ct_guard_u1", "triggered_rules": ["U1"]}

        a = _ns(_argv(_gold(tmp_path), tmp_path / "o"))
        bundle = s2.dry_bundle(a)
        rows, stopped, _ = s2.run_steps(
            model=bundle["model"],
            ref_adapter_off=bundle["ref_adapter_off"],
            tok=bundle["tok"],
            optimiser=bundle["optimiser"],
            prompts=[_row("g00")],
            rollout=bundle["rollout"],
            steps=2,
            g=2,
            max_new=6,
            reward_fn=s2.dry_reward,
            device_sync=lambda: None,
            peak_alloc_mib=lambda: None,
            groups_per_update=1,
            ct_guard=_AlwaysHalt(),
            trunc_guard=s2.TruncationHaltGuard(window_updates=1, halt_rate=0.01),
            train_seed=3,
        )
        assert len(rows) == 1 and stopped["reason"] == "ct_guard_u1"
        assert stopped["also_triggered"]["reason"] == "trunc_halt"

    def test_a_two_value_rollout_is_retried_and_blank_evidence_lines_are_skipped(self, tmp_path):
        gold = _gold(tmp_path)

        def build(a):
            bundle = s2.dry_bundle(a)
            tok = bundle["tok"]

            def rollout(prompt_text, g, max_new, temp):
                return tok.encode(prompt_text)[:16], [tok.encode(_reply("cannot_tell"))] * g

            bundle["rollout"] = rollout
            return bundle

        out = tmp_path / "o"
        assert _run(_argv(gold, out, arm="critic", steps=1, gpu=1), build=build) == 0
        retries = s2.read_save_all(out / "s2-hf-critic-retries.jsonl.gz")
        assert len(retries) == 2 and all(r["truncated"] is True for r in retries)  # no EOS, at the cap
        hint = s2.critic_hint("The parser counts tokens.", ["\n\nthe parser\n\n"], [("x", False)])
        assert hint["retrieved_lines"] == 1


# produced by aspire-si 7f9b4cb3's run_steps on TestMechArmMatchesTheBase's scenario: 4 updates, _fix_weights
WEIGHT_GOLDEN = {
    "adapter_sum": [
        0.11292799317743629,
        0.3390660954755731,
        -0.11871607862121891,
        -0.6830788990482688,
        0.12282950707594864,
        1.0100286953384057,
        -0.12478516815463081,
        -1.3184345296467654,
        0.019002936824108474,
        2.2615917898219777,
        -0.0015995634021237493,
        -2.098695441207383,
        -0.015840491112612654,
        1.8920997587265447,
        0.03313214038826118,
        -1.648422405931342,
    ],
    "adapter_abs": [
        16.329101423209067,
        324.9488071927917,
        16.335935196679202,
        324.9901679949835,
        16.339718816656386,
        325.30805834999774,
        16.334276767491247,
        325.68734987609787,
        16.24931084875425,
        326.57224320569367,
        16.246759076049784,
        326.38898786896607,
        16.246399645686324,
        326.19154973712284,
        16.249602493664497,
        326.0166613989204,
    ],
    "exp_avg_abs": [
        9.014724451591495e-06,
        7.186527331337405e-08,
        5.682040783887732e-06,
        6.726039318142088e-08,
        0.02120312451855,
        0.00014203357862163646,
        0.0005873367884721148,
        0.0006613491875258615,
        7.221958479316061e-08,
        7.670555628221431e-10,
        4.025583206394097e-08,
        7.678001072381164e-10,
        0.013829851717034103,
        0.00015453866363772384,
        0.0005963033239355653,
        0.00040298870662480724,
    ],
    "exp_avg_sq_sum": [
        1.0660882610803158e-14,
        7.645731326004864e-19,
        4.1046141239414e-15,
        6.752353964837532e-19,
        5.7853564927994384e-08,
        2.1036079562680724e-12,
        3.554067441640244e-11,
        5.1113971188619555e-11,
        6.368710167929185e-19,
        6.491283415209773e-23,
        2.6850480494990146e-19,
        8.90216374269743e-23,
        2.0877380041983172e-08,
        2.3937062620664808e-12,
        4.0839744395778257e-11,
        2.0065377914235978e-11,
    ],
    "steps": [
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
        4.0,
    ],
}
# A KL beta of 0.05 instead of 0.04 moves exp_avg_abs by 5.5e-4 and exp_avg_sq_sum by 7.7e-4 (relative), so
# the 1e-4 lines catch it with a 5x margin over float32 BLAS noise.
WEIGHT_TOLERANCE = {
    "adapter_sum": {"abs": 1e-8, "rel": 0.0},
    "adapter_abs": {"rel": 1e-5, "abs": 0.0},
    "exp_avg_abs": {"rel": 1e-4, "abs": 0.0},
    "exp_avg_sq_sum": {"rel": 1e-4, "abs": 0.0},
    "steps": {"abs": 0.0},
}


# -- fix round 1: periodic saves, kill and resume, header, RNG restore, forked critic RNG --
class _Killed(BaseException):
    """A simulated kill: escapes every handler, as a hard stop would."""


def _killing_build(kill_at_call, exc=_Killed):
    calls = {"n": 0}

    def build(a):
        bundle = s2.dry_bundle(a)
        real = bundle["rollout"]

        def rollout(*args):
            if calls["n"] == kill_at_call:
                calls["n"] += 1
                raise exc("killed mid-update")
            calls["n"] += 1
            return real(*args)

        bundle["rollout"] = rollout
        return bundle

    return build


def _segment(d, name, cut_copy=None):
    path = Path(d) / (cut_copy or name)
    if name.endswith(".gz"):
        return s2.read_save_all(path) if path.exists() else []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class TestPeriodicStateAndKill:
    @pytest.mark.parametrize("exc", [_Killed, TimeoutError])
    def test_k_in_one_go_equal_a_run_killed_after_a_periodic_save_and_resumed(self, tmp_path, exc):
        """Required: K = 6 in one go == saves every 2, killed in update 3 (after the save at 2), resumed."""
        gold = _gold(tmp_path)
        full, killed, resumed = tmp_path / "full", tmp_path / "killed", tmp_path / "resumed"
        every = ("--state-every-updates", "2")
        assert _run(_argv(gold, full, *every, steps=6)) == 0
        build = _killing_build(kill_at_call=7, exc=exc)  # update 3, group 1 (2 groups per update)
        if exc is _Killed:
            with pytest.raises(_Killed):
                _run(_argv(gold, killed, *every, steps=6), build=build)
        else:  # a timeout is caught by the trainer: the chunk fails and saves no final state
            assert _run(_argv(gold, killed, *every, steps=6), build=build) == 1
            assert _receipt(killed)["verdict"] == "S2-TRAIN-FAIL"
        killed_state, _, _ = _state(killed)
        assert killed_state["kind"] == "periodic" and killed_state["next_step"] == 2
        assert _run(_argv(gold, resumed, *every, "--resume-from", str(killed), steps=6)) == 0

        st_f, ten_f, opt_f = _state(full)
        st_r, ten_r, opt_r = _state(resumed)
        _assert_tensors_equal(ten_f, ten_r)
        _assert_optim_equal(opt_f, opt_r)
        assert st_f["trunc_guard"] == st_r["trunc_guard"] and st_f["rng_plain"] == st_r["rng_plain"]
        cuts = {c["archive"]: c for c in _receipt(resumed)["training"]["resumed_from"]["cuts"]}
        assert {"save_all", "guard_history", "steps"} <= set(cuts)
        for c in cuts.values():
            assert c["original_bytes"] > c["bytes"] and (killed / c["copy"]).is_file()
            assert (killed / c["file"]).stat().st_size == c["original_bytes"]  # the original is kept whole
        names = s2.tracked_archives()
        for key in ("save_all", "guard_history", "steps"):
            whole = _segment(full, names[key])
            split = _segment(killed, names[key], cuts[key]["copy"]) + _segment(resumed, names[key])
            if key == "steps":
                whole, split = _untimed(whole), _untimed(split)
            assert whole == split, key
        link = st_r["save_all_chain"][0]
        assert link["chunk_dir"] == "killed"
        assert link["archives"]["save_all"]["cut_copy"] == cuts["save_all"]["copy"]

    def test_only_the_last_two_states_are_kept(self, tmp_path):
        out = tmp_path / "o"
        assert _run(_argv(_gold(tmp_path), out, "--state-every-updates", "1", steps=4)) == 0
        dirs = s2.state_dirs(out)
        assert [d.name for d in dirs] == ["0002-step-00003", "0003-step-00004"]
        kinds = [json.loads((d / "state.json").read_text(encoding="utf-8"))["kind"] for d in dirs]
        assert kinds == ["periodic", "final"]

    def test_a_change_inside_the_saved_prefix_refuses(self, tmp_path):
        gold = _gold(tmp_path)
        killed = tmp_path / "killed"
        with pytest.raises(_Killed):
            _run(_argv(gold, killed, "--state-every-updates", "2", steps=6), build=_killing_build(7))
        steps_file = killed / "s2-hf-steps.jsonl"
        steps_file.write_bytes(b"X" + steps_file.read_bytes()[1:])
        with pytest.raises(SystemExit, match="RESUME_SAVE_ALL_CHANGED"):
            _run(
                _argv(
                    gold, tmp_path / "r", "--state-every-updates", "2", "--resume-from", str(killed), steps=6
                )
            )

    def test_the_pt_files_are_fsynced(self, tmp_path, monkeypatch):
        synced = []
        real = s2.os.fsync

        def spy(fd):
            synced.append(fd)
            real(fd)

        monkeypatch.setattr(s2.os, "fsync", spy)
        a = _ns(_argv(_gold(tmp_path), tmp_path / "o"))
        bundle = s2.dry_bundle(a)
        s2.save_train_state(
            tmp_path / "o",
            model=bundle["model"],
            optimiser=bundle["optimiser"],
            header={"settings": {}},
            next_step=1,
            steps_total=2,
            chunk_index=1,
            ct_guard=None,
            trunc_guard=s2.TruncationHaltGuard(window_updates=2, halt_rate=0.5),
            prior_chain=[],
            chunk_link={"chunk_dir": "o"},
            adapter_sha=None,
            halted=None,
            resumed_from=None,
            kind="periodic",
        )
        assert len(synced) == 4  # trainable.pt, optimizer.pt, rng.pt, state.json

    def test_bad_state_interval_is_refused_and_recorded_otherwise(self, tmp_path):
        with pytest.raises(SystemExit, match="invalid_state_every_updates"):
            s2.validate_training_args(
                _ns(_argv(_gold(tmp_path), tmp_path / "o", "--state-every-updates", "0"))
            )
        with pytest.raises(SystemExit, match="training_flags_required"):
            s2.validate_training_args(
                _ns(_argv(_gold(tmp_path), tmp_path / "o", "--state-every-updates", "3", train=False))
            )
        out = tmp_path / "rec"
        assert _run(_argv(_gold(tmp_path), out, steps=1)) == 0
        hdr = json.loads((out / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert hdr["state_every_updates"] == 10 and hdr["settings"]["state_every_updates"] == 10


class TestHeaderFix:
    @pytest.mark.parametrize("key", ["quote_rule_sha256", "s2_draw_sha256"])
    def test_a_changed_reward_or_draw_module_refuses_resume(self, tmp_path, monkeypatch, key):
        gold = _gold(tmp_path)
        c1 = tmp_path / "c1"
        assert _run(_argv(gold, c1, "--stop-after-updates", "1", steps=2)) == 0
        hdr = json.loads((c1 / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert len(hdr[key]) == 64
        real = s2.dependency_sha256s
        monkeypatch.setattr(s2, "dependency_sha256s", lambda: {**real(), key: "0" * 64})
        with pytest.raises(SystemExit, match=f"RESUME_HEADER_MISMATCH.*{key}"):
            _run(_argv(gold, tmp_path / "c2", "--resume-from", str(c1), steps=2))

    def test_the_header_records_that_cuda_kernels_are_not_deterministic(self, tmp_path):
        out = tmp_path / "o"
        assert _run(_argv(_gold(tmp_path), out, steps=1)) == 0
        hdr = json.loads((out / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert hdr["cuda_deterministic"] is False
        for k in ("torch_deterministic_algorithms", "cudnn_deterministic", "cudnn_benchmark"):
            assert isinstance(hdr[k], bool)
        assert "cublas_workspace_config" in hdr

    def test_model_free_checks_run_before_the_model_is_built(self, tmp_path):
        gold = _gold(tmp_path)
        done = tmp_path / "done"
        assert _run(_argv(gold, done, steps=1)) == 0

        def no_build(a):
            raise AssertionError("the model was built before the resume checks")

        with pytest.raises(SystemExit, match="RESUME_RUN_COMPLETE"):
            _run(_argv(gold, tmp_path / "x", "--resume-from", str(done), steps=1), build=no_build)


class TestRngRestore:
    def test_resume_starts_training_from_the_saved_global_rng_state(self, tmp_path, monkeypatch):
        """Fails if the RNG restore is removed: the generators at run_steps entry equal the saved ones."""
        import random

        import numpy as np

        gold = _gold(tmp_path)
        c1 = tmp_path / "c1"
        assert _run(_argv(gold, c1, "--stop-after-updates", "1", steps=2)) == 0
        sdir = s2.latest_state_dir(c1)
        saved = torch.load(sdir / "rng.pt", weights_only=True)
        plain = json.loads((sdir / "state.json").read_text(encoding="utf-8"))["rng_plain"]
        seen = {}
        real = s2.run_steps

        def spy(**kw):
            seen["torch"] = torch.get_rng_state().clone()
            seen["python"] = random.getstate()
            seen["numpy"] = np.random.get_state()
            return real(**kw)

        monkeypatch.setattr(s2, "run_steps", spy)
        assert _run(_argv(gold, tmp_path / "c2", "--resume-from", str(c1), steps=2)) == 0
        assert torch.equal(seen["torch"], saved["torch_cpu"])
        assert list(seen["python"][1]) == plain["python"][1]
        assert [int(x) for x in seen["numpy"][1]] == plain["numpy"][1]

    def test_the_critic_retry_leaves_the_global_rng_where_the_mech_arm_leaves_it(self, tmp_path):
        gold = _gold(tmp_path)
        mech, critic = tmp_path / "mech", tmp_path / "critic"
        assert _run(_argv(gold, mech, arm="mech", steps=2)) == 0
        assert _run(_argv(gold, critic, arm="critic", steps=2)) == 0
        rng_m = torch.load(s2.latest_state_dir(mech) / "rng.pt", weights_only=True)
        rng_c = torch.load(s2.latest_state_dir(critic) / "rng.pt", weights_only=True)
        assert torch.equal(rng_m["torch_cpu"], rng_c["torch_cpu"])


# -- fix round 2: resume lineage, overrides, the cut re-check, CUT.json --
class TestResumeLineage:
    def _chain(self, tmp_path):
        gold = _gold(tmp_path)
        c1, c2 = tmp_path / "c1", tmp_path / "c2"
        assert _run(_argv(gold, c1, "--stop-after-updates", "1", steps=4)) == 0
        assert _run(_argv(gold, c2, "--stop-after-updates", "1", "--resume-from", str(c1), steps=4)) == 0
        return gold, c1, c2

    def test_the_normal_chain_marks_each_consumed_chunk_once(self, tmp_path):
        gold, c1, c2 = self._chain(tmp_path)
        markers = s2.consumed_markers(c1)
        assert [m.name for m in markers] == ["CONSUMED_BY.json"]
        mark = json.loads(markers[0].read_text(encoding="utf-8"))
        assert mark["consumed_by"] == "c2" and mark["resumed_at_step"] == 1
        assert s2.consumed_markers(c2) == []
        c3 = tmp_path / "c3"
        assert _run(_argv(gold, c3, "--resume-from", str(c2), steps=4)) == 0
        assert _receipt(c3)["verdict"] == "S2-TRAIN-COMPLETE"
        assert _receipt(c3)["training"]["forked_or_rewound"] is False

    def test_resuming_an_older_chunk_and_forking_are_refused(self, tmp_path):
        gold, c1, c2 = self._chain(tmp_path)
        with pytest.raises(SystemExit, match="RESUME_ALREADY_CONSUMED"):  # older chunk / a second child
            _run(_argv(gold, tmp_path / "fork", "--resume-from", str(c1), steps=4))
        for m in s2.consumed_markers(c1):  # with the marker gone, the descendant chunk still shows it
            m.unlink()
        with pytest.raises(SystemExit, match="RESUME_NOT_LATEST"):
            _run(_argv(gold, tmp_path / "fork2", "--resume-from", str(c1), steps=4))

    def test_an_older_state_than_the_chunks_newest_is_refused(self, tmp_path):
        gold = _gold(tmp_path)
        c1 = tmp_path / "c1"
        assert _run(_argv(gold, c1, "--stop-after-updates", "2", "--state-every-updates", "1", steps=4)) == 0
        newer, older = s2.state_dirs(c1)[-1], s2.state_dirs(c1)[0]
        import shutil

        # a later-step state that sorts first: the latest-by-sequence state is then not the newest step
        shutil.copytree(newer, c1 / s2.TRAIN_STATE_DIR / "0000-step-00009")
        shutil.rmtree(older)
        state = json.loads(
            (c1 / s2.TRAIN_STATE_DIR / "0000-step-00009" / "state.json").read_text(encoding="utf-8")
        )
        state["next_step"] = 3
        (c1 / s2.TRAIN_STATE_DIR / "0000-step-00009" / "state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )
        with pytest.raises(SystemExit, match="RESUME_NOT_LATEST"):
            s2.precheck_resume(c1, steps_total=4)

    def test_an_override_is_recorded_and_marks_the_chunk_and_every_later_one(self, tmp_path):
        gold, c1, c2 = self._chain(tmp_path)
        fork = tmp_path / "fork"
        reason = ["--resume-override-reason", "rewind to chunk 1 after a bad node"]
        assert (
            _run(_argv(gold, fork, "--stop-after-updates", "1", "--resume-from", str(c1), *reason, steps=4))
            == 0
        )
        rec = _receipt(fork)
        assert rec["verdict"] == "S2-TRAIN-CHUNK-OK FORKED_OR_REWOUND"
        assert rec["training"]["forked_or_rewound"] is True
        assert rec["training"]["override_reasons"][0]["reason"] == "rewind to chunk 1 after a bad node"
        assert rec["training"]["override_reasons"][0]["overridden"] == [
            "RESUME_ALREADY_CONSUMED",
            "RESUME_NOT_LATEST",
        ]
        assert rec["run"]["resume_override_reason"] == "rewind to chunk 1 after a bad node"
        hdr = json.loads((fork / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert hdr["forked_or_rewound"] is True and hdr["resume_override_reason"].startswith("rewind")
        assert [m.name for m in s2.consumed_markers(c1)] == ["CONSUMED_BY.1.json", "CONSUMED_BY.json"]
        later = tmp_path / "later"
        assert _run(_argv(gold, later, "--resume-from", str(fork), steps=4)) == 0
        rec2 = _receipt(later)
        assert (
            rec2["verdict"] == "S2-TRAIN-COMPLETE FORKED_OR_REWOUND" and s2.exit_status(rec2["verdict"]) == 0
        )
        hdr2 = json.loads((later / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert hdr2["forked_or_rewound"] is True and hdr2["resume_override_reason"] is None
        assert hdr2["override_reasons"][0]["chunk_dir"] == "fork"
        assert _state(later)[0]["lineage"]["forked_or_rewound"] is True

    def test_an_empty_or_orphan_override_reason_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="invalid_resume_override_reason"):
            s2.validate_training_args(
                _ns(_argv(_gold(tmp_path), tmp_path / "o", "--resume-override-reason", "x"))
            )
        with pytest.raises(SystemExit, match="invalid_resume_override_reason"):
            s2.validate_training_args(
                _ns(
                    _argv(
                        _gold(tmp_path), tmp_path / "o", "--resume-from", "p", "--resume-override-reason", " "
                    )
                )
            )


class TestCutRecheck:
    def test_the_cut_rechecks_the_prefix_hashes_the_copy_and_writes_cut_json(self, tmp_path):
        gold = _gold(tmp_path)
        killed, resumed = tmp_path / "killed", tmp_path / "resumed"
        with pytest.raises(_Killed):
            _run(_argv(gold, killed, "--state-every-updates", "2", steps=6), build=_killing_build(7))
        assert (
            _run(_argv(gold, resumed, "--state-every-updates", "2", "--resume-from", str(killed), steps=6))
            == 0
        )
        cut = json.loads((killed / "CUT.json").read_text(encoding="utf-8"))
        assert cut["resumed_by"] == "resumed" and cut["next_step"] == 2
        for entry in cut["copies"].values():
            assert entry["copy_sha256"] == entry["sha256"] == s2._file_sha256(killed / entry["copy"])
        link = _state(resumed)[0]["save_all_chain"][0]["archives"]["save_all"]
        assert link["cut_copy_sha256"] == cut["copies"]["save_all"]["copy_sha256"]
        cuts = _receipt(resumed)["training"]["resumed_from"]["cuts"]
        assert all(c["copy_sha256"] == c["sha256"] for c in cuts)

    def test_a_prefix_changed_between_the_precheck_and_the_cut_refuses(self, tmp_path):
        gold = _gold(tmp_path)
        killed = tmp_path / "killed"
        with pytest.raises(_Killed):
            _run(_argv(gold, killed, "--state-every-updates", "2", steps=6), build=_killing_build(7))
        state = s2.precheck_resume(killed, steps_total=6)
        assert state["_cuts"]
        archive = killed / state["_cuts"][0]["file"]
        archive.write_bytes(b"Z" + archive.read_bytes()[1:])
        with pytest.raises(SystemExit, match="changed before it was cut"):
            s2.finish_resume(state, header=state["header"], out_dir=tmp_path / "r")


# -- fix round 3: two resumes racing for one chunk --
class TestResumeRace:
    def _racing(self, tmp_path, *loser_extra, winner_steps=4, loser_steps=4, every=None):
        """B passes the precheck first; while B builds its model, A resumes from the same chunk and finishes.
        Both passed the precheck before either wrote a marker."""
        gold = _gold(tmp_path)
        c1, a_out, b_out = tmp_path / "c1", tmp_path / "A", tmp_path / "B"
        extra = ("--state-every-updates", str(every)) if every else ()
        assert _run(_argv(gold, c1, *extra, "--stop-after-updates", "1", steps=winner_steps)) == 0

        def build_b(ns):
            assert _run(_argv(gold, a_out, *extra, "--resume-from", str(c1), steps=winner_steps)) == 0
            return s2.dry_bundle(ns)

        rc = _run(
            _argv(gold, b_out, *extra, *loser_extra, "--resume-from", str(c1), steps=loser_steps),
            build=build_b,
        )
        return c1, a_out, b_out, rc

    def test_exactly_one_of_two_racing_resumes_continues(self, tmp_path):
        c1, a_out, b_out, rc = self._racing(tmp_path)
        assert _receipt(a_out)["verdict"] == "S2-TRAIN-COMPLETE"
        rec = _receipt(b_out)
        assert rc == 2 and rec["verdict"] == "S2-TRAIN-HALT-RESUME_RACE_LOST"
        assert (
            rec["stopped"]["reason"] == "resume_race_lost" and "RESUME_RACE_LOST" in rec["stopped"]["error"]
        )
        assert not (b_out / s2.TRAIN_STATE_DIR).exists() and rec["training"]["train_state_saved"] is False
        assert [m.name for m in s2.consumed_markers(c1)] == ["CONSUMED_BY.json"]
        assert json.loads((c1 / "CONSUMED_BY.json").read_text(encoding="utf-8"))["consumed_by"] == "A"

    def test_the_race_is_lost_at_a_periodic_save_before_the_state_is_written(self, tmp_path):
        c1, a_out, b_out, rc = self._racing(tmp_path, every=1)
        rec = _receipt(b_out)
        assert rc == 2 and rec["verdict"] == "S2-TRAIN-HALT-RESUME_RACE_LOST"
        assert rec["stopped"]["step"] == 2 and not (b_out / s2.TRAIN_STATE_DIR).exists()

    def test_an_override_continues_marked_forked(self, tmp_path):
        c1, a_out, b_out, rc = self._racing(tmp_path, "--resume-override-reason", "deliberate second branch")
        rec = _receipt(b_out)
        assert rc == 0 and rec["verdict"] == "S2-TRAIN-COMPLETE FORKED_OR_REWOUND"
        assert [m.name for m in s2.consumed_markers(c1)] == ["CONSUMED_BY.1.json", "CONSUMED_BY.json"]

    def test_a_marker_naming_this_chunk_is_its_own_claim(self, tmp_path):
        rec = {"consumed_by": "A"}
        first = s2.write_consumed_marker(tmp_path, rec)
        assert s2.write_consumed_marker(tmp_path, rec) == first
        with pytest.raises(s2.ResumeRaceLostError):
            s2.write_consumed_marker(tmp_path, {"consumed_by": "B"})


# -- Part 1, Part 3.1 & Part 3.3 (a): --ct-guard-mode halt|monitor --
class TestCtGuardModeMonitor:
    def test_registered_path_default_and_explicit_halt_are_byte_identical(self, tmp_path, monkeypatch):
        real_loader = s2.load_prepass_baseline

        def loader(path, sha, ids, **kw):
            if path is not None and str(path).endswith("pass-a.jsonl"):
                return {str(i): 8 for i in ids}, "a" * 64
            return real_loader(path, sha, ids, **kw)

        monkeypatch.setattr(s2, "load_prepass_baseline", loader)
        gold = _gold(tmp_path)
        guard_flags = [
            "--ct-guard",
            "--guard-window-groups",
            "2",
            "--pass-a-jsonl",
            "pass-a.jsonl",
            "--pass-a-sha256",
            "a" * 64,
        ]
        # 1. Normal chunk stop: omitted vs --ct-guard-mode halt
        c_def, c_hlt = tmp_path / "c_def", tmp_path / "c_hlt"
        assert _run(_argv(gold, c_def, *guard_flags, "--stop-after-updates", "2", steps=4, gpu=2)) == 0
        assert (
            _run(
                _argv(
                    gold,
                    c_hlt,
                    *guard_flags,
                    "--ct-guard-mode",
                    "halt",
                    "--stop-after-updates",
                    "2",
                    steps=4,
                    gpu=2,
                )
            )
            == 0
        )
        hdr_def = json.loads((c_def / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        hdr_hlt = json.loads((c_hlt / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert hdr_def == hdr_hlt
        assert "ct_guard_mode" not in hdr_def["settings"] and "exploratory" not in hdr_def
        assert (c_def / "s2-hf-save-all.jsonl.gz").read_bytes() == (
            c_hlt / "s2-hf-save-all.jsonl.gz"
        ).read_bytes()
        assert (c_def / "s2-hf-guard-history.jsonl.gz").read_bytes() == (
            c_hlt / "s2-hf-guard-history.jsonl.gz"
        ).read_bytes()
        assert _untimed(_rows(c_def)) == _untimed(_rows(c_hlt))
        _, ten_d, opt_d = _state(c_def)
        _, ten_h, opt_h = _state(c_hlt)
        _assert_tensors_equal(ten_d, ten_h)
        _assert_optim_equal(opt_d, opt_h)

        # 2. Dry halt: omitted vs --ct-guard-mode halt
        h_def, h_hlt = tmp_path / "h_def", tmp_path / "h_hlt"
        assert _run(_argv(gold, h_def, *guard_flags, steps=4, gpu=2, window=1, halt=0.01)) == 2
        assert (
            _run(
                _argv(
                    gold,
                    h_hlt,
                    *guard_flags,
                    "--ct-guard-mode",
                    "halt",
                    steps=4,
                    gpu=2,
                    window=1,
                    halt=0.01,
                )
            )
            == 2
        )
        assert _receipt(h_def)["verdict"] == _receipt(h_hlt)["verdict"] == "S2-TRAIN-HALT-TRUNC"
        assert _receipt(h_def)["stopped"] == _receipt(h_hlt)["stopped"]
        assert (h_def / "s2-hf-save-all.jsonl.gz").read_bytes() == (
            h_hlt / "s2-hf-save-all.jsonl.gz"
        ).read_bytes()
        assert (h_def / "s2-hf-guard-history.jsonl.gz").read_bytes() == (
            h_hlt / "s2-hf-guard-history.jsonl.gz"
        ).read_bytes()

    def test_ct_guard_mode_bad_value_is_refused(self, tmp_path):
        with pytest.raises(SystemExit, match="--ct-guard-mode"):
            s2.CtCollapseGuard(window_groups=2, pass_a_k={}, mode="invalid")
        a = _ns(_argv(_gold(tmp_path), tmp_path / "o"))
        a.ct_guard_mode = "invalid"
        with pytest.raises(SystemExit, match="--ct-guard-mode"):
            s2.active_flags(a)
        with pytest.raises(SystemExit, match="--ct-guard-mode"):
            s2.execute(a, build=s2.dry_bundle)

    def test_monitor_mode_logs_identical_windows_without_halting_or_zeroing_gradients(
        self, tmp_path, monkeypatch
    ):
        real_loader = s2.load_prepass_baseline

        def loader(path, sha, ids, **kw):
            if path is not None and str(path).endswith("pass-a.jsonl"):
                return {str(i): 8 for i in ids}, "a" * 64
            return real_loader(path, sha, ids, **kw)

        monkeypatch.setattr(s2, "load_prepass_baseline", loader)
        gold = _gold(tmp_path)
        # Every group returns cannot_tell + supported so S1 (3 distinct supported items with CT and positive
        # advantage) triggers in a 3-group window.
        build_h, _ = _fake_rollout_build(
            lambda i, p: [_reply("cannot_tell"), _reply("unsupported")], lambda i: [False, False]
        )
        build_m, _ = _fake_rollout_build(
            lambda i, p: [_reply("cannot_tell"), _reply("unsupported")], lambda i: [False, False]
        )
        guard_flags = [
            "--ct-guard",
            "--guard-window-groups",
            "3",
            "--pass-a-jsonl",
            "pass-a.jsonl",
            "--pass-a-sha256",
            "a" * 64,
        ]
        out_halt = tmp_path / "halt"
        out_mon = tmp_path / "monitor"
        rc_h = _run(_argv(gold, out_halt, *guard_flags, steps=3, gpu=3), build=build_h)
        rc_m = _run(
            _argv(gold, out_mon, *guard_flags, "--ct-guard-mode", "monitor", steps=3, gpu=3), build=build_m
        )
        rec_h = _receipt(out_halt)
        rec_m = _receipt(out_mon)
        assert rc_h == 2 and rec_h["verdict"] == "S2-TRAIN-HALT-S1"
        assert rc_m == 0 and rec_m["verdict"] == "S2-TRAIN-COMPLETE"
        assert rec_m["stopped"] is None
        hdr_h = json.loads((out_halt / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        hdr_m = json.loads((out_mon / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert "ct_guard_mode" not in hdr_h["settings"]
        assert hdr_m["settings"]["ct_guard_mode"] == "monitor"
        assert "exploratory" not in hdr_h and "exploratory" not in hdr_m

        hist_h = [h for h in _history(out_halt) if h["kind"] == "ct_guard_window"]
        hist_m = [h for h in _history(out_mon) if h["kind"] == "ct_guard_window"]
        assert len(hist_h) == 1 and len(hist_m) == 3
        assert hist_h[0] == hist_m[0] and hist_m[0]["triggered_rules"] == ["S1"]
        assert all(w["triggered_rules"] == ["S1"] for w in hist_m)

        # In monitor mode, gradients were applied across all 3 updates (optimizer steps = 3).
        _, _, opt_h = _state(out_halt)
        _, _, opt_m = _state(out_mon)
        assert opt_h["state"] == {}  # halted on update 0 -> gradients discarded
        assert all(int(st["step"]) == 3 for st in opt_m["state"].values())

    def test_truncation_halt_still_fires_in_monitor_mode(self, tmp_path, monkeypatch):
        real_loader = s2.load_prepass_baseline

        def loader(path, sha, ids, **kw):
            if path is not None and str(path).endswith("pass-a.jsonl"):
                return {str(i): 8 for i in ids}, "a" * 64
            return real_loader(path, sha, ids, **kw)

        monkeypatch.setattr(s2, "load_prepass_baseline", loader)
        gold = _gold(tmp_path)
        build, _ = _fake_rollout_build(
            lambda i, p: [_reply("cannot_tell"), _reply("unsupported")], lambda i: [True, True]
        )
        out = tmp_path / "mon-trunc"
        rc = _run(
            _argv(
                gold,
                out,
                "--ct-guard-mode",
                "monitor",
                "--guard-window-groups",
                "2",
                "--pass-a-jsonl",
                "pass-a.jsonl",
                "--pass-a-sha256",
                "a" * 64,
                steps=3,
                gpu=2,
                window=1,
                halt=0.20,
            ),
            build=build,
        )
        rec = _receipt(out)
        assert rc == 2 and rec["verdict"] == "S2-TRAIN-HALT-TRUNC"
        assert rec["stopped"]["reason"] == "trunc_halt"


# -- Part 2 & Part 3.3 (b)-(d): --resume-from-halted-explore "<reason>" --
class TestExploreResumeFromHalted:
    def _halted_chunk(self, tmp_path, monkeypatch, *, arm="mech"):
        real_loader = s2.load_prepass_baseline

        def loader(path, sha, ids, **kw):
            if path is not None and str(path).endswith("pass-a.jsonl"):
                return {str(i): 8 for i in ids}, "a" * 64
            return real_loader(path, sha, ids, **kw)

        monkeypatch.setattr(s2, "load_prepass_baseline", loader)
        gold = _gold(tmp_path)

        # Step 0 has normal replies; step 1 triggers S1 in a 3-group window so step 0 is saved and step 1 halts
        def texts(i, prompt):
            if i < 2:
                return [_reply("supported"), _reply("unsupported")]
            return [_reply("cannot_tell"), _reply("unsupported")]

        build, _ = _fake_rollout_build(texts, lambda i: [False, False])
        guard_flags = [
            "--ct-guard",
            "--guard-window-groups",
            "3",
            "--pass-a-jsonl",
            "pass-a.jsonl",
            "--pass-a-sha256",
            "a" * 64,
            "--state-every-updates",
            "1",
        ]
        orig = tmp_path / "run-training" / "seed-42" / "chunk-01"
        rc = _run(_argv(gold, orig, *guard_flags, arm=arm, steps=4, gpu=3), build=build)
        assert rc == 2 and _receipt(orig)["verdict"] == "S2-TRAIN-HALT-S1"
        return gold, orig, guard_flags

    def test_every_explore_refusal_code(self, tmp_path, monkeypatch):
        gold, orig, guard_flags = self._halted_chunk(tmp_path, monkeypatch)
        import shutil

        halted_copy = tmp_path / "explore-work" / "chunk-01-copy"
        shutil.copytree(orig, halted_copy)
        out_ok = tmp_path / "explore-work" / "chunk-02-explore"

        # 1. EXPLORE_REASON_REQUIRED (empty/whitespace reason)
        with pytest.raises(SystemExit, match="EXPLORE_REASON_REQUIRED"):
            s2.validate_training_args(
                _ns(
                    _argv(
                        gold,
                        out_ok,
                        *guard_flags,
                        "--ct-guard-mode",
                        "monitor",
                        "--resume-from",
                        str(halted_copy),
                        "--resume-from-halted-explore",
                        "   ",
                    )
                )
            )
        with pytest.raises(SystemExit, match="EXPLORE_REASON_REQUIRED"):
            s2.precheck_resume(halted_copy, steps_total=4, explore_reason="  ")

        # 2. EXPLORE_RESUME_FROM_REQUIRED (--resume-from omitted)
        with pytest.raises(SystemExit, match="EXPLORE_RESUME_FROM_REQUIRED"):
            s2.validate_training_args(
                _ns(
                    _argv(
                        gold,
                        out_ok,
                        *guard_flags,
                        "--ct-guard-mode",
                        "monitor",
                        "--resume-from-halted-explore",
                        "post-halt check",
                    )
                )
            )

        # 3. EXPLORE_OUT_IN_RUN_TRAINING (--out inside run-training)
        bad_out = tmp_path / "run-training" / "explore-out"
        with pytest.raises(SystemExit, match="EXPLORE_OUT_IN_RUN_TRAINING"):
            s2.validate_training_args(
                _ns(
                    _argv(
                        gold,
                        bad_out,
                        *guard_flags,
                        "--ct-guard-mode",
                        "monitor",
                        "--resume-from",
                        str(halted_copy),
                        "--resume-from-halted-explore",
                        "post-halt check",
                    )
                )
            )

        # 4. EXPLORE_RESUME_FROM_IN_RUN_TRAINING (--resume-from inside run-training)
        with pytest.raises(SystemExit, match="EXPLORE_RESUME_FROM_IN_RUN_TRAINING"):
            s2.validate_training_args(
                _ns(
                    _argv(
                        gold,
                        out_ok,
                        *guard_flags,
                        "--ct-guard-mode",
                        "monitor",
                        "--resume-from",
                        str(orig),
                        "--resume-from-halted-explore",
                        "post-halt check",
                    )
                )
            )
        with pytest.raises(SystemExit, match="EXPLORE_RESUME_FROM_IN_RUN_TRAINING"):
            s2.precheck_resume(orig, steps_total=4, explore_reason="post-halt check")

        # 5. EXPLORE_REQUIRES_CT_GUARD_MONITOR (--ct-guard-mode != monitor)
        with pytest.raises(SystemExit, match="EXPLORE_REQUIRES_CT_GUARD_MONITOR"):
            s2.validate_training_args(
                _ns(
                    _argv(
                        gold,
                        out_ok,
                        *guard_flags,
                        "--resume-from",
                        str(halted_copy),
                        "--resume-from-halted-explore",
                        "post-halt check",
                    )
                )
            )

        # 6. EXPLORE_STATE_NOT_HALTED (state.json has halted == null)
        non_halted = tmp_path / "explore-work" / "chunk-ok"
        assert _run(_argv(gold, non_halted, *guard_flags, "--stop-after-updates", "1", steps=4, gpu=3)) == 0
        with pytest.raises(SystemExit, match="EXPLORE_STATE_NOT_HALTED"):
            _run(
                _argv(
                    gold,
                    out_ok,
                    *guard_flags,
                    "--ct-guard-mode",
                    "monitor",
                    "--resume-from",
                    str(non_halted),
                    "--resume-from-halted-explore",
                    "post-halt check",
                    steps=4,
                    gpu=3,
                )
            )

        # 7. RESUME_HEADER_MISMATCH on exploratory resume when a non-allowed field differs (with full JSON diff)
        with pytest.raises(SystemExit) as exc_info:
            _run(
                _argv(
                    gold,
                    out_ok,
                    *guard_flags,
                    "--ct-guard-mode",
                    "monitor",
                    "--resume-from",
                    str(halted_copy),
                    "--resume-from-halted-explore",
                    "post-halt check",
                    seed=99,
                    steps=4,
                    gpu=3,
                )
            )
        msg = str(exc_info.value)
        assert "RESUME_HEADER_MISMATCH" in msg and "train_seed" in msg
        assert '"saved": 42' in msg and '"current": 99' in msg

    def test_explore_resume_end_to_end_marks_outputs_and_leaves_original_untouched(
        self, tmp_path, monkeypatch
    ):
        gold, orig, guard_flags = self._halted_chunk(tmp_path, monkeypatch, arm="critic")
        import shutil

        orig_before = {p.relative_to(orig): p.read_bytes() for p in orig.rglob("*") if p.is_file()}
        halted_copy = tmp_path / "explore-work" / "chunk-01-halted-copy"
        shutil.copytree(orig, halted_copy)

        # Simulate code SHA change on the exploratory branch (allowed by EXPLORE_ALLOWED_HEADER_DIFFS)
        monkeypatch.setattr(s2, "aspire_commit", lambda cwd=None: "e" * 40)

        build_exp, _ = _fake_rollout_build(
            lambda i, p: [_reply("cannot_tell"), _reply("unsupported")], lambda i: [False, False]
        )
        out_exp = tmp_path / "explore-work" / "chunk-02-explore"
        reason_str = "post-halt curve shape check after S1"
        rc = _run(
            _argv(
                gold,
                out_exp,
                *guard_flags,
                "--ct-guard-mode",
                "monitor",
                "--resume-from",
                str(halted_copy),
                "--resume-from-halted-explore",
                reason_str,
                arm="critic",
                steps=4,
                gpu=3,
            ),
            build=build_exp,
        )
        assert rc == 0
        # 1. Original halted directory in run-training is completely untouched (no CONSUMED_BY.json)
        assert s2.consumed_markers(orig) == []
        orig_after = {p.relative_to(orig): p.read_bytes() for p in orig.rglob("*") if p.is_file()}
        assert orig_before == orig_after
        # 2. The copy outside run-training received CONSUMED_BY.json
        assert [m.name for m in s2.consumed_markers(halted_copy)] == ["CONSUMED_BY.json"]

        # 3. Verdict is S2-EXPLORE-CHUNK-OK and exploratory metadata is in header, receipt, eval-summary, state
        expected_exp = {
            "exploratory": True,
            "reason": reason_str,
            "original_halt": {"step": 1, "rule": "S1"},
        }
        rec = _receipt(out_exp)
        assert rec["verdict"] == "S2-EXPLORE-CHUNK-OK"
        assert rec["exploratory"] == expected_exp
        assert rec["training"]["exploratory"] == expected_exp
        hdr = json.loads((out_exp / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert hdr["exploratory"] == expected_exp
        eval_sum = json.loads((out_exp / "s2-hf-eval-summary.json").read_text(encoding="utf-8"))
        assert eval_sum["exploratory"] == expected_exp
        for sdir in s2.state_dirs(out_exp):
            st_doc = json.loads((sdir / "state.json").read_text(encoding="utf-8"))
            assert st_doc["exploratory"] == expected_exp

        # 4. Every JSONL record carries "exploratory": True
        sa_recs = s2.read_save_all(out_exp / "s2-hf-save-all.jsonl.gz")
        assert sa_recs and all(r.get("exploratory") is True for r in sa_recs)
        gh_recs = _history(out_exp)
        assert gh_recs and all(r.get("exploratory") is True for r in gh_recs)
        cr_recs = s2.read_save_all(out_exp / "s2-hf-critic-retries.jsonl.gz")
        assert cr_recs and all(r.get("exploratory") is True for r in cr_recs)
        step_recs = _rows(out_exp)
        assert step_recs and all(r.get("exploratory") is True for r in step_recs)

        # Also check chunk_stop verdict path for exploratory=True
        assert (
            s2.train_verdict_line(
                stopped=None,
                oom=False,
                next_step=2,
                steps_total=4,
                chunk_stop={"next_step": 2},
                fits=True,
                dry=True,
                exploratory=True,
            )
            == "S2-EXPLORE-CHUNK-OK"
        )

    def test_phase_and_group_loggers_tag_exploratory_records(self, tmp_path):
        import types

        fake_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            memory_allocated=lambda: 1024,
            memory_reserved=lambda: 2048,
            max_memory_allocated=lambda: 4096,
            max_memory_reserved=lambda: 8192,
            reset_peak_memory_stats=lambda: None,
        )
        p_path = tmp_path / "phases.jsonl"
        g_path = tmp_path / "groups.jsonl"
        pl = s2.PhaseLogger(p_path, enabled=True, cuda=fake_cuda, exploratory=True)
        gl = s2.GroupLogger(g_path, enabled=True, cuda=fake_cuda, exploratory=True)
        try:
            pl.record(step=0, group=0, phase="pre_rollout", prompt_tokens=10, completion_lengths=(4, 5))
            gl.record(
                step=0,
                group=0,
                prompt_id="g00",
                prompt_tokens=10,
                k_correct=1,
                g=2,
                reward_std=0.5,
                verdict_reward_std=0.5,
                truncations=0,
                mean_thinking_tokens=4.0,
                zero_variance_label=None,
            )
        finally:
            pl.close()
            gl.close()
        p_rec = json.loads(p_path.read_text(encoding="utf-8").strip())
        g_rec = json.loads(g_path.read_text(encoding="utf-8").strip())
        assert p_rec["exploratory"] is True
        assert g_rec["exploratory"] is True
