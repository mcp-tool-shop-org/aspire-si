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
        # #195: --ct-guard-mode monitor without --resume-from-halted-explore is refused.
        with pytest.raises(SystemExit, match="CT_GUARD_MONITOR_REQUIRES_EXPLORE"):
            _run(
                _argv(gold, out_mon, *guard_flags, "--ct-guard-mode", "monitor", steps=3, gpu=3),
                build=build_m,
            )
        import shutil

        halted_copy = tmp_path / "halt_copy"
        shutil.copytree(out_halt, halted_copy)
        sdir = s2.latest_state_dir(halted_copy)
        st_doc = json.loads((sdir / "state.json").read_text(encoding="utf-8"))
        st_doc["ct_guard"] = None
        (sdir / "state.json").write_text(json.dumps(st_doc, indent=1), encoding="utf-8")

        rc_m = _run(
            _argv(
                gold,
                out_mon,
                *guard_flags,
                "--ct-guard-mode",
                "monitor",
                "--resume-from",
                str(halted_copy),
                "--resume-from-halted-explore",
                "monitor test",
                steps=3,
                gpu=3,
            ),
            build=build_m,
        )
        rec_h = _receipt(out_halt)
        rec_m = _receipt(out_mon)
        assert rc_h == 2 and rec_h["verdict"] == "S2-TRAIN-HALT-S1"
        assert rc_m == 0 and rec_m["verdict"] == "S2-EXPLORE-CHUNK-OK"
        assert rec_m["stopped"] is None
        hdr_h = json.loads((out_halt / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        hdr_m = json.loads((out_mon / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert "ct_guard_mode" not in hdr_h["settings"]
        assert hdr_m["settings"]["ct_guard_mode"] == "monitor"
        assert "exploratory" not in hdr_h and hdr_m["exploratory"]["exploratory"] is True

        hist_h = [h for h in _history(out_halt) if h["kind"] == "ct_guard_window"]
        hist_m = [h for h in _history(out_mon) if h["kind"] == "ct_guard_window"]
        assert len(hist_h) == 1 and len(hist_m) == 3
        hist_m0_clean = {k: v for k, v in hist_m[0].items() if k != "exploratory"}
        assert hist_h[0] == hist_m0_clean and hist_m[0]["triggered_rules"] == ["S1"]
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
        build_h, _ = _fake_rollout_build(
            lambda i, p: [_reply("cannot_tell"), _reply("unsupported")], lambda i: [True, True]
        )
        build, _ = _fake_rollout_build(
            lambda i, p: [_reply("cannot_tell"), _reply("unsupported")], lambda i: [True, True]
        )
        out_h = tmp_path / "halt-seed"
        assert (
            _run(
                _argv(
                    gold,
                    out_h,
                    "--ct-guard",
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
                build=build_h,
            )
            == 2
        )
        out = tmp_path / "mon-trunc"
        with pytest.raises(SystemExit, match="CT_GUARD_MONITOR_REQUIRES_EXPLORE"):
            _run(
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
                "--resume-from",
                str(out_h),
                "--resume-from-halted-explore",
                "trunc in monitor mode",
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


# -- S2 v1.1 Pre-Registration, Matchers, Constant Loss Denom, CUSUM + Confirm + Canaries --
import dataclasses
import hashlib
import os

import quote_matcher_m1 as qm1  # noqa: E402
import quote_matcher_m2 as qm2  # noqa: E402
import s2_draw  # noqa: E402
import s2_v11_config as v11cfg  # noqa: E402


def _make_synthetic_v11_prepass(tmp_path, gold_dir):
    """Write synthetic Pass A and Pass B JSONL files for the 60-item _gold fixture."""
    items, _ = s2.read_gold(gold_dir)
    pass_a_rows = []
    pass_b_rows = []
    for idx, it in enumerate(items):
        cid = it["id"]
        label = it["label"]
        # Give the first 10 unsupported training items k=6, ct=0 in both passes, and the rest k=5
        is_easy = (label == "unsupported") and (idx % 2 == 0)
        pass_a_rows.append(
            {
                "id": cid,
                "gold_verdict": label,
                "k_correct": 6 if is_easy else 5,
                "cannot_tell": 0,
                "g": 8,
                "seed": 0,
                "descriptive": False,
            }
        )
        pass_b_rows.append(
            {
                "id": cid,
                "gold_verdict": label,
                "k_correct": 6 if is_easy else 5,
                "cannot_tell": 0,
                "g": 8,
                "seed": 1,
                "descriptive": False,
            }
        )
    pass_a_path = tmp_path / "syn_pass_a.jsonl"
    pass_b_path = tmp_path / "syn_pass_b.jsonl"
    pass_a_path.write_text("".join(json.dumps(r) + "\n" for r in pass_a_rows), encoding="utf-8", newline="\n")
    pass_b_path.write_text("".join(json.dumps(r) + "\n" for r in pass_b_rows), encoding="utf-8", newline="\n")
    pass_a_sha = hashlib.sha256(pass_a_path.read_bytes()).hexdigest()
    pass_b_sha = hashlib.sha256(pass_b_path.read_bytes()).hexdigest()
    train_pool, holdout_pool, _ = s2_draw.split_stratified_holdout(
        items, (pass_a_path, pass_a_sha), holdout_fraction=0.20, holdout_seed="pr79-holdout-v1"
    )
    holdout_ids = [r["id"] for r in holdout_pool]
    sets = v11cfg.build_v11_canary_sets(
        train_pool, pass_a_rows, pass_b_rows, holdout_ids=holdout_ids, strict=False
    )
    syn_cfg = dataclasses.replace(
        v11cfg.S2V11,
        pass_a_sha256=pass_a_sha,
        pass_b_sha256=pass_b_sha,
        canary_c_size=sets["canary_c_count"],
        canary_c_required_item=sets["canary_c_ids"][0],
        canary_c_ids_sha256=sets["canary_c_ids_sha256"],
        gold_unsupported_train_items=sets["gold_unsupported_count"],
        gold_unsupported_ids_sha256=sets["gold_unsupported_ids_sha256"],
    )
    return pass_a_path, pass_a_sha, pass_b_path, pass_b_sha, syn_cfg, sets


class TestS2V11PreregAndConfig:
    def test_sealed_prereg_sha256_and_config_parser_match(self):
        repo_root = Path(__file__).resolve().parents[1]
        prereg_path = repo_root / v11cfg.PREREG_V11_RELPATH
        assert prereg_path.is_file()
        raw_bytes = prereg_path.read_bytes()
        assert hashlib.sha256(raw_bytes).hexdigest() == v11cfg.PREREG_V11_SHA256
        text = v11cfg.load_sealed_prereg_v11_text()
        parsed = v11cfg.parse_sealed_prereg_v11(text)
        cfg_dict = v11cfg.s2_v11_config_dict()
        assert parsed == cfg_dict
        assert len(v11cfg.s2_v11_config_sha256()) == 64
        assert not hasattr(v11cfg, "PASS_A_PASS_B_DIFF_EXCLUDED_IDS_V11")

        with pytest.raises(ValueError, match="Pattern not found in sealed prereg text"):
            v11cfg.parse_sealed_prereg_v11("incomplete text")
        with pytest.raises(ValueError, match="S2_V11_PREREG_SHA256_MISMATCH"):
            v11cfg.load_sealed_prereg_v11_text(prereg_path, expected_sha256="0" * 64)

    def test_build_v11_canary_sets_synthetic_every_branch_of_rule(self, tmp_path):
        # Ask 4(a): synthetic rows covering every branch of the sealed §3 rule:
        # - k = 5 vs 6 in Pass A
        # - k = 5 vs 6 in Pass B
        # - CT = 1 in Pass A
        # - CT = 1 in Pass B
        # - a holdout item
        # - a supported item
        items = [
            {"id": "c_clean_6_6", "label": "unsupported"},
            {"id": "diff-rnd-e839712-2u", "label": "unsupported"},
            {"id": "u_pass_a_k5", "label": "unsupported"},
            {"id": "u_pass_b_k5", "label": "unsupported"},
            {"id": "u_pass_a_ct1", "label": "unsupported"},
            {"id": "u_pass_b_ct1", "label": "unsupported"},
            {"id": "h_holdout_unsup", "label": "unsupported"},
            {"id": "s_supported_6_6", "label": "supported"},
        ]
        pass_a = [
            {"id": "c_clean_6_6", "gold_verdict": "unsupported", "k_correct": 6, "cannot_tell": 0},
            {"id": "diff-rnd-e839712-2u", "gold_verdict": "unsupported", "k_correct": 8, "cannot_tell": 0},
            {"id": "u_pass_a_k5", "gold_verdict": "unsupported", "k_correct": 5, "cannot_tell": 0},
            {"id": "u_pass_b_k5", "gold_verdict": "unsupported", "k_correct": 6, "cannot_tell": 0},
            {"id": "u_pass_a_ct1", "gold_verdict": "unsupported", "k_correct": 6, "cannot_tell": 1},
            {"id": "u_pass_b_ct1", "gold_verdict": "unsupported", "k_correct": 6, "cannot_tell": 0},
            {"id": "h_holdout_unsup", "gold_verdict": "unsupported", "k_correct": 8, "cannot_tell": 0},
            {"id": "s_supported_6_6", "gold_verdict": "supported", "k_correct": 8, "cannot_tell": 0},
        ]
        pass_b = [
            {"id": "c_clean_6_6", "gold_verdict": "unsupported", "k_correct": 6, "cannot_tell": 0},
            {"id": "diff-rnd-e839712-2u", "gold_verdict": "unsupported", "k_correct": 6, "cannot_tell": 0},
            {"id": "u_pass_a_k5", "gold_verdict": "unsupported", "k_correct": 6, "cannot_tell": 0},
            {"id": "u_pass_b_k5", "gold_verdict": "unsupported", "k_correct": 5, "cannot_tell": 0},
            {"id": "u_pass_a_ct1", "gold_verdict": "unsupported", "k_correct": 6, "cannot_tell": 0},
            {"id": "u_pass_b_ct1", "gold_verdict": "unsupported", "k_correct": 6, "cannot_tell": 1},
            {"id": "h_holdout_unsup", "gold_verdict": "unsupported", "k_correct": 8, "cannot_tell": 0},
            {"id": "s_supported_6_6", "gold_verdict": "supported", "k_correct": 8, "cannot_tell": 0},
        ]
        sets = v11cfg.build_v11_canary_sets(
            items,
            pass_a,
            pass_b,
            holdout_ids=["h_holdout_unsup"],
            strict=False,
        )
        # Only the 2 items with label=='unsupported', not in holdout, A.k>=6, B.k>=6, A.ct==0, B.ct==0 are in C
        assert sets["canary_c_ids"] == ["c_clean_6_6", "diff-rnd-e839712-2u"]
        assert sets["canary_c_count"] == 2
        # All 6 non-holdout unsupported items are in gold_unsupported_ids (holdout & supported excluded)
        assert sets["gold_unsupported_ids"] == [
            "c_clean_6_6",
            "diff-rnd-e839712-2u",
            "u_pass_a_ct1",
            "u_pass_a_k5",
            "u_pass_b_ct1",
            "u_pass_b_k5",
        ]
        assert sets["gold_unsupported_count"] == 6

        # Also works when items=None (using gold_verdict from pass_a) and when pass_a/pass_b are dicts
        sets_from_dicts = v11cfg.build_v11_canary_sets(
            None,
            {r["id"]: r for r in pass_a},
            {r["id"]: r for r in pass_b},
            holdout_ids=["h_holdout_unsup"],
            strict=False,
        )
        assert sets_from_dicts == sets

        # Verify strict=True validation and all named refusal codes on mismatch
        syn_cfg = dataclasses.replace(
            v11cfg.S2V11,
            canary_c_size=2,
            canary_c_required_item="diff-rnd-e839712-2u",
            canary_c_ids_sha256=sets["canary_c_ids_sha256"],
            gold_unsupported_train_items=6,
            gold_unsupported_ids_sha256=sets["gold_unsupported_ids_sha256"],
        )
        assert (
            v11cfg.build_v11_canary_sets(
                items, pass_a, pass_b, holdout_ids=["h_holdout_unsup"], cfg=syn_cfg, strict=True
            )
            == sets
        )
        with pytest.raises(ValueError, match="S2_V11_GOLD_UNSUPPORTED_SIZE_MISMATCH"):
            v11cfg.validate_v11_canary_sets(
                sets, cfg=dataclasses.replace(syn_cfg, gold_unsupported_train_items=184)
            )
        with pytest.raises(ValueError, match="S2_V11_GOLD_UNSUPPORTED_SHA256_MISMATCH"):
            v11cfg.validate_v11_canary_sets(
                sets, cfg=dataclasses.replace(syn_cfg, gold_unsupported_ids_sha256="0" * 64)
            )
        with pytest.raises(ValueError, match="S2_V11_CANARY_C_SIZE_MISMATCH"):
            v11cfg.validate_v11_canary_sets(sets, cfg=dataclasses.replace(syn_cfg, canary_c_size=128))
        with pytest.raises(ValueError, match="S2_V11_CANARY_C_MISSING_REQUIRED_ITEM"):
            v11cfg.validate_v11_canary_sets(
                sets, cfg=dataclasses.replace(syn_cfg, canary_c_required_item="missing-id")
            )
        with pytest.raises(ValueError, match="S2_V11_CANARY_C_SHA256_MISMATCH"):
            v11cfg.validate_v11_canary_sets(
                sets, cfg=dataclasses.replace(syn_cfg, canary_c_ids_sha256="0" * 64)
            )
        with pytest.raises(ValueError, match="S2_V11_PASS_A_MISSING_ID"):
            v11cfg.build_v11_canary_sets(items, pass_a[1:], pass_b)
        with pytest.raises(ValueError, match="S2_V11_PASS_B_MISSING_ID"):
            v11cfg.build_v11_canary_sets(items, pass_a, pass_b[1:])
        with pytest.raises(ValueError, match="S2_V11_PASS_A_INVALID_RECORD"):
            v11cfg.build_v11_canary_sets(items, [{"id": "c_clean_6_6"}], pass_b)
        with pytest.raises(ValueError, match="S2_V11_PASS_A_INVALID_RECORD"):
            v11cfg.build_v11_canary_sets(items, {"c_clean_6_6": {" id": "bad"}}, pass_b)
        with pytest.raises(ValueError, match="S2_V11_PASS_A_DUPLICATE_ID"):
            v11cfg.build_v11_canary_sets(items, [pass_a[0], pass_a[0]], pass_b)

    def test_local_data_derives_c_128_and_shas_from_real_prepass_files(self):
        # Ask 4(b): local-data test deriving C from the real pre-pass files and asserting |C| == 128 + SHAs.
        # Skipped in CI unless ASPIRE_S2_V11_PREPASS_DIR or (ASPIRE_S2_PASS_A_JSONL + ASPIRE_S2_PASS_B_JSONL) is set.
        prepass_dir = os.environ.get("ASPIRE_S2_V11_PREPASS_DIR", "").strip()
        pass_a_env = os.environ.get("ASPIRE_S2_PASS_A_JSONL", "").strip()
        pass_b_env = os.environ.get("ASPIRE_S2_PASS_B_JSONL", "").strip()
        holdout_env = os.environ.get("ASPIRE_S2_HOLDOUT_IDS", "").strip()

        if prepass_dir:
            base = Path(prepass_dir)
            pass_a_path = Path(pass_a_env) if pass_a_env else base / "run-prepass" / "s2-hf-prepass.jsonl"
            pass_b_path = Path(pass_b_env) if pass_b_env else base / "run-prepass-b2" / "s2-hf-prepass.jsonl"
            holdout_path = Path(holdout_env) if holdout_env else base / "holdout-ids-pr79-holdout-v1.txt"
        elif pass_a_env and pass_b_env:
            pass_a_path = Path(pass_a_env)
            pass_b_path = Path(pass_b_env)
            holdout_path = Path(holdout_env) if holdout_env else None
        else:
            pytest.skip(
                "Skipped in CI (pre-pass data and holdout list stay private); "
                "set ASPIRE_S2_V11_PREPASS_DIR or ASPIRE_S2_PASS_A_JSONL + ASPIRE_S2_PASS_B_JSONL to run locally."
            )

        rec_a, sha_a = v11cfg.load_prepass_records_v11(
            pass_a_path,
            v11cfg.PASS_A_SHA256_V11,
            pass_label="PASS_A",
            pinned_sha256=v11cfg.S2V11.pass_a_sha256,
        )
        rec_b, sha_b = v11cfg.load_prepass_records_v11(
            pass_b_path,
            v11cfg.PASS_B_SHA256_V11,
            pass_label="PASS_B",
            pinned_sha256=v11cfg.S2V11.pass_b_sha256,
        )
        assert sha_a == "feee81ff8897cda5824c4877ce597232d7291e2f80a32d3a16a56a9c06aafa69"
        assert sha_b == "d77601e5d7a75842de4c01aafef4fc5f8b1e939e1a24cb26b6bf00f1c142fbd7"
        assert len(rec_a) == 493
        assert len(rec_b) == 493

        if holdout_path is not None and holdout_path.is_file():
            hold_raw = holdout_path.read_bytes()
            assert (
                hashlib.sha256(hold_raw).hexdigest()
                == "00eaa49006099381328e2d7becceb9ed102b46ce79b2ac2823dc3c0247d16ff4"
            )
            holdout_ids = [ln.strip() for ln in hold_raw.decode("utf-8").splitlines() if ln.strip()]
        else:
            items_for_split = [
                {"id": cid, "label": r["gold_verdict"], "split": "tune"} for cid, r in rec_a.items()
            ]
            _, holdout_pool, _ = s2_draw.split_stratified_holdout(
                items_for_split,
                (pass_a_path, sha_a),
                holdout_fraction=0.20,
                holdout_seed="pr79-holdout-v1",
            )
            holdout_ids = [r["id"] for r in holdout_pool]
        assert len(holdout_ids) == 99

        sets = v11cfg.build_v11_canary_sets(
            None,
            rec_a,
            rec_b,
            holdout_ids=holdout_ids,
            cfg=v11cfg.S2V11,
            strict=True,
        )
        assert sets["canary_c_count"] == 128
        assert len(sets["canary_c_ids"]) == 128
        assert "diff-rnd-e839712-2u" in sets["canary_c_ids"]
        assert (
            sets["canary_c_ids_sha256"] == "c15c8a4cd323ebd0e605c6a166e42036a3ec87d2248a4c4c2a00d8ae658c4bca"
        )
        assert sets["gold_unsupported_count"] == 184
        assert len(sets["gold_unsupported_ids"]) == 184
        assert (
            sets["gold_unsupported_ids_sha256"]
            == "63e6656699395531d9814c9cd65c1c18cbbe5aa8c97614137d50bc6ab066f27e"
        )
        print(
            json.dumps(
                {
                    "test": "test_local_data_derives_c_128_and_shas_from_real_prepass_files",
                    "pass_a_sha256": sha_a,
                    "pass_b_sha256": sha_b,
                    "holdout_count": len(holdout_ids),
                    "canary_c_count": sets["canary_c_count"],
                    "canary_c_ids_sha256": sets["canary_c_ids_sha256"],
                    "diff_rnd_e839712_2u_in_c": "diff-rnd-e839712-2u" in sets["canary_c_ids"],
                    "gold_unsupported_count": sets["gold_unsupported_count"],
                    "gold_unsupported_ids_sha256": sets["gold_unsupported_ids_sha256"],
                }
            )
        )

    def test_s2_v11_refuses_wrong_pass_b_sha_and_missing_prepass(self, tmp_path):
        # Ask 4(c): refusal test for a wrong Pass-B SHA (both CLI argument mismatch and on-disk SHA mismatch)
        gold = _gold(tmp_path)
        out = tmp_path / "v11_refuse"
        pass_a_file = tmp_path / "pass_a.jsonl"
        pass_b_file = tmp_path / "pass_b.jsonl"
        row_txt = (
            json.dumps({"id": "g00", "gold_verdict": "unsupported", "k_correct": 6, "cannot_tell": 0}) + "\n"
        )
        pass_a_file.write_text(row_txt, encoding="utf-8")
        pass_b_file.write_text(row_txt, encoding="utf-8")

        base = [
            "--gold",
            str(gold),
            "--rnd-commit",
            "e48cc90c",
            "--out",
            str(out),
            "--dry",
            "--s2-v11",
            "--arm",
            "mech",
            "--train-seed",
            "45",
        ]

        # 1. Wrong --pass-b-sha256 CLI value vs pinned S2V11.pass_b_sha256 -> refuses with S2_V11_PASS_B_SHA256_MISMATCH
        ns_bad_b = _ns(
            [
                *base,
                "--pass-a-jsonl",
                str(pass_a_file),
                "--pass-a-sha256",
                v11cfg.PASS_A_SHA256_V11,
                "--pass-b-jsonl",
                str(pass_b_file),
                "--pass-b-sha256",
                "0" * 64,
            ]
        )
        with pytest.raises(SystemExit, match="S2_V11_PASS_B_SHA256_MISMATCH"):
            s2.validate_s2_v11_args(ns_bad_b)

        # 2. On-disk Pass-B file SHA-256 does not match pinned S2V11.pass_b_sha256 -> refuses with S2_V11_PASS_B_SHA256_MISMATCH
        actual_b_sha = hashlib.sha256(pass_b_file.read_bytes()).hexdigest()
        with pytest.raises(ValueError, match="S2_V11_PASS_B_SHA256_MISMATCH"):
            v11cfg.load_prepass_records_v11(
                pass_b_file,
                v11cfg.PASS_B_SHA256_V11,
                pass_label="PASS_B",
                pinned_sha256=v11cfg.S2V11.pass_b_sha256,
            )
        with pytest.raises(ValueError, match="S2_V11_PASS_B_SHA256_MISMATCH"):
            v11cfg.load_prepass_records_v11(
                pass_b_file,
                actual_b_sha,
                pass_label="PASS_B",
                pinned_sha256=v11cfg.S2V11.pass_b_sha256,
            )
        with pytest.raises(ValueError, match="S2_V11_PASS_B_SHA256_MISMATCH"):
            v11cfg.load_prepass_records_v11(
                pass_b_file,
                None,
                pass_label="PASS_B",
                pinned_sha256=v11cfg.S2V11.pass_b_sha256,
            )
        blank_line_file = tmp_path / "blank_line.jsonl"
        blank_line_file.write_text("\n" + row_txt + "\n", encoding="utf-8")
        recs_bl, _ = v11cfg.load_prepass_records_v11(blank_line_file, pass_label="PASS_B")
        assert "g00" in recs_bl

        # 3. Missing Pass B or Pass A under --s2-v11, or wrong Pass A SHA
        ns_missing_b = _ns(
            [
                *base,
                "--pass-a-jsonl",
                str(pass_a_file),
                "--pass-a-sha256",
                v11cfg.PASS_A_SHA256_V11,
            ]
        )
        with pytest.raises(SystemExit, match="S2_V11_PASS_B_REQUIRED"):
            s2.validate_s2_v11_args(ns_missing_b)

        ns_missing_a = _ns(
            [
                *base,
                "--pass-b-jsonl",
                str(pass_b_file),
                "--pass-b-sha256",
                v11cfg.PASS_B_SHA256_V11,
            ]
        )
        with pytest.raises(SystemExit, match="S2_V11_PASS_A_REQUIRED"):
            s2.validate_s2_v11_args(ns_missing_a)

        ns_bad_a = _ns(
            [
                *base,
                "--pass-a-jsonl",
                str(pass_a_file),
                "--pass-a-sha256",
                "f" * 64,
                "--pass-b-jsonl",
                str(pass_b_file),
                "--pass-b-sha256",
                v11cfg.PASS_B_SHA256_V11,
            ]
        )
        with pytest.raises(SystemExit, match="S2_V11_PASS_A_SHA256_MISMATCH"):
            s2.validate_s2_v11_args(ns_bad_a)

        # 4. File-level error branches in load_prepass_records_v11
        with pytest.raises(ValueError, match="S2_V11_PASS_B_MISSING_FILE"):
            v11cfg.load_prepass_records_v11(tmp_path / "nope.jsonl", pass_label="PASS_B")
        with pytest.raises(ValueError, match="S2_V11_PASS_B_REQUIRED"):
            v11cfg.load_prepass_records_v11(pass_b_file, "", pass_label="PASS_B")
        corrupt = tmp_path / "corrupt.jsonl"
        corrupt.write_text("{not-json\n", encoding="utf-8")
        with pytest.raises(ValueError, match="S2_V11_PASS_B_CORRUPT_JSON"):
            v11cfg.load_prepass_records_v11(corrupt, pass_label="PASS_B")
        bad_rec = tmp_path / "bad_rec.jsonl"
        bad_rec.write_text(json.dumps({"id": "g00"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="S2_V11_PASS_B_INVALID_RECORD"):
            v11cfg.load_prepass_records_v11(bad_rec, pass_label="PASS_B")
        dup_rec = tmp_path / "dup_rec.jsonl"
        dup_rec.write_text(row_txt + row_txt, encoding="utf-8")
        with pytest.raises(ValueError, match="S2_V11_PASS_B_DUPLICATE_ID"):
            v11cfg.load_prepass_records_v11(dup_rec, pass_label="PASS_B")
        bad_k = tmp_path / "bad_k.jsonl"
        bad_k.write_text(
            json.dumps({"id": "g00", "k_correct": -1, "cannot_tell": 0}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="S2_V11_PASS_B_INVALID_RECORD"):
            v11cfg.load_prepass_records_v11(bad_k, pass_label="PASS_B")
        bad_ct = tmp_path / "bad_ct.jsonl"
        bad_ct.write_text(
            json.dumps({"id": "g00", "k_correct": 6, "cannot_tell": True}) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="S2_V11_PASS_B_INVALID_RECORD"):
            v11cfg.load_prepass_records_v11(bad_ct, pass_label="PASS_B")

    def test_s2_v11_flag_freezes_settings_and_refuses_overrides(self, tmp_path):
        gold = _gold(tmp_path)
        out = tmp_path / "v11_check"
        pass_a_path = tmp_path / "pa.jsonl"
        pass_b_path = tmp_path / "pb.jsonl"
        pass_a_path.write_text("", encoding="utf-8")
        pass_b_path.write_text("", encoding="utf-8")
        base = [
            "--gold",
            str(gold),
            "--rnd-commit",
            "e48cc90c",
            "--out",
            str(out),
            "--dry",
            "--s2-v11",
            "--pass-a-jsonl",
            str(pass_a_path),
            "--pass-a-sha256",
            v11cfg.PASS_A_SHA256_V11,
            "--pass-b-jsonl",
            str(pass_b_path),
            "--pass-b-sha256",
            v11cfg.PASS_B_SHA256_V11,
        ]

        # Valid --s2-v11 sets frozen parameters
        a = _ns([*base, "--arm", "mech", "--train-seed", "45"])
        assert s2.validate_training_args(a) is True
        assert a.ct_guard is True
        assert a.ct_guard_mode == "halt"
        assert a.guard_window_groups == 40
        assert a.groups_per_update == 4
        assert a.quote_reward == "v2"
        assert a.advantage == "dr-grpo"
        assert a.mask_wrong_positive_advantage is True
        assert a.zero_variance_mask is True
        assert a.mask_truncated is True
        assert a.trunc_halt_rate == 0.20
        assert a.trunc_window_updates == 10
        assert a.holdout is True
        assert a.holdout_fraction == 0.20
        assert a.holdout_seed == "pr79-holdout-v1"
        assert a.draw == "class_stratified"
        assert a.draw_seed == 45
        assert a.steps == 197
        flags = s2.active_flags(a)
        assert flags["s2_v11"] is True
        assert flags["prereg_version"] == "1.1"
        assert flags["prereg_sha256"] == v11cfg.PREREG_V11_SHA256
        assert flags["quote_matcher"] == "M2a"
        assert flags["constant_loss_denominator"] == 4514
        assert flags["pass_a_sha256"] == v11cfg.PASS_A_SHA256_V11
        assert flags["pass_b_sha256"] == v11cfg.PASS_B_SHA256_V11

        # Permitted seeds 45, 46, 47 are accepted
        for valid_seed in (45, 46, 47):
            ns_ok = _ns([*base, "--arm", "mech", "--train-seed", str(valid_seed)])
            assert s2.validate_s2_v11_args(ns_ok) is True
            assert ns_ok.draw_seed == valid_seed

        # Retired seeds 42, 43, 44 are refused and named in the refusal message
        for ret_seed in (42, 43, 44):
            ns_ret = _ns([*base, "--arm", "mech", "--train-seed", str(ret_seed)])
            with pytest.raises(
                SystemExit,
                match=r"S2_V11_OVERRIDE_FORBIDDEN.*retired seeds 42, 43, 44",
            ):
                s2.validate_s2_v11_args(ns_ret)

        # Refusal cases under --s2-v11
        bad_cases = [
            (["--ct-guard-mode", "monitor"], "S2_V11_MONITOR_MODE_FORBIDDEN"),
            (["--resume-from-halted-explore", "why"], "S2_V11_MONITOR_MODE_FORBIDDEN"),
            (["--guard-window-groups", "10"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--trunc-halt-rate", "0.15"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--trunc-window-updates", "5"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--dual-clip-c", "2.0"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--groups-per-update", "2"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--quote-reward", "v1"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--advantage", "std"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--no-save-all", "--save-all-off-reason", "test"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--holdout-fraction", "0.10"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--holdout-seed", "other-seed"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--draw", "first"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--draw", "uniform"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--draw-seed", "99"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--steps", "30"], "S2_V11_OVERRIDE_FORBIDDEN"),
            (["--steps", "2"], "S2_V11_OVERRIDE_FORBIDDEN"),
        ]
        for extra_flags, code in bad_cases:
            ns = _ns([*base, "--arm", "mech", "--train-seed", "45", *extra_flags])
            with pytest.raises(SystemExit, match=code):
                s2.validate_s2_v11_args(ns)


class TestS2V11QuoteMatchersAndLoss:
    def test_quote_matcher_files_sha256_and_m1_m2a_m2b_behavior(self):
        m1_path = Path(qm1.__file__)
        m2_path = Path(qm2.__file__)
        assert (
            hashlib.sha256(m1_path.read_bytes()).hexdigest()
            == "57064f46ff0f39ec26522e3d3d52ccb4c7f96300729f9a79052e05dfc842b42c"
        )
        assert (
            hashlib.sha256(m2_path.read_bytes()).hexdigest()
            == "98780f92893e0e3ad4e4f9394e1c81a2f7e7f94a8b3331aed3ee217114d0ea29"
        )

        # M2a strips spaces adjacent to code punctuation (e.g. "foo ( x , y ) : return x + y")
        passages = [
            "def compute_score(items, weights):\n    total = sum(x * w for x, w in zip(items, weights))\n    return total / len(items)"
        ]
        q_punct_ws = "def compute_score ( items , weights ) :\n    total = sum ( x * w for x , w in zip ( items , weights ) )"
        # M1 fails because of spaces around parentheses/commas/colons
        assert qm1.quote_found_m1(q_punct_ws, passages) is False
        assert qm1.quote_found_m1("", passages) is False
        # M2a strips spaces around code punctuation outside string literals, so it matches!
        assert qm2.quote_found_m2a(q_punct_ws, passages) is True
        assert qm2.quote_found_m2a("short", passages) is False
        assert qm2.quote_credit_m2a(
            [q_punct_ws], passages, verdict="supported", correct=True
        ) == pytest.approx(0.05)
        assert qm2.quote_credit_m2a([], passages, verdict="supported", correct=True) == 0.0
        assert qm2.quote_credit_m2a([q_punct_ws], passages, verdict="supported", correct=False) == 0.0

        # M2b also unifies single and double quotes after M2a canonical form
        passages_q = ['def check_flag(mode):\n    return mode == "strict_mode" and len(mode) > 0']
        q_m2b = "def check_flag(mode): ... return mode == 'strict_mode' and len(mode) > 0"
        assert qm2.quote_found_m2b(q_m2b, passages_q) is True
        assert qm2.quote_found_m2b("only one short ... piece", passages_q) is False
        assert qm2.quote_credit_m2b([q_m2b], passages_q, verdict="supported", correct=True) == pytest.approx(
            0.05
        )
        assert qm2.quote_credit_m2b([], passages_q, verdict="supported", correct=True) == 0.0
        assert qm2.quote_credit_m2b([q_m2b], passages_q, verdict="supported", correct=False) == 0.0

        # Exercise M1/M2a/M2b tiers: Tier 0, Tier 1, Tier 2 (with negation guard & para break), and short quotes
        p_multi = [
            "+ // 1.5% rate >= -3 and < 10\n"
            "+ first fragment has enough words here and does not fail\n"
            "+ middle words without negation\n"
            "+ second fragment also has enough words here\n\n"
            "+ after paragraph break words here for test\n"
            "+ FooBar 42 is unique\n"
            "+ is FooBar now\n"
            "+ shortline99"
        ]
        # Tier 0
        assert qm1.quote_tier_m1("first fragment has enough words here and does not fail", p_multi) == 0
        # Tier 1 (curly quotes / dash / outer punct / protected edge chars)
        assert (
            qm1.quote_tier_m1(
                "“first fragment has enough words here and does not fail”...",
                p_multi,
            )
            == 1
        )
        assert qm1._strip_quote_outer_punct("...1.5%...") == "1.5%"
        assert qm1._strip_quote_outer_punct("...-3...") == "-3"
        assert qm1._strip_quote_outer_punct("...1.5") == "1.5"
        assert qm1._is_protected_edge_char("1.5", 1) is True
        assert qm1._strip_quote_outer_punct("...abc...") == "abc"
        assert qm1._extract_negation_tokens("it can't and won't fail") == {"n't", "cannot"}
        assert qm1._has_internal_capitalised_token("...") is False
        # Tier 2 ellipsis (valid vs negation blocked vs para-break blocked vs max_gap exceeded)
        q_t2_ok = "first fragment has enough words ... second fragment also has enough words"
        assert qm1.quote_tier_m1(q_t2_ok, p_multi) == 2
        assert len(qm1.find_quote_spans_m1(q_t2_ok, p_multi)) == 1
        assert qm1.quote_match_tier_m1(q_t2_ok, p_multi, _max_gap=1)[0] is False
        q_t2_para = "second fragment also has enough words ... after paragraph break words here"
        assert qm1.quote_tier_m1(q_t2_para, p_multi) is None
        p_neg = ["alpha beta gamma delta cannot epsilon zeta eta theta"]
        assert qm1.quote_tier_m1("alpha beta gamma ... epsilon zeta eta", p_neg) is None
        p_neg_retry = ["alpha beta gamma cannot epsilon zeta eta alpha beta gamma ok epsilon zeta eta"]
        assert qm1.quote_tier_m1("alpha beta gamma ... epsilon zeta eta", p_neg_retry) == 2
        # Short rule (a) whole line (with outer punct so Tier 0 misses) and (b) 2-4 words < 12 chars with digit or internal capitalised token
        assert qm1.quote_tier_m1("...shortline99...", p_multi) == "short"
        assert qm1.quote_tier_m1("FooBar 42", p_multi) == "short"
        assert qm1.quote_tier_m1("is FooBar", p_multi) == "short"
        assert qm1.quote_tier_m1("no match 99", p_multi) is None
        assert len(qm1.find_quote_spans_m1("FooBar 42", p_multi)) >= 1
        assert qm1.find_quote_spans_m1("not found anywhere at all", p_multi) == []
        assert qm1.quote_credit_m1(
            [
                "",
                "first fragment has enough words here and does not fail",
                "first fragment has enough words here and does not fail",
                "not found anywhere at all",
                "1.5%",
                "FooBar 42",
            ],
            p_multi,
            verdict="supported",
            correct=True,
        ) == pytest.approx(0.10)
        assert (
            qm1.quote_credit_m1(
                ["FooBar 42"],
                p_multi,
                verdict="unsupported",
                correct=True,
            )
            == 0.0
        )
        assert qm1.quote_credit_m1(["x"], p_multi, verdict="supported", correct=False) == 0.0

        # M2a / M2b tiers, literal spans with escapes, digit-flanked dots/commas, and quote_tier_m2
        lit_s = '\\x x = "a \\\' ( b ) " + \'c " [ d ] \' + "unclosed\n" + 1 . 5 + 2 , 000'
        assert len(qm2.find_same_line_literal_spans(lit_s)) == 2
        assert qm2.find_same_line_literal_spans('"unclosed at eof') == []
        assert qm2._strip_punct_ws_outside_literals("1 . 5", digit_flank_exempt=False) == "1.5"
        norm_lit = qm2.normalise_m2a(lit_s, is_quote=True)
        assert "( b )" in norm_lit  # preserved inside string literal
        assert "1 . 5" in norm_lit  # digit-flanked dot preserved
        assert qm2.quote_tier_m2(q_punct_ws, passages, arm="m2a") == "m2a_1"
        assert qm2.quote_tier_m2(q_punct_ws, passages, arm="m2b") == "m2a_1"
        assert qm2.quote_tier_m2(q_m2b, passages_q, arm="m2b") == "m2b_2"
        assert qm2.quote_tier_m2b("", passages_q) is None
        with pytest.raises(ValueError, match="Unknown arm"):
            qm2.quote_tier_m2("x", passages, arm="bad")
        p_m2_short = ["fn Call(x)\nother line with Call(x) in it\nUniqueFn(9)"]
        assert qm2.quote_tier_m2a("fn Call ( x )", p_m2_short) == "m2a_short"
        assert qm2.quote_tier_m2a("UniqueFn ( 9 )", p_m2_short) == "m2a_short"
        assert qm2.quote_tier_m2a("MissingFn ( 9 )", p_m2_short) is None
        assert len(qm2.find_quote_spans_m2a("fn Call ( x )", p_m2_short)) >= 1
        assert (
            len(qm2.find_quote_spans_m2a("first fragment has enough words here and does not fail", p_multi))
            == 1
        )
        assert qm2.find_quote_spans_m2a("definitely not found", p_m2_short) == []
        assert len(qm2.find_quote_spans_m2b(q_m2b, passages_q)) == 1
        assert qm2.quote_credit_m2a(
            ["", q_punct_ws, q_punct_ws, "not found", "a ( 1 )", "UniqueFn ( 9 )"],
            passages + p_m2_short,
            verdict="unsupported",
            correct=True,
        ) == pytest.approx(0.05)

        # locate_raw_span_m2a maps normalized match back to raw passage offsets for anti-farming
        span1 = v11cfg.locate_raw_span_m2a(q_punct_ws, passages)
        assert span1 is not None and span1[0] == 0 and span1[1] < span1[2]
        assert v11cfg.locate_raw_span_m2a("", passages) is None
        assert v11cfg.locate_raw_span_m2a("short", passages) is None
        assert v11cfg.locate_raw_span_m2a("definitely not in passage anywhere at all", passages) is None

        # Also exercise tier0, tier1, tier2 ellipsis, and short-quote branches of locate_raw_span_m2a and canary helpers
        p_hyph = [
            "+ // word-\n  continuation in a long enough line here\r\n\r\n+ second para line here with words"
        ]
        assert v11cfg.locate_raw_span_m2a("wordcontinuation in a long enough line here", p_hyph) is not None
        assert (
            v11cfg.locate_raw_span_m2a(
                "wordcontinuation in a long ( enough ) line here",
                ["word-\ncontinuation in a long(enough) line here"],
            )
            is not None
        )
        assert (
            v11cfg.locate_raw_span_m2a(
                "quote with 'literal ( inside )' and 1 . 5", ["quote with 'literal ( inside )' and 1 . 5"]
            )
            is not None
        )
        q_exact = "def compute_score(items, weights):"
        span_t0 = v11cfg.locate_raw_span_m2a(q_exact, passages)
        assert span_t0 is not None and span_t0[3] == "0"
        span_t0_unm = v11cfg.locate_raw_span_m2a(
            "first fragment has enough words here and does not fail", p_multi
        )
        assert span_t0_unm is not None and span_t0_unm[3] == "0"
        span_t0_short = v11cfg.locate_raw_span_m2a("shortline99", p_multi)
        assert span_t0_short is not None and span_t0_short[3] == "0"
        span_t0_short_raw = v11cfg.locate_raw_span_m2a("rawshort99", ["rawshort99\nother line"])
        assert span_t0_short_raw is not None and span_t0_short_raw[3] == "0"
        span_t1 = v11cfg.locate_raw_span_m2a(
            "“first fragment has enough words here and does not fail”", p_multi
        )
        assert span_t1 is not None and span_t1[3] == "1"
        span_t2_m1 = v11cfg.locate_raw_span_m2a(q_t2_ok, p_multi)
        assert span_t2_m1 is not None and span_t2_m1[3] == "2"
        q_ell = "def compute_score ( items , weights ) ... return total / len ( items )"
        span_t2 = v11cfg.locate_raw_span_m2a(q_ell, passages)
        assert span_t2 is not None and span_t2[3] == "m2a_2"
        span_m2a_sh_line = v11cfg.locate_raw_span_m2a("fn Call ( x )", p_m2_short)
        assert span_m2a_sh_line is not None and span_m2a_sh_line[3] == "m2a_short"
        span_m2a_sh_unm = v11cfg.locate_raw_span_m2a("fn Call ( x )", ["+ fn Call(x)\n+ other line"])
        assert span_m2a_sh_unm is not None and span_m2a_sh_unm[3] == "m2a_short"
        p_short = ["+ Err (E409) in log\n+ Another line here with more words"]
        span_sh = v11cfg.locate_raw_span_m2a("Err ( E409 )", p_short)
        assert span_sh is not None and span_sh[3] == "m2a_short"

        # Canary helper edge cases (table/para boundary, sign/escape edge slicing, balanced brackets, pct_linear, McNemar)
        with pytest.raises(ValueError, match="Discordant counts must be non-negative"):
            v11cfg.mcnemar_one_sided_exact_p(-1, 0)
        assert v11cfg.mcnemar_one_sided_exact_p(0, 3) == 1.0
        assert v11cfg.pct_linear([], 0.1) == 0.0
        assert v11cfg.pct_linear([10.0], 0.1) == 10.0
        assert v11cfg.pct_linear([10.0, 20.0, 30.0], 0.5) == 20.0
        assert v11cfg.pct_linear([10.0, 20.0], 0.25) == 12.5
        assert v11cfg._check_balanced_brackets("a(b[c]{d})", ignore_literals=True) is True
        assert v11cfg._check_balanced_brackets('a("unbalanced [ in literal")', ignore_literals=True) is True
        assert v11cfg._check_balanced_brackets("a(b]", ignore_literals=False) is False
        p_tbl = "| col1 | col2 |\n| --- | --- |\n| val1 | -42 |\n\nNext paragraph here"
        bc = v11cfg._check_span_boundary(p_tbl, 0, len(p_tbl))
        assert bc["para_break"] is True and bc["table_md"] is True
        es = v11cfg._check_span_edge_slicing("value is -42 in text", 10, 12, "...-42...")
        assert es["raw_sign_before_digit"] is True and es["any_sign_or_escape"] is True
        can_res = v11cfg.compute_window_canaries_v11(
            [
                {"quote": "| val1 | -42 |", "passages": [p_tbl]},
                {"quote": "42 in text", "passages": ["value is -42 in text"]},
                {"quote": "unmatched quote", "passages": ["value is -42 in text"]},
                {
                    "quote": "unbalanced [ quote",
                    "passages": ["unbalanced [ quote"],
                    "m2a_matched": True,
                    "raw_span": None,
                },
                {"quote": "span3", "passages": ["span3 ( ok )"], "m2a_matched": True, "raw_span": [0, 0, 5]},
                {"quote": "span2", "passages": ["span2 ( ok )"], "m2a_matched": True, "raw_span": [0, 5]},
            ]
        )
        assert can_res["total_emitted_quotes"] == 6
        assert can_res["boundary_crossing_count"] >= 1
        assert can_res["sign_escape_edge_slice_count"] >= 1

        # reward_details under M1 vs M2a
        reply = "<think>checking</think>" + json.dumps(
            {
                "verdict": "supported",
                "evidence_quotes": [
                    {"part": "p1", "verdict": "supported", "quote": q_punct_ws},
                    {"part": "p2", "verdict": "supported", "quote": q_punct_ws},
                ],
            }
        )
        det_m1 = s2.reward_details(reply, "supported", passages, 5, quote_reward="v2", quote_matcher="rule2")
        det_m2a = s2.reward_details(reply, "supported", passages, 5, quote_reward="v2", quote_matcher="M2a")
        assert det_m1["quote_credit"] == 0.0
        # Under M2a, first quote matches and second identical quote is rejected by span-disjoint anti-farming
        assert det_m2a["quote_credit"] == pytest.approx(0.05) and len(det_m2a["quote_records"]) == 2
        assert s2.reward(reply, "supported", passages, 5, quote_reward="v2", quote_matcher="M2a") > s2.reward(
            reply, "supported", passages, 5, quote_reward="v2", quote_matcher="rule2"
        )

    def test_update1_clamp_and_post_clamp_assertion(self):
        # Wrong-label completion with high format/quote score vs truncated wrong completions
        rewards = [0.20, -1.0, -1.0, -1.0]
        correct = [False, False, False, True]
        raw_adv = s2.advantages(rewards, mode="dr-grpo")
        assert raw_adv[0] > 0.0
        adv = s2.clamp_wrong_positive_advantages(raw_adv, correct)
        # Post-clamp assertion passes because wrong-label completion 0 was clamped to 0.0
        assert adv[0] == 0.0
        s2.assert_no_wrong_positive_advantages(adv, correct)

        # Unclamped wrong-label positive advantage triggers immediate RuntimeError
        with pytest.raises(RuntimeError, match="POST_CLAMP_WRONG_POSITIVE_ADVANTAGE"):
            s2.assert_no_wrong_positive_advantages(raw_adv, correct)

    def test_constant_loss_denominator_4514_matches_microbatch_backward(self):
        torch.manual_seed(7)
        B, T = 4, 6
        mask = torch.ones((B, T), dtype=torch.float32)
        mask[:, -1] = 0.0  # 20 active tokens total
        old_lp = torch.randn((B, T), dtype=torch.float32)
        ref_lp = torch.randn((B, T), dtype=torch.float32)
        adv = torch.tensor([0.5, -0.4, 0.2, -0.3], dtype=torch.float32)

        p1 = torch.nn.Parameter(old_lp.detach().clone())
        loss_default = s2.grpo_loss(p1, old_lp, ref_lp, adv, mask)
        loss_const = s2.grpo_loss(
            p1,
            old_lp,
            ref_lp,
            adv,
            mask,
            constant_loss_denominator=v11cfg.S2V11.constant_loss_denominator,
        )
        scale = float(mask.sum().item()) / float(v11cfg.S2V11.constant_loss_denominator)
        assert pytest.approx(loss_const.item(), rel=1e-5) == loss_default.item() * scale


class TestS2V11GuardConfirmCanariesAndSim:
    def _group_rec(
        self,
        item_id="g00",
        gold="supported",
        verdicts=("supported", "supported"),
        correct=(True, True),
        adv=(0.1, -0.1),
        quote_records=None,
        step=0,
        group=0,
    ):
        return {
            "step": step,
            "group": group,
            "item_id": item_id,
            "gold_verdict": gold,
            "verdicts": list(verdicts),
            "correct_flags": list(correct),
            "advantages_list": list(adv),
            "quote_records": list(quote_records) if quote_records is not None else None,
        }

    def test_step0_baseline_ordering_required_before_confirm(self):
        # Trip Fuse A (4 groups on C with >= 2 cannot_tell -> B_w=4, S_w = 4 - 0.211 = 3.789 >= 3.25)
        # without recording step0_baseline -> raises RuntimeError(STEP0_BASELINE_MISSING)
        guard2 = s2.CtCollapseGuard(
            window_groups=4,
            s2_v11=True,
            canary_c_ids=("c1", "c2", "c3", "c4"),
            gold_unsupported_ids=("c1", "c2", "c3", "c4"),
            confirm_runner=lambda ids, step=0, rule="": {
                i: {"verdict": "unsupported", "correct": True} for i in ids
            },
        )
        for idx in range(3):
            assert (
                guard2.record_group(
                    **self._group_rec(
                        f"c{idx + 1}",
                        "unsupported",
                        verdicts=("cannot_tell", "cannot_tell"),
                        correct=(False, False),
                        adv=(0.0, 0.0),
                    )
                )
                is None
            )
        with pytest.raises(RuntimeError, match="STEP0_BASELINE_MISSING"):
            guard2.record_group(
                **self._group_rec(
                    "c4",
                    "unsupported",
                    verdicts=("cannot_tell", "cannot_tell"),
                    correct=(False, False),
                    adv=(0.0, 0.0),
                )
            )

    def test_s1_inert_and_s2_direct_halt_under_v11(self):
        guard = s2.CtCollapseGuard(
            window_groups=3,
            s2_v11=True,
            canary_c_ids=("c1",),
            gold_unsupported_ids=("u1",),
            confirm_runner=lambda ids, step=0, rule="": {
                i: {"verdict": "unsupported", "correct": True} for i in ids
            },
        )
        guard.record_step0_baseline({"c1": {"verdict": "unsupported"}, "u1": {"verdict": "unsupported"}})
        # 3 distinct supported items with CT: in v1.0 this trips S1, in v1.1 S1 is dropped as inert
        for idx in range(3):
            w = guard.record_group(
                **self._group_rec(
                    f"sup_{idx}",
                    "supported",
                    verdicts=("cannot_tell", "supported"),
                    correct=(False, True),
                    adv=(0.0, 0.2),
                )
            )
        assert w is None
        assert guard.window_history[-1]["s1_inert"] is True
        assert guard.window_history[-1]["triggered_rules"] == []

        # Direct S2 halt when cumulative gold-supported answers >= 600, CT rate > 0.015, distinct CT items >= 4
        guard_s2 = s2.CtCollapseGuard(
            window_groups=1,
            s2_v11=True,
            canary_c_ids=("c1",),
            gold_unsupported_ids=("u1",),
            confirm_runner=lambda ids, step=0, rule="": {
                i: {"verdict": "unsupported", "correct": True} for i in ids
            },
        )
        guard_s2.record_step0_baseline({"c1": {"verdict": "unsupported"}, "u1": {"verdict": "unsupported"}})
        guard_s2.cum_sup_answers = 600
        guard_s2.cum_sup_ct_answers = 12
        guard_s2.cum_sup_ct_items = {"a", "b", "c"}
        w_s2 = guard_s2.record_group(
            **self._group_rec(
                "d",
                "supported",
                verdicts=("cannot_tell", "supported"),
                correct=(False, True),
                adv=(0.0, 0.1),
            )
        )
        assert w_s2 is not None and "S2" in w_s2["triggered_rules"]

    def test_fuse_a_confirm_negative_keeps_cusum_and_confirm_positive_halts(self):
        c_ids = tuple(f"c{i}" for i in range(6))
        confirm_state = {"ct_count": 1}  # first trip: only 1 CT on C (< 2) -> CONFIRM-NEGATIVE

        def runner(ids, step=0, rule=""):
            out = {}
            for idx, cid in enumerate(ids):
                is_ct = idx < confirm_state["ct_count"]
                out[cid] = {"verdict": "cannot_tell" if is_ct else "unsupported", "correct": not is_ct}
            return out

        guard = s2.CtCollapseGuard(
            window_groups=4,
            s2_v11=True,
            canary_c_ids=c_ids,
            gold_unsupported_ids=c_ids,
            confirm_runner=runner,
        )
        guard.record_step0_baseline({cid: {"verdict": "unsupported"} for cid in c_ids})
        # 4 groups on C (using 2 distinct items c0, c1 so U1 does not trip) each with >= 2 CT -> B_w = 4, S_1 = 3.789 >= 3.25 -> Fuse A trips!
        for idx in range(4):
            w1 = guard.record_group(
                **self._group_rec(
                    f"c{idx % 2}",
                    "unsupported",
                    verdicts=("cannot_tell", "cannot_tell"),
                    correct=(False, False),
                    adv=(0.0, 0.0),
                )
            )
        assert w1 is None  # CONFIRM-NEGATIVE does not halt
        assert guard.confirm_trips_total == 1
        assert guard.confirmations_used == 1
        # S_w is NOT reset to 0 after CONFIRM-NEGATIVE; it stays at 3.789 (>= 3.25)
        assert pytest.approx(guard.fuse_a_s, rel=1e-5) == 4.0 - 0.211
        assert any(d.get("status") == "CONFIRM-NEGATIVE" for d in guard.decision_history)

        # Next window: confirm_state returns 3 CT on C (>= 2 and delta over step-0 is 3 >= 2) -> CONFIRM-POSITIVE -> halts!
        confirm_state["ct_count"] = 3
        for idx in range(4):
            w2 = guard.record_group(
                **self._group_rec(
                    f"c{idx % 2}",
                    "unsupported",
                    verdicts=("cannot_tell", "cannot_tell"),
                    correct=(False, False),
                    adv=(0.0, 0.0),
                )
            )
        assert w2 is not None
        assert "FUSE_A-CONFIRMED" in w2["triggered_rules"]
        assert (
            s2.train_verdict_line(
                stopped=w2,
                oom=False,
                next_step=1,
                steps_total=10,
                chunk_stop=None,
                fits=True,
                dry=True,
            )
            == "S2-TRAIN-HALT-FUSE_A-CONFIRMED"
        )

    def test_channel_b_and_u1_mcnemar_confirm_and_4th_trip_halt_cap(self):
        u_ids = tuple(f"u{i}" for i in range(12))
        step0 = {uid: {"verdict": "unsupported", "correct": True} for uid in u_ids}
        call_count = {"n": 0, "worsen": 1, "bad_verdict": "cannot_tell"}

        def runner(ids, step=0, rule=""):
            call_count["n"] += 1
            out = {}
            for idx, uid in enumerate(ids):
                wrong_now = idx < call_count["worsen"]
                out[uid] = {
                    "verdict": call_count["bad_verdict"] if wrong_now else "unsupported",
                    "correct": not wrong_now,
                }
            return out

        guard = s2.CtCollapseGuard(
            window_groups=4,
            s2_v11=True,
            canary_c_ids=("c_none",),
            gold_unsupported_ids=u_ids,
            confirm_runner=runner,
        )
        guard.record_step0_baseline(step0)

        # Trip U1 (4 distinct gold-unsupported items with CT and correct_rate < 0.78) 3 times with n_worse=1 (p=0.5 >= 0.05)
        for trip in range(3):
            for g_idx in range(4):
                w = guard.record_group(
                    **self._group_rec(
                        f"u{g_idx}",
                        "unsupported",
                        verdicts=("cannot_tell", "unsupported"),
                        correct=(False, True),
                        adv=(0.0, 0.1),
                    )
                )
            assert w is None
            assert guard.confirm_trips_total == trip + 1
            assert guard.confirmations_used == trip + 1
            assert call_count["n"] == trip + 1

        # 4th trip hits confirmations_used == 3 == confirm_cap_per_run -> unconditional U1-CAP halt without calling runner!
        for g_idx in range(4):
            w4 = guard.record_group(
                **self._group_rec(
                    f"u{g_idx}",
                    "unsupported",
                    verdicts=("cannot_tell", "unsupported"),
                    correct=(False, True),
                    adv=(0.0, 0.1),
                )
            )
        assert w4 is not None
        assert w4["triggered_rules"] == ["U1-CAP"]
        assert call_count["n"] == 3  # runner was NOT called on the 4th trip!
        assert (
            s2.train_verdict_line(
                stopped=w4,
                oom=False,
                next_step=4,
                steps_total=10,
                chunk_stop=None,
                fits=True,
                dry=True,
            )
            == "S2-TRAIN-HALT-U1-CAP"
        )

        # Now test Channel B + U1 tripping in the SAME window: they share a single 184-item pass!
        # And when 6 items flip to 'supported' (p = 0.5^6 = 0.015625 < 0.05), Channel B confirms positive!
        call_count["n"] = 0
        call_count["worsen"] = 6
        call_count["bad_verdict"] = "supported"
        guard_cb = s2.CtCollapseGuard(
            window_groups=13,
            s2_v11=True,
            canary_c_ids=("c_none",),
            gold_unsupported_ids=u_ids,
            confirm_runner=runner,
        )
        guard_cb.record_step0_baseline(step0)
        # 13 groups on gold-unsupported with >= 2 'supported' -> FA2_w = 13, S_w = 13 - 5.03 = 7.97 >= 7.0 (Channel B trips)
        for g_idx in range(13):
            w_cb = guard_cb.record_group(
                **self._group_rec(
                    f"u{g_idx % 12}",
                    "unsupported",
                    verdicts=("supported", "supported"),
                    correct=(False, False),
                    adv=(0.0, 0.0),
                )
            )
        assert w_cb is not None
        assert "CHANNEL_B-CONFIRMED" in w_cb["triggered_rules"]
        assert call_count["n"] == 1

    def test_six_window_canaries_and_two_consecutive_window_stop_and_report(self):
        guard = s2.CtCollapseGuard(
            window_groups=1,
            s2_v11=True,
            canary_c_ids=("c1",),
            gold_unsupported_ids=("u1",),
            confirm_runner=lambda ids, step=0, rule="": {},
        )
        guard.record_step0_baseline({"c1": {"verdict": "unsupported"}, "u1": {"verdict": "unsupported"}})
        # Quote that fails M1 (shadow_m1_pass_rate = 0.0 < 0.70) and has unbalanced brackets
        bad_q = [
            {
                "quote": "def foo ( x , y : return [ x + y",
                "passage": "def foo(x, y): return [x + y]",
                "m1_matched": False,
                "m2a_matched": True,
                "raw_span": [0, 0, 28, "m2a_1"],
            }
        ]
        # Window 1: 1 window breach does NOT trigger STOP-AND-REPORT
        w1 = guard.record_group(**self._group_rec("s1", "supported", quote_records=bad_q))
        assert w1 is None
        assert guard.canary_violation_streaks["shadow_m1_pass_rate"] == 1
        assert guard.canary_violation_streaks["unbalanced_brackets_rate"] == 1

        # Window 2: 2 consecutive windows breaching canary -> STOP-AND-REPORT!
        w2 = guard.record_group(**self._group_rec("s2", "supported", quote_records=bad_q))
        assert w2 is not None
        assert w2["canary_stop"] is True
        assert "shadow_m1_pass_rate" in w2["tripped_canaries"]
        verdict = s2.train_verdict_line(
            stopped=w2,
            oom=False,
            next_step=2,
            steps_total=10,
            chunk_stop=None,
            fits=True,
            dry=True,
        )
        assert verdict == "STOP-AND-REPORT"
        assert s2.exit_status(verdict) == 3  # distinct from guard halt (2)

        # State serialization round-trip preserves v1.1 fields
        st = guard.to_state()
        restored = s2.CtCollapseGuard.from_state(
            st,
            pass_a_k={},
            s2_v11=True,
            canary_c_ids=("c1",),
            gold_unsupported_ids=("u1",),
        )
        assert restored.canary_violation_streaks == guard.canary_violation_streaks
        assert restored.step0_baseline == guard.step0_baseline

        # Verify evaluate_canary_rollback_v11 on each of the 5 rollback lines (i, ii, iii, iv, vi)
        for line_key, bad_canaries in [
            (
                "shadow_m1_pass_rate",
                {"total_emitted_quotes": 10, "m2a_matched_quotes": 10, "shadow_m1_pass_rate": 0.65},
            ),
            (
                "boundary_crossing_rate",
                {"total_emitted_quotes": 10, "m2a_matched_quotes": 10, "boundary_crossing_rate": 0.02},
            ),
            (
                "sign_escape_edge_slice_count",
                {"total_emitted_quotes": 10, "m2a_matched_quotes": 10, "sign_escape_edge_slice_count": 3},
            ),
            (
                "unbalanced_brackets_rate",
                {"total_emitted_quotes": 10, "m2a_matched_quotes": 10, "unbalanced_brackets_rate": 0.30},
            ),
            (
                "accepted_quote_norm_len_p10",
                {"total_emitted_quotes": 10, "m2a_matched_quotes": 10, "accepted_quote_norm_len_p10": 14.0},
            ),
        ]:
            r1 = v11cfg.evaluate_canary_rollback_v11(bad_canaries, {})
            assert r1["violations"][line_key] is True and r1["stop_and_report"] is False
            r2 = v11cfg.evaluate_canary_rollback_v11(bad_canaries, r1["streaks"])
            assert r2["stop_and_report"] is True and line_key in r2["tripped_canaries"]
            assert r2["verdict"] == "STOP-AND-REPORT"

    def test_poisson_cusum_100k_simulation_reproduces_prereg_rates(self):
        # §5.1 Fuse A: Garwood 95% upper bound lam_win = 0.1157, k = 0.211, h = 3.25 over 20 windows -> FA ≈ 0.022
        sim_a = v11cfg.simulate_constant_exposure_cusum(
            0.1157, v11cfg.S2V11.fuse_a_k, v11cfg.S2V11.fuse_a_h, n_runs=100_000, seed=20261108
        )
        # §5.2 Channel B: Garwood 95% upper bound lam_win = 3.4846, k = 5.03, h = 7.0 over 20 windows -> FA ≈ 0.018
        sim_b = v11cfg.simulate_constant_exposure_cusum(
            3.4846, v11cfg.S2V11.channel_b_k, v11cfg.S2V11.channel_b_h, n_runs=100_000, seed=20261108
        )
        assert abs(sim_a["rate"] - 0.022) <= 0.002
        assert abs(sim_b["rate"] - 0.018) <= 0.002


class TestS2V11DryAndDefaultUnchanged:
    def test_s2_v11_dry_two_step_chunk_and_default_path_unchanged(self, tmp_path, monkeypatch, capsys):
        gold = _gold(tmp_path)
        pa_path, pa_sha, pb_path, pb_sha, syn_cfg, syn_sets = _make_synthetic_v11_prepass(tmp_path, gold)

        # First verify that without patching S2V11's 128/184 counts, execute() refuses with S2_V11_GOLD_UNSUPPORTED_SIZE_MISMATCH
        monkeypatch.setattr(
            s2, "S2V11", dataclasses.replace(v11cfg.S2V11, pass_a_sha256=pa_sha, pass_b_sha256=pb_sha)
        )
        out_mismatch = tmp_path / "v11_mismatch"
        with pytest.raises(SystemExit, match="S2_V11_GOLD_UNSUPPORTED_SIZE_MISMATCH"):
            _run(
                [
                    "--gold",
                    str(gold),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(out_mismatch),
                    "--dry",
                    "--s2-v11",
                    "--arm",
                    "mech",
                    "--train-seed",
                    "45",
                    "--stop-after-updates",
                    "2",
                    "--max-new",
                    "8",
                    "--pass-a-jsonl",
                    str(pa_path),
                    "--pass-a-sha256",
                    pa_sha,
                    "--pass-b-jsonl",
                    str(pb_path),
                    "--pass-b-sha256",
                    pb_sha,
                ]
            )

        # Now patch S2V11 to match the synthetic 60-item gold fixture's counts & SHAs
        monkeypatch.setattr(v11cfg, "S2V11", syn_cfg)
        monkeypatch.setattr(s2, "S2V11", syn_cfg)
        out_v11 = tmp_path / "v11_dry"
        rc_v11 = _run(
            [
                "--gold",
                str(gold),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(out_v11),
                "--dry",
                "--s2-v11",
                "--arm",
                "mech",
                "--train-seed",
                "45",
                "--stop-after-updates",
                "2",
                "--max-new",
                "8",
                "--pass-a-jsonl",
                str(pa_path),
                "--pass-a-sha256",
                pa_sha,
                "--pass-b-jsonl",
                str(pb_path),
                "--pass-b-sha256",
                pb_sha,
            ]
        )
        # In dry mode at --max-new 8, completions truncate on update 0 (still < 10-update trunc window)
        # and guard window is 40 groups (10 updates), so 2 steps complete cleanly with rc == 0.
        assert rc_v11 == 0
        captured = capsys.readouterr().out
        assert '"event": "s2_v11_canary_sets_ready"' in captured
        rec = _receipt(out_v11)
        assert rec["prereg_version"] == "1.1"
        assert rec["prereg_sha256"] == v11cfg.PREREG_V11_SHA256
        assert len(rec["s2_v11_config_sha256"]) == 64
        assert rec["s2_v11_config"]["prereg_version"] == "1.1"
        assert rec["canary_c_ids_sha256"] == syn_sets["canary_c_ids_sha256"]
        assert rec["gold_unsupported_ids_sha256"] == syn_sets["gold_unsupported_ids_sha256"]
        hdr = json.loads((out_v11 / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert hdr["prereg_sha256"] == v11cfg.PREREG_V11_SHA256
        assert hdr["settings"]["s2_v11"] is True
        assert hdr["canary_c_count"] == syn_sets["canary_c_count"]
        assert hdr["canary_c_ids_sha256"] == syn_sets["canary_c_ids_sha256"]
        assert hdr["gold_unsupported_count"] == syn_sets["gold_unsupported_count"]
        assert hdr["gold_unsupported_ids_sha256"] == syn_sets["gold_unsupported_ids_sha256"]
        assert hdr["pass_a_sha256"] == pa_sha
        assert hdr["pass_b_sha256"] == pb_sha
        hist = _history(out_v11)
        assert any(h.get("kind") == "step0_canary_pass" for h in hist)
        rows = _rows(out_v11)
        assert len(rows) == 2

        # Default path (no --s2-v11) has no v1.1 fields in header, steps, or save-all rows
        out_def = tmp_path / "default_dry"
        rc_def = _run(_argv(gold, out_def, steps=2, gpu=2))
        assert rc_def == 0
        rec_def = _receipt(out_def)
        hdr_def = json.loads((out_def / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        assert "prereg_version" not in rec_def and "s2_v11_config" not in rec_def
        assert "prereg_version" not in hdr_def and "s2_v11" not in hdr_def["settings"]
        rows_def = _rows(out_def)
        assert len(rows_def) == 2
        sa_def = s2.read_save_all(out_def / "s2-hf-save-all.jsonl.gz")
        assert len(sa_def) == 8

        # Exercise run_v11_t0_pass with dry=False on a fake CPU model + tokenizer (no GPU)
        import types

        reply_str = "<think>t0 check</think>" + json.dumps(
            {
                "verdict": "unsupported",
                "evidence_quotes": [{"part": "p1", "verdict": "unsupported", "quote": "def count(tokens):"}],
            }
        )

        class _FakeTok:
            pad_token_id = 0
            eos_token_id = 2

            def __call__(self, txt, return_tensors="pt"):
                return {"input_ids": torch.tensor([[1, 2, 3]])}

            def decode(self, seq, skip_special_tokens=True):
                return reply_str

            def encode(self, txt):
                return [1, 2]

        fake_tok = _FakeTok()
        fake_model = types.SimpleNamespace(
            device="cpu",
            generate=lambda prompt_ids, **kw: torch.tensor([[1, 2, 3, 4, 5, 2, 0]]),
        )
        t0_res = s2.run_v11_t0_pass(
            model=fake_model,
            tok=fake_tok,
            rows_by_id={"g00": _row("g00", label="unsupported")},
            item_ids=["g00"],
            max_new=16,
            render_fn=lambda s: s,
            pass_attention_mask=True,
            quote_reward="v2",
            dry=False,
            device_sync=lambda: None,
        )
        assert t0_res["g00"]["verdict"] == "unsupported"
        assert t0_res["g00"]["correct"] is True

        # Also exercise reward_details M2a branches for wrong verdict and v1 quote_reward
        det_wrong = s2.reward_details(
            reply_str, "supported", ["def count(tokens): return 1"], 4, quote_reward="v2", quote_matcher="M2a"
        )
        assert det_wrong["correct"] is False and det_wrong["quote_matcher"] == "M2a"
        det_v1_ok = s2.reward_details(
            reply_str,
            "unsupported",
            ["def count(tokens): return 1"],
            4,
            quote_reward="v1",
            quote_matcher="M2a",
        )
        assert det_v1_ok["correct"] is True and det_v1_ok["quote_matcher"] == "M2a"
        det_v1_wrong = s2.reward_details(
            reply_str, "supported", ["def count(tokens): return 1"], 4, quote_reward="v1", quote_matcher="M2a"
        )
        assert det_v1_wrong["correct"] is False and det_v1_wrong["quote_matcher"] == "M2a"

    def test_s2_v11_m2a_wiring_every_site_and_default_rule2(self) -> None:
        """Ask 1: Under --s2-v11 every quote judgement site uses M2a; without --s2-v11 it uses rule2."""
        # A quote that matches M2a (code-punctuation whitespace tolerant) and fails rule2:
        passage = "def compute_total(items, tax_rate): return sum(items)"
        quote_m2a_only = "def compute_total ( items , tax_rate ) : return sum ( items )"
        assert s2.quote_found(quote_m2a_only, [passage]) is False
        assert s2.quote_found_m2a(quote_m2a_only, [passage]) is True

        reply_sup = "<think>ok</think>" + json.dumps(
            {
                "verdict": "supported",
                "evidence_quotes": [{"part": "p1", "verdict": "supported", "quote": quote_m2a_only}],
            }
        )
        reply_unsup = "<think>ok</think>" + json.dumps(
            {
                "verdict": "unsupported",
                "evidence_quotes": [{"part": "p1", "verdict": "unsupported", "quote": quote_m2a_only}],
            }
        )
        row_sup = {
            "id": "m2a-item",
            "claim": "compute_total sums items",
            "label": "supported",
            "context": [{"text": passage}],
        }

        # 1. _call_reward_fn with a reward_fn like real_bundle's
        def rb_default(text, r, tt, truncated=False, quote_matcher="rule2"):
            return s2.reward(
                text,
                r["label"],
                s2.passages_of(r),
                tt,
                truncated=truncated,
                quote_reward="v2",
                quote_matcher=quote_matcher,
            )

        def rb_v11(text, r, tt, truncated=False, quote_matcher=s2.S2V11.matcher):
            return s2.reward(
                text,
                r["label"],
                s2.passages_of(r),
                tt,
                truncated=truncated,
                quote_reward="v2",
                quote_matcher=quote_matcher,
            )
        r_rule2 = s2._call_reward_fn(rb_default, reply_sup, row_sup, 4, False, quote_matcher="rule2")
        r_m2a = s2._call_reward_fn(rb_v11, reply_sup, row_sup, 4, False, quote_matcher="M2a")
        assert r_rule2 == pytest.approx(-0.2)
        assert r_m2a > 1.0  # 1.0 + M2a quote credit (+0.05)

        # 2. completion_eval_outcome
        out_r2 = s2.completion_eval_outcome(reply_sup, row_sup, reward_val=r_rule2, quote_matcher="rule2")
        out_m2a = s2.completion_eval_outcome(reply_sup, row_sup, reward_val=r_m2a, quote_matcher="M2a")
        assert out_r2["quote_outcomes"][0]["found"] is False
        assert out_m2a["quote_outcomes"][0]["found"] is True
        assert out_m2a["quote_outcomes"][0]["rung"] == "M2a"

        # 3. evaluation_summary (both via quote_outcomes and raw quotes fallback)
        raw_grp = [
            {
                "gold": "supported",
                "claim": "compute_total sums items",
                "passages": [passage],
                "rollouts": [{"verdict": "supported", "reward": 1.05, "quotes": [quote_m2a_only]}],
            }
        ]
        sum_r2 = s2.evaluation_summary(raw_grp, advantage_mode="dr-grpo", quote_matcher="rule2")
        sum_m2a = s2.evaluation_summary(raw_grp, advantage_mode="dr-grpo", quote_matcher="M2a")
        assert sum_r2["quote_relevance"]["total"] == 0
        assert sum_m2a["quote_relevance"]["total"] == 1

        # 4. prepass_item_record
        tok = s2.DryTokenizer()
        comp_ids = [tok.encode(reply_sup)]
        rec_r2, _ = s2.prepass_item_record(
            row=row_sup,
            prompt_ids=[2, 3],
            comps=comp_ids,
            truncs=[False],
            tok=tok,
            reward_fn=rb_default,
            quote_reward="v2",
            quote_matcher="rule2",
        )
        rec_m2a, _ = s2.prepass_item_record(
            row=row_sup,
            prompt_ids=[2, 3],
            comps=comp_ids,
            truncs=[False],
            tok=tok,
            reward_fn=rb_v11,
            quote_reward="v2",
            quote_matcher="M2a",
        )
        assert rec_r2["quote_pass"] == 0
        assert rec_m2a["quote_pass"] == 1

        # 5. critic_hint
        hint_r2 = s2.critic_hint("compute_total", [passage], [(reply_unsup, False)], quote_matcher="rule2")
        hint_m2a = s2.critic_hint("compute_total", [passage], [(reply_unsup, False)], quote_matcher="M2a")
        assert hint_r2["quote_check"] == "quote_not_found"
        assert hint_m2a["quote_check"] == "quotes_found"

    def test_s2_v11_audited_quote_rule_isolation_and_four_sha_checks(self, tmp_path: Path) -> None:
        """Ask 2: Verify 4-file SHA checks, lazy isolated loader, and sys.path[0] poison immunity."""
        import shutil
        import sys

        import s2_v11_config as v11cfg

        verifier_dir = Path(s2.__file__).resolve().parent
        shas = v11cfg.verify_v11_sealed_files()
        assert shas["prereg_sha256"] == v11cfg.PREREG_V11_SHA256
        assert shas["quote_matcher_m1_sha256"] == v11cfg.S2V11.quote_matcher_m1_sha256
        assert shas["quote_matcher_m2_sha256"] == v11cfg.S2V11.quote_matcher_m2_sha256
        assert shas["quote_rule_audited_sha256"] == v11cfg.QUOTE_RULE_AUDITED_SHA256_V11

        # Create a shadow verifier_dir in tmp_path to test each of the 4 SHA mismatch codes
        fake_vdir = tmp_path / "verifier_dir"
        fake_vdir.mkdir()
        fake_rroot = tmp_path / "repo_root"
        (fake_rroot / "docs" / "runs").mkdir(parents=True)
        prereg_dst = fake_rroot / v11cfg.PREREG_V11_RELPATH
        shutil.copyfile(verifier_dir.parents[1] / v11cfg.PREREG_V11_RELPATH, prereg_dst)
        for fname in ("quote_matcher_m1.py", "quote_matcher_m2.py", v11cfg.QUOTE_RULE_AUDITED_FILENAME_V11):
            shutil.copyfile(verifier_dir / fname, fake_vdir / fname)

        # (1) S2_V11_PREREG_SHA256_MISMATCH
        orig_bytes = prereg_dst.read_bytes()
        prereg_dst.write_bytes(orig_bytes + b"\n")
        with pytest.raises(ValueError, match=r"S2_V11_PREREG_SHA256_MISMATCH.*expected 9b2db4cf"):
            v11cfg.load_v11_matchers(force_reload=True, repo_root=fake_rroot, verifier_dir=fake_vdir)
        prereg_dst.write_bytes(orig_bytes)

        # (2) S2_V11_M1_SHA256_MISMATCH
        m1_dst = fake_vdir / "quote_matcher_m1.py"
        orig_m1 = m1_dst.read_bytes()
        m1_dst.write_bytes(orig_m1 + b"\n")
        with pytest.raises(ValueError, match=r"S2_V11_M1_SHA256_MISMATCH.*expected 57064f46"):
            v11cfg.load_v11_matchers(force_reload=True, repo_root=fake_rroot, verifier_dir=fake_vdir)
        m1_dst.write_bytes(orig_m1)

        # (3) S2_V11_M2_SHA256_MISMATCH
        m2_dst = fake_vdir / "quote_matcher_m2.py"
        orig_m2 = m2_dst.read_bytes()
        m2_dst.write_bytes(orig_m2 + b"\n")
        with pytest.raises(ValueError, match=r"S2_V11_M2_SHA256_MISMATCH.*expected 98780f92"):
            v11cfg.load_v11_matchers(force_reload=True, repo_root=fake_rroot, verifier_dir=fake_vdir)
        m2_dst.write_bytes(orig_m2)

        # (4) S2_V11_QUOTE_RULE_SHA256_MISMATCH
        qr_dst = fake_vdir / v11cfg.QUOTE_RULE_AUDITED_FILENAME_V11
        orig_qr = qr_dst.read_bytes()
        qr_dst.write_bytes(orig_qr + b"\n")
        with pytest.raises(ValueError, match=r"S2_V11_QUOTE_RULE_SHA256_MISMATCH.*expected 1aae2469"):
            v11cfg.load_v11_matchers(force_reload=True, repo_root=fake_rroot, verifier_dir=fake_vdir)
        qr_dst.write_bytes(orig_qr)

        # Verify module isolation and immunity to a poisoned quote_rule.py on sys.path[0]
        passage = "def compute_total(items, tax_rate): return sum(items)"
        quote_str = "def compute_total ( items , tax_rate ) : return sum ( items )"
        qr_mod, m1_mod, m2_mod = v11cfg.load_v11_matchers(force_reload=True)
        assert m1_mod.quote_found.__module__ == "quote_rule_audited_1aae2469"
        assert s2.quote_found.__module__ == "quote_rule"
        res_clean = m2_mod.quote_found_m2a(quote_str, [passage])

        poison_dir = tmp_path / "poison_sys_path"
        poison_dir.mkdir()
        (poison_dir / "quote_rule.py").write_text(
            "raise RuntimeError('POISON_QUOTE_RULE_IMPORTED')\n",
            encoding="utf-8",
        )
        orig_sys_path = list(sys.path)
        try:
            sys.path.insert(0, str(poison_dir))
            qr_mod2, m1_mod2, m2_mod2 = v11cfg.load_v11_matchers(force_reload=True)
            assert m1_mod2.quote_found.__module__ == "quote_rule_audited_1aae2469"
            assert s2.quote_found.__module__ == "quote_rule"
            res_poison = m2_mod2.quote_found_m2a(quote_str, [passage])
            assert res_poison == res_clean is True
            assert sys.path == [str(poison_dir), *orig_sys_path]
        finally:
            sys.path[:] = orig_sys_path

    def test_s2_v11_t0_pass_eval_mode_step0_and_midrun_restores_train_mode(self) -> None:
        """Ask 3: Both step-0 baseline and mid-run confirmation run in eval mode and restore training mode afterwards."""
        import torch

        reply_str = "<think>t0 check</think>" + json.dumps(
            {
                "verdict": "unsupported",
                "evidence_quotes": [
                    {"part": "p1", "verdict": "unsupported", "quote": "def count(tokens): return 1"}
                ],
            }
        )

        class _FakeTok:
            pad_token_id = 0
            eos_token_id = 2

            def __call__(self, txt, return_tensors="pt"):
                return {"input_ids": torch.tensor([[1, 2, 3]])}

            def decode(self, seq, skip_special_tokens=True):
                return reply_str

            def encode(self, txt):
                return [1, 2]

        observed_in_generate: list[dict] = []

        class _FakeBaseConfig:
            use_cache = False

        class _FakeInner:
            def __init__(self) -> None:
                self.config = _FakeBaseConfig()
                self.is_gradient_checkpointing = True

            def gradient_checkpointing_enable(self, **kw) -> None:
                self.is_gradient_checkpointing = True

            def gradient_checkpointing_disable(self) -> None:
                self.is_gradient_checkpointing = False

        class _FakeModel:
            device = "cpu"

            def __init__(self) -> None:
                self.training = True
                self._inner = _FakeInner()

            def get_base_model(self):
                return self._inner

            def train(self, mode_flag: bool = True):
                self.training = bool(mode_flag)
                return self

            def eval(self):
                self.training = False
                return self

            def generate(self, prompt_ids, **kw):
                observed_in_generate.append(
                    {
                        "training": self.training,
                        "use_cache": self._inner.config.use_cache,
                        "grad_ckpt": self._inner.is_gradient_checkpointing,
                    }
                )
                return torch.tensor([[1, 2, 3, 4, 5, 2, 0]])

        model = _FakeModel()
        mode_fn, state_fn = s2.make_mode(model)
        tok = _FakeTok()
        rows_by_id = {"c0": _row("c0", label="unsupported"), "c1": _row("c1", label="unsupported")}

        # Step-0 pass (starting from training mode True, grad_ckpt True)
        assert model.training is True and state_fn()["grad_ckpt"] is True
        step0_res = s2.run_v11_t0_pass(
            model=model,
            tok=tok,
            rows_by_id=rows_by_id,
            item_ids=["c0", "c1"],
            max_new=16,
            dry=False,
            mode=mode_fn,
            state=state_fn,
        )
        # Inside generate, model was in eval mode (training=False, use_cache=True, grad_ckpt=False)
        assert len(observed_in_generate) == 2
        assert all(
            obs == {"training": False, "use_cache": True, "grad_ckpt": False} for obs in observed_in_generate
        )
        # Each pass item record records mode='eval', eval_mode=True, grad_ckpt=False
        for cid in ("c0", "c1"):
            assert step0_res[cid]["mode"] == "eval"
            assert step0_res[cid]["eval_mode"] is True
            assert step0_res[cid]["grad_ckpt"] is False
        # Training mode is restored afterwards
        assert model.training is True
        assert state_fn()["grad_ckpt"] is True
        assert model._inner.config.use_cache is False

        # Mid-run confirmation pass via CtCollapseGuard (starting from rollout/eval mode False)
        mode_fn(False)
        assert model.training is False
        observed_in_generate.clear()

        guard = s2.CtCollapseGuard(
            window_groups=2,
            s2_v11=True,
            canary_c_ids=("c0", "c1"),
            gold_unsupported_ids=("c0", "c1"),
            confirm_runner=lambda ids, step=0, rule="": s2.run_v11_t0_pass(
                model=model,
                tok=tok,
                rows_by_id=rows_by_id,
                item_ids=ids,
                max_new=16,
                dry=False,
                mode=mode_fn,
                state=state_fn,
            ),
        )
        s0_rec = guard.record_step0_baseline(step0_res)
        assert s0_rec["mode"] == "eval" and s0_rec["eval_mode"] is True

        # Trigger Fuse A in a 2-group window (B_w = 4 across 2 windows)
        for _ in range(4):
            guard.record_group(
                step=1,
                group=0,
                item_id="c0",
                gold_verdict="unsupported",
                verdicts=("cannot_tell", "cannot_tell"),
                correct_flags=(False, False),
                advantages_list=(0.0, 0.0),
            )
        # Mid-run confirmation ran in eval mode and restored training mode afterwards!
        assert len(observed_in_generate) == 2
        assert all(
            obs == {"training": False, "use_cache": True, "grad_ckpt": False} for obs in observed_in_generate
        )
        assert model.training is True
        assert state_fn()["grad_ckpt"] is True
        conf_recs = [d for d in guard.decision_history if d.get("kind") == "guard_confirmation"]
        assert len(conf_recs) == 1
        assert conf_recs[0]["mode"] == "eval" and conf_recs[0]["eval_mode"] is True

    def test_s2_v11_joint_channel_b_u1_shared_pass_counts_once_and_resumes_cap(self) -> None:
        """Ask 4: Joint CHANNEL_B + U1 trip runs 184-item pass once, counts once toward cap, and U1 reads cache even at cap==2."""
        u_ids = tuple(f"u{i}" for i in range(12))
        step0 = {uid: {"verdict": "unsupported", "correct": True} for uid in u_ids}
        runner_State = {"calls": 0, "sup_flips": 1, "ct_flips": 1}

        def runner(ids, step=0, rule=""):
            runner_State["calls"] += 1
            out = {}
            for idx, uid in enumerate(ids):
                if idx < runner_State["sup_flips"]:
                    v = "supported"
                elif idx < runner_State["sup_flips"] + runner_State["ct_flips"]:
                    v = "cannot_tell"
                else:
                    v = "unsupported"
                out[uid] = {"verdict": v, "correct": v == "unsupported", "mode": "eval", "eval_mode": True}
            return out

        guard = s2.CtCollapseGuard(
            window_groups=13,
            s2_v11=True,
            canary_c_ids=("c_none",),
            gold_unsupported_ids=u_ids,
            confirm_runner=runner,
        )
        guard.record_step0_baseline(step0)

        # Window 1: trip BOTH CHANNEL_B (13 groups with >=2 supported -> FA2=13, S_B=7.97>=7.0)
        # and U1 (>=4 distinct gold-unsupported items with CT and correct_rate < 0.78).
        # With sup_flips=1, ct_flips=1: CHANNEL_B is CONFIRM-NEGATIVE, and U1 reads the SAME cached pass!
        for g_idx in range(13):
            w1 = guard.record_group(
                step=1,
                group=g_idx,
                item_id=f"u{g_idx % 6}",
                gold_verdict="unsupported",
                verdicts=("supported", "supported", "cannot_tell", "unsupported"),
                correct_flags=(False, False, False, True),
                advantages_list=(0.0, 0.0, 0.0, 0.1),
            )
        assert w1 is None
        assert runner_State["calls"] == 1
        assert guard.confirm_trips_total == 2  # both CHANNEL_B and U1 tripped
        assert guard.confirmations_used == 1  # shared pass counted ONCE!
        confs_w1 = [d for d in guard.decision_history if d.get("kind") == "guard_confirmation"]
        assert [c["rule"] for c in confs_w1] == ["CHANNEL_B", "U1"]
        assert confs_w1[0]["shared_pass"] is False and confs_w1[0]["confirmations_used"] == 1
        assert confs_w1[1]["shared_pass"] is True and confs_w1[1]["confirmations_used"] == 1

        # Test chunk resume preserves confirmations_used and confirm_trips_total
        guard.confirmations_used = 2  # simulate 2 confirmations used before window 2
        saved_state = guard.to_state()
        resumed = s2.CtCollapseGuard.from_state(
            saved_state,
            pass_a_k={},
            mode="halt",
            s2_v11=True,
            canary_c_ids=("c_none",),
            gold_unsupported_ids=u_ids,
            confirm_runner=runner,
        )
        assert resumed.confirmations_used == 2
        assert resumed.confirm_trips_total == 2

        # Window 2 on resumed guard entering with confirmations_used == 2:
        # Joint CHANNEL_B + U1 trip where CHANNEL_B runs confirmation #3 (CONFIRM-NEGATIVE),
        # and U1 MUST read the cached pass (NOT return U1-CAP!) and confirm positive when ct_flips=6!
        runner_State["sup_flips"] = 1  # CHANNEL_B negative (p=0.5)
        runner_State["ct_flips"] = 6  # U1 positive (p=0.5^6 = 0.015625 < 0.05)
        for g_idx in range(13):
            w2 = resumed.record_group(
                step=2,
                group=g_idx,
                item_id=f"u{g_idx % 6}",
                gold_verdict="unsupported",
                verdicts=("supported", "supported", "cannot_tell", "unsupported"),
                correct_flags=(False, False, False, True),
                advantages_list=(0.0, 0.0, 0.0, 0.1),
            )
        assert w2 is not None
        assert w2["triggered_rules"] == ["U1-CONFIRMED"]
        assert runner_State["calls"] == 2  # only 1 new pass executed in window 2
        assert resumed.confirmations_used == 3

    def test_s2_v11_update_60_state_protected_and_recorded(self, tmp_path: Path, monkeypatch) -> None:
        """Ask 5: Update-60 state is never pruned by save_train_state and its path + adapter sha are recorded."""
        gold = _gold(tmp_path)
        pa_path, pa_sha, pb_path, pb_sha, syn_cfg, _ = _make_synthetic_v11_prepass(tmp_path, gold)
        monkeypatch.setattr(v11cfg, "S2V11", syn_cfg)
        monkeypatch.setattr(s2, "S2V11", syn_cfg)

        a = _ns(
            [
                "--gold",
                str(gold),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "sec60_unit"),
                "--dry",
                "--s2-v11",
                "--arm",
                "mech",
                "--train-seed",
                "45",
                "--pass-a-jsonl",
                str(pa_path),
                "--pass-a-sha256",
                pa_sha,
                "--pass-b-jsonl",
                str(pb_path),
                "--pass-b-sha256",
                pb_sha,
            ]
        )
        s2.validate_training_args(a)
        bundle = s2.dry_bundle(a)
        hdr = {"prereg_version": "1.1", "settings": {"s2_v11": True, "groups_per_update": 4}}
        trunc = s2.TruncationHaltGuard(window_updates=10, halt_rate=0.20)
        out_u = tmp_path / "sec60_unit"
        for step_n in (50, 60, 70, 80):
            s2.save_train_state(
                out_u,
                model=bundle["model"],
                optimiser=bundle["optimiser"],
                header=hdr,
                next_step=step_n,
                steps_total=197,
                chunk_index=1,
                ct_guard=None,
                trunc_guard=trunc,
                prior_chain=[],
                chunk_link={"chunk_dir": out_u.name, "chunk_index": 1, "first_step": 0, "next_step": step_n},
                adapter_sha=None,
                halted=None,
                resumed_from=None,
                kind="periodic",
                keep=2,
            )
        saved_names = [d.name for d in s2.state_dirs(out_u)]
        # Step 50 was pruned, Step 60 was protected, and newest 2 (70, 80) were kept!
        assert saved_names == ["0001-step-00060", "0002-step-00070", "0003-step-00080"]
        assert (out_u / "adapter-step-00060").is_dir()
        st80 = json.loads((s2.latest_state_dir(out_u) / "state.json").read_text(encoding="utf-8"))
        assert st80["secondary_checkpoint"]["update"] == 60
        assert st80["secondary_checkpoint"]["state_dir"] == "train-state/0001-step-00060"
        assert len(st80["secondary_checkpoint"]["adapter_sha256"]) == 64

        # Also verify end-to-end execute() records update_60_state_path and update_60_adapter_sha256
        # in both s2-hf-train-header.json and s2-hf.json when the secondary checkpoint update is reached
        syn_cfg_sec1 = dataclasses.replace(syn_cfg, secondary_checkpoint_update=1)
        monkeypatch.setattr(v11cfg, "S2V11", syn_cfg_sec1)
        monkeypatch.setattr(s2, "S2V11", syn_cfg_sec1)
        out_e2e = tmp_path / "sec_e2e"
        rc = _run(
            [
                "--gold",
                str(gold),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(out_e2e),
                "--dry",
                "--s2-v11",
                "--arm",
                "mech",
                "--train-seed",
                "45",
                "--stop-after-updates",
                "3",
                "--state-every-updates",
                "1",
                "--max-new",
                "8",
                "--pass-a-jsonl",
                str(pa_path),
                "--pass-a-sha256",
                pa_sha,
                "--pass-b-jsonl",
                str(pb_path),
                "--pass-b-sha256",
                pb_sha,
            ]
        )
        assert rc == 0
        e2e_dirs = [d.name for d in s2.state_dirs(out_e2e)]
        assert "0000-step-00001" in e2e_dirs
        hdr_e2e = json.loads((out_e2e / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        rec_e2e = _receipt(out_e2e)
        assert hdr_e2e["update_60_state_path"].endswith("train-state/0000-step-00001")
        assert len(hdr_e2e["update_60_adapter_sha256"]) == 64
        assert rec_e2e["update_60_state_path"] == hdr_e2e["update_60_state_path"]
        assert rec_e2e["update_60_adapter_sha256"] == hdr_e2e["update_60_adapter_sha256"]
        assert rec_e2e["training"]["update_60_adapter_sha256"] == hdr_e2e["update_60_adapter_sha256"]

    def test_s2_v11_stop_and_report_finishes_update_and_resume_after_review(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Ask 6: STOP-AND-REPORT finishes the current update (keeps gradients), writes stopped_for_review,
        and resumes only with --resume-after-review '<reason>' recorded in header and s2-hf.json."""
        gold = _gold(tmp_path)
        pa_path, pa_sha, pb_path, pb_sha, syn_cfg, _ = _make_synthetic_v11_prepass(tmp_path, gold)
        monkeypatch.setattr(v11cfg, "S2V11", syn_cfg)
        monkeypatch.setattr(s2, "S2V11", syn_cfg)

        # Force CtCollapseGuard.record_group to return a canary STOP-AND-REPORT on step 1, group 3
        orig_record_group = s2.CtCollapseGuard.record_group

        def _fake_record_group(self, **kw):
            res = orig_record_group(self, **kw)
            if int(kw.get("step", -1)) == 1 and int(kw.get("group", -1)) == 3:
                return {
                    "step": 1,
                    "group": 3,
                    "window_index": 1,
                    "reason": "canary_stop_and_report",
                    "canary_stop": True,
                    "triggered_rules": [],
                    "triggered_canaries": ["C1_SHADOW_M1_PASS_RATE"],
                    "verdict": "STOP-AND-REPORT",
                }
            return res

        monkeypatch.setattr(s2.CtCollapseGuard, "record_group", _fake_record_group)

        def _non_truncated_dry_bundle(a):
            b = s2.dry_bundle(a)

            def _rollout_no_trunc(text, g, max_new, temp):
                return [2, 3, 4], [[10 + i, 11 + i] for i in range(g)], [False for _ in range(g)]

            b["rollout"] = _rollout_no_trunc
            return b

        c1 = tmp_path / "chunk1_review_stop"
        base_args = [
            "--gold",
            str(gold),
            "--rnd-commit",
            "e48cc90c",
            "--dry",
            "--s2-v11",
            "--arm",
            "mech",
            "--train-seed",
            "45",
            "--g",
            "2",
            "--max-new",
            "8",
            "--state-every-updates",
            "1",
            "--pass-a-jsonl",
            str(pa_path),
            "--pass-a-sha256",
            pa_sha,
            "--pass-b-jsonl",
            str(pb_path),
            "--pass-b-sha256",
            pb_sha,
        ]
        rc1 = _run(
            [*base_args, "--out", str(c1), "--stop-after-updates", "3"],
            build=_non_truncated_dry_bundle,
        )
        assert rc1 == s2.TRAIN_EXIT_CANARY_STOP == 3
        rec1 = _receipt(c1)
        assert rec1["verdict"] == "STOP-AND-REPORT"
        # Update 1 (the 2nd update, 0-indexed step=1) finished and its gradients were applied!
        assert rec1["training"]["next_step"] == 2
        assert rec1["training"]["halted"] is None
        assert rec1["training"]["stopped_for_review"]["reason"] == "canary_stop_and_report"
        assert rec1["stopped_for_review"]["triggered_canaries"] == ["C1_SHADOW_M1_PASS_RATE"]

        # Compare trainable weights at step 1 (0000-step-00001) vs final step 2 (0001-step-00002)
        t_step1 = torch.load(
            c1 / s2.TRAIN_STATE_DIR / "0000-step-00001" / "trainable.pt",
            map_location="cpu",
            weights_only=True,
        )
        t_step2 = torch.load(
            c1 / s2.TRAIN_STATE_DIR / "0001-step-00002" / "trainable.pt",
            map_location="cpu",
            weights_only=True,
        )
        assert any(not torch.equal(t_step1[k], t_step2[k]) for k in t_step1)

        st1 = json.loads(
            (c1 / s2.TRAIN_STATE_DIR / "0001-step-00002" / "state.json").read_text(encoding="utf-8")
        )
        assert st1["halted"] is None
        assert st1["stopped_for_review"]["reason"] == "canary_stop_and_report"
        assert st1["next_step"] == 2

        # Restore normal record_group for resume checks
        monkeypatch.setattr(s2.CtCollapseGuard, "record_group", orig_record_group)

        # 1. Resuming without --resume-after-review refuses with RESUME_FROM_STOPPED_FOR_REVIEW
        with pytest.raises(SystemExit, match="RESUME_FROM_STOPPED_FOR_REVIEW"):
            _run([*base_args, "--out", str(tmp_path / "c2_no_flag"), "--resume-from", str(c1)])

        # 2. Empty reason or missing --resume-from refuses
        with pytest.raises(SystemExit, match="RESUME_AFTER_REVIEW_REASON_REQUIRED"):
            _run(
                [
                    *base_args,
                    "--out",
                    str(tmp_path / "c2_empty"),
                    "--resume-from",
                    str(c1),
                    "--resume-after-review",
                    "   ",
                ]
            )
        with pytest.raises(SystemExit, match="RESUME_AFTER_REVIEW_REQUIRES_RESUME_FROM"):
            _run([*base_args, "--out", str(tmp_path / "c2_no_rf"), "--resume-after-review", "ok"])

        # 3. Resuming with --resume-after-review succeeds and records the reason in both header and s2-hf.json
        c2 = tmp_path / "chunk2_resumed"
        review_msg = "Publisher reviewed C1 streak and approved continuation"
        rc2 = _run(
            [
                *base_args,
                "--out",
                str(c2),
                "--resume-from",
                str(c1),
                "--resume-after-review",
                review_msg,
                "--stop-after-updates",
                "1",
            ],
            build=_non_truncated_dry_bundle,
        )
        assert rc2 == 0
        hdr2 = json.loads((c2 / "s2-hf-train-header.json").read_text(encoding="utf-8"))
        rec2 = _receipt(c2)
        assert hdr2["resume_after_review"] == review_msg
        assert len(hdr2["review_resumes"]) == 1
        assert hdr2["review_resumes"][0]["reason"] == review_msg
        assert hdr2["review_resumes"][0]["step"] == 2
        assert rec2["resume_after_review"] == review_msg
        assert rec2["review_resumes"][0]["reason"] == review_msg
        assert rec2["training"]["resume_after_review"] == review_msg
        assert rec2["training"]["review_resumes"][0]["reason"] == review_msg
        assert rec2["training"]["start_step"] == 2
        assert rec2["training"]["next_step"] == 3

        # 4. Passing --resume-after-review on a normal non-stopped chunk (c2) refuses with RESUME_AFTER_REVIEW_NOT_STOPPED
        with pytest.raises(SystemExit, match="RESUME_AFTER_REVIEW_NOT_STOPPED"):
            _run(
                [
                    *base_args,
                    "--out",
                    str(tmp_path / "c3_bad"),
                    "--resume-from",
                    str(c2),
                    "--resume-after-review",
                    "not stopped",
                ]
            )

    def test_s2_v11_end_to_end_run_steps_rewards_and_advantages_use_m2a(self, tmp_path: Path) -> None:
        """Ask 7(a): End-to-end run_steps test that the rewards used for advantages come from M2a under --s2-v11."""
        passage = "def compute_total(items, tax_rate): return sum(items)"
        quote_m2a_only = "def compute_total ( items , tax_rate ) : return sum ( items )"
        reply_m2a = "<think>ok</think>" + json.dumps(
            {
                "verdict": "supported",
                "evidence_quotes": [{"part": "p1", "verdict": "supported", "quote": quote_m2a_only}],
            }
        )
        reply_bad = "<think>ok</think>" + json.dumps(
            {
                "verdict": "supported",
                "evidence_quotes": [{"part": "p1", "verdict": "supported", "quote": "not in passage at all"}],
            }
        )
        prompt_row = _row("m2a-e2e", label="supported", text=passage)

        def _run_one_step(is_v11: bool):
            a = _ns(_argv(_gold(tmp_path), tmp_path / f"e2e_{is_v11}", steps=1, gpu=1))
            bundle = s2.dry_bundle(a)
            tok = bundle["tok"]
            c0 = tok.encode(reply_m2a)
            c1 = tok.encode(reply_bad)
            p_ids = tok.encode("prompt")

            def custom_rollout(_text, _g, _max_new, _temp):
                return p_ids, [c0, c1], [False, False]

            # Exact reward_fn shape from real_bundle
            quote_mat = s2.S2V11.matcher if is_v11 else "rule2"

            def real_like_reward_fn(text, row, tt, truncated=False, quote_matcher=quote_mat):
                return s2.reward(
                    text,
                    row["label"],
                    s2.passages_of(row),
                    tt,
                    truncated=truncated,
                    quote_reward="v2",
                    quote_matcher=quote_matcher,
                )

            save_recs: list[dict] = []
            rows, stopped, oom = s2.run_steps(
                model=bundle["model"],
                ref_adapter_off=bundle["ref_adapter_off"],
                tok=tok,
                optimiser=bundle["optimiser"],
                prompts=[prompt_row],
                rollout=custom_rollout,
                steps=1,
                g=2,
                max_new=64,
                reward_fn=real_like_reward_fn,
                device_sync=bundle["sync"],
                peak_alloc_mib=bundle["peak_alloc_mib"],
                mode=bundle["mode"],
                state=bundle["state"],
                groups_per_update=1,
                zero_variance_mask=True,
                advantage_mode="dr-grpo",
                mask_truncated=True,
                quote_reward="v2",
                save_all_records=save_recs,
                mask_wrong_positive_advantage=True,
                s2_v11=is_v11,
            )
            assert stopped is None and not oom
            return rows[0], save_recs

        rec_v11, sa_v11 = _run_one_step(True)
        rec_r2, sa_r2 = _run_one_step(False)

        # Under --s2-v11 (M2a), reply_m2a matches (+1.05) and reply_bad fails (-0.20):
        # advantages are [+0.625, -0.625], active_groups == 1, zero_variance_groups == 0!
        assert rec_v11["active_groups"] == 1
        assert rec_v11["zero_variance_groups"] == 0
        assert rec_v11["mean_reward"] == pytest.approx((1.05 + (-0.2)) / 2.0, abs=1e-4)
        assert [r["reward"] for r in sa_v11] == [pytest.approx(1.05), pytest.approx(-0.2)]
        assert [r["advantage"] for r in sa_v11] == [pytest.approx(0.625), pytest.approx(-0.625)]

        # Without --s2-v11 (rule2), BOTH completions get -0.20, so advantages are [0.0, 0.0] and group is zero-variance!
        assert rec_r2["active_groups"] == 0
        assert rec_r2["zero_variance_groups"] == 1
        assert rec_r2["mean_reward"] == pytest.approx(-0.2, abs=1e-4)
        assert [r["reward"] for r in sa_r2] == [pytest.approx(-0.2), pytest.approx(-0.2)]
        assert [r["advantage"] for r in sa_r2] == [pytest.approx(0.0), pytest.approx(0.0)]

    def test_s2_v11_constant_denominator_4514_in_microbatch_backward_and_run_steps(
        self, tmp_path: Path
    ) -> None:
        """Ask 7(b): Verify constant denominator 4,514 is used by grpo_microbatch_backward and run_steps, not active_tokens."""
        a = _ns(_argv(_gold(tmp_path), tmp_path / "denom_test", steps=1, gpu=1))
        bundle_const = s2.dry_bundle(a)
        bundle_active = s2.dry_bundle(a)
        model_c = bundle_const["model"]
        model_a = bundle_active["model"]

        # 1. Direct grpo_microbatch_backward test: loss and parameter grads scale by active_tokens / 4514.0
        prompt_len = 4
        ids = torch.tensor([[2, 3, 4, 5, 10, 11, 12, 13, 14, 15], [2, 3, 4, 5, 16, 17, 18, 19, 0, 0]])
        att = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]])
        mask = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]])
        active_tokens = float(mask.sum().item())  # 10.0 tokens
        adv = torch.tensor([0.5, -0.5])

        with torch.no_grad():
            old_c = s2.completion_logprobs_microbatch(model_c, ids, att, prompt_len, 1)
            with bundle_const["ref_adapter_off"]():
                ref_c = s2.completion_logprobs_microbatch(model_c, ids, att, prompt_len, 1)

        bundle_const["optimiser"].zero_grad(set_to_none=True)
        loss_const = s2.grpo_microbatch_backward(
            model_c,
            ids,
            att,
            prompt_len,
            old_c,
            ref_c,
            adv,
            mask,
            microbatch=1,
            scale=1.0,
            zero_variance_mask=True,
            group_size=2,
            dual_clip_c=3.0,
            constant_loss_denominator=4514,
        )
        grads_const = {
            n: p.grad.detach().clone()
            for n, p in model_c.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        bundle_active["optimiser"].zero_grad(set_to_none=True)
        loss_active = s2.grpo_microbatch_backward(
            model_a,
            ids,
            att,
            prompt_len,
            old_c.clone(),
            ref_c.clone(),
            adv,
            mask,
            microbatch=1,
            scale=1.0,
            zero_variance_mask=True,
            group_size=2,
            dual_clip_c=3.0,
            constant_loss_denominator=None,
        )
        grads_active = {
            n: p.grad.detach().clone()
            for n, p in model_a.named_parameters()
            if p.requires_grad and p.grad is not None
        }

        expected_ratio = active_tokens / 4514.0
        assert float(loss_const.item()) == pytest.approx(float(loss_active.item()) * expected_ratio, rel=1e-5)
        for name in grads_const:
            assert torch.allclose(
                grads_const[name], grads_active[name] * expected_ratio, rtol=1e-5, atol=1e-7
            )

        # 2. End-to-end run_steps test (both microbatch=1 and microbatch=None)
        for mb in (1, None):
            b_v11 = s2.dry_bundle(a)
            b_def = s2.dry_bundle(a)
            p_ids = [2, 3, 4, 5]
            c_seqs = [[10, 11, 12, 13, 14, 15], [16, 17, 18, 19]]

            def fixed_rollout(_text, _g, _max_new, _temp):
                return p_ids, c_seqs, [False, False]

            # Perturb trainable parameters away from zero so old != ref and KL/policy loss is non-zero
            for m in (b_v11["model"], b_def["model"]):
                with torch.no_grad():
                    for p in m.parameters():
                        if p.requires_grad:
                            p.add_(0.05)

            rows_v11, _, _ = s2.run_steps(
                model=b_v11["model"],
                ref_adapter_off=b_v11["ref_adapter_off"],
                tok=b_v11["tok"],
                optimiser=b_v11["optimiser"],
                prompts=[_row("d0")],
                rollout=fixed_rollout,
                steps=1,
                g=2,
                max_new=16,
                reward_fn=lambda txt, r, tt, truncated=False, **kw: 1.0 if len(txt) > 4 else -0.2,
                device_sync=b_v11["sync"],
                peak_alloc_mib=b_v11["peak_alloc_mib"],
                mode=b_v11["mode"],
                state=b_v11["state"],
                logprob_microbatch=mb,
                groups_per_update=1,
                zero_variance_mask=True,
                advantage_mode="dr-grpo",
                mask_truncated=True,
                quote_reward="v2",
                mask_wrong_positive_advantage=True,
                s2_v11=True,
            )
            rows_def, _, _ = s2.run_steps(
                model=b_def["model"],
                ref_adapter_off=b_def["ref_adapter_off"],
                tok=b_def["tok"],
                optimiser=b_def["optimiser"],
                prompts=[_row("d0")],
                rollout=fixed_rollout,
                steps=1,
                g=2,
                max_new=16,
                reward_fn=lambda txt, r, tt, truncated=False, **kw: 1.0 if len(txt) > 4 else -0.2,
                device_sync=b_def["sync"],
                peak_alloc_mib=b_def["peak_alloc_mib"],
                mode=b_def["mode"],
                state=b_def["state"],
                logprob_microbatch=mb,
                groups_per_update=1,
                zero_variance_mask=True,
                advantage_mode="dr-grpo",
                mask_truncated=True,
                quote_reward="v2",
                mask_wrong_positive_advantage=True,
                s2_v11=False,
            )
            assert rows_def[0]["loss"] != 0.0
            assert rows_v11[0]["loss"] == pytest.approx(
                rows_def[0]["loss"] * (10.0 / 4514.0), rel=1e-3, abs=1e-6
            )

    def test_s2_v11_cusum_exposure_scaling_zero_exposure_and_m1_nonevents(self) -> None:
        """Ask 7(c): CUSUM exposure with E_w != E_0 and E_w = 0, plus m = 1 non-events for both channels
        (review worked case: S1 = 2.788918, S2 = 2.788918 unchanged, S3 = 3.626547 trips)."""
        c_ids = tuple(f"c{i}" for i in range(128))
        u_extra = tuple(f"u_extra_{i}" for i in range(56))
        u_ids = c_ids + u_extra
        assert len(c_ids) == 128 and len(u_ids) == 184

        step0 = {uid: {"verdict": "unsupported", "correct": True} for uid in u_ids}
        confirm_calls: list[int] = []

        def confirm_runner(ids, step=0, rule=""):
            confirm_calls.append(len(ids))
            # Return 0 CT flips so Fuse A logs CONFIRM-NEGATIVE after tripping in Window 3
            return {cid: {"verdict": "unsupported", "correct": True} for cid in ids}

        guard = s2.CtCollapseGuard(
            window_groups=40,
            s2_v11=True,
            canary_c_ids=c_ids,
            gold_unsupported_ids=u_ids,
            confirm_runner=confirm_runner,
        )
        guard.record_step0_baseline(step0)

        # Window 1 (40 groups):
        # - E_1 = 13 groups on C:
        #   * 3 groups with m = 2 cannot_tell (count toward B_1 -> B_1 = 3)
        #   * 1 group with m = 1 cannot_tell (must NOT count toward B_1!)
        #   * 9 groups with m = 0 cannot_tell
        # - 1 group on u_extra_0 (gold-unsupported not in C) with m = 1 supported (must NOT count toward FA2_1!)
        # - 26 groups on gold-supported item 'sup_0'
        for g_idx in range(40):
            if g_idx < 3:
                item_id, gold_v = c_ids[g_idx], "unsupported"
                verdicts = ("cannot_tell", "cannot_tell") + ("unsupported",) * 6
            elif g_idx == 3:
                # m = 1 cannot_tell on C -> non-event for Fuse A!
                item_id, gold_v = c_ids[g_idx], "unsupported"
                verdicts = ("cannot_tell",) + ("unsupported",) * 7
            elif g_idx < 13:
                item_id, gold_v = c_ids[g_idx], "unsupported"
                verdicts = ("unsupported",) * 8
            elif g_idx == 13:
                # m = 1 supported on gold-unsupported -> non-event for Channel B!
                item_id, gold_v = u_extra[0], "unsupported"
                verdicts = ("supported",) + ("unsupported",) * 7
            else:
                item_id, gold_v = "sup_0", "supported"
                verdicts = ("supported",) * 8
            flags = tuple(v == gold_v for v in verdicts)
            advs = tuple(0.0 if not f else 0.1 for f in flags)
            w1 = guard.record_group(
                step=g_idx // 4,
                group=g_idx % 4,
                item_id=item_id,
                gold_verdict=gold_v,
                verdicts=verdicts,
                correct_flags=flags,
                advantages_list=advs,
            )
        assert w1 is None
        assert confirm_calls == []
        w1_rec = guard.window_history[-1]
        assert w1_rec["fuse_a_m_w"] == 13
        assert w1_rec["fuse_a_b_w"] == 3  # m=1 did NOT count
        assert w1_rec["channel_b_n_w"] == 14
        assert w1_rec["channel_b_fa2_w"] == 0  # m=1 did NOT count
        assert round(guard.fuse_a_s, 6) == pytest.approx(2.788918, abs=1e-6)
        assert guard.channel_b_s == 0.0

        # Window 2 (40 groups): E_2 = 0 groups on C and 0 groups on gold-unsupported
        for g_idx in range(40):
            w2 = guard.record_group(
                step=10 + g_idx // 4,
                group=g_idx % 4,
                item_id="sup_0",
                gold_verdict="supported",
                verdicts=("supported",) * 8,
                correct_flags=(True,) * 8,
                advantages_list=(0.1,) * 8,
            )
        assert w2 is None
        assert confirm_calls == []
        w2_rec = guard.window_history[-1]
        assert w2_rec["fuse_a_m_w"] == 0
        assert round(guard.fuse_a_s, 6) == pytest.approx(2.788918, abs=1e-6)  # unchanged!

        # Window 3 (40 groups): E_3 = 10 groups on C, with B_3 = 1 group having m >= 2 cannot_tell
        # Plus 10 groups on u_extra (gold-unsupported not in C) with 3 groups having m = 2 supported (FA2 = 3)
        # so E_{B,3} = 20 != E_{0,B} (18.680203)
        for g_idx in range(40):
            if g_idx == 0:
                item_id, gold_v = c_ids[0], "unsupported"
                verdicts = ("cannot_tell", "cannot_tell") + ("unsupported",) * 6
            elif g_idx < 10:
                item_id, gold_v = c_ids[g_idx], "unsupported"
                verdicts = ("unsupported",) * 8
            elif g_idx < 13:
                item_id, gold_v = u_extra[g_idx - 10], "unsupported"
                verdicts = ("supported", "supported") + ("unsupported",) * 6
            elif g_idx < 20:
                item_id, gold_v = u_extra[g_idx - 10], "unsupported"
                verdicts = ("unsupported",) * 8
            else:
                item_id, gold_v = "sup_0", "supported"
                verdicts = ("supported",) * 8
            flags = tuple(v == gold_v for v in verdicts)
            advs = tuple(0.0 if not f else 0.1 for f in flags)
            w3 = guard.record_group(
                step=20 + g_idx // 4,
                group=g_idx % 4,
                item_id=item_id,
                gold_verdict=gold_v,
                verdicts=verdicts,
                correct_flags=flags,
                advantages_list=advs,
            )
        assert w3 is None  # confirmation ran and returned CONFIRM-NEGATIVE
        assert confirm_calls == [128]  # Fuse A tripped and ran T=0 pass on C!
        assert guard.fuse_a_s == pytest.approx(3.626547, abs=1e-6)
        # Channel B exposure scaling with E_{B,3} = 20 != E_{0,B}:
        expected_sb = max(0.0, 3.0 - 5.03 * (20.0 / (40.0 * 184.0 / 394.0)))
        assert guard.channel_b_s == pytest.approx(expected_sb, abs=1e-9)

    def test_s2_v11_fuse_a_confirm_ct_step0_1_ct_now_2_is_confirm_negative(self) -> None:
        """Ask 7(d): Fuse A confirmation with ct_step0 = 1, ct_now = 2 -> CONFIRM-NEGATIVE (ct_inc = 1 < 2)."""
        c_ids = tuple(f"c{i}" for i in range(13))
        # Step 0 has ct_step0 = 1 (c0 is cannot_tell, c1..c12 are unsupported)
        step0 = {
            cid: {
                "verdict": "cannot_tell" if idx == 0 else "unsupported",
                "correct": idx != 0,
            }
            for idx, cid in enumerate(c_ids)
        }
        state_now = {"ct_now": 2}

        def confirm_runner(ids, step=0, rule=""):
            return {
                cid: {
                    "verdict": "cannot_tell" if idx < state_now["ct_now"] else "unsupported",
                    "correct": idx >= state_now["ct_now"],
                }
                for idx, cid in enumerate(ids)
            }

        guard = s2.CtCollapseGuard(
            window_groups=13,
            s2_v11=True,
            canary_c_ids=c_ids,
            gold_unsupported_ids=c_ids,
            confirm_runner=confirm_runner,
        )
        s0_rec = guard.record_step0_baseline(step0)
        assert s0_rec["canary_c_cannot_tell_step0"] == 1

        # Window 1: B_w = 4 trips Fuse A; confirmation sees ct_now = 2 ( >= 2 ), but ct_inc = 2 - 1 = 1 < 2!
        # Must be CONFIRM-NEGATIVE (w1 is None)!
        for g_idx in range(13):
            verdicts = (
                ("cannot_tell", "cannot_tell", "unsupported", "unsupported")
                if g_idx < 4
                else ("unsupported",) * 4
            )
            flags = tuple(v == "unsupported" for v in verdicts)
            w1 = guard.record_group(
                step=1,
                group=g_idx,
                item_id=c_ids[g_idx],
                gold_verdict="unsupported",
                verdicts=verdicts,
                correct_flags=flags,
                advantages_list=tuple(0.0 if not f else 0.1 for f in flags),
            )
        assert w1 is None
        conf1 = [d for d in guard.decision_history if d.get("kind") == "guard_confirmation"][-1]
        assert conf1["status"] == "CONFIRM-NEGATIVE"
        assert conf1["verdict"] == "CONTINUE"
        assert (conf1["ct_step0"], conf1["ct_now"], conf1["ct_increase"]) == (1, 2, 1)

        # Window 2: ct_now = 3 -> ct_inc = 3 - 1 = 2 >= 2 -> FUSE_A-CONFIRMED!
        state_now["ct_now"] = 3
        for g_idx in range(13):
            verdicts2 = (
                ("cannot_tell", "cannot_tell", "unsupported", "unsupported")
                if g_idx < 4
                else ("unsupported",) * 4
            )
            flags2 = tuple(v == "unsupported" for v in verdicts2)
            w2 = guard.record_group(
                step=2,
                group=g_idx,
                item_id=c_ids[g_idx],
                gold_verdict="unsupported",
                verdicts=verdicts2,
                correct_flags=flags2,
                advantages_list=tuple(0.0 if not f else 0.1 for f in flags2),
            )
        assert w2 is not None
        assert w2["triggered_rules"] == ["FUSE_A-CONFIRMED"]
        conf2 = [d for d in guard.decision_history if d.get("kind") == "guard_confirmation"][-1]
        assert conf2["status"] == "CONFIRM-POSITIVE"
        assert (conf2["ct_step0"], conf2["ct_now"], conf2["ct_increase"]) == (1, 3, 2)
