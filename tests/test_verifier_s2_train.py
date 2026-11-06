"""The S2 training run: resume across chunks (B1), the truncation halt (B2), seeding (B3), the critic arm (B4),
the guard history (B5) and the training verdict (B6).

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


# -- fixtures ----------------------------------------------------------------------------------------------------------
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
    sdir = Path(out) / s2.TRAIN_STATE_DIR
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


# -- B1: resume across chunks ------------------------------------------------------------------------------------------
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
        assert [c["records"] for c in chain] == [8, 8]

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
        opt = c1 / s2.TRAIN_STATE_DIR / "optimizer.pt"
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


# -- B2: the truncation halt -------------------------------------------------------------------------------------------
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
        # update 0: 1 of 4 truncated (25%); update 1: 2 of 4 (50%); window 2, line 0.30 -> fires after update 1
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
        assert _run(_argv(gold, c1, "--stop-after-updates", "1", steps=4, window=2, halt=0.30), build=build2) == 0
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
        assert s2.train_verdict_line(
            stopped={**ct, "also_triggered": {"reason": "trunc_halt"}},
            oom=False,
            next_step=3,
            steps_total=10,
            chunk_stop=None,
            fits=True,
            dry=True,
        ) == "S2-TRAIN-HALT-S1"


# -- B3: seeding -----------------------------------------------------------------------------------------------------
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


# -- B4: the critic arm ------------------------------------------------------------------------------------------------
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
        assert all(r["in_gradient"] is False and r["hint"].startswith("Hint from a tool check") for r in retries)
        assert {r["seed"] for r in retries} == {
            s2.train_rollout_seed(42, st, gr, purpose="critic-retry") for st in (0, 1) for gr in (0, 1)
        }
        rec = _receipt(critic)
        assert rec["training"]["critic"]["hinted_groups"] == 4 and rec["training"]["critic"]["signal_drops"] == 0
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


class TestMechArmMatchesTheBase:
    """Rewards and advantages on fixed replies with the registered reward (v2, dr-grpo, masks on). The golden
    values were produced by aspire-si 7f9b4cb3's run_steps on this exact scenario (counts only)."""

    GOLDEN = {
        "rewards": [
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
        ],
    }

    def test_rewards_and_advantages_equal_the_base_commit(self, tmp_path):
        a = _ns(_argv(_gold(tmp_path), tmp_path / "o", gpu=2, steps=2))
        bundle = s2.dry_bundle(a)
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
            steps=2,
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


# -- B5: the guard history ---------------------------------------------------------------------------------------------
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
        second = s2.CtCollapseGuard.from_state(
            json.loads(json.dumps(first.to_state())), pass_a_k={"i00": 8}
        )
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


# -- B6: the training verdict ------------------------------------------------------------------------------------------
class TestTrainingVerdict:
    @pytest.mark.parametrize(
        ("kw", "line", "code"),
        [
            ({"next_step": 197, "chunk_stop": None}, "S2-TRAIN-COMPLETE", 0),
            ({"next_step": 40, "chunk_stop": {"next_step": 40}}, "S2-TRAIN-CHUNK-OK", 0),
            ({"next_step": 39, "chunk_stop": None}, "S2-TRAIN-FAIL", 1),
            ({"next_step": 12, "stopped": {"reason": "trunc_halt", "triggered_rules": ["TRUNC"]}}, "S2-TRAIN-HALT-TRUNC", 2),
            ({"next_step": 12, "stopped": {"reason": "ct_guard_u2", "triggered_rules": ["U2"]}}, "S2-TRAIN-HALT-U2", 2),
            ({"next_step": 12, "stopped": {"error": "non-finite loss"}}, "S2-TRAIN-FAIL", 1),
            ({"next_step": 12, "oom": True}, "S2-TRAIN-FAIL", 1),
            ({"next_step": 197, "fits": False, "dry": False}, "S2-TRAIN-FAIL", 1),
            ({"next_step": 197, "fits": False, "dry": True}, "S2-TRAIN-COMPLETE", 0),
        ],
    )
    def test_verdicts_and_exit_codes(self, kw, line, code):
        args = {"stopped": None, "oom": False, "steps_total": 197, "chunk_stop": None, "fits": True, "dry": False}
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
