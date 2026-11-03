"""Smoke test S2's parts: the prompt rule, the reward, the pass line, the guards, the receipt, and a CPU rehearsal.

Everything but the last group is plain Python and runs on any interpreter (torch is imported lazily by the
script). The tests marked ``needs_torch`` run the tiny-model rehearsal; they are skipped only when torch is not
installed at all. A torch that is installed but broken fails them (the import error is not swallowed), and
``S2_REQUIRE_TORCH=1`` turns a missing torch into a failure instead of a skip. Under the repo's conftest, which
imports torch, they always run; on a Python without a torch wheel, run with ``--noconftest``.
"""

import importlib
import importlib.util
import itertools
import json
import os
import re
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "verifier-distill"))

import s2_grpo_smoke as s2  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch is not installed in this Python")


# -- fixtures ----------------------------------------------------------------------------------------------------------------
def _row(cid, split="tune", label="supported", claim="Claim.", text="alpha beta\ngamma"):
    return {
        "id": cid,
        "check_type": "grounded",
        "claim": claim,
        "label": label,
        "split": split,
        "context": [{"source": f"src@{cid}", "text": text}],
    }


def _gold(tmp_path, rows_by_file=None):
    rows_by_file = rows_by_file or {
        "grounded.jsonl": [_row(f"g{i:02d}") for i in range(20)],
        "prs/grounded-prs.jsonl": [_row(f"p{i:02d}") for i in range(20)] + [_row("h1", split="heldout")],
        "diffs/reasoning-diffs.jsonl": [_row(f"r{i:02d}", label="unsupported") for i in range(20)],
    }
    for name, rows in rows_by_file.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return tmp_path


def _args(tmp_path, gold, **kw):
    base = dict(
        path="hf",
        steps=s2.STEPS,
        prompts=s2.PROMPTS,
        g=s2.G,
        max_new=s2.MAX_NEW_TOKENS,
        gold=gold,
        rnd_commit="e48cc90c",
        out=tmp_path / "out",
        dry=False,
        revision=s2.STUDENT_REVISION,
        heartbeat=None,
        llama_server=None,
        gguf=None,
        convert_script=None,
        timeout=10.0,
        gen_timeout=120.0,
    )
    return Namespace(**{**base, **kw})


def _steps(n=20, **kw):
    row = {
        "step": 0,
        "groups": 2,
        "zero_variance_groups": 0,
        "tokens": 800,
        "rollout_s": 4.0,
        "train_s": 6.0,
        "step_s": 10.0,
        "peak_alloc_mib": 20000,
    }
    return [{**row, "step": i, **kw} for i in range(n)]


def _result(steps=None, peak=25000, requested=20, stopped=None, oom=False, cut=False, dry=False):
    return s2.readout(
        _steps() if steps is None else steps,
        peak_mib=peak,
        steps_requested=requested,
        stopped=stopped,
        oom=oom,
        cut=cut,
        dry=dry,
    )


# -- the prompt rule -------------------------------------------------------------------------------------------------------
class TestPromptRule:
    def test_the_first_32_tune_claims_by_sorted_id_whatever_the_file_order(self, tmp_path):
        rows, hashes = s2.read_gold(_gold(tmp_path))
        picked = s2.select_prompts(rows)
        assert [r["id"] for r in picked] == sorted(r["id"] for r in rows if r["split"] == "tune")[:32]
        assert len(picked) == 32 and all(r["split"] == "tune" for r in picked)
        assert [r["id"] for r in s2.select_prompts(list(reversed(rows)))] == [r["id"] for r in picked]
        assert sorted(hashes) == sorted(s2.GOLD_FILES)

    def test_a_gold_set_with_too_few_tune_claims_stops_the_run(self, tmp_path):
        rows, _ = s2.read_gold(_gold(tmp_path))
        with pytest.raises(SystemExit, match="needs 32"):
            s2.select_prompts([r for r in rows if r["id"] < "p"], 32)

    def test_the_prompt_carries_the_contract_the_evidence_and_the_claim(self):
        text = s2.build_prompt(_row("a", claim="It adds one.", text="x = x + 1"))
        assert (
            "evidence_quotes" in text
            and "src@a" in text
            and "x = x + 1" in text
            and "Claim: It adds one." in text
        )

    def test_gold_hashes_are_the_sha256_of_the_bytes_parsed(self, tmp_path):
        import hashlib

        gold = _gold(tmp_path)
        _, hashes = s2.read_gold(gold)
        for name, digest in hashes.items():
            assert digest == hashlib.sha256((gold / name).read_bytes()).hexdigest()


# -- the reward --------------------------------------------------------------------------------------------------------------
def _reply(verdict="supported", quote="alpha beta", parts=None, think="short"):
    parts = parts if parts is not None else [{"part": "p", "verdict": verdict, "quote": quote}]
    return f"<think>{think}</think>" + json.dumps({"verdict": verdict, "evidence_quotes": parts})


EVIDENCE = "[src]\nalpha   beta\ngamma"


class TestReward:
    def test_parse_reply_accepts_contract_v3_and_refuses_the_rest(self):
        assert s2.parse_reply(_reply())["verdict"] == "supported"
        for bad in (
            "no json at all",
            "{",
            json.dumps({"verdict": "yes", "evidence_quotes": []}),
            json.dumps({"verdict": "supported", "evidence_quotes": []}),  # 0 parts
            json.dumps(
                {"verdict": "supported", "evidence_quotes": [{"part": "p", "verdict": "x", "quote": "q"}]}
            ),
            json.dumps(
                {"verdict": "supported", "evidence_quotes": [{"part": "p", "verdict": "supported"}] * 5}
            ),
            json.dumps([1, 2]),
        ):
            assert s2.parse_reply(bad) is None

    def test_format_gate_macro_outcomes_and_the_quote_rule(self):
        r = s2.reward
        assert r("garbage", "supported", EVIDENCE, 5) == -1.0  # the format gate
        assert r(_reply("supported", "alpha beta"), "supported", EVIDENCE, 5) == pytest.approx(
            1.05
        )  # + a found quote
        assert (
            r(_reply("supported", "not in the evidence"), "supported", EVIDENCE, 5) == -3.0
        )  # a false accept
        assert r(_reply("supported", "alpha beta"), "unsupported", EVIDENCE, 5) == pytest.approx(
            -3.0
        )  # false accept
        assert r(_reply("unsupported", "alpha beta"), "supported", EVIDENCE, 5) == pytest.approx(
            -1.0
        )  # false reject
        assert r(_reply("cannot_tell", "alpha beta"), "supported", EVIDENCE, 5) == pytest.approx(-0.3)
        assert r(_reply("unsupported", "alpha beta"), "cannot_tell", EVIDENCE, 5) == pytest.approx(
            1.05
        )  # also counts
        assert r(_reply("supported"), "supported", EVIDENCE, 8192) == -1.0  # the hard cap
        assert r(_reply("supported"), "supported", EVIDENCE, 5, truncated=True) == -1.0

    def test_quotes_match_through_whitespace_normalisation(self):
        assert s2.reward(_reply("supported", "alpha beta gamma"), "supported", EVIDENCE, 5) == pytest.approx(
            1.05
        )

    def test_consistency_length_and_loop_terms(self):
        r = s2.reward
        parts = [
            {"part": "a", "verdict": "supported", "quote": "alpha beta"},
            {"part": "b", "verdict": "unsupported", "quote": "gamma"},
        ]
        # the claim says supported but a part is unsupported: the consistency penalty (-0.5), and the false accept
        assert r(_reply("supported", parts=parts), "unsupported", EVIDENCE, 5) == pytest.approx(-3.5)
        assert r(_reply("unsupported", parts=parts), "unsupported", EVIDENCE, 5) == pytest.approx(1.1)
        # the length penalty on a correct sample: 1,536 thinking tokens over a 1,024 budget is -0.05, capped at -0.5
        assert r(_reply("supported"), "supported", EVIDENCE, 1536) == pytest.approx(1.0 - 0.05 + 0.05)
        assert r(_reply("supported"), "supported", EVIDENCE, 8000) == pytest.approx(1.0 - 0.5 + 0.05)
        looped = _reply("supported", think=("one two three four five six seven eight " * 41))
        assert s2.loop_detected(looped) and not s2.loop_detected("a b c d e f g h")
        assert r(looped, "supported", EVIDENCE, 5) == pytest.approx(
            -0.3 - 0.5
        )  # read as cannot_tell, plus the loop

    def test_the_plans_invariants_hold_over_every_case(self):
        """No wrong verdict scores above -0.3; every correct one scores at least 0.0; extras never exceed +0.3."""
        verdicts = ("supported", "unsupported", "cannot_tell")
        evidence_quotes = ("alpha beta", "absent text")
        for gold, claim_v, n_parts, quote, think, loop in itertools.product(
            verdicts,
            verdicts,
            (1, 2, 4),
            evidence_quotes,
            (0, 300, 1100, 5000, 8000),
            (False, True),
        ):
            for part_vs in itertools.product(verdicts, repeat=min(n_parts, 2)):
                parts = [
                    {"part": str(i), "verdict": part_vs[i % len(part_vs)], "quote": quote}
                    for i in range(n_parts)
                ]
                text = _reply(claim_v, parts=parts, think=("a b c d e f g h " * 41) if loop else "t")
                value = s2.reward(text, gold, EVIDENCE, think)
                looped = loop
                eff = "cannot_tell" if looped else claim_v
                _, correct = s2.macro(eff, gold)
                unfound_accept = eff == "supported" and any(
                    p["verdict"] == "supported" and p["quote"] not in "alpha beta gamma" for p in parts
                )
                if correct and not unfound_accept:
                    assert value >= 0.0, (gold, claim_v, part_vs, quote, think, loop, value)
                else:
                    assert value <= -0.3 + 1e-9, (gold, claim_v, part_vs, quote, think, loop, value)
                assert value <= 1.0 + 0.3 + 1e-9

    def test_advantages_are_group_normalised_and_a_flat_group_is_zero_variance(self):
        adv = s2.advantages([1.0, 0.0, 1.0, 0.0])
        assert sum(adv) == pytest.approx(0.0) and adv[0] > 0 > adv[1]
        assert s2.advantages([0.5] * 8) == [0.0] * 8
        assert s2.zero_variance([0.5] * 8) and not s2.zero_variance([0.5, 0.6])


# -- the pass line -----------------------------------------------------------------------------------------------------------
class TestPassLine:
    def test_a_whole_run_that_fits_passes(self):
        res = _result()
        assert res["pass"] is True and res["complete"] is True
        assert s2.verdict_line(res, stopped=None, cut=False, dry=False) == "S2-OK"
        assert s2.exit_status("S2-OK") == 0

    def test_peak_vram_must_be_under_the_line(self):
        assert _result(peak=s2.VRAM_LINE_MIB - 1)["pass"] is True
        over = _result(peak=s2.VRAM_LINE_MIB)  # at the line is not under it
        assert over["pass"] is False and over["fits"] is False
        assert s2.verdict_line(over, stopped=None, cut=False, dry=False) == "S2-FAIL"
        assert _result(peak=None)["pass"] is False  # an unread sampler proves nothing

    def test_19_of_20_steps_is_not_a_pass(self):
        res = _result(steps=_steps(19))
        assert res["pass"] is False and res["complete"] is False
        assert s2.verdict_line(res, stopped=None, cut=False, dry=False) == "S2-FAIL"

    def test_a_steps_cut_reads_limited_never_ok_and_exits_non_zero(self):
        res = _result(steps=_steps(5), requested=5, cut=True)  # every requested step ran and fit
        assert res["pass"] is False and res["complete"] is False
        line = s2.verdict_line(res, stopped=None, cut=True, dry=False)
        assert line == "S2-LIMITED" and s2.exit_status(line) == 1

    def test_a_prompts_or_g_cut_at_20_steps_is_limited_too(self):
        res = _result(
            steps=_steps(20), cut=True
        )  # --prompts 8 or --g 4: every step ran and fit, at another shape
        assert res["pass"] is False and res["complete"] is False
        assert s2.verdict_line(res, stopped=None, cut=True, dry=False) == "S2-LIMITED"

    def test_a_dry_run_that_somehow_has_20_steps_still_never_passes(self):
        res = _result(steps=_steps(20), dry=True)
        assert res["pass"] is False and res["complete"] is False

    def test_the_oom_flag_alone_fails_a_run_whose_steps_all_counted(self):
        res = _result(steps=_steps(20), oom=True)
        assert res["pass"] is False and res["complete"] is False

    def test_a_stop_reads_stopped_never_ok_and_exits_non_zero(self):
        stop = {"step": 7, "error": "timeout: /completion did not answer in 120s"}
        res = _result(steps=_steps(7), stopped=stop)
        assert res["pass"] is False
        line = s2.verdict_line(res, stopped=stop, cut=False, dry=False)
        assert line == "S2-STOPPED" and s2.exit_status(line) == 1

    def test_an_oom_is_a_failed_measurement_and_never_a_pass(self):
        res = _result(steps=_steps(3), oom=True, stopped={"step": 3, "error": "out of memory"})
        assert res["oom"] is True and res["pass"] is False

    def test_a_dry_rehearsal_never_passes_and_exits_zero_when_it_finishes(self):
        res = _result(steps=_steps(2), requested=2, peak=None, dry=True)
        assert res["pass"] is False
        line = s2.verdict_line(res, stopped=None, cut=True, dry=True)
        assert line == "S2-DRY" and s2.exit_status(line) == 0
        broken = s2.verdict_line(
            _result(steps=_steps(1), requested=2, dry=True), stopped=None, cut=True, dry=True
        )
        assert broken == "S2-STOPPED"

    def test_the_readout_reports_speed_and_zero_variance_groups(self):
        res = _result(steps=_steps(20, zero_variance_groups=1))
        assert res["rollout_tok_s"] == 200.0 and res["step_s"] == {"p50": 10.0, "max": 10.0}
        assert (
            res["zero_variance_groups"] == 20 and res["groups"] == 40 and res["peak_vram_mib_torch"] == 20000
        )


# -- the guards --------------------------------------------------------------------------------------------------------------
class TestGuards:
    def test_only_the_single_nvidia_gpu_is_accepted(self):
        gpu = {"name": "NVIDIA GeForce RTX 5090", "driver": "590.1"}
        assert s2.require_single_nvidia_gpu([gpu]) == gpu
        with pytest.raises(SystemExit, match="2 GPUs"):
            s2.require_single_nvidia_gpu([gpu, {"name": "NVIDIA RTX A2000", "driver": "590.1"}])
        with pytest.raises(SystemExit, match="not an NVIDIA"):
            s2.require_single_nvidia_gpu([{"name": "Intel(R) Graphics", "driver": "1"}])
        with pytest.raises(SystemExit, match="0 GPUs"):
            s2.require_single_nvidia_gpu([])

    def test_another_process_on_the_card_refuses_the_start(self):
        s2.require_exclusive([])
        with pytest.raises(SystemExit, match="1 other process"):
            s2.require_exclusive(["4242"])

    def test_the_watchdog_heartbeat_must_exist_and_be_recent(self):
        s2.require_heartbeat(1000.0, 1030.0)
        for mtime in (None, 900.0):
            with pytest.raises(SystemExit, match="refusing to start"):
                s2.require_heartbeat(mtime, 1030.0)

    def test_a_real_run_without_the_rig_stops_before_any_model_is_built(self, tmp_path):
        built = []

        class Rig:
            def preflight(self):
                raise SystemExit("refusing to start: no watchdog heartbeat")

        with pytest.raises(SystemExit, match="no watchdog heartbeat"):
            s2.execute(_args(tmp_path, _gold(tmp_path)), build=lambda a: built.append(a), rig=Rig())
        assert built == []


# -- the sampler and the receipt -----------------------------------------------------------------------------------------
class TestSamplerAndReceipt:
    def test_the_sampler_keeps_the_peak_and_ignores_a_failed_read(self):
        readings = iter([9000, None, 31000, 12000])
        sampler = s2.VramSampler(lambda: next(readings, None))
        for _ in range(4):
            sampler.sample_once()
        assert sampler.peak_mib == 31000 and sampler.samples == 3

    def test_the_sampler_runs_in_a_thread_and_takes_a_last_reading(self):
        calls = []
        with s2.VramSampler(lambda: calls.append(1) or 5000, period_s=0.01) as sampler:
            pass
        assert sampler.peak_mib == 5000 and len(calls) >= 2

    def test_the_receipt_carries_the_gold_hashes_the_rnd_commit_and_every_environment_field(self, tmp_path):
        rows, hashes = s2.read_gold(_gold(tmp_path))
        env = s2.environment(
            gpu={"name": "NVIDIA GeForce RTX 5090", "driver": "590.1"},
            dtype="bf16",
            batch=8,
            exclusive=True,
            cuda="13.0",
        )
        assert set(env) == set(s2.ENVIRONMENT_FIELDS)
        assert (
            env["gpu_name"] == "NVIDIA GeForce RTX 5090"
            and env["driver"] == "590.1"
            and env["cuda"] == "13.0"
        )
        assert env["python"] == sys.version.split()[0] and env["gpu_exclusive"] is True
        receipt = s2.build_receipt(
            path="hf",
            dry=False,
            args={"steps": 20},
            rnd_commit="e48cc90c",
            gold_hashes=hashes,
            prompt_ids=[r["id"] for r in s2.select_prompts(rows)],
            env=env,
            revision="b968826d9c46aa",
            llama=None,
            result=_result(),
            stopped=None,
            verdict="S2-OK",
        )
        assert receipt["gold_files_sha256"] == hashes and len(receipt["gold_files_sha256"]) == 3
        assert receipt["rnd_commit"] == "e48cc90c" and set(receipt["environment"]) == set(
            s2.ENVIRONMENT_FIELDS
        )
        assert receipt["registered"]["G"] == 8 and receipt["registered"]["beta"] == 0.04
        assert receipt["registered"]["lora"] == {
            "r": 16,
            "alpha": 32,
            "targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        }
        assert (
            receipt["pass_line"]["vram_line_mib"] == 31200
            and receipt["revision_resolved"] == "b968826d9c46aa"
        )
        assert len(receipt["prompt_ids"]) == 32 and receipt["verdict"] == "S2-OK"

    def test_the_llamacpp_rollout_reloads_the_adapter_with_the_control_timeout_and_samples_g_completions(
        self,
    ):
        calls = []

        def post(path, payload, timeout_s):
            calls.append((path, timeout_s))
            return {"tokens": [5, 6, 7]}

        class Tok:
            def __call__(self, text):
                return {"input_ids": [1, 2]}

        roll = s2.LlamaCppRollout(post, lambda step: Path(f"adapter-{step}.gguf"), Tok(), 10.0, 120.0)
        roll.reload(3)
        ids, comps = roll("prompt", 4, 64, 0.7)
        assert calls[0] == ("/lora-adapters", 10.0) and [c[1] for c in calls[1:]] == [120.0] * 4
        assert ids == [1, 2] and comps == [[5, 6, 7]] * 4

    def test_the_dry_rehearsal_takes_the_hugging_face_path_only(self):
        with pytest.raises(SystemExit, match="Hugging Face"):
            s2.apply_dry_defaults(Namespace(path="llamacpp", steps=20, g=8, max_new=2048))
        a = s2.apply_dry_defaults(Namespace(path="hf", steps=20, g=8, max_new=2048))
        assert (a.steps, a.g, a.max_new) == (2, 2, 24)
        a = s2.apply_dry_defaults(Namespace(path="hf", steps=3, g=4, max_new=100))
        assert (a.steps, a.g, a.max_new) == (3, 4, 100)


# -- torch: lazy, and not allowed to hide a real import error ---------------------------------------------------------
class TestLazyTorch:
    def test_importing_the_script_does_not_import_torch(self):
        import subprocess

        here = str(Path(__file__).parent.parent / "examples" / "verifier-distill")
        code = f"import sys; sys.path.insert(0, {here!r}); import s2_grpo_smoke; assert 'torch' not in sys.modules"
        done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr

    def test_torch_missing_is_a_refusal_naming_it(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "torch", None)
        with pytest.raises(SystemExit, match="torch is not installed"):
            s2.require_torch()

    def test_a_broken_torch_raises_its_real_import_error_and_is_never_swallowed(self, monkeypatch):
        class Broken:
            @staticmethod
            def find_spec(name, path=None, target=None):
                if name == "torch":
                    raise ImportError("DLL load failed while importing torch")
                return None

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(sys, "meta_path", [Broken, *sys.meta_path])
        with pytest.raises(ImportError, match="DLL load failed") as caught:
            s2.require_torch()
        assert not isinstance(caught.value, SystemExit)

    def test_a_missing_dependency_of_torch_is_re_raised_not_swallowed(self, monkeypatch):
        class Missing:
            @staticmethod
            def find_spec(name, path=None, target=None):
                if name == "torch":
                    raise ModuleNotFoundError("No module named 'torch_dep'", name="torch_dep")
                return None

        monkeypatch.delitem(sys.modules, "torch", raising=False)
        monkeypatch.setattr(sys, "meta_path", [Missing, *sys.meta_path])
        with pytest.raises(ModuleNotFoundError, match="torch_dep") as caught:
            s2.require_torch()
        assert caught.value.name == "torch_dep"

    def test_ci_that_has_torch_does_not_skip_the_torch_tests(self):
        if os.environ.get("S2_REQUIRE_TORCH") == "1":
            assert HAS_TORCH, "S2_REQUIRE_TORCH=1 but torch is not installed: the rehearsal would be skipped"


# -- the CPU rehearsal on a tiny random model (needs torch; no network, never CUDA) ----------------------------------
@needs_torch
class TestDryRehearsal:
    def test_grpo_loss_is_the_clipped_ratio_loss_with_a_kl_term(self):
        import torch

        adv = torch.tensor([1.0, -1.0])
        mask = torch.ones(2, 3)
        logp = torch.zeros(2, 3)
        zero = s2.grpo_loss(logp, logp, logp, adv, mask)
        assert float(zero) == pytest.approx(0.0)  # ratio 1, no KL, advantages cancel
        # a ratio far above 1+eps on a positive advantage is clipped: its gradient is gone
        up = s2.grpo_loss(
            torch.full((1, 1), 2.0),
            torch.zeros(1, 1),
            torch.full((1, 1), 2.0),
            torch.tensor([1.0]),
            torch.ones(1, 1),
            beta=0.0,
        )
        assert float(up) == pytest.approx(-(1 + s2.EPSILON))
        # the KL term is positive when the policy leaves the reference
        kl = s2.grpo_loss(
            torch.zeros(1, 1),
            torch.zeros(1, 1),
            torch.full((1, 1), 1.0),
            torch.tensor([0.0]),
            torch.ones(1, 1),
            beta=0.04,
        )
        assert float(kl) > 0.0

    def test_a_dry_run_end_to_end_writes_the_receipt_and_never_touches_cuda(
        self, tmp_path, monkeypatch, capsys
    ):
        import torch

        # third-party imports (bitsandbytes, when installed) probe CUDA once at import; ours must not
        importlib.import_module("peft")
        importlib.import_module("transformers")

        def refuse(*a, **k):
            raise AssertionError("the dry rehearsal touched CUDA")

        for name in (
            "is_available",
            "synchronize",
            "max_memory_allocated",
            "reset_peak_memory_stats",
            "device_count",
        ):
            monkeypatch.setattr(torch.cuda, name, refuse)
        gold = _gold(tmp_path)
        a = s2.apply_dry_defaults(_args(tmp_path, gold, dry=True))
        assert s2.execute(a, build=s2.dry_bundle) == 0
        body = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        steps = [
            json.loads(x) for x in (a.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert body["verdict"] == "S2-DRY" and body["dry"] is True and body["readout"]["pass"] is False
        assert len(steps) == 2 and all(s["tokens"] > 0 and s["step_s"] > 0 for s in steps)
        assert (
            set(body["environment"]) == set(s2.ENVIRONMENT_FIELDS) and body["environment"]["gpu_name"] is None
        )
        assert len(body["gold_files_sha256"]) == 3 and body["rnd_commit"] == "e48cc90c"
        assert capsys.readouterr().out.strip().splitlines()[-1] == "S2-DRY"

    def test_a_rollout_that_times_out_stops_the_run_and_a_stopped_run_exits_non_zero(self, tmp_path):
        gold = _gold(tmp_path)
        a = s2.apply_dry_defaults(_args(tmp_path, gold, dry=True))
        real = s2.dry_bundle

        def build(args):
            bundle = real(args)
            calls = {"n": 0}
            inner = bundle["rollout"]

            def rollout(*x):
                calls["n"] += 1
                if calls["n"] > s2.GROUPS_PER_STEP:  # the first step is whole, the second times out
                    raise TimeoutError("/completion did not answer in 120s")
                return inner(*x)

            bundle["rollout"] = rollout
            return bundle

        assert s2.execute(a, build=build) == 1
        body = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert (
            body["verdict"] == "S2-STOPPED"
            and body["stopped"]["step"] == 1
            and "timeout" in body["stopped"]["error"]
        )
        assert body["readout"]["steps_completed"] == 1 and body["readout"]["pass"] is False

    def test_a_non_finite_loss_stops_the_run(self, tmp_path, monkeypatch):
        gold = _gold(tmp_path)
        a = s2.apply_dry_defaults(_args(tmp_path, gold, dry=True))
        monkeypatch.setattr(s2, "grpo_loss", lambda *x, **k: x[0].sum() * float("nan"))
        assert s2.execute(a, build=s2.dry_bundle) == 1
        body = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert body["stopped"]["error"] == "non-finite loss" and body["verdict"] == "S2-STOPPED"

    def test_an_out_of_memory_step_is_recorded_as_oom_and_not_as_a_pass(self, tmp_path):
        import torch

        gold = _gold(tmp_path)
        a = s2.apply_dry_defaults(_args(tmp_path, gold, dry=True))
        real = s2.dry_bundle

        def build(args):
            bundle = real(args)

            def rollout(*x):
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")

            bundle["rollout"] = rollout
            return bundle

        assert s2.execute(a, build=build) == 1
        body = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert body["readout"]["oom"] is True and body["readout"]["pass"] is False


# -- review jo: the stopped mutants, the cut, the llama.cpp server, the receipt, gradient checkpointing --------------
class TestStoppedAndCut:
    STOP = {"step": 7, "error": "timeout: /completion did not answer in 120s"}

    def test_a_20_step_run_with_a_stop_record_never_passes(self):
        res = _result(
            steps=_steps(20), stopped=self.STOP
        )  # every step counted, yet something stopped the run
        assert res["pass"] is False and res["complete"] is False

    def test_a_dry_run_with_all_its_steps_but_a_stop_reads_stopped(self):
        res = _result(steps=_steps(2), requested=2, dry=True, stopped=self.STOP)
        assert s2.verdict_line(res, stopped=self.STOP, cut=True, dry=True) == "S2-STOPPED"
        assert s2.exit_status("S2-STOPPED") == 1

    def test_a_changed_max_new_is_not_the_registered_shape(self):
        shape = {"steps": s2.STEPS, "prompts": s2.PROMPTS, "g": s2.G, "max_new": s2.MAX_NEW_TOKENS}
        assert s2.is_cut(Namespace(**shape)) is False
        for key, value in (("steps", 5), ("prompts", 8), ("g", 4), ("max_new", 1024)):
            assert s2.is_cut(Namespace(**{**shape, key: value})) is True, key


class _Resp:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return json.dumps(self.body).encode()


class TestLlamaServer:
    def test_http_json_posts_json_and_gets_when_there_is_no_payload(self):
        seen = []

        def urlopen(req, timeout):
            seen.append((req.full_url, req.data, req.get_method(), timeout))
            return _Resp({"ok": True})

        assert s2.http_json("http://h:1", "/completion", {"n": 1}, 120.0, urlopen=urlopen) == {"ok": True}
        assert s2.http_json("http://h:1", "/health", None, 10.0, urlopen=urlopen) == {"ok": True}
        assert seen[0][1] == b'{"n": 1}' and seen[0][2] == "POST" and seen[0][3] == 120.0
        assert seen[1][1] is None and seen[1][2] == "GET" and seen[1][3] == 10.0

    def test_a_socket_timeout_however_urllib_wraps_it_becomes_timeout_error(self):
        import urllib.error

        for exc in (TimeoutError("timed out"), urllib.error.URLError(TimeoutError("timed out"))):

            def urlopen(req, timeout, exc=exc):
                raise exc

            with pytest.raises(TimeoutError, match="/completion did not answer in 120s"):
                s2.http_json("http://h:1", "/completion", {}, 120.0, urlopen=urlopen)

    def test_other_url_errors_are_not_turned_into_timeouts(self):
        import urllib.error

        def urlopen(req, timeout):
            raise urllib.error.URLError(ConnectionRefusedError("refused"))

        with pytest.raises(urllib.error.URLError) as caught:
            s2.http_json("http://h:1", "/health", None, 10.0, urlopen=urlopen)
        assert not isinstance(caught.value, TimeoutError)

    def test_wait_ready_polls_until_health_says_ok(self):
        import urllib.error

        replies = iter([urllib.error.URLError("refused"), {"status": "loading model"}, {"status": "ok"}])
        polls, sleeps = [], []

        def get(path, timeout_s):
            polls.append((path, timeout_s))
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply

        clock = iter(range(1000))
        s2.wait_ready(
            get,
            lambda: None,
            timeout_s=100.0,
            control_timeout_s=10.0,
            sleep=sleeps.append,
            now=lambda: next(clock),
        )
        assert (
            polls == [("/health", 10.0)] * 3 and len(sleeps) == 2
        )  # it waited between polls, not after the ok

    def test_a_server_that_is_never_ready_is_a_structured_stop(self):
        clock = iter(range(0, 10_000, 40))
        sleeps = []

        def sleep(seconds):
            sleeps.append(seconds)
            assert len(sleeps) < 100, "wait_ready kept polling: it has no deadline"

        with pytest.raises(SystemExit, match="llama_server_not_ready"):
            s2.wait_ready(
                lambda p, t: {"status": "loading model"},
                lambda: None,
                timeout_s=100.0,
                sleep=sleep,
                now=lambda: next(clock),
            )

    def test_a_server_that_exited_is_a_structured_stop_and_is_not_polled_further(self):
        called = []
        with pytest.raises(SystemExit, match="llama_server_exited.*code 1"):
            s2.wait_ready(lambda p, t: called.append(p) or {"status": "ok"}, lambda: 1, sleep=lambda s: None)
        assert called == []

    def test_terminate_and_wait_stops_a_running_server_kills_a_stuck_one_and_leaves_an_exited_one(self):
        import subprocess

        class Proc:
            def __init__(self, code=None, stuck=False):
                self.code, self.stuck, self.calls = code, stuck, []

            def poll(self):
                return self.code

            def terminate(self):
                self.calls.append("terminate")

            def kill(self):
                self.calls.append("kill")
                self.code = -9

            def wait(self, timeout=None):
                self.calls.append("wait")
                if self.stuck and "kill" not in self.calls:
                    raise subprocess.TimeoutExpired("llama-server", timeout)

        running, stuck, gone = Proc(), Proc(stuck=True), Proc(code=0)
        for proc in (running, stuck, gone):
            s2.terminate_and_wait(proc, timeout_s=0.1)
        assert running.calls == ["terminate", "wait"]
        assert stuck.calls == ["terminate", "wait", "kill", "wait"]
        assert gone.calls == []

    def _args(self, tmp_path, **kw):
        return Namespace(
            llama_server=tmp_path / "llama-server",
            gguf=tmp_path / "m.gguf",
            ready_timeout=100.0,
            timeout=10.0,
            **kw,
        )

    def test_start_llama_server_waits_for_ready_then_hands_back_the_process_and_a_post(self, tmp_path):
        launched = []

        class Proc:
            terminated = False

            def poll(self):
                return None

            def terminate(self):
                self.terminated = True

            def wait(self, timeout=None):
                pass

        proc = Proc()
        answers = iter([{"status": "loading model"}, {"status": "ok"}, {"echo": 1}])

        def urlopen(req, timeout):
            return _Resp(next(answers))

        got, post = s2.start_llama_server(
            self._args(tmp_path),
            popen=lambda cmd: launched.append(cmd) or proc,
            urlopen=urlopen,
            sleep=lambda s: None,
            now=iter(range(1000)).__next__,
        )
        assert got is proc and not proc.terminated and "-m" in launched[0] and "--port" in launched[0]
        assert post("/completion", {}, 120.0) == {"echo": 1}

    def test_a_server_that_never_gets_ready_is_terminated_before_the_error_leaves(self, tmp_path):
        class Proc:
            calls = []

            def poll(self):
                return None if "terminate" not in self.calls else -15

            def terminate(self):
                self.calls.append("terminate")

            def wait(self, timeout=None):
                self.calls.append("wait")

        proc = Proc()
        clock = iter(range(0, 10_000, 60))

        def urlopen(req, timeout):
            return _Resp({"status": "loading model"})

        with pytest.raises(SystemExit, match="llama_server_not_ready"):
            s2.start_llama_server(
                self._args(tmp_path),
                popen=lambda cmd: proc,
                urlopen=urlopen,
                sleep=lambda s: None,
                now=lambda: next(clock),
            )
        assert proc.calls == ["terminate", "wait"]

    def test_file_record_hashes_the_bytes(self, tmp_path):
        import hashlib

        f = tmp_path / "x.gguf"
        f.write_bytes(b"abc")
        assert s2.file_record(f) == {"path": str(f), "sha256": hashlib.sha256(b"abc").hexdigest()}


class TestReceiptSeries:
    def test_the_sampler_keeps_its_series_and_the_readout_carries_it(self):
        readings = iter([9000, None, 31000, 12000])
        sampler = s2.VramSampler(lambda: next(readings, None))
        for _ in range(4):
            sampler.sample_once()
        assert sampler.series == [9000, 31000, 12000]
        res = s2.readout(
            _steps(),
            peak_mib=sampler.peak_mib,
            steps_requested=20,
            stopped=None,
            oom=False,
            cut=False,
            dry=False,
            series=sampler.series,
        )
        assert res["vram_mib_sampler_series"] == [9000, 31000, 12000] and res["sampler_samples"] == 3

    def test_a_sampler_that_read_nothing_has_no_peak_and_does_not_fit(self):
        res = s2.readout(
            _steps(),
            peak_mib=None,
            steps_requested=20,
            stopped=None,
            oom=False,
            cut=False,
            dry=False,
            series=[],
        )
        assert res["sampler_samples"] == 0 and res["peak_vram_mib_sampler"] is None
        assert res["vram_mib_sampler_series"] == [] and res["fits"] is False and res["pass"] is False

    def test_the_environment_records_the_interpreter_path(self):
        env = s2.environment(gpu=None, dtype="x", batch=1, exclusive=None, cuda=None)
        assert env["python_executable"] == sys.executable and "python_executable" in s2.ENVIRONMENT_FIELDS


@needs_torch
class TestReviewJoOnTheDryBundle:
    def test_checkpointing_is_off_in_the_rollout_and_on_in_every_training_pass(self, tmp_path):
        gold = _gold(tmp_path)
        a = s2.apply_dry_defaults(_args(tmp_path, gold, dry=True))
        inside = []

        def build(args):
            bundle = s2.dry_bundle(args)
            inner = bundle["rollout"]

            def rollout(*x):
                inside.append(bundle["state"]()["grad_ckpt"])  # the state while generate runs
                return inner(*x)

            bundle["rollout"] = rollout
            return bundle

        assert s2.execute(a, build=build) == 0
        steps = [
            json.loads(x) for x in (a.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert inside and inside == [False] * len(inside)
        assert [s["grad_ckpt"] for s in steps] == [True, True]
        assert [s["grad_ckpt_rollout"] for s in steps] == [False, False]

    def test_the_mode_switch_toggles_checkpointing_the_cache_and_train_mode(self):
        bundle = s2.dry_bundle(Namespace())
        model, mode, state = bundle["model"], bundle["mode"], bundle["state"]
        inner = model.get_base_model()
        mode(True)
        assert state() == {"grad_ckpt": True} and inner.config.use_cache is False and model.training is True
        mode(False)
        assert state() == {"grad_ckpt": False} and inner.config.use_cache is True and model.training is False

    def test_the_server_is_stopped_in_a_finally_on_success_and_on_an_exception(self, tmp_path):
        gold = _gold(tmp_path)
        closed = []

        def build_with(rollout_error):
            def build(args):
                bundle = s2.dry_bundle(args)
                bundle["close"] = lambda: closed.append(1)
                if rollout_error:

                    def rollout(*x):
                        raise RuntimeError("the server fell over")

                    bundle["rollout"] = rollout
                return bundle

            return build

        a = s2.apply_dry_defaults(_args(tmp_path, gold, dry=True))
        assert s2.execute(a, build=build_with(False)) == 0 and closed == [1]
        with pytest.raises(RuntimeError, match="fell over"):
            s2.execute(a, build=build_with(True))
        assert closed == [1, 1]

    def test_the_series_the_python_path_and_the_environment_land_in_the_receipt(self, tmp_path):
        gold = _gold(tmp_path)

        class Rig:
            def preflight(self):
                return {"name": "NVIDIA GeForce RTX 5090", "driver": "590.1"}

            def vram_query(self):
                return 12345

            def cuda_version(self):
                return "13.0"

        a = _args(tmp_path, gold, steps=1, g=2, max_new=24)
        assert s2.execute(a, build=s2.dry_bundle, rig=Rig()) == 1  # a cut run: S2-LIMITED, non-zero
        body = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        res = body["readout"]
        assert res["vram_mib_sampler_series"] and set(res["vram_mib_sampler_series"]) == {12345}
        assert res["sampler_samples"] == len(res["vram_mib_sampler_series"]) >= 2
        assert res["peak_vram_mib_sampler"] == 12345 and body["verdict"] == "S2-LIMITED"
        env = body["environment"]
        assert env["python_executable"] == sys.executable and env["gpu_name"] == "NVIDIA GeForce RTX 5090"
        assert env["cuda"] == "13.0" and set(env) == set(s2.ENVIRONMENT_FIELDS)


# -- the exclusive-GPU guard: rules (a) and (b), fail closed -----------------------------------------------------------------
def _xml(procs):
    """nvidia-smi -q -x output with the given (type, process_name) pairs; memory is N/A, as under WDDM."""
    infos = "".join(
        f"<process_info><pid>{100 + i}</pid><type>{t}</type><process_name>{n}</process_name>"
        f"<used_memory>N/A</used_memory></process_info>"
        for i, (t, n) in enumerate(procs)
    )
    return f"<nvidia_smi_log><gpu><processes>{infos}</processes></gpu></nvidia_smi_log>"


def _smi(procs=(), used="2400", xml=None):
    def run(args):
        if list(args) == ["-q", "-x"]:
            return xml if xml is not None else _xml(procs)
        assert list(args)[0] == "--query-gpu=memory.used", args
        return used

    return run


_DESKTOP_IMAGES = s2.DRY_DESKTOP_IMAGES  # one OS-only source, shared with the --dry canned output
DESKTOP = [("C+G", rf"C:\Windows\{n}") for n in _DESKTOP_IMAGES]


def _refusal(smi):
    with pytest.raises(SystemExit) as caught:
        s2.check_exclusive(smi)
    return str(caught.value)


class TestExclusiveGuard:
    def test_the_rigs_desktop_with_na_memory_and_idle_memory_passes(self):
        assert len(DESKTOP) == 26
        rec = s2.check_exclusive(_smi(DESKTOP, used="2400"))
        assert rec["result"] == "passed" and rec["memory_used_mib"] == 2400
        assert rec["rules"] == {"a_no_compute_runner": "passed", "b_memory_baseline": "passed"}

    def test_a_python_process_fails_rule_a_whatever_its_case_or_suffix(self):
        for name in (
            "python.exe",
            "PYTHON.EXE",
            "pythonw.exe",
            "python3.12",
            r"C:\envs\gpu-py312\python.exe",
        ):
            msg = _refusal(_smi([*DESKTOP, ("C+G", name)]))
            assert "rule (a)" in msg and "python" in msg.lower(), name

    def test_an_ollama_or_llama_server_process_fails_rule_a(self):
        for name in ("ollama.exe", "ollama_llama_server.exe", "llama-server.exe", "Ollama app.exe"):
            msg = _refusal(_smi([*DESKTOP, ("C", name)]))
            assert "rule (a)" in msg, name
        assert "ollama_llama_server.exe" in _refusal(_smi([*DESKTOP, ("C+G", "ollama_llama_server.exe")]))

    def test_a_compute_only_process_with_an_unlisted_name_fails_rule_a(self):
        msg = _refusal(_smi([*DESKTOP, ("C", "mystery.exe")]))
        assert "rule (a)" in msg and "mystery.exe (type C)" in msg

    def test_a_desktop_type_with_an_unlisted_name_passes(self):
        assert s2.check_exclusive(_smi([("G", "whatever.exe"), ("C+G", "Xorg")]))["result"] == "passed"

    def test_a_listed_name_with_a_desktop_type_still_refuses(self):
        assert "rule (a)" in _refusal(_smi([("C+G", "pythonista_notes.exe")]))

    def test_memory_just_over_the_baseline_fails_rule_b(self):
        msg = _refusal(_smi(DESKTOP, used="4097"))
        assert "rule (b)" in msg and "4097 MiB" in msg and "4096 MiB baseline" in msg

    def test_memory_exactly_at_the_baseline_passes(self):
        assert s2.check_exclusive(_smi(DESKTOP, used="4096"))["result"] == "passed"
        assert s2.EXCLUSIVE_BASELINE_MIB == 4096

    def test_both_rules_firing_are_both_named(self):
        msg = _refusal(_smi([("C", "python.exe")], used="9000"))
        assert "rule (a)" in msg and "rule (b)" in msg

    def test_nvidia_smi_missing_failing_slow_or_unreadable_each_refuses_with_a_named_reason(self):
        import subprocess

        def raises(exc):
            def run(args):
                raise exc

            return run

        cases = [
            (raises(FileNotFoundError("nvidia-smi")), "nvidia_smi_missing"),
            (raises(subprocess.CalledProcessError(9, "nvidia-smi")), "nvidia_smi_failed"),
            (raises(subprocess.CalledProcessError(9, "nvidia-smi")), "exited 9"),
            (raises(subprocess.TimeoutExpired("nvidia-smi", 10)), "nvidia_smi_timeout"),
            (raises(PermissionError("denied")), "nvidia_smi_unavailable"),
            (_smi(xml="this is not xml"), "nvidia_smi_garbage"),
            (_smi(xml="<other_root/>"), "unexpected root element"),
            (_smi(xml=_xml([("Z", "x.exe")])), "unknown process type"),
            (_smi(xml="<nvidia_smi_log/>"), "no <gpu> element"),
            (_smi(xml="<nvidia_smi_log><gpu/></nvidia_smi_log>"), "has no <processes> element"),
            (
                _smi(xml="<nvidia_smi_log><gpu><processes>N/A</processes></gpu></nvidia_smi_log>"),
                "outside any <process_info>",
            ),
            (
                _smi(xml="<nvidia_smi_log><gpu><processes>not available</processes></gpu></nvidia_smi_log>"),
                "holds text",
            ),
            (
                _smi(xml="<nvidia_smi_log><gpu><processes/></gpu><gpu><processes/></gpu></nvidia_smi_log>"),
                "more than one <gpu> element",
            ),
            (
                _smi(xml=_xml(DESKTOP).replace("</gpu>", "</gpu><gpu><processes/></gpu>")),
                "more than one <gpu>",
            ),
            (_smi(DESKTOP, used="lots"), "not a MiB integer"),
            (_smi(DESKTOP, used="-5"), "not a MiB integer"),
            (_smi(DESKTOP, used="24.5"), "not a MiB integer"),
            (_smi(DESKTOP, used="2000\n2200\n"), "2 memory.used readings for one GPU"),
            (_smi(DESKTOP, used="\n"), "no memory.used reading"),
        ]
        for smi, reason in cases:
            assert reason in _refusal(smi), reason

    def test_a_failure_in_the_second_call_also_refuses(self):
        def run(args):
            if list(args) == ["-q", "-x"]:
                return _xml(DESKTOP)
            raise FileNotFoundError("nvidia-smi")

        assert "nvidia_smi_missing" in _refusal(run)

    def test_the_receipt_carries_image_names_only_the_reading_the_baseline_and_the_rule_results(self):
        procs = [*DESKTOP, ("G", "/usr/lib/xorg/Xorg"), ("C+G", "N/A"), ("G", r"D:\a b\c d.exe")]
        rec = s2.check_exclusive(_smi(procs, used="2572"))
        assert all("\\" not in n and "/" not in n for n in rec["processes"])
        assert (
            "Xorg" in rec["processes"] and "c d.exe" in rec["processes"] and "<unnamed>" in rec["processes"]
        )
        assert len(rec["processes"]) == len(procs)
        assert rec["memory_used_mib"] == 2572 and rec["baseline_mib"] == 4096
        assert rec["runner_patterns"] == [
            "python*",
            "ollama*",
            "llama-server*",
            "llama-cli*",
            "llama-bench*",
            "uv*",
        ]
        assert rec["process_types"] == ["C+G", "G"]
        assert rec["rules"] == {"a_no_compute_runner": "passed", "b_memory_baseline": "passed"}
        assert rec["result"] == "passed" and rec["source"] == "nvidia-smi"
        json.dumps(rec)

    def test_no_processes_at_all_is_a_pass_not_a_parse_failure(self):
        assert (
            s2.check_exclusive(_smi(xml="<nvidia_smi_log><gpu><processes/></gpu></nvidia_smi_log>"))[
                "processes"
            ]
            == []
        )

    def test_an_empty_processes_element_is_the_real_nothing_on_the_card_shape_and_passes_to_rule_b(self):
        for empty in ("<processes/>", "<processes></processes>", "<processes>  </processes>"):
            xml = f"<nvidia_smi_log><gpu>{empty}</gpu></nvidia_smi_log>"
            rec = s2.check_exclusive(_smi(xml=xml, used="2400"))
            assert rec["processes"] == [] and rec["result"] == "passed", empty
            assert "rule (b)" in _refusal(_smi(xml=xml, used="4097")), empty

    def test_a_runner_like_directory_with_a_non_runner_file_name_passes_rule_a(self):
        sep = chr(92)  # built at runtime: no drive letter or user name is written out
        for folder in ("python", "ollama", "llama-server"):
            for path in (
                sep.join(["D:", folder, "explorer.exe"]),
                "/".join(["", "opt", folder, "explorer.exe"]),
            ):
                rec = s2.check_exclusive(_smi([("C+G", path)]))
                assert rec["processes"] == ["explorer.exe"] and rec["result"] == "passed", path

    def test_the_old_plain_rule_is_unchanged(self):
        s2.require_exclusive([])
        with pytest.raises(SystemExit, match="hold the GPU"):
            s2.require_exclusive(["1"])


class TestReportMarkupAndStructure:
    """Shapes the first guard let through as "no processes", and the markup a real report never needs."""

    DOCTYPE = '<?xml version="1.0" ?>\n<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v13.dtd">\n'

    @staticmethod
    def _wrap(inner, head=""):
        return f"{head}<nvidia_smi_log><gpu>{inner}</gpu></nvidia_smi_log>"

    def test_a_processes_element_with_a_foreign_child_refuses_and_names_the_tag(self):
        msg = _refusal(_smi(xml=self._wrap("<processes><other>N/A</other></processes>")))
        assert "nvidia_smi_garbage" in msg and "<other>" in msg

    def test_a_namespaced_process_info_refuses_as_a_foreign_child(self):
        xml = self._wrap('<processes><process_info xmlns="urn:x"><type>C+G</type></process_info></processes>')
        assert "{urn:x}process_info" in _refusal(_smi(xml=xml))

    def test_a_second_gpu_nested_inside_the_one_direct_gpu_refuses(self):
        msg = _refusal(_smi(xml=self._wrap("<processes/><gpu><processes/></gpu>")))
        assert "more than one <gpu> element (2)" in msg

    def test_a_namespaced_root_or_gpu_refuses(self):
        assert "unexpected root element" in _refusal(
            _smi(xml='<nvidia_smi_log xmlns="urn:x"><gpu/></nvidia_smi_log>')
        )
        xml = '<nvidia_smi_log><gpu xmlns="urn:x"><processes/></gpu></nvidia_smi_log>'
        assert "no <gpu> element" in _refusal(_smi(xml=xml))

    def test_the_genuine_empty_shapes_still_pass_to_rule_b(self):
        for inner in ("<processes/>", "<processes></processes>", "<processes><!-- nothing --></processes>"):
            rec = s2.check_exclusive(_smi(xml=self._wrap(inner), used="2400"))
            assert rec["processes"] == [] and rec["result"] == "passed", inner
            assert "rule (b)" in _refusal(_smi(xml=self._wrap(inner), used="4097")), inner

    def test_a_process_info_with_attributes_and_the_real_fields_is_accepted(self):
        info = (
            '<process_info id="7" kind="x"><gpu_instance_id>N/A</gpu_instance_id>'
            "<compute_instance_id>N/A</compute_instance_id><pid>1000</pid><type> c+g </type>"
            "<process_name>explorer.exe</process_name><used_memory>N/A</used_memory></process_info>"
        )
        rec = s2.check_exclusive(_smi(xml=self._wrap(f"<processes>\n  {info}\n</processes>")))
        assert rec["processes"] == ["explorer.exe"] and rec["process_types"] == ["C+G"]
        assert rec["result"] == "passed"

    def test_a_process_info_whose_type_is_missing_or_unreadable_refuses(self):
        for typ, why in (
            ("", "has no <type>"),
            ("<type/>", "unknown process type"),
            ("<type>X</type>", "unknown process type"),
            ("<type>c;g</type>", "unknown process type"),
        ):
            info = f"<process_info>{typ}<process_name>a.exe</process_name></process_info>"
            assert why in _refusal(_smi(xml=self._wrap(f"<processes>{info}</processes>"))), typ

    def _info(self, inner):
        return self._wrap(f"<processes><process_info>{inner}</process_info></processes>")

    def test_a_second_type_cannot_hide_a_compute_process_behind_the_first(self):
        xml = self._info("<type>C+G</type><type>C</type><process_name>python.exe</process_name>")
        assert "repeats <type>" in _refusal(_smi(xml=xml))

    def test_a_second_processes_block_cannot_hide_a_compute_process(self):
        hidden = "<processes><process_info><type>C</type><process_name>python.exe</process_name></process_info></processes>"
        xml = self._wrap("<processes/>" + hidden)
        assert "more than one <processes> element (2)" in _refusal(_smi(xml=xml))
        deeper = self._wrap("<processes/><other>" + hidden + "</other>")
        assert "more than one <processes> element (2)" in _refusal(_smi(xml=deeper))

    def test_memory_that_is_not_ascii_digits_refuses_by_name_and_never_crashes(self):
        for used in ("\u00b2", "\u0663\u0664", "\uff12\uff14", "12\u00b2", " \u2082 "):
            msg = _refusal(_smi(DESKTOP, used=used))
            assert "not a MiB integer" in msg and "nvidia_smi_garbage" in msg, repr(used)

    def test_a_memory_reading_is_bounded_to_nine_digits(self):
        for used in ("9" * 5000, "1" * 4301, "1234567890", "0" * 10):
            msg = _refusal(_smi(DESKTOP, used=used))
            assert "not a MiB integer of 1 to 9 digits" in msg and len(msg) < 200, used[:20]
        assert "rule (b)" in _refusal(_smi(DESKTOP, used="999999999"))
        assert s2.check_exclusive(_smi(DESKTOP, used="000000009"))["memory_used_mib"] == 9
        assert "no memory.used reading" in _refusal(_smi(DESKTOP, used=""))

    def test_a_process_info_outside_the_one_processes_element_refuses_wherever_it_sits(self):
        stray = "<process_info><type>C</type><process_name>python.exe</process_name></process_info>"
        cases = [
            (self._wrap("<processes/>" + stray), "sits under <gpu>"),
            (
                f"<nvidia_smi_log><gpu><processes/></gpu>{stray}</nvidia_smi_log>",
                "sits under <nvidia_smi_log>",
            ),
            (self._wrap(f"<processes/><other>{stray}</other>"), "sits under <other>"),
            (self._wrap(f"<processes/><other><deep>{stray}</deep></other>"), "sits under <deep>"),
            (self._wrap("<processes/>").replace("</gpu>", "</gpu><x/>") + "", None),
        ]
        for xml, why in cases[:4]:
            msg = _refusal(_smi(xml=xml))
            assert why in msg and "not under the one <processes>" in msg, why

    def test_a_stray_process_info_next_to_real_entries_also_refuses(self):
        real = "<process_info><type>G</type><process_name>explorer.exe</process_name></process_info>"
        stray = "<process_info><type>C</type><process_name>x.exe</process_name></process_info>"
        xml = self._wrap(f"<processes>{real}</processes>{stray}")
        assert "sits under <gpu>" in _refusal(_smi(xml=xml))

    def test_the_other_compute_tools_are_runner_names_too(self):
        for name in ("llama-cli.exe", "llama-bench.exe", "uv.exe", "uvx.exe", "LLAMA-CLI", "uv"):
            assert "rule (a)" in _refusal(_smi([*DESKTOP, ("C+G", name)])), name

    def test_text_outside_the_entries_refuses(self):
        entry = "<process_info><type>G</type></process_info>"
        cases = [
            (f"<processes>stray{entry}</processes>", "outside any <process_info>"),
            (f"<processes>{entry}between{entry}</processes>", "after a </process_info>"),
            (f"<processes>{entry}tail</processes>", "after a </process_info>"),
            ("<processes><process_info>loose<type>G</type></process_info></processes>", "outside its fields"),
            ("<processes><process_info><type>G</type>loose</process_info></processes>", "after a <type>"),
            (
                "<processes><process_info><type>G</type><pid>1</pid>x</process_info></processes>",
                "after a <pid>",
            ),
        ]
        for inner, why in cases:
            assert why in _refusal(_smi(xml=self._wrap(inner))), inner

    def test_an_element_outside_the_allowlist_a_repeat_or_a_nested_element_refuses(self):
        cases = [
            ("<type>G</type><foo/>", "unexpected <foo>"),
            ("<type>G</type><extra><deep/></extra>", "unexpected <extra>"),
            ("<type><b>C</b></type>", "nested <b>"),
            ("<type>G</type><pid>1</pid><pid>2</pid>", "repeats <pid>"),
            (
                "<type>G</type><process_name>a.exe</process_name><process_name>b.exe</process_name>",
                "repeats <process_name>",
            ),
            ("<pid>1</pid>", "has no <type>"),
            ("<type>G</type><pid><x/></pid>", "nested <x>"),
        ]
        for inner, why in cases:
            assert why in _refusal(_smi(xml=self._info(inner))), inner

    def test_a_predefined_entity_in_a_name_is_just_text(self):
        info = "<process_info><type>G</type><process_name>a&amp;b.exe</process_name></process_info>"
        rec = s2.check_exclusive(_smi(xml=self._wrap(f"<processes>{info}</processes>")))
        assert rec["processes"] == ["a&b.exe"]

    def test_the_drivers_declaration_doctype_and_comments_are_accepted(self):
        xml = self.DOCTYPE + "<!-- header -->" + self._wrap("<!-- a --><processes><!-- b --></processes>")
        assert s2.check_exclusive(_smi(xml=xml))["result"] == "passed"
        assert (
            s2.check_exclusive(_smi(xml=self._wrap("<processes/>")))["result"] == "passed"
        )  # no DOCTYPE at all

    def test_markup_a_real_report_never_needs_refuses_with_a_named_reason(self):
        sys_ok = '<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v13.dtd">'
        wrap = self._wrap("<processes/>")
        cases = [
            ("<?foo bar?>" + wrap, "processing instruction"),
            (wrap.replace("<gpu>", "<gpu><?foo bar?>"), "processing instruction"),
            ('<!DOCTYPE nvidia_smi_log [<!ENTITY x "y">]>' + wrap, "an ENTITY or any other declaration"),
            ("<!DOCTYPE nvidia_smi_log [ ]>" + wrap, "internal subset"),
            (sys_ok + wrap.replace("<processes/>", "<processes>&xxe;</processes>"), "entity reference &xxe;"),
            (wrap.replace("<processes/>", "<processes>&xxe;</processes>"), "not XML"),
            ('<!DOCTYPE nvidia_smi_log PUBLIC "-//x//y" "nvsmi_device_v13.dtd">' + wrap, "PUBLIC identifier"),
            ('<!DOCTYPE other SYSTEM "nvsmi_device_v13.dtd">' + wrap, "names 'other'"),
            ('<!DOCTYPE nvidia_smi_log SYSTEM "http://example.invalid/x.dtd">' + wrap, "x.dtd"),
        ]
        for xml, why in cases:
            assert why in _refusal(_smi(xml=xml)), why

    def test_any_numbered_nvsmi_device_dtd_is_accepted_and_nothing_else_is(self):
        wrap = self._wrap("<processes/>")
        for dtd in (
            "nvsmi_device_v13.dtd",
            "nvsmi_device_v14.dtd",
            "nvsmi_device_v99.dtd",
            "nvsmi_device_v1.dtd",
        ):
            xml = f'<!DOCTYPE nvidia_smi_log SYSTEM "{dtd}">' + wrap
            assert s2.check_exclusive(_smi(xml=xml))["result"] == "passed", dtd
        for dtd in (
            "nvsmi_device_v13a.dtd",
            "nvsmi_device_v.dtd",
            "../x.dtd",
            "../nvsmi_device_v13.dtd",
            "sub/nvsmi_device_v13.dtd",
            "nvsmi_device_v13.dtd.evil",
            "Nvsmi_device_v13.dtd",
            "nvsmi_device_v13_.dtd",
            "",
        ):
            xml = f'<!DOCTYPE nvidia_smi_log SYSTEM "{dtd}">' + wrap
            msg = _refusal(_smi(xml=xml))
            assert "nvidia_smi_garbage" in msg and "nvsmi_device_v<digits>.dtd" in msg, dtd

    def test_the_external_dtd_is_never_opened_or_fetched(self, tmp_path, monkeypatch):
        import builtins
        import socket
        import urllib.request

        # a DTD sits where a naive parser would look for it; it declares &x;, so loading it would make &x; legal
        (tmp_path / "nvsmi_device_v13.dtd").write_text('<!ENTITY x "from-the-dtd">', encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        touched = []
        real_open = builtins.open

        def spy_open(file, *a, **kw):
            if str(file).endswith(".dtd"):
                touched.append(str(file))
            return real_open(file, *a, **kw)

        def no_network(*a, **kw):
            touched.append("network")
            raise AssertionError("network access while parsing")

        monkeypatch.setattr(builtins, "open", spy_open)
        monkeypatch.setattr(socket, "create_connection", no_network)
        monkeypatch.setattr(socket.socket, "connect", no_network)
        monkeypatch.setattr(urllib.request, "urlopen", no_network)
        head = '<!DOCTYPE nvidia_smi_log SYSTEM "nvsmi_device_v13.dtd">'
        ok = head + self._wrap("<processes/>")
        assert s2.check_exclusive(_smi(xml=ok))["result"] == "passed"
        url = '<!DOCTYPE nvidia_smi_log SYSTEM "http://example.invalid/nvsmi_device_v13.dtd">'
        assert "DTD is" in _refusal(_smi(xml=url + self._wrap("<processes/>")))
        # if the DTD had been read, this reference would resolve and the report would pass
        ref = head + self._wrap(
            "<processes><process_info><type>G</type><process_name>&x;</process_name></process_info></processes>"
        )
        assert "entity reference &x;" in _refusal(_smi(xml=ref))
        assert touched == []


class TestRigPreflight:
    def _rig(self, tmp_path, smi):
        beat = tmp_path / "beat"
        beat.write_text("x", encoding="utf-8")
        return s2.Rig(Namespace(heartbeat=beat), smi=smi)

    def _smi(self, gpus, procs, used="2400"):
        inner = _smi(procs, used)

        def run(args):
            if list(args)[0] == "--query-gpu=name,driver_version":
                return gpus
            return inner(args)

        return run

    def test_a_real_preflight_on_the_rigs_desktop_passes_and_returns_the_guard_record(self, tmp_path):
        rig = self._rig(tmp_path, self._smi("NVIDIA GeForce RTX 5090, 590.1\n", DESKTOP))
        got = rig.preflight()
        assert got["name"] == "NVIDIA GeForce RTX 5090" and got["driver"] == "590.1"
        assert got["guard"]["result"] == "passed" and got["guard"]["source"] == "nvidia-smi"

    def test_a_runner_on_the_card_refuses_the_preflight(self, tmp_path):
        rig = self._rig(
            tmp_path, self._smi("NVIDIA GeForce RTX 5090, 590.1\n", [*DESKTOP, ("C", "python.exe")])
        )
        with pytest.raises(SystemExit, match="rule .a."):
            rig.preflight()

    def test_two_gpus_a_non_nvidia_gpu_and_a_garbled_gpu_line_still_refuse(self, tmp_path):
        for gpus, why in (
            ("NVIDIA A, 1\nNVIDIA B, 1\n", "2 GPUs"),
            ("AMD Radeon, 1\n", "not an NVIDIA GPU"),
            ("NVIDIA A\n", "nvidia_smi_garbage"),
        ):
            with pytest.raises(SystemExit, match=why):
                self._rig(tmp_path, self._smi(gpus, DESKTOP)).preflight()

    def test_a_missing_heartbeat_still_refuses_first(self, tmp_path):
        rig = s2.Rig(Namespace(heartbeat=tmp_path / "nope"), smi=self._smi("NVIDIA A, 1\n", DESKTOP))
        with pytest.raises(SystemExit, match="heartbeat"):
            rig.preflight()

    @needs_torch
    def test_a_run_carries_the_guard_record_into_the_receipt(self, tmp_path):
        class Rig:
            def preflight(self):
                return {"name": "NVIDIA GeForce RTX 5090", "driver": "590.1", "guard": {"result": "passed"}}

            def vram_query(self):
                return 12345

            def cuda_version(self):
                return "13.0"

        a = _args(tmp_path, _gold(tmp_path), steps=1, g=2, max_new=24)
        assert s2.execute(a, build=s2.dry_bundle, rig=Rig()) == 1
        body = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert body["environment"]["exclusive_guard"] == {"result": "passed"}
        assert body["environment"]["gpu_exclusive"] is True


@needs_torch
class TestDryUsesTheRealGuard:
    def test_the_dry_receipt_records_the_guards_result_without_running_nvidia_smi(
        self, tmp_path, monkeypatch
    ):
        import subprocess

        def forbidden(*args, **kw):
            raise AssertionError("the real nvidia-smi was started")

        monkeypatch.setattr(subprocess, "run", forbidden)
        a = s2.apply_dry_defaults(_args(tmp_path, _gold(tmp_path), dry=True))
        assert s2.execute(a, build=s2.dry_bundle) == 0
        body = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        guard = body["environment"]["exclusive_guard"]
        assert guard["result"] == "passed" and guard["source"].startswith("dry:")
        assert guard["memory_used_mib"] == s2.DRY_IDLE_MIB and len(guard["processes"]) == 26
        assert body["environment"]["gpu_exclusive"] is None

    def test_a_dry_run_whose_canned_card_is_not_the_desktop_would_refuse(self, tmp_path, monkeypatch):
        monkeypatch.setattr(s2, "DRY_IDLE_MIB", 5000)
        a = s2.apply_dry_defaults(_args(tmp_path, _gold(tmp_path), dry=True))
        with pytest.raises(SystemExit, match="rule .b."):
            s2.execute(a, build=s2.dry_bundle)


class TestTheRealNvidiaSmiShape:
    """The fixture has the real ``nvidia-smi -q -x`` shape at the real process count and type mix (26 desktop
    processes, all ``C+G``, memory ``N/A``), with OS and shell image names only. These tests assert the guard's
    verdict on it, not the file's contents."""

    FIXTURE = Path(__file__).parent / "fixtures" / "nvidia-smi-q-x-fixture-public-2026-11-03.xml"

    @staticmethod
    def _runner(xml, used="2400"):
        def run(args):
            return xml if list(args) == ["-q", "-x"] else used + "\n"

        return run

    def test_the_real_shape_passes_with_26_desktop_processes(self):
        rec = s2.check_exclusive(self._runner(self.FIXTURE.read_text(encoding="utf-8")))
        assert rec["result"] == "passed" and len(rec["processes"]) == 26
        assert rec["rules"] == {"a_no_compute_runner": "passed", "b_memory_baseline": "passed"}
        assert all("\\" not in n and "/" not in n for n in rec["processes"])

    def test_the_same_shape_refuses_when_a_runner_joins_it(self):
        xml = self.FIXTURE.read_text(encoding="utf-8")
        extra = (
            "<process_info><pid>9</pid><type>C</type><process_name>python.exe</process_name>"
            "<used_memory>N/A</used_memory></process_info></processes>"
        )
        with pytest.raises(SystemExit, match="rule .a.*python.exe"):
            s2.check_exclusive(self._runner(xml.replace("</processes>", extra, 1)))

    def test_the_same_shape_refuses_above_the_baseline(self):
        xml = self.FIXTURE.read_text(encoding="utf-8")
        with pytest.raises(SystemExit, match="rule .b."):
            s2.check_exclusive(self._runner(xml, used="4097"))


# -- no personal desktop names in a public file ------------------------------------------------------------------------------
_DENIED_SHA256 = frozenset(
    {
        "01b8d4f932aad0f172b98b20f92b51540de4852d58dbb8f7785e74a280b3d0d1",
        "11b496da7187c8892b2f8513c8380c59e4bd2b9dd951455b592d3fbfc7085b85",
        "39b4e46bc88704567baed958805a49910dbf165cbd0d2cf13177d14429e22dba",
        "3f1f29444c093e2890d2163174cce5d5db40386b84168e015ec9da46cff1a6f9",
        "5fc6ebae246b6dfea7b1bf8b1a5ce3df9c4aa3b6b5dcf716dc203876669a8077",
        "8462663f5bf0fb4bbabebf3aa7a04aa34a7fe6433587544a7b5a262b8086cc0e",
        "aa55de0728673d0e50853396ed87a303cd089d164800ad97872f90744358ffed",
        "ac0a3dfd6dddb20962cecff6ee5fe65e19d3923be20e52c5ab52ff877f7e4c32",
        "b7b479e0123d49074fac95b0d264ab37897447a8c3b0ce7cf7ac655c0dc5a2c7",
        "b7e12ae231da3d56982ba5291dd62f0eca40cb050e3639c20b3c67a8bda368bc",
        "babe90b8d3d59ebb6be72807c7c89d8606c9dccec48b9251d51d1886a0524312",
        "cb9c3b54d7e3f3d8d854d314a6e77b8bd5ce4a9e3bccc6b68f525645e6344f64",
        "d9e579df77e2827be5ceb25a24b5346e8ae24a4e743b7f7ebdfc001e1bc6a232",
        "e2f26e98726bde0e9a893dda21a7cede7fe27f9221922aab442feae55af04d0b",
    }
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9._-]+")


def _scan(files, *, denied=None):
    """Where a denied token sits, as ``file:line (hash n)``; the matched text is never echoed."""
    import hashlib

    active = _DENIED_SHA256 if denied is None else frozenset(denied)
    ordered = sorted(active)
    index_of = {h: idx for idx, h in enumerate(ordered)}
    hits = []
    for path in files:
        for no, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            seen_on_line = set()
            for raw in _TOKEN_RE.findall(line):
                tok = raw.lower()
                candidates = (tok, tok[:-4]) if tok.endswith(".exe") and len(tok) > 4 else (tok,)
                for cand in candidates:
                    digest = hashlib.sha256(cand.encode("utf-8")).hexdigest()
                    if digest in index_of and index_of[digest] not in seen_on_line:
                        seen_on_line.add(index_of[digest])
                        hits.append(f"{path.name}:{no} (hash {index_of[digest]})")
    return hits


class TestNoPersonalNamesInPublicFiles:
    ROOT = Path(__file__).parent.parent

    def _public_files(self):
        files = [self.ROOT / "examples" / "verifier-distill" / "s2_grpo_smoke.py", Path(__file__)]
        for folder in (self.ROOT / "docs", self.ROOT / "tests" / "fixtures"):
            files += sorted(p for p in folder.rglob("*") if p.is_file())
        return files

    def test_the_module_the_tests_the_docs_and_the_fixtures_hold_no_personal_desktop_names(self):
        assert _scan(self._public_files()) == []

    def test_the_scan_finds_a_planted_name_and_does_not_echo_it(self, tmp_path):
        import hashlib

        first = "".join(("planted", "_", "alpha"))
        second = "".join(("planted", "_", "beta"))
        h1 = hashlib.sha256(first.encode("utf-8")).hexdigest()
        h2 = hashlib.sha256(second.encode("utf-8")).hexdigest()
        ordered = sorted((h1, h2))
        planted = tmp_path / "planted.txt"
        planted.write_text(f"ok line\n{first.upper()}.exe is here\n{second} too\n", encoding="utf-8")
        assert _scan([planted], denied=frozenset({h1, h2})) == [
            f"planted.txt:2 (hash {ordered.index(h1)})",
            f"planted.txt:3 (hash {ordered.index(h2)})",
        ]

    def test_ordinary_words_are_not_flagged(self, tmp_path):
        ok = tmp_path / "ok.txt"
        ok.write_text("a moment, momentum, mainstream, the steps of a lawnmower, chromatic, antigen\n")
        assert _scan([ok]) == []

    def test_the_dry_canned_desktop_and_the_public_fixture_are_the_same_os_only_set(self):
        fixture = self.ROOT / "tests" / "fixtures" / "nvidia-smi-q-x-fixture-public-2026-11-03.xml"
        names = re.findall(r"<process_name>([^<]*)</process_name>", fixture.read_text(encoding="utf-8"))
        assert len(names) == 26 and sorted(names) == sorted(s2.DRY_DESKTOP_IMAGES)
        assert _scan([fixture]) == []

    @needs_torch
    def test_the_dry_receipt_holds_only_the_os_only_names(self, tmp_path):
        a = s2.apply_dry_defaults(_args(tmp_path, _gold(tmp_path), dry=True))
        assert s2.execute(a, build=s2.dry_bundle) == 0
        receipt = a.out / "s2-hf.json"
        body = json.loads(receipt.read_text(encoding="utf-8"))
        processes = body["environment"]["exclusive_guard"]["processes"]
        assert sorted(processes) == sorted(s2.DRY_DESKTOP_IMAGES)
        assert _scan([receipt]) == []

    def test_planted_token_built_from_split_literals_is_caught_by_hash_scan(self, tmp_path):
        import hashlib

        token = "".join(("planted", "_", "denied", "_", "marker"))
        token_hash = hashlib.sha256(token.lower().encode("utf-8")).hexdigest()
        planted = tmp_path / "planted_hash.txt"
        planted.write_text(f"clean line\nrun {token.upper()}.EXE now\nand {token} bare\n", encoding="utf-8")
        hits = _scan([planted], denied=frozenset({token_hash}))
        assert hits == ["planted_hash.txt:2 (hash 0)", "planted_hash.txt:3 (hash 0)"]
        assert token not in " ".join(hits)
        assert len(_DENIED_SHA256) == 14
        assert not hasattr(sys.modules[__name__], "_personal_patterns")

    def test_diff_added_lines_contain_no_denied_hashes_and_no_exe_in_new_log_fields(self, tmp_path):
        import subprocess

        assert hasattr(s2, "PhaseLogger") and hasattr(s2, "run_probe_memory")
        phase_path = tmp_path / "s2-hf-phases.jsonl"
        fake_cuda = _FakeCuda()
        logger = s2.PhaseLogger(phase_path, enabled=True, cuda=fake_cuda)
        logger.record(step=0, group=0, phase="pre_rollout", prompt_tokens=12, completion_lengths=[])
        logger.record(step=0, group=0, phase="step_end", prompt_tokens=12, completion_lengths=[4, 6])
        logger.close()
        phase_text = phase_path.read_text(encoding="utf-8")
        assert re.findall(r"[A-Za-z0-9._-]+\.exe\b", phase_text, re.IGNORECASE) == []
        assert _scan([phase_path]) == []

        a = _args(
            tmp_path,
            _gold(tmp_path),
            probe_memory=True,
            probe_lengths="4,8",
        )
        rig = _FakeRig(a)
        assert (
            s2.execute(
                a,
                build=lambda args: _fake_probe_bundle(args, fake_cuda),
                rig=rig,
                counter_runner=lambda: (
                    '"\\\\HOST\\GPU Adapter Memory(luid_0x0_0x1_phys_0)\\Shared Usage",'
                    '"\\\\HOST\\GPU Adapter Memory(luid_0x0_0x1_phys_0)\\Dedicated Usage"\n'
                    '"1048576","2097152"\n'
                ),
            )
            == 0
        )
        probe_path = a.out / "s2-hf-probe.json"
        probe_text = probe_path.read_text(encoding="utf-8")
        assert re.findall(r"[A-Za-z0-9._-]+\.exe\b", probe_text, re.IGNORECASE) == []
        assert _scan([probe_path]) == []

        base_sha = "9837195bb763c5d0ff220246d28b65057749676f"
        check = subprocess.run(
            ["git", "rev-parse", "--verify", f"{base_sha}^{{commit}}"],
            cwd=self.ROOT,
            capture_output=True,
            text=True,
        )
        if check.returncode == 0:
            diff_all = subprocess.run(
                [
                    "git",
                    "diff",
                    base_sha,
                    "--",
                    "examples/verifier-distill/s2_grpo_smoke.py",
                    "tests/test_verifier_s2.py",
                    "docs/runs/2026-10-10-smoke-s2-prereg.md",
                ],
                cwd=self.ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            added_lines = [
                line[1:]
                for line in diff_all.splitlines()
                if line.startswith("+") and not line.startswith("+++")
            ]
            assert added_lines, "expected non-empty diff against base sha"
            diff_file = tmp_path / "diff_added.txt"
            diff_file.write_text("\n".join(added_lines) + "\n", encoding="utf-8")
            assert _scan([diff_file]) == []


# -- helpers for memory-fix, phase-logger and probe tests --------------------------------------------------------------------
class _FakeCuda:
    def __init__(self):
        self.calls: list[str] = []
        self.alloc = 100 * 2**20
        self.reserved = 128 * 2**20
        self.max_alloc = 110 * 2**20
        self.max_reserved = 140 * 2**20

    def is_available(self) -> bool:
        return True

    def memory_allocated(self) -> int:
        self.calls.append("memory_allocated")
        return self.alloc

    def memory_reserved(self) -> int:
        self.calls.append("memory_reserved")
        return self.reserved

    def max_memory_allocated(self) -> int:
        self.calls.append("max_memory_allocated")
        return self.max_alloc

    def max_memory_reserved(self) -> int:
        self.calls.append("max_memory_reserved")
        return self.max_reserved

    def reset_peak_memory_stats(self) -> None:
        self.calls.append("reset_peak_memory_stats")
        self.max_alloc = self.alloc
        self.max_reserved = self.reserved

    def empty_cache(self) -> None:
        self.calls.append("empty_cache")


class _FakeRig:
    def __init__(self, a):
        self.a = a
        self.preflight_called = 0

    def preflight(self) -> dict:
        self.preflight_called += 1
        return {
            "name": "NVIDIA GeForce RTX 5090",
            "driver": "595.71",
            "guard": s2.check_exclusive(s2.dry_smi, source="fake preflight"),
        }

    def vram_query(self) -> int | None:
        return 2500

    def cuda_version(self) -> str | None:
        return "13.0"


def _tiny_qwen3(dtype=None):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    target_dtype = torch.bfloat16 if dtype is None else dtype
    cfg = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        torch_dtype=target_dtype,
    )
    base = Qwen3ForCausalLM(cfg).to(target_dtype)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    model = get_peft_model(
        base,
        LoraConfig(r=4, lora_alpha=8, target_modules=list(s2.LORA_TARGETS), lora_dropout=0.0),
    )
    return model


def _fake_probe_bundle(a, fake_cuda: _FakeCuda):
    bundle = s2.dry_bundle(a)
    if getattr(a, "bf16_restore", False):
        s2.restore_bf16(bundle["model"])
    bundle["cuda"] = fake_cuda
    return bundle


# -- 1. --bf16-restore -------------------------------------------------------------------------------------------------------
@needs_torch
class TestBf16Restore:
    def test_without_bf16_restore_prepare_upcasts_to_fp32_and_cache_is_fp32(self):
        import torch

        model = _tiny_qwen3(torch.bfloat16)
        non_trainable = [p for p in model.parameters() if not p.requires_grad and p.is_floating_point()]
        assert non_trainable and all(p.dtype == torch.float32 for p in non_trainable)
        model.eval()
        model.get_base_model().gradient_checkpointing_disable()
        model.get_base_model().config.use_cache = True
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[2, 3, 4, 5]]), use_cache=True)
        cache = out.past_key_values
        keys = [layer.keys for layer in cache.layers]
        values = [layer.values for layer in cache.layers]
        assert keys and all(k.dtype == torch.float32 for k in keys)
        assert values and all(v.dtype == torch.float32 for v in values)

    def test_bf16_restore_casts_non_trainable_float_params_to_bf16_keeps_lora_fp32_and_skips_params4bit(self):
        import torch

        class Params4bit(torch.nn.Parameter):
            pass

        model = _tiny_qwen3(torch.bfloat16)
        fake_4bit = Params4bit(torch.ones(4, dtype=torch.float32), requires_grad=False)
        model.get_base_model().register_parameter("fake_4bit", fake_4bit)

        census = s2.restore_bf16(model)
        for name, param in model.named_parameters():
            if name.endswith("fake_4bit"):
                assert param.dtype == torch.float32
            elif param.requires_grad:
                assert "lora_" in name
                assert param.dtype == torch.float32
            elif param.is_floating_point():
                assert param.dtype == torch.bfloat16, f"{name} remained {param.dtype}"
        inner = model.get_base_model().model
        assert inner.embed_tokens.weight.dtype == torch.bfloat16
        assert model.get_base_model().lm_head.weight.dtype == torch.bfloat16
        assert inner.norm.weight.dtype == torch.bfloat16
        assert census["trainable"]["float32"] > 0
        assert census["non_trainable"]["bfloat16"] > 0

    def test_bf16_restore_forward_with_use_cache_produces_bf16_kv_cache(self):
        import torch

        model = _tiny_qwen3(torch.bfloat16)
        s2.restore_bf16(model)
        model.eval()
        model.get_base_model().gradient_checkpointing_disable()
        model.get_base_model().config.use_cache = True
        with torch.no_grad():
            out = model(input_ids=torch.tensor([[2, 3, 4, 5]]), use_cache=True)
        keys, values = s2.cache_kv_tensors(out.past_key_values)
        assert keys and all(k.dtype == torch.bfloat16 for k in keys)
        assert values and all(v.dtype == torch.bfloat16 for v in values)

    def test_bf16_restore_recorded_in_receipt_and_readout(self, tmp_path):
        a = s2.apply_dry_defaults(_args(tmp_path, _gold(tmp_path), dry=True, bf16_restore=True))
        assert s2.execute(a, build=s2.dry_bundle) == 0
        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt["run"]["bf16_restore"] is True
        assert receipt["readout"]["flags"]["bf16_restore"] is True
        assert receipt["dtype_census"]["non_trainable"]["bfloat16"] > 0
        assert receipt["dtype_census"]["trainable"]["float32"] > 0


# -- 2. --logprob-microbatch M -----------------------------------------------------------------------------------------------
@needs_torch
class TestLogprobMicrobatch:
    def test_logprob_microbatch_matches_old_path_logprobs_loss_and_lora_grads_for_m_1_2_g(self):
        import torch

        model = _tiny_qwen3(torch.float32)
        model.eval()
        model.get_base_model().gradient_checkpointing_disable()
        torch.manual_seed(42)
        for name, param in model.named_parameters():
            if param.requires_grad and "lora_B" in name:
                param.data.normal_(0.0, 0.05)

        prompt_ids = [2, 3, 4, 5, 6, 7]
        comps = [
            [8, 9, 10],
            [11, 12, 13, 14, 15, 16, 17],
            [18],
            [19, 20, 21, 22, 23],
        ]
        g = len(comps)
        width = max(len(c) for c in comps)
        pad = 0
        ids = torch.tensor([prompt_ids + c + [pad] * (width - len(c)) for c in comps], dtype=torch.long)
        att = torch.tensor(
            [[1] * (len(prompt_ids) + len(c)) + [0] * (width - len(c)) for c in comps], dtype=torch.long
        )
        mask = torch.tensor([[1.0] * len(c) + [0.0] * (width - len(c)) for c in comps], dtype=torch.float32)
        adv = torch.tensor([0.75, -1.25, 0.5, 0.0], dtype=torch.float32)
        prompt_len = len(prompt_ids)

        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            old_full = s2.completion_logprobs(model, ids, att, prompt_len)
            with model.disable_adapter():
                ref_full = s2.completion_logprobs(model, ids, att, prompt_len)
        logp_full = s2.completion_logprobs(model, ids, att, prompt_len)
        loss_full = s2.grpo_loss(logp_full, old_full, ref_full, adv, mask)
        loss_full.backward()
        grads_full = {
            name: param.grad.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        assert all(torch.any(g_tensor != 0) for g_tensor in grads_full.values())

        for m in (1, 2, g):
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                old_mb = s2.completion_logprobs_microbatch(model, ids, att, prompt_len, m)
                with model.disable_adapter():
                    ref_mb = s2.completion_logprobs_microbatch(model, ids, att, prompt_len, m)
            assert torch.allclose(old_mb * mask, old_full * mask, atol=1e-5, rtol=0.0)
            assert torch.allclose(ref_mb * mask, ref_full * mask, atol=1e-5, rtol=0.0)
            assert float((old_mb - old_full).abs().max()) <= 1e-5
            assert float((ref_mb - ref_full).abs().max()) <= 1e-5

            loss_mb = s2.grpo_microbatch_backward(
                model,
                ids,
                att,
                prompt_len,
                old_mb,
                ref_mb,
                adv,
                mask,
                microbatch=m,
                scale=1.0,
            )
            assert abs(float(loss_mb) - float(loss_full.detach())) <= 1e-6
            for name, param in model.named_parameters():
                if param.requires_grad:
                    diff = float((param.grad.detach() - grads_full[name]).abs().max())
                    assert diff <= 1e-5, f"M={m} {name} grad diff {diff}"

    def test_logprob_microbatch_passes_logits_to_keep_and_never_calls_log_softmax(self, monkeypatch):
        import torch

        model = _tiny_qwen3(torch.float32)
        model.eval()
        seen_kwargs: list[dict] = []
        orig_forward = model.get_base_model().forward

        def spy_forward(*args, **kwargs):
            seen_kwargs.append(dict(kwargs))
            return orig_forward(*args, **kwargs)

        monkeypatch.setattr(model.get_base_model(), "forward", spy_forward)
        monkeypatch.setattr(
            torch,
            "log_softmax",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("log_softmax must not be called")),
        )
        monkeypatch.setattr(
            torch.nn.functional,
            "log_softmax",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("F.log_softmax must not be called")),
        )

        ids = torch.tensor([[2, 3, 4, 8, 9, 0], [2, 3, 4, 10, 11, 12]], dtype=torch.long)
        att = torch.tensor([[1, 1, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1]], dtype=torch.long)
        with torch.no_grad():
            lp = s2.completion_logprobs_microbatch(model, ids, att, prompt_len=3, microbatch=1)
        assert lp.shape == (2, 3)
        assert len(seen_kwargs) == 2
        assert all(k.get("logits_to_keep") == 4 for k in seen_kwargs)


# -- 3. --alloc-conf expandable and --empty-cache-phases ---------------------------------------------------------------------
class TestAllocConfAndEmptyCache:
    def test_alloc_conf_expandable_passes_after_1mib_cuda_alloc_and_settings_check(self, monkeypatch):
        import builtins

        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        s2.configure_alloc_conf("expandable")
        real_import = builtins.__import__
        env_at_torch_import: list[str | None] = []
        events: list[str] = []

        class _ProbeTensor:
            def __del__(self):
                events.append("freed")

        class _FakeCudaMod:
            @staticmethod
            def is_available() -> bool:
                events.append("is_available")
                return True

            @staticmethod
            def empty_cache() -> None:
                events.append("empty_cache")

        class _FakeC:
            @staticmethod
            def _accelerator_getAllocatorSettings() -> str:
                events.append("get_settings")
                return "expandable_segments:True"

        class _FakeTorch:
            uint8 = "uint8"
            cuda = _FakeCudaMod
            _C = _FakeC

            @staticmethod
            def empty(size, *, dtype=None, device=None):
                events.append(f"empty:{size}:{device}")
                return _ProbeTensor()

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                env_at_torch_import.append(os.environ.get("PYTORCH_CUDA_ALLOC_CONF"))
                return _FakeTorch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            mod = s2.require_torch()
            assert mod is _FakeTorch
            assert env_at_torch_import == ["expandable_segments:True"]
            assert f"empty:{1024 * 1024}:cuda" in events
            assert "freed" in events
            assert "get_settings" in events
            assert events.index(f"empty:{1024 * 1024}:cuda") < events.index("get_settings")
        finally:
            s2.configure_alloc_conf(None)

    def test_alloc_conf_expandable_refuses_on_settings_mismatch(self, monkeypatch):
        import builtins

        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        s2.configure_alloc_conf("expandable")
        real_import = builtins.__import__

        class _FakeCudaMod:
            @staticmethod
            def is_available() -> bool:
                return True

        class _FakeC:
            @staticmethod
            def _accelerator_getAllocatorSettings() -> str:
                return "expandable_segments:False"

        class _MismatchTorch:
            uint8 = "uint8"
            cuda = _FakeCudaMod
            _C = _FakeC

            @staticmethod
            def empty(size, *, dtype=None, device=None):
                return object()

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return _MismatchTorch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            with pytest.raises(SystemExit) as exc_info:
                s2.require_torch()
            msg = str(exc_info.value)
            assert "expandable segments check failed:" in msg
            assert "unsupported" not in msg.lower()
        finally:
            s2.configure_alloc_conf(None)

    def test_alloc_conf_expandable_refuses_on_exception(self, monkeypatch):
        import builtins

        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        s2.configure_alloc_conf("expandable")
        real_import = builtins.__import__

        class _FakeCudaMod:
            @staticmethod
            def is_available() -> bool:
                return True

        class _ExceptionTorch:
            uint8 = "uint8"
            cuda = _FakeCudaMod

            @staticmethod
            def empty(size, *, dtype=None, device=None):
                raise RuntimeError("driver alloc boom")

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return _ExceptionTorch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            with pytest.raises(SystemExit) as exc_info:
                s2.require_torch()
            msg = str(exc_info.value)
            assert "expandable segments check failed:" in msg
            assert "driver alloc boom" in msg
            assert "unsupported" not in msg.lower()
        finally:
            s2.configure_alloc_conf(None)

    def test_alloc_conf_expandable_refuses_when_cuda_unavailable(self, monkeypatch):
        import builtins

        monkeypatch.delenv("PYTORCH_CUDA_ALLOC_CONF", raising=False)
        s2.configure_alloc_conf("expandable")
        real_import = builtins.__import__

        class _NoCudaMod:
            @staticmethod
            def is_available() -> bool:
                return False

        class _NoCudaTorch:
            cuda = _NoCudaMod

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                return _NoCudaTorch
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        try:
            with pytest.raises(SystemExit) as exc_info:
                s2.require_torch()
            msg = str(exc_info.value)
            assert "expandable segments check failed:" in msg
            assert "unsupported" not in msg.lower()
        finally:
            s2.configure_alloc_conf(None)

    @needs_torch
    def test_empty_cache_phases_calls_empty_cache_after_rollout_and_after_ref(self, tmp_path):
        a = s2.apply_dry_defaults(_args(tmp_path, _gold(tmp_path), dry=True, steps=1, g=2))
        bundle = s2.dry_bundle(a)
        rows, _ = s2.read_gold(a.gold)
        prompts = s2.select_prompts(rows, a.prompts)
        events: list[str] = []
        orig_rollout = bundle["rollout"]
        orig_ref = bundle["ref_adapter_off"]

        def tracked_rollout(*args, **kwargs):
            res = orig_rollout(*args, **kwargs)
            events.append("rollout")
            return res

        from contextlib import contextmanager

        @contextmanager
        def tracked_ref():
            with orig_ref():
                yield
            events.append("ref")

        s2.run_steps(
            model=bundle["model"],
            ref_adapter_off=tracked_ref,
            tok=bundle["tok"],
            optimiser=bundle["optimiser"],
            prompts=prompts,
            rollout=tracked_rollout,
            steps=1,
            g=2,
            max_new=8,
            reward_fn=bundle["reward_fn"],
            device_sync=bundle["sync"],
            peak_alloc_mib=bundle["peak_alloc_mib"],
            mode=bundle["mode"],
            state=bundle["state"],
            empty_cache_phases=True,
            empty_cache=lambda: events.append("empty_cache"),
        )
        assert events == [
            "rollout",
            "empty_cache",
            "ref",
            "empty_cache",
            "rollout",
            "empty_cache",
            "ref",
            "empty_cache",
        ]


# -- 4. --attention-mask -----------------------------------------------------------------------------------------------------
@needs_torch
class TestAttentionMaskFlag:
    def test_attention_mask_flag_passes_explicit_all_ones_mask_to_generate(self):
        import torch

        tok = s2.DryTokenizer()
        recorded: list[dict] = []

        class _FakeGenModel:
            device = torch.device("cpu")

            def generate(self, input_ids, **kwargs):
                recorded.append({"input_ids": input_ids, **kwargs})
                extra = torch.full((input_ids.shape[0], 3), 5, dtype=torch.long)
                return torch.cat([input_ids, extra], dim=1)

        fake_model = _FakeGenModel()
        s2.hf_rollout(fake_model, tok, "abc", g=3, max_new=4, temperature=0.7, pass_attention_mask=False)
        assert "attention_mask" not in recorded[-1]

        s2.hf_rollout(fake_model, tok, "abc", g=3, max_new=4, temperature=0.7, pass_attention_mask=True)
        assert "attention_mask" in recorded[-1]
        mask = recorded[-1]["attention_mask"]
        assert mask.shape == recorded[-1]["input_ids"].shape
        assert torch.all(mask == 1)


# -- 5. Phase allocator logging ----------------------------------------------------------------------------------------------
@needs_torch
class TestPhaseAllocatorLogging:
    def test_phase_logger_records_call_order_and_resets_peak_stats_after_each_boundary(self, tmp_path):
        a = s2.apply_dry_defaults(_args(tmp_path, _gold(tmp_path), dry=True, steps=1, g=2))
        bundle = s2.dry_bundle(a)
        rows, _ = s2.read_gold(a.gold)
        prompts = s2.select_prompts(rows, a.prompts)
        fake_cuda = _FakeCuda()
        phase_file = tmp_path / "s2-hf-phases.jsonl"
        fsync_count = 0

        def fake_fsync(fd: int) -> None:
            nonlocal fsync_count
            fsync_count += 1

        logger = s2.PhaseLogger(phase_file, enabled=True, cuda=fake_cuda, fsync=fake_fsync)
        try:
            s2.run_steps(
                model=bundle["model"],
                ref_adapter_off=bundle["ref_adapter_off"],
                tok=bundle["tok"],
                optimiser=bundle["optimiser"],
                prompts=prompts,
                rollout=bundle["rollout"],
                steps=1,
                g=2,
                max_new=8,
                reward_fn=bundle["reward_fn"],
                device_sync=bundle["sync"],
                peak_alloc_mib=bundle["peak_alloc_mib"],
                mode=bundle["mode"],
                state=bundle["state"],
                phase_logger=logger,
            )
        finally:
            logger.close()

        lines = [json.loads(line) for line in phase_file.read_text(encoding="utf-8").splitlines()]
        expected_phases = [
            (0, "pre_rollout"),
            (0, "post_rollout"),
            (0, "post_ref"),
            (0, "post_train_forward"),
            (0, "post_train_backward"),
            (1, "pre_rollout"),
            (1, "post_rollout"),
            (1, "post_ref"),
            (1, "post_train_forward"),
            (1, "post_train_backward"),
            (1, "step_end"),
        ]
        assert [(r["group"], r["phase"]) for r in lines] == expected_phases
        assert fsync_count == len(expected_phases)
        unit = [
            "memory_allocated",
            "memory_reserved",
            "max_memory_allocated",
            "max_memory_reserved",
            "reset_peak_memory_stats",
        ]
        assert fake_cuda.calls == unit * len(expected_phases)
        for rec in lines:
            assert set(rec.keys()) == {
                "step",
                "group",
                "phase",
                "monotonic_s",
                "prompt_tokens",
                "completion_tokens",
                "memory_allocated",
                "memory_reserved",
                "max_memory_allocated",
                "max_memory_reserved",
            }
            assert set(rec["completion_tokens"].keys()) == {"min", "max", "sum"}

    def test_phase_logger_simulated_kill_mid_phase_leaves_complete_lines_only(self, tmp_path):
        class _SimulatedKill(BaseException):
            pass

        a = s2.apply_dry_defaults(_args(tmp_path, _gold(tmp_path), dry=True, steps=1, g=2))
        bundle = s2.dry_bundle(a)
        rows, _ = s2.read_gold(a.gold)
        prompts = s2.select_prompts(rows, a.prompts)
        fake_cuda = _FakeCuda()
        phase_file = tmp_path / "s2-hf-phases.jsonl"
        orig_ref = bundle["ref_adapter_off"]

        from contextlib import contextmanager

        @contextmanager
        def kill_after_ref():
            with orig_ref():
                yield
            raise _SimulatedKill("killed right after reference forward")

        logger = s2.PhaseLogger(phase_file, enabled=True, cuda=fake_cuda)
        with pytest.raises(_SimulatedKill):
            try:
                s2.run_steps(
                    model=bundle["model"],
                    ref_adapter_off=kill_after_ref,
                    tok=bundle["tok"],
                    optimiser=bundle["optimiser"],
                    prompts=prompts,
                    rollout=bundle["rollout"],
                    steps=1,
                    g=2,
                    max_new=8,
                    reward_fn=bundle["reward_fn"],
                    device_sync=bundle["sync"],
                    peak_alloc_mib=bundle["peak_alloc_mib"],
                    mode=bundle["mode"],
                    state=bundle["state"],
                    phase_logger=logger,
                )
            finally:
                logger.close()

        raw = phase_file.read_text(encoding="utf-8")
        assert raw.endswith("\n")
        parsed = [json.loads(line) for line in raw.splitlines()]
        assert [r["phase"] for r in parsed] == ["pre_rollout", "post_rollout"]

    def test_phase_logger_keeps_run_max_across_resets_and_writes_to_readout(self, tmp_path):
        fake_cuda = _FakeCuda()
        a = _args(tmp_path, _gold(tmp_path), steps=1, g=2, max_new=8)
        rig = _FakeRig(a)

        def build_spike(args):
            bundle = s2.dry_bundle(args)
            bundle["cuda"] = fake_cuda
            bundle["peak_alloc_mib"] = lambda: int(fake_cuda.max_memory_allocated() / 2**20)
            orig_rollout = bundle["rollout"]

            def spike_rollout(*r_args, **r_kwargs):
                res = orig_rollout(*r_args, **r_kwargs)
                # True peak occurs inside group 0's rollout (between pre_rollout and post_rollout resets)
                if fake_cuda.max_alloc < 18000 * 2**20:
                    fake_cuda.max_alloc = 18500 * 2**20
                    fake_cuda.max_reserved = 20100 * 2**20
                return res

            bundle["rollout"] = spike_rollout
            return bundle

        assert s2.execute(a, build=build_spike, rig=rig) == 1  # cut run (steps=1): S2-LIMITED
        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt["readout"]["peak_vram_mib_torch"] == 18500
        assert receipt["readout"]["peak_reserved_mib_torch"] == 20100
        doc = (Path(__file__).parent.parent / "docs" / "runs" / "2026-10-10-smoke-s2-prereg.md").read_text(
            encoding="utf-8"
        )
        assert "a run without flags behaves identically" not in doc
        assert "changes how the torch peak is measured, and nothing else" in doc


# -- 6. Probe mode (--probe-memory) and --prompt-lengths ---------------------------------------------------------------------
class TestProbeMemoryAndPromptLengths:
    def test_prompt_lengths_cpu_only_writes_stats_without_model_or_gpu(self, tmp_path):
        a = _args(tmp_path, _gold(tmp_path), prompt_lengths=True, dry=True)
        assert (
            s2.execute(
                a,
                build=lambda _: (_ for _ in ()).throw(AssertionError("model build must not run")),
                rig=None,
            )
            == 0
        )
        out_file = a.out / "s2-hf-prompt-lengths.json"
        data = json.loads(out_file.read_text(encoding="utf-8"))
        assert data["count"] == 32
        assert data["min"] <= data["median"] <= data["p90"] <= data["max"]

    def test_windows_gpu_adapter_counter_sums_bytes_strips_instance_names_and_handles_failure(self):
        sample_csv = (
            '"(PDH-CSV 4.0) (Eastern Standard Time)(300)",'
            '"\\\\MACHINE\\GPU Adapter Memory(luid_0x00000000_0x00012345_phys_0)\\Shared Usage",'
            '"\\\\MACHINE\\GPU Adapter Memory(luid_0x00000000_0x00067890_phys_0)\\Shared Usage",'
            '"\\\\MACHINE\\GPU Adapter Memory(luid_0x00000000_0x00012345_phys_0)\\Dedicated Usage",'
            '"\\\\MACHINE\\GPU Adapter Memory(luid_0x00000000_0x00067890_phys_0)\\Dedicated Usage"\n'
            '"11/03/2026 11:00:00.000","1000.000000","2500.000000","4000.000000","6000.000000"\n'
        )
        res = s2.read_gpu_adapter_memory(runner=lambda: sample_csv)
        assert res == {
            "shared_usage_bytes": 3500,
            "dedicated_usage_bytes": 10000,
            "error": None,
        }
        serialized = json.dumps(res)
        assert "MACHINE" not in serialized and "luid_" not in serialized

        failed = s2.read_gpu_adapter_memory(
            runner=lambda: (_ for _ in ()).throw(OSError("counter unavailable"))
        )
        assert failed["shared_usage_bytes"] is None
        assert failed["dedicated_usage_bytes"] is None
        assert "counter unavailable" in failed["error"]

    @needs_torch
    def test_probe_memory_on_fake_bundle_forces_lengths_and_records_phases_in_order(self, tmp_path, capsys):
        rows_by_file = {
            "grounded.jsonl": [
                _row(f"g{i:02d}", text="short" if i < 19 else "longest_evidence " * 30) for i in range(20)
            ],
            "prs/grounded-prs.jsonl": [_row(f"p{i:02d}") for i in range(20)],
            "diffs/reasoning-diffs.jsonl": [_row(f"r{i:02d}", label="unsupported") for i in range(20)],
        }
        gold_dir = _gold(tmp_path, rows_by_file)
        a = _args(
            tmp_path,
            gold_dir,
            probe_memory=True,
            probe_lengths="6,12",
            bf16_restore=True,
            logprob_microbatch=2,
            g=4,
        )
        fake_cuda = _FakeCuda()
        rig = _FakeRig(a)
        seen_gen_kwargs: list[dict] = []

        def build_probe(args):
            bundle = _fake_probe_bundle(args, fake_cuda)
            orig_gen = bundle["model"].generate

            def spy_gen(*g_args, **g_kwargs):
                seen_gen_kwargs.append(dict(g_kwargs))
                fake_cuda.alloc += 4 * 2**20
                fake_cuda.max_alloc = max(fake_cuda.max_alloc, fake_cuda.alloc)
                return orig_gen(*g_args, **g_kwargs)

            bundle["model"].generate = spy_gen
            return bundle

        rc = s2.execute(
            a,
            build=build_probe,
            rig=rig,
            counter_runner=lambda: (
                '"(PDH-CSV 4.0)",'
                '"\\\\H\\GPU Adapter Memory(luid_0)\\Shared Usage",'
                '"\\\\H\\GPU Adapter Memory(luid_0)\\Dedicated Usage"\n'
                '"t","1024","2048"\n'
            ),
        )
        assert rc == 0
        assert rig.preflight_called == 1
        probe_file = a.out / "s2-hf-probe.json"
        probe = json.loads(probe_file.read_text(encoding="utf-8"))
        assert probe["prompt_id"] == "g19"
        assert probe["g"] == 4
        assert probe["model_info"]["embed_tokens_dtype"] == "bfloat16"
        assert probe["model_info"]["lm_head_dtype"] == "bfloat16"
        assert probe["model_info"]["kv_cache_key_dtype"] == "bfloat16"
        assert isinstance(probe["model_info"]["use_cache"], bool)
        assert probe["model_info"]["attn_implementation"] is not None
        assert [item["length"] for item in probe["lengths"]] == [6, 12]
        assert [kw["min_new_tokens"] for kw in seen_gen_kwargs] == [6, 12]
        assert [kw["max_new_tokens"] for kw in seen_gen_kwargs] == [6, 12]
        assert all(kw.get("return_dict_in_generate") is True for kw in seen_gen_kwargs)
        for item in probe["lengths"]:
            assert list(item["phases"].keys()) == [
                "before_generate",
                "after_generate",
                "after_ref_forward",
            ]
            assert item["shared_gpu_memory"]["before_generate"]["shared_usage_bytes"] == 1024
            assert item["shared_gpu_memory"]["after_generate"]["dedicated_usage_bytes"] == 2048
        stdout = capsys.readouterr().out
        assert "alloc_mib_per_token" in stdout and "reserved_minus_allocated_mib" in stdout

    @needs_torch
    def test_probe_memory_releases_carry_over_between_lengths_and_records_allocated_after_release(
        self, tmp_path
    ):
        a = _args(
            tmp_path,
            _gold(tmp_path),
            probe_memory=True,
            probe_lengths="6,12,24",
            bf16_restore=True,
            logprob_microbatch=2,
            g=2,
        )
        fake_cuda = _FakeCuda()
        baseline_alloc = fake_cuda.alloc
        rig = _FakeRig(a)

        class _OwnedGenOutput:
            def __init__(self, sequences, past_key_values, bytes_held: int):
                self.sequences = sequences
                self.past_key_values = past_key_values
                self._bytes_held = bytes_held
                self._released = False
                fake_cuda.alloc += bytes_held
                fake_cuda.max_alloc = max(fake_cuda.max_alloc, fake_cuda.alloc)

            def __del__(self):
                if not self._released:
                    fake_cuda.alloc -= self._bytes_held
                    self._released = True

        def build_probe(args):
            bundle = _fake_probe_bundle(args, fake_cuda)
            orig_gen = bundle["model"].generate

            def releasing_gen(*g_args, **g_kwargs):
                out = orig_gen(*g_args, **g_kwargs)
                t_len = int(g_kwargs["max_new_tokens"])
                seqs = out.sequences.clone() if hasattr(out, "sequences") else out.clone()
                cache = getattr(out, "past_key_values", None)
                return _OwnedGenOutput(seqs, cache, t_len * 2**20)

            bundle["model"].generate = releasing_gen
            return bundle

        rc = s2.execute(
            a,
            build=build_probe,
            rig=rig,
            counter_runner=lambda: (
                '"(PDH-CSV 4.0)",'
                '"\\\\H\\GPU Adapter Memory(luid_0)\\Shared Usage",'
                '"\\\\H\\GPU Adapter Memory(luid_0)\\Dedicated Usage"\n'
                '"t","1024","2048"\n'
            ),
        )
        assert rc == 0
        probe = json.loads((a.out / "s2-hf-probe.json").read_text(encoding="utf-8"))
        lengths = probe["lengths"]
        assert [item["length"] for item in lengths] == [6, 12, 24]
        first_before = lengths[0]["phases"]["before_generate"]["memory_allocated"]
        assert first_before == baseline_alloc
        assert [item["phases"]["before_generate"]["memory_allocated"] for item in lengths] == [
            first_before
        ] * len(lengths)
        for item in lengths:
            assert (
                item["phases"]["after_generate"]["memory_allocated"] == first_before + item["length"] * 2**20
            )
            assert item["allocated_after_release"] == first_before
        assert "empty_cache" in fake_cuda.calls


# -- 7. Stress prefix (--stress-prefix) and prereg amendment -----------------------------------------------------------------
@needs_torch
class TestStressPrefix:
    def test_stress_prefix_runs_before_registered_steps_and_is_excluded_from_readout(self, tmp_path):
        rows_by_file = {
            "grounded.jsonl": [
                _row(f"g{i:02d}", text="short" if i < 19 else "longest_prompt_text " * 25) for i in range(20)
            ],
            "prs/grounded-prs.jsonl": [_row(f"p{i:02d}") for i in range(20)],
            "diffs/reasoning-diffs.jsonl": [_row(f"r{i:02d}", label="unsupported") for i in range(20)],
        }
        gold_dir = _gold(tmp_path, rows_by_file)
        a = s2.apply_dry_defaults(
            _args(tmp_path, gold_dir, dry=True, steps=2, g=2, max_new=10, stress_prefix=2)
        )
        opt_steps = 0

        def build_tracked(args):
            bundle = s2.dry_bundle(args)
            orig_step = bundle["optimiser"].step

            def counted_step(*s_args, **s_kwargs):
                nonlocal opt_steps
                opt_steps += 1
                return orig_step(*s_args, **s_kwargs)

            bundle["optimiser"].step = counted_step
            return bundle

        assert s2.execute(a, build=build_tracked) == 0
        assert opt_steps == 4  # 2 stress prefix steps + 2 registered steps
        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        step_lines = [
            json.loads(line)
            for line in (a.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(step_lines) == 2
        assert [s["step"] for s in step_lines] == [0, 1]
        assert receipt["readout"]["steps_completed"] == 2
        assert receipt["readout"]["steps_requested"] == 2
        assert receipt["run"]["stress_prefix"] == 2
        assert receipt["readout"]["flags"]["stress_prefix"] == 2

    def test_prereg_doc_has_2026_11_03_amendment_and_marks_stress_prefix_option_a_chosen(self):
        doc = Path(__file__).parent.parent / "docs" / "runs" / "2026-10-10-smoke-s2-prereg.md"
        text = doc.read_text(encoding="utf-8")
        assert "## 2026-11-03" in text
        for needle in (
            "--bf16-restore",
            "--logprob-microbatch",
            "--alloc-conf expandable",
            "--empty-cache-phases",
            "--attention-mask",
            "--probe-memory",
            "--prompt-lengths",
            "--stress-prefix 2",
        ):
            assert needle in text
        assert "(A)" in text and "(B)" in text and "chosen" in text.lower()

    def test_prereg_doc_has_2026_11_03_v3_run_record_with_no_exe_names_or_paths(self, tmp_path):
        doc = Path(__file__).parent.parent / "docs" / "runs" / "2026-10-10-smoke-s2-prereg.md"
        text = doc.read_text(encoding="utf-8")
        heading = "## 2026-11-03 run record: path (a) v3"
        assert heading in text
        section = text[text.index(heading) :]
        assert "Verdict: S2-OK." in section
        assert re.findall(r"[A-Za-z0-9._-]+\.exe\b", section, re.IGNORECASE) == []
        assert re.findall(r"[A-Za-z]:[\\/]", section) == []
        assert "\\" not in section
        assert re.findall(r"(?<!tokens)/(?![\s)])[A-Za-z0-9._-]+/[A-Za-z0-9._-]+", section) == []
        sec_file = tmp_path / "v3_section.md"
        sec_file.write_text(section, encoding="utf-8")
        assert _scan([sec_file]) == []
