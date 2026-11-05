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
        assert r(_reply("supported"), "supported", EVIDENCE, 8192) == pytest.approx(
            1.0 - 0.5 + 0.05
        )  # dead 8,192 hard gate removed; smooth length penalty capped at -0.5
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
        files = [
            self.ROOT / "examples" / "verifier-distill" / "s2_grpo_smoke.py",
            self.ROOT / "examples" / "verifier-distill" / "quote_rule.py",
            Path(__file__),
        ]
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


# ---------------------------------------------------------------------------
# 2026-11-03 GRPO loss (zero-variance mask, Dr. GRPO, groups-per-update),
# root truncation + unclosed thinking, offrig quote rule 2 port,
# quote-reward v2, per-group logging + evaluation summary, prereg amendment
# ---------------------------------------------------------------------------


OFFRIG_RULE2_PARITY_FIXTURE = [
    # Actual strings from offrig crates/offrig-core/src/verify.rs:1810-1822
    # (quotes_match_across_whitespace_and_curly_marks):
    {
        "name": "quotes_match_across_whitespace_and_curly_marks_found",
        "passages": ['He said\n  "it\'s   fine"  and left.'],
        "quote": "\u201cit\u2019s fine\u201d and left.",
        "expected": True,
    },
    {
        "name": "quotes_match_across_whitespace_and_curly_marks_paraphrase_not_found",
        "passages": ['He said\n  "it\'s   fine"  and left.'],
        "quote": "it is fine",
        "expected": False,
    },
    {
        "name": "quotes_match_across_whitespace_and_curly_marks_empty_not_found",
        "passages": ['He said\n  "it\'s   fine"  and left.'],
        "quote": "   ",
        "expected": False,
    },
    {
        "name": "quotes_match_across_whitespace_and_curly_marks_empty_evidence_not_found",
        "passages": [],
        "quote": "x",
        "expected": False,
    },
    # Actual strings from offrig crates/offrig-core/src/verify.rs:1824-1830
    # (a_quote_too_short_to_prove_anything_is_not_found):
    {
        "name": "a_quote_too_short_to_prove_anything_is_not_found_one_word",
        "passages": ["the cache is cleared on the next start"],
        "quote": "the",
        "expected": False,
    },
    {
        "name": "a_quote_too_short_to_prove_anything_is_not_found_under_minimum",
        "passages": ["the cache is cleared on the next start"],
        "quote": "the next",
        "expected": False,
    },
    {
        "name": "a_quote_too_short_to_prove_anything_is_not_found_long_enough",
        "passages": ["the cache is cleared on the next start"],
        "quote": "cleared on the next start",
        "expected": True,
    },
    # Additional rule 2 cases (whole short lines, diff/comment markers, cross-passage):
    {
        "name": "a_short_quote_that_is_a_whole_line_is_found_1",
        "passages": ["def f(v):\n    v = v * 3\n    return v\n}"],
        "quote": "v = v * 3",
        "expected": True,
    },
    {
        "name": "a_short_quote_that_is_a_whole_line_is_found_2",
        "passages": ["def f(v):\n    v = v * 3\n    return v\n}"],
        "quote": "return v",
        "expected": True,
    },
    {
        "name": "a_short_quote_that_is_a_whole_line_fragment_not_found",
        "passages": ["def f(v):\n    v = v * 3\n    return v\n}"],
        "quote": "v * 3",
        "expected": False,
    },
    {
        "name": "a_short_quote_that_is_a_whole_line_brace_only_not_found",
        "passages": ["def f(v):\n    v = v * 3\n    return v\n}"],
        "quote": "}",
        "expected": False,
    },
    {
        "name": "a_short_quote_that_is_a_whole_diff_line_marker_removed",
        "passages": ["@@ -1,3 +1,3 @@\n-    x += 1\n+    x += 2"],
        "quote": "x += 2",
        "expected": True,
    },
    {
        "name": "diff_and_comment_markers_stripped_multiline",
        "passages": ["+ // return Err(Error::Timeout);\n+ // retry_count += 1;"],
        "quote": "return Err(Error::Timeout); retry_count += 1;",
        "expected": True,
    },
    {
        "name": "diff_and_comment_markers_stripped_short_whole_line",
        "passages": ["  - # x = 1\n"],
        "quote": "x = 1",
        "expected": True,
    },
    {
        "name": "diff_header_double_plus_or_minus_not_stripped_as_single_diff_marker",
        "passages": ["+++ b/lib.rs\n--- a/lib.rs"],
        "quote": "b/lib.rs",
        "expected": False,
    },
    {
        "name": "cross_passage_boundary_does_not_match",
        "passages": ["alpha beta", "gamma delta"],
        "quote": "alpha beta gamma delta",
        "expected": False,
    },
    # Python/Rust edge cases (str::lines() and char::is_whitespace):
    {
        "name": "rust_edge_cr_alone_not_line_break_short_line",
        "passages": ["x = 1\ry = 2"],
        "quote": "x = 1",
        "expected": False,
    },
    {
        "name": "rust_edge_cr_alone_not_line_break_strip_markers",
        "passages": ["+ // first line\r+ // second line"],
        "quote": "first line second line",
        "expected": False,
    },
    {
        "name": "rust_edge_vt_x0b_not_line_break",
        "passages": ["x = 1\x0by = 2"],
        "quote": "x = 1",
        "expected": False,
    },
    {
        "name": "rust_edge_ff_x0c_not_line_break",
        "passages": ["x = 1\x0cy = 2"],
        "quote": "x = 1",
        "expected": False,
    },
    {
        "name": "rust_edge_nel_x85_not_line_break",
        "passages": ["x = 1\x85y = 2"],
        "quote": "x = 1",
        "expected": False,
    },
    {
        "name": "rust_edge_ls_u2028_not_line_break",
        "passages": ["x = 1\u2028y = 2"],
        "quote": "x = 1",
        "expected": False,
    },
    {
        "name": "rust_edge_ps_u2029_not_line_break",
        "passages": ["x = 1\u2029y = 2"],
        "quote": "x = 1",
        "expected": False,
    },
    {
        "name": "rust_edge_vt_ff_nel_ls_ps_collapsed_as_whitespace",
        "passages": ["alpha\x0bbeta\x0cgamma\x85delta\u2028eps\u2029zeta"],
        "quote": "alpha beta gamma delta eps zeta",
        "expected": True,
    },
    {
        "name": "rust_edge_fs_x1c_not_whitespace",
        "passages": ["alpha\x1cbeta gamma delta"],
        "quote": "alpha beta gamma delta",
        "expected": False,
    },
    {
        "name": "rust_edge_gs_x1d_not_whitespace",
        "passages": ["alpha\x1dbeta gamma delta"],
        "quote": "alpha beta gamma delta",
        "expected": False,
    },
    {
        "name": "rust_edge_rs_x1e_not_whitespace",
        "passages": ["alpha\x1ebeta gamma delta"],
        "quote": "alpha beta gamma delta",
        "expected": False,
    },
    {
        "name": "rust_edge_us_x1f_not_whitespace",
        "passages": ["alpha\x1fbeta gamma delta"],
        "quote": "alpha beta gamma delta",
        "expected": False,
    },
    {
        "name": "rust_edge_fs_gs_rs_us_preserved_in_normalise",
        "passages": ["alpha\x1cbeta\x1dgamma\x1edelta\x1feps"],
        "quote": "alpha\x1cbeta\x1dgamma\x1edelta\x1feps",
        "expected": True,
    },
]


@needs_torch
class TestZeroVarianceMask:
    def test_zero_variance_group_contributes_zero_loss_and_zero_lora_grad(self):
        import torch

        model = _tiny_qwen3(torch.float32)
        model.eval()
        model.get_base_model().gradient_checkpointing_disable()
        torch.manual_seed(7)
        for name, param in model.named_parameters():
            if param.requires_grad and "lora_B" in name:
                param.data.normal_(0.0, 0.05)

        ids = torch.tensor([[10, 11, 20, 21, 22], [10, 11, 23, 24, 25]], dtype=torch.long)
        att = torch.ones_like(ids)
        mask = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float32)
        adv = torch.tensor([0.0, 0.0], dtype=torch.float32)
        prompt_len = 2

        with torch.no_grad():
            old_lp = s2.completion_logprobs(model, ids, att, prompt_len) + 0.02
            with model.disable_adapter():
                ref_lp = s2.completion_logprobs(model, ids, att, prompt_len)
        pi_lp = s2.completion_logprobs(model, ids, att, prompt_len)

        loss = s2.grpo_loss(
            pi_lp,
            old_lp,
            ref_lp,
            adv,
            mask,
            zero_variance_mask=True,
            group_size=2,
        )
        assert float(loss.detach()) == 0.0

        model.zero_grad(set_to_none=True)
        mb_loss = s2.grpo_microbatch_backward(
            model,
            ids,
            att,
            prompt_len,
            old_lp,
            ref_lp,
            adv,
            mask,
            microbatch=1,
            zero_variance_mask=True,
            group_size=2,
        )
        assert float(mb_loss) == 0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.all(p.grad == 0.0)

    def test_equality_no_zero_variance_and_one_zero_variance_group_atol_1e6(self):
        import copy

        import torch

        torch.manual_seed(123)
        model_old = _tiny_qwen3(torch.float32)
        model_old.eval()
        model_old.get_base_model().gradient_checkpointing_disable()
        with torch.no_grad():
            for n, p in model_old.named_parameters():
                if p.requires_grad and "lora_B" in n:
                    p.uniform_(-0.05, 0.05)

        model_new = copy.deepcopy(model_old)
        prompt_len = 2

        ids_all = torch.tensor(
            [
                [10, 11, 20, 21, 22],
                [10, 11, 23, 24, 25],
                [12, 13, 30, 31, 32],
                [12, 13, 33, 34, 35],
            ],
            dtype=torch.long,
        )
        att_all = torch.tensor(
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0],
            ],
            dtype=torch.long,
        )
        mask_all = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        adv_g0 = torch.tensor(s2.advantages([1.0, -0.3]), dtype=torch.float32)
        adv_g1 = torch.tensor(s2.advantages([0.8, -1.0]), dtype=torch.float32)

        with torch.no_grad():
            old_lp_all = s2.completion_logprobs(model_old, ids_all, att_all, prompt_len).detach() + 0.03
            with model_old.disable_adapter():
                ref_lp_all = s2.completion_logprobs(model_old, ids_all, att_all, prompt_len).detach()

        # Case 1: no zero-variance group in the batch (2 active groups)
        model_old.zero_grad(set_to_none=True)
        loss_old = float(
            s2.grpo_microbatch_backward(
                model_old,
                ids_all[:2],
                att_all[:2],
                prompt_len,
                old_lp_all[:2],
                ref_lp_all[:2],
                adv_g0,
                mask_all[:2],
                microbatch=1,
                scale=1.0 / 2,
                zero_variance_mask=False,
            )
            + s2.grpo_microbatch_backward(
                model_old,
                ids_all[2:],
                att_all[2:],
                prompt_len,
                old_lp_all[2:],
                ref_lp_all[2:],
                adv_g1,
                mask_all[2:],
                microbatch=1,
                scale=1.0 / 2,
                zero_variance_mask=False,
            )
        )

        model_new.zero_grad(set_to_none=True)
        loss_new = float(
            s2.grpo_microbatch_backward(
                model_new,
                ids_all[:2],
                att_all[:2],
                prompt_len,
                old_lp_all[:2],
                ref_lp_all[:2],
                adv_g0,
                mask_all[:2],
                microbatch=1,
                scale=1.0 / 2,
                zero_variance_mask=True,
                group_size=2,
            )
            + s2.grpo_microbatch_backward(
                model_new,
                ids_all[2:],
                att_all[2:],
                prompt_len,
                old_lp_all[2:],
                ref_lp_all[2:],
                adv_g1,
                mask_all[2:],
                microbatch=1,
                scale=1.0 / 2,
                zero_variance_mask=True,
                group_size=2,
            )
        )

        assert abs(loss_old - loss_new) <= 1e-6
        for (n1, p1), (n2, p2) in zip(
            model_old.named_parameters(), model_new.named_parameters(), strict=True
        ):
            assert n1 == n2
            if p1.requires_grad:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-6)

        # Case 2: one zero-variance group (g0) and one active group (g1)
        adv_zv = torch.tensor(s2.advantages([1.0, 1.0]), dtype=torch.float32)
        adv_mixed = torch.cat([adv_zv, adv_g1], dim=0)

        model_old.zero_grad(set_to_none=True)
        loss_active_only = float(
            s2.grpo_microbatch_backward(
                model_old,
                ids_all[2:],
                att_all[2:],
                prompt_len,
                old_lp_all[2:],
                ref_lp_all[2:],
                adv_g1,
                mask_all[2:],
                microbatch=1,
                scale=1.0,
                zero_variance_mask=False,
            )
        )

        model_new.zero_grad(set_to_none=True)
        loss_masked_batch = float(
            s2.grpo_microbatch_backward(
                model_new,
                ids_all,
                att_all,
                prompt_len,
                old_lp_all,
                ref_lp_all,
                adv_mixed,
                mask_all,
                microbatch=1,
                scale=1.0,
                zero_variance_mask=True,
                group_size=2,
            )
        )

        assert abs(loss_active_only - loss_masked_batch) <= 1e-6
        for (_, p1), (_, p2) in zip(model_old.named_parameters(), model_new.named_parameters(), strict=True):
            if p1.requires_grad:
                assert torch.allclose(p1.grad, p2.grad, atol=1e-6)

    def test_cli_and_step_log_active_groups(self, tmp_path):
        gold_dir = _gold(tmp_path)
        ap = s2.build_parser()
        base_argv = ["--gold", str(gold_dir), "--rnd-commit", "e48cc90c", "--out", str(tmp_path / "out")]
        a0 = ap.parse_args(base_argv)
        assert a0.zero_variance_mask is False

        a1 = s2.apply_dry_defaults(
            ap.parse_args([*base_argv, "--dry", "--steps", "1", "--zero-variance-mask"])
        )
        assert a1.zero_variance_mask is True

        call_idx = {"n": 0}

        def build_mixed(args):
            bundle = s2.dry_bundle(args)

            def mixed_rollout(text, g, max_new, temp):
                call_idx["n"] += 1
                if call_idx["n"] == 1:
                    return [10, 11], [[20, 21] for _ in range(g)], [False] * g
                comps = [[20, 21] for _ in range(g)]
                comps[0] = [25, 26]
                return [10, 11], comps, [False] * g

            bundle["rollout"] = mixed_rollout
            return bundle

        assert s2.execute(a1, build=build_mixed) == 0
        receipt = json.loads((a1.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt["run"]["zero_variance_mask"] is True
        assert receipt["readout"]["flags"]["zero_variance_mask"] is True
        step_lines = [
            json.loads(line)
            for line in (a1.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert step_lines[0]["active_groups"] == 1
        assert step_lines[0]["zero_variance_groups"] == 1


class TestDrGrpoAndGroupsPerUpdate:
    def test_dr_grpo_advantages_are_centered_not_divided_by_std(self):
        rewards = [1.0, -0.2, -0.2, -0.2]
        adv_dr = s2.advantages(rewards, mode="dr-grpo")
        mean = sum(rewards) / len(rewards)
        assert adv_dr == pytest.approx([r - mean for r in rewards], abs=1e-6)
        adv_std = s2.advantages(rewards)
        assert adv_std != pytest.approx(adv_dr, abs=1e-3)
        assert s2.advantages([1.0, 1.0, 1.0], mode="dr-grpo") == [0.0, 0.0, 0.0]

    @needs_torch
    def test_groups_per_update_flag_and_accumulation(self, tmp_path):
        gold_dir = _gold(tmp_path)
        ap = s2.build_parser()
        base_argv = ["--gold", str(gold_dir), "--rnd-commit", "e48cc90c", "--out", str(tmp_path / "out")]
        a_def = ap.parse_args(base_argv)
        assert a_def.advantage == "std"
        assert a_def.groups_per_update == 2

        a = s2.apply_dry_defaults(
            ap.parse_args(
                [
                    *base_argv,
                    "--dry",
                    "--steps",
                    "1",
                    "--advantage",
                    "dr-grpo",
                    "--groups-per-update",
                    "4",
                ]
            )
        )
        assert a.advantage == "dr-grpo"
        assert a.groups_per_update == 4

        seen_prompts: list[str] = []

        def build_tracked(args):
            bundle = s2.dry_bundle(args)
            orig_rollout = bundle["rollout"]

            def tracking_rollout(text, g, max_new, temp):
                seen_prompts.append(text)
                return orig_rollout(text, g, max_new, temp)

            bundle["rollout"] = tracking_rollout
            return bundle

        assert s2.execute(a, build=build_tracked) == 0
        assert len(seen_prompts) == 4
        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt["readout"]["flags"]["advantage"] == "dr-grpo"
        assert receipt["readout"]["flags"]["groups_per_update"] == 4
        assert receipt["readout"]["groups"] == 4


class TestTruncationAndUnclosedThinking:
    @needs_torch
    def test_hf_rollout_reports_per_completion_truncation_and_passes_to_reward(self, tmp_path):
        import torch

        tok = s2.DryTokenizer()
        eos = tok.eos_token_id  # 1

        class _FakeGenModel:
            device = torch.device("cpu")

            def generate(self, input_ids, **kwargs):
                max_new = kwargs["max_new_tokens"]
                row0 = torch.cat([input_ids[0], torch.tensor([10] * max_new, dtype=torch.long)])
                row1 = torch.cat([input_ids[1], torch.tensor([10] * (max_new - 1) + [eos], dtype=torch.long)])
                return torch.stack([row0, row1], dim=0)

        p_ids, comps, truncs = s2.hf_rollout(
            _FakeGenModel(),
            tok,
            "hello world",
            g=2,
            max_new=4,
            temperature=0.7,
        )
        assert len(comps) == 2
        assert truncs == [True, False]

        gold_dir = _gold(tmp_path)
        a = s2.apply_dry_defaults(_args(tmp_path, gold_dir, dry=True, steps=1, g=2))
        seen_truncs: list[bool] = []

        def build_trunc(args):
            bundle = s2.dry_bundle(args)
            bundle["rollout"] = lambda text, g, max_new, temp: (
                [10, 11],
                [[20, 21, 22, 23], [20, 21, eos]],
                [True, False],
            )
            bundle["reward_fn"] = lambda text, row, tt, truncated=False: (
                seen_truncs.append(bool(truncated)) or (-1.0 if truncated else 1.0)
            )
            return bundle

        assert s2.execute(a, build=build_trunc) == 0
        assert seen_truncs == [True, False, True, False]

    def test_unclosed_thinking_block_never_parsed_for_verdict(self):
        raw = "<think>Considering " + json.dumps(
            {
                "verdict": "supported",
                "evidence_quotes": [{"part": "p", "verdict": "supported", "quote": "alpha beta"}],
            }
        )
        thinking, answer = s2.split_reply(raw)
        assert answer == ""
        assert "Considering" in thinking
        assert s2.parse_reply(raw) is None
        assert s2.reward(raw, "supported", EVIDENCE, 50, truncated=False) == -1.0

    @needs_torch
    def test_mask_truncated_leaves_truncated_completions_out_of_loss(self, tmp_path):
        gold_dir = _gold(tmp_path)
        ap = s2.build_parser()
        a = s2.apply_dry_defaults(
            ap.parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / "out"),
                    "--dry",
                    "--steps",
                    "1",
                    "--mask-truncated",
                ]
            )
        )
        assert a.mask_truncated is True

        def build_masked(args):
            bundle = s2.dry_bundle(args)
            bundle["rollout"] = lambda text, g, max_new, temp: (
                [10, 11],
                [[20, 21, 22], [20, 21, 1]],
                [True, False],
            )
            bundle["reward_fn"] = lambda text, row, tt, truncated=False: -1.0 if truncated else 1.0
            return bundle

        assert s2.execute(a, build=build_masked) == 0
        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt["readout"]["flags"]["mask_truncated"] is True
        step_lines = [
            json.loads(line)
            for line in (a.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        # With completion 0 truncated and masked out, only 1 non-truncated completion remains per group -> zero variance
        assert step_lines[0]["zero_variance_groups"] == 2

    def test_dead_8192_thinking_token_gate_removed(self):
        val = s2.reward(_reply("supported", "alpha beta"), "supported", EVIDENCE, 8192, truncated=False)
        assert val == pytest.approx(1.0 - 0.5 + 0.05)


class TestOffrigQuoteRuleParity:
    def test_offrig_rule_2_parity_fixture(self):
        import quote_rule as qr

        assert qr.QUOTE_RULE == 2
        assert qr.MIN_QUOTE_CHARS == 12
        assert qr.MIN_LINE_QUOTE_CHARS == 5

        for case in OFFRIG_RULE2_PARITY_FIXTURE:
            got = qr.quote_found(case["quote"], case["passages"])
            assert got is case["expected"], f"Mismatch on {case['name']}: got {got}"

    def test_gold_row_context_maps_to_passage_list_not_joined_string(self):
        row = {
            "id": "p1",
            "claim": "alpha beta gamma delta",
            "label": "supported",
            "context": [
                {"source": "a.rs:1", "text": "alpha beta"},
                {"source": "b.rs:2", "text": "gamma delta"},
            ],
        }
        passages = s2.passages_of(row)
        assert passages == ["alpha beta", "gamma delta"]
        assert s2.reward(_reply("supported", "alpha beta gamma delta"), "supported", passages, 5) == -3.0


class TestQuoteRewardV2:
    PASSAGES = [
        "alpha beta gamma delta epsilon zeta eta theta",
        "second passage with status_code = 404 and retry_limit = 3",
    ]

    def test_table_rows_under_quote_reward_v2(self):
        # 1. gold=supported, verdict=supported, m=k (1/1) -> +1.0 + 0.05 = +1.05
        assert s2.reward(
            _reply("supported", "alpha beta gamma"),
            "supported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(1.05)

        # 2. gold=supported, verdict=supported, 0 < m < k (1/2) -> -0.2 + 1.2*(1/2) = +0.4
        parts_1_of_2 = [
            {"part": "p1", "verdict": "supported", "quote": "alpha beta gamma"},
            {"part": "p2", "verdict": "supported", "quote": "missing quote text here"},
        ]
        assert s2.reward(
            _reply("supported", parts=parts_1_of_2),
            "supported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(0.4)

        # 3. gold=supported, verdict=supported, m=0 -> -0.2
        assert s2.reward(
            _reply("supported", "completely fabricated quote"),
            "supported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(-0.2)

        # Default v1 stays -3.0 on missed quote
        assert s2.reward(
            _reply("supported", "completely fabricated quote"),
            "supported",
            self.PASSAGES,
            10,
        ) == pytest.approx(-3.0)

        # 4. gold=unsupported, verdict=supported, all found -> -3.0 (false accept)
        assert s2.reward(
            _reply("supported", "alpha beta gamma"),
            "unsupported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(-3.0)

        # 5. gold=unsupported, verdict=supported, any missed -> -3.2 (fabricating never helps)
        assert s2.reward(
            _reply("supported", "completely fabricated quote"),
            "unsupported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(-3.2)

        # 6. gold=unsupported, verdict=unsupported, none given -> +1.0
        assert s2.reward(
            _reply("unsupported", ""),
            "unsupported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(1.0)

        # 7. gold=unsupported, verdict=unsupported, given and all found (>= 20 chars) -> +1.0 + 0.05 = +1.05
        assert s2.reward(
            _reply("unsupported", "alpha beta gamma delta"),
            "unsupported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(1.05)

        # 8. gold=unsupported, verdict=unsupported, given and any missed -> +0.8
        assert s2.reward(
            _reply("unsupported", "completely fabricated quote"),
            "unsupported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(0.8)

        # 9. decidable gold, verdict=cannot_tell -> -0.3
        assert s2.reward(
            _reply("cannot_tell", ""),
            "supported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(-0.3)

        # 10. gold=supported, verdict=unsupported -> -1.0
        assert s2.reward(
            _reply("unsupported", ""),
            "supported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(-1.0)

    def test_anti_farming_duplicates_overlap_density_unsupported_length_and_4way_split(self):
        # Duplicate quotes earn once (+0.05)
        dup_parts = [
            {"part": "p1", "verdict": "supported", "quote": "alpha beta gamma"},
            {"part": "p2", "verdict": "supported", "quote": "alpha   beta   gamma"},
        ]
        assert s2.reward(
            _reply("supported", parts=dup_parts),
            "supported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(1.05)

        # Overlapping spans (>= 25% of shorter span in same passage) earn only once
        overlap_parts = [
            {"part": "p1", "verdict": "supported", "quote": "alpha beta gamma delta"},
            {"part": "p2", "verdict": "supported", "quote": "beta gamma delta"},
        ]
        assert s2.reward(
            _reply("supported", parts=overlap_parts),
            "supported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(1.05)

        # Low density (< 8 alphanumeric chars, e.g. whole short line "x = 1") is found by rule 2 but gets 0 credit
        assert s2.reward(
            _reply("supported", "x = 1"),
            "supported",
            ["x = 1\nalpha beta gamma"],
            10,
            quote_reward="v2",
        ) == pytest.approx(1.0)

        # Unsupported quote < 20 chars gets 0 credit (+1.0 total)
        assert s2.reward(
            _reply("unsupported", "alpha beta gamma"),
            "unsupported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(1.0)

        # 4-way split of a simple claim:
        # (a) 4 overlapping quotes from the same span earn only +0.05
        split_same = [
            {"part": "1", "verdict": "supported", "quote": "alpha beta gamma delta"},
            {"part": "2", "verdict": "supported", "quote": "alpha beta gamma"},
            {"part": "3", "verdict": "supported", "quote": "beta gamma delta"},
            {"part": "4", "verdict": "supported", "quote": "alpha beta gamma delta epsilon"},
        ]
        assert s2.reward(
            _reply("supported", parts=split_same),
            "supported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(1.05)

        # (b) 4 genuinely disjoint spans are capped at +0.15
        split_disjoint = [
            {"part": "1", "verdict": "supported", "quote": "alpha beta gamma"},
            {"part": "2", "verdict": "supported", "quote": "delta epsilon zeta"},
            {"part": "3", "verdict": "supported", "quote": "second passage with"},
            {"part": "4", "verdict": "supported", "quote": "status_code = 404 and"},
        ]
        assert s2.reward(
            _reply("supported", parts=split_disjoint),
            "supported",
            self.PASSAGES,
            10,
            quote_reward="v2",
        ) == pytest.approx(1.15)


class TestPerGroupLoggingAndEvaluationSummary:
    @needs_torch
    def test_per_group_logging_gpu_vs_cpu_and_zero_variance_labels(self, tmp_path):
        assert (
            s2.classify_zero_variance_group(
                [{"correct": True, "quote_credit": 0.05, "truncated_or_malformed": False} for _ in range(2)]
            )
            == "all correct with full credit"
        )
        assert (
            s2.classify_zero_variance_group(
                [{"correct": True, "quote_credit": 0.0, "truncated_or_malformed": False} for _ in range(2)]
            )
            == "all correct without quote credit"
        )
        assert (
            s2.classify_zero_variance_group(
                [{"correct": False, "quote_credit": 0.0, "truncated_or_malformed": False} for _ in range(2)]
            )
            == "all wrong"
        )
        assert (
            s2.classify_zero_variance_group(
                [{"correct": False, "quote_credit": 0.0, "truncated_or_malformed": True} for _ in range(2)]
            )
            == "all truncated or malformed"
        )

        # Dry/CPU run -> GroupLogger is a no-op
        gold_dir = _gold(tmp_path)
        a_dry = s2.apply_dry_defaults(_args(tmp_path / "dry", gold_dir, dry=True, steps=1, g=2))
        assert s2.execute(a_dry, build=s2.dry_bundle) == 0
        assert not (a_dry.out / "s2-hf-groups.jsonl").exists()

        # GPU run -> writes s2-hf-groups.jsonl with required fields and no text/quotes
        a_gpu = _args(tmp_path / "gpu", gold_dir, steps=1, g=2, max_new=8)
        fake_cuda = _FakeCuda()
        rig = _FakeRig(a_gpu)

        def build_gpu(args):
            bundle = s2.dry_bundle(args)
            bundle["cuda"] = fake_cuda
            bundle["peak_alloc_mib"] = lambda: int(fake_cuda.max_memory_allocated() / 2**20)
            return bundle

        assert s2.execute(a_gpu, build=build_gpu, rig=rig) == 1  # cut run -> S2-LIMITED
        g_file = a_gpu.out / "s2-hf-groups.jsonl"
        assert g_file.exists()
        g_lines = [json.loads(line) for line in g_file.read_text(encoding="utf-8").splitlines()]
        assert len(g_lines) == 2
        required_keys = {
            "step",
            "group",
            "prompt_id",
            "prompt_tokens",
            "k_correct",
            "g",
            "reward_std",
            "verdict_reward_std",
            "truncations",
            "mean_thinking_tokens",
            "zero_variance_label",
        }
        for rec in g_lines:
            assert required_keys.issubset(rec.keys())
            assert "claim" not in rec
            assert "quotes" not in rec
            assert "context" not in rec

    def test_evaluation_summary_metrics_and_7_rung_waterfall(self):
        records = [
            {
                "group": 0,
                "gold": "supported",
                "claim": "The handler sets retry_limit to 3 when timeout occurs.",
                "passages": [
                    "When timeout occurs, the handler sets retry_limit = 3 in config.",
                    "Fallback path logs a warning.",
                ],
                "rollouts": [
                    {
                        "verdict": "supported",
                        "quotes": ["handler sets retry_limit = 3 in config"],
                        "reward": 1.05,
                    },
                    {
                        "verdict": "cannot_tell",
                        "quotes": [],
                        "reward": -0.3,
                    },
                ],
            },
            {
                "group": 1,
                "gold": "supported",
                "claim": "Timeout triggers fallback warning.",
                "passages": [
                    "First part of evidence",
                    "Second part of evidence",
                ],
                "rollouts": [
                    {
                        "verdict": "supported",
                        "quotes": [
                            "First part",
                            "FIRST PART OF EVIDENCE",
                            "First part, of evidence!",
                            "First part ... of evidence",
                            "First part of evidence Second part",
                            "First part of the evidence",
                        ],
                        "reward": -3.0,
                    },
                    {
                        "verdict": "cannot_tell",
                        "quotes": [],
                        "reward": -0.3,
                    },
                ],
            },
            {
                "group": 2,
                "gold": "unsupported",
                "claim": "It deletes the database.",
                "passages": ["Read-only query path."],
                "rollouts": [
                    {
                        "verdict": "supported",
                        "quotes": ["Read-only query path."],
                        "reward": -3.0,
                    },
                    {
                        "verdict": "unsupported",
                        "quotes": [],
                        "reward": 1.0,
                    },
                ],
            },
        ]
        summary = s2.evaluation_summary(records)
        assert summary["supported_pred_to_prior_ratio"] == pytest.approx((3 / 6) / (4 / 6))
        assert summary["p_cannot_tell_given_gold_supported"] == pytest.approx(2 / 4)
        assert summary["gold_supported_groups_with_positive_cannot_tell_advantage"] == pytest.approx(1 / 2)
        assert summary["false_accepts"] == 1
        assert summary["false_rejects"] == 0
        wf = summary["miss_waterfall"]
        assert wf["rule_2"] == 0
        assert wf["no_minimum"] == 1
        assert wf["case"] == 1
        assert wf["punctuation"] == 1
        assert wf["ellipsis_split"] == 1
        assert wf["joined_passages"] == 1
        assert wf["fuzzy"] == 1
        assert summary["quote_relevance"]["relevant"] == 1
        assert summary["quote_relevance"]["total"] == 2
        assert summary["quote_relevance"]["share"] == pytest.approx(0.5)
        raw_json = json.dumps(summary)
        assert "retry_limit" not in raw_json
        assert "First part" not in raw_json


class TestPreregAmendmentLossAndQuote:
    def test_prereg_doc_has_loss_and_quote_amendment(self):
        doc = Path(__file__).parent.parent / "docs" / "runs" / "2026-10-10-smoke-s2-prereg.md"
        text = doc.read_text(encoding="utf-8")
        for needle in (
            "--zero-variance-mask",
            "--advantage dr-grpo",
            "--groups-per-update",
            "--mask-truncated",
            "--quote-reward v2",
            "QUOTE_RULE = 2",
        ):
            assert needle in text


class TestFollowup2EvalSummaryParityEdgeCasesAndLoss:
    def test_offrig_named_parity_strings_and_rust_lines_whitespace_edge_cases(self):
        import quote_rule as qr

        by_name = {c["name"]: c for c in OFFRIG_RULE2_PARITY_FIXTURE}
        assert by_name["quotes_match_across_whitespace_and_curly_marks_found"]["passages"] == [
            'He said\n  "it\'s   fine"  and left.'
        ]
        assert by_name["a_quote_too_short_to_prove_anything_is_not_found_long_enough"]["passages"] == [
            "the cache is cleared on the next start"
        ]
        assert qr.normalise("  a \n\t b\u00a0c ") == "a b c"
        assert qr.normalise("\u201cit\u2019s\u201d") == '"it\'s"'

        # Rust str::lines() vs Python splitlines():
        # \r alone, \x0b, \x0c, \x85, \u2028, \u2029 are NOT line breaks in Rust str::lines()
        for sep in ("\r", "\x0b", "\x0c", "\x85", "\u2028", "\u2029"):
            assert qr.quote_found("x = 1", [f"x = 1{sep}y = 2"]) is False, repr(sep)
        assert qr.quote_found("first line second line", ["+ // first line\r+ // second line"]) is False
        assert qr.quote_found("x = 1", ["x = 1\r\ny = 2"]) is True

        # Rust char::is_whitespace vs Python str.split():
        # \x1c-\x1f are NOT whitespace in Rust
        for ctrl in ("\x1c", "\x1d", "\x1e", "\x1f"):
            assert qr.normalise(f"a{ctrl}b") == f"a{ctrl}b", repr(ctrl)
            assert qr.quote_found("alpha beta gamma delta", [f"alpha{ctrl}beta gamma delta"]) is False
            assert qr.quote_found(f"alpha{ctrl}beta gamma delta", [f"alpha{ctrl}beta gamma delta"]) is True

    @needs_torch
    def test_fake_run_of_2_steps_writes_eval_summary_and_readout_with_no_item_text(self, tmp_path):
        marker_claim = "".join(("PLANTED", "_", "CLAIM", "_", "TEXT", "_", "9941"))
        marker_passage = "".join(("PLANTED", "_", "PASSAGE", "_", "TEXT", "_", "8832"))
        marker_quote = "".join(("PLANTED", "_", "QUOTE", "_", "TEXT", "_", "7723"))

        # Build gold where the first 4 tune items (2 steps * 2 groups/step) have known labels and passages:
        # g00: gold=supported, passage has marker_passage + " handler sets retry_limit = 3 in config"
        # g01: gold=supported, passage has "First part of evidence" and "Second part of evidence"
        # g02: gold=unsupported, passage has "Read-only query path."
        # g03: gold=supported, passage has "When timeout occurs, retry_limit = 3"
        g_rows = []
        for i in range(20):
            if i == 0:
                g_rows.append(
                    {
                        "id": "g00",
                        "check_type": "grounded",
                        "claim": f"{marker_claim} handler sets retry_limit to 3.",
                        "label": "supported",
                        "split": "tune",
                        "context": [
                            {
                                "source": "src@g00",
                                "text": f"{marker_passage} handler sets retry_limit = 3 in config.",
                            }
                        ],
                    }
                )
            elif i == 1:
                g_rows.append(
                    {
                        "id": "g01",
                        "check_type": "grounded",
                        "claim": f"{marker_claim} Timeout triggers fallback warning.",
                        "label": "supported",
                        "split": "tune",
                        "context": [
                            {"source": "src@g01a", "text": "First part of evidence"},
                            {"source": "src@g01b", "text": "Second part of evidence"},
                        ],
                    }
                )
            elif i == 2:
                g_rows.append(
                    {
                        "id": "g02",
                        "check_type": "grounded",
                        "claim": f"{marker_claim} It deletes the database.",
                        "label": "unsupported",
                        "split": "tune",
                        "context": [{"source": "src@g02", "text": "Read-only query path."}],
                    }
                )
            elif i == 3:
                g_rows.append(
                    {
                        "id": "g03",
                        "check_type": "grounded",
                        "claim": f"{marker_claim} Timeout sets retry_limit.",
                        "label": "supported",
                        "split": "tune",
                        "context": [
                            {"source": "src@g03", "text": "When timeout occurs, retry_limit = 3 in config."}
                        ],
                    }
                )
            else:
                g_rows.append(_row(f"g{i:02d}"))

        gold_dir = _gold(
            tmp_path,
            {
                "grounded.jsonl": g_rows,
                "prs/grounded-prs.jsonl": [_row(f"p{i:02d}") for i in range(20)],
                "diffs/reasoning-diffs.jsonl": [_row(f"r{i:02d}", label="unsupported") for i in range(20)],
            },
        )
        a = s2.apply_dry_defaults(_args(tmp_path, gold_dir, dry=True, steps=2, g=2, max_new=64))

        # Scripted replies for the 4 groups (2 completions each = 8 rollouts total):
        # Group 0 (g00, gold=supported):
        #   r0: verdict=supported, quote="handler sets retry_limit = 3 in config" (found, relevant -> relevant=1, total=1)
        #   r1: verdict=cannot_tell, quote="short" (not found, no_minimum=0 since not in passage -> unmatched=1)
        # Group 1 (g01, gold=supported):
        #   r0: verdict=supported, quotes=["First part", "FIRST PART OF EVIDENCE", marker_quote]
        #       -> "First part" is no_minimum, "FIRST PART OF EVIDENCE" is case, marker_quote is unmatched.
        #       Because unfound supported quote -> reward = -3.0.
        #   r1: verdict=cannot_tell, quote="First part, of evidence!" (punctuation=1)
        #       -> reward = -0.3 > -3.0, so cannot_tell has positive advantage in this gold-supported group!
        # Group 2 (g02, gold=unsupported):
        #   r0: verdict=supported, quote="Read-only query path." (found, not relevant -> relevant=0, total=1; false_accepts=1)
        #   r1: verdict=unsupported, quote="First part ... of evidence" (unmatched=1)
        # Group 3 (g03, gold=supported):
        #   r0: verdict=unsupported, quote="When timeout occurs, retry_limit = 3" (found, but verdict=unsupported; false_rejects=1)
        #   r1: verdict=supported, quote="When timeout occurs, retry_limit = 3" (found, relevant -> relevant=1, total=1)
        scripted_replies = [
            [
                _reply("supported", "handler sets retry_limit = 3 in config"),
                _reply("cannot_tell", "absent_xyz"),
            ],
            [
                _reply(
                    "supported",
                    parts=[
                        {"part": "p1", "verdict": "supported", "quote": "First part"},
                        {"part": "p2", "verdict": "supported", "quote": "FIRST PART OF EVIDENCE"},
                        {"part": "p3", "verdict": "supported", "quote": marker_quote},
                    ],
                ),
                _reply("cannot_tell", "First part, of evidence!"),
            ],
            [
                _reply("supported", "Read-only query path."),
                _reply("unsupported", "absent_uvw"),
            ],
            [
                _reply("unsupported", "When timeout occurs, retry_limit = 3"),
                _reply("supported", "When timeout occurs, retry_limit = 3"),
            ],
        ]
        grp_counter = {"idx": 0}

        def build_scripted(args):
            bundle = s2.dry_bundle(args)
            tok = bundle["tok"]

            def scripted_rollout(text, g, max_new, temp):
                idx = grp_counter["idx"]
                grp_counter["idx"] += 1
                replies = scripted_replies[idx]
                p_ids = tok.encode(text)
                comps = [tok.encode(rep) + [tok.eos_token_id] for rep in replies]
                return p_ids, comps, [False] * len(comps)

            bundle["rollout"] = scripted_rollout
            bundle["reward_fn"] = lambda text, row, tt, truncated=False: s2.reward(
                text,
                row["label"],
                s2.passages_of(row),
                tt,
                truncated=truncated,
            )
            return bundle

        assert s2.execute(a, build=build_scripted) == 0

        summary_path = a.out / "s2-hf-eval-summary.json"
        assert summary_path.exists()
        raw_summary = summary_path.read_text(encoding="utf-8")
        for m in (marker_claim, marker_passage, marker_quote):
            assert m not in raw_summary
        receipt_raw = (a.out / "s2-hf.json").read_text(encoding="utf-8")
        for m in (marker_claim, marker_passage, marker_quote):
            assert m not in receipt_raw

        summary = json.loads(raw_summary)
        # Hand-computed expectations over 8 rollouts (4 groups * 2):
        # pred_supported = 4 (g0.r0, g1.r0, g2.r0, g3.r1) -> pred_rate = 4/8 = 0.5
        # gold_supported_rollouts = 6 (g0, g1, g3 each have 2) -> prior_rate = 6/8 = 0.75
        # supported_pred_to_prior_ratio = (4/8) / (6/8) = 4/6 = 2/3
        assert summary["supported_pred_to_prior_ratio"] == pytest.approx(4 / 6)
        # cannot_tell on gold_supported = 2 (g0.r1, g1.r1) out of 6 -> 2/6 = 1/3
        assert summary["p_cannot_tell_given_gold_supported"] == pytest.approx(2 / 6)
        # gold_supported_groups = 3 (g0, g1, g3); positive cannot_tell advantage in g1 only -> 1/3
        assert summary["gold_supported_groups_with_positive_cannot_tell_advantage"] == pytest.approx(1 / 3)
        # false_accepts = 1 (g2.r0), false_rejects = 1 (g3.r0)
        assert summary["false_accepts"] == 1
        assert summary["false_rejects"] == 1
        # miss_waterfall: no_minimum=1, case=1, punctuation=1, unmatched=3 (absent_xyz, marker_quote, absent_uvw)
        assert summary["miss_waterfall"] == {
            "rule_2": 0,
            "no_minimum": 1,
            "case": 1,
            "punctuation": 1,
            "ellipsis_split": 0,
            "joined_passages": 0,
            "fuzzy": 0,
            "unmatched": 3,
        }
        # quote_relevance: 3 found quotes on supported verdicts (g0.r0 relevant, g2.r0 not relevant, g3.r1 relevant) -> 2/3
        assert summary["quote_relevance"]["relevant"] == 2
        assert summary["quote_relevance"]["total"] == 3
        assert summary["quote_relevance"]["share"] == pytest.approx(2 / 3)

        # Headline numbers in readout
        receipt = json.loads(receipt_raw)
        ro = receipt["readout"]
        assert "eval_summary" in ro
        assert ro["eval_summary"]["supported_pred_to_prior_ratio"] == pytest.approx(4 / 6)
        assert ro["eval_summary"]["p_cannot_tell_given_gold_supported"] == pytest.approx(2 / 6)
        assert ro["eval_summary"][
            "gold_supported_groups_with_positive_cannot_tell_advantage"
        ] == pytest.approx(1 / 3)
        assert ro["eval_summary"]["false_accepts"] == 1
        assert ro["eval_summary"]["false_rejects"] == 1
        assert ro["eval_summary"]["quote_relevance_share"] == pytest.approx(2 / 3)

    @needs_torch
    def test_logged_step_loss_is_mean_over_step_groups_in_both_mask_off_and_mask_on_modes(self, tmp_path):
        import copy

        import torch

        gold_dir = _gold(tmp_path)

        # Group 0: zero-variance (all completions identical -> reward equal -> advantage 0, group loss 0 under mask,
        # or KL=0 and policy=0 when old == pi == ref).
        # Let's make old != pi or ref != pi so we can also test active vs zero-variance groups!
        # Wait: in run_steps, old is computed from current model, and ref from model with adapter off.
        # If LoRA B weights are non-zero, ref != pi, so KL > 0 on an unmasked group, and on an active group policy+KL != 0!
        def run_mode(zero_variance_mask: bool):
            a = s2.apply_dry_defaults(
                _args(
                    tmp_path / ("on" if zero_variance_mask else "off"),
                    gold_dir,
                    dry=True,
                    steps=1,
                    g=2,
                    groups_per_update=2,
                    zero_variance_mask=zero_variance_mask,
                )
            )
            bundle = s2.dry_bundle(a)
            torch.manual_seed(99)
            with torch.no_grad():
                for n, p in bundle["model"].named_parameters():
                    if p.requires_grad and "lora_B" in n:
                        p.normal_(0.0, 0.2)

            model_snap = copy.deepcopy(bundle["model"])
            model_snap.eval()
            model_snap.get_base_model().gradient_checkpointing_disable()

            call_n = {"n": 0}
            grp_Data = []

            def rollout(text, g, max_new, temp):
                call_n["n"] += 1
                p_ids = [10, 11, 12]
                if call_n["n"] == 1:
                    # Group 0: identical completions -> zero variance
                    comps = [[20, 21, 22, 1], [20, 21, 22, 1]]
                else:
                    # Group 1: distinct completions -> active group
                    comps = [[20, 21, 22, 1], [30, 31, 32, 1]]
                grp_Data.append((p_ids, comps))
                return p_ids, comps, [False, False]

            bundle["rollout"] = rollout
            s2.execute(a, build=lambda _: bundle)
            step_rec = json.loads((a.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()[0])

            # Compute each group's unscaled loss directly on model_snap (before optimizer.step())
            grp_losses = []
            for idx, (p_ids, comps) in enumerate(grp_Data):
                rewards = [bundle["reward_fn"](bundle["tok"].decode(c), {}, 0) for c in comps]
                adv_t = torch.tensor(s2.advantages(rewards), dtype=torch.float32)
                ids_t = torch.tensor([p_ids + c for c in comps], dtype=torch.long)
                att_t = torch.ones_like(ids_t)
                mask_t = torch.ones((2, len(comps[0])), dtype=torch.float32)
                with torch.no_grad():
                    old_lp = s2.completion_logprobs(model_snap, ids_t, att_t, len(p_ids))
                    with model_snap.disable_adapter():
                        ref_lp = s2.completion_logprobs(model_snap, ids_t, att_t, len(p_ids))
                    gl = s2.grpo_loss(
                        old_lp,
                        old_lp,
                        ref_lp,
                        adv_t,
                        mask_t,
                        zero_variance_mask=zero_variance_mask,
                        group_size=2,
                    )
                    grp_losses.append(float(gl))

            expected_mean_over_groups = sum(grp_losses) / len(grp_losses)
            return step_rec["loss"], round(expected_mean_over_groups, 6), grp_losses

        loss_off, exp_off, gl_off = run_mode(False)
        loss_on, exp_on, gl_on = run_mode(True)
        assert gl_off[0] > 0.0 and gl_off[1] > 0.0
        assert gl_on[0] == 0.0 and gl_on[1] > 0.0
        assert loss_off == pytest.approx(exp_off, abs=1e-6)
        assert loss_on == pytest.approx(exp_on, abs=1e-6)

        # Also check prereg amendment dated 2026-11-03
        doc = Path(__file__).parent.parent / "docs" / "runs" / "2026-10-10-smoke-s2-prereg.md"
        doc_text = doc.read_text(encoding="utf-8")
        assert "2026-11-03" in doc_text
        assert "mean over the step's groups of each group's loss" in doc_text


def _run_stress_prefix_fake(tmp_path: Path, monkeypatch, *, force_base_prefix_masks: bool = False) -> dict:
    from contextlib import nullcontext

    import torch

    monkeypatch.setattr(s2.time, "time", lambda: 1000.0)
    rows_by_file = {
        "grounded.jsonl": [
            _row(f"g{i:02d}", text="short" if i < 19 else "longest_stress_prompt " * 20) for i in range(20)
        ],
        "prs/grounded-prs.jsonl": [_row(f"p{i:02d}") for i in range(20)],
        "diffs/reasoning-diffs.jsonl": [_row(f"r{i:02d}", label="unsupported") for i in range(20)],
    }
    gold_dir = _gold(tmp_path / "gold", rows_by_file)
    ap = s2.build_parser()
    a = s2.apply_dry_defaults(
        ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "out"),
                "--dry",
                "--steps",
                "1",
                "--g",
                "4",
                "--max-new",
                "12",
                "--stress-prefix",
                "1",
                "--mask-truncated",
                "--zero-variance-mask",
                "--groups-per-update",
                "2",
                "--logprob-microbatch",
                "2",
            ]
        )
    )
    tok = s2.DryTokenizer()
    all_rows, _ = s2.read_gold(a.gold)
    selected_prompts = s2.select_prompts(all_rows, a.prompts)
    longest_row = max(selected_prompts, key=lambda r: (len(tok.encode(s2.build_prompt(r))), r["id"]))
    longest_prompt_len = len(tok.encode(s2.build_prompt(longest_row)))

    phase = {"stage": "prefix", "reg_group": -1}
    prefix_train_widths: list[int] = []
    reg_train_widths_by_group: dict[int, list[int]] = {0: [], 1: []}

    class _FakeModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def parameters(self):
            return [self._param]

        def generate(self, input_ids, **kwargs):
            max_n = int(kwargs["max_new_tokens"])
            cap_fill = torch.full((input_ids.shape[0], max_n), 10, dtype=torch.long)
            return torch.cat([input_ids, cap_fill], dim=1)

        def __call__(self, input_ids, attention_mask=None, logits_to_keep=None, **kwargs):
            if torch.is_grad_enabled():
                width = int(input_ids.shape[1])
                if phase["stage"] == "prefix":
                    prefix_train_widths.append(width)
                else:
                    grp = phase["reg_group"]
                    reg_train_widths_by_group[grp].append(width)
            logits = (
                torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], tok.vocab_size),
                    dtype=torch.float32,
                )
                + self._param * 0.0
            )
            return Namespace(logits=logits)

    fake_model = _FakeModel()

    def fake_reg_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
        phase["stage"] = "registered"
        phase["reg_group"] += 1
        grp = phase["reg_group"]
        p_ids = tok.encode(prompt_text)
        if grp == 0:
            # Group 0: row 0 truncated at max_n, rows 1..3 non-truncated with identical replies/rewards
            comps = [
                [10] * max_n,
                [20, 21, 22, tok.eos_token_id],
                [20, 21, 22, tok.eos_token_id],
                [20, 21, 22, tok.eos_token_id],
            ]
            truncs = [True, False, False, False]
        else:
            # Group 1: microbatch 0 (rows 0, 1) truncated; microbatch 1 (rows 2, 3) non-truncated with distinct rewards
            comps = [
                [10] * max_n,
                [10] * max_n,
                [20, 21, 22, tok.eos_token_id],
                [30, 31, 32, tok.eos_token_id],
            ]
            truncs = [True, True, False, False]
        return p_ids, comps, truncs

    def fake_reward(text: str, row: dict, tt: int, truncated: bool = False) -> float:
        if truncated:
            return -1.0
        if text == tok.decode([30, 31, 32]):
            return -0.3
        return 1.0

    def build_fake(_args_ns):
        return {
            "model": fake_model,
            "tok": tok,
            "mode": lambda training: None,
            "state": lambda: {"grad_ckpt": True},
            "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
            "ref_adapter_off": nullcontext,
            "rollout": fake_reg_rollout,
            "reward_fn": fake_reward,
            "sync": lambda: None,
            "peak_alloc_mib": lambda: None,
            "revision": None,
            "llama": None,
            "dtype_census": None,
        }

    if force_base_prefix_masks:
        orig_run_steps = s2.run_steps
        call_n = {"n": 0}

        def base_run_steps(**kw):
            call_n["n"] += 1
            if call_n["n"] == 1:
                kw["mask_truncated"] = True
                kw["zero_variance_mask"] = True
            return orig_run_steps(**kw)

        monkeypatch.setattr(s2, "run_steps", base_run_steps)

    rc = s2.execute(a, build=build_fake)
    assert rc == 0
    step_rows = [
        json.loads(line) for line in (a.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
    return {
        "args": a,
        "longest_prompt_len": longest_prompt_len,
        "prefix_train_widths": prefix_train_widths,
        "reg_train_widths_by_group": reg_train_widths_by_group,
        "step_rows": step_rows,
        "receipt": receipt,
    }


class TestStressPrefixMasksOff:
    @needs_torch
    def test_t1_prefix_trains_at_full_width_under_plan_flags(self, tmp_path, monkeypatch):
        run = _run_stress_prefix_fake(tmp_path, monkeypatch)
        a = run["args"]
        expected_count = a.groups_per_update * a.g // a.logprob_microbatch
        expected_width = run["longest_prompt_len"] + a.max_new
        assert len(run["prefix_train_widths"]) == expected_count
        assert run["prefix_train_widths"] == [expected_width] * expected_count

    @needs_torch
    def test_t2_registered_steps_unchanged_and_equal_to_base(self, tmp_path, monkeypatch):
        head_run = _run_stress_prefix_fake(tmp_path / "head", monkeypatch, force_base_prefix_masks=False)
        base_run = _run_stress_prefix_fake(tmp_path / "base", monkeypatch, force_base_prefix_masks=True)

        # Registered group 0 (1 truncated row + 3 identical rewards) masked the truncated row, became
        # zero-variance, and skipped both all-masked microbatches:
        assert head_run["reg_train_widths_by_group"][0] == []
        # Registered group 1 skipped microbatch 0 (both rows truncated -> all-masked) and ran only microbatch 1:
        assert len(head_run["reg_train_widths_by_group"][1]) == 1
        assert head_run["reg_train_widths_by_group"] == base_run["reg_train_widths_by_group"]

        # Registered step rows match base across all fields:
        assert head_run["step_rows"] == base_run["step_rows"]
        assert head_run["step_rows"][0]["zero_variance_groups"] == 1
        assert head_run["step_rows"][0]["active_groups"] == 1
        # While in the same run, the prefix did run all microbatches (unlike base where prefix ran 0):
        assert len(head_run["prefix_train_widths"]) == 4 and len(base_run["prefix_train_widths"]) == 0

    @needs_torch
    def test_t3_readout_stress_prefix_masks_off_when_on_and_absent_when_off(self, tmp_path, monkeypatch):
        on_run = _run_stress_prefix_fake(tmp_path / "on", monkeypatch)
        assert on_run["receipt"]["readout"]["flags"]["stress_prefix_masks"] == "off"

        gold_dir = _gold(tmp_path / "gold_off")
        a_off = s2.apply_dry_defaults(
            _args(
                tmp_path / "off",
                gold_dir,
                dry=True,
                steps=1,
                g=2,
                stress_prefix=0,
                mask_truncated=True,
            )
        )
        assert s2.execute(a_off, build=s2.dry_bundle) == 0
        receipt_off = json.loads((a_off.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert "stress_prefix_masks" not in receipt_off["readout"].get("flags", {})
        assert "stress_prefix_masks" not in s2.active_flags(a_off)

        stress_help = next(
            act.help for act in s2.build_parser()._actions if "--stress-prefix" in act.option_strings
        )
        assert "both masks off" in stress_help and "full-width training forward" in stress_help


# ---------------------------------------------------------------------------
# 2026-11-04 Rollout-only pre-pass (--rollout-only): T1..T6
# ---------------------------------------------------------------------------


def _prepass_gold(tmp_path: Path) -> Path:
    rows_by_file = {
        "grounded.jsonl": [
            _row("t_z", split="tune", label="supported", claim="Claim tz", text="alpha beta gamma delta"),
            _row("h_a", split="heldout", label="supported", claim="Heldout ha", text="alpha beta gamma"),
            _row("t_b", split="tune", label="unsupported", claim="Claim tb", text="alpha beta gamma delta"),
        ],
        "prs/grounded-prs.jsonl": [
            _row("t_m", split="tune", label="cannot_tell", claim="Claim tm", text="alpha beta gamma delta"),
            _row("h_b", split="heldout", label="unsupported", claim="Heldout hb", text="alpha beta gamma"),
            _row("t_a", split="tune", label="supported", claim="Claim ta", text="alpha beta gamma delta"),
        ],
        "diffs/reasoning-diffs.jsonl": [
            _row("t_k", split="tune", label="unsupported", claim="Claim tk", text="alpha beta gamma delta"),
            _row("h_c", split="heldout", label="cannot_tell", claim="Heldout hc", text="alpha beta gamma"),
            _row("t_d", split="tune", label="supported", claim="Claim td", text="alpha beta gamma delta"),
        ],
    }
    return _gold(tmp_path, rows_by_file)


class TestRolloutOnlyPrepass:
    @needs_torch
    def test_t1_no_training(self, tmp_path):
        from contextlib import contextmanager

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        a = _args(
            tmp_path,
            gold_dir,
            rollout_only=True,
            steps=3,
            prompts=2,
            g=2,
            max_new=16,
        )
        tok = s2.DryTokenizer()
        calls = {
            "optimiser_step": 0,
            "optimiser_zero_grad": 0,
            "ref_forward": 0,
            "train_forward": 0,
            "backward": 0,
            "rollout": 0,
        }

        class _TrackedTensor(torch.Tensor):
            def backward(self, *args, **kwargs):
                calls["backward"] += 1
                return super().backward(*args, **kwargs)

        class _FakeModel:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def parameters(self):
                return [self._p]

            def __call__(self, input_ids, attention_mask=None, **kwargs):
                calls["train_forward"] += 1
                logits = (
                    torch.zeros(
                        (input_ids.shape[0], input_ids.shape[1], tok.vocab_size),
                        dtype=torch.float32,
                    )
                    + self._p * 0.0
                ).as_subclass(_TrackedTensor)
                return Namespace(logits=logits)

        @contextmanager
        def tracked_ref():
            calls["ref_forward"] += 1
            yield

        class _TrackedOptim:
            def zero_grad(self, set_to_none: bool = True) -> None:
                calls["optimiser_zero_grad"] += 1

            def step(self) -> None:
                calls["optimiser_step"] += 1

        def tracked_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            calls["rollout"] += 1
            p_ids = tok.encode(prompt_text)
            rep = tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]
            return p_ids, [rep for _ in range(g_cnt)], [False] * g_cnt

        def build_fake(_args_ns):
            return {
                "model": _FakeModel(),
                "tok": tok,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": _TrackedOptim(),
                "ref_adapter_off": tracked_ref,
                "rollout": tracked_rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        rc = s2.execute(a, build=build_fake, rig=_FakeRig(a))
        assert calls["optimiser_step"] == 0
        assert calls["ref_forward"] == 0
        assert calls["train_forward"] == 0
        assert calls["backward"] == 0
        assert calls["rollout"] == 6
        assert rc == 0

    @needs_torch
    def test_t2_selection_and_counts(self, tmp_path):
        from contextlib import nullcontext

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        a = _args(
            tmp_path,
            gold_dir,
            rollout_only=True,
            steps=1,
            prompts=2,
            g=4,
            max_new=32,
        )
        tok = s2.DryTokenizer()
        seen_prompts: list[str] = []

        valid_q = "alpha beta gamma"
        bad_q = "missing quote text"
        scripted_by_idx = [
            (
                [_reply("supported", valid_q)] * 4,
                [False, False, False, False],
            ),
            (
                [
                    _reply("unsupported", valid_q),
                    _reply("unsupported", valid_q),
                    _reply("supported", valid_q),
                    "not json at all",
                ],
                [False, False, False, False],
            ),
            (
                [
                    _reply("supported", valid_q),
                    _reply("cannot_tell", bad_q),
                    _reply("cannot_tell", bad_q),
                    _reply("supported", valid_q),
                ],
                [False, False, False, True],
            ),
            (
                [_reply("supported", valid_q)] * 4,
                [False, False, False, False],
            ),
            (
                [
                    _reply("cannot_tell", bad_q),
                    _reply("cannot_tell", bad_q),
                    _reply("unsupported", valid_q),
                    _reply("supported", valid_q),
                ],
                [False, False, False, False],
            ),
            (
                [
                    _reply("supported", valid_q),
                    _reply("supported", bad_q),
                    _reply("unsupported", valid_q),
                    "broken { json",
                ],
                [False, False, False, False],
            ),
        ]

        def scripted_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            idx = len(seen_prompts)
            seen_prompts.append(prompt_text)
            replies, truncs = scripted_by_idx[idx]
            p_ids = tok.encode(prompt_text)
            comps = [tok.encode(r) + ([] if tr else [tok.eos_token_id]) for r, tr in zip(replies, truncs)]
            return p_ids, comps, truncs

        def build_fake(_args_ns):
            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": scripted_rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        rc = s2.execute(a, build=build_fake, rig=_FakeRig(a))
        assert rc == 0
        prepass_path = a.out / "s2-hf-prepass.jsonl"
        assert prepass_path.exists()
        lines = [json.loads(x) for x in prepass_path.read_text(encoding="utf-8").splitlines()]
        assert [r["id"] for r in lines] == ["t_a", "t_b", "t_d", "t_k", "t_m", "t_z"]
        assert all(not r["id"].startswith("h_") for r in lines)

        expected_truth = [
            ("t_a", "prs/grounded-prs.jsonl", "supported", 4, 4, 0, 0, 0, 0, 4, True),
            ("t_b", "grounded.jsonl", "unsupported", 2, 1, 2, 0, 1, 0, 3, False),
            ("t_d", "diffs/reasoning-diffs.jsonl", "supported", 1, 1, 0, 2, 0, 1, 1, False),
            ("t_k", "diffs/reasoning-diffs.jsonl", "unsupported", 0, 4, 0, 0, 0, 0, 4, True),
            ("t_m", "prs/grounded-prs.jsonl", "cannot_tell", 3, 1, 1, 2, 0, 0, 2, False),
            ("t_z", "grounded.jsonl", "supported", 1, 2, 1, 0, 1, 0, 2, False),
        ]
        for rec, (
            cid,
            gfile,
            gverd,
            k_corr,
            sup,
            unsup,
            ct,
            malf,
            trunc_c,
            q_pass,
            zv,
        ) in zip(lines, expected_truth, strict=True):
            assert rec["id"] == cid
            assert rec["gold_file"] == gfile
            assert rec["gold_verdict"] == gverd
            assert rec["g"] == 4
            assert rec["k_correct"] == k_corr
            assert rec["supported"] == sup
            assert rec["unsupported"] == unsup
            assert rec["cannot_tell"] == ct
            assert rec["malformed"] == malf
            assert rec["truncations"] == trunc_c
            assert rec["quote_pass"] == q_pass
            assert rec["zero_variance"] is zv

        summary = json.loads((a.out / "s2-hf-prepass-summary.json").read_text(encoding="utf-8"))
        assert summary["item_count"] == 6
        assert summary["k_correct_histogram"] == {"0": 1, "1": 2, "2": 1, "3": 1, "4": 1}
        assert summary["all_correct_share"] == pytest.approx(1 / 6)
        assert summary["zero_variance_share"] == pytest.approx(2 / 6)
        assert summary["total_truncations"] == 1
        assert summary["p_cannot_tell_given_gold_supported"] == pytest.approx(2 / 12)
        assert summary["by_gold_file"]["grounded.jsonl"] == {"0": 0, "1": 1, "2": 1, "3": 0, "4": 0}
        assert summary["by_gold_file"]["prs/grounded-prs.jsonl"] == {"0": 0, "1": 0, "2": 0, "3": 1, "4": 1}
        assert summary["by_gold_file"]["diffs/reasoning-diffs.jsonl"] == {
            "0": 1,
            "1": 1,
            "2": 0,
            "3": 0,
            "4": 0,
        }
        assert summary["by_gold_verdict"]["supported"] == {"0": 0, "1": 2, "2": 0, "3": 0, "4": 1}
        assert summary["by_gold_verdict"]["unsupported"] == {"0": 1, "1": 0, "2": 1, "3": 0, "4": 0}
        assert summary["by_gold_verdict"]["cannot_tell"] == {"0": 0, "1": 0, "2": 0, "3": 1, "4": 0}

    @needs_torch
    def test_t3_no_text(self, tmp_path, capsys):
        from contextlib import nullcontext

        import torch

        marker = "SECRET_PREPASS_MARKER_8472"
        rows_by_file = {
            "grounded.jsonl": [
                _row(
                    "g01",
                    split="tune",
                    label="supported",
                    claim=f"Claim with {marker}",
                    text=f"Evidence passage with {marker} inside it",
                )
            ],
            "prs/grounded-prs.jsonl": [
                _row(
                    "p01",
                    split="tune",
                    label="unsupported",
                    claim=f"PR claim {marker}",
                    text=f"PR evidence {marker}",
                )
            ],
            "diffs/reasoning-diffs.jsonl": [
                _row(
                    "r01",
                    split="tune",
                    label="cannot_tell",
                    claim=f"Diff claim {marker}",
                    text=f"Diff evidence {marker}",
                )
            ],
        }
        gold_dir = _gold(tmp_path / "gold", rows_by_file)
        a = _args(tmp_path, gold_dir, rollout_only=True, g=2, max_new=32)
        tok = s2.DryTokenizer()
        fake_cuda = _FakeCuda()

        def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            p_ids = tok.encode(prompt_text)
            reply_str = _reply(
                "supported",
                quote=f"Evidence passage with {marker} inside it",
                think=f"thinking about {marker}",
            )
            comp = tok.encode(reply_str) + [tok.eos_token_id]
            return p_ids, [comp for _ in range(g_cnt)], [False] * g_cnt

        def build_fake(_args_ns):
            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        assert s2.execute(a, build=build_fake, rig=_FakeRig(a)) == 0
        assert (a.out / "s2-hf-prepass.jsonl").exists()
        assert (a.out / "s2-hf-prepass-header.json").exists()
        assert (a.out / "s2-hf-prepass-summary.json").exists()
        assert (a.out / "s2-hf-phases.jsonl").exists()

        captured = capsys.readouterr()
        assert marker not in captured.out
        assert marker not in captured.err
        assert (a.out / "s2-hf-save-all.jsonl.gz").is_file()
        for path in a.out.rglob("*"):
            if path.is_file() and not path.name.endswith(".jsonl.gz"):
                assert marker not in path.read_text(encoding="utf-8"), f"marker leaked into {path.name}"

    @needs_torch
    def test_t4_resume(self, tmp_path):
        from contextlib import nullcontext

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()

        def make_builder(stop_after: int | None = None, seen_prompts: list[str] | None = None):
            calls = {"n": 0}

            def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
                calls["n"] += 1
                if stop_after is not None and calls["n"] > stop_after:
                    raise TimeoutError("simulated stop after k items")
                if seen_prompts is not None:
                    seen_prompts.append(prompt_text)
                p_ids = tok.encode(prompt_text)
                comps = []
                truncs = []
                for _ in range(g_cnt):
                    draw = int(torch.randint(0, 4, (1,)).item())
                    if draw == 0:
                        rep = _reply("supported", "alpha beta gamma")
                        tr = False
                    elif draw == 1:
                        rep = _reply("unsupported", "alpha beta gamma")
                        tr = False
                    elif draw == 2:
                        rep = _reply("cannot_tell", "absent quote")
                        tr = False
                    else:
                        rep = "malformed reply"
                        tr = True
                    comps.append(tok.encode(rep) + ([] if tr else [tok.eos_token_id]))
                    truncs.append(tr)
                return p_ids, comps, truncs

            def build_fake(_args_ns):
                return {
                    "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                    "tok": tok,
                    "mode": lambda training: None,
                    "state": lambda: {"grad_ckpt": False},
                    "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                    "ref_adapter_off": nullcontext,
                    "rollout": rollout,
                    "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                        text, row["label"], s2.passages_of(row), tt, truncated=truncated
                    ),
                    "sync": lambda: None,
                    "peak_alloc_mib": lambda: 1000,
                    "revision": "rev123",
                    "llama": None,
                    "dtype_census": None,
                }

            return build_fake

        # 1. Full uninterrupted run
        a_full = _args(tmp_path / "full", gold_dir, rollout_only=True, g=4, max_new=32)
        assert s2.execute(a_full, build=make_builder(), rig=_FakeRig(a_full)) == 0
        assert (a_full.out / "s2-hf-prepass.jsonl").exists()
        full_jsonl = (a_full.out / "s2-hf-prepass.jsonl").read_text(encoding="utf-8").splitlines()
        full_summary = json.loads((a_full.out / "s2-hf-prepass-summary.json").read_text(encoding="utf-8"))
        assert len(full_jsonl) == 6

        # 2. Interrupted run after k = 3 items + half-written trailing line
        a_res = _args(tmp_path / "resumed", gold_dir, rollout_only=True, g=4, max_new=32)
        assert s2.execute(a_res, build=make_builder(stop_after=3), rig=_FakeRig(a_res)) == 1
        res_jsonl_path = a_res.out / "s2-hf-prepass.jsonl"
        with res_jsonl_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write('{"id": "t_k", "gold_file": "diffs/reasoning-diffs.jsonl", "k_correct": ')

        # 3. Rerun to resume
        resumed_prompts: list[str] = []
        assert (
            s2.execute(
                a_res,
                build=make_builder(stop_after=None, seen_prompts=resumed_prompts),
                rig=_FakeRig(a_res),
            )
            == 0
        )
        assert len(resumed_prompts) == 3  # first 3 done ids skipped, only remaining 3 rolled out
        res_jsonl = res_jsonl_path.read_text(encoding="utf-8").splitlines()
        res_summary = json.loads((a_res.out / "s2-hf-prepass-summary.json").read_text(encoding="utf-8"))
        assert [json.loads(x) for x in res_jsonl] == [json.loads(x) for x in full_jsonl]
        assert res_summary == full_summary

    @needs_torch
    def test_t5_refuse_on_mismatch(self, tmp_path):
        from contextlib import nullcontext

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()

        def build_fake(_args_ns):
            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": lambda text, g_cnt, max_n, temp: (
                    tok.encode(text),
                    [tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]] * g_cnt,
                    [False] * g_cnt,
                ),
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        a1 = _args(tmp_path / "run", gold_dir, rollout_only=True, g=4, max_new=32)
        assert s2.execute(a1, build=build_fake, rig=_FakeRig(a1)) == 0
        header_path = a1.out / "s2-hf-prepass-header.json"
        jsonl_path = a1.out / "s2-hf-prepass.jsonl"
        assert header_path.exists() and jsonl_path.exists()
        before_bytes = jsonl_path.read_bytes()

        # Change G (4 -> 2) and rerun on the same out dir
        a_mismatch = _args(tmp_path / "run", gold_dir, rollout_only=True, g=2, max_new=32)
        with pytest.raises(SystemExit, match="mismatch"):
            s2.execute(a_mismatch, build=build_fake, rig=_FakeRig(a_mismatch))
        assert jsonl_path.read_bytes() == before_bytes

        # Also check corrupt header, non-dict header, orphan JSONL, and corrupt mid-file JSONL line
        exp_hdr = json.loads(header_path.read_text(encoding="utf-8"))
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        hp = bad_dir / "h.json"
        jp = bad_dir / "p.jsonl"
        jp.write_text("{}", encoding="utf-8")
        with pytest.raises(SystemExit, match="without"):
            s2._prepare_prepass_resume(hp, jp, exp_hdr)
        jp.unlink()
        hp.write_text("{not json", encoding="utf-8")
        with pytest.raises(SystemExit, match="not valid JSON"):
            s2._prepare_prepass_resume(hp, jp, exp_hdr)
        hp.write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(SystemExit, match="not a JSON object"):
            s2._prepare_prepass_resume(hp, jp, exp_hdr)
        hp.write_text(json.dumps(exp_hdr), encoding="utf-8")
        assert s2._prepare_prepass_resume(hp, jp, exp_hdr) == []
        jp.write_text("{corrupt\n{}\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="corrupt line 1"):
            s2._prepare_prepass_resume(hp, jp, exp_hdr)
        jp.write_text('{"id": "only_id"}\n{"id": "second"}\n', encoding="utf-8")
        with pytest.raises(SystemExit, match="incomplete record on line 1"):
            s2._prepare_prepass_resume(hp, jp, exp_hdr)
        # Trailing incomplete record on last line is truncated cleanly
        first_valid_line = before_bytes.splitlines(keepends=True)[0].decode("utf-8")
        jp.write_text(first_valid_line + '{"id": "partial_record"}\n', encoding="utf-8")
        recs = s2._prepare_prepass_resume(hp, jp, exp_hdr)
        assert len(recs) == 1
        assert jp.read_text(encoding="utf-8") == first_valid_line

    @needs_torch
    def test_t6_readout_and_flags(self, tmp_path, capsys):
        import torch

        gold_dir = _gold(tmp_path / "gold")
        ap = s2.build_parser()
        ro_actions = [act for act in ap._actions if "--rollout-only" in act.option_strings]
        assert ro_actions, "missing --rollout-only flag"
        assert "it trains nothing and writes counts per item" in ro_actions[0].help

        base_argv = ["--gold", str(gold_dir), "--rnd-commit", "e48cc90c", "--out", str(tmp_path / "out")]
        parsed = ap.parse_args([*base_argv, "--rollout-only", "--dry"])
        assert parsed.rollout_only is True
        assert s2.active_flags(parsed)["rollout_only"] is True

        # --rollout-only with --stress-prefix > 0 is refused
        bad = ap.parse_args([*base_argv, "--rollout-only", "--stress-prefix", "1"])
        with pytest.raises(SystemExit, match="--rollout-only"):
            s2.execute(bad, build=s2.dry_bundle)

        # --dry --rollout-only runs on CPU with the tiny model over DRY_PREPASS_ITEMS (4) items and exercises resume
        rc = s2.main([*base_argv, "--dry", "--rollout-only", "--prompts", "11", "--steps", "7"])
        assert rc == 0
        out_lines = capsys.readouterr().out.strip().splitlines()
        assert out_lines[-1] == "S2-PREPASS"
        receipt = json.loads((Path(base_argv[5]) / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt["verdict"] == "S2-PREPASS"
        assert receipt["readout"]["flags"]["rollout_only"] is True
        assert receipt["readout"]["ignored_flags"] == ["prompts", "steps"]
        prepass_lines = [
            json.loads(x)
            for x in (Path(base_argv[5]) / "s2-hf-prepass.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(prepass_lines) == s2.DRY_PREPASS_ITEMS == 4

        # Check verdict_line failure branches and 2-tuple rollout + OOM handling in run_prepass
        assert (
            s2.verdict_line(
                {"items_requested": 4, "items_completed": 2, "oom": True},
                stopped=None,
                cut=False,
                dry=False,
                rollout_only=True,
            )
            == "S2-FAIL"
        )
        assert (
            s2.verdict_line(
                {"items_requested": 4, "items_completed": 2, "oom": True},
                stopped=None,
                cut=False,
                dry=True,
                rollout_only=True,
            )
            == "S2-STOPPED"
        )
        tok = s2.DryTokenizer()
        rows_g, hashes_g = s2.read_gold(gold_dir)
        prompts_2 = s2.select_prompts(rows_g, 2)
        a_2t = _args(
            tmp_path / "twotuple",
            gold_dir,
            rollout_only=True,
            empty_cache_phases=True,
            g=2,
            max_new=16,
        )
        before_called: list[int] = []
        bundle_2t = {
            "tok": tok,
            "rollout": lambda text, g_cnt, max_n, temp: (
                tok.encode(text),
                [tok.encode(_reply("supported", "x")) + [tok.eos_token_id]] * g_cnt,
            ),
            "reward_fn": s2.dry_reward,
            "sync": lambda: None,
            "peak_alloc_mib": lambda: 100,
            "before_step": before_called.append,
            "mode": None,
            "state": None,
            "revision": "rev123",
        }
        s_rows, c_recs, st_rec, is_oom = s2.run_prepass(
            a=a_2t,
            prompts=prompts_2,
            gold_hashes=hashes_g,
            bundle=bundle_2t,
            phase_logger=None,
            eval_records=None,
        )
        assert len(s_rows) == 2 and len(c_recs) == 2 and st_rec is None and not is_oom
        assert before_called == [0]

        oom_cls = getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None)
        if oom_cls is not None:
            a_oom = _args(tmp_path / "oom_prepass", gold_dir, rollout_only=True, g=2, max_new=16)
            bundle_oom = {
                **bundle_2t,
                "rollout": lambda text, g_cnt, max_n, temp: (_ for _ in ()).throw(oom_cls("fake oom")),
            }
            _, _, st_oom, flag_oom = s2.run_prepass(
                a=a_oom,
                prompts=prompts_2,
                gold_hashes=hashes_g,
                bundle=bundle_oom,
            )
            assert flag_oom is True and st_oom == {"step": 0, "error": "out of memory"}

    @needs_torch
    def test_t7_llamacpp_resume_refusal(self, tmp_path):
        from contextlib import nullcontext

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()

        def build_fake(_args_ns):
            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": lambda text, g_cnt, max_n, temp: (
                    tok.encode(text),
                    [tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]] * g_cnt,
                    [False] * g_cnt,
                ),
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": {"path": "llama-server", "sha256": "00" * 32, "pid": 1},
                "dtype_census": None,
            }

        a = _args(tmp_path / "llama_run", gold_dir, path="llamacpp", rollout_only=True, g=2, max_new=16)
        assert s2.execute(a, build=build_fake, rig=_FakeRig(a)) == 0
        header_path = a.out / "s2-llamacpp-prepass-header.json"
        jsonl_path = a.out / "s2-llamacpp-prepass.jsonl"
        assert header_path.exists()
        assert jsonl_path.exists()
        before_header_bytes = header_path.read_bytes()
        before_jsonl_bytes = jsonl_path.read_bytes()

        with pytest.raises(
            SystemExit,
            match="refusing to resume prepass: --path llamacpp samples are not seeded per item, so a resumed run is not reproducible",
        ):
            s2.execute(a, build=build_fake, rig=_FakeRig(a))
        assert header_path.read_bytes() == before_header_bytes
        assert jsonl_path.read_bytes() == before_jsonl_bytes

    @needs_torch
    def test_t8_fresh_llamacpp_header_and_summary_reproducible_false(self, tmp_path):
        from contextlib import nullcontext

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()

        def build_fake(_args_ns):
            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": lambda text, g_cnt, max_n, temp: (
                    tok.encode(text),
                    [tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]] * g_cnt,
                    [False] * g_cnt,
                ),
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": {"path": "llama-server", "sha256": "00" * 32, "pid": 1},
                "dtype_census": None,
            }

        a_llama = _args(
            tmp_path / "llama_fresh", gold_dir, path="llamacpp", rollout_only=True, g=2, max_new=16
        )
        assert s2.execute(a_llama, build=build_fake, rig=_FakeRig(a_llama)) == 0
        header = json.loads((a_llama.out / "s2-llamacpp-prepass-header.json").read_text(encoding="utf-8"))
        summary = json.loads((a_llama.out / "s2-llamacpp-prepass-summary.json").read_text(encoding="utf-8"))
        receipt = json.loads((a_llama.out / "s2-llamacpp.json").read_text(encoding="utf-8"))
        assert header["reproducible"] is False
        assert summary["reproducible"] is False
        assert receipt["readout"]["prepass_summary"]["reproducible"] is False

        a_hf = _args(tmp_path / "hf_fresh", gold_dir, path="hf", rollout_only=True, g=2, max_new=16)
        assert s2.execute(a_hf, build=build_fake, rig=_FakeRig(a_hf)) == 0
        hf_header = json.loads((a_hf.out / "s2-hf-prepass-header.json").read_text(encoding="utf-8"))
        hf_summary = json.loads((a_hf.out / "s2-hf-prepass-summary.json").read_text(encoding="utf-8"))
        hf_receipt = json.loads((a_hf.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert hf_header["reproducible"] is True
        assert hf_summary["reproducible"] is True
        assert hf_receipt["readout"]["prepass_summary"]["reproducible"] is True

    @needs_torch
    def test_t9_hf_header_without_reproducible_key_still_resumes(self, tmp_path):
        from contextlib import nullcontext

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        calls = {"n": 0}

        def make_builder(stop_after: int | None):
            def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
                calls["n"] += 1
                if stop_after is not None and calls["n"] > stop_after:
                    raise TimeoutError("stop early")
                return (
                    tok.encode(prompt_text),
                    [tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]] * g_cnt,
                    [False] * g_cnt,
                )

            def build_fake(_args_ns):
                return {
                    "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                    "tok": tok,
                    "mode": lambda training: None,
                    "state": lambda: {"grad_ckpt": False},
                    "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                    "ref_adapter_off": nullcontext,
                    "rollout": rollout,
                    "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                        text, row["label"], s2.passages_of(row), tt, truncated=truncated
                    ),
                    "sync": lambda: None,
                    "peak_alloc_mib": lambda: 1000,
                    "revision": "rev123",
                    "llama": None,
                    "dtype_census": None,
                }

            return build_fake

        a = _args(tmp_path / "hf_compat", gold_dir, path="hf", rollout_only=True, g=2, max_new=16)
        assert s2.execute(a, build=make_builder(stop_after=2), rig=_FakeRig(a)) == 1
        header_path = a.out / "s2-hf-prepass-header.json"
        orig_hdr = json.loads(header_path.read_text(encoding="utf-8"))
        assert orig_hdr["reproducible"] is True

        # Negative case 1: deleting another header key (quote_reward) refuses with header mismatch
        hdr_missing_qr = dict(orig_hdr)
        del hdr_missing_qr["quote_reward"]
        header_path.write_text(json.dumps(hdr_missing_qr, indent=1) + "\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="refusing to resume prepass: header mismatch on quote_reward"):
            s2.execute(a, build=make_builder(stop_after=None), rig=_FakeRig(a))

        # Negative case 2: setting reproducible to False in an hf header refuses
        hdr_repro_false = dict(orig_hdr)
        hdr_repro_false["reproducible"] = False
        header_path.write_text(json.dumps(hdr_repro_false, indent=1) + "\n", encoding="utf-8")
        with pytest.raises(SystemExit, match="refusing to resume prepass: header mismatch on reproducible"):
            s2.execute(a, build=make_builder(stop_after=None), rig=_FakeRig(a))

        # Positive legacy case: deleting reproducible key on hf still resumes
        hdr = dict(orig_hdr)
        del hdr["reproducible"]
        header_path.write_text(json.dumps(hdr, indent=1) + "\n", encoding="utf-8")

        calls["n"] = 0
        assert s2.execute(a, build=make_builder(stop_after=None), rig=_FakeRig(a)) == 0
        assert calls["n"] == 4
        lines = (a.out / "s2-hf-prepass.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 6

    @needs_torch
    def test_t10_quote_pass_role_in_summary_readout_and_help(self, tmp_path):
        from contextlib import nullcontext

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()

        def build_fake(_args_ns):
            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": lambda text, g_cnt, max_n, temp: (
                    tok.encode(text),
                    [tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]] * g_cnt,
                    [False] * g_cnt,
                ),
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text,
                    row["label"],
                    s2.passages_of(row),
                    tt,
                    truncated=truncated,
                    quote_reward="v2",
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        a = _args(
            tmp_path / "qrole_v2",
            gold_dir,
            path="hf",
            rollout_only=True,
            quote_reward="v2",
            g=2,
            max_new=16,
        )
        assert s2.execute(a, build=build_fake, rig=_FakeRig(a)) == 0
        expected_v2 = (
            "diagnostic: every evidence quote found in the passages; "
            "not the registered reward (quote rule v2)"
        )
        summary = json.loads((a.out / "s2-hf-prepass-summary.json").read_text(encoding="utf-8"))
        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert summary["quote_pass_role"] == expected_v2
        assert summary["quote_pass"] == 12
        assert receipt["readout"]["quote_pass_role"] == expected_v2
        assert receipt["readout"]["prepass_summary"]["quote_pass_role"] == expected_v2

        ro_help = next(
            act.help for act in s2.build_parser()._actions if "--rollout-only" in act.option_strings
        )
        assert (
            "diagnostic: every evidence quote found in the passages; "
            "not the registered reward (quote rule <v>)"
        ) in ro_help
        ro_no_summary = s2.readout(
            [],
            peak_mib=100,
            steps_requested=2,
            stopped=None,
            oom=False,
            cut=False,
            dry=False,
            rollout_only=True,
            quote_reward="v1",
        )
        assert ro_no_summary["quote_pass_role"] == (
            "diagnostic: every evidence quote found in the passages; "
            "not the registered reward (quote rule v1)"
        )


# ---------------------------------------------------------------------------
# 2026-11-04 Stress pool (--stress-pool {selected,tune}): T1..T5
# ---------------------------------------------------------------------------


def _stress_pool_gold(tmp_path: Path, *, tie_outside: bool = False, marker: str = "") -> Path:
    long_outside = f"outside_selected_longest_prompt {marker} " * 25
    rows_by_file = {
        "grounded.jsonl": [
            _row(f"g{i:02d}", text="short" if i < 19 else "selected_longest_prompt " * 10) for i in range(20)
        ],
        "prs/grounded-prs.jsonl": [_row(f"p{i:02d}", text="short") for i in range(20)]
        + [
            _row("zz_heldout", split="heldout", text="heldout_even_longer " * 50),
        ],
        "diffs/reasoning-diffs.jsonl": [
            _row(
                f"r{i:02d}",
                label="unsupported",
                claim=f"Claim {marker}" if (not tie_outside and i == 15 and marker) else "Claim.",
                text=(
                    long_outside
                    if (tie_outside and i in (5, 10)) or (not tie_outside and i == 15)
                    else "short"
                ),
            )
            for i in (range(19, -1, -1) if tie_outside else range(20))
        ],
    }
    return _gold(tmp_path, rows_by_file)


def _make_stress_pool_fake_bundle(tok, seen_prefix_prompts: list[str], seen_reg_prompts: list[str]):
    from contextlib import nullcontext

    import torch

    class _FakeModel:
        device = torch.device("cpu")

        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def parameters(self):
            return [self._param]

        def generate(self, input_ids, **kwargs):
            row0_ids = input_ids[0].tolist()
            seen_prefix_prompts.append(tok.decode(row0_ids))
            max_n = int(kwargs["max_new_tokens"])
            cap_fill = torch.full((input_ids.shape[0], max_n), 10, dtype=torch.long)
            return torch.cat([input_ids, cap_fill], dim=1)

        def __call__(self, input_ids, attention_mask=None, logits_to_keep=None, **kwargs):
            logits = (
                torch.zeros(
                    (input_ids.shape[0], input_ids.shape[1], tok.vocab_size),
                    dtype=torch.float32,
                )
                + self._param * 0.0
            )
            return Namespace(logits=logits)

    fake_model = _FakeModel()

    def fake_reg_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
        seen_reg_prompts.append(prompt_text)
        p_ids = tok.encode(prompt_text)
        comps = [[20, 21, 22, tok.eos_token_id] for _ in range(g_cnt)]
        if g_cnt >= 2:
            comps[0] = [30, 31, 32, tok.eos_token_id]
        truncs = [False] * g_cnt
        return p_ids, comps, truncs

    def fake_reward(text: str, row: dict, tt: int, truncated: bool = False) -> float:
        if truncated:
            return -1.0
        if text == tok.decode([30, 31, 32]):
            return 1.0
        return -0.3

    def build_fake(_args_ns):
        return {
            "model": fake_model,
            "tok": tok,
            "mode": lambda training: None,
            "state": lambda: {"grad_ckpt": True},
            "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
            "ref_adapter_off": nullcontext,
            "rollout": fake_reg_rollout,
            "reward_fn": fake_reward,
            "sync": lambda: None,
            "peak_alloc_mib": lambda: 1000,
            "revision": None,
            "llama": None,
            "dtype_census": None,
        }

    return build_fake


class TestStressPool:
    @needs_torch
    def test_t1_tune_picks_longer_prompt_outside_selected_and_prompt_lengths_tune(self, tmp_path):
        gold_dir = _stress_pool_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        rows, _ = s2.read_gold(gold_dir)
        selected_32 = s2.select_prompts(rows, 32)
        assert all(not r["id"].startswith("r") for r in selected_32)

        # Full fake run with --stress-prefix 1 --stress-pool tune
        seen_prefix: list[str] = []
        seen_reg: list[str] = []
        a_run = s2.apply_dry_defaults(
            _args(
                tmp_path / "run_tune",
                gold_dir,
                dry=True,
                steps=1,
                g=2,
                max_new=8,
                stress_prefix=1,
                stress_pool="tune",
            )
        )
        assert s2.execute(a_run, build=_make_stress_pool_fake_bundle(tok, seen_prefix, seen_reg)) == 0
        assert len(seen_prefix) == 2
        assert all("[src@r15]" in p for p in seen_prefix)
        # Registered steps still use select_prompts (g00, g01)
        assert len(seen_reg) == 2
        assert "[src@g00]" in seen_reg[0] and "[src@g01]" in seen_reg[1]

        # --prompt-lengths --stress-pool tune stays CPU only and covers all 60 tune rows
        a_pl = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "pl_tune"),
                "--dry",
                "--prompt-lengths",
                "--stress-pool",
                "tune",
            ]
        )
        assert (
            s2.execute(
                a_pl,
                build=lambda _: (_ for _ in ()).throw(AssertionError("model build must not run")),
                rig=None,
            )
            == 0
        )
        pl_doc = json.loads((a_pl.out / "s2-hf-prompt-lengths.json").read_text(encoding="utf-8"))
        assert pl_doc["count"] == 60
        assert pl_doc["stress_pool"] == "tune"
        assert pl_doc["stress_prompt_id"] == "r15"
        r15_row = next(r for r in rows if r["id"] == "r15")
        r15_tokens = len(tok.encode(s2.build_prompt(r15_row)))
        assert pl_doc["stress_prompt_tokens"] == r15_tokens == pl_doc["max"]

    @needs_torch
    def test_t2_ties_break_by_larger_id(self, tmp_path):
        rows_by_file = {
            "grounded.jsonl": [
                _row(
                    f"g{i:02d}",
                    text="tied_longest_selected " * 15 if i in (2, 9) else "short",
                )
                for i in range(19, -1, -1)
            ],
            "prs/grounded-prs.jsonl": [_row(f"p{i:02d}", text="short") for i in range(20)],
            "diffs/reasoning-diffs.jsonl": [
                _row(
                    f"r{i:02d}",
                    label="unsupported",
                    text="tied_longest_tune " * 25 if i in (5, 10) else "short",
                )
                for i in range(19, -1, -1)
            ],
        }
        gold_dir = _gold(tmp_path / "gold", rows_by_file)
        tok = s2.DryTokenizer()

        # Check selected pool tie break (g02 vs g09 -> g09 wins by larger id)
        seen_sel_prefix: list[str] = []
        seen_sel_reg: list[str] = []
        a_sel = s2.apply_dry_defaults(
            _args(
                tmp_path / "tie_sel",
                gold_dir,
                dry=True,
                steps=1,
                g=2,
                max_new=8,
                stress_prefix=1,
                stress_pool="selected",
            )
        )
        assert s2.execute(a_sel, build=_make_stress_pool_fake_bundle(tok, seen_sel_prefix, seen_sel_reg)) == 0
        assert all("[src@g09]" in p for p in seen_sel_prefix)

        # Check tune pool tie break (r05 vs r10 -> r10 wins by larger id)
        seen_tune_prefix: list[str] = []
        seen_tune_reg: list[str] = []
        a_tune = s2.apply_dry_defaults(
            s2.build_parser().parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / "tie_tune"),
                    "--dry",
                    "--steps",
                    "1",
                    "--g",
                    "2",
                    "--max-new",
                    "8",
                    "--stress-prefix",
                    "1",
                    "--stress-pool",
                    "tune",
                ]
            )
        )
        assert (
            s2.execute(a_tune, build=_make_stress_pool_fake_bundle(tok, seen_tune_prefix, seen_tune_reg)) == 0
        )
        assert all("[src@r10]" in p for p in seen_tune_prefix)
        receipt = json.loads((a_tune.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt["readout"]["flags"]["stress_prompt_id"] == "r10"

    def test_t2b_select_stress_prompt_matches_base_max_expression(self):
        tok = s2.DryTokenizer()
        pool = [
            _row("id_03", text="tied_maximum_text " * 12),
            _row("id_01", text="short"),
            _row("id_09", text="tied_maximum_text " * 12),
            _row("id_05", text="tied_maximum_text " * 12),
        ]
        expected_row = max(pool, key=lambda r: (len(tok.encode(s2.build_prompt(r))), r["id"]))
        expected_tok = len(tok.encode(s2.build_prompt(expected_row)))
        chosen_row, chosen_tok = s2.select_stress_prompt(pool, tok)
        assert chosen_row == expected_row
        assert chosen_row["id"] == "id_09"
        assert chosen_tok == expected_tok

    @needs_torch
    def test_t3_default_path_unchanged(self, tmp_path):
        gold_dir = _stress_pool_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        rows, _ = s2.read_gold(gold_dir)
        g19_row = next(r for r in rows if r["id"] == "g19")
        g19_tokens = len(tok.encode(s2.build_prompt(g19_row)))
        expected_reg_prompts = [
            s2.build_prompt(next(r for r in rows if r["id"] == pid)) for pid in ("g00", "g01")
        ]

        # Independent baseline run at --stress-prefix 0
        pref_no_prefix: list[str] = []
        reg_no_prefix: list[str] = []
        a_no_prefix = s2.apply_dry_defaults(
            s2.build_parser().parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / "no_prefix_run"),
                    "--dry",
                    "--steps",
                    "1",
                    "--g",
                    "2",
                    "--max-new",
                    "8",
                    "--stress-prefix",
                    "0",
                ]
            )
        )
        assert (
            s2.execute(
                a_no_prefix,
                build=_make_stress_pool_fake_bundle(tok, pref_no_prefix, reg_no_prefix),
            )
            == 0
        )
        assert pref_no_prefix == []
        assert reg_no_prefix == expected_reg_prompts
        step_rows_no_prefix = [
            json.loads(x)
            for x in (a_no_prefix.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]

        # Run default (no --stress-pool passed -> defaults to "selected")
        pref_def: list[str] = []
        reg_def: list[str] = []
        a_def = s2.apply_dry_defaults(
            s2.build_parser().parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / "def_run"),
                    "--dry",
                    "--steps",
                    "1",
                    "--g",
                    "2",
                    "--max-new",
                    "8",
                    "--stress-prefix",
                    "1",
                ]
            )
        )
        assert s2.execute(a_def, build=_make_stress_pool_fake_bundle(tok, pref_def, reg_def)) == 0
        assert all("[src@g19]" in p for p in pref_def)
        assert reg_def == expected_reg_prompts == reg_no_prefix

        step_rows_def = [
            json.loads(x) for x in (a_def.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(step_rows_def) == len(step_rows_no_prefix) == 1
        for k in (
            "step",
            "groups",
            "active_groups",
            "zero_variance_groups",
            "tokens",
            "mean_reward",
            "loss",
            "peak_alloc_mib",
            "grad_ckpt",
            "grad_ckpt_rollout",
        ):
            assert [r[k] for r in step_rows_def] == [r[k] for r in step_rows_no_prefix]

        receipt_def = json.loads((a_def.out / "s2-hf.json").read_text(encoding="utf-8"))
        receipt_no_prefix = json.loads((a_no_prefix.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert (
            receipt_def["prompt_ids"]
            == receipt_no_prefix["prompt_ids"]
            == [r["id"] for r in s2.select_prompts(rows, 32)]
        )
        readout_def = receipt_def["readout"]

        assert readout_def["flags"]["stress_pool"] == "selected"
        assert readout_def["flags"]["stress_prompt_id"] == "g19"
        assert readout_def["flags"]["stress_prompt_tokens"] == g19_tokens
        assert readout_def["flags"]["stress_width"] == g19_tokens + 8

        # Compare readout against base readout from independent --stress-prefix 0 step rows
        aligned_no_prefix_steps = [
            {
                **r,
                "rollout_s": d["rollout_s"],
                "train_s": d["train_s"],
                "step_s": d["step_s"],
            }
            for r, d in zip(step_rows_no_prefix, step_rows_def, strict=True)
        ]
        expected_base_readout = s2.readout(
            aligned_no_prefix_steps,
            peak_mib=None,
            steps_requested=1,
            stopped=None,
            oom=False,
            cut=True,
            dry=True,
            flags={"stress_prefix": 1, "stress_prefix_masks": "off"},
            eval_summary_record=json.loads((a_no_prefix.out / "s2-hf-eval-summary.json").read_text()),
        )
        stripped_readout = {
            **readout_def,
            "flags": {
                k: v
                for k, v in readout_def["flags"].items()
                if k not in ("stress_pool", "stress_prompt_id", "stress_prompt_tokens", "stress_width")
            },
        }
        assert stripped_readout == expected_base_readout

    @needs_torch
    def test_t4_recorded_fields_present_and_text_free(self, tmp_path, capsys):
        marker = "SECRET_STRESS_" + "POOL_MARKER_991"
        gold_dir = _stress_pool_gold(tmp_path / "gold", marker=marker)
        tok = s2.DryTokenizer()
        rows, _ = s2.read_gold(gold_dir)
        r15_row = next(r for r in rows if r["id"] == "r15")
        r15_tokens = len(tok.encode(s2.build_prompt(r15_row)))

        seen_prefix: list[str] = []
        seen_reg: list[str] = []
        a = s2.apply_dry_defaults(
            _args(
                tmp_path / "text_free",
                gold_dir,
                dry=True,
                steps=1,
                g=2,
                max_new=12,
                stress_prefix=1,
                stress_pool="tune",
            )
        )
        assert s2.execute(a, build=_make_stress_pool_fake_bundle(tok, seen_prefix, seen_reg)) == 0
        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        for container in (receipt["readout"]["flags"], receipt["run"], receipt):
            assert container["stress_pool"] == "tune"
            assert container["stress_prompt_id"] == "r15"
            assert container["stress_prompt_tokens"] == r15_tokens
            assert container["stress_width"] == r15_tokens + 12

        captured = capsys.readouterr()
        assert marker not in captured.out
        assert marker not in captured.err
        for p in a.out.rglob("*"):
            if p.is_file() and not p.name.endswith(".jsonl.gz"):
                assert marker not in p.read_text(encoding="utf-8"), f"marker leaked into {p.name}"

        # Also run --prompt-lengths --stress-pool tune on the marker gold
        a_pl = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "pl_text_free"),
                "--dry",
                "--prompt-lengths",
                "--stress-pool",
                "tune",
            ]
        )
        assert (
            s2.execute(
                a_pl,
                build=lambda _: (_ for _ in ()).throw(AssertionError("model build must not run")),
                rig=None,
            )
            == 0
        )
        captured_pl = capsys.readouterr()
        assert marker not in captured_pl.out
        assert marker not in captured_pl.err
        for p in a_pl.out.rglob("*"):
            if p.is_file():
                assert marker not in p.read_text(encoding="utf-8"), f"marker leaked into {p.name}"

    def test_t5_rollout_only_with_stress_prefix_still_refuses(self, tmp_path):
        gold_dir = _stress_pool_gold(tmp_path / "gold")
        ap = s2.build_parser()
        for pool in ("selected", "tune"):
            a = ap.parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / f"refuse_{pool}"),
                    "--rollout-only",
                    "--stress-prefix",
                    "1",
                    "--stress-pool",
                    pool,
                ]
            )
            with pytest.raises(
                SystemExit,
                match="refusing to start: --rollout-only cannot be used with --stress-prefix > 0",
            ):
                s2.execute(a, build=s2.dry_bundle)
        with pytest.raises(SystemExit, match="unknown --stress-pool 'invalid'"):
            s2.stress_prompt_pool([], [], "invalid")
        empty_stats = s2.compute_prompt_lengths([], s2.DryTokenizer(), stress_pool="tune")
        assert empty_stats["count"] == 0
        assert empty_stats["stress_pool"] == "tune"
        assert empty_stats["stress_prompt_id"] is None
        assert empty_stats["stress_prompt_tokens"] == 0


# ---------------------------------------------------------------------------
# 2026-11-04 Prompt draw setting (s2_draw.draw_prompts & --draw): (a)..(f)
# ---------------------------------------------------------------------------


def _draw_gold_40(tmp_path: Path, *, marker: str = "") -> Path:
    claim_str = f"Claim {marker}." if marker else "Claim."
    text_str = f"alpha beta gamma {marker}" if marker else "alpha beta gamma"
    rows_by_file = {
        "grounded.jsonl": [_row(f"t{i:02d}", claim=claim_str, text=text_str) for i in range(15)],
        "prs/grounded-prs.jsonl": [_row(f"t{i:02d}", claim=claim_str, text=text_str) for i in range(15, 30)]
        + [_row("h00", split="heldout", claim=claim_str, text=text_str)],
        "diffs/reasoning-diffs.jsonl": [
            _row(f"t{i:02d}", label="unsupported", claim=claim_str, text=text_str) for i in range(30, 40)
        ],
    }
    return _gold(tmp_path, rows_by_file)


def _write_fake_prepass(path: Path, ids: list[str], k_fn=lambda idx: idx % 9, g: int = 8) -> str:
    import hashlib

    lines = [
        json.dumps(
            {
                "id": cid,
                "gold_file": "grounded.jsonl",
                "gold_verdict": "supported",
                "g": g,
                "k_correct": k_fn(idx),
                "supported": k_fn(idx),
                "unsupported": g - k_fn(idx),
                "cannot_tell": 0,
                "malformed": 0,
                "truncations": 0,
                "quote_pass": k_fn(idx),
                "reward_mean": 0.1,
                "reward_std": 0.5,
                "zero_variance": k_fn(idx) in (0, g),
                "prompt_tokens": 24,
                "completion_tokens_mean": 12.0, "pass_seed": 0, "descriptive": False,
            }
        )
        + "\n"
        for idx, cid in enumerate(ids)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8", newline="\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestDrawSetting:
    def test_a_first_equals_select_prompts_and_calls_it(self, tmp_path, monkeypatch):
        import s2_draw
        from s2_draw import draw_prompts

        assert s2.select_prompts is s2_draw.select_prompts

        gold_dir = _draw_gold_40(tmp_path / "gold")
        rows, _ = s2.read_gold(gold_dir)
        expected = s2.select_prompts(rows, 12)
        picked, rec = draw_prompts(rows, 12, mode="first")
        assert picked == expected
        assert [r["id"] for r in picked] == [f"t{i:02d}" for i in range(12)]
        assert rec == {
            "draw_mode": "first",
            "draw_n": 12,
            "draw_seed": None,
            "drawn_ids_sha256": s2.ids_sha256([r["id"] for r in expected]),
        }

        calls: list[tuple[int, str]] = []
        real_select = s2_draw.select_prompts

        def spy_select(r_list, n_val=s2.PROMPTS, split=s2.PROMPT_SPLIT):
            calls.append((n_val, split))
            return real_select(r_list, n_val, split=split)

        monkeypatch.setattr(s2_draw, "select_prompts", spy_select)
        picked2, _ = draw_prompts(rows, 10, mode="first")
        assert calls == [(10, "tune")]
        assert len(picked2) == 10

    def test_s2_draw_subprocess_imports_no_smoke_and_no_torch(self):
        import subprocess

        verifier_dir = Path(__file__).parent.parent / "examples" / "verifier-distill"
        code = (
            "import sys\n"
            f"sys.path.insert(0, {str(verifier_dir)!r})\n"
            "import s2_draw\n"
            "assert 's2_grpo_smoke' not in sys.modules, 's2_grpo_smoke loaded on import s2_draw'\n"
            "assert 'torch' not in sys.modules, 'torch loaded on import s2_draw'\n"
            "rows = [{'id': 't00', 'split': 'tune'}, {'id': 't01', 'split': 'tune'}]\n"
            "picked, rec = s2_draw.draw_prompts(rows, 1, mode='first')\n"
            "assert [r['id'] for r in picked] == ['t00']\n"
            "assert 's2_grpo_smoke' not in sys.modules, 's2_grpo_smoke loaded on draw_prompts(mode=first)'\n"
            "assert 'torch' not in sys.modules, 'torch loaded on draw_prompts(mode=first)'\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert proc.returncode == 0, f"subprocess failed:\nstdout={proc.stdout}\nstderr={proc.stderr}"

    def test_b_uniform_pinned_draw_and_local_rng(self, tmp_path):
        import random

        from s2_draw import draw_prompts

        gold_dir = _draw_gold_40(tmp_path / "gold")
        rows, _ = s2.read_gold(gold_dir)

        random.seed(99999)
        before_global = random.getstate()
        picked_1, rec_1 = draw_prompts(rows, 10, mode="uniform", seed=1234)
        after_global = random.getstate()
        assert before_global == after_global, "draw_prompts must never touch global random state"

        pinned_ids = ["t00", "t02", "t05", "t06", "t07", "t15", "t22", "t28", "t36", "t37"]
        assert [r["id"] for r in picked_1] == pinned_ids
        assert [r["id"] for r in picked_1] == sorted(r["id"] for r in picked_1)
        assert all(r["split"] == "tune" for r in picked_1)
        assert rec_1 == {
            "draw_mode": "uniform",
            "draw_n": 10,
            "draw_seed": 1234,
            "drawn_ids_sha256": "0f33b135252d80b75ccc9486f3e2a3e3a188fb143741b503dd05d3bfabaf96b7",
        }

        # Same seed gives the exact same ids and record on a reversed input row order
        picked_2, rec_2 = draw_prompts(list(reversed(rows)), 10, mode="uniform", seed=1234)
        assert [r["id"] for r in picked_2] == pinned_ids
        assert rec_2 == rec_1

        # Different seed gives a different list
        picked_diff, rec_diff = draw_prompts(rows, 10, mode="uniform", seed=5678)
        assert [r["id"] for r in picked_diff] != pinned_ids
        assert rec_diff["draw_seed"] == 5678
        assert rec_diff["drawn_ids_sha256"] != rec_1["drawn_ids_sha256"]

    def test_c_kband_keeps_only_in_band_and_matches_pinned_list(self, tmp_path):
        import hashlib
        import random

        from s2_draw import draw_prompts

        gold_dir = _draw_gold_40(tmp_path / "gold")
        rows, _ = s2.read_gold(gold_dir)
        tune_ids = [f"t{i:02d}" for i in range(40)]
        prepass_path = tmp_path / "prepass.jsonl"
        sha = _write_fake_prepass(prepass_path, tune_ids, k_fn=lambda idx: idx % 9, g=8)

        random.seed(99999)
        before_global = random.getstate()
        picked, rec = draw_prompts(
            rows,
            8,
            mode="kband",
            seed=1234,
            prepass=(prepass_path, sha),
            k_min=2,
            k_max=5,
        )
        after_global = random.getstate()
        assert before_global == after_global, "kband draw_prompts must never touch global random state"

        pinned_kband_ids = ["t02", "t03", "t05", "t21", "t22", "t29", "t31", "t32"]
        assert [r["id"] for r in picked] == pinned_kband_ids
        for r in picked:
            idx = int(r["id"][1:])
            assert 2 <= (idx % 9) <= 5
        assert rec == {
            "draw_mode": "kband",
            "draw_n": 8,
            "draw_seed": 1234,
            "drawn_ids_sha256": "dcee2a8bdbc8f7d4576cc3b6e89a09273c676e70cdbabf8f522a103b286a0bf8",
            "prepass_sha256": sha,
            "k_min": 2,
            "k_max": 5,
            "band_pool_size": 18,
        }

        # Same result for shuffled rows
        shuffled_rows = list(rows)
        random.Random(42).shuffle(shuffled_rows)
        assert [r["id"] for r in shuffled_rows] != [r["id"] for r in rows]
        picked_shuf_rows, rec_shuf_rows = draw_prompts(
            shuffled_rows,
            8,
            mode="kband",
            seed=1234,
            prepass=(prepass_path, sha),
            k_min=2,
            k_max=5,
        )
        assert [r["id"] for r in picked_shuf_rows] == pinned_kband_ids
        assert rec_shuf_rows == rec

        # Same result for a shuffled pre-pass file
        prepass_lines = prepass_path.read_text(encoding="utf-8").splitlines(keepends=True)
        shuf_prepass_lines = list(prepass_lines)
        random.Random(42).shuffle(shuf_prepass_lines)
        assert shuf_prepass_lines != prepass_lines
        shuf_prepass_path = tmp_path / "prepass_shuffled.jsonl"
        shuf_prepass_path.write_text("".join(shuf_prepass_lines), encoding="utf-8", newline="\n")
        shuf_sha = hashlib.sha256(shuf_prepass_path.read_bytes()).hexdigest()
        picked_shuf_prepass, rec_shuf_prepass = draw_prompts(
            shuffled_rows,
            8,
            mode="kband",
            seed=1234,
            prepass=(shuf_prepass_path, shuf_sha),
            k_min=2,
            k_max=5,
        )
        assert [r["id"] for r in picked_shuf_prepass] == pinned_kband_ids
        assert {**rec_shuf_prepass, "prepass_sha256": sha} == rec

        # A second seed gives a different list
        picked_diff, rec_diff = draw_prompts(
            rows,
            8,
            mode="kband",
            seed=5678,
            prepass=(prepass_path, sha),
            k_min=2,
            k_max=5,
        )
        assert [r["id"] for r in picked_diff] != pinned_kband_ids
        assert rec_diff["draw_seed"] == 5678
        assert rec_diff["drawn_ids_sha256"] != rec["drawn_ids_sha256"]

    def test_d_refusals_by_name(self, tmp_path):
        from s2_draw import draw_prompts

        gold_dir = _draw_gold_40(tmp_path / "gold")
        rows, _ = s2.read_gold(gold_dir)
        tune_ids = [f"t{i:02d}" for i in range(40)]
        prepass_path = tmp_path / "prepass.jsonl"
        sha = _write_fake_prepass(prepass_path, tune_ids, k_fn=lambda idx: idx % 9, g=8)

        # 1. uniform without seed or with seed=True -> missing_seed
        with pytest.raises(SystemExit, match="missing_seed"):
            draw_prompts(rows, 10, mode="uniform", seed=None)
        with pytest.raises(SystemExit, match="missing_seed"):
            draw_prompts(rows, 10, mode="uniform", seed=True)

        # 2. uniform n exceeds pool -> n_exceeds_pool
        with pytest.raises(SystemExit, match="n_exceeds_pool"):
            draw_prompts(rows, 41, mode="uniform", seed=1234)

        # 3. kband without seed or with seed=True -> missing_seed
        with pytest.raises(SystemExit, match="missing_seed"):
            draw_prompts(rows, 5, mode="kband", seed=None, prepass=(prepass_path, sha), k_min=1, k_max=3)
        with pytest.raises(SystemExit, match="missing_seed"):
            draw_prompts(rows, 5, mode="kband", seed=True, prepass=(prepass_path, sha), k_min=1, k_max=3)

        # 4. kband sha mismatch -> prepass_sha_mismatch
        with pytest.raises(SystemExit, match="prepass_sha_mismatch"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(prepass_path, "0" * 64),
                k_min=1,
                k_max=3,
            )

        # 5. kband missing file -> prepass_missing_file
        with pytest.raises(SystemExit, match="prepass_missing_file"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(tmp_path / "nonexistent.jsonl", sha),
                k_min=1,
                k_max=3,
            )

        # 6. split row with no pre-pass record -> prepass_missing_split_id
        incomplete_path = tmp_path / "incomplete.jsonl"
        incomplete_sha = _write_fake_prepass(incomplete_path, tune_ids[:-1])
        with pytest.raises(SystemExit, match="prepass_missing_split_id"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(incomplete_path, incomplete_sha),
                k_min=1,
                k_max=3,
            )

        # 7. pre-pass id not in the split -> prepass_extra_id
        extra_path = tmp_path / "extra.jsonl"
        extra_sha = _write_fake_prepass(extra_path, [*tune_ids, "h00"])
        with pytest.raises(SystemExit, match="prepass_extra_id"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(extra_path, extra_sha),
                k_min=1,
                k_max=3,
            )

        # 7b. duplicate id in the pre-pass -> prepass_duplicate_id
        dup_path = tmp_path / "dup.jsonl"
        dup_sha = _write_fake_prepass(dup_path, [*tune_ids, "t00"])
        with pytest.raises(SystemExit, match="prepass_duplicate_id"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(dup_path, dup_sha),
                k_min=1,
                k_max=3,
            )

        # 7c. mixed g values across pre-pass records -> prepass_mixed_g
        import hashlib

        mixed_g_path = tmp_path / "mixed_g.jsonl"
        mixed_g_lines = []
        for idx, cid in enumerate(tune_ids):
            g_row = 16 if idx == 0 else 8
            mixed_g_lines.append(
                json.dumps(
                    {
                        "id": cid,
                        "gold_file": "grounded.jsonl",
                        "gold_verdict": "supported",
                        "g": g_row,
                        "k_correct": 2, "pass_seed": 0, "descriptive": False,
                    }
                )
                + "\n"
            )
        mixed_g_path.write_text("".join(mixed_g_lines), encoding="utf-8", newline="\n")
        mixed_g_sha = hashlib.sha256(mixed_g_path.read_bytes()).hexdigest()
        with pytest.raises(SystemExit, match="prepass_mixed_g"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(mixed_g_path, mixed_g_sha),
                k_min=1,
                k_max=4,
            )

        # 8. k_min > k_max -> k_min_gt_k_max
        with pytest.raises(SystemExit, match="k_min_gt_k_max"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(prepass_path, sha),
                k_min=5,
                k_max=2,
            )

        # 9. out-of-range values -> k_out_of_range (k_min < 0, k_max > g, record k_correct > g, or k_correct < 0)
        with pytest.raises(SystemExit, match="k_out_of_range"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(prepass_path, sha),
                k_min=-1,
                k_max=4,
            )
        with pytest.raises(SystemExit, match="k_out_of_range"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(prepass_path, sha),
                k_min=1,
                k_max=9,
            )
        bad_k_path = tmp_path / "bad_k.jsonl"
        bad_k_sha = _write_fake_prepass(bad_k_path, tune_ids, k_fn=lambda idx: 99 if idx == 0 else 2, g=8)
        with pytest.raises(SystemExit, match="k_out_of_range"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(bad_k_path, bad_k_sha),
                k_min=1,
                k_max=4,
            )
        neg_k_path = tmp_path / "neg_k.jsonl"
        neg_k_sha = _write_fake_prepass(neg_k_path, tune_ids, k_fn=lambda idx: -1 if idx == 0 else 2, g=8)
        with pytest.raises(SystemExit, match="k_out_of_range"):
            draw_prompts(
                rows,
                5,
                mode="kband",
                seed=1234,
                prepass=(neg_k_path, neg_k_sha),
                k_min=1,
                k_max=4,
            )

        # 10. n larger than the band -> n_exceeds_band
        with pytest.raises(SystemExit, match="n_exceeds_band"):
            draw_prompts(
                rows,
                19,
                mode="kband",
                seed=1234,
                prepass=(prepass_path, sha),
                k_min=2,
                k_max=5,
            )

        # 11. draw flags given for a mode that doesn't use them -> unused_draw_flags
        # (a) --draw first with --draw-seed, --prepass-*, or --k-*
        for bad_kw in (
            {"seed": 1234},
            {"prepass": (prepass_path, None)},
            {"prepass": (None, sha)},
            {"k_min": 2},
            {"k_max": 5},
        ):
            with pytest.raises(SystemExit, match="unused_draw_flags"):
                draw_prompts(rows, 10, mode="first", **bad_kw)
        for extra_cli in (
            ["--draw-seed", "1234"],
            ["--prepass-jsonl", str(prepass_path)],
            ["--prepass-sha256", sha],
            ["--k-min", "2"],
            ["--k-max", "5"],
        ):
            a_first_bad = s2.build_parser().parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / "first_bad"),
                    "--draw",
                    "first",
                    *extra_cli,
                ]
            )
            with pytest.raises(SystemExit, match="unused_draw_flags"):
                s2.execute(a_first_bad, build=s2.dry_bundle)

        # (b) --draw uniform with --prepass-* or --k-*
        for bad_kw in (
            {"prepass": (prepass_path, None)},
            {"prepass": (None, sha)},
            {"k_min": 2},
            {"k_max": 5},
        ):
            with pytest.raises(SystemExit, match="unused_draw_flags"):
                draw_prompts(rows, 10, mode="uniform", seed=1234, **bad_kw)
        for extra_cli in (
            ["--prepass-jsonl", str(prepass_path)],
            ["--prepass-sha256", sha],
            ["--k-min", "2"],
            ["--k-max", "5"],
        ):
            a_uni_bad = s2.build_parser().parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / "uni_bad"),
                    "--draw",
                    "uniform",
                    "--draw-seed",
                    "1234",
                    *extra_cli,
                ]
            )
            with pytest.raises(SystemExit, match="unused_draw_flags"):
                s2.execute(a_uni_bad, build=s2.dry_bundle)

        # (c) any draw flag with --rollout-only -> unused_draw_flags
        for extra_cli in (
            ["--draw", "uniform"],
            ["--draw", "kband"],
            ["--draw-seed", "1234"],
            ["--prepass-jsonl", str(prepass_path)],
            ["--prepass-sha256", sha],
            ["--k-min", "2"],
            ["--k-max", "5"],
        ):
            a_ro = s2.build_parser().parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / "ro_bad"),
                    "--rollout-only",
                    *extra_cli,
                ]
            )
            with pytest.raises(SystemExit, match="unused_draw_flags"):
                s2.execute(a_ro, build=s2.dry_bundle)

    @needs_torch
    def test_e_default_first_is_byte_identical_to_base_shape(self, tmp_path, capsys):
        import statistics

        from s2_draw import draw_prompts

        gold_dir = _draw_gold_40(tmp_path / "gold")
        rows, gold_hashes = s2.read_gold(gold_dir)
        picked_first, _ = draw_prompts(rows, 10, mode="first")
        assert [r["id"] for r in picked_first] == [f"t{i:02d}" for i in range(10)]

        tok = s2.DryTokenizer()
        ap = s2.build_parser()
        assert (
            ap.parse_args(["--gold", str(gold_dir), "--rnd-commit", "e48cc90c", "--out", str(tmp_path)]).draw
            == "first"
        )

        seen_pref: list[str] = []
        seen_reg: list[str] = []
        a = s2.apply_dry_defaults(
            ap.parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / "out_first"),
                    "--dry",
                    "--steps",
                    "5",
                    "--prompts",
                    "10",
                    "--g",
                    "2",
                    "--max-new",
                    "8",
                ]
            )
        )
        assert a.draw == "first"
        assert s2.execute(a, build=_make_stress_pool_fake_bundle(tok, seen_pref, seen_reg)) == 0

        # Assert the prompt ids that run_steps actually received equal select_prompts(rows, 10)
        expected_first_rows = s2.select_prompts(rows, 10)
        assert seen_reg == [s2.build_prompt(r) for r in expected_first_rows]
        seen_first_ids = [re.search(r"\[src@([^\]]+)\]", p).group(1) for p in seen_reg]
        assert seen_first_ids == [r["id"] for r in expected_first_rows] == [f"t{i:02d}" for i in range(10)]

        # Check stdout carries no draw_ keys and matches literal key set
        captured = capsys.readouterr()
        assert "draw_" not in captured.out
        stdout_lines = captured.out.strip().splitlines()
        assert len(stdout_lines) == 2
        stdout_summary = json.loads(stdout_lines[0])
        expected_stdout_keys = {
            "path",
            "stopped",
            "steps_completed",
            "peak_vram_mib_sampler",
            "rollout_tok_s",
            "step_s",
            "zero_variance_groups",
            "oom",
        }
        assert set(stdout_summary.keys()) == expected_stdout_keys
        assert not any(k.startswith("draw_") for k in stdout_summary)
        assert stdout_lines[1] == "S2-DRY"

        # Check s2-hf-eval-summary.json key set and literal expected contents
        eval_sum = json.loads((a.out / "s2-hf-eval-summary.json").read_text(encoding="utf-8"))
        expected_eval_summary_keys = {
            "supported_pred_to_prior_ratio",
            "p_cannot_tell_given_gold_supported",
            "gold_supported_groups_with_positive_cannot_tell_advantage",
            "false_accepts",
            "false_rejects",
            "miss_waterfall",
            "quote_relevance",
        }
        assert set(eval_sum.keys()) == expected_eval_summary_keys
        expected_eval_summary_doc = {
            "supported_pred_to_prior_ratio": 0.0,
            "p_cannot_tell_given_gold_supported": 0.0,
            "gold_supported_groups_with_positive_cannot_tell_advantage": 0.0,
            "false_accepts": 0,
            "false_rejects": 0,
            "miss_waterfall": {
                "rule_2": 0,
                "no_minimum": 0,
                "case": 0,
                "punctuation": 0,
                "ellipsis_split": 0,
                "joined_passages": 0,
                "fuzzy": 0,
                "unmatched": 0,
            },
            "quote_relevance": {
                "relevant": 0,
                "total": 0,
                "share": 0.0,
            },
        }
        assert eval_sum == expected_eval_summary_doc

        # Check every row in s2-hf-steps.jsonl against literal key set and literal deterministic values
        step_rows = [
            json.loads(x) for x in (a.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(step_rows) == 5
        expected_step_row_keys = {
            "step",
            "groups",
            "active_groups",
            "zero_variance_groups",
            "tokens",
            "rollout_s",
            "train_s",
            "step_s",
            "mean_reward",
            "loss",
            "peak_alloc_mib",
            "grad_ckpt",
            "grad_ckpt_rollout",
        }
        for idx, sr in enumerate(step_rows):
            assert set(sr.keys()) == expected_step_row_keys
            assert not any(k.startswith("draw_") for k in sr)
            assert sr["step"] == idx
            assert sr["groups"] == 2
            assert sr["active_groups"] == 2
            assert sr["zero_variance_groups"] == 0
            assert sr["tokens"] == 16
            assert sr["mean_reward"] == 0.35
            assert sr["loss"] == 0.0
            assert sr["peak_alloc_mib"] == 1000
            assert sr["grad_ckpt"] is True
            assert sr["grad_ckpt_rollout"] is True

        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        readout_doc = receipt["readout"]

        # Literal base-shaped key sets (independent of the run's output)
        expected_receipt_keys = {
            "test",
            "path",
            "dry",
            "model",
            "revision_registered",
            "revision_resolved",
            "registered",
            "run",
            "rnd_commit",
            "gold_files_sha256",
            "prompt_split",
            "prompt_ids",
            "environment",
            "llama_cpp",
            "pass_line",
            "stopped",
            "readout",
            "verdict",
            "save_all",
            "save_all_sha256",
            "stack_versions",
            "pip_freeze_sha256",
        }
        expected_run_keys = {"steps", "prompts", "g", "max_new", "dry"}
        expected_readout_keys = {
            "steps_requested",
            "steps_completed",
            "peak_vram_mib_sampler",
            "sampler_samples",
            "vram_mib_sampler_series",
            "peak_vram_mib_torch",
            "peak_reserved_mib_torch",
            "fits",
            "oom",
            "rollout_tok_s",
            "step_s",
            "wall_s",
            "zero_variance_groups",
            "active_groups",
            "groups",
            "complete",
            "pass",
            "eval_summary",
        }
        assert set(receipt.keys()) == expected_receipt_keys
        assert set(receipt["run"].keys()) == expected_run_keys
        assert set(readout_doc.keys()) == expected_readout_keys
        assert "flags" not in readout_doc
        assert receipt["prompt_ids"] == [f"t{i:02d}" for i in range(10)]

        # Construct literal base-shaped expected receipt and compare byte-for-byte
        step_s_list = [r["step_s"] for r in step_rows]
        total_rollout_s = sum(r["rollout_s"] for r in step_rows)
        base_readout = {
            "steps_requested": 5,
            "steps_completed": 5,
            "peak_vram_mib_sampler": None,
            "sampler_samples": 0,
            "vram_mib_sampler_series": [],
            "peak_vram_mib_torch": 1000,
            "peak_reserved_mib_torch": None,
            "fits": False,
            "oom": False,
            "rollout_tok_s": round(80 / total_rollout_s, 2) if total_rollout_s > 0 else None,
            "step_s": {"p50": statistics.median(step_s_list), "max": max(step_s_list)},
            "wall_s": round(sum(step_s_list), 2),
            "zero_variance_groups": 0,
            "active_groups": 10,
            "groups": 10,
            "complete": False,
            "pass": False,
            "eval_summary": {
                "supported_pred_to_prior_ratio": 0.0,
                "p_cannot_tell_given_gold_supported": 0.0,
                "gold_supported_groups_with_positive_cannot_tell_advantage": 0.0,
                "false_accepts": 0,
                "false_rejects": 0,
                "miss_waterfall": {
                    "rule_2": 0,
                    "no_minimum": 0,
                    "case": 0,
                    "punctuation": 0,
                    "ellipsis_split": 0,
                    "joined_passages": 0,
                    "fuzzy": 0,
                    "unmatched": 0,
                },
                "quote_relevance_share": 0.0,
            },
        }
        guard_rec = s2.check_exclusive(
            s2.dry_smi, source="dry: a canned nvidia-smi output, never the real one"
        )
        base_env = s2.environment(
            gpu=None,
            dtype="float32 (dry, CPU)",
            batch=2,
            exclusive=None,
            cuda=None,
            guard=guard_rec,
        )
        base_receipt = {
            "test": "S2",
            "path": "hf",
            "dry": True,
            "model": "Qwen/Qwen3-8B",
            "revision_registered": "b968826d9c46",
            "revision_resolved": None,
            "registered": {
                "steps": 20,
                "prompts": 32,
                "groups_per_step": 2,
                "G": 8,
                "temperature": 0.7,
                "lora": {"r": 16, "alpha": 32, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                "learning_rate": 1e-5,
                "beta": 0.04,
                "epsilon": 0.2,
                "max_new_tokens": 2048,
            },
            "run": {"steps": 5, "prompts": 10, "g": 2, "max_new": 8, "dry": True},
            "rnd_commit": "e48cc90c",
            "gold_files_sha256": gold_hashes,
            "prompt_split": "tune",
            "prompt_ids": [f"t{i:02d}" for i in range(10)],
            "environment": base_env,
            "stack_versions": {k: s2.package_version(k) for k in s2.FROZEN_STACK_VERSIONS},
            "pip_freeze_sha256": None,
            "llama_cpp": None,
            "pass_line": {"vram_line_mib": 31200, "steps": 20, "oom": False},
            "stopped": None,
            "readout": base_readout,
            "verdict": "S2-DRY",
            "save_all": True,
            "save_all_sha256": s2.hashlib.sha256((a.out / "s2-hf-save-all.jsonl.gz").read_bytes()).hexdigest(),
        }
        expected_bytes = json.dumps(base_receipt, indent=1).encode("utf-8")
        actual_bytes = (a.out / "s2-hf.json").read_bytes()
        assert actual_bytes == expected_bytes

    @needs_torch
    def test_f_no_item_text_in_record_or_any_output_for_uniform_and_kband(self, tmp_path, capsys):
        from s2_draw import draw_prompts

        marker = "SECRET_DRAW_" + "ITEM_TEXT_7742"
        gold_dir = _draw_gold_40(tmp_path / "gold", marker=marker)
        rows, _ = s2.read_gold(gold_dir)
        tune_ids = [f"t{i:02d}" for i in range(40)]
        prepass_path = tmp_path / "prepass.jsonl"
        sha = _write_fake_prepass(prepass_path, tune_ids, k_fn=lambda idx: idx % 9, g=8)

        # Direct draw_prompts records must not contain marker
        picked_u, rec_u = draw_prompts(rows, 10, mode="uniform", seed=1234)
        picked_k, rec_k = draw_prompts(
            rows, 8, mode="kband", seed=1234, prepass=(prepass_path, sha), k_min=2, k_max=5
        )
        assert marker not in json.dumps(rec_u)
        assert marker not in json.dumps(rec_k)

        tok = s2.DryTokenizer()

        # Execute smoke script with --draw uniform
        seen_u_reg: list[str] = []
        a_u = s2.apply_dry_defaults(
            s2.build_parser().parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / "out_uniform"),
                    "--dry",
                    "--steps",
                    "5",
                    "--prompts",
                    "10",
                    "--g",
                    "2",
                    "--max-new",
                    "8",
                    "--draw",
                    "uniform",
                    "--draw-seed",
                    "1234",
                ]
            )
        )
        assert s2.execute(a_u, build=_make_stress_pool_fake_bundle(tok, [], seen_u_reg)) == 0
        receipt_u = json.loads((a_u.out / "s2-hf.json").read_text(encoding="utf-8"))
        expected_u_ids = [
            "t00",
            "t02",
            "t05",
            "t06",
            "t07",
            "t15",
            "t22",
            "t28",
            "t36",
            "t37",
        ]
        assert receipt_u["prompt_ids"] == expected_u_ids
        seen_u_ids = [re.search(r"\[src@([^\]]+)\]", p).group(1) for p in seen_u_reg]
        assert seen_u_reg == [s2.build_prompt(r) for r in picked_u]
        assert seen_u_ids == [r["id"] for r in picked_u] == expected_u_ids
        assert (
            s2.ids_sha256(seen_u_ids)
            == rec_u["drawn_ids_sha256"]
            == "0f33b135252d80b75ccc9486f3e2a3e3a188fb143741b503dd05d3bfabaf96b7"
        )
        for container in (receipt_u["readout"]["flags"], receipt_u["run"], receipt_u):
            assert container["draw_mode"] == "uniform"
            assert container["draw_n"] == 10
            assert container["draw_seed"] == 1234
            assert (
                container["drawn_ids_sha256"]
                == "0f33b135252d80b75ccc9486f3e2a3e3a188fb143741b503dd05d3bfabaf96b7"
            )
            assert "prepass_sha256" not in container

        # Execute smoke script with --draw kband
        seen_k_reg: list[str] = []
        a_k = s2.apply_dry_defaults(
            s2.build_parser().parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / "out_kband"),
                    "--dry",
                    "--steps",
                    "4",
                    "--prompts",
                    "8",
                    "--g",
                    "2",
                    "--max-new",
                    "8",
                    "--draw",
                    "kband",
                    "--draw-seed",
                    "1234",
                    "--prepass-jsonl",
                    str(prepass_path),
                    "--prepass-sha256",
                    sha,
                    "--k-min",
                    "2",
                    "--k-max",
                    "5",
                ]
            )
        )
        assert s2.execute(a_k, build=_make_stress_pool_fake_bundle(tok, [], seen_k_reg)) == 0
        receipt_k = json.loads((a_k.out / "s2-hf.json").read_text(encoding="utf-8"))
        expected_k_ids = ["t02", "t03", "t05", "t21", "t22", "t29", "t31", "t32"]
        assert receipt_k["prompt_ids"] == expected_k_ids
        seen_k_ids = [re.search(r"\[src@([^\]]+)\]", p).group(1) for p in seen_k_reg]
        assert seen_k_reg == [s2.build_prompt(r) for r in picked_k]
        assert seen_k_ids == [r["id"] for r in picked_k] == expected_k_ids
        assert (
            s2.ids_sha256(seen_k_ids)
            == rec_k["drawn_ids_sha256"]
            == "dcee2a8bdbc8f7d4576cc3b6e89a09273c676e70cdbabf8f522a103b286a0bf8"
        )
        for container in (receipt_k["readout"]["flags"], receipt_k["run"], receipt_k):
            assert container["draw_mode"] == "kband"
            assert container["draw_n"] == 8
            assert container["draw_seed"] == 1234
            assert (
                container["drawn_ids_sha256"]
                == "dcee2a8bdbc8f7d4576cc3b6e89a09273c676e70cdbabf8f522a103b286a0bf8"
            )
            assert container["prepass_sha256"] == sha
            assert container["k_min"] == 2
            assert container["k_max"] == 5
            assert container["band_pool_size"] == 18

        captured = capsys.readouterr()
        assert marker not in captured.out
        assert marker not in captured.err
        for out_dir in (a_u.out, a_k.out):
            for p in out_dir.rglob("*"):
                if p.is_file() and not p.name.endswith(".jsonl.gz"):
                    assert marker not in p.read_text(encoding="utf-8"), f"marker leaked into {p}"


# ---------------------------------------------------------------------------
# 2026-11-04 Diagnostics & Pre-pass seed (Items 1..5)
# ---------------------------------------------------------------------------


class TestPrepassSeed:
    @needs_torch
    def test_prepass_seed_0_equals_base_header_form_and_resumes_when_script_hash_matches(
        self, tmp_path, monkeypatch
    ):
        import hashlib
        from contextlib import nullcontext

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        ap = s2.build_parser()
        a0 = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "seed0_out"),
                "--rollout-only",
                "--g",
                "2",
                "--max-new",
                "16",
                "--quote-reward",
                "v2",
            ]
        )
        assert a0.prepass_seed == 0
        assert "prepass_seed" not in s2.active_flags(a0)

        fixed_script_sha = "906cae4b804b10614c42bd1e9f4240b6915cb1bb936153c97bb23d5f6730b48a"
        monkeypatch.setattr(s2, "script_sha256", lambda: fixed_script_sha)

        seen_seeds: list[int] = []
        orig_manual_seed = torch.manual_seed

        def spy_manual_seed(seed_val):
            seen_seeds.append(int(seed_val))
            return orig_manual_seed(seed_val)

        monkeypatch.setattr(torch, "manual_seed", spy_manual_seed)

        def build_fake(_args_ns):
            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": lambda text, g_cnt, max_n, temp: (
                    tok.encode(text),
                    [tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]] * g_cnt,
                    [False] * g_cnt,
                ),
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text,
                    row["label"],
                    s2.passages_of(row),
                    tt,
                    truncated=truncated,
                    quote_reward="v2",
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "llama": None,
                "dtype_census": None,
            }

        assert s2.execute(a0, build=build_fake, rig=_FakeRig(a0)) == 0
        expected_ids = ["t_a", "t_b", "t_d", "t_k", "t_m", "t_z"]
        assert seen_seeds == [s2.item_seed(cid, 0) for cid in expected_ids]

        header_bytes = (a0.out / "s2-hf-prepass-header.json").read_bytes()
        expected_header_dict = {
            "ids_sha256": "1faaea262a971c66357c551cd28fbe6c8f94b922cc36fa1a8ac041e3fd7373ca",
            "script_sha256": "906cae4b804b10614c42bd1e9f4240b6915cb1bb936153c97bb23d5f6730b48a",
            "gold_files_sha256": {
                "grounded.jsonl": "9cdc8be6611000289f539cb41573173827e3a4417249964496614a69fe4a8c31",
                "prs/grounded-prs.jsonl": "2ad0462e5bb52f899dd802ea505cd62514a93690393d05401e8971924a1ba2e9",
                "diffs/reasoning-diffs.jsonl": "aa66a79b4276ef90e68856ea30427d53d2058cc9c56e685f87bb92f259c2858d",
            },
            "g": 2,
            "max_new": 16,
            "quote_reward": "v2",
            "seed": 0,
            "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "reproducible": True,
            "descriptive": False,
            "stack_versions": {k: s2.package_version(k) for k in s2.FROZEN_STACK_VERSIONS},
            "pip_freeze_sha256": None,
        }
        expected_literal_header_bytes = (json.dumps(expected_header_dict, indent=1) + "\n").encode("utf-8")
        assert header_bytes == expected_literal_header_bytes

        # Also verify resume works when the script hash matches and seed is default 0:
        tonight_header_bytes = (
            b"{\n"
            b' "ids_sha256": "5c41d3fb437be23d4cc7ef673963bbe3caba78d832dd23665c58c882679d72f8",\n'
            b' "script_sha256": "906cae4b804b10614c42bd1e9f4240b6915cb1bb936153c97bb23d5f6730b48a",\n'
            b' "gold_files_sha256": {\n'
            b'  "grounded.jsonl": "59601d10dfea03351e506d08f5296f285d7f7cb37ba5ace79ae2bd006940aabd",\n'
            b'  "prs/grounded-prs.jsonl": "6e2d068eb7d2f3bdbce5a229dd9eedac50c09e3f57abbacba78e06f3e3c34f13",\n'
            b'  "diffs/reasoning-diffs.jsonl": "4f4c058a67f9f564bbb29123e580bee5c6f42f97ad334e4ceb089abfb3586989"\n'
            b" },\n"
            b' "g": 8,\n'
            b' "max_new": 2048,\n'
            b' "quote_reward": "v2",\n'
            b' "seed": 0,\n'
            b' "model_revision": "b968826d9c46dd6066d109eabc6255188de91218"\n'
            b"}\n"
        )
        assert (
            hashlib.sha256(tonight_header_bytes).hexdigest()
            == "1e749d8b897804fff93125c0888fa42cfb07a560ab93f731fe37370d49f3020f"
        )
        tonight_dir = tmp_path / "tonight_resume"
        tonight_dir.mkdir(parents=True, exist_ok=True)
        tonight_hdr_path = tonight_dir / "s2-hf-prepass-header.json"
        tonight_jsonl_path = tonight_dir / "s2-hf-prepass.jsonl"
        tonight_hdr_path.write_bytes(tonight_header_bytes)
        tonight_jsonl_path.write_bytes(b"")

        monkeypatch.setattr(
            s2,
            "ids_sha256",
            lambda _ids: "5c41d3fb437be23d4cc7ef673963bbe3caba78d832dd23665c58c882679d72f8",
        )
        tonight_gold_hashes = {
            "grounded.jsonl": "59601d10dfea03351e506d08f5296f285d7f7cb37ba5ace79ae2bd006940aabd",
            "prs/grounded-prs.jsonl": "6e2d068eb7d2f3bdbce5a229dd9eedac50c09e3f57abbacba78e06f3e3c34f13",
            "diffs/reasoning-diffs.jsonl": "4f4c058a67f9f564bbb29123e580bee5c6f42f97ad334e4ceb089abfb3586989",
        }
        a_tonight = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tonight_dir),
                "--rollout-only",
                "--quote-reward",
                "v2",
            ]
        )
        rows_g, _ = s2.read_gold(gold_dir)
        prompts_g = s2.select_prompts(rows_g, None)
        s_rows, c_recs, st_rec, is_oom = s2.run_prepass(
            a=a_tonight,
            prompts=prompts_g,
            gold_hashes=tonight_gold_hashes,
            bundle=build_fake(a_tonight),
        )
        assert st_rec is None and not is_oom and len(c_recs) == 6
        assert (
            hashlib.sha256(tonight_hdr_path.read_bytes()).hexdigest()
            == "1e749d8b897804fff93125c0888fa42cfb07a560ab93f731fe37370d49f3020f"
        )

    @needs_torch
    def test_prepass_seed_1_changes_item_seeds_and_writes_seed_1_in_header(self, tmp_path, monkeypatch):
        from contextlib import nullcontext

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        ap = s2.build_parser()
        a0 = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "seed0_draws_out"),
                "--rollout-only",
                "--g",
                "2",
                "--max-new",
                "16",
            ]
        )
        a1 = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "seed1_out"),
                "--rollout-only",
                "--prepass-seed",
                "1",
                "--g",
                "2",
                "--max-new",
                "16",
            ]
        )
        assert a1.prepass_seed == 1
        assert s2.active_flags(a1)["prepass_seed"] == 1

        seen_seeds: list[int] = []
        orig_manual_seed = torch.manual_seed

        def spy_manual_seed(seed_val):
            seen_seeds.append(int(seed_val))
            return orig_manual_seed(seed_val)

        monkeypatch.setattr(torch, "manual_seed", spy_manual_seed)

        draws_by_run: list[list[float]] = []

        def make_build_fake(draw_sink: list[list[float]]):
            def _rollout(text: str, g_cnt: int, max_n: int, temp: float):
                draw_sink.append(torch.rand(4).tolist())
                return (
                    tok.encode(text),
                    [tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]] * g_cnt,
                    [False] * g_cnt,
                )

            def _build(_args_ns):
                return {
                    "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                    "tok": tok,
                    "mode": lambda training: None,
                    "state": lambda: {"grad_ckpt": False},
                    "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                    "ref_adapter_off": nullcontext,
                    "rollout": _rollout,
                    "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                        text, row["label"], s2.passages_of(row), tt, truncated=truncated
                    ),
                    "sync": lambda: None,
                    "peak_alloc_mib": lambda: 1000,
                    "revision": "rev123",
                    "llama": None,
                    "dtype_census": None,
                }

            return _build

        draws_0: list[list[float]] = []
        assert s2.execute(a0, build=make_build_fake(draws_0), rig=_FakeRig(a0)) == 0
        seen_seeds_0 = list(seen_seeds)
        seen_seeds.clear()

        draws_1: list[list[float]] = []
        assert s2.execute(a1, build=make_build_fake(draws_1), rig=_FakeRig(a1)) == 0
        seen_seeds_1 = list(seen_seeds)
        _ = draws_by_run

        expected_ids = ["t_a", "t_b", "t_d", "t_k", "t_m", "t_z"]
        seeds_0 = [s2.item_seed(cid, 0) for cid in expected_ids]
        seeds_1 = [s2.item_seed(cid, 1) for cid in expected_ids]
        assert len(draws_0) == len(expected_ids) and len(draws_1) == len(expected_ids)
        assert all(d0 != d1 for d0, d1 in zip(draws_0, draws_1, strict=True))
        assert seen_seeds_0 == seeds_0
        assert seen_seeds_1 == seeds_1
        assert all(s0 != s1 for s0, s1 in zip(seeds_0, seeds_1, strict=True))

        header_1 = json.loads((a1.out / "s2-hf-prepass-header.json").read_text(encoding="utf-8"))
        assert header_1["seed"] == 1
        receipt_1 = json.loads((a1.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt_1["readout"]["flags"]["prepass_seed"] == 1
        assert receipt_1["run"]["prepass_seed"] == 1

    @needs_torch
    def test_prepass_seed_mismatch_refuses_resume(self, tmp_path):
        from contextlib import nullcontext

        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        ap = s2.build_parser()

        def build_fake(_args_ns):
            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": lambda text, g_cnt, max_n, temp: (
                    tok.encode(text),
                    [tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]] * g_cnt,
                    [False] * g_cnt,
                ),
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        out_dir = tmp_path / "mismatch_seed"
        a_seed0 = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(out_dir),
                "--rollout-only",
                "--g",
                "2",
                "--max-new",
                "16",
            ]
        )
        assert s2.execute(a_seed0, build=build_fake, rig=_FakeRig(a_seed0)) == 0

        a_seed1 = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(out_dir),
                "--rollout-only",
                "--prepass-seed",
                "1",
                "--g",
                "2",
                "--max-new",
                "16",
            ]
        )
        with pytest.raises(SystemExit, match="refusing to resume prepass: header mismatch on seed"):
            s2.execute(a_seed1, build=build_fake, rig=_FakeRig(a_seed1))

        # And starting with seed 1, then attempting to resume with seed 0 or seed 2 also refuses
        out_dir_1 = tmp_path / "mismatch_from_1"
        a_start1 = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(out_dir_1),
                "--rollout-only",
                "--prepass-seed",
                "1",
                "--g",
                "2",
                "--max-new",
                "16",
            ]
        )
        assert s2.execute(a_start1, build=build_fake, rig=_FakeRig(a_start1)) == 0
        for other_seed in ("0", "2"):
            a_other = ap.parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(out_dir_1),
                    "--rollout-only",
                    "--prepass-seed",
                    other_seed,
                    "--g",
                    "2",
                    "--max-new",
                    "16",
                ]
            )
            with pytest.raises(SystemExit, match="refusing to resume prepass: header mismatch on seed"):
                s2.execute(a_other, build=build_fake, rig=_FakeRig(a_other))


class TestSignDiagnostics:
    def test_sign_diagnostics_help_text_describes_outcome_var_and_correct_var_in_one_line_each(self):
        ap = s2.build_parser()
        sign_action = next(act for act in ap._actions if "--sign-diagnostics" in act.option_strings)
        lines = [ln.strip() for ln in (sign_action.help or "").splitlines() if ln.strip()]
        outcome_lines = [ln for ln in lines if ln.startswith("outcome_var:")]
        correct_lines = [ln for ln in lines if ln.startswith("correct_var:")]
        assert len(outcome_lines) == 1
        assert len(correct_lines) == 1
        assert "outcome" in outcome_lines[0] and "quote" in outcome_lines[0]
        assert "0/1" in correct_lines[0] and "correct" in correct_lines[0]

    @needs_torch
    def test_sign_diagnostics_group_and_step_counts_and_variances_on_v2_supported_group(
        self, tmp_path, capsys
    ):
        from contextlib import nullcontext

        import torch

        marker = "".join(("PLANTED", "_", "SIGN", "_", "DIAG", "_", "3104"))
        gold_dir = _gold(
            tmp_path / "gold",
            {
                "grounded.jsonl": [
                    _row(
                        f"g{i:02d}",
                        label="supported",
                        claim=f"Claim {marker}.",
                        text=f"alpha beta gamma delta {marker}",
                    )
                    for i in range(20)
                ],
                "prs/grounded-prs.jsonl": [_row(f"p{i:02d}") for i in range(20)],
                "diffs/reasoning-diffs.jsonl": [_row(f"r{i:02d}", label="unsupported") for i in range(20)],
            },
        )
        tok = s2.DryTokenizer()
        fake_cuda = _FakeCuda()

        # Group 0 (active, gold="supported", quote_reward="v2"):
        #   2 rollouts with correct "supported" verdict and found quote ("alpha beta gamma")
        #     -> outcome = 1.0, quote_credit = 0.05, total = 1.05, correct = True
        #   1 rollout with correct "supported" verdict and unfound quote ("missing quote text here")
        #     -> outcome = -0.2, quote_credit = 0.0, total = -0.2, correct = True
        #   1 rollout with wrong "unsupported" verdict ("alpha beta gamma")
        #     -> outcome = -1.0, quote_credit = 0.0, total = -1.0, correct = False
        #   Group mean reward = (1.05 + 1.05 - 0.2 - 1.0) / 4 = 0.9 / 4 = 0.225.
        #   Rollout 2 has correct=True and advantage < 0 (-0.2 - 0.225 = -0.425 < 0)
        #     -> correct_n = 3, correct_negative_adv_n = 1, truncated_n = 0.
        #   outcome_var = pvariance([1.0, 1.0, -0.2, -1.0]) = 0.72.
        #   correct_var = pvariance([1.0, 1.0, 1.0, 0.0]) = 0.1875.
        #   total_reward_var = pvariance([1.05, 1.05, -0.2, -1.0]) = 0.760625.
        # Group 1 (zero-variance, gold="supported", quote_reward="v2"):
        #   4 rollouts with correct "supported" verdict and found quote ("alpha beta gamma")
        #     -> correct_n = 4, correct_negative_adv_n = 0, truncated_n = 0,
        #        outcome_var = 0.0, correct_var = 0.0, total_reward_var = 0.0.
        # Step 0 summary:
        #   step_correct_active_n = 3, step_correct_zv_n = 4.
        #   correct_negative_adv_share = 1 / 3 = 0.333333 (over active groups only, not 1 / 7).
        #   outcome_to_total_var_ratio = 0.72 / 0.760625 = 0.94659.
        #   correct_to_total_var_ratio = 0.1875 / 0.760625 = 0.246508.
        grp_call = {"n": 0}

        def scripted_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            grp_call["n"] += 1
            p_ids = tok.encode(prompt_text)
            found_rep = tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]
            unfound_rep = tok.encode(_reply("supported", "missing quote text here")) + [tok.eos_token_id]
            wrong_rep = tok.encode(_reply("unsupported", "alpha beta gamma")) + [tok.eos_token_id]
            if grp_call["n"] == 1:
                comps = [found_rep, found_rep, unfound_rep, wrong_rep]
            else:
                comps = [found_rep, found_rep, found_rep, found_rep]
            return p_ids, comps, [False] * g_cnt

        class _FakeTrainModel:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def parameters(self):
                return [self._p]

            def __call__(self, input_ids, attention_mask=None, **kwargs):
                logits = (
                    torch.zeros((input_ids.shape[0], input_ids.shape[1], tok.vocab_size), dtype=torch.float32)
                    + self._p * 0.0
                )
                return Namespace(logits=logits)

        def build_fake(_args_ns):
            return {
                "model": _FakeTrainModel(),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": True},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": scripted_rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text,
                    row["label"],
                    s2.passages_of(row),
                    tt,
                    truncated=truncated,
                    quote_reward="v2",
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        ap = s2.build_parser()
        a = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "sign_out"),
                "--steps",
                "1",
                "--prompts",
                "4",
                "--g",
                "4",
                "--max-new",
                "32",
                "--quote-reward",
                "v2",
                "--sign-diagnostics",
            ]
        )
        assert a.sign_diagnostics is True
        assert s2.active_flags(a)["sign_diagnostics"] is True
        assert s2.execute(a, build=build_fake, rig=_FakeRig(a)) == 1  # cut run -> S2-LIMITED

        group_rows = [
            json.loads(line)
            for line in (a.out / "s2-hf-groups.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(group_rows) == 2

        g0, g1 = group_rows[0], group_rows[1]
        assert g0["step"] == 0 and g0["group"] == 0 and g0["prompt_id"] == "g00"
        assert g0["k_correct"] == 3
        assert g0["correct_n"] == 3
        assert g0["correct_negative_adv_n"] == 1
        assert g0["truncated_n"] == 0
        assert "verdict_score_var" not in g0
        assert g0["outcome_var"] == 0.72
        assert g0["correct_var"] == 0.1875
        assert g0["total_reward_var"] == 0.760625
        assert g0["zero_variance_label"] is None

        assert g1["step"] == 0 and g1["group"] == 1 and g1["prompt_id"] == "g01"
        assert g1["k_correct"] == 4
        assert g1["correct_n"] == 4
        assert g1["correct_negative_adv_n"] == 0
        assert g1["truncated_n"] == 0
        assert "verdict_score_var" not in g1
        assert g1["outcome_var"] == 0.0
        assert g1["correct_var"] == 0.0
        assert g1["total_reward_var"] == 0.0
        assert g1["zero_variance_label"] == "all correct with full credit"

        step_rows = [
            json.loads(line)
            for line in (a.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(step_rows) == 1
        s0 = step_rows[0]
        assert s0["active_groups"] == 1
        assert s0["zero_variance_groups"] == 1
        assert s0["step_correct_active_n"] == 3
        assert s0["step_correct_zv_n"] == 4
        assert s0["correct_negative_adv_share"] == 0.333333
        assert "verdict_to_total_var_ratio" not in s0
        assert s0["outcome_to_total_var_ratio"] == 0.94659
        assert s0["correct_to_total_var_ratio"] == 0.246508

        captured = capsys.readouterr()
        assert marker not in captured.out and marker not in captured.err
        for p in a.out.rglob("*"):
            if p.is_file() and not p.name.endswith(".jsonl.gz"):
                assert marker not in p.read_text(encoding="utf-8")

    @needs_torch
    def test_sign_diagnostics_mask_truncated_excludes_truncated_rollouts_and_records_truncated_n(
        self, tmp_path
    ):
        from contextlib import nullcontext

        import torch

        gold_dir = _gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        fake_cuda = _FakeCuda()

        # 4 rollouts in group 0 (gold="supported", quote_reward="v2", mask_truncated=True):
        #   rollout 0 (non-truncated): correct supported + found quote -> outcome=1.0, total=1.05, correct=True
        #   rollout 1 (non-truncated): correct supported + unfound quote -> outcome=-0.2, total=-0.2, correct=True
        #   rollout 2 (non-truncated): wrong unsupported -> outcome=-1.0, total=-1.0, correct=False
        #   rollout 3 (TRUNCATED): reward_details gives outcome=-1.0, total=-1.0, correct=False,
        #     AND we pass a custom reward_fn that gives -9.0 on truncated so if truncated were included
        #     total_reward_var would jump dramatically, and we also test a monkeypatched reward_details
        #     where truncated rollout has correct=True to prove correct_n and correct_negative_adv_n
        #     are counted over non-truncated rollouts only.
        # Over the 3 non-truncated rollouts:
        #   outcomes = [1.0, -0.2, -1.0] -> pvariance = 0.675556
        #   correct  = [1.0, 1.0, 0.0]   -> pvariance = 0.222222
        #   rewards  = [1.05, -0.2, -1.0] -> mean = -0.05; pvariance = 0.711667
        #   advantages on non-truncated = [1.05 - (-0.05), -0.2 - (-0.05), -1.0 - (-0.05)] scaled
        #     rollout 1 has correct=True and advantage < 0 (-0.15 / sd < 0)
        #   correct_n = 2, correct_negative_adv_n = 1, truncated_n = 1.
        def scripted_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            p_ids = tok.encode(prompt_text)
            r_found = tok.encode(_reply("supported", "alpha beta")) + [tok.eos_token_id]
            r_unfound = tok.encode(_reply("supported", "missing quote text here")) + [tok.eos_token_id]
            r_wrong = tok.encode(_reply("unsupported", "alpha beta")) + [tok.eos_token_id]
            r_trunc = tok.encode(_reply("supported", "alpha beta"))
            return p_ids, [r_found, r_unfound, r_wrong, r_trunc], [False, False, False, True]

        class _FakeTrainModel:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def parameters(self):
                return [self._p]

            def __call__(self, input_ids, attention_mask=None, **kwargs):
                logits = (
                    torch.zeros((input_ids.shape[0], input_ids.shape[1], tok.vocab_size), dtype=torch.float32)
                    + self._p * 0.0
                )
                return Namespace(logits=logits)

        def build_fake(_args_ns):
            return {
                "model": _FakeTrainModel(),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": True},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": scripted_rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text,
                    row["label"],
                    s2.passages_of(row),
                    tt,
                    truncated=truncated,
                    quote_reward="v2",
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        ap = s2.build_parser()
        a = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "mask_trunc_sign"),
                "--steps",
                "1",
                "--prompts",
                "2",
                "--g",
                "4",
                "--max-new",
                "32",
                "--groups-per-update",
                "1",
                "--quote-reward",
                "v2",
                "--mask-truncated",
                "--sign-diagnostics",
            ]
        )
        assert s2.execute(a, build=build_fake, rig=_FakeRig(a)) == 1

        group_rows = [
            json.loads(line)
            for line in (a.out / "s2-hf-groups.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(group_rows) == 1
        g0 = group_rows[0]
        assert g0["truncations"] == 1
        assert g0["truncated_n"] == 1
        assert g0["correct_n"] == 2
        assert g0["correct_negative_adv_n"] == 1
        assert g0["outcome_var"] == 0.675556
        assert g0["correct_var"] == 0.222222
        assert g0["total_reward_var"] == 0.711667

    @needs_torch
    def test_sign_diagnostics_empty_active_correct_step_writes_null_share(self, tmp_path):
        from contextlib import nullcontext

        import torch

        gold_dir = _gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        fake_cuda = _FakeCuda()

        # Group 0 (active, gold="supported", quote_reward="v2"):
        #   2 rollouts wrong "unsupported" (-1.0) and 2 rollouts wrong "cannot_tell" (-0.3)
        #   -> active group (non-zero variance), 0 correct rollouts!
        # Group 1 (zero-variance, gold="supported", quote_reward="v2"):
        #   4 rollouts correct "supported" with found quote (1.05)
        #   -> zero-variance group, 4 correct rollouts!
        # Step has step_correct_active_n == 0 and step_correct_zv_n == 4.
        # correct_negative_adv_share MUST be None (JSON null), never 0.0 and never ZeroDivisionError.
        grp_call = {"n": 0}

        def scripted_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            grp_call["n"] += 1
            p_ids = tok.encode(prompt_text)
            r_unsup = tok.encode(_reply("unsupported", "alpha beta")) + [tok.eos_token_id]
            r_ct = tok.encode(_reply("cannot_tell", "alpha beta")) + [tok.eos_token_id]
            r_sup = tok.encode(_reply("supported", "alpha beta")) + [tok.eos_token_id]
            if grp_call["n"] == 1:
                return p_ids, [r_unsup, r_unsup, r_ct, r_ct], [False] * g_cnt
            return p_ids, [r_sup, r_sup, r_sup, r_sup], [False] * g_cnt

        class _FakeTrainModel:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def parameters(self):
                return [self._p]

            def __call__(self, input_ids, attention_mask=None, **kwargs):
                logits = (
                    torch.zeros((input_ids.shape[0], input_ids.shape[1], tok.vocab_size), dtype=torch.float32)
                    + self._p * 0.0
                )
                return Namespace(logits=logits)

        def build_fake(_args_ns):
            return {
                "model": _FakeTrainModel(),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": True},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": scripted_rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text,
                    row["label"],
                    s2.passages_of(row),
                    tt,
                    truncated=truncated,
                    quote_reward="v2",
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        ap = s2.build_parser()
        a = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "empty_active_correct"),
                "--steps",
                "1",
                "--prompts",
                "2",
                "--g",
                "4",
                "--max-new",
                "32",
                "--quote-reward",
                "v2",
                "--sign-diagnostics",
            ]
        )
        assert s2.execute(a, build=build_fake, rig=_FakeRig(a)) == 1

        raw_step_line = (a.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").strip()
        assert '"correct_negative_adv_share": null' in raw_step_line
        s0 = json.loads(raw_step_line)
        assert s0["step_correct_active_n"] == 0
        assert s0["step_correct_zv_n"] == 4
        assert s0["correct_negative_adv_share"] is None
        assert s0["outcome_to_total_var_ratio"] == 1.0
        assert s0["correct_to_total_var_ratio"] == 0.0


class TestPrepassBaseline:
    @needs_torch
    def test_prepass_baseline_writes_group_residuals_and_readout_k_band_means_with_no_item_text(
        self, tmp_path, capsys
    ):
        import hashlib
        from contextlib import nullcontext

        import torch

        marker = "".join(("PLANTED", "_", "BASELINE", "_", "CLAIM", "_", "4419"))
        gold_dir = _gold(
            tmp_path / "gold",
            {
                "grounded.jsonl": [
                    _row(
                        f"g{i:02d}",
                        label="supported",
                        claim=f"Claim {marker}.",
                        text=f"alpha beta gamma delta {marker}",
                    )
                    for i in range(20)
                ],
                "prs/grounded-prs.jsonl": [_row(f"p{i:02d}") for i in range(20)],
                "diffs/reasoning-diffs.jsonl": [_row(f"r{i:02d}", label="unsupported") for i in range(20)],
            },
        )

        # Fake pre-pass JSONL in tmp_path:
        #   g00: k_prepass = 2 (band [0,5]), online k = 6 -> residual = (6 - 2) / 8 = +0.5
        #   g01: k_prepass = 5 (band [0,5]), online k = 3 -> residual = (3 - 5) / 8 = -0.25
        #   g02: k_prepass = 7 (band [6,7]), online k = 5 -> residual = (5 - 7) / 8 = -0.25
        #   g03: k_prepass = 8 (band 8),     online k = 6 -> residual = (6 - 8) / 8 = -0.25
        # Mean residual by band:
        #   "[0,5]": (0.5 + (-0.25)) / 2 = 0.125
        #   "[6,7]": -0.25
        #   "8":     -0.25
        k_prepass_map = {"g00": 2, "g01": 5, "g02": 7, "g03": 8}
        k_online_map = {"g00": 6, "g01": 3, "g02": 5, "g03": 6}
        baseline_path = tmp_path / "fake_prepass_baseline.jsonl"
        baseline_sha = _write_fake_prepass(
            baseline_path,
            [f"g{i:02d}" for i in range(4)],
            k_fn=lambda idx: k_prepass_map[f"g{idx:02d}"],
            g=8,
        )
        assert baseline_sha == hashlib.sha256(baseline_path.read_bytes()).hexdigest()

        tok = s2.DryTokenizer()
        fake_cuda = _FakeCuda()
        grp_idx = {"n": 0}
        ordered_ids = ["g00", "g01", "g02", "g03"]

        def scripted_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            cid = ordered_ids[grp_idx["n"]]
            grp_idx["n"] += 1
            k_on = k_online_map[cid]
            p_ids = tok.encode(prompt_text)
            ok_comp = tok.encode(_reply("supported", "alpha beta gamma")) + [tok.eos_token_id]
            bad_comp = tok.encode(_reply("unsupported", "alpha beta gamma")) + [tok.eos_token_id]
            comps = [ok_comp] * k_on + [bad_comp] * (g_cnt - k_on)
            return p_ids, comps, [False] * g_cnt

        class _FakeTrainModel:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def parameters(self):
                return [self._p]

            def __call__(self, input_ids, attention_mask=None, **kwargs):
                logits = (
                    torch.zeros((input_ids.shape[0], input_ids.shape[1], tok.vocab_size), dtype=torch.float32)
                    + self._p * 0.0
                )
                return Namespace(logits=logits)

        def build_fake(_args_ns):
            return {
                "model": _FakeTrainModel(),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": True},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": scripted_rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        ap = s2.build_parser()
        a = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "baseline_out"),
                "--steps",
                "2",
                "--prompts",
                "4",
                "--g",
                "8",
                "--max-new",
                "32",
                "--prepass-baseline",
                str(baseline_path),
                "--prepass-baseline-sha256",
                baseline_sha,
            ]
        )
        assert s2.execute(a, build=build_fake, rig=_FakeRig(a)) == 1  # cut run -> S2-LIMITED

        group_rows = [
            json.loads(line)
            for line in (a.out / "s2-hf-groups.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(group_rows) == 4
        expected_group_baseline = [
            ("g00", 6, 2, 0.5),
            ("g01", 3, 5, -0.25),
            ("g02", 5, 7, -0.25),
            ("g03", 6, 8, -0.25),
        ]
        for gr, (cid, exp_k_on, exp_k_pre, exp_res) in zip(group_rows, expected_group_baseline, strict=True):
            assert gr["prompt_id"] == cid
            assert gr["k_correct"] == exp_k_on
            assert gr["k_prepass"] == exp_k_pre
            assert gr["residual"] == exp_res

        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt["readout"]["mean_residual_by_k_band"] == {
            "[0,5]": 0.125,
            "[6,7]": -0.25,
            "8": -0.25,
            "n_prompts": {"[0,5]": 2, "[6,7]": 1, "8": 1},
            "prompt_visits_max": 1,
        }
        assert receipt["readout"]["flags"]["prepass_baseline_sha256"] == baseline_sha

        captured = capsys.readouterr()
        assert marker not in captured.out and marker not in captured.err
        for p in a.out.rglob("*"):
            if p.is_file() and not p.name.endswith(".jsonl.gz"):
                content = p.read_text(encoding="utf-8")
                assert marker not in content
                assert str(baseline_path) not in content

    @needs_torch
    def test_prepass_baseline_wraps_and_counts_each_prompt_once_with_actual_group_size(self, tmp_path):
        from contextlib import nullcontext

        import torch

        gold_dir = _gold(tmp_path / "gold")
        # 3 prompts (g00, g01, g02), steps=3, groups_per_update=2 -> 6 group visits (wraps: 3*2 > 3):
        #   visit 0: g00 (k_prepass=1, band [0,5]) -> online k=3 -> residual = (3 - 1) / 4 = +0.5
        #   visit 1: g01 (k_prepass=2, band [0,5]) -> online k=1 -> residual = (1 - 2) / 4 = -0.25
        #   visit 2: g02 (k_prepass=3, band [0,5]) -> online k=3 -> residual = (3 - 3) / 4 = 0.0
        #   visit 3: g00 (2nd visit)               -> online k=1 -> residual = (1 - 1) / 4 = 0.0
        #   visit 4: g01 (2nd visit)               -> online k=1 -> residual = (1 - 2) / 4 = -0.25
        #   visit 5: g00 (if 5 visits; with 6 visits: visit 5 is g02, let's do steps=5, groups_per_update=1 -> 5 visits!)
        # With steps=5, groups_per_update=1, prompts=3 (5 * 1 > 3):
        #   visit 0 (g00): k_pre=1, k_on=3 -> res = (3-1)/4 = +0.5
        #   visit 1 (g01): k_pre=3, k_on=1 -> res = (1-3)/4 = -0.5
        #   visit 2 (g02): k_pre=2, k_on=2 -> res = (2-2)/4 = 0.0
        #   visit 3 (g00): k_pre=1, k_on=1 -> res = (1-1)/4 = 0.0
        #   visit 4 (g01): k_pre=3, k_on=1 -> res = (1-3)/4 = -0.5
        # Prompt mean residuals:
        #   g00: mean([+0.5, 0.0]) = +0.25 (visited twice)
        #   g01: mean([-0.5, -0.5]) = -0.5 (visited twice)
        #   g02: mean([0.0]) = 0.0 (visited once)
        # Band "[0,5]" mean across the 3 prompts (each counted once):
        #   mean([0.25, -0.5, 0.0]) = -0.25 / 3 = -0.083333
        # (If pooled raw over all 5 visits instead of per-prompt, it would be (0.5 - 0.5 + 0 + 0 - 0.5)/5 = -0.1!)
        # (If divided by literal 8.0 instead of g=4, residuals would be halved!)
        k_prepass_map = {"g00": 1, "g01": 3, "g02": 2}
        visit_k_online = [3, 1, 2, 1, 1]
        baseline_path = tmp_path / "wrap_baseline.jsonl"
        baseline_sha = _write_fake_prepass(
            baseline_path,
            ["g00", "g01", "g02"],
            k_fn=lambda idx: k_prepass_map[f"g{idx:02d}"],
            g=4,
        )

        tok = s2.DryTokenizer()
        fake_cuda = _FakeCuda()
        visit_idx = {"n": 0}

        def scripted_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            k_on = visit_k_online[visit_idx["n"]]
            visit_idx["n"] += 1
            p_ids = tok.encode(prompt_text)
            ok_comp = tok.encode(_reply("supported", "alpha beta")) + [tok.eos_token_id]
            bad_comp = tok.encode(_reply("unsupported", "alpha beta")) + [tok.eos_token_id]
            comps = [ok_comp] * k_on + [bad_comp] * (g_cnt - k_on)
            return p_ids, comps, [False] * g_cnt

        class _FakeTrainModel:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def parameters(self):
                return [self._p]

            def __call__(self, input_ids, attention_mask=None, **kwargs):
                logits = (
                    torch.zeros((input_ids.shape[0], input_ids.shape[1], tok.vocab_size), dtype=torch.float32)
                    + self._p * 0.0
                )
                return Namespace(logits=logits)

        def build_fake(_args_ns):
            return {
                "model": _FakeTrainModel(),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": True},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": scripted_rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        ap = s2.build_parser()
        a = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "wrap_out"),
                "--steps",
                "5",
                "--groups-per-update",
                "1",
                "--prompts",
                "3",
                "--g",
                "4",
                "--max-new",
                "32",
                "--prepass-baseline",
                str(baseline_path),
                "--prepass-baseline-sha256",
                baseline_sha,
            ]
        )
        assert s2.execute(a, build=build_fake, rig=_FakeRig(a)) == 1

        group_rows = [
            json.loads(line)
            for line in (a.out / "s2-hf-groups.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(group_rows) == 5
        assert [gr["residual"] for gr in group_rows] == [0.5, -0.5, 0.0, 0.0, -0.5]

        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt["readout"]["mean_residual_by_k_band"] == {
            "[0,5]": -0.083333,
            "[6,7]": None,
            "8": None,
            "n_prompts": {"[0,5]": 3, "[6,7]": 0, "8": 0},
            "prompt_visits_max": 2,
        }

    def test_prepass_baseline_refusals_on_sha_mismatch_and_missing_id(self, tmp_path):
        import hashlib

        gold_dir = _gold(tmp_path / "gold")
        baseline_path = tmp_path / "baseline.jsonl"
        sha = _write_fake_prepass(baseline_path, [f"g{i:02d}" for i in range(4)], k_fn=lambda idx: 2, g=2)
        ap = s2.build_parser()

        # 1. SHA-256 mismatch refuses
        a_bad_sha = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "bad_sha"),
                "--dry",
                "--prompt-lengths",
                "--prompts",
                "4",
                "--g",
                "2",
                "--prepass-baseline",
                str(baseline_path),
                "--prepass-baseline-sha256",
                "0" * 64,
            ]
        )
        with pytest.raises(SystemExit, match="prepass_baseline_sha_mismatch"):
            s2.execute(a_bad_sha, build=s2.dry_bundle)

        # 2. Missing prompt ID refuses
        missing_id_path = tmp_path / "missing_id.jsonl"
        missing_id_sha = _write_fake_prepass(missing_id_path, ["g00", "g01", "g02"], k_fn=lambda idx: 2, g=2)
        a_missing_id = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "missing_id"),
                "--dry",
                "--prompt-lengths",
                "--prompts",
                "4",
                "--g",
                "2",
                "--prepass-baseline",
                str(missing_id_path),
                "--prepass-baseline-sha256",
                missing_id_sha,
            ]
        )
        with pytest.raises(SystemExit, match="prepass_baseline_missing_id"):
            s2.execute(a_missing_id, build=s2.dry_bundle)

        # 3. Missing companion flag or missing file refuses
        a_no_sha = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "no_sha"),
                "--dry",
                "--prompt-lengths",
                "--prompts",
                "4",
                "--g",
                "2",
                "--prepass-baseline",
                str(baseline_path),
            ]
        )
        with pytest.raises(SystemExit, match="prepass_baseline_missing_arg"):
            s2.execute(a_no_sha, build=s2.dry_bundle)

        a_no_file = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "no_file"),
                "--dry",
                "--prompt-lengths",
                "--prompts",
                "4",
                "--g",
                "2",
                "--prepass-baseline",
                str(tmp_path / "does_not_exist.jsonl"),
                "--prepass-baseline-sha256",
                sha,
            ]
        )
        with pytest.raises(SystemExit, match="prepass_baseline_missing_file"):
            s2.execute(a_no_file, build=s2.dry_bundle)

        # 4. Baseline g mismatch with run's --g refuses with prepass_baseline_g_mismatch
        with pytest.raises(SystemExit, match="prepass_baseline_g_mismatch"):
            s2.load_prepass_baseline(baseline_path, sha, ["g00", "g01", "g02", "g03"], g=8)

        # 5. Baseline k_correct out of range 0..g or non-int (including bool) refuses with prepass_baseline_k_out_of_range
        for idx_bad, bad_k in enumerate([True, False, "2", 1.5, -1, 5]):
            bad_k_path = tmp_path / f"bad_k_{idx_bad}.jsonl"
            rec_line = json.dumps({"id": "g00", "g": 4, "k_correct": bad_k}) + "\n"
            bad_k_path.write_text(rec_line, encoding="utf-8")
            bad_k_sha = hashlib.sha256(bad_k_path.read_bytes()).hexdigest()
            with pytest.raises(SystemExit, match="prepass_baseline_k_out_of_range"):
                s2.load_prepass_baseline(bad_k_path, bad_k_sha, ["g00"], g=4)


class TestLogSdpaBackend:
    @needs_torch
    def test_log_sdpa_backend_records_allowed_backend_on_cpu_and_unknown_with_reason(
        self, tmp_path, monkeypatch
    ):
        from contextlib import nullcontext

        import torch
        import torch.nn.functional as F
        from torch.nn.attention import SDPBackend, sdpa_kernel

        gold_dir = _gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        fake_cuda = _FakeCuda()
        allowed = {"flash", "efficient", "math", "cudnn", "unknown"}

        def fake_attention_call(requires_grad: bool = False):
            q = torch.randn(1, 2, 4, 16, requires_grad=requires_grad)
            return F.scaled_dot_product_attention(q, q, q, is_causal=True)

        class _FakeSdpaModel:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def parameters(self):
                return [self._p]

            def __call__(self, input_ids, attention_mask=None, **kwargs):
                attn_out = fake_attention_call(requires_grad=torch.is_grad_enabled())
                logits = (
                    torch.zeros((input_ids.shape[0], input_ids.shape[1], tok.vocab_size), dtype=torch.float32)
                    + self._p * 0.0
                    + attn_out.sum() * 0.0
                )
                return Namespace(logits=logits)

        def fake_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            with sdpa_kernel(SDPBackend.MATH):
                _ = fake_attention_call(requires_grad=False)
            p_ids = tok.encode(prompt_text)
            comp = tok.encode(_reply("supported", "alpha beta")) + [tok.eos_token_id]
            return p_ids, [comp for _ in range(g_cnt)], [False] * g_cnt

        def fake_mode(training: bool) -> None:
            if not training:
                _ = fake_attention_call(requires_grad=False)

        def build_fake(_args_ns):
            return {
                "model": _FakeSdpaModel(),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": fake_mode,
                "state": lambda: {"grad_ckpt": True},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": fake_rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 1000,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        before_sdp_flags = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
        )

        ap = s2.build_parser()
        a = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "sdpa_out"),
                "--steps",
                "1",
                "--prompts",
                "2",
                "--g",
                "2",
                "--max-new",
                "16",
                "--log-sdpa-backend",
            ]
        )
        assert a.log_sdpa_backend is True
        assert s2.active_flags(a)["log_sdpa_backend"] is True
        assert s2.execute(a, build=build_fake, rig=_FakeRig(a)) == 1

        after_sdp_flags = (
            torch.backends.cuda.flash_sdp_enabled(),
            torch.backends.cuda.mem_efficient_sdp_enabled(),
            torch.backends.cuda.math_sdp_enabled(),
            torch.backends.cuda.cudnn_sdp_enabled(),
        )
        assert before_sdp_flags == after_sdp_flags

        phase_rows = [
            json.loads(line)
            for line in (a.out / "s2-hf-phases.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        target_phases = {
            "pre_rollout",
            "post_rollout",
            "post_ref",
            "post_train_forward",
            "post_train_backward",
        }
        seen_target_phases = {r["phase"] for r in phase_rows if r["phase"] in target_phases}
        assert seen_target_phases == target_phases
        step_end_rows = [r for r in phase_rows if r["phase"] == "step_end"]
        assert len(step_end_rows) >= 1
        for row in phase_rows:
            assert "sdpa_backend" in row
            assert row["sdpa_backend"] in allowed
            if row["phase"] == "post_rollout":
                assert row["sdpa_backend"] == "math"
            elif row["phase"] in ("pre_rollout", "post_ref", "post_train_forward", "post_train_backward"):
                assert row["sdpa_backend"] in {"flash", "efficient", "math", "cudnn"}
            elif row["phase"] == "step_end":
                assert row["sdpa_backend"] == "unknown"
                assert isinstance(row.get("sdpa_backend_reason"), str) and row["sdpa_backend_reason"]
            elif row["sdpa_backend"] == "unknown":
                assert isinstance(row.get("sdpa_backend_reason"), str) and row["sdpa_backend_reason"]


class TestDefaultPathLiteralExpectation:
    @needs_torch
    def test_default_path_with_no_new_flags_matches_literal_base_expectations(self, tmp_path, monkeypatch):
        from contextlib import nullcontext

        import torch

        monkeypatch.setattr(s2.time, "time", lambda: 1000.0)
        monkeypatch.setattr(s2.time, "monotonic", lambda: 500.0)
        monkeypatch.setitem(s2.PhaseLogger.__init__.__kwdefaults__, "now", lambda: 500.0)

        gold_dir = _gold(tmp_path / "gold")
        _, gold_hashes = s2.read_gold(gold_dir)
        ap = s2.build_parser()
        a = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "default_out"),
                "--steps",
                "1",
                "--prompts",
                "2",
                "--g",
                "2",
                "--max-new",
                "8",
            ]
        )
        assert a.prepass_seed == 0
        assert a.sign_diagnostics is False
        assert a.prepass_baseline is None
        assert a.prepass_baseline_sha256 is None
        assert a.log_sdpa_backend is False
        assert s2.active_flags(a) == {}

        tok = s2.DryTokenizer()
        fake_cuda = _FakeCuda()

        class _FakeModel:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def parameters(self):
                return [self._p]

            def __call__(self, input_ids, attention_mask=None, **kwargs):
                logits = (
                    torch.zeros((input_ids.shape[0], input_ids.shape[1], tok.vocab_size), dtype=torch.float32)
                    + self._p * 0.0
                )
                return Namespace(logits=logits)

        def fake_rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            p_ids = tok.encode(prompt_text)
            c0 = tok.encode(_reply("supported", "alpha beta")) + [tok.eos_token_id]
            c1 = tok.encode(_reply("unsupported", "alpha beta")) + [tok.eos_token_id]
            return p_ids, [c0, c1], [False, False]

        def build_fake(_args_ns):
            return {
                "model": _FakeModel(),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": True},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": fake_rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 100,
                "revision": "rev123",
                "llama": None,
                "dtype_census": None,
            }

        assert s2.execute(a, build=build_fake, rig=_FakeRig(a)) == 1  # cut run -> S2-LIMITED

        # 1. Literal steps rows expectation
        step_rows = [
            json.loads(line)
            for line in (a.out / "s2-hf-steps.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        expected_step_rows = [
            {
                "step": 0,
                "groups": 2,
                "active_groups": 2,
                "zero_variance_groups": 0,
                "tokens": 520,
                "rollout_s": 0.0,
                "train_s": 0.0,
                "step_s": 0.0,
                "mean_reward": 0.025,
                "loss": 0.015385,
                "peak_alloc_mib": 110,
                "grad_ckpt": True,
                "grad_ckpt_rollout": True,
            }
        ]
        assert step_rows == expected_step_rows

        # 2. Literal groups rows expectation
        group_rows = [
            json.loads(line)
            for line in (a.out / "s2-hf-groups.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        expected_group_rows = [
            {
                "step": 0,
                "group": 0,
                "prompt_id": "g00",
                "prompt_tokens": 438,
                "k_correct": 1,
                "g": 2,
                "reward_std": 1.025,
                "verdict_reward_std": 1.0,
                "truncations": 0,
                "mean_thinking_tokens": 5.0,
                "zero_variance_label": None,
            },
            {
                "step": 0,
                "group": 1,
                "prompt_id": "g01",
                "prompt_tokens": 438,
                "k_correct": 1,
                "g": 2,
                "reward_std": 1.025,
                "verdict_reward_std": 1.0,
                "truncations": 0,
                "mean_thinking_tokens": 5.0,
                "zero_variance_label": None,
            },
        ]
        assert group_rows == expected_group_rows

        # 3. Literal phase rows expectation
        phase_rows = [
            json.loads(line)
            for line in (a.out / "s2-hf-phases.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        zero_comp = {"min": 0, "max": 0, "sum": 0}
        full_comp = {"min": 128, "max": 132, "sum": 260}
        mem_first = {
            "memory_allocated": 104857600,
            "memory_reserved": 134217728,
            "max_memory_allocated": 115343360,
            "max_memory_reserved": 146800640,
        }
        mem_rest = {
            "memory_allocated": 104857600,
            "memory_reserved": 134217728,
            "max_memory_allocated": 104857600,
            "max_memory_reserved": 134217728,
        }
        expected_phase_rows = [
            {
                "step": 0,
                "group": 0,
                "phase": "pre_rollout",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": zero_comp,
                **mem_first,
            },
            {
                "step": 0,
                "group": 0,
                "phase": "post_rollout",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": full_comp,
                **mem_rest,
            },
            {
                "step": 0,
                "group": 0,
                "phase": "post_ref",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": full_comp,
                **mem_rest,
            },
            {
                "step": 0,
                "group": 0,
                "phase": "post_train_forward",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": full_comp,
                **mem_rest,
            },
            {
                "step": 0,
                "group": 0,
                "phase": "post_train_backward",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": full_comp,
                **mem_rest,
            },
            {
                "step": 0,
                "group": 1,
                "phase": "pre_rollout",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": zero_comp,
                **mem_rest,
            },
            {
                "step": 0,
                "group": 1,
                "phase": "post_rollout",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": full_comp,
                **mem_rest,
            },
            {
                "step": 0,
                "group": 1,
                "phase": "post_ref",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": full_comp,
                **mem_rest,
            },
            {
                "step": 0,
                "group": 1,
                "phase": "post_train_forward",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": full_comp,
                **mem_rest,
            },
            {
                "step": 0,
                "group": 1,
                "phase": "post_train_backward",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": full_comp,
                **mem_rest,
            },
            {
                "step": 0,
                "group": 1,
                "phase": "step_end",
                "monotonic_s": 500.0,
                "prompt_tokens": 438,
                "completion_tokens": full_comp,
                **mem_rest,
            },
        ]
        assert phase_rows == expected_phase_rows

        # 4. Literal readout and receipt expectation
        receipt = json.loads((a.out / "s2-hf.json").read_text(encoding="utf-8"))
        expected_readout = {
            "steps_requested": 1,
            "steps_completed": 1,
            "peak_vram_mib_sampler": 2500,
            "sampler_samples": len(receipt["readout"]["vram_mib_sampler_series"]),
            "vram_mib_sampler_series": [2500] * len(receipt["readout"]["vram_mib_sampler_series"]),
            "peak_vram_mib_torch": 110,
            "peak_reserved_mib_torch": 140,
            "fits": True,
            "oom": False,
            "rollout_tok_s": None,
            "step_s": {"p50": 0.0, "max": 0.0},
            "wall_s": 0.0,
            "zero_variance_groups": 0,
            "active_groups": 2,
            "groups": 2,
            "complete": False,
            "pass": False,
            "eval_summary": {
                "supported_pred_to_prior_ratio": 0.5,
                "p_cannot_tell_given_gold_supported": 0.0,
                "gold_supported_groups_with_positive_cannot_tell_advantage": 0.0,
                "false_accepts": 0,
                "false_rejects": 2,
                "miss_waterfall": {
                    "rule_2": 0,
                    "no_minimum": 0,
                    "case": 0,
                    "punctuation": 0,
                    "ellipsis_split": 0,
                    "joined_passages": 0,
                    "fuzzy": 0,
                    "unmatched": 0,
                },
                "quote_relevance_share": 0.0,
            },
        }
        assert receipt["readout"] == expected_readout

        guard_rec = s2.check_exclusive(s2.dry_smi, source="fake preflight")
        expected_env = s2.environment(
            gpu={"name": "NVIDIA GeForce RTX 5090", "driver": "595.71"},
            dtype="bfloat16 compute on a 4-bit base",
            batch=2,
            exclusive=True,
            cuda="13.0",
            guard=guard_rec,
        )
        expected_receipt = {
            "test": "S2",
            "path": "hf",
            "dry": False,
            "model": "Qwen/Qwen3-8B",
            "revision_registered": "b968826d9c46",
            "revision_resolved": "rev123",
            "registered": {
                "steps": 20,
                "prompts": 32,
                "groups_per_step": 2,
                "G": 8,
                "temperature": 0.7,
                "lora": {"r": 16, "alpha": 32, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
                "learning_rate": 1e-5,
                "beta": 0.04,
                "epsilon": 0.2,
                "max_new_tokens": 2048,
            },
            "run": {"steps": 1, "prompts": 2, "g": 2, "max_new": 8, "dry": False},
            "rnd_commit": "e48cc90c",
            "gold_files_sha256": gold_hashes,
            "prompt_split": "tune",
            "prompt_ids": ["g00", "g01"],
            "environment": expected_env,
            "stack_versions": {k: s2.package_version(k) for k in s2.FROZEN_STACK_VERSIONS},
            "pip_freeze_sha256": None,
            "llama_cpp": None,
            "pass_line": {"vram_line_mib": 31200, "steps": 20, "oom": False},
            "stopped": None,
            "readout": expected_readout,
            "verdict": "S2-LIMITED",
            "save_all": True,
            "save_all_sha256": s2.hashlib.sha256((a.out / "s2-hf-save-all.jsonl.gz").read_bytes()).hexdigest(),
        }
        expected_receipt_bytes = json.dumps(expected_receipt, indent=1).encode("utf-8")
        assert (a.out / "s2-hf.json").read_bytes() == expected_receipt_bytes


# ---------------------------------------------------------------------------
# 2026-11-05 Part A: Save-All
# ---------------------------------------------------------------------------


class TestSaveAll:
    def test_save_all_is_on_by_default_and_refuses_no_save_all_without_reason(self, tmp_path):
        gold_dir = _gold(tmp_path / "gold")
        ap = s2.build_parser()
        a_def = ap.parse_args(
            ["--gold", str(gold_dir), "--rnd-commit", "e48cc90c", "--out", str(tmp_path / "out_def")]
        )
        assert a_def.save_all is True
        assert a_def.save_all_off_reason is None
        assert s2.validate_save_all_args(a_def) == (True, None)
        assert "save_all" not in s2.active_flags(a_def)

        a_no_reason = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "out_no_reason"),
                "--no-save-all",
            ]
        )
        assert a_no_reason.save_all is False
        with pytest.raises(SystemExit, match="save_all_off_reason_required"):
            s2.validate_save_all_args(a_no_reason)
        with pytest.raises(SystemExit, match="save_all_off_reason_required"):
            s2.execute(
                a_no_reason,
                build=lambda _: (_ for _ in ()).throw(AssertionError("build must not run")),
            )

        a_blank_reason = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "out_blank"),
                "--no-save-all",
                "--save-all-off-reason",
                "   ",
            ]
        )
        with pytest.raises(SystemExit, match="save_all_off_reason_required"):
            s2.validate_save_all_args(a_blank_reason)

        a_unused_reason = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "out_unused"),
                "--save-all-off-reason",
                "accidental reason while save-all is on",
            ]
        )
        with pytest.raises(SystemExit, match="save_all_off_reason_unused"):
            s2.validate_save_all_args(a_unused_reason)

        a_valid_off = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "out_valid_off"),
                "--no-save-all",
                "--save-all-off-reason",
                "  write-only equivalence test  ",
                "--pass-id",
                "custom_pass_01",
            ]
        )
        assert s2.validate_save_all_args(a_valid_off) == (False, "write-only equivalence test")
        flags = s2.active_flags(a_valid_off)
        assert flags["save_all"] is False
        assert flags["save_all_off_reason"] == "write-only equivalence test"
        assert flags["pass_id"] == "custom_pass_01"

        # Also test read_save_all on plain .jsonl as well as .jsonl.gz
        plain_path = tmp_path / "plain.jsonl"
        plain_path.write_text('{"item_id": "x1", "kind": "pass"}\n', encoding="utf-8")
        assert s2.read_save_all(plain_path) == [{"item_id": "x1", "kind": "pass"}]

    @needs_torch
    def test_save_all_pass_and_training_write_only_equivalence_and_sha256(self, tmp_path, monkeypatch):
        import hashlib
        from contextlib import nullcontext

        import torch

        monkeypatch.setattr(s2.time, "time", lambda: 1000.0)
        monkeypatch.setattr(s2.time, "monotonic", lambda: 500.0)
        monkeypatch.setitem(s2.PhaseLogger.__init__.__kwdefaults__, "now", lambda: 500.0)

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()

        def make_pass_bundle(_args_ns):
            fake_cuda = _FakeCuda()

            def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
                p_ids = tok.encode(prompt_text)
                c_ok = tok.encode(_reply("supported", "alpha beta gamma", think="careful thought")) + [
                    tok.eos_token_id
                ]
                c_mal = tok.encode("<think>bad json</think>{not valid json") + [tok.eos_token_id]
                c_trunc = tok.encode("<think>unclosed thinking without eos")
                comps = [c_ok, c_mal, c_trunc][:g_cnt]
                truncs = [False, False, True][:g_cnt]
                return p_ids, comps, truncs

            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 500,
                "revision": "rev_pass_1",
                "llama": None,
                "dtype_census": None,
                "dry_prepass_cap": 4,
            }

        ap = s2.build_parser()
        # 1. Pass run with save-all ON (including dry resume across two halves) vs save-all OFF
        a_pass_on = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "pass_on"),
                "--rollout-only",
                "--dry",
                "--g",
                "3",
                "--max-new",
                "32",
                "--quote-reward",
                "v2",
                "--pass-id",
                "passA_seed0",
            ]
        )
        a_pass_off = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "pass_off"),
                "--rollout-only",
                "--dry",
                "--g",
                "3",
                "--max-new",
                "32",
                "--quote-reward",
                "v2",
                "--pass-id",
                "passA_seed0",
                "--no-save-all",
                "--save-all-off-reason",
                "testing write-only equivalence on pass",
            ]
        )
        assert s2.execute(a_pass_on, build=make_pass_bundle) == 0
        assert s2.execute(a_pass_off, build=make_pass_bundle) == 0

        sa_pass_path = a_pass_on.out / "s2-hf-save-all.jsonl.gz"
        assert sa_pass_path.is_file()
        assert not (a_pass_off.out / "s2-hf-save-all.jsonl.gz").exists()

        rec_pass_on = json.loads((a_pass_on.out / "s2-hf.json").read_text(encoding="utf-8"))
        rec_pass_off = json.loads((a_pass_off.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert rec_pass_on["save_all"] is True
        assert (
            rec_pass_on["save_all_sha256"]
            == hashlib.sha256(sa_pass_path.read_bytes()).hexdigest()
        )
        assert "save_all_off_reason" not in rec_pass_on
        assert rec_pass_off["save_all"] is False
        assert rec_pass_off["save_all_sha256"] is None
        assert rec_pass_off["save_all_off_reason"] == "testing write-only equivalence on pass"

        saved_pass_rows = s2.read_save_all(sa_pass_path)
        assert len(saved_pass_rows) == 4 * 3  # 4 items * g=3 completions
        for idx, row_sa in enumerate(saved_pass_rows):
            assert row_sa["kind"] == "pass"
            assert row_sa["pass_id"] == "passA_seed0"
            assert row_sa["pass_seed"] == 0
            assert row_sa["sample_index"] == idx % 3
            assert row_sa["seed"] == s2.item_seed(row_sa["item_id"], 0)
            if idx % 3 == 0:
                assert row_sa["parse_status"] == "ok"
                assert row_sa["thinking"] == "careful thought"
                assert row_sa["answer"].startswith("{")
                assert row_sa["verdict"] == "supported"
                assert row_sa["truncated"] is False
            elif idx % 3 == 1:
                assert row_sa["parse_status"] == "malformed"
                assert row_sa["verdict"] is None
                assert row_sa["truncated"] is False
            else:
                assert row_sa["parse_status"] == "truncated"
                assert row_sa["verdict"] is None
                assert row_sa["truncated"] is True

        # Byte-identical prepass.jsonl, prepass-summary.json, eval-summary.json
        for fname in (
            "s2-hf-prepass.jsonl",
            "s2-hf-prepass-summary.json",
            "s2-hf-eval-summary.json",
        ):
            assert (a_pass_on.out / fname).read_bytes() == (a_pass_off.out / fname).read_bytes(), fname

        # Resuming a save-all=False directory with save-all=True refuses on header mismatch
        a_mismatch_resume = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "pass_off"),
                "--rollout-only",
                "--dry",
                "--g",
                "3",
                "--max-new",
                "32",
                "--quote-reward",
                "v2",
            ]
        )
        with pytest.raises(SystemExit, match="header mismatch on save_all"):
            s2.execute(a_mismatch_resume, build=make_pass_bundle)

        # 2. Training run with save-all ON vs save-all OFF: byte-identical steps, groups, eval_summary, and receipt
        def make_train_bundle(_args_ns):
            torch.manual_seed(42)
            fake_cuda = _FakeCuda()

            class _TinyModel:
                device = torch.device("cpu")

                def __init__(self) -> None:
                    self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

                def parameters(self):
                    return [self._p]

                def __call__(self, input_ids, attention_mask=None, **kwargs):
                    logits = (
                        torch.zeros(
                            (input_ids.shape[0], input_ids.shape[1], tok.vocab_size), dtype=torch.float32
                        )
                        + self._p * 0.0
                    )
                    return Namespace(logits=logits)

            def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
                p_ids = tok.encode(prompt_text)
                c0 = tok.encode(_reply("supported", "alpha beta gamma", think="step thought")) + [
                    tok.eos_token_id
                ]
                c1 = tok.encode(_reply("unsupported", "alpha beta gamma", think="alt thought")) + [
                    tok.eos_token_id
                ]
                c2 = tok.encode("<think>truncated rollout")
                return p_ids, [c0, c1, c2][:g_cnt], [False, False, True][:g_cnt]

            return {
                "model": _TinyModel(),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": True},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 250,
                "revision": "rev_train_1",
                "llama": None,
                "dtype_census": None,
            }

        a_train_on = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "train_on"),
                "--steps",
                "2",
                "--prompts",
                "4",
                "--g",
                "3",
                "--max-new",
                "32",
                "--quote-reward",
                "v2",
            ]
        )
        a_train_off = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "train_off"),
                "--steps",
                "2",
                "--prompts",
                "4",
                "--g",
                "3",
                "--max-new",
                "32",
                "--quote-reward",
                "v2",
                "--no-save-all",
                "--save-all-off-reason",
                "testing write-only equivalence on training",
            ]
        )
        assert s2.execute(a_train_on, build=make_train_bundle, rig=_FakeRig(a_train_on)) == 1
        assert s2.execute(a_train_off, build=make_train_bundle, rig=_FakeRig(a_train_off)) == 1

        sa_train_path = a_train_on.out / "s2-hf-save-all.jsonl.gz"
        assert sa_train_path.is_file()
        assert not (a_train_off.out / "s2-hf-save-all.jsonl.gz").exists()

        saved_rollouts = s2.read_save_all(sa_train_path)
        assert len(saved_rollouts) == 2 * 2 * 3  # 2 steps * 2 groups/step * 3 samples/group = 12
        for r_item in saved_rollouts:
            assert r_item["kind"] == "rollout"
            assert set(r_item.keys()) == {
                "kind",
                "item_id",
                "seed",
                "step",
                "group",
                "sample_index",
                "text",
                "thinking",
                "answer",
                "parse_status",
                "verdict",
                "reward",
                "advantage",
                "truncated",
            }

        for fname in ("s2-hf-steps.jsonl", "s2-hf-groups.jsonl", "s2-hf-eval-summary.json"):
            assert (a_train_on.out / fname).read_bytes() == (a_train_off.out / fname).read_bytes(), fname

        rec_tr_on = json.loads((a_train_on.out / "s2-hf.json").read_text(encoding="utf-8"))
        rec_tr_off = json.loads((a_train_off.out / "s2-hf.json").read_text(encoding="utf-8"))
        assert rec_tr_on["save_all"] is True
        assert (
            rec_tr_on["save_all_sha256"]
            == hashlib.sha256(sa_train_path.read_bytes()).hexdigest()
        )
        assert rec_tr_off["save_all"] is False
        assert rec_tr_off["save_all_sha256"] is None
        assert rec_tr_off["save_all_off_reason"] == "testing write-only equivalence on training"

        def strip_save_all_fields(doc: dict) -> dict:
            out = {
                k: v
                for k, v in doc.items()
                if k not in ("save_all", "save_all_sha256", "save_all_off_reason")
            }
            out["run"] = {
                k: v for k, v in doc["run"].items() if k not in ("save_all", "save_all_off_reason")
            }
            out["readout"] = dict(doc["readout"])
            if "flags" in out["readout"]:
                out["readout"]["flags"] = {
                    k: v
                    for k, v in out["readout"]["flags"].items()
                    if k not in ("save_all", "save_all_off_reason")
                }
            return out

        assert strip_save_all_fields(rec_tr_on) == strip_save_all_fields(rec_tr_off)
        assert strip_save_all_fields(rec_pass_on) == strip_save_all_fields(rec_pass_off)


class TestLoggingPassAndPostpass:
    def test_pass_mode_validation_and_checkpoint_sha(self, tmp_path: Path) -> None:
        import hashlib

        ap = s2.build_parser()
        gold_dir = _prepass_gold(tmp_path / "gold")

        # Conflicting --logging-pass and --postpass
        a_conflict = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "conflict"),
                "--logging-pass",
                "--postpass",
            ]
        )
        with pytest.raises(SystemExit, match="conflicting_pass_flags"):
            s2.validate_pass_mode_args(a_conflict)

        # --logging-pass cannot take --checkpoint
        ckpt_file = tmp_path / "adapter.bin"
        ckpt_file.write_bytes(b"weights-v1")
        a_log_ckpt = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "log_ckpt"),
                "--logging-pass",
                "--checkpoint",
                str(ckpt_file),
            ]
        )
        with pytest.raises(SystemExit, match="logging_pass_checkpoint_forbidden"):
            s2.validate_pass_mode_args(a_log_ckpt)

        # --logging-pass refuses seed != 1 via --pass-seed or --prepass-seed
        a_log_bad_seed = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "log_bad_seed"),
                "--logging-pass",
                "--pass-seed",
                "2",
            ]
        )
        with pytest.raises(SystemExit, match="logging_pass_seed_must_be_1"):
            s2.validate_pass_mode_args(a_log_bad_seed)

        a_log_bad_pre_seed = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "log_bad_pre_seed"),
                "--logging-pass",
                "--prepass-seed",
                "2",
            ]
        )
        with pytest.raises(SystemExit, match="logging_pass_seed_must_be_1"):
            s2.validate_pass_mode_args(a_log_bad_pre_seed)

        # --postpass refuses reserved seeds 0 and 1
        for bad_seed in ("0", "1"):
            a_post_reserved = ap.parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / f"post_reserved_{bad_seed}"),
                    "--postpass",
                    "--pass-seed",
                    bad_seed,
                ]
            )
            with pytest.raises(SystemExit, match="postpass_reserved_seed"):
                s2.validate_pass_mode_args(a_post_reserved)

        # Missing checkpoint and empty directory checkpoint are refused
        with pytest.raises(SystemExit, match="checkpoint_missing"):
            s2.checkpoint_sha256(tmp_path / "no_such_ckpt")
        empty_ckpt_dir = tmp_path / "empty_ckpt"
        empty_ckpt_dir.mkdir()
        with pytest.raises(SystemExit, match="checkpoint_empty"):
            s2.checkpoint_sha256(empty_ckpt_dir)

        # Directory checkpoint hashing is deterministic
        ckpt_dir = tmp_path / "lora_ckpt"
        (ckpt_dir / "sub").mkdir(parents=True)
        (ckpt_dir / "adapter_config.json").write_text('{"r": 16}\n', encoding="utf-8")
        (ckpt_dir / "sub" / "adapter_model.safetensors").write_bytes(b"safetensor-bytes")
        sha_dir_1 = s2.checkpoint_sha256(ckpt_dir)
        sha_dir_2 = s2.checkpoint_sha256(ckpt_dir)
        assert sha_dir_1 is not None and len(sha_dir_1) == 64 and sha_dir_1 == sha_dir_2

        # File checkpoint hashing
        sha_file = s2.checkpoint_sha256(ckpt_file)
        assert sha_file == hashlib.sha256(b"weights-v1").hexdigest()

        # Valid --postpass with --checkpoint and --pass-seed 2
        a_post_ok = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "post_ok"),
                "--postpass",
                "--checkpoint",
                str(ckpt_dir),
                "--pass-seed",
                "2",
            ]
        )
        is_pm, is_desc, eff_seed, ckpt_sha = s2.validate_pass_mode_args(a_post_ok)
        assert (is_pm, is_desc, eff_seed, ckpt_sha) == (True, False, 2, sha_dir_1)
        assert s2.active_flags(a_post_ok)["postpass"] is True
        assert s2.active_flags(a_post_ok)["prepass_seed"] == 2

    def test_logging_pass_descriptive_guard_and_postpass_truncation_by_band(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import hashlib
        from contextlib import nullcontext

        torch = pytest.importorskip("torch")
        ap = s2.build_parser()
        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        fixed_script_sha = "77" * 32
        monkeypatch.setattr(s2, "script_sha256", lambda: fixed_script_sha)

        # Create a Pass-A baseline JSONL (seed 0, non-descriptive) with items across all 3 bands:
        # t_a, t_b in [0,5] (k=2, 5); t_d, t_k in [6,7] (k=6, 7); t_m, t_z in 8 (k=8, 8)
        pass_a_dir = tmp_path / "pass_a"
        pass_a_dir.mkdir(parents=True)
        pass_a_jsonl = pass_a_dir / "s2-hf-prepass.jsonl"
        pass_a_rows = [
            {"id": "t_a", "g": 8, "k_correct": 2, "pass_seed": 0, "descriptive": False},
            {"id": "t_b", "g": 8, "k_correct": 5, "pass_seed": 0, "descriptive": False},
            {"id": "t_d", "g": 8, "k_correct": 6, "pass_seed": 0, "descriptive": False},
            {"id": "t_k", "g": 8, "k_correct": 7, "pass_seed": 0, "descriptive": False},
            {"id": "t_m", "g": 8, "k_correct": 8, "pass_seed": 0, "descriptive": False},
            {"id": "t_z", "g": 8, "k_correct": 8, "pass_seed": 0, "descriptive": False},
        ]
        pass_a_jsonl.write_text(
            "".join(json.dumps(r) + "\n" for r in pass_a_rows),
            encoding="utf-8",
            newline="\n",
        )
        pass_a_sha = hashlib.sha256(pass_a_jsonl.read_bytes()).hexdigest()

        def make_bundle(_args_ns):
            fake_cuda = _FakeCuda()

            def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
                p_ids = tok.encode(prompt_text)
                # Make t_a (band [0,5]) have 2 truncations, t_d (band [6,7]) have 1 truncation,
                # and t_m (band 8) have 3 truncations.
                if "Claim ta" in prompt_text:
                    n_trunc = 2
                elif "Claim td" in prompt_text:
                    n_trunc = 1
                elif "Claim tm" in prompt_text:
                    n_trunc = 3
                else:
                    n_trunc = 0
                comps = []
                truncs = []
                for idx in range(g_cnt):
                    if idx < n_trunc:
                        comps.append(tok.encode("<think>unclosed"))
                        truncs.append(True)
                    else:
                        comps.append(
                            tok.encode(_reply("supported", "alpha beta gamma", think="ok"))
                            + [tok.eos_token_id]
                        )
                        truncs.append(False)
                return p_ids, comps, truncs

            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 100,
                "revision": "rev_pass_test",
                "llama": None,
                "dtype_census": None,
                "adapter_loaded": getattr(_args_ns, "checkpoint", None) is not None,
            }

        # 1. Run starting-policy --logging-pass (uses seed 1, sets descriptive: true)
        log_dir = tmp_path / "logging_pass"
        a_log = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(log_dir),
                "--logging-pass",
                "--path",
                "hf",
                "--revision",
                s2.STUDENT_FULL_REVISION,
                "--bf16-restore",
                "--logprob-microbatch",
                "2",
                "--attention-mask",
                "--alloc-conf",
                "expandable",
                "--empty-cache-phases",
                "--g",
                "8",
                "--max-new",
                "2048",
                "--quote-reward",
                "v2",
                "--replay-check-off-reason",
                "synthetic truncation band test",
                "--pass-a-jsonl",
                str(pass_a_jsonl),
                "--pass-a-sha256",
                pass_a_sha,
            ]
        )
        assert s2.execute(a_log, build=make_bundle, rig=_FakeRig(a_log)) == 0

        log_hdr = json.loads((log_dir / "s2-hf-prepass-header.json").read_text(encoding="utf-8"))
        assert log_hdr["seed"] == 1
        assert log_hdr["descriptive"] is True
        assert log_hdr["pass_a_sha256"] == pass_a_sha

        log_summary = json.loads((log_dir / "s2-hf-prepass-summary.json").read_text(encoding="utf-8"))
        assert log_summary["descriptive"] is True
        assert log_summary["total_truncations"] == 6
        assert log_summary["truncations_by_band"] == {"[0,5]": 2, "[6,7]": 1, "8": 3}

        log_jsonl_path = log_dir / "s2-hf-prepass.jsonl"
        log_recs = [
            json.loads(line) for line in log_jsonl_path.read_text(encoding="utf-8").splitlines() if line
        ]
        assert len(log_recs) == 6
        assert all(r.get("descriptive") is True for r in log_recs)
        by_id = {r["id"]: r for r in log_recs}
        assert by_id["t_a"]["pass_a_k"] == 2 and by_id["t_a"]["pass_a_band"] == "[0,5]"
        assert by_id["t_d"]["pass_a_k"] == 6 and by_id["t_d"]["pass_a_band"] == "[6,7]"
        assert by_id["t_m"]["pass_a_k"] == 8 and by_id["t_m"]["pass_a_band"] == "8"

        sa_log_recs = s2.read_save_all(log_dir / "s2-hf-save-all.jsonl.gz")
        assert len(sa_log_recs) == 6 * 8
        assert all(r["pass_id"] == "logging_seed1" and r["pass_seed"] == 1 for r in sa_log_recs)

        # 2. Refusal when passing the descriptive logging pass as --prepass-baseline
        log_sha = hashlib.sha256(log_jsonl_path.read_bytes()).hexdigest()
        with pytest.raises(SystemExit, match="prepass_baseline_descriptive"):
            s2.load_prepass_baseline(log_jsonl_path, log_sha, ["t_a"], g=8)

        # Even if the companion header is removed, the per-record descriptive=true guard still refuses
        copied_jsonl = tmp_path / "copied_descriptive.jsonl"
        copied_jsonl.write_bytes(log_jsonl_path.read_bytes())
        with pytest.raises(SystemExit, match="prepass_baseline_descriptive"):
            s2.load_prepass_baseline(copied_jsonl, log_sha, ["t_a"], g=8)

        # And if a record has descriptive omitted but companion *-header.json has descriptive=true, it refuses
        hdr_only_dir = tmp_path / "hdr_only_desc"
        hdr_only_dir.mkdir()
        hdr_only_jsonl = hdr_only_dir / "s2-hf-prepass.jsonl"
        hdr_only_jsonl.write_bytes(pass_a_jsonl.read_bytes())
        (hdr_only_dir / "s2-hf-prepass-header.json").write_text(
            json.dumps({"descriptive": True}), encoding="utf-8"
        )
        with pytest.raises(SystemExit, match="prepass_baseline_descriptive"):
            s2.load_prepass_baseline(hdr_only_jsonl, pass_a_sha, ["t_a"], g=8)

        # 3. Run --postpass for a trained checkpoint with seed 2 and verify header, summary, and resume
        ckpt_dir = tmp_path / "trained_ckpt"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapter_model.bin").write_bytes(b"trained-adapter-bytes")
        expected_ckpt_sha = s2.checkpoint_sha256(ckpt_dir)

        post_dir = tmp_path / "postpass_seed2"
        a_post = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(post_dir),
                "--postpass",
                "--checkpoint",
                str(ckpt_dir),
                "--pass-seed",
                "2",
                "--g",
                "8",
                "--max-new",
                "2048",
                "--quote-reward",
                "v2",
                "--pass-a-jsonl",
                str(pass_a_jsonl),
                "--pass-a-sha256",
                pass_a_sha,
            ]
        )
        assert s2.execute(a_post, build=make_bundle, rig=_FakeRig(a_post)) == 0

        post_hdr = json.loads((post_dir / "s2-hf-prepass-header.json").read_text(encoding="utf-8"))
        assert post_hdr["seed"] == 2
        assert post_hdr["descriptive"] is False
        assert post_hdr["checkpoint_sha256"] == expected_ckpt_sha
        assert post_hdr["pass_a_sha256"] == pass_a_sha

        post_summary = json.loads((post_dir / "s2-hf-prepass-summary.json").read_text(encoding="utf-8"))
        assert "descriptive" not in post_summary
        assert post_summary["checkpoint_sha256"] == expected_ckpt_sha
        assert post_summary["truncations_by_band"] == {"[0,5]": 2, "[6,7]": 1, "8": 3}

        sa_post_recs = s2.read_save_all(post_dir / "s2-hf-save-all.jsonl.gz")
        assert len(sa_post_recs) == 6 * 8
        assert all(r["pass_id"] == "postpass_seed2" and r["pass_seed"] == 2 for r in sa_post_recs)


class TestTrainingSideRulesPartC:
    def test_frozen_env_check_and_refusals(self, tmp_path: Path) -> None:
        import hashlib

        # Build a synthetic pip-freeze text and verify check_frozen_env against its sha256 and versions
        freeze_lines = [
            "accelerate==1.15.0\n",
            "bitsandbytes==0.50.2\n",
            "peft==0.21.2\n",
            "torch==2.14.1+cu130\n",
            "transformers==5.19.0\n",
        ]
        pf_path = tmp_path / "pip-freeze.txt"
        pf_bytes = "".join(freeze_lines).encode("utf-8")
        pf_path.write_bytes(pf_bytes)
        pf_sha = hashlib.sha256(pf_bytes).hexdigest()

        res = s2.check_frozen_env(
            pf_path,
            expected_sha256=pf_sha,
            installed_versions=s2.FROZEN_STACK_VERSIONS,
            require_exact_versions=True,
        )
        assert res["pip_freeze_sha256"] == pf_sha
        assert res["stack_versions"] == s2.FROZEN_STACK_VERSIONS
        assert (
            s2.FROZEN_PIP_FREEZE_SHA256
            == "27c98212c6cd9bd0303b089e76bed0f167cfeeec0eedd0cc9b0baf076c055324"
        )

        # Missing file -> frozen_env_missing_file
        with pytest.raises(SystemExit, match="frozen_env_missing_file"):
            s2.check_frozen_env(tmp_path / "missing-freeze.txt")

        # SHA mismatch -> frozen_env_sha_mismatch
        with pytest.raises(SystemExit, match="frozen_env_sha_mismatch"):
            s2.check_frozen_env(pf_path, expected_sha256="00" * 32)

        # Version mismatch inside pip-freeze -> frozen_env_version_mismatch
        bad_pf = tmp_path / "bad-freeze.txt"
        bad_bytes = (
            b"accelerate==1.15.0\nbitsandbytes==0.50.2\npeft==0.21.2\n"
            b"torch==2.6.0\ntransformers==5.19.0\n"
        )
        bad_pf.write_bytes(bad_bytes)
        bad_sha = hashlib.sha256(bad_bytes).hexdigest()
        with pytest.raises(SystemExit, match="frozen_env_version_mismatch"):
            s2.check_frozen_env(bad_pf, expected_sha256=bad_sha)

        # Installed version mismatch when require_exact_versions=True -> frozen_env_version_mismatch
        with pytest.raises(SystemExit, match="frozen_env_version_mismatch"):
            s2.check_frozen_env(
                installed_versions={
                    "torch": "2.14.1+cu130",
                    "transformers": "4.50.0",
                    "bitsandbytes": "0.50.2",
                    "peft": "0.21.2",
                    "accelerate": "1.15.0",
                },
                require_exact_versions=True,
            )

    def test_stratified_20_percent_holdout_and_class_stratified_epoch_draw(self, tmp_path: Path) -> None:
        import hashlib
        import random
        from collections import Counter

        from s2_draw import (
            build_class_stratified_epoch,
            build_class_stratified_schedule,
            draw_prompts,
            split_stratified_holdout,
        )

        # Build a 493-item synthetic tune split with the exact 9-cell Pass-A band x gold class counts:
        # ('[0,5]', 'cannot_tell'): 11, ('[0,5]', 'supported'): 37, ('[0,5]', 'unsupported'): 57
        # ('[6,7]', 'cannot_tell'): 6,  ('[6,7]', 'supported'): 53, ('[6,7]', 'unsupported'): 59
        # ('8', 'cannot_tell'): 17,     ('8', 'supported'): 139,    ('8', 'unsupported'): 114
        cell_specs = [
            ("[0,5]", "cannot_tell", 11, 3),
            ("[0,5]", "supported", 37, 4),
            ("[0,5]", "unsupported", 57, 5),
            ("[6,7]", "cannot_tell", 6, 6),
            ("[6,7]", "supported", 53, 7),
            ("[6,7]", "unsupported", 59, 6),
            ("8", "cannot_tell", 17, 8),
            ("8", "supported", 139, 8),
            ("8", "unsupported", 114, 8),
        ]
        rows_493: list[dict] = []
        prepass_recs: list[dict] = []
        idx = 0
        for _band, gold_cls, count, k_val in cell_specs:
            for _ in range(count):
                cid = f"item_{idx:03d}"
                rows_493.append(
                    {
                        "id": cid,
                        "split": "tune",
                        "label": gold_cls,
                        "claim": f"claim {cid}",
                        "context": [{"source": "s", "text": "alpha beta gamma"}],
                    }
                )
                prepass_recs.append({"id": cid, "g": 8, "k_correct": k_val, "gold_verdict": gold_cls, "pass_seed": 0, "descriptive": False})
                idx += 1
        assert len(rows_493) == 493

        prepass_path = tmp_path / "pass_a_493.jsonl"
        prepass_path.write_text(
            "".join(json.dumps(r) + "\n" for r in prepass_recs),
            encoding="utf-8",
            newline="\n",
        )
        prepass_sha = hashlib.sha256(prepass_path.read_bytes()).hexdigest()

        train_rows, holdout_rows, meta = split_stratified_holdout(
            rows_493,
            (prepass_path, prepass_sha),
            holdout_fraction=0.20,
            holdout_seed="pr79-holdout-v1",
        )
        assert len(holdout_rows) == 99
        assert len(train_rows) == 394
        assert meta["holdout_count"] == 99
        assert meta["train_count"] == 394
        holdout_ids = {r["id"] for r in holdout_rows}
        train_ids = {r["id"] for r in train_rows}
        assert holdout_ids.isdisjoint(train_ids)
        assert len(holdout_ids | train_ids) == 493

        # Verify each of the 9 cells has proportional 20% holdout within 1 item of exact quota
        for band, gold_cls, count, _ in cell_specs:
            cell_info = meta["cells"][f"{band}:{gold_cls}"]
            assert cell_info["total"] == count
            assert abs(cell_info["holdout"] - (0.20 * count)) < 1.0

        # Determinism across shuffled input rows and dictionary input
        shuffled_rows = list(rows_493)
        random.Random(99).shuffle(shuffled_rows)
        train_2, holdout_2, meta_2 = split_stratified_holdout(
            shuffled_rows,
            {r["id"]: r["k_correct"] for r in prepass_recs},
            holdout_fraction=0.20,
            holdout_seed="pr79-holdout-v1",
        )
        assert [r["id"] for r in holdout_2] == [r["id"] for r in holdout_rows]
        assert [r["id"] for r in train_2] == [r["id"] for r in train_rows]
        assert meta_2["holdout_ids_sha256"] == meta["holdout_ids_sha256"]

        # Class-stratified epoch draw without replacement:
        # 1. Exact class counts per epoch, zero repeats within an epoch, reshuffled per epoch
        expected_train_class_counts = Counter(r["label"] for r in train_rows)
        ep0 = build_class_stratified_epoch(train_rows, seed=101, epoch=0)
        ep1 = build_class_stratified_epoch(train_rows, seed=101, epoch=1)
        assert len(ep0) == 394 and len(ep1) == 394
        assert len({r["id"] for r in ep0}) == 394
        assert len({r["id"] for r in ep1}) == 394
        assert Counter(r["label"] for r in ep0) == expected_train_class_counts
        assert Counter(r["label"] for r in ep1) == expected_train_class_counts
        assert [r["id"] for r in ep0] != [r["id"] for r in ep1]
        assert {r["id"] for r in ep0}.isdisjoint(holdout_ids)

        # Prefix of 50 items is also class-stratified within 1 item of exact class share
        prefix_50 = ep0[:50]
        prefix_counts = Counter(r["label"] for r in prefix_50)
        for cls, total_cls in expected_train_class_counts.items():
            expected_prefix_cls = 50.0 * (total_cls / 394.0)
            assert abs(prefix_counts[cls] - expected_prefix_cls) <= 1.0

        # Multi-epoch schedule (800 groups = 2+ epochs over 394 training items) via draw_prompts
        sched_800, sched_rec = draw_prompts(
            rows_493,
            800,
            mode="class_stratified",
            seed=101,
            prepass=(prepass_path, prepass_sha),
            holdout_seed="pr79-holdout-v1",
        )
        assert len(sched_800) == 800
        assert sched_rec["draw_mode"] == "class_stratified"
        assert sched_rec["holdout_count"] == 99
        assert sched_rec["holdout_ids_sha256"] == meta["holdout_ids_sha256"]
        assert {r["id"] for r in sched_800}.isdisjoint(holdout_ids)
        # First 394 and second 394 are each a full permutation of train_rows without replacement
        assert {r["id"] for r in sched_800[:394]} == train_ids
        assert {r["id"] for r in sched_800[394:788]} == train_ids
        assert [r["id"] for r in sched_800[:394]] != [r["id"] for r in sched_800[394:788]]

        # Refusal when a holdout item is in the training pool
        with pytest.raises(SystemExit, match="holdout_item_in_training"):
            build_class_stratified_schedule(
                train_rows + [holdout_rows[0]],
                10,
                seed=101,
                holdout_ids=holdout_ids,
            )

    def test_ct_collapse_guard_s1_s2_u1_u2_and_run_steps_halt(self, tmp_path: Path) -> None:
        from contextlib import nullcontext

        torch = pytest.importorskip("torch")

        # 1. S1: >= 3 distinct gold-supported items in window have CT AND >= 2 have positive CT advantage.
        # Same item repeated 3 times does NOT trigger S1 (distinct item count = 1).
        guard_s1 = s2.CtCollapseGuard(window_groups=4)
        assert (
            guard_s1.record_group(
                step=0,
                group=0,
                item_id="sup_1",
                gold_verdict="supported",
                verdicts=["cannot_tell", "supported"],
                correct_flags=[False, True],
                advantages_list=[0.5, -0.5],
            )
            is None
        )
        assert (
            guard_s1.record_group(
                step=0,
                group=1,
                item_id="sup_1",
                gold_verdict="supported",
                verdicts=["cannot_tell", "supported"],
                correct_flags=[False, True],
                advantages_list=[0.5, -0.5],
            )
            is None
        )
        assert (
            guard_s1.record_group(
                step=1,
                group=0,
                item_id="sup_1",
                gold_verdict="supported",
                verdicts=["cannot_tell", "supported"],
                correct_flags=[False, True],
                advantages_list=[0.5, -0.5],
            )
            is None
        )
        # 4th group completes window: only 2 distinct items (sup_1, sup_2) -> does NOT halt
        assert (
            guard_s1.record_group(
                step=1,
                group=1,
                item_id="sup_2",
                gold_verdict="supported",
                verdicts=["cannot_tell", "supported"],
                correct_flags=[False, True],
                advantages_list=[0.5, -0.5],
            )
            is None
        )

        # Next window of 4 groups: 3 distinct gold-supported items with CT, 2 with positive CT adv -> halts S1!
        guard_s1.record_group(
            step=2,
            group=0,
            item_id="sup_1",
            gold_verdict="supported",
            verdicts=["cannot_tell", "unsupported"],
            correct_flags=[False, False],
            advantages_list=[0.8, -0.8],
        )
        guard_s1.record_group(
            step=2,
            group=1,
            item_id="sup_2",
            gold_verdict="supported",
            verdicts=["cannot_tell", "unsupported"],
            correct_flags=[False, False],
            advantages_list=[0.8, -0.8],
        )
        guard_s1.record_group(
            step=3,
            group=0,
            item_id="sup_3",
            gold_verdict="supported",
            verdicts=["cannot_tell", "supported"],
            correct_flags=[False, True],
            advantages_list=[-0.5, 0.5],
        )
        # 4th group is a zero-variance cannot_tell group (counted in the 4-group window, no guard on gold CT)
        stop_s1 = guard_s1.record_group(
            step=3,
            group=1,
            item_id="ct_1",
            gold_verdict="cannot_tell",
            verdicts=["cannot_tell", "cannot_tell"],
            correct_flags=[True, True],
            advantages_list=[0.0, 0.0],
            is_zero_variance=True,
        )
        assert stop_s1 is not None
        assert stop_s1["triggered_rules"] == ["S1"]
        assert stop_s1["reason"] == "ct_guard_s1"

        # 2. S2 (cumulative): CT > 1.5% once >= 600 gold-supported answers have accrued across >= 4 distinct items
        guard_s2 = s2.CtCollapseGuard(window_groups=4)
        # Feed 76 groups of 8 answers = 608 gold-supported answers across 4 distinct items with 10 CTs (1.64%)
        stop_s2 = None
        for idx in range(76):
            cid = f"sup_item_{idx % 4}"
            # Put 1 CT in each of the first 10 windows across all 4 distinct items (idx % 4 = 0,1,2,3)
            has_ct = idx in (0, 5, 10, 15, 20, 25, 30, 35, 40, 45)
            v_list = (["cannot_tell"] + ["supported"] * 7) if has_ct else (["supported"] * 8)
            c_list = ([False] + [True] * 7) if has_ct else ([True] * 8)
            a_list = ([-0.8] + [0.11] * 7) if has_ct else ([0.0] * 8)
            res = guard_s2.record_group(
                step=idx // 2,
                group=idx % 2,
                item_id=cid,
                gold_verdict="supported",
                verdicts=v_list,
                correct_flags=c_list,
                advantages_list=a_list,
                is_zero_variance=not has_ct,
            )
            if res is not None:
                stop_s2 = res
                break
        assert stop_s2 is not None
        assert stop_s2["triggered_rules"] == ["S2"]
        assert stop_s2["guard_metrics"]["s2_cum_sup_answers"] == 608
        assert stop_s2["guard_metrics"]["s2_cum_distinct_ct_items"] == 4

        # 3. U1: >= 4 distinct gold-unsupported items in window have CT AND window correct rate < 78%
        guard_u1 = s2.CtCollapseGuard(window_groups=4)
        for g_idx in range(3):
            assert (
                guard_u1.record_group(
                    step=0,
                    group=g_idx,
                    item_id=f"u_{g_idx}",
                    gold_verdict="unsupported",
                    verdicts=["cannot_tell", "cannot_tell", "unsupported", "unsupported"],
                    correct_flags=[False, False, True, True],
                    advantages_list=[-0.5, -0.5, 0.5, 0.5],
                )
                is None
            )
        stop_u1 = guard_u1.record_group(
            step=1,
            group=1,
            item_id="u_3",
            gold_verdict="unsupported",
            verdicts=["cannot_tell", "cannot_tell", "unsupported", "unsupported"],
            correct_flags=[False, False, True, True],
            advantages_list=[-0.5, -0.5, 0.5, 0.5],
        )
        assert stop_u1 is not None
        assert stop_u1["triggered_rules"] == ["U1"]
        assert stop_u1["guard_metrics"]["u1_correct_rate"] == 0.5

        # 4. U2: >= 2 distinct gold-unsupported items with Pass-A k >= 6 have any CT in one window
        guard_u2 = s2.CtCollapseGuard(
            window_groups=2,
            pass_a_k={"u_easy_1": 6, "u_easy_2": 8, "u_hard": 3},
        )
        assert (
            guard_u2.record_group(
                step=0,
                group=0,
                item_id="u_easy_1",
                gold_verdict="unsupported",
                verdicts=["cannot_tell"] + ["unsupported"] * 7,
                correct_flags=[False] + [True] * 7,
                advantages_list=[-0.5] + [0.07] * 7,
            )
            is None
        )
        stop_u2 = guard_u2.record_group(
            step=0,
            group=1,
            item_id="u_easy_2",
            gold_verdict="unsupported",
            verdicts=["cannot_tell"] + ["unsupported"] * 7,
            correct_flags=[False] + [True] * 7,
            advantages_list=[-0.5] + [0.07] * 7,
        )
        assert stop_u2 is not None
        assert stop_u2["triggered_rules"] == ["U2"]

        # 5. End-to-end run_steps halts on U2 at window boundary and refuses holdout items
        tok = s2.DryTokenizer()

        class _TinyModel:
            device = torch.device("cpu")

            def __init__(self) -> None:
                self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

            def parameters(self):
                return [self._p]

            def __call__(self, input_ids, attention_mask=None, **kwargs):
                logits = (
                    torch.zeros((input_ids.shape[0], input_ids.shape[1], tok.vocab_size), dtype=torch.float32)
                    + self._p * 0.0
                )
                return Namespace(logits=logits)

        prompts_u = [
            {
                "id": "u_easy_1",
                "label": "unsupported",
                "claim": "c1",
                "context": [{"source": "s", "text": "alpha beta gamma"}],
            },
            {
                "id": "u_easy_2",
                "label": "unsupported",
                "claim": "c2",
                "context": [{"source": "s", "text": "alpha beta gamma"}],
            },
        ]
        # Refuses if any prompt is in holdout_ids
        with pytest.raises(SystemExit, match="holdout_item_in_training"):
            s2.run_steps(
                model=_TinyModel(),
                ref_adapter_off=nullcontext,
                tok=tok,
                optimiser=Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                prompts=prompts_u,
                rollout=lambda text, g_cnt, max_n, temp: (
                    tok.encode(text),
                    [tok.encode(_reply("unsupported", "alpha beta gamma")) + [tok.eos_token_id]] * g_cnt,
                    [False] * g_cnt,
                ),
                steps=2,
                g=2,
                max_new=32,
                reward_fn=lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
                ),
                device_sync=lambda: None,
                peak_alloc_mib=lambda: 100,
                holdout_ids={"u_easy_2"},
            )

        # Halts at end of step 0 (2 groups = 1 window of 2 groups) when U2 triggers
        steps_out, stopped_out, oom_out = s2.run_steps(
            model=_TinyModel(),
            ref_adapter_off=nullcontext,
            tok=tok,
            optimiser=Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
            prompts=prompts_u,
            rollout=lambda text, g_cnt, max_n, temp: (
                tok.encode(text),
                [
                    tok.encode(_reply("cannot_tell", "alpha beta gamma")) + [tok.eos_token_id],
                    tok.encode(_reply("unsupported", "alpha beta gamma")) + [tok.eos_token_id],
                ][:g_cnt],
                [False] * g_cnt,
            ),
            steps=5,
            g=2,
            max_new=32,
            reward_fn=lambda text, row, tt, truncated=False: s2.reward(
                text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
            ),
            device_sync=lambda: None,
            peak_alloc_mib=lambda: 100,
            ct_guard=s2.CtCollapseGuard(window_groups=2, pass_a_k={"u_easy_1": 7, "u_easy_2": 8}),
        )
        assert not oom_out
        assert len(steps_out) == 1
        assert stopped_out is not None
        assert stopped_out["triggered_rules"] == ["U2"]

    def test_mask_wrong_positive_advantage_dual_clip_and_truncation_out_of_group_mean(self) -> None:
        torch = pytest.importorskip("torch")

        # 1. Masking fallback: wrong-label completions with positive advantage are clamped to <= 0
        # after mean-centring, with NO re-centring after the clamp.
        # Suppose on a gold-supported item:
        # - 1 completion is 'supported' (+1.0, correct=True)
        # - 2 completions are 'cannot_tell' (-0.3, correct=False)
        # - 5 completions are 'supported' with unfound quote on gold unsupported? Or suppose on gold supported:
        #   5 completions are malformed (-1.0, correct=False), 2 are cannot_tell (-0.3, correct=False), 1 is supported (+1.0, correct=True)
        rewards = [1.0, -0.3, -0.3, -1.0, -1.0, -1.0, -1.0, -1.0]
        correct_flags = [True, False, False, False, False, False, False, False]
        raw_adv = s2.advantages(rewards, mode="dr-grpo")
        # Mean is -0.575, so cannot_tell (-0.3) gets positive advantage +0.275 before clamping!
        assert raw_adv[0] == pytest.approx(1.575)
        assert raw_adv[1] == pytest.approx(0.275)
        assert raw_adv[3] == pytest.approx(-0.425)

        clamped_adv = s2.clamp_wrong_positive_advantages(raw_adv, correct_flags)
        # Wrong-label positive advantages (indices 1, 2) are clamped to 0.0;
        # right-label positive (index 0) and wrong-label negative (indices 3..7) are UNCHANGED (no re-centring!)
        assert clamped_adv[0] == pytest.approx(1.575)
        assert clamped_adv[1] == 0.0
        assert clamped_adv[2] == 0.0
        for idx in range(3, 8):
            assert clamped_adv[idx] == pytest.approx(-0.425)
        assert sum(clamped_adv) != pytest.approx(0.0)

        # 2. Dual-clip c = 3.0 caps the policy surrogate on negative advantages when ratio > 3.0
        old_logp = torch.zeros((1, 4), dtype=torch.float32)
        ref_logp = torch.zeros((1, 4), dtype=torch.float32)
        # logp = log(10.0) -> ratio = 10.0 > c = 3.0
        logp_high = torch.full((1, 4), float(torch.log(torch.tensor(10.0))), dtype=torch.float32)
        adv_neg = torch.tensor([-2.0], dtype=torch.float32)
        mask = torch.ones((1, 4), dtype=torch.float32)

        loss_unclipped = s2.grpo_loss(logp_high, old_logp, ref_logp, adv_neg, mask, beta=0.0, dual_clip_c=None)
        loss_dual_clipped = s2.grpo_loss(
            logp_high, old_logp, ref_logp, adv_neg, mask, beta=0.0, dual_clip_c=3.0
        )
        # Unclipped policy term is -ratio * adv = -10.0 * (-2.0) = +20.0
        # Dual-clipped policy term at c=3.0 is -c * adv = -3.0 * (-2.0) = +6.0
        assert float(loss_unclipped) == pytest.approx(20.0, rel=1e-5)
        assert float(loss_dual_clipped) == pytest.approx(6.0, rel=1e-5)

        # 3. Truncation out of the group mean:
        # When 2 of 8 completions are truncated (-1.0) and the 6 survivors all score +1.0:
        # - Without mask_truncated, survivors get positive advantage against the -1.0 truncations.
        # - With mask_truncated, truncated completions are excluded from the group mean & std BEFORE
        #   computing advantages, so the 6 identical survivors have zero variance and 0.0 advantage!
        r_group = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0]
        truncs = [False, False, False, False, False, False, True, True]
        active_r = [r for r, tr in zip(r_group, truncs, strict=True) if not tr]
        assert s2.zero_variance(active_r) is True
        active_adv = s2.advantages(active_r, mode="dr-grpo")
        it = iter(active_adv)
        masked_adv = [0.0 if tr else next(it) for tr in truncs]
        assert masked_adv == [0.0] * 8

        # Even when survivors vary (e.g. three +1.0 and three +0.8), their mean is 0.9 (not pulled down by -1.0)
        r_varied = [1.0, 1.0, 1.0, 0.8, 0.8, 0.8, -1.0, -1.0]
        active_varied = [r for r, tr in zip(r_varied, truncs, strict=True) if not tr]
        adv_varied = s2.advantages(active_varied, mode="dr-grpo")
        it_v = iter(adv_varied)
        masked_varied = [0.0 if tr else next(it_v) for tr in truncs]
        assert masked_varied[:3] == [pytest.approx(0.1)] * 3
        assert masked_varied[3:6] == [pytest.approx(-0.1)] * 3
        assert masked_varied[6:] == [0.0, 0.0]

    def test_postpass_confident_errors_confusion_matrix_hedge_flag_and_paired_estimators(self) -> None:
        # Build Pass A, Pass B, and 3 seeds x 2 post-passes over 3 items (1 per band & gold class)
        pass_a = [
            {
                "id": "i_sup_b0",
                "gold_verdict": "supported",
                "g": 8,
                "k_correct": 4,
                "supported": 4,
                "unsupported": 3,
                "cannot_tell": 1,
                "malformed": 0,
                "truncations": 0,
                "reward_mean": 0.2, "pass_seed": 0, "descriptive": False,
            },
            {
                "id": "i_unsup_b1",
                "gold_verdict": "unsupported",
                "g": 8,
                "k_correct": 6,
                "supported": 2,
                "unsupported": 6,
                "cannot_tell": 0,
                "malformed": 0,
                "truncations": 0,
                "reward_mean": 0.4, "pass_seed": 0, "descriptive": False,
            },
            {
                "id": "i_ct_b2",
                "gold_verdict": "cannot_tell",
                "g": 8,
                "k_correct": 8,
                "supported": 0,
                "unsupported": 4,
                "cannot_tell": 4,
                "malformed": 0,
                "truncations": 0,
                "reward_mean": 1.0, "pass_seed": 0, "descriptive": False,
            },
        ]
        pass_b = [
            {
                "id": "i_sup_b0",
                "gold_verdict": "supported",
                "g": 8,
                "k_correct": 6,
                "supported": 6,
                "unsupported": 2,
                "cannot_tell": 0,
                "malformed": 0,
                "truncations": 0,
                "reward_mean": 0.5, "pass_seed": 1, "descriptive": False,
            },
            {
                "id": "i_unsup_b1",
                "gold_verdict": "unsupported",
                "g": 8,
                "k_correct": 6,
                "supported": 2,
                "unsupported": 6,
                "cannot_tell": 0,
                "malformed": 0,
                "truncations": 0,
                "reward_mean": 0.4, "pass_seed": 1, "descriptive": False,
            },
            {
                "id": "i_ct_b2",
                "gold_verdict": "cannot_tell",
                "g": 8,
                "k_correct": 6,
                "supported": 2,
                "unsupported": 3,
                "cannot_tell": 3,
                "malformed": 0,
                "truncations": 0,
                "reward_mean": 0.4, "pass_seed": 1, "descriptive": False,
            },
        ]

        # Confident errors on Pass A:
        # i_sup_b0 (gold supported): unsupported = 3
        # i_unsup_b1 (gold unsupported): supported = 2
        # i_ct_b2 (gold cannot_tell): supported = 0
        assert [s2.item_confident_errors(r) for r in pass_a] == [3, 2, 0]
        assert [s2.item_confident_errors(r) for r in pass_b] == [2, 2, 2]

        # Post-passes for 3 seeds:
        # Make CT share on gold cannot_tell drop to 1/8 = 12.5% (< 21% threshold) so hedge_flag triggers
        def make_post(k_sup: int, unsup_on_sup: int, ct_on_ct: int, pass_seed: int = 2):
            return [
                {
                    "id": "i_sup_b0",
                    "gold_verdict": "supported",
                    "g": 8,
                    "k_correct": k_sup,
                    "supported": k_sup,
                    "unsupported": unsup_on_sup,
                    "cannot_tell": 8 - k_sup - unsup_on_sup,
                    "malformed": 0,
                    "truncations": 0,
                    "reward_mean": 0.8, "pass_seed": pass_seed, "descriptive": False,
                },
                {
                    "id": "i_unsup_b1",
                    "gold_verdict": "unsupported",
                    "g": 8,
                    "k_correct": 7,
                    "supported": 1,
                    "unsupported": 7,
                    "cannot_tell": 0,
                    "malformed": 0,
                    "truncations": 0,
                    "reward_mean": 0.7, "pass_seed": pass_seed, "descriptive": False,
                },
                {
                    "id": "i_ct_b2",
                    "gold_verdict": "cannot_tell",
                    "g": 8,
                    "k_correct": 8,
                    "supported": 0,
                    "unsupported": 8 - ct_on_ct,
                    "cannot_tell": ct_on_ct,
                    "malformed": 0,
                    "truncations": 0,
                    "reward_mean": 1.0, "pass_seed": pass_seed, "descriptive": False,
                },
            ]

        seed_postpasses = [
            (make_post(7, 1, 1, 2), make_post(8, 0, 1, 3)),  # seed 1 post_mean k on i_sup_b0 = 7.5, ce = 0.5
            (make_post(7, 1, 1, 2), make_post(7, 1, 1, 3)),  # seed 2 post_mean k on i_sup_b0 = 7.0, ce = 1.0
            (make_post(8, 0, 1, 2), make_post(7, 1, 1, 3)),  # seed 3 post_mean k on i_sup_b0 = 7.5, ce = 0.5
        ]
        # Across the 3 seeds, i_sup_b0 has post_mean_k = (7.5 + 7.0 + 7.5)/3 = 7.333333, ce = 2.0/3 = 0.666667
        est = s2.compute_paired_arm_estimators(
            pass_a,
            pass_b,
            seed_postpasses,
            holdout_ids={"i_ct_b2"},
            baseline_ct_share=0.42,
        )
        assert est["n_items"] == 3
        assert est["n_seeds"] == 3
        assert est["per_item_post_mean_k"]["i_sup_b0"] == pytest.approx(22.0 / 3.0, abs=1e-5)
        assert est["per_item_post_mean_confident_errors"]["i_sup_b0"] == pytest.approx(2.0 / 3.0, abs=1e-5)

        # Band [0,5] (i_sup_b0): estimator is post_mean - B (B has k=6, ce=2)
        b0 = est["by_pass_a_band"]["[0,5]"]
        assert b0["estimator"] == "post_mean - B"
        assert b0["k_diff"] == pytest.approx((22.0 / 3.0) - 6.0, abs=1e-5)
        assert b0["confident_error_diff"] == pytest.approx((2.0 / 3.0) - 2.0, abs=1e-5)

        # Unbanded holdout (i_ct_b2): estimator is post_mean - (A + B) / 2
        # A has k=8, B has k=6 -> (A+B)/2 = 7.0; post_mean k = 8.0 -> diff = +1.0
        # A has ce=0, B has ce=2 -> (A+B)/2 = 1.0; post_mean ce = 0.0 -> diff = -1.0
        hold_est = est["unbanded"]["holdout"]
        assert hold_est["estimator"] == "post_mean - (A + B) / 2"
        assert hold_est["k_diff"] == pytest.approx(1.0, abs=1e-5)
        assert hold_est["confident_error_diff"] == pytest.approx(-1.0, abs=1e-5)

        # Hedge flag triggered because ct_share_of_correct = 1/8 = 0.125 < 0.21
        assert est["hedge_flag_after"]["ct_share_of_correct"] == pytest.approx(0.125)
        assert est["hedge_flag_after"]["triggered"] is True
        assert est["confusion_matrix_after"]["cannot_tell"]["cannot_tell"] == 6
        assert est["confusion_matrix_after"]["cannot_tell"]["unsupported"] == 42


# ---------------------------------------------------------------------------
# 2026-11-05 Part D: CT Interview Runner & Two-Judge Tagging Pipeline
# ---------------------------------------------------------------------------
class TestInterviewAndTagging:
    def test_cohort_selection_first_ct_rates_and_resolution_mcnemar(self):
        import s2_interview as s2i

        def _samp(cid: str, idx: int, verdict: str, part: str = "subclaim A") -> dict:
            body = json.dumps(
                {
                    "verdict": verdict,
                    "reason": "r",
                    "evidence_quotes": [{"part": part, "verdict": verdict, "quote": ""}],
                }
            )
            return {
                "item_id": cid,
                "sample_index": idx,
                "verdict": verdict,
                "text": f"<think>trace {cid} s{idx}</think>{body}",
            }

        # Item 1: both_CT (baseline has CT at sample 1 and 2 -> 2/4 = 0.5, majority (>=0.5);
        # checkpoint has CT at sample 0, 1, 2 -> 3/4 = 0.75, majority)
        # Item 2: baseline_only_CT (baseline has CT at sample 2; checkpoint has 0 CT)
        # Item 3: checkpoint_only_CT (baseline has 0 CT; checkpoint has CT at sample 3)
        # Item 4: neither CT (excluded)
        base_recs = [
            _samp("item_1", 0, "supported"),
            _samp("item_1", 2, "cannot_tell", "second ct"),
            _samp("item_1", 1, "cannot_tell", "first ct"),
            _samp("item_1", 3, "unsupported"),
            _samp("item_2", 0, "supported"),
            _samp("item_2", 1, "cannot_tell", "b_only ct"),
            _samp("item_3", 0, "supported"),
            _samp("item_4", 0, "supported"),
        ]
        ckpt_recs = [
            _samp("item_1", 0, "cannot_tell", "ckpt first ct"),
            _samp("item_1", 1, "cannot_tell"),
            _samp("item_1", 2, "cannot_tell"),
            _samp("item_1", 3, "supported"),
            _samp("item_2", 0, "supported"),
            _samp("item_2", 1, "supported"),
            _samp("item_3", 0, "unsupported"),
            _samp("item_3", 1, "cannot_tell", "ckpt_only ct"),
            _samp("item_4", 0, "unsupported"),
        ]

        cohort = s2i.select_interview_cohort(base_recs, ckpt_recs)
        assert [c["item_id"] for c in cohort] == ["item_1", "item_2", "item_3"]
        c1, c2, c3 = cohort
        assert c1["cohort"] == "both_CT"
        assert c1["baseline_ct_rate"] == pytest.approx(0.5)
        assert c1["baseline_majority_ct"] is True
        assert c1["baseline_first_ct"]["sample_index"] == 1
        assert "first ct" in c1["baseline_first_ct"]["text"]
        assert c1["checkpoint_ct_rate"] == pytest.approx(0.75)
        assert c1["checkpoint_majority_ct"] is True
        assert c1["checkpoint_first_ct"]["sample_index"] == 0

        assert c2["cohort"] == "baseline_only_CT"
        assert c2["baseline_first_ct"]["sample_index"] == 1
        assert c2["checkpoint_first_ct"] is None

        assert c3["cohort"] == "checkpoint_only_CT"
        assert c3["baseline_first_ct"] is None
        assert c3["checkpoint_first_ct"]["sample_index"] == 1

        res = s2i.summarize_cohort_resolution(
            cohort,
            checkpoint_verdicts_by_item={"item_2": "supported"},
            baseline_q3_lean_by_item={"item_2": "supported"},
            both_ct_paired_outcomes={"item_1": (False, True)},
        )
        assert res["cohort_counts"] == {
            "both_CT": 1,
            "baseline_only_CT": 1,
            "checkpoint_only_CT": 1,
        }
        assert res["resolved_cross_tab_lean_vs_checkpoint"]["supported"]["supported"] == 1
        assert res["both_ct_mcnemar"] == {
            "both_true": 0,
            "baseline_only_true_minus": 0,
            "checkpoint_only_true_plus": 1,
            "both_false": 0,
        }

    def test_turn1_thinking_stripped_in_turn2_prompt_and_violation_caught(self):
        import s2_interview as s2i

        tok = s2i.Qwen3ChatStubTokenizer()
        row = _row("item_strip_01", claim="The cache timeout is 30 seconds.")
        secret_think = "SECRET_TURN1_THINKING_TRACE_THAT_MUST_BE_STRIPPED"
        turn1_reply = (
            f"<think>{secret_think}</think>"
            '{"verdict":"cannot_tell","reason":"no timeout given",'
            '"evidence_quotes":[{"part":"timeout is 30 seconds","verdict":"cannot_tell","quote":""}]}'
        )
        msgs = s2i.build_interview_messages(row, turn1_reply, "Q1")
        rendered_text, prompt_ids = s2i.render_interview_prompt(tok, msgs)

        assert s2i.verify_turn1_thinking_stripped(rendered_text, prompt_ids, secret_think) is True
        assert secret_think not in rendered_text
        assert s2i.QWEN3_END_THINK_TOKEN_ID not in prompt_ids
        # Generation prompt at the end opens Turn 2 thinking (`<think>\n`), so exactly 1 `<think>` token is at the tail
        assert prompt_ids.count(s2i.QWEN3_THINK_TOKEN_ID) == 1

        # Verify assertion triggers if `</think>` or Turn 1 thinking text leaks into prompt
        with pytest.raises(AssertionError, match="leaked"):
            s2i.verify_turn1_thinking_stripped(
                rendered_text + "</think>",
                [*prompt_ids, s2i.QWEN3_END_THINK_TOKEN_ID],
                secret_think,
            )
        with pytest.raises(AssertionError, match="leaked"):
            s2i.verify_turn1_thinking_stripped(
                rendered_text + secret_think,
                prompt_ids,
                secret_think,
            )

    @needs_torch
    def test_two_stage_thinking_cap_truncates_at_256_and_natural_close_not_truncated(self):
        import s2_interview as s2i
        import torch

        tok = s2i.Qwen3ChatStubTokenizer()

        class _UnclosedThinkModel:
            device = torch.device("cpu")

            def __init__(self):
                self.calls = []

            def generate(self, input_ids, max_new_tokens, **kwargs):
                self.calls.append((int(input_ids.shape[1]), int(max_new_tokens)))
                if len(self.calls) == 1:
                    # Emit max_new_tokens of 'a' without closing </think>
                    new_toks = tok.encode("thinking word " * 40)[:max_new_tokens]
                else:
                    # Stage 2 answer tokens after forced \n</think>\n\n
                    new_toks = tok.encode("The missing piece is the cache TTL.")[:max_new_tokens]
                out = torch.tensor([list(input_ids[0].tolist()) + new_toks], dtype=torch.long)
                return out

        m_trunc = _UnclosedThinkModel()
        res_trunc = s2i.generate_two_stage_thinking(
            m_trunc,
            tok,
            [10, 11, 12],
            seed=42,
            thinking_cap=256,
            answer_cap=200,
        )
        assert res_trunc["think_truncated"] is True
        assert res_trunc["thinking_tokens"] == 256
        assert len(m_trunc.calls) == 2
        assert m_trunc.calls[0][1] == 256
        assert m_trunc.calls[1][1] == 200
        assert "The missing piece is the cache TTL." in res_trunc["answer"]
        assert res_trunc["ngram_4_rep_rate"] > 0.0

        class _ClosedThinkModel:
            device = torch.device("cpu")

            def generate(self, input_ids, max_new_tokens, **kwargs):
                # Emit short thinking + </think> + answer + EOS inside Stage 1
                seq = (
                    tok.encode("short reasoning")
                    + [s2i.QWEN3_END_THINK_TOKEN_ID]
                    + tok.encode("LEAN: supported — matches context.")
                    + [s2i.QWEN3_EOS_TOKEN_ID]
                )
                return torch.tensor([list(input_ids[0].tolist()) + seq], dtype=torch.long)

        res_closed = s2i.generate_two_stage_thinking(
            _ClosedThinkModel(),
            tok,
            [10, 11, 12],
            seed=99,
            thinking_cap=256,
            answer_cap=200,
        )
        assert res_closed["think_truncated"] is False
        assert "short reasoning" in res_closed["thinking"]
        assert res_closed["answer"].startswith("LEAN: supported")

    def test_mechanical_measures_part_match_whole_claim_guard_quotes_counterfactual_and_q3_lean(self):
        import s2_interview as s2i

        claim = (
            "The authentication service rotates signing keys every twenty four hours "
            "and stores backup keys in an offline hardware security module."
        )
        json_ct_parts = ["stores backup keys in an offline hardware security module"]

        # Specific Q1 answer matching the CT sub-claim -> high containment & match_at_0_3/0_5/0_7 True
        good_q1 = "stores backup keys in an offline hardware security module"
        sc_good = s2i.score_part_match(good_q1, json_ct_parts, claim)
        assert sc_good["whole_claim_copy"] is False
        assert sc_good["containment"] >= 0.7
        assert sc_good["match_at_0_3"] is True
        assert sc_good["match_at_0_5"] is True
        assert sc_good["match_at_0_7"] is True

        # Whole-claim copy (>85% of claim tokens while JSON CT part is <70% of claim tokens) -> gated off
        copy_q1 = (
            "The authentication service rotates signing keys every twenty four hours "
            "and stores backup keys in an offline hardware security module."
        )
        sc_copy = s2i.score_part_match(copy_q1, json_ct_parts, claim)
        assert sc_copy["whole_claim_copy"] is True
        assert sc_copy["match_at_0_3"] is False
        assert sc_copy["match_at_0_5"] is False
        assert sc_copy["match_at_0_7"] is False

        # Quote validity & consistency (Q4)
        evidence = ["The authentication service rotates signing keys every 24 hours."]
        t1_json = json.dumps(
            {
                "verdict": "cannot_tell",
                "reason": "backup storage not mentioned",
                "evidence_quotes": [
                    {
                        "part": "rotates signing keys",
                        "verdict": "supported",
                        "quote": "rotates signing keys every 24 hours",
                    }
                ],
            }
        )
        q4_valid = s2i.score_quote_validity_and_consistency(
            "rotates signing keys every 24 hours", evidence, t1_json
        )
        assert q4_valid["quote_valid"] is True
        assert q4_valid["in_evidence"] is True
        assert q4_valid["quote_consistency_f1"] == pytest.approx(1.0)

        q4_none = s2i.score_quote_validity_and_consistency("NONE", evidence, t1_json)
        assert q4_none["is_none"] is True
        assert q4_none["quote_valid"] is True

        q4_hallucinated = s2i.score_quote_validity_and_consistency(
            "backup keys are kept in AWS KMS vault", evidence, t1_json
        )
        assert q4_hallucinated["quote_valid"] is False

        # Counterfactual check (Q2 MISSING_FACT)
        row = _row("cf_item_1", claim=claim, text=evidence[0])
        q2_ans = "MISSING_FACT: Backup keys are stored in an offline hardware security module."

        def fake_verifier(cf_row: dict, seed: int) -> str:
            joined = " ".join(c["text"] for c in cf_row["context"])
            assert "offline hardware security module" in joined
            return json.dumps(
                {
                    "verdict": "supported",
                    "reason": "now supported",
                    "evidence_quotes": [
                        {
                            "part": "offline hardware security module",
                            "verdict": "supported",
                            "quote": "Backup keys are stored in an offline hardware security module",
                        }
                    ],
                }
            )

        cf_res = s2i.run_counterfactual_check(row, q2_ans, fake_verifier, base_seed=7)
        assert cf_res["format_ok"] is True
        assert cf_res["left_cannot_tell"] is True
        assert cf_res["new_verdict"] == "supported"
        assert cf_res["direction"] == "supported"

        cf_bad = s2i.run_counterfactual_check(row, "No prefix here.", fake_verifier, base_seed=7)
        assert cf_bad["format_ok"] is False
        assert cf_bad["left_cannot_tell"] is False

        # Q3 forced lean & counterbalancing
        orders = {s2i.q3_choice_order(f"id_{i}") for i in range(20)}
        assert orders == {("supported", "unsupported"), ("unsupported", "supported")}

        l_sup = s2i.score_q3_lean_item(
            "LEAN: supported — partial evidence aligns.", "supported", "id_1"
        )
        l_unsup = s2i.score_q3_lean_item(
            "LEAN: supported — partial evidence aligns.", "unsupported", "id_2"
        )
        l_ct = s2i.score_q3_lean_item(
            "LEAN: unsupported — leans against.", "cannot_tell", "id_3"
        )
        assert l_sup["accuracy_scored"] is True and l_sup["correct"] is True
        assert l_unsup["accuracy_scored"] is True and l_unsup["correct"] is False
        assert l_ct["accuracy_scored"] is False and l_ct["correct"] is None

        lean_summary = s2i.summarize_q3_lean([l_sup, l_unsup, l_ct])
        assert lean_summary["balanced_accuracy"] == pytest.approx(0.5)
        assert lean_summary["gold_cannot_tell"]["accuracy_scored"] is False
        assert lean_summary["gold_cannot_tell"]["unsupported_share"] == pytest.approx(1.0)

    @needs_torch
    def test_cpu_3_item_synthetic_interview_end_to_end(self, tmp_path):
        import hashlib

        import s2_interview as s2i
        import torch

        init_model, _ = s2i.build_cpu_qwen3_stub(with_adapter=True)
        with torch.no_grad():
            for name, param in init_model.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.05)
        ckpt_dir = tmp_path / "syn_ckpt"
        init_model.save_pretrained(str(ckpt_dir))
        ckpt_sha = s2.checkpoint_sha256(ckpt_dir)
        model, tok = s2i.build_cpu_qwen3_stub(checkpoint_path=ckpt_dir)
        runner = s2i.InterviewModelRunner(model, tok, expected_checkpoint_sha256=ckpt_sha)

        rows_by_id = {
            f"syn_{i}": _row(
                f"syn_{i}",
                claim=f"Synthetic claim number {i} states the buffer size is 64 bytes.",
                text=f"Passage for item {i}: the module initialises a local buffer.",
                label="supported" if i == 0 else ("unsupported" if i == 1 else "cannot_tell"),
            )
            for i in range(3)
        }

        def _make_ct_rec(cid: str) -> dict:
            t1 = (
                f"<think>Thinking for {cid}: buffer size 64 bytes is missing.</think>"
                + json.dumps(
                    {
                        "verdict": "cannot_tell",
                        "reason": "size missing",
                        "evidence_quotes": [
                            {"part": "buffer size is 64 bytes", "verdict": "cannot_tell", "quote": ""}
                        ],
                    }
                )
            )
            return {"item_id": cid, "sample_index": 0, "verdict": "cannot_tell", "text": t1}

        base_pass = [_make_ct_rec("syn_0"), _make_ct_rec("syn_1")]
        ckpt_pass = [_make_ct_rec("syn_0"), _make_ct_rec("syn_2")]
        cohort = s2i.select_interview_cohort(base_pass, ckpt_pass)
        assert len(cohort) == 3

        out_dir = tmp_path / "interview_out"
        receipt = s2i.run_interview_for_cohort(
            cohort=cohort,
            rows_by_id=rows_by_id,
            runner=runner,
            out_dir=out_dir,
            base_seed=20261105,
            thinking_cap=8,
            answer_cap=8,
            include_cold_control=True,
        )
        assert receipt["cohort_size"] == 3
        # syn_0 (both_CT -> 2*4 + 2*2 cold = 12), syn_1 (baseline_only -> 4 + 2 = 6), syn_2 (ckpt_only -> 4 + 2 = 6) => 24 turns
        assert receipt["total_turns_recorded"] == 24
        save_all_file = out_dir / receipt["save_all_path"]
        assert save_all_file.exists()
        assert hashlib.sha256(save_all_file.read_bytes()).hexdigest() == receipt["save_all_sha256"]
        records = s2.read_save_all(save_all_file)
        assert len(records) == 24
        assert all("ngram_4_rep_rate" in r and "think_truncated" in r for r in records)

    def test_two_judge_tagging_refuses_qwen_gates_controls_computes_ac1_and_exports_blinded_audit(
        self, tmp_path
    ):
        import s2_interview_tag as s2t

        # 1. Refuse Qwen-family judges or duplicate judges
        with pytest.raises(SystemExit, match="judge_qwen_family_forbidden"):
            s2t.validate_judge_name("Qwen/Qwen3-32B-Instruct")
        with pytest.raises(SystemExit, match="judge_qwen_family_forbidden"):
            s2t.require_two_non_qwen_judges("claude-3-5-sonnet", "qwen2.5-72b")
        with pytest.raises(SystemExit, match="judges_must_be_distinct"):
            s2t.require_two_non_qwen_judges("llama-3.3-70b", "Llama-3.3-70B")

        # 2. Build 30 shuffled, 30 verbatim, 15 empty-thinking controls
        controls = s2t.build_judge_controls([])
        assert len(controls["shuffled"]) == 30
        assert len(controls["verbatim"]) == 30
        assert len(controls["empty_thinking"]) == 15

        def accurate_judge(thinking: str, q1_answer: str) -> dict[str, str]:
            t_clean = thinking.strip()
            if not t_clean or "not enough information to decide" in t_clean or "not sure whether" in t_clean or "Cannot tell from" in t_clean:
                return {"missing_element_match": "none", "reason_type": "generic_or_none"}
            if t_clean in q1_answer or "release date" in t_clean and "release date" in q1_answer and t_clean.split()[-1] == q1_answer.split()[-1]:
                return {"missing_element_match": "same", "reason_type": "lacks"}
            return {"missing_element_match": "different", "reason_type": "conflicts"}

        def bad_judge(thinking: str, q1_answer: str) -> dict[str, str]:
            return {"missing_element_match": "same", "reason_type": "lacks"}

        # Gate fails when accuracy < 0.90
        with pytest.raises(SystemExit, match="judge_control_gate_failed"):
            s2t.evaluate_judge_controls(controls, bad_judge, min_accuracy=0.90, enforce_gate=True)

        # Gate passes for accurate judge
        gate_ok = s2t.evaluate_judge_controls(controls, accurate_judge, min_accuracy=0.90, enforce_gate=True)
        assert gate_ok["passed"] is True
        assert all(acc >= 0.90 for acc in gate_ok["accuracies"].values())

        # 3. Run full two-judge pipeline over 35 synthetic Q1 records (with 10 disagreements + 25 agreements)
        q1_records = []
        for i in range(35):
            q1_records.append(
                {
                    "item_id": f"item_{i:02d}",
                    "policy": "baseline" if i % 2 == 0 else "checkpoint",
                    "original_turn1_thinking_tokens": 50 + i * 3,
                    "turn1_thinking": f"Thinking trace {i}: release date for module {i} is missing.",
                    "q1_answer": (
                        f"As noted in the reasoning, Thinking trace {i}: release date for module {i} is missing."
                        if i < 20
                        else f"Different explanation {i}"
                    ),
                }
            )

        call_counter = {"b": 0}

        def judge_b_with_disagreements(thinking: str, q1_answer: str) -> dict[str, str]:
            base = accurate_judge(thinking, q1_answer)
            # Only flip on non-control items (which start with "Thinking trace ")
            if thinking.startswith("Thinking trace "):
                idx = call_counter["b"]
                call_counter["b"] += 1
                if idx < 10:
                    return {
                        "missing_element_match": "different"
                        if base["missing_element_match"] == "same"
                        else "same",
                        "reason_type": "conflicts",
                    }
            return base

        out_dir = tmp_path / "tagging_out"
        summary = s2t.run_two_judge_tagging(
            interview_q1_records=q1_records,
            judge_a_name="meta-llama/Llama-3.3-70B-Instruct",
            judge_b_name="mistralai/Mistral-Large-Instruct",
            judge_a_fn=accurate_judge,
            judge_b_fn=judge_b_with_disagreements,
            out_dir=out_dir,
            controls=controls,
        )
        assert summary["agreement_missing_element_match"]["n"] == 35
        assert summary["agreement_missing_element_match"]["disagreements"] == 10
        assert summary["agreement_missing_element_match"]["raw_agreement"] == pytest.approx(25.0 / 35.0, abs=1e-5)
        assert summary["agreement_missing_element_match"]["cohens_kappa"] is not None
        assert summary["agreement_missing_element_match"]["gwets_ac1"] is not None
        assert summary["thinking_length_confound"]["n"] == 35

        # 4. Verify 30-item blinded human audit export (10 disagreements + 20 stratified)
        audit_meta = summary["blinded_human_audit"]
        assert audit_meta["audit_count"] == 30
        assert audit_meta["disagreements_included"] == 10
        assert audit_meta["stratified_agreements_included"] == 20

        audit_lines = [
            json.loads(line)
            for line in (out_dir / "s2-interview-blinded-audit.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(audit_lines) == 30
        for row in audit_lines:
            assert "policy" not in row
            assert "model" not in row
            assert "item_id" not in row
            assert row["blind_source_code"].startswith("BLIND_")


class TestFix1Items:
    @needs_torch
    def test_item_01_load_checkpoint_into_peft_model(self, tmp_path: Path) -> None:
        import torch

        assert hasattr(s2, "attach_or_load_adapter")
        gold_dir = _prepass_gold(tmp_path / "gold")

        # 1. Build a dry PEFT model, perturb its LoRA weights, and save the adapter to ckpt_dir
        a_init = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(tmp_path / "init_out"),
                "--rnd-commit",
                "deadbee",
                "--dry",
            ]
        )
        b_init = s2.dry_bundle(a_init)
        assert b_init["adapter_loaded"] is False
        assert b_init["checkpoint_sha256"] is None
        assert b_init["checkpoint_files_sha256"] is None
        with torch.no_grad():
            for name, p in b_init["model"].named_parameters():
                if "lora_" in name:
                    p.fill_(0.125)
        ckpt_dir = tmp_path / "saved_lora_ckpt"
        b_init["model"].save_pretrained(str(ckpt_dir))

        # 2. Load the saved adapter via dry_bundle in --postpass mode and verify weights + file shas
        a_post = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(tmp_path / "post_out"),
                "--rnd-commit",
                "deadbee",
                "--dry",
                "--postpass",
                "--pass-seed",
                "2",
                "--checkpoint",
                str(ckpt_dir),
            ]
        )
        b_loaded = s2.dry_bundle(a_post)
        assert b_loaded["adapter_loaded"] is True
        assert b_loaded["checkpoint_sha256"] == s2.checkpoint_sha256(ckpt_dir)
        assert b_loaded["checkpoint_files_sha256"] == s2.checkpoint_file_shas(ckpt_dir)
        assert all(not p.requires_grad for p in b_loaded["model"].parameters())
        lora_b_vals = [
            float(p.abs().mean().item())
            for name, p in b_loaded["model"].named_parameters()
            if "lora_B" in name
        ]
        assert lora_b_vals and all(abs(v - 0.125) < 1e-5 for v in lora_b_vals)

        # 3. Refuse --postpass without --checkpoint (postpass_requires_checkpoint)
        a_no_ckpt = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(tmp_path / "no_ckpt"),
                "--rnd-commit",
                "deadbee",
                "--postpass",
                "--pass-seed",
                "2",
            ]
        )
        with pytest.raises(SystemExit, match="postpass_requires_checkpoint"):
            s2.validate_pass_mode_args(a_no_ckpt)

        # 4. Refuse corrupt/unloaded adapter in attach_or_load_adapter and in execute
        bad_ckpt = tmp_path / "bad_ckpt"
        bad_ckpt.mkdir(parents=True, exist_ok=True)
        (bad_ckpt / "not_an_adapter.txt").write_text("corrupt\n", encoding="utf-8")
        a_bad = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(tmp_path / "bad_out"),
                "--rnd-commit",
                "deadbee",
                "--dry",
                "--checkpoint",
                str(bad_ckpt),
                "--pass-seed",
                "2",
            ]
        )
        with pytest.raises(SystemExit, match="checkpoint_adapter_not_loaded"):
            s2.dry_bundle(a_bad)

        with pytest.raises(SystemExit, match="checkpoint_adapter_not_loaded"):
            s2.execute(a_post, build=lambda _a: {**b_init, "adapter_loaded": False})

    @needs_torch
    def test_item_02_save_adapter_at_end_of_training(self, tmp_path: Path) -> None:
        gold_dir = _prepass_gold(tmp_path / "gold")
        out_dir = tmp_path / "train_save_adapter"
        a = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(out_dir),
                "--rnd-commit",
                "deadbee",
                "--dry",
                "--steps",
                "1",
                "--prompts",
                "1",
                "--g",
                "2",
            ]
        )
        assert s2.execute(a, build=s2.dry_bundle) == 0
        adapter_dir = out_dir / "adapter"
        assert adapter_dir.is_dir()
        receipt = json.loads((out_dir / "s2-hf.json").read_text(encoding="utf-8"))
        assert receipt["adapter_sha256"] == s2.checkpoint_sha256(adapter_dir)
        assert receipt["adapter_files_sha256"] == s2.checkpoint_file_shas(adapter_dir)

        # Verify the saved adapter can be loaded into a subsequent --postpass run
        post_out = tmp_path / "post_from_saved"
        a_post = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(post_out),
                "--rnd-commit",
                "deadbee",
                "--dry",
                "--postpass",
                "--pass-seed",
                "2",
                "--checkpoint",
                str(adapter_dir),
                "--g",
                "2",
                "--max-new",
                "8",
            ]
        )
        assert s2.execute(a_post, build=s2.dry_bundle) == 0
        post_hdr = json.loads((post_out / "s2-hf-prepass-header.json").read_text(encoding="utf-8"))
        post_sum = json.loads((post_out / "s2-hf-prepass-summary.json").read_text(encoding="utf-8"))
        post_rec = json.loads((post_out / "s2-hf.json").read_text(encoding="utf-8"))
        assert post_hdr["checkpoint_sha256"] == receipt["adapter_sha256"]
        assert post_hdr["checkpoint_files_sha256"] == receipt["adapter_files_sha256"]
        assert post_sum["checkpoint_files_sha256"] == receipt["adapter_files_sha256"]
        assert post_rec["checkpoint_sha256"] == receipt["adapter_sha256"]
        assert post_rec["checkpoint_files_sha256"] == receipt["adapter_files_sha256"]

    def test_item_03_frozen_stack_versions_match_aspire_gpu(self) -> None:
        assert s2.FROZEN_STACK_VERSIONS == {
            "torch": "2.14.1+cu130",
            "transformers": "5.19.0",
            "bitsandbytes": "0.50.2",
            "peft": "0.21.2",
            "accelerate": "1.15.0",
        }
        assert (
            s2.FROZEN_PIP_FREEZE_SHA256
            == "27c98212c6cd9bd0303b089e76bed0f167cfeeec0eedd0cc9b0baf076c055324"
        )

    @needs_torch
    def test_item_04_always_check_frozen_env_on_non_dry_and_record_in_headers_and_receipts(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        gold_dir = _prepass_gold(tmp_path / "gold")
        # 1. Non-dry run without --require-frozen-env still calls check_frozen_env and refuses version mismatch
        a_nondry = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(tmp_path / "nondry_mismatch"),
                "--rnd-commit",
                "deadbee",
                "--rollout-only",
                "--g",
                "2",
                "--max-new",
                "8",
            ]
        )
        monkeypatch.setattr(
            s2,
            "FROZEN_STACK_VERSIONS",
            {**s2.FROZEN_STACK_VERSIONS, "transformers": "999.0.0"},
        )
        with pytest.raises(SystemExit, match="frozen_env_version_mismatch"):
            s2.execute(a_nondry, build=s2.real_bundle, rig=_FakeRig(a_nondry))

        # 2. Restore FROZEN_STACK_VERSIONS and verify stack_versions + pip_freeze_sha256 in header and receipt
        monkeypatch.undo()
        freeze_lines = [f"{pkg}=={ver}\n" for pkg, ver in sorted(s2.FROZEN_STACK_VERSIONS.items())]
        pf_path = tmp_path / "FROZEN-pip-freeze-2026-11-05.txt"
        pf_bytes = "".join(freeze_lines).encode("utf-8")
        pf_path.write_bytes(pf_bytes)
        pf_sha = s2.hashlib.sha256(pf_bytes).hexdigest()
        monkeypatch.setattr(s2, "FROZEN_PIP_FREEZE_SHA256", pf_sha)
        monkeypatch.setattr(s2, "package_version", lambda k: s2.FROZEN_STACK_VERSIONS.get(k))

        out_ro = tmp_path / "ro_with_freeze"
        a_ro = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(out_ro),
                "--rnd-commit",
                "deadbee",
                "--rollout-only",
                "--g",
                "2",
                "--max-new",
                "8",
                "--frozen-pip-freeze",
                str(pf_path),
            ]
        )
        assert s2.execute(a_ro, build=s2.dry_bundle, rig=_FakeRig(a_ro)) == 0
        hdr = json.loads((out_ro / "s2-hf-prepass-header.json").read_text(encoding="utf-8"))
        rec = json.loads((out_ro / "s2-hf.json").read_text(encoding="utf-8"))
        assert hdr["pip_freeze_sha256"] == pf_sha
        assert rec["pip_freeze_sha256"] == pf_sha
        assert set(hdr["stack_versions"].keys()) == set(s2.FROZEN_STACK_VERSIONS.keys())
        assert set(rec["stack_versions"].keys()) == set(s2.FROZEN_STACK_VERSIONS.keys())

    def test_item_05_prepass_and_paired_estimators_refuse_descriptive_and_wrong_seeds(
        self, tmp_path: Path
    ) -> None:
        from s2_draw import draw_prompts, split_stratified_holdout

        gold_dir = _draw_gold_40(tmp_path / "gold")
        rows, _ = s2.read_gold(gold_dir)
        tune_ids = [f"t{i:02d}" for i in range(40)]

        # 1. Prepass file with seed=1 header or record is refused by draw_prompts, split_stratified_holdout, and load_prepass_baseline
        seed1_dir = tmp_path / "seed1_pass"
        seed1_dir.mkdir(parents=True, exist_ok=True)
        seed1_jsonl = seed1_dir / "s2-hf-prepass.jsonl"
        seed1_sha = _write_fake_prepass(seed1_jsonl, tune_ids, k_fn=lambda i: i % 9, g=8)
        (seed1_dir / "s2-hf-prepass-header.json").write_text(
            json.dumps({"seed": 1, "g": 8}) + "\n", encoding="utf-8"
        )

        with pytest.raises(SystemExit, match="prepass_wrong_seed"):
            draw_prompts(
                rows,
                8,
                mode="kband",
                seed=42,
                prepass=(seed1_jsonl, seed1_sha),
                k_min=0,
                k_max=5,
            )
        with pytest.raises(SystemExit, match="prepass_wrong_seed"):
            split_stratified_holdout(
                rows,
                (seed1_jsonl, seed1_sha),
                holdout_seed="pr79-holdout-v1",
            )
        with pytest.raises(SystemExit, match="prepass_baseline_wrong_seed"):
            s2.load_prepass_baseline(seed1_jsonl, seed1_sha, tune_ids[:4], g=8)

        # Also check record-level pass_seed != 0 without header
        rec_seed2_jsonl = tmp_path / "rec_seed2.jsonl"
        rec_lines = [
            json.dumps({"id": cid, "g": 8, "k_correct": 4, "pass_seed": 2}) + "\n"
            for cid in tune_ids
        ]
        rec_bytes = "".join(rec_lines).encode("utf-8")
        rec_seed2_jsonl.write_bytes(rec_bytes)
        rec_seed2_sha = s2.hashlib.sha256(rec_bytes).hexdigest()
        with pytest.raises(SystemExit, match="prepass_wrong_seed"):
            draw_prompts(
                rows,
                8,
                mode="kband",
                seed=42,
                prepass=(rec_seed2_jsonl, rec_seed2_sha),
                k_min=0,
                k_max=5,
            )
        with pytest.raises(SystemExit, match="prepass_baseline_wrong_seed"):
            s2.load_prepass_baseline(rec_seed2_jsonl, rec_seed2_sha, tune_ids[:4], g=8)

        # 2. compute_paired_arm_estimators checks Pass A seed=0, Pass B seed=1, and post-passes non-reserved & not descriptive
        base_rec = {
            "id": "t00",
            "gold_verdict": "supported",
            "g": 8,
            "k_correct": 4,
            "supported": 4,
            "unsupported": 2,
            "cannot_tell": 2,
            "malformed": 0,
            "truncations": 0,
            "reward_mean": 0.2,
        }
        pa_ok = [{**base_rec, "pass_seed": 0}]
        pb_ok = [{**base_rec, "pass_seed": 1}]
        post2_ok = [{**base_rec, "pass_seed": 2}]
        post3_ok = [{**base_rec, "pass_seed": 3}]

        with pytest.raises(SystemExit, match="paired_estimator_wrong_seed"):
            s2.compute_paired_arm_estimators(
                [{**base_rec, "pass_seed": 1}], pb_ok, [(post2_ok, post3_ok)]
            )
        with pytest.raises(SystemExit, match="paired_estimator_wrong_seed"):
            s2.compute_paired_arm_estimators(
                pa_ok, [{**base_rec, "pass_seed": 0}], [(post2_ok, post3_ok)]
            )
        with pytest.raises(SystemExit, match="paired_estimator_wrong_seed"):
            s2.compute_paired_arm_estimators(
                pa_ok, pb_ok, [([{**base_rec, "pass_seed": 1}], post3_ok)]
            )
        with pytest.raises(SystemExit, match="paired_estimator_wrong_seed"):
            s2.compute_paired_arm_estimators(pa_ok, pb_ok, [(post2_ok, post2_ok)])
        with pytest.raises(SystemExit, match="paired_estimator_descriptive"):
            s2.compute_paired_arm_estimators(
                pa_ok, pb_ok, [([{**base_rec, "pass_seed": 2, "descriptive": True}], post3_ok)]
            )

    @needs_torch
    def test_item_06_guard_halts_before_optimiser_step_and_records_all_groups_in_step(self) -> None:
        from contextlib import nullcontext

        import torch

        class _TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros((1, 256), dtype=torch.float32))
                self.device = torch.device("cpu")

            def forward(self, input_ids, attention_mask=None, logits_to_keep=None):
                b, t = input_ids.shape
                k = logits_to_keep if logits_to_keep is not None else t
                logits = self.w.unsqueeze(0).expand(b, k, 256)
                return Namespace(logits=logits)

        tok = s2.DryTokenizer()
        prompts_u = [
            {
                "id": f"u_easy_{i}",
                "split": "tune",
                "gold_file": "grounded.jsonl",
                "claim": f"Unsupported claim {i}.",
                "label": "unsupported",
                "context": [{"title": "T", "sentences": ["alpha beta gamma"]}],
            }
            for i in range(4)
        ]
        opt_steps = {"count": 0}
        zero_grads = {"count": 0}
        guard = s2.CtCollapseGuard(
            window_groups=2,
            pass_a_k={f"u_easy_{i}": 8 for i in range(4)},
        )
        steps_out, stopped_out, oom_out = s2.run_steps(
            model=_TinyModel(),
            ref_adapter_off=nullcontext,
            tok=tok,
            optimiser=Namespace(
                zero_grad=lambda set_to_none=True: zero_grads.__setitem__("count", zero_grads["count"] + 1),
                step=lambda: opt_steps.__setitem__("count", opt_steps["count"] + 1),
            ),
            prompts=prompts_u,
            rollout=lambda text, g_cnt, max_n, temp: (
                tok.encode(text),
                [
                    tok.encode(_reply("cannot_tell", "alpha beta gamma")) + [tok.eos_token_id],
                    tok.encode(_reply("unsupported", "alpha beta gamma")) + [tok.eos_token_id],
                ][:g_cnt],
                [False] * g_cnt,
            ),
            steps=2,
            g=2,
            max_new=32,
            reward_fn=lambda text, row, tt, truncated=False: s2.reward(
                text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
            ),
            device_sync=lambda: None,
            peak_alloc_mib=lambda: 100,
            groups_per_update=3,
            ct_guard=guard,
        )
        assert not oom_out
        assert len(steps_out) == 1
        assert stopped_out is not None
        assert opt_steps["count"] == 0
        assert zero_grads["count"] >= 2
        assert len(guard.records) == 3

    @needs_torch
    def test_item_07_guard_computes_ct_pos_adv_on_unclamped_advantages(self) -> None:
        from contextlib import nullcontext

        import torch

        class _TinyModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.w = torch.nn.Parameter(torch.zeros((1, 256), dtype=torch.float32))
                self.device = torch.device("cpu")

            def forward(self, input_ids, attention_mask=None, logits_to_keep=None):
                b, t = input_ids.shape
                k = logits_to_keep if logits_to_keep is not None else t
                logits = self.w.unsqueeze(0).expand(b, k, 256)
                return Namespace(logits=logits)

        tok = s2.DryTokenizer()
        prompts_sup = [
            {
                "id": f"sup_item_{i}",
                "split": "tune",
                "gold_file": "grounded.jsonl",
                "claim": f"Supported claim {i}.",
                "label": "supported",
                "context": [{"title": "T", "sentences": ["alpha beta gamma"]}],
            }
            for i in range(4)
        ]
        # On gold supported: 1 completion is 'cannot_tell' (-0.3) and 3 are 'unsupported' (-1.0).
        # Raw unclamped advantage on 'cannot_tell' is positive (-0.3 > mean -0.825),
        # while clamp_wrong_positive_advantages clamps it to 0.0.
        # S1 must still see ct_pos_adv=True on the unclamped advantages!
        guard = s2.CtCollapseGuard(window_groups=4)
        _, stopped_out, _ = s2.run_steps(
            model=_TinyModel(),
            ref_adapter_off=nullcontext,
            tok=tok,
            optimiser=Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
            prompts=prompts_sup,
            rollout=lambda text, g_cnt, max_n, temp: (
                tok.encode(text),
                [
                    tok.encode(_reply("cannot_tell", "alpha beta gamma")) + [tok.eos_token_id],
                    tok.encode(_reply("unsupported", "alpha beta gamma")) + [tok.eos_token_id],
                    tok.encode(_reply("unsupported", "alpha beta gamma")) + [tok.eos_token_id],
                    tok.encode(_reply("unsupported", "alpha beta gamma")) + [tok.eos_token_id],
                ][:g_cnt],
                [False] * g_cnt,
            ),
            steps=2,
            g=4,
            max_new=32,
            reward_fn=lambda text, row, tt, truncated=False: s2.reward(
                text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
            ),
            device_sync=lambda: None,
            peak_alloc_mib=lambda: 100,
            groups_per_update=2,
            mask_wrong_positive_advantage=True,
            ct_guard=guard,
        )
        assert stopped_out is not None
        assert "S1" in stopped_out["triggered_rules"]
        assert all(r["ct_pos_adv"] is True for r in guard.records)

    @needs_torch
    def test_item_08_u2_loads_pass_a_k_from_prepass_jsonl_and_refuses_without_pass_a_k(
        self, tmp_path: Path
    ) -> None:
        gold_dir = _draw_gold_40(tmp_path / "gold")
        tune_ids = [f"t{i:02d}" for i in range(40)]
        prepass_path = tmp_path / "pass_a.jsonl"
        prepass_sha = _write_fake_prepass(prepass_path, tune_ids, k_fn=lambda _i: 7, g=8)

        # 1. Refuses --ct-guard when neither --pass-a-jsonl nor --prepass-jsonl is given
        a_no_pa = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(tmp_path / "no_pa"),
                "--rnd-commit",
                "deadbee",
                "--dry",
                "--steps",
                "1",
                "--prompts",
                "2",
                "--g",
                "2",
                "--ct-guard",
            ]
        )
        with pytest.raises(SystemExit, match="ct_guard_requires_pass_a_k"):
            s2.execute(a_no_pa, build=s2.dry_bundle)

        # 2. Loads pass_a_k from --prepass-jsonl when --pass-a-jsonl is not separately passed
        a_with_prepass = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(tmp_path / "with_prepass"),
                "--rnd-commit",
                "deadbee",
                "--dry",
                "--steps",
                "1",
                "--prompts",
                "2",
                "--g",
                "2",
                "--draw",
                "class_stratified",
                "--draw-seed",
                "42",
                "--holdout-seed",
                "pr79-holdout-v1",
                "--prepass-jsonl",
                str(prepass_path),
                "--prepass-sha256",
                prepass_sha,
                "--ct-guard",
            ]
        )
        assert s2.execute(a_with_prepass, build=s2.dry_bundle) == 0

    def test_item_09_prepass_item_record_does_not_count_truncations_in_malformed(self) -> None:
        tok = s2.DryTokenizer()
        row = {
            "id": "item_trunc",
            "gold_file": "grounded.jsonl",
            "label": "supported",
            "context": [{"title": "T", "sentences": ["alpha beta gamma"]}],
        }
        c_ok = tok.encode(_reply("supported", "alpha beta gamma"))
        c_trunc = tok.encode("<think>unclosed thinking that got truncated")
        rec, _ = s2.prepass_item_record(
            row=row,
            prompt_ids=[1, 2, 3],
            comps=[c_ok, c_trunc],
            truncs=[False, True],
            tok=tok,
            reward_fn=lambda text, r, tt, truncated=False: s2.reward(
                text, r["label"], s2.passages_of(r), tt, truncated=truncated
            ),
        )
        assert rec["truncations"] == 1
        assert rec["malformed"] == 0
        assert (
            rec["supported"]
            + rec["unsupported"]
            + rec["cannot_tell"]
            + rec["malformed"]
            + rec["truncations"]
            == rec["g"]
        )

    def test_item_10_paired_estimators_by_band_and_gold_class_and_cli_entry(
        self, tmp_path: Path
    ) -> None:
        # 3 bands x 3 gold classes = 9 items
        items = []
        idx = 0
        for band_k in (2, 6, 8):
            for gv in ("supported", "unsupported", "cannot_tell"):
                items.append((f"id_{idx}", gv, band_k))
                idx += 1

        def _make_records(seed_val: int, delta_k: int = 0) -> list[dict]:
            out = []
            for cid, gv, bk in items:
                k_c = min(8, max(0, bk + delta_k))
                out.append(
                    {
                        "id": cid,
                        "gold_verdict": gv,
                        "g": 8,
                        "k_correct": k_c,
                        "supported": k_c if gv == "supported" else (8 - k_c),
                        "unsupported": k_c if gv == "unsupported" else 0,
                        "cannot_tell": k_c if gv == "cannot_tell" else 0,
                        "malformed": 0,
                        "truncations": 0,
                        "reward_mean": 0.5,
                        "pass_seed": seed_val, "descriptive": False,
                    }
                )
            return out

        pa = _make_records(0, 0)
        pb = _make_records(1, 0)
        p2 = _make_records(2, 1)
        p3 = _make_records(3, 1)

        doc = s2.compute_paired_arm_estimators(pa, pb, [(p2, p3)])
        assert "by_band_and_gold_class" in doc
        for band in ("[0,5]", "[6,7]", "8"):
            assert set(doc["by_band_and_gold_class"][band].keys()) == {
                "supported",
                "unsupported",
                "cannot_tell",
            }
            for gv in ("supported", "unsupported", "cannot_tell"):
                cell = doc["by_band_and_gold_class"][band][gv]
                assert cell["n_items"] == 1
                assert cell["estimator"] == "post_mean - B"

        # Verify CLI entry --paired-estimators writes s2-paired-estimators.json
        def _write_pass_dir(name: str, seed_val: int, recs: list[dict]) -> dict[str, str]:
            d = tmp_path / name
            d.mkdir(parents=True, exist_ok=True)
            jpath = d / "s2-hf-prepass.jsonl"
            raw = "".join(json.dumps(r) + "\n" for r in recs).encode("utf-8")
            jpath.write_bytes(raw)
            (d / "s2-hf-prepass-header.json").write_text(
                json.dumps({"seed": seed_val, "g": 8}) + "\n", encoding="utf-8"
            )
            return {"jsonl": str(jpath), "sha256": s2.hashlib.sha256(raw).hexdigest()}

        spec_path = tmp_path / "paired_spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "pass_a": _write_pass_dir("pa", 0, pa),
                    "pass_b": _write_pass_dir("pb", 1, pb),
                    "seed_postpasses": [[_write_pass_dir("p2", 2, p2), _write_pass_dir("p3", 3, p3)]],
                }
            ),
            encoding="utf-8",
        )
        out_cli = tmp_path / "paired_out"
        a_cli = s2.build_parser().parse_args(
            [
                "--gold",
                str(tmp_path),
                "--rnd-commit",
                "deadbee",
                "--out",
                str(out_cli),
                "--paired-estimators",
                str(spec_path),
            ]
        )
        assert s2.execute(a_cli, build=s2.dry_bundle) == 0
        written = json.loads((out_cli / "s2-paired-estimators.json").read_text(encoding="utf-8"))
        assert written["by_band_and_gold_class"] == doc["by_band_and_gold_class"]

    @needs_torch
    def test_item_11_class_stratified_total_groups_and_explicit_holdout_seed_required(
        self, tmp_path: Path
    ) -> None:
        from s2_draw import draw_prompts, split_stratified_holdout

        gold_dir = _draw_gold_40(tmp_path / "gold")
        rows, _ = s2.read_gold(gold_dir)
        tune_ids = [f"t{i:02d}" for i in range(40)]
        prepass_path = tmp_path / "pass_a.jsonl"
        prepass_sha = _write_fake_prepass(prepass_path, tune_ids, k_fn=lambda i: i % 9, g=8)

        # 1. Missing holdout_seed is refused by split_stratified_holdout and draw_prompts
        with pytest.raises(SystemExit, match="missing_holdout_seed"):
            split_stratified_holdout(rows, (prepass_path, prepass_sha))
        with pytest.raises(SystemExit, match="missing_holdout_seed"):
            draw_prompts(
                rows,
                10,
                mode="class_stratified",
                seed=42,
                prepass=(prepass_path, prepass_sha),
            )

        # 2. Under --draw class_stratified, explicit --prompts < steps * groups_per_update is refused
        a_too_small = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(tmp_path / "too_small"),
                "--rnd-commit",
                "deadbee",
                "--dry",
                "--steps",
                "5",
                "--groups-per-update",
                "2",
                "--prompts",
                "6",
                "--g",
                "2",
                "--draw",
                "class_stratified",
                "--draw-seed",
                "42",
                "--holdout-seed",
                "pr79-holdout-v1",
                "--prepass-jsonl",
                str(prepass_path),
                "--prepass-sha256",
                prepass_sha,
            ]
        )
        with pytest.raises(SystemExit, match="prompts_smaller_than_total_groups"):
            s2.execute(a_too_small, build=s2.dry_bundle)

        # 3. When --prompts is omitted under --draw class_stratified, it defaults to steps * groups_per_update
        out_ok = tmp_path / "cs_default_prompts"
        a_ok = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--out",
                str(out_ok),
                "--rnd-commit",
                "deadbee",
                "--dry",
                "--steps",
                "3",
                "--groups-per-update",
                "2",
                "--g",
                "2",
                "--draw",
                "class_stratified",
                "--draw-seed",
                "42",
                "--holdout-seed",
                "pr79-holdout-v1",
                "--prepass-jsonl",
                str(prepass_path),
                "--prepass-sha256",
                prepass_sha,
            ]
        )
        assert s2.execute(a_ok, build=s2.dry_bundle) == 0
        rec = json.loads((out_ok / "s2-hf.json").read_text(encoding="utf-8"))
        assert rec["draw_n"] == 6
        assert len(rec["prompt_ids"]) == 6
        assert rec["holdout_seed"] == "pr79-holdout-v1"

    def test_item_12_interview_question_text_matches_design_v2(self):
        import s2_interview as s2i

        expected = {
            "Q1": (
                "Which part of the claim could you not decide from the evidence? "
                "Copy the exact words of that part from the claim."
            ),
            "Q4": (
                "Copy exactly one line from the evidence that is closest to deciding that part, "
                "or write NONE."
            ),
            "Q2": (
                "What would the evidence need to say for you to decide it? "
                "End with one line: "
                "MISSING_FACT: <one sentence>."
            ),
            "Q3": (
                "If you had to choose between {A} and {B}, "
                "which would you choose, and why?"
            ),
        }
        assert s2i.INTERVIEW_QUESTIONS == expected
        assert s2i.COLD_CONTROL_QUESTIONS == {
            "Q1": expected["Q1"],
            "Q2": expected["Q2"],
        }
        # Q3 option order comes from item hash, with nothing hard-coded in format line
        q3_text = s2i.format_interview_question("Q3", "item_abc")
        assert "supported | unsupported" not in q3_text
        # Also verify natural prose and LEAN: both parse cleanly
        assert s2i.parse_q3_lean("I would lean toward supported because the passage agrees.")["lean"] == "supported"
        assert s2i.parse_q3_lean("LEAN: unsupported — contradicts the date.")["lean"] == "unsupported"

    @needs_torch
    def test_item_13_interview_seed_uses_item_id_and_question_id_only(self):
        import hashlib
        import inspect

        import s2_interview as s2i

        s_base = s2i.interview_seed("item_42", "Q1", base_seed=7)
        sig = inspect.signature(s2i.interview_seed)
        assert "policy" not in sig.parameters
        digest = hashlib.sha256(b"interview:7:item_42:Q1").digest()
        expected = int.from_bytes(digest[:8], "big") & 0x7FFF_FFFF_FFFF_FFFF
        assert s_base == expected

    @needs_torch
    def test_item_14_stage1_stops_at_end_think_and_slices_answer_to_answer_cap(self):
        import s2_interview as s2i
        import torch

        tok = s2i.Qwen3ChatStubTokenizer()

        class _CaptureEosAndOverflowAnswerModel:
            device = torch.device("cpu")

            def __init__(self):
                self.stage1_eos = None

            def generate(self, input_ids, max_new_tokens, **kwargs):
                if self.stage1_eos is None:
                    self.stage1_eos = kwargs.get("eos_token_id")
                    # Simulate model emitting </think> followed by 50 answer tokens in Stage 1
                    seq = (
                        tok.encode("brief thought")
                        + [s2i.QWEN3_END_THINK_TOKEN_ID]
                        + [20 + (i % 50) for i in range(50)]
                    )
                    return torch.tensor([list(input_ids[0].tolist()) + seq], dtype=torch.long)
                seq2 = [30 + (i % 50) for i in range(max_new_tokens)]
                return torch.tensor([list(input_ids[0].tolist()) + seq2], dtype=torch.long)

        m = _CaptureEosAndOverflowAnswerModel()
        res = s2i.generate_two_stage_thinking(
            m,
            tok,
            [10, 11, 12],
            seed=11,
            thinking_cap=64,
            answer_cap=10,
        )
        eos_set = set(m.stage1_eos) if isinstance(m.stage1_eos, (list, tuple)) else {m.stage1_eos}
        assert s2i.QWEN3_END_THINK_TOKEN_ID in eos_set
        assert tok.eos_token_id in eos_set
        assert res["answer_tokens"] <= 10

    @needs_torch
    def test_item_15_cold_control_omits_json_contract_and_runs_both_policies(self, tmp_path):
        import s2_interview as s2i

        row = _row("cold_item_1", claim="The cache TTL is 60 seconds.", text="A local cache is used.")
        msgs = s2i.build_cold_control_messages(row, "Q1")
        user_text = msgs[0]["content"]
        assert "The cache TTL is 60 seconds." in user_text
        assert "A local cache is used." in user_text
        assert "evidence_quotes" not in user_text
        assert '{"verdict"' not in user_text

        import torch

        init_model, _ = s2i.build_cpu_qwen3_stub(with_adapter=True)
        with torch.no_grad():
            for name, param in init_model.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.05)
        ckpt_dir = tmp_path / "cold_ckpt"
        init_model.save_pretrained(str(ckpt_dir))
        ckpt_sha = s2.checkpoint_sha256(ckpt_dir)
        model, tok = s2i.build_cpu_qwen3_stub(checkpoint_path=ckpt_dir)
        policy_seen: list[str] = []
        runner = s2i.InterviewModelRunner(model, tok, expected_checkpoint_sha256=ckpt_sha)
        orig_ctx = runner._policy_context

        def _spy_ctx(pol: str, *, expected_checkpoint_sha256: str | None = None):
            policy_seen.append(pol)
            return orig_ctx(pol, expected_checkpoint_sha256=expected_checkpoint_sha256)

        runner._policy_context = _spy_ctx  # type: ignore[method-assign]
        t1 = (
            "<think>missing TTL</think>"
            + json.dumps(
                {
                    "verdict": "cannot_tell",
                    "reason": "missing",
                    "evidence_quotes": [{"part": "TTL is 60 seconds", "verdict": "cannot_tell", "quote": ""}],
                }
            )
        )
        cohort = s2i.select_interview_cohort(
            [{"item_id": "cold_item_1", "sample_index": 0, "verdict": "cannot_tell", "text": t1}],
            [{"item_id": "cold_item_1", "sample_index": 0, "verdict": "cannot_tell", "text": t1}],
        )
        out_dir = tmp_path / "cold_both"
        s2i.run_interview_for_cohort(
            cohort=cohort,
            rows_by_id={"cold_item_1": row},
            runner=runner,
            out_dir=out_dir,
            base_seed=1,
            thinking_cap=4,
            answer_cap=4,
            include_cold_control=True,
        )
        recs = s2.read_save_all(out_dir / "s2-interview-save-all.jsonl.gz")
        cold_recs = [r for r in recs if r.get("cold_control") is True or str(r.get("policy", "")).startswith("cold_control")]
        cold_policies = {r.get("policy") for r in cold_recs}
        assert len(cold_recs) == 4
        assert len(cold_policies) == 2

    def test_item_16_part_match_f1_containment_fallback_and_unspecific_full_claim_copy(self):
        import s2_interview as s2i

        claim = (
            "The authentication service rotates signing keys every twenty four hours "
            "and stores backup keys in an offline hardware security module."
        )
        # Q1 has 4 tokens ("stores backup keys module"), JSON has 6 tokens ("stores backup keys in offline module")
        # Intersection = 4 -> |Q1 ∩ JSON| / |Q1| = 4/4 = 1.0, while |Q1 ∩ JSON| / |JSON| = 4/6 = 0.666667
        q1_short = "stores backup keys module"
        json_part = ["stores backup keys in offline module"]
        sc = s2i.score_part_match(q1_short, json_part, claim)
        assert sc["containment"] == pytest.approx(1.0)
        assert sc["token_f1"] == pytest.approx(0.8)
        assert "unspecific_full_claim_copy" in sc
        assert "f1_at_0_3" in sc and "f1_at_0_5" in sc and "f1_at_0_7" in sc
        assert "containment_at_0_3" in sc and "containment_at_0_5" in sc and "containment_at_0_7" in sc

        # Fallback flag when JSON has no cannot_tell entry in evidence_quotes
        t1_no_ct_part = json.dumps(
            {
                "verdict": "cannot_tell",
                "reason": "unsure",
                "evidence_quotes": [{"part": "rotates signing keys", "verdict": "supported", "quote": "rotates"}],
            }
        )
        parts_res = s2i.extract_ct_parts_from_json(t1_no_ct_part)
        sc_fb = s2i.score_part_match(q1_short, parts_res, claim)
        assert sc_fb.get("ct_part_fallback") is True

    @needs_torch
    def test_item_17_counterfactual_runs_by_default_with_interviewed_policy_and_writes_summary(
        self, tmp_path
    ):
        import s2_interview as s2i

        model, tok = s2i.build_cpu_qwen3_stub()
        runner = s2i.InterviewModelRunner(model, tok)
        row = _row("cf_def_1", claim="The retry limit is 5.", text="Requests are retried.")
        t1 = (
            "<think>retry limit missing</think>"
            + json.dumps(
                {
                    "verdict": "cannot_tell",
                    "reason": "missing",
                    "evidence_quotes": [{"part": "retry limit is 5", "verdict": "cannot_tell", "quote": ""}],
                }
            )
        )
        cohort = s2i.select_interview_cohort(
            [{"item_id": "cf_def_1", "sample_index": 0, "verdict": "cannot_tell", "text": t1}],
            [],
        )
        out_dir = tmp_path / "cf_default"
        receipt = s2i.run_interview_for_cohort(
            cohort=cohort,
            rows_by_id={"cf_def_1": row},
            runner=runner,
            out_dir=out_dir,
            base_seed=9,
            thinking_cap=4,
            answer_cap=4,
            include_cold_control=False,
        )
        assert "counterfactual_summary" in receipt
        recs = s2.read_save_all(out_dir / "s2-interview-save-all.jsonl.gz")
        q2_rec = next(r for r in recs if r["question_id"] == "Q2")
        assert "left_cannot_tell" in q2_rec["mechanical_score"]
        assert "direction" in q2_rec["mechanical_score"]

    @needs_torch
    def test_item_18_majority_ct_uses_half_or_more_and_copies_to_turn_records(self, tmp_path):
        import s2_interview as s2i

        t1 = (
            "<think>missing</think>"
            + json.dumps(
                {
                    "verdict": "cannot_tell",
                    "reason": "missing",
                    "evidence_quotes": [{"part": "x", "verdict": "cannot_tell", "quote": ""}],
                }
            )
        )
        t_sup = (
            "<think>ok</think>"
            + json.dumps(
                {
                    "verdict": "supported",
                    "reason": "ok",
                    "evidence_quotes": [{"part": "x", "verdict": "supported", "quote": "x"}],
                }
            )
        )
        # 4 of 8 = 0.5 -> majority_ct must be True under 2*k >= K
        base_recs = [
            {"item_id": "maj_1", "sample_index": i, "verdict": "cannot_tell" if i < 4 else "supported", "text": t1 if i < 4 else t_sup}
            for i in range(8)
        ]
        cohort = s2i.select_interview_cohort(base_recs, [])
        assert cohort[0]["baseline_majority_ct"] is True

        model, tok = s2i.build_cpu_qwen3_stub()
        runner = s2i.InterviewModelRunner(model, tok)
        out_dir = tmp_path / "maj_out"
        s2i.run_interview_for_cohort(
            cohort=cohort,
            rows_by_id={"maj_1": _row("maj_1", claim="x", text="x")},
            runner=runner,
            out_dir=out_dir,
            base_seed=3,
            thinking_cap=4,
            answer_cap=4,
            include_cold_control=False,
        )
        recs = s2.read_save_all(out_dir / "s2-interview-save-all.jsonl.gz")
        for r in recs:
            assert "ct_rate" in r
            assert "majority_ct" in r
            assert r["ct_rate"] == pytest.approx(0.5)
            assert r["majority_ct"] is True

    def test_item_19_q4_is_none_requires_exact_none_not_prefix(self):
        import s2_interview as s2i

        t1 = json.dumps(
            {
                "verdict": "cannot_tell",
                "reason": "missing",
                "evidence_quotes": [{"part": "x", "verdict": "cannot_tell", "quote": ""}],
            }
        )
        sc = s2i.score_quote_validity_and_consistency(
            "None of the passages state the exact latency.", ["Passage one."], t1
        )
        assert sc["is_none"] is False
        assert sc["quote_valid"] is False

        sc_exact = s2i.score_quote_validity_and_consistency("  NONE. ", ["Passage one."], t1)
        assert sc_exact["is_none"] is True
        assert sc_exact["quote_valid"] is True

    def test_item_20_q3_unparsed_answers_count_as_wrong_in_recall_and_balanced_accuracy(self):
        import s2_interview as s2i

        r_sup_ok = s2i.score_q3_lean_item("LEAN: supported — yes", "supported", "id_1")
        r_sup_unparsed = s2i.score_q3_lean_item("I refuse to guess.", "supported", "id_2")
        r_unsup_ok = s2i.score_q3_lean_item("LEAN: unsupported — no", "unsupported", "id_3")

        summary = s2i.summarize_q3_lean([r_sup_ok, r_sup_unparsed, r_unsup_ok])
        assert summary["unparsed_count"] == 1
        # supported recall is 1/2 = 0.5 (unparsed counts in denominator), unsupported recall is 1/1 = 1.0 -> bal_acc = 0.75
        assert summary["decidable_supported_acc"] == pytest.approx(0.5)
        assert summary["balanced_accuracy"] == pytest.approx(0.75)

    @needs_torch
    def test_item_21_tagger_tags_q2_requires_turn1_thinking_truly_empty_controls_and_hash_only_tags_jsonl(
        self, tmp_path
    ):
        import s2_interview as s2i
        import s2_interview_tag as s2t

        # 1. All 15 empty_thinking controls must have truly empty thinking (thinking.strip() == "")
        controls = s2t.build_judge_controls([])
        assert len(controls["empty_thinking"]) == 15
        assert all(c["thinking"].strip() == "" for c in controls["empty_thinking"])

        # 2. InterviewModelRunner.ask_question stores turn1_thinking on non-cold records
        model, tok = s2i.build_cpu_qwen3_stub()
        runner = s2i.InterviewModelRunner(model, tok)
        row = _row("tag_item_1", claim="The cache size is 16 MB.", text="A local cache is configured.")
        t1 = (
            "<think>Original Turn-1 secret thinking about 16 MB cache size.</think>"
            + json.dumps(
                {
                    "verdict": "cannot_tell",
                    "reason": "size missing",
                    "evidence_quotes": [{"part": "cache size is 16 MB", "verdict": "cannot_tell", "quote": ""}],
                }
            )
        )
        q2_turn = runner.ask_question(
            row=row,
            turn1_completion_text=t1,
            question_id="Q2",
            policy="baseline",
            base_seed=5,
            thinking_cap=4,
            answer_cap=4,
        )
        assert q2_turn.get("turn1_thinking") == "Original Turn-1 secret thinking about 16 MB cache size."

        # 3. build_tagging_prompt compares Turn 1 thinking against Turn 2 Q2 answer
        prompt_str = s2t.build_tagging_prompt("turn 1 trace", "Q2 missing fact answer")
        assert "Turn 1 <think>" in prompt_str
        assert "Turn 2 Q2 answer" in prompt_str

        def accurate_judge(thinking: str, q2_answer: str) -> dict[str, str]:
            t_clean = thinking.strip()
            if not t_clean:
                return {"missing_element_match": "none", "reason_type": "generic_or_none"}
            if (
                t_clean in q2_answer
                or (
                    "release date" in t_clean
                    and "release date" in q2_answer
                    and t_clean.split()[-1].rstrip(".") in q2_answer
                )
            ):
                return {"missing_element_match": "same", "reason_type": "lacks"}
            return {"missing_element_match": "different", "reason_type": "conflicts"}

        # 4. Refuses when turn1_thinking is missing (never falls back to Turn 2's thinking)
        with pytest.raises(SystemExit, match="missing_turn1_thinking"):
            s2t.run_two_judge_tagging(
                interview_q2_records=[
                    {
                        "item_id": "bad_1",
                        "policy": "baseline",
                        "question_id": "Q2",
                        "thinking": "Turn 2 thinking only",
                        "answer": "MISSING_FACT: x",
                    }
                ],
                judge_a_name="llama-3.3-70b",
                judge_b_name="mistral-large",
                judge_a_fn=accurate_judge,
                judge_b_fn=accurate_judge,
                out_dir=tmp_path / "bad_no_t1",
                controls=controls,
            )

        with pytest.raises(SystemExit, match="missing_turn1_thinking"):
            s2t.export_blinded_human_audit(
                [{"item_id": "bad_1", "policy": "baseline", "q2_answer": "x"}],
                tmp_path / "bad_audit",
            )

        with pytest.raises(SystemExit, match="missing_q2_records"):
            s2t.run_two_judge_tagging(
                interview_q2_records=[
                    {
                        "item_id": "only_q1",
                        "policy": "baseline",
                        "question_id": "Q1",
                        "turn1_thinking": "trace",
                        "answer": "part",
                    }
                ],
                judge_a_name="llama-3.3-70b",
                judge_b_name="mistral-large",
                judge_a_fn=accurate_judge,
                judge_b_fn=accurate_judge,
                out_dir=tmp_path / "bad_no_q2",
                controls=controls,
            )

        # 5. Tags Q2 records (filtering out Q1/cold records when passed a mixed save-all list)
        # and writes ONLY ids, tags, token counts, and sha256 hashes (no raw text) to s2-interview-tags.jsonl
        mixed_records = [
            {
                "item_id": "item_01",
                "policy": "baseline",
                "question_id": "Q1",
                "turn1_thinking": "RAW_TURN1_THINKING_SECRET_TEXT",
                "answer": "RAW_Q1_ANSWER_SHOULD_NOT_BE_TAGGED",
                "original_turn1_thinking_tokens": 12,
            },
            {
                "item_id": "item_01",
                "policy": "baseline",
                "question_id": "Q2",
                "turn1_thinking": "RAW_TURN1_THINKING_SECRET_TEXT",
                "answer": "As noted in the reasoning, RAW_TURN1_THINKING_SECRET_TEXT\nMISSING_FACT: fact",
                "original_turn1_thinking_tokens": 12,
            },
            {
                "item_id": "item_01",
                "policy": "cold_control:baseline",
                "question_id": "Q2",
                "cold_control": True,
                "turn1_thinking": None,
                "answer": "COLD_Q2_SHOULD_BE_SKIPPED",
                "original_turn1_thinking_tokens": 0,
            },
        ]
        out_ok = tmp_path / "tag_ok"
        res = s2t.run_two_judge_tagging(
            interview_q2_records=mixed_records,
            judge_a_name="llama-3.3-70b",
            judge_b_name="mistral-large",
            judge_a_fn=accurate_judge,
            judge_b_fn=accurate_judge,
            out_dir=out_ok,
            controls=controls,
        )
        assert res["tagged_question_id"] == "Q2"
        assert res["agreement_missing_element_match"]["n"] == 1
        tags_raw = (out_ok / "s2-interview-tags.jsonl").read_text(encoding="utf-8")
        assert "RAW_TURN1_THINKING_SECRET_TEXT" not in tags_raw
        assert "MISSING_FACT: fact" not in tags_raw
        tag_row = json.loads(tags_raw.splitlines()[0])
        assert tag_row["question_id"] == "Q2"
        assert "turn1_thinking_sha256" in tag_row
        assert "q2_answer_sha256" in tag_row
        assert "turn1_thinking" not in tag_row
        assert "q1_answer" not in tag_row
        assert "q2_answer" not in tag_row


class TestFix2Items:
    @needs_torch
    def test_fix2_item_01_checkpoint_policy_requires_loaded_adapter(self, tmp_path: Path) -> None:
        import s2_interview as s2i

        row = _row("ckpt_item_1", claim="The buffer is 64 bytes.", text="A local buffer is allocated.")
        t1 = (
            "<think>buffer size missing</think>"
            + json.dumps(
                {
                    "verdict": "cannot_tell",
                    "reason": "missing",
                    "evidence_quotes": [{"part": "buffer is 64 bytes", "verdict": "cannot_tell", "quote": ""}],
                }
            )
        )
        bare_model, tok = s2i.build_cpu_qwen3_stub()
        bare_runner = s2i.InterviewModelRunner(bare_model, tok)

        # Bare model succeeds for baseline and cold_control:baseline
        res_base = bare_runner.ask_question(
            row=row,
            turn1_completion_text=t1,
            question_id="Q1",
            policy="baseline",
            base_seed=1,
            thinking_cap=4,
            answer_cap=4,
        )
        assert res_base["policy"] == "baseline"
        res_cold_base = bare_runner.ask_question(
            row=row,
            turn1_completion_text=None,
            question_id="Q1",
            policy="cold_control:baseline",
            cold_control=True,
            base_seed=1,
            thinking_cap=4,
            answer_cap=4,
        )
        assert res_cold_base["cold_control"] is True

        # Bare model (no PEFT adapter loaded) must refuse checkpoint and cold_control:checkpoint
        with pytest.raises(SystemExit, match="checkpoint_adapter_not_loaded"):
            bare_runner.ask_question(
                row=row,
                turn1_completion_text=t1,
                question_id="Q1",
                policy="checkpoint",
                base_seed=1,
                thinking_cap=4,
                answer_cap=4,
            )
        with pytest.raises(SystemExit, match="checkpoint_adapter_not_loaded"):
            bare_runner.ask_question(
                row=row,
                turn1_completion_text=None,
                question_id="Q1",
                policy="cold_control:checkpoint",
                cold_control=True,
                base_seed=1,
                thinking_cap=4,
                answer_cap=4,
            )

        # Model with loaded PEFT adapter succeeds on baseline, checkpoint, and both cold controls
        import torch

        init_model, _ = s2i.build_cpu_qwen3_stub(with_adapter=True)
        with torch.no_grad():
            for name, param in init_model.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.05)
        ckpt_dir = tmp_path / "fix2_ckpt"
        init_model.save_pretrained(str(ckpt_dir))
        ckpt_sha = s2.checkpoint_sha256(ckpt_dir)
        peft_model, peft_tok = s2i.build_cpu_qwen3_stub(checkpoint_path=ckpt_dir)
        peft_runner = s2i.InterviewModelRunner(
            peft_model, peft_tok, expected_checkpoint_sha256=ckpt_sha
        )
        for pol in ("baseline", "checkpoint", "cold_control:baseline", "cold_control:checkpoint"):
            is_cold = pol.startswith("cold_control:")
            out = peft_runner.ask_question(
                row=row,
                turn1_completion_text=None if is_cold else t1,
                question_id="Q1",
                policy=pol,
                cold_control=is_cold,
                base_seed=2,
                thinking_cap=4,
                answer_cap=4,
            )
            assert out["policy"] == pol
        with pytest.raises(SystemExit, match="checkpoint_adapter_not_loaded"):
            peft_runner.ask_question(
                row=row,
                turn1_completion_text=t1,
                question_id="Q1",
                policy="checkpoint:missing_adapter",
                base_seed=2,
                thinking_cap=4,
                answer_cap=4,
            )

    @needs_torch
    def test_fix2_item_02_counterfactual_works_end_to_end_with_turn1_budget(self, tmp_path: Path, monkeypatch) -> None:
        import s2_interview as s2i

        model, tok = s2i.build_cpu_qwen3_stub()
        runner = s2i.InterviewModelRunner(model, tok)
        row = _row("cf_e2e_1", claim="The retry limit is 5.", text="Requests are retried on failure.")

        # 1. Direct run_verifier call must return generated text without KeyError: 'raw_text'
        direct_reply = runner.run_verifier(row, seed=7, policy="baseline", thinking_cap=4, answer_cap=4)
        assert isinstance(direct_reply, str) and len(direct_reply) > 0

        # 2. End-to-end cohort run where Q2 emits MISSING_FACT: <sentence>
        import torch

        init_model, _ = s2i.build_cpu_qwen3_stub(with_adapter=True)
        with torch.no_grad():
            for name, param in init_model.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.05)
        ckpt_dir = tmp_path / "cf_ckpt"
        init_model.save_pretrained(str(ckpt_dir))
        ckpt_sha = s2.checkpoint_sha256(ckpt_dir)
        peft_model, peft_tok = s2i.build_cpu_qwen3_stub(checkpoint_path=ckpt_dir)
        peft_runner = s2i.InterviewModelRunner(
            peft_model, peft_tok, expected_checkpoint_sha256=ckpt_sha
        )
        seen_verifier_calls: list[tuple[str, int, int]] = []
        orig_gen = s2i.generate_two_stage_thinking

        def scripted_gen(m, tk, prompt_ids, *, seed, thinking_cap=256, answer_cap=200, sampling=s2i.INTERVIEW_SAMPLING):
            decoded_prompt = tk.decode(prompt_ids, skip_special_tokens=False)
            if "MISSING_FACT:" in decoded_prompt and "Question:" not in decoded_prompt:
                # Turn 2 Q2 interview prompt
                txt = (
                    "<think>need retry limit</think>"
                    "The evidence does not state the retry count.\n"
                    "MISSING_FACT: The retry limit is 5."
                )
                return {
                    "seed": int(seed),
                    "text": txt,
                    "thinking": "need retry limit",
                    "answer": "The evidence does not state the retry count.\nMISSING_FACT: The retry limit is 5.",
                    "thinking_tokens": 3,
                    "answer_tokens": 12,
                    "think_truncated": False,
                    "ngram_4_rep_rate": 0.0,
                }
            if "The retry limit is 5." in decoded_prompt and "Which part of the claim" not in decoded_prompt and "Copy exactly one line" not in decoded_prompt and "If you had to choose" not in decoded_prompt:
                # Counterfactual Turn-1 verification task rerun
                is_disabled = any(getattr(sub, "disable_adapters", None) is True for sub in m.modules())
                pol_used = "baseline" if is_disabled else "checkpoint"
                seen_verifier_calls.append((pol_used, int(thinking_cap), int(answer_cap)))
                ver_json = json.dumps(
                    {
                        "verdict": "supported",
                        "reason": "retry limit is 5",
                        "evidence_quotes": [
                            {"part": "retry limit is 5", "verdict": "supported", "quote": "The retry limit is 5."}
                        ],
                    }
                )
                txt = f"<think>now supported</think>{ver_json}"
                return {
                    "seed": int(seed),
                    "text": txt,
                    "thinking": "now supported",
                    "answer": ver_json,
                    "thinking_tokens": 2,
                    "answer_tokens": 15,
                    "think_truncated": False,
                    "ngram_4_rep_rate": 0.0,
                }
            return orig_gen(
                m,
                tk,
                prompt_ids,
                seed=seed,
                thinking_cap=thinking_cap,
                answer_cap=answer_cap,
                sampling=sampling,
            )

        monkeypatch.setattr(s2i, "generate_two_stage_thinking", scripted_gen)
        t1 = (
            "<think>missing retry limit</think>"
            + json.dumps(
                {
                    "verdict": "cannot_tell",
                    "reason": "missing",
                    "evidence_quotes": [{"part": "retry limit is 5", "verdict": "cannot_tell", "quote": ""}],
                }
            )
        )
        cohort = s2i.select_interview_cohort(
            [{"item_id": "cf_e2e_1", "sample_index": 0, "verdict": "cannot_tell", "text": t1}],
            [{"item_id": "cf_e2e_1", "sample_index": 0, "verdict": "cannot_tell", "text": t1}],
        )
        out_dir = tmp_path / "cf_e2e_out"
        receipt = s2i.run_interview_for_cohort(
            cohort=cohort,
            rows_by_id={"cf_e2e_1": row},
            runner=peft_runner,
            out_dir=out_dir,
            base_seed=5,
            thinking_cap=4,
            answer_cap=4,
            include_cold_control=False,
        )
        assert seen_verifier_calls == [
            ("baseline", s2.MAX_NEW_TOKENS, s2.MAX_NEW_TOKENS),
            ("checkpoint", s2.MAX_NEW_TOKENS, s2.MAX_NEW_TOKENS),
        ]
        for pol in ("baseline", "checkpoint"):
            cf_sum = receipt["counterfactual_summary"][pol]
            assert cf_sum["n"] == 1
            assert cf_sum["format_ok_count"] == 1
            assert cf_sum["left_cannot_tell_count"] == 1
            assert cf_sum["direction_counts"] == {"supported": 1}

    def test_fix2_item_03_question_text_matches_design_v2_exact_strings(self) -> None:
        import s2_interview as s2i

        assert (
            s2i.INTERVIEW_QUESTIONS["Q1"]
            == "Which part of the claim could you not decide from the evidence? Copy the exact words of that part from the claim."
        )
        assert (
            s2i.INTERVIEW_QUESTIONS["Q4"]
            == "Copy exactly one line from the evidence that is closest to deciding that part, or write NONE."
        )
        assert (
            s2i.INTERVIEW_QUESTIONS["Q2"]
            == "What would the evidence need to say for you to decide it? End with one line: MISSING_FACT: <one sentence>."
        )
        assert (
            s2i.INTERVIEW_QUESTIONS["Q3"]
            == "If you had to choose between {A} and {B}, which would you choose, and why?"
        )
        assert s2i.COLD_CONTROL_QUESTIONS == {
            "Q1": s2i.INTERVIEW_QUESTIONS["Q1"],
            "Q2": s2i.INTERVIEW_QUESTIONS["Q2"],
        }

    def test_fix2_item_04_part_match_summaries_use_gated_values_and_keep_raw_fields(self, tmp_path: Path) -> None:
        import s2_interview as s2i

        claim = (
            "The authentication service rotates signing keys every twenty four hours "
            "and stores backup keys in an offline hardware security module."
        )
        json_ct_parts = ["stores backup keys in an offline hardware security module"]
        sc_copy = s2i.score_part_match(claim, json_ct_parts, claim)
        assert sc_copy["unspecific_full_claim_copy"] is True
        assert sc_copy["token_f1"] == 0.0
        assert sc_copy["containment"] == 0.0
        assert sc_copy["max_overlap"] == 0.0
        assert sc_copy["raw_token_f1"] == pytest.approx(4.0 / 7.0, abs=1e-5)
        assert sc_copy["raw_containment"] == pytest.approx(8.0 / 19.0, abs=1e-5)
        assert sc_copy["raw_max_overlap"] == pytest.approx(4.0 / 7.0, abs=1e-5)

        sc_good = s2i.score_part_match(json_ct_parts[0], json_ct_parts, claim)
        assert sc_good["unspecific_full_claim_copy"] is False
        assert sc_good["token_f1"] == pytest.approx(1.0)
        assert sc_good["containment"] == pytest.approx(1.0)
        assert sc_good["raw_token_f1"] == pytest.approx(1.0)
        assert sc_good["raw_containment"] == pytest.approx(1.0)

        class _StubRunner:
            def __init__(self, q1_by_item: dict[str, str]) -> None:
                self.q1_by_item = q1_by_item

            def ask_question(self, *, row, turn1_completion_text, question_id, policy="baseline", cold_control=False, **kw):
                cid = str(row["id"])
                ans = self.q1_by_item[cid] if question_id == "Q1" else "NONE"
                return {
                    "item_id": cid,
                    "policy": policy,
                    "question_id": question_id,
                    "cold_control": cold_control,
                    "prompt_text": "",
                    "prompt_tokens": 4,
                    "turn1_thinking": "think",
                    "original_turn1_thinking_tokens": 1,
                    "seed": 1,
                    "text": ans,
                    "thinking": "",
                    "answer": ans,
                    "thinking_tokens": 0,
                    "answer_tokens": 2,
                    "think_truncated": False,
                    "ngram_4_rep_rate": 0.0,
                }

            def run_verifier(self, row, seed, **kw) -> str:
                return '{"verdict":"cannot_tell","reason":"r","evidence_quotes":[]}'

        t1 = json.dumps(
            {
                "verdict": "cannot_tell",
                "reason": "missing",
                "evidence_quotes": [{"part": json_ct_parts[0], "verdict": "cannot_tell", "quote": ""}],
            }
        )
        cohort = s2i.select_interview_cohort(
            [
                {"item_id": "i_good", "sample_index": 0, "verdict": "cannot_tell", "text": t1},
                {"item_id": "i_copy", "sample_index": 0, "verdict": "cannot_tell", "text": t1},
            ],
            [],
        )
        rows_by_id = {
            "i_good": _row("i_good", claim=claim, text="x"),
            "i_copy": _row("i_copy", claim=claim, text="x"),
        }
        receipt = s2i.run_interview_for_cohort(
            cohort=cohort,
            rows_by_id=rows_by_id,
            runner=_StubRunner({"i_good": json_ct_parts[0], "i_copy": claim}),  # type: ignore[arg-type]
            out_dir=tmp_path / "pm_sum",
            include_cold_control=False,
        )
        pm_base = receipt["part_match_summary"]["baseline"]
        assert pm_base["mean_token_f1"] == pytest.approx(0.5)
        assert pm_base["median_token_f1"] == pytest.approx(0.5)
        assert pm_base["mean_containment"] == pytest.approx(0.5)
        assert pm_base["median_containment"] == pytest.approx(0.5)
        assert pm_base["mean_overlap"] == pytest.approx(0.5)
        assert pm_base["median_overlap"] == pytest.approx(0.5)
        assert pm_base["raw_mean_containment"] == pytest.approx(27.0 / 38.0, abs=1e-5)
        assert pm_base["raw_median_containment"] == pytest.approx(27.0 / 38.0, abs=1e-5)
        assert pm_base["raw_mean_token_f1"] > 0.5

    def test_fix2_item_05_q3_parsing_refuses_both_labels_and_negations(self) -> None:
        import s2_interview as s2i

        refusal_both = s2i.parse_q3_lean("I cannot choose between supported and unsupported without more evidence.")
        assert refusal_both["format_ok"] is False
        assert refusal_both["lean"] is None

        neither_both = s2i.parse_q3_lean("Neither unsupported nor supported is justified by the text.")
        assert neither_both["format_ok"] is False
        assert neither_both["lean"] is None

        mixed_both = s2i.parse_q3_lean("I would lean toward supported rather than unsupported.")
        assert mixed_both["format_ok"] is False
        assert mixed_both["lean"] is None

        negated_single = s2i.parse_q3_lean("I cannot choose supported without seeing the logs.")
        assert negated_single["format_ok"] is False
        assert negated_single["lean"] is None

        not_label = s2i.parse_q3_lean("The claim is not supported by the passage.")
        assert not_label["format_ok"] is False
        assert not_label["lean"] is None

        scored = [
            s2i.score_q3_lean_item("I cannot choose between supported and unsupported.", "supported", "id_1"),
            s2i.score_q3_lean_item("Neither unsupported nor supported applies.", "unsupported", "id_2"),
            s2i.score_q3_lean_item("I would choose supported because the numbers match.", "supported", "id_3"),
            s2i.score_q3_lean_item("LEAN: unsupported — contradicts the passage.", "unsupported", "id_4"),
        ]
        summary = s2i.summarize_q3_lean(scored)
        assert summary["unparsed_count"] == 2
        assert summary["decidable_supported_acc"] == pytest.approx(0.5)
        assert summary["decidable_unsupported_acc"] == pytest.approx(0.5)
        assert summary["balanced_accuracy"] == pytest.approx(0.5)

    @needs_torch
    def test_fix2_item_06_freeze_file_required_and_probe_records_stack_versions_and_sha(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # 1. When require_exact_versions=True and no freeze file is given or at sys.prefix, refuse with frozen_env_missing_file
        monkeypatch.setattr(s2.sys, "prefix", str(tmp_path / "empty_prefix"))
        with pytest.raises(SystemExit, match="frozen_env_missing_file"):
            s2.check_frozen_env(
                None,
                installed_versions=s2.FROZEN_STACK_VERSIONS,
                require_exact_versions=True,
            )

        # 2. run_probe_memory writes stack_versions and pip_freeze_sha256 into s2-<path>-probe.json
        freeze_lines = [f"{pkg}=={ver}\n" for pkg, ver in sorted(s2.FROZEN_STACK_VERSIONS.items())]
        pf_path = tmp_path / "FROZEN-pip-freeze-2026-11-05.txt"
        pf_bytes = "".join(freeze_lines).encode("utf-8")
        pf_path.write_bytes(pf_bytes)
        pf_sha = s2.hashlib.sha256(pf_bytes).hexdigest()
        monkeypatch.setattr(s2, "FROZEN_PIP_FREEZE_SHA256", pf_sha)
        monkeypatch.setattr(s2, "package_version", lambda k: s2.FROZEN_STACK_VERSIONS.get(k))

        gold_dir = _gold(tmp_path / "gold")
        out_probe = tmp_path / "probe_out"
        a_probe = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "deadbee",
                "--out",
                str(out_probe),
                "--dry",
                "--probe-memory",
                "--probe-lengths",
                "4,8",
                "--prompts",
                "2",
                "--g",
                "2",
                "--frozen-pip-freeze",
                str(pf_path),
            ]
        )
        assert s2.execute(a_probe, build=s2.dry_bundle) == 0
        probe_doc = json.loads((out_probe / "s2-hf-probe.json").read_text(encoding="utf-8"))
        assert probe_doc["stack_versions"] == s2.FROZEN_STACK_VERSIONS
        assert probe_doc["pip_freeze_sha256"] == pf_sha

    def test_fix2_item_07_seed_and_descriptive_provenance_required_everywhere(self, tmp_path: Path) -> None:
        from s2_draw import _load_and_verify_prepass

        # 1. prepass_item_record always writes pass_seed and descriptive, including for Pass A (seed 0, descriptive=False)
        tok = s2.DryTokenizer()
        row = {
            "id": "t00",
            "gold_file": "grounded.jsonl",
            "label": "supported",
            "context": [{"title": "T", "sentences": ["alpha beta gamma"]}],
        }
        c_ok = tok.encode(_reply("supported", "alpha beta gamma"))
        rec_a, _ = s2.prepass_item_record(
            row=row,
            prompt_ids=[1, 2],
            comps=[c_ok],
            truncs=[False],
            tok=tok,
            reward_fn=lambda text, r, tt, truncated=False: s2.reward(
                text, r["label"], s2.passages_of(r), tt, truncated=truncated
            ),
            pass_seed=0,
            descriptive=False,
        )
        assert rec_a["pass_seed"] == 0
        assert rec_a["descriptive"] is False

        # 2. compute_paired_arm_estimators defaults to require_seeds=True and refuses missing seed metadata
        bare_item = {
            "id": "t00",
            "gold_verdict": "supported",
            "g": 8,
            "k_correct": 4,
            "supported": 4,
            "unsupported": 2,
            "cannot_tell": 2,
            "malformed": 0,
            "truncations": 0,
            "reward_mean": 0.2,
        }
        with pytest.raises(SystemExit, match="paired_estimator_wrong_seed"):
            s2.compute_paired_arm_estimators(
                [bare_item],
                [{**bare_item, "pass_seed": 1}],
                [([{**bare_item, "pass_seed": 2}], [{**bare_item, "pass_seed": 3}])],
            )

        # 3. _load_and_verify_prepass and load_prepass_baseline refuse when neither record nor header has seed/descriptive
        no_prov_jsonl = tmp_path / "no_prov.jsonl"
        no_prov_bytes = (json.dumps({"id": "t00", "g": 8, "k_correct": 4}) + "\n").encode("utf-8")
        no_prov_jsonl.write_bytes(no_prov_bytes)
        no_prov_sha = s2.hashlib.sha256(no_prov_bytes).hexdigest()

        with pytest.raises(SystemExit, match="prepass_missing_seed"):
            _load_and_verify_prepass((no_prov_jsonl, no_prov_sha), {"t00"})
        with pytest.raises(SystemExit, match="prepass_baseline_missing_seed"):
            s2.load_prepass_baseline(no_prov_jsonl, no_prov_sha, ["t00"], g=8)

        no_desc_jsonl = tmp_path / "no_desc.jsonl"
        no_desc_bytes = (json.dumps({"id": "t00", "g": 8, "k_correct": 4, "pass_seed": 0}) + "\n").encode("utf-8")
        no_desc_jsonl.write_bytes(no_desc_bytes)
        no_desc_sha = s2.hashlib.sha256(no_desc_bytes).hexdigest()

        with pytest.raises(SystemExit, match="prepass_missing_descriptive"):
            _load_and_verify_prepass((no_desc_jsonl, no_desc_sha), {"t00"})
        with pytest.raises(SystemExit, match="prepass_baseline_missing_descriptive"):
            s2.load_prepass_baseline(no_desc_jsonl, no_desc_sha, ["t00"], g=8)

    @needs_torch
    def test_fix2_item_08_class_stratified_requires_holdout_and_draw_prompts_filters_holdout_everywhere(
        self, tmp_path: Path
    ) -> None:
        from s2_draw import draw_prompts, split_stratified_holdout

        gold_dir = _draw_gold_40(tmp_path / "gold")
        rows, _ = s2.read_gold(gold_dir)
        tune_ids = [f"t{i:02d}" for i in range(40)]
        prepass_path = tmp_path / "pass_a.jsonl"
        prepass_sha = _write_fake_prepass(prepass_path, tune_ids, k_fn=lambda i: i % 9, g=8)

        # 1. --draw class_stratified without prepass / holdout split is refused
        with pytest.raises(SystemExit, match="prepass_missing"):
            draw_prompts(rows, 10, mode="class_stratified", seed=42, holdout_seed="pr79-holdout-v1")

        a_cs_no_prepass = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "deadbee",
                "--out",
                str(tmp_path / "cs_no_pre"),
                "--dry",
                "--draw",
                "class_stratified",
                "--draw-seed",
                "42",
                "--holdout-seed",
                "pr79-holdout-v1",
            ]
        )
        with pytest.raises(SystemExit, match="prepass_missing"):
            s2.execute(a_cs_no_prepass, build=s2.dry_bundle)

        # 2. draw_prompts filters holdout_ids out of first, uniform, kband, and class_stratified when a split is active
        _, _, hold_meta = split_stratified_holdout(
            rows,
            (prepass_path, prepass_sha),
            holdout_fraction=0.20,
            holdout_seed="pr79-holdout-v1",
        )
        hold_set = set(hold_meta["holdout_ids"])
        assert len(hold_set) == 8

        p_first, _ = draw_prompts(rows, 32, mode="first", holdout_ids=hold_set)
        assert {r["id"] for r in p_first}.isdisjoint(hold_set)

        p_uni, _ = draw_prompts(rows, 32, mode="uniform", seed=42, holdout_ids=hold_set)
        assert {r["id"] for r in p_uni}.isdisjoint(hold_set)

        p_kb, _ = draw_prompts(
            rows,
            10,
            mode="kband",
            seed=42,
            prepass=(prepass_path, prepass_sha),
            k_min=0,
            k_max=8,
            holdout_ids=hold_set,
        )
        assert {r["id"] for r in p_kb}.isdisjoint(hold_set)

        # End-to-end execute with --holdout and --draw uniform filters holdout items before run_steps
        out_uni_hold = tmp_path / "uni_hold"
        a_uni_hold = s2.build_parser().parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "deadbee",
                "--out",
                str(out_uni_hold),
                "--dry",
                "--steps",
                "1",
                "--prompts",
                "32",
                "--g",
                "2",
                "--draw",
                "uniform",
                "--draw-seed",
                "42",
                "--holdout",
                "--holdout-seed",
                "pr79-holdout-v1",
                "--pass-a-jsonl",
                str(prepass_path),
                "--pass-a-sha256",
                prepass_sha,
            ]
        )
        assert s2.execute(a_uni_hold, build=s2.dry_bundle) == 0
        rec_uni = json.loads((out_uni_hold / "s2-hf.json").read_text(encoding="utf-8"))
        assert set(rec_uni["prompt_ids"]).isdisjoint(hold_set)


class TestFix3Items:
    @needs_torch
    def test_fix3_item_01_checkpoint_policy_proves_named_checkpoint_and_nonzero_lora_b(
        self, tmp_path: Path
    ) -> None:
        import s2_interview as s2i
        import torch

        row = _row("f3_1", claim="Buffer size is 64 bytes.", text="The local buffer is initialised.")
        t1 = (
            "<think>missing size</think>"
            + json.dumps(
                {
                    "verdict": "cannot_tell",
                    "reason": "missing",
                    "evidence_quotes": [{"part": "64 bytes", "verdict": "cannot_tell", "quote": ""}],
                }
            )
        )

        # 1. Fresh get_peft_model (checkpoint_path=None) has no _loaded_checkpoint_sha256 and all-zero lora_B.
        # It must fail both checks: first missing/mismatched checkpoint sha, and even if sha is forged, all-zero lora_B.
        fresh_model, fresh_tok = s2i.build_cpu_qwen3_stub(with_adapter=True)
        assert getattr(fresh_model, "_loaded_checkpoint_sha256", None) is None
        fresh_runner_no_exp = s2i.InterviewModelRunner(fresh_model, fresh_tok)
        with pytest.raises(SystemExit, match="checkpoint_sha_missing"):
            fresh_runner_no_exp.ask_question(
                row=row,
                turn1_completion_text=t1,
                question_id="Q1",
                policy="checkpoint",
                base_seed=1,
                thinking_cap=4,
                answer_cap=4,
            )
        fresh_runner = s2i.InterviewModelRunner(
            fresh_model, fresh_tok, expected_checkpoint_sha256="deadbeef" * 8
        )
        with pytest.raises(SystemExit, match="checkpoint_sha_missing"):
            fresh_runner.ask_question(
                row=row,
                turn1_completion_text=t1,
                question_id="Q1",
                policy="checkpoint",
                base_seed=1,
                thinking_cap=4,
                answer_cap=4,
            )

        # Even if a zero-lora_B adapter is saved to disk and loaded via attach_or_load_adapter (so sha matches),
        # _policy_context must refuse because every lora_B weight of the active adapter is 0.
        zero_ckpt_dir = tmp_path / "zero_ckpt"
        fresh_model.save_pretrained(str(zero_ckpt_dir))
        zero_ckpt_sha = s2.checkpoint_sha256(zero_ckpt_dir)
        zero_loaded_model, zero_loaded_tok = s2i.build_cpu_qwen3_stub(checkpoint_path=zero_ckpt_dir)
        assert getattr(zero_loaded_model, "_loaded_checkpoint_sha256", None) == zero_ckpt_sha
        zero_runner = s2i.InterviewModelRunner(
            zero_loaded_model, zero_loaded_tok, expected_checkpoint_sha256=zero_ckpt_sha
        )
        with pytest.raises(SystemExit, match="checkpoint_lora_b_all_zero"):
            zero_runner.ask_question(
                row=row,
                turn1_completion_text=t1,
                question_id="Q1",
                policy="checkpoint",
                base_seed=1,
                thinking_cap=4,
                answer_cap=4,
            )

        # 2. A trained checkpoint (non-zero lora_B) loaded via attach_or_load_adapter records _loaded_checkpoint_sha256,
        # refuses on sha mismatch, and succeeds when expected_checkpoint_sha256 matches.
        with torch.no_grad():
            for name, param in fresh_model.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.05)
        trained_ckpt_dir = tmp_path / "trained_ckpt"
        fresh_model.save_pretrained(str(trained_ckpt_dir))
        trained_sha = s2.checkpoint_sha256(trained_ckpt_dir)

        loaded_model, loaded_tok = s2i.build_cpu_qwen3_stub(checkpoint_path=trained_ckpt_dir)
        assert getattr(loaded_model, "_loaded_checkpoint_sha256", None) == trained_sha

        wrong_sha_runner = s2i.InterviewModelRunner(
            loaded_model, loaded_tok, expected_checkpoint_sha256="00" * 32
        )
        with pytest.raises(SystemExit, match="checkpoint_sha_mismatch"):
            wrong_sha_runner.ask_question(
                row=row,
                turn1_completion_text=t1,
                question_id="Q1",
                policy="checkpoint",
                base_seed=1,
                thinking_cap=4,
                answer_cap=4,
            )

        ok_runner = s2i.InterviewModelRunner(
            loaded_model, loaded_tok, expected_checkpoint_sha256=trained_sha
        )
        ans = ok_runner.ask_question(
            row=row,
            turn1_completion_text=t1,
            question_id="Q1",
            policy="checkpoint",
            base_seed=1,
            thinking_cap=4,
            answer_cap=4,
        )
        assert ans["policy"] == "checkpoint"

    def test_fix3_item_02_legacy_and_nondescriptive_pass_headers_accepted_consistently(
        self, tmp_path: Path
    ) -> None:
        import hashlib

        from s2_draw import _extract_k_by_id, _load_and_verify_prepass

        # 1. Verify legacy Pass A and Pass B2 header shapes (no 'descriptive' field)
        pa_hdr = {
            "ids_sha256": "54848c6fa915b3a6ab0c366304b276ce42beac43252a7bf3b1291321c97b9e95",
            "script_sha256": "be8c800987ac982a9732158d1e48aa2a0e19ab96a8dd24b26085d3c674f5c63a",
            "gold_files_sha256": {},
            "g": 8,
            "max_new": 2048,
            "quote_reward": "v2",
            "seed": 0,
            "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        }
        b2_hdr = {
            "ids_sha256": "54848c6fa915b3a6ab0c366304b276ce42beac43252a7bf3b1291321c97b9e95",
            "script_sha256": "4577ddba30c9b18661f12d4b0f35742dd02d73aa26243cd4f99754d2b453ab19",
            "gold_files_sha256": {},
            "g": 8,
            "max_new": 2048,
            "quote_reward": "v2",
            "seed": 1,
            "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "reproducible": True,
        }
        assert "descriptive" not in pa_hdr
        assert "descriptive" not in b2_hdr

        # Legacy records also lack 'pass_seed' and 'descriptive'
        legacy_line = (
            json.dumps(
                {
                    "id": "fe_0",
                    "gold_file": "fever_500.jsonl",
                    "gold_verdict": "cannot_tell",
                    "g": 8,
                    "k_correct": 4,
                    "supported": 4,
                    "unsupported": 0,
                    "cannot_tell": 4,
                    "malformed": 0,
                    "truncations": 0,
                    "quote_pass": 0,
                    "reward_mean": 0.25,
                    "reward_std": 0.65,
                    "zero_variance": False,
                    "prompt_tokens": 227,
                    "completion_tokens_mean": 300.5,
                }
            )
            + "\n"
        ).encode("utf-8")
        legacy_sha = hashlib.sha256(legacy_line).hexdigest()

        pa_dir = tmp_path / "legacy_pa"
        pa_dir.mkdir()
        (pa_dir / "s2-hf-prepass-header.json").write_text(json.dumps(pa_hdr), encoding="utf-8")
        pa_jsonl = pa_dir / "s2-hf-prepass.jsonl"
        pa_jsonl.write_bytes(legacy_line)

        assert s2._check_header_not_descriptive(pa_jsonl, expected_seeds=(0,)) == (True, True)
        k_pa, _ = s2.load_prepass_baseline(pa_jsonl, legacy_sha, ["fe_0"], g=8, expected_seeds=(0,))
        assert k_pa == {"fe_0": 4}

        b2_dir = tmp_path / "legacy_b2"
        b2_dir.mkdir()
        (b2_dir / "s2-hf-prepass-header.json").write_text(json.dumps(b2_hdr), encoding="utf-8")
        b2_jsonl = b2_dir / "s2-hf-prepass.jsonl"
        b2_jsonl.write_bytes(legacy_line)

        assert s2._check_header_not_descriptive(b2_jsonl, expected_seeds=(1,)) == (True, True)
        k_b2, _ = s2.load_prepass_baseline(b2_jsonl, legacy_sha, ["fe_0"], g=8, expected_seeds=(1,))
        assert k_b2 == {"fe_0": 4}

        # 2. Non-bool 'descriptive' in header or record is refused everywhere
        for bad_desc in ("false", "true", 0, 1, None, ["no"]):
            bad_hdr_dir = tmp_path / f"bad_hdr_{type(bad_desc).__name__}_{bad_desc}"
            bad_hdr_dir.mkdir(exist_ok=True)
            (bad_hdr_dir / "s2-hf-prepass-header.json").write_text(
                json.dumps({**pa_hdr, "descriptive": bad_desc}), encoding="utf-8"
            )
            bad_hdr_jsonl = bad_hdr_dir / "s2-hf-prepass.jsonl"
            bad_hdr_jsonl.write_bytes(legacy_line)
            with pytest.raises(SystemExit, match="prepass_baseline_descriptive"):
                s2._check_header_not_descriptive(bad_hdr_jsonl, expected_seeds=(0,))
            with pytest.raises(SystemExit, match="prepass_baseline_descriptive"):
                s2.load_prepass_baseline(bad_hdr_jsonl, legacy_sha, ["fe_0"], g=8)
            with pytest.raises(SystemExit, match="prepass_descriptive"):
                _load_and_verify_prepass((bad_hdr_jsonl, legacy_sha), {"fe_0"})

            bad_rec_dir = tmp_path / f"bad_rec_{type(bad_desc).__name__}_{bad_desc}"
            bad_rec_dir.mkdir(exist_ok=True)
            bad_rec_bytes = (
                json.dumps({"id": "fe_0", "g": 8, "k_correct": 4, "pass_seed": 0, "descriptive": bad_desc})
                + "\n"
            ).encode("utf-8")
            bad_rec_jsonl = bad_rec_dir / "s2-hf-prepass.jsonl"
            bad_rec_jsonl.write_bytes(bad_rec_bytes)
            bad_rec_sha = hashlib.sha256(bad_rec_bytes).hexdigest()
            with pytest.raises(SystemExit, match="prepass_baseline_descriptive"):
                s2.load_prepass_baseline(bad_rec_jsonl, bad_rec_sha, ["fe_0"], g=8)
            with pytest.raises(SystemExit, match="prepass_descriptive"):
                _load_and_verify_prepass((bad_rec_jsonl, bad_rec_sha), {"fe_0"})
            with pytest.raises(SystemExit, match="prepass_descriptive"):
                _extract_k_by_id(
                    {"fe_0": {"k_correct": 4, "pass_seed": 0, "descriptive": bad_desc}},
                    {"fe_0"},
                )
            with pytest.raises(SystemExit, match="paired_estimator_descriptive"):
                s2._verify_pass_provenance(
                    [{"id": "fe_0", "k_correct": 4, "pass_seed": 0, "descriptive": bad_desc}],
                    role="Pass A",
                )

        # 3. Newly written non-descriptive pass headers explicitly include "descriptive": False
        new_hdr = s2.build_prepass_header(
            prompt_ids=["fe_0"],
            gold_hashes={"fever_500.jsonl": "aa" * 32},
            g=8,
            max_new=2048,
            quote_reward="v2",
            seed=2,
            model_revision=s2.STUDENT_FULL_REVISION,
            descriptive=False,
        )
        assert "descriptive" in new_hdr and new_hdr["descriptive"] is False

    def test_fix3_item_03_logging_pass_requires_and_records_b2_replay_settings(
        self, tmp_path: Path
    ) -> None:
        ap = s2.build_parser()
        gold_dir = _prepass_gold(tmp_path / "gold")

        # 1. Verify --logging-pass refuses when B2 replay settings are missing or altered
        a_default_missing_b2 = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "out_missing_b2"),
                "--logging-pass",
            ]
        )
        with pytest.raises(SystemExit, match="logging_pass_b2_settings_mismatch"):
            s2.validate_pass_mode_args(a_default_missing_b2)

        expected_b2 = {
            "path": "hf",
            "revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "bf16_restore": True,
            "logprob_microbatch": 2,
            "attention_mask": True,
            "alloc_conf": "expandable",
            "empty_cache_phases": True,
            "quote_reward": "v2",
            "g": 8,
            "max_new": 2048,
        }
        assert s2.B2_REPLAY_SETTINGS == expected_b2

        base_cli = [
            "--gold",
            str(gold_dir),
            "--rnd-commit",
            "e48cc90c",
            "--out",
            str(tmp_path / "out"),
            "--logging-pass",
            "--path",
            "hf",
            "--revision",
            "b968826d9c46dd6066d109eabc6255188de91218",
            "--bf16-restore",
            "--logprob-microbatch",
            "2",
            "--attention-mask",
            "--alloc-conf",
            "expandable",
            "--empty-cache-phases",
            "--quote-reward",
            "v2",
            "--g",
            "8",
            "--max-new",
            "2048",
            "--replay-check-off-reason",
            "unit test",
        ]
        a_ok = ap.parse_args(base_cli)
        assert s2.validate_pass_mode_args(a_ok)[:3] == (True, True, 1)

        # Verify each B2 replay setting is enforced and refused when missing or altered
        for key, bad_val in (
            ("path", "llamacpp"),
            ("revision", "b968826d"),
            ("bf16_restore", False),
            ("logprob_microbatch", 1),
            ("logprob_microbatch", None),
            ("attention_mask", False),
            ("alloc_conf", None),
            ("empty_cache_phases", False),
            ("quote_reward", "v1"),
            ("g", 4),
            ("max_new", 1024),
        ):
            a_bad = ap.parse_args(base_cli)
            setattr(a_bad, key, bad_val)
            with pytest.raises(SystemExit, match=f"logging_pass_b2_settings_mismatch.*{key}"):
                s2.validate_pass_mode_args(a_bad)

        hdr = s2.build_prepass_header(
            prompt_ids=["t_a"],
            gold_hashes={"grounded.jsonl": "11" * 32},
            g=8,
            max_new=2048,
            quote_reward="v2",
            seed=1,
            model_revision=s2.STUDENT_FULL_REVISION,
            descriptive=True,
        )
        assert hdr["descriptive"] is True
        assert hdr["b2_replay_settings"] == expected_b2

    @needs_torch
    def test_fix3_item_04_logging_pass_replay_check_and_mismatch_halt(
        self, tmp_path: Path
    ) -> None:
        import hashlib
        from contextlib import nullcontext

        import torch

        ap = s2.build_parser()
        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        tune_ids = ["t_a", "t_b", "t_d", "t_k", "t_m", "t_z"]

        b2_cli_flags = [
            "--logging-pass",
            "--path",
            "hf",
            "--revision",
            "b968826d9c46dd6066d109eabc6255188de91218",
            "--bf16-restore",
            "--logprob-microbatch",
            "2",
            "--attention-mask",
            "--alloc-conf",
            "expandable",
            "--empty-cache-phases",
            "--quote-reward",
            "v2",
            "--g",
            "8",
            "--max-new",
            "2048",
        ]

        # 1. --logging-pass without --replay-check and without --replay-check-off-reason is refused
        a_no_rc = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "no_rc"),
                *b2_cli_flags,
            ]
        )
        with pytest.raises(SystemExit, match="logging_pass_requires_replay_check"):
            s2.validate_pass_mode_args(a_no_rc)

        # Empty/whitespace --replay-check-off-reason is also refused
        a_empty_reason = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "empty_reason"),
                *b2_cli_flags,
                "--replay-check-off-reason",
                "   ",
            ]
        )
        with pytest.raises(SystemExit, match="logging_pass_requires_replay_check"):
            s2.validate_pass_mode_args(a_empty_reason)

        # Write a B2 JSONL where every item has k_correct=8 except t_d which has k_correct=7
        b2_jsonl = tmp_path / "b2_baseline.jsonl"
        b2_rows = [
            {"id": cid, "g": 8, "k_correct": 7 if cid == "t_d" else 8, "pass_seed": 1, "descriptive": False}
            for cid in tune_ids
        ]
        b2_jsonl.write_text(
            "".join(json.dumps(r) + "\n" for r in b2_rows),
            encoding="utf-8",
            newline="\n",
        )
        b2_sha = hashlib.sha256(b2_jsonl.read_bytes()).hexdigest()

        seen_prompts: list[str] = []

        def make_bundle(_args_ns):
            fake_cuda = _FakeCuda()

            def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
                seen_prompts.append(prompt_text)
                p_ids = tok.encode(prompt_text)
                # Make t_b (the 2nd item, index 1) return only 5 correct completions instead of 8!
                n_wrong = 3 if "Claim tb" in prompt_text else 0
                comps = []
                truncs = []
                for idx in range(g_cnt):
                    if idx < n_wrong:
                        comps.append(
                            tok.encode(_reply("unsupported", "alpha beta gamma", think="wrong"))
                            + [tok.eos_token_id]
                        )
                    else:
                        comps.append(
                            tok.encode(_reply("supported", "alpha beta gamma", think="ok"))
                            + [tok.eos_token_id]
                        )
                    truncs.append(False)
                return p_ids, comps, truncs

            return {
                "model": Namespace(device=torch.device("cpu"), parameters=lambda: []),
                "tok": tok,
                "cuda": fake_cuda,
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 100,
                "revision": s2.STUDENT_FULL_REVISION,
                "llama": None,
                "dtype_census": None,
                "adapter_loaded": False,
            }

        # 2. Run --logging-pass with --replay-check: t_a (index 0) matches (k=8),
        # t_b (index 1) mismatches (k_correct=5 vs b2_k_correct=8) -> halts immediately with LOGGING_PASS_REPLAY_MISMATCH
        mismatch_dir = tmp_path / "logging_mismatch"
        a_mismatch = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(mismatch_dir),
                *b2_cli_flags,
                "--replay-check",
                str(b2_jsonl),
                "--replay-check-sha256",
                b2_sha,
            ]
        )
        with pytest.raises(SystemExit) as exc_info:
            s2.execute(a_mismatch, build=make_bundle, rig=_FakeRig(a_mismatch))
        err_msg = str(exc_info.value)
        assert "LOGGING_PASS_REPLAY_MISMATCH" in err_msg
        assert "t_b" in err_msg
        assert "k_correct=3" in err_msg
        assert "b2_k_correct=8" in err_msg

        # Only t_a and t_b were evaluated; subsequent items (t_d, t_k, t_m, t_z) were not run
        assert len(seen_prompts) == 2
        partial_jsonl = mismatch_dir / "s2-hf-prepass.jsonl"
        assert partial_jsonl.is_file()
        partial_recs = [
            json.loads(line) for line in partial_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        assert [r["id"] for r in partial_recs] == ["t_a", "t_b"]

        partial_sa = s2.read_save_all(mismatch_dir / "s2-hf-save-all.jsonl.gz")
        assert len(partial_sa) == 2 * 8

        partial_summary = json.loads(
            (mismatch_dir / "s2-hf-prepass-summary.json").read_text(encoding="utf-8")
        )
        assert partial_summary["item_count"] == 2
        assert partial_summary["stopped"]["reason"] == "LOGGING_PASS_REPLAY_MISMATCH"
        assert partial_summary["stopped"]["item_id"] == "t_b"
        assert partial_summary["stopped"]["k_correct"] == 3
        assert partial_summary["stopped"]["b2_k_correct"] == 8

        partial_receipt = json.loads((mismatch_dir / "s2-hf.json").read_text(encoding="utf-8"))
        assert partial_receipt["stopped"]["reason"] == "LOGGING_PASS_REPLAY_MISMATCH"
        assert partial_receipt["run"]["replay_check_sha256"] == b2_sha
        hdr_mismatch = json.loads(
            (mismatch_dir / "s2-hf-prepass-header.json").read_text(encoding="utf-8")
        )
        assert hdr_mismatch["replay_check_sha256"] == b2_sha

        # 2b. Relaunching into the same out dir with the same arguments refuses before running any new item
        seen_prompts.clear()
        with pytest.raises(SystemExit, match="LOGGING_PASS_REPLAY_MISMATCH"):
            s2.execute(a_mismatch, build=make_bundle, rig=_FakeRig(a_mismatch))
        assert len(seen_prompts) == 0

        # Even if the summary file is absent (or before summary check), _prepare_prepass_resume re-checks
        # every completed record against replay_check_k and refuses before running any new item.
        (mismatch_dir / "s2-hf-prepass-summary.json").unlink()
        with pytest.raises(SystemExit, match="LOGGING_PASS_REPLAY_MISMATCH") as relaunch_exc:
            s2.execute(a_mismatch, build=make_bundle, rig=_FakeRig(a_mismatch))
        assert "item_id=t_b" in str(relaunch_exc.value)
        assert len(seen_prompts) == 0

        # 2c. Refusal on a wrong --replay-check-sha256 (replay_check_sha_mismatch)
        a_bad_sha = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(tmp_path / "logging_bad_sha"),
                *b2_cli_flags,
                "--replay-check",
                str(b2_jsonl),
                "--replay-check-sha256",
                "00" * 32,
            ]
        )
        with pytest.raises(SystemExit, match="replay_check_sha_mismatch"):
            s2.execute(a_bad_sha, build=make_bundle, rig=_FakeRig(a_bad_sha))

        # 2d. Full run where every item's k_correct matches B2 and the pass completes
        matching_b2_jsonl = tmp_path / "b2_matching.jsonl"
        expected_k_by_id = {"t_a": 8, "t_b": 3, "t_d": 8, "t_k": 0, "t_m": 0, "t_z": 8}
        matching_b2_rows = [
            {
                "id": cid,
                "g": 8,
                "k_correct": expected_k_by_id[cid],
                "pass_seed": 1,
                "descriptive": False,
            }
            for cid in tune_ids
        ]
        matching_b2_jsonl.write_text(
            "".join(json.dumps(r) + "\n" for r in matching_b2_rows),
            encoding="utf-8",
            newline="\n",
        )
        matching_b2_sha = hashlib.sha256(matching_b2_jsonl.read_bytes()).hexdigest()
        match_dir = tmp_path / "logging_match"
        a_match = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(match_dir),
                *b2_cli_flags,
                "--replay-check",
                str(matching_b2_jsonl),
                "--replay-check-sha256",
                matching_b2_sha,
            ]
        )
        seen_prompts.clear()
        assert s2.execute(a_match, build=make_bundle, rig=_FakeRig(a_match)) == 0
        assert len(seen_prompts) == len(tune_ids)
        match_summary = json.loads(
            (match_dir / "s2-hf-prepass-summary.json").read_text(encoding="utf-8")
        )
        assert match_summary["item_count"] == len(tune_ids)
        assert "stopped" not in match_summary
        match_receipt = json.loads((match_dir / "s2-hf.json").read_text(encoding="utf-8"))
        assert match_receipt["stopped"] is None
        assert match_receipt["run"]["replay_check_sha256"] == matching_b2_sha

        # 3. Also verify --replay-check-off-reason is recorded when provided
        off_dir = tmp_path / "logging_rc_off"
        a_off = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(off_dir),
                *b2_cli_flags,
                "--replay-check-off-reason",
                "explicitly bypassed in unit test",
            ]
        )
        assert s2.execute(a_off, build=make_bundle, rig=_FakeRig(a_off)) == 0
        hdr_off = json.loads((off_dir / "s2-hf-prepass-header.json").read_text(encoding="utf-8"))
        assert hdr_off["replay_check_off_reason"] == "explicitly bypassed in unit test"
        rec_off = json.loads((off_dir / "s2-hf.json").read_text(encoding="utf-8"))
        assert rec_off["run"]["replay_check_off_reason"] == "explicitly bypassed in unit test"

    @needs_torch
    def test_fix4_item_04_checkpoint_sha_stored_and_checked_per_adapter_name(
        self, tmp_path: Path
    ) -> None:
        import s2_interview as s2i
        import torch

        row = _row("f4_4", claim="Buffer size is 64 bytes.", text="The local buffer is initialised.")
        t1 = (
            "<think>missing size</think>"
            + json.dumps(
                {
                    "verdict": "cannot_tell",
                    "reason": "missing",
                    "evidence_quotes": [{"part": "64 bytes", "verdict": "cannot_tell", "quote": ""}],
                }
            )
        )

        # Create two distinct trained checkpoints on disk (ckpt_a and ckpt_b) with different non-zero lora_B
        base_m, _ = s2i.build_cpu_qwen3_stub(with_adapter=True)
        with torch.no_grad():
            for name, param in base_m.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.05)
        ckpt_a_dir = tmp_path / "ckpt_a"
        base_m.save_pretrained(str(ckpt_a_dir))
        sha_a = s2.checkpoint_sha256(ckpt_a_dir)

        with torch.no_grad():
            for name, param in base_m.named_parameters():
                if "lora_B" in name:
                    param.fill_(0.12)
        ckpt_b_dir = tmp_path / "ckpt_b"
        base_m.save_pretrained(str(ckpt_b_dir))
        sha_b = s2.checkpoint_sha256(ckpt_b_dir)
        assert sha_a != sha_b

        # Load adapter_a via attach_or_load_adapter, then load a second trained adapter_b onto the same model
        raw_model, tok = s2i.build_cpu_qwen3_stub(with_adapter=False)
        model, _, _, _ = s2.attach_or_load_adapter(
            raw_model, ckpt_a_dir, for_pass=True, adapter_name="adapter_a"
        )
        model, _, _, _ = s2.attach_or_load_adapter(
            model, ckpt_b_dir, for_pass=True, adapter_name="adapter_b"
        )
        assert model._loaded_checkpoint_sha256_by_adapter == {
            "adapter_a": sha_a,
            "adapter_b": sha_b,
        }

        # Runner expecting sha_b (the active/last-loaded adapter) must REFUSE policy="checkpoint:adapter_a"
        runner_expect_b = s2i.InterviewModelRunner(model, tok, expected_checkpoint_sha256=sha_b)
        with pytest.raises(SystemExit, match="checkpoint_sha_mismatch"):
            runner_expect_b.ask_question(
                row=row,
                turn1_completion_text=t1,
                question_id="Q1",
                policy="checkpoint:adapter_a",
                base_seed=1,
                thinking_cap=4,
                answer_cap=4,
            )

        # And if a second trained adapter is loaded via raw model.load_adapter without attach_or_load_adapter,
        # it has no entry in _loaded_checkpoint_sha256_by_adapter and must refuse with checkpoint_sha_missing
        model.load_adapter(str(ckpt_b_dir), adapter_name="untracked_adapter", is_trainable=False)
        with pytest.raises(SystemExit, match="checkpoint_sha_missing"):
            runner_expect_b.ask_question(
                row=row,
                turn1_completion_text=t1,
                question_id="Q1",
                policy="checkpoint:untracked_adapter",
                base_seed=1,
                thinking_cap=4,
                answer_cap=4,
            )

        # Each tracked adapter succeeds when its own expected sha is supplied
        ans_a = runner_expect_b.ask_question(
            row=row,
            turn1_completion_text=t1,
            question_id="Q1",
            policy="checkpoint:adapter_a",
            base_seed=1,
            thinking_cap=4,
            answer_cap=4,
            expected_checkpoint_sha256=sha_a,
        )
        assert ans_a["policy"] == "checkpoint:adapter_a"

        ans_b = runner_expect_b.ask_question(
            row=row,
            turn1_completion_text=t1,
            question_id="Q1",
            policy="checkpoint:adapter_b",
            base_seed=1,
            thinking_cap=4,
            answer_cap=4,
        )
        assert ans_b["policy"] == "checkpoint:adapter_b"


class _SimulatedHardKill(BaseException):
    """Simulates a hard process kill (taskkill /T /F) that aborts execution immediately."""


class TestSaveAllCrashSafeAndStopAfterNew:
    @needs_torch
    def test_crashsafe_01_kill_and_resume_pass(self, tmp_path: Path) -> None:
        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        ap = s2.build_parser()
        out_dir = tmp_path / "pass_kill_resume"
        a = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(out_dir),
                "--rollout-only",
                "--g",
                "2",
                "--max-new",
                "32",
                "--quote-reward",
                "v2",
                "--pass-id",
                "pass_kill_test",
            ]
        )
        rows, gold_hashes = s2.read_gold(a.gold)
        prompts = s2.select_prompts(rows, None)
        assert len(prompts) == 6

        calls = 0
        kill_after = 2
        sa_path = s2.save_all_file_path(out_dir, "hf")
        sa_bytes_at_kill: bytes | None = None

        def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            nonlocal calls, sa_bytes_at_kill
            calls += 1
            if calls == kill_after + 1:
                sa_bytes_at_kill = sa_path.read_bytes() if sa_path.exists() else None
                raise _SimulatedHardKill("taskkill /T /F after 2 items")
            p_ids = tok.encode(prompt_text)
            c_ok = tok.encode(_reply("supported", "alpha beta gamma", think=f"thought_{calls}")) + [
                tok.eos_token_id
            ]
            c_ct = tok.encode(_reply("cannot_tell", "", think=f"ct_{calls}")) + [tok.eos_token_id]
            return p_ids, [c_ok, c_ct][:g_cnt], [False, False][:g_cnt]

        bundle = {
            "tok": tok,
            "rollout": rollout,
            "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
            ),
            "sync": lambda: None,
            "peak_alloc_mib": lambda: 300,
            "revision": s2.STUDENT_FULL_REVISION,
        }

        with pytest.raises(_SimulatedHardKill):
            s2.run_prepass(a=a, prompts=prompts, gold_hashes=gold_hashes, bundle=bundle)

        # At the exact instant of the hard kill (before any finally block could ever run),
        # save-all on disk must already hold the K=2 completed items (4 records).
        assert sa_bytes_at_kill is not None
        sa_path.write_bytes(sa_bytes_at_kill)
        partial_records = s2.read_save_all(sa_path)
        assert len(partial_records) == kill_after * 2
        assert [r["item_id"] for r in partial_records] == [
            prompts[0]["id"],
            prompts[0]["id"],
            prompts[1]["id"],
            prompts[1]["id"],
        ]

        # Also simulate a truncated trailing gzip member on disk from a kill mid-write, then resume:
        sa_path.write_bytes(sa_bytes_at_kill + b"\x1f\x8b\x08\x00truncated_gzip")
        _, completed, stopped, oom = s2.run_prepass(
            a=a, prompts=prompts, gold_hashes=gold_hashes, bundle=bundle
        )
        assert stopped is None and not oom
        assert len(completed) == len(prompts)

        final_records = s2.read_save_all(sa_path)
        assert len(final_records) == len(prompts) * 2
        seen_pairs = [(r["item_id"], r["sample_index"]) for r in final_records]
        expected_pairs = [(p["id"], s_idx) for p in prompts for s_idx in range(2)]
        assert seen_pairs == expected_pairs

    @needs_torch
    def test_crashsafe_02_no_duplicates_kill_mid_item_before_jsonl(self, tmp_path: Path) -> None:
        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        ap = s2.build_parser()
        out_dir = tmp_path / "pass_mid_item_kill"
        a = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(out_dir),
                "--rollout-only",
                "--g",
                "2",
                "--max-new",
                "32",
            ]
        )
        rows, gold_hashes = s2.read_gold(a.gold)
        prompts = s2.select_prompts(rows, None)
        sa_path = s2.save_all_file_path(out_dir, "hf")
        jsonl_path = out_dir / "s2-hf-prepass.jsonl"

        def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
            p_ids = tok.encode(prompt_text)
            c0 = tok.encode(_reply("supported", "alpha beta gamma", think="t0")) + [tok.eos_token_id]
            c1 = tok.encode(_reply("unsupported", "alpha beta gamma", think="t1")) + [tok.eos_token_id]
            return p_ids, [c0, c1][:g_cnt], [False, False][:g_cnt]

        bundle = {
            "tok": tok,
            "rollout": rollout,
            "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                text, row["label"], s2.passages_of(row), tt, truncated=truncated
            ),
            "sync": lambda: None,
            "peak_alloc_mib": lambda: 300,
            "revision": s2.STUDENT_FULL_REVISION,
        }

        # Kill on the 2nd item right after append_save_all fsyncs the 2nd item's save-all records,
        # BEFORE jfh.write writes the 2nd item's line to s2-hf-prepass.jsonl.
        sa_fsync_count = 0

        def kill_after_second_save_all_fsync(fd: int) -> None:
            nonlocal sa_fsync_count
            os.fsync(fd)
            if sa_path.exists() and jsonl_path.exists():
                sa_rows_now = s2.read_save_all(sa_path)
                jsonl_lines_now = [
                    ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()
                ]
                # When item 2's save-all member has just been fsynced (4 records total),
                # but jsonl_path still only has item 1 (1 line): kill!
                if len(sa_rows_now) == 4 and len(jsonl_lines_now) == 1:
                    sa_fsync_count += 1
                    raise _SimulatedHardKill("killed after save-all write before jsonl write")

        with pytest.raises(_SimulatedHardKill):
            s2.run_prepass(
                a=a,
                prompts=prompts,
                gold_hashes=gold_hashes,
                bundle=bundle,
                fsync=kill_after_second_save_all_fsync,
            )

        assert sa_fsync_count == 1
        # At this point save-all has 4 records (items 0 and 1), while prepass.jsonl has 1 line (item 0).
        assert len(s2.read_save_all(sa_path)) == 4
        assert len(jsonl_path.read_text(encoding="utf-8").splitlines()) == 1

        # Now resume to completion: item 1 must be rewritten cleanly on its redo with zero duplicate records.
        _, completed, stopped, oom = s2.run_prepass(
            a=a, prompts=prompts, gold_hashes=gold_hashes, bundle=bundle
        )
        assert stopped is None and not oom
        assert len(completed) == len(prompts)
        final_sa = s2.read_save_all(sa_path)
        assert len(final_sa) == len(prompts) * 2
        assert [(r["item_id"], r["sample_index"]) for r in final_sa] == [
            (p["id"], s_idx) for p in prompts for s_idx in range(2)
        ]

    @needs_torch
    def test_crashsafe_03_replay_check_survives_resume(self, tmp_path: Path) -> None:
        import hashlib
        from contextlib import nullcontext

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        ap = s2.build_parser()
        rows, _ = s2.read_gold(gold_dir)
        prompts = s2.select_prompts(rows, None)
        tune_ids = [r["id"] for r in prompts]
        assert len(tune_ids) == 6

        # Plant a B2 k_correct mismatch on item index 3 (the 4th item, which runs in chunk 2 after resume).
        b2_dir = tmp_path / "b2_ref"
        b2_dir.mkdir(parents=True, exist_ok=True)
        b2_jsonl = b2_dir / "s2-hf-prepass.jsonl"
        b2_records = []
        for idx, p_row in enumerate(prompts):
            tid = p_row["id"]
            gv = str(p_row.get("label", "supported"))
            expected_k = 8 if gv == "supported" else 0
            b2_records.append(
                {
                    "id": tid,
                    "gold_file": "grounded.jsonl",
                    "gold_verdict": gv,
                    "g": 8,
                    "k_correct": expected_k if idx != 3 else (0 if expected_k == 8 else 8),
                    "supported": 8,
                    "unsupported": 0,
                    "cannot_tell": 0,
                    "malformed": 0,
                    "truncations": 0,
                    "quote_pass": 8,
                    "reward_mean": 1.0,
                    "reward_std": 0.0,
                    "zero_variance": True,
                    "prompt_tokens": 12,
                    "completion_tokens_mean": 10.0,
                    "descriptive": False,
                    "pass_seed": 1,
                }
            )
        b2_jsonl.write_text("".join(json.dumps(r) + "\n" for r in b2_records), encoding="utf-8", newline="\n")
        b2_sha = hashlib.sha256(b2_jsonl.read_bytes()).hexdigest()

        def make_bundle(_args_ns):
            def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
                p_ids = tok.encode(prompt_text)
                c_ok = tok.encode(_reply("supported", "alpha beta gamma", think="b2 thought")) + [
                    tok.eos_token_id
                ]
                return p_ids, [list(c_ok) for _ in range(g_cnt)], [False] * g_cnt

            return {
                "model": Namespace(device="cpu"),
                "tok": tok,
                "cuda": _FakeCuda(),
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": rollout,
                "reward_fn": lambda text, row, tt, truncated=False: 1.0,
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 400,
                "revision": s2.STUDENT_FULL_REVISION,
                "llama": None,
                "dtype_census": None,
            }

        out_dir = tmp_path / "logging_chunked_mismatch"
        common_flags = [
            "--gold",
            str(gold_dir),
            "--rnd-commit",
            "e48cc90c",
            "--out",
            str(out_dir),
            "--logging-pass",
            "--path",
            "hf",
            "--revision",
            s2.STUDENT_FULL_REVISION,
            "--bf16-restore",
            "--logprob-microbatch",
            "2",
            "--attention-mask",
            "--alloc-conf",
            "expandable",
            "--empty-cache-phases",
            "--quote-reward",
            "v2",
            "--g",
            "8",
            "--max-new",
            "2048",
            "--replay-check",
            str(b2_jsonl),
            "--replay-check-sha256",
            b2_sha,
        ]

        # Chunk 1: run first 2 items cleanly via --stop-after-new 2
        a_chunk1 = ap.parse_args([*common_flags, "--stop-after-new", "2"])
        assert s2.execute(a_chunk1, build=make_bundle, rig=_FakeRig(a_chunk1)) == 1
        sa_path = out_dir / "s2-hf-save-all.jsonl.gz"
        assert len(s2.read_save_all(sa_path)) == 2 * 8

        # Chunk 2: resume with --stop-after-new 3; item index 3 (the 4th item) triggers LOGGING_PASS_REPLAY_MISMATCH
        a_chunk2 = ap.parse_args([*common_flags, "--stop-after-new", "3"])
        with pytest.raises(SystemExit, match="LOGGING_PASS_REPLAY_MISMATCH"):
            s2.execute(a_chunk2, build=make_bundle, rig=_FakeRig(a_chunk2))

        # Verify the first 4 items (including the mismatching 4th item) are all preserved in save-all on disk
        assert len(s2.read_save_all(sa_path)) == 4 * 8
        summary = json.loads((out_dir / "s2-hf-prepass-summary.json").read_text(encoding="utf-8"))
        assert summary["stopped"]["reason"] == "LOGGING_PASS_REPLAY_MISMATCH"
        assert summary["stopped"]["item_id"] == tune_ids[3]

    @needs_torch
    def test_crashsafe_04_chunking_byte_equal_to_straight_through(self, tmp_path: Path) -> None:
        from contextlib import nullcontext

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        ap = s2.build_parser()

        def make_bundle(_args_ns):
            def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
                p_ids = tok.encode(prompt_text)
                c0 = tok.encode(_reply("supported", "alpha beta gamma", think="chunk test 0")) + [
                    tok.eos_token_id
                ]
                c1 = tok.encode(_reply("cannot_tell", "", think="chunk test 1")) + [tok.eos_token_id]
                return p_ids, [c0, c1][:g_cnt], [False, False][:g_cnt]

            return {
                "model": Namespace(device="cpu"),
                "tok": tok,
                "cuda": _FakeCuda(),
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated, quote_reward="v2"
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 350,
                "revision": s2.STUDENT_FULL_REVISION,
                "llama": None,
                "dtype_census": None,
            }

        straight_dir = tmp_path / "straight"
        chunked_dir = tmp_path / "chunked"
        base_args = [
            "--gold",
            str(gold_dir),
            "--rnd-commit",
            "e48cc90c",
            "--rollout-only",
            "--g",
            "2",
            "--max-new",
            "32",
            "--quote-reward",
            "v2",
            "--pass-id",
            "replay_eq_pass",
        ]

        # 1. Run straight through
        a_straight = ap.parse_args([*base_args, "--out", str(straight_dir)])
        assert s2.execute(a_straight, build=make_bundle, rig=_FakeRig(a_straight)) == 0

        # 2. Run in 3 chunks of --stop-after-new 2 (6 items total)
        for chunk_idx in range(3):
            a_chunk = ap.parse_args([*base_args, "--out", str(chunked_dir), "--stop-after-new", "2"])
            rc = s2.execute(a_chunk, build=make_bundle, rig=_FakeRig(a_chunk))
            if chunk_idx < 2:
                assert rc == 1
                assert (chunked_dir / "s2-hf-prepass-summary.json").is_file()
                assert (chunked_dir / "s2-hf.json").is_file()
            else:
                assert rc == 0

        # Assert byte-equal s2-hf-prepass.jsonl, s2-hf-prepass-summary.json, and s2-hf-save-all.jsonl.gz!
        assert (straight_dir / "s2-hf-prepass.jsonl").read_bytes() == (
            chunked_dir / "s2-hf-prepass.jsonl"
        ).read_bytes()
        assert (straight_dir / "s2-hf-prepass-summary.json").read_bytes() == (
            chunked_dir / "s2-hf-prepass-summary.json"
        ).read_bytes()
        assert (straight_dir / "s2-hf-save-all.jsonl.gz").read_bytes() == (
            chunked_dir / "s2-hf-save-all.jsonl.gz"
        ).read_bytes()
        assert s2.read_save_all(straight_dir / "s2-hf-save-all.jsonl.gz") == s2.read_save_all(
            chunked_dir / "s2-hf-save-all.jsonl.gz"
        )

    @needs_torch
    def test_crashsafe_05_all_item2_writers_crash_safe(self, tmp_path: Path) -> None:
        from contextlib import nullcontext

        import s2_interview as s2i
        import s2_interview_tag as s2t
        import torch

        gold_dir = _prepass_gold(tmp_path / "gold")
        tok = s2.DryTokenizer()
        ap = s2.build_parser()

        # (A) Training-rollout save-all writer + steps.jsonl writer in s2_grpo_smoke.py
        train_dir = tmp_path / "train_kill"
        a_train = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(train_dir),
                "--steps",
                "2",
                "--prompts",
                "4",
                "--g",
                "2",
                "--max-new",
                "32",
            ]
        )
        train_rollout_calls = 0

        def make_train_bundle(_args_ns):
            class _TinyModel:
                device = torch.device("cpu")

                def __init__(self) -> None:
                    self._p = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

                def parameters(self):
                    return [self._p]

                def __call__(self, input_ids, attention_mask=None, **kwargs):
                    logits = (
                        torch.zeros(
                            (input_ids.shape[0], input_ids.shape[1], tok.vocab_size), dtype=torch.float32
                        )
                        + self._p * 0.0
                    )
                    return Namespace(logits=logits)

            def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
                nonlocal train_rollout_calls
                train_rollout_calls += 1
                # Step 0 has 2 groups (calls 1 and 2). Step 1 has 2 groups (calls 3 and 4).
                # Kill on call 4 (after step 0 committed and step 1 group 0 wrote save-all, before step 1 finishes).
                if train_rollout_calls == 4:
                    raise _SimulatedHardKill("kill mid step 1 in training")
                p_ids = tok.encode(prompt_text)
                c0 = tok.encode(_reply("supported", "alpha beta gamma", think="tr0")) + [tok.eos_token_id]
                c1 = tok.encode(_reply("unsupported", "alpha beta gamma", think="tr1")) + [tok.eos_token_id]
                return p_ids, [c0, c1][:g_cnt], [False, False][:g_cnt]

            return {
                "model": _TinyModel(),
                "tok": tok,
                "cuda": _FakeCuda(),
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": True},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 250,
                "revision": s2.STUDENT_FULL_REVISION,
                "llama": None,
                "dtype_census": None,
            }

        with pytest.raises(_SimulatedHardKill):
            s2.execute(a_train, build=make_train_bundle, rig=_FakeRig(a_train))

        sa_train_path = train_dir / "s2-hf-save-all.jsonl.gz"
        assert sa_train_path.is_file()
        # Step 0 (2 groups * 2 samples = 4) + Step 1 Group 0 (2 samples) = 6 records on disk at kill
        assert len(s2.read_save_all(sa_train_path)) == 6

        # Resume training run: the 3 already-completed groups are kept and only the remaining group is run
        assert s2.execute(a_train, build=make_train_bundle, rig=_FakeRig(a_train)) == 1
        final_train_sa = s2.read_save_all(sa_train_path)
        assert len(final_train_sa) == 2 * 2 * 2  # 2 steps * 2 groups * 2 samples = 8
        assert [(r["step"], r["group"], r["sample_index"]) for r in final_train_sa] == [
            (s_i, g_i, samp_i) for s_i in range(2) for g_i in range(2) for samp_i in range(2)
        ]

        # (B) Post-pass writer in s2_grpo_smoke.py
        ckpt_dir = tmp_path / "ckpt_post"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / "adapter_model.safetensors").write_bytes(b"postpass-ckpt-bytes")
        post_dir = tmp_path / "postpass_kill"
        a_post = ap.parse_args(
            [
                "--gold",
                str(gold_dir),
                "--rnd-commit",
                "e48cc90c",
                "--out",
                str(post_dir),
                "--postpass",
                "--pass-seed",
                "2",
                "--checkpoint",
                str(ckpt_dir),
                "--g",
                "2",
                "--max-new",
                "32",
            ]
        )
        post_calls = 0

        def make_post_bundle(_args_ns):
            def rollout(prompt_text: str, g_cnt: int, max_n: int, temp: float):
                nonlocal post_calls
                post_calls += 1
                if post_calls == 3:
                    raise _SimulatedHardKill("kill postpass on item 3")
                p_ids = tok.encode(prompt_text)
                c0 = tok.encode(_reply("supported", "alpha beta gamma", think="p0")) + [tok.eos_token_id]
                c1 = tok.encode(_reply("cannot_tell", "", think="p1")) + [tok.eos_token_id]
                return p_ids, [c0, c1][:g_cnt], [False, False][:g_cnt]

            return {
                "model": Namespace(device="cpu"),
                "tok": tok,
                "cuda": _FakeCuda(),
                "mode": lambda training: None,
                "state": lambda: {"grad_ckpt": False},
                "optimiser": Namespace(zero_grad=lambda set_to_none=True: None, step=lambda: None),
                "ref_adapter_off": nullcontext,
                "rollout": rollout,
                "reward_fn": lambda text, row, tt, truncated=False: s2.reward(
                    text, row["label"], s2.passages_of(row), tt, truncated=truncated
                ),
                "sync": lambda: None,
                "peak_alloc_mib": lambda: 300,
                "revision": s2.STUDENT_FULL_REVISION,
                "adapter_loaded": True,
                "llama": None,
                "dtype_census": None,
            }

        with pytest.raises(_SimulatedHardKill):
            s2.execute(a_post, build=make_post_bundle, rig=_FakeRig(a_post))
        sa_post_path = post_dir / "s2-hf-save-all.jsonl.gz"
        assert len(s2.read_save_all(sa_post_path)) == 2 * 2
        assert s2.execute(a_post, build=make_post_bundle, rig=_FakeRig(a_post)) == 0
        assert len(s2.read_save_all(sa_post_path)) == 6 * 2

        # (C) s2_interview.py writer (run_interview_for_cohort)
        rows_by_id = {
            f"iv_{i}": _row(f"iv_{i}", claim=f"Claim {i} has 64 bytes.", text="Passage text.")
            for i in range(3)
        }
        t1_ct = "<think>missing bytes</think>" + json.dumps(
            {
                "verdict": "cannot_tell",
                "reason": "missing",
                "evidence_quotes": [{"part": "64 bytes", "verdict": "cannot_tell", "quote": ""}],
            }
        )
        cohort = [
            {
                "item_id": f"iv_{i}",
                "cohort": "unresolved_ct",
                "baseline_ct_rate": 0.5,
                "baseline_majority_ct": True,
                "checkpoint_ct_rate": 0.0,
                "checkpoint_majority_ct": False,
                "baseline_first_ct": {"text": t1_ct, "sample_index": 0},
                "checkpoint_first_ct": None,
            }
            for i in range(3)
        ]

        class _StubInterviewRunner:
            def __init__(self, fail_on_item: str | None = None) -> None:
                self.fail_on_item = fail_on_item

            def ask_question(self, **kw) -> dict:
                cid = str(kw["row"]["id"])
                if self.fail_on_item == cid and kw["question_id"] == "Q2":
                    raise _SimulatedHardKill(f"kill interview on {cid}")
                q_id = kw["question_id"]
                ans = (
                    "64 bytes"
                    if q_id == "Q1"
                    else "NONE"
                    if q_id == "Q4"
                    else "Need size.\nMISSING_FACT: The buffer size is 64 bytes."
                    if q_id == "Q2"
                    else "I choose supported because of context."
                )
                return {
                    "item_id": cid,
                    "policy": kw["policy"],
                    "question_id": q_id,
                    "cold_control": bool(kw.get("cold_control", False)),
                    "seed": 1,
                    "turn1_ stripped_in_prompt": True,
                    "original_turn1_thinking_tokens": 5,
                    "turn1_thinking": "missing bytes",
                    "prompt_tokens": 20,
                    "thinking_tokens": 4,
                    "answer_tokens": 6,
                    "think_truncated": False,
                    "answer_truncated": False,
                    "thinking": "short thought",
                    "answer": ans,
                    "text": f"<think>short thought</think>{ans}",
                    "ngram_repetition_rate_4": 0.0,
                }

            def run_verifier(self, cf_row, cf_seed, **kw) -> str:
                return _reply("supported", "Passage text.", think="cf ok")

        iv_kill_dir = tmp_path / "interview_kill"
        with pytest.raises(_SimulatedHardKill):
            s2i.run_interview_for_cohort(
                cohort=cohort,
                rows_by_id=rows_by_id,
                runner=_StubInterviewRunner(fail_on_item="iv_1"),
                out_dir=iv_kill_dir,
                include_cold_control=False,
            )
        iv_sa_path = iv_kill_dir / "s2-interview-save-all.jsonl.gz"
        assert iv_sa_path.is_file()
        assert len(s2.read_save_all(iv_sa_path)) == 4  # item iv_0's 4 questions already on disk!

        # Resume interview to completion and compare against a clean straight-through run
        receipt_resumed = s2i.run_interview_for_cohort(
            cohort=cohort,
            rows_by_id=rows_by_id,
            runner=_StubInterviewRunner(fail_on_item=None),
            out_dir=iv_kill_dir,
            include_cold_control=False,
        )
        iv_clean_dir = tmp_path / "interview_clean"
        receipt_clean = s2i.run_interview_for_cohort(
            cohort=cohort,
            rows_by_id=rows_by_id,
            runner=_StubInterviewRunner(fail_on_item=None),
            out_dir=iv_clean_dir,
            include_cold_control=False,
        )
        assert receipt_resumed == receipt_clean
        assert s2.read_save_all(iv_sa_path) == s2.read_save_all(
            iv_clean_dir / "s2-interview-save-all.jsonl.gz"
        )

        # (D) s2_interview_tag.py writer (run_two_judge_tagging)
        q2_recs = [
            {
                "item_id": f"tag_{i}",
                "policy": "baseline",
                "question_id": "Q2",
                "cold_control": False,
                "original_turn1_thinking_tokens": 10 + i,
                "turn1_thinking": f"missing fact {i} in evidence",
                "answer": f"MISSING_FACT: missing fact {i} in evidence",
            }
            for i in range(4)
        ]
        controls = s2t.build_judge_controls(q2_recs, seed=123)

        def make_judge(fail_on_call: int | None = None):
            call_n = 0

            def _judge(t1_think: str, q2_ans: str) -> dict[str, str]:
                nonlocal call_n
                if not t1_think.strip():
                    return {"missing_element_match": "none", "reason_type": "generic_or_none"}
                if t1_think.strip() in q2_ans:
                    # Only count non-control tagging calls if we want, or count all calls after controls (75 controls)
                    call_n += 1
                    if fail_on_call is not None and call_n == fail_on_call:
                        raise _SimulatedHardKill("kill tagging mid-loop")
                    return {"missing_element_match": "same", "reason_type": "lacks"}
                return {"missing_element_match": "different", "reason_type": "conflicts"}

            return _judge

        tag_kill_dir = tmp_path / "tag_kill"
        # 30 verbatim controls have t1_think in q2_ans, so fail_on_call=33 fails on the 3rd real Q2 item (after 2 items written)
        with pytest.raises(_SimulatedHardKill):
            s2t.run_two_judge_tagging(
                interview_q2_records=q2_recs,
                judge_a_name="claude-3-5-sonnet",
                judge_b_name="gpt-4o",
                judge_a_fn=make_judge(fail_on_call=33),
                judge_b_fn=make_judge(fail_on_call=None),
                out_dir=tag_kill_dir,
                controls=controls,
            )
        tags_path = tag_kill_dir / "s2-interview-tags.jsonl"
        assert tags_path.is_file()
        partial_tags = [json.loads(line) for line in tags_path.read_text(encoding="utf-8").splitlines() if line]
        assert len(partial_tags) == 2

        # Resume tagging to completion and compare against clean run
        sum_resumed = s2t.run_two_judge_tagging(
            interview_q2_records=q2_recs,
            judge_a_name="claude-3-5-sonnet",
            judge_b_name="gpt-4o",
            judge_a_fn=make_judge(fail_on_call=None),
            judge_b_fn=make_judge(fail_on_call=None),
            out_dir=tag_kill_dir,
            controls=controls,
        )
        tag_clean_dir = tmp_path / "tag_clean"
        sum_clean = s2t.run_two_judge_tagging(
            interview_q2_records=q2_recs,
            judge_a_name="claude-3-5-sonnet",
            judge_b_name="gpt-4o",
            judge_a_fn=make_judge(fail_on_call=None),
            judge_b_fn=make_judge(fail_on_call=None),
            out_dir=tag_clean_dir,
            controls=controls,
        )
        assert sum_resumed == sum_clean
        assert tags_path.read_bytes() == (tag_clean_dir / "s2-interview-tags.jsonl").read_bytes()

    def test_crashsafe_06_stop_after_new_zero_refuses(self, tmp_path: Path) -> None:
        gold_dir = _prepass_gold(tmp_path / "gold")
        ap = s2.build_parser()
        help_text = ap.format_help()
        assert "--stop-after-new" in help_text

        for bad_n in ("0", "-1"):
            a_bad = ap.parse_args(
                [
                    "--gold",
                    str(gold_dir),
                    "--rnd-commit",
                    "e48cc90c",
                    "--out",
                    str(tmp_path / f"bad_{bad_n}"),
                    "--rollout-only",
                    "--stop-after-new",
                    bad_n,
                ]
            )
            with pytest.raises(SystemExit, match="invalid_stop_after_new"):
                s2.validate_pass_mode_args(a_bad)
            with pytest.raises(SystemExit, match="invalid_stop_after_new"):
                s2.execute(a_bad, build=s2.dry_bundle)

