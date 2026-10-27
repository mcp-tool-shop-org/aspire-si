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
