"""The determinism wrapper: the target runs unchanged inside the pinned settings; the machine line is a hard stop;
a strict deterministic raise stops for a --warn-only rerun; warned ops and the settings are recorded; a resume under
a different wrapper config is refused. A fake target and machine line: no GPU, no model."""

import json
import sys
import types
import warnings

import pytest
import torch

import det_run

LINE = {"gpu": "NVIDIA GeForce RTX 5090", "driver": "617.42", "cuda": "13.4", "torch": "2.16.0",
        "transformers": "5.19.0"}


@pytest.fixture
def rig(tmp_path, monkeypatch):
    seen = {}

    def main(argv):
        seen["argv"] = argv
        seen["deterministic"] = torch.are_deterministic_algorithms_enabled()
        seen["warn_only"] = torch.is_deterministic_algorithms_warn_only_enabled()
        seen["benchmark"] = torch.backends.cudnn.benchmark
        if seen.get("raise"):
            raise RuntimeError("index_copy_cuda does not have a deterministic implementation")
        if seen.get("warn"):
            warnings.warn("index_copy_cuda does not have a deterministic implementation, but you set warn_only")
        return 0
    monkeypatch.setitem(sys.modules, "harness", types.SimpleNamespace(main=main))
    monkeypatch.setattr(det_run, "machine_line", lambda: dict(LINE))
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(LINE), encoding="utf-8")
    yield {"seen": seen, "frozen": frozen, "out": tmp_path / "m1"}
    torch.use_deterministic_algorithms(False)


def args(rig, *own):
    return [*own, "--", "harness", "--tasks", "t.jsonl", "--out", str(rig["out"]), "--seed", "0"]


def config(rig):
    return json.loads((rig["out"] / "wrapper-config.json").read_text(encoding="utf-8"))


def test_the_target_runs_unchanged_inside_the_pinned_settings_and_they_are_recorded(rig):
    assert det_run.main(args(rig, "--machine-line", str(rig["frozen"]))) == 0
    s = rig["seen"]
    assert s["argv"] == ["--tasks", "t.jsonl", "--out", str(rig["out"]), "--seed", "0"]
    assert s["deterministic"] and not s["warn_only"] and s["benchmark"] is False
    c = config(rig)
    assert c["sdpa_backend"] == "flash" and c["warn_only"] is False and c["cublas_workspace_config"] == ":4096:8"
    assert c["machine_line"] == LINE and c["outcome"] == "exit 0" and len(c["wrapper_sha256"]) == 64
    assert "cpu" in c["rng_at_start"]


def test_a_machine_line_difference_stops_before_the_target(rig, monkeypatch, capsys):
    monkeypatch.setattr(det_run, "machine_line", lambda: {**LINE, "driver": "617.14"})
    assert det_run.main(args(rig, "--machine-line", str(rig["frozen"]))) == 5
    assert "argv" not in rig["seen"] and "driver: '617.42' -> '617.14'" in capsys.readouterr().out


def test_g0_writes_the_frozen_line(rig, tmp_path):
    line = tmp_path / "new.json"
    assert det_run.main(args(rig, "--write-machine-line", str(line))) == 0
    assert json.loads(line.read_text(encoding="utf-8")) == LINE


def test_writing_the_frozen_line_refuses_an_existing_file(rig, capsys):
    """A wrong flag on a later sitting must not re-freeze the line and pass (R&D on #122)."""
    rig["frozen"].write_text(json.dumps({**LINE, "driver": "617.14"}), encoding="utf-8")
    assert det_run.main(args(rig, "--write-machine-line", str(rig["frozen"]))) == 7
    assert "argv" not in rig["seen"] and "refusing to overwrite" in capsys.readouterr().out
    assert json.loads(rig["frozen"].read_text(encoding="utf-8"))["driver"] == "617.14"  # untouched


def test_recorded_warnings_are_still_shown(rig, capsys):
    rig["seen"]["warn"] = True
    with pytest.warns(UserWarning, match="deterministic implementation"):
        det_run.main(args(rig, "--machine-line", str(rig["frozen"]), "--warn-only"))


def test_a_strict_deterministic_raise_stops_for_a_warn_only_rerun(rig):
    rig["seen"]["raise"] = True
    assert det_run.main(args(rig, "--machine-line", str(rig["frozen"]))) == 4
    assert config(rig)["outcome"].startswith("strict deterministic op raised")


def test_warn_only_logs_every_warned_op(rig):
    rig["seen"]["warn"] = True
    assert det_run.main(args(rig, "--machine-line", str(rig["frozen"]), "--warn-only")) == 0
    assert rig["seen"]["warn_only"] is True
    assert config(rig)["warned_ops"] == [
        "index_copy_cuda does not have a deterministic implementation, but you set warn_only"]


def test_a_resume_under_a_different_wrapper_config_is_refused(rig):
    assert det_run.main(args(rig, "--machine-line", str(rig["frozen"]))) == 0
    assert det_run.main(args(rig, "--machine-line", str(rig["frozen"]), "--warn-only")) == 6
    assert det_run.main(args(rig, "--machine-line", str(rig["frozen"]))) == 0  # the same config resumes


def test_a_target_outside_the_three_is_refused(rig):
    with pytest.raises(SystemExit, match="must be one of"):
        det_run.main(["--machine-line", str(rig["frozen"]), "--", "train", "--out", str(rig["out"])])
