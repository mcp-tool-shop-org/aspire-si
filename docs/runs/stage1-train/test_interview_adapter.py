"""The pre-interview for a trained adapter: interview.py runs unchanged with the adapter attached to the student it
loads; a smoke adapter never spends the sealed questions; one run directory never mixes two adapters."""

import json

import pytest

import interview_adapter as ia
from interview_adapter import interview, student

BASE = object()


class Attached:
    def __init__(self, base, path):
        self.base, self.path, self.evaluated = base, path, False

    def eval(self):
        self.evaluated = True


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A fake student and generate: no weights, no GPU. The model each generate call sees is recorded."""
    seen = []
    monkeypatch.setattr(student, "load_student", lambda: BASE)
    monkeypatch.setattr(student, "load_tokenizer", lambda: "tok")
    monkeypatch.setattr(student, "verify_weights", lambda: {"weights": "ok"})
    monkeypatch.setattr(student, "identity", lambda: {"model": "fake"})
    monkeypatch.setattr("peft.PeftModel.from_pretrained", lambda base, path: Attached(base, path))

    def generate(model, tok, messages, decoding, seed):
        seen.append(model)
        return {"text": "<think>t</think>an answer", "prompt_tokens": 10, "new_tokens": 3, "truncated": False,
                "seconds": 0.1}
    monkeypatch.setattr(interview.harness, "generate", generate)

    def adapter(name, manifest):
        a = tmp_path / name / "adapter"
        a.mkdir(parents=True)
        (a / "adapter_model.safetensors").write_bytes(name.encode())
        if manifest is not None:
            (a.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return a
    return {"seen": seen, "adapter": adapter, "out": tmp_path / "interview"}


FROZEN = {"freeze_report": {"shortfalls": []}}


def test_every_question_is_asked_of_the_student_with_the_adapter_attached(rig):
    a = rig["adapter"]("r1", FROZEN)
    assert ia.main(["--adapter", str(a), "--out", str(rig["out"]), "--seeds", "0", "--sitting", "round1",
                    "--arm", "ROLE"]) == 0
    rows = [json.loads(x) for x in (rig["out"] / "interview.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == len(interview.QUESTIONS) == len(rig["seen"])
    assert {r["sitting"] for r in rows} == {"round1"} and {r["arm"] for r in rows} == {"ROLE"}
    m = rig["seen"][0]
    assert isinstance(m, Attached) and m.base is BASE and m.path == str(a) and m.evaluated
    assert all(x is m for x in rig["seen"])
    cfg = json.loads((rig["out"] / "adapter-config.json").read_text(encoding="utf-8"))
    assert cfg["interview_py_sha256"] == json.loads(
        (rig["out"] / "interview-config.json").read_text(encoding="utf-8"))["interview_py_sha256"]
    assert student.load_student() is BASE  # the wrapper is undone afterwards


@pytest.mark.parametrize("manifest, why", [(None, "no manifest"),
                                           ({"freeze_report": {"partial_set": True}}, "partial-set")])
def test_a_smoke_adapter_never_spends_the_sealed_questions(rig, manifest, why):
    a = rig["adapter"]("smoke", manifest)
    with pytest.raises(SystemExit, match=why):
        ia.main(["--adapter", str(a), "--out", str(rig["out"])])
    assert rig["seen"] == [] and not rig["out"].exists()


def test_a_resume_under_a_different_adapter_is_refused(rig):
    first = rig["adapter"]("r1", FROZEN)
    ia.main(["--adapter", str(first), "--out", str(rig["out"]), "--seeds", "0"])
    other = rig["adapter"]("r1b", FROZEN)
    with pytest.raises(SystemExit, match="different adapter"):
        ia.main(["--adapter", str(other), "--out", str(rig["out"]), "--seeds", "0"])
    n = len(rig["seen"])
    ia.main(["--adapter", str(first), "--out", str(rig["out"]), "--seeds", "0"])  # same adapter: resumes, asks none
    assert len(rig["seen"]) == n
