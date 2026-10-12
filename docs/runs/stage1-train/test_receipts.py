"""The receipt writer and its firewall (schema stage1-receipt-v1). The held-out files here are planted: the real
sealed set never enters a test."""

import ast
import hashlib
import json
from pathlib import Path

import pytest

import receipts

HERE = Path(__file__).resolve().parent
SEALED_LINE = "the planted sealed line says the north gate opens at seven every morning except holidays"
DEV_LINE = "the planted dev material records that the ferry leaves the harbour at noon on fridays only"
QUESTION = "In a few sentences, describe how you decide whether a statement is backed by the material in front of you?"


@pytest.fixture
def fw(tmp_path):
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text(json.dumps({"id": "sx-0123456789", "material": SEALED_LINE,
                                  "ideal_tree": {"node_id": "held-node-7"}}) + "\n", encoding="utf-8")
    dev = tmp_path / "dev.jsonl"
    dev.write_text(json.dumps({"id": "pl-abcdef0123", "material": DEV_LINE}) + "\n", encoding="utf-8")
    sha = receipts.sha256_file  # the pin is of the content with CRLF read as LF
    return receipts.Firewall.load(sealed=sealed, sealed_sha256=sha(sealed), dev={dev: sha(dev)},
                                  questions=[QUESTION])


def writer(tmp_path, fw, **kw):
    args = dict(root=tmp_path / "out", round=2, arm="ROLE", seed=0, run_id="r2-s0-ROLE", adapter_sha256="ab" * 32,
                gates_sha256="0e" * 32, stuck_rule_sha256="71" * 32, firewall=fw)
    args.update(kw)
    return receipts.ReceiptWriter(**args)


QUIZ = {"question_id": "T05-D-03-v2", "base_item_id": "T05-D-03", "variant": 2, "lesson": "T05", "kind": "D",
        "tier": 4, "attempt": 2, "ledger_box": 1, "ledger_state": {"box": 1, "misses": 1},
        "verdict_given": "unsupported", "trace": "I read the notice. It is closed on Sundays.", "answer": "x"}


def test_a_clean_row_lands_at_the_path_its_header_names(tmp_path, fw):
    w = writer(tmp_path, fw)
    row = w.write("train_quiz", QUIZ)
    p = tmp_path / "out" / "2" / "ROLE" / "0" / "receipts" / "train_quiz.jsonl"
    got = json.loads(p.read_text(encoding="utf-8"))
    assert got == row
    assert list(got)[:10] == ["schema", "source", "round", "seed", "arm", "run_id", "adapter_sha256",
                              "gates_sha256", "stuck_rule_sha256", "written_at"]
    assert got["schema"] == "stage1-receipt-v1" and got["round"] == 2 and got["seed"] == 0


@pytest.mark.parametrize("change, code", [
    ({"question_id": "sx-0123456789"}, "HELD_OUT_ID"),
    ({"base_item_id": "pl-0000000000"}, "HELD_OUT_ID"),
    ({"base_item_id": "rv-0000000000"}, "HELD_OUT_ID"),
    ({"ledger_state": {"item_id": "held-node-7"}}, "HELD_OUT_ID"),  # an id held in the sealed file, nested
    ({"trace": "Thinking: " + SEALED_LINE + "."}, "HELD_OUT_TEXT"),
    ({"answer": "As the notice says, " + DEV_LINE}, "HELD_OUT_TEXT"),
    ({"trace": "Asked: " + QUESTION}, "HELD_OUT_TEXT"),
    ({"q": 3}, "EVAL_RECORD"),
    ({"sitting": "baseline"}, "EVAL_RECORD"),
])
def test_the_firewall_refuses(tmp_path, fw, change, code):
    w = writer(tmp_path, fw)
    with pytest.raises(receipts.Refused) as e:
        w.write("train_quiz", {**QUIZ, **change})
    assert e.value.code == code
    assert not (tmp_path / "out").exists()  # nothing written
    assert w.refused == {code: 1}
    assert SEALED_LINE not in str(e.value) and DEV_LINE not in str(e.value)  # never echoes held-out text


def test_an_evaluation_record_is_refused(tmp_path, fw):
    """The eval-record firewall test (R&D): a record shaped like the harness's raw.jsonl, or eval_adapter's, can't
    be written as a receipt under any source."""
    raw = {"task_id": "T05-D-03", "phase": "A-after", "arm": "ROLE", "seed": 0, "harness_sha256": "4f" * 32,
           "turns": [{"thinking": "...", "answer": "..."}], "final": {"verdict": "supported"},
           "grade": {"verdict_correct": True}}
    w = writer(tmp_path, fw)
    for source in receipts.SOURCES:
        with pytest.raises(receipts.Refused) as e:
            w.write(source, {k: v for k, v in raw.items() if k not in ("arm", "seed")})
        assert e.value.code == "EVAL_RECORD"
    assert not (tmp_path / "out").exists()


def test_sources_and_arms_are_closed(tmp_path, fw):
    with pytest.raises(receipts.Refused) as e:
        writer(tmp_path, fw).write("dev", QUIZ)
    assert e.value.code == "SOURCE"
    for bad in ("sealed", "interview", "eval", "generation"):
        with pytest.raises(receipts.Refused):
            writer(tmp_path, fw).write(bad, QUIZ)
    with pytest.raises(ValueError):
        writer(tmp_path, fw, arm="BASE")
    with pytest.raises(ValueError):
        writer(tmp_path, fw, round="2")
    with pytest.raises(ValueError):
        writer(tmp_path, None)
    with pytest.raises(ValueError):
        writer(tmp_path, fw).write("train_quiz", {**QUIZ, "arm": "NO-ROLE"})  # a field can't overwrite the header


def test_the_firewall_refuses_to_arm_on_a_changed_or_missing_file(tmp_path):
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text("{}\n")
    with pytest.raises(SystemExit):
        receipts.Firewall.load(sealed=sealed, sealed_sha256="0" * 64, dev={}, questions=[])
    with pytest.raises(SystemExit):
        receipts.Firewall.load(sealed=tmp_path / "missing.jsonl", sealed_sha256="0" * 64, dev={}, questions=[])


def test_the_interview_questions_are_read_without_importing_interview_py():
    qs = receipts.interview_questions()
    assert len(qs) == 12
    assert hashlib.sha256(json.dumps(qs).encode()).hexdigest().startswith("b59927e1")


def imports_of(path: Path) -> set:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module.split(".")[0])
    return out


def test_the_evaluation_code_never_imports_the_receipt_writer():
    """Sealed and pre-interview sittings run through stage1-eval (harness, interview, student, mapper, the scorers)
    and eval_adapter.py. None of them may import receipts or run, so no sitting can write a receipt."""
    eval_files = sorted((HERE.parent / "stage1-eval").glob("*.py")) + [HERE / "eval_adapter.py"]
    assert len(eval_files) >= 8
    for f in eval_files:
        assert not imports_of(f) & {"receipts", "run", "role_store", "role_set"}, f.name


def test_a_crlf_checkout_of_a_held_out_file_still_arms_the_firewall(tmp_path):
    """Found on the 5090 smoke (2026-10-11): with core.autocrlf the DEV files check out with CRLF and failed
    their LF pins, so the firewall refused to arm and no receipt was written."""
    lf = json.dumps({"id": "sx-0123456789", "material": SEALED_LINE}) + "\n"
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_bytes(lf.replace("\n", "\r\n").encode())
    pin = hashlib.sha256(lf.encode()).hexdigest()  # the pin is of the LF bytes, as git stores them
    fw = receipts.Firewall.load(sealed=sealed, sealed_sha256=pin, dev={}, questions=[])
    assert "sx-0123456789" in fw.held_ids


def test_the_repository_dev_files_match_their_pins_however_they_are_checked_out():
    for name, pin in receipts.DEV_FILES.items():
        assert receipts.sha256_file(receipts.DEV_DIR / name) == pin, name
