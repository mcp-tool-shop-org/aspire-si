"""The freeze orchestrator, on the CPU. No model weights, except one test that skips unless the student's
tokenizer is already cached.

The four failure fixtures each feed a broken input and assert the named check is the one that fails:
a leaked claim, a spent n = 0 variant, a NO-ROLE composition mismatch over one item, a malformed apply item.
A partial or failed report cannot write FROZEN.md. The same inputs write the same bytes.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
import freeze  # noqa: E402
import no_role  # noqa: E402
import receipts  # noqa: E402

DATA = HERE.parent / "stage1-training-data"
ORDER = (
    "pilot-t04.jsonl",
    "batch2-t01.jsonl", "batch2-t14.jsonl", "batch2-t16.jsonl",
    "batch3-t05.jsonl", "batch3-t13.jsonl", "batch3-t17.jsonl",
    "batch4-t02.jsonl", "batch4-t03.jsonl", "batch4-t06.jsonl",
    "batch5-t07.jsonl", "batch5-t08.jsonl", "batch5-t09.jsonl",
    "batch6-t10.jsonl", "batch6-t11.jsonl", "batch6-t12.jsonl",
)
CLAIM = "the quartz lantern on the north pier was lit only on the third thursday of frost"
PHRASE = "quartz lantern frost thursday never appears inside this maintenance card at all"


def tokenizer_available():
    """The same cache check the runner uses. A missing snapshot skips; it does not download."""
    sys.path.insert(0, str(HERE.parent / "stage1-eval"))
    try:
        import student
        return (student.snapshot_dir() / "tokenizer.json").is_file()
    except BaseException:  # snapshot_dir exits when HF_HOME is unset
        return False


def write_rows(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8", newline="\n")


def planted_sealed(tmp: Path, claim: str = "a planted sealed claim that no training item states") -> Path:
    path = tmp / "sealed.jsonl"
    path.write_text(json.dumps({"id": "sx-planted0001", "claim": claim,
                                "material": "one planted line"}) + "\n", encoding="utf-8", newline="\n")
    return path


def assert_public(report) -> None:
    blob = json.dumps(report)
    assert not re.search(r"[A-Za-z]:[\\/]", blob)
    assert "/Users/" not in blob and "/home/" not in blob


def check(tmp, files, monkeypatch, *, partial=True, sealed=None, apply=None, count="words"):
    """Run `freeze check` in process. Word counts stand in for the tokenizer, so CI needs no snapshot."""
    if count == "words":
        monkeypatch.setattr(freeze, "cli_count", lambda: (lambda text: len(str(text).split())))
    elif count == "missing":
        def missing():
            raise SystemExit("tokenizer: not available")

        monkeypatch.setattr(freeze, "cli_count", missing)
    out = tmp / "out"
    argv = ["check", "--data", *[str(p) for p in files], "--out", str(out),
            "--sealed", str(sealed if sealed is not None else tmp / "no-sealed.jsonl")]
    if partial:
        argv.append("--partial-set")
    if apply:
        argv += ["--apply", *[str(p) for p in apply]]
    code = freeze.main(argv)
    report = json.loads((out / freeze.REPORT_NAME).read_text(encoding="utf-8"))
    return code, report


def test_a_partial_pilot_passes_and_cannot_write_frozen(tmp_path, monkeypatch):
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch)
    assert code == 0, report["failed"]
    assert report["partial_set"] is True
    validate = report["checks"]["validate_training_set"]
    assert validate["ok"] and validate["freeze_shortfalls"]  # floors fall short; the check still passes
    assert report["checks"]["apply"]["skipped"] is True
    assert report["checks"]["apply"]["predicted_holds"] == []
    assert report["checks"]["apply"]["predicted_holds_by_lesson"] == {}
    assert report["no_role"]["pin"] == freeze.NO_ROLE_PIN
    assert report["training_files"][0]["path"] == "docs/runs/stage1-training-data/pilot-t04.jsonl"
    assert_public(report)
    blob = (tmp_path / "out" / freeze.NO_ROLE_NAME).read_bytes()
    assert hashlib.sha256(blob).hexdigest() == report["no_role"]["sha256"]
    frozen, pin = tmp_path / "FROZEN.md", tmp_path / "pin.jsonl"
    status = freeze.main(["write", "--report", str(tmp_path / "out" / freeze.REPORT_NAME),
                          "--out", str(frozen), "--no-role-out", str(pin)])
    assert status == 1 and not frozen.exists() and not pin.exists()
    assert not (freeze.REPO / freeze.NO_ROLE_PIN).exists()


@pytest.mark.skipif(not tokenizer_available(), reason="the student's tokenizer isn't cached here")
@pytest.mark.timeout(600)
def test_batches_1_to_6_pass_apart_from_the_freeze_floors(tmp_path):
    """The real count, so the NO-ROLE sha256 equals `no_role.py`.

    Floors may already be met. A partial run is still recorded, and it cannot write FROZEN.md.
    """
    files = [DATA / name for name in ORDER]
    assert all(p.is_file() for p in files)
    sealed = planted_sealed(tmp_path)
    out = tmp_path / "out"
    code = freeze.main(["check", "--partial-set", "--data", *[str(p) for p in files],
                        "--out", str(out), "--sealed", str(sealed)])
    report = json.loads((out / freeze.REPORT_NAME).read_text(encoding="utf-8"))
    assert code == 0, report["failed"]
    assert report["partial_set"] is True
    assert [Path(f["path"]).name for f in report["training_files"]] == list(ORDER)
    assert report["sealed"]["matches_pin"] is False  # a planted file, not the local sealed task
    assert_public(report)
    cli = tmp_path / "cli-no-role.jsonl"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HF_HUB_OFFLINE"] = "1"
    subprocess.run([sys.executable, str(HERE / "no_role.py"), "--data", *[str(p) for p in files],
                    "--out", str(cli)], check=True, env=env)
    assert receipts.sha256_file(out / freeze.NO_ROLE_NAME) == receipts.sha256_file(cli)
    assert report["checks"]["apply"]["skipped"] is True
    assert report["checks"]["apply"]["predicted_holds"] == []
    assert report["checks"]["apply"]["predicted_holds_by_lesson"] == {}


def test_a_leaked_claim_fails_leakage(tmp_path, monkeypatch):
    rows = data.load(DATA / "pilot-t04.jsonl")
    rows[0]["statement"] = CLAIM
    path = tmp_path / "pilot-t04.jsonl"
    write_rows(path, rows)
    code, report = check(tmp_path, [path], monkeypatch, sealed=planted_sealed(tmp_path, CLAIM))
    assert code == 1
    assert report["failed"] == ["leakage"]
    assert any(rows[0]["id"] in hit and "claim" in hit for hit in report["checks"]["leakage"]["hits"])
    assert_public(report)


def test_a_spent_n0_variant_fails_the_spent_check(tmp_path, monkeypatch):
    """The authoring gate refuses the same span, so validate fails too.

    The spent check is what names the item and n = 0.
    """
    rows = data.load(DATA / "batch2-t01.jsonl")
    item = next(r for r in rows if r["id"] == "T01-E-08")
    item["turns"][0]["thinking"] = (item["turns"][0].get("thinking") or "") + " " + PHRASE
    item["variant"] = {**item["variant"], "user": PHRASE + " What do you answer?"}
    path = tmp_path / "batch2-t01.jsonl"
    write_rows(path, rows)
    code, report = check(tmp_path, [path], monkeypatch, sealed=planted_sealed(tmp_path))
    assert code == 1
    assert "spent_variants" in report["failed"]
    assert set(report["failed"]) <= {"spent_variants", "validate_training_set"}
    hits = [h for h in report["checks"]["spent_variants"]["spent"]
            if h["id"] == "T01-E-08" and h["n"] == 0 and h["arm"] == "ROLE"]
    assert hits and hits[0]["reason"] == "spent"
    err = report["checks"]["validate_training_set"].get("error") or ""
    if "validate_training_set" in report["failed"]:
        assert "T01-E-08" in err and "spent" in err


def test_a_no_role_composition_mismatch_over_one_item_fails_no_role(tmp_path, monkeypatch):
    real = no_role.build

    def flipped(rows, count):
        built, exposure = real(rows, count)
        victim = next(r for r in built if r.get("kind") == "C")
        victim["verdict"] = "supported" if victim.get("verdict") != "supported" else "unsupported"
        return built, exposure

    monkeypatch.setattr(no_role, "build", flipped)
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch)
    assert code == 1
    assert report["failed"] == ["no_role"]
    diffs = report["checks"]["no_role"]["differences"]
    assert any(not d["within_tolerance"] and d["item_ids"] for d in diffs)


def test_a_one_sequence_difference_is_listed_and_allowed(tmp_path, monkeypatch):
    real = no_role.build

    def extra(rows, count):
        built, exposure = real(rows, count)
        built[0]["turns"] = list(built[0]["turns"]) + [dict(built[0]["turns"][-1])]
        return built, exposure

    monkeypatch.setattr(no_role, "build", extra)
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch)
    assert code == 0, report["failed"]
    diffs = report["checks"]["no_role"]["differences"]
    assert any(d["cell"] == "sequences" and d["within_tolerance"] and d["item_ids"] for d in diffs)
    assert report["checks"]["no_role"]["target_token_ratio"].count(".") == 1


def test_a_malformed_apply_item_fails_the_apply_check(tmp_path, monkeypatch):
    bad = tmp_path / "apply.jsonl"
    bad.write_text(json.dumps({"id": "T04-C-01", "split": "train", "lesson": "T04", "kind": "C"}) + "\n",
                   encoding="utf-8", newline="\n")
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch, apply=[bad])
    assert code == 1
    assert report["failed"] == ["apply"]
    assert report["checks"]["leakage"]["ok"] is True
    assert report["checks"]["leakage"]["apply_rows_checked"] is False
    assert report["checks"]["apply"]["problems"]
    assert report["checks"]["apply"]["predicted_holds"] == []


def test_a_models_sidecar_and_a_glob_order_are_refused(tmp_path, monkeypatch):
    code, report = check(tmp_path, [tmp_path / "pilot-t04.models.jsonl"], monkeypatch, count=None)
    assert code == 1 and report["failed"] == ["training_order"]
    assert "models" in report["checks"]["training_order"]["error"]
    # A shell glob sorts batch* before pilot. That order is refused, and the later checks do not run.
    swapped = [DATA / "batch2-t01.jsonl", DATA / "pilot-t04.jsonl"]
    code, report = check(tmp_path, swapped, monkeypatch, count=None)
    assert code == 1 and report["failed"] == ["training_order"]
    assert set(report["checks"]) == {"training_order"}


def test_a_build_error_is_a_failed_check(tmp_path, monkeypatch):
    def boom(rows, count):
        raise no_role.BuildError("T04-C-01: planted build failure")

    monkeypatch.setattr(no_role, "build", boom)
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch)
    assert code == 1
    assert report["failed"] == ["no_role"]
    assert "planted build failure" in report["checks"]["no_role"]["build_error"]


def test_composition_tolerance_is_one_item_or_sequence():
    def row(verdict="supported", turns=1):
        return {"id": "T04-C-01", "lesson": "T04", "kind": "C", "tier": 1, "verdict": verdict,
                "deciding": ["the line"], "escalate": "no",
                "turns": [{"thinking": "alpha beta", "reply": "VERDICT: supported"} for _ in range(turns)]}

    def count(text):
        return len(str(text).split())

    allowed = freeze.composition_gate([row()], [row(turns=2)], count)
    assert allowed["ok"]
    assert any(d["cell"] == "sequences" and d["within_tolerance"] and d["item_ids"] == ["T04-C-01"]
               for d in allowed["differences"])
    refused = freeze.composition_gate([row()], [row(verdict="unsupported")], count)
    assert not refused["ok"]
    assert any(d["cell"].startswith("verdict") and not d["within_tolerance"] and "T04-C-01" in d["item_ids"]
               for d in refused["differences"])


def test_a_missing_sealed_file_blocks_a_real_freeze(tmp_path, monkeypatch):
    _code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch)
    assert report["sealed"]["status"] == "not available"
    assert "sealed: not available" not in report["checks"]["leakage"]["hits"]
    # A directory, including a sealed output directory, is never read.
    _code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch, sealed=tmp_path)
    assert report["sealed"]["status"] == "not available"
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch, partial=False)
    assert code == 1
    assert "sealed: not available" in report["checks"]["leakage"]["hits"]
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch, partial=False,
                         sealed=planted_sealed(tmp_path))
    assert code == 1
    assert any(hit.startswith("sealed:") and receipts.SEALED_SHA256[:8] in hit
               for hit in report["checks"]["leakage"]["hits"])


def _passing_report(nr_sha: str) -> dict:
    checks = {name: {"ok": True} for name in freeze.CHECKS}
    checks["validate_training_set"] = {"ok": True, "freeze_shortfalls": []}
    checks["apply"] = {"ok": True, "given": True, "complete": True}
    checks["no_role"] = {"ok": True, "target_token_ratio": "0.624000"}
    return {
        "partial_set": False, "ok": True, "failed": [],
        "training_files": [
            {"path": "docs/runs/stage1-training-data/pilot-t04.jsonl", "items": 40, "sha256": "ab" * 32},
            {"path": "docs/runs/stage1-training-data/batch2-t01.jsonl", "items": 40, "sha256": "cd" * 32},
        ],
        "apply_files": [{"path": "docs/runs/stage1-apply/round1.jsonl", "items": 210, "sha256": "ef" * 32}],
        "no_role": {"file": freeze.NO_ROLE_NAME, "sha256": nr_sha, "pin": freeze.NO_ROLE_PIN, "items": 40},
        "counts": [{"lesson": "T04", "A": 8, "B": 8, "C": 8, "D": 8, "E": 8}],
        "sources": {"real": [{"repo": "mcp-tool-shop-org/roll", "commit": "abc", "path": "README.md",
                              "lines": "L1", "licence": "MIT License"}], "invented_items": 3},
        "sealed": {"status": "ok", "sha256": receipts.SEALED_SHA256, "matches_pin": True,
                   "path": "outputs/stage1-sealed-FROZEN/taskset-sealed.jsonl"},
        "checks": checks,
    }


def _write(tmp: Path, report: dict, nr: bytes, capsys):
    folder = tmp / "report"
    folder.mkdir(parents=True)
    (folder / freeze.NO_ROLE_NAME).write_bytes(nr)
    (folder / freeze.REPORT_NAME).write_text(json.dumps(report), encoding="utf-8", newline="\n")
    frozen, pin = tmp / "FROZEN.md", tmp / "pin.jsonl"
    code = freeze.main(["write", "--report", str(folder / freeze.REPORT_NAME),
                        "--out", str(frozen), "--no-role-out", str(pin)])
    out = capsys.readouterr().out.strip()
    return code, out, frozen, pin


def test_write_refuses_a_failed_report_and_is_byte_stable(tmp_path, capsys):
    nr = b'{"id": "T04-A-01-nr"}\n'
    report = _passing_report(hashlib.sha256(nr).hexdigest())
    code, sha1, frozen, pin = _write(tmp_path / "a", report, nr, capsys)
    assert code == 0
    text = frozen.read_bytes()
    assert b"\r" not in text and hashlib.sha256(text).hexdigest() == sha1
    # Training order is the report's order. A path sort would put batch2 ahead of pilot.
    body = text.decode("utf-8")
    assert body.index("pilot-t04.jsonl") < body.index("batch2-t01.jsonl")
    assert freeze.NO_ROLE_PIN in body and "round1.jsonl" in body
    assert "sha256 of the content with CRLF read as LF (`receipts.sha256_file`)" in body
    assert f"aspire-si-gates {freeze.GATES_VERSION}" in body
    assert f"freeze.py sha256: {receipts.sha256_file(Path(freeze.__file__))}" in body
    assert "sealed pin: e1e515b4" in body
    assert receipts.sha256_file(frozen) == sha1
    assert pin.read_bytes() == nr
    assert not (freeze.REPO / freeze.NO_ROLE_PIN).exists()
    code2, sha2, frozen2, _pin2 = _write(tmp_path / "b", report, nr, capsys)
    assert code2 == 0 and sha2 == sha1 and frozen2.read_bytes() == text

    refused = json.loads(json.dumps(report))
    refused["ok"] = False
    refused["failed"] = ["leakage"]
    refused["checks"]["leakage"] = {"ok": False}
    code, _sha, frozen3, pin3 = _write(tmp_path / "c", refused, nr, capsys)
    assert code == 1 and not frozen3.exists() and not pin3.exists()

    partial = json.loads(json.dumps(report))
    partial["partial_set"] = True
    code, _sha, frozen4, _pin4 = _write(tmp_path / "d", partial, nr, capsys)
    assert code == 1 and not frozen4.exists()

    short = json.loads(json.dumps(report))
    short["checks"]["validate_training_set"]["freeze_shortfalls"] = ["floors"]
    code, _sha, frozen5, _pin5 = _write(tmp_path / "e", short, nr, capsys)
    assert code == 1 and not frozen5.exists()

    wrong = json.loads(json.dumps(report))
    wrong["no_role"]["sha256"] = "00" * 32
    code, _sha, frozen6, pin6 = _write(tmp_path / "f", wrong, nr, capsys)
    assert code == 1 and not frozen6.exists() and not pin6.exists()


def test_a_missing_apply_path_fails_the_apply_check(tmp_path, monkeypatch):
    missing = tmp_path / "no-such-apply.jsonl"
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch, apply=[missing])
    assert code == 1
    assert report["failed"] == ["apply"]
    assert "apply file: not available" in report["checks"]["apply"]["problems"]
    assert report["checks"]["leakage"]["apply_rows_checked"] is False
    assert_public(report)


def test_a_bad_apply_json_line_fails_the_apply_check(tmp_path, monkeypatch):
    bad = tmp_path / "apply.jsonl"
    bad.write_text("{not json\n", encoding="utf-8", newline="\n")
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch, apply=[bad])
    assert code == 1
    assert report["failed"] == ["apply"]
    assert "apply file: not JSON" in report["checks"]["apply"]["problems"]
    assert report["checks"]["leakage"]["apply_rows_checked"] is False
    assert_public(report)


def test_write_refuses_an_outside_repo_path(tmp_path, capsys):
    nr = b'{"id": "T04-A-01-nr"}\n'
    training = _passing_report(hashlib.sha256(nr).hexdigest())
    training["training_files"][0]["path"] = "<outside-repo>/pilot-t04.jsonl"
    code, _sha, frozen, pin = _write(tmp_path / "train", training, nr, capsys)
    assert code == 1 and not frozen.exists() and not pin.exists()
    apply_files = _passing_report(hashlib.sha256(nr).hexdigest())
    apply_files["apply_files"][0]["path"] = "<outside-repo>/round1.jsonl"
    code, _sha, frozen2, pin2 = _write(tmp_path / "apply", apply_files, nr, capsys)
    assert code == 1 and not frozen2.exists() and not pin2.exists()


DEPOT = (
    "The bramblewick depot unlocks at dawn on odd weekdays only.\n"
    "It stays shuttered on frost-sundays entirely here."
)
SHUTTERED = "It stays shuttered on frost-sundays entirely here."
RIGHT_REPLY = f"VERDICT: unsupported\nDECIDING: {SHUTTERED}\nESCALATE: no"
CANNOT_TELL = "VERDICT: cannot_tell\nDECIDING: NONE\nESCALATE: no"
THINKING = "Bramblewick hours settle from the shuttered frost line alone."


def apply_row(lesson, kind, n, *, verdict="unsupported", deciding=(SHUTTERED,), user=None, reply=None,
              **extra):
    """One synthetic apply item. The wording is not in the curriculum, so spans and leakage stay clear."""
    statement = f"the bramblewick depot case {lesson}{kind}{n} stays open on a frost sunday"
    if user is None:
        user = f"Material:\n{DEPOT}\nStatement: {statement}."
    if reply is None:
        reply = CANNOT_TELL if verdict == "cannot_tell" else RIGHT_REPLY
    row = {"id": f"{lesson}-{kind}-AP-{n:02d}", "split": "apply", "lesson": lesson, "trait": "t",
           "kind": kind, "side": "n/a", "content_kind": "notice", "tier": 2, "material": DEPOT,
           "sources": [], "invented": True, "key_basis": "construction", "tags": [],
           "verdict": verdict, "deciding": list(deciding), "escalate": "no",
           "turns": [{"user": user, "thinking": THINKING, "reply": reply}]}
    if kind == "C":
        row["planted_flaw"] = "rubber_stamp"
    row.update(extra)
    return row


def test_expected_twin_holds_are_reported_and_do_not_fail(tmp_path, monkeypatch):
    """One item of each hold the runtime is expected to make. The apply check passes."""
    rows = [
        apply_row("T03", "D", 1, verdict="cannot_tell", deciding=()),
        apply_row("T01", "D", 1, silent_without_deciding=False),
        apply_row("T09", "C", 1, order_free=False),
    ]
    path = tmp_path / "apply.jsonl"
    write_rows(path, rows)
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch,
                         sealed=planted_sealed(tmp_path), apply=[path])
    apply_c = report["checks"]["apply"]
    assert code == 0, apply_c["problems"]
    assert apply_c["ok"] is True
    by_id = {h["id"]: h for h in apply_c["predicted_holds"]}
    # cannot_tell drop_deciding is not a hold: twin_applies is false, so make_twin is not called.
    assert "T03-D-AP-01" not in by_id
    reason, expected = freeze.hold_reason(rows[0])
    assert reason == "drop_deciding: not keyed supported or unsupported" and expected is False
    assert set(by_id) == {"T01-D-AP-01", "T09-C-AP-01"}
    assert by_id["T01-D-AP-01"]["reason"] == "drop_deciding: silent_without_deciding is false"
    assert by_id["T09-C-AP-01"]["reason"] == "permute: order_free is false"
    assert all(h["expected"] is True for h in by_id.values())
    assert apply_c["predicted_holds_by_lesson"] == {"T01": 1, "T09": 1}


def test_an_unexplained_twin_none_fails_apply(tmp_path, monkeypatch):
    """The deciding line is in the material and the reply, and missing from the user turn."""
    user = (
        "Material:\nThe bramblewick depot unlocks at dawn on odd weekdays only.\n"
        "A notice about hours sits below the bramblewick clock.\n"
        "Statement: the bramblewick depot case T01D9 stays open on a frost sunday."
    )
    row = apply_row("T01", "D", 1, silent_without_deciding=True, user=user)
    path = tmp_path / "apply.jsonl"
    write_rows(path, [row])
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch,
                         sealed=planted_sealed(tmp_path), apply=[path])
    apply_c = report["checks"]["apply"]
    assert code == 1
    assert report["failed"] == ["apply"]
    assert apply_c["predicted_holds"] == [{
        "id": "T01-D-AP-01", "lesson": "T01", "expected": False,
        "reason": "drop_deciding: a deciding segment is not on a material line",
    }]
    assert apply_c["predicted_holds_by_lesson"] == {}
    assert any("T01-D-AP-01" in p for p in apply_c["problems"])


def test_a_real_freeze_without_a_token_count_records_the_skipped_rules(tmp_path, monkeypatch):
    """No count means the stuck rule and the set rules did not run. A partial run does not say so."""
    path = tmp_path / "apply.jsonl"
    write_rows(path, [apply_row("T06", "D", 1)])
    phrase = "stuck and set rules not run: no token count"
    code, report = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch, partial=False,
                         count="missing", apply=[path])
    assert code == 1
    assert phrase in report["checks"]["apply"]["problems"]
    assert report["checks"]["apply"]["ok"] is False
    _code, partial = check(tmp_path, [DATA / "pilot-t04.jsonl"], monkeypatch, partial=True,
                           count="missing", apply=[path])
    assert phrase not in partial["checks"]["apply"]["problems"]
