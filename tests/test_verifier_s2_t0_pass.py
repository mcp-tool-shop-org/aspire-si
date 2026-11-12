"""The S2 ``--t0-pass`` mode: the step-0 greedy T=0, n=1 pass over Set U union C, on the base model or a saved
adapter, batch 1, with its header, per-item rows (fsynced), resume and text-free summary.

Every test runs on the CPU with the dry rehearsal's tiny random model (or a stub). No GPU, no download.
"""

import dataclasses
import gzip
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "verifier-distill"))

import s2_grpo_smoke as s2  # noqa: E402
import s2_v11_config as v11cfg  # noqa: E402

torch = pytest.importorskip("torch")

from tests.test_verifier_s2_train import _gold, _make_synthetic_v11_prepass  # noqa: E402

REAL_CHECK_FROZEN_ENV = s2.check_frozen_env
REAL_ASPIRE_COMMIT = s2.aspire_commit
A100 = "NVIDIA A100-SXM4-40GB, 40960, 580.82.07\n"
TEXT_KEYS = {"text", "thinking", "answer"}


class Fx:
    """The synthetic 60-item gold, its Pass A/B files, and the sets the sealed rule derives from them."""


@pytest.fixture
def fx(tmp_path, monkeypatch):
    f = Fx()
    f.tmp = tmp_path
    f.gold = _gold(tmp_path)
    f.pa, f.pas, f.pb, f.pbs, cfg, f.sets = _make_synthetic_v11_prepass(tmp_path, f.gold)
    monkeypatch.setattr(v11cfg, "S2V11", cfg)
    monkeypatch.setattr(s2, "S2V11", cfg)
    monkeypatch.setattr(s2, "MAX_NEW_TOKENS", 8)  # a non-dry pass on the random model must stay short
    f.cfg = cfg
    f.items = sorted(set(f.sets["gold_unsupported_ids"]) | set(f.sets["canary_c_ids"]))
    return f


def _argv(fx, out, *extra, dry=True, pass_id="t0-a"):
    argv = [
        "--gold", str(fx.gold),
        "--rnd-commit", "e48cc90c",
        "--out", str(out),
        "--s2-v11",
        "--t0-pass",
        "--pass-id", pass_id,
        "--attention-mask",
        "--bf16-restore",
        "--max-new", "8",
        "--pass-a-jsonl", str(fx.pa),
        "--pass-a-sha256", fx.pas,
        "--pass-b-jsonl", str(fx.pb),
        "--pass-b-sha256", fx.pbs,
    ]  # fmt: skip
    if dry:
        argv.append("--dry")
    return [*argv, *extra]


def _ns(argv):
    a = s2.build_parser().parse_args(argv)
    return s2.apply_dry_defaults(a) if a.dry else a


def _run(fx, out, *extra, dry=True, build=s2.dry_bundle, rig=None, **kw):
    return s2.execute(_ns(_argv(fx, out, *extra, dry=dry, **kw)), build=build, rig=rig)


def _smi(text=A100, calls=None):
    def run(args):
        if calls is not None:
            calls.append(list(args))
        return text

    return run


def _rig(smi=None):
    return types.SimpleNamespace(
        preflight=lambda: {"name": "NVIDIA GeForce RTX 5090", "driver": "580.1", "guard": {}},
        smi=smi or _smi(),
    )


def _colab_rig(smi=None):
    def boom():
        raise AssertionError("the rig's preflight must not run on colab")

    return types.SimpleNamespace(preflight=boom, smi=smi or _smi())


def _rows(out):
    return s2.read_save_all(Path(out) / s2.T0_PASS_ROWS)


def _header(out):
    return json.loads((Path(out) / s2.T0_PASS_HEADER).read_text(encoding="utf-8"))


def _summary(out):
    return json.loads((Path(out) / s2.T0_PASS_SUMMARY).read_text(encoding="utf-8"))


def _keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _keys(v)


def _live(fx, out, *extra, **kw):
    """A non-dry pass (the model really generates, on the CPU) on the colab venue with a mocked nvidia-smi."""
    return _run(fx, out, "--venue", "colab", *extra, dry=False, rig=_colab_rig(), **kw)


# -- flag refusals --------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("extra", "code"),
    [
        (["--arm", "mech", "--train-seed", "45"], "T0_PASS_CONFLICTING_FLAGS"),
        (["--train-seed", "45"], "T0_PASS_CONFLICTING_FLAGS"),
        (["--rollout-only"], "T0_PASS_CONFLICTING_FLAGS"),
        (["--logging-pass"], "T0_PASS_CONFLICTING_FLAGS"),
        (["--postpass"], "T0_PASS_CONFLICTING_FLAGS"),
        (["--probe-memory"], "T0_PASS_CONFLICTING_FLAGS"),
        (["--prompt-lengths"], "T0_PASS_CONFLICTING_FLAGS"),
        (["--resume-from", "somewhere"], "T0_PASS_CONFLICTING_FLAGS"),
        (["--stress-prefix", "1"], "T0_PASS_CONFLICTING_FLAGS"),
        (["--stop-after-new", "2"], "T0_PASS_CONFLICTING_FLAGS"),
        (["--steps", "5"], "T0_PASS_STEPS_FORBIDDEN"),
        (["--draw", "uniform"], "T0_PASS_DRAW_FLAGS_FORBIDDEN"),
        (["--draw-seed", "3"], "T0_PASS_DRAW_FLAGS_FORBIDDEN"),
        (["--k-min", "1"], "T0_PASS_DRAW_FLAGS_FORBIDDEN"),
    ],
)
def test_conflicting_flags_refused(fx, extra, code):
    with pytest.raises(SystemExit, match=code):
        _run(fx, fx.tmp / "o", *extra)
    assert not (fx.tmp / "o").exists()


def test_other_refusals(fx):
    with pytest.raises(SystemExit, match="T0_PASS_REQUIRES_S2_V11"):
        s2.execute(_ns([a for a in _argv(fx, fx.tmp / "o") if a != "--s2-v11"]), build=s2.dry_bundle)
    argv = _argv(fx, fx.tmp / "o")
    with pytest.raises(SystemExit, match="T0_PASS_PATH_HF_ONLY"):
        s2.execute(s2.build_parser().parse_args([*argv, "--path", "llamacpp"]), build=s2.dry_bundle)
    for bad in ("Has Caps", "a" * 65, "", "x/y"):
        with pytest.raises(SystemExit, match="T0_PASS_ID_INVALID"):
            _run(fx, fx.tmp / "o", pass_id=bad)
    no_id = [a for a in argv if a not in ("--pass-id", "t0-a")]
    with pytest.raises(SystemExit, match="T0_PASS_ID_INVALID"):
        s2.execute(_ns(no_id), build=s2.dry_bundle)
    for flag in ("--attention-mask", "--bf16-restore"):
        with pytest.raises(SystemExit, match="T0_PASS_TRAINER_SETTING_MISSING"):
            s2.execute(_ns([a for a in argv if a != flag]), build=s2.dry_bundle)
    with pytest.raises(SystemExit, match="T0_PASS_FLAG_WITHOUT_MODE"):
        s2.execute(_ns([a for a in argv if a != "--t0-pass"] + ["--venue", "colab"]), build=s2.dry_bundle)
    with pytest.raises(SystemExit, match="T0_PASS_FLAG_WITHOUT_MODE"):
        s2.execute(
            _ns([a for a in argv if a != "--t0-pass"] + ["--code-commit", "a" * 40]), build=s2.dry_bundle
        )
    assert s2.t0_pass_requested(types.SimpleNamespace()) is False
    with pytest.raises(SystemExit, match="T0_PASS_VENUE_INVALID"):
        s2.validate_t0_pass_args(types.SimpleNamespace(**{**vars(_ns(argv)), "venue": "mars"}))
    assert not (fx.tmp / "o").exists()


def test_v11_freezes_still_apply_in_the_mode(fx):
    with pytest.raises(SystemExit, match="S2_V11_OVERRIDE_FORBIDDEN"):
        _run(fx, fx.tmp / "o", "--no-save-all", "--save-all-off-reason", "x")
    with pytest.raises(SystemExit, match="S2_V11_OVERRIDE_FORBIDDEN"):
        _run(fx, fx.tmp / "o", "--quote-reward", "v1")
    with pytest.raises(SystemExit, match="S2_V11_PASS_A_SHA256_MISMATCH"):
        s2.execute(_ns(_swap(_argv(fx, fx.tmp / "o"), "--pass-a-sha256", "0" * 64)), build=s2.dry_bundle)


def _swap(argv, flag, value):
    out = list(argv)
    out[out.index(flag) + 1] = value
    return out


def test_non_dry_max_new_must_be_the_trainers(fx, monkeypatch):
    monkeypatch.setattr(s2, "MAX_NEW_TOKENS", 2048)
    with pytest.raises(SystemExit, match="T0_PASS_MAX_NEW_MISMATCH"):
        _run(fx, fx.tmp / "o", dry=False, rig=_rig())


def test_does_not_change_the_namespace_the_trainer_sees(fx):
    a = _ns(_argv(fx, fx.tmp / "o"))
    before = dict(vars(a))
    s2.validate_t0_pass_args(a)
    assert {k: v for k, v in vars(a).items() if k != "quote_reward"} == {
        k: v for k, v in before.items() if k != "quote_reward"
    }
    assert a.quote_reward == "v2"


# -- the item set ---------------------------------------------------------------------------------------------
def test_item_set_is_u_union_c_and_digests_are_checked(fx, capsys):
    assert _run(fx, fx.tmp / "o") == 0
    planned = json.loads(capsys.readouterr().out.splitlines()[0])
    assert planned["event"] == "t0_pass_planned"
    assert planned["item_count"] == len(fx.items)
    assert planned["c_count"] == fx.sets["canary_c_count"]
    assert planned["u_only_count"] == len(fx.items) - fx.sets["canary_c_count"]
    assert [Path(p).name for p in planned["outputs"]] == [
        s2.T0_PASS_ROWS,
        s2.T0_PASS_HEADER,
        s2.T0_PASS_SUMMARY,
    ]
    rows = _rows(fx.tmp / "o")
    assert [r["item_id"] for r in rows] == fx.items
    c = set(fx.sets["canary_c_ids"])
    assert {r["item_id"] for r in rows if r["set"] == "C"} == c
    assert all(r["set"] == "U-only" for r in rows if r["item_id"] not in c)
    hdr = _header(fx.tmp / "o")
    assert hdr["canary_c_ids_sha256"] == fx.sets["canary_c_ids_sha256"]
    assert hdr["gold_unsupported_ids_sha256"] == fx.sets["gold_unsupported_ids_sha256"]
    assert hdr["item_count"] == len(fx.items)


def test_digest_mismatch_refuses(fx, monkeypatch):
    bad_c = dataclasses.replace(fx.cfg, canary_c_ids_sha256="0" * 64)
    monkeypatch.setattr(s2, "S2V11", bad_c)
    with pytest.raises(SystemExit, match=r"T0_PASS_SET_DIGEST_MISMATCH: S2_V11_CANARY_C_SHA256_MISMATCH"):
        _run(fx, fx.tmp / "o")
    bad_u = dataclasses.replace(fx.cfg, gold_unsupported_ids_sha256="0" * 64)
    monkeypatch.setattr(s2, "S2V11", bad_u)
    with pytest.raises(
        SystemExit, match="T0_PASS_SET_DIGEST_MISMATCH: S2_V11_GOLD_UNSUPPORTED_SHA256_MISMATCH"
    ):
        _run(fx, fx.tmp / "o")
    monkeypatch.setattr(s2, "S2V11", dataclasses.replace(fx.cfg, gold_unsupported_train_items=999))
    with pytest.raises(
        SystemExit, match="T0_PASS_SET_DIGEST_MISMATCH: S2_V11_GOLD_UNSUPPORTED_SIZE_MISMATCH"
    ):
        _run(fx, fx.tmp / "o")
    assert not (fx.tmp / "o").exists()


def test_a_pass_file_missing_an_item_refuses_before_anything_is_written(fx, monkeypatch):
    with pytest.raises(SystemExit, match="prepass_missing_split_id"):
        _drop_pass_a_row(fx, monkeypatch)
    assert not (fx.tmp / "o").exists()


def _drop_pass_a_row(fx, monkeypatch):
    rows = [json.loads(x) for x in fx.pa.read_text(encoding="utf-8").splitlines()]
    short = fx.tmp / "short_a.jsonl"
    short.write_text("".join(json.dumps(r) + "\n" for r in rows[1:]), encoding="utf-8", newline="\n")
    sha = hashlib.sha256(short.read_bytes()).hexdigest()
    monkeypatch.setattr(s2, "S2V11", dataclasses.replace(fx.cfg, pass_a_sha256=sha))
    argv = _swap(_swap(_argv(fx, fx.tmp / "o"), "--pass-a-jsonl", str(short)), "--pass-a-sha256", sha)
    return s2.execute(_ns(argv), build=s2.dry_bundle)


def test_derived_ids_are_rechecked_even_if_the_builder_is_lenient(fx, monkeypatch):
    real = s2.build_v11_canary_sets
    tampered = {}

    def lenient(*args, **kw):
        out = dict(real(*args, **{**kw, "strict": False}))
        out.update(tampered)
        return out

    monkeypatch.setattr(s2, "build_v11_canary_sets", lenient)
    tampered["canary_c_ids"] = list(fx.sets["canary_c_ids"])[:-1]
    with pytest.raises(SystemExit, match="T0_PASS_SET_DIGEST_MISMATCH"):
        _run(fx, fx.tmp / "o")
    tampered.clear()
    tampered["gold_unsupported_ids"] = list(fx.sets["gold_unsupported_ids"])
    monkeypatch.setattr(
        s2,
        "ids_sha256_v11",
        lambda ids: (
            fx.sets["gold_unsupported_ids_sha256"]
            if len(ids) == len(fx.sets["gold_unsupported_ids"])
            else fx.sets["canary_c_ids_sha256"]
        ),
    )
    tampered["canary_c_ids"] = ["not-in-u", *fx.sets["canary_c_ids"][: fx.sets["canary_c_count"] - 1]]
    with pytest.raises(SystemExit, match="T0_PASS_SET_NOT_NESTED"):
        _run(fx, fx.tmp / "o")


# -- batch-1 rows equal the trainer's own pass ----------------------------------------------------------------
def _direct(fx, **ns_extra):
    a = _ns(_argv(fx, fx.tmp / "direct", dry=False))
    for k, v in ns_extra.items():
        setattr(a, k, v)
    bundle = s2.dry_bundle(a)
    rows = {str(r["id"]): r for r in s2.read_gold(fx.gold)[0]}
    return s2.run_v11_t0_pass(
        model=bundle["model"],
        tok=bundle["tok"],
        rows_by_id=rows,
        item_ids=fx.items,
        max_new=8,
        render_fn=bundle.get("render"),
        pass_attention_mask=True,
        quote_reward="v2",
        dry=False,
        device_sync=bundle["sync"],
        mode=bundle["mode"],
        state=bundle["state"],
    )


PASS_FIELDS = (
    "gold_verdict", "verdict", "correct", "truncated_or_malformed", "temperature", "n_samples", "mode",
    "eval_mode", "grad_ckpt",
)  # fmt: skip


def test_batch_one_rows_equal_a_direct_run_v11_t0_pass_call(fx):
    assert _live(fx, fx.tmp / "o") == 0
    direct = _direct(fx)
    rows = _rows(fx.tmp / "o")
    assert [r["item_id"] for r in rows] == fx.items == list(direct)
    for r in rows:
        want = direct[r["item_id"]]
        assert {k: r[k] for k in PASS_FIELDS} == {k: want[k] for k in PASS_FIELDS}
        assert r["kind"] == "t0_pass" and r["pass_id"] == "t0-a" and r["checkpoint"] == "base"
        assert r["set"] in ("C", "U-only")
        assert isinstance(r["text"], str) and isinstance(r["thinking"], str) and isinstance(r["answer"], str)
        assert 0 < r["completion_tokens"] <= 8
        assert r["parse_status"] in ("ok", "malformed", "truncated")
        if r["truncated"]:
            assert r["parse_status"] == "truncated" and r["truncated_or_malformed"] is True
        if r["parse_status"] == "ok":
            assert r["verdict"] in s2.VERDICTS and not r["truncated"]
    assert any(r["truncated"] for r in rows)  # the random model never closes its answer at 8 tokens


def test_dry_rows_equal_a_direct_dry_call_and_carry_no_text(fx):
    assert _run(fx, fx.tmp / "o") == 0
    a = _ns(_argv(fx, fx.tmp / "d"))
    bundle = s2.dry_bundle(a)
    rows_by_id = {str(r["id"]): r for r in s2.read_gold(fx.gold)[0]}
    direct = s2.run_v11_t0_pass(
        model=bundle["model"], tok=bundle["tok"], rows_by_id=rows_by_id, item_ids=fx.items,
        max_new=8, pass_attention_mask=True, quote_reward="v2", dry=True, mode=bundle["mode"], state=bundle["state"],
    )  # fmt: skip
    rows = _rows(fx.tmp / "o")
    for r in rows:
        assert {k: r[k] for k in PASS_FIELDS} == {k: direct[r["item_id"]][k] for k in PASS_FIELDS}
        assert r["parse_status"] == "dry" and r["text"] is None and r["completion_tokens"] is None


def test_the_trainers_own_step0_pass_is_the_same_call(fx):
    """The trainer calls run_v11_t0_pass on all the ids at once; one id at a time gives the same results."""
    assert _live(fx, fx.tmp / "o") == 0
    once = _direct(fx)
    for r in _rows(fx.tmp / "o"):
        assert r["verdict"] == once[r["item_id"]]["verdict"]


# -- resume, kill, torn tail ----------------------------------------------------------------------------------
def test_resume_after_a_kill_and_a_torn_write(fx, capsys, monkeypatch):
    assert _live(fx, fx.tmp / "full") == 0
    full = _rows(fx.tmp / "full")
    real = s2.run_v11_t0_pass
    n = {"calls": 0}

    def killer(**kw):
        if n["calls"] == 3:
            raise RuntimeError("simulated kill")
        n["calls"] += 1
        return real(**kw)

    monkeypatch.setattr(s2, "run_v11_t0_pass", killer)
    with pytest.raises(RuntimeError, match="simulated kill"):
        _live(fx, fx.tmp / "o")
    assert len(_rows(fx.tmp / "o")) == 3
    assert not (fx.tmp / "o" / s2.T0_PASS_SUMMARY).exists()
    rows_path = fx.tmp / "o" / s2.T0_PASS_ROWS
    member = gzip.compress(b'{"item_id": "half-written"}\n', mtime=0)
    with open(rows_path, "ab") as fh:
        fh.write(member[:11])  # the write the kill cut
    n["calls"] = -(10**6)
    calls = []

    def counting(**kw):
        calls.append(kw["item_ids"])
        return real(**kw)

    monkeypatch.setattr(s2, "run_v11_t0_pass", counting)
    capsys.readouterr()
    assert _live(fx, fx.tmp / "o") == 0
    out = capsys.readouterr().out
    assert '"event": "t0_pass_resume"' in out and '"done": 3' in out and '"torn_bytes_dropped": 11' in out
    assert calls == [[i] for i in fx.items[3:]]  # only the items not done, one at a time
    assert _rows(fx.tmp / "o") == full
    assert _summary(fx.tmp / "o")["complete"] is True
    assert _live(fx, fx.tmp / "o") == 0  # a complete pass resumes to a no-op
    assert calls == [[i] for i in fx.items[3:]]
    assert _rows(fx.tmp / "o") == full


def test_each_row_is_fsynced_as_it_lands(fx, monkeypatch):
    synced = []
    real_fsync = s2.os.fsync
    monkeypatch.setattr(s2.os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    assert _live(fx, fx.tmp / "o") == 0
    assert len(synced) >= len(fx.items)  # at least one per row (the header and summary add more)


def test_rows_are_one_gzip_member_each(fx):
    assert _run(fx, fx.tmp / "o") == 0
    raw = (fx.tmp / "o" / s2.T0_PASS_ROWS).read_bytes()
    assert raw.count(b"\x1f\x8b\x08") >= len(fx.items)
    rows, good, total = s2._t0_read_rows(fx.tmp / "o" / s2.T0_PASS_ROWS)
    assert len(rows) == len(fx.items) and good == total == len(raw)


# -- header: pins, mismatch, tamper ---------------------------------------------------------------------------
def test_header_holds_every_pin(fx):
    assert _live(fx, fx.tmp / "o") == 0
    h = _header(fx.tmp / "o")
    assert h["schema"] == "s2-t0pass-header-v1" and h["pass_id"] == "t0-a" and h["checkpoint"] == "base"
    assert h["adapter_files_sha256"] is None and h["adapter_sha256"] is None
    assert h["script_sha256"] == s2.script_sha256()
    assert h["code_commit"] == s2.aspire_commit()
    assert h["student_revision"] == s2.STUDENT_REVISION and h["student"] == s2.STUDENT
    assert h["pass_a_sha256"] == fx.pas and h["pass_b_sha256"] == fx.pbs
    assert h["venue"] == "colab" and h["gpu"] == {
        "name": "NVIDIA A100-SXM4-40GB",
        "memory_total_mib": 40960,
        "driver": "580.82.07",
    }
    assert h["max_new"] == 8 and h["quote_reward"] == "v2" and h["attention_mask"] and h["bf16_restore"]
    assert h["torch"] == s2.package_version("torch") and h["transformers"] == s2.package_version(
        "transformers"
    )
    assert set(h["frozen_stack"]) == set(s2.FROZEN_STACK_VERSIONS)
    assert h["header_sha256"] == s2._t0_header_sha256(h) and h["created_utc"]
    assert set(s2.read_gold(fx.gold)[1]) == set(h["gold_sha256s"])


def test_mismatched_header_refuses(fx, monkeypatch):
    assert _run(fx, fx.tmp / "o") == 0
    rows_before = (fx.tmp / "o" / s2.T0_PASS_ROWS).read_bytes()
    with pytest.raises(SystemExit, match=r"T0_PASS_HEADER_MISMATCH.*pass_id"):
        _run(fx, fx.tmp / "o", pass_id="t0-b")
    monkeypatch.setattr(s2, "aspire_commit", lambda cwd=None: "f" * 40)
    with pytest.raises(SystemExit, match=r"T0_PASS_HEADER_MISMATCH.*code_commit"):
        _run(fx, fx.tmp / "o")
    monkeypatch.setattr(s2, "aspire_commit", REAL_ASPIRE_COMMIT)
    with pytest.raises(SystemExit, match=r"T0_PASS_HEADER_MISMATCH.*checkpoint"):
        _run(fx, fx.tmp / "o", "--checkpoint", str(_adapter(fx)))
    assert (fx.tmp / "o" / s2.T0_PASS_ROWS).read_bytes() == rows_before


def test_a_changed_resolved_revision_refuses_after_the_build(fx):
    assert _run(fx, fx.tmp / "o") == 0

    def other_revision(a):
        return {**s2.dry_bundle(a), "revision": "c0ffee"}

    with pytest.raises(SystemExit, match=r"T0_PASS_HEADER_MISMATCH.*student_revision_resolved"):
        _run(fx, fx.tmp / "o", build=other_revision)


def test_tampered_missing_and_foreign_files_refuse(fx):
    out = fx.tmp / "o"
    assert _run(fx, out) == 0
    hp, rp = out / s2.T0_PASS_HEADER, out / s2.T0_PASS_ROWS
    good_header, good_rows = hp.read_text(encoding="utf-8"), rp.read_bytes()

    hdr = json.loads(good_header)
    hdr["max_new"] = 99  # edited without re-hashing
    hp.write_text(json.dumps(hdr), encoding="utf-8")
    with pytest.raises(SystemExit, match="T0_PASS_HEADER_CORRUPT"):
        _run(fx, out)
    hp.write_text("not json", encoding="utf-8")
    with pytest.raises(SystemExit, match="T0_PASS_HEADER_CORRUPT"):
        _run(fx, out)
    hp.write_text(good_header, encoding="utf-8")

    hp.unlink()
    with pytest.raises(SystemExit, match="T0_PASS_HEADER_MISSING"):
        _run(fx, out)
    hp.write_text(good_header, encoding="utf-8")

    rows = s2.read_save_all(rp)
    rows[0]["header_sha256"] = "0" * 64
    rp.write_bytes(b"".join(gzip.compress((json.dumps(r) + "\n").encode(), mtime=0) for r in rows))
    with pytest.raises(SystemExit, match="T0_PASS_OUT_FOREIGN_ROW"):
        _run(fx, out)
    rows[0]["header_sha256"] = json.loads(good_header)["header_sha256"]
    rp.write_bytes(
        b"".join(gzip.compress((json.dumps(r) + "\n").encode(), mtime=0) for r in [rows[0], rows[0]])
    )
    with pytest.raises(SystemExit, match="T0_PASS_OUT_FOREIGN_ROW"):  # a duplicate item
        _run(fx, out)

    rp.write_bytes(b"this is not gzip at all")
    with pytest.raises(SystemExit, match="T0_PASS_OUT_CORRUPT"):
        _run(fx, out)
    rp.write_bytes(gzip.compress(b"{not json\n", mtime=0))
    with pytest.raises(SystemExit, match="T0_PASS_OUT_CORRUPT"):
        _run(fx, out)
    rp.write_bytes(good_rows)
    assert _run(fx, out) == 0  # the untouched files resume cleanly


def test_a_header_without_rows_resumes(fx, monkeypatch):
    real = s2.run_v11_t0_pass
    monkeypatch.setattr(s2, "run_v11_t0_pass", lambda **kw: (_ for _ in ()).throw(RuntimeError("kill")))
    with pytest.raises(RuntimeError):
        _run(fx, fx.tmp / "o")
    assert (fx.tmp / "o" / s2.T0_PASS_HEADER).exists() and not (fx.tmp / "o" / s2.T0_PASS_ROWS).exists()
    monkeypatch.setattr(s2, "run_v11_t0_pass", real)
    assert _run(fx, fx.tmp / "o") == 0
    assert len(_rows(fx.tmp / "o")) == len(fx.items)


# -- summary: counts only -------------------------------------------------------------------------------------
def test_summary_is_counts_only_and_matches_the_rows(fx):
    assert _live(fx, fx.tmp / "o") == 0
    summary, rows = _summary(fx.tmp / "o"), _rows(fx.tmp / "o")
    assert not (set(_keys(summary)) & TEXT_KEYS)
    assert summary["complete"] is True and summary["planned_items"] == summary["done_items"] == len(fx.items)
    assert summary["all"]["items"] == len(rows)
    assert summary["sets"]["C"]["items"] == fx.sets["canary_c_count"]
    assert summary["sets"]["C"]["items"] + summary["sets"]["U-only"]["items"] == len(rows)
    for name, group in (
        ("C", [r for r in rows if r["set"] == "C"]),
        ("U-only", [r for r in rows if r["set"] == "U-only"]),
    ):
        s = summary["sets"][name]
        assert sum(s["verdict_counts"].values()) == len(group) == sum(s["parse_status_counts"].values())
        assert s["truncated"] == sum(1 for r in group if r["truncated"])
        assert s["correct"] == sum(1 for r in group if r["correct"])
    assert set(summary) == {
        "schema",
        "kind",
        "pass_id",
        "header_sha256",
        "planned_items",
        "done_items",
        "complete",
        "sets",
        "all",
    }
    assert all(r["text"] not in json.dumps(summary) for r in rows if r["text"])


def test_summary_of_a_partial_set_says_so(fx):
    hdr = {"pass_id": "p", "header_sha256": "h"}
    s = s2.t0_pass_summary(
        [{"set": "C", "verdict": "supported", "parse_status": "ok", "truncated": False, "correct": True}],
        header=hdr,
        planned=5,
    )
    assert s["complete"] is False and s["done_items"] == 1 and s["sets"]["U-only"]["items"] == 0


# -- venue ----------------------------------------------------------------------------------------------------
def test_colab_skips_the_rigs_preflight_and_records_nvidia_smi(fx):
    calls: list = []
    rig = _colab_rig(_smi(A100, calls))
    assert _run(fx, fx.tmp / "o", "--venue", "colab", dry=False, rig=rig) == 0
    assert calls == [["--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]]
    assert _header(fx.tmp / "o")["gpu"]["name"] == "NVIDIA A100-SXM4-40GB"
    # an injected runner (the `smi` argument) is used before the rig's
    other: list = []
    a = _ns(_argv(fx, fx.tmp / "o2", "--venue", "colab", dry=False))
    assert s2.execute_t0_pass(a, build=s2.dry_bundle, smi=_smi("\nNVIDIA L4, 23034, 570.0\n", other)) == 0
    assert len(other) == 1 and _header(fx.tmp / "o2")["gpu"] == {
        "name": "NVIDIA L4",
        "memory_total_mib": 23034,
        "driver": "570.0",
    }


def test_colab_refuses_without_a_gpu_in_non_dry_mode(fx):
    def run(rig):
        return _run(fx, fx.tmp / "o", "--venue", "colab", dry=False, rig=rig)

    with pytest.raises(SystemExit, match="T0_PASS_NO_GPU"):
        run(_colab_rig(_smi("")))

    def missing(args):
        raise FileNotFoundError("nvidia-smi")

    with pytest.raises(SystemExit, match="nvidia_smi_missing"):
        run(_colab_rig(missing))
    with pytest.raises(SystemExit, match="nvidia_smi_garbage"):
        run(_colab_rig(_smi("NVIDIA A100, lots, 580\n")))
    with pytest.raises(SystemExit, match="nvidia_smi_garbage"):
        run(_colab_rig(_smi("NVIDIA A100, 40960\n")))
    with pytest.raises(SystemExit, match="2 GPUs visible"):
        run(_colab_rig(_smi(A100 + A100)))
    with pytest.raises(SystemExit, match="not an NVIDIA GPU"):
        run(_colab_rig(_smi("Tesla T4, 15360, 550\n".replace("Tesla", "AMD"))))
    assert not (fx.tmp / "o").exists()


def test_colab_dry_asks_nothing_of_the_machine(fx):
    def boom(args):
        raise AssertionError("dry must not call nvidia-smi")

    assert _run(fx, fx.tmp / "o", "--venue", "colab", rig=_colab_rig(boom)) == 0
    h = _header(fx.tmp / "o")
    assert h["venue"] == "colab" and h["gpu"] is None and h["dry"] is True


def test_rig_venue_keeps_the_preflight(fx):
    seen = []
    rig = _rig()
    rig.preflight = lambda: seen.append(1) or {"name": "NVIDIA GeForce RTX 5090", "driver": "580.1"}
    assert _run(fx, fx.tmp / "o", dry=False, rig=rig) == 0
    assert seen == [1]
    h = _header(fx.tmp / "o")
    assert h["venue"] == "rig" and h["gpu"] == {"name": "NVIDIA GeForce RTX 5090", "driver": "580.1"}
    with pytest.raises(SystemExit, match="T0_PASS_RIG_REQUIRED"):
        _run(fx, fx.tmp / "o2", dry=False, rig=None)

    # a failed preflight (no watchdog) stops a rig pass before anything is written
    def refuse():
        raise SystemExit("refusing to start: no watchdog heartbeat")

    with pytest.raises(SystemExit, match="no watchdog heartbeat"):
        _run(fx, fx.tmp / "o3", dry=False, rig=types.SimpleNamespace(preflight=refuse))
    assert not (fx.tmp / "o3").exists()
    assert _run(fx, fx.tmp / "o4") == 0  # dry on the rig venue uses the canned guard, no rig


def test_require_frozen_stack_works_on_both_venues(fx, monkeypatch):
    seen = []
    monkeypatch.setattr(
        s2, "check_frozen_env",
        lambda path=None, **kw: seen.append(kw) or {"stack_versions": {"torch": "x"}, "pip_freeze_sha256": "ab" * 32},
    )  # fmt: skip
    for venue, rig in (("rig", _rig()), ("colab", _colab_rig())):
        out = fx.tmp / venue
        assert _run(fx, out, "--venue", venue, "--require-frozen-stack", dry=False, rig=rig) == 0
        assert seen[-1] == {"require_exact_versions": True}
        h = _header(out)
        assert h["require_frozen_stack"] is True and h["frozen_pip_freeze_sha256"] == "ab" * 32
    monkeypatch.setattr(s2, "check_frozen_env", REAL_CHECK_FROZEN_ENV)
    for venue in ("rig", "colab"):  # the real check: this CPU stack is not the frozen one
        with pytest.raises(SystemExit, match="frozen_env_version_mismatch"):
            _run(fx, fx.tmp / f"x-{venue}", "--venue", venue, "--require-frozen-stack")


# -- code commit ----------------------------------------------------------------------------------------------
def test_code_commit_arg_and_refusals(fx, monkeypatch):
    here = s2.aspire_commit()
    assert _run(fx, fx.tmp / "o", "--code-commit", here) == 0
    assert _header(fx.tmp / "o")["code_commit"] == here
    with pytest.raises(SystemExit, match="T0_PASS_CODE_COMMIT_INVALID"):
        _run(fx, fx.tmp / "o2", "--code-commit", "abc")
    with pytest.raises(SystemExit, match="T0_PASS_CODE_COMMIT_MISMATCH"):
        _run(fx, fx.tmp / "o2", "--code-commit", "0" * 40)
    monkeypatch.setattr(s2, "aspire_commit", lambda cwd=None: None)
    with pytest.raises(SystemExit, match="T0_PASS_CODE_COMMIT_UNKNOWN"):
        _live(fx, fx.tmp / "o2")
    assert _run(fx, fx.tmp / "o3") == 0  # a dry rehearsal may have no commit
    assert _header(fx.tmp / "o3")["code_commit"] is None
    assert _run(fx, fx.tmp / "o4", "--code-commit", "1" * 40) == 0  # no checkout: the arg stands alone
    assert _header(fx.tmp / "o4")["code_commit"] == "1" * 40


# -- adapter --------------------------------------------------------------------------------------------------
def _adapter(fx, name="adapter"):
    d = fx.tmp / name
    s2.dry_bundle(_ns(_argv(fx, fx.tmp / "scratch")))["model"].save_pretrained(d)
    return d


def test_adapter_pass_records_file_shas_and_loads_frozen(fx):
    ad = _adapter(fx)
    seen = {}

    def spy(a):
        seen.update(postpass=a.postpass, checkpoint=a.checkpoint)
        b = s2.dry_bundle(a)
        seen["trainable"] = sum(p.numel() for p in b["model"].parameters() if p.requires_grad)
        return b

    a = _ns(_argv(fx, fx.tmp / "o", "--checkpoint", str(ad)))
    assert s2.execute(a, build=spy) == 0
    assert seen["postpass"] is True and seen["trainable"] == 0  # loaded for a pass: frozen
    assert a.postpass is False  # the caller's namespace is untouched
    h = _header(fx.tmp / "o")
    assert h["checkpoint"] == str(ad)
    assert h["adapter_files_sha256"] == s2.checkpoint_file_shas(ad)
    assert h["adapter_sha256"] == s2.checkpoint_sha256(ad)
    assert all(r["checkpoint"] == str(ad) for r in _rows(fx.tmp / "o"))
    base = _adapter(fx, "adapter2")
    assert s2.checkpoint_file_shas(base) == s2.checkpoint_file_shas(ad)  # B = 0 saves are reproducible
    assert all(len(v) == 64 for v in h["adapter_files_sha256"].values())


def test_adapter_that_is_not_loaded_or_missing_refuses(fx):
    ad = _adapter(fx)

    def forgetful(a):
        return {**s2.dry_bundle(a), "adapter_loaded": False}

    with pytest.raises(SystemExit, match="checkpoint_adapter_not_loaded"):
        _run(fx, fx.tmp / "o", "--checkpoint", str(ad), build=forgetful)
    with pytest.raises(SystemExit, match="checkpoint_missing"):
        _run(fx, fx.tmp / "o2", "--checkpoint", str(fx.tmp / "nope"))
    assert not (fx.tmp / "o2").exists()


def test_a_trained_adapter_changes_the_pass_and_the_base_pass_is_a_fresh_zero_adapter(fx):
    """Base mode is a fresh LoRA with B = 0, as at the trainer's step 0: its output is the base model's."""
    assert _live(fx, fx.tmp / "base") == 0
    ad = _adapter(fx)
    from safetensors.torch import load_file, save_file

    f = ad / "adapter_model.safetensors"
    t = load_file(f)
    for k in t:
        if "lora_B" in k:
            t[k] = torch.full_like(t[k], 0.5)
    save_file(t, f)
    out = fx.tmp / "trained"
    assert _live(fx, out, "--checkpoint", str(ad)) == 0
    base_rows, trained_rows = _rows(fx.tmp / "base"), _rows(out)
    assert [r["item_id"] for r in base_rows] == [r["item_id"] for r in trained_rows]
    assert [r["text"] for r in base_rows] != [r["text"] for r in trained_rows]
    assert _header(out)["adapter_sha256"] != _header(fx.tmp / "base")["adapter_sha256"]


# -- the trainer's code is not touched ------------------------------------------------------------------------
def test_the_dispatch_adds_nothing_to_a_training_run(fx):
    argv = [a for a in _argv(fx, fx.tmp / "o") if a not in ("--t0-pass", "--pass-id", "t0-a")]
    a = _ns(argv)
    assert s2.t0_pass_requested(a) is False
    assert a.t0_pass is False and a.venue is None and a.code_commit is None
