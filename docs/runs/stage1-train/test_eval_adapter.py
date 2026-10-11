"""A smoke adapter never spends a sealed read (R&D, 2026-10-11)."""

import hashlib
import json

import eval_adapter


def setup(tmp_path, monkeypatch, manifest):
    sealed = tmp_path / "sealed.jsonl"
    sealed.write_text('{"id": "sx-1"}\n', encoding="utf-8")
    monkeypatch.setattr(eval_adapter, "SEALED_SHA256", hashlib.sha256(b'{"id": "sx-1"}\n').hexdigest())
    adapter = tmp_path / "round" / "adapter"
    adapter.mkdir(parents=True)
    if manifest is not None:
        (adapter.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return sealed, adapter


def test_a_partial_set_adapter_is_refused_on_the_sealed_set(tmp_path, monkeypatch):
    sealed, adapter = setup(tmp_path, monkeypatch, {"freeze_report": {"partial_set": True, "freeze_refused": "x"}})
    assert "partial-set" in eval_adapter.sealed_read_allowed(sealed, adapter)


def test_an_adapter_without_a_manifest_is_refused_on_the_sealed_set(tmp_path, monkeypatch):
    sealed, adapter = setup(tmp_path, monkeypatch, None)
    assert "no manifest" in eval_adapter.sealed_read_allowed(sealed, adapter)


def test_a_frozen_set_adapter_may_sit_the_sealed_set(tmp_path, monkeypatch):
    sealed, adapter = setup(tmp_path, monkeypatch, {"freeze_report": {"shortfalls": []}})
    assert eval_adapter.sealed_read_allowed(sealed, adapter) is None


def test_any_adapter_may_sit_tasks_other_than_the_sealed_set(tmp_path, monkeypatch):
    _, adapter = setup(tmp_path, monkeypatch, {"freeze_report": {"partial_set": True}})
    dev = tmp_path / "dev.jsonl"
    dev.write_text('{"id": "pl-1"}\n', encoding="utf-8")
    assert eval_adapter.sealed_read_allowed(dev, adapter) is None
