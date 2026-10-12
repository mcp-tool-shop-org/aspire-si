"""The puzzle signature keyed on the sidecar model, and the sidecar's row_sha256 verified as role-set stamps it
(R&D, 2026-10-11)."""

import shutil
from pathlib import Path

import pytest

import aspire_si_gates as G
from aspire_si_gates import gates

FIX = Path(__file__).resolve().parent / "fixtures" / "role-set-stamp"  # role-set tests/fixtures/puzzle-check, as stamped


def test_a_row_stamped_by_role_set_verifies_and_one_edited_character_fails(tmp_path):
    models = G.load_models(FIX / "sample.jsonl")  # both lines' hashes verify; only the puzzle carries a model
    assert sorted(models) == ["T00-B-01"]
    shutil.copy(FIX / "sample.models.jsonl", tmp_path / "sample.models.jsonl")
    data = (FIX / "sample.jsonl").read_bytes()
    i = data.index(b"Ash")
    (tmp_path / "sample.jsonl").write_bytes(data[:i] + b"Asp" + data[i + 3:])  # one character edited
    with pytest.raises(G.DataError, match="row_sha256 doesn't match"):
        G.load_models(tmp_path / "sample.jsonl")


def test_the_stamp_ignores_a_crlf_line_ending():
    line = '{"id": "x"}'
    assert G.row_sha256(line) == G.row_sha256(line + "\r\n") == G.row_sha256(line + "\n")


def test_a_sidecar_line_for_an_item_the_file_lacks_is_refused_and_no_sidecar_means_no_models(tmp_path):
    shutil.copy(FIX / "sample.models.jsonl", tmp_path / "sample.models.jsonl")
    first = (FIX / "sample.jsonl").read_text(encoding="utf-8").splitlines()[0]
    (tmp_path / "sample.jsonl").write_text(first + "\n", encoding="utf-8")
    with pytest.raises(G.DataError, match="doesn't have"):
        G.load_models(tmp_path / "sample.jsonl")
    (tmp_path / "solo.jsonl").write_text(first + "\n", encoding="utf-8")
    assert G.load_models(tmp_path / "solo.jsonl") == {}


def model(n, *rules):
    return {"entities": [f"E{i}" for i in range(n)], "facts": [{"rule": [r, "E0"]} for r in rules]}


def test_the_signature_is_entities_and_the_multiset_of_rules_with_after_folded_into_before():
    assert G.model_signature(model(3, "is", "after")) == G.model_signature(model(3, "before", "is"))
    assert G.model_signature(model(3, "is", "is")) != G.model_signature(model(3, "is"))  # counts matter
    assert G.model_signature(model(3, "is_not")) != G.model_signature(model(3, "is"))  # elimination stays apart
    assert G.model_signature(model(4, "is")) != G.model_signature(model(3, "is"))


def puzzle(iid, text):
    return {"id": iid, "kind": "B", "lesson": "T05", "tags": ["non-verification"],
            "turns": [{"user": text, "thinking": "t.", "reply": "Ann holds the red."}]}


def test_the_model_decides_the_signature_and_the_regex_is_only_the_fallback():
    a = puzzle("T05-B-90", "Three crates, one each: Ann is not first.")
    b = puzzle("T05-B-91", "Four jars stand in a row; the salt is before the sugar.")  # different regex cues
    same = {"T05-B-90": model(3, "is", "after"), "T05-B-91": model(3, "before", "is")}
    with pytest.raises(G.DataError, match=r"share the structure 3 entities, rules before \+ is"):
        gates.check_puzzles([a, b], models=same)
    c = puzzle("T05-B-92", "Three crates, one each: Ann is not first.")  # the regex signature of a
    gates.check_puzzles([a, c], models={"T05-B-90": model(3, "is"), "T05-B-92": model(3, "is", "is_not")})
    with pytest.raises(G.DataError, match="share the structure"):
        gates.check_puzzles([a, c])  # without models, the regex fallback sees them as one shape
