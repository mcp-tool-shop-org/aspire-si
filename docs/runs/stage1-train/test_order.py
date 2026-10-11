"""The training order as a setting (--order), and R&D's check that a variant's text is never trained on."""

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
import train  # noqa: E402

DATA_DIR = HERE.parent / "stage1-training-data"


def seqs_of(lessons_sizes):
    return [{"lesson": ls, "id": f"{ls}-{i}"} for ls, n in lessons_sizes for i in range(n)]


def test_shuffled_draws_exactly_as_the_trainer_always_has():
    seqs = seqs_of([("T01", 7), ("T04", 5), ("T14", 9)])
    random.seed(0)
    got = train.training_order(seqs, "shuffled", None, random)
    random.seed(0)
    legacy = list(range(len(seqs)))
    random.shuffle(legacy)
    assert got == legacy


@pytest.mark.parametrize("mode", train.ORDERS)
def test_every_order_trains_every_sequence_once(mode):
    seqs = seqs_of([("T01", 7), ("T04", 5), ("T14", 9)])
    order = train.training_order(seqs, mode, None, random.Random(0))
    assert sorted(order) == list(range(len(seqs)))


def test_listed_trains_one_lesson_at_a_time_in_the_given_order():
    seqs = seqs_of([("T01", 3), ("T04", 2), ("T14", 4)])
    order = train.training_order(seqs, "listed", ["T14", "T01", "T04"], random.Random(0))
    assert [seqs[i]["lesson"] for i in order] == ["T14"] * 4 + ["T01"] * 3 + ["T04"] * 2
    default = train.training_order(seqs, "listed", None, random.Random(0))
    assert [seqs[i]["lesson"] for i in default] == ["T01"] * 3 + ["T04"] * 2 + ["T14"] * 4  # data order


def test_round_robin_takes_one_from_each_lesson_in_turn():
    seqs = seqs_of([("T01", 3), ("T04", 1), ("T14", 2)])
    order = train.training_order(seqs, "round-robin", ["T04", "T01", "T14"], random.Random(0))
    assert [seqs[i]["lesson"] for i in order] == ["T04", "T01", "T14", "T01", "T14", "T01"]


def test_the_order_is_reproducible_from_the_seed_and_changes_with_it():
    seqs = seqs_of([("T01", 6), ("T04", 6)])
    for mode in train.ORDERS:
        a = train.training_order(seqs, mode, None, random.Random(0))
        assert a == train.training_order(seqs, mode, None, random.Random(0))
        assert a != train.training_order(seqs, mode, None, random.Random(1))


def test_a_lesson_order_must_name_every_lesson_once():
    seqs = seqs_of([("T01", 2), ("T04", 2)])
    for bad in (["T01"], ["T01", "T04", "T09"], ["T01", "T01", "T04"]):
        with pytest.raises(SystemExit):
            train.training_order(seqs, "listed", bad, random.Random(0))
    with pytest.raises(SystemExit):
        train.training_order(seqs, "backwards", None, random.Random(0))


class TextTok:
    """Stands in for the tokenizer: 'token ids' are the text itself, so a sequence can be searched as text."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True):
        return "\n".join(m["content"] for m in msgs)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [text]}


def test_a_variants_text_never_appears_in_a_training_sequence():
    """R&D, 2026-10-11: the first quiz asks the variant (n = 0), so the trainer must never train on it. Sequences
    come from an item's turns only; this checks every kind-E variant in the training data against them."""
    rows = [json.loads(x) for f in sorted(DATA_DIR.glob("*.jsonl")) if not f.name.endswith(".models.jsonl")
            for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    tok = TextTok()
    trained = "\n".join(t for r in rows for s in data.sequences(r, tok) for t in s["prompt_ids"] + s["target_ids"])
    variants = [r["variant"]["user"] for r in rows if r.get("kind") == "E" and r.get("variant")]
    assert len(variants) >= 40
    spent = [v[:60] for v in variants if v.strip() and v.strip() in trained]
    assert not spent, f"variant text found in training sequences: {spent[:3]}"


def test_a_resume_under_another_order_is_refused(tmp_path):
    sys.path.insert(0, str(HERE.parent / "stage1-eval"))
    import student
    try:
        if not (student.snapshot_dir() / "tokenizer.json").exists():
            pytest.skip("the student's tokenizer isn't cached here")
    except BaseException:
        pytest.skip("the student's tokenizer isn't cached here")
    files = [str(DATA_DIR / f) for f in ("pilot-t04.jsonl", "batch2-t01.jsonl")]
    out = tmp_path / "r"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    base = [sys.executable, str(HERE / "tiny_train.py"), "--data", *files, "--out", str(out), "--ckpt-every", "1"]
    subprocess.run(base + ["--max-steps", "1", "--order", "listed", "--lesson-order", "T04,T01"], env=env, check=True,
                   capture_output=True, timeout=600)
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert man["order"] == {"mode": "listed", "seed": 0, "lessons": ["T04", "T01"]}
    p = subprocess.run(base + ["--max-steps", "2", "--order", "listed", "--lesson-order", "T01,T04"], env=env,
                       capture_output=True, text=True, timeout=600)
    assert p.returncode != 0 and "training order differs" in p.stderr
