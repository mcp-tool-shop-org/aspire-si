"""The Skeptic control (Auditor plan, addendum 2): paraphrase pairs, the balanced permutation null,
and the committed reading, on synthetic data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXPERIMENT = Path(__file__).resolve().parents[1] / "examples" / "sft-experiment"
sys.path.insert(0, str(EXPERIMENT))

import critic_heads as ch  # noqa: E402
import critic_heads_run as run  # noqa: E402
import lib  # noqa: E402
import skeptic_pairs as sk  # noqa: E402

ANSWER = (
    "Water boils at 100 C at sea level, where the air pressure is about one atmosphere. "
    "It freezes at 0 C under the same conditions. Ice floats on water because it is less dense. "
    "These facts follow from the hydrogen bonds between water molecules."
)


class TestBalancedNull:
    def test_each_prompt_is_flipped_exactly_half(self):
        ids = [0, 0, 1, 1, 1, 1, 2, 2, 2]
        for k in range(5):
            flips = ch.balanced_flips(ids, k)
            for g in set(ids):
                members = [f for f, i in zip(flips, ids) if i == g]
                assert sum(members) == len(members) // 2

    def test_the_odd_pair_left_out_rotates(self):
        ids = [0, 0, 0]
        unflipped = set()
        for k in range(6):
            flips = ch.balanced_flips(ids, k)
            assert sum(flips) == 1
        assert len({tuple(ch.balanced_flips(ids, k)) for k in range(6)}) > 1
        del unflipped

    def test_permutation_p(self):
        assert ch.permutation_p(0.9, [0.5] * 20) == pytest.approx(1 / 21)
        assert ch.permutation_p(0.5, [0.5] * 20) == 1.0


class TestSkepticClass:
    def test_rows_are_mutually_exclusive(self):
        assert ch.skeptic_class([0.44, 0.66], [0.1, 0.3]) == "error-specific"
        assert ch.skeptic_class([0.6, 0.8], [-0.05, 0.1]) == "an edit detector"
        assert ch.skeptic_class([0.6, 0.8], [-0.3, -0.1]) == "an edit detector"
        assert ch.skeptic_class([0.6, 0.8], [0.05, 0.3]) == "partly an edit detector"
        assert ch.skeptic_class([0.2, 0.4], [0.3, 0.5]) == "prefers the edited copy"

    def test_the_skeptic_role_reads_like_the_auditor(self):
        assert ch.role_scores("skeptic", [1.0], [2.0]) == ([2.0], [1.0])


def head(role, pooling, seed, sets, control="none"):
    return {"role": role, "pooling": pooling, "seed": seed, "control": control, "sets": sets}


def scores(n, good, ids, role="auditor"):
    strong, flawed = [], []
    for i in range(n):
        win = int((i + 1) * good) > int(i * good)
        hi, lo = (5.5 + i / 1000, 4.5) if win else (4.5, 5.5 + i / 1000)
        s, f = (lo, hi) if role in ("auditor", "skeptic") else (hi, lo)
        strong.append(s)
        flawed.append(f)
    return {"prompt_ids": ids, "strong": strong, "flawed": flawed}


class TestSkepticReadout:
    ERR_IDS = [i // 2 for i in range(80)]  # 40 strong answers, 2 error pairs each
    PARA_IDS = [i // 2 for i in range(80)]  # the same 40, 2 paraphrases each
    PARA_PAIR_IDS = [f"{i // 2}-p{i % 2 + 1}" for i in range(80)]

    def results(self, edit_good):
        return [
            head(
                "auditor",
                "mean",
                s,
                {
                    "confirm": scores(80, 0.95, self.ERR_IDS),
                    "pconfirm": scores(80, edit_good, self.PARA_IDS),
                },
            )
            for s in ch.SEEDS
        ]

    def test_an_error_specific_head(self):
        r = run.skeptic_readout(self.results(0.5), "confirm", "pconfirm", set(), self.PARA_PAIR_IDS)
        row = r["auditor-mean"]
        assert row["reading"] == "error-specific" and row["strong_answers"] == 40
        assert row["error_margin"] == pytest.approx(0.45, abs=0.03)

    def test_an_edit_detector(self):
        r = run.skeptic_readout(self.results(0.95), "confirm", "pconfirm", set(), self.PARA_PAIR_IDS)
        assert r["auditor-mean"]["reading"] == "an edit detector"

    def test_partly_an_edit_detector(self):
        r = run.skeptic_readout(self.results(0.7), "confirm", "pconfirm", set(), self.PARA_PAIR_IDS)
        assert r["auditor-mean"]["reading"] == "partly an edit detector"

    def test_flagged_paraphrases_are_left_out(self):
        dropped = {pid for pid in self.PARA_PAIR_IDS if pid.endswith("-p2")}
        r = run.skeptic_readout(self.results(0.5), "confirm", "pconfirm", dropped, self.PARA_PAIR_IDS)
        assert r["auditor-mean"]["paraphrase_pairs"] == 40

    def test_permutation_and_retraining_checks(self):
        results = self.results(0.5)
        perm = [
            head("auditor", "mean", 42, {"confirm": scores(80, 0.5, self.ERR_IDS)}, f"perm{k}")
            for k in range(20)
        ]
        step4 = [head("auditor", "mean", s, {"confirm": scores(80, 0.95, self.ERR_IDS)}) for s in ch.SEEDS]
        r = run.skeptic_readout(results, "confirm", "pconfirm", set(), self.PARA_PAIR_IDS, perm, step4)
        row = r["auditor-mean"]
        assert row["permutation"]["passes"] and row["permutation"]["p"] == pytest.approx(1 / 21)
        assert row["retrain_reproduces"] and row["retrain_max_validation_change"] == 0
        readable = run.skeptic_role_readable(r, {"auditor-mean": {"passes": True}})
        assert readable == {"auditor-mean": True}
        assert run.skeptic_role_readable(r, {"auditor-mean": {"passes": False}}) == {"auditor-mean": False}


class TestParaphrasePairs:
    def test_strong_items_are_distinct_answers(self):
        pairs = [{"prompt_id": "t1", "prompt": "q", "strong": "a", "flawed": "b"}] * 2 + [
            {"prompt_id": "t2", "prompt": "q2", "strong": "c", "flawed": "d"}
        ]
        assert [it["prompt_id"] for it in sk.strong_items(pairs)] == ["t1", "t2"]

    def test_two_paraphrases_on_different_sentences(self):
        def reword(chat):
            text = chat.turns[0][1]
            if "different sentence" in text and "Ice floats" in text.split("which must stay")[1]:
                return json.dumps(
                    {
                        "original": "It freezes at 0 C under the same conditions.",
                        "edited": "It freezes at 0 C in the same conditions.",
                    }
                )
            return json.dumps(
                {
                    "original": "Ice floats on water because it is less dense.",
                    "edited": "Ice floats on water as it is less dense.",
                }
            )

        items = [{"prompt_id": "t1", "topic": "x", "prompt": "Why?", "strong": ANSWER}]
        pairs, report = sk.plant(lib.FakeBackend(reword), "sys", items)
        assert [p["pair_id"] for p in pairs] == ["t1-p1", "t1-p2"]
        assert sk.sentences(pairs[0])[0] != sk.sentences(pairs[1])[0]
        assert report["planted"] == 2 and report["dropped_second_on_same_sentence"] == 0

    def test_a_second_paraphrase_on_the_same_sentence_is_dropped(self):
        def reword(chat):
            return json.dumps(
                {
                    "original": "It freezes at 0 C under the same conditions.",
                    "edited": "It freezes at 0 C in the same conditions.",
                }
            )

        items = [{"prompt_id": "t1", "topic": "x", "prompt": "Why?", "strong": ANSWER}]
        pairs, report = sk.plant(lib.FakeBackend(reword), "sys", items)
        assert len(pairs) == 1 and report["dropped_second_on_same_sentence"] == 1

    def test_the_meaning_check_flags_and_counts_unparsed(self):
        pairs = [
            {
                "pair_id": f"t{i}-p1",
                "strong": ANSWER,
                "flawed": ANSWER.replace("freezes at 0 C", f"freezes at {i} C"),
            }
            for i in range(1, 4)
        ]
        replies = iter(
            ['{"same_meaning": true, "reason": "ok"}', '{"same_meaning": false, "reason": "number"}', "?"]
        )
        result = sk.verify(lib.FakeBackend(lambda chat: next(replies)), "sys", pairs)
        assert result["kept_meaning"] == 1
        assert result["flagged_changed_meaning"] == ["t2-p1"] and result["unparsed"] == ["t3-p1"]
        original, edited = sk.sentences(pairs[0])
        assert original == "It freezes at 0 C under the same conditions."
        assert edited == "It freezes at 1 C under the same conditions."
