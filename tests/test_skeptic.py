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
    def test_each_prompt_is_flipped_half_and_the_total_is_balanced(self):
        ids = [0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4]
        for k in range(6):
            flips = ch.balanced_flips(ids, k)
            for g in set(ids):
                members = [f for f, i in zip(flips, ids) if i == g]
                assert len(members) // 2 <= sum(members) <= (len(members) + 1) // 2
            assert abs(sum(flips) - (len(ids) - sum(flips))) <= 1

    def test_which_odd_prompt_gets_its_spare_flipped_varies(self):
        ids = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        assert len({tuple(ch.balanced_flips(ids, k)) for k in range(6)}) > 1
        assert all(sum(ch.balanced_flips(ids, k)) == 6 for k in range(6))

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
        rounds = [1 if i % 4 else 2 for i in range(80)]
        r = run.skeptic_readout(
            self.results(0.5), "confirm", "pconfirm", set(), self.PARA_PAIR_IDS, para_rounds=rounds
        )
        row = r["auditor-mean"]
        assert row["reading"] == "error-specific" and row["strong_answers"] == 40
        assert row["edit_rate_first_round"]["pairs"] == 60 and row["edit_rate_later_rounds"]["pairs"] == 20
        assert row["error_margin"] == pytest.approx(0.45, abs=0.03)

    def test_an_edit_detector(self):
        r = run.skeptic_readout(self.results(0.95), "confirm", "pconfirm", set(), self.PARA_PAIR_IDS)
        assert r["auditor-mean"]["reading"] == "an edit detector"

    def test_partly_an_edit_detector(self):
        r = run.skeptic_readout(self.results(0.7), "confirm", "pconfirm", set(), self.PARA_PAIR_IDS)
        assert r["auditor-mean"]["reading"] == "partly an edit detector"

    def test_two_arms_that_disagree_leave_the_reading_unresolved(self):
        ids2 = [f"{i // 2}-r{i % 2 + 1}" for i in range(80)]
        results = self.results(0.5)
        for r, s in zip(results, ch.SEEDS):
            r["sets"]["prewritten"] = scores(80, 0.95, self.PARA_IDS)
        r = run.skeptic_readout(
            results, "confirm", "pconfirm", set(), self.PARA_PAIR_IDS, secondary=("prewritten", set(), ids2)
        )
        row = r["auditor-mean"]
        assert row["arms_disagree"] and row["reading"] == "unresolved (depends on paraphrase method)"
        assert row["reading_primary_arm"] == "error-specific" and row["secondary_arm"]["pairs"] == 80

    def test_two_arms_that_agree_keep_the_primary_reading(self):
        ids2 = [f"{i // 2}-r{i % 2 + 1}" for i in range(80)]
        results = self.results(0.5)
        for r in results:
            r["sets"]["prewritten"] = scores(80, 0.5, self.PARA_IDS)
        dropped = {pid for pid in ids2 if int(pid.split("-")[0]) >= 20}
        r = run.skeptic_readout(
            results, "confirm", "pconfirm", set(), self.PARA_PAIR_IDS, secondary=("prewritten", dropped, ids2)
        )
        row = r["auditor-mean"]
        assert not row["arms_disagree"] and row["reading"] == "error-specific"
        assert row["secondary_arm"]["pairs"] == 40

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
        assert row["permutation"]["observed_seed42"] == pytest.approx(0.95, abs=0.01)
        assert row["retrain_reproduces"] and row["retrain_max_validation_change"] == 0
        readable = run.skeptic_role_readable(r, {"auditor-mean": {"passes": True}})
        assert readable == {"auditor-mean": True}
        assert run.skeptic_role_readable(r, {"auditor-mean": {"passes": False}}) == {"auditor-mean": False}


class TestLexicalGuard:
    @pytest.mark.parametrize(
        "before, after, why",
        [
            ("Most cells divide.", "All cells divide.", "quantifier"),
            ("It can fail.", "It will fail.", "modal"),
            ("It usually holds.", "It always holds.", "frequency"),
            ("It does work.", "It doesn't work.", "negation"),
            ("The tree is tall.", "The tree is taller.", "comparative"),
            ("It weighs 5 kg here.", "It weighs 6 kg here.", "number"),
            ("They met in the city.", "They met in Paris.", "named entity"),
            ("It takes two hours.", "It takes three hours.", "number word"),
        ],
    )
    def test_claim_flipping_swaps_are_stopped(self, before, after, why):
        assert sk.lexical_guard({"strong": before, "flawed": after}) == why

    def test_a_plain_synonym_passes(self):
        pair = {"strong": "Ice is less dense, so it floats.", "flawed": "Ice is less dense, hence it floats."}
        assert sk.lexical_guard(pair) is None
        assert sk.lexical_guard({"strong": "The method is quick.", "flawed": "The method is fast."}) is None


class TestSizeCap:
    def test_the_cap_follows_each_answers_own_error_with_a_floor(self):
        pairs = [
            {
                "prompt_id": "a",
                "prompt": "q",
                "strong": "The value is 7 today.",
                "flawed": "The value is 9 today.",
            },
            {
                "prompt_id": "b",
                "prompt": "q",
                "strong": "x" * 40 + " end.",
                "flawed": "y" * 20 + "x" * 20 + " end.",
            },
        ]
        items = sk.strong_items(pairs)
        assert [sk.size_cap(it) for it in items] == [8, 40]

    def test_quantiles(self):
        assert sk.quantiles([1, 2, 3, 4, 5]) == [2, 3, 4] and sk.quantiles([]) is None


def swap(sentence, word, synonym):
    return json.dumps({"sentence": sentence, "word": word, "synonym": synonym})


class TestWordSwap:
    S = "Ice floats on water because it is less dense."

    @pytest.mark.parametrize(
        "reply, why",
        [
            ("no json", "no JSON swap"),
            (swap("Ice sinks in oil.", "sinks", "drops"), "sentence not in the answer"),
            (swap(S, "on water", "atop"), "not a single word"),
            (swap(S, "melts", "thaws"), "word not found exactly once"),
            (swap(S, "floats", "Floats"), "synonym unchanged"),
            (swap(S, "floats", "floated"), "synonym is an inflection"),
            (swap(S, "floats", "drift"), "synonym changes the form"),
        ],
    )
    def test_each_rejection_says_why(self, reply, why):
        assert sk.apply_swap(reply, ANSWER) == (None, why)

    def test_a_word_twice_in_the_sentence_is_refused(self):
        answer = "It is what it is."
        assert sk.apply_swap(swap(answer, "is", "remains"), answer) == (None, "word not found exactly once")

    def test_the_swap_is_made_in_code_with_case_kept(self):
        edited, why = sk.apply_swap(swap(self.S, "Ice", "frost"), ANSWER)
        assert why == "ok" and edited == ANSWER.replace("Ice floats", "Frost floats")
        edited, _ = sk.apply_swap(swap(self.S, "floats", "Drifts"), ANSWER)
        assert edited == ANSWER.replace("floats", "drifts", 1)

    def test_the_article_follows_the_synonym(self):
        answer = "It is a large effect. Nothing else."
        edited, why = sk.apply_swap(swap("It is a large effect.", "large", "immense"), answer)
        assert why == "ok" and edited.startswith("It is an immense effect.")
        answer = "It has an enormous effect."
        edited, _ = sk.apply_swap(swap(answer, "enormous", "big"), answer)
        assert edited == "It has a big effect."

    def test_planting_two_swaps_on_different_sentences(self):
        def pick(chat):
            text = chat.turns[0][1]
            if "different sentence" in text:
                return swap("It freezes at 0 C under the same conditions.", "conditions", "circumstances")
            return swap(self.S, "floats", "drifts")

        items = sk.strong_items(
            [
                {
                    "prompt_id": "t1",
                    "topic": "x",
                    "prompt": "Why?",
                    "strong": ANSWER,
                    "flawed": ANSWER.replace("less", "more"),
                }
            ]
        )
        items[0]["error_chars"] = 10
        pairs, report = sk.plant_swaps(lib.FakeBackend(pick), "sys", items)
        assert [p["pair_id"] for p in pairs] == ["t1-w1", "t1-w2"]
        assert pairs[0]["flawed"] == ANSWER.replace("floats", "drifts")
        assert report["planted"] == 2 and report["attempt_outcomes"] == {"ok": 2}

    def test_planting_counts_the_guard_and_the_same_sentence(self):
        def pick(chat):
            return swap(self.S, "less", "fewer")

        items = [{"prompt_id": "t1", "topic": "x", "prompt": "Why?", "strong": ANSWER, "error_chars": 10}]
        pairs, report = sk.plant_swaps(lib.FakeBackend(pick), "sys", items, rounds=2)
        assert pairs == [] and report["attempt_outcomes"] == {"lexical guard: comparative": 2}


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
        assert len(pairs) == 1 and report["dropped_second_on_same_sentence"] >= 1
        assert report["answers_with_a_pair"] == 1 and report["answers_with_two_pairs"] == 0

    def test_the_size_gate_rejects_large_rewordings_and_counts_what_it_costs(self):
        def reword(chat):
            return json.dumps(
                {
                    "original": "Ice floats on water because it is less dense.",
                    "edited": "Floating on water, ice is less dense than the liquid around it.",
                }
            )

        items = [{"prompt_id": "t1", "topic": "x", "prompt": "Why?", "strong": ANSWER}]
        pairs, report = sk.plant(lib.FakeBackend(reword), "sys", items, rounds=3, max_chars=10)
        assert pairs == [] and report["answers_with_no_pair"] == 1
        assert (
            report["attempt_outcomes"].get("over the size gate", 0)
            + sum(v for k, v in report["attempt_outcomes"].items() if k != "ok" and k != "over the size gate")
            >= 1
        )
        assert report["size_gate"] == 10

    def test_edit_chars_matches_the_difflib_measure(self):
        assert sk.edit_chars({"strong": "The value is 7 today.", "flawed": "The value is 9 today."}) == 1
        assert sk.edit_chars({"strong": "It is big.", "flawed": "It is large."}) == 4

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
            [
                '{"same_meaning": true, "claim_changed": false, "reason": "ok"}',
                '{"same_meaning": true, "claim_changed": true, "reason": "number"}',
                "?",
            ]
        )
        result = sk.verify(lib.FakeBackend(lambda chat: next(replies)), "sys", pairs)
        assert result["kept_meaning"] == 1
        assert result["flagged_changed_meaning"] == ["t2-p1"] and result["unparsed"] == ["t3-p1"]
        original, edited = sk.sentences(pairs[0])
        assert original == "It freezes at 0 C under the same conditions."
        assert edited == "It freezes at 1 C under the same conditions."
