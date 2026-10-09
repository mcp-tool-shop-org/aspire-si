"""Property tests for sft.py's label spans (step 0 of the distillation exercise, rnd
experiments/distill-ladder). Hypothesis generates the chats; a failure prints its reproduce blob."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")  # a dev dependency; without it these skip instead of erroring

from hypothesis import given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "sft-experiment"))

import sft  # noqa: E402

TEXT = st.text(alphabet=st.characters(codec="utf-8", exclude_categories=("Cs",)), min_size=1, max_size=40)


class OneTokenPerChar:
    """A consistent template: every turn renders the same wherever it sits."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return {"input_ids": [ord(c) for c in text]} if tokenize else text


class QwenShaped:
    """Qwen3's shape: the last assistant turn renders an empty think block; with thinking off the
    generation prompt carries it too, and earlier assistant turns render without it."""

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True
    ):
        text = ""
        for i, m in enumerate(messages):
            if m["role"] == "assistant" and i == len(messages) - 1:
                text += f"<assistant><think></think>{m['content']}</assistant>"
            else:
                text += f"<{m['role']}>{m['content']}</{m['role']}>"
        if add_generation_prompt:
            text += "<assistant>" if enable_thinking else "<assistant><think></think>"
        return {"input_ids": [ord(c) for c in text]} if tokenize else text


@st.composite
def single_turn(draw):
    chat = [{"role": "system", "content": draw(TEXT)}] if draw(st.booleans()) else []
    return [*chat, {"role": "user", "content": draw(TEXT)}, {"role": "assistant", "content": draw(TEXT)}]


@st.composite
def multi_turn(draw):
    turns = draw(st.lists(st.tuples(TEXT, TEXT), min_size=2, max_size=4))
    return [
        m for q, a in turns for m in ({"role": "user", "content": q}, {"role": "assistant", "content": a})
    ]


def labelled(example):
    return "".join(chr(i) for i, label in zip(example["input_ids"], example["labels"]) if label != sft.IGNORE)


@given(single_turn(), st.sampled_from([("consistent", None), ("qwen", {"enable_thinking": False})]))
def test_single_turn_labels_are_exactly_the_answer_and_never_the_prompt(chat, template):
    kind, kwargs = template
    tok = OneTokenPerChar() if kind == "consistent" else QwenShaped()
    example = sft.tokenize_example(tok, chat, 10_000, kwargs)
    assert labelled(example) == chat[-1]["content"] + "</assistant>"
    prompt = sft._chat_ids(tok, chat[:-1], add_generation_prompt=True, **(kwargs or {}))
    assert all(label == sft.IGNORE for label in example["labels"][: len(prompt)])


@given(multi_turn())
def test_multi_turn_labels_on_a_consistent_template_are_every_answer(chat):
    example = sft.tokenize_example(OneTokenPerChar(), chat, 10_000)
    answers = [m["content"] for m in chat if m["role"] == "assistant"]
    assert labelled(example) == "".join(a + "</assistant>" for a in answers)


@given(multi_turn())
def test_multi_turn_with_thinking_off_raises_rather_than_mislabelling(chat):
    with pytest.raises(ValueError, match="renders differently inside the whole chat"):
        sft.tokenize_example(QwenShaped(), chat, 10_000, {"enable_thinking": False})


@given(single_turn(), st.integers(min_value=1, max_value=200))
def test_drop_and_truncate_never_mix(chat, max_length):
    full = sft.tokenize_example(OneTokenPerChar(), chat, 10_000)
    dropped = sft.tokenize_example(OneTokenPerChar(), chat, max_length, overlong="drop")
    truncated = sft.tokenize_example(OneTokenPerChar(), chat, max_length, overlong="truncate")
    if len(full["input_ids"]) > max_length:
        assert dropped is None
        assert truncated["input_ids"] == full["input_ids"][-max_length:]
        assert truncated["labels"] == full["labels"][-max_length:]
    else:
        assert dropped == full == truncated


@given(st.lists(single_turn(), min_size=1, max_size=12), st.integers(min_value=5, max_value=150))
def test_prepare_accounts_for_every_item(chats, max_length):
    rows = [{"messages": c} for c in chats]
    examples, dropped, unlabelled = sft.prepare_examples(OneTokenPerChar(), rows, max_length, overlong="drop")
    assert len(examples) + dropped + unlabelled == len(rows)
    assert unlabelled == 0  # in drop mode a kept single-turn chat always has its answer
