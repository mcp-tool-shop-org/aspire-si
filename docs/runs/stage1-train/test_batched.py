"""Batched generation (B4) on a tiny random Qwen3, CPU, float32: a row's tokens don't depend on its batch-mates
(per-item generators), continuations carry token ids and the row's generator, batches keep to the plan and the
budget, and the sampling is the harness's own settings."""

import pytest
import torch

import batched

STOP = {1}
SAMPLING = batched.SAMPLING


@pytest.fixture(scope="module")
def model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
                      attn_implementation="sdpa")
    m = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():  # flatten the logits a little, so samples vary and rarely hit the stop token early
        m.lm_head.weight.mul_(0.2)
    return m


def work(n, lengths=None):
    g = torch.Generator().manual_seed(7)
    lengths = lengths or [3 + (i * 5) % 11 for i in range(n)]
    return [{"key": f"q{i}-s{i % 3}", "item_id": f"q{i}", "sample": i % 3,
             "ids": torch.randint(2, 64, (lengths[i],), generator=g).tolist()} for i in range(n)]


def run(model, w, **kw):
    kw = {"pad_id": 0, "stop_ids": STOP, "max_new": 40, "sampling": SAMPLING, "kernel": "math", **kw}  # no
    # efficient kernel on CPU; the rig pins efficient
    return batched.generate_all(model, w, **kw)


def test_a_rows_tokens_do_not_depend_on_its_batch_mates(model):
    w = work(11)
    together, log = run(model, w, batch=8, b_tok=8 * 200, chunk_max=64, chunk_min=8)
    alone = {x["key"]: run(model, [x], batch=1, b_tok=200, chunk_max=64, chunk_min=8)[0][x["key"]] for x in w}
    assert {k: r["new"] for k, r in together.items()} == {k: r["new"] for k, r in alone.items()}
    assert len({tuple(r["new"]) for r in together.values()}) == 11  # not vacuous: every row differs
    assert max(r["new_tokens"] for r in together.values()) == 40  # and some run to the cap
    assert [b["size"] for b in log if b["kind"] in ("first", "padded")] == [8, 8]  # 11 rows: 8, then 3 padded to 8


def test_continuations_carry_ids_and_the_generator_so_chunking_changes_nothing(model):
    w = work(5)
    one, _ = run(model, w, batch=8, b_tok=8 * 200, chunk_max=64, chunk_min=8)
    chunked, log = run(model, w, batch=8, b_tok=8 * 60, chunk_max=10, chunk_min=8)
    assert {k: r["new"] for k, r in one.items()} == {k: r["new"] for k, r in chunked.items()}
    assert any(b["kind"] == "continuation" for b in log) and max(r["chunks"] for r in chunked.values()) > 1


def test_the_cap_and_the_stop_token_are_the_harness_rules(model):
    out, _ = run(model, work(6), batch=8, b_tok=8 * 200, chunk_max=64, chunk_min=8, max_new=12)
    for r in out.values():
        ended = bool(r["new"]) and r["new"][-1] in STOP
        assert r["new_tokens"] == len(r["new"]) <= 12 and r["truncated"] == (not ended and r["new_tokens"] >= 12)
        assert STOP.isdisjoint(r["new"][:-1])  # nothing after a stop
        assert r["path"] == batched.PATH and r["seconds_shared"] and r["first_logprob"] <= 0


def test_the_seed_is_per_item_and_sample_and_stable():
    assert batched.seed_for("T05-E-01-v0", 0) == batched.seed_for("T05-E-01-v0", 0)
    assert len({batched.seed_for("T05-E-01-v0", k) for k in range(3)}) == 3
    assert batched.seed_for("T05-E-01-v0", 0) < 2 ** 63


def test_the_plan_holds_first_batches_at_8_and_keeps_every_batch_in_the_budget():
    rows = [{"key": f"k{i}", "ids": [0] * (100 + i * 37), "new": []} for i in range(19)]
    p = batched.plan(rows, b_tok=28000, batch=8, chunk_max=2048, chunk_min=512)
    assert [len(k) for k, _, _ in p] == [8, 8, 3] and [kind for *_, kind in p] == ["first", "first", "padded"]
    for keys, chunk, _ in p:
        longest = max(100 + int(k[1:]) * 37 for k in keys)
        assert 8 * (longest + chunk) <= 28000 and 512 <= chunk <= 2048
    assert p == batched.plan(list(reversed(rows)), b_tok=28000)  # the same work makes the same batches
    long = [{"key": f"L{i}", "ids": [0] * 3400, "new": []} for i in range(8)]  # 8 x (3400 + 512) > 28000
    lp = batched.plan(long, b_tok=28000)
    assert {kind for *_, kind in lp} == {"long-prompt"} and all(len(k) * (3400 + 512) <= 28000 for k, _, _ in lp)
    cont = batched.plan(rows, b_tok=28000, first=False)
    assert all(len(k) * (max(100 + int(x[1:]) * 37 for x in k) + c) <= 28000 for k, c, _ in cont)


def test_the_sampling_settings_are_the_harnesses():
    d = batched.harness.DECODING["sampled"]
    assert SAMPLING == {"temperature": 0.6, "top_k": 20, "top_p": 0.95} and d["min_p"] == 0.0
    assert {k: d[k] for k in SAMPLING} == SAMPLING and batched.MAX_NEW == batched.harness.MAX_NEW_TOKENS == 16384
    logits = torch.tensor([[5.0, 4.0, 3.0, 0.0, -1.0]])
    s = batched.process_logits(logits, 1.0, top_k=3, top_p=1.0)
    assert torch.isinf(s[0, 3:]).all() and torch.isfinite(s[0, :3]).all()
    s = batched.process_logits(logits, 1.0, top_k=0, top_p=0.5)
    assert torch.isfinite(s[0, 0]) and torch.isinf(s[0, 1:]).all()  # 5.0 alone holds over half the mass


def test_duplicate_work_keys_are_refused(model):
    w = work(2)
    with pytest.raises(ValueError, match="duplicate"):
        run(model, [w[0], w[0]])


class FakeTok:
    """Just enough of a tokenizer: the chat template joins the user text; ids are character codes mod 60 + 2."""
    eos_token_id, pad_token_id = 1, 0

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert not tokenize and add_generation_prompt and enable_thinking is batched.harness.THINKING
        return "|".join(m["content"] for m in messages)

    def __call__(self, text):
        return {"input_ids": [2 + ord(c) % 60 for c in text]}

    def convert_tokens_to_ids(self, token):
        return 1

    def decode(self, ids, skip_special_tokens):
        return "".join(chr(97 + i % 26) for i in ids if not (skip_special_tokens and i in (0, 1)))


def test_generate_many_returns_the_harness_fields_for_every_ask(model):
    log = []
    gm = batched.make_generate_many(model, FakeTok(), log, batch=8, b_tok=8 * 200, kernel="math")
    asks = [{"key": f"T05-E-0{i}-v0|{s}", "item_id": f"T05-E-0{i}-v0", "sample": s,
             "messages": [{"role": "user", "content": f"Is item {i} supported?"}]} for i in range(3) for s in range(3)]
    out = gm(asks)
    assert set(out) == {a["key"] for a in asks} and log
    for r in out.values():
        assert {"text", "prompt_tokens", "new_tokens", "truncated", "seconds"} <= set(r)
        assert r["prompt_tokens"] == len("Is item 0 supported?")
    assert batched.stop_ids(model, FakeTok()) >= {1}


def test_on_done_fires_once_per_finished_row_even_across_chunks(model):
    seen = []
    w = work(5)
    out, _ = run(model, w, batch=8, b_tok=8 * 60, chunk_max=10, chunk_min=8,
                 on_done=lambda key, tokens, seconds: seen.append((key, tokens)))
    assert sorted(k for k, _ in seen) == sorted(x["key"] for x in w)
    assert {k: t for k, t in seen} == {k: r["new_tokens"] for k, r in out.items()}


class CountingLM(torch.nn.Module):
    """A stand-in model recording every call: half the probability mass on the stop token (1), the rest spread, so
    rows stop at random, each at its own generator's draw."""

    def __init__(self, vocab=64):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(1))
        self.vocab, self.calls = vocab, []

    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, use_cache=True,
                logits_to_keep=None):
        from types import SimpleNamespace
        self.calls.append({"width": input_ids.shape[1], "logits_to_keep": logits_to_keep})
        n = input_ids.shape[1] if logits_to_keep is None else logits_to_keep
        logits = torch.full((input_ids.shape[0], n, self.vocab), 0.0)
        logits[..., 1] = float(torch.log(torch.tensor(self.vocab - 1.0)))  # p(stop) = 0.5
        return SimpleNamespace(logits=logits, past_key_values=None)


def test_prefill_keeps_only_the_last_logits_and_padded_rows_never_run_on_alone():
    """Argon's finding 4: logits_to_keep=1 on the prefill, and decoding stops once every real row is done."""
    lm = CountingLM()
    w = work(3)  # 3 real rows, padded to 8
    out, log = batched.generate_all(lm, w, pad_id=0, stop_ids=STOP, max_new=200, sampling=SAMPLING, kernel="math",
                                    batch=8, b_tok=8 * 300, chunk_max=200, chunk_min=8)
    assert lm.calls[0]["logits_to_keep"] == 1 and lm.calls[0]["width"] > 1  # the prefill
    longest = max(r["new_tokens"] for r in out.values())
    assert len(lm.calls) == 1 + longest  # one prefill and one step per real token: no step for padded rows alone
    assert log[0]["kind"] == "padded" and log[0]["size"] == 8
