# Vendored from mcp-tool-shop-org/rnd experiments/cuda-graphs/graphed_heads.py at d7e5622b (MIT,
# same author), written by the R&D session and benchmarked on this repo's Llama cache (rnd 1.1.4.1.5):
# pooled heads 7.2-7.7x, attention 25x, batched scoring 12-31x; exact gate 4/5 (the graphed path
# uses AdamW(capturable=True), a slightly different optimizer), outcome gate 5/5.
"""Faster critic-head training and scoring, with an optional CUDA-graph training step.

A drop-in alternative to aspire-si's `critic_heads.train_head` and `score_set`. It runs the same
model, loss, optimizer, batch order and label flips. Three changes, each separately measurable:

1. **Features live on the GPU, stacked and padded once.** The original builds each batch on the
   CPU, row by row, and copies it over. Here every answer's states sit in one [N, T, D] tensor in
   their stored dtype. A batch is an `index_select` by answer number, then `.float()`, which gives
   the same values the original's per-row `.float()` copy gives.
2. **Scoring is batched.** The original scores one pair per forward and calls `float()` twice per
   pair, which waits for the GPU each time. Here a chunk of answers goes in one forward, with one
   transfer to the CPU at the end.
3. **Optional CUDA graph** (`graphs=True`). The whole training step (gather, forward, loss,
   backward, AdamW) is captured once per batch size and replayed. The only per-step work left on
   the CPU is one device-to-device copy of the batch's answer numbers and one replay call.

Padding does not change the result: mean pooling multiplies padding by a zero mask, and attention
pooling fills it with -inf before the softmax. Sums over a longer row can differ in the last bits
(another reduction order), so results match to a tolerance, not bit for bit.

Capture rules this code follows (each one breaks capture or silently corrupts it if ignored):
- Shapes are static: features are padded to one length, and each batch size gets its own graph
  (a full batch, and the shorter last batch of an epoch).
- No CPU sync inside the step: no `.item()`, `float()`, `.tolist()` or data-dependent branches.
  The loss is cloned into a list on the device and read once at the end.
- Warmup runs on a side stream before capture, so kernels, cuBLAS workspaces and AdamW's lazy state
  are created outside the graph.
- **Warmup steps are real steps**: they move the weights and AdamW's moments. Both are restored in
  place afterwards (same storage, so the captured addresses stay valid), so the graphed run starts
  from the same state as an eager run.
- AdamW is built with `capturable=True`, which keeps its step count on the device.
- Seeding covers the CUDA generator too, and one RNG fork spans init and training: dropout runs on
  the GPU, so seeding only the CPU made each head depend on the heads trained before it in the same
  process. The graphed path reseeds CUDA after warmup, which also draws dropout masks.
- Dropout works under capture (PyTorch registers the generator with the graph), but its masks are
  drawn differently from eager mode. Bit-level comparisons need dropout 0. With dropout on, the
  comparison is statistical: the validation result must fall within the eager result's interval.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def batch_plan(
    pair_index: Sequence[tuple[int, int]],
    labels_flip: Sequence[bool] | None,
    seed: int,
    epochs: int,
    batch_pairs: int,
) -> list[list[int]]:
    """The answer numbers of every training batch, in order, exactly as the original loop draws
    them: one `random.Random(seed)` reshuffles the pair order at the start of each epoch, and a
    flipped pair swaps its strong and flawed answers."""
    order_rng = random.Random(seed)
    pairs = list(range(len(pair_index)))
    plan = []
    for _ in range(epochs):
        order_rng.shuffle(pairs)
        for start in range(0, len(pairs), batch_pairs):
            ks: list[int] = []
            for i in pairs[start : start + batch_pairs]:
                s, f = pair_index[i]
                if labels_flip and labels_flip[i]:
                    s, f = f, s
                ks += [s, f]
            plan.append(ks)
    return plan


def stack_features(features: list, masks: list, device: str = "cuda", dtype=None):
    """Stack per-answer [T_k, D] states into one zero-padded [N, T, D] tensor on `device`, with a
    [N, T] mask and the per-answer lengths. `dtype` defaults to the stored dtype (fp16 in the
    caches), which halves the memory of a float32 stack; batches are converted when gathered."""
    import torch

    n = len(features)
    length = max(f.shape[0] for f in features)
    dim = features[0].shape[-1]
    dtype = dtype or features[0].dtype
    x = torch.zeros(n, length, dim, dtype=dtype, device=device)
    m = torch.zeros(n, length, dtype=torch.long, device=device)
    lengths = []
    for k, (f, mk) in enumerate(zip(features, masks)):
        t = f.shape[0]
        x[k, :t] = f.to(device=device, dtype=dtype)
        m[k, :t] = mk.to(device)
        lengths.append(t)
    return x, m, lengths


def pair_loss(role: str, scores):
    """The original's loss: binary cross-entropy on scores/10 for the Auditor, a pairwise logistic
    loss (strong minus flawed) for every other role. Scores alternate strong, flawed."""
    import torch
    import torch.nn.functional as fn

    strong, flawed = scores[0::2], scores[1::2]
    if role == "auditor":
        p = (torch.cat([strong, flawed]) / 10).clamp(1e-6, 1 - 1e-6)
        target = torch.cat([torch.zeros_like(strong), torch.ones_like(flawed)])
        return fn.binary_cross_entropy(p, target)
    return -fn.logsigmoid(strong - flawed).mean()


def _forward(head, x, m, idx, length: int | None = None):
    states, mask = x, m
    if length is not None:  # eager only: trim to the batch's longest answer (known on the host)
        states, mask = x[:, :length], m[:, :length]
    return head(
        hidden_states=states.index_select(0, idx).float(), attention_mask=mask.index_select(0, idx)
    ).score


class GraphedStep:
    """One captured training step per batch size. `step(idx)` copies the batch's answer numbers
    into the graph's static index buffer and replays it; it returns the step's loss tensor, which
    the next replay overwrites (clone it to keep it)."""

    def __init__(
        self, head, opt, x, m, role: str, sizes: Sequence[int], warmup: int = 3, seed: int | None = None
    ):
        import torch

        if opt.state:
            raise ValueError("GraphedStep needs a fresh optimizer: warmup is undone by zeroing its state")
        if not all(g.get("capturable") for g in opt.param_groups):
            raise ValueError("build the optimizer with capturable=True")
        params = [p for p in head.parameters() if p.requires_grad]
        saved = [p.detach().clone() for p in params]
        self.idx, self.loss, self.graphs = {}, {}, {}
        for size in sorted(set(sizes)):
            self.idx[size] = torch.arange(size, device=x.device) % x.shape[0]

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for size in self.idx:
                for _ in range(warmup):
                    opt.zero_grad(set_to_none=True)
                    pair_loss(role, _forward(head, x, m, self.idx[size])).backward()
                    opt.step()
        torch.cuda.current_stream().wait_stream(side)

        # Undo the warmup in place: a fresh AdamW has zero moments and step 0.
        with torch.no_grad():
            for p, s in zip(params, saved):
                p.copy_(s)
            for state in opt.state.values():
                for v in state.values():
                    if torch.is_tensor(v):
                        v.zero_()
        # Warmup also drew dropout masks from the CUDA generator. Reseed it, so the captured graphs
        # start from the head's own seed (replays then advance philox offsets deterministically).
        if seed is not None:
            torch.cuda.manual_seed(seed)

        for size in self.idx:
            opt.zero_grad(set_to_none=True)  # each graph allocates its own grads in its own pool
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                loss = pair_loss(role, _forward(head, x, m, self.idx[size]))
                loss.backward()
                opt.step()
            self.graphs[size], self.loss[size] = g, loss

    def step(self, idx):
        size = idx.numel()
        self.idx[size].copy_(idx)
        self.graphs[size].replay()
        return self.loss[size]


def train_head_fast(
    role: str,
    pooling: str,
    seed: int,
    x,
    m,
    lengths: Sequence[int],
    pair_index: list[tuple[int, int]],
    labels_flip: list[bool] | None = None,
    hparams: dict | None = None,
    graphs: bool = False,
    head_factory=None,
):
    """Train one CriticHead on stacked features (`stack_features`). Same initialisation, loss,
    optimizer settings, batch order and flips as aspire-si's `train_head`. Returns the head (in eval
    mode) and the per-step losses as a list of floats, read from the GPU once at the end."""
    import torch

    if hparams is None:
        from critic_heads import HPARAMS

        hparams = HPARAMS
    if head_factory is None:
        from aspire.critic import CriticHead

        head_factory = CriticHead

    # One fork over init AND training, seeding CPU and CUDA: dropout runs on the GPU, so an unseeded
    # CUDA generator made each head depend on how many heads trained before it in the process
    # (found by ASPIRE, 2026-10-08). The fork restores the caller's RNG state afterwards.
    cuda = x.device.type == "cuda"
    with torch.random.fork_rng(devices=[x.device] if cuda else []):
        torch.manual_seed(seed)
        return _train(
            role, pooling, seed, x, m, lengths, pair_index, labels_flip, hparams, graphs, head_factory
        )


def _train(role, pooling, seed, x, m, lengths, pair_index, labels_flip, hparams, graphs, head_factory):
    import torch

    head = head_factory(
        input_dim=x.shape[-1],
        hidden_dim=hparams["hidden_dim"],
        num_layers=hparams["num_layers"],
        dropout=hparams["dropout"],
        pooling="attention" if pooling == "attention" else "mean",
    )
    head = head.to(x.device)
    opt = torch.optim.AdamW(
        head.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"], capturable=graphs
    )
    plan = batch_plan(pair_index, labels_flip, seed, hparams["epochs"], hparams["batch_pairs"])
    flat = torch.tensor([k for ks in plan for k in ks], dtype=torch.long, device=x.device)
    bounds, at = [], 0
    for ks in plan:
        bounds.append((at, at + len(ks)))
        at += len(ks)

    head.train()
    losses = []
    if graphs:
        runner = GraphedStep(head, opt, x, m, role, [len(ks) for ks in plan], seed=seed)
        for lo, hi in bounds:
            losses.append(runner.step(flat[lo:hi]).detach().clone())
    else:
        for ks, (lo, hi) in zip(plan, bounds):
            loss = pair_loss(role, _forward(head, x, m, flat[lo:hi], max(lengths[k] for k in ks)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.detach())
    head.eval()
    return head, torch.stack(losses).tolist()


def score_set_fast(head, x, m, lengths: Sequence[int], pair_index, chunk: int = 64):
    """Strong and flawed scores per pair, scoring `chunk` answers per forward (trimmed to the
    chunk's longest answer) and reading the GPU once."""
    import torch

    order = [k for pair in pair_index for k in pair]
    out = []
    with torch.no_grad():
        for start in range(0, len(order), chunk):
            ks = order[start : start + chunk]
            idx = torch.tensor(ks, dtype=torch.long, device=x.device)
            out.append(_forward(head, x, m, idx, max(lengths[k] for k in ks)))
    scores = torch.cat(out).tolist()
    return scores[0::2], scores[1::2]
