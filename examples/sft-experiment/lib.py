"""Shared pieces of the fine-tune-then-ASPIRE experiment (aspire-si#11).

The teacher's requests and parsing are aspire-si's own (`evaluation_request`, `parse_evaluation`,
the student's system message), so scores here are on the same footing as an ASPIRE run's.

Backends: "vllm" for the pod (batched, fast), "hf" for transformers, and "fake" for CPU tests,
which answers from a callable instead of a model.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from aspire.teachers.base import BaseTeacher, ChallengeType, EvaluationDimension
from aspire.teachers.local import evaluation_request, extract_json_object, parse_evaluation

# The student's system message in ASPIRE's dialogues (aspire.dialogue.generator).
STUDENT_SYSTEM = (
    "You are a helpful assistant engaged in a learning dialogue. "
    "Respond thoughtfully and be willing to revise your thinking when challenged."
)
DIMENSIONS = [d.value for d in EvaluationDimension]

# The 32 control prompts' domain, as topic areas.
TOPICS = {
    "programming-concepts": "programming concepts: recursion, data structures, hash tables, types, naming",
    "software-engineering": (
        "software engineering judgment: testing, rewrites vs patches, debugging, race conditions"
    ),
    "computer-systems": "how computers and networks work: compilers, interpreters, HTTP/HTTPS, web requests",
    "algorithms": "algorithms and complexity: big-O, searching, sorting, gradient descent",
    "databases": "data storage trade-offs: SQL vs NoSQL, indexing, consistency",
    "everyday-physics": "everyday physics: density, light and colour, heat, seasons, buoyancy",
    "biology-health": "biology and health: vaccines, photosynthesis, the immune system, food safety",
    "earth-climate": "earth science and climate: weather vs climate, the water cycle, energy",
    "probability-puzzles": (
        "probability and reasoning puzzles: Monty Hall, conditional probability, word problems"
    ),
    "statistics-causation": "statistics and causation: correlation, sampling, misleading statistics",
    "ethics": "applied ethics: promises, honesty, utilitarianism and its objections, fairness",
    "economics-policy": "economics and public policy: inflation, transport, energy, trade-offs",
}


def teacher_system_prompt(model: str) -> str:
    """The system prompt a `local:<model>` teacher used in the control runs."""
    name = "local:" + model.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    shim = SimpleNamespace(name=name, description="A teacher powered by a local language model")
    return BaseTeacher.get_system_prompt(shim)  # type: ignore[arg-type]


def teacher_name(model: str) -> str:
    return "local:" + model.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


# ---------------------------------------------------------------- text similarity


_STOPWORDS = set(
    "a an the and or but if then than so of to in on at by for with from into onto about over under "
    "is are was were be been being do does did done have has had having can could would should will "
    "may might must shall it its this that these those there their they them you your we our he she "
    "his her what which who whom whose when where why how not no yes as also just very more most "
    "some any each every all both either neither one two between explain describe give example "
    "short brief someone something".split()
)


def content_words(text: str) -> set[str]:
    """Lower-cased words of three or more letters that carry content (stopwords and phrasing removed)."""
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 3 and w not in _STOPWORDS}


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of content words: 1.0 for the same subject matter, near 0 for unrelated.

    Shared phrasing ("how would you explain ... ?") does not count, so two questions on different
    subjects in the same template are not duplicates, while a paraphrase of the same question is.
    """
    wa, wb = content_words(a), content_words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def dedupe(
    candidates: Sequence[tuple[str, str]],
    exclude: Sequence[str],
    exclude_threshold: float = 0.5,
    duplicate_threshold: float = 0.7,
    embed: Callable[[Sequence[str]], Any] | None = None,
    embed_exclude_threshold: float = 0.85,
    embed_duplicate_threshold: float = 0.92,
) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Keep (topic, prompt) pairs unlike every excluded prompt and every prompt already kept.

    Two checks, and either drops a prompt: content-word overlap (`similarity`), and, when `embed`
    is given (texts -> unit vectors), embedding cosine, which also catches paraphrases that share
    no words. Returns the kept pairs and a record of each one dropped, with the reason.
    """
    vectors = embed([p for _, p in candidates]) if embed else None
    exclude_vectors = embed(list(exclude)) if embed and exclude else None
    kept: list[tuple[str, str]] = []
    kept_idx: list[int] = []
    dropped: list[dict[str, Any]] = []
    for i, (topic, prompt) in enumerate(candidates):
        near_eval = max((similarity(prompt, e) for e in exclude), default=0.0)
        cos_eval = float(max(exclude_vectors @ vectors[i])) if exclude_vectors is not None else 0.0
        if near_eval >= exclude_threshold or cos_eval >= embed_exclude_threshold:
            dropped.append(
                {
                    "prompt": prompt,
                    "reason": "near an evaluation prompt",
                    "similarity": near_eval,
                    "cosine": cos_eval,
                }
            )
            continue
        near_kept = max((similarity(prompt, k) for _, k in kept), default=0.0)
        cos_kept = float(max(vectors[kept_idx] @ vectors[i])) if vectors is not None and kept_idx else 0.0
        if near_kept >= duplicate_threshold or cos_kept >= embed_duplicate_threshold:
            dropped.append(
                {"prompt": prompt, "reason": "duplicate", "similarity": near_kept, "cosine": cos_kept}
            )
            continue
        kept.append((topic, prompt))
        kept_idx.append(i)
    return kept, dropped


class Embedder:  # pragma: no cover - downloads a model
    """Unit sentence vectors from BAAI/bge-small-en-v1.5 (MIT): CLS pooling, normalized."""

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5", device: str = "cpu"):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).to(device).eval()
        self.device = device

    def __call__(self, texts: Sequence[str]):
        import numpy as np

        out = []
        with self.torch.no_grad():
            for start in range(0, len(texts), 64):
                batch = self.tokenizer(
                    list(texts[start : start + 64]),
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)
                cls = self.model(**batch).last_hidden_state[:, 0]
                out.append(self.torch.nn.functional.normalize(cls, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 384))


def split_held_out(
    pairs: Sequence[tuple[str, str]], held_out: int, train_cap: int, seed: int = 42
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Stratified by topic: about held_out/len(topics) per topic held out, the rest (capped) for training."""
    rng = random.Random(seed)
    by_topic: dict[str, list[tuple[str, str]]] = {}
    for topic, prompt in pairs:
        by_topic.setdefault(topic, []).append((topic, prompt))
    for items in by_topic.values():
        rng.shuffle(items)
    topics = sorted(by_topic)
    held: list[tuple[str, str]] = []
    i = 0
    while len(held) < held_out and any(by_topic[t] for t in topics):
        t = topics[i % len(topics)]
        if by_topic[t]:
            held.append(by_topic[t].pop())
        i += 1
    rest = [p for t in topics for p in by_topic[t]]
    rng.shuffle(rest)
    return rest[:train_cap], held


def parse_question_list(reply: str) -> list[str]:
    """Questions from a reply that should be a JSON list of strings (or {"questions": [...]})."""
    data: Any = None
    match = re.search(r"\[.*\]", reply, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0), strict=False)
        except json.JSONDecodeError:
            data = None
    if data is None:
        obj = extract_json_object(reply)
        data = obj.get("questions") if obj else None
    if not isinstance(data, list):
        return []
    return [q.strip() for q in data if isinstance(q, str) and 15 <= len(q.strip()) <= 400]


# ---------------------------------------------------------------- requests


def question_request(topic_description: str, count: int) -> str:
    """The teacher never sees the evaluation prompts, so it cannot paraphrase them."""
    return f"""Write {count} distinct questions a curious adult might ask about {topic_description}.
Each should call for an explanation or a reasoned answer of one to three paragraphs, not a
one-word fact. Vary the style: explain-why, compare, judge a trade-off, solve a short problem,
argue a position.

Reply with a JSON list of {count} strings and nothing else."""


def challenge_request(prompt: str, response: str, challenge_type: str) -> str:
    """The local teacher's challenge request (aspire.teachers.local.LocalTeacher.challenge)."""
    return f"""Generate a {challenge_type} challenge for this student response.

Original prompt: {prompt}

Student's response: {response}


Generate a single challenging question or statement. Reply with the challenge only."""


FLAW_KINDS = (
    "one wrong fact or number",
    "one reasoning step that does not follow from the step before it",
)

EDIT_REQUEST = """Here is a question and a strong answer to it.

Question: {prompt}

Strong answer: {answer}

Plant {kind} in this answer by changing a single sentence, so that the answer becomes wrong in
substance while still reading as confident and fluent. Change as few words of that sentence as you
can. Do not mark, hint at, explain or correct the change.

Reply with a JSON object only:
{{"original": "<the sentence, copied exactly from the answer>",
  "edited": "<the same sentence with the error planted>"}}"""

# JSON decodes \f, \b, \t and \r in a model's LaTeX (\frac, \beta, \times, \rho) as control
# characters; put the backslash back before looking for the sentence in the answer.
_UNESCAPE = {"\f": "\\f", "\b": "\\b", "\t": "\\t", "\r": "\\r"}


def _restore_latex(text: str) -> str:
    for char, escaped in _UNESCAPE.items():
        text = text.replace(char, escaped)
    return text


def apply_edit(reply: str, answer: str) -> tuple[str | None, str]:
    """The answer with the model's one-sentence edit applied, or None and why it was rejected."""
    data = extract_json_object(reply)
    if not data or not isinstance(data.get("original"), str) or not isinstance(data.get("edited"), str):
        return None, "no JSON edit"
    for original, edited in (
        (data["original"], data["edited"]),
        (_restore_latex(data["original"]), _restore_latex(data["edited"])),
    ):
        original, edited = original.strip(), edited.strip()
        if not original or original not in answer:
            continue
        if edited == original:
            return None, "edit unchanged"
        flags = [m for m in SELF_FLAG_MARKERS if edited.lower().count(m) > original.lower().count(m)]
        if flags:
            return None, "self-flagging: " + ", ".join(flags)
        return answer.replace(original, edited, 1), "ok"
    return None, "original not in the answer"


# Words a rewrite uses when it points at its own planted error.
SELF_FLAG_MARKERS = ("error", "(this is", "should be", "incorrect", "mistake", "note:")


def scoring_request(prompt: str, response: str, history: str = "") -> str:
    """The exact request ASPIRE's local teacher scores with (no improved response)."""
    return evaluation_request(prompt, response, DIMENSIONS, history, generate_improved=False)


def challenge_types(seed: int) -> Callable[[], str]:
    rng = random.Random(seed)
    kinds = [c.value for c in ChallengeType]
    return lambda: rng.choice(kinds)


# ---------------------------------------------------------------- generation backends


@dataclass
class Chat:
    system: str
    turns: list[tuple[str, str]] = field(default_factory=list)  # (role, content)

    def messages(self) -> list[dict[str, str]]:
        return [{"role": "system", "content": self.system}] + [
            {"role": r, "content": c} for r, c in self.turns
        ]


class Backend:
    """Batched chat generation. `generate(chats, max_tokens, temperature)` returns one text per chat.

    After each call, `last_truncated[i]` says whether reply i stopped at `max_tokens`.
    """

    last_truncated: list[bool] = []

    def generate(self, chats: Sequence[Chat], max_tokens: int, temperature: float) -> list[str]:
        raise NotImplementedError


class FakeBackend(Backend):
    """Answers from a function of the chat; for CPU tests. One character counts as one token, so a
    reply longer than `max_tokens` is cut there and flagged as truncated."""

    def __init__(self, answer: Callable[[Chat], str]):
        self.answer = answer
        self.calls: list[Chat] = []
        self.last_truncated = []

    def generate(self, chats: Sequence[Chat], max_tokens: int, temperature: float) -> list[str]:
        self.calls.extend(chats)
        replies = [self.answer(c) for c in chats]
        self.last_truncated = [len(r) > max_tokens for r in replies]
        return [r[:max_tokens] for r in replies]


class VllmBackend(Backend):  # pragma: no cover - needs a GPU and vllm
    def __init__(
        self, model: str, max_model_len: int = 8192, gpu_memory_utilization: float = 0.9, seed: int = 42
    ):
        from vllm import LLM

        self.llm = LLM(
            model=model,
            dtype="bfloat16",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
        )
        self.seed = seed

    def generate(self, chats: Sequence[Chat], max_tokens: int, temperature: float) -> list[str]:
        from vllm import SamplingParams

        # No per-request seed: the engine is seeded once (reproducible run to run), and repeated
        # identical requests still sample independently, which the noise-floor rescoring needs.
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        outputs = self.llm.chat([c.messages() for c in chats], params, use_tqdm=False)
        self.last_truncated = [o.outputs[0].finish_reason == "length" for o in outputs]
        return [o.outputs[0].text for o in outputs]


class ServerBackend(Backend):  # pragma: no cover - needs a running server
    """An OpenAI-compatible chat server (llama.cpp's llama-server on the local GPU), with
    `workers` requests in flight to fill the server's parallel slots."""

    def __init__(self, url: str = "http://127.0.0.1:8010", workers: int = 4, timeout: float = 600):
        self.url = url.rstrip("/") + "/v1/chat/completions"
        self.workers = workers
        self.timeout = timeout

    def _one(self, chat: Chat, max_tokens: int, temperature: float) -> tuple[str, bool]:
        import urllib.request

        body = {"messages": chat.messages(), "max_tokens": max_tokens, "temperature": temperature}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), headers={"content-type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            choice = json.loads(response.read())["choices"][0]
        return choice["message"]["content"] or "", choice.get("finish_reason") == "length"

    def generate(self, chats: Sequence[Chat], max_tokens: int, temperature: float) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.workers) as pool:
            out = list(pool.map(lambda c: self._one(c, max_tokens, temperature), chats))
        self.last_truncated = [t for _, t in out]
        return [text for text, _ in out]


def refuse_cloud(model: str) -> str:
    """The studio's standing rule: no Ollama Cloud models. Any name containing "cloud" is refused."""
    if "cloud" in model.lower():
        raise ValueError(f"{model!r} is an Ollama Cloud model; only local models may be used")
    return model


class OllamaBackend(Backend):  # pragma: no cover - needs the local Ollama daemon
    """The local Ollama daemon's chat API, one request at a time. Cloud models are refused.

    `unload()` asks the daemon to drop the model from GPU memory (keep_alive 0) without the
    `ollama` CLI, which can hang automation on this rig."""

    def __init__(self, model: str, url: str = "http://127.0.0.1:11434", timeout: float = 900):
        self.model = refuse_cloud(model)
        self.url = url.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, body: dict) -> dict:
        import urllib.request

        req = urllib.request.Request(
            self.url + path,
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read())

    def model_id(self) -> str:
        """The local model's digest, as `ollama list` shows its ID (first 12 characters)."""
        import urllib.request

        with urllib.request.urlopen(self.url + "/api/tags", timeout=30) as response:
            models = json.loads(response.read())["models"]
        for m in models:
            if m["name"] == self.model:
                return m["digest"][:12]
        raise ValueError(f"{self.model!r} is not installed locally")

    def generate(self, chats: Sequence[Chat], max_tokens: int, temperature: float) -> list[str]:
        out, cut = [], []
        for chat in chats:
            reply = self._post(
                "/api/chat",
                {
                    "model": self.model,
                    "messages": chat.messages(),
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            )
            out.append(reply["message"]["content"] or "")
            cut.append(reply.get("done_reason") == "length")
        self.last_truncated = cut
        return out

    def unload(self) -> None:
        self._post("/api/generate", {"model": self.model, "keep_alive": 0})


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_of(reply: str, model: str) -> tuple[float, str, dict[str, float]]:
    """Overall score, how the reply was parsed, and the dimension scores, as ASPIRE reads them."""
    ev = parse_evaluation(reply, False, teacher_name(model))
    dims = {d.dimension.value: float(d.score) for d in ev.dimension_scores}
    return float(ev.overall_score), str(ev.metadata.get("parse")), dims


# ---------------------------------------------------------------- statistics


def pairwise_accuracy(strong: Sequence[float], flawed: Sequence[float]) -> float:
    """The fraction of pairs where the strong answer scores higher (ties count half)."""
    wins = sum(1.0 if s > f else 0.5 if s == f else 0.0 for s, f in zip(strong, flawed))
    return wins / len(strong)


def bootstrap_ci(
    strong: Sequence[float],
    flawed: Sequence[float],
    resamples: int = 2000,
    seed: int = 0,
    groups: Sequence[Any] | None = None,
) -> tuple[float, float]:
    """A 95% percentile bootstrap interval for pairwise_accuracy.

    With `groups` (one label per pair, e.g. the prompt), whole groups are resampled, so pairs that
    share a prompt are not counted as independent evidence.
    """
    rng = random.Random(seed)
    labels = list(groups) if groups is not None else list(range(len(strong)))
    members: dict[Any, list[int]] = {}
    for i, g in enumerate(labels):
        members.setdefault(g, []).append(i)
    clusters = list(members.values())
    stats = []
    for _ in range(resamples):
        idx = [i for _ in clusters for i in clusters[rng.randrange(len(clusters))]]
        stats.append(pairwise_accuracy([strong[i] for i in idx], [flawed[i] for i in idx]))
    stats.sort()
    return stats[int(0.025 * resamples)], stats[int(0.975 * resamples) - 1]


def mean_ci(values: Sequence[float], resamples: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and its 95% percentile bootstrap interval."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return sum(values) / n, means[int(0.025 * resamples)], means[int(0.975 * resamples) - 1]
