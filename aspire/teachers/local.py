"""
Local model teacher implementation.

Enables using local LLMs (via transformers) as teachers, useful for:
- Faster iteration during development
- Privacy-sensitive applications
- Cost reduction at scale
- Specialized domain teachers from fine-tuned models
"""

import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from aspire.teachers.base import (
    BaseTeacher,
    ChallengeType,
    DialogueHistory,
    DimensionScore,
    EvaluationDimension,
    TeacherChallenge,
    TeacherEvaluation,
)

# The most request tokens a teacher is given, whatever its context.
MAX_INPUT_TOKENS = 4096


def default_device() -> str:
    """`cuda` when torch can use it, `cpu` otherwise."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def extract_json_object(text: str) -> dict | None:
    """The first JSON object in a model's reply, fenced or bare, or None."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _clamp_score(value: object) -> float | None:
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if score != score:  # NaN
        return None
    return min(10.0, max(0.0, score))


class LocalTeacher(BaseTeacher):
    """
    Teacher powered by a local model.

    Useful for faster iteration, privacy, or using specialized
    fine-tuned models as domain experts.

    The model is asked for JSON with an overall score and a score per evaluation dimension.
    Small models do not always comply, so a reply that is not JSON falls back to finding a
    score in the text, and one with no score at all gets 5.0. Which of the three happened is
    kept in the evaluation's ``metadata["parse"]`` (``json``, ``text`` or ``default``).
    """

    def __init__(
        self,
        model_name_or_path: str,
        load_in_4bit: bool = True,
        load_in_8bit: bool = False,
        device: str | None = None,
        name: str = "Local Teacher",
        description: str = "A teacher powered by a local language model",
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.model_name_or_path = model_name_or_path
        self.device = device or default_device()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config
        quantization_config = None
        if load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        self.model.eval()

    def _format(self, user: str) -> str:
        """The model's own chat format when its tokenizer has one, a generic one otherwise."""
        if isinstance(getattr(self.tokenizer, "chat_template", None), str):
            return self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": user},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"<|system|>\n{self.get_system_prompt()}<|end|>\n<|user|>\n{user}<|end|>\n<|assistant|>\n"

    def _input_limit(self, max_new_tokens: int) -> int:
        """Prompt tokens that fit the model's context with room for the reply."""
        context = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        if not isinstance(context, int) or context <= 0:
            return MAX_INPUT_TOKENS
        return max(64, min(MAX_INPUT_TOKENS, context - max_new_tokens))

    def _generate(self, text: str, max_new_tokens: int, temperature: float) -> str:
        chat = isinstance(getattr(self.tokenizer, "chat_template", None), str)
        # Too long a request loses its beginning, never the instructions at its end.
        self.tokenizer.truncation_side = "left"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self._input_limit(max_new_tokens),
            # A chat template already starts with the model's special tokens.
            add_special_tokens=not chat,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    async def challenge(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        challenge_type: ChallengeType | None = None,
    ) -> TeacherChallenge:
        """Generate a challenge using local model."""
        # Validate inputs
        self._validate_input(prompt=prompt, student_response=student_response)

        if challenge_type is None:
            challenge_type = self.select_challenge_type(dialogue_history)

        history_context = ""
        if dialogue_history and dialogue_history.turns:
            history_context = "\n\nPrevious dialogue:\n"
            for turn in dialogue_history.turns:
                history_context += f"Challenge: {turn.challenge.content}\n"
                history_context += f"Student: {turn.student_response}\n\n"

        request = f"""Generate a {challenge_type.value} challenge for this student response.

Original prompt: {prompt}

Student's response: {student_response}
{history_context}

Generate a single challenging question or statement. Reply with the challenge only."""

        response = self._generate(self._format(request), 256, self.temperature)

        return TeacherChallenge(
            challenge_type=challenge_type,
            content=response.strip(),
            difficulty=0.5,
        )

    async def evaluate(
        self,
        prompt: str,
        student_response: str,
        dialogue_history: DialogueHistory | None = None,
        generate_improved: bool = True,
    ) -> TeacherEvaluation:
        """Evaluate using local model."""
        # Validate inputs
        self._validate_input(prompt=prompt, student_response=student_response)

        history_context = ""
        if dialogue_history and dialogue_history.turns:
            history_context = "\n\nDialogue history:\n"
            for turn in dialogue_history.turns:
                history_context += f"Challenge: {turn.challenge.content}\n"
                history_context += f"Student: {turn.student_response}\n\n"

        dimensions = [d.value for d in self.evaluation_dimensions]
        dimension_lines = ",\n".join(
            f'    {{"dimension": "{d}", "score": 0-10, "explanation": "one sentence"}}' for d in dimensions
        )
        improved_line = (
            ',\n  "improved_response": "an improved version of the response"' if generate_improved else ""
        )
        request = f"""Evaluate this student response on a scale of 0-10.

Original prompt: {prompt}

Student's response: {student_response}
{history_context}

Score it overall and on each of these dimensions: {", ".join(dimensions)}.
Reply with JSON only, in this shape:
{{
  "overall_score": 0-10,
  "dimension_scores": [
{dimension_lines}
  ],
  "reasoning": "two or three sentences",
  "strengths": ["..."],
  "weaknesses": ["..."]{improved_line}
}}"""

        response = self._generate(self._format(request), 768, 0.3)
        return self._parse_evaluation(response, generate_improved)

    def _parse_evaluation(self, response: str, generate_improved: bool) -> TeacherEvaluation:
        """Read the model's reply: JSON if it gave JSON, a score found in the text otherwise."""
        data = extract_json_object(response)
        overall = _clamp_score(data.get("overall_score")) if data else None
        if data is not None and overall is not None:
            dimension_scores = []
            for item in data.get("dimension_scores") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    dimension = EvaluationDimension(str(item.get("dimension", "")).strip().lower())
                except ValueError:
                    continue
                score = _clamp_score(item.get("score"))
                if score is None:
                    continue
                dimension_scores.append(
                    DimensionScore(dimension, score, str(item.get("explanation", "")))
                )
            improved = data.get("improved_response") if generate_improved else None
            return TeacherEvaluation(
                overall_score=overall,
                dimension_scores=dimension_scores,
                reasoning=str(data.get("reasoning", "")) or response,
                improved_response=improved if isinstance(improved, str) and improved.strip() else None,
                strengths=[str(s) for s in data.get("strengths") or [] if isinstance(s, str)],
                weaknesses=[str(s) for s in data.get("weaknesses") or [] if isinstance(s, str)],
                metadata={"teacher": self.name, "parse": "json"},
            )

        score = self._extract_score(response)
        return TeacherEvaluation(
            overall_score=5.0 if score is None else score,
            dimension_scores=[],
            reasoning=response,
            improved_response=self._extract_improved(response) if generate_improved else None,
            metadata={"teacher": self.name, "parse": "default" if score is None else "text"},
        )

    def _extract_score(self, response: str) -> float | None:
        """Extract a 0-10 score from free text, or None when there is none."""
        # Look for patterns like "Score: 7" or "7/10" or just a number
        patterns = [
            r"[Ss]core:?\s*(\d+(?:\.\d+)?)",
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"^(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                score = float(match.group(1))
                return min(10.0, max(0.0, score))

        return None

    def _extract_improved(self, response: str) -> str | None:
        """Extract improved response if present."""
        markers = ["improved version:", "better response:", "improved:"]
        lower_response = response.lower()

        for marker in markers:
            if marker in lower_response:
                idx = lower_response.index(marker)
                return response[idx + len(marker) :].strip()

        return None
