"""The three-line answer parser. A copy of parse_answer in docs/runs/stage1-eval/harness.py, which is
hash-frozen with the evaluation and can't import this package. A test in docs/runs/stage1-train asserts the two
sources are identical, so the copy can never drift."""

import re


def parse_answer(answer: str) -> dict:
    def line(tag):
        m = re.search(rf"^\s*{tag}\s*:\s*(.*)$", answer, re.M | re.I)
        return m.group(1).strip() if m else None
    verdict = (line("VERDICT") or "").lower().strip(" .*`")
    deciding = line("DECIDING")
    escalate = line("ESCALATE")
    return {"verdict": verdict if verdict in ("supported", "unsupported", "cannot_tell") else None,
            "verdict_raw": line("VERDICT"), "deciding": deciding, "escalate": escalate}
