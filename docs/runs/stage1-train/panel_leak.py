"""Leakage check for the panel (design §3). Built and tested before FROZEN.md exists.

A real run refuses to report a clean pass when FROZEN.md is missing or cannot be parsed.
Pass A is `gates.leakage` on kind-E wraps. Pass B is `gates._shingles(..., 8)` and does not
call `gates._spans`. A hit on the shared instruction stops the check: the instruction is
rewritten, and the source is not dropped for that hit. A hit on the source text drops the item.

The claim of a training or apply row is its statement, else `gates._statement` of the first
user turn, else that user turn. Kind A, kind B, written E and NO-ROLE recall rows are not skipped.
"""

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_PKG = HERE.parents[2] / "packages" / "aspire-si-gates" / "src"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))
import interview  # noqa: E402
import receipts  # noqa: E402
from aspire_si_gates import gates  # noqa: E402

SEALED_SHA_PREFIX = "e1e515b4"
_SECTION = re.compile(r"^## (.+)$", re.M)
_TRAIN = re.compile(r"^\|\s*\d+\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*([0-9a-f]{64})\s*\|$", re.M)
_NOROLE = re.compile(r"^\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|\s*([0-9a-f]{64})\s*\|$", re.M)


class LeakError(Exception):
    """The check cannot run, or the mapping is impossible. The message names no item text."""


def claim_of(row: dict) -> str:
    """Statement, else the parsed statement, else the first user turn. Empty mapping stops."""
    statement = row.get("statement")
    if isinstance(statement, str) and statement.strip():
        return statement.strip()
    turns = row.get("turns") or []
    if turns and isinstance(turns[0], dict):
        user = turns[0].get("user") or ""
        if isinstance(user, str):
            parsed = gates._statement(user)
            if parsed:
                return parsed
            if user.strip():
                return user
    raise LeakError("a row has no turns and no statement")


def held_out_of(row: dict) -> dict:
    return {"id": row.get("id", "?"), "claim": claim_of(row), "material": row.get("material") or ""}


def as_kind_e(item: dict) -> dict:
    return {"id": item["id"], "kind": "E",
            "turns": [{"user": item.get("prompt") or "", "reply": ""}],
            "statement": "", "variant": None, "material": ""}


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, dict):
        out = []
        for inner in value.values():
            out.extend(_strings(inner))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for inner in value:
            out.extend(_strings(inner))
        return out
    return []


def row_shingle_texts(row: dict) -> list[str]:
    """User, thinking, reply, material, variant and r10_twin. Not `gates._spans`."""
    texts = []
    for turn in row.get("turns") or []:
        if isinstance(turn, dict):
            for key in ("user", "thinking", "reply"):
                value = turn.get(key)
                if isinstance(value, str) and value:
                    texts.append(value)
    material = row.get("material")
    if isinstance(material, str) and material:
        texts.append(material)
    texts.extend(_strings(row.get("variant")))
    texts.extend(_strings(row.get("r10_twin")))
    return texts


def task_shingle_texts(task: dict) -> list[str]:
    return [task[key] for key in ("claim", "material") if isinstance(task.get(key), str) and task[key]]


def shingles_of(texts) -> set:
    found = set()
    for text in texts:
        found |= gates._shingles(text, 8)
    return found


class Corpus:
    """The two passes, over one held-out list and one shingle set."""

    def __init__(self, held_out: list[dict], shingles: set, questions: list[str]):
        self.held_out = held_out
        self.shingles = shingles
        self.questions = questions

    def source_hits(self, item: dict) -> list[str]:
        """Pass A hits, then a source-text shingle hit. Instruction hits are not source hits."""
        hits = gates.leakage([as_kind_e(item)], self.held_out, self.questions)
        source = gates._shingles(item.get("source_text") or "", 8)
        if source & self.shingles:
            hits = list(hits) + [f"{item['id']}: source shingle"]
        return hits

    def instruction_hit(self, item: dict) -> bool:
        """The prompt shares an 8-word span that the source text does not."""
        source = gates._shingles(item.get("source_text") or "", 8)
        prompt = gates._shingles(item.get("prompt") or "", 8)
        return bool((prompt - source) & self.shingles)


def corpus_from_parts(training: list[dict], apply_rows: list[dict], tasks: list[dict],
                      questions: list[str]) -> Corpus:
    held = [held_out_of(row) for row in training + apply_rows]
    held += [{"id": task.get("id", "?"), "claim": task.get("claim") or "",
              "material": task.get("material") or ""} for task in tasks]
    texts = []
    for row in training + apply_rows:
        texts.extend(row_shingle_texts(row))
    for task in tasks:
        texts.extend(task_shingle_texts(task))
    texts.extend(questions)
    return Corpus(held, shingles_of(texts), list(questions))


def reject_reason(item: dict, corpus: Corpus) -> str | None:
    """None keeps the item. `instruction` stops the build. Any other string skips the item."""
    if corpus.instruction_hit(item):
        raise LeakError("the shared instruction shares an 8-word span; rewrite it, do not drop the source")
    hits = corpus.source_hits(item)
    if hits:
        return "leakage"
    return None


def parse_frozen(text: str) -> dict:
    """The three tables `freeze.py` renders. A missing section or a bad row fails closed."""
    if "sha256 of the content with CRLF read as LF" not in text:
        raise LeakError("FROZEN.md does not carry the hash rule")
    parts = _SECTION.split(text)
    # split keeps the text before the first heading, then alternating title, body.
    sections = {}
    if len(parts) < 2:
        raise LeakError("FROZEN.md has no sections")
    it = iter(parts[1:])
    for title, body in zip(it, it):
        sections[title.strip()] = body
    for name in ("Training files", "Apply files", "NO-ROLE"):
        if name not in sections:
            raise LeakError("FROZEN.md is missing a section")
    training = [_row(match) for match in _TRAIN.finditer(sections["Training files"])]
    apply_rows = [_row(match) for match in _TRAIN.finditer(sections["Apply files"])]
    no_role = [_row(match) for match in _NOROLE.finditer(sections["NO-ROLE"])]
    if not training or len(no_role) != 1:
        raise LeakError("FROZEN.md tables are incomplete")
    return {"training": training, "apply": apply_rows, "no_role": no_role}


def _row(match: re.Match) -> dict:
    file, items, digest = match.group(1).strip(), match.group(2), match.group(3)
    if file.startswith("/") or ".." in Path(file).parts or ":" in file:
        raise LeakError("FROZEN.md lists a path that is not repo-relative")
    return {"file": file, "items": int(items), "sha256": digest}


def load_listed(root: Path, rows: list[dict]) -> list[dict]:
    """Load each listed file and require `receipts.sha256_file` to match the table."""
    loaded = []
    for row in rows:
        path = root / row["file"]
        if path.is_dir() or not path.is_file():
            raise LeakError("a listed path is not a file")
        if receipts.sha256_file(path) != row["sha256"]:
            raise LeakError("a listed file's sha256 does not match FROZEN.md")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                loaded.append(json.loads(line))
    return loaded


def load_jsonl_file(path: Path, expect_sha: str | None) -> list[dict]:
    """A task file. A directory is refused. `expect_sha` is checked with CRLF read as LF."""
    if path.is_dir() or not path.is_file():
        raise LeakError("the task path is not a file")
    if expect_sha is not None and receipts.sha256_file(path) != expect_sha:
        raise LeakError("the task file's sha256 does not match the pin")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_dev(dev_dir: Path | None = None) -> list[dict]:
    root = dev_dir or receipts.DEV_DIR
    rows = []
    for name, pin in receipts.DEV_FILES.items():
        rows.extend(load_jsonl_file(root / name, pin))
    return rows


def check_items(items: list[dict], corpus: Corpus) -> dict:
    """Classify kept items. An instruction hit raises. Source hits are listed by id only."""
    dropped = []
    for item in items:
        reason = reject_reason(item, corpus)
        if reason:
            dropped.append({"id": item["id"], "reason": reason})
    return {"dropped": dropped, "clean": not dropped}


def corpus_from_frozen(frozen: Path, *, repo: Path | None = None, sealed: Path | None = None,
                       dev_dir: Path | None = None, questions: list[str] | None = None) -> Corpus:
    """The real corpus. The sealed path must be a file and must match the sealed pin."""
    if not frozen.is_file():
        raise LeakError("FROZEN.md is not here")
    tables = parse_frozen(frozen.read_text(encoding="utf-8"))
    root = repo or receipts.REPO
    training = load_listed(root, tables["training"])
    apply_rows = load_listed(root, tables["apply"])
    training += load_listed(root, tables["no_role"])
    sealed_path = sealed if sealed is not None else receipts.SEALED_FILE
    if not receipts.SEALED_SHA256.startswith(SEALED_SHA_PREFIX):
        raise LeakError("the sealed pin does not have the recorded prefix")
    tasks = load_jsonl_file(sealed_path, receipts.SEALED_SHA256)
    tasks += load_dev(dev_dir)
    return corpus_from_parts(training, apply_rows, tasks, questions if questions is not None
                             else list(interview.QUESTIONS))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Panel leakage check. Refuses to pass without FROZEN.md.")
    parser.add_argument("--items", required=True)
    parser.add_argument("--frozen", required=True)
    parser.add_argument("--sealed", default=None)
    args = parser.parse_args(argv)
    frozen = Path(args.frozen)
    if not frozen.is_file():
        print("FROZEN.md is not here; the leakage check has not run", file=sys.stderr)
        return 2
    try:
        corpus = corpus_from_frozen(frozen, sealed=Path(args.sealed) if args.sealed else None)
        items = []
        for line in Path(args.items).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("decision") == "kept":
                items.append(row)
        result = check_items(items, corpus)
    except LeakError as exc:
        if "shared instruction" in str(exc):
            print(str(exc), file=sys.stderr)
            return 1
        print("FROZEN.md could not be checked; this is not a clean pass", file=sys.stderr)
        return 2
    except (json.JSONDecodeError, OSError):
        print("FROZEN.md could not be checked; this is not a clean pass", file=sys.stderr)
        return 2
    if not result["clean"]:
        print(f"leakage hits: {len(result['dropped'])}", file=sys.stderr)
        return 1
    print("leakage: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
