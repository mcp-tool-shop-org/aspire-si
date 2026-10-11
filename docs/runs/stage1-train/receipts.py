"""Stage 1 receipts (schema stage1-receipt-v1): what the runner writes for role-set's ingest, and the firewall.

    w = ReceiptWriter(root, round=2, arm="ROLE", seed=0, run_id="r2-s0-ROLE", adapter_sha256=..., firewall=fw)
    w.write("train_quiz", {...})

Rows go to <root>/<round>/<arm>/<seed>/receipts/<source>.jsonl (root is outputs/stage1-train). The path is built
from the row's own header, so the two can't disagree. role-set ingests these files into its store; the trainer
never opens the store. Field names are agreed with the Dataset Specialist: tell them before renaming any.

The firewall (agreed with R&D and the Dataset Specialist; role-set's ingest checks again at the door). A row is
refused, and nothing is written, when it:
- has a source outside train_quiz, rehearsal and diagnostics, or an arm outside ROLE and NO-ROLE;
- carries an id from the sealed set (sx-), the DEV set (pl-, rv-), or any id the sealed or DEV files hold;
- carries a field only an evaluation sitting writes: q and sitting (the pre-interview), phase, task_id and
  harness_sha256 (the harness and eval_adapter records);
- has any text, at any depth, sharing an 8-word span with the sealed set, the DEV set or the 12 pre-interview
  questions.
The sealed file is read only here, on aspire-si's side, and only to build its spans in memory; it is checked
against its frozen sha256 first. The evaluation code (stage1-eval, eval_adapter.py) never imports this module;
a test checks the import graph.
"""

import ast
import hashlib
import json
import re
import time
from pathlib import Path

SCHEMA = "stage1-receipt-v1"
SOURCES = ("train_quiz", "rehearsal", "diagnostics")
ARMS = ("ROLE", "NO-ROLE")
REFUSED_ID_PREFIXES = ("sx-", "pl-", "rv-")
EVAL_ONLY_KEYS = ("q", "sitting", "phase", "task_id", "harness_sha256")
SPAN = 8
GAP_UNIT = "tokens (no streaming timestamps)"

REPO = Path(__file__).resolve().parents[3]
SEALED_FILE = REPO / "outputs" / "stage1-sealed-FROZEN" / "taskset-sealed.jsonl"
SEALED_SHA256 = "e1e515b4201b3341c363d0cf1061356acf7775c3d4b5b5194afa9790223fc683"  # the sealed-freeze record
DEV_FILES = {  # pilot-30: the DEV set
    "taskset-pilot-20.jsonl": "6db74657a50d59be229f7ea81e5a55170b5a76d8856eb23797540822996b967b",
    "taskset-review-10.jsonl": "b97652353deb1d674d1542dbf4b5fe582c8934297153c87cbfada71760ea7771",
}
DEV_DIR = Path(__file__).resolve().parents[1] / "stage1-taskset"
INTERVIEW_FILE = Path(__file__).resolve().parents[1] / "stage1-eval" / "interview.py"
INTERVIEW_QUESTIONS_SHA256_PREFIX = "b59927e1"


class Refused(Exception):
    """A row the firewall won't write. `code` is stable; the message never repeats held-out text."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code, self.detail = code, detail


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def words(text: str) -> list:
    return re.findall(r"[a-z0-9']+", text.lower())


def spans(text: str, n: int = SPAN) -> set:
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def strings(x):
    """Every string in a JSON value, at any depth (dict keys excluded)."""
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from strings(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from strings(v)


def ids_in(x):
    """Every string value under a key ending in "id" (id, question_id, base_item_id, item_id, ...)."""
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "id" or k.endswith("_id"):
                if isinstance(v, str):
                    yield v
            yield from ids_in(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from ids_in(v)


def interview_questions(path: Path = INTERVIEW_FILE) -> list:
    """The 12 pre-interview questions, read from interview.py's QUESTIONS assignment without importing it (which
    would load the harness and the student), and checked against the pinned sha256 prefix."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "QUESTIONS" for t in node.targets):
            qs = ast.literal_eval(node.value)
            got = hashlib.sha256(json.dumps(qs).encode()).hexdigest()
            if not got.startswith(INTERVIEW_QUESTIONS_SHA256_PREFIX):
                raise SystemExit(f"interview QUESTIONS sha256 {got[:8]} != pinned {INTERVIEW_QUESTIONS_SHA256_PREFIX}")
            return qs
    raise SystemExit("QUESTIONS not found in interview.py")


class Firewall:
    """Held-out ids and 8-word spans, in memory only. Build it with Firewall.load(); tests build it from planted
    files with their own hashes."""

    def __init__(self, held_ids: set, held_spans: set, sources: dict):
        self.held_ids, self.held_spans, self.sources = held_ids, held_spans, sources

    @classmethod
    def load(cls, sealed: Path = SEALED_FILE, sealed_sha256: str = SEALED_SHA256, dev: dict | None = None,
             questions: list | None = None):
        dev = {DEV_DIR / k: v for k, v in DEV_FILES.items()} if dev is None else dev
        files = {Path(sealed): sealed_sha256, **{Path(k): v for k, v in dev.items()}}
        held_ids, held_spans, sources = set(), set(), {}
        for p, want in files.items():
            if not p.exists():
                raise SystemExit(f"firewall: held-out file missing: {p.name} (the receipts can't be written unarmed)")
            got = sha256_file(p)
            if got != want:
                raise SystemExit(f"firewall: {p.name} sha256 {got[:8]} != frozen {want[:8]}")
            sources[p.name] = got
            for line in p.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                held_ids.update(ids_in(rec))
                for s in strings(rec):
                    held_spans |= spans(s)
        qs = interview_questions() if questions is None else questions
        for q in qs:
            held_spans |= spans(q)
        sources["interview.QUESTIONS"] = hashlib.sha256(json.dumps(qs).encode()).hexdigest()
        return cls(held_ids, held_spans, sources)

    def check(self, row: dict):
        if row.get("source") not in SOURCES:
            raise Refused("SOURCE", f"source {row.get('source')!r} is not one of {', '.join(SOURCES)}")
        if row.get("arm") not in ARMS:
            raise Refused("ARM", f"arm {row.get('arm')!r} is not ROLE or NO-ROLE")
        for k in EVAL_ONLY_KEYS:
            if k in row:
                raise Refused("EVAL_RECORD", f"field {k!r} belongs to an evaluation sitting")
        for i in ids_in(row):
            if i.startswith(REFUSED_ID_PREFIXES):
                raise Refused("HELD_OUT_ID", f"id prefix {i[:3]!r} (sealed or DEV)")
            if i in self.held_ids:
                raise Refused("HELD_OUT_ID", "an id from the sealed or DEV files")
        for s in strings(row):
            if spans(s) & self.held_spans:
                raise Refused("HELD_OUT_TEXT", "text shares an 8-word span with the sealed set, DEV or the interview")


class ReceiptWriter:
    def __init__(self, root: Path, round: int, arm: str, seed: int, run_id: str, adapter_sha256: str | None,
                 gates_sha256: str, stuck_rule_sha256: str, firewall: Firewall):
        if type(round) is not int or type(seed) is not int:
            raise ValueError("round and seed are plain integers")
        if arm not in ARMS:
            raise ValueError(f"arm must be ROLE or NO-ROLE, not {arm!r}")
        if not isinstance(firewall, Firewall):
            raise ValueError("a receipt writer needs an armed firewall")
        self.firewall = firewall
        self.header = {"schema": SCHEMA, "source": None, "round": round, "seed": seed, "arm": arm, "run_id": run_id,
                       "adapter_sha256": adapter_sha256, "gates_sha256": gates_sha256,
                       "stuck_rule_sha256": stuck_rule_sha256, "written_at": None}
        self.dir = Path(root) / str(round) / arm / str(seed) / "receipts"
        self.refused: dict = {}

    def write(self, source: str, fields: dict) -> dict:
        """Write one row; raises Refused (and writes nothing) when the firewall refuses it. Refusals are counted
        by code in self.refused, for the round's diagnostics."""
        clash = set(fields) & set(self.header)
        if clash:
            raise ValueError(f"fields overlap the header: {sorted(clash)}")
        row = {**self.header, "source": source,
               "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **fields}
        try:
            self.firewall.check(row)
        except Refused as e:
            self.refused[e.code] = self.refused.get(e.code, 0) + 1
            raise
        self.dir.mkdir(parents=True, exist_ok=True)
        with open(self.dir / f"{source}.jsonl", "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

