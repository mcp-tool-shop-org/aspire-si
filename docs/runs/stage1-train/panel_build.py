"""Build the Stage 1 general-capability panel item file (design §1 and §2).

The two source files are local. Each one's sha256 is checked against the pin of the file at
the commit the design names. The design pinned those commits, not these hashes. Fetching is
optional. The item file, the GSM8K notice and the WinoGrande attribution are written outside
the repository. Nothing here commits an item.

Without a frozen corpus the walk is hash order, gold and the within-panel 8-word rule. The
meta line then says leakage was not run, and the file is not the final pin.
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_PKG = HERE.parents[2] / "packages" / "aspire-si-gates" / "src"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
import panel_leak  # noqa: E402
import panel_score  # noqa: E402
import receipts  # noqa: E402
from aspire_si_gates import gates  # noqa: E402

ORDER = "panel-b3-v1"
PER_SOURCE = 140
GSM8K_COMMIT = "3101c7d5072418e28b9008a6636bde82a006892c"
WINOGRANDE_COMMIT = "727e837f77521ef38bcc56df3b275c8da43f45af"
# Raw bytes of the named files at those commits. Not a hash the design recorded.
GSM8K_SHA256 = "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
WINOGRANDE_SHA256 = "1aeac79cf46a3dcbe59a71ca7e5372a1490cde2b4ae2387dc56f00378104d2e8"
GSM8K_URL = ("https://raw.githubusercontent.com/openai/grade-school-math/"
             f"{GSM8K_COMMIT}/grade_school_math/data/test.jsonl")
WINOGRANDE_URL = "https://storage.googleapis.com/ai2-mosaic/public/winogrande/winogrande_1.1.zip"

GSM_TEMPLATE = (
    "Solve the problem. Think it through, then end your reply with one line: "
    "ANSWER: <the final integer>. No units and no commas.\n\n")
WIN_TEMPLATE = (
    "The sentence has a blank, shown as an underscore. Choose the option that fills the blank. "
    "Think it through, then end your reply with one line: ANSWER: A or ANSWER: B.\n\n")

_GSM_GOLD = re.compile(r"####\s*(-?\d[\d,]*)")
_GSM_ID = re.compile(r"^gsm8k-test-[0-9]+$")
_WIN_ID = re.compile(r"^winogrande-[0-9A-Z-]{32}$")

# The LICENSE file at the GSM8K commit named in the design. Written beside the item file, not committed.
GSM8K_NOTICE = """MIT License

Copyright (c) 2021 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

WINOGRANDE_NOTICE = f"""WinoGrande attribution

Creators: Sakaguchi, Le Bras, Bhagavatula, Choi; AllenAI, 2019.
https://github.com/allenai/winogrande

Licence, from the README at commit {WINOGRANDE_COMMIT}:
Winogrande (codebase) is licensed under the Apache License 2.0. The dataset is licensed under CC-BY.

The dataset licence is CC-BY, version not stated by AllenAI.

Modifications: reformatted into prompts; options relabelled A and B.
"""


def sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def order_key(source: str, item_id: str) -> str:
    return hashlib.sha256(f"{ORDER}\n{source}\n{item_id}".encode()).hexdigest()


def prompt_sha256() -> dict:
    return {"gsm8k": hashlib.sha256(GSM_TEMPLATE.encode()).hexdigest(),
            "winogrande": hashlib.sha256(WIN_TEMPLATE.encode()).hexdigest()}


def outside_repo(path: Path) -> None:
    """Refuse a path that is the repository or inside it."""
    repo = receipts.REPO.resolve()
    resolved = path.resolve()
    if resolved == repo or repo in resolved.parents:
        raise SystemExit("refusing a path inside the repository")


def gsm_gold(answer: str) -> str | None:
    """The last #### integer, commas removed. The answer must end on that integer."""
    text = (answer or "").rstrip()
    found = list(_GSM_GOLD.finditer(text))
    if not found or found[-1].end() != len(text):
        return None
    raw = found[-1].group(1).replace(",", "")
    if not re.fullmatch(r"-?\d+", raw):
        return None
    return str(int(raw))


def win_gold(answer: str) -> str | None:
    return {"1": "A", "2": "B"}.get(str(answer))


def gsm_prompt(question: str) -> str:
    return GSM_TEMPLATE + question


def win_prompt(sentence: str, option1: str, option2: str) -> str:
    return f"{WIN_TEMPLATE}Sentence: {sentence}\nA. {option1}\nB. {option2}"


def load_gsm8k(path: Path) -> list[dict]:
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        obj = json.loads(line)
        item_id = f"gsm8k-test-{index}"
        question = obj.get("question") or ""
        rows.append({"source": "gsm8k", "id": item_id, "key": order_key("gsm8k", item_id),
                     "gold": gsm_gold(obj.get("answer") or ""), "source_text": question,
                     "prompt": gsm_prompt(question)})
    return rows


def load_winogrande(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        item_id = f"winogrande-{obj.get('qID')}"
        sentence = obj.get("sentence") or ""
        option1 = obj.get("option1") or ""
        option2 = obj.get("option2") or ""
        rows.append({"source": "winogrande", "id": item_id, "key": order_key("winogrande", item_id),
                     "gold": win_gold(obj.get("answer")),
                     "source_text": f"{sentence} {option1} {option2}",
                     "prompt": win_prompt(sentence, option1, option2)})
    return rows


def walk(rows: list[dict], per_source: int, blocked: dict[str, str], reject) -> list[dict]:
    """Hash order. Stop at `per_source` kept rows. `blocked` maps an 8-word span to the kept id."""
    looked = []
    kept = 0
    for row in sorted(rows, key=lambda item: item["key"]):
        if kept >= per_source:
            break
        if row["gold"] is None:
            looked.append(_skipped(row, "gold"))
            continue
        spans = gates._shingles(row["source_text"], 8)
        other = next((blocked[span] for span in spans if span in blocked), None)
        if other is not None:
            looked.append(_skipped(row, f"dedupe:{other}"))
            continue
        if reject is not None:
            reason = reject(row)
            if reason:
                looked.append(_skipped(row, reason))
                continue
        looked.append(_kept(row))
        for span in spans:
            blocked.setdefault(span, row["id"])
        kept += 1
    return looked


def _skipped(row: dict, reason: str) -> dict:
    return {"record": "row", "source": row["source"], "id": row["id"], "key": row["key"],
            "decision": "skipped", "reason": reason}


def _kept(row: dict) -> dict:
    return {"record": "row", "source": row["source"], "id": row["id"], "key": row["key"],
            "decision": "kept", "reason": "kept", "slice": row["source"], "gold": row["gold"],
            "prompt": row["prompt"], "source_text": row["source_text"]}


def check_kept(kept: list[dict]) -> None:
    """Ids, the within-panel 8-word rule, and every gold line scoring right."""
    seen = set()
    spans: dict[str, str] = {}
    for row in kept:
        ident = row["id"]
        pattern = _GSM_ID if row["source"] == "gsm8k" else _WIN_ID
        if pattern.fullmatch(ident) is None or ident in seen:
            raise SystemExit("refusing to write: an item id is missing, duplicated, or the wrong form")
        seen.add(ident)
        own = gates._shingles(row["source_text"], 8)
        if own & spans.keys():
            raise SystemExit("refusing to write: an 8-word span is shared inside the panel")
        spans.update(dict.fromkeys(own, ident))
        gold_text = panel_score.gold_reply(row["gold"])
        scored = panel_score.score_generation(
            {"text": gold_text, "truncated": False}, row["gold"], row["source"])
        if not scored["right"]:
            raise SystemExit("refusing to write: a gold answer does not score right")


def render(meta: dict, rows: list[dict]) -> str:
    lines = [json.dumps(meta, ensure_ascii=False)]
    lines.extend(json.dumps(row, ensure_ascii=False) for row in rows)
    return "\n".join(lines) + "\n"


def build(gsm_path: Path, wino_path: Path, out_path: Path, *, per_source: int = PER_SOURCE,
          gsm_sha: str = GSM8K_SHA256, wino_sha: str = WINOGRANDE_SHA256, reject=None,
          leakage: str = "not run") -> str:
    """Write the item file and the two notices. Return the item file's sha256."""
    outside_repo(out_path)
    if sha256_bytes(gsm_path) != gsm_sha:
        raise SystemExit("gsm8k sha256 does not match the pin")
    if sha256_bytes(wino_path) != wino_sha:
        raise SystemExit("winogrande sha256 does not match the pin")
    gsm_rows, wino_rows = load_gsm8k(gsm_path), load_winogrande(wino_path)
    blocked: dict[str, str] = {}
    looked = walk(gsm_rows, per_source, blocked, reject)
    looked.extend(walk(wino_rows, per_source, blocked, reject))
    kept = [row for row in looked if row["decision"] == "kept"]
    counts = {"gsm8k": sum(1 for row in kept if row["source"] == "gsm8k"),
              "winogrande": sum(1 for row in kept if row["source"] == "winogrande")}
    if counts["gsm8k"] != per_source or counts["winogrande"] != per_source:
        raise SystemExit(f"refusing to write: kept gsm8k {counts['gsm8k']}, "
                         f"winogrande {counts['winogrande']}")
    check_kept(kept)
    meta = {"record": "meta", "order": ORDER, "leakage": leakage, "per_source": per_source,
            "prompt_sha256": prompt_sha256(),
            "gsm8k": {"commit": GSM8K_COMMIT, "sha256": gsm_sha, "rows": len(gsm_rows)},
            "winogrande": {"commit": WINOGRANDE_COMMIT, "sha256": wino_sha, "rows": len(wino_rows)},
            "kept": counts}
    text = render(meta, looked)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write(out_path, text)
    _write(out_path.parent / "GSM8K-MIT-LICENSE.txt", GSM8K_NOTICE)
    _write(out_path.parent / "WINOGRANDE-ATTRIBUTION.txt", WINOGRANDE_NOTICE)
    return sha256_bytes(out_path)


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def fetch_into(dest: Path, opener=urllib.request.urlopen) -> dict[str, Path]:
    """Download the two source files. The directory must be outside the repository."""
    outside_repo(dest)
    dest.mkdir(parents=True, exist_ok=True)
    gsm_path = dest / "gsm8k-test.jsonl"
    with opener(GSM8K_URL) as response, gsm_path.open("wb") as handle:
        handle.write(response.read())
    zip_path = dest / "winogrande_1.1.zip"
    with opener(WINOGRANDE_URL) as response, zip_path.open("wb") as handle:
        handle.write(response.read())
    wino_path = dest / "winogrande-dev.jsonl"
    with zipfile.ZipFile(zip_path) as archive:
        name = next(info.filename for info in archive.infolist() if info.filename.endswith("dev.jsonl"))
        wino_path.write_bytes(archive.read(name))
    return {"gsm8k": gsm_path, "winogrande": wino_path}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the panel item file outside the repository.")
    parser.add_argument("--gsm8k")
    parser.add_argument("--winogrande")
    parser.add_argument("--out")
    parser.add_argument("--fetch-dir", help="download both source files into this outside-repo directory")
    parser.add_argument("--per-source", type=int, default=PER_SOURCE)
    parser.add_argument("--frozen", help="FROZEN.md. When omitted, leakage is not run and the file says so.")
    args = parser.parse_args(argv)
    if args.fetch_dir:
        fetched = fetch_into(Path(args.fetch_dir))
        gsm_path, wino_path = fetched["gsm8k"], fetched["winogrande"]
    else:
        if not args.gsm8k or not args.winogrande:
            parser.error("--gsm8k and --winogrande are required unless --fetch-dir is set")
        gsm_path, wino_path = Path(args.gsm8k), Path(args.winogrande)
    if not args.out:
        parser.error("--out is required")
    reject = None
    leakage = "not run"
    if args.frozen:
        frozen = Path(args.frozen)
        if not frozen.is_file():
            print("FROZEN.md is not here; the item file was not written", file=sys.stderr)
            return 2
        corpus = panel_leak.corpus_from_frozen(frozen)

        def reject(row, corpus=corpus):
            return panel_leak.reject_reason(row, corpus)

        leakage = "run"
    try:
        digest = build(gsm_path, wino_path, Path(args.out), per_source=args.per_source, reject=reject,
                       leakage=leakage)
    except panel_leak.LeakError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"item sha256: {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
