"""Stage 1 freeze: one orchestrator for the checks, then FROZEN.md.

    python docs/runs/stage1-train/freeze.py check --data <files> --out <dir>
        [--apply <files>] [--partial-set] [--sealed <path>]
    python docs/runs/stage1-train/freeze.py write --report <dir>/freeze-report.json
        --out docs/runs/stage1-training-data/FROZEN.md [--no-role-out <path>]

`check` is read-only on the data and writes only under `--out`. It calls the checks that already exist.
It does not reimplement them:

- `validate_training_set` (what `train.py` calls), which itself calls `freeze_report` (what
  `data.py --freeze` runs) and, per batch, `check_puzzles` and `framing_repeats` (what `data.py --batch`
  runs). `checks.py` in the training-data directory has no `--batch` flag. Its per-file quote gate stays
  the writer's gate.
- `no_role.build` and `no_role.composition`, with the same token count `no_role.py`'s CLI uses.
- `gates.leakage`, against the sealed task file, the DEV files and `interview.QUESTIONS`, loaded the way
  `train.py` loads them.
- `quiz.variant_checked_spans` against `quiz.trained_spans`.
- `apply.load_items`, then the apply dry run (`make_twin`, `set_rule`, `score_three_lines`, `single`,
  `any_item`, and the frozen stuck rule). A None from `make_twin` is a predicted hold. The three holds the
  runtime is expected to make are reported and do not fail. Any other None fails the apply check.

`--partial-set` is the batches-1-to-6 run: `require_freeze=False`, and `--apply` may be omitted. It is
recorded on the report. `write` refuses that report, any failed check, any freeze shortfall, a missing or
incomplete apply set, and any sealed sha other than the pinned `e1e515b4...`. A test or a pre-freeze
report cannot write FROZEN.md.

The pinned NO-ROLE jsonl for round 1 (what `train.py` and `apply.py star --training-no-role` read) is
`docs/runs/stage1-no-role/round1.jsonl`. R&D approved that path. It sits outside `stage1-training-data/`
so a glob of that directory cannot pick it up as curriculum. `write` copies it there.

Paths in the report and in FROZEN.md are repo-relative POSIX. Both files use LF, so their sha256 does not
depend on the system. The sealed file is the task file `receipts.SEALED_FILE` only. A sealed output
directory is never read. CPU only.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]  # stage1-train -> runs -> docs -> the repo root
sys.path.insert(0, str(HERE))

import apply  # noqa: E402
import apply_rules  # noqa: E402
import data  # noqa: E402
import no_role  # noqa: E402
import quiz  # noqa: E402
import receipts  # noqa: E402
from aspire_si_gates import GATES_VERSION, gates  # noqa: E402

# Approved by R&D. It is the file the round reads, not a scratch copy.
NO_ROLE_PIN = "docs/runs/stage1-no-role/round1.jsonl"
NO_ROLE_NAME = "no-role.jsonl"
REPORT_NAME = "freeze-report.json"
CHECKS = ("training_order", "validate_training_set", "no_role", "leakage", "spent_variants", "apply")
APPLY_TOTAL = 210  # 21 lessons x (5 C + 5 D)
_FILE = re.compile(r"^(pilot|batch([2-8]))-([tl])(\d+)\.jsonl$")
_DRIVE = re.compile(r"[A-Za-z]:[\\/]")


class OrderError(Exception):
    pass


class CountError(Exception):
    pass


def scrub(text) -> str:
    """Drop machine paths from a message that will be stored in the report."""
    s = str(text).replace("\\", "/")
    root = str(REPO.resolve()).replace("\\", "/")
    if root and root in s:
        s = s.replace(root + "/", "").replace(root, "")
    s = re.sub(r"/Users/[^\s'\"]+", "<path>", s)
    s = re.sub(r"/home/[^\s'\"]+", "<path>", s)
    s = _DRIVE.sub("<path>", s)
    return s


def public_path(path: Path) -> str:
    """A repo-relative POSIX path, or a bare name when the file is a planted one outside the repo."""
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return "<outside-repo>/" + path.name


def cli_count():
    """The token count `no_role.py`'s CLI uses, so the NO-ROLE file's sha256 equals that CLI's."""
    sys.path.insert(0, str(data.EVAL_DIR))
    import student
    tok = student.load_tokenizer()

    def count(text):
        return len(tok(text, add_special_tokens=False)["input_ids"])

    return count


def load_count():
    try:
        return cli_count()
    except SystemExit as e:
        raise CountError(scrub(e) or "tokenizer: not available") from e


def training_key(path: Path):
    """(batch, letter, number) and the lesson id. pilot is batch 1 and comes before batch2."""
    name = path.name
    if name.endswith(".models.jsonl"):
        raise OrderError(f"{name}: a .models.jsonl sidecar is not a training file")
    m = _FILE.match(name)
    if not m:
        raise OrderError(f"{name}: not a training file (pilot, or batch2 to batch8, then the lesson)")
    batch = 1 if m.group(1) == "pilot" else int(m.group(2))
    letter = m.group(3).upper()
    number = int(m.group(4))
    lesson = f"T{number:02d}" if letter == "T" else f"L{number}"
    return (batch, 0 if letter == "T" else 1, number), lesson


def training_order(paths):
    """The given order must already be pilot, then batch2 to batch8, lessons ascending inside a batch.

    A shell glob sorts batch* before pilot. That order is refused. This does not sort the files.
    """
    parsed = []
    for p in paths:
        key, lesson = training_key(p)
        parsed.append((p, key, lesson))
    keys = [k for _, k, _ in parsed]
    if keys != sorted(keys) or len(keys) != len(set(keys)):
        raise OrderError(
            "files are not in training order: pilot first, then batch2 to batch8, in lesson order "
            "within a batch (a shell glob sorts batch* before pilot)")
    loaded = []
    for p, _key, lesson in parsed:
        if not p.is_file():
            raise OrderError(f"{p.name}: not a file")
        try:
            rows = data.load(p)
        except (OSError, json.JSONDecodeError) as e:
            raise OrderError(f"{p.name}: {scrub(e)}") from e
        got = {r.get("lesson") for r in rows}
        if got != {lesson}:
            raise OrderError(f"{p.name}: file lesson {lesson} but rows are {sorted(map(str, got))}")
        loaded.append((p, lesson, rows))
    return loaded


def item_cells(r: dict) -> dict:
    """The composition cells one item increments. The totals come from `no_role.composition`."""
    c = {"items": 1, "sequences": len(r.get("turns") or [])}
    c[f"tier {r.get('tier')}"] = 1
    keys = data._verdict_keys(r)
    if isinstance(keys, dict):
        c[f"verdict {keys.get('verdict')}"] = 1
        if (keys.get("escalate") or "").startswith("yes ("):
            c["escalate yes"] = 1
        deciding = keys.get("deciding") or []
        if 2 <= len(deciding) <= 4:
            c["multi-line"] = 1
        if not deciding:
            c["NONE"] = 1
        if r.get("pair") is not None:
            c["paired"] = 1
        if r.get("kind") == "D" and len(r.get("turns") or []) >= 2:
            first = (r.get("turn_keys") or {}).get("1", {}).get("verdict")
            name = "two-turn changed" if first != keys.get("verdict") else "two-turn stood"
            c[name] = 1
    return c


def composition_gate(role, nr, count) -> dict:
    """±1 on the items and sequences cells only. Every other count cell must match.

    Token totals are a ratio, not a gate.
    """
    a, b = no_role.composition(role, count), no_role.composition(nr, count)
    diffs = []
    failed = False
    for key in sorted(set(a) | set(b)):
        for cell in sorted((set(a[key]) | set(b[key])) - {"target_tokens"}):
            va, vb = a[key][cell], b[key][cell]
            if va == vb:
                continue
            within = abs(va - vb) <= (1 if cell in ("items", "sequences") else 0)
            ids = []
            for ra, rb in zip(role, nr):
                if (ra.get("lesson"), ra.get("kind")) != key and (rb.get("lesson"), rb.get("kind")) != key:
                    continue
                if item_cells(ra).get(cell, 0) != item_cells(rb).get(cell, 0):
                    ids.append(ra.get("id"))
            if not ids:
                ids = [r.get("id") for r in role if (r.get("lesson"), r.get("kind")) == key]
            diffs.append({"lesson": key[0], "kind": key[1], "cell": cell, "role": va, "no_role": vb,
                          "item_ids": ids, "within_tolerance": within})
            failed = failed or not within
    if len(role) != len(nr):
        failed = True
        extra = [r.get("id") for r in role[len(nr):]] + [r.get("id") for r in nr[len(role):]]
        diffs.append({"lesson": None, "kind": None, "cell": "length", "role": len(role), "no_role": len(nr),
                      "item_ids": extra, "within_tolerance": False})
    tot_a = sum(a[k]["target_tokens"] for k in a)
    tot_b = sum(b[k]["target_tokens"] for k in b)
    ratio = (tot_b / tot_a) if tot_a else 1.0
    return {"ok": not failed, "differences": diffs,
            "target_token_ratio": f"{ratio:.6f}", "role_target_tokens": tot_a,
            "no_role_target_tokens": tot_b, "build_error": None}


def write_jsonl(path: Path, rows) -> None:
    """The same bytes `no_role.py` writes: UTF-8, LF, one JSON object per line."""
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def no_role_check(role, count, out_dir: Path) -> tuple:
    """Returns (check, rows or None). A BuildError is this check failing, not a crash."""
    try:
        nr, _exposure = no_role.build(role, count)
    except no_role.BuildError as e:
        return {"ok": False, "build_error": scrub(e), "differences": [], "target_token_ratio": None,
                "role_target_tokens": None, "no_role_target_tokens": None}, None
    gate = composition_gate(role, nr, count)
    path = out_dir / NO_ROLE_NAME
    write_jsonl(path, nr)
    gate["sha256"] = receipts.sha256_file(path)
    gate["items"] = len(nr)
    return gate, nr


def spent_hits(rows, arm: str) -> list:
    """Testable kind-E variants (n = 0) whose checked spans meet a trained span. Free text is not testable."""
    if not rows:
        return []
    trained = quiz.trained_spans(rows, quiz.SPENT_RULE_DEFAULT["span"])
    hits = []
    for r in rows:
        if r.get("kind") != "E" or not isinstance(r.get("variant"), dict):
            continue
        if "quiz" not in r:
            hits.append({"id": r.get("id"), "arm": arm, "n": 0, "reason": "no quiz"})
            continue
        kind = quiz.quiz_type(r, r["variant"])
        if kind == "free_text":
            continue
        if quiz.variant_checked_spans(r, r["variant"], quiz.SPENT_RULE_DEFAULT) & trained:
            hits.append({"id": r.get("id"), "arm": arm, "n": 0, "reason": "spent"})
    return hits


def load_questions():
    """`interview.QUESTIONS`, the import `train.py` uses."""
    sys.path.insert(0, str(data.EVAL_DIR))
    from interview import QUESTIONS
    return list(QUESTIONS)


def load_held(sealed_path: Path, partial: bool):
    """The sealed task file plus the DEV files.

    A missing sealed file is reported, and it fails a real freeze.
    """
    info = {"status": "not available", "sha256": None, "matches_pin": False, "path": public_path(sealed_path)}
    problems = []
    held = []
    # A directory is never read, including a sealed output directory.
    if sealed_path.is_file():
        info["sha256"] = receipts.sha256_file(sealed_path)
        info["matches_pin"] = info["sha256"] == receipts.SEALED_SHA256
        info["status"] = "ok"
        try:
            held.extend(data.load(sealed_path))
        except (OSError, json.JSONDecodeError) as e:
            problems.append(f"sealed: {scrub(e)}")
        if not partial and not info["matches_pin"]:
            problems.append(f"sealed: sha256 {info['sha256'][:8]} is not {receipts.SEALED_SHA256[:8]}")
    elif not partial:
        problems.append("sealed: not available")
    for name in sorted(receipts.DEV_FILES):
        path = receipts.DEV_DIR / name
        want = receipts.DEV_FILES[name]
        if not path.is_file():
            problems.append(f"dev: {name} not available")
            continue
        got = receipts.sha256_file(path)
        if got != want:
            problems.append(f"dev: {name} sha256 {got[:8]} is not {want[:8]}")
            continue
        held.extend(data.load(path))
    return info, held, problems


def leakage_check(role, nr, apply_rows, sealed_path: Path, partial: bool) -> tuple:
    info, held, problems = load_held(sealed_path, partial)
    try:
        questions = load_questions()
    except (SystemExit, OSError, SyntaxError) as e:
        return {"ok": False, "hits": problems + [f"interview: {scrub(e)}"]}, info
    groups = [("training", role), ("no-role", nr or []), ("apply", apply_rows or [])]
    hits = list(problems)
    for name, rows in groups:
        if not rows:
            continue
        try:
            found = gates.leakage(rows, held, questions)
        except (KeyError, TypeError) as e:
            hits.append(f"{name}: {scrub(e)}")
            continue
        hits.extend(f"{name}: {h}" for h in found)
    return {"ok": not hits, "hits": hits}, info


def hold_reason(r: dict) -> tuple:
    """The make_twin branch that returned None, and whether that hold is one the runtime expects.

    The three expected holds are reported and do not fail. Each reason names the branch it mirrors,
    in make_twin's order. Any other None fails the apply check.
    """
    lesson = r.get("lesson")
    rule = apply_rules.TWIN[lesson]
    if rule == "drop_deciding":
        segs = [s for s in (r.get("deciding") or []) if s and str(s).upper() != "NONE"]
        if r.get("verdict") not in ("supported", "unsupported"):
            return "drop_deciding: not keyed supported or unsupported", True
        if not segs:
            return "drop_deciding: no deciding segment", False
        # `is not True` is make_twin's branch. Only the false value is the expected hold.
        if lesson in apply_rules.SILENT_NEEDED and r.get("silent_without_deciding") is not True:
            if r.get("silent_without_deciding") is False:
                return "drop_deciding: silent_without_deciding is false", True
            return "drop_deciding: silent_without_deciding is not true", False
        return "drop_deciding: a deciding segment is not on a material line", False
    if rule == "permute":
        if r.get("order_free") is False:
            return "permute: order_free is false", True
        if r.get("order_free") is not True:
            return "permute: order_free is not true", False
        lines = apply_rules._user_lines(r)
        idx = apply_rules._material_line_idx(r, lines)
        if len(idx) < 2:
            return "permute: fewer than two material lines", False
        new = list(lines)
        for i, j in zip(idx, reversed(idx)):
            new[i] = lines[j]
        if new == lines:
            return "permute: the reversed lines equal the original", False
        return "permute: make_twin returned None", False
    if rule == "inject":
        return "inject: no material line", False
    if rule == "r10":
        return "r10: malformed r10_twin", False
    return "make_twin returned None", False


def apply_report(ok, *, given, skipped, complete, count, problems, holds=()) -> dict:
    expected = [h for h in holds if h["expected"]]
    return {"ok": ok, "given": given, "skipped": skipped, "complete": complete, "count": count,
            "problems": list(problems), "predicted_holds": list(holds),
            "predicted_holds_by_lesson": dict(Counter(h["lesson"] for h in expected))}


def apply_complete(items) -> bool:
    if len(items) != APPLY_TOTAL:
        return False
    per = {}
    for r in items:
        per.setdefault(r.get("lesson"), Counter())
        per[r.get("lesson")][r.get("kind")] += 1
    if set(per) != set(data.LESSONS):
        return False
    return all(per[lesson]["C"] == 5 and per[lesson]["D"] == 5 for lesson in per)


def apply_check(paths, train_rows, count, partial: bool) -> dict:
    """`load_items`, then the dry run. Omitted apply files are allowed only with `--partial-set`."""
    if not paths:
        if partial:
            return apply_report(True, given=False, skipped=True, complete=False, count=0, problems=[])
        return apply_report(False, given=False, skipped=False, complete=False, count=0,
                            problems=["apply items were not given"])
    try:
        items = apply.load_items(paths)
    except apply.ApplyItemError as e:
        return apply_report(False, given=True, skipped=False, complete=False, count=None,
                            problems=[scrub(e)])
    problems = []
    holds = []
    # A real freeze must not pass the stuck rule and the set rules by skipping them.
    if count is None and not partial:
        problems.append("stuck and set rules not run: no token count")
    train_ids = {r.get("id") for r in train_rows}
    trained = quiz.trained_spans(train_rows, quiz.SPENT_RULE_DEFAULT["span"])
    span_n = quiz.SPENT_RULE_DEFAULT["span"]
    for r in items:
        if r["id"] in train_ids:
            problems.append(f"{r['id']}: id collides with the training union")
        if receipts.spans(r["turns"][0]["user"], span_n) & trained:
            problems.append(f"{r['id']}: an {span_n}-word span of the user turn collides with training")
    rule = None
    try:
        import stuck_adapter
        rule = stuck_adapter.load_frozen()
    except SystemExit as e:
        problems.append(f"stuck rule: {scrub(e)}")
    for r in items:
        try:
            reply = r["turns"][0]["reply"]
            thinking = r["turns"][0].get("thinking") or ""
            if not quiz.score_three_lines(apply.key_of(r), r.get("material") or "", reply)["right"]:
                problems.append(f"{r['id']}: the reference answer's three lines do not pass")
            parsed = quiz.harness.parse_answer(reply)
            graded = quiz.harness.grade({"key": apply.key_of(r), "material": r.get("material") or ""}, parsed)
            ok_any, why_any = apply_rules.any_item(graded["deciding_verbatim"], thinking, False)
            if not ok_any:
                problems.append(f"{r['id']}: {why_any}")
            ok_single, why_single = apply_rules.single(r["lesson"], reply, r["kind"])
            if not ok_single:
                problems.append(f"{r['id']}: {why_single}")
            if rule is not None and count is not None:
                target = count(thinking)
                fields = quiz.stuck_fields(thinking, r.get("material") or "", r["tier"], target,
                                           target + count(reply), count, rule)
                if fields["stuck"]:
                    problems.append(f"{r['id']}: the reference answer is stuck")
            lesson = r.get("lesson")
            if lesson in apply_rules.TWIN and apply_rules.twin_applies(r, lesson):
                if apply_rules.make_twin(r, apply_rules.TWIN[lesson], reply) is None:
                    reason, expected = hold_reason(r)
                    holds.append({"id": r["id"], "lesson": lesson, "reason": reason, "expected": expected})
                    if not expected:
                        problems.append(f"{r['id']}: {reason}")
        except (KeyError, TypeError, ValueError) as e:
            problems.append(f"{r.get('id', '?')}: {scrub(e)}")
    if rule is not None and count is not None:
        for lesson in sorted(apply_rules.SET):
            rows = [r for r in items if r.get("lesson") == lesson]
            if not rows:
                if not partial:
                    problems.append(f"{lesson}: the set rule has no items")
                continue
            if lesson == "T07":
                spec = [{"id": r["id"], "tier": r["tier"],
                         "thinking_tokens": count(r["turns"][0].get("thinking") or "")} for r in rows]
            else:
                spec = [{"id": r["id"], "key_verdict": r["verdict"],
                         "given": quiz.harness.parse_answer(r["turns"][0]["reply"])["verdict"]} for r in rows]
            result = apply_rules.set_rule(lesson, spec, rule)
            passed = result.get("testable") and all(r["id"] in result.get("pass_ids", ()) for r in rows)
            if not passed:
                problems.append(f"{lesson}: set rule did not pass ({result.get('why') or 'not testable'})")
    complete = apply_complete(items)
    if not partial and not complete:
        problems.append(f"apply count {len(items)} is not 21 lessons x (5 C + 5 D) = {APPLY_TOTAL}")
    return apply_report(not problems, given=True, skipped=False, complete=complete,
                        count=len(items), problems=problems, holds=holds)


def counts_of(loaded) -> list:
    order, table = [], {}
    for _p, lesson, rows in loaded:
        if lesson not in table:
            order.append(lesson)
            table[lesson] = {k: 0 for k in "ABCDE"}
        for r in rows:
            if r.get("kind") in table[lesson]:
                table[lesson][r["kind"]] += 1
    return [{"lesson": lesson, **table[lesson]} for lesson in order]


def sources_of(rows) -> dict:
    real = set()
    invented = 0
    for r in rows:
        cited = False
        for s in r.get("sources") or []:
            if isinstance(s, dict) and s.get("repo"):
                cited = True
                licence = s.get("licence") or ""
                first = licence.splitlines()[0] if isinstance(licence, str) else ""
                real.add((s.get("repo"), s.get("commit"), s.get("path"), str(s.get("lines")), first))
        if not cited:
            invented += 1
    listed = [{"repo": a, "commit": b, "path": c, "lines": d, "licence": e} for a, b, c, d, e in sorted(real)]
    return {"real": listed, "invented_items": invented}


def validate_check(files, partial: bool) -> dict:
    import train
    try:
        body = data.validate_training_set(
            files, require_freeze=not partial, star_check=train.star_three_lines_ok)
    except data.DataError as e:
        return {"ok": False, "error": scrub(e), "freeze_shortfalls": [], "star_rows": None}
    short = []
    if body.get("freeze_refused"):
        short = [scrub(body["freeze_refused"])]
    elif body.get("shortfalls"):
        short = [scrub(s) for s in body["shortfalls"]]
    # A partial run reports shortfalls and still passes this check. A real freeze does not.
    ok = True if partial else not short
    return {"ok": ok, "error": None, "freeze_shortfalls": short, "star_rows": body.get("star_rows"),
            "freeze_report": {k: body[k] for k in body if k in (
                "verdict_bearing", "cannot_tell", "supported", "supported_or_unsupported", "multi_line",
                "lessons_without_escalate_yes", "exempt_without_escalate_yes", "repeated_framings",
                "shortfalls", "star_rows")}}


def file_record(path: Path, rows, lesson=None) -> dict:
    rec = {"path": public_path(path), "sha256": receipts.sha256_file(path), "items": len(rows)}
    if lesson is not None:
        rec["lesson"] = lesson
    return rec


def finish(report, out_dir: Path) -> int:
    report["failed"] = [name for name in CHECKS
                        if name in report["checks"] and not report["checks"][name]["ok"]]
    report["ok"] = not report["failed"] and "training_order" in report["checks"]
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    (out_dir / REPORT_NAME).write_bytes(text.encode("utf-8"))
    if report["ok"]:
        ratio = (report["checks"].get("no_role") or {}).get("target_token_ratio")
        short = (report["checks"].get("validate_training_set") or {}).get("freeze_shortfalls") or []
        print("freeze: pass")
        print(f"partial-set: {str(report['partial_set']).lower()}")
        print("freeze shortfalls: " + (short[0] if short else "none"))
        print(f"no-role target-token ratio: {ratio}")
        nr = report.get("no_role") or {}
        print(f"no-role sha256: {nr.get('sha256')}")
        sealed = report.get("sealed") or {}
        print(f"sealed: {sealed.get('status')} {sealed.get('sha256') or ''}".rstrip())
        holds = (report["checks"].get("apply") or {}).get("predicted_holds") or []
        print(f"predicted holds: {len(holds)}")
    else:
        print("freeze: fail: " + ", ".join(report["failed"]), file=sys.stderr)
    return 0 if report["ok"] else 1


def cmd_check(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    partial = bool(args.partial_set)
    report = {"partial_set": partial, "training_files": [], "apply_files": [], "sealed": None,
              "no_role": {"file": None, "sha256": None, "pin": NO_ROLE_PIN, "items": None},
              "counts": [], "sources": {"real": [], "invented_items": 0}, "checks": {}}
    try:
        loaded = training_order([Path(p) for p in args.data])
    except OrderError as e:
        report["checks"]["training_order"] = {"ok": False, "error": scrub(e)}
        return finish(report, out)
    report["checks"]["training_order"] = {"ok": True}
    report["training_files"] = [file_record(p, rows, lesson) for p, lesson, rows in loaded]
    report["counts"] = counts_of(loaded)
    role = [r for _p, _lesson, rows in loaded for r in rows]
    report["sources"] = sources_of(role)
    files = {str(p): rows for p, _lesson, rows in loaded}
    report["checks"]["validate_training_set"] = validate_check(files, partial)

    try:
        count = load_count()
    except CountError as e:
        report["checks"]["no_role"] = {"ok": False, "build_error": scrub(e), "differences": [],
                                       "target_token_ratio": None}
        count = None
        nr = None
    else:
        gate, nr = no_role_check(role, count, out)
        report["checks"]["no_role"] = gate
        if nr is not None:
            report["no_role"] = {"file": NO_ROLE_NAME, "sha256": gate.get("sha256"), "pin": NO_ROLE_PIN,
                                 "items": gate.get("items")}

    spent = spent_hits(role, "ROLE") + spent_hits(nr, "NO-ROLE")
    report["checks"]["spent_variants"] = {"ok": not spent, "spent": spent}

    apply_paths = [Path(p) for p in (args.apply or [])]
    report["apply_files"] = [{"path": public_path(p),
                              "sha256": receipts.sha256_file(p) if p.is_file() else None}
                             for p in apply_paths]
    leak, sealed = leakage_check(role, nr, None, Path(args.sealed), partial)
    # Apply rows join leakage once they load. A schema failure belongs to the apply check: leakage records
    # that those rows were not checked, and does not fail for that.
    apply_rows = None
    if apply_paths:
        try:
            apply_rows = apply.load_items(apply_paths)
        except apply.ApplyItemError:
            apply_rows = None
            leak["apply_rows_checked"] = False
        else:
            extra, _info = leakage_check((), (), apply_rows, Path(args.sealed), partial)
            # `extra` reloads the held-out sets. Keep the training result and add only the apply hits.
            leak["hits"] = list(leak["hits"]) + [h for h in extra["hits"] if h.startswith("apply:")]
            leak["ok"] = not leak["hits"]
            leak["apply_rows_checked"] = True
    report["checks"]["leakage"] = leak
    report["sealed"] = sealed
    report["checks"]["apply"] = apply_check(apply_paths, role, count, partial)
    if apply_rows is not None:
        for rec, rows in zip(report["apply_files"], _group_apply(apply_paths, apply_rows)):
            rec["items"] = len(rows)
    return finish(report, out)


def _group_apply(paths, items):
    """Split the loaded items back into the files they came from, in the given order."""
    counts = []
    for p in paths:
        n = sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip())
        counts.append(n)
    out, i = [], 0
    for n in counts:
        out.append(items[i:i + n])
        i += n
    return out


def refuse_reason(report: dict) -> str | None:
    """Why this report must not become FROZEN.md. None means it may."""
    if not isinstance(report, dict):
        return "the report is not an object"
    if report.get("partial_set") is not False:
        return "the report is a partial set"
    if report.get("ok") is not True or report.get("failed"):
        return "a check failed: " + ", ".join(report.get("failed") or ["unknown"])
    checks = report.get("checks") or {}
    for name in CHECKS:
        if name not in checks or not checks[name].get("ok"):
            return f"{name} did not pass"
    short = (checks.get("validate_training_set") or {}).get("freeze_shortfalls") or []
    if short:
        return "the freeze has shortfalls"
    apply_c = checks.get("apply") or {}
    if not apply_c.get("given") or not apply_c.get("complete"):
        return "the apply items were not given and complete"
    sealed = report.get("sealed") or {}
    if sealed.get("sha256") != receipts.SEALED_SHA256 or sealed.get("status") != "ok":
        return "the sealed file's sha256 is not the pinned file"
    if not (report.get("no_role") or {}).get("sha256") or not (report.get("no_role") or {}).get("file"):
        return "the report has no NO-ROLE file"
    return None


def _cell(value) -> str:
    return str(value).replace("|", "/").replace("\n", " ")


def render_frozen(report: dict) -> str:
    """The fixed FROZEN.md. Training files stay in the report's order, which is the training order.

    The header records the hash rule and the two versions a reader needs to reproduce every digest.
    It is computed at write time, so two writes of the same report stay byte-identical.
    """
    lines = ["# Stage 1 frozen data", "",
             "Each file's sha256 is the sha256 of the content with CRLF read as LF (`receipts.sha256_file`).",
             f"aspire-si-gates {GATES_VERSION}",
             f"freeze.py sha256: {receipts.sha256_file(Path(__file__))}",
             f"sealed pin: {receipts.SEALED_SHA256[:8]}",
             "",
             "Training order is the order of the rows below. Do not sort this list by path.", "",
             "## Training files", "",
             "| # | file | items | sha256 |",
             "|---:|---|---:|---|"]
    for i, f in enumerate(report.get("training_files") or [], 1):
        lines.append(f"| {i} | {_cell(f['path'])} | {f['items']} | {f['sha256']} |")
    lines += ["", "## Apply files", "",
              "| # | file | items | sha256 |",
              "|---:|---|---:|---|"]
    for i, f in enumerate(report.get("apply_files") or [], 1):
        lines.append(f"| {i} | {_cell(f['path'])} | {f.get('items', '')} | {f['sha256']} |")
    nr = report.get("no_role") or {}
    lines += ["", "## NO-ROLE", "",
              "| file | items | sha256 |",
              "|---|---:|---|",
              f"| {_cell(nr.get('pin') or NO_ROLE_PIN)} | {nr.get('items', '')} | {nr.get('sha256')} |",
              "", "## Items per lesson and kind", "",
              "| lesson | A | B | C | D | E |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in report.get("counts") or []:
        lines.append("| " + " | ".join(_cell(row[k]) for k in ("lesson", "A", "B", "C", "D", "E")) + " |")
    ratio = (report.get("checks") or {}).get("no_role", {}).get("target_token_ratio")
    lines += ["", "## Gates", "",
              "| check | result |",
              "|---|---|"]
    for name in CHECKS:
        lines.append(f"| {name} | pass |")
    lines += ["", f"target-token ratio (NO-ROLE / ROLE): {ratio}", "",
              "## Sources", "",
              "| repo | commit | path | lines | licence |",
              "|---|---|---|---|---|"]
    for s in (report.get("sources") or {}).get("real") or []:
        cells = " | ".join(_cell(s[k]) for k in ("repo", "commit", "path", "lines", "licence"))
        lines.append("| " + cells + " |")
    lines.append("")
    lines.append(f"invented items: {(report.get('sources') or {}).get('invented_items', 0)}")
    lines.append("")
    return "\n".join(lines)


def cmd_write(args) -> int:
    report_path = Path(args.report)
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"refusing FROZEN.md: {scrub(e)}", file=sys.stderr)
        return 1
    why = refuse_reason(report)
    if why:
        print(f"refusing FROZEN.md: {why}", file=sys.stderr)
        return 1
    src = report_path.resolve().parent / report["no_role"]["file"]
    if not src.is_file():
        print("refusing FROZEN.md: the NO-ROLE file is not beside the report", file=sys.stderr)
        return 1
    blob = src.read_bytes().replace(b"\r\n", b"\n")
    if receipts.sha256_file(src) != report["no_role"]["sha256"]:
        print("refusing FROZEN.md: the NO-ROLE file's sha256 does not match the report", file=sys.stderr)
        return 1
    dest = Path(args.no_role_out) if args.no_role_out else REPO / NO_ROLE_PIN
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(blob)
    text = render_frozen(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(text.encode("utf-8"))
    print(receipts.sha256_file(out))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 1 freeze checks, then FROZEN.md.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="run the freeze checks and write freeze-report.json")
    c.add_argument("--data", nargs="+", required=True)
    c.add_argument("--apply", nargs="*", default=None)
    c.add_argument("--out", required=True)
    c.add_argument("--partial-set", action="store_true")
    c.add_argument("--sealed", default=str(receipts.SEALED_FILE))
    w = sub.add_parser("write", help="write FROZEN.md from a report that passed in full")
    w.add_argument("--report", required=True)
    w.add_argument("--out", required=True)
    w.add_argument("--no-role-out", help="where the pinned NO-ROLE file is copied "
                   f"(default: {NO_ROLE_PIN})")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        return cmd_check(args)
    return cmd_write(args)


if __name__ == "__main__":
    sys.exit(main())
