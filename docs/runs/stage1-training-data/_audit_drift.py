import json, re, sys
from pathlib import Path
# The validator from this checkout (beside this directory), never from another clone.
RUNS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RUNS / "stage1-train"))
sys.path.insert(0, str(RUNS / "stage1-eval"))
import data as D

def ws(x): return re.sub(r"\s+", " ", x).strip()

for f in ["batch2-t01.jsonl", "batch2-t14.jsonl", "batch2-t16.jsonl"]:
    rows = [json.loads(x) for x in Path(f).read_text(encoding="utf-8").splitlines() if x.strip()]
    # 1) deciding verbatim w/ markup (ws) — for verdict-keyed kinds: C/D items and E verdict quizzes
    for r in rows:
        mat = ws((r.get("material") or "") + "\n" + "\n".join(t["user"] for t in r["turns"]))
        keysets = []
        if r["kind"] in ("C", "D"): keysets.append((r["id"], r.get("deciding") or []))
        if r["kind"] == "E" and r.get("quiz", {}).get("type") == "verdict":
            keysets.append((r["id"], r["quiz"].get("deciding") or []))
        for where, ks in keysets:
            for d in ks:
                if ws(d) not in mat:
                    print(f"{f} {where}: deciding NOT verbatim (markup-kept): {d[:70]!r}")
    # 2) E written variant overlap
    for r in rows:
        if r["kind"] == "E" and r.get("quiz", {}).get("type") == "written":
            orig = D._key_points(r["quiz"]); var = " ".join(D._key_points(r["variant"]["key"]))
            if not any(k and (k in var or var in k) for k in orig):
                print(f"{f} {r['id']}: variant key shares NO key point with original")
                print("   orig:", orig)
                print("   var :", D._key_points(r["variant"]["key"]))
    # 3) frames (first 6 words, C/D/E, max 2 per trait)
    seen = {}
    for r in rows:
        if r["kind"] in ("C", "D", "E"):
            op = " ".join(re.findall(r"[a-z']+", r["turns"][0]["user"].lower())[:6])
            if len(op.split()) == 6: seen.setdefault((r["lesson"], op), []).append(r["id"])
    for (l, op), ids in seen.items():
        if len(ids) > 2: print(f"{f}: frame x{len(ids)}: {op!r} -> {ids}")
    # 4) B stamps (8-word shingles >2 per trait)
    seen = {}
    for r in rows:
        if r["kind"] == "B":
            text = " ".join(t["thinking"] + " " + t["reply"] for t in r["turns"])
            for sh in D._shingles(text): seen.setdefault(sh, set()).add(r["id"])
    for sh, ids in seen.items():
        if len(ids) > 2: print(f"{f}: B stamp x{len(ids)} {sorted(ids)}: {sh!r}")
    # 5) time words in statements (C/D/E) where material undated
    for r in rows:
        if r["kind"] in ("C", "D", "E"):
            stmt = D._statement(r["turns"][0]["user"])
            if stmt and D.TIME_WORDS.search(stmt) and not D.DATED.search(r.get("material") or ""):
                print(f"{f} {r['id']}: time word {D.TIME_WORDS.search(stmt).group(0)!r} in statement")
    # 6) A: 2 sentences before question, trait named, first-person reply, no VERDICT
    for r in rows:
        if r["kind"] == "A":
            fu = r["turns"][0]["user"]; q = fu.rfind("?")
            before = fu[:fu.rfind(".", 0, q) + 1] if q > 0 else fu
            if D._sentences(before) < 2: print(f"{f} {r['id']}: A passage < 2 sentences")
            names = D.TRAIT_NAMES.get(r["lesson"], [])
            if names and not any(n in fu.lower() for n in names): print(f"{f} {r['id']}: A passage doesn't name trait")
    # 7) C: verdict word in user turn
    for r in rows:
        if r["kind"] == "C":
            if not re.search(r"\b(verdict|supported|unsupported|cannot_tell|can't tell)\b", r["turns"][0]["user"], re.I):
                print(f"{f} {r['id']}: C user turn lacks checker verdict word")
    # 8) full data.validate per file (catch anything I missed, collecting first error)
    try:
        D.validate(rows)
    except D.DataError as e:
        print(f"{f}: VALIDATOR(first): {e}")
print("audit done")
