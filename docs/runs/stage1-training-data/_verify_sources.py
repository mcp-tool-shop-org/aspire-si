"""Source-fidelity check: for every real citation in sources[], prove the material is verbatim text
from the cited repo file at the pinned commit. Entries marked {"invented": ...} are skipped (they are
the stated invented parts — the checker trace in C, or a fully invented everyday scenario).
Reconstructs the cited lines from the local clones and requires
  - the pinned commit to be an ancestor of the clone's origin/HEAD,
  - the item's material to be a contiguous excerpt of the cited window (whitespace-collapsed,
    markup kept — the landed drift check's rule),
  - every DECIDING key to sit verbatim (markup kept) in the cited window,
  - no duplicate citations across the batch.
Run: python _verify_sources.py batch2-t14.jsonl ...
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Local clones of the cited repos sit side by side in one directory, named as on GitHub. STAGE1_CLONES names that
# directory; by default it's the one holding this aspire-si checkout.
CLONES = Path(os.environ.get("STAGE1_CLONES") or Path(__file__).resolve().parents[4])
REPO_DIR = {f"mcp-tool-shop-org/{name}": str(CLONES / name)
            for name in ("rig-bridge", "roll", "style-dataset-lab", "polyglot-mcp", "repo-knowledge", "site-theme")}


def ws(x):
    """Whitespace collapsed, markup kept — matches the landed data.py rule for quoted lines."""
    return re.sub(r"\s+", " ", x).strip()


def is_real(s):
    return isinstance(s, dict) and not s.get("invented")


def cited_text(src):
    repo = REPO_DIR[src["repo"]]
    blob = subprocess.run(["git", "-C", repo, "show", f"{src['commit']}:{src['path']}"],
                          capture_output=True, check=True).stdout.decode("utf-8")
    # One range ("L79-L86") or several, comma-separated ("L7, L21-L25"); the cited window is their lines in order.
    lines = blob.splitlines()
    window = []
    for part in src["lines"].split(","):
        lo_hi = re.fullmatch(r"L(\d+)(?:-L(\d+))?", part.strip())
        lo, hi = int(lo_hi.group(1)), int(lo_hi.group(2) or lo_hi.group(1))
        window += lines[lo - 1:hi]
    return "\n".join(window)


def main(paths):
    bad = 0
    seen = set()
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            reals = [s for s in r.get("sources", []) if is_real(s)]
            for s in reals:
                cited_text(s)  # raises if the blob/path is wrong
                head = subprocess.run(["git", "-C", REPO_DIR[s["repo"]], "merge-base", "--is-ancestor",
                                       s["commit"], "origin/HEAD"], capture_output=True).returncode
                if head != 0:
                    print(f"{r['id']}: {s['repo']}@{s['commit'][:12]} not on origin/HEAD"); bad += 1
                key = (s["repo"], s["commit"], s["path"], s["lines"])
                if key in seen:
                    print(f"{r['id']}: duplicate citation {key}"); bad += 1
                seen.add(key)
            if reals and r.get("material") is not None:
                blob = ws(cited_text(reals[0]))
                if ws(r["material"]) not in blob:
                    print(f"{r['id']}: material not a contiguous excerpt of cited window {reals[0]['lines']}"); bad += 1
                for keys in [r] + list((r.get("turn_keys") or {}).values()):
                    for d in (keys.get("deciding") or []):
                        if ws(d) not in blob:
                            print(f"{r['id']}: deciding line not verbatim in cited window: {d[:60]!r}"); bad += 1
    print(f"source check: {len(paths)} file(s), {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
