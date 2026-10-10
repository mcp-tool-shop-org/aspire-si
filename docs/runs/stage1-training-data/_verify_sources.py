"""Source-fidelity check: for every item with sources[], prove the material is verbatim text
from the cited repo file at the pinned commit. Reconstructs the cited lines from the local
clones (which sit at the pinned origin/HEAD commits recorded in batch2) and require the item's
material to match the cited window after the scorer's normalisation. Run: python _verify_sources.py batch2-t14.jsonl ...
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_DIR = {
    "mcp-tool-shop-org/rig-bridge": r"E:/AI/rig-bridge",
    "mcp-tool-shop-org/roll": r"E:/AI/roll",
    "mcp-tool-shop-org/style-dataset-lab": r"E:/AI/style-dataset-lab",
    "mcp-tool-shop-org/polyglot-mcp": r"E:/AI/polyglot-mcp",
    "mcp-tool-shop-org/repo-knowledge": r"E:/AI/repo-knowledge",
    "mcp-tool-shop-org/site-theme": r"E:/AI/site-theme",
}


def norm(x):
    return re.sub(r"\s+", " ", x.replace("**", "").replace("__", "")).strip()


def cited_text(src):
    repo = REPO_DIR[src["repo"]]
    blob = subprocess.run(["git", "-C", repo, "show", f"{src['commit']}:{src['path']}"],
                          capture_output=True, check=True).stdout.decode("utf-8")
    lines = blob.splitlines()
    m = re.fullmatch(r"L(\d+)(?:-L(\d+))?", src["lines"])
    lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
    return "\n".join(lines[lo - 1:hi])


def main(paths):
    bad = 0
    seen = set()
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            for s in r.get("sources", []):
                blob = cited_text(s)
                # pinned commit must be an ancestor of (or equal to) the clone's origin/HEAD
                head = subprocess.run(["git", "-C", REPO_DIR[s["repo"]], "merge-base", "--is-ancestor",
                                       s["commit"], "origin/HEAD"], capture_output=True).returncode
                if head != 0:
                    print(f"{r['id']}: {s['repo']}@{s['commit'][:12]} not on origin/HEAD"); bad += 1
                key = (s["repo"], s["commit"], s["path"], s["lines"])
                if key in seen:
                    print(f"{r['id']}: duplicate citation {key}"); bad += 1
                seen.add(key)
            if r.get("sources") and r.get("material") is not None:
                blob = norm(cited_text(r["sources"][0]))
                if norm(r["material"]) not in blob:
                    print(f"{r['id']}: material not a contiguous excerpt of cited window {r['sources'][0]['lines']}"); bad += 1
                for d in (r.get("deciding") or []):
                    if norm(d) not in blob:
                        print(f"{r['id']}: deciding line not in cited window: {d[:60]!r}"); bad += 1
    print(f"source check: {len(paths)} file(s), {bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
