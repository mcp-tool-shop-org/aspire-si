"""Stage-1 training data: the gates live in the aspire-si-gates package (packages/aspire-si-gates), shared with
the dataset generator; this module re-exports them under their old names and keeps the trainer-side part:
turning items into training sequences.

Every assistant turn is a target, whole: `<think>{thinking}</think>` then the reply. One sequence per assistant
turn; the prefix is the conversation so far, with earlier assistant turns shown by their reply only (as the
chat template renders history at inference). Loss falls on the target only.
"""

import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[3] / "packages" / "aspire-si-gates" / "src"
if _PKG.is_dir() and str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))  # the repo copy; an installed aspire-si-gates works the same

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _evaldir import EVAL_DIR  # noqa: E402

sys.path.insert(0, str(EVAL_DIR))  # kept: no_role.py (hash-frozen) imports harness through this path

from aspire_si_gates import gates as _gates  # noqa: E402

globals().update({k: v for k, v in vars(_gates).items() if not k.startswith("__")})


# ---------------------------------------------------------------- rendering

def sequences(r: dict, tok) -> list[dict]:
    """[{prompt_ids, target_ids}] for each assistant turn of one item."""
    msgs, out = [], []
    for t in r["turns"]:
        msgs.append({"role": "user", "content": t["user"]})
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        think = t["thinking"].strip()
        target = (f"<think>\n{think}\n</think>\n\n" if think else "<think>\n\n</think>\n\n") + t["reply"].strip() \
            + "<|im_end|>"
        out.append({"prompt_ids": tok(prompt, add_special_tokens=False)["input_ids"],
                    "target_ids": tok(target, add_special_tokens=False)["input_ids"]})
        msgs.append({"role": "assistant", "content": t["reply"].strip()})
    return out


if __name__ == "__main__":
    # python data.py --freeze <file.jsonl> [<file.jsonl> ...]: the whole-dataset check, over all files together
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", nargs="+", metavar="FILE")
    ap.add_argument("--batch", nargs="+", metavar="FILE", help="cross-file diversity checks for one batch")
    a = ap.parse_args()
    if a.batch:
        batch = [r for f in a.batch for r in load(f)]
        print(json.dumps(batch_report(batch), indent=1))
        try:
            check_puzzles(batch)
            if framing_repeats(batch):
                raise DataError(f"{len(framing_repeats(batch))} framing line(s) repeated across the batch")
            print("batch: clean")
        except DataError as e:
            print(e)
            sys.exit(1)
        sys.exit(0)
    if not a.freeze:
        ap.error("--freeze or --batch is required")
    everything = [r for f in a.freeze for r in load(f)]
    try:
        print(json.dumps(freeze_report(everything), indent=1))
        print("freeze: clean")
    except DataError as e:
        print(e)
        sys.exit(1)
