"""Write the four ASPIRE configs for one more training seed (run 1 of the 2026-10-07 plan).

The control (examples/pod-run/real-*-teacher.yaml) and the fine-tune condition
(configs/sft-*-teacher.yaml) differ only in their student, run name and output folder. For a
seed N this writes, into --out:

  control-local-sN.yaml   control-composite-sN.yaml   from the base student
  sft-local-sN.yaml       sft-composite-sN.yaml       from --sft-student (that seed's fine-tune)

each with `seed: N`, `experiment_name` set to the file's name and `training.output_dir` to
outputs/<name>, so no two runs share a dialogue cache: a new seed samples new dialogues. The CLI
has no seed option, which is why the configs are written out.

With --prompts N (steps 2 and 3 of the 2026-10-08 plan) it writes one local-teacher config for
that seed, with its own output folder; the N prompts are passed to `aspire train --prompts`
(make_prompts.py writes them):
  - without --sft-student: control-local-pN-sSEED, from the base student (step 2);
  - with --sft-student: sft-local-pN-sSEED, from that seed's fine-tune (step 3).

Usage: python seed_configs.py --seed 43 --sft-student /workspace/job/sft-s43/merged --out configs-s43
       python seed_configs.py --seed 43 --prompts 128 --out configs-p128
       python seed_configs.py --seed 43 --prompts 128 --sft-student sft-s43/merged --out configs-f
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

HERE = Path(__file__).parent
SOURCES = {
    "control-local": HERE.parent / "pod-run" / "real-local-teacher.yaml",
    "control-composite": HERE.parent / "pod-run" / "real-composite-teacher.yaml",
    "sft-local": HERE / "configs" / "sft-local-teacher.yaml",
    "sft-composite": HERE / "configs" / "sft-composite-teacher.yaml",
}


def seed_configs(seed: int, sft_student: str, out: Path) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for condition, source in SOURCES.items():
        config = yaml.safe_load(source.read_text(encoding="utf-8"))
        name = f"{condition}-s{seed}"
        config["seed"] = seed
        config["experiment_name"] = name
        config["training"]["output_dir"] = f"outputs/{name}"
        if condition.startswith("sft-"):
            config["student"]["model_name_or_path"] = sft_student
        path = out / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
        written[name] = path
    return written


def prompt_config(seed: int, prompts: int, out: Path, sft_student: str | None = None) -> Path:
    """The local-teacher config for one seed at a given training-prompt count: control-local from
    the base student, or sft-local from `sft_student` (that seed's merged fine-tune)."""
    out.mkdir(parents=True, exist_ok=True)
    condition = "sft-local" if sft_student else "control-local"
    config = yaml.safe_load(SOURCES[condition].read_text(encoding="utf-8"))
    name = f"{condition}-p{prompts}-s{seed}"
    config["seed"] = seed
    config["experiment_name"] = name
    config["training"]["output_dir"] = f"outputs/{name}"
    if sft_student:
        config["student"]["model_name_or_path"] = sft_student
    path = out / f"{name}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sft-student", help="that seed's merged fine-tune (all four conditions)")
    parser.add_argument("--prompts", type=int, help="one local-teacher config, named for this prompt count")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.prompts:
        print(prompt_config(args.seed, args.prompts, args.out, args.sft_student))
        return
    if not args.sft_student:
        parser.error("give --sft-student (four conditions) or --prompts N (control-local only)")
    for name, path in seed_configs(args.seed, args.sft_student, args.out).items():
        print(name, path)


if __name__ == "__main__":
    main()
