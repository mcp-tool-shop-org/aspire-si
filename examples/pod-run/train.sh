#!/usr/bin/env bash
# One run of the 2026-10-06 pod session, from /workspace/job on an offrig `job` pod.
# One ASPIRE run on the pod: $1 = condition (real-local-teacher | real-composite-teacher).
set -euo pipefail
cd /workspace/job/aspire-si
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_HOME=/workspace/hf
aspire --verbose train --config "examples/pod-run/$1.yaml" --prompts examples/local-run/prompts.json --geometry
python - "$1" <<'PY'
import sys
from aspire.judge import Judge
run = sys.argv[1]
judge = Judge.from_checkpoint(f"outputs/{run}/checkpoint-3")
pairs = [
    ("What is 2+2?", "2 + 2 = 4."),
    ("What is 2+2?", "It is 5, because numbers are flexible."),
    ("Why is the sky blue?", "Sunlight scatters off air molecules, and shorter blue wavelengths scatter far more (Rayleigh scattering), so blue light reaches our eyes from every direction."),
    ("Why is the sky blue?", "Because the ocean reflects onto it."),
]
for prompt, response in pairs:
    print(f"JUDGE {judge.score(prompt, response):.2f}  {prompt} -> {response[:60]}")
PY
echo "RUN-OK $1"
