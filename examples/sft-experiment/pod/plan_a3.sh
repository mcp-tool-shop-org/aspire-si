#!/usr/bin/env bash
# Plan A3 of the fine-tune-then-ASPIRE experiment, on an offrig `job` pod (RTX PRO 6000).
# Rebuilds only the judge set of Plan A2: keeps A2's training data, strong answers and passing
# pairs, plants errors as one-sentence JSON edits where a pair is missing, then cleans.
# Expects in /workspace/job:
#   aspire-si.tar.gz     a `git archive` of this repository
#   prev/data-raw/       Plan A2's raw output
# Writes /workspace/job/data-raw (with the new judge set) and /workspace/job/data (cleaned).
set -euo pipefail
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
cd /workspace/job
rm -rf aspire-si && mkdir aspire-si && tar -xzf aspire-si.tar.gz -C aspire-si
python -V; nvidia-smi --query-gpu=name,memory.total --format=csv
python -m pip install -q --upgrade pip
# vLLM brings its own torch (and torchvision), replacing the image's nightly builds.
python -m pip install -q vllm
python -m pip install -q -e ./aspire-si
python -c "import torch, vllm, transformers; print('torch', torch.__version__, torch.cuda.get_device_name(0), 'vllm', vllm.__version__, 'transformers', transformers.__version__)"
hf download Qwen/Qwen2.5-32B-Instruct --quiet > /dev/null
echo "model downloaded"
cd aspire-si/examples/sft-experiment
python build_dataset.py --teacher Qwen/Qwen2.5-32B-Instruct --judge-only --from /workspace/job/prev/data-raw \
  --flaws-per-prompt 2 --edit-attempts 5 --out /workspace/job/data-raw
# Exit 3 means too few judge pairs survived: the data is kept for review, Plan B must not start.
python clean_dataset.py --data /workspace/job/data-raw --out /workspace/job/data
echo PLAN-A3-OK
