#!/usr/bin/env bash
# Plan A of the fine-tune-then-ASPIRE experiment, on an offrig `job` pod (RTX PRO 6000).
# Expects aspire-si.tar.gz (a `git archive` of this repository) in /workspace/job.
# Builds the dataset, held-out prompts, judge set and noise floor with Qwen2.5-32B on vLLM.
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
hf download BAAI/bge-small-en-v1.5 --quiet > /dev/null
echo "models downloaded"
cd aspire-si/examples/sft-experiment
python build_dataset.py --teacher Qwen/Qwen2.5-32B-Instruct --eval-prompts ../local-run/prompts.json \
  --out /workspace/job/data
echo PLAN-A-OK
