#!/usr/bin/env bash
# Pod setup for the 2026-10-06 runs: on an offrig `job` pod, with aspire-si.tar.gz (a `git archive` of
# this repository) in /workspace/job.
# Pod setup: unpack aspire-si, install it on a CUDA build of torch, fetch the models.
set -euo pipefail
# Over ssh the pod's own environment is not inherited: put downloads on the volume explicitly.
export HF_HOME=/workspace/hf
cd /workspace/job
rm -rf aspire-si && mkdir -p aspire-si && tar -xzf aspire-si.tar.gz -C aspire-si
python -V; nvidia-smi --query-gpu=name,memory.total --format=csv
python -m pip install -q --upgrade pip
python -m pip install -q --upgrade torch --index-url https://download.pytorch.org/whl/cu128
# The image ships nightly torchvision/torchaudio built for its own torch; ASPIRE needs neither.
python -m pip uninstall -q -y torchvision torchaudio
python -m pip install -q -e ./aspire-si "huggingface_hub[hf_transfer]"
python -c "import torch,transformers,peft,bitsandbytes;print('torch',torch.__version__,torch.cuda.get_device_name(0),'transformers',transformers.__version__,'peft',peft.__version__,'bnb',bitsandbytes.__version__)"
export HF_HUB_ENABLE_HF_TRANSFER=1
for m in Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-32B-Instruct google/gemma-4-31B-it; do
  hf download "$m" --quiet > /dev/null
  echo "downloaded $m"
done
du -sh "$HF_HOME"
echo SETUP-OK
