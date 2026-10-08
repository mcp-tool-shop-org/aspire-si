#!/usr/bin/env bash
# Step 2 of the 2026-10-08 plan (docs/runs/2026-10-08-next-step-plan.md): critic variance against
# the number of training prompts. Three seeds of the control-local condition (base student,
# Qwen2.5-32B teacher) train side by side on 128 prompts, on an offrig `job` pod (RTX PRO 6000).
# Each critic is then judged on the 127 planted-error pairs, and probed on the control's 32 fixed
# exchanges.
# Usage: HF_HOME=/root/hf bash plan_e.sh 42 43 44
# Expects in /workspace/job:
#   aspire-si.tar.gz   a `git archive` of this repository
#   data/              the experiment's data (judge_set.json)
#   control/           real-local-teacher/dialogue_cache (the probes' fixed exchanges)
# Each stage leaves a marker in /workspace/job/stages-e, so running this again resumes.
set -euo pipefail
SEEDS=("$@")
[ ${#SEEDS[@]} -gt 0 ] || { echo "usage: plan_e.sh SEED [SEED ...]"; exit 2; }
# HF_HOME may be set by the caller, e.g. to a container disk: HF_HOME=/root/hf bash plan_e.sh 42 43 44
# HF_XET_HIGH_PERFORMANCE: more parallel xet transfers; the speed probe measures with it set.
export HF_HOME=${HF_HOME:-/workspace/hf} PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1
J=/workspace/job
E=$J/aspire-si/examples/sft-experiment
R=$J/results-e
O=$J/aspire-si/outputs
BASE=Qwen/Qwen2.5-1.5B-Instruct
N=128
mkdir -p $J/stages-e $R

stage() {
  local name=$1
  shift
  if [ -f "$J/stages-e/$name.done" ]; then echo "== skip $name"; return; fi
  echo "== $name $(date -u +%T)"
  "$@"
  touch "$J/stages-e/$name.done"
  echo "== $name done $(date -u +%T)"
}

setup() {
  cd $J
  rm -rf aspire-si && mkdir aspire-si && tar -xzf aspire-si.tar.gz -C aspire-si
  # Refuse a host whose driver or GPU cannot run this, before installing anything.
  python aspire-si/examples/sft-experiment/host_check.py --min-cuda 13.0 --min-memory-gb 90
  python -m pip install -q --upgrade pip
  python -m pip install -q vllm
  python -m pip install -q -e ./aspire-si
  python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
  # Refuse a host too slow to fetch the teacher in time: cache writes, and one shard of the
  # teacher through the same hf/xet path as the download below. The budget is the plan's:
  # 66 GB (teacher and student) in 45 minutes.
  python aspire-si/examples/sft-experiment/host_check.py --speed --download-gb 66 --download-minutes 45
  for m in $BASE Qwen/Qwen2.5-32B-Instruct; do
    hf download "$m" --quiet > /dev/null
    echo "downloaded $m"
  done
}

configs() {
  for s in "${SEEDS[@]}"; do
    python $E/seed_configs.py --seed "$s" --prompts $N --out $J/configs-e
  done
}

# All seeds side by side (about 25 GB each); all must succeed.
aspire_runs() {
  cd $J/aspire-si
  local pids=()
  for s in "${SEEDS[@]}"; do
    local name=control-local-p$N-s$s
    aspire --verbose train --config $J/configs-e/$name.yaml --prompts $E/prompts-$N.json \
      --geometry > $R/aspire-$name.log 2>&1 &
    pids+=($!)
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  for s in "${SEEDS[@]}"; do tail -n 3 $R/aspire-control-local-p$N-s$s.log; done
  return $failed
}

probes() {
  for s in "${SEEDS[@]}"; do
    local o=$O/control-local-p$N-s$s
    python $E/probe_models.py --seed "$s" --pairs $J/control/real-local-teacher/dialogue_cache \
      --out $R/probe-control-local-p$N-s$s base=$BASE control-1=$BASE+$o/checkpoint-1/student \
      control-2=$BASE+$o/checkpoint-2/student control-3=$BASE+$o/checkpoint-3/student --drift-from base
  done
}

judge() {
  local entries=()
  for s in "${SEEDS[@]}"; do entries+=("control-local-p$N-s$s=$O/control-local-p$N-s$s/checkpoint-3"); done
  python $E/judge_eval.py --judge-set $J/data/judge_set.json --out $R/judge.json "${entries[@]}"
}

stage setup setup
stage configs configs
stage aspire aspire_runs
stage probes probes
stage judge judge
echo "PLAN-E-OK seeds ${SEEDS[*]}"
