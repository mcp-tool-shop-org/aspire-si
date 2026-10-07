#!/usr/bin/env bash
# Run 1 of the 2026-10-07 plan (docs/runs/2026-10-07-next-runs-plan.md): one more training seed
# for both conditions and both teachers, on an offrig `job` pod (RTX PRO 6000).
# Usage: bash plan_c.sh SEED   (43 or 44; seed 42 is the 2026-10-07 experiment)
# Expects in /workspace/job:
#   aspire-si.tar.gz   a `git archive` of this repository
#   data/              the experiment's data (train.jsonl, judge_set.json)
#   control/           real-local-teacher/ and real-composite-teacher/ (their dialogue caches are
#                      the probes' fixed exchanges)
# Each stage leaves a marker in /workspace/job/stages-sSEED, so running this again resumes.
set -euo pipefail
SEED=${1:?usage: plan_c.sh SEED}
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
J=/workspace/job
E=$J/aspire-si/examples/sft-experiment
R=$J/results-s$SEED
O=$J/aspire-si/outputs
BASE=Qwen/Qwen2.5-1.5B-Instruct
SFT=$J/sft-s$SEED/merged
mkdir -p $J/stages-s$SEED $R

stage() {
  local name=$1
  shift
  if [ -f "$J/stages-s$SEED/$name.done" ]; then echo "== skip $name"; return; fi
  echo "== $name $(date -u +%T)"
  "$@"
  touch "$J/stages-s$SEED/$name.done"
  echo "== $name done $(date -u +%T)"
}

setup() {
  cd $J
  rm -rf aspire-si && mkdir aspire-si && tar -xzf aspire-si.tar.gz -C aspire-si
  python -m pip install -q --upgrade pip
  python -m pip install -q vllm
  python -m pip install -q -e ./aspire-si
  python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
  for m in $BASE Qwen/Qwen2.5-32B-Instruct google/gemma-4-31B-it; do
    hf download "$m" --quiet > /dev/null
    echo "downloaded $m"
  done
}

sft() {
  cd $J && python $E/sft.py --data $J/data/train.jsonl --out $J/sft-s$SEED --seed $SEED
}

configs() {
  python $E/seed_configs.py --seed $SEED --sft-student $SFT --out $J/configs-s$SEED
}

# Two ASPIRE runs side by side; both must succeed.
pair() {
  cd $J/aspire-si
  local pids=()
  for name in "$@"; do
    aspire --verbose train --config $J/configs-s$SEED/$name.yaml --prompts examples/local-run/prompts.json \
      --geometry > $R/aspire-$name.log 2>&1 &
    pids+=($!)
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  for name in "$@"; do tail -n 3 $R/aspire-$name.log; done
  return $failed
}

composite() { pair control-composite-s$SEED sft-composite-s$SEED; }
local_pair() { pair control-local-s$SEED sft-local-s$SEED; }

probes() {
  for c in local composite; do
    local ctl=$O/control-$c-s$SEED sft=$O/sft-$c-s$SEED
    python $E/probe_models.py --pairs $J/control/real-$c-teacher/dialogue_cache --out $R/probe-control-$c \
      base=$BASE control-1=$BASE+$ctl/checkpoint-1/student control-2=$BASE+$ctl/checkpoint-2/student \
      control-3=$BASE+$ctl/checkpoint-3/student --drift-from base
    python $E/probe_models.py --pairs $J/control/real-$c-teacher/dialogue_cache --out $R/probe-sft-$c \
      base=$BASE sft=$SFT sft-aspire-1=$SFT+$sft/checkpoint-1/student \
      sft-aspire-2=$SFT+$sft/checkpoint-2/student sft-aspire-3=$SFT+$sft/checkpoint-3/student \
      --drift-from base --drift-from sft
  done
}

judge() {
  python $E/judge_eval.py --judge-set $J/data/judge_set.json --out $R/judge.json \
    control-local=$O/control-local-s$SEED/checkpoint-3 \
    control-composite=$O/control-composite-s$SEED/checkpoint-3 \
    sft-local=$O/sft-local-s$SEED/checkpoint-3 \
    sft-composite=$O/sft-composite-s$SEED/checkpoint-3
}

stage setup setup
stage sft sft
stage configs configs
stage aspire-composite composite
stage aspire-local local_pair
stage probes probes
stage judge judge
echo "PLAN-C-OK seed $SEED"
