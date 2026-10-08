#!/usr/bin/env bash
# Step 3 of the 2026-10-08 plan (docs/runs/2026-10-08-step-3-plan.md): the fine-tune comparison
# at 128 prompts, under rule 3 of docs/runs/2026-10-08-next-step-plan.md. Three seeds of the
# sft-local condition (each seed's own fine-tune from run 1 as the student, Qwen2.5-32B teacher)
# train side by side on the same 128 prompts as step 2's control-local runs (plan_e.sh), with
# everything else as there. Each critic is judged on the 127 planted-error pairs, and probed on
# the control's 32 fixed exchanges.
# Usage: HF_HOME=/root/hf bash plan_f.sh 42 43 44
# Expects in /workspace/job:
#   aspire-si.tar.gz   a `git archive` of this repository
#   data/              the experiment's data (judge_set.json)
#   control/           real-local-teacher/dialogue_cache (the probes' fixed exchanges)
#   sft-sSEED/epoch-2/ each seed's fine-tune adapter from run 1 (2026-10-07), merged here
# Each stage leaves a marker in /workspace/job/stages-f, so running this again resumes.
set -euo pipefail
SEEDS=("$@")
[ ${#SEEDS[@]} -gt 0 ] || { echo "usage: plan_f.sh SEED [SEED ...]"; exit 2; }
# HF_HOME may be set by the caller, e.g. to a container disk: HF_HOME=/root/hf bash plan_e.sh 42 43 44
# HF_XET_HIGH_PERFORMANCE: more parallel xet transfers; the speed probe measures with it set.
export HF_HOME=${HF_HOME:-/workspace/hf} PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1
J=/workspace/job
E=$J/aspire-si/examples/sft-experiment
R=$J/results-f
O=$J/aspire-si/outputs
BASE=Qwen/Qwen2.5-1.5B-Instruct
N=128
mkdir -p $J/stages-f $R

stage() {
  local name=$1
  shift
  if [ -f "$J/stages-f/$name.done" ]; then echo "== skip $name"; return; fi
  echo "== $name $(date -u +%T)"
  "$@"
  touch "$J/stages-f/$name.done"
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

# Each seed's run 1 fine-tune (2 epochs, saved as an adapter), merged into a bf16 student.
merge() {
  for s in "${SEEDS[@]}"; do
    python $E/sft.py --merge-only $J/sft-s$s/epoch-2 --out $J/sft-s$s --student $BASE
  done
}

configs() {
  for s in "${SEEDS[@]}"; do
    python $E/seed_configs.py --seed "$s" --prompts $N --sft-student $J/sft-s$s/merged --out $J/configs-f
  done
}

# A progress line every PROGRESS_SECONDS (600) while ASPIRE runs, for the deadline check:
#   progress HH:MM:SS s42 dialogues 37/128 epochs 0/3 | s43 ...
# Each seed caches one dialogue per prompt in epoch 1, so dialogues/128 is epoch 1's progress;
# "epochs" counts the "Epoch k/3 - Loss" lines its log has printed. An epoch's end is also
# printed once, as "epoch-end HH:MM:SS s42 epoch 1".
progress() {
  local seen=()
  while true; do
    local line="progress $(date -u +%T)" i=0
    for s in "${SEEDS[@]}"; do
      local name=sft-local-p$N-s$s
      local n=$(ls $O/$name/dialogue_cache 2>/dev/null | wc -l)
      local e=$(grep -c -E "^Epoch [0-9]+/[0-9]+ - Loss" $R/aspire-$name.log 2>/dev/null || true)
      e=${e:-0}
      line="$line s$s dialogues $n/$N epochs $e/3 |"
      while [ "${seen[$i]:-0}" -lt "$e" ]; do
        seen[$i]=$(( ${seen[$i]:-0} + 1 ))
        echo "epoch-end $(date -u +%T) s$s epoch ${seen[$i]}"
      done
      i=$((i + 1))
    done
    echo "$line"
    sleep "${PROGRESS_SECONDS:-600}"
  done
}

# All seeds side by side (about 25 GB each); all must succeed.
aspire_runs() {
  cd $J/aspire-si
  local pids=()
  for s in "${SEEDS[@]}"; do
    local name=sft-local-p$N-s$s
    aspire --verbose train --config $J/configs-f/$name.yaml --prompts $E/prompts-$N.json \
      --geometry > $R/aspire-$name.log 2>&1 &
    pids+=($!)
  done
  progress &
  local watcher=$!
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  kill "$watcher" 2>/dev/null || true
  for s in "${SEEDS[@]}"; do tail -n 3 $R/aspire-sft-local-p$N-s$s.log; done
  return $failed
}

probes() {
  for s in "${SEEDS[@]}"; do
    local o=$O/sft-local-p$N-s$s sft=$J/sft-s$s/merged
    python $E/probe_models.py --seed "$s" --pairs $J/control/real-local-teacher/dialogue_cache \
      --out $R/probe-sft-local-p$N-s$s base=$BASE sft=$sft sft-aspire-1=$sft+$o/checkpoint-1/student \
      sft-aspire-2=$sft+$o/checkpoint-2/student sft-aspire-3=$sft+$o/checkpoint-3/student \
      --drift-from base --drift-from sft
  done
}

judge() {
  local entries=()
  for s in "${SEEDS[@]}"; do entries+=("sft-local-p$N-s$s=$O/sft-local-p$N-s$s/checkpoint-3"); done
  python $E/judge_eval.py --judge-set $J/data/judge_set.json --out $R/judge.json "${entries[@]}"
}

stage setup setup
stage merge merge
stage configs configs
stage aspire aspire_runs
stage probes probes
stage judge judge
echo "PLAN-F-OK seeds ${SEEDS[*]}"
