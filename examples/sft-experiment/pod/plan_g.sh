#!/usr/bin/env bash
# The 2026-10-08 critic-init test (docs/runs/2026-10-08-critic-init-plan.md): which seed carries the
# critic's spread between runs. The 32-prompt control-local run (base student, Qwen2.5-32B teacher,
# examples/local-run/prompts.json) is trained with its two sources of randomness seeded apart:
#   the run's seed R        student adapter initialisation, data order, student and teacher sampling
#   the critic's seed C     the critic head's initial weights only (critic.init_seed)
# Arm A varies C with R fixed; arm B varies R with C fixed. They share the run R=42, C=42.
# A repeat of 42:42 (R:C:TAG, here 42:42:b) measures the noise between runs with identical seeds.
# Usage: HF_HOME=/root/hf bash plan_g.sh 42:42 42:43 43:42 42:44 44:42 42:42:b
#        (R:C or R:C:TAG; a TAG gives a repeat its own name and output folder)
# Runs go three at a time (about 23 GB each on one 96 GB GPU), in the order given.
# Expects in /workspace/job:
#   aspire-si.tar.gz   a `git archive` of this repository
#   data/              the experiment's data (judge_set.json)
#   control/           real-local-teacher/dialogue_cache (the probes' fixed exchanges)
# Each stage leaves a marker in /workspace/job/stages-g, so running this again resumes.
set -euo pipefail
PAIRS=("$@")
[ ${#PAIRS[@]} -gt 0 ] || { echo "usage: plan_g.sh R:C [R:C ...]"; exit 2; }
# HF_XET_HIGH_PERFORMANCE: more parallel xet transfers; the speed probe measures with it set.
export HF_HOME=${HF_HOME:-/workspace/hf} PYTHONUNBUFFERED=1 HF_XET_HIGH_PERFORMANCE=1
J=/workspace/job
E=$J/aspire-si/examples/sft-experiment
R=$J/results-g
O=$J/aspire-si/outputs
BASE=Qwen/Qwen2.5-1.5B-Instruct
N=32
# R:C or R:C:TAG -> run seed, critic seed, run name
run_seed() { echo "${1%%:*}"; }
critic_seed() { local rest=${1#*:}; echo "${rest%%:*}"; }
run_tag() { local rest=${1#*:}; [ "$rest" = "${rest#*:}" ] && echo "" || echo "${rest#*:}"; }
run_name() {
  local tag
  tag=$(run_tag "$1")
  echo "control-local-r$(run_seed "$1")-c$(critic_seed "$1")${tag:+-$tag}"
}
NAMES=()
for p in "${PAIRS[@]}"; do NAMES+=("$(run_name "$p")"); done
mkdir -p $J/stages-g $R

stage() {
  local name=$1
  shift
  if [ -f "$J/stages-g/$name.done" ]; then echo "== skip $name"; return; fi
  echo "== $name $(date -u +%T)"
  "$@"
  touch "$J/stages-g/$name.done"
  echo "== $name done $(date -u +%T)"
}

setup() {
  cd $J
  rm -rf aspire-si && mkdir aspire-si && tar -xzf aspire-si.tar.gz -C aspire-si
  # Refuse a host whose driver or GPU cannot run this, before installing anything.
  python aspire-si/examples/sft-experiment/host_check.py --min-cuda 13.0 --min-memory-gb 90
  python -m pip install -q --upgrade pip
  python -m pip install -q vllm pytest
  python -m pip install -q -e ./aspire-si
  python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
  # The seed split's tests, here with CUDA, so the GPU-stream test runs too.
  (cd aspire-si && python -m pytest tests/test_critic_init_seed.py -q -p no:cacheprovider)
  # Refuse a host too slow to fetch the teacher in time: 66 GB (teacher and student) in 30 minutes.
  python aspire-si/examples/sft-experiment/host_check.py --speed --download-gb 66 --download-minutes 30
  for m in $BASE Qwen/Qwen2.5-32B-Instruct; do
    hf download "$m" --quiet > /dev/null
    echo "downloaded $m"
  done
}

configs() {
  for p in "${PAIRS[@]}"; do
    python $E/seed_configs.py --seed "$(run_seed "$p")" --critic-seed "$(critic_seed "$p")" \
      --tag "$(run_tag "$p")" --out $J/configs-g
  done
}

# A progress line every PROGRESS_SECONDS (600) while a batch runs, for the deadline check:
#   progress HH:MM:SS control-local-r42-c42 dialogues 12/32 epochs 0/3 | ...
progress() {
  local seen=()
  while true; do
    local line="progress $(date -u +%T)" i=0
    for name in "$@"; do
      local n=$(ls $O/$name/dialogue_cache 2>/dev/null | wc -l)
      local e=$(grep -c -E "^Epoch [0-9]+/[0-9]+ - Loss" $R/aspire-$name.log 2>/dev/null || true)
      e=${e:-0}
      line="$line $name dialogues $n/$N epochs $e/3 |"
      while [ "${seen[$i]:-0}" -lt "$e" ]; do
        seen[$i]=$(( ${seen[$i]:-0} + 1 ))
        echo "epoch-end $(date -u +%T) $name epoch ${seen[$i]}"
      done
      i=$((i + 1))
    done
    echo "$line"
    sleep "${PROGRESS_SECONDS:-600}"
  done
}

# One batch of runs side by side; all must succeed.
batch() {
  cd $J/aspire-si
  local pids=()
  for name in "$@"; do
    aspire --verbose train --config $J/configs-g/$name.yaml --prompts examples/local-run/prompts.json \
      --geometry > $R/aspire-$name.log 2>&1 &
    pids+=($!)
  done
  progress "$@" &
  local watcher=$!
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  kill "$watcher" 2>/dev/null || true
  for name in "$@"; do tail -n 3 $R/aspire-$name.log; done
  return $failed
}

batch_1() { batch "${NAMES[@]:0:3}"; }
batch_2() { if [ ${#NAMES[@]} -gt 3 ]; then batch "${NAMES[@]:3}"; fi; }

probes() {
  for p in "${PAIRS[@]}"; do
    local name o
    name=$(run_name "$p")
    o=$O/$name
    python $E/probe_models.py --seed "$(run_seed "$p")" --pairs $J/control/real-local-teacher/dialogue_cache \
      --out $R/probe-$name base=$BASE control-1=$BASE+$o/checkpoint-1/student \
      control-2=$BASE+$o/checkpoint-2/student control-3=$BASE+$o/checkpoint-3/student --drift-from base
  done
}

judge() {
  local entries=()
  for name in "${NAMES[@]}"; do entries+=("$name=$O/$name/checkpoint-3"); done
  python $E/judge_eval.py --judge-set $J/data/judge_set.json --out $R/judge.json "${entries[@]}"
}

stage setup setup
stage configs configs
stage batch-1 batch_1
stage batch-2 batch_2
stage probes probes
stage judge judge
echo "PLAN-G-OK ${PAIRS[*]}"
