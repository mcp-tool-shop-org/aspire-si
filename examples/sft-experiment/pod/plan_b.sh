#!/usr/bin/env bash
# Plan B of the fine-tune-then-ASPIRE experiment, on an offrig `job` pod (RTX PRO 6000).
# Expects in /workspace/job:
#   aspire-si.tar.gz   a `git archive` of this repository
#   data/              Plan A's output (train.jsonl, held_out.json, judge_set.json, ...)
#   control/           the control runs: real-local-teacher/ and real-composite-teacher/
#                      (checkpoints and dialogue caches, from the 2026-10-06 pod run)
# Each stage leaves a marker in /workspace/job/stages, so running this again resumes.
set -euo pipefail
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
J=/workspace/job
E=$J/aspire-si/examples/sft-experiment
R=$J/results
BASE=Qwen/Qwen2.5-1.5B-Instruct
SFT=$J/sft/merged
mkdir -p $J/stages $R

stage() {
  local name=$1
  shift
  if [ -f "$J/stages/$name.done" ]; then echo "== skip $name"; return; fi
  echo "== $name $(date -u +%T)"
  "$@"
  touch "$J/stages/$name.done"
  echo "== $name done $(date -u +%T)"
}

setup() {
  cd $J
  rm -rf aspire-si && mkdir aspire-si && tar -xzf aspire-si.tar.gz -C aspire-si
  python -m pip install -q --upgrade pip
  python -m pip install -q vllm
  python -m pip install -q -e ./aspire-si
  python -c "import torch, vllm, transformers, peft; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
  for m in $BASE Qwen/Qwen2.5-32B-Instruct google/gemma-4-31B-it; do
    hf download "$m" --quiet > /dev/null
    echo "downloaded $m"
  done
}

sft() {
  cd $J && python $E/sft.py --data $J/data/train.jsonl --out $J/sft
}

probe_sft() {
  for c in local composite; do
    python $E/probe_models.py --seed 42 --pairs $J/control/real-$c-teacher/dialogue_cache --out $R/probe-sft-$c \
      base=$BASE sft-epoch-1=$BASE+$J/sft/epoch-1 sft-epoch-2=$BASE+$J/sft/epoch-2 sft=$SFT --drift-from base
  done
}

aspire_runs() {
  cd $J/aspire-si
  local pids=()
  for c in local composite; do
    aspire --verbose train --config $E/configs/sft-$c-teacher.yaml --prompts examples/local-run/prompts.json \
      --geometry > $R/aspire-sft-$c.log 2>&1 &
    pids+=($!)
  done
  local failed=0
  for p in "${pids[@]}"; do wait "$p" || failed=1; done
  tail -n 5 $R/aspire-sft-local.log $R/aspire-sft-composite.log
  return $failed
}

probe_aspire() {
  for c in local composite; do
    local out=$J/aspire-si/outputs/sft-$c-teacher
    python $E/probe_models.py --seed 42 --pairs $J/control/real-$c-teacher/dialogue_cache --out $R/probe-aspire-$c \
      base=$BASE sft=$SFT sft-aspire-1=$SFT+$out/checkpoint-1/student \
      sft-aspire-2=$SFT+$out/checkpoint-2/student sft-aspire-3=$SFT+$out/checkpoint-3/student \
      --drift-from base --drift-from sft
  done
}

answers() {
  local o=$J/aspire-si/outputs
  python $E/eval_heldout.py answer --held-out $J/data/held_out.json --out $R/eval \
    base=$BASE sft=$SFT \
    control-local=$BASE+$J/control/real-local-teacher/checkpoint-3/student \
    control-composite=$BASE+$J/control/real-composite-teacher/checkpoint-3/student \
    sft-local=$SFT+$o/sft-local-teacher/checkpoint-3/student \
    sft-composite=$SFT+$o/sft-composite-teacher/checkpoint-3/student
}

score() {
  python $E/eval_heldout.py score --out $R/eval --teacher Qwen/Qwen2.5-32B-Instruct
}

judge() {
  local o=$J/aspire-si/outputs
  python $E/judge_eval.py --judge-set $J/data/judge_set.json --out $R/judge.json \
    control-local=$J/control/real-local-teacher/checkpoint-3 \
    control-composite=$J/control/real-composite-teacher/checkpoint-3 \
    sft-local=$o/sft-local-teacher/checkpoint-3 \
    sft-composite=$o/sft-composite-teacher/checkpoint-3
}

stage setup setup
stage sft sft
stage probe-sft probe_sft
stage aspire aspire_runs
stage probe-aspire probe_aspire
stage answers answers
stage score score
stage judge judge
echo PLAN-B-OK
