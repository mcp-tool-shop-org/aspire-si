#!/usr/bin/env bash
# Runs 2 and 3 of the 2026-10-07 plan (docs/runs/2026-10-07-next-runs-plan.md), on an offrig
# `job` pod (RTX PRO 6000). No training: it re-reads the seed-42 models of the
# fine-tune-then-ASPIRE experiment.
#   run 2: the six seed-42 students answer the 64 held-out questions with up to 1024 new tokens,
#          and the teacher scores them.
#   run 3: the teacher compares the two sides of every judge pair in both orders, and the
#          seed-42 critics' recorded pair scores are re-read on the pairs it separates.
# Expects in /workspace/job:
#   aspire-si.tar.gz      a `git archive` of this repository
#   data/                 the experiment's data (held_out.json, judge_set.json)
#   control/              real-local-teacher/ and real-composite-teacher/ (checkpoint-3/student)
#   sft/epoch-2/          the fine-tune's adapter (the merged student is rebuilt from it)
#   sft-outputs/          sft-local-teacher/ and sft-composite-teacher/ (checkpoint-3/student)
#   prev/judge.json       the seed-42 judge results
# Each stage leaves a marker in /workspace/job/stages-d, so running this again resumes.
set -euo pipefail
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1
J=/workspace/job
E=$J/aspire-si/examples/sft-experiment
R=$J/results-d
BASE=Qwen/Qwen2.5-1.5B-Instruct
TEACHER=Qwen/Qwen2.5-32B-Instruct
SFT=$J/sft/merged
mkdir -p $J/stages-d $R

stage() {
  local name=$1
  shift
  if [ -f "$J/stages-d/$name.done" ]; then echo "== skip $name"; return; fi
  echo "== $name $(date -u +%T)"
  "$@"
  touch "$J/stages-d/$name.done"
  echo "== $name done $(date -u +%T)"
}

setup() {
  cd $J
  rm -rf aspire-si && mkdir aspire-si && tar -xzf aspire-si.tar.gz -C aspire-si
  python -m pip install -q --upgrade pip
  python -m pip install -q vllm
  python -m pip install -q -e ./aspire-si
  python -c "import torch, vllm, transformers, peft; print('torch', torch.__version__, 'vllm', vllm.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)"
  for m in $BASE $TEACHER; do
    hf download "$m" --quiet > /dev/null
    echo "downloaded $m"
  done
}

merge() {
  python $E/sft.py --merge-only $J/sft/epoch-2 --out $J/sft --student $BASE
}

answers() {
  python $E/eval_heldout.py answer --held-out $J/data/held_out.json --out $R/eval --max-new-tokens 1024 \
    base=$BASE sft=$SFT \
    control-local=$BASE+$J/control/real-local-teacher/checkpoint-3/student \
    control-composite=$BASE+$J/control/real-composite-teacher/checkpoint-3/student \
    sft-local=$SFT+$J/sft-outputs/sft-local-teacher/checkpoint-3/student \
    sft-composite=$SFT+$J/sft-outputs/sft-composite-teacher/checkpoint-3/student
}

score() {
  python $E/eval_heldout.py score --out $R/eval --teacher $TEACHER
}

pairwise() {
  python $E/pairwise_teacher.py --judge-set $J/data/judge_set.json --out $R/pairwise.json --teacher $TEACHER
}

separable() {
  python $E/judge_eval.py --judge-set $J/data/judge_set.json --out $R/judge-separable.json \
    --reuse $J/prev/judge.json --subset $R/pairwise.json
}

stage setup setup
stage merge merge
stage answers answers
stage score score
stage pairwise pairwise
stage separable separable
echo PLAN-D-OK
