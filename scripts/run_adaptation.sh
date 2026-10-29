#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SIGNALFEED_ADAPTATION_RUN:?Set a fresh prepared run}"
: "${SIGNALFEED_HOME:?Point to the immutable bound baseline}"
run="$SIGNALFEED_ADAPTATION_RUN"
if [[ ! -f "$run/dataset/manifest.json" ]]; then
  echo 'Data preparation must complete before evaluation' >&2
  exit 1
fi
if [[ -e "$run/reports/select-42/predictions.jsonl" ]]; then
  echo 'Refusing to overwrite a previous run' >&2
  exit 1
fi
export HF_HUB_OFFLINE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python_bin="${SIGNALFEED_PYTHON:-.venv/bin/python}"
"$python_bin" -m signalfeed.adaptation_experiments collect --run "$run" --split select --seed 42 > "$run/logs/select-42.log" 2>&1
"$python_bin" -m signalfeed.adaptation_experiments select --run "$run" > "$run/logs/selection.log" 2>&1
for split in guard test; do
  pids=()
  for seed in 42 43 44; do
    "$python_bin" -m signalfeed.adaptation_experiments collect --run "$run" --split "$split" --seed "$seed" > "$run/logs/$split-$seed.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  if [[ "$failed" != 0 ]]; then exit 1; fi
  if [[ "$split" == guard ]]; then
    "$python_bin" -m signalfeed.adaptation_experiments guard --run "$run" > "$run/logs/guard.log" 2>&1
  fi
done
touch "$run/predictions-completed"
