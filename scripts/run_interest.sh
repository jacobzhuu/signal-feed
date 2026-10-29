#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SIGNALFEED_INTEREST_RUN:?Set a new prepared run}"
run="$SIGNALFEED_INTEREST_RUN"
python_bin="${SIGNALFEED_PYTHON:-.venv/bin/python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1
if [[ -e "$run/reports/select" ]]; then
  echo 'Refusing to overwrite an existing research run' >&2
  exit 1
fi
for split in select guard test; do
  "$python_bin" -m signalfeed.interest_experiments evaluate --run "$run" --split "$split" > "$run/logs/$split.log" 2>&1
  if [[ "$split" == select ]]; then
    "$python_bin" -m signalfeed.interest_experiments select --run "$run" > "$run/logs/selection.log" 2>&1
  elif [[ "$split" == guard ]]; then
    "$python_bin" -m signalfeed.interest_experiments guard --run "$run" > "$run/logs/guard-decision.log" 2>&1
  fi
done
