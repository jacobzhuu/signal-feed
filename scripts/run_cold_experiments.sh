#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${SIGNALFEED_HOME:?Set a fresh independent output directory}"
: "${SIGNALFEED_BASELINE_HOME:?Set the audited baseline directory}"
if [[ "$(realpath -m "$SIGNALFEED_HOME")" == "$(realpath -m "$SIGNALFEED_BASELINE_HOME")" ]]; then
  echo 'ColdFuse output must differ from the audited baseline directory' >&2
  exit 1
fi
mkdir -p "$SIGNALFEED_HOME/models" "$SIGNALFEED_HOME/reports" "$SIGNALFEED_HOME/logs"
if [[ ! -f "$SIGNALFEED_HOME/mind/news.json" ]]; then
  cp -a "$SIGNALFEED_BASELINE_HOME/mind" "$SIGNALFEED_HOME/"
  ln -s "$SIGNALFEED_BASELINE_HOME/hf" "$SIGNALFEED_HOME/hf"
fi
for seed in 42 43 44; do
  for kind in sid ranker; do
    if [[ ! -d "$SIGNALFEED_HOME/models/$kind-$seed" ]]; then
      cp -a "$SIGNALFEED_BASELINE_HOME/models/$kind-$seed" "$SIGNALFEED_HOME/models/"
    fi
  done
done
if [[ ! -f "$SIGNALFEED_HOME/models/fusion.json" ]]; then
  cp "$SIGNALFEED_BASELINE_HOME/models/fusion.json" "$SIGNALFEED_HOME/models/"
fi
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
for seed in 42 43 44; do
  if [[ ! -f "$SIGNALFEED_HOME/models/coldfuse-$seed/train.json" ]]; then
    .venv/bin/python -m signalfeed.cli train-cold --seed "$seed" > "$SIGNALFEED_HOME/logs/train-cold-$seed.log" 2>&1
  fi
done
if [[ ! -f "$SIGNALFEED_HOME/models/coldfuse.json" ]]; then
  .venv/bin/python -m signalfeed.cold_experiments tune > "$SIGNALFEED_HOME/logs/cold-tune.log" 2>&1
fi
for seed in 42 43 44; do
  if [[ ! -f "$SIGNALFEED_HOME/reports/cold-test-$seed/metrics.json" ]]; then
    .venv/bin/python -m signalfeed.cold_experiments evaluate --seed "$seed" > "$SIGNALFEED_HOME/logs/cold-test-$seed.log" 2>&1
  fi
  if [[ ! -f "$SIGNALFEED_HOME/reports/cold-natural-$seed/metrics.json" ]]; then
    .venv/bin/python -m signalfeed.cold_experiments evaluate --seed "$seed" --natural > "$SIGNALFEED_HOME/logs/cold-natural-$seed.log" 2>&1
  fi
done
.venv/bin/python scripts/summarize_cold.py --run "$SIGNALFEED_HOME" > "$SIGNALFEED_HOME/logs/cold-summary.log" 2>&1
touch "$SIGNALFEED_HOME/completed"
