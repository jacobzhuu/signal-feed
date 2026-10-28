#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p "$SIGNALFEED_HOME/logs"
python scripts/download_data.py
signalfeed prepare
signalfeed encode
signalfeed features
signalfeed upstream-smoke --steps 32
for seed in 42 43 44; do
  signalfeed train --model sid --seed "$seed"
  signalfeed train --model ranker --seed "$seed"
done
python -c 'from signalfeed.ranker import train; train(personal=False)'
signalfeed evaluate --split valid --ranking-limit 2000 --retrieval-limit 2000
signalfeed tune-fusion
for seed in 42 43 44; do
  signalfeed evaluate --seed "$seed" --retrieval-limit 1000
done
python scripts/summarize.py
