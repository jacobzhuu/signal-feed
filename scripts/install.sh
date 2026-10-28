#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
uv venv --allow-existing --python 3.11 "$SIGNALFEED_HOME/venv"
uv pip install --python "$SIGNALFEED_HOME/venv/bin/python" torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
uv pip install --python "$SIGNALFEED_HOME/venv/bin/python" -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu118 --index-strategy unsafe-best-match
(cd frontend && npm ci && npm run build)
