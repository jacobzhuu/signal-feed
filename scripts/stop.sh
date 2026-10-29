#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python scripts/stop_server.py "$SIGNALFEED_HOME/server.pid" --timeout "${SIGNALFEED_STOP_TIMEOUT:-45}"
