#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
mkdir -p "$SIGNALFEED_HOME/logs"
if [[ -f "$SIGNALFEED_HOME/server.pid" ]] && kill -0 "$(cat "$SIGNALFEED_HOME/server.pid")" 2>/dev/null; then
  echo "SignalFeed is already running: http://127.0.0.1:8765"
  exit 0
fi
nohup signalfeed serve >> "$SIGNALFEED_HOME/logs/server.log" 2>&1 &
echo "$!" > "$SIGNALFEED_HOME/server.pid"
for attempt in {1..1200}; do
  if curl --noproxy '*' --max-time 1 -fsS http://127.0.0.1:8765/api/ready >/dev/null 2>&1; then
    echo "SignalFeed ready: http://127.0.0.1:8765"
    exit 0
  fi
  sleep 0.1
done
echo "Startup did not become ready; inspect $SIGNALFEED_HOME/logs/server.log" >&2
exit 1
