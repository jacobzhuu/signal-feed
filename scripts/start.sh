#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
export PYTHONPATH="$PWD/src"
if [[ -f RELEASE ]]; then export SIGNALFEED_RELEASE="$(cat RELEASE)"; fi
host="${SIGNALFEED_HOST:-127.0.0.1}"
port="${SIGNALFEED_PORT:-8765}"
probe_host="$host"
if [[ "$host" == "0.0.0.0" ]]; then probe_host=127.0.0.1; fi
if [[ "$probe_host" == *:* ]]; then probe_host="[$probe_host]"; fi
base_url="http://$probe_host:$port"
mkdir -p "$SIGNALFEED_HOME/logs"
if [[ -f "$SIGNALFEED_HOME/server.pid" ]] && kill -0 "$(cat "$SIGNALFEED_HOME/server.pid")" 2>/dev/null; then
  echo "SignalFeed is already running (PID $(cat "$SIGNALFEED_HOME/server.pid")). Stop it before changing host/port."
  exit 0
fi
# Refuse a live listener even when a PID file is missing.
python - "$host" "$port" <<'PYPORT'
import socket, sys
with socket.socket(socket.AF_INET6 if ':' in sys.argv[1] else socket.AF_INET) as sock:
    sock.bind((sys.argv[1], int(sys.argv[2])))
PYPORT
nohup setsid python -m signalfeed.cli serve --host "$host" --port "$port" >> "$SIGNALFEED_HOME/logs/server.log" 2>&1 &
echo "$!" > "$SIGNALFEED_HOME/server.pid"
for attempt in {1..1200}; do
  if ! kill -0 "$(cat "$SIGNALFEED_HOME/server.pid")" 2>/dev/null; then
    echo "SignalFeed exited; inspect $SIGNALFEED_HOME/logs/server.log" >&2
    exit 1
  fi
  if curl --noproxy '*' --max-time 1 -fsS "$base_url/api/ready" >/dev/null 2>&1; then
    echo "SignalFeed ready: $base_url (listening on $host:$port)"
    exit 0
  fi
  sleep 0.1
done
echo "Startup did not become ready; inspect $SIGNALFEED_HOME/logs/server.log" >&2
exit 1
