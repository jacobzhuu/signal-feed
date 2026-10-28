#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
if [[ -f "$SIGNALFEED_HOME/server.pid" ]]; then
  pid="$(cat "$SIGNALFEED_HOME/server.pid")"
  if [[ -r "/proc/$pid/cmdline" ]] && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'signalfeed serve'; then
    kill "$pid"
    for attempt in {1..100}; do
      if ! kill -0 "$pid" 2>/dev/null; then break; fi
      if [[ -r "/proc/$pid/stat" ]] && [[ "$(awk '{print $3}' "/proc/$pid/stat")" == "Z" ]]; then break; fi
      sleep 0.1
    done
  fi
  rm "$SIGNALFEED_HOME/server.pid"
fi
