#!/usr/bin/env bash
export SIGNALFEED_HOME="${SIGNALFEED_HOME:-/data1/zhuzy/signalfeed}"
config="${SIGNALFEED_ENV_FILE:-$SIGNALFEED_HOME/deploy.env}"
if [[ -f "$config" ]]; then
  source "$config"
fi
export HF_HOME="$SIGNALFEED_HOME/hf"
export UV_CACHE_DIR="$SIGNALFEED_HOME/uv-cache"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export PATH="$SIGNALFEED_HOME/venv/bin:$PATH"
