#!/usr/bin/env bash
# vast_train.sh — lightweight wrapper for running ai-t9-run on a Vast.ai instance.
#
# This script is a convenience wrapper for manual SSH use.  The recommended
# approach is to use vast_orchestrate.py, which handles provisioning, file
# upload, and teardown automatically.
#
# Usage:
#   # Upload your config, then:
#   bash vast_train.sh /root/train_config.yaml
#
#   # With step overrides:
#   bash vast_train.sh /root/train_config.yaml --step train
#
# The script:
#   1. Installs ai-t9[train,data] (if not already installed)
#   2. Runs ai-t9-run with the given config file
#
# S3 credentials must be set as environment variables so ${VAR} references
# in the YAML config resolve correctly.

set -euo pipefail

CONFIG="${1:?Usage: vast_train.sh <config.yaml> [--step/--skip flags...]}"
shift  # remaining args forwarded to ai-t9-run

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
log "Installing ai-t9[train,data]..."
pip install --quiet "ai-t9[train,data]" pyyaml 2>&1 | tail -3

# Check for GPU
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    log "GPU detected: ${GPU_NAME}"
else
    log "WARNING: No CUDA GPU found — training will use CPU"
fi

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
log "Running: ai-t9-run ${CONFIG} $*"
ai-t9-run "${CONFIG}" "$@"

log "Done."
