#!/usr/bin/env bash
# vast_train.sh — run ai-t9 training on a Vast.ai GPU instance.
#
# This script is designed to be uploaded and executed on the instance by
# vast_orchestrate.py, but can also be run manually:
#
#   bash vast_train.sh [OPTIONS]
#
# Required environment variables:
#   AI_T9_S3_ENDPOINT      e.g. https://s3.us-east-1.amazonaws.com
#   AI_T9_S3_BUCKET        e.g. my-ai-t9-bucket
#   AI_T9_S3_ACCESS_KEY    AWS / Cloudflare R2 access key
#   AI_T9_S3_SECRET_KEY    AWS / Cloudflare R2 secret key
#
# Optional environment variables (override defaults below):
#   VOCAB_REMOTE           remote path for vocab.json    (default: vocab/vocab.json)
#   PAIRS_REMOTE           remote path / prefix for pairs (default: pairs/pairs)
#   MODEL_REMOTE           remote path for model output  (default: models/model.npz)
#   NGRAM_REMOTE           remote path for bigram model  (default: ngrams/bigram.npz)
#   CORPUS_REMOTE          remote path / prefix for corpus files (default: "")
#   EPOCHS                 training epochs               (default: 5)
#   EMBED_DIM              embedding dimension           (default: 64)
#   BATCH_SIZE             micro-batch size              (default: 8192)
#   ACCUMULATE             gradient accumulation steps   (default: 4)
#   TEMPERATURE            in-batch negative temperature (default: 0.07)
#   WEIGHT_DECAY           AdamW weight decay            (default: 0.0001)
#   WARMUP_FRAC            LR warmup fraction            (default: 0.05)
#   LR                     peak learning rate            (default: 0.001)
#   CONTEXT_WINDOW         context words                 (default: 3)
#   MODEL_TYPE             dual-encoder or char-ngram    (default: char-ngram)
#   SHARD_SIZE             pairs per shard (prep step)   (default: 10000000)
#   SKIP_PREP              set to 1 to skip vocab+pairs step
#   SKIP_TRAIN             set to 1 to skip training (pairs-only run)
#   SAVE_NGRAM             set to 1 to train and upload bigram model
#
# Example:
#   export AI_T9_S3_ENDPOINT=https://s3.us-east-1.amazonaws.com
#   export AI_T9_S3_BUCKET=my-bucket
#   export AI_T9_S3_ACCESS_KEY=AKIA...
#   export AI_T9_S3_SECRET_KEY=...
#   export EPOCHS=10 EMBED_DIM=128
#   bash vast_train.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VOCAB_REMOTE="${VOCAB_REMOTE:-vocab/vocab.json}"
PAIRS_REMOTE="${PAIRS_REMOTE:-pairs/pairs}"
MODEL_REMOTE="${MODEL_REMOTE:-models/model.npz}"
NGRAM_REMOTE="${NGRAM_REMOTE:-ngrams/bigram.npz}"
CORPUS_REMOTE="${CORPUS_REMOTE:-}"

EPOCHS="${EPOCHS:-5}"
EMBED_DIM="${EMBED_DIM:-64}"
BATCH_SIZE="${BATCH_SIZE:-8192}"
ACCUMULATE="${ACCUMULATE:-4}"
TEMPERATURE="${TEMPERATURE:-0.07}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
WARMUP_FRAC="${WARMUP_FRAC:-0.05}"
LR="${LR:-0.001}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-3}"
MODEL_TYPE="${MODEL_TYPE:-char-ngram}"
SHARD_SIZE="${SHARD_SIZE:-10000000}"

SKIP_PREP="${SKIP_PREP:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SAVE_NGRAM="${SAVE_NGRAM:-0}"

WORKDIR="${HOME}/ai_t9_workdir"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log() { echo "[$(date '+%H:%M:%S')] $*"; }

require_env() {
    for var in "$@"; do
        if [[ -z "${!var:-}" ]]; then
            echo "ERROR: Required environment variable $var is not set." >&2
            exit 1
        fi
    done
}

s3_download() {
    # Download a single file from S3-compatible storage using aws CLI.
    local remote="$1" local_path="$2"
    aws s3 cp "s3://${AI_T9_S3_BUCKET}/${remote}" "${local_path}" \
        --endpoint-url "${AI_T9_S3_ENDPOINT}"
}

s3_upload() {
    local local_path="$1" remote="$2"
    aws s3 cp "${local_path}" "s3://${AI_T9_S3_BUCKET}/${remote}" \
        --endpoint-url "${AI_T9_S3_ENDPOINT}"
}

s3_sync_from() {
    # Sync a remote prefix to a local directory.
    local remote_prefix="$1" local_dir="$2"
    aws s3 sync "s3://${AI_T9_S3_BUCKET}/${remote_prefix}" "${local_dir}" \
        --endpoint-url "${AI_T9_S3_ENDPOINT}"
}

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
require_env AI_T9_S3_ENDPOINT AI_T9_S3_BUCKET AI_T9_S3_ACCESS_KEY AI_T9_S3_SECRET_KEY

# Configure aws CLI credentials
export AWS_ACCESS_KEY_ID="${AI_T9_S3_ACCESS_KEY}"
export AWS_SECRET_ACCESS_KEY="${AI_T9_S3_SECRET_KEY}"
export AWS_DEFAULT_REGION="${AI_T9_S3_REGION:-us-east-1}"

# ---------------------------------------------------------------------------
# Set up environment
# ---------------------------------------------------------------------------
log "=== Setting up environment ==="

mkdir -p "${WORKDIR}"
cd "${WORKDIR}"

# Create venv if it doesn't exist
if [[ ! -d ".venv" ]]; then
    log "Creating Python virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

log "Installing ai-t9[train]..."
pip install --quiet --upgrade "ai-t9[train]"

# Check for GPU
if python3 -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'" 2>/dev/null; then
    DEVICE="cuda"
    GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
    log "GPU detected: ${GPU_NAME}"
else
    DEVICE="cpu"
    log "WARNING: No CUDA GPU found, training on CPU (will be slow)"
fi

# ---------------------------------------------------------------------------
# Step 1: Download vocabulary (always needed)
# ---------------------------------------------------------------------------
log "=== Downloading vocabulary ==="
mkdir -p vocab
s3_download "${VOCAB_REMOTE}" vocab/vocab.json
log "  vocab.json downloaded"

# ---------------------------------------------------------------------------
# Step 2: Prep — download corpus + compute pairs (skippable)
# ---------------------------------------------------------------------------
if [[ "${SKIP_PREP}" == "1" ]]; then
    log "=== Skipping prep (SKIP_PREP=1) — downloading existing pairs ==="
    mkdir -p pairs
    s3_sync_from "$(dirname "${PAIRS_REMOTE}")" pairs/
    HAVE_SHARDS=$(ls pairs/pairs_*.npz 2>/dev/null | wc -l)
    if [[ "${HAVE_SHARDS}" -eq 0 ]]; then
        # Fall back to single-file pairs
        s3_download "${PAIRS_REMOTE}.npz" pairs/pairs.npz 2>/dev/null \
            || { log "ERROR: No pairs found at ${PAIRS_REMOTE} or ${PAIRS_REMOTE}.npz"; exit 1; }
    fi
else
    log "=== Step 2: Downloading corpus and computing pairs ==="
    if [[ -n "${CORPUS_REMOTE}" ]]; then
        mkdir -p corpus
        s3_sync_from "${CORPUS_REMOTE}" corpus/
        CORPUS_ARG="--corpus corpus/"
    else
        log "No CORPUS_REMOTE set; using NLTK Brown corpus for pairs"
        CORPUS_ARG=""
    fi

    mkdir -p pairs
    ai-t9-train \
        --vocab vocab/vocab.json \
        ${CORPUS_ARG} \
        --save-pairs pairs/pairs \
        --shard-size "${SHARD_SIZE}" \
        --pairs-only \
        --context-window "${CONTEXT_WINDOW}" \
        --output /dev/null

    log "Uploading pairs shards to S3..."
    for shard in pairs/pairs_*.npz; do
        remote_shard="$(dirname "${PAIRS_REMOTE}")/$(basename "${shard}")"
        s3_upload "${shard}" "${remote_shard}"
    done
fi

# ---------------------------------------------------------------------------
# Step 3: Train
# ---------------------------------------------------------------------------
if [[ "${SKIP_TRAIN}" == "1" ]]; then
    log "=== Skipping training (SKIP_TRAIN=1) ==="
else
    EFFECTIVE_BATCH=$(( BATCH_SIZE * ACCUMULATE ))
    log "=== Step 3: Training (effective batch=${EFFECTIVE_BATCH}) ==="

    # Determine pairs input — sharded dir if available, single file otherwise
    if ls pairs/pairs_*.npz &>/dev/null; then
        PAIRS_ARG="--pairs-dir pairs/"
    else
        PAIRS_ARG="--load-pairs pairs/pairs.npz"
    fi

    ai-t9-train \
        --vocab vocab/vocab.json \
        ${PAIRS_ARG} \
        --output model.npz \
        --model-type "${MODEL_TYPE}" \
        --epochs "${EPOCHS}" \
        --embed-dim "${EMBED_DIM}" \
        --context-window "${CONTEXT_WINDOW}" \
        --batch-size "${BATCH_SIZE}" \
        --accumulate-grad-batches "${ACCUMULATE}" \
        --temperature "${TEMPERATURE}" \
        --weight-decay "${WEIGHT_DECAY}" \
        --warmup-frac "${WARMUP_FRAC}" \
        --lr "${LR}" \
        --device "${DEVICE}" \
        --debug

    log "Uploading model to S3: ${MODEL_REMOTE}"
    s3_upload model.npz "${MODEL_REMOTE}"

    # Optional: train and upload bigram model
    if [[ "${SAVE_NGRAM}" == "1" ]] && [[ -n "${CORPUS_REMOTE}" ]]; then
        log "=== Training bigram model ==="
        ai-t9-train \
            --vocab vocab/vocab.json \
            --corpus corpus/ \
            --output /dev/null \
            --save-ngram bigram.npz \
            --epochs 1

        log "Uploading bigram model to S3: ${NGRAM_REMOTE}"
        s3_upload bigram.npz "${NGRAM_REMOTE}"
    fi
fi

log "=== Done ==="
