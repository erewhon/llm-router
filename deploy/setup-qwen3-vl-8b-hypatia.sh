#!/bin/bash
# One-shot: fetch Qwen3-VL-8B-Instruct GGUF + mmproj to hypatia.
# Idempotent — re-running re-checks but does not re-download existing files.
#
# Run on hypatia (or run on euclid then rsync; this script assumes hypatia).
#
# Total download: ~10 GB (Q8_0 ~9 GB + mmproj F16 ~1.5 GB).
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/erewhon/models/qwen3-vl-8b}"
HF_REPO="${HF_REPO:-Qwen/Qwen3-VL-8B-Instruct-GGUF}"
GGUF_NAME="${GGUF_NAME:-Qwen3VL-8B-Instruct-Q8_0.gguf}"
MMPROJ_NAME="${MMPROJ_NAME:-mmproj-Qwen3VL-8B-Instruct-F16.gguf}"

mkdir -p "${MODEL_DIR}"

if ! command -v hf >/dev/null 2>&1; then
    echo "hf CLI not found. Install with:" >&2
    echo "  uv tool install huggingface_hub" >&2
    exit 1
fi

echo "==> Downloading ${HF_REPO} files to ${MODEL_DIR}..."
hf download "${HF_REPO}" \
    --include "${GGUF_NAME}" \
    --include "${MMPROJ_NAME}" \
    --local-dir "${MODEL_DIR}"

echo
echo "Files in ${MODEL_DIR}:"
ls -lh "${MODEL_DIR}"
echo
echo "Next: ./run-qwen3-vl-8b-hypatia.sh"
