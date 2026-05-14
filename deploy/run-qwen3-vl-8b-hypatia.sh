#!/bin/bash
# Launch llama.cpp CUDA serving Qwen3-VL-8B-Instruct Q8_0 on hypatia as a
# co-resident alongside the existing SGLang Qwen3.6-35B-A3B container.
#
# Memory math (verified 2026-06-03):
#   hypatia total unified VRAM:  ~119.6 GB
#   SGLang Qwen3.6 reservation:  ~92 GB (mem-fraction-static 0.70 * 128)
#   Free pool:                   ~27 GB
#   This server (Q8_0 8B + 32K ctx + mmproj): expected ~12-15 GB
#
# Comfortable margin. Co-resident, no shutdowns required.
#
# Run on hypatia:
#   ./run-qwen3-vl-8b-hypatia.sh
#
# Prereqs:
#   - Model files at MODEL_DIR (see deploy/setup-qwen3-vl-8b-hypatia.sh)
#   - UFW: sudo ufw allow from 192.168.42.0/24 to any port 5398 \
#       comment "Qwen3-VL-8B llama.cpp"
set -euo pipefail

NAME="${NAME:-llamacpp-qwen3-vl-8b}"
IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
MODEL_DIR="${MODEL_DIR:-/home/erewhon/models/qwen3-vl-8b}"
PORT="${PORT:-5398}"
CTX="${CTX:-32768}"

GGUF_NAME="${GGUF_NAME:-Qwen3VL-8B-Instruct-Q8_0.gguf}"
MMPROJ_NAME="${MMPROJ_NAME:-mmproj-Qwen3VL-8B-Instruct-F16.gguf}"

GGUF="${MODEL_DIR}/${GGUF_NAME}"
MMPROJ="${MODEL_DIR}/${MMPROJ_NAME}"

for f in "$GGUF" "$MMPROJ"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing $f" >&2
        echo "Run deploy/setup-qwen3-vl-8b-hypatia.sh first to download." >&2
        exit 1
    fi
done

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "Removing existing container ${NAME}..."
    docker rm -f "${NAME}"
fi

exec docker run -d \
    --name "${NAME}" \
    --network host \
    --restart unless-stopped \
    --gpus all \
    -v "${MODEL_DIR}":/models:ro \
    "${IMAGE}" \
    --model "/models/${GGUF_NAME}" \
    --mmproj "/models/${MMPROJ_NAME}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    -ngl 999 \
    --ctx-size "${CTX}" \
    --jinja \
    --metrics \
    --no-webui
