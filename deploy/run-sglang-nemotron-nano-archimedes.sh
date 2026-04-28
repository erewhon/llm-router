#!/bin/bash
# Launch SGLang serving NVIDIA Nemotron-3 Nano 30B-A3B NVFP4 on archimedes.
#
# Smaller sibling of Nemotron-3 Super: same nemotron_h architecture (Mamba-2
# + MoE + attention hybrid), 30B total / 3B active. Faster end-to-end than
# Super, at the cost of less raw capability.
#
# Uses the SGLang dev image with nemotron_h architecture and the nemotron_3
# reasoning parser. Default reasoning mode is ON; clients can disable via
# `chat_template_kwargs.enable_thinking: false` per request.
#
# Run on archimedes:
#   ./run-sglang-nemotron-nano-archimedes.sh
set -euo pipefail

NAME="sglang-nemotron-nano"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3}"
MODEL_DIR="${MODEL_DIR:-/home/erewhon/models/nemotron-3-nano-nvfp4}"
MEM_FRACTION="${MEM_FRACTION:-0.40}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-8}"
PORT="${PORT:-5391}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
    echo "ERROR: ${MODEL_DIR}/config.json not found. Are the weights staged?" >&2
    exit 1
fi

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "Removing existing container ${NAME}..."
    docker rm -f "${NAME}"
fi

exec docker run -d \
    --name "${NAME}" \
    --network host \
    --restart unless-stopped \
    --gpus all \
    --shm-size=16g \
    -v "${MODEL_DIR}":/model:ro \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
        --model-path /model \
        --served-model-name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 \
        --port "${PORT}" \
        --host 0.0.0.0 \
        --trust-remote-code \
        --quantization modelopt_fp4 \
        --attention-backend flashinfer \
        --mem-fraction-static "${MEM_FRACTION}" \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --context-length "${CONTEXT_LENGTH}" \
        --tool-call-parser qwen3_coder \
        --reasoning-parser nemotron_3 \
        --disable-piecewise-cuda-graph \
        --enable-metrics
