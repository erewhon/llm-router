#!/bin/bash
# Launch SGLang serving NVIDIA Nemotron-3-Super 120B (A12B) NVFP4 on archimedes.
#
# Replaces Qwen3.6-27B-FP8 and Qwen3-VL-32B-Thinking. Inherits the
# thinker/research/coder-alt aliases from the retired Qwen3.6-27B.
#
# Requires the SGLang dev image with the nemotron_h architecture and the
# super_v3/nemotron_3 reasoning parser baked in:
#   lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3
#
# Model weights expected at MODEL_DIR (default /home/erewhon/models/nemotron-3-super-nvfp4).
# Rsync from euclid NAS to that path before starting; HF cache layout is not used.
#
# Run on archimedes:
#   ./run-sglang-nemotron-super-archimedes.sh
set -euo pipefail

NAME="sglang-nemotron-super"
IMAGE="${IMAGE:-lmsysorg/sglang:dev-cu13-nemotronh-nano-omni-reasoning-v3}"
MODEL_DIR="${MODEL_DIR:-/home/erewhon/models/nemotron-3-super-nvfp4}"
MEM_FRACTION="${MEM_FRACTION:-0.80}"
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
        --served-model-name nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
        --port "${PORT}" \
        --host 0.0.0.0 \
        --trust-remote-code \
        --quantization modelopt_fp4 \
        --mem-fraction-static "${MEM_FRACTION}" \
        --max-running-requests "${MAX_RUNNING_REQUESTS}" \
        --context-length "${CONTEXT_LENGTH}" \
        --tool-call-parser qwen3_coder \
        --reasoning-parser nemotron_3 \
        --disable-piecewise-cuda-graph \
        --enable-metrics
