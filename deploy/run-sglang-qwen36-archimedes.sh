#!/bin/bash
# Launch SGLang serving Qwen3.6-27B-FP8 (dense thinker) on archimedes.
#
# Coexists with two other llama.cpp CUDA models on the GB10 (UI-TARS and
# Qwen3-VL-32B-Thinking) which together occupy ~35GB. Qwen3-Coder-FIM was
# disabled 2026-05-10 to free room for SGLang's 0.50 reservation.
#
# Run on archimedes:
#   ./run-sglang-qwen36-archimedes.sh
set -euo pipefail

NAME="sglang-qwen36-27b"
IMAGE="lmsysorg/sglang:v0.5.10.post1-cu130"
HF_CACHE="${HF_CACHE:-/home/erewhon/.cache/huggingface}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/etc/llm-router/qwen36-chat-template.jinja}"
MEM_FRACTION="${MEM_FRACTION:-0.60}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"
PORT="${PORT:-5391}"

if docker ps -a --format '{{.Names}}' | grep -qx "${NAME}"; then
    echo "Removing existing container ${NAME}..."
    docker rm -f "${NAME}"
fi

exec docker run -d \
    --name "${NAME}" \
    --network host \
    --restart unless-stopped \
    --gpus all \
    --shm-size=10g \
    -v "${CHAT_TEMPLATE}":/opt/chat-template.jinja:ro \
    -v "${HF_CACHE}":/root/.cache/huggingface \
    "${IMAGE}" \
    python3 -m sglang.launch_server \
        --model-path Qwen/Qwen3.6-27B-FP8 \
        --port "${PORT}" \
        --host 0.0.0.0 \
        --reasoning-parser qwen3 \
        --tool-call-parser qwen3_coder \
        --mem-fraction-static "${MEM_FRACTION}" \
        --context-length "${CONTEXT_LENGTH}" \
        --trust-remote-code \
        --enable-metrics
