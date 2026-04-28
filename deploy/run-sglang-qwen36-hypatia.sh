#!/bin/bash
# Launch SGLang serving Qwen3.6-35B-A3B-FP8 (MoE coder, thinking OFF) on hypatia.
#
# Hypatia runs SGLang as its only inference workload (creative-content models
# all migrated to delphi 2026-05-12/13). MEM_FRACTION reduced from 0.75 to
# 0.70 on 2026-05-13 after a GPU OOM crash mid-use — NVRM driver was logging
# NV_ERR_NO_MEMORY before SGLang shut itself down cleanly. 0.70 leaves more
# runtime headroom for KV cache spikes + MoE expert routing intermediates at
# 262K context.
#
# Run on hypatia:
#   ./run-sglang-qwen36-hypatia.sh
set -euo pipefail

NAME="sglang-qwen36"
IMAGE="lmsysorg/sglang:v0.5.10.post1-cu130"
HF_CACHE="${HF_CACHE:-/home/erewhon/.cache/huggingface}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-/etc/llm-router/qwen36-chat-template-nothink.jinja}"
MEM_FRACTION="${MEM_FRACTION:-0.70}"
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
        --model-path Qwen/Qwen3.6-35B-A3B-FP8 \
        --port "${PORT}" \
        --host 0.0.0.0 \
        --chat-template /opt/chat-template.jinja \
        --tool-call-parser qwen3_coder \
        --mem-fraction-static "${MEM_FRACTION}" \
        --context-length "${CONTEXT_LENGTH}" \
        --trust-remote-code \
        --enable-metrics
