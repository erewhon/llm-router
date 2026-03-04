#!/bin/bash
# Launch a multi-node vLLM cluster across two DGX Spark nodes.
#
# Usage:
#   ./launch.sh                  # uses defaults from env or config below
#   ./launch.sh --model Qwen/Qwen2.5-1.5B-Instruct --tp 2
#   ./launch.sh --stop           # tear down the cluster
#
# Prerequisites:
#   - Docker + nvidia-container-toolkit on both nodes
#   - SSH access from this machine to both nodes
#   - Same Docker image on both nodes

set -euo pipefail

# --- Configuration (override via env vars) ---
# Use .local (mDNS) hostnames to ensure LAN routing, not Tailscale.
HEAD_NODE="${HEAD_NODE:-archimedes.local}"
WORKER_NODE="${WORKER_NODE:-hypatia.local}"
HEAD_IP="${HEAD_IP:-192.168.42.134}"
WORKER_IP="${WORKER_IP:-192.168.42.52}"
ETH_IF="${ETH_IF:-enP7s7}"

IMAGE="${VLLM_IMAGE:-nvcr.io/nvidia/vllm:25.12.post1-py3}"
MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TP_SIZE="${TP_SIZE:-2}"
PORT="${VLLM_PORT:-5391}"
RAY_PORT="${RAY_PORT:-6379}"

CONTAINER_NAME="vllm-multinode"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# --- Parse args ---
STOP=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --tp) TP_SIZE="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --stop) STOP=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

DOCKER_OPTS=(
    --gpus all
    --network host
    --ipc=host
    --shm-size=64g
    --rm -d
    --name "$CONTAINER_NAME"
    -e "NCCL_SOCKET_IFNAME=$ETH_IF"
    -e "GLOO_SOCKET_IFNAME=$ETH_IF"
    -e "NCCL_IGNORE_CPU_AFFINITY=1"
    -e "NCCL_DEBUG=WARN"
    -v "$HF_CACHE:/root/.cache/huggingface"
)

stop_cluster() {
    echo "Stopping cluster..."
    ssh "$HEAD_NODE" "docker stop $CONTAINER_NAME 2>/dev/null" || true
    ssh "$WORKER_NODE" "docker stop $CONTAINER_NAME 2>/dev/null" || true
    echo "Cluster stopped."
}

if $STOP; then
    stop_cluster
    exit 0
fi

echo "=== Multi-node vLLM cluster ==="
echo "  Head:   $HEAD_NODE ($HEAD_IP)"
echo "  Worker: $WORKER_NODE ($WORKER_IP)"
echo "  Model:  $MODEL"
echo "  TP:     $TP_SIZE"
echo "  Image:  $IMAGE"
echo "  Port:   $PORT"
echo ""

# Stop any existing cluster
stop_cluster 2>/dev/null || true

# --- Start Ray head on head node ---
echo "Starting Ray head on $HEAD_NODE..."
ssh "$HEAD_NODE" docker run \
    "${DOCKER_OPTS[@]}" \
    -e "VLLM_HOST_IP=$HEAD_IP" \
    "$IMAGE" \
    bash -c "ray start --head --port=$RAY_PORT --node-ip-address=$HEAD_IP --block" &

sleep 5

# --- Start Ray worker on worker node ---
echo "Starting Ray worker on $WORKER_NODE..."
ssh "$WORKER_NODE" docker run \
    "${DOCKER_OPTS[@]}" \
    -e "VLLM_HOST_IP=$WORKER_IP" \
    "$IMAGE" \
    bash -c "ray start --address=$HEAD_IP:$RAY_PORT --node-ip-address=$WORKER_IP --block" &

sleep 10

# --- Wait for Ray cluster to form ---
echo "Waiting for Ray cluster (2 nodes)..."
for i in $(seq 1 30); do
    NODE_COUNT=$(ssh "$HEAD_NODE" "docker exec $CONTAINER_NAME ray status 2>/dev/null | grep -c 'node_' || echo 0")
    if [[ "$NODE_COUNT" -ge 2 ]]; then
        echo "Ray cluster ready: $NODE_COUNT nodes"
        break
    fi
    echo "  waiting... ($i/30, nodes: $NODE_COUNT)"
    sleep 5
done

if [[ "$NODE_COUNT" -lt 2 ]]; then
    echo "ERROR: Ray cluster did not form (only $NODE_COUNT nodes). Check logs:"
    echo "  ssh $HEAD_NODE docker logs $CONTAINER_NAME"
    echo "  ssh $WORKER_NODE docker logs $CONTAINER_NAME"
    exit 1
fi

# --- Launch vLLM serve on the head node ---
echo "Starting vLLM serve..."
ssh "$HEAD_NODE" docker exec -d "$CONTAINER_NAME" \
    bash -c "vllm serve '$MODEL' \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --distributed-executor-backend ray \
        2>&1 | tee /tmp/vllm-serve.log"

echo ""
echo "vLLM is starting on $HEAD_NODE:$PORT"
echo "  Model:  $MODEL"
echo "  TP:     $TP_SIZE"
echo ""
echo "Monitor logs:  ssh $HEAD_NODE docker exec $CONTAINER_NAME tail -f /tmp/vllm-serve.log"
echo "Test:          curl http://$HEAD_NODE:$PORT/v1/models"
echo "Stop:          $0 --stop"
