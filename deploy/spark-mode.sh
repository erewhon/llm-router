#!/bin/bash
# Switch between inference modes on the two DGX Spark nodes.
#
# Modes:
#   off      - Stop all inference on both Sparks
#   default  - Individual models: LMStudio + nspawn on archimedes,
#              vLLM + ComfyUI on hypatia
#   big      - Single Qwen3.5-397B-A17B across both Sparks (TP=2, Ray)
#   status   - Show what's running on each Spark
#
# Usage:
#   ./spark-mode.sh status
#   ./spark-mode.sh off
#   ./spark-mode.sh default
#   ./spark-mode.sh big
#   ./spark-mode.sh big --model Qwen/Other-Model --image vllm/vllm-openai:latest

set -euo pipefail

# --- Configuration ---
ARCHIMEDES="${ARCHIMEDES:-archimedes.local}"
HYPATIA="${HYPATIA:-hypatia.local}"
# QSFP direct-connect IPs (200GbE point-to-point link)
ARCHIMEDES_IP="${ARCHIMEDES_IP:-192.168.100.10}"
HYPATIA_IP="${HYPATIA_IP:-192.168.100.11}"
ETH_IF="${ETH_IF:-enp1s0f0np0}"

BIG_MODEL="${BIG_MODEL:-Qwen/Qwen3.5-122B-A10B-FP8}"
BIG_IMAGE="${BIG_IMAGE:-vllm/vllm-openai:cu130-nightly-old}"
BIG_PORT="${BIG_PORT:-5391}"
RAY_PORT="${RAY_PORT:-6379}"
TP_SIZE="${TP_SIZE:-2}"

# Local SSD HF cache (pre-synced from NFS for fast loading + proper mmap)
HF_CACHE="/opt/llm-vllm/hf-cache/hub"
MULTINODE_CONTAINER="vllm-multinode"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

PROJ="/home/erewhon/Projects/erewhon/llm-router"
EUCLID="${EUCLID:-euclid.local}"
UV="/home/linuxbrew/.linuxbrew/bin/uv"

info()  { echo -e "${BLUE}==>${RESET} $*"; }
ok()    { echo -e "${GREEN}==>${RESET} $*"; }
warn()  { echo -e "${YELLOW}==>${RESET} $*"; }

regenerate_and_deploy() {
    local mode=$1
    info "Regenerating LiteLLM config (mode: $mode)..."
    cd "$PROJ"
    $UV run llm-router-generate --mode "$mode"
    info "Deploying config to euclid and restarting proxy..."
    scp -q deploy/litellm/config.yaml "erewhon@${EUCLID}:${PROJ}/deploy/litellm/config.yaml"
    ssh "$EUCLID" "sudo systemctl restart litellm-proxy"
    # Restart tool proxy so it picks up the new model registry for backend routing
    sudo systemctl restart llm-router-tool-proxy 2>/dev/null || true
    ok "LiteLLM proxy + tool proxy restarted with $mode mode config"
}
err()   { echo -e "${RED}==>${RESET} $*" >&2; }

ssh_cmd() {
    local host=$1; shift
    ssh -o ConnectTimeout=5 "$host" "$@"
}

# --- Stop functions ---

stop_lmstudio() {
    local host=$1 name=$2
    info "[$name] Unloading LMStudio models..."
    ssh_cmd "$host" "~/.lmstudio/bin/lms unload --all 2>/dev/null" || true
    info "[$name] Stopping LMStudio server..."
    ssh_cmd "$host" "~/.lmstudio/bin/lms server stop 2>/dev/null" || true
}

stop_nspawn() {
    local host=$1 name=$2
    info "[$name] Stopping nspawn containers..."
    ssh_cmd "$host" "sudo machinectl stop llm 2>/dev/null" || true
    ssh_cmd "$host" "sudo machinectl stop svc-sys-llm-vpn 2>/dev/null" || true
    # Wait for containers to actually stop
    for i in $(seq 1 12); do
        local count
        count=$(ssh_cmd "$host" "sudo machinectl list --no-legend 2>/dev/null | wc -l" 2>/dev/null || echo 0)
        if [[ "$count" -eq 0 ]]; then
            break
        fi
        sleep 5
    done
}

stop_vllm_units() {
    local host=$1 name=$2
    info "[$name] Stopping vLLM systemd units..."
    local units
    units=$(ssh_cmd "$host" "systemctl list-units 'vllm@*' --no-legend --plain 2>/dev/null | awk '{print \$1}'" || true)
    if [[ -n "$units" ]]; then
        for unit in $units; do
            ssh_cmd "$host" "sudo systemctl stop $unit 2>/dev/null" || true
        done
    fi
}

stop_comfyui() {
    local host=$1 name=$2
    info "[$name] Stopping ComfyUI..."
    ssh_cmd "$host" "sudo systemctl stop comfyui 2>/dev/null" || true
}

stop_multinode() {
    info "Stopping multi-node vLLM cluster..."
    ssh_cmd "$ARCHIMEDES" "docker stop $MULTINODE_CONTAINER 2>/dev/null" || true
    ssh_cmd "$HYPATIA" "docker stop $MULTINODE_CONTAINER 2>/dev/null" || true
}

stop_default_containers() {
    ssh_cmd "$ARCHIMEDES" "docker stop vllm-default 2>/dev/null" || true
    ssh_cmd "$HYPATIA" "docker stop vllm-default 2>/dev/null" || true
}

stop_all() {
    local host=$1 name=$2
    stop_multinode
    stop_default_containers
    stop_vllm_units "$host" "$name"
    stop_lmstudio "$host" "$name"
    stop_nspawn "$host" "$name"
    stop_comfyui "$host" "$name"
}

# --- Start functions ---

start_default_archimedes() {
    local model="Qwen/Qwen3-Coder-30B-A3B-Instruct"
    local image="$BIG_IMAGE"
    local port=5391

    info "[archimedes] Starting vLLM: $model"
    ssh_cmd "$ARCHIMEDES" "docker run --gpus all --network host --ipc=host --shm-size=16g \
        --rm -d --entrypoint bash --name vllm-default \
        -e HF_HUB_OFFLINE=1 \
        -v $HF_CACHE:/root/.cache/huggingface/hub:ro \
        $image -c 'vllm serve $model \
            --host 0.0.0.0 --port $port \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --reasoning-parser qwen3 \
            --gpu-memory-utilization 0.90 --enforce-eager \
            --max-model-len 131072 \
            2>&1 | tee /tmp/vllm-serve.log'"

    ok "[archimedes] Default mode started (Qwen3-Coder-30B-A3B on port $port)"
}

start_default_hypatia() {
    local model="Qwen/Qwen3.5-27B"
    local image="$BIG_IMAGE"
    local port=5391

    info "[hypatia] Starting vLLM: $model"
    ssh_cmd "$HYPATIA" "docker run --gpus all --network host --ipc=host --shm-size=16g \
        --rm -d --entrypoint bash --name vllm-default \
        -e HF_HUB_OFFLINE=1 \
        -v $HF_CACHE:/root/.cache/huggingface/hub:ro \
        $image -c 'vllm serve $model \
            --host 0.0.0.0 --port $port \
            --enable-auto-tool-choice --tool-call-parser qwen3_xml \
            --reasoning-parser qwen3 \
            --gpu-memory-utilization 0.90 --enforce-eager \
            --max-model-len 131072 \
            2>&1 | tee /tmp/vllm-serve.log'"

    info "[hypatia] Starting ComfyUI..."
    ssh_cmd "$HYPATIA" "sudo systemctl start comfyui 2>/dev/null" || true

    ok "[hypatia] Default mode started (Qwen3.5-27B on port $port + ComfyUI)"
}

start_big() {
    local model="$BIG_MODEL"
    local image="$BIG_IMAGE"
    local port="$BIG_PORT"

    echo ""
    info "Starting multi-node vLLM cluster"
    echo -e "  ${CYAN}Head:${RESET}   $ARCHIMEDES ($ARCHIMEDES_IP)"
    echo -e "  ${CYAN}Worker:${RESET} $HYPATIA ($HYPATIA_IP)"
    echo -e "  ${CYAN}Model:${RESET}  $model"
    echo -e "  ${CYAN}Image:${RESET}  $image"
    echo -e "  ${CYAN}TP:${RESET}     $TP_SIZE"
    echo -e "  ${CYAN}Port:${RESET}   $port"
    echo ""

    # Build docker run command as a string to avoid SSH array quoting issues
    local docker_common="docker run --gpus all --network host --ipc=host --shm-size=64g"
    docker_common+=" --rm -d --entrypoint bash --name $MULTINODE_CONTAINER"
    docker_common+=" -e NCCL_SOCKET_IFNAME=$ETH_IF -e GLOO_SOCKET_IFNAME=$ETH_IF"
    docker_common+=" -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN -e HF_HUB_OFFLINE=1"
    # Disable Ray memory monitor — unified memory GPUs confuse it (CUDA allocs look like RAM usage)
    docker_common+=" -e RAY_memory_monitor_refresh_ms=0"
    # Increase Ray compiled DAG timeout — 300s default is too short for long-context 122B generation
    docker_common+=" -e RAY_CGRAPH_get_timeout=3600"
    docker_common+=" -v $HF_CACHE:/root/.cache/huggingface/hub:ro"

    # Start Ray head on archimedes
    info "Starting Ray head on archimedes..."
    ssh_cmd "$ARCHIMEDES" "$docker_common -e VLLM_HOST_IP=$ARCHIMEDES_IP $image -c 'ray start --head --port=$RAY_PORT --node-ip-address=$ARCHIMEDES_IP --block'"

    sleep 8

    # Start Ray worker on hypatia
    info "Starting Ray worker on hypatia..."
    ssh_cmd "$HYPATIA" "$docker_common -e VLLM_HOST_IP=$HYPATIA_IP $image -c 'ray start --address=$ARCHIMEDES_IP:$RAY_PORT --node-ip-address=$HYPATIA_IP --block'"

    # Wait for Ray cluster
    info "Waiting for Ray cluster (2 nodes)..."
    local node_count=0
    for i in $(seq 1 60); do
        node_count=$(ssh_cmd "$ARCHIMEDES" \
            "docker exec $MULTINODE_CONTAINER ray status 2>/dev/null | grep -c node_ || echo 0" \
            2>/dev/null | tail -1) || node_count=0
        if [[ "$node_count" -ge 2 ]]; then
            ok "Ray cluster ready: $node_count nodes"
            break
        fi
        echo "  waiting... ($i/60, nodes: $node_count)"
        sleep 5
    done

    if [[ "$node_count" -lt 2 ]]; then
        err "Ray cluster did not form (only $node_count nodes)"
        echo "  Head logs:   ssh $ARCHIMEDES docker logs $MULTINODE_CONTAINER"
        echo "  Worker logs: ssh $HYPATIA docker logs $MULTINODE_CONTAINER"
        return 1
    fi

    # Launch vLLM serve with aliases so LiteLLM can route coder/thinker/vision
    info "Starting vLLM serve..."
    ssh_cmd "$ARCHIMEDES" "docker exec -d $MULTINODE_CONTAINER bash -c 'vllm serve $model \
        --host 0.0.0.0 --port $port \
        --tensor-parallel-size $TP_SIZE \
        --distributed-executor-backend ray \
        --served-model-name $model coder thinker vision \
        --enable-auto-tool-choice --tool-call-parser qwen3_xml \
        --reasoning-parser qwen3 \
        --gpu-memory-utilization 0.90 --enforce-eager \
        --kv-cache-dtype fp8 \
        --max-model-len 262144 \
        2>&1 | tee /tmp/vllm-serve.log'"

    echo ""
    ok "vLLM starting on $ARCHIMEDES:$port"
    echo -e "  ${CYAN}Aliases:${RESET} coder, thinker, vision"
    echo -e "  ${CYAN}Monitor:${RESET} ssh $ARCHIMEDES docker exec $MULTINODE_CONTAINER tail -f /tmp/vllm-serve.log"
    echo -e "  ${CYAN}Test:${RESET}    curl http://$ARCHIMEDES:$port/v1/models"
    echo -e "  ${CYAN}Stop:${RESET}    $0 off"
}

# --- Status ---

show_status() {
    echo -e "${BOLD}=== Spark Cluster Status ===${RESET}"
    echo ""

    for host_info in "$ARCHIMEDES:archimedes" "$HYPATIA:hypatia"; do
        local host="${host_info%%:*}"
        local name="${host_info##*:}"
        echo -e "${BOLD}--- $name ($host) ---${RESET}"

        # Memory
        local mem
        mem=$(ssh_cmd "$host" "awk '/MemTotal/{t=\$2} /MemAvailable/{a=\$2} END{printf \"%.0f GB free / %.0f GB total\", a/1048576, t/1048576}' /proc/meminfo" 2>/dev/null) || mem="unreachable"
        echo -e "  ${CYAN}Memory:${RESET} $mem"

        # LMStudio
        local lms_status
        lms_status=$(ssh_cmd "$host" "~/.lmstudio/bin/lms ps --json 2>/dev/null | python3 -c \"
import sys, json
data = json.load(sys.stdin)
if not data: print('no models loaded')
else: print(', '.join(m.get('identifier','?') + ' (' + m.get('size','?') + ')' for m in data))
\" 2>/dev/null" || echo "not running")
        echo -e "  ${CYAN}LMStudio:${RESET} $lms_status"

        # vLLM systemd units
        local vllm_units
        vllm_units=$(ssh_cmd "$host" "systemctl list-units 'vllm@*' --no-legend --plain 2>/dev/null | awk '{print \$1, \$3}'" || true)
        if [[ -n "$vllm_units" ]]; then
            echo -e "  ${CYAN}vLLM units:${RESET}"
            while IFS= read -r line; do
                echo "    $line"
            done <<< "$vllm_units"
        fi

        # Docker containers (inference-related)
        local docker_ps
        docker_ps=$(ssh_cmd "$host" "docker ps --format '{{.Names}} ({{.Image}})' 2>/dev/null | grep -iE 'vllm|multinode'" || true)
        if [[ -n "$docker_ps" ]]; then
            echo -e "  ${CYAN}Docker:${RESET}"
            while IFS= read -r line; do
                echo "    $line"
            done <<< "$docker_ps"
        fi

        # nspawn containers
        local nspawn
        nspawn=$(ssh_cmd "$host" "sudo machinectl list --no-legend 2>/dev/null" || true)
        if [[ -n "$nspawn" ]]; then
            echo -e "  ${CYAN}nspawn:${RESET}"
            while IFS= read -r line; do
                echo "    $line"
            done <<< "$nspawn"
        fi

        # ComfyUI
        local comfy
        comfy=$(ssh_cmd "$host" "systemctl is-active comfyui 2>/dev/null" || echo "inactive")
        if [[ "$comfy" == "active" ]]; then
            echo -e "  ${CYAN}ComfyUI:${RESET} running"
        fi

        echo ""
    done
}

# --- Parse args ---
MODE="${1:-status}"
shift || true

while [[ $# -gt 0 ]]; do
    case $1 in
        --model) BIG_MODEL="$2"; shift 2 ;;
        --image) BIG_IMAGE="$2"; shift 2 ;;
        --port)  BIG_PORT="$2"; shift 2 ;;
        --tp)    TP_SIZE="$2"; shift 2 ;;
        *) err "Unknown arg: $1"; exit 1 ;;
    esac
done

case "$MODE" in
    status)
        show_status
        ;;
    off)
        info "Switching to OFF mode — stopping everything on both Sparks"
        stop_all "$ARCHIMEDES" "archimedes"
        stop_all "$HYPATIA" "hypatia"
        ok "All inference stopped on both Sparks"
        echo ""
        show_status
        ;;
    default)
        info "Switching to DEFAULT mode"
        # Stop everything first
        stop_all "$ARCHIMEDES" "archimedes"
        stop_all "$HYPATIA" "hypatia"
        echo ""
        # Start default services
        start_default_archimedes
        echo ""
        start_default_hypatia
        echo ""
        regenerate_and_deploy "default"
        echo ""
        show_status
        ;;
    big)
        info "Switching to BIG mode ($BIG_MODEL)"
        # Stop everything first
        stop_all "$ARCHIMEDES" "archimedes"
        stop_all "$HYPATIA" "hypatia"
        echo ""
        start_big
        echo ""
        regenerate_and_deploy "big"
        ;;
    *)
        echo "Usage: $0 {status|off|default|big} [--model MODEL] [--image IMAGE] [--port PORT] [--tp N]"
        exit 1
        ;;
esac
