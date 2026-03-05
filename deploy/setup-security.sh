#!/usr/bin/env bash
# Setup script for LLM Router security hardening.
# Run as root on each node.
#
# Usage:
#   sudo ./setup-security.sh [--node-type TYPE]
#
# Node types:
#   delphi      — AMD GPU (ROCm), LMStudio, node agent
#   nvidia-node — NVIDIA GPU (archimedes, hypatia), node agent + vLLM
#   all         — install everything (default)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
NODE_TYPE="${1:---node-type}"
if [[ "$NODE_TYPE" == "--node-type" ]]; then
    NODE_TYPE="${2:-all}"
fi

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

if [[ $EUID -ne 0 ]]; then
    error "This script must be run as root (use sudo)"
    exit 1
fi

# --- Step 1: Create system users ---

info "Creating system user: llm-router"
if ! id -u llm-router &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --home-dir /nonexistent llm-router
    info "  Created llm-router user"
else
    info "  User llm-router already exists"
fi

if [[ "$NODE_TYPE" == "nvidia-node" || "$NODE_TYPE" == "all" ]]; then
    info "Creating system user: llm-vllm"
    if ! id -u llm-vllm &>/dev/null; then
        useradd --system --shell /usr/sbin/nologin --home-dir /nonexistent llm-vllm
        info "  Created llm-vllm user"
    else
        info "  User llm-vllm already exists"
    fi

    # Add llm-vllm to GPU groups
    info "Adding llm-vllm to GPU groups"
    if getent group video &>/dev/null; then
        usermod -aG video llm-vllm
        info "  Added to video group"
    fi
    if getent group render &>/dev/null; then
        usermod -aG render llm-vllm
        info "  Added to render group"
    fi
fi

if [[ "$NODE_TYPE" == "delphi" || "$NODE_TYPE" == "all" ]]; then
    # On delphi (AMD), llm-vllm might be needed too
    if ! id -u llm-vllm &>/dev/null; then
        useradd --system --shell /usr/sbin/nologin --home-dir /nonexistent llm-vllm
        info "  Created llm-vllm user (delphi)"
    fi
    if getent group video &>/dev/null; then
        usermod -aG video llm-vllm 2>/dev/null || true
    fi
    if getent group render &>/dev/null; then
        usermod -aG render llm-vllm 2>/dev/null || true
    fi
fi

# --- Step 2: Create directories ---

info "Creating directories"

# vLLM env file directory (node agent writes, vLLM reads)
mkdir -p /etc/llm-router/vllm-env
chown llm-router:llm-vllm /etc/llm-router/vllm-env
chmod 0750 /etc/llm-router/vllm-env
info "  /etc/llm-router/vllm-env (llm-router:llm-vllm, 0750)"

# StateDirectory is created by systemd, but ensure base exists
mkdir -p /var/lib/llm-router
chown llm-router:llm-router /var/lib/llm-router
chmod 0750 /var/lib/llm-router
info "  /var/lib/llm-router (llm-router:llm-router, 0750)"

# --- Step 3: Deploy systemd units ---

info "Deploying systemd service files"

cp "$SCRIPT_DIR/node-agent.service" /etc/systemd/system/llm-router-agent.service
info "  Installed llm-router-agent.service"

if [[ "$NODE_TYPE" == "nvidia-node" || "$NODE_TYPE" == "delphi" || "$NODE_TYPE" == "all" ]]; then
    cp "$SCRIPT_DIR/vllm@.service" /etc/systemd/system/vllm@.service
    info "  Installed vllm@.service"
fi

if [[ "$NODE_TYPE" == "delphi" || "$NODE_TYPE" == "all" ]]; then
    cp "$SCRIPT_DIR/lmstudio.service" /etc/systemd/system/lmstudio.service
    info "  Installed lmstudio.service"
fi

# --- Step 4: Deploy sudoers ---

info "Deploying sudoers rules"
cp "$SCRIPT_DIR/sudoers-llm-router" /etc/sudoers.d/llm-router
chmod 0440 /etc/sudoers.d/llm-router
# Validate
if visudo -c -f /etc/sudoers.d/llm-router &>/dev/null; then
    info "  Sudoers syntax OK"
else
    error "  Sudoers syntax check FAILED — removing to prevent lockout"
    rm -f /etc/sudoers.d/llm-router
    exit 1
fi

# --- Step 5: Reload systemd ---

info "Reloading systemd daemon"
systemctl daemon-reload

# --- Step 6: Enable services ---

info "Enabling services"
systemctl enable llm-router-agent.service
info "  Enabled llm-router-agent"

# --- Summary ---

echo ""
info "Setup complete for node type: $NODE_TYPE"
echo ""
echo "Next steps:"
echo "  1. Review service files in /etc/systemd/system/"
echo "  2. Start the node agent:  sudo systemctl restart llm-router-agent"
echo "  3. Verify sandboxing:     systemd-analyze security llm-router-agent"
if [[ "$NODE_TYPE" == "delphi" || "$NODE_TYPE" == "all" ]]; then
    echo "  4. Start LMStudio:       sudo systemctl start lmstudio"
fi
echo ""
echo "Verification commands:"
echo "  systemd-analyze security llm-router-agent"
echo "  systemd-analyze security vllm@test"
echo "  journalctl -u llm-router-agent -f"
