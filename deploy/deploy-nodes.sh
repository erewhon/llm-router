#!/bin/bash
# Deploy llm-router code and restart services on all nodes.
#
# Usage:
#   ./deploy/deploy-nodes.sh              # Deploy to all nodes
#   ./deploy/deploy-nodes.sh agents       # Node agents only
#   ./deploy/deploy-nodes.sh dashboard    # Dashboard + proxy only
#   ./deploy/deploy-nodes.sh sync         # Sync code only (no restart)

set -euo pipefail

PROJ="/home/erewhon/Projects/erewhon/llm-router"
RSYNC_OPTS=(-av --delete --exclude='.git' --exclude='__pycache__' --exclude='.venv')

# GPU nodes run node agents
AGENT_NODES=(archimedes.local hypatia.local)
# delphi is localhost
DELPHI_IS_LOCAL=true

# euclid runs dashboard + proxy
DASHBOARD_HOST="euclid.local"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
RESET='\033[0m'

info()  { echo -e "${BLUE}==>${RESET} $*"; }
ok()    { echo -e "${GREEN}==>${RESET} $*"; }
err()   { echo -e "${RED}==>${RESET} $*" >&2; }

sync_node() {
    local host=$1
    info "Syncing code to $host..."
    rsync "${RSYNC_OPTS[@]}" "$PROJ/" "erewhon@${host}:${PROJ}/"
}

rebuild_venv() {
    local host=$1
    info "Syncing venv on $host..."
    ssh "$host" "cd $PROJ && /home/linuxbrew/.linuxbrew/bin/uv sync --all-extras -q"
}

restart_agent() {
    local host=$1
    info "Restarting node agent on $host..."
    ssh "$host" "sudo systemctl restart llm-router-agent"
    local status
    status=$(ssh "$host" "systemctl is-active llm-router-agent")
    if [[ "$status" == "active" ]]; then
        ok "Node agent on $host: active"
    else
        err "Node agent on $host: $status"
        return 1
    fi
}

deploy_agents() {
    # Remote GPU nodes
    for host in "${AGENT_NODES[@]}"; do
        sync_node "$host"
        rebuild_venv "$host"
        restart_agent "$host"
        echo ""
    done

    # delphi (localhost)
    if [[ "$DELPHI_IS_LOCAL" == "true" ]]; then
        info "Rebuilding venv on delphi (localhost)..."
        uv sync --all-extras -q 2>&1 | tail -1 || true
        info "Restarting node agent on delphi..."
        sudo systemctl restart llm-router-agent
        if systemctl is-active llm-router-agent >/dev/null 2>&1; then
            ok "Node agent on delphi: active"
        else
            err "Node agent on delphi: failed"
        fi
        echo ""
    fi
}

install_services() {
    local host=$1
    info "Installing service files on $host..."
    ssh "$host" "sudo cp $PROJ/deploy/litellm-proxy.service /etc/systemd/system/ && \
        sudo cp $PROJ/deploy/litellm-dashboard.service /etc/systemd/system/ && \
        sudo systemctl daemon-reload"
}

deploy_dashboard() {
    sync_node "$DASHBOARD_HOST"
    rebuild_venv "$DASHBOARD_HOST"
    install_services "$DASHBOARD_HOST"

    info "Restarting dashboard + proxy on $DASHBOARD_HOST..."
    ssh "$DASHBOARD_HOST" "sudo systemctl restart litellm-dashboard litellm-proxy"

    local dash_status proxy_status
    dash_status=$(ssh "$DASHBOARD_HOST" "systemctl is-active litellm-dashboard")
    proxy_status=$(ssh "$DASHBOARD_HOST" "systemctl is-active litellm-proxy")
    [[ "$dash_status" == "active" ]] && ok "Dashboard on $DASHBOARD_HOST: active" || err "Dashboard on $DASHBOARD_HOST: $dash_status"
    [[ "$proxy_status" == "active" ]] && ok "Proxy on $DASHBOARD_HOST: active" || err "Proxy on $DASHBOARD_HOST: $proxy_status"
    echo ""
}

MODE="${1:-all}"

case "$MODE" in
    agents)
        deploy_agents
        ;;
    dashboard)
        deploy_dashboard
        ;;
    sync)
        for host in "${AGENT_NODES[@]}" "$DASHBOARD_HOST"; do
            sync_node "$host"
        done
        ;;
    all)
        deploy_agents
        deploy_dashboard
        ok "All nodes deployed."
        ;;
    *)
        echo "Usage: $0 {all|agents|dashboard|sync}"
        exit 1
        ;;
esac
