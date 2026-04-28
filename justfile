# LLM Router operations.
# Recipes wrap existing shell scripts and one-off ops. Run `just` (no args) to list.

# Show available recipes
default:
    @just --list

# ─── Deploy ────────────────────────────────────────────────────────────────────

# Full deploy: code + venv + restart on every node (agents + dashboard/proxy)
deploy:
    ./deploy/deploy-nodes.sh all

# Node agents only (every GPU node)
deploy-agents:
    ./deploy/deploy-nodes.sh agents

# Dashboard + LiteLLM proxy on euclid
deploy-dashboard:
    ./deploy/deploy-nodes.sh dashboard

# Sync code to every node (no venv rebuild, no service restart)
sync:
    ./deploy/deploy-nodes.sh sync

# ─── Config ────────────────────────────────────────────────────────────────────

# Regenerate deploy/litellm/config.yaml from models.yaml
generate-config mode="default":
    uv run python -m llm_router.generate_config --mode {{mode}}

# After editing models.yaml: regen config, push everywhere, restart services
push mode="default": (generate-config mode) deploy

# ─── Inference mode (Sparks) ───────────────────────────────────────────────────

# default | big | off | status (passes through to deploy/spark-mode.sh)
spark-mode mode="status":
    ./deploy/spark-mode.sh {{mode}}

# ─── SGLang containers ─────────────────────────────────────────────────────────

# Launch SGLang Qwen3.6-27B-FP8 on archimedes (dense, thinking on)
sglang-archimedes:
    ./deploy/run-sglang-qwen36-archimedes.sh

# Launch SGLang Qwen3.6-35B-A3B-FP8 on hypatia (MoE, nothink)
sglang-hypatia:
    ./deploy/run-sglang-qwen36-hypatia.sh

# ─── Service restarts ──────────────────────────────────────────────────────────

restart-proxy:
    ssh erewhon@euclid.local "sudo systemctl restart litellm-proxy"

restart-dashboard:
    ssh erewhon@euclid.local "sudo systemctl restart litellm-dashboard"

restart-tool-proxy:
    ssh erewhon@euclid.local "sudo systemctl restart llm-router-tool-proxy"

# host = archimedes | hypatia | euclid | delphi
restart-agent host:
    ssh erewhon@{{host}}.local "sudo systemctl restart llm-router-agent"

# ─── Probes ────────────────────────────────────────────────────────────────────

# What each node agent thinks is running
probe-agents:
    #!/usr/bin/env bash
    for h in archimedes hypatia delphi euclid; do
        echo "=== $h ==="
        curl -sS -m 3 "http://$h.local:8100/models" | python3 -m json.tool 2>/dev/null || echo "(unreachable)"
    done

# What the LiteLLM proxy is serving (alias list)
probe-proxy:
    @curl -sS https://llm.peacock-bramble.ts.net/v1/models -H "Authorization: Bearer sk-litellm-master" | python3 -m json.tool

# What the Tailscale Service config looks like
probe-tailscale:
    @ssh erewhon@euclid.local "sudo tailscale serve --service=svc:llm get-config"

# Where the deployed copy of models.yaml is on each node and its mtime
probe-deployed-models:
    #!/usr/bin/env bash
    for h in archimedes hypatia euclid; do
        echo -n "$h: "
        ssh -o ConnectTimeout=3 erewhon@$h.local "stat -c '%y' /home/erewhon/Projects/erewhon/llm-router/models.yaml 2>&1" 2>&1
    done

# ─── Tailscale ─────────────────────────────────────────────────────────────────

# Re-apply svc:llm path handlers on euclid (idempotent; also runs at boot)
tailscale-restore:
    ssh erewhon@euclid.local "sudo /usr/local/bin/tailscale-serve-restore-svc-llm.sh"

# ─── Code quality ──────────────────────────────────────────────────────────────

test:
    uv run pytest

lint:
    uv run ruff check src tests

fmt:
    uv run ruff format src tests

fmt-check:
    uv run ruff format --check src tests

# ─── One-time setup (kept for discoverability) ─────────────────────────────────

setup-intel-gpu:
    ./deploy/setup-intel-gpu.sh

setup-security:
    ./deploy/setup-security.sh
