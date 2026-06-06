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

# ─── llama.cpp containers ─────────────────────────────────────────────────────

# Download Qwen3-VL-8B weights to hypatia (~10GB, one-time; idempotent)
qwen3-vl-setup:
    ssh erewhon@hypatia.local "bash /home/erewhon/Projects/erewhon/llm-router/deploy/setup-qwen3-vl-8b-hypatia.sh"

# Start Qwen3-VL-8B llama.cpp container on hypatia (port 5398)
qwen3-vl-up:
    ssh erewhon@hypatia.local "bash /home/erewhon/Projects/erewhon/llm-router/deploy/run-qwen3-vl-8b-hypatia.sh"

# Stop and remove the Qwen3-VL-8B container on hypatia
qwen3-vl-down:
    ssh erewhon@hypatia.local "docker rm -f llamacpp-qwen3-vl-8b 2>/dev/null || true"

# Tail Qwen3-VL-8B container logs (Ctrl-C to detach)
qwen3-vl-logs lines="80":
    ssh -t erewhon@hypatia.local "docker logs --tail {{lines}} -f llamacpp-qwen3-vl-8b"

# Show Qwen3-VL-8B container status on hypatia
qwen3-vl-status:
    @ssh erewhon@hypatia.local "docker ps -a --filter name=llamacpp-qwen3-vl-8b --format 'table {{ '{{' }}.Names{{ '}}' }}\t{{ '{{' }}.Status{{ '}}' }}\t{{ '{{' }}.Image{{ '}}' }}'"

# ─── Service restarts ──────────────────────────────────────────────────────────

# Restart the Go router on euclid (cut over from LiteLLM 2026-06-06).
# To roll back to LiteLLM, use `just rollback-to-litellm`.
restart-proxy:
    ssh erewhon@euclid.local "sudo systemctl restart llm-router-go"

# Roll back: stop Go router, restart LiteLLM. Use if Go router has a regression.
rollback-to-litellm:
    ssh erewhon@euclid.local "set -e ; \
        sudo systemctl stop llm-router-go ; \
        sudo systemctl disable llm-router-go ; \
        sudo systemctl enable --now litellm-proxy ; \
        systemctl is-active litellm-proxy"

# Re-cut to Go router after a rollback.
cutover-to-go-router:
    ssh erewhon@euclid.local "set -e ; \
        sudo systemctl stop litellm-proxy ; \
        sudo systemctl disable litellm-proxy ; \
        sudo systemctl enable --now llm-router-go ; \
        systemctl is-active llm-router-go"

# Tail Go router logs (Ctrl-C detaches).
proxy-logs lines="80":
    ssh -t erewhon@euclid.local "sudo journalctl -u llm-router-go -f --no-pager -n {{lines}}"

# Build + deploy the Go router. `--cutover` flips services automatically.
deploy-proxy *args:
    ~/code/llm-router-go/deploy/scripts/deploy-router.sh {{args}}

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

# Smoke-test Qwen3-VL-8B: direct container probe + via LiteLLM
probe-qwen3-vl:
    @echo "=== direct: hypatia:5398/v1/models ==="
    @curl -sS -m 3 http://hypatia.local:5398/v1/models | python3 -m json.tool || echo "(unreachable — container not running?)"
    @echo "=== via LiteLLM: vision-fast in /v1/models ==="
    @curl -sS https://llm.peacock-bramble.ts.net/v1/models -H "Authorization: Bearer sk-litellm-master" | python3 -c 'import json,sys; d=json.load(sys.stdin); hits=[m["id"] for m in d["data"] if "qwen3-vl" in m["id"] or m["id"]=="vision-fast"]; print(*hits, sep="\n") if hits else print("(not registered)")'

# Where the deployed copy of models.yaml is on each node and its mtime
probe-deployed-models:
    #!/usr/bin/env bash
    for h in archimedes hypatia euclid; do
        echo -n "$h: "
        ssh -o ConnectTimeout=3 erewhon@$h.local "stat -c '%y' /home/erewhon/Projects/erewhon/llm-router/models.yaml 2>&1" 2>&1
    done

# ─── RAG ───────────────────────────────────────────────────────────────────────

# Ingest a code repo into the Qdrant `code` collection (default: this repo)
rag-ingest-code path=".":
    uv run llm-router-rag ingest-code {{path}}

# Search the code collection
rag-search query top-k="10":
    uv run llm-router-rag search-code "{{query}}" --top-k {{top-k}}

# Qdrant collection stats
rag-stats:
    @curl -s http://euclid.local:6333/collections/code | python3 -m json.tool

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
