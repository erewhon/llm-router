"""Lightweight model dashboard — single-file FastAPI app.

Serves a status page showing all models from the registry with
live health/status from the LiteLLM proxy API.

Run standalone:  uv run python -m llm_router.dashboard
Or as a service:  see deploy/litellm-dashboard.service
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click
import httpx
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from llm_router.config import ModelRegistry, load_registry

logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Router Dashboard")

_registry: ModelRegistry | None = None
_litellm_url: str = "http://localhost:4010"
_litellm_key: str = "sk-litellm-master"


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the dashboard HTML page."""
    return DASHBOARD_HTML


async def _fetch_node_metrics(
    name: str, host: str, agent_port: int
) -> dict:
    """Fetch GPU metrics and model states from a single node agent."""
    result: dict = {
        "reachable": False,
        "vram_used_gb": None,
        "vram_total_gb": None,
        "vram_pct": None,
        "gpu_busy_pct": None,
        "models": [],
    }
    try:
        base = f"http://{host}:{agent_port}"
        async with httpx.AsyncClient(timeout=5) as client:
            health_resp, models_resp = await asyncio.gather(
                client.get(f"{base}/health"),
                client.get(f"{base}/models"),
                return_exceptions=True,
            )

            if isinstance(health_resp, Exception):
                return result
            if health_resp.status_code == 200:
                h = health_resp.json()
                total = h.get("total_vram_gb")
                free = h.get("free_vram_gb")
                if total is not None and free is not None:
                    used = round(total - free, 1)
                    result["vram_used_gb"] = used
                    result["vram_total_gb"] = round(total, 1)
                    result["vram_pct"] = (
                        round(used / total * 100, 1) if total > 0 else 0
                    )
                result["gpu_busy_pct"] = h.get("gpu_busy_pct")
                result["services"] = h.get("services", [])
                result["disk_free_gb"] = h.get("disk_free_gb")
                result["disk_total_gb"] = h.get("disk_total_gb")
                result["reachable"] = True

            if not isinstance(models_resp, Exception) and models_resp.status_code == 200:
                for m in models_resp.json():
                    result["models"].append({
                        "model_id": m.get("model_id", ""),
                        "state": m.get("state", "unknown"),
                    })
    except Exception:
        pass
    return result


@app.get("/api/node-metrics")
async def node_metrics():
    """Fast endpoint: only GPU metrics from node agents (no LiteLLM)."""
    assert _registry is not None
    metrics: dict[str, dict] = {}
    tasks = {
        name: _fetch_node_metrics(name, node.host, node.agent_port)
        for name, node in _registry.nodes.items()
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for name, result in zip(tasks.keys(), results, strict=True):
        if isinstance(result, Exception):
            metrics[name] = {
                "reachable": False,
                "vram_used_gb": None,
                "vram_total_gb": None,
                "vram_pct": None,
                "gpu_busy_pct": None,
                "models": [],
            }
        else:
            metrics[name] = result
    return metrics


@app.get("/api/models")
async def api_models():
    """Aggregate model info from registry + LiteLLM health + node metrics."""
    assert _registry is not None

    # Fetch live data from LiteLLM
    live_models: dict = {}
    health_info: dict = {}
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{_litellm_url}/model/info",
                headers={"Authorization": f"Bearer {_litellm_key}"},
            )
            if resp.status_code == 200:
                for entry in resp.json().get("data", []):
                    name = entry.get("model_name", "")
                    live_models[name] = entry

            resp = await client.get(
                f"{_litellm_url}/health",
                headers={"Authorization": f"Bearer {_litellm_key}"},
            )
            if resp.status_code == 200:
                hdata = resp.json()
                for ep in hdata.get("healthy_endpoints", []):
                    model = ep.get("model", "").removeprefix("openai/")
                    health_info[model] = "healthy"
                for ep in hdata.get("unhealthy_endpoints", []):
                    model = ep.get("model", "").removeprefix("openai/")
                    health_info[model] = "unhealthy"
    except Exception as e:
        logger.warning(f"Could not reach LiteLLM at {_litellm_url}: {e}")

    # Fetch node metrics in parallel
    node_metrics: dict[str, dict] = {}
    tasks = {
        name: _fetch_node_metrics(name, node.host, node.agent_port)
        for name, node in _registry.nodes.items()
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    for name, result in zip(tasks.keys(), results, strict=True):
        if isinstance(result, Exception):
            node_metrics[name] = {
                "reachable": False,
                "vram_used_gb": None,
                "vram_total_gb": None,
                "vram_pct": None,
                "gpu_busy_pct": None,
                "models": [],
            }
        else:
            node_metrics[name] = result

    # Build model_id -> agent state lookup
    agent_model_states: dict[str, str] = {}
    for nm in node_metrics.values():
        for m in nm.get("models", []):
            agent_model_states[m["model_id"]] = m["state"]

    # Build response from registry
    models = []
    for model_id, model in _registry.models.items():
        # Determine node(s)
        if model.multi_node:
            nodes = model.multi_node.nodes
            head = model.multi_node.head_node or nodes[0]
        elif model.node:
            nodes = [model.node]
            head = model.node
        else:
            nodes = []
            head = None

        # Check if this model (or its hf_repo) is healthy
        health = health_info.get(model.hf_repo, "unknown")
        if model_id in live_models:
            health = "routed"
            if model.hf_repo in health_info:
                health = health_info[model.hf_repo]

        # Node agent state (running/stopped/starting/error)
        agent_state = agent_model_states.get(model_id)

        api_base = _registry.get_api_base(model_id)

        models.append({
            "id": model_id,
            "hf_repo": model.hf_repo,
            "backend": model.backend.value,
            "nodes": nodes,
            "head_node": head,
            "vram_gb": model.vram_gb,
            "always_on": model.always_on,
            "enabled": model.enabled,
            "tool_proxy": model.tool_proxy,
            "aliases": model.aliases,
            "capabilities": [c.value for c in model.capabilities],
            "tags": model.tags,
            "api_base": api_base,
            "health": health if model.enabled else "disabled",
            "agent_state": agent_state if model.enabled else None,
            "gguf_file": model.gguf_file,
        })

    return {
        "litellm_url": _litellm_url,
        "node_count": len(_registry.nodes),
        "model_count": len(_registry.models),
        "nodes": {
            name: {
                "host": node.host,
                "gpu": node.gpu.value,
                "vram_gb": node.vram_gb,
                "agent_port": node.agent_port,
            }
            for name, node in _registry.nodes.items()
        },
        "node_metrics": node_metrics,
        "models": models,
    }


DASHBOARD_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Router</title>
<style>
  :root {
    --bg: #0f1117;
    --surface: #1a1d27;
    --border: #2a2d3a;
    --text: #e1e4ed;
    --text-dim: #8b8fa3;
    --accent: #6c8cff;
    --green: #4ade80;
    --yellow: #fbbf24;
    --red: #f87171;
    --orange: #fb923c;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    padding: 2rem;
    max-width: 1200px;
    margin: 0 auto;
  }
  h1 { font-size: 1.5rem; font-weight: 600; margin-bottom: 0.25rem; }
  .subtitle { color: var(--text-dim); font-size: 0.875rem; margin-bottom: 2rem; }
  .stats {
    display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap;
  }
  .stat {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    min-width: 140px;
  }
  .stat-value { font-size: 1.5rem; font-weight: 700; color: var(--accent); }
  .stat-label {
    font-size: 0.75rem; color: var(--text-dim);
    text-transform: uppercase; letter-spacing: 0.05em;
  }

  .section-title {
    font-size: 1rem; font-weight: 600; margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border);
  }

  /* Nodes */
  .nodes { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
  .node-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.25rem;
    min-width: 200px;
    flex: 1;
  }
  .node-name { font-weight: 600; margin-bottom: 0.5rem; }
  .node-detail { font-size: 0.8rem; color: var(--text-dim); }
  .node-detail span { color: var(--text); }

  /* GPU metric rows */
  .metric-row {
    display: flex; align-items: center; gap: 0.4rem;
    margin-top: 0.4rem;
  }
  .metric-label {
    font-size: 0.65rem; font-weight: 600;
    color: var(--text-dim); width: 2rem;
    text-transform: uppercase; flex-shrink: 0;
  }
  .vram-bar-track {
    flex: 1; height: 6px; min-width: 40px;
    background: var(--border); border-radius: 3px;
    overflow: hidden;
  }
  .vram-bar-fill {
    height: 100%; border-radius: 3px;
    transition: width 0.6s ease, background 0.6s ease;
  }
  .vram-bar-fill.green { background: var(--green); }
  .vram-bar-fill.yellow { background: var(--yellow); }
  .vram-bar-fill.red { background: var(--red); }
  .vram-bar-fill.grey { background: var(--text-dim); }
  .vram-text {
    font-size: 0.7rem; color: var(--text-dim);
    white-space: nowrap; flex-shrink: 0;
  }
  .vram-text strong { color: var(--text); font-weight: 600; }
  .node-offline {
    font-size: 0.8rem; color: var(--text-dim);
    margin-top: 0.5rem; font-style: italic;
  }

  /* Models table */
  table {
    width: 100%;
    border-collapse: collapse;
    background: var(--surface);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  th {
    text-align: left;
    padding: 0.75rem 1rem;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-dim);
    background: var(--bg);
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: rgba(108, 140, 255, 0.04); }

  .model-id { font-weight: 600; }
  .model-repo { font-size: 0.75rem; color: var(--text-dim); }

  .badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 500;
    margin: 0.1rem 0.15rem;
  }
  .badge-backend { background: rgba(108, 140, 255, 0.15); color: var(--accent); }
  .badge-alias { background: rgba(139, 143, 163, 0.15); color: var(--text-dim); }
  .badge-cap { background: rgba(74, 222, 128, 0.12); color: var(--green); }
  .badge-on { background: rgba(74, 222, 128, 0.15); color: var(--green); }
  .badge-off { background: rgba(139, 143, 163, 0.1); color: var(--text-dim); }
  .badge-tool { background: rgba(251, 191, 36, 0.15); color: var(--yellow); }
  .badge-multi { background: rgba(251, 146, 60, 0.15); color: var(--orange); }
  .badge-tag { background: rgba(168, 85, 247, 0.15); color: #c084fc; }

  .health-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 0.4rem;
    vertical-align: middle;
  }
  .health-healthy { background: var(--green); }
  .health-routed { background: var(--green); }
  .health-unhealthy { background: var(--red); }
  .health-unknown { background: var(--text-dim); }
  .health-running { background: var(--green); }
  .health-starting { background: var(--yellow); }
  .health-stopped { background: var(--text-dim); }
  .health-error { background: var(--red); }
  .health-disabled { background: var(--text-dim); }

  tr.disabled td { opacity: 0.45; }

  .toggle-row {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.75rem; font-size: 0.8rem; color: var(--text-dim);
  }
  .toggle-row label { cursor: pointer; user-select: none; }
  .toggle-row input[type="checkbox"] { cursor: pointer; }

  .api-base {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
  }

  .copy-btn {
    background: none; border: 1px solid var(--border); border-radius: 4px;
    color: var(--text-dim); cursor: pointer; padding: 2px 6px;
    font-size: 0.65rem; margin-left: 0.4rem; vertical-align: middle;
    transition: all 0.2s;
  }
  .copy-btn:hover { border-color: var(--accent); color: var(--accent); }
  .copy-btn.copied { border-color: var(--green); color: var(--green); }

  .alias-list {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 0.75rem 2rem;
  }
  .alias-item {
    display: flex; gap: 0.5rem; align-items: baseline;
  }
  .alias-name {
    font-weight: 600; font-size: 0.85rem;
    color: var(--accent); white-space: nowrap;
  }
  .alias-desc {
    font-size: 0.8rem; color: var(--text-dim);
  }

  .loading { color: var(--text-dim); padding: 2rem; text-align: center; }
  .error { color: var(--red); padding: 1rem; }

  @media (max-width: 768px) {
    body { padding: 1rem; }
    .stats { flex-direction: column; }
    table { font-size: 0.8rem; }
    th, td { padding: 0.5rem; }
  }
</style>
</head>
<body>
  <h1>LLM Router</h1>
  <p class="subtitle">Model registry and routing status</p>

  <div id="content"><p class="loading">Loading...</p></div>

<script>
// Sparkline history per node: { vram_pct: [], gpu_busy_pct: [] }
const nodeHistory = {};
const SPARK_MAX = 60; // ~2 min at 2s intervals

// Last full data from /api/models (refreshed every 30s)
let lastData = null;

function updateHistory(nm) {
  for (const [name, m] of Object.entries(nm)) {
    if (!nodeHistory[name]) nodeHistory[name] = {vram: [], gpu: []};
    if (m.reachable) {
      if (m.vram_pct !== null) {
        nodeHistory[name].vram.push(m.vram_pct);
        if (nodeHistory[name].vram.length > SPARK_MAX)
          nodeHistory[name].vram.shift();
      }
      if (m.gpu_busy_pct !== null) {
        nodeHistory[name].gpu.push(m.gpu_busy_pct);
        if (nodeHistory[name].gpu.length > SPARK_MAX)
          nodeHistory[name].gpu.shift();
      }
    }
  }
}

async function load() {
  const el = document.getElementById('content');
  try {
    const r = await fetch('/api/models');
    lastData = await r.json();
    updateHistory(lastData.node_metrics || {});
    el.innerHTML = render(lastData);
  } catch (e) {
    el.innerHTML = `<p class="error">Failed to load: ${e.message}</p>`;
  }
}

async function pollMetrics() {
  if (!lastData) return;
  try {
    const r = await fetch('/api/node-metrics');
    const nm = await r.json();
    lastData.node_metrics = nm;
    // Also update agent_state on models from fresh metrics
    const stateMap = {};
    for (const m of Object.values(nm)) {
      for (const mdl of (m.models || [])) {
        stateMap[mdl.model_id] = mdl.state;
      }
    }
    for (const m of lastData.models) {
      m.agent_state = stateMap[m.id] || null;
    }
    updateHistory(nm);
    document.getElementById('content').innerHTML = render(lastData);
  } catch (_) { /* next poll will retry */ }
}

function sparklineSvg(vals, color) {
  if (!vals || vals.length < 2) return '';
  const w = 48, h = 16;
  const coords = vals.map((v, i) => {
    const x = (i / (vals.length - 1)) * w;
    const y = h - (v / 100) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}"` +
    ` style="vertical-align:middle">` +
    `<polyline points="${coords}" fill="none" stroke="${color}"` +
    ` stroke-width="1.5" stroke-linecap="round"` +
    ` stroke-linejoin="round"/></svg>`;
}

let showDisabled = false;
function toggleDisabled(checked) {
  showDisabled = checked;
  document.querySelectorAll('tr.disabled').forEach(tr => {
    tr.style.display = checked ? '' : 'none';
  });
}

function copyText(text, btn) {
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = 'copied';
    btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'copy'; btn.classList.remove('copied'); }, 1500);
  });
}

function vramBarColor(pct) {
  if (pct >= 90) return 'red';
  if (pct >= 70) return 'yellow';
  return 'green';
}

function render(data) {
  const { nodes, models, model_count, node_count, litellm_url, node_metrics } = data;
  const nm = node_metrics || {};

  // Stats
  const alwaysOn = models.filter(m => m.always_on).length;
  const healthy = models.filter(m => m.health === 'healthy' || m.health === 'routed').length;
  let html = `
    <div class="stats">
      <div class="stat">
        <div class="stat-value">${model_count}</div>
        <div class="stat-label">Models</div></div>
      <div class="stat">
        <div class="stat-value">${node_count}</div>
        <div class="stat-label">Nodes</div></div>
      <div class="stat">
        <div class="stat-value">${alwaysOn}</div>
        <div class="stat-label">Always On</div></div>
      <div class="stat">
        <div class="stat-value">${healthy}</div>
        <div class="stat-label">Healthy</div></div>
    </div>`;

  // Connection info
  const tsUrl = 'https://llm.peacock-bramble.ts.net';
  html += `
    <div class="section-title">Connection</div>
    <div class="nodes">
      <div class="node-card" style="flex:2">
        <div class="node-name">Quick Start</div>
        <div class="node-detail" style="margin-top:0.5rem">
          <strong>Host:</strong> <span class="api-base">${tsUrl}</span>
          <button class="copy-btn" onclick="copyText('${tsUrl}', this)">copy</button>
          <span style="color:var(--text-dim); font-size:0.75rem">(preferred)</span>
        </div>
        <div class="node-detail">
          <strong>Direct:</strong> <span class="api-base">${litellm_url}</span>
          <button class="copy-btn" onclick="copyText('${litellm_url}', this)">copy</button>
          <span style="color:var(--text-dim); font-size:0.75rem">(from delphi only)</span>
        </div>
        <div class="node-detail" style="margin-top:0.5rem">
          <strong>API Key:</strong> <span class="api-base">sk-litellm-master</span>
          <button class="copy-btn" onclick="copyText('sk-litellm-master', this)">copy</button>
        </div>
        <div class="node-detail" style="margin-top:0.5rem">
          <strong>Model:</strong> <span>use the model ID or any alias from the table below</span>
        </div>
      </div>
      <div class="node-card" style="flex:3">
        <div class="node-name">Example</div>
        <div style="margin-top:0.5rem">
          <span class="api-base">curl ${tsUrl}/v1/chat/completions \\<br>
          &nbsp;&nbsp;-H "Authorization: Bearer sk-litellm-master" \\<br>
          &nbsp;&nbsp;-H "Content-Type: application/json" \\<br>
          &nbsp;&nbsp;-d '{"model":"${models[0]?.aliases?.[0] || models[0]?.id || 'MODEL'}","messages":[{"role":"user","content":"Hello"}]}'</span>
        </div>
      </div>
    </div>`;

  // Alias descriptions
  html += `
    <div class="section-title">Model Aliases</div>
    <div class="alias-list" style="grid-template-columns: repeat(4, 1fr)">
      <div class="alias-item">
        <span class="alias-name">coder</span>
        <span class="alias-desc">Code generation and debugging</span>
      </div>
      <div class="alias-item">
        <span class="alias-name">thinker</span>
        <span class="alias-desc">Reasoning and problem solving</span>
      </div>
      <div class="alias-item">
        <span class="alias-name">research</span>
        <span class="alias-desc">Web search via VPN (27B dense)</span>
      </div>
      <div class="alias-item">
        <span class="alias-name">vision</span>
        <span class="alias-desc">Image understanding</span>
      </div>
      <div class="alias-item">
        <span class="alias-name">coder-fast</span>
        <span class="alias-desc">Quick code tasks (9B, no thinking)</span>
      </div>
      <div class="alias-item">
        <span class="alias-name">coder-veryfast</span>
        <span class="alias-desc">AI grep (4B, no thinking)</span>
      </div>
      <div class="alias-item" style="opacity: 0"></div>
      <div class="alias-item" style="opacity: 0"></div>
    </div>`;

  // Nodes
  html += `<div class="section-title">Nodes</div><div class="nodes">`;
  for (const [name, n] of Object.entries(nodes)) {
    const m = nm[name] || {};
    const reachable = m.reachable === true;
    html += `<div class="node-card">
        <div class="node-name">${name}</div>
        <div class="node-detail">Host: <span>${n.host}</span></div>
        <div class="node-detail">GPU: <span>${n.gpu.toUpperCase()}</span>
          &middot; <span>${n.vram_gb} GB</span></div>`;

    if (reachable && m.vram_pct !== null) {
      const color = vramBarColor(m.vram_pct);
      const hist = nodeHistory[name] || {vram:[], gpu:[]};
      const busy = m.gpu_busy_pct;
      html += `
        <div class="metric-row">
          <span class="metric-label">MEM</span>
          <div class="vram-bar-track">
            <div class="vram-bar-fill ${color}"
              style="width:${m.vram_pct}%"></div>
          </div>
          <span class="vram-text"><strong>${m.vram_used_gb}</strong>
            / ${m.vram_total_gb} GB</span>
          ${sparklineSvg(hist.vram, 'var(--accent)')}
        </div>`;
      if (busy !== null && busy !== undefined) {
        const bColor = busy >= 90 ? 'red' : busy >= 50 ? 'yellow' : 'green';
        html += `
        <div class="metric-row">
          <span class="metric-label">GPU</span>
          <div class="vram-bar-track">
            <div class="vram-bar-fill ${bColor}"
              style="width:${busy}%"></div>
          </div>
          <span class="vram-text"><strong>${busy}%</strong></span>
          ${sparklineSvg(hist.gpu, 'var(--green)')}
        </div>`;
      }
    } else if (reachable) {
      html += `<div class="node-offline">No GPU metrics</div>`;
    } else {
      html += `<div class="node-offline">Agent offline</div>`;
    }

    // Disk usage
    if (reachable && m.disk_free_gb !== null && m.disk_free_gb !== undefined
        && m.disk_total_gb !== null && m.disk_total_gb !== undefined) {
      const diskUsed = (m.disk_total_gb - m.disk_free_gb).toFixed(0);
      const diskPct = ((m.disk_total_gb - m.disk_free_gb) / m.disk_total_gb * 100).toFixed(1);
      const diskColor = diskPct >= 90 ? 'red' : diskPct >= 75 ? 'yellow' : 'green';
      html += `
        <div class="metric-row">
          <span class="metric-label">DISK</span>
          <div class="vram-bar-track">
            <div class="vram-bar-fill ${diskColor}"
              style="width:${diskPct}%"></div>
          </div>
          <span class="vram-text"><strong>${diskUsed}</strong>
            / ${m.disk_total_gb.toFixed(0)} GB
            (${m.disk_free_gb.toFixed(0)} free)</span>
        </div>`;
    }

    // Services (ComfyUI, etc.)
    const svcs = m.services || [];
    for (const svc of svcs) {
      html += `<div style="margin-top:.5rem;padding-top:.5rem;border-top:1px solid var(--border)">`;
      const dot = svc.reachable ? 'health-running' : 'health-stopped';
      const lbl = svc.label || svc.name;
      html += `<div style="font-size:0.8rem;font-weight:500">` +
        `<span class="health-dot ${dot}"></span>${lbl}</div>`;
      if (svc.reachable && svc.vram_used_gb !== null && svc.vram_total_gb !== null) {
        const svcPct = (svc.vram_used_gb / svc.vram_total_gb * 100).toFixed(1);
        const svcColor = vramBarColor(parseFloat(svcPct));
        html += `
          <div class="metric-row">
            <span class="metric-label">MEM</span>
            <div class="vram-bar-track">
              <div class="vram-bar-fill ${svcColor}"
                style="width:${svcPct}%"></div>
            </div>
            <span class="vram-text"><strong>${svc.vram_used_gb}</strong>
              / ${svc.vram_total_gb} GB</span>
          </div>`;
      }
      if (svc.queue_running > 0 || svc.queue_pending > 0) {
        html += `<div style="font-size:0.7rem;color:var(--text-dim);margin-top:2px">` +
          `Queue: ${svc.queue_running} running, ${svc.queue_pending} pending</div>`;
      }
      html += `</div>`;
    }

    html += `</div>`;
  }
  html += `</div>`;

  // Models table
  const disabledCount = models.filter(m => m.enabled === false).length;
  html += `<div class="section-title">Models</div>`;
  if (disabledCount > 0) {
    html += `<div class="toggle-row">
      <input type="checkbox" id="show-disabled" ${showDisabled ? 'checked' : ''} onchange="toggleDisabled(this.checked)">
      <label for="show-disabled">Show disabled models (${disabledCount})</label>
    </div>`;
  }
  html += `<table><thead><tr>
    <th>Model</th><th>Backend</th><th>Node(s)</th>
    <th>VRAM</th><th>Aliases</th><th>Capabilities</th>
    <th>Flags</th><th>Health</th><th>API Base</th>
  </tr></thead><tbody>`;

  for (const m of models) {
    const nodeStr = m.nodes.map(n =>
      n === m.head_node && m.nodes.length > 1 ? `<strong>${n}</strong>` : n
    ).join(', ');

    const aliases = m.aliases.map(a => `<span class="badge badge-alias">${a}</span>`).join(' ');
    const caps = m.capabilities.map(c => `<span class="badge badge-cap">${c}</span>`).join(' ');

    let flags = '';
    if (m.enabled === false) flags += '<span class="badge badge-off">disabled</span> ';
    if (m.always_on) flags += '<span class="badge badge-on">always-on</span> ';
    else flags += '<span class="badge badge-off">on-demand</span> ';
    if (m.tool_proxy) flags += '<span class="badge badge-tool">tool-proxy</span> ';
    if (m.nodes.length > 1) flags += '<span class="badge badge-multi">multi-node</span> ';
    if (m.tags) m.tags.forEach(t => { flags += `<span class="badge badge-tag">${t}</span> `; });

    // Determine status: prefer agent_state when available, fall back to LiteLLM health
    let statusClass, statusLabel;
    if (m.agent_state) {
      statusClass = `health-${m.agent_state}`;
      statusLabel = m.agent_state;
    } else {
      statusClass = `health-${m.health}`;
      statusLabel = m.health === 'routed' ? 'healthy' : m.health;
    }

    // Secondary indicator: show LiteLLM health alongside agent state
    let healthExtra = '';
    if (m.agent_state && m.health !== 'unknown') {
      const hLabel = m.health === 'routed' ? 'healthy' : m.health;
      const hDot = `health-${m.health}`;
      healthExtra = `<div style="font-size:0.7rem;` +
        `color:var(--text-dim);margin-top:2px">` +
        `<span class="health-dot ${hDot}"` +
        ` style="width:6px;height:6px"></span>` +
        `litellm: ${hLabel}</div>`;
    }

    const hideDisabled = m.enabled === false && !showDisabled;
    const rowClass = m.enabled === false ? ` class="disabled"${hideDisabled ? ' style="display:none"' : ''}` : '';
    html += `<tr${rowClass}>
      <td><div class="model-id">${m.id}</div><div class="model-repo">${m.hf_repo}</div>
        ${m.gguf_file ? `<div class="model-repo">${m.gguf_file}</div>` : ''}</td>
      <td><span class="badge badge-backend">${m.backend}</span></td>
      <td>${nodeStr}</td>
      <td>${m.vram_gb ? m.vram_gb + ' GB' : '<span style="color:var(--text-dim)">—</span>'}</td>
      <td>${aliases || '<span style="color:var(--text-dim)">—</span>'}</td>
      <td>${caps}</td>
      <td>${flags}</td>
      <td><span style="white-space:nowrap"><span class="health-dot ${statusClass}"></span>${statusLabel}</span>${healthExtra}</td>
      <td><span class="api-base">${m.api_base}</span></td>
    </tr>`;
  }
  html += `</tbody></table>`;

  return html;
}

load();
setInterval(load, 30000);       // full refresh (LiteLLM + nodes)
setInterval(pollMetrics, 2000);  // fast GPU metrics only
</script>
</body>
</html>
"""


def create_dashboard_app(
    registry: ModelRegistry,
    litellm_url: str = "http://localhost:4010",
    litellm_key: str = "sk-litellm-master",
) -> FastAPI:
    global _registry, _litellm_url, _litellm_key
    _registry = registry
    _litellm_url = litellm_url
    _litellm_key = litellm_key
    return app


@click.command()
@click.option(
    "--registry", "-r",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to models.yaml",
)
@click.option("--litellm-url", default="http://localhost:4010", help="LiteLLM proxy URL")
@click.option("--litellm-key", default="sk-litellm-master", help="LiteLLM API key")
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=4011, type=int, help="Bind port")
def cli(
    registry: Path | None,
    litellm_url: str,
    litellm_key: str,
    host: str,
    port: int,
) -> None:
    """Start the LLM Router dashboard."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    reg = load_registry(registry)
    create_dashboard_app(reg, litellm_url, litellm_key)
    logger.info(f"Dashboard at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    cli()
