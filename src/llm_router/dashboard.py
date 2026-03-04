"""Lightweight model dashboard — single-file FastAPI app.

Serves a status page showing all models from the registry with
live health/status from the LiteLLM proxy API.

Run standalone:  uv run python -m llm_router.dashboard
Or as a service:  see deploy/litellm-dashboard.service
"""

from __future__ import annotations

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


@app.get("/api/models")
async def api_models():
    """Aggregate model info from registry + LiteLLM health."""
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

        api_base = _registry.get_api_base(model_id)

        models.append({
            "id": model_id,
            "hf_repo": model.hf_repo,
            "backend": model.backend.value,
            "nodes": nodes,
            "head_node": head,
            "vram_gb": model.vram_gb,
            "always_on": model.always_on,
            "tool_proxy": model.tool_proxy,
            "aliases": model.aliases,
            "capabilities": [c.value for c in model.capabilities],
            "tags": model.tags,
            "api_base": api_base,
            "health": health,
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

  .api-base {
    font-family: 'SF Mono', 'Fira Code', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
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
async function load() {
  const el = document.getElementById('content');
  try {
    const r = await fetch('/api/models');
    const data = await r.json();
    el.innerHTML = render(data);
  } catch (e) {
    el.innerHTML = `<p class="error">Failed to load: ${e.message}</p>`;
  }
}

function render(data) {
  const { nodes, models, model_count, node_count, litellm_url } = data;

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

  // Nodes
  html += `<div class="section-title">Nodes</div><div class="nodes">`;
  for (const [name, n] of Object.entries(nodes)) {
    html += `
      <div class="node-card">
        <div class="node-name">${name}</div>
        <div class="node-detail">Host: <span>${n.host}</span></div>
        <div class="node-detail">GPU: <span>${n.gpu.toUpperCase()}</span></div>
        <div class="node-detail">VRAM: <span>${n.vram_gb} GB</span></div>
        <div class="node-detail">Agent: <span>:${n.agent_port}</span></div>
      </div>`;
  }
  html += `</div>`;

  // Models table
  html += `<div class="section-title">Models</div>`;
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
    if (m.always_on) flags += '<span class="badge badge-on">always-on</span> ';
    else flags += '<span class="badge badge-off">on-demand</span> ';
    if (m.tool_proxy) flags += '<span class="badge badge-tool">tool-proxy</span> ';
    if (m.nodes.length > 1) flags += '<span class="badge badge-multi">multi-node</span> ';
    if (m.tags) m.tags.forEach(t => { flags += `<span class="badge badge-tag">${t}</span> `; });

    const hclass = `health-${m.health}`;
    const hlabel = m.health === 'routed' ? 'healthy' : m.health;

    html += `<tr>
      <td><div class="model-id">${m.id}</div><div class="model-repo">${m.hf_repo}</div>
        ${m.gguf_file ? `<div class="model-repo">${m.gguf_file}</div>` : ''}</td>
      <td><span class="badge badge-backend">${m.backend}</span></td>
      <td>${nodeStr}</td>
      <td>${m.vram_gb} GB</td>
      <td>${aliases || '<span style="color:var(--text-dim)">—</span>'}</td>
      <td>${caps}</td>
      <td>${flags}</td>
      <td><span class="health-dot ${hclass}"></span>${hlabel}</td>
      <td><span class="api-base">${m.api_base}</span></td>
    </tr>`;
  }
  html += `</tbody></table>`;

  // Connection info
  html += `
    <div class="section-title">Connection</div>
    <div class="node-card" style="max-width:500px">
      <div class="node-detail">Proxy: <span>${litellm_url}</span></div>
      <div class="node-detail" style="margin-top:0.5rem">
        <span class="api-base">curl ${litellm_url}/v1/chat/completions \\<br>
        &nbsp;&nbsp;-H "Authorization: Bearer $KEY" \\<br>
        &nbsp;&nbsp;-H "Content-Type: application/json" \\<br>
        &nbsp;&nbsp;-d '{"model":"${models[0]?.id || 'MODEL'}","messages":[...]}'</span>
      </div>
    </div>`;

  return html;
}

load();
setInterval(load, 30000);
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
