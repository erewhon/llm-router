"""Node agent — FastAPI service managing inference backends on a single node."""

from __future__ import annotations

import logging
import shutil
import socket
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import click
import uvicorn
from fastapi import FastAPI, HTTPException

from llm_router.config import BackendType, ModelRegistry, load_registry
from llm_router.node_agent.backends.base import Backend
from llm_router.node_agent.backends.llamacpp import LlamaCppBackend
from llm_router.node_agent.backends.lmstudio import LmStudioBackend
from llm_router.node_agent.backends.vllm import VllmBackend
from llm_router.node_agent.models import (
    ModelListEntry,
    ModelStartRequest,
    ModelState,
    ModelStatusResponse,
    NodeHealthResponse,
    RayJoinRequest,
    RayStatusResponse,
    ServiceStatusResponse,
)
from llm_router.node_agent.ray_cluster import RayClusterManager

logger = logging.getLogger(__name__)

# Module-level state populated at startup
_registry: ModelRegistry | None = None
_node_name: str = ""
_backends: dict[BackendType, Backend] = {}
_ray: RayClusterManager = RayClusterManager()


def _get_backend(backend_type: BackendType) -> Backend:
    if backend_type not in _backends:
        raise HTTPException(status_code=501, detail=f"Backend {backend_type.value!r} not available")
    return _backends[backend_type]


def _get_model(model_id: str) -> tuple[str, Any]:
    """Look up a model in the registry, scoped to this node."""
    assert _registry is not None
    node_models = _registry.models_for_node(_node_name)
    if model_id not in node_models:
        raise HTTPException(
            status_code=404,
            detail=f"Model {model_id!r} not found on node {_node_name!r}",
        )
    return model_id, node_models[model_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start always-on models at startup."""
    assert _registry is not None
    node_models = _registry.models_for_node(_node_name)
    for model_id, model in node_models.items():
        if model.backend == BackendType.EXTERNAL:
            # External models are managed outside the node agent (e.g. OVMS, Docker)
            logger.info(f"Skipping external model: {model_id}")
            continue
        if model.always_on and model.mode_tag is None:
            # Only auto-start models without a mode tag.
            # Mode-tagged models (mode:big, mode:default) are managed by spark-mode.sh.
            backend = _get_backend(model.backend)
            logger.info(f"Auto-starting always-on model: {model_id}")
            try:
                await backend.start(model_id, model)
            except Exception:
                logger.exception(f"Failed to auto-start {model_id}")
    yield


app = FastAPI(title="LLM Router Node Agent", lifespan=lifespan)


@app.post("/models/{model_id}/start")
async def start_model(model_id: str, req: ModelStartRequest | None = None) -> ModelStatusResponse:
    """Start an inference backend for a model."""
    model_id, model = _get_model(model_id)
    if model.backend == BackendType.EXTERNAL:
        return ModelStatusResponse(
            model_id=model_id, state=ModelState.RUNNING, backend=model.backend.value,
            hf_repo=model.hf_repo,
        )
    backend = _get_backend(model.backend)

    status = await backend.status(model_id, model=model)
    if status.state == ModelState.RUNNING:
        return ModelStatusResponse(
            model_id=model_id,
            state=status.state,
            pid=status.pid,
            port=status.port,
            backend=model.backend.value,
            hf_repo=model.hf_repo,
        )

    await backend.start(model_id, model)
    status = await backend.status(model_id, model=model)

    return ModelStatusResponse(
        model_id=model_id,
        state=status.state,
        pid=status.pid,
        port=status.port,
        backend=model.backend.value,
        hf_repo=model.hf_repo,
        error=status.error,
    )


@app.post("/models/{model_id}/stop")
async def stop_model(model_id: str) -> ModelStatusResponse:
    """Stop an inference backend for a model."""
    model_id, model = _get_model(model_id)
    if model.backend == BackendType.EXTERNAL:
        return ModelStatusResponse(
            model_id=model_id, state=ModelState.RUNNING, backend=model.backend.value,
            hf_repo=model.hf_repo,
        )
    backend = _get_backend(model.backend)

    await backend.stop(model_id)
    status = await backend.status(model_id, model=model)

    return ModelStatusResponse(
        model_id=model_id,
        state=status.state,
        backend=model.backend.value,
        hf_repo=model.hf_repo,
    )


@app.get("/models/{model_id}/status")
async def model_status(model_id: str) -> ModelStatusResponse:
    """Get the status of a model's inference backend."""
    model_id, model = _get_model(model_id)
    if model.backend == BackendType.EXTERNAL:
        return ModelStatusResponse(
            model_id=model_id,
            state=ModelState.RUNNING,
            pid=None,
            port=None,
            backend=model.backend.value,
            hf_repo=model.hf_repo,
            error=None,
        )
    backend = _get_backend(model.backend)
    status = await backend.status(model_id, model=model)

    return ModelStatusResponse(
        model_id=model_id,
        state=status.state,
        pid=status.pid,
        port=status.port,
        backend=model.backend.value,
        hf_repo=model.hf_repo,
        error=status.error,
    )


@app.get("/models")
async def list_models() -> list[ModelListEntry]:
    """List all models assigned to this node with their status."""
    assert _registry is not None
    node_models = _registry.models_for_node(_node_name)
    entries = []
    for model_id, model in node_models.items():
        if model.backend == BackendType.EXTERNAL:
            entries.append(
                ModelListEntry(
                    model_id=model_id,
                    state=ModelState.RUNNING,
                    hf_repo=model.hf_repo,
                    backend=model.backend.value,
                    always_on=model.always_on,
                    vram_gb=model.vram_gb,
                    requests_running=0,
                    requests_waiting=0,
                    avg_tok_per_s=None,
                    total_requests=0,
                )
            )
            continue
        backend = _get_backend(model.backend)
        status = await backend.status(model_id, model=model)
        running, waiting = (0, 0)
        avg_tok_s = None
        total_reqs = 0
        if status.state == ModelState.RUNNING:
            running, waiting = await backend.get_request_counts(model_id)
            throughput = await backend.get_throughput(model_id)
            avg_tok_s = throughput.get("avg_tok_per_s")
            total_reqs = throughput.get("request_count", 0)
        entries.append(
            ModelListEntry(
                model_id=model_id,
                state=status.state,
                hf_repo=model.hf_repo,
                backend=model.backend.value,
                always_on=model.always_on,
                vram_gb=model.vram_gb,
                requests_running=running,
                requests_waiting=waiting,
                avg_tok_per_s=avg_tok_s,
                total_requests=total_reqs,
            )
        )
    return entries


@app.get("/health")
async def health() -> NodeHealthResponse:
    """Health check with node GPU info."""
    assert _registry is not None
    node_models = _registry.models_for_node(_node_name)

    running = []
    for model_id, model in node_models.items():
        if model.backend == BackendType.EXTERNAL:
            running.append(model_id)
            continue
        backend = _get_backend(model.backend)
        status = await backend.status(model_id, model=model)
        if status.state == ModelState.RUNNING:
            running.append(model_id)

    gpu_type = None
    total_vram = None
    free_vram = None
    gpu_busy = None
    try:
        from llm_router.node_agent.gpu import get_gpu_info

        # Use the configured GPU type from the registry to avoid
        # misdetection on multi-GPU systems (e.g. AMD iGPU + Intel dGPU)
        node_def = _registry.nodes.get(_node_name)
        configured_gpu = node_def.gpu if node_def else None
        info = get_gpu_info(override_type=configured_gpu)
        gpu_type = info.gpu_type.value
        total_vram = info.total_vram_gb
        free_vram = info.free_vram_gb
        gpu_busy = info.gpu_busy_pct
    except Exception:
        pass

    # Root volume disk space
    disk_free: float | None = None
    disk_total: float | None = None
    try:
        usage = shutil.disk_usage("/")
        disk_free = round(usage.free / (1024**3), 1)
        disk_total = round(usage.total / (1024**3), 1)
    except OSError:
        pass

    # Probe co-located services (ComfyUI, etc.)
    service_results: list[ServiceStatusResponse] = []
    node_def = _registry.nodes.get(_node_name)
    if node_def and node_def.services:
        from llm_router.node_agent.services import probe_service

        for svc_name, svc_def in node_def.services.items():
            try:
                svc_status = await probe_service(svc_name, svc_def, node_def.host)
                service_results.append(ServiceStatusResponse(
                    name=svc_status.name,
                    service_type=svc_status.service_type,
                    label=svc_status.label,
                    reachable=svc_status.reachable,
                    vram_used_gb=svc_status.vram_used_gb,
                    vram_total_gb=svc_status.vram_total_gb,
                    queue_running=svc_status.queue_running,
                    queue_pending=svc_status.queue_pending,
                ))
            except Exception:
                logger.debug(f"Failed to probe service {svc_name}", exc_info=True)

    return NodeHealthResponse(
        node=_node_name,
        gpu_type=gpu_type,
        total_vram_gb=total_vram,
        free_vram_gb=free_vram,
        gpu_busy_pct=gpu_busy,
        disk_free_gb=disk_free,
        disk_total_gb=disk_total,
        running_models=running,
        services=service_results,
    )


@app.post("/ray/join")
async def ray_join(req: RayJoinRequest) -> RayStatusResponse:
    """Join or form a Ray cluster."""
    if req.role == "head":
        status = await _ray.start_head(port=req.port)
    elif req.role == "worker":
        if not req.head_address:
            raise HTTPException(status_code=400, detail="head_address required for worker role")
        status = await _ray.start_worker(req.head_address)
    else:
        raise HTTPException(status_code=400, detail=f"Invalid role: {req.role!r}")

    return RayStatusResponse(
        role=status.role.value,
        state=status.state.value,
        head_address=status.head_address,
        pid=status.pid,
        error=status.error,
    )


@app.post("/ray/leave")
async def ray_leave() -> RayStatusResponse:
    """Leave the Ray cluster and stop Ray on this node."""
    status = await _ray.stop()
    return RayStatusResponse(
        role=status.role.value,
        state=status.state.value,
    )


@app.get("/ray/status")
async def ray_status() -> RayStatusResponse:
    """Get this node's Ray cluster status."""
    status = _ray.status
    return RayStatusResponse(
        role=status.role.value,
        state=status.state.value,
        head_address=status.head_address,
        pid=status.pid,
        error=status.error,
    )


def create_app(
    registry: ModelRegistry,
    node_name: str,
) -> FastAPI:
    """Create the FastAPI app with configured state."""
    global _registry, _node_name, _backends
    _registry = registry
    _node_name = node_name
    lmstudio = LmStudioBackend()
    _backends = {
        BackendType.VLLM: VllmBackend(),
        BackendType.LLAMACPP: LlamaCppBackend(),
        BackendType.LMSTUDIO: lmstudio,
    }
    # Register ports for lmstudio models so health probes work
    for mid, m in registry.models_for_node(node_name).items():
        if m.backend == BackendType.LMSTUDIO:
            lmstudio.register_model(mid, m)
    return app


@click.command()
@click.option(
    "--registry",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to models.yaml",
)
@click.option(
    "--node",
    "-n",
    type=str,
    default=None,
    help="Node name (default: hostname)",
)
@click.option("--host", default="0.0.0.0", help="Bind address")
@click.option("--port", default=8100, type=int, help="Bind port")
def cli(registry: Path | None, node: str | None, host: str, port: int) -> None:
    """Start the node agent."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

    node_name = node or socket.gethostname()
    reg = load_registry(registry)

    if node_name not in reg.nodes:
        raise click.ClickException(
            f"Node {node_name!r} not found in registry. "
            f"Available: {', '.join(reg.nodes)}"
        )

    create_app(reg, node_name)
    logger.info(f"Starting node agent for {node_name} on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    cli()
