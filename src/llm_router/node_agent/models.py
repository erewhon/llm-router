"""Pydantic request/response models for the node agent API."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel


class ModelState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class ProcessStatus(BaseModel):
    model_id: str
    state: ModelState
    pid: int | None = None
    port: int | None = None
    error: str | None = None
    vram_gb: int = 0


class ModelStartRequest(BaseModel):
    """Optional overrides when starting a model."""

    vram_gb: int | None = None
    max_model_len: int | None = None
    extra_args: list[str] | None = None


class ModelStatusResponse(BaseModel):
    model_id: str
    state: ModelState
    pid: int | None = None
    port: int | None = None
    backend: str | None = None
    hf_repo: str | None = None
    error: str | None = None


class ServiceStatusResponse(BaseModel):
    name: str
    service_type: str
    label: str = ""
    reachable: bool = False
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    queue_running: int = 0
    queue_pending: int = 0


class NodeHealthResponse(BaseModel):
    status: str = "ok"
    node: str
    gpu_type: str | None = None
    total_vram_gb: float | None = None
    free_vram_gb: float | None = None
    gpu_busy_pct: int | None = None
    disk_free_gb: float | None = None
    disk_total_gb: float | None = None
    running_models: list[str] = []
    services: list[ServiceStatusResponse] = []


class ModelListEntry(BaseModel):
    model_id: str
    state: ModelState
    hf_repo: str
    backend: str
    always_on: bool
    vram_gb: int
    requests_running: int = 0
    requests_waiting: int = 0


class RayJoinRequest(BaseModel):
    """Request to join or form a Ray cluster."""

    role: str  # "head" or "worker"
    head_address: str | None = None  # required when role == "worker"
    port: int = 6379  # Ray port (for head)


class RayStatusResponse(BaseModel):
    role: str
    state: str
    head_address: str | None = None
    pid: int | None = None
    error: str | None = None
