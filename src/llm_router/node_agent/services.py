"""Service prober — check health/metrics of co-located services (ComfyUI, etc.)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import httpx

from llm_router.config import ServiceDefinition, ServiceType

logger = logging.getLogger(__name__)

PROBE_TIMEOUT = 3.0


@dataclass
class ServiceStatus:
    name: str
    service_type: str
    label: str = ""
    reachable: bool = False
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    queue_running: int = 0
    queue_pending: int = 0
    extra: dict[str, object] = field(default_factory=dict)


async def probe_comfyui(name: str, host: str, port: int, label: str = "") -> ServiceStatus:
    """Probe a ComfyUI instance via /system_stats and /queue."""
    status = ServiceStatus(name=name, service_type=ServiceType.COMFYUI, label=label)
    base = f"http://{host}:{port}"

    try:
        async with httpx.AsyncClient(timeout=PROBE_TIMEOUT) as client:
            # Fetch system stats (VRAM info)
            stats_resp = await client.get(f"{base}/system_stats")
            if stats_resp.status_code == 200:
                # Note: ComfyUI /system_stats reports system-wide GPU VRAM,
                # not ComfyUI-specific usage. Don't report it as service VRAM
                # since the node-level VRAM bar already shows the total.
                status.reachable = True

            # Fetch queue info
            try:
                queue_resp = await client.get(f"{base}/queue")
                if queue_resp.status_code == 200:
                    q = queue_resp.json()
                    status.queue_running = len(q.get("queue_running", []))
                    status.queue_pending = len(q.get("queue_pending", []))
            except httpx.HTTPError:
                pass  # queue endpoint is optional
    except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
        logger.debug(f"ComfyUI at {base} unreachable")
    except Exception:
        logger.debug(f"Error probing ComfyUI at {base}", exc_info=True)

    return status


async def probe_service(name: str, svc: ServiceDefinition, host: str) -> ServiceStatus:
    """Dispatch to the correct prober based on service type."""
    if svc.type == ServiceType.COMFYUI:
        return await probe_comfyui(name, host, svc.port, label=svc.label)

    # Unknown service type — return unreachable
    return ServiceStatus(name=name, service_type=svc.type, label=svc.label)
