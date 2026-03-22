"""LMStudio backend — external process, probe-only."""

from __future__ import annotations

import logging

import httpx

from llm_router.config import ModelDefinition
from llm_router.node_agent.backends.base import Backend
from llm_router.node_agent.models import ModelState, ProcessStatus

logger = logging.getLogger(__name__)


class LmStudioBackend(Backend):
    """Read-only backend for LMStudio instances.

    LMStudio runs as an external process (not managed by the node agent).
    This backend probes the API port to report whether it's running.
    """

    def __init__(self) -> None:
        self._ports: dict[str, int] = {}

    def _port_for(self, model_id: str, model: ModelDefinition | None = None) -> int:
        if model_id in self._ports:
            return self._ports[model_id]
        if model and model.api_port:
            self._ports[model_id] = model.api_port
            return model.api_port
        return 1234  # LMStudio default

    async def start(self, model_id: str, model: ModelDefinition) -> None:
        """No-op — LMStudio is managed externally."""
        self._ports[model_id] = self._port_for(model_id, model)
        logger.info(
            f"LMStudio model {model_id} is externally managed "
            f"(port {self._ports[model_id]})"
        )

    async def stop(self, model_id: str) -> None:
        """No-op — LMStudio is managed externally."""
        logger.info(f"LMStudio model {model_id} cannot be stopped by the agent")

    async def status(self, model_id: str, model: ModelDefinition | None = None) -> ProcessStatus:
        """Probe the LMStudio API to determine if the model is running."""
        port = self._port_for(model_id)
        running = await self.health_check(model_id)
        return ProcessStatus(
            model_id=model_id,
            state=ModelState.RUNNING if running else ModelState.STOPPED,
            port=port if running else None,
        )

    async def health_check(self, model_id: str, model: ModelDefinition | None = None) -> bool:
        """Check if LMStudio is responding and serving the expected model."""
        port = self._port_for(model_id)
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(f"http://localhost:{port}/v1/models")
                if resp.status_code != 200:
                    return False
                if model is None:
                    return True
                # Verify the specific model is loaded
                data = resp.json()
                served_ids = {m.get("id", "") for m in data.get("data", [])}
                hf_base = model.hf_repo.split("#")[0]
                return hf_base in served_ids or model.hf_repo in served_ids
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            return False

    def register_model(self, model_id: str, model: ModelDefinition) -> None:
        """Register a model's port so status() can probe it."""
        self._ports[model_id] = self._port_for(model_id, model)
