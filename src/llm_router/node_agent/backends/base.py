"""Abstract backend interface for inference servers."""

from __future__ import annotations

import abc
from enum import StrEnum

from llm_router.config import ModelDefinition
from llm_router.node_agent.models import ProcessStatus


class BackendState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


class Backend(abc.ABC):
    """Abstract interface for an inference backend (vLLM, llama.cpp, etc.)."""

    @abc.abstractmethod
    async def start(self, model_id: str, model: ModelDefinition) -> None:
        """Start the inference server for a model."""

    @abc.abstractmethod
    async def stop(self, model_id: str) -> None:
        """Stop the inference server for a model."""

    @abc.abstractmethod
    async def status(self, model_id: str, model: ModelDefinition | None = None) -> ProcessStatus:
        """Get the status of a model's inference server."""

    @abc.abstractmethod
    async def health_check(self, model_id: str, model: ModelDefinition | None = None) -> bool:
        """Check if the model's inference server is ready to serve requests."""

    async def get_request_counts(self, model_id: str) -> tuple[int, int]:
        """Get (running, waiting) request counts. Override for backends that support it."""
        return 0, 0
