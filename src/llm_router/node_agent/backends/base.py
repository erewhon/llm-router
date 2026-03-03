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
    async def status(self, model_id: str) -> ProcessStatus:
        """Get the status of a model's inference server."""

    @abc.abstractmethod
    async def health_check(self, model_id: str) -> bool:
        """Check if the model's inference server is ready to serve requests."""
