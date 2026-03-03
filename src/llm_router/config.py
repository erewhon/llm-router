"""Pydantic models for the LLM Router model registry (models.yaml)."""

from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

REGISTRY_PATH = Path(__file__).parents[2] / "models.yaml"


class BackendType(StrEnum):
    VLLM = "vllm"
    LLAMACPP = "llamacpp"


class ModelCapability(StrEnum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"


class GpuType(StrEnum):
    AMD = "amd"
    NVIDIA = "nvidia"


class NodeDefinition(BaseModel):
    host: str
    gpu: GpuType
    vram_gb: int
    agent_port: int = 8100


class MultiNodeConfig(BaseModel):
    """Configuration for models that span multiple nodes via Ray."""

    nodes: list[str]
    tensor_parallel_size: int
    head_node: str | None = None  # defaults to first node


class VllmArgs(BaseModel):
    """Extra vLLM-specific arguments."""

    tool_call_parser: str | None = None
    max_model_len: int | None = None
    gpu_memory_utilization: float | None = None
    extra_args: list[str] = Field(default_factory=list)


class ModelDefinition(BaseModel):
    """A single model entry in the registry."""

    hf_repo: str
    backend: BackendType = BackendType.VLLM
    node: str | None = None  # single-node deployment
    multi_node: MultiNodeConfig | None = None
    vram_gb: int = 0
    always_on: bool = False
    tool_proxy: bool = False
    aliases: list[str] = Field(default_factory=list)
    capabilities: list[ModelCapability] = Field(default_factory=lambda: [ModelCapability.TEXT])
    vllm_args: VllmArgs = Field(default_factory=VllmArgs)

    # For llamacpp backend
    gguf_file: str | None = None

    @model_validator(mode="after")
    def validate_node_or_multi(self) -> "ModelDefinition":
        if not self.node and not self.multi_node:
            raise ValueError("Model must specify either 'node' or 'multi_node'")
        if self.node and self.multi_node:
            raise ValueError("Model cannot specify both 'node' and 'multi_node'")
        return self


class ModelRegistry(BaseModel):
    """Top-level registry parsed from models.yaml."""

    nodes: dict[str, NodeDefinition]
    models: dict[str, ModelDefinition]

    def get_node(self, model_id: str) -> NodeDefinition:
        """Get the node definition for a model (single-node only)."""
        model = self.models[model_id]
        if not model.node:
            raise ValueError(f"Model {model_id!r} is multi-node, use get_nodes()")
        return self.nodes[model.node]

    def get_api_base(self, model_id: str) -> str:
        """Get the API base URL for a model's inference backend.

        For multi-node models, uses the head node.
        """
        model = self.models[model_id]
        if model.multi_node:
            head = model.multi_node.head_node or model.multi_node.nodes[0]
            host = self.nodes[head].host
        else:
            host = self.get_node(model_id).host
        if model.tool_proxy:
            return f"http://{host}:5392/v1"
        return f"http://{host}:5391/v1"

    def models_for_node(self, node_name: str) -> dict[str, ModelDefinition]:
        """Get all models assigned to a specific node."""
        return {
            mid: m
            for mid, m in self.models.items()
            if m.node == node_name
            or (m.multi_node and node_name in m.multi_node.nodes)
        }


def load_registry(path: Path | None = None) -> ModelRegistry:
    """Load and validate the model registry from YAML."""
    path = path or REGISTRY_PATH
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelRegistry.model_validate(data)


def load_registry_from_dict(data: dict[str, Any]) -> ModelRegistry:
    """Load and validate the model registry from a dict (for testing)."""
    return ModelRegistry.model_validate(data)
