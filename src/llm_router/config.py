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
    LMSTUDIO = "lmstudio"
    EXTERNAL = "external"


class ModelCapability(StrEnum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    IMAGE_GEN = "image_gen"
    TOOL_CALLING = "tool_calling"


class GpuType(StrEnum):
    AMD = "amd"
    NVIDIA = "nvidia"
    INTEL = "intel"


class ServiceType(StrEnum):
    COMFYUI = "comfyui"


class ServiceDefinition(BaseModel):
    type: ServiceType
    port: int
    label: str = ""


class NodeDefinition(BaseModel):
    host: str
    gpu: GpuType
    vram_gb: int
    agent_port: int = 8100
    services: dict[str, ServiceDefinition] = Field(default_factory=dict)


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
    enabled: bool = True  # False = model shown in dashboard but not started/health-checked
    tool_proxy: bool = False
    aliases: list[str] = Field(default_factory=list)
    capabilities: list[ModelCapability] = Field(default_factory=lambda: [ModelCapability.TEXT])
    tags: list[str] = Field(default_factory=list)

    @property
    def mode_tag(self) -> str | None:
        """Return the mode value if a 'mode:xxx' tag is present."""
        for tag in self.tags:
            if tag.startswith("mode:"):
                return tag.split(":", 1)[1]
        return None
    vllm_args: VllmArgs = Field(default_factory=VllmArgs)

    # For llamacpp backend
    gguf_file: str | None = None

    # For external backends (lmstudio, etc.) — custom port
    api_port: int | None = None

    # For external/cloud models — full API base URL (e.g. https://opencode.ai/zen/v1)
    api_base: str | None = None
    api_key: str | None = None  # env var name or literal key

    @model_validator(mode="after")
    def validate_node_or_multi(self) -> "ModelDefinition":
        if self.backend == BackendType.EXTERNAL:
            if not self.api_base:
                raise ValueError("External models must specify 'api_base'")
            return self
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
        For external models, returns the configured api_base.
        """
        model = self.models[model_id]
        if model.api_base:
            return model.api_base
        if model.multi_node:
            head = model.multi_node.head_node or model.multi_node.nodes[0]
            host = self.nodes[head].host
        else:
            host = self.get_node(model_id).host
        if model.tool_proxy:
            # Tool proxy runs on delphi — route all tool_proxy models there
            tp_host = self.nodes["delphi"].host if "delphi" in self.nodes else host
            return f"http://{tp_host}:5392/v1"
        if model.api_port:
            return f"http://{host}:{model.api_port}/v1"
        return f"http://{host}:5391/v1"

    def models_for_node(
        self, node_name: str, *, enabled_only: bool = True
    ) -> dict[str, ModelDefinition]:
        """Get all models assigned to a specific node.

        By default only returns enabled models.  Pass enabled_only=False
        to include disabled models (e.g. for dashboard display).
        """
        return {
            mid: m
            for mid, m in self.models.items()
            if (m.node == node_name or (m.multi_node and node_name in m.multi_node.nodes))
            and (not enabled_only or m.enabled)
        }

    def models_for_mode(self, mode: str | None = None) -> dict[str, ModelDefinition]:
        """Filter models by mode tag.

        - None: all enabled models (no mode filtering, backward compatible)
        - "big"/"default"/etc: include models with matching mode tag + models with no mode tag
        - Models with a different mode tag are excluded
        - enabled=False models are always excluded
        """
        result = {}
        for mid, m in self.models.items():
            if not m.enabled:
                continue
            if mode is None:
                result[mid] = m
                continue
            model_mode = m.mode_tag
            if model_mode is None or model_mode == mode:
                result[mid] = m
        return result


def load_registry(path: Path | None = None) -> ModelRegistry:
    """Load and validate the model registry from YAML."""
    path = path or REGISTRY_PATH
    with open(path) as f:
        data = yaml.safe_load(f)
    return ModelRegistry.model_validate(data)


def load_registry_from_dict(data: dict[str, Any]) -> ModelRegistry:
    """Load and validate the model registry from a dict (for testing)."""
    return ModelRegistry.model_validate(data)
