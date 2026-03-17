"""Tests for config parsing and validation."""


import pytest

from llm_router.config import (
    BackendType,
    GpuType,
    ModelCapability,
    ModelRegistry,
    ServiceType,
    load_registry,
    load_registry_from_dict,
)

SAMPLE_REGISTRY = {
    "nodes": {
        "archimedes": {
            "host": "archimedes",
            "gpu": "nvidia",
            "vram_gb": 128,
            "agent_port": 8100,
        },
        "delphi": {
            "host": "delphi",
            "gpu": "amd",
            "vram_gb": 64,
            "agent_port": 8100,
        },
    },
    "models": {
        "test-model": {
            "hf_repo": "org/TestModel-7B",
            "backend": "vllm",
            "node": "archimedes",
            "vram_gb": 10,
            "always_on": True,
            "tool_proxy": True,
            "aliases": ["small-test"],
            "capabilities": ["text"],
        },
        "vision-model": {
            "hf_repo": "org/VisionModel-13B",
            "backend": "vllm",
            "node": "delphi",
            "vram_gb": 30,
            "always_on": False,
            "tool_proxy": False,
            "capabilities": ["text", "vision"],
        },
    },
}


def test_load_registry_from_dict():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    assert isinstance(reg, ModelRegistry)
    assert len(reg.nodes) == 2
    assert len(reg.models) == 2


def test_node_gpu_type():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    assert reg.nodes["archimedes"].gpu == GpuType.NVIDIA
    assert reg.nodes["delphi"].gpu == GpuType.AMD


def test_model_fields():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    m = reg.models["test-model"]
    assert m.hf_repo == "org/TestModel-7B"
    assert m.backend == BackendType.VLLM
    assert m.node == "archimedes"
    assert m.vram_gb == 10
    assert m.always_on is True
    assert m.tool_proxy is True
    assert m.aliases == ["small-test"]
    assert ModelCapability.TEXT in m.capabilities


def test_model_capabilities_vision():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    m = reg.models["vision-model"]
    assert ModelCapability.TEXT in m.capabilities
    assert ModelCapability.VISION in m.capabilities


def test_get_node():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    node = reg.get_node("test-model")
    assert node.host == "archimedes"
    assert node.gpu == GpuType.NVIDIA


def test_get_api_base_with_tool_proxy():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    base = reg.get_api_base("test-model")
    assert base == "http://delphi:5392/v1"


def test_get_api_base_without_tool_proxy():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    base = reg.get_api_base("vision-model")
    assert base == "http://delphi:5391/v1"


def test_models_for_node():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    arch_models = reg.models_for_node("archimedes")
    assert "test-model" in arch_models
    assert "vision-model" not in arch_models

    delphi_models = reg.models_for_node("delphi")
    assert "vision-model" in delphi_models
    assert "test-model" not in delphi_models


def test_model_must_have_node_or_multi_node():
    data = {
        "nodes": {"n": {"host": "n", "gpu": "nvidia", "vram_gb": 32}},
        "models": {
            "bad": {
                "hf_repo": "org/Model",
                "backend": "vllm",
                # neither node nor multi_node
            }
        },
    }
    with pytest.raises(ValueError):
        load_registry_from_dict(data)


def test_model_cannot_have_both_node_and_multi_node():
    data = {
        "nodes": {
            "a": {"host": "a", "gpu": "nvidia", "vram_gb": 32},
            "b": {"host": "b", "gpu": "nvidia", "vram_gb": 32},
        },
        "models": {
            "bad": {
                "hf_repo": "org/Model",
                "backend": "vllm",
                "node": "a",
                "multi_node": {
                    "nodes": ["a", "b"],
                    "tensor_parallel_size": 2,
                },
            }
        },
    }
    with pytest.raises(ValueError):
        load_registry_from_dict(data)


def test_load_real_registry():
    """Validate the actual models.yaml in the project."""
    reg = load_registry()
    assert len(reg.models) > 0
    assert len(reg.nodes) > 0
    for model_id, model in reg.models.items():
        if model.node:
            assert model.node in reg.nodes, f"Model {model_id} references unknown node {model.node}"


def test_default_capabilities():
    """Model with no capabilities specified gets [text]."""
    data = {
        "nodes": {"n": {"host": "n", "gpu": "nvidia", "vram_gb": 32}},
        "models": {
            "m": {
                "hf_repo": "org/Model",
                "node": "n",
            }
        },
    }
    reg = load_registry_from_dict(data)
    assert reg.models["m"].capabilities == [ModelCapability.TEXT]


def test_node_without_services():
    """Node without services field defaults to empty dict."""
    data = {
        "nodes": {"n": {"host": "n", "gpu": "nvidia", "vram_gb": 32}},
        "models": {"m": {"hf_repo": "org/Model", "node": "n"}},
    }
    reg = load_registry_from_dict(data)
    assert reg.nodes["n"].services == {}


def test_node_with_services():
    """Node with services parses correctly."""
    data = {
        "nodes": {
            "n": {
                "host": "n",
                "gpu": "nvidia",
                "vram_gb": 128,
                "services": {
                    "comfyui": {
                        "type": "comfyui",
                        "port": 8188,
                        "label": "ComfyUI",
                    }
                },
            }
        },
        "models": {"m": {"hf_repo": "org/Model", "node": "n"}},
    }
    reg = load_registry_from_dict(data)
    assert "comfyui" in reg.nodes["n"].services
    svc = reg.nodes["n"].services["comfyui"]
    assert svc.type == ServiceType.COMFYUI
    assert svc.port == 8188
    assert svc.label == "ComfyUI"
