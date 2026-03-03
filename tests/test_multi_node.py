"""Tests for multi-node model config, routing, and generation."""

from llm_router.config import load_registry_from_dict
from llm_router.generate_config import generate_litellm_config, generate_node_config

MULTI_NODE_REGISTRY = {
    "nodes": {
        "archimedes": {
            "host": "archimedes",
            "gpu": "nvidia",
            "vram_gb": 128,
            "agent_port": 8100,
        },
        "hypatia": {
            "host": "hypatia",
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
        "llama4-scout": {
            "hf_repo": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            "backend": "vllm",
            "multi_node": {
                "nodes": ["archimedes", "hypatia"],
                "tensor_parallel_size": 2,
            },
            "vram_gb": 120,
            "tool_proxy": True,
            "capabilities": ["text", "vision"],
        },
        "gguf-model": {
            "hf_repo": "Qwen/Qwen3-30B-A3B-GGUF",
            "backend": "llamacpp",
            "node": "delphi",
            "gguf_file": "/data/models/test.gguf",
            "vram_gb": 20,
        },
    },
}


def test_multi_node_model_routing():
    """Multi-node model should route through head node (first in list)."""
    reg = load_registry_from_dict(MULTI_NODE_REGISTRY)
    base = reg.get_api_base("llama4-scout")
    # tool_proxy=True → port 5392, head node is archimedes
    assert base == "http://archimedes:5392/v1"


def test_multi_node_model_assigned_to_both_nodes():
    """Multi-node model should appear in models_for_node for all nodes."""
    reg = load_registry_from_dict(MULTI_NODE_REGISTRY)
    arch = reg.models_for_node("archimedes")
    hyp = reg.models_for_node("hypatia")
    assert "llama4-scout" in arch
    assert "llama4-scout" in hyp
    # GGUF model only on delphi
    assert "gguf-model" not in arch
    assert "gguf-model" not in hyp


def test_multi_node_litellm_config():
    """LiteLLM config should include multi_node metadata."""
    reg = load_registry_from_dict(MULTI_NODE_REGISTRY)
    config = generate_litellm_config(reg)

    scout_entry = next(
        e for e in config["model_list"] if e["model_name"] == "llama4-scout"
    )
    info = scout_entry["model_info"]
    assert "multi_node" in info
    assert info["multi_node"]["nodes"] == ["archimedes", "hypatia"]
    assert info["multi_node"]["tensor_parallel_size"] == 2
    assert info["multi_node"]["head_node"] == "archimedes"


def test_llamacpp_model_routing():
    """llamacpp model should route directly through 5391."""
    reg = load_registry_from_dict(MULTI_NODE_REGISTRY)
    base = reg.get_api_base("gguf-model")
    assert base == "http://delphi:5391/v1"


def test_node_config_includes_gguf_file():
    """Node config for delphi should include gguf_file for llamacpp model."""
    reg = load_registry_from_dict(MULTI_NODE_REGISTRY)
    node_cfg = generate_node_config(reg, "delphi")
    assert "gguf-model" in node_cfg["models"]
    assert node_cfg["models"]["gguf-model"]["gguf_file"] == "/data/models/test.gguf"
    assert node_cfg["models"]["gguf-model"]["backend"] == "llamacpp"


def test_multi_node_with_custom_head():
    """Multi-node model with explicit head_node."""
    data = {
        "nodes": {
            "a": {"host": "a", "gpu": "nvidia", "vram_gb": 128},
            "b": {"host": "b", "gpu": "nvidia", "vram_gb": 128},
        },
        "models": {
            "m": {
                "hf_repo": "org/Model",
                "backend": "vllm",
                "multi_node": {
                    "nodes": ["a", "b"],
                    "tensor_parallel_size": 2,
                    "head_node": "b",
                },
                "vram_gb": 100,
                "tool_proxy": False,
            }
        },
    }
    reg = load_registry_from_dict(data)
    # Head is explicitly "b", not first in list
    base = reg.get_api_base("m")
    assert base == "http://b:5391/v1"


def test_vllm_build_command_multi_node():
    """vLLM backend should add tensor-parallel and ray flags for multi-node."""
    from unittest.mock import patch

    from llm_router.config import BackendType, ModelDefinition, MultiNodeConfig
    from llm_router.node_agent.backends.vllm import VllmBackend

    model = ModelDefinition(
        hf_repo="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        backend=BackendType.VLLM,
        multi_node=MultiNodeConfig(
            nodes=["archimedes", "hypatia"],
            tensor_parallel_size=2,
        ),
        vram_gb=120,
    )

    backend = VllmBackend()

    with patch(
        "llm_router.node_agent.backends.vllm.get_gpu_info",
        side_effect=RuntimeError("no GPU"),
    ):
        cmd = backend._build_command("llama4-scout", model)

    assert "--tensor-parallel-size" in cmd
    idx = cmd.index("--tensor-parallel-size")
    assert cmd[idx + 1] == "2"
    assert "--distributed-executor-backend" in cmd
    idx2 = cmd.index("--distributed-executor-backend")
    assert cmd[idx2 + 1] == "ray"
