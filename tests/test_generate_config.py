"""Tests for LiteLLM config generation."""

from llm_router.config import load_registry_from_dict
from llm_router.generate_config import generate_litellm_config, generate_node_config

SAMPLE_REGISTRY = {
    "nodes": {
        "archimedes": {
            "host": "archimedes",
            "gpu": "nvidia",
            "vram_gb": 128,
            "agent_port": 8100,
        },
    },
    "models": {
        "qwen3-30b": {
            "hf_repo": "Qwen/Qwen3-30B-A3B",
            "backend": "vllm",
            "node": "archimedes",
            "vram_gb": 20,
            "always_on": True,
            "tool_proxy": True,
            "aliases": ["medium-coding", "medium-research"],
            "capabilities": ["text"],
            "vllm_args": {"tool_call_parser": "qwen3_xml"},
        },
        "devstral": {
            "hf_repo": "mistralai/Devstral-Small-2505",
            "backend": "vllm",
            "node": "archimedes",
            "vram_gb": 28,
            "always_on": False,
            "tool_proxy": False,
        },
    },
}


def test_generate_litellm_config_model_count():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    config = generate_litellm_config(reg)
    # qwen3-30b + 2 aliases + devstral = 4
    assert len(config["model_list"]) == 4


def test_tool_proxy_routes_through_5392():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    config = generate_litellm_config(reg)
    qwen_entry = config["model_list"][0]
    assert qwen_entry["model_name"] == "qwen3-30b"
    assert "5392" in qwen_entry["litellm_params"]["api_base"]


def test_no_tool_proxy_routes_through_5391():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    config = generate_litellm_config(reg)
    devstral_entry = next(e for e in config["model_list"] if e["model_name"] == "devstral")
    assert "5391" in devstral_entry["litellm_params"]["api_base"]


def test_aliases_generate_entries():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    config = generate_litellm_config(reg)
    names = [e["model_name"] for e in config["model_list"]]
    assert "medium-coding" in names
    assert "medium-research" in names


def test_model_info_metadata():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    config = generate_litellm_config(reg)
    entry = config["model_list"][0]
    info = entry["model_info"]
    assert info["id"] == "qwen3-30b"
    assert info["always_on"] is True
    assert info["tool_proxy"] is True
    assert info["vram_gb"] == 20


def test_generate_node_config():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    node_cfg = generate_node_config(reg, "archimedes")
    assert node_cfg["node"]["name"] == "archimedes"
    assert node_cfg["node"]["gpu"] == "nvidia"
    assert "qwen3-30b" in node_cfg["models"]
    assert "devstral" in node_cfg["models"]


def test_litellm_settings():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    config = generate_litellm_config(reg)
    assert config["litellm_settings"]["drop_params"] is True
    assert "general_settings" in config
