"""Generate LiteLLM proxy config from models.yaml."""

from pathlib import Path

import click
import yaml

from llm_router.config import ModelDefinition, ModelRegistry, load_registry

DEFAULT_OUTPUT = Path(__file__).parents[2] / "deploy" / "litellm" / "config.yaml"


def _litellm_model_entry(
    model_id: str,
    model: ModelDefinition,
    registry: ModelRegistry,
) -> dict:
    """Build a single LiteLLM model_list entry."""
    api_base = registry.get_api_base(model_id)

    entry: dict = {
        "model_name": model_id,
        "litellm_params": {
            "model": f"openai/{model.hf_repo}",
            "api_base": api_base,
            "api_key": "not-needed",
        },
        "model_info": {
            "id": model_id,
            "node": model.node,
            "backend": model.backend.value,
            "always_on": model.always_on,
            "tool_proxy": model.tool_proxy,
            "vram_gb": model.vram_gb,
            "capabilities": [c.value for c in model.capabilities],
            **({"tags": model.tags} if model.tags else {}),
            **(
                {
                    "multi_node": {
                        "nodes": model.multi_node.nodes,
                        "tensor_parallel_size": model.multi_node.tensor_parallel_size,
                        "head_node": model.multi_node.head_node or model.multi_node.nodes[0],
                    }
                }
                if model.multi_node
                else {}
            ),
        },
    }
    return entry


def generate_litellm_config(registry: ModelRegistry) -> dict:
    """Generate the full LiteLLM config dict."""
    model_list: list[dict] = []

    for model_id, model in registry.models.items():
        if not model.enabled:
            continue
        # Primary entry
        entry = _litellm_model_entry(model_id, model, registry)
        model_list.append(entry)

        # Alias entries — each alias points to the same backend
        for alias in model.aliases:
            alias_entry = _litellm_model_entry(model_id, model, registry)
            alias_entry["model_name"] = alias
            model_list.append(alias_entry)

    config: dict = {
        "model_list": model_list,
        "litellm_settings": {
            "drop_params": True,
            "request_timeout": 600,
        },
        "general_settings": {},
    }
    return config


def generate_node_config(registry: ModelRegistry, node_name: str) -> dict:
    """Generate a node-specific config for the node agent."""
    node = registry.nodes[node_name]
    models = registry.models_for_node(node_name)

    return {
        "node": {
            "name": node_name,
            "host": node.host,
            "gpu": node.gpu.value,
            "vram_gb": node.vram_gb,
            "agent_port": node.agent_port,
        },
        "models": {
            mid: {
                "hf_repo": m.hf_repo,
                "backend": m.backend.value,
                "vram_gb": m.vram_gb,
                "always_on": m.always_on,
                "tool_proxy": m.tool_proxy,
                **({"gguf_file": m.gguf_file} if m.gguf_file else {}),
                "vllm_args": m.vllm_args.model_dump(exclude_none=True),
            }
            for mid, m in models.items()
        },
    }


@click.command()
@click.option(
    "--registry",
    "-r",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to models.yaml (default: project root)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for LiteLLM config (default: deploy/litellm/config.yaml)",
)
@click.option(
    "--node-configs",
    is_flag=True,
    default=False,
    help="Also generate per-node config files in deploy/",
)
def main(registry: Path | None, output: Path | None, node_configs: bool) -> None:
    """Generate LiteLLM proxy config from models.yaml."""
    reg = load_registry(registry)
    output = output or DEFAULT_OUTPUT

    litellm_config = generate_litellm_config(reg)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        yaml.dump(litellm_config, f, default_flow_style=False, sort_keys=False)
    click.echo(f"Wrote LiteLLM config: {output}")
    click.echo(f"  {len(litellm_config['model_list'])} model entries")

    if node_configs:
        for node_name in reg.nodes:
            node_cfg = generate_node_config(reg, node_name)
            node_path = output.parent.parent / f"node-{node_name}.yaml"
            with open(node_path, "w") as f:
                yaml.dump(node_cfg, f, default_flow_style=False, sort_keys=False)
            click.echo(f"Wrote node config: {node_path}")


if __name__ == "__main__":
    main()
