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
    *,
    effective_tool_proxy: bool | None = None,
) -> dict:
    """Build a single LiteLLM model_list entry.

    When `effective_tool_proxy` is set (used by alias entries with a per-alias
    override), it overrides the model's tool_proxy flag for routing + naming.
    """
    use_tool_proxy = effective_tool_proxy if effective_tool_proxy is not None else model.tool_proxy
    api_base = registry.get_api_base(model_id, tool_proxy=use_tool_proxy)

    # Resolve API key: external models use their own key, local models don't need one
    if model.api_key:
        api_key = f"os.environ/{model.api_key}" if not model.api_key.startswith("sk-") else model.api_key
    else:
        api_key = "not-needed"

    # For tool_proxy entries, send model_id (not hf_repo) so the tool proxy
    # can disambiguate when multiple models share the same hf_repo.
    model_name_for_backend = model_id if use_tool_proxy else model.hf_repo

    entry: dict = {
        "model_name": model_id,
        "litellm_params": {
            "model": f"openai/{model_name_for_backend}",
            "api_base": api_base,
            "api_key": api_key,
        },
        "model_info": {
            "id": model_id,
            "node": model.node,
            "backend": model.backend.value,
            "always_on": model.always_on,
            "tool_proxy": use_tool_proxy,
            "vram_gb": model.vram_gb,
            **({"health_check": False} if model.backend.value == "external" else {}),
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
            **(
                {"input_cost_per_token": model.input_cost_per_million / 1_000_000}
                if model.input_cost_per_million
                else {}
            ),
            **(
                {"output_cost_per_token": model.output_cost_per_million / 1_000_000}
                if model.output_cost_per_million
                else {}
            ),
        },
    }
    return entry


def generate_litellm_config(registry: ModelRegistry, *, mode: str | None = None) -> dict:
    """Generate the full LiteLLM config dict."""
    model_list: list[dict] = []

    for model_id, model in registry.models_for_mode(mode).items():
        # Primary entry
        entry = _litellm_model_entry(model_id, model, registry)
        model_list.append(entry)

        # Alias entries — each alias points to the same backend.
        # Per-alias `tool_proxy` override (from alias_overrides) flips routing
        # for that single alias. When an alias is tool-proxy-routed, forward
        # the alias name (not the canonical id) so the tool proxy can apply
        # alias_overrides logic against the matched alias.
        for alias in model.aliases:
            override = model.alias_overrides.get(alias)
            override_tool_proxy = override.tool_proxy if override else None
            alias_entry = _litellm_model_entry(
                model_id, model, registry, effective_tool_proxy=override_tool_proxy,
            )
            alias_entry["model_name"] = alias
            use_tool_proxy = (
                override_tool_proxy if override_tool_proxy is not None else model.tool_proxy
            )
            if use_tool_proxy:
                alias_entry["litellm_params"]["model"] = f"openai/{alias}"
            model_list.append(alias_entry)

    config: dict = {
        "model_list": model_list,
        "litellm_settings": {
            "drop_params": True,
            "request_timeout": 600,
            "success_callback": ["prometheus", "db"],
        },
        "general_settings": {
            "database_url": "postgresql://litellm:litellm-local@127.0.0.1:5432/litellm",
            "background_health_checks": True,
            "health_check_interval": 300,  # 5 min
        },
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
@click.option(
    "--mode", "-m",
    type=str,
    default=None,
    help="Active mode tag (e.g. 'big', 'default'). Only models with this mode or no mode tag are included.",
)
def main(registry: Path | None, output: Path | None, node_configs: bool, mode: str | None) -> None:
    """Generate LiteLLM proxy config from models.yaml."""
    reg = load_registry(registry)
    output = output or DEFAULT_OUTPUT

    litellm_config = generate_litellm_config(reg, mode=mode)
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
