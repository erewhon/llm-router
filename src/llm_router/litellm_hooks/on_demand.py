"""LiteLLM custom hook for on-demand model startup.

When a request arrives for an on-demand model, this hook calls the
node agent to start the model if it's not already running, then
waits for it to become ready before allowing the request through.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("on-demand-hook")

# How long to wait for a model to start (seconds)
START_TIMEOUT = 180
# How often to poll status (seconds)
POLL_INTERVAL = 2


class OnDemandModelHook(CustomLogger):
    """LiteLLM CustomLogger that starts on-demand models before requests."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: dict,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> dict:
        """Called before each LLM API call. Start model if needed."""
        model_info = data.get("metadata", {}).get("model_info", {})

        # Only intervene for on-demand models
        if model_info.get("always_on", True):
            return data

        model_id = model_info.get("id")
        node = model_info.get("node")
        if not model_id or not node:
            return data

        # Check if model is running by querying the node agent
        agent_url = f"http://{node}:8100"

        async with httpx.AsyncClient() as client:
            try:
                status_resp = await client.get(
                    f"{agent_url}/models/{model_id}/status",
                    timeout=5,
                )
                status = status_resp.json()

                if status.get("state") == "running":
                    return data

                # Model not running — start it
                logger.info(f"Starting on-demand model {model_id} on {node}")
                await client.post(
                    f"{agent_url}/models/{model_id}/start",
                    timeout=10,
                )

                # Wait for model to become ready
                for _ in range(START_TIMEOUT // POLL_INTERVAL):
                    await asyncio.sleep(POLL_INTERVAL)
                    status_resp = await client.get(
                        f"{agent_url}/models/{model_id}/status",
                        timeout=5,
                    )
                    status = status_resp.json()

                    if status.get("state") == "running":
                        logger.info(f"Model {model_id} is ready")
                        return data

                    if status.get("state") == "error":
                        error = status.get("error", "unknown error")
                        logger.error(f"Model {model_id} failed to start: {error}")
                        raise Exception(f"Model {model_id} failed to start: {error}")

                raise TimeoutError(
                    f"Model {model_id} did not become ready within {START_TIMEOUT}s"
                )

            except httpx.ConnectError as e:
                logger.error(f"Cannot reach node agent at {agent_url}")
                raise Exception(f"Node agent unreachable at {agent_url}") from e

        return data
