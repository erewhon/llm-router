"""LiteLLM custom hook for on-demand model startup.

When a request arrives for an on-demand model, this hook calls the
node agent to start the model if it's not already running, then
waits for it to become ready before allowing the request through.

For multi-node models, it coordinates Ray cluster formation across
all participating nodes before starting vLLM.
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
# Longer timeout for multi-node (Ray cluster formation + model load)
MULTI_NODE_START_TIMEOUT = 300
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
        multi_node = model_info.get("multi_node")

        if not model_id:
            return data

        if multi_node:
            await self._start_multi_node(model_id, multi_node)
        elif node:
            await self._start_single_node(model_id, node)

        return data

    async def _start_single_node(self, model_id: str, node: str) -> None:
        """Start a single-node on-demand model."""
        agent_url = f"http://{node}:8100"

        async with httpx.AsyncClient() as client:
            try:
                status = await self._get_status(client, agent_url, model_id)
                if status.get("state") == "running":
                    return

                logger.info(f"Starting on-demand model {model_id} on {node}")
                await client.post(
                    f"{agent_url}/models/{model_id}/start",
                    timeout=10,
                )

                await self._wait_for_ready(client, agent_url, model_id, START_TIMEOUT)

            except httpx.ConnectError as e:
                logger.error(f"Cannot reach node agent at {agent_url}")
                raise Exception(f"Node agent unreachable at {agent_url}") from e

    async def _start_multi_node(self, model_id: str, multi_node: dict) -> None:
        """Start a multi-node on-demand model with Ray cluster coordination."""
        nodes = multi_node.get("nodes", [])
        head_node = multi_node.get("head_node") or nodes[0]
        worker_nodes = [n for n in nodes if n != head_node]

        head_agent = f"http://{head_node}:8100"

        async with httpx.AsyncClient() as client:
            # Check if already running on head
            try:
                status = await self._get_status(client, head_agent, model_id)
                if status.get("state") == "running":
                    return
            except httpx.ConnectError as e:
                raise Exception(f"Head node agent unreachable at {head_agent}") from e

            # Form Ray cluster: start head
            logger.info(f"Starting Ray head on {head_node} for {model_id}")
            try:
                await client.post(
                    f"{head_agent}/ray/join",
                    json={"role": "head"},
                    timeout=15,
                )
            except Exception as e:
                raise Exception(f"Failed to start Ray head on {head_node}: {e}") from e

            # Join workers
            head_address = f"{head_node}:6379"
            for worker in worker_nodes:
                worker_agent = f"http://{worker}:8100"
                logger.info(f"Joining Ray worker {worker} to cluster")
                try:
                    await client.post(
                        f"{worker_agent}/ray/join",
                        json={"role": "worker", "head_address": head_address},
                        timeout=15,
                    )
                except Exception as e:
                    raise Exception(f"Failed to join worker {worker}: {e}") from e

            # Start model on head node (vLLM with Ray distributed executor)
            logger.info(f"Starting multi-node model {model_id} on {head_node}")
            await client.post(
                f"{head_agent}/models/{model_id}/start",
                timeout=10,
            )

            await self._wait_for_ready(
                client, head_agent, model_id, MULTI_NODE_START_TIMEOUT
            )

    async def _get_status(
        self, client: httpx.AsyncClient, agent_url: str, model_id: str
    ) -> dict:
        resp = await client.get(
            f"{agent_url}/models/{model_id}/status",
            timeout=5,
        )
        return resp.json()

    async def _wait_for_ready(
        self,
        client: httpx.AsyncClient,
        agent_url: str,
        model_id: str,
        timeout: int,
    ) -> None:
        """Poll model status until running or failed."""
        for _ in range(timeout // POLL_INTERVAL):
            await asyncio.sleep(POLL_INTERVAL)
            status = await self._get_status(client, agent_url, model_id)

            if status.get("state") == "running":
                logger.info(f"Model {model_id} is ready")
                return

            if status.get("state") == "error":
                error = status.get("error", "unknown error")
                logger.error(f"Model {model_id} failed to start: {error}")
                raise Exception(f"Model {model_id} failed to start: {error}")

        raise TimeoutError(
            f"Model {model_id} did not become ready within {timeout}s"
        )
