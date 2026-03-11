"""Tests for the service prober module."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from llm_router.config import ServiceDefinition, ServiceType
from llm_router.node_agent.services import probe_comfyui, probe_service


@pytest.mark.asyncio
async def test_probe_comfyui_success():
    """ComfyUI probe returns VRAM and queue info on success."""
    system_stats = {
        "devices": [
            {
                "name": "cuda:0",
                "vram_total": 128_849_018_880,  # ~120 GB
                "vram_free": 68_719_476_736,  # ~64 GB
            }
        ]
    }
    queue_data = {
        "queue_running": [["item1"]],
        "queue_pending": [],
    }

    async def mock_get(url, **kwargs):
        resp = AsyncMock(spec=httpx.Response)
        resp.status_code = 200
        if "/system_stats" in url:
            resp.json.return_value = system_stats
        elif "/queue" in url:
            resp.json.return_value = queue_data
        return resp

    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("llm_router.node_agent.services.httpx.AsyncClient", return_value=mock_client):
        status = await probe_comfyui("comfyui", "hypatia.local", 8188, label="ComfyUI")

    assert status.reachable is True
    # ComfyUI /system_stats reports system-wide VRAM, not ComfyUI-specific,
    # so we no longer extract per-service VRAM from it.
    assert status.vram_total_gb is None
    assert status.vram_used_gb is None
    assert status.queue_running == 1
    assert status.queue_pending == 0


@pytest.mark.asyncio
async def test_probe_comfyui_unreachable():
    """ComfyUI probe returns reachable=False when connection fails."""
    mock_client = AsyncMock(spec=httpx.AsyncClient)
    mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("llm_router.node_agent.services.httpx.AsyncClient", return_value=mock_client):
        status = await probe_comfyui("comfyui", "hypatia.local", 8188)

    assert status.reachable is False
    assert status.vram_used_gb is None
    assert status.vram_total_gb is None


@pytest.mark.asyncio
async def test_probe_service_dispatches_comfyui():
    """probe_service dispatches to probe_comfyui for ComfyUI type."""
    svc = ServiceDefinition(type=ServiceType.COMFYUI, port=8188, label="ComfyUI")

    with patch(
        "llm_router.node_agent.services.probe_comfyui",
        new_callable=AsyncMock,
    ) as mock_probe:
        mock_probe.return_value = AsyncMock(
            name="comfyui",
            service_type="comfyui",
            label="ComfyUI",
            reachable=True,
        )
        await probe_service("comfyui", svc, "hypatia.local")
        mock_probe.assert_awaited_once_with("comfyui", "hypatia.local", 8188, label="ComfyUI")
