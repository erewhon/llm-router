"""Tests for node agent API endpoints."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from llm_router.config import load_registry_from_dict
from llm_router.node_agent.main import create_app
from llm_router.node_agent.models import ModelState, ProcessStatus

SAMPLE_REGISTRY = {
    "nodes": {
        "testnode": {
            "host": "testnode",
            "gpu": "nvidia",
            "vram_gb": 64,
            "agent_port": 8100,
            "services": {
                "comfyui": {
                    "type": "comfyui",
                    "port": 8188,
                    "label": "ComfyUI",
                }
            },
        },
    },
    "models": {
        "test-model": {
            "hf_repo": "org/TestModel-7B",
            "backend": "vllm",
            "node": "testnode",
            "vram_gb": 10,
            "always_on": False,
            "tool_proxy": False,
        },
    },
}


@pytest.fixture
def client():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    app = create_app(reg, "testnode")
    return TestClient(app, raise_server_exceptions=False)


def test_health(client):
    # gpu.get_gpu_info is imported locally inside the health endpoint
    from llm_router.node_agent.services import ServiceStatus

    mock_svc = ServiceStatus(
        name="comfyui",
        service_type="comfyui",
        label="ComfyUI",
        reachable=True,
    )
    with (
        patch("llm_router.node_agent.gpu.get_gpu_info", side_effect=RuntimeError("no GPU")),
        patch(
            "llm_router.node_agent.services.probe_service",
            new_callable=AsyncMock,
            return_value=mock_svc,
        ),
    ):
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["node"] == "testnode"
    assert data["status"] == "ok"
    assert len(data["services"]) == 1
    svc = data["services"][0]
    assert svc["name"] == "comfyui"
    assert svc["reachable"] is True


def test_list_models(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["model_id"] == "test-model"
    assert data[0]["state"] == "stopped"


def test_model_status(client):
    resp = client.get("/models/test-model/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "test-model"
    assert data["state"] == "stopped"


def test_model_not_found(client):
    resp = client.get("/models/nonexistent/status")
    assert resp.status_code == 404


def test_start_model(client):
    mock_status = ProcessStatus(
        model_id="test-model",
        state=ModelState.STARTING,
        pid=12345,
        port=5391,
    )
    with (
        patch(
            "llm_router.node_agent.backends.vllm.VllmBackend.start",
            new_callable=AsyncMock,
        ),
        patch(
            "llm_router.node_agent.backends.vllm.VllmBackend.status",
            new_callable=AsyncMock,
            return_value=mock_status,
        ),
    ):
        resp = client.post("/models/test-model/start")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_id"] == "test-model"


def test_stop_model(client):
    mock_status = ProcessStatus(
        model_id="test-model",
        state=ModelState.STOPPED,
    )
    with (
        patch(
            "llm_router.node_agent.backends.vllm.VllmBackend.stop",
            new_callable=AsyncMock,
        ),
        patch(
            "llm_router.node_agent.backends.vllm.VllmBackend.status",
            new_callable=AsyncMock,
            return_value=mock_status,
        ),
    ):
        resp = client.post("/models/test-model/stop")
    assert resp.status_code == 200
    data = resp.json()
    assert data["state"] == "stopped"
