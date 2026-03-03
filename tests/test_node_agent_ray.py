"""Tests for node agent Ray API endpoints."""

from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from llm_router.config import load_registry_from_dict
from llm_router.node_agent.main import create_app
from llm_router.node_agent.ray_cluster import RayNodeStatus, RayRole, RayState

SAMPLE_REGISTRY = {
    "nodes": {
        "testnode": {
            "host": "testnode",
            "gpu": "nvidia",
            "vram_gb": 64,
            "agent_port": 8100,
        },
    },
    "models": {
        "test-model": {
            "hf_repo": "org/TestModel-7B",
            "backend": "vllm",
            "node": "testnode",
            "vram_gb": 10,
        },
    },
}


def _make_client():
    reg = load_registry_from_dict(SAMPLE_REGISTRY)
    app = create_app(reg, "testnode")
    return TestClient(app, raise_server_exceptions=False)


def test_ray_status_initial():
    client = _make_client()
    resp = client.get("/ray/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "none"
    assert data["state"] == "stopped"


def test_ray_join_head():
    client = _make_client()
    mock_status = RayNodeStatus(
        role=RayRole.HEAD,
        state=RayState.RUNNING,
        head_address="127.0.0.1:6379",
        pid=12345,
    )
    with patch(
        "llm_router.node_agent.ray_cluster.RayClusterManager.start_head",
        new_callable=AsyncMock,
        return_value=mock_status,
    ):
        resp = client.post("/ray/join", json={"role": "head"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "head"
    assert data["state"] == "running"
    assert data["head_address"] == "127.0.0.1:6379"


def test_ray_join_worker():
    client = _make_client()
    mock_status = RayNodeStatus(
        role=RayRole.WORKER,
        state=RayState.RUNNING,
        head_address="head:6379",
        pid=54321,
    )
    with patch(
        "llm_router.node_agent.ray_cluster.RayClusterManager.start_worker",
        new_callable=AsyncMock,
        return_value=mock_status,
    ):
        resp = client.post(
            "/ray/join",
            json={"role": "worker", "head_address": "head:6379"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "worker"
    assert data["state"] == "running"


def test_ray_join_worker_missing_address():
    client = _make_client()
    resp = client.post("/ray/join", json={"role": "worker"})
    assert resp.status_code == 400


def test_ray_join_invalid_role():
    client = _make_client()
    resp = client.post("/ray/join", json={"role": "invalid"})
    assert resp.status_code == 400


def test_ray_leave():
    client = _make_client()
    mock_status = RayNodeStatus(
        role=RayRole.NONE,
        state=RayState.STOPPED,
    )
    with patch(
        "llm_router.node_agent.ray_cluster.RayClusterManager.stop",
        new_callable=AsyncMock,
        return_value=mock_status,
    ):
        resp = client.post("/ray/leave")

    assert resp.status_code == 200
    data = resp.json()
    assert data["role"] == "none"
    assert data["state"] == "stopped"
