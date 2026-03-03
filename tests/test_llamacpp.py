"""Tests for llama.cpp backend."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from llm_router.config import BackendType, ModelDefinition, VllmArgs
from llm_router.node_agent.backends.llamacpp import LlamaCppBackend
from llm_router.node_agent.models import ModelState


def _make_model(**kwargs) -> ModelDefinition:
    defaults = {
        "hf_repo": "Qwen/Qwen3-30B-A3B-GGUF",
        "backend": BackendType.LLAMACPP,
        "node": "delphi",
        "gguf_file": "/data/models/test.gguf",
        "vram_gb": 20,
    }
    defaults.update(kwargs)
    return ModelDefinition(**defaults)


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "llm_router.node_agent.backends.llamacpp.PID_DIR", tmp_path / "pids"
    )
    monkeypatch.setattr(
        "llm_router.node_agent.backends.llamacpp.LOG_DIR", tmp_path / "logs"
    )
    return LlamaCppBackend(port=15391)


def test_build_command_basic(backend):
    model = _make_model()
    with patch(
        "llm_router.node_agent.backends.llamacpp._find_llama_server",
        return_value="/usr/bin/llama-server",
    ):
        cmd = backend._build_command("test-model", model)

    assert cmd[0] == "/usr/bin/llama-server"
    assert "--model" in cmd
    assert "/data/models/test.gguf" in cmd
    assert "--port" in cmd
    assert "15391" in cmd
    assert "--n-gpu-layers" in cmd
    assert "999" in cmd


def test_build_command_with_ctx_size(backend):
    model = _make_model(vllm_args=VllmArgs(max_model_len=32768))
    with patch(
        "llm_router.node_agent.backends.llamacpp._find_llama_server",
        return_value="/usr/bin/llama-server",
    ):
        cmd = backend._build_command("test-model", model)

    assert "--ctx-size" in cmd
    idx = cmd.index("--ctx-size")
    assert cmd[idx + 1] == "32768"


def test_build_command_with_extra_args(backend):
    model = _make_model(
        vllm_args=VllmArgs(extra_args=["--threads", "8"])
    )
    with patch(
        "llm_router.node_agent.backends.llamacpp._find_llama_server",
        return_value="/usr/bin/llama-server",
    ):
        cmd = backend._build_command("test-model", model)

    assert "--threads" in cmd
    assert "8" in cmd


@pytest.mark.asyncio
async def test_status_default_stopped(backend):
    status = await backend.status("nonexistent")
    assert status.state == ModelState.STOPPED
    assert status.pid is None
    assert status.port is None


@pytest.mark.asyncio
async def test_start_and_status(backend):
    model = _make_model()
    mock_proc = AsyncMock()
    mock_proc.pid = 99999
    mock_proc.returncode = None

    with (
        patch(
            "llm_router.node_agent.backends.llamacpp._find_llama_server",
            return_value="/usr/bin/llama-server",
        ),
        patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ),
        patch(
            "asyncio.create_task",
            return_value=MagicMock(),
        ),
    ):
        await backend.start("test-model", model)

    # After start, state should be STARTING (readiness check runs in bg)
    status = await backend.status("test-model")
    assert status.state == ModelState.STARTING


@pytest.mark.asyncio
async def test_stop(backend):
    model = _make_model()
    mock_proc = AsyncMock()
    mock_proc.pid = 99999
    mock_proc.returncode = None
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.send_signal = MagicMock()

    with (
        patch(
            "llm_router.node_agent.backends.llamacpp._find_llama_server",
            return_value="/usr/bin/llama-server",
        ),
        patch(
            "asyncio.create_subprocess_exec",
            return_value=mock_proc,
        ),
        patch(
            "asyncio.create_task",
            return_value=MagicMock(),
        ),
    ):
        await backend.start("test-model", model)

    await backend.stop("test-model")
    status = await backend.status("test-model")
    assert status.state == ModelState.STOPPED


@pytest.mark.asyncio
async def test_health_check_success(backend):
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_client
        )
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await backend.health_check("test-model")

    assert result is True


@pytest.mark.asyncio
async def test_health_check_failure(backend):
    with patch("httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value.__aenter__ = AsyncMock(
            return_value=mock_client
        )
        mock_client_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        result = await backend.health_check("test-model")

    assert result is False


def test_find_llama_server_not_found():
    with (
        patch("shutil.which", return_value=None),
        pytest.raises(RuntimeError, match="llama-server binary not found"),
    ):
        from llm_router.node_agent.backends.llamacpp import _find_llama_server

        _find_llama_server()
