"""Tests for Ray cluster management."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llm_router.node_agent.ray_cluster import (
    RayClusterManager,
    RayRole,
    RayState,
)

FIND_RAY = "llm_router.node_agent.ray_cluster._find_ray"


@pytest.fixture
def manager():
    return RayClusterManager()


def test_initial_status(manager):
    status = manager.status
    assert status.role == RayRole.NONE
    assert status.state == RayState.STOPPED
    assert status.pid is None
    assert status.head_address is None
    assert status.error is None


@pytest.mark.asyncio
async def test_start_head(manager):
    mock_proc = AsyncMock()
    mock_proc.pid = 12345
    mock_proc.returncode = None
    mock_proc.stdout = AsyncMock()

    with (
        patch(FIND_RAY, return_value="/usr/bin/ray"),
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
    ):
        status = await manager.start_head(port=6379)

    assert status.role == RayRole.HEAD
    assert status.state == RayState.RUNNING
    assert status.pid == 12345
    assert status.head_address == "127.0.0.1:6379"


@pytest.mark.asyncio
async def test_start_head_failure(manager):
    mock_proc = AsyncMock()
    mock_proc.pid = 12345
    mock_proc.returncode = 1  # failed immediately
    mock_proc.stdout = AsyncMock()
    mock_proc.stdout.read = AsyncMock(return_value=b"error msg")

    with (
        patch(FIND_RAY, return_value="/usr/bin/ray"),
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
    ):
        status = await manager.start_head()

    assert status.state == RayState.ERROR
    assert "failed to start" in status.error.lower()


@pytest.mark.asyncio
async def test_start_worker(manager):
    mock_proc = AsyncMock()
    mock_proc.pid = 54321
    mock_proc.returncode = None
    mock_proc.stdout = AsyncMock()

    with (
        patch(FIND_RAY, return_value="/usr/bin/ray"),
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
    ):
        status = await manager.start_worker("head-node:6379")

    assert status.role == RayRole.WORKER
    assert status.state == RayState.RUNNING
    assert status.pid == 54321
    assert status.head_address == "head-node:6379"


@pytest.mark.asyncio
async def test_stop(manager):
    mock_proc = AsyncMock()
    mock_proc.pid = 12345
    mock_proc.returncode = None
    mock_proc.stdout = AsyncMock()
    mock_proc.wait = AsyncMock(return_value=0)
    mock_proc.terminate = MagicMock()

    with (
        patch(FIND_RAY, return_value="/usr/bin/ray"),
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
    ):
        await manager.start_head()

    # Now stop — also mock the `ray stop` subprocess
    mock_proc.returncode = None  # still running when stop() is called
    stop_proc = AsyncMock()
    stop_proc.wait = AsyncMock(return_value=0)

    with (
        patch("shutil.which", return_value="/usr/bin/ray"),
        patch("asyncio.create_subprocess_exec", return_value=stop_proc),
    ):
        mock_proc.wait = AsyncMock(return_value=0)
        status = await manager.stop()

    assert status.role == RayRole.NONE
    assert status.state == RayState.STOPPED


@pytest.mark.asyncio
async def test_already_running_returns_status(manager):
    mock_proc = AsyncMock()
    mock_proc.pid = 12345
    mock_proc.returncode = None
    mock_proc.stdout = AsyncMock()

    with (
        patch(FIND_RAY, return_value="/usr/bin/ray"),
        patch("asyncio.create_subprocess_exec", return_value=mock_proc),
    ):
        await manager.start_head()
        # Second call should return existing status without starting again
        status = await manager.start_head()

    assert status.state == RayState.RUNNING
    assert status.pid == 12345


def test_status_detects_dead_process(manager):
    """If the process dies, status should reflect ERROR."""
    mock_proc = MagicMock()
    mock_proc.pid = 12345
    mock_proc.returncode = 1  # dead

    manager._process = mock_proc
    manager._state = RayState.RUNNING
    manager._role = RayRole.HEAD

    status = manager.status
    assert status.state == RayState.ERROR
    assert "exited" in status.error.lower()
