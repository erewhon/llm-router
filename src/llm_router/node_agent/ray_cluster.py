"""Ray cluster formation and teardown for multi-node inference.

Manages Ray head and worker processes so that vLLM can run tensor-parallel
inference across multiple GPU nodes (e.g. both Spark machines).
"""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from enum import StrEnum

import httpx

logger = logging.getLogger(__name__)

RAY_DEFAULT_PORT = 6379
RAY_DASHBOARD_PORT = 8265


class RayRole(StrEnum):
    NONE = "none"
    HEAD = "head"
    WORKER = "worker"


class RayState(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class RayNodeStatus:
    role: RayRole = RayRole.NONE
    state: RayState = RayState.STOPPED
    head_address: str | None = None
    pid: int | None = None
    error: str | None = None
    connected_nodes: list[str] = field(default_factory=list)


class RayClusterManager:
    """Manages Ray head/worker processes on a single node."""

    def __init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None
        self._role: RayRole = RayRole.NONE
        self._state: RayState = RayState.STOPPED
        self._head_address: str | None = None
        self._error: str | None = None

    @property
    def status(self) -> RayNodeStatus:
        # Check if process died
        if (
            self._process
            and self._process.returncode is not None
            and self._state == RayState.RUNNING
        ):
            self._state = RayState.ERROR
            self._error = f"Ray process exited with code {self._process.returncode}"

        return RayNodeStatus(
            role=self._role,
            state=self._state,
            head_address=self._head_address,
            pid=self._process.pid if self._process and self._process.returncode is None else None,
            error=self._error,
        )

    async def start_head(self, port: int = RAY_DEFAULT_PORT) -> RayNodeStatus:
        """Start this node as a Ray head node."""
        if self._state == RayState.RUNNING:
            return self.status

        ray_bin = _find_ray()
        self._state = RayState.STARTING
        self._role = RayRole.HEAD
        self._error = None

        cmd = [
            ray_bin,
            "start",
            "--head",
            "--port",
            str(port),
            "--dashboard-port",
            str(RAY_DASHBOARD_PORT),
            "--block",
        ]

        logger.info(f"Starting Ray head on port {port}")
        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            # Give it a moment to start
            await asyncio.sleep(2)

            if self._process.returncode is not None:
                stdout = await self._process.stdout.read() if self._process.stdout else b""
                self._state = RayState.ERROR
                self._error = f"Ray head failed to start: {stdout.decode()[:500]}"
                logger.error(self._error)
            else:
                self._state = RayState.RUNNING
                self._head_address = f"127.0.0.1:{port}"
                logger.info(f"Ray head running (pid {self._process.pid})")

        except Exception as e:
            self._state = RayState.ERROR
            self._error = str(e)
            logger.error(f"Failed to start Ray head: {e}")

        return self.status

    async def start_worker(self, head_address: str) -> RayNodeStatus:
        """Join a Ray cluster as a worker node."""
        if self._state == RayState.RUNNING:
            return self.status

        ray_bin = _find_ray()
        self._state = RayState.STARTING
        self._role = RayRole.WORKER
        self._head_address = head_address
        self._error = None

        cmd = [
            ray_bin,
            "start",
            "--address",
            head_address,
            "--block",
        ]

        logger.info(f"Joining Ray cluster at {head_address}")
        try:
            self._process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            await asyncio.sleep(2)

            if self._process.returncode is not None:
                stdout = await self._process.stdout.read() if self._process.stdout else b""
                self._state = RayState.ERROR
                self._error = f"Ray worker failed to join: {stdout.decode()[:500]}"
                logger.error(self._error)
            else:
                self._state = RayState.RUNNING
                logger.info(f"Ray worker joined cluster (pid {self._process.pid})")

        except Exception as e:
            self._state = RayState.ERROR
            self._error = str(e)
            logger.error(f"Failed to start Ray worker: {e}")

        return self.status

    async def stop(self) -> RayNodeStatus:
        """Stop Ray on this node (head or worker)."""
        if self._process and self._process.returncode is None:
            logger.info(f"Stopping Ray {self._role.value} (pid {self._process.pid})")
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=10)
            except TimeoutError:
                self._process.kill()
                await self._process.wait()

        # Also run `ray stop` to clean up any orphaned processes
        ray_bin = shutil.which("ray")
        if ray_bin:
            try:
                proc = await asyncio.create_subprocess_exec(
                    ray_bin, "stop", "--force",
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=10)
            except (TimeoutError, Exception):
                pass

        self._process = None
        self._role = RayRole.NONE
        self._state = RayState.STOPPED
        self._head_address = None
        self._error = None
        return self.status

    async def get_cluster_nodes(self) -> list[str]:
        """Query Ray for connected nodes (only works on head node)."""
        if self._role != RayRole.HEAD or self._state != RayState.RUNNING:
            return []

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://127.0.0.1:{RAY_DASHBOARD_PORT}/api/cluster_status",
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Extract node IPs from cluster status
                    nodes = []
                    active = (
                        data.get("result", {})
                        .get("autoscaler_report", {})
                        .get("active_nodes", {})
                    )
                    for node in active:
                        nodes.append(str(node))
                    return nodes
        except Exception:
            pass
        return []


async def coordinate_multi_node_start(
    head_node: str,
    worker_nodes: list[str],
    agent_port: int = 8100,
) -> bool:
    """Coordinate Ray cluster formation across multiple nodes.

    Called by the on-demand hook or head node agent when starting a multi-node model.
    1. Start Ray head on head_node
    2. Tell each worker node to join
    3. Wait for all nodes to be connected
    """
    async with httpx.AsyncClient(timeout=30) as client:
        # Start head
        logger.info(f"Starting Ray head on {head_node}")
        resp = await client.post(
            f"http://{head_node}:{agent_port}/ray/join",
            json={"role": "head"},
        )
        if resp.status_code != 200:
            logger.error(f"Failed to start Ray head on {head_node}: {resp.text}")
            return False

        head_data = resp.json()
        head_address = head_data.get("head_address")
        if not head_address:
            # Construct from node hostname
            head_address = f"{head_node}:{RAY_DEFAULT_PORT}"

        # Start workers
        for worker in worker_nodes:
            logger.info(f"Joining Ray worker {worker} to {head_address}")
            resp = await client.post(
                f"http://{worker}:{agent_port}/ray/join",
                json={"role": "worker", "head_address": head_address},
            )
            if resp.status_code != 200:
                logger.error(f"Failed to join worker {worker}: {resp.text}")
                return False

        logger.info("Ray cluster formation complete")
        return True


async def coordinate_multi_node_stop(
    nodes: list[str],
    agent_port: int = 8100,
) -> None:
    """Stop Ray on all nodes."""
    async with httpx.AsyncClient(timeout=15) as client:
        for node in nodes:
            try:
                await client.post(f"http://{node}:{agent_port}/ray/leave")
            except Exception:
                logger.warning(f"Failed to stop Ray on {node}")


def _find_ray() -> str:
    """Find the ray binary."""
    ray_bin = shutil.which("ray")
    if not ray_bin:
        raise RuntimeError("ray binary not found in PATH")
    return ray_bin
