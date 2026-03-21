"""vLLM backend — manage vLLM via systemd template units (vllm@.service)."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from pathlib import Path

import httpx

from llm_router.config import ModelDefinition
from llm_router.node_agent.backends.base import Backend
from llm_router.node_agent.gpu import compute_gpu_memory_utilization, get_gpu_info
from llm_router.node_agent.models import ModelState, ProcessStatus

logger = logging.getLogger(__name__)

VLLM_PORT = 5391
_state_dir = Path(os.environ.get("STATE_DIRECTORY", "/tmp/llm-router"))
ENV_DIR = Path(os.environ.get("VLLM_ENV_DIR", "/etc/llm-router/vllm-env"))

# Map model name patterns to vLLM tool-call-parser names
TOOL_PARSER_MAP: list[tuple[str, str]] = [
    ("nemotron", "qwen3_coder"),
    ("qwen3-coder", "qwen3_coder"),
    ("qwen3", "qwen3_xml"),
    ("qwen2", "hermes"),
    ("llama-3", "llama3_json"),
    ("llama3", "llama3_json"),
    ("llama-4", "llama3_json"),
    ("llama4", "llama3_json"),
    ("mistral", "mistral"),
    ("mixtral", "mistral"),
    ("hermes", "hermes"),
    ("deepseek", "deepseek_v3"),
    ("jamba", "jamba"),
]


def _auto_detect_tool_parser(model_name: str) -> str:
    """Auto-detect vLLM tool-call-parser from model name."""
    lower = model_name.lower()
    for pattern, parser in TOOL_PARSER_MAP:
        if pattern in lower:
            return parser
    return "hermes"  # fallback


def _sanitize_unit_id(model_id: str) -> str:
    """Sanitize a model ID for use as a systemd unit instance name.

    Replaces / with - and removes other unsafe characters.
    """
    return re.sub(r"[^a-zA-Z0-9._-]", "-", model_id)


class VllmBackend(Backend):
    """Manages vLLM inference via systemd template units (vllm@.service).

    Instead of spawning subprocesses directly, this backend:
    1. Writes vLLM args to an env file in ENV_DIR
    2. Uses `sudo systemctl start/stop vllm@<instance>` to manage the service
    3. Queries systemd for process state (survives agent restarts)
    """

    def __init__(self) -> None:
        self._ports: dict[str, int] = {}  # model_id -> port
        self._models: dict[str, ModelDefinition] = {}  # model_id -> definition
        self._states: dict[str, ModelState] = {}
        self._errors: dict[str, str] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()

    @staticmethod
    def _get_port(model: ModelDefinition) -> int:
        """Return the port for a model (api_port override or default)."""
        return model.api_port or VLLM_PORT

    async def start(self, model_id: str, model: ModelDefinition) -> None:
        """Start vLLM for a model via systemd."""
        unit_id = _sanitize_unit_id(model_id)
        port = self._get_port(model)
        self._ports[model_id] = port
        self._models[model_id] = model

        # Check if already running via systemd
        if await self._is_active(unit_id):
            logger.info(f"Model {model_id} already running (unit vllm@{unit_id})")
            self._states[model_id] = ModelState.RUNNING
            return

        self._states[model_id] = ModelState.STARTING
        self._errors.pop(model_id, None)

        # Write args file (plain text, read by bash $(cat ...) in vllm@.service)
        vllm_args = self._build_vllm_args(model_id, model, port)
        args_file = ENV_DIR / f"{unit_id}.args"
        logger.info(f"Writing vLLM args for {model_id}: {args_file}")

        try:
            args_file.write_text(vllm_args + "\n")
            args_file.chmod(0o644)  # readable by llm-vllm user
        except OSError as e:
            self._states[model_id] = ModelState.ERROR
            self._errors[model_id] = f"Failed to write env file: {e}"
            logger.error(f"Failed to write env file for {model_id}: {e}")
            raise

        # Start the systemd unit
        logger.info(f"Starting vllm@{unit_id}")
        try:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "systemctl", "start", f"vllm@{unit_id}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode != 0:
                err_msg = stderr.decode().strip() or f"systemctl start exited {proc.returncode}"
                self._states[model_id] = ModelState.ERROR
                self._errors[model_id] = err_msg
                logger.error(f"Failed to start vllm@{unit_id}: {err_msg}")
                raise RuntimeError(err_msg)
        except TimeoutError:
            self._states[model_id] = ModelState.ERROR
            self._errors[model_id] = "Timed out starting systemd unit"
            logger.error(f"Timed out starting vllm@{unit_id}")
            raise

        # Wait for readiness in background
        task = asyncio.create_task(self._wait_for_ready(model_id, unit_id))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _wait_for_ready(self, model_id: str, unit_id: str) -> None:
        """Poll vLLM health until ready or failed."""
        for _attempt in range(120):  # up to ~2 minutes
            # Check if systemd unit died
            if not await self._is_active(unit_id):
                self._states[model_id] = ModelState.ERROR
                self._errors[model_id] = "systemd unit stopped unexpectedly"
                logger.error(f"vllm@{unit_id} stopped unexpectedly")
                return

            if await self.health_check(model_id):
                self._states[model_id] = ModelState.RUNNING
                pid = await self._get_main_pid(unit_id)
                logger.info(f"vLLM for {model_id} is ready (unit vllm@{unit_id}, pid {pid})")
                return

            await asyncio.sleep(1)

        self._states[model_id] = ModelState.ERROR
        self._errors[model_id] = "Timed out waiting for vLLM to become ready"
        logger.error(f"vLLM for {model_id} timed out")

    async def stop(self, model_id: str) -> None:
        """Stop vLLM for a model via systemd."""
        unit_id = _sanitize_unit_id(model_id)
        logger.info(f"Stopping vllm@{unit_id}")

        proc = await asyncio.create_subprocess_exec(
            "sudo", "systemctl", "stop", f"vllm@{unit_id}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            await asyncio.wait_for(proc.communicate(), timeout=45)
        except TimeoutError:
            logger.warning(f"Timed out stopping vllm@{unit_id}")

        self._states[model_id] = ModelState.STOPPED
        self._errors.pop(model_id, None)

    async def status(self, model_id: str, model: ModelDefinition | None = None) -> ProcessStatus:
        """Get current status of a model from systemd."""
        if model is not None:
            self._models[model_id] = model
        unit_id = _sanitize_unit_id(model_id)
        state = self._states.get(model_id, ModelState.STOPPED)

        # If we think it's running or starting, verify with systemd
        is_starting_or_running = state in (ModelState.RUNNING, ModelState.STARTING)
        if is_starting_or_running and not await self._is_active(unit_id):
            state = ModelState.ERROR
            self._states[model_id] = state
            self._errors[model_id] = "systemd unit is not active"

        # If we think it's stopped, check if systemd knows about a running instance
        # (handles agent restart scenario)
        if state == ModelState.STOPPED and await self._is_active(unit_id):
            # Rediscover running model after agent restart
            if await self.health_check(model_id):
                state = ModelState.RUNNING
                self._states[model_id] = state
                logger.info(f"Rediscovered running vllm@{unit_id}")
            else:
                state = ModelState.STARTING
                self._states[model_id] = state

        port = self._ports.get(model_id, VLLM_PORT)

        # For models not managed by systemd (e.g. multi-node Docker),
        # fall back to API health check — verify the correct model is serving
        if state in (ModelState.STOPPED, ModelState.ERROR):
            if await self.health_check(model_id, model=self._models.get(model_id)):
                state = ModelState.RUNNING
                self._states[model_id] = state
                self._errors.pop(model_id, None)
                logger.info(f"Discovered running vLLM on port {port} for {model_id}")

        pid = await self._get_main_pid(unit_id) if state == ModelState.RUNNING else None

        return ProcessStatus(
            model_id=model_id,
            state=state,
            pid=pid,
            port=port if state == ModelState.RUNNING else None,
            error=self._errors.get(model_id),
        )

    async def health_check(self, model_id: str, model: ModelDefinition | None = None) -> bool:
        """Check if vLLM is responding and serving the expected model."""
        port = self._ports.get(model_id, VLLM_PORT)
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://localhost:{port}/v1/models",
                    timeout=5,
                )
                if resp.status_code != 200:
                    return False
                if model is None:
                    return True
                # Verify the served model matches the expected hf_repo
                data = resp.json()
                served_ids = {m.get("id", "") for m in data.get("data", [])}
                return model.hf_repo in served_ids
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            return False

    async def _is_active(self, unit_id: str) -> bool:
        """Check if a systemd unit is active."""
        proc = await asyncio.create_subprocess_exec(
            "sudo", "systemctl", "is-active", f"vllm@{unit_id}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        return stdout.decode().strip() == "active"

    async def _get_main_pid(self, unit_id: str) -> int | None:
        """Get the MainPID of a systemd unit."""
        proc = await asyncio.create_subprocess_exec(
            "sudo", "systemctl", "show", f"vllm@{unit_id}",
            "--property=MainPID",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        # Output: "MainPID=12345\n"
        output = stdout.decode().strip()
        if output.startswith("MainPID="):
            try:
                pid = int(output.split("=", 1)[1])
                return pid if pid > 0 else None
            except ValueError:
                return None
        return None

    def _build_vllm_args(self, model_id: str, model: ModelDefinition, port: int) -> str:
        """Build vLLM command-line args as a single string for the env file."""
        args: list[str] = [
            "--model", model.hf_repo,
            "--host", "0.0.0.0",
            "--port", str(port),
        ]

        # GPU memory utilization
        gpu_util = model.vllm_args.gpu_memory_utilization
        if gpu_util is None:
            try:
                gpu_info = get_gpu_info()
                gpu_util = compute_gpu_memory_utilization(gpu_info, model.vram_gb)
            except RuntimeError:
                gpu_util = 0.80  # safe default
        args.extend(["--gpu-memory-utilization", str(gpu_util)])

        # Tool calling
        parser = model.vllm_args.tool_call_parser
        if parser is None:
            parser = _auto_detect_tool_parser(model.hf_repo)
        args.extend([
            "--enable-auto-tool-choice",
            "--tool-call-parser", parser,
        ])

        # Max model len
        if model.vllm_args.max_model_len:
            args.extend(["--max-model-len", str(model.vllm_args.max_model_len)])

        # Multi-node / tensor parallelism
        if model.multi_node:
            tp = model.multi_node.tensor_parallel_size
            args.extend([
                "--tensor-parallel-size", str(tp),
                "--distributed-executor-backend", "ray",
            ])

        # Extra args (values must not contain spaces — use compact JSON)
        args.extend(model.vllm_args.extra_args)

        return " ".join(args)
