"""llama.cpp backend — start/stop llama-server for GGUF models."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
from pathlib import Path

import httpx

from llm_router.config import GpuType, ModelDefinition
from llm_router.node_agent.backends.base import Backend
from llm_router.node_agent.gpu import detect_gpu_type
from llm_router.node_agent.models import ModelState, ProcessStatus

logger = logging.getLogger(__name__)

LLAMACPP_PORT = 5391
_state_dir = Path(os.environ.get("STATE_DIRECTORY", "/tmp/llm-router"))
PID_DIR = _state_dir / "pids"
LOG_DIR = _state_dir / "logs"


class LlamaCppBackend(Backend):
    """Manages llama-server (llama.cpp) processes for GGUF models."""

    def __init__(self, port: int = LLAMACPP_PORT) -> None:
        self._port = port
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._states: dict[str, ModelState] = {}
        self._errors: dict[str, str] = {}
        self._log_handles: dict[str, object] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        PID_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    async def start(self, model_id: str, model: ModelDefinition) -> None:
        """Start llama-server for a GGUF model."""
        if model_id in self._processes:
            proc = self._processes[model_id]
            if proc.returncode is None:
                logger.info(f"Model {model_id} already running (pid {proc.pid})")
                return

        if not model.gguf_file:
            raise ValueError(f"Model {model_id} has no gguf_file specified")

        self._states[model_id] = ModelState.STARTING
        self._errors.pop(model_id, None)

        cmd = self._build_command(model_id, model)
        log_file = LOG_DIR / f"{model_id}.log"
        logger.info(f"Starting llama-server for {model_id}: {' '.join(cmd)}")

        try:
            log_fh = log_file.open("w")
            self._log_handles[model_id] = log_fh
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=log_fh,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self._processes[model_id] = proc

            pid_file = PID_DIR / f"{model_id}.pid"
            pid_file.write_text(str(proc.pid))

            task = asyncio.create_task(self._wait_for_ready(model_id, proc))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        except Exception as e:
            self._states[model_id] = ModelState.ERROR
            self._errors[model_id] = str(e)
            logger.error(f"Failed to start llama-server for {model_id}: {e}")
            raise

    async def _wait_for_ready(
        self, model_id: str, proc: asyncio.subprocess.Process
    ) -> None:
        """Poll llama-server health until ready or failed."""
        for _attempt in range(90):  # up to ~90 seconds
            if proc.returncode is not None:
                self._states[model_id] = ModelState.ERROR
                self._errors[model_id] = f"Process exited with code {proc.returncode}"
                logger.error(f"llama-server for {model_id} exited early: code {proc.returncode}")
                return

            if await self.health_check(model_id):
                self._states[model_id] = ModelState.RUNNING
                logger.info(f"llama-server for {model_id} is ready (pid {proc.pid})")
                return

            await asyncio.sleep(1)

        self._states[model_id] = ModelState.ERROR
        self._errors[model_id] = "Timed out waiting for llama-server to become ready"
        logger.error(f"llama-server for {model_id} timed out")

    async def stop(self, model_id: str) -> None:
        """Stop llama-server for a model."""
        proc = self._processes.get(model_id)
        if proc and proc.returncode is None:
            logger.info(f"Stopping llama-server for {model_id} (pid {proc.pid})")
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=10)
            except TimeoutError:
                logger.warning(f"Force-killing llama-server for {model_id}")
                proc.kill()
                await proc.wait()

        self._processes.pop(model_id, None)
        self._states[model_id] = ModelState.STOPPED
        self._errors.pop(model_id, None)

        log_fh = self._log_handles.pop(model_id, None)
        if log_fh and hasattr(log_fh, "close"):
            log_fh.close()

        pid_file = PID_DIR / f"{model_id}.pid"
        pid_file.unlink(missing_ok=True)

    async def status(self, model_id: str) -> ProcessStatus:
        """Get current status of a model."""
        state = self._states.get(model_id, ModelState.STOPPED)
        proc = self._processes.get(model_id)

        if proc and proc.returncode is not None and state == ModelState.RUNNING:
            state = ModelState.ERROR
            self._states[model_id] = state
            self._errors[model_id] = f"Process exited with code {proc.returncode}"

        return ProcessStatus(
            model_id=model_id,
            state=state,
            pid=proc.pid if proc and proc.returncode is None else None,
            port=self._port if state == ModelState.RUNNING else None,
            error=self._errors.get(model_id),
        )

    async def health_check(self, model_id: str) -> bool:
        """Check if llama-server is responding."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://localhost:{self._port}/health",
                    timeout=5,
                )
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            return False

    def _build_command(self, model_id: str, model: ModelDefinition) -> list[str]:
        """Build the llama-server startup command."""
        server_bin = _find_llama_server()

        cmd = [
            server_bin,
            "--model",
            model.gguf_file or "",
            "--host",
            "0.0.0.0",
            "--port",
            str(self._port),
        ]

        # GPU layers — offload all layers by default
        cmd.extend(["--n-gpu-layers", "999"])

        # Context size
        if model.vllm_args.max_model_len:
            cmd.extend(["--ctx-size", str(model.vllm_args.max_model_len)])

        # GPU-specific flags
        try:
            gpu_type = detect_gpu_type()
            if gpu_type == GpuType.AMD:
                # ROCm-specific: ensure HIP is used
                pass  # llama.cpp auto-detects ROCm when compiled with GGML_HIP
        except RuntimeError:
            pass

        # Extra args from model config
        cmd.extend(model.vllm_args.extra_args)

        return cmd


def _find_llama_server() -> str:
    """Find the llama-server binary."""
    for name in ("llama-server", "llama-server-rocm", "server"):
        path = shutil.which(name)
        if path:
            return path
    raise RuntimeError(
        "llama-server binary not found in PATH. "
        "Install llama.cpp or set PATH to include the binary location."
    )
