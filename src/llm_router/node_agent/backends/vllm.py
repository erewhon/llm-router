"""vLLM backend — start/stop vLLM via subprocess."""

from __future__ import annotations

import asyncio
import logging
import signal
from pathlib import Path

import httpx

from llm_router.config import ModelDefinition
from llm_router.node_agent.backends.base import Backend
from llm_router.node_agent.gpu import compute_gpu_memory_utilization, get_gpu_info
from llm_router.node_agent.models import ModelState, ProcessStatus

logger = logging.getLogger(__name__)

VLLM_PORT = 5391
PID_DIR = Path("/tmp/llm-router/pids")
LOG_DIR = Path("/tmp/llm-router/logs")

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


class VllmBackend(Backend):
    """Manages vLLM inference server processes."""

    def __init__(self, vllm_port: int = VLLM_PORT) -> None:
        self._port = vllm_port
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._states: dict[str, ModelState] = {}
        self._errors: dict[str, str] = {}
        self._log_handles: dict[str, object] = {}
        self._background_tasks: set[asyncio.Task[None]] = set()
        PID_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    async def start(self, model_id: str, model: ModelDefinition) -> None:
        """Start vLLM for a model."""
        if model_id in self._processes:
            proc = self._processes[model_id]
            if proc.returncode is None:
                logger.info(f"Model {model_id} already running (pid {proc.pid})")
                return

        self._states[model_id] = ModelState.STARTING
        self._errors.pop(model_id, None)

        cmd = self._build_command(model_id, model)
        log_file = LOG_DIR / f"{model_id}.log"
        logger.info(f"Starting vLLM for {model_id}: {' '.join(cmd)}")

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

            # Write PID file
            pid_file = PID_DIR / f"{model_id}.pid"
            pid_file.write_text(str(proc.pid))

            # Wait for readiness in background
            task = asyncio.create_task(self._wait_for_ready(model_id, proc))
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        except Exception as e:
            self._states[model_id] = ModelState.ERROR
            self._errors[model_id] = str(e)
            logger.error(f"Failed to start vLLM for {model_id}: {e}")
            raise

    async def _wait_for_ready(
        self, model_id: str, proc: asyncio.subprocess.Process
    ) -> None:
        """Poll vLLM health until ready or failed."""
        for _attempt in range(120):  # up to ~2 minutes
            if proc.returncode is not None:
                self._states[model_id] = ModelState.ERROR
                self._errors[model_id] = f"Process exited with code {proc.returncode}"
                logger.error(f"vLLM for {model_id} exited early: code {proc.returncode}")
                return

            if await self.health_check(model_id):
                self._states[model_id] = ModelState.RUNNING
                logger.info(f"vLLM for {model_id} is ready (pid {proc.pid})")
                return

            await asyncio.sleep(1)

        self._states[model_id] = ModelState.ERROR
        self._errors[model_id] = "Timed out waiting for vLLM to become ready"
        logger.error(f"vLLM for {model_id} timed out")

    async def stop(self, model_id: str) -> None:
        """Stop vLLM for a model."""
        proc = self._processes.get(model_id)
        if proc and proc.returncode is None:
            logger.info(f"Stopping vLLM for {model_id} (pid {proc.pid})")
            proc.send_signal(signal.SIGTERM)
            try:
                await asyncio.wait_for(proc.wait(), timeout=15)
            except TimeoutError:
                logger.warning(f"Force-killing vLLM for {model_id}")
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

        # Check if process died unexpectedly
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
        """Check if vLLM is responding to requests."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    f"http://localhost:{self._port}/v1/models",
                    timeout=5,
                )
                return resp.status_code == 200
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout):
            return False

    def _build_command(self, model_id: str, model: ModelDefinition) -> list[str]:
        """Build the vLLM startup command."""
        cmd = [
            "python3",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model.hf_repo,
            "--host",
            "0.0.0.0",
            "--port",
            str(self._port),
        ]

        # GPU memory utilization
        gpu_util = model.vllm_args.gpu_memory_utilization
        if gpu_util is None:
            try:
                gpu_info = get_gpu_info()
                gpu_util = compute_gpu_memory_utilization(gpu_info, model.vram_gb)
            except RuntimeError:
                gpu_util = 0.80  # safe default
        cmd.extend(["--gpu-memory-utilization", str(gpu_util)])

        # Tool calling
        parser = model.vllm_args.tool_call_parser
        if parser is None:
            parser = _auto_detect_tool_parser(model.hf_repo)
        cmd.extend([
            "--enable-auto-tool-choice",
            "--tool-call-parser",
            parser,
        ])

        # Max model len
        if model.vllm_args.max_model_len:
            cmd.extend(["--max-model-len", str(model.vllm_args.max_model_len)])

        # Extra args
        cmd.extend(model.vllm_args.extra_args)

        return cmd
