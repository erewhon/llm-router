"""GPU detection and VRAM tracking."""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from llm_router.config import GpuType

logger = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    gpu_type: GpuType
    total_vram_mb: int
    free_vram_mb: int
    unified_memory: bool
    gpu_busy_pct: int | None = None  # 0-100, None if unavailable

    @property
    def total_vram_gb(self) -> float:
        return self.total_vram_mb / 1024

    @property
    def free_vram_gb(self) -> float:
        return self.free_vram_mb / 1024


def detect_gpu_type() -> GpuType:
    """Detect whether the system has AMD or NVIDIA GPU."""
    if Path("/dev/kfd").exists():
        return GpuType.AMD
    if Path("/dev/nvidia0").exists():
        return GpuType.NVIDIA
    # Fallback: check for nvidia-smi or rocm-smi
    if shutil.which("nvidia-smi"):
        return GpuType.NVIDIA
    if shutil.which("rocm-smi"):
        return GpuType.AMD
    raise RuntimeError("No GPU detected (no /dev/kfd or /dev/nvidia0)")


def _get_nvidia_vram() -> tuple[int, int, bool]:
    """Query NVIDIA GPU memory. Returns (free_mb, total_mb, unified)."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            free_mb = int(parts[0].strip())
            total_mb = int(parts[1].strip())
            return free_mb, total_mb, False
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    # nvidia-smi reports [N/A] on unified memory GPUs (e.g. GB10)
    # Fall back to PyTorch
    try:
        result = subprocess.run(
            [
                "python3",
                "-c",
                "import torch; f,t=torch.cuda.mem_get_info(); "
                "print(f//1048576, t//1048576)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            return int(parts[0]), int(parts[1]), True
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    # Fallback: unified memory GPU — use system RAM from /proc/meminfo
    return _get_meminfo_vram()


def _get_meminfo_vram() -> tuple[int, int, bool]:
    """Read system memory from /proc/meminfo (unified memory fallback)."""
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        raise RuntimeError("Could not determine NVIDIA GPU memory")
    text = meminfo.read_text()
    total_kb = free_kb = 0
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            total_kb = int(line.split()[1])
        elif line.startswith("MemAvailable:"):
            free_kb = int(line.split()[1])
    if total_kb == 0:
        raise RuntimeError("Could not parse /proc/meminfo")
    return free_kb // 1024, total_kb // 1024, True


def _get_amd_vram() -> tuple[int, int, bool]:
    """Query AMD GPU memory. Returns (free_mb, total_mb, unified).

    Strix Halo uses unified memory, so we always report unified=True for AMD.
    Tries sysfs first (no dependencies), then PyTorch, then rocm-smi.
    """
    # Primary: sysfs (works on all amdgpu-driven GPUs, no deps)
    try:
        drm = Path("/sys/class/drm")
        for card in sorted(drm.glob("card[0-9]*/device/mem_info_vram_total")):
            device = card.parent
            total_b = int((device / "mem_info_vram_total").read_text().strip())
            used_b = int((device / "mem_info_vram_used").read_text().strip())
            total_mb = total_b // (1024 * 1024)
            free_mb = (total_b - used_b) // (1024 * 1024)
            return free_mb, total_mb, True
    except (OSError, ValueError):
        pass

    # Fallback: PyTorch
    try:
        result = subprocess.run(
            [
                "python3",
                "-c",
                "import torch; f,t=torch.cuda.mem_get_info(); "
                "print(f//1048576, t//1048576)",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            return int(parts[0]), int(parts[1]), True
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    # Fallback: rocm-smi
    try:
        result = subprocess.run(
            ["rocm-smi", "--showmeminfo", "vram", "--json"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            import json

            data = json.loads(result.stdout)
            # rocm-smi JSON format varies; handle common structures
            for _gpu_id, info in data.items():
                if isinstance(info, dict):
                    total = int(info.get("VRAM Total Memory (B)", 0)) // (1024 * 1024)
                    used = int(info.get("VRAM Total Used Memory (B)", 0)) // (1024 * 1024)
                    return total - used, total, True
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    raise RuntimeError("Could not determine AMD GPU memory")


def _get_gpu_busy_pct() -> int | None:
    """Read GPU compute utilization from sysfs (amdgpu) or nvidia-smi."""
    # sysfs (amdgpu)
    try:
        for p in sorted(Path("/sys/class/drm").glob(
            "card[0-9]*/device/gpu_busy_percent"
        )):
            return int(p.read_text().strip())
    except (OSError, ValueError):
        pass

    # nvidia-smi
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            val = result.stdout.strip().split("\n")[0].strip()
            if val != "[N/A]":
                return int(val)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass

    return None


def get_gpu_info() -> GpuInfo:
    """Detect GPU type and query current memory status."""
    gpu_type = detect_gpu_type()
    if gpu_type == GpuType.NVIDIA:
        free_mb, total_mb, unified = _get_nvidia_vram()
    else:
        free_mb, total_mb, unified = _get_amd_vram()

    busy = _get_gpu_busy_pct()

    info = GpuInfo(
        gpu_type=gpu_type,
        total_vram_mb=total_mb,
        free_vram_mb=free_mb,
        unified_memory=unified,
        gpu_busy_pct=busy,
    )
    logger.info(
        f"GPU: {gpu_type.value}, {info.free_vram_gb:.1f}GB free / "
        f"{info.total_vram_gb:.1f}GB total (unified={unified}), "
        f"busy={busy}%"
    )
    return info


def compute_gpu_memory_utilization(
    gpu_info: GpuInfo,
    requested_vram_gb: int,
) -> float:
    """Compute vLLM --gpu-memory-utilization fraction.

    If requested_vram_gb is specified, compute the fraction of total VRAM.
    Otherwise, use 80% of free memory, capped at 90% of free.
    """
    total_mb = gpu_info.total_vram_mb
    free_mb = gpu_info.free_vram_mb

    if requested_vram_gb > 0:
        requested_mb = requested_vram_gb * 1024
        util = requested_mb / total_mb
    else:
        util = (free_mb * 0.8) / total_mb

    # Clamp to [0.10, 0.95]
    util = max(0.10, min(0.95, util))

    # Cap at 90% of free memory to leave headroom
    max_util = (free_mb * 0.9) / total_mb
    if util > max_util:
        util = max(0.10, max_util)

    return round(util, 2)
