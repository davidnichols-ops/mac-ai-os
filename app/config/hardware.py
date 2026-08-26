"""Hardware autodetection for sensible MLX/model defaults on Apple Silicon."""

from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class HardwareProfile:
    """A snapshot of the host machine relevant to model sizing."""

    macos_version: str
    arch: str
    model: str
    cpu_cores: int
    memory_gb: int
    has_gpu: bool
    has_neural_engine: bool

    @property
    def is_apple_silicon(self) -> bool:
        return self.arch == "arm64"

    @property
    def can_run_mlx(self) -> bool:
        return self.is_apple_silicon

    def recommended_planner(self) -> str:
        """Pick a planner model that fits comfortably in RAM."""
        if not self.can_run_mlx:
            return "qwen3.5:4b"  # ollama fallback
        # Qwen3.5-4B-OptiQ-4bit — mixed-precision sensitivity-aware quant.
        # 2.8GB on disk, Capability Score 68.76, sweet spot for laptops.
        return "mlx-community/Qwen3.5-4B-OptiQ-4bit"

    def recommended_utility(self) -> str:
        if self.can_run_mlx:
            return "mlx-community/gemma-3-4b-it-4bit"
        return "gemma3:4b"

    def recommended_embedding(self) -> str:
        # mlx-lm has no embed() API; the reliable local default is chroma's
        # built-in ONNX MiniLM. ollama/nomic-embed-text is the alternative.
        return "all-MiniLM-L6-v2-onnx"


def _sysctl(name: str) -> str:
    try:
        out = subprocess.check_output(["sysctl", "-n", name], text=True, timeout=2)
        return out.strip()
    except Exception:
        return ""


def _detect_neural_engine() -> bool:
    # The Neural Engine is present on all Apple Silicon; no clean public sysctl
    # exposes it, so infer from arch.
    return platform.machine() == "arm64"


def detect() -> HardwareProfile:
    macos = platform.mac_ver()[0]
    arch = platform.machine()
    model = _sysctl("hw.model") or "unknown"
    try:
        cpu_cores = int(_sysctl("hw.ncpu"))
    except ValueError:
        cpu_cores = 0
    try:
        mem_bytes = int(_sysctl("hw.memsize"))
        memory_gb = round(mem_bytes / (1024**3))
    except ValueError:
        memory_gb = 0
    has_gpu = arch == "arm64"
    return HardwareProfile(
        macos_version=macos,
        arch=arch,
        model=model,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        has_gpu=has_gpu,
        has_neural_engine=_detect_neural_engine(),
    )


def parse_model_name(model: str) -> str:
    """Strip a registry path down to a short display name."""
    return re.sub(r"^mlx-community/", "", model)
