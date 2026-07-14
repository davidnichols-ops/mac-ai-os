"""Configuration models and YAML loading.

Everything configurable lives here. Settings are layered:

    defaults (code)  <  configs/default.yaml  <  configs/user.yaml  <  env vars
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from app.config.hardware import HardwareProfile, detect


class ModelConfig(BaseModel):
    provider: str = "mlx"  # mlx | ollama
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    # Optional endpoint for remote/openai-compatible providers.
    base_url: str | None = None


class EmbeddingsConfig(BaseModel):
    provider: str = "chroma_default"  # chroma_default | mlx | ollama
    model: str = ""
    # Chroma collection name and persistence directory.
    collection: str = "mac-ai-os"
    persist_dir: str = "data/embeddings"


class RetrievalConfig(BaseModel):
    backend: str = "chroma"  # chroma | (future: lancedb, qdrant, pgvector)
    top_k: int = 5
    chunk_size: int = 800
    chunk_overlap: int = 120


class TelemetryConfig(BaseModel):
    enabled: bool = True
    interval_seconds: float = 5.0


class OllamaConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    auto_start: bool = True
    startup_timeout_seconds: float = 30.0


class LoggingConfig(BaseModel):
    level: str = "INFO"
    json_logs: bool = False
    log_dir: str = "logs"


class PathsConfig(BaseModel):
    data_dir: str = "data"
    models_dir: str = "models"
    logs_dir: str = "logs"


class Settings(BaseModel):
    planner: ModelConfig = Field(default_factory=ModelConfig)
    utility: ModelConfig = Field(default_factory=ModelConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    hardware: HardwareProfile | None = None

    def with_hardware(self, hw: HardwareProfile | None = None) -> Settings:
        """Fill in any empty model names using detected hardware defaults."""
        hw = hw or self.hardware or detect()
        updated = self.model_copy()
        updated.hardware = hw
        if not updated.planner.model:
            updated.planner.model = hw.recommended_planner()
        if not updated.utility.model:
            updated.utility.model = hw.recommended_utility()
        if not updated.embeddings.model:
            updated.embeddings.model = hw.recommended_embedding()
        return updated

    def resolved_data_dir(self) -> Path:
        """Resolve the data directory to an absolute path.

        If data_dir is relative and the source repo's data/ exists, use that
        (development mode). Otherwise use ~/.local/share/maos/ (installed mode).
        """
        p = Path(self.paths.data_dir)
        if p.is_absolute():
            return p
        # Check if we're running from the source repo.
        repo_data = Path(__file__).resolve().parents[2] / "data"
        if repo_data.exists() and (repo_data / "maos.db").exists():
            return repo_data
        # Installed mode — use XDG data directory.
        return Path.home() / ".local" / "share" / "maos"


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping")
    return data


def _find_config_dir() -> Path | None:
    """Find the configs directory.

    Checks in order:
    1. Next to the source repo (development)
    2. Bundled in the installed package
    3. ~/.config/maos/ (XDG user config)
    """
    # 1. Source repo — configs/ next to app/
    repo_dir = Path(__file__).resolve().parents[2] / "configs"
    if (repo_dir / "default.yaml").exists():
        return repo_dir

    # 2. Installed package — configs/ bundled alongside
    pkg_dir = Path(__file__).resolve().parent.parent / "configs"
    if (pkg_dir / "default.yaml").exists():
        return pkg_dir

    # 3. XDG user config
    xdg = Path.home() / ".config" / "maos" / "configs"
    if (xdg / "default.yaml").exists():
        return xdg

    return None


def load_settings(
    config_path: str | Path | None = None,
    *,
    extra_yaml: str | Path | None = None,
    detect_hardware: bool = True,
) -> Settings:
    """Load settings from YAML files, then apply hardware-derived defaults."""
    base_dir = _find_config_dir()
    defaults_yaml = _load_yaml(base_dir / "default.yaml") if base_dir else {}

    user_yaml: dict[str, Any] = {}
    if config_path:
        user_yaml = _load_yaml(Path(config_path))
    elif base_dir:
        user_yaml = _load_yaml(base_dir / "user.yaml")

    # Also check XDG user config for overrides (highest priority YAML).
    xdg_user = Path.home() / ".config" / "maos" / "configs" / "user.yaml"
    xdg_yaml = _load_yaml(xdg_user) if xdg_user.exists() else {}

    extra = _load_yaml(Path(extra_yaml)) if extra_yaml else {}

    merged: dict[str, Any] = {}
    merged.update(defaults_yaml)
    merged.update(user_yaml)
    merged.update(xdg_yaml)
    merged.update(extra)

    hw = detect() if detect_hardware else None
    settings = Settings.model_validate(merged)
    return settings.with_hardware(hw)
