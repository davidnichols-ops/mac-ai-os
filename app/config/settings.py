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
    provider: str = "mlx"  # mlx | ollama | chroma_default
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


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping")
    return data


def load_settings(
    config_path: str | Path | None = None,
    *,
    extra_yaml: str | Path | None = None,
    detect_hardware: bool = True,
) -> Settings:
    """Load settings from YAML files, then apply hardware-derived defaults."""
    base_dir = Path(__file__).resolve().parents[2] / "configs"
    defaults_yaml = _load_yaml(base_dir / "default.yaml")

    user_yaml: dict[str, Any] = {}
    if config_path:
        user_yaml = _load_yaml(Path(config_path))
    else:
        user_yaml = _load_yaml(base_dir / "user.yaml")

    extra = _load_yaml(Path(extra_yaml)) if extra_yaml else {}

    merged: dict[str, Any] = {}
    merged.update(defaults_yaml)
    merged.update(user_yaml)
    merged.update(extra)

    hw = detect() if detect_hardware else None
    settings = Settings.model_validate(merged)
    return settings.with_hardware(hw)
