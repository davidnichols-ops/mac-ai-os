"""Configuration models and YAML loading.

Everything configurable lives here. Settings are layered:

    defaults (code)  <  configs/default.yaml  <  configs/user.yaml  <  env vars
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from app.config.hardware import HardwareProfile, detect


class ModelConfig(BaseModel):
    provider: str = "mlx"  # mlx | ollama | api | rapid_mlx
    model: str = ""
    temperature: float = 0.7
    max_tokens: int = 1024
    # Optional endpoint for remote/openai-compatible providers.
    base_url: str | None = None
    api_key: str | None = None
    # Rapid-MLX / local server options.
    model_path: str | None = None
    binary: str | None = None
    auto_start: bool = True


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


class LanVaultConfig(BaseModel):
    """Durable LANBlockchain archive for MAOS-owned persistence."""

    enabled: bool = False
    binary: str = "lan-vault"
    archive_root: str = "personal"
    outbox_dir: str = "data/lan-vault-outbox"
    timeout_seconds: int = 120
    verify_after_push: bool = True
    reconcile_after_push: bool = False


class SlackConfig(BaseModel):
    """Slack integration configuration.

    Tokens are loaded from env vars (never hardcoded in YAML):
      SLACK_BOT_TOKEN  — xoxb-... (from installing the app to your workspace)
      SLACK_APP_TOKEN  — xapp-... (from enabling Socket Mode)
      SLACK_USER_TOKEN — xoxp-... (read-only history for the installing user)
      SLACK_TEAM_ID    — T... workspace ID (required for organization tokens)

    The channel and allowlist can be set in user.yaml or env vars.
    """
    enabled: bool = False
    bot_token: str = ""        # set via SLACK_BOT_TOKEN env var
    app_token: str = ""        # set via SLACK_APP_TOKEN env var
    user_token: str = ""       # set via SLACK_USER_TOKEN; read-only history sync
    team_id: str = ""          # target workspace for Enterprise Grid org tokens
    channel: str = "#maos"     # command channel and default response target
    alerts_channel: str = ""   # operational alerts; defaults to channel
    daily_channel: str = ""    # scheduled/on-demand digests; defaults to channel
    audit_channel: str = ""    # lifecycle and action audit; defaults to alerts
    allowlist: list[str] = []  # Slack user IDs allowed to send commands (empty = allow all workspace members)
    # Which event types trigger push alerts. Empty = all alertable events.
    alert_types: list[str] = []
    # Daily digest — set to HH:MM (24h UTC) to enable, empty to disable.
    digest_time: str = ""
    # Require explicit approval in Slack for HIGH/CRITICAL risk commands.
    require_approval: bool = True
    # Optional Q&A model for Slack `ask` — man-in-the-middle with RAG.
    # provider: 'api' = OpenAI-compatible (e.g. OpenRouter)
    qa_model: str = "nvidia/nemotron-3-ultra-550b-a55b:free"
    qa_provider: str = "api"
    qa_api_key: str = ""
    qa_base_url: str = "https://openrouter.ai/api/v1"
    # Incremental Slack history ingestion into a local SQLite cache + RAG.
    ingest_enabled: bool = True
    ingest_interval_seconds: int = 900
    ingest_days: int = 30
    ingest_max_messages: int = 10_000
    ingest_include_dms: bool = True
    ingest_channels: list[str] = []  # channel IDs or names; empty = all visible
    ingest_db: str = "data/slack.db"


class Settings(BaseModel):
    planner: ModelConfig = Field(default_factory=ModelConfig)
    planner_alt: ModelConfig = Field(default_factory=ModelConfig)
    utility: ModelConfig = Field(default_factory=ModelConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    telemetry: TelemetryConfig = Field(default_factory=TelemetryConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    lan_vault: LanVaultConfig = Field(default_factory=LanVaultConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
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
    # The repository-local .env is the documented home for Slack tokens and
    # channel overrides. Load it before applying environment precedence so
    # `maos slack status` and the bridge use the same configuration.
    repo_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(repo_env, override=False)
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

    # Env var overrides for secrets (never stored in YAML).
    import os
    lan_cfg = merged.setdefault("lan_vault", {})
    if isinstance(lan_cfg, dict):
        bool_env = os.environ.get("MAOS_LAN_VAULT_ENABLED")
        if bool_env is not None:
            lan_cfg["enabled"] = bool_env.strip().lower() in {"1", "true", "yes", "on"}
        for env_name, field_name in (
            ("MAOS_LAN_VAULT_BINARY", "binary"),
            ("MAOS_LAN_VAULT_ARCHIVE_ROOT", "archive_root"),
            ("MAOS_LAN_VAULT_OUTBOX_DIR", "outbox_dir"),
        ):
            value = os.environ.get(env_name, "").strip()
            if value:
                lan_cfg[field_name] = value
        timeout = os.environ.get("MAOS_LAN_VAULT_TIMEOUT_SECONDS", "").strip()
        if timeout:
            try:
                lan_cfg["timeout_seconds"] = int(timeout)
            except ValueError:
                pass

    slack_cfg = merged.setdefault("slack", {})
    if isinstance(slack_cfg, dict):
        bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
        app_token = os.environ.get("SLACK_APP_TOKEN", "")
        user_token = os.environ.get("SLACK_USER_TOKEN", "")
        team_id = os.environ.get("SLACK_TEAM_ID", "")
        if bot_token:
            slack_cfg["bot_token"] = bot_token
        if app_token:
            slack_cfg["app_token"] = app_token
        if user_token:
            slack_cfg["user_token"] = user_token
        if team_id:
            slack_cfg["team_id"] = team_id
        if bot_token and app_token:
            slack_cfg["enabled"] = True
        # Allow channel + allowlist overrides via env too.
        channel = os.environ.get("SLACK_CHANNEL", "")
        if channel:
            slack_cfg["channel"] = channel
        for env_name, field_name in (
            ("SLACK_ALERTS_CHANNEL", "alerts_channel"),
            ("SLACK_DAILY_CHANNEL", "daily_channel"),
            ("SLACK_AUDIT_CHANNEL", "audit_channel"),
        ):
            value = os.environ.get(env_name, "")
            if value:
                slack_cfg[field_name] = value
        allowlist = os.environ.get("SLACK_ALLOWLIST", "")
        if allowlist:
            slack_cfg["allowlist"] = [u.strip() for u in allowlist.split(",") if u.strip()]
        ingest_channels = os.environ.get("SLACK_INGEST_CHANNELS", "")
        if ingest_channels:
            slack_cfg["ingest_channels"] = [
                value.strip().lstrip("#")
                for value in ingest_channels.split(",")
                if value.strip()
            ]
        bool_env = {
            "SLACK_INGEST_ENABLED": "ingest_enabled",
            "SLACK_INGEST_INCLUDE_DMS": "ingest_include_dms",
        }
        for env_name, field_name in bool_env.items():
            raw = os.environ.get(env_name)
            if raw is not None:
                slack_cfg[field_name] = raw.strip().lower() in {"1", "true", "yes", "on"}
        int_env = {
            "SLACK_INGEST_INTERVAL_SECONDS": "ingest_interval_seconds",
            "SLACK_INGEST_DAYS": "ingest_days",
            "SLACK_INGEST_MAX_MESSAGES": "ingest_max_messages",
        }
        for env_name, field_name in int_env.items():
            raw = os.environ.get(env_name, "").strip()
            if raw:
                try:
                    slack_cfg[field_name] = int(raw)
                except ValueError:
                    pass
        ingest_db = os.environ.get("SLACK_INGEST_DB", "").strip()
        if ingest_db:
            slack_cfg["ingest_db"] = ingest_db
        # Resolve the Q&A model API key from the environment.
        # OPENROUTER_API_KEY is the canonical name; OPENAI_API_KEY is a
        # fallback for users who reuse the same key.  This ensures the
        # Slack bridge can build the qa_model even when the process was
        # started without sourcing .env (e.g. via launchd or a bare
        # `uv run app slack start`).
        qa_key = (
            os.environ.get("OPENROUTER_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if qa_key:
            slack_cfg["qa_api_key"] = qa_key

    hw = detect() if detect_hardware else None
    settings = Settings.model_validate(merged)
    return settings.with_hardware(hw)
