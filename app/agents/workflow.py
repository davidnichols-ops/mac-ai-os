"""Provider-neutral startup and credential diagnostics for MAOS agents."""

from __future__ import annotations

import os
import shlex
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from dotenv import dotenv_values, load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
DEVICE_WORKFLOW = Path.home() / "AGENTS_WORKFLOW.md"

PROVIDERS: dict[str, dict[str, Any]] = {
    "generic": {
        "binary": None,
        "required_env": (),
        "auth": "provider-managed",
        "config": (),
    },
    "devin": {
        "binary": ("DEVIN_PATH", "devin"),
        "required_env": (),
        "auth": "Devin CLI managed authentication",
        "config": ("~/.config/devin/mcp_config.json", "~/.config/devin/AGENTS.md"),
    },
    "grokhack": {
        "binary": ("GROK_PATH", "grok"),
        "required_env": ("OPENROUTER_API_KEY",),
        "auth": "OPENROUTER_API_KEY from the protected MAOS environment",
        "config": ("~/.grok/config.toml", "~/.grok/rules/04-personal-ecosystem.md"),
    },
    "claude-code": {
        "binary": ("CLAUDE_PATH", "claude"),
        "required_env": (),
        "auth": "Claude CLI managed authentication",
        "config": ("~/.claude",),
    },
    "codex": {
        "binary": ("CODEX_PATH", "codex"),
        "required_env": (),
        "auth": "Codex application managed authentication",
        "config": ("~/.codex",),
    },
    "local": {
        "binary": ("UV_PATH", "uv"),
        "required_env": (),
        "auth": "none; local execution",
        "config": ("configs/user.yaml",),
    },
    "openrouter": {
        "binary": ("DEVIN_PATH", "devin"),
        "required_env": ("OPENROUTER_API_KEY",),
        "auth": "OPENROUTER_API_KEY from the protected MAOS environment",
        "config": ("configs/user.yaml",),
    },
}

INTEGRATION_ENV = (
    "SLACK_BOT_TOKEN",
    "SLACK_APP_TOKEN",
    "SLACK_USER_TOKEN",
    "SLACK_TEAM_ID",
)

OPTIONAL_ENV_SOURCES: dict[str, tuple[str, ...]] = {
    "TOGETHER_API_KEY": (".env",),
    "DEEPINFRA_API_KEY": (".env",),
    "FIRECRAWL_API_KEY": (".env",),
}


def _resolve_binary(
    spec: tuple[str, str] | None,
    environ: Mapping[str, str],
    which: Callable[[str], str | None],
) -> dict[str, Any]:
    if spec is None:
        return {"required": False, "configured": True, "path": None}
    env_name, default = spec
    requested = environ.get(env_name) or default
    path = requested if Path(requested).is_file() else which(requested)
    return {
        "required": True,
        "configured": bool(path),
        "path": str(path) if path else None,
        "override_env": env_name,
    }


def _dotenv_status(path: Path, key: str) -> bool:
    if not path.is_file():
        return False
    return bool(dotenv_values(path).get(key))


def _credential_status(
    key: str,
    *,
    environ: Mapping[str, str],
    sources: tuple[Path, ...],
) -> dict[str, Any]:
    if environ.get(key):
        return {"configured": True, "source": "process environment"}
    for source in sources:
        if _dotenv_status(source, key):
            return {"configured": True, "source": str(source)}
    return {"configured": False, "source": None}


def _file_status(path: Path, *, sensitive: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "secure": not sensitive}
    mode = path.stat().st_mode & 0o777
    secure = not sensitive or mode & 0o077 == 0
    return {
        "path": str(path),
        "exists": True,
        "mode": f"{mode:03o}",
        "secure": secure,
    }


def _config_status(path: str, *, root: Path, home: Path) -> dict[str, Any]:
    if path.startswith("~/"):
        resolved = home / path[2:]
    else:
        candidate = Path(path)
        resolved = candidate if candidate.is_absolute() else root / candidate
    return _file_status(resolved)


def doctor_report(
    provider: str = "all",
    *,
    root: Path = REPO_ROOT,
    home: Path | None = None,
    environ: Mapping[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> dict[str, Any]:
    """Return readiness without ever returning credential values."""
    if provider != "all" and provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")

    home = home or Path.home()
    repo_env = root / ".env"
    if root == REPO_ROOT:
        load_dotenv(repo_env, override=False)
    env = dict(os.environ if environ is None else environ)
    providers = PROVIDERS if provider == "all" else {provider: PROVIDERS[provider]}

    provider_checks: dict[str, Any] = {}
    required_keys = set(INTEGRATION_ENV)
    for name, spec in providers.items():
        binary = _resolve_binary(spec["binary"], env, which)
        required_env = list(spec["required_env"])
        required_keys.update(required_env)
        config_files = [
            _config_status(path, root=root, home=home) for path in spec["config"]
        ]
        provider_checks[name] = {
            "binary": binary,
            "required_env": required_env,
            "auth": spec["auth"],
            "config_files": config_files,
        }

    credentials = {
        key: _credential_status(key, environ=env, sources=(repo_env,))
        for key in sorted(required_keys)
    }
    optional_credentials = {
        key: _credential_status(
            key,
            environ=env,
            sources=tuple(
                (root / item).resolve() if not Path(item).is_absolute() else Path(item)
                for item in source_names
            ),
        )
        for key, source_names in OPTIONAL_ENV_SOURCES.items()
    }

    workflow_files = {
        "device": _file_status(home / "AGENTS_WORKFLOW.md"),
        "repository": _file_status(root / "AGENTS.md"),
        "agent_package": _file_status(root / "app" / "agents" / "AGENTS.md"),
        "environment": _file_status(repo_env, sensitive=True),
    }
    provider_ok = all(
        check["binary"]["configured"]
        and all(credentials[key]["configured"] for key in check["required_env"])
        and all(item["exists"] for item in check["config_files"])
        for check in provider_checks.values()
    )
    integration_ok = all(credentials[key]["configured"] for key in INTEGRATION_ENV)
    files_ok = all(item["exists"] and item["secure"] for item in workflow_files.values())
    return {
        "healthy": provider_ok and integration_ok and files_ok,
        "provider": provider,
        "providers": provider_checks,
        "credentials": credentials,
        "optional_credentials": optional_credentials,
        "workflow_files": workflow_files,
        "secret_policy": "values never leave protected environment files or process memory",
    }


def build_agent_brief(provider: str = "generic", task: str = "") -> str:
    """Build the authoritative provider-specific handoff instructions."""
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}")
    task_arg = shlex.quote(task or "<task>")
    return f"""# MAOS Agent Brief

Provider: {provider}
Task: {task or '<task not supplied>'}

## Required sequence

1. Read `{DEVICE_WORKFLOW}` and `{REPO_ROOT / 'AGENTS.md'}`.
2. Run `cd {REPO_ROOT} && uv run app agent doctor --provider {provider} --json`.
3. Register the run with
   `uv run app agent start --provider {provider} --role operator --task {task_arg}`.
4. Run `uv run app onboard --json`, then `uv run app refresh --json`.
5. Inspect `uv run app workspace active`, `uv run app blackboard show`, and
   `uv run app task list` before inventing new work.
6. Execute only requested or clearly implied work. Preserve unrelated dirty
   changes and use MAOS MCP as the context/action boundary when available.
7. Verify with direct evidence: focused tests, lint, live behavior, and diff
   inspection. A second agent's agreement is not verification.
8. Close the session with `uv run app agent end <agent-id> --summary <summary>
   --result <evidence> --workspace <workspace>`.

## Credentials

MAOS loads secrets from `{REPO_ROOT / '.env'}` and provider-managed stores.
Never print, copy into prompts, commit, or place raw values in session records.
Use `agent doctor` for names, sources, and readiness only. Optional training
API keys may be placed in the MAOS `.env` or a path of your choice.

## Slack handoff

Commands belong in `#maos-bridge`, operational handoffs in `#maos-ops`, and
daily summaries in `#daily-brief`. Slack is a console, not the source of truth.
"""
