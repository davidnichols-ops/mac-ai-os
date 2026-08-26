"""Read a Trajectory OS context pack without coupling MAOS to its internals.

Trajectory OS remains the owner of the personal knowledge graph. MAOS only
consumes its provider-neutral Markdown context command, so either project can
evolve independently and no personal database is imported or mutated here.

Observability:
    Every failure path emits a structured log event so context-ingestion
    gaps are debuggable instead of silent. Use ``status()`` for a
    configuration/availability snapshot suitable for diagnostics.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any

from app.log import get_logger

log = get_logger(__name__)


class TrajectoryContextProvider:
    """Best-effort provider for a local Trajectory OS context pack."""

    def __init__(
        self,
        *,
        command: str = "trajectory",
        db_path: str | None = None,
        timeout_s: float = 10.0,
    ) -> None:
        self.command = command
        self.db_path = db_path
        self.timeout_s = timeout_s

    def get(self, purpose: str = "daily") -> str | None:
        """Return a Markdown context pack, or ``None`` when unavailable.

        Every failure path is logged so context-ingestion gaps are
        observable rather than silent. The ``None`` return contract is
        preserved — callers (e.g. the agent loop) treat absence as
        "no Trajectory context available" and continue normally.
        """
        args = [self.command, "context", purpose, "--format", "markdown"]
        if self.db_path:
            args.extend(["--db-path", self.db_path])
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                check=False,
                env=os.environ.copy(),
            )
        except FileNotFoundError:
            log.debug(
                "trajectory.context.unavailable",
                reason="command_not_found",
                command=self.command,
                purpose=purpose,
            )
            return None
        except subprocess.TimeoutExpired:
            log.warning(
                "trajectory.context.timeout",
                command=self.command,
                purpose=purpose,
                timeout_s=self.timeout_s,
            )
            return None
        except OSError as e:
            log.warning(
                "trajectory.context.os_error",
                command=self.command,
                purpose=purpose,
                error=str(e),
            )
            return None

        if result.returncode != 0:
            log.warning(
                "trajectory.context.nonzero_exit",
                command=self.command,
                purpose=purpose,
                returncode=result.returncode,
                stderr=result.stderr.strip()[:500] if result.stderr else "",
            )
            return None
        if not result.stdout.strip():
            log.debug(
                "trajectory.context.empty_output",
                command=self.command,
                purpose=purpose,
            )
            return None
        return result.stdout.strip()

    def status(self) -> dict[str, Any]:
        """Return a configuration/availability snapshot for diagnostics.

        This does NOT invoke the external command. It reports what the
        provider is configured to do and whether the command appears to
        be on ``PATH``. Safe to call at any time.
        """
        from shutil import which

        on_path = which(self.command) is not None
        return {
            "configured_command": self.command,
            "db_path": self.db_path,
            "timeout_s": self.timeout_s,
            "command_on_path": on_path,
            "available": on_path,
        }

    @classmethod
    def from_environment(cls) -> TrajectoryContextProvider:
        """Build from optional MAOS configuration environment variables."""
        return cls(
            command=os.environ.get("TRAJECTORY_COMMAND", "trajectory"),
            db_path=os.environ.get("TRAJECTORY_DB_PATH") or None,
        )
