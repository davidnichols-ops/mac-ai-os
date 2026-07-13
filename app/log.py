"""Structured logging setup using structlog.

Call `configure_logging(settings)` once at startup. Every module then does:

    import structlog
    log = structlog.get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog

from app.config.settings import Settings


def configure_logging(settings: Settings) -> None:
    level = getattr(logging, settings.logging.level.upper(), logging.INFO)
    log_dir = Path(settings.logging.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Standard logging handlers (console + file).
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    file_handler = logging.FileHandler(log_dir / "mac-ai-os.log")
    file_handler.setLevel(level)
    handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=handlers,
        force=True,
    )

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.logging.json_logs:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=shared_processors + [renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
