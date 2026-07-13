"""Plugin discovery.

Plugins are Python modules that expose a `register(registry)` callable or a
`tools` list. They are loaded from a configured directory (default:
`app/plugins/` — future) or from installed entry points.

This is a minimal, dependency-free loader: it imports a module path and
calls `register` if present. A full entry-point based loader can replace it
later without touching the rest of the codebase.
"""

from __future__ import annotations

import importlib

from app.log import get_logger
from app.tools.registry import ToolRouter

log = get_logger(__name__)


def load_plugins(router: ToolRouter, modules: list[str] | None = None) -> int:
    """Import each module path; if it defines `register(router)`, call it."""
    count = 0
    for mod_path in modules or []:
        try:
            mod = importlib.import_module(mod_path)
        except Exception as e:
            log.warning("plugin.load.failed", module=mod_path, error=str(e))
            continue
        register = getattr(mod, "register", None)
        if callable(register):
            register(router)
            count += 1
        else:
            log.warning("plugin.no_register", module=mod_path)
    return count


def discover_entry_points() -> list[str]:  # pragma: no cover - placeholder
    """Future: read `[project.entry-points."mac_ai_os.tools"]`."""
    return []
