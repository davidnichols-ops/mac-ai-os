"""Observers package — sensors that capture user activity."""

from app.observers.shell import import_shell_history, search_history

__all__ = ["import_shell_history", "search_history"]
