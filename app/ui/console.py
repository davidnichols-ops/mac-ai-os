"""Console UI helpers (rich)."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

console = Console()


def print_panel(text: str, *, title: str | None = None) -> None:
    console.print(Panel(text, title=title, border_style="cyan"))


def print_markdown(text: str) -> None:
    console.print(Markdown(text))
