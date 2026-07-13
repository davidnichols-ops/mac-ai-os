"""`maos ask "..."` — one-shot local prompt to the planner (no retrieval)."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_markdown


def ask(
    prompt: str = typer.Argument(..., help="Prompt for the planner model"),
) -> None:
    """Answer a single prompt using the planner LLM directly."""
    rt = build_runtime()
    from app.models.types import Message

    system = "You are Mac AI OS. Answer concisely and accurately."
    resp = rt.planner.chat([Message("system", system), Message("user", prompt)])
    print_markdown(resp.text)
