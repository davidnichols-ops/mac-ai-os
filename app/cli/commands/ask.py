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

    # For v2 models, the model self-selects mode via control tokens.
    # For v1 models with an adaptive router, use the router.
    enable_thinking = None
    if rt.adaptive_router is not None:
        mode, confidence, meta = rt.adaptive_router.route(prompt)
        enable_thinking = (mode == "THINK")
        typer.echo(f"  [{mode}] confidence={confidence:.0%} family={meta.get('family', '?')} "
                   f"diff={meta.get('difficulty', '?')}", err=True)

    system = "You are Mac AI OS. Answer concisely and accurately."
    kwargs = {}
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    resp = rt.planner.chat([Message("system", system), Message("user", prompt)], **kwargs)

    # Show mode if the model self-selected via control tokens (v2)
    if hasattr(resp, "mode") and resp.mode:
        typer.echo(f"  [{resp.mode.upper()}]", err=True)

    print_markdown(resp.text)
