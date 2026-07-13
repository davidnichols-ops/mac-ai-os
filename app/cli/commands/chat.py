"""`maos chat` — interactive session routed through the orchestrator."""

from __future__ import annotations

import typer

from app.conversation import Conversation
from app.runtime import build_runtime
from app.ui.console import print_markdown, print_panel


def chat() -> None:
    """Start an interactive conversation through the agent orchestrator."""
    rt = build_runtime()
    convo = Conversation(orchestrator=rt.orchestrator)
    print_panel("Mac AI OS chat. Ctrl-D / Ctrl-C to exit.", title="Chat")
    while True:
        try:
            text = typer.prompt("you", prompt_suffix="> ", default="", show_default=False)
        except (typer.Abort, EOFError, KeyboardInterrupt):
            typer.echo("\nbye.")
            return
        text = text.strip()
        if not text:
            continue
        answer = convo.send(text)
        print_markdown(answer)
