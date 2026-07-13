"""CLI entrypoint (Typer).

    uv run maos verify
    uv run maos models
    uv run maos index <path>
    uv run maos ask "..."
    uv run maos rag "..."
    uv run maos chat
"""

from __future__ import annotations

import typer

from app.cli.commands import ask, chat, index, models, rag, verify, watch
from app.cli.commands.git import git_app

app = typer.Typer(
    name="maos",
    help="Mac AI OS — a local-first intelligence layer for macOS.",
    no_args_is_help=True,
    add_completion=False,
)

app.command()(verify.verify)
app.command()(models.models)
app.command()(index.index)
app.command()(watch.watch)
app.command()(ask.ask)
app.command()(rag.rag)
app.command()(chat.chat)
app.add_typer(git_app, name="git")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
