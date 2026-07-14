"""maos serve — start the local Context API daemon."""

from __future__ import annotations

import typer
from rich.console import Console

from app.runtime import build_runtime

console = Console()


def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address."),
    port: int = typer.Option(7777, "--port", "-p", help="Port number."),
) -> None:
    """Start the MAOS Context API (local HTTP daemon)."""
    from app.server import run_server

    rt = build_runtime()
    run_server(rt, host=host, port=port)
