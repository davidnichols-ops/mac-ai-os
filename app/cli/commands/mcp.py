"""maos mcp serve — start the MCP Context Layer server."""

from __future__ import annotations

import typer
from rich.console import Console

console = Console()

mcp_app = typer.Typer(
    name="mcp",
    help="MCP Context Layer — expose MAOS to external AI clients.",
    no_args_is_help=True,
)


@mcp_app.command("serve")
def serve(
    transport: str = typer.Option(
        "stdio",
        "--transport",
        "-t",
        help="Transport: 'stdio' (default) or 'http'.",
    ),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Bind address (HTTP only)."),
    port: int = typer.Option(7778, "--port", "-p", help="Port (HTTP only)."),
    no_token: bool = typer.Option(
        False, "--no-token", help="Disable capability token (not recommended)."
    ),
) -> None:
    """Start the MAOS MCP server.

    By default uses stdio transport for integration with CLI-based MCP
    clients (Claude Code, Codex, etc.). Use --transport http for
    HTTP-based clients.

    Examples:
        maos mcp serve                           # stdio (default)
        maos mcp serve --transport http          # HTTP on :7778
        maos mcp serve -t http -p 9000           # HTTP on port 9000
    """
    from app.mcp import run_mcp_server
    from app.runtime import build_runtime

    rt = build_runtime()
    run_mcp_server(
        rt,
        transport=transport,
        host=host,
        port=port,
        require_token=not no_token,
    )
