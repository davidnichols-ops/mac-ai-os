"""maos mcp — MCP Context Layer commands.

Subcommands:
  serve   — start the MCP server
  status  — show MCP server status, resources, tools, token
  test    — connect to a running MCP server and verify responses
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from app.runtime import build_runtime
from app.ui.console import print_panel

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

    rt = build_runtime()
    run_mcp_server(
        rt,
        transport=transport,
        host=host,
        port=port,
        require_token=not no_token,
    )


@mcp_app.command("status")
def status() -> None:
    """Show MCP server status, registered resources, tools, and token info."""
    from app.mcp.schemas import ALL_RESOURCES, ALL_TOOLS
    from app.mcp.server import TOKEN_FILE

    rt = build_runtime()

    # Check if MCP server is running on the default HTTP port.
    port = 7778
    running = _is_port_open("127.0.0.1", port)

    # Token status.
    token_exists = TOKEN_FILE.exists()
    token_preview = ""
    if token_exists:
        token = TOKEN_FILE.read_text().strip()
        token_preview = token[:8] + "..." + token[-4:] if len(token) > 12 else token

    # MCP request stats from telemetry.
    stats = rt.db.mcp_request_stats()

    # Build output.
    lines = [
        f"Server:        {'running' if running else 'not running'}",
        f"Transport:     {'http' if running else 'stdio (default)'}",
        f"Port:          {port}" + (" (listening)" if running else " (not listening)"),
        f"Token:         {'exists at ' + str(TOKEN_FILE) if token_exists else 'not generated'}",
    ]
    if token_preview:
        lines.append(f"Token preview: {token_preview}")

    lines.append("")
    lines.append(f"Resources ({len(ALL_RESOURCES)}):")
    for r in ALL_RESOURCES:
        lines.append(f"  {r.uri}")
        lines.append(f"    {r.description[:80]}")

    lines.append("")
    lines.append(f"Tools ({len(ALL_TOOLS)}):")
    for t in ALL_TOOLS:
        lines.append(f"  {t.name}")
        lines.append(f"    {t.description[:80]}")

    # Telemetry stats.
    lines.append("")
    lines.append("MCP Telemetry:")
    lines.append(f"  Total requests:  {stats['total']}")
    lines.append(f"  Successful:      {stats['success']}")
    lines.append(f"  Avg latency:     {stats['avg_latency_ms']}ms")
    if stats["by_kind"]:
        for kind, kstats in stats["by_kind"].items():
            lines.append(f"  {kind}: {kstats['count']} calls, avg {kstats['avg_latency_ms']}ms")

    print_panel("\n".join(lines), title="MCP Status")


@mcp_app.command("test")
def test(
    port: int = typer.Option(7778, "--port", "-p", help="Port of running MCP server."),
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host of running MCP server."),
) -> None:
    """Connect to a running MCP server and verify tool responses.

    Tests:
      - maos_get_context() — returns valid context packet
      - maos_search_memory() — returns search results
      - maos_list_tasks() — returns task list via ActionExecutor
    """
    from app.mcp.server import TOKEN_FILE

    # Check server is running.
    if not _is_port_open(host, port):
        console.print(f"[red]Error: No MCP server running at {host}:{port}[/red]")
        console.print("Start it with: maos mcp serve --transport http --port " + str(port))
        raise typer.Exit(1)

    # Get token.
    token = TOKEN_FILE.read_text().strip() if TOKEN_FILE.exists() else ""

    results: list[dict[str, Any]] = []

    # Test 1: maos_get_context
    results.append(_test_tool(host, port, token, "maos_get_context", {}))

    # Test 2: maos_search_memory
    results.append(_test_tool(host, port, token, "maos_search_memory", {"query": "test"}))

    # Test 3: maos_list_tasks
    results.append(_test_tool(host, port, token, "maos_list_tasks", {}))

    # Render results.
    table = Table(title="MCP Test Results")
    table.add_column("Tool", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Latency", style="yellow")
    table.add_column("Details")

    all_passed = True
    for r in results:
        status_str = "PASS" if r["ok"] else "FAIL"
        if not r["ok"]:
            all_passed = False
        table.add_row(
            r["tool"],
            status_str,
            f"{r['latency_ms']}ms",
            r["detail"][:60],
        )

    console.print(table)

    if all_passed:
        console.print("\n[green]All tests passed.[/green]")
    else:
        console.print("\n[red]Some tests failed.[/red]")
        raise typer.Exit(1)


# --- helpers ----------------------------------------------------------------


def _is_port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Check if a port is listening."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except OSError:
        return False


def _test_tool(
    host: str, port: int, token: str, tool_name: str, args: dict[str, Any],
) -> dict[str, Any]:
    """Call an MCP tool via HTTP and verify the response."""
    import urllib.request

    url = f"http://{host}:{port}/mcp"

    # MCP uses JSON-RPC. We need to initialize a session first, then call the tool.
    # For simplicity, we use the streamable-http protocol's single-request mode.

    # Step 1: Initialize session.
    init_req = json.dumps({
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "maos-test", "version": "0.1"},
        },
        "id": 1,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    start = time.time()
    try:
        req = urllib.request.Request(url, data=init_req, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read().decode()  # consume init response

        # Step 2: Call the tool.
        tool_req = json.dumps({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": args,
            },
            "id": 2,
        }).encode()

        req2 = urllib.request.Request(url, data=tool_req, headers=headers, method="POST")
        with urllib.request.urlopen(req2, timeout=15) as resp2:
            tool_resp = resp2.read().decode()

        latency_ms = int((time.time() - start) * 1000)

        # Parse the response — could be SSE format or plain JSON.
        # Try to extract the JSON from the response.
        result_text = tool_resp
        if "data: " in tool_resp:
            # SSE format — extract the JSON payload.
            for line in tool_resp.split("\n"):
                if line.startswith("data: "):
                    result_text = line[6:]
                    break

        result = json.loads(result_text)

        # Check if the result has an error.
        if "error" in result:
            return {
                "tool": tool_name,
                "ok": False,
                "latency_ms": latency_ms,
                "detail": f"error: {result['error'].get('message', '?')[:60]}",
            }

        # Check if the result content is valid.
        content = result.get("result", {}).get("content", [])
        if content:
            text = content[0].get("text", "")
            # Verify the response is valid JSON.
            try:
                parsed = json.loads(text)
                if "error" in parsed and "workspace" not in parsed:
                    return {
                        "tool": tool_name,
                        "ok": False,
                        "latency_ms": latency_ms,
                        "detail": f"error: {str(parsed['error'])[:60]}",
                    }
            except json.JSONDecodeError:
                pass  # Not all responses are JSON

        return {
            "tool": tool_name,
            "ok": True,
            "latency_ms": latency_ms,
            "detail": "valid response",
        }

    except Exception as e:
        latency_ms = int((time.time() - start) * 1000)
        return {
            "tool": tool_name,
            "ok": False,
            "latency_ms": latency_ms,
            "detail": f"exception: {str(e)[:60]}",
        }
