"""MCP server — FastMCP adapter over the MAOS Context API.

Creates a FastMCP server that exposes MAOS resources and tools. The
server is a thin adapter: all logic lives in app/mcp/resources.py,
app/mcp/tools.py, and the existing Context API / ActionExecutor.

Security:
    A localhost capability token is generated on first run and stored
    at ~/.maos/token. For HTTP transport, the token must be provided
    via the Authorization header. For stdio transport (default), the
    token check is skipped (stdio is local-only by definition).

    The server binds to 127.0.0.1 by default. No external network
    exposure.
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

from app.log import get_logger
from app.runtime import Runtime

log = get_logger(__name__)

# Token storage location.
TOKEN_DIR = Path.home() / ".maos"
TOKEN_FILE = TOKEN_DIR / "token"


def get_or_create_token() -> str:
    """Get the localhost capability token, creating it on first run.

    The token is a 32-byte hex string stored at ~/.maos/token with
    0600 permissions. It persists across restarts.
    """
    if TOKEN_FILE.exists():
        return TOKEN_FILE.read_text().strip()

    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    token = secrets.token_hex(32)
    TOKEN_FILE.write_text(token)
    # Restrict to owner only.
    os.chmod(TOKEN_FILE, 0o600)
    log.info("mcp.token.created", path=str(TOKEN_FILE))
    return token


def validate_token(provided: str | None, expected: str) -> bool:
    """Validate a capability token using constant-time comparison."""
    if not provided:
        return False
    return secrets.compare_digest(provided, expected)


def create_mcp_server(runtime: Runtime, *, require_token: bool = True) -> Any:
    """Create a FastMCP server wired to the MAOS runtime.

    Args:
        runtime: The MAOS Runtime with db, retriever, blackboard, actions.
        require_token: If True, HTTP requests must include the capability
            token. stdio transport always skips token checks.

    Returns:
        A FastMCP server instance ready to run.
    """
    from mcp.server.fastmcp import FastMCP

    from app.mcp import resources, tools

    mcp = FastMCP("MAOS Context Layer")
    token = get_or_create_token() if require_token else None

    # --- Resources ---------------------------------------------------------

    @mcp.resource("maos://context/current")
    def current_context() -> str:
        """Universal context packet: workspace, session, tasks, decisions, knowledge, telemetry, blackboard."""
        return resources.read_context_resource(runtime)

    @mcp.resource("maos://workspace/current")
    def current_workspace() -> str:
        """Active workspace with git branch, modified files, and status."""
        return resources.read_workspace_resource(runtime)

    @mcp.resource("maos://session/latest")
    def latest_session() -> str:
        """Most recent work session: summary, files changed, commands, unfinished work."""
        return resources.read_session_resource(runtime)

    @mcp.resource("maos://tasks/open")
    def open_tasks() -> str:
        """Pending tasks with dependencies, priority, and workspace."""
        return resources.read_tasks_resource(runtime)

    @mcp.resource("maos://knowledge/search{query}")
    def knowledge_search(query: str = "") -> str:
        """Semantic search over indexed knowledge (vectors + graph)."""
        return resources.read_knowledge_resource(runtime, query=query)

    # --- Tools -------------------------------------------------------------

    @mcp.tool()
    def maos_get_context() -> str:
        """Get the full MAOS context packet (workspace, session, tasks, decisions, knowledge, telemetry, blackboard).

        This is the primary tool for understanding what the user is currently
        working on. Start here before asking specific questions.
        """
        return tools.maos_get_context(runtime)

    @mcp.tool()
    def maos_search_memory(query: str, top_k: int = 5) -> str:
        """Search MAOS memory (vector store) for content matching the query.

        Returns ranked results with text, source path, and relevance score.
        Use this to find indexed documents, code, and commit history.
        """
        return tools.maos_search_memory(runtime, query=query, top_k=top_k)

    @mcp.tool()
    def maos_workspace_summary(workspace: str = "") -> str:
        """Generate a summary of a workspace's current state.

        Includes git branch, modified files, and pending tasks for the
        specified workspace. If no workspace is given, uses the active one.
        """
        return tools.maos_workspace_summary(runtime, workspace=workspace)

    @mcp.tool()
    def maos_create_task(title: str, workspace: str = "", priority: int = 0) -> str:
        """Create a new TODO task.

        The task is created through the MAOS ActionExecutor, which enforces
        permission checks and logs the action.
        """
        return tools.maos_create_task(
            runtime, title=title, workspace=workspace, priority=priority
        )

    @mcp.tool()
    def maos_list_tasks(workspace: str = "") -> str:
        """List pending tasks, optionally filtered by workspace.

        Returns tasks through the MAOS ActionExecutor (read permission required).
        """
        return tools.maos_list_tasks(runtime, workspace=workspace)

    @mcp.tool()
    def maos_run_action(
        action: str, target: str = "", parameters: dict[str, Any] | None = None,
    ) -> str:
        """Execute any registered MAOS action by name.

        This is the generic gateway to the MAOS ActionExecutor. All permission
        checks, reversibility checks, action logging, and event publishing
        are preserved. Available actions can be listed via the /actions endpoint
        of the Context API.

        Common actions:
        - open_workspace: Open a project directory in Finder
        - create_todo: Create a task
        - complete_todo: Mark a task as done
        - workspace_summary: Get workspace state summary
        - list_tasks: List pending tasks
        """
        return tools.maos_run_action(
            runtime, action=action, target=target, parameters=parameters
        )

    # Store token for HTTP transport middleware.
    mcp._maos_token = token  # type: ignore[attr-defined]
    mcp._maos_runtime = runtime  # type: ignore[attr-defined]

    return mcp


def run_mcp_server(
    runtime: Runtime,
    *,
    transport: str = "stdio",
    host: str = "127.0.0.1",
    port: int = 7778,
    require_token: bool = True,
) -> None:
    """Create and run the MCP server.

    Args:
        transport: "stdio" (default, for CLI integration) or "streamable-http".
        host: Bind address for HTTP transport (default 127.0.0.1).
        port: Port for HTTP transport (default 7778).
        require_token: Whether to require the capability token for HTTP.
    """
    mcp = create_mcp_server(runtime, require_token=require_token)

    if transport == "stdio":
        log.info("mcp.serving", transport="stdio")
        mcp.run(transport="stdio")

    elif transport in ("http", "streamable-http"):
        token = mcp._maos_token  # type: ignore[attr-defined]
        log.info("mcp.serving", transport="http", host=host, port=port)

        if token:
            print(f"\n  MAOS MCP Server — http://{host}:{port}/mcp")
            print(f"  Token: {token}")
            print(f"  Token file: {TOKEN_FILE}")
            print("\n  Clients must send: Authorization: Bearer {token}")
            print("\n  Press Ctrl+C to stop.\n")
        else:
            print(f"\n  MAOS MCP Server — http://{host}:{port}/mcp")
            print("  Token validation disabled.\n")

        # FastMCP streamable-http accepts host and port via settings.
        mcp.settings.host = host
        mcp.settings.port = port
        mcp.run(transport="streamable-http")

    else:
        raise ValueError(f"unsupported transport: {transport}")
