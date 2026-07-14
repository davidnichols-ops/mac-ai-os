"""MAOS MCP Context Layer — Model Context Protocol adapter.

Exposes MAOS memory, context, and permissioned actions through MCP so
external AI clients (Codex, Claude, ChatGPT-compatible clients, local
agents) can use MAOS as their local context backend.

Architecture:

    AI Client  →  MCP Adapter  →  Context API  →  MAOS Core

The MCP layer is a thin adapter. All business logic lives in the existing
Context API (app/server/) and ActionExecutor (app/actions/). This package
contains NO business logic — it only translates between MCP protocol
messages and existing MAOS functions.

Usage:
    maos mcp serve                          # stdio transport (default)
    maos mcp serve --transport http         # HTTP transport
    maos mcp serve --port 7778 --host 127.0.0.1
"""

from app.mcp.server import create_mcp_server, run_mcp_server

__all__ = ["create_mcp_server", "run_mcp_server"]
