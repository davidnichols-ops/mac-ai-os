"""MAOS Context API — local HTTP daemon.

Exposes the system's memory, context, and actions over a simple REST API
so any AI client (Codex, ChatGPT, Claude, local LLMs) can query it.

    maos serve                  # starts localhost:7777

Endpoints:
    GET  /                       # API info
    GET  /context                # universal context packet
    GET  /context/current        # alias for /context
    GET  /memory/search?q=       # RAG search
    GET  /workspace/current      # active workspace
    GET  /workspace/list         # all workspaces
    GET  /session/latest         # most recent session
    GET  /session/list           # recent sessions
    GET  /tasks                  # pending tasks
    GET  /tasks/all              # all tasks
    GET  /knowledge/entities?q=  # entity search
    GET  /knowledge/stats        # graph stats
    GET  /timeline?hours=        # recent events
    GET  /telemetry/latest       # latest telemetry snapshot
    GET  /blackboard             # blackboard state
    GET  /actions                # available actions
    POST /actions/request        # execute an action
    GET  /health                 # health check
"""

from app.server.app import create_server, run_server

__all__ = ["create_server", "run_server"]
