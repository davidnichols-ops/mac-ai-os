"""HTTP server for the MAOS Context API.

Uses stdlib http.server — no external dependencies. Runs a threaded
server on localhost:7777 by default.

The server holds a reference to the Runtime, so every request has
access to the full system: database, retriever, blackboard, actions.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from app.log import get_logger
from app.runtime import Runtime
from app.server.context import build_context_packet

log = get_logger(__name__)


class _Handler(BaseHTTPRequestHandler):
    """Request handler — has access to the runtime via the server."""

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        routes: dict[str, Any] = {
            "/": self._api_info,
            "/context": self._context,
            "/context/current": self._context,
            "/memory/search": self._memory_search,
            "/workspace/current": self._workspace_current,
            "/workspace/list": self._workspace_list,
            "/session/latest": self._session_latest,
            "/session/list": self._session_list,
            "/tasks": self._tasks,
            "/tasks/all": self._tasks_all,
            "/knowledge/entities": self._knowledge_entities,
            "/knowledge/stats": self._knowledge_stats,
            "/timeline": self._timeline,
            "/telemetry/latest": self._telemetry_latest,
            "/blackboard": self._blackboard,
            "/actions": self._actions,
            "/health": self._health,
        }

        handler = routes.get(path)
        if handler:
            try:
                handler(params)
            except Exception as e:
                log.warning("server.request_failed", path=path, error=str(e))
                self._json(500, {"error": str(e)})
        else:
            self._json(404, {"error": f"not found: {path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/actions/request":
            self._action_request()
        else:
            self._json(404, {"error": f"not found: {path}"})

    # --- route handlers ---

    def _api_info(self, _params: dict) -> None:
        self._json(200, {
            "name": "MAOS Context API",
            "version": "0.1.0",
            "endpoints": [
                "GET /context",
                "GET /memory/search?q=",
                "GET /workspace/current",
                "GET /workspace/list",
                "GET /session/latest",
                "GET /session/list",
                "GET /tasks",
                "GET /tasks/all",
                "GET /knowledge/entities?q=",
                "GET /knowledge/stats",
                "GET /timeline?hours=",
                "GET /telemetry/latest",
                "GET /blackboard",
                "GET /actions",
                "POST /actions/request",
                "GET /health",
            ],
        })

    def _context(self, params: dict) -> None:
        rt = self._runtime()
        rag_query = params.get("q")
        lookback = int(params.get("hours", "24"))
        packet = build_context_packet(
            rt.db, rt.blackboard,
            retriever=rt.retriever,
            lookback_hours=lookback,
            rag_query=rag_query,
        )
        self._json(200, packet)

    def _memory_search(self, params: dict) -> None:
        rt = self._runtime()
        q = params.get("q", "")
        if not q:
            self._json(400, {"error": "missing query parameter 'q'"})
            return
        top_k = int(params.get("top_k", "5"))
        try:
            results = rt.retriever.search(q, top_k=top_k)
            self._json(200, {
                "query": q,
                "count": len(results),
                "results": [
                    {
                        "text": r.text[:1000],
                        "source": r.source,
                        "score": round(r.score, 3),
                        "metadata": r.metadata,
                    }
                    for r in results
                ],
            })
        except Exception as e:
            self._json(500, {"error": str(e)})

    def _workspace_current(self, _params: dict) -> None:
        from app.workspace import detect_active_workspace

        rt = self._runtime()
        ws = detect_active_workspace(rt.db)
        self._json(200, ws or {"error": "no active workspace"})

    def _workspace_list(self, _params: dict) -> None:
        rt = self._runtime()
        rows = rt.db.all_workspaces()
        self._json(200, {
            "count": len(rows),
            "workspaces": [dict(r) for r in rows],
        })

    def _session_latest(self, _params: dict) -> None:
        rt = self._runtime()
        s = rt.db.latest_session()
        self._json(200, dict(s) if s else {"error": "no sessions"})

    def _session_list(self, params: dict) -> None:
        rt = self._runtime()
        limit = int(params.get("limit", "20"))
        rows = rt.db.recent_sessions(limit=limit)
        self._json(200, {
            "count": len(rows),
            "sessions": [dict(r) for r in rows],
        })

    def _tasks(self, _params: dict) -> None:
        rt = self._runtime()
        rows = rt.db.pending_tasks()
        self._json(200, {
            "count": len(rows),
            "tasks": [dict(r) for r in rows],
        })

    def _tasks_all(self, params: dict) -> None:
        rt = self._runtime()
        limit = int(params.get("limit", "50"))
        rows = rt.db.all_tasks(limit=limit)
        self._json(200, {
            "count": len(rows),
            "tasks": [dict(r) for r in rows],
        })

    def _knowledge_entities(self, params: dict) -> None:
        rt = self._runtime()
        q = params.get("q", "")
        if q:
            rows = rt.db.search_entities(q, limit=int(params.get("limit", "20")))
        else:
            rows = rt.db.query(
                "SELECT * FROM entities ORDER BY created_at DESC LIMIT ?",
                (int(params.get("limit", "20")),),
            )
        self._json(200, {
            "count": len(rows),
            "entities": [dict(r) for r in rows],
        })

    def _knowledge_stats(self, _params: dict) -> None:
        rt = self._runtime()
        entity_count = rt.db.query_one("SELECT COUNT(*) as n FROM entities")
        rel_count = rt.db.query_one("SELECT COUNT(*) as n FROM relationships")
        self._json(200, {
            "entities": entity_count["n"] if entity_count else 0,
            "relationships": rel_count["n"] if rel_count else 0,
        })

    def _timeline(self, params: dict) -> None:
        rt = self._runtime()
        hours = int(params.get("hours", "1"))
        from datetime import UTC, datetime, timedelta

        since = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        rows = rt.db.query_events(since=since, limit=int(params.get("limit", "100")))
        self._json(200, {
            "hours": hours,
            "count": len(rows),
            "events": [dict(r) for r in rows],
        })

    def _telemetry_latest(self, _params: dict) -> None:
        rt = self._runtime()
        rows = rt.db.query(
            "SELECT * FROM telemetry_snapshots ORDER BY timestamp DESC LIMIT 1"
        )
        if rows:
            self._json(200, dict(rows[0]))
        else:
            self._json(200, {"error": "no telemetry data"})

    def _blackboard(self, _params: dict) -> None:
        rt = self._runtime()
        self._json(200, rt.blackboard.snapshot())

    def _actions(self, _params: dict) -> None:
        rt = self._runtime()
        actions = []
        for name, action in rt.action_registry._actions.items():
            actions.append({
                "name": name,
                "description": action.description,
                "permissions": action.permissions,
                "reversible": action.reversible,
                "parameters": action.parameters,
            })
        self._json(200, {"count": len(actions), "actions": actions})

    def _action_request(self) -> None:
        rt = self._runtime()
        body = self._read_body()
        if body is None:
            return

        from app.actions.models import ActionRequest

        try:
            req = ActionRequest(
                action=body.get("action", ""),
                target=body.get("target", ""),
                parameters=body.get("parameters", {}),
            )
        except Exception as e:
            self._json(400, {"error": f"invalid request: {e}"})
            return

        result = rt.action_executor.execute(req, context={"db": rt.db})
        self._json(200, result.to_dict())

    def _health(self, _params: dict) -> None:
        rt = self._runtime()
        self._json(200, {
            "status": "ok",
            "database": rt.db.query_one("SELECT 1 as ok") is not None,
            "vectors": rt.retriever.count() if hasattr(rt.retriever, "count") else "unknown",
        })

    # --- helpers ---

    def _runtime(self) -> Runtime:
        return self.server.runtime  # type: ignore[attr-defined]

    def _read_body(self) -> dict | None:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._json(400, {"error": "empty body"})
            return None
        try:
            raw = self.rfile.read(content_length)
            return json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            self._json(400, {"error": f"invalid JSON: {e}"})
            return None

    def _json(self, status: int, data: Any) -> None:
        body = json.dumps(data, default=str, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        # Suppress default stderr logging — use structlog instead.
        log.debug("server.request", method=self.command, path=self.path,
                  status=self._response_code if hasattr(self, "_response_code") else 0)


class _Server(ThreadingHTTPServer):
    """HTTP server with a runtime reference."""

    runtime: Runtime

    def __init__(self, addr: tuple[str, int], runtime: Runtime) -> None:
        self.runtime = runtime
        super().__init__(addr, _Handler)


def create_server(runtime: Runtime, *, host: str = "127.0.0.1", port: int = 7777) -> _Server:
    """Create the HTTP server (doesn't start it)."""
    server = _Server((host, port), runtime)
    log.info("server.created", host=host, port=port)
    return server


def run_server(runtime: Runtime, *, host: str = "127.0.0.1", port: int = 7777) -> None:
    """Create and run the HTTP server (blocking)."""
    server = create_server(runtime, host=host, port=port)
    log.info("server.serving", url=f"http://{host}:{port}")
    print(f"\n  MAOS Context API — http://{host}:{port}\n")
    print("  Endpoints:")
    print("    GET  /context              — universal context packet")
    print("    GET  /memory/search?q=     — RAG search")
    print("    GET  /workspace/current    — active workspace")
    print("    GET  /session/latest       — latest session")
    print("    GET  /tasks                — pending tasks")
    print("    GET  /knowledge/entities   — graph search")
    print("    GET  /blackboard           — working memory")
    print("    POST /actions/request      — execute an action")
    print("    GET  /health               — health check")
    print("\n  Press Ctrl+C to stop.\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...")
        server.shutdown()
