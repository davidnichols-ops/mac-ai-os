"""Tests for the Context API server."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from http.client import HTTPConnection

from app.blackboard import Blackboard
from app.db import Database
from app.server.app import create_server
from app.server.context import build_context_packet


@dataclass
class FakeResult:
    text: str
    source: str
    score: float
    metadata: dict


class FakeRetriever:
    """Minimal retriever for testing — no model loading."""

    def __init__(self, results: list[FakeResult] | None = None):
        self._results = results or []

    def search(self, query: str, *, top_k: int = 5, where: dict | None = None):
        return self._results[:top_k]

    def index(self, source: str, chunks: list[dict]) -> int:
        return len(chunks)

    def count(self) -> int:
        return len(self._results)


def _make_db(tmp_path) -> Database:
    return Database(str(tmp_path / "test_server.db"))


def test_context_packet_empty_db(tmp_path):
    """Context packet builds without errors on an empty database."""
    db = _make_db(tmp_path)
    bb = Blackboard()
    packet = build_context_packet(db, bb)
    assert "timestamp" in packet
    assert "workspace" in packet
    assert "session" in packet
    assert "tasks" in packet
    assert "recent_decisions" in packet
    assert "related_knowledge" in packet
    assert "rag_results" in packet
    assert "system" in packet
    assert "blackboard" in packet
    assert "recent_activity" in packet
    assert packet["tasks"] == []
    assert packet["session"]["active"] is False
    db.close()


def test_context_packet_with_data(tmp_path):
    """Context packet includes tasks and blackboard state."""
    db = _make_db(tmp_path)
    bb = Blackboard()
    db.create_task("Test task", source="test")
    bb.set_goal("Build MCP server")

    packet = build_context_packet(db, bb)
    assert len(packet["tasks"]) == 1
    assert packet["tasks"][0]["title"] == "Test task"
    assert packet["blackboard"]["current_goal"] == "Build MCP server"
    db.close()


def test_context_packet_with_rag(tmp_path):
    """Context packet includes RAG results when a query is provided."""
    db = _make_db(tmp_path)
    bb = Blackboard()
    retriever = FakeRetriever([
        FakeResult(text="ActionExecutor validates permissions.", source="test.py", score=0.95, metadata={})
    ])

    packet = build_context_packet(db, bb, retriever=retriever, rag_query="action executor")
    assert len(packet["rag_results"]) == 1
    assert "ActionExecutor" in packet["rag_results"][0]["text"]
    db.close()


def test_server_health(tmp_path):
    """Server responds to /health."""
    db = _make_db(tmp_path)
    bb = Blackboard()
    retriever = FakeRetriever()

    # Create a minimal mock runtime.
    class FakeRuntime:
        pass

    rt = FakeRuntime()
    rt.db = db
    rt.blackboard = bb
    rt.retriever = retriever

    server = create_server(rt, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.2)
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["status"] == "ok"
        conn.close()
    finally:
        server.shutdown()
        db.close()


def test_server_context(tmp_path):
    """Server responds to /context with a valid packet."""
    db = _make_db(tmp_path)
    bb = Blackboard()
    retriever = FakeRetriever()

    class FakeRuntime:
        pass

    rt = FakeRuntime()
    rt.db = db
    rt.blackboard = bb
    rt.retriever = retriever

    server = create_server(rt, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.2)
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/context")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert "workspace" in data
        assert "session" in data
        assert "tasks" in data
        conn.close()
    finally:
        server.shutdown()
        db.close()


def test_server_api_info(tmp_path):
    """Server responds to / with API info."""
    db = _make_db(tmp_path)
    bb = Blackboard()
    retriever = FakeRetriever()

    class FakeRuntime:
        pass

    rt = FakeRuntime()
    rt.db = db
    rt.blackboard = bb
    rt.retriever = retriever

    server = create_server(rt, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.2)
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        resp = conn.getresponse()
        assert resp.status == 200
        data = json.loads(resp.read())
        assert data["name"] == "MAOS Context API"
        assert any("context" in e for e in data["endpoints"])
        conn.close()
    finally:
        server.shutdown()
        db.close()


def test_server_404(tmp_path):
    """Server returns 404 for unknown paths."""
    db = _make_db(tmp_path)
    bb = Blackboard()
    retriever = FakeRetriever()

    class FakeRuntime:
        pass

    rt = FakeRuntime()
    rt.db = db
    rt.blackboard = bb
    rt.retriever = retriever

    server = create_server(rt, host="127.0.0.1", port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        time.sleep(0.2)
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/nonexistent")
        resp = conn.getresponse()
        assert resp.status == 404
        conn.close()
    finally:
        server.shutdown()
        db.close()


def test_server_auth_and_device_rag_status(tmp_path):
    """Sensitive endpoints require a bearer token when production auth is enabled."""
    db = _make_db(tmp_path)
    bb = Blackboard()
    retriever = FakeRetriever()

    class FakeRuntime:
        pass

    rt = FakeRuntime()
    rt.db = db
    rt.blackboard = bb
    rt.retriever = retriever
    server = create_server(rt, host="127.0.0.1", port=0)
    server.capability_token = "test-capability"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/rag/status")
        assert conn.getresponse().status == 401
        conn.close()

        conn = HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request(
            "GET",
            "/rag/status",
            headers={"Authorization": "Bearer test-capability"},
        )
        response = conn.getresponse()
        assert response.status == 200
        payload = json.loads(response.read())
        assert payload["files"] == 0
        assert payload["policy"]["sensitive_files_indexed"] is False
        conn.close()
    finally:
        server.shutdown()
        server.server_close()
        db.close()


def test_server_rejects_non_loopback_bind(tmp_path):
    db = _make_db(tmp_path)

    class FakeRuntime:
        pass

    rt = FakeRuntime()
    rt.db = db
    rt.blackboard = Blackboard()
    rt.retriever = FakeRetriever()
    try:
        import pytest

        with pytest.raises(ValueError, match="loopback"):
            create_server(rt, host="0.0.0.0", port=0)
    finally:
        db.close()
