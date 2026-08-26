"""Tests for durable personal memory capture."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.actions import ActionExecutor, ActionRegistry, PermissionChecker, builtin_actions
from app.db import Database
from app.events.bus import EventBus
from app.memory.service import PersonalMemoryService
from app.persistence.lan_vault import LanVaultReceipt
from app.slack.auth import SlackAuth
from app.slack.commands import SlackCommandRouter


class FakeRetriever:
    def __init__(self) -> None:
        self.calls: list[tuple[str, list[dict]]] = []

    def update(self, source: str, chunks: list[dict]) -> int:
        self.calls.append((source, chunks))
        return len(chunks)


class FakeVault:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[tuple[Path, str]] = []

    def archive_file(self, path: Path, *, archive_name: str) -> LanVaultReceipt:
        self.calls.append((path, archive_name))
        return LanVaultReceipt(
            "verified", f"personal/{archive_name}", "h" * 64, "h" * 64,
            "remote hash verified",
        )


def test_capture_is_local_first_indexed_and_archived(tmp_path):
    db = Database(str(tmp_path / "maos.db"))
    retriever = FakeRetriever()
    vault = FakeVault(tmp_path)
    service = PersonalMemoryService(
        db,
        retriever,
        vault=vault,
        notes_dir=tmp_path / "notes",
    )

    result = service.capture(
        "  Remember the LAN archive is the durable secondary store.  ",
        tags=("Slack", "personal"),
        source="slack",
        source_ref="C123:123.456",
        author_id="U123",
    )

    assert result.ok is True
    assert result.indexed is True
    assert result.archive is not None and result.archive.verified is True
    assert len(retriever.calls) == 1
    row = db.personal_memory(result.memory_id)
    assert row is not None
    assert row["archive_status"] == "verified"
    assert row["source_ref"] == "C123:123.456"
    assert list((tmp_path / "notes").glob("*.md"))
    db.close()


def test_capture_deduplicates_content(tmp_path):
    db = Database(str(tmp_path / "maos.db"))
    retriever = FakeRetriever()
    service = PersonalMemoryService(db, retriever, notes_dir=tmp_path / "notes")

    first = service.capture("same thought", archive=False)
    second = service.capture("  same   thought ", archive=False)

    assert first.ok and second.ok
    assert first.memory_id == second.memory_id
    assert second.duplicate is True
    assert len(retriever.calls) == 1
    db.close()


def test_slack_remember_uses_audited_action_path(tmp_path):
    db = Database(str(tmp_path / "maos.db"))
    retriever = FakeRetriever()
    service = PersonalMemoryService(db, retriever, notes_dir=tmp_path / "notes")
    registry = ActionRegistry()
    for action, handler in builtin_actions():
        registry.register(action, handler)
    executor = ActionExecutor(
        registry,
        PermissionChecker({"memory.write"}),
        db,
        bus=EventBus(db=db),
    )
    runtime = SimpleNamespace(
        db=db,
        memory=service,
        action_executor=executor,
        settings=SimpleNamespace(lan_vault=SimpleNamespace(archive_root="personal")),
        vault=SimpleNamespace(),
    )
    router = SlackCommandRouter(runtime, SlackAuth(rate_limit_seconds=0))

    response = router.handle(
        "U123",
        "remember The bridge should preserve Slack provenance.",
        source_context={"channel": "C123", "ts": "123.456"},
    )

    assert response["ok"] is True
    assert "Indexed personal memory" in response["text"]
    action = db.query_one(
        "SELECT action_name, approved, executed FROM action_log ORDER BY id DESC LIMIT 1"
    )
    assert dict(action) == {
        "action_name": "capture_personal_memory",
        "approved": 1,
        "executed": 1,
    }
    memory = db.list_personal_memories(1)[0]
    assert memory["source_ref"] == "C123:123.456"
    db.close()
