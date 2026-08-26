"""Device-wide RAG indexing, safety, hybrid retrieval, and grounding tests."""

from __future__ import annotations

from pathlib import Path

from app.retrieval.device import (
    DeviceCatalog,
    DeviceIndexPolicy,
    DeviceRAGService,
)
from tests.conftest import FakeModel, FakeRetriever


def _service(tmp_path: Path) -> tuple[DeviceRAGService, FakeRetriever]:
    retriever = FakeRetriever()
    service = DeviceRAGService(
        retriever,
        DeviceCatalog(tmp_path / "device_rag.db"),
        policy=DeviceIndexPolicy(roots=(tmp_path,)),
        chunk_size=120,
        chunk_overlap=20,
    )
    return service, retriever


def test_policy_excludes_credentials_dependencies_and_binaries(tmp_path):
    policy = DeviceIndexPolicy(roots=(tmp_path,))
    safe = tmp_path / "notes.md"
    secret = tmp_path / ".env.local"
    key = tmp_path / "server.key"
    binary = tmp_path / "archive.zip"
    for path in (safe, secret, key, binary):
        path.write_text("value")

    assert policy.accepts(safe) == (True, "accepted")
    assert policy.accepts(secret)[0] is False
    assert policy.accepts(key)[0] is False
    assert policy.accepts(binary)[0] is False


def test_incremental_sync_hybrid_search_and_deletion(tmp_path):
    service, retriever = _service(tmp_path)
    note = tmp_path / "intel.md"
    note.write_text("gameID 10864031 belongs to Player 67 and country Iowa")
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=do-not-index")
    dependency = tmp_path / "node_modules"
    dependency.mkdir()
    (dependency / "leak.md").write_text("must not be indexed")

    first = service.sync()
    assert first.indexed == 1
    assert first.errors == 0
    assert service.status()["files"] == 1
    assert str(note) in retriever._store
    assert not any("do-not-index" in chunk["text"] for chunks in retriever._store.values() for chunk in chunks)

    results = service.search("10864031", top_k=3)
    assert results
    assert results[0].source == str(note)
    assert "lexical" in results[0].metadata["retrieval_signals"]

    second = service.sync()
    assert second.indexed == 0
    assert second.unchanged == 1

    note.write_text("gameID 10864031 now maps to Player 53")
    changed = service.sync()
    assert changed.indexed == 1

    note.unlink()
    deleted = service.sync()
    assert deleted.deleted == 1
    assert service.status()["files"] == 0
    assert str(note) not in retriever._store
    service.catalog.close()


def test_limited_scan_never_deletes_unseen_files(tmp_path):
    service, _ = _service(tmp_path)
    first = tmp_path / "a.md"
    second = tmp_path / "b.md"
    first.write_text("alpha")
    second.write_text("beta")
    service.sync()
    assert service.status()["files"] == 2

    limited = service.sync(force=True, max_files=1)
    assert limited.limited is True
    assert limited.deleted == 0
    assert service.status()["files"] == 2
    service.catalog.close()


def test_grounded_answer_returns_numbered_sources(tmp_path):
    service, _ = _service(tmp_path)
    note = tmp_path / "answer.md"
    note.write_text("The launch code name is Orchard Blue.")
    service.sync()

    payload = service.answer("What is the launch code name?", FakeModel(), top_k=2)
    assert payload["sources"][0]["id"] == 1
    assert payload["sources"][0]["path"] == str(note)
    assert "Evidence:" in payload["answer"]
    service.catalog.close()
