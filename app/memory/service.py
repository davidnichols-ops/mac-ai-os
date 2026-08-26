"""Capture personal notes into MAOS, RAG, and the audited LAN archive."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.db import Database
from app.events.bus import EventBus
from app.events.models import Event
from app.persistence.lan_vault import LanVaultClient, LanVaultReceipt
from app.retrieval.base import Retriever

PERSONAL_MEMORY_CAPTURED = "PERSONAL_MEMORY_CAPTURED"
PERSONAL_MEMORY_ARCHIVED = "PERSONAL_MEMORY_ARCHIVED"


@dataclass(frozen=True)
class MemoryCaptureResult:
    ok: bool
    memory_id: str = ""
    content_hash: str = ""
    indexed: bool = False
    archive: LanVaultReceipt | None = None
    duplicate: bool = False
    error: str = ""


class PersonalMemoryService:
    """Single write path for user-authored, durable personal information."""

    def __init__(
        self,
        db: Database,
        retriever: Retriever,
        *,
        bus: EventBus | None = None,
        vault: LanVaultClient | None = None,
        notes_dir: Path,
    ) -> None:
        self.db = db
        self.retriever = retriever
        self.bus = bus
        self.vault = vault
        self.notes_dir = notes_dir
        self.notes_dir.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        content: str,
        *,
        title: str = "",
        tags: tuple[str, ...] = (),
        source: str = "manual",
        source_ref: str = "",
        author_id: str = "",
        archive: bool = True,
    ) -> MemoryCaptureResult:
        normalized = " ".join(content.strip().split())
        if not normalized:
            return MemoryCaptureResult(False, error="memory content is empty")
        if len(normalized) > 50_000:
            return MemoryCaptureResult(False, error="memory content exceeds 50,000 characters")

        content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        existing = self.db.personal_memory_by_hash(content_hash)
        if existing:
            return MemoryCaptureResult(
                True,
                memory_id=str(existing["id"]),
                content_hash=content_hash,
                indexed=True,
                duplicate=True,
            )

        memory_id = uuid.uuid4().hex
        safe_title = self._clean_title(title) or self._clean_title(normalized[:72])
        safe_tags = tuple(sorted({self._clean_tag(tag) for tag in tags if self._clean_tag(tag)}))
        created_at = datetime.now(UTC).isoformat()
        note_path = self.notes_dir / f"{memory_id}.md"
        note_path.write_text(self._render_note(safe_title, normalized, safe_tags, created_at), encoding="utf-8")
        archive_status = "pending" if archive else "disabled"
        self.db.upsert_personal_memory(
            memory_id,
            content_hash,
            normalized,
            title=safe_title,
            tags=json.dumps(safe_tags),
            source=source,
            source_ref=source_ref,
            author_id=author_id,
            archive_status=archive_status,
        )

        source_key = f"personal-memory:{memory_id}"
        indexed = False
        try:
            self.retriever.update(
                source_key,
                [{
                    "text": normalized,
                    "metadata": {
                        "source": "personal_memory",
                        "memory_id": memory_id,
                        "title": safe_title,
                        "tags": ",".join(safe_tags),
                        "origin": source,
                        "source_ref": source_ref,
                        "created_at": created_at,
                    },
                }],
            )
            indexed = True
            self.db.execute(
                "UPDATE personal_memories SET indexed_at = ?, updated_at = ? WHERE id = ?",
                (created_at, created_at, memory_id),
            )
        except Exception as exc:
            self.db.update_personal_memory_archive(memory_id, status="failed", error=f"index: {exc}")
            return MemoryCaptureResult(
                False, memory_id, content_hash, error=f"memory saved but indexing failed: {exc}"
            )

        if self.bus:
            self.bus.publish(Event(PERSONAL_MEMORY_CAPTURED, {
                "memory_id": memory_id,
                "source": source,
                "source_ref": source_ref,
                "content_hash": content_hash,
            }))

        receipt: LanVaultReceipt | None = None
        if archive and self.vault:
            receipt = self.vault.archive_file(note_path, archive_name=f"{memory_id}.md")
            self.db.update_personal_memory_archive(
                memory_id,
                status=receipt.status,
                archive_path=receipt.archive_path,
                archive_hash=receipt.remote_hash,
                error=receipt.detail if not receipt.verified else "",
            )
            if self.bus:
                self.bus.publish(Event(PERSONAL_MEMORY_ARCHIVED, {
                    "memory_id": memory_id,
                    "status": receipt.status,
                    "archive_path": receipt.archive_path,
                    "verified": receipt.verified,
                }))

        return MemoryCaptureResult(
            True, memory_id, content_hash, indexed=indexed, archive=receipt
        )

    def archive_pending(self, *, limit: int = 100) -> dict[str, int]:
        """Retry durable archive receipts without changing local memory."""
        if not self.vault:
            return {"attempted": 0, "verified": 0, "failed": 0}
        attempted = verified = failed = 0
        for row in self.db.pending_personal_memories(limit):
            note_path = self.notes_dir / f"{row['id']}.md"
            if not note_path.is_file():
                self.db.update_personal_memory_archive(
                    row["id"], status="failed", error="local note missing from outbox"
                )
                failed += 1
                continue
            attempted += 1
            receipt = self.vault.archive_file(note_path, archive_name=f"{row['id']}.md")
            self.db.update_personal_memory_archive(
                row["id"],
                status=receipt.status,
                archive_path=receipt.archive_path,
                archive_hash=receipt.remote_hash,
                error=receipt.detail if not receipt.verified else "",
            )
            if receipt.verified:
                verified += 1
            else:
                failed += 1
        return {"attempted": attempted, "verified": verified, "failed": failed}

    @staticmethod
    def _clean_title(value: str) -> str:
        return re.sub(r"[^\w .-]+", "", value, flags=re.UNICODE).strip(" .")[:120]

    @staticmethod
    def _clean_tag(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "", value.strip().lower())[:40]

    @staticmethod
    def _render_note(title: str, content: str, tags: tuple[str, ...], created_at: str) -> str:
        tag_line = " ".join(f"#{tag}" for tag in tags)
        return f"# {title}\n\n{content}\n\n---\nCaptured: {created_at}\nTags: {tag_line}\n"
