"""Durable, incremental Slack history ingestion for MAOS."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _oldest_timestamp(days: int) -> str:
    return str((datetime.now(UTC) - timedelta(days=days)).timestamp())


def _next_cursor(response: Any) -> str:
    metadata = response.get("response_metadata") or {}
    return str(metadata.get("next_cursor") or "").strip()


@dataclass(frozen=True)
class SlackSyncResult:
    channels_seen: int
    channels_synced: int
    messages_seen: int
    messages_changed: int
    users_seen: int
    errors: tuple[str, ...]
    rag_refreshed: bool
    completed_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "channels_seen": self.channels_seen,
            "channels_synced": self.channels_synced,
            "messages_seen": self.messages_seen,
            "messages_changed": self.messages_changed,
            "users_seen": self.users_seen,
            "errors": list(self.errors),
            "rag_refreshed": self.rag_refreshed,
            "completed_at": self.completed_at,
        }


class SlackStore:
    """SQLite-backed normalized cache for Slack objects and sync checkpoints."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS slack_channels (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    channel_type TEXT NOT NULL,
                    is_archived INTEGER NOT NULL DEFAULT 0,
                    is_member INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS slack_users (
                    id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    real_name TEXT NOT NULL,
                    is_bot INTEGER NOT NULL DEFAULT 0,
                    deleted INTEGER NOT NULL DEFAULT 0,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS slack_messages (
                    channel_id TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    thread_ts TEXT,
                    user_id TEXT,
                    text TEXT NOT NULL,
                    subtype TEXT,
                    edited_ts TEXT,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    ingested_at TEXT NOT NULL,
                    PRIMARY KEY (channel_id, ts),
                    FOREIGN KEY (channel_id) REFERENCES slack_channels(id)
                );

                CREATE INDEX IF NOT EXISTS idx_slack_messages_ts
                    ON slack_messages(ts);
                CREATE INDEX IF NOT EXISTS idx_slack_messages_thread
                    ON slack_messages(channel_id, thread_ts);

                CREATE TABLE IF NOT EXISTS slack_sync_state (
                    channel_id TEXT PRIMARY KEY,
                    latest_ts TEXT NOT NULL,
                    synced_at TEXT NOT NULL,
                    FOREIGN KEY (channel_id) REFERENCES slack_channels(id)
                );

                CREATE TABLE IF NOT EXISTS slack_sync_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    status TEXT NOT NULL,
                    result_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )

    def upsert_channel(self, channel: dict[str, Any]) -> None:
        channel_type = "public_channel"
        if channel.get("is_im"):
            channel_type = "im"
        elif channel.get("is_mpim"):
            channel_type = "mpim"
        elif channel.get("is_private"):
            channel_type = "private_channel"

        name = str(channel.get("name") or channel.get("user") or channel["id"])
        metadata = {
            key: channel.get(key)
            for key in ("purpose", "topic", "user", "num_members")
            if channel.get(key) is not None
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO slack_channels (
                    id, name, channel_type, is_archived, is_member,
                    metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name = excluded.name,
                    channel_type = excluded.channel_type,
                    is_archived = excluded.is_archived,
                    is_member = excluded.is_member,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    channel["id"],
                    name,
                    channel_type,
                    int(bool(channel.get("is_archived"))),
                    int(bool(channel.get("is_member"))),
                    json.dumps(metadata, sort_keys=True),
                    _utc_now(),
                ),
            )

    def upsert_user(self, user: dict[str, Any]) -> None:
        profile = user.get("profile") or {}
        display_name = str(
            profile.get("display_name_normalized")
            or profile.get("display_name")
            or profile.get("real_name_normalized")
            or profile.get("real_name")
            or user.get("name")
            or user["id"]
        )
        real_name = str(
            profile.get("real_name_normalized")
            or profile.get("real_name")
            or display_name
        )
        metadata = {
            key: profile.get(key)
            for key in ("title", "team", "status_text", "status_emoji")
            if profile.get(key) is not None
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO slack_users (
                    id, display_name, real_name, is_bot, deleted,
                    metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    display_name = excluded.display_name,
                    real_name = excluded.real_name,
                    is_bot = excluded.is_bot,
                    deleted = excluded.deleted,
                    metadata_json = excluded.metadata_json,
                    updated_at = excluded.updated_at
                """,
                (
                    user["id"],
                    display_name,
                    real_name,
                    int(bool(user.get("is_bot"))),
                    int(bool(user.get("deleted"))),
                    json.dumps(metadata, sort_keys=True),
                    _utc_now(),
                ),
            )

    def upsert_message(self, channel_id: str, message: dict[str, Any]) -> bool:
        ts = str(message.get("ts") or "")
        if not ts:
            return False

        metadata = {
            key: message.get(key)
            for key in (
                "attachments",
                "blocks",
                "files",
                "reactions",
                "reply_count",
                "reply_users_count",
            )
            if message.get(key) is not None
        }
        edited = message.get("edited") or {}
        values = (
            channel_id,
            ts,
            message.get("thread_ts"),
            message.get("user") or message.get("bot_id"),
            str(message.get("text") or ""),
            message.get("subtype"),
            edited.get("ts"),
            json.dumps(metadata, sort_keys=True),
        )
        with self._connect() as connection:
            existing = connection.execute(
                """
                SELECT thread_ts, user_id, text, subtype, edited_ts, metadata_json
                FROM slack_messages WHERE channel_id = ? AND ts = ?
                """,
                (channel_id, ts),
            ).fetchone()
            comparable = values[2:]
            if existing is not None and tuple(existing) == comparable:
                return False
            connection.execute(
                """
                INSERT INTO slack_messages (
                    channel_id, ts, thread_ts, user_id, text, subtype,
                    edited_ts, metadata_json, ingested_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(channel_id, ts) DO UPDATE SET
                    thread_ts = excluded.thread_ts,
                    user_id = excluded.user_id,
                    text = excluded.text,
                    subtype = excluded.subtype,
                    edited_ts = excluded.edited_ts,
                    metadata_json = excluded.metadata_json,
                    ingested_at = excluded.ingested_at
                """,
                (*values, _utc_now()),
            )
        return True

    def checkpoint(self, channel_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT latest_ts FROM slack_sync_state WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
        return str(row["latest_ts"]) if row else None

    def set_checkpoint(self, channel_id: str, latest_ts: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO slack_sync_state(channel_id, latest_ts, synced_at)
                VALUES (?, ?, ?)
                ON CONFLICT(channel_id) DO UPDATE SET
                    latest_ts = excluded.latest_ts,
                    synced_at = excluded.synced_at
                """,
                (channel_id, latest_ts, _utc_now()),
            )

    def start_run(self) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                "INSERT INTO slack_sync_runs(started_at, status) VALUES (?, 'running')",
                (_utc_now(),),
            )
            return int(cursor.lastrowid)

    def finish_run(self, run_id: int, status: str, result: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE slack_sync_runs
                SET completed_at = ?, status = ?, result_json = ?
                WHERE id = ?
                """,
                (_utc_now(), status, json.dumps(result, sort_keys=True), run_id),
            )

    def chunks(self, oldest: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    m.channel_id, m.ts, m.thread_ts, m.text, m.subtype,
                    c.name AS channel_name, c.channel_type,
                    COALESCE(u.display_name, m.user_id, 'unknown') AS author
                FROM slack_messages m
                JOIN slack_channels c ON c.id = m.channel_id
                LEFT JOIN slack_users u ON u.id = m.user_id
                WHERE CAST(m.ts AS REAL) >= CAST(? AS REAL)
                  AND m.text != ''
                ORDER BY CAST(m.ts AS REAL) ASC
                """,
                (oldest,),
            ).fetchall()

        chunks: list[dict[str, Any]] = []
        for row in rows:
            timestamp = datetime.fromtimestamp(float(row["ts"]), tz=UTC).isoformat()
            location = f"#{row['channel_name']}" if row["channel_type"] != "im" else "direct message"
            chunks.append(
                {
                    "text": (
                        f"Slack {location} at {timestamp}\n"
                        f"{row['author']}: {row['text']}"
                    ),
                    "metadata": {
                        "source": "slack",
                        "channel_id": row["channel_id"],
                        "channel_name": row["channel_name"],
                        "channel_type": row["channel_type"],
                        "message_ts": row["ts"],
                        "thread_ts": row["thread_ts"] or "",
                        "author": row["author"],
                        "subtype": row["subtype"] or "",
                        "timestamp": timestamp,
                    },
                }
            )
        return chunks

    def stats(self) -> dict[str, Any]:
        with self._connect() as connection:
            counts = {
                table: int(
                    connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                )
                for table in ("slack_channels", "slack_users", "slack_messages")
            }
            latest = connection.execute(
                """
                SELECT completed_at, status, result_json
                FROM slack_sync_runs
                ORDER BY id DESC LIMIT 1
                """
            ).fetchone()
        return {
            "path": str(self.path),
            "channels": counts["slack_channels"],
            "users": counts["slack_users"],
            "messages": counts["slack_messages"],
            "last_sync": dict(latest) if latest else None,
        }


class SlackIngestor:
    """Sync Slack history into SQLite and the existing MAOS retriever."""

    def __init__(
        self,
        token: str,
        store: SlackStore,
        *,
        team_id: str = "",
        days: int = 90,
        max_messages: int = 25_000,
        include_dms: bool = True,
        channels: tuple[str, ...] = (),
        retriever: Any | None = None,
        client: Any | None = None,
    ) -> None:
        self.store = store
        self.team_id = team_id
        self.days = days
        self.max_messages = max_messages
        self.include_dms = include_dms
        self.channels = {value.strip().lstrip("#") for value in channels if value.strip()}
        self.retriever = retriever
        if client is not None:
            self.client = client
            return

        from slack_sdk import WebClient
        from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler

        self.client = WebClient(token=token)
        self.client.retry_handlers.append(RateLimitErrorRetryHandler(max_retry_count=3))

    def auth(self) -> dict[str, Any]:
        response = self.client.auth_test()
        return {
            "ok": bool(response.get("ok")),
            "team": response.get("team"),
            "team_id": response.get("team_id"),
            "user": response.get("user"),
            "user_id": response.get("user_id"),
        }

    def _list_users(self) -> list[dict[str, Any]]:
        users: list[dict[str, Any]] = []
        cursor = ""
        while True:
            response = self.client.users_list(
                team_id=self.team_id or None,
                limit=200,
                cursor=cursor or None,
            )
            users.extend(response.get("members") or [])
            cursor = _next_cursor(response)
            if not cursor:
                return users

    def _list_channels(self) -> list[dict[str, Any]]:
        channels: list[dict[str, Any]] = []
        cursor = ""
        while True:
            response = self.client.conversations_list(
                types="public_channel,private_channel,im,mpim",
                team_id=self.team_id or None,
                exclude_archived=True,
                limit=200,
                cursor=cursor or None,
            )
            channels.extend(response.get("channels") or [])
            cursor = _next_cursor(response)
            if not cursor:
                return channels

    def _selected(self, channel: dict[str, Any]) -> bool:
        if channel.get("is_archived"):
            return False
        if channel.get("is_im") or channel.get("is_mpim"):
            return self.include_dms and bool(channel.get("is_member"))
        if channel.get("is_private") and not channel.get("is_member"):
            return False
        if not self.channels:
            return True
        return channel.get("id") in self.channels or channel.get("name") in self.channels

    def _sync_channel(
        self,
        channel: dict[str, Any],
        *,
        full: bool,
        remaining: int,
    ) -> tuple[int, int, str | None]:
        channel_id = str(channel["id"])
        oldest = _oldest_timestamp(self.days)
        checkpoint = None if full else self.store.checkpoint(channel_id)
        if checkpoint and float(checkpoint) > float(oldest):
            oldest = checkpoint

        seen = 0
        changed = 0
        latest: str | None = checkpoint
        cursor = ""
        while seen < remaining:
            response = self.client.conversations_history(
                channel=channel_id,
                team_id=self.team_id or None,
                oldest=oldest,
                inclusive=full,
                limit=min(200, remaining - seen),
                cursor=cursor or None,
            )
            messages = response.get("messages") or []
            for message in messages:
                seen += 1
                changed += int(self.store.upsert_message(channel_id, message))
                message_ts = str(message.get("ts") or "")
                if message_ts and (latest is None or float(message_ts) > float(latest)):
                    latest = message_ts
                if message.get("reply_count") and seen < remaining:
                    thread_seen, thread_changed, thread_latest = self._sync_thread(
                        channel_id,
                        message_ts,
                        remaining - seen,
                    )
                    seen += thread_seen
                    changed += thread_changed
                    if thread_latest and (latest is None or float(thread_latest) > float(latest)):
                        latest = thread_latest
            cursor = _next_cursor(response)
            if not cursor or not messages:
                break
        return seen, changed, latest

    def _sync_thread(
        self, channel_id: str, thread_ts: str, remaining: int
    ) -> tuple[int, int, str | None]:
        if not thread_ts or remaining <= 0:
            return 0, 0, None
        seen = 0
        changed = 0
        latest: str | None = None
        cursor = ""
        while seen < remaining:
            response = self.client.conversations_replies(
                channel=channel_id,
                ts=thread_ts,
                team_id=self.team_id or None,
                limit=min(200, remaining - seen),
                cursor=cursor or None,
            )
            messages = response.get("messages") or []
            for message in messages:
                if str(message.get("ts")) == thread_ts:
                    continue
                seen += 1
                changed += int(self.store.upsert_message(channel_id, message))
                message_ts = str(message.get("ts") or "")
                if message_ts and (latest is None or float(message_ts) > float(latest)):
                    latest = message_ts
            cursor = _next_cursor(response)
            if not cursor or not messages:
                break
        return seen, changed, latest

    def sync(self, *, full: bool = False) -> SlackSyncResult:
        run_id = self.store.start_run()
        errors: list[str] = []
        messages_seen = 0
        messages_changed = 0
        channels_synced = 0
        users_seen = 0
        channels: list[dict[str, Any]] = []
        rag_refreshed = False

        try:
            self.auth()
            try:
                users = self._list_users()
                for user in users:
                    self.store.upsert_user(user)
                users_seen = len(users)
            except Exception as exc:
                errors.append(f"users: {exc}")

            channels = self._list_channels()
            for channel in channels:
                self.store.upsert_channel(channel)

            for channel in channels:
                if not self._selected(channel) or messages_seen >= self.max_messages:
                    continue
                try:
                    seen, changed, latest = self._sync_channel(
                        channel,
                        full=full,
                        remaining=self.max_messages - messages_seen,
                    )
                    messages_seen += seen
                    messages_changed += changed
                    channels_synced += 1
                    if latest:
                        self.store.set_checkpoint(str(channel["id"]), latest)
                except Exception as exc:
                    label = channel.get("name") or channel.get("id")
                    errors.append(f"{label}: {exc}")

            if self.retriever is not None:
                self.retriever.update("slack", self.store.chunks(_oldest_timestamp(self.days)))
                rag_refreshed = True

            result = SlackSyncResult(
                channels_seen=len(channels),
                channels_synced=channels_synced,
                messages_seen=messages_seen,
                messages_changed=messages_changed,
                users_seen=users_seen,
                errors=tuple(errors),
                rag_refreshed=rag_refreshed,
                completed_at=_utc_now(),
            )
            self.store.finish_run(run_id, "partial" if errors else "ok", result.as_dict())
            return result
        except Exception as exc:
            self.store.finish_run(run_id, "error", {"error": str(exc)})
            raise
