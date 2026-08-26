"""Safe MAOS integration with the LANBlockchain archive.

The adapter treats the LAN vault as a durable secondary store, never as the
source of truth for live MAOS state. Local writes complete first; archiving is
idempotent, time-bounded, and records a receipt only after a remote hash check.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.config.settings import LanVaultConfig


@dataclass(frozen=True)
class LanVaultReceipt:
    status: str
    archive_path: str
    content_hash: str
    remote_hash: str = ""
    detail: str = ""

    @property
    def verified(self) -> bool:
        return self.status == "verified" and self.content_hash == self.remote_hash


class LanVaultClient:
    """Bounded subprocess client for the local LANBlockchain CLI."""

    def __init__(self, config: LanVaultConfig, *, base_dir: Path) -> None:
        self.config = config
        self.base_dir = base_dir
        outbox = Path(config.outbox_dir).expanduser()
        if not outbox.is_absolute():
            outbox = base_dir / outbox
        self.outbox = outbox
        self.archive_root = config.archive_root.strip("/\\") or "personal"

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled and self.binary)

    @property
    def binary(self) -> str:
        configured = self.config.binary.strip()
        if os.path.sep in configured or configured.startswith("."):
            path = Path(configured).expanduser()
            if not path.is_absolute():
                path = self.base_dir / path
            return str(path) if path.exists() else ""
        return shutil.which(configured) or ""

    def _run(self, *args: str, timeout: int | None = None) -> subprocess.CompletedProcess[str]:
        if not self.binary:
            raise FileNotFoundError("lan-vault CLI is not installed or not on PATH")
        return subprocess.run(
            [self.binary, *args],
            capture_output=True,
            text=True,
            timeout=timeout or self.config.timeout_seconds,
            check=False,
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def archive_file(self, path: Path, *, archive_name: str) -> LanVaultReceipt:
        """Push one file via the audited CLI and verify its remote hash.

        The file is copied into a persistent outbox directory first. This
        makes a transient LAN outage retryable without losing the local note.
        ``lan-vault push`` archives the containing directory, so the remote
        path is ``<archive_root>/<archive_name>``.
        """
        if not self.enabled:
            return LanVaultReceipt("disabled", "", self._sha256(path), detail="LAN vault disabled")
        if not path.is_file():
            return LanVaultReceipt("failed", "", "", detail=f"not a file: {path}")

        target_dir = self.outbox / self.archive_root
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / archive_name
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=target_dir, prefix=f".{archive_name}.", delete=False
        ) as temp:
            temp_path = Path(temp.name)
            with path.open("rb") as source:
                shutil.copyfileobj(source, temp)
        os.replace(temp_path, target)
        content_hash = self._sha256(target)
        archive_path = f"{self.archive_root}/{archive_name}"

        try:
            pushed = self._run(str(target_dir))
        except (OSError, subprocess.TimeoutExpired) as exc:
            return LanVaultReceipt("failed", archive_path, content_hash, detail=str(exc))
        if pushed.returncode != 0:
            detail = (pushed.stderr or pushed.stdout or "lan-vault push failed").strip()
            return LanVaultReceipt("failed", archive_path, content_hash, detail=detail[-1000:])

        if not self.config.verify_after_push:
            return LanVaultReceipt("submitted", archive_path, content_hash, detail="push completed")

        try:
            verified = self._run("verify", archive_path, "--json")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return LanVaultReceipt("failed", archive_path, content_hash, detail=f"verify: {exc}")
        if verified.returncode != 0:
            detail = (verified.stderr or verified.stdout or "remote verification failed").strip()
            return LanVaultReceipt("failed", archive_path, content_hash, detail=detail[-1000:])
        try:
            payload: dict[str, Any] = json.loads(verified.stdout)
        except json.JSONDecodeError:
            return LanVaultReceipt("failed", archive_path, content_hash, detail="verify returned invalid JSON")
        remote_hash = str(payload.get("hash") or "")
        if remote_hash != content_hash:
            return LanVaultReceipt(
                "failed", archive_path, content_hash, remote_hash,
                detail="remote hash does not match local hash",
            )
        return LanVaultReceipt("verified", archive_path, content_hash, remote_hash, "remote hash verified")

    def status(self) -> dict[str, Any]:
        """Return local chain status plus bounded remote health information."""
        if not self.binary:
            return {"ok": False, "error": "lan-vault CLI is not installed or not on PATH"}
        try:
            result = self._run("status", timeout=min(self.config.timeout_seconds, 45))
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc)}
        output = (result.stdout or "").strip()
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "output": output[-4000:],
            "error": (result.stderr or "").strip()[-1000:],
        }

    def reconcile(self, path: str | None = None) -> dict[str, Any]:
        args = ["reconcile"]
        if path:
            args.extend(["--path", path])
        try:
            result = self._run(*args, timeout=self.config.timeout_seconds)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"ok": False, "error": str(exc)}
        return {
            "ok": result.returncode == 0,
            "exit_code": result.returncode,
            "output": (result.stdout or "")[-4000:],
            "error": (result.stderr or "")[-1000:],
        }
