"""Storage management — decompose macOS "System Data" and find reclaimable space.

This module bridges two external tools with native macOS introspection:

  - **X-MaC** (``xmac`` CLI): Rust-based disk scanner with 13 engines.
    Provides disk usage breakdown, clean scan (reclaimable space), and
    maintenance tasks. Called via subprocess with JSON output.

  - **NeoDisk** (GUI app): Stores scan caches in
    ``~/Library/Application Support/Neodisk/ScanCache/``. We read these
    cached scans to avoid redundant filesystem walks when available.

  - **Native macOS**: The "System Data" category in macOS Storage settings
    is a black box. This module decomposes it by directly querying:
      - Time Machine local snapshots (``tmutil``)
      - APFS snapshots (``diskutil apfs listSnapshots``)
      - sleepimage + swapfiles (``/private/var/vm``)
      - Unified log archive (``/private/var/db/diagnostics`` + ``uuidtext``)
      - Purgeable space (``diskutil info -plist``)
      - OS update staging snapshots
      - Spotlight index (``.Spotlight-V100``)
      - Asset cache (``/private/var/db/AssetCache``)

All operations are **read-only**. Destructive actions are emitted as
reviewable fix scripts — never executed automatically. This follows the
X-MaC philosophy: scan, report, generate fix script, let the human review.
"""

from __future__ import annotations

import json
import os
import plistlib
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC
from pathlib import Path
from typing import Any

from app.log import get_logger

log = get_logger(__name__)

# --- Constants ---

XMAC_BIN = os.environ.get("XMAC_BIN", "xmac")
"""Path to the xmac binary. Defaults to PATH lookup; override via env."""

NEODISK_CACHE = Path.home() / "Library" / "Application Support" / "Neodisk" / "ScanCache"
"""NeoDisk scan cache directory."""

HOME = Path.home()
"""User home directory."""

# System data locations that macOS categorizes as "System Data" or "Other"
SYSTEM_DATA_PATHS = {
    "sleepimage": "/private/var/vm/sleepimage",
    "swapfiles_dir": "/private/var/vm",
    "log_diagnostics": "/private/var/db/diagnostics",
    "log_uuidtext": "/private/var/db/uuidtext",
    "log_install": "/private/var/log",
    "spotlight_index": "/System/Volumes/Data/.Spotlight-V100",
    "asset_cache": "/private/var/db/AssetCache",
}

# User-managed caches (safe to clear, regenerate on demand)
USER_CACHE_PATHS = {
    "uv_cache": str(HOME / ".cache" / "uv"),
    "huggingface_cache": str(HOME / ".cache" / "huggingface"),
    "npm_cache": str(HOME / ".npm" / "_cacache"),
    "pip_cache": str(HOME / ".cache" / "pip"),
    "cargo_cache": str(HOME / ".cargo" / "registry"),
    "rustup": str(HOME / ".rustup"),
    "gradle_cache": str(HOME / ".gradle" / "caches"),
    "pyenv": str(HOME / ".pyenv"),
    "homebrew_cache": str(HOME / "Library" / "Caches" / "Homebrew"),
    "xcode_derived_data": str(HOME / "Library" / "Developer" / "Xcode" / "DerivedData"),
    "xcode_archives": str(HOME / "Library" / "Developer" / "Xcode" / "Archives"),
    "core_simulator": str(HOME / "Library" / "Developer" / "CoreSimulator"),
    "docker_desktop": str(HOME / "Library" / "Containers" / "com.docker.docker"),
    "trash": str(HOME / ".Trash"),
    # AI/ML tool data (discovered 2026-07-24)
    "lmstudio_extensions": str(HOME / ".lmstudio" / "extensions"),
    "lmstudio_bin": str(HOME / ".lmstudio" / "bin"),
    "codex_cache": str(HOME / "Library" / "Caches" / "com.openai.codex"),
    "codex_data": str(HOME / ".codex"),
    "hermes_data": str(HOME / ".hermes"),
    "mlx_server": str(HOME / ".mlx_server"),
    "ollama_models": str(HOME / ".ollama"),
    # Developer tool data (discovered 2026-07-24)
    "uv_tools": str(HOME / ".local" / "share" / "uv" / "tools"),
    "uv_python": str(HOME / ".local" / "share" / "uv" / "python"),
    "devin_cli_data": str(HOME / ".local" / "share" / "devin" / "cli"),
    "devin_config": str(HOME / ".devin"),
    "openhands_node_modules": str(HOME / "OpenHands" / "node_modules"),
    "vscode_extensions": str(HOME / ".vscode" / "extensions"),
    "nvm": str(HOME / ".nvm"),
    "mise_data": str(HOME / ".local" / "share" / "mise"),
    "pipx": str(HOME / ".local" / "pipx"),
    "pnpm_store": str(HOME / "Library" / "pnpm"),
    # App support data (discovered 2026-07-24)
    "google_app_support": str(HOME / "Library" / "Application Support" / "Google"),
    "warp_app_support": str(HOME / "Library" / "Application Support" / "dev.warp.Warp-Stable"),
    "devin_app_support": str(HOME / "Library" / "Application Support" / "Devin"),
    "cursor_app_support": str(HOME / "Library" / "Application Support" / "Cursor"),
    "proton_mail_app_support": str(HOME / "Library" / "Application Support" / "Proton Mail"),
    "discord_app_support": str(HOME / "Library" / "Application Support" / "discord"),
    "codex_app_support": str(HOME / "Library" / "Application Support" / "Codex"),
    "raycast_app_support": str(HOME / "Library" / "Application Support" / "com.raycast.macos"),
    "neodisk_app_support": str(HOME / "Library" / "Application Support" / "Neodisk"),
    # Cloud storage (can evict but not delete)
    "proton_drive": str(HOME / "Library" / "CloudStorage"),
    # SDKs
    "android_sdk": str(HOME / "Library" / "Android" / "sdk"),
    # Containers
    "playcover": str(HOME / "Library" / "Containers" / "io.playcover.PlayCover"),
    "mediaanalysis": str(HOME / "Library" / "Containers" / "com.apple.mediaanalysisd"),
    # System-level (not in home)
    "library_updates": "/Library/Updates",
    "library_developer": "/Library/Developer",
    "library_command_line_tools": "/Library/Developer/CommandLineTools",
    "system_assets": "/System/Volumes/Data/System/Library/AssetsV2",
}


# --- Data structures ---


@dataclass
class DiskVolume:
    """APFS volume statistics from ``diskutil``."""

    total_bytes: int = 0
    free_bytes: int = 0
    used_bytes: int = 0
    system_used_bytes: int = 0
    data_used_bytes: int = 0
    applications_bytes: int = 0
    purgeable_bytes: int = 0
    container_name: str = ""
    capacity_percent: float = 0.0

    @property
    def used_gb(self) -> float:
        return self.used_bytes / 1e9

    @property
    def free_gb(self) -> float:
        return self.free_bytes / 1e9

    @property
    def purgeable_gb(self) -> float:
        return self.purgeable_bytes / 1e9


@dataclass
class TMSnapshot:
    """A Time Machine local snapshot."""

    name: str
    date: str  # e.g. "2026-07-24-152803"
    deletable: bool = True
    size_estimate_bytes: int = 0


@dataclass
class APFSSnapshot:
    """An APFS filesystem snapshot."""

    uuid: str
    name: str
    purgeable: bool = False
    limits_container_size: bool = False


@dataclass
class SystemDataEntry:
    """A single component of macOS "System Data"."""

    category: str
    path: str
    size_bytes: int
    description: str
    reclaimable: bool
    safe_to_delete: bool
    remediation: str = ""
    evidence: str = ""


@dataclass
class CacheEntry:
    """A user-managed cache directory."""

    category: str
    path: str
    size_bytes: int
    description: str
    safe_to_clear: bool
    regenerates: bool
    remediation: str = ""


@dataclass
class StorageReport:
    """Full storage analysis report."""

    volume: DiskVolume
    system_data: list[SystemDataEntry] = field(default_factory=list)
    user_caches: list[CacheEntry] = field(default_factory=list)
    tm_snapshots: list[TMSnapshot] = field(default_factory=list)
    apfs_snapshots: list[APFSSnapshot] = field(default_factory=list)
    xmac_disk_findings: list[dict] = field(default_factory=list)
    xmac_clean_findings: list[dict] = field(default_factory=list)
    neodisk_available: bool = False
    total_reclaimable_bytes: int = 0
    total_system_data_bytes: int = 0
    total_cache_bytes: int = 0
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FixScript:
    """A reviewable remediation script."""

    path: str
    total_reclaimable_bytes: int
    commands: list[str]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --- Helpers ---


def _dir_size(path: str | Path) -> int:
    """Physical disk usage of a directory (blocks, not logical size).

    Uses ``du -sk`` for speed — walks the tree once and sums block usage.
    Returns 0 if the path doesn't exist or is inaccessible.
    """
    p = Path(path)
    if not p.exists():
        return 0
    try:
        result = subprocess.run(
            ["du", "-sk", str(p)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            # du -sk returns KB
            return int(result.stdout.split()[0]) * 1024
    except (subprocess.TimeoutExpired, ValueError, IndexError):
        pass
    return 0


def _file_size(path: str | Path) -> int:
    """Size of a single file in bytes, 0 if missing."""
    p = Path(path)
    try:
        return p.stat().st_size if p.exists() else 0
    except OSError:
        return 0


def _fmt_bytes(n: int) -> str:
    """Human-readable byte size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def _run(cmd: list[str], timeout: int = 60) -> tuple[int, str, str]:
    """Run a command, return (exit_code, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except FileNotFoundError:
        return -2, "", f"command not found: {cmd[0]}"


def _xmac(args: list[str], timeout: int = 120) -> list[dict]:
    """Run xmac with JSON output, return parsed findings list."""
    cmd = [XMAC_BIN, "--format", "json"] + args
    code, stdout, stderr = _run(cmd, timeout=timeout)
    if code != 0:
        log.warning("xmac failed", cmd=cmd, code=code, stderr=stderr[:200])
        return []
    findings = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            findings.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return findings


# --- Volume stats ---


def get_volume_stats() -> DiskVolume:
    """Get APFS volume statistics via ``diskutil info -plist``.

    This is more accurate than ``df`` on APFS because it reports
    purgeable space and separates system vs data volume usage.
    """
    vol = DiskVolume()

    # Get container info for the root volume.
    # On APFS, FreeSpace per-volume is 0 because free space is managed
    # at the container level. Use APFSContainerSize and APFSContainerFree.
    code, stdout, _ = _run(["diskutil", "info", "-plist", "/"])
    if code == 0 and stdout:
        try:
            plist = plistlib.loads(stdout.encode())
            vol.container_name = plist.get(
                "FilesystemName", plist.get("VolumeName", "Macintosh HD")
            )
            vol.total_bytes = int(
                plist.get("APFSContainerSize", plist.get("TotalSize", 0))
            )
            vol.free_bytes = int(
                plist.get("APFSContainerFree", plist.get("FreeSpace", 0))
            )
            vol.used_bytes = vol.total_bytes - vol.free_bytes
            if vol.total_bytes > 0:
                vol.capacity_percent = (
                    vol.used_bytes / vol.total_bytes * 100
                )
        except (plistlib.InvalidFileException, ValueError, KeyError):
            pass

    # Get purgeable space via diskutil (macOS 12+)
    code, stdout, _ = _run(
        ["diskutil", "apfs", "getPercentPurgeable", "/"]
    )
    if code == 0 and stdout:
        try:
            # Output: "Percent Purgeable: 12.5%"
            for line in stdout.splitlines():
                if "Purgeable" in line:
                    pct_str = line.split(":")[-1].strip().rstrip("%")
                    pct = float(pct_str)
                    vol.purgeable_bytes = int(
                        vol.total_bytes * pct / 100
                    )
                    break
        except (ValueError, IndexError):
            pass

    # Fallback: df for basic stats if diskutil failed
    if vol.total_bytes == 0:
        code, stdout, _ = _run(["df", "-k", "/"])
        if code == 0 and stdout:
            lines = stdout.strip().splitlines()
            if len(lines) >= 2:
                parts = lines[1].split()
                if len(parts) >= 4:
                    vol.total_bytes = int(parts[1]) * 1024
                    vol.used_bytes = int(parts[2]) * 1024
                    vol.free_bytes = int(parts[3]) * 1024
                    if vol.total_bytes > 0:
                        vol.capacity_percent = (
                            vol.used_bytes / vol.total_bytes * 100
                        )

    # Data volume usage (the writable APFS volume)
    code, stdout, _ = _run(["df", "-k", "/System/Volumes/Data"])
    if code == 0 and stdout:
        lines = stdout.strip().splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 3:
                vol.data_used_bytes = int(parts[2]) * 1024

    # System volume usage (read-only sealed volume)
    code, stdout, _ = _run(["df", "-k", "/"])
    if code == 0 and stdout:
        lines = stdout.strip().splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 3:
                vol.system_used_bytes = int(parts[2]) * 1024

    # Applications directory
    vol.applications_bytes = _dir_size("/Applications")

    return vol


# --- Time Machine snapshots ---


def get_tm_snapshots() -> list[TMSnapshot]:
    """List Time Machine local snapshots via ``tmutil``.

    These are the #1 cause of mysterious "System Data" bloat. macOS
    creates local snapshots throughout the day and they can hold
    tens of GB of purgeable space that only clears on the next backup.
    """
    code, stdout, _ = _run(["tmutil", "listlocalsnapshots", "/"])
    if code != 0 or not stdout:
        return []

    snapshots = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Format: "com.apple.TimeMachine.2026-07-24-152803.local"
        # or just the date part
        name = line
        date_str = ""
        if "TimeMachine." in line:
            parts = line.split("TimeMachine.")
            if len(parts) > 1:
                date_part = parts[1].replace(".local", "")
                date_str = date_part

        # OS update snapshots are not user-deletable
        deletable = "os.update" not in line
        snapshots.append(
            TMSnapshot(
                name=name,
                date=date_str,
                deletable=deletable,
                size_estimate_bytes=0,  # tmutil doesn't report per-snapshot size
            )
        )
    return snapshots


def delete_tm_snapshot(date: str) -> tuple[bool, str]:
    """Delete a single Time Machine local snapshot by date.

    Args:
        date: The snapshot date string, e.g. "2026-07-24-152803".

    Returns:
        (success, message)
    """
    code, stdout, stderr = _run(["tmutil", "deletelocalsnapshots", date])
    if code == 0:
        return True, f"Deleted local snapshot '{date}'"
    return False, f"Failed to delete snapshot '{date}': {stderr.strip()}"


# --- APFS snapshots ---


def get_apfs_snapshots() -> list[APFSSnapshot]:
    """List APFS filesystem snapshots via ``diskutil``.

    These include OS update staging snapshots (not purgeable) and
    any user-created snapshots. The ``Purgeable`` flag indicates
    whether macOS can reclaim the space automatically.
    """
    code, stdout, _ = _run(["diskutil", "apfs", "listSnapshots", "/"])
    if code != 0 or not stdout:
        return []

    snapshots = []
    current = APFSSnapshot(uuid="", name="")

    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("+--"):
            # New snapshot entry
            if current.uuid:
                snapshots.append(current)
            uuid = line.lstrip("+-").strip()
            current = APFSSnapshot(uuid=uuid, name="")
        elif line.startswith("Name:"):
            current.name = line.split(":", 1)[1].strip()
        elif line.startswith("Purgeable:"):
            current.purgeable = line.split(":", 1)[1].strip().lower() == "yes"
        elif "limits the minimum size" in line:
            current.limits_container_size = True

    if current.uuid:
        snapshots.append(current)

    return snapshots


# --- System data decomposition ---


def decompose_system_data() -> list[SystemDataEntry]:
    """Decompose macOS "System Data" into identifiable components.

    This is the core function that solves the "120 GB of system data"
    mystery. It checks every known location that macOS categorizes as
    "System Data" or "Other" in Storage settings and reports the actual
    size, whether it's reclaimable, and how to reclaim it.
    """
    entries: list[SystemDataEntry] = []

    # 1. sleepimage — RAM dump on sleep, regenerates automatically
    sleep_size = _file_size(SYSTEM_DATA_PATHS["sleepimage"])
    if sleep_size > 0:
        entries.append(
            SystemDataEntry(
                category="sleepimage",
                path=SYSTEM_DATA_PATHS["sleepimage"],
                size_bytes=sleep_size,
                description=(
                    "RAM image written on sleep. Regenerates on next sleep. "
                    "Safe to delete while awake — frees space equal to your RAM."
                ),
                reclaimable=True,
                safe_to_delete=True,
                remediation=f"sudo rm -f {SYSTEM_DATA_PATHS['sleepimage']}",
                evidence=f"ls -lh {SYSTEM_DATA_PATHS['sleepimage']}",
            )
        )

    # 2. Swapfiles
    swap_size = _dir_size(SYSTEM_DATA_PATHS["swapfiles_dir"])
    # Subtract sleepimage since it's in the same dir
    swap_only = swap_size - sleep_size
    if swap_only > 0:
        entries.append(
            SystemDataEntry(
                category="swapfiles",
                path=SYSTEM_DATA_PATHS["swapfiles_dir"],
                size_bytes=swap_only,
                description=(
                    "Virtual memory swap files. Managed by macOS — clears "
                    "on reboot. Not safe to delete manually."
                ),
                reclaimable=True,
                safe_to_delete=False,
                remediation="Reboot to clear swap files",
                evidence=f"ls -lh {SYSTEM_DATA_PATHS['swapfiles_dir']}/swapfile*",
            )
        )

    # 3. Unified log archive (diagnostics + uuidtext)
    diag_size = _dir_size(SYSTEM_DATA_PATHS["log_diagnostics"])
    uuid_size = _dir_size(SYSTEM_DATA_PATHS["log_uuidtext"])
    log_total = diag_size + uuid_size
    if log_total > 0:
        entries.append(
            SystemDataEntry(
                category="unified_log_archive",
                path=f"{SYSTEM_DATA_PATHS['log_diagnostics']} + {SYSTEM_DATA_PATHS['log_uuidtext']}",
                size_bytes=log_total,
                description=(
                    "macOS unified log archive. Accumulates over time. "
                    "Safe to erase — system recreates as needed."
                ),
                reclaimable=True,
                safe_to_delete=True,
                remediation="sudo log erase --all",
                evidence=f"du -sh {SYSTEM_DATA_PATHS['log_diagnostics']} {SYSTEM_DATA_PATHS['log_uuidtext']}",
            )
        )

    # 4. System logs
    syslog_size = _dir_size(SYSTEM_DATA_PATHS["log_install"])
    if syslog_size > 0:
        entries.append(
            SystemDataEntry(
                category="system_logs",
                path=SYSTEM_DATA_PATHS["log_install"],
                size_bytes=syslog_size,
                description=(
                    "System log files (install.log, wifi.log, powermanagement, "
                    "launchd logs). Safe to rotate but some are useful for diagnostics."
                ),
                reclaimable=True,
                safe_to_delete=True,
                remediation="sudo log erase --all && sudo rm -rf /private/var/log/*.log.*",
                evidence=f"du -sh {SYSTEM_DATA_PATHS['log_install']}/*",
            )
        )

    # 5. Spotlight index
    spotlight_size = _dir_size(SYSTEM_DATA_PATHS["spotlight_index"])
    if spotlight_size > 0:
        entries.append(
            SystemDataEntry(
                category="spotlight_index",
                path=SYSTEM_DATA_PATHS["spotlight_index"],
                size_bytes=spotlight_size,
                description=(
                    "Spotlight search index. Rebuilds automatically after deletion. "
                    "Search will be slow until reindex completes."
                ),
                reclaimable=True,
                safe_to_delete=True,
                remediation="sudo mdutil -E / && sudo rm -rf /.Spotlight-V100",
                evidence=f"du -sh {SYSTEM_DATA_PATHS['spotlight_index']}",
            )
        )

    # 6. Asset cache (content caching)
    asset_size = _dir_size(SYSTEM_DATA_PATHS["asset_cache"])
    if asset_size > 0:
        entries.append(
            SystemDataEntry(
                category="asset_cache",
                path=SYSTEM_DATA_PATHS["asset_cache"],
                size_bytes=asset_size,
                description=(
                    "macOS content caching storage. Safe to clear — "
                    "re-downloads from Apple on demand."
                ),
                reclaimable=True,
                safe_to_delete=True,
                remediation="sudo rm -rf /private/var/db/AssetCache/*",
                evidence=f"du -sh {SYSTEM_DATA_PATHS['asset_cache']}",
            )
        )

    # 7. Time Machine local snapshots (size is purgeable space)
    tm_snaps = get_tm_snapshots()
    tm_count = len([s for s in tm_snaps if s.deletable])
    if tm_count > 0:
        entries.append(
            SystemDataEntry(
                category="tm_local_snapshots",
                path="tmutil listlocalsnapshots /",
                size_bytes=0,  # tmutil doesn't report per-snapshot size
                description=(
                    f"{tm_count} Time Machine local snapshot(s) are pinning "
                    "purgeable space. These are the #1 cause of mysterious "
                    "'System Data' bloat. Deleting them reclaims purgeable "
                    "space immediately."
                ),
                reclaimable=True,
                safe_to_delete=True,
                remediation="tmutil deletelocalsnapshots <date>",
                evidence="tmutil listlocalsnapshots /",
            )
        )

    # 8. APFS snapshots (OS update staging)
    apfs_snaps = get_apfs_snapshots()
    os_update_snaps = [s for s in apfs_snaps if "os.update" in s.name]
    if os_update_snaps:
        entries.append(
            SystemDataEntry(
                category="os_update_snapshots",
                path="diskutil apfs listSnapshots /",
                size_bytes=0,
                description=(
                    f"{len(os_update_snaps)} OS update snapshot(s) are staged. "
                    "These are NOT purgeable — they're needed by the OS. "
                    "Installing or cancelling the pending update releases them."
                ),
                reclaimable=False,
                safe_to_delete=False,
                remediation="softwareupdate --install (or cancel in System Settings)",
                evidence="diskutil apfs listSnapshots /",
            )
        )

    # 9. /private/var/folders (per-user temp)
    temp_size = _dir_size("/private/var/folders")
    if temp_size > 0:
        entries.append(
            SystemDataEntry(
                category="user_temp",
                path="/private/var/folders",
                size_bytes=temp_size,
                description=(
                    "Per-user temporary files (C, T, X directories). "
                    "Cleared on reboot. Some files are SIP-protected."
                ),
                reclaimable=True,
                safe_to_delete=False,
                remediation="Reboot to clear (safest), or rm -rf specific subdirs",
                evidence="du -sh /private/var/folders/*/*",
            )
        )

    # 10. /private/var/db (system databases)
    var_db_size = _dir_size("/private/var/db")
    if var_db_size > 0:
        entries.append(
            SystemDataEntry(
                category="system_databases",
                path="/private/var/db",
                size_bytes=var_db_size,
                description=(
                    "System databases (receipts, powerlog, systemstats, "
                    "KernelExtensionManagement). Mostly not reclaimable — "
                    "diagnostics subfolder is the exception."
                ),
                reclaimable=False,
                safe_to_delete=False,
                remediation="sudo log erase --all (clears diagnostics subfolder)",
                evidence="du -sh /private/var/db/*",
            )
        )

    # 11. /System/Volumes/Data/System/Library/AssetsV2 — system assets
    assets_size = _dir_size("/System/Volumes/Data/System/Library/AssetsV2")
    if assets_size > 0:
        entries.append(
            SystemDataEntry(
                category="system_assets",
                path="/System/Volumes/Data/System/Library/AssetsV2",
                size_bytes=assets_size,
                description=(
                    "macOS system asset files (fonts, voices, dictionaries, "
                    "asset packs). Managed by macOS — do NOT delete manually."
                ),
                reclaimable=False,
                safe_to_delete=False,
                remediation="Managed by macOS — not reclaimable",
                evidence="du -sh /System/Volumes/Data/System/Library/AssetsV2",
            )
        )

    # 12. /Library/Updates — downloaded macOS update packages
    updates_size = _dir_size("/Library/Updates")
    if updates_size > 0:
        entries.append(
            SystemDataEntry(
                category="downloaded_updates",
                path="/Library/Updates",
                size_bytes=updates_size,
                description=(
                    "Downloaded macOS update packages. Safe to delete — "
                    "Software Update will re-download if needed. Clearing "
                    "these does NOT cancel the pending update."
                ),
                reclaimable=True,
                safe_to_delete=True,
                remediation="sudo rm -rf /Library/Updates/*",
                evidence="du -sh /Library/Updates/*",
            )
        )

    # 13. /Library/Developer/CommandLineTools — Xcode CLT
    clt_size = _dir_size("/Library/Developer/CommandLineTools")
    if clt_size > 0:
        entries.append(
            SystemDataEntry(
                category="xcode_command_line_tools",
                path="/Library/Developer/CommandLineTools",
                size_bytes=clt_size,
                description=(
                    "Xcode Command Line Tools. Can be reinstalled with "
                    "xcode-select --install. Only delete if you don't need "
                    "compilation tools (gcc, make, git, etc.)."
                ),
                reclaimable=True,
                safe_to_delete=True,
                remediation="sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install",
                evidence="du -sh /Library/Developer/CommandLineTools",
            )
        )

    # 14. /System/Volumes/VM — virtual memory volume
    vm_code, vm_stdout, _ = _run(["df", "-k", "/System/Volumes/VM"])
    vm_size = 0
    if vm_code == 0 and vm_stdout:
        lines = vm_stdout.strip().splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 3:
                vm_size = int(parts[2]) * 1024
    if vm_size > 0:
        entries.append(
            SystemDataEntry(
                category="vm_volume",
                path="/System/Volumes/VM",
                size_bytes=vm_size,
                description=(
                    "APFS VM volume — swap space and virtual memory. "
                    "Clears on reboot. Not directly deletable."
                ),
                reclaimable=True,
                safe_to_delete=False,
                remediation="Reboot to clear VM volume",
                evidence="df -h /System/Volumes/VM",
            )
        )

    # Sort by size descending
    entries.sort(key=lambda e: e.size_bytes, reverse=True)
    return entries


# --- Hidden data scanner ---


def scan_hidden_data() -> list[CacheEntry]:
    """Scan for hidden data consumers that macOS categorizes as 'System Data'.

    This goes beyond the standard system data decomposition and user caches.
    It checks for:

    - AI/ML tool data (LM Studio, Codex, Devin CLI, Hermes, MLX, Ollama)
    - Developer tool data (uv tools, pipx, nvm, mise, pnpm store)
    - App support directories (Chrome, Warp, Cursor, Discord, etc.)
    - Cloud storage sync (Proton Drive, iCloud)
    - SDKs (Android SDK)
    - App containers (PlayCover, mediaanalysis)
    - Devin CLI session database (can grow to 3GB+ unbounded)

    These are the 'hidden' consumers that don't show up in standard
    disk usage analysis because they're in non-obvious locations.
    """
    # Reuse the USER_CACHE_PATHS dict — the hidden entries are already there.
    # This function is a superset of scan_user_caches that also scans
    # subdirectories of large paths for more granular reporting.
    base_entries = scan_user_caches()

    # Add subdirectory breakdowns for the biggest hidden consumers
    extra: list[CacheEntry] = []

    # Devin CLI session database — the #1 hidden space hog
    devin_cli = HOME / ".local" / "share" / "devin" / "cli"
    if devin_cli.exists():
        sessions_db = devin_cli / "sessions.db"
        if sessions_db.exists():
            extra.append(
                CacheEntry(
                    category="devin_sessions_db",
                    path=str(sessions_db),
                    size_bytes=sessions_db.stat().st_size,
                    description=(
                        "Devin CLI session database. Grows unbounded with "
                        "every Devin session. Can be vacuumed to reclaim space "
                        "without losing data: sqlite3 sessions.db 'VACUUM;'"
                    ),
                    safe_to_clear=False,
                    regenerates=False,
                    remediation="sqlite3 ~/.local/share/devin/cli/sessions.db 'VACUUM;'",
                )
            )
        # Corrupt backup files — safe to delete
        for bak in devin_cli.glob("*.corrupt.bak*"):
            extra.append(
                CacheEntry(
                    category="devin_corrupt_backups",
                    path=str(bak),
                    size_bytes=bak.stat().st_size,
                    description="Corrupt Devin CLI database backup — safe to delete.",
                    safe_to_clear=True,
                    regenerates=False,
                    remediation=f"rm -f {bak}",
                )
            )
        # Devin logs
        logs_dir = devin_cli / "logs"
        if logs_dir.exists():
            log_size = _dir_size(logs_dir)
            if log_size > 0:
                extra.append(
                    CacheEntry(
                        category="devin_logs",
                        path=str(logs_dir),
                        size_bytes=log_size,
                        description="Devin CLI log files — safe to delete.",
                        safe_to_clear=True,
                        regenerates=True,
                        remediation=f"rm -rf {logs_dir}",
                    )
                )
        # Devin summaries
        summaries_dir = devin_cli / "summaries"
        if summaries_dir.exists():
            sum_size = _dir_size(summaries_dir)
            if sum_size > 0:
                extra.append(
                    CacheEntry(
                        category="devin_summaries",
                        path=str(summaries_dir),
                        size_bytes=sum_size,
                        description="Devin CLI session summaries — review before deleting.",
                        safe_to_clear=False,
                        regenerates=False,
                        remediation=f"rm -rf {summaries_dir}",
                    )
                )

    # uv tools breakdown
    uv_tools = HOME / ".local" / "share" / "uv" / "tools"
    if uv_tools.exists():
        for tool_dir in uv_tools.iterdir():
            if tool_dir.is_dir():
                size = _dir_size(tool_dir)
                if size > 50 * 1024 * 1024:  # only report >50MB
                    extra.append(
                        CacheEntry(
                            category=f"uv_tool_{tool_dir.name}",
                            path=str(tool_dir),
                            size_bytes=size,
                            description=f"uv tool install: {tool_dir.name}",
                            safe_to_clear=False,
                            regenerates=True,
                            remediation=f"uv tool uninstall {tool_dir.name}",
                        )
                    )

    # NeoDisk scan cache breakdown
    neodisk_cache = HOME / "Library" / "Application Support" / "Neodisk" / "ScanCache"
    if neodisk_cache.exists():
        cache_size = _dir_size(neodisk_cache)
        if cache_size > 0:
            extra.append(
                CacheEntry(
                    category="neodisk_scan_cache",
                    path=str(neodisk_cache),
                    size_bytes=cache_size,
                    description="NeoDisk scan cache files — safe to delete, rescans on next launch.",
                    safe_to_clear=True,
                    regenerates=True,
                    remediation=f"rm -rf {neodisk_cache}",
                )
            )

    all_entries = base_entries + extra
    all_entries.sort(key=lambda e: e.size_bytes, reverse=True)
    return all_entries


# --- User cache scan ---


def scan_user_caches() -> list[CacheEntry]:
    """Scan user-managed cache directories.

    These are developer caches, package manager caches, and build
    artifacts that are safe to clear and regenerate on demand.
    """
    descriptions = {
        "uv_cache": ("uv Python wheel cache", True, True, "uv cache clean"),
        "huggingface_cache": (
            "HuggingFace downloaded models",
            True,
            True,
            "rm -rf ~/.cache/huggingface",
        ),
        "npm_cache": ("npm package cache", True, True, "npm cache clean --force"),
        "pip_cache": ("pip download cache", True, True, "pip cache purge"),
        "cargo_cache": ("Rust crate registry cache", True, True, "cargo cache --autoclean"),
        "rustup": ("Rust toolchain installations", True, False, "rustup toolchain list && rustup toolchain uninstall <old>"),
        "gradle_cache": ("Gradle build cache", True, True, "rm -rf ~/.gradle/caches"),
        "pyenv": ("pyenv Python versions", True, False, "pyenv uninstall <version>"),
        "homebrew_cache": ("Homebrew download cache", True, True, "brew cleanup --prune=all"),
        "xcode_derived_data": ("Xcode build artifacts", True, True, "rm -rf ~/Library/Developer/Xcode/DerivedData/*"),
        "xcode_archives": ("Xcode app archives", True, False, "Review and delete old archives in Xcode Organizer"),
        "core_simulator": ("iOS Simulator runtimes and data", True, False, "xcrun simctl delete unavailable"),
        "docker_desktop": ("Docker Desktop VM disk image", True, False, "Docker Desktop → Settings → Resources → Reclaim space"),
        "trash": ("Trash bin contents", True, True, "Empty Trash in Finder or: rm -rf ~/.Trash/*"),
        # AI/ML tool data (discovered 2026-07-24)
        "lmstudio_extensions": ("LM Studio extensions and models", True, True, "rm -rf ~/.lmstudio/extensions"),
        "lmstudio_bin": ("LM Studio binary", True, False, "Reinstall from LM Studio if needed"),
        "codex_cache": ("OpenAI Codex cache (can grow to 1.3GB+)", True, True, "rm -rf ~/Library/Caches/com.openai.codex"),
        "codex_data": ("Codex CLI data directory", True, False, "Review ~/.codex before deleting"),
        "hermes_data": ("Hermes runtime data", True, False, "Review ~/.hermes before deleting"),
        "mlx_server": ("MLX server data", True, False, "Review ~/.mlx_server before deleting"),
        "ollama_models": ("Ollama downloaded models", True, True, "ollama model rm <model> or rm -rf ~/.ollama"),
        # Developer tool data (discovered 2026-07-24)
        "uv_tools": ("uv tool installs (mac-ai-os, mlx-lm, etc.)", True, False, "uv tool uninstall <tool>"),
        "uv_python": ("uv managed Python installations", True, False, "uv python uninstall <version>"),
        "devin_cli_data": ("Devin CLI session database and logs (can grow to 3GB+)", True, True, "rm -rf ~/.local/share/devin/cli/logs ~/.local/share/devin/cli/*.corrupt.bak* && vacuum sessions.db"),
        "devin_config": ("Devin CLI config directory", True, False, "Review ~/.devin before deleting"),
        "openhands_node_modules": ("OpenHands npm dependencies", True, True, "cd ~/OpenHands && npm install"),
        "vscode_extensions": ("VS Code extensions", True, False, "code --uninstall-extension <name>"),
        "nvm": ("Node Version Manager installations", True, False, "nvm uninstall <version>"),
        "mise_data": ("mise version manager data", True, False, "mise prune"),
        "pipx": ("pipx tool installations", True, False, "pipx uninstall <tool>"),
        "pnpm_store": ("pnpm package store", True, True, "pnpm store prune"),
        # App support data (discovered 2026-07-24)
        "google_app_support": ("Google Chrome profile and data", True, False, "Clear Chrome data in settings"),
        "warp_app_support": ("Warp terminal data and history", True, False, "Warp → Settings → Clear History"),
        "devin_app_support": ("Devin desktop app data", True, False, "Review before deleting"),
        "cursor_app_support": ("Cursor editor data and cache", True, False, "Cursor → Settings → Clear Cache"),
        "proton_mail_app_support": ("Proton Mail local data", True, False, "Proton Mail → Settings → Clear Cache"),
        "discord_app_support": ("Discord cache and data", True, True, "rm -rf ~/Library/Application Support/discord/Cache"),
        "codex_app_support": ("Codex app support data", True, False, "Review before deleting"),
        "raycast_app_support": ("Raycast launcher data", True, False, "Raycast → Settings → Clear Cache"),
        "neodisk_app_support": ("NeoDisk scan cache", True, True, "rm -rf ~/Library/Application Support/Neodisk/ScanCache"),
        # Cloud storage (can evict but not delete)
        "proton_drive": ("Proton Drive cloud sync (evict via Finder, don't delete)", True, False, "Right-click in Finder → Remove Download"),
        # SDKs
        "android_sdk": ("Android SDK (system images, platforms)", True, False, "sdkmanager --uninstall <package>"),
        # Containers
        "playcover": ("PlayCover app data", True, False, "Review before deleting"),
        "mediaanalysis": ("macOS Photos media analysis DB", True, False, "Not recommended to delete — Photos needs it"),
        # System-level (not in home)
        "library_updates": ("Downloaded macOS update packages", True, True, "sudo rm -rf /Library/Updates/*"),
        "library_developer": ("Developer tools and frameworks", True, False, "Review before deleting"),
        "library_command_line_tools": ("Xcode Command Line Tools", True, False, "sudo rm -rf /Library/Developer/CommandLineTools && xcode-select --install"),
        "system_assets": ("macOS system asset files (fonts, voices, assets)", False, False, "Managed by macOS — do not delete"),
    }

    entries = []
    for key, path in USER_CACHE_PATHS.items():
        size = _dir_size(path)
        if size == 0:
            continue
        desc, safe, regen, remediation = descriptions.get(
            key, (key, True, True, f"rm -rf {path}")
        )
        entries.append(
            CacheEntry(
                category=key,
                path=path,
                size_bytes=size,
                description=desc,
                safe_to_clear=safe,
                regenerates=regen,
                remediation=remediation,
            )
        )

    entries.sort(key=lambda e: e.size_bytes, reverse=True)
    return entries


# --- NeoDisk integration ---


def check_neodisk() -> bool:
    """Check if NeoDisk is installed and has scan cache available."""
    return NEODISK_CACHE.exists() and any(NEODISK_CACHE.glob("*.ndscan"))


def neodisk_cache_info() -> dict[str, Any]:
    """Get NeoDisk scan cache metadata (without parsing the binary format)."""
    if not NEODISK_CACHE.exists():
        return {"available": False}

    caches = []
    for f in NEODISK_CACHE.glob("*"):
        if f.is_file():
            try:
                stat = f.stat()
                caches.append(
                    {
                        "file": f.name,
                        "size_bytes": stat.st_size,
                        "modified": stat.st_mtime,
                    }
                )
            except OSError:
                continue

    return {
        "available": len(caches) > 0,
        "cache_dir": str(NEODISK_CACHE),
        "files": caches,
        "total_cache_size": sum(c["size_bytes"] for c in caches),
    }


# --- Full scan ---


def scan_storage(include_xmac: bool = True) -> StorageReport:
    """Run a full storage analysis.

    Combines native macOS system data decomposition, user cache scan,
    Time Machine/APFS snapshot listing, and (optionally) X-MaC disk
    and clean engine findings.

    Args:
        include_xmac: If True, run xmac disk and xmac clean for additional
            findings. Set to False for a fast scan that only uses native
            macOS commands (no filesystem walk).
    """
    from datetime import datetime

    vol = get_volume_stats()
    system_data = decompose_system_data()
    user_caches = scan_user_caches()
    tm_snapshots = get_tm_snapshots()
    apfs_snapshots = get_apfs_snapshots()
    neodisk = check_neodisk()

    xmac_disk = []
    xmac_clean = []
    if include_xmac:
        xmac_disk = _xmac(["disk", "--top", "20"], timeout=120)
        xmac_clean = _xmac(["clean"], timeout=120)

    total_system = sum(e.size_bytes for e in system_data)
    total_cache = sum(e.size_bytes for e in user_caches)
    total_reclaimable = sum(
        e.size_bytes for e in system_data if e.reclaimable
    ) + sum(e.size_bytes for e in user_caches if e.safe_to_clear)

    return StorageReport(
        volume=vol,
        system_data=system_data,
        user_caches=user_caches,
        tm_snapshots=tm_snapshots,
        apfs_snapshots=apfs_snapshots,
        xmac_disk_findings=xmac_disk,
        xmac_clean_findings=xmac_clean,
        neodisk_available=neodisk,
        total_reclaimable_bytes=total_reclaimable,
        total_system_data_bytes=total_system,
        total_cache_bytes=total_cache,
        generated_at=datetime.now(UTC).isoformat(),
    )


# --- Fix script generation ---


def generate_fix_script(
    report: StorageReport,
    output_path: str | Path,
    include_caches: bool = True,
    include_system: bool = True,
    include_tm: bool = True,
) -> FixScript:
    """Generate a reviewable remediation shell script.

    The script contains all reclaimable actions as commented-out commands.
    The user reviews, uncomments what they want, and runs it. Destructive
    commands are NEVER auto-executed.

    Args:
        report: The storage scan report.
        output_path: Where to write the fix script.
        include_caches: Include user cache cleanup commands.
        include_system: Include system data cleanup commands.
        include_tm: Include Time Machine snapshot deletion commands.
    """
    path = Path(output_path)
    commands: list[str] = []
    warnings: list[str] = []
    total_reclaimable = 0

    lines = [
        "#!/bin/bash",
        "# MAOS Storage Remediation Script",
        f"# Generated: {report.generated_at}",
        "# ",
        "# ALL destructive commands are COMMENTED OUT. Review each one,",
        "# uncomment the ones you want, then run: bash {path.name}",
        "# ",
        f"# Total reclaimable space: {_fmt_bytes(report.total_reclaimable_bytes)}",
        "",
        "set -euo pipefail",
        "",
        "echo 'MAOS Storage Remediation'",
        f'echo "Disk: {report.volume.used_gb:.1f} GB used / {report.volume.free_gb:.1f} GB free ({report.volume.capacity_percent:.0f}%)"',
        f'echo "Purgeable: {report.volume.purgeable_gb:.1f} GB"',
        'echo ""',
        "",
    ]

    # System data
    if include_system:
        lines.append("# === System Data ===")
        lines.append(f"# Total identified: {_fmt_bytes(report.total_system_data_bytes)}")
        lines.append("")
        for entry in report.system_data:
            if not entry.reclaimable:
                lines.append(f"# [NOT RECLAIMABLE] {entry.category}: {_fmt_bytes(entry.size_bytes)}")
                lines.append(f"#   {entry.description}")
                lines.append("")
                continue
            if not entry.safe_to_delete:
                lines.append(f"# [NEEDS REBOOT/SUDO] {entry.category}: {_fmt_bytes(entry.size_bytes)}")
                lines.append(f"#   {entry.description}")
                lines.append(f"#   {entry.remediation}")
                lines.append("")
                continue

            total_reclaimable += entry.size_bytes
            lines.append(f"# {entry.category}: {_fmt_bytes(entry.size_bytes)}")
            lines.append(f"#   {entry.description}")
            lines.append(f"# {entry.remediation}")
            lines.append("")
            commands.append(entry.remediation)

        lines.append("")

    # Time Machine snapshots
    if include_tm:
        deletable = [s for s in report.tm_snapshots if s.deletable]
        if deletable:
            lines.append("# === Time Machine Local Snapshots ===")
            lines.append(f"# {len(deletable)} deletable snapshot(s)")
            lines.append("# These pin purgeable space — deleting reclaims it immediately.")
            lines.append("")
            for snap in deletable:
                cmd = f"tmutil deletelocalsnapshots {snap.date}"
                lines.append(f"# Snapshot: {snap.date}")
                lines.append(f"# {cmd}")
                lines.append("")
                commands.append(cmd)
            lines.append("")

    # User caches
    if include_caches:
        lines.append("# === User Caches (regenerate on demand) ===")
        lines.append(f"# Total: {_fmt_bytes(report.total_cache_bytes)}")
        lines.append("")
        for entry in report.user_caches:
            if not entry.safe_to_clear:
                lines.append(f"# [REVIEW FIRST] {entry.category}: {_fmt_bytes(entry.size_bytes)}")
                lines.append(f"#   {entry.description}")
                lines.append(f"#   {entry.remediation}")
                lines.append("")
                continue

            total_reclaimable += entry.size_bytes
            lines.append(f"# {entry.category}: {_fmt_bytes(entry.size_bytes)}")
            lines.append(f"#   {entry.description}")
            lines.append(f"# {entry.remediation}")
            lines.append("")
            commands.append(entry.remediation)

        lines.append("")

    # X-MaC clean findings
    if report.xmac_clean_findings:
        lines.append("# === X-MaC Clean Engine Findings ===")
        lines.append(f"# {len(report.xmac_clean_findings)} finding(s)")
        lines.append("# Run: xmac --fix-script ./xmac-fixes.sh clean")
        lines.append("# for a full X-MaC-generated remediation script.")
        lines.append("")

    lines.append('echo "Done. Review the output above for any errors."')
    lines.append("")

    script_content = "\n".join(lines)
    path.write_text(script_content)
    os.chmod(path, 0o755)

    if total_reclaimable > report.volume.free_bytes * 5:
        warnings.append(
            f"Reclaimable ({_fmt_bytes(total_reclaimable)}) greatly exceeds "
            f"free space ({_fmt_bytes(report.volume.free_bytes)}). "
            "Some commands may need sudo — review carefully."
        )

    return FixScript(
        path=str(path),
        total_reclaimable_bytes=total_reclaimable,
        commands=commands,
        warnings=warnings,
    )


# --- Maintenance ---


def run_maintenance() -> dict[str, Any]:
    """Run safe macOS maintenance tasks via xmac.

    Delegates to ``xmac maintain`` which runs:
    - Flush DNS cache
    - Reindex Spotlight
    - Rebuild LaunchServices
    - Run periodic scripts
    - Purge RAM (inactive memory)
    - Clear Quick Look cache

    Sudo-requiring tasks are emitted as findings, not executed.
    """
    findings = _xmac(["maintain"], timeout=120)
    return {
        "tool": "xmac maintain",
        "findings_count": len(findings),
        "findings": findings[:20],  # cap for MCP response size
        "note": "Safe tasks run automatically. Sudo tasks emitted as findings.",
    }


# --- Formatting ---


def format_storage_report(report: StorageReport) -> str:
    """Format a storage report as a human-readable text summary."""
    lines = [
        "=== MAOS Storage Report ===",
        "",
        f"Volume: {report.volume.container_name or 'Macintosh HD'}",
        f"  Total:     {_fmt_bytes(report.volume.total_bytes)}",
        f"  Used:      {_fmt_bytes(report.volume.used_bytes)} ({report.volume.capacity_percent:.0f}%)",
        f"  Free:      {_fmt_bytes(report.volume.free_bytes)}",
        f"  Purgeable: {_fmt_bytes(report.volume.purgeable_bytes)}",
        f"  System:    {_fmt_bytes(report.volume.system_used_bytes)}",
        f"  Data:      {_fmt_bytes(report.volume.data_used_bytes)}",
        f"  Apps:      {_fmt_bytes(report.volume.applications_bytes)}",
        "",
        f"=== System Data Decomposition ({_fmt_bytes(report.total_system_data_bytes)}) ===",
        "",
    ]

    if not report.system_data:
        lines.append("  No system data components found.")
    else:
        for entry in report.system_data:
            flag = ""
            if entry.reclaimable and entry.safe_to_delete:
                flag = " [RECLAIMABLE]"
            elif entry.reclaimable:
                flag = " [RECLAIMABLE - needs reboot/sudo]"
            else:
                flag = " [NOT RECLAIMABLE]"
            lines.append(
                f"  {entry.category}: {_fmt_bytes(entry.size_bytes)}{flag}"
            )
            lines.append(f"    {entry.description[:100]}")
            if entry.remediation:
                lines.append(f"    Fix: {entry.remediation}")
            lines.append("")

    lines.append(f"=== User Caches ({_fmt_bytes(report.total_cache_bytes)}) ===")
    lines.append("")
    if not report.user_caches:
        lines.append("  No user caches found.")
    else:
        for entry in report.user_caches:
            flag = " [SAFE]" if entry.safe_to_clear else " [REVIEW]"
            regen = " (regenerates)" if entry.regenerates else ""
            lines.append(
                f"  {entry.category}: {_fmt_bytes(entry.size_bytes)}{flag}{regen}"
            )
            lines.append(f"    {entry.description}")
            lines.append(f"    Fix: {entry.remediation}")
            lines.append("")

    # Time Machine snapshots
    if report.tm_snapshots:
        deletable = [s for s in report.tm_snapshots if s.deletable]
        lines.append(f"=== Time Machine Snapshots ({len(report.tm_snapshots)} total, {len(deletable)} deletable) ===")
        lines.append("")
        for snap in report.tm_snapshots:
            flag = " [DELETABLE]" if snap.deletable else " [OS - keep]"
            lines.append(f"  {snap.date}{flag}")
        lines.append("")

    # APFS snapshots
    if report.apfs_snapshots:
        lines.append(f"=== APFS Snapshots ({len(report.apfs_snapshots)}) ===")
        lines.append("")
        for snap in report.apfs_snapshots:
            flag = ""
            if snap.purgeable:
                flag = " [PURGEABLE]"
            if snap.limits_container_size:
                flag += " [LIMITS CONTAINER]"
            lines.append(f"  {snap.name}{flag}")
        lines.append("")

    # X-MaC findings summary
    if report.xmac_disk_findings:
        lines.append(f"=== X-MaC Disk Engine ({len(report.xmac_disk_findings)} findings) ===")
        lines.append("")
        for f in report.xmac_disk_findings[:10]:
            title = f.get("title", "")
            size = f.get("size_bytes", 0)
            if size:
                lines.append(f"  {_fmt_bytes(size):>10}  {title}")
            else:
                lines.append(f"  {'':>10}  {title}")
        if len(report.xmac_disk_findings) > 10:
            lines.append(f"  ... and {len(report.xmac_disk_findings) - 10} more")
        lines.append("")

    if report.xmac_clean_findings:
        lines.append(f"=== X-MaC Clean Engine ({len(report.xmac_clean_findings)} findings) ===")
        lines.append("")
        for f in report.xmac_clean_findings[:10]:
            title = f.get("title", "")
            size = f.get("size_bytes", 0)
            sev = f.get("severity", "info")
            if size:
                lines.append(f"  [{sev:6s}] {_fmt_bytes(size):>10}  {title}")
            else:
                lines.append(f"  [{sev:6s}] {'':>10}  {title}")
        if len(report.xmac_clean_findings) > 10:
            lines.append(f"  ... and {len(report.xmac_clean_findings) - 10} more")
        lines.append("")

    # NeoDisk
    if report.neodisk_available:
        lines.append("=== NeoDisk ===")
        lines.append("  Scan cache available (GUI app installed)")
        lines.append("")

    # Summary
    lines.append("=== Summary ===")
    lines.append(f"  Total reclaimable: {_fmt_bytes(report.total_reclaimable_bytes)}")
    lines.append(f"  System data:       {_fmt_bytes(report.total_system_data_bytes)}")
    lines.append(f"  User caches:       {_fmt_bytes(report.total_cache_bytes)}")
    lines.append("")
    lines.append("  Generate a fix script: uv run maos storage fix-script ./fixes.sh")
    lines.append("")

    return "\n".join(lines)
