"""Google Drive — file storage via Drive API v3 (OAuth).

This is the cloud storage channel for MAOS. Agents can list, search,
upload, download, and organize files in the user's Workspace Drive.
Useful for backing up session artifacts, sharing reports, and keeping
MAOS data organized in a durable location.

Architecture:
    Google Drive OAuth credentials live in ~/.gmail-mcp/ (shared with the
    gmail-github-to-maos.js sync script and the Gmail module). The
    specific Google account is determined by the stored credentials, so
    this module remains account-agnostic.

    Tokens expire. This module refreshes them on demand using the
    stored refresh_token + client credentials. No interactive auth flow.

    All API calls go through `_api_request`, which auto-refreshes the
    access token on 401 and retries once. No external dependencies —
    just stdlib (urllib, json, base64, mimetypes, pathlib).

Security:
    - Credentials are read from ~/.gmail-mcp/, never written to logs.
    - Refreshed tokens are saved back to the credentials file so other
      consumers benefit.
    - The Drive account is whichever account the stored credentials
      belong to; no addresses are hardcoded.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC
from pathlib import Path
from typing import Any

from app.log import get_logger

log = get_logger(__name__)

DRIVE_DIR = Path.home() / ".gmail-mcp"
# Primary: ~/.gdrive-token.json (gdrive CLI format — has client_id/secret embedded)
# Fallback: ~/.gmail-mcp/drive-credentials.json + account-a-keys.json
GDRIVE_TOKEN_FILE = Path.home() / ".gdrive-token.json"
CREDS_FILE = DRIVE_DIR / "drive-credentials.json"
KEYS_FILE = DRIVE_DIR / "account-a-keys.json"

DRIVE_API_BASE = "https://www.googleapis.com/drive/v3"
DRIVE_UPLOAD_BASE = "https://www.googleapis.com/upload/drive/v3"
OAUTH_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Fields we request for file metadata — keeps responses lean and consistent.
_FILE_FIELDS = (
    "id,name,mimeType,size,modifiedTime,createdTime,parents,"
    "trashed,webViewLink,iconLink,description"
)
_LIST_FIELDS = f"nextPageToken,files({_FILE_FIELDS})"

# Threshold for switching from simple/multipart upload to resumable upload.
_RESUMABLE_THRESHOLD = 5 * 1024 * 1024  # 5MB


class DriveError(Exception):
    """Failed to perform a Drive API operation."""


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------

def refresh_access_token() -> str:
    """Refresh the Google Drive OAuth access token using the stored refresh token.

    Supports two credential formats:
    1. ~/.gdrive-token.json (gdrive CLI format — client_id/secret embedded)
    2. ~/.gmail-mcp/drive-credentials.json + account-a-keys.json (split format)

    Returns a fresh access token. Also saves it back to the credentials
    file so the sync script and other consumers benefit.

    Raises:
        DriveError: If the credentials files are missing, the refresh
            request fails, or the response lacks an access_token.
    """
    # Try gdrive-token.json first (has everything in one file)
    if GDRIVE_TOKEN_FILE.exists():
        creds = json.loads(GDRIVE_TOKEN_FILE.read_text())
        refresh_token = creds.get("refresh_token")
        client_id = creds.get("client_id")
        client_secret = creds.get("client_secret")
        if refresh_token and client_id and client_secret:
            access_token = _do_refresh(
                client_id, client_secret, refresh_token, creds, GDRIVE_TOKEN_FILE
            )
            if access_token:
                return access_token

    # Fallback to split format
    if not KEYS_FILE.exists():
        raise DriveError(f"Drive keys file not found: {KEYS_FILE}")
    if not CREDS_FILE.exists():
        raise DriveError(f"Drive credentials file not found: {CREDS_FILE}")

    keys = json.loads(KEYS_FILE.read_text())
    installed = keys.get("installed", keys)
    creds = json.loads(CREDS_FILE.read_text())

    refresh_token = creds.get("refresh_token")
    if not refresh_token:
        raise DriveError("No refresh_token in credentials file")

    return _do_refresh(
        installed["client_id"],
        installed["client_secret"],
        refresh_token,
        creds,
        CREDS_FILE,
    )


def _do_refresh(
    client_id: str,
    client_secret: str,
    refresh_token: str,
    creds: dict[str, Any],
    save_path: Path,
) -> str:
    """Perform the token refresh and save back to save_path."""
    data = urllib.parse.urlencode(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
    ).encode()

    req = urllib.request.Request(OAUTH_TOKEN_URL, data=data)
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        token_data = json.loads(resp.read())
    except Exception as e:
        raise DriveError(f"Token refresh failed: {e}") from e

    access_token = token_data.get("access_token")
    if not access_token:
        raise DriveError("No access_token in refresh response")

    # Save refreshed token back so other consumers benefit.
    creds["token"] = access_token
    if "access_token" in creds:
        creds["access_token"] = access_token
    if "expires_in" in token_data:
        if "expiry_date" in creds:
            creds["expiry_date"] = token_data["expires_in"]
        if "expiry" in creds:
            from datetime import datetime, timedelta
            creds["expiry"] = (
                datetime.now(UTC) + timedelta(seconds=token_data["expires_in"])
            ).isoformat()
    try:
        save_path.write_text(json.dumps(creds, indent=2))
    except Exception as e:
        log.warning("gdrive.token_save_failed", error=str(e))

    log.info("gdrive.token.refreshed")
    return access_token


_cached_token: str | None = None
_cached_token_time: float = 0


def _load_access_token() -> str:
    """Load an access token with a 50-minute cache.

    Tokens last 60 minutes; we refresh at 50 to avoid edge cases.
    """
    import time

    global _cached_token, _cached_token_time
    if _cached_token and (time.time() - _cached_token_time) < 3000:
        return _cached_token
    _cached_token = refresh_access_token()
    _cached_token_time = time.time()
    return _cached_token


# ---------------------------------------------------------------------------
# Core API request helper
# ---------------------------------------------------------------------------

def _api_request(
    path: str,
    method: str = "GET",
    body: Any = None,
    params: dict[str, Any] | None = None,
    *,
    base_url: str = DRIVE_API_BASE,
    content_type: str = "application/json",
    raw_body: bytes | None = None,
    _retry: bool = True,
) -> dict[str, Any]:
    """Make an authenticated Drive API v3 request.

    Auto-refreshes the access token on 401 and retries once.

    Args:
        path: API path (e.g. "/files" or "/files/{id}"). May include a
            leading slash; it will be normalized.
        method: HTTP method (GET, POST, PATCH, DELETE).
        body: Request body — will be JSON-encoded if not None. Ignored
            when raw_body is provided.
        params: Query parameters.
        base_url: Base URL for the request (defaults to DRIVE_API_BASE).
        content_type: Content-Type header for JSON bodies.
        raw_body: Raw bytes to send as the request body (bypasses JSON
            encoding of `body`).
        _retry: Internal — whether to retry once after a token refresh.

    Returns:
        Parsed JSON response as a dict. For endpoints that return no
        body (e.g. DELETE), returns an empty dict.

    Raises:
        DriveError: On HTTP errors (after retry) or network failures.
    """
    path = path.lstrip("/")
    url = f"{base_url}/{path}"
    if params:
        # Drop None values so callers can pass optional params inline.
        clean = {k: v for k, v in params.items() if v is not None}
        if clean:
            url = f"{url}?{urllib.parse.urlencode(clean)}"

    access_token = _load_access_token()

    if raw_body is not None:
        data: bytes | None = raw_body
    elif body is not None:
        data = json.dumps(body).encode()
    else:
        data = None

    headers = {
        "Authorization": f"Bearer {access_token}",
    }
    if data is not None:
        headers["Content-Type"] = content_type

    req = urllib.request.Request(url, data=data, headers=headers, method=method)

    try:
        resp = urllib.request.urlopen(req, timeout=60)
        raw = resp.read()
        if not raw:
            return {}
        return json.loads(raw)
    except urllib.error.HTTPError as e:
        status = e.code
        error_body = ""
        try:
            error_body = e.read().decode()[:500]
        except Exception:
            pass

        # 401 → refresh and retry once.
        if status == 401 and _retry:
            log.warning("gdrive.401_refreshing")
            refresh_access_token()
            return _api_request(
                path,
                method=method,
                body=body,
                params=params,
                base_url=base_url,
                content_type=content_type,
                raw_body=raw_body,
                _retry=False,
            )

        log.error("gdrive.api_error", status=status, method=method, path=path, error=error_body)
        raise DriveError(f"Drive API error {status} {method} {path}: {error_body}") from e
    except Exception as e:
        log.error("gdrive.request_failed", method=method, path=path, error=str(e))
        raise DriveError(f"Request failed {method} {path}: {e}") from e


# ---------------------------------------------------------------------------
# File listing & metadata
# ---------------------------------------------------------------------------

def list_files(
    folder_id: str | None = None,
    page_size: int = 100,
    query: str | None = None,
    order_by: str | None = None,
) -> list[dict[str, Any]]:
    """List files in a folder (or root if folder_id is None).

    Args:
        folder_id: Parent folder ID. None lists files in root / My Drive.
            Note: Drive's root is special — passing None uses no parent
            filter, which includes all files the user can see.
        page_size: Max results (1–1000).
        query: Additional Drive query string to AND with the folder filter
            (e.g. "trashed = false").
        order_by: Sort key (e.g. "modifiedTime desc").

    Returns:
        List of file dicts with id, name, mimeType, size, modifiedTime,
        parents, etc.
    """
    clauses: list[str] = ["trashed = false"]
    if folder_id:
        clauses.append(f"'{folder_id}' in parents")
    if query:
        clauses.append(f"({query})")
    q = " and ".join(clauses)

    params = {
        "q": q,
        "pageSize": page_size,
        "fields": _LIST_FIELDS,
        "orderBy": order_by,
    }

    files: list[dict[str, Any]] = []
    result = _api_request("/files", params=params)
    files.extend(result.get("files", []))

    # Paginate if needed.
    while result.get("nextPageToken"):
        params["pageToken"] = result["nextPageToken"]
        result = _api_request("/files", params=params)
        files.extend(result.get("files", []))

    log.info("gdrive.listed", count=len(files), folder=folder_id or "root")
    return files


def get_file_metadata(file_id: str) -> dict[str, Any]:
    """Get full metadata for a single file.

    Args:
        file_id: The Drive file ID.

    Returns:
        Dict with id, name, mimeType, size, modifiedTime, parents, etc.
    """
    return _api_request(f"/files/{file_id}", params={"fields": _FILE_FIELDS})


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_file(file_id: str) -> bytes:
    """Download a file's content as bytes.

    Args:
        file_id: The Drive file ID.

    Returns:
        Raw file content as bytes.

    Raises:
        DriveError: On HTTP errors or network failures.
    """
    path = f"/files/{file_id}"
    url = f"{DRIVE_API_BASE}/{path.lstrip('/')}?alt=media"
    access_token = _load_access_token()

    def _do_fetch(token: str) -> bytes:
        req = urllib.request.Request(
            url,
            headers={"Authorization": f"Bearer {token}"},
            method="GET",
        )
        resp = urllib.request.urlopen(req, timeout=120)
        return resp.read()

    try:
        return _do_fetch(access_token)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            log.warning("gdrive.download.401_refreshing")
            new_token = refresh_access_token()
            try:
                return _do_fetch(new_token)
            except urllib.error.HTTPError as e2:
                error_body = ""
                try:
                    error_body = e2.read().decode()[:500]
                except Exception:
                    pass
                log.error("gdrive.download_failed", file_id=file_id, status=e2.code, error=error_body)
                raise DriveError(f"Download failed {e2.code} for {file_id}: {error_body}") from e2
        error_body = ""
        try:
            error_body = e.read().decode()[:500]
        except Exception:
            pass
        log.error("gdrive.download_failed", file_id=file_id, status=e.code, error=error_body)
        raise DriveError(f"Download failed {e.code} for {file_id}: {error_body}") from e
    except Exception as e:
        log.error("gdrive.download_failed", file_id=file_id, error=str(e))
        raise DriveError(f"Download failed for {file_id}: {e}") from e


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

def _simple_multipart_upload(
    name: str,
    content: bytes,
    mime_type: str,
    parent_folder_id: str | None,
) -> dict[str, Any]:
    """Upload a small file (<5MB) via multipart upload.

    Uses the Drive API's multipart upload endpoint. The metadata and
    file content are sent in a single multipart/related request.
    """
    metadata: dict[str, Any] = {"name": name}
    if parent_folder_id:
        metadata["parents"] = [parent_folder_id]

    boundary = "maos-drive-" + base64.b64encode(name.encode()).decode().replace("=", "")[:24]
    delimiter = f"--{boundary}\r\n".encode()

    body = b""
    body += delimiter
    body += b"Content-Type: application/json; charset=UTF-8\r\n\r\n"
    body += json.dumps(metadata).encode()
    body += b"\r\n"
    body += delimiter
    body += f"Content-Type: {mime_type}\r\n\r\n".encode()
    body += content
    body += b"\r\n"
    body += f"--{boundary}--\r\n".encode()

    params = {"uploadType": "multipart", "fields": _FILE_FIELDS}
    return _api_request(
        "/files",
        method="POST",
        raw_body=body,
        params=params,
        base_url=DRIVE_UPLOAD_BASE,
        content_type=f"multipart/related; boundary={boundary}",
    )


def _resumable_upload(
    name: str,
    content: bytes,
    mime_type: str,
    parent_folder_id: str | None,
) -> dict[str, Any]:
    """Upload a larger file (>=5MB) via resumable upload.

    Initiates a resumable session, then uploads the content in a single
    PUT (chunked transfer handled by urllib). For very large files this
    could be extended to upload in chunks with resume support, but a
    single PUT is sufficient for the 5TB Workspace use case.
    """
    metadata: dict[str, Any] = {"name": name}
    if parent_folder_id:
        metadata["parents"] = [parent_folder_id]

    # Step 1: initiate the resumable session.
    session_url = f"{DRIVE_UPLOAD_BASE}/files?uploadType=resumable&fields={urllib.parse.quote(_FILE_FIELDS)}"
    access_token = _load_access_token()

    init_req = urllib.request.Request(
        session_url,
        data=json.dumps(metadata).encode(),
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Upload-Content-Type": mime_type,
            "X-Upload-Content-Length": str(len(content)),
        },
        method="POST",
    )

    try:
        init_resp = urllib.request.urlopen(init_req, timeout=30)
        upload_url = init_resp.headers.get("Location", "")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            log.warning("gdrive.upload.init.401_refreshing")
            access_token = refresh_access_token()
            init_req.add_header("Authorization", f"Bearer {access_token}")
            try:
                init_resp = urllib.request.urlopen(init_req, timeout=30)
                upload_url = init_resp.headers.get("Location", "")
            except urllib.error.HTTPError as e2:
                error_body = ""
                try:
                    error_body = e2.read().decode()[:500]
                except Exception:
                    pass
                raise DriveError(f"Resumable init failed {e2.code}: {error_body}") from e2
        else:
            error_body = ""
            try:
                error_body = e.read().decode()[:500]
            except Exception:
                pass
            raise DriveError(f"Resumable init failed {e.code}: {error_body}") from e
    except Exception as e:
        raise DriveError(f"Resumable init failed: {e}") from e

    if not upload_url:
        raise DriveError("Resumable session did not return a Location URL")

    # Step 2: PUT the content to the session URL.
    put_req = urllib.request.Request(
        upload_url,
        data=content,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": mime_type,
            "Content-Length": str(len(content)),
        },
        method="PUT",
    )

    try:
        put_resp = urllib.request.urlopen(put_req, timeout=300)
        raw = put_resp.read()
        if not raw:
            return {}
        return json.loads(raw)
    except urllib.error.HTTPError as e:
        error_body = ""
        try:
            error_body = e.read().decode()[:500]
        except Exception:
            pass
        log.error("gdrive.upload_failed", name=name, status=e.code, error=error_body)
        raise DriveError(f"Resumable upload failed {e.code}: {error_body}") from e
    except Exception as e:
        log.error("gdrive.upload_failed", name=name, error=str(e))
        raise DriveError(f"Resumable upload failed: {e}") from e


def upload_file(
    name: str,
    content: bytes,
    mime_type: str,
    parent_folder_id: str | None = None,
) -> dict[str, Any]:
    """Upload a new file to Google Drive.

    Uses multipart upload for files < 5MB and resumable upload for
    larger files.

    Args:
        name: The file name in Drive.
        content: File content as bytes.
        mime_type: MIME type (e.g. "text/plain", "application/json").
        parent_folder_id: Parent folder ID. None uploads to root.

    Returns:
        File metadata dict (id, name, mimeType, etc.).
    """
    if not content and content != b"":
        raise DriveError("content must be bytes")

    if len(content) < _RESUMABLE_THRESHOLD:
        result = _simple_multipart_upload(name, content, mime_type, parent_folder_id)
    else:
        result = _resumable_upload(name, content, mime_type, parent_folder_id)

    log.info("gdrive.uploaded", name=name, size=len(content), file_id=result.get("id"))
    return result


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

def create_folder(name: str, parent_folder_id: str | None = None) -> dict[str, Any]:
    """Create a folder in Google Drive.

    Args:
        name: Folder name.
        parent_folder_id: Parent folder ID. None creates in root.

    Returns:
        Folder metadata dict.
    """
    body: dict[str, Any] = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
    }
    if parent_folder_id:
        body["parents"] = [parent_folder_id]

    result = _api_request("/files", method="POST", body=body, params={"fields": _FILE_FIELDS})
    log.info("gdrive.folder_created", name=name, folder_id=result.get("id"))
    return result


def get_or_create_folder(name: str, parent_folder_id: str | None = None) -> dict[str, Any]:
    """Find a folder by name under a parent, creating it if not found.

    Useful for organizing MAOS data into a predictable folder structure.

    Args:
        name: Folder name to find or create.
        parent_folder_id: Parent folder ID. None searches/creates in root.

    Returns:
        Folder metadata dict (existing or newly created).
    """
    clauses = [
        "mimeType = 'application/vnd.google-apps.folder'",
        "trashed = false",
        f"name = '{name}'",
    ]
    if parent_folder_id:
        clauses.append(f"'{parent_folder_id}' in parents")
    q = " and ".join(clauses)

    result = _api_request(
        "/files",
        params={"q": q, "fields": _LIST_FIELDS, "pageSize": 1},
    )
    files = result.get("files", [])
    if files:
        log.info("gdrive.folder_found", name=name, folder_id=files[0].get("id"))
        return files[0]

    return create_folder(name, parent_folder_id)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_files(query: str, page_size: int = 50) -> list[dict[str, Any]]:
    """Search Google Drive using a Drive query string.

    Examples:
        search_files("name contains 'report'")
        search_files("mimeType = 'application/pdf'")
        search_files("fullText contains 'invoice'")

    Args:
        query: Drive query string (see Google Drive API query docs).
        page_size: Max results (1–1000).

    Returns:
        List of matching file dicts.
    """
    q = f"({query}) and trashed = false"
    params = {"q": q, "pageSize": page_size, "fields": _LIST_FIELDS}

    files: list[dict[str, Any]] = []
    result = _api_request("/files", params=params)
    files.extend(result.get("files", []))

    while result.get("nextPageToken") and len(files) < page_size:
        params["pageToken"] = result["nextPageToken"]
        result = _api_request("/files", params=params)
        files.extend(result.get("files", []))
        if len(files) >= page_size:
            files = files[:page_size]
            break

    log.info("gdrive.searched", query=query, count=len(files))
    return files


# ---------------------------------------------------------------------------
# Delete / move
# ---------------------------------------------------------------------------

def delete_file(file_id: str) -> bool:
    """Delete a file permanently from Google Drive.

    Note: this is a hard delete, not a move-to-trash. Drive does not
    offer a "trash" via the delete endpoint — trashing requires a
    PATCH with `trashed=true`. This function permanently deletes.

    Args:
        file_id: The Drive file ID.

    Returns:
        True if deleted successfully.
    """
    _api_request(f"/files/{file_id}", method="DELETE")
    log.info("gdrive.deleted", file_id=file_id)
    return True


def move_file(file_id: str, new_parent_folder_id: str) -> dict[str, Any]:
    """Move a file to a different folder.

    Removes all existing parents and adds the new one. If you need to
    add a parent without removing existing ones, use the API directly.

    Args:
        file_id: The Drive file ID to move.
        new_parent_folder_id: The destination folder ID.

    Returns:
        Updated file metadata dict.
    """
    # First, fetch current parents.
    meta = get_file_metadata(file_id)
    current_parents = ",".join(meta.get("parents", []))

    params = {
        "addParents": new_parent_folder_id,
        "removeParents": current_parents or None,
        "fields": _FILE_FIELDS,
    }
    result = _api_request(f"/files/{file_id}", method="PATCH", params=params)
    log.info("gdrive.moved", file_id=file_id, new_parent=new_parent_folder_id)
    return result


# ---------------------------------------------------------------------------
# Storage quota
# ---------------------------------------------------------------------------

def get_storage_info() -> dict[str, Any]:
    """Get storage quota info from the Drive about endpoint.

    Returns:
        Dict with storageQuota (limit, usage, usageInDrive, etc.) and
        user info.
    """
    return _api_request("/about", params={"fields": "storageQuota,user"})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_format_file(f: dict[str, Any]) -> str:
    """Format a file dict for CLI display."""
    fid = f.get("id", "")
    name = f.get("name", "")
    mime = f.get("mimeType", "")
    size = f.get("size", "")
    modified = f.get("modifiedTime", "")
    parents = ",".join(f.get("parents", []))
    return f"{fid}\t{name}\t{mime}\t{size}\t{modified}\t{parents}"


def _cli_format_quota(q: dict[str, Any]) -> str:
    """Format storage quota for CLI display."""
    limit = int(q.get("limit", 0))
    usage = int(q.get("usage", 0))
    in_drive = int(q.get("usageInDrive", 0))
    in_trash = int(q.get("usageInDriveTrash", 0))

    def _human(n: int) -> str:
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if n < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024
        return f"{n:.1f} PB"

    pct = (usage / limit * 100) if limit else 0
    lines = [
        f"Limit:           {_human(limit)}",
        f"Used:            {_human(usage)} ({pct:.1f}%)",
        f"  In Drive:      {_human(in_drive)}",
        f"  In Drive trash:{_human(in_trash)}",
        f"  Free:          {_human(limit - usage)}",
    ]
    return "\n".join(lines)


def _cli(argv: list[str]) -> int:
    """CLI entry point. Returns exit code."""
    if len(argv) < 2:
        print(
            "Usage: python3 app/gdrive.py <command> [args]\n"
            "\n"
            "Commands:\n"
            "  list [folder_id]              List files in a folder (root if omitted)\n"
            "  search <query>                Search Drive (e.g. \"name contains 'x'\")\n"
            "  upload <local_path> [name] [folder_id]  Upload a file\n"
            "  download <file_id> [local_path]        Download a file\n"
            "  mkdir <name> [parent_id]      Create a folder\n"
            "  quota                         Show storage quota\n"
            "  info <file_id>                Show file metadata\n",
            file=sys.stderr,
        )
        return 1

    cmd = argv[1]

    try:
        if cmd == "list":
            folder_id = argv[2] if len(argv) > 2 else None
            files = list_files(folder_id=folder_id)
            if not files:
                print("(no files)")
            for f in files:
                print(_cli_format_file(f))
            return 0

        elif cmd == "search":
            if len(argv) < 3:
                print("Usage: search <query>", file=sys.stderr)
                return 1
            query = argv[2]
            files = search_files(query)
            if not files:
                print("(no matches)")
            for f in files:
                print(_cli_format_file(f))
            return 0

        elif cmd == "upload":
            if len(argv) < 3:
                print("Usage: upload <local_path> [remote_name] [folder_id]", file=sys.stderr)
                return 1
            local_path = Path(argv[2])
            remote_name = argv[3] if len(argv) > 3 else local_path.name
            folder_id = argv[4] if len(argv) > 4 else None
            if not local_path.exists():
                print(f"File not found: {local_path}", file=sys.stderr)
                return 1
            content = local_path.read_bytes()
            mime_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
            result = upload_file(remote_name, content, mime_type, parent_folder_id=folder_id)
            print(f"Uploaded: {result.get('id')}  {result.get('name')}  ({len(content)} bytes)")
            return 0

        elif cmd == "download":
            if len(argv) < 3:
                print("Usage: download <file_id> [local_path]", file=sys.stderr)
                return 1
            file_id = argv[2]
            local_path = Path(argv[3]) if len(argv) > 3 else Path(file_id)
            content = download_file(file_id)
            local_path.write_bytes(content)
            print(f"Downloaded {len(content)} bytes → {local_path}")
            return 0

        elif cmd == "mkdir":
            if len(argv) < 3:
                print("Usage: mkdir <name> [parent_id]", file=sys.stderr)
                return 1
            name = argv[2]
            parent_id = argv[3] if len(argv) > 3 else None
            result = create_folder(name, parent_folder_id=parent_id)
            print(f"Created folder: {result.get('id')}  {result.get('name')}")
            return 0

        elif cmd == "quota":
            info = get_storage_info()
            quota = info.get("storageQuota", {})
            user = info.get("user", {})
            print(f"Account: {user.get('emailAddress', 'unknown')}")
            print(_cli_format_quota(quota))
            return 0

        elif cmd == "info":
            if len(argv) < 3:
                print("Usage: info <file_id>", file=sys.stderr)
                return 1
            file_id = argv[2]
            meta = get_file_metadata(file_id)
            print(json.dumps(meta, indent=2))
            return 0

        else:
            print(f"Unknown command: {cmd}", file=sys.stderr)
            return 1

    except DriveError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(_cli(sys.argv))
