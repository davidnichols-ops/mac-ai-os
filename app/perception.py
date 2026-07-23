"""Perception layer — shells out to the AAFP CLI for web search, browsing,
PDF extraction, and OCR.

This module wraps the AAFP perception commands (`aafp search`, `aafp browse`,
`aafp read-pdf`, `aafp ocr`) and returns structured Python objects. It is the
bridge between MAOS agents and the external internet.

The AAFP CLI binary must be on PATH (or at the configured path). Build it with:
    cd ~/Projects/aafp && cargo build --release --bin aafp

All commands support JSON output via `--json`, which we parse here.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from app.log import get_logger

log = get_logger(__name__)

# Default binary path — checks PATH first, falls back to the release build.
_DEFAULT_AAFP_BIN = str(Path.home() / "Projects/aafp/target/release/aafp")


def _find_aafp_bin() -> str:
    """Locate the aafp binary."""
    p = shutil.which("aafp")
    if p:
        return p
    if os.path.isfile(_DEFAULT_AAFP_BIN) and os.access(_DEFAULT_AAFP_BIN, os.X_OK):
        return _DEFAULT_AAFP_BIN
    return ""


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    score: float = 0.0
    source: str = "duckduckgo"


@dataclass
class BrowseResult:
    url: str
    title: str = ""
    content: str = ""
    markdown: str = ""
    error: str | None = None


@dataclass
class PdfResult:
    path: str
    pages: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def text(self) -> str:
        return "\n\n---\n\n".join(self.pages)


@dataclass
class OcrResult:
    path: str
    text: str = ""
    language: str = "eng"
    error: str | None = None


class Perception:
    """Thin wrapper over the AAFP perception CLI commands."""

    def __init__(self, aafp_bin: str | None = None, timeout: int = 30):
        self.bin = aafp_bin or _find_aafp_bin()
        self.timeout = timeout
        if not self.bin:
            log.warning("perception.no_binary", msg="aafp binary not found on PATH or default location")

    @property
    def available(self) -> bool:
        return bool(self.bin)

    def _run(self, args: list[str]) -> dict | str:
        """Run an aafp subcommand with --json and parse the output."""
        if not self.bin:
            raise RuntimeError("aafp binary not available")
        cmd = [self.bin] + args + ["--json"]
        log.info("perception.exec", cmd=" ".join(cmd[:3]))
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"aafp command timed out after {self.timeout}s") from None
        if proc.returncode != 0:
            err = proc.stderr.strip() or proc.stdout.strip()
            raise RuntimeError(f"aafp exited {proc.returncode}: {err[:500]}")
        out = proc.stdout.strip()
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return out

    def search(self, query: str, num: int = 10) -> list[SearchResult]:
        """Web search via DuckDuckGo (free, no API key)."""
        data = self._run(["search", query, "--num", str(num)])
        if isinstance(data, str):
            return []
        results = []
        for r in data.get("results", []):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                snippet=r.get("snippet", ""),
                score=r.get("score", 0.0),
                source=r.get("source", "duckduckgo"),
            ))
        log.info("perception.search.done", query=query, results=len(results))
        return results

    def browse(self, url: str) -> BrowseResult:
        """Browse a web page via Firecrawl (requires FIRECRAWL_API_KEY)."""
        data = self._run(["browse", url])
        if isinstance(data, str):
            return BrowseResult(url=url, error=data[:500])
        return BrowseResult(
            url=url,
            title=data.get("title", ""),
            content=data.get("content", ""),
            markdown=data.get("markdown", data.get("content", "")),
            error=data.get("error"),
        )

    def read_pdf(self, path: str) -> PdfResult:
        """Extract text from a PDF via PyMuPDF."""
        data = self._run(["read-pdf", path])
        if isinstance(data, str):
            return PdfResult(path=path, error=data[:500])
        pages = data.get("pages", [])
        if isinstance(pages, list):
            pages = [p if isinstance(p, str) else p.get("text", "") for p in pages]
        return PdfResult(path=path, pages=pages, error=data.get("error"))

    def ocr(self, path: str, lang: str = "eng") -> OcrResult:
        """Run OCR on an image via Tesseract."""
        data = self._run(["ocr", path, "--lang", lang])
        if isinstance(data, str):
            return OcrResult(path=path, text=data, language=lang)
        return OcrResult(
            path=path,
            text=data.get("text", ""),
            language=data.get("language", lang),
            error=data.get("error"),
        )
