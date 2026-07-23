"""`maos search/browse/read-pdf/ocr` — perception commands via AAFP CLI."""

from __future__ import annotations

import json as json_mod

import typer

from app.perception import Perception
from app.ui.console import print_markdown, print_panel


def search(
    query: str = typer.Argument(..., help="Search query"),
    num: int = typer.Option(10, help="Max results"),
    json: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Web search via DuckDuckGo (free, no API key). Powered by AAFP perception."""
    perc = Perception()
    if not perc.available:
        print_panel("aafp binary not found. Build it: cd ~/Projects/aafp && cargo build --release --bin aafp", title="Error")
        raise typer.Exit(1) from None
    try:
        results = perc.search(query, num=num)
    except Exception as e:
        print_panel(str(e), title="Search Error")
        raise typer.Exit(1) from None
    if json:
        typer.echo(json_mod.dumps([r.__dict__ for r in results], indent=2))
        return
    if not results:
        print_panel("No results.", title="Search")
        return
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. **{r.title}**\n   {r.url}\n   {r.snippet[:200]}")
    print_panel("\n\n".join(lines), title=f"Search: {query}")


def browse(
    url: str = typer.Argument(..., help="URL to browse"),
    json: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Browse a web page via Firecrawl (requires FIRECRAWL_API_KEY). Powered by AAFP perception."""
    perc = Perception()
    if not perc.available:
        print_panel("aafp binary not found. Build it: cd ~/Projects/aafp && cargo build --release --bin aafp", title="Error")
        raise typer.Exit(1) from None
    try:
        result = perc.browse(url)
    except Exception as e:
        print_panel(str(e), title="Browse Error")
        raise typer.Exit(1) from None
    if json:
        typer.echo(json_mod.dumps(result.__dict__, indent=2))
        return
    if result.error:
        print_panel(result.error, title="Browse Error")
        raise typer.Exit(1) from None
    if result.markdown:
        print_markdown(result.markdown)
    else:
        print_panel(result.content[:5000] or "No content extracted.", title=result.title or url)


def read_pdf(
    path: str = typer.Argument(..., help="Path to PDF file"),
    json: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Extract text from a PDF via PyMuPDF. Powered by AAFP perception."""
    perc = Perception()
    if not perc.available:
        print_panel("aafp binary not found. Build it: cd ~/Projects/aafp && cargo build --release --bin aafp", title="Error")
        raise typer.Exit(1) from None
    try:
        result = perc.read_pdf(path)
    except Exception as e:
        print_panel(str(e), title="PDF Error")
        raise typer.Exit(1) from None
    if json:
        typer.echo(json_mod.dumps({"path": result.path, "pages": result.pages, "error": result.error}, indent=2))
        return
    if result.error:
        print_panel(result.error, title="PDF Error")
        raise typer.Exit(1) from None
    print_panel(f"{len(result.pages)} pages extracted from {path}", title="PDF")
    print_markdown(result.text[:10000])


def ocr(
    path: str = typer.Argument(..., help="Path to image file"),
    lang: str = typer.Option("eng", help="Language hint (e.g. eng, fra)"),
    json: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Run OCR on an image via Tesseract. Powered by AAFP perception."""
    perc = Perception()
    if not perc.available:
        print_panel("aafp binary not found. Build it: cd ~/Projects/aafp && cargo build --release --bin aafp", title="Error")
        raise typer.Exit(1) from None
    try:
        result = perc.ocr(path, lang=lang)
    except Exception as e:
        print_panel(str(e), title="OCR Error")
        raise typer.Exit(1) from None
    if json:
        typer.echo(json_mod.dumps(result.__dict__, indent=2))
        return
    if result.error:
        print_panel(result.error, title="OCR Error")
        raise typer.Exit(1) from None
    print_panel(f"OCR result for {path} ({result.language})", title="OCR")
    print_markdown(result.text[:10000])
