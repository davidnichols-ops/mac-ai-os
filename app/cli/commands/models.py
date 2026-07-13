"""`maos models` — list models from MLX cache and Ollama."""

from __future__ import annotations

import typer

from app.ui.console import print_panel


def models() -> None:
    """List locally available models from MLX and Ollama."""
    sections: list[str] = []

    try:
        import ollama  # type: ignore
        try:
            resp = ollama.Client(host="http://localhost:11434").list()
            items = resp.get("models", []) if isinstance(resp, dict) else getattr(resp, "models", [])
            names = [m.get("model", str(m)) if isinstance(m, dict) else getattr(m, "model", str(m)) for m in items]
            sections.append("Ollama:\n" + ("\n".join(f"  - {n}" for n in names) if names else "  (none)"))
        except Exception as e:
            sections.append(f"Ollama: unreachable ({e})")
    except ImportError:
        sections.append("Ollama: client not installed")

    try:
        from huggingface_hub import list_repo_files  # type: ignore # noqa
        sections.append("MLX: (models live in ~/.cache/huggingface/hub; use setup_models.sh)")
    except Exception:
        sections.append("MLX: models are cached by mlx-lm under ~/.cache/huggingface/hub")

    print_panel("\n\n".join(sections), title="Available models")
    typer.echo("Tip: run ./scripts/setup_models.sh to pull required models.")
