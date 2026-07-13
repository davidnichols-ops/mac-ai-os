"""`maos verify` — check MLX, Ollama, hardware, and model availability."""

from __future__ import annotations

import typer

from app.config import detect
from app.runtime import build_runtime
from app.ui.console import print_panel


def verify() -> None:
    """Verify MLX, Ollama, hardware detection, and model health."""
    hw = detect()
    hw_block = (
        f"macOS {hw.macos_version} ({hw.arch})\n"
        f"Model: {hw.model}\n"
        f"CPU cores: {hw.cpu_cores}\n"
        f"Memory: {hw.memory_gb} GB\n"
        f"GPU: {'yes' if hw.has_gpu else 'no'}\n"
        f"Neural Engine: {'yes' if hw.has_neural_engine else 'no'}\n"
        f"MLX capable: {'yes' if hw.can_run_mlx else 'no'}"
    )
    print_panel(hw_block, title="Hardware")

    # MLX check
    mlx_ok = False
    try:
        import mlx_lm  # type: ignore  # noqa: F401
        mlx_ok = True
    except ImportError:
        pass
    print_panel(f"mlx-lm installed: {'yes' if mlx_ok else 'no'}", title="MLX")

    # Ollama check
    ollama_ok = False
    try:
        import ollama  # type: ignore
        try:
            ollama.Client(host="http://localhost:11434").list()
            ollama_ok = True
        except Exception:
            ollama_ok = False
    except ImportError:
        pass
    print_panel(f"ollama reachable: {'yes' if ollama_ok else 'no'}", title="Ollama")

    # Build runtime (loads config + hardware defaults) and report planned models.
    try:
        rt = build_runtime(configure_logs=False)
        planned = (
            f"planner:    {rt.settings.planner.provider} / {rt.settings.planner.model}\n"
            f"utility:    {rt.settings.utility.provider} / {rt.settings.utility.model}\n"
            f"embeddings: {rt.settings.embeddings.provider} / {rt.settings.embeddings.model}\n"
            f"retrieval:  {rt.settings.retrieval.backend} (top_k={rt.settings.retrieval.top_k})"
        )
        print_panel(planned, title="Configured models")
    except Exception as e:
        print_panel(f"runtime build failed: {e}", title="Runtime")

    typer.echo("\nVerify complete.")
