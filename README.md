# Mac AI OS

A local-first AI operating system for macOS — an intelligence layer over the
computer, not a chatbot. It reasons, retrieves, chooses tools, executes,
verifies, and explains.

## One-command bootstrap

```bash
uv sync --extra mlx --extra dev
./scripts/setup_models.sh
uv run app
```

## CLI

```bash
uv run maos verify          # verify MLX, Ollama, hardware, models
uv run maos models          # list detected models
uv run maos index <path>    # index a folder into the vector store
uv run maos ask "..."       # one-shot local prompt (planner LLM)
uv run maos rag "..."       # retrieval-augmented question
uv run maos chat            # interactive session
```

See `configs/default.yaml` and `AGENTS.md` for architecture and conventions.
