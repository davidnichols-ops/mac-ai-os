# AGENTS.md — Mac AI OS

Conventions, commands, and architecture notes for agents working on this repo.

## Quick start

```bash
uv sync --extra mlx --extra dev          # install deps (Apple Silicon)
./scripts/setup_models.sh                # verify MLX, install Ollama, pull models
uv run app                               # launch the CLI (== uv run maos)
```

## Common commands

| Task | Command |
| --- | --- |
| Run tests | `uv run pytest` |
| Lint | `uv run ruff check app tests` |
| Verify runtime | `uv run app verify` |
| List models | `uv run app models` |
| Index a folder | `uv run app index <path>` |
| One-shot prompt | `uv run app ask "..."` |
| RAG question | `uv run app rag "..."` |
| Interactive chat | `uv run app chat` |
| Run the app directly | `uv run app` (entry: `app/__main__.py`) |

## Configuration

- `configs/default.yaml` — spec target models (Qwen3 14B, gemma3 4b). Do not edit.
- `configs/user.yaml` — machine overrides (gitignored from edits; safe to tailor).
- `configs/example.yaml` — reference for switching providers.
- Empty model fields are auto-filled from `app.config.hardware.detect()`.
- Layering: code defaults < `default.yaml` < `user.yaml` < env vars.

## Architecture

```
Conversation API  ->  Agent Orchestrator  ->  { Planner LLM, Retrieval Engine, Tool Router }
                                              { MLX/Ollama models, Chroma backend, permissioned tools }
```

- `app/models/` — `Model` + `EmbeddingModel` ABCs; providers: `mlx`, `ollama`, `chroma_default`.
  Built via `app/models/registry.py`. **mlx-lm has no embed() API** — use `chroma_default`
  (ONNX MiniLM, default) or `ollama` (nomic-embed-text) for embeddings.
- `app/retrieval/` — `Retriever` ABC (`index`/`search`/`delete`/`update`); Chroma backend;
  indexers for folders, markdown, code, PDF. Indexers return chunk dicts; they never touch the vector store.
- `app/tools/` — `Tool` ABC (`execute`/`schema`/`permissions`); `ToolRouter` gates on granted
  permissions. The planner never runs raw shell — everything goes through permissioned tools.
- `app/agents/` — `Agent` ABC (`plan`/`execute`/`describe`/`health`); initial agents: repository,
  filesystem, research, system. Dependencies injected via constructor (no globals/singletons).
- `app/orchestrator.py` — routes a goal to an agent (keyword + planner-assisted), runs it.
- `app/conversation.py` — stateful wrapper keeping message history.
- `app/runtime.py` — single `build_runtime()` that wires everything from settings.
- `app/config/` — Pydantic Settings + YAML + hardware autodetect.
- `app/log.py` — structlog setup (file named `log.py` to avoid shadowing stdlib `logging`).

## Coding standards

- Python 3.12+, strongly typed, `from __future__ import annotations`.
- Composition over inheritance. Dependency injection. No global state, no singletons.
- Lazy-import heavy backends (`mlx_lm`, `ollama`, `chromadb`, `pypdf`) so the project imports
  without them and tests stay light.
- Don't add/remove comments unless asked.
- New providers: register a factory in `app/models/registry.py` (or `retrieval/registry.py`).
  Don't scatter provider knowledge across the codebase.

## Tests

- `tests/conftest.py` ships `FakeModel`, `FakeEmbeddingModel`, `FakeRetriever` so agent/
  orchestrator tests run without MLX, Ollama, or Chroma.
- Tests must pass with `uv run pytest` and no network.

## Known constraints on this host

- 16 GB RAM, ~13 GB free disk. `configs/user.yaml` uses `Llama-3.2-3B-Instruct-4bit` (~2 GB)
  instead of the spec's Qwen3 14B (~8 GB) to fit on disk. Switch back on a bigger machine.
- `mlx-lm` (0.31.3) has no embedding API → default embeddings provider is `chroma_default`.
