# Mac AI OS (MAOS)

A local-first AI operating system for macOS — an intelligence layer over the
computer, not a chatbot. It reasons, retrieves, chooses tools, executes,
verifies, and explains, all while keeping your data on-device.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## What is MAOS?

MAOS is a personal, local-first agent OS for macOS. It runs on your machine,
uses MLX on Apple Silicon, falls back to Ollama, and exposes its capabilities
through a CLI, an MCP server, and a set of composable Python services.

Key design goals:

- **Local-first** — models and data stay on your Mac by default.
- **Tool-native** — the runtime can call tools, inspect results, and iterate.
- **Verifiable** — guards, evidence capture, and diagnostics are first-class.
- **MCP-ready** — exposes `maos_*` tools to any MCP-compatible client.
- **Provider-agnostic** — works with Ollama, MLX, OpenRouter, and more.

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/davidnichols-ops/mac-ai-os.git
cd mac-ai-os
uv sync --extra mlx --extra dev
```

[`uv`](https://github.com/astral-sh/uv) is the recommended Python package and
project manager. If you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set up models

```bash
./scripts/setup_models.sh
```

This installs `uv`, creates a `.venv`, pulls Ollama/MLX models, and runs a
first-inference benchmark.

### 3. Copy the user config (optional)

```bash
cp configs/example.yaml configs/user.yaml
# edit configs/user.yaml to override models
```

`configs/user.yaml` is gitignored. `configs/default.yaml` is the safe baseline.

### 4. Run MAOS

```bash
uv run maos --help
uv run maos verify        # verify MLX, Ollama, hardware, models
uv run maos onboard       # get the user briefing + lessons
uv run maos refresh       # run the unified context refresh
uv run maos task list     # check open tasks
```

You can also use `uv run app` as a legacy alias.

## CLI overview

| Command | What it does |
| --- | --- |
| `maos verify` | Check hardware, models, and integrations |
| `maos onboard` | Load user briefing, lessons, and active context |
| `maos refresh` | Refresh workspace, memory, and indexes |
| `maos task list` | Show open tasks |
| `maos ask "..."` | One-shot local prompt |
| `maos rag "..."` | Retrieval-augmented question |
| `maos chat` | Interactive session |
| `maos mcp serve` | Start the MCP server |
| `maos guards ...` | Run evidence-producing verifications |

Run `uv run maos --help` for the full command tree.

## Architecture

```
mac-ai-os/
├── app/                # Core runtime (CLI, models, MCP, agents)
├── configs/            # Configuration templates
├── data/               # Runtime data (ignored, except .gitkeep files)
├── docs/               # Architecture, dogfooding, integration guides
├── engineering-intelligence/  # Research artifacts and security campaigns
├── training/           # Model training pipeline and experiments
├── eval/               # Evaluation harness
├── scripts/            # Setup and helper scripts
└── tests/              # Pytest suite
```

See `docs/architecture.md` and `AGENTS.md` for a deeper walkthrough.

## Configuration

Configuration layers (later wins):

1. Code defaults
2. `configs/default.yaml`
3. `configs/user.yaml` (gitignored, your local overrides)
4. Environment variables (e.g. `OPENROUTER_API_KEY`)

Copy `configs/example.yaml` to `configs/user.yaml` to get started.

## MCP server

MAOS exposes tools via the Model Context Protocol:

```bash
uv run maos mcp serve --transport http --port 7778
```

Clients can then call tools like `maos_get_context`, `maos_list_tasks`,
`maos_search_memory`, and `maos_run_subagent`.

## Testing and linting

```bash
uv run pytest
uv run ruff check app tests
```

## Security and secrets

- Keep API keys in a `.env` file (not in any tracked file). `.env` is
  gitignored.
- Use `configs/user.yaml` for model/provider preferences, not credentials.
- Review `AGENTS.md` for the security and verification contract.

## Training (optional)

The `training/` directory contains the v4 pipeline for fine-tuning an
`adaptive-operator` model. It is Apple-Silicon-oriented and depends on extra
packages (`mlx`, `transformers`, `peft`, `together`). See
`training/v4_pipeline/PIPELINE_ARCHITECTURE.md` for details.

## License

MIT — see `LICENSE`.

## Contributing

This is a personal research project that is opening up for visibility and
feedback. Issues and discussions are welcome. If you run into setup problems on
non-Apple Silicon hardware, the Ollama fallback path should still work.
