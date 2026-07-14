# Getting Started with MAOS

This guide is for someone who has installed MAOS and wants to connect an
external AI agent (Codex, Claude, ChatGPT, local LLM) to use MAOS as their
personal context backend.

## Prerequisites

- macOS on Apple Silicon (MLX models)
- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Ollama (optional, for non-MLX models)

## Installation

```bash
# Clone the repo
git clone <repo-url> ~/mac-ai-os
cd ~/mac-ai-os

# Install dependencies
uv sync --extra mlx --extra dev

# Verify the install
uv run maos verify
```

## Bootstrap (First Run)

After installation, your database is empty. Run bootstrap to populate it:

```bash
uv run maos bootstrap
```

This will:
1. **Scan for workspaces** — discovers project directories in `~/Projects`
2. **Import shell history** — loads your `~/.zsh_history` into the database
3. **Collect telemetry** — samples CPU, memory, and processes for 10 seconds
4. **Index git history** — indexes commits from discovered git repos
5. **Generate initial context** — runs an attention cycle to populate the
   blackboard with your current goal and workspace

You can customize the bootstrap:

```bash
# Scan a different directory
uv run maos bootstrap --scan-path ~/Developer

# Index specific git repos
uv run maos bootstrap --git-repo ~/my-project --git-repo ~/other-project

# Skip telemetry (faster)
uv run maos bootstrap --skip-telemetry

# Skip everything except workspace scan
uv run maos bootstrap --skip-shell --skip-telemetry --skip-git
```

After bootstrap, check your context quality:

```bash
uv run maos context evaluate
```

You should see scores for each dimension. If any are low, run the
specific commands to populate that data:

```bash
uv run maos workspace scan          # workspace awareness
uv run maos history import          # shell history
uv run maos telemetry collect -s 60 # telemetry
uv run maos git index ~/my-repo     # git history
uv run maos task add "My task"      # task awareness
uv run maos graph extract "topic"   # knowledge graph
```

## Starting the MCP Server

MAOS exposes context through the Model Context Protocol (MCP). Start the
server:

### For CLI-based clients (Claude Code, Codex)

```bash
uv run maos mcp serve
```

This uses stdio transport — the AI client spawns MAOS as a subprocess and
communicates over stdin/stdout.

### For web-based or HTTP clients

```bash
uv run maos mcp serve --transport http
```

This starts an HTTP server on `http://127.0.0.1:7778/mcp`. A capability
token is printed on startup and stored at `~/.maos/token`.

### Check server status

```bash
uv run maos mcp status
```

Shows whether the server is running, registered resources and tools,
token status, and MCP telemetry stats.

### Test the server

```bash
# Start the server in one terminal
uv run maos mcp serve --transport http &

# Test it in another terminal
uv run maos mcp test
```

This connects to the running server, calls `maos_get_context()`,
`maos_search_memory()`, and `maos_list_tasks()`, and verifies the
responses are valid.

## Connecting an AI Client

### Claude Code

```bash
claude mcp add maos -- uv run --project ~/mac-ai-os maos mcp serve
```

### Codex / Generic MCP Client

Add to your client's MCP configuration:

```json
{
  "mcpServers": {
    "maos": {
      "command": "uv",
      "args": ["run", "--project", "/Users/you/mac-ai-os", "maos", "mcp", "serve"]
    }
  }
}
```

### HTTP-based Clients

```json
{
  "mcpServers": {
    "maos": {
      "url": "http://127.0.0.1:7778/mcp",
      "headers": {
        "Authorization": "Bearer <token-from-~/.maos/token>"
      }
    }
  }
}
```

## Example Questions

Once connected, try asking your AI client:

### "What should I work on today?"

The client calls `maos_get_context()` and receives:
- Your active workspace and git branch
- Your current goal (from the blackboard)
- Pending tasks with priorities
- Recent activity and decisions
- Suggested next actions

### "What was I working on last?"

The client reads `maos://session/latest` and receives:
- The most recent session summary
- Commands you ran
- Commits you made
- Unfinished tasks

### "Search my codebase for X"

The client calls `maos_search_memory(query="X")` and receives ranked
results from your indexed files, code, and commit history.

### "Create a task to fix the login bug"

The client calls `maos_create_task(title="fix the login bug")` which goes
through the MAOS ActionExecutor — permissions are checked, the action is
logged, and an event is published.

## How It Works

```
You install MAOS
    ↓
maos bootstrap → populates database (workspaces, history, telemetry, git)
    ↓
maos mcp serve → starts MCP server (stdio or HTTP)
    ↓
AI client connects → calls maos_get_context() or reads resources
    ↓
MAOS assembles context from: SQLite + Chroma + Blackboard + Knowledge Graph
    ↓
AI client receives context packet → uses it to answer your question
```

The AI client doesn't need to know anything about MAOS internals. It just
calls tools and reads resources through MCP.

## Security

- The MCP server binds to `127.0.0.1` only — no external network exposure
- A capability token is generated at `~/.maos/token` (0600 permissions)
- HTTP transport requires `Authorization: Bearer <token>` header
- All action tools go through ActionExecutor (permission checks, logging,
  event publishing)

## Troubleshooting

### "No MCP server running"

Start the server first:
```bash
uv run maos mcp serve --transport http
```

### "Context quality scores are low"

Run `maos bootstrap` again, or run individual commands:
```bash
uv run maos workspace scan
uv run maos history import
uv run maos git index ~/my-project
```

### "Token not found"

The token is generated on first server start. Or generate it manually:
```bash
uv run maos mcp serve --transport http  # generates token on startup
cat ~/.maos/token
```

### "MCP test fails"

Check that the server is running and the token is correct:
```bash
uv run maos mcp status
uv run maos mcp test
```

## Next Steps

- Read [docs/mcp.md](mcp.md) for the full MCP API reference
- Read [docs/operators-manual.md](operators-manual.md) for all CLI commands
- Run `maos context evaluate` periodically to track context quality over time
- Run `maos context history` to see trends
