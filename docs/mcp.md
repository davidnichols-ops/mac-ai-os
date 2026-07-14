# MAOS MCP Context Layer

MAOS exposes a personal context engine through the Model Context Protocol (MCP),
allowing external AI clients (Codex, Claude, ChatGPT-compatible clients, local
agents) to use MAOS as their local context backend.

## What MAOS Provides

MAOS is a local-first intelligence layer that indexes your code, documents, git
history, shell commands, and system telemetry. Through MCP, AI clients can:

- **Read context**: What am I working on? What's my active workspace? What tasks
  are pending?
- **Search memory**: Semantic search over indexed files, code, and commits.
- **Execute actions**: Create tasks, generate workspace summaries — all through
  a permissioned action executor with full audit logging.

The MCP layer is a thin adapter. All business logic lives in the existing MAOS
Context API and ActionExecutor. The MCP server contains no business logic.

## Architecture

```
AI Client  →  MCP Adapter  →  Context API  →  MAOS Core
                                (app/server/)   (app/actions/)
                                                (app/retrieval/)
                                                (app/db.py)
```

The MCP adapter (`app/mcp/`) translates between MCP protocol messages and
existing MAOS functions. It does not duplicate business logic.

## Starting the Server

### stdio transport (default, for CLI-based clients)

```bash
maos mcp serve
```

This starts the server in stdio mode, which is the standard transport for
CLI-based MCP clients like Claude Code and Codex.

### HTTP transport (for web-based clients)

```bash
maos mcp serve --transport http
maos mcp serve --transport http --port 9000 --host 127.0.0.1
```

HTTP mode runs on `http://127.0.0.1:7778/mcp` by default. A capability token
is printed on startup and must be included in requests as:

```
Authorization: Bearer <token>
```

## Available Resources

Resources are read-only data that AI clients can load into context.

### `maos://context/current`

The universal context packet. This is the primary resource — most queries
start here. Returns:

- Active workspace (name, path, language, git status)
- Current session (start, end, project, event count)
- Blackboard state (current goal, open problems, suggested actions, confidence)
- Open tasks (title, priority, workspace)
- Recent activity (event counts by type)
- Recent decisions (agent, decision, confidence)
- Related knowledge (graph entities)
- RAG results (vector search based on current goal)
- System health (CPU, memory, thermal state)

### `maos://workspace/current`

Active workspace with live git status:

- Workspace name, path, language
- Git branch
- Modified files (with count)
- Git status (clean / modified / unavailable)

### `maos://session/latest`

Most recent work session:

- Session summary (if available)
- Start and end times
- Commands run during the session
- Commits made during the session
- Events recorded during the session
- Unfinished tasks (pending across all workspaces)

### `maos://tasks/open`

All pending tasks:

- Task ID, title, status
- Priority
- Associated workspace
- Dependencies (list of task IDs)
- Whether the task is blocked by incomplete dependencies

### `maos://knowledge/search{query}`

Semantic search over indexed knowledge. Combines:

- Vector store search (RAG over indexed files, code, commits)
- Knowledge graph entity lookup

Pass a `query` parameter to search. Returns ranked results with text, source,
and relevance score.

## Available Tools

Tools perform actions. Every action tool goes through the MAOS ActionExecutor,
which preserves:

- **Permission checks**: Each action declares required permissions. The
  executor validates them before execution.
- **Reversibility checks**: Irreversible actions require explicit confirmation.
- **Action logging**: Every execution (or denial) is logged to the `action_log`
  table with the full request, result, and duration.
- **Event publishing**: `ACTION_EXECUTED` or `ACTION_DENIED` events are
  published to the EventBus and persisted to the `events` table.

No tool may bypass ActionExecutor.

### `maos_get_context()`

Get the full MAOS context packet. Equivalent to reading
`maos://context/current` but as a tool call. Returns workspace, session, tasks,
decisions, knowledge, telemetry, and blackboard state.

### `maos_search_memory(query: str, top_k: int = 5)`

Search MAOS memory (vector store) for content matching the query. Returns
ranked results with text, source path, and relevance score. Use this to find
indexed documents, code, and commit history.

### `maos_workspace_summary(workspace: str = "")`

Generate a summary of a workspace's current state. Includes git branch,
modified files, and pending tasks. Maps to the `workspace_summary` action
through ActionExecutor.

### `maos_create_task(title: str, workspace: str = "", priority: int = 0)`

Create a new TODO task. Maps to the `create_todo` action through
ActionExecutor (requires `tasks.write` permission).

### `maos_list_tasks(workspace: str = "")`

List pending tasks, optionally filtered by workspace. Maps to the `list_tasks`
action through ActionExecutor (requires `tasks.read` permission).

### `maos_run_action(action: str, target: str = "", parameters: dict = {})`

Execute any registered MAOS action by name. This is the generic gateway to
ActionExecutor. All checks, logging, and event publishing are preserved.

Available actions:

| Action | Description | Permissions |
|--------|-------------|-------------|
| `open_workspace` | Open a project directory in Finder | `filesystem.read`, `application.launch` |
| `create_todo` | Create a task | `tasks.write` |
| `complete_todo` | Mark a task as done | `tasks.write` |
| `workspace_summary` | Get workspace state summary | `filesystem.read`, `git.exec` |
| `list_tasks` | List pending tasks | `tasks.read` |

## Connecting External AI Clients

### Claude Code

Add the MAOS MCP server to Claude Code:

```bash
# stdio transport
claude mcp add maos -- uv run maos mcp serve

# Or with a specific project
cd ~/my-project
claude mcp add maos -- uv run --project ~/mac-ai-os maos mcp serve
```

Once connected, you can ask Claude:

> "What am I working on?"

Claude will call `maos_get_context()` and receive your full context packet.

### Codex / Other MCP Clients

For any MCP-compatible client, configure the server command:

```json
{
  "mcpServers": {
    "maos": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/mac-ai-os", "maos", "mcp", "serve"]
    }
  }
}
```

### HTTP transport clients

For web-based or HTTP-based clients:

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

## Security Model

### Localhost-only binding

The MCP server binds to `127.0.0.1` by default. It is not exposed to external
networks. The `--host` option exists for advanced configurations but should
not be set to `0.0.0.0` without understanding the risks.

### Capability token

A capability token is generated on first run and stored at `~/.maos/token`
with `0600` permissions (owner read/write only).

- **HTTP transport**: The token must be provided via the
  `Authorization: Bearer <token>` header. Requests without a valid token are
  rejected.
- **stdio transport**: Token validation is skipped (stdio is local-only by
  definition — the client process has direct access).

To disable the token requirement (not recommended):

```bash
maos mcp serve --transport http --no-token
```

To regenerate the token, delete `~/.maos/token` and restart the server.

### ActionExecutor as trust boundary

All action tools route through ActionExecutor, which:

1. Validates the action exists in the registry
2. Checks the caller has the required permissions
3. Checks reversibility (irreversible actions need explicit confirmation)
4. Executes the handler
5. Logs the result to `action_log` (audit trail)
6. Publishes `ACTION_EXECUTED` or `ACTION_DENIED` to the EventBus

This means even if an AI client requests a dangerous action, the executor
will refuse it if permissions are insufficient or the action is irreversible
without confirmation.

## Verification

To verify the MCP server is working:

```bash
# Run the test suite
uv run pytest tests/test_mcp.py -v

# Start the server and test with MCP Inspector
maos mcp serve --transport http --port 7778
# In another terminal:
npx -y @modelcontextprotocol/inspector
# Connect to http://localhost:7778/mcp
```

## Implementation Notes

- The MCP layer uses the official `mcp` Python SDK (v1.x, `FastMCP` API).
- The server is a thin adapter — all logic is in `app/mcp/resources.py` (which
  delegates to `app/server/context.py`) and `app/mcp/tools.py` (which delegates
  to `app/actions/executor.py`).
- No business logic is duplicated. Adding a new MAOS capability automatically
  makes it available through MCP by registering a new resource or tool.
