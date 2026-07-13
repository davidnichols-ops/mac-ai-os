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
| Index a folder (incremental) | `uv run app index <path>` |
| Force re-index | `uv run app index <path> --force` |
| Watch a directory | `uv run app watch <path>` |
| One-shot prompt | `uv run app ask "..."` |
| RAG question | `uv run app rag "..."` |
| Interactive chat | `uv run app chat` |
| Index git history | `uv run app git index <repo>` |
| Recent commits | `uv run app git log [repo]` |
| Changes since N days | `uv run app git changes [repo] --days 1` |
| Most modified files | `uv run app git files [repo]` |
| Explain commits (RAG) | `uv run app git explain "..."` |
| Telemetry status | `uv run app telemetry status` |
| Collect telemetry | `uv run app telemetry collect --seconds 30` |
| Telemetry history | `uv run app telemetry history --hours 1` |
| Activity timeline | `uv run app timeline --hours 1` |
| Event types | `uv run app event-types` |
| Graph stats | `uv run app graph stats` |
| Graph search | `uv run app graph search <name>` |
| Graph extract (LLM) | `uv run app graph extract "<topic>"` |
| Graph add relation | `uv run app graph add <src> <rel> <tgt>` |
| Session detect | `uv run app session detect` |
| Session list | `uv run app session list` |
| Session summarize | `uv run app session summarize [id]` |
| Session ask | `uv run app session ask "What was I working on?"` |
| Workspace scan | `uv run app workspace scan [path]` |
| Workspace list | `uv run app workspace list` |
| Active workspace | `uv run app workspace active` |
| Import shell history | `uv run app history import` |
| Search history | `uv run app history search "pattern"` |
| Recent commands | `uv run app history recent` |
| Add task | `uv run app task add "title" [--workspace ws]` |
| List tasks | `uv run app task list [--workspace ws]` |
| Complete task | `uv run app task done <id>` |
| List actions | `uv run app actions list` |
| Run action | `uv run app actions run <name> [target] [--params '{}']` |
| Action log | `uv run app actions log` |
| Decision log | `uv run app actions decisions` |
| Capture snapshot | `uv run app snapshot capture [workspace]` |
| List snapshots | `uv run app snapshot list` |
| Latest snapshot | `uv run app snapshot latest [workspace]` |
| Morning routine | `uv run app workflow morning` |
| Workflow runs | `uv run app workflow list` |
| Workflow details | `uv run app workflow show <id>` |
| Submit approval | `uv run app approval submit <action> [target]` |
| List approvals | `uv run app approval list` |
| Approve | `uv run app approval approve <id>` |
| Deny | `uv run app approval deny <id>` |
| Set preference | `uv run app preferences set <key> <value>` |
| Get preference | `uv run app preferences get <key>` |
| List preferences | `uv run app preferences list` |
| Show blackboard | `uv run app blackboard show` |
| Set goal | `uv run app blackboard goal "goal text"` |
| Run attention cycle | `uv run app blackboard cycle` |
| Attention status | `uv run app blackboard attention` |
| Clear blackboard | `uv run app blackboard clear` |
| Learn summary | `uv run app learn summary` |
| Record action outcome | `uv run app learn action <name> --success/--fail` |
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

Observation Layer:
  EventBus  <──  IndexingPipeline (hash-aware)  ──>  Chroma + SQLite
          <──  ContinuousIndexer (watchdog)
          <──  GitIndexer  ──>  git_commits + vector store
          <──  TelemetryCollector (psutil)  ──>  telemetry_snapshots + process_snapshots
          <──  ShellHistoryObserver  ──>  shell_commands
          ──>  events table (activity timeline)

Cognitive Layer:
  ContextBuilder  ──>  Context { session, workspace, events, commits, telemetry, commands, graph, rag }
  SessionAgent  ──>  correlates all sources  ──>  LLM synthesis  ──>  session summary
  SessionDetector  ──>  sessions table (episodes from event gaps)
  WorkspaceDetector  ──>  workspaces table (project detection)
  IntentRouter  ──>  utility model classifies query  ──>  agent selection  ──>  decision log

Agency Layer:
  ActionRegistry  ──>  Action { name, permissions, reversible, handler }
  PermissionChecker  ──>  validates required permissions
  ActionExecutor  ──>  validates → checks perms → executes → logs → publishes event
  DecisionLogger  ──>  decisions table (agent, decision, confidence, reasoning, outcome)
  ModelManager  ──>  tracks loaded models, unloads idle, manages memory pressure
  SnapshotCollector  ──>  workspace_snapshots (git state, modified files, processes)

Workflow Layer:
  WorkflowEngine  ──>  executes steps sequentially → verifies → handles failures
  MorningWorkflow  ──>  reconstruct → summarize → tasks → health → prepare → plan
  VerificationResult  ──>  success + evidence (not just "ok" but "why")
  ApprovalQueue  ──>  submit → list → approve/deny → execute
  TaskGraph  ──>  tasks with parent_id + dependencies (not just a flat list)
  UserModel  ──>  procedural memory (preferences, patterns, confidence)

Knowledge Graph:
  KnowledgeGraph  ──>  entities + relationships (SQLite)
                  ──>  LLM extraction from indexed content
```

### Phase 1 — skeleton (complete)

- `app/models/` — `Model` + `EmbeddingModel` ABCs; providers: `mlx`, `ollama`, `chroma_default`.
  Built via `app/models/registry.py`. **mlx-lm has no embed() API** — use `chroma_default`
  (ONNX MiniLM, default) or `ollama` (nomic-embed-text) for embeddings.
- `app/retrieval/` — `Retriever` ABC (`index`/`search`/`delete`/`update`); Chroma backend
  with metadata `where` filtering; indexers for folders, markdown, code, PDF.
  Indexers return chunk dicts; they never touch the vector store.
- `app/tools/` — `Tool` ABC (`execute`/`schema`/`permissions`); `ToolRouter` gates on granted
  permissions. The planner never runs raw shell — everything goes through permissioned tools.
- `app/agents/` — `Agent` ABC (`plan`/`execute`/`describe`/`health`); agents: repository,
  filesystem, research, system, git. Dependencies injected via constructor (no globals/singletons).
- `app/orchestrator.py` — routes a goal to an agent (keyword + planner-assisted), runs it.
- `app/conversation.py` — stateful wrapper keeping message history.
- `app/runtime.py` — single `build_runtime()` that wires everything from settings.
- `app/config/` — Pydantic Settings + YAML + hardware autodetect.
- `app/log.py` — structlog setup (file named `log.py` to avoid shadowing stdlib `logging`).

### Phase 2 — observation layer (complete)

- `app/db.py` — SQLite database (`data/maos.db`) with 7 tables:
  `file_hashes`, `git_commits`, `events`, `telemetry_snapshots`, `process_snapshots`,
  `entities`, `relationships`. Typed helpers for each domain.
- `app/events/` — in-process event bus (`EventBus`). Typed events (`FILE_INDEXED`,
  `FILE_SKIPPED`, `FILE_DELETED`, `GIT_REPO_INDEXED`, `MODEL_LOADED`, `SYSTEM_SNAPSHOT`,
  etc.). Synchronous dispatch, handler isolation (one crashing handler doesn't stop
  others). Auto-persists every event to the `events` table for the activity timeline.
  Wildcard subscription with `"*"`. The bus is the decoupling layer — components
  publish events without knowing who consumes them.
- `app/telemetry/collector.py` — `TelemetryCollector` runs in a background thread,
  samples system state via `psutil` every N seconds. Stores snapshots in
  `telemetry_snapshots` + top processes in `process_snapshots`. Publishes
  `SYSTEM_SNAPSHOT` events on the bus. **Honest about limitations**: GPU usage and
  Neural Engine utilization are `None` (not available without root / no public API).
  Thermal state is best-effort via `pmset -g therm`.
- `app/retrieval/pipeline.py` — `IndexingPipeline` wraps indexer + retriever + database + bus.
  Hash-aware: skips unchanged files, re-indexes changed files (delete-then-insert per file),
  removes deleted files. Publishes `FILE_INDEXED`/`FILE_SKIPPED`/`FILE_DELETED` events.
- `app/retrieval/watcher.py` — `ContinuousIndexer` uses `watchdog` to watch directories.
  Debounces events (default 2s), filters by extension and ignore rules, triggers incremental
  indexing on change, removal on delete. Runs in a background thread.
- `app/retrieval/indexers/git.py` — `GitIndexer` extracts commit-level semantic units from
  `git log --numstat`. Each commit becomes a chunk with rich metadata (commit_hash, author,
  date, files, insertions/deletions) and a structured row in `git_commits`.
- `app/agents/git.py` — `GitAgent` combines structured SQLite queries (time-based, most-modified,
  recent commits) with RAG over commit chunks (metadata-filtered by `kind=git_commit`).
- `app/knowledge/graph.py` — `KnowledgeGraph` uses the LLM to extract entities and
  relationships from indexed content, stores them in `entities` + `relationships` tables.
  JSON parsing handles markdown code fences and partial responses. Supports manual
  relationship addition and graph search with relationship traversal.
- `knowledge/` — canonical directory for indexed reference material (apple, metal, coreml,
  mlx, xnu, papers, notes, personal). Content is gitignored; structure is kept.

### Phase 3 — cognitive layer (in progress)

- `app/sessions.py` — session auto-detection from event gaps (30-min gap = new session).
  Sessions group events into episodes. `summarize_session()` gathers all data
  (events, commits, telemetry, commands, files) for a given session.
- `app/workspace.py` — workspace detection. Scans standard project directories,
  detects language from file extensions, registers in `workspaces` table.
  `detect_active_workspace()` finds the most recently active project from events.
- `app/observers/shell.py` — shell history observer. Parses `~/.zsh_history`
  (both extended `: timestamp:duration;command` and plain formats) and
  `~/.bash_history`. Incremental import (skips already-imported commands).
- `app/context.py` — `Context` dataclass + `build_context()`. Gathers all
  relevant data for a query into a single object: active session, workspace,
  recent events, commits, telemetry, commands, graph entities, RAG results.
  `to_prompt()` renders the context as a text block for LLM prompts.
- `app/agents/session.py` — `SessionAgent` (the "killer demo"). Correlates git
  commits, events, telemetry, shell commands, and knowledge graph to answer
  "What was I working on?" Uses the planner LLM with assembled context to
  synthesize a human-readable session summary. Falls back to structured
  summary if the LLM fails. Routed by the orchestrator via session keywords.

### Phase 3.5 — agency layer (complete)

- `app/actions/` — the agency layer. The LLM never executes; it requests.
  - `models.py` — `Action` (schema: name, permissions, reversible, parameters),
    `ActionRequest` (LLM output), `ActionResult` (executor output).
  - `registry.py` — `ActionRegistry` maps action names to schemas + handlers.
  - `permissions.py` — `PermissionChecker` validates required permissions.
  - `executor.py` — `ActionExecutor` is the trust boundary: validates action
    exists → checks permissions → checks reversibility → executes handler →
    logs to `action_log` table → publishes `ACTION_EXECUTED`/`ACTION_DENIED` event.
  - `builtin.py` — 5 initial safe actions: `open_workspace`, `create_todo`,
    `complete_todo`, `workspace_summary`, `list_tasks`.
- `app/router.py` — `IntentRouter` replaces keyword-only routing. Uses the
  utility model (3B) for fast, cheap classification. Falls back to keyword
  matching if the LLM is unavailable. Logs every routing decision to the
  `decisions` table with agent, confidence, and reasoning.
- `app/models/manager.py` — `ModelManager` tracks loaded models, estimated
  memory, and use count. Unloads idle models when memory pressure exceeds
  the limit. `deactivate_all()` for cleanup.
- `app/observers/snapshots.py` — `SnapshotCollector` captures periodic
  workspace state: git branch, git status, modified files, recent commands,
  active processes. Stores in `workspace_snapshots` table. Publishes
  `SNAPSHOT_CAPTURED` events. Can run as a background thread.
- `decisions` table — operational audit trail: agent, decision, reasoning,
  confidence, outcome, query. Not chain-of-thought — just the record of
  what the system decided and why.
- `tasks` table — TODO items with status, source (manual/AI), workspace,
  priority, due date.
- `action_log` table — every action request: approved, executed, result,
  error, duration.

### Phase 3 — remaining (future)

- Permissioned action layer — controlled actions through ToolRouter (done in 3.5)
- Workspace snapshots — periodic state capture (done in 3.5)

### Phase 4 — autonomous workflows (complete)

- `app/workflows/` — the workflow layer. Workflows orchestrate agents and actions
  into multi-step sequences with verification.
  - `models.py` — `Workflow` (named sequence of steps), `WorkflowStep` (handler +
    optional verification), `StepResult`, `WorkflowResult`, `VerificationResult`
    (success + evidence, not just "ok").
  - `engine.py` — `WorkflowEngine` executes steps sequentially, runs verification
    after each step, handles failures (required steps stop the workflow, optional
    steps continue), records to `workflow_runs` table, publishes events.
  - `morning.py` — the morning routine workflow: reconstruct_last_session →
    summarize_changes → check_pending_tasks → check_system_health →
    prepare_workspace → create_daily_plan. `format_morning_report()` renders
    the result as a readable "Good morning" report.
- Task graph: `tasks` table now has `parent_id` and `dependencies` columns.
  `task_tree()` returns nested structure. `task_is_blocked()` checks dependencies.
  Database migration handles existing tables (ALTER TABLE + CREATE INDEX).
- Approval queue: `approval_queue` table with submit → approve/deny → execute
  flow. Actions that need human approval are queued, not executed immediately.
- User model (procedural memory): `user_model` table stores preferences, patterns,
  and learned behaviors with confidence scores. Three memory types now:
  - Episodic: sessions, events, snapshots ("what happened")
  - Semantic: knowledge graph, RAG ("what do I know")
  - Procedural: user_model ("how do I work")
- `workflow_runs` table — records every workflow execution with status, steps
  completed, result summary, duration.

### Phase 4.5 — attention, blackboard, learning (complete)

The system now has a concept of attention and shared working memory. Instead
of rediscovering context from scratch on every interaction, the blackboard
maintains state that all capabilities inherit.

- `app/blackboard.py` — `Blackboard`: shared working memory with current goal,
  workspace, context, open problems, recent decisions, current plan, suggested
  actions, and confidence. Thread-safe. Persists to `user_model` table for
  recovery after restart. Every capability reads and writes to it.
- `app/attention.py` — `AttentionDaemon`: proactive context computation. Every
  N seconds, computes: active workspace, current goal (from pending tasks),
  system health (from telemetry), open problems (blocked tasks, failed actions,
  resource pressure), suggested actions. Updates the blackboard and persists.
  Can run as a background thread.
- `app/learning.py` — `Learner`: the final stage of the causal loop
  (Observe → Remember → Understand → Plan → Act → Verify → **Learn**).
  Records action success rates, common errors, workflow outcomes, and user
  preferences. Confidence in learned patterns grows with each observation.
  Not RL — just: action succeeded → store why → increase confidence;
  action failed → remember failure → avoid next time.

The causal loop is now complete:

```
Observe → Remember → Understand → Plan → Act → Verify → Learn
   |          |           |          |       |       |        |
   v          v           v          v       v       v        v
 events   sessions    blackboard   tasks   actions  verify  user_model
 telemetry snapshots  attention    graph   executor result   confidence
 shell    context     daemon               action_log         |
                                                                |
                                                                v
                                                    next interaction
                                                    inherits context
```

## Coding standards

- Python 3.12+, strongly typed, `from __future__ import annotations`.
- Composition over inheritance. Dependency injection. No global state, no singletons.
- Lazy-import heavy backends (`mlx_lm`, `ollama`, `chromadb`, `pypdf`, `watchdog`) so the
  project imports without them and tests stay light.
- Don't add/remove comments unless asked.
- New providers: register a factory in `app/models/registry.py` (or `retrieval/registry.py`).
  Don't scatter provider knowledge across the codebase.
- New agents: extend `Agent` ABC, register in `app/agents/registry.py`, add routing hints
  in `app/orchestrator.py` if keyword routing should recognize them.

## Tests

- `tests/conftest.py` ships `FakeModel`, `FakeEmbeddingModel`, `FakeRetriever` so agent/
  orchestrator tests run without MLX, Ollama, or Chroma. `FakeRetriever.search` supports
  `where` metadata filtering.
- `tests/test_db.py` — SQLite file_hashes and git_commits helpers.
- `tests/test_pipeline.py` — hash-aware indexing (index, skip, force, remove).
- `tests/test_git.py` — GitIndexer parsing, GitAgent structured + RAG answers (uses a
  temporary git repo with real commits).
- `tests/test_events.py` — event bus subscribe/publish, wildcard, persistence, handler
  isolation, unsubscribe.
- `tests/test_telemetry.py` — snapshot collection, storage, history queries.
- `tests/test_graph.py` — entity/relationship storage, LLM extraction, graph search.
- `tests/test_sessions.py` — session storage, auto-detection from event gaps, summarization.
- `tests/test_workspace.py` — project detection, workspace storage, active workspace detection.
- `tests/test_shell_history.py` — zsh history parsing (extended + plain), incremental import, search.
- `tests/test_session_agent.py` — session agent structured query, matching, context builder.
- `tests/test_actions.py` — action registry, permissions, executor, built-in actions, decision log.
- `tests/test_snapshots.py` — snapshot storage, capture with git state, batch capture.
- `tests/test_router.py` — keyword routing, LLM routing, decision logging.
- `tests/test_model_manager.py` — model lifecycle, memory pressure, deactivate all.
- `tests/test_workflows.py` — workflow engine, step execution, verification, failure handling, morning routine.
- `tests/test_task_graph.py` — task parent/child, dependencies, blocking, approval queue, user model.
- `tests/test_blackboard.py` — blackboard state, persistence, thread safety, attention cycle, learner.
- Tests must pass with `uv run pytest` and no network.

## Known constraints on this host

- 16 GB RAM, 47 GB free disk. `configs/user.yaml` now uses `Qwen3-14B-4bit` (~8 GB)
  as the planner (spec target) with `Llama-3.2-3B-Instruct-4bit` as the utility model.
  The 14B model uses ~9-10 GB RAM when loaded — close to the 16 GB limit.
- `mlx-lm` (0.31.3) has no embedding API → default embeddings provider is `chroma_default`.
- `datetime.utcnow()` is deprecated in Python 3.13+ → use `datetime.now(timezone.utc)`.
- GPU and Neural Engine utilization are not available without root (`powermetrics` needs `sudo`).
- zsh history may not have extended timestamps (`: timestamp:duration;command`) — the parser
  handles both extended and plain formats (uses file mtime for plain format).
