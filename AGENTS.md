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
| Web search | `uv run app search "query" [--num N] [--json]` |
| Browse web page | `uv run app browse <url> [--json]` |
| Read PDF | `uv run app read-pdf <path> [--json]` |
| OCR image | `uv run app ocr <path> [--lang eng] [--json]` |
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
| Profile all workspaces | `uv run app workspace profile [--force] [--no-llm]` |
| Profile single workspace | `uv run app workspace profile <path>` |
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
| Start Context API | `uv run app serve [--port 7777]` |
| Start MCP server (stdio) | `uv run app mcp serve` |
| Start MCP server (HTTP) | `uv run app mcp serve --transport http [--port 7778]` |
| MCP status | `uv run app mcp status` |
| MCP test | `uv run app mcp test` |
| **Onboarding briefing** | `uv run app onboard` |
| Onboarding (JSON) | `uv run app onboard --json` |
| List lessons | `uv run app onboard lessons` |
| Record a lesson | `uv run app onboard record <slug> "<text>"` |
| Send email to david | `uv run app onboard email "<subject>" "<body>"` |
| Escalate (needs attention) | `uv run app onboard escalate "<subject>" "<body>"` |
| **Scan for vibes** | `uv run app guards scan "<text>"` |
| Run all guards | `uv run app guards check "<text>" --pr --bug-fix` |
| Scan from stdin (pipe) | `echo "<text>" \| uv run app guards stdin` |
| **Produce test evidence** | `uv run app guards verify "uv run pytest" --save test.json` |
| **Produce lint evidence** | `uv run app guards verify "uv run ruff check app tests" --save lint.json` |
| **Inspect git diff** | `uv run app guards diff --json` |
| **PR readiness check** | `uv run app guards readiness --test-evidence test.json --lint-evidence lint.json --pr-description pr.md` |
| **Tool diagnostics** | `uv run app diagnostics` |
| Diagnostics (JSON) | `uv run app diagnostics --json` |
| **Time awareness** | `uv run app time` |
| Time (JSON) | `uv run app time --json` |
| Bootstrap (first run) | `uv run app bootstrap [--scan-path ~/Projects]` |
| Evaluate context quality | `uv run app context evaluate` |
| Context eval history | `uv run app context history` |
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

### Phase 5 — intelligence fabric (in progress)

MAOS pivots from a standalone AI assistant to a personal context engine
that external AI agents can consume. The CLI stays, but the primary
interface becomes the Context API and MCP server.

- `app/server/` — Context API: local HTTP daemon (`maos serve`) on
  localhost:7777. Exposes 16 REST endpoints including `GET /context`
  (the universal context packet), `GET /memory/search`, `GET /tasks`,
  `POST /actions/request`. Uses stdlib `http.server` — zero new deps.
  The server holds a Runtime reference, so every request has access to
  the full system.
- `app/mcp/` — MCP Context Layer: thin adapter over the Context API
  and ActionExecutor. Exposes 5 resources and 14 tools through the
  Model Context Protocol (official `mcp` Python SDK, FastMCP API).
  Supports stdio transport (default, for Claude Code / Codex) and
  streamable-http transport. All action tools route through
  ActionExecutor — no bypass of permissions, logging, or events.
- `app/mcp/server.py` — security: localhost capability token generated
  on first run, stored at `~/.maos/token` (0600). HTTP transport
  requires `Authorization: Bearer <token>`. stdio skips token check
  (local-only by definition). Server binds to 127.0.0.1 by default.
- `app/mcp/resources.py` — 5 MCP resources: `maos://context/current`,
  `maos://workspace/current`, `maos://session/latest`,
  `maos://tasks/open`, `maos://knowledge/search{query}`. All delegate
  to `app/server/context.py` or direct DB queries. No business logic.
- `app/mcp/tools.py` — 14 MCP tools: `maos_get_context`,
  `maos_search_memory`, `maos_workspace_summary`, `maos_create_task`,
  `maos_list_tasks`, `maos_run_action`, 3 onboarding tools
  (`maos_onboard`, `maos_record_lesson`, `maos_list_lessons`),
  2 email tools (`maos_send_email`, `maos_escalate`), 2 guard
  tools (`maos_scan_vibes`, `maos_check_guards`), and 1 diagnostic
  tool (`maos_diagnostics`). Read-only tools call the Context API
  directly. Action tools go through ActionExecutor with
  `context={"db": runtime.db}` — preserving permissions, reversibility,
  action_log, and EventBus publishing.
- `app/onboarding.py` — Session onboarding system. Assembles a briefing
  for new AI sessions from: user preferences (user_model), hard-won
  lessons (lesson.* keys in user_model), recent session, active
  workspace, open tasks, and known failure patterns. Lessons are
  recorded via `maos_record_lesson` / `maos onboard record` and
  surfaced in every future briefing. This is the cold-start antidote —
  it prevents new sessions from repeating past mistakes.
- `app/gmail.py` — Email sending via Gmail OAuth. Uses credentials
  from `~/.gmail-mcp/` (account-b = david.nichols.ops@gmail.com).
  Refreshes tokens automatically. Two functions: `send_email` (generic)
  and `escalate` (the anti-vibe-overload channel — sends a [MAOS]
  prefixed email when something needs human attention). No external
  deps, just stdlib urllib + base64 + email.
- `app/guards.py` — PR governance system (evidence-producing, not
  evidence-consuming). The core principle: a boolean should represent the
  RESULT of a verification process, not REPLACE it. An agent can always
  claim `--has-tests=True`; it cannot fake a captured command execution
  tied to a commit, timestamp, and exit code.
  - **Legacy guards** (still present, for text scanning only):
    `VibeLanguageGuard`, `PrPreflightGuard`, `LessonComplianceGuard`,
    `SubagentProofGuard`. Each returns PASS/WARN/BLOCK. Use `guards scan`
    for vibe detection on arbitrary text. Do NOT use `guards check --pr`
    for PR readiness — it accepts self-attested booleans.
  - **Evidence-producing modules** (the real PR governance system):
    - `app/guards_evidence.py` — `CommandEvidence` captures exit code,
      stdout, stderr, duration, timestamp, working dir, git commit, dirty
      flag. `ManualVerificationEvidence` captures before/after states.
      `ReviewEvidence` captures human review decisions.
    - `app/guards_diff.py` — inspects `git diff` for: secrets/credentials
      (BLOCK), debug statements (WARN), MCP tool/schema changes
      (REQUIRE_REVIEW), security-sensitive paths (REQUIRE_REVIEW),
      dependency changes (WARN), binary files (WARN), scope violations
      (WARN), large diffs (WARN).
    - `app/guards_readiness.py` — PR readiness state machine:
      DRAFT → UNDERSTOOD → VERIFIED → PR_READY. Requires evidence objects,
      structured PR description, and diff inspection. If state is not
      PR_READY, do NOT open the PR.
  - **CLI**: `guards verify`, `guards diff`, `guards readiness` are the
    evidence-producing commands. `guards scan` and `guards check` are
    legacy text-scanning only.
  - Full architecture: `docs/pr-governance.md`.
- `app/diagnostics.py` — Self-diagnostic for all MAOS MCP tools. Actually
  calls each read-only tool and verifies it returns valid JSON. Side-effect
  tools (email, task creation) are checked for handler existence but not
  called. Returns per-tool status: working, degraded, broken, skipped.
  Use `maos_diagnostics` / `maos diagnostics` to verify system health.
- `docs/mcp.md` — MCP documentation: resources, tools, connecting
  external AI clients, security model.

Architecture rule: the MCP layer contains NO business logic. It is a
protocol translator. All intelligence remains in existing MAOS
components.

```
AI Client  →  MCP Adapter  →  Context API  →  MAOS Core
              (app/mcp/)      (app/server/)   (app/actions/)
                                              (app/retrieval/)
                                              (app/db.py)
```

### Phase 5.4 — external agent validation layer (complete)

Validates that MAOS improves external AI agent performance by providing
persistent context. Focus on observability, integration, and developer
experience — no new intelligence systems.

- `app/evaluation.py` — `evaluate_context()`: scores context quality across
  6 dimensions (workspace, session, tasks, knowledge, preferences, activity)
  on a 0-100 scale. Weighted average produces overall score. Results stored
  in `context_evaluations` table for trend tracking.
- `app/cli/commands/context.py` — `maos context evaluate` and
  `maos context history` commands. Shows scores, details, and trend arrows.
- `app/cli/commands/mcp.py` — added `maos mcp status` (server state,
  resources, tools, token, telemetry stats) and `maos mcp test` (connects
  to running HTTP server, calls 3 tools, verifies responses).
- `app/cli/commands/bootstrap.py` — `maos bootstrap`: first-run data
  acquisition. Runs workspace scan → shell history import → telemetry
  collection → git indexing → attention cycle. Takes empty DB to usable
  context engine. Idempotent — safe to re-run.
- MCP telemetry: `mcp_requests` table records every resource read and
  tool call (client, kind, name, success, latency, error). Two new event
  types: `MCP_RESOURCE_READ`, `MCP_TOOL_CALLED`. All MCP resources and
  tools are instrumented via `_instrument()` wrappers.
- `docs/getting-started.md` — end-user guide: installation, bootstrap,
  starting MCP server, connecting AI clients, example questions,
  troubleshooting.
- `tests/test_phase54.py` — 19 tests covering evaluation, telemetry,
  bootstrap, MCP status port check, token validation.

### Phase 5.5 — workspace profiling (complete)

Turns "28 directories with files" into "28 understood projects." For each
workspace, the profiler gathers README content, recent git commits, current
branch, uncommitted changes, TODO/FIXME comments, and open branches, then
asks the utility LLM (3B) to produce a structured profile:

  - SUMMARY: what the project is (1-2 sentences)
  - STATUS: active / abandoned / complete / maintenance
  - OPEN_WORK: discovered incomplete features
  - STACK: tech stack summary

- `app/workspace_profiler.py` — `profile_workspace()` profiles a single
  workspace; `profile_all_workspaces()` iterates all registered workspaces.
  Falls back to heuristic profiling (README + git recency) if LLM is
  unavailable. Discovered open work items are automatically created as tasks
  (with duplicate detection).
- `app/cli/commands/workspace.py` — `maos workspace profile [path] [--force]
  [--no-llm]` command. Profiles all or a single workspace. `workspace list`
  now shows project_status column.
- DB migration: `workspaces` table gains `profile`, `project_status`, and
  `profiled_at` columns.
- Context packet (`app/server/context.py`) now includes `workspace_profiles`:
  a sorted list of all profiled workspaces (active first), each with name,
  path, status, language, and truncated profile text.
- Context evaluator (`app/evaluation.py`) credits profiled workspaces —
  `with_profile` is now the biggest contributor to the workspace score (25%).
- `tests/test_phase55.py` — 28 tests covering fallback profiling, LLM
  response parsing, TODO extraction, git info gathering, task discovery,
  context packet inclusion, evaluator scoring, and DB migration.

### Phase 5.6 — perception layer (AAFP integration)

MAOS agents can now see the internet. The perception layer shells out to the
AAFP CLI (`aafp search/browse/read-pdf/ocr`) for web search, page browsing,
PDF text extraction, and image OCR. The AAFP binary must be on PATH (or at
`~/Projects/aafp/target/release/aafp`). Build it with:
`cd ~/Projects/aafp && cargo build --release --bin aafp`

- `app/perception.py` — `Perception` class wraps the AAFP CLI commands.
  Returns structured dataclasses (`SearchResult`, `BrowseResult`, `PdfResult`,
  `OcrResult`). All commands use `--json` output for reliable parsing.
- `app/agents/research.py` — `ResearchAgent` now does web search (via
  perception) + local RAG (via retriever) + LLM synthesis. Falls back to
  local-only RAG if perception is unavailable. Cites both web and local sources.
- `app/cli/commands/perception.py` — CLI commands: `maos search`, `maos browse`,
  `maos read-pdf`, `maos ocr`. Each supports `--json` for raw output.
- `app/runtime.py` — `Runtime` dataclass now includes `perception: Perception`.
  Wired into `default_agents()` for the ResearchAgent.
- `app/agents/base.py` — `Agent` ABC now accepts `perception: Perception | None`
  via constructor injection.

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
- `tests/test_server.py` — Context API HTTP server: context packet, health, 404, action request.
- `tests/test_mcp.py` — MCP resources (context, workspace, session, tasks, knowledge), MCP tools
  (action executor routing, action logging, event publishing), token generation/validation,
  server creation with registered tools.
- `tests/test_phase54.py` — context evaluation (empty DB, with data, persistence, history,
  details), MCP telemetry (tool calls, resource reads, event publishing, failures, stats),
  MCP status port check, bootstrap (workspace scan, shell history, telemetry, full run),
  token validation.
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
