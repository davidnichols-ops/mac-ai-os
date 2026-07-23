# Phase 0: Architectural Reconnaissance — MAOS Workflow Runtime Evolution

**Status**: Complete
**Date**: 2026-08-18
**Scope**: Map current architecture, workflow lifecycle, and architectural debt as the foundation for evolving MAOS into a durable, stateful, interruptible, recoverable workflow runtime.

---

## Output A: Current Architecture Map

### A.1 Component Inventory (by layer)

#### Observation Layer (inputs → events → state)

| Component | File | Dependencies | State (in-memory) | State (persisted) | Events Published | External Effects |
|---|---|---|---|---|---|---|
| `IndexingPipeline` | `app/retrieval/pipeline.py` | Retriever, Database, EventBus | none | `file_hashes` | `FILE_INDEXED`, `FILE_SKIPPED`, `FILE_DELETED` | Chroma writes |
| `ContinuousIndexer` | `app/retrieval/watcher.py` | IndexingPipeline, watchdog | `_pending: dict[path→(ts, deleted)]`, `_thread`, `_stop_event` | none | `WATCHER_STARTED`, `WATCHER_STOPPED` | filesystem reads |
| `GitIndexer` | `app/retrieval/indexers/git.py` | subprocess git | none | `git_commits` | (via pipeline) | git subprocess |
| `TelemetryCollector` | `app/telemetry/collector.py` | psutil, Database, EventBus | `_thread`, `_stop_event` | `telemetry_snapshots`, `process_snapshots` | `SYSTEM_SNAPSHOT` | psutil reads, `pmset` subprocess |
| `SnapshotCollector` | `app/observers/snapshots.py` | Database, EventBus, psutil, subprocess | `_thread`, `_stop_event` | `workspace_snapshots` | `SNAPSHOT_CAPTURED` | git subprocess, psutil |
| `ShellHistoryObserver` | `app/observers/shell.py` | Database | none | `shell_commands` | (none direct) | reads `~/.zsh_history` |
| `Perception` | `app/perception.py` | AAFP CLI binary | none | none | (none) | subprocess to AAFP (web search, browse, PDF, OCR) |

#### Cognitive Layer (state → context → decisions)

| Component | File | Dependencies | State (in-memory) | State (persisted) | Events Published | External Effects |
|---|---|---|---|---|---|---|
| `Context` / `build_context()` | `app/context.py` | Database, Retriever, workspace detection | ephemeral per query | none | none | none |
| `IntentRouter` | `app/router.py` | utility Model, Database | none | `decisions` | `DECISION_LOGGED` (via db.log_decision) | LLM call |
| `KnowledgeGraph` | `app/knowledge/graph.py` | Database, planner Model | none | `entities`, `relationships` | `ENTITY_EXTRACTED` | LLM call |
| `Blackboard` | `app/blackboard.py` | threading.Lock | `current_goal`, `current_workspace`, `current_context`, `open_problems`, `recent_decisions`, `current_plan`, `suggested_actions`, `confidence` | `user_model[_blackboard_state]` | none | none |
| `AttentionDaemon` | `app/attention.py` | Database, Blackboard | `_thread`, `_stop_event` | (writes via Blackboard.persist) | none | none |
| `Learner` | `app/learning.py` | Database | none | `user_model` (action success rates, preferences, workflow outcomes) | none | none |

#### Agency Layer (decisions → authorized actions → effects)

| Component | File | Dependencies | State (in-memory) | State (persisted) | Events Published | External Effects |
|---|---|---|---|---|---|---|
| `ActionRegistry` | `app/actions/registry.py` | none | `_actions`, `_handlers` dicts | none | none | none |
| `PermissionChecker` | `app/actions/permissions.py` | none | `granted: set[str]` | none | none | none |
| `ActionExecutor` (**trust boundary**) | `app/actions/executor.py` | Registry, Permissions, Database, EventBus | `allow_irreversible: bool` | `action_log` | `ACTION_EXECUTED`, `ACTION_DENIED` | delegates to action handlers |
| `ToolRouter` | `app/tools/registry.py` | Tool instances | `_granted: set[str]` | none | none | shell exec, file read/write, git (via tools) |
| `AgentOrchestrator` | `app/orchestrator.py` | planner Model, agents dict, ToolRouter | none | (via agent.execute) | none | LLM calls, tool calls |

#### Workflow Layer (the evolution target)

| Component | File | Dependencies | State (in-memory) | State (persisted) | Events Published | External Effects |
|---|---|---|---|---|---|---|
| `WorkflowEngine` | `app/workflows/engine.py` | Database, EventBus | none | `workflow_runs` | `WORKFLOW_STARTED`, `WORKFLOW_STEP_COMPLETED`, `WORKFLOW_STEP_FAILED`, `WORKFLOW_COMPLETED` | delegates to step handlers |
| `Workflow` / `WorkflowStep` / `StepResult` / `WorkflowResult` / `VerificationResult` | `app/workflows/models.py` | none (dataclasses) | ephemeral | none | none | none |
| `morning_workflow()` | `app/workflows/morning.py` | Database, agents, actions | none (rebuilt each call) | none (writes via handlers) | (via engine) | git reads, LLM calls, task writes |

#### External Interface Layer

| Component | File | Dependencies | State (in-memory) | State (persisted) | Events Published | External Effects |
|---|---|---|---|---|---|---|
| `create_mcp_server()` | `app/mcp/server.py` | Runtime, FastMCP | token at `~/.maos/token` | none | (via tool instrumentation) | MCP stdio/HTTP |
| MCP Resources (6) | `app/mcp/resources.py` | Runtime | none | `mcp_requests` | `MCP_RESOURCE_READ` | none |
| MCP Tools (16) | `app/mcp/tools.py` | Runtime, ActionExecutor | none | `mcp_requests` | `MCP_TOOL_CALLED` | delegates to actions/tools |
| Context API HTTP server | `app/server/app.py` | Runtime | none | (via handlers) | none | HTTP server on 127.0.0.1:7777 |
| `GmailSender` | `app/gmail.py` | OAuth creds at `~/.gmail-mcp/` | none | none | (via ActionExecutor) | **sends real emails** |

#### Infrastructure Layer

| Component | File | Dependencies | State (in-memory) | State (persisted) | Events Published | External Effects |
|---|---|---|---|---|---|---|
| `Database` | `app/db.py` | sqlite3 | connection, WAL mode | 20 tables in `data/maos.db` | none | file I/O |
| `EventBus` | `app/events/bus.py` | Database (optional) | `_handlers: dict[type→list]`, `_wildcard_handlers`, `RLock` | `events` table | none | none |
| `ModelManager` | `app/models/manager.py` | Model instances | `_models: dict[name→entry]` with last-used timestamps | none | `MODEL_LOADED` | GPU/memory allocation |
| `build_runtime()` | `app/runtime.py` | all of the above | assembles `Runtime` dataclass | restores Blackboard | none | none |

### A.2 Database Schema (20 tables)

| Table | Purpose | Key Relationships |
|---|---|---|
| `file_hashes` | Incremental indexing skip-unchanged | PK (path, source) |
| `git_commits` | Structured commit metadata | PK (repo, hash) |
| `events` | Activity timeline (all EventBus events) | auto-increment id |
| `telemetry_snapshots` | System health time-series | → `process_snapshots` (1:N) |
| `process_snapshots` | Top processes per snapshot | FK → telemetry_snapshots (CASCADE) |
| `entities` | Knowledge graph nodes | PK id, UNIQUE name |
| `relationships` | Knowledge graph edges | FK → entities (CASCADE, both ends) |
| `sessions` | Work episodes (event-gap detection) | none |
| `workspaces` | Detected projects | UNIQUE path |
| `shell_commands` | Shell history | FK → sessions (SET NULL) |
| `decisions` | Routing/decision audit trail | none |
| `tasks` | TODOs with hierarchy + deps | FK → tasks (CASCADE, parent_id) |
| `approval_queue` | Pending human approvals | `workflow_id` (no FK enforced) |
| `user_model` | Procedural memory (prefs, lessons, blackboard) | UNIQUE key |
| `workflow_runs` | Workflow execution history | none (no FK to anything) |
| `workspace_snapshots` | Periodic workspace state | FK → telemetry_snapshots (SET NULL) |
| `action_log` | Every action request (approved/denied/executed) | none |
| `mcp_requests` | MCP tool/resource call telemetry | none |
| `context_evaluations` | Context quality scores | none |
| `file_reads` | File reading behavior tracking | FK → sessions (SET NULL) |

**Notable schema gaps:**
- `workflow_runs` has no FK to any step-level table (step results are JSON-blobbed into `result` column)
- `approval_queue.workflow_id` is an integer but has no FK constraint to `workflow_runs.id`
- No table for workflow *definitions* (only runs)
- No table for step-level execution records
- No table for checkpoints or snapshots of workflow context

### A.3 Event Taxonomy (28 event types)

**Indexing**: `FILE_INDEXED`, `FILE_SKIPPED`, `FILE_DELETED`, `INDEX_UPDATED`
**Git**: `GIT_COMMIT_INDEXED`, `GIT_REPO_INDEXED`
**Models**: `MODEL_LOADED`, `MODEL_GENERATED`
**System**: `SYSTEM_SNAPSHOT`, `COMMAND_RUN`, `WATCHER_STARTED`, `WATCHER_STOPPED`
**Knowledge**: `ENTITY_EXTRACTED`
**Conversation**: `CONVERSATION_MESSAGE`
**Actions**: `ACTION_EXECUTED`, `ACTION_DENIED`
**Tasks**: `TASK_CREATED`, `TASK_COMPLETED`
**Snapshots**: `SNAPSHOT_CAPTURED`
**Decisions**: `DECISION_LOGGED`
**Workflows**: `WORKFLOW_STARTED`, `WORKFLOW_STEP_COMPLETED`, `WORKFLOW_STEP_FAILED`, `WORKFLOW_COMPLETED`
**MCP**: `MCP_RESOURCE_READ`, `MCP_TOOL_CALLED`
**Sessions**: `SESSION_STARTED`, `SESSION_ENDED`

**EventBus semantics:**
- Events persisted to `events` table BEFORE handler dispatch (crash-safe audit)
- Handler isolation: try/except per handler, one crash doesn't stop others
- Wildcard `"*"` subscription supported
- Thread-safe via RLock
- Synchronous dispatch (no async, no queues, no backpressure)

### A.4 Trust Boundary Map

```
External Caller (Devin/LLM/HTTP)
     │
     ▼
  MCP Tool (app/mcp/tools.py)        ──instrumented──→ mcp_requests table + MCP_TOOL_CALLED event
     │
     ▼
  ActionExecutor (app/actions/executor.py)   ◄── THE TRUST BOUNDARY
     │  1. Action exists? (registry lookup)
     │  2. Permissions satisfied? (PermissionChecker)
     │  3. Reversible OR explicitly allowed?
     │  4. Execute handler
     │  5. Log to action_log
     │  6. Publish ACTION_EXECUTED/ACTION_DENIED
     ▼
  Action Handler (app/actions/builtin.py)
     │
     ▼
  External Effect (subprocess, db write, email)
```

**Trustcard integration:** Trustcard is NOT integrated into the MAOS codebase. It exists only at the Devin config level — `mcp-proxy.js` wraps the MAOS MCP server with a manifest at `~/.config/devin/manifests/maos.json`. MAOS's own trust boundary is `ActionExecutor` with string-based permissions. There is no manifest-based enforcement, no tool poisoning detection, no schema shadowing defense within MAOS itself.

### A.5 Runtime Assembly Order

`build_runtime()` in `app/runtime.py` assembles components in this order:
1. Settings → 2. Models (planner, utility, embedding) → 3. Retriever (Chroma) → 4. Database → 5. EventBus → 6. KnowledgeGraph → 7. ToolRouter → 8. Perception → 9. Agents → 10. Orchestrator → 11. ActionRegistry → 12. PermissionChecker → 13. ActionExecutor → 14. IntentRouter → 15. ModelManager → 16. WorkflowEngine → 17. Blackboard (restored from DB) → 18. AttentionDaemon → 19. Learner → 20. Runtime dataclass

**Granted permissions (hardcoded at build time):**
- Actions: `filesystem.read`, `application.launch`, `tasks.read`, `tasks.write`, `git.exec`
- Tools: `shell.exec`, `file.read`, `file.write`, `git.exec`

**No dynamic permission changes. No per-workflow permission scoping. No permission persistence.**

---

## Output B: Current Workflow Lifecycle

### B.1 States and Transitions

```
                    ┌─────────────┐
                    │  (not in DB) │  Workflow object exists in memory only
                    │  constructed │  (morning_workflow() builds it fresh each call)
                    └──────┬──────┘
                           │ WorkflowEngine.run(workflow, context) called
                           ▼
                    ┌─────────────┐
                    │   running    │  db.start_workflow_run() → status='running', steps_done=0
                    │              │  Event: WORKFLOW_STARTED
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
              ▼            ▼            ▼
     ┌────────────┐ ┌───────────┐ ┌───────────┐
     │ step ok    │ │ step fail │ │ step fail │
     │ (verify    │ │ (required)│ │ (optional)│
     │  passes)   │ │           │ │           │
     └─────┬──────┘ └─────┬─────┘ └─────┬─────┘
           │              │             │
           │              │ continue    │ continue
           │              │ to next     │ to next
           │              ▼             ▼
           │        ┌─────────────┐
           │        │   failed    │  db.finish_workflow_run(status='failed')
           │        │             │  Event: WORKFLOW_COMPLETED (with ok=False)
           │        └─────────────┘
           │
           ▼ (all steps done)
    ┌─────────────┐
    │  completed  │  db.finish_workflow_run(status='completed')
    │             │  Event: WORKFLOW_COMPLETED (with ok=True)
    └─────────────┘
```

### B.2 Step Execution Detail

For each step in `workflow.steps`:
1. Record `step_start = time.time()`
2. Create `StepResult(step_name=step.name, ok=False)`
3. **Try:** `data = step.handler(ctx)` where `ctx` is the merged context dict
4. Set `step_result.ok = bool(data.get("ok", True))`
5. Set `step_result.data = data`
6. **If ok:** merge `data` into `ctx` (subsequent steps see prior outputs)
7. **If verify callable exists AND step ok:** `verification = step.verify({**ctx, **data})`
   - If `verification.success == False`: step ok = False, error = "verification failed: ..."
8. **Except:** step ok = False, error = str(exception)
9. Record `step_result.duration_ms`
10. `db.update_workflow_run(run_id, i + 1)` — updates `steps_done` only
11. Publish `WORKFLOW_STEP_COMPLETED` or `WORKFLOW_STEP_FAILED`
12. **If step failed AND step.required:** set result.ok = False, break loop
13. **If step failed AND NOT step.required:** continue to next step

### B.3 What is Persisted vs. Lost

**Persisted to `workflow_runs` table:**
- `id`, `timestamp`, `workflow` (name), `status` ('running'|'completed'|'failed')
- `steps_total`, `steps_done` (integer count only)
- `result` (JSON blob of entire WorkflowResult — written only at end)
- `error` (string, written only at end)
- `duration_ms` (written only at end)

**NOT persisted (lost on crash):**
- Per-step results during execution (only `steps_done` count is updated mid-run)
- Workflow context (the `ctx` dict that accumulates across steps)
- Workflow definition (the `Workflow` object is rebuilt fresh each call)
- Which step is currently executing
- Verification results per step
- Step durations per step

**Implication:** If MAOS crashes mid-workflow, the `workflow_runs` row will be stuck at `status='running'` forever. There is no recovery mechanism, no way to resume from the last completed step, no way to know which step failed, and no way to inspect intermediate context.

### B.4 The Morning Workflow (only concrete workflow)

`morning_workflow()` in `app/workflows/morning.py` builds a `Workflow` with 6 steps:

| # | Step | Required | Handler | Verify |
|---|---|---|---|---|
| 1 | `reconstruct_last_session` | No | SessionAgent → session summary | checks session exists |
| 2 | `summarize_changes` | No | git log since last session | checks commits returned |
| 3 | `check_pending_tasks` | No | db query for pending tasks | checks query succeeded |
| 4 | `check_system_health` | No | telemetry snapshot | checks telemetry exists |
| 5 | `prepare_workspace` | No | detect active workspace | checks workspace detected |
| 6 | `create_daily_plan` | **Yes** | LLM generates plan from context | none |

All steps except `create_daily_plan` are optional (`required=False`), meaning the workflow can "succeed" even if session reconstruction, git summarization, task checking, health monitoring, and workspace detection all fail. Only the final LLM plan generation is required.

---

## Output C: Architectural Debt Report

### C.1 Critical Gaps (block durable workflows)

#### C.1.1 No Step-Level Persistence
**Problem:** `workflow_runs` stores only `steps_done` (an integer count) during execution. Individual `StepResult` objects (with data, verification, error, duration) are only kept in memory and written as a JSON blob at the end.
**Impact:** Crash mid-workflow = total loss of step history. Cannot resume. Cannot inspect what happened. Cannot debug failures.
**Fix:** Add `workflow_step_runs` table with per-step records (step_name, status, data, verification, error, duration, started_at, finished_at).

#### C.1.2 No Workflow Definition Persistence
**Problem:** `Workflow` objects are built fresh in code each time (`morning_workflow()` reconstructs the entire workflow on every call). There is no `workflows` table.
**Impact:** Cannot version workflows. Cannot reference a workflow by ID. Cannot pause and resume because the definition doesn't exist outside memory. Cannot have dynamic/workflow-as-data patterns.
**Fix:** Add `workflow_definitions` table (id, name, version, definition_json, created_at).

#### C.1.3 No Resumability / Checkpointing
**Problem:** No checkpoint mechanism. No way to save workflow context mid-execution. No way to resume from a specific step.
**Impact:** Long-running workflows are fragile. Crash = restart from scratch. No pause/resume for human-in-the-loop steps.
**Fix:** Add checkpoint records (workflow_run_id, step_index, context_snapshot_json, created_at). Add `resume(run_id, from_step=None)` to WorkflowEngine.

#### C.1.4 No Retry Mechanism
**Problem:** Steps execute exactly once. No retry count, no backoff, no retry policy.
**Impact:** Transient failures (network blip, LLM timeout, file lock) kill the step permanently.
**Fix:** Add `retry_count`, `retry_delay_ms`, `max_retries` to `WorkflowStep`. Implement retry loop in engine.

#### C.1.5 No Compensation / Rollback
**Problem:** When a step fails, there is no cleanup. If step 3 sent an email and step 4 failed, the email is already sent with no way to undo.
**Impact:** Partial workflow execution leaves the system in an inconsistent state with external side effects.
**Fix:** Add `compensation_handler` to `WorkflowStep`. Execute compensations on failure for already-completed steps.

#### C.1.6 Stale `running` Status on Crash
**Problem:** If MAOS crashes during a workflow, `workflow_runs.status` stays `'running'` forever. No cleanup, no detection, no recovery.
**Impact:** Orphaned runs accumulate. Status queries are misleading. No way to distinguish "actually running" from "crashed while running."
**Fix:** On startup, scan for `status='running'` runs with no heartbeat. Mark as `'crashed'` or `'interrupted'`. Add heartbeat column.

### C.2 Structural Debt

#### C.2.1 Workflow Context is an Unstructured Dict
**Problem:** `ctx: dict[str, Any]` accumulates data across steps. No schema, no type safety, no validation. Step 2 silently overwrites step 1's keys if they collide.
**Impact:** Fragile, hard to debug, impossible to validate. Step contracts are implicit.
**Fix:** Define `WorkflowContext` as a typed dataclass or Pydantic model with per-step input/output contracts.

#### C.2.2 No Step Identity
**Problem:** Steps are identified only by `name` (a string). No stable ID. If two steps have the same name, they're indistinguishable in logs and DB.
**Impact:** Cannot reliably reference a specific step execution. Cannot track step versions.
**Fix:** Add `step_id: str` (UUID or deterministic hash of name+workflow_version).

#### C.2.3 No Workflow Composition
**Problem:** A `WorkflowStep.handler` is a callable. It cannot be "run this other workflow." No sub-workflow support, no nesting.
**Impact:** Cannot build complex workflows from simpler ones. Everything must be flat.
**Fix:** Allow `WorkflowStep.action` to reference another workflow by ID. Implement sub-workflow execution with context isolation.

#### C.2.4 No Conditional Branching / Parallelism
**Problem:** Steps run sequentially in list order. No `if`/`else`, no parallel groups, no loops.
**Impact:** Cannot express non-trivial control flow. All workflows are linear pipelines.
**Fix:** Add `condition: Callable[[ctx], bool]` to skip steps. Add `parallel_group: str` for concurrent execution. (Longer term: consider a DAG-based engine.)

#### C.2.5 No Workflow Cancellation
**Problem:** Once `WorkflowEngine.run()` starts, there is no way to cancel it from outside. No cancellation token, no interrupt mechanism.
**Impact:** A stuck step (e.g., LLM hang) blocks forever. No timeout per step.
**Fix:** Add per-step timeout. Add cancellation token checked between steps. Add `cancel(run_id)` method.

### C.3 Trust & Security Debt

#### C.3.1 No Trustcard Integration in MAOS
**Problem:** Trustcard exists only at the Devin/MCP-proxy level. MAOS's own `ActionExecutor` uses simple string permissions with no manifest, no danger detection, no tool poisoning defense.
**Impact:** If MAOS exposes new tools via MCP, they are not scanned for danger. A compromised tool could bypass ActionExecutor if called directly.
**Fix:** Integrate Trustcard manifest generation into MAOS MCP server. Add danger scoring to action registry. Enforce manifest at the ActionExecutor level, not just MCP proxy.

#### C.3.2 Permissions are Global and Static
**Problem:** `PermissionChecker` is initialized once in `build_runtime()` with a hardcoded set. No per-workflow permissions, no dynamic grant/revoke, no permission persistence.
**Impact:** All workflows have the same permissions. Cannot scope a workflow to read-only. Cannot escalate permissions for a specific step with human approval.
**Fix:** Add per-workflow permission scope. Add permission escalation via approval queue. Persist permission grants.

#### C.3.3 No Approval Queue Integration with Workflows
**Problem:** `approval_queue` table exists with a `workflow_id` column, but `WorkflowEngine` never writes to it and never checks it. The approval system is disconnected from workflows.
**Impact:** Cannot pause a workflow for human approval. Cannot gate dangerous steps.
**Fix:** Add `requires_approval: bool` to `WorkflowStep`. Engine pauses and writes to `approval_queue`. Resume on approval.

#### C.3.4 Irreversible Actions Not Tracked at Workflow Level
**Problem:** `Action.allow_irreversible` exists in `ActionExecutor`, but workflows don't declare which steps have irreversible side effects. No way to know if a workflow is safe to retry.
**Impact:** Retrying a workflow might re-send emails, re-create tasks, etc.
**Fix:** Mark steps with `side_effects: list[str]`. Track idempotency keys. Block retry if non-idempotent side effects already executed.

### C.4 Hidden Coupling

| Coupling | Location | Risk |
|---|---|---|
| EventBus → Database (direct `db.execute` in `_persist`) | `bus.py:88-94` | Cannot use EventBus without DB. Testing requires DB mock. |
| `build_context()` → `detect_active_workspace()` (direct import) | `context.py:128` | Context builder cannot work without workspace detection. Circular dep risk with SessionAgent. |
| SessionAgent → `build_context()` | `session.py:24,104` | SessionAgent used in context building → circular dependency potential. |
| GitAgent → Database (bypasses retrieval for structured queries) | `git.py:48` | Direct SQL coupling. Schema changes break GitAgent. |
| KnowledgeGraph → Database (bypasses abstraction) | `graph.py:51` | Direct `db.get_or_create_entity()`, `db.add_relationship()`. |
| All collectors → EventBus | multiple | Hard to test collectors in isolation. |
| IntentRouter → Database (decision logging) | `router.py:122` | Routing cannot work without DB. |
| WorkflowEngine → step handlers (opaque callables) | `engine.py:71` | Engine cannot inspect what a handler will do before calling it. No dry-run. |

### C.5 In-Memory State Lost on Crash

| Component | Lost State | Recovery Behavior |
|---|---|---|
| EventBus | handler subscriptions | Handlers must re-subscribe on restart (no auto-recovery) |
| ContinuousIndexer | pending debounced events | File changes during crash may be missed until next change triggers re-index |
| TelemetryCollector | background thread | Must restart collector manually |
| SnapshotCollector | background thread | Must restart collector manually |
| AttentionDaemon | background thread | Must restart daemon manually |
| Blackboard | in-memory fields | **Recovered** via `restore(db)` from `user_model` table |
| WorkflowEngine | current execution state | **NOT recovered**. Stuck as 'running' in DB. |
| ModelManager | loaded models, idle timestamps | Models re-loaded on demand. No memory of what was loaded. |
| Workflow `ctx` dict | accumulated context | **NOT recovered**. Total loss. |

### C.6 Missing Transitions / States

**Current workflow states:** `running` → `completed` | `failed`

**Missing states needed for durable workflows:**
- `pending` — workflow created but not yet started
- `paused` — intentionally paused (e.g., awaiting approval)
- `cancelled` — cancelled by user or system
- `crashed` — process died during execution (detected on restart)
- `timed_out` — step or workflow exceeded time limit
- `compensating` — running compensation handlers after failure

**Missing transitions:**
- `running` → `paused` (pause for approval)
- `paused` → `running` (resume after approval)
- `running` → `crashed` (detected on restart)
- `running` → `cancelled` (user cancel)
- `running` → `timed_out` (step timeout)
- `failed` → `compensating` (start rollback)
- `compensating` → `compensated` (rollback complete)
- `crashed` → `running` (resume from checkpoint)
- `any` → `pending` (re-queue for retry)

### C.7 Unsafe Side Effects

| Side Effect | Component | Idempotent? | Tracked? | Compensable? |
|---|---|---|---|---|
| Send email | `app/gmail.py` | **No** (sends real email) | Yes (action_log) | No (can't unsend) |
| Create task | `db.create_task()` | No (creates new row) | Yes (action_log) | Yes (delete task) |
| Complete task | `db.complete_task()` | Yes (idempotent update) | Yes (action_log) | Yes (un-complete) |
| Write file | `FileWriteTool` | Yes (overwrites) | Yes (tool router) | No (original content lost) |
| Shell command | `ShellTool` | **No** (arbitrary effects) | Yes (tool router) | No (depends on command) |
| Git command | `GitTool` | No (mutates repo) | Yes (tool router) | No (depends on command) |
| Open workspace | `subprocess.run(["open", ...])` | Yes (no-op if already open) | Yes (action_log) | N/A (benign) |
| Chroma index write | `ChromaRetriever.index()` | Yes (upsert by stable ID) | No (no action_log) | Yes (delete chunks) |
| DB schema migration | `Database._migrate()` | Yes (ALTER IF NOT EXISTS) | No | No |

**Key risk:** Email sending is the most dangerous side effect — non-idempotent, non-compensable, and external. If a workflow sends an email at step 3 and fails at step 4, retrying the workflow sends a duplicate email.

### C.8 Test Coverage Gaps

**Current:** ~150 tests, all passing. Good coverage of individual components.

**Missing for durable workflows:**
- No test for crash recovery (kill mid-workflow, restart, verify state)
- No test for workflow resumption from checkpoint
- No test for stale `running` status detection on startup
- No test for concurrent workflow execution
- No test for workflow cancellation
- No test for step timeout
- No test for compensation execution
- No test for approval queue integration with workflows
- No test for permission escalation mid-workflow
- No integration test for full runtime wiring with workflows
- No test for email idempotency / duplicate prevention

---

## Summary: What Exists vs. What's Needed

| Capability | Current State | Target State |
|---|---|---|
| Workflow identity | Name string only | Stable ID + version + persisted definition |
| Workflow lifecycle | running → completed/failed | pending → running → paused → completed/failed/crashed/cancelled/timed_out/compensating |
| Step identity | Name string only | Stable ID + input/output contract |
| Step persistence | None (JSON blob at end) | Per-step records with data, verification, timing |
| Context | Unstructured dict | Typed, validated, checkpointable |
| Retry | None | Per-step retry policy with backoff |
| Timeout | None | Per-step and per-workflow timeout |
| Compensation | None | Per-step compensation handler |
| Checkpoints | None | Periodic context snapshots, resumable |
| Crash recovery | None | Detect stale runs, resume from checkpoint |
| Cancellation | None | Cancel token, checked between steps |
| Approval integration | Table exists, disconnected | Step-level approval gates, pause/resume |
| Permission scoping | Global, static | Per-workflow scope, dynamic escalation |
| Trustcard | External (MCP proxy only) | Integrated at ActionExecutor + manifest generation |
| Side effect tracking | action_log only | Per-step side effect manifest, idempotency keys |
| Parallel execution | None | Parallel step groups |
| Conditional branching | None | Per-step conditions |
| Sub-workflows | None | Workflow composition with context isolation |
| External API | MCP tools (run morning) | Start/pause/resume/cancel/inspect/retry workflows |
