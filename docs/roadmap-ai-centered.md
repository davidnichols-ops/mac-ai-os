# MAOS: AI at the Center — Implementation Roadmap

> **Vision**: A local-first AI operating system where the model is the brain. It thinks, decides, acts, observes, and iterates — entirely on-device. No cloud. No deterministic routing. No hardcoded heuristics. Just the model, tools, and a loop.

---

## Executive Summary

MAOS today has 19+ fully implemented subsystems — agents, tools, retrieval, knowledge graph, blackboard, attention daemon, workflow engine, guards, telemetry, sessions, onboarding, learning, perception, and more. The architecture is clean, modular, and local-first.

**The problem**: The AI is not at the center. Deterministic code makes the decisions:

- **Routing** is keyword-based (`orchestrator.py` lines 27-35) — the model is never consulted
- **Tool calling doesn't exist in chat mode** — the model is never given tool definitions, never parses tool calls, never executes tools. The system prompt says "you choose tools, execute" but this is a lie.
- **Agents retrieve data directly** — they call `retriever.search()`, query SQLite, parse regex — instead of asking the model to call tools
- **Question classification** uses regex patterns (`git.py`, `session.py`) instead of LLM understanding
- **Tool infrastructure exists but is unused internally** — `ToolRouter`, `Tool` base class, schemas, permissions — all only used by the MCP server for external clients (Claude, Codex)

**The solution**: Implement a full agentic loop adapted from the local-operator project, where the model runs in a cycle: **think -> act (tool call) -> observe (tool result) -> think -> ... until done**. Replace deterministic routing with model-driven tool selection. Convert all agent capabilities into tools the model can call.

**The constraint**: Everything runs locally on the M4 16GB Mac with the v2 MLX 4-bit adaptive operator model. No cloud API calls. Ever.

---

## Current Architecture (What We Have)

```
User Input
    |
    v
+--------------+     keyword match     +--------------+     direct retrieval    +--------------+
| Orchestrator |--------------------->|   Agent      |------------------------>|  Retriever   |
|  (no AI)     |                       | (7 types)    |                        |  / DB / etc  |
+--------------+                       +------+-------+                        +--------------+
                                              | planner.chat()
                                              v
                                       +--------------+
                                       |   Model      |     text synthesis only
                                       |  (MLX 4-bit) |     no tool definitions
                                       +--------------+     no tool call parsing
```

**Key gaps**:
1. Model providers (`mlxvlm_provider.py`, etc.) don't accept `tools` parameter
2. `Message` type has no `tool_calls` or `tool_call_id` fields
3. `ModelResponse` has no `tool_calls` field
4. No agent loop — single-shot `planner.chat()` -> text -> done
5. No tool call parsing from model output
6. No tool result feedback to model
7. `orchestrator.available_tools()` and `orchestrator.call_tool()` exist but are never called

## Target Architecture (What We're Building)

```
User Input
    |
    v
+--------------+     system prompt +     +------------------------------------------+
|  Agent Loop  |     tool definitions    |              AGENTIC LOOP                |
|  (maos do)   |----------------------->|                                          |
+--------------+                        |  +---------+  +---------+  +---------+  |
                                        |  |  THINK  |->|  ACT    |->| OBSERVE |  |
                                        |  | (model) |  |(tool    |  |(result  |  |
                                        |  |         |  | call)   |  | feed-   |  |
                                        |  |         |<-|         |<-| back)   |  |
                                        |  +----+----+  +---------+  +---------+  |
                                        |       | done?                          |
                                        |       v                                |
                                        |  +---------+                            |
                                        |  | ANSWER  |                            |
                                        |  +---------+                            |
                                        +------------------------------------------+
                                                        |
                                                        v
                                        +----------------------------------+
                                        |        TOOL LAYER               |
                                        | shell | file_read | file_write  |
                                        | git   | search    | rag          |
                                        | session| system   | graph        |
                                        | tasks | workspace | web          |
                                        | ...30+ tools...                 |
                                        +----------------------------------+
```

**Two modes**:
- `maos chat` — pure conversation, no tools, just the model talking
- `maos do "..."` — full agent loop with all tools available

---

## Roadmap: 6 Phases

### Phase 1: Foundation — Agent Loop & Tool Calling
**Goal**: Make the model able to call tools and iterate. This is the critical missing piece.

**1.1 Extend type system** (`app/models/types.py`)
- Add `ToolCall` dataclass: `id`, `name`, `arguments: dict`
- Add `tool_calls: list[ToolCall]` to `ModelResponse`
- Add `tool_calls`, `tool_call_id`, `name` fields to `Message`
- Add `tools: list[dict]` parameter to `Model.chat()` signature

**1.2 Implement tool call parsing** (`app/models/mlxvlm_provider.py`)
- The v2 model doesn't have native function calling, so we use text-based parsing
- Define a structured output format the model can produce (XML-style tags)
- Parse from model output (adapt from local-operator's 4-format parser)
- Support: XML tags, pipe-delimited, code blocks, bare JSON formats
- Inject tool definitions into the system prompt as formatted text

**1.3 Create AgentLoop class** (`app/agent_loop.py`)
- Adapted from local-operator's `BenchmarkRunner._run_loop()`
- Core cycle:
  1. Build system prompt with tool schemas + context
  2. Call model with messages + tools
  3. Parse tool calls from response
  4. If tool calls: execute each, append results as tool messages, go to 2
  5. If no tool calls: model is done, return final answer
- Safety: step budget (max 20 steps), infinite loop detection (same tool call 3x), timeout
- Trajectory recording: log every step to `agent_trajectories` table
- Nudge mechanism: if model stalls, inject "You have tools available. Use them."

**1.4 Add `maos do` command** (`app/cli/commands/do.py`)
- `maos do "fix the failing tests in src/auth/"`
- Starts agent loop with all tools available
- Shows step-by-step progress (tool calls + results)
- Streams model output to terminal

**1.5 Keep `maos chat` as pure conversation**
- No tool definitions passed
- Just the model talking with conversation history
- System prompt: "You are Mac AI OS. You're having a conversation."

**1.6 Wire existing tools into the loop**
- The 4 built-in tools (`ShellTool`, `FileReadTool`, `FileWriteTool`, `GitTool`) become available
- Tool schemas formatted and injected into system prompt
- ToolRouter executes calls with permission checks

**Deliverable**: `maos do "list files in the current directory"` -> model calls `shell` tool -> gets `ls` output -> responds with summary.

---

### Phase 2: AI-Centered Orchestration
**Goal**: Replace deterministic routing with model-driven decisions. Eliminate the 7 specialized agents.

**2.1 Remove keyword-based orchestrator routing**
- The agent loop doesn't need routing — the model decides what to do
- `orchestrator.py` becomes a thin wrapper: build system prompt, start agent loop
- The 7 agents (`general`, `filesystem`, `git`, `repository`, `research`, `session`, `system`) are replaced by tools

**2.2 Convert agent capabilities into tools**
Each agent's retrieval logic becomes a tool the model can call:

| Old Agent | New Tool(s) |
|-----------|-------------|
| `FilesystemAgent` | `rag_search` (search indexed files) |
| `GitAgent` | `git_log`, `git_changes`, `git_files`, `git_explain` |
| `RepositoryAgent` | `code_search` (RAG over code) |
| `ResearchAgent` | `web_search`, `web_browse` |
| `SessionAgent` | `session_list`, `session_summarize`, `session_ask` |
| `SystemAgent` | `system_status` (CPU, memory, battery) |
| `GeneralAgent` | (eliminated — the model itself is the general agent) |

**2.3 Redesign system prompt**
- The system prompt is the "OS kernel" — it tells the model what it is, what tools it has, how to behave
- Includes: identity, available tools (with schemas), current context (workspace, session, time), behavioral guidelines
- Injected dynamically at the start of each agent loop

**2.4 Context injection**
- Onboarding briefing (lessons, recent work, active workspace) injected into system prompt
- Time awareness footer
- Blackboard state (current goal, open problems, suggested actions)

**Deliverable**: `maos do "what was I working on yesterday?"` -> model calls `session_summarize` tool -> synthesizes answer. No keyword routing needed.

---

### Phase 3: Tool Expansion
**Goal**: Give the model a rich toolset so it can actually do things, not just answer questions.

**3.1 File operations tools**
- `file_read` (exists) — read file contents
- `file_write` (exists) — write file contents
- `file_edit` — surgical edit (find/replace in file)
- `file_move` — move/rename
- `file_copy` — copy
- `file_delete` — delete (with confirmation)
- `file_list` — list directory
- `file_search` — find files by name pattern
- `file_grep` — search file contents (uses ripgrep)

**3.2 Shell tool** (exists, enhance)
- `shell` — execute command, return stdout/stderr/exit code
- Add: working directory tracking, timeout configuration, env var support
- Add: command history (stored in blackboard for context)

**3.3 Git tools**
- `git_status` — working tree status
- `git_diff` — show changes
- `git_log` — recent commits
- `git_commit` — stage + commit (with message)
- `git_branch` — list/create/switch branches
- `git_merge` — merge branches
- `git_explain` — RAG-based commit explanation

**3.4 Search & retrieval tools**
- `rag_search` — semantic search over indexed content
- `web_search` — DuckDuckGo search (via perception)
- `web_browse` — fetch web page (via perception)
- `code_search` — search code in indexed repos
- `knowledge_search` — search knowledge graph

**3.5 System tools**
- `system_status` — CPU, memory, battery, thermal
- `process_list` — running processes
- `process_kill` — kill process (with confirmation)
- `disk_usage` — disk space
- `network_status` — network interfaces, connectivity

**3.6 Session & workspace tools**
- `session_list` — recent work sessions
- `session_summarize` — LLM summary of a session
- `workspace_list` — registered workspaces
- `workspace_active` — current workspace
- `workspace_scan` — scan for projects

**3.7 Task management tools**
- `task_create` — create task
- `task_list` — list tasks
- `task_complete` — mark done
- `task_delete` — delete task

**3.8 Knowledge tools**
- `graph_search` — search knowledge graph
- `graph_extract` — extract entities from text
- `graph_stats` — graph statistics

**3.9 Code intelligence tools**
- `code_read` — read file with syntax awareness
- `code_edit` — edit with diff preview
- `code_lint` — run linter
- `code_test` — run tests
- `code_build` — run build

**3.10 Perception tools**
- `ocr` — OCR an image
- `read_pdf` — extract text from PDF
- `screenshot` — capture screen (if permitted)

**Deliverable**: `maos do "find all Python files with TODO comments and list them"` -> model calls `file_grep` -> parses results -> responds with organized list.

---

### Phase 4: Cognitive Architecture
**Goal**: Wire the existing cognitive subsystems into the agent loop so the model has memory, attention, and learning.

**4.1 Blackboard as agent working memory**
- The agent loop reads blackboard state at start (current goal, open problems, suggestions)
- The agent loop writes to blackboard during execution (decisions made, tools called, results)
- Blackboard persists across sessions (already implemented in SQLite)

**4.2 Attention daemon feeds the agent**
- Attention daemon (60s cycle) updates blackboard with proactive context
- When agent loop starts, it reads the latest blackboard state
- Suggestions from attention daemon are injected as system prompt context
- Example: "You have uncommitted changes in 3 files. A task 'fix auth bug' is pending."

**4.3 Learning system tracks tool outcomes**
- After each tool call, record outcome (success/failure) in learning system
- Model can query learned success rates: "Has this tool failed before in this context?"
- Failure patterns become lessons (already implemented in `learning.py`)

**4.4 Knowledge graph built from agent interactions**
- After agent completes a task, extract entities/relationships from the conversation
- Store in knowledge graph (already implemented in `app/knowledge/`)
- Model can search graph via `graph_search` tool

**4.5 Onboarding briefing in system prompt**
- At agent loop start, build onboarding packet (lessons, recent work, active workspace)
- Inject as context in system prompt
- Model starts every task already briefed on context

**4.6 Time awareness in tool results**
- Every tool result includes time footer (already implemented in MCP tools)
- Agent loop injects time awareness into system prompt
- Model can reason about time ("I worked on this 2 hours ago")

**4.7 Trajectory recording**
- New `agent_trajectories` table: step number, tool called, arguments, result, model output, timing
- Enables: replay, debugging, distillation (train v3 on successful trajectories)
- Queryable by the model: "What did I do last time I worked on this?"

**Deliverable**: `maos do "continue where I left off"` -> model reads blackboard + onboarding -> calls `session_summarize` -> resumes work.

---

### Phase 5: Proactive & Background Operations
**Goal**: MAOS becomes proactive, not just reactive. It works in the background, suggests actions, and runs scheduled tasks.

**5.1 Background agent**
- Long-running agent loop that operates without user input
- Triggered by: file changes, schedule, events, attention daemon suggestions
- Example: "Index new files when they appear in workspace"
- Example: "Run tests when files change"
- Example: "Summarize session when idle for 30 minutes"

**5.2 Proactive suggestions**
- Attention daemon identifies open problems -> agent generates suggestions
- Suggestions surfaced via: terminal notification, menubar icon, `maos suggest` command
- Example: "You have uncommitted changes. Want me to review them?"
- Example: "The test suite is failing. Want me to investigate?"

**5.3 Scheduled tasks**
- AI-driven cron: "Every morning at 9am, summarize yesterday's work and list today's tasks"
- Implemented as workflow definitions (already have workflow engine)
- Agent loop executes workflow steps, model decides how to complete each step

**5.4 File watching triggers**
- `maos watch` command already exists for indexing
- Extend: file change events trigger agent actions
- Example: ".py file changed -> run linter -> if errors, suggest fix"
- Example: "new file in inbox/ -> summarize -> create task"

**5.5 Event-driven activation**
- Event bus (already implemented) triggers agent loop
- Events: git commit, task created, session ended, build failed
- Agent responds to events autonomously (with user-configured permissions)

**Deliverable**: MAOS notices you have failing tests, investigates, and suggests a fix before you ask.

---

### Phase 6: Full Local-First AI OS
**Goal**: MAOS becomes the primary interface to the computer — not the terminal, not the GUI, but the model.

**6.1 Natural language command interface**
- `maos "..."` (no subcommand) -> model decides: chat or agent mode?
- If simple question -> chat mode (fast, no tools)
- If task -> agent mode (full tool loop)
- The model itself is the command parser

**6.2 Voice input/output**
- `maos voice` — listen via macOS speech recognition, respond via TTS
- Hands-free interaction: "Hey MAOS, what's my battery level?"
- Whisper.cpp for STT (local), macOS AVSpeechSynthesizer for TTS (local)

**6.3 System-level integration**
- macOS menubar app showing MAOS status
- Notification center integration for proactive suggestions
- Spotlight-like quick access (Cmd+Space -> type -> MAOS responds)
- Background daemon always running

**6.4 Multi-agent collaboration**
- Local model as orchestrator
- Spawns sub-agents for specific tasks (each a separate agent loop)
- Sub-agents can use different models (if user has multiple local models)
- Example: "Research X" -> orchestrator spawns research sub-agent -> sub-agent uses web_search -> reports back

**6.5 Plugin system**
- External tools registered as plugins (entry-point discovery)
- Each plugin: tool schema + handler + permissions
- `maos plugin install <name>` -> registers tool in registry
- Community plugins: Spotify control, calendar, reminders, etc.

**6.6 Self-improvement**
- Agent loop trajectories become training data
- Successful trajectories -> fine-tune v3 model
- Failed trajectories -> lessons (already implemented)
- Model learns from its own actions and gets better over time
- "MAOS got better at fixing tests because it practiced 100 times"

**6.7 Workspace awareness**
- MAOS knows which project you're in, what you're doing, what's broken
- Auto-indexes new files, auto-runs tests on change, auto-summarizes sessions
- The model has persistent context across all interactions

**6.8 Cross-device sync (optional, local network)**
- Sync blackboard, lessons, knowledge graph across Macs on same network
- No cloud — direct Mac-to-Mac sync over LAN
- "I was working on this on my MacBook, continue on my Mac Studio"

**Deliverable**: You talk to your Mac through MAOS. It knows your context, has tools to act, learns from experience, and gets better over time. All local. All private.

---

## Implementation Priority

| Phase | Effort | Impact | Dependency |
|-------|--------|--------|------------|
| 1. Agent Loop & Tool Calling | High | Critical | None |
| 2. AI-Centered Orchestration | Medium | High | Phase 1 |
| 3. Tool Expansion | Medium | High | Phase 1 |
| 4. Cognitive Architecture | Medium | High | Phase 1, 2 |
| 5. Proactive Operations | High | Medium | Phase 1-4 |
| 6. Full AI OS | Very High | Transformative | Phase 1-5 |

**Critical path**: Phase 1 -> Phase 2 -> Phase 3 (can parallel with 2) -> Phase 4 -> Phase 5 -> Phase 6

---

## Technical Decisions

### Tool Call Format (Text-Based)
The v2 MLX model doesn't support native function calling. We use text-based parsing with XML-style tags. The system prompt teaches the model this format. We parse it from output. This is the same approach local-operator uses successfully with Qwen models.

The model outputs tool calls in a structured format that we parse from the text response. We support multiple formats for robustness: XML tags, pipe-delimited, fenced code blocks, and bare JSON.

### System Prompt Design
The system prompt is dynamically assembled for each agent loop invocation:

1. **Identity**: "You are Mac AI OS (MAOS), a local-first AI operating system running on Apple Silicon."
2. **Available Tools**: Formatted tool schemas (name, description, parameters)
3. **Current Context**: Workspace, time, session info, onboarding briefing
4. **Behavioral Guidelines**: Think step by step, call tools when needed, assess after each result, provide clear answer when done, try different approaches on failure
5. **Tool Call Format**: Instructions for how to call tools in the output

### Safety Boundaries
- **Step budget**: max 20 tool calls per task (configurable)
- **Infinite loop detection**: same tool + same args 3 times -> stop
- **Timeout**: 5 minutes per task (configurable)
- **Permissions**: tools check permissions before executing (already implemented)
- **Destructive operations**: `file_delete`, `process_kill`, `git_commit` require user confirmation
- **Sandboxing**: shell tool can be restricted to workspace directory

### Memory Management
- The v2 MLX 4-bit model uses ~5GB of unified memory
- Tool results can be large (file contents, command output) — truncate to 4000 chars
- Conversation history: keep last 10 messages, summarize older ones
- Blackboard: in-memory, persisted to SQLite

### Trajectory Storage
New SQLite table for recording agent loop executions:

```sql
CREATE TABLE agent_trajectories (
    id INTEGER PRIMARY KEY,
    session_id TEXT,
    step INTEGER,
    tool_name TEXT,
    tool_arguments TEXT,  -- JSON
    tool_result TEXT,     -- truncated
    model_output TEXT,
    mode TEXT,            -- FAST / THINK
    inference_time_ms INTEGER,
    timestamp TEXT
);
```

This enables replay, debugging, and future distillation (training v3 on successful trajectories).

---

## What We're NOT Changing

These subsystems work well and stay as-is:

- **Database** (SQLite) — solid storage layer
- **Retrieval** (Chroma) — vector search works
- **Embeddings** — chroma_default is fine
- **Telemetry** — comprehensive logging
- **Event bus** — decoupled pub/sub
- **Permission system** — tool-level permissions
- **MCP server** — stays for external client access (Claude, Codex)
- **Guards** — PR governance stays for development workflows
- **Workflow engine** — stays for scheduled/multi-step tasks
- **Config system** — layered config is good
- **Hardware detection** — works well

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Tool calling in chat | No | Yes (agent mode) |
| Model-driven routing | No (keywords) | Yes (model decides) |
| Tools available to model | 0 | 30+ |
| Agent loop iterations | 0 | Unlimited (with budget) |
| Local-only | Yes | Yes (stays local) |
| Proactive suggestions | No | Yes (Phase 5) |
| Self-improvement | No | Yes (Phase 6) |

---

## Next Action

**Start Phase 1**: Implement the agent loop and tool calling. This is the foundation everything else builds on. Without it, MAOS is just a RAG system with a chat interface.

The local-operator project has a working agent loop (`BenchmarkRunner._run_loop()`) that we can adapt directly. The tool call parsing, trajectory recording, and safety mechanisms are all proven code.

### Phase 1 Implementation Order

1. **types.py** — Add `ToolCall`, extend `Message` and `ModelResponse` (30 min)
2. **mlxvlm_provider.py** — Add `tools` param, implement tool call parsing (1 hr)
3. **agent_loop.py** — Create `AgentLoop` class with think-act-observe cycle (2 hr)
4. **do.py** — Add `maos do` CLI command (30 min)
5. **Test** — `maos do "list files in current directory"` (30 min)
6. **Iterate** — Fix parsing edge cases, improve system prompt (ongoing)
