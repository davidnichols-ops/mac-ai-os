# Mac AI OS — Operator's Manual

## What this is

Mac AI OS (maos) is a local-first intelligence layer for your Mac. It runs
entirely on-device — no cloud, no API keys, no telemetry leaving your machine.
It observes what you do, remembers it, understands context, plans, takes
permissioned actions, verifies results, and learns from outcomes.

It is not a chatbot. It is an operating environment with a causal loop:

```
observe → remember → understand → plan → act → verify → learn
```

## Installation

### From source (development)

```bash
git clone <repo> ~/mac-ai-os
cd ~/mac-ai-os
uv sync --extra mlx --extra dev
./scripts/setup_models.sh
```

### System-wide (daily use)

```bash
cd ~/mac-ai-os
uv tool install . --with mlx-lm
```

This puts `maos` and `mac-ai-os` on your PATH. Verify:

```bash
maos --help
maos verify
```

### Requirements

- macOS on Apple Silicon (M1 or later)
- Python 3.12+
- ~10 GB disk for models
- 16 GB RAM recommended (8 GB minimum)

## Data locations

| What | Development | Installed |
|------|-------------|-----------|
| Database | `~/mac-ai-os/data/maos.db` | `~/.local/share/maos/maos.db` |
| Embeddings | `~/mac-ai-os/data/embeddings/` | `~/.local/share/maos/embeddings/` |
| Configs | `~/mac-ai-os/configs/` | bundled in package |
| User overrides | `~/mac-ai-os/configs/user.yaml` | `~/.config/maos/configs/user.yaml` |

The system auto-detects which mode to use. If you run from the source repo,
it uses the repo's `data/` directory. If you run the installed `maos` from
anywhere else, it uses `~/.local/share/maos/`.

## Models

Three models run on-device via MLX:

| Role | Model | Size | Purpose |
|------|-------|------|---------|
| Planner | Qwen3-14B-4bit | ~8 GB | Reasoning, synthesis, session reconstruction |
| Utility | Llama-3.2-3B-Instruct-4bit | ~2 GB | Fast classification, intent routing |
| Embeddings | all-MiniLM-L6-v2 (ONNX) | ~80 MB | Text vectorization for RAG |

Models load on first use (~5 seconds cold start) and stay in memory. The
ModelManager tracks memory usage and unloads idle models under pressure.

To change models, edit `configs/user.yaml`:

```yaml
planner:
  provider: mlx
  model: mlx-community/Qwen3-14B-4bit
utility:
  provider: mlx
  model: mlx-community/Llama-3.2-3B-Instruct-4bit
```

Run `maos models` to see what's available.

## Daily workflow

### Start of day

```bash
maos workflow morning
```

This is the flagship command. It runs six steps:

1. **Reconstruct last session** — finds your most substantial recent work
   session, shows what you were doing, what files you touched, what commits
   you made.
2. **Summarize changes** — detects your active workspace (CWD-aware), shows
   git branch and modified files.
3. **Check pending tasks** — lists tasks you haven't completed.
4. **Check system health** — collects a telemetry snapshot (CPU, memory,
   processes) if none exists recently.
5. **Prepare workspace** — opens the workspace in Finder.
6. **Create daily plan** — synthesizes everything above into a recommended
   plan of action.

Output looks like:

```
Good morning.

Last session: 2026-07-13T03:18:08 → 2026-07-13T03:20:42
  Project: sample_docs
  Events: 5, Commits: 0, Files touched: 2

Current workspace: mac-ai-os
  Branch: master
  Modified files (10): M AGENTS.md, M app/agents/git.py, ...

Pending tasks (1): [1] Benchmark Qwen3 14B latency

System: normal (CPU 8.6%, Memory 6.9/16.0GB)

Recommended plan:
  Continue work on sample_docs
  Address 1 pending task(s)
    - [1] Benchmark Qwen3 14B latency
  Review 10 modified files in mac-ai-os
```

### During work

**Ask about your code:**
```bash
maos rag "how does the action executor work?"
```

RAG searches indexed content and feeds results to the planner LLM. Only
works on files you've indexed (see Indexing below).

**One-shot prompt (no retrieval):**
```bash
maos ask "explain the difference between epoll and kqueue"
```

**Interactive chat:**
```bash
maos chat
```

Routes your message through the intent router to the best agent (git,
session, research, filesystem, repository, system).

**Ask about past work:**
```bash
maos session ask "what was I working on yesterday?"
```

The session agent correlates events, git commits, shell commands, telemetry,
and the knowledge graph to reconstruct what you did.

**Manage tasks:**
```bash
maos task add "implement SwiftUI menu bar"
maos task list
maos task done 3
```

**Set your current goal:**
```bash
maos blackboard goal "ship Phase 5 UI"
```

The goal persists in the blackboard and influences suggestions.

### End of day

```bash
maos snapshot capture mac-ai-os
```

Captures workspace state: git branch, modified files, recent commands,
active processes. Stored in `workspace_snapshots` table.

## Indexing

The system only knows about files you've indexed. Indexing is hash-aware
and incremental — re-running skips unchanged files.

```bash
# Index a project's source code
maos index ~/mac-ai-os/app

# Index another project
maos index ~/Projects/X-MaC/src

# Force re-index everything (after bulk changes)
maos index ~/mac-ai-os/app --force

# Watch a directory for changes (continuous indexing)
maos watch ~/mac-ai-os/app
```

**What gets indexed:**

- Code: `.py`, `.js`, `.ts`, `.tsx`, `.go`, `.rs`, `.c`, `.h`, `.cpp`, `.swift`,
  `.java`, `.kt`, `.rb`, `.sh`, `.bash`, `.zsh`, `.sql`, `.html`, `.css`
- Text: `.txt`, `.md`, `.rst`, `.log`, `.ini`, `.cfg`, `.toml`, `.yaml`, `.yml`

**What gets skipped:**

- Directories: `.git`, `.venv`, `venv`, `__pycache__`, `node_modules`,
  `.pytest_cache`, `dist`, `build`
- Files over 256 KB
- `.DS_Store`

**Git history indexing:**
```bash
maos git index ~/mac-ai-os
maos git log
maos git changes --days 1
maos git files
```

Each commit becomes a searchable chunk with metadata (hash, author, date,
files, insertions/deletions).

## Memory

The system has three types of memory:

### Episodic (what happened)

- **Events** — every observable action (file indexed, model loaded, snapshot
  captured, action executed). View with `maos timeline`.
- **Sessions** — auto-detected from event gaps (30 min idle = new session).
  View with `maos session list`.
- **Snapshots** — periodic workspace state captures.
  View with `maos snapshot list`.

### Semantic (what I know)

- **Knowledge graph** — entities and relationships extracted by the LLM from
  indexed content. View with `maos graph stats`, `maos graph search "term"`.
- **RAG** — vector-indexed content for retrieval.
  Search with `maos rag "question"`.

### Procedural (how I work)

- **User model** — your preferences, patterns, and learned behaviors with
  confidence scores. View with `maos preferences list`.
- **Action outcomes** — success/failure rates per action.
  View with `maos learn summary`.
- **Blackboard** — shared working memory (current goal, workspace, context,
  open problems, suggested actions, confidence). View with `maos blackboard show`.

## The blackboard

The blackboard is shared working memory. Every capability reads and writes
to it. Instead of rediscovering context from scratch on each interaction,
the blackboard maintains state across runs.

```bash
# Show current state
maos blackboard show

# Set your current goal
maos blackboard goal "ship the SwiftUI menu bar app"

# Run an attention cycle (proactive context computation)
maos blackboard cycle

# Clear everything
maos blackboard clear
```

The attention daemon computes: active workspace, current goal, system health,
open problems, suggested actions, and confidence. It updates the blackboard
and persists to the database so state survives restarts.

## Actions

Actions are the trust boundary. The LLM never executes anything directly —
it requests actions, which are validated, permission-checked, logged, and
reversible where possible.

### Available actions

| Action | Permissions | Reversible | Description |
|--------|-------------|------------|-------------|
| `open_workspace` | filesystem.read, application.launch | yes | Opens a workspace in Finder |
| `create_todo` | tasks.write | yes | Creates a task |
| `complete_todo` | tasks.write | no | Marks a task complete |
| `workspace_summary` | filesystem.read, git.exec | yes | Summarizes workspace state |
| `list_tasks` | tasks.read | yes | Lists pending tasks |

### Running actions

```bash
# List available actions
maos actions list

# Run a safe action directly
maos actions run workspace_summary mac-ai-os
maos actions run create_todo "" --params '{"title": "Fix tests"}'

# View action log
maos actions log

# View decision log (intent routing decisions)
maos actions decisions
```

### Approval queue

For actions that need human approval:

```bash
# Submit an action for approval
maos approval submit git_commit mac-ai-os --reason "3 files changed" --risk medium

# List pending approvals
maos approval list

# Approve and execute
maos approval approve 1

# Deny
maos approval deny 1
```

## Workflows

Workflows are multi-step sequences with verification.

```bash
# Morning routine (the main one)
maos workflow morning

# List past workflow runs
maos workflow list

# Show details of a specific run
maos workflow show 3
```

Each step produces a `VerificationResult` with evidence — not just "ok" but
"why it's ok." Failed optional steps don't stop the workflow; failed required
steps do.

## Shell history

The system imports your shell history so it knows what commands you've run.

```bash
# Import from ~/.zsh_history and ~/.bash_history
maos history import

# Search past commands
maos history search "pytest"

# Recent commands
maos history recent
```

Import is incremental — re-running skips already-imported commands.

## Telemetry

```bash
# Collect a snapshot now
maos telemetry collect --seconds 30

# View recent telemetry
maos telemetry history --hours 1

# Telemetry status
maos telemetry status
```

Telemetry captures: CPU usage, memory usage, process count, top processes
by CPU and memory. GPU and Neural Engine are not available (no public API
without root).

## Knowledge graph

```bash
# Graph statistics
maos graph stats

# Search entities
maos graph search "workflow"

# List all entities
maos graph entities

# Extract entities from indexed content (uses planner LLM)
maos graph extract "action executor"

# Add a relationship manually
maos graph add ActionExecutor uses PermissionChecker
```

## Preferences

```bash
# List all preferences
maos preferences list

# Set a preference with confidence
maos preferences set coding_style "typed Python with dataclasses" --confidence 0.9

# Get a specific preference
maos preferences get coding_style
```

Preferences are procedural memory — the system uses them to adapt behavior.
Confidence scores (0.0–1.0) reflect how certain the system is about each
preference.

## Learning

```bash
# Summary of what the system has learned
maos learn summary

# Record an action outcome manually
maos learn action open_workspace --success
maos learn action git_commit --fail --error "merge conflict"
```

The learner tracks:
- Action success rates (e.g., `open_workspace: 2/3 = 67%`)
- Common errors per action
- Workflow outcomes
- User preference confirmations (confidence increases when observations
  match stated preferences)

## Activity timeline

```bash
# Recent activity (all event types)
maos timeline --hours 1

# List all event types
maos event-types
```

The timeline is the complete record of everything the system has observed
and done — file indexing, model loading, telemetry, actions, workflows,
snapshots.

## Workspaces

```bash
# Scan for project directories
maos workspace scan

# List discovered workspaces
maos workspace list

# Show active workspace
maos workspace active
```

Workspace detection scans standard project directories (`~/Projects`,
`~/Developer`, `~/Code`, `~/repos`, `~/src`, `~/Documents/Projects`,
`~/Desktop`). A directory is a workspace if it contains `.git`,
`package.json`, `pyproject.toml`, `Cargo.toml`, `go.mod`, `Package.swift`,
or similar markers.

The active workspace is detected by: (1) current working directory if it's
a project, (2) most recently indexed workspace from events, (3) most
recently registered workspace.

## Troubleshooting

### "maos: command not found"

The installed binary is at `~/.local/bin/maos`. Ensure `~/.local/bin` is on
your PATH:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Models fail to load

```bash
maos verify
```

This checks MLX, Ollama, hardware, and model availability. If MLX models
aren't downloaded, they'll download on first use (~8 GB for the planner).

### RAG returns nothing

You need to index content first:

```bash
maos index ~/your-project/src
maos rag "your question"
```

### Wrong workspace detected

The system prioritizes your current working directory. Run `maos` from the
project you want to work on, or set it explicitly:

```bash
maos blackboard goal "working on X"
```

### Database is empty when running installed maos

The installed `maos` uses `~/.local/share/maos/maos.db`. The development
version uses `~/mac-ai-os/data/maos.db`. These are separate databases. To
migrate data:

```bash
cp ~/mac-ai-os/data/maos.db ~/.local/share/maos/maos.db
cp -r ~/mac-ai-os/data/embeddings/* ~/.local/share/maos/embeddings/
```

### Embedding model 401 error

This was a config loading bug (fixed). If you see HuggingFace 401 errors for
`all-MiniLM-L6-v2-onnx`, reinstall:

```bash
cd ~/mac-ai-os
uv tool install . --with mlx-lm --force
```

## Configuration

### Layering

```
code defaults  <  configs/default.yaml  <  configs/user.yaml  <  env vars
```

### User overrides

Edit `configs/user.yaml` (development) or `~/.config/maos/configs/user.yaml`
(installed):

```yaml
planner:
  provider: mlx
  model: mlx-community/Qwen3-14B-4bit
  temperature: 0.7
  max_tokens: 1024

utility:
  provider: mlx
  model: mlx-community/Llama-3.2-3B-Instruct-4bit
  temperature: 0.4
  max_tokens: 256

embeddings:
  provider: chroma_default
  collection: mac-ai-os
  persist_dir: data/embeddings
```

### Environment variables

All settings can be overridden via env vars with `MAOS_` prefix:

```bash
MAOS_PLANNER__MODEL=mlx-community/Qwen3-32B-4bit maos ask "hello"
```

## Command reference

| Command | Purpose |
|---------|---------|
| `maos verify` | Check MLX, Ollama, hardware, models |
| `maos models` | List available models |
| `maos index <path>` | Index a folder into vector store |
| `maos index <path> --force` | Re-index everything |
| `maos watch <path>` | Watch directory for changes |
| `maos ask "..."` | One-shot LLM prompt |
| `maos rag "..."` | Retrieval-augmented question |
| `maos chat` | Interactive chat session |
| `maos session ask "..."` | Ask about past work |
| `maos session list` | List detected sessions |
| `maos session summarize [id]` | Summarize a session |
| `maos workflow morning` | Morning routine |
| `maos workflow list` | List workflow runs |
| `maos workflow show <id>` | Show workflow details |
| `maos task add "title"` | Create a task |
| `maos task list` | List tasks |
| `maos task done <id>` | Complete a task |
| `maos actions list` | List available actions |
| `maos actions run <name> [target]` | Run an action |
| `maos actions log` | View action log |
| `maos actions decisions` | View decision log |
| `maos approval submit <action> [target]` | Submit for approval |
| `maos approval list` | List pending approvals |
| `maos approval approve <id>` | Approve and execute |
| `maos approval deny <id>` | Deny |
| `maos blackboard show` | Show working memory |
| `maos blackboard goal "text"` | Set current goal |
| `maos blackboard cycle` | Run attention cycle |
| `maos blackboard attention` | Attention daemon status |
| `maos blackboard clear` | Clear blackboard |
| `maos learn summary` | What the system learned |
| `maos learn action <name> --success/--fail` | Record outcome |
| `maos preferences list` | List preferences |
| `maos preferences set <key> <value>` | Set preference |
| `maos preferences get <key>` | Get preference |
| `maos snapshot capture [workspace]` | Capture workspace state |
| `maos snapshot list` | List snapshots |
| `maos snapshot latest [workspace]` | Latest snapshot |
| `maos git index <repo>` | Index git history |
| `maos git log [repo]` | Recent commits |
| `maos git changes [repo] --days N` | Changes since N days |
| `maos git files [repo]` | Most modified files |
| `maos git explain "..."` | Explain commits via RAG |
| `maos workspace scan [path]` | Scan for projects |
| `maos workspace list` | List workspaces |
| `maos workspace active` | Show active workspace |
| `maos history import` | Import shell history |
| `maos history search "pattern"` | Search shell history |
| `maos history recent` | Recent commands |
| `maos telemetry collect --seconds N` | Collect telemetry |
| `maos telemetry history --hours N` | View telemetry |
| `maos telemetry status` | Telemetry status |
| `maos timeline --hours N` | Activity timeline |
| `maos event-types` | List event types |
| `maos graph stats` | Graph statistics |
| `maos graph search "name"` | Search entities |
| `maos graph entities` | List entities |
| `maos graph extract "topic"` | LLM entity extraction |
| `maos graph add <src> <rel> <tgt>` | Add relationship |
