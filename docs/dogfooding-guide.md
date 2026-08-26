# MAOS Dogfooding Guide — Start, Index, and Work on Projects

This is the hands-on guide for using MAOS as your daily operating intelligence.
It covers the full lifecycle: initial setup, indexing your projects, working
with the local model, connecting external AI agents, and the daily workflow
that makes MAOS actually useful.

---

## Table of Contents

1. [The 30-Second Mental Model](#1-the-30-second-mental-model)
2. [First-Time Setup](#2-first-time-setup)
3. [Indexing Your Projects](#3-indexing-your-projects)
4. [The Daily Workflow](#4-the-daily-workflow)
5. [Working on a Project with MAOS](#5-working-on-a-project-with-maos)
6. [Using the Local Model (maos chat / maos ask)](#6-using-the-local-model)
7. [Connecting External AI Agents (MCP)](#7-connecting-external-ai-agents)
8. [The Blackboard: Persistent Goal Tracking](#8-the-blackboard)
9. [Knowledge Graph and RAG](#9-knowledge-graph-and-rag)
10. [Guards and PR Governance](#10-guards-and-pr-governance)
11. [Onboarding and Lessons](#11-onboarding-and-lessons)
12. [Troubleshooting](#12-troubleshooting)

---

## 1. The 30-Second Mental Model

MAOS is not a chatbot. It is a **context engine** that runs locally on your Mac.

```
You work → MAOS observes → MAOS remembers → MAOS understands →
MAOS plans → MAOS acts → MAOS verifies → MAOS learns
```

Two ways to use it:

| Mode | How | Best For |
|------|-----|----------|
| **Direct** | `maos chat` / `maos ask` | Quick questions, local inference |
| **Context backend** | `maos mcp serve` + external AI agent | Deep work, coding, research |

The real power is the **context backend** mode. You connect Claude, Devin, or
any MCP-compatible AI agent to MAOS. The agent calls `maos_get_context()` and
instantly knows: what you're working on, what changed recently, what tasks are
pending, what your codebase looks like, and what lessons you've learned.

---

## 2. First-Time Setup

### Install

```bash
# From source (development)
git clone <repo> ~/mac-ai-os
cd ~/mac-ai-os
uv sync --extra mlx --extra dev
./scripts/setup_models.sh

# Or system-wide (daily use)
cd ~/mac-ai-os
uv tool install . --with mlx-lm
```

### Bootstrap (one command to populate everything)

```bash
maos bootstrap
```

This runs five steps automatically:

1. **Workspace scan** — discovers all project directories in `~/Projects`
2. **Shell history import** — loads `~/.zsh_history` into the database
3. **Telemetry collection** — samples CPU, memory, and processes for 10 seconds
4. **Git indexing** — indexes commits from discovered git repos
5. **Attention cycle** — populates the blackboard with your current goal

Customize the bootstrap:

```bash
# Scan a different directory
maos bootstrap --scan-path ~/Developer

# Skip specific steps
maos bootstrap --skip-telemetry --skip-shell

# Re-run anytime — it's idempotent
maos bootstrap
```

### Verify

```bash
maos verify          # check runtime health
maos context evaluate  # check context quality (aim for 80%+)
```

You should see scores across six dimensions:

| Dimension | What It Measures | How to Improve |
|-----------|-----------------|----------------|
| Workspace | Project discovery + profiling | `maos workspace scan` |
| Session | Activity continuity | Just use your Mac — sessions auto-detect |
| Tasks | TODO awareness | `maos task add "..."` |
| Knowledge | Indexed files + graph | `maos index <path>` |
| Preferences | Learned user habits | Set via `maos preferences set` |
| Activity | Recent events | Use MAOS — events accumulate |

---

## 3. Indexing Your Projects

Indexing is what makes MAOS smart about your code. Without indexing, MAOS knows
your projects exist but can't search their contents.

### Index a single project

```bash
# Index all files in a project (incremental — skips unchanged files)
maos index ~/Projects/my-app

# Force re-index everything (use after major changes)
maos index ~/Projects/my-app --force
```

### Index git history

```bash
# Index commits from a repo (each commit becomes a searchable chunk)
maos git index ~/Projects/my-app

# Check what was indexed
maos git log ~/Projects/my-app
maos git changes ~/Projects/my-app --days 7
maos git files ~/Projects/my-app
```

### Index reference material

MAOS has a `knowledge/` directory for reference content:

```bash
ls ~/mac-ai-os/knowledge/
# apple/ coreml/ metal/ mlx/ papers/ notes/ personal/ xnu/

# Index any of these
maos index ~/mac-ai-os/knowledge/apple
maos index ~/mac-ai-os/knowledge/papers
```

### Watch a directory for live updates

```bash
# Watches for file changes and re-indexes automatically (background thread)
maos watch ~/Projects/my-app
```

### Profile your workspaces

```bash
# Profile all workspaces (uses the utility LLM to summarize each project)
maos workspace profile

# Profile a single workspace
maos workspace profile ~/Projects/my-app

# Profile without the LLM (heuristic only — faster, less detailed)
maos workspace profile --no-llm
```

Profiling discovers: project summary, status (active/abandoned/complete),
open work items (TODO/FIXME comments), and tech stack. Open work items are
automatically created as tasks.

### What to index (recommended order)

| Priority | What | Command |
|----------|------|---------|
| 1 | Your active project | `maos index ~/Projects/<active>` |
| 2 | Git history of active project | `maos git index ~/Projects/<active>` |
| 3 | All workspace code | `maos index ~/Projects` |
| 4 | Reference docs | `maos index ~/mac-ai-os/knowledge/apple` |
| 5 | Personal notes | `maos index ~/mac-ai-os/knowledge/notes` |

### Check what's indexed

```bash
# Context quality report (shows indexed_files count)
maos context evaluate

# Search the index
maos rag "how does the model loader work"

# Knowledge graph stats
maos graph stats
```

---

## 4. The Daily Workflow

### Morning routine

```bash
maos workflow morning
```

This runs the morning workflow:
1. **Reconstruct last session** — what you were doing when you stopped
2. **Summarize changes** — what happened since then (commits, events)
3. **Check pending tasks** — what's on your plate
4. **Check system health** — CPU, memory, disk pressure
5. **Prepare workspace** — open your active project
6. **Create daily plan** — suggest what to work on

### Set your goal for the day

```bash
maos blackboard goal "Ship the authentication refactor by end of day"
```

The blackboard persists this across sessions. Every MAOS interaction inherits
this context — the model knows what you're trying to accomplish.

### Check your context

```bash
# What does MAOS know about me right now?
maos blackboard show

# What tasks are pending?
maos task list

# What was I working on?
maos session summarize
maos session ask "What was I working on yesterday?"
```

### End of day

```bash
# Capture a snapshot of your workspace state
maos snapshot capture

# Check what MAOS learned today
maos learn summary
```

---

## 5. Working on a Project with MAOS

This is the core workflow for using MAOS while actively developing a project.

### Step 1: Make sure your project is indexed

```bash
# Index the code (if not already done)
maos index ~/Projects/my-app

# Index git history
maos git index ~/Projects/my-app

# Profile the workspace
maos workspace profile ~/Projects/my-app
```

### Step 2: Set the goal

```bash
maos blackboard goal "Fix the race condition in the payment processor"
maos blackboard show
```

### Step 3: Create tasks

```bash
maos task add "Write regression test for race condition"
maos task add "Identify the synchronization bug"
maos task add "Fix the locking strategy"
maos task list
```

### Step 4: Start the MCP server

```bash
# For external AI agents (Claude, Devin, etc.)
maos mcp serve --transport http &

# Or for stdio-based clients (Claude Code, Codex)
maos mcp serve
```

### Step 5: Work with your AI agent

Now your AI agent (connected via MCP) can:

- **Get full context**: `maos_get_context()` — returns your goal, workspace,
  tasks, recent activity, knowledge graph entities
- **Search your codebase**: `maos_search_memory("payment processor")` — RAG
  search over indexed files and git commits
- **Create tasks**: `maos_create_task("fix the lock")` — goes through the
  permissioned action executor
- **List tasks**: `maos_list_tasks()` — see what's pending
- **Get workspace summary**: `maos_workspace_summary()` — git state, modified
  files, recent commits
- **Record lessons**: `maos_record_lesson("race_condition_fix", "The bug was
  in the double-checked locking pattern...")` — persists for future sessions
- **Scan for vibes**: `maos_scan_vibes("everything is working")` — checks PR
  text for vague claims without evidence

### Step 6: Use the local model for quick questions

```bash
# One-shot question (uses the local fine-tuned model)
maos ask "What files changed in the last commit?"

# Interactive chat
maos chat

# RAG question (searches indexed content)
maos rag "How does the model loader work in this codebase?"
```

### Step 7: Track progress

```bash
# Complete tasks
maos task done <task-id>

# Check the blackboard (it updates as you work)
maos blackboard show

# Run an attention cycle (updates suggested actions)
maos blackboard cycle
```

### Step 8: Before opening a PR

```bash
# Produce test evidence (actually runs the tests)
maos guards verify "uv run pytest" --save test-evidence.json

# Produce lint evidence
maos guards verify "ruff check app tests" --save lint-evidence.json

# Inspect the diff for secrets, debug statements, scope violations
maos guards diff --json

# Check PR readiness (state machine: DRAFT → UNDERSTOOD → VERIFIED → PR_READY)
maos guards readiness \
  --test-evidence test-evidence.json \
  --lint-evidence lint-evidence.json \
  --pr-description pr-description.md
```

If the state is not `PR_READY`, do NOT open the PR. Fix the blockers first.

---

## 6. Using the Local Model

MAOS ships with a local model for on-device inference. The current setup uses
a fine-tuned Qwen3.5-9B in MLX 4-bit format (~5.6GB in memory).

### Quick questions

```bash
# FAST mode (default for simple questions)
maos ask "What is 2+2?"

# THINK mode (for complex questions — uses reasoning)
maos ask "Design a microservices architecture for a payment system"
```

The adaptive router automatically classifies your prompt and selects FAST or
THINK mode. You'll see the routing decision:

```
[FAST] confidence=95% family=unknown diff=4
Hello! How can I help you today?
```

### Interactive chat

```bash
maos chat
```

Type messages and get responses. The conversation history is maintained within
the session. Ctrl-D or Ctrl-C to exit.

### RAG questions (search indexed content)

```bash
maos rag "How does the authentication system work?"
```

This searches your indexed files and git commits, then uses the model to
synthesize an answer with citations.

### Model configuration

Models are configured in `configs/user.yaml`:

```yaml
planner:
  provider: mlxvlm
  model: /path/to/your/model
  temperature: 0.7
  max_tokens: 1024

utility:
  provider: mlx
  model: mlx-community/Llama-3.2-3B-Instruct-4bit
```

Check available models:

```bash
maos models
```

### Performance notes

- **Cold start**: ~5 seconds to load the 9B model into unified memory
- **Warm inference**: ~3-20 seconds per response depending on output length
- **Memory**: ~6GB for the 9B planner model (utility model is lazy-loaded)
- **FAST mode**: Should be ~3-5 seconds for simple questions
- **THINK mode**: 15-30+ seconds for complex reasoning

If responses are slow:
1. Check if another heavy process is running (`maos telemetry status`)
2. Make sure the utility model is lazy-loaded (check logs for `model.build`
   with `role=utility` — it should NOT appear at startup)
3. Use `maos ask` for one-shot questions instead of `maos chat` if you
   don't need conversation history

---

## 7. Connecting External AI Agents

This is where MAOS becomes a force multiplier. Instead of replacing your AI
tools, MAOS makes them smarter by providing persistent context.

### Start the MCP server

```bash
# Verify the selected agent path without exposing credential values
maos agent doctor --provider all --json
maos agent brief --provider <provider> --task "<task>"

# HTTP transport (for web-based agents; authenticated by default)
maos mcp serve --transport http --port 7778 &

# stdio transport (for Claude Code, Codex)
maos mcp serve
```

Do not disable MCP authentication for convenience. The default token is kept
outside the repository; clients should receive it through protected runtime
configuration rather than prompts, logs, or tracked agent files.

### Connect Claude Code

```bash
claude mcp add maos -- uv run --project ~/mac-ai-os maos mcp serve
```

### Connect Devin CLI

Add to `~/.config/devin/config.json`:

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

### Connect any MCP-compatible client

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

### What the AI agent gets

When your AI agent calls `maos_get_context()`, it receives:

```json
{
  "goal": "Fix the race condition in the payment processor",
  "workspace": {
    "name": "my-app",
    "path": "/Users/you/Projects/my-app",
    "language": "python",
    "git_branch": "fix/race-condition",
    "git_status": "2 modified, 1 untracked",
    "profile": "Payment processing app with Flask..."
  },
  "tasks": [
    {"title": "Write regression test", "status": "pending", "priority": "high"},
    {"title": "Fix the locking strategy", "status": "pending", "priority": "high"}
  ],
  "recent_activity": [...],
  "recent_commits": [...],
  "knowledge_entities": [...],
  "suggested_actions": [...]
}
```

This means the AI agent starts every conversation already knowing what you're
working on. No more "let me look at the codebase first" — it already knows.

### MCP tools available to the agent

| Tool | What it does |
|------|-------------|
| `maos_get_context` | Full context packet (goal, workspace, tasks, activity) |
| `maos_search_memory` | RAG search over indexed files + git commits |
| `maos_workspace_summary` | Git state, modified files, recent commits |
| `maos_create_task` | Create a TODO item (permissioned) |
| `maos_list_tasks` | List pending tasks |
| `maos_run_action` | Execute a registered action (permissioned) |
| `maos_onboard` | Get the onboarding briefing (lessons, preferences, state) |
| `maos_record_lesson` | Record a hard-won lesson for future sessions |
| `maos_list_lessons` | List all recorded lessons |
| `maos_send_email` | Send an email via Gmail OAuth |
| `maos_escalate` | Send an urgent email (needs human attention) |
| `maos_scan_vibes` | Scan text for vague claims without evidence |
| `maos_check_guards` | Run PR governance checks |
| `maos_diagnostics` | Check all MCP tools are working |

---

## 8. The Blackboard

The blackboard is MAOS's shared working memory. It persists across sessions
and is inherited by every capability.

### Set your current goal

```bash
maos blackboard goal "Ship the authentication refactor"
```

This is the single most important command. Every MAOS interaction and every
external AI agent conversation inherits this goal. The agent knows what you're
trying to accomplish without you repeating it.

### Check the blackboard

```bash
maos blackboard show
```

Shows: current goal, active workspace, open problems, recent decisions,
suggested actions, and confidence level.

### Run an attention cycle

```bash
maos blackboard cycle
```

This recomputes: active workspace, current goal (from pending tasks), system
health, open problems (blocked tasks, failed actions), and suggested actions.
Updates the blackboard and persists.

### Clear the blackboard

```bash
maos blackboard clear
```

Use this when you're switching to a completely new project or context.

---

## 9. Knowledge Graph and RAG

### Build the knowledge graph

```bash
# Extract entities from indexed content using the LLM
maos graph extract "payment processing"

# Add a relationship manually
maos graph add PaymentProcessor uses Stripe

# Search the graph
maos graph search "payment"

# Stats
maos graph stats
```

### RAG queries

```bash
# Search indexed files + git commits
maos rag "How does the authentication middleware work?"

# The research agent combines web search + local RAG + LLM synthesis
# It cites both web and local sources
```

### What gets indexed

| Source | What's Stored | How to Index |
|--------|--------------|-------------|
| Code files | Full content, chunked | `maos index <path>` |
| Git commits | Commit message + files changed | `maos git index <repo>` |
| Markdown | Full content | `maos index <path>` |
| PDFs | Extracted text | `maos index <path>` |
| Reference docs | Full content | `maos index ~/mac-ai-os/knowledge/` |

The indexer is **hash-aware**: it skips unchanged files, re-indexes changed
files, and removes deleted files. Running `maos index` on the same directory
twice is fast — only new/changed files are processed.

---

## 10. Guards and PR Governance

MAOS includes an evidence-producing PR governance system. The core principle:
**a boolean should represent the RESULT of a verification process, not REPLACE
the verification process.**

### Quick vibe scan

```bash
# Check any text for vague claims
maos guards scan "everything is working and all tests pass"
```

### Full PR readiness workflow

```bash
# 1. Produce test evidence (actually runs the tests)
maos guards verify "uv run pytest" --save test-evidence.json

# 2. Produce lint evidence
maos guards verify "ruff check app tests" --save lint-evidence.json

# 3. Write a structured PR description (pr-description.md)
# Must include: Summary, Why, What Changed, Scope
# For bug fixes: Root Cause, Regression Test, Before, After

# 4. Check PR readiness
maos guards readiness \
  --test-evidence test-evidence.json \
  --lint-evidence lint-evidence.json \
  --pr-description pr-description.md \
  --bug-fix
```

The state machine: `DRAFT → UNDERSTOOD → VERIFIED → PR_READY`

If the state is not `PR_READY`, do NOT open the PR. Fix the blockers.

### What the guards check for

| Check | Severity | What It Catches |
|-------|----------|----------------|
| Secrets in diff | BLOCK | API keys, tokens, passwords |
| Debug statements | WARN | print(), console.log(), debugger |
| MCP schema changes | REQUIRE_REVIEW | Tool definitions changed |
| Security paths | REQUIRE_REVIEW | Auth, crypto, permission files |
| Dependency changes | WARN | New packages in requirements |
| Scope violations | WARN | Changes outside stated scope |
| Vibe language | BLOCK | "everything works" without evidence |
| Missing tests | BLOCK | No test evidence for PR |

---

## 11. Onboarding and Lessons

MAOS learns from your mistakes so future AI sessions don't repeat them.

### Get the onboarding briefing

```bash
maos onboard
```

Returns: user bio, communication style, coding preferences, 46+ recorded
lessons (with confidence scores), recent session, active workspace, open tasks,
and known failure patterns.

This is what external AI agents call via `maos_onboard()` at the start of every
session. It's the cold-start antidote.

### Record a lesson

When you learn something the hard way (a fix was wrong, an assumption was
incorrect, a workflow rule was violated):

```bash
maos onboard record race_condition_fix "The bug was in the double-checked locking pattern. Always use a single lock acquisition, not a check-then-lock-then-check pattern."
```

Or via MCP (from your AI agent):

```
maos_record_lesson(slug="race_condition_fix", text="...")
```

Lessons are persisted and surfaced in every future onboarding briefing. They
include trigger keywords so they fire when relevant patterns appear.

### Escalate when something needs human attention

```bash
maos onboard escalate "Gmail OAuth expired" "The Gmail integration has been failing for 3 days. Token refresh returns 400. User needs to re-authenticate at ~/.gmail-mcp/"
```

This sends a real email to the configured MAOS recipient with a [MAOS] prefix.
Use it when:
- A fix didn't work and you need to explain why
- You're about to do something irreversible
- Context quality is degrading
- A lesson was just learned the hard way

### List all lessons

```bash
maos onboard lessons
```

---

## 12. Troubleshooting

### "Context quality is low"

```bash
maos context evaluate
```

Check which dimension is low, then run the specific command:

| Low Dimension | Fix |
|---------------|-----|
| Workspace | `maos workspace scan` then `maos workspace profile` |
| Session | Just use your Mac — sessions auto-detect from activity |
| Tasks | `maos task add "..."` for each TODO |
| Knowledge | `maos index ~/Projects` and `maos git index` each repo |
| Preferences | `maos preferences set <key> <value>` |
| Activity | Run `maos bootstrap` or just use MAOS more |

### "Local model is slow"

1. Check for competing processes: `maos telemetry status`
2. Make sure utility model is lazy-loaded (no `role=utility` in startup logs)
3. Use `maos ask` for one-shot questions (no conversation overhead)
4. Check memory pressure: `maos storage scan --mode fast`

### "MCP server won't start"

```bash
maos mcp status    # check if it's already running
maos mcp test      # test the connection
maos diagnostics   # check all tools
```

### "Model not found"

```bash
maos models        # list configured models
maos verify        # check runtime health
```

Check `configs/user.yaml` — the model path must point to a valid MLX model
directory.

### "Gmail integration failing"

The Gmail OAuth token expires periodically. Re-authenticate:

```bash
# Check the credentials directory
ls ~/.gmail-mcp/

# The token needs to be refreshed via the OAuth flow
# See docs/getting-started.md for setup instructions
```

### "Disk space issues"

```bash
# Scan for what's taking up space
maos storage scan --mode full

# Find hidden system data
maos storage hidden

# Generate a cleanup script
maos storage fix-script ~/Downloads --no-caches

# Run maintenance
maos storage maintain
```

---

## Quick Reference Card

```bash
# Daily workflow
maos workflow morning              # morning routine
maos blackboard goal "..."         # set today's goal
maos blackboard show               # check current state
maos task add "..."                # create a task
maos task list                     # see pending tasks
maos task done <id>                # complete a task

# Indexing
maos index ~/Projects/my-app       # index a project
maos git index ~/Projects/my-app   # index git history
maos workspace profile             # profile all workspaces
maos rag "..."                     # search indexed content

# Local model
maos ask "..."                     # one-shot question
maos chat                          # interactive chat
maos rag "..."                     # RAG question

# MCP server (for external AI agents)
maos mcp serve --transport http &  # start HTTP server
maos mcp status                    # check status
maos mcp test                      # test connection

# PR governance
maos guards verify "pytest" --save test.json
maos guards diff --json
maos guards readiness --test-evidence test.json --pr-description pr.md

# Onboarding and lessons
maos onboard                       # get briefing
maos onboard record <slug> "..."   # record a lesson
maos onboard escalate "..." "..."  # send urgent email

# Context and health
maos context evaluate              # context quality score
maos diagnostics                   # system health
maos telemetry status              # system resources
maos time                          # current time awareness
```

---

## The Bottom Line

MAOS works best when you:

1. **Index your projects** — `maos index` and `maos git index` for each project
2. **Set your goal** — `maos blackboard goal "..."` every time you switch context
3. **Create tasks** — `maos task add "..."` for everything you need to do
4. **Connect your AI agent** — `maos mcp serve` + Claude/Devin/Codex
5. **Record lessons** — `maos onboard record` when you learn something
6. **Run the morning workflow** — `maos workflow morning` to start each day
7. **Use guards before PRs** — `maos guards readiness` to verify before shipping

The system gets smarter the more you use it. Every event, every task, every
lesson, every commit feeds back into the context that makes the next
interaction better.
