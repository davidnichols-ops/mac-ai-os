# Safety — Risk Classification and Approval Enforcement

## Overview

The safety layer is the trust boundary between the model's intent and the
environment's execution. The model decides what to do; the safety gate
decides whether it's allowed.

## Risk Levels

```
LOW (0)      → read-only, no side effects
MEDIUM (1)   → side effects, but reversible
HIGH (2)     → potentially destructive, may not be reversible
CRITICAL (3) → irreversible damage, always requires approval
```

## Evaluation Flow

```
Tool Call (name, args)
    |
    v
SafetyGate.evaluate()
    |
    +-- Check blocked commands (rm -rf /, mkfs, dd, etc.) → BLOCK
    +-- Check high-risk command patterns (rm, git reset, sudo, etc.) → APPROVAL
    +-- Check tool-level risk (file_delete=HIGH, process_kill=HIGH) → APPROVAL
    +-- Auto-approve LOW and MEDIUM
    +-- Request user approval for HIGH and CRITICAL
    |
    v
SafetyDecision { allowed, risk, requires_approval }
```

## Blocked Commands (CRITICAL — always blocked)

- `rm -rf /`, `rm -rf ~`, `rm -rf *`, `rm -rf .`
- `mkfs`, `dd if=`, `shred`
- `diskutil eraseDisk`
- `chmod 777 /`, `chown -R root`

## High-Risk Commands (require approval)

- `rm`, `rmdir`
- `git reset --hard`, `git push --force`, `git push -f`
- `git clean -fd`, `git branch -D`, `git rebase`
- `sudo`, `su`
- `pip install`, `npm install`, `brew install`
- `kill -9`, `killall`, `pkill`
- `curl`, `wget`, `scp`, `rsync`

## High-Risk Tools

- `file_delete` → HIGH
- `process_kill` → HIGH
- `git_push` → HIGH
- `git_reset` → HIGH

## Approval Callback

When a HIGH or CRITICAL operation is requested, the safety gate calls the
approval callback:

```python
def approval_callback(risk: RiskLevel, tool_name: str, args: dict) -> bool:
    # Show the user what's being requested
    # Return True to approve, False to deny
    ...
```

In the CLI (`maos do`), this shows a panel with the tool name, risk level,
and arguments, then asks for confirmation.

## Implementation

- `app/core/safety.py` — SafetyGate, RiskLevel, SafetyDecision
- `app/core/tool_executor.py` — ToolExecutor wraps ToolRouter with SafetyGate
- `app/cli/commands/do.py` — CLI approval callback

## Design Principle

The model must know what is dangerous and what requires confirmation.
The system prompt tells the model: "Dangerous operations require user approval.
Respect safety decisions."

The safety gate is NOT intelligence — it's a hard boundary. The model can
request anything; the gate decides what's permitted.
