# Tool System — Capabilities for the Adaptive Operator

## Overview

Tools are the model's hands. The model decides what to do; tools execute it.
Every tool has a schema (what the model sees), permissions (what's allowed),
and a risk level (how dangerous it is).

## Tool Schema

Every tool provides a JSON schema that's injected into the system prompt:

```python
def schema(self) -> dict[str, Any]:
    return {
        "name": "shell",
        "description": "Run a shell command...",
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The command to run"},
        },
        "required": ["command"],
    }
```

## Built-in Tools

| Tool | Risk | Permissions | Description |
|------|------|-------------|-------------|
| `shell` | MEDIUM | shell.exec | Run shell commands |
| `file_read` | LOW | file.read | Read file contents |
| `file_write` | MEDIUM | file.write | Write file contents |
| `file_edit` | MEDIUM | file.write | Surgical find/replace edit |
| `file_list` | LOW | file.read | List directory contents |
| `git` | MEDIUM | git.exec | Run git subcommands |

## Risk Levels

| Level | Description | Auto-approved? |
|-------|-------------|----------------|
| LOW | Read-only, no side effects | Yes |
| MEDIUM | Side effects, reversible | Yes |
| HIGH | Potentially destructive | No — requires user approval |
| CRITICAL | Irreversible damage | No — always blocked |

## Safety Gate

The SafetyGate (`app/core/safety.py`) evaluates every tool call:

1. Check for hard-blocked commands (rm -rf /, mkfs, dd, etc.)
2. Check for high-risk command patterns (rm, git reset --hard, sudo, etc.)
3. Check tool-level risk (file_delete=HIGH, process_kill=HIGH)
4. Auto-approve LOW and MEDIUM
5. Request user approval for HIGH and CRITICAL

## Tool Executor

The ToolExecutor (`app/core/tool_executor.py`) wraps the existing ToolRouter with:
- Safety gate evaluation
- Trajectory recording
- Result formatting (truncation, error handling)
- Approval callback integration

## Adding New Tools

```python
from app.tools.base import Tool, ToolResult

class MyTool(Tool):
    name = "my_tool"
    description = "Does something useful"
    permissions = {"my.permission"}

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "A parameter"},
            },
            "required": ["param"],
        }

    def execute(self, args: dict[str, Any]) -> ToolResult:
        param = args.get("param")
        if not param:
            return ToolResult(ok=False, error="missing 'param'")
        # ... do the work ...
        return ToolResult(ok=True, output="result")
```

Register in `app/tools/builtin.py`:
```python
def builtin_tools() -> list[Tool]:
    return [..., MyTool()]
```

And grant the permission in `app/runtime.py`:
```python
tools = ToolRouter(builtin_tools(), granted_permissions={
    ..., "my.permission",
})
```
