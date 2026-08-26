"""Execution state — mutable state tracking during an agent loop run.

The ExecutionState tracks everything that happens during a single agent loop
invocation. It is the runtime's working memory — distinct from the Blackboard
(which is cross-session shared memory) and the Trajectory (which is the
immutable record).

State tracked:
    - Current step number
    - Messages (conversation history)
    - Seen actions (for infinite loop detection)
    - Tool call count
    - Error count
    - Whether the loop is finished
    - The final answer (if any)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.models.types import Message


@dataclass
class ExecutionState:
    """Mutable state during a single agent loop execution.

    This is created fresh for each `maos do` invocation. It is NOT
    persisted across sessions — that's what the Blackboard and Trajectory
    are for. This is just the working state of the current loop.
    """
    # The user's goal
    goal: str = ""

    # Conversation history (system + user + assistant + tool messages)
    messages: list[Message] = field(default_factory=list)

    # Step tracking
    step: int = 0
    max_steps: int = 30

    # Action tracking for infinite loop detection
    # Stores (tool_name, args_hash) tuples
    seen_actions: list[tuple[str, str]] = field(default_factory=list)

    # Error tracking
    errors: int = 0
    max_errors: int = 10

    # Tool call tracking
    tool_calls: int = 0

    # Completion
    finished: bool = False
    final_answer: str = ""

    # Workspace (for file operations)
    workspace: str = "."

    # Config
    max_history_messages: int = 20

    @property
    def is_step_budget_exceeded(self) -> bool:
        return self.step >= self.max_steps

    @property
    def is_error_budget_exceeded(self) -> bool:
        return self.errors >= self.max_errors

    def check_infinite_loop(self, tool_name: str, args: dict[str, Any]) -> bool:
        """Check if we've seen this exact tool call recently.

        Returns True if this (tool_name, args) pair was seen in the last
        3 calls — indicates an infinite loop.
        """
        key = (tool_name, json.dumps(args, sort_keys=True))
        if key in self.seen_actions[-3:]:
            return True
        self.seen_actions.append(key)
        return False

    def truncate_history(self) -> None:
        """Truncate conversation history to prevent unbounded growth.

        Keeps the system prompt + initial user message + last N messages.
        """
        if len(self.messages) <= self.max_history_messages:
            return
        # Keep first 2 (system + user) and last N-2
        keep = self.max_history_messages
        self.messages = self.messages[:2] + self.messages[-(keep - 2):]

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(self, content: str, tool_calls: list | None = None) -> None:
        self.messages.append(Message(
            role="assistant", content=content,
            tool_calls=tool_calls or [],
        ))

    def add_tool_result(self, tool_call_id: str, name: str, content: str) -> None:
        self.messages.append(Message(
            role="tool", content=content,
            tool_call_id=tool_call_id, name=name,
        ))
