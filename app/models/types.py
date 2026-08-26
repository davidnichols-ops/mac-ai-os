"""Shared types for model providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A parsed tool call from model output.

    The model emits tool calls in a structured text format. The provider
    parses them into ToolCall objects. This is model-agnostic — any model
    that can produce structured output can generate tool calls.
    """
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """A conversation message.

    Roles: system, user, assistant, tool.
    Tool messages include tool_call_id and name for correlation.
    Assistant messages may include tool_calls when the model requests actions.
    """
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None  # tool name for tool result messages

    def to_dict(self) -> dict[str, Any]:
        import json as _json
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {
                    "name": tc.name, "arguments": _json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else str(tc.arguments)
                }}
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ModelResponse:
    """Response from a model generation.

    text: the raw text output (may contain tool calls in structured format)
    tool_calls: parsed tool calls (empty if model didn't call any tools)
    mode: adaptive compute mode ("fast" | "think" | None for v2 control tokens)
    finish_reason: why generation stopped ("stop" | "tool_calls" | "length")
    """
    text: str
    model: str
    provider: str
    raw: Any = None
    usage: dict[str, int] = field(default_factory=dict)
    mode: str | None = None  # "fast" | "think" | None (for v2 control tokens)
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"  # "stop" | "tool_calls" | "length"
    # Inference metrics
    generation_time_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0
    peak_memory_mb: float = 0.0

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    def __str__(self) -> str:
        return self.text


@dataclass
class EmbeddingResponse:
    vector: list[float]
    model: str
    provider: str
