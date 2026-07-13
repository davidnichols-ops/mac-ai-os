"""Shared types for model providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ModelResponse:
    text: str
    model: str
    provider: str
    raw: Any = None
    usage: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text


@dataclass
class EmbeddingResponse:
    vector: list[float]
    model: str
    provider: str
