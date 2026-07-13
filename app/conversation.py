"""Conversation API.

A thin stateful wrapper over the orchestrator that keeps message history and
exposes a single `send(text) -> str` entry point for the CLI/GUI/voice.
"""

from __future__ import annotations

from app.log import get_logger
from app.models.types import Message
from app.orchestrator import AgentOrchestrator

log = get_logger(__name__)


class Conversation:
    def __init__(self, *, orchestrator: AgentOrchestrator, system_prompt: str | None = None):
        self.orchestrator = orchestrator
        self.system_prompt = system_prompt or (
            "You are Mac AI OS, a local-first intelligence layer for macOS. "
            "You reason, retrieve, choose tools, execute, verify, and explain."
        )
        self.history: list[Message] = [Message("system", self.system_prompt)]

    def send(self, text: str) -> str:
        log.info("conversation.send", text=text[:80])
        result = self.orchestrator.run(text)
        self.history.append(Message("user", text))
        self.history.append(Message("assistant", result.answer or result.error or ""))
        return result.answer or result.error or "(no answer)"

    def reset(self) -> None:
        self.history = [Message("system", self.system_prompt)]
