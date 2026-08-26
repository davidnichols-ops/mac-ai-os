"""Conversation API.

A thin stateful wrapper over the orchestrator that keeps message history and
exposes a single `send(text) -> str` entry point for the CLI/GUI/voice.

For v2 models with control tokens, the model self-selects FAST/THINK mode
via <FAST>/<THINK> tokens at the start of generation. No external routing
needed. For v1 models, the adaptive router can still be used.
"""

from __future__ import annotations

from app.adaptive_router import AdaptiveRouter
from app.log import get_logger
from app.models.types import Message
from app.orchestrator import AgentOrchestrator

log = get_logger(__name__)


class Conversation:
    def __init__(
        self,
        *,
        orchestrator: AgentOrchestrator,
        system_prompt: str | None = None,
        adaptive_router: AdaptiveRouter | None = None,
    ):
        self.orchestrator = orchestrator
        self.adaptive_router = adaptive_router
        self.system_prompt = system_prompt or (
            "You are Mac AI OS, a local-first intelligence layer for macOS. "
            "You reason, retrieve, choose tools, execute, verify, and explain."
        )
        self.history: list[Message] = [Message("system", self.system_prompt)]
        self._last_routing: dict | None = None

    def send(self, text: str) -> str:
        log.info("conversation.send", text=text[:80])

        # For v2 models with control tokens, the model self-selects mode.
        # We pass enable_thinking=None to let the model decide.
        # For v1 models with an adaptive router, use the router.
        if self.adaptive_router is not None:
            mode, confidence, meta = self.adaptive_router.route(text)
            self._last_routing = {"mode": mode, "confidence": confidence, **meta}
            log.info(
                "conversation.adaptive_route",
                mode=mode,
                confidence=confidence,
                family=meta.get("family"),
                difficulty=meta.get("difficulty"),
                method=meta.get("method"),
            )
            result = self.orchestrator.run(text, enable_thinking=(mode == "THINK"))
        else:
            # No router — let the model self-select via control tokens (v2)
            self._last_routing = None
            result = self.orchestrator.run(text, enable_thinking=None)

        self.history.append(Message("user", text))
        self.history.append(Message("assistant", result.answer or result.error or ""))
        return result.answer or result.error or "(no answer)"

    @property
    def last_routing(self) -> dict | None:
        """Return metadata about the last adaptive routing decision."""
        return self._last_routing

    def reset(self) -> None:
        self.history = [Message("system", self.system_prompt)]
        self._last_routing = None
