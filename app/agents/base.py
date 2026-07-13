"""Agent abstraction.

Agents are independent. Each one owns a domain (repos, filesystem, research,
the system) and exposes the same four-method surface:

    plan(goal)      -> list[str]   # proposed steps
    execute(goal)   -> AgentResult # carry out the plan
    describe()      -> dict        # capabilities
    health()        -> bool        # ready to run?

Agents receive their dependencies (model, retriever, tools) via constructor
injection — no globals, no singletons.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.models.base import Model
from app.retrieval.base import Retriever
from app.tools.registry import ToolRouter


@dataclass
class AgentResult:
    ok: bool
    answer: str = ""
    steps: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class Agent(ABC):
    name: str = "base"
    description: str = ""

    def __init__(
        self,
        *,
        planner: Model | None = None,
        utility: Model | None = None,
        retriever: Retriever | None = None,
        tools: ToolRouter | None = None,
    ):
        self.planner = planner
        self.utility = utility
        self.retriever = retriever
        self.tools = tools

    @abstractmethod
    def plan(self, goal: str) -> list[str]:
        ...

    @abstractmethod
    def execute(self, goal: str) -> AgentResult:
        ...

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description}

    @abstractmethod
    def health(self) -> bool:
        ...
