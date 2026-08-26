"""CEO operating office for portfolio vision, delegation, and human briefs.

The office turns MAOS observations into an auditable operating loop:

    observe -> choose a portfolio thesis -> delegate bounded work -> verify -> brief David

It intentionally produces recommendations and tasks, not silent high-impact actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.db import Database


@dataclass(frozen=True, slots=True)
class Delegation:
    """A bounded assignment for an agent or the human operator."""

    title: str
    owner: str
    why: str
    deliverable: str
    verification: str
    priority: int = 0


@dataclass(frozen=True, slots=True)
class PortfolioVision:
    """A durable, human-readable portfolio decision artifact."""

    thesis: str
    current_focus: str
    projects: tuple[str, ...]
    delegations: tuple[Delegation, ...]
    insight: tuple[str, ...]
    generated_at: str


def _project_name(row: Any) -> str:
    return str(row["name"] if row["name"] is not None else "unknown")


def build_vision(workspaces: list[Any], pending_tasks: list[Any]) -> PortfolioVision:
    """Build a deterministic portfolio brief from MAOS evidence.

    This is deliberately evidence-first. A model can later enrich the prose,
    but it must not invent the inventory or silently change priorities.
    """
    names = tuple(_project_name(row) for row in workspaces)
    normalized = {name.lower(): name for name in names}
    flagship = next(
        (normalized[key] for key in ("mac-ai-os", "infernet", "trajectory-os") if key in normalized),
        names[0] if names else "the portfolio",
    )
    active = next((name for name in names if name.lower() in {"infernet", "mac-ai-os"}), flagship)
    pending_count = len(pending_tasks)

    thesis = (
        "Build a coherent personal AI infrastructure company: MAOS is the operating layer, "
        "Infernet is the distributed compute fabric, and the remaining projects become "
        "validated capabilities rather than competing destinations."
    )
    focus = (
        f"Make {active} the next execution surface. Keep one measurable milestone active; "
        "everything else is either a dependency, a research track, or explicitly parked."
    )
    insight = (
        f"MAOS currently sees {len(names)} workspace(s) and {pending_count} pending task(s).",
        "The highest-leverage bottleneck is portfolio sequencing, not idea generation.",
        "Every delegation needs an owner, a concrete artifact, and an executable verification command.",
    )
    delegations = (
        Delegation(
            title="Produce the next milestone implementation plan",
            owner="architecture-agent",
            why="Convert the active project vision into a bounded sequence of shippable changes.",
            deliverable="A dated milestone plan with dependencies, risks, and explicit non-goals.",
            verification="Review the plan against the repository state and run the listed checks.",
            priority=100,
        ),
        Delegation(
            title="Audit active project readiness",
            owner="verification-agent",
            why="Prevent strategy from outrunning the actual code and test evidence.",
            deliverable="A short readiness report covering tests, lint, security, docs, and blockers.",
            verification="Run the repository's real test and lint commands; attach captured evidence.",
            priority=90,
        ),
        Delegation(
            title="Execute the highest-value implementation slice",
            owner="implementation-agent",
            why="Turn the approved milestone into one reviewable change instead of parallel drift.",
            deliverable="One focused commit with tests and a concise operator handoff.",
            verification="Run tests, inspect the diff, and record before/after behavior.",
            priority=80,
        ),
        Delegation(
            title="Human decision gate",
            owner="david",
            why="The CEO layer advises and delegates; irreversible priorities remain human-owned.",
            deliverable="Choose proceed, reprioritize, or park after reading the insight and evidence.",
            verification="Record the decision and rationale in MAOS before execution continues.",
            priority=70,
        ),
    )
    return PortfolioVision(
        thesis=thesis,
        current_focus=focus,
        projects=names,
        delegations=delegations,
        insight=insight,
        generated_at=datetime.now(UTC).isoformat(),
    )


def render_vision(vision: PortfolioVision) -> str:
    """Render the portfolio vision as Markdown for humans and agents."""
    lines = [
        "# MAOS CEO Office — Portfolio Vision",
        "",
        f"Generated: `{vision.generated_at}`",
        "",
        "## Thesis",
        vision.thesis,
        "",
        "## Current focus",
        vision.current_focus,
        "",
        "## Evidence and insight",
        *[f"- {item}" for item in vision.insight],
        "",
        "## Portfolio",
        *( [f"- `{name}`" for name in vision.projects] or ["- No workspaces registered yet."] ),
        "",
        "## Delegation queue",
    ]
    for index, item in enumerate(vision.delegations, start=1):
        lines.extend(
            [
                f"### {index}. {item.title}",
                f"- **Owner:** `{item.owner}`",
                f"- **Why:** {item.why}",
                f"- **Deliverable:** {item.deliverable}",
                f"- **Verification:** `{item.verification}`",
                f"- **Priority:** {item.priority}",
                "",
            ]
        )
    lines.extend(
        [
            "## Operating contract",
            "1. Observe from MAOS evidence before proposing work.",
            "2. State the portfolio thesis and the reason for the current focus.",
            "3. Delegate bounded artifacts to named agents or David.",
            "4. Require executable verification; agreement is not proof.",
            "5. Surface uncertainty and blockers in the human brief.",
            "6. Log decisions and outcomes so the next agent can continue the pattern.",
            "",
            "## Agent handoff format",
            "Use `owner`, `why`, `deliverable`, `verification`, `priority`, and `status`. Do not emit vague goals such as `make it better`.",
            "",
        ]
    )
    return "\n".join(lines)


def generate_vision(db: Database, output: str | Path) -> PortfolioVision:
    """Generate and persist the current CEO portfolio artifact."""
    vision = build_vision(db.all_workspaces(), db.pending_tasks())
    destination = Path(output).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_vision(vision), encoding="utf-8")
    db.log_decision(
        "ceo-office",
        f"set portfolio focus: {vision.current_focus}",
        reasoning=vision.thesis,
        confidence=0.75,
        kind="portfolio_vision",
        tags=["ceo", "vision", "delegation"],
    )
    return vision


def delegate(db: Database, vision: PortfolioVision) -> list[int]:
    """Materialize the delegation queue as auditable MAOS tasks."""
    task_ids: list[int] = []
    for item in vision.delegations:
        task_ids.append(
            db.create_task(
                f"[{item.owner}] {item.title} — {item.deliverable}",
                source="ceo-office",
                priority=item.priority,
            )
        )
    return task_ids
    return task_ids
