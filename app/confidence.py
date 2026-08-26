"""Provider-neutral confidence composition with explicit semantic boundaries."""

from __future__ import annotations

from typing import Any


def compose(*, trajectory: float | None, trustcard: float | None, cvconform: float | None) -> dict[str, Any]:
    """Compose confidence without allowing tool/runtime scores to prove truth.

    Trajectory is the epistemic anchor. Trustcard gates provenance/authorization;
    CVConform describes runtime conformance only. Missing components lower the
    completeness flag but never become fabricated zero-confidence claims.
    """
    components = {"trajectory": trajectory, "trustcard": trustcard, "cvconform": cvconform}
    available = [v for v in components.values() if v is not None]
    return {
        "components": components,
        "epistemic_confidence": trajectory,
        "operational_confidence": min(available) if available else None,
        "complete": len(available) == 3,
        "rule": "trajectory determines truth confidence; trustcard gates source/tool trust; cvconform is runtime-only",
    }
