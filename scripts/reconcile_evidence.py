"""Idempotent evidence reconciliation for the Local Operator release."""

from __future__ import annotations

import json
from pathlib import Path

from app.confidence import compose
from app.db import Database

RELEASE = "https://huggingface.co/davidnichols-ops/adaptive-operator-v4-gguf"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    db = Database(root / "data/maos.db")
    data_dir = root / "data"
    artifact_path = data_dir / "external_artifacts.json"
    artifact = {
        "artifact": "adaptive-operator-v4-gguf",
        "url": RELEASE,
        "project": "Local Operator Distillation",
        "status": "shipped_experiment",
        "result_classification": "useful_failure",
        "source": "user_confirmed_external_release",
        "validation_limit": "release existence does not prove every Phase 0-3 milestone",
    }
    artifact_path.write_text(json.dumps(artifact, indent=2) + "\n")

    parent = db.query_one("SELECT * FROM tasks WHERE id = 522")
    changes: list[str] = []
    if parent and parent["status"] != "completed":
        db.complete_task(522)
        changes.append("completed parent task 522 as shipped experiment")
    existing = {row["title"] for row in db.pending_tasks()}
    followups = [
        ("Validate adaptive-operator-v4-gguf release against reproducible inference checks", 522, 9),
        ("Document useful-failure lessons from Local Operator Distillation", 522, 8),
    ]
    for title, parent_id, priority in followups:
        if title not in existing:
            db.create_task_with_parent(title, parent_id=parent_id, source="evidence-derived", related_workspace="local-operator", priority=priority)
            changes.append(f"created proposed follow-up: {title}")
    report = {
        "artifact": artifact,
        "changes": changes,
        "confidence": compose(trajectory=None, trustcard=None, cvconform=None),
        "stale_phase_guard": "parent shipped; no Phase 0-3 continuation tasks generated",
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
