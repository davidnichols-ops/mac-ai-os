from __future__ import annotations

from app.ceo_office import build_vision, delegate, generate_vision, render_vision
from app.db import Database


def test_build_vision_is_evidence_backed() -> None:
    vision = build_vision([], [])

    assert "MAOS" in vision.thesis
    assert vision.projects == ()
    assert len(vision.delegations) == 4
    assert all(item.owner and item.deliverable and item.verification for item in vision.delegations)


def test_generate_vision_logs_decision_and_writes_artifact(tmp_path) -> None:
    db = Database(str(tmp_path / "maos.db"))
    db.upsert_workspace(str(tmp_path / "infernet"), "infernet", language="python", git_repo=True)
    output = tmp_path / "vision.md"

    vision = generate_vision(db, output)

    assert output.exists()
    assert "Infernet" in output.read_text()
    assert db.recent_decisions(1)[0]["kind"] == "portfolio_vision"
    assert vision.current_focus
    db.close()


def test_delegate_materializes_auditable_tasks(tmp_path) -> None:
    db = Database(str(tmp_path / "maos.db"))
    vision = build_vision([], [])

    task_ids = delegate(db, vision)

    assert len(task_ids) == 4
    tasks = db.all_tasks(limit=10)
    assert all(task["source"] == "ceo-office" for task in tasks)
    assert "architecture-agent" in tasks[-1]["title"] or "architecture-agent" in " ".join(t["title"] for t in tasks)
    db.close()


def test_render_vision_contains_continuation_contract() -> None:
    rendered = render_vision(build_vision([], []))

    assert "Operating contract" in rendered
    assert "verification" in rendered
    assert "human" in rendered.lower()
