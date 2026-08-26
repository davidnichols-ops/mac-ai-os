from pathlib import Path

import pytest

from app.engineering_intelligence import EngineeringFinding, ValidationEvidence, rank_findings


def test_validation_evidence_round_trips(tmp_path: Path) -> None:
    evidence = ValidationEvidence(
        subject="trajectory-os",
        commit="abc123",
        command="pytest -q",
        verdict="passed",
        exit_code=0,
        duration_seconds=1.25,
    )
    target = tmp_path / "evidence.json"
    evidence.write(target)
    assert '"verdict": "passed"' in target.read_text()
    assert evidence.to_dict()["subject"] == "trajectory-os"


def test_validation_evidence_rejects_invalid_verdict() -> None:
    with pytest.raises(ValueError, match="verdict"):
        ValidationEvidence("repo", "abc", "true", "green", 0, 0)


def test_findings_rank_by_transparent_priority() -> None:
    high = EngineeringFinding("F1", "shared evidence", 5, 5, 5, 5, 1)
    low = EngineeringFinding("F2", "cosmetic", 1, 1, 1, 1, 5)
    assert rank_findings([low, high]) == [high, low]
    assert high.priority == 625.0
