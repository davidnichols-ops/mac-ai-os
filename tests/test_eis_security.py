from pathlib import Path

from app.eis_loop import EngineeringIntelligenceLoop
from app.eis_security.attack_surface import discover_attack_surface
from app.eis_security.dependency_audit import audit_dependencies
from app.eis_security.finding_engine import Finding, FindingStatus
from app.eis_security.fuzzer import generate_mutation_cases
from app.eis_security.source_analyzer import analyze_source
from app.eis_security.threat_model import generate_threats
from app.engineering_intelligence import ValidationEvidence


def test_eis_loop_preserves_evidence_and_sorts_decisions() -> None:
    class Adapter:
        name = "fixture"

        def observe(self) -> dict[str, object]:
            return {"changed": ["src/main.py"]}

        def model(self, observation: dict[str, object]) -> dict[str, object]:
            return {"surface": observation["changed"]}

        def validate(self, model: dict[str, object]) -> list[ValidationEvidence]:
            return [ValidationEvidence("fixture", "abc", "true", "passed", 0, 0.01)]

        def challenge(self, model: dict[str, object], evidence: list[ValidationEvidence]) -> list[Finding]:
            return [Finding("F1", "fixture", FindingStatus.MISSING, 4, 4, 5, 4, 2, "h", "t", "o", "i")]

    run = EngineeringIntelligenceLoop().run(Adapter())
    assert run.adapter == "fixture"
    assert run.evidence[0].verdict == "passed"
    assert run.decisions[0].finding_id == "F1"


def test_security_inventory_and_source_signals(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("import subprocess\nsubprocess.run(user_input, shell=True)\n")
    (tmp_path / "pyproject.toml").write_text("[project]\ndependencies=['httpx']\n")
    surface = discover_attack_surface(tmp_path)
    assert "main.py" in surface.entry_points
    assert any(signal.pattern == "shell-true" for signal in analyze_source(tmp_path))
    assert audit_dependencies(tmp_path)
    assert generate_threats(surface)


def test_mutation_generator_is_bounded_and_reproducible() -> None:
    first = generate_mutation_cases(b"frame", 16, random_seed=7)
    second = generate_mutation_cases(b"frame", 16, random_seed=7)
    assert first == second
    assert len(first) == 16
