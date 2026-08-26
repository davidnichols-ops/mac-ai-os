from pathlib import Path

from app.eis_adapters import analyze_repository
from app.eis_campaigns import run_source_campaign
from app.eis_contracts import ClaimStatus
from app.research_operator import campaign_decisions, select_campaigns


def test_repository_adapter_emits_claims_and_challenges(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print('fixture')\n")
    snapshot = analyze_repository(tmp_path, "fixture")
    assert snapshot.claims[0].status == ClaimStatus.UNVERIFIED
    assert snapshot.challenges[0].claim_id == snapshot.claims[0].claim_id
    assert snapshot.attack_surface.entry_points == ["main.py"]


def test_source_campaign_does_not_promote_static_signal_to_confirmed(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("import subprocess\nsubprocess.run(command, shell=True)\n")
    result = run_source_campaign("fixture", tmp_path)
    assert result.findings
    assert all(finding.status.value != "confirmed_vulnerability" for finding in result.findings)


def test_research_operator_requires_approval() -> None:
    result = run_source_campaign("fixture", Path("."))
    campaigns = select_campaigns(result.findings)
    assert campaigns
    assert all(campaign.requires_approval for campaign in campaigns)
    assert all(decision.status == "approval_required" for decision in campaign_decisions(campaigns))
