"""Phase 4 planning primitives: select high-value research campaigns, do not self-authorize execution."""
from __future__ import annotations

from dataclasses import dataclass

from app.eis_contracts import Decision
from app.eis_security.finding_engine import Finding


@dataclass(frozen=True)
class ResearchCampaign:
    campaign_id: str
    target: str
    objective: str
    rationale: str
    command: list[str]
    requires_approval: bool = True


def select_campaigns(findings: list[Finding], limit: int = 5) -> list[ResearchCampaign]:
    if limit < 1:
        raise ValueError("limit must be positive")
    ranked = sorted(findings, key=lambda finding: finding.priority, reverse=True)
    campaigns: list[ResearchCampaign] = []
    for finding in ranked[:limit]:
        campaigns.append(ResearchCampaign(
            campaign_id=f"campaign:{finding.finding_id}",
            target=", ".join(finding.affected_systems),
            objective=finding.test,
            rationale=f"Priority {finding.priority}: {finding.title}",
            command=["echo", "approval-required", finding.finding_id],
        ))
    return campaigns


def campaign_decisions(campaigns: list[ResearchCampaign]) -> list[Decision]:
    return [Decision(
        decision_id=campaign.campaign_id,
        title=f"Authorize validation campaign for {campaign.target}",
        action="review_and_approve",
        rationale=campaign.rationale,
        priority=0.0,
        affected_systems=[campaign.target],
        status="approval_required",
    ) for campaign in campaigns]
