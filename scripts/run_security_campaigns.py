#!/usr/bin/env python3
"""Run bounded EIS source campaigns and regenerate security artifacts."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.eis_campaigns import CampaignResult, run_source_campaign
from app.eis_security.finding_engine import write_findings

TARGETS = {
    "trustcard": Path.home() / "Projects/trustcard",
    "maos": Path.home() / "mac-ai-os",
    "aafp": Path.home() / "Projects/aafp",
    "infernet": Path.home() / "Projects/infernet",
    "cvconform": Path.home() / "Projects/cvconform",
    "repository-infrastructure": Path.home() / "Projects/repo-archaeologist",
}


def main() -> None:
    output = Path.home() / "mac-ai-os/engineering-intelligence"
    results: list[CampaignResult] = []
    for target, path in TARGETS.items():
        if path.exists():
            results.append(run_source_campaign(target, path))
    findings = [finding for result in results for finding in result.findings]
    write_findings(findings, output / "FINDINGS_DATABASE.json")
    summary = {
        "campaigns": [
            {
                "target": result.target,
                "repository": str(result.snapshot.repository),
                "attack_surface": asdict(result.snapshot.attack_surface),
                "source_signal_count": len(result.snapshot.source_signals),
                "dependency_signal_count": len(result.snapshot.dependency_signals),
                "threat_count": len(result.snapshot.threats),
                "claims": [claim.to_dict() for claim in result.snapshot.claims],
                "challenges": [challenge.to_dict() for challenge in result.snapshot.challenges],
                "evidence": [evidence.to_dict() for evidence in result.evidence],
            }
            for result in results
        ],
        "finding_count": len(findings),
        "confirmed_vulnerability_count": sum(f.status.value == "confirmed_vulnerability" for f in findings),
        "limitations": [
            "Static source campaigns do not establish exploitability.",
            "No confirmed vulnerability is reported without a reproducible runtime test.",
            "Dependency signals are not advisory vulnerability results.",
        ],
    }
    (output / "security_campaigns.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"campaigns={len(results)} findings={len(findings)} confirmed=0")


if __name__ == "__main__":
    main()
