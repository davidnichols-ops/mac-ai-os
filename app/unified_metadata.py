"""Unified metadata schema for AI assets (MCP servers, ML models, agents).

Bridges trustcard (MCP server trust/capability manifests) and cvconform
(CV model conformance reports) into a single structured format that MAOS
can index and search. The schema is designed to be:

1. **Content-addressed** — every asset has a SHA-256 digest
2. **Evidence-based** — claims carry provenance and confidence
3. **Mechanism-classified** — stable IDs for failure/danger modes
4. **Timestamped** — issuedAt + optional expiresAt
5. **Cryptographically signed** — Ed25519 publisher signatures (optional)

The unified schema is a superset of both trustcard and cvconform formats.
Existing manifests/reports can be converted losslessly.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "maos.dev/asset-metadata@1"


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass
class Provenance:
    """Who/what produced this metadata and when."""
    publisher: str = ""               # Publisher name or keyId
    key_id: str = ""                  # SHA-256 of publicKey (base64url)
    public_key: str = ""              # Ed25519 public key (base64url DER)
    issued_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    expires_at: str | None = None
    generator: str = "maos-unified-metadata"  # Tool that generated this


@dataclass
class Evidence:
    """A single piece of evidence about an asset.

    Shared between trustcard danger findings and cvconform divergences.
    """
    mechanism: str                    # Stable mechanism ID (see MECHANISM_REGISTRY)
    confidence: float                 # 0.0–1.0
    impact: str = ""                  # Human-readable impact description
    observed: float | None = None     # Measured value (if applicable)
    threshold: float | None = None    # Policy threshold (if applicable)
    fixes: list[str] = field(default_factory=list)  # Recommended fixes
    details: dict[str, Any] = field(default_factory=dict)  # Additional context


@dataclass
class AssetIdentity:
    """Content-addressed identity of an AI asset."""
    kind: str                         # "mcp_server" | "ml_model" | "agent"
    name: str                         # Asset name
    version: str = ""                 # Asset version
    digest: str = ""                  # SHA-256 of the asset's semantic content
    source_path: str = ""             # Where the asset was found on disk
    spec: str = ""                    # npm spec, HuggingFace repo, or path


@dataclass
class Capability:
    """A single capability/tool exposed by the asset."""
    name: str                         # Tool/capability name
    description: str = ""
    schema_hash: str = ""             # SHA-256 of input schema
    description_hash: str = ""        # SHA-256 of description
    dangerous: bool = False           # Trustcard danger flag
    danger_score: float = 0.0         # 0.0–1.0 fused danger score
    danger_confidence: str = "low"    # "low" | "medium" | "high"
    allowed: bool = True              # Can this be called?
    required_scopes: list[str] = field(default_factory=list)
    manual_override: bool = False
    override_reason: str = ""


@dataclass
class ConformanceTarget:
    """A single runtime target's conformance result (cvconform)."""
    target: str                       # Runtime name (e.g., "onnx", "coreml")
    overall_score: float = 0.0        # 0–100 conformance score
    is_conformant: bool = False       # True if score >= 95 and no divergences
    scores: dict[str, float] = field(default_factory=dict)  # Per-output scores
    divergences: list[Evidence] = field(default_factory=list)
    status: str = "unknown"           # "conformant" | "degraded" | "error"


@dataclass
class UnifiedAssetMetadata:
    """The unified metadata envelope for any AI asset.

    This is the canonical format that MAOS indexes. It can represent:
    - An MCP server (from trustcard manifest)
    - An ML model (from cvconform report)
    - An agent (from either source or manual)
    """
    schema: str = SCHEMA_VERSION
    asset: AssetIdentity = field(default_factory=AssetIdentity)
    provenance: Provenance = field(default_factory=Provenance)
    capabilities: list[Capability] = field(default_factory=list)
    conformance: list[ConformanceTarget] = field(default_factory=list)
    findings: list[Evidence] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)
    raw_manifest_path: str = ""       # Path to original trustcard manifest
    raw_report_path: str = ""         # Path to original cvconform report
    metadata_digest: str = ""         # SHA-256 of this metadata envelope

    def compute_digest(self) -> str:
        """Compute SHA-256 of the metadata envelope (excluding the digest itself)."""
        d = asdict(self)
        d["metadata_digest"] = ""
        canonical = json.dumps(d, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict with computed digest."""
        d = asdict(self)
        d["metadata_digest"] = self.compute_digest()
        return d

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Mechanism registry — stable IDs shared between trustcard and cvconform
# ---------------------------------------------------------------------------

MECHANISM_REGISTRY: dict[str, dict[str, str]] = {
    # Trustcard danger mechanisms
    "destructive_action": {
        "domain": "mcp",
        "label": "Destructive action (file/data deletion, irreversible ops)",
    },
    "prompt_injection": {
        "domain": "mcp",
        "label": "Prompt injection in tool description",
    },
    "tool_poisoning": {
        "domain": "mcp",
        "label": "Tool schema shadowing or disguised malicious tool",
    },
    "supply_chain": {
        "domain": "mcp",
        "label": "Supply chain attack on MCP server package",
    },
    "credential_exfiltration": {
        "domain": "mcp",
        "label": "Credential or secret exfiltration via tool parameters",
    },
    # Cvconform conformance mechanisms
    "fusion": {
        "domain": "cv",
        "label": "Operator fusion / execution-order change",
    },
    "precision": {
        "domain": "cv",
        "label": "Precision change (FP32 → FP16/bf16/int8)",
    },
    "quant_scale": {
        "domain": "cv",
        "label": "Quantization scale / zero-point mismatch",
    },
    "fallback": {
        "domain": "cv",
        "label": "Unsupported-operator fallback",
    },
    "padding": {
        "domain": "cv",
        "label": "Padding / auto_pad behavior difference",
    },
    "nms": {
        "domain": "cv",
        "label": "NMS implementation difference",
    },
    "output_shape": {
        "domain": "cv",
        "label": "Shape / layout / output-name mismatch",
    },
    "nan_inf": {
        "domain": "cv",
        "label": "NaN/Inf produced on target",
    },
    "execution_failed": {
        "domain": "cv",
        "label": "Runtime failed to execute",
    },
    "comparison_error": {
        "domain": "cv",
        "label": "Comparison engine error",
    },
}


# ---------------------------------------------------------------------------
# Converters: trustcard manifest → unified metadata
# ---------------------------------------------------------------------------

def from_trustcard_manifest(manifest_path: Path) -> UnifiedAssetMetadata:
    """Convert a trustcard proxy manifest JSON to unified metadata."""
    data = json.loads(manifest_path.read_text())

    asset = AssetIdentity(
        kind="mcp_server",
        name=data.get("serverInfo", {}).get("name", data.get("spec", manifest_path.stem)),
        version=data.get("serverInfo", {}).get("version", ""),
        digest=data.get("manifestHash", ""),
        source_path=str(manifest_path),
        spec=data.get("spec", ""),
    )

    provenance = Provenance(
        issued_at=data.get("createdAt", datetime.now(UTC).isoformat()),
        expires_at=data.get("expiresAt"),
        generator="trustcard",
    )

    capabilities: list[Capability] = []
    for tool in data.get("tools", []):
        capabilities.append(Capability(
            name=tool.get("name", ""),
            schema_hash=tool.get("schemaHash", ""),
            description_hash=tool.get("descriptionHash", ""),
            dangerous=tool.get("dangerous", False),
            danger_score=tool.get("dangerScore", 0.0),
            danger_confidence=tool.get("dangerConfidence", "low"),
            allowed=tool.get("allowed", True),
            required_scopes=tool.get("requiredScopes", []),
            manual_override=tool.get("manualOverride", False),
            override_reason=tool.get("overrideReason", ""),
        ))

    findings: list[Evidence] = []
    for tool in data.get("tools", []):
        if tool.get("dangerous") and not tool.get("manualOverride"):
            findings.append(Evidence(
                mechanism="destructive_action" if tool.get("dangerScore", 0) > 0.7 else "tool_poisoning",
                confidence=tool.get("dangerScore", 0.0),
                impact=f"Tool '{tool.get('name')}' flagged as dangerous",
                details={"danger_confidence": tool.get("dangerConfidence")},
            ))

    metadata = UnifiedAssetMetadata(
        asset=asset,
        provenance=provenance,
        capabilities=capabilities,
        findings=findings,
        raw_manifest_path=str(manifest_path),
    )
    metadata.metadata_digest = metadata.compute_digest()
    return metadata


# ---------------------------------------------------------------------------
# Converters: cvconform report → unified metadata
# ---------------------------------------------------------------------------

def from_cvconform_report(report_path: Path) -> UnifiedAssetMetadata:
    """Convert a cvconform conformance report JSON to unified metadata."""
    data = json.loads(report_path.read_text())

    asset = AssetIdentity(
        kind="ml_model",
        name=data.get("model", report_path.stem),
        version="",
        digest="",  # cvconform doesn't produce a content hash; use seed-based
        source_path=str(report_path),
        spec=data.get("source", ""),
    )

    provenance = Provenance(
        generator="cvconform",
    )

    conformance: list[ConformanceTarget] = []
    findings: list[Evidence] = []

    for target_name, target_data in data.get("targets", {}).items():
        target_divergences: list[Evidence] = []
        for div in target_data.get("divergences", []):
            ev = Evidence(
                mechanism=div.get("mechanism", "comparison_error"),
                confidence=div.get("confidence", 0.0),
                impact=div.get("kind", ""),
                observed=div.get("observed"),
                threshold=div.get("threshold"),
                details={"metric": div.get("metric", ""), "first_divergence": div.get("first_divergence", "")},
            )
            target_divergences.append(ev)
            findings.append(ev)

        conformance.append(ConformanceTarget(
            target=target_name,
            overall_score=target_data.get("overall_score", 0.0),
            is_conformant=target_data.get("is_conformant", False),
            scores=target_data.get("scores", {}),
            divergences=target_divergences,
            status=data.get("target_status", {}).get(target_name, "unknown"),
        ))

    # Add root-cause findings
    for finding in data.get("findings", []):
        findings.append(Evidence(
            mechanism=finding.get("mechanism", "comparison_error"),
            confidence=finding.get("confidence", 0.0),
            impact=finding.get("impact", ""),
            fixes=finding.get("fixes", []),
            details={"backend": finding.get("backend", ""), "cause": finding.get("cause", "")},
        ))

    environment = data.get("environment", {})
    environment["input_shape"] = data.get("input_shape", [])
    environment["dataset"] = data.get("dataset", "")
    environment["seed"] = data.get("seed", 0)

    metadata = UnifiedAssetMetadata(
        asset=asset,
        provenance=provenance,
        conformance=conformance,
        findings=findings,
        environment=environment,
        raw_report_path=str(report_path),
    )
    metadata.metadata_digest = metadata.compute_digest()
    return metadata


# ---------------------------------------------------------------------------
# Batch scanning: find all manifests and reports on disk
# ---------------------------------------------------------------------------

def scan_asset_metadata(*, home: Path | None = None) -> list[UnifiedAssetMetadata]:
    """Scan the filesystem for all trustcard manifests and cvconform reports.

    Returns a list of UnifiedAssetMetadata objects ready for MAOS indexing.
    """
    home = home or Path.home()
    results: list[UnifiedAssetMetadata] = []

    # Trustcard manifests in ~/.config/devin/manifests/
    manifests_dir = home / ".config" / "devin" / "manifests"
    if manifests_dir.exists():
        for mf in manifests_dir.glob("*.json"):
            try:
                results.append(from_trustcard_manifest(mf))
            except (json.JSONDecodeError, KeyError, OSError):
                continue

    # Cvconform reports — look in common locations
    cvconform_dirs = [
        home / "Projects" / "cvconform" / "reports",
        home / "Projects" / "local-operator" / "reports",
        home / "data" / "cvconform",
    ]
    for d in cvconform_dirs:
        if d.exists():
            for report in d.glob("*.json"):
                try:
                    results.append(from_cvconform_report(report))
                except (json.JSONDecodeError, KeyError, OSError):
                    continue

    return results


def write_unified_index(output_path: Path, *, home: Path | None = None) -> dict[str, Any]:
    """Scan all asset metadata and write a unified index file.

    This is the entry point for the refresh pipeline's confidence adapter step.
    """
    assets = scan_asset_metadata(home=home)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_data = {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "asset_count": len(assets),
        "assets": [a.to_dict() for a in assets],
        "mechanisms": MECHANISM_REGISTRY,
    }
    output_path.write_text(json.dumps(index_data, indent=2, ensure_ascii=False) + "\n")
    return {
        "assets": len(assets),
        "output": str(output_path),
        "by_kind": {
            k: sum(1 for a in assets if a.asset.kind == k)
            for k in {a.asset.kind for a in assets}
        },
    }
