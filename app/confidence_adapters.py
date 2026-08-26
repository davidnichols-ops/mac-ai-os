"""Live Trustcard and CVConform adapters with explicit unknown results."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


def _run(command: list[str], cwd: Path, timeout: int = 15) -> dict[str, Any]:
    try:
        result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "unknown", "reason": f"timeout after {timeout}s", "command": command}
    return {"status": "ok" if result.returncode == 0 else "unknown", "exit_code": result.returncode,
            "command": command, "stdout_tail": result.stdout[-2000:], "stderr_tail": result.stderr[-2000:]}


def trustcard(root: Path) -> dict[str, Any]:
    trust_root = root.parent / "Projects" / "trustcard"
    command = ["node", "bin/mcp-trustcard.js", "scan", "--", "uv", "run", "app", "mcp", "status", "--json"]
    result = _run(command, trust_root)
    result.update({"system": "trustcard", "semantic_type": "tool_identity_authorization"})
    return result


def cvconform(root: Path) -> dict[str, Any]:
    cv_root = root.parent / "Projects" / "cvconform"
    candidates = list((root.parent / "Projects" / "local-operator").rglob("*.onnx"))
    if not candidates:
        return {"system": "cvconform", "semantic_type": "runtime_conformance", "status": "unknown",
                "reason": "no compatible ONNX/CoreML/PyTorch artifact; GGUF is outside CVConform's domain"}
    command = ["uv", "run", "python", "-m", "cvconform.cli", "verify", str(candidates[0]), "--json"]
    result = _run(command, cv_root, timeout=60)
    result.update({"system": "cvconform", "semantic_type": "runtime_conformance", "artifact": str(candidates[0])})
    return result


def run(root: Path, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = output if output.is_absolute() else root / output
    report = {"trustcard": trustcard(root), "cvconform": cvconform(root)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    return report
