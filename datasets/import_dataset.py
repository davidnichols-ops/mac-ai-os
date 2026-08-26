#!/usr/bin/env python3
"""Dataset import infrastructure for MAOS — adaptive_operator_v1.

Imports JSONL files in the adaptive_operator_dataset_v1 format, validates them,
scrubs sensitive information, deduplicates, checks for train/test leakage, and
emits a human-readable import report plus cleaned/validation/statistics outputs.

Usage:
    uv run python datasets/import_dataset.py \
        --input /path/to/adaptive_operator_dataset_v1/ \
        --output datasets/adaptive_operator_v1/

Stdlib only (json, re, argparse, pathlib, collections, statistics, hashlib).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Regex patterns for secret / sensitive data scrubbing
# ---------------------------------------------------------------------------

# Secret patterns. Each entry: (label, compiled regex, replacement).
# We flag (do not silently rewrite) secrets by default — they cause rejection.
SECRET_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("OpenAI API key", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("GitHub PAT", re.compile(r"ghp_[A-Za-z0-9]{36}")),
    ("GitHub fine-grained PAT", re.compile(r"github_pat_[A-Za-z0-9_]{82}")),
    ("GitHub OAuth token", re.compile(r"gho_[A-Za-z0-9]{36}")),
    ("GitHub app token", re.compile(r"ghu_[A-Za-z0-9]{36}")),
    ("GitHub refresh token", re.compile(r"ghr_[A-Za-z0-9]{76}")),
    ("AWS access key id", re.compile(r"AKIA[0-9A-Z]{16}")),
    ("AWS secret access key", re.compile(r"(?i)aws(.{0,20})?(secret|priv).{0,5}['\"][0-9a-zA-Z/+]{40}['\"]")),
    ("Slack token", re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}")),
    ("Stripe key", re.compile(r"(?:sk|pk|rk)_(?:live|test)_[0-9a-zA-Z]{24,}")),
    ("Google API key", re.compile(r"AIza[0-9A-Za-z_\-]{35}")),
    ("Generic password assignment", re.compile(r"(?i)(password|passwd|pwd)\s*[=:]\s*['\"][^'\"]{4,}['\"]")),
    ("Generic token assignment", re.compile(r"(?i)(token|api[_-]?key|secret[_-]?key)\s*[=:]\s*['\"][^'\"]{8,}['\"]")),
    ("Bearer token", re.compile(r"(?i)bearer\s+[0-9a-zA-Z_\-\.]{20,}")),
    ("JWT", re.compile(r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |)PRIVATE KEY-----")),
]

# Patterns we *rewrite* in place (sanitization, not rejection).
# Project / temp paths -> /workspace/
PROJECT_PATH_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"/var/folders/[A-Za-z0-9_/]+"), "/workspace"),
    (re.compile(r"/private/var/folders/[A-Za-z0-9_/]+"), "/workspace"),
    (re.compile(r"/tmp/[A-Za-z0-9_./\-]+"), "/workspace"),
    (re.compile(r"/var/tmp/[A-Za-z0-9_./\-]+"), "/workspace"),
    # Absolute user home paths
    (re.compile(r"/Users/[A-Za-z0-9_\-]+/[A-Za-z0-9_./\-]+"), "/workspace"),
    (re.compile(r"/home/[A-Za-z0-9_\-]+/[A-Za-z0-9_./\-]+"), "/workspace"),
]

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
# IPv4 (avoid matching version numbers like 1.2.3 by requiring context)
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def content_hash(sample: dict[str, Any]) -> str:
    """Stable SHA-256 hash of the canonical JSON encoding of a sample."""
    payload = json.dumps(sample, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token, min 1."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def sample_token_count(sample: dict[str, Any]) -> int:
    """Sum token estimates across all message contents + tool_calls."""
    total = 0
    for msg in sample.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += estimate_tokens(str(part.get("text", "")))
        # tool_calls arguments contribute tokens too
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            total += estimate_tokens(str(fn.get("arguments", "")))
    return total


def stringify_sample(sample: dict[str, Any]) -> str:
    """Flatten a sample to a single string for scanning."""
    parts: list[str] = []
    for msg in sample.get("messages", []):
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text", "")))
        for tc in msg.get("tool_calls", []) or []:
            fn = tc.get("function", {}) if isinstance(tc, dict) else {}
            parts.append(str(fn.get("arguments", "")))
    return "\n".join(parts)


def find_secrets(text: str) -> list[str]:
    """Return labels of any secret patterns matched in text."""
    found: list[str] = []
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(text):
            found.append(label)
    return found


def scrub_paths(text: str) -> str:
    """Replace project / temp / absolute paths with /workspace/."""
    for pattern, replacement in PROJECT_PATH_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def scrub_emails(text: str) -> str:
    return EMAIL_RE.sub("[REDACTED_EMAIL]", text)


def scrub_ips(text: str) -> str:
    # Leave loopback / 0.0.0.0 alone — they're not sensitive.
    def _replace(m: re.Match[str]) -> str:
        ip = m.group(0)
        if ip in {"127.0.0.1", "0.0.0.0", "255.255.255.255"}:
            return ip
        return "[REDACTED_IP]"
    return IPV4_RE.sub(_replace, text)


def scrub_string(value: str) -> str:
    value = scrub_paths(value)
    value = scrub_emails(value)
    value = scrub_ips(value)
    return value


def scrub_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Apply path/email/IP scrubbing to every string in a sample (deep)."""
    def _scrub(obj: Any) -> Any:
        if isinstance(obj, str):
            return scrub_string(obj)
        if isinstance(obj, list):
            return [_scrub(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _scrub(v) for k, v in obj.items()}
        return obj
    return _scrub(sample)


# ---------------------------------------------------------------------------
# Core import logic
# ---------------------------------------------------------------------------

class RejectedSample:
    __slots__ = ("source", "line_no", "reason", "detail")

    def __init__(self, source: str, line_no: int, reason: str, detail: str = ""):
        self.source = source
        self.line_no = line_no
        self.reason = reason
        self.detail = detail

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "line": self.line_no,
            "reason": self.reason,
            "detail": self.detail,
        }


def validate_schema(sample: Any) -> tuple[bool, str]:
    """Check that a parsed sample has the minimal schema."""
    if not isinstance(sample, dict):
        return False, "not a JSON object"
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return False, "missing or non-list 'messages' field"
    if len(messages) == 0:
        return False, "empty 'messages' list"
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return False, f"message[{i}] is not an object"
        if "role" not in msg:
            return False, f"message[{i}] missing 'role'"
    return True, ""


def discover_jsonl_files(input_dir: Path) -> list[Path]:
    """Find all .jsonl files under input_dir (recursive, sorted)."""
    return sorted(input_dir.rglob("*.jsonl"))


def split_name(path: Path) -> str:
    """Infer a split name (train/test/val) from a filename, else 'data'."""
    name = path.stem.lower()
    if "train" in name:
        return "train"
    if "test" in name or "eval" in name:
        return "test"
    if "val" in name or "validation" in name:
        return "val"
    return "data"


def import_dataset(input_dir: Path, output_dir: Path, validation_ratio: float) -> int:
    """Run the full import pipeline. Returns process exit code."""
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"ERROR: input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    files = discover_jsonl_files(input_dir)
    if not files:
        print(f"ERROR: no .jsonl files found under {input_dir}", file=sys.stderr)
        return 2

    # Prepare output subdirectories
    processed_dir = output_dir / "processed"
    validation_dir = output_dir / "validation"
    stats_dir = output_dir / "statistics"
    for d in (processed_dir, validation_dir, stats_dir):
        d.mkdir(parents=True, exist_ok=True)

    rejected: list[RejectedSample] = []
    seen_hashes: set[str] = set()
    seen_task_ids: dict[str, str] = {}  # task_id -> split
    task_id_leakage: list[tuple[str, str, str]] = []  # (task_id, split_a, split_b)

    # Accumulators per split
    split_samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    # category distribution (task_family or metadata.type)
    category_counter: Counter[str] = Counter()
    # compute label distribution (metadata.compute_label)
    compute_labels: Counter[str] = Counter()
    token_counts: list[int] = []

    total_lines = 0
    parsed = 0
    duplicates = 0
    secret_rejected = 0
    schema_rejected = 0
    malformed = 0

    for file_path in files:
        split = split_name(file_path)
        print(f"  reading {file_path} (split={split})")
        with file_path.open("r", encoding="utf-8") as fh:
            for line_no, raw_line in enumerate(fh, start=1):
                total_lines += 1
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    sample = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    malformed += 1
                    rejected.append(RejectedSample(str(file_path), line_no, "malformed_json", str(exc)))
                    continue

                ok, why = validate_schema(sample)
                if not ok:
                    schema_rejected += 1
                    rejected.append(RejectedSample(str(file_path), line_no, "schema_invalid", why))
                    continue

                # Secret scan BEFORE scrubbing (secrets are a rejection reason).
                text = stringify_sample(sample)
                secrets = find_secrets(text)
                if secrets:
                    secret_rejected += 1
                    rejected.append(
                        RejectedSample(
                            str(file_path),
                            line_no,
                            "secret_detected",
                            ", ".join(secrets),
                        )
                    )
                    continue

                # Dedup by task_id (if present) then content hash.
                task_id = sample.get("task_id")
                if task_id is not None:
                    tid = str(task_id)
                    if tid in seen_task_ids:
                        prev_split = seen_task_ids[tid]
                        if prev_split != split:
                            task_id_leakage.append((tid, prev_split, split))
                        duplicates += 1
                        rejected.append(
                            RejectedSample(str(file_path), line_no, "duplicate_task_id", tid)
                        )
                        continue
                    seen_task_ids[tid] = split

                chash = content_hash(sample)
                if chash in seen_hashes:
                    duplicates += 1
                    rejected.append(RejectedSample(str(file_path), line_no, "duplicate_content", chash[:12]))
                    continue
                seen_hashes.add(chash)

                # Scrub sensitive/path info.
                sample = scrub_sample(sample)

                # Track category — check top-level, metadata, or extract from system message
                family = sample.get("task_family")
                if family is None:
                    # Check metadata field (control_token / production format)
                    meta = sample.get("metadata", {})
                    if isinstance(meta, dict):
                        family = meta.get("task_family") or meta.get("type")
                if family is None:
                    # Extract from system message: "completing task: refactor_006"
                    for msg in sample.get("messages", []):
                        if msg.get("role") == "system" and "task: " in msg.get("content", ""):
                            task_name = msg["content"].split("task: ")[-1].strip()
                            # task_family = prefix before last _NNN
                            parts = task_name.rsplit("_", 1)
                            family = parts[0] if len(parts) == 2 and parts[1].isdigit() else task_name
                            break
                if family is not None:
                    category_counter[str(family)] += 1

                # Track compute label distribution (production format)
                meta = sample.get("metadata", {})
                if isinstance(meta, dict) and meta.get("compute_label"):
                    compute_labels[meta["compute_label"]] += 1

                # Token stats
                tc = sample_token_count(sample)
                token_counts.append(tc)

                sample["_import_meta"] = {
                    "source": str(file_path),
                    "line": line_no,
                    "split": split,
                    "tokens": tc,
                    "content_hash": chash,
                }

                split_samples[split].append(sample)
                parsed += 1

    # ---- Write processed output (one file per split) ----
    written_files: list[Path] = []
    for split, samples in split_samples.items():
        out_path = processed_dir / f"{split}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for s in samples:
                # Strip import meta before writing? Keep it — useful for tracing.
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")
        written_files.append(out_path)

    # ---- Build validation split (stratified-ish by simple stride) ----
    validation_samples: list[dict[str, Any]] = []
    remaining: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, samples in split_samples.items():
        n_val = int(round(len(samples) * validation_ratio))
        if n_val < 1 and samples:
            n_val = 1 if len(samples) >= 5 else 0
        # Even stride sampling for a rough stratified split.
        if n_val and len(samples):
            stride = max(1, len(samples) // n_val)
            val_idx = set(range(0, len(samples), stride)) 
            # cap to n_val
            val_list = sorted(val_idx)[:n_val]
            val_set = set(val_list)
            for i, s in enumerate(samples):
                if i in val_set:
                    validation_samples.append(s)
                else:
                    remaining[split].append(s)
        else:
            remaining[split].extend(samples)

    val_path = validation_dir / "validation.jsonl"
    with val_path.open("w", encoding="utf-8") as fh:
        for s in validation_samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Rewrite processed files excluding validation samples (by content_hash).
    val_hashes = {s["_import_meta"]["content_hash"] for s in validation_samples}
    for split, samples in split_samples.items():
        out_path = processed_dir / f"{split}.jsonl"
        kept = [s for s in samples if s["_import_meta"]["content_hash"] not in val_hashes]
        with out_path.open("w", encoding="utf-8") as fh:
            for s in kept:
                fh.write(json.dumps(s, ensure_ascii=False) + "\n")

    # ---- Statistics ----
    stats: dict[str, Any] = {
        "total_lines_read": total_lines,
        "parsed_samples": parsed,
        "malformed_lines": malformed,
        "schema_rejected": schema_rejected,
        "secret_rejected": secret_rejected,
        "duplicates_removed": duplicates,
        "validation_samples": len(validation_samples),
        "task_id_leakage": [
            {"task_id": tid, "splits": [a, b]} for tid, a, b in task_id_leakage
        ],
        "per_split_counts": {s: len(v) for s, v in split_samples.items()},
        "category_distribution": dict(category_counter.most_common()),
        "compute_label_distribution": dict(compute_labels.most_common()),
        "tokens": {},
    }
    if token_counts:
        stats["tokens"] = {
            "min": min(token_counts),
            "max": max(token_counts),
            "mean": round(statistics.mean(token_counts), 2),
            "median": statistics.median(token_counts),
            "stdev": round(statistics.pstdev(token_counts), 2) if len(token_counts) > 1 else 0.0,
        }

    stats_path = stats_dir / "statistics.json"
    with stats_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    # ---- Validation errors (separate JSON file) ----
    validation_errors_path = output_dir / "validation_errors.json"
    with validation_errors_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "total_rejected": len(rejected),
                "errors_by_reason": dict(Counter(r.reason for r in rejected).most_common()),
                "errors": [r.to_dict() for r in rejected],
            },
            fh, indent=2, ensure_ascii=False,
        )

    # ---- Manifest (dataset metadata for reproducibility) ----
    manifest: dict[str, Any] = {
        "dataset_name": "adaptive_operator_v1",
        "schema_version": "1.0",
        "import_timestamp": datetime.now(UTC).isoformat(),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "input_files": [str(f) for f in files],
        "validation_ratio": validation_ratio,
        "statistics": stats,
        "outputs": {
            "processed_dir": str(processed_dir),
            "validation_dir": str(validation_dir),
            "statistics_file": str(stats_path),
            "report_file": str(stats_dir / "dataset_import_report.md"),
            "validation_errors_file": str(validation_errors_path),
        },
        "integrity": {
            "total_samples_written": sum(len(v) for v in split_samples.values()),
            "total_validation_samples": len(validation_samples),
            "rejected_count": len(rejected),
            "leakage_count": len(task_id_leakage),
        },
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # ---- Import report (markdown) ----
    report_path = stats_dir / "dataset_import_report.md"
    write_report(
        report_path,
        files=files,
        stats=stats,
        token_counts=token_counts,
        samples=[s for samples in split_samples.values() for s in samples],
        rejected=rejected,
        leakage=task_id_leakage,
        processed_dir=processed_dir,
        validation_dir=validation_dir,
    )

    # ---- Console summary ----
    print()
    print("=" * 60)
    print("Dataset import complete")
    print("=" * 60)
    print(f"  Input files:          {len(files)}")
    print(f"  Lines read:           {total_lines}")
    print(f"  Parsed samples:       {parsed}")
    print(f"  Malformed lines:      {malformed}")
    print(f"  Schema rejected:      {schema_rejected}")
    print(f"  Secret rejected:      {secret_rejected}")
    print(f"  Duplicates removed:   {duplicates}")
    print(f"  Validation samples:   {len(validation_samples)}")
    print(f"  Task-id leakage:      {len(task_id_leakage)}")
    if token_counts:
        print(f"  Tokens (min/med/max): {stats['tokens']['min']} / "
              f"{stats['tokens']['median']} / {stats['tokens']['max']}")
    print(f"  Processed output:     {processed_dir}")
    print(f"  Validation output:    {validation_dir}")
    print(f"  Statistics:           {stats_path}")
    print(f"  Manifest:             {manifest_path}")
    print(f"  Validation errors:    {validation_errors_path}")
    print(f"  Report:               {report_path}")
    print("=" * 60)
    return 0


def write_report(
    path: Path,
    files: list[Path],
    stats: dict[str, Any],
    token_counts: list[int],
    samples: list[dict[str, Any]],
    rejected: list[RejectedSample],
    leakage: list[tuple[str, str, str]],
    processed_dir: Path,
    validation_dir: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Dataset Import Report — adaptive_operator_v1")
    lines.append("")
    lines.append("Generated by `datasets/import_dataset.py`.")
    lines.append("")

    lines.append("## Input")
    lines.append("")
    lines.append(f"- Files scanned: {len(files)}")
    for f in files:
        lines.append(f"  - `{f}`")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total lines read: **{stats['total_lines_read']}**")
    lines.append(f"- Parsed samples: **{stats['parsed_samples']}**")
    lines.append(f"- Malformed lines: {stats['malformed_lines']}")
    lines.append(f"- Schema rejected: {stats['schema_rejected']}")
    lines.append(f"- Secret rejected: {stats['secret_rejected']}")
    lines.append(f"- Duplicates removed: {stats['duplicates_removed']}")
    lines.append(f"- Validation samples: {stats['validation_samples']}")
    lines.append("")

    lines.append("## Per-split counts")
    lines.append("")
    lines.append("| Split | Samples |")
    lines.append("| --- | --- |")
    for s, c in sorted(stats["per_split_counts"].items()):
        lines.append(f"| {s} | {c} |")
    lines.append("")

    cat = stats.get("category_distribution", {})
    if cat:
        lines.append("## Category distribution (task_family)")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("| --- | --- |")
        for k, v in cat.items():
            lines.append(f"| {k} | {v} |")
        lines.append("")

    tk = stats.get("tokens", {})
    if tk:
        lines.append("## Token statistics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| min | {tk['min']} |")
        lines.append(f"| max | {tk['max']} |")
        lines.append(f"| mean | {tk['mean']} |")
        lines.append(f"| median | {tk['median']} |")
        lines.append(f"| stdev | {tk['stdev']} |")
        lines.append("")

    # Longest / shortest samples
    if samples:
        sortable = sorted(samples, key=lambda s: s["_import_meta"]["tokens"], reverse=True)
        lines.append("## Longest samples (top 5 by token count)")
        lines.append("")
        lines.append("| Rank | Tokens | task_id | task_family | Source |")
        lines.append("| --- | --- | --- | --- | --- |")
        for i, s in enumerate(sortable[:5], 1):
            meta = s["_import_meta"]
            tid = s.get("task_id", "—")
            fam = s.get("task_family", "—")
            lines.append(f"| {i} | {meta['tokens']} | {tid} | {fam} | `{meta['source']}:{meta['line']}` |")
        lines.append("")

        lines.append("## Shortest samples (bottom 5 by token count)")
        lines.append("")
        lines.append("| Rank | Tokens | task_id | task_family | Source |")
        lines.append("| --- | --- | --- | --- | --- |")
        shortest = sorted(samples, key=lambda s: s["_import_meta"]["tokens"])
        for i, s in enumerate(shortest[:5], 1):
            meta = s["_import_meta"]
            tid = s.get("task_id", "—")
            fam = s.get("task_family", "—")
            lines.append(f"| {i} | {meta['tokens']} | {tid} | {fam} | `{meta['source']}:{meta['line']}` |")
        lines.append("")

    # Train/test leakage
    lines.append("## Train/test leakage (shared task_id)")
    lines.append("")
    if leakage:
        lines.append("| task_id | Split A | Split B |")
        lines.append("| --- | --- | --- |")
        for tid, a, b in leakage:
            lines.append(f"| {tid} | {a} | {b} |")
    else:
        lines.append("No task_id leakage detected.")
    lines.append("")

    # Rejected samples
    lines.append("## Rejected samples")
    lines.append("")
    if not rejected:
        lines.append("No samples were rejected.")
    else:
        # Group by reason
        by_reason: Counter[str] = Counter(r.reason for r in rejected)
        lines.append("### By reason")
        lines.append("")
        lines.append("| Reason | Count |")
        lines.append("| --- | --- |")
        for reason, count in by_reason.most_common():
            lines.append(f"| {reason} | {count} |")
        lines.append("")
        lines.append("### Details (first 200)")
        lines.append("")
        lines.append("| Source | Line | Reason | Detail |")
        lines.append("| --- | --- | --- | --- |")
        for r in rejected[:200]:
            src = r.source.split("/")[-1]
            detail = r.detail.replace("|", "\\|")
            lines.append(f"| {src} | {r.line_no} | {r.reason} | {detail} |")
        if len(rejected) > 200:
            lines.append("")
            lines.append(f"_…and {len(rejected) - 200} more (see statistics.json for full data)._")
    lines.append("")

    lines.append("## Outputs")
    lines.append("")
    lines.append(f"- Processed: `{processed_dir}`")
    lines.append(f"- Validation: `{validation_dir}`")
    lines.append(f"- Statistics: `{processed_dir.parent / 'statistics' / 'statistics.json'}`")
    lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="import_dataset.py",
        description="Import and validate JSONL datasets for MAOS adaptive_operator_v1.",
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory containing .jsonl files (adaptive_operator_dataset_v1 format).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/adaptive_operator_v1/"),
        help="Output dataset root (default: datasets/adaptive_operator_v1/).",
    )
    p.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Fraction of samples to reserve for the validation split (default: 0.1).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.validation_ratio < 1.0:
        print("ERROR: --validation-ratio must be in (0.0, 1.0)", file=sys.stderr)
        return 2
    return import_dataset(args.input, args.output, args.validation_ratio)


if __name__ == "__main__":
    sys.exit(main())
