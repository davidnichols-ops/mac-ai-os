#!/usr/bin/env python3
"""Build an evidence report from indexed local Git history.

This deliberately does not label projects successful or failed. It reports
observable activity so planning can distinguish missing metadata from inactivity.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path


def build(db_path: Path) -> dict[str, object]:
    now = datetime.now(UTC)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT repo, author_date FROM git_commits WHERE repo != ''"
        ).fetchall()
    finally:
        conn.close()
    projects: dict[str, dict[str, object]] = {}
    heatmap: dict[str, int] = {}
    for row in rows:
        repo = str(row["repo"])
        name = Path(repo).name
        item = projects.setdefault(name, {"repo": repo, "commits": 0, "last_commit": None, "commits_30d": 0, "commits_90d": 0})
        item["commits"] = int(item["commits"]) + 1
        stamp = str(row["author_date"])
        if item["last_commit"] is None or stamp > item["last_commit"]:
            item["last_commit"] = stamp
        try:
            dt = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            age = now - dt.astimezone(UTC)
            if age <= timedelta(days=30):
                item["commits_30d"] = int(item["commits_30d"]) + 1
            if age <= timedelta(days=90):
                item["commits_90d"] = int(item["commits_90d"]) + 1
            day = dt.date().isoformat()
            heatmap[day] = heatmap.get(day, 0) + 1
        except ValueError:
            continue
    for item in projects.values():
        c30 = int(item["commits_30d"])
        item["activity_band"] = "hot" if c30 >= 10 else "warm" if c30 else "cold"
    return {"generated_at": now.isoformat(), "source": "maos.git_commits", "projects": sorted(projects.values(), key=lambda x: (-int(x["commits_30d"]), -int(x["commits"]), str(x["repo"]))), "commit_heatmap": dict(sorted(heatmap.items()))}


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    report = build(root / "data" / "maos.db")
    (root / "data" / "project_activity.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"projects": len(report["projects"]), "heatmap_days": len(report["commit_heatmap"])}))
