"""Slack command router — maps Slack messages to MAOS actions.

Understands both natural language and slash-command-style messages:

    status          → system + network + tasks summary
    network         → network health
    telemetry       → current telemetry snapshot
    tasks           → list pending tasks
    task done <id>  → complete a task
    brief           → AI-synthesized daily brief (Nemotron Ultra)
    project <ws>    → dynamically load and summarize a project's state
    trajectory      → trajectory OS / MAOS status and next milestone
    run <cmd>       → run a shell command (with approval for HIGH/CRITICAL)
    ask <question>  → RAG question
    do <goal>       → agentic loop (like `maos do`)
    help            → list available commands
    digest          → post a daily digest now

The router goes through ActionExecutor for actions, preserving permissions,
logging, and event publishing. Read-only queries call the Context API directly.
"""

from __future__ import annotations

import contextvars
import json
import shlex
from typing import Any

from app.actions.models import ActionRequest
from app.log import get_logger
from app.runtime import Runtime
from app.slack.auth import SlackAuth

log = get_logger(__name__)

# Max output length for Slack messages (Slack limits ~40k, we keep it readable).
MAX_OUTPUT = 3000


class SlackCommandRouter:
    """Routes Slack messages to MAOS actions and queries."""

    def __init__(self, runtime: Runtime, auth: SlackAuth):
        self.runtime = runtime
        self.auth = auth
        self._source_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
            "slack_source_context", default=None
        )
        self._handlers = {
            "status": self._cmd_status,
            "network": self._cmd_network,
            "telemetry": self._cmd_telemetry,
            "tasks": self._cmd_tasks,
            "task": self._cmd_task,
            "brief": self._cmd_brief,
            "ceo": self._cmd_brief,
            "project": self._cmd_project,
            "trajectory": self._cmd_trajectory,
            "umbrella": self._cmd_umbrella,
            "portfolio": self._cmd_umbrella,
            "run": self._cmd_run,
            "shell": self._cmd_run,
            "ask": self._cmd_ask,
            "do": self._cmd_do,
            "help": self._cmd_help,
            "digest": self._cmd_digest,
            "context": self._cmd_context,
            "remember": self._cmd_remember,
            "memory": self._cmd_remember,
            "note": self._cmd_remember,
            "index": self._cmd_remember,
            "vault": self._cmd_vault,
        }

    def handle(
        self,
        user_id: str,
        text: str,
        *,
        source_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Handle a Slack message. Returns a response dict.

        Response format:
            {"ok": bool, "text": str, "needs_approval": bool, "risk": str}
        """
        text = text.strip()
        self._source_context.set(dict(source_context or {}))
        if not text:
            return {"ok": False, "text": "Empty command. Send 'help' for available commands."}

        # Parse the command.
        try:
            parts = shlex.split(text)
        except ValueError:
            parts = text.split()

        cmd = parts[0].lower().lstrip("/")

        # Authorize.
        authorized, reason = self.auth.authorize(user_id, text)
        if not authorized:
            return {"ok": False, "text": reason}

        # Dispatch.
        handler = self._handlers.get(cmd)
        if not handler:
            # Try natural language: if it looks like a question, route to ask.
            if text.endswith("?") or any(w in text.lower() for w in ["what", "how", "why", "when", "where"]):
                return self._cmd_ask(user_id, [], text)
            return {"ok": False, "text": f"Unknown command: '{cmd}'. Send 'help' for available commands."}

        try:
            return handler(user_id, parts[1:] if len(parts) > 1 else [], text)
        except Exception as e:
            log.error("slack.command.error", cmd=cmd, error=str(e))
            return {"ok": False, "text": f"Error executing command: {e}"}

    def _cmd_status(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """System status: network + telemetry + tasks."""
        from app.server.context import build_context_packet

        packet = build_context_packet(
            self.runtime.db, self.runtime.blackboard,
            retriever=self.runtime.retriever,
        )
        lines = ["📊 *MAOS Status*"]

        # Network
        n = packet.get("network", {})
        if n.get("healthy"):
            lines.append(f"\n🌐 Network: ✅ healthy ({n.get('network_type', '?')})")
            if n.get("latency_ms") is not None:
                lines.append(f"   Latency: {n['latency_ms']:.0f}ms, Quality: {n.get('bandwidth_quality', 0)}/100")
        else:
            issues = n.get("issues", ["unknown"])
            lines.append(f"\n🌐 Network: ⚠️ {'/'.join(issues)}")

        # System
        s = packet.get("system", {})
        if s.get("available"):
            lines.append(f"\n💻 CPU: {s.get('cpu_percent', 0):.1f}%")
            lines.append(f"   Memory: {s.get('memory_used_gb', 0):.1f}/{s.get('memory_total_gb', 0):.1f}GB")

        # Tasks
        tasks = packet.get("tasks", [])
        lines.append(f"\n📋 Tasks: {len(tasks)} pending")
        for t in tasks[:3]:
            lines.append(f"   [{t.get('id', '?')}] {t.get('title', '?')[:50]}")

        # Workspace
        w = packet.get("workspace", {})
        if w.get("name"):
            lines.append(f"\n📁 Workspace: {w['name']}")

        return {"ok": True, "text": "\n".join(lines)}

    def _cmd_network(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Network health."""
        status = self.runtime.network.get_status()
        lines = ["🌐 *Network Health*"]
        lines.append(f"Healthy: {'✅' if status.healthy else '⚠️'}")
        lines.append(f"Interface: {status.interface} ({status.network_type.value})")
        lines.append(f"State: {status.connection_state.value}")
        lines.append(f"RX: {_fmt_bps(status.rx_rate_bps)}")
        lines.append(f"TX: {_fmt_bps(status.tx_rate_bps)}")
        lines.append(f"Latency: {_fmt_latency(status.latency_ms)}")
        lines.append(f"Quality: {status.bandwidth_quality}/100")
        if status.issues:
            lines.append(f"Issues: {', '.join(status.issues)}")
        return {"ok": True, "text": "\n".join(lines)}

    def _cmd_telemetry(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Current telemetry."""
        from app.telemetry.collector import _collect_snapshot
        snap = _collect_snapshot()
        lines = ["💻 *Telemetry*"]
        lines.append(f"CPU: {snap.get('cpu_percent', 0):.1f}% ({snap.get('cpu_cores', 0)} cores)")
        lines.append(f"Memory: {snap.get('memory_used_gb', 0):.1f}/{snap.get('memory_total_gb', 0):.1f}GB ({snap.get('memory_percent', 0):.0f}%)")
        if snap.get("battery_percent") is not None:
            plugged = "plugged" if snap.get("battery_plugged") else "battery"
            lines.append(f"Battery: {snap['battery_percent']}% ({plugged})")
        lines.append(f"Thermal: {snap.get('thermal_state', 'unknown')}")
        return {"ok": True, "text": "\n".join(lines)}

    def _cmd_tasks(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """List pending tasks."""
        tasks = self.runtime.db.pending_tasks()
        if not tasks:
            return {"ok": True, "text": "📋 No pending tasks."}
        lines = [f"📋 *{len(tasks)} Pending Tasks*"]
        for t in tasks[:10]:
            title = t["title"] if hasattr(t, "__getitem__") else t.get("title", "?")
            tid = t["id"] if hasattr(t, "__getitem__") else t.get("id", "?")
            priority = t["priority"] if hasattr(t, "__getitem__") else t.get("priority", 0)
            lines.append(f"  [{tid}] {title[:60]}")
            if priority and priority > 0:
                lines.append(f"      priority: {priority}")
        return {"ok": True, "text": "\n".join(lines)}

    def _cmd_task(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Task management: 'task done <id>'."""
        if not args:
            return {"ok": False, "text": "Usage: task done <id>"}
        sub = args[0].lower()
        if sub == "done" and len(args) >= 2:
            try:
                task_id = int(args[1])
            except ValueError:
                return {"ok": False, "text": f"Invalid task ID: {args[1]}"}
            req = ActionRequest(action="complete_todo", target=str(task_id), parameters={"task_id": task_id})
            result = self.runtime.action_executor.execute(req, context={"db": self.runtime.db})
            if result.ok:
                return {"ok": True, "text": f"✅ Task {task_id} completed."}
            return {"ok": False, "text": f"Failed: {result.error}"}
        return {"ok": False, "text": "Usage: task done <id>"}

    def _cmd_run(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Run a shell command. HIGH/CRITICAL risk requires approval."""
        if not args:
            return {"ok": False, "text": "Usage: run <command>"}
        command = " ".join(args)
        risk = SlackAuth.classify_risk(command)

        if risk in ("HIGH", "CRITICAL") and self.runtime.settings.slack.require_approval:
            return {
                "ok": False,
                "text": f"⚠️ Command classified as {risk} risk: `{command}`\n"
                        f"This requires explicit approval. Reply with `approve {command}` to proceed.",
                "needs_approval": True,
                "risk": risk,
                "command": command,
            }

        # Execute via shell.
        import subprocess
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=30,
            )
            output = result.stdout[:MAX_OUTPUT]
            if result.stderr:
                output += f"\nstderr: {result.stderr[:1000]}"
            if result.returncode != 0:
                output += f"\n(exit code: {result.returncode})"
            # Log the action.
            log.info("slack.shell.exec", user=user_id, command=command, exit_code=result.returncode)
            return {"ok": True, "text": f"```\n{output}\n```"}
        except subprocess.TimeoutExpired:
            return {"ok": False, "text": "⏱ Command timed out (30s limit)."}
        except Exception as e:
            return {"ok": False, "text": f"Error: {e}"}

    def _cmd_ask(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """RAG + model-in-the-middle answer."""
        query = " ".join(args) if args else full_text
        if not query:
            return {"ok": False, "text": "Usage: ask <question>"}

        from app.models.base import Model
        from app.slack.ask import answer_question

        # Prefer the dedicated Slack Q&A model, then the planner.
        model: Model | None = getattr(self.runtime, "qa_model", None)
        if not isinstance(model, Model):
            model = getattr(self.runtime, "planner", None)
        if not isinstance(model, Model):
            model = None

        return answer_question(
            query,
            self.runtime.db,
            retriever=getattr(self.runtime, "retriever", None),
            model=model,
        )

    def _cmd_do(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Agentic loop — like `maos do` but from Slack."""
        goal = " ".join(args) if args else ""
        if not goal:
            return {"ok": False, "text": "Usage: do <goal>"}

        # For safety, agentic loops from Slack are limited to read-only goals.
        # The agent loop is too powerful for unattended Slack control without
        # more sophisticated approval flows. This is a deliberate guardrail.
        risk = SlackAuth.classify_risk(goal)
        if risk in ("HIGH", "CRITICAL"):
            return {
                "ok": False,
                "text": f"⚠️ Goal classified as {risk} risk. Agentic loops with {risk} risk "
                        f"are not supported via Slack. Use `run <command>` with approval instead.",
            }

        # Route through the orchestrator.
        try:
            response = self.runtime.orchestrator.run(goal)
            output = str(response)[:MAX_OUTPUT]
            return {"ok": True, "text": f"🤖 {output}"}
        except Exception as e:
            return {"ok": False, "text": f"Agent error: {e}"}

    def _cmd_brief(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """AI-synthesized daily brief — Nemotron Ultra turns all tasks into clear assignments."""
        from app.models.base import Model
        from app.slack.brief import generate_brief

        model: Model | None = getattr(self.runtime, "qa_model", None)
        if not isinstance(model, Model):
            model = None

        # Parse max assignments from args (default 5).
        max_assignments = 5
        if args:
            try:
                max_assignments = int(args[0])
            except ValueError:
                pass

        result = generate_brief(
            self.runtime.db,
            model=model,
            retriever=getattr(self.runtime, "retriever", None),
            max_assignments=max_assignments,
        )

        if not result.get("ok"):
            return result

        # Prepend header with stats.
        stats = result.get("stats", {})
        total = stats.get("total", 0)
        by_unit = stats.get("by_unit", {})
        unit_summary = " | ".join(f"{u}: {c}" for u, c in sorted(by_unit.items(), key=lambda x: -x[1])[:5])

        text = f"*Daily Brief — {total} tasks*\n_{unit_summary}_\n\n{result['text']}"
        return {"ok": True, "text": text[:MAX_OUTPUT + 500]}  # Brief gets extra room

    def _cmd_project(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Dynamically load and summarize a specific project's state."""
        from app.models.base import Model
        from app.slack.brief import generate_project_brief

        if not args:
            # List available workspaces.
            try:
                rows = self.runtime.db.query(
                    "SELECT name, status FROM workspaces ORDER BY name"
                )
                if rows:
                    names = [r["name"] for r in rows]
                    return {
                        "ok": True,
                        "text": f"*Available workspaces* ({len(names)}):\n" + "\n".join(f"  • {n}" for n in names[:30]),
                    }
            except Exception:
                pass
            return {"ok": False, "text": "Usage: project <workspace-name>"}

        workspace = " ".join(args)
        model: Model | None = getattr(self.runtime, "qa_model", None)
        if not isinstance(model, Model):
            model = None

        result = generate_project_brief(
            self.runtime.db,
            workspace,
            model=model,
            retriever=getattr(self.runtime, "retriever", None),
        )
        return {"ok": result.get("ok", False), "text": result.get("text", "Error")[:MAX_OUTPUT + 500]}

    def _cmd_trajectory(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Trajectory OS / MAOS status and next milestone."""
        from app.models.base import Model
        from app.slack.brief import generate_trajectory_brief

        model: Model | None = getattr(self.runtime, "qa_model", None)
        if not isinstance(model, Model):
            model = None

        result = generate_trajectory_brief(
            self.runtime.db,
            model=model,
            retriever=getattr(self.runtime, "retriever", None),
        )
        return {"ok": result.get("ok", False), "text": result.get("text", "Error")[:MAX_OUTPUT + 500]}

    def _cmd_help(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """List available commands."""
        lines = [
            "*Available Commands*\n",
            "*Core*",
            "• `status` — system + network + tasks summary",
            "• `network` — network health and bandwidth",
            "• `telemetry` — CPU, memory, battery, thermal",
            "• `tasks` — list pending tasks",
            "• `task done <id>` — complete a task",
            "",
            "*AI Brief (Nemotron Ultra)*",
            "• `brief` — daily brief: 3-5 clear assignments from all tasks",
            "• `project <workspace>` — dynamically load and summarize a project",
            "• `trajectory` — trajectory OS / MAOS status and next milestone",
            "",
            "*System*",
            "• `run <cmd>` — run a shell command (HIGH/CRITICAL needs approval)",
            "• `ask <question>` — search local knowledge (RAG + Nemotron)",
            "• `do <goal>` — run an agentic loop (read-only goals only)",
            "• `remember <thought>` — save, index, and archive a personal memory",
            "• `index <thought>` / `note <thought>` — aliases for remember",
            "• `vault status` — LANBlockchain health and chain status",
            "• `vault reconcile [subpath]` — scoped remote drift check",
            "• `digest` — post a daily digest now",
            "• `context` — full context packet (JSON)",
            "• `umbrella` — unified computer, portfolio, repository, CLI, and MCP view",
            "• `portfolio` — alias for `umbrella`",
            "• `help` — this message",
            "\n_Natural language questions are auto-routed to `ask`._",
        ]
        return {"ok": True, "text": "\n".join(lines)}

    def _cmd_remember(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Capture user-authored text into local memory, RAG, and LAN archive."""
        content = " ".join(args).strip()
        if not content:
            return {
                "ok": False,
                "text": "Usage: remember <thought or fact>\n"
                        "Aliases: `memory`, `note`, `index`.",
            }

        source_context = self._source_context.get() or {}
        source_ref = ":".join(
            str(source_context.get(key, ""))
            for key in ("channel", "ts")
            if source_context.get(key)
        )
        action_result = self.runtime.action_executor.execute(
            ActionRequest(
                action="capture_personal_memory",
                target=content[:120],
                parameters={
                    "content": content,
                    "title": str(source_context.get("title", "")),
                    "tags": ["slack", "personal"],
                    "source": "slack",
                    "source_ref": source_ref,
                    "author_id": user_id,
                    "archive": True,
                },
            ),
            context={"db": self.runtime.db, "memory": self.runtime.memory},
        )
        if not action_result.ok:
            return {"ok": False, "text": f"Memory capture failed: {action_result.error}"}
        try:
            payload = json.loads(action_result.result)
        except json.JSONDecodeError:
            return {"ok": True, "text": f"✅ Indexed personal memory `{action_result.target}`."}
        if payload.get("duplicate"):
            return {
                "ok": True,
                "text": f"Already indexed as `{payload.get('memory_id', action_result.target)}` (duplicate content).",
            }
        archive_status = payload.get("archive_status", "disabled")
        if payload.get("archive_verified"):
            archive_text = f"LAN archive verified: `{payload.get('archive_path', '')}`"
        elif archive_status == "disabled":
            archive_text = "LAN archive disabled; local memory is durable."
        else:
            archive_text = f"LAN archive pending ({archive_status}): {payload.get('archive_detail', '')[:240]}"
        return {
            "ok": True,
            "text": f"✅ Indexed personal memory `{payload.get('memory_id', action_result.target)}`.\n{archive_text}",
            "memory_id": payload.get("memory_id", action_result.target),
        }

    def _cmd_vault(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Read-only LANBlockchain health and scoped reconciliation."""
        subcommand = args[0].lower() if args else "status"
        if subcommand == "status":
            result = self.runtime.vault.status()
            output = result.get("output") or result.get("error") or "No status returned."
            return {"ok": bool(result.get("ok")), "text": f"*LANBlockchain status*\n```\n{output[:MAX_OUTPUT]}\n```"}
        if subcommand == "reconcile":
            path = args[1] if len(args) > 1 else self.runtime.settings.lan_vault.archive_root
            result = self.runtime.vault.reconcile(path)
            output = result.get("output") or result.get("error") or "No reconcile output."
            return {"ok": bool(result.get("ok")), "text": f"*LANBlockchain reconcile `{path}`*\n```\n{output[:MAX_OUTPUT]}\n```"}
        return {"ok": False, "text": "Usage: vault status | vault reconcile [subpath]"}

    def _cmd_digest(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Post a daily digest now."""
        from app.slack.digest import build_digest
        digest = build_digest(self.runtime.db)
        lines = ["📋 *Digest*\n"]
        for section, data in digest.items():
            if isinstance(data, dict):
                lines.append(f"*{section}*:")
                for k, v in data.items():
                    lines.append(f"  {k}: {v}")
        return {"ok": True, "text": "\n".join(lines)}

    def _cmd_umbrella(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Show the unified computer, repository, CLI, MCP, and MAOS surface."""
        from pathlib import Path

        from app.umbrella import build_snapshot, render_snapshot

        home = getattr(self.runtime, "home", None)
        if home is None:
            home = Path.home()
        snapshot = build_snapshot(self.runtime.db, home=home)
        return {"ok": True, "text": render_snapshot(snapshot)}

    def _cmd_context(self, user_id: str, args: list[str], full_text: str) -> dict[str, Any]:
        """Full context packet as JSON."""
        from app.server.context import build_context_packet
        packet = build_context_packet(
            self.runtime.db, self.runtime.blackboard,
            retriever=self.runtime.retriever,
        )
        # Trim to fit Slack.
        output = json.dumps(packet, default=str, indent=2)[:MAX_OUTPUT]
        return {"ok": True, "text": f"```json\n{output}\n```"}


def _fmt_bps(bps: float) -> str:
    if bps <= 0:
        return "0 bps"
    units = ["bps", "Kbps", "Mbps", "Gbps"]
    i = 0
    v = float(bps)
    while v >= 1000 and i < len(units) - 1:
        v /= 1000
        i += 1
    return f"{v:.1f} {units[i]}"


def _fmt_latency(ms: float | None) -> str:
    if ms is None:
        return "n/a"
    return f"{ms:.0f} ms"
