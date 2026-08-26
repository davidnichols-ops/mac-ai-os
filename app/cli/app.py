"""CLI entrypoint (Typer).

    uv run maos verify
    uv run maos models
    uv run maos index <path>
    uv run maos watch <path>
    uv run maos ask "..."
    uv run maos rag "..."
    uv run maos chat
    uv run maos git <subcommand>
    uv run maos telemetry <subcommand>
    uv run maos timeline
    uv run maos graph <subcommand>
    uv run maos session <subcommand>
    uv run maos workspace <subcommand>
    uv run maos history <subcommand>
    uv run maos task <subcommand>
    uv run maos actions <subcommand>
    uv run maos snapshot <subcommand>
    uv run maos workflow <subcommand>
    uv run maos approval <subcommand>
    uv run maos preferences <subcommand>
"""

from __future__ import annotations

import typer

from app.cli.commands import (
    ask,
    chat,
    do,
    models,
    refresh,
    serve,
    umbrella,
    verify,
    watch,
)
from app.cli.commands import nevernote_rag as rag
from app.cli.commands.actions import actions_app
from app.cli.commands.agent import agent_app
from app.cli.commands.approval import approval_app
from app.cli.commands.blackboard import blackboard_app
from app.cli.commands.bootstrap import bootstrap
from app.cli.commands.context import context_app
from app.cli.commands.device_rag import device_rag_app
from app.cli.commands.diagnostics import diagnostics_app
from app.cli.commands.git import git_app
from app.cli.commands.graph import graph_app
from app.cli.commands.guards import guards_app
from app.cli.commands.history import history_app
from app.cli.commands.learn import learn_app
from app.cli.commands.load import load_app
from app.cli.commands.mcp import mcp_app
from app.cli.commands.memory import memory_app
from app.cli.commands.network import network_app
from app.cli.commands.office import office_app
from app.cli.commands.onboarding import onboarding_app
from app.cli.commands.perception import browse, ocr, read_pdf, search
from app.cli.commands.preferences import preferences_app
from app.cli.commands.session import session_app
from app.cli.commands.slack import slack_app
from app.cli.commands.snapshot import snapshot_app
from app.cli.commands.storage import storage_app
from app.cli.commands.task import task_app
from app.cli.commands.telemetry import telemetry_app
from app.cli.commands.time import time_app
from app.cli.commands.timeline import event_types, timeline
from app.cli.commands.workflow import workflow_app
from app.cli.commands.workspace import workspace_app

app = typer.Typer(
    name="maos",
    help="Mac AI OS — a local-first intelligence layer for macOS.",
    no_args_is_help=True,
    add_completion=False,
)

app.command()(verify.verify)
app.command()(models.models)
app.command()(rag.index)
app.command()(watch.watch)
app.command()(ask.ask)
app.command()(rag.rag)
app.command()(chat.chat)
app.command()(do.do)
app.command(name="umbrella")(umbrella.umbrella)
app.command(name="portfolio")(umbrella.portfolio)
app.command()(serve.serve)
app.command()(refresh.refresh)
app.command()(bootstrap)
app.command(name="timeline")(timeline)
app.command(name="event-types")(event_types)
app.command(name="search")(search)
app.command(name="browse")(browse)
app.command(name="read-pdf")(read_pdf)
app.command(name="ocr")(ocr)
app.command(name="rag-status")(rag.rag_status)
app.add_typer(git_app, name="git")
app.add_typer(telemetry_app, name="telemetry")
app.add_typer(graph_app, name="graph")
app.add_typer(session_app, name="session")
app.add_typer(workspace_app, name="workspace")
app.add_typer(history_app, name="history")
app.add_typer(task_app, name="task")
app.add_typer(actions_app, name="actions")
app.add_typer(snapshot_app, name="snapshot")
app.add_typer(workflow_app, name="workflow")
app.add_typer(approval_app, name="approval")
app.add_typer(agent_app, name="agent")
app.add_typer(preferences_app, name="preferences")
app.add_typer(blackboard_app, name="blackboard")
app.add_typer(learn_app, name="learn")
app.add_typer(load_app, name="load")
app.add_typer(memory_app, name="memory")
app.add_typer(mcp_app, name="mcp")
app.add_typer(context_app, name="context")
app.add_typer(onboarding_app, name="onboard")
app.add_typer(guards_app, name="guards")
app.add_typer(diagnostics_app, name="diagnostics")
app.add_typer(time_app, name="time")
app.add_typer(storage_app, name="storage")
app.add_typer(network_app, name="network")
app.add_typer(slack_app, name="slack")
app.add_typer(office_app, name="office")
app.add_typer(device_rag_app, name="device-rag")


def run() -> None:
    app()


if __name__ == "__main__":
    run()
