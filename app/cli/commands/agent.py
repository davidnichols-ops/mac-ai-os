"""Agent identity and handoff commands."""

from __future__ import annotations

import json

import typer

from app.agents.identity import end_session, recent, start_session
from app.agents.workflow import PROVIDERS, build_agent_brief, doctor_report
from app.config.settings import load_settings

agent_app = typer.Typer(help="Register agent identities and publish session handoffs.")


def _provider_help() -> str:
    return "Provider: " + ", ".join(PROVIDERS)


@agent_app.command("doctor")
def doctor(
    provider: str = typer.Option("all", help=_provider_help()),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Verify agent tooling, workflow files, and secret readiness safely."""
    try:
        report = doctor_report(provider)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if json_output:
        typer.echo(json.dumps(report, indent=2))
    else:
        state = "healthy" if report["healthy"] else "not ready"
        typer.echo(f"Agent workflow: {state}")
        for name, check in report["providers"].items():
            binary = check["binary"]
            path = binary.get("path") or "not required"
            typer.echo(f"  {name}: {path}")
        for name, check in report["credentials"].items():
            typer.echo(f"  {name}: {'set' if check['configured'] else 'missing'}")
    if not report["healthy"]:
        raise typer.Exit(1)


@agent_app.command("brief")
def brief(
    provider: str = typer.Option("generic", help=_provider_help()),
    task: str = typer.Option("", help="Task to include in the handoff"),
) -> None:
    """Print the authoritative startup and handoff instructions."""
    try:
        typer.echo(build_agent_brief(provider, task))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@agent_app.command("start")
def start(provider: str = typer.Option("unknown"), role: str = typer.Option("operator"), task: str = typer.Option("")) -> None:
    settings = load_settings(detect_hardware=False)
    record = start_session(settings.resolved_data_dir(), provider=provider, role=role, task=task)
    typer.echo(json.dumps(record, indent=2))


@agent_app.command("end")
def end(
    agent_id: str,
    summary: str = typer.Option(""),
    result: str = typer.Option(""),
    status: str = typer.Option("completed"),
    workspace: str = typer.Option("", help="Project/workspace used for Slack routing"),
) -> None:
    settings = load_settings(detect_hardware=False)
    record = end_session(settings.resolved_data_dir(), agent_id, status=status, summary=summary, result=result)
    if not record:
        raise typer.BadParameter(f"No active agent session found for {agent_id}")
    if settings.slack.enabled and settings.slack.bot_token:
        from app.slack.notifier import SlackNotifier

        SlackNotifier(
            settings.slack.bot_token,
            settings.slack.channel,
            team_id=settings.slack.team_id,
            alerts_channel=settings.slack.alerts_channel,
            daily_channel=settings.slack.daily_channel,
            audit_channel=settings.slack.audit_channel,
        ).post_alert(
            "AGENT_SESSION_COMPLETED",
            {
                "agent_id": agent_id,
                "provider": record.get("provider", "unknown"),
                "role": record.get("role", "operator"),
                "workspace": workspace,
                "summary": summary or "(no summary)",
                "result": result or status,
            },
        )
    typer.echo(json.dumps(record, indent=2))


@agent_app.command("list")
def list_agents(limit: int = typer.Option(20)) -> None:
    settings = load_settings(detect_hardware=False)
    typer.echo(json.dumps(recent(settings.resolved_data_dir(), limit), indent=2))
