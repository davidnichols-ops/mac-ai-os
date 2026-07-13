"""`maos approval` — approval queue commands."""

from __future__ import annotations

import json

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

approval_app = typer.Typer(name="approval", help="Approval queue commands.", no_args_is_help=True)


@approval_app.command("list")
def list_pending() -> None:
    """List pending approval requests."""
    rt = build_runtime()
    approvals = rt.db.pending_approvals()
    if not approvals:
        typer.echo("No pending approvals.")
        return

    lines = [f"{'ID':>4s}  {'Time':>19s}  {'Action':>20s}  {'Target':>15s}  {'Risk':>6s}  Reason"]
    for a in approvals:
        lines.append(
            f"{a['id']:4d}  {a['timestamp'][:19]:>19s}  {a['action_name']:>20s}  "
            f"{(a['target'] or '')[:15]:>15s}  {a['risk']:>6s}  {a['reason'][:40]}"
        )
    print_panel("\n".join(lines), title=f"Pending Approvals ({len(approvals)})")


@approval_app.command("approve")
def approve(
    approval_id: int = typer.Argument(..., help="Approval ID to approve"),
) -> None:
    """Approve a pending action."""
    rt = build_runtime()
    approval = rt.db.approval_by_id(approval_id)
    if not approval:
        typer.echo(f"Approval {approval_id} not found.")
        raise typer.Exit(1)
    if approval["status"] != "pending":
        typer.echo(f"Approval {approval_id} is already {approval['status']}.")
        raise typer.Exit(1)

    rt.db.decide_approval(approval_id, approved=True)

    # Execute the action if it's safe.
    from app.actions.models import ActionRequest
    params = json.loads(approval["parameters"]) if approval["parameters"] else {}
    request = ActionRequest(
        action=approval["action_name"],
        target=approval["target"] or "",
        parameters=params,
    )
    result = rt.action_executor.execute(request, context={"db": rt.db})
    if result.ok:
        typer.echo(f"Approved and executed: {result.result}")
    else:
        typer.echo(f"Approved but execution failed: {result.error}", err=True)


@approval_app.command("deny")
def deny(
    approval_id: int = typer.Argument(..., help="Approval ID to deny"),
) -> None:
    """Deny a pending action."""
    rt = build_runtime()
    approval = rt.db.approval_by_id(approval_id)
    if not approval:
        typer.echo(f"Approval {approval_id} not found.")
        raise typer.Exit(1)
    if approval["status"] != "pending":
        typer.echo(f"Approval {approval_id} is already {approval['status']}.")
        raise typer.Exit(1)

    rt.db.decide_approval(approval_id, approved=False)
    typer.echo(f"Denied approval #{approval_id}")


@approval_app.command("submit")
def submit(
    action: str = typer.Argument(..., help="Action name"),
    target: str = typer.Argument("", help="Action target"),
    reason: str = typer.Option("", "--reason", "-r", help="Reason for the action"),
    risk: str = typer.Option("low", "--risk", help="Risk level: low, medium, high"),
    params: str = typer.Option("{}", "--params", "-p", help="JSON parameters"),
) -> None:
    """Submit an action for approval."""
    rt = build_runtime()
    approval_id = rt.db.submit_approval(
        action, target=target, parameters=params, reason=reason, risk=risk,
    )
    typer.echo(f"Submitted approval #{approval_id} for {action} (risk: {risk})")
