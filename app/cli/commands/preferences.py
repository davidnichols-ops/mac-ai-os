"""`maos preferences` — user model (procedural memory) commands."""

from __future__ import annotations

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

preferences_app = typer.Typer(name="preferences", help="User model / preferences commands.", no_args_is_help=True)


@preferences_app.command("set")
def set_pref(
    key: str = typer.Argument(..., help="Preference key (e.g. 'coding_style')"),
    value: str = typer.Argument(..., help="Preference value"),
    confidence: float = typer.Option(0.8, "--confidence", "-c", help="Confidence 0.0-1.0"),
    source: str = typer.Option("manual", "--source", "-s", help="Source: manual, observed, inferred"),
) -> None:
    """Set a user model entry."""
    rt = build_runtime()
    rt.db.set_user_model_entry(key, value, confidence=confidence, source=source)
    typer.echo(f"Set {key} = {value} (confidence: {confidence}, source: {source})")


@preferences_app.command("get")
def get_pref(
    key: str = typer.Argument(..., help="Preference key"),
) -> None:
    """Get a user model entry."""
    rt = build_runtime()
    entry = rt.db.get_user_model_entry(key)
    if not entry:
        typer.echo(f"No entry for '{key}'.")
        return
    print_panel(
        f"Key:       {entry['key']}\n"
        f"Value:     {entry['value']}\n"
        f"Confidence: {entry['confidence']}\n"
        f"Source:    {entry['source']}\n"
        f"Updated:   {entry['updated_at'][:19]}",
        title=f"Preference: {key}",
    )


@preferences_app.command("list")
def list_prefs() -> None:
    """List all user model entries."""
    rt = build_runtime()
    entries = rt.db.all_user_model_entries()
    if not entries:
        typer.echo("No user model entries.")
        return

    lines = [f"{'Key':>20s}  {'Value':>30s}  {'Conf':>5s}  {'Source':>10s}"]
    for e in entries:
        lines.append(
            f"{e['key']:>20s}  {e['value'][:30]:>30s}  "
            f"{e['confidence']:5.2f}  {e['source']:>10s}"
        )
    print_panel("\n".join(lines), title=f"User Model ({len(entries)})")


@preferences_app.command("delete")
def delete_pref(
    key: str = typer.Argument(..., help="Preference key to delete"),
) -> None:
    """Delete a user model entry."""
    rt = build_runtime()
    rt.db.delete_user_model_entry(key)
    typer.echo(f"Deleted {key}")
