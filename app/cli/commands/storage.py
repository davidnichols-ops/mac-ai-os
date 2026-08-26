"""`maos storage` — disk storage analysis and management.

Decomposes macOS "System Data", scans user caches, lists Time Machine
snapshots, and generates reviewable remediation scripts. Bridges X-MaC
(Rust disk scanner), NeoDisk (GUI scan cache), and native macOS commands.

Usage:
    uv run maos storage scan [--mode full|fast|system|caches] [--json]
    uv run maos storage system-data [--json]
    uv run maos storage fix-script <path> [--no-caches] [--no-system] [--no-tm]
    uv run maos storage maintain
    uv run maos storage delete-snapshot <date>
    uv run maos storage volume
"""

from __future__ import annotations

import json

import typer

from app.runtime import build_runtime

storage_app = typer.Typer(
    name="storage",
    help="Disk storage analysis — decompose System Data, find reclaimable space.",
    no_args_is_help=True,
)


@storage_app.command("scan")
def scan_cmd(
    mode: str = typer.Option(
        "full",
        "--mode",
        "-m",
        help="Scan mode: full (default), fast, system, or caches.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Run a comprehensive storage analysis.

    Decomposes macOS "System Data", scans user caches, and (in full mode)
    runs X-MaC disk + clean engines for a complete picture.
    """
    from app.mcp.tools import maos_storage_scan

    rt = build_runtime()
    fmt = "json" if json_output else "text"
    result = maos_storage_scan(rt, mode=mode, format=fmt)
    typer.echo(result)


@storage_app.command("system-data")
def system_data_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Decompose macOS "System Data" into identifiable components.

    Shows exactly what makes up the mysterious "System Data" category:
    sleepimage, swapfiles, log archives, Spotlight index, Time Machine
    snapshots, APFS snapshots, temp files, and system databases.
    """
    from app.mcp.tools import maos_storage_system_data

    rt = build_runtime()
    fmt = "json" if json_output else "text"
    result = maos_storage_system_data(rt, format=fmt)
    typer.echo(result)


@storage_app.command("fix-script")
def fix_script_cmd(
    output_path: str = typer.Argument("./storage-fixes.sh", help="Output path for the fix script"),
    include_caches: bool = typer.Option(True, "--caches/--no-caches", help="Include user cache cleanup"),
    include_system: bool = typer.Option(True, "--system/--no-system", help="Include system data cleanup"),
    include_tm: bool = typer.Option(True, "--tm/--no-tm", help="Include Time Machine snapshot deletion"),
) -> None:
    """Generate a reviewable remediation shell script.

    All destructive commands are COMMENTED OUT. Review the script,
    uncomment what you want, then run: bash <output_path>
    """
    from app.mcp.tools import maos_storage_fix_script

    rt = build_runtime()
    result = maos_storage_fix_script(
        rt,
        output_path=output_path,
        include_caches=include_caches,
        include_system=include_system,
        include_tm=include_tm,
    )
    parsed = json.loads(result)
    typer.echo(f"Fix script written to: {parsed['path']}")
    typer.echo(
        f"Total reclaimable: {parsed['total_reclaimable_bytes'] / 1e9:.1f} GB"
    )
    typer.echo(f"Commands: {len(parsed['commands'])}")
    if parsed.get("warnings"):
        for w in parsed["warnings"]:
            typer.echo(f"  WARNING: {w}")
    typer.echo(f"\nReview and run: bash {parsed['path']}")


@storage_app.command("maintain")
def maintain_cmd() -> None:
    """Run safe macOS maintenance tasks via X-MaC.

    Flushes DNS, reindexes Spotlight, rebuilds LaunchServices, runs
    periodic scripts, purges RAM, and clears Quick Look cache.
    """
    from app.mcp.tools import maos_storage_maintain

    rt = build_runtime()
    result = maos_storage_maintain(rt)
    parsed = json.loads(result)
    typer.echo(f"X-MaC maintain: {parsed['findings_count']} findings")
    for f in parsed.get("findings", [])[:10]:
        title = f.get("title", "")
        typer.echo(f"  {title}")


@storage_app.command("delete-snapshot")
def delete_snapshot_cmd(
    date: str = typer.Argument(..., help="Snapshot date, e.g. 2026-07-24-152803"),
) -> None:
    """Delete a Time Machine local snapshot by date.

    This is the #1 way to reclaim "System Data" space. Use `maos storage scan`
    first to see available snapshots.
    """
    from app.mcp.tools import maos_storage_delete_tm_snapshot

    rt = build_runtime()
    result = maos_storage_delete_tm_snapshot(rt, date=date)
    parsed = json.loads(result)
    if parsed["success"]:
        typer.echo(f"✓ {parsed['message']}")
    else:
        typer.echo(f"✗ {parsed['message']}", err=True)
        raise typer.Exit(1)


@storage_app.command("volume")
def volume_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show APFS volume statistics (total, used, free, purgeable)."""
    from app.storage import get_volume_stats

    vol = get_volume_stats()
    if json_output:
        typer.echo(json.dumps(vol.__dict__, indent=2))
    else:
        typer.echo(f"Volume: {vol.container_name or 'Macintosh HD'}")
        typer.echo(f"  Total:     {vol.total_bytes / 1e9:.1f} GB")
        typer.echo(f"  Used:      {vol.used_bytes / 1e9:.1f} GB ({vol.capacity_percent:.0f}%)")
        typer.echo(f"  Free:      {vol.free_bytes / 1e9:.1f} GB")
        typer.echo(f"  Purgeable: {vol.purgeable_bytes / 1e9:.1f} GB")
        typer.echo(f"  System:    {vol.system_used_bytes / 1e9:.1f} GB")
        typer.echo(f"  Data:      {vol.data_used_bytes / 1e9:.1f} GB")
        typer.echo(f"  Apps:      {vol.applications_bytes / 1e9:.1f} GB")


@storage_app.command("hidden")
def hidden_cmd(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Scan for hidden data consumers that macOS categorizes as 'System Data'.

    Finds the hidden stuff that doesn't show up in standard disk usage:
    Devin CLI session DB, AI/ML tool data, app support dirs, cloud storage,
    SDKs, containers, and system-level data.
    """
    from app.mcp.tools import maos_storage_hidden

    rt = build_runtime()
    fmt = "json" if json_output else "text"
    result = maos_storage_hidden(rt, format=fmt)
    typer.echo(result)
