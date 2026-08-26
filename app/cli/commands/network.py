"""`maos network` — network intelligence commands.

Invisible OS sensor: query current network health, collect samples in the
background, or view historical bandwidth/latency data. No UI, no
notifications — just data for MAOS reasoning and the user.
"""

from __future__ import annotations

import json
import time

import typer

from app.runtime import build_runtime
from app.ui.console import print_panel

network_app = typer.Typer(
    name="network",
    help="Network intelligence — bandwidth, latency, health, history.",
    no_args_is_help=True,
)


@network_app.command("status")
def status(
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show current network health and bandwidth status."""
    rt = build_runtime()
    s = rt.network.get_status()

    if json_output:
        typer.echo(json.dumps(s.to_dict(), default=str, indent=2))
        return

    lines = [
        f"Healthy:     {s.healthy}",
        f"Interface:   {s.interface} ({s.network_type.value})",
        f"State:       {s.connection_state.value}",
        f"RX rate:     {_fmt_bps(s.rx_rate_bps)}",
        f"TX rate:     {_fmt_bps(s.tx_rate_bps)}",
        f"Latency:     {_fmt_latency(s.latency_ms)}",
        f"Packet loss: {_fmt_loss(s.packet_loss)}",
        f"Quality:     {s.bandwidth_quality}/100",
    ]
    if s.issues:
        lines.append(f"Issues:      {', '.join(s.issues)}")
    if s.timestamp:
        lines.append(f"Sampled:     {s.timestamp[:19]}")
    print_panel("\n".join(lines), title="Network Status")


@network_app.command("collect")
def collect(
    seconds: int = typer.Option(30, "--seconds", "-s", help="How long to collect"),
    interval: float = typer.Option(5.0, "--interval", "-i", help="Seconds between samples"),
) -> None:
    """Collect network samples for a duration, then show a summary."""
    rt = build_runtime()
    from app.network import NetworkCollector

    collector = NetworkCollector(
        rt.db, rt.bus, interval_seconds=interval,
    )
    collector.start()
    typer.echo(f"Collecting network samples for {seconds}s (every {interval}s)...")
    time.sleep(seconds)
    collector.stop()

    count = rt.db.network_sample_count()
    latest = rt.db.latest_network_sample()
    if latest:
        print_panel(
            f"Collected {count} samples total.\n"
            f"Latest: iface={latest['interface']} "
            f"rx={latest['rx_rate_bps']:.0f}bps "
            f"tx={latest['tx_rate_bps']:.0f}bps "
            f"latency={latest['latency_ms']}",
            title="Network Collection",
        )
    else:
        print_panel(f"Collected {count} samples.", title="Network Collection")


@network_app.command("history")
def history(
    hours: int = typer.Option(24, "--hours", "-h", help="Look back N hours"),
    resolution: str = typer.Option(
        "hourly", "--resolution", "-r",
        help="hourly (default) or minute",
    ),
    json_output: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Show historical network bandwidth and latency data."""
    rt = build_runtime()
    h = rt.network.get_history(hours=hours, resolution=resolution)

    if json_output:
        typer.echo(json.dumps(h, default=str, indent=2))
        return

    series = h["series"]
    totals = h["totals"]
    events = h["events"]

    lines = [
        f"Resolution: {h['resolution']}",
        f"Since:      {h['since'][:19]}",
        f"Total RX:   {_fmt_bytes(totals['rx_bytes'])}",
        f"Total TX:   {_fmt_bytes(totals['tx_bytes'])}",
        f"Samples:    {len(series)}",
        f"Events:     {len(events)}",
    ]
    if series:
        lines.append(f"\n{'Bucket':>16s}  {'Iface':>6s}  {'RX':>9s}  {'TX':>9s}  {'Lat':>6s}")
        for r in series[-15:]:
            lat = f"{r['latency_avg']:.0f}ms" if r.get("latency_avg") is not None else "-"
            lines.append(
                f"{r['bucket']:>16s}  {r['interface']:>6s}  "
                f"{_fmt_bytes(r['rx_bytes']):>9s}  {_fmt_bytes(r['tx_bytes']):>9s}  {lat:>6s}"
            )
    if events:
        lines.append("\nRecent events:")
        for e in events[:10]:
            lines.append(f"  {e['timestamp'][:19]} {e['event_type']} {e['interface']}")
    print_panel("\n".join(lines), title=f"Network History ({hours}h)")


@network_app.command("quality")
def quality() -> None:
    """Show the current network quality score (0-100)."""
    rt = build_runtime()
    score = rt.network.get_quality_score()
    s = rt.network.get_status()
    bar = "█" * (score // 5) + "░" * (20 - score // 5)
    print_panel(
        f"{bar} {score}/100\n\n"
        f"Interface: {s.interface} ({s.network_type.value})\n"
        f"Latency:   {_fmt_latency(s.latency_ms)}\n"
        f"Loss:      {_fmt_loss(s.packet_loss)}\n"
        f"Healthy:   {s.healthy}",
        title="Network Quality",
    )


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


def _fmt_bytes(n: int) -> str:
    if n <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    v = float(n)
    while v >= 1024 and i < len(units) - 1:
        v /= 1024
        i += 1
    return f"{v:.1f} {units[i]}"


def _fmt_latency(ms: float | None) -> str:
    if ms is None:
        return "n/a"
    return f"{ms:.0f} ms"


def _fmt_loss(loss: float | None) -> str:
    if loss is None:
        return "n/a"
    return f"{loss * 100:.3f}%"
