"""Network intelligence layer — an invisible OS sensor for MAOS.

This package makes network awareness another foundational capability that
MAOS reasoning systems can consume, the same way macOS knows battery state.

Public API:
    NetworkProvider — the only interface the rest of MAOS should use.
        .get_status()       -> current network health snapshot
        .get_history()      -> aggregated historical bandwidth/latency
        .get_quality_score() -> 0-100 health score

Internals (not for external use):
    NetworkCollector — background thread that samples native macOS APIs
    types            — dataclasses for samples, status, aggregates

Architecture:
    MAOS Core
        |
    Context / Intelligence API
        |
    NetworkProvider  (app/network/provider.py)
        |
    Network Collector Service  (app/network/collector.py)
        |
    macOS Native APIs  (psutil.net_io_counters, getifaddrs, networksetup,
                        SystemConfiguration, TCP RTT probe)

No UI. No notifications. No packet inspection. No Network Extension.
"""

from __future__ import annotations

from app.network.collector import NetworkCollector
from app.network.provider import NetworkProvider
from app.network.types import (
    ConnectionState,
    NetworkSample,
    NetworkStatus,
    NetworkType,
)

__all__ = [
    "ConnectionState",
    "NetworkCollector",
    "NetworkProvider",
    "NetworkSample",
    "NetworkStatus",
    "NetworkType",
]
