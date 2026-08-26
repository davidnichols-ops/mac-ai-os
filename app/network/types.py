"""Typed data structures for the network intelligence layer.

These dataclasses are the wire format between the collector, the provider,
and the rest of MAOS. They are intentionally plain data — no behavior, no
side effects. The provider converts them to dicts for the context packet
and MCP responses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class NetworkType(StrEnum):
    """Coarse interface classification. 'unknown' is honest — we don't guess."""
    WIFI = "wifi"
    ETHERNET = "ethernet"
    UNKNOWN = "unknown"


class ConnectionState(StrEnum):
    """Coarse link state. 'unknown' when we can't determine it."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNKNOWN = "unknown"


@dataclass
class NetworkSample:
    """One raw sample from the collector (every 5s by default).

    Rates (rx_rate_bps / tx_rate_bps) are computed by the collector from
    the delta of cumulative byte counters between consecutive samples.
    Bytes are cumulative counters (monotonic, wrap at 2^64 on macOS).
    """
    timestamp: str
    interface: str
    network_type: NetworkType = NetworkType.UNKNOWN
    connection_state: ConnectionState = ConnectionState.UNKNOWN
    rx_bytes: int = 0
    tx_bytes: int = 0
    rx_rate_bps: float = 0.0
    tx_rate_bps: float = 0.0
    latency_ms: float | None = None
    packet_loss: float | None = None
    rtt_ok: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "interface": self.interface,
            "network_type": self.network_type.value,
            "connection_state": self.connection_state.value,
            "rx_bytes": self.rx_bytes,
            "tx_bytes": self.tx_bytes,
            "rx_rate_bps": round(self.rx_rate_bps, 1),
            "tx_rate_bps": round(self.tx_rate_bps, 1),
            "latency_ms": self.latency_ms,
            "packet_loss": self.packet_loss,
            "rtt_ok": self.rtt_ok,
        }


@dataclass
class NetworkStatus:
    """The public 'is the network healthy?' answer.

    This is what `NetworkProvider.get_status()` returns and what flows into
    the context packet. Future MAOS modules ask this question without
    knowing anything about interfaces, RTT probes, or byte counters.
    """
    healthy: bool
    interface: str
    network_type: NetworkType
    connection_state: ConnectionState
    rx_rate_bps: float = 0.0
    tx_rate_bps: float = 0.0
    latency_ms: float | None = None
    packet_loss: float | None = None
    bandwidth_quality: int = 0  # 0-100
    issues: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "healthy": self.healthy,
            "interface": self.interface,
            "network_type": self.network_type.value,
            "connection_state": self.connection_state.value,
            "rx_rate_bps": round(self.rx_rate_bps, 1),
            "tx_rate_bps": round(self.tx_rate_bps, 1),
            "latency_ms": self.latency_ms,
            "packet_loss": self.packet_loss,
            "bandwidth_quality": self.bandwidth_quality,
            "issues": self.issues,
            "timestamp": self.timestamp,
        }


def utc_now_iso() -> str:
    """ISO 8601 UTC timestamp. Centralized so all samples share a format."""
    return datetime.now(UTC).isoformat()
