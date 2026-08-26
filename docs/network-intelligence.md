# Network Intelligence Provider

A foundational OS sensor layer that makes network awareness another invisible
capability MAOS reasoning systems can consume — the same way macOS knows
battery state.

This is **not** a standalone app, menu bar widget, dashboard, or user-facing
monitoring tool. It's a substrate: small, native, invisible, composable.

---

## Architecture

```
MAOS Core
    |
Context / Intelligence API  (app/server/context.py, app/context.py)
    |
NetworkProvider  (app/network/provider.py)      ← public interface
    |
Network Collector Service  (app/network/collector.py)
    |
macOS Native APIs  (psutil.net_io_counters, getifaddrs, networksetup,
                    route, ifconfig, netstat, TCP RTT probe)
```

### Layering principle

The rest of MAOS **never** touches the collector or the storage schema
directly. It goes through `NetworkProvider`, which exposes three methods:

| Method | Returns | Purpose |
| --- | --- | --- |
| `get_status()` | `NetworkStatus` | Current health snapshot ("is the network healthy?") |
| `get_history(hours, resolution)` | `dict` | Aggregated historical bandwidth/latency |
| `get_quality_score()` | `int` (0-100) | Single-number health score |

The collector is a background thread that samples native APIs every 5 seconds,
computes rates, stores raw samples, rolls them up to minute/hourly
aggregates, and publishes events. It's started explicitly (`maos network
collect` or a future daemon) — the provider is read-only against whatever's
already in the DB.

---

## Data flow

```
macOS APIs
    ↓ (every 5s)
NetworkCollector.collect_once()
    ↓
    ├── compute rates from previous sample
    ├── detect transitions (interrupt / reconnect / degradation)
    │       ↓
    │   network_events table + EventBus
    ├── insert raw sample → network_samples table
    ├── publish NETWORK_SAMPLE event on EventBus
    ├── on minute boundary → roll up to network_minute
    ├── on hour boundary  → roll up to network_hourly
    └── prune raw > 24h, minute > 30d
```

### Query path

```
AI agent / MAOS module
    ↓
NetworkProvider.get_status()
    ↓
latest row from network_samples
    ↓
compute quality score (latency + loss + state)
    ↓
NetworkStatus dataclass
```

---

## API contract

### `NetworkProvider.get_status() -> NetworkStatus`

```python
{
    "healthy": true,
    "interface": "en0",
    "network_type": "wifi",
    "connection_state": "connected",
    "rx_rate_bps": 1250000.0,
    "tx_rate_bps": 480000.0,
    "latency_ms": 18.0,
    "packet_loss": 0.0,
    "bandwidth_quality": 94,
    "issues": [],
    "timestamp": "2026-08-03T14:05:30+00:00"
}
```

### `NetworkProvider.get_quality_score() -> int`

0-100. Computed from the latest sample:

| Condition | Penalty |
| --- | --- |
| Disconnected | -100 (score = 0) |
| Latency > 500ms | -40 |
| Latency 200-500ms | -20 to -40 (linear) |
| Latency 50-200ms | -0 to -10 (linear) |
| Packet loss > 5% | -30 |
| Packet loss 1-5% | -15 to -30 (linear) |
| Packet loss 0-1% | -0 to -5 (linear) |
| RTT probe failed (connected but no latency) | -15 |

Score bands:
- **100**: perfect
- **70-99**: good (usable for cloud LLM calls)
- **40-69**: degraded (consider local models, delay large uploads)
- **0-39**: poor (cloud calls likely to fail or be very slow)

### `NetworkProvider.get_history(hours=24, resolution="hourly") -> dict`

```python
{
    "resolution": "hourly",
    "since": "2026-08-02T14:05:30+00:00",
    "totals": {"rx_bytes": 123456789, "tx_bytes": 98765432},
    "series": [{"bucket": "2026-08-02T14", "interface": "en0", ...}, ...],
    "events": [{"timestamp": "...", "event_type": "NETWORK_INTERRUPTED", ...}, ...]
}
```

### `NetworkProvider.get_daily_usage(days=7) -> list[dict]`

Per-day RX/TX totals from hourly aggregates.

### `NetworkProvider.get_hourly_pattern(days=30) -> list[dict]`

Average usage by hour-of-day (0-23). Reveals daily patterns.

---

## Storage

Integrated into the existing MAOS SQLite database (`data/maos.db`). The spec's
`~/Library/Application Support/MAOS/network/` fallback is **not** used because
MAOS already has a telemetry storage layer (`telemetry_snapshots`); creating a
parallel one would violate the "no parallel architecture" rule.

### Tables

| Table | Retention | Purpose |
| --- | --- | --- |
| `network_samples` | 24 hours | Raw per-interface samples (every 5s) |
| `network_minute` | 30 days | Minute aggregates (rates, peaks, p95 latency) |
| `network_hourly` | indefinite | Hourly aggregates (long-term patterns) |
| `network_events` | indefinite | Interruptions, degradation, interface transitions |

### Schema (designed for future expansion)

```sql
network_samples:
  id, timestamp, interface, network_type, connection_state,
  rx_bytes, tx_bytes, rx_rate_bps, tx_rate_bps,
  latency_ms, packet_loss, rtt_ok
```

Future columns (signal strength, SSID, DNS latency, jitter) can be added via
the migration system in `Database._run_migrations()` without breaking existing
data.

---

## macOS integration

Native APIs only. No Network Extension, no kernel extensions, no packet
inspection, no VPN extensions.

| Source | What it provides |
| --- | --- |
| `psutil.net_io_counters(pernic=True)` | Per-interface RX/TX byte counters |
| `route -n get default` | Active interface (primary route) |
| `networksetup -listallhardwareports` | Interface → hardware port mapping (wifi/ethernet) |
| `ifconfig <iface>` | Link state (active/inactive) |
| `netstat -i -b -n` | Interface error/collision counts (packet loss proxy) |
| TCP connect probe (`socket.create_connection`) | Latency estimate (RTT to a stable endpoint) |

### Honest about limitations

Like the existing telemetry collector (which reports `gpu_usage = None` and
`neural_engine = None` because they need root), the network collector is
honest about what it can't measure:

- **Per-flow bandwidth**: needs Network Extension / packet inspection — not attempted.
- **True end-to-end packet loss**: needs active probing at scale — the `netstat -i` error rate is a coarse proxy, not real loss.
- **DNS resolution time**: would need to spawn `dig`/`host` on every sample — not worth the CPU.
- **Signal strength / SSID**: requires CoreWLAN private API or `wdutil info` (needs root). Future extension point.

The latency probe defaults to `dns.google:443` (anycast, stable). It's a TCP
connect, not a DNS query, so it measures reachability + RTT, not DNS health.
Configurable via the `NetworkCollector` constructor.

---

## Event system

The collector publishes events on the existing `EventBus`:

| Event type | When | Severity |
| --- | --- | --- |
| `NETWORK_SAMPLE` | Every sample (5s) | info |
| `NETWORK_INTERRUPTED` | connected → disconnected | warning |
| `NETWORK_RECONNECTED` | disconnected → connected | info |
| `NETWORK_DEGRADED` | latency spike (>2x baseline, >100ms) or packet loss appears | warning |
| `NETWORK_INTERFACE_CHANGED` | active interface changes | info |

All events are also persisted to the `network_events` table (the spec's
`events.jsonl` equivalent, but in SQLite so it joins with the activity
timeline).

---

## Intelligence layer integration

Network context is injected into **every** context packet via
`app/server/context.py::_network_info()`. This means any AI client connected
via MCP or the Context API automatically receives network state without
calling a tool explicitly.

The agent-facing `Context` dataclass (`app/context.py`) also includes a
`network` field, rendered into the prompt:

```
Network: healthy
  Latency: 18ms
  Interface: en0 (wifi)
```

or, when degraded:

```
Network: degraded
  Latency: 220ms
  Interface: en0 (wifi)
  Issues: high_latency
```

### Future behaviors this enables (not yet implemented)

- **Choose local models when cloud latency is poor** — if `bandwidth_quality < 50`, route to MLX instead of an API.
- **Explain slow responses** — "Your network latency is 220ms; cloud LLM calls will be slower than usual."
- **Delay large uploads** — if `bandwidth_quality < 70`, defer non-urgent uploads.
- **Detect ISP problems** — sustained `NETWORK_DEGRADED` events over an hour suggest an ISP issue, not a transient spike.
- **Captive portal detection** — `rtt_ok=False` while `connection_state=connected` is a captive portal signal.

The provider only exposes intelligence. It does not take autonomous action.

---

## Privacy model

- **No packet inspection.** The collector reads byte counters and interface
  metadata, not packet contents. It cannot see what you're downloading, only
  how much.
- **No DNS query logging.** The latency probe is a TCP connect, not a DNS
  query.
- **No SSID / BSSID logging.** Network type is classified as wifi/ethernet/
  unknown only — the specific network name is not recorded. (Future extension
  point if explicitly opted in.)
- **Local-only storage.** All data stays in `data/maos.db` on the local
  machine. Nothing is transmitted.
- **No external reporting.** The latency probe connects to one anycast
  endpoint (`dns.google:443`) to measure RTT. No data is sent — it's a TCP
  handshake that's immediately closed.

---

## CLI

```bash
maos network status              # current health snapshot
maos network status --json       # structured output
maos network quality             # 0-100 score with a bar chart
maos network collect -s 60       # collect samples for 60s
maos network history -h 24       # last 24h, hourly resolution
maos network history -r minute   # minute resolution (30-day retention)
maos network history --json      # structured output
```

## MCP

| Tool | Purpose |
| --- | --- |
| `maos_network_status` | Current health (text or JSON) |
| `maos_network_history` | Historical bandwidth/latency |

| Resource | Purpose |
| --- | --- |
| `maos://network/current` | Current health snapshot (JSON) |

Network state is also injected into every `maos://context/current` resource
and `maos_get_context` tool response as the `network` field.

---

## Future extension points

1. **Signal strength / SSID** — via CoreWLAN (`airport -I` or `wdutil info`
   with root). Add `signal_dbm` and `ssid` columns to `network_samples`.
2. **DNS latency** — spawn `dig +stats` periodically (every minute, not every
   5s) and store as a separate column.
3. **Jitter** — compute from the variance of consecutive latency probes.
4. **Per-process network usage** — via `nettop` or `nethogs` (if available).
   Would require a new `network_process_snapshots` table mirroring
   `process_snapshots`.
5. **Autonomous actions** — the provider exposes intelligence; a future
   `NetworkAwareRouter` could consume it to switch between cloud and local
   models based on `bandwidth_quality`.
6. **Daemon mode** — run the collector as a launchd agent so it's always
   sampling, not just when `maos network collect` is running.
7. **Network Extension (deep)** — only if a future requirement demands
   per-flow bandwidth or content filtering. Would require a signed Network
   Extension entitlement and a separate process.

---

## Files

| File | Purpose |
| --- | --- |
| `app/network/__init__.py` | Package exports |
| `app/network/types.py` | `NetworkSample`, `NetworkStatus`, `NetworkType`, `ConnectionState` |
| `app/network/collector.py` | `NetworkCollector` + macOS API integration |
| `app/network/provider.py` | `NetworkProvider` — public interface |
| `app/db.py` | Network tables + helpers |
| `app/events/models.py` | Network event constants |
| `app/runtime.py` | Wires `NetworkProvider` into `Runtime` |
| `app/context.py` | Injects network into agent `Context` |
| `app/server/context.py` | Injects network into the context packet |
| `app/mcp/tools.py` | `maos_network_status`, `maos_network_history` |
| `app/mcp/server.py` | Registers MCP tools + resource |
| `app/mcp/resources.py` | `maos://network/current` resource |
| `app/cli/commands/network.py` | `maos network` CLI |
| `app/cli/app.py` | Registers the CLI subapp |
| `tests/test_network.py` | Unit + integration tests |
