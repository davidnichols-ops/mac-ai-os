"""macOS system telemetry using documented APIs + psutil.

Avoids private APIs. Uses psutil for cross-platform process/memory stats and
subprocess sysctl calls for Apple-specific hardware info (thermal, GPU, NE).
"""

from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass, field


@dataclass
class TelemetrySnapshot:
    macos_version: str = ""
    arch: str = ""
    cpu_percent: float = 0.0
    cpu_cores: int = 0
    memory_total_gb: float = 0.0
    memory_used_gb: float = 0.0
    memory_percent: float = 0.0
    process_count: int = 0
    thermal_state: str = "unknown"
    battery: dict = field(default_factory=dict)
    top_processes: list[dict] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [
            f"macOS {self.macos_version} ({self.arch})",
            f"CPU: {self.cpu_percent:.1f}% across {self.cpu_cores} cores",
            f"Memory: {self.memory_used_gb:.1f} / {self.memory_total_gb:.1f} GB ({self.memory_percent:.0f}%)",
            f"Processes: {self.process_count}",
            f"Thermal: {self.thermal_state}",
        ]
        if self.battery:
            lines.append(
                f"Battery: {self.battery.get('percent', '?')}% "
                f"({'plugged' if self.battery.get('plugged') else 'on battery'})"
            )
        if self.top_processes:
            lines.append("Top processes:")
            for p in self.top_processes[:5]:
                lines.append(f"  {p['name']} (pid={p['pid']}) cpu={p['cpu']:.1f}% mem={p['mem_mb']:.0f}MB")
        return "\n".join(lines)


class SystemTelemetry:
    def __init__(self, *, top_n: int = 5):
        self.top_n = top_n

    def snapshot(self) -> TelemetrySnapshot:
        try:
            import psutil  # type: ignore
        except ImportError:
            return TelemetrySnapshot(macos_version=platform.mac_ver()[0], arch=platform.machine())

        snap = TelemetrySnapshot(
            macos_version=platform.mac_ver()[0],
            arch=platform.machine(),
            cpu_cores=psutil.cpu_count(logical=True) or 0,
        )
        try:
            snap.cpu_percent = psutil.cpu_percent(interval=0.5)
        except Exception:
            pass
        vm = psutil.virtual_memory()
        snap.memory_total_gb = vm.total / (1024**3)
        snap.memory_used_gb = vm.used / (1024**3)
        snap.memory_percent = vm.percent
        snap.process_count = len(psutil.pids())

        batt = psutil.sensors_battery()
        if batt is not None:
            snap.battery = {"percent": round(batt.percent), "plugged": batt.power_plugged}

        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = p.info
                procs.append({
                    "pid": info["pid"],
                    "name": info["name"],
                    "cpu": float(info.get("cpu_percent") or 0.0),
                    "mem_mb": float(info.get("memory_percent") or 0.0) * snap.memory_total_gb * 1024 / 100,
                })
            except Exception:
                continue
        procs.sort(key=lambda x: x["cpu"], reverse=True)
        snap.top_processes = procs[: self.top_n]
        snap.thermal_state = self._thermal_state()
        return snap

    def _thermal_state(self) -> str:
        # No public API; best-effort via powermetrics (needs root) -> skip gracefully.
        try:
            out = subprocess.check_output(
                ["pmset", "-g", "therm"], text=True, timeout=2
            )
            for line in out.splitlines():
                if "CPU_Power" in line or "thermal" in line.lower():
                    return line.strip()
        except Exception:
            pass
        return "unknown"
