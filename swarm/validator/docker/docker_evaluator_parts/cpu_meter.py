"""Container CPU accounting for the compute term.

Scoring prices what a miner's container actually burned, not how long the
validator waited for it. A loaded host inflates the wait without changing the
work; a background thread inflates the work without touching the wait. Only the
CPU counter tracks the thing we are paying for.

The counter is read on the host, from the container's own cgroup, so nothing
running inside the container can steer it. Every failure path returns ``None``,
which the caller must treat as "this seed carries no compute term" rather than
as zero cost.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

_CGROUP_ROOT = Path("/sys/fs/cgroup")


def resolve_cpu_counter(container_pid: Optional[int]) -> Optional[Path]:
    """The file holding this container's cumulative CPU time, or None.

    Both cgroup layouts are supported because either driver may be in use:
    unified v2 reports microseconds in ``cpu.stat``, legacy v1 reports
    nanoseconds in ``cpuacct.usage``. Resolving from ``/proc/<pid>/cgroup``
    rather than from the container id keeps this independent of which one the
    host runs and of how Docker names its scopes.
    """
    if container_pid is None:
        return None
    try:
        entries = Path(f"/proc/{int(container_pid)}/cgroup").read_text().splitlines()
    except (OSError, ValueError):
        return None

    v1_relative: Optional[str] = None
    for entry in entries:
        parts = entry.split(":", 2)
        if len(parts) != 3:
            continue
        controllers, relative = parts[1], parts[2]
        if controllers == "":
            candidate = _CGROUP_ROOT / relative.lstrip("/") / "cpu.stat"
            if candidate.is_file():
                return candidate
        elif "cpuacct" in controllers.split(","):
            v1_relative = relative

    if v1_relative is not None:
        candidate = _CGROUP_ROOT / "cpuacct" / v1_relative.lstrip("/") / "cpuacct.usage"
        if candidate.is_file():
            return candidate
    return None


def read_cpu_seconds(counter: Optional[Path]) -> Optional[float]:
    """Cumulative CPU seconds burned inside the container, or None if unreadable."""
    if counter is None:
        return None
    try:
        raw = counter.read_text()
    except OSError:
        return None
    if counter.name == "cpuacct.usage":
        try:
            return int(raw.strip()) / 1e9
        except ValueError:
            return None
    for line in raw.splitlines():
        if line.startswith("usage_usec"):
            try:
                return int(line.split()[1]) / 1e6
            except (IndexError, ValueError):
                return None
    return None
