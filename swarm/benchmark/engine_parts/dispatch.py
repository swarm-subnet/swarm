# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Which batch runs next and how many may run at once, decided from live host memory and a per-group RAM prior."""

from __future__ import annotations

import os
import time

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency in tests.
    psutil = None

from swarm.constants import available_vcpu_count

from ._shared import (
    BENCH_GROUP_ORDER,
    Any,
    Counter,
    Dict,
    List,
    Optional,
    Tuple,
    dataclass,
    field,
)


@dataclass(frozen=True)
class _ResourceSnapshot:
    """One reading of host CPU, load average and free memory, stamped with the monotonic clock."""

    cpu_percent: float
    load_ratio: float
    mem_available_mb: float
    mem_total_mb: float
    ts: float


# These estimates reserve memory before a worker starts. They are intentionally
# independent per group: there are no resource classes or token conversions.
_GROUP_RAM_ESTIMATES_MB = {
    "type1_city": 1400.0,
    "type2_open": 1800.0,
    "type5_warehouse": 1900.0,
    "type4_village": 2200.0,
    "type3_mountain": 2300.0,
    "type6_forest": 2400.0,
    "type7_office": 1900.0,
}
_DEFAULT_RAM_ESTIMATE_MB = max(_GROUP_RAM_ESTIMATES_MB.values())
_RESOURCE_POLL_INTERVAL_SEC = 2.0

_PARENT_WORKER_HEARTBEAT_SEC = 15.0
_PARENT_WORKER_STALL_TIMEOUT_SEC = 90.0
_TIMEOUT_RETRY_STATUSES = frozenset(
    {
        "batch_timeout",
        "batch_timeout_partial",
        "seed_cancelled",
    }
)
_RPC_TRANSPORT_RETRY_STATUSES = frozenset(
    {
        "rpc_connect_failed",
        "rpc_ping_timeout",
        "seed_rpc_disconnected",
    }
)
_INFRA_FAILURE_STATUSES = frozenset(
    {
        "batch_exception",
        "worker_stall_timeout",
    }
)


def _detect_total_ram_mb() -> float:
    """Installed host memory in MiB, read from psutil then /proc/meminfo, falling back to 8192 when neither answers."""
    try:
        if psutil is not None:
            return float(psutil.virtual_memory().total) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        pass
    return 8192.0


def _read_mem_available_mb() -> float:
    """Free host memory in MiB from psutil or /proc/meminfo, 0.0 when neither source can be reached."""
    try:
        if psutil is not None:
            return float(psutil.virtual_memory().available) / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _read_cpu_percent() -> float:
    """Host CPU utilisation since the previous psutil sample, 0.0 without psutil or on any error."""
    try:
        if psutil is not None:
            return float(psutil.cpu_percent(interval=None))
    except Exception:
        pass
    return 0.0


def _read_load_ratio(machine_vcpus: int) -> float:
    """One-minute load average divided by the vCPU count, 0.0 where getloadavg is unavailable."""
    try:
        if hasattr(os, "getloadavg"):
            return float(os.getloadavg()[0]) / float(max(1, machine_vcpus))
    except Exception:
        pass
    return 0.0


def _sample_resource_snapshot(machine_vcpus: int) -> _ResourceSnapshot:
    """Read CPU, load, free memory and installed memory once and stamp them with the monotonic clock."""
    return _ResourceSnapshot(
        cpu_percent=_read_cpu_percent(),
        load_ratio=_read_load_ratio(machine_vcpus),
        mem_available_mb=_read_mem_available_mb(),
        mem_total_mb=_detect_total_ram_mb(),
        ts=time.monotonic(),
    )


def _default_ram_reserve_mb(total_ram_mb: float) -> float:
    """Headroom kept for the host: 12% of installed memory, clamped between 2 GiB and 6 GiB."""
    return max(2048.0, min(6144.0, float(total_ram_mb) * 0.12))


def _ram_estimate_for_group(group_name: str) -> float:
    """Memory one seed of that map type is assumed to take, defaulting to the heaviest prior for an unknown name."""
    return float(
        _GROUP_RAM_ESTIMATES_MB.get(
            str(group_name),
            _DEFAULT_RAM_ESTIMATE_MB,
        )
    )


def _resource_cost_dict_for_group(group_name: str) -> Dict[str, Any]:
    """The group's memory prior wrapped as a single-key mapping, the shape the benchmark report records."""
    return {"ram_mb": _ram_estimate_for_group(group_name)}


def _resource_model_rows() -> List[Dict[str, Any]]:
    """One row per benchmark group carrying its memory prior, in the order the groups are declared."""
    return [
        {
            "group": group_name,
            "ram_mb": _ram_estimate_for_group(group_name),
        }
        for group_name in BENCH_GROUP_ORDER
    ]


def _is_clean_execution_status(status: str) -> bool:
    """True only for 'seed_done', the one outcome that means the seed flew to the end without interference."""
    return status == "seed_done"


def _is_timeout_retry_status(status: str) -> bool:
    """True for a batch that ran out of time and the seeds cancelled with it, which bucket as timeouts, not failures."""
    return status in _TIMEOUT_RETRY_STATUSES


def _is_rpc_transport_status(status: str) -> bool:
    """True when the link to the model container broke: connect refused, ping unanswered, or the socket died mid-seed."""
    return status in _RPC_TRANSPORT_RETRY_STATUSES


def _is_infra_failure_status(status: str) -> bool:
    """True for a batch that raised or a worker that went silent: the host misbehaved, not the model."""
    return status in _INFRA_FAILURE_STATUSES


def _build_worker_stall_seed_meta(
    task: Any,
    *,
    uid: int,
    elapsed_sec: float,
    error: str,
) -> Dict[str, Any]:
    """The per-seed record written when a worker goes quiet: no success, zero sim time, and the stall message."""
    return {
        "uid": int(uid),
        "map_seed": int(getattr(task, "map_seed", -1)),
        "challenge_type": int(getattr(task, "challenge_type", -1)),
        "horizon_sec": float(getattr(task, "horizon", 0.0)),
        "status": "worker_stall_timeout",
        "success": False,
        "sim_time_sec": 0.0,
        "seed_wall_sec": max(0.0, float(elapsed_sec)),
        "step_idx": 0,
        "error": error,
    }


@dataclass
class _RamWorkerScheduler:
    """Fill configured CPU-pinned workers while reserving enough host RAM."""

    requested_workers: int
    machine_vcpus: Optional[int] = None
    machine_total_ram_mb: Optional[float] = None
    resource_provider: Optional[Any] = None
    active_worker_cap: int = field(init=False)
    max_worker_cap: int = field(init=False)
    ram_reserve_mb: float = field(init=False)
    latest_snapshot: Optional[_ResourceSnapshot] = None
    group_dispatch_counts: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        """Resolve vCPUs and installed memory, then cap the slots at what the lightest seed prior fits in the budget."""
        if self.machine_vcpus is None:
            self.machine_vcpus = available_vcpu_count()
        self.machine_vcpus = max(1, int(self.machine_vcpus))

        if self.machine_total_ram_mb is None:
            self.machine_total_ram_mb = _detect_total_ram_mb()
        self.machine_total_ram_mb = max(2048.0, float(self.machine_total_ram_mb))
        self.ram_reserve_mb = _default_ram_reserve_mb(self.machine_total_ram_mb)

        usable_ram_mb = max(
            0.0,
            self.machine_total_ram_mb - self.ram_reserve_mb,
        )
        smallest_seed_ram_mb = min(_GROUP_RAM_ESTIMATES_MB.values())
        ram_limited_workers = max(
            1,
            int(usable_ram_mb // smallest_seed_ram_mb),
        )
        self.max_worker_cap = max(
            1,
            min(int(self.requested_workers), ram_limited_workers),
        )
        self.active_worker_cap = self.max_worker_cap

        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass

    @property
    def enabled(self) -> bool:
        """True once more than one worker slot survived the memory cap, so batches can overlap."""
        return self.max_worker_cap > 1

    def cost_model(self) -> List[Dict[str, Any]]:
        """Memory prior per benchmark group, the table admission is weighed against."""
        return _resource_model_rows()

    def _snapshot_from_provider(self) -> _ResourceSnapshot:
        """Take a reading from the injected provider where one was supplied, otherwise measure the real host."""
        if callable(self.resource_provider):
            raw = self.resource_provider()
            if isinstance(raw, _ResourceSnapshot):
                return raw
            raw_dict = raw if isinstance(raw, dict) else {}
            return _ResourceSnapshot(
                cpu_percent=float(
                    getattr(raw, "cpu_percent", raw_dict.get("cpu_percent", 0.0))
                ),
                load_ratio=float(
                    getattr(raw, "load_ratio", raw_dict.get("load_ratio", 0.0))
                ),
                mem_available_mb=float(
                    getattr(
                        raw,
                        "mem_available_mb",
                        raw_dict.get("mem_available_mb", 0.0),
                    )
                ),
                mem_total_mb=float(
                    getattr(
                        raw,
                        "mem_total_mb",
                        raw_dict.get("mem_total_mb", self.machine_total_ram_mb),
                    )
                ),
                ts=float(
                    getattr(raw, "ts", raw_dict.get("ts", time.monotonic()))
                ),
            )
        return _sample_resource_snapshot(self.machine_vcpus)

    def refresh_resources(self) -> None:
        """Store a fresh reading so the next admission decision weighs current free memory."""
        self.latest_snapshot = self._snapshot_from_provider()

    def _status_dict(self, snapshot: Optional[_ResourceSnapshot]) -> Dict[str, Any]:
        """Worker caps alongside the reading's CPU, load and free memory, zeros where no reading was taken."""
        return {
            "active_worker_cap": int(self.active_worker_cap),
            "max_worker_cap": int(self.max_worker_cap),
            "cpu_percent": (
                float(snapshot.cpu_percent) if snapshot is not None else 0.0
            ),
            "load_ratio": (
                float(snapshot.load_ratio) if snapshot is not None else 0.0
            ),
            "mem_available_mb": (
                float(snapshot.mem_available_mb) if snapshot is not None else 0.0
            ),
        }

    def status_dict(self) -> Dict[str, Any]:
        """Caps and machine figures from the last stored reading, without touching the host again."""
        return self._status_dict(self.latest_snapshot)

    def live_status_dict(self) -> Dict[str, Any]:
        """Caps and machine figures measured on the spot, ignoring whatever was last stored."""
        return self._status_dict(self._snapshot_from_provider())

    def _format_status_line(self, state: Dict[str, Any]) -> str:
        """Render caps, CPU, load and free memory as the single line the dispatch log prints."""
        return (
            f"cap={state['active_worker_cap']}/{state['max_worker_cap']} "
            f"cpu={state['cpu_percent']:.1f}% "
            f"load={state['load_ratio']:.2f} "
            f"mem_avail={state['mem_available_mb']:.0f}MiB"
        )

    def format_status_line(self) -> str:
        """The last stored reading rendered for the log, printed beside every dispatch."""
        return self._format_status_line(self.status_dict())

    def format_live_status_line(self) -> str:
        """A reading taken now and rendered for the log, handed to the heartbeat printer as a callback."""
        return self._format_status_line(self.live_status_dict())

    def describe_configuration_lines(self) -> List[str]:
        """Startup banner: vCPUs, installed and reserved memory, worker slots, then one entry per group prior."""
        lines = [
            (
                "Scheduler machine: "
                f"vcpus={self.machine_vcpus} "
                f"total_ram={self.machine_total_ram_mb / 1024.0:.1f}GiB "
                f"reserve_ram={self.ram_reserve_mb / 1024.0:.1f}GiB "
                f"workers={self.max_worker_cap}"
            )
        ]
        lines.extend(
            (
                "Scheduler RAM prior: "
                f"group={row['group']} ram={row['ram_mb']:.0f}MiB"
            )
            for row in self.cost_model()
        )
        return lines

    def note_group_dispatched(self, group_name: str) -> None:
        """Count one more batch sent out for that map type, which feeds the fairness term of the sort key."""
        self.group_dispatch_counts[str(group_name)] += 1

    def dispatch_sort_key(
        self,
        group_name: str,
        batch_id: int,
    ) -> Tuple[int, float, int]:
        """Order candidates by fewest batches already sent for the map type, then heaviest memory prior, then lowest id."""
        return (
            int(self.group_dispatch_counts.get(str(group_name), 0)),
            -_ram_estimate_for_group(group_name),
            int(batch_id),
        )

    def _ram_budget_mb(self) -> float:
        """Installed memory minus the host headroom, the ceiling everything in flight must stay under."""
        return max(
            0.0,
            float(self.machine_total_ram_mb) - float(self.ram_reserve_mb),
        )

    def _reserved_ram_mb(self, active_groups: List[str]) -> float:
        """Sum of the memory priors for the map types currently in flight."""
        return sum(_ram_estimate_for_group(group) for group in active_groups)

    def can_admit_group(self, group_name: str, active_groups: List[str]) -> bool:
        """True only when a slot is free, the priors of everything in flight still fit the budget, and the host has the memory spare."""
        if len(active_groups) >= self.active_worker_cap:
            return False

        seed_ram_mb = _ram_estimate_for_group(group_name)
        if (
            self._reserved_ram_mb(active_groups) + seed_ram_mb
            > self._ram_budget_mb()
        ):
            return False

        snapshot = self.latest_snapshot
        if (
            snapshot is not None
            and snapshot.mem_available_mb > 0.0
            and snapshot.mem_available_mb - self.ram_reserve_mb < seed_ram_mb
        ):
            return False
        return True


def _select_next_batch_index(
    *,
    pending_batch_ids: List[int],
    batch_plan: List[List[int]],
    task_meta: List[Dict[str, Any]],
    active_batch_ids: List[int],
    active_worker_cap: int,
    scheduler: Optional[_RamWorkerScheduler] = None,
) -> Optional[int]:
    """Pick the pending batch to dispatch: the lowest id without a scheduler, otherwise the best admissible one, or None when nothing fits."""
    if not pending_batch_ids:
        return None

    active_groups = [
        str(task_meta[batch_plan[batch_id][0]]["group"])
        for batch_id in active_batch_ids
        if batch_plan[batch_id]
    ]

    if scheduler is None:
        return min(pending_batch_ids)

    admissible = [
        batch_id
        for batch_id in pending_batch_ids
        if batch_plan[batch_id]
        and scheduler.can_admit_group(
            str(task_meta[batch_plan[batch_id][0]]["group"]),
            active_groups,
        )
    ]
    if not admissible:
        return None

    return min(
        admissible,
        key=lambda batch_id: scheduler.dispatch_sort_key(
            str(task_meta[batch_plan[batch_id][0]]["group"]),
            batch_id,
        ),
    )
