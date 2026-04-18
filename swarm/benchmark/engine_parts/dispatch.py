from __future__ import annotations

import os

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency in tests.
    psutil = None

from swarm.constants import available_vcpu_count

from ._shared import (
    Any,
    Counter,
    Dict,
    List,
    Optional,
    Tuple,
    BENCH_GROUP_TO_TYPE,
    dataclass,
    deque,
    field,
)


@dataclass(frozen=True)
class _SeedResourceCost:
    resource_class: str
    cpu_units: float
    ram_mb: float
    heavy_tokens: int


@dataclass(frozen=True)
class _ResourceSnapshot:
    cpu_percent: float
    load_ratio: float
    mem_available_mb: float
    mem_total_mb: float
    ts: float


_CLASS_RANK = {
    "light": 0,
    "medium": 1,
    "heavy": 2,
}
_RESOURCE_COSTS = {
    "light": _SeedResourceCost(
        resource_class="light",
        cpu_units=0.90,
        ram_mb=2200.0,
        heavy_tokens=0,
    ),
    "medium": _SeedResourceCost(
        resource_class="medium",
        cpu_units=1.15,
        ram_mb=2850.0,
        heavy_tokens=1,
    ),
    "heavy": _SeedResourceCost(
        resource_class="heavy",
        cpu_units=1.35,
        ram_mb=3600.0,
        heavy_tokens=2,
    ),
}
_GROUP_BASE_RESOURCE_COSTS = {
    "type1_city": _SeedResourceCost("light", 0.90, 2200.0, 0),
    "type2_open": _SeedResourceCost("light", 0.95, 2300.0, 0),
    "type5_warehouse": _SeedResourceCost("medium", 1.05, 2650.0, 1),
    "type4_village": _SeedResourceCost("medium", 1.10, 2900.0, 1),
    "type3_mountain": _SeedResourceCost("heavy", 1.25, 3300.0, 2),
    "type6_forest": _SeedResourceCost("heavy", 1.35, 3600.0, 2),
}
_GROUP_RESOURCE_CLASS = {
    group_name: cost.resource_class
    for group_name, cost in _GROUP_BASE_RESOURCE_COSTS.items()
}
_RESOURCE_CLASS_GROUPS = {
    "light": ("type1_city", "type2_open"),
    "medium": ("type5_warehouse", "type4_village"),
    "heavy": ("type3_mountain", "type6_forest"),
}
_GROUP_CONCURRENCY_LIMITS = {
    "type3_mountain": 1,
}
_HEAVY_GROUPS = frozenset(
    group_name
    for group_name, cost in _GROUP_BASE_RESOURCE_COSTS.items()
    if cost.resource_class == "heavy"
)
_GROUP_FROM_CHALLENGE_TYPE = {
    int(challenge_type): group_name
    for group_name, challenge_type in BENCH_GROUP_TO_TYPE.items()
}
_BACKOFF_ACTIVE_WORKERS = 2
_BACKOFF_COOLDOWN_COMPLETIONS = 4
_COLD_START_SMALL_WORKERS = 2
_COLD_START_LARGE_WORKERS = 3
_COLD_START_LARGE_THRESHOLD = 6
_PRESSURE_HIGH_CPU_PERCENT = 88.0
_PRESSURE_CRITICAL_CPU_PERCENT = 94.0
_PRESSURE_HEALTHY_CPU_PERCENT = 72.0
_PRESSURE_HIGH_LOAD_RATIO = 0.95
_PRESSURE_CRITICAL_LOAD_RATIO = 1.10
_PRESSURE_HEALTHY_LOAD_RATIO = 0.75
_RESOURCE_WINDOW = 6
_PRESSURE_HIGH_STREAK = 3
_PRESSURE_CRITICAL_STREAK = 2
_PARENT_WORKER_HEARTBEAT_SEC = 15.0
_PARENT_WORKER_STALL_TIMEOUT_SEC = 90.0
_BACKOFF_TIMEOUT_STATUSES = frozenset(
    {
        "batch_timeout",
        "batch_timeout_partial",
        "seed_cancelled",
    }
)
_BACKOFF_INFRA_FAILURE_STATUSES = frozenset(
    {
        "batch_exception",
        "worker_stall_timeout",
    }
)
_RESOURCE_SOFT_SIGNAL_STATUSES = frozenset(
    {
        "seed_timeout_strikes",
        "seed_rpc_disconnected",
        "seed_exception",
    }
)
_NON_RESOURCE_FAILURE_STATUSES = frozenset(
    {
        "container_start_failed",
        "pip_install_failed",
        "network_lockdown_failed",
        "submission_start_failed",
        "rpc_connection_failed",
    }
)
_PROFILE_WINDOW = 12
_PROFILE_MIN_TRUSTED_SAMPLES_FOR_DEMOTION = 4
_RELAX_FAST_STREAK = 2
_RELAX_NORMAL_STREAK = 3


@dataclass
class _GroupRuntimeProfile:
    samples: int = 0
    trusted_samples: int = 0
    timeout_like_count: int = 0
    calibration_spike_count: int = 0
    wall_sec_ema: float = 0.0
    ratio_ema: float = 0.0
    recent_wall_sec: deque[float] = field(
        default_factory=lambda: deque(maxlen=_PROFILE_WINDOW)
    )
    recent_ratio: deque[float] = field(
        default_factory=lambda: deque(maxlen=_PROFILE_WINDOW)
    )


def _detect_total_ram_mb() -> float:
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
    try:
        if psutil is not None:
            return float(psutil.cpu_percent(interval=None))
    except Exception:
        pass
    return 0.0


def _read_load_ratio(machine_vcpus: int) -> float:
    try:
        if hasattr(os, "getloadavg"):
            load1 = float(os.getloadavg()[0])
            return load1 / float(max(1, machine_vcpus))
    except Exception:
        pass
    return 0.0


def _sample_resource_snapshot(machine_vcpus: int) -> _ResourceSnapshot:
    return _ResourceSnapshot(
        cpu_percent=_read_cpu_percent(),
        load_ratio=_read_load_ratio(machine_vcpus),
        mem_available_mb=_read_mem_available_mb(),
        mem_total_mb=_detect_total_ram_mb(),
        ts=float(os.times().elapsed),
    )


def _default_ram_reserve_mb(total_ram_mb: float) -> float:
    return max(2048.0, min(6144.0, float(total_ram_mb) * 0.12))


def _group_name_from_challenge_type(challenge_type: Any) -> Optional[str]:
    try:
        return _GROUP_FROM_CHALLENGE_TYPE.get(int(challenge_type))
    except Exception:
        return None


def _resource_class_rank(resource_class: str) -> int:
    return int(_CLASS_RANK.get(str(resource_class), 1))


def _resource_class_from_rank(rank: int) -> str:
    normalized = max(0, min(2, int(rank)))
    for resource_class, resource_rank in _CLASS_RANK.items():
        if resource_rank == normalized:
            return resource_class
    return "medium"


def _resource_class_from_heavy_tokens(heavy_tokens: int) -> str:
    if int(heavy_tokens) >= 2:
        return "heavy"
    if int(heavy_tokens) >= 1:
        return "medium"
    return "light"


def _base_resource_cost_for_group(group_name: str) -> _SeedResourceCost:
    return _GROUP_BASE_RESOURCE_COSTS.get(
        str(group_name),
        _RESOURCE_COSTS["medium"],
    )


def _resource_class_for_group(group_name: str) -> str:
    return str(_base_resource_cost_for_group(group_name).resource_class)


def _resource_cost_for_group(group_name: str) -> _SeedResourceCost:
    return _base_resource_cost_for_group(group_name)


def _resource_cost_dict_for_group(group_name: str) -> Dict[str, Any]:
    cost = _resource_cost_for_group(group_name)
    return {
        "resource_class": cost.resource_class,
        "cpu_units": float(cost.cpu_units),
        "ram_mb": float(cost.ram_mb),
        "heavy_tokens": int(cost.heavy_tokens),
    }


def _resource_model_rows() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for resource_class in ("light", "medium", "heavy"):
        group_names = list(_RESOURCE_CLASS_GROUPS[resource_class])
        if group_names:
            cpu_units = sum(
                float(_base_resource_cost_for_group(group_name).cpu_units)
                for group_name in group_names
            ) / float(len(group_names))
            ram_mb = sum(
                float(_base_resource_cost_for_group(group_name).ram_mb)
                for group_name in group_names
            ) / float(len(group_names))
            heavy_tokens = max(
                int(_base_resource_cost_for_group(group_name).heavy_tokens)
                for group_name in group_names
            )
        else:
            fallback = _RESOURCE_COSTS[resource_class]
            cpu_units = float(fallback.cpu_units)
            ram_mb = float(fallback.ram_mb)
            heavy_tokens = int(fallback.heavy_tokens)
        rows.append(
            {
                "resource_class": resource_class,
                "groups": group_names,
                "cpu_units": float(cpu_units),
                "ram_mb": float(ram_mb),
                "heavy_tokens": int(heavy_tokens),
            }
        )
    return rows


def _is_clean_execution_status(status: str) -> bool:
    return status == "seed_done"


def _max_heavy_active(active_worker_cap: int) -> int:
    worker_cap = max(1, int(active_worker_cap))
    return max(1, min(4, (worker_cap + 1) // 3))


def _is_backoff_timeout_status(status: str) -> bool:
    return status in _BACKOFF_TIMEOUT_STATUSES


def _is_backoff_infra_status(status: str) -> bool:
    return status in _BACKOFF_INFRA_FAILURE_STATUSES


def _worker_cap_levels(requested_workers: int) -> Tuple[int, ...]:
    floor = 1 if int(requested_workers) <= 1 else _BACKOFF_ACTIVE_WORKERS
    current = max(1, int(requested_workers))
    levels = [current]
    while current > floor:
        if current >= 8:
            current -= 2
        else:
            current -= 1
        current = max(floor, current)
        if current != levels[-1]:
            levels.append(current)
    return tuple(levels)


def _initial_worker_cap(max_worker_cap: int) -> int:
    worker_cap = max(1, int(max_worker_cap))
    if worker_cap <= 1:
        return 1
    if worker_cap >= _COLD_START_LARGE_THRESHOLD:
        return min(worker_cap, _COLD_START_LARGE_WORKERS)
    return min(worker_cap, _COLD_START_SMALL_WORKERS)


def _seed_runtime_ratio(seed_meta: Optional[Dict[str, Any]]) -> float:
    if not isinstance(seed_meta, dict):
        return 0.0
    try:
        wall_sec = max(0.0, float(seed_meta.get("seed_wall_sec", 0.0)))
    except Exception:
        wall_sec = 0.0
    try:
        sim_time_sec = max(0.0, float(seed_meta.get("sim_time_sec", 0.0)))
    except Exception:
        sim_time_sec = 0.0
    try:
        horizon_sec = max(0.0, float(seed_meta.get("horizon_sec", 0.0)))
    except Exception:
        horizon_sec = 0.0
    reference_sim = sim_time_sec if sim_time_sec > 0.0 else horizon_sec
    reference_sim = max(10.0, reference_sim)
    if wall_sec <= 0.0:
        return 0.0
    return wall_sec / reference_sim


def _seed_has_calibration_spike(seed_meta: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(seed_meta, dict):
        return False
    try:
        overhead = seed_meta.get("calibration_overhead_sec")
        cpu_factor = seed_meta.get("calibration_cpu_factor")
        overhead_val = float(overhead) if overhead is not None else 0.0
        cpu_factor_val = float(cpu_factor) if cpu_factor is not None else 1.0
    except Exception:
        return False
    return overhead_val >= 0.25 or cpu_factor_val >= 1.5


def _format_seed_desc(seed_meta: Optional[Dict[str, Any]]) -> str:
    if not isinstance(seed_meta, dict):
        return "seed=unknown"
    parts = []
    if "map_seed" in seed_meta:
        parts.append(f"seed={seed_meta.get('map_seed')}")
    if "challenge_type" in seed_meta:
        parts.append(f"type={seed_meta.get('challenge_type')}")
    return " ".join(parts) if parts else "seed=unknown"


def _build_worker_stall_seed_meta(
    task: Any,
    *,
    uid: int,
    elapsed_sec: float,
    error: str,
) -> Dict[str, Any]:
    return {
        "uid": int(uid),
        "map_seed": int(getattr(task, "map_seed", -1)),
        "challenge_type": int(getattr(task, "challenge_type", -1)),
        "horizon_sec": float(getattr(task, "horizon", 0.0)),
        "moving_platform": bool(getattr(task, "moving_platform", False)),
        "status": "worker_stall_timeout",
        "success": False,
        "sim_time_sec": 0.0,
        "seed_wall_sec": max(0.0, float(elapsed_sec)),
        "step_idx": 0,
        "error": error,
    }


@dataclass
class _AdaptiveBackoffController:
    requested_workers: int
    machine_vcpus: Optional[int] = None
    machine_total_ram_mb: Optional[float] = None
    resource_provider: Optional[Any] = None
    start_worker_cap: int = field(init=False)
    active_worker_cap: int = field(init=False)
    active_heavy_cap: int = field(init=False)
    worker_cap_levels: Tuple[int, ...] = field(init=False)
    max_worker_cap: int = field(init=False)
    max_heavy_cap: int = field(init=False)
    ram_reserve_mb: float = field(init=False)
    healthy_completion_streak: int = 0
    cooldown_remaining: int = 0
    pressure_high_streak: int = 0
    pressure_critical_streak: int = 0
    recent_outcome_signals: deque[str] = field(
        default_factory=lambda: deque(maxlen=12)
    )
    recent_snapshots: deque[_ResourceSnapshot] = field(
        default_factory=lambda: deque(maxlen=_RESOURCE_WINDOW)
    )
    group_profiles: Dict[str, _GroupRuntimeProfile] = field(default_factory=dict)
    latest_pressure: str = "unknown"
    latest_snapshot: Optional[_ResourceSnapshot] = None

    def __post_init__(self) -> None:
        if self.machine_vcpus is None:
            self.machine_vcpus = available_vcpu_count()
        self.machine_vcpus = max(1, int(self.machine_vcpus))
        if self.machine_total_ram_mb is None:
            self.machine_total_ram_mb = _detect_total_ram_mb()
        self.machine_total_ram_mb = max(2048.0, float(self.machine_total_ram_mb))
        self.ram_reserve_mb = _default_ram_reserve_mb(self.machine_total_ram_mb)
        if psutil is not None:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
        ram_limited_workers = max(
            1,
            int((self.machine_total_ram_mb - self.ram_reserve_mb) // _RESOURCE_COSTS["light"].ram_mb),
        )
        self.max_worker_cap = max(
            1,
            min(
                int(self.requested_workers),
                int(self.machine_vcpus),
                int(ram_limited_workers),
            ),
        )
        self.worker_cap_levels = _worker_cap_levels(self.max_worker_cap)
        self.start_worker_cap = _initial_worker_cap(self.max_worker_cap)
        self.active_worker_cap = self.start_worker_cap
        self.max_heavy_cap = _max_heavy_active(self.max_worker_cap)
        self.active_heavy_cap = min(1, self.max_heavy_cap)
        for group_name in _GROUP_BASE_RESOURCE_COSTS:
            self.group_profiles.setdefault(group_name, _GroupRuntimeProfile())

    @property
    def enabled(self) -> bool:
        return self.max_worker_cap > 1

    def machine_summary(self) -> Dict[str, Any]:
        return {
            "requested_workers": int(self.requested_workers),
            "machine_vcpus": int(self.machine_vcpus),
            "machine_total_ram_mb": float(self.machine_total_ram_mb),
            "ram_reserve_mb": float(self.ram_reserve_mb),
            "start_worker_cap": int(self.start_worker_cap),
            "max_worker_cap": int(self.max_worker_cap),
            "max_heavy_cap": int(self.max_heavy_cap),
        }

    def cost_model(self) -> List[Dict[str, Any]]:
        return _resource_model_rows()

    def status_dict(self) -> Dict[str, Any]:
        snapshot = self.latest_snapshot
        return {
            "start_worker_cap": int(self.start_worker_cap),
            "active_worker_cap": int(self.active_worker_cap),
            "active_heavy_cap": int(self.active_heavy_cap),
            "max_worker_cap": int(self.max_worker_cap),
            "max_heavy_cap": int(self.max_heavy_cap),
            "pressure": str(self.latest_pressure),
            "cpu_percent": (
                float(snapshot.cpu_percent)
                if snapshot is not None
                else 0.0
            ),
            "load_ratio": (
                float(snapshot.load_ratio)
                if snapshot is not None
                else 0.0
            ),
            "mem_available_mb": (
                float(snapshot.mem_available_mb)
                if snapshot is not None
                else 0.0
            ),
        }

    def format_status_line(self) -> str:
        state = self.status_dict()
        return (
            f"cap={state['active_worker_cap']}/{state['max_worker_cap']} "
            f"heavy={state['active_heavy_cap']}/{state['max_heavy_cap']} "
            f"pressure={state['pressure']} "
            f"cpu={state['cpu_percent']:.1f}% "
            f"load={state['load_ratio']:.2f} "
            f"mem_avail={state['mem_available_mb']:.0f}MiB"
        )

    def describe_configuration_lines(self) -> List[str]:
        lines = [
            (
                "Scheduler machine: "
                f"vcpus={self.machine_vcpus} total_ram={self.machine_total_ram_mb / 1024.0:.1f}GiB "
                f"reserve_ram={self.ram_reserve_mb / 1024.0:.1f}GiB "
                f"start_workers={self.start_worker_cap} max_workers={self.max_worker_cap} "
                f"start_heavy={self.active_heavy_cap} max_heavy={self.max_heavy_cap}"
            )
        ]
        for row in self.cost_model():
            lines.append(
                "Scheduler cost prior: "
                f"{row['resource_class']} groups={','.join(row['groups'])} "
                f"cpu={row['cpu_units']:.2f} ram={row['ram_mb']:.0f}MiB "
                f"heavy_tokens={row['heavy_tokens']}"
            )
        return lines

    def _profile_for_group(self, group_name: str) -> _GroupRuntimeProfile:
        return self.group_profiles.setdefault(str(group_name), _GroupRuntimeProfile())

    def _learning_weight(self) -> float:
        if self.latest_pressure in {"high"}:
            return 0.35
        if self.latest_pressure == "critical":
            return 0.0
        return 1.0

    def _profile_reference_ratio(self, profile: _GroupRuntimeProfile) -> float:
        if profile.recent_ratio:
            avg_recent = sum(float(v) for v in profile.recent_ratio) / float(len(profile.recent_ratio))
            return max(float(profile.ratio_ema), float(avg_recent))
        return float(profile.ratio_ema)

    def _profile_reference_wall(self, profile: _GroupRuntimeProfile) -> float:
        if profile.recent_wall_sec:
            avg_recent = sum(float(v) for v in profile.recent_wall_sec) / float(len(profile.recent_wall_sec))
            return max(float(profile.wall_sec_ema), float(avg_recent))
        return float(profile.wall_sec_ema)

    def _learned_cost_for_group(self, group_name: str) -> _SeedResourceCost:
        base_cost = _base_resource_cost_for_group(group_name)
        profile = self.group_profiles.get(str(group_name))
        if profile is None or profile.samples <= 0:
            return base_cost

        ratio_ref = self._profile_reference_ratio(profile)
        wall_ref = self._profile_reference_wall(profile)
        timeout_rate = float(profile.timeout_like_count) / float(max(1, profile.samples))
        calibration_rate = float(profile.calibration_spike_count) / float(max(1, profile.samples))

        learned_rank = _resource_class_rank(base_cost.resource_class)
        if timeout_rate >= 0.30 or wall_ref >= 220.0 or ratio_ref >= 8.0:
            learned_rank = 2
        elif timeout_rate >= 0.12 or wall_ref >= 110.0 or ratio_ref >= 4.5:
            learned_rank = 1
        else:
            learned_rank = 0

        prior_rank = _resource_class_rank(base_cost.resource_class)
        if profile.trusted_samples < _PROFILE_MIN_TRUSTED_SAMPLES_FOR_DEMOTION:
            target_rank = max(prior_rank, learned_rank)
        else:
            target_rank = max(max(0, prior_rank - 1), learned_rank)

        resource_class = _resource_class_from_rank(target_rank)
        tier_cost = _RESOURCE_COSTS[resource_class]
        cpu_floor = (
            float(base_cost.cpu_units) * 0.90
            if target_rank < prior_rank
            else float(base_cost.cpu_units)
        )
        ram_floor = (
            float(base_cost.ram_mb) * 0.90
            if target_rank < prior_rank
            else float(base_cost.ram_mb)
        )
        cpu_units = max(float(tier_cost.cpu_units), cpu_floor)
        ram_mb = max(float(tier_cost.ram_mb), ram_floor)
        heavy_tokens = int(tier_cost.heavy_tokens)

        if target_rank == 2:
            cpu_units = max(cpu_units, 1.30 + (0.10 * calibration_rate))
            ram_mb = max(ram_mb, 3400.0 + (400.0 * timeout_rate))
            heavy_tokens = max(heavy_tokens, 2)
        elif target_rank == 1:
            cpu_units = max(cpu_units, 1.05 + (0.08 * timeout_rate))
            ram_mb = max(ram_mb, 2700.0 + (250.0 * timeout_rate))
            heavy_tokens = max(heavy_tokens, 1)

        return _SeedResourceCost(
            resource_class=_resource_class_from_heavy_tokens(heavy_tokens),
            cpu_units=float(cpu_units),
            ram_mb=float(ram_mb),
            heavy_tokens=int(heavy_tokens),
        )

    def _cost_for_group(self, group_name: str) -> _SeedResourceCost:
        return self._learned_cost_for_group(group_name)

    def dispatch_sort_key(self, group_name: str, batch_id: int) -> Tuple[int, int, float, float, int]:
        cost = self._cost_for_group(group_name)
        profile = self.group_profiles.get(str(group_name))
        expected_wall = float(profile.wall_sec_ema) if profile and profile.wall_sec_ema > 0.0 else 0.0
        warmup_penalty = 0
        if self.active_worker_cap <= max(self.start_worker_cap, _COLD_START_SMALL_WORKERS):
            warmup_penalty = int(cost.heavy_tokens)
        return (
            int(warmup_penalty),
            int(cost.heavy_tokens),
            float(cost.cpu_units),
            float(expected_wall),
            int(batch_id),
        )

    def _sample_resources(self) -> _ResourceSnapshot:
        if callable(self.resource_provider):
            raw = self.resource_provider()
            if isinstance(raw, _ResourceSnapshot):
                snapshot = raw
            else:
                raw_dict = raw if isinstance(raw, dict) else {}
                snapshot = _ResourceSnapshot(
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
                    ts=float(getattr(raw, "ts", raw_dict.get("ts", 0.0))),
                )
        else:
            snapshot = _sample_resource_snapshot(self.machine_vcpus)
        self.latest_snapshot = snapshot
        self.recent_snapshots.append(snapshot)
        return snapshot

    def _pressure_state(self) -> str:
        if not self.recent_snapshots:
            return "unknown"
        samples = list(self.recent_snapshots)[-3:]
        cpu_values = [float(sample.cpu_percent) for sample in samples if sample.cpu_percent > 0.0]
        avg_cpu = sum(cpu_values) / len(cpu_values) if cpu_values else 0.0
        load_values = [float(sample.load_ratio) for sample in samples if sample.load_ratio > 0.0]
        avg_load = sum(load_values) / len(load_values) if load_values else 0.0
        min_mem_available = min(float(sample.mem_available_mb) for sample in samples)
        if (
            avg_cpu >= _PRESSURE_CRITICAL_CPU_PERCENT
            or avg_load >= _PRESSURE_CRITICAL_LOAD_RATIO
            or min_mem_available <= (self.ram_reserve_mb + _RESOURCE_COSTS["heavy"].ram_mb)
        ):
            return "critical"
        if (
            avg_cpu >= _PRESSURE_HIGH_CPU_PERCENT
            or avg_load >= _PRESSURE_HIGH_LOAD_RATIO
            or min_mem_available <= (self.ram_reserve_mb + (_RESOURCE_COSTS["medium"].ram_mb * 1.25))
        ):
            return "high"
        if (
            (avg_cpu == 0.0 or avg_cpu <= _PRESSURE_HEALTHY_CPU_PERCENT)
            and (avg_load == 0.0 or avg_load <= _PRESSURE_HEALTHY_LOAD_RATIO)
            and min_mem_available >= (self.ram_reserve_mb + (_RESOURCE_COSTS["medium"].ram_mb * 2.0))
        ):
            return "healthy"
        return "neutral"

    def _cpu_budget_units(self) -> float:
        return max(
            float(_RESOURCE_COSTS["heavy"].cpu_units),
            min(
                float(self.machine_vcpus) * 1.10,
                float(max(1, self.active_worker_cap)) * 1.15,
            ),
        )

    def _ram_budget_mb(self) -> float:
        return max(0.0, float(self.machine_total_ram_mb) - float(self.ram_reserve_mb))

    def _heavy_token_budget(self) -> int:
        return max(2, int(self.active_worker_cap) + max(0, int(self.active_heavy_cap) - 1))

    def _active_group_costs(self, active_groups: List[str]) -> Dict[str, float]:
        cpu_units = 0.0
        ram_mb = 0.0
        heavy_tokens = 0
        heavy_count = 0
        for group_name in active_groups:
            cost = self._cost_for_group(group_name)
            cpu_units += float(cost.cpu_units)
            ram_mb += float(cost.ram_mb)
            heavy_tokens += int(cost.heavy_tokens)
            if cost.resource_class == "heavy":
                heavy_count += 1
        return {
            "cpu_units": cpu_units,
            "ram_mb": ram_mb,
            "heavy_tokens": float(heavy_tokens),
            "heavy_count": float(heavy_count),
        }

    def _recent_signal_count(self, signal: str) -> int:
        return sum(1 for item in self.recent_outcome_signals if item == signal)

    def _default_heavy_cap_for_current_workers(self, worker_cap: int) -> int:
        return _max_heavy_active(worker_cap)

    def _reduce_caps(self, *, reason: str, severity: str) -> Optional[str]:
        previous_worker_cap = int(self.active_worker_cap)
        previous_heavy_cap = int(self.active_heavy_cap)
        if severity == "critical":
            next_worker_cap = max(
                self.start_worker_cap if self.max_worker_cap > 1 else 1,
                min(self.active_worker_cap - 2, int(self.active_worker_cap * 0.7)),
            )
            next_heavy_cap = max(1, min(self.active_heavy_cap - 1, next_worker_cap))
            self.cooldown_remaining = max(self.cooldown_remaining, _BACKOFF_COOLDOWN_COMPLETIONS + 2)
        else:
            next_worker_cap = max(
                self.start_worker_cap if self.max_worker_cap > 1 else 1,
                self.active_worker_cap - 1,
            )
            next_heavy_cap = max(1, min(self.active_heavy_cap, self._default_heavy_cap_for_current_workers(next_worker_cap)))
            if next_heavy_cap == previous_heavy_cap and next_heavy_cap > 1:
                next_heavy_cap -= 1
            self.cooldown_remaining = max(self.cooldown_remaining, _BACKOFF_COOLDOWN_COMPLETIONS)

        self.active_worker_cap = max(1, min(next_worker_cap, self.max_worker_cap))
        self.active_heavy_cap = max(1, min(next_heavy_cap, self._default_heavy_cap_for_current_workers(self.active_worker_cap)))
        self.healthy_completion_streak = 0
        if self.active_worker_cap < previous_worker_cap or self.active_heavy_cap < previous_heavy_cap:
            return (
                f"Scheduler pressure backoff: workers {previous_worker_cap}->{self.active_worker_cap}, "
                f"heavy {previous_heavy_cap}->{self.active_heavy_cap} ({reason})"
            )
        return (
            f"Scheduler pressure hold: workers {self.active_worker_cap}/{self.max_worker_cap}, "
            f"heavy {self.active_heavy_cap}/{self.max_heavy_cap} ({reason})"
        )

    def _maybe_relax_caps(self) -> Optional[str]:
        if self.cooldown_remaining > 0 or self.latest_pressure not in {"healthy", "neutral"}:
            return None
        min_streak = (
            _RELAX_FAST_STREAK
            if self.active_worker_cap < min(self.max_worker_cap, 6)
            else _RELAX_NORMAL_STREAK
        )
        if self.healthy_completion_streak < min_streak:
            return None
        previous_worker_cap = int(self.active_worker_cap)
        previous_heavy_cap = int(self.active_heavy_cap)
        if self.active_worker_cap < self.max_worker_cap:
            self.active_worker_cap += 1
        target_heavy_cap = self._default_heavy_cap_for_current_workers(self.active_worker_cap)
        if (
            self.active_heavy_cap < target_heavy_cap
            and self.healthy_completion_streak >= max(min_streak + 1, 3)
            and self.active_worker_cap >= 4
        ):
            self.active_heavy_cap += 1
        self.active_heavy_cap = min(self.active_heavy_cap, target_heavy_cap)
        if (
            self.active_worker_cap == previous_worker_cap
            and self.active_heavy_cap == previous_heavy_cap
        ):
            return None
        self.healthy_completion_streak = 0
        self.cooldown_remaining = 1 if self.active_worker_cap < self.max_worker_cap else 0
        if self.active_worker_cap >= self.max_worker_cap and self.active_heavy_cap >= self.max_heavy_cap:
            return (
                f"Scheduler relaxed: restored full dispatch "
                f"{self.active_worker_cap}/{self.max_worker_cap} workers "
                f"and {self.active_heavy_cap}/{self.max_heavy_cap} heavy"
            )
        return (
            f"Scheduler relaxed: workers {previous_worker_cap}->{self.active_worker_cap}, "
            f"heavy {previous_heavy_cap}->{self.active_heavy_cap}"
        )

    def _observe_group_profile(
        self,
        group_name: Optional[str],
        seed_meta: Optional[Dict[str, Any]],
        *,
        timeout_like: bool,
        calibration_spike: bool,
    ) -> None:
        resolved_group = str(group_name) if group_name else None
        if resolved_group is None and isinstance(seed_meta, dict):
            resolved_group = _group_name_from_challenge_type(seed_meta.get("challenge_type"))
        if not resolved_group:
            return
        profile = self._profile_for_group(resolved_group)
        profile.samples += 1
        if timeout_like:
            profile.timeout_like_count += 1
        if calibration_spike:
            profile.calibration_spike_count += 1

        wall_sec = 0.0
        try:
            wall_sec = max(0.0, float(seed_meta.get("seed_wall_sec", 0.0))) if isinstance(seed_meta, dict) else 0.0
        except Exception:
            wall_sec = 0.0
        ratio = _seed_runtime_ratio(seed_meta)
        if wall_sec > 0.0:
            profile.recent_wall_sec.append(float(wall_sec))
        if ratio > 0.0:
            profile.recent_ratio.append(float(ratio))

        trust = self._learning_weight()
        if trust <= 0.0 or wall_sec <= 0.0:
            return
        profile.trusted_samples += 1
        alpha = 0.35 if profile.trusted_samples <= 4 else 0.20
        alpha *= trust
        if profile.wall_sec_ema <= 0.0:
            profile.wall_sec_ema = float(wall_sec)
        else:
            profile.wall_sec_ema = ((1.0 - alpha) * float(profile.wall_sec_ema)) + (alpha * float(wall_sec))
        if ratio > 0.0:
            if profile.ratio_ema <= 0.0:
                profile.ratio_ema = float(ratio)
            else:
                profile.ratio_ema = ((1.0 - alpha) * float(profile.ratio_ema)) + (alpha * float(ratio))

    def observe_resources(self, active_groups: List[str]) -> Optional[str]:
        self._sample_resources()
        self.latest_pressure = self._pressure_state()
        if self.latest_pressure == "critical":
            self.pressure_critical_streak += 1
            self.pressure_high_streak += 1
        elif self.latest_pressure == "high":
            self.pressure_high_streak += 1
            self.pressure_critical_streak = 0
        else:
            self.pressure_high_streak = 0
            self.pressure_critical_streak = 0

        if not self.enabled:
            return None

        snapshot = self.latest_snapshot
        if snapshot is None:
            return None

        if self.pressure_critical_streak >= _PRESSURE_CRITICAL_STREAK:
            self.pressure_critical_streak = 0
            self.pressure_high_streak = 0
            return self._reduce_caps(
                reason=(
                    f"host pressure cpu={snapshot.cpu_percent:.1f}% "
                    f"load={snapshot.load_ratio:.2f} "
                    f"mem_avail={snapshot.mem_available_mb:.0f}MiB"
                ),
                severity="critical",
            )

        if self.pressure_high_streak >= _PRESSURE_HIGH_STREAK:
            self.pressure_high_streak = 0
            return self._reduce_caps(
                reason=(
                    f"host pressure cpu={snapshot.cpu_percent:.1f}% "
                    f"load={snapshot.load_ratio:.2f} "
                    f"mem_avail={snapshot.mem_available_mb:.0f}MiB"
                ),
                severity="moderate",
            )
        return None

    def can_admit_group(self, group_name: str, active_groups: List[str]) -> bool:
        if len(active_groups) >= self.active_worker_cap:
            return False
        cost = self._cost_for_group(group_name)
        class_name = cost.resource_class
        active_costs = self._active_group_costs(active_groups)
        group_limit = int(_GROUP_CONCURRENCY_LIMITS.get(group_name, self.active_worker_cap))
        if sum(1 for group in active_groups if group == group_name) >= group_limit:
            return False
        if (
            class_name == "heavy"
            and int(active_costs["heavy_count"]) >= int(self.active_heavy_cap)
        ):
            return False
        if (active_costs["cpu_units"] + float(cost.cpu_units)) > self._cpu_budget_units():
            return False
        if (active_costs["ram_mb"] + float(cost.ram_mb)) > self._ram_budget_mb():
            return False
        if (active_costs["heavy_tokens"] + float(cost.heavy_tokens)) > float(self._heavy_token_budget()):
            return False
        snapshot = self.latest_snapshot
        if (
            snapshot is not None
            and snapshot.mem_available_mb > 0.0
            and (snapshot.mem_available_mb - self.ram_reserve_mb) < float(cost.ram_mb)
        ):
            return False
        if self.latest_pressure == "critical" and class_name != "light":
            return False
        return True

    def observe_seed(
        self,
        seed_meta: Optional[Dict[str, Any]],
        *,
        group_name: Optional[str] = None,
    ) -> Optional[str]:
        if not isinstance(seed_meta, dict):
            return None

        status = str(seed_meta.get("status", "")).strip() or "unknown"
        if status in _NON_RESOURCE_FAILURE_STATUSES:
            self.healthy_completion_streak = 0
            return None

        hard_issue = _is_backoff_infra_status(status)
        timeout_issue = _is_backoff_timeout_status(status)
        soft_signal = status in _RESOURCE_SOFT_SIGNAL_STATUSES
        calibration_spike = _seed_has_calibration_spike(seed_meta)
        timeout_like = timeout_issue or soft_signal

        signal = "clean"
        if hard_issue:
            signal = "hard"
        elif timeout_like or calibration_spike:
            signal = "soft"
        self.recent_outcome_signals.append(signal)
        self._observe_group_profile(
            group_name,
            seed_meta,
            timeout_like=timeout_like,
            calibration_spike=calibration_spike,
        )

        reason: Optional[str] = None
        severity = "moderate"
        if hard_issue:
            reason = f"final outcome {status} {_format_seed_desc(seed_meta)}"
            severity = "critical" if self.latest_pressure == "critical" else "moderate"
        elif timeout_issue and self.latest_pressure in {"high", "critical"}:
            reason = f"timeout under {self.latest_pressure} pressure ({status} {_format_seed_desc(seed_meta)})"
        elif timeout_issue and self._recent_signal_count("soft") >= 4:
            reason = f"repeated timeout-like outcomes ({status} {_format_seed_desc(seed_meta)})"
        elif soft_signal and self.latest_pressure == "critical":
            reason = f"resource-sensitive failure under critical pressure ({status} {_format_seed_desc(seed_meta)})"
        elif calibration_spike and self.latest_pressure in {"high", "critical"}:
            try:
                overhead_ms = float(seed_meta.get("calibration_overhead_sec") or 0.0) * 1000.0
            except Exception:
                overhead_ms = 0.0
            try:
                cpu_factor = float(seed_meta.get("calibration_cpu_factor") or 1.0)
            except Exception:
                cpu_factor = 1.0
            reason = (
                f"calibration spike {_format_seed_desc(seed_meta)} "
                f"overhead={overhead_ms:.1f}ms cpu_factor={cpu_factor:.2f}x"
            )

        if reason is not None and self.enabled:
            return self._reduce_caps(reason=reason, severity=severity)

        if status == "seed_done" and not calibration_spike:
            if self.latest_pressure in {"healthy", "neutral", "unknown"}:
                self.healthy_completion_streak += 1
                if self.cooldown_remaining > 0:
                    self.cooldown_remaining = max(0, self.cooldown_remaining - 1)
                return self._maybe_relax_caps()
        else:
            self.healthy_completion_streak = 0
            if self.cooldown_remaining > 0:
                self.cooldown_remaining = max(0, self.cooldown_remaining - 1)
        return None


def _select_next_batch_index(
    *,
    pending_batch_ids: List[int],
    batch_plan: List[List[int]],
    task_meta: List[Dict[str, Any]],
    active_batch_ids: List[int],
    active_worker_cap: int,
    scheduler: Optional[_AdaptiveBackoffController] = None,
) -> Optional[int]:
    if not pending_batch_ids:
        return None

    active_groups = [
        str(task_meta[batch_plan[batch_id][0]]["group"])
        for batch_id in active_batch_ids
        if batch_plan[batch_id]
    ]
    active_group_counts = Counter(active_groups)
    active_heavy = sum(1 for group_name in active_groups if group_name in _HEAVY_GROUPS)

    def _is_preferred(batch_id: int) -> bool:
        if not batch_plan[batch_id]:
            return False
        group_name = str(task_meta[batch_plan[batch_id][0]]["group"])
        if scheduler is not None:
            return scheduler.can_admit_group(group_name, active_groups)
        group_limit = int(_GROUP_CONCURRENCY_LIMITS.get(group_name, active_worker_cap))
        if active_group_counts[group_name] >= group_limit:
            return False
        if group_name in _HEAVY_GROUPS and active_heavy >= _max_heavy_active(active_worker_cap):
            return False
        return True

    preferred = [batch_id for batch_id in pending_batch_ids if _is_preferred(batch_id)]
    if scheduler is not None and not preferred:
        return None
    candidate_pool = preferred if preferred else pending_batch_ids

    def _sort_key(batch_id: int) -> Tuple[Any, ...]:
        group_name = str(task_meta[batch_plan[batch_id][0]]["group"])
        if scheduler is not None:
            return scheduler.dispatch_sort_key(group_name, batch_id)
        return (
            int(1 if group_name in _HEAVY_GROUPS else 0),
            int(_GROUP_CONCURRENCY_LIMITS.get(group_name, active_worker_cap)),
            int(batch_id),
        )

    return min(candidate_pool, key=_sort_key)
