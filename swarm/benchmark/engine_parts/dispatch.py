from __future__ import annotations

from ._shared import Any, Counter, Dict, List, Optional, Tuple, dataclass, deque, field

_GROUP_DISPATCH_WEIGHTS = {
    "type1_city": 1,
    "type2_open": 1,
    "type3_mountain": 4,
    "type4_village": 2,
    "type5_warehouse": 3,
    "type6_forest": 2,
}
_GROUP_CONCURRENCY_LIMITS = {
    "type3_mountain": 1,
    "type5_warehouse": 1,
}
_HEAVY_GROUPS = frozenset({"type3_mountain", "type5_warehouse", "type6_forest"})
_BACKOFF_ACTIVE_WORKERS = 2
_BACKOFF_RECENT_WINDOW = 6
_BACKOFF_MIN_SAMPLES = 4
_BACKOFF_COOLDOWN_COMPLETIONS = 6
_BACKOFF_CLEAN_RATE_THRESHOLD = 0.85
_BACKOFF_CALIBRATION_OVERHEAD_SPIKE_SEC = 0.25
_BACKOFF_CALIBRATION_CPU_FACTOR_SPIKE = 1.5
_PARENT_WORKER_HEARTBEAT_SEC = 15.0
_PARENT_WORKER_STALL_TIMEOUT_SEC = 90.0
_BACKOFF_FAILURE_STATUSES = frozenset(
    {
        "batch_timeout",
        "batch_exception",
        "rpc_connection_failed",
        "rpc_ping_timeout",
        "rpc_connect_failed",
        "rpc_agent_unavailable",
        "seed_timeout_strikes",
        "seed_rpc_disconnected",
        "seed_exception",
        "seed_cancelled",
        "worker_stall_timeout",
    }
)


def _is_clean_execution_status(status: str) -> bool:
    return status == "seed_done"


def _group_dispatch_weight(group_name: str) -> int:
    return int(_GROUP_DISPATCH_WEIGHTS.get(group_name, 1))


def _max_heavy_active(active_worker_cap: int) -> int:
    worker_cap = max(1, int(active_worker_cap))
    return max(1, min(worker_cap // 2, 4))


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
    return (
        overhead_val >= _BACKOFF_CALIBRATION_OVERHEAD_SPIKE_SEC
        or cpu_factor_val >= _BACKOFF_CALIBRATION_CPU_FACTOR_SPIKE
    )


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
    active_worker_cap: int = field(init=False)
    cooldown_remaining: int = 0
    recent_statuses: deque[str] = field(
        default_factory=lambda: deque(maxlen=_BACKOFF_RECENT_WINDOW)
    )
    recent_spikes: deque[bool] = field(
        default_factory=lambda: deque(maxlen=_BACKOFF_RECENT_WINDOW)
    )

    def __post_init__(self) -> None:
        self.active_worker_cap = max(1, int(self.requested_workers))

    @property
    def enabled(self) -> bool:
        return self.requested_workers > _BACKOFF_ACTIVE_WORKERS

    def recent_clean_rate(self) -> float:
        if not self.recent_statuses:
            return 1.0
        clean = sum(1 for status in self.recent_statuses if _is_clean_execution_status(status))
        return clean / float(len(self.recent_statuses))

    def recent_window_is_healthy(self) -> bool:
        if len(self.recent_statuses) < _BACKOFF_RECENT_WINDOW:
            return False
        return (
            all(_is_clean_execution_status(status) for status in self.recent_statuses)
            and not any(self.recent_spikes)
        )

    def observe_seed(self, seed_meta: Optional[Dict[str, Any]]) -> Optional[str]:
        if not self.enabled or not isinstance(seed_meta, dict):
            return None

        status = str(seed_meta.get("status", "")).strip() or "unknown"
        spike = _seed_has_calibration_spike(seed_meta)
        self.recent_statuses.append(status)
        self.recent_spikes.append(spike)

        trigger_reason: Optional[str] = None
        if status in _BACKOFF_FAILURE_STATUSES:
            trigger_reason = f"status={status} {_format_seed_desc(seed_meta)}"
        elif spike:
            try:
                overhead_ms = float(seed_meta.get("calibration_overhead_sec") or 0.0) * 1000.0
            except Exception:
                overhead_ms = 0.0
            try:
                cpu_factor = float(seed_meta.get("calibration_cpu_factor") or 1.0)
            except Exception:
                cpu_factor = 1.0
            trigger_reason = (
                f"calibration spike {_format_seed_desc(seed_meta)} "
                f"overhead={overhead_ms:.1f}ms cpu_factor={cpu_factor:.2f}x"
            )
        elif (
            len(self.recent_statuses) >= _BACKOFF_MIN_SAMPLES
            and self.recent_clean_rate() < _BACKOFF_CLEAN_RATE_THRESHOLD
        ):
            trigger_reason = (
                f"recent clean execution {self.recent_clean_rate() * 100.0:.0f}% "
                f"over last {len(self.recent_statuses)} seeds"
            )

        if trigger_reason is not None:
            previous_cap = self.active_worker_cap
            self.active_worker_cap = min(self.active_worker_cap, _BACKOFF_ACTIVE_WORKERS)
            self.cooldown_remaining = max(
                self.cooldown_remaining,
                _BACKOFF_COOLDOWN_COMPLETIONS,
            )
            if previous_cap > self.active_worker_cap:
                return (
                    f"Adaptive backoff active: limiting dispatch to "
                    f"{self.active_worker_cap}/{self.requested_workers} workers "
                    f"({trigger_reason})"
                )
            return (
                f"Adaptive backoff extended: keeping dispatch at "
                f"{self.active_worker_cap}/{self.requested_workers} workers "
                f"({trigger_reason})"
            )

        if self.active_worker_cap < self.requested_workers:
            self.cooldown_remaining = max(0, self.cooldown_remaining - 1)
            if self.cooldown_remaining == 0 and self.recent_window_is_healthy():
                self.active_worker_cap = self.requested_workers
                return (
                    f"Adaptive backoff cleared: restoring dispatch to "
                    f"{self.requested_workers} workers"
                )

        return None


def _select_next_batch_index(
    *,
    pending_batch_ids: List[int],
    batch_plan: List[List[int]],
    task_meta: List[Dict[str, Any]],
    active_batch_ids: List[int],
    active_worker_cap: int,
) -> Optional[int]:
    if not pending_batch_ids:
        return None

    active_groups = Counter(
        str(task_meta[batch_plan[batch_id][0]]["group"])
        for batch_id in active_batch_ids
        if batch_plan[batch_id]
    )
    active_heavy = sum(
        1
        for batch_id in active_batch_ids
        if batch_plan[batch_id]
        and str(task_meta[batch_plan[batch_id][0]]["group"]) in _HEAVY_GROUPS
    )

    def _is_preferred(batch_id: int) -> bool:
        if not batch_plan[batch_id]:
            return False
        group_name = str(task_meta[batch_plan[batch_id][0]]["group"])
        group_limit = int(_GROUP_CONCURRENCY_LIMITS.get(group_name, active_worker_cap))
        if active_groups[group_name] >= group_limit:
            return False
        if group_name in _HEAVY_GROUPS and active_heavy >= _max_heavy_active(active_worker_cap):
            return False
        return True

    preferred = [batch_id for batch_id in pending_batch_ids if _is_preferred(batch_id)]
    candidate_pool = preferred if preferred else pending_batch_ids

    def _sort_key(batch_id: int) -> Tuple[int, int]:
        group_name = str(task_meta[batch_plan[batch_id][0]]["group"])
        return (_group_dispatch_weight(group_name), -batch_id)

    return max(candidate_pool, key=_sort_key)
