from __future__ import annotations

from ._shared import (
    BENCH_GROUP_ORDER,
    Any,
    Counter,
    Dict,
    List,
    Tuple,
    _BatchStat,
    asdict,
    deque,
    statistics,
    time,
)
from .dispatch import _is_clean_execution_status


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _avg(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _bytes_to_mib(value: Any) -> float:
    return _safe_float(value) / (1024.0 * 1024.0)


def _batch_share_breakdown(batch_stat: _BatchStat | None) -> Dict[str, float]:
    if batch_stat is None:
        return {
            "workspace_prepare_share_sec": 0.0,
            "container_start_share_sec": 0.0,
            "pip_install_share_sec": 0.0,
            "template_restage_share_sec": 0.0,
            "container_pid_lookup_share_sec": 0.0,
            "network_lockdown_share_sec": 0.0,
            "submission_launch_share_sec": 0.0,
            "rpc_ready_wait_share_sec": 0.0,
            "rpc_internal_share_sec": 0.0,
            "container_cleanup_share_sec": 0.0,
            "tmpdir_cleanup_share_sec": 0.0,
            "batch_unaccounted_share_sec": 0.0,
            "shared_total_sec": 0.0,
        }

    seed_count = max(1, int(batch_stat.seed_count))
    timing = dict(getattr(batch_stat, "timing_breakdown", {}) or {})
    rpc_batch_wait_sec = _safe_float(timing.get("rpc_batch_wait_sec"))
    rpc_internal_share_sec = max(
        0.0, rpc_batch_wait_sec - _safe_float(batch_stat.seed_processing_sec)
    ) / seed_count
    share = {
        "workspace_prepare_share_sec": _safe_float(
            timing.get("workspace_prepare_total_sec")
        )
        / seed_count,
        "container_start_share_sec": _safe_float(timing.get("container_start_sec"))
        / seed_count,
        "pip_install_share_sec": _safe_float(timing.get("pip_install_wait_sec"))
        / seed_count,
        "template_restage_share_sec": _safe_float(
            timing.get("template_restage_sec")
        )
        / seed_count,
        "container_pid_lookup_share_sec": _safe_float(
            timing.get("container_pid_lookup_sec")
        )
        / seed_count,
        "network_lockdown_share_sec": _safe_float(
            timing.get("network_lockdown_sec")
        )
        / seed_count,
        "submission_launch_share_sec": _safe_float(
            timing.get("submission_launch_sec")
        )
        / seed_count,
        "rpc_ready_wait_share_sec": _safe_float(timing.get("rpc_ready_wait_sec"))
        / seed_count,
        "rpc_internal_share_sec": rpc_internal_share_sec,
        "container_cleanup_share_sec": _safe_float(
            timing.get("container_cleanup_sec")
        )
        / seed_count,
        "tmpdir_cleanup_share_sec": _safe_float(
            timing.get("tmpdir_cleanup_wait_sec")
        )
        / seed_count,
        "batch_unaccounted_share_sec": _safe_float(
            timing.get("batch_unaccounted_sec")
        )
        / seed_count,
    }
    share["shared_total_sec"] = sum(share.values())
    return share


def _build_seed_timing_detail(
    *,
    processing_wall: float,
    wall: float,
    seed_detail: Dict[str, Any],
    batch_stat: _BatchStat | None,
) -> Dict[str, Any]:
    processing_breakdown = dict(seed_detail.get("timing_breakdown", {}) or {})
    rollout_breakdown = dict(seed_detail.get("rollout_breakdown", {}) or {})
    latency_stats = dict(seed_detail.get("latency_stats", {}) or {})
    step_metrics = dict(seed_detail.get("step_metrics", {}) or {})
    memory_breakdown = dict(seed_detail.get("memory_breakdown", {}) or {})
    processing_total = _safe_float(
        processing_breakdown.get("seed_total_sec"), processing_wall
    )
    batch_share = _batch_share_breakdown(batch_stat)
    shared_total = _safe_float(batch_share.get("shared_total_sec"))
    return {
        "batch_index": seed_detail.get(
            "batch_index", getattr(batch_stat, "batch_index", None)
        ),
        "worker_id": seed_detail.get(
            "worker_id", getattr(batch_stat, "worker_id", None)
        ),
        "seed_processing_sec": processing_total,
        "container_shared_sec": shared_total,
        "full_wall_sec": wall,
        "full_wall_unaccounted_sec": max(
            0.0, wall - (processing_total + shared_total)
        ),
        "seed_processing_breakdown": processing_breakdown,
        "rollout_breakdown": rollout_breakdown,
        "container_share_breakdown": batch_share,
        "latency_stats": latency_stats,
        "step_metrics": step_metrics,
        "memory_breakdown": memory_breakdown,
    }


def _phase_averages(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}

    def _collect(path: tuple[str, ...]) -> List[float]:
        values: List[float] = []
        for row in rows:
            current: Any = row
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if current is None:
                continue
            try:
                values.append(float(current))
            except Exception:
                continue
        return values

    return {
        "full_wall_sec": _avg(_collect(("wall_time",))),
        "container_shared_sec": _avg(_collect(("timing_detail", "container_shared_sec"))),
        "workspace_prepare_share_sec": _avg(
            _collect(("timing_detail", "container_share_breakdown", "workspace_prepare_share_sec"))
        ),
        "container_start_share_sec": _avg(
            _collect(("timing_detail", "container_share_breakdown", "container_start_share_sec"))
        ),
        "rpc_ready_wait_share_sec": _avg(
            _collect(("timing_detail", "container_share_breakdown", "rpc_ready_wait_share_sec"))
        ),
        "rpc_internal_share_sec": _avg(
            _collect(("timing_detail", "container_share_breakdown", "rpc_internal_share_sec"))
        ),
        "container_cleanup_share_sec": _avg(
            _collect(("timing_detail", "container_share_breakdown", "container_cleanup_share_sec"))
        ),
        "seed_processing_sec": _avg(_collect(("timing_detail", "seed_processing_sec"))),
        "env_build_sec": _avg(
            _collect(("timing_detail", "seed_processing_breakdown", "env_build_sec"))
        ),
        "env_reset_sec": _avg(
            _collect(("timing_detail", "seed_processing_breakdown", "env_reset_sec"))
        ),
        "agent_reset_sec": _avg(
            _collect(("timing_detail", "seed_processing_breakdown", "agent_reset_sec"))
        ),
        "calibration_sec": _avg(
            _collect(("timing_detail", "seed_processing_breakdown", "calibration_sec"))
        ),
        "rollout_total_sec": _avg(
            _collect(("timing_detail", "rollout_breakdown", "rollout_total_sec"))
        ),
        "observation_serialize_sec": _avg(
            _collect(("timing_detail", "rollout_breakdown", "observation_serialize_sec"))
        ),
        "agent_act_sec": _avg(
            _collect(("timing_detail", "rollout_breakdown", "agent_act_sec"))
        ),
        "action_decode_sec": _avg(
            _collect(("timing_detail", "rollout_breakdown", "action_decode_sec"))
        ),
        "action_postprocess_sec": _avg(
            _collect(("timing_detail", "rollout_breakdown", "action_postprocess_sec"))
        ),
        "env_step_sec": _avg(
            _collect(("timing_detail", "rollout_breakdown", "env_step_sec"))
        ),
        "rollout_other_sec": _avg(
            _collect(("timing_detail", "rollout_breakdown", "rollout_other_sec"))
        ),
        "env_cleanup_sec": _avg(
            _collect(("timing_detail", "seed_processing_breakdown", "env_cleanup_sec"))
        ),
        "seed_processing_unaccounted_sec": _avg(
            _collect(("timing_detail", "seed_processing_breakdown", "seed_unaccounted_sec"))
        ),
        "full_wall_unaccounted_sec": _avg(
            _collect(("timing_detail", "full_wall_unaccounted_sec"))
        ),
    }


def _memory_averages(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    if not rows:
        return {}

    def _collect(path: tuple[str, ...]) -> List[float]:
        values: List[float] = []
        for row in rows:
            current: Any = row
            for key in path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if current is None:
                continue
            try:
                values.append(float(current))
            except Exception:
                continue
        return values

    return {
        "validator_avg_rss_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "validator_process_rss", "avg_bytes"))]
        ),
        "validator_peak_rss_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "validator_process_rss", "peak_bytes"))]
        ),
        "validator_peak_delta_rss_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "validator_process_rss", "peak_delta_from_first_bytes"))]
        ),
        "validator_avg_vms_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "validator_process_vms", "avg_bytes"))]
        ),
        "validator_peak_vms_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "validator_process_vms", "peak_bytes"))]
        ),
        "validator_peak_delta_vms_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "validator_process_vms", "peak_delta_from_first_bytes"))]
        ),
        "container_avg_mem_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "docker_container_memory", "avg_bytes"))]
        ),
        "container_peak_mem_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "docker_container_memory", "peak_bytes"))]
        ),
        "container_peak_delta_mem_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "docker_container_memory", "peak_delta_from_first_bytes"))]
        ),
        "container_limit_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "docker_container_memory_limit_bytes"))]
        ),
        "host_total_ram_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "system_total_bytes"))]
        ),
        "host_available_ram_avg_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "system_available_memory", "avg_bytes"))]
        ),
        "host_available_ram_min_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "system_available_memory", "min_bytes"))]
        ),
        "host_available_ram_drop_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "system_available_memory", "min_delta_from_first_bytes"))]
        ),
        "host_used_ram_avg_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "system_used_memory", "avg_bytes"))]
        ),
        "host_used_ram_peak_mib": _avg(
            [_bytes_to_mib(v) for v in _collect(("timing_detail", "memory_breakdown", "system_used_memory", "peak_bytes"))]
        ),
    }


def _batch_phase_averages(batch_stats: List[_BatchStat]) -> Dict[str, float]:
    if not batch_stats:
        return {}
    timing_dicts = [dict(getattr(stat, "timing_breakdown", {}) or {}) for stat in batch_stats]
    keys = [
        "batch_total_sec",
        "workspace_prepare_total_sec",
        "container_start_sec",
        "pip_install_wait_sec",
        "template_restage_sec",
        "container_pid_lookup_sec",
        "network_lockdown_sec",
        "submission_launch_sec",
        "rpc_ready_wait_sec",
        "rpc_batch_wait_sec",
        "container_cleanup_sec",
        "tmpdir_cleanup_wait_sec",
        "batch_unaccounted_sec",
    ]
    return {
        key: _avg([_safe_float(timing.get(key)) for timing in timing_dicts])
        for key in keys
    }


def _weighted_latency_summary(rows: List[Dict[str, Any]], name: str) -> Dict[str, float]:
    total_sec = 0.0
    total_count = 0
    max_ms = 0.0
    p95_values: List[float] = []
    for row in rows:
        stats = (
            row.get("timing_detail", {})
            .get("latency_stats", {})
            .get(name, {})
        )
        total_sec += _safe_float(stats.get("total_sec"))
        total_count += int(_safe_float(stats.get("count")))
        max_ms = max(max_ms, _safe_float(stats.get("max_ms")))
        p95 = _safe_float(stats.get("p95_ms"))
        if p95 > 0.0:
            p95_values.append(p95)
    return {
        "count": total_count,
        "total_sec": total_sec,
        "avg_ms": (total_sec / total_count * 1000.0) if total_count > 0 else 0.0,
        "p95_ms_seed_avg": _avg(p95_values),
        "max_ms": max_ms,
    }


def _print_results(
    task_meta: List[Dict[str, Any]],
    results: list,
    seed_times: List[float],
    seed_wall_by_key: Dict[Tuple[int, int], deque[float]],
    seed_status_by_key: Dict[Tuple[int, int], deque[str]],
    seed_detail_by_key: Dict[Tuple[int, int], deque[Dict[str, Any]]],
    full_wall_by_key: Dict[Tuple[int, int], float],
    batch_stats: List[_BatchStat],
    elapsed: float,
    eval_start: float,
    num_workers: int,
    host_parallelism: str = "process",
) -> Dict[str, Any]:
    group_order = BENCH_GROUP_ORDER
    seed_wall_queues: Dict[Tuple[int, int], deque[float]] = {
        key: deque(values) for key, values in seed_wall_by_key.items()
    }
    seed_status_queues: Dict[Tuple[int, int], deque[str]] = {
        key: deque(values) for key, values in seed_status_by_key.items()
    }
    seed_detail_queues: Dict[Tuple[int, int], deque[Dict[str, Any]]] = {
        key: deque(dict(value) for value in values) for key, values in seed_detail_by_key.items()
    }
    batch_by_index = {int(stat.batch_index): stat for stat in batch_stats}

    group_results: Dict[str, List[Dict[str, Any]]] = {}
    timing_rows: List[Dict[str, Any]] = []
    for i, meta in enumerate(task_meta):
        group = meta["group"]
        group_results.setdefault(group, [])

        result = results[i] if i < len(results) else None
        score = float(result.score) if result else 0.0
        success = bool(result.success) if result else False
        sim_time = float(result.time_sec) if result else 0.0

        seed_key = (int(meta["seed"]), int(meta["challenge_type"]))
        status_q = seed_status_queues.get(seed_key)
        execution_status = str(status_q.popleft()) if status_q and len(status_q) > 0 else "unknown"
        execution_ok = _is_clean_execution_status(execution_status)
        wall_q = seed_wall_queues.get(seed_key)
        if wall_q and len(wall_q) > 0:
            processing_wall = float(wall_q.popleft())
        elif i < len(seed_times):
            processing_wall = (seed_times[i] - seed_times[i - 1]) if i > 0 else (seed_times[0] - eval_start)
        else:
            processing_wall = 0.0

        seed_detail_q = seed_detail_queues.get(seed_key)
        seed_detail = dict(seed_detail_q.popleft()) if seed_detail_q and len(seed_detail_q) > 0 else {}
        batch_index = seed_detail.get("batch_index")
        batch_stat = batch_by_index.get(int(batch_index)) if batch_index is not None else None
        wall = float(full_wall_by_key.get(seed_key, 0.0))
        if wall <= 0.0:
            wall = processing_wall + _safe_float(
                _batch_share_breakdown(batch_stat).get("shared_total_sec")
            )

        timing_detail = _build_seed_timing_detail(
            processing_wall=processing_wall,
            wall=wall,
            seed_detail=seed_detail,
            batch_stat=batch_stat,
        )

        row = {
            "group": group,
            "seed": meta["seed"],
            "score": score,
            "success": success,
            "sim_time": sim_time,
            "wall_time": wall,
            "processing_wall_time": processing_wall,
            "execution_status": execution_status,
            "execution_ok": execution_ok,
            "timeout_zero": wall < 0.5 and i > 0,
            "timing_detail": timing_detail,
        }
        group_results[group].append(row)
        timing_rows.append(row)

    print()
    print(f"  {'Group':<18} {'Seed':>8} {'Score':>7} {'OK?':>5} {'SimT':>7} {'WallT':>7}")
    print(f"  {'-'*18} {'-'*8} {'-'*7} {'-'*5} {'-'*7} {'-'*7}")

    for group in group_order:
        if group not in group_results:
            continue
        for row in group_results[group]:
            ok = "Y" if row["success"] else "N"
            print(
                f"  {group:<18} {row['seed']:>8} {row['score']:>7.4f} {ok:>5} "
                f"{row['sim_time']:>6.2f}s {row['wall_time']:>6.1f}s"
            )
        walls = [row["wall_time"] for row in group_results[group]]
        scores = [row["score"] for row in group_results[group]]
        avg_w = sum(walls) / len(walls) if walls else 0
        avg_s = sum(scores) / len(scores) if scores else 0
        print(f"  {'  -> AVG':<18} {'':>8} {avg_s:>7.4f} {'':>5} {'':>6} {avg_w:>6.1f}s")
        print()

    all_rows = [row for group in group_order if group in group_results for row in group_results[group]]
    all_seed_walls = [float(row["wall_time"]) for row in all_rows if float(row["wall_time"]) > 0.0]
    all_sim_times = [float(row["sim_time"]) for row in all_rows]
    success_count = sum(1 for row in all_rows if bool(row["success"]))
    execution_success_count = sum(1 for row in all_rows if bool(row["execution_ok"]))
    execution_status_counts = Counter(str(row["execution_status"]) for row in all_rows)
    total_seeds = len(all_rows)
    workers_used = max(1, int(num_workers))

    avg_wall_per_seed = (sum(all_seed_walls) / len(all_seed_walls)) if all_seed_walls else 0.0
    med_wall_per_seed = statistics.median(all_seed_walls) if all_seed_walls else 0.0
    if all_seed_walls:
        sorted_walls = sorted(all_seed_walls)
        p90_idx = max(0, int(round(0.9 * len(sorted_walls) + 0.5)) - 1)
        p90_wall_per_seed = sorted_walls[min(p90_idx, len(sorted_walls) - 1)]
    else:
        p90_wall_per_seed = 0.0
    avg_sim_per_seed = (sum(all_sim_times) / len(all_sim_times)) if all_sim_times else 0.0

    throughput_seeds_per_min = (total_seeds / elapsed * 60.0) if elapsed > 0 else 0.0
    throughput_per_worker = throughput_seeds_per_min / workers_used
    total_seed_worker_sec = sum(float(stat.elapsed_sec) for stat in batch_stats) if batch_stats else sum(all_seed_walls)
    effective_parallelism = (total_seed_worker_sec / elapsed) if elapsed > 0 else 0.0
    worker_utilization = min(1.0, effective_parallelism / workers_used) if workers_used > 0 else 0.0
    total_startup_overhead_sec = sum(float(stat.startup_overhead_sec) for stat in batch_stats)
    avg_startup_overhead_sec = total_startup_overhead_sec / len(batch_stats) if batch_stats else 0.0
    avg_batch_size = (
        sum(int(stat.seed_count) for stat in batch_stats) / len(batch_stats)
        if batch_stats
        else 0.0
    )

    print("  Run summary:")
    print(f"    Seeds evaluated:           {total_seeds}")
    print(
        f"    Success rate:              {success_count}/{total_seeds} "
        f"({(100.0 * success_count / total_seeds) if total_seeds else 0.0:.1f}%)"
    )
    print(
        f"    Clean execution rate:      {execution_success_count}/{total_seeds} "
        f"({(100.0 * execution_success_count / total_seeds) if total_seeds else 0.0:.1f}%)"
    )
    execution_failures = [
        f"{status}={count}"
        for status, count in sorted(execution_status_counts.items())
        if not _is_clean_execution_status(status)
    ]
    if execution_failures:
        print(f"    Execution failure modes:   {', '.join(execution_failures)}")
    print(f"    Total wall-clock:          {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"    Avg wall / seed:           {avg_wall_per_seed:.2f}s")
    print(f"    Median wall / seed:        {med_wall_per_seed:.2f}s")
    print(f"    P90 wall / seed:           {p90_wall_per_seed:.2f}s")
    print(f"    Avg sim time / seed:       {avg_sim_per_seed:.2f}s")
    print(f"    Total seed-worker time:    {total_seed_worker_sec:.1f}s")
    print(f"    Throughput:                {throughput_seeds_per_min:.2f} seeds/min")
    print(f"    Throughput per worker:     {throughput_per_worker:.2f} seeds/min/worker")
    print(
        f"    Effective parallelism:     {effective_parallelism:.2f}x "
        f"(utilization {worker_utilization * 100.0:.1f}% of {workers_used} workers)"
    )
    print(f"    Batches run:               {len(batch_stats)}")
    print(f"    Avg seeds / container:     {avg_batch_size:.2f}")
    print(f"    Total startup overhead:    {total_startup_overhead_sec:.1f}s")
    print(f"    Avg startup / container:   {avg_startup_overhead_sec:.1f}s")
    print()

    phase_avgs = _phase_averages(all_rows)
    memory_avgs = _memory_averages(all_rows)
    act_latency_summary = _weighted_latency_summary(all_rows, "act")
    env_step_latency_summary = _weighted_latency_summary(all_rows, "env_step")
    step_latency_summary = _weighted_latency_summary(all_rows, "step_total")

    print("  Timing breakdown (avg per seed):")
    print(f"    Container shared:          {phase_avgs.get('container_shared_sec', 0.0):.2f}s")
    print(f"    Workspace prepare share:   {phase_avgs.get('workspace_prepare_share_sec', 0.0):.2f}s")
    print(f"    RPC ready/share gap:       {(phase_avgs.get('rpc_ready_wait_share_sec', 0.0) + phase_avgs.get('rpc_internal_share_sec', 0.0)):.2f}s")
    print(f"    Seed processing:           {phase_avgs.get('seed_processing_sec', 0.0):.2f}s")
    print(f"    Env build + reset:         {(phase_avgs.get('env_build_sec', 0.0) + phase_avgs.get('env_reset_sec', 0.0)):.2f}s")
    print(f"    Agent reset + calibrate:   {(phase_avgs.get('agent_reset_sec', 0.0) + phase_avgs.get('calibration_sec', 0.0)):.2f}s")
    print(f"    Rollout total:             {phase_avgs.get('rollout_total_sec', 0.0):.2f}s")
    print(f"    act() total:               {phase_avgs.get('agent_act_sec', 0.0):.2f}s")
    print(f"    env.step() total:          {phase_avgs.get('env_step_sec', 0.0):.2f}s")
    print(f"    Env cleanup:               {phase_avgs.get('env_cleanup_sec', 0.0):.2f}s")
    print(f"    Full-wall unaccounted:     {phase_avgs.get('full_wall_unaccounted_sec', 0.0):.2f}s")
    print(
        f"    act() latency:             avg={act_latency_summary['avg_ms']:.1f}ms "
        f"seed-p95-avg={act_latency_summary['p95_ms_seed_avg']:.1f}ms max={act_latency_summary['max_ms']:.1f}ms"
    )
    print(
        f"    env.step() latency:        avg={env_step_latency_summary['avg_ms']:.1f}ms "
        f"seed-p95-avg={env_step_latency_summary['p95_ms_seed_avg']:.1f}ms max={env_step_latency_summary['max_ms']:.1f}ms"
    )
    print(
        f"    Step total latency:        avg={step_latency_summary['avg_ms']:.1f}ms "
        f"seed-p95-avg={step_latency_summary['p95_ms_seed_avg']:.1f}ms max={step_latency_summary['max_ms']:.1f}ms"
    )
    print()

    print("  Memory breakdown (avg per seed):")
    print(
        f"    Validator RSS:             avg={memory_avgs.get('validator_avg_rss_mib', 0.0):.1f} MiB "
        f"peak={memory_avgs.get('validator_peak_rss_mib', 0.0):.1f} MiB "
        f"peak-delta={memory_avgs.get('validator_peak_delta_rss_mib', 0.0):.1f} MiB"
    )
    print(
        f"    Validator VMS:             avg={memory_avgs.get('validator_avg_vms_mib', 0.0):.1f} MiB "
        f"peak={memory_avgs.get('validator_peak_vms_mib', 0.0):.1f} MiB "
        f"peak-delta={memory_avgs.get('validator_peak_delta_vms_mib', 0.0):.1f} MiB"
    )
    container_limit_mib = memory_avgs.get("container_limit_mib", 0.0)
    container_limit_txt = (
        f" limit={container_limit_mib:.0f} MiB"
        if container_limit_mib > 0.0
        else ""
    )
    print(
        f"    Docker container mem:      avg={memory_avgs.get('container_avg_mem_mib', 0.0):.1f} MiB "
        f"peak={memory_avgs.get('container_peak_mem_mib', 0.0):.1f} MiB "
        f"peak-delta={memory_avgs.get('container_peak_delta_mem_mib', 0.0):.1f} MiB"
        f"{container_limit_txt}"
    )
    host_total_ram_mib = memory_avgs.get("host_total_ram_mib", 0.0)
    host_total_txt = (
        f" total={host_total_ram_mib:.0f} MiB"
        if host_total_ram_mib > 0.0
        else ""
    )
    print(
        f"    Host available RAM:        avg={memory_avgs.get('host_available_ram_avg_mib', 0.0):.1f} MiB "
        f"min={memory_avgs.get('host_available_ram_min_mib', 0.0):.1f} MiB "
        f"drop={memory_avgs.get('host_available_ram_drop_mib', 0.0):.1f} MiB"
        f"{host_total_txt}"
    )
    print()

    slowest_rows = sorted(all_rows, key=lambda row: float(row["wall_time"]), reverse=True)[:5]
    if slowest_rows:
        print("  Slowest seeds:")
        for row in slowest_rows:
            timing = row.get("timing_detail", {})
            print(
                f"    {row['group']:<18} seed={row['seed']} wall={row['wall_time']:.1f}s "
                f"shared={_safe_float(timing.get('container_shared_sec')):.1f}s "
                f"rollout={_safe_float(timing.get('rollout_breakdown', {}).get('rollout_total_sec')):.1f}s "
                f"act={_safe_float(timing.get('rollout_breakdown', {}).get('agent_act_sec')):.1f}s "
                f"step={_safe_float(timing.get('rollout_breakdown', {}).get('env_step_sec')):.1f}s"
            )
        print()

    highest_memory_rows = sorted(
        all_rows,
        key=lambda row: max(
            _safe_float(
                row.get("timing_detail", {})
                .get("memory_breakdown", {})
                .get("validator_process_rss", {})
                .get("peak_bytes")
            ),
            _safe_float(
                row.get("timing_detail", {})
                .get("memory_breakdown", {})
                .get("validator_process_vms", {})
                .get("peak_bytes")
            ),
            _safe_float(
                row.get("timing_detail", {})
                .get("memory_breakdown", {})
                .get("docker_container_memory", {})
                .get("peak_bytes")
            ),
        ),
        reverse=True,
    )[:5]
    highest_memory_peak_bytes = max(
        [
            max(
                _safe_float(
                    row.get("timing_detail", {})
                    .get("memory_breakdown", {})
                    .get("validator_process_rss", {})
                    .get("peak_bytes")
                ),
                _safe_float(
                    row.get("timing_detail", {})
                    .get("memory_breakdown", {})
                    .get("validator_process_vms", {})
                    .get("peak_bytes")
                ),
                _safe_float(
                    row.get("timing_detail", {})
                    .get("memory_breakdown", {})
                    .get("docker_container_memory", {})
                    .get("peak_bytes")
                ),
            )
            for row in highest_memory_rows
        ],
        default=0.0,
    )
    if highest_memory_rows and highest_memory_peak_bytes > 0.0:
        print("  Highest-memory seeds:")
        for row in highest_memory_rows:
            memory = row.get("timing_detail", {}).get("memory_breakdown", {})
            validator_peak_mib = _bytes_to_mib(
                memory.get("validator_process_rss", {}).get("peak_bytes")
            )
            validator_vms_peak_mib = _bytes_to_mib(
                memory.get("validator_process_vms", {}).get("peak_bytes")
            )
            container_peak_mib = _bytes_to_mib(
                memory.get("docker_container_memory", {}).get("peak_bytes")
            )
            host_available_min_mib = _bytes_to_mib(
                memory.get("system_available_memory", {}).get("min_bytes")
            )
            print(
                f"    {row['group']:<18} seed={row['seed']} "
                f"validator_peak={validator_peak_mib:.1f} MiB "
                f"validator_vms_peak={validator_vms_peak_mib:.1f} MiB "
                f"container_peak={container_peak_mib:.1f} MiB "
                f"host_avail_min={host_available_min_mib:.1f} MiB"
            )
        print()

    from math import floor

    from swarm.constants import CHALLENGE_TYPE_DISTRIBUTION

    def _allocate(total: int, weights: Dict[Any, float], keys: List[Any]) -> Dict[Any, int]:
        raw = {k: max(0.0, float(weights.get(k, 0.0))) * total for k in keys}
        base = {k: int(floor(v)) for k, v in raw.items()}
        rem = max(0, total - sum(base.values()))
        order = sorted(keys, key=lambda k: (raw[k] - base[k]), reverse=True)
        for i in range(rem):
            base[order[i % len(order)]] += 1
        return base

    type_counts = _allocate(1000, CHALLENGE_TYPE_DISTRIBUTION, [1, 2, 3, 4, 5, 6])
    dist = {
        "type1_city": type_counts[1],
        "type2_open": type_counts[2],
        "type3_mountain": type_counts[3],
        "type4_village": type_counts[4],
        "type5_warehouse": type_counts[5],
        "type6_forest": type_counts[6],
    }

    total_extrap_worker_sec = 0.0
    print("  Extrapolation to 1,000 seeds (using measured per-seed worker time):")
    for group, count in dist.items():
        rows = group_results.get(group, [])
        if rows:
            real_walls = [float(row["wall_time"]) for row in rows if float(row["wall_time"]) > 0.0]
            avg = sum(real_walls) / len(real_walls) if real_walls else avg_wall_per_seed
            source = "observed"
        else:
            avg = avg_wall_per_seed
            source = "fallback-global"
        group_worker_sec = count * avg
        total_extrap_worker_sec += group_worker_sec
        print(
            f"    {group:<18} {count:>4} seeds x {avg:.2f}s = {group_worker_sec:.0f}s "
            f"({source})"
        )

    print()
    est_wall_1000 = total_extrap_worker_sec / workers_used
    est_avg_seed_1000 = est_wall_1000 / 1000.0
    est_tput_1000 = (1000.0 / est_wall_1000 * 60.0) if est_wall_1000 > 0 else 0.0
    print(f"    Workers used:              {workers_used}")
    print(f"    Estimated worker-time:     {total_extrap_worker_sec:.0f}s")
    print(f"    Estimated wall-clock:      {est_wall_1000:.0f}s ({est_wall_1000 / 60.0:.1f} min)")
    print(f"    Estimated avg wall / seed: {est_avg_seed_1000:.2f}s")
    print(f"    Estimated throughput:      {est_tput_1000:.2f} seeds/min")
    print()

    group_phase_averages = {
        group: _phase_averages(rows) for group, rows in group_results.items()
    }
    batch_phase_avgs = _batch_phase_averages(batch_stats)
    timing_analysis = {
        "notes": [
            "full_wall_sec = seed_processing_sec + container_shared_sec + full_wall_unaccounted_sec",
            "container_shared_sec allocates non-seed batch costs equally across seeds in the same container",
            "rpc_internal_share_sec is rpc_batch_wait_sec minus summed per-seed processing time, allocated equally across seeds in the batch",
            "memory_overview_mib reports averages of per-seed validator RSS/VMS, container memory, and host available-memory samples",
        ],
        "overall_phase_averages_sec": phase_avgs,
        "group_phase_averages_sec": group_phase_averages,
        "batch_phase_averages_sec": batch_phase_avgs,
        "memory_overview_mib": memory_avgs,
        "latency_overview": {
            "act": act_latency_summary,
            "env_step": env_step_latency_summary,
            "step_total": step_latency_summary,
        },
        "slowest_seeds": [
            {
                "group": row["group"],
                "seed": row["seed"],
                "wall_time_sec": row["wall_time"],
                "processing_wall_time_sec": row["processing_wall_time"],
                "timing_detail": row["timing_detail"],
            }
            for row in slowest_rows
        ],
        "highest_memory_seeds": [
            {
                "group": row["group"],
                "seed": row["seed"],
                "validator_peak_mib": _bytes_to_mib(
                    row.get("timing_detail", {})
                    .get("memory_breakdown", {})
                    .get("validator_process_rss", {})
                    .get("peak_bytes")
                ),
                "validator_vms_peak_mib": _bytes_to_mib(
                    row.get("timing_detail", {})
                    .get("memory_breakdown", {})
                    .get("validator_process_vms", {})
                    .get("peak_bytes")
                ),
                "container_peak_mib": _bytes_to_mib(
                    row.get("timing_detail", {})
                    .get("memory_breakdown", {})
                    .get("docker_container_memory", {})
                    .get("peak_bytes")
                ),
                "host_available_min_mib": _bytes_to_mib(
                    row.get("timing_detail", {})
                    .get("memory_breakdown", {})
                    .get("system_available_memory", {})
                    .get("min_bytes")
                ),
                "timing_detail": row["timing_detail"],
            }
            for row in highest_memory_rows
        ],
        "per_seed": [
            {
                "group": row["group"],
                "seed": row["seed"],
                "score": row["score"],
                "success": row["success"],
                "sim_time_sec": row["sim_time"],
                "wall_time_sec": row["wall_time"],
                "processing_wall_time_sec": row["processing_wall_time"],
                "execution_status": row["execution_status"],
                "timing_detail": row["timing_detail"],
            }
            for row in all_rows
        ],
    }

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_workers": workers_used,
        "host_parallelism": host_parallelism,
        "total_seeds": len(task_meta),
        "wall_clock_sec": elapsed,
        "startup_overhead_sec": total_startup_overhead_sec,
        "seed_timings_note": "wall_time includes equal share of per-container startup and cleanup overhead",
        "run_metrics": {
            "success_count": success_count,
            "execution_success_count": execution_success_count,
            "execution_success_rate": (
                (float(execution_success_count) / float(total_seeds))
                if total_seeds
                else 0.0
            ),
            "avg_wall_per_seed_sec": avg_wall_per_seed,
            "median_wall_per_seed_sec": med_wall_per_seed,
            "p90_wall_per_seed_sec": p90_wall_per_seed,
            "avg_sim_per_seed_sec": avg_sim_per_seed,
            "total_seed_worker_sec": total_seed_worker_sec,
            "throughput_seeds_per_min": throughput_seeds_per_min,
            "throughput_per_worker_seeds_per_min": throughput_per_worker,
            "effective_parallelism": effective_parallelism,
            "worker_utilization": worker_utilization,
            "batch_count": len(batch_stats),
            "avg_seeds_per_container": avg_batch_size,
            "avg_startup_overhead_per_container_sec": avg_startup_overhead_sec,
            "total_startup_overhead_sec": total_startup_overhead_sec,
        },
        "execution_status_counts": dict(sorted(execution_status_counts.items())),
        "group_results": {g: rs for g, rs in group_results.items()},
        "batch_stats": [asdict(stat) for stat in batch_stats],
        "timing_analysis": timing_analysis,
        "extrapolation_1000_seeds": {
            "workers_used": workers_used,
            "total_seed_worker_sec": total_extrap_worker_sec,
            "estimated_wall_clock_sec": est_wall_1000,
            "estimated_avg_wall_per_seed_sec": est_avg_seed_1000,
            "estimated_throughput_seeds_per_min": est_tput_1000,
        },
    }
