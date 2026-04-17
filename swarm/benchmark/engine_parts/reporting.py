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
from .dispatch import (
    _is_clean_execution_status,
    _resource_class_for_group,
    _resource_cost_dict_for_group,
    _resource_model_rows,
)


def _mean(values: List[float]) -> float:
    return (sum(values) / len(values)) if values else 0.0


def _pctl(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(value) for value in values)
    idx = max(0, int(round(float(fraction) * len(sorted_values) + 0.5)) - 1)
    return float(sorted_values[min(idx, len(sorted_values) - 1)])


def _rows_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    wall_times = [float(row["wall_time"]) for row in rows if float(row["wall_time"]) > 0.0]
    processing_walls = [
        float(row["processing_wall_time"])
        for row in rows
        if float(row["processing_wall_time"]) > 0.0
    ]
    sim_times = [float(row["sim_time"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    wall_per_sim = [
        float(row["wall_time"]) / float(row["sim_time"])
        for row in rows
        if float(row["sim_time"]) > 0.0
    ]
    success_count = sum(1 for row in rows if bool(row["success"]))
    execution_success_count = sum(1 for row in rows if bool(row["execution_ok"]))
    return {
        "seed_count": len(rows),
        "success_count": success_count,
        "execution_success_count": execution_success_count,
        "success_rate": (
            float(success_count) / float(len(rows))
            if rows
            else 0.0
        ),
        "execution_success_rate": (
            float(execution_success_count) / float(len(rows))
            if rows
            else 0.0
        ),
        "avg_wall_sec": _mean(wall_times),
        "median_wall_sec": statistics.median(wall_times) if wall_times else 0.0,
        "p90_wall_sec": _pctl(wall_times, 0.90),
        "max_wall_sec": max(wall_times) if wall_times else 0.0,
        "avg_processing_wall_sec": _mean(processing_walls),
        "avg_sim_sec": _mean(sim_times),
        "avg_score": _mean(scores),
        "avg_wall_per_sim_ratio": _mean(wall_per_sim),
    }


def _print_results(
    task_meta: List[Dict[str, Any]],
    results: list,
    seed_times: List[float],
    seed_wall_by_key: Dict[Tuple[int, int], deque[float]],
    seed_status_by_key: Dict[Tuple[int, int], deque[str]],
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

    group_results: Dict[str, List[Dict[str, Any]]] = {}
    for i, meta in enumerate(task_meta):
        group = str(meta["group"])
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
        wall = float(full_wall_by_key.get(seed_key, processing_wall))
        resource_class = _resource_class_for_group(group)

        group_results[group].append(
            {
                "group": group,
                "seed": int(meta["seed"]),
                "challenge_type": int(meta["challenge_type"]),
                "score": score,
                "success": success,
                "sim_time": sim_time,
                "wall_time": wall,
                "processing_wall_time": processing_wall,
                "execution_status": execution_status,
                "execution_ok": execution_ok,
                "timeout_zero": wall < 0.5 and i > 0,
                "resource_class": resource_class,
                "estimated_resource_cost": _resource_cost_dict_for_group(group),
                "wall_per_sim_ratio": (wall / sim_time) if sim_time > 0.0 else 0.0,
            }
        )

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
        walls = [float(row["wall_time"]) for row in group_results[group]]
        scores = [float(row["score"]) for row in group_results[group]]
        avg_w = _mean(walls)
        avg_s = _mean(scores)
        print(f"  {'  -> AVG':<18} {'':>8} {avg_s:>7.4f} {'':>5} {'':>6} {avg_w:>6.1f}s")
        print()

    all_rows = [
        row
        for group in group_order
        if group in group_results
        for row in group_results[group]
    ]
    all_seed_walls = [float(row["wall_time"]) for row in all_rows if float(row["wall_time"]) > 0.0]
    all_sim_times = [float(row["sim_time"]) for row in all_rows]
    success_count = sum(1 for row in all_rows if bool(row["success"]))
    execution_success_count = sum(1 for row in all_rows if bool(row["execution_ok"]))
    execution_status_counts = Counter(str(row["execution_status"]) for row in all_rows)
    total_seeds = len(all_rows)
    workers_used = max(1, int(num_workers))

    avg_wall_per_seed = _mean(all_seed_walls)
    med_wall_per_seed = statistics.median(all_seed_walls) if all_seed_walls else 0.0
    p90_wall_per_seed = _pctl(all_seed_walls, 0.90)
    avg_sim_per_seed = _mean(all_sim_times)

    throughput_seeds_per_min = (total_seeds / elapsed * 60.0) if elapsed > 0 else 0.0
    throughput_per_worker = throughput_seeds_per_min / workers_used
    total_seed_worker_sec = (
        sum(float(stat.elapsed_sec) for stat in batch_stats)
        if batch_stats
        else sum(all_seed_walls)
    )
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

    resource_cost_model = _resource_model_rows()
    resource_class_summary: Dict[str, Dict[str, Any]] = {}
    print("  Scheduler cost model (initial estimate):")
    for row in resource_cost_model:
        print(
            f"    {row['resource_class']:<6} groups={','.join(row['groups'])} "
            f"cpu={row['cpu_units']:.2f} ram={row['ram_mb']:.0f}MiB "
            f"heavy_tokens={row['heavy_tokens']}"
        )
    print()

    print("  Observed by resource class:")
    for resource_class in ("light", "medium", "heavy"):
        class_rows = [row for row in all_rows if str(row["resource_class"]) == resource_class]
        if not class_rows:
            continue
        summary = _rows_summary(class_rows)
        resource_class_summary[resource_class] = {
            "groups": [
                group_name
                for group_name in BENCH_GROUP_ORDER
                if _resource_class_for_group(group_name) == resource_class
            ],
            "estimated_cost": next(
                (
                    row
                    for row in resource_cost_model
                    if str(row["resource_class"]) == resource_class
                ),
                {},
            ),
            **summary,
        }
        print(
            f"    {resource_class:<6} seeds={summary['seed_count']:<3} "
            f"success={summary['success_count']}/{summary['seed_count']} "
            f"clean={summary['execution_success_count']}/{summary['seed_count']} "
            f"avg_wall={summary['avg_wall_sec']:.1f}s "
            f"p90={summary['p90_wall_sec']:.1f}s "
            f"max={summary['max_wall_sec']:.1f}s "
            f"avg_proc={summary['avg_processing_wall_sec']:.1f}s "
            f"avg_sim={summary['avg_sim_sec']:.1f}s"
        )
    print()

    expensive_seed_rows = sorted(
        all_rows,
        key=lambda row: (
            float(row["wall_time"]),
            float(row["processing_wall_time"]),
            float(row["sim_time"]),
        ),
        reverse=True,
    )[: min(10, len(all_rows))]
    if expensive_seed_rows:
        print("  Slowest seeds:")
        for row in expensive_seed_rows:
            print(
                f"    {row['group']:<18} seed={int(row['seed'])} "
                f"class={row['resource_class']} wall={float(row['wall_time']):.1f}s "
                f"proc={float(row['processing_wall_time']):.1f}s "
                f"sim={float(row['sim_time']):.1f}s "
                f"status={row['execution_status']} score={float(row['score']):.4f}"
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
            avg = _mean(real_walls) if real_walls else avg_wall_per_seed
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

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_workers": workers_used,
        "host_parallelism": host_parallelism,
        "total_seeds": len(task_meta),
        "wall_clock_sec": elapsed,
        "startup_overhead_sec": total_startup_overhead_sec,
        "seed_timings_note": "wall_time includes equal share of per-container startup overhead",
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
        "resource_cost_model_estimate": resource_cost_model,
        "resource_class_summary": resource_class_summary,
        "top_expensive_seeds": expensive_seed_rows,
        "group_results": {g: rs for g, rs in group_results.items()},
        "batch_stats": [asdict(stat) for stat in batch_stats],
        "extrapolation_1000_seeds": {
            "workers_used": workers_used,
            "total_seed_worker_sec": total_extrap_worker_sec,
            "estimated_wall_clock_sec": est_wall_1000,
            "estimated_avg_wall_per_seed_sec": est_avg_seed_1000,
            "estimated_throughput_seeds_per_min": est_tput_1000,
        },
    }
