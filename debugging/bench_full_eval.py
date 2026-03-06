#!/usr/bin/env python3
"""
Full Evaluation Benchmark
=========================
Measures real end-to-end wall-clock time per seed using the actual
Docker evaluation pipeline with a miner model zip.

Reports per-type timing, success rates, and extrapolates to 1,000 seeds.

Usage:
    python3 bench_full_eval.py --model path/to/model.zip
    python3 bench_full_eval.py --model path/to/model.zip --workers 4 --seeds-per-group 5
"""
from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import sys
import threading
import time
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SWARM_PRIVATE_BENCHMARK_SECRET", "bench_test_key_2026")

_HIDDEN = argparse.SUPPRESS
BENCH_GROUP_ORDER = [
    "type1_city",
    "type2_open",
    "type3_mountain",
    "type4_village",
    "type5_warehouse",
]
BENCH_GROUP_TO_TYPE = {
    "type1_city": 1,
    "type2_open": 2,
    "type3_mountain": 3,
    "type4_village": 4,
    "type5_warehouse": 5,
}
TYPE_TO_BENCH_GROUP = {v: k for k, v in BENCH_GROUP_TO_TYPE.items()}
MAP_TYPE_ALIASES = {
    "1": 1,
    "type1": 1,
    "type1_city": 1,
    "city": 1,
    "2": 2,
    "type2": 2,
    "type2_open": 2,
    "open": 2,
    "3": 3,
    "type3": 3,
    "type3_mountain": 3,
    "mountain": 3,
    "4": 4,
    "type4": 4,
    "type4_village": 4,
    "village": 4,
    "5": 5,
    "type5": 5,
    "type5_warehouse": 5,
    "warehouse": 5,
}


@dataclass
class _RunOptions:
    progress_every: int = 1
    heartbeat_sec: float = 30.0
    rpc_trace: bool = False
    rpc_trace_every: int = 250
    rpc_heartbeat_sec: float = 150.0
    serialize_pybullet: bool = True
    max_batch_timeout_sec: float = 900.0
    timeout_multiplier: float = 1.0
    extend_timeout_on_progress: bool = True
    timeout_extend_sec: float = 30.0
    timeout_progress_stale_sec: float = 3.0
    timeout_progress_min_sim_advance: float = 0.02
    max_seed_walltime_sec: float = 0.0
    default_log_out: Optional[str] = None


_RUN_PROFILES: Dict[str, _RunOptions] = {
    "standard": _RunOptions(),
    "debug": _RunOptions(
        heartbeat_sec=15.0,
        rpc_trace=True,
        rpc_trace_every=100,
        rpc_heartbeat_sec=100.0,
        serialize_pybullet=False,
        max_batch_timeout_sec=300.0,
        timeout_multiplier=1.0,
        extend_timeout_on_progress=True,
        timeout_extend_sec=30.0,
        timeout_progress_stale_sec=3.0,
        timeout_progress_min_sim_advance=0.02,
        max_seed_walltime_sec=1800.0,
        default_log_out="/tmp/bench_full_eval.log",
    ),
}


@contextmanager
def _temporary_env(overrides: Dict[str, Optional[str]]):
    previous = {k: os.environ.get(k) for k in overrides}
    try:
        for key, value in overrides.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full Docker evaluation benchmark across all challenge types.",
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to miner submission zip (e.g. model/UID_178.zip).",
    )
    parser.add_argument(
        "--uid", type=int, default=0,
        help="Miner UID (default: 0).",
    )
    parser.add_argument(
        "--seeds-per-group", type=int, default=3,
        help="Number of seeds per benchmark map group (default: 3).",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel Docker workers (default: 2).",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(_RUN_PROFILES.keys()),
        default="standard",
        help="Aggregated run profile for advanced settings (default: standard).",
    )
    parser.add_argument(
        "--log-out", type=Path, default=None,
        help="Path to write log file (default depends on profile).",
    )
    parser.add_argument(
        "--relax-timeouts", action="store_true", default=False,
        help="Override timing constants for slow machines (longer timeouts, more strikes).",
    )
    parser.add_argument(
        "--progress-every", type=int, default=1,
        help="Print progress every N completed seeds (default: 1).",
    )
    parser.add_argument(
        "--include-map-types",
        type=str,
        default=None,
        help="Comma-separated map types to include (ids or names): e.g. 3,5 or mountain,warehouse.",
    )
    parser.add_argument(
        "--exclude-map-types",
        type=str,
        default=None,
        help="Comma-separated map types to exclude (ids or names).",
    )
    # Advanced overrides (kept for compatibility, hidden from normal help output).
    parser.add_argument("--heartbeat-sec", type=float, default=None, help=_HIDDEN)
    parser.add_argument(
        "--rpc-trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=_HIDDEN,
    )
    parser.add_argument("--rpc-trace-every", type=int, default=None, help=_HIDDEN)
    parser.add_argument("--rpc-heartbeat-sec", type=float, default=None, help=_HIDDEN)
    parser.add_argument(
        "--serialize-pybullet",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=_HIDDEN,
    )
    parser.add_argument("--max-batch-timeout-sec", type=float, default=None, help=_HIDDEN)
    parser.add_argument("--timeout-multiplier", type=float, default=None, help=_HIDDEN)
    parser.add_argument(
        "--extend-timeout-on-progress",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=_HIDDEN,
    )
    parser.add_argument("--timeout-extend-sec", type=float, default=None, help=_HIDDEN)
    parser.add_argument("--timeout-progress-stale-sec", type=float, default=None, help=_HIDDEN)
    parser.add_argument("--timeout-progress-min-sim-advance", type=float, default=None, help=_HIDDEN)
    parser.add_argument("--max-seed-walltime-sec", type=float, default=None, help=_HIDDEN)
    return parser.parse_args()


class _Tee:
    """Write to multiple file objects simultaneously."""
    def __init__(self, *files):
        self.files = files

    def write(self, data):
        for f in self.files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def _parse_map_type_list(spec: Optional[str]) -> Optional[set[int]]:
    if spec is None:
        return None
    values: set[int] = set()
    for raw in spec.split(","):
        token = raw.strip().lower()
        if not token:
            continue
        if token not in MAP_TYPE_ALIASES:
            valid = ", ".join(["1", "2", "3", "4", "5", "city", "open", "mountain", "village", "warehouse"])
            raise ValueError(f"Invalid map type '{raw}'. Valid values: {valid}")
        values.add(MAP_TYPE_ALIASES[token])
    if not values:
        return None
    return values


def _resolve_selected_groups(
    include_types: Optional[set[int]],
    exclude_types: Optional[set[int]],
) -> List[str]:
    active = set(BENCH_GROUP_TO_TYPE.values()) if include_types is None else set(include_types)
    if exclude_types:
        active -= set(exclude_types)
    if not active:
        raise ValueError("Map-type filter removed all groups. Adjust include/exclude filters.")
    return [TYPE_TO_BENCH_GROUP[t] for t in sorted(active) if t in TYPE_TO_BENCH_GROUP]


def _resolve_run_options(args: argparse.Namespace) -> _RunOptions:
    opts = replace(_RUN_PROFILES[args.profile])
    opts.progress_every = max(1, int(args.progress_every))

    if args.heartbeat_sec is not None:
        opts.heartbeat_sec = max(0.0, float(args.heartbeat_sec))
    if args.rpc_trace is not None:
        opts.rpc_trace = bool(args.rpc_trace)
    if args.rpc_trace_every is not None:
        opts.rpc_trace_every = max(1, int(args.rpc_trace_every))
    if args.rpc_heartbeat_sec is not None:
        opts.rpc_heartbeat_sec = max(0.0, float(args.rpc_heartbeat_sec))
    if args.serialize_pybullet is not None:
        opts.serialize_pybullet = bool(args.serialize_pybullet)
    if args.max_batch_timeout_sec is not None:
        opts.max_batch_timeout_sec = float(args.max_batch_timeout_sec)
    if args.timeout_multiplier is not None:
        opts.timeout_multiplier = max(1.0, float(args.timeout_multiplier))
    if args.extend_timeout_on_progress is not None:
        opts.extend_timeout_on_progress = bool(args.extend_timeout_on_progress)
    if args.timeout_extend_sec is not None:
        opts.timeout_extend_sec = max(1.0, float(args.timeout_extend_sec))
    if args.timeout_progress_stale_sec is not None:
        opts.timeout_progress_stale_sec = max(0.5, float(args.timeout_progress_stale_sec))
    if args.timeout_progress_min_sim_advance is not None:
        opts.timeout_progress_min_sim_advance = max(0.0, float(args.timeout_progress_min_sim_advance))
    if args.max_seed_walltime_sec is not None:
        opts.max_seed_walltime_sec = max(0.0, float(args.max_seed_walltime_sec))

    return opts


def _apply_relaxed_overrides() -> Dict[str, Any]:
    """Patch swarm.constants for slow VPS benchmarking. Returns applied overrides."""
    import swarm.constants as _c

    overrides = {
        "MINER_COMPUTE_BUDGET_SEC": 3.0,
        "CALIBRATION_OVERHEAD_CAP_SEC": 1.0,
        "RPC_MAX_STRIKES_PER_SEED": 15,
        "RPC_STEP_TIMEOUT_SEC": 4.0,
        "DOCKER_WORKER_CPUS": "4",
        "GLOBAL_EVAL_BASE_SEC": 7200.0,
        "GLOBAL_EVAL_PER_SEED_SEC": 600.0,
        "GLOBAL_EVAL_CAP_SEC": 7200.0,
    }
    for attr, val in overrides.items():
        if hasattr(_c, attr):
            setattr(_c, attr, val)
    return overrides


def _infer_bench_group(challenge_type: int, seed: int) -> Optional[str]:
    """Map internal task challenge IDs to benchmark map groups (5-way)."""
    if challenge_type == 1:
        return "type1_city"
    if challenge_type == 2:
        return "type2_open"
    if challenge_type == 3:
        return "type3_mountain"
    if challenge_type == 4:
        return "type4_village"
    if challenge_type == 5:
        return "type5_warehouse"
    return None


def _find_seeds(seeds_per_group: int, selected_groups: Optional[List[str]] = None) -> Dict[str, List[int]]:
    """Find seeds covering selected benchmark map groups."""
    from swarm.constants import SIM_DT
    from swarm.validator.task_gen import random_task

    group_order = selected_groups if selected_groups is not None else BENCH_GROUP_ORDER
    groups: Dict[str, List[int]] = {g: [] for g in group_order}

    seed = random.randint(100000, 900000)
    max_search = seed + 500000
    while seed < max_search:
        task = random_task(sim_dt=SIM_DT, seed=seed)
        group = _infer_bench_group(int(task.challenge_type), seed)
        if group is not None and group in groups and len(groups[group]) < seeds_per_group:
            groups[group].append(seed)

        if all(len(v) >= seeds_per_group for v in groups.values()):
            break
        seed += 1

    missing = [g for g, seeds in groups.items() if len(seeds) < seeds_per_group]
    if missing:
        raise RuntimeError(
            "Could not find enough seeds for groups: "
            + ", ".join(f"{g} ({len(groups[g])}/{seeds_per_group})" for g in missing)
        )

    return groups


async def _run_benchmark(
    model_path: Path,
    uid: int,
    type_seeds: Dict[str, List[int]],
    num_workers: int,
    run_opts: _RunOptions,
) -> tuple:
    """Run Docker evaluation and return (task_meta, results, seed_times, total_elapsed)."""
    from swarm.constants import SIM_DT
    from swarm.validator.task_gen import random_task
    from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator

    all_tasks = []
    task_meta: List[Dict[str, Any]] = []
    for group_name, seeds in type_seeds.items():
        for s in seeds:
            task = random_task(sim_dt=SIM_DT, seed=s)
            all_tasks.append(task)
            bench_type = BENCH_GROUP_TO_TYPE.get(group_name, int(task.challenge_type))
            task_meta.append({
                "group": group_name,
                "bench_type": bench_type,
                "seed": s,
                "challenge_type": task.challenge_type,
                "horizon": task.horizon,
                "moving_platform": getattr(task, "moving_platform", False),
            })

    print(f"[{_ts()}] Initializing DockerSecureEvaluator...")
    evaluator = DockerSecureEvaluator()
    if not evaluator._base_ready:
        raise RuntimeError("Docker evaluator base image is not ready.")

    seed_times: List[float] = []
    seed_wall_by_key: Dict[Tuple[int, int], deque[float]] = {}
    total_seeds = len(all_tasks)
    progress_every = run_opts.progress_every
    heartbeat_sec = run_opts.heartbeat_sec

    eval_start = time.time()
    progress_lock = threading.Lock()
    done_count = 0
    last_done_at = eval_start
    stop_heartbeat = threading.Event()

    def _eta_minutes(elapsed_sec: float, done: int) -> float:
        if done <= 0:
            return float("inf")
        remaining = max(0, total_seeds - done)
        return (elapsed_sec / done) * remaining / 60.0

    def _on_seed_done(seed_meta: Optional[Dict[str, Any]] = None):
        nonlocal done_count, last_done_at
        now = time.time()
        with progress_lock:
            seed_times.append(now)
            if seed_meta is not None:
                try:
                    seed_key = (
                        int(seed_meta.get("map_seed")),
                        int(seed_meta.get("challenge_type")),
                    )
                    seed_wall = max(0.0, float(seed_meta.get("seed_wall_sec", 0.0)))
                    seed_wall_by_key.setdefault(seed_key, deque()).append(seed_wall)
                except Exception:
                    pass
            done_count += 1
            last_done_at = now
            done_snapshot = done_count

        if done_snapshot % progress_every == 0 or done_snapshot >= total_seeds:
            elapsed = now - eval_start
            eta_min = _eta_minutes(elapsed, done_snapshot)
            eta_txt = "--" if eta_min == float("inf") else f"{eta_min:.1f}m"
            print(
                f"[{_ts()}] Progress: {done_snapshot}/{total_seeds} seeds complete | "
                f"elapsed {elapsed/60.0:.1f}m | ETA {eta_txt}",
                flush=True,
            )

    def _heartbeat() -> None:
        try:
            if heartbeat_sec <= 0:
                return
            while not stop_heartbeat.wait(timeout=heartbeat_sec):
                now = time.time()
                with progress_lock:
                    done_snapshot = done_count
                    last_done_snapshot = last_done_at

                elapsed = now - eval_start
                idle_for = now - last_done_snapshot
                eta_min = _eta_minutes(elapsed, done_snapshot)
                eta_txt = "--" if eta_min == float("inf") else f"{eta_min:.1f}m"
                print(
                    f"[{_ts()}] Heartbeat: {done_snapshot}/{total_seeds} done | "
                    f"elapsed {elapsed/60.0:.1f}m | last completion {idle_for:.0f}s ago | ETA {eta_txt}",
                    flush=True,
                )
        except Exception as e:
            print(f"[{_ts()}] Heartbeat thread error: {type(e).__name__}: {e}", flush=True)

    timeout_mult = run_opts.timeout_multiplier
    extend_sec = run_opts.timeout_extend_sec
    stale_sec = run_opts.timeout_progress_stale_sec
    min_sim_advance = run_opts.timeout_progress_min_sim_advance
    max_total = run_opts.max_seed_walltime_sec

    env_overrides: Dict[str, Optional[str]] = {
        "SWARM_LOG_RPC_TRACE": "1" if run_opts.rpc_trace else None,
        "SWARM_LOG_RPC_TRACE_EVERY": str(max(1, int(run_opts.rpc_trace_every))) if run_opts.rpc_trace else None,
        "SWARM_LOG_RPC_HEARTBEAT_SEC": (
            f"{float(run_opts.rpc_heartbeat_sec):.3f}"
            if run_opts.rpc_trace and run_opts.rpc_heartbeat_sec > 0
            else None
        ),
        "SWARM_SERIALIZE_PYB": "1" if run_opts.serialize_pybullet else "0",
        "SWARM_BATCH_TIMEOUT_HARD_CAP_SEC": (
            f"{float(run_opts.max_batch_timeout_sec):.3f}"
            if run_opts.max_batch_timeout_sec > 0
            else None
        ),
        "SWARM_BATCH_TIMEOUT_MULT": f"{timeout_mult:.6f}",
        "SWARM_BATCH_TIMEOUT_EXTEND_ON_PROGRESS": "1" if run_opts.extend_timeout_on_progress else None,
        "SWARM_BATCH_TIMEOUT_EXTEND_SEC": (
            f"{extend_sec:.6f}" if run_opts.extend_timeout_on_progress else None
        ),
        "SWARM_BATCH_TIMEOUT_PROGRESS_STALE_SEC": (
            f"{stale_sec:.6f}" if run_opts.extend_timeout_on_progress else None
        ),
        "SWARM_BATCH_TIMEOUT_PROGRESS_MIN_SIM_ADVANCE": (
            f"{min_sim_advance:.6f}" if run_opts.extend_timeout_on_progress else None
        ),
        "SWARM_BATCH_TIMEOUT_MAX_TOTAL_SEC": (
            f"{max_total:.6f}" if run_opts.extend_timeout_on_progress else None
        ),
    }

    if run_opts.rpc_trace:
        print(
            f"[{_ts()}] RPC trace enabled (logging ping/reset plus every "
            f"{max(1, int(run_opts.rpc_trace_every))} act() steps; "
            f"phase heartbeat every {max(0.0, float(run_opts.rpc_heartbeat_sec)):.1f}s)"
        )
    if run_opts.serialize_pybullet:
        print(f"[{_ts()}] PyBullet serialization enabled (stable mode).")
    else:
        print(f"[{_ts()}] PyBullet serialization disabled (parallel env mode).")
    if run_opts.max_batch_timeout_sec > 0:
        print(f"[{_ts()}] Worker batch timeout hard cap: {float(run_opts.max_batch_timeout_sec):.1f}s")
    else:
        print(f"[{_ts()}] Worker batch timeout hard cap: disabled")
    if timeout_mult > 1.0:
        print(f"[{_ts()}] Worker timeout multiplier: x{timeout_mult:.2f}")
    if run_opts.extend_timeout_on_progress:
        print(
            f"[{_ts()}] Progress timeout extension: +{extend_sec:.1f}s "
            f"(stale<={stale_sec:.1f}s, min_sim_advance={min_sim_advance:.3f}s, "
            f"max_total={'unbounded' if max_total <= 0 else f'{max_total:.1f}s'})"
        )
    else:
        print(f"[{_ts()}] Progress timeout extension: disabled")

    effective_workers = max(1, int(num_workers))
    print(
        f"[{_ts()}] Running evaluation ({effective_workers} workers, {len(all_tasks)} seeds, "
        f"dynamic dispatch)..."
    )
    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        with _temporary_env(env_overrides):
            results: List[Optional[Any]] = [None] * len(all_tasks)
            pending_indices = deque(range(len(all_tasks)))
            pending_lock = asyncio.Lock()

            async def _claim_next_index() -> Optional[int]:
                async with pending_lock:
                    if not pending_indices:
                        return None
                    return pending_indices.popleft()

            async def _worker_loop(worker_slot: int) -> None:
                while True:
                    idx = await _claim_next_index()
                    if idx is None:
                        return

                    meta = task_meta[idx]
                    task = all_tasks[idx]
                    print(
                        f"[{_ts()}] Dispatch {idx + 1}/{len(all_tasks)} | "
                        f"worker={worker_slot} | seed={meta['seed']} | group={meta['group']}",
                        flush=True,
                    )
                    seed_start = time.time()
                    seed_results = await evaluator.evaluate_seeds_batch(
                        tasks=[task],
                        uid=uid,
                        model_path=model_path,
                        worker_id=worker_slot,
                        on_seed_complete=_on_seed_done,
                        task_offset=idx,
                        task_total=len(all_tasks),
                    )
                    if len(seed_results) != 1:
                        raise RuntimeError(
                            f"Worker {worker_slot}: unexpected result count "
                            f"{len(seed_results)} for single-task dispatch (seed {meta['seed']})."
                        )
                    results[idx] = seed_results[0]
                    print(
                        f"[{_ts()}] Worker {worker_slot} complete | seed={meta['seed']} "
                        f"in {time.time() - seed_start:.1f}s",
                        flush=True,
                    )

            worker_count = min(effective_workers, len(all_tasks))
            await asyncio.gather(*(_worker_loop(i) for i in range(worker_count)))

            if any(r is None for r in results):
                raise RuntimeError("Dynamic dispatch ended with missing seed result(s).")
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2.0)

    elapsed = time.time() - eval_start
    return task_meta, results, seed_times, seed_wall_by_key, elapsed, eval_start


def _print_results(
    task_meta: List[Dict[str, Any]],
    results: list,
    seed_times: List[float],
    seed_wall_by_key: Dict[Tuple[int, int], deque[float]],
    elapsed: float,
    eval_start: float,
    num_workers: int,
) -> Dict[str, Any]:
    """Print results table and return JSON-serializable summary."""
    GROUP_ORDER = BENCH_GROUP_ORDER
    seed_wall_queues: Dict[Tuple[int, int], deque[float]] = {
        key: deque(values) for key, values in seed_wall_by_key.items()
    }

    group_results: Dict[str, List[Dict[str, Any]]] = {}
    for i, meta in enumerate(task_meta):
        group = meta["group"]
        if group not in group_results:
            group_results[group] = []

        result = results[i] if i < len(results) else None
        score = float(result.score) if result else 0.0
        success = bool(result.success) if result else False
        sim_time = float(result.time_sec) if result else 0.0

        seed_key = (int(meta["seed"]), int(meta["challenge_type"]))
        wall_q = seed_wall_queues.get(seed_key)
        if wall_q and len(wall_q) > 0:
            wall = float(wall_q.popleft())
        elif i < len(seed_times):
            # Fallback for compatibility if callback metadata was not provided.
            wall = (seed_times[i] - seed_times[i - 1]) if i > 0 else (seed_times[0] - eval_start)
        else:
            wall = 0.0

        is_timeout = wall < 0.5 and i > 0
        group_results[group].append({
            "seed": meta["seed"],
            "score": score,
            "success": success,
            "sim_time": sim_time,
            "wall_time": wall,
            "timeout_zero": is_timeout,
        })

    print()
    print(f"  {'Group':<18} {'Seed':>8} {'Score':>7} {'OK?':>5} {'SimT':>7} {'WallT':>7}")
    print(f"  {'-'*18} {'-'*8} {'-'*7} {'-'*5} {'-'*7} {'-'*7}")

    for group in GROUP_ORDER:
        if group not in group_results:
            continue
        for r in group_results[group]:
            ok = "Y" if r["success"] else "N"
            print(
                f"  {group:<18} {r['seed']:>8} {r['score']:>7.4f} {ok:>5} "
                f"{r['sim_time']:>6.2f}s {r['wall_time']:>6.1f}s"
            )
        walls = [r["wall_time"] for r in group_results[group]]
        scores = [r["score"] for r in group_results[group]]
        avg_w = sum(walls) / len(walls) if walls else 0
        avg_s = sum(scores) / len(scores) if scores else 0
        print(f"  {'  -> AVG':<18} {'':>8} {avg_s:>7.4f} {'':>5} {'':>6} {avg_w:>6.1f}s")
        print()

    all_rows = [row for group in GROUP_ORDER if group in group_results for row in group_results[group]]
    all_seed_walls = [float(r["wall_time"]) for r in all_rows if float(r["wall_time"]) > 0.0]
    all_sim_times = [float(r["sim_time"]) for r in all_rows]
    success_count = sum(1 for r in all_rows if bool(r["success"]))
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
    total_seed_worker_sec = sum(all_seed_walls)
    effective_parallelism = (total_seed_worker_sec / elapsed) if elapsed > 0 else 0.0
    worker_utilization = min(1.0, effective_parallelism / workers_used) if workers_used > 0 else 0.0

    print("  Run summary:")
    print(f"    Seeds evaluated:           {total_seeds}")
    print(
        f"    Success rate:              {success_count}/{total_seeds} "
        f"({(100.0 * success_count / total_seeds) if total_seeds else 0.0:.1f}%)"
    )
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
    print()

    # Optional startup overhead estimate for single-worker runs.
    startup_overhead = 0.0
    real_deltas: List[float] = []
    if workers_used == 1:
        for i in range(len(seed_times)):
            dt = (seed_times[i] - seed_times[i - 1]) if i > 0 else (seed_times[0] - eval_start)
            if dt >= 0.5:
                real_deltas.append(dt)
    if real_deltas:
        first_wall = real_deltas[0]
        steady = real_deltas[1:] if len(real_deltas) > 1 else []
        avg_steady = sum(steady) / len(steady) if steady else 0
        startup_overhead = max(0, first_wall - avg_steady) if avg_steady > 0 else first_wall
        print(f"  Container startup overhead: ~{startup_overhead:.0f}s")
        if steady:
            print(f"  Steady-state per seed:      ~{avg_steady:.1f}s")
        print()

    # Extrapolation (derived from challenge type distribution).
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

    type_counts = _allocate(1000, CHALLENGE_TYPE_DISTRIBUTION, [1, 2, 3, 4, 5])
    DIST = {
        "type1_city": type_counts[1],
        "type2_open": type_counts[2],
        "type3_mountain": type_counts[3],
        "type4_village": type_counts[4],
        "type5_warehouse": type_counts[5],
    }

    total_extrap_worker_sec = 0.0
    print("  Extrapolation to 1,000 seeds (using measured per-seed worker time):")
    for group, count in DIST.items():
        rows = group_results.get(group, [])
        if rows:
            real_walls = [float(r["wall_time"]) for r in rows if float(r["wall_time"]) > 0.0]
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

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_workers": workers_used,
        "total_seeds": len(task_meta),
        "wall_clock_sec": elapsed,
        "startup_overhead_sec": startup_overhead,
        "run_metrics": {
            "success_count": success_count,
            "avg_wall_per_seed_sec": avg_wall_per_seed,
            "median_wall_per_seed_sec": med_wall_per_seed,
            "p90_wall_per_seed_sec": p90_wall_per_seed,
            "avg_sim_per_seed_sec": avg_sim_per_seed,
            "total_seed_worker_sec": total_seed_worker_sec,
            "throughput_seeds_per_min": throughput_seeds_per_min,
            "throughput_per_worker_seeds_per_min": throughput_per_worker,
            "effective_parallelism": effective_parallelism,
            "worker_utilization": worker_utilization,
        },
        "group_results": {g: rs for g, rs in group_results.items()},
        "extrapolation_1000_seeds": {
            "workers_used": workers_used,
            "total_seed_worker_sec": total_extrap_worker_sec,
            "estimated_wall_clock_sec": est_wall_1000,
            "estimated_avg_wall_per_seed_sec": est_avg_seed_1000,
            "estimated_throughput_seeds_per_min": est_tput_1000,
        },
    }


def main() -> None:
    args = _parse_args()
    run_opts = _resolve_run_options(args)
    include_types = _parse_map_type_list(args.include_map_types)
    exclude_types = _parse_map_type_list(args.exclude_map_types)
    selected_groups = _resolve_selected_groups(include_types, exclude_types)

    requested_workers = max(1, int(args.workers))
    effective_workers = requested_workers
    # In no-serialize mode, high worker counts are unstable in practice and can hard-freeze
    # inside PyBullet env calls. Cap to a safer concurrency level for benchmark reliability.
    if not run_opts.serialize_pybullet and requested_workers > 2:
        effective_workers = 2

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    effective_log_out = args.log_out
    if effective_log_out is None and run_opts.default_log_out:
        effective_log_out = Path(run_opts.default_log_out)

    log_fh = None
    if effective_log_out:
        log_fh = open(effective_log_out, "w")
        sys.stdout = _Tee(sys.__stdout__, log_fh)
        sys.stderr = _Tee(sys.__stderr__, log_fh)

    overrides = {}
    if args.relax_timeouts:
        overrides = _apply_relaxed_overrides()

    print(f"[{_ts()}] === FULL EVALUATION BENCHMARK ===")
    print(f"[{_ts()}] Model: {model_path}")
    print(f"[{_ts()}] Workers requested: {requested_workers}")
    if effective_workers != requested_workers:
        print(
            f"[{_ts()}] Workers effective:  {effective_workers} "
            f"(capped for no-serialize stability)"
        )
    else:
        print(f"[{_ts()}] Workers effective:  {effective_workers}")
    print(f"[{_ts()}] Profile: {args.profile}")
    if overrides:
        print(f"[{_ts()}] Relaxed timeouts: {overrides}")
    print(f"[{_ts()}] Map types selected: {', '.join(selected_groups)}")
    if effective_log_out:
        print(f"[{_ts()}] Log file: {effective_log_out}")
    print()

    print(f"[{_ts()}] Finding {args.seeds_per_group} seeds per group...")
    type_seeds = _find_seeds(args.seeds_per_group, selected_groups=selected_groups)
    total_seeds = sum(len(v) for v in type_seeds.values())
    for group, seeds in type_seeds.items():
        print(f"  {group}: {seeds}")
    print(f"  Total: {total_seeds}")
    print()

    task_meta, results, seed_times, seed_wall_by_key, elapsed, eval_start = asyncio.run(
        _run_benchmark(
            model_path,
            args.uid,
            type_seeds,
            effective_workers,
            run_opts=run_opts,
        )
    )

    print(f"\n[{_ts()}] === RESULTS ===")
    _print_results(
        task_meta,
        results,
        seed_times,
        seed_wall_by_key,
        elapsed,
        eval_start,
        effective_workers,
    )

    print(f"[{_ts()}] === BENCHMARK COMPLETE ===")
    if log_fh:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_fh.close()


if __name__ == "__main__":
    main()
