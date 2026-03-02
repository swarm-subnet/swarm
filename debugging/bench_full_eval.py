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
    python3 bench_full_eval.py --model path/to/model.zip --json-out results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SWARM_PRIVATE_BENCHMARK_SECRET", "bench_test_key_2026")


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
        help="Number of seeds per challenge type/subtype (default: 3).",
    )
    parser.add_argument(
        "--workers", type=int, default=2,
        help="Number of parallel Docker workers (default: 2).",
    )
    parser.add_argument(
        "--json-out", type=Path, default=None,
        help="Path to write JSON results (default: <cwd>/bench_full_eval_results.json).",
    )
    parser.add_argument(
        "--log-out", type=Path, default=None,
        help="Path to write log file (default: no file logging).",
    )
    parser.add_argument(
        "--relax-timeouts", action="store_true", default=False,
        help="Override timing constants for slow machines (longer timeouts, more strikes).",
    )
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


def _find_seeds(seeds_per_group: int) -> Dict[str, List[int]]:
    """Find seeds covering all 4 type/subtype groups."""
    from swarm.constants import SIM_DT
    from swarm.validator.task_gen import random_task
    from swarm.core.mountain_generator import get_mountain_subtype

    groups: Dict[str, List[int]] = {
        "type1_city": [],
        "type2_open": [],
        "type3_mountain": [],
        "type3_village": [],
    }

    seed = random.randint(100000, 900000)
    max_search = seed + 500000
    while seed < max_search:
        task = random_task(sim_dt=SIM_DT, seed=seed)
        ct = task.challenge_type

        if ct == 1 and len(groups["type1_city"]) < seeds_per_group:
            groups["type1_city"].append(seed)
        elif ct == 2 and len(groups["type2_open"]) < seeds_per_group:
            groups["type2_open"].append(seed)
        elif ct == 3:
            sub = get_mountain_subtype(seed)
            if sub == 1 and len(groups["type3_mountain"]) < seeds_per_group:
                groups["type3_mountain"].append(seed)
            elif sub == 2 and len(groups["type3_village"]) < seeds_per_group:
                groups["type3_village"].append(seed)

        if all(len(v) >= seeds_per_group for v in groups.values()):
            break
        seed += 1

    return groups


async def _run_benchmark(
    model_path: Path,
    uid: int,
    type_seeds: Dict[str, List[int]],
    num_workers: int,
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
            task_meta.append({
                "group": group_name,
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
    eval_start = time.time()

    def _on_seed_done():
        seed_times.append(time.time())

    print(f"[{_ts()}] Running evaluation ({num_workers} workers, {len(all_tasks)} seeds)...")
    results = await evaluator.evaluate_seeds_parallel(
        tasks=all_tasks,
        uid=uid,
        model_path=model_path,
        num_workers=num_workers,
        on_seed_complete=_on_seed_done,
    )

    elapsed = time.time() - eval_start
    return task_meta, results, seed_times, elapsed, eval_start


def _print_results(
    task_meta: List[Dict[str, Any]],
    results: list,
    seed_times: List[float],
    elapsed: float,
    eval_start: float,
    num_workers: int,
) -> Dict[str, Any]:
    """Print results table and return JSON-serializable summary."""
    GROUP_ORDER = ["type1_city", "type2_open", "type3_mountain", "type3_village"]

    group_results: Dict[str, List[Dict[str, Any]]] = {}
    for i, meta in enumerate(task_meta):
        group = meta["group"]
        if group not in group_results:
            group_results[group] = []

        result = results[i] if i < len(results) else None
        score = float(result.score) if result else 0.0
        success = bool(result.success) if result else False
        sim_time = float(result.time_sec) if result else 0.0

        if i < len(seed_times):
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
    print(f"  {'Group':<18} {'Seed':>8} {'Score':>7} {'OK?':>5} {'SimT':>6} {'WallT':>7}")
    print(f"  {'-'*18} {'-'*8} {'-'*7} {'-'*5} {'-'*6} {'-'*7}")

    for group in GROUP_ORDER:
        if group not in group_results:
            continue
        for r in group_results[group]:
            ok = "Y" if r["success"] else "N"
            print(
                f"  {group:<18} {r['seed']:>8} {r['score']:>7.4f} {ok:>5} "
                f"{r['sim_time']:>5.1f}s {r['wall_time']:>6.1f}s"
            )
        walls = [r["wall_time"] for r in group_results[group]]
        scores = [r["score"] for r in group_results[group]]
        avg_w = sum(walls) / len(walls) if walls else 0
        avg_s = sum(scores) / len(scores) if scores else 0
        print(f"  {'  -> AVG':<18} {'':>8} {avg_s:>7.4f} {'':>5} {'':>6} {avg_w:>6.1f}s")
        print()

    # Overhead analysis
    real_deltas = []
    for i in range(len(seed_times)):
        dt = (seed_times[i] - seed_times[i - 1]) if i > 0 else (seed_times[0] - eval_start)
        if dt >= 0.5:
            real_deltas.append(dt)

    startup_overhead = 0.0
    if real_deltas:
        first_wall = real_deltas[0]
        steady = real_deltas[1:] if len(real_deltas) > 1 else []
        avg_steady = sum(steady) / len(steady) if steady else 0
        startup_overhead = max(0, first_wall - avg_steady) if avg_steady > 0 else first_wall
        print(f"  Container startup overhead: ~{startup_overhead:.0f}s")
        if steady:
            print(f"  Steady-state per seed:      ~{avg_steady:.1f}s")
    print(f"  Total wall-clock:           {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print()

    # Extrapolation
    DIST = {"type1_city": 500, "type2_open": 200, "type3_mountain": 225, "type3_village": 75}

    total_extrap = 0.0
    print(f"  Extrapolation to 1,000 seeds:")
    for group, count in DIST.items():
        if group in group_results and group_results[group]:
            real_walls = [r["wall_time"] for r in group_results[group] if not r.get("timeout_zero")]
            avg = sum(real_walls) / len(real_walls) if real_walls else 0
            group_total = count * avg
            total_extrap += group_total
            print(f"    {group:<18} {count:>4} seeds x {avg:.1f}s = {group_total:.0f}s")

    print()
    for w in [1, 2, 4]:
        print(f"    {w} worker(s): {total_extrap / w:.0f}s ({total_extrap / w / 60:.1f} min)")
    print()

    return {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_workers": num_workers,
        "total_seeds": len(task_meta),
        "wall_clock_sec": elapsed,
        "startup_overhead_sec": startup_overhead,
        "group_results": {g: rs for g, rs in group_results.items()},
        "extrapolation_1000_seeds": {
            f"{w}_workers_sec": total_extrap / w for w in [1, 2, 4]
        },
    }


def main() -> None:
    args = _parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    log_fh = None
    if args.log_out:
        log_fh = open(args.log_out, "w")
        sys.stdout = _Tee(sys.__stdout__, log_fh)
        sys.stderr = _Tee(sys.__stderr__, log_fh)

    overrides = {}
    if args.relax_timeouts:
        overrides = _apply_relaxed_overrides()

    print(f"[{_ts()}] === FULL EVALUATION BENCHMARK ===")
    print(f"[{_ts()}] Model: {model_path}")
    print(f"[{_ts()}] Workers: {args.workers}")
    if overrides:
        print(f"[{_ts()}] Relaxed timeouts: {overrides}")
    print()

    print(f"[{_ts()}] Finding {args.seeds_per_group} seeds per group...")
    type_seeds = _find_seeds(args.seeds_per_group)
    total_seeds = sum(len(v) for v in type_seeds.values())
    for group, seeds in type_seeds.items():
        print(f"  {group}: {seeds}")
    print(f"  Total: {total_seeds}")
    print()

    task_meta, results, seed_times, elapsed, eval_start = asyncio.run(
        _run_benchmark(model_path, args.uid, type_seeds, args.workers)
    )

    print(f"\n[{_ts()}] === RESULTS ===")
    summary = _print_results(task_meta, results, seed_times, elapsed, eval_start, args.workers)
    summary["model"] = str(model_path)

    json_out = args.json_out or Path.cwd() / "bench_full_eval_results.json"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[{_ts()}] JSON saved: {json_out}")

    if log_fh:
        log_fh.close()

    print(f"[{_ts()}] === BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    main()
