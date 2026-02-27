#!/usr/bin/env python3
"""Validator-faithful local model test using Docker RPC evaluation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from swarm.constants import SIM_DT
from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator
from swarm.validator.task_gen import random_task

LOGGER = logging.getLogger("swarm.rl.test")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run validator-faithful Docker + Cap'n Proto RPC evaluation across "
            "multiple seeds and report aggregate stats."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model/UID_178.zip"),
        help="Path to miner submission zip (default: model/UID_178.zip)",
    )
    parser.add_argument(
        "--uid",
        type=int,
        default=None,
        help="Miner UID (default: inferred from filename UID_<n>.zip, else 0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Seed control: with --num-seeds=1 it is the exact map seed; with "
            "--num-seeds>1 it is used as deterministic seed source for sampling."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of seeds to evaluate (default: 10).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel Docker workers per wave. Multi-seed runs are split into "
            "waves with at most one seed per worker to avoid batch timeouts."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print final result in JSON format.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console log level.",
    )
    parser.add_argument(
        "--show-actions",
        action="store_true",
        help="Print per-tick RPC action vectors returned by the miner.",
    )
    parser.add_argument(
        "--verbose-eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable evaluator progress diagnostics (default: enabled).",
    )
    parser.add_argument(
        "--action-log-every",
        type=int,
        default=1,
        help="When --show-actions is enabled, log every N ticks (default: 1).",
    )
    parser.add_argument(
        "--progress-log-every",
        type=int,
        default=25,
        help="When verbose eval is enabled, log progress every N simulation ticks (default: 25).",
    )
    parser.add_argument(
        "--timeout-multiplier",
        type=float,
        default=1.0,
        help=(
            "Multiply worker batch timeout budget for local runs (default: 1.0). "
            "Use >1.0 on slow machines to reduce false timeouts."
        ),
    )
    parser.add_argument(
        "--extend-timeout-on-progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If enabled, extend batch timeout when evaluator progress is active. "
            "Useful for slow local machines (default: enabled)."
        ),
    )
    parser.add_argument(
        "--timeout-extend-sec",
        type=float,
        default=30.0,
        help="Seconds to add each time active progress extension triggers (default: 30).",
    )
    parser.add_argument(
        "--timeout-progress-stale-sec",
        type=float,
        default=3.0,
        help="Do not extend timeout if progress signal is older than this many seconds (default: 3).",
    )
    parser.add_argument(
        "--timeout-progress-min-sim-advance",
        type=float,
        default=0.02,
        help=(
            "Minimum simulation-time advance (seconds) between extensions to count as active progress "
            "(default: 0.02)."
        ),
    )
    parser.add_argument(
        "--max-seed-walltime-sec",
        type=float,
        default=0.0,
        help=(
            "Optional hard cap for one seed wall time (seconds). "
            "0 disables cap and allows extensions while progress continues."
        ),
    )
    return parser.parse_args()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )


def _resolve_uid(model_path: Path, uid_arg: int | None) -> int:
    if uid_arg is not None:
        return uid_arg

    match = re.search(r"UID_(\d+)\.zip$", model_path.name)
    if match:
        return int(match.group(1))
    return 0


def _task_meta(task: Any) -> Dict[str, Any]:
    return {
        "map_seed": int(task.map_seed),
        "challenge_type": int(task.challenge_type),
        "moving_platform": bool(getattr(task, "moving_platform", False)),
        "start": [float(v) for v in task.start],
        "goal": [float(v) for v in task.goal],
        "horizon": float(task.horizon),
    }


def _build_seed_list(seed_arg: int | None, num_seeds: int) -> List[int]:
    count = max(1, int(num_seeds))
    if count == 1 and seed_arg is not None:
        return [int(seed_arg)]

    rng = random.Random(seed_arg) if seed_arg is not None else random.SystemRandom()
    seeds: List[int] = []
    seen: set[int] = set()

    if seed_arg is not None:
        fixed_seed = int(seed_arg)
        seeds.append(fixed_seed)
        seen.add(fixed_seed)

    while len(seeds) < count:
        candidate = int(rng.randint(1, 2_147_483_647))
        if candidate in seen:
            continue
        seeds.append(candidate)
        seen.add(candidate)

    return seeds


async def _evaluate_seeds(
    *,
    model_path: Path,
    uid: int,
    seeds: List[int],
    workers: int,
    show_actions: bool,
    action_log_every: int,
    verbose_eval: bool,
    progress_log_every: int,
    timeout_multiplier: float,
    extend_timeout_on_progress: bool,
    timeout_extend_sec: float,
    timeout_progress_stale_sec: float,
    timeout_progress_min_sim_advance: float,
    max_seed_walltime_sec: float,
) -> Tuple[List[Dict[str, Any]], List[Any], float]:
    tasks = []
    task_infos: List[Dict[str, Any]] = []
    LOGGER.info("Generating %d task(s)", len(seeds))
    for i, map_seed in enumerate(seeds):
        task = random_task(sim_dt=SIM_DT, seed=map_seed)
        task_info = _task_meta(task)
        tasks.append(task)
        task_infos.append(task_info)
        LOGGER.info(
            "Task %d/%d | map_seed=%d challenge_type=%d moving_platform=%s",
            i + 1,
            len(seeds),
            task_info["map_seed"],
            task_info["challenge_type"],
            task_info["moving_platform"],
        )

    LOGGER.info("Initializing DockerSecureEvaluator")
    previous_log_actions = os.environ.get("SWARM_LOG_ACTIONS")
    previous_log_every = os.environ.get("SWARM_LOG_ACTION_EVERY")
    previous_eval_progress = os.environ.get("SWARM_LOG_EVAL_PROGRESS")
    previous_progress_every = os.environ.get("SWARM_LOG_PROGRESS_EVERY")
    previous_timeout_multiplier = os.environ.get("SWARM_BATCH_TIMEOUT_MULT")
    previous_timeout_extend_on_progress = os.environ.get("SWARM_BATCH_TIMEOUT_EXTEND_ON_PROGRESS")
    previous_timeout_extend_sec = os.environ.get("SWARM_BATCH_TIMEOUT_EXTEND_SEC")
    previous_timeout_progress_stale_sec = os.environ.get("SWARM_BATCH_TIMEOUT_PROGRESS_STALE_SEC")
    previous_timeout_progress_min_sim_advance = os.environ.get("SWARM_BATCH_TIMEOUT_PROGRESS_MIN_SIM_ADVANCE")
    previous_timeout_max_total_sec = os.environ.get("SWARM_BATCH_TIMEOUT_MAX_TOTAL_SEC")
    try:
        if show_actions:
            os.environ["SWARM_LOG_ACTIONS"] = "1"
            os.environ["SWARM_LOG_ACTION_EVERY"] = str(max(1, action_log_every))
            LOGGER.info(
                "Action trace enabled: logging miner RPC actions every %d tick(s)",
                max(1, action_log_every),
            )
        else:
            os.environ.pop("SWARM_LOG_ACTIONS", None)
            os.environ.pop("SWARM_LOG_ACTION_EVERY", None)

        if verbose_eval:
            os.environ["SWARM_LOG_EVAL_PROGRESS"] = "1"
            os.environ["SWARM_LOG_PROGRESS_EVERY"] = str(max(1, progress_log_every))
            LOGGER.info(
                "Verbose evaluator diagnostics enabled: progress every %d tick(s)",
                max(1, progress_log_every),
            )
        else:
            os.environ.pop("SWARM_LOG_EVAL_PROGRESS", None)
            os.environ.pop("SWARM_LOG_PROGRESS_EVERY", None)

        timeout_mult = max(1.0, float(timeout_multiplier))
        os.environ["SWARM_BATCH_TIMEOUT_MULT"] = f"{timeout_mult:.6f}"
        if timeout_mult > 1.0:
            LOGGER.info("Local timeout multiplier enabled: x%.2f", timeout_mult)

        if extend_timeout_on_progress:
            extend_sec = max(1.0, float(timeout_extend_sec))
            stale_sec = max(0.5, float(timeout_progress_stale_sec))
            min_sim_advance = max(0.0, float(timeout_progress_min_sim_advance))
            max_total = max(0.0, float(max_seed_walltime_sec))
            os.environ["SWARM_BATCH_TIMEOUT_EXTEND_ON_PROGRESS"] = "1"
            os.environ["SWARM_BATCH_TIMEOUT_EXTEND_SEC"] = f"{extend_sec:.6f}"
            os.environ["SWARM_BATCH_TIMEOUT_PROGRESS_STALE_SEC"] = f"{stale_sec:.6f}"
            os.environ["SWARM_BATCH_TIMEOUT_PROGRESS_MIN_SIM_ADVANCE"] = f"{min_sim_advance:.6f}"
            os.environ["SWARM_BATCH_TIMEOUT_MAX_TOTAL_SEC"] = f"{max_total:.6f}"
            LOGGER.info(
                "Progress timeout extension enabled: +%.1fs when stale<=%.1fs and sim advances>=%.3fs "
                "(max_seed_walltime=%s)",
                extend_sec,
                stale_sec,
                min_sim_advance,
                "unbounded" if max_total <= 0 else f"{max_total:.1f}s",
            )
        else:
            os.environ.pop("SWARM_BATCH_TIMEOUT_EXTEND_ON_PROGRESS", None)
            os.environ.pop("SWARM_BATCH_TIMEOUT_EXTEND_SEC", None)
            os.environ.pop("SWARM_BATCH_TIMEOUT_PROGRESS_STALE_SEC", None)
            os.environ.pop("SWARM_BATCH_TIMEOUT_PROGRESS_MIN_SIM_ADVANCE", None)
            os.environ.pop("SWARM_BATCH_TIMEOUT_MAX_TOTAL_SEC", None)

        evaluator = DockerSecureEvaluator()
        if not evaluator._base_ready:
            raise RuntimeError("Docker evaluator base image is not ready.")

        LOGGER.info(
            "Running validator-style evaluation via RPC (uid=%d, workers=%d, seeds=%d)",
            uid,
            max(1, workers),
            len(tasks),
        )
        t0 = time.time()
        effective_workers = max(1, workers)
        wave_size = effective_workers  # one seed per worker per wave
        results = []
        total_waves = (len(tasks) + wave_size - 1) // wave_size

        for wave_idx, start_idx in enumerate(range(0, len(tasks), wave_size), start=1):
            end_idx = min(start_idx + wave_size, len(tasks))
            wave_tasks = tasks[start_idx:end_idx]
            wave_infos = task_infos[start_idx:end_idx]
            wave_workers = min(effective_workers, len(wave_tasks))
            wave_seed_str = ", ".join(str(info["map_seed"]) for info in wave_infos)

            LOGGER.info(
                "Wave %d/%d | seeds=%s | workers=%d",
                wave_idx,
                total_waves,
                wave_seed_str,
                wave_workers,
            )

            wave_results = await evaluator.evaluate_seeds_parallel(
                tasks=wave_tasks,
                uid=uid,
                model_path=model_path,
                num_workers=wave_workers,
            )
            if len(wave_results) != len(wave_tasks):
                raise RuntimeError(
                    f"Wave {wave_idx}: unexpected result count "
                    f"{len(wave_results)} for {len(wave_tasks)} tasks."
                )
            results.extend(wave_results)

        elapsed = time.time() - t0
    finally:
        if previous_log_actions is None:
            os.environ.pop("SWARM_LOG_ACTIONS", None)
        else:
            os.environ["SWARM_LOG_ACTIONS"] = previous_log_actions
        if previous_log_every is None:
            os.environ.pop("SWARM_LOG_ACTION_EVERY", None)
        else:
            os.environ["SWARM_LOG_ACTION_EVERY"] = previous_log_every
        if previous_eval_progress is None:
            os.environ.pop("SWARM_LOG_EVAL_PROGRESS", None)
        else:
            os.environ["SWARM_LOG_EVAL_PROGRESS"] = previous_eval_progress
        if previous_progress_every is None:
            os.environ.pop("SWARM_LOG_PROGRESS_EVERY", None)
        else:
            os.environ["SWARM_LOG_PROGRESS_EVERY"] = previous_progress_every
        if previous_timeout_multiplier is None:
            os.environ.pop("SWARM_BATCH_TIMEOUT_MULT", None)
        else:
            os.environ["SWARM_BATCH_TIMEOUT_MULT"] = previous_timeout_multiplier
        if previous_timeout_extend_on_progress is None:
            os.environ.pop("SWARM_BATCH_TIMEOUT_EXTEND_ON_PROGRESS", None)
        else:
            os.environ["SWARM_BATCH_TIMEOUT_EXTEND_ON_PROGRESS"] = previous_timeout_extend_on_progress
        if previous_timeout_extend_sec is None:
            os.environ.pop("SWARM_BATCH_TIMEOUT_EXTEND_SEC", None)
        else:
            os.environ["SWARM_BATCH_TIMEOUT_EXTEND_SEC"] = previous_timeout_extend_sec
        if previous_timeout_progress_stale_sec is None:
            os.environ.pop("SWARM_BATCH_TIMEOUT_PROGRESS_STALE_SEC", None)
        else:
            os.environ["SWARM_BATCH_TIMEOUT_PROGRESS_STALE_SEC"] = previous_timeout_progress_stale_sec
        if previous_timeout_progress_min_sim_advance is None:
            os.environ.pop("SWARM_BATCH_TIMEOUT_PROGRESS_MIN_SIM_ADVANCE", None)
        else:
            os.environ["SWARM_BATCH_TIMEOUT_PROGRESS_MIN_SIM_ADVANCE"] = previous_timeout_progress_min_sim_advance
        if previous_timeout_max_total_sec is None:
            os.environ.pop("SWARM_BATCH_TIMEOUT_MAX_TOTAL_SEC", None)
        else:
            os.environ["SWARM_BATCH_TIMEOUT_MAX_TOTAL_SEC"] = previous_timeout_max_total_sec

    if not results:
        raise RuntimeError("No validation result returned by evaluator.")

    if len(results) != len(task_infos):
        raise RuntimeError(
            f"Unexpected result count: got {len(results)} for {len(task_infos)} tasks."
        )

    return task_infos, results, elapsed


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    model_path = args.model.resolve()
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Model zip not found: {model_path}")

    uid = _resolve_uid(model_path, args.uid)
    seeds = _build_seed_list(args.seed, args.num_seeds)

    LOGGER.info("Model: %s", model_path)
    LOGGER.info("UID: %d", uid)
    if args.seed is None:
        LOGGER.info("Seed source: random (no --seed provided)")
    else:
        LOGGER.info("Seed source: deterministic (--seed=%d)", int(args.seed))
    LOGGER.info("Seeds (%d): %s", len(seeds), ", ".join(str(s) for s in seeds))

    task_infos, results, elapsed = asyncio.run(
        _evaluate_seeds(
            model_path=model_path,
            uid=uid,
            seeds=seeds,
            workers=args.workers,
            show_actions=args.show_actions,
            action_log_every=args.action_log_every,
            verbose_eval=args.verbose_eval,
            progress_log_every=args.progress_log_every,
            timeout_multiplier=args.timeout_multiplier,
            extend_timeout_on_progress=args.extend_timeout_on_progress,
            timeout_extend_sec=args.timeout_extend_sec,
            timeout_progress_stale_sec=args.timeout_progress_stale_sec,
            timeout_progress_min_sim_advance=args.timeout_progress_min_sim_advance,
            max_seed_walltime_sec=args.max_seed_walltime_sec,
        )
    )

    per_seed = []
    for i, (seed, task_info, result) in enumerate(zip(seeds, task_infos, results)):
        per_seed.append(
            {
                "index": i,
                "seed": int(seed),
                "task": task_info,
                "result": {
                    "success": bool(result.success),
                    "time_sec": float(result.time_sec),
                    "score": float(result.score),
                },
            }
        )

    scores = [entry["result"]["score"] for entry in per_seed]
    times = [entry["result"]["time_sec"] for entry in per_seed]
    success_count = sum(1 for entry in per_seed if entry["result"]["success"])
    hard_fail_count = sum(
        1
        for entry in per_seed
        if (not entry["result"]["success"])
        and abs(entry["result"]["time_sec"]) < 1e-9
        and abs(entry["result"]["score"]) < 1e-12
    )
    n = len(per_seed)
    aggregate = {
        "num_seeds": n,
        "success_count": int(success_count),
        "success_rate": float(success_count / n if n else 0.0),
        "hard_fail_count": int(hard_fail_count),
        "score_mean": float(statistics.fmean(scores) if scores else 0.0),
        "score_median": float(statistics.median(scores) if scores else 0.0),
        "score_std": float(statistics.pstdev(scores) if len(scores) > 1 else 0.0),
        "score_min": float(min(scores) if scores else 0.0),
        "score_max": float(max(scores) if scores else 0.0),
        "time_mean_sec": float(statistics.fmean(times) if times else 0.0),
        "time_median_sec": float(statistics.median(times) if times else 0.0),
        "time_min_sec": float(min(times) if times else 0.0),
        "time_max_sec": float(max(times) if times else 0.0),
        "wall_time_sec": float(elapsed),
    }

    type_name = {1: "city", 2: "open", 3: "mountain"}
    by_challenge_type: Dict[str, Dict[str, Any]] = {}
    for challenge_type in sorted({entry["task"]["challenge_type"] for entry in per_seed}):
        rows = [entry for entry in per_seed if entry["task"]["challenge_type"] == challenge_type]
        row_scores = [row["result"]["score"] for row in rows]
        row_times = [row["result"]["time_sec"] for row in rows]
        row_success = sum(1 for row in rows if row["result"]["success"])
        row_hard_fail = sum(
            1
            for row in rows
            if (not row["result"]["success"])
            and abs(row["result"]["time_sec"]) < 1e-9
            and abs(row["result"]["score"]) < 1e-12
        )
        key = f"{challenge_type}:{type_name.get(challenge_type, 'unknown')}"
        by_challenge_type[key] = {
            "count": len(rows),
            "success_count": int(row_success),
            "success_rate": float(row_success / len(rows) if rows else 0.0),
            "hard_fail_count": int(row_hard_fail),
            "score_mean": float(statistics.fmean(row_scores) if row_scores else 0.0),
            "score_median": float(statistics.median(row_scores) if row_scores else 0.0),
            "time_mean_sec": float(statistics.fmean(row_times) if row_times else 0.0),
            "time_median_sec": float(statistics.median(row_times) if row_times else 0.0),
            "time_max_sec": float(max(row_times) if row_times else 0.0),
        }

    by_moving_platform: Dict[str, Dict[str, Any]] = {}
    for moving in (False, True):
        rows = [entry for entry in per_seed if bool(entry["task"]["moving_platform"]) is moving]
        if not rows:
            continue
        row_scores = [row["result"]["score"] for row in rows]
        row_times = [row["result"]["time_sec"] for row in rows]
        row_success = sum(1 for row in rows if row["result"]["success"])
        row_hard_fail = sum(
            1
            for row in rows
            if (not row["result"]["success"])
            and abs(row["result"]["time_sec"]) < 1e-9
            and abs(row["result"]["score"]) < 1e-12
        )
        key = "moving_platform" if moving else "static_platform"
        by_moving_platform[key] = {
            "count": len(rows),
            "success_count": int(row_success),
            "success_rate": float(row_success / len(rows) if rows else 0.0),
            "hard_fail_count": int(row_hard_fail),
            "score_mean": float(statistics.fmean(row_scores) if row_scores else 0.0),
            "score_median": float(statistics.median(row_scores) if row_scores else 0.0),
            "time_mean_sec": float(statistics.fmean(row_times) if row_times else 0.0),
            "time_median_sec": float(statistics.median(row_times) if row_times else 0.0),
            "time_max_sec": float(max(row_times) if row_times else 0.0),
        }

    summary = {
        "model": str(model_path),
        "uid": uid,
        "seeds": [int(s) for s in seeds],
        "aggregate": aggregate,
        "by_challenge_type": by_challenge_type,
        "by_moving_platform": by_moving_platform,
        "per_seed": per_seed,
    }

    if args.json:
        print(json.dumps(summary, indent=2))
        return

    print("----------------------------------------------------")
    print(f"Model      : {summary['model']}")
    print(f"UID        : {summary['uid']}")
    print(f"Seeds      : {', '.join(str(s) for s in summary['seeds'])}")
    print("Aggregate  :")
    print(f"  num_seeds    = {aggregate['num_seeds']}")
    print(f"  success_rate = {aggregate['success_rate']:.3f} ({aggregate['success_count']}/{aggregate['num_seeds']})")
    print(f"  hard_fails   = {aggregate['hard_fail_count']} (time=0 and score=0)")
    print(
        f"  score        = mean {aggregate['score_mean']:.6f} | median {aggregate['score_median']:.6f} "
        f"| std {aggregate['score_std']:.6f} | min {aggregate['score_min']:.6f} | max {aggregate['score_max']:.6f}"
    )
    print(
        f"  time_sec     = mean {aggregate['time_mean_sec']:.3f} | median {aggregate['time_median_sec']:.3f} "
        f"| min {aggregate['time_min_sec']:.3f} | max {aggregate['time_max_sec']:.3f}"
    )
    print(f"  wall_time_sec= {aggregate['wall_time_sec']:.3f}")
    print("By challenge type:")
    for key, stats in by_challenge_type.items():
        print(
            f"  {key:<12} count={stats['count']:>3} "
            f"success_rate={stats['success_rate']:.3f} ({stats['success_count']}/{stats['count']}) "
            f"hard_fails={stats['hard_fail_count']} "
            f"score_mean={stats['score_mean']:.6f} "
            f"time_mean_sec={stats['time_mean_sec']:.3f}"
        )
    print("By platform:")
    for key, stats in by_moving_platform.items():
        print(
            f"  {key:<15} count={stats['count']:>3} "
            f"success_rate={stats['success_rate']:.3f} ({stats['success_count']}/{stats['count']}) "
            f"hard_fails={stats['hard_fail_count']} "
            f"score_mean={stats['score_mean']:.6f} "
            f"time_mean_sec={stats['time_mean_sec']:.3f}"
        )
    print("Per-seed   :")
    for entry in per_seed:
        print(
            f"  seed={entry['seed']} type={entry['task']['challenge_type']} "
            f"success={entry['result']['success']} time_sec={entry['result']['time_sec']:.3f} "
            f"score={entry['result']['score']:.6f}"
        )
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()
