import asyncio
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np

from swarm.constants import (
    SCREENING_BOOTSTRAP_THRESHOLD,
    SCREENING_CHECKPOINT_SIZE,
    SCREENING_EARLY_FAIL_FACTORS,
    SCREENING_EARLY_PASS_FACTORS,
    SCREENING_TOP_MODEL_FACTOR,
    SIM_DT,
)
from swarm.validator.task_gen import random_task
from swarm.validator.runtime_telemetry import tracker_call

from .heartbeat import HeartbeatManager


def _utils_facade():
    from swarm.validator import utils as validator_utils

    return validator_utils


async def _evaluate_seeds(
    self,
    uid: int,
    model_path: Path,
    seeds: List[int],
    description: str = "benchmark",
    on_seed_complete: Optional[Callable[[], None]] = None,
) -> Tuple[List[float], Dict[str, List[float]]]:
    """Evaluate a model on multiple seeds using parallel Docker containers."""
    all_scores = []
    per_type_scores = {
        "city": [], "open": [], "mountain": [],
        "village": [], "warehouse": [], "forest": [], "moving_platform": [],
    }

    challenge_type_to_name = {
        1: "city",
        2: "open",
        3: "mountain",
        4: "village",
        5: "warehouse",
        6: "forest",
    }

    bt.logging.info(f"🔬 Starting {description} for UID {uid}: {len(seeds)} seeds (parallel)")

    tasks = []
    for seed in seeds:
        try:
            task = random_task(sim_dt=SIM_DT, seed=seed)
            tasks.append(task)
        except Exception as e:
            bt.logging.warning(f"Failed to create task for seed {seed}: {e}")
            tasks.append(None)

    valid_tasks = [t for t in tasks if t is not None]
    if not valid_tasks:
        bt.logging.warning(f"No valid tasks created for UID {uid}")
        return [], per_type_scores

    results = await self.docker_evaluator.evaluate_seeds_parallel(
        tasks=valid_tasks,
        uid=uid,
        model_path=model_path,
        on_seed_complete=on_seed_complete,
    )

    task_idx = 0
    for i, task in enumerate(tasks):
        if task is None:
            all_scores.append(0.0)
            continue

        if task_idx < len(results):
            result = results[task_idx]
            score = result.score if result else 0.0
            all_scores.append(score)

            type_name = challenge_type_to_name.get(task.challenge_type, "unknown")
            if getattr(task, 'moving_platform', False):
                per_type_scores["moving_platform"].append(score)
            elif type_name in per_type_scores:
                per_type_scores[type_name].append(score)

            task_idx += 1
        else:
            all_scores.append(0.0)

    bt.logging.info(f"✅ {description} complete for UID {uid}: {len(all_scores)} seeds evaluated")
    return all_scores, per_type_scores


def _get_screening_threshold(self) -> float:
    """Compute the screening pass/fail threshold from the current top model."""
    current_top = getattr(self, '_current_top', None)
    if not current_top or not current_top.get('score'):
        return float(SCREENING_BOOTSTRAP_THRESHOLD)
    return float(current_top.get('score', 0.0)) * SCREENING_TOP_MODEL_FACTOR


async def _run_screening(
    self, uid: int, model_path: Path
) -> Tuple[float, List[float], Dict[str, List[float]]]:
    """Run screening benchmark with early termination on clearly failing/passing models.

    Returns (avg_score, all_scores, per_type_scores).
    """
    screening_seeds = self.seed_manager.get_screening_seeds()
    threshold = _get_screening_threshold(self)
    total_seeds = len(screening_seeds)
    tracker_call(
        self,
        "mark_screening_started",
        uid=int(uid),
        total_seeds=int(total_seeds),
        threshold=float(threshold),
    )

    hb = HeartbeatManager(self.backend_api, asyncio.get_running_loop())
    hb.start("evaluating_screening", uid, total_seeds)

    all_per_type: Dict[str, List[float]] = {
        "city": [], "open": [], "mountain": [],
        "village": [], "warehouse": [], "forest": [], "moving_platform": [],
    }

    try:
        all_scores: List[float] = []
        checkpoints = list(range(
            SCREENING_CHECKPOINT_SIZE, total_seeds + 1, SCREENING_CHECKPOINT_SIZE
        ))
        if not checkpoints or checkpoints[-1] < total_seeds:
            checkpoints.append(total_seeds)
        completion_note = ""
        for checkpoint in checkpoints:
            batch_seeds = screening_seeds[len(all_scores):checkpoint]
            if not batch_seeds:
                break

            batch_scores, batch_per_type = await _utils_facade()._evaluate_seeds(
                self, uid, model_path, batch_seeds,
                f"screening [{len(all_scores) + 1}..{checkpoint}]",
                on_seed_complete=hb.on_seed_complete,
            )
            all_scores.extend(batch_scores)
            for type_name, scores in batch_per_type.items():
                if type_name in all_per_type:
                    all_per_type[type_name].extend(scores)

            evaluated = len(all_scores)
            if evaluated < SCREENING_CHECKPOINT_SIZE:
                continue

            running_avg = float(np.mean(all_scores))
            tracker_call(
                self,
                "mark_screening_progress",
                uid=int(uid),
                progress=int(evaluated),
                total_seeds=int(total_seeds),
                running_median=float(running_avg),
                note=f"checkpoint {evaluated}/{total_seeds}",
            )

            fail_factor = SCREENING_EARLY_FAIL_FACTORS.get(evaluated)
            if fail_factor is not None and running_avg < threshold * fail_factor:
                completion_note = (
                    f"early_fail {evaluated}/{total_seeds} avg={running_avg:.4f}"
                )
                bt.logging.info(
                    f"⏩ Screening early fail for UID {uid}: "
                    f"avg={running_avg:.4f} < {threshold * fail_factor:.4f} "
                    f"after {evaluated}/{total_seeds} seeds"
                )
                break

            pass_factor = SCREENING_EARLY_PASS_FACTORS.get(evaluated)
            if pass_factor is not None and running_avg > threshold * pass_factor:
                completion_note = (
                    f"early_pass {evaluated}/{total_seeds} avg={running_avg:.4f}"
                )
                bt.logging.info(
                    f"⏩ Screening early pass for UID {uid}: "
                    f"avg={running_avg:.4f} > {threshold * pass_factor:.4f} "
                    f"after {evaluated}/{total_seeds} seeds"
                )
                break
    finally:
        hb.finish()

    avg_score = float(np.mean(all_scores)) if all_scores else 0.0
    tracker_call(
        self,
        "mark_screening_completed",
        uid=int(uid),
        evaluated=len(all_scores),
        total_seeds=int(total_seeds),
        median_score=float(avg_score),
        note=completion_note,
    )
    bt.logging.info(
        f"📊 Screening result for UID {uid}: "
        f"avg={avg_score:.4f} ({len(all_scores)}/{total_seeds} seeds)"
    )
    return avg_score, all_scores, all_per_type


async def _run_full_benchmark(
    self, uid: int, model_path: Path, seeds: Optional[List[int]] = None
) -> Tuple[float, Dict[str, float], List[float], Dict[str, List[float]]]:
    """Run full benchmark. Uses benchmark seeds by default, or custom seeds if provided.

    Returns (avg_score, per_type_avgs, all_scores, per_type_raw).
    """
    benchmark_seeds = seeds if seeds is not None else self.seed_manager.get_benchmark_seeds()
    tracker_call(
        self,
        "mark_benchmark_started",
        uid=int(uid),
        total_seeds=len(benchmark_seeds),
        note="full benchmark" if seeds is None else "custom seeds",
    )

    hb = HeartbeatManager(self.backend_api, asyncio.get_running_loop())
    hb.start("evaluating_benchmark", uid, len(benchmark_seeds))

    try:
        all_scores, per_type_raw = await _utils_facade()._evaluate_seeds(
            self, uid, model_path, benchmark_seeds, "full benchmark",
            on_seed_complete=hb.on_seed_complete
        )
    finally:
        hb.finish()

    avg_score = float(np.mean(all_scores)) if all_scores else 0.0

    per_type_avgs = {}
    for type_name, scores in per_type_raw.items():
        per_type_avgs[type_name] = float(np.mean(scores)) if scores else 0.0

    tracker_call(
        self,
        "mark_benchmark_completed",
        uid=int(uid),
        evaluated=len(all_scores),
        total_seeds=len(benchmark_seeds),
        median_score=float(avg_score),
        note="full benchmark" if seeds is None else "custom seeds",
    )
    bt.logging.info(f"📊 Full benchmark result for UID {uid}: avg={avg_score:.4f}")
    return avg_score, per_type_avgs, all_scores, per_type_raw


# ──────────────────────────────────────────────────────────────────────────
# Scoring & detection helpers
# ──────────────────────────────────────────────────────────────────────────

def _passes_screening(self, screening_score: float) -> bool:
    """Check if screening score meets the threshold."""
    current_top = getattr(self, '_current_top', None)

    if not current_top or not current_top.get('score'):
        threshold = SCREENING_BOOTSTRAP_THRESHOLD
        passed = screening_score >= threshold
        bt.logging.info(
            f"Screening (bootstrap mode): {screening_score:.4f} >= {threshold} = {passed}"
        )
        return passed

    top_score = current_top.get('score', 0.0)
    threshold = top_score * SCREENING_TOP_MODEL_FACTOR
    passed = screening_score >= threshold
    bt.logging.info(
        f"Screening: {screening_score:.4f} >= {threshold:.4f} "
        f"({SCREENING_TOP_MODEL_FACTOR:.0%} of top {top_score:.4f}) = {passed}"
    )
    return passed
