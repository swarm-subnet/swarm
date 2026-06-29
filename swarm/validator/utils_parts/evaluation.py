import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np

from swarm.constants import (
    BENCHMARK_SCREENING_SEED_COUNT,
    BENCHMARK_TEMPLATE,
    BENCHMARK_VERSION,
    MAX_INFLIGHT_SEED_UPLOADS,
    SCREENING_TEMPLATE,
    SIM_DT,
    UNIFIED_CHUNK_SIZE,
)
from swarm.validator.backend_api import BackendTransportError, authorize_with_retry
from swarm.validator.task_gen import random_task, screening_task
from swarm.validator.runtime_telemetry import tracker_call

from .heartbeat import HeartbeatManager


_EMPTY_PER_TYPE = (
    "city", "open", "mountain", "village", "warehouse", "forest", "moving_platform",
)


def _utils_facade():
    from swarm.validator import utils as validator_utils

    return validator_utils


def _empty_per_type() -> Dict[str, List[float]]:
    return {name: [] for name in _EMPTY_PER_TYPE}


def _task_for_seed_index(abs_idx: int, seed: int):
    """Task for an absolute seed index: screening range uses SCREENING_TEMPLATE, benchmark range BENCHMARK_TEMPLATE."""
    if abs_idx < BENCHMARK_SCREENING_SEED_COUNT:
        slot = SCREENING_TEMPLATE[abs_idx % len(SCREENING_TEMPLATE)]
    else:
        bench_idx = abs_idx - BENCHMARK_SCREENING_SEED_COUNT
        slot = BENCHMARK_TEMPLATE[bench_idx % len(BENCHMARK_TEMPLATE)]
    try:
        return screening_task(
            sim_dt=SIM_DT, seed=seed,
            challenge_type=slot["challenge_type"],
            distance_range=slot["distance_range"],
            goal_height_range=slot.get("goal_height_range"),
            moving_platform=slot["moving_platform"],
        )
    except Exception as e:
        bt.logging.warning(f"Failed to build templated task for seed {seed} (idx {abs_idx}): {e}")
        return None


async def _evaluate_seeds(
    self,
    uid: int,
    model_path: Path,
    seeds: List[int],
    description: str = "benchmark",
    on_seed_complete: Optional[Callable[[], None]] = None,
    prior_seeds_done: int = 0,
    prior_total_seeds: int = 0,
    prior_avg: float = 0.0,
    pre_built_tasks: Optional[List] = None,
) -> Tuple[List[float], Dict[str, List[float]], List[dict]]:
    """Evaluate a model on multiple seeds using parallel Docker containers."""
    all_scores = []
    per_type_scores = _empty_per_type()

    challenge_type_to_name = {
        1: "city",
        2: "open",
        3: "mountain",
        4: "village",
        5: "warehouse",
        6: "forest",
    }

    model_hash_short = ""
    try:
        from swarm.utils.hash import sha256sum as _sha
        model_hash_short = _sha(model_path)[:12]
    except Exception:
        pass
    bt.logging.info(f"━━━ {description.upper()} UID {uid} | {model_hash_short} ━━━")

    if pre_built_tasks is not None:
        tasks = list(pre_built_tasks)
    else:
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
        return [], per_type_scores, []

    phase = "screening" if "screening" in description.lower() else "benchmark"
    results = await self.docker_evaluator.evaluate_seeds_parallel(
        tasks=valid_tasks,
        uid=uid,
        model_path=model_path,
        on_seed_complete=on_seed_complete,
        phase_label=phase,
        prior_seeds_done=prior_seeds_done,
        prior_total_seeds=prior_total_seeds,
        prior_avg=prior_avg,
    )

    seed_details = []
    task_idx = 0
    for i, task in enumerate(tasks):
        if task is None:
            all_scores.append(0.0)
            seed_details.append({"score": 0.0, "map_type": "unknown"})
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

            seed_details.append({"score": score, "map_type": type_name})
            task_idx += 1
        else:
            type_name = challenge_type_to_name.get(task.challenge_type, "unknown")
            all_scores.append(0.0)
            seed_details.append({"score": 0.0, "map_type": type_name})

    bt.logging.info(f"✅ {description} complete for UID {uid}: {len(all_scores)} seeds evaluated")
    return all_scores, per_type_scores, seed_details


async def _run_streaming_phase(
    self,
    uid: int,
    model_path: Path,
    seeds: List[int],
    *,
    phase_description: str,
    seed_offset: int,
    epoch_number: int,
    hb: HeartbeatManager,
    task_id: Optional[int] = None,
    pre_built_tasks: Optional[List] = None,
    re_authorize: Optional[Callable[[], Awaitable[Dict[str, Any]]]] = None,
    should_stop: Optional[Callable[[], Optional[str]]] = None,
    on_chunk_complete: Optional[Callable[..., None]] = None,
    chunk_size: int = UNIFIED_CHUNK_SIZE,
    max_inflight: int = MAX_INFLIGHT_SEED_UPLOADS,
    evaluator_prior_done: int = 0,
    evaluator_total_seeds: Optional[int] = None,
) -> Tuple[List[float], Dict[str, List[float]], List[dict], Optional[str]]:
    """Evaluate seeds in chunks, streaming scores with fire-and-forget uploads.

    The caller owns heartbeat lifecycle and outer telemetry. Returns accumulated
    scores, per-type scores, seed details, and an optional cancel reason set
    either by ``re_authorize`` (task no longer authorized) or ``should_stop``
    (backend requested stop via heartbeat response).
    """
    all_scores: List[float] = []
    all_per_type: Dict[str, List[float]] = _empty_per_type()
    all_details: List[dict] = []
    inflight: List[asyncio.Task] = []
    failed_batches: List[List[dict]] = []
    cancel_reason: Optional[str] = None
    total_for_evaluator = (
        evaluator_total_seeds if evaluator_total_seeds is not None else len(seeds)
    )

    if pre_built_tasks is None:
        # Build stratified-template tasks for this window from each seed's absolute index.
        pre_built_tasks = [
            _task_for_seed_index(seed_offset + i, seed) for i, seed in enumerate(seeds)
        ]

    async def _safe_upload(batch: List[dict]) -> None:
        try:
            result = await self.backend_api.post_seed_scores_batch(
                model_uid=uid, epoch_number=epoch_number, scores=batch,
                task_id=task_id,
            )
        except Exception as exc:
            bt.logging.warning(f"Seed score upload failed for UID {uid}: {exc}")
            failed_batches.append(batch)
            return
        if not result or not result.get("recorded"):
            failed_batches.append(batch)

    async def _wait_for_slot() -> None:
        while len(inflight) >= max_inflight:
            done, _pending = await asyncio.wait(
                inflight, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                if task in inflight:
                    inflight.remove(task)

    async def _drain_inflight() -> None:
        if inflight:
            await asyncio.gather(*inflight, return_exceptions=True)
            inflight.clear()

    try:
        for chunk_start in range(0, len(seeds), chunk_size):
            if should_stop is not None:
                stop_reason = should_stop()
                if stop_reason:
                    await _drain_inflight()
                    cancel_reason = f"backend stop_required: {stop_reason}"
                    break

            if re_authorize is not None:
                await _drain_inflight()
                auth = await authorize_with_retry(
                    re_authorize,
                    log_prefix=f"UID {uid} mid-{phase_description}: ",
                )
                if not auth.get("authorized"):
                    cancel_reason = str(auth.get("reason") or "unauthorized")
                    break

            batch_seeds = seeds[chunk_start:chunk_start + chunk_size]
            batch_tasks = (
                pre_built_tasks[chunk_start:chunk_start + chunk_size]
                if pre_built_tasks is not None else None
            )
            if not batch_seeds:
                break

            prior_avg = float(np.mean(all_scores)) if all_scores else 0.0
            batch_scores, batch_per_type, batch_details = await _utils_facade()._evaluate_seeds(
                self, uid, model_path, batch_seeds,
                f"{phase_description} [{chunk_start + 1}..{chunk_start + len(batch_seeds)}]",
                on_seed_complete=hb.on_seed_complete,
                prior_seeds_done=evaluator_prior_done + len(all_scores),
                prior_total_seeds=total_for_evaluator,
                prior_avg=prior_avg,
                pre_built_tasks=batch_tasks,
            )

            seed_batch = [
                {
                    "seed_index": seed_offset + chunk_start + j,
                    "score": detail["score"],
                    "map_type": detail["map_type"],
                }
                for j, detail in enumerate(batch_details)
                if detail.get("map_type") != "unknown"
            ]
            if seed_batch:
                await _wait_for_slot()
                inflight.append(asyncio.create_task(_safe_upload(seed_batch)))

            all_scores.extend(batch_scores)
            for type_name, scores in batch_per_type.items():
                if type_name in all_per_type:
                    all_per_type[type_name].extend(scores)
            all_details.extend(batch_details)

            if on_chunk_complete is not None:
                try:
                    on_chunk_complete(
                        evaluated=len(all_scores),
                        total=len(seeds),
                        running_avg=float(np.mean(all_scores)) if all_scores else 0.0,
                        chunk_scores=list(batch_scores),
                        chunk_per_type={k: list(v) for k, v in batch_per_type.items()},
                        chunk_details=list(batch_details),
                    )
                except Exception as exc:
                    bt.logging.warning(f"on_chunk_complete callback failed for UID {uid}: {exc}")
    finally:
        await _drain_inflight()
        if failed_batches:
            retry_queue = list(failed_batches)
            failed_batches.clear()
            for batch in retry_queue:
                try:
                    result = await self.backend_api.post_seed_scores_batch(
                        model_uid=uid, epoch_number=epoch_number, scores=batch,
                        task_id=task_id,
                    )
                except Exception as exc:
                    bt.logging.warning(
                        f"Final retry of {len(batch)} seed scores failed for UID {uid}: {exc}"
                    )
                    continue
                if not result or not result.get("recorded"):
                    bt.logging.warning(
                        f"Final retry of {len(batch)} seed scores not recorded for UID {uid}"
                    )

    return all_scores, all_per_type, all_details, cancel_reason


async def _run_screening(
    self, uid: int, model_path: Path, reeval: bool = False,
    task_id: Optional[int] = None,
    *,
    seeds_from: int = 0,
    seeds_to: Optional[int] = None,
    cancel_flag: Optional[asyncio.Event] = None,
    batch_id: Optional[int] = None,
) -> Tuple[float, List[float], Dict[str, List[float]], Optional[str], bool]:
    """Run a screening seed window and stream per-seed scores.

    Returns ``(avg, all_scores, per_type, cancel_reason, early_failed)``.

    ``seeds_from`` / ``seeds_to`` carve the batch window out of the epoch's
    screening seed list. ``cancel_flag`` is set by the SSE listener; when
    set the streaming phase aborts at the next chunk boundary. Loser /
    copy early-fail is decided backend-side, so ``early_failed`` is always
    ``False`` here.
    """
    full_seeds = self.seed_manager.get_screening_seeds()
    upper = seeds_to if seeds_to is not None else len(full_seeds)
    upper = max(seeds_from, min(upper, len(full_seeds)))
    screening_seeds = full_seeds[seeds_from:upper]
    total_seeds = len(screening_seeds)
    # Cumulative-progress framing for the heartbeat: report the full
    # screening range as total and the resume offset as already-done so
    # the dashboard renders honest progress (e.g. 130/300) instead of
    # 0/(remaining-slice) on resume.
    heartbeat_total = upper
    progress_offset = seeds_from
    epoch = self.seed_manager.epoch_number

    tracker_call(
        self,
        "mark_screening_started",
        uid=int(uid),
        total_seeds=int(total_seeds),
    )

    hb = HeartbeatManager(self.backend_api, asyncio.get_running_loop())
    hb_queue = getattr(self, '_heartbeat_queue', None)
    decision_version = None
    if hb_queue:
        matched = next((item for item in hb_queue if int(item.get("uid", -1)) == uid), None)
        if matched is not None:
            decision_version = matched.get("backend_decision_version")
    active_task = {
        "uid": uid,
        "phase": "REEVAL" if reeval else "SCREENING",
        "assignment_id": task_id,
        "batch_id": batch_id,
        "epoch_number": epoch,
        "benchmark_version": BENCHMARK_VERSION,
    }
    hb.start(
        "evaluating_screening",
        uid,
        heartbeat_total,
        queue=hb_queue,
        active_task=active_task,
        backend_decision_version=decision_version,
        progress_offset=progress_offset,
    )

    def _on_chunk(**info) -> None:
        tracker_call(
            self,
            "mark_screening_progress",
            uid=int(uid),
            progress=int(info["evaluated"]),
            total_seeds=int(info["total"]),
            running_median=float(info["running_avg"]),
            note=f"checkpoint {info['evaluated']}/{info['total']}",
        )

    def _should_stop() -> Optional[str]:
        if cancel_flag is not None and cancel_flag.is_set():
            return "cancel_flag_set"
        return hb.should_stop()

    try:
        all_scores, all_per_type, _details, cancel_reason = await _run_streaming_phase(
            self,
            uid,
            model_path,
            screening_seeds,
            phase_description="screening",
            seed_offset=seeds_from,
            epoch_number=epoch,
            hb=hb,
            task_id=task_id,
            should_stop=_should_stop,
            on_chunk_complete=_on_chunk,
        )
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
    )
    if cancel_reason is None:
        bt.logging.info(
            f"📊 Screening result for UID {uid}: "
            f"avg={avg_score:.4f} ({len(all_scores)}/{total_seeds} seeds)"
        )
    else:
        bt.logging.warning(
            f"Screening cancelled for UID {uid} after "
            f"{len(all_scores)}/{total_seeds} seeds: {cancel_reason}"
        )
    return avg_score, all_scores, all_per_type, cancel_reason, False


async def _run_full_benchmark(
    self, uid: int, model_path: Path, seeds: Optional[List[int]] = None,
    reeval: bool = False, task_id: Optional[int] = None,
    *,
    cancel_flag: Optional[asyncio.Event] = None,
    seeds_from: Optional[int] = None,
    seeds_to: Optional[int] = None,
    batch_id: Optional[int] = None,
) -> Tuple[float, Dict[str, float], List[float], Dict[str, List[float]], Optional[str]]:
    """Run full benchmark. Uses benchmark seeds by default, or custom seeds if provided.

    Returns (avg_score, per_type_avgs, all_scores, per_type_raw, cancel_reason).

    When ``reeval`` is True the heartbeat labels the active task as REEVAL and a
    backend authorization check runs before each streaming chunk so a stale
    champion re-eval can be halted mid-flight (epoch rollover, admin override,
    wrong-model edge cases).
    """
    if seeds is None:
        # REEVAL covers all 1100 seeds; benchmark covers only 300..1099.
        # Caller signals REEVAL by passing seeds_from < 300.
        if seeds_from is not None and seeds_from < BENCHMARK_SCREENING_SEED_COUNT:
            benchmark_seeds = self.seed_manager.seeds[seeds_from:]
            seed_offset = seeds_from
            heartbeat_total = len(self.seed_manager.seeds)
            progress_offset = seeds_from
        elif seeds_from is not None and seeds_from > BENCHMARK_SCREENING_SEED_COUNT:
            full_benchmark = self.seed_manager.get_benchmark_seeds()
            offset = seeds_from - BENCHMARK_SCREENING_SEED_COUNT
            benchmark_seeds = full_benchmark[offset:]
            seed_offset = seeds_from
            heartbeat_total = len(full_benchmark)
            progress_offset = offset
        else:
            benchmark_seeds = self.seed_manager.get_benchmark_seeds()
            seed_offset = BENCHMARK_SCREENING_SEED_COUNT
            heartbeat_total = len(benchmark_seeds)
            progress_offset = 0
    else:
        benchmark_seeds = seeds
        seed_offset = 0
        heartbeat_total = len(seeds)
        progress_offset = 0

    if seeds is None and seeds_to is not None:
        count = max(0, int(seeds_to) - int(seed_offset))
        benchmark_seeds = benchmark_seeds[:count]
        heartbeat_total = len(benchmark_seeds)
        progress_offset = 0

    # Full benchmark and REEVAL (seeds is None) build stratified-template tasks from each
    # seed's absolute index inside _run_streaming_phase. Custom-seed runs stay random.
    pre_built_tasks: Optional[List] = None
    if seeds is not None:
        pre_built_tasks = []
        for seed in benchmark_seeds:
            try:
                pre_built_tasks.append(random_task(sim_dt=SIM_DT, seed=seed))
            except Exception as e:
                bt.logging.warning(f"Failed to create task for seed {seed}: {e}")
                pre_built_tasks.append(None)

    total_seeds = len(benchmark_seeds)
    note = "full benchmark" if seeds is None else "custom seeds"
    epoch = self.seed_manager.epoch_number

    tracker_call(
        self,
        "mark_benchmark_started",
        uid=int(uid),
        total_seeds=total_seeds,
        note=note,
    )

    hb = HeartbeatManager(self.backend_api, asyncio.get_running_loop())
    hb_queue = getattr(self, '_heartbeat_queue', None)
    decision_version = None
    if hb_queue:
        matched = next((item for item in hb_queue if int(item.get("uid", -1)) == uid), None)
        if matched is not None:
            decision_version = matched.get("backend_decision_version")
    active_task = {
        "uid": uid,
        "phase": "REEVAL" if reeval else "BENCHMARK",
        "assignment_id": task_id,
        "batch_id": batch_id,
        "epoch_number": epoch,
        "benchmark_version": BENCHMARK_VERSION,
    }
    hb.start(
        "evaluating_benchmark",
        uid,
        heartbeat_total,
        queue=hb_queue,
        active_task=active_task,
        backend_decision_version=decision_version,
        progress_offset=progress_offset,
    )

    def _on_chunk(**info) -> None:
        tracker_call(
            self,
            "mark_benchmark_progress",
            uid=int(uid),
            progress=int(info["evaluated"]),
            total_seeds=int(info["total"]),
            note=f"checkpoint {info['evaluated']}/{info['total']}",
        )

    def _should_stop() -> Optional[str]:
        if cancel_flag is not None and cancel_flag.is_set():
            return "cancel_flag_set"
        return hb.should_stop()

    try:
        all_scores, per_type_raw, _details, cancel_reason = await _run_streaming_phase(
            self,
            uid,
            model_path,
            benchmark_seeds,
            phase_description="full benchmark",
            seed_offset=seed_offset,
            epoch_number=epoch,
            hb=hb,
            task_id=task_id,
            pre_built_tasks=pre_built_tasks,
            should_stop=_should_stop,
            on_chunk_complete=_on_chunk,
        )
    finally:
        hb.finish()

    avg_score = float(np.mean(all_scores)) if all_scores else 0.0
    per_type_avgs: Dict[str, float] = {}
    for type_name, scores in per_type_raw.items():
        per_type_avgs[type_name] = float(np.mean(scores)) if scores else 0.0

    completed_note = note if cancel_reason is None else f"{note} (cancelled: {cancel_reason})"
    tracker_call(
        self,
        "mark_benchmark_completed",
        uid=int(uid),
        evaluated=len(all_scores),
        total_seeds=total_seeds,
        median_score=float(avg_score),
        note=completed_note,
    )
    if cancel_reason is None:
        bt.logging.info(f"📊 Full benchmark result for UID {uid}: avg={avg_score:.4f}")
    else:
        bt.logging.warning(
            f"Full benchmark cancelled for UID {uid} after "
            f"{len(all_scores)}/{total_seeds} seeds: {cancel_reason}"
        )
    return avg_score, per_type_avgs, all_scores, per_type_raw, cancel_reason
