"""Process-based parallel seed evaluation for the regular validator path.

This mirrors the benchmark scheduler:
- one seed per batch
- resource-aware dispatch with class-based admission
- host-pressure-driven cap adjustments
- parent-side worker stall detection and replacement

The main difference is that validator evaluation degrades failed batches into
per-seed failures and keeps going, instead of aborting the full run.
"""

from __future__ import annotations

import asyncio
import queue as queue_mod
import re
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import bittensor as bt

from swarm.constants import N_DOCKER_WORKERS
from swarm.protocol import ValidationResult
from swarm.validator.runtime_telemetry import tracker_call

from ._shared import _docker_evaluator_facade

_MAX_TIMEOUT_RETRIES = 10


def _benchmark_engine():
    import swarm.benchmark.engine as bench_engine

    return bench_engine


def _task_seed(task: Any, index: int) -> int:
    try:
        return int(getattr(task, "map_seed", index))
    except Exception:
        return int(index)


def _task_challenge_type(task: Any) -> int:
    try:
        return int(getattr(task, "challenge_type", 0))
    except Exception:
        return 0


def _task_group_name(task: Any, index: int) -> str:
    bench_engine = _benchmark_engine()
    seed = _task_seed(task, index)
    challenge_type = _task_challenge_type(task)
    group_name = bench_engine._infer_bench_group(challenge_type, seed)
    if group_name:
        return str(group_name)
    return f"type{challenge_type}_unknown"


def _emit_seed_complete(
    on_seed_complete: Optional[Callable[..., None]],
    seed_meta: Optional[Dict[str, Any]],
) -> None:
    if on_seed_complete is None:
        return
    try:
        on_seed_complete(seed_meta)
    except TypeError:
        try:
            on_seed_complete()
        except Exception:
            pass
    except Exception:
        pass


def _failure_seed_meta(
    task: Any,
    *,
    uid: int,
    status: str,
    error: str,
    elapsed_sec: float = 0.0,
) -> Dict[str, Any]:
    return {
        "uid": int(uid),
        "map_seed": _task_seed(task, -1),
        "challenge_type": _task_challenge_type(task),
        "horizon_sec": float(getattr(task, "horizon", 0.0)),
        "moving_platform": bool(getattr(task, "moving_platform", False)),
        "status": status,
        "success": False,
        "sim_time_sec": 0.0,
        "seed_wall_sec": max(0.0, float(elapsed_sec)),
        "step_idx": 0,
        "error": error,
    }


def _log_scheduler_note(note: str) -> None:
    if "Scheduler pressure backoff" in note or "Scheduler pressure hold" in note:
        bt.logging.warning(note)
    else:
        bt.logging.info(note)


def _seed_status(seed_meta: Optional[Dict[str, Any]]) -> str:
    if not isinstance(seed_meta, dict):
        return ""
    return str(seed_meta.get("status", "")).strip()


def _build_result_seed_meta(
    meta: Optional[dict[str, Any]],
    *,
    uid: int,
    result_obj: ValidationResult,
    status: str,
) -> Dict[str, Any]:
    return {
        "uid": int(uid),
        "map_seed": int(meta.get("seed", -1)) if isinstance(meta, dict) else -1,
        "challenge_type": int(meta.get("challenge_type", -1)) if isinstance(meta, dict) else -1,
        "horizon_sec": float(meta.get("horizon", 0.0)) if isinstance(meta, dict) else 0.0,
        "moving_platform": bool(meta.get("moving_platform", False)) if isinstance(meta, dict) else False,
        "status": status,
        "success": bool(result_obj.success),
        "sim_time_sec": float(result_obj.time_sec),
        "seed_wall_sec": 0.0,
        "step_idx": 0,
        "error": "",
    }


def _summary_bucket_for_result(
    bench_engine: Any,
    *,
    status: str,
    result_obj: Optional[ValidationResult],
    is_infra_failure: bool = False,
) -> str:
    if bench_engine._is_backoff_timeout_status(status):
        return "timeout"
    if is_infra_failure or bench_engine._is_backoff_infra_status(status):
        return "runtime"
    if result_obj is not None and bool(result_obj.success):
        return "ok"
    return "failed"


def _pop_batch_seed_meta(
    batch_seed_meta: dict[int, Dict[str, Any]],
    *,
    batch_index: int,
    meta: Optional[dict[str, Any]],
    uid: int,
    result_obj: Optional[ValidationResult] = None,
    fallback_seed_meta: Optional[Dict[str, Any]] = None,
    prefer_fallback: bool = False,
) -> Dict[str, Any]:
    observed = batch_seed_meta.pop(int(batch_index), None)
    if prefer_fallback and fallback_seed_meta is not None:
        return fallback_seed_meta
    if isinstance(observed, dict):
        return observed
    if fallback_seed_meta is not None:
        return fallback_seed_meta
    if result_obj is None:
        result_obj = ValidationResult(int(uid), False, 0.0, 0.0)
    status = "seed_done" if bool(result_obj.success) else "seed_failed"
    return _build_result_seed_meta(
        meta,
        uid=uid,
        result_obj=result_obj,
        status=status,
    )


async def _run_process_parallel(
    *,
    all_tasks: list,
    task_meta: list[dict[str, Any]],
    batch_plan: list[list[int]],
    uid: int,
    model_path: Path,
    effective_workers: int,
    runtime_tracker: Any = None,
    on_seed_complete: Optional[Callable[..., None]] = None,
    phase_label: str = "eval",
    prior_seeds_done: int = 0,
    prior_total_seeds: int = 0,
    prior_avg: float = 0.0,
    heartbeat_sec: float = 30.0,
    model_image: Optional[str] = None,
) -> list:
    bench_engine = _benchmark_engine()
    ctx = bench_engine._benchmark_mp_context()
    result_queue = ctx.Queue()
    progress_queue = ctx.Queue()
    worker_queues: dict[int, Any] = {}
    workers: dict[int, Any] = {}
    worker_active_requests: dict[int, Any] = {}
    worker_last_heartbeat: dict[int, float] = {}
    worker_started_at: dict[int, float] = {}
    results: list[Optional[ValidationResult]] = [None] * len(all_tasks)
    scheduler = bench_engine._AdaptiveBackoffController(requested_workers=effective_workers)
    pending_batch_ids = list(range(len(batch_plan)))
    batch_seed_meta: dict[int, Dict[str, Any]] = {}
    batch_retry_counts: dict[int, int] = {}
    timeout_retries_used = 0
    stall_timeout_sec = max(
        bench_engine._PARENT_WORKER_STALL_TIMEOUT_SEC,
        max(0.0, float(heartbeat_sec)) * 2.0,
    )
    pressure_poll_interval_sec = max(
        0.0,
        float(getattr(bench_engine, "_PRESSURE_POLL_INTERVAL_SEC", 2.0)),
    )
    last_pressure_poll_at = 0.0

    from swarm.constants import (
        EVAL_SUMMARY_INTERVAL_SEC,
        GLOBAL_EVAL_BASE_SEC,
        GLOBAL_EVAL_CAP_SEC,
        GLOBAL_EVAL_PER_SEED_SEC,
    )

    max_batch_size = max((len(indices) for indices in batch_plan), default=1)
    wall_timeout = GLOBAL_EVAL_BASE_SEC + (GLOBAL_EVAL_PER_SEED_SEC * max_batch_size)
    if GLOBAL_EVAL_CAP_SEC > 0:
        wall_timeout = min(wall_timeout, GLOBAL_EVAL_CAP_SEC)
    bt.logging.info(
        f"    {len(all_tasks)} seeds | {effective_workers} workers | "
        f"timeout={wall_timeout:.0f}s | scheduler={'on' if scheduler.enabled else 'static'}"
    )
    for line in scheduler.describe_configuration_lines():
        bt.logging.info(f"    {line}")

    def _spawn_worker(worker_slot: int) -> None:
        task_queue = ctx.Queue()
        worker = ctx.Process(
            target=bench_engine._benchmark_worker_main,
            args=(worker_slot, task_queue, result_queue, progress_queue),
            name=f"validator_host_worker_{worker_slot}",
            daemon=True,
        )
        worker.start()
        worker_queues[worker_slot] = task_queue
        workers[worker_slot] = worker

    def _close_queue(queue_obj: Any) -> None:
        try:
            queue_obj.close()
        except Exception:
            pass

    def _restart_worker(worker_slot: int) -> None:
        worker = workers.get(worker_slot)
        if worker is not None:
            try:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=2.0)
            except Exception:
                pass
        queue_obj = worker_queues.pop(worker_slot, None)
        if queue_obj is not None:
            _close_queue(queue_obj)
        workers.pop(worker_slot, None)
        worker_last_heartbeat.pop(worker_slot, None)
        worker_started_at.pop(worker_slot, None)
        worker_active_requests.pop(worker_slot, None)
        _spawn_worker(worker_slot)
        tracker_call(runtime_tracker, "mark_docker_worker_restart", worker_slot=int(worker_slot))

    def _active_group_names() -> list[str]:
        active_groups: list[str] = []
        for request in worker_active_requests.values():
            try:
                if not request.batch_indices:
                    continue
                first_idx = int(request.batch_indices[0])
                active_groups.append(str(task_meta[first_idx]["group"]))
            except Exception:
                continue
        return active_groups

    def _maybe_poll_scheduler(*, force: bool = False) -> None:
        nonlocal last_pressure_poll_at
        now = time.monotonic()
        if (
            not force
            and pressure_poll_interval_sec > 0.0
            and (now - last_pressure_poll_at) < pressure_poll_interval_sec
        ):
            return
        last_pressure_poll_at = now
        note = scheduler.observe_resources(_active_group_names())
        if note:
            _log_scheduler_note(note)
            tracker_call(
                runtime_tracker,
                "mark_docker_backoff",
                active_worker_cap=int(scheduler.active_worker_cap),
                note=str(note),
            )

    def _dispatch_available_batches() -> None:
        while pending_batch_ids and len(worker_active_requests) < scheduler.active_worker_cap:
            idle_worker_slots = [
                slot
                for slot in range(effective_workers)
                if slot not in worker_active_requests
                and slot in workers
                and workers[slot].is_alive()
            ]
            if not idle_worker_slots:
                return
            batch_index = bench_engine._select_next_batch_index(
                pending_batch_ids=pending_batch_ids,
                batch_plan=batch_plan,
                task_meta=task_meta,
                active_batch_ids=[
                    int(request.batch_index) for request in worker_active_requests.values()
                ],
                active_worker_cap=scheduler.active_worker_cap,
                scheduler=scheduler,
            )
            if batch_index is None:
                return
            pending_batch_ids.remove(batch_index)
            batch_indices = list(batch_plan[batch_index])
            batch_meta = [task_meta[idx] for idx in batch_indices]
            worker_slot = min(idle_worker_slots)
            request = bench_engine._ProcessBatchRequest(
                batch_index=batch_index,
                batch_indices=batch_indices,
                tasks=[all_tasks[idx] for idx in batch_indices],
                uid=uid,
                model_path=str(model_path),
                task_total=len(all_tasks),
                model_image=model_image,
            )
            worker_active_requests[worker_slot] = request
            now = time.time()
            worker_started_at[worker_slot] = now
            worker_last_heartbeat[worker_slot] = now
            worker_queues[worker_slot].put(request)
            group_name = str(batch_meta[0]["group"]) if batch_meta else "unknown"
            seed_list = [int(meta["seed"]) for meta in batch_meta]
            tracker_call(
                runtime_tracker,
                "mark_docker_dispatch",
                worker_slot=int(worker_slot),
                batch_index=int(batch_index),
                group=str(group_name),
                seed=int(seed_list[0]),
                active_worker_cap=int(scheduler.active_worker_cap),
            )

    def _remember_seed_meta(batch_index: int, seed_meta: Optional[Dict[str, Any]]) -> None:
        if isinstance(seed_meta, dict):
            batch_seed_meta[int(batch_index)] = dict(seed_meta)

    def _observe_final_seed(
        meta: Optional[dict[str, Any]],
        seed_meta: Optional[Dict[str, Any]],
    ) -> None:
        group_name = str(meta.get("group", "")) if isinstance(meta, dict) else None
        note = scheduler.observe_seed(seed_meta, group_name=group_name)
        if note:
            _log_scheduler_note(note)
            tracker_call(
                runtime_tracker,
                "mark_docker_backoff",
                active_worker_cap=int(scheduler.active_worker_cap),
                note=str(note),
            )

    def _drain_progress_events() -> None:
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue_mod.Empty:
                return
            if isinstance(event, bench_engine._ProcessWorkerHeartbeat):
                worker_slot = int(event.worker_id)
                worker_last_heartbeat[worker_slot] = float(event.ts)
                if event.event_type == "batch_started":
                    worker_started_at[worker_slot] = float(event.ts)
                continue
            _remember_seed_meta(
                int(getattr(event, "batch_index", -1)),
                getattr(event, "seed_meta", None),
            )

    def _complete_failed_request(
        worker_slot: int,
        *,
        request: Any,
        status: str,
        error: str,
        elapsed_sec: float,
    ) -> None:
        bt.logging.warning(
            f"[Validator eval] worker {worker_slot} failed batch {request.batch_index + 1}/{len(batch_plan)} "
            f"with status={status}: {error}"
        )
        tracker_call(
            runtime_tracker,
            "mark_docker_worker_failure",
            status=str(status),
            worker_slot=int(worker_slot),
            error=str(error),
        )
        for idx, task in zip(request.batch_indices, request.tasks):
            if status == "worker_stall_timeout":
                seed_meta = bench_engine._build_worker_stall_seed_meta(
                    task,
                    uid=request.uid,
                    elapsed_sec=elapsed_sec,
                    error=error,
                )
            else:
                seed_meta = _failure_seed_meta(
                    task,
                    uid=request.uid,
                    status=status,
                    error=error,
                    elapsed_sec=elapsed_sec,
                )
            meta = task_meta[int(idx)] if 0 <= int(idx) < len(task_meta) else None
            _observe_final_seed(meta, seed_meta)
            _emit_seed_complete(on_seed_complete, seed_meta)

        for idx in request.batch_indices:
            vr = ValidationResult(int(uid), False, 0.0, 0.0)
            results[idx] = vr
            meta = task_meta[idx] if idx < len(task_meta) else None
            _record_seed_result(
                vr,
                meta,
                status=str(status),
                is_runtime_failure=True,
            )

        batch_seed_meta.pop(int(request.batch_index), None)

        worker_active_requests.pop(worker_slot, None)
        worker_last_heartbeat.pop(worker_slot, None)
        worker_started_at.pop(worker_slot, None)

    def _check_for_stalled_workers() -> int:
        completed_now = 0
        now = time.time()
        for worker_slot, request in list(worker_active_requests.items()):
            last_hb = worker_last_heartbeat.get(worker_slot)
            if last_hb is None or (now - last_hb) < stall_timeout_sec:
                continue
            elapsed_sec = max(0.0, now - worker_started_at.get(worker_slot, now))
            _complete_failed_request(
                worker_slot,
                request=request,
                status="worker_stall_timeout",
                error=f"worker {worker_slot} heartbeat stalled for {elapsed_sec:.1f}s",
                elapsed_sec=elapsed_sec,
            )
            completed_now += 1
            _restart_worker(worker_slot)
            _dispatch_available_batches()
        return completed_now

    _TYPE_PREFIX = re.compile(r"^type\d+_")
    seed_stats: dict[str, Any] = {
        "ok": 0,
        "failed": 0,
        "timeout": 0,
        "runtime": 0,
        "retried_timeout": 0,
        "scores": [],
        "per_type": {},
    }
    last_summary_chunk_done = 0

    def _type_name(meta: Optional[dict]) -> str:
        raw = meta.get("group", "unknown") if meta else "unknown"
        return _TYPE_PREFIX.sub("", raw)

    def _seed_label(meta: Optional[dict]) -> str:
        seed_id = meta.get("seed", "?") if meta else "?"
        return f"{_type_name(meta)}:{seed_id}"

    def _record_seed_result(
        result_obj: Any,
        meta: Optional[dict],
        *,
        status: str,
        is_runtime_failure: bool = False,
    ) -> None:
        if result_obj is None:
            result_obj = ValidationResult(int(uid), False, 0.0, 0.0)
        score = float(result_obj.score) if result_obj else 0.0
        seed_stats["scores"].append(score)
        tname = _type_name(meta)
        if tname not in seed_stats["per_type"]:
            seed_stats["per_type"][tname] = []
        seed_stats["per_type"][tname].append(score)
        bucket = _summary_bucket_for_result(
            bench_engine,
            status=str(status),
            result_obj=result_obj,
            is_infra_failure=is_runtime_failure,
        )
        seed_stats[bucket] += 1
        if bucket == "timeout":
            seed_stats.setdefault("timeout_seeds", []).append(_seed_label(meta))
        elif bucket == "runtime":
            seed_stats.setdefault("runtime_seeds", []).append(_seed_label(meta))

    def _record_timeout_retry(meta: Optional[dict]) -> None:
        seed_stats["retried_timeout"] += 1
        seed_stats.setdefault("retried_timeout_seeds", []).append(_seed_label(meta))

    def _log_summary() -> None:
        nonlocal last_summary_chunk_done
        chunk_done = len(seed_stats["scores"])
        if chunk_done == 0 or chunk_done == last_summary_chunk_done:
            return
        overall_done = prior_seeds_done + chunk_done
        overall_total = prior_total_seeds if prior_total_seeds > 0 else len(all_tasks)
        chunk_sum = sum(seed_stats["scores"])
        prior_sum = prior_avg * prior_seeds_done if prior_seeds_done > 0 else 0.0
        overall_avg = (prior_sum + chunk_sum) / overall_done if overall_done > 0 else 0.0
        type_parts = " ".join(
            f"{t}={sum(s)/len(s):.2f}"
            for t, s in sorted(seed_stats["per_type"].items()) if s
        )
        active = len(worker_active_requests)
        counts = (
            f"{seed_stats['ok']} ok, {seed_stats['failed']} failed, "
            f"{seed_stats['timeout']} timeout, {seed_stats['runtime']} runtime, "
            f"{seed_stats['retried_timeout']} retried_timeout"
        )
        line = (
            f"[{phase_label} {overall_done}/{overall_total}] avg={overall_avg:.4f} | "
            f"{counts} | {type_parts} | {active}/{effective_workers} workers "
            f"({scheduler.format_live_status_line()})"
        )
        bt.logging.info(line)
        last_summary_chunk_done = chunk_done
        if seed_stats.get("retried_timeout_seeds"):
            bt.logging.info(
                f"  retried_timeouts: {', '.join(seed_stats['retried_timeout_seeds'])}"
            )
        if seed_stats.get("timeout_seeds"):
            bt.logging.info(f"  timeouts: {', '.join(seed_stats['timeout_seeds'])}")
        if seed_stats.get("runtime_seeds"):
            bt.logging.info(
                f"  runtime_failures: {', '.join(seed_stats['runtime_seeds'])}"
            )

    last_summary_time = time.time()

    try:
        for worker_slot in range(effective_workers):
            _spawn_worker(worker_slot)
        _maybe_poll_scheduler(force=True)
        _dispatch_available_batches()

        completed_batches = 0
        while completed_batches < len(batch_plan):
            _drain_progress_events()
            _maybe_poll_scheduler()
            completed_batches += _check_for_stalled_workers()
            if completed_batches >= len(batch_plan):
                break

            try:
                payload = result_queue.get(timeout=0.2)
            except queue_mod.Empty:
                _maybe_poll_scheduler()
                now = time.time()
                for worker_slot, worker in list(workers.items()):
                    if worker.is_alive() or worker.exitcode in (0, None):
                        continue
                    request = worker_active_requests.get(worker_slot)
                    if request is None:
                        bt.logging.warning(
                            f"[Validator eval] idle worker {worker_slot} crashed "
                            f"(exitcode={worker.exitcode}); restarting"
                        )
                        _restart_worker(worker_slot)
                        continue
                    elapsed_sec = max(0.0, now - worker_started_at.get(worker_slot, now))
                    _complete_failed_request(
                        worker_slot,
                        request=request,
                        status="batch_exception",
                        error=f"worker crashed (exitcode={worker.exitcode})",
                        elapsed_sec=elapsed_sec,
                    )
                    completed_batches += 1
                    _restart_worker(worker_slot)
                    _dispatch_available_batches()
                await asyncio.sleep(0)
                continue

            _drain_progress_events()
            _maybe_poll_scheduler()
            completed_batches += _check_for_stalled_workers()
            if completed_batches >= len(batch_plan):
                break

            worker_slot = int(payload.worker_id)
            request = worker_active_requests.get(worker_slot)
            if request is None or int(payload.batch_index) != int(request.batch_index):
                bt.logging.warning(
                    f"[Validator eval] ignoring late result from worker {worker_slot} "
                    f"for batch {int(payload.batch_index) + 1}"
                )
                continue

            if payload.error:
                _complete_failed_request(
                    worker_slot,
                    request=request,
                    status="batch_exception",
                    error=str(payload.error),
                    elapsed_sec=float(payload.elapsed_sec),
                )
                completed_batches += 1
                _restart_worker(worker_slot)
                _dispatch_available_batches()
                continue

            if len(payload.results) != len(request.batch_indices):
                _complete_failed_request(
                    worker_slot,
                    request=request,
                    status="batch_exception",
                    error=(
                        f"unexpected result count {len(payload.results)} "
                        f"for batch of {len(request.batch_indices)} seeds"
                    ),
                    elapsed_sec=float(payload.elapsed_sec),
                )
                completed_batches += 1
                _restart_worker(worker_slot)
                _dispatch_available_batches()
                continue

            prebuilt_seed_meta: dict[int, Dict[str, Any]] = {}
            if len(request.batch_indices) == 1:
                idx = int(request.batch_indices[0])
                vr = ValidationResult(*payload.results[0])
                meta = task_meta[idx] if idx < len(task_meta) else None
                final_seed_meta = _pop_batch_seed_meta(
                    batch_seed_meta,
                    batch_index=int(request.batch_index),
                    meta=meta,
                    uid=uid,
                    result_obj=vr,
                )
                final_status = _seed_status(final_seed_meta)
                prior_retries = int(batch_retry_counts.get(int(request.batch_index), 0))
                if (
                    bench_engine._is_backoff_timeout_status(final_status)
                    and prior_retries < 1
                    and timeout_retries_used < _MAX_TIMEOUT_RETRIES
                ):
                    timeout_retries_used += 1
                    batch_retry_counts[int(request.batch_index)] = prior_retries + 1
                    _record_timeout_retry(meta)
                    bt.logging.warning(
                        f"[Validator eval] retrying timed-out seed {_seed_label(meta)} "
                        f"for UID {uid} (retry {prior_retries + 1}/1, "
                        f"global retries {timeout_retries_used}/{_MAX_TIMEOUT_RETRIES}, "
                        f"status={final_status})"
                    )
                    worker_active_requests.pop(worker_slot, None)
                    worker_last_heartbeat.pop(worker_slot, None)
                    worker_started_at.pop(worker_slot, None)
                    pending_batch_ids.append(int(request.batch_index))
                    _dispatch_available_batches()
                    continue
                prebuilt_seed_meta[idx] = final_seed_meta

            for idx, packed in zip(request.batch_indices, payload.results):
                vr = ValidationResult(*packed)
                results[idx] = vr
                meta = task_meta[idx] if idx < len(task_meta) else None
                final_seed_meta = prebuilt_seed_meta.pop(int(idx), None)
                if final_seed_meta is None:
                    final_seed_meta = _pop_batch_seed_meta(
                        batch_seed_meta,
                        batch_index=int(request.batch_index),
                        meta=meta,
                        uid=uid,
                        result_obj=vr,
                    )
                _observe_final_seed(meta, final_seed_meta)
                _emit_seed_complete(on_seed_complete, final_seed_meta)
                _record_seed_result(
                    vr,
                    meta,
                    status=_seed_status(final_seed_meta),
                )

            worker_active_requests.pop(worker_slot, None)
            worker_last_heartbeat.pop(worker_slot, None)
            worker_started_at.pop(worker_slot, None)
            completed_batches += 1
            _maybe_poll_scheduler()
            _dispatch_available_batches()

            now_ts = time.time()
            if now_ts - last_summary_time >= EVAL_SUMMARY_INTERVAL_SEC:
                _log_summary()
                last_summary_time = now_ts

        _drain_progress_events()
        _log_summary()
        return [
            result if result is not None else ValidationResult(int(uid), False, 0.0, 0.0)
            for result in results
        ]
    finally:
        for worker_slot, queue_obj in list(worker_queues.items()):
            try:
                queue_obj.put(None)
            except Exception:
                pass
        for worker in workers.values():
            try:
                worker.join(timeout=5.0)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=2.0)
            except Exception:
                pass
        for queue_obj in worker_queues.values():
            _close_queue(queue_obj)
        for queue_obj in (result_queue, progress_queue):
            _close_queue(queue_obj)


async def evaluate_seeds_parallel(
    self,
    tasks: list,
    uid: int,
    model_path: Path,
    num_workers: int = None,
    on_seed_complete: Optional[Callable[..., None]] = None,
    phase_label: str = "eval",
    prior_seeds_done: int = 0,
    prior_total_seeds: int = 0,
    prior_avg: float = 0.0,
) -> list:
    """Evaluate validator seeds using the benchmark-grade process scheduler."""
    if not tasks:
        return []

    if num_workers is None:
        num_workers = N_DOCKER_WORKERS
    effective_workers = max(1, min(int(num_workers), len(tasks)))
    runtime_tracker = getattr(self, "runtime_tracker", None)
    tracker_call(
        runtime_tracker,
        "mark_docker_run_started",
        requested_workers=int(num_workers),
        effective_workers=int(effective_workers),
        total_tasks=int(len(tasks)),
    )

    if not _docker_evaluator_facade().DockerSecureEvaluator._base_ready:
        return await self.evaluate_seeds_batch(
            tasks,
            uid,
            model_path,
            worker_id=0,
            on_seed_complete=on_seed_complete,
            task_offset=0,
            task_total=len(tasks),
        )

    from .batch import prepare_model_image, remove_model_image

    model_image = prepare_model_image(self, uid, model_path)

    task_meta = [
        {
            "group": _task_group_name(task, index),
            "seed": _task_seed(task, index),
            "challenge_type": _task_challenge_type(task),
            "horizon": float(getattr(task, "horizon", 0.0)),
            "moving_platform": bool(getattr(task, "moving_platform", False)),
        }
        for index, task in enumerate(tasks)
    ]
    batch_plan = _benchmark_engine()._batch_indices(len(tasks))

    try:
        return await _run_process_parallel(
            all_tasks=list(tasks),
            task_meta=task_meta,
            batch_plan=batch_plan,
            uid=uid,
            model_path=model_path,
            effective_workers=effective_workers,
            runtime_tracker=runtime_tracker,
            on_seed_complete=on_seed_complete,
            heartbeat_sec=30.0,
            phase_label=phase_label,
            prior_seeds_done=prior_seeds_done,
            prior_total_seeds=prior_total_seeds,
            prior_avg=prior_avg,
            model_image=model_image,
        )
    finally:
        if model_image:
            remove_model_image(model_image)


__all__ = [name for name in globals() if not name.startswith("__")]
