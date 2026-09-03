# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import bittensor as bt
import numpy as np

from swarm.challenge_families import (
    DEFAULT_RUNTIME_FAMILY_ID,
    build_benchmark_tasks,
    build_random_task,
    build_screening_tasks,
)
from swarm.constants import (
    BENCHMARK_FULL_SEED_COUNT,
    BENCHMARK_SCREENING_SEED_COUNT,
    BENCHMARK_VERSION,
    MAX_INFLIGHT_SEED_UPLOADS,
    RE_AUTH_INTERVAL_SEC,
    SIM_DT,
    UNIFIED_CHUNK_SIZE,
)
from swarm.domain_model import CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE, ENVIRONMENT_TYPES
from swarm.core.faults import EvaluationFault, INFRA_FAULT_CODES, ReasonCode
from swarm.core.submission_policy import (
    EXECUTION_PROFILE_ID,
    RUNNER_ABI,
    profile_digest,
)
from swarm.protocol import FailureReason
from swarm.utils.hash import sha256sum
from swarm.validator.backend_api import BackendTransportError, authorize_with_retry
from swarm.validator.runtime_telemetry import tracker_call

from .heartbeat import HeartbeatManager


_EMPTY_PER_TYPE = tuple(ENVIRONMENT_TYPES) + ("moving_platform",)

_INFRA_FAILURE_REASONS = frozenset(
    {FailureReason.INFRA.value} | {code.value for code in INFRA_FAULT_CODES}
)


def _is_infra_failure(reason) -> bool:
    """Infrastructure faults are the validator's problem and are never uploaded as miner scores."""
    return reason in _INFRA_FAILURE_REASONS


def _seed_upload_provenance(self, model_path: Path) -> Dict[str, Any]:
    """Provenance fields the backend seed-score schema requires on every upload.

    ``runner_image_digest`` carries the runner image's build-input fingerprint
    (the ``swarm.code_hash`` label), not an OCI digest. An absent or stale label
    means the scores cannot be attributed to a known image, so the upload is
    refused as an infrastructure fault rather than stamped with a placeholder."""
    evaluator = self.docker_evaluator
    try:
        label = str(evaluator._get_image_hash_label() or "")
        expected = str(evaluator._calculate_docker_hash())
    except Exception as exc:
        raise EvaluationFault(
            ReasonCode.INFRA_IMAGE_MISMATCH, f"runner image provenance unavailable: {exc}"
        ) from exc
    if not label or label != expected:
        raise EvaluationFault(
            ReasonCode.INFRA_IMAGE_MISMATCH,
            f"runner image label {label!r} does not match expected build hash {expected!r}",
        )
    return {
        "artifact_sha256": sha256sum(model_path),
        "execution_profile_id": EXECUTION_PROFILE_ID,
        "execution_profile_digest": profile_digest(),
        "runner_abi": RUNNER_ABI,
        "runner_image_digest": label,
    }


def _utils_facade():
    from swarm.validator import utils as validator_utils

    return validator_utils


def _empty_per_type() -> Dict[str, List[float]]:
    return {name: [] for name in _EMPTY_PER_TYPE}


def _seed_manager_call(seed_manager, method_name: str, family_id: str, epoch: Optional[int] = None):
    method = getattr(seed_manager, method_name)
    if epoch is not None and epoch > getattr(seed_manager, "epoch_number", 0):
        # A pre-evaluation task is scored on its own epoch's seeds, not today's.
        return method(family_id=family_id, epoch=epoch)
    try:
        return method(family_id=family_id)
    except TypeError:
        return method()


async def _evaluate_seeds(
    self,
    uid: int,
    model_path: Path,
    seeds: List[int],
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    description: str = "benchmark",
    on_seed_complete: Optional[Callable[[], None]] = None,
    on_seed_result: Optional[Callable[[int, Any, str], None]] = None,
    should_stop: Optional[Callable[[], Optional[str]]] = None,
    prior_seeds_done: int = 0,
    prior_total_seeds: int = 0,
    prior_avg: float = 0.0,
    pre_built_tasks: Optional[List] = None,
    retry_budget: Optional[Dict[str, int]] = None,
    seed_feeder: Optional[Callable[[int], Any]] = None,
    initial_pending: Optional[List[int]] = None,
    on_held_seeds: Optional[Callable[[List[int]], None]] = None,
) -> Tuple[List[float], Dict[str, List[float]], List[dict]]:
    """Evaluate a model on multiple seeds using parallel Docker containers.

    ``on_seed_result`` fires once per finally-scored seed with the seed's
    position in ``seeds`` and its detail dict (score, map_type, metric_key,
    failure_reason, moving_platform). ``should_stop`` is polled by the
    dispatcher; a non-None reason stops new dispatches, in-flight seeds
    finish, and undispatched seeds are skipped (returned lists then cover
    only the evaluated seeds)."""
    all_scores = []
    per_type_scores = _empty_per_type()

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
                task = build_random_task(
                    sim_dt=SIM_DT,
                    seed=seed,
                    family_id=family_id,
                )
                tasks.append(task)
            except Exception as e:
                bt.logging.warning(f"Failed to create a seed task: {e}")
                tasks.append(None)

    valid_tasks = [t for t in tasks if t is not None]
    if not valid_tasks:
        bt.logging.warning(f"No valid tasks created for UID {uid}")
        return [], per_type_scores, []

    valid_positions = [i for i, t in enumerate(tasks) if t is not None]

    def _forward_seed_result(valid_idx: int, result: Any, status: str) -> None:
        if on_seed_result is None or result is None:
            return
        if not (0 <= int(valid_idx) < len(valid_positions)):
            return
        position = valid_positions[int(valid_idx)]
        task = tasks[position]
        type_name = CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE.get(
            task.challenge_type, "unknown"
        )
        on_seed_result(
            position,
            {
                "score": float(getattr(result, "score", 0.0)),
                "metric_key": type_name,
                "map_type": type_name,
                "failure_reason": str(
                    getattr(result, "failure_reason", "NONE") or "NONE"
                ),
                "moving_platform": bool(getattr(task, "moving_platform", False)),
            },
        )

    engine_feeder = None
    engine_initial: Optional[List[int]] = None
    engine_held: Optional[Callable[[List[int]], None]] = None
    if seed_feeder is not None:
        # The feeder speaks absolute seed indexes; the engine indexes valid_tasks.
        abs_to_valid = {pos: vi for vi, pos in enumerate(valid_positions)}

        if on_held_seeds is not None:
            def engine_held(valid_indexes: List[int]) -> None:
                on_held_seeds([
                    valid_positions[int(i)]
                    for i in valid_indexes
                    if 0 <= int(i) < len(valid_positions)
                ])

        async def engine_feeder(free_slots: int):
            granted, drained = await seed_feeder(free_slots)
            return (
                [abs_to_valid[int(i)] for i in granted if int(i) in abs_to_valid],
                drained,
            )

        engine_initial = [
            abs_to_valid[int(i)]
            for i in (initial_pending or [])
            if int(i) in abs_to_valid
        ]

    phase = "screening" if "screening" in description.lower() else "benchmark"
    results = await self.docker_evaluator.evaluate_seeds_parallel(
        tasks=valid_tasks,
        uid=uid,
        model_path=model_path,
        on_seed_complete=on_seed_complete,
        on_seed_result=_forward_seed_result if on_seed_result is not None else None,
        should_stop=should_stop,
        phase_label=phase,
        prior_seeds_done=prior_seeds_done,
        prior_total_seeds=prior_total_seeds,
        prior_avg=prior_avg,
        retry_budget=retry_budget,
        seed_feeder=engine_feeder,
        initial_pending=engine_initial,
        on_held_seeds=engine_held,
    )

    seed_details = []
    task_idx = 0
    for i, task in enumerate(tasks):
        if task is None:
            all_scores.append(0.0)
            seed_details.append(
                {
                    "score": 0.0,
                    "metric_key": "unknown",
                    "map_type": "unknown",
                    "failure_reason": FailureReason.INFRA.value,
                    "metrics": {},
                }
            )
            continue

        if task_idx < len(results):
            result = results[task_idx]
            task_idx += 1
            if result is None:
                # Skipped after a stop request: not evaluated, not scored.
                continue
            score = result.score
            reason = getattr(result, "failure_reason", "NONE")
            metrics = dict(getattr(result, "metrics", {}) or {})
            all_scores.append(score)

            type_name = CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE.get(
                task.challenge_type,
                "unknown",
            )
            if getattr(task, "moving_platform", False):
                per_type_scores["moving_platform"].append(score)
            elif type_name in per_type_scores:
                per_type_scores[type_name].append(score)

            seed_details.append(
                {
                    "score": score,
                    "metric_key": type_name,
                    "map_type": type_name,
                    "failure_reason": reason,
                    "metrics": metrics,
                }
            )
        else:
            type_name = CHALLENGE_TYPE_TO_ENVIRONMENT_TYPE.get(
                task.challenge_type,
                "unknown",
            )
            all_scores.append(0.0)
            seed_details.append(
                {
                    "score": 0.0,
                    "metric_key": type_name,
                    "map_type": type_name,
                    "failure_reason": FailureReason.INFRA.value,
                    "metrics": {},
                }
            )

    bt.logging.info(f"✅ {description} complete for UID {uid}: {len(all_scores)} seeds evaluated")
    return all_scores, per_type_scores, seed_details


async def _run_streaming_phase(
    self,
    uid: int,
    model_path: Path,
    seeds: List[int],
    *,
    phase_description: str,
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
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
    re_auth_interval_sec: float = RE_AUTH_INTERVAL_SEC,
    seed_feeder: Optional[Callable[[int], Any]] = None,
    initial_pending: Optional[List[int]] = None,
) -> Tuple[List[float], Dict[str, List[float]], List[dict], Optional[str]]:
    """Evaluate the full seed range as one rolling queue with streamed uploads.

    Every seed goes to the parallel evaluator in a single call: a worker picks
    up the next pending seed the moment it frees, with no barrier between
    upload groups. Completed scores upload in groups of ``chunk_size`` as they
    arrive (fire-and-forget, capped at ``max_inflight``). ``re_authorize``
    (when given) re-checks the task every ``re_auth_interval_sec`` alongside
    evaluation; a denial — like a backend stop via ``should_stop`` — halts new
    seed dispatches, lets in-flight seeds finish, and returns the accumulated
    partials with the cancel reason.
    """
    all_scores: List[float] = []
    all_per_type: Dict[str, List[float]] = _empty_per_type()
    all_details: List[dict] = []
    inflight: List[asyncio.Task] = []
    failed_batches: List[List[dict]] = []
    # Scored here but not yet acknowledged by the backend; still ours to hold.
    unacked: set = set()
    retry_budget: Dict[str, int] = {"timeout": 0, "rpc_transport": 0}
    total_for_evaluator = (
        evaluator_total_seeds if evaluator_total_seeds is not None else len(seeds)
    )
    provenance = _seed_upload_provenance(self, model_path)

    async def _safe_upload(batch: List[dict]) -> None:
        for delay in (0.0, 2.0, 4.0):
            if delay:
                await asyncio.sleep(delay)
            try:
                result = await self.backend_api.post_seed_scores_batch(
                    model_uid=uid, epoch_number=epoch_number, scores=batch,
                    task_id=task_id,
                    family_id=family_id,
                    provenance=provenance,
                )
            except Exception as exc:
                bt.logging.warning(f"Seed score upload failed for UID {uid}: {exc}")
                continue
            if result and result.get("recorded"):
                unacked.difference_update(int(row["seed_index"]) for row in batch)
                return
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

    upload_queue: asyncio.Queue = asyncio.Queue()

    async def _upload_pump() -> None:
        group: List[dict] = []
        while True:
            row = await upload_queue.get()
            if row is None:
                break
            group.append(row)
            if len(group) >= chunk_size:
                batch, group = group, []
                await _wait_for_slot()
                inflight.append(asyncio.create_task(_safe_upload(batch)))
        if group:
            await _wait_for_slot()
            inflight.append(asyncio.create_task(_safe_upload(group)))

    stop_state: Dict[str, Any] = {"cancel": None, "raise": None}
    seen: set = set()
    completed_scores: List[float] = []
    window: Dict[str, Any] = {
        "scores": [], "per_type": _empty_per_type(), "details": [],
    }

    def _fire_chunk_complete() -> None:
        if on_chunk_complete is None or not window["scores"]:
            return
        chunk_scores = window["scores"]
        chunk_per_type = window["per_type"]
        chunk_details = window["details"]
        window["scores"] = []
        window["per_type"] = _empty_per_type()
        window["details"] = []
        try:
            on_chunk_complete(
                evaluated=len(completed_scores),
                total=len(seeds),
                running_avg=(
                    float(np.mean(completed_scores)) if completed_scores else 0.0
                ),
                chunk_scores=list(chunk_scores),
                chunk_per_type={k: list(v) for k, v in chunk_per_type.items()},
                chunk_details=list(chunk_details),
            )
        except Exception as exc:
            bt.logging.warning(f"on_chunk_complete callback failed for UID {uid}: {exc}")

    def _on_result(idx: int, detail: dict) -> None:
        if idx in seen or not isinstance(detail, dict):
            return
        seen.add(idx)
        score = float(detail.get("score", 0.0))
        reason = str(detail.get("failure_reason", "NONE") or "NONE")
        type_name = str(detail.get("metric_key") or detail.get("map_type") or "unknown")
        completed_scores.append(score)
        window["scores"].append(score)
        if detail.get("moving_platform"):
            window["per_type"]["moving_platform"].append(score)
        elif type_name in window["per_type"]:
            window["per_type"][type_name].append(score)
        window["details"].append(
            {
                "score": score,
                "metric_key": type_name,
                "map_type": type_name,
                "failure_reason": reason,
            }
        )
        if type_name != "unknown" and not _is_infra_failure(reason):
            unacked.add(seed_offset + idx)
            upload_queue.put_nowait(
                {
                    "seed_index": seed_offset + idx,
                    "score": score,
                    "metric_key": type_name,
                    "map_type": type_name,
                    "failure_reason": reason,
                }
            )
        if len(completed_scores) % chunk_size == 0:
            _fire_chunk_complete()

    def _combined_stop() -> Optional[str]:
        if stop_state["cancel"] is not None:
            return stop_state["cancel"]
        if should_stop is not None:
            reason = should_stop()
            if reason:
                stop_state["cancel"] = f"backend stop_required: {reason}"
                return stop_state["cancel"]
        return None

    reauth_task: Optional[asyncio.Task] = None
    if re_authorize is not None:
        async def _reauth_loop() -> None:
            while stop_state["cancel"] is None:
                await asyncio.sleep(re_auth_interval_sec)
                try:
                    auth = await authorize_with_retry(
                        re_authorize,
                        log_prefix=f"UID {uid} mid-{phase_description}: ",
                    )
                except BackendTransportError as exc:
                    stop_state["raise"] = exc
                    stop_state["cancel"] = str(exc)
                    return
                if not auth.get("authorized"):
                    stop_state["cancel"] = str(auth.get("reason") or "unauthorized")
                    return

        reauth_task = asyncio.create_task(_reauth_loop())

    pump_task = asyncio.create_task(_upload_pump())

    try:
        all_scores, all_per_type, all_details = await _utils_facade()._evaluate_seeds(
            self,
            uid,
            model_path,
            seeds,
            family_id=family_id,
            description=phase_description,
            on_seed_complete=hb.on_seed_complete,
            on_seed_result=_on_result,
            should_stop=_combined_stop,
            prior_seeds_done=evaluator_prior_done,
            prior_total_seeds=total_for_evaluator,
            prior_avg=0.0,
            pre_built_tasks=pre_built_tasks,
            retry_budget=retry_budget,
            seed_feeder=seed_feeder,
            initial_pending=initial_pending,
            on_held_seeds=(
                None if seed_feeder is None
                else lambda held: hb.set_in_flight(sorted(
                    {seed_offset + int(position) for position in held} | unacked
                ))
            ),
        )
        _fire_chunk_complete()
    finally:
        if reauth_task is not None:
            reauth_task.cancel()
            try:
                await reauth_task
            except (asyncio.CancelledError, Exception):
                pass
        upload_queue.put_nowait(None)
        try:
            await pump_task
        except Exception as exc:
            bt.logging.warning(f"Seed score upload pump failed for UID {uid}: {exc}")
        if not seen and all_details:
            rows = [
                {
                    "seed_index": seed_offset + j,
                    "score": detail["score"],
                    "metric_key": detail.get("metric_key") or detail["map_type"],
                    "map_type": detail["map_type"],
                    "failure_reason": detail.get("failure_reason", "NONE"),
                }
                for j, detail in enumerate(all_details)
                if (detail.get("metric_key") or detail.get("map_type")) != "unknown"
                and not _is_infra_failure(detail.get("failure_reason"))
            ]
            for start in range(0, len(rows), chunk_size):
                await _wait_for_slot()
                inflight.append(
                    asyncio.create_task(_safe_upload(rows[start:start + chunk_size]))
                )
        await _drain_inflight()
        if failed_batches:
            retry_queue = list(failed_batches)
            failed_batches.clear()
            for batch in retry_queue:
                try:
                    result = await self.backend_api.post_seed_scores_batch(
                        model_uid=uid, epoch_number=epoch_number, scores=batch,
                        task_id=task_id,
                        family_id=family_id,
                        provenance=provenance,
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
        if seed_feeder is not None:
            # Every score is uploaded by now, so anything still leased was dropped.
            hb.set_in_flight([])

    if stop_state["raise"] is not None:
        raise stop_state["raise"]
    return all_scores, all_per_type, all_details, stop_state["cancel"]


async def _run_screening(
    self, uid: int, model_path: Path, reeval: bool = False,
    task_id: Optional[int] = None,
    *,
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    seeds_from: int = 0,
    seeds_to: Optional[int] = None,
    cancel_flag: Optional[asyncio.Event] = None,
    batch_id: Optional[int] = None,
    seed_feeder: Optional[Callable[[int], Any]] = None,
) -> Tuple[float, List[float], Dict[str, List[float]], Optional[str], bool]:
    """Run screening seeds and stream per-seed scores.

    Returns ``(avg, all_scores, per_type, cancel_reason, early_failed)`` with
    ``early_failed`` always ``False`` — pass/fail and copy-detection are
    decided by the backend from the streamed seed scores.

    ``seeds_from`` / ``seeds_to`` carve a sub-range out of the epoch's full
    screening seed list — used for resuming an interrupted task. ``cancel_flag``
    is set by the SSE listener; when set the streaming phase stops dispatching
    at the next seed.
    """
    full_seeds = _seed_manager_call(
        self.seed_manager, "get_screening_seeds", family_id
    )
    upper = seeds_to if seeds_to is not None else len(full_seeds)
    upper = max(seeds_from, min(upper, len(full_seeds)))
    screening_seeds = full_seeds[seeds_from:upper]
    total_seeds = len(screening_seeds)
    # Cumulative-progress framing for the heartbeat: report the full
    # screening range as total and the resume offset as already-done so
    # the dashboard renders honest progress (e.g. 130/200) instead of
    # 0/(remaining-slice) on resume.
    heartbeat_total = upper
    progress_offset = seeds_from
    epoch = self.seed_manager.epoch_number

    screening_tasks: List = []
    try:
        screening_tasks = build_screening_tasks(
            sim_dt=SIM_DT,
            seeds=screening_seeds,
            family_id=family_id,
            offset=seeds_from,
            total_seed_count=len(full_seeds),
        )
    except Exception as e:
        bt.logging.warning(f"Failed to create screening tasks: {e}")
        screening_tasks = [None for _ in screening_seeds]

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
        "family_id": family_id,
        "epoch_number": epoch,
        "benchmark_version": BENCHMARK_VERSION,
        "batch_id": batch_id,
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
        evaluated = int(info["evaluated"])
        running_avg = float(info["running_avg"])
        tracker_call(
            self,
            "mark_screening_progress",
            uid=int(uid),
            progress=evaluated,
            total_seeds=int(info["total"]),
            running_median=running_avg,
            note=f"checkpoint {evaluated}/{info['total']}",
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
            family_id=family_id,
            seed_offset=seeds_from,
            epoch_number=epoch,
            hb=hb,
            task_id=task_id,
            pre_built_tasks=screening_tasks,
            should_stop=_should_stop,
            on_chunk_complete=_on_chunk,
            chunk_size=2 if seed_feeder is not None else UNIFIED_CHUNK_SIZE,
            seed_feeder=seed_feeder,
            initial_pending=[] if seed_feeder is not None else None,
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
    family_id: str = DEFAULT_RUNTIME_FAMILY_ID,
    cancel_flag: Optional[asyncio.Event] = None,
    seeds_from: Optional[int] = None,
    seeds_to: Optional[int] = None,
    batch_id: Optional[int] = None,
    seed_feeder: Optional[Callable[[int], Any]] = None,
    epoch_number: Optional[int] = None,
) -> Tuple[float, Dict[str, float], List[float], Dict[str, List[float]], Optional[str]]:
    """Run full benchmark. Uses benchmark seeds by default, or custom seeds if provided.

    Returns (avg_score, per_type_avgs, all_scores, per_type_raw, cancel_reason).

    When ``reeval`` is True the heartbeat labels the active task as REEVAL; a
    stale champion re-eval is halted mid-flight via ``cancel_flag`` (SSE) or a
    heartbeat stop, both polled before every seed dispatch.
    """
    if seeds is None:
        # A range starting below the screening boundary (REEVAL, or BENCHMARK when the backend runs without screening) spans the full seed list.
        if seeds_from is not None and seeds_from < BENCHMARK_SCREENING_SEED_COUNT:
            all_family_seeds = _seed_manager_call(
                self.seed_manager, "get_all_seeds", family_id, epoch_number
            )
            benchmark_seeds = all_family_seeds[seeds_from:]
            seed_offset = seeds_from
            heartbeat_total = len(all_family_seeds)
            progress_offset = seeds_from
        elif seeds_from is not None and seeds_from > BENCHMARK_SCREENING_SEED_COUNT:
            full_benchmark = _seed_manager_call(
                self.seed_manager, "get_benchmark_seeds", family_id, epoch_number
            )
            offset = seeds_from - BENCHMARK_SCREENING_SEED_COUNT
            benchmark_seeds = full_benchmark[offset:]
            seed_offset = seeds_from
            heartbeat_total = len(full_benchmark)
            progress_offset = offset
        else:
            benchmark_seeds = _seed_manager_call(
                self.seed_manager, "get_benchmark_seeds", family_id, epoch_number
            )
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

    # Template each seed by absolute index (screening below the boundary, benchmark above); custom-seed runs keep random.
    pre_built_tasks: Optional[List] = None
    if seeds is None:
        full_screening_seeds = None
        pre_built_tasks = []
        for i, seed in enumerate(benchmark_seeds):
            abs_idx = seed_offset + i
            try:
                if abs_idx < BENCHMARK_SCREENING_SEED_COUNT:
                    if full_screening_seeds is None:
                        full_screening_seeds = _seed_manager_call(
                            self.seed_manager, "get_screening_seeds", family_id, epoch_number
                        )
                    task = build_screening_tasks(
                        sim_dt=SIM_DT, seeds=[seed], family_id=family_id,
                        offset=abs_idx, total_seed_count=len(full_screening_seeds),
                    )[0]
                else:
                    task = build_benchmark_tasks(
                        sim_dt=SIM_DT, seeds=[seed], family_id=family_id,
                        offset=abs_idx - BENCHMARK_SCREENING_SEED_COUNT,
                        total_seed_count=BENCHMARK_FULL_SEED_COUNT,
                    )[0]
            except Exception as e:
                bt.logging.warning(
                    f"Failed to create benchmark task idx {abs_idx}: {e}"
                )
                task = None
            pre_built_tasks.append(task)

    total_seeds = len(benchmark_seeds)
    note = "full benchmark" if seeds is None else "custom seeds"
    # The epoch that leased these seeds, not today's, or the leases never complete.
    epoch = int(
        epoch_number if epoch_number is not None else self.seed_manager.epoch_number
    )

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
        "family_id": family_id,
        "epoch_number": epoch,
        "benchmark_version": BENCHMARK_VERSION,
        "batch_id": batch_id,
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
            phase_description="seed-flow benchmark" if seed_feeder is not None else "full benchmark",
            family_id=family_id,
            seed_offset=seed_offset,
            epoch_number=epoch,
            hb=hb,
            task_id=task_id,
            pre_built_tasks=pre_built_tasks,
            should_stop=_should_stop,
            on_chunk_complete=_on_chunk,
            chunk_size=2 if seed_feeder is not None else UNIFIED_CHUNK_SIZE,
            seed_feeder=seed_feeder,
            initial_pending=[] if seed_feeder is not None else None,
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
