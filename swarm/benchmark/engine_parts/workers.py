from __future__ import annotations

from ._shared import (
    BENCH_GROUP_TO_TYPE,
    Any,
    Dict,
    List,
    Optional,
    Path,
    Tuple,
    _BatchStat,
    _ProcessBatchRequest,
    _ProcessBatchResult,
    _ProcessSeedEvent,
    _ProcessWorkerHeartbeat,
    _RunOptions,
    asyncio,
    deque,
    gc,
    mp,
    queue_mod,
    sys,
    threading,
    time,
    traceback,
)
from .config import _build_progress_bar, _temporary_env, _ts
from .dispatch import (
    _BACKOFF_ACTIVE_WORKERS,
    _PARENT_WORKER_HEARTBEAT_SEC,
    _PARENT_WORKER_STALL_TIMEOUT_SEC,
    _AdaptiveBackoffController,
    _build_worker_stall_seed_meta,
    _is_clean_execution_status,
    _max_heavy_active,
    _select_next_batch_index,
)


def _engine_facade():
    import swarm.benchmark.engine as engine

    return engine


def _benchmark_mp_context() -> mp.context.BaseContext:
    if sys.platform != "win32":
        try:
            return mp.get_context("fork")
        except ValueError:
            pass
    return mp.get_context("spawn")


def _create_prepared_benchmark_evaluator():
    from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator

    evaluator = DockerSecureEvaluator.__new__(DockerSecureEvaluator)
    evaluator.base_image = "swarm_evaluator_base:latest"
    evaluator.last_fake_model_info = None
    evaluator.base_ready = True
    DockerSecureEvaluator._base_ready = True
    return evaluator


def _pack_validation_result(result: Any) -> Tuple[int, bool, float, float]:
    return (
        int(getattr(result, "uid")),
        bool(getattr(result, "success")),
        float(getattr(result, "time_sec")),
        float(getattr(result, "score")),
    )


def _benchmark_worker_main(
    process_slot: int,
    task_queue: Any,
    result_queue: Any,
    progress_queue: Any,
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        evaluator = _engine_facade()._create_prepared_benchmark_evaluator()

        while True:
            request = task_queue.get()
            if request is None:
                return

            batch_start = time.time()
            heartbeat_stop = threading.Event()

            def _on_seed_complete(seed_meta: Optional[Dict[str, Any]] = None) -> None:
                try:
                    progress_queue.put(
                        _ProcessSeedEvent(
                            worker_id=process_slot,
                            batch_index=request.batch_index,
                            seed_meta=seed_meta,
                        )
                    )
                except Exception:
                    pass

            def _emit_worker_heartbeat(event_type: str) -> None:
                try:
                    progress_queue.put(
                        _ProcessWorkerHeartbeat(
                            worker_id=process_slot,
                            batch_index=request.batch_index,
                            event_type=event_type,
                            ts=time.time(),
                        )
                    )
                except Exception:
                    pass

            def _heartbeat_loop() -> None:
                while not heartbeat_stop.wait(
                    timeout=_engine_facade()._PARENT_WORKER_HEARTBEAT_SEC
                ):
                    _emit_worker_heartbeat("heartbeat")

            _emit_worker_heartbeat("batch_started")
            heartbeat_thread = threading.Thread(
                target=_heartbeat_loop,
                name=f"bench_worker_hb_{process_slot}",
                daemon=True,
            )
            heartbeat_thread.start()

            try:
                seed_results = loop.run_until_complete(
                    evaluator.evaluate_seeds_batch(
                        tasks=request.tasks,
                        uid=request.uid,
                        model_path=Path(request.model_path),
                        worker_id=process_slot,
                        on_seed_complete=_on_seed_complete,
                        task_offset=request.batch_indices[0] if request.batch_indices else 0,
                        task_total=request.task_total,
                        model_image=getattr(request, "model_image", None),
                    )
                )
                result_queue.put(
                    _ProcessBatchResult(
                        worker_id=process_slot,
                        batch_index=request.batch_index,
                        batch_indices=list(request.batch_indices),
                        results=[_pack_validation_result(result) for result in seed_results],
                        elapsed_sec=time.time() - batch_start,
                    )
                )
            except Exception as exc:
                result_queue.put(
                    _ProcessBatchResult(
                        worker_id=process_slot,
                        batch_index=request.batch_index,
                        batch_indices=list(request.batch_indices),
                        results=[],
                        elapsed_sec=time.time() - batch_start,
                        error=f"{type(exc).__name__}: {exc}",
                        traceback_text=traceback.format_exc(),
                    )
                )
                return
            finally:
                heartbeat_stop.set()
                heartbeat_thread.join(timeout=1.0)
                gc.collect()
    finally:
        asyncio.set_event_loop(None)
        loop.close()


async def _run_benchmark_process_mode(
    *,
    all_tasks: List[Any],
    task_meta: List[Dict[str, Any]],
    batch_plan: List[List[int]],
    uid: int,
    model_path: Path,
    effective_workers: int,
    record_batch_completion: Any,
    on_seed_done: Any,
    run_opts: _RunOptions,
) -> int:
    engine = _engine_facade()
    ctx = engine._benchmark_mp_context()
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    progress_queue = ctx.Queue()
    workers: Dict[int, Any] = {}
    scheduler = _AdaptiveBackoffController(requested_workers=effective_workers)
    pending_batch_ids: List[int] = list(range(len(batch_plan)))
    inflight_batches: Dict[int, _ProcessBatchRequest] = {}
    worker_active_batches: Dict[int, int] = {}
    worker_last_heartbeat: Dict[int, float] = {}
    worker_started_at: Dict[int, float] = {}
    stall_timeout_sec = max(
        engine._PARENT_WORKER_STALL_TIMEOUT_SEC,
        max(0.0, float(getattr(run_opts, "heartbeat_sec", 0.0))) * 2.0,
    )

    def _spawn_worker(worker_slot: int) -> Any:
        worker = ctx.Process(
            target=_benchmark_worker_main,
            args=(worker_slot, task_queue, result_queue, progress_queue),
            name=f"bench_host_worker_{worker_slot}",
            daemon=True,
        )
        worker.start()
        workers[worker_slot] = worker
        return worker

    print(
        f"[{_ts()}] Dispatch policy: heavy-aware scheduling enabled "
        f"(heavy_groups=mountain,warehouse,forest; "
        f"mountain<=1, warehouse<=1, max_heavy={_max_heavy_active(scheduler.active_worker_cap)})"
    )
    if scheduler.enabled:
        from .dispatch import (
            _BACKOFF_CALIBRATION_CPU_FACTOR_SPIKE,
            _BACKOFF_CALIBRATION_OVERHEAD_SPIKE_SEC,
            _BACKOFF_COOLDOWN_COMPLETIONS,
        )

        print(
            f"[{_ts()}] Adaptive backoff: enabled "
            f"(fallback cap={_BACKOFF_ACTIVE_WORKERS}, cooldown={_BACKOFF_COOLDOWN_COMPLETIONS} seeds, "
            f"calibration spike>={_BACKOFF_CALIBRATION_OVERHEAD_SPIKE_SEC*1000:.0f}ms "
            f"or cpu_factor>={_BACKOFF_CALIBRATION_CPU_FACTOR_SPIKE:.2f}x)"
        )
    else:
        print(f"[{_ts()}] Adaptive backoff: disabled (requested workers <= {_BACKOFF_ACTIVE_WORKERS})")
    print(
        f"[{_ts()}] Parent worker stall watchdog: "
        f"{stall_timeout_sec:.1f}s without worker heartbeat -> discard seed and replace worker"
    )

    def _drain_progress_events() -> None:
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue_mod.Empty:
                return
            if isinstance(event, _ProcessWorkerHeartbeat):
                worker_last_heartbeat[int(event.worker_id)] = float(event.ts)
                if event.event_type == "batch_started":
                    worker_active_batches[int(event.worker_id)] = int(event.batch_index)
                    worker_started_at[int(event.worker_id)] = float(event.ts)
                continue
            seed_meta = getattr(event, "seed_meta", None)
            on_seed_done(seed_meta)
            note = scheduler.observe_seed(seed_meta)
            if note:
                print(f"[{_ts()}] {note}", flush=True)

    def _restart_worker(worker_slot: int) -> None:
        worker = workers.get(worker_slot)
        if worker is not None:
            try:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=2.0)
            except Exception:
                pass
        _spawn_worker(worker_slot)

    def _dispatch_available_batches() -> None:
        while pending_batch_ids and len(inflight_batches) < scheduler.active_worker_cap:
            batch_index = _select_next_batch_index(
                pending_batch_ids=pending_batch_ids,
                batch_plan=batch_plan,
                task_meta=task_meta,
                active_batch_ids=list(inflight_batches.keys()),
                active_worker_cap=scheduler.active_worker_cap,
            )
            if batch_index is None:
                return
            pending_batch_ids.remove(batch_index)
            batch_indices = list(batch_plan[batch_index])
            batch_meta = [task_meta[idx] for idx in batch_indices]
            seed_list = [meta["seed"] for meta in batch_meta]
            group_name = str(batch_meta[0]["group"]) if batch_meta else "unknown"
            print(
                f"[{_ts()}] Dispatch batch {batch_index + 1}/{len(batch_plan)} | "
                f"worker=queued | group={group_name} | seeds={len(batch_indices)} | "
                f"first_seed={seed_list[0]} | last_seed={seed_list[-1]} | "
                f"active_limit={scheduler.active_worker_cap}",
                flush=True,
            )
            request = _ProcessBatchRequest(
                batch_index=batch_index,
                batch_indices=list(batch_indices),
                tasks=[all_tasks[idx] for idx in batch_indices],
                uid=uid,
                model_path=str(model_path),
                task_total=len(all_tasks),
            )
            inflight_batches[batch_index] = request
            task_queue.put(request)

    def _check_for_stalled_workers() -> int:
        completed_now = 0
        now = time.time()
        for worker_slot, batch_index in list(worker_active_batches.items()):
            last_hb = worker_last_heartbeat.get(worker_slot)
            if last_hb is None or (now - last_hb) < stall_timeout_sec:
                continue
            request = inflight_batches.pop(batch_index, None)
            worker_active_batches.pop(worker_slot, None)
            started_at = worker_started_at.pop(worker_slot, now)
            worker_last_heartbeat.pop(worker_slot, None)
            elapsed_sec = max(0.0, now - started_at)
            if request is None:
                _restart_worker(worker_slot)
                continue

            batch_meta = [task_meta[idx] for idx in request.batch_indices]
            seed_list = [int(meta["seed"]) for meta in batch_meta]
            print(
                f"[{_ts()}] Worker {worker_slot} stalled on batch {batch_index + 1}/{len(batch_plan)} "
                f"for {elapsed_sec:.1f}s without heartbeat | seeds={seed_list} | replacing worker",
                flush=True,
            )
            stall_error = f"worker {worker_slot} heartbeat stalled for {elapsed_sec:.1f}s"
            for task in request.tasks:
                seed_meta = _build_worker_stall_seed_meta(
                    task,
                    uid=request.uid,
                    elapsed_sec=elapsed_sec,
                    error=stall_error,
                )
                on_seed_done(seed_meta)
                note = scheduler.observe_seed(seed_meta)
                if note:
                    print(f"[{_ts()}] {note}", flush=True)

            from swarm.protocol import ValidationResult

            record_batch_completion(
                worker_slot,
                batch_index,
                list(request.batch_indices),
                [
                    ValidationResult(int(request.uid), False, 0.0, 0.0)
                    for _ in request.batch_indices
                ],
                elapsed_sec,
            )
            completed_now += 1
            _restart_worker(worker_slot)
            _dispatch_available_batches()
        return completed_now

    try:
        for worker_slot in range(effective_workers):
            _spawn_worker(worker_slot)
        _dispatch_available_batches()

        completed_batches = 0
        while completed_batches < len(batch_plan):
            _drain_progress_events()
            completed_batches += _check_for_stalled_workers()
            if completed_batches >= len(batch_plan):
                break
            try:
                payload = result_queue.get(timeout=0.2)
            except queue_mod.Empty:
                if any(
                    (not worker.is_alive()) and worker.exitcode not in (0, None)
                    for worker in workers.values()
                ):
                    crashed = [
                        f"{worker.name}(exitcode={worker.exitcode})"
                        for worker in workers.values()
                        if (not worker.is_alive()) and worker.exitcode not in (0, None)
                    ]
                    raise RuntimeError(
                        "Benchmark host worker crashed before returning results: "
                        + ", ".join(crashed)
                    )
                await asyncio.sleep(0)
                continue

            _drain_progress_events()
            completed_batches += _check_for_stalled_workers()
            if completed_batches >= len(batch_plan):
                break
            if int(payload.batch_index) not in inflight_batches:
                print(
                    f"[{_ts()}] Ignoring late batch result from worker {payload.worker_id} "
                    f"for already-resolved batch {payload.batch_index + 1}",
                    flush=True,
                )
                continue
            if payload.error:
                raise RuntimeError(
                    f"Host worker {payload.worker_id} failed on batch {payload.batch_index + 1}: "
                    f"{payload.error}\n{payload.traceback_text or ''}".rstrip()
                )

            from swarm.protocol import ValidationResult

            inflight_batches.pop(int(payload.batch_index), None)
            worker_active_batches.pop(int(payload.worker_id), None)
            worker_last_heartbeat.pop(int(payload.worker_id), None)
            worker_started_at.pop(int(payload.worker_id), None)
            record_batch_completion(
                int(payload.worker_id),
                int(payload.batch_index),
                list(payload.batch_indices),
                [ValidationResult(*packed) for packed in payload.results],
                float(payload.elapsed_sec),
            )
            completed_batches += 1
            _dispatch_available_batches()

        _drain_progress_events()
        return effective_workers
    finally:
        for _ in workers.values():
            task_queue.put(None)
        for worker in workers.values():
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)
        for q in (task_queue, result_queue, progress_queue):
            try:
                q.close()
            except Exception:
                pass


async def _run_benchmark(
    model_path: Path,
    uid: int,
    type_seeds: Dict[str, List[int]],
    num_workers: int,
    run_opts: _RunOptions,
) -> tuple:
    from swarm.constants import SIM_DT
    from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator
    from swarm.validator.task_gen import random_task

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
    del evaluator

    seed_times: List[float] = []
    seed_wall_by_key: Dict[Tuple[int, int], deque[float]] = {}
    seed_status_by_key: Dict[Tuple[int, int], deque[str]] = {}
    full_wall_by_key: Dict[Tuple[int, int], float] = {}
    batch_stats: List[_BatchStat] = []
    total_seeds = len(all_tasks)
    heartbeat_sec = run_opts.heartbeat_sec

    eval_start = time.time()
    progress_bar = _build_progress_bar(total_seeds)
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
                    seed_status = str(seed_meta.get("status", "")).strip()
                    if seed_status:
                        seed_status_by_key.setdefault(seed_key, deque()).append(seed_status)
                except Exception:
                    pass
            done_count += 1
            done_snapshot = done_count
            last_done_at = now
            progress_bar.update(1)

        elapsed = now - eval_start
        eta_min = _eta_minutes(elapsed, done_snapshot)
        eta_txt = "--" if eta_min == float("inf") else f"{eta_min:.1f}m"
        progress_bar.set_postfix_str(
            f"done={done_snapshot}/{total_seeds}, elapsed={elapsed/60.0:.1f}m, eta={eta_txt}",
            refresh=False,
        )
        if progress_bar.__class__.__name__ == "_NoopProgressBar":
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
    from .seeds import _batch_indices
    batch_plan = _batch_indices(total_tasks=len(all_tasks))
    print(
        f"[{_ts()}] Running evaluation ({effective_workers} workers, {len(all_tasks)} seeds, "
        f"container_mode=single-seed, host_parallelism=process, batches={len(batch_plan)})..."
    )
    heartbeat_thread = threading.Thread(target=_heartbeat, daemon=True)
    heartbeat_thread.start()
    try:
        with _temporary_env(env_overrides):
            results: List[Optional[Any]] = [None] * len(all_tasks)

            def _record_batch_completion(
                worker_slot: int,
                batch_index: int,
                batch_indices: List[int],
                seed_results: List[Any],
                batch_elapsed: float,
            ) -> None:
                if len(seed_results) != len(batch_indices):
                    raise RuntimeError(
                        f"Worker {worker_slot}: unexpected result count {len(seed_results)} "
                        f"for batch of {len(batch_indices)} seeds."
                    )

                batch_meta = [task_meta[idx] for idx in batch_indices]
                seed_list = [meta["seed"] for meta in batch_meta]
                batch_processing_sec = 0.0
                for idx, result in zip(batch_indices, seed_results):
                    results[idx] = result
                    meta = task_meta[idx]
                    seed_key = (int(meta["seed"]), int(meta["challenge_type"]))
                    seed_values = seed_wall_by_key.get(seed_key)
                    seed_processing = float(seed_values[0]) if seed_values else 0.0
                    batch_processing_sec += seed_processing

                startup_overhead_sec = max(0.0, batch_elapsed - batch_processing_sec)
                startup_share = startup_overhead_sec / len(batch_indices) if batch_indices else 0.0
                for idx in batch_indices:
                    meta = task_meta[idx]
                    seed_key = (int(meta["seed"]), int(meta["challenge_type"]))
                    seed_values = seed_wall_by_key.get(seed_key)
                    seed_processing = float(seed_values[0]) if seed_values else 0.0
                    full_wall_by_key[seed_key] = seed_processing + startup_share

                batch_stats.append(
                    _BatchStat(
                        worker_id=worker_slot,
                        batch_index=batch_index,
                        seed_count=len(batch_indices),
                        elapsed_sec=batch_elapsed,
                        seed_processing_sec=batch_processing_sec,
                        startup_overhead_sec=startup_overhead_sec,
                        seeds=seed_list,
                    )
                )
                print(
                    f"[{_ts()}] Worker {worker_slot} complete | batch {batch_index + 1} "
                    f"| seeds={len(batch_indices)} | elapsed={batch_elapsed:.1f}s "
                    f"| startup={startup_overhead_sec:.1f}s",
                    flush=True,
                )

            worker_count = await _engine_facade()._run_benchmark_process_mode(
                all_tasks=all_tasks,
                task_meta=task_meta,
                batch_plan=batch_plan,
                uid=uid,
                model_path=model_path,
                effective_workers=effective_workers,
                record_batch_completion=_record_batch_completion,
                on_seed_done=_on_seed_done,
                run_opts=run_opts,
            )

            if any(r is None for r in results):
                raise RuntimeError("Dynamic dispatch ended with missing seed result(s).")
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=2.0)
        progress_bar.close()

    elapsed = time.time() - eval_start
    return (
        task_meta,
        results,
        seed_times,
        seed_wall_by_key,
        seed_status_by_key,
        full_wall_by_key,
        batch_stats,
        elapsed,
        eval_start,
        worker_count,
    )
