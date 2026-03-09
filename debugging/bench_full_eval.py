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
import io
import json
import multiprocessing as mp
import os
import queue as queue_mod
import random
import re
import statistics
import sys
import threading
import time
import traceback
import gc
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - fallback path when tqdm is unavailable.
    _tqdm = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SWARM_PRIVATE_BENCHMARK_SECRET", "bench_test_key_2026")

_UID_RE = re.compile(r"uid[_-]?(\d+)", re.IGNORECASE)

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


@dataclass
class _RunOptions:
    heartbeat_sec: float = 30.0
    rpc_trace: bool = False
    rpc_trace_every: int = 250
    rpc_heartbeat_sec: float = 150.0
    max_batch_timeout_sec: float = 900.0
    timeout_multiplier: float = 1.0
    extend_timeout_on_progress: bool = True
    timeout_extend_sec: float = 30.0
    timeout_progress_stale_sec: float = 3.0
    timeout_progress_min_sim_advance: float = 0.02
    max_seed_walltime_sec: float = 0.0
    default_log_out: Optional[str] = None


@dataclass
class _BatchStat:
    worker_id: int
    batch_index: int
    seed_count: int
    elapsed_sec: float
    seed_processing_sec: float
    startup_overhead_sec: float
    seeds: List[int]


@dataclass
class _ProcessBatchRequest:
    batch_index: int
    batch_indices: List[int]
    tasks: List[Any]
    uid: int
    model_path: str
    task_total: int


@dataclass
class _ProcessBatchResult:
    worker_id: int
    batch_index: int
    batch_indices: List[int]
    results: List[Tuple[int, bool, float, float]]
    elapsed_sec: float
    error: Optional[str] = None
    traceback_text: Optional[str] = None


@dataclass
class _ProcessSeedEvent:
    worker_id: int
    batch_index: int
    seed_meta: Optional[Dict[str, Any]] = None


def _debug_profile_options() -> _RunOptions:
    return _RunOptions(
        heartbeat_sec=15.0,
        rpc_trace=True,
        rpc_trace_every=250,
        rpc_heartbeat_sec=120.0,
        max_batch_timeout_sec=300.0,
        timeout_multiplier=1.0,
        extend_timeout_on_progress=True,
        timeout_extend_sec=30.0,
        timeout_progress_stale_sec=3.0,
        timeout_progress_min_sim_advance=0.02,
        max_seed_walltime_sec=1800.0,
        default_log_out="/tmp/bench_full_eval.log",
    )


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
        "--uid", type=int, default=None,
        help="Miner UID (default: inferred from --model filename, fallback 0).",
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
        "--log-out", type=Path, default=None,
        help="Path to write log file (default: /tmp/bench_full_eval.log).",
    )
    parser.add_argument(
        "--relax-timeouts", action="store_true", default=False,
        help="Override timing constants for slow machines (longer timeouts, more strikes).",
    )
    parser.add_argument(
        "--rpc-verbosity",
        choices=["low", "mid", "high"],
        default="mid",
        help="RPC log verbosity level (default: mid).",
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=None,
        help="JSON file with exact benchmark seeds to reuse across runs.",
    )
    parser.add_argument(
        "--save-seed-file",
        type=Path,
        default=None,
        help="Write the resolved benchmark seed map to JSON.",
    )
    parser.add_argument(
        "--seed-search-rng",
        type=int,
        default=None,
        help="Seed used for reproducible seed discovery when --seed-file is not provided.",
    )
    parser.add_argument(
        "--summary-json-out",
        type=Path,
        default=None,
        help="Write benchmark summary JSON to this path.",
    )
    return parser.parse_args()


def _infer_uid_from_model_path(model_path: Path) -> Optional[int]:
    """Infer miner UID from model filename patterns like UID_178.zip."""
    for candidate in (model_path.stem, model_path.name):
        match = _UID_RE.search(candidate)
        if match:
            try:
                return int(match.group(1))
            except Exception:
                continue
    return None


def _normalize_type_seeds(raw: Any) -> Dict[str, List[int]]:
    if not isinstance(raw, dict):
        raise ValueError("Seed file must contain a JSON object mapping benchmark groups to seed lists.")

    normalized: Dict[str, List[int]] = {}
    missing = [group for group in BENCH_GROUP_ORDER if group not in raw]
    if missing:
        raise ValueError(f"Seed file missing groups: {', '.join(missing)}")

    for group in BENCH_GROUP_ORDER:
        values = raw.get(group)
        if not isinstance(values, list) or not values:
            raise ValueError(f"Seed group {group} must be a non-empty list.")
        normalized[group] = [int(seed) for seed in values]

    return normalized


def _load_type_seeds(seed_file: Path) -> Dict[str, List[int]]:
    return _normalize_type_seeds(json.loads(seed_file.read_text()))


def _save_type_seeds(seed_file: Path, type_seeds: Dict[str, List[int]]) -> None:
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    seed_file.write_text(json.dumps(type_seeds, indent=2, sort_keys=True))


class _Tee(io.TextIOBase):
    """Write to multiple file objects simultaneously."""
    def __init__(self, *files):
        self.files = files
        self._primary = files[0] if files else sys.__stdout__
        self._lock = threading.Lock()

    def write(self, data):
        with self._lock:
            for f in self.files:
                try:
                    if getattr(f, "closed", False):
                        continue
                    f.write(data)
                    f.flush()
                except Exception:
                    # Best-effort tee: ignore late writes during shutdown races.
                    continue
        return len(data)

    def flush(self):
        with self._lock:
            for f in self.files:
                try:
                    if getattr(f, "closed", False):
                        continue
                    f.flush()
                except Exception:
                    continue

    @property
    def buffer(self):
        return getattr(self._primary, "buffer", None)

    @property
    def encoding(self):
        return getattr(self._primary, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._primary, "errors", "strict")

    def reconfigure(self, *args, **kwargs):
        for f in self.files:
            try:
                reconfigure = getattr(f, "reconfigure", None)
                if callable(reconfigure):
                    reconfigure(*args, **kwargs)
            except Exception:
                continue

    def fileno(self):
        return self._primary.fileno()

    def isatty(self):
        return self._primary.isatty()

    def writable(self):
        return True

    def __getattr__(self, name: str):
        return getattr(self._primary, name)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


class _NoopProgressBar:
    def update(self, _n: int) -> None:
        return None

    def set_postfix_str(self, _text: str, refresh: bool = True) -> None:
        _ = refresh
        return None

    def close(self) -> None:
        return None


def _build_progress_bar(total_seeds: int):
    if _tqdm is None:
        return _NoopProgressBar()
    return _tqdm(
        total=total_seeds,
        desc="Seed progress",
        unit="seed",
        dynamic_ncols=True,
        mininterval=0.5,
        leave=True,
    )


def _resolve_run_options(args: argparse.Namespace) -> _RunOptions:
    opts = _debug_profile_options()
    if args.rpc_verbosity == "low":
        opts.rpc_trace = False
        opts.rpc_trace_every = 1000
        opts.rpc_heartbeat_sec = 0.0
    elif args.rpc_verbosity == "mid":
        opts.rpc_trace = True
        opts.rpc_trace_every = 250
        opts.rpc_heartbeat_sec = 120.0
    else:
        opts.rpc_trace = True
        opts.rpc_trace_every = 25
        opts.rpc_heartbeat_sec = 30.0
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


def _find_seeds(seeds_per_group: int) -> Dict[str, List[int]]:
    """Find seeds covering all benchmark map groups."""
    from swarm.constants import SIM_DT
    from swarm.validator.task_gen import random_task

    groups: Dict[str, List[int]] = {g: [] for g in BENCH_GROUP_ORDER}

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


def _batch_indices(total_tasks: int) -> List[List[int]]:
    if total_tasks <= 0:
        return []
    return [
        [index]
        for index in range(total_tasks)
    ]


def _active_runtime_overrides() -> Dict[str, str]:
    keys = [
        "SWARM_DOCKER_THREAD_CAPS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "SWARM_TORCH_NUM_THREADS",
        "SWARM_TORCH_INTEROP_THREADS",
        "SWARM_DOCKER_WORKER_CPUS_OVERRIDE",
        "SWARM_DOCKER_WORKER_MEMORY_OVERRIDE",
        "SWARM_DOCKER_WORKER_CPUSETS",
    ]
    active: Dict[str, str] = {}
    for key in keys:
        value = os.getenv(key)
        if value not in (None, ""):
            active[key] = value
    for key, value in os.environ.items():
        if key.startswith("SWARM_DOCKER_WORKER_CPUSET_CPUS_") and value not in ("",):
            active[key] = value
    return active


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


def _is_clean_execution_status(status: str) -> bool:
    return status == "seed_done"


def _benchmark_worker_main(
    process_slot: int,
    task_queue: Any,
    result_queue: Any,
    progress_queue: Any,
) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        evaluator = _create_prepared_benchmark_evaluator()

        while True:
            request = task_queue.get()
            if request is None:
                return

            batch_start = time.time()

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
) -> int:
    ctx = _benchmark_mp_context()
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    progress_queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_benchmark_worker_main,
            args=(worker_slot, task_queue, result_queue, progress_queue),
            name=f"bench_host_worker_{worker_slot}",
            daemon=True,
        )
        for worker_slot in range(effective_workers)
    ]

    def _drain_progress_events() -> None:
        while True:
            try:
                event = progress_queue.get_nowait()
            except queue_mod.Empty:
                return
            on_seed_done(getattr(event, "seed_meta", None))

    try:
        for worker in workers:
            worker.start()

        for batch_index, batch_indices in enumerate(batch_plan):
            batch_meta = [task_meta[idx] for idx in batch_indices]
            seed_list = [meta["seed"] for meta in batch_meta]
            print(
                f"[{_ts()}] Dispatch batch {batch_index + 1}/{len(batch_plan)} | "
                f"worker=queued | seeds={len(batch_indices)} | "
                f"first_seed={seed_list[0]} | last_seed={seed_list[-1]}",
                flush=True,
            )
            task_queue.put(
                _ProcessBatchRequest(
                    batch_index=batch_index,
                    batch_indices=list(batch_indices),
                    tasks=[all_tasks[idx] for idx in batch_indices],
                    uid=uid,
                    model_path=str(model_path),
                    task_total=len(all_tasks),
                )
            )

        completed_batches = 0
        while completed_batches < len(batch_plan):
            _drain_progress_events()
            try:
                payload = result_queue.get(timeout=0.2)
            except queue_mod.Empty:
                if any((not worker.is_alive()) and worker.exitcode not in (0, None) for worker in workers):
                    crashed = [
                        f"{worker.name}(exitcode={worker.exitcode})"
                        for worker in workers
                        if (not worker.is_alive()) and worker.exitcode not in (0, None)
                    ]
                    raise RuntimeError(
                        "Benchmark host worker crashed before returning results: "
                        + ", ".join(crashed)
                    )
                await asyncio.sleep(0)
                continue

            _drain_progress_events()
            if payload.error:
                raise RuntimeError(
                    f"Host worker {payload.worker_id} failed on batch {payload.batch_index + 1}: "
                    f"{payload.error}\n{payload.traceback_text or ''}".rstrip()
                )

            from swarm.protocol import ValidationResult

            record_batch_completion(
                int(payload.worker_id),
                int(payload.batch_index),
                list(payload.batch_indices),
                [ValidationResult(*packed) for packed in payload.results],
                float(payload.elapsed_sec),
            )
            completed_batches += 1

        _drain_progress_events()
        return effective_workers
    finally:
        for _ in workers:
            task_queue.put(None)
        for worker in workers:
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
    """Run Docker evaluation and return benchmark artifacts plus launched worker count."""
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
            last_done_at = now
            done_snapshot = done_count
            progress_bar.update(1)

        elapsed = now - eval_start
        eta_min = _eta_minutes(elapsed, done_snapshot)
        eta_txt = "--" if eta_min == float("inf") else f"{eta_min:.1f}m"
        progress_bar.set_postfix_str(
            f"done={done_snapshot}/{total_seeds}, elapsed={elapsed/60.0:.1f}m, eta={eta_txt}",
            refresh=False,
        )
        if _tqdm is None:
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
                startup_share = (
                    startup_overhead_sec / len(batch_indices) if batch_indices else 0.0
                )
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

            worker_count = await _run_benchmark_process_mode(
                all_tasks=all_tasks,
                task_meta=task_meta,
                batch_plan=batch_plan,
                uid=uid,
                model_path=model_path,
                effective_workers=effective_workers,
                record_batch_completion=_record_batch_completion,
                on_seed_done=_on_seed_done,
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
    """Print results table and return JSON-serializable summary."""
    GROUP_ORDER = BENCH_GROUP_ORDER
    seed_wall_queues: Dict[Tuple[int, int], deque[float]] = {
        key: deque(values) for key, values in seed_wall_by_key.items()
    }
    seed_status_queues: Dict[Tuple[int, int], deque[str]] = {
        key: deque(values) for key, values in seed_status_by_key.items()
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
        status_q = seed_status_queues.get(seed_key)
        execution_status = str(status_q.popleft()) if status_q and len(status_q) > 0 else "unknown"
        execution_ok = _is_clean_execution_status(execution_status)
        wall_q = seed_wall_queues.get(seed_key)
        if wall_q and len(wall_q) > 0:
            processing_wall = float(wall_q.popleft())
        elif i < len(seed_times):
            # Fallback for compatibility if callback metadata was not provided.
            processing_wall = (seed_times[i] - seed_times[i - 1]) if i > 0 else (seed_times[0] - eval_start)
        else:
            processing_wall = 0.0
        wall = float(full_wall_by_key.get(seed_key, processing_wall))

        is_timeout = wall < 0.5 and i > 0
        group_results[group].append({
            "seed": meta["seed"],
            "score": score,
            "success": success,
            "sim_time": sim_time,
            "wall_time": wall,
            "processing_wall_time": processing_wall,
            "execution_status": execution_status,
            "execution_ok": execution_ok,
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
    execution_success_count = sum(1 for r in all_rows if bool(r["execution_ok"]))
    execution_status_counts = Counter(str(r["execution_status"]) for r in all_rows)
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
    avg_startup_overhead_sec = (
        total_startup_overhead_sec / len(batch_stats) if batch_stats else 0.0
    )
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


def main() -> None:
    args = _parse_args()
    run_opts = _resolve_run_options(args)
    selected_groups = BENCH_GROUP_ORDER

    requested_workers = max(1, int(args.workers))
    effective_workers = requested_workers

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    inferred_uid = _infer_uid_from_model_path(model_path)
    if args.uid is None:
        uid = inferred_uid if inferred_uid is not None else 0
    else:
        uid = int(args.uid)

    effective_log_out = args.log_out
    if effective_log_out is None and run_opts.default_log_out:
        effective_log_out = Path(run_opts.default_log_out)

    log_fh = None
    run_error: Optional[BaseException] = None
    report_error: Optional[BaseException] = None
    task_meta: List[Dict[str, Any]] = []
    results: list = []
    seed_times: List[float] = []
    seed_wall_by_key: Dict[Tuple[int, int], deque[float]] = {}
    full_wall_by_key: Dict[Tuple[int, int], float] = {}
    seed_status_by_key: Dict[Tuple[int, int], deque[str]] = {}
    batch_stats: List[_BatchStat] = []
    elapsed = 0.0
    eval_start = time.time()
    launched_workers = effective_workers
    overrides: Dict[str, Any] = {}
    summary: Optional[Dict[str, Any]] = None

    try:
        if effective_log_out:
            log_fh = open(effective_log_out, "w")
            sys.stdout = _Tee(sys.__stdout__, log_fh)
            sys.stderr = _Tee(sys.__stderr__, log_fh)

        if args.relax_timeouts:
            overrides = _apply_relaxed_overrides()

        print(f"[{_ts()}] === FULL EVALUATION BENCHMARK ===")
        print(f"[{_ts()}] Model: {model_path}")
        if args.uid is None and inferred_uid is not None:
            print(f"[{_ts()}] UID: {uid} (inferred from model filename)")
        elif args.uid is None:
            print(f"[{_ts()}] UID: {uid} (default fallback)")
        else:
            print(f"[{_ts()}] UID: {uid} (from --uid)")
        print(f"[{_ts()}] Workers requested: {requested_workers}")
        print(f"[{_ts()}] Workers effective:  {effective_workers}")
        print(f"[{_ts()}] Profile: debug")
        print(f"[{_ts()}] RPC verbosity: {args.rpc_verbosity}")
        print(f"[{_ts()}] Host parallelism: process")
        print(f"[{_ts()}] Container mode: single-seed")
        if overrides:
            print(f"[{_ts()}] Relaxed timeouts: {overrides}")
        runtime_overrides = _active_runtime_overrides()
        if runtime_overrides:
            print(f"[{_ts()}] Runtime overrides: {runtime_overrides}")
        print(f"[{_ts()}] Map types selected: {', '.join(selected_groups)}")
        if effective_log_out:
            print(f"[{_ts()}] Log file: {effective_log_out}")
        print()

        if args.seed_file is not None:
            print(f"[{_ts()}] Loading seeds from {args.seed_file}...")
            type_seeds = _load_type_seeds(args.seed_file)
        else:
            if args.seed_search_rng is not None:
                print(f"[{_ts()}] Seed search RNG: {args.seed_search_rng}")
                random.seed(args.seed_search_rng)
            print(f"[{_ts()}] Finding {args.seeds_per_group} seeds per group...")
            type_seeds = _find_seeds(args.seeds_per_group)
        if args.save_seed_file is not None:
            _save_type_seeds(args.save_seed_file, type_seeds)
            print(f"[{_ts()}] Saved seed file: {args.save_seed_file}")
        total_seeds = sum(len(v) for v in type_seeds.values())
        for group, seeds in type_seeds.items():
            print(f"  {group}: {seeds}")
        print(f"  Total: {total_seeds}")
        print()

        (
            task_meta,
            results,
            seed_times,
            seed_wall_by_key,
            seed_status_by_key,
            full_wall_by_key,
            batch_stats,
            elapsed,
            eval_start,
            launched_workers,
        ) = asyncio.run(
            _run_benchmark(
                model_path,
                uid,
                type_seeds,
                effective_workers,
                run_opts=run_opts,
            )
        )
    except BaseException as exc:
        run_error = exc
    finally:
        # Always restore real stdio before final report to avoid wrapper edge-cases.
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        out_stream = sys.__stdout__
        err_stream = sys.__stderr__
        if out_stream is None or getattr(out_stream, "closed", False):
            out_stream = err_stream if err_stream is not None else sys.stdout
        if err_stream is None or getattr(err_stream, "closed", False):
            err_stream = out_stream if out_stream is not None else sys.stderr
        sys.stdout = out_stream
        sys.stderr = err_stream

    print(f"\n[{_ts()}] === RESULTS ===", flush=True)
    if task_meta and results:
        if run_error is not None:
            print(f"[{_ts()}] Printing partial results collected before failure.", flush=True)
        try:
            summary = _print_results(
                task_meta,
                results,
                seed_times,
                seed_wall_by_key,
                seed_status_by_key,
                full_wall_by_key,
                batch_stats,
                elapsed,
                eval_start,
                launched_workers,
                host_parallelism="process",
            )
        except BaseException as exc:
            report_error = exc
            print(f"[{_ts()}] Report generation failed: {type(exc).__name__}: {exc}", flush=True)

    if run_error is not None:
        print(
            f"[{_ts()}] Benchmark failed before report generation: {type(run_error).__name__}: {run_error}",
            flush=True,
        )
        traceback.print_exception(type(run_error), run_error, run_error.__traceback__)
        if report_error is not None:
            traceback.print_exception(type(report_error), report_error, report_error.__traceback__)
        print(f"[{_ts()}] === BENCHMARK FAILED ===", flush=True)
    elif report_error is not None:
        traceback.print_exception(type(report_error), report_error, report_error.__traceback__)
        print(f"[{_ts()}] === BENCHMARK FAILED ===", flush=True)
    else:
        print(f"[{_ts()}] === BENCHMARK COMPLETE ===", flush=True)

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass

    if log_fh:
        try:
            log_fh.flush()
        except Exception:
            pass
        finally:
            log_fh.close()

    if args.summary_json_out is not None and summary is not None:
        args.summary_json_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    if run_error is not None:
        raise run_error
    if report_error is not None:
        raise report_error


if __name__ == "__main__":
    main()
