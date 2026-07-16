import asyncio
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import bittensor as bt

from swarm.config import DockerBatchTimeoutSettings, RpcTraceSettings
from swarm.constants import GLOBAL_EVAL_BASE_SEC, GLOBAL_EVAL_CAP_SEC, GLOBAL_EVAL_PER_SEED_SEC, SIM_DT
from swarm.model_graph import ReasonCode, admit_artifact_subprocess
from swarm.model_graph.constants import RUNNER_STARTUP_WALL_SEC
from swarm.protocol import (
    FailureReason,
    SCHEMA_VERSION,
    ValidationResult,
    is_supported_schema,
    normalize_version,
)
from swarm.validator.calibration import (
    CALIBRATION_STATE,
    baseline_model_available,
    baseline_model_path,
    load_baseline_manifest,
    normalize_speed_factor,
    percentile,
)
from swarm.validator.task_gen import task_for_seed_and_type

from ._shared import (
    _docker_evaluator_facade,
    _runtime_profile_env,
    _runtime_profile_from_payload,
)

_CALIBRATION_MAX_AGE_SEC = 6 * 3600  # re-measure the host speed factor at least this often


def _docker_cmd_quiet(cmd: list[str], timeout_sec: float = 30.0) -> None:
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# evaluate_seeds_batch — extracted phases
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _BatchHelpers:
    """Bundle of internal closures built once at orchestrator entry."""
    phase: Callable[[str], None]
    on_seed_complete_guarded: Callable
    build_failure_seed_meta: Callable
    notify_all_failed: Callable
    run_docker_cmd_quiet: Callable
    cleanup_tmpdir_quiet: Callable


@dataclass
class _BatchContext:
    """Mutable shared state for evaluate_seeds_batch phase helpers."""

    # Function parameters
    self: Any
    tasks: list
    uid: int
    model_path: Path
    worker_id: int = 0
    on_seed_complete: Optional[Callable[..., None]] = None
    rollout_observer: Optional[Callable[[dict], None]] = None
    task_offset: int = 0
    task_total: Optional[int] = None
    runtime_profile_payload: Optional[dict[str, Any]] = None
    speed_factor: Optional[float] = None

    # Trace + sync primitives (built in _init_batch_state)
    trace_rpc: bool = False
    stop_event: Optional[threading.Event] = None
    completed_lock: Optional[threading.Lock] = None
    progress_state: Optional[dict] = None

    # Pre-try state (set by _setup_pretry_state)
    container_name: Optional[str] = None
    host_port: Optional[int] = None
    tmpdir: Optional[str] = None

    # Graph runner state
    run_image: Optional[str] = None
    current_uid: Optional[int] = None
    current_gid: Optional[int] = None
    worker_limits: Optional[dict] = None
    docker_envs: Optional[dict] = None
    validator_ip: Optional[str] = None
    runtime_profile: Optional[Any] = None

    connected: bool = False

    # Closure bundle (built in _init_batch_state)
    helpers: Optional[_BatchHelpers] = None


def _init_batch_state(ctx: _BatchContext) -> None:
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    on_seed_complete = ctx.on_seed_complete

    ctx.trace_rpc = RpcTraceSettings.from_env().enabled
    ctx.stop_event = threading.Event()
    ctx.progress_state = {
        "uid": uid,
        "worker_id": worker_id,
        "phase": "init",
        "task": "n/a",
        "step_idx": 0,
        "sim_t": 0.0,
        "ts": time.time(),
    }
    ctx.completed_lock = threading.Lock()
    completed_count = 0

    trace_rpc = ctx.trace_rpc
    completed_lock = ctx.completed_lock

    def _phase(msg: str) -> None:
        if not trace_rpc:
            return
        line = f"[{time.strftime('%H:%M:%S')}] [RPC TRACE][Worker {worker_id}][UID {uid}] {msg}"
        print(line, flush=True)
        bt.logging.info(line)

    def _on_seed_complete_guarded(seed_meta: Optional[dict] = None) -> None:
        nonlocal completed_count
        if on_seed_complete is None:
            return
        with completed_lock:
            if completed_count >= len(tasks):
                return
            completed_count += 1
        try:
            on_seed_complete(seed_meta)
        except TypeError:
            try:
                on_seed_complete()
            except Exception:
                pass
        except Exception:
            pass

    def _build_failure_seed_meta(task_obj, *, status: str, error: str = "") -> dict:
        return {
            "uid": int(uid),
            "map_seed": int(getattr(task_obj, "map_seed", -1)),
            "challenge_type": int(getattr(task_obj, "challenge_type", -1)),
            "horizon_sec": float(getattr(task_obj, "horizon", 0.0)),
            "status": status,
            "success": False,
            "sim_time_sec": 0.0,
            "seed_wall_sec": 0.0,
            "step_idx": 0,
            "error": error,
        }

    def _notify_all_failed(*, status: str = "batch_failed", error: str = ""):
        """Call on_seed_complete for all pending tasks when batch fails early."""
        with completed_lock:
            start_index = min(completed_count, len(tasks))
            remaining_tasks = list(tasks[start_index:])
        _phase(
            f"batch failing early; marking {len(remaining_tasks)} pending seed(s) as failed "
            f"with status={status}"
        )
        for failed_task in remaining_tasks:
            _on_seed_complete_guarded(
                _build_failure_seed_meta(
                    failed_task,
                    status=status,
                    error=error,
                )
            )

    def _run_docker_cmd_quiet(cmd: list[str], timeout_sec: float = 30.0) -> None:
        """Run cleanup docker command without letting hangs block benchmark completion."""
        try:
            subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
        except Exception:
            pass

    def _cleanup_tmpdir_quiet(
        path: Optional[str], timeout_sec: float = 16.0
    ) -> None:
        """Best-effort tmpdir cleanup without blocking benchmark completion."""
        if not path:
            return
        done = threading.Event()

        def _rm() -> None:
            try:
                shutil.rmtree(path, ignore_errors=True)
            finally:
                done.set()

        t = threading.Thread(
            target=_rm,
            name=f"tmp_cleanup_uid{uid}_w{worker_id}",
            daemon=True,
        )
        t.start()
        if not done.wait(timeout=timeout_sec):
            bt.logging.warning(
                f"[Worker {worker_id}] tmpdir cleanup still running in background: {path}"
            )

    ctx.helpers = _BatchHelpers(
        phase=_phase,
        on_seed_complete_guarded=_on_seed_complete_guarded,
        build_failure_seed_meta=_build_failure_seed_meta,
        notify_all_failed=_notify_all_failed,
        run_docker_cmd_quiet=_run_docker_cmd_quiet,
        cleanup_tmpdir_quiet=_cleanup_tmpdir_quiet,
    )


def check_task_versions(
    uid: int, worker_id: int, tasks: list
) -> Optional[list]:
    for task in tasks:
        task_version = getattr(task, "version", None)
        if task_version is None:
            continue
        if not is_supported_schema(task_version):
            bt.logging.warning(
                f"[Worker {worker_id}] UID {uid} task schema {task_version!r} not in allow-list; rejecting batch"
            )
            return [
                ValidationResult(
                    uid, False, 0.0, 0.0,
                    failure_reason=FailureReason.INFRA.value,
                )
                for _ in tasks
            ]
        if normalize_version(task_version) != SCHEMA_VERSION:
            bt.logging.warning(
                f"[Worker {worker_id}] UID {uid} task schema {task_version!r} supported but not current ({SCHEMA_VERSION})"
            )
    return None


def _validate_inputs(ctx: _BatchContext) -> Optional[list]:
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    model_path = ctx.model_path
    _notify_all_failed = ctx.helpers.notify_all_failed

    schema_reject = check_task_versions(uid, worker_id, tasks)
    if schema_reject is not None:
        _notify_all_failed(status="unsupported_schema_version")
        return schema_reject

    if not model_path.is_file():
        bt.logging.warning(f"[Worker {worker_id}] Model path missing: {model_path}")
        _notify_all_failed(status="model_path_missing")
        return [
            ValidationResult(uid, False, 0.0, 0.0, failure_reason=FailureReason.INFRA.value)
            for _ in tasks
        ]

    if not _docker_evaluator_facade().DockerSecureEvaluator._base_ready:
        bt.logging.warning(f"[Worker {worker_id}] Docker not ready for UID {uid}")
        _notify_all_failed(status="docker_not_ready")
        return [
            ValidationResult(uid, False, 0.0, 0.0, failure_reason=ReasonCode.INFRA_DOCKER.value)
            for _ in tasks
        ]

    admission = admit_artifact_subprocess(model_path)
    if not admission.accepted:
        bt.logging.warning(
            f"[Worker {worker_id}] UID {uid} graph admission failed: "
            f"{admission.reason_code}: {admission.detail}"
        )
        _notify_all_failed(status=admission.reason_code, error=admission.detail)
        return [
            ValidationResult(
                uid, False, 0.0, 0.0, failure_reason=admission.reason_code
            )
            for _ in tasks
        ]

    return None


def _setup_pretry_state(ctx: _BatchContext) -> None:
    self = ctx.self
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    _phase = ctx.helpers.phase

    ctx.container_name = f"swarm_eval_{uid}_w{worker_id}_{int(time.time() * 1000)}"
    ctx.host_port = self._find_free_port(worker_id)

    _phase(
        f"prepare container={ctx.container_name} host_port={ctx.host_port} seeds={len(tasks)}"
    )



OBS_SHM_BYTES = 32 * 1024 * 1024


def _obs_shm_host_path(host_port: int) -> str:
    return f"/dev/shm/swarm_obs_{host_port}.bin"


def _create_obs_shm(host_port: int) -> Optional[str]:
    path = _obs_shm_host_path(host_port)
    try:
        with open(path, "wb") as f:
            f.truncate(OBS_SHM_BYTES)
        os.chmod(path, 0o644)
        return path
    except OSError:
        return None


async def _run_rpc_phase(ctx: _BatchContext) -> list:
    """Owns the inner try/finally entirely: diagnostics and result validation
    run inside the try, before the cleanup in the finally block."""
    self = ctx.self
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    container_name = ctx.container_name
    host_port = ctx.host_port
    rollout_observer = ctx.rollout_observer
    stop_event = ctx.stop_event
    progress_state = ctx.progress_state
    task_offset = ctx.task_offset
    task_total = ctx.task_total
    runtime_profile = ctx.runtime_profile
    _phase = ctx.helpers.phase
    _on_seed_complete_guarded = ctx.helpers.on_seed_complete_guarded
    _run_docker_cmd_quiet = ctx.helpers.run_docker_cmd_quiet
    _notify_all_failed = ctx.helpers.notify_all_failed

    try:
        profile_base_sec = (
            float(runtime_profile.global_eval_base_sec)
            if runtime_profile is not None and runtime_profile.global_eval_base_sec is not None
            else float(GLOBAL_EVAL_BASE_SEC)
        )
        profile_per_seed_sec = (
            float(runtime_profile.global_eval_per_seed_sec)
            if runtime_profile is not None and runtime_profile.global_eval_per_seed_sec is not None
            else float(GLOBAL_EVAL_PER_SEED_SEC)
        )
        profile_cap_sec = (
            float(runtime_profile.global_eval_cap_sec)
            if runtime_profile is not None and runtime_profile.global_eval_cap_sec is not None
            else float(GLOBAL_EVAL_CAP_SEC)
        )
        base_batch_timeout = profile_base_sec + profile_per_seed_sec * len(tasks)
        if profile_cap_sec > 0:
            base_batch_timeout = min(base_batch_timeout, profile_cap_sec)
        timeout_settings = DockerBatchTimeoutSettings.from_env()
        profile_timeout_multiplier = (
            float(runtime_profile.batch_timeout_multiplier)
            if runtime_profile is not None
            else 1.0
        )
        timeout_multiplier = timeout_settings.multiplier * profile_timeout_multiplier
        batch_timeout = base_batch_timeout * timeout_multiplier
        hard_cap_timeout = timeout_settings.hard_cap_sec
        if hard_cap_timeout > 0:
            batch_timeout = min(batch_timeout, hard_cap_timeout)
        extend_on_progress = timeout_settings.extend_on_progress
        extend_by_sec = timeout_settings.extend_by_sec
        progress_stale_sec = timeout_settings.progress_stale_sec
        progress_min_sim_advance = timeout_settings.progress_min_sim_advance
        max_total_timeout_sec = timeout_settings.max_total_timeout_sec

        if hard_cap_timeout > 0:
            _phase(
                f"starting rpc batch with timeout={batch_timeout:.1f}s "
                f"(base={base_batch_timeout:.1f}s x {timeout_multiplier:.2f} "
                f"hard_cap={hard_cap_timeout:.1f}s)"
            )
        else:
            _phase(
                f"starting rpc batch with timeout={batch_timeout:.1f}s "
                f"(base={base_batch_timeout:.1f}s x {timeout_multiplier:.2f})"
            )
        if extend_on_progress:
            _phase(
                f"progress timeout extension enabled: +{extend_by_sec:.1f}s when "
                f"stale<={progress_stale_sec:.1f}s and sim advances>={progress_min_sim_advance:.3f}s "
                f"(max_total={'unbounded' if max_total_timeout_sec <= 0 else f'{max_total_timeout_sec:.1f}s'})"
            )

        rpc_done = threading.Event()
        rpc_payload: dict[str, object] = {}

        def _rpc_worker():
            try:
                rpc_payload["results"] = self._run_multi_seed_rpc_sync(
                    tasks,
                    uid,
                    host_port,
                    _on_seed_complete_guarded,
                    rollout_observer,
                    stop_event,
                    progress_state,
                    task_offset,
                    task_total,
                    runtime_profile.as_dict() if runtime_profile is not None else None,
                    ctx.speed_factor,
                )
            except Exception as e:
                rpc_payload["error"] = e
            finally:
                rpc_done.set()

        rpc_thread = threading.Thread(
            target=_rpc_worker,
            name=f"rpc_eval_uid{uid}_w{worker_id}",
            daemon=True,
        )
        rpc_thread.start()

        timed_out = False
        eval_start = time.time()
        timeout_deadline = eval_start + batch_timeout
        extension_count = 0
        last_extended_sim_t = -1.0
        last_extended_step_idx = -1
        while not rpc_done.is_set():
            now = time.time()
            if now >= timeout_deadline:
                if extend_on_progress:
                    try:
                        last_ts = float(progress_state.get("ts", eval_start))
                    except Exception:
                        last_ts = eval_start
                    stale_for = max(0.0, now - last_ts)
                    try:
                        current_sim_t = float(progress_state.get("sim_t", -1.0))
                    except Exception:
                        current_sim_t = -1.0
                    try:
                        current_step_idx = int(
                            progress_state.get("step_idx", -1)
                        )
                    except Exception:
                        current_step_idx = -1

                    sim_advanced = current_sim_t >= (
                        last_extended_sim_t + progress_min_sim_advance
                    )
                    step_advanced = current_step_idx > last_extended_step_idx

                    within_total_cap = True
                    hard_deadline = None
                    if max_total_timeout_sec > 0:
                        hard_deadline = eval_start + max_total_timeout_sec
                        within_total_cap = now < hard_deadline

                    if (
                        stale_for <= progress_stale_sec
                        and (sim_advanced or step_advanced)
                        and within_total_cap
                    ):
                        old_deadline = timeout_deadline
                        timeout_deadline = old_deadline + extend_by_sec
                        if hard_deadline is not None:
                            timeout_deadline = min(
                                timeout_deadline, hard_deadline
                            )

                        if timeout_deadline > old_deadline:
                            extension_count += 1
                            last_extended_sim_t = current_sim_t
                            last_extended_step_idx = current_step_idx
                            _phase(
                                f"timeout extended by {timeout_deadline - old_deadline:.1f}s "
                                f"(#{extension_count}) phase={progress_state.get('phase', 'unknown')} "
                                f"task={progress_state.get('task', 'n/a')} "
                                f"step={current_step_idx} sim_t={current_sim_t:.2f}s stale_for={stale_for:.1f}s"
                            )
                            await asyncio.sleep(0)
                            continue
                timed_out = True
                break
            await asyncio.sleep(0.2)

        if timed_out:
            stop_event.set()
            elapsed = time.time() - eval_start
            timeout_limit_elapsed = timeout_deadline - eval_start
            bt.logging.warning(
                f"[Worker {worker_id}] Batch timeout for UID {uid} after {elapsed:.1f}s "
                f"(limit={timeout_limit_elapsed:.1f}s, base_limit={batch_timeout:.1f}s, "
                f"extensions={extension_count})"
            )
            try:
                last_ts = float(progress_state.get("ts", eval_start))
            except Exception:
                last_ts = eval_start
            stale_sec = max(0.0, time.time() - last_ts)
            _phase(
                f"batch timeout after {timeout_limit_elapsed:.1f}s; last progress "
                f"phase={progress_state.get('phase', 'unknown')} "
                f"task={progress_state.get('task', 'n/a')} "
                f"step={progress_state.get('step_idx', 'n/a')} "
                f"sim_t={progress_state.get('sim_t', 'n/a')} stale_for={stale_sec:.1f}s; "
                f"collecting diagnostics"
            )
            # Give RPC thread short grace period to notice stop_event.
            for _ in range(10):
                if rpc_done.wait(0.2):
                    break
                await asyncio.sleep(0)

            try:
                top_result = subprocess.run(
                    ["docker", "top", container_name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if top_result.returncode == 0 and top_result.stdout.strip():
                    top_snapshot = top_result.stdout[:1200]
                    bt.logging.warning(
                        f"[Worker {worker_id}] Container process snapshot at timeout:\n{top_snapshot}"
                    )
                    _phase(f"container top snapshot:\n{top_snapshot}")
                else:
                    _phase("container top snapshot unavailable")
            except Exception as e:
                _phase(
                    f"container top snapshot failed: {type(e).__name__}: {e}"
                )

            try:
                logs_result = subprocess.run(
                    ["docker", "logs", "--tail", "200", container_name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if logs_result.returncode == 0 and logs_result.stdout.strip():
                    logs_tail = logs_result.stdout[-3000:]
                    bt.logging.warning(
                        f"[Worker {worker_id}] Container logs tail at timeout:\n{logs_tail}"
                    )
                    _phase(f"container logs tail:\n{logs_tail}")
                else:
                    _phase("container logs tail empty")
            except Exception as e:
                _phase(f"container logs tail failed: {type(e).__name__}: {e}")

            partial_results = rpc_payload.get("results")
            if isinstance(partial_results, list) and len(partial_results) == len(tasks):
                completed = sum(1 for r in partial_results if r.score > 0.0)
                bt.logging.warning(
                    f"[Worker {worker_id}] Using partial results: {completed}/{len(tasks)} seeds completed before timeout"
                )
                _notify_all_failed(status="batch_timeout_partial")
                return partial_results
            _notify_all_failed(status="batch_timeout")
            return [
                ValidationResult(
                    uid, False, 0.0, 0.0, failure_reason=FailureReason.INFRA.value
                )
                for _ in tasks
            ]

        if "error" in rpc_payload:
            raise RuntimeError(f"RPC worker failed: {rpc_payload['error']}")

        results_obj = rpc_payload.get("results")
        if not isinstance(results_obj, list):
            raise RuntimeError("RPC worker returned invalid results payload")
        results = results_obj

        valid_results = []
        for r in results:
            score = float(r.score)
            if 0.0 <= score <= 1.0:
                valid_results.append(r)
            else:
                bt.logging.warning(
                    f"[Worker {worker_id}] Invalid score {score}"
                )
                valid_results.append(
                    ValidationResult(
                        uid, False, 0.0, 0.0,
                        failure_reason=ReasonCode.OUTPUT_CONTRACT.value,
                    )
                )

        _phase(f"batch complete ({len(valid_results)} result(s))")
        return valid_results
    finally:
        stop_event.set()
        _run_docker_cmd_quiet(["docker", "kill", container_name])
        _run_docker_cmd_quiet(["docker", "rm", "-f", container_name])
        _phase("container cleaned up")


async def _run_baseline_calibration(self, worker_id: int):
    """Measure this worker's speed factor against the committed baseline model."""
    manifest = load_baseline_manifest()
    model = manifest["baseline_model"]
    measurement = manifest.get("measurement", {})
    seeds = [int(s) for s in measurement.get("sample_seeds", [1001])]
    warmup = int(measurement.get("warmup_steps", 1))
    tasks = [
        task_for_seed_and_type(
            SIM_DT,
            seed=seed,
            challenge_type=int(model["run_as_challenge_type"]),
            family_id=str(model["run_as_family_id"]),
        )
        for seed in seeds
    ]
    sample_horizon = measurement.get("sample_horizon_sec")
    if sample_horizon:
        # Timing only needs a few hundred acts; a shorter episode keeps startup cheap.
        for task in tasks:
            task.horizon = min(float(task.horizon), float(sample_horizon))

    act_ms: list[float] = []
    overhead = {"ms": 0.0}

    def _observer(event: dict) -> None:
        if event.get("event") != "step":
            return
        if int(event.get("step_idx", 0)) > warmup:
            value = float(event.get("act_ms", 0.0))
            if value > 0.0:
                act_ms.append(value)

    def _on_seed(meta=None) -> None:
        if isinstance(meta, dict) and meta.get("calibration_overhead_sec") is not None:
            overhead["ms"] = float(meta["calibration_overhead_sec"]) * 1000.0

    try:
        await evaluate_seeds_batch(
            self,
            tasks,
            0,
            baseline_model_path(),
            worker_id=worker_id,
            on_seed_complete=_on_seed,
            rollout_observer=_observer,
            is_calibration_run=True,
        )
    except Exception as e:
        bt.logging.warning(f"[Worker {worker_id}] baseline calibration failed: {e}")
        return None

    compute = [a - overhead["ms"] for a in act_ms if a - overhead["ms"] > 0.0]
    if len(compute) < 100:
        bt.logging.warning(
            f"[Worker {worker_id}] baseline calibration produced {len(compute)} samples; "
            f"falling back to legacy timing"
        )
        return None

    local_p90 = percentile(compute, 90)
    try:
        speed = normalize_speed_factor(local_p90)
    except ValueError as e:
        bt.logging.warning(f"[Worker {worker_id}] invalid speed factor: {e}")
        return None

    CALIBRATION_STATE.set(worker_id, speed, overhead["ms"], manifest["calibration_version"])
    summary = (
        f"[Worker {worker_id}] reference calibration: speed_factor={speed.factor:.2f}x "
        f"(local_p90={local_p90:.0f}ms / owner_p90={speed.owner_p90_ms:.0f}ms, n={len(compute)})"
    )
    if speed.eligible:
        bt.logging.info(summary)
    else:
        bt.logging.warning(summary + " — host slower than the eligibility limit; it will not score miners")
    return speed


async def _ensure_worker_speed_factor(self, worker_id: int):
    """Return the cached/freshly-measured SpeedFactor, or None to use legacy timing."""
    if not baseline_model_available():
        return None
    if CALIBRATION_STATE.is_stale(worker_id, max_age_sec=_CALIBRATION_MAX_AGE_SEC):
        await _run_baseline_calibration(self, worker_id)
    entry = CALIBRATION_STATE.get(worker_id)
    return entry.speed if entry is not None else None


def _setup_graph_workspace(ctx: _BatchContext) -> Optional[list]:
    """Prepare immutable artifact execution state; no submission is extracted."""
    runtime_profile = _runtime_profile_from_payload(ctx.runtime_profile_payload, ctx.tasks)
    worker_limits = ctx.self._resolve_worker_limits(ctx.worker_id, runtime_profile=runtime_profile)
    docker_envs = ctx.self._docker_env_overrides()
    docker_envs.update(_runtime_profile_env(runtime_profile))
    docker_envs.update({
        "SWARM_MODEL_GRAPH_ARTIFACT": "/workspace/model_graph.zip",
        "SWARM_AGENT_PORT": "8000",
        "SWARM_START_GATE": _START_GATE_PATH,
    })
    ctx.current_uid = os.getuid()
    ctx.current_gid = os.getgid()
    ctx.worker_limits = worker_limits
    ctx.docker_envs = docker_envs
    ctx.run_image = ctx.self.base_image
    ctx.validator_ip = ctx.self._get_docker_host_ip()
    ctx.runtime_profile = runtime_profile
    ctx.self.last_selected_runtime_profile = runtime_profile.as_dict()
    ctx.self.last_selected_worker_limits = dict(worker_limits)
    ctx.self.last_selected_runtime_env = dict(docker_envs)
    ctx.self.last_selected_run_image = str(ctx.run_image)
    return None


_START_GATE_PATH = "/tmp/swarm_start.gate"


def _open_start_gate(container_name: str) -> bool:
    """Signal the runner to load the artifact, only after the network lockdown."""
    try:
        result = subprocess.run(
            ["docker", "exec", container_name, "touch", _START_GATE_PATH],
            capture_output=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _container_is_gone(container_name: str) -> bool:
    """True only when docker positively confirms the container no longer runs.

    An unresponsive daemon (timeout, error) returns False so the failure is
    charged to infrastructure, never to the miner.
    """
    try:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Pid}}", container_name],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return True
    try:
        return int(result.stdout.strip()) <= 0
    except ValueError:
        return False


def _launch_graph_container(ctx: _BatchContext) -> Optional[list]:
    obs_shm_path = _create_obs_shm(ctx.host_port)
    cmd = [
        "docker", "run", "--rm", "-d", "--name", ctx.container_name,
        "--user", f"{ctx.current_uid}:{ctx.current_gid}",
        f"--memory={ctx.worker_limits['memory']}",
        f"--cpus={ctx.worker_limits['cpus']}",
        "--pids-limit=50", "--ulimit", "nofile=256:256",
        "--ulimit", "fsize=52428800:52428800", "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL", "--network", "bridge", "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "-p", f"127.0.0.1:{ctx.host_port}:8000",
        "-v", f"{ctx.model_path.resolve()}:/workspace/model_graph.zip:ro",
    ]
    if ctx.worker_limits["cpuset_cpus"]:
        cmd.extend(["--cpuset-cpus", str(ctx.worker_limits["cpuset_cpus"])])
    for key, value in ctx.docker_envs.items():
        cmd.extend(["-e", f"{key}={value}"])
    if obs_shm_path:
        cmd.extend(["-v", f"{obs_shm_path}:/workspace/obs_shm.bin:ro"])
        cmd.extend(["-e", "SWARM_OBS_SHM=/workspace/obs_shm.bin"])
    cmd.append(ctx.run_image)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        ctx.helpers.notify_all_failed(status=ReasonCode.INFRA_DOCKER.value, error=result.stderr[:300])
        return [ValidationResult(ctx.uid, False, 0.0, 0.0, failure_reason=ReasonCode.INFRA_DOCKER.value) for _ in ctx.tasks]
    return None


async def _prepare_graph_network_and_rpc(ctx: _BatchContext) -> Optional[list]:
    container_pid = ctx.self._get_container_pid(ctx.container_name)
    if (
        not container_pid
        or not ctx.self._apply_network_lockdown(container_pid, ctx.validator_ip)
        or not _open_start_gate(ctx.container_name)
    ):
        ctx.helpers.run_docker_cmd_quiet(["docker", "rm", "-f", ctx.container_name])
        ctx.helpers.notify_all_failed(status=ReasonCode.INFRA_DOCKER.value)
        return [ValidationResult(ctx.uid, False, 0.0, 0.0, failure_reason=ReasonCode.INFRA_DOCKER.value) for _ in ctx.tasks]
    deadline = time.monotonic() + RUNNER_STARTUP_WALL_SEC
    while time.monotonic() < deadline:
        if ctx.self._check_rpc_ready(ctx.host_port):
            ctx.connected = True
            return None
        await asyncio.sleep(0.1)
    gone = _container_is_gone(ctx.container_name)
    ctx.helpers.run_docker_cmd_quiet(["docker", "rm", "-f", ctx.container_name])
    reason = ReasonCode.LOAD_FAILED if gone else ReasonCode.INFRA_DOCKER
    ctx.helpers.notify_all_failed(status=reason.value)
    return [ValidationResult(ctx.uid, False, 0.0, 0.0, failure_reason=reason.value) for _ in ctx.tasks]


async def evaluate_seeds_batch(
    self,
    tasks: list,
    uid: int,
    model_path: Path,
    worker_id: int = 0,
    on_seed_complete: Optional[Callable[..., None]] = None,
    rollout_observer: Optional[Callable[[dict], None]] = None,
    task_offset: int = 0,
    task_total: Optional[int] = None,
    runtime_profile_payload: Optional[dict[str, Any]] = None,
    is_calibration_run: bool = False,
) -> list:
    """Evaluate multiple seeds in a single container.

    Args:
        tasks: List of MapTask objects (one per seed)
        uid: Miner UID
        model_path: Path to model zip file
        worker_id: Worker ID for logging (0 to N_DOCKER_WORKERS-1)

    Returns:
        List of ValidationResult objects (one per seed)
    """
    if not tasks:
        return []

    ctx = _BatchContext(
        self=self,
        tasks=tasks,
        uid=uid,
        model_path=model_path,
        worker_id=worker_id,
        on_seed_complete=on_seed_complete,
        rollout_observer=rollout_observer,
        task_offset=task_offset,
        task_total=task_total,
        runtime_profile_payload=runtime_profile_payload,
    )

    _init_batch_state(ctx)

    early = _validate_inputs(ctx)
    if early is not None:
        return early

    if not is_calibration_run:
        speed = await _ensure_worker_speed_factor(self, worker_id)
        if speed is None or not speed.eligible:
            detail = (
                "reference calibration is unavailable"
                if speed is None
                else f"host speed factor {speed.factor:.2f}x is not eligible to score"
            )
            bt.logging.warning(
                f"[Worker {worker_id}] {detail}; excluding this host from scoring UID {uid}"
            )
            ctx.helpers.notify_all_failed(
                status=ReasonCode.INFRA_CALIBRATION.value, error=detail
            )
            return [
                ValidationResult(
                    uid, False, 0.0, 0.0,
                    failure_reason=ReasonCode.INFRA_CALIBRATION.value,
                )
                for _ in tasks
            ]
        ctx.speed_factor = speed.factor

    _setup_pretry_state(ctx)

    try:
        early = _setup_graph_workspace(ctx)
        if early is not None:
            return early

        early = _launch_graph_container(ctx)
        if early is not None:
            return early

        early = await _prepare_graph_network_and_rpc(ctx)
        if early is not None:
            return early

        return await _run_rpc_phase(ctx)

    except Exception as e:
        bt.logging.warning(f"[Worker {ctx.worker_id}] Batch evaluation failed: {e}")
        ctx.helpers.phase(f"batch evaluation exception: {type(e).__name__}: {e}")
        ctx.helpers.notify_all_failed(
            status="batch_exception",
            error=f"{type(e).__name__}: {e}",
        )
        try:
            ctx.helpers.run_docker_cmd_quiet(["docker", "kill", ctx.container_name])
            ctx.helpers.run_docker_cmd_quiet(["docker", "rm", "-f", ctx.container_name])
        except Exception:
            pass
    finally:
        ctx.helpers.cleanup_tmpdir_quiet(ctx.tmpdir)
        if getattr(ctx, "host_port", None):
            try:
                os.unlink(_obs_shm_host_path(ctx.host_port))
            except OSError:
                pass

    return [
        ValidationResult(uid, False, 0.0, 0.0, failure_reason=FailureReason.INFRA.value)
        for _ in ctx.tasks
    ]


def cleanup(self):
    """Clean up any orphaned containers and prune unused images/cache"""
    for stale in Path("/dev/shm").glob("swarm_obs_*.bin"):
        try:
            stale.unlink()
        except OSError:
            pass
    try:
        # List all swarm evaluation containers
        result = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "name=swarm_eval_",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0 and result.stdout:
            containers = result.stdout.strip().split("\n")
            for container in containers:
                if container:
                    subprocess.run(
                        ["docker", "rm", "-f", container],
                        capture_output=True,
                        timeout=30,
                    )
                    bt.logging.debug(f"Cleaned up orphaned container: {container}")

        # Also clean up verification containers
        result_verify = subprocess.run(
            [
                "docker",
                "ps",
                "-a",
                "--filter",
                "name=swarm_verify_",
                "--format",
                "{{.Names}}",
            ],
            capture_output=True,
            text=True,
        )
        if result_verify.returncode == 0 and result_verify.stdout:
            containers_v = result_verify.stdout.strip().split("\n")
            for container in containers_v:
                if container:
                    subprocess.run(
                        ["docker", "rm", "-f", container],
                        capture_output=True,
                        timeout=30,
                    )
                    bt.logging.debug(
                        f"Cleaned up orphaned verify container: {container}"
                    )

        subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
        subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
        subprocess.run(
            ["docker", "builder", "prune", "-f", "--keep-storage", "5GB"],
            capture_output=True,
        )

    except Exception as e:
        bt.logging.warning(f"Container cleanup failed: {e}")
