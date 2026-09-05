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
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import bittensor as bt

from swarm.config import DockerBatchTimeoutSettings, DockerRuntimeSettings, RpcTraceSettings
from swarm.constants import (
    AGENT_STARTUP_WALL_SEC,
    DOCKER_WORKER_CPUS,
    GLOBAL_EVAL_BASE_SEC,
    GLOBAL_EVAL_CAP_SEC,
    GLOBAL_EVAL_PER_SEED_SEC,
    MODEL_DIR,
    SIM_DT,
    SPEED_FACTOR_MAX_ELIGIBLE,
)
from swarm.core.faults import ReasonCode
from swarm.core.submission_lane import is_model_graph_artifact
from swarm.core.submission_policy import check_safety, validate_submission_zip
from swarm.protocol import (
    SCHEMA_VERSION,
    FailureReason,
    ValidationResult,
    is_supported_schema,
    normalize_version,
)
from swarm.utils.hash import sha256sum
from swarm.validator.calibration import (
    SpeedFactor,
    baseline_model_available,
    baseline_model_path,
    load_baseline_manifest,
    normalize_speed_factor,
    percentile,
)
from swarm.validator.task_gen import task_for_seed_and_type

from ._shared import (
    _docker_evaluator_facade,
    _graph_runtime_template_dir,
    _runtime_profile_env,
    _runtime_profile_from_payload,
    _submission_template_dir,
)

_CALIBRATION_MAX_AGE_SEC = 6 * 3600  # re-measure the host speed factor at least this often


@dataclass(frozen=True)
class HostSpeedCalibration:
    """In-memory, host-level calibration measured under concurrent worker load."""

    speed: SpeedFactor
    worker_count: int
    worker_speeds: tuple[SpeedFactor, ...]
    calibration_version: str
    computed_at: float


_HOST_SPEED_CALIBRATION: Optional[HostSpeedCalibration] = None
_CALIBRATION_CACHE_PATH = (
    Path(__file__).resolve().parents[4] / "state" / "host_speed_factor.json"
)


def _calibration_host_fingerprint() -> str:
    """A measurement is only reusable on the machine that produced it, and only
    until it reboots: a box that came back different must measure itself again."""
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except Exception:
        boot_id = ""
    return f"{os.cpu_count() or 0}:{boot_id}"


def _write_calibration_cache(calibration: HostSpeedCalibration) -> None:
    """Persist an eligible measurement so a restart does not repay for it.

    Only eligible factors are kept: a host that failed the limit must re-measure
    rather than sit out the whole cache window on one bad reading."""
    if not calibration.speed.eligible:
        return
    try:
        _CALIBRATION_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "host": _calibration_host_fingerprint(),
            "calibration_version": calibration.calibration_version,
            "worker_count": int(calibration.worker_count),
            "computed_at": float(calibration.computed_at),
            "local_p90_ms": float(calibration.speed.local_p90_ms),
            "worker_local_p90_ms": [
                float(speed.local_p90_ms) for speed in calibration.worker_speeds
            ],
        }
        temp_path = _CALIBRATION_CACHE_PATH.with_suffix(".tmp")
        temp_path.write_text(json.dumps(payload, sort_keys=True))
        temp_path.replace(_CALIBRATION_CACHE_PATH)
    except Exception as e:
        bt.logging.warning(f"Could not cache the host calibration: {e}")


def _read_calibration_cache(
    *, worker_count: int, calibration_version: str
) -> Optional[HostSpeedCalibration]:
    """Load a cached measurement, or None if it does not describe this host now."""
    try:
        payload = json.loads(_CALIBRATION_CACHE_PATH.read_text())
    except Exception:
        return None
    try:
        if payload["host"] != _calibration_host_fingerprint():
            return None
        if payload["calibration_version"] != str(calibration_version):
            return None
        if int(payload["worker_count"]) < int(worker_count):
            return None
        computed_at = float(payload["computed_at"])
        if (time.time() - computed_at) > _CALIBRATION_MAX_AGE_SEC:
            return None
        speed = normalize_speed_factor(float(payload["local_p90_ms"]))
        worker_speeds = tuple(
            normalize_speed_factor(float(value))
            for value in payload["worker_local_p90_ms"]
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not speed.eligible:
        return None
    return HostSpeedCalibration(
        speed=speed,
        worker_count=int(payload["worker_count"]),
        worker_speeds=worker_speeds,
        calibration_version=str(calibration_version),
        computed_at=computed_at,
    )


def _calibration_mp_context() -> mp.context.BaseContext:
    try:
        return mp.get_context("fork")
    except ValueError:
        return mp.get_context("spawn")


def _prepared_calibration_evaluator(base_image: str):
    from swarm.validator.docker.docker_evaluator import DockerSecureEvaluator

    evaluator = DockerSecureEvaluator.__new__(DockerSecureEvaluator)
    evaluator.base_image = str(base_image)
    evaluator.base_images = {"base": str(base_image)}
    evaluator.base_ready = True
    evaluator.last_fake_model_info = None
    evaluator.family_runtime_profiles = {}
    evaluator.last_runtime_profile_info = None
    evaluator.last_selected_run_image = None
    evaluator.last_selected_worker_limits = None
    evaluator.last_selected_runtime_profile = None
    evaluator.last_selected_runtime_env = None
    DockerSecureEvaluator._base_ready = True
    return evaluator


def _host_calibration_worker_main(worker_id: int, base_image: str, result_queue: Any) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        evaluator = _prepared_calibration_evaluator(base_image)
        speed = loop.run_until_complete(_run_baseline_calibration(evaluator, int(worker_id)))
        if speed is None:
            result_queue.put(
                {
                    "worker_id": int(worker_id),
                    "error": "baseline calibration returned no speed factor",
                }
            )
            return
        result_queue.put(
            {
                "worker_id": int(worker_id),
                "speed": speed,
            }
        )
    except Exception as exc:
        result_queue.put(
            {
                "worker_id": int(worker_id),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    finally:
        asyncio.set_event_loop(None)
        loop.close()


def _docker_cmd_quiet(cmd: list[str], timeout_sec: float = 30.0) -> None:
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────────
# Per-model images — miner dependencies installed before the sandbox closes
# ──────────────────────────────────────────────────────────────────────

_PIP_INSTALL_TIMEOUT_SEC = 120
_BUILD_CACHE_PRUNE_FREE_GB = 25.0


def model_image_tag(model_hash: str) -> str:
    return f"swarm_eval_model_{model_hash[:12]}:latest"


def _image_exists(image_tag: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_tag],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def prepare_model_image(
    self,
    uid: int,
    model_path: Path,
    runtime_profile_payload: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Build a per-model image with the miner's declared dependencies baked in.

    Dependencies are installed here, in a throwaway container that never runs
    miner code, so the evaluation container itself needs no network. Returns the
    image tag, or None when the submission declares no dependencies or the
    install fails. Images are cached by content hash; cleanup() reaps them once
    the model zip is gone.
    """
    if not model_path.is_file():
        return None
    if is_model_graph_artifact(model_path):
        # the graph runner ships with the image; a legacy artifact never
        # contributes dependencies of its own
        return None
    image_tag = model_image_tag(sha256sum(model_path))
    if not model_path.with_suffix(".private").exists() and _image_exists(image_tag):
        return image_tag

    tmpdir = None
    container_name = f"swarm_pip_{uid}_{int(time.time() * 1000)}"
    try:
        current_uid = os.getuid()
        current_gid = os.getgid()
        runtime_profile = (
            _runtime_profile_from_payload(runtime_profile_payload, [])
            if runtime_profile_payload is not None
            else None
        )
        worker_limits = self._resolve_worker_limits(0, runtime_profile=runtime_profile)
        run_image = self.base_image
        if runtime_profile is not None:
            run_image = self._resolve_base_image_for_key(runtime_profile.image_key)

        tmpdir = tempfile.mkdtemp()
        os.chmod(tmpdir, 0o755)
        submission_dir = Path(tmpdir) / "submission"
        submission_dir.mkdir()
        os.chmod(submission_dir, 0o755)

        _extract_submission(model_path, submission_dir)

        # The archive may carry a .pip_done of its own; left in place the wait below
        # would see it immediately and commit the image mid-install.
        (submission_dir / ".pip_done").unlink(missing_ok=True)

        miner_requirements = submission_dir / "requirements.txt"
        if not miner_requirements.exists():
            return None

        if model_path.with_suffix(".private").exists():
            bt.logging.warning(
                f"UID {uid}: private model with requirements.txt rejected "
                "(no pre-lockdown dependency install for private submissions)"
            )
            return None

        if not self._validate_requirements(miner_requirements, uid):
            bt.logging.warning(f"UID {uid}: requirements.txt rejected during image build")
            return None

        startup_script = submission_dir / "startup.sh"
        startup_script.write_text(
            "#!/bin/bash\n"
            "uv pip install --no-cache --python /opt/env/bin/python "
            "-r /workspace/submission/requirements.txt\n"
            "if [ $? -ne 0 ]; then exit 1; fi\n"
            "touch /workspace/submission/.pip_done\n"
            "sleep infinity\n"
        )
        os.chmod(startup_script, 0o755)
        os.chown(startup_script, current_uid, current_gid)

        cmd = [
            "docker", "run", "--rm", "-d",
            "--name", container_name,
            "--user", f"{current_uid}:{current_gid}",
            f"--memory={worker_limits['memory']}",
            f"--cpus={worker_limits['cpus'] or DOCKER_WORKER_CPUS}",
            "--pids-limit=50",
            "--ulimit", "nofile=256:256",
            "--ulimit", "fsize=524288000:524288000",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--network", "bridge",
            "-v", f"{submission_dir}:/workspace/submission:rw",
        ]
        if runtime_profile is not None:
            for key, value in _runtime_profile_env(runtime_profile).items():
                cmd.extend(["-e", f"{key}={value}"])
        cmd.extend([run_image, "bash", "/workspace/submission/startup.sh"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            bt.logging.warning(f"UID {uid}: dependency container start failed: {result.stderr[:200]}")
            return None

        pip_done_flag = submission_dir / ".pip_done"
        pip_start = time.time()
        pip_done = False

        bt.logging.info(f"UID {uid}: installing dependencies...")
        while time.time() - pip_start < _PIP_INSTALL_TIMEOUT_SEC:
            if pip_done_flag.exists():
                pip_done = True
                break
            check = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
                capture_output=True, text=True, timeout=10,
            )
            if check.returncode != 0 or check.stdout.strip() != "true":
                break
            time.sleep(2)

        if not pip_done:
            bt.logging.warning(f"UID {uid}: dependency install failed during image build")
            _docker_cmd_quiet(["docker", "kill", container_name])
            _docker_cmd_quiet(["docker", "rm", "-f", container_name])
            return None

        elapsed = time.time() - pip_start
        commit_result = subprocess.run(
            [
                "docker", "commit",
                "--change", 'CMD ["python", "/workspace/submission/main.py"]',
                container_name, image_tag,
            ],
            capture_output=True, text=True, timeout=120,
        )
        _docker_cmd_quiet(["docker", "kill", container_name])
        _docker_cmd_quiet(["docker", "rm", "-f", container_name])

        if commit_result.returncode != 0:
            bt.logging.warning(f"UID {uid}: docker commit failed: {commit_result.stderr[:200]}")
            return None

        bt.logging.info(f"UID {uid}: model image ready ({image_tag}, install took {elapsed:.1f}s)")
        return image_tag

    except Exception as e:
        bt.logging.warning(f"UID {uid}: prepare_model_image failed: {e}")
        _docker_cmd_quiet(["docker", "kill", container_name])
        _docker_cmd_quiet(["docker", "rm", "-f", container_name])
        return None
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def prune_build_cache_if_disk_low() -> None:
    """Drop the layer cache only when the disk is genuinely tight.

    The cache holds the apt and dependency layers of the base image, so clearing it
    turns the next version bump into a full rebuild that costs the validator
    its evaluation time. Bound the disk, but not on every cleanup.
    """
    try:
        free_gb = shutil.disk_usage("/").free / (1024 ** 3)
    except Exception:
        return
    if free_gb >= _BUILD_CACHE_PRUNE_FREE_GB:
        return
    bt.logging.info(f"Pruning docker build cache: {free_gb:.1f}GiB free")
    try:
        subprocess.run(
            ["docker", "builder", "prune", "-f", "--keep-storage", "5GB"],
            capture_output=True, timeout=120,
        )
    except Exception:
        pass


def remove_model_image(image_tag: str) -> None:
    """Remove a per-model image once its evaluation is finished."""
    try:
        subprocess.run(["docker", "rmi", image_tag], capture_output=True, timeout=15)
    except Exception:
        pass


def remove_all_model_images() -> None:
    """Drop every cached per-model image; stale bases must not survive a base rebuild."""
    try:
        result = subprocess.run(
            [
                "docker", "images", "--format", "{{.Repository}}:{{.Tag}}",
                "--filter", "reference=swarm_eval_model_*",
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0 and result.stdout:
            for img in result.stdout.strip().split("\n"):
                if img:
                    remove_model_image(img)
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

    # Agent runner state
    is_model_graph: bool = False
    submission_dir: Optional[Path] = None
    model_image: Optional[str] = None
    run_image: Optional[str] = None
    current_uid: Optional[int] = None
    current_gid: Optional[int] = None
    worker_limits: Optional[dict] = None
    docker_envs: Optional[dict] = None
    validator_ip: Optional[str] = None
    runtime_profile: Optional[Any] = None

    connected: bool = False
    container_started_at: Optional[float] = None
    container_startup_sec: Optional[float] = None

    # Closure bundle (built in _init_batch_state)
    helpers: Optional[_BatchHelpers] = None


def _init_batch_state(ctx: _BatchContext) -> None:
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    on_seed_complete = ctx.on_seed_complete

    ctx.is_model_graph = is_model_graph_artifact(ctx.model_path)
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

    # a legacy graph artifact carries no drone_agent.py; the runner admits it
    # inside the container, where the ONNX rules are enforced
    if ctx.is_model_graph:
        accepted, detail = check_safety(model_path)
    else:
        accepted, detail = validate_submission_zip(model_path)
    if not accepted:
        bt.logging.warning(
            f"[Worker {worker_id}] UID {uid} submission rejected: {detail}"
        )
        _notify_all_failed(status=ReasonCode.LOAD_FAILED.value, error=detail)
        return [
            ValidationResult(
                uid, False, 0.0, 0.0, failure_reason=ReasonCode.LOAD_FAILED.value
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
                        failure_reason=FailureReason.EVAL_ERROR.value,
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

    summary = (
        f"[Worker {worker_id}] reference calibration: speed_factor={speed.factor:.2f}x "
        f"(local_p90={local_p90:.0f}ms / owner_p90={speed.owner_p90_ms:.0f}ms, n={len(compute)})"
    )
    if speed.eligible:
        bt.logging.info(summary)
    else:
        bt.logging.warning(summary + " — host slower than the eligibility limit; it will not score miners")
    return speed


def _host_calibration_is_valid(
    calibration: Optional[HostSpeedCalibration],
    *,
    worker_count: int,
    calibration_version: str,
) -> bool:
    if calibration is None:
        return False
    if calibration.calibration_version != str(calibration_version):
        return False
    if int(calibration.worker_count) < int(worker_count):
        return False
    return (time.time() - float(calibration.computed_at)) <= _CALIBRATION_MAX_AGE_SEC


def _average_host_speed(worker_speeds: list[SpeedFactor]) -> Optional[SpeedFactor]:
    if not worker_speeds:
        return None
    avg_local_p90 = math.fsum(
        float(speed.local_p90_ms) for speed in worker_speeds
    ) / len(worker_speeds)
    return normalize_speed_factor(avg_local_p90)


async def _run_host_baseline_calibration(self, worker_count: int):
    """Measure one host speed factor under concurrent worker load."""
    global _HOST_SPEED_CALIBRATION

    if not baseline_model_available():
        return None

    manifest = load_baseline_manifest()
    requested = max(1, int(worker_count))
    bt.logging.info(
        f"Starting host reference calibration with {requested} concurrent worker(s)"
    )
    base_image = str(getattr(self, "base_image", "swarm_evaluator_base:latest"))
    ctx = _calibration_mp_context()
    result_queue = ctx.Queue()
    processes = [
        ctx.Process(
            target=_host_calibration_worker_main,
            args=(worker_id, base_image, result_queue),
            name=f"swarm_host_calibration_{worker_id}",
            daemon=True,
        )
        for worker_id in range(requested)
    ]
    for proc in processes:
        proc.start()

    payloads: list[dict[str, Any]] = []
    # A host at the eligibility ceiling needs proportionally longer than the reference.
    deadline = time.monotonic() + SPEED_FACTOR_MAX_ELIGIBLE * max(
        180.0, 90.0 + (requested * 30.0)
    )
    while len(payloads) < requested and time.monotonic() < deadline:
        try:
            payload = result_queue.get(timeout=0.5)
        except queue.Empty:
            if all(not proc.is_alive() for proc in processes):
                break
            await asyncio.sleep(0)
            continue
        if isinstance(payload, dict):
            payloads.append(payload)

    for proc in processes:
        proc.join(timeout=5.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
    try:
        result_queue.close()
    except Exception:
        pass

    speeds_by_worker: dict[int, SpeedFactor] = {}
    seen_workers: set[int] = set()
    for payload in payloads:
        worker_id = int(payload.get("worker_id", -1))
        seen_workers.add(worker_id)
        error = payload.get("error")
        if error:
            bt.logging.warning(
                f"[Worker {worker_id}] host calibration failed: {error}"
            )
            continue
        speed = payload.get("speed")
        if isinstance(speed, SpeedFactor):
            speeds_by_worker[worker_id] = speed

    missing = sorted(set(range(requested)) - seen_workers)
    for worker_id in missing:
        bt.logging.warning(f"[Worker {worker_id}] host calibration produced no result")

    worker_speeds = [
        speeds_by_worker[worker_id]
        for worker_id in range(requested)
        if worker_id in speeds_by_worker
    ]
    if len(worker_speeds) != requested:
        bt.logging.warning(
            f"Host reference calibration failed: {len(worker_speeds)}/{requested} "
            "workers produced usable speed factors"
        )
        return None

    try:
        host_speed = _average_host_speed(worker_speeds)
    except ValueError as e:
        bt.logging.warning(f"Host reference calibration produced invalid speed factor: {e}")
        return None
    if host_speed is None:
        return None
    local_p90s = [float(speed.local_p90_ms) for speed in worker_speeds]
    avg_local_p90 = float(host_speed.local_p90_ms)
    min_local_p90 = min(local_p90s)
    max_local_p90 = max(local_p90s)
    spread = max_local_p90 / min_local_p90 if min_local_p90 > 0.0 else float("inf")

    _HOST_SPEED_CALIBRATION = HostSpeedCalibration(
        speed=host_speed,
        worker_count=requested,
        worker_speeds=tuple(worker_speeds),
        calibration_version=str(manifest["calibration_version"]),
        computed_at=time.time(),
    )
    _write_calibration_cache(_HOST_SPEED_CALIBRATION)
    per_worker = ", ".join(
        f"w{i}={speed.factor:.2f}x/{speed.local_p90_ms:.1f}ms"
        for i, speed in enumerate(worker_speeds)
    )
    summary = (
        f"Host reference calibration: speed_factor={host_speed.factor:.2f}x "
        f"(avg_local_p90={avg_local_p90:.1f}ms / "
        f"owner_p90={host_speed.owner_p90_ms:.1f}ms, workers={requested}, "
        f"min={min_local_p90:.1f}ms, max={max_local_p90:.1f}ms, spread={spread:.2f}x; "
        f"{per_worker})"
    )
    if host_speed.eligible:
        bt.logging.info(summary)
    else:
        bt.logging.warning(
            summary + " — host slower than the eligibility limit; it will not score miners"
        )
    return host_speed


def host_speed_factor_is_fresh(worker_count: int) -> bool:
    """True when scoring can start without stopping to measure the host first."""
    global _HOST_SPEED_CALIBRATION

    try:
        manifest = load_baseline_manifest()
    except Exception:
        return False
    requested = max(1, int(worker_count))
    version = str(manifest["calibration_version"])
    if _host_calibration_is_valid(
        _HOST_SPEED_CALIBRATION, worker_count=requested, calibration_version=version,
    ):
        return True
    cached = _read_calibration_cache(
        worker_count=requested, calibration_version=version,
    )
    if cached is None:
        return False
    _HOST_SPEED_CALIBRATION = cached
    bt.logging.info(
        f"Reusing the cached host calibration: speed_factor={cached.speed.factor:.2f}x "
        f"(measured {(time.time() - cached.computed_at) / 60.0:.0f} min ago, "
        f"workers={cached.worker_count})"
    )
    return True


async def _ensure_host_speed_factor(self, worker_count: int):
    """Return the in-memory host speed factor, calibrated under concurrent load."""
    global _HOST_SPEED_CALIBRATION

    if not baseline_model_available():
        return None
    requested = max(1, int(worker_count))
    if host_speed_factor_is_fresh(requested):
        return _HOST_SPEED_CALIBRATION.speed
    return await _run_host_baseline_calibration(self, requested)


def _extract_submission(model_path: Path, submission_dir: Path) -> None:
    """Extract the miner's zip, flattening a single wrapping directory if present."""
    with zipfile.ZipFile(model_path, "r") as zf:
        zf.extractall(submission_dir)

    contents = list(submission_dir.iterdir())
    if len(contents) == 1 and contents[0].is_dir():
        nested_dir = contents[0]
        for item in nested_dir.iterdir():
            target = submission_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
        nested_dir.rmdir()


def _setup_workspace(ctx: _BatchContext) -> Optional[list]:
    """Extract the submission and stage the RPC server next to the miner's agent."""
    runtime_profile = _runtime_profile_from_payload(ctx.runtime_profile_payload, ctx.tasks)
    worker_limits = ctx.self._resolve_worker_limits(ctx.worker_id, runtime_profile=runtime_profile)
    thread_cap = DockerRuntimeSettings.worker_thread_cap(worker_limits)
    docker_envs = ctx.self._docker_env_overrides(thread_cap=thread_cap)
    docker_envs.update(_runtime_profile_env(runtime_profile))
    docker_envs.update({
        "SWARM_AGENT_PORT": "8000",
        "SWARM_START_GATE": _START_GATE_PATH,
    })

    current_uid = os.getuid()
    current_gid = os.getgid()

    tmpdir = tempfile.mkdtemp()
    ctx.tmpdir = tmpdir  # set before chown/chmod so the outer finally still cleans up
    os.chown(tmpdir, current_uid, current_gid)
    os.chmod(tmpdir, 0o755)

    submission_dir = Path(tmpdir) / "submission"
    submission_dir.mkdir(exist_ok=True)
    os.chown(submission_dir, current_uid, current_gid)
    os.chmod(submission_dir, 0o755)

    try:
        if ctx.is_model_graph:
            # the graph runner reads the archive itself, so nothing is unpacked
            # and no file from it can shadow the bootstrap we stage below
            shutil.copy(ctx.model_path, submission_dir / _GRAPH_ARTIFACT_NAME)
        else:
            _extract_submission(ctx.model_path, submission_dir)
    except Exception as exc:
        ctx.helpers.notify_all_failed(
            status=ReasonCode.LOAD_FAILED.value, error=f"extract failed: {exc}"
        )
        return [
            ValidationResult(ctx.uid, False, 0.0, 0.0, failure_reason=ReasonCode.LOAD_FAILED.value)
            for _ in ctx.tasks
        ]

    template_dir = _submission_template_dir()
    if ctx.is_model_graph:
        shutil.copy(template_dir / "agent.capnp", submission_dir)
        shutil.copy(_graph_runtime_template_dir() / "main.py", submission_dir)
        docker_envs["SWARM_MODEL_GRAPH_ARTIFACT"] = (
            f"/workspace/submission/{_GRAPH_ARTIFACT_NAME}"
        )
    else:
        for name in ("agent.capnp", "agent_server.py", "main.py", "runtime_caps.py"):
            shutil.copy(template_dir / name, submission_dir)

    for f in submission_dir.iterdir():
        if f.is_file():
            os.chown(f, current_uid, current_gid)
            os.chmod(f, 0o644)

    ctx.submission_dir = submission_dir
    ctx.current_uid = current_uid
    ctx.current_gid = current_gid
    ctx.worker_limits = worker_limits
    ctx.docker_envs = docker_envs
    ctx.run_image = ctx.model_image or ctx.self._resolve_base_image_for_key(
        runtime_profile.image_key
    )
    ctx.validator_ip = ctx.self._get_docker_host_ip()
    ctx.runtime_profile = runtime_profile
    ctx.self.last_selected_runtime_profile = runtime_profile.as_dict()
    ctx.self.last_selected_worker_limits = dict(worker_limits)
    ctx.self.last_selected_runtime_env = dict(docker_envs)
    ctx.self.last_selected_run_image = str(ctx.run_image)
    return None


_GRAPH_ARTIFACT_NAME = "model_graph.zip"
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


def _launch_container(ctx: _BatchContext) -> Optional[list]:
    obs_shm_path = _create_obs_shm(ctx.host_port)
    cmd = [
        "docker", "run", "--rm", "-d", "--name", ctx.container_name,
        "--user", f"{ctx.current_uid}:{ctx.current_gid}",
        f"--memory={ctx.worker_limits['memory']}",
        "--pids-limit=50", "--ulimit", "nofile=256:256",
        "--ulimit", "fsize=52428800:52428800", "--security-opt", "no-new-privileges",
        "--cap-drop", "ALL", "--network", "bridge", "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "-p", f"127.0.0.1:{ctx.host_port}:8000",
        "-v", f"{ctx.submission_dir}:/workspace/submission:rw",
    ]
    if ctx.worker_limits["cpus"]:
        cmd.append(f"--cpus={ctx.worker_limits['cpus']}")
    if ctx.worker_limits["cpuset_cpus"]:
        cmd.extend(["--cpuset-cpus", str(ctx.worker_limits["cpuset_cpus"])])
    for key, value in ctx.docker_envs.items():
        cmd.extend(["-e", f"{key}={value}"])
    if obs_shm_path:
        cmd.extend(["-v", f"{obs_shm_path}:/workspace/obs_shm.bin:ro"])
        cmd.extend(["-e", "SWARM_OBS_SHM=/workspace/obs_shm.bin"])
    cmd.extend([ctx.run_image, "python", "/workspace/submission/main.py"])
    ctx.container_started_at = time.monotonic()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        ctx.helpers.notify_all_failed(status=ReasonCode.INFRA_DOCKER.value, error=result.stderr[:300])
        return [ValidationResult(ctx.uid, False, 0.0, 0.0, failure_reason=ReasonCode.INFRA_DOCKER.value) for _ in ctx.tasks]
    return None


async def _prepare_network_and_rpc(ctx: _BatchContext) -> Optional[list]:
    container_pid = ctx.self._get_container_pid(ctx.container_name)
    if (
        not container_pid
        or not ctx.self._apply_network_lockdown(container_pid, ctx.validator_ip)
        or not _open_start_gate(ctx.container_name)
    ):
        ctx.helpers.run_docker_cmd_quiet(["docker", "rm", "-f", ctx.container_name])
        ctx.helpers.notify_all_failed(status=ReasonCode.INFRA_DOCKER.value)
        return [ValidationResult(ctx.uid, False, 0.0, 0.0, failure_reason=ReasonCode.INFRA_DOCKER.value) for _ in ctx.tasks]
    deadline = time.monotonic() + AGENT_STARTUP_WALL_SEC
    while time.monotonic() < deadline:
        if ctx.self._check_rpc_ready(ctx.host_port):
            ctx.connected = True
            started_at = getattr(ctx, "container_started_at", None)
            if started_at is not None:
                startup_sec = time.monotonic() - started_at
                ctx.container_startup_sec = startup_sec
                bt.logging.info(
                    f"[Worker {ctx.worker_id}] container ready in {startup_sec:.1f}s"
                )
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
    host_speed_factor: Optional[float] = None,
    model_image: Optional[str] = None,
) -> list:
    """Evaluate multiple seeds in a single container.

    Args:
        tasks: List of MapTask objects (one per seed)
        uid: Miner UID
        model_path: Path to model zip file
        worker_id: Worker ID for logging (0 to N_DOCKER_WORKERS-1)
        model_image: Pre-built image carrying the miner's declared dependencies

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
        model_image=model_image,
    )

    _init_batch_state(ctx)

    early = _validate_inputs(ctx)
    if early is not None:
        return early

    if not is_calibration_run:
        speed = None
        if host_speed_factor is not None:
            try:
                factor = float(host_speed_factor)
            except (TypeError, ValueError):
                factor = 0.0
            if math.isfinite(factor) and factor > 0.0:
                ctx.speed_factor = factor
            else:
                host_speed_factor = None

        if ctx.speed_factor is None:
            speed = await _ensure_host_speed_factor(self, 1)
            if speed is not None and speed.eligible:
                ctx.speed_factor = speed.factor

        if ctx.speed_factor is None or (speed is not None and not speed.eligible):
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

    _setup_pretry_state(ctx)

    try:
        t0 = time.monotonic()
        early = _setup_workspace(ctx)
        if early is not None:
            return early

        t1 = time.monotonic()
        early = _launch_container(ctx)
        if early is not None:
            return early

        t2 = time.monotonic()
        early = await _prepare_network_and_rpc(ctx)
        if early is not None:
            return early

        t3 = time.monotonic()
        results = await _run_rpc_phase(ctx)
        t4 = time.monotonic()
        bt.logging.info(
            f"[Worker {ctx.worker_id}] seed timing: setup {t1 - t0:.1f}s · "
            f"container {t2 - t1:.1f}s · rpc {t3 - t2:.1f}s · "
            f"mission {t4 - t3:.1f}s · total {t4 - t0:.1f}s"
        )
        return results

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

        result_pip = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=swarm_pip_", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
        )
        if result_pip.returncode == 0 and result_pip.stdout:
            for container in result_pip.stdout.strip().split("\n"):
                if container:
                    subprocess.run(
                        ["docker", "rm", "-f", container], capture_output=True, timeout=30
                    )

        result_images = subprocess.run(
            [
                "docker", "images", "--format", "{{.Repository}}:{{.Tag}}",
                "--filter", "reference=swarm_eval_model_*",
            ],
            capture_output=True,
            text=True,
        )
        if result_images.returncode == 0 and result_images.stdout:
            live_tags = set()
            for zip_fp in MODEL_DIR.glob("*.zip"):
                try:
                    live_tags.add(model_image_tag(sha256sum(zip_fp)))
                except Exception:
                    continue
            for img in result_images.stdout.strip().split("\n"):
                if img and img not in live_tags:
                    remove_model_image(img)

        subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
        subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
        prune_build_cache_if_disk_low()

    except Exception as e:
        bt.logging.warning(f"Container cleanup failed: {e}")
