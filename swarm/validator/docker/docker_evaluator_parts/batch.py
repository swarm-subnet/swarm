import asyncio
import math
import os
import random
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

from swarm.config import DockerBatchTimeoutSettings, RpcTraceSettings
from swarm.constants import (
    GLOBAL_EVAL_BASE_SEC,
    GLOBAL_EVAL_CAP_SEC,
    GLOBAL_EVAL_PER_SEED_SEC,
    HORIZON_SEC,
    SIM_DT,
    USE_REFERENCE_TIMING,
)
from swarm.core.model_verify import add_to_blacklist
from swarm.core.submission_policy import REQUIRED_ROOT_FILES
from swarm.protocol import MapTask, ValidationResult
from swarm.utils.hash import sha256sum
from swarm.validator.calibration import (
    CALIBRATION_STATE,
    baseline_model_available,
    baseline_model_path,
    load_baseline_manifest,
    normalize_speed_factor,
    percentile,
)

_CALIBRATION_MAX_AGE_SEC = 6 * 3600  # re-measure the host speed factor at least this often
_CALIBRATION_MIN_SAMPLES = 100       # below this, the measurement is untrustworthy

from ._shared import _docker_evaluator_facade, _submission_template_dir


def _docker_cmd_quiet(cmd: list[str], timeout_sec: float = 30.0) -> None:
    try:
        subprocess.run(cmd, capture_output=True, timeout=timeout_sec)
    except Exception:
        pass


def prepare_model_image(
    self,
    uid: int,
    model_path: Path,
) -> Optional[str]:
    """Build a per-model Docker image with pip dependencies pre-installed.

    Returns the image tag if requirements.txt exists and pip succeeds, None otherwise.
    The caller must call remove_model_image() when done.
    """
    if not model_path.is_file():
        return None

    tmpdir = None
    container_name = f"swarm_pip_{uid}_{int(time.time() * 1000)}"
    try:
        current_uid = os.getuid()
        current_gid = os.getgid()
        worker_limits = self._resolve_worker_limits(0)

        tmpdir = tempfile.mkdtemp()
        os.chmod(tmpdir, 0o755)
        submission_dir = Path(tmpdir) / "submission"
        submission_dir.mkdir()
        os.chmod(submission_dir, 0o755)

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

        template_dir = _submission_template_dir()
        shutil.copy(template_dir / "agent.capnp", submission_dir)
        shutil.copy(template_dir / "agent_server.py", submission_dir)
        shutil.copy(template_dir / "main.py", submission_dir)

        for f in submission_dir.iterdir():
            os.chown(f, current_uid, current_gid)
            os.chmod(f, 0o644)

        miner_requirements = submission_dir / "requirements.txt"
        if not miner_requirements.exists():
            return None

        if not self._validate_requirements(miner_requirements, uid):
            bt.logging.warning(f"UID {uid}: requirements.txt rejected during image build")
            return None

        startup_script = submission_dir / "startup.sh"
        with open(startup_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write("pip install --no-cache-dir --user -r /workspace/submission/requirements.txt\n")
            f.write("if [ $? -ne 0 ]; then exit 1; fi\n")
            f.write("touch /workspace/submission/.pip_done\n")
            f.write("sleep infinity\n")
        os.chmod(startup_script, 0o755)
        os.chown(startup_script, current_uid, current_gid)

        cmd = [
            "docker", "run", "--rm", "-d",
            "--name", container_name,
            "--user", f"{current_uid}:{current_gid}",
            f"--memory={worker_limits['memory']}",
            f"--cpus={worker_limits['cpus']}",
            "--pids-limit=50",
            "--ulimit", "nofile=256:256",
            "--ulimit", "fsize=524288000:524288000",
            "--security-opt", "no-new-privileges",
            "--cap-drop", "ALL",
            "--network", "bridge",
            "-v", f"{submission_dir}:/workspace/submission:rw",
            self.base_image,
            "bash", "/workspace/submission/startup.sh",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            bt.logging.warning(f"UID {uid}: pip container start failed: {result.stderr[:200]}")
            return None

        pip_done_flag = submission_dir / ".pip_done"
        pip_start = time.time()
        pip_timeout = 120
        pip_done = False

        bt.logging.info(f"UID {uid}: installing pip dependencies...")
        while time.time() - pip_start < pip_timeout:
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
            bt.logging.warning(f"UID {uid}: pip install failed during image build")
            _docker_cmd_quiet(["docker", "kill", container_name])
            _docker_cmd_quiet(["docker", "rm", "-f", container_name])
            return None

        elapsed = time.time() - pip_start
        model_hash = sha256sum(model_path)[:12]
        image_tag = f"swarm_eval_model_{model_hash}:latest"

        commit_result = subprocess.run(
            ["docker", "commit", container_name, image_tag],
            capture_output=True, text=True, timeout=30,
        )
        _docker_cmd_quiet(["docker", "kill", container_name])
        _docker_cmd_quiet(["docker", "rm", "-f", container_name])

        if commit_result.returncode != 0:
            bt.logging.warning(f"UID {uid}: docker commit failed: {commit_result.stderr[:200]}")
            return None

        bt.logging.info(f"UID {uid}: model image ready ({image_tag}, pip took {elapsed:.1f}s)")
        return image_tag

    except Exception as e:
        bt.logging.warning(f"UID {uid}: prepare_model_image failed: {e}")
        try:
            _docker_cmd_quiet(["docker", "kill", container_name])
            _docker_cmd_quiet(["docker", "rm", "-f", container_name])
        except Exception:
            pass
        return None
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def remove_model_image(image_tag: str) -> None:
    """Remove a per-model Docker image after evaluation completes."""
    try:
        subprocess.run(["docker", "rmi", image_tag], capture_output=True, timeout=15)
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
    model_image: Optional[str] = None
    speed_factor: Optional[float] = None
    is_calibration_run: bool = False

    # Trace + sync primitives (built in _init_batch_state)
    trace_rpc: bool = False
    stop_event: Optional[threading.Event] = None
    completed_lock: Optional[threading.Lock] = None
    progress_state: Optional[dict] = None

    # Pre-try state (set by _setup_pretry_state)
    container_name: Optional[str] = None
    host_port: Optional[int] = None
    tmpdir: Optional[str] = None

    # Workspace state (set by _setup_workspace)
    submission_dir: Optional[Path] = None
    template_dir: Optional[Path] = None
    has_requirements: bool = False
    run_image: Optional[str] = None
    current_uid: Optional[int] = None
    current_gid: Optional[int] = None
    worker_limits: Optional[dict] = None
    docker_envs: Optional[dict] = None
    validator_ip: Optional[str] = None

    # Pip / readiness flags
    pip_done: bool = False
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
            "moving_platform": bool(getattr(task_obj, "moving_platform", False)),
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


def _validate_inputs(ctx: _BatchContext) -> Optional[list]:
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    model_path = ctx.model_path
    _notify_all_failed = ctx.helpers.notify_all_failed

    if not model_path.is_file():
        bt.logging.warning(f"[Worker {worker_id}] Model path missing: {model_path}")
        _notify_all_failed(status="model_path_missing")
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    if not _docker_evaluator_facade().DockerSecureEvaluator._base_ready:
        bt.logging.warning(f"[Worker {worker_id}] Docker not ready for UID {uid}")
        _notify_all_failed(status="docker_not_ready")
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            namelist = zf.namelist()
            missing = [f for f in REQUIRED_ROOT_FILES if f not in namelist]
            if missing:
                bt.logging.warning(
                    f"[Worker {worker_id}] Model {uid} missing required files: {missing}"
                )
                _notify_all_failed(status="submission_missing_drone_agent")
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
    except Exception as e:
        bt.logging.warning(
            f"[Worker {worker_id}] Failed to validate model {uid}: {e}"
        )
        _notify_all_failed(
            status="submission_validation_failed",
            error=f"{type(e).__name__}: {e}",
        )
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    return None


def _setup_pretry_state(ctx: _BatchContext) -> None:
    self = ctx.self
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    _phase = ctx.helpers.phase

    ctx.container_name = f"swarm_eval_{uid}_w{worker_id}_{int(time.time() * 1000)}"
    ctx.host_port = self._find_free_port()

    _phase(
        f"prepare container={ctx.container_name} host_port={ctx.host_port} seeds={len(tasks)}"
    )
    # ctx.tmpdir defaults to None on the dataclass — matches original line 349.


def _setup_workspace(ctx: _BatchContext) -> Optional[list]:
    self = ctx.self
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    model_path = ctx.model_path
    model_image = ctx.model_image
    _notify_all_failed = ctx.helpers.notify_all_failed

    current_uid = os.getuid()
    current_gid = os.getgid()
    worker_limits = self._resolve_worker_limits(worker_id)
    docker_envs = self._docker_env_overrides()

    tmpdir = tempfile.mkdtemp()
    ctx.tmpdir = tmpdir  # set BEFORE chown/chmod so outer finally cleans up if either raises
    os.chown(tmpdir, current_uid, current_gid)
    os.chmod(tmpdir, 0o755)

    submission_dir = Path(tmpdir) / "submission"
    submission_dir.mkdir(exist_ok=True)
    os.chown(submission_dir, current_uid, current_gid)
    os.chmod(submission_dir, 0o755)

    template_dir = _submission_template_dir()

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

    shutil.copy(template_dir / "agent.capnp", submission_dir)
    shutil.copy(template_dir / "agent_server.py", submission_dir)
    shutil.copy(template_dir / "main.py", submission_dir)

    for f in submission_dir.iterdir():
        os.chown(f, current_uid, current_gid)
        os.chmod(f, 0o644)

    miner_requirements = submission_dir / "requirements.txt"
    has_requirements = miner_requirements.exists() and model_image is None

    if has_requirements and not self._validate_requirements(
        miner_requirements, uid
    ):
        bt.logging.warning(
            f"[Worker {worker_id}] UID {uid} requirements.txt rejected"
        )
        _notify_all_failed(status="requirements_rejected")
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    validator_ip = self._get_docker_host_ip()
    run_image = model_image or self.base_image

    if has_requirements:
        bt.logging.info(
            f"[Worker {worker_id}] Miner has requirements.txt for UID {uid}"
        )
        startup_script = submission_dir / "startup.sh"
        with open(startup_script, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(
                "pip install --no-cache-dir --user -r /workspace/submission/requirements.txt\n"
            )
            f.write("if [ $? -ne 0 ]; then exit 1; fi\n")
            f.write("touch /workspace/submission/.pip_done\n")
            f.write("sleep infinity\n")
        os.chmod(startup_script, 0o755)
        os.chown(startup_script, current_uid, current_gid)

    ctx.current_uid = current_uid
    ctx.current_gid = current_gid
    ctx.worker_limits = worker_limits
    ctx.docker_envs = docker_envs
    ctx.submission_dir = submission_dir
    ctx.template_dir = template_dir
    ctx.has_requirements = has_requirements
    ctx.run_image = run_image
    ctx.validator_ip = validator_ip
    return None


def _launch_container(ctx: _BatchContext) -> Optional[list]:
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    container_name = ctx.container_name
    host_port = ctx.host_port
    submission_dir = ctx.submission_dir
    has_requirements = ctx.has_requirements
    run_image = ctx.run_image
    current_uid = ctx.current_uid
    current_gid = ctx.current_gid
    worker_limits = ctx.worker_limits
    docker_envs = ctx.docker_envs
    _phase = ctx.helpers.phase
    _notify_all_failed = ctx.helpers.notify_all_failed

    if has_requirements:
        cmd = [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "--user",
            f"{current_uid}:{current_gid}",
            f"--memory={worker_limits['memory']}",
            f"--cpus={worker_limits['cpus']}",
            "--pids-limit=50",
            "--ulimit",
            "nofile=256:256",
            "--ulimit",
            "fsize=524288000:524288000",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--network",
            "bridge",
            "-p",
            f"127.0.0.1:{host_port}:8000",
            "-v",
            f"{submission_dir}:/workspace/submission:rw",
        ]
        if worker_limits["cpuset_cpus"]:
            cmd.extend(["--cpuset-cpus", str(worker_limits["cpuset_cpus"])])
        for key, value in docker_envs.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(
            [
                run_image,
                "bash",
                "/workspace/submission/startup.sh",
            ]
        )
    else:
        cmd = [
            "docker",
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "--user",
            f"{current_uid}:{current_gid}",
            f"--memory={worker_limits['memory']}",
            f"--cpus={worker_limits['cpus']}",
            "--pids-limit=20",
            "--ulimit",
            "nofile=256:256",
            "--ulimit",
            "fsize=524288000:524288000",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--network",
            "bridge",
            "-p",
            f"127.0.0.1:{host_port}:8000",
            "-v",
            f"{submission_dir}:/workspace/submission:ro",
        ]
        if worker_limits["cpuset_cpus"]:
            cmd.extend(["--cpuset-cpus", str(worker_limits["cpuset_cpus"])])
        for key, value in docker_envs.items():
            cmd.extend(["-e", f"{key}={value}"])
        cmd.extend(
            [
                run_image,
                "bash",
                "-c",
                "sleep infinity",
            ]
        )

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

    if result.returncode != 0:
        bt.logging.warning(
            f"[Worker {worker_id}] Container start failed: {result.stderr[:300]}"
        )
        _notify_all_failed(
            status="container_start_failed",
            error=result.stderr[:300],
        )
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
    _phase("container started successfully")
    return None


async def _wait_for_pip_done(ctx: _BatchContext) -> Optional[list]:
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    container_name = ctx.container_name
    submission_dir = ctx.submission_dir
    has_requirements = ctx.has_requirements
    _phase = ctx.helpers.phase
    _run_docker_cmd_quiet = ctx.helpers.run_docker_cmd_quiet
    _notify_all_failed = ctx.helpers.notify_all_failed

    pip_timeout = 120 if has_requirements else 0
    rpc_timeout = 30
    connected = False
    pip_done = False

    if has_requirements:
        bt.logging.info(
            f"[Worker {worker_id}] Waiting for pip install (max {pip_timeout}s)..."
        )
        _phase(f"waiting pip install (timeout={pip_timeout}s)")
        pip_start = time.time()
        pip_done_flag = submission_dir / ".pip_done"

        while time.time() - pip_start < pip_timeout:
            if pip_done_flag.exists():
                pip_done = True
                elapsed = time.time() - pip_start
                bt.logging.info(
                    f"[Worker {worker_id}] pip done in {elapsed:.1f}s"
                )
                _phase(f"pip install complete in {elapsed:.1f}s")
                break

            check = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "-f",
                    "{{.State.Running}}",
                    container_name,
                ],
                capture_output=True,
                text=True,
            )
            if check.returncode != 0 or check.stdout.strip() != "true":
                bt.logging.warning(
                    f"[Worker {worker_id}] Container stopped during pip"
                )
                break

            await asyncio.sleep(2)

        if not pip_done:
            bt.logging.warning(
                f"[Worker {worker_id}] pip install failed for UID {uid}"
            )
            _phase("pip install failed")
            _run_docker_cmd_quiet(["docker", "kill", container_name])
            _run_docker_cmd_quiet(["docker", "rm", "-f", container_name])
            _notify_all_failed(status="pip_install_failed")
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
    else:
        pip_done = True
        _phase("no pip install required")

    ctx.pip_done = pip_done
    ctx.connected = connected
    return None


async def _prepare_network_and_rpc(ctx: _BatchContext) -> Optional[list]:
    self = ctx.self
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    container_name = ctx.container_name
    host_port = ctx.host_port
    submission_dir = ctx.submission_dir
    template_dir = ctx.template_dir
    has_requirements = ctx.has_requirements
    validator_ip = ctx.validator_ip
    trace_rpc = ctx.trace_rpc
    _phase = ctx.helpers.phase
    _run_docker_cmd_quiet = ctx.helpers.run_docker_cmd_quiet
    _notify_all_failed = ctx.helpers.notify_all_failed

    if has_requirements:
        shutil.copy(template_dir / "agent.capnp", submission_dir)
        shutil.copy(template_dir / "agent_server.py", submission_dir)
        shutil.copy(template_dir / "main.py", submission_dir)

    container_pid = self._get_container_pid(container_name)
    if not container_pid:
        bt.logging.warning(f"[Worker {worker_id}] Failed to get container PID")
        _phase("failed to resolve container pid")
        _run_docker_cmd_quiet(["docker", "kill", container_name])
        _run_docker_cmd_quiet(["docker", "rm", "-f", container_name])
        _notify_all_failed(status="container_pid_missing")
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    _phase(
        f"applying network lockdown pid={container_pid} validator_ip={validator_ip}"
    )
    if not self._apply_network_lockdown(container_pid, validator_ip):
        bt.logging.warning(f"[Worker {worker_id}] Network lockdown failed")
        _phase("network lockdown failed")
        _run_docker_cmd_quiet(["docker", "kill", container_name])
        _run_docker_cmd_quiet(["docker", "rm", "-f", container_name])
        _notify_all_failed(status="network_lockdown_failed")
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
    _phase("network lockdown applied")

    exec_result = subprocess.run(
        [
            "docker",
            "exec",
            "-d",
            container_name,
            "python",
            "/workspace/submission/main.py",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if exec_result.returncode != 0:
        bt.logging.warning(
            f"[Worker {worker_id}] Failed to start main.py: {exec_result.stderr[:200]}"
        )
        _phase("failed to launch submission main.py")
        _run_docker_cmd_quiet(["docker", "kill", container_name])
        _run_docker_cmd_quiet(["docker", "rm", "-f", container_name])
        _notify_all_failed(
            status="submission_start_failed",
            error=exec_result.stderr[:200],
        )
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
    _phase("submission main.py launched")

    rpc_start = time.time()
    max_rpc_wait = 30
    rpc_check_interval = 0.1
    rpc_check_count = 0
    connected = False
    _phase(f"waiting for rpc readiness (max_wait={max_rpc_wait}s)")

    while time.time() - rpc_start < max_rpc_wait:
        rpc_check_count += 1
        if self._check_rpc_ready(host_port):
            connected = True
            elapsed = time.time() - rpc_start
            _phase(f"rpc ready after {elapsed:.1f}s")
            break
        if trace_rpc and rpc_check_count % 30 == 0:
            waited = time.time() - rpc_start
            _phase(f"rpc not ready yet ({waited:.1f}s elapsed)")
        await asyncio.sleep(rpc_check_interval)

    if not connected:
        bt.logging.warning(f"[Worker {worker_id}] RPC connection failed")
        _phase("rpc readiness failed")
        _run_docker_cmd_quiet(["docker", "kill", container_name])
        _run_docker_cmd_quiet(["docker", "rm", "-f", container_name])
        _notify_all_failed(status="rpc_connection_failed")
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    container_check = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if (
        container_check.returncode != 0
        or "true" not in container_check.stdout.lower()
    ):
        bt.logging.warning(
            f"[Worker {worker_id}] Container stopped before evaluation"
        )
        _phase("container not running before rpc batch")
        _run_docker_cmd_quiet(["docker", "rm", "-f", container_name])
        _notify_all_failed(status="container_stopped_before_eval")
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    ctx.connected = connected
    return None


async def _run_rpc_phase(ctx: _BatchContext) -> list:
    """Owns the inner try/finally entirely. Diagnostics + result validation
    happen INSIDE the try, BEFORE the inner finally (mirrors original 667-900)."""
    self = ctx.self
    uid = ctx.uid
    worker_id = ctx.worker_id
    tasks = ctx.tasks
    model_path = ctx.model_path
    container_name = ctx.container_name
    host_port = ctx.host_port
    rollout_observer = ctx.rollout_observer
    stop_event = ctx.stop_event
    progress_state = ctx.progress_state
    task_offset = ctx.task_offset
    task_total = ctx.task_total
    _phase = ctx.helpers.phase
    _on_seed_complete_guarded = ctx.helpers.on_seed_complete_guarded
    _run_docker_cmd_quiet = ctx.helpers.run_docker_cmd_quiet
    _notify_all_failed = ctx.helpers.notify_all_failed

    try:
        base_batch_timeout = min(
            GLOBAL_EVAL_BASE_SEC + GLOBAL_EVAL_PER_SEED_SEC * len(tasks),
            GLOBAL_EVAL_CAP_SEC,
        )
        timeout_settings = DockerBatchTimeoutSettings.from_env()
        timeout_multiplier = timeout_settings.multiplier
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
                    speed_factor=ctx.speed_factor,
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
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

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
                model_hash = sha256sum(model_path)
                add_to_blacklist(model_hash)
                valid_results.append(ValidationResult(uid, False, 0.0, 0.0))

        _phase(f"batch complete ({len(valid_results)} result(s))")
        return valid_results
    finally:
        stop_event.set()
        _run_docker_cmd_quiet(["docker", "kill", container_name])
        _run_docker_cmd_quiet(["docker", "rm", "-f", container_name])
        _phase("container cleaned up")


def _calibration_task(seed: int, challenge_type: int) -> MapTask:
    """A long, known-goal flight (no search, obstacle-light) so the baseline model
    survives long enough to yield enough act() timing samples — independent of the
    live map difficulty. Per-act() compute is map-independent, so the measured
    speed factor stays comparable to the owner's committed baseline.
    """
    rng = random.Random(seed)
    angle = rng.uniform(0.0, 2.0 * math.pi)
    dist = 45.0
    goal = (dist * math.cos(angle), dist * math.sin(angle), 4.0)
    return MapTask(
        map_seed=seed,
        start=(0.0, 0.0, 1.121),
        goal=goal,
        sim_dt=SIM_DT,
        horizon=HORIZON_SEC,
        challenge_type=challenge_type,
        search_radius=0.0,
        moving_platform=False,
        version="1",
    )


async def _run_baseline_calibration(self, worker_id: int):
    """Measure this worker's speed factor by flying the committed baseline model.

    speed_factor = local_p90 / owner_p90, so every miner act() can be judged in
    baseline-equivalent time regardless of how fast this validator host is.
    """
    manifest = load_baseline_manifest()
    model = manifest["baseline_model"]
    measurement = manifest.get("measurement", {})
    seeds = [int(s) for s in measurement.get("sample_seeds", [1001])]
    warmup = int(measurement.get("warmup_steps", 1))
    ctype = int(model["run_as_challenge_type"])
    tasks = [_calibration_task(seed, ctype) for seed in seeds]

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
            self, tasks, 0, baseline_model_path(),
            worker_id=worker_id, on_seed_complete=_on_seed,
            rollout_observer=_observer, is_calibration_run=True,
        )
    except Exception as e:
        bt.logging.warning(f"[Worker {worker_id}] baseline calibration failed: {e}")
        return None

    compute = [a - overhead["ms"] for a in act_ms if a - overhead["ms"] > 0.0]
    if len(compute) < _CALIBRATION_MIN_SAMPLES:
        bt.logging.warning(
            f"[Worker {worker_id}] baseline calibration produced {len(compute)} samples "
            f"(<{_CALIBRATION_MIN_SAMPLES}); not enough to trust"
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
        bt.logging.warning(summary + " — host much slower than baseline; scores may be noisier")
    return speed


async def _ensure_worker_speed_factor(self, worker_id: int):
    """Cached/freshly-measured speed factor for this worker; ``None`` -> legacy timing.

    When the baseline model is present we retry once before conceding, and log
    loudly, so a silent legacy fallback never hides the hardware-fairness bug.
    """
    if not baseline_model_available():
        return None
    if CALIBRATION_STATE.is_stale(worker_id, max_age_sec=_CALIBRATION_MAX_AGE_SEC):
        if await _run_baseline_calibration(self, worker_id) is None:
            await _run_baseline_calibration(self, worker_id)  # retry once
    entry = CALIBRATION_STATE.get(worker_id)
    if entry is None:
        bt.logging.warning(
            f"[Worker {worker_id}] reference calibration unavailable after retry; "
            f"falling back to legacy timing for this batch"
        )
        return None
    return entry.speed.factor


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
    model_image: Optional[str] = None,
    is_calibration_run: bool = False,
) -> list:
    """Evaluate multiple seeds in a single container.

    Args:
        tasks: List of MapTask objects (one per seed)
        uid: Miner UID
        model_path: Path to model zip file
        worker_id: Worker ID for logging (0 to N_DOCKER_WORKERS-1)
        model_image: Pre-built image with pip deps (skips pip install when set)

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
        model_image=model_image,
        is_calibration_run=is_calibration_run,
    )

    _init_batch_state(ctx)

    early = _validate_inputs(ctx)
    if early is not None:
        return early

    if not ctx.is_calibration_run and USE_REFERENCE_TIMING and baseline_model_available():
        ctx.speed_factor = await _ensure_worker_speed_factor(self, worker_id)

    _setup_pretry_state(ctx)

    try:
        early = _setup_workspace(ctx)
        if early is not None:
            return early

        early = _launch_container(ctx)
        if early is not None:
            return early

        early = await _wait_for_pip_done(ctx)
        if early is not None:
            return early

        early = await _prepare_network_and_rpc(ctx)
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

    return [ValidationResult(uid, False, 0.0, 0.0) for _ in ctx.tasks]


def cleanup(self):
    """Clean up any orphaned containers and prune unused images/cache"""
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
            [
                "docker", "ps", "-a",
                "--filter", "name=swarm_pip_",
                "--format", "{{.Names}}",
            ],
            capture_output=True, text=True,
        )
        if result_pip.returncode == 0 and result_pip.stdout:
            for c in result_pip.stdout.strip().split("\n"):
                if c:
                    subprocess.run(["docker", "rm", "-f", c], capture_output=True, timeout=30)

        result_images = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "--filter", "reference=swarm_eval_model_*"],
            capture_output=True, text=True,
        )
        if result_images.returncode == 0 and result_images.stdout:
            for img in result_images.stdout.strip().split("\n"):
                if img:
                    subprocess.run(["docker", "rmi", img], capture_output=True, timeout=15)

        subprocess.run(["docker", "image", "prune", "-f"], capture_output=True)
        subprocess.run(["docker", "volume", "prune", "-f"], capture_output=True)
        subprocess.run(
            ["docker", "builder", "prune", "-f", "--keep-storage", "5GB"],
            capture_output=True,
        )

    except Exception as e:
        bt.logging.warning(f"Container cleanup failed: {e}")
