import asyncio
import os
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

import bittensor as bt
try:
    import psutil
except Exception:  # pragma: no cover - psutil should exist in validator env.
    psutil = None

from swarm.config import DockerBatchTimeoutSettings, RpcTraceSettings
from swarm.constants import GLOBAL_EVAL_BASE_SEC, GLOBAL_EVAL_CAP_SEC, GLOBAL_EVAL_PER_SEED_SEC
from swarm.core.model_verify import add_to_blacklist
from swarm.protocol import ValidationResult
from swarm.utils.hash import sha256sum

from ._shared import _docker_evaluator_facade, _submission_template_dir

_MEMORY_SAMPLE_INTERVAL_SEC = 0.25


def _new_memory_metric() -> dict[str, object]:
    return {
        "count": 0,
        "total_bytes": 0.0,
        "peak_bytes": 0,
        "min_bytes": None,
        "first_bytes": None,
        "last_bytes": 0,
    }


def _observe_memory_metric(metric: dict[str, object], value_bytes: int) -> None:
    if value_bytes <= 0:
        return
    count = int(metric.get("count", 0)) + 1
    total_bytes = float(metric.get("total_bytes", 0.0)) + float(value_bytes)
    peak_bytes = max(int(metric.get("peak_bytes", 0)), int(value_bytes))
    min_existing = metric.get("min_bytes")
    min_bytes = int(value_bytes) if min_existing is None else min(int(min_existing), int(value_bytes))
    first_existing = metric.get("first_bytes")
    first_bytes = int(value_bytes) if first_existing is None else int(first_existing)
    metric["count"] = count
    metric["total_bytes"] = total_bytes
    metric["peak_bytes"] = peak_bytes
    metric["min_bytes"] = min_bytes
    metric["first_bytes"] = first_bytes
    metric["last_bytes"] = int(value_bytes)


def _finalize_memory_metric(metric: dict[str, object]) -> dict[str, float | int]:
    count = int(metric.get("count", 0))
    if count <= 0:
        return {
            "sample_count": 0,
            "avg_bytes": 0.0,
            "peak_bytes": 0,
            "min_bytes": 0,
            "first_bytes": 0,
            "last_bytes": 0,
            "peak_delta_from_first_bytes": 0,
            "min_delta_from_first_bytes": 0,
        }
    total_bytes = float(metric.get("total_bytes", 0.0))
    peak_bytes = int(metric.get("peak_bytes", 0))
    min_bytes = int(metric.get("min_bytes") or 0)
    first_bytes = int(metric.get("first_bytes") or 0)
    last_bytes = int(metric.get("last_bytes") or 0)
    return {
        "sample_count": count,
        "avg_bytes": total_bytes / count,
        "peak_bytes": peak_bytes,
        "min_bytes": min_bytes,
        "first_bytes": first_bytes,
        "last_bytes": last_bytes,
        "peak_delta_from_first_bytes": max(0, peak_bytes - first_bytes),
        "min_delta_from_first_bytes": max(0, first_bytes - min_bytes),
    }


def _new_seed_memory_stats() -> dict[str, object]:
    return {
        "validator_process_rss": _new_memory_metric(),
        "validator_process_vms": _new_memory_metric(),
        "docker_container_memory": _new_memory_metric(),
        "system_available_memory": _new_memory_metric(),
        "system_used_memory": _new_memory_metric(),
        "sample_count": 0,
        "first_ts": None,
        "last_ts": None,
        "phase_counts": {},
        "system_total_bytes": 0,
        "docker_container_memory_limit_bytes": 0,
        "host_memory_cap_bytes": 0,
    }


def _read_process_memory_bytes(pid: int) -> tuple[int, int]:
    try:
        if psutil is not None:
            memory_info = psutil.Process(pid).memory_info()
            return int(memory_info.rss), int(memory_info.vms)
    except Exception:
        pass
    rss_bytes = 0
    vms_bytes = 0
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss_bytes = int(parts[1]) * 1024
                elif line.startswith("VmSize:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        vms_bytes = int(parts[1]) * 1024
    except Exception:
        pass
    return rss_bytes, vms_bytes


def _read_system_memory_bytes() -> tuple[int, int, int]:
    try:
        if psutil is not None:
            vm = psutil.virtual_memory()
            return int(vm.available), int(vm.used), int(vm.total)
    except Exception:
        pass
    mem_available = 0
    mem_total = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        mem_available = int(parts[1]) * 1024
                elif line.startswith("MemTotal:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        mem_total = int(parts[1]) * 1024
    except Exception:
        pass
    mem_used = max(0, mem_total - mem_available)
    return mem_available, mem_used, mem_total


def _read_cgroup_memory_bytes_for_pid(pid: int) -> int:
    if pid <= 0:
        return 0
    try:
        with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except Exception:
        return 0

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel_path = parts
        if controllers == "":
            candidate = Path("/sys/fs/cgroup") / rel_path.lstrip("/") / "memory.current"
            try:
                if candidate.is_file():
                    return int(candidate.read_text(encoding="utf-8").strip())
            except Exception:
                pass

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel_path = parts
        controller_names = [name.strip() for name in controllers.split(",") if name.strip()]
        if "memory" not in controller_names:
            continue
        candidates = [
            Path("/sys/fs/cgroup/memory") / rel_path.lstrip("/") / "memory.usage_in_bytes",
            Path("/sys/fs/cgroup") / rel_path.lstrip("/") / "memory.usage_in_bytes",
        ]
        for candidate in candidates:
            try:
                if candidate.is_file():
                    return int(candidate.read_text(encoding="utf-8").strip())
            except Exception:
                pass
    return 0


def _read_cgroup_memory_limit_bytes_for_pid(pid: int) -> int:
    if pid <= 0:
        return 0
    try:
        with open(f"/proc/{pid}/cgroup", "r", encoding="utf-8") as handle:
            lines = [line.strip() for line in handle if line.strip()]
    except Exception:
        return 0

    def _read_limit_file(candidate: Path) -> int:
        try:
            if not candidate.is_file():
                return 0
            raw = candidate.read_text(encoding="utf-8").strip()
            if raw in ("", "max"):
                return 0
            value = int(raw)
            return max(0, value)
        except Exception:
            return 0

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel_path = parts
        if controllers == "":
            value = _read_limit_file(
                Path("/sys/fs/cgroup") / rel_path.lstrip("/") / "memory.max"
            )
            if value > 0:
                return value

    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, rel_path = parts
        controller_names = [name.strip() for name in controllers.split(",") if name.strip()]
        if "memory" not in controller_names:
            continue
        for candidate in (
            Path("/sys/fs/cgroup/memory") / rel_path.lstrip("/") / "memory.limit_in_bytes",
            Path("/sys/fs/cgroup") / rel_path.lstrip("/") / "memory.limit_in_bytes",
        ):
            value = _read_limit_file(candidate)
            if value > 0:
                return value
    return 0


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


async def evaluate_seeds_batch(
    self,
    tasks: list,
    uid: int,
    model_path: Path,
    worker_id: int = 0,
    on_seed_complete: Optional[Callable[..., None]] = None,
    on_batch_complete: Optional[Callable[..., None]] = None,
    rollout_observer: Optional[Callable[[dict], None]] = None,
    task_offset: int = 0,
    task_total: Optional[int] = None,
    model_image: Optional[str] = None,
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

    trace_rpc = RpcTraceSettings.from_env().enabled
    stop_event = threading.Event()
    batch_wall_start = time.perf_counter()
    batch_phase_sec = {
        "workspace_prepare_total_sec": 0.0,
        "container_start_sec": 0.0,
        "pip_install_wait_sec": 0.0,
        "template_restage_sec": 0.0,
        "container_pid_lookup_sec": 0.0,
        "network_lockdown_sec": 0.0,
        "submission_launch_sec": 0.0,
        "rpc_ready_wait_sec": 0.0,
        "rpc_batch_wait_sec": 0.0,
        "container_cleanup_sec": 0.0,
        "tmpdir_cleanup_wait_sec": 0.0,
    }
    workspace_detail_sec = {
        "tmpdir_create_sec": 0.0,
        "submission_extract_sec": 0.0,
        "submission_flatten_sec": 0.0,
        "template_copy_sec": 0.0,
        "permission_fixup_sec": 0.0,
        "requirements_validate_sec": 0.0,
    }
    batch_status = "batch_exception"
    batch_error = ""
    batch_results_count = 0
    batch_has_requirements = False
    progress_state: dict[str, object] = {
        "uid": uid,
        "worker_id": worker_id,
        "phase": "init",
        "task": "n/a",
        "step_idx": 0,
        "sim_t": 0.0,
        "map_seed": -1,
        "challenge_type": -1,
        "seed_active": False,
        "ts": time.time(),
    }
    completed_lock = threading.Lock()
    completed_count = 0
    container_pid = 0
    container_memory_limit_bytes = 0
    host_memory_cap_bytes = 0
    memory_lock = threading.Lock()
    seed_memory_stats: dict[tuple[int, int], dict[str, object]] = {}
    try:
        raw_host_memory_mb = os.getenv("SWARM_HOST_WORKER_MEMORY_MB")
        if raw_host_memory_mb not in (None, ""):
            parsed_memory_mb = int(raw_host_memory_mb)
            if parsed_memory_mb > 0:
                host_memory_cap_bytes = parsed_memory_mb * 1024 * 1024
    except Exception:
        host_memory_cap_bytes = 0

    def _seed_key_from_meta(meta: Optional[dict]) -> Optional[tuple[int, int]]:
        if not isinstance(meta, dict):
            return None
        try:
            map_seed = int(meta.get("map_seed", -1))
            challenge_type = int(meta.get("challenge_type", -1))
        except Exception:
            return None
        if map_seed < 0 or challenge_type < 0:
            return None
        return (map_seed, challenge_type)

    def _observe_seed_memory(
        seed_key: tuple[int, int],
        *,
        phase: str,
        validator_rss_bytes: int,
        validator_vms_bytes: int,
        container_mem_bytes: int,
        system_available_bytes: int,
        system_used_bytes: int,
        system_total_bytes: int,
    ) -> None:
        now_ts = time.time()
        with memory_lock:
            stats = seed_memory_stats.setdefault(seed_key, _new_seed_memory_stats())
            if stats.get("first_ts") is None:
                stats["first_ts"] = now_ts
            if int(stats.get("system_total_bytes", 0)) <= 0 and system_total_bytes > 0:
                stats["system_total_bytes"] = int(system_total_bytes)
            if (
                int(stats.get("docker_container_memory_limit_bytes", 0)) <= 0
                and container_memory_limit_bytes > 0
            ):
                stats["docker_container_memory_limit_bytes"] = int(container_memory_limit_bytes)
            if int(stats.get("host_memory_cap_bytes", 0)) <= 0 and host_memory_cap_bytes > 0:
                stats["host_memory_cap_bytes"] = int(host_memory_cap_bytes)
            stats["last_ts"] = now_ts
            stats["sample_count"] = int(stats.get("sample_count", 0)) + 1
            phase_counts = stats.setdefault("phase_counts", {})
            phase_counts[str(phase)] = int(phase_counts.get(str(phase), 0)) + 1
            _observe_memory_metric(
                stats.setdefault("validator_process_rss", _new_memory_metric()),
                int(validator_rss_bytes),
            )
            _observe_memory_metric(
                stats.setdefault("validator_process_vms", _new_memory_metric()),
                int(validator_vms_bytes),
            )
            _observe_memory_metric(
                stats.setdefault("docker_container_memory", _new_memory_metric()),
                int(container_mem_bytes),
            )
            _observe_memory_metric(
                stats.setdefault("system_available_memory", _new_memory_metric()),
                int(system_available_bytes),
            )
            _observe_memory_metric(
                stats.setdefault("system_used_memory", _new_memory_metric()),
                int(system_used_bytes),
            )

    def _sample_seed_memory(seed_key: Optional[tuple[int, int]] = None) -> None:
        target_key = seed_key
        phase = "unknown"
        if target_key is None:
            try:
                active = bool(progress_state.get("seed_active", False))
                map_seed = int(progress_state.get("map_seed", -1))
                challenge_type = int(progress_state.get("challenge_type", -1))
                if active and map_seed >= 0 and challenge_type >= 0:
                    target_key = (map_seed, challenge_type)
                phase = str(progress_state.get("phase", "unknown"))
            except Exception:
                target_key = None
        if target_key is None:
            return
        validator_rss_bytes, validator_vms_bytes = _read_process_memory_bytes(os.getpid())
        container_mem_bytes = _read_cgroup_memory_bytes_for_pid(container_pid)
        system_available_bytes, system_used_bytes, system_total_bytes = _read_system_memory_bytes()
        _observe_seed_memory(
            target_key,
            phase=phase,
            validator_rss_bytes=validator_rss_bytes,
            validator_vms_bytes=validator_vms_bytes,
            container_mem_bytes=container_mem_bytes,
            system_available_bytes=system_available_bytes,
            system_used_bytes=system_used_bytes,
            system_total_bytes=system_total_bytes,
        )

    def _snapshot_seed_memory(seed_key: Optional[tuple[int, int]]) -> dict[str, object]:
        if seed_key is None:
            return {}
        with memory_lock:
            stats = dict(seed_memory_stats.get(seed_key, {}) or {})
        if not stats:
            return {}
        first_ts = stats.get("first_ts")
        last_ts = stats.get("last_ts")
        return {
            "sample_interval_sec": _MEMORY_SAMPLE_INTERVAL_SEC,
            "sample_count": int(stats.get("sample_count", 0)),
            "observed_span_sec": (
                max(0.0, float(last_ts) - float(first_ts))
                if first_ts is not None and last_ts is not None
                else 0.0
            ),
            "phase_counts": dict(stats.get("phase_counts", {}) or {}),
            "system_total_bytes": int(stats.get("system_total_bytes", 0) or 0),
            "docker_container_memory_limit_bytes": int(
                stats.get("docker_container_memory_limit_bytes", 0) or 0
            ),
            "host_memory_cap_bytes": int(stats.get("host_memory_cap_bytes", 0) or 0),
            "validator_process_rss": _finalize_memory_metric(
                dict(stats.get("validator_process_rss", {}) or {})
            ),
            "validator_process_vms": _finalize_memory_metric(
                dict(stats.get("validator_process_vms", {}) or {})
            ),
            "docker_container_memory": _finalize_memory_metric(
                dict(stats.get("docker_container_memory", {}) or {})
            ),
            "system_available_memory": _finalize_memory_metric(
                dict(stats.get("system_available_memory", {}) or {})
            ),
            "system_used_memory": _finalize_memory_metric(
                dict(stats.get("system_used_memory", {}) or {})
            ),
        }

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
        payload = dict(seed_meta) if isinstance(seed_meta, dict) else None
        seed_key = _seed_key_from_meta(payload)
        if seed_key is not None:
            try:
                _sample_seed_memory(seed_key)
            except Exception:
                pass
            memory_breakdown = _snapshot_seed_memory(seed_key)
            if payload is not None and memory_breakdown:
                payload["memory_breakdown"] = memory_breakdown
        with completed_lock:
            if completed_count >= len(tasks):
                return
            completed_count += 1
        try:
            on_seed_complete(payload)
        except TypeError:
            try:
                on_seed_complete()
            except Exception:
                pass
        except Exception:
            pass

    def _emit_batch_complete() -> None:
        if on_batch_complete is None:
            return
        workspace_known_sec = float(sum(workspace_detail_sec.values()))
        workspace_total_sec = max(
            workspace_known_sec,
            float(batch_phase_sec.get("workspace_prepare_total_sec", 0.0)),
        )
        batch_phase_sec["workspace_prepare_total_sec"] = workspace_total_sec
        batch_total_sec = max(0.0, time.perf_counter() - batch_wall_start)
        top_level_known_sec = (
            workspace_total_sec
            + float(batch_phase_sec.get("container_start_sec", 0.0))
            + float(batch_phase_sec.get("pip_install_wait_sec", 0.0))
            + float(batch_phase_sec.get("template_restage_sec", 0.0))
            + float(batch_phase_sec.get("container_pid_lookup_sec", 0.0))
            + float(batch_phase_sec.get("network_lockdown_sec", 0.0))
            + float(batch_phase_sec.get("submission_launch_sec", 0.0))
            + float(batch_phase_sec.get("rpc_ready_wait_sec", 0.0))
            + float(batch_phase_sec.get("rpc_batch_wait_sec", 0.0))
            + float(batch_phase_sec.get("container_cleanup_sec", 0.0))
            + float(batch_phase_sec.get("tmpdir_cleanup_wait_sec", 0.0))
        )
        payload = {
            "uid": int(uid),
            "worker_id": int(worker_id),
            "seed_count": int(len(tasks)),
            "result_count": int(batch_results_count),
            "status": str(batch_status),
            "error": str(batch_error),
            "has_requirements": bool(batch_has_requirements),
            "used_model_image": bool(model_image),
            "timing_breakdown": {
                "batch_total_sec": batch_total_sec,
                **{key: float(value) for key, value in batch_phase_sec.items()},
                "batch_unaccounted_sec": max(0.0, batch_total_sec - top_level_known_sec),
                "workspace_detail_sec": {
                    **{key: float(value) for key, value in workspace_detail_sec.items()},
                    "workspace_prepare_other_sec": max(
                        0.0, workspace_total_sec - workspace_known_sec
                    ),
                },
            },
        }
        try:
            on_batch_complete(payload)
        except TypeError:
            try:
                on_batch_complete()
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

    if not model_path.is_file():
        bt.logging.warning(f"[Worker {worker_id}] Model path missing: {model_path}")
        _notify_all_failed(status="model_path_missing")
        batch_status = "model_path_missing"
        batch_results_count = len(tasks)
        _emit_batch_complete()
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    if not _docker_evaluator_facade().DockerSecureEvaluator._base_ready:
        bt.logging.warning(f"[Worker {worker_id}] Docker not ready for UID {uid}")
        _notify_all_failed(status="docker_not_ready")
        batch_status = "docker_not_ready"
        batch_results_count = len(tasks)
        _emit_batch_complete()
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    try:
        with zipfile.ZipFile(model_path, "r") as zf:
            if "drone_agent.py" not in zf.namelist():
                bt.logging.warning(
                    f"[Worker {worker_id}] Model {uid} missing drone_agent.py"
                )
                _notify_all_failed(status="submission_missing_drone_agent")
                batch_status = "submission_missing_drone_agent"
                batch_results_count = len(tasks)
                _emit_batch_complete()
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
    except Exception as e:
        bt.logging.warning(
            f"[Worker {worker_id}] Failed to validate model {uid}: {e}"
        )
        _notify_all_failed(
            status="submission_validation_failed",
            error=f"{type(e).__name__}: {e}",
        )
        batch_status = "submission_validation_failed"
        batch_error = f"{type(e).__name__}: {e}"
        batch_results_count = len(tasks)
        _emit_batch_complete()
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    container_name = f"swarm_eval_{uid}_w{worker_id}_{int(time.time() * 1000)}"
    host_port = self._find_free_port()

    _phase(
        f"prepare container={container_name} host_port={host_port} seeds={len(tasks)}"
    )

    tmpdir = None
    has_requirements = False
    try:
        current_uid = os.getuid()
        current_gid = os.getgid()
        worker_limits = self._resolve_worker_limits(worker_id)
        docker_envs = self._docker_env_overrides()

        workspace_start = time.perf_counter()

        tmpdir_start = time.perf_counter()
        tmpdir = tempfile.mkdtemp()
        os.chown(tmpdir, current_uid, current_gid)
        os.chmod(tmpdir, 0o755)

        submission_dir = Path(tmpdir) / "submission"
        submission_dir.mkdir(exist_ok=True)
        os.chown(submission_dir, current_uid, current_gid)
        os.chmod(submission_dir, 0o755)
        workspace_detail_sec["tmpdir_create_sec"] += max(
            0.0, time.perf_counter() - tmpdir_start
        )

        template_dir = _submission_template_dir()

        extract_start = time.perf_counter()
        with zipfile.ZipFile(model_path, "r") as zf:
            zf.extractall(submission_dir)
        workspace_detail_sec["submission_extract_sec"] += max(
            0.0, time.perf_counter() - extract_start
        )

        flatten_start = time.perf_counter()
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
        workspace_detail_sec["submission_flatten_sec"] += max(
            0.0, time.perf_counter() - flatten_start
        )

        template_copy_start = time.perf_counter()
        shutil.copy(template_dir / "agent.capnp", submission_dir)
        shutil.copy(template_dir / "agent_server.py", submission_dir)
        shutil.copy(template_dir / "main.py", submission_dir)
        workspace_detail_sec["template_copy_sec"] += max(
            0.0, time.perf_counter() - template_copy_start
        )

        permissions_start = time.perf_counter()
        for f in submission_dir.iterdir():
            os.chown(f, current_uid, current_gid)
            os.chmod(f, 0o644)
        workspace_detail_sec["permission_fixup_sec"] += max(
            0.0, time.perf_counter() - permissions_start
        )

        miner_requirements = submission_dir / "requirements.txt"
        has_requirements = miner_requirements.exists() and model_image is None
        batch_has_requirements = bool(has_requirements)
        validate_start = time.perf_counter()

        if has_requirements and not self._validate_requirements(
            miner_requirements, uid
        ):
            workspace_detail_sec["requirements_validate_sec"] += max(
                0.0, time.perf_counter() - validate_start
            )
            batch_phase_sec["workspace_prepare_total_sec"] = max(
                0.0, time.perf_counter() - workspace_start
            )
            bt.logging.warning(
                f"[Worker {worker_id}] UID {uid} requirements.txt rejected"
            )
            _notify_all_failed(status="requirements_rejected")
            batch_status = "requirements_rejected"
            batch_results_count = len(tasks)
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
        workspace_detail_sec["requirements_validate_sec"] += max(
            0.0, time.perf_counter() - validate_start
        )
        batch_phase_sec["workspace_prepare_total_sec"] = max(
            0.0, time.perf_counter() - workspace_start
        )

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

        container_start_begin = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        batch_phase_sec["container_start_sec"] += max(
            0.0, time.perf_counter() - container_start_begin
        )

        if result.returncode != 0:
            bt.logging.warning(
                f"[Worker {worker_id}] Container start failed: {result.stderr[:300]}"
            )
            _notify_all_failed(
                status="container_start_failed",
                error=result.stderr[:300],
            )
            batch_status = "container_start_failed"
            batch_error = result.stderr[:300]
            batch_results_count = len(tasks)
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
        _phase("container started successfully")

        pip_timeout = 120 if has_requirements else 0
        rpc_timeout = 30
        connected = False
        pip_done = False

        if has_requirements:
            bt.logging.info(
                f"[Worker {worker_id}] Waiting for pip install (max {pip_timeout}s)..."
            )
            _phase(f"waiting pip install (timeout={pip_timeout}s)")
            pip_wait_start = time.perf_counter()
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

            batch_phase_sec["pip_install_wait_sec"] += max(
                0.0, time.perf_counter() - pip_wait_start
            )

            if not pip_done:
                bt.logging.warning(
                    f"[Worker {worker_id}] pip install failed for UID {uid}"
                )
                _phase("pip install failed")
                _notify_all_failed(status="pip_install_failed")
                batch_status = "pip_install_failed"
                batch_results_count = len(tasks)
                return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
        else:
            pip_done = True
            _phase("no pip install required")

        if has_requirements:
            template_restage_start = time.perf_counter()
            shutil.copy(template_dir / "agent.capnp", submission_dir)
            shutil.copy(template_dir / "agent_server.py", submission_dir)
            shutil.copy(template_dir / "main.py", submission_dir)
            batch_phase_sec["template_restage_sec"] += max(
                0.0, time.perf_counter() - template_restage_start
            )

        container_pid_start = time.perf_counter()
        container_pid = self._get_container_pid(container_name)
        batch_phase_sec["container_pid_lookup_sec"] += max(
            0.0, time.perf_counter() - container_pid_start
        )
        if not container_pid:
            bt.logging.warning(f"[Worker {worker_id}] Failed to get container PID")
            _phase("failed to resolve container pid")
            _notify_all_failed(status="container_pid_missing")
            batch_status = "container_pid_missing"
            batch_results_count = len(tasks)
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
        container_memory_limit_bytes = _read_cgroup_memory_limit_bytes_for_pid(container_pid)

        _phase(
            f"applying network lockdown pid={container_pid} validator_ip={validator_ip}"
        )
        network_lockdown_start = time.perf_counter()
        network_ok = self._apply_network_lockdown(container_pid, validator_ip)
        batch_phase_sec["network_lockdown_sec"] += max(
            0.0, time.perf_counter() - network_lockdown_start
        )
        if not network_ok:
            bt.logging.warning(f"[Worker {worker_id}] Network lockdown failed")
            _phase("network lockdown failed")
            _notify_all_failed(status="network_lockdown_failed")
            batch_status = "network_lockdown_failed"
            batch_results_count = len(tasks)
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
        _phase("network lockdown applied")

        submission_launch_start = time.perf_counter()
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
        batch_phase_sec["submission_launch_sec"] += max(
            0.0, time.perf_counter() - submission_launch_start
        )
        if exec_result.returncode != 0:
            bt.logging.warning(
                f"[Worker {worker_id}] Failed to start main.py: {exec_result.stderr[:200]}"
            )
            _phase("failed to launch submission main.py")
            _notify_all_failed(
                status="submission_start_failed",
                error=exec_result.stderr[:200],
            )
            batch_status = "submission_start_failed"
            batch_error = exec_result.stderr[:200]
            batch_results_count = len(tasks)
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]
        _phase("submission main.py launched")

        rpc_ready_wait_start = time.perf_counter()
        rpc_start = time.time()
        max_rpc_wait = 30
        rpc_check_interval = 2
        rpc_check_count = 0
        connected = False
        _phase(f"waiting for rpc readiness (max_wait={max_rpc_wait}s)")

        await asyncio.sleep(4)

        while time.time() - rpc_start < max_rpc_wait:
            rpc_check_count += 1
            try:
                if self._check_rpc_ready(container_name, timeout=5.0):
                    connected = True
                    elapsed = time.time() - rpc_start
                    _phase(f"rpc ready after {elapsed:.1f}s")
                    break
            except Exception:
                pass
            if trace_rpc and rpc_check_count % 3 == 0:
                waited = time.time() - rpc_start
                _phase(f"rpc not ready yet ({waited:.1f}s elapsed)")
            await asyncio.sleep(rpc_check_interval)
        batch_phase_sec["rpc_ready_wait_sec"] += max(
            0.0, time.perf_counter() - rpc_ready_wait_start
        )

        if not connected:
            bt.logging.warning(f"[Worker {worker_id}] RPC connection failed")
            _phase("rpc readiness failed")
            _notify_all_failed(status="rpc_connection_failed")
            batch_status = "rpc_connection_failed"
            batch_results_count = len(tasks)
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
            _notify_all_failed(status="container_stopped_before_eval")
            batch_status = "container_stopped_before_eval"
            batch_results_count = len(tasks)
            return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

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
            memory_sampler_stop = threading.Event()

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
                    )
                except Exception as e:
                    rpc_payload["error"] = e
                finally:
                    rpc_done.set()

            def _memory_sampler_worker() -> None:
                while not memory_sampler_stop.wait(timeout=_MEMORY_SAMPLE_INTERVAL_SEC):
                    try:
                        _sample_seed_memory()
                    except Exception:
                        pass

            rpc_thread = threading.Thread(
                target=_rpc_worker,
                name=f"rpc_eval_uid{uid}_w{worker_id}",
                daemon=True,
            )
            memory_sampler_thread = threading.Thread(
                target=_memory_sampler_worker,
                name=f"rpc_mem_uid{uid}_w{worker_id}",
                daemon=True,
            )
            rpc_thread.start()
            memory_sampler_thread.start()

            timed_out = False
            rpc_batch_wait_start = time.perf_counter()
            eval_start = time.time()
            timeout_deadline = eval_start + batch_timeout
            extension_count = 0
            last_extended_sim_t = -1.0
            last_extended_step_idx = -1
            try:
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
            finally:
                memory_sampler_stop.set()
                memory_sampler_thread.join(timeout=1.0)
            batch_phase_sec["rpc_batch_wait_sec"] += max(
                0.0, time.perf_counter() - rpc_batch_wait_start
            )

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
                    batch_status = "batch_timeout_partial"
                    batch_error = (
                        f"batch timed out after {timeout_limit_elapsed:.1f}s and returned partial results"
                    )
                    batch_results_count = len(partial_results)
                    return partial_results
                _notify_all_failed(status="batch_timeout")
                batch_status = "batch_timeout"
                batch_error = f"batch timed out after {timeout_limit_elapsed:.1f}s"
                batch_results_count = len(tasks)
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
            batch_status = "batch_done"
            batch_results_count = len(valid_results)
            return valid_results
        finally:
            stop_event.set()
            cleanup_start = time.perf_counter()
            _run_docker_cmd_quiet(["docker", "kill", container_name])
            _run_docker_cmd_quiet(["docker", "rm", "-f", container_name])
            batch_phase_sec["container_cleanup_sec"] += max(
                0.0, time.perf_counter() - cleanup_start
            )
            _phase("container cleaned up")

    except Exception as e:
        bt.logging.warning(f"[Worker {worker_id}] Batch evaluation failed: {e}")
        _phase(f"batch evaluation exception: {type(e).__name__}: {e}")
        _notify_all_failed(
            status="batch_exception",
            error=f"{type(e).__name__}: {e}",
        )
        batch_status = "batch_exception"
        batch_error = f"{type(e).__name__}: {e}"
        batch_results_count = len(tasks)
    finally:
        tmpdir_cleanup_start = time.perf_counter()
        _cleanup_tmpdir_quiet(tmpdir)
        batch_phase_sec["tmpdir_cleanup_wait_sec"] += max(
            0.0, time.perf_counter() - tmpdir_cleanup_start
        )
        _emit_batch_complete()

    return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

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
