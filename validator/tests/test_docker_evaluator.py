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

"""Unit tests for the Docker sandbox evaluator: cache keys, worker limits, RPC transport and the process scheduler."""

from __future__ import annotations

import asyncio
import queue
import re
import socket
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.benchmark import engine as bench_full_eval
from swarm.challenge_families.base import ChallengeFamilyRuntimeProfile
from swarm.protocol import FailureReason, ValidationResult
from swarm.validator.calibration import SpeedFactor
from swarm.validator.docker import docker_evaluator as de
from swarm.validator.docker.docker_evaluator_parts import lifecycle
from swarm.validator.runtime_telemetry import ValidatorRuntimeTracker


class _ProcResult:
    """Stand-in for a completed subprocess.run result."""
    def __init__(self, returncode=0, stdout="", stderr=""):
        """Record the exit code and the streams a faked docker call hands back."""
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _new_evaluator() -> de.DockerSecureEvaluator:
    """Return a DockerSecureEvaluator built without __init__, with its base image marked ready."""
    ev = de.DockerSecureEvaluator.__new__(de.DockerSecureEvaluator)
    ev.base_image = "swarm_evaluator_base:latest"
    ev.base_images = {"base": ev.base_image}
    ev.base_ready = True
    de.DockerSecureEvaluator._base_ready = True
    return ev


def _eligible_speed(factor: float = 1.0) -> SpeedFactor:
    """Return a SpeedFactor that passes the eligibility check at the given ratio."""
    return SpeedFactor(
        raw=factor,
        factor=factor,
        eligible=True,
        owner_p90_ms=100.0,
        local_p90_ms=100.0 * factor,
    )


class _ScriptedQueue:
    """In-process stand-in for a multiprocessing queue, drained by a handler or buffered."""
    def __init__(self):
        """Start with nothing buffered and no handler attached."""
        self._items = []
        self._handler = None

    def set_handler(self, handler):
        """Route later puts straight to handler instead of buffering them."""
        self._handler = handler

    def put(self, item):
        """Hand the item to the handler when one is set, otherwise append it to the buffer."""
        if self._handler is not None:
            self._handler(item)
            return
        self._items.append(item)

    def get(self, timeout=None):
        """Pop the oldest buffered item, raising Empty when nothing is waiting; timeout is ignored."""
        _ = timeout
        if self._items:
            return self._items.pop(0)
        raise de.parallel.queue_mod.Empty

    def get_nowait(self):
        """Pop the oldest buffered item, raising Empty at once when the buffer is dry."""
        return self.get(timeout=0.0)

    def close(self):
        """Do nothing; the scripted queue holds no resources."""
        return None


class _ScriptedProcess:
    """Stand-in for a worker process that answers each request inline on the caller's thread."""
    def __init__(self, ctx, args):
        """Record the scripted context and the worker argument tuple, with a fixed fake pid."""
        self._ctx = ctx
        self._args = args
        self.exitcode = None
        self._alive = False
        self.pid = 4242

    def start(self):
        """Mark the worker alive and wire the task queue so every request is answered inline."""
        self._alive = True
        worker_slot, task_queue, result_queue, progress_queue = self._args
        task_queue.set_handler(
            lambda request: self._ctx.handle_request(
                worker_slot,
                request,
                result_queue,
                progress_queue,
            )
        )

    def join(self, timeout=None):
        """Return at once; the scripted worker has no thread to wait on."""
        _ = timeout
        return None

    def terminate(self):
        """Mark the worker dead and record the SIGTERM exit code."""
        self._alive = False
        self.exitcode = -15

    def is_alive(self):
        """Return True while the scripted worker has been started and not yet terminated."""
        return self._alive


class _ScriptedContext:
    """Multiprocessing context stand-in that replays a canned outcome for each batch attempt."""
    def __init__(self, bench_engine, scripted_attempts):
        """Hold the benchmark engine module and the per-batch attempt plans, with nothing run yet."""
        self._bench_engine = bench_engine
        self._scripted_attempts = scripted_attempts
        self._attempts = {}
        self._queues = []
        self.requests = []

    def Queue(self):
        """Return a fresh scripted queue and keep a reference to it."""
        queue_obj = _ScriptedQueue()
        self._queues.append(queue_obj)
        return queue_obj

    def Process(self, target, args, name, daemon):
        """Return a scripted process bound to this context; target, name and daemon are ignored."""
        _ = target, name, daemon
        return _ScriptedProcess(self, args)

    def handle_request(self, worker_slot, request, result_queue, progress_queue):
        """Replay the plan for this batch attempt: push its seed events, then its batch result."""
        if request is None:
            return
        self.requests.append(request)
        batch_index = int(request.batch_index)
        attempt = self._attempts.get(batch_index, 0)
        self._attempts[batch_index] = attempt + 1
        plan = self._scripted_attempts[batch_index][attempt]
        for seed_meta in plan.get("seed_events", []):
            progress_queue.put(
                self._bench_engine._ProcessSeedEvent(
                    worker_id=int(worker_slot),
                    batch_index=batch_index,
                    seed_meta=seed_meta,
                )
            )
        result_queue.put(
            self._bench_engine._ProcessBatchResult(
                worker_id=int(worker_slot),
                batch_index=batch_index,
                batch_indices=list(request.batch_indices),
                results=list(plan.get("results", [])),
                elapsed_sec=float(plan.get("elapsed_sec", 0.1)),
                error=plan.get("error"),
            )
        )

    @property
    def attempts(self):
        """Return a copy of the run count recorded against each batch index."""
        return dict(self._attempts)


def test_serialize_observation_dict():
    """A dict observation becomes one entry per key, carrying each array's shape and dtype."""
    class _ObsMessage:
        """Capnp Observation message stand-in whose init allocates plain namespace entries."""
        def init(self, field, n):
            """Allocate n blank entries on the message and return them."""
            assert field == "entries"
            self.entries = [
                SimpleNamespace(
                    key="",
                    tensor=SimpleNamespace(data=b"", shape=[], dtype=""),
                )
                for _ in range(n)
            ]
            return self.entries

    class _Observation:
        """Capnp schema stand-in that hands out a fresh fake message."""
        @staticmethod
        def new_message():
            """Return a fresh fake Observation message."""
            return _ObsMessage()

    schema = SimpleNamespace(Observation=_Observation)
    msg = de.DockerSecureEvaluator._serialize_observation(
        schema,
        {"state": np.array([1.0, 2.0], dtype=np.float32)},
    )
    assert len(msg.entries) == 1
    assert msg.entries[0].key == "state"
    assert msg.entries[0].tensor.shape == [2]
    assert msg.entries[0].tensor.dtype == "float32"


def test_serialize_observation_array_sets_value_key():
    """A bare array observation lands under the reserved __value__ key with its own shape."""
    class _ObsMessage:
        """Capnp Observation message stand-in whose init allocates plain namespace entries."""
        def init(self, field, n):
            """Allocate n blank entries on the message and return them."""
            assert field == "entries"
            self.entries = [
                SimpleNamespace(
                    key="",
                    tensor=SimpleNamespace(data=b"", shape=[], dtype=""),
                )
                for _ in range(n)
            ]
            return self.entries

    class _Observation:
        """Capnp schema stand-in that hands out a fresh fake message."""
        @staticmethod
        def new_message():
            """Return a fresh fake Observation message."""
            return _ObsMessage()

    schema = SimpleNamespace(Observation=_Observation)
    msg = de.DockerSecureEvaluator._serialize_observation(
        schema, np.array([5, 6], dtype=np.float32)
    )
    assert msg.entries[0].key == "__value__"
    assert msg.entries[0].tensor.shape == [2]


def test_serialize_observation_zero_tensor_roundtrips_compact():
    """An all-zero array ships no bytes and rebuilds exactly; negative zero still ships its bytes."""
    capnp = pytest.importorskip("capnp")
    from swarm.validator.docker.docker_evaluator_parts._shared import (
        _submission_template_dir,
    )

    schema = capnp.load(str(_submission_template_dir() / "agent.capnp"))
    obs = {
        "depth": np.random.rand(4, 4, 1).astype(np.float32),
        "rgb": np.zeros((4, 4, 3), dtype=np.float32),
        "state": np.array([1.0, 0.0, 2.5], dtype=np.float32),
        "neg_zero": np.array([-0.0, 0.0], dtype=np.float32),
    }
    msg = de.DockerSecureEvaluator._serialize_observation(schema, obs)

    by_key = {e.key: e for e in msg.entries}
    assert len(by_key["rgb"].tensor.data) == 0
    assert len(by_key["depth"].tensor.data) == obs["depth"].nbytes
    assert len(by_key["state"].tensor.data) == obs["state"].nbytes
    assert by_key["neg_zero"].tensor.data == obs["neg_zero"].tobytes()

    with schema.Observation.from_bytes(msg.to_bytes()) as parsed:
        for entry in parsed.entries:
            shape = tuple(entry.tensor.shape)
            dtype = np.dtype(entry.tensor.dtype)
            if len(entry.tensor.data) == 0:
                rebuilt = np.zeros(shape, dtype=dtype)
            else:
                rebuilt = np.frombuffer(entry.tensor.data, dtype=dtype).reshape(shape)
            assert rebuilt.dtype == obs[entry.key].dtype
            assert rebuilt.shape == obs[entry.key].shape
            assert rebuilt.tobytes() == obs[entry.key].tobytes()


def test_serialize_observation_shm_roundtrips_via_buffer():
    """Tensors travel in the shared buffer under a __shm__ manifest and rebuild byte for byte; a buffer too small raises BufferError."""
    import json as _json

    capnp = pytest.importorskip("capnp")
    from swarm.validator.docker.docker_evaluator_parts._shared import (
        _submission_template_dir,
    )
    from swarm.validator.docker.docker_evaluator_parts.submission import (
        _serialize_observation_shm,
    )

    schema = capnp.load(str(_submission_template_dir() / "agent.capnp"))
    obs = {
        "depth": np.random.rand(8, 8, 1).astype(np.float32),
        "rgb": np.zeros((8, 8, 3), dtype=np.float32),
        "state": np.array([1.0, -2.0, 0.5], dtype=np.float32),
    }
    shm_buf = bytearray(4096)
    msg = _serialize_observation_shm(schema, obs, shm_buf)

    with schema.Observation.from_bytes(msg.to_bytes()) as parsed:
        manifest = {}
        tensor_entries = []
        for entry in parsed.entries:
            if entry.key == "__shm__":
                for key, off, nbytes in _json.loads(bytes(entry.tensor.data).decode()):
                    manifest[key] = (off, nbytes)
            else:
                assert len(entry.tensor.data) == 0
                tensor_entries.append(entry)

        assert set(manifest) == {"depth", "state"}
        for entry in tensor_entries:
            shape = tuple(entry.tensor.shape)
            dtype = np.dtype(entry.tensor.dtype)
            if entry.key in manifest:
                off, nbytes = manifest[entry.key]
                rebuilt = np.frombuffer(
                    bytes(shm_buf[off:off + nbytes]), dtype=dtype
                ).reshape(shape)
            else:
                rebuilt = np.zeros(shape, dtype=dtype)
            assert rebuilt.tobytes() == obs[entry.key].tobytes()

    tiny = bytearray(8)
    with pytest.raises(BufferError):
        _serialize_observation_shm(schema, obs, tiny)


def test_check_docker_available_true(monkeypatch):
    """A docker version call that exits zero reports the daemon as usable."""
    ev = _new_evaluator()
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda *a, **k: _ProcResult(returncode=0, stdout="Docker version 26"),
    )
    assert ev._check_docker_available() is True


def test_check_docker_available_false_on_missing_binary(monkeypatch):
    """A docker binary absent from PATH reads as unavailable rather than raising to the caller."""
    ev = _new_evaluator()

    def _raise(*args, **kwargs):
        """Raise FileNotFoundError the way an absent docker binary would."""
        _ = args, kwargs
        raise FileNotFoundError("docker")

    monkeypatch.setattr(de.subprocess, "run", _raise)
    assert ev._check_docker_available() is False


def test_cleanup_env_quietly_closes_env():
    """A passed environment is closed exactly once."""
    calls = {"count": 0}

    class _Env:
        """Environment stand-in that counts how often it is closed."""
        def close(self):
            """Count one close call."""
            calls["count"] += 1

    de._cleanup_env_quietly(_Env())
    assert calls["count"] == 1


def test_submission_template_dir_points_to_swarm_template():
    """The template directory resolves inside the installed swarm package and holds agent.capnp."""
    template_dir = de._submission_template_dir()
    assert template_dir == (Path(__file__).resolve().parents[2] / "swarm" / "submission_template")
    assert (template_dir / "agent.capnp").is_file()


def test_get_image_hash_label(monkeypatch):
    """The stored hash comes back from docker inspect with the trailing newline stripped."""
    ev = _new_evaluator()
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda *a, **k: _ProcResult(returncode=0, stdout="abc123\n"),
    )
    assert ev._get_image_hash_label() == "abc123"


def test_should_rebuild_base_image_when_image_missing(monkeypatch):
    """No image id from docker images means a rebuild, whatever the source hash says."""
    ev = _new_evaluator()
    monkeypatch.setattr(ev, "_calculate_docker_hash", lambda: "hash1")
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda cmd, **k: _ProcResult(returncode=0, stdout=""),
    )
    assert ev._should_rebuild_base_image() is True


def test_should_rebuild_base_image_false_when_hash_matches(monkeypatch):
    """An existing image whose label equals the source hash is reused, not rebuilt."""
    ev = _new_evaluator()
    monkeypatch.setattr(ev, "_calculate_docker_hash", lambda: "hash1")
    monkeypatch.setattr(ev, "_get_image_hash_label", lambda: "hash1")
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda cmd, **k: _ProcResult(returncode=0, stdout="imageid\n"),
    )
    assert ev._should_rebuild_base_image() is False


def test_should_rebuild_base_image_true_when_hash_differs(monkeypatch):
    """An image labelled with a stale source hash is rebuilt."""
    ev = _new_evaluator()
    monkeypatch.setattr(ev, "_calculate_docker_hash", lambda: "hash-new")
    monkeypatch.setattr(ev, "_get_image_hash_label", lambda: "hash-old")
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda cmd, **k: _ProcResult(returncode=0, stdout="imageid\n"),
    )
    assert ev._should_rebuild_base_image() is True


def test_setup_base_container_uses_real_docker_paths_after_split(monkeypatch):
    """Setup issues exactly one docker build, with -f aimed at the repository's own Dockerfile."""
    ev = _new_evaluator()
    build_cmds = []

    monkeypatch.setattr(ev, "_check_docker_available", lambda: True)
    monkeypatch.setattr(ev, "_should_rebuild_base_image", lambda: True)
    monkeypatch.setattr(ev, "_calculate_docker_hash", lambda: "hash1")

    def _run(cmd, **kwargs):
        """Record every docker build invocation and report success for any command."""
        _ = kwargs
        if isinstance(cmd, list) and cmd[:2] == ["docker", "build"]:
            build_cmds.append(cmd)
        return _ProcResult(returncode=0, stdout="")

    monkeypatch.setattr(de.subprocess, "run", _run)
    de.DockerSecureEvaluator._base_ready = False

    ev._setup_base_container()

    assert ev.base_ready is True
    assert len(build_cmds) == 1
    dockerfile_path = Path(__file__).resolve().parents[2] / "swarm" / "validator" / "docker" / "Dockerfile"
    assert build_cmds[0][build_cmds[0].index("-f") + 1] == str(dockerfile_path)


def test_check_rpc_ready_open_port():
    """A socket that is listening is reported ready."""
    ev = _new_evaluator()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]
        assert ev._check_rpc_ready(port, timeout=0.5) is True
    finally:
        server.close()


def test_check_rpc_ready_closed_port():
    """A port with nothing bound to it is reported not ready before the timeout elapses."""
    ev = _new_evaluator()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.close()
    assert ev._check_rpc_ready(port, timeout=0.2) is False


def test_check_rpc_ready_late_bind():
    """A bound socket that is not yet listening reads as not ready, and as ready once it listens."""
    ev = _new_evaluator()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    assert ev._check_rpc_ready(port, timeout=0.1) is False
    server.listen(1)
    try:
        assert ev._check_rpc_ready(port, timeout=0.5) is True
    finally:
        server.close()


def test_get_docker_host_ip_and_fallback(monkeypatch):
    """The bridge address comes from docker inspect, falling back to 172.17.0.1 when it fails."""
    ev = _new_evaluator()
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda *a, **k: _ProcResult(returncode=0, stdout="172.18.0.1\n"),
    )
    assert ev._get_docker_host_ip() == "172.18.0.1"

    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda *a, **k: _ProcResult(returncode=1, stdout=""),
    )
    assert ev._get_docker_host_ip() == "172.17.0.1"


def test_get_container_pid(monkeypatch):
    """A reported pid of 0 comes back as None, any other value as an int."""
    ev = _new_evaluator()
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda *a, **k: _ProcResult(returncode=0, stdout="123\n"),
    )
    assert ev._get_container_pid("c1") == 123

    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda *a, **k: _ProcResult(returncode=0, stdout="0\n"),
    )
    assert ev._get_container_pid("c1") is None


def test_apply_network_lockdown_success(monkeypatch):
    """A full lockdown lays down six rules: four on IPv4 and two on IPv6."""
    ev = _new_evaluator()
    calls = {"count": 0}

    def _run(*args, **kwargs):
        """Count every firewall call and report success."""
        _ = args, kwargs
        calls["count"] += 1
        return _ProcResult(returncode=0)

    monkeypatch.setattr(de.subprocess, "run", _run)
    assert ev._apply_network_lockdown(9999, "10.0.0.1") is True
    assert calls["count"] == 6  # 4 IPv4 iptables rules + 2 IPv6 ip6tables rules


def test_apply_network_lockdown_failure_when_rule_fails(monkeypatch):
    """One rejected IPv4 rule aborts the lockdown and reports failure."""
    ev = _new_evaluator()
    calls = {"count": 0}

    def _run(*args, **kwargs):
        """Fail the second firewall call and let every other one succeed."""
        _ = args, kwargs
        calls["count"] += 1
        if calls["count"] == 2:
            return _ProcResult(returncode=1, stderr="iptables failed")
        return _ProcResult(returncode=0)

    monkeypatch.setattr(de.subprocess, "run", _run)
    assert ev._apply_network_lockdown(9999, "10.0.0.1") is False


def test_apply_network_lockdown_fails_closed_on_ipv6_rule_failure(monkeypatch):
    """A rejected DROP rule on a host that does have IPv6 leaves the lockdown failed."""
    ev = _new_evaluator()
    calls = {"count": 0}

    def _run(*args, **kwargs):
        """Fail the last ip6tables call and let every other one succeed."""
        _ = args, kwargs
        calls["count"] += 1
        if calls["count"] == 6:  # last IPv6 ip6tables DROP rule
            return _ProcResult(returncode=1, stderr="ip6tables failed")
        return _ProcResult(returncode=0)

    monkeypatch.setattr(de.subprocess, "run", _run)
    assert ev._apply_network_lockdown(9999, "10.0.0.1") is False


def test_apply_network_lockdown_tolerates_ipv6_disabled_host(monkeypatch):
    """A host with IPv6 switched off skips the remaining v6 rules and still locks down."""
    ev = _new_evaluator()
    calls = {"count": 0}

    def _run(*args, **kwargs):
        """Fail the first ip6tables call the way an ipv6.disable=1 host does."""
        _ = args, kwargs
        calls["count"] += 1
        if calls["count"] == 5:  # first IPv6 rule on an ipv6.disable=1 host
            return _ProcResult(
                returncode=1,
                stderr="ip6tables: can't initialize ip6tables table 'filter': "
                       "Address family not supported by protocol",
            )
        return _ProcResult(returncode=0)

    monkeypatch.setattr(de.subprocess, "run", _run)
    assert ev._apply_network_lockdown(9999, "10.0.0.1") is True
    assert calls["count"] == 5  # remaining IPv6 rules skipped


def test_docker_env_overrides_enable_thread_caps_by_default(monkeypatch):
    """With nothing set in the environment, every thread variable is pinned to the cap and interop to one."""
    monkeypatch.delenv("SWARM_DOCKER_THREAD_CAPS", raising=False)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    envs = de.DockerSecureEvaluator._docker_env_overrides(thread_cap=2)
    assert envs["OMP_NUM_THREADS"] == "2"
    assert envs["SWARM_INFERENCE_THREADS"] == "2"
    assert envs["SWARM_ORT_INTRA_OP_THREADS"] == "2"
    assert envs["SWARM_TORCH_NUM_THREADS"] == "2"
    assert envs["SWARM_TORCH_THREADS"] == "2"
    assert envs["SWARM_TORCH_INTEROP_THREADS"] == "1"


def test_docker_env_overrides_clamp_operator_values_to_worker_cap(monkeypatch):
    """A larger thread setting left by the operator is clamped down, never honoured as given."""
    monkeypatch.setenv("OMP_NUM_THREADS", "64")
    monkeypatch.setenv("SWARM_TORCH_THREADS", "32")
    monkeypatch.delenv("SWARM_TORCH_NUM_THREADS", raising=False)

    envs = de.DockerSecureEvaluator._docker_env_overrides(thread_cap=2)

    assert envs["OMP_NUM_THREADS"] == "2"
    assert envs["SWARM_TORCH_NUM_THREADS"] == "2"
    assert envs["SWARM_TORCH_THREADS"] == "2"


def test_worker_thread_cap_uses_narrowest_cpu_limit():
    """The cap is the smaller of the cpuset width and the quota, never their sum."""
    from swarm.config import DockerRuntimeSettings

    assert DockerRuntimeSettings.worker_thread_cap(
        {"cpus": None, "cpuset_cpus": "4-5"}
    ) == 2
    assert DockerRuntimeSettings.worker_thread_cap(
        {"cpus": "1.5", "cpuset_cpus": "4-7"}
    ) == 2
    assert DockerRuntimeSettings.worker_thread_cap(
        {"cpus": "1", "cpuset_cpus": "4-5"}
    ) == 1


def test_resolve_worker_limits_uses_env_overrides(monkeypatch):
    """Environment variables supply the cpu quota, the memory ceiling and this worker's cpuset."""
    monkeypatch.setenv("SWARM_DOCKER_WORKER_CPUS_OVERRIDE", "1.0")
    monkeypatch.setenv("SWARM_DOCKER_WORKER_MEMORY_OVERRIDE", "4g")
    monkeypatch.setenv("SWARM_DOCKER_WORKER_CPUSETS", "0-1;2-3")
    limits = de.DockerSecureEvaluator._resolve_worker_limits(worker_id=1)
    assert limits["cpus"] == "1.0"
    assert limits["memory"] == "4g"
    assert limits["cpuset_cpus"] == "2-3"


def test_resolve_worker_limits_accepts_runtime_profile_overrides(monkeypatch):
    """A challenge family's profile sets cpu and memory when the environment says nothing."""
    monkeypatch.delenv("SWARM_DOCKER_WORKER_CPUS_OVERRIDE", raising=False)
    monkeypatch.delenv("SWARM_DOCKER_WORKER_MEMORY_OVERRIDE", raising=False)
    profile = ChallengeFamilyRuntimeProfile(
        family_id="cf_search_and_rescue",
        docker_worker_cpus="1.5",
        docker_worker_memory="5g",
    )

    limits = de.DockerSecureEvaluator._resolve_worker_limits(
        worker_id=0,
        runtime_profile=profile,
    )

    assert limits["cpus"] == "1.5"
    assert limits["memory"] == "5g"


def test_resolve_base_image_for_key_uses_mapping_and_env_override(monkeypatch):
    """A key resolves through the built image map, and an environment variable beats that map."""
    ev = _new_evaluator()
    ev.base_images["mission"] = "swarm_evaluator_mission:latest"

    assert ev._resolve_base_image_for_key("mission") == "swarm_evaluator_mission:latest"

    monkeypatch.setenv("SWARM_DOCKER_BASE_IMAGE_MISSION", "custom-mission-image:dev")
    assert ev._resolve_base_image_for_key("mission") == "custom-mission-image:dev"


def test_auto_worker_cpuset_map_partitions_host():
    """Workers get disjoint, contiguous cpu lists handed out from cpu 0; spare cpus stay unassigned."""
    from swarm.config import auto_worker_cpuset_map

    assert auto_worker_cpuset_map(n_workers=6, cpus_per_worker=2, total_cpus=12) == (
        "0,1;2,3;4,5;6,7;8,9;10,11"
    )
    assert auto_worker_cpuset_map(n_workers=4, cpus_per_worker=2, total_cpus=8) == (
        "0,1;2,3;4,5;6,7"
    )
    assert auto_worker_cpuset_map(n_workers=2, cpus_per_worker=2, total_cpus=4) == (
        "0,1;2,3"
    )
    assert auto_worker_cpuset_map(n_workers=1, cpus_per_worker=4, total_cpus=8) == "0,1,2,3"


def test_auto_worker_cpuset_map_returns_none_when_partition_oversized():
    """No pinning is proposed when the workers would need more cpus than the machine has."""
    from swarm.config import auto_worker_cpuset_map

    assert auto_worker_cpuset_map(n_workers=6, cpus_per_worker=2, total_cpus=8) is None
    assert auto_worker_cpuset_map(n_workers=1, cpus_per_worker=2, total_cpus=1) is None
    assert auto_worker_cpuset_map(n_workers=0, cpus_per_worker=2, total_cpus=8) is None


def test_resolve_worker_limits_auto_pins_when_env_unset(monkeypatch):
    """Each worker gets its own slice of the machine's cpus when nothing is configured."""
    monkeypatch.delenv("SWARM_DOCKER_WORKER_CPUSETS", raising=False)
    for i in range(16):
        monkeypatch.delenv(f"SWARM_DOCKER_WORKER_CPUSET_CPUS_{i}", raising=False)
    monkeypatch.setattr("swarm.config.runtime.N_DOCKER_WORKERS", 4)
    monkeypatch.setattr("swarm.config.runtime.cpus_per_docker_worker", lambda: 2)
    monkeypatch.setattr("swarm.config.runtime.available_vcpu_count", lambda: 8)

    assert de.DockerSecureEvaluator._resolve_worker_limits(worker_id=0)["cpuset_cpus"] == "0,1"
    assert de.DockerSecureEvaluator._resolve_worker_limits(worker_id=1)["cpuset_cpus"] == "2,3"
    assert de.DockerSecureEvaluator._resolve_worker_limits(worker_id=3)["cpuset_cpus"] == "6,7"


def test_resolve_worker_limits_skips_auto_pin_on_undersized_host(monkeypatch):
    """No cpuset is chosen at all when the machine has fewer cpus than the workers need."""
    monkeypatch.delenv("SWARM_DOCKER_WORKER_CPUSETS", raising=False)
    for i in range(16):
        monkeypatch.delenv(f"SWARM_DOCKER_WORKER_CPUSET_CPUS_{i}", raising=False)
    monkeypatch.setattr("swarm.config.runtime.N_DOCKER_WORKERS", 6)
    monkeypatch.setattr("swarm.config.runtime.cpus_per_docker_worker", lambda: 2)
    monkeypatch.setattr("swarm.config.runtime.available_vcpu_count", lambda: 4)

    assert de.DockerSecureEvaluator._resolve_worker_limits(worker_id=0)["cpuset_cpus"] is None


def test_resolve_worker_limits_env_var_overrides_auto(monkeypatch):
    """A configured cpuset list wins over the partition computed from the machine."""
    monkeypatch.setenv("SWARM_DOCKER_WORKER_CPUSETS", "10-11;12-13")
    monkeypatch.setattr("swarm.config.runtime.N_DOCKER_WORKERS", 4)
    monkeypatch.setattr("swarm.config.runtime.cpus_per_docker_worker", lambda: 2)
    monkeypatch.setattr("swarm.config.runtime.available_vcpu_count", lambda: 16)

    assert de.DockerSecureEvaluator._resolve_worker_limits(worker_id=0)["cpuset_cpus"] == "10-11"
    assert de.DockerSecureEvaluator._resolve_worker_limits(worker_id=1)["cpuset_cpus"] == "12-13"


def test_resolve_worker_limits_per_worker_env_overrides_auto(monkeypatch):
    """A single worker's own variable wins for that slot and leaves the rest computed."""
    monkeypatch.delenv("SWARM_DOCKER_WORKER_CPUSETS", raising=False)
    monkeypatch.setenv("SWARM_DOCKER_WORKER_CPUSET_CPUS_2", "20-21")
    monkeypatch.setattr("swarm.config.runtime.N_DOCKER_WORKERS", 4)
    monkeypatch.setattr("swarm.config.runtime.cpus_per_docker_worker", lambda: 2)
    monkeypatch.setattr("swarm.config.runtime.available_vcpu_count", lambda: 8)

    assert de.DockerSecureEvaluator._resolve_worker_limits(worker_id=2)["cpuset_cpus"] == "20-21"
    assert de.DockerSecureEvaluator._resolve_worker_limits(worker_id=0)["cpuset_cpus"] == "0,1"


def test_host_worker_runtime_settings_use_env_overrides(monkeypatch):
    """The memory ceiling and the per-slot cpuset are both read from the environment."""
    from swarm.config import HostWorkerRuntimeSettings

    monkeypatch.setenv("SWARM_HOST_WORKER_MEMORY_MB", "2048")
    monkeypatch.setenv("SWARM_HOST_WORKER_CPUSETS", "0;1;2-3")
    limits = HostWorkerRuntimeSettings.from_env().resolve_worker_limits(worker_id=2)
    assert limits.memory_mb == 2048
    assert limits.cpuset_cpus == "2-3"


def test_host_worker_runtime_parse_cpuset_spec():
    """A cpuset string expands its ranges into the full set of individual cpu indices."""
    from swarm.config import HostWorkerRuntimeSettings

    parsed = HostWorkerRuntimeSettings.parse_cpuset_spec("0,2-4,7")
    assert parsed == {0, 2, 3, 4, 7}


def test_run_multi_seed_rpc_sync_uses_initial_obs_without_extra_reset(monkeypatch):
    """The observation the env was built with feeds calibration and the first act, so no second reset is issued."""
    rpc_mod = de.rpc
    ev = _new_evaluator()
    monkeypatch.setattr(rpc_mod.capnp, "load", lambda _path: SimpleNamespace(Agent=object()))
    monkeypatch.setattr(rpc_mod.RpcTraceSettings, "from_env", lambda: SimpleNamespace(enabled=False, trace_every=1, heartbeat_sec=0.0))

    class _Ping:
        """Ping reply carrying the expected pong string."""
        response = "pong"

    class _Cal:
        """Calibration reply reporting a twelve millisecond benchmark."""
        benchmarkNs = 12_000_000

    class _ActionTensor:
        """Zero-filled five element float32 action payload."""
        data = np.zeros(5, dtype=np.float32).tobytes()
        dtype = "float32"
        shape = [5]

    class _ActionResp:
        """Act reply wrapping the fixed action payload."""
        action = _ActionTensor()

    class _Agent:
        """Capnp agent stand-in recording every reset, calibrate and act call."""
        def __init__(self):
            """Start with no resets recorded and empty calibrate and act logs."""
            self.reset_calls = 0
            self.calibrate_obs = []
            self.act_obs = []

        async def ping(self, _msg):
            """Return the pong reply."""
            return _Ping()

        async def reset(self):
            """Count the call and hand nothing back to the caller."""
            self.reset_calls += 1
            return None

        async def calibrate(self, obs):
            """Log the observation and return the canned calibration reply."""
            self.calibrate_obs.append(obs)
            return _Cal()

        async def act(self, obs):
            """Log the observation and return the fixed action reply."""
            self.act_obs.append(obs)
            return _ActionResp()

    agent = _Agent()

    class _Bootstrap:
        """Capnp bootstrap stand-in that casts to the single fake agent."""
        def cast_as(self, _schema):
            """Return the fake agent whatever schema is asked for."""
            return agent

    class _Client:
        """Two-party client stand-in handing out the fake bootstrap capability."""
        def bootstrap(self):
            """Return the stand-in bootstrap capability."""
            return _Bootstrap()

    monkeypatch.setattr(rpc_mod.capnp, "TwoPartyClient", lambda _stream: _Client())

    class _Loop:
        """Async context manager standing in for the capnp kj event loop."""
        async def __aenter__(self):
            """Enter the fake event loop and yield nothing."""
            return None

        async def __aexit__(self, exc_type, exc, tb):
            """Leave the fake event loop without swallowing an exception."""
            return False

    monkeypatch.setattr(rpc_mod.capnp, "kj_loop", lambda: _Loop())

    class _StreamFactory:
        """Stream factory stand-in whose connections are bare objects."""
        @staticmethod
        async def create_connection(**_kwargs):
            """Return a bare object in place of a socket stream."""
            return object()

    monkeypatch.setattr(rpc_mod.capnp, "AsyncIoStream", _StreamFactory)
    monkeypatch.setattr(ev, "_serialize_observation", lambda _schema, obs: ("serialized", obs["marker"]))
    monkeypatch.setattr(rpc_mod, "CALIBRATION_RECAL_INTERVAL", 1)
    calibration_obs = []

    async def _fake_calibrate(_agent, _schema, obs, uid):
        """Record the uid and observation marker, and report a ten millisecond overhead."""
        calibration_obs.append((uid, obs["marker"]))
        return 0.01, 1.0

    monkeypatch.setattr(ev, "_calibrate_rpc_overhead_async", _fake_calibrate)

    closed = []

    class _Env:
        """Drone environment stand-in that ends the episode on its first step."""
        def __init__(self):
            """Build a five dimensional action space and start the step counter at zero."""
            self.action_space = SimpleNamespace(
                low=np.full(5, -1.0, dtype=np.float32),
                high=np.full(5, 1.0, dtype=np.float32),
                shape=(5,),
            )
            self.ACT_TYPE = None
            self.SPEED_LIMIT = None
            self.step_calls = 0

        def step(self, _action):
            """Count the call and return a terminated transition that reached the goal."""
            self.step_calls += 1
            return {"marker": "next"}, 0.0, True, False, {
                "success": True,
                "min_clearance": 1.0,
                "collision": False,
            }

        def close(self):
            """Record that the environment was shut down."""
            closed.append(True)

    env = _Env()
    make_calls = []

    def _fake_make_env_with_initial_obs(task, gui=False):
        """Record the seed and gui flag, and return the scripted env with a marker observation."""
        make_calls.append((task.map_seed, gui))
        return env, {"marker": "initial"}

    monkeypatch.setattr(rpc_mod, "make_env_with_initial_obs", _fake_make_env_with_initial_obs)

    task = SimpleNamespace(
        map_seed=77,
        challenge_type=1,
        horizon=0.04,
        start=(0.0, 0.0, 1.0),
        goal=(1.0, 1.0, 1.0),
    )
    results = ev._run_multi_seed_rpc_sync([task], uid=9, rpc_port=8000)

    assert len(results) == 1
    assert results[0].success is True
    assert make_calls == [(77, False)]
    assert agent.reset_calls == 1
    assert calibration_obs == [(9, "initial")]
    assert agent.act_obs == [("serialized", "initial")]
    assert env.step_calls == 1
    assert closed == [True]


def test_evaluate_seeds_parallel_uses_process_scheduler(monkeypatch, tmp_path):
    """Every seed goes to the process scheduler as its own batch, tagged by map group and carrying the family profile and host speed factor."""
    ev = _new_evaluator()
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    tasks = [
        SimpleNamespace(challenge_type=3, map_seed=1001, seed_id=0, family_id="cf_search_and_rescue"),
        SimpleNamespace(challenge_type=2, map_seed=1002, seed_id=1, family_id="cf_search_and_rescue"),
        SimpleNamespace(challenge_type=6, map_seed=1003, seed_id=2, family_id="cf_search_and_rescue"),
    ]
    callback_payloads = []
    captured = {}

    async def _fake_run_process_parallel(**kwargs):
        """Capture the scheduler arguments, fire a seed callback per task, and return one result each."""
        captured["batch_plan"] = kwargs["batch_plan"]
        captured["effective_workers"] = kwargs["effective_workers"]
        captured["task_meta"] = kwargs["task_meta"]
        captured["runtime_profile"] = kwargs["runtime_profile"]
        captured["host_speed_factor"] = kwargs["host_speed_factor"]
        for task in kwargs["all_tasks"]:
            kwargs["on_seed_complete"](
                {
                    "map_seed": int(task.map_seed),
                    "challenge_type": int(task.challenge_type),
                    "status": "seed_done",
                }
            )
        return [
            ValidationResult(kwargs["uid"], True, float(task.seed_id), 0.5)
            for task in kwargs["all_tasks"]
        ]

    monkeypatch.setattr(de.parallel, "_run_process_parallel", _fake_run_process_parallel)
    monkeypatch.setattr(
        de.batch, "_ensure_host_speed_factor",
        lambda _self, _worker_count: asyncio.sleep(0, result=_eligible_speed(1.25)),
    )
    results = asyncio.run(
        ev.evaluate_seeds_parallel(
            tasks,
            uid=11,
            model_path=model_path,
            num_workers=3,
            on_seed_complete=lambda payload=None: callback_payloads.append(payload),
        )
    )
    assert len(results) == 3
    assert [r.time_sec for r in results] == [0.0, 1.0, 2.0]
    assert captured["batch_plan"] == [[0], [1], [2]]
    assert captured["effective_workers"] == 3
    assert [meta["group"] for meta in captured["task_meta"]] == [
        "type3_mountain",
        "type2_open",
        "type6_forest",
    ]
    assert captured["runtime_profile"]["family_id"] == "cf_search_and_rescue"
    assert captured["runtime_profile"]["profile_name"] == "search_and_rescue"
    assert captured["host_speed_factor"] == pytest.approx(1.25)
    assert [payload["status"] for payload in callback_payloads] == [
        "seed_done",
        "seed_done",
        "seed_done",
    ]


def test_evaluate_seeds_parallel_uses_default_worker_count_of_three(monkeypatch, tmp_path):
    """With no worker count given, the scheduler gets the configured maximum capped by the task count."""
    ev = _new_evaluator()
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    tasks = [
        SimpleNamespace(challenge_type=1, map_seed=2001, family_id="cf_autopilot"),
        SimpleNamespace(challenge_type=2, map_seed=2002, family_id="cf_autopilot"),
        SimpleNamespace(challenge_type=3, map_seed=2003, family_id="cf_autopilot"),
        SimpleNamespace(challenge_type=4, map_seed=2004, family_id="cf_autopilot"),
        SimpleNamespace(challenge_type=5, map_seed=2005, family_id="cf_autopilot"),
    ]
    captured = {}

    async def _fake_run_process_parallel(**kwargs):
        """Capture the worker count, family profile and speed factor, and pass every task."""
        captured["effective_workers"] = kwargs["effective_workers"]
        captured["runtime_profile"] = kwargs["runtime_profile"]
        captured["host_speed_factor"] = kwargs["host_speed_factor"]
        return [ValidationResult(kwargs["uid"], True, 1.0, 0.5) for _ in kwargs["all_tasks"]]

    monkeypatch.setattr(de.parallel, "_run_process_parallel", _fake_run_process_parallel)
    monkeypatch.setattr(
        de.batch, "_ensure_host_speed_factor",
        lambda _self, _worker_count: asyncio.sleep(0, result=_eligible_speed(1.1)),
    )
    results = asyncio.run(ev.evaluate_seeds_parallel(tasks, uid=17, model_path=model_path))

    assert len(results) == 5
    assert captured["effective_workers"] == min(len(tasks), de.parallel.N_DOCKER_WORKERS)
    assert captured["runtime_profile"]["family_id"] == "cf_autopilot"
    assert captured["runtime_profile"]["profile_name"] == "autopilot_navigation"
    assert captured["host_speed_factor"] == pytest.approx(1.1)


def test_evaluate_seeds_parallel_updates_runtime_tracker(monkeypatch, tmp_path):
    """The tracker records both the worker count asked for and the smaller number actually used."""
    ev = _new_evaluator()
    ev.runtime_tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    tasks = [
        SimpleNamespace(challenge_type=2, map_seed=2101),
        SimpleNamespace(challenge_type=3, map_seed=2102),
    ]
    captured = {}

    async def _fake_run_process_parallel(**kwargs):
        """Capture the runtime tracker and speed factor, and pass every task."""
        captured["runtime_tracker"] = kwargs["runtime_tracker"]
        captured["host_speed_factor"] = kwargs["host_speed_factor"]
        return [ValidationResult(kwargs["uid"], True, 1.0, 0.5) for _ in kwargs["all_tasks"]]

    monkeypatch.setattr(de.parallel, "_run_process_parallel", _fake_run_process_parallel)
    monkeypatch.setattr(
        de.batch, "_ensure_host_speed_factor",
        lambda _self, _worker_count: asyncio.sleep(0, result=_eligible_speed(0.9)),
    )
    results = asyncio.run(
        ev.evaluate_seeds_parallel(tasks, uid=19, model_path=model_path, num_workers=4)
    )

    snapshot = ev.runtime_tracker.snapshot_copy()
    assert len(results) == 2
    assert captured["runtime_tracker"] is ev.runtime_tracker
    assert captured["host_speed_factor"] == pytest.approx(0.9)
    assert snapshot["docker"]["requested_workers"] == 4
    assert snapshot["docker"]["effective_workers"] == 2


@pytest.mark.full
def test_run_process_parallel_retries_wall_timeout_once(monkeypatch, tmp_path):
    """A seed cancelled on its wall clock is run a second time and the successful attempt counts."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    task = SimpleNamespace(
        challenge_type=4,
        map_seed=3101,
        horizon=60.0,
    )
    scripted_context = _ScriptedContext(
        bench_full_eval,
        {
            0: [
                {
                    "seed_events": [
                        {
                            "uid": 41,
                            "map_seed": 3101,
                            "challenge_type": 4,
                            "status": "seed_cancelled",
                            "success": False,
                            "sim_time_sec": 12.0,
                            "seed_wall_sec": 240.1,
                            "step_idx": 123,
                            "error": "",
                        }
                    ],
                    "results": [(41, False, 12.0, 0.0)],
                    "elapsed_sec": 240.1,
                },
                {
                    "seed_events": [
                        {
                            "uid": 41,
                            "map_seed": 3101,
                            "challenge_type": 4,
                            "status": "seed_done",
                            "success": True,
                            "sim_time_sec": 18.0,
                            "seed_wall_sec": 300.0,
                            "step_idx": 180,
                            "error": "",
                        }
                    ],
                    "results": [(41, True, 18.0, 0.8)],
                    "elapsed_sec": 18.0,
                },
            ]
        },
    )
    callback_payloads = []
    log_lines = []

    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: scripted_context)
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: log_lines.append(str(msg)))
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: log_lines.append(str(msg)))

    results = asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=[task],
            task_meta=[
                {
                    "group": "type4_village",
                    "seed": 3101,
                    "index": 0,
                    "challenge_type": 4,
                    "horizon": 60.0,
                }
            ],
            batch_plan=[[0]],
            uid=41,
            model_path=model_path,
            effective_workers=1,
            on_seed_complete=lambda payload=None: callback_payloads.append(payload),
            phase_label="eval",
        )
    )

    assert scripted_context.attempts == {0: 2}
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].score == pytest.approx(0.8)
    assert [payload["status"] for payload in callback_payloads] == ["seed_done"]
    assert any("retrying timed-out seed village:#0" in line for line in log_lines)
    assert any("1 retried_timeout" in line for line in log_lines)


@pytest.mark.full
def test_run_process_parallel_hands_a_timed_out_seed_back_in_seed_flow(monkeypatch, tmp_path):
    """Under the seed feeder a wall-clock timeout is final here: the pool retries it, not this host."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    task = SimpleNamespace(
        challenge_type=4,
        map_seed=3101,
        horizon=60.0,
    )
    timed_out = {
        "seed_events": [
            {
                "uid": 41,
                "map_seed": 3101,
                "challenge_type": 4,
                "status": "seed_cancelled",
                "success": False,
                "sim_time_sec": 12.0,
                "seed_wall_sec": 840.0,
                "step_idx": 123,
                "error": "",
            }
        ],
        "results": [(41, False, 12.0, 0.0)],
        "elapsed_sec": 840.0,
    }
    scripted_context = _ScriptedContext(bench_full_eval, {0: [timed_out, timed_out]})
    log_lines = []

    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: scripted_context)
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: log_lines.append(str(msg)))
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: log_lines.append(str(msg)))

    async def _drained_feeder(_free_slots):
        """Offer nothing more and report the pool drained."""
        return [], True

    results = asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=[task],
            task_meta=[
                {
                    "group": "type4_village",
                    "seed": 3101,
                    "index": 0,
                    "challenge_type": 4,
                    "horizon": 60.0,
                }
            ],
            batch_plan=[[0]],
            uid=41,
            model_path=model_path,
            effective_workers=1,
            phase_label="eval",
            seed_feeder=_drained_feeder,
            initial_pending=[0],
        )
    )

    assert scripted_context.attempts == {0: 1}, "the seed must not be flown twice on this host"
    assert results[0].failure_reason == FailureReason.INFRA.value
    assert not any("retrying timed-out seed" in line for line in log_lines)


@pytest.mark.full
def test_run_process_parallel_does_not_retry_seed_timeout_strikes(monkeypatch, tmp_path):
    """A seed killed for slow inference is never re-run; the failure stands and counts as slow_act."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    task = SimpleNamespace(
        challenge_type=3,
        map_seed=3201,
        horizon=60.0,
    )
    scripted_context = _ScriptedContext(
        bench_full_eval,
        {
            0: [
                {
                    "seed_events": [
                        {
                            "uid": 51,
                            "map_seed": 3201,
                            "challenge_type": 3,
                            "status": "seed_timeout_strikes",
                            "success": False,
                            "sim_time_sec": 4.0,
                            "seed_wall_sec": 30.0,
                            "step_idx": 15,
                            "error": "",
                        }
                    ],
                    "results": [(51, False, 4.0, 0.0)],
                    "elapsed_sec": 30.0,
                }
            ]
        },
    )
    callback_payloads = []
    log_lines = []

    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: scripted_context)
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: log_lines.append(str(msg)))
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: log_lines.append(str(msg)))

    results = asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=[task],
            task_meta=[
                {
                    "group": "type3_mountain",
                    "seed": 3201,
                    "index": 0,
                    "challenge_type": 3,
                    "horizon": 60.0,
                }
            ],
            batch_plan=[[0]],
            uid=51,
            model_path=model_path,
            effective_workers=1,
            on_seed_complete=lambda payload=None: callback_payloads.append(payload),
            phase_label="eval",
        )
    )

    assert scripted_context.attempts == {0: 1}
    assert len(results) == 1
    assert results[0].success is False
    assert results[0].score == pytest.approx(0.0)
    assert [payload["status"] for payload in callback_payloads] == ["seed_timeout_strikes"]
    assert not any("retrying timed-out seed mountain:#0" in line for line in log_lines)
    assert any(
        "0 failed, 1 slow_act, 0 timeout, 0 runtime, 0 retried_timeout" in line
        for line in log_lines
    )
    assert any("slow_act_failures: mountain:#0" in line for line in log_lines)


def test_is_rpc_transport_status_classifies_transport_failures():
    """Only connect, ping-timeout and disconnect count as transport faults; seed outcomes do not."""
    assert bench_full_eval._is_rpc_transport_status("rpc_connect_failed")
    assert bench_full_eval._is_rpc_transport_status("rpc_ping_timeout")
    assert bench_full_eval._is_rpc_transport_status("seed_rpc_disconnected")
    assert not bench_full_eval._is_rpc_transport_status("seed_timeout_strikes")
    assert not bench_full_eval._is_rpc_transport_status("seed_exception")
    assert not bench_full_eval._is_rpc_transport_status("seed_done")
    assert not bench_full_eval._is_rpc_transport_status("rpc_connection_failed")


@pytest.mark.full
def test_run_process_parallel_retries_rpc_transport_once(monkeypatch, tmp_path):
    """A broken pipe to the sandbox is run again and the second attempt's result is the one kept."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    task = SimpleNamespace(
        challenge_type=4,
        map_seed=3401,
        horizon=60.0,
    )
    scripted_context = _ScriptedContext(
        bench_full_eval,
        {
            0: [
                {
                    "seed_events": [
                        {
                            "uid": 61,
                            "map_seed": 3401,
                            "challenge_type": 4,
                            "status": "seed_rpc_disconnected",
                            "success": False,
                            "sim_time_sec": 3.0,
                            "seed_wall_sec": 5.0,
                            "step_idx": 20,
                            "error": "[Errno 32] Broken pipe",
                        }
                    ],
                    "results": [(61, False, 3.0, 0.0)],
                    "elapsed_sec": 5.0,
                },
                {
                    "seed_events": [
                        {
                            "uid": 61,
                            "map_seed": 3401,
                            "challenge_type": 4,
                            "status": "seed_done",
                            "success": True,
                            "sim_time_sec": 18.0,
                            "seed_wall_sec": 300.0,
                            "step_idx": 180,
                            "error": "",
                        }
                    ],
                    "results": [(61, True, 18.0, 0.8)],
                    "elapsed_sec": 18.0,
                },
            ]
        },
    )
    callback_payloads = []
    log_lines = []

    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: scripted_context)
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: log_lines.append(str(msg)))
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: log_lines.append(str(msg)))

    results = asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=[task],
            task_meta=[
                {
                    "group": "type4_village",
                    "seed": 3401,
                    "index": 0,
                    "challenge_type": 4,
                    "horizon": 60.0,
                }
            ],
            batch_plan=[[0]],
            uid=61,
            model_path=model_path,
            effective_workers=1,
            on_seed_complete=lambda payload=None: callback_payloads.append(payload),
            phase_label="eval",
        )
    )

    assert scripted_context.attempts == {0: 2}
    assert len(results) == 1
    assert results[0].success is True
    assert results[0].score == pytest.approx(0.8)
    assert [payload["status"] for payload in callback_payloads] == ["seed_done"]
    assert any("retrying RPC-transport seed village:#0" in line for line in log_lines)
    assert any("1 retried_rpc_transport" in line for line in log_lines)


@pytest.mark.full
def test_run_process_parallel_caps_rpc_transport_retries_at_one(monkeypatch, tmp_path):
    """A second disconnect is not run again; the seed fails after exactly one repeat."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    task = SimpleNamespace(
        challenge_type=4,
        map_seed=3402,
        horizon=60.0,
    )
    disconnect_attempt = {
        "seed_events": [
            {
                "uid": 62,
                "map_seed": 3402,
                "challenge_type": 4,
                "status": "seed_rpc_disconnected",
                "success": False,
                "sim_time_sec": 3.0,
                "seed_wall_sec": 5.0,
                "step_idx": 20,
                "error": "[Errno 32] Broken pipe",
            }
        ],
        "results": [(62, False, 3.0, 0.0)],
        "elapsed_sec": 5.0,
    }
    scripted_context = _ScriptedContext(
        bench_full_eval,
        {0: [disconnect_attempt, disconnect_attempt]},
    )
    callback_payloads = []
    log_lines = []

    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: scripted_context)
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: log_lines.append(str(msg)))
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: log_lines.append(str(msg)))

    results = asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=[task],
            task_meta=[
                {
                    "group": "type4_village",
                    "seed": 3402,
                    "challenge_type": 4,
                    "horizon": 60.0,
                }
            ],
            batch_plan=[[0]],
            uid=62,
            model_path=model_path,
            effective_workers=1,
            on_seed_complete=lambda payload=None: callback_payloads.append(payload),
            phase_label="eval",
        )
    )

    assert scripted_context.attempts == {0: 2}
    assert len(results) == 1
    assert results[0].success is False
    assert results[0].score == pytest.approx(0.0)
    assert [payload["status"] for payload in callback_payloads] == ["seed_rpc_disconnected"]
    assert len([line for line in log_lines if "retrying RPC-transport seed" in line]) == 1


@pytest.mark.full
def test_run_process_parallel_honors_exhausted_shared_retry_budget(monkeypatch, tmp_path):
    """A budget already spent by an earlier call blocks the repeat and is left untouched."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    task = SimpleNamespace(
        challenge_type=4,
        map_seed=3501,
        horizon=60.0,
    )
    scripted_context = _ScriptedContext(
        bench_full_eval,
        {
            0: [
                {
                    "seed_events": [
                        {
                            "uid": 62,
                            "map_seed": 3501,
                            "challenge_type": 4,
                            "status": "seed_rpc_disconnected",
                            "success": False,
                            "sim_time_sec": 3.0,
                            "seed_wall_sec": 5.0,
                            "step_idx": 20,
                            "error": "[Errno 32] Broken pipe",
                        }
                    ],
                    "results": [(62, False, 3.0, 0.0)],
                    "elapsed_sec": 5.0,
                },
            ]
        },
    )
    log_lines = []

    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: scripted_context)
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: log_lines.append(str(msg)))
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: log_lines.append(str(msg)))

    retry_budget = {
        "timeout": de.parallel._MAX_TIMEOUT_RETRIES,
        "rpc_transport": de.parallel._MAX_RPC_TRANSPORT_RETRIES,
    }
    results = asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=[task],
            task_meta=[
                {
                    "group": "type4_village",
                    "seed": 3501,
                    "challenge_type": 4,
                    "horizon": 60.0,
                }
            ],
            batch_plan=[[0]],
            uid=62,
            model_path=model_path,
            effective_workers=1,
            phase_label="eval",
            retry_budget=retry_budget,
        )
    )

    assert scripted_context.attempts == {0: 1}
    assert len(results) == 1
    assert results[0].success is False
    assert not any("retrying RPC-transport seed" in line for line in log_lines)
    assert retry_budget["rpc_transport"] == de.parallel._MAX_RPC_TRANSPORT_RETRIES


@pytest.mark.full
def test_run_process_parallel_summary_uses_live_scheduler_status(monkeypatch, tmp_path):
    """The closing log prints the scheduler's freshly sampled line, not the cached one."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    task = SimpleNamespace(
        challenge_type=5,
        map_seed=3301,
        horizon=60.0,
    )
    scripted_context = _ScriptedContext(
        bench_full_eval,
        {
            0: [
                {
                    "seed_events": [
                        {
                            "uid": 61,
                            "map_seed": 3301,
                            "challenge_type": 5,
                            "status": "seed_done",
                            "success": True,
                            "sim_time_sec": 12.0,
                            "seed_wall_sec": 40.0,
                            "step_idx": 120,
                            "error": "",
                        }
                    ],
                    "results": [(61, True, 12.0, 0.9)],
                    "elapsed_sec": 40.0,
                }
            ]
        },
    )
    log_lines = []

    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: scripted_context)
    monkeypatch.setattr(
        bench_full_eval._RamWorkerScheduler,
        "format_status_line",
        lambda self: "stale-status",
    )
    monkeypatch.setattr(
        bench_full_eval._RamWorkerScheduler,
        "format_live_status_line",
        lambda self: "live-status",
    )
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: log_lines.append(str(msg)))
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: log_lines.append(str(msg)))

    results = asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=[task],
            task_meta=[
                {
                    "group": "type5_warehouse",
                    "seed": 3301,
                    "challenge_type": 5,
                    "horizon": 60.0,
                }
            ],
            batch_plan=[[0]],
            uid=61,
            model_path=model_path,
            effective_workers=1,
            on_seed_complete=None,
            phase_label="eval",
        )
    )

    assert len(results) == 1
    assert results[0].success is True
    assert any("live-status" in line for line in log_lines)
    assert not any("stale-status" in line for line in log_lines)


@pytest.mark.full
def test_run_process_parallel_refreshes_resources_while_waiting(monkeypatch, tmp_path):
    """Machine load is sampled repeatedly during a busy batch, not once at dispatch."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    refresh_calls = []
    original_refresh_resources = bench_full_eval._RamWorkerScheduler.refresh_resources

    def _counting_refresh_resources(self):
        """Count the call and delegate to the scheduler's own sampling."""
        refresh_calls.append(True)
        return original_refresh_resources(self)

    monkeypatch.setattr(
        bench_full_eval._RamWorkerScheduler,
        "refresh_resources",
        _counting_refresh_resources,
    )
    monkeypatch.setattr(
        bench_full_eval,
        "_RESOURCE_POLL_INTERVAL_SEC",
        0.05,
        raising=False,
    )

    class _DelayedProcess:
        """Worker process stand-in that serves one batch on a thread after a quarter second."""
        def __init__(self, ctx, args):
            """Record the context and worker arguments, with no thread yet and the stop flag clear."""
            self._ctx = ctx
            self._args = args
            self.exitcode = None
            self._alive = False
            self._thread = None
            self._stop = threading.Event()

        def start(self):
            """Spawn the daemon thread that serves a single batch after the delay."""
            self._alive = True
            worker_slot, task_queue, result_queue, progress_queue = self._args

            def _run():
                """Send a batch-started heartbeat, wait out the delay, then post one passing result."""
                request = task_queue.get()
                if request is None:
                    self.exitcode = 0
                    self._alive = False
                    return
                progress_queue.put(
                    self._ctx._bench_engine._ProcessWorkerHeartbeat(
                        worker_id=int(worker_slot),
                        batch_index=int(request.batch_index),
                        event_type="batch_started",
                        ts=time.time(),
                    )
                )
                if self._stop.wait(0.25):
                    self.exitcode = -15
                    self._alive = False
                    return
                result_queue.put(
                    self._ctx._bench_engine._ProcessBatchResult(
                        worker_id=int(worker_slot),
                        batch_index=int(request.batch_index),
                        batch_indices=list(request.batch_indices),
                        results=[(int(request.uid), True, 12.0, 0.9)],
                        elapsed_sec=0.25,
                    )
                )
                self.exitcode = 0
                self._alive = False

            self._thread = threading.Thread(target=_run, daemon=True)
            self._thread.start()

        def join(self, timeout=None):
            """Wait on the serving thread when one was ever started."""
            if self._thread is not None:
                self._thread.join(timeout=timeout)

        def terminate(self):
            """Set the stop flag so the serving thread exits without posting a result."""
            self.exitcode = -15
            self._stop.set()
            self._alive = False

        def is_alive(self):
            """Return True until the serving thread finishes or is stopped."""
            return bool(self._alive)

    class _DelayedCtx:
        """Multiprocessing context stand-in handing out real queues and delayed processes."""
        def __init__(self, bench_engine):
            """Hold the benchmark engine module whose event types the workers post."""
            self._bench_engine = bench_engine

        def Queue(self):
            """Return a plain thread-safe queue."""
            return queue.Queue()

        def Process(self, target, args, name, daemon):
            """Return a delayed process bound to this context; target, name and daemon are ignored."""
            _ = target, name, daemon
            return _DelayedProcess(self, args)

    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(
        bench_full_eval,
        "_benchmark_mp_context",
        lambda: _DelayedCtx(bench_full_eval),
    )
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: None)
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: None)

    task = SimpleNamespace(
        challenge_type=5,
        map_seed=3401,
        horizon=60.0,
    )
    results = asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=[task],
            task_meta=[
                {
                    "group": "type5_warehouse",
                    "seed": 3401,
                    "index": 0,
                    "challenge_type": 5,
                    "horizon": 60.0,
                }
            ],
            batch_plan=[[0]],
            uid=62,
            model_path=model_path,
            effective_workers=1,
            on_seed_complete=None,
            phase_label="eval",
        )
    )

    assert len(results) == 1
    assert results[0].success is True
    assert len(refresh_calls) >= 2


def test_evaluate_seeds_parallel_falls_back_to_batch_when_docker_not_ready(monkeypatch, tmp_path):
    """With no base image built, every seed runs through one sequential call on slot 0, still carrying the family profile and speed factor."""
    ev = _new_evaluator()
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    tasks = [
        SimpleNamespace(challenge_type=3, map_seed=3001, family_id="cf_search_and_rescue"),
        SimpleNamespace(challenge_type=1, map_seed=3002, family_id="cf_search_and_rescue"),
    ]
    captured = {}

    async def _fake_batch(
        chunk,
        uid,
        model_path,
        worker_id=0,
        on_seed_complete=None,
        task_offset=0,
        task_total=None,
        runtime_profile_payload=None,
        host_speed_factor=None,
        model_image=None,
    ):
        """Capture the sequential call's arguments and return one failing result per task."""
        captured["chunk"] = list(chunk)
        captured["worker_id"] = worker_id
        captured["task_offset"] = task_offset
        captured["task_total"] = task_total
        captured["runtime_profile_payload"] = dict(runtime_profile_payload or {})
        captured["host_speed_factor"] = host_speed_factor
        _ = model_path, on_seed_complete
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in chunk]

    monkeypatch.setattr(de.DockerSecureEvaluator, "_base_ready", False)
    monkeypatch.setattr(ev, "evaluate_seeds_batch", _fake_batch)
    monkeypatch.setattr(
        de.batch, "_ensure_host_speed_factor",
        lambda _self, _worker_count: asyncio.sleep(0, result=_eligible_speed(1.4)),
    )
    try:
        results = asyncio.run(
            ev.evaluate_seeds_parallel(tasks, uid=5, model_path=model_path, num_workers=3)
        )
    finally:
        monkeypatch.setattr(de.DockerSecureEvaluator, "_base_ready", True)

    assert len(results) == 2
    assert captured["chunk"] == tasks
    assert captured["worker_id"] == 0
    assert captured["task_offset"] == 0
    assert captured["task_total"] == 2
    assert captured["runtime_profile_payload"]["family_id"] == "cf_search_and_rescue"
    assert captured["runtime_profile_payload"]["profile_name"] == "search_and_rescue"
    assert captured["host_speed_factor"] == pytest.approx(1.4)


def test_evaluate_seeds_batch_returns_failures_when_model_missing(tmp_path):
    """A model file that is not on disk scores every seed zero and reports model_path_missing."""
    ev = _new_evaluator()
    de.DockerSecureEvaluator._base_ready = True
    payloads = []
    tasks = [SimpleNamespace(map_seed=1), SimpleNamespace(map_seed=2)]
    results = asyncio.run(
        ev.evaluate_seeds_batch(
            tasks,
            uid=1,
            model_path=tmp_path / "missing.zip",
            on_seed_complete=lambda payload=None: payloads.append(payload),
        )
    )
    assert len(results) == 2
    assert all(r.score == 0.0 for r in results)
    assert len(payloads) == 2
    assert [payload["status"] for payload in payloads] == ["model_path_missing", "model_path_missing"]


def test_evaluate_seeds_batch_returns_failures_when_docker_not_ready(tmp_path):
    """An unbuilt base image scores the seed zero and reports docker_not_ready."""
    ev = _new_evaluator()
    de.DockerSecureEvaluator._base_ready = False
    model = tmp_path / "model.zip"
    with zipfile.ZipFile(model, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", "{}")
    payloads = []
    tasks = [SimpleNamespace(map_seed=3)]
    results = asyncio.run(
        ev.evaluate_seeds_batch(
            tasks,
            uid=2,
            model_path=model,
            on_seed_complete=lambda payload=None: payloads.append(payload),
        )
    )
    assert len(results) == 1
    assert results[0].score == 0.0
    assert len(payloads) == 1
    assert payloads[0]["status"] == "docker_not_ready"


def test_run_multi_seed_rpc_sync_isolated_payload_transforms_results(monkeypatch):
    """ValidationResults cross the process boundary as plain tuples, field order preserved."""
    sample = [
        ValidationResult(uid=1, success=True, time_sec=2.5, score=0.7),
        ValidationResult(uid=1, success=False, time_sec=1.0, score=0.0),
    ]
    monkeypatch.setattr(
        de.DockerSecureEvaluator, "_run_multi_seed_rpc_sync", lambda *a, **k: sample
    )
    payload = de._run_multi_seed_rpc_sync_isolated_payload(
        tasks=[1, 2], uid=1, rpc_port=9000
    )
    assert payload == [(1, True, 2.5, 0.7), (1, False, 1.0, 0.0)]


def test_constructor_uses_class_state_without_module_global(monkeypatch):
    """A fresh evaluator reads its readiness off the class attribute, which stays False without docker."""
    monkeypatch.setattr(
        de.DockerSecureEvaluator, "_check_docker_available", lambda self: False
    )
    de.DockerSecureEvaluator._instance = None
    de.DockerSecureEvaluator._base_ready = False

    evaluator = de.DockerSecureEvaluator()

    assert evaluator.base_ready is False
    assert de.DockerSecureEvaluator._base_ready is False


def _recycle_scripted_run(monkeypatch, tmp_path, *, seeds, log_lines):
    """Drive the given number of one-seed batches through a scripted context, returning the results and that context."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    tasks = [
        SimpleNamespace(challenge_type=1, map_seed=5200 + i, horizon=60.0)
        for i in range(seeds)
    ]
    scripted_context = _ScriptedContext(
        bench_full_eval,
        {
            i: [{"results": [(51, True, 1.0, 0.5)], "elapsed_sec": 0.1}]
            for i in range(seeds)
        },
    )
    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: scripted_context)
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: log_lines.append(str(msg)))
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: log_lines.append(str(msg)))
    results = asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=tasks,
            task_meta=[
                {"group": "type1_city", "seed": int(t.map_seed), "challenge_type": 1, "horizon": 60.0}
                for t in tasks
            ],
            batch_plan=[[i] for i in range(seeds)],
            uid=51,
            model_path=model_path,
            effective_workers=1,
            phase_label="eval",
        )
    )
    return results, scripted_context


@pytest.mark.full
def test_run_process_parallel_recycles_worker_after_seed_budget(monkeypatch, tmp_path):
    """A worker past its seed budget is replaced while idle, and all the work still completes."""
    monkeypatch.setattr(de.parallel, "_WORKER_RECYCLE_SEED_BUDGET", 3)
    monkeypatch.setattr(de.parallel, "_WORKER_RECYCLE_MIN_SEEDS", 3)
    log_lines = []

    results, _ = _recycle_scripted_run(monkeypatch, tmp_path, seeds=5, log_lines=log_lines)

    recycle_lines = [line for line in log_lines if "recycling idle worker" in line]
    assert recycle_lines and "seeds served" in recycle_lines[0]
    assert len(results) == 5
    assert all(r.success for r in results)
    assert not any("crashed" in line for line in log_lines)


@pytest.mark.full
def test_run_process_parallel_recycles_worker_on_rss_threshold(monkeypatch, tmp_path):
    """A worker over the resident memory ceiling is replaced while idle, with no crash reported."""
    monkeypatch.setattr(de.parallel, "_WORKER_RECYCLE_MIN_SEEDS", 1)

    class _FakeMem:
        """Memory reading of three gigabytes resident."""
        rss = 3000 * 1024 * 1024

    class _FakeProc:
        """Process handle stand-in reporting a fixed resident size for any pid."""
        def __init__(self, pid):
            """Keep the pid the scheduler asked about."""
            self.pid = pid

        def memory_info(self):
            """Return the fixed three gigabyte resident reading."""
            return _FakeMem()

    monkeypatch.setattr(de.parallel, "psutil", SimpleNamespace(Process=_FakeProc))
    log_lines = []

    results, _ = _recycle_scripted_run(monkeypatch, tmp_path, seeds=3, log_lines=log_lines)

    recycle_lines = [line for line in log_lines if "recycling idle worker" in line]
    assert recycle_lines and "rss 3000" in recycle_lines[0]
    assert len(results) == 3
    assert all(r.success for r in results)
    assert not any("crashed" in line for line in log_lines)


@pytest.mark.full
def test_run_process_parallel_no_recycle_below_thresholds(monkeypatch, tmp_path):
    """A worker under both the seed budget and the memory ceiling is left running."""
    log_lines = []

    results, _ = _recycle_scripted_run(monkeypatch, tmp_path, seeds=4, log_lines=log_lines)

    assert not any("recycling idle worker" in line for line in log_lines)
    assert len(results) == 4
    assert all(r.success for r in results)


def test_release_freed_memory_is_safe_and_wired():
    """The allocator trim returns quietly and the worker entry point still calls it."""
    from swarm.benchmark.engine_parts import workers

    assert workers._release_freed_memory() is None
    import inspect
    body = inspect.getsource(workers._benchmark_worker_main)
    assert "_release_freed_memory()" in body


def test_resolve_worker_limits_drops_quota_for_pinned_workers(monkeypatch):
    """A pinned worker keeps its cpuset and loses the cpu quota, since the pin already bounds it."""
    monkeypatch.setenv("SWARM_DOCKER_WORKER_CPUSET_CPUS_1", "6,7")
    monkeypatch.delenv("SWARM_DOCKER_KEEP_CPU_QUOTA", raising=False)
    limits = de.DockerSecureEvaluator._resolve_worker_limits(1)
    assert limits["cpuset_cpus"] == "6,7"
    assert limits["cpus"] is None


def test_resolve_worker_limits_keeps_quota_when_requested(monkeypatch):
    """Asking for it in the environment leaves the cpu share in place beside the pin."""
    monkeypatch.setenv("SWARM_DOCKER_WORKER_CPUSET_CPUS_1", "6,7")
    monkeypatch.setenv("SWARM_DOCKER_KEEP_CPU_QUOTA", "1")
    limits = de.DockerSecureEvaluator._resolve_worker_limits(1)
    assert limits["cpus"] == de.batch.DOCKER_WORKER_CPUS


def test_resolve_worker_limits_keeps_explicit_quota_when_pinned(monkeypatch):
    """An operator's own cpu override survives the pin, unlike the default share."""
    monkeypatch.setenv("SWARM_DOCKER_WORKER_CPUSET_CPUS_1", "6,7")
    monkeypatch.setenv("SWARM_DOCKER_WORKER_CPUS_OVERRIDE", "1.5")
    monkeypatch.delenv("SWARM_DOCKER_KEEP_CPU_QUOTA", raising=False)
    limits = de.DockerSecureEvaluator._resolve_worker_limits(1)
    assert limits["cpuset_cpus"] == "6,7"
    assert limits["cpus"] == "1.5"


def test_die_with_parent_survives_missing_libc(monkeypatch):
    """Without libc loaded the call returns quietly instead of raising into the worker."""
    from swarm.benchmark.engine_parts import workers

    monkeypatch.setattr(workers, "_libc", None)
    assert workers._die_with_parent() is None


def test_die_with_parent_requests_kill_signal(monkeypatch):
    """The worker asks the kernel for SIGKILL on parent death through PR_SET_PDEATHSIG."""
    import signal as signal_mod

    from swarm.benchmark.engine_parts import workers

    calls = []

    class _FakeLibc:
        """Libc stand-in that records every prctl call and reports success."""
        def prctl(self, *args):
            """Record the arguments and return zero."""
            calls.append(args)
            return 0

    monkeypatch.setattr(workers, "_libc", _FakeLibc())
    workers._die_with_parent()
    assert calls and calls[0][0] == workers._PR_SET_PDEATHSIG
    assert calls[0][1] == signal_mod.SIGKILL


# ── docker cache key ──────────────────────────────────────────────────────────
# The key decides whether a validator rebuilds its runner image. Anything it
# misses is a stale image running against real submissions, so these drive the
# real calculation rather than a stand-in.


@pytest.fixture
def hash_tree(tmp_path, monkeypatch):
    """A miniature repo the cache key can be computed over."""
    pkg = tmp_path / "swarm"
    (pkg / "validator" / "docker").mkdir(parents=True)
    (pkg / "assets").mkdir()
    (tmp_path / ".dockerignore").write_text("**/__pycache__\n**/*.pyc\nswarm/assets\n")
    (pkg / "validator" / "docker" / "Dockerfile").write_text("FROM python:3.11-slim\n")
    (pkg / "validator" / "docker" / "docker-requirements.txt").write_text("numpy\n")
    (pkg / "runner.py").write_text("x = 1\n")
    (pkg / "profile.json").write_text('{"ops": []}\n')
    (pkg / "assets" / "big.bin").write_text("ignored\n")

    monkeypatch.setattr(lifecycle, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(lifecycle, "_swarm_package_dir", lambda: pkg)
    monkeypatch.setattr(
        lifecycle, "_docker_dir", lambda: pkg / "validator" / "docker"
    )
    return pkg


def _digest():
    """Return the docker cache key computed over the fixture tree."""
    return lifecycle._calculate_docker_hash(None)


def test_docker_hash_changes_when_python_changes(hash_tree):
    """Editing a source file inside the build context moves the key."""
    before = _digest()
    (hash_tree / "runner.py").write_text("x = 2\n")
    assert _digest() != before


def test_docker_hash_changes_when_data_file_changes(hash_tree):
    """The bug this replaces: the key only ever hashed *.py."""
    before = _digest()
    (hash_tree / "profile.json").write_text('{"ops": ["Add"]}\n')
    assert _digest() != before


def test_docker_hash_changes_when_file_added(hash_tree):
    """A new file in the build context moves the key, even one nothing imports."""
    before = _digest()
    (hash_tree / "agent.capnp").write_text("@0xdeadbeef;\n")
    assert _digest() != before


def test_docker_hash_changes_when_file_deleted(hash_tree):
    """Removing a file from the build context moves the key."""
    before = _digest()
    (hash_tree / "profile.json").unlink()
    assert _digest() != before


def test_docker_hash_changes_when_mode_changes(hash_tree):
    """COPY preserves the mode, so a chmod alone still changes the image."""
    target = hash_tree / "runner.py"
    before = _digest()
    target.chmod(0o755)
    assert _digest() != before


def test_docker_hash_changes_when_symlink_target_changes(hash_tree):
    """Repointing a link moves the key even though no file content was touched."""
    link = hash_tree / "current.py"
    link.symlink_to("runner.py")
    before = _digest()
    link.unlink()
    link.symlink_to("profile.json")
    assert _digest() != before


def test_docker_hash_distinguishes_file_from_symlink_with_same_bytes(hash_tree):
    """A symlink hashes its target, never the bytes it resolves to."""
    (hash_tree / "runner.py").write_text("payload\n")
    plain = hash_tree / "copy.py"
    plain.write_text("payload\n")
    as_file = _digest()

    plain.unlink()
    plain.symlink_to("runner.py")
    assert _digest() != as_file


def test_docker_hash_ignores_files_excluded_from_the_build(hash_tree):
    """Assets are not in the image, so they must not force a rebuild."""
    before = _digest()
    (hash_tree / "assets" / "big.bin").write_text("still ignored, different\n")
    assert _digest() == before


def test_docker_hash_marks_a_missing_required_input(hash_tree):
    """Deleting the Dockerfile moves the key rather than leaving it unchanged."""
    before = _digest()
    (hash_tree / "validator" / "docker" / "Dockerfile").unlink()
    assert _digest() != before


def test_docker_hash_covers_a_file_a_negation_puts_back(hash_tree, tmp_path):
    """`!pattern` re-includes a file the line above excluded, so it is in the image."""
    (tmp_path / ".dockerignore").write_text(
        "swarm/assets\n!swarm/assets/needed.json\n"
    )
    (hash_tree / "assets" / "needed.json").write_text("first\n")
    before = _digest()
    (hash_tree / "assets" / "needed.json").write_text("second\n")
    assert _digest() != before


def test_docker_hash_covers_a_directory_mode_change(hash_tree):
    """COPY carries directory permissions, so this changes the image on its own."""
    (hash_tree / "nested").mkdir()
    (hash_tree / "nested" / "keep.py").write_text("x = 1\n")
    (hash_tree / "nested").chmod(0o755)
    before = _digest()
    (hash_tree / "nested").chmod(0o700)
    assert _digest() != before


def test_docker_hash_covers_a_symlink_to_a_directory(hash_tree):
    """Docker copies the link itself, so retargeting it changes the image."""
    (hash_tree / "one").mkdir()
    (hash_tree / "two").mkdir()
    link = hash_tree / "current"
    link.symlink_to("one", target_is_directory=True)
    before = _digest()
    link.unlink()
    link.symlink_to("two", target_is_directory=True)
    assert _digest() != before


def test_every_name_read_through_the_evaluator_facade_exists():
    """The parts read some constants off the evaluator module at call time, so a
    name dropped from its import list only fails on a live host."""
    parts_dir = Path(de.__file__).parent / "docker_evaluator_parts"
    pattern = re.compile(r"(?:_docker_evaluator_facade\(\)|\bfacade)\.([A-Za-z_]+)")
    names = set()
    for source in parts_dir.glob("*.py"):
        names.update(pattern.findall(source.read_text()))
    assert names, "no facade reads found; the pattern is stale"
    missing = sorted(n for n in names if not hasattr(de, n))
    assert not missing, f"read through the facade but not on the module: {missing}"
