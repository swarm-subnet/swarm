from __future__ import annotations

import asyncio
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.protocol import ValidationResult
from swarm.validator.docker import docker_evaluator as de


class _ProcResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _new_evaluator() -> de.DockerSecureEvaluator:
    ev = de.DockerSecureEvaluator.__new__(de.DockerSecureEvaluator)
    ev.base_image = "swarm_evaluator_base:latest"
    ev.base_ready = True
    de.DockerSecureEvaluator._base_ready = True
    return ev


def test_normalize_package_name():
    assert (
        de.DockerSecureEvaluator._normalize_package_name("NumPy_Pkg.Name")
        == "numpy-pkg-name"
    )


def test_validate_requirements_accepts_whitelisted_packages(tmp_path):
    ev = _new_evaluator()
    req = tmp_path / "requirements.txt"
    req.write_text(
        "\n".join(
            [
                "# comment",
                "numpy>=1.24",
                "torch==2.0.0",
            ]
        )
    )
    assert ev._validate_requirements(req, uid=1) is True


@pytest.mark.parametrize(
    "line",
    [
        "-r other.txt",
        "git+https://example.com/repo.git",
        "mypkg @ https://example.com/pkg.whl",
        "httpx==0.27.0",  # not in DOCKER_PIP_WHITELIST
    ],
)
def test_validate_requirements_rejects_disallowed_lines(tmp_path, line):
    ev = _new_evaluator()
    req = tmp_path / "requirements.txt"
    req.write_text(line + "\n")
    assert ev._validate_requirements(req, uid=2) is False


def test_serialize_observation_dict():
    class _ObsMessage:
        def init(self, field, n):
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
        @staticmethod
        def new_message():
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
    class _ObsMessage:
        def init(self, field, n):
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
        @staticmethod
        def new_message():
            return _ObsMessage()

    schema = SimpleNamespace(Observation=_Observation)
    msg = de.DockerSecureEvaluator._serialize_observation(
        schema, np.array([5, 6], dtype=np.float32)
    )
    assert msg.entries[0].key == "__value__"
    assert msg.entries[0].tensor.shape == [2]


def test_check_docker_available_true(monkeypatch):
    ev = _new_evaluator()
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda *a, **k: _ProcResult(returncode=0, stdout="Docker version 26"),
    )
    assert ev._check_docker_available() is True


def test_check_docker_available_false_on_missing_binary(monkeypatch):
    ev = _new_evaluator()

    def _raise(*args, **kwargs):
        _ = args, kwargs
        raise FileNotFoundError("docker")

    monkeypatch.setattr(de.subprocess, "run", _raise)
    assert ev._check_docker_available() is False


def test_cleanup_env_quietly_closes_env():
    calls = {"count": 0}

    class _Env:
        def close(self):
            calls["count"] += 1

    de._cleanup_env_quietly(_Env())
    assert calls["count"] == 1


def test_submission_template_dir_points_to_swarm_template():
    template_dir = de._submission_template_dir()
    assert template_dir == Path("swarm/submission_template").resolve()
    assert (template_dir / "agent.capnp").is_file()


def test_get_image_hash_label(monkeypatch):
    ev = _new_evaluator()
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda *a, **k: _ProcResult(returncode=0, stdout="abc123\n"),
    )
    assert ev._get_image_hash_label() == "abc123"


def test_should_rebuild_base_image_when_image_missing(monkeypatch):
    ev = _new_evaluator()
    monkeypatch.setattr(ev, "_calculate_docker_hash", lambda: "hash1")
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda cmd, **k: _ProcResult(returncode=0, stdout=""),
    )
    assert ev._should_rebuild_base_image() is True


def test_should_rebuild_base_image_false_when_hash_matches(monkeypatch):
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
    ev = _new_evaluator()
    build_cmds = []

    monkeypatch.setattr(ev, "_check_docker_available", lambda: True)
    monkeypatch.setattr(ev, "_should_rebuild_base_image", lambda: True)
    monkeypatch.setattr(ev, "_calculate_docker_hash", lambda: "hash1")

    def _run(cmd, **kwargs):
        _ = kwargs
        if isinstance(cmd, list) and cmd[:2] == ["docker", "build"]:
            build_cmds.append(cmd)
        return _ProcResult(returncode=0, stdout="")

    monkeypatch.setattr(de.subprocess, "run", _run)
    de.DockerSecureEvaluator._base_ready = False

    ev._setup_base_container()

    assert ev.base_ready is True
    assert len(build_cmds) == 1
    dockerfile_path = Path("swarm/validator/docker/Dockerfile").resolve()
    assert build_cmds[0][build_cmds[0].index("-f") + 1] == str(dockerfile_path)


def test_check_rpc_ready(monkeypatch):
    ev = _new_evaluator()
    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda *a, **k: _ProcResult(returncode=0, stdout="PID CMD\n1 python main.py\n"),
    )
    assert ev._check_rpc_ready("container") is True

    monkeypatch.setattr(
        de.subprocess,
        "run",
        lambda *a, **k: _ProcResult(returncode=0, stdout="PID CMD\n1 sleep infinity\n"),
    )
    assert ev._check_rpc_ready("container") is False


def test_get_docker_host_ip_and_fallback(monkeypatch):
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
    ev = _new_evaluator()
    calls = {"count": 0}

    def _run(*args, **kwargs):
        _ = args, kwargs
        calls["count"] += 1
        return _ProcResult(returncode=0)

    monkeypatch.setattr(de.subprocess, "run", _run)
    assert ev._apply_network_lockdown(9999, "10.0.0.1") is True
    assert calls["count"] == 4


def test_apply_network_lockdown_failure_when_rule_fails(monkeypatch):
    ev = _new_evaluator()
    calls = {"count": 0}

    def _run(*args, **kwargs):
        _ = args, kwargs
        calls["count"] += 1
        if calls["count"] == 2:
            return _ProcResult(returncode=1, stderr="iptables failed")
        return _ProcResult(returncode=0)

    monkeypatch.setattr(de.subprocess, "run", _run)
    assert ev._apply_network_lockdown(9999, "10.0.0.1") is False


def test_docker_env_overrides_enable_thread_caps(monkeypatch):
    monkeypatch.setenv("SWARM_DOCKER_THREAD_CAPS", "1")
    envs = de.DockerSecureEvaluator._docker_env_overrides()
    assert envs["OMP_NUM_THREADS"] == "1"
    assert envs["SWARM_TORCH_NUM_THREADS"] == "1"
    assert envs["SWARM_TORCH_INTEROP_THREADS"] == "1"


def test_resolve_worker_limits_uses_env_overrides(monkeypatch):
    monkeypatch.setenv("SWARM_DOCKER_WORKER_CPUS_OVERRIDE", "1.0")
    monkeypatch.setenv("SWARM_DOCKER_WORKER_MEMORY_OVERRIDE", "4g")
    monkeypatch.setenv("SWARM_DOCKER_WORKER_CPUSETS", "0-1;2-3")
    limits = de.DockerSecureEvaluator._resolve_worker_limits(worker_id=1)
    assert limits["cpus"] == "1.0"
    assert limits["memory"] == "4g"
    assert limits["cpuset_cpus"] == "2-3"


def test_calibrate_rpc_overhead_fallback_on_failures(monkeypatch):
    ev = _new_evaluator()
    monkeypatch.setattr(ev, "_serialize_observation", lambda *a, **k: object())
    monkeypatch.setattr(de, "CALIBRATION_ROUNDS", 4)

    class _Agent:
        async def calibrate(self, obs):
            _ = obs
            raise asyncio.TimeoutError()

    overhead, cpu_factor = asyncio.run(
        ev._calibrate_rpc_overhead_async(
            _Agent(), object(), {"state": np.zeros(2)}, uid=5
        )
    )
    fallback = max(de.RPC_STEP_TIMEOUT_SEC - de.MINER_COMPUTE_BUDGET_SEC, 0.010)
    assert overhead == fallback
    assert cpu_factor == 1.0


def test_calibrate_rpc_overhead_success(monkeypatch):
    ev = _new_evaluator()
    monkeypatch.setattr(ev, "_serialize_observation", lambda *a, **k: object())
    monkeypatch.setattr(de, "CALIBRATION_ROUNDS", 4)

    timeline = iter([0.0, 0.05, 1.0, 1.06, 2.0, 2.07, 3.0, 3.08])
    last = {"t": 3.08}

    def _fake_time():
        try:
            last["t"] = next(timeline)
            return last["t"]
        except StopIteration:
            return last["t"]

    monkeypatch.setattr(de.time, "time", _fake_time)

    class _Resp:
        def __init__(self, benchmark_ns):
            self.benchmarkNs = benchmark_ns

    class _Agent:
        async def calibrate(self, obs):
            _ = obs
            return _Resp(12_000_000)

    overhead, cpu_factor = asyncio.run(
        ev._calibrate_rpc_overhead_async(
            _Agent(), object(), {"state": np.zeros(2)}, uid=6
        )
    )
    expected_overheads = [0.038, 0.048, 0.058, 0.068]
    assert overhead == pytest.approx(np.median(expected_overheads), abs=1e-9)
    assert cpu_factor == pytest.approx(1.0, abs=1e-9)


def test_evaluate_seeds_parallel_uses_process_scheduler(monkeypatch, tmp_path):
    ev = _new_evaluator()
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    tasks = [
        SimpleNamespace(challenge_type=3, map_seed=1001, seed_id=0),
        SimpleNamespace(challenge_type=2, map_seed=1002, seed_id=1),
        SimpleNamespace(challenge_type=6, map_seed=1003, seed_id=2),
    ]
    callback_payloads = []
    captured = {}

    async def _fake_run_process_parallel(**kwargs):
        captured["batch_plan"] = kwargs["batch_plan"]
        captured["effective_workers"] = kwargs["effective_workers"]
        captured["task_meta"] = kwargs["task_meta"]
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
    assert [payload["status"] for payload in callback_payloads] == [
        "seed_done",
        "seed_done",
        "seed_done",
    ]


def test_evaluate_seeds_parallel_uses_default_worker_count_of_three(monkeypatch, tmp_path):
    ev = _new_evaluator()
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    tasks = [
        SimpleNamespace(challenge_type=1, map_seed=2001),
        SimpleNamespace(challenge_type=2, map_seed=2002),
        SimpleNamespace(challenge_type=3, map_seed=2003),
        SimpleNamespace(challenge_type=4, map_seed=2004),
        SimpleNamespace(challenge_type=5, map_seed=2005),
    ]
    captured = {}

    async def _fake_run_process_parallel(**kwargs):
        captured["effective_workers"] = kwargs["effective_workers"]
        return [ValidationResult(kwargs["uid"], True, 1.0, 0.5) for _ in kwargs["all_tasks"]]

    monkeypatch.setattr(de.parallel, "_run_process_parallel", _fake_run_process_parallel)
    results = asyncio.run(ev.evaluate_seeds_parallel(tasks, uid=17, model_path=model_path))

    assert len(results) == 5
    assert captured["effective_workers"] == 3


def test_heavy_aware_chunk_distributes_heavy_maps_evenly():
    tasks = [
        SimpleNamespace(challenge_type=3),
        SimpleNamespace(challenge_type=5),
        SimpleNamespace(challenge_type=3),
        SimpleNamespace(challenge_type=5),
        SimpleNamespace(challenge_type=1),
        SimpleNamespace(challenge_type=2),
        SimpleNamespace(challenge_type=1),
        SimpleNamespace(challenge_type=4),
    ]

    chunks, index_map = de._heavy_aware_chunk(tasks, 4)

    assert len(chunks) == 4
    assert sum(len(c) for c in chunks) == 8
    assert set(idx for indices in index_map for idx in indices) == set(range(8))

    for chunk in chunks:
        heavy_count = sum(
            1 for t in chunk if t.challenge_type in de._HEAVY_CHALLENGE_TYPES
        )
        assert heavy_count <= 1


def test_evaluate_seeds_parallel_falls_back_to_batch_when_docker_not_ready(monkeypatch, tmp_path):
    ev = _new_evaluator()
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    tasks = [
        SimpleNamespace(challenge_type=3, map_seed=3001),
        SimpleNamespace(challenge_type=1, map_seed=3002),
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
    ):
        captured["chunk"] = list(chunk)
        captured["worker_id"] = worker_id
        captured["task_offset"] = task_offset
        captured["task_total"] = task_total
        _ = model_path, on_seed_complete
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in chunk]

    monkeypatch.setattr(de.DockerSecureEvaluator, "_base_ready", False)
    monkeypatch.setattr(ev, "evaluate_seeds_batch", _fake_batch)
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


def test_evaluate_seeds_batch_returns_failures_when_model_missing(tmp_path):
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
    ev = _new_evaluator()
    de.DockerSecureEvaluator._base_ready = False
    model = tmp_path / "model.zip"
    with zipfile.ZipFile(model, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("drone_agent.py", "class Agent: pass")
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
    monkeypatch.setattr(
        de.DockerSecureEvaluator, "_check_docker_available", lambda self: False
    )
    de.DockerSecureEvaluator._instance = None
    de.DockerSecureEvaluator._base_ready = False

    evaluator = de.DockerSecureEvaluator()

    assert evaluator.base_ready is False
    assert de.DockerSecureEvaluator._base_ready is False
