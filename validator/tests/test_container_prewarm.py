"""The next seed's container starts while the current seed flies, on the same worker."""

from __future__ import annotations

import asyncio
import queue
import socket
import time
import zipfile
from types import SimpleNamespace

import pytest

from swarm.benchmark import engine as bench_full_eval
from swarm.benchmark.engine_parts import workers
from swarm.protocol import ValidationResult
from swarm.validator.docker import docker_evaluator as de
from swarm.validator.docker.docker_evaluator_parts import batch
from validator.tests.test_docker_evaluator import _ScriptedContext

_LOAD_SEC = 1.0
_FLY_SEC = 2.5


class _FakeDocker:
    """A docker daemon in memory: containers become ready ``_LOAD_SEC`` after their run
    and hold their published host port until they are killed, as docker-proxy does."""

    def __init__(self, update_ok: bool = True) -> None:
        """Record every command; ``update_ok`` scripts the CPU-weight restore."""
        self.events: list[tuple[float, str, str]] = []
        self.runs: list[list[str]] = []
        self.updates: list[list[str]] = []
        self.ready_at: dict[int, float] = {}
        self.port_of: dict[str, int] = {}
        self.owner: dict[int, str] = {}
        self.sockets: dict[str, socket.socket] = {}
        self.update_ok = update_ok

    def _note(self, kind: str, subject: str) -> None:
        """Append one timestamped event."""
        self.events.append((time.monotonic(), kind, subject))

    def run(self, cmd, **kwargs):
        """Stand in for ``subprocess.run`` on every docker command the batch issues."""
        _ = kwargs
        verb = cmd[1] if len(cmd) > 1 else ""
        if verb == "run":
            name = cmd[cmd.index("--name") + 1]
            port = int(cmd[cmd.index("-p") + 1].split(":")[1])
            self.runs.append(list(cmd))
            self.port_of[name] = port
            self.owner[port] = name
            self.ready_at[port] = time.monotonic() + _LOAD_SEC
            held = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            held.bind(("127.0.0.1", port))
            self.sockets[name] = held
            self._note("run", name)
        elif verb == "update":
            self.updates.append(list(cmd))
            self._note("update", cmd[-1])
            return SimpleNamespace(returncode=0 if self.update_ok else 1, stdout="", stderr="no")
        elif verb in ("kill", "rm"):
            held = self.sockets.pop(cmd[-1], None)
            if held is not None:
                held.close()
            self._note(verb, cmd[-1])
        elif verb == "inspect":
            return SimpleNamespace(returncode=0, stdout="4242\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def rpc_ready(self, port: int) -> bool:
        """True once the container behind ``port`` has finished loading."""
        return port in self.ready_at and time.monotonic() >= self.ready_at[port]

    def ready_time(self, name: str) -> float:
        """When the container ``name`` finished loading."""
        return self.ready_at[self.port_of[name]]

    def fly(self, tasks, uid, rpc_port, *args, **kwargs):
        """Stand in for the RPC flight: one scored result per task after ``_FLY_SEC``."""
        _ = args, kwargs
        name = self.owner[int(rpc_port)]
        self._note("fly_start", name)
        time.sleep(_FLY_SEC)
        self._note("fly_end", name)
        return [ValidationResult(uid, True, 1.0, 0.5) for _ in tasks]

    def first(self, kind: str, subject: str) -> float:
        """Time of the first event of ``kind`` on ``subject``."""
        return next(t for t, k, s in self.events if k == kind and s == subject)

    def names(self) -> list[str]:
        """Container names in launch order."""
        return [cmd[cmd.index("--name") + 1] for cmd in self.runs]


@pytest.fixture
def fake_docker(monkeypatch, tmp_path):
    """Wire the fake daemon into the evaluator and hand back a real, tiny submission."""
    docker = _FakeDocker()
    monkeypatch.setattr(batch.subprocess, "run", docker.run)
    monkeypatch.setattr(de.DockerSecureEvaluator, "_get_container_pid", lambda self, name: 4242)
    monkeypatch.setattr(de.DockerSecureEvaluator, "_apply_network_lockdown", lambda self, pid, ip: True)
    monkeypatch.setattr(de.DockerSecureEvaluator, "_get_docker_host_ip", lambda self: "172.17.0.1")
    monkeypatch.setattr(
        de.DockerSecureEvaluator, "_check_rpc_ready",
        lambda self, port, timeout=0.2: docker.rpc_ready(int(port)),
    )
    monkeypatch.setattr(de.DockerSecureEvaluator, "_run_multi_seed_rpc_sync", docker.fly)
    monkeypatch.setattr(batch, "validate_submission_zip", lambda path: (True, ""))
    monkeypatch.setattr(workers, "_die_with_parent", lambda: None)
    monkeypatch.setattr(workers, "_apply_host_worker_limits", lambda slot: None)
    monkeypatch.delenv("SWARM_DOCKER_PREWARM", raising=False)
    de.DockerSecureEvaluator._base_ready = True
    model_path = tmp_path / "UID_7.zip"
    with zipfile.ZipFile(model_path, "w") as zf:
        zf.writestr("drone_agent.py", "class DroneFlightController: pass\n")
    docker.model_path = model_path
    return docker


def _request(index: int, uid: int, model_path, prewarm_next: bool):
    """One single-seed batch request as the dispatchers build it."""
    return bench_full_eval._ProcessBatchRequest(
        batch_index=index,
        batch_indices=[index],
        tasks=[SimpleNamespace(map_seed=1000 + index, challenge_type=2, horizon=30.0)],
        uid=uid,
        model_path=str(model_path),
        task_total=2,
        runtime_profile=None,
        host_speed_factor=1.0,
        prewarm_next=prewarm_next,
    )


def _run_worker(requests) -> list:
    """Drive one worker process body through ``requests`` and collect its results."""
    task_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    for request in requests:
        task_queue.put(request)
    task_queue.put(None)
    workers._benchmark_worker_main(0, task_queue, result_queue, queue.Queue())
    results = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue.Empty:
            return results


def test_next_container_starts_during_the_flight(fake_docker):
    """The second seed's container launches mid-flight, at low CPU weight, and is
    raised to the default weight before it flies; both seeds score normally."""
    results = _run_worker([
        _request(0, 7, fake_docker.model_path, prewarm_next=True),
        _request(1, 7, fake_docker.model_path, prewarm_next=False),
    ])

    first, second = fake_docker.names()
    assert len(fake_docker.runs) == 2
    assert fake_docker.ready_time(first) <= fake_docker.first("run", second)
    assert fake_docker.first("run", second) < fake_docker.first("fly_end", first)
    assert "--cpu-shares=2" in fake_docker.runs[1] and "--cpus=0.5" in fake_docker.runs[1]
    assert "--cpu-shares=2" not in fake_docker.runs[0] and "--cpus=0.5" not in fake_docker.runs[0]
    assert "--cpu-shares=1024" in fake_docker.updates[0]
    assert any(arg == "--cpu-quota=-1" or arg.startswith("--cpus=") for arg in fake_docker.updates[0])
    assert fake_docker.first("fly_end", first) < fake_docker.first("update", second)
    assert fake_docker.first("update", second) < fake_docker.first("fly_start", second)
    assert [t for _, k, t in fake_docker.events if k == "kill"] == [first, second]
    assert [r.results[0][3] for r in results] == [0.5, 0.5]
    assert all(r.error is None for r in results)


def test_second_seed_does_not_wait_for_its_container(fake_docker, monkeypatch):
    """Serially the second flight starts a full model load after the first one ends;
    with the start hidden behind the first flight that gap shrinks below the load time."""
    def _gap_between_flights() -> float:
        """Seconds from the end of the first flight to the start of the second."""
        fake_docker.events.clear()
        _run_worker([
            _request(0, 7, fake_docker.model_path, prewarm_next=True),
            _request(1, 7, fake_docker.model_path, prewarm_next=False),
        ])
        first, second = fake_docker.names()[-2:]
        return fake_docker.first("fly_start", second) - fake_docker.first("fly_end", first)

    monkeypatch.setenv("SWARM_DOCKER_PREWARM", "0")
    assert _gap_between_flights() > _LOAD_SEC
    monkeypatch.setenv("SWARM_DOCKER_PREWARM", "1")
    assert _gap_between_flights() < _LOAD_SEC


def test_spare_for_the_last_seed_is_killed_at_shutdown(fake_docker):
    """A spare that never gets a seed is killed when the worker stops, never flown."""
    _run_worker([_request(0, 7, fake_docker.model_path, prewarm_next=True)])

    first, spare = fake_docker.names()
    assert [t for _, k, t in fake_docker.events if k == "fly_start"] == [first]
    assert fake_docker.first("kill", spare) > fake_docker.first("fly_end", first)


def test_prewarm_switch_keeps_the_serial_flow(fake_docker, monkeypatch):
    """With SWARM_DOCKER_PREWARM off every container starts after the previous flight."""
    monkeypatch.setenv("SWARM_DOCKER_PREWARM", "0")
    _run_worker([
        _request(0, 7, fake_docker.model_path, prewarm_next=True),
        _request(1, 7, fake_docker.model_path, prewarm_next=True),
    ])

    first, second = fake_docker.names()
    assert len(fake_docker.runs) == 2
    assert fake_docker.first("run", second) > fake_docker.first("fly_end", first)
    assert not any("--cpu-shares" in arg or arg == "--cpus=0.5" for cmd in fake_docker.runs for arg in cmd)


def test_spare_for_another_miner_is_discarded(fake_docker, tmp_path):
    """A spare only serves the miner it was started for; another miner gets a fresh start."""
    other = tmp_path / "UID_8.zip"
    other.write_bytes(fake_docker.model_path.read_bytes())
    results = _run_worker([
        _request(0, 7, fake_docker.model_path, prewarm_next=True),
        _request(1, 8, other, prewarm_next=False),
    ])

    first, spare, fresh = fake_docker.names()
    flown = [t for _, k, t in fake_docker.events if k == "fly_start"]
    assert flown == [first, fresh]
    assert fake_docker.first("kill", spare) < fake_docker.first("run", fresh)
    assert [r.results[0][3] for r in results] == [0.5, 0.5]


def test_failed_weight_restore_falls_back_to_a_fresh_start(fake_docker):
    """When the spare's CPU limits cannot be restored it is dropped and the seed starts
    its own container, so the flight never runs at the spare's low weight or quota."""
    fake_docker.update_ok = False
    results = _run_worker([
        _request(0, 7, fake_docker.model_path, prewarm_next=True),
        _request(1, 7, fake_docker.model_path, prewarm_next=False),
    ])

    first, spare, fresh = fake_docker.names()
    flown = [t for _, k, t in fake_docker.events if k == "fly_start"]
    assert flown == [first, fresh]
    assert fake_docker.first("kill", spare) < fake_docker.first("run", fresh)
    assert "--cpu-shares=2" not in fake_docker.runs[2]
    assert [r.results[0][3] for r in results] == [0.5, 0.5]


def test_validator_dispatcher_asks_for_a_spare_until_the_last_seed(monkeypatch, tmp_path):
    """A one-worker plan of three seeds carries the hint on the first two only."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    plan = {
        "seed_events": [],
        "results": [(41, True, 10.0, 0.7)],
        "elapsed_sec": 10.0,
    }
    scripted_context = _ScriptedContext(bench_full_eval, {0: [plan], 1: [plan], 2: [plan]})
    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: scripted_context)
    tasks = [SimpleNamespace(challenge_type=2, map_seed=s, horizon=30.0) for s in (1, 2, 3)]

    asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=tasks,
            task_meta=[
                {"group": "type2_open", "seed": t.map_seed, "index": i, "challenge_type": 2, "horizon": 30.0}
                for i, t in enumerate(tasks)
            ],
            batch_plan=[[0], [1], [2]],
            uid=41,
            model_path=model_path,
            effective_workers=1,
        )
    )

    assert [r.prewarm_next for r in scripted_context.requests] == [True, True, False]
