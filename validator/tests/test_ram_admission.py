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

"""RAM admission for Swarm Sentinel patrols: the prior covers a measured patrol, no reading is spent twice,
and a host short of memory runs one seed at a time instead of stalling."""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from swarm.benchmark import engine as bench_full_eval
from swarm.benchmark.engine_parts import dispatch, workers
from swarm.validator.docker import docker_evaluator as de

MEASURED_PATROL_PEAK_MB = 3344.0
SIXTEEN_GIB_MB = 16384.0
EIGHT_GIB_MB = 8192.0
PATROLS = 7


def _fixed_reading(mem_available_mb: float, mem_total_mb: float):
    """A resource provider that always reports the same free memory, as when started patrols have not grown yet."""
    return lambda: {
        "cpu_percent": 0.0,
        "load_ratio": 0.0,
        "mem_available_mb": mem_available_mb,
        "mem_total_mb": mem_total_mb,
        "ts": 1.0,
    }


def _scheduler(total_mb: float, free_mb: float, workers_requested: int = PATROLS):
    """A scheduler on a host of that size, holding one reading of that much free memory."""
    scheduler = dispatch._RamWorkerScheduler(
        requested_workers=workers_requested,
        machine_vcpus=16,
        machine_total_ram_mb=total_mb,
        resource_provider=_fixed_reading(free_mb, total_mb),
    )
    scheduler.refresh_resources()
    return scheduler


def _admit_until_refused(scheduler, group_name: str) -> list[str]:
    """Start seeds of one group the way a dispatch pass does, until the scheduler refuses the next one."""
    active: list[str] = []
    while scheduler.can_admit_group(group_name, active):
        alone = not active
        active.append(group_name)
        scheduler.note_group_started(group_name, alone=alone)
    return active


def test_sentinel_prior_covers_the_measured_patrol():
    """The memory booked for a patrol is at least the peak measured over 100 real patrols."""
    assert dispatch._ram_estimate_for_group("type8_solar") >= MEASURED_PATROL_PEAK_MB


def test_sixteen_gib_box_admits_only_the_patrols_that_fit():
    """A 16 GiB box with seven slots starts three patrols, and three measured peaks fit beside the host reserve."""
    scheduler = _scheduler(SIXTEEN_GIB_MB, 15000.0)

    admitted = _admit_until_refused(scheduler, "type8_solar")

    assert len(admitted) == 3
    assert len(admitted) * MEASURED_PATROL_PEAK_MB + scheduler.ram_reserve_mb <= SIXTEEN_GIB_MB


def test_patrols_started_since_a_reading_count_against_it():
    """One reading of 9 GiB free admits one patrol; the next waits for a fresh reading, which clears the count."""
    scheduler = _scheduler(32768.0, 9000.0, workers_requested=8)

    admitted = _admit_until_refused(scheduler, "type8_solar")
    assert len(admitted) == 1

    scheduler.refresh_resources()
    assert scheduler.can_admit_group("type8_solar", admitted)


@pytest.mark.parametrize(
    ("total_mb", "free_mb"),
    [
        pytest.param(EIGHT_GIB_MB, 5000.0, id="free-memory-short"),
        pytest.param(4096.0, 3900.0, id="host-smaller-than-a-patrol"),
    ],
)
def test_a_host_short_of_memory_runs_one_patrol_at_a_time(total_mb, free_mb):
    """With nothing in flight a patrol starts even when it does not fit, warns once, and a second waits for the first."""
    scheduler = _scheduler(total_mb, free_mb)

    assert scheduler.can_admit_group("type8_solar", [])
    warning = scheduler.note_group_started("type8_solar", alone=True)
    assert "type8_solar" in warning
    assert "one at a time" in warning
    assert not scheduler.can_admit_group("type8_solar", ["type8_solar"])

    scheduler.refresh_resources()
    assert scheduler.note_group_started("type8_solar", alone=True) is None


def test_a_patrol_that_fits_starts_without_a_warning():
    """A patrol with the memory it needs logs nothing, alone or not."""
    scheduler = _scheduler(SIXTEEN_GIB_MB, 15000.0)

    assert scheduler.note_group_started("type8_solar", alone=True) is None
    assert scheduler.note_group_started("type8_solar", alone=False) is None


@pytest.mark.parametrize(
    ("total_mb", "at_once"),
    [
        pytest.param(SIXTEEN_GIB_MB, 3, id="16-GiB"),
        pytest.param(32768.0, 7, id="32-GiB"),
        pytest.param(4096.0, 1, id="4-GiB"),
    ],
)
def test_startup_banner_says_how_many_patrols_run_at_once(total_mb, at_once):
    """The scheduler's startup lines give each map type's prior and how many of its seeds the host runs side by side."""
    scheduler = _scheduler(total_mb, total_mb - 1000.0, workers_requested=8)

    lines = scheduler.describe_configuration_lines()

    assert f"Scheduler RAM prior: group=type8_solar ram=3700MiB at_once={at_once}" in lines


class _ConcurrencyProbe:
    """Counts the batches the stand-in workers hold at once and keeps the highest count seen."""

    def __init__(self):
        """Start with nothing running."""
        self._lock = threading.Lock()
        self.running = 0
        self.peak = 0

    def enter(self):
        """Record one more batch in flight."""
        with self._lock:
            self.running += 1
            self.peak = max(self.peak, self.running)

    def leave(self):
        """Record one batch finished."""
        with self._lock:
            self.running -= 1


class _ThreadWorker:
    """A worker process stand-in on a thread: it holds each batch briefly, then posts a passing result."""

    def __init__(self, probe, args):
        """Unpack the slot and the three queues from the spawn arguments."""
        self._probe = probe
        self.worker_slot, self.task_queue, self.result_queue, self.progress_queue = args[:4]
        self.exitcode = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        """Serve batches on a daemon thread until a None request or a terminate."""

        def _serve():
            """Announce each batch, hold it while counted as running, then post its result."""
            while True:
                request = self.task_queue.get()
                if request is None:
                    self.exitcode = 0
                    return
                self.progress_queue.put(
                    bench_full_eval._ProcessWorkerHeartbeat(
                        worker_id=self.worker_slot,
                        batch_index=request.batch_index,
                        event_type="batch_started",
                        ts=time.time(),
                    )
                )
                self._probe.enter()
                stopped = self._stop.wait(0.1)
                self._probe.leave()
                if stopped:
                    self.exitcode = -15
                    return
                self.result_queue.put(
                    bench_full_eval._ProcessBatchResult(
                        worker_id=self.worker_slot,
                        batch_index=request.batch_index,
                        batch_indices=list(request.batch_indices),
                        results=[(int(request.uid), True, 12.0, 0.9)],
                        elapsed_sec=0.1,
                    )
                )

        self._thread = threading.Thread(target=_serve, daemon=True)
        self._thread.start()

    def is_alive(self):
        """True while the serving thread runs."""
        return bool(self._thread and self._thread.is_alive())

    def join(self, timeout=None):
        """Wait on the serving thread when one was started."""
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def terminate(self):
        """Stop the serving thread without posting a result."""
        self.exitcode = -15
        self._stop.set()


class _ThreadContext:
    """Multiprocessing context stand-in handing out plain queues, events and thread workers."""

    def __init__(self, probe):
        """Keep the probe every worker reports to."""
        self._probe = probe

    def Queue(self):
        """Return an in-process queue."""
        return queue.Queue()

    def Event(self):
        """Return the cancel signal the validator shares with its workers."""
        return threading.Event()

    def Process(self, target, args, name=None, daemon=None):
        """Return a thread worker in place of a forked process; target, name and daemon are unused."""
        _ = target, name, daemon
        return _ThreadWorker(self._probe, args)


@pytest.fixture
def fake_host(monkeypatch):
    """Return a function that makes the scheduler see a host of a given size whose free memory never moves.

    Workers run as threads. The function returns the probe of patrols running at once and the list of started groups."""
    note_group_started = dispatch._RamWorkerScheduler.note_group_started

    def _configure(total_mb: float, free_mb: float):
        """Point the scheduler at a host of total_mb holding free_mb free, and record every start."""
        probe = _ConcurrencyProbe()
        started: list[str] = []

        def _recording_note_group_started(self, group_name, **kwargs):
            """Remember the start, then charge it as the scheduler does."""
            started.append(str(group_name))
            return note_group_started(self, group_name, **kwargs)

        monkeypatch.setattr(dispatch, "_detect_total_ram_mb", lambda: total_mb)
        monkeypatch.setattr(
            dispatch,
            "_sample_resource_snapshot",
            lambda machine_vcpus: dispatch._ResourceSnapshot(
                cpu_percent=0.0,
                load_ratio=0.0,
                mem_available_mb=free_mb,
                mem_total_mb=total_mb,
                ts=time.monotonic(),
            ),
        )
        monkeypatch.setattr(dispatch._RamWorkerScheduler, "note_group_started", _recording_note_group_started)
        monkeypatch.setattr(bench_full_eval, "_RESOURCE_POLL_INTERVAL_SEC", 0.02, raising=False)
        monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: _ThreadContext(probe))
        return SimpleNamespace(probe=probe, started=started)

    return _configure


def _sentinel_seeds():
    """Seven one-seed Swarm Sentinel batches: the tasks, their metadata and the batch plan."""
    tasks = [SimpleNamespace(map_seed=5000 + i, challenge_type=8, horizon=390.0) for i in range(PATROLS)]
    meta = [
        {"group": "type8_solar", "seed": task.map_seed, "index": i, "challenge_type": 8, "horizon": 390.0}
        for i, task in enumerate(tasks)
    ]
    return tasks, meta, [[i] for i in range(PATROLS)]


def _run_validator_pool(monkeypatch, tmp_path):
    """Fly the seven patrols through the validator's worker pool; return the results and the warnings it logged."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    warnings: list[str] = []
    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: None)
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: warnings.append(str(msg)))
    tasks, meta, plan = _sentinel_seeds()
    results = asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=tasks,
            task_meta=meta,
            batch_plan=plan,
            uid=46,
            model_path=model_path,
            effective_workers=PATROLS,
            on_seed_complete=None,
            phase_label="eval",
        )
    )
    return results, warnings


def _run_benchmark_pool(monkeypatch, tmp_path):
    """Fly the seven patrols through the benchmark's worker pool; return how many workers ran and the batches recorded."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")

    async def _no_calibration(worker_count):
        """Skip the host speed measurement, which needs the baseline model."""
        _ = worker_count
        return True

    monkeypatch.setattr(workers, "_precalibrate_host", _no_calibration)
    tasks, meta, plan = _sentinel_seeds()
    recorded = []
    launched = asyncio.run(
        bench_full_eval._run_benchmark_process_mode(
            all_tasks=tasks,
            task_meta=meta,
            batch_plan=plan,
            uid=46,
            model_path=model_path,
            effective_workers=PATROLS,
            record_batch_completion=lambda *args: recorded.append(args),
            on_seed_done=lambda payload=None: None,
            run_opts=bench_full_eval._RunOptions(heartbeat_sec=0.0),
        )
    )
    return launched, recorded


def test_validator_runs_at_most_three_patrols_at_once_on_sixteen_gib(monkeypatch, tmp_path, fake_host):
    """The validator's worker pool, asked for seven workers on 16 GiB, never holds more than three patrols and charges each start."""
    host = fake_host(SIXTEEN_GIB_MB, 15000.0)

    results, warnings = _run_validator_pool(monkeypatch, tmp_path)

    assert all(result is not None and result.success for result in results)
    assert host.probe.peak == 3
    assert host.started == ["type8_solar"] * PATROLS
    assert not any("Low memory" in line for line in warnings)


def test_benchmark_runs_at_most_three_patrols_at_once_on_sixteen_gib(monkeypatch, tmp_path, fake_host):
    """The benchmark's worker pool, asked for seven workers on 16 GiB, never holds more than three patrols and charges each start."""
    host = fake_host(SIXTEEN_GIB_MB, 15000.0)

    launched, recorded = _run_benchmark_pool(monkeypatch, tmp_path)

    assert launched == PATROLS
    assert len(recorded) == PATROLS
    assert host.probe.peak == 3
    assert host.started == ["type8_solar"] * PATROLS


def test_validator_short_of_memory_flies_every_patrol_one_at_a_time(monkeypatch, tmp_path, fake_host):
    """An 8 GiB validator with 5 GiB free finishes all seven patrols one by one and logs the shortage once."""
    host = fake_host(EIGHT_GIB_MB, 5000.0)

    results, warnings = _run_validator_pool(monkeypatch, tmp_path)

    assert all(result is not None and result.success for result in results)
    assert host.probe.peak == 1
    assert sum("Low memory for type8_solar" in line for line in warnings) == 1


def test_benchmark_short_of_memory_flies_every_patrol_one_at_a_time(monkeypatch, tmp_path, fake_host, capsys):
    """An 8 GiB benchmark host with 5 GiB free finishes all seven patrols one by one and prints the shortage once."""
    host = fake_host(EIGHT_GIB_MB, 5000.0)

    launched, recorded = _run_benchmark_pool(monkeypatch, tmp_path)

    assert launched == PATROLS
    assert len(recorded) == PATROLS
    assert host.probe.peak == 1
    assert capsys.readouterr().out.count("Low memory for type8_solar") == 1
