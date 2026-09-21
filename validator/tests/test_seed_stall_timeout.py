"""The stall clock: a seed that stops reporting progress is cut, one that keeps reporting flies on."""

from __future__ import annotations

import queue
import time

import pytest

from swarm.benchmark.engine_parts import workers
from swarm.benchmark.engine_parts._shared import _ProcessSeedEvent
from swarm.constants import (
    CALIBRATION_TIMEOUT_SEC,
    FIRST_STEP_HARD_CAP_REF_SEC,
    GLOBAL_EVAL_BASE_SEC,
    HARD_CAP_REF_SEC,
    RPC_CONNECT_MAX_WAIT_SEC,
    RPC_RESET_TIMEOUT_SEC,
    SEED_STALL_TIMEOUT_SEC,
    SPEED_FACTOR_MAX_ELIGIBLE,
)
from swarm.protocol import ValidationResult
from swarm.validator.docker import docker_evaluator as de
from swarm.validator.docker.docker_evaluator_parts import batch
from validator.tests.test_container_prewarm import _request
from validator.tests.test_container_prewarm import fake_docker as _shared_fake_docker  # noqa: F401

_STALL_SEC = 0.5


@pytest.fixture
def fake(request):
    """The shared fake daemon under a local name."""
    return request.getfixturevalue("_shared_fake_docker")


def _stamp(progress_state: dict, step: int) -> None:
    """Report one more step, the way the flight loop does."""
    progress_state["phase"] = "rpc_act"
    progress_state["step_idx"] = step
    progress_state["ts"] = time.time()


def _fly_until_stopped(stamp_every: float | None, first_stamp_after: float = 0.0):
    """A flight that only ends when the batch stops it, reporting progress every ``stamp_every`` seconds or never."""

    def fly(self, tasks, uid, rpc_port, on_seed_complete, observer, stop_event, progress_state, *args):
        """Wait on the stop event, stamping progress on the given cadence."""
        _ = self, rpc_port, on_seed_complete, observer, args
        started = time.monotonic()
        step = 0
        while not stop_event.wait(0.05):
            if stamp_every is not None and time.monotonic() - started >= first_stamp_after:
                step += 1
                _stamp(progress_state, step)
                stop_event.wait(stamp_every)
            if time.monotonic() - started > 10.0:
                break
        return [ValidationResult(uid, False, 0.0, 0.0) for _ in tasks]

    return fly


def _run(monkeypatch, fake, fly, stall_sec: float = _STALL_SEC):
    """Fly one seed through the worker body with the given flight and stall clock; return elapsed, score, statuses."""
    monkeypatch.setattr(de.DockerSecureEvaluator, "_run_multi_seed_rpc_sync", fly)
    monkeypatch.setattr(batch, "SEED_STALL_TIMEOUT_SEC", stall_sec)
    task_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    progress_queue: queue.Queue = queue.Queue()
    task_queue.put(_request(0, 7, fake.model_path, prewarm_next=False))
    task_queue.put(None)
    started = time.monotonic()
    workers._benchmark_worker_main(0, task_queue, result_queue, progress_queue)
    elapsed = time.monotonic() - started
    result = result_queue.get_nowait()
    statuses = []
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            break
        if isinstance(event, _ProcessSeedEvent) and isinstance(event.seed_meta, dict):
            statuses.append(event.seed_meta.get("status"))
    return elapsed, result, statuses


def test_a_seed_with_no_progress_is_cut_by_the_stall_clock(fake, monkeypatch):
    """A flight that never reports progress ends at the stall limit, long before the batch limit."""
    elapsed, result, statuses = _run(monkeypatch, fake, _fly_until_stopped(stamp_every=None))

    assert statuses == ["batch_timeout_partial"]
    assert result.results[0][3] == 0.0
    assert elapsed < GLOBAL_EVAL_BASE_SEC
    assert elapsed < _STALL_SEC + 4.0


def test_a_slow_seed_that_keeps_reporting_is_not_cut(fake, monkeypatch):
    """A flight that stamps progress more often than the stall limit runs to its own end and keeps its score."""

    def fly(self, tasks, uid, rpc_port, on_seed_complete, observer, stop_event, progress_state, *args):
        """Report a step every tenth of a second for three stall limits, then land."""
        _ = self, rpc_port, on_seed_complete, observer, args
        stop_event.wait(0.3)
        for step in range(1, int(3 * _STALL_SEC / 0.1) + 1):
            if stop_event.is_set():
                break
            _stamp(progress_state, step)
            time.sleep(0.1)
        return [ValidationResult(uid, True, 1.0, 0.5) for _ in tasks]

    elapsed, result, statuses = _run(monkeypatch, fake, fly)

    assert statuses == []
    assert result.results[0][3] == 0.5
    assert elapsed >= 3 * _STALL_SEC


def test_the_batch_limit_still_cuts_a_seed_that_keeps_reporting(fake, monkeypatch):
    """Progress keeps the stall clock quiet but never lifts the batch limit."""
    monkeypatch.setenv("SWARM_BATCH_TIMEOUT_HARD_CAP_SEC", "0.4")

    elapsed, result, statuses = _run(monkeypatch, fake, _fly_until_stopped(stamp_every=0.05))

    assert statuses == ["batch_timeout_partial"]
    assert result.results[0][3] == 0.0
    assert elapsed < 4.0


def test_the_stall_limit_covers_every_bounded_wait_inside_a_seed():
    """The stall limit sits above the longest wait a healthy seed may make between two progress stamps."""
    longest_quiet_wait = max(
        RPC_CONNECT_MAX_WAIT_SEC,
        RPC_RESET_TIMEOUT_SEC,
        CALIBRATION_TIMEOUT_SEC,
        FIRST_STEP_HARD_CAP_REF_SEC * SPEED_FACTOR_MAX_ELIGIBLE,
        HARD_CAP_REF_SEC * SPEED_FACTOR_MAX_ELIGIBLE,
    )
    assert SEED_STALL_TIMEOUT_SEC > longest_quiet_wait
    assert SEED_STALL_TIMEOUT_SEC < GLOBAL_EVAL_BASE_SEC
