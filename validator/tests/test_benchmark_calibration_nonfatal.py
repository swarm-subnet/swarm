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

"""A local benchmark reports scores even when the host cannot be calibrated.

Nothing is at stake in a local run, so an unmeasurable host still gets its results;
only the comparison against the validator's budget is withheld.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.benchmark import engine
from swarm.benchmark.engine_parts import workers
from swarm.core.faults import ReasonCode
from swarm.protocol import ValidationResult
from swarm.validator.calibration import normalize_speed_factor
from swarm.validator.docker.docker_evaluator_parts import batch

TOO_SLOW_P90_MS = 260.8 * 10


@pytest.fixture(autouse=True)
def _reset_flag():
    """Pin workers.host_timings_normalized True around every test so one run cannot leak into the next."""
    workers.host_timings_normalized = True
    yield
    workers.host_timings_normalized = True


def _run(coro):
    """Drive a coroutine to completion on a fresh event loop."""
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


def _patrol_tasks(count: int = 1) -> list:
    """Swarm Sentinel task stand-ins carrying only what dispatch and the container key read."""
    return [SimpleNamespace(map_seed=7000 + i, challenge_type=8, horizon=390.0) for i in range(count)]


@pytest.fixture
def flight(monkeypatch):
    """Stub every docker step of a batch and leave the host unmeasured, so only the calibration gate decides."""

    async def _container_ready(ctx):
        """Report the container as started without touching docker."""
        return None

    async def _fly(ctx):
        """Fly every seed to a passing score and keep the speed factor the flight was judged at."""
        return [
            ValidationResult(ctx.uid, True, 12.0, 0.9, metrics={"speed_factor": ctx.speed_factor})
            for _ in ctx.tasks
        ]

    monkeypatch.setattr(batch, "_validate_inputs", lambda ctx: None)
    monkeypatch.setattr(batch, "baseline_model_available", lambda: True)
    monkeypatch.setattr(batch, "host_speed_factor_is_fresh", lambda n: False)
    monkeypatch.setattr(batch, "_setup_pretry_state", lambda ctx: None)
    monkeypatch.setattr(batch, "_prepare_container", _container_ready)
    monkeypatch.setattr(batch, "_run_rpc_phase", _fly)
    monkeypatch.setenv("SWARM_DOCKER_PREWARM", "0")


@pytest.fixture
def measurements(monkeypatch):
    """Replace the host measurement with one that always fails, and return the list of attempts it saw."""
    attempts: list[int] = []

    async def _measure(self, worker_count):
        """Record the attempt and come back with no speed factor."""
        attempts.append(int(worker_count))
        return None

    monkeypatch.setattr(batch, "_run_host_baseline_calibration", _measure)
    return attempts


def _evaluate(tmp_path: Path, **kwargs) -> list:
    """Run one patrol batch through the real evaluator entry point with the flight stubbed."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    return _run(
        batch.evaluate_seeds_batch(
            workers._create_prepared_benchmark_evaluator(),
            tasks=_patrol_tasks(),
            uid=61,
            model_path=model_path,
            **kwargs,
        )
    )


def test_failed_calibration_does_not_abort_the_run(monkeypatch):
    """A host that cannot be timed still benchmarks: precalibration reports False instead of raising."""
    monkeypatch.setattr(workers, "baseline_model_available", lambda: True)
    monkeypatch.setattr(workers, "host_speed_factor_is_fresh", lambda n: False)
    monkeypatch.setattr(workers, "_create_prepared_benchmark_evaluator", lambda: object())

    async def _no_speed(evaluator, count):
        """Stand in for a host speed measurement that came back with nothing."""
        return None

    monkeypatch.setattr(workers, "_ensure_host_speed_factor", _no_speed)

    normalized = _run(workers._precalibrate_host(8))

    assert normalized is False
    assert workers.host_timings_normalized is False


def test_successful_calibration_marks_timings_normalized(monkeypatch):
    """A speed factor that measures cleanly sets the module flag the rest of the benchmark reads."""
    monkeypatch.setattr(workers, "baseline_model_available", lambda: True)
    monkeypatch.setattr(workers, "host_speed_factor_is_fresh", lambda n: False)
    monkeypatch.setattr(workers, "_create_prepared_benchmark_evaluator", lambda: object())

    async def _speed(evaluator, count):
        """Stand in for a host measurement that produced a usable factor."""
        return normalize_speed_factor(200.0)

    monkeypatch.setattr(workers, "_ensure_host_speed_factor", _speed)

    assert _run(workers._precalibrate_host(8)) is True
    assert workers.host_timings_normalized is True


def test_a_fresh_cache_skips_calibration_entirely(monkeypatch):
    """A cached speed factor still in date short-circuits: the measurement is never awaited."""
    monkeypatch.setattr(workers, "baseline_model_available", lambda: True)
    monkeypatch.setattr(workers, "host_speed_factor_is_fresh", lambda n: True)

    async def _boom(evaluator, count):
        """Fail loudly if a measurement is attempted while the cached factor is still valid."""
        raise AssertionError("calibration must not run when the cache is fresh")

    monkeypatch.setattr(workers, "_ensure_host_speed_factor", _boom)

    assert _run(workers._precalibrate_host(4)) is True
    assert workers.host_timings_normalized is True


def test_a_host_too_slow_to_score_flies_raw(monkeypatch):
    """A host measured past the eligibility limit is benchmarked on raw act times instead of scoring zero."""
    monkeypatch.setattr(workers, "baseline_model_available", lambda: True)
    monkeypatch.setattr(workers, "host_speed_factor_is_fresh", lambda n: False)
    monkeypatch.setattr(workers, "_create_prepared_benchmark_evaluator", lambda: object())

    async def _too_slow(evaluator, count):
        """Stand in for a measurement that found the host slower than any validator may be."""
        return normalize_speed_factor(TOO_SLOW_P90_MS)

    monkeypatch.setattr(workers, "_ensure_host_speed_factor", _too_slow)

    assert _run(workers._precalibrate_host(4)) is False
    assert workers.host_timings_normalized is False


def test_a_missing_baseline_model_flies_raw(monkeypatch):
    """Without the reference model there is nothing to measure against, so timings are reported raw."""
    monkeypatch.setattr(workers, "baseline_model_available", lambda: False)

    assert _run(workers._precalibrate_host(4)) is False
    assert workers.host_timings_normalized is False


def test_a_raw_batch_flies_without_measuring_the_host(tmp_path, flight, measurements):
    """A batch marked raw never measures the host and flies on the per-step timeout, with no speed factor."""
    results = _evaluate(tmp_path, raw_timing=True)

    assert measurements == []
    assert [(r.success, r.score) for r in results] == [(True, 0.9)]
    assert results[0].metrics["speed_factor"] is None


def test_the_validator_still_refuses_to_score_an_unmeasured_host(tmp_path, flight, measurements):
    """Without the raw flag an unmeasurable host is excluded from scoring, as the validator requires."""
    results = _evaluate(tmp_path)

    assert measurements == [1]
    assert [(r.success, r.failure_reason) for r in results] == [
        (False, ReasonCode.INFRA_CALIBRATION.value)
    ]


@pytest.mark.skipif(sys.platform == "win32", reason="daemon workers are forked")
def test_a_failed_speed_check_still_scores_every_seed_in_forked_workers(monkeypatch, tmp_path, flight):
    """Real daemon worker processes fly every seed after a failed speed check instead of crashing on a re-measure."""

    async def _failed_check(worker_count):
        """Stand in for a speed check that came back with nothing."""
        return False

    monkeypatch.setattr(workers, "_precalibrate_host", _failed_check)
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    tasks = _patrol_tasks(2)
    meta = [
        {"group": "type8_solar", "seed": task.map_seed, "challenge_type": 8, "horizon": 390.0}
        for task in tasks
    ]
    recorded = []

    _run(
        engine._run_benchmark_process_mode(
            all_tasks=tasks,
            task_meta=meta,
            batch_plan=[[0], [1]],
            uid=61,
            model_path=model_path,
            effective_workers=2,
            record_batch_completion=lambda *args: recorded.append(args),
            on_seed_done=lambda payload=None: None,
            run_opts=engine._RunOptions(heartbeat_sec=0.0),
        )
    )

    results = [result for _slot, _batch, _indices, batch_results, _sec in recorded for result in batch_results]
    assert [(r.success, r.score) for r in results] == [(True, 0.9), (True, 0.9)]
    assert all(r.metrics["speed_factor"] is None for r in results)
