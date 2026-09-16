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

import pytest

from swarm.benchmark.engine_parts import workers


@pytest.fixture(autouse=True)
def _reset_flag():
    """Pin workers.host_timings_normalized True around every test so one run cannot leak into the next."""
    workers.host_timings_normalized = True
    yield
    workers.host_timings_normalized = True


def _run(coro):
    """Drive a coroutine to completion on a fresh event loop."""
    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(coro)


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
        """Stand in for a host measurement that produced a usable factor of 1.0."""
        return 1.0

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
