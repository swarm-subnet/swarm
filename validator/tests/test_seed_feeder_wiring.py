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

"""Wiring between _evaluate_seeds and the container evaluator: seed indices translated both ways, and what a drained pool means."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.validator.utils_parts.evaluation import _evaluate_seeds
from swarm.validator.utils_parts.run_task import _repeat_handouts


class _StubEvaluator:
    """An evaluator that records the keyword arguments it was handed and answers with canned per-seed results."""
    def __init__(self, results):
        """Store the canned results and start with nothing captured."""
        self._results = results
        self.captured = None

    async def evaluate_seeds_parallel(self, **kwargs):
        """Capture the call's keyword arguments for inspection and hand back the canned results."""
        self.captured = kwargs
        return self._results


def _stub_task():
    """A task stand-in carrying only the challenge type and moving-platform flag the caller reads."""
    return SimpleNamespace(challenge_type=1, moving_platform=False)


def _stub_self(results):
    """A validator stand-in whose docker_evaluator is the recording stub."""
    return SimpleNamespace(docker_evaluator=_StubEvaluator(results))


@pytest.mark.asyncio
async def test_feeder_indexes_map_between_absolute_and_valid_positions():
    """A seed with no built task shifts the numbering, so pending indices and feeder grants both reach the evaluator renumbered over the built tasks."""
    tasks = [_stub_task(), None, _stub_task()]

    async def feeder(free_slots):
        """Grant the absolute seed indices 0 and 2, and report the pool drained."""
        return [0, 2], True

    stub = _stub_self([None, None])
    await _evaluate_seeds(
        stub,
        uid=1,
        model_path=Path("/nonexistent"),
        seeds=[111, 222, 333],
        pre_built_tasks=tasks,
        seed_feeder=feeder,
        initial_pending=[2],
    )
    captured = stub.docker_evaluator.captured
    assert captured["initial_pending"] == [1]
    granted, drained = await captured["seed_feeder"](4)
    assert granted == [0, 1]
    assert drained is True


@pytest.mark.asyncio
async def test_sparse_feeder_results_score_nothing():
    """A feeder that leases nothing leaves no scores and no seed details, rather than scoring seeds that never flew."""
    tasks = [_stub_task(), _stub_task(), _stub_task()]

    async def feeder(free_slots):
        """Lease no seeds at all and report the pool drained."""
        return [], True

    stub = _stub_self([None, None, None])
    all_scores, _per_type, details = await _evaluate_seeds(
        stub,
        uid=1,
        model_path=Path("/nonexistent"),
        seeds=[111, 222, 333],
        pre_built_tasks=tasks,
        seed_feeder=feeder,
        initial_pending=[],
    )
    assert all_scores == []
    assert details == []


def test_pool_drained_ignores_seeds_flying_elsewhere():
    """Drained means nothing granted and nothing pending here, whatever other validators are still flying."""
    from swarm.validator.utils_parts.run_task import _pool_drained

    assert _pool_drained([], 0) is True
    assert _pool_drained([], 5) is False
    assert _pool_drained([7], 0) is False


def test_only_seeds_handed_out_before_are_reported():
    """A first hand-out is ordinary work; only a repeat is named, with JSON's string keys read back as indexes."""
    reply = {"granted": [3, 40], "handouts": {"3": 1, "40": 4}, "max_handouts": 6}

    assert _repeat_handouts(reply) == {40: 4}


def test_a_backend_that_sends_no_hand_out_numbers_reports_nothing():
    """An older backend omits the field, and the claim loop must carry on unchanged."""
    assert _repeat_handouts({"granted": [3]}) == {}
