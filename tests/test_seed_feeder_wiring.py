from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.validator.utils_parts.evaluation import _evaluate_seeds


class _StubEvaluator:
    def __init__(self, results):
        self._results = results
        self.captured = None

    async def evaluate_seeds_parallel(self, **kwargs):
        self.captured = kwargs
        return self._results


def _stub_task():
    return SimpleNamespace(challenge_type=1, moving_platform=False)


def _stub_self(results):
    return SimpleNamespace(docker_evaluator=_StubEvaluator(results))


@pytest.mark.asyncio
async def test_feeder_indexes_map_between_absolute_and_valid_positions():
    tasks = [_stub_task(), None, _stub_task()]

    async def feeder(free_slots):
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
    tasks = [_stub_task(), _stub_task(), _stub_task()]

    async def feeder(free_slots):
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
