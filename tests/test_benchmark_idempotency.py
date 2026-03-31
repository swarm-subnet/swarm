from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.benchmark import idempotency


def test_summarize_idempotency_runs_marks_identical_scores_as_idempotent():
    summary = idempotency.summarize_idempotency_runs(
        [
            {"run": 1, "score": 0.42, "success": False, "time_sec": 1.2, "wall_time_sec": 2.3},
            {"run": 2, "score": 0.42, "success": False, "time_sec": 1.2, "wall_time_sec": 2.3},
            {"run": 3, "score": 0.42, "success": False, "time_sec": 1.2, "wall_time_sec": 2.3},
        ]
    )

    assert summary["idempotent_score"] is True
    assert summary["idempotent_success"] is True
    assert summary["idempotent_sim_time"] is True
    assert summary["strict_idempotent"] is True


def test_summarize_idempotency_runs_detects_mismatch():
    summary = idempotency.summarize_idempotency_runs(
        [
            {"run": 1, "score": 0.42, "success": False, "time_sec": 1.2, "wall_time_sec": 2.3},
            {"run": 2, "score": 0.41, "success": True, "time_sec": 1.3, "wall_time_sec": 2.8},
        ]
    )

    assert summary["idempotent_score"] is False
    assert summary["idempotent_success"] is False
    assert summary["idempotent_sim_time"] is False
    assert summary["strict_idempotent"] is False


def test_run_idempotency_rebuilds_the_same_task_each_time(monkeypatch):
    built_tasks: list[SimpleNamespace] = []
    evaluator_calls: list[tuple[list[SimpleNamespace], int, Path, int]] = []

    def _fake_task_for_seed_and_type(sim_dt, *, seed, challenge_type, moving_platform=None):
        task = SimpleNamespace(
            sim_dt=sim_dt,
            map_seed=seed,
            challenge_type=challenge_type,
            moving_platform=moving_platform,
            build_index=len(built_tasks) + 1,
        )
        built_tasks.append(task)
        return task

    class _FakeEvaluator:
        _base_ready = True

        async def evaluate_seeds_batch(self, *, tasks, uid, model_path, worker_id):
            evaluator_calls.append((list(tasks), uid, Path(model_path), worker_id))
            return [SimpleNamespace(success=False, time_sec=0.68, score=0.01)]

    monkeypatch.setattr(idempotency, "task_for_seed_and_type", _fake_task_for_seed_and_type)
    monkeypatch.setattr(idempotency, "DockerSecureEvaluator", _FakeEvaluator)

    summary = idempotency.run_idempotency(
        model_path=Path("model/UID_178.zip"),
        uid=178,
        seed=101678,
        challenge_type=6,
        runs=3,
        moving_platform=False,
        worker_id=7,
    )

    assert len(built_tasks) == 3
    assert [task.build_index for task in built_tasks] == [1, 2, 3]
    assert all(task.map_seed == 101678 for task in built_tasks)
    assert all(task.challenge_type == 6 for task in built_tasks)
    assert all(task.moving_platform is False for task in built_tasks)

    assert len(evaluator_calls) == 3
    assert all(len(tasks) == 1 for tasks, *_ in evaluator_calls)
    assert [tasks[0].build_index for tasks, *_ in evaluator_calls] == [1, 2, 3]
    assert all(uid == 178 for _, uid, _, _ in evaluator_calls)
    assert all(model_path == Path("model/UID_178.zip") for _, _, model_path, _ in evaluator_calls)
    assert all(worker_id == 7 for _, _, _, worker_id in evaluator_calls)

    assert summary["idempotent_score"] is True
    assert summary["runs_requested"] == 3
    assert summary["challenge_type"] == 6
    assert len(summary["unique_wall_times"]) == 3


def test_run_idempotency_requires_positive_runs():
    with pytest.raises(ValueError, match="runs must be positive"):
        idempotency.run_idempotency(
            model_path=Path("model/UID_178.zip"),
            uid=178,
            seed=101678,
            challenge_type=6,
            runs=0,
        )
