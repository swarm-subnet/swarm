from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.protocol import ValidationResult
from swarm.validator import utils as validator_utils
from swarm.validator.runtime_telemetry import (
    ValidatorRuntimeTracker,
    load_recent_events,
)
from swarm.validator.utils_parts import evaluation as validator_evaluation


def _queue_item(model_path: Path, model_hash: str = "hash1", uid: int = 1) -> dict:
    return {
        "uid": uid,
        "model_hash": model_hash,
        "model_path": str(model_path),
        "status": "pending",
        "registered": False,
        "screening_recorded": False,
        "score_recorded": False,
        "retry_attempts": 0,
        "next_retry_at": 0,
        "last_error": "",
        "created_at": 0,
        "updated_at": 0,
    }


def _validator(epoch: int = 7) -> SimpleNamespace:
    async def _post_heartbeat(**kwargs):
        _ = kwargs
        return {"ok": True}

    async def _post_seed_scores_batch(**kwargs):
        _ = kwargs
        return {"recorded": True}

    async def _authorize_task(*args, **kwargs):
        _ = args, kwargs
        return {
            "authorized": True,
            "reason": "ok",
            "task_id": 1,
            "decision_version": 1,
        }

    return SimpleNamespace(
        seed_manager=SimpleNamespace(
            epoch_number=epoch,
            get_benchmark_seeds=lambda: [700001],
        ),
        backend_api=SimpleNamespace(
            post_heartbeat=_post_heartbeat,
            post_seed_scores_batch=_post_seed_scores_batch,
            authorize_task=_authorize_task,
        ),
        metagraph=SimpleNamespace(hotkeys=["hotkey0", "hotkey1", "hotkey2"]),
    )


def test_evaluate_seeds_tracks_forest_scores(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "UID_9.zip"
    model_path.write_bytes(b"zip-bytes")

    tasks = [
        SimpleNamespace(challenge_type=6),
        SimpleNamespace(challenge_type=6),
    ]

    def _fake_random_task(*args, **kwargs):
        _ = args, kwargs
        return tasks.pop(0)

    async def _fake_parallel(*args, **kwargs):
        _ = args, kwargs
        return [
            ValidationResult(9, False, 1.0, 0.25),
            ValidationResult(9, True, 2.0, 0.75),
        ]

    validator = SimpleNamespace(
        docker_evaluator=SimpleNamespace(evaluate_seeds_parallel=_fake_parallel),
    )

    monkeypatch.setattr(validator_evaluation, "build_random_task", _fake_random_task)

    all_scores, per_type_scores, _details = asyncio.run(
        validator_utils._evaluate_seeds(
            validator,
            uid=9,
            model_path=model_path,
            seeds=[600001, 600002],
            description="forest check",
        )
    )

    assert all_scores == [0.25, 0.75]
    assert per_type_scores["forest"] == [0.25, 0.75]
    assert per_type_scores["moving_platform"] == []


def test_build_heartbeat_queue_snapshot_contains_full_queue_metadata(monkeypatch):
    monkeypatch.setattr(validator_utils.time, "time", lambda: 100.0)

    queue = {
        "items": {
            "11:hash11": {
                "uid": 11,
                "model_hash": "hash11",
                "status": "pending",
                "created_at": 10.0,
                "updated_at": 15.0,
                "retry_attempts": 0,
                "next_retry_at": 0,
                "assignment_id": 3,
                "backend_authorized": True,
                "backend_decision_version": 7,
            },
            "12:hash12": {
                "uid": 12,
                "model_hash": "hash12",
                "status": "retry",
                "created_at": 20.0,
                "updated_at": 25.0,
                "retry_attempts": 2,
                "next_retry_at": 150.0,
                "last_error": "waiting for lease",
                "screening_recorded": True,
            },
            "13:hash13": {
                "uid": 13,
                "model_hash": "hash13",
                "status": "cancelled",
                "created_at": 30.0,
                "updated_at": 35.0,
                "retry_attempts": 0,
                "next_retry_at": 0,
                "last_error": "backend authorization failed",
            },
        }
    }

    snapshot = validator_utils.build_heartbeat_queue_snapshot(queue)

    assert [entry["uid"] for entry in snapshot] == [11, 12, 13]
    assert snapshot[0]["assignment_id"] == 3
    assert snapshot[0]["processable"] is True
    assert snapshot[1]["phase"] == "benchmark"
    assert snapshot[1]["processable"] is False
    assert snapshot[1]["blocked_reason"] == "waiting for lease"
    assert snapshot[2]["status"] == "cancelled"
    assert snapshot[2]["blocked_reason"] == "backend authorization failed"
