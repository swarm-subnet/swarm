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
