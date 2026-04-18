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


def test_process_normal_queue_item_happy_path(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "UID_1.zip"
    model_path.write_bytes(b"zip-bytes")

    key = "1:hash1"
    queue = {"items": {key: _queue_item(model_path)}}
    validator = _validator(epoch=11)

    cache_calls: list[tuple[str, int, dict]] = []
    marked: list[tuple[int, str]] = []

    monkeypatch.setattr(validator_utils, "sha256sum", lambda path: "hash1")
    monkeypatch.setattr(validator_utils, "has_cached_score", lambda *_: False)

    async def _register(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _run_screening(*args, **kwargs):
        _ = args, kwargs
        return 0.8, [0.8], {"city": [], "open": [0.8]}

    async def _submit_screening(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _evaluate(*args, **kwargs):
        _ = args, kwargs
        return [0.9], {"open": [0.9]}, [{"score": 0.9, "map_type": "open"}]

    async def _submit_score(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    monkeypatch.setattr(validator_utils, "_register_new_model_with_ack", _register)
    monkeypatch.setattr(validator_utils, "_run_screening", _run_screening)
    monkeypatch.setattr(validator_utils, "_submit_screening_with_ack", _submit_screening)
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _evaluate)
    monkeypatch.setattr(validator_utils, "_submit_score_with_ack", _submit_score)
    monkeypatch.setattr(
        validator_utils,
        "set_cached_score",
        lambda model_hash, epoch, result: cache_calls.append((model_hash, epoch, result)),
    )
    monkeypatch.setattr(
        validator_utils,
        "mark_model_hash_processed",
        lambda uid, model_hash: marked.append((uid, model_hash)),
    )

    asyncio.run(
        validator_utils._process_normal_queue_item(
            validator,
            queue=queue,
            key=key,
            validator_hotkey="validator_hotkey",
            validator_stake=123.0,
        )
    )

    item = queue["items"][key]
    assert item["status"] == "completed"
    assert item["registered"] is True
    assert item["screening_recorded"] is True
    assert item["score_recorded"] is True
    assert item["seeds_evaluated"] == 2
    assert item["total_score"] == pytest.approx(0.85)
    assert "screening_scores" not in item
    assert marked == [(1, "hash1")]
    assert len(cache_calls) == 1
    assert cache_calls[0][0] == "hash1"
    assert cache_calls[0][1] == 11


def test_process_normal_queue_item_updates_runtime_tracker(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "UID_1.zip"
    model_path.write_bytes(b"zip-bytes")

    key = "1:hash1"
    queue = {"items": {key: _queue_item(model_path)}}
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    validator = _validator(epoch=13)
    validator.runtime_tracker = tracker

    monkeypatch.setattr(validator_utils, "sha256sum", lambda path: "hash1")
    monkeypatch.setattr(validator_utils, "has_cached_score", lambda *_: False)

    async def _register(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _run_screening(*args, **kwargs):
        _ = args, kwargs
        return 0.8, [0.8], {"open": [0.8]}

    async def _submit_screening(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _evaluate(*args, **kwargs):
        _ = args, kwargs
        return [0.9], {"open": [0.9]}, [{"score": 0.9, "map_type": "open"}]

    async def _submit_score(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    monkeypatch.setattr(validator_utils, "_register_new_model_with_ack", _register)
    monkeypatch.setattr(validator_utils, "_run_screening", _run_screening)
    monkeypatch.setattr(validator_utils, "_submit_screening_with_ack", _submit_screening)
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _evaluate)
    monkeypatch.setattr(validator_utils, "_submit_score_with_ack", _submit_score)
    monkeypatch.setattr(validator_utils, "set_cached_score", lambda *args, **kwargs: None)
    monkeypatch.setattr(validator_utils, "mark_model_hash_processed", lambda *args, **kwargs: None)

    asyncio.run(
        validator_utils._process_normal_queue_item(
            validator,
            queue=queue,
            key=key,
            validator_hotkey="validator_hotkey",
            validator_stake=123.0,
        )
    )

    snapshot = tracker.snapshot_copy()
    events = load_recent_events(tmp_path / "validator_events.jsonl", limit=32)
    assert snapshot["queue"]["counts"]["completed"] == 1
    assert snapshot["queue"]["active_items"][0]["progress_done"] == 1
    assert snapshot["queue"]["active_items"][0]["progress_total"] == 1
    assert snapshot["counters"]["models_processed_total"] == 1
    assert snapshot["counters"]["screening_submit_success_total"] == 1
    assert snapshot["counters"]["score_submit_success_total"] == 1
    assert any(
        event["event"] == "queue_item_stage"
        and event["fields"].get("stage") == "completed"
        for event in events
    )


def test_process_normal_queue_item_register_retry(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "UID_1.zip"
    model_path.write_bytes(b"zip-bytes")

    key = "1:hash1"
    queue = {"items": {key: _queue_item(model_path)}}
    validator = _validator()

    monkeypatch.setattr(validator_utils, "sha256sum", lambda path: "hash1")
    monkeypatch.setattr(validator_utils.time, "time", lambda: 100.0)

    async def _register(*args, **kwargs):
        _ = args, kwargs
        return False, False, "backend temporary outage"

    async def _unexpected_screening(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("screening should not run")

    monkeypatch.setattr(validator_utils, "_register_new_model_with_ack", _register)
    monkeypatch.setattr(validator_utils, "_run_screening", _unexpected_screening)

    asyncio.run(
        validator_utils._process_normal_queue_item(
            validator,
            queue=queue,
            key=key,
            validator_hotkey="validator_hotkey",
            validator_stake=123.0,
        )
    )

    item = queue["items"][key]
    assert item["status"] == "retry"
    assert item["retry_attempts"] == 1
    assert item["next_retry_at"] == pytest.approx(102.0)
    assert "register failed" in item["last_error"]


def test_process_normal_queue_item_backend_cancels_benchmark_authorization(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "UID_1.zip"
    model_path.write_bytes(b"zip-bytes")

    key = "1:hash1"
    queue = {"items": {key: _queue_item(model_path)}}
    validator = _validator()

    auth_phases: list[str] = []

    async def _authorize_task(_uid, phase, **_kwargs):
        auth_phases.append(str(phase))
        if phase == "BENCHMARK":
            return {"authorized": False, "reason": "model already completed", "decision_version": 1}
        return {"authorized": True, "reason": "ok", "task_id": 1, "decision_version": 1}

    validator.backend_api.authorize_task = _authorize_task

    marked: list[tuple[int, str]] = []
    full_called = {"value": False}

    monkeypatch.setattr(validator_utils, "sha256sum", lambda path: "hash1")
    monkeypatch.setattr(validator_utils, "has_cached_score", lambda *_: False)

    async def _register(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _run_screening(*args, **kwargs):
        _ = args, kwargs
        return 0.1, [0.1], {"city": [0.1]}

    async def _submit_screening(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _unexpected_evaluate(*args, **kwargs):
        _ = args, kwargs
        full_called["value"] = True
        return [], {}, []

    monkeypatch.setattr(validator_utils, "_register_new_model_with_ack", _register)
    monkeypatch.setattr(validator_utils, "_run_screening", _run_screening)
    monkeypatch.setattr(validator_utils, "_submit_screening_with_ack", _submit_screening)
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _unexpected_evaluate)
    monkeypatch.setattr(validator_utils, "set_cached_score", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        validator_utils,
        "mark_model_hash_processed",
        lambda uid, model_hash: marked.append((uid, model_hash)),
    )

    asyncio.run(
        validator_utils._process_normal_queue_item(
            validator,
            queue=queue,
            key=key,
            validator_hotkey="validator_hotkey",
            validator_stake=123.0,
        )
    )

    item = queue["items"][key]
    assert item["status"] == "cancelled"
    assert item["screening_recorded"] is True
    assert item["score_recorded"] is False
    assert full_called["value"] is False
    assert marked == [(1, "hash1")]
    assert "BENCHMARK" in auth_phases


def test_process_normal_queue_item_terminal_score_rejection(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "UID_1.zip"
    model_path.write_bytes(b"zip-bytes")

    key = "1:hash1"
    queue = {"items": {key: _queue_item(model_path)}}
    validator = _validator()

    marked: list[tuple[int, str]] = []

    monkeypatch.setattr(validator_utils, "sha256sum", lambda path: "hash1")
    monkeypatch.setattr(validator_utils, "has_cached_score", lambda *_: False)

    async def _register(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _run_screening(*args, **kwargs):
        _ = args, kwargs
        return 0.8, [0.8], {"city": [], "open": [0.8]}

    async def _submit_screening(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _evaluate(*args, **kwargs):
        _ = args, kwargs
        return [0.9], {"open": [0.9]}, [{"score": 0.9, "map_type": "open"}]

    async def _submit_score(*args, **kwargs):
        _ = args, kwargs
        return False, True, "terminal reject"

    monkeypatch.setattr(validator_utils, "_register_new_model_with_ack", _register)
    monkeypatch.setattr(validator_utils, "_run_screening", _run_screening)
    monkeypatch.setattr(validator_utils, "_submit_screening_with_ack", _submit_screening)
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _evaluate)
    monkeypatch.setattr(validator_utils, "_submit_score_with_ack", _submit_score)
    monkeypatch.setattr(
        validator_utils,
        "set_cached_score",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        validator_utils,
        "mark_model_hash_processed",
        lambda uid, model_hash: marked.append((uid, model_hash)),
    )

    asyncio.run(
        validator_utils._process_normal_queue_item(
            validator,
            queue=queue,
            key=key,
            validator_hotkey="validator_hotkey",
            validator_stake=123.0,
        )
    )

    item = queue["items"][key]
    assert item["status"] == "terminal_rejected"
    assert item["score_recorded"] is False
    assert marked == [(1, "hash1")]


def test_process_normal_queue_item_uses_cached_scores(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "UID_1.zip"
    model_path.write_bytes(b"zip-bytes")

    key = "1:hash1"
    queue = {"items": {key: _queue_item(model_path)}}
    validator = _validator(epoch=9)

    marked: list[tuple[int, str]] = []

    monkeypatch.setattr(validator_utils, "sha256sum", lambda path: "hash1")
    monkeypatch.setattr(validator_utils, "has_cached_score", lambda *_: True)
    monkeypatch.setattr(
        validator_utils,
        "get_cached_score",
        lambda *_: {
            "screening_score": 0.8,
            "full_score": 0.9,
            "total_score": 0.85,
            "per_type_scores": {"open": 0.9},
            "seeds_evaluated": 2,
        },
    )

    async def _register(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _submit_screening(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _submit_score(*args, **kwargs):
        _ = args, kwargs
        return True, False, ""

    async def _fail_if_called(*args, **kwargs):
        _ = args, kwargs
        raise AssertionError("Live screening/benchmark should not run on cache hit")

    monkeypatch.setattr(validator_utils, "_register_new_model_with_ack", _register)
    monkeypatch.setattr(validator_utils, "_run_screening", _fail_if_called)
    monkeypatch.setattr(validator_utils, "_evaluate_seeds", _fail_if_called)
    monkeypatch.setattr(validator_utils, "_submit_screening_with_ack", _submit_screening)
    monkeypatch.setattr(validator_utils, "_submit_score_with_ack", _submit_score)
    monkeypatch.setattr(validator_utils, "set_cached_score", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        validator_utils,
        "mark_model_hash_processed",
        lambda uid, model_hash: marked.append((uid, model_hash)),
    )

    asyncio.run(
        validator_utils._process_normal_queue_item(
            validator,
            queue=queue,
            key=key,
            validator_hotkey="validator_hotkey",
            validator_stake=123.0,
        )
    )

    item = queue["items"][key]
    assert item["status"] == "completed"
    assert item["score_recorded"] is True
    assert item["total_score"] == pytest.approx(0.85)
    assert marked == [(1, "hash1")]


def test_evaluate_seeds_tracks_forest_scores(monkeypatch, tmp_path: Path):
    model_path = tmp_path / "UID_9.zip"
    model_path.write_bytes(b"zip-bytes")

    tasks = [
        SimpleNamespace(challenge_type=6, moving_platform=False),
        SimpleNamespace(challenge_type=6, moving_platform=True),
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

    monkeypatch.setattr(validator_evaluation, "random_task", _fake_random_task)

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
    assert per_type_scores["forest"] == [0.25]
    assert per_type_scores["moving_platform"] == [0.75]


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
