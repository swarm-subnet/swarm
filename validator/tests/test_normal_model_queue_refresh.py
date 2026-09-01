from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from swarm.validator import utils as validator_utils
from swarm.validator.utils_parts import state as state_mod


@pytest.fixture
def queue_env(tmp_path: Path, monkeypatch):
    queue_file = tmp_path / "normal_model_queue.json"
    monkeypatch.setattr(state_mod, "STATE_DIR", tmp_path)
    monkeypatch.setattr(state_mod, "NORMAL_MODEL_QUEUE_FILE", queue_file)
    monkeypatch.setattr(validator_utils, "STATE_DIR", tmp_path)
    monkeypatch.setattr(validator_utils, "NORMAL_MODEL_QUEUE_FILE", queue_file)
    return queue_file


def _seed_queue(queue_file: Path, items: dict) -> None:
    queue_file.write_text(json.dumps({"items": items}))


def _key(uid: int, model_hash: str) -> str:
    return f"{uid}:{model_hash}"


def test_cancelled_item_reactivated_when_backend_reauthorizes(queue_env, tmp_path):
    key = _key(43, "abc123")
    _seed_queue(queue_env, {
        key: {
            "uid": 43,
            "model_hash": "abc123",
            "model_path": "old/path.zip",
            "github_url": "https://example.com/old",
            "status": "cancelled",
            "last_error": "backend authorization failed: epoch5_relaunch_cleanup",
            "backend_authorized": False,
            "backend_reason": "epoch5_relaunch_cleanup",
            "screening_recorded": False,
            "score_recorded": False,
            "retry_attempts": 5,
            "next_retry_at": time.time() + 3600,
            "created_at": 1000.0,
            "updated_at": 1500.0,
        }
    })

    new_path = tmp_path / "UID_43.zip"
    queue = state_mod._refresh_normal_model_queue({
        43: (new_path, "abc123", "https://example.com/new"),
    })

    item = queue["items"][key]
    assert item["status"] == "pending"
    assert item["last_error"] == ""
    assert item["retry_attempts"] == 0
    assert item["next_retry_at"] == 0
    assert item["backend_authorized"] is True
    assert item["backend_reason"] == ""
    assert item["created_at"] == 1000.0
    assert item["model_path"] == str(new_path)
    assert item["github_url"] == "https://example.com/new"

    processable = state_mod._get_processable_queue_keys(queue, 100)
    assert processable == [key]


def test_cancelled_item_preserves_screening_recorded_when_reactivated(queue_env, tmp_path):
    key = _key(118, "fa84")
    _seed_queue(queue_env, {
        key: {
            "uid": 118,
            "model_hash": "fa84",
            "model_path": "old.zip",
            "github_url": "https://example.com/old",
            "status": "cancelled",
            "screening_recorded": True,
            "screening_score": 0.65,
            "score_recorded": False,
            "retry_attempts": 0,
            "next_retry_at": 0,
            "last_error": "backend cancelled at benchmark",
            "created_at": 2000.0,
            "updated_at": 2500.0,
        }
    })

    new_path = tmp_path / "UID_118.zip"
    queue = state_mod._refresh_normal_model_queue({
        118: (new_path, "fa84", "https://example.com/new"),
    })

    item = queue["items"][key]
    assert item["status"] == "pending"
    assert item["screening_recorded"] is True
    assert item["screening_score"] == 0.65
    assert item["score_recorded"] is False


def test_terminal_rejected_item_not_reactivated(queue_env, tmp_path):
    key = _key(55, "deadbeef")
    _seed_queue(queue_env, {
        key: {
            "uid": 55,
            "model_hash": "deadbeef",
            "status": "terminal_rejected",
            "last_error": "permanent reject",
            "screening_recorded": False,
            "score_recorded": False,
            "retry_attempts": 0,
            "next_retry_at": 0,
            "created_at": 3000.0,
            "updated_at": 3500.0,
        }
    })

    new_path = tmp_path / "UID_55.zip"
    queue = state_mod._refresh_normal_model_queue({
        55: (new_path, "deadbeef", "https://example.com/repo"),
    })

    item = queue["items"][key]
    assert item["status"] == "terminal_rejected"
    assert item["last_error"] == "permanent reject"
    assert state_mod._get_processable_queue_keys(queue, 100) == []


def test_completed_item_not_reactivated(queue_env, tmp_path):
    key = _key(77, "feedface")
    _seed_queue(queue_env, {
        key: {
            "uid": 77,
            "model_hash": "feedface",
            "status": "completed",
            "screening_recorded": True,
            "score_recorded": True,
            "retry_attempts": 0,
            "next_retry_at": 0,
            "last_error": "",
            "created_at": 4000.0,
            "updated_at": 4500.0,
        }
    })

    new_path = tmp_path / "UID_77.zip"
    queue = state_mod._refresh_normal_model_queue({
        77: (new_path, "feedface", "https://example.com/repo"),
    })

    item = queue["items"][key]
    assert item["status"] == "completed"
    assert state_mod._get_processable_queue_keys(queue, 100) == []


def test_pending_item_unchanged_except_path_and_url(queue_env, tmp_path):
    key = _key(11, "hash11")
    _seed_queue(queue_env, {
        key: {
            "uid": 11,
            "model_hash": "hash11",
            "model_path": "stale.zip",
            "github_url": "https://example.com/stale",
            "status": "pending",
            "screening_recorded": False,
            "score_recorded": False,
            "retry_attempts": 2,
            "next_retry_at": 0,
            "last_error": "",
            "created_at": 5000.0,
            "updated_at": 5500.0,
        }
    })

    new_path = tmp_path / "UID_11.zip"
    queue = state_mod._refresh_normal_model_queue({
        11: (new_path, "hash11", "https://example.com/fresh"),
    })

    item = queue["items"][key]
    assert item["status"] == "pending"
    assert item["retry_attempts"] == 2
    assert item["model_path"] == str(new_path)
    assert item["github_url"] == "https://example.com/fresh"
    assert item["created_at"] == 5000.0


def test_fresh_uid_creates_new_pending_entry(queue_env, tmp_path):
    _seed_queue(queue_env, {})

    new_path = tmp_path / "UID_99.zip"
    queue = state_mod._refresh_normal_model_queue({
        99: (new_path, "newhash", "https://example.com/fresh"),
    })

    key = _key(99, "newhash")
    item = queue["items"][key]
    assert item["status"] == "pending"
    assert item["from_backend"] is True
    assert item["registered"] is False
    assert item["screening_recorded"] is False
    assert item["score_recorded"] is False
    assert item["retry_attempts"] == 0
    assert item["last_error"] == ""
    assert state_mod._get_processable_queue_keys(queue, 100) == [key]


def test_stale_hash_for_same_uid_removed_unless_terminal_rejected(queue_env, tmp_path):
    old_key = _key(20, "old_hash")
    other_key = _key(20, "rejected_hash")
    _seed_queue(queue_env, {
        old_key: {
            "uid": 20,
            "model_hash": "old_hash",
            "status": "pending",
            "created_at": 100.0,
            "updated_at": 100.0,
        },
        other_key: {
            "uid": 20,
            "model_hash": "rejected_hash",
            "status": "terminal_rejected",
            "created_at": 50.0,
            "updated_at": 50.0,
        },
    })

    new_path = tmp_path / "UID_20.zip"
    queue = state_mod._refresh_normal_model_queue({
        20: (new_path, "fresh_hash", "https://example.com/repo"),
    })

    assert old_key not in queue["items"]
    assert other_key in queue["items"]
    new_key = _key(20, "fresh_hash")
    assert new_key in queue["items"]
    assert queue["items"][new_key]["status"] == "pending"
