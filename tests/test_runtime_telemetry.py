from __future__ import annotations

from pathlib import Path

from swarm.validator.runtime_telemetry import (
    ValidatorRuntimeTracker,
    compute_alerts,
    load_recent_events,
    load_runtime_snapshot,
)


def test_runtime_tracker_writes_snapshot_and_events(tmp_path: Path) -> None:
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)

    tracker.mark_worker_thread_alive(True)
    tracker.mark_forward_started(3)
    tracker.mark_backend_sync_started()
    tracker.mark_backend_sync_completed(
        fallback=False,
        pending_models_count=4,
        reeval_queue_count=2,
        leaderboard_version=9,
    )
    tracker.mark_forward_completed(3)
    tracker.flush()

    snapshot = load_runtime_snapshot(tmp_path / "validator_runtime.json")
    events = load_recent_events(tmp_path / "validator_events.jsonl", limit=10)

    assert snapshot["forward"]["count"] == 3
    assert snapshot["forward"]["last_completed_forward_count"] == 3
    assert snapshot["backend"]["pending_models_count"] == 4
    assert snapshot["backend"]["reeval_queue_count"] == 2
    assert snapshot["process"]["worker_thread_alive"] is True
    assert [event["event"] for event in events][-2:] == [
        "backend_sync_completed",
        "forward_completed",
    ]


def test_runtime_tracker_summarizes_queue_stage_and_progress(tmp_path: Path) -> None:
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    queue = {
        "items": {
            "1:hash1": {
                "uid": 1,
                "model_hash": "hash1",
                "status": "processing",
                "retry_attempts": 2,
                "next_retry_at": 0,
                "created_at": 10.0,
                "updated_at": 20.0,
                "last_error": "",
                "seeds_evaluated": 150,
            }
        }
    }

    tracker.mark_queue_item_stage(
        queue=queue,
        key="1:hash1",
        item=queue["items"]["1:hash1"],
        stage="benchmark",
        progress_done=150,
        progress_total=800,
        note="chunk 150/800",
    )
    tracker.flush()

    snapshot = load_runtime_snapshot(tmp_path / "validator_runtime.json")
    item = snapshot["queue"]["active_items"][0]
    assert snapshot["queue"]["counts"]["processing"] == 1
    assert snapshot["queue"]["processable_count"] == 1
    assert item["stage"] == "benchmark"
    assert item["progress_done"] == 150
    assert item["progress_total"] == 800
    assert item["note"] == "chunk 150/800"


def test_compute_alerts_flags_fallback_freeze_and_stalls(tmp_path: Path) -> None:
    snapshot = ValidatorRuntimeTracker(state_dir=tmp_path).snapshot_copy()
    now = 10_000.0
    snapshot["process"]["worker_thread_alive"] = False
    snapshot["forward"]["count"] = 2
    snapshot["forward"]["in_progress"] = True
    snapshot["forward"]["last_started_at"] = now - 301
    snapshot["backend"]["fallback"] = True
    snapshot["backend"]["last_sync_success_at"] = now - 301
    snapshot["queue"]["oldest_age_sec"] = 901.0
    snapshot["queue"]["processable_count"] = 1
    snapshot["queue"]["max_retry_attempts"] = 6
    snapshot["epoch"]["freeze_active"] = True
    snapshot["chain_sync"]["in_progress"] = True
    snapshot["chain_sync"]["last_started_at"] = now - 181

    alerts = compute_alerts(snapshot, now=now)
    codes = {alert["code"] for alert in alerts}

    assert "worker_thread_dead" in codes
    assert "forward_stalled" in codes
    assert "backend_fallback" in codes
    assert "queue_oldest_warning" in codes
    assert "queue_retry_warning" in codes
    assert "freeze_backlog" in codes
    assert "chain_sync_stalled" in codes
