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

"""What the validator records in its snapshot and event log, and the alerts a bad state raises."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from swarm.validator import forward as forward_mod
from swarm.validator.runtime_telemetry import (
    ValidatorRuntimeTracker,
    compute_alerts,
    load_recent_events,
    load_runtime_snapshot,
)


def test_runtime_tracker_writes_snapshot_and_events(tmp_path: Path) -> None:
    """A forward pass and a backend sync land in the snapshot and append both events in the order they happened."""
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
    """A queued model marked at a stage appears in active_items with its progress and note, counted as processing."""
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
        progress_total=700,
        note="chunk 150/700",
    )
    tracker.flush()

    snapshot = load_runtime_snapshot(tmp_path / "validator_runtime.json")
    item = snapshot["queue"]["active_items"][0]
    assert snapshot["queue"]["counts"]["processing"] == 1
    assert snapshot["queue"]["processable_count"] == 1
    assert item["stage"] == "benchmark"
    assert item["progress_done"] == 150
    assert item["progress_total"] == 700
    assert item["note"] == "chunk 150/700"


def test_compute_alerts_flags_fallback_freeze_and_stalls(tmp_path: Path) -> None:
    """A snapshot with a dead worker, an overrunning forward and chain sync, an offline backend and a stale queue raises all seven codes."""
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


def test_an_excluded_host_and_a_denied_lockdown_raise_critical_alerts(tmp_path: Path) -> None:
    """Both states stay on the snapshot as critical alerts until they clear, instead of one log line."""
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    tracker.mark_lockdown_permission(permitted=False, detail="iptables lacks cap_net_admin")
    tracker.mark_scoring_excluded("no valid reference calibration")
    tracker.mark_forward_completed(1)

    alerts = {alert["code"]: alert for alert in tracker.snapshot_copy()["alerts"]}
    assert alerts["sandbox_lockdown_denied"]["severity"] == "critical"
    assert "cap_net_admin" in alerts["sandbox_lockdown_denied"]["message"]
    assert alerts["host_excluded_from_scoring"]["severity"] == "critical"
    assert "calibration" in alerts["host_excluded_from_scoring"]["message"]

    tracker.mark_lockdown_permission(permitted=True, detail="binary capabilities detected")
    tracker.mark_scoring_eligible()
    codes = {alert["code"] for alert in tracker.snapshot_copy()["alerts"]}
    assert not codes & {"sandbox_lockdown_denied", "host_excluded_from_scoring"}


def test_a_repeated_exclusion_keeps_its_start_and_logs_one_event(tmp_path: Path) -> None:
    """Every pass re-marks the exclusion, so only a change is written to the event log."""
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    tracker.mark_scoring_excluded("no valid reference calibration")
    since = tracker.snapshot_copy()["scoring"]["excluded_since"]
    tracker.mark_scoring_excluded("no valid reference calibration")

    assert tracker.snapshot_copy()["scoring"]["excluded_since"] == since
    events = [e["event"] for e in load_recent_events(tracker.events_file, limit=50)]
    assert events.count("scoring_excluded") == 1


def _validator_that_cannot_lock_down(monkeypatch, tmp_path: Path, speed) -> tuple:
    """A validator whose lockdown check fails and whose calibration returns the given speed, with its error log."""
    errors: list[str] = []
    checks: list[int] = []

    async def _speed(evaluator, workers):
        """Hand back the calibration result the test asked for."""
        return speed

    def _denied():
        """Count the check and report the capability as missing."""
        checks.append(1)
        return False, "iptables lacks cap_net_admin"

    monkeypatch.setattr(forward_mod, "_image_provenance_ok", lambda self: True)
    monkeypatch.setattr(forward_mod, "baseline_model_available", lambda: False)
    monkeypatch.setattr(forward_mod, "_ensure_host_speed_factor", _speed)
    monkeypatch.setattr(forward_mod, "lockdown_permission", _denied)
    monkeypatch.setattr(forward_mod.bt.logging, "error", lambda msg: errors.append(str(msg)))
    validator = SimpleNamespace(
        docker_evaluator=None, runtime_tracker=ValidatorRuntimeTracker(state_dir=tmp_path),
    )
    return validator, errors, checks


def test_a_denied_lockdown_is_reported_once_a_minute_while_it_lasts(monkeypatch, tmp_path: Path) -> None:
    """The cause is logged and alerted on the first pass and again each minute, never on every 2 s pass."""
    validator, errors, checks = _validator_that_cannot_lock_down(monkeypatch, tmp_path, speed=None)

    assert asyncio.run(forward_mod._host_may_score(validator)) is False
    assert asyncio.run(forward_mod._host_may_score(validator)) is False
    assert len(checks) == 1
    assert sum("SANDBOX LOCKDOWN NOT PERMITTED" in line for line in errors) == 1

    validator._lockdown_checked_at -= forward_mod.LOCKDOWN_RECHECK_SEC
    asyncio.run(forward_mod._host_may_score(validator))
    assert len(checks) == 2
    assert sum("SANDBOX LOCKDOWN NOT PERMITTED" in line for line in errors) == 2

    scoring = validator.runtime_tracker.snapshot_copy()["scoring"]
    assert scoring["lockdown_permitted"] is False
    assert scoring["excluded"] is True


def test_a_host_that_calibrates_again_is_no_longer_excluded(monkeypatch, tmp_path: Path) -> None:
    """An eligible speed factor clears the exclusion the failed calibration set."""
    validator, _errors, _checks = _validator_that_cannot_lock_down(monkeypatch, tmp_path, speed=None)
    asyncio.run(forward_mod._host_may_score(validator))
    assert validator.runtime_tracker.snapshot_copy()["scoring"]["excluded"] is True

    async def _eligible(evaluator, workers):
        """A calibration that measured the host and found it fast enough."""
        return SimpleNamespace(factor=1.0, eligible=True)

    monkeypatch.setattr(forward_mod, "_ensure_host_speed_factor", _eligible)
    assert asyncio.run(forward_mod._host_may_score(validator)) is True
    assert validator.runtime_tracker.snapshot_copy()["scoring"]["excluded"] is False
