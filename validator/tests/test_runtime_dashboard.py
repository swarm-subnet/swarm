from __future__ import annotations

import json
from pathlib import Path

from swarm import cli
from swarm.validator.runtime_dashboard import render_runtime_dashboard
from swarm.validator.runtime_telemetry import ValidatorRuntimeTracker


def test_render_runtime_dashboard_includes_sections(tmp_path: Path) -> None:
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    tracker.mark_worker_thread_alive(True)
    tracker.mark_forward_started(5)
    tracker.mark_backend_sync_started()
    tracker.mark_backend_sync_completed(
        fallback=True,
        pending_models_count=3,
        reeval_queue_count=1,
        leaderboard_version=12,
        error="timeout",
    )
    tracker.mark_forward_failed("timeout")
    tracker.flush()

    snapshot = json.loads((tmp_path / "validator_runtime.json").read_text())
    events = [json.loads(line) for line in (tmp_path / "validator_events.jsonl").read_text().splitlines()]
    frame = render_runtime_dashboard(snapshot, events=events, now=snapshot["updated_at"] + 5)

    assert "Swarm Validator Monitor" in frame
    assert "Alerts" in frame
    assert "Backend" in frame
    assert "Queue Items" in frame
    assert "Recent Events" in frame
    assert "pending models" in frame


def test_monitor_cli_once_renders_snapshot(monkeypatch, tmp_path: Path, capsys) -> None:
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    tracker.mark_worker_thread_alive(True)
    tracker.mark_forward_started(2)
    tracker.mark_forward_completed(2)
    tracker.flush()

    rc = cli.main(
        [
            "monitor",
            "--snapshot",
            str(tmp_path / "validator_runtime.json"),
            "--events",
            str(tmp_path / "validator_events.jsonl"),
            "--once",
            "--no-clear",
        ]
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "Swarm Validator Monitor" in output
    assert "Forward" in output
    assert "Chain / Weights" in output
