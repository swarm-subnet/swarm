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
