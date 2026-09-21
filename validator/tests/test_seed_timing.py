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

"""The per-seed timing record survives from the flight loop to the tracker, for good seeds, failed seeds and retried ones."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from swarm.benchmark import engine as bench_full_eval
from swarm.validator import runtime_telemetry
from swarm.validator.docker import docker_evaluator as de
from swarm.validator.docker.docker_evaluator_parts import batch
from swarm.validator.runtime_dashboard import render_runtime_dashboard
from swarm.validator.runtime_telemetry import (
    ValidatorRuntimeTracker,
    format_seed_timing_line,
    load_runtime_snapshot,
    summarize_seed_timing,
)
from swarm.validator.utils_parts.heartbeat import HeartbeatManager
from validator.tests.test_docker_evaluator import _new_evaluator, _ScriptedContext

_FLIGHT_PHASES = (
    "env_build_sec",
    "reset_sec",
    "calibration_sec",
    "fly_sec",
    "act_sec",
    "act_max_sec",
    "sim_sec",
    "cleanup_sec",
)


class _Env:
    """Drone environment stand-in whose step, and slow close, are scripted per test."""

    def __init__(self, step):
        """Build a five dimensional action space around the scripted step function."""
        self.action_space = SimpleNamespace(
            low=np.full(5, -1.0, dtype=np.float32),
            high=np.full(5, 1.0, dtype=np.float32),
            shape=(5,),
        )
        self.step = step

    def close(self):
        """Take long enough to tear down that the cleanup clock cannot read zero."""
        time.sleep(0.01)


def _fly_one_seed(monkeypatch, step) -> list:
    """Fly one seed through the real RPC loop against a fake agent and return every record it emitted."""
    rpc_mod = de.rpc
    ev = _new_evaluator()
    records: list = []

    class _Agent:
        """Capnp agent stand-in answering ping, reset and act with canned replies."""

        async def ping(self, _msg):
            """Return the pong reply."""
            return SimpleNamespace(response="pong")

        async def reset(self):
            """Accept the reset and hand nothing back."""
            return None

        async def act(self, _obs):
            """Return a zero action of the expected shape."""
            tensor = SimpleNamespace(
                data=np.zeros(5, dtype=np.float32).tobytes(), dtype="float32", shape=[5]
            )
            return SimpleNamespace(action=tensor)

    class _Loop:
        """Async context manager standing in for the capnp kj event loop."""

        async def __aenter__(self):
            """Enter the fake event loop and yield nothing."""
            return None

        async def __aexit__(self, exc_type, exc, tb):
            """Leave the fake event loop without swallowing an exception."""
            return False

    class _StreamFactory:
        """Stream factory stand-in whose connections are bare objects."""

        @staticmethod
        async def create_connection(**_kwargs):
            """Return a bare object in place of a socket stream."""
            return object()

    async def _fake_calibrate(_agent, _schema, _obs, _uid):
        """Report a ten millisecond overhead on a reference-speed host."""
        return 0.01, 1.0

    client = SimpleNamespace(bootstrap=lambda: SimpleNamespace(cast_as=lambda _schema: _Agent()))
    monkeypatch.setattr(rpc_mod.capnp, "load", lambda _path: SimpleNamespace(Agent=object()))
    monkeypatch.setattr(
        rpc_mod.RpcTraceSettings,
        "from_env",
        lambda: SimpleNamespace(enabled=False, trace_every=1, heartbeat_sec=0.0),
    )
    monkeypatch.setattr(rpc_mod.capnp, "TwoPartyClient", lambda _stream: client)
    monkeypatch.setattr(rpc_mod.capnp, "kj_loop", lambda: _Loop())
    monkeypatch.setattr(rpc_mod.capnp, "AsyncIoStream", _StreamFactory)
    monkeypatch.setattr(ev, "_serialize_observation", lambda _schema, obs: obs)
    monkeypatch.setattr(ev, "_calibrate_rpc_overhead_async", _fake_calibrate)
    monkeypatch.setattr(
        rpc_mod, "make_env_with_initial_obs", lambda _task, gui=False: (_Env(step), {"marker": 0})
    )

    task = SimpleNamespace(
        map_seed=77, challenge_type=1, horizon=0.04, start=(0.0, 0.0, 1.0), goal=(1.0, 1.0, 1.0)
    )
    ev._run_multi_seed_rpc_sync(
        [task], uid=9, rpc_port=8000, on_seed_complete=lambda record=None: records.append(record)
    )
    return records


def test_heartbeat_seed_callback_accepts_the_record():
    """The heartbeat callback takes the seed record, so the emitter never falls back to a bare call."""
    loop = asyncio.new_event_loop()
    hb = HeartbeatManager(SimpleNamespace(), loop)
    hb._active = True

    hb.on_seed_complete({"status": "seed_done", "seed_wall_sec": 12.5})
    hb.on_seed_complete()
    loop.close()

    assert hb._progress == 2


def test_flown_seed_record_carries_every_phase_clock(monkeypatch):
    """A seed that flies reports the old fields plus a clock for each phase, sent once the env is closed."""

    def _step(_action):
        """End the episode on the first step with the goal reached."""
        return {"marker": 1}, 0.0, True, False, {
            "success": True, "min_clearance": 1.0, "collision": False,
        }

    records = _fly_one_seed(monkeypatch, _step)

    assert [record["status"] for record in records] == ["seed_done"]
    record = records[0]
    assert record["map_seed"] == 77
    assert record["calibration_overhead_sec"] == pytest.approx(0.01)
    assert record["calibration_cpu_factor"] == pytest.approx(1.0)
    assert record["seed_wall_sec"] > 0.0
    assert set(_FLIGHT_PHASES) <= set(record)
    assert record["cleanup_sec"] >= 0.01
    assert record["fly_sec"] >= record["sim_sec"] > 0.0
    assert record["act_sec"] >= record["act_max_sec"] > 0.0


def test_failed_seed_record_still_carries_its_clocks(monkeypatch):
    """A seed whose env breaks twice reports the failure with the time spent building and tearing down."""

    def _step(_action):
        """Break the simulator on every step."""
        raise RuntimeError("physics exploded")

    records = _fly_one_seed(monkeypatch, _step)

    assert [record["status"] for record in records] == ["seed_env_failure"]
    assert "physics exploded" in records[0]["error"]
    assert records[0]["env_build_sec"] > 0.0
    assert records[0]["cleanup_sec"] >= 0.01


def test_batch_callback_adds_container_timing(tmp_path):
    """Every record leaving a batch carries how long its container took to start and to answer."""
    records: list = []
    ctx = batch._BatchContext(
        self=SimpleNamespace(),
        tasks=[SimpleNamespace(map_seed=5)],
        uid=3,
        model_path=tmp_path / "model.zip",
        on_seed_complete=lambda record=None: records.append(record),
    )
    batch._init_batch_state(ctx)
    ctx.setup_sec, ctx.launch_sec, ctx.lockdown_sec, ctx.serve_sec = 0.5, 1.0, 0.25, 4.0
    ctx.progress_state.update(rpc_started_ts=100.0, ping_ok_ts=102.5, phase="rpc_act")

    ctx.helpers.on_seed_complete_guarded({"status": "seed_done", "seed_wall_sec": 9.0})

    assert records[0]["seed_wall_sec"] == 9.0
    assert records[0]["container_start_sec"] == pytest.approx(5.75)
    assert records[0]["serve_sec"] == pytest.approx(4.0)
    assert records[0]["connect_sec"] == pytest.approx(2.5)
    assert records[0]["prewarmed"] is False
    assert records[0]["last_phase"] == "rpc_act"


def _seed_event(status: str, wall_sec: float, *, success: bool = False) -> dict:
    """A worker-side seed record with one phase clock, as the flight loop would emit it."""
    return {
        "uid": 41,
        "map_seed": 3101,
        "challenge_type": 4,
        "status": status,
        "success": success,
        "sim_time_sec": 12.0,
        "seed_wall_sec": wall_sec,
        "step_idx": 123,
        "error": "",
        "fly_sec": wall_sec - 1.0,
    }


def _run_scripted(monkeypatch, tmp_path, attempts: list[dict]) -> tuple[ValidatorRuntimeTracker, list]:
    """Run one seed through the parent engine against scripted worker attempts, with a real tracker attached."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"x")
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path / "state")
    log_lines: list = []
    monkeypatch.setattr(de.parallel, "_benchmark_engine", lambda: bench_full_eval)
    monkeypatch.setattr(
        bench_full_eval,
        "_benchmark_mp_context",
        lambda: _ScriptedContext(bench_full_eval, {0: attempts}),
    )
    monkeypatch.setattr(de.parallel.bt.logging, "info", lambda msg: log_lines.append(str(msg)))
    monkeypatch.setattr(de.parallel.bt.logging, "warning", lambda msg: log_lines.append(str(msg)))
    asyncio.run(
        de.parallel._run_process_parallel(
            all_tasks=[SimpleNamespace(challenge_type=4, map_seed=3101, horizon=60.0)],
            task_meta=[
                {"group": "type4_village", "seed": 3101, "index": 0, "challenge_type": 4, "horizon": 60.0}
            ],
            batch_plan=[[0]],
            uid=41,
            model_path=model_path,
            effective_workers=1,
            runtime_tracker=tracker,
            on_seed_complete=lambda record=None: None,
            phase_label="benchmark",
        )
    )
    return tracker, log_lines


def _logged_records(tracker: ValidatorRuntimeTracker) -> list[dict]:
    """Every line of the tracker's seed timing log, parsed."""
    return [json.loads(line) for line in tracker.seed_timing_file.read_text().splitlines()]


def test_engine_logs_the_record_of_a_normal_seed(monkeypatch, tmp_path):
    """A finished seed lands in the timing log with the worker's fields and the parent's total, and in the run summary line."""
    tracker, log_lines = _run_scripted(
        monkeypatch,
        tmp_path,
        [
            {
                "seed_events": [_seed_event("seed_done", 30.0, success=True)],
                "results": [(41, True, 12.0, 0.8)],
            }
        ],
    )

    records = _logged_records(tracker)
    assert [(r["status"], r["attempt"], r["retried"]) for r in records] == [("seed_done", 1, False)]
    assert records[0]["seed_wall_sec"] == 30.0
    assert records[0]["fly_sec"] == 29.0
    assert records[0]["eval_phase"] == "benchmark"
    assert records[0]["total_sec"] >= 0.0
    assert tracker.snapshot_copy()["seed_timing"]["by_status"] == {"seed_done": 1}
    assert any("seed timing n=1 (seed_done 1)" in line for line in log_lines)


def test_engine_logs_the_attempt_that_timed_out_before_its_retry(monkeypatch, tmp_path):
    """The timed-out first attempt is logged as retried, then the second attempt as the final one."""
    tracker, _ = _run_scripted(
        monkeypatch,
        tmp_path,
        [
            {
                "seed_events": [_seed_event("seed_cancelled", 240.0)],
                "results": [(41, False, 12.0, 0.0)],
            },
            {
                "seed_events": [_seed_event("seed_done", 30.0, success=True)],
                "results": [(41, True, 12.0, 0.8)],
            },
        ],
    )

    assert [(r["status"], r["attempt"], r["retried"], r["seed_wall_sec"]) for r in _logged_records(tracker)] == [
        ("seed_cancelled", 1, True, 240.0),
        ("seed_done", 2, False, 30.0),
    ]


def test_engine_logs_a_batch_that_died(monkeypatch, tmp_path):
    """A worker that returns an error still leaves a record for its seed, under the failure status."""
    tracker, _ = _run_scripted(
        monkeypatch, tmp_path, [{"results": [], "error": "RuntimeError: boom", "elapsed_sec": 7.5}]
    )

    records = _logged_records(tracker)
    assert [(r["status"], r["success"]) for r in records] == [("batch_exception", False)]
    assert records[0]["seed_wall_sec"] == 7.5
    assert "boom" in records[0]["error"]


def test_tracker_aggregates_only_the_phases_a_seed_reached(tmp_path: Path):
    """The snapshot holds each phase's spread and the total per status, and a phase left at zero is not a sample."""
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    for total in (10.0, 20.0, 30.0):
        tracker.record_seed_timing({"status": "seed_done", "total_sec": total, "fly_sec": total / 2})
    tracker.record_seed_timing({"status": "batch_timeout", "total_sec": 300.0, "fly_sec": 0.0})
    tracker.flush()

    timing = load_runtime_snapshot(tmp_path / "validator_runtime.json")["seed_timing"]
    assert timing["seeds"] == 4
    assert timing["by_status"] == {"seed_done": 3, "batch_timeout": 1}
    assert timing["phases"]["fly_sec"] == {"n": 3, "median": 10.0, "p90": 15.0, "max": 15.0}
    assert timing["phases"]["total_sec"]["max"] == 300.0
    assert timing["total_by_status"]["seed_done"] == {"n": 3, "median": 20.0, "p90": 30.0, "max": 30.0}
    assert timing["total_by_status"]["batch_timeout"]["median"] == 300.0
    assert "env_build_sec" not in timing["phases"]


def test_seed_timing_log_rolls_over_when_full(monkeypatch, tmp_path: Path):
    """Past the size cap the log moves to a single .1 file and a fresh one starts."""
    monkeypatch.setattr(runtime_telemetry, "_SEED_TIMING_MAX_BYTES", 200)
    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    for index in range(6):
        tracker.record_seed_timing({"status": "seed_done", "map_seed": index, "total_sec": 1.0})

    rolled = tracker.seed_timing_file.with_name(tracker.seed_timing_file.name + ".1")
    assert rolled.exists()
    kept = rolled.read_text().splitlines() + tracker.seed_timing_file.read_text().splitlines()
    assert 0 < len(tracker.seed_timing_file.read_text().splitlines()) < 6
    assert json.loads(kept[-1])["map_seed"] == 5


def test_summary_line_and_dashboard_show_the_spread(tmp_path: Path):
    """The log line and the monitor frame both print median, p90 and max for the measured phases in seed order."""
    records = [{"status": "seed_done", "total_sec": 42.0, "fly_sec": 30.0, "queue_wait_sec": 0.5}]
    line = format_seed_timing_line(summarize_seed_timing(records))
    assert line.startswith("seed timing n=1 (seed_done 1)")
    assert line.index("total 42.00/42.00/42.00") < line.index("queue_wait 0.50") < line.index("fly 30.00")

    tracker = ValidatorRuntimeTracker(state_dir=tmp_path)
    tracker.record_seed_timing(records[0])
    tracker.flush()
    frame = render_runtime_dashboard(load_runtime_snapshot(tmp_path / "validator_runtime.json"))
    assert "Seed Timing (last 1 seeds, median / p90 / max)" in frame
    assert frame.index("queue wait") < frame.index("fly  ")
    assert "42.0s / 42.0s / 42.0s" in frame
