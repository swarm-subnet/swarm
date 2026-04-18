from __future__ import annotations

import asyncio
import io
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarm.benchmark import engine as bench_full_eval
from swarm.constants import N_DOCKER_WORKERS


def _argv_for_model(model_path, *extra: str) -> list[str]:
    return [
        "bench_full_eval.py",
        "--model",
        str(model_path),
        "--workers",
        "1",
        "--seeds-per-group",
        "1",
        *extra,
    ]


def test_tee_write_ignores_closed_secondary_stream():
    primary = io.StringIO()
    secondary = io.StringIO()
    tee = bench_full_eval._Tee(primary, secondary)

    secondary.close()
    written = tee.write("hello")
    tee.flush()
    tee.reconfigure(line_buffering=True)

    assert written == 5
    assert primary.getvalue() == "hello"


def test_infer_uid_from_model_path():
    assert bench_full_eval._infer_uid_from_model_path(Path("model/UID_178.zip")) == 178
    assert bench_full_eval._infer_uid_from_model_path(Path("model/uid-42.zip")) == 42
    assert bench_full_eval._infer_uid_from_model_path(Path("model/submission.zip")) is None


def test_batch_indices_creates_one_seed_per_batch():
    assert bench_full_eval._batch_indices(5) == [
        [0],
        [1],
        [2],
        [3],
        [4],
    ]


def test_parse_args_defaults_workers_to_dynamic_count(tmp_path):
    model_path = tmp_path / "submission.zip"
    model_path.write_bytes(b"x")

    args = bench_full_eval._parse_args(["--model", str(model_path)])

    assert args.workers == N_DOCKER_WORKERS


def test_build_worker_stall_seed_meta_marks_failure():
    task = SimpleNamespace(
        map_seed=123,
        challenge_type=5,
        horizon=60.0,
        moving_platform=False,
    )

    meta = bench_full_eval._build_worker_stall_seed_meta(
        task,
        uid=7,
        elapsed_sec=91.5,
        error="worker stalled",
    )

    assert meta["uid"] == 7
    assert meta["map_seed"] == 123
    assert meta["challenge_type"] == 5
    assert meta["status"] == "worker_stall_timeout"
    assert meta["success"] is False
    assert meta["seed_wall_sec"] == pytest.approx(91.5)


def test_resource_class_assignments_match_expected_groups():
    assert bench_full_eval._resource_class_for_group("type1_city") == "light"
    assert bench_full_eval._resource_class_for_group("type2_open") == "light"
    assert bench_full_eval._resource_class_for_group("type5_warehouse") == "medium"
    assert bench_full_eval._resource_class_for_group("type4_village") == "medium"
    assert bench_full_eval._resource_class_for_group("type3_mountain") == "heavy"
    assert bench_full_eval._resource_class_for_group("type6_forest") == "heavy"


def test_select_next_batch_index_prefers_light_when_heavy_is_already_active():
    batch_plan = [[0], [1], [2]]
    task_meta = [
        {"group": "type3_mountain"},
        {"group": "type5_warehouse"},
        {"group": "type1_city"},
    ]
    scheduler = bench_full_eval._AdaptiveBackoffController(
        requested_workers=2,
        machine_vcpus=2,
        machine_total_ram_mb=8192,
    )

    selected = bench_full_eval._select_next_batch_index(
        pending_batch_ids=[1, 2],
        batch_plan=batch_plan,
        task_meta=task_meta,
        active_batch_ids=[0],
        active_worker_cap=2,
        scheduler=scheduler,
    )

    assert selected == 2


def test_select_next_batch_index_waits_when_only_extra_heavy_seed_is_pending():
    batch_plan = [[0], [1]]
    task_meta = [
        {"group": "type6_forest"},
        {"group": "type3_mountain"},
    ]
    scheduler = bench_full_eval._AdaptiveBackoffController(
        requested_workers=3,
        machine_vcpus=3,
        machine_total_ram_mb=16384,
    )

    selected = bench_full_eval._select_next_batch_index(
        pending_batch_ids=[1],
        batch_plan=batch_plan,
        task_meta=task_meta,
        active_batch_ids=[0],
        active_worker_cap=3,
        scheduler=scheduler,
    )

    assert selected is None


def test_select_next_batch_index_allows_heavy_seed_when_capacity_is_available():
    batch_plan = [[0], [1]]
    task_meta = [
        {"group": "type1_city"},
        {"group": "type6_forest"},
    ]
    scheduler = bench_full_eval._AdaptiveBackoffController(
        requested_workers=3,
        machine_vcpus=3,
        machine_total_ram_mb=16384,
    )

    selected = bench_full_eval._select_next_batch_index(
        pending_batch_ids=[1],
        batch_plan=batch_plan,
        task_meta=task_meta,
        active_batch_ids=[0],
        active_worker_cap=3,
        scheduler=scheduler,
    )

    assert selected == 1


def test_max_heavy_active_scales_with_worker_count():
    assert bench_full_eval._max_heavy_active(1) == 1
    assert bench_full_eval._max_heavy_active(2) == 1
    assert bench_full_eval._max_heavy_active(3) == 1
    assert bench_full_eval._max_heavy_active(4) == 1
    assert bench_full_eval._max_heavy_active(6) == 2
    assert bench_full_eval._max_heavy_active(8) == 3
    assert bench_full_eval._max_heavy_active(12) == 4


def test_scheduler_ignores_non_resource_failure_statuses():
    controller = bench_full_eval._AdaptiveBackoffController(
        requested_workers=8,
        machine_vcpus=8,
        machine_total_ram_mb=65536,
    )

    for status in (
        "container_start_failed",
        "pip_install_failed",
        "network_lockdown_failed",
        "submission_start_failed",
        "rpc_connection_failed",
    ):
        note = controller.observe_seed({"status": status, "map_seed": 10, "challenge_type": 2})
        assert note is None

    assert controller.start_worker_cap == 3
    assert controller.active_worker_cap == controller.start_worker_cap
    assert controller.max_worker_cap == 8


def test_scheduler_reduces_cap_on_critical_pressure_and_relaxes_after_recovery():
    snapshots = deque(
        [
            {"cpu_percent": 95.0, "load_ratio": 1.20, "mem_available_mb": 24000.0, "mem_total_mb": 65536.0, "ts": 1.0},
            {"cpu_percent": 96.0, "load_ratio": 1.18, "mem_available_mb": 23500.0, "mem_total_mb": 65536.0, "ts": 2.0},
            {"cpu_percent": 40.0, "load_ratio": 0.40, "mem_available_mb": 36000.0, "mem_total_mb": 65536.0, "ts": 3.0},
        ]
    )

    def _provider():
        if snapshots:
            return snapshots.popleft()
        return {
            "cpu_percent": 38.0,
            "load_ratio": 0.35,
            "mem_available_mb": 36500.0,
            "mem_total_mb": 65536.0,
            "ts": time.time(),
        }

    controller = bench_full_eval._AdaptiveBackoffController(
        requested_workers=8,
        machine_vcpus=8,
        machine_total_ram_mb=65536,
        resource_provider=_provider,
    )
    controller.active_worker_cap = 8
    controller.active_heavy_cap = 3

    note_one = controller.observe_resources([])
    note_two = controller.observe_resources([])

    assert note_one is None
    assert "Scheduler pressure backoff" in str(note_two)
    assert controller.worker_cap_levels == (8, 6, 5, 4, 3, 2)
    assert controller.active_worker_cap == 5
    assert controller.active_heavy_cap == 2

    relax_notes = []
    for idx in range(8):
        controller.observe_resources([])
        relax_notes.append(
            controller.observe_seed(
                {
                    "status": "seed_done",
                    "map_seed": 600 + idx,
                    "challenge_type": 2,
                }
            )
        )

    assert controller.active_worker_cap > 5
    assert any(note and "Scheduler relaxed" in note for note in relax_notes)


def test_scheduler_cold_starts_low_with_single_heavy_slot():
    controller = bench_full_eval._AdaptiveBackoffController(
        requested_workers=12,
        machine_vcpus=12,
        machine_total_ram_mb=65536,
    )

    assert controller.max_worker_cap == 12
    assert controller.start_worker_cap == 3
    assert controller.active_worker_cap == 3
    assert controller.active_heavy_cap == 1


def test_scheduler_promotes_light_group_when_observed_runtime_is_expensive():
    controller = bench_full_eval._AdaptiveBackoffController(
        requested_workers=12,
        machine_vcpus=12,
        machine_total_ram_mb=65536,
    )
    controller.latest_pressure = "healthy"

    for idx in range(4):
        controller.observe_seed(
            {
                "status": "seed_done",
                "map_seed": 1000 + idx,
                "challenge_type": 1,
                "sim_time_sec": 20.0,
                "seed_wall_sec": 190.0,
                "horizon_sec": 60.0,
            },
            group_name="type1_city",
        )

    learned = controller._cost_for_group("type1_city")
    assert learned.resource_class == "heavy"
    assert learned.heavy_tokens == 2


def test_scheduler_can_demote_heavy_group_one_step_after_consistent_healthy_samples():
    controller = bench_full_eval._AdaptiveBackoffController(
        requested_workers=12,
        machine_vcpus=12,
        machine_total_ram_mb=65536,
    )
    controller.latest_pressure = "healthy"

    for idx in range(5):
        controller.observe_seed(
            {
                "status": "seed_done",
                "map_seed": 2000 + idx,
                "challenge_type": 3,
                "sim_time_sec": 20.0,
                "seed_wall_sec": 36.0,
                "horizon_sec": 60.0,
            },
            group_name="type3_mountain",
        )

    learned = controller._cost_for_group("type3_mountain")
    assert learned.resource_class == "medium"
    assert learned.heavy_tokens == 1


def test_save_and_load_type_seeds(tmp_path):
    seed_file = tmp_path / "seeds.json"
    payload = {group: [i + 1] for i, group in enumerate(bench_full_eval.BENCH_GROUP_ORDER)}
    bench_full_eval._save_type_seeds(seed_file, payload)
    assert bench_full_eval._load_type_seeds(seed_file) == payload


def test_main_infers_uid_from_model_filename(monkeypatch, tmp_path):
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")
    captured = {}

    async def _fake_run_benchmark(model_path, uid, type_seeds, num_workers, run_opts):
        _ = model_path, type_seeds, num_workers, run_opts
        captured["uid"] = uid
        return ([], [], [], {}, {}, {}, [], 0.0, 0.0, 1)

    monkeypatch.setattr(
        bench_full_eval,
        "_find_seeds",
        lambda seeds_per_group: {"type5_warehouse": [200662]},
    )
    monkeypatch.setattr(bench_full_eval, "_run_benchmark", _fake_run_benchmark)
    monkeypatch.setattr(sys, "argv", _argv_for_model(model_path))

    bench_full_eval.main()
    assert captured["uid"] == 178


def test_main_explicit_uid_overrides_model_inference(monkeypatch, tmp_path):
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")
    captured = {}

    async def _fake_run_benchmark(model_path, uid, type_seeds, num_workers, run_opts):
        _ = model_path, type_seeds, num_workers, run_opts
        captured["uid"] = uid
        return ([], [], [], {}, {}, {}, [], 0.0, 0.0, 1)

    monkeypatch.setattr(
        bench_full_eval,
        "_find_seeds",
        lambda seeds_per_group: {"type5_warehouse": [200662]},
    )
    monkeypatch.setattr(bench_full_eval, "_run_benchmark", _fake_run_benchmark)
    monkeypatch.setattr(sys, "argv", _argv_for_model(model_path, "--uid", "12"))

    bench_full_eval.main()
    assert captured["uid"] == 12


def test_main_prints_results_and_completion_footer(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")
    out = io.StringIO()
    err = io.StringIO()

    seed = 200662
    task_meta = [{
        "group": "type5_warehouse",
        "bench_type": 5,
        "seed": seed,
        "challenge_type": 5,
        "horizon": 60.0,
        "moving_platform": False,
    }]
    fake_result = SimpleNamespace(success=False, score=0.01, time_sec=60.0)

    async def _fake_run_benchmark(model_path, uid, type_seeds, num_workers, run_opts):
        _ = model_path, uid, type_seeds, num_workers, run_opts
        eval_start = 1000.0
        return (
            task_meta,
            [fake_result],
            [1060.0],
            {(seed, 5): deque([60.0])},
            {(seed, 5): deque(["seed_done"])},
            {(seed, 5): 61.0},
            [bench_full_eval._BatchStat(0, 0, 1, 61.0, 60.0, 1.0, [seed])],
            61.0,
            eval_start,
            1,
        )

    monkeypatch.setattr(
        bench_full_eval,
        "_find_seeds",
        lambda seeds_per_group: {"type5_warehouse": [seed]},
    )
    monkeypatch.setattr(bench_full_eval, "_run_benchmark", _fake_run_benchmark)
    monkeypatch.setattr(sys, "__stdout__", out)
    monkeypatch.setattr(sys, "__stderr__", err)
    monkeypatch.setattr(sys, "argv", _argv_for_model(model_path))

    bench_full_eval.main()
    combined = out.getvalue() + err.getvalue()

    assert "=== RESULTS ===" in combined
    assert "Run summary:" in combined
    assert "Clean execution rate:      1/1 (100.0%)" in combined
    assert "=== BENCHMARK COMPLETE ===" in combined


def test_main_writes_final_report_to_log_file(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")
    log_path = tmp_path / "bench.log"
    out = io.StringIO()
    err = io.StringIO()

    seed = 200662
    task_meta = [{
        "group": "type5_warehouse",
        "bench_type": 5,
        "seed": seed,
        "challenge_type": 5,
        "horizon": 60.0,
        "moving_platform": False,
    }]
    fake_result = SimpleNamespace(success=False, score=0.01, time_sec=60.0)

    async def _fake_run_benchmark(model_path, uid, type_seeds, num_workers, run_opts):
        _ = model_path, uid, type_seeds, num_workers, run_opts
        eval_start = 1000.0
        return (
            task_meta,
            [fake_result],
            [1060.0],
            {(seed, 5): deque([60.0])},
            {(seed, 5): deque(["seed_done"])},
            {(seed, 5): 61.0},
            [bench_full_eval._BatchStat(0, 0, 1, 61.0, 60.0, 1.0, [seed])],
            61.0,
            eval_start,
            1,
        )

    monkeypatch.setattr(
        bench_full_eval,
        "_find_seeds",
        lambda seeds_per_group: {"type5_warehouse": [seed]},
    )
    monkeypatch.setattr(bench_full_eval, "_run_benchmark", _fake_run_benchmark)
    monkeypatch.setattr(sys, "__stdout__", out)
    monkeypatch.setattr(sys, "__stderr__", err)
    monkeypatch.setattr(sys, "argv", _argv_for_model(model_path, "--log-out", str(log_path)))

    bench_full_eval.main()

    log_text = log_path.read_text()
    assert "Run summary:" in log_text
    assert "Clean execution rate:      1/1 (100.0%)" in log_text
    assert "=== BENCHMARK COMPLETE ===" in log_text


def test_main_prints_failed_footer_when_benchmark_raises(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")
    out = io.StringIO()
    err = io.StringIO()

    async def _fake_run_benchmark(model_path, uid, type_seeds, num_workers, run_opts):
        _ = model_path, uid, type_seeds, num_workers, run_opts
        raise RuntimeError("simulated benchmark failure")

    monkeypatch.setattr(
        bench_full_eval,
        "_find_seeds",
        lambda seeds_per_group: {"type5_warehouse": [200662]},
    )
    monkeypatch.setattr(bench_full_eval, "_run_benchmark", _fake_run_benchmark)
    monkeypatch.setattr(sys, "__stdout__", out)
    monkeypatch.setattr(sys, "__stderr__", err)
    monkeypatch.setattr(sys, "argv", _argv_for_model(model_path))

    with pytest.raises(RuntimeError, match="simulated benchmark failure"):
        bench_full_eval.main()

    combined = out.getvalue() + err.getvalue()
    assert "=== RESULTS ===" in combined
    assert "Benchmark failed before report generation: RuntimeError: simulated benchmark failure" in combined
    assert "=== BENCHMARK FAILED ===" in combined


def test_main_report_uses_runtime_worker_count(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")
    out = io.StringIO()
    err = io.StringIO()

    seed = 200662
    task_meta = [{
        "group": "type5_warehouse",
        "bench_type": 5,
        "seed": seed,
        "challenge_type": 5,
        "horizon": 60.0,
        "moving_platform": False,
    }]
    fake_result = SimpleNamespace(success=False, score=0.01, time_sec=60.0)

    async def _fake_run_benchmark(model_path, uid, type_seeds, num_workers, run_opts):
        _ = model_path, uid, type_seeds, num_workers, run_opts
        eval_start = 1000.0
        return (
            task_meta,
            [fake_result],
            [1060.0],
            {(seed, 5): deque([60.0])},
            {(seed, 5): deque(["seed_done"])},
            {(seed, 5): 61.0},
            [bench_full_eval._BatchStat(0, 0, 1, 61.0, 60.0, 1.0, [seed])],
            61.0,
            eval_start,
            3,
        )

    monkeypatch.setattr(
        bench_full_eval,
        "_find_seeds",
        lambda seeds_per_group: {"type5_warehouse": [seed]},
    )
    monkeypatch.setattr(bench_full_eval, "_run_benchmark", _fake_run_benchmark)
    monkeypatch.setattr(sys, "__stdout__", out)
    monkeypatch.setattr(sys, "__stderr__", err)
    monkeypatch.setattr(sys, "argv", _argv_for_model(model_path, "--workers", "2"))

    bench_full_eval.main()
    combined = out.getvalue() + err.getvalue()
    assert "Workers used:              3" in combined


def test_main_prints_failed_footer_when_seed_selection_raises(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")
    out = io.StringIO()
    err = io.StringIO()

    def _fake_find_seeds(seeds_per_group):
        _ = seeds_per_group
        raise ValueError("simulated seed selection failure")

    monkeypatch.setattr(bench_full_eval, "_find_seeds", _fake_find_seeds)
    monkeypatch.setattr(sys, "__stdout__", out)
    monkeypatch.setattr(sys, "__stderr__", err)
    monkeypatch.setattr(sys, "argv", _argv_for_model(model_path))

    with pytest.raises(ValueError, match="simulated seed selection failure"):
        bench_full_eval.main()

    combined = out.getvalue() + err.getvalue()
    assert "=== RESULTS ===" in combined
    assert "Benchmark failed before report generation: ValueError: simulated seed selection failure" in combined
    assert "=== BENCHMARK FAILED ===" in combined


def test_run_benchmark_keeps_requested_worker_count(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")
    captured = {}

    class _FakeEvaluator:
        _base_ready = True

    def _fake_random_task(sim_dt, seed):
        _ = sim_dt
        return SimpleNamespace(
            map_seed=seed,
            challenge_type=5,
            horizon=60.0,
            moving_platform=False,
            start=(0.0, 0.0, 0.0),
        )

    import swarm.validator.docker.docker_evaluator as docker_eval_mod
    import swarm.validator.task_gen as task_gen

    monkeypatch.setattr(task_gen, "random_task", _fake_random_task)
    monkeypatch.setattr(docker_eval_mod, "DockerSecureEvaluator", _FakeEvaluator)
    async def _fake_process_mode(**kwargs):
        captured["effective_workers"] = kwargs["effective_workers"]
        kwargs["on_seed_done"](
            {
                "map_seed": 123456,
                "challenge_type": 5,
                "seed_wall_sec": 0.1,
                "status": "seed_done",
            }
        )
        kwargs["record_batch_completion"](
            0,
            0,
            [0],
            [SimpleNamespace(uid=0, success=False, time_sec=0.0, score=0.0)],
            0.1,
        )
        return kwargs["effective_workers"]

    monkeypatch.setattr(bench_full_eval, "_run_benchmark_process_mode", _fake_process_mode)

    out = asyncio.run(
        bench_full_eval._run_benchmark(
            model_path=model_path,
            uid=0,
            type_seeds={"type5_warehouse": [123456]},
            num_workers=30,
            run_opts=bench_full_eval._RunOptions(),
        )
    )

    launched_workers = out[-1]
    assert captured["effective_workers"] == 30
    assert launched_workers == 30


def test_run_benchmark_uses_process_mode_runner(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")
    captured = {}

    class _FakeEvaluator:
        _base_ready = True

    def _fake_random_task(sim_dt, seed):
        _ = sim_dt
        return SimpleNamespace(
            map_seed=seed,
            challenge_type=5,
            horizon=60.0,
            moving_platform=False,
            start=(0.0, 0.0, 0.0),
        )

    async def _fake_process_mode(**kwargs):
        captured["called"] = True
        kwargs["on_seed_done"](
            {
                "map_seed": 123456,
                "challenge_type": 5,
                "seed_wall_sec": 0.25,
                "status": "seed_done",
            }
        )
        kwargs["record_batch_completion"](
            1,
            0,
            [0],
            [SimpleNamespace(uid=0, success=True, time_sec=1.0, score=0.5)],
            0.5,
        )
        return kwargs["effective_workers"]

    import swarm.validator.docker.docker_evaluator as docker_eval_mod
    import swarm.validator.task_gen as task_gen

    monkeypatch.setattr(task_gen, "random_task", _fake_random_task)
    monkeypatch.setattr(docker_eval_mod, "DockerSecureEvaluator", _FakeEvaluator)
    monkeypatch.setattr(
        bench_full_eval,
        "_run_benchmark_process_mode",
        _fake_process_mode,
    )

    out = asyncio.run(
        bench_full_eval._run_benchmark(
            model_path=model_path,
            uid=0,
            type_seeds={"type5_warehouse": [123456]},
            num_workers=2,
            run_opts=bench_full_eval._RunOptions(),
        )
    )

    assert captured["called"] is True
    assert out[-1] == 2
    assert out[1][0].score == 0.5
    assert out[4][(123456, 5)][0] == "seed_done"
    assert out[5][(123456, 5)] == 0.5


def test_benchmark_worker_main_emits_progress_and_results(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")
    task_queue: queue.Queue = queue.Queue()
    result_queue: queue.Queue = queue.Queue()
    progress_queue: queue.Queue = queue.Queue()

    class _FakeEvaluator:
        async def evaluate_seeds_batch(
            self,
            tasks,
            uid,
            model_path,
            worker_id=0,
            on_seed_complete=None,
            task_offset=0,
            task_total=None,
            model_image=None,
        ):
            _ = uid, model_path, worker_id, task_offset, task_total, model_image
            for task in tasks:
                if on_seed_complete is not None:
                    on_seed_complete(
                        {
                            "map_seed": task.map_seed,
                            "challenge_type": task.challenge_type,
                            "seed_wall_sec": 0.2,
                            "status": "seed_done",
                        }
                    )
            return [SimpleNamespace(uid=uid, success=False, time_sec=0.0, score=0.0) for _ in tasks]

    monkeypatch.setattr(
        bench_full_eval,
        "_create_prepared_benchmark_evaluator",
        lambda: _FakeEvaluator(),
    )

    task_queue.put(
        bench_full_eval._ProcessBatchRequest(
            batch_index=0,
            batch_indices=[0, 1],
            tasks=[
                SimpleNamespace(map_seed=10, challenge_type=5),
                SimpleNamespace(map_seed=11, challenge_type=5),
            ],
            uid=7,
            model_path=str(model_path),
            task_total=2,
        )
    )
    task_queue.put(None)

    bench_full_eval._benchmark_worker_main(0, task_queue, result_queue, progress_queue)

    queue_events = []
    while True:
        try:
            queue_events.append(progress_queue.get_nowait())
        except queue.Empty:
            break
    progress_events = [
        event for event in queue_events if isinstance(event, bench_full_eval._ProcessSeedEvent)
    ]
    heartbeat_events = [
        event for event in queue_events if isinstance(event, bench_full_eval._ProcessWorkerHeartbeat)
    ]
    result = result_queue.get_nowait()

    assert [event.seed_meta["map_seed"] for event in progress_events] == [10, 11]
    assert [event.seed_meta["status"] for event in progress_events] == ["seed_done", "seed_done"]
    assert heartbeat_events[0].event_type == "batch_started"
    assert heartbeat_events[0].worker_id == 0
    assert heartbeat_events[0].batch_index == 0
    assert result.worker_id == 0
    assert result.batch_index == 0
    assert len(result.results) == 2


def test_process_mode_discards_stalled_seed_and_replaces_worker(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")

    monkeypatch.setattr(bench_full_eval, "_PARENT_WORKER_STALL_TIMEOUT_SEC", 0.05)
    monkeypatch.setattr(bench_full_eval, "_PARENT_WORKER_HEARTBEAT_SEC", 0.01)

    class _FakeProcess:
        generations: dict[int, int] = {}

        def __init__(self, target, args, name=None, daemon=None):
            _ = target, name, daemon
            self.worker_slot = int(args[0])
            self.task_queue = args[1]
            self.result_queue = args[2]
            self.progress_queue = args[3]
            self.generation = self.generations.get(self.worker_slot, 0)
            self.generations[self.worker_slot] = self.generation + 1
            self._thread = None
            self._stop = threading.Event()
            self.exitcode = None

        def start(self):
            if self.generation == 0:
                def _stall():
                    request = self.task_queue.get()
                    if request is None:
                        self.exitcode = 0
                        return
                    self.progress_queue.put(
                        bench_full_eval._ProcessWorkerHeartbeat(
                            worker_id=self.worker_slot,
                            batch_index=request.batch_index,
                            event_type="batch_started",
                            ts=time.time(),
                        )
                    )
                    while not self._stop.wait(0.01):
                        pass
                    if self.exitcode is None:
                        self.exitcode = -15

                self._thread = threading.Thread(target=_stall, daemon=True)
            else:
                def _idle():
                    request = self.task_queue.get()
                    if request is None:
                        self.exitcode = 0
                        return
                    self.progress_queue.put(
                        bench_full_eval._ProcessWorkerHeartbeat(
                            worker_id=self.worker_slot,
                            batch_index=request.batch_index,
                            event_type="batch_started",
                            ts=time.time(),
                        )
                    )
                    self.exitcode = 0

                self._thread = threading.Thread(target=_idle, daemon=True)
            self._thread.start()

        def is_alive(self):
            return bool(self._thread and self._thread.is_alive())

        def join(self, timeout=None):
            if self._thread is not None:
                self._thread.join(timeout=timeout)

        def terminate(self):
            self.exitcode = -15
            self._stop.set()

    class _FakeCtx:
        @staticmethod
        def Queue():
            return queue.Queue()

        @staticmethod
        def Process(*args, **kwargs):
            return _FakeProcess(*args, **kwargs)

    monkeypatch.setattr(bench_full_eval, "_benchmark_mp_context", lambda: _FakeCtx())

    recorded = []
    seed_events = []

    def _record_batch_completion(worker_slot, batch_index, batch_indices, seed_results, batch_elapsed):
        recorded.append(
            {
                "worker_slot": worker_slot,
                "batch_index": batch_index,
                "batch_indices": list(batch_indices),
                "seed_results": list(seed_results),
                "batch_elapsed": batch_elapsed,
            }
        )

    def _on_seed_done(seed_meta):
        seed_events.append(seed_meta)

    task = SimpleNamespace(
        map_seed=123456,
        challenge_type=5,
        horizon=60.0,
        moving_platform=False,
    )

    launched = asyncio.run(
        bench_full_eval._run_benchmark_process_mode(
            all_tasks=[task],
            task_meta=[{"group": "type5_warehouse", "seed": 123456, "challenge_type": 5}],
            batch_plan=[[0]],
            uid=7,
            model_path=model_path,
            effective_workers=1,
            record_batch_completion=_record_batch_completion,
            on_seed_done=_on_seed_done,
            run_opts=bench_full_eval._RunOptions(heartbeat_sec=0.0),
        )
    )

    assert launched == 1
    assert len(recorded) == 1
    assert len(recorded[0]["seed_results"]) == 1
    assert recorded[0]["seed_results"][0].success is False
    assert seed_events[0]["status"] == "worker_stall_timeout"
