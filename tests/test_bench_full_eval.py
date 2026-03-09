from __future__ import annotations

import asyncio
import io
import queue
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest

from debugging import bench_full_eval as bench_full_eval


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

    import swarm.validator.task_gen as task_gen
    import swarm.validator.docker.docker_evaluator as docker_eval_mod

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

    import swarm.validator.task_gen as task_gen
    import swarm.validator.docker.docker_evaluator as docker_eval_mod

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
        ):
            _ = uid, model_path, worker_id, task_offset, task_total
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

    progress_events = [progress_queue.get_nowait(), progress_queue.get_nowait()]
    result = result_queue.get_nowait()

    assert [event.seed_meta["map_seed"] for event in progress_events] == [10, 11]
    assert [event.seed_meta["status"] for event in progress_events] == ["seed_done", "seed_done"]
    assert result.worker_id == 0
    assert result.batch_index == 0
    assert len(result.results) == 2
