from __future__ import annotations

import io
import sys
from collections import deque
from types import SimpleNamespace

import pytest

from debugging import bench_full_eval as bench_full_eval


def _argv_for_model(model_path, *extra: str) -> list[str]:
    return [
        "bench_full_eval.py",
        "--model",
        str(model_path),
        "--profile",
        "standard",
        "--workers",
        "1",
        "--seeds-per-group",
        "1",
        "--include-map-types",
        "5",
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
            61.0,
            eval_start,
        )

    monkeypatch.setattr(
        bench_full_eval,
        "_find_seeds",
        lambda seeds_per_group, selected_groups=None: {"type5_warehouse": [seed]},
    )
    monkeypatch.setattr(bench_full_eval, "_run_benchmark", _fake_run_benchmark)
    monkeypatch.setattr(sys, "__stdout__", out)
    monkeypatch.setattr(sys, "__stderr__", err)
    monkeypatch.setattr(sys, "argv", _argv_for_model(model_path))

    bench_full_eval.main()
    combined = out.getvalue() + err.getvalue()

    assert "=== RESULTS ===" in combined
    assert "Run summary:" in combined
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
        lambda seeds_per_group, selected_groups=None: {"type5_warehouse": [200662]},
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


def test_main_prints_failed_footer_when_seed_selection_raises(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"zip")
    out = io.StringIO()
    err = io.StringIO()

    def _fake_find_seeds(seeds_per_group, selected_groups=None):
        _ = seeds_per_group, selected_groups
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
