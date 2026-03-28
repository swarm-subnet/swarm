from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def test_screening_count_matches_validator_ratio(reload_module):
    mod = reload_module("scripts.stress_benchmark_compare")

    assert mod._screening_count_for_total(1000) == 200
    assert mod._screening_count_for_total(100) == 20
    assert mod._screening_count_for_total(6) == 1


def test_classify_raw_seeds_preserves_order_and_phase(reload_module, monkeypatch):
    mod = reload_module("scripts.stress_benchmark_compare")

    monkeypatch.setattr(
        mod,
        "random_task",
        lambda sim_dt, seed: SimpleNamespace(challenge_type=((int(seed) - 10) % 6) + 1),
    )

    grouped, manifest = mod._classify_raw_seeds([10, 11, 12, 13, 14, 15], screening_count=2)

    assert grouped == {
        "type1_city": [10],
        "type2_open": [11],
        "type3_mountain": [12],
        "type4_village": [13],
        "type5_warehouse": [14],
        "type6_forest": [15],
    }
    assert [row["phase"] for row in manifest] == [
        "screening",
        "screening",
        "benchmark",
        "benchmark",
        "benchmark",
        "benchmark",
    ]


def test_summarize_run_computes_expected_averages(reload_module):
    mod = reload_module("scripts.stress_benchmark_compare")

    manifest = [
        {"index": 0, "seed": 100, "challenge_type": 1, "group": "type1_city", "phase": "screening"},
        {"index": 1, "seed": 200, "challenge_type": 2, "group": "type2_open", "phase": "screening"},
        {"index": 2, "seed": 300, "challenge_type": 3, "group": "type3_mountain", "phase": "benchmark"},
        {"index": 3, "seed": 400, "challenge_type": 4, "group": "type4_village", "phase": "benchmark"},
    ]
    summary = {
        "group_results": {
            "type1_city": [
                {"seed": 100, "score": 0.10, "success": True, "sim_time": 10.0, "wall_time": 1.0, "processing_wall_time": 1.0, "execution_status": "completed", "execution_ok": True}
            ],
            "type2_open": [
                {"seed": 200, "score": 0.30, "success": True, "sim_time": 20.0, "wall_time": 2.0, "processing_wall_time": 2.0, "execution_status": "completed", "execution_ok": True}
            ],
            "type3_mountain": [
                {"seed": 300, "score": 0.50, "success": False, "sim_time": 30.0, "wall_time": 3.0, "processing_wall_time": 3.0, "execution_status": "completed", "execution_ok": True}
            ],
            "type4_village": [
                {"seed": 400, "score": 0.90, "success": True, "sim_time": 40.0, "wall_time": 4.0, "processing_wall_time": 4.0, "execution_status": "completed", "execution_ok": True}
            ],
        },
        "run_metrics": {"avg_wall_per_seed_sec": 2.5},
        "execution_status_counts": {"completed": 4},
        "wall_clock_sec": 12.0,
    }

    metrics = mod._summarize_run(
        summary=summary,
        manifest=manifest,
        raw_seed_count=4,
        screening_count=2,
    )

    assert metrics["overall_avg_score"] == 0.45
    assert metrics["screening_avg_score"] == 0.20
    assert metrics["benchmark_avg_score"] == 0.70
    assert metrics["group_averages"]["type1_city"] == 0.10
    assert metrics["group_averages"]["type4_village"] == 0.90
    assert metrics["group_counts"]["type5_warehouse"] == 0
    assert metrics["overall_success_rate"] == 0.75


def test_main_writes_report_with_fake_benchmark(reload_module, monkeypatch, tmp_path):
    mod = reload_module("scripts.stress_benchmark_compare")

    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"fake")

    bundles = [
        (
            [10, 11, 12, 13, 14, 15],
            {
                "type1_city": [10],
                "type2_open": [11],
                "type3_mountain": [12],
                "type4_village": [13],
                "type5_warehouse": [14],
                "type6_forest": [15],
            },
            [
                {"index": 0, "seed": 10, "challenge_type": 1, "group": "type1_city", "phase": "screening"},
                {"index": 1, "seed": 11, "challenge_type": 2, "group": "type2_open", "phase": "benchmark"},
                {"index": 2, "seed": 12, "challenge_type": 3, "group": "type3_mountain", "phase": "benchmark"},
                {"index": 3, "seed": 13, "challenge_type": 4, "group": "type4_village", "phase": "benchmark"},
                {"index": 4, "seed": 14, "challenge_type": 5, "group": "type5_warehouse", "phase": "benchmark"},
                {"index": 5, "seed": 15, "challenge_type": 6, "group": "type6_forest", "phase": "benchmark"},
            ],
        ),
        (
            [20, 21, 22, 23, 24, 25],
            {
                "type1_city": [20],
                "type2_open": [21],
                "type3_mountain": [22],
                "type4_village": [23],
                "type5_warehouse": [24],
                "type6_forest": [25],
            },
            [
                {"index": 0, "seed": 20, "challenge_type": 1, "group": "type1_city", "phase": "screening"},
                {"index": 1, "seed": 21, "challenge_type": 2, "group": "type2_open", "phase": "benchmark"},
                {"index": 2, "seed": 22, "challenge_type": 3, "group": "type3_mountain", "phase": "benchmark"},
                {"index": 3, "seed": 23, "challenge_type": 4, "group": "type4_village", "phase": "benchmark"},
                {"index": 4, "seed": 24, "challenge_type": 5, "group": "type5_warehouse", "phase": "benchmark"},
                {"index": 5, "seed": 25, "challenge_type": 6, "group": "type6_forest", "phase": "benchmark"},
            ],
        ),
    ]

    monkeypatch.setattr(mod, "_infer_uid_from_model_path", lambda path: 178)
    monkeypatch.setattr(mod, "_build_seed_set", lambda count, rng, screening_count: bundles.pop(0))

    def _fake_run(argv: list[str]) -> None:
        seed_file = Path(argv[argv.index("--seed-file") + 1])
        summary_path = Path(argv[argv.index("--summary-json-out") + 1])
        grouped = json.loads(seed_file.read_text())
        group_results = {}
        for group, seeds in grouped.items():
            rows = []
            for seed in seeds:
                rows.append(
                    {
                        "seed": int(seed),
                        "score": float(seed) / 100.0,
                        "success": True,
                        "sim_time": 10.0,
                        "wall_time": 1.0,
                        "processing_wall_time": 1.0,
                        "execution_status": "completed",
                        "execution_ok": True,
                    }
                )
            group_results[group] = rows
        summary_path.write_text(
            json.dumps(
                {
                    "group_results": group_results,
                    "run_metrics": {"avg_wall_per_seed_sec": 1.0},
                    "execution_status_counts": {"completed": 6},
                    "wall_clock_sec": 6.0,
                }
            )
        )

    monkeypatch.setattr(mod, "_run_single_benchmark", _fake_run)

    run_dir = tmp_path / "stress"
    rc = mod.main(
        [
            "--model",
            str(model_path),
            "--repetitions",
            "2",
            "--seed-count",
            "6",
            "--workers",
            "1",
            "--run-dir",
            str(run_dir),
            "--seed-rng",
            "123",
        ]
    )

    assert rc == 0
    report = json.loads((run_dir / "report.json").read_text())
    assert report["aggregate"]["successful_runs"] == 2
    assert len(report["runs"]) == 2
    assert report["runs"][0]["metrics"]["overall_avg_score"] == 0.125
    assert report["runs"][1]["metrics"]["overall_avg_score"] == 0.225
    assert (run_dir / "report.txt").exists()
