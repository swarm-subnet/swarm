from __future__ import annotations

import json
from pathlib import Path

from scripts import benchmark_with_videos as runner
from scripts import generate_video


def test_runner_invokes_benchmark_then_video(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")
    run_dir = tmp_path / "artifacts"
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        "swarm.benchmark.engine.main",
        lambda argv: captured.setdefault("benchmark", list(argv)),
    )
    monkeypatch.setattr(
        "scripts.generate_video.main",
        lambda argv: captured.setdefault("video", list(argv)),
    )

    rc = runner.main(
        [
            "--model",
            str(model_path),
            "--run-dir",
            str(run_dir),
        ]
    )

    assert rc == 0
    assert captured["benchmark"] == [
        "--model",
        str(model_path.resolve()),
        "--workers",
        "1",
        "--log-out",
        str((run_dir / "benchmark.log").resolve()),
        "--seeds-per-group",
        "1",
        "--save-seed-file",
        str((run_dir / "seeds.json").resolve()),
        "--summary-json-out",
        str((run_dir / "summary.json").resolve()),
        "--rpc-verbosity",
        "mid",
    ]
    assert captured["video"] == [
        "--model",
        str(model_path.resolve()),
        "--seed-file",
        str((run_dir / "seeds.json").resolve()),
        "--summary-json",
        str((run_dir / "summary.json").resolve()),
        "--backend",
        "benchmark",
        "--mode",
        "chase",
        "--out",
        str((run_dir / "videos").resolve()),
        "--width",
        "960",
        "--height",
        "540",
        "--fps",
        "15",
        "--chase-back",
        "2.5",
        "--chase-up",
        "1.0",
        "--chase-fov",
        "65.0",
        "--fpv-fov",
        "90.0",
        "--overview-fov",
        "55.0",
        "--save-actions",
        str((run_dir / "actions").resolve()),
    ]


def test_runner_forwards_seed_file_and_skip_existing(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")
    seed_file = tmp_path / "input_seeds.json"
    seed_file.write_text("{}")
    captured: dict[str, list[str]] = {}

    monkeypatch.setattr(
        "swarm.benchmark.engine.main",
        lambda argv: captured.setdefault("benchmark", list(argv)),
    )
    monkeypatch.setattr(
        "scripts.generate_video.main",
        lambda argv: captured.setdefault("video", list(argv)),
    )

    rc = runner.main(
        [
            "--model",
            str(model_path),
            "--run-dir",
            str(tmp_path / "artifacts"),
            "--seed-file",
            str(seed_file),
            "--skip-existing",
        ]
    )

    assert rc == 0
    assert "--seed-file" in captured["benchmark"]
    assert str(seed_file.resolve()) in captured["benchmark"]
    assert "--skip-existing" in captured["video"]


def test_runner_only_type_filters_benchmark_and_replays_in_batch(
    monkeypatch, tmp_path: Path
) -> None:
    model_path = tmp_path / "UID_178.zip"
    model_path.write_bytes(b"zip")
    run_dir = tmp_path / "artifacts"
    captured: dict[str, list[str]] = {}
    filtered_benchmark_calls: list[dict[str, object]] = []
    discovered_seeds = [401, 402]

    def _fake_filtered_benchmark(**kwargs) -> None:
        filtered_benchmark_calls.append(dict(kwargs))
        Path(kwargs["saved_seed_file"]).write_text(
            json.dumps(
                {
                    "type4_village": [401, 402],
                }
            )
        )
        Path(kwargs["summary_json_out"]).write_text(
            json.dumps(
                {
                    "group_results": {
                        "type4_village": [
                            {"seed": 401, "score": 0.01, "success": False, "sim_time": 10.0},
                            {"seed": 402, "score": 0.01, "success": False, "sim_time": 11.0},
                        ]
                    }
                }
            )
        )

    monkeypatch.setattr(runner, "_run_filtered_benchmark", _fake_filtered_benchmark)
    monkeypatch.setattr(
        runner,
        "_find_seeds_for_type",
        lambda **kwargs: list(discovered_seeds),
    )
    monkeypatch.setattr(
        "scripts.generate_video.main",
        lambda argv: captured.setdefault("video", list(argv)),
    )

    rc = runner.main(
        [
            "--model",
            str(model_path),
            "--run-dir",
            str(run_dir),
                "--only-type",
                "4",
                "--seeds-per-group",
                "2",
                "--mode",
                "chase",
            ]
        )

    assert rc == 0
    assert filtered_benchmark_calls == [
        {
            "model": model_path.resolve(),
                "uid": None,
                "workers": 1,
                "challenge_type": 4,
                "seeds": discovered_seeds,
                "rpc_verbosity": "mid",
                "relax_timeouts": False,
                "log_out": (run_dir / "benchmark.log").resolve(),
                "saved_seed_file": (run_dir / "seeds.json").resolve(),
            "summary_json_out": (run_dir / "summary.json").resolve(),
        }
    ]
    assert captured["video"] == [
        "--model",
        str(model_path.resolve()),
        "--seed-file",
        str((run_dir / "seeds.json").resolve()),
        "--summary-json",
        str((run_dir / "summary.json").resolve()),
        "--backend",
        "benchmark",
        "--mode",
        "chase",
        "--out",
        str((run_dir / "videos").resolve()),
        "--width",
        "960",
        "--height",
        "540",
        "--fps",
        "15",
        "--chase-back",
        "2.5",
        "--chase-up",
        "1.0",
        "--chase-fov",
        "65.0",
        "--fpv-fov",
        "90.0",
        "--overview-fov",
        "55.0",
        "--save-actions",
        str((run_dir / "actions").resolve()),
    ]


def test_generate_video_accepts_subset_seed_file(tmp_path: Path) -> None:
    seed_file = tmp_path / "subset_seeds.json"
    seed_file.write_text(json.dumps({"type4_village": [401, 402]}))

    jobs = generate_video._load_seed_jobs(seed_file)

    assert [(job.seed, job.challenge_type) for job in jobs] == [(401, 4), (402, 4)]


def test_runner_fails_fast_for_missing_model(tmp_path: Path) -> None:
    rc = runner.main(["--model", str(tmp_path / "missing.zip")])

    assert rc == 1


def test_resolve_run_dir_appends_timestamp_when_directory_exists(
    monkeypatch, tmp_path: Path
) -> None:
    existing = tmp_path / "uid178_bench_video_run"
    existing.mkdir()
    monkeypatch.setattr(runner, "_timestamp_suffix", lambda: "20260320_111111")

    resolved = runner._resolve_run_dir(existing)

    assert resolved == tmp_path / "uid178_bench_video_run_20260320_111111"


def test_resolve_run_dir_preserves_new_requested_directory(tmp_path: Path) -> None:
    requested = tmp_path / "uid178_bench_video_run"

    resolved = runner._resolve_run_dir(requested)

    assert resolved == requested.resolve()
