from __future__ import annotations

import math
from pathlib import Path

import pytest

from scripts import generate_video as video_mod
from swarm import constants as C


def test_build_parser_accepts_forest_type() -> None:
    args = video_mod._build_parser().parse_args(
        ["--model", "model/UID_178.zip", "--seed", "431623", "--type", "6"]
    )
    assert args.type == 6


def test_resolve_jobs_supports_seed_file(tmp_path: Path) -> None:
    seed_file = tmp_path / "seeds.json"
    seed_file.write_text(
        """{
  "type1_city": [101],
  "type2_open": [102],
  "type3_mountain": [103],
  "type4_village": [104],
  "type5_warehouse": [105],
  "type6_forest": [106]
}"""
    )

    args = video_mod._build_parser().parse_args(
        ["--model", "model/UID_178.zip", "--seed-file", str(seed_file)]
    )
    jobs = video_mod._resolve_jobs(args)

    assert [(job.seed, job.challenge_type) for job in jobs] == [
        (101, 1),
        (102, 2),
        (103, 3),
        (104, 4),
        (105, 5),
        (106, 6),
    ]


def test_expected_output_paths_match_video_filenames(tmp_path: Path) -> None:
    jobs = [video_mod.VideoJob(seed=42, challenge_type=1)]
    paths = video_mod._expected_output_paths(tmp_path, jobs, ["chase", "depth"])

    assert paths[jobs[0]] == [
        tmp_path / "seed42_city_chase.mp4",
        tmp_path / "seed42_city_depth.mp4",
    ]


def test_load_benchmark_expectations_reads_group_results(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        """{
  "group_results": {
    "type1_city": [{"seed": 101, "score": 0.5, "success": true, "sim_time": 12.0}],
    "type2_open": [{"seed": 102, "score": 0.6, "success": false, "sim_time": 8.0}],
    "type3_mountain": [{"seed": 103, "score": 0.01, "success": false, "sim_time": 0.4}],
    "type4_village": [{"seed": 104, "score": 0.01, "success": false, "sim_time": 14.0}],
    "type5_warehouse": [{"seed": 105, "score": 0.01, "success": false, "sim_time": 3.1}],
    "type6_forest": [{"seed": 106, "score": 0.01, "success": false, "sim_time": 27.0}]
  }
}"""
    )

    expectations = video_mod._load_benchmark_expectations(summary_path)

    assert expectations[(101, 1)] == video_mod.BenchmarkExpectation(True, 0.5, 12.0)
    assert expectations[(106, 6)] == video_mod.BenchmarkExpectation(False, 0.01, 27.0)


def test_assert_replay_matches_expected_accepts_exact_match() -> None:
    video_mod._assert_replay_matches_expected(
        job=video_mod.VideoJob(seed=101, challenge_type=1),
        expected=video_mod.BenchmarkExpectation(success=True, score=0.75, sim_time_sec=11.0),
        success=True,
        score=0.75,
        sim_time_sec=11.0,
    )


def test_assert_replay_matches_expected_rejects_mismatch() -> None:
    with pytest.raises(RuntimeError, match="Benchmark replay mismatch"):
        video_mod._assert_replay_matches_expected(
            job=video_mod.VideoJob(seed=101, challenge_type=1),
            expected=video_mod.BenchmarkExpectation(success=True, score=0.75, sim_time_sec=11.0),
            success=False,
            score=0.75,
            sim_time_sec=11.0,
        )


def test_infer_uid_from_model_path_reads_uid_digits() -> None:
    assert video_mod._infer_uid_from_model_path(Path("UID_178.zip")) == 178
    assert video_mod._infer_uid_from_model_path(Path("/tmp/model/uid-42.zip")) == 42


def test_video_benchmark_env_overrides_enable_generous_timeout_profile() -> None:
    overrides = video_mod._video_benchmark_env_overrides()

    assert overrides["SWARM_BATCH_TIMEOUT_MULT"] == "20.0"
    assert overrides["SWARM_BATCH_TIMEOUT_EXTEND_ON_PROGRESS"] == "1"
    assert overrides["SWARM_BATCH_TIMEOUT_HARD_CAP_SEC"] == "7200.0"


def test_status_labels_distinguish_failed_from_timeout() -> None:
    assert video_mod._outcome_label(True) == "SUCCESS"
    assert video_mod._outcome_label(False) == "FAILED"
    assert video_mod._summary_tag(success=True, verified=True) == "MATCH_OK"
    assert video_mod._summary_tag(success=False, verified=True) == "MATCH_FAIL"
    assert video_mod._summary_tag(success=False, verified=False) == "FAILED"


@pytest.mark.parametrize("challenge_type", [1, 2, 3, 4, 5, 6])
def test_build_task_supports_all_map_types(challenge_type: int) -> None:
    task = video_mod.build_task(seed=431600 + challenge_type, challenge_type=challenge_type)

    assert task.challenge_type == challenge_type
    assert task.horizon > 0.0
    assert C.SEARCH_RADIUS_MIN <= task.search_radius <= C.SEARCH_RADIUS_MAX
    assert isinstance(task.moving_platform, bool)


def test_build_task_uses_shared_moving_platform_resolver(monkeypatch) -> None:
    monkeypatch.setattr(
        video_mod,
        "resolve_moving_platform",
        lambda seed, challenge_type: seed == 404 and challenge_type == 2,
    )

    task = video_mod.build_task(seed=404, challenge_type=2)

    assert task.moving_platform is True


def test_build_task_warehouse_uses_current_world_ranges() -> None:
    task = video_mod.build_task(seed=323518, challenge_type=5)

    assert abs(task.start[0]) <= C.TYPE_4_WORLD_RANGE_X
    assert abs(task.start[1]) <= C.TYPE_4_WORLD_RANGE_Y
    assert abs(task.goal[0]) <= C.TYPE_4_WORLD_RANGE_X
    assert abs(task.goal[1]) <= C.TYPE_4_WORLD_RANGE_Y


def test_build_task_village_uses_village_behavior() -> None:
    task = video_mod.build_task(seed=323530, challenge_type=4)

    assert abs(task.start[0]) <= C.TYPE_3_VILLAGE_RANGE
    assert abs(task.start[1]) <= C.TYPE_3_VILLAGE_RANGE
    assert math.isclose(task.start[2], C.START_PLATFORM_TAKEOFF_BUFFER, rel_tol=0.0, abs_tol=1e-6)
    assert task.goal[2] == 0.0


def test_type_labels_include_forest() -> None:
    assert video_mod.TYPE_LABELS[6] == "forest"


def test_link_workspace_skips_when_no_onnx(tmp_path: Path) -> None:
    extracted = tmp_path / "submission"
    extracted.mkdir()
    (extracted / "drone_agent.py").write_text("class DroneFlightController: pass\n")

    video_mod._link_workspace(extracted)
