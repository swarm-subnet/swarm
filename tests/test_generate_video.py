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


@pytest.mark.parametrize("challenge_type", [1, 2, 3, 4, 5, 6])
def test_build_task_supports_all_map_types(challenge_type: int) -> None:
    task = video_mod.build_task(seed=431600 + challenge_type, challenge_type=challenge_type)

    assert task.challenge_type == challenge_type
    assert task.horizon > 0.0
    assert C.SEARCH_RADIUS_MIN <= task.search_radius <= C.SEARCH_RADIUS_MAX
    assert isinstance(task.moving_platform, bool)


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
