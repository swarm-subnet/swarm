from __future__ import annotations

import math

import pybullet as p

from swarm.constants import (
    SIM_DT,
    TYPE_1_R_MAX,
    TYPE_1_R_MIN,
)
from swarm.constants import START_PLATFORM_TAKEOFF_BUFFER
from swarm.core.env_builder import build as build_mod
from swarm.validator.task_gen import task_for_seed_and_type


class _DummyPyBullet:
    GEOM_CYLINDER = 1
    VISUAL_SHAPE_DOUBLE_SIDED = 2

    def __init__(self) -> None:
        self._next_uid = 10

    def getNumBodies(self, physicsClientId: int) -> int:
        _ = physicsClientId
        return 0

    def createCollisionShape(self, *args, **kwargs) -> int:
        _ = args, kwargs
        self._next_uid += 1
        return self._next_uid

    def createVisualShape(self, *args, **kwargs) -> int:
        _ = args, kwargs
        self._next_uid += 1
        return self._next_uid

    def createMultiBody(self, *args, **kwargs) -> int:
        _ = args, kwargs
        self._next_uid += 1
        return self._next_uid

    def changeDynamics(self, *args, **kwargs) -> None:
        _ = args, kwargs
        return None

    def changeVisualShape(self, *args, **kwargs) -> None:
        _ = args, kwargs
        return None

    def loadTexture(self, *args, **kwargs) -> int:
        _ = args, kwargs
        self._next_uid += 1
        return self._next_uid


def test_build_world_reuses_cleared_forest_start_surface(monkeypatch) -> None:
    dummy_p = _DummyPyBullet()
    start = (1.0, 2.0, 3.0)
    cleared_surface_z = 12.5

    monkeypatch.setattr(build_mod.shared, "p", dummy_p)
    monkeypatch.setattr(build_mod, "_build_static_world", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        build_mod,
        "_find_clear_platform_position",
        lambda *args, **kwargs: (7.0, 8.0, cleared_surface_z),
    )

    def _unexpected_random_height(seed: int, challenge_type: int) -> float:
        raise AssertionError(
            f"forest start platform height should come from the cleared surface, got random helper call for seed={seed} type={challenge_type}"
        )

    monkeypatch.setattr(build_mod.shared, "get_platform_height_for_seed", _unexpected_random_height)

    result = build_mod.build_world(
        seed=123,
        cli=99,
        start=start,
        goal=None,
        challenge_type=6,
    )

    _, start_platform_uids, start_surface_z, _, adjusted_start, _ = result

    assert len(start_platform_uids) == 3
    assert start_surface_z == cleared_surface_z
    assert adjusted_start == (7.0, 8.0, cleared_surface_z + START_PLATFORM_TAKEOFF_BUFFER)


def test_build_world_goal_relocation_uses_type_distance_bounds(monkeypatch) -> None:
    dummy_p = _DummyPyBullet()
    calls: list[dict[str, object]] = []

    def _capture_position(*args, **kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            return (7.0, 8.0, 12.5)
        return (9.0, 10.0, 0.5)

    monkeypatch.setattr(build_mod.shared, "p", dummy_p)
    monkeypatch.setattr(build_mod.shared, "START_PLATFORM", True)
    monkeypatch.setattr(build_mod.shared, "PLATFORM", False)
    monkeypatch.setattr(build_mod, "_build_static_world", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_mod, "_find_clear_platform_position", _capture_position)

    build_mod.build_world(
        seed=123,
        cli=99,
        start=(1.0, 2.0, 3.0),
        goal=(4.0, 5.0, 0.6),
        challenge_type=1,
    )

    assert len(calls) == 2
    goal_call = calls[1]
    assert goal_call["avoid_pos"] == (7.0, 8.0, 12.5 + START_PLATFORM_TAKEOFF_BUFFER)
    assert goal_call["min_distance"] == TYPE_1_R_MIN
    assert goal_call["required_distance_min"] == TYPE_1_R_MIN
    assert goal_call["required_distance_max"] == TYPE_1_R_MAX
    assert goal_call["preferred_distance"] == math.hypot(4.0 - 1.0, 5.0 - 2.0)
    assert goal_call["distance_mode"] == "xy"
    assert goal_call["allow_candidate_fallback"] is False


def test_city_adjusted_goal_distance_stays_within_type_bounds(monkeypatch) -> None:
    problematic_seeds = [2, 31, 57, 74, 85, 98, 113]
    for seed in problematic_seeds:
        task = task_for_seed_and_type(sim_dt=SIM_DT, seed=seed, challenge_type=1)
        cli = p.connect(p.DIRECT)
        try:
            _, _, _, _, adjusted_start, adjusted_goal = build_mod.build_world(
                seed=task.map_seed,
                cli=cli,
                start=task.start,
                goal=task.goal,
                challenge_type=task.challenge_type,
            )
        finally:
            p.disconnect(cli)

        start = adjusted_start or task.start
        goal = adjusted_goal or task.goal
        dist_xy = math.hypot(goal[0] - start[0], goal[1] - start[1])
        assert TYPE_1_R_MIN <= dist_xy <= TYPE_1_R_MAX, (
            f"seed={seed} adjusted city distance escaped bounds: {dist_xy:.3f}m "
            f"start={start} goal={goal}"
        )


def test_build_world_village_skips_generic_platform_relocation(monkeypatch) -> None:
    dummy_p = _DummyPyBullet()

    def _unexpected_find_clear(*args, **kwargs):
        raise AssertionError("village should not use generic platform relocation")

    monkeypatch.setattr(build_mod.shared, "p", dummy_p)
    monkeypatch.setattr(build_mod.shared, "START_PLATFORM", True)
    monkeypatch.setattr(build_mod.shared, "PLATFORM", False)
    monkeypatch.setattr(build_mod, "_build_static_world", lambda *args, **kwargs: None)
    monkeypatch.setattr(build_mod, "_find_clear_platform_position", _unexpected_find_clear)
    monkeypatch.setattr(build_mod, "_raycast_surface_z", lambda cli, x, y: 0.25)

    _, _, start_surface_z, goal_surface_z, adjusted_start, adjusted_goal = build_mod.build_world(
        seed=123,
        cli=99,
        start=(1.0, 2.0, 0.121),
        goal=(4.0, 5.0, 0.0),
        challenge_type=4,
    )

    assert start_surface_z == 0.25
    assert goal_surface_z == 0.25
    assert adjusted_start is None
    assert adjusted_goal is None
