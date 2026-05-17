from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from swarm.constants import (
    DRONE_HULL_RADIUS,
    LANDING_ALTITUDE_BUFFER,
    LANDING_COLUMN_PADDING,
    LANDING_FLOOR_MAX_HEIGHT,
    LANDING_PLATFORM_RADIUS,
    SAFETY_DISTANCE_SAFE,
    TYPE_6_SAFETY_DISTANCE_SAFE,
)
from swarm.core import moving_drone as moving_drone_mod


LANDING_R = LANDING_PLATFORM_RADIUS + DRONE_HULL_RADIUS + LANDING_COLUMN_PADDING


def _make_env(
    *,
    challenge_type: int,
    goal: tuple,
    platform_pos=None,
    aabb=None,
    cli: int = 7,
):
    env = moving_drone_mod.MovingDroneAviary.__new__(moving_drone_mod.MovingDroneAviary)
    env.task = SimpleNamespace(challenge_type=challenge_type)
    env.GOAL_POS = np.asarray(goal, dtype=float)
    env._current_platform_pos = (
        np.asarray(platform_pos, dtype=float) if platform_pos is not None else env.GOAL_POS.copy()
    )
    env.CLIENT = cli
    env._aabb = aabb if aabb is not None else ((-50.0, -50.0, -0.05), (50.0, 50.0, 0.05))
    return env


def _patch_aabb(monkeypatch, env):
    def _fake_aabb(uid, physicsClientId=None):
        return env._aabb
    monkeypatch.setattr(moving_drone_mod.p, "getAABB", _fake_aabb)


def test_city_low_platform_flat_road_below_is_skipped(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(10.0, 5.0, 0.6),
        aabb=((-40.0, -40.0, -0.002), (40.0, 40.0, 0.102)),
    )
    _patch_aabb(monkeypatch, env)
    assert env._is_landing_floor_body(99, drone_pos=np.array([10.05, 5.05, 0.7]))


def test_forest_low_platform_ground_box_is_skipped(monkeypatch) -> None:
    platform_z = 0.4
    env = _make_env(
        challenge_type=6,
        goal=(0.0, 0.0, platform_z),
        aabb=((-50.0, -50.0, -0.10), (50.0, 50.0, 0.0)),
    )
    _patch_aabb(monkeypatch, env)
    assert platform_z < TYPE_6_SAFETY_DISTANCE_SAFE
    assert env._is_landing_floor_body(42, drone_pos=np.array([0.1, -0.1, 0.5]))


def test_warehouse_low_platform_thin_floor_is_skipped(monkeypatch) -> None:
    env = _make_env(
        challenge_type=5,
        goal=(-3.0, 2.0, 0.5),
        aabb=((-3.4, 1.6, 0.0), (-2.6, 2.4, 0.024)),
    )
    _patch_aabb(monkeypatch, env)
    assert env._is_landing_floor_body(91, drone_pos=np.array([-3.0, 2.0, 0.6]))


def test_village_low_platform_flat_ground_is_skipped(monkeypatch) -> None:
    env = _make_env(
        challenge_type=4,
        goal=(5.0, -2.0, 0.3),
        aabb=((-25.0, -25.0, -0.05), (25.0, 25.0, 0.05)),
    )
    _patch_aabb(monkeypatch, env)
    assert env._is_landing_floor_body(17, drone_pos=np.array([5.1, -1.9, 0.4]))


def test_wall_under_low_platform_is_counted(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(10.0, 5.0, 0.6),
        aabb=((9.4, 4.4, 0.0), (10.6, 5.6, 0.59)),
    )
    _patch_aabb(monkeypatch, env)
    assert not env._is_landing_floor_body(7, drone_pos=np.array([10.0, 5.0, 0.7]))


def test_floor_during_cruise_is_counted(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(10.0, 5.0, 0.6),
        aabb=((-40.0, -40.0, -0.002), (40.0, 40.0, 0.102)),
    )
    _patch_aabb(monkeypatch, env)
    cruise_z = 0.6 + SAFETY_DISTANCE_SAFE + LANDING_ALTITUDE_BUFFER + 0.001
    assert not env._is_landing_floor_body(99, drone_pos=np.array([10.0, 5.0, cruise_z]))


def test_floor_outside_landing_column_is_counted(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(10.0, 5.0, 0.6),
        aabb=((-40.0, -40.0, -0.002), (40.0, 40.0, 0.102)),
    )
    _patch_aabb(monkeypatch, env)
    drone = np.array([10.0 + LANDING_R + 0.05, 5.0, 0.7])
    assert not env._is_landing_floor_body(99, drone_pos=drone)


def test_high_platform_warehouse_is_not_suppressed(monkeypatch) -> None:
    env = _make_env(
        challenge_type=5,
        goal=(0.0, 0.0, 4.0),
        aabb=((-10.0, -10.0, -0.05), (10.0, 10.0, 0.05)),
    )
    _patch_aabb(monkeypatch, env)
    assert not env._is_landing_floor_body(1, drone_pos=np.array([0.0, 0.0, 4.1]))


def test_ceiling_body_above_platform_is_counted(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(2.0, 2.0, 0.5),
        aabb=((1.5, 1.5, 0.6), (2.5, 2.5, 0.7)),
    )
    _patch_aabb(monkeypatch, env)
    assert not env._is_landing_floor_body(3, drone_pos=np.array([2.0, 2.0, 0.5]))


@pytest.mark.parametrize("challenge_type", [2, 3])
def test_non_eligible_challenge_type_never_suppressed(
    monkeypatch, challenge_type
) -> None:
    env = _make_env(
        challenge_type=challenge_type,
        goal=(0.0, 0.0, 0.5),
        aabb=((-50.0, -50.0, -0.05), (50.0, 50.0, 0.05)),
    )
    _patch_aabb(monkeypatch, env)
    assert not env._is_landing_floor_body(1, drone_pos=np.array([0.0, 0.0, 0.6]))


def test_floor_off_to_side_not_skipped_when_drone_over_pad(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(10.0, 5.0, 0.6),
        aabb=(
            (10.0 + LANDING_R + 0.5, 5.0 - 0.2, -0.02),
            (10.0 + LANDING_R + 1.5, 5.0 + 0.2, 0.05),
        ),
    )
    _patch_aabb(monkeypatch, env)
    assert not env._is_landing_floor_body(8, drone_pos=np.array([10.0, 5.0, 0.7]))


def test_flatness_exactly_at_threshold_is_skipped(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(0.0, 0.0, 0.6),
        aabb=((-10.0, -10.0, 0.0), (10.0, 10.0, LANDING_FLOOR_MAX_HEIGHT)),
    )
    _patch_aabb(monkeypatch, env)
    assert env._is_landing_floor_body(1, drone_pos=np.array([0.0, 0.0, 0.7]))


def test_flatness_just_above_threshold_is_counted(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(0.0, 0.0, 0.6),
        aabb=(
            (-10.0, -10.0, 0.0),
            (10.0, 10.0, LANDING_FLOOR_MAX_HEIGHT + 0.001),
        ),
    )
    _patch_aabb(monkeypatch, env)
    assert not env._is_landing_floor_body(1, drone_pos=np.array([0.0, 0.0, 0.7]))


def test_current_platform_pos_none_falls_back_to_goal_pos(monkeypatch) -> None:
    env = _make_env(challenge_type=1, goal=(0.0, 0.0, 0.6))
    env._current_platform_pos = None
    _patch_aabb(monkeypatch, env)
    assert env._is_landing_floor_body(1, drone_pos=np.array([0.0, 0.0, 0.7]))


def test_moving_platform_position_overrides_goal_pos(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(0.0, 0.0, 0.6),
        platform_pos=(3.0, 4.0, 0.6),
        aabb=((-40.0, -40.0, -0.002), (40.0, 40.0, 0.102)),
    )
    _patch_aabb(monkeypatch, env)
    assert env._is_landing_floor_body(1, drone_pos=np.array([3.05, 4.05, 0.7]))
    assert not env._is_landing_floor_body(1, drone_pos=np.array([0.0, 0.0, 0.7]))


def test_body_flush_with_platform_top_is_counted(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(0.0, 0.0, 0.6),
        aabb=((-10.0, -10.0, 0.5), (10.0, 10.0, 0.6)),
    )
    _patch_aabb(monkeypatch, env)
    assert not env._is_landing_floor_body(1, drone_pos=np.array([0.0, 0.0, 0.7]))


def test_body_too_far_below_platform_is_counted(monkeypatch) -> None:
    env = _make_env(
        challenge_type=1,
        goal=(0.0, 0.0, 0.95),
        aabb=((-10.0, -10.0, -0.10), (10.0, 10.0, -0.05)),
    )
    _patch_aabb(monkeypatch, env)
    assert (env._current_platform_pos[2] - (-0.05)) >= SAFETY_DISTANCE_SAFE
    assert not env._is_landing_floor_body(1, drone_pos=np.array([0.0, 0.0, 1.0]))


def test_forest_altitude_gate_uses_type_six_safe_distance(monkeypatch) -> None:
    env = _make_env(
        challenge_type=6,
        goal=(0.0, 0.0, 0.4),
        aabb=((-50.0, -50.0, -0.10), (50.0, 50.0, 0.0)),
    )
    _patch_aabb(monkeypatch, env)
    boundary = 0.4 + TYPE_6_SAFETY_DISTANCE_SAFE + LANDING_ALTITUDE_BUFFER
    assert env._is_landing_floor_body(1, drone_pos=np.array([0.0, 0.0, boundary]))
    assert not env._is_landing_floor_body(
        1, drone_pos=np.array([0.0, 0.0, boundary + 0.001])
    )


def test_update_min_clearance_skips_floor_body(monkeypatch) -> None:
    env = moving_drone_mod.MovingDroneAviary.__new__(moving_drone_mod.MovingDroneAviary)
    env.task = SimpleNamespace(challenge_type=1)
    env.GOAL_POS = np.array([0.0, 0.0, 0.6])
    env._current_platform_pos = env.GOAL_POS.copy()
    env.CLIENT = 7
    env.DRONE_IDS = np.array([100], dtype=np.int64)
    env.PLANE_ID = 0
    env._end_platform_uids = []
    env._start_platform_uids = []
    env._collision = False
    env._min_clearance_episode = SAFETY_DISTANCE_SAFE
    env.pos = np.array([[0.0, 0.0, 0.7]], dtype=float)

    floor_aabb = ((-50.0, -50.0, -0.002), (50.0, 50.0, 0.102))
    wall_aabb = ((0.4, -0.5, 0.0), (0.5, 0.5, 1.5))

    def _fake_aabb(uid, physicsClientId=None):
        if uid == env.DRONE_IDS[0]:
            return ((-0.1, -0.1, 0.6), (0.1, 0.1, 0.8))
        if uid == 200:
            return floor_aabb
        if uid == 300:
            return wall_aabb
        raise AssertionError(f"unexpected uid {uid}")

    monkeypatch.setattr(moving_drone_mod.p, "getAABB", _fake_aabb)
    monkeypatch.setattr(
        moving_drone_mod.p,
        "getOverlappingObjects",
        lambda mn, mx, physicsClientId=None: [(200, -1), (300, -1)],
    )

    closest_calls: list[int] = []

    def _fake_closest_points(bodyA, bodyB, distance, physicsClientId=None):
        closest_calls.append(int(bodyB))
        return [(0, bodyA, bodyB, (0, 0, 0), (0, 0, 0), (0, 0, 1), 0.4, 0.4, 0.42)]

    monkeypatch.setattr(moving_drone_mod.p, "getClosestPoints", _fake_closest_points)

    env._update_min_clearance()

    assert closest_calls == [300]
    assert env._min_clearance_episode == pytest.approx(0.42)


def test_update_min_clearance_counts_floor_outside_eligible_type(monkeypatch) -> None:
    env = moving_drone_mod.MovingDroneAviary.__new__(moving_drone_mod.MovingDroneAviary)
    env.task = SimpleNamespace(challenge_type=3)
    env.GOAL_POS = np.array([0.0, 0.0, 0.6])
    env._current_platform_pos = env.GOAL_POS.copy()
    env.CLIENT = 7
    env.DRONE_IDS = np.array([100], dtype=np.int64)
    env.PLANE_ID = 0
    env._end_platform_uids = []
    env._start_platform_uids = []
    env._collision = False
    env._min_clearance_episode = SAFETY_DISTANCE_SAFE
    env.pos = np.array([[0.0, 0.0, 0.7]], dtype=float)

    def _fake_aabb(uid, physicsClientId=None):
        if uid == env.DRONE_IDS[0]:
            return ((-0.1, -0.1, 0.6), (0.1, 0.1, 0.8))
        return ((-50.0, -50.0, -0.002), (50.0, 50.0, 0.102))

    monkeypatch.setattr(moving_drone_mod.p, "getAABB", _fake_aabb)
    monkeypatch.setattr(
        moving_drone_mod.p,
        "getOverlappingObjects",
        lambda mn, mx, physicsClientId=None: [(400, -1)],
    )

    closest_calls: list[int] = []

    def _fake_closest_points(bodyA, bodyB, distance, physicsClientId=None):
        closest_calls.append(int(bodyB))
        return [(0, bodyA, bodyB, (0, 0, 0), (0, 0, 0), (0, 0, 1), 0.5, 0.5, 0.55)]

    monkeypatch.setattr(moving_drone_mod.p, "getClosestPoints", _fake_closest_points)

    env._update_min_clearance()

    assert closest_calls == [400]
    assert env._min_clearance_episode == pytest.approx(0.55)
