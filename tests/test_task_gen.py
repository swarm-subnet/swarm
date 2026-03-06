from __future__ import annotations

import math
import random

import pytest

from swarm.constants import START_PLATFORM_TAKEOFF_BUFFER
from swarm.validator import task_gen


def test_get_type_params_returns_default_for_unknown_type():
    assert task_gen.get_type_params(999) == task_gen.TYPE_PARAMS[1]


def test_get_platform_height_for_mountain_and_village_is_zero():
    assert task_gen.get_platform_height_for_seed(123, challenge_type=3) == 0.0
    assert task_gen.get_platform_height_for_seed(123, challenge_type=4) == 0.0


def test_get_platform_height_respects_non_randomized_platform(monkeypatch):
    monkeypatch.setattr(task_gen, "START_PLATFORM", False)
    h = task_gen.get_platform_height_for_seed(42, challenge_type=1)
    assert h == task_gen.START_PLATFORM_SURFACE_Z


def test_random_start_for_warehouse_stays_within_bounds():
    rng = random.Random(1234)
    params = task_gen.get_type_params(5)
    x, y, z = task_gen._random_start(rng, params, challenge_type=5, seed=7)

    assert -params["world_range_x"] <= x <= params["world_range_x"]
    assert -params["world_range_y"] <= y <= params["world_range_y"]
    assert z >= params["start_h_min"] + START_PLATFORM_TAKEOFF_BUFFER
    assert z <= params["start_h_max"] + START_PLATFORM_TAKEOFF_BUFFER


def test_goal_from_start_warehouse_respects_distance_and_bounds():
    rng = random.Random(7)
    params = task_gen.get_type_params(5)
    start = (0.0, 0.0, 1.0)
    goal = task_gen._goal_from_start_warehouse(rng, start, params)
    gx, gy, gz = goal

    assert -params["world_range_x"] <= gx <= params["world_range_x"]
    assert -params["world_range_y"] <= gy <= params["world_range_y"]
    assert params["h_min"] <= gz <= params["h_max"]
    assert math.hypot(gx - start[0], gy - start[1]) >= params["r_min"]


def test_goal_from_start_type1_respects_world_bounds_and_min_distance():
    rng = random.Random(9)
    params = task_gen.get_type_params(1)
    start = (0.0, 0.0, 1.0)
    gx, gy, gz = task_gen._goal_from_start(rng, start, params, challenge_type=1, seed=1)

    assert -params["world_range"] <= gx <= params["world_range"]
    assert -params["world_range"] <= gy <= params["world_range"]
    assert params["h_min"] <= gz <= params["h_max"]
    assert math.hypot(gx - start[0], gy - start[1]) >= params["r_min"]


def test_random_task_is_deterministic_for_fixed_seed():
    t1 = task_gen.random_task(sim_dt=0.02, seed=12345)
    t2 = task_gen.random_task(sim_dt=0.02, seed=12345)
    assert t1 == t2


def test_random_task_can_be_forced_to_warehouse(monkeypatch):
    monkeypatch.setattr(task_gen, "CHALLENGE_TYPE_DISTRIBUTION", {5: 1.0})
    monkeypatch.setattr(task_gen, "MOVING_PLATFORM_PROB", {5: 0.0})
    task = task_gen.random_task(sim_dt=0.02, seed=111)

    params = task_gen.get_type_params(5)
    sx, sy, _ = task.start
    gx, gy, gz = task.goal
    assert task.challenge_type == 5
    assert -params["world_range_x"] <= sx <= params["world_range_x"]
    assert -params["world_range_x"] <= gx <= params["world_range_x"]
    assert -params["world_range_y"] <= sy <= params["world_range_y"]
    assert -params["world_range_y"] <= gy <= params["world_range_y"]
    assert params["h_min"] <= gz <= params["h_max"]
    assert task.moving_platform is False


def test_random_task_type3_uses_terrain_surface(monkeypatch):
    monkeypatch.setattr(task_gen, "CHALLENGE_TYPE_DISTRIBUTION", {3: 1.0})
    monkeypatch.setattr(task_gen, "MOVING_PLATFORM_PROB", {3: 0.0})
    task = task_gen.random_task(sim_dt=0.02, seed=321)

    sx, sy, sz = task.start
    gx, gy, gz = task.goal
    start_surface = task_gen._get_type3_surface_z(sx, sy, task.map_seed)
    goal_surface = task_gen._get_type3_surface_z(gx, gy, task.map_seed)

    assert task.challenge_type == 3
    assert pytest.approx(start_surface + START_PLATFORM_TAKEOFF_BUFFER, rel=1e-6, abs=1e-6) == sz
    assert pytest.approx(goal_surface, rel=1e-6, abs=1e-6) == gz
