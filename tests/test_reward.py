from __future__ import annotations

import math

import pytest

from swarm.constants import (
    HOVER_SEC,
    SAFETY_DISTANCE_DANGER,
    SAFETY_DISTANCE_SAFE,
    SPEED_LIMIT,
)
from swarm.protocol import MapTask
from swarm.validator.reward import (
    _calculate_safety_term,
    _calculate_target_time,
    _clamp,
    flight_reward,
)


def _sample_task():
    return MapTask(
        map_seed=1,
        start=(0.0, 0.0, 0.0),
        goal=(3.0, 4.0, 0.0),  # distance=5
        sim_dt=0.02,
        horizon=60.0,
        challenge_type=1,
    )


def test_clamp_bounds():
    assert _clamp(-1.0) == 0.0
    assert _clamp(2.0) == 1.0
    assert _clamp(0.3) == 0.3


def test_calculate_target_time_matches_formula():
    task = _sample_task()
    expected_min = (5.0 / SPEED_LIMIT) + HOVER_SEC
    expected = expected_min * 1.06
    assert math.isclose(_calculate_target_time(task), expected, rel_tol=1e-9)


def test_calculate_safety_term_thresholds_and_interpolation():
    assert _calculate_safety_term(min_clearance=2.0, collision=False) == 1.0
    assert _calculate_safety_term(min_clearance=0.0, collision=False) == 0.0
    assert _calculate_safety_term(min_clearance=0.5, collision=True) == 0.0

    mid = (SAFETY_DISTANCE_SAFE + SAFETY_DISTANCE_DANGER) / 2.0
    assert math.isclose(_calculate_safety_term(mid, collision=False), 0.5, rel_tol=1e-9)


def test_flight_reward_requires_positive_horizon():
    with pytest.raises(ValueError):
        flight_reward(success=True, t=1.0, horizon=0.0)


def test_flight_reward_collision_penalty_for_legitimate_model():
    assert flight_reward(success=True, t=1.0, horizon=10.0, collision=True, legitimate_model=True) == 0.01


def test_flight_reward_collision_zero_for_illegitimate_or_zero_time():
    assert flight_reward(success=True, t=1.0, horizon=10.0, collision=True, legitimate_model=False) == 0.0
    assert flight_reward(success=True, t=0.0, horizon=10.0, collision=True, legitimate_model=True) == 0.0


def test_flight_reward_failed_mission_returns_base_score_for_legitimate():
    assert flight_reward(success=False, t=1.0, horizon=10.0, legitimate_model=True) == 0.01


def test_flight_reward_failed_mission_zero_for_illegitimate():
    assert flight_reward(success=False, t=1.0, horizon=10.0, legitimate_model=False) == 0.0


def test_flight_reward_success_with_fast_time_and_safe_clearance_is_one():
    task = _sample_task()
    score = flight_reward(
        success=True,
        t=0.1,
        horizon=60.0,
        task=task,
        min_clearance=SAFETY_DISTANCE_SAFE + 0.1,
    )
    assert score == 1.0


def test_flight_reward_success_without_task_uses_linear_time_term():
    score = flight_reward(
        success=True,
        t=5.0,
        horizon=10.0,
        task=None,
        min_clearance=None,
    )
    # success=1.0, time=0.5, safety=1.0
    assert math.isclose(score, 0.45 + 0.45 * 0.5 + 0.10, rel_tol=1e-9)


def test_flight_reward_with_task_and_horizon_below_target_forces_zero_time_term():
    task = _sample_task()
    target = _calculate_target_time(task)
    score = flight_reward(
        success=True,
        t=target + 1.0,
        horizon=target - 0.1,
        task=task,
        min_clearance=SAFETY_DISTANCE_SAFE,
    )
    # success=1.0, time=0.0, safety=1.0
    assert math.isclose(score, 0.55, rel_tol=1e-9)


def test_flight_reward_uses_safety_interpolation():
    task = _sample_task()
    mid = (SAFETY_DISTANCE_SAFE + SAFETY_DISTANCE_DANGER) / 2.0
    score = flight_reward(
        success=True,
        t=0.1,
        horizon=60.0,
        task=task,
        min_clearance=mid,
    )
    # success=1, time=1, safety=0.5
    assert math.isclose(score, 0.45 + 0.45 + 0.05, rel_tol=1e-9)
