from __future__ import annotations

import math

import pytest

from swarm.constants import (
    HORIZON_SEC,
    SAR_DWELL_SEC,
    SAR_SEARCH_RADIUS,
    SAR_SWEEP_WIDTH,
    SAR_TIME_TERM_BUFFER,
    SPEED_LIMIT,
)
from swarm.protocol import MapTask
from swarm.validator.reward import (
    _calculate_sar_target_time,
    flight_reward,
)


def _task(start=(0.0, 0.0, 1.5), search_centre=(0.0, 0.0)):
    return MapTask(
        map_seed=1,
        start=start,
        goal=(8.0, 8.0, 1.5),
        sim_dt=1 / 30,
        horizon=HORIZON_SEC,
        challenge_type=2,
        version="5.0.0",
        search_centre=search_centre,
    )


def _expected(d):
    sweep = 0.70 * math.pi * (SAR_SEARCH_RADIUS ** 2) / (SAR_SWEEP_WIDTH * SPEED_LIMIT)
    return SAR_TIME_TERM_BUFFER * (d / SPEED_LIMIT + sweep + SAR_DWELL_SEC)


def test_target_time_per_map_distances():
    for d in (0.0, 5.0, 15.0, 30.0, 55.0):
        sx, sy = d, 0.0
        task = _task(start=(sx, sy, 1.5), search_centre=(0.0, 0.0))
        observed = _calculate_sar_target_time(task)
        expected = _expected(d)
        assert observed == pytest.approx(expected, abs=1e-6)


def test_time_term_plateau_until_target():
    task = _task(start=(0.0, 0.0, 1.5), search_centre=(0.0, 0.0))
    target = _calculate_sar_target_time(task)
    score_inside = flight_reward(
        success=True, t=target - 1.0, horizon=HORIZON_SEC, task=task,
        sar_mode=True, min_clearance=None,
    )
    score_at_target = flight_reward(
        success=True, t=target, horizon=HORIZON_SEC, task=task,
        sar_mode=True, min_clearance=None,
    )
    assert score_inside == pytest.approx(score_at_target, abs=1e-6)


def test_time_term_linear_decay_beyond_target():
    task = _task(start=(0.0, 0.0, 1.5), search_centre=(0.0, 0.0))
    target = _calculate_sar_target_time(task)
    midpoint = (target + HORIZON_SEC) / 2.0
    score_at_horizon = flight_reward(
        success=True, t=HORIZON_SEC, horizon=HORIZON_SEC, task=task,
        sar_mode=True, min_clearance=None,
    )
    score_mid = flight_reward(
        success=True, t=midpoint, horizon=HORIZON_SEC, task=task,
        sar_mode=True, min_clearance=None,
    )
    score_at_target = flight_reward(
        success=True, t=target, horizon=HORIZON_SEC, task=task,
        sar_mode=True, min_clearance=None,
    )
    assert score_at_target > score_mid > score_at_horizon


def test_participation_reward_per_failure_reason():
    task = _task()
    for reason in (
        "OBSTACLE_COLLISION", "NO_TOUCH_SPHERE", "INFEASIBLE",
        "SPAWN_FAILURE", "TILT", "TIMEOUT",
    ):
        r = flight_reward(
            success=False, t=5.0, horizon=HORIZON_SEC, task=task,
            failure_reason=reason, sar_mode=True, min_clearance=None,
        )
        assert r == 0.01, f"{reason} expected 0.01 got {r}"


def test_spawn_failure_t_zero_returns_participation():
    task = _task()
    r = flight_reward(
        success=False, t=0.0, horizon=HORIZON_SEC, task=task,
        failure_reason="SPAWN_FAILURE", sar_mode=True, min_clearance=None,
    )
    assert r == 0.01


def test_illegitimate_model_returns_zero_regardless_of_reason():
    task = _task()
    for reason in ("NONE", "INFEASIBLE", "EVAL_ERROR"):
        r = flight_reward(
            success=False, t=5.0, horizon=HORIZON_SEC, task=task,
            failure_reason=reason, legitimate_model=False, sar_mode=True,
            min_clearance=None,
        )
        assert r == 0.0


def test_sar_collision_labeled_gives_participation():
    task = _task()
    r = flight_reward(
        success=False, t=5.0, horizon=HORIZON_SEC, task=task,
        failure_reason="OBSTACLE_COLLISION", collision=True,
        sar_mode=True, min_clearance=None,
    )
    assert r == 0.01


def test_sar_collision_unlabeled_returns_zero():
    """Fail-closed: collision without a failure_reason label is treated as
    missed plumbing and scores zero so the bug is observable."""
    task = _task()
    r = flight_reward(
        success=False, t=5.0, horizon=HORIZON_SEC, task=task,
        failure_reason="NONE", collision=True,
        sar_mode=True, min_clearance=None,
    )
    assert r == 0.0


def test_eval_error_returns_zero_even_legitimate():
    task = _task()
    r = flight_reward(
        success=False, t=5.0, horizon=HORIZON_SEC, task=task,
        failure_reason="EVAL_ERROR", legitimate_model=True, sar_mode=True,
        min_clearance=None,
    )
    assert r == 0.0
