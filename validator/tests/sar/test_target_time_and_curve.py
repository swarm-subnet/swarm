# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""The SAR reward curve: how the per-map target time is built, where the time term stops paying, and which failures still earn the participation floor."""
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
    """A SAR MapTask on a fixed map, with only the start and the search centre varied."""
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
    """The target time written out longhand: cruise over distance d, a 70% sweep of the search disc, the dwell, all scaled by the buffer."""
    sweep = 0.70 * math.pi * (SAR_SEARCH_RADIUS ** 2) / (SAR_SWEEP_WIDTH * SPEED_LIMIT)
    return SAR_TIME_TERM_BUFFER * (d / SPEED_LIMIT + sweep + SAR_DWELL_SEC)


def test_target_time_per_map_distances():
    """Target time follows the straight-line run to the search centre at every distance tried, matching the closed form exactly."""
    for d in (0.0, 5.0, 15.0, 30.0, 55.0):
        sx, sy = d, 0.0
        task = _task(start=(sx, sy, 1.5), search_centre=(0.0, 0.0))
        observed = _calculate_sar_target_time(task)
        expected = _expected(d)
        assert observed == pytest.approx(expected, abs=1e-6)


def test_time_term_plateau_until_target():
    """Finishing early pays no more than finishing on the target, so racing below it buys a miner nothing."""
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
    """Past the target the payout falls with elapsed time: on target beats the midpoint, which beats arriving at the horizon."""
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
    """Each of the six participation reasons, collision through timeout, still pays the 0.01 floor."""
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
    """A spawn failure at t=0 is not mistaken for an instant finish, it pays the 0.01 floor."""
    task = _task()
    r = flight_reward(
        success=False, t=0.0, horizon=HORIZON_SEC, task=task,
        failure_reason="SPAWN_FAILURE", sar_mode=True, min_clearance=None,
    )
    assert r == 0.01


def test_sar_collision_labeled_gives_participation():
    """A crash that arrives with its OBSTACLE_COLLISION label still pays the 0.01 floor, not nothing."""
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


def test_eval_error_returns_zero():
    """An EVAL_ERROR pays nothing at all, not even the participation floor."""
    task = _task()
    r = flight_reward(
        success=False, t=5.0, horizon=HORIZON_SEC, task=task,
        failure_reason="EVAL_ERROR", sar_mode=True,
        min_clearance=None,
    )
    assert r == 0.0
