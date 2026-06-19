# swarm/validator/reward.py
"""Reward function for flight missions.

The score is a weighted combination of mission success, time efficiency, and safety::

    score = 0.45 * success_term + 0.45 * time_term + 0.10 * safety_term

where

* ``success_term`` is ``1`` if the mission reaches its goal and ``0``
  otherwise.
* ``time_term`` is based on minimum theoretical time with 6% buffer.
* ``safety_term`` is based on minimum obstacle clearance during flight.

All weights sum to one. The final score is clamped to ``[0, 1]``.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from swarm.protocol import MapTask

from swarm.constants import (
    HOVER_SEC,
    REWARD_W_SAFETY,
    REWARD_W_SUCCESS,
    REWARD_W_TIME,
    SAFETY_DISTANCE_DANGER,
    SAFETY_DISTANCE_SAFE,
    SAFETY_DISTANCE_SAFE_BY_TYPE,
    SAR_DWELL_SEC,
    SAR_SEARCH_RADIUS,
    SAR_SWEEP_WIDTH,
    SAR_TIME_TERM_BUFFER,
    SPEED_LIMIT,
    SWARM_CONGESTION_PER_NEIGHBOR_SEC,
)

__all__ = [
    "PARTICIPATION_REASONS",
    "PARTICIPATION_REWARD",
    "_calculate_safety_term",
    "_calculate_sar_target_time",
    "_calculate_swarm_sar_target_time",
    "_calculate_swarm_target_time",
    "_calculate_target_time",
    "_clamp",
    "_score_single_drone",
    "calculate_time_term",
    "flight_reward",
]


PARTICIPATION_REASONS = frozenset({
    "OBSTACLE_COLLISION",
    "NO_TOUCH_SPHERE",
    "INFEASIBLE",
    "SPAWN_FAILURE",
    "TILT",
    "TIMEOUT",
})
PARTICIPATION_REWARD = 0.01


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp *value* to the inclusive range [*lower*, *upper*]."""
    return max(lower, min(upper, value))


def _calculate_target_time(task: "MapTask") -> float:
    """Calculate target time based on distance and 6% buffer."""
    start_pos = np.array(task.start)
    goal_pos = np.array(task.goal)
    distance = np.linalg.norm(goal_pos - start_pos)

    min_time = (distance / SPEED_LIMIT) + HOVER_SEC
    return min_time * 1.06


def _calculate_sar_target_time(task: "MapTask") -> float:
    """Candidate-C SAR target time: travel + sweep + dwell, with buffer."""
    sc = getattr(task, "search_centre", None)
    if sc is None:
        sc = (0.0, 0.0)
    sx, sy = float(task.start[0]), float(task.start[1])
    d = math.hypot(sx - float(sc[0]), sy - float(sc[1]))
    sweep = 0.70 * math.pi * (SAR_SEARCH_RADIUS ** 2) / max(
        SAR_SWEEP_WIDTH * SPEED_LIMIT, 1e-6
    )
    return SAR_TIME_TERM_BUFFER * (
        d / max(SPEED_LIMIT, 1e-6) + sweep + SAR_DWELL_SEC
    )


def _calculate_safety_term(
    min_clearance: float, collision: bool, challenge_type: int = 0
) -> float:
    """Calculate safety term based on minimum obstacle clearance."""
    if collision:
        return 0.0
    safe = SAFETY_DISTANCE_SAFE_BY_TYPE.get(challenge_type, SAFETY_DISTANCE_SAFE)
    if min_clearance >= safe:
        return 1.0
    if min_clearance <= SAFETY_DISTANCE_DANGER:
        return 0.0
    return (min_clearance - SAFETY_DISTANCE_DANGER) / (safe - SAFETY_DISTANCE_DANGER)


def calculate_time_term(
    *,
    t: float,
    horizon: float,
    target_time: Optional[float],
) -> float:
    """Calculate a normalized time-efficiency term in ``[0, 1]``."""
    if target_time is None:
        return _clamp(1.0 - t / horizon)
    if t <= target_time:
        return 1.0
    if horizon <= target_time:
        return 0.0
    return _clamp(1.0 - (t - target_time) / (horizon - target_time))


def _calculate_swarm_sar_target_time(starts, search_centre, n_drones: int, search_radius: float) -> float:
    """Team SAR target time: travel from the start cluster to the clue, plus the
    area sweep divided across the swarm, plus the dwell. Dividing the sweep by the
    drone count means more drones are expected to find the victim faster."""
    pts = [np.asarray(s, dtype=float) for s in starts] or [np.zeros(2)]
    cx = float(np.mean([p[0] for p in pts]))
    cy = float(np.mean([p[1] for p in pts]))
    sc = search_centre if search_centre is not None else (0.0, 0.0)
    d = math.hypot(cx - float(sc[0]), cy - float(sc[1]))
    sweep = 0.70 * math.pi * (float(search_radius) ** 2) / max(SAR_SWEEP_WIDTH * SPEED_LIMIT, 1e-6)
    return SAR_TIME_TERM_BUFFER * (
        d / max(SPEED_LIMIT, 1e-6) + sweep / max(int(n_drones), 1) + SAR_DWELL_SEC
    )


def _calculate_swarm_target_time(start, goal, n_congested: int) -> float:
    """Straight-line autopilot target time plus a per-neighbour congestion slack
    so a drone that detours to deconflict is not punished as merely slow."""
    distance = float(np.linalg.norm(np.asarray(goal, dtype=float) - np.asarray(start, dtype=float)))
    base = (distance / SPEED_LIMIT) + HOVER_SEC
    return base * 1.06 + SWARM_CONGESTION_PER_NEIGHBOR_SEC * int(n_congested)


def _score_single_drone(
    *,
    success: bool,
    t: float,
    horizon: float,
    target_time: Optional[float],
    min_clearance: Optional[float],
    collision: bool,
    challenge_type: int,
    legitimate_model: bool,
    failure_reason: str,
) -> float:
    """Per-drone autopilot score (0.45 success + 0.45 time + 0.10 safety).

    Numerically identical to AutopilotChallengeFamily.normalize_rollout_metrics
    for one drone; the swarm family averages this over its drones.
    """
    if not legitimate_model or failure_reason == "EVAL_ERROR":
        return 0.0
    if not success:
        return PARTICIPATION_REWARD if t > 0.0 else 0.0
    if collision:
        return PARTICIPATION_REWARD if t > 0.0 else 0.0

    time_term = calculate_time_term(t=t, horizon=horizon, target_time=target_time)
    if min_clearance is not None:
        safety_term = _calculate_safety_term(float(min_clearance), collision=False, challenge_type=challenge_type)
    else:
        safety_term = 1.0
    return _clamp((0.45 * 1.0) + (0.45 * time_term) + (0.10 * safety_term))


def flight_reward(
    success: bool,
    t: float,
    horizon: float,
    task: Optional["MapTask"] = None,
    *,
    min_clearance: Optional[float] = None,
    collision: bool = False,
    w_success: float = REWARD_W_SUCCESS,
    w_t: float = REWARD_W_TIME,
    w_safety: float = REWARD_W_SAFETY,
    legitimate_model: bool = True,
    failure_reason: str = "NONE",
    sar_mode: bool = False,
) -> float:
    """Compute the reward for a single flight mission.

    Parameters
    ----------
    success
        ``True`` if the mission successfully reached its objective.
    t
        Time (in seconds) taken to complete the mission.
    horizon
        Maximum time allowed to complete the mission.
    task
        MapTask object containing start and goal positions for distance calculation.
    min_clearance
        Minimum distance (meters) to any obstacle during flight. If None, safety
        term is set to 1.0 (full score).
    collision
        ``True`` if the drone collided with an obstacle. Forces safety term to 0.
    w_success, w_t, w_safety
        Weights for success, time, and safety terms. They should sum to ``1``.
    legitimate_model
        ``True`` if the model passed verification. Legitimate models that fail
        missions receive a base reward of 0.01.

    Returns
    -------
    float
        A score in the range ``[0, 1]``.
    """

    if horizon <= 0:
        raise ValueError("'horizon' must be positive")

    if not legitimate_model or failure_reason == "EVAL_ERROR":
        return 0.0

    if not success:
        if failure_reason in PARTICIPATION_REASONS:
            return PARTICIPATION_REWARD
        if not sar_mode and legitimate_model and t > 0.0:
            return PARTICIPATION_REWARD
        return 0.0

    if collision:
        if legitimate_model and t > 0.0:
            return PARTICIPATION_REWARD
        return 0.0

    success_term = 1.0

    target_time = None
    if task is not None:
        target_time = (
            _calculate_sar_target_time(task)
            if sar_mode
            else _calculate_target_time(task)
        )
    time_term = calculate_time_term(t=t, horizon=horizon, target_time=target_time)

    challenge_type = getattr(task, "challenge_type", 0) if task is not None else 0
    if min_clearance is not None:
        safety_term = _calculate_safety_term(min_clearance, collision, challenge_type)
    else:
        safety_term = 1.0 if not collision else 0.0

    score = (w_success * success_term) + (w_t * time_term) + (w_safety * safety_term)
    return _clamp(score)
