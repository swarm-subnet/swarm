# swarm/validator/reward.py
"""Reward function for flight missions.

The score is a weighted combination of mission success and time efficiency::

    score = 0.50 * success_term + 0.50 * time_term

where

* ``success_term`` is ``1`` if the mission reaches its goal and ``0``
  otherwise.
* ``time_term`` is based on minimum theoretical time with 2% buffer.

Both weights sum to one. The final score is clamped to ``[0, 1]``.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from swarm.protocol import MapTask

from swarm.constants import SPEED_LIMIT, STABLE_LANDING_SEC

__all__ = ["flight_reward"]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp *value* to the inclusive range [*lower*, *upper*]."""
    return max(lower, min(upper, value))


def _calculate_target_time(task: "MapTask") -> float:
    """Calculate target time based on distance and 2% buffer."""
    import numpy as np
    
    start_pos = np.array(task.start)
    goal_pos = np.array(task.goal)
    distance = np.linalg.norm(goal_pos - start_pos)
    
    min_time = (distance / SPEED_LIMIT) + STABLE_LANDING_SEC
    return min_time * 1.02


def flight_reward(
    success: bool,
    t: float,
    horizon: float,
    task: Optional["MapTask"] = None,
    *,
    w_success: float = 0.5,
    w_t: float = 0.5,
) -> float:
    """Compute the reward for a single flight mission.

    Parameters
    ----------
    success
        ``True`` if the mission successfully reached its objective.
    t
        Time (in seconds) taken to reach the goal.
    horizon
        Maximum time allowed to complete the mission.
    task
        MapTask object containing start and goal positions for distance calculation.
    w_success, w_t
        Weights for the success and time terms. They should sum to ``1``.

    Returns
    -------
    float
        A score in the range ``[0, 1]``.
    """

    if horizon <= 0:
        raise ValueError("'horizon' must be positive")

    success_term = 1.0 if success else 0.0
    
    if success_term == 0.0:
        return 0.0

    if task is not None:
        target_time = _calculate_target_time(task)
        
        if t <= target_time:
            time_term = 1.0
        else:
            time_term = _clamp(1.0 - (t - target_time) / (horizon - target_time))
    else:
        time_term = _clamp(1.0 - t / horizon)

    score = (w_success * success_term) + (w_t * time_term)
    return _clamp(score)