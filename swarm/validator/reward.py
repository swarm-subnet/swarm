# swarm/validator/reward.py
"""Reward function for flight missions.

The score is a weighted combination of mission success and time efficiency::

    score = 0.50 * success_term + 0.50 * time_term

where

* ``success_term`` is ``1`` if the mission reaches its goal and ``0``
  otherwise.
* ``time_term``   is ``1 - t / horizon`` clamped to ``[0, 1]`` with ``t``
  the time to goal and ``horizon`` the maximum allowed time.

Both weights sum to one. The final score is clamped to ``[0, 1]``.
"""
from __future__ import annotations

__all__ = ["flight_reward"]


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp *value* to the inclusive range [*lower*, *upper*]."""
    return max(lower, min(upper, value))


def flight_reward(
    success: bool,
    t: float,
    horizon: float,
    *,
    w_success: float = 0.5,
    w_t: float = 0.5,
) -> float:
    """Compute the reward for a single flight mission.

    Parameters
    ----------
    success
        ``True`` if the mission successfully reached its objective - landing 
        on platform (PLATFORM=True) or hovering at goal (PLATFORM=False).
    t
        Time (in seconds) taken to reach the goal.
    horizon
        Maximum time allowed to complete the mission.
    w_success, w_t
        Weights for the success and time terms. They should sum to ``1``.

    Returns
    -------
    float
        A score in the range ``[0, 1]``.
    """

    if horizon <= 0:
        raise ValueError("'horizon' must be positive")

    # Individual terms (all in [0, 1])
    success_term = 1.0 if success else 0.0
    time_term = _clamp(1.0 - t / horizon)

    # Weighted sum of the terms
    if success_term == 0.0:
        return 0.0  # No reward if the mission failed
    else:
        score = (w_success * success_term) + (w_t * time_term)

    return _clamp(score)