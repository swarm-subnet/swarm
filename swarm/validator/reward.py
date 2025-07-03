# ──────────────────────────────────────────────────────────────────────────
#  swarm/validator/reward.py          episodic score (validator / stats)
# ──────────────────────────────────────────────────────────────────────────
"""
Final mission score.

Idea
~~~~
* When the goal **is reached** we care about finishing **fast** and **cheaply**.
* When the goal **is NOT reached** we give at most *half* of the total weight
  for “good behaviour” (hovering economically), but there is **no time bonus** –
  faster failures are not better than slower ones.

The score still lives in **[0 … 1]**.
"""
from __future__ import annotations

__all__ = ["flight_reward"]


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def flight_reward(
    success: bool,
    t: float,
    e: float,
    horizon: float,
    e_budget: float = 00.1,
    *,
    w_success: float = 0.99,
    w_t: float = 0.30,
    w_e: float = 0.20,
) -> float:
    """Return a score in `[0.0 … 1.0]`.

    *If `success` is False the time term is *ignored* and the energy term is
    down-weighted by ½ so that any failure caps the total score at 0.5.*
    """
    if horizon <= 0:
        raise ValueError("'horizon' must be positive")
    if e_budget <= 0:
        raise ValueError("'e_budget' must be positive")

    time_term   = _clamp(1.0 - t / horizon)
    energy_term = _clamp(1.0 - e / e_budget)

    if success:
        score = (
            w_success * 1.0 +
            w_t       * time_term +
            w_e       * energy_term
        )
    else:
        # Half-weight the energy term, drop the time term completely
        score = (
            w_success * 0.0 +
            0.0               +
            (w_e * 0.5) * energy_term
        )

    return _clamp(score)
