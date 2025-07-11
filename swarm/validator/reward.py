# swarm/validator/reward.py
"""
Unified reward function for the drone-landing task.

The score is a linear combination of **five** terms, each capped so its
*contribution* to the final score never exceeds **0.20**.  All weights sum
to 1, therefore the total score is always in [0 … 1].

Weighted sum
============
    score = Σ wᵢ · termᵢ
    score ∈ [0, 1]

Terms
-----
success_term   (w = 0.20)  – 1 if the goal is reached, else 0
alive_term     (w = 0.20)  – • 0.50 (→ 0.10 score) if *success* is True
                           – • up to 1.00 (→ 0.20 score) if *success* is False,
                             increasing linearly with time-alive until 30 s
progress_term  (w = 0.20)  – Linear fraction of the start→goal distance closed
time_term      (w = 0.20)  – 1 − t / horizon, only if *success* is True
energy_term    (w = 0.20)  – 1 − e / e_budget, only if *success* is True

All intermediate values are clamped to [0, 1] before weighting.
"""
from __future__ import annotations

__all__ = ["flight_reward"]

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp *value* to [lo, hi]."""
    return max(lo, min(hi, value))


# -----------------------------------------------------------------------------
# reward
# -----------------------------------------------------------------------------
def flight_reward(
    success: bool,
    *,
    # episode-level measurements ------------------------------------------------
    t_alive: float,            # [s] simulated time until termination
    d_start: float,            # [m] start→goal distance at t = 0
    d_final: float,            # [m] distance to goal when the episode ends
    # only needed when the goal is actually reached ----------------------------
    t_to_goal: float | None = None,   # [s] time from start until touchdown
    e_used: float | None = None,      # [J] energy consumed until touchdown
    # environment limits -------------------------------------------------------
    horizon: float = 60.0,     # [s] maximum episode length
    e_budget: float = 50.0,    # [J] nominal energy budget
    goal_tol: float = 0.10,    # [m] distance considered “on the goal”
    # weights (must sum to 1) ---------------------------------------------------
    w_success: float = 0.20,
    w_alive: float   = 0.20,
    w_progress: float = 0.20,
    w_time: float    = 0.20,
    w_energy: float  = 0.20,
) -> float:
    """
    Compute the mission score.

    Parameters
    ----------
    success
        ``True`` if the drone landed on the platform within ``goal_tol``.
    t_alive
        Total simulated time until episode termination [s].
    d_start
        Distance from drone to goal at the very beginning [m].
    d_final
        Distance from drone to goal when the episode ends [m].
    t_to_goal
        Time from episode start until touchdown [s].  **Required** if
        ``success`` is ``True`` – ignored otherwise.
    e_used
        Energy consumed until touchdown [J].  **Required** if ``success`` is
        ``True`` – ignored otherwise.
    horizon
        Maximum episode length allowed by the environment [s].
    e_budget
        Nominal energy budget allowed for the mission [J].
    goal_tol
        Radius around the goal that counts as “on target” [m].
    weights
        Five weights that must add up to 1.0.  They determine each term’s share
        in the final score.

    Returns
    -------
    float
        A score in the range [0 … 1].
    """
    # -------------------------------------------------------------------------
    # basic sanity checks
    # -------------------------------------------------------------------------
    if horizon <= 0:
        raise ValueError("'horizon' must be positive")
    if d_start <= 0:
        raise ValueError("'d_start' must be positive")
    if e_budget <= 0:
        raise ValueError("'e_budget' must be positive")
    if not abs(
        (w_success + w_alive + w_progress + w_time + w_energy) - 1.0
    ) < 1e-6:
        raise ValueError("weights must sum to 1.0")

    # -------------------------------------------------------------------------
    # ① success term
    # -------------------------------------------------------------------------
    success_term = 1.0 if success else 0.0

    # -------------------------------------------------------------------------
    # ② alive term
    #     • If success → fixed 0.10 contribution (term = 0.5).
    #     • Else       → linear up to 30 s, max contribution 0.20.
    # -------------------------------------------------------------------------
    if success:
        alive_term = 0.5  # weight (0.2) × 0.5 → 0.1
    else:
        alive_term = _clamp(t_alive / 30.0)  # 30 s → term = 1.0 → 0.20

    # -------------------------------------------------------------------------
    # ③ progress term
    # -------------------------------------------------------------------------
    d_final_clamped = max(d_final - goal_tol, 0.0)
    progress_term = _clamp((d_start - d_final_clamped) / d_start)

    # -------------------------------------------------------------------------
    # ④ time term (only meaningful if goal reached)
    # -------------------------------------------------------------------------
    if success:
        if t_to_goal is None:
            raise ValueError("'t_to_goal' must be provided when success is True")
        time_term = _clamp(1.0 - t_to_goal / horizon)
    else:
        time_term = 0.0

    # -------------------------------------------------------------------------
    # ⑤ energy term (only meaningful if goal reached)
    # -------------------------------------------------------------------------
    if success:
        if e_used is None:
            raise ValueError("'e_used' must be provided when success is True")
        energy_term = _clamp(1.0 - e_used / e_budget)
    else:
        energy_term = 0.0

    # -------------------------------------------------------------------------
    # final weighted sum
    # -------------------------------------------------------------------------
    score = (
        w_success * success_term
        + w_alive * alive_term
        + w_progress * progress_term
        + w_time * time_term
        + w_energy * energy_term
    )

    # keep within [0, 1] in case of numerical mishaps
    return _clamp(score)
