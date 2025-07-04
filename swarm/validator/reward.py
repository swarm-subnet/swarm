# ──────────────────────────────────────────────────────────────────────────
#  swarm/validator/reward.py                episodic score (validator / stats)
# ──────────────────────────────────────────────────────────────────────────
"""
Final mission score (∈ [0 … 1]).

Principles
~~~~~~~~~~
* **Reaching the goal** is paramount – it carries the majority of the weight.
* If the goal is *not* reached we still give credit for:
  – surviving as long as possible,
  – ending closer to the goal than we started.
* The components are combined linearly then clamped to [0, 1].

Arguments
~~~~~~~~~
success       – True if the “success” flag from the env is set.
t_alive       – Simulated time until termination [s].
d_start       – Distance start→goal at t = 0 [m].
d_final       – Distance current→goal when episode ends [m].
horizon       – Maximum episode length [s].

Weights (tweak to taste)
~~~~~~~~~~~~~~~~~~~~~~~~
w_success  : 0.80   # dominant term – must succeed!
w_alive    : 0.10   # stay in the air
w_progress : 0.10   # finish closer than you started
"""
from __future__ import annotations

__all__ = ["flight_reward"]

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def flight_reward(
    success: bool,
    t_alive: float,
    d_start: float,
    d_final: float,
    horizon: float,
    *,
    w_success: float = 0.60,
    w_alive: float = 0.10,
    w_progress: float = 0.30,
) -> float:
    if horizon <= 0:
        raise ValueError("'horizon' must be positive")
    if d_start <= 0:
        raise ValueError("'d_start' must be positive")

    # ① success term
    success_term = 1.0 if success else 0.0

    # ② time‑alive term   (0 → 1 over the episode)
    alive_term = _clamp(t_alive / horizon)

    # ③ progress term     (0 if we end further away, up to 1 if we finish on the goal)
    progress_term = _clamp((d_start - d_final) / d_start)

    score = (
        w_success  * success_term +
        w_alive    * alive_term +
        w_progress * progress_term
    )
    return _clamp(score)
