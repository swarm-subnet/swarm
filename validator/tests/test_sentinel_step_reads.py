"""Small per-step shortcuts retain the original scalar and engine bits."""

from types import SimpleNamespace

import numpy as np

from swarm.challenge_families.solar_patrol import dock3
from swarm.core.moving_drone import MovingDroneAviary


def test_direct_state_reads():
    """Direct positions and Euler angles equal the original twenty-value state path."""
    env = MovingDroneAviary.__new__(MovingDroneAviary)
    env.NUM_DRONES = 1
    env._collision, env._success, env._failure_reason = False, False, "NONE"
    env._prev_score, env._t_to_goal, env._min_clearance_episode = 0.25, None, 3.0
    env.family_runtime = SimpleNamespace(compute_truncated=lambda env, **kwargs: kwargs, build_info=lambda env: {})
    rng = np.random.default_rng(4)
    for _ in range(1000):
        for field, size in [("pos", 3), ("quat", 4), ("rpy", 3), ("vel", 3), ("ang_v", 3), ("last_clipped_action", 4)]:
            setattr(env, field, rng.normal(size=(1, size)))
        env.GOAL_POS = rng.normal(size=3)
        state = env._getDroneStateVector(0)
        got = env._computeTruncated()
        assert np.float64(got["roll"]).tobytes() == state[7].tobytes()
        assert np.float64(got["pitch"]).tobytes() == state[8].tobytes()
        assert np.float64(env._computeInfo()["distance_to_goal"]).tobytes() == np.float64(
            np.linalg.norm(state[:3] - env.GOAL_POS)).tobytes()


def test_settled_lids_skip_clip(monkeypatch):
    """Resting lids return the same result without any engine or numpy work."""
    def fail(*args, **kwargs):
        """Reject work on settled lids."""
        raise AssertionError("settled lids did work")
    monkeypatch.setattr(dock3.np, "clip", fail)
    for opening in (0.0, 1.0):
        assert dock3.move_lids(None, SimpleNamespace(opening=opening), bool(opening), 0.02)
