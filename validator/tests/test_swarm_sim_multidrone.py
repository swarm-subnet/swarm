from __future__ import annotations

import numpy as np

from swarm.protocol import MapTask
from swarm.utils.env_factory import make_env


def _multi_task() -> MapTask:
    return MapTask(
        map_seed=7, start=(0.0, 0.0, 1.5), goal=(6.0, 0.0, 1.5),
        sim_dt=1 / 30, horizon=5.0, challenge_type=2, family_id="cf_autopilot",
        num_drones=3,
        starts=((0.0, 0.0, 1.5), (2.0, 0.0, 1.5), (-2.0, 0.0, 1.5)),
        goals=((6.0, 0.0, 1.5), (6.0, 2.0, 1.5), (6.0, -2.0, 1.5)),
    )


def test_multi_drone_builds_steps_and_freezes():
    env = make_env(_multi_task(), gui=False)
    try:
        env.reset(seed=7)
        assert env.NUM_DRONES == 3
        assert len(env.DRONE_IDS) == 3

        for _ in range(5):
            _obs, _r, term, trunc, _info = env.step(np.zeros((3, 5), dtype=np.float32))
            if term or trunc:
                break

        env._freeze_drone(1)
        assert bool(env._frozen[1]) is True
        pos_before = np.array(env.pos[1], dtype=float)

        for _ in range(4):
            act = np.zeros((3, 5), dtype=np.float32)
            act[1] = np.array([1.0, 0.0, 1.0, 1.0, 0.0], dtype=np.float32)
            env.step(act)

        pos_after = np.array(env.pos[1], dtype=float)
        assert np.allclose(pos_before, pos_after, atol=1e-3), (pos_before, pos_after)
    finally:
        env.close()
