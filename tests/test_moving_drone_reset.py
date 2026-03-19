from __future__ import annotations

from types import SimpleNamespace

import gymnasium.spaces as spaces
import numpy as np
from gym_pybullet_drones.envs.BaseRLAviary import BaseRLAviary

from swarm.core import moving_drone as moving_drone_mod


def test_reset_generates_search_area_after_world_spawn(monkeypatch) -> None:
    env = moving_drone_mod.MovingDroneAviary.__new__(moving_drone_mod.MovingDroneAviary)
    env.task = SimpleNamespace(map_seed=123, goal=(1.0, 2.0, 3.0))
    env.GOAL_POS = np.array([1.0, 2.0, 3.0], dtype=float)
    env.EP_LEN_SEC = 60.0
    env._state_dim = 4
    env.CLIENT = 7
    env.observation_space = {
        "depth": spaces.Box(low=0.0, high=1.0, shape=(2, 2, 1), dtype=np.float32),
        "state": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
    }

    monkeypatch.setattr(
        BaseRLAviary,
        "reset",
        lambda self, **kwargs: (
            {
                "depth": np.zeros((2, 2, 1), dtype=np.float32),
                "state": np.zeros((4,), dtype=np.float32),
            },
            {"seed": kwargs.get("seed")},
        ),
    )
    monkeypatch.setattr(moving_drone_mod, "flight_reward", lambda **kwargs: 0.0)
    monkeypatch.setattr(
        moving_drone_mod.p,
        "setPhysicsEngineParameter",
        lambda **kwargs: None,
    )

    def _spawn_task_world() -> None:
        env.GOAL_POS = np.array([100.0, 200.0, 300.0], dtype=float)

    env._spawn_task_world = _spawn_task_world
    env._computeObs = lambda: {
        "depth": np.zeros((2, 2, 1), dtype=np.float32),
        "state": np.zeros((4,), dtype=np.float32),
    }
    env._generate_search_area_center = lambda seed=None: env.GOAL_POS.copy()

    obs, info = moving_drone_mod.MovingDroneAviary.reset(env, seed=123)

    assert np.array_equal(env._search_area_center, np.array([100.0, 200.0, 300.0]))
    assert np.array_equal(obs["state"], np.zeros((4,), dtype=np.float32))
    assert info == {"seed": 123}
