from __future__ import annotations

import numpy as np

from swarm.constants import SIM_DT
from swarm.validator.task_gen import task_for_seed_and_type
from swarm.utils.env_factory import make_env


def test_village_env_builds_for_reported_seed() -> None:
    task = task_for_seed_and_type(sim_dt=SIM_DT, seed=657398, challenge_type=4)
    env = make_env(task, gui=False)
    try:
        assert env.task.challenge_type == 4
        assert env.GOAL_POS.shape == (3,)
        assert float(env.task.start[2]) > 0.0
        assert float(env.task.goal[2]) >= 0.0
    finally:
        env.close()


def test_mountain_env_renders_depth_without_er_depth_only() -> None:
    task = task_for_seed_and_type(sim_dt=SIM_DT, seed=657393, challenge_type=3)
    env = make_env(task, gui=False)
    try:
        obs, info = env.reset(seed=task.map_seed)
        assert obs["depth"].size > 0
        assert np.isfinite(obs["depth"]).all()

        obs, reward, terminated, truncated, info = env.step(
            np.zeros((1, 5), dtype=np.float32)
        )
        assert obs["depth"].size > 0
        assert np.isfinite(obs["depth"]).all()
        assert np.isfinite(float(reward))
        assert isinstance(bool(terminated), bool)
        assert isinstance(bool(truncated), bool)
        assert "distance_to_goal" in info
    finally:
        env.close()
