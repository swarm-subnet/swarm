from __future__ import annotations

import math

import numpy as np
import pybullet as p
import pytest

from swarm.constants import SIM_DT
from swarm.protocol import MapTask
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import task_for_seed_and_type


def _manual_open_world_task() -> MapTask:
    return MapTask(
        map_seed=31415,
        start=(0.0, 0.0, 1.5),
        goal=(8.0, 0.0, 1.5),
        sim_dt=1 / 30,
        horizon=20.0,
        challenge_type=2,
        family_id="cf_autopilot",
        version="4.9.0",
    )


def _goal_directed_policy(observation: dict[str, np.ndarray]) -> np.ndarray:
    state = np.asarray(observation["state"], dtype=np.float32)
    goal_offset = state[-3:]
    horizontal_offset = np.array([goal_offset[0], goal_offset[1], 0.0], dtype=np.float32)
    norm = float(np.linalg.norm(horizontal_offset))
    if norm > 1e-6:
        direction = horizontal_offset / norm
    else:
        direction = np.zeros(3, dtype=np.float32)
    yaw = float(math.atan2(float(direction[1]), float(direction[0]))) / math.pi if norm > 1e-6 else 0.0
    speed = min(0.7, max(0.15, norm / 10.0))
    z_term = float(np.clip(goal_offset[2] * 0.5, -0.4, 0.4))
    return np.array([direction[0], direction[1], z_term, speed, yaw], dtype=np.float32)


def _random_policy(
    rng: np.random.RandomState,
    observation: dict[str, np.ndarray],
) -> np.ndarray:
    _ = observation
    action = rng.uniform(-1.0, 1.0, size=5).astype(np.float32)
    action[3] = float(rng.uniform(0.0, 1.0))
    return action


def _run_policy_episode(
    task: MapTask,
    *,
    controller,
    seed: int,
    max_steps: int | None = None,
) -> tuple[dict[str, object], bool, bool]:
    env = make_env(task, gui=False)
    try:
        obs, _ = env.reset(seed=seed)
        terminated = False
        truncated = False
        info: dict[str, object] = {}
        if max_steps is None:
            max_steps = int(round(task.horizon / task.sim_dt)) + 5

        for _ in range(max_steps):
            action = np.asarray(controller(obs), dtype=np.float32)
            obs, _reward, terminated, truncated, info = env.step(action[None, :])
            if terminated or truncated:
                break

        return info, bool(terminated), bool(truncated)
    finally:
        env.close()


def test_autopilot_runtime_marks_success_when_drone_enters_goal_radius():
    task = _manual_open_world_task()
    env = make_env(task, gui=False)
    try:
        env.reset(seed=task.map_seed)
        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[0],
            [float(task.goal[0]), float(task.goal[1]), float(task.goal[2])],
            p.getQuaternionFromEuler([0.0, 0.0, 0.0]),
            physicsClientId=env.CLIENT,
        )
        p.resetBaseVelocity(
            env.DRONE_IDS[0],
            linearVelocity=[0.0, 0.0, 0.0],
            angularVelocity=[0.0, 0.0, 0.0],
            physicsClientId=env.CLIENT,
        )
        env._updateAndStoreKinematicInformation()

        env._time_alive = 0.0
        env._step_processed = False
        env._process_step_updates()

        assert env._success is True
        assert env._t_to_goal == pytest.approx(env._sim_dt)
        assert env._failure_reason == "NONE"
    finally:
        env.close()


def test_autopilot_generated_scenario_builds_and_steps():
    task = task_for_seed_and_type(
        sim_dt=SIM_DT,
        seed=657398,
        challenge_type=4,
        family_id="cf_autopilot",
    )
    env = make_env(task, gui=False)
    try:
        obs, info = env.reset(seed=task.map_seed)
        assert obs["depth"].size > 0
        assert np.isfinite(obs["depth"]).all()
        assert obs["state"].shape[-1] >= 16

        obs, reward, terminated, truncated, info = env.step(
            np.zeros((1, 5), dtype=np.float32)
        )
        assert obs["depth"].size > 0
        assert np.isfinite(float(reward))
        assert isinstance(bool(terminated), bool)
        assert isinstance(bool(truncated), bool)
        assert "distance_to_goal" in info
        assert "autopilot_goal_reach_radius_m" in info
    finally:
        env.close()


def test_goal_directed_baseline_beats_random_policy_on_easy_autopilot_seed():
    task = _manual_open_world_task()

    baseline_info, baseline_terminated, baseline_truncated = _run_policy_episode(
        task,
        controller=_goal_directed_policy,
        seed=task.map_seed,
    )
    rng = np.random.RandomState(123)
    random_info, random_terminated, random_truncated = _run_policy_episode(
        task,
        controller=lambda obs: _random_policy(rng, obs),
        seed=task.map_seed,
    )

    assert baseline_terminated or baseline_truncated
    assert random_terminated or random_truncated
    assert float(baseline_info["distance_to_goal"]) < float(random_info["distance_to_goal"])
    assert float(baseline_info["score"]) > float(random_info["score"])
