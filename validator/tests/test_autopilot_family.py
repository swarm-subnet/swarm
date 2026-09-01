from __future__ import annotations

import math

import numpy as np
import pybullet as p
import pytest

from swarm.constants import SEARCH_RADIUS_MAX, SIM_DT
from swarm.protocol import MapTask
from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import task_for_seed_and_type


def _autopilot_center_and_goal(seed):
    task = task_for_seed_and_type(sim_dt=SIM_DT, seed=seed, challenge_type=2, family_id="cf_autopilot")
    env = make_env(task, gui=False)
    try:
        obs, _ = env.reset(seed=task.map_seed)
        sv = env._getDroneStateVector(0)
        return (
            np.array(env._search_area_center, dtype=float),
            np.array(env.GOAL_POS, dtype=float),
            np.array(obs["state"][-3:], dtype=np.float32),
            np.array(sv[0:3], dtype=np.float32),
        )
    finally:
        env.close()


def test_autopilot_observation_is_noisy_search_clue_not_exact_goal():
    center, _goal, state_tail, pos = _autopilot_center_and_goal(4242)
    # the last 3 state numbers are the offset to the (noisy) search centre, exactly like main
    np.testing.assert_allclose(state_tail, (center - pos).astype(np.float32), atol=1e-5)


def test_autopilot_search_clue_is_deterministic_and_within_radius():
    c1, g1, _t1, _p1 = _autopilot_center_and_goal(7)
    c2, _g2, _t2, _p2 = _autopilot_center_and_goal(7)
    np.testing.assert_array_equal(c1, c2)

    offsets = [float(np.hypot(c1[0] - g1[0], c1[1] - g1[1]))]
    for seed in (1, 2, 3):
        c, g, _t, _p = _autopilot_center_and_goal(seed)
        off = float(np.hypot(c[0] - g[0], c[1] - g[1]))
        offsets.append(off)
        assert off <= SEARCH_RADIUS_MAX * 1.5
    assert max(offsets) > 0.5, "autopilot clue is never noticeably offset from the goal"


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
    # take off and cruise while still far, then descend onto the goal pad
    if norm > 2.0:
        z_term = 0.6
        speed = 0.7
    else:
        z_term = float(np.clip(goal_offset[2] * 0.6, -0.5, 0.5))
        speed = min(0.4, max(0.1, norm / 5.0))
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


def test_autopilot_runtime_marks_success_on_stable_landing():
    from swarm.constants import LANDING_STABLE_SEC

    task = _manual_open_world_task()
    env = make_env(task, gui=False)
    try:
        env.reset(seed=task.map_seed)
        # park the drone upright and motionless on the goal pad
        p.resetBasePositionAndOrientation(
            env.DRONE_IDS[0],
            [float(env.GOAL_POS[0]), float(env.GOAL_POS[1]), float(env.GOAL_POS[2]) + 0.05],
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
        env._success = False
        env._collision = False
        env._landing_stable_time = 0.0

        # a brief air gap does not count as landing
        env._update_landing_state(False)
        assert env._success is False

        # continuous, stable contact for >= LANDING_STABLE_SEC completes the landing
        steps = int(LANDING_STABLE_SEC / env._sim_dt) + 2
        for _ in range(steps):
            env._update_landing_state(True)

        assert env._success is True
        assert env._t_to_goal is not None
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
