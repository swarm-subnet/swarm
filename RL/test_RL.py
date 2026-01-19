#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from swarm.constants import SIM_DT, SPEED_LIMIT
from swarm.core.drone import track_drone
from swarm.protocol import ValidationResult
from swarm.utils.env_factory import make_env
from swarm.validator.reward import flight_reward
from swarm.validator.task_gen import random_task
from gym_pybullet_drones.utils.enums import ActionType


def _run_episode_speed_limit(task, uid, model, *, gui=False):
    class _Pilot:
        def __init__(self, m):
            self.m = m

        def reset(self, task):
            pass

        def act(self, obs, t):
            act, _ = self.m.predict(obs, deterministic=True)
            return act.squeeze()

    pilot = _Pilot(model)
    env = make_env(task, gui=gui)
    obs = env._computeObs()

    pos0 = np.asarray(task.start, dtype=float)
    t_sim = 0.0
    success = False
    speeds = []
    step_count = 0

    frames_per_cam = max(1, int(round(1.0 / (SIM_DT * 60.0))))
    lo, hi = env.action_space.low.flatten(), env.action_space.high.flatten()
    last_pos = pos0

    cli_id = getattr(env, "CLIENT", getattr(env, "_cli", 0))
    while t_sim < task.horizon:
        act = np.clip(np.asarray(pilot.act(obs, t_sim), dtype=np.float32).reshape(-1), lo, hi)

        if hasattr(env, 'ACT_TYPE') and hasattr(env, 'SPEED_LIMIT'):
            if env.ACT_TYPE == ActionType.VEL and env.SPEED_LIMIT:
                n = max(np.linalg.norm(act[:3]), 1e-6)
                scale = min(1.0, SPEED_LIMIT / n)
                act[:3] *= scale
                act = np.clip(act, lo, hi)

        prev = last_pos
        obs, _r, terminated, truncated, info = env.step(act[None, :])
        last_pos = env._getDroneStateVector(0)[0:3]

        speed = np.linalg.norm(last_pos - prev) / SIM_DT
        speeds.append(speed)

        t_sim += SIM_DT

        if gui and step_count % frames_per_cam == 0:
            try:
                track_drone(cli=cli_id, drone_id=env.DRONE_IDS[0])
            except Exception:
                pass
        if gui:
            time.sleep(SIM_DT)

        if terminated or truncated:
            success = info.get("success", False)
            break
        step_count += 1

    if not gui:
        env.close()

    score = flight_reward(success=success, t=t_sim, horizon=task.horizon, task=task)
    avg_speed = np.mean(speeds) if speeds else 0.0
    result = ValidationResult(uid, success, t_sim, score)
    return result, avg_speed


def _run_episode(task, uid, model, *, gui=False):
    result, _ = _run_episode_speed_limit(task, uid, model, gui=gui)
    return result


def main():
    parser = argparse.ArgumentParser(description="Local Swarm policy validator")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("model/ppo_policy.zip"),
        help="Path to the Stable-Baselines3 .zip file.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for MapTask generation")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="After evaluation, replay the episode in a PyBullet GUI")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Policy file not found: {args.model}")

    task = random_task(sim_dt=SIM_DT, seed=args.seed)

    print(f"Evaluating policy at {args.model} …")
    _init_env = make_env(task, gui=False)
    try:
        model = PPO.load(str(args.model), device="cpu")
    finally:
        try:
            _init_env.close()
        except Exception:
            pass

    result, avg_speed = _run_episode_speed_limit(
        task=task, uid=0, model=model, gui=args.gui
    )

    print("----------------------------------------------------")
    print(f"Success : {result.success}")
    print(f"Time    : {result.time_sec:.2f} s")
    print(f"Score   : {result.score:.3f}")
    print(f"Avg Speed: {avg_speed:.3f} m/s")
    print("----------------------------------------------------")


if __name__ == "__main__":
    main()
