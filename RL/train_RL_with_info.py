#!/usr/bin/env python3

import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT, HORIZON_SEC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)

    def _build_env():
        return make_env(task, gui=False)

    env = DummyVecEnv([_build_env])
    env = VecTransposeImage(env)
    model = PPO("MultiInputPolicy", env, verbose=1)

    model.learn(args.timesteps)

    os.makedirs("model", exist_ok=True)
    model.save("model/ppo_policy")

    env.close()


if __name__ == "__main__":
    main()
