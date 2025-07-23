#!/usr/bin/env python3

import argparse
import os


from stable_baselines3 import PPO

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.validator.forward import SIM_DT, HORIZON_SEC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1)
    args = parser.parse_args()

    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)
    env = make_env(task, gui=False)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(args.timesteps)
    
    # Create model directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    model.save("model/ppo_policy")

    env.close()


if __name__ == "__main__":
    main()
