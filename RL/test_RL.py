#!/usr/bin/env python3

from stable_baselines3 import PPO

from swarm.utils.env_factory  import make_env
from swarm.validator.task_gen import random_task
from swarm.validator.forward  import SIM_DT, HORIZON_SEC


task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)
env  = make_env(task, gui=False)

model = PPO.load("model/ppo_policy.zip")

obs, _ = env.reset()
done = False
episode_return = 0.0

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    episode_return += reward
    done = terminated or truncated

print(f"Episode return: {episode_return}")
env.close()
