#!/usr/bin/env python3
"""
Train a PPO model for drone navigation.

Workflow:
1. Train model → saved to swarm/submission_template/ppo_policy.zip
2. Test with: python tests/test_rpc.py swarm/submission_template/ --zip
3. Submission.zip created in Submission/
4. Run miner (reads from Submission/submission.zip)
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from swarm.utils.env_factory import make_env
from swarm.validator.task_gen import random_task
from swarm.constants import SIM_DT, HORIZON_SEC


def main():
    parser = argparse.ArgumentParser(description="Train PPO model for Swarm subnet")
    parser.add_argument("--timesteps", type=int, default=10000, help="Training timesteps")
    args = parser.parse_args()

    task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=1)
    env = make_env(task, gui=False)

    model = PPO("MultiInputPolicy", env, verbose=1)
    model.learn(args.timesteps)

    # Save model to submission_template folder
    output_dir = Path(__file__).parent.parent / "swarm" / "submission_template"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "ppo_policy.zip"
    model.save(str(model_path))
    
    print(f"\n✅ Model saved to: {model_path}")
    print("\n📋 Next steps:")
    print("   1. Test: python tests/test_rpc.py swarm/submission_template/ --zip")
    print("   2. Run miner (reads from Submission/submission.zip)")

    env.close()


if __name__ == "__main__":
    main()
