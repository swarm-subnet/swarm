#!/usr/bin/env python3
"""
Quick offline validator for a single Stable‑Baselines 3 PPO policy.

It re‑uses the **exact same** `_run_episode` helper as the on‑chain
validator, so the score you see here should match what your miner would
receive for the same MapTask.

Usage
-----
$ python test_RL.py --model model/ppo_policy.zip
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from swarm.validator.task_gen   import random_task
from swarm.validator.forward    import _run_episode          # public helper
from swarm.constants            import SIM_DT, HORIZON_SEC


# ----------------------------------------------------------------------
# 1. CLI / paths
# ----------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Local Swarm policy validator")
parser.add_argument(
    "--model",
    type=Path,
    default=Path("model/ppo_policy.zip"),
    help="Path to the Stable‑Baselines3 .zip file",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,
    help="Random seed for MapTask generation",
)
args = parser.parse_args()

if not args.model.exists():
    raise FileNotFoundError(f"Policy file not found: {args.model}")

# ----------------------------------------------------------------------
# 2. Build the same MapTask the on‑chain validator uses
# ----------------------------------------------------------------------
task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)

# ----------------------------------------------------------------------
# 3. Load policy  ➜ run episode  ➜ show result
# ----------------------------------------------------------------------
print(f"Evaluating policy at {args.model} …")
model = PPO.load(args.model, device="cpu")        # CPU to match validator
result = _run_episode(task=task, uid=0, model=model)

print("----------------------------------------------------")
print(f"Success: {result.success}")
print(f"Time    : {result.time_sec:.2f} s")
print(f"Energy  : {result.energy:.1f} J")
print(f"Score   : {result.score:.3f}")
print("----------------------------------------------------")
