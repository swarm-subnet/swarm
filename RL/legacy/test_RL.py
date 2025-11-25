#!/usr/bin/env python3
"""
Quick offline validator for a single Stable‑Baselines 3 PPO policy.

If you pass the optional ``--gui`` flag the policy is first evaluated
head‑less (exactly like the on‑chain validator) and the metrics are
printed.  Afterwards **the identical episode is replayed once more**
with a PyBullet GUI.

Usage
-----
$ python test_RL.py --model model/ppo_policy.zip [--seed 42] [--gui]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO

from swarm.constants              import SIM_DT, HORIZON_SEC
from swarm.validator.task_gen     import random_task
from ..test_RL                    import _run_episode          # public helper
from swarm.utils.gui_isolation    import run_isolated


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Local Swarm policy validator")
parser.add_argument(
    "--model", type=Path, default=Path("model/ppo_policy.zip"),
    help="Path to the Stable‑Baselines 3 .zip file",
)
parser.add_argument(
    "--seed", type=int, default=1,
    help="Random seed for MapTask generation",
)
parser.add_argument(
    "--gui", action="store_true", default=False,
    help="After evaluation, replay the episode in a PyBullet GUI",
)
args = parser.parse_args()

if not args.model.exists():
    raise FileNotFoundError(f"Policy file not found: {args.model}")


# ──────────────────────────────────────────────────────────────────────
# Deterministic MapTask
# ──────────────────────────────────────────────────────────────────────
task = random_task(sim_dt=SIM_DT, horizon=HORIZON_SEC, seed=args.seed)

# ──────────────────────────────────────────────────────────────────────
# 1) Head‑less evaluation (bit‑for‑bit like the validator)
# ──────────────────────────────────────────────────────────────────────
print(f"Evaluating policy at {args.model} …")
model = PPO.load(args.model, device="cpu")               # CPU to match chain
result = _run_episode(task=task, uid=0, model=model, gui=args.gui)     # gui=False (default)

print("----------------------------------------------------")
print(f"Success: {result.success}")
print(f"Time    : {result.time_sec:.2f} s")
print(f"Score   : {result.score:.3f}")
print("----------------------------------------------------")