#!/usr/bin/env python3
"""Baseline PPO starter for cf_swarm_autopilot: one shared policy lands every drone on its pad."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import train_family

if __name__ == "__main__":
    train_family("cf_swarm_autopilot", supports_drone_count=True)
