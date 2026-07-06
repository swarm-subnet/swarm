#!/usr/bin/env python3
"""Baseline PPO starter for cf_search_and_rescue: find the victim and hover above it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import train_family

if __name__ == "__main__":
    train_family("cf_search_and_rescue")
