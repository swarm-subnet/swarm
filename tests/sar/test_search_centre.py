from __future__ import annotations

import math
import random
import statistics

import pytest

from swarm.core.env_builder.search_clue import (
    SEARCH_RADIUS_M,
    sample_search_centre,
)


def test_within_30m_circle():
    victim = (12.3, -4.5)
    rng = random.Random(0)
    for _ in range(1000):
        sc = sample_search_centre(rng, victim)
        dx, dy = sc[0] - victim[0], sc[1] - victim[1]
        assert math.hypot(dx, dy) <= SEARCH_RADIUS_M + 1e-6


def test_circle_uniformity():
    victim = (0.0, 0.0)
    rng = random.Random(42)
    n = 10000
    bins = [0] * 10
    for _ in range(n):
        sc = sample_search_centre(rng, victim)
        r = math.hypot(sc[0], sc[1])
        slot = min(int(r / (SEARCH_RADIUS_M / 10.0)), 9)
        bins[slot] += 1
    # Expected count per ring is proportional to (r_outer^2 - r_inner^2).
    expected = []
    for i in range(10):
        r_in = (i / 10.0) * SEARCH_RADIUS_M
        r_out = ((i + 1) / 10.0) * SEARCH_RADIUS_M
        expected.append(n * (r_out * r_out - r_in * r_in) / (SEARCH_RADIUS_M * SEARCH_RADIUS_M))
    chi_sq = sum((o - e) ** 2 / e for o, e in zip(bins, expected))
    assert chi_sq < 50.0, f"chi-square {chi_sq:.2f} suggests non-uniform radial density"


def test_anchored_to_victim_not_task_goal():
    # Simulate spawn-retry: victim ends up far from original task.goal.
    task_goal_xy = (10.0, 0.0)
    victim_after_retry = (-15.0, 22.0)
    rng = random.Random(7)
    sc = sample_search_centre(rng, victim_after_retry)
    d_to_victim = math.hypot(sc[0] - victim_after_retry[0], sc[1] - victim_after_retry[1])
    d_to_goal = math.hypot(sc[0] - task_goal_xy[0], sc[1] - task_goal_xy[1])
    assert d_to_victim <= SEARCH_RADIUS_M
    assert d_to_goal > 1.0  # anchored to victim, not task goal
