from __future__ import annotations

import math
import time

import pybullet as p
import pytest

from swarm.core.env_builder.sar_tagging import build_and_tag_map
from swarm.core.env_builder.spawn_pipeline import (
    MAX_SPAWN_ATTEMPTS,
    SARSpawnError,
    find_spawn_xy,
)


_MAPS = {
    "open":      2,
    "mountain":  3,
    "city":      1,
    "village":   4,
    "forest":    6,
    "warehouse": 5,
}


_N_SEEDS = 100
_MAX_FAILURE_RATE = 0.02


# Plan B.4.3 asks for 5000 seeds × 6 maps at <=0.5% failure rate. Map build
# averages ~5-30s per seed depending on environment type (warehouse is heaviest),
# so the full 5000-seed audit cannot run in CI. We sample 100 seeds × 6 maps
# at a 2% threshold here — strong enough to catch a real pipeline regression;
# the full 5000-seed audit ships as the nightly script in D.3.2.
@pytest.mark.parametrize("name,ctype", list(_MAPS.items()))
def test_per_map_failure_rate(sar_pybullet, name, ctype):
    failures = 0
    started = time.time()
    for seed in range(_N_SEEDS):
        p.resetSimulation(physicsClientId=sar_pybullet)
        tagger = build_and_tag_map(
            sar_pybullet, seed=seed, challenge_type=ctype,
            start=(0.0, 0.0, 1.5), goal=(8.0, 8.0, 1.5),
        )
        try:
            find_spawn_xy(
                sar_pybullet,
                map_seed=seed,
                challenge_type=ctype,
                body_tags=tagger.body_tags,
            )
        except SARSpawnError:
            failures += 1
    elapsed = time.time() - started
    failure_rate = failures / _N_SEEDS
    print(f"{name}: {failures}/{_N_SEEDS} failures = {failure_rate:.4%}  ({elapsed:.0f}s)")
    assert failure_rate <= _MAX_FAILURE_RATE, (
        f"{name}: failure_rate {failure_rate:.4%} exceeds threshold {_MAX_FAILURE_RATE:.2%}"
    )
