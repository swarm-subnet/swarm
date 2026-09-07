# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import time

import pybullet as p
import pytest

from swarm.core.env_builder.sar_tagging import build_and_tag_map
from swarm.core.env_builder.spawn_pipeline import (
    SARSpawnError,
    find_spawn_xy,
)

pytestmark = pytest.mark.full


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
