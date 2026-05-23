"""B.3.3 — fuzz 2000 random XY positions per map family and assert minimum
acceptance rates. Catches resolver bugs that synthetic tests miss."""
from __future__ import annotations

import random
from typing import Iterable

import pybullet as p
import pytest

from swarm.core.env_builder.body_tagger import BodyTagger
from swarm.core.env_builder.sar_tagging import build_and_tag_map
from swarm.core.env_builder.sar_types import BodyCategory, SUPPORT_CATEGORIES
from swarm.core.env_builder.surface_resolver import resolve_surface


_FUZZ_N = 2000
_RNG_SEED = 7

_MAP_THRESHOLDS = {
    "open":      (2, (-30.0, 30.0), 0.30),
    "mountain":  (3, (-30.0, 30.0), 0.30),
    "city":      (1, (-30.0, 30.0), 0.15),
    "village":   (4, (-30.0, 30.0), 0.25),
    "forest":    (6, (-25.0, 25.0), 0.10),
    "warehouse": (5, (-15.0, 15.0), 0.10),
}


def _fuzz_one_map(cli, name, challenge_type, sample_lo_hi, min_accept):
    p.resetSimulation(physicsClientId=cli)
    tagger = build_and_tag_map(
        cli, seed=20_000 + challenge_type, challenge_type=challenge_type,
        start=(0.0, 0.0, 1.5), goal=(8.0, 8.0, 1.5),
    )
    body_tags = tagger.body_tags

    rng = random.Random(_RNG_SEED + challenge_type)
    lo, hi = sample_lo_hi
    accepts = 0
    bad_category = 0
    for _ in range(_FUZZ_N):
        x = rng.uniform(lo, hi)
        y = rng.uniform(lo, hi)
        hit = resolve_surface(cli, x, y, body_tags, SUPPORT_CATEGORIES)
        if hit is None:
            continue
        accepts += 1
        # Invariant: accepted hit must be on a support_* body.
        if not hit.category.startswith("SUPPORT_"):
            bad_category += 1

    accept_rate = accepts / _FUZZ_N
    assert bad_category == 0, f"{name}: {bad_category} non-support accepts"
    assert accept_rate >= min_accept, (
        f"{name}: acceptance rate {accept_rate:.2%} < threshold {min_accept:.2%}"
    )


@pytest.mark.parametrize("name,info", list(_MAP_THRESHOLDS.items()))
def test_per_map_acceptance(sar_pybullet, name, info):
    challenge_type, sample_box, min_accept = info
    _fuzz_one_map(sar_pybullet, name, challenge_type, sample_box, min_accept)
