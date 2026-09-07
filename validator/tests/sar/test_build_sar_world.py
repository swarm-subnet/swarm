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

import math

import pybullet as p
import pytest

from swarm.core.env_builder.sar_types import (
    BodyCategory,
)
from swarm.core.env_builder.sar_world import build_sar_world

_MAPS = {
    "open":      2,
    "mountain":  3,
    "city":      1,
    "village":   4,
    "forest":    6,
    "warehouse": 5,
}


@pytest.mark.parametrize("name,ctype", list(_MAPS.items()))
def test_per_map_well_formed(sar_pybullet, name, ctype):
    p.resetSimulation(physicsClientId=sar_pybullet)
    world = build_sar_world(
        sar_pybullet, seed=1234 + ctype, challenge_type=ctype,
        start=(0.0, 0.0, 1.5), goal=(8.0, 8.0, 1.5),
    )
    assert len(world.victim_uids) >= 1
    for u in world.victim_uids:
        assert world.body_tags.get(u) == BodyCategory.VICTIM.value
    assert world.support_uid in world.body_tags
    assert world.body_tags[world.support_uid].startswith("SUPPORT_")
    assert math.isfinite(world.surface_z)
    assert world.safety_patch.support_uid == world.support_uid
    assert world.safety_patch.surface_z == pytest.approx(world.surface_z)
    mn, mx = world.victim_aabb
    assert all(b > a for a, b in zip(mn, mx))
    assert world.search_centre is not None
    cx, cy = world.search_centre
    vx, vy = world.victim_centre_xy
    assert math.hypot(cx - vx, cy - vy) <= 30.0 + 1e-6
