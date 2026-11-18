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

"""The shared decoy ground of the flat stand-in parks: the same world reuses it, and a world with anything new standing
in it gets a ground of its own."""

from __future__ import annotations

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families.solar_patrol import decoys

FENCE = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 30.0], [0.0, 30.0]])
DOCK = np.array([27.0, 27.0])


@pytest.fixture
def slab():
    """A physics world holding a 40 m flat slab, and the slab's body id."""
    cli = p.connect(p.DIRECT)
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[20.0, 20.0, 0.5], physicsClientId=cli)
    floor = p.createMultiBody(0, shape, -1, [15.0, 15.0, -0.5], physicsClientId=cli)
    yield cli, floor
    p.disconnect(cli)


def _ground(cli: int, floor: int) -> decoys.Ground:
    """The open ground of the test fence, found through whatever open_ground is in place."""
    return decoys.open_ground(cli, FENCE, frozenset({floor}), DOCK, np.zeros((0, 2)))


def test_the_same_world_reuses_its_ground_and_a_new_body_gets_a_fresh_one(shared_open_ground, slab):
    """Proves a second reset of an unchanged world gets the shared ground, and a box stood on the slab afterwards gets
    a ground of its own with the cells under the box closed."""
    cli, floor = slab
    first = _ground(cli, floor)
    assert _ground(cli, floor) is first

    box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2.0, 2.0, 1.0], physicsClientId=cli)
    p.createMultiBody(0, box, -1, [12.0, 12.0, 1.0], physicsClientId=cli)
    second = _ground(cli, floor)
    assert second is not first
    assert first.open_at(np.array([[12.0, 12.0]]))[0]
    assert not second.open_at(np.array([[12.0, 12.0]]))[0]
