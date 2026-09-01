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

"""Office map builder: geometry, collision, and determinism checks."""

import numpy as np
import pybullet as p
import pytest

from swarm.core.maps.office import OFFICE_CEILING_M, build_office_map
from swarm.core.maps.office.builder import office_scale


SEED = 0
SCALE = office_scale(SEED)


@pytest.fixture()
def office_client():
    cli = p.connect(p.DIRECT)
    info = build_office_map(seed=SEED, cli=cli)
    yield cli, info
    p.disconnect(cli)


def _at(x=0.0, y=0.0, z=0.0):
    """A nominal point placed in this seed's room."""
    sx, sy, sz = SCALE
    return [x * sx, y * sy, z * sz]


def _ray_hit(cli, start, end):
    hit = p.rayTest(start, end, physicsClientId=cli)[0]
    return hit[0], hit[3]


def test_office_map_bodies(office_client):
    cli, info = office_client
    assert set(info["bodies"]) == {"floor", "shell", "west", "mid", "east", "led", "backdrop"}
    assert info["window_plug"] >= 0
    assert info["ceiling_m"] == pytest.approx(OFFICE_CEILING_M * SCALE[2])


def test_office_map_shell_geometry(office_client):
    cli, info = office_client
    _, floor_pt = _ray_hit(cli, _at(9, 3.8, 2.0), _at(9, 3.8, -1))
    assert abs(floor_pt[2] - 0.0) < 1e-3, "the floor stays at z=0 whatever the room size"
    _, ceil_pt = _ray_hit(cli, _at(9, 3.8, 1.0), _at(9, 3.8, 5))
    assert abs(ceil_pt[2] - 3.0 * SCALE[2]) < 1e-3
    _, col_pt = _ray_hit(cli, _at(13.0, 5.96, 1.5), _at(16, 5.96, 1.5))
    assert abs(col_pt[0] - 14.065 * SCALE[0]) < 5e-3


def test_office_map_furniture_collision(office_client):
    cli, info = office_client
    body, table_pt = _ray_hit(cli, _at(7.0, 1.55, 2.0), _at(7.0, 1.55, 0.2))
    assert body == info["bodies"]["mid"]
    assert abs(table_pt[2] - 0.74 * SCALE[2]) < 2e-2


def test_office_map_window_plug(office_client):
    cli, info = office_client
    for y, z in [(3.2, 1.8), (4.4, 1.2)]:
        body, pt = _ray_hit(cli, _at(16.5, y, z), _at(21, y, z))
        assert body == info["window_plug"]
        assert pt[0] < 18.1 * SCALE[0]


def _aabbs(seed):
    cli = p.connect(p.DIRECT)
    info = build_office_map(seed=seed, cli=cli)
    out = [p.getAABB(b, physicsClientId=cli) for b in sorted(info["bodies"].values())]
    p.disconnect(cli)
    return np.array(out)


def test_office_map_deterministic():
    """One seed builds one room, every time and on every validator."""
    assert np.allclose(_aabbs(1), _aabbs(1))


def test_office_map_size_follows_the_seed():
    """Different seeds build differently sized rooms, so a floorplan fitted to one
    episode does not line up with the next."""
    a, b = _aabbs(1), _aabbs(999)
    assert not np.allclose(a, b), "two seeds produced the same room"
    span_a = float(a[:, 1, 0].max() - a[:, 0, 0].min())
    span_b = float(b[:, 1, 0].max() - b[:, 0, 0].min())
    assert abs(span_a - span_b) > 0.05, f"room length barely moved: {span_a:.2f} vs {span_b:.2f}"
