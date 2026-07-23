"""Office map builder: geometry, collision, and determinism checks."""

import numpy as np
import pybullet as p
import pytest

from swarm.core.maps.office import (
    OFFICE_CEILING_M,
    build_office_map,
    clear_office_shape_cache,
)


@pytest.fixture()
def office_client():
    cli = p.connect(p.DIRECT)
    info = build_office_map(seed=0, cli=cli)
    yield cli, info
    p.disconnect(cli)
    clear_office_shape_cache(cli)


def _ray_hit(cli, start, end):
    hit = p.rayTest(start, end, physicsClientId=cli)[0]
    return hit[0], hit[3]


def test_office_map_bodies(office_client):
    cli, info = office_client
    assert set(info["bodies"]) == {"floor", "shell", "west", "mid", "east", "led", "backdrop"}
    assert info["window_plug"] >= 0
    assert info["ceiling_m"] == OFFICE_CEILING_M


def test_office_map_shell_geometry(office_client):
    cli, info = office_client
    _, floor_pt = _ray_hit(cli, [9, 3.8, 2.0], [9, 3.8, -1])
    assert abs(floor_pt[2] - 0.0) < 1e-3
    _, ceil_pt = _ray_hit(cli, [9, 3.8, 1.0], [9, 3.8, 5])
    assert abs(ceil_pt[2] - 3.0) < 1e-3
    _, col_pt = _ray_hit(cli, [13.0, 5.96, 1.5], [16, 5.96, 1.5])
    assert abs(col_pt[0] - 14.065) < 5e-3


def test_office_map_furniture_collision(office_client):
    cli, info = office_client
    body, table_pt = _ray_hit(cli, [7.0, 1.55, 2.0], [7.0, 1.55, 0.2])
    assert body == info["bodies"]["mid"]
    assert abs(table_pt[2] - 0.74) < 2e-2


def test_office_map_window_plug(office_client):
    cli, info = office_client
    for y, z in [(3.2, 1.8), (4.4, 1.2)]:
        body, pt = _ray_hit(cli, [16.5, y, z], [21, y, z])
        assert body == info["window_plug"]
        assert pt[0] < 18.1


def test_office_map_deterministic():
    aabbs = []
    for seed in (1, 999):
        cli = p.connect(p.DIRECT)
        info = build_office_map(seed=seed, cli=cli)
        aabbs.append([p.getAABB(b, physicsClientId=cli) for b in sorted(info["bodies"].values())])
        p.disconnect(cli)
        clear_office_shape_cache(cli)
    assert np.allclose(np.array(aabbs[0]), np.array(aabbs[1]))
