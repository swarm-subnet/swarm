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

"""Solar map forest: every tree of the scene in one table, thinned by the seed, drawn as one body, trunks near the fence.

Skipped unless SOLAR_ASSET_DIR points at a built copy of the map with its forest table, like the movers tests.
"""
from __future__ import annotations

import glob
import os
import tempfile

import numpy as np
import pybullet as p
import pytest

from swarm.core.maps.solar.builder import CONFIG, SOLAR_ASSET_DIR, _forest_table, build_solar_map, solar_densities, solar_manifest

ASSET_DIR = os.environ.get("SOLAR_ASSET_DIR", SOLAR_ASSET_DIR)


def _has_forest(asset_dir):
    """Whether the built map at asset_dir lists a forest."""
    path = os.path.join(asset_dir, "manifest.json")
    return os.path.exists(path) and "forest" in solar_manifest(asset_dir)


pytestmark = pytest.mark.skipif(not _has_forest(ASSET_DIR), reason=f"solar map forest not built at {ASSET_DIR}")


def _expected(seed):
    """The table, which of its trees a seed keeps, and which of those get a trunk, computed apart from the builder."""
    forest = solar_manifest(ASSET_DIR)["forest"]
    table = _forest_table(os.path.join(ASSET_DIR, forest["folder"], forest["table"]))
    densities = solar_densities(seed)
    keep = table["rank"] < np.array([densities.get(name, 1.0) for name in forest["tiers"]])[table["tier"]]
    trunk = keep & (table["tier"] == forest["tiers"].index("near")) & (table["scale"][:, 2] >= CONFIG["forest_trunk_min_m"])
    return keep, trunk


@pytest.fixture
def client():
    """A DIRECT client, dropped after the test."""
    cli = p.connect(p.DIRECT)
    yield cli
    p.disconnect(cli)


def test_a_seed_stands_the_trees_its_densities_keep(client):
    """The builder stands exactly the trees whose rank is below the seed's density for their zone."""
    world = build_solar_map(seed=0, cli=client, asset_dir=ASSET_DIR, groups=("plants",))
    keep, _ = _expected(0)
    assert world["trees"] == int(keep.sum())


def test_near_tall_trees_get_a_collision_only_trunk(client):
    """One visual body carries the forest; the other plants bodies hold a cylinder no camera sees for every near tall tree."""
    world = build_solar_map(seed=0, cli=client, asset_dir=ASSET_DIR, groups=("plants",))
    forest, *trunks = world["bodies"]["plants"]
    _, trunk = _expected(0)
    shapes = [p.getCollisionShapeData(body, -1, physicsClientId=client) for body in trunks]
    assert sum(len(parts) for parts in shapes) == int(trunk.sum())
    assert all(len(parts) <= CONFIG["forest_trunks_per_body"] for parts in shapes)
    assert all(part[2] == p.GEOM_CYLINDER for parts in shapes for part in parts)
    assert p.getVisualShapeData(forest, physicsClientId=client)
    for body in trunks[:50]:
        visual = p.getVisualShapeData(body, physicsClientId=client)[0]
        assert visual[2] == p.GEOM_MESH and visual[7][3] == 0.0
    (x, y, z), _ = p.getBasePositionAndOrientation(trunks[0], physicsClientId=client)
    view = p.computeViewMatrix([x + 2.0, y, z], [x, y, z], [0, 0, 1])
    proj = p.computeProjectionMatrixFOV(20, 1, 0.05, 50)
    seg = np.asarray(p.getCameraImage(32, 32, view, proj, renderer=p.ER_TINY_RENDERER,
                                      flags=getattr(p, "ER_SWARM_RAYCAST", 0), physicsClientId=client)[4])
    assert trunks[0] not in set(seg.ravel().tolist())
    base, _ = p.getBasePositionAndOrientation(trunks[0], physicsClientId=client)
    centre = np.add(base, shapes[0][0][5])
    hit = p.rayTest((centre + [1.0, 0.0, 0.0]).tolist(), (centre - [1.0, 0.0, 0.0]).tolist(), physicsClientId=client)[0]
    assert hit[0] in trunks


def test_two_seeds_stand_different_forests():
    """Seeds that draw different densities stand different numbers of trees."""
    counts = []
    for seed in (0, 1):
        cli = p.connect(p.DIRECT)
        counts.append(build_solar_map(seed=seed, cli=cli, asset_dir=ASSET_DIR, groups=("plants",))["trees"])
        p.disconnect(cli)
    assert counts[0] != counts[1]


def test_the_forest_file_is_removed_after_the_build(client):
    """The per-build forest file the renderer reads is gone once the body stands."""
    before = set(glob.glob(os.path.join(tempfile.gettempdir(), "*.fst")))
    build_solar_map(seed=0, cli=client, asset_dir=ASSET_DIR, groups=("plants",))
    assert set(glob.glob(os.path.join(tempfile.gettempdir(), "*.fst"))) == before


def test_an_engine_without_instancing_is_refused(client, monkeypatch):
    """A wheel without the instanced flag fails loudly instead of drawing no forest."""
    monkeypatch.delattr(p, "VISUAL_SHAPE_RENDER_INSTANCED", raising=False)
    with pytest.raises(RuntimeError, match="VISUAL_SHAPE_RENDER_INSTANCED"):
        build_solar_map(seed=0, cli=client, asset_dir=ASSET_DIR, groups=("plants",))
