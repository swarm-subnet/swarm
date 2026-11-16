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
"""The test site builder: every manifest placement stands where it is put, terrain and field are the land, houses
and walls block, and visual-only pieces have no body to hit."""
from __future__ import annotations

import json

import pybullet as p
import pytest

from swarm.core.maps.testsite import builder

_BOX = """v -0.5 -0.5 0
v 0.5 -0.5 0
v 0.5 0.5 0
v -0.5 0.5 0
v -0.5 -0.5 1
v 0.5 -0.5 1
v 0.5 0.5 1
v -0.5 0.5 1
f 1 3 2
f 1 4 3
f 5 6 7
f 5 7 8
f 1 2 6
f 1 6 5
f 2 3 7
f 2 7 6
f 3 4 8
f 3 8 7
f 4 1 5
f 4 5 8
"""


def _item(name, group, collision):
    """A record for the unit box, as the export writes them."""
    return {"name": name, "obj": "box.obj", "texture": "", "triangles": 12, "group": group, "folder": "kit",
            "flags": [], "collision": collision, "specular": 0.0, "bounds_min": [-0.5, -0.5, 0.0],
            "bounds_max": [0.5, 0.5, 1.0]}


@pytest.fixture
def site(tmp_path):
    """An asset folder with a terrain tile, the field, a house, a wall and a visual-only weed patch."""
    (tmp_path / "kit").mkdir()
    (tmp_path / "kit" / "box.obj").write_text(_BOX)
    items = {"tile": _item("tile", "terrain", "mesh"), "field": _item("field", "field", "mesh"),
             "house": _item("house", "buildings", "mesh"), "wall": _item("wall", "edges", "box"),
             "weeds": _item("weeds", "edges", "none")}
    placements = [{"item": "tile", "position": [0.0, 0.0, -1.0], "quaternion": [0, 0, 0, 1], "scale": [200.0, 200.0, 1.0]},
                  {"item": "field", "position": [-20.0, 0.0, -0.5], "quaternion": [0, 0, 0, 1], "scale": [20.0, 20.0, 0.6]},
                  {"item": "house", "position": [30.0, 0.0, 0.0], "quaternion": [0, 0, 0, 1], "scale": [8.0, 10.0, 7.0]},
                  {"item": "wall", "position": [0.0, 20.0, 0.0], "quaternion": [0, 0, 0, 1], "scale": [30.0, 0.6, 1.2]},
                  {"item": "weeds", "position": [-20.0, 0.0, 0.1], "quaternion": [0, 0, 0, 1], "scale": [1.0, 1.0, 0.3]}]
    (tmp_path / "manifest.json").write_text(json.dumps({"version": 1, "items": items, "placements": placements}))
    builder.testsite_manifest.cache_clear()
    yield str(tmp_path)
    builder.testsite_manifest.cache_clear()


def test_the_site_stands_as_its_manifest_says(site):
    """Each placement is one body; terrain and field are land, the house and wall block, the weeds only show."""
    cli = p.connect(p.DIRECT)
    try:
        built = builder.build_testsite_map(seed=7, cli=cli, asset_dir=site)
        assert built["body_count"] == 5 and built["triangles"] == 60 and built["trees"] == 0
        lift = -built["datum_m"]
        groups = built["bodies"]
        assert built["land"] == frozenset(groups["terrain"] + groups["field"])
        assert set(groups["buildings"]) <= built["blockers"] and not built["land"] & built["blockers"]
        down = p.rayTest([30.0, 0.0, 50.0], [30.0, 0.0, -5.0], physicsClientId=cli)[0]
        assert down[0] in groups["buildings"] and down[3][2] == pytest.approx(7.0 + lift, abs=1e-3)
        across = p.rayTest([0.0, 10.0, 0.6 + lift], [0.0, 30.0, 0.6 + lift], physicsClientId=cli)[0]
        assert across[0] in groups["edges"]
        weed = p.rayTest([-20.0, 0.0, 5.0], [-20.0, 0.0, -5.0], physicsClientId=cli)[0]
        assert weed[0] in groups["field"], "a visual-only piece must not stop a ray"
    finally:
        p.disconnect(cli)


def test_the_same_seed_builds_the_same_site(site):
    """The real place does not move with the seed: two seeds give the same bodies in the same places."""
    poses = []
    for seed in (1, 2):
        cli = p.connect(p.DIRECT)
        try:
            built = builder.build_testsite_map(seed=seed, cli=cli, asset_dir=site)
            poses.append([p.getBasePositionAndOrientation(uid, physicsClientId=cli) for group in sorted(built["bodies"])
                          for uid in built["bodies"][group]])
        finally:
            p.disconnect(cli)
    assert poses[0] == poses[1]


def test_the_site_is_lowered_by_the_fields_lowest_point(site):
    """The export keeps heights above sea level; the site is lowered by the field's lowest point, down to the metre,
    so the field stands just above zero wherever the real place lies."""
    manifest = builder.testsite_manifest(site)
    assert builder.datum_m(manifest) == -1.0
    high = dict(manifest, placements=[dict(place, position=[x, y, z + 1005.4]) for place in manifest["placements"]
                                      for x, y, z in [place["position"]]])
    assert builder.datum_m(high) == 1004.0
    assert builder.datum_m(dict(manifest, placements=[])) == 0.0
