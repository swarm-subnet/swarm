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

"""Builder for the Andorra test site map: the real field, its walls and lane, the trees and houses around it, and the
valley out to the horizon, all from the manifest the Blender export wrote.

The place is fixed, so every placement stands where the manifest puts it, the whole site lowered by the field's lowest
point: the export keeps heights above sea level, and the world's rays and floor expect the ground near zero. The
terrain rings and the ploughed field are
the ground a drone lands on; every other solid piece is a blocker. Trees stand as one instanced forest body with a
collision trunk for each near tree, the way the solar park's forest does.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, List, Optional

from swarm.core.maps.solar.builder import CONFIG as SOLAR_CONFIG
from swarm.core.maps.solar.builder import _fixed_body, _Shapes, _stand_forest

try:
    import swarm_worlds
except ImportError:                      # Blender's Python and bare checkouts read a path they are given
    swarm_worlds = None

# Always the installed swarm-worlds package, as for the solar park, so every validator scores the same world.
TESTSITE_ASSET_DIR = os.path.join(swarm_worlds.maps_dir(), "custom", "testsite") if swarm_worlds is not None else ""

CONFIG: Dict[str, Any] = {
    "land_groups": ("terrain", "field"),         # what a drone can land on
    "not_blockers": ("terrain", "field", "plants_visual"),
    "forest_density": {"near": 1.0, "mid": 1.0, "far": 1.0},   # the real place: every tree stands in every seed
}


@lru_cache(maxsize=2)
def testsite_manifest(asset_dir: str) -> Dict[str, Any]:
    """The manifest of the test site's assets, read once per process."""
    with open(os.path.join(asset_dir, "manifest.json")) as handle:
        return json.load(handle)


def datum_m(manifest: Dict[str, Any]) -> float:
    """How far the site is lowered: the field's lowest point, down to the whole metre; 0 without a field."""
    lows = [place["position"][2] + manifest["items"][place["item"]]["bounds_min"][2] * place["scale"][2]
            for place in manifest["placements"] if manifest["items"][place["item"]]["group"] == "field"]
    return float(math.floor(min(lows))) if lows else 0.0


def build_testsite_map(seed: int = 0, cli: int = 0, asset_dir: Optional[str] = None) -> Dict[str, Any]:
    """Build the test site inside an existing PyBullet world, lowered by datum_m.

    Returns the body ids per group, the land bodies (terrain and field), every other solid body a drone can meet, the
    number of trees standing, the placed triangle count and how far the site was lowered.
    """
    asset_dir = asset_dir or TESTSITE_ASSET_DIR
    manifest = testsite_manifest(asset_dir)
    datum = datum_m(manifest)
    shapes = _Shapes(cli, asset_dir, manifest["items"])
    bodies: Dict[str, List[int]] = {}
    triangles = 0
    for place in manifest["placements"]:
        item = manifest["items"][place["item"]]
        visual, collision = shapes.get(place["item"], place["scale"])
        x, y, z = place["position"]
        body = _fixed_body(cli, collision, visual, [x, y, z - datum], place["quaternion"])
        bodies.setdefault(item["group"], []).append(body)
        triangles += int(item.get("triangles", 0))
    trees = 0
    forest = manifest.get("forest")
    if forest:
        standing, trees = _stand_forest(cli, asset_dir, forest, CONFIG["forest_density"], lift_m=-datum)
        bodies.setdefault("plants", []).extend(standing)
    land = frozenset(uid for group in CONFIG["land_groups"] for uid in bodies.get(group, ()))
    blockers = frozenset(uid for group, uids in bodies.items() if group not in CONFIG["not_blockers"] for uid in uids)
    return {"bodies": bodies, "land": land, "blockers": blockers, "trees": trees, "triangles": triangles,
            "asset_dir": asset_dir, "seed": seed, "body_count": sum(len(ids) for ids in bodies.values()), "datum_m": datum,
            "trunk_radius_m": SOLAR_CONFIG["cylinder_radius_m"]}
