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

"""Builder for the solar park map.

The park is a real site, so its rows, fence, road and terrain never move. The map is a manifest of reusable pieces
with their placements; this builder loads what the manifest lists and lets the seed decide only what may vary:
how much of the vegetation outside the fence stands. Sun, sky and wind belong to the family's options.

Adding a piece to the map is a new file and a manifest entry, never a change here. Changing how the seed varies
the world is a number in CONFIG.
"""

from __future__ import annotations

import json
import os
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import pybullet as p
import swarm_worlds

SOLAR_ASSET_DIR = os.path.join(swarm_worlds.maps_dir(), "custom", "solar")

CONFIG: Dict[str, Any] = {
    "seed_offset": 0x501A2,                      # own stream: the map must not ride the other seed draws
    "density": {"near": (0.6, 1.0),              # share of the trees by the fence and the road a seed keeps
                "mid": (0.5, 1.0),               # share of the trees on the slopes a seed keeps
                "far": (1.0, 1.0)},              # the forest mass on the far hills always stands
    "cylinder_radius_m": 0.13,                   # trunk and post stand-in for the drone to hit
    "cylinder_height_share": 0.6,                # of the piece's height, so a crown is flown through, not into
}

_FLAG_BITS = {"double_sided": "VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY", "glass": "VISUAL_SHAPE_GLASS",
              "tree_cache": "VISUAL_SHAPE_RENDER_TREE_CACHE"}


@lru_cache(maxsize=2)
def solar_manifest(asset_dir: str = SOLAR_ASSET_DIR) -> Dict[str, Any]:
    """The map's manifest: every item with its files and flags, and every placement with its tags."""
    path = os.path.join(asset_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"solar map manifest missing: {path}")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def solar_densities(seed: int) -> Dict[str, float]:
    """The share of each vegetation tier this seed keeps, drawn from the map's own random stream."""
    rng = random.Random((int(seed) ^ CONFIG["seed_offset"]) & 0xFFFFFFFF)
    return {tier: rng.uniform(low, high) for tier, (low, high) in CONFIG["density"].items()}


def _stands(place: Dict[str, Any], densities: Dict[str, float]) -> bool:
    """Whether a placement is part of this seed's world: fixed pieces always, vegetation by its rank."""
    tier = place.get("tier")
    return tier is None or place.get("rank", 0.0) < densities.get(tier, 1.0)


class _Shapes:
    """Visual and collision shapes of the map's items, created once per item and scale and shared by its bodies."""

    def __init__(self, cli: int, asset_dir: str, items: Dict[str, Any]):
        """Remember where the files are; nothing is loaded until a placement asks."""
        self.cli = cli
        self.asset_dir = asset_dir
        self.items = items
        self.cache: Dict[Tuple[str, Tuple[float, ...]], Tuple[int, int]] = {}

    def get(self, name: str, scale: List[float]) -> Tuple[int, int]:
        """The visual and collision shape ids of an item at a scale."""
        key = (name, tuple(round(float(s), 4) for s in scale))
        if key not in self.cache:
            item = self.items[name]
            path = os.path.join(self.asset_dir, item["folder"], item["obj"])
            if not os.path.exists(path):
                raise FileNotFoundError(f"solar map asset missing: {path}")
            self.cache[key] = (self._visual(item, path, scale), self._collision(item, path, scale))
        return self.cache[key]

    def _visual(self, item: Dict[str, Any], path: str, scale: List[float]) -> int:
        """The item's render shape with the flags its record asks for; a matte piece returns no sheen."""
        flags = 0
        for flag in item.get("flags", []):
            flags |= getattr(p, _FLAG_BITS.get(flag, ""), 0)
        return p.createVisualShape(p.GEOM_MESH, fileName=path, meshScale=list(scale), flags=flags,
                                   specularColor=[float(item.get("specular", 0.0))] * 3, physicsClientId=self.cli)

    def _collision(self, item: Dict[str, Any], path: str, scale: List[float]) -> int:
        """What the drone can hit: the mesh itself, its bounding box, a trunk-sized cylinder, or nothing."""
        kind = item.get("collision", "none")
        low = [a * s for a, s in zip(item["bounds_min"], scale)]
        high = [a * s for a, s in zip(item["bounds_max"], scale)]
        if kind == "mesh":
            flags = p.GEOM_FORCE_CONCAVE_TRIMESH | getattr(p, "GEOM_CONCAVE_BVH_CACHE", 0)
            return p.createCollisionShape(p.GEOM_MESH, fileName=path, meshScale=list(scale), flags=flags,
                                          physicsClientId=self.cli)
        if kind == "box":
            half = [max((h - l) / 2.0, 0.01) for l, h in zip(low, high)]
            centre = [(h + l) / 2.0 for l, h in zip(low, high)]
            return p.createCollisionShape(p.GEOM_BOX, halfExtents=half, collisionFramePosition=centre,
                                          physicsClientId=self.cli)
        if kind == "cylinder":
            height = (high[2] - low[2]) * CONFIG["cylinder_height_share"]
            return p.createCollisionShape(p.GEOM_CYLINDER, radius=CONFIG["cylinder_radius_m"], height=height,
                                          collisionFramePosition=[0.0, 0.0, low[2] + height / 2.0],
                                          physicsClientId=self.cli)
        return -1


def build_solar_map(seed: int = 0, cli: int = 0, asset_dir: Optional[str] = None) -> Dict[str, Any]:
    """Build the solar park inside an existing PyBullet world.

    Returns the body ids per group, the densities the seed drew, and the counts of what was placed.
    """
    asset_dir = asset_dir or SOLAR_ASSET_DIR
    manifest = solar_manifest(asset_dir)
    shapes = _Shapes(cli, asset_dir, manifest["items"])
    densities = solar_densities(seed)
    bodies: Dict[str, List[int]] = {}
    triangles = 0
    for place in manifest["placements"]:
        if not _stands(place, densities):
            continue
        item = manifest["items"][place["item"]]
        visual, collision = shapes.get(place["item"], place["scale"])
        body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
                                 basePosition=place["position"], baseOrientation=place["quaternion"],
                                 physicsClientId=cli)
        p.changeVisualShape(body, -1, rgbaColor=[1, 1, 1, 1], specularColor=[float(item.get("specular", 0.0))] * 3,
                            physicsClientId=cli)
        bodies.setdefault(item["group"], []).append(body)
        triangles += item["triangles"]
    return {"bodies": bodies, "densities": densities, "triangles": triangles,
            "body_count": sum(len(ids) for ids in bodies.values())}
