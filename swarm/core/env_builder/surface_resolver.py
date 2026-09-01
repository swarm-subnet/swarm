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

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import pybullet as p

from .sar_types import BodyCategory

NORMAL_Z_FLAT = 0.85
NORMAL_Z_SLOPE = 0.70
AABB_TOP_TOLERANCE = 0.20
# Clears the tallest mountain peak; a lower start drops the ray inside the rock.
RAYCAST_TOP_Z = 250.0
RAYCAST_BOTTOM_Z = -5.0


@dataclass
class SurfaceHit:
    support_uid: int
    surface_z: float
    normal: Tuple[float, float, float]
    category: str
    is_slope: bool


def _to_value(category) -> str:
    if isinstance(category, BodyCategory):
        return category.value
    return str(category)


def resolve_surface(
    cli: int,
    x: float,
    y: float,
    body_tags: Dict[int, str],
    accepted_categories: Iterable,
    *,
    max_descent_iterations: int = 12,
) -> Optional[SurfaceHit]:
    accepted: Set[str] = {_to_value(c) for c in accepted_categories}
    top_z = RAYCAST_TOP_Z
    for _ in range(max_descent_iterations):
        top = (float(x), float(y), top_z)
        bottom = (float(x), float(y), RAYCAST_BOTTOM_Z)
        hits = p.rayTest(top, bottom, physicsClientId=cli)
        if not hits:
            return None
        raw = hits[0]
        uid = int(raw[0])
        if uid < 0:
            return None
        hit_pos = raw[3]
        hit_normal = raw[4]
        if body_tags.get(uid) in accepted:
            # Below standable ground is the inside of the world, never a surface.
            return _classify_hit(cli, uid, hit_pos, hit_normal, body_tags, accepted)
        next_top = float(hit_pos[2]) - 1e-3
        if next_top <= RAYCAST_BOTTOM_Z:
            return None
        top_z = next_top
    return None


def _classify_hit(
    cli: int,
    uid: int,
    hit_pos,
    hit_normal,
    body_tags: Dict[int, str],
    accepted: Set[str],
) -> Optional[SurfaceHit]:
    category = body_tags.get(int(uid))
    if category is None:
        return None
    if category not in accepted:
        return None
    nx, ny, nz = float(hit_normal[0]), float(hit_normal[1]), float(hit_normal[2])
    is_slope_tag = category == BodyCategory.SUPPORT_SLOPE.value
    is_terrain_tag = category == BodyCategory.SUPPORT_TERRAIN.value
    if is_slope_tag or is_terrain_tag:
        if nz < NORMAL_Z_SLOPE:
            return None
    else:
        if nz < NORMAL_Z_FLAT:
            return None
    surface_z = float(hit_pos[2])
    if not _on_surface(cli, uid, hit_pos, surface_z):
        return None
    is_slope = (nz < 0.985) or is_slope_tag
    return SurfaceHit(
        support_uid=int(uid),
        surface_z=surface_z,
        normal=(nx, ny, nz),
        category=category,
        is_slope=is_slope,
    )


def _on_surface(cli: int, uid: int, hit_pos, surface_z: float) -> bool:
    try:
        mn, mx = p.getAABB(uid, physicsClientId=cli)
    except p.error:
        return False
    if abs(surface_z - mx[2]) <= AABB_TOP_TOLERANCE:
        return True
    if (
        mn[0] - 0.05 <= hit_pos[0] <= mx[0] + 0.05
        and mn[1] - 0.05 <= hit_pos[1] <= mx[1] + 0.05
    ):
        body_height = max(0.5, mx[2] - mn[2])
        if surface_z >= mn[2] + 0.5 * body_height:
            return True
    return False
