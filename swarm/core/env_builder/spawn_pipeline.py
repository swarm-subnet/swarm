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

import random
from typing import Optional, Tuple

import pybullet as p

from .sar_types import SUPPORT_CATEGORIES, BodyCategory
from .surface_resolver import SurfaceHit, resolve_surface
from .victim import accepted_categories_for, terrain_slope_deg

MAX_SPAWN_ATTEMPTS = 50
NO_TOUCH_SPHERE_RADIUS = 0.8
HOVER_COLUMN_TOP_Z = 5.0
MAX_SPAWN_SLOPE_DEG = 22.0

# Reached only after the strict pass fails; a dense map can leave no fully clear spot.
RELAXED_SPAWN_ATTEMPTS = 100
_RELAXED_STAGES = ((0.6, 0.5, 1.0), (0.3, 0.25, 1.5))


class SARSpawnError(RuntimeError):
    pass


_DEFAULT_MAP_BOUNDS = {
    1: 25.0,
    2: 30.0,
    3: 30.0,
    4: 25.0,
    5: 12.0,
    6: 20.0,
}


def _hover_column_clear(
    cli: int,
    x: float,
    y: float,
    surface_z: float,
    *,
    body_tags,
    support_uid: int,
    top_z: float = HOVER_COLUMN_TOP_Z,
) -> bool:
    bottom = (x, y, surface_z + 0.05)
    top = (x, y, surface_z + top_z)
    hits = p.rayTest(bottom, top, physicsClientId=cli)
    if not hits:
        return True
    raw = hits[0]
    uid = int(raw[0])
    if uid < 0:
        return True
    if uid == support_uid:
        return True
    tag = body_tags.get(uid)
    if tag == BodyCategory.VICTIM.value:
        return True
    return False


def _sphere_obstacle_clear(
    cli: int,
    x: float,
    y: float,
    surface_z: float,
    *,
    body_tags,
    support_uid: int,
    radius: float = NO_TOUCH_SPHERE_RADIUS,
) -> bool:
    r = radius
    aabb_min = (x - r, y - r, surface_z + 0.01)
    aabb_max = (x + r, y + r, surface_z + r)
    overlaps = p.getOverlappingObjects(aabb_min, aabb_max, physicsClientId=cli)
    if not overlaps:
        return True
    for entry in overlaps:
        uid = int(entry[0])
        if uid == support_uid:
            continue
        tag = body_tags.get(uid)
        if tag is None or tag == BodyCategory.VICTIM.value:
            continue
        if isinstance(tag, str) and tag.startswith("SUPPORT_"):
            continue
        return False
    return True


def _sample_candidate(
    map_seed: int, attempt: int, bounds: float,
) -> Tuple[float, float]:
    rng = random.Random((map_seed * 1_000_003) ^ (attempt * 9_176_531))
    x = rng.uniform(-bounds, bounds)
    y = rng.uniform(-bounds, bounds)
    return x, y


def find_spawn_xy(
    cli: int,
    *,
    map_seed: int,
    challenge_type: int,
    body_tags,
    bounds: Optional[float] = None,
    near: Optional[Tuple[float, float]] = None,
    max_dist: Optional[float] = None,
) -> Tuple[float, float, SurfaceHit]:
    bound = float(bounds) if bounds is not None else _DEFAULT_MAP_BOUNDS.get(challenge_type, 20.0)
    accepted = accepted_categories_for(challenge_type)
    last_reason = "no_attempts"
    fallback: Optional[Tuple[float, Tuple[float, float, SurfaceHit]]] = None
    flattest: Optional[Tuple[float, Tuple[float, float, SurfaceHit]]] = None
    grounded: Optional[Tuple[float, float, SurfaceHit]] = None
    any_support: Optional[Tuple[float, float, SurfaceHit]] = None
    for attempt in range(MAX_SPAWN_ATTEMPTS + RELAXED_SPAWN_ATTEMPTS):
        if attempt < MAX_SPAWN_ATTEMPTS:
            column_scale, sphere_scale, bound_scale = 1.0, 1.0, 1.0
        else:
            stage = (attempt - MAX_SPAWN_ATTEMPTS) * len(_RELAXED_STAGES) // RELAXED_SPAWN_ATTEMPTS
            column_scale, sphere_scale, bound_scale = _RELAXED_STAGES[stage]
        x, y = _sample_candidate(map_seed, attempt, bound * bound_scale)
        hit = resolve_surface(cli, x, y, body_tags, accepted)
        if hit is None:
            if any_support is None and attempt >= MAX_SPAWN_ATTEMPTS:
                loose = resolve_surface(cli, x, y, body_tags, SUPPORT_CATEGORIES)
                if loose is not None:
                    any_support = (x, y, loose)
            last_reason = "no_support_hit"
            continue
        if grounded is None:
            grounded = (x, y, hit)
        if not _hover_column_clear(
            cli, x, y, hit.surface_z, body_tags=body_tags, support_uid=hit.support_uid,
            top_z=HOVER_COLUMN_TOP_Z * column_scale,
        ):
            last_reason = "hover_column_blocked"
            continue
        if not _sphere_obstacle_clear(
            cli, x, y, hit.surface_z, body_tags=body_tags, support_uid=hit.support_uid,
            radius=NO_TOUCH_SPHERE_RADIUS * sphere_scale,
        ):
            last_reason = "no_touch_sphere_blocked"
            continue
        slope = max(
            terrain_slope_deg(cli, x, y, hit.surface_z, radius=0.4),
            terrain_slope_deg(cli, x, y, hit.surface_z, radius=1.0),
        )
        if slope > MAX_SPAWN_SLOPE_DEG:
            # keep the flattest valid spot so an all-steep map still spawns
            if flattest is None or slope < flattest[0]:
                flattest = (slope, (x, y, hit))
            last_reason = "too_steep"
            continue
        if near is not None and max_dist is not None:
            dist = ((x - near[0]) ** 2 + (y - near[1]) ** 2) ** 0.5
            if dist > max_dist:
                if fallback is None or dist < fallback[0]:
                    fallback = (dist, (x, y, hit))
                last_reason = "beyond_max_dist"
                continue
        return x, y, hit
    if fallback is not None:
        return fallback[1]
    if flattest is not None:
        return flattest[1]
    if grounded is not None:
        return grounded
    if any_support is not None:
        return any_support
    raise SARSpawnError(
        f"spawn exhausted {MAX_SPAWN_ATTEMPTS + RELAXED_SPAWN_ATTEMPTS} attempts for seed={map_seed} "
        f"challenge_type={challenge_type}: last_reason={last_reason}"
    )
