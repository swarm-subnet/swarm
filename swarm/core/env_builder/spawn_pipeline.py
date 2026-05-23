from __future__ import annotations

import random
from typing import Optional, Tuple

import pybullet as p

from .sar_types import BodyCategory
from .surface_resolver import SurfaceHit, resolve_surface
from .victim import accepted_categories_for


MAX_SPAWN_ATTEMPTS = 50
NO_TOUCH_SPHERE_RADIUS = 0.8
HOVER_COLUMN_TOP_Z = 5.0


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
) -> bool:
    bottom = (x, y, surface_z + 0.05)
    top = (x, y, surface_z + HOVER_COLUMN_TOP_Z)
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
) -> bool:
    r = NO_TOUCH_SPHERE_RADIUS
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
    on_attempt=None,
) -> Tuple[float, float, SurfaceHit]:
    bound = float(bounds) if bounds is not None else _DEFAULT_MAP_BOUNDS.get(challenge_type, 20.0)
    accepted = accepted_categories_for(challenge_type)
    last_reason = "no_attempts"
    for attempt in range(MAX_SPAWN_ATTEMPTS):
        x, y = _sample_candidate(map_seed, attempt, bound)
        hit = resolve_surface(cli, x, y, body_tags, accepted)
        if hit is None:
            last_reason = "no_support_hit"
            if on_attempt is not None:
                on_attempt(attempt, "no_support_hit", x, y)
            continue
        if not _hover_column_clear(
            cli, x, y, hit.surface_z, body_tags=body_tags, support_uid=hit.support_uid,
        ):
            last_reason = "hover_column_blocked"
            if on_attempt is not None:
                on_attempt(attempt, "hover_column_blocked", x, y)
            continue
        if not _sphere_obstacle_clear(
            cli, x, y, hit.surface_z, body_tags=body_tags, support_uid=hit.support_uid,
        ):
            last_reason = "no_touch_sphere_blocked"
            if on_attempt is not None:
                on_attempt(attempt, "no_touch_sphere_blocked", x, y)
            continue
        if on_attempt is not None:
            on_attempt(attempt, "accept", x, y)
        return x, y, hit
    raise SARSpawnError(
        f"spawn exhausted {MAX_SPAWN_ATTEMPTS} attempts for seed={map_seed} "
        f"challenge_type={challenge_type}: last_reason={last_reason}"
    )
