from __future__ import annotations

from typing import Iterable, Optional, Set

import pybullet as p

from .body_tagger import BodyTagger
from .sar_types import BodyCategory


def classify_body(cli: int, uid: int, *, challenge_type: int) -> str:
    try:
        shape_data = p.getCollisionShapeData(uid, -1, physicsClientId=cli)
    except p.error:
        shape_data = None
    shape_type = shape_data[0][2] if shape_data else None

    try:
        mn, mx = p.getAABB(uid, physicsClientId=cli)
    except p.error:
        return BodyCategory.OBSTACLE_OTHER.value

    aabb_size_x = mx[0] - mn[0]
    aabb_size_y = mx[1] - mn[1]
    aabb_size_z = mx[2] - mn[2]
    is_flat = aabb_size_z < 0.5
    is_large = aabb_size_x > 5.0 and aabb_size_y > 5.0

    if shape_type in (p.GEOM_MESH, p.GEOM_PLANE):
        if is_large:
            return BodyCategory.SUPPORT_TERRAIN.value
        return BodyCategory.OBSTACLE_OTHER.value

    if shape_type == p.GEOM_BOX:
        if is_flat and is_large:
            return (
                BodyCategory.SUPPORT_FLOOR.value
                if challenge_type == 5
                else BodyCategory.SUPPORT_TERRAIN.value
            )
        if challenge_type in (1, 4):
            if aabb_size_z > 2.0 and (aabb_size_x > 1.5 or aabb_size_y > 1.5):
                return BodyCategory.SUPPORT_ROOFTOP.value
        return BodyCategory.OBSTACLE_OTHER.value

    if shape_type is None:
        if is_flat and is_large:
            return (
                BodyCategory.SUPPORT_FLOOR.value
                if challenge_type == 5
                else BodyCategory.SUPPORT_TERRAIN.value
            )
        if challenge_type in (1, 4):
            if aabb_size_z > 2.0 and (aabb_size_x > 1.5 or aabb_size_y > 1.5):
                return BodyCategory.SUPPORT_ROOFTOP.value

    return BodyCategory.OBSTACLE_OTHER.value


def tag_world_after_build(
    cli: int,
    tagger: BodyTagger,
    *,
    challenge_type: int,
    body_range: Iterable[int],
    victim_uids: Optional[Iterable[int]] = None,
    support_uid: Optional[int] = None,
) -> None:
    victim_set: Set[int] = set(int(u) for u in (victim_uids or []))
    pre_tagged = set(tagger.body_tags.keys())
    for uid in body_range:
        if uid in pre_tagged:
            continue
        if uid in victim_set:
            tagger.tag_existing(uid, BodyCategory.VICTIM)
            continue
        if support_uid is not None and uid == support_uid:
            continue
        cat = classify_body(cli, uid, challenge_type=challenge_type)
        tagger.tag_existing(uid, cat)


def enumerate_bodies(cli: int) -> list[int]:
    n = p.getNumBodies(physicsClientId=cli)
    return [int(p.getBodyUniqueId(i, physicsClientId=cli)) for i in range(n)]


def build_and_tag_map(
    cli: int,
    seed: int,
    challenge_type: int,
    *,
    start=None,
    goal=None,
    safe_zone_radius: float = 5.0,
    sar_mode: bool = False,
) -> BodyTagger:
    from swarm.core.maps.city import build_city as build_city_map
    from swarm.core.maps.forest import build_forest_map
    from swarm.core.maps.mountain import build_mountain_map
    from swarm.core.maps.open import build_open_world
    from swarm.core.maps.village import build_village_map
    from swarm.core.maps.warehouse import build_warehouse_map

    tagger = BodyTagger(cli)
    n_before = p.getNumBodies(physicsClientId=cli)

    safe_zones: list = []
    if start is not None:
        safe_zones.append((float(start[0]), float(start[1])))
    if goal is not None:
        safe_zones.append((float(goal[0]), float(goal[1])))

    forest_assets = None
    if challenge_type == 1:
        build_city_map(cli, seed, safe_zones, safe_zone_radius)
    elif challenge_type == 2:
        build_open_world(cli, seed, start=start, goal=goal, sar_mode=sar_mode)
    elif challenge_type == 3:
        build_mountain_map(cli, seed, safe_zones, safe_zone_radius)
    elif challenge_type == 4:
        build_village_map(cli, seed, safe_zones, safe_zone_radius)
    elif challenge_type == 5:
        build_warehouse_map(seed=seed, cli=cli, start=start, goal=goal)
    elif challenge_type == 6:
        forest_assets = build_forest_map(cli, seed, safe_zones, max(safe_zone_radius, 8.0))
    else:
        raise ValueError(f"unknown challenge_type {challenge_type}")

    if forest_assets:
        for uid in forest_assets.get("trees", ()):
            tagger.tag_existing(uid, BodyCategory.OBSTACLE_CANOPY)
        for uid in forest_assets.get("props", ()):
            tagger.tag_existing(uid, BodyCategory.OBSTACLE_CLUTTER)

    n_after = p.getNumBodies(physicsClientId=cli)
    new_uids = enumerate_bodies(cli)[n_before:n_after]
    tag_world_after_build(
        cli,
        tagger,
        challenge_type=challenge_type,
        body_range=new_uids,
    )
    return tagger
