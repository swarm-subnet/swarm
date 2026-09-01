"""B.3.1 — surface_resolver synthetic cases.

Builds a small tagged scene covering 20 representative situations:
- flat terrain, flat building rooftops, warehouse floors
- gentle slopes (acceptable), steep slopes (rejected)
- walls (vertical normal, must reject)
- obstacle-only categories (must reject)
- empty positions (no body, must return None)"""
from __future__ import annotations

import pytest
import pybullet as p

from swarm.core.env_builder.body_tagger import BodyTagger
from swarm.core.env_builder.sar_types import BodyCategory, SUPPORT_CATEGORIES
from swarm.core.env_builder.surface_resolver import resolve_surface


def _make_box(cli, tagger, category, *, position, half_extents, yaw=0.0):
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half_extents, physicsClientId=cli,
    )
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    uid = tagger.create_body(
        category,
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=position,
        baseOrientation=quat,
    )
    return uid


def _make_slope(cli, tagger, category, *, position, half_extents, tilt_rad):
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half_extents, physicsClientId=cli,
    )
    quat = p.getQuaternionFromEuler([0.0, tilt_rad, 0.0])
    uid = tagger.create_body(
        category,
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=position,
        baseOrientation=quat,
    )
    return uid


def test_all_20_synthetic_cases(sar_pybullet):
    cli = sar_pybullet
    tagger = BodyTagger(cli)

    # 1. Flat ground tile (SUPPORT_TERRAIN, large flat box).
    ground = _make_box(
        cli, tagger, BodyCategory.SUPPORT_TERRAIN,
        position=[0.0, 0.0, 0.0], half_extents=[20.0, 20.0, 0.1],
    )
    # 2. Building rooftop (SUPPORT_ROOFTOP, tall box) at (10, 0).
    rooftop = _make_box(
        cli, tagger, BodyCategory.SUPPORT_ROOFTOP,
        position=[10.0, 0.0, 3.0], half_extents=[2.0, 2.0, 3.0],
    )
    # 3. Warehouse floor (SUPPORT_FLOOR) at (-15, 0).
    floor = _make_box(
        cli, tagger, BodyCategory.SUPPORT_FLOOR,
        position=[-15.0, 0.0, 0.05], half_extents=[3.0, 3.0, 0.05],
    )
    # 4. Gentle slope (SUPPORT_SLOPE, 30deg).
    import math
    gentle_slope = _make_slope(
        cli, tagger, BodyCategory.SUPPORT_SLOPE,
        position=[20.0, 0.0, 1.0], half_extents=[2.0, 2.0, 0.1],
        tilt_rad=math.radians(30.0),
    )
    # 5. Steep slope (SUPPORT_SLOPE, 60deg) — should be rejected.
    steep_slope = _make_slope(
        cli, tagger, BodyCategory.SUPPORT_SLOPE,
        position=[25.0, 0.0, 1.0], half_extents=[2.0, 2.0, 0.1],
        tilt_rad=math.radians(60.0),
    )
    # 6. Pillar (OBSTACLE_OTHER, cylinder) at (30, 0).
    col_pillar = p.createCollisionShape(
        p.GEOM_CYLINDER, radius=0.5, height=4.0, physicsClientId=cli,
    )
    pillar = tagger.create_body(
        BodyCategory.OBSTACLE_OTHER,
        baseMass=0.0, baseCollisionShapeIndex=col_pillar,
        basePosition=[30.0, 0.0, 2.0],
    )

    sup_set = SUPPORT_CATEGORIES

    # (1) ground at (0,0) — accept, SUPPORT_TERRAIN
    hit = resolve_surface(cli, 0.0, 0.0, tagger.body_tags, sup_set)
    assert hit is not None and hit.category == "SUPPORT_TERRAIN"

    # (2) far-from-anything XY at (0, 50) — still hits ground? ground extent only ±20 → no hit
    hit = resolve_surface(cli, 0.0, 50.0, tagger.body_tags, sup_set)
    assert hit is None

    # (3) ground edge (19, 19) — still on ground
    hit = resolve_surface(cli, 19.0, 19.0, tagger.body_tags, sup_set)
    assert hit is not None and hit.category == "SUPPORT_TERRAIN"

    # (4) rooftop centre (10, 0) — accept SUPPORT_ROOFTOP
    hit = resolve_surface(cli, 10.0, 0.0, tagger.body_tags, sup_set)
    assert hit is not None and hit.category == "SUPPORT_ROOFTOP"

    # (5) building side / wall at (12.5, 0) — outside rooftop XY, falls to ground
    hit = resolve_surface(cli, 12.5, 0.0, tagger.body_tags, sup_set)
    assert hit is not None and hit.category == "SUPPORT_TERRAIN"

    # (6) warehouse floor (-15, 0) — accept SUPPORT_FLOOR
    hit = resolve_surface(cli, -15.0, 0.0, tagger.body_tags, sup_set)
    assert hit is not None and hit.category == "SUPPORT_FLOOR"

    # (7) gentle slope centre (20, 0) — accept SUPPORT_SLOPE, mark is_slope
    hit = resolve_surface(cli, 20.0, 0.0, tagger.body_tags, sup_set)
    assert hit is not None
    assert hit.category == "SUPPORT_SLOPE"
    assert hit.is_slope

    # (8) steep slope centre (25, 0) — reject (normal too steep)
    hit = resolve_surface(cli, 25.0, 0.0, tagger.body_tags, sup_set)
    # Steep slope rejects — falls through to ground if any overlap, else None
    assert hit is None or hit.category != "SUPPORT_SLOPE"

    # (9) pillar top at (30, 0) with OBSTACLE_OTHER not in accepted → reject
    hit = resolve_surface(cli, 30.0, 0.0, tagger.body_tags, sup_set)
    # Could be no hit (pillar caps ray then category-rejected → None) or fall to ground
    if hit is not None:
        assert hit.category != "OBSTACLE_OTHER"

    # (10) accept-only-OBSTACLE filter at (0, 0) → reject ground
    hit = resolve_surface(
        cli, 0.0, 0.0, tagger.body_tags, {BodyCategory.OBSTACLE_OTHER}
    )
    assert hit is None

    # (11) accept-only-FLOOR at (0, 0) where it's TERRAIN → reject
    hit = resolve_surface(
        cli, 0.0, 0.0, tagger.body_tags, {BodyCategory.SUPPORT_FLOOR}
    )
    assert hit is None

    # (12) accept-only-FLOOR at (-15, 0) → accept
    hit = resolve_surface(
        cli, -15.0, 0.0, tagger.body_tags, {BodyCategory.SUPPORT_FLOOR}
    )
    assert hit is not None and hit.category == "SUPPORT_FLOOR"

    # (13) ground hit returns surface_z near 0.1 (top of the box)
    hit = resolve_surface(cli, 5.0, 5.0, tagger.body_tags, sup_set)
    assert hit is not None
    assert abs(hit.surface_z - 0.1) < 0.05

    # (14) rooftop surface_z near 6.0 (3.0 + 3.0 half_extent)
    hit = resolve_surface(cli, 10.0, 0.0, tagger.body_tags, sup_set)
    assert hit is not None
    assert abs(hit.surface_z - 6.0) < 0.05

    # (15) is_slope false on flat ground
    hit = resolve_surface(cli, 0.0, 0.0, tagger.body_tags, sup_set)
    assert hit is not None and not hit.is_slope

    # (16) is_slope true on gentle slope
    hit = resolve_surface(cli, 20.0, 0.0, tagger.body_tags, sup_set)
    assert hit is not None and hit.is_slope

    # (17) untagged body in the column — multi-hit descent ignores it and
    # falls through to the ground below.
    col_extra = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.3, 0.3, 0.3], physicsClientId=cli,
    )
    raw_uid = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_extra,
        basePosition=[7.0, 7.0, 5.0],
        physicsClientId=cli,
    )
    hit = resolve_surface(cli, 7.0, 7.0, tagger.body_tags, sup_set)
    assert hit is not None and hit.category == "SUPPORT_TERRAIN"

    # (18) Hit normal is roughly upward on ground
    hit = resolve_surface(cli, 0.0, 0.0, tagger.body_tags, sup_set)
    assert hit is not None and hit.normal[2] > 0.85

    # (19) is_slope is True for slope-tagged body even if normal almost flat
    almost_flat_slope = _make_slope(
        cli, tagger, BodyCategory.SUPPORT_SLOPE,
        position=[35.0, 0.0, 0.1], half_extents=[2.0, 2.0, 0.05],
        tilt_rad=math.radians(5.0),
    )
    hit = resolve_surface(cli, 35.0, 0.0, tagger.body_tags, sup_set)
    assert hit is not None and hit.is_slope

    # (20) acceptance: hit category matches the body's tag
    hit = resolve_surface(cli, 0.0, 0.0, tagger.body_tags, sup_set)
    assert hit is not None
    assert tagger.body_tags[hit.support_uid] == hit.category
