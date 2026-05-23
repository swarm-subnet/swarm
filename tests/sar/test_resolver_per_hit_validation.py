"""B.3.2 — resolver rejects wall-side hits, accepts gentle slopes, rejects
steep slopes."""
from __future__ import annotations

import math

import pybullet as p

from swarm.core.env_builder.body_tagger import BodyTagger
from swarm.core.env_builder.sar_types import BodyCategory, SUPPORT_CATEGORIES
from swarm.core.env_builder.surface_resolver import resolve_surface


def _flat_ground(cli, tagger, *, z=0.0):
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[40.0, 40.0, 0.1], physicsClientId=cli,
    )
    return tagger.create_body(
        BodyCategory.SUPPORT_TERRAIN,
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=[0.0, 0.0, z],
    )


def _building(cli, tagger, *, position, half_extents=(2.0, 2.0, 5.0)):
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=list(half_extents), physicsClientId=cli,
    )
    return tagger.create_body(
        BodyCategory.SUPPORT_ROOFTOP,
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=position,
    )


def _slope(cli, tagger, *, position, tilt_deg, category=BodyCategory.SUPPORT_SLOPE):
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[3.0, 3.0, 0.1], physicsClientId=cli,
    )
    quat = p.getQuaternionFromEuler([0.0, math.radians(tilt_deg), 0.0])
    return tagger.create_body(
        category,
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        basePosition=list(position),
        baseOrientation=quat,
    )


def test_rejects_wall_side_hit(sar_pybullet):
    cli = sar_pybullet
    tagger = BodyTagger(cli)
    _flat_ground(cli, tagger, z=-0.1)
    _building(cli, tagger, position=[0.0, 0.0, 5.0])
    # Right at the rooftop centre — accept.
    hit = resolve_surface(cli, 0.0, 0.0, tagger.body_tags, SUPPORT_CATEGORIES)
    assert hit is not None and hit.category == "SUPPORT_ROOFTOP"
    # Side, just outside the rooftop XY — must NOT return rooftop; ground only.
    hit = resolve_surface(cli, 2.5, 0.0, tagger.body_tags, SUPPORT_CATEGORIES)
    assert hit is not None
    assert hit.category != "SUPPORT_ROOFTOP"


def test_slope_accept_35deg(sar_pybullet):
    cli = sar_pybullet
    tagger = BodyTagger(cli)
    _slope(cli, tagger, position=[0.0, 0.0, 1.5], tilt_deg=35.0)
    hit = resolve_surface(cli, 0.0, 0.0, tagger.body_tags, SUPPORT_CATEGORIES)
    assert hit is not None
    assert hit.category == "SUPPORT_SLOPE"
    assert hit.is_slope


def test_slope_reject_55deg(sar_pybullet):
    cli = sar_pybullet
    tagger = BodyTagger(cli)
    _slope(cli, tagger, position=[0.0, 0.0, 1.5], tilt_deg=55.0)
    hit = resolve_surface(cli, 0.0, 0.0, tagger.body_tags, SUPPORT_CATEGORIES)
    # Normal at 55deg tilt has nz = cos(55) ~= 0.57 < NORMAL_Z_SLOPE (0.70) → reject.
    assert hit is None or hit.category != "SUPPORT_SLOPE"
