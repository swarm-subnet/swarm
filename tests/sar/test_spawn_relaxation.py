from __future__ import annotations

import pybullet as p
import pytest

from swarm.core.env_builder.sar_types import BodyCategory
from swarm.core.env_builder.spawn_pipeline import (
    HOVER_COLUMN_TOP_Z,
    SARSpawnError,
    find_spawn_xy,
)


def _slab(cli, half_extents, position, tags, category):
    shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=half_extents, physicsClientId=cli
    )
    uid = p.createMultiBody(
        baseCollisionShapeIndex=shape, basePosition=position, physicsClientId=cli
    )
    tags[uid] = category.value
    return uid


def _canopy_world(cli, ceiling_z):
    """Flat ground under a solid ceiling, so no spot clears the full hover column."""
    tags: dict[int, str] = {}
    _slab(cli, [60.0, 60.0, 0.05], [0.0, 0.0, -0.05], tags, BodyCategory.SUPPORT_TERRAIN)
    _slab(cli, [60.0, 60.0, 0.05], [0.0, 0.0, ceiling_z], tags, BodyCategory.OBSTACLE_CANOPY)
    return tags


def test_relaxed_pass_places_under_a_low_canopy(sar_pybullet):
    tags = _canopy_world(sar_pybullet, ceiling_z=HOVER_COLUMN_TOP_Z * 0.8)

    x, y, hit = find_spawn_xy(
        sar_pybullet, map_seed=7, challenge_type=3, body_tags=tags,
    )

    assert hit is not None
    assert abs(hit.surface_z) < 0.5


def test_places_on_an_off_type_support_when_nothing_else_exists(sar_pybullet):
    tags: dict[int, str] = {}
    _slab(sar_pybullet, [60.0, 60.0, 0.05], [0.0, 0.0, -0.05], tags, BodyCategory.SUPPORT_WALKWAY)

    # challenge_type 1 accepts terrain only
    x, y, hit = find_spawn_xy(
        sar_pybullet, map_seed=7, challenge_type=1, body_tags=tags,
    )

    assert hit is not None
    assert abs(hit.surface_z) < 0.5


def test_still_raises_when_there_is_no_ground(sar_pybullet):
    tags: dict[int, str] = {}
    _slab(sar_pybullet, [60.0, 60.0, 0.05], [0.0, 0.0, 4.0], tags, BodyCategory.OBSTACLE_CANOPY)

    with pytest.raises(SARSpawnError):
        find_spawn_xy(
            sar_pybullet, map_seed=7, challenge_type=3, body_tags=tags,
        )
