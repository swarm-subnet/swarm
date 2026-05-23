from __future__ import annotations

import pickle

from swarm.core.env_builder.sar_types import (
    BodyCategory,
    SafetyPatch,
    SARWorld,
    SUPPORT_CATEGORIES,
)


def test_enum_members():
    expected = {
        "SUPPORT_TERRAIN", "SUPPORT_ROOFTOP", "SUPPORT_FLOOR",
        "SUPPORT_SLOPE", "SUPPORT_WALKWAY", "VICTIM",
        "OBSTACLE_CANOPY", "OBSTACLE_BEAM",
        "OBSTACLE_CLUTTER", "OBSTACLE_OTHER",
    }
    assert {m.name for m in BodyCategory} == expected
    for member in BodyCategory:
        assert member.value == member.name
    assert BodyCategory.SUPPORT_TERRAIN in SUPPORT_CATEGORIES
    assert BodyCategory.VICTIM not in SUPPORT_CATEGORIES
    assert BodyCategory.OBSTACLE_BEAM not in SUPPORT_CATEGORIES


def test_sar_world_round_trip():
    patch = SafetyPatch(support_uid=2, xy=(1.5, -2.0), surface_z=0.3)
    world = SARWorld(
        victim_uids=[10, 11, 12],
        victim_aabb=((1.0, -1.0, 0.0), (2.0, 1.0, 1.8)),
        victim_centre=(1.5, 0.0, 0.9),
        support_uid=2,
        support_category=BodyCategory.SUPPORT_TERRAIN.value,
        surface_z=0.3,
        safety_patch=patch,
        body_tags={2: "SUPPORT_TERRAIN", 10: "VICTIM", 11: "VICTIM", 12: "VICTIM"},
        adjusted_start=(0.0, 0.0, 1.5),
        search_centre=(3.5, -1.0),
    )
    blob = pickle.dumps(world)
    back = pickle.loads(blob)
    assert back.victim_uids == [10, 11, 12]
    assert back.victim_centre == (1.5, 0.0, 0.9)
    assert back.victim_centre_xy == (1.5, 0.0)
    assert back.safety_patch.radius == 2.5
    assert back.support_category == "SUPPORT_TERRAIN"
    assert back.search_centre == (3.5, -1.0)
    blob2 = world.to_bytes()
    back2 = SARWorld.from_bytes(blob2)
    assert back2.body_tags == world.body_tags
