"""B.2.2 — every body created by each map generator must be tagged after
the build. We use post-build classification via `build_and_tag_map`.

Per-map tests assert: every uid in the scene appears in `tagger.body_tags`.
The grep-style test runs all six maps and asserts the same invariant at
runtime, which is the operational expression of "no untagged bodies in
the SAR-active path"."""
from __future__ import annotations

import pybullet as p
import pytest

from swarm.core.env_builder.sar_tagging import build_and_tag_map, enumerate_bodies
from swarm.core.env_builder.sar_world import build_sar_world


_CHALLENGE_TYPES = {
    "city": 1,
    "open": 2,
    "mountain": 3,
    "village": 4,
    "warehouse": 5,
    "forest": 6,
}


def _assert_all_tagged(cli, tagger):
    scene = set(enumerate_bodies(cli))
    tagged = set(tagger.body_tags.keys())
    missing = scene - tagged
    extra = tagged - scene
    assert not missing, f"untagged bodies in scene: {sorted(missing)}"
    assert not extra, f"tagger has uids no longer in scene: {sorted(extra)}"
    return scene, tagged


@pytest.mark.parametrize("name,ctype", list(_CHALLENGE_TYPES.items()))
def test_per_map_all_bodies_tagged(sar_pybullet, name, ctype):
    start = (0.0, 0.0, 1.5)
    goal = (10.0, 10.0, 1.5)
    tagger = build_and_tag_map(
        sar_pybullet, seed=1000 + ctype, challenge_type=ctype,
        start=start, goal=goal,
    )
    scene, tagged = _assert_all_tagged(sar_pybullet, tagger)
    # At minimum some support_* tag must exist or the resolver cannot work.
    support_tags = {
        v for v in tagger.body_tags.values() if v.startswith("SUPPORT_")
    }
    assert support_tags, f"{name}: no SUPPORT_* body found"


@pytest.mark.parametrize("name,ctype", list(_CHALLENGE_TYPES.items()))
def test_no_untagged_bodies_anywhere(sar_pybullet, name, ctype):
    """Runtime equivalent of the grep check: every body in a SAR-active map
    build must be in body_tags. Split per map so the builds parallelize."""
    tagger = build_and_tag_map(
        sar_pybullet, seed=50 + ctype, challenge_type=ctype,
        start=(0.0, 0.0, 1.5), goal=(10.0, 10.0, 1.5),
    )
    scene = set(enumerate_bodies(sar_pybullet))
    tagged = set(tagger.body_tags.keys())
    missing = scene - tagged
    assert not missing, f"{name}: untagged bodies {sorted(missing)}"


# Seeds 1014 and 1030 previously seated the victim ~17-21 m up on a tree
# canopy, because a wide tree mesh tripped the >5x5 SUPPORT_TERRAIN heuristic.
@pytest.mark.parametrize("seed", [1014, 1030, 1007])
def test_forest_trees_never_classified_as_support(sar_pybullet, seed):
    p.resetSimulation(physicsClientId=sar_pybullet)
    world = build_sar_world(
        sar_pybullet, seed=seed, challenge_type=6,
        start=(0.0, 0.0, 1.5), goal=(8.0, 8.0, 1.5),
    )

    canopy = [u for u, t in world.body_tags.items() if t == "OBSTACLE_CANOPY"]
    assert canopy, f"seed {seed}: trees were not tagged OBSTACLE_CANOPY"

    # The only standable bodies in a forest are the ground and hills, which
    # span the map; a tree-sized SUPPORT footprint means a canopy became ground.
    for uid, tag in world.body_tags.items():
        if not tag.startswith("SUPPORT_"):
            continue
        mn, mx = p.getAABB(uid, physicsClientId=sar_pybullet)
        fx, fy = mx[0] - mn[0], mx[1] - mn[1]
        assert fx >= 15.0 or fy >= 15.0, (
            f"seed {seed}: SUPPORT body uid={uid} has a tree-sized footprint "
            f"{fx:.1f}x{fy:.1f} m — a tree was mis-tagged as standable ground"
        )

    assert world.support_category.startswith("SUPPORT_")
