"""B.2.2 — every body created by each map generator must be tagged after
the build. We use post-build classification via `build_and_tag_map`.

Per-map tests assert: every uid in the scene appears in `tagger.body_tags`.
The grep-style test runs all six maps and asserts the same invariant at
runtime, which is the operational expression of "no untagged bodies in
the SAR-active path"."""
from __future__ import annotations

import pytest

from swarm.core.env_builder.sar_tagging import build_and_tag_map, enumerate_bodies


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


def test_no_untagged_bodies_anywhere(sar_pybullet):
    """Runtime equivalent of the grep check: every body in every SAR-active
    map build must be in body_tags."""
    for name, ctype in _CHALLENGE_TYPES.items():
        cli = sar_pybullet
        # Reset between maps to avoid uid accumulation across builds
        import pybullet as p
        p.resetSimulation(physicsClientId=cli)
        tagger = build_and_tag_map(
            cli, seed=50 + ctype, challenge_type=ctype,
            start=(0.0, 0.0, 1.5), goal=(10.0, 10.0, 1.5),
        )
        scene = set(enumerate_bodies(cli))
        tagged = set(tagger.body_tags.keys())
        missing = scene - tagged
        assert not missing, f"{name}: untagged bodies {sorted(missing)}"
