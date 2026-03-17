from __future__ import annotations

from swarm.core.env_builder import generation as generation_mod
from swarm.core.forest_generator_parts import placement as placement_mod


def test_build_static_world_passes_safe_zones_to_forest(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _capture_build_forest_map(cli, seed, safe_zones, safe_zone_radius):
        captured["cli"] = cli
        captured["seed"] = seed
        captured["safe_zones"] = safe_zones
        captured["safe_zone_radius"] = safe_zone_radius

    monkeypatch.setattr(generation_mod.shared, "build_forest_map", _capture_build_forest_map)

    start = (1.0, 2.0, 3.0)
    goal = (4.0, 5.0, 6.0)
    generation_mod._build_static_world(
        seed=123,
        cli=7,
        start=start,
        goal=goal,
        challenge_type=6,
    )

    assert captured["cli"] == 7
    assert captured["seed"] == 123
    assert captured["safe_zones"] == [start, goal]
    assert captured["safe_zone_radius"] == (
        generation_mod.shared.SAFE_ZONE_RADIUS
        + max(
            generation_mod.shared.START_PLATFORM_RADIUS,
            generation_mod.shared.LANDING_PLATFORM_RADIUS,
        )
    )


def test_pick_tree_instances_avoids_safe_zone(monkeypatch) -> None:
    monkeypatch.setattr(
        placement_mod,
        "_tree_candidate_points",
        lambda _rng, _count: [(0.0, 0.0), (10.0, 10.0)],
    )
    monkeypatch.setattr(
        placement_mod,
        "_build_tree_family_assets",
        lambda _assets: {
            placement_mod.COMMON_TREE_PREFIX: {
                "total_weight": 1.0,
                "weighted_assets": [(1.0, "normal", "CommonTree_1.obj")],
            }
        },
    )
    monkeypatch.setattr(
        placement_mod,
        "_rank_tree_families_for_point",
        lambda *_args, **_kwargs: [placement_mod.COMMON_TREE_PREFIX],
    )
    monkeypatch.setattr(
        placement_mod,
        "_pick_weighted_tree_from_family",
        lambda _rng, _info: ("normal", "CommonTree_1.obj"),
    )
    monkeypatch.setattr(placement_mod.os.path, "exists", lambda _path: True)
    monkeypatch.setattr(placement_mod, "_obj_planar_radius_cached", lambda _path: 1.0)
    monkeypatch.setattr(
        placement_mod,
        "_tree_spacing_radius",
        lambda _obj_name, canopy_radius, _difficulty_id: canopy_radius,
    )
    monkeypatch.setattr(
        placement_mod,
        "_tree_occupancy_radius",
        lambda _obj_name, canopy_radius: canopy_radius,
    )
    monkeypatch.setattr(placement_mod, "_tree_dual_rects_for_scale", lambda *_args, **_kwargs: None)

    safe_zone_circles = [(0.0, 0.0, 2.0)]
    placed = placement_mod._pick_tree_instances(
        placement_mod.random.Random(123),
        count=1,
        assets=[("normal", "CommonTree_1.obj")],
        clearance_m=0.0,
        difficulty_id=1,
        safe_zone_circles=safe_zone_circles,
        safe_zone_rects=placement_mod._safe_zone_rects(safe_zone_circles),
    )

    assert len(placed) == 1
    assert placed[0][0:2] == (10.0, 10.0)
