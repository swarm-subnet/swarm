from __future__ import annotations

from swarm.core import city_generator as cg


class _FixedRng:
    def __init__(self, value):
        self.value = value

    def next_float(self):
        return self.value


def test_seeded_rng_is_deterministic():
    a = cg.SeededRNG(123)
    b = cg.SeededRNG(123)
    seq_a = [a.rand_int(1, 100), round(a.range(0, 1), 6), round(a.next_float(), 6)]
    seq_b = [b.rand_int(1, 100), round(b.range(0, 1), 6), round(b.next_float(), 6)]
    assert seq_a == seq_b


def test_generate_road_positions_cover_map_bounds():
    rng = cg.SeededRNG(1)
    positions = cg.generate_road_positions(rng, min_spacing=15, target_area=400, tile_size=10)
    assert positions[0] == 0
    assert positions[-1] == 190
    assert all(positions[i] < positions[i + 1] for i in range(len(positions) - 1))


def test_extract_blocks_merges_cells_when_segment_removed():
    v = [0, 5, 10]
    h = [0, 5, 10]
    blocks_plain = cg.extract_blocks(v, h, min_area=1, effective_tile_size=1)
    blocks_merged = cg.extract_blocks(
        v,
        h,
        min_area=1,
        removed_v_segments=[(5, 0, 5)],
        effective_tile_size=1,
    )
    assert len(blocks_plain) >= 1
    assert len(blocks_merged) <= len(blocks_plain)


def test_ceil_half_rounds_up_to_nearest_half():
    assert cg.ceil_half(1.01) == 1.5
    assert cg.ceil_half(2.5) == 2.5


def test_get_building_zone_returns_center_middle_outer():
    assert cg.get_building_zone(100, 100, map_size=200) == "center"
    assert cg.get_building_zone(150, 100, map_size=200) == "middle"
    assert cg.get_building_zone(195, 195, map_size=200) == "outer"


def test_get_zone_building_type_city_type_one_always_house():
    assert cg.get_zone_building_type("outer", city_type=1, rng=_FixedRng(0.99)) == "house"


def test_get_zone_building_type_respects_thresholds():
    assert cg.get_zone_building_type("outer", city_type=2, rng=_FixedRng(0.79)) == "house"
    assert cg.get_zone_building_type("outer", city_type=2, rng=_FixedRng(0.90)) == "apt"
    assert cg.get_zone_building_type("outer", city_type=2, rng=_FixedRng(0.99)) == "tower"


def test_safe_zone_intersection_helpers():
    safe_zones = [(0.0, 0.0)]
    assert cg._in_safe_zone(0.2, 0.2, safe_zones, 1.0) is True
    assert cg._in_safe_zone(2.0, 2.0, safe_zones, 1.0) is False

    assert cg._rect_intersects_safe_zone(0.5, 0.0, 0.4, 0.4, safe_zones, 1.0) is True
    assert cg._rect_intersects_safe_zone(3.0, 3.0, 0.4, 0.4, safe_zones, 1.0) is False


def test_pick_city_variant_maps_hard_mode_to_city_type_3_difficulty_3():
    assert cg._pick_city_variant(_FixedRng(0.95)) == (3, 3)
