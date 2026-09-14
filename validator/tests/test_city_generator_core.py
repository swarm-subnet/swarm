# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Core city generator pieces: the seeded RNG, road layout, block merging and the zone building mix."""
from __future__ import annotations

from swarm.core import city_generator as cg


class _FixedRng:
    """RNG stand-in whose next_float always answers the same value, to pin a threshold draw."""
    def __init__(self, value):
        """Hold the value every draw will hand back."""
        self.value = value

    def next_float(self):
        """Return the held value, ignoring any seeded stream."""
        return self.value


def test_seeded_rng_is_deterministic():
    """Two generators on seed 123 emit an identical sequence across all three draw methods."""
    a = cg.SeededRNG(123)
    b = cg.SeededRNG(123)
    seq_a = [a.rand_int(1, 100), round(a.range(0, 1), 6), round(a.next_float(), 6)]
    seq_b = [b.rand_int(1, 100), round(b.range(0, 1), 6), round(b.next_float(), 6)]
    assert seq_a == seq_b


def test_generate_road_positions_cover_map_bounds():
    """Road lines run from 0 to one tile short of the 200 m map and never repeat or go backwards."""
    rng = cg.SeededRNG(1)
    positions = cg.generate_road_positions(rng, min_spacing=15, target_area=400, tile_size=10)
    assert positions[0] == 0
    assert positions[-1] == 190
    assert all(positions[i] < positions[i + 1] for i in range(len(positions) - 1))


def test_extract_blocks_merges_cells_when_segment_removed():
    """Deleting a road segment fuses the cells on either side, so the plot count can only fall."""
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
    """A value just over 1.0 lands on 1.5 while an exact 2.5 is left where it is."""
    assert cg.ceil_half(1.01) == 1.5
    assert cg.ceil_half(2.5) == 2.5


def test_get_building_zone_returns_center_middle_outer():
    """The ring a point belongs to follows its distance from the map centre outwards."""
    assert cg.get_building_zone(100, 100, map_size=200) == "center"
    assert cg.get_building_zone(150, 100, map_size=200) == "middle"
    assert cg.get_building_zone(195, 195, map_size=200) == "outer"


def test_get_zone_building_type_city_type_one_always_house():
    """A city of type 1 is houses throughout, whatever ring the plot sits in and however the draw falls."""
    assert cg.get_zone_building_type("outer", city_type=1, rng=_FixedRng(0.99)) == "house"


def test_get_zone_building_type_respects_thresholds():
    """In the outer ring of a type 2 city the draw splits at 0.80 and 0.95 into house, apt and tower."""
    assert cg.get_zone_building_type("outer", city_type=2, rng=_FixedRng(0.79)) == "house"
    assert cg.get_zone_building_type("outer", city_type=2, rng=_FixedRng(0.90)) == "apt"
    assert cg.get_zone_building_type("outer", city_type=2, rng=_FixedRng(0.99)) == "tower"


def test_safe_zone_intersection_helpers():
    """A point and a rectangle each count as protected only while they come within the guard radius."""
    safe_zones = [(0.0, 0.0)]
    assert cg._in_safe_zone(0.2, 0.2, safe_zones, 1.0) is True
    assert cg._in_safe_zone(2.0, 2.0, safe_zones, 1.0) is False

    assert cg._rect_intersects_safe_zone(0.5, 0.0, 0.4, 0.4, safe_zones, 1.0) is True
    assert cg._rect_intersects_safe_zone(3.0, 3.0, 0.4, 0.4, safe_zones, 1.0) is False


def test_pick_city_variant_maps_hard_mode_to_city_type_3_difficulty_3():
    """The last bucket of the variant distribution draws the urban layout at the raised difficulty."""
    assert cg._pick_city_variant(_FixedRng(0.95)) == (3, 3)
