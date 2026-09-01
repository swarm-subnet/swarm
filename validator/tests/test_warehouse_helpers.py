from __future__ import annotations

import math
import random
from pathlib import Path

import pytest

from swarm.core.warehouse import helpers as wh


class _DummyLoader:
    def __init__(self, obj_dir: str = ""):
        self.obj_dir = obj_dir

    def model_size(self, model_name, scale):
        _ = model_name, scale
        return (2.0, 4.0, 1.0)

    def _parse_vertices(self, model_name):
        _ = model_name
        return [(-1.0, -2.0, -3.0), (1.0, 2.0, 3.0)]


def test_slot_point_and_dock_yaw_for_all_slots():
    assert wh.slot_point("north", along=1.0, inward=2.0) == (1.0, wh.HALF_Y - 2.0)
    assert wh.slot_point("south", along=1.0, inward=2.0) == (1.0, -wh.HALF_Y + 2.0)
    assert wh.slot_point("east", along=1.0, inward=2.0) == (wh.HALF_X - 2.0, 1.0)
    assert wh.slot_point("west", along=1.0, inward=2.0) == (-wh.HALF_X + 2.0, 1.0)
    assert wh.dock_inward_yaw_for_slot("north") == 180.0
    assert wh.wall_yaw_for_slot("west") == 270.0

    with pytest.raises(ValueError):
        wh.slot_point("invalid", along=0.0, inward=0.0)


def test_tiled_centers_validates_tile_size_and_positions():
    centers = wh.tiled_centers(total_size=10.0, tile_size=2.0)
    assert centers == [-4.0, -2.0, 0.0, 2.0, 4.0]

    with pytest.raises(ValueError):
        wh.tiled_centers(total_size=10.0, tile_size=0.0)


def test_oriented_xy_size_rotates_dimensions_and_uses_cache():
    loader = _DummyLoader()
    xy_90 = wh.oriented_xy_size(loader, "model.obj", scale=1.0, yaw_deg=90.0)
    xy_90_again = wh.oriented_xy_size(loader, "model.obj", scale=1.0, yaw_deg=90.0)
    assert xy_90 == pytest.approx((4.0, 2.0), rel=1e-12, abs=1e-12)
    assert xy_90_again == xy_90


def test_model_bounds_xyz_from_vertices():
    loader = _DummyLoader()
    min_v, max_v = wh.model_bounds_xyz(loader, "x.obj", (1.0, 2.0, 3.0))
    assert min_v == [-1.0, -4.0, -9.0]
    assert max_v == [1.0, 4.0, 9.0]


def test_resolve_optional_model_case_insensitive(tmp_path):
    root = tmp_path / "assets"
    root.mkdir()
    (root / "Truck.OBJ").write_text("v 0 0 0\n")
    found_dir, found_name = wh._resolve_optional_model(str(root), ["truck.obj"])
    assert Path(found_dir) == root
    assert found_name == "Truck.OBJ"


def test_first_existing_model_name(tmp_path):
    loader = _DummyLoader(obj_dir=str(tmp_path))
    (tmp_path / "b.obj").write_text("v 0 0 0\n")
    assert wh._first_existing_model_name(loader, ["a.obj", "b.obj"]) == "b.obj"


def test_shell_mesh_scale_xy_uses_config_override():
    sx, sy = wh._shell_mesh_scale_xy({"config": {"warehouse_size_x": 52.0, "warehouse_size_y": 36.0}})
    assert math.isclose(sx, wh.WAREHOUSE_SIZE_X / 52.0, rel_tol=1e-9)
    assert math.isclose(sy, wh.WAREHOUSE_SIZE_Y / 36.0, rel_tol=1e-9)


def test_truck_extra_gap_by_gate_name():
    assert wh._truck_extra_gap_for_gate_state("dock-door-closed.obj") == wh.LOADING_TRUCK_EXTRA_GAP_CLOSED
    assert wh._truck_extra_gap_for_gate_state("dock-door-half.obj") == wh.LOADING_TRUCK_EXTRA_GAP_HALF
    assert wh._truck_extra_gap_for_gate_state("dock-door-open.obj") == 0.0


def test_rect_helpers_and_overlap():
    a = {"cx": 0.0, "cy": 0.0, "sx": 2.0, "sy": 2.0}
    b = {"cx": 1.0, "cy": 0.0, "sx": 2.0, "sy": 2.0}
    c = {"cx": 10.0, "cy": 10.0, "sx": 1.0, "sy": 1.0}
    assert wh._candidate_rect_bounds(a) == (-1.0, 1.0, -1.0, 1.0)
    assert wh._rects_overlap(a, b, gap=0.0) is True
    assert wh._rects_overlap(a, c, gap=0.0) is False


def test_wall_and_span_geometry_helpers():
    assert wh._size_fits_half_span(2.0, 3.0, half_x=5.0, half_y=5.0, margin=0.5) is True
    assert wh._size_fits_half_span(20.0, 3.0, half_x=5.0, half_y=5.0, margin=0.5) is False

    lo, hi = wh._wall_along_limits("north", sx=2.0, sy=3.0, half_x=10.0, half_y=8.0, margin=1.0)
    assert lo < hi
    x, y = wh._wall_attached_center("south", along=0.0, sx=2.0, sy=3.0, half_x=10.0, half_y=8.0, margin=1.0)
    assert y < 0

    assert wh._orient_dims_long_side_on_wall("north", 2.0, 5.0) == (5.0, 2.0)
    assert wh._orient_dims_long_side_on_wall("east", 2.0, 5.0) == (2.0, 5.0)


def test_attached_wall_from_area_bounds_prefers_nearest_wall():
    half_y = wh.WAREHOUSE_SIZE_Y * 0.5
    wall = wh._attached_wall_from_area_bounds(area_sx=5.0, area_sy=10.0, area_cx=0.0, area_cy=half_y - 5.0)
    assert wall == "north"


def test_window_index_helpers():
    assert wh.mirrored_window_indices(5) == set()
    assert wh.mirrored_window_indices(12) == {2, 6, 9}

    starts = wh.mirrored_wide_window_starts(segment_count=12, span_steps=2, seed_key=10)
    assert isinstance(starts, list)
    assert starts == sorted(starts)


def test_door_blocking_and_span_math():
    blocked = wh._indices_blocked_by_doors(along_values=[-2, -1, 0, 1, 2], door_centers=[0], door_span=1.0)
    assert blocked == {2}

    merged = wh._merge_spans_1d([(0, 1), (0.9, 2.0), (3.0, 4.0)])
    assert merged == [(0.0, 2.0), (3.0, 4.0)]

    remaining = wh._subtract_spans_1d(base_spans=[(0, 10)], cut_spans=[(2, 3), (5, 7)])
    assert remaining == [(0.0, 2.0), (3.0, 5.0), (7.0, 10.0)]


def test_mirrored_window_filters():
    singles = wh._filter_mirrored_single_windows({2, 3, 9}, blocked_indices={9}, segment_count=12)
    assert singles == {3, 8}

    wides = wh._filter_mirrored_wide_windows(candidate_starts=[2], span_steps=2, blocked_indices=set(), segment_count=12)
    assert wides == [2, 8]

    wides_blocked = wh._filter_mirrored_wide_windows(candidate_starts=[2], span_steps=2, blocked_indices={8, 9}, segment_count=12)
    assert wides_blocked == []


def test_sample_random_center_within_safe_bounds():
    rng = random.Random(3)
    x, y = wh._sample_random_center(rng, sx=2.0, sy=2.0, floor_half_x=10.0, floor_half_y=8.0, margin=1.0)
    assert -8.0 <= x <= 8.0
    assert -6.0 <= y <= 6.0
