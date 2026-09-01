from __future__ import annotations

import pytest

from swarm.core.warehouse import office


def test_slot_config_for_each_wall_and_invalid():
    assert office.slot_config("north")["wall_yaw"] == 0.0
    assert office.slot_config("east")["normal"] == (-1.0, 0.0)
    with pytest.raises(ValueError):
        office.slot_config("invalid")


def test_snap_helpers():
    assert office.snap_cardinal(46) == 90.0
    assert office.snap_cardinal(359) == 0.0
    assert office.snap_octant(23) == 45.0
    assert office.snap_octant(359) == 0.0


def test_wall_face_and_tangent_yaws():
    assert office.wall_face_yaw("north") == 180.0
    assert office.wall_face_yaw("west") == 270.0
    assert office.wall_tangent_yaw("east") == 90.0


def test_desk_lr_along_offsets_depend_on_slot():
    assert office.desk_lr_along_offsets("north", 2.0) == (-2.0, 2.0)
    assert office.desk_lr_along_offsets("south", 2.0) == (2.0, -2.0)


def test_slot_xy_places_points_on_expected_edges():
    edge = office.FLOOR_SIZE[0] / 2.0
    x_n, y_n = office.slot_xy("north", along=0.0, inward=0.0)
    x_s, y_s = office.slot_xy("south", along=0.0, inward=0.0)
    x_e, y_e = office.slot_xy("east", along=0.0, inward=0.0)
    x_w, y_w = office.slot_xy("west", along=0.0, inward=0.0)

    assert y_n == office.ROOM_CENTER[1] + edge
    assert y_s == office.ROOM_CENTER[1] - edge
    assert x_e == office.ROOM_CENTER[0] + edge
    assert x_w == office.ROOM_CENTER[0] - edge


def test_corner_points_and_nearest_corner_index():
    corners = office.corner_points()
    assert len(corners) == 4
    idx = office.nearest_corner_index(corners[0][0], corners[0][1], corners)
    assert idx == 0


def test_workstation_corner_helpers():
    assert office.workstation_l_corner_index("north") == 1
    assert office.adjacent_slot_for_l("west") == "north"
    assert office.along_sign_for_corner("north", 0) == -1.0
    assert office.workstation_right_is_positive_along("west") is True
    assert office.workstation_right_is_positive_along("east") is False
