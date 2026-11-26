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

"""An area's placement tables, from a small manifest and meshes written in the test: every vertex of a tree lies in
its crown circle, a house's pieces make one rectangle round it, and bridges, lift ropes and water never become houses
but mark the cells a field keeps clear of."""
from __future__ import annotations

import math

import numpy as np
import pytest

from swarm.core.maps.andorra import placements

SHAPE = (100, 100)


def _write_obj(path, vertices, faces):
    """Write an OBJ file with texture indices on its faces, as the converter writes them."""
    lines = [f"v {x:.3f} {y:.3f} {z:.3f}" for x, y, z in vertices]
    lines += ["f " + " ".join(f"{i + 1}/{i + 1}" for i in face) for face in faces]
    path.write_text("\n".join(lines) + "\n")


def _box(x0, y0, x1, y1, z0=0.0, z1=3.0):
    """The vertices and faces of an upright box, as a house piece or a bridge deck."""
    vertices = [(x, y, z) for z in (z0, z1) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    faces = [(0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6),
             (3, 0, 4), (3, 4, 7)]
    return vertices, faces


@pytest.fixture()
def area(tmp_path):
    """A small area folder: a tree whose widest leaf is off its box's axes, a two-piece house turned 30 degrees, a
    bridge, a long thin rope and a pond."""
    (tmp_path / "shared" / "trees").mkdir(parents=True)
    (tmp_path / "buildings").mkdir()
    (tmp_path / "water").mkdir()
    tree = [(0.0, 0.0, 0.0), (0.1, 0.0, 5.0), (1.2, 1.6, 6.0), (-1.0, 0.5, 4.0), (0.5, -1.2, 7.0)]
    _write_obj(tmp_path / "shared" / "trees" / "tree.obj", tree, [(0, 1, 2), (0, 3, 4)])
    turn = math.radians(30.0)
    along, across = np.array([math.cos(turn), math.sin(turn)]), np.array([-math.sin(turn), math.cos(turn)])
    centre = np.array([-20.0, 20.0])

    def turned(u0, v0, u1, v1, z1):
        """A box piece turned 30 degrees about the house's centre, baked in place as the converter does."""
        vertices, faces = _box(u0, v0, u1, v1, 0.0, z1)
        return [(*(centre + along * x + across * y), z) for x, y, z in vertices], faces

    _write_obj(tmp_path / "buildings" / "buildings_vf1_7_walls.obj", *turned(-6.0, -4.0, 6.0, 4.0, 5.0))
    _write_obj(tmp_path / "buildings" / "buildings_vf1_7_roof.obj", *turned(-6.5, -4.5, 6.5, 4.5, 7.0))
    _write_obj(tmp_path / "buildings" / "buildings_bridge_wood.obj", *_box(10.0, -30.0, 14.0, 10.0, 0.0, 1.0))
    _write_obj(tmp_path / "buildings" / "buildings_Lift_rope.obj", [(-45.0, -40.0, 12.0), (45.0, 40.0, 14.0), (45.0, 40.1, 14.0)],
               [(0, 1, 2)])
    _write_obj(tmp_path / "water" / "water_pond.obj", [(-5.0, -5.0, 0.0), (5.0, -5.0, 0.0), (5.0, 5.0, 0.0), (-5.0, 5.0, 0.0)],
               [(0, 1, 2, 3)])

    def item(name, group, folder, shared=False):
        """A manifest item."""
        row = {"group": group, "folder": folder, "obj": name + ".obj", "bounds_min": [-1, -1, 0], "bounds_max": [1, 1, 1]}
        if shared:
            row["root"] = "shared"
        return row

    manifest = {"andorra": {"shared_root": "shared"},
                "items": {"trees_t": item("tree", "trees", "trees", shared=True),
                          "buildings_vf1_7_walls": item("buildings_vf1_7_walls", "buildings", "buildings"),
                          "buildings_vf1_7_roof": item("buildings_vf1_7_roof", "buildings", "buildings"),
                          "buildings_bridge_wood": item("buildings_bridge_wood", "buildings", "buildings"),
                          "buildings_Lift_rope": item("buildings_Lift_rope", "buildings", "buildings"),
                          "water_pond": item("water_pond", "water", "water")},
                "placements": []}
    identity = [0.0, 0.0, 0.0, 1.0]
    yaw = [0.0, 0.0, math.sin(0.6), math.cos(0.6)]
    for name, position, quaternion, scale in (("trees_t", [30.0, -30.0, 2.0], yaw, [1.5, 1.5, 1.2]),
                                               ("buildings_vf1_7_walls", [0.0, 0.0, 0.0], identity, [1, 1, 1]),
                                               ("buildings_vf1_7_roof", [0.0, 0.0, 0.0], identity, [1, 1, 1]),
                                               ("buildings_bridge_wood", [0.0, 0.0, 0.0], identity, [1, 1, 1]),
                                               ("buildings_Lift_rope", [0.0, 0.0, 0.0], identity, [1, 1, 1]),
                                               ("water_pond", [25.0, 25.0, 0.0], identity, [1, 1, 1])):
        manifest["placements"].append({"item": name, "position": position, "quaternion": quaternion, "scale": scale})
    placements.read_mesh.cache_clear()
    placements.item_reach_m.cache_clear()
    return tmp_path, manifest


def _cell(x, y):
    """The grid cell holding a point in area metres."""
    return int(math.floor(SHAPE[0] / 2 - y)), int(math.floor(x + SHAPE[1] / 2))


def test_every_vertex_of_a_tree_lies_inside_its_crown_circle(area):
    """The circle reaches the farthest vertex, not the box, times the tree's scale, plus the drone's margin, and
    every vertex of the turned and scaled tree lies inside it."""
    root, manifest = area
    crowns = placements.crown_circles(root, manifest)
    assert crowns.shape == (1, 3)
    x, y, radius = crowns[0]
    assert (x, y) == (30.0, -30.0)
    assert radius == pytest.approx(2.0 * 1.5 + placements.CROWN_MARGIN_M)
    place = manifest["placements"][0]
    vertices = placements.placed(placements.read_mesh(str(root / "shared" / "trees" / "tree.obj")).vertices, place)
    assert (np.hypot(vertices[:, 0] - x, vertices[:, 1] - y) <= radius - placements.CROWN_MARGIN_M + 1e-9).all()


def test_house_pieces_join_into_one_turned_rectangle_round_them(area):
    """The walls and the roof of one house make a single row: the roof's 13 by 9 m turned to a heading of 60
    degrees, centred where the house stands; the bridge and the rope make none."""
    root, manifest = area
    rows, corners = placements.houses(root, manifest)
    assert rows.shape == (1, 5) and corners.shape == (1, 4, 2)
    x, y, length, width, heading = rows[0]
    assert (x, y) == pytest.approx((-20.0, 20.0), abs=1e-3)
    assert (length, width) == pytest.approx((13.0, 9.0), abs=1e-3)
    assert heading == pytest.approx(60.0, abs=1e-3)


def test_bridges_ropes_houses_and_water_mark_the_cells_a_field_keeps_clear_of(area):
    """The house, the bridge deck, every cell along the thin rope and the pond are marked; open ground far from
    all of them is not."""
    root, manifest = area
    grid = placements.keep_out(root, manifest, SHAPE)
    for point in ((-20.0, 20.0), (12.0, -10.0), (25.0, 25.0), (-40.0, -35.56), (0.0, 0.0), (40.0, 35.56)):
        assert grid[_cell(*point)], point
    assert not grid[_cell(-35.0, -10.0)]
    assert not grid[_cell(35.0, -20.0)]


def test_the_tables_come_out_the_same_on_every_read(area):
    """Reading the same files twice, with the mesh cache emptied between, gives identical bytes."""
    root, manifest = area
    first = placements.tables(root, manifest, SHAPE)
    placements.read_mesh.cache_clear()
    placements.item_reach_m.cache_clear()
    second = placements.tables(root, manifest, SHAPE)
    for a, b in zip(first, second):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


def test_stones_and_bushes_mark_their_tops_and_nothing_else_does():
    """A solid stone marks the cells its turned, scaled box reaches, grown by one, with its top in whole centimetres
    above the survey's base; a pebble that never collides, a tree and a house mark nothing."""
    manifest = {"items": {"stone": {"group": "rocks", "collision": "mesh", "bounds_min": [-1.0, -0.5, -0.2], "bounds_max": [1.0, 0.5, 1.0]},
                          "pebble": {"group": "rocks", "collision": "mesh", "bounds_min": [-0.1, -0.1, 0.0], "bounds_max": [0.1, 0.1, 0.1]},
                          "tree": {"group": "trees", "collision": "cylinder", "bounds_min": [-2, -2, 0], "bounds_max": [2, 2, 9]},
                          "house": {"group": "buildings", "collision": "mesh", "bounds_min": [-3, -3, 0], "bounds_max": [3, 3, 6]}},
                "placements": [
                    {"item": "stone", "position": [10.2, -5.3, 4.0], "quaternion": [0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4)],
                     "scale": [2.0, 2.0, 1.5]},
                    {"item": "pebble", "position": [-20.5, 0.5, 3.0], "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1], "collision": "none"},
                    {"item": "tree", "position": [30.5, 30.5, 2.0], "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]},
                    {"item": "house", "position": [-30.5, -30.5, 2.0], "quaternion": [0, 0, 0, 1], "scale": [1, 1, 1]}]}
    grid = placements.solid_tops(manifest, SHAPE, z_shift=-2.0)
    # Turned a quarter, the stone reaches 1 m east-west and 2 m north-south; its top is 4 + 1.5 m, 7.5 m over the base.
    marked = np.argwhere(grid > placements.NO_TOP)
    assert set(np.unique(grid[grid > placements.NO_TOP])) == {750}
    assert (marked[:, 0].min(), marked[:, 0].max()) == (_cell(10.2, -3.3)[0] - 1, _cell(10.2, -7.3)[0] + 1)
    assert (marked[:, 1].min(), marked[:, 1].max()) == (_cell(9.2, -5.3)[1] - 1, _cell(11.2, -5.3)[1] + 1)
    for point in ((-20.5, 0.5), (30.5, 30.5), (-30.5, -30.5)):
        assert grid[_cell(*point)] == placements.NO_TOP
