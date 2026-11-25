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

"""A small stand-in area package for the tests, laid out like the real one: a catalogue, a shared folder and one
160 m area on a gentle slope with trees, rocks, a pebble, two houses, a pond, the horizon, grass tufts and far trees.

The real areas are not in the released swarm-worlds package the tests run on, so the area tests write this one into a
folder instead and load it through the real loader. Every piece stands where its constant below says, so a test can
aim a ray at it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from swarm.core.maps.andorra.rendering import LIGHT_GRADE
from swarm.core.maps.andorra.survey import PERSON_KEY

SIDE_M = 160                           # the area square, metres
TILE_M = 32
SLOPE = 0.05                           # the ground rises this much per metre east
LIFT_M = 10.0                          # world ground height at the area's centre line
ORIGIN_EN = [530000.0, 20000.0]
FRAME = {"origin_en": ORIGIN_EN, "base_height_m": 1000.0, "z_offset_m": -2.0}
SURVEY_BASE_M = 1001.5                 # a survey base apart from the frame's, so the shift between them is tested
SLUG = "01_stand_in"
TREES = [(-60.0, 60.0), (10.0, 10.0), (20.0, -5.0), (65.0, -65.0)]
TREE_HALF_M, TREE_HEIGHT_M = 2.0, 8.0
STONES = [(5.0, -20.0), (70.0, 70.0)]  # solid
PEBBLE = (8.0, -20.0)                  # marked "collision": "none"
HOUSES = [(30.0, 30.0), (-70.0, -10.0)]
HOUSE_HALF_M, HOUSE_HEIGHT_M = (4.0, 3.0), 5.0
POND = (-30.0, -40.0)
TUFTS = [(0.0, 0.0), (15.0, 5.0), (-70.0, 70.0), (75.0, -70.0)]
FAR_TREES = [(0.0, 900.0), (900.0, 0.0)]
NO_DATA_CELL = (0, 0)                  # the survey has no data in its north-west corner cell
GROUND_LUMINANCE = 0.131               # the brightness the area's table gives its one ground picture


def ground_z(x: float, y: float = 0.0) -> float:
    """World height of the stand-in's ground at a point."""
    _ = y
    return LIFT_M + SLOPE * x


def _quad(half_x: float, half_y: float, slope: float = 0.0) -> str:
    """An .obj quad centred on the origin, rising `slope` per metre east."""
    corners = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
    lines = [f"v {x} {y} {slope * x:.4f}" for x, y in corners]
    return "\n".join(lines + ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1", "f 1/1 2/2 3/3", "f 1/1 3/3 4/4"]) + "\n"


def _box(half_x: float, half_y: float, height: float) -> str:
    """An .obj box standing on the origin."""
    corners = [(x, y, z) for z in (0.0, height) for x, y in ((-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y))]
    faces = ["1 3 2", "1 4 3", "5 6 7", "5 7 8", "1 2 6", "1 6 5", "2 3 7", "2 7 6", "3 4 8", "3 8 7", "4 1 5", "4 5 8"]
    return "\n".join([f"v {x} {y} {z}" for x, y, z in corners] + [f"f {face}" for face in faces]) + "\n"


def _pyramid(half: float, height: float) -> str:
    """An .obj four-sided pyramid standing on the origin: the stand-in's tree."""
    corners = [(-half, -half, 0.0), (half, -half, 0.0), (half, half, 0.0), (-half, half, 0.0), (0.0, 0.0, height)]
    faces = ["1 2 5", "2 3 5", "3 4 5", "4 1 5", "1 3 2", "1 4 3"]
    return "\n".join([f"v {x} {y} {z}" for x, y, z in corners] + [f"f {face}" for face in faces]) + "\n"


def _item(name: str, group: str, folder: str, obj: str, collision: str, low: list, high: list, flags: list = (),
          shared: bool = False) -> dict:
    """A manifest item record as the converter writes one."""
    item = {"name": name, "group": group, "folder": folder, "obj": obj, "texture": "paint.png", "flags": list(flags),
            "collision": collision, "bounds_min": low, "bounds_max": high, "specular": 0.0}
    if shared:
        item["root"] = "shared"
    return item


def _place(item: str, x: float, y: float, z: float, yaw_deg: float = 0.0, **extra) -> dict:
    """A placement record standing an item at a point, turned about z."""
    half = np.radians(yaw_deg) / 2.0
    return {"item": item, "position": [x, y, z], "quaternion": [0.0, 0.0, float(np.sin(half)), float(np.cos(half))],
            "scale": [1.0, 1.0, 1.0], **extra}


def _write(path: Path, text: str) -> None:
    """Write a text file, making its folder first."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _paint(folder: Path, rgb: tuple) -> None:
    """Write the 2 x 2 texture every item in a folder is painted with."""
    folder.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), rgb).save(folder / "paint.png")


def tile_centres() -> list:
    """Centres of the stand-in's terrain tiles, west to east, south to north."""
    edges = np.arange(-SIDE_M / 2 + TILE_M / 2, SIDE_M / 2, TILE_M)
    return [(float(x), float(y)) for y in edges for x in edges]


def _survey(area: Path) -> None:
    """Write the area's 1 m grids: the slope, the objects on it, and open ground, where a person may stand too, off the
    houses, the pond, the trees and the outer 10 m."""
    shape = (SIDE_M, SIDE_M)
    rows, cols = np.indices(shape)
    x, y = cols + 0.5 - SIDE_M / 2, SIDE_M / 2 - rows - 0.5
    shift = SURVEY_BASE_M - FRAME["base_height_m"] + FRAME["z_offset_m"]
    ground = np.rint((ground_z(x) - shift) * 100.0).astype(np.uint16)
    ceiling = ground.copy()
    blocked = (np.abs(x) > SIDE_M / 2 - 10) | (np.abs(y) > SIDE_M / 2 - 10)
    for n, (hx, hy) in enumerate(HOUSES):
        half_x, half_y = HOUSE_HALF_M[::-1] if n % 2 else HOUSE_HALF_M     # every other house stands turned 90 degrees
        house = (np.abs(x - hx) <= half_x) & (np.abs(y - hy) <= half_y)
        ceiling[house] += int(HOUSE_HEIGHT_M * 100)
        blocked |= house
    for tx, ty in TREES:
        crown = np.hypot(x - tx, y - ty) <= TREE_HALF_M
        ceiling[crown] += int(TREE_HEIGHT_M * 100)
        blocked |= crown
    blocked |= (np.abs(x - POND[0]) <= 5) & (np.abs(y - POND[1]) <= 5)
    ground[NO_DATA_CELL] = ceiling[NO_DATA_CELL] = 65535
    kind = np.where(blocked, 0, 11).astype(np.uint8)
    open_ground = ~blocked
    dock = open_ground & (np.abs(x) < SIDE_M / 2 - 20) & (np.abs(y) < SIDE_M / 2 - 20)
    metadata = {"version": 2, "id": 1, "place": "Stand-in slope", "cell_m": 1, "row_order": "north_to_south"}
    np.savez_compressed(area / "spots.npz", metadata=np.asarray(json.dumps(metadata)), shape=np.asarray(shape, np.uint16),
                        origin_en=np.asarray(ORIGIN_EN), base_height_m=np.asarray(SURVEY_BASE_M), kind=kind,
                        ground_cm=ground, ceiling_cm=ceiling, start_bits=np.packbits(open_ground, bitorder="little"),
                        dock_bits=np.packbits(dock, bitorder="little"),
                        **{PERSON_KEY: np.packbits(open_ground, bitorder="little")})


def write_area(root: Path) -> Path:
    """Write the stand-in package into a folder: its catalogue, shared folder and area; returns the area's folder."""
    root = Path(root)
    shared, area = root / "shared", root / SLUG
    _write(shared / "terrain" / "tile.obj", _quad(TILE_M / 2, TILE_M / 2, SLOPE))
    _paint(shared / "terrain", (95, 110, 60))
    _write(shared / "trees" / "tree.obj", _pyramid(TREE_HALF_M, TREE_HEIGHT_M))
    _paint(shared / "trees", (40, 70, 30))
    _write(area / "rocks" / "stone.obj", _box(0.5, 0.5, 0.6))
    _paint(area / "rocks", (120, 120, 115))
    _write(area / "buildings" / "house.obj", _box(*HOUSE_HALF_M, HOUSE_HEIGHT_M))
    _paint(area / "buildings", (170, 150, 130))
    _write(area / "water" / "pond.obj", _quad(5.0, 5.0))
    _paint(area / "water", (40, 60, 70))
    _write(area / "horizon" / "ridge.obj", _quad(2000.0, 50.0))
    _write(area / "horizon" / "far_tree.obj", _pyramid(3.0, 12.0))
    _paint(area / "horizon", (90, 100, 110))
    _write(area / "grass" / "tuft.obj", _pyramid(0.1, 0.3))
    _paint(area / "grass", (100, 130, 60))
    np.save(area / "grass" / "tufts.npy", np.array([[0, x, y, ground_z(x), 0.0, 1.0, 1.0] for x, y in TUFTS]))
    np.save(area / "horizon" / "far_trees.npy", np.array([[0, x, y, 50.0, 0.0, 1.0, 1.0] for x, y in FAR_TREES]))

    slope_z = SLOPE * TILE_M / 2
    items = {
        "tile": _item("tile", "terrain", "terrain", "tile.obj", "mesh", [-16, -16, -slope_z], [16, 16, slope_z], shared=True),
        "tree": _item("tree", "trees", "trees", "tree.obj", "cylinder", [-TREE_HALF_M, -TREE_HALF_M, 0], [TREE_HALF_M, TREE_HALF_M, TREE_HEIGHT_M],
                      ("instanced", "double_sided"), shared=True),
        "stone": _item("stone", "rocks", "rocks", "stone.obj", "mesh", [-0.5, -0.5, 0], [0.5, 0.5, 0.6], ("instanced",)),
        "house": _item("house", "buildings", "buildings", "house.obj", "mesh", [-HOUSE_HALF_M[0], -HOUSE_HALF_M[1], 0],
                       [HOUSE_HALF_M[0], HOUSE_HALF_M[1], HOUSE_HEIGHT_M]),
        "pond": _item("pond", "water", "water", "pond.obj", "none", [-5, -5, 0], [5, 5, 0]),
        "ridge": _item("ridge", "horizon", "horizon", "ridge.obj", "none", [-2000, -50, 0], [2000, 50, 0]),
        "tuft": _item("tuft", "grass", "grass", "tuft.obj", "none", [-0.1, -0.1, 0], [0.1, 0.1, 0.3], ("instanced", "double_sided")),
        "far_tree": _item("far_tree", "horizon", "horizon", "far_tree.obj", "none", [-3, -3, 0], [3, 3, 12], ("instanced",)),
    }
    placements = [_place("tile", x, y, ground_z(x)) for x, y in tile_centres()]
    placements += [_place("tree", x, y, ground_z(x)) for x, y in TREES]
    placements += [_place("stone", x, y, ground_z(x)) for x, y in STONES]
    placements += [_place("stone", *PEBBLE, ground_z(PEBBLE[0]), collision="none")]
    placements += [_place("house", x, y, ground_z(x), yaw_deg=90.0 * n) for n, (x, y) in enumerate(HOUSES)]
    placements += [_place("pond", *POND, ground_z(POND[0]) + 0.05)]
    placements += [_place("ridge", 0.0, 1500.0, 100.0)]
    manifest = {"version": 1, "andorra": {"core_m": float(SIDE_M), "frame": FRAME, "light": dict(LIGHT_GRADE), "shared_root": "../shared"},
                "items": items, "placements": placements,
                "tuft_field": {"folder": "grass", "table": "tufts.npy", "items": ["tuft"]},
                "far_trees": {"folder": "horizon", "table": "far_trees.npy", "items": ["far_tree"]}}
    _write(area / "manifest.json", json.dumps(manifest) + "\n")
    _write(area / "ground.json", json.dumps({"luminance": {"tile": GROUND_LUMINANCE}}) + "\n")
    _survey(area)
    rows = [{"id": 2, "slug": "02_not_installed", "place": "Missing", "kind": "town", "parish": "test", "frame": FRAME},
            {"id": 1, "slug": SLUG, "place": "Stand-in slope", "kind": "valley", "parish": "test", "frame": FRAME}]
    _write(root / "areas.json", json.dumps({"version": 1, "areas": rows}) + "\n")
    return area
