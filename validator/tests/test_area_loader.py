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

"""The area loader on the stand-in area: the catalogue, every piece built where the manifest puts it, the two load
boxes, the survey grids under the built ground, and an optional check on a real package."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pybullet as p
import pytest

from swarm.core.maps import andorra
from swarm.core.maps.andorra import builder
from swarm.core.maps.andorra.survey import PERSON_KEY, Survey
from validator.tests import area_stand_in as stand_in

KEEP = (-20.0, -20.0, 40.0, 40.0)      # nine of the 25 tiles, two trees, two rocks, one house, two tufts
COLLIDE = (0.0, 0.0, 25.0, 25.0)       # one tree, no rock, no house


@pytest.fixture
def root(tmp_path):
    """A stand-in package in its own folder."""
    stand_in.write_area(tmp_path / "andorra")
    return tmp_path / "andorra"


@pytest.fixture
def client():
    """One DIRECT client, closed after the test."""
    cli = p.connect(p.DIRECT)
    yield cli
    p.disconnect(cli)


def _down(cli, x, y):
    """The first body a ray dropped from high above a point meets, any collision group."""
    return p.rayTest([x, y, 500.0], [x, y, -500.0], physicsClientId=cli)[0][0]


def _across(cli, x, y, half=5.0):
    """The first body a level ray 1 m over the ground meets, crossing a point from west to east."""
    z = stand_in.ground_z(x) + 1.0
    return p.rayTest([x - half, y, z], [x + half, y, z], physicsClientId=cli)[0][0]


def _forest_rows(path: Path) -> int:
    """How many placements a forest file lists, binary or text."""
    data = path.read_bytes()
    lines = data.split(b"\n")
    binary = [line for line in lines if line.startswith(b"binary ")]
    if binary:
        return int(binary[0].split()[1])
    return sum(1 for line in lines if line and not line.startswith(b"mesh "))


def test_catalogue_lists_the_areas_by_id(root):
    """Rows come back sorted by id, whatever order the file holds them in."""
    assert [row["id"] for row in andorra.catalog(root)] == [1, 2]


def test_root_follows_the_environment(root, monkeypatch):
    """SWARM_ANDORRA_ROOT points the package at another folder; unset, it is the installed worlds' maps."""
    monkeypatch.setenv(andorra.ROOT_ENV, str(root))
    assert andorra.andorra_root() == root
    assert andorra.area_folder(1) == root / stand_in.SLUG
    monkeypatch.delenv(andorra.ROOT_ENV)
    assert andorra.andorra_root().name == "andorra"


def test_area_folder_by_row_or_id_and_clear_failures(root, tmp_path):
    """An area is found by its row or its id; a missing catalogue, id or folder is named in the error."""
    rows = andorra.catalog(root)
    assert andorra.area_folder(rows[0], root) == andorra.area_folder(1, root) == root / stand_in.SLUG
    with pytest.raises(FileNotFoundError, match="02_not_installed"):
        andorra.area_folder(2, root)
    with pytest.raises(KeyError, match="no area 7"):
        andorra.area_folder(7, root)
    with pytest.raises(FileNotFoundError, match="catalogue missing"):
        andorra.catalog(tmp_path / "nowhere")
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "areas.json").write_text(json.dumps({"version": 1, "areas": []}))
    with pytest.raises(ValueError, match="empty"):
        andorra.catalog(tmp_path / "empty")


def test_shared_pieces_resolve_through_the_manifest(root):
    """An item marked shared is read from the manifest's shared folder, any other from the area's own."""
    area = root / stand_in.SLUG
    manifest = json.loads((area / "manifest.json").read_text())
    tile = builder.asset_path(area, manifest, manifest["items"]["tile"], "tile.obj")
    assert tile.resolve() == (root / "shared" / "terrain" / "tile.obj").resolve()
    assert builder.asset_path(area, manifest, manifest["items"]["house"], "house.obj") == area / "buildings" / "house.obj"


def test_whole_area_stands_as_its_manifest_says(root, client, tmp_path):
    """Every tile is ground under its slope; trunks, stones and houses block; the pebble and the pond only show; the
    forests, tufts and far trees are one instanced body each."""
    area = builder.load_area(root / stand_in.SLUG, client, work=tmp_path)
    assert len(area.terrain) == len(stand_in.tile_centres())
    # Four instanced bodies, the tiles, two houses, the pond and the ridge.
    assert len(area.bodies) == 4 + len(area.terrain) + len(stand_in.HOUSES) + 2
    for x, y in ((0.0, 0.0), (-70.0, 50.0), (75.0, -75.0)):
        assert builder.ground_height(x, y, client, area.terrain) == pytest.approx(stand_in.ground_z(x), abs=1e-3)
    colliders = set(area.colliders)
    assert all(_across(client, x, y) in colliders for x, y in stand_in.TREES)
    assert all(_down(client, x, y) in colliders for x, y in stand_in.STONES + stand_in.HOUSES)
    assert _down(client, *stand_in.PEBBLE) in area.terrain
    assert _down(client, *stand_in.POND) in area.terrain
    assert _forest_rows(tmp_path / "trees.fst") == len(stand_in.TREES)
    assert _forest_rows(tmp_path / "rocks.fst") == len(stand_in.STONES) + 1
    assert _forest_rows(tmp_path / "tufts.fst") == len(stand_in.TUFTS)
    assert _forest_rows(tmp_path / "far_trees.fst") == len(stand_in.FAR_TREES)


def test_each_group_drawn_a_body_a_placement_names_its_bodies(root, client, tmp_path):
    """The load names the bodies of every group drawn one placement to a body: the tiles, the two houses, the pond."""
    area = builder.load_area(root / stand_in.SLUG, client, work=tmp_path)
    assert area.groups["terrain"] == area.terrain
    assert len(area.groups["buildings"]) == len(stand_in.HOUSES) and len(area.groups["water"]) == 1
    assert set().union(*area.groups.values()) <= set(area.bodies)


def test_each_forest_names_its_one_body(root, client, tmp_path):
    """The load names the instanced body of the trees and of the rocks, apart from every body a group names; a load
    whose box holds no tree names none for the trees."""
    area = builder.load_area(root / stand_in.SLUG, client, work=tmp_path)
    assert set(area.forests) == {"trees", "rocks"} and area.forests["trees"] != area.forests["rocks"]
    assert set(area.forests.values()) <= set(area.bodies)
    assert not set(area.forests.values()) & set().union(*area.groups.values())
    bare = builder.load_area(root / stand_in.SLUG, client, work=tmp_path, keep_xy=(-50.0, -50.0, -40.0, -40.0))
    assert "trees" not in bare.forests


def test_tufts_stand_as_one_body_a_cell_inside_their_own_box(root, client, tmp_path):
    """With a cell size the tufts are stood as one body a grid cell that holds any, west to east then south to north,
    each with the box its tufts fill; a tuft box cuts them closer than the keep box; without a cell there is no
    patch."""
    assert builder.load_area(root / stand_in.SLUG, client, work=tmp_path, keep_xy=KEEP).patches == ()
    whole = builder.load_area(root / stand_in.SLUG, client, work=tmp_path, tuft_cell_m=100.0)
    # Cells (-1, 0), (0, -1) and (0, 0) of a 100 m grid: the far north-west tuft, the far south-east one, the two near.
    assert [_forest_rows(tmp_path / f"tufts_{cell}.fst") for cell in ("-1_0", "0_-1", "0_0")] == [1, 1, 2]
    assert len(whole.patches) == 3 and len({patch.body for patch in whole.patches}) == 3
    assert all(patch.body in whole.bodies for patch in whole.patches)
    near = whole.patches[2]
    (west, south), (east, north) = stand_in.TUFTS[0], stand_in.TUFTS[1]
    assert near.low == pytest.approx((west - builder.TUFT_REACH, south - builder.TUFT_REACH, stand_in.ground_z(west) - 0.5))
    assert near.high == pytest.approx((east + builder.TUFT_REACH, north + builder.TUFT_REACH, stand_in.ground_z(east) + 1.0))
    cut = builder.load_area(root / stand_in.SLUG, client, work=tmp_path, keep_xy=KEEP, tuft_xy=(10.0, 0.0, 20.0, 10.0),
                            tuft_cell_m=10.0)
    assert len(cut.patches) == 1 and _forest_rows(tmp_path / "tufts_1_0.fst") == 1
    assert len(cut.terrain) == 9
    nothing = builder.load_area(root / stand_in.SLUG, client, work=tmp_path, tuft_xy=(40.0, 40.0, 50.0, 50.0), tuft_cell_m=10.0)
    assert nothing.patches == ()


def test_a_mesh_every_area_shares_is_one_tidy_path(root):
    """A stone drawn with a mesh of the shared folder, named from the area's own folder, is read at the shared
    folder's own path with no step back left in it, so the engine sees one file whichever area names it."""
    area = root / stand_in.SLUG
    manifest = json.loads((area / "manifest.json").read_text())
    stone = dict(manifest["items"]["stone"], visual="../../shared/rocks/stone_drawn.obj")
    drawn = builder.visual_path(area, manifest, stone)
    assert drawn == root / "shared" / "rocks" / "stone_drawn.obj" and ".." not in drawn.parts
    assert builder.asset_path(area, manifest, stone, stone["obj"]) == area / "rocks" / "stone.obj"


def test_keep_box_builds_only_what_reaches_into_it(root, client, tmp_path):
    """Tiles, trees, rocks, houses and tufts wholly outside the keep box are skipped; the horizon and the far trees
    stay; what is built still collides everywhere."""
    area = builder.load_area(root / stand_in.SLUG, client, work=tmp_path, keep_xy=KEEP)
    assert len(area.terrain) == 9
    assert _forest_rows(tmp_path / "trees.fst") == 2
    assert _forest_rows(tmp_path / "rocks.fst") == 2
    assert _forest_rows(tmp_path / "tufts.fst") == 2
    assert _forest_rows(tmp_path / "far_trees.fst") == len(stand_in.FAR_TREES)
    # Four instanced bodies, nine tiles, the one house that reaches in, and the ridge; the pond lies outside.
    assert len(area.bodies) == 4 + 9 + 1 + 1
    assert _down(client, *stand_in.HOUSES[0]) in area.colliders
    assert _down(client, *stand_in.HOUSES[1]) == -1
    with pytest.raises(RuntimeError, match="no terrain"):
        builder.ground_height(-70.0, 70.0, client, area.terrain)


def test_a_piece_reaching_into_the_keep_box_is_kept_whole(root, client, tmp_path):
    """A house whose centre lies outside the box but whose walls reach into it is built and solid."""
    house = stand_in.HOUSES[0]
    box = (house[0] + stand_in.HOUSE_HALF_M[0] - 1.0, -10.0, 60.0, 60.0)
    area = builder.load_area(root / stand_in.SLUG, client, work=tmp_path, keep_xy=box)
    assert _down(client, *house) in area.colliders


def test_collide_box_limits_solid_shapes_but_not_the_ground(root, client, tmp_path):
    """Inside the collide box a trunk blocks; outside it a tree and a house are drawn but never hit, while every
    built tile is still ground."""
    area = builder.load_area(root / stand_in.SLUG, client, work=tmp_path, keep_xy=KEEP, collide_xy=COLLIDE)
    assert _across(client, 10.0, 10.0) in area.colliders
    assert _across(client, 20.0, -5.0) == -1
    assert _down(client, *stand_in.HOUSES[0]) in area.terrain
    assert _forest_rows(tmp_path / "trees.fst") == 2
    assert builder.ground_height(-30.0, -10.0, client, area.terrain) == pytest.approx(stand_in.ground_z(-30.0), abs=1e-3)


def test_footprint_turns_with_the_placement(root):
    """A house turned 90 degrees covers its length north to south."""
    manifest = json.loads((root / stand_in.SLUG / "manifest.json").read_text())
    houses = [pl for pl in manifest["placements"] if pl["item"] == "house"]
    straight, turned = builder.footprints(manifest, houses)
    (hx, hy), (tx, ty) = stand_in.HOUSES
    half_x, half_y = stand_in.HOUSE_HALF_M
    assert straight == pytest.approx([hx - half_x, hy - half_y, hx + half_x, hy + half_y])
    assert turned == pytest.approx([tx - half_y, ty - half_x, tx + half_y, ty + half_x])


def test_another_light_grade_or_a_missing_piece_is_refused(root, client):
    """An area converted under another grade, or one missing a mesh, fails before any body is built."""
    path = root / stand_in.SLUG / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["andorra"]["light"]["exposure_scale"] = 0.5
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="light grade"):
        builder.load_area(root / stand_in.SLUG, client)
    manifest["andorra"]["light"]["exposure_scale"] = 0.85
    path.write_text(json.dumps(manifest))
    (root / stand_in.SLUG / "buildings" / "house.obj").unlink()
    with pytest.raises(FileNotFoundError, match="house.obj"):
        builder.load_area(root / stand_in.SLUG, client)
    assert p.getNumBodies(physicsClientId=client) == 0


def test_survey_reads_the_grids_and_every_layer(root):
    """The grids keep their centimetres, a cell with no data reads NaN ground and an endless top, and the open, person
    and dock layers leave out the houses, the pond and the border."""
    survey = Survey(root / stand_in.SLUG / "spots.npz")
    assert survey.shape == (stand_in.SIDE_M, stand_in.SIDE_M) and survey.metadata["id"] == 1
    assert np.isnan(survey.ground_m[stand_in.NO_DATA_CELL]) and np.isinf(survey.ceiling_m[stand_in.NO_DATA_CELL])
    for x, y in stand_in.HOUSES + [stand_in.POND, (-75.0, 0.0)]:
        assert not survey.open[survey.cell(x, y)] and not survey.person[survey.cell(x, y)]
    assert survey.open[survey.cell(0.0, -40.0)] and survey.dock[survey.cell(0.0, -40.0)]
    assert survey.person[survey.cell(0.0, -40.0)]
    assert not (survey.dock & ~survey.open).any()
    house = survey.cell(*stand_in.HOUSES[0])
    assert survey.ceiling_cm[house] - survey.ground_cm[house] == stand_in.HOUSE_HEIGHT_M * 100


def test_survey_cells_and_centres_agree():
    """A cell's centre lies in that cell, half a metre from its edges, rows counting north to south."""
    survey = Survey.__new__(Survey)
    survey.shape = (stand_in.SIDE_M, stand_in.SIDE_M)
    assert survey.cell(0.2, 0.2) == (79, 80) and survey.cell(-80.0, 79.9) == (0, 0)
    rows, cols = np.array([0, 79, 159]), np.array([5, 80, 159])
    x, y = survey.centre(rows, cols)
    back = survey.cell(x, y)
    assert (back[0] == rows).all() and (back[1] == cols).all()
    assert survey.centre(0, 0) == (-79.5, 79.5)


def test_survey_heights_sit_on_the_built_ground(root, client):
    """A survey height plus the frame's shift is the built ground's height at that cell, within the grid's 1 cm."""
    area = builder.load_area(root / stand_in.SLUG, client)
    survey = Survey(root / stand_in.SLUG / "spots.npz")
    shift = survey.z_shift(area.manifest["andorra"]["frame"])
    for x, y in ((0.5, 0.5), (-60.5, -20.5), (70.5, 40.5)):
        assert builder.ground_height(x, y, client, area.terrain) == pytest.approx(survey.ground_m[survey.cell(x, y)] + shift, abs=0.01)
    with pytest.raises(ValueError, match="origin"):
        survey.z_shift({**stand_in.FRAME, "origin_en": [0.0, 0.0]})


def test_survey_without_a_dock_or_person_layer_reads_none(root):
    """A package from before the dock and person layers loads with neither."""
    path = root / stand_in.SLUG / "spots.npz"
    with np.load(path) as data:
        arrays = {key: data[key] for key in data.files if key not in ("dock_bits", PERSON_KEY)}
    np.savez(path, **arrays)
    survey = Survey(path)
    assert survey.dock is None and survey.person is None


@pytest.fixture
def package():
    """A real area package named by SWARM_ANDORRA_PACKAGE; the test is skipped without one."""
    configured = os.environ.get("SWARM_ANDORRA_PACKAGE")
    if not configured or not (Path(configured) / "areas.json").is_file():
        pytest.skip("set SWARM_ANDORRA_PACKAGE to a real area package to run this check")
    return Path(configured)


def test_real_area_loads_whole_and_cut_on_its_survey(package, client):
    """The first real area builds whole and cut round its centre, and its ground lies on its survey within 0.3 m."""
    folder = andorra.area_folder(andorra.catalog(package)[0], package)
    whole = builder.load_area(folder, client)
    survey = Survey(folder / "spots.npz")
    shift = survey.z_shift(whole.manifest["andorra"]["frame"])
    rows, cols = np.nonzero(survey.open)
    for n in np.linspace(0, len(rows) - 1, 12).astype(int):
        x, y = survey.centre(rows[n], cols[n])
        assert builder.ground_height(float(x), float(y), client, whole.terrain) == pytest.approx(
            survey.ground_m[rows[n], cols[n]] + shift, abs=0.3)
    p.resetSimulation(physicsClientId=client)
    cut = builder.load_area(folder, client, keep_xy=(-100.0, -100.0, 100.0, 100.0), collide_xy=(-30.0, -30.0, 30.0, 30.0))
    assert len(cut.terrain) < len(whole.terrain) and len(cut.colliders) < len(whole.colliders)
