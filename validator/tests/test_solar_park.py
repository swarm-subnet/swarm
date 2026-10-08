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

"""Solar Patrol's park (task 17): each seed shifts the real site a little and flies it by day or by night.

The shifts are checked against the map as it ships, with the geometry worked out here rather than read back from
the builder: every row, building and tree moves, rows stay straight, nothing leaves its side of the fence or comes
near another piece, the survey is left as it is, and a built park stands each moved table on its own ground.
"""
from __future__ import annotations

import copy
import math
import os
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pybullet_data
import pytest

from swarm.challenge_families.solar_patrol import park
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.challenge_families.solar_patrol.family import SolarPatrolChallengeFamily
from swarm.core.daylight import SunLight, max_elevation_deg, seeded_sun
from swarm.core.maps.solar.builder import (
    CONFIG,
    SOLAR_ASSET_DIR,
    _Shapes,
    build_solar_map,
    solar_fence,
    solar_manifest,
    solar_shifts,
)

ASSET_DIR = os.environ.get("SOLAR_ASSET_DIR", SOLAR_ASSET_DIR)


def _has_rows(asset_dir):
    """Whether the built map at asset_dir stands panel tables in rows."""
    path = os.path.join(asset_dir, "manifest.json")
    return os.path.exists(path) and any("row" in place for place in solar_manifest(asset_dir)["placements"])


needs_park = pytest.mark.skipif(not _has_rows(ASSET_DIR), reason=f"solar park not built at {ASSET_DIR}")


def _yaw(quaternion):
    """Heading of a rotation about the vertical, radians."""
    x, y, z, w = quaternion
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _outline(place, items):
    """A placed piece seen from above: the corners of its bounds at its size and heading."""
    item = items[place["item"]]
    (x0, y0), (x1, y1) = (np.array(item[key][:2]) * place["scale"][:2] for key in ("bounds_min", "bounds_max"))
    yaw = _yaw(place["quaternion"])
    turn = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ turn.T + place["position"][:2]


def _inside(ring, points):
    """Which of the points lie inside a closed ring, each by the parity of the sides a ray to its east crosses."""
    x, y = points[:, 0:1], points[:, 1:2]
    x1, y1 = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    spans = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        east = x < x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return (spans & east).sum(axis=1) % 2 == 1


def _ring_gap(points, ring):
    """Smallest distance from any of the points to the sides of a closed ring."""
    start, side = ring, np.roll(ring, -1, axis=0) - ring
    offset = points[:, None, :] - start[None, :, :]
    along = np.clip((offset * side).sum(axis=2) / (side * side).sum(axis=1), 0.0, 1.0)
    return float(np.linalg.norm(offset - along[..., None] * side, axis=2).min())


def _crosses(a, b):
    """Whether two convex outlines overlap, by looking for an axis that separates them."""
    for poly in (a, b):
        for start, end in zip(poly, np.roll(poly, -1, axis=0)):
            normal = np.array([start[1] - end[1], end[0] - start[0]])
            if (a @ normal).max() < (b @ normal).min() or (b @ normal).max() < (a @ normal).min():
                return False
    return True


def _apart(a, b):
    """Distance between two convex outlines, 0 when they overlap."""
    if _crosses(a, b):
        return 0.0
    return min(_ring_gap(a, b), _ring_gap(b, a))


def _units(placements):
    """Placement indices of every unit that moves as one, keyed by kind and number, without the legs under tables."""
    units = {}
    for index, place in enumerate(placements):
        if "row" in place:
            units.setdefault(("table", place["row"]), []).append(index)
    for kind, groups in CONFIG["shift_units"].items():
        for number, names in enumerate(groups):
            units[(kind, number)] = [i for i, place in enumerate(placements) if place["item"] in names]
    return units


def _carrier(leg, frames, placements, items):
    """The table a leg stands under: the one it lies deepest inside, measured in each table's own axes."""
    def outside(index):
        """How far the leg lies outside a table's outline along its worse axis, negative inside it."""
        place, item = placements[index], items[placements[index]["item"]]
        yaw = _yaw(place["quaternion"])
        rel = np.subtract(placements[leg]["position"][:2], place["position"][:2])
        local = np.array([rel[0] * math.cos(yaw) + rel[1] * math.sin(yaw), -rel[0] * math.sin(yaw) + rel[1] * math.cos(yaw)])
        low, high = (np.array(item[key][:2]) * place["scale"][:2] for key in ("bounds_min", "bounds_max"))
        return float(np.max(np.abs(local - (low + high) / 2.0) - (high - low) / 2.0))

    return min(frames, key=outside)


def _fence_side(outlines, ring):
    """+1 when every corner stands inside the fence, -1 when every one stands outside, 0 when the fence runs through."""
    corners = _inside(ring, np.vstack(outlines))
    if corners.any() != corners.all() or any(_inside(outline, ring).any() for outline in outlines):
        return 0
    return 1 if corners.all() else -1


@needs_park
def test_two_seeds_stand_every_piece_somewhere_else_on_the_same_layout():
    """Two seeds shift every table with its legs, every building and the tree, each to a spot of its own, and leave the
    fence, its posts and its gate where the survey stands them."""
    placements = solar_manifest(ASSET_DIR)["placements"]
    a, b = solar_shifts(1, ASSET_DIR), solar_shifts(2, ASSET_DIR)
    assert set(a) == set(b)
    moved = {placements[index]["item"] for index in a}
    assert {"full_table_frame", "full_table_glass", "full_table_racking", "half_table_frame", "pile",
            "white_unit_north", "white_unit_south", "slab", "olive_bark", "olive_leaves_0"} <= moved
    assert not moved & {"fence_post", "fence_panel", "gate"}
    tables = [index for index, place in enumerate(placements) if "row" in place]
    apart = [math.dist(a[i]["position"][:2], b[i]["position"][:2]) for i in tables]
    assert min(apart) > 0.0
    assert np.median(apart) > 0.2


@needs_park
def test_a_row_moves_as_one_straight_string_with_its_legs():
    """Every table in a row turns by the same angle and takes the same size, and its tables and the legs under them
    keep their places against each other, turned and sized with the row, so a row stays straight and its tables
    never run into each other."""
    manifest = solar_manifest(ASSET_DIR)
    placements, items = manifest["placements"], manifest["items"]
    frames = [i for i, place in enumerate(placements) if "row" in place and place["item"].endswith("_frame")]
    carried = {i: _carrier(i, frames, placements, items) for i, place in enumerate(placements)
               if place["item"] in CONFIG["table_legs"]}
    for seed in range(12):
        shifts = solar_shifts(seed, ASSET_DIR)
        for row in {placements[i]["row"] for i in frames}:
            members = [i for i in frames if placements[i]["row"] == row]
            turns = [_yaw(shifts[i]["quaternion"]) - _yaw(placements[i]["quaternion"]) for i in members]
            sizes = [shifts[i]["scale"][0] / placements[i]["scale"][0] for i in members]
            assert np.ptp(turns) < 1e-9 and np.ptp(sizes) < 1e-9
            turn, size = turns[0], sizes[0]
            rotate = np.array([[math.cos(turn), -math.sin(turn)], [math.sin(turn), math.cos(turn)]]) * size
            first = members[0]
            for i in members[1:] + [leg for leg, frame in carried.items() if frame in members]:
                before = np.subtract(placements[i]["position"][:2], placements[first]["position"][:2])
                after = np.subtract(shifts[i]["position"][:2], shifts[first]["position"][:2])
                assert np.allclose(after, rotate @ before, atol=2e-4)


@needs_park
def test_moved_pieces_keep_to_their_side_of_the_fence_and_apart():
    """Over many seeds every unit keeps to the side of the fence it is surveyed on and at least the configured metre
    from it and from every other unit, or no nearer than the survey stands them; the one built into the fence stays."""
    manifest = solar_manifest(ASSET_DIR)
    placements, items = manifest["placements"], manifest["items"]
    ring = np.asarray(solar_fence(ASSET_DIR))
    units = _units(placements)
    survey = {key: [_outline(placements[i], items) for i in members] for key, members in units.items()}
    sides = {key: _fence_side(outlines, ring) for key, outlines in survey.items()}
    near = {(k1, k2): min(_apart(a, b) for a in survey[k1] for b in survey[k2])
            for k1 in units for k2 in units if k1 < k2}
    near = {pair: gap for pair, gap in near.items() if gap < CONFIG["shift_reach_m"]}
    assert sorted(sides.values()).count(0) == 1
    for seed in range(40):
        shifts = solar_shifts(seed, ASSET_DIR)
        moved = {key: [_outline(dict(placements[i], **{k: shifts[i][k] for k in ("position", "quaternion", "scale")}),
                                items) for i in members] for key, members in units.items()}
        for key, outlines in moved.items():
            if sides[key] == 0:
                assert all(np.allclose(a, b) for a, b in zip(outlines, survey[key]))
                continue
            assert _fence_side(outlines, ring) == sides[key]
            surveyed = min(_ring_gap(o, ring) for o in survey[key])
            assert min(_ring_gap(o, ring) for o in outlines) >= min(CONFIG["fence_clear_m"], surveyed) - 1e-6
        for (k1, k2), surveyed in near.items():
            gap = min(_apart(a, b) for a in moved[k1] for b in moved[k2])
            assert gap >= min(CONFIG["piece_gap_m"], surveyed) - 1e-6, (seed, k1, k2)


@needs_park
def test_each_table_takes_its_own_tint_never_brighter_than_the_survey():
    """Every table's parts share one tint, each colour a little darker than the survey's and never brighter, since a
    tint over 1 overflows the renderer; different tables take different tints."""
    placements = solar_manifest(ASSET_DIR)["placements"]
    shifts = solar_shifts(4, ASSET_DIR)
    darkest = 1.0 - CONFIG["shifts"]["table"]["tint"]
    tints = {}
    for index, place in enumerate(placements):
        if "row" in place:
            tint = shifts[index]["tint"]
            assert all(darkest <= channel <= 1.0 for channel in tint)
            tints.setdefault(tuple(place["position"]), set()).add(tuple(tint))
    assert all(len(shared) == 1 for shared in tints.values())
    assert len({next(iter(shared)) for shared in tints.values()}) == len(tints)


@needs_park
def test_the_same_seed_stands_the_same_park_and_the_survey_is_left_as_it_ships():
    """A seed always stands the same park, and drawing it never changes the survey the site map is read from."""
    before = copy.deepcopy(solar_manifest(ASSET_DIR)["placements"])
    assert solar_shifts(9, ASSET_DIR) == solar_shifts(9, ASSET_DIR)
    assert solar_shifts(9, ASSET_DIR) != solar_shifts(10, ASSET_DIR)
    assert solar_manifest(ASSET_DIR)["placements"] == before


@needs_park
@pytest.mark.timeout(600)
def test_a_built_park_stands_each_moved_table_on_its_own_ground():
    """Built with its terrain, every moved table stands over the ground under it as high as the survey's does over
    its own, give or take what its size adds, so no table floats or sinks where the seed moved it."""
    manifest = solar_manifest(ASSET_DIR)
    placements, items = manifest["placements"], manifest["items"]
    seed = 3
    shifts = solar_shifts(seed, ASSET_DIR)
    ground_cli, park_cli = p.connect(p.DIRECT), p.connect(p.DIRECT)
    try:
        build_solar_map(seed=seed, cli=ground_cli, asset_dir=ASSET_DIR, groups=("terrain",))
        world = build_solar_map(seed=seed, cli=park_cli, asset_dir=ASSET_DIR, groups=("park", "terrain"))

        def ground(xy):
            """Terrain height under a point, read on the terrain alone."""
            return p.rayTest([xy[0], xy[1], 1000.0], [xy[0], xy[1], -1000.0], physicsClientId=ground_cli)[0][3][2]

        frames = {tuple(np.round(placements[i]["position"], 4)): i for i, place in enumerate(placements)
                  if "row" in place and place["item"].endswith("_frame")}
        built = {}
        for body in world["bodies"]["park"]:
            position, _ = p.getBasePositionAndOrientation(body, physicsClientId=park_cli)
            built[tuple(np.round(position[:2], 3))] = position[2]
        for index in frames.values():
            place, shift = placements[index], shifts[index]
            parts = [placements[i] for i in range(len(placements)) if placements[i]["position"] == place["position"]]
            low = min(items[part["item"]]["bounds_min"][2] * part["scale"][2] for part in parts)
            size = shift["scale"][0] / place["scale"][0]
            z = built[tuple(np.round(shift["position"][:2], 3))]
            above = z - ground(shift["position"][:2])
            surveyed = place["position"][2] - ground(place["position"][:2])
            assert above == pytest.approx(surveyed - low * (size - 1.0), abs=0.01)
    finally:
        p.disconnect(ground_cli)
        p.disconnect(park_cli)


@needs_park
def test_building_the_park_sinks_the_default_floor_under_its_lowest_valley(monkeypatch):
    """The environment's flat floor stands at height 0, where it shows through the valleys south-east of the fence;
    building the park sinks it under the map's lowest ground, so a look straight down there meets nothing of it."""
    manifest = solar_manifest(ASSET_DIR)
    lowest = min(place["position"][2] + manifest["items"][place["item"]]["bounds_min"][2] * place["scale"][2]
                 for place in manifest["placements"] if manifest["items"][place["item"]].get("group") == "terrain")
    monkeypatch.setattr(park, "build_solar_map", lambda seed, cli: {"asset_dir": ASSET_DIR, "bodies": {}})
    monkeypatch.setattr(park, "build_solar_movers", lambda world, seed, cli: None)
    cli = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cli)
        floor = p.loadURDF("plane.urdf", physicsClientId=cli)
        park.reset(SimpleNamespace(CLIENT=cli, PLANE_ID=floor, _sun=None), SolarEpisode(seed=1))
        assert p.getBasePositionAndOrientation(floor, physicsClientId=cli)[0][2] < lowest
        assert p.rayTest([86.0, -76.0, 100.0], [86.0, -76.0, lowest], physicsClientId=cli)[0][0] == -1
    finally:
        p.disconnect(cli)


def _light(elevation_deg, night=False):
    """A sun, or a moon, at an elevation."""
    return SunLight(hour=12.0, elevation_deg=elevation_deg, azimuth_deg=0.0, direction=(0.0, 0.0, 1.0),
                    color=(1.0, 1.0, 1.0), ambient=0.5, diffuse=0.3, night=night, sky=None)


def test_a_noon_sun_warms_the_panel_glass_to_its_measured_range():
    """Under the noon sun every table's glass reads 45 to 65 C, the day's air lies in its range, and the clear sky
    reads 30 to 45 K under the air, for every seed."""
    for seed in range(200):
        heat = park.heat(_light(max_elevation_deg()), seed, 58)
        assert park.AIR_DAY_C[0] <= heat["air_c"] <= park.AIR_DAY_C[1]
        assert 30.0 - 0.1 <= heat["air_c"] - heat["sky_c"] <= 45.0 + 0.1
        assert len(heat["panel_c"]) == 58
        assert 45.0 - 0.1 <= min(heat["panel_c"]) and max(heat["panel_c"]) <= 65.0 + 0.1
        assert len(set(heat["panel_c"])) > 1


def test_a_low_sun_barely_warms_the_glass_and_a_night_leaves_it_to_the_engine():
    """A sun a few degrees up warms the glass only a little over the air; at night the air is the night's and no
    glass is set, so the engine keeps it a few degrees under the air as the cold sky it mirrors does."""
    low = park.heat(_light(5.0), 1, 10)
    assert max(low["panel_c"]) < low["air_c"] + 5.0 + park.PANEL_SPREAD_C
    night = park.heat(_light(40.0, night=True), 1, 10)
    assert park.AIR_NIGHT_C[0] <= night["air_c"] <= park.AIR_NIGHT_C[1]
    assert night["panel_c"] == []
    assert park.heat(_light(40.0), 1, 10) == park.heat(_light(40.0), 1, 10)


def test_half_the_seeds_fly_at_night_under_the_seed_s_own_light():
    """The family lights every seed with its own sun or moon, and half the seeds are nights."""
    family = SolarPatrolChallengeFamily
    assert family.seeded_sun is True and family.night_share == 0.5
    assert family.sky_from_sun == park.SKY_FROM_SUN and family.daylight == park.DAYLIGHT
    nights = sum(seeded_sun(seed, family.night_share).night for seed in range(4000))
    assert 0.47 < nights / 4000 < 0.53


@pytest.mark.skipif(not hasattr(p, "VISUAL_SHAPE_GLASS_BACKED"), reason="engine without backed glass")
def test_only_the_panel_glass_is_drawn_over_a_backsheet(monkeypatch):
    """The table glass of the park asks for the backsheet; the pickup's windows stay glass that looks through."""
    asked = {}

    def record(*_args, **kwargs):
        """Keep the flags a visual shape was asked with."""
        asked["flags"] = kwargs["flags"]
        return 0

    monkeypatch.setattr(p, "createVisualShape", record)
    items = solar_manifest(SOLAR_ASSET_DIR)["items"]
    shapes = _Shapes(0, SOLAR_ASSET_DIR, items)
    backed = p.VISUAL_SHAPE_GLASS_BACKED
    for name, want in (("full_table_glass", backed), ("half_table_glass", backed), ("pickup_glass", 0)):
        shapes._visual(items[name], "unused.obj", [1.0, 1.0, 1.0])
        assert asked["flags"] & p.VISUAL_SHAPE_GLASS
        assert asked["flags"] & backed == want, name


@needs_park
def test_an_engine_without_backed_glass_is_refused(monkeypatch):
    """A wheel that cannot draw the panel glass over its backsheet fails loudly instead of drawing another picture."""
    monkeypatch.setattr(p, "createVisualShape", lambda *args, **kwargs: 0)
    monkeypatch.delattr(p, "VISUAL_SHAPE_GLASS_BACKED", raising=False)
    items = solar_manifest(SOLAR_ASSET_DIR)["items"]
    with pytest.raises(RuntimeError, match="VISUAL_SHAPE_GLASS_BACKED"):
        _Shapes(0, SOLAR_ASSET_DIR, items)._visual(items["full_table_glass"], "unused.obj", [1.0, 1.0, 1.0])
