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

"""Solar map movers: the farmers on the public road, or the truck on its path where the map has no road, the bird on
its loop, and what a seed draws.

The map's assets are far too large to live in the repository, so the whole file is skipped unless SOLAR_ASSET_DIR
points at a built copy of them. Only the terrain and the movers are placed: the vegetation adds a minute to the
build and nothing here reads it.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pybullet as p
import pytest

from swarm.core.env_builder.sar_tagging import classify_body
from swarm.core.env_builder.sar_types import SUPPORT_CATEGORIES
from swarm.core.maps.solar import farmers
from swarm.core.maps.solar.builder import CONFIG, SOLAR_ASSET_DIR, _inside, _point_gap, build_solar_map, build_solar_movers, solar_fence, solar_manifest, solar_mover_rules

ASSET_DIR = os.environ.get("SOLAR_ASSET_DIR", SOLAR_ASSET_DIR)
SEED = 0
HORIZON_S = 390.0
SUPPORT = {category.value for category in SUPPORT_CATEGORIES}

ROAD = os.path.join(ASSET_DIR, "movers", CONFIG["road_file"])
ROAD_SHIPPED = os.path.exists(ROAD)

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ASSET_DIR, "manifest.json")),
                                reason=f"solar map assets not built at {ASSET_DIR}")
needs_road = pytest.mark.skipif(not ROAD_SHIPPED, reason=f"the installed swarm-worlds has no {ROAD}")
table_truck = pytest.mark.skipif(ROAD_SHIPPED, reason="the map ships the farmers' road, so the truck follows no table")


@pytest.fixture(scope="module")
def world():
    """A DIRECT client holding this seed's terrain and movers, and the runtime that drives them."""
    cli = p.connect(p.DIRECT)
    built = build_solar_map(SEED, cli, asset_dir=ASSET_DIR, groups=("terrain", "movers"))
    yield cli, built, build_solar_movers(built, seed=SEED, cli=cli)
    p.disconnect(cli)


def _by_item(built, item):
    """The body ids of every mover placement of one item."""
    return [body for place, body in built["movers"] if place["item"] == item]


def test_the_runtime_holds_every_mover_body_and_the_herd_is_left_out(world):
    """Every placement tagged as a mover is in the runtime, the truck and the bird both there, and no goat is built
    though the manifest still lists the herd."""
    _, built, movers = world
    assert len(built["movers"]) == len(movers.body_uids) > 0
    assert {place["mover"] for place, _ in built["movers"]} == {"pickup", "bird"}
    listed = {place["mover"] for place in solar_manifest(ASSET_DIR)["placements"] if "mover" in place}
    assert set(CONFIG["movers_left_out"]) <= listed | set(CONFIG["movers_left_out"])


@table_truck
def test_the_pickup_follows_its_table(world):
    """The truck waits out its delay at the first row, walks the table, and holds the last row at the dead end."""
    cli, built, movers = world
    with open(os.path.join(ASSET_DIR, "movers", "pickup_path.json"), encoding="utf-8") as handle:
        rows = json.load(handle)["rows"]
    delay = int(round(movers.rules["pickup_delay_s"] * CONFIG["step_hz"]))
    body = _by_item(built, "pickup_body")[0]
    movers.advance(max(delay - 1, 0))
    assert p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] == pytest.approx(rows[0][:3], abs=1e-4)
    movers.advance(delay + 300)
    assert p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] == pytest.approx(rows[300][:3], abs=1e-4)
    movers.advance(delay + len(rows) + 5000)
    assert p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] == pytest.approx(rows[-1][:3], abs=1e-4)


@table_truck
def test_the_pickup_wheels_ride_and_spin(world):
    """Each wheel holds its place in the truck's frame and turns there by exactly what the table's spin column says."""
    cli, built, movers = world
    with open(os.path.join(ASSET_DIR, "movers", "pickup_path.json"), encoding="utf-8") as handle:
        rows = json.load(handle)["rows"]
    delay = int(round(movers.rules["pickup_delay_s"] * CONFIG["step_hz"]))
    body = _by_item(built, "pickup_body")[0]
    wheel = _by_item(built, "pickup_wheel_front_left")[0]
    carried = []
    for step in (100, 105):
        movers.advance(delay + step)
        truck = p.getBasePositionAndOrientation(body, physicsClientId=cli)
        spun = p.getBasePositionAndOrientation(wheel, physicsClientId=cli)
        carried.append(p.multiplyTransforms(*p.invertTransform(*truck), spun[0], spun[1]))
    assert carried[0][0] == pytest.approx(carried[1][0], abs=1e-4)
    turned = 2.0 * math.acos(min(abs(p.getDifferenceQuaternion(carried[0][1], carried[1][1])[3]), 1.0))
    assert turned == pytest.approx(rows[105][7] - rows[100][7], abs=1e-4)


def test_the_bird_wraps_its_loop(world):
    """A step a whole loop later puts every bird part exactly where it was, and the step between does not."""
    cli, built, movers = world
    with open(os.path.join(ASSET_DIR, "movers", "bird_path.json"), encoding="utf-8") as handle:
        steps = json.load(handle)["steps"]
    bodies = [body for place, body in built["movers"] if place["mover"] == "bird"]
    movers.advance(0)
    first = [p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] for body in bodies]
    movers.advance(steps)
    assert [p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] for body in bodies] == pytest.approx(first)
    movers.advance(steps // 3)
    flown = [p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] for body in bodies]
    assert np.linalg.norm(np.array(flown[0]) - np.array(first[0])) > 1.0


def test_no_mover_is_ever_landable_terrain(world):
    """No mover reads as ground anywhere on its path, and the one that reads as a roof is the one that owns a box.

    The truck's box stands 2.4 m in its own axis-aligned bounds once the road tilts it, which is over the 2 m the
    rooftop rule uses, so under the two challenge types that have roofs it reads as one. Nothing in the map can
    argue it down: the family that takes this map has to tag the runtime's body_uids before the world is tagged.
    """
    cli, built, movers = world
    truck = _by_item(built, "pickup_body")[0]
    seen = {}
    for step in (0, 250, 500, 750, 998, 1500, 3000):
        movers.advance(step)
        for _, body in built["movers"]:
            for challenge in range(1, 7):
                seen.setdefault((body, challenge), set()).add(classify_body(cli, body, challenge_type=challenge))
    for (body, challenge), kinds in seen.items():
        roof = body == truck and challenge in (1, 4)
        assert kinds == ({"SUPPORT_ROOFTOP"} if roof else {"OBSTACLE_OTHER"}), f"body {body} type {challenge}: {kinds}"
    ground = SUPPORT - {"SUPPORT_ROOFTOP"}
    assert not ground & set().union(*seen.values())


def test_a_seed_draws_the_same_rules_twice_and_two_seeds_draw_different_ones():
    """The mover rules are a pure function of the seed, and they move with it."""
    assert solar_mover_rules(11) == solar_mover_rules(11)
    assert solar_mover_rules(11) != solar_mover_rules(12)
    drawn = [solar_mover_rules(seed) for seed in range(200)]
    assert len({rules["bird_phase_share"] for rules in drawn}) == len(drawn)
    assert len({rules["pickup_delay_s"] for rules in drawn}) == len(drawn)


@table_truck
def test_the_truck_drives_every_seed_after_take_off_and_before_the_patrol_ends():
    """Every seed's truck leaves once the 40 s take-off is over, at a moment spread across the patrol, and reaches
    the dead end before the 390 s are out."""
    with open(os.path.join(ASSET_DIR, "movers", "pickup_path.json"), encoding="utf-8") as handle:
        drive_s = len(json.load(handle)["rows"]) / CONFIG["step_hz"]
    delays = np.array([solar_mover_rules(seed)["pickup_delay_s"] for seed in range(2000)])
    assert delays.min() >= 40.0
    assert delays.max() + drive_s <= HORIZON_S
    assert np.histogram(delays, bins=5, range=CONFIG["pickup_delay_s"])[0].min() > 300


def _truck(built):
    """The pickup's body id, and its wheels' ids with their placements."""
    body = _by_item(built, "pickup_body")[0]
    return body, [(place, b) for place, b in built["movers"] if place["item"].startswith("pickup_wheel")]


@needs_road
def test_every_patrol_has_a_farmer_coming_after_another_has_gone():
    """Each seed brings farmers one at a time, never two on the road at once, and the second comes in within the
    patrol after the first has left; the first is already on the road or comes as the patrol starts."""
    road = farmers.load_road(ROAD)
    for seed in range(300):
        trips = farmers.plan(seed, road)
        assert len(trips) >= 2
        assert trips[0].start_s <= 0.0 < trips[0].end_s
        assert all(a.end_s < b.start_s for a, b in zip(trips, trips[1:]))
        assert 0.0 < trips[1].start_s < HORIZON_S
        assert trips[-1].start_s < HORIZON_S


@needs_road
def test_farmers_are_drawn_from_the_seed_and_change_with_it():
    """The same seed brings the same farmers at the same moments, and different seeds bring them at different ones."""
    road = farmers.load_road(ROAD)
    again = [(t.arrive_by, t.leave_by, t.start_s, t.cruise_m_s, t.stop_s) for t in farmers.plan(11, road)]
    assert again == [(t.arrive_by, t.leave_by, t.start_s, t.cruise_m_s, t.stop_s) for t in farmers.plan(11, road)]
    second = {round(farmers.plan(seed, road)[1].start_s, 3) for seed in range(100)}
    assert len(second) == 100
    trips = [t for seed in range(100) for t in farmers.plan(seed, road)]
    assert {(t.arrive_by, t.leave_by) for t in trips} == {(a, b) for a in farmers.ARMS for b in farmers.ARMS}
    assert 0.3 < np.mean([t.stop_s > 0 for t in trips]) < 0.7


@needs_road
def test_every_farmer_stays_outside_the_fence():
    """At every moment of every trip, all four corners of the truck stand outside the fence and at least a metre off it."""
    road = farmers.load_road(ROAD)
    body = solar_manifest(ASSET_DIR)["items"]["pickup_body"]
    (x0, y0), (x1, y1) = body["bounds_min"][:2], body["bounds_max"][:2]
    ring = np.asarray(solar_fence(ASSET_DIR))
    for arrive_by in farmers.ARMS:
        for leave_by in farmers.ARMS:
            pose = farmers._path(road, arrive_by, leave_by)[0]
            yaw = np.arctan2(2 * (pose[:, 6] * pose[:, 5] + pose[:, 3] * pose[:, 4]), 1 - 2 * (pose[:, 4] ** 2 + pose[:, 5] ** 2))
            c, s = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
            lx, ly = np.array([x0, x1, x1, x0]), np.array([y0, y0, y1, y1])
            corners = np.stack([pose[:, :1] + c * lx - s * ly, pose[:, 1:2] + s * lx + c * ly], axis=2).reshape(-1, 2)
            assert not _inside(ring, corners).any()
            assert _point_gap(corners, ring) >= 1.0


@needs_road
def test_a_farmer_drives_in_turns_round_backing_up_and_drives_out():
    """A trip keeps to its speeds, backs up only in the yard and only slowly, and halts each time it changes gear."""
    road = farmers.load_road(ROAD)
    trip = farmers._trip(road, "north", "east", 5.0, 20.0)
    gear = farmers._path(road, "north", "east")[2]
    assert trip.speed.max() <= 5.0 + 1e-9
    assert set(gear[gear < 0]) == {-1.0} and trip.speed[gear < 0].max() <= farmers.CONFIG["reverse_m_s"] + 1e-9
    changes = np.flatnonzero(gear[1:] != gear[:-1])
    assert len(changes) == 2 and (trip.speed[changes] == 0.0).all()
    yard = len(road.lines["north"]) - road.turn_from
    assert trip.leave[yard] - trip.reach[yard] == pytest.approx(20.0)
    backed = np.flatnonzero(gear < 0)
    assert backed.min() > yard and backed.max() < len(gear) - (len(road.lines["east"]) - road.turn_to)


@needs_road
def test_the_wheels_ride_on_the_ground_and_the_front_ones_steer(world):
    """Through a whole patrol every wheel sits within a few centimetres of the terrain under it, the front wheels
    turn about the vertical on full lock in the yard while the rear ones never do, and the empty road leaves the truck
    under the map."""
    cli, built, movers = world
    terrain = set(built["bodies"]["terrain"])
    body, wheels = _truck(built)
    gaps, steered, under = [], {}, 0
    for step in range(0, int(HORIZON_S * CONFIG["step_hz"]), 10):
        movers.advance(step)
        position, orientation = p.getBasePositionAndOrientation(body, physicsClientId=cli)
        if position[2] < -100.0:
            under += 1
            continue
        for place, wheel in wheels:
            at, turned = p.getBasePositionAndOrientation(wheel, physicsClientId=cli)
            hit = p.rayTest([at[0], at[1], at[2] - 0.25], [at[0], at[1], at[2] - 6.0], physicsClientId=cli)[0]
            assert hit[0] in terrain
            gaps.append(at[2] - 0.36 - hit[3][2])
            local = p.multiplyTransforms(*p.invertTransform(position, orientation), at, turned)[1]
            axle = p.rotateVector(local, [0.0, 1.0, 0.0])
            steered.setdefault(place["item"], []).append(abs(math.degrees(math.atan2(-axle[0], axle[1]))))
    gaps = np.abs(gaps)
    assert np.percentile(gaps, 99) < 0.03 and gaps.max() < 0.08
    assert max(steered["pickup_wheel_front_left"]) > 25.0 and max(steered["pickup_wheel_rear_left"]) < 2.0
    assert under > 0
