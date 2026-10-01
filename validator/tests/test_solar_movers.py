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

"""Solar map movers: the truck on its path, the bird on its loop, and what a seed draws.

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
from swarm.core.maps.solar.builder import CONFIG, SOLAR_ASSET_DIR, build_solar_map, build_solar_movers, solar_manifest, solar_mover_rules

ASSET_DIR = os.environ.get("SOLAR_ASSET_DIR", SOLAR_ASSET_DIR)
SEED = 0
HORIZON_S = 390.0
SUPPORT = {category.value for category in SUPPORT_CATEGORIES}

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ASSET_DIR, "manifest.json")),
                                reason=f"solar map assets not built at {ASSET_DIR}")


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


def test_the_truck_drives_every_seed_after_take_off_and_before_the_patrol_ends():
    """Every seed's truck leaves once the 40 s take-off is over, at a moment spread across the patrol, and reaches
    the dead end before the 390 s are out."""
    with open(os.path.join(ASSET_DIR, "movers", "pickup_path.json"), encoding="utf-8") as handle:
        drive_s = len(json.load(handle)["rows"]) / CONFIG["step_hz"]
    delays = np.array([solar_mover_rules(seed)["pickup_delay_s"] for seed in range(2000)])
    assert delays.min() >= 40.0
    assert delays.max() + drive_s <= HORIZON_S
    assert np.histogram(delays, bins=5, range=CONFIG["pickup_delay_s"])[0].min() > 300
