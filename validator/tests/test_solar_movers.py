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

"""Solar map movers: the truck on its path, the bird on its loop, the herd on the terrain, and what a seed draws.

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
from swarm.core.maps.solar.builder import CONFIG, SOLAR_ASSET_DIR, _stride_pose, build_solar_map, build_solar_movers, solar_manifest, solar_mover_rules

ASSET_DIR = os.environ.get("SOLAR_ASSET_DIR", SOLAR_ASSET_DIR)
SEED = 0
WALK_STEPS = 100
SUPPORT = {category.value for category in SUPPORT_CATEGORIES}

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(ASSET_DIR, "manifest.json")),
                                reason=f"solar map assets not built at {ASSET_DIR}")


def _herd(asset_dir):
    """The herd table the movers fragment names, read straight off disk."""
    with open(os.path.join(asset_dir, "movers", "goat_herd.json"), encoding="utf-8") as handle:
        return json.load(handle)


def _synthetic_stride(asset_dir, target):
    """An asset directory like the real one but with a stride for the herd, for a build that shipped without one.

    The stand-in is the rest pose repeated, in the face-corner order the engine holds the mesh in, so it has the
    shape and the vertex count the real table has and the walk can be driven before that table lands.
    """
    os.mkdir(target)
    for name in os.listdir(asset_dir):
        if name != "movers":
            os.symlink(os.path.join(asset_dir, name), os.path.join(target, name))
    movers = os.path.join(target, "movers")
    os.mkdir(movers)
    for name in os.listdir(os.path.join(asset_dir, "movers")):
        if name not in ("goat_herd.json", "goat_poses.npz"):
            os.symlink(os.path.join(asset_dir, "movers", name), os.path.join(movers, name))
    vertices, corners = [], []
    with open(os.path.join(asset_dir, "movers", "goat.obj"), encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
            elif line.startswith("f "):
                corners += [int(token.split("/")[0]) - 1 for token in line.split()[1:4]]
    herd = _herd(asset_dir)
    rest = np.asarray(vertices, dtype=np.float32)[np.asarray(corners)]
    np.savez_compressed(os.path.join(movers, "goat_poses.npz"), hz=herd["cycle_fps"], stride_m=herd["stride_m"],
                        poses=np.repeat(rest[None], herd["cycle_frames"], axis=0))
    herd["poses"] = "goat_poses.npz"
    with open(os.path.join(movers, "goat_herd.json"), "w", encoding="utf-8") as handle:
        json.dump(herd, handle)
    return target


@pytest.fixture(scope="module")
def walking_assets(tmp_path_factory):
    """An asset directory whose herd carries a stride: the real one where it does, a stand-in where it does not."""
    if _herd(ASSET_DIR).get("poses"):
        return ASSET_DIR
    return _synthetic_stride(ASSET_DIR, str(tmp_path_factory.mktemp("solar") / "assets"))


@pytest.fixture(scope="module")
def world(walking_assets):
    """A DIRECT client holding this seed's terrain and movers, and the runtime that drives them."""
    cli = p.connect(p.DIRECT)
    built = build_solar_map(SEED, cli, asset_dir=walking_assets, groups=("terrain", "movers"))
    yield cli, built, build_solar_movers(built, seed=SEED, cli=cli)
    p.disconnect(cli)


def _ground(cli, x, y):
    """Terrain height under a point, from a ray dropped from well above the park."""
    hit = p.rayTest([float(x), float(y), 400.0], [float(x), float(y), -80.0], physicsClientId=cli)[0]
    return float(hit[3][2]) if int(hit[0]) >= 0 else None


def _by_item(built, item):
    """The body ids of every mover placement of one item."""
    return [body for place, body in built["movers"] if place["item"] == item]


def test_the_runtime_holds_every_mover_body(world):
    """Every placement tagged as a mover is in the runtime, and the manifest's three movers are all there."""
    _, built, movers = world
    assert len(built["movers"]) == len(movers.body_uids) > 0
    kinds = {place["mover"] for place, _ in built["movers"]}
    assert kinds == {"pickup", "bird", "goat"}
    assert movers.goat_stride


def test_the_stride_is_read_from_the_pose_table(world, walking_assets):
    """How long the walk is, how fast it plays and how far it carries the animal all come off the table itself."""
    _, _, movers = world
    herd = _herd(walking_assets)
    with np.load(os.path.join(walking_assets, "movers", herd["poses"])) as table:
        assert movers.stride == {"frames": len(table["poses"]), "fps": float(table["hz"]),
                                 "stride_m": float(table["stride_m"])}


def test_the_stride_blends_between_its_frames():
    """A whole phase is one frame of the table, a part phase is the two it falls between, and the end wraps round."""
    poses = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    assert _stride_pose(poses, 0.0) == pytest.approx(poses[0])
    assert _stride_pose(poses, 1.0) == pytest.approx(poses[1])
    assert _stride_pose(poses, 2.0) == pytest.approx(poses[0])
    assert _stride_pose(poses, 0.25) == pytest.approx(0.75 * poses[0] + 0.25 * poses[1])
    assert _stride_pose(poses, 1.5) == pytest.approx(0.5 * (poses[0] + poses[1]))


def test_the_pickup_follows_its_table(world, walking_assets):
    """The truck waits out its delay at the first row, walks the table, and holds the last row at the dead end."""
    cli, built, movers = world
    with open(os.path.join(walking_assets, "movers", "pickup_path.json"), encoding="utf-8") as handle:
        rows = json.load(handle)["rows"]
    delay = int(round(movers.rules["pickup_delay_s"] * CONFIG["step_hz"]))
    body = _by_item(built, "pickup_body")[0]
    movers.advance(max(delay - 1, 0))
    assert p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] == pytest.approx(rows[0][:3], abs=1e-4)
    movers.advance(delay + 300)
    assert p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] == pytest.approx(rows[300][:3], abs=1e-4)
    movers.advance(delay + len(rows) + 5000)
    assert p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] == pytest.approx(rows[-1][:3], abs=1e-4)


def test_the_pickup_wheels_ride_and_spin(world, walking_assets):
    """Each wheel holds its place in the truck's frame and turns there by exactly what the table's spin column says."""
    cli, built, movers = world
    with open(os.path.join(walking_assets, "movers", "pickup_path.json"), encoding="utf-8") as handle:
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


def test_the_bird_wraps_its_loop(world, walking_assets):
    """A step a whole loop later puts every bird part exactly where it was, and the step between does not."""
    cli, built, movers = world
    with open(os.path.join(walking_assets, "movers", "bird_path.json"), encoding="utf-8") as handle:
        steps = json.load(handle)["steps"]
    bodies = [body for place, body in built["movers"] if place["mover"] == "bird"]
    movers.advance(0)
    first = [p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] for body in bodies]
    movers.advance(steps)
    assert [p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] for body in bodies] == pytest.approx(first)
    movers.advance(steps // 3)
    flown = [p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] for body in bodies]
    assert np.linalg.norm(np.array(flown[0]) - np.array(first[0])) > 1.0


def test_the_goats_keep_their_feet_on_the_terrain(world, walking_assets):
    """Over a hundred steps every goat walks, and its foot stays within a few centimetres of the ground under it."""
    cli, built, movers = world
    foot = float(solar_manifest(walking_assets)["items"]["goat"]["bounds_min"][2])
    bodies = _by_item(built, "goat")
    movers.advance(0)
    start = [p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] for body in bodies]
    worst, relief = 0.0, 0.0
    for step in range(WALK_STEPS):
        movers.advance(step)
        for body, was in zip(bodies, start):
            position = p.getBasePositionAndOrientation(body, physicsClientId=cli)[0]
            ground = _ground(cli, position[0], position[1])
            assert ground is not None
            worst = max(worst, abs(position[2] + foot - ground))
            relief = max(relief, abs(ground - was[2] - foot))
    walked = [np.linalg.norm(np.array(p.getBasePositionAndOrientation(body, physicsClientId=cli)[0][:2])
                             - np.array(was[:2])) for body, was in zip(bodies, start)]
    assert min(walked) > 0.5
    # A goat that never left its first height would pass on flat ground, so the ground has to move under it too.
    assert relief > 0.05
    assert worst < 0.05


def test_the_goats_face_the_way_they_walk(world):
    """Each goat's nose, its own -y turned by its pose, points along the ground it covers in its next steps."""
    cli, built, movers = world
    for body in _by_item(built, "goat"):
        movers.advance(0)
        start, pose = p.getBasePositionAndOrientation(body, physicsClientId=cli)
        movers.advance(10)
        travel = np.array(p.getBasePositionAndOrientation(body, physicsClientId=cli)[0][:2]) - np.array(start[:2])
        nose = np.array(p.getMatrixFromQuaternion(pose)).reshape(3, 3) @ np.array([0.0, -1.0, 0.0])
        assert float(nose[:2] @ travel) / np.linalg.norm(travel) > 0.98


def test_the_goats_stand_still_without_a_stride(walking_assets, tmp_path):
    """A herd table with no pose file leaves the animals where the manifest put them."""
    target = str(tmp_path / "still")
    os.mkdir(target)
    for name in os.listdir(walking_assets):
        if name != "movers":
            os.symlink(os.path.join(walking_assets, name), os.path.join(target, name))
    os.mkdir(os.path.join(target, "movers"))
    for name in os.listdir(os.path.join(walking_assets, "movers")):
        if name != "goat_herd.json":
            os.symlink(os.path.join(walking_assets, "movers", name), os.path.join(target, "movers", name))
    herd = dict(_herd(walking_assets), poses=None)
    with open(os.path.join(target, "movers", "goat_herd.json"), "w", encoding="utf-8") as handle:
        json.dump(herd, handle)
    cli = p.connect(p.DIRECT)
    try:
        built = build_solar_map(SEED, cli, asset_dir=target, groups=("movers",))
        movers = build_solar_movers(built, seed=SEED, cli=cli)
        assert not movers.goat_stride
        bodies = _by_item(built, "goat")
        before = [p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] for body in bodies]
        for step in range(20):
            movers.advance(step)
        assert [p.getBasePositionAndOrientation(body, physicsClientId=cli)[0] for body in bodies] == before
    finally:
        p.disconnect(cli)


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
    assert len({rules["goat_seed"] for rules in drawn}) == len(drawn)
    parked = sum(rules["pickup_parked"] for rules in drawn)
    assert 0 < parked < len(drawn)
    delays = [rules["pickup_delay_s"] for rules in drawn if not rules["pickup_parked"]]
    assert min(CONFIG["pickup_delay_s"]) <= min(delays) and max(delays) <= max(CONFIG["pickup_delay_s"])


def test_two_seeds_walk_the_herd_differently(walking_assets):
    """The same herd table walks every goat somewhere else once the seed changes."""
    tracks = []
    for seed in (SEED, SEED + 1):
        cli = p.connect(p.DIRECT)
        try:
            built = build_solar_map(seed, cli, asset_dir=walking_assets, groups=("movers",))
            build_solar_movers(built, seed=seed, cli=cli).advance(WALK_STEPS)
            tracks.append([p.getBasePositionAndOrientation(body, physicsClientId=cli)[0][:2]
                           for body in _by_item(built, "goat")])
        finally:
            p.disconnect(cli)
    assert all(np.linalg.norm(np.array(one) - np.array(other)) > 0.5 for one, other in zip(*tracks))
