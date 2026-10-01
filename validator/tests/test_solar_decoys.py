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

"""Solar Patrol decoys: the open ground the dogs live on, their planned patrols, when they are posed, and the tags
the report check reads.

The ground and the planner run on a flat stand-in world and stand-in clips, so they run everywhere. The dogs'
bodies ship in swarm-worlds, so the tests that pose one are skipped on an installed release without them.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest

from swarm.challenge_families.solar_patrol import camera, decoys
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.constants import SIM_DT

FENCE = np.array([[0.0, 0.0], [60.0, 0.0], [60.0, 40.0], [0.0, 40.0]])
DOCK = np.array([52.0, 34.0])
TRUNK = np.array([8.0, 30.0])
# Three panel rows across the middle, each 40 m long and 3 m deep, leaving aisles between them.
ROWS = [(30.0, 10.0), (30.0, 18.0), (30.0, 26.0)]
CLIPS = {
    "walk": {"start": 0, "frames": 25, "fps": 30, "loop": True, "speed_m_s": 0.74},
    "trot": {"start": 25, "frames": 12, "fps": 30, "loop": True, "speed_m_s": 1.72},
    "run": {"start": 37, "frames": 13, "fps": 36, "loop": True, "speed_m_s": 3.73},
    "stand": {"start": 50, "frames": 80, "fps": 24, "loop": True, "speed_m_s": 0.0},
    "look": {"start": 130, "frames": 80, "fps": 24, "loop": True, "speed_m_s": 0.0},
    "sniff": {"start": 210, "frames": 96, "fps": 24, "loop": True, "speed_m_s": 0.0},
    "nose_down": {"start": 306, "frames": 64, "fps": 24, "loop": True, "speed_m_s": 0.0},
    "sniff_walk": {"start": 370, "frames": 99, "fps": 24, "loop": True, "speed_m_s": 0.45},
    "lie_down": {"start": 469, "frames": 41, "fps": 24, "loop": False, "speed_m_s": 0.0, "then": "lying"},
    "lying": {"start": 510, "frames": 96, "fps": 24, "loop": True, "speed_m_s": 0.0},
    "get_up": {"start": 606, "frames": 29, "fps": 24, "loop": False, "speed_m_s": 0.0, "then": "stand"},
}
MOVING = {"walk", "trot", "run", "sniff_walk"}
HORIZON_S = 400.0
BLEND_S = 0.25

needs_dogs = pytest.mark.skipif(not decoys.DOGS_SHIPPED, reason=f"the installed swarm-worlds has no {decoys.DOGS_DIR}")


@pytest.fixture(scope="module")
def ground():
    """The open ground of a flat 60 x 40 m park with three panel rows, a dock, a tree trunk and a walled pocket."""
    cli = p.connect(p.DIRECT)
    try:
        slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[100.0, 100.0, 0.5], physicsClientId=cli)
        terrain = p.createMultiBody(0, slab, -1, [30.0, 20.0, 1.5], physicsClientId=cli)
        panel = p.createCollisionShape(p.GEOM_BOX, halfExtents=[20.0, 1.5, 0.05], physicsClientId=cli)
        for x, y in ROWS:
            p.createMultiBody(0, panel, -1, [x, y, 3.2], physicsClientId=cli)
        # A pocket walled off on all four sides, 3 x 3 m inside: open to the sky, too small to live in.
        wall = p.createCollisionShape(p.GEOM_BOX, halfExtents=[2.5, 0.25, 1.0], physicsClientId=cli)
        side = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.25, 2.5, 1.0], physicsClientId=cli)
        for shape, (x, y) in ((wall, (8.0, 3.0)), (wall, (8.0, 7.5)), (side, (5.75, 5.25)), (side, (10.25, 5.25))):
            p.createMultiBody(0, shape, -1, [x, y, 3.0], physicsClientId=cli)
        yield decoys.open_ground(cli, FENCE, frozenset({terrain}), DOCK, TRUNK[None, :])
    finally:
        p.disconnect(cli)


def _plan(ground, mood, seed, size=1.0, pace=1.0):
    """One dog's patrol on the stand-in ground, as a Dog with no body."""
    rng = np.random.default_rng(seed)
    start = decoys._start(rng, ground, np.argwhere(ground.planned), [])
    legs = decoys.Planner(rng, ground, CLIPS, size, pace, BLEND_S).plan(mood, start, HORIZON_S)
    return decoys.Dog(body=-1, breed="husky", coat="agouti", size=size, pace=pace, mood=mood, length=0.95 * size,
                      blend=BLEND_S, legs=legs)


def _track(dog, until=390.0):
    """Where the dog is at every control step of a patrol: east, north, heading."""
    return np.array([decoys.locate(dog, t)[:3] for t in np.arange(0.0, until, SIM_DT)])


def _time_in(dog, clip, until=390.0):
    """Seconds of the patrol a dog spends in one clip."""
    ends = [leg.t0 for leg in dog.legs[1:]] + [math.inf]
    return sum(max(0.0, min(end, until) - leg.t0) for leg, end in zip(dog.legs, ends) if leg.clip == clip)


def test_the_open_ground_keeps_off_the_panels_the_fence_the_dock_and_the_trunk(ground):
    """Cells under a panel, near the fence, round the dock or the trunk are closed; the aisles are open and flat."""
    for x, y in ROWS:
        assert not ground.open_at(np.array([[x, y], [x - 19.0, y], [x + 19.0, y + 1.4]])).any()
    assert not ground.open_at(np.array([[0.5, 20.0], [59.5, 20.0], [30.0, 0.5]])).any()
    assert not ground.open_at(DOCK + np.array([[0.0, 0.0], [5.5, 0.0], [0.0, -5.5]])).any()
    assert not ground.open_at(TRUNK[None, :]).any()
    aisles = np.array([[30.0, 14.0], [30.0, 22.0], [5.0, 20.0], [55.0, 14.0]])
    assert ground.open_at(aisles).all()
    assert ground.height_at(aisles) == pytest.approx([2.0] * len(aisles))


def test_a_pocket_too_small_to_live_in_is_dropped(ground):
    """Open sky inside a walled 3 x 3 m pocket is not ground a dog is ever put on."""
    assert not ground.open_at(np.array([[8.0, 5.25], [7.0, 4.5]])).any()


@pytest.mark.parametrize("mood", sorted(decoys.MOODS))
def test_every_planned_moment_stays_on_open_ground_without_a_jump(ground, mood):
    """Over many seeds, a dog is on open ground at every control step, and it never moves faster than it runs or
    turns faster than it steps round."""
    fastest = CLIPS["run"]["speed_m_s"] * 1.1 * 1.1
    for seed in range(25):
        dog = _plan(ground, mood, seed, size=1.08, pace=1.1)
        track = _track(dog)
        assert ground.open_at(track[:, :2]).all(), f"seed {seed}"
        assert np.hypot(*np.diff(track[:, :2], axis=0).T).max() <= fastest * SIM_DT + 1e-6
        turned = np.abs((np.diff(track[:, 2]) + math.pi) % (2.0 * math.pi) - math.pi)
        tightest = fastest / min(decoys.TURN_RADIUS_M)
        assert turned.max() <= max(tightest, decoys.PIVOT_RAD_S) * SIM_DT + 1e-6


def test_a_plan_is_the_seed_s_own(ground):
    """The same seed plans the same patrol to the last number, and another seed plans another one."""
    one, again, other = (_plan(ground, "roamer", seed).legs for seed in (3, 3, 4))
    assert one == again
    assert one != other


def test_the_clips_follow_how_a_dog_moves(ground):
    """Only the gaits travel, lying down is always followed by lying, getting up always follows lying, and stepping
    round on the spot is done in the walk."""
    for mood in decoys.MOODS:
        for seed in range(10):
            legs = _plan(ground, mood, seed).legs
            for leg, after in zip(legs, legs[1:]):
                if leg.v > 0.0:
                    assert leg.clip in MOVING
                if leg.spin:
                    assert leg.clip == "walk" and leg.v == leg.v_in == 0.0
                if leg.clip == "lie_down":
                    assert after.clip == "lying"
                if after.clip == "get_up":
                    assert leg.clip == "lying"


def test_each_mood_lives_the_way_it_is_named(ground):
    """Resters spend most of the patrol lying, roamers cover the most ground, sniffers stay near where they began."""
    lying, covered, strayed = {m: [] for m in decoys.MOODS}, {m: [] for m in decoys.MOODS}, []
    for mood in decoys.MOODS:
        for seed in range(12):
            dog = _plan(ground, mood, seed)
            track = _track(dog)
            lying[mood].append(_time_in(dog, "lying") / 390.0)
            covered[mood].append(np.hypot(*np.diff(track[:, :2], axis=0).T).sum())
            if mood == "sniffer":
                strayed.append(np.hypot(*(track[:, :2] - track[0, :2]).T).max())
    assert np.mean(lying["rester"]) > 0.5
    assert np.mean(covered["roamer"]) > 1.5 * max(np.mean(covered["sniffer"]), np.mean(covered["rester"]))
    assert max(strayed) <= decoys.HOME_RADIUS_M + 1.0


def test_distance_and_the_time_it_takes_are_inverse():
    """time_for undoes distance, whether the body speeds up, slows down or keeps its pace through the blend."""
    for v_in, v in ((0.0, 0.74), (1.7, 0.74), (0.74, 0.74), (3.7, 0.45)):
        for tau in (0.05, 0.2, 0.25, 1.0, 7.3):
            assert decoys.time_for(decoys.distance(tau, v_in, v, BLEND_S), v_in, v, BLEND_S) == pytest.approx(tau)


def test_a_frame_shows_what_is_in_front_of_it_and_nothing_else():
    """A dog under a camera looking straight down stands its true length on the frame; one behind it or off to
    the side is not shown at all."""
    shot = camera.View(feed="thermal", eye=(0.0, 0.0, 20.0), forward=(0.0, 0.0, -1.0), up=(0.0, 1.0, 0.0),
                       width=640, height=512, vertical_fov_deg=30.0, sees=True, step=0)
    focal = 256 / math.tan(math.radians(15.0))
    assert decoys._pixels(shot, np.array([0.0, 0.0, 0.0]), 1.0) == pytest.approx(focal / 20.0)
    assert decoys._pixels(shot, np.array([0.0, 0.0, 30.0]), 1.0) == 0.0
    assert decoys._pixels(shot, np.array([40.0, 0.0, 0.0]), 1.0) == 0.0


def _episode(cli, ground, dogs):
    """A patrol holding only what the decoys read: the park's movers, the dogs and their ground."""
    ep = SolarEpisode(seed=0)
    ep.park = {"world": {"movers": [({"mover": "pickup"}, 101), ({"mover": "bird"}, 102)]}}
    ep.decoys = {"dogs": dogs, "ground": ground}
    return ep, SimpleNamespace(CLIENT=cli)


def test_every_decoy_is_tagged_for_the_report_check(ground):
    """The truck's and the bird's bodies and every dog carry what they are, and none of them is left untagged."""
    dog = _plan(ground, "roamer", 1)
    dog.body = 103
    ep, _ = _episode(0, ground, [dog])
    assert decoys.bodies(ep) == {101: "truck", 102: "bird", 103: "dog"}


@needs_dogs
def test_the_plans_use_only_clips_every_dog_ships():
    """Every clip the planner can ask for is in every dog body the package ships, with the same looping."""
    for body in decoys._catalogue()["bodies"].values():
        for name, spec in CLIPS.items():
            assert body["clips"][name]["loop"] == spec["loop"]


@needs_dogs
def test_a_dog_is_posed_only_when_a_frame_would_show_the_change(ground, monkeypatch):
    """With frames coming, a dog is posed only when its new pose moves at least half a pixel on one of them: not
    when no frame looks at it, not when its pose is unchanged, not when the change is too small to see from far
    off; the same change seen up close is posed. With no frame named, every dog is posed."""
    cli = p.connect(p.DIRECT)
    try:
        rng = np.random.default_rng(5)
        draw = {"breed": "husky", "coat": "agouti", "size": 1.0, "pace": 1.0, "mood": "roamer",
                "legs": decoys.Planner(rng, ground, decoys._catalogue()["bodies"]["husky"]["clips"], 1.0, 1.0,
                                       BLEND_S).plan("roamer", np.array([30.0, 14.0]), HORIZON_S)}
        dog = decoys._spawn(cli, draw)
        ep, env = _episode(cli, ground, [dog])
        posed = []
        real = p.resetMeshData
        monkeypatch.setattr(decoys.p, "resetMeshData", lambda body, *a, **k: (posed.append(body), real(body, *a, **k)))

        def looking(step, height, forward=(0.0, 0.0, -1.0)):
            """A frame from straight above the dog at a step, or turned away from it."""
            x, y = decoys.positions(ep, step)[0, :2]
            return camera.View(feed="colour", eye=(x, y, height), forward=forward, up=(0.0, 1.0, 0.0), width=640,
                               height=480, vertical_fov_deg=60.0, sees=True, step=step)

        decoys.place(env, ep, 0)
        assert posed == [dog.body]
        moving = next(round(leg.t0 / SIM_DT) + 25 for leg in dog.legs if leg.clip == "walk" and leg.t0 > 2.0)
        for step, shot, expected in ((moving, looking(moving, 22.0, (0.0, 0.0, 1.0)), []),
                                     (moving, looking(moving, 22.0), [dog.body]),
                                     (moving, looking(moving, 22.0), []),
                                     (moving + 10, looking(moving + 10, 3000.0), []),
                                     (moving + 10, looking(moving + 10, 22.0), [dog.body])):
            posed.clear()
            decoys.place(env, ep, step, [shot])
            assert posed == expected, (step, shot.eye)
    finally:
        p.disconnect(cli)


@needs_dogs
def test_a_dog_stands_on_the_ground_and_shows_on_the_object_map_as_a_decoy(ground):
    """A posed dog's paws rest on the terrain, and a frame's object map finds the dog's own body under it, which
    the tags call a dog: a report boxed on it can never be a threat."""
    cli = p.connect(p.DIRECT)
    try:
        slab = p.createCollisionShape(p.GEOM_BOX, halfExtents=[100.0, 100.0, 0.5], physicsClientId=cli)
        p.createMultiBody(0, slab, -1, [30.0, 20.0, 1.5], physicsClientId=cli)
        rng = np.random.default_rng(2)
        clips = decoys._catalogue()["bodies"]["shepherd"]["clips"]
        draw = {"breed": "shepherd", "coat": "sable", "size": 1.0, "pace": 1.0, "mood": "sniffer",
                "legs": decoys.Planner(rng, ground, clips, 1.0, 1.0, BLEND_S).plan("sniffer", np.array([30.0, 14.0]),
                                                                                    HORIZON_S)}
        dog = decoys._spawn(cli, draw)
        ep, env = _episode(cli, ground, [dog])
        decoys.place(env, ep, 250)
        (x, y, z), _ = p.getBasePositionAndOrientation(dog.body, physicsClientId=cli)
        assert z == pytest.approx(2.0, abs=0.02)
        vertices = decoys.shape(dog, *decoys.locate(dog, 250 * SIM_DT)[3:])
        assert float(vertices[:, 2].min()) == pytest.approx(0.0, abs=0.03)
        view = p.computeViewMatrix([x, y, 22.0], [x, y, 2.0], [0.0, 1.0, 0.0])
        projection = p.computeProjectionMatrixFOV(10.0, 1.0, 0.1, 100.0)
        mask = np.asarray(p.getCameraImage(64, 64, view, projection, renderer=p.ER_TINY_RENDERER,
                                           physicsClientId=cli)[4]).reshape(64, 64)
        assert (mask == dog.body).sum() > 20
        assert decoys.bodies(ep)[dog.body] == "dog"
    finally:
        p.disconnect(cli)
