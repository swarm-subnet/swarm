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

"""Theft scenarios (task 19): one seed in five, the same theft for the same seed, everyone in before the take-off is
over, nobody inside a table, the four reactions to the drone once its camera shows a man (task 50), nobody leaving
unseen, and the fence really cut in the engine."""

from __future__ import annotations

import contextlib
import io
import math
import os
from functools import lru_cache

import numpy as np
import pybullet as p
import pytest
import swarm_worlds
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union

from swarm.challenge_families import build_benchmark_tasks
from swarm.challenge_families.solar_patrol import camera, intruders, theft, theft_moves, theft_site
from swarm.challenge_families.solar_patrol.contract import FAMILY_ID
from swarm.challenge_families.solar_patrol.episode import SolarEpisode
from swarm.constants import SIM_DT
from swarm.core.maps.solar import builder
from swarm.utils.env_factory import make_env_with_initial_obs

MAPS = swarm_worlds.maps_dir()
pytestmark = pytest.mark.skipif(
    not (os.path.isfile(os.path.join(MAPS, intruders.ASSET_DIR, "intruders.json"))
         and os.path.isfile(os.path.join(MAPS, theft_moves.ASSET_DIR, "motions.json"))),
    reason="the installed swarm-worlds has no intruders or motion library yet")

THEFT_SEEDS = [s for s in range(200) if theft.has_theft(s)][:8]


@lru_cache(maxsize=8)
def _site(seed: int) -> theft_site.Site:
    """A seed's park as a thief sees it, built once per test run."""
    return theft_site.Site(seed, builder.SOLAR_ASSET_DIR)


def _crew(seed: int, reaction: str = "", **first):
    """A seed's story and crew, the first man's reaction and any other field of his overridden when asked."""
    site = _site(seed)
    story = theft.write(seed, site)
    if reaction:
        story.thieves[0].reaction = reaction
    for name, value in first.items():
        setattr(story.thieves[0], name, value)
    return story, site, theft.cast(seed, story, site)


def _picture(ep: SolarEpisode, shows: dict, sees: bool = True) -> camera.View:
    """A wide frame taken at the episode's step whose object map shows each man given, n -> pixels across, as an
    upright block of his body id; every man gets a stand-in body first."""
    for n, man in enumerate(ep.theft.men):
        if man.body is None:
            man.body = type("Body", (), {"bodies": [1001 + n]})()
    objects = np.full((480, 640), -1, dtype=np.int32)
    for n, across in shows.items():
        objects[100:160, 40 * n:40 * n + across] = ep.theft.men[n].body.bodies[0]
    return camera.View("colour", (0.0, 0.0, 20.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0), 640, 480, 60.0, sees, ep.step,
                       objects)


class _Drone:
    """Stands in for the environment: where the drone is."""

    pos = np.zeros((1, 3))


def _episode(seed: int, story, site, men) -> SolarEpisode:
    """A patrol state carrying a theft, with the dock at the park's first free cell and the drone flying."""
    ep = SolarEpisode(seed=seed)
    ep.dock_position = np.zeros(3)
    ep.phase = "flying"
    ep.theft = theft.Theft(story, site, men)
    return ep


def test_one_seed_in_five_has_a_theft():
    """The share of seeds with intruders is one in five over many seeds."""
    share = np.mean([theft.has_theft(s) for s in range(10000)])
    assert 0.19 <= share <= 0.21


def test_the_single_point_fence_check_matches_the_batch_one():
    """The per-step inside-the-fence check answers exactly as the batch check, on points spread over and around the
    park and on the fence's own corners."""
    site = _site(THEFT_SEEDS[0])
    low, high = site.ring.min(0) - 5.0, site.ring.max(0) + 5.0
    points = np.vstack([np.random.default_rng(0).uniform(low, high, (4000, 2)), site.ring])
    assert [site.holds(x, y) for x, y in points] == theft_site.inside(site.ring, points).tolist()


def test_the_same_seed_writes_the_same_theft():
    """Two draws of one seed agree on the way in, the hole and every man's dress, role, spots and reaction."""
    seed = THEFT_SEEDS[0]
    a = theft.write(seed, theft_site.Site(seed, builder.SOLAR_ASSET_DIR))
    b = theft.write(seed, _site(seed))
    assert (a.way, a.opening.index, a.cut_along, a.cut_width, a.hinge, a.cycles) == \
        (b.way, b.opening.index, b.cut_along, b.cut_width, b.hinge, b.cycles)
    for x, y in zip(a.thieves, b.thieves):
        assert (x.role, x.dress, x.start, x.reaction, x.hear_m, x.spots) == (y.role, y.dress, y.start, y.reaction,
                                                                            y.hear_m, y.spots)


@pytest.mark.timeout(1200)
def test_every_crew_is_in_before_the_take_off_is_over_and_never_inside_a_table():
    """Played for a whole patrol with no drone, every man steps inside the fence by ENTRY_S, never stands inside a
    table, nobody leaves, since no camera ever saw him, no two men ever walk into each other, and the crews come in all
    three ways."""
    ways = set()
    for seed in THEFT_SEEDS:
        story, site, men = _crew(seed)
        ways.add(story.way)
        assert 1 <= len(men) <= 3
        tables = unary_union([Polygon(table.corners(-0.05)) for table in site.tables])
        for man in men:
            entered, track = None, []
            for k in range(0, int(390 / SIM_DT), 10):
                frame = man.actor.at(k * SIM_DT)
                assert frame.index != theft_moves.GONE, (seed, k * SIM_DT)
                here = np.array([frame.x, frame.y])
                if entered is None and theft_site.inside(site.ring, here)[0]:
                    entered = k * SIM_DT
                track.append(here)
            track = np.array(track)
            assert not contains_xy(tables, track[:, 0], track[:, 1]).any(), seed
            assert entered is not None and entered <= theft.ENTRY_S, (seed, entered)
        for k in range(0, int(390 / SIM_DT), 5):
            spots = [man.actor.at(k * SIM_DT) for man in men]
            for i in range(len(spots)):
                for j in range(i + 1, len(spots)):
                    if theft_moves.GONE in (spots[i].index, spots[j].index):
                        continue
                    gap = math.hypot(spots[i].x - spots[j].x, spots[i].y - spots[j].y)
                    assert gap >= theft.SPACING_M, (seed, k * SIM_DT, i, j, gap)
    assert ways == {"road", "gate", "forest"}


def test_a_walk_stops_on_its_mark():
    """A walk of six metres or more ends within ten centimetres of its mark."""
    lib = theft_moves.library()
    for distance in (6.0, 11.0, 23.0):
        mark = np.array([distance, 2.0])
        route = theft_moves.Route([mark])

        def script(route=route):
            """Start walking, walk to the mark, stop."""
            yield theft_moves.Step("stand_to_walk", steer=route)
            yield theft_moves.Step("loop", cycles=None, steer=route, then="walk_to_stand")
            yield theft_moves.Step("walk_to_stand", steer=route)

        actor = theft_moves.Actor(lib, (0.0, 0.0), 0.5, script=script())
        actor.context = {"hub": "stand"}
        actor.run_to(4000)
        assert np.linalg.norm(actor.pos - mark) < 0.1


def _hover(monkeypatch, seed: int, reaction: str, docked: bool = False, dock_z: float = 0.0, pictures: bool = True,
           near: tuple = (120.0, 390.0), **first):
    """Over flat ground, keep the drone 200 m up, and between the near times fly it 20 m straight over the first man,
    its camera showing him twice a second when pictures is set; return his frames and the episode. dock_z stands the
    dock that high, as it is up the park's slope."""
    monkeypatch.setattr(theft, "_ground", lambda env, ep, x, y: 0.0)
    story, site, men = _crew(seed, reaction, **first)
    ep = _episode(seed, story, site, men)
    ep.dock_position = np.array([0.0, 0.0, dock_z])
    ep.phase = "docked" if docked else "flying"
    first_man = men[0].actor
    for step in range(int(390 / SIM_DT)):
        ep.step = step
        t = (step + 1) * SIM_DT
        frame = first_man.at(t)
        over = near[0] <= t < near[1]
        _Drone.pos = np.array([[frame.x, frame.y, 20.0 if over else 200.0]])
        if step % camera.FRAME_STEPS == 0:
            ep.camera = {"view": _picture(ep, {0: 12} if over and pictures else {})}
        theft.advance(_Drone, ep)
    return [first_man.at(k * SIM_DT) for k in range(int(390 / SIM_DT))], ep, site, story


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("reaction", ["work_on", "freeze", "hide", "run"])
def test_each_reaction_plays_out_once_the_camera_shows_him(monkeypatch, reaction):
    """Nobody reacts to a drone 200 m up; once its camera shows him, a man working on never reacts, a freezer goes back
    to work, a hidden man lies flat inside or gets up again, and a runner leaves through the hole the crew cut."""
    frames, ep, site, story = _hover(monkeypatch, THEFT_SEEDS[1], reaction)
    stages = [f.stage for f in frames]
    early = stages[:int(120 / SIM_DT)]
    assert not any(s.startswith("reaction") for s in early)
    late = stages[int(120 / SIM_DT):]
    if reaction == "work_on":
        assert not any(s.startswith("reaction") for s in late)
    elif reaction == "freeze":
        k = next(k for k, s in enumerate(late) if s == "reaction: freeze")
        assert any(not s.startswith("reaction") for s in late[k:])
    elif reaction == "hide":
        k = next(k for k, s in enumerate(late) if s == "reaction: hide")
        assert all(s == "reaction: hide" for s in late[k:]) or "prone_to_stand" in theft_moves.library().kinds(
            {"hub": "prone"})
        last = frames[-1]
        assert theft_site.inside(site.ring, np.array([last.x, last.y]))[0]
    else:
        assert frames[-1].index == theft_moves.GONE
        out = [np.array([f.x, f.y]) for f in frames if f.index != theft_moves.GONE]
        crossing = next(k for k in range(len(out) - 1, 0, -1) if theft_site.inside(site.ring, out[k - 1])[0]
                        and not theft_site.inside(site.ring, out[k])[0])
        assert np.linalg.norm(out[crossing] - story.hole) < 1.5
    assert all(man.entered for man in ep.theft.men)


def test_a_drone_below_the_dock_over_low_ground_still_sets_him_off(monkeypatch):
    """The park rises 39 m: a drone 20 m over a man on low ground can fly below its dock, and he still reacts once its
    camera shows him."""
    frames, _ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", dock_z=39.0)
    assert any(f.stage.startswith("reaction") for f in frames)


def test_nobody_reacts_to_a_drone_still_in_its_dock(monkeypatch):
    """A drone that never takes off never sets anyone off, even when its pictures show him."""
    frames, _ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", docked=True)
    assert not any(f.stage.startswith("reaction") for f in frames)


def test_a_near_drone_whose_camera_never_shows_him_sets_nobody_off(monkeypatch):
    """A drone 20 m over a man for the rest of the patrol, its camera never showing him, sets nobody off and lets
    nobody leave: he reacts to its camera, not to how near it is."""
    frames, ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", pictures=False, carry_after_s=60.0)
    assert not any(f.stage.startswith("reaction") for f in frames)
    assert all(f.index != theft_moves.GONE for f in frames)
    assert all(man.seen_at is None for man in ep.theft.men)


def test_a_reaction_due_after_the_drone_has_gone_is_dropped(monkeypatch):
    """A man shown for one second, whose reaction is due only after the drone has left him unseen for longer than
    SEEN_QUIET_S and flown out of earshot, never reacts."""
    delay = theft.SEEN_QUIET_S + 10.0
    frames, ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", near=(120.0, 121.0), react_delay_s=delay)
    assert ep.theft.men[0].seen_at is not None
    assert not any(f.stage.startswith("reaction") for f in frames)


def test_nobody_leaves_before_the_camera_has_seen_him(monkeypatch):
    """A worker whose time to carry the coil out came at 60 s stays until the camera shows him at 120 s, then carries
    it out."""
    frames, ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "work_on", carry_after_s=60.0)
    seen = ep.theft.men[0].seen_at
    gone = next(k for k, f in enumerate(frames) if f.index == theft_moves.GONE) * SIM_DT
    assert 120.0 <= seen < gone
    assert "carrying" in {f.stage for f in frames}


def test_a_picture_shows_a_man_only_when_it_can():
    """A man counts as seen from a picture that shows him inside the fence at least SEEN_PX across, once per picture,
    never from a thinner shape, a dark frame, or a picture taken while he still stood outside."""
    story, site, men = _crew(THEFT_SEEDS[0])
    ep = _episode(THEFT_SEEDS[0], story, site, men)
    ep.step = 0
    ep.camera = {"view": _picture(ep, {0: 12})}
    assert theft._shown(ep, ep.theft) == set()
    ep.step = int(theft.ENTRY_S / SIM_DT)
    wide = int(theft.SEEN_PX)
    for shows, sees, expected in (({0: wide}, True, {0}), ({0: wide - 2}, True, set()), ({0: wide}, False, set())):
        ep.camera = {"view": _picture(ep, shows, sees)}
        assert theft._shown(ep, ep.theft) == expected
    assert theft._shown(ep, ep.theft) == set()


def test_a_man_who_has_left_never_comes_back():
    """A reaction still waiting when a man's track takes him out of the scene never brings him back into it."""
    lib = theft_moves.library()

    def script():
        """Walk two strides and leave."""
        yield theft_moves.Step("stand_to_walk")
        yield theft_moves.Step("loop", cycles=2)
        yield theft_moves.Step("gone")

    def freeze(actor):
        """Freeze where he stands."""
        return iter([theft_moves.Step("walk_freeze", stage="reaction: freeze")])

    actor = theft_moves.Actor(lib, (0.0, 0.0), 0.0, script=script())
    actor.context = {"hub": "stand"}
    while not actor.gone:
        actor.run_to(len(actor.frames) + 1)
    actor.interrupt(freeze)
    gone = len(actor.frames)
    actor.run_to(gone + 300)
    assert all(f.index == theft_moves.GONE for f in actor.frames[gone - 1:])


def test_every_loop_a_thief_is_in_can_freeze_and_drop_flat():
    """Every take of every loop a thief can be in while the drone is about has a freeze that goes back to it and a
    drop flat, and a man lying flat can get up again."""
    lib = theft_moves.library()
    for kind in ("walk", "walk_wary", "sneak", "carry", "cut_fence", "cut_cable", "pull_cable", "look_around",
                 "crouch_watch"):
        takes = [name for name, block in lib.blocks.items() if block["loop"] and block["kind"] == kind]
        assert takes, kind
        for take in takes:
            follows = lib.kinds({"loop": take})
            for way in ("freeze", "hide"):
                assert any(pattern.format(kind) in follows for pattern in theft._REACT[way]), (take, way)
    assert "prone_to_stand" in lib.kinds({"hub": "prone"})


def test_threats_are_the_men_inside_the_fence():
    """At the start every man waits outside and is no threat; once in, each one's bodies are threats."""
    story, site, men = _crew(THEFT_SEEDS[0])
    ep = _episode(THEFT_SEEDS[0], story, site, men)
    for man in men:
        man.body = type("Body", (), {"bodies": [id(man)]})()
    ep.step = 0
    assert not theft.threat_bodies(ep) and all(not theft.inside(ep, n, 0) for n in range(len(men)))
    ep.step = int(theft.ENTRY_S / SIM_DT)
    assert theft.threat_bodies(ep) == {id(man) for man in men}
    assert all(theft.inside(ep, n, ep.step) for n in range(len(men)))
    assert theft.bodies(ep) == {id(man): n for n, man in enumerate(men)}


@pytest.mark.timeout(1200)
def test_the_first_man_is_not_played_ahead_at_reset():
    """Staging the theft times the fence cut on a copy of the first man, so his own track is written no further than
    the patrol clock and a reaction can take over at his first seam."""
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[THEFT_SEEDS[2]], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, _ = make_env_with_initial_obs(task)
    try:
        assert env._solar.theft.men[0].actor.time_s < 10.0
    finally:
        env.close()


@pytest.mark.timeout(1200)
def test_every_thief_counts_as_a_threat_from_the_start():
    """A patrol that ends before the crew is through the fence still answers for every man of the story."""
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[THEFT_SEEDS[2]], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, _ = make_env_with_initial_obs(task)
    try:
        ep = env._solar
        assert not theft.threat_bodies(ep)
        assert ep.outcome.threats == len(ep.theft.men) > 0
    finally:
        env.close()


@pytest.mark.timeout(1200)
def test_the_fence_is_cut_and_the_thief_posed_in_the_engine():
    """In a real patrol the panel or gate stands whole at the start and gives way to its cut or opened pieces once the
    first man is through, and a picture of a thief poses him where he stands."""
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[THEFT_SEEDS[2]], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, _ = make_env_with_initial_obs(task)
    try:
        ep = env._solar
        prop = ep.theft.props[0]
        panel = prop.panel if isinstance(prop, theft.FenceHole) else prop.gate

        def picture(t: float, at: np.ndarray) -> None:
            """Pose everything for a picture of a ground point at patrol time t."""
            eye = np.array([at[0] - 8.0, at[1] - 8.0, float(ep.dock_position[2]) + 20.0])
            forward = np.append(at, float(ep.dock_position[2])) - eye
            view = camera.View("colour", tuple(eye), tuple(forward / np.linalg.norm(forward)), (0.0, 0.0, 1.0), 640,
                               480, 50.0, True, int(round(t / SIM_DT)))
            theft.show(env, ep, view)

        picture(0.0, ep.theft.story.hole)
        assert p.getBasePositionAndOrientation(panel, physicsClientId=env.CLIENT)[0][2] > -100.0
        picture(theft.ENTRY_S, ep.theft.story.hole)
        assert p.getBasePositionAndOrientation(panel, physicsClientId=env.CLIENT)[0][2] < -100.0
        man = ep.theft.men[0]
        frame = man.actor.at(60.0)
        picture(60.0, np.array([frame.x, frame.y]))
        assert math.dist(man.posed[:2], (frame.x, frame.y)) < 1e-6
    finally:
        env.close()
