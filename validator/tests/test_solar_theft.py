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

"""Theft scenarios (task 19): two seeds in five, crews of one to five, the same theft for the same seed, about a third
of crews at the fence from the start and the rest walking up later from out of the scene, crews keeping together,
spreading out once in or coming in two groups, everyone in within ENTRY_S of his group's arrival, nobody inside a
table, the four reactions to the drone once a man hears it, how far he hears it with his own ears and in the seed's
wind, night and direction, never to a picture taken from further, a coil carried out only once seen, runners who
flee unseen, and the fence really cut in the engine."""

from __future__ import annotations

import contextlib
import io
import math
import os
from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pytest
import swarm_worlds
from shapely import contains_xy
from shapely.geometry import Polygon
from shapely.ops import unary_union

from swarm.challenge_families import build_benchmark_tasks
from swarm.challenge_families.solar_patrol import camera, intruders, theft, theft_moves, theft_site, wind
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

# The theft seeds the scenes below were written against, kept fixed so a change of THEFT_SHARE leaves them alone. The
# crews of 4, 8, 29 and 51 are at the fence from the start; those of 27, 32 and 40 walk up later. The crews of 8 and 40
# spread out once in, and that of 29 comes in two groups.
THEFT_SEEDS = [4, 8, 27, 29, 32, 40, 51, 52]
# A crew of three at the fence from the start that spreads out, its lookout among them, and one that comes later through
# the gate in two groups, the second the lookout alone.
LOOSE_SEED = 150
SPLIT_SEED = 183
# A tight crew of four at the fence from the start and a loose crew of five that walks up later, found by searching,
# kept fixed so a change of CREW leaves them alone.
FOUR_MAN_SEED = 14
FIVE_MAN_SEED = 515
PACE_COPY_SEED = 273                    # a crew of two whose follower takes the first man's pace to keep apart


@lru_cache(maxsize=10)
def _site(seed: int) -> theft_site.Site:
    """A seed's park as a thief sees it, built once per test run."""
    return theft_site.Site(seed, builder.SOLAR_ASSET_DIR)


def _crew(seed: int, reaction: str = "", **first):
    """A seed's story and crew, the first man's reaction and any other field of his overridden when asked."""
    site = _site(seed)
    story = theft.write(seed, site)
    if reaction:
        story.thieves[0].reaction = reaction
        story.thieves[0].bolts_unseen = False
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


def test_two_seeds_in_five_have_a_theft():
    """The share of seeds with intruders is two in five over many seeds."""
    share = np.mean([theft.has_theft(s) for s in range(10000)])
    assert 0.39 <= share <= 0.41


def test_the_scene_seeds_all_have_a_theft():
    """Every seed the scenes below are written against still draws a theft."""
    assert all(theft.has_theft(s) for s in THEFT_SEEDS)


def test_crews_come_in_sizes_one_to_five_by_their_weights():
    """Over many theft seeds every crew size from one to five comes about as often as CREW says."""
    seeds = [s for s in range(6000) if theft.has_theft(s)]
    sizes = []
    for seed in seeds:
        rng = np.random.default_rng([theft.THEFT_SEED_STREAM, seed])
        rng.random()
        sizes.append(theft._pick(rng, theft.CREW))
    assert set(sizes) == set(theft.CREW) == {1, 2, 3, 4, 5}
    for size, weight in theft.CREW.items():
        assert abs(sizes.count(size) / len(sizes) - weight) <= 0.03, size


def test_the_single_point_fence_check_matches_the_batch_one():
    """The per-step inside-the-fence check answers exactly as the batch check, on points spread over and around the
    park and on the fence's own corners."""
    site = _site(THEFT_SEEDS[0])
    low, high = site.ring.min(0) - 5.0, site.ring.max(0) + 5.0
    points = np.vstack([np.random.default_rng(0).uniform(low, high, (4000, 2)), site.ring])
    assert [site.holds(x, y) for x, y in points] == theft_site.inside(site.ring, points).tolist()


def test_a_third_of_crews_are_at_the_fence_from_the_start_and_the_rest_come_later():
    """Over many seeds ARRIVE_EARLY_SHARE of crews are at the fence when the patrol starts, and the others reach it at
    times spread over the whole of ARRIVE_S."""
    arrivals = np.array([theft.arrival(s) for s in range(10000)])
    early = arrivals == 0.0
    assert abs(early.mean() - theft.ARRIVE_EARLY_SHARE) < 0.015
    later = arrivals[~early]
    assert theft.ARRIVE_S[0] <= later.min() < theft.ARRIVE_S[0] + 2.0
    assert theft.ARRIVE_S[1] - 2.0 < later.max() <= theft.ARRIVE_S[1]
    assert abs(np.median(later) - sum(theft.ARRIVE_S) / 2.0) < 5.0


def test_the_arrival_moves_no_other_draw_of_the_theft(monkeypatch):
    """A crew that comes later is the crew the seed writes for the start, the time each worker shoulders his coil
    counted from the crew's arrival: the arrival has a stream of its own."""
    late = [seed for seed in THEFT_SEEDS if theft.arrival(seed) > 0.0]
    assert late
    for seed in late:
        story = theft.write(seed, _site(seed))
        monkeypatch.setattr(theft, "arrival", lambda seed: 0.0)
        early = theft.write(seed, _site(seed))
        monkeypatch.undo()
        assert story.arrive_s == theft.arrival(seed) and early.arrive_s == 0.0
        assert (story.way, story.opening.index, story.cut_along, story.cut_width, story.hinge, story.open_deg,
                story.cycles) == (early.way, early.opening.index, early.cut_along, early.cut_width, early.hinge,
                                  early.open_deg, early.cycles)
        for x, y in zip(story.thieves, early.thieves):
            if x.role == "worker":
                assert x.carry_after_s == pytest.approx(y.carry_after_s + story.arrive_s)
            y.carry_after_s = x.carry_after_s
            assert x == y, seed


def test_crews_keep_together_spread_out_or_come_in_two_groups_in_their_shares():
    """A lone man has no style; over many seeds crews at the fence from the start take each style in its CREW_STYLES
    share, and a split crew's second group always reaches the fence SPLIT_LATER_S after the first and by the end of
    ARRIVE_S, so a crew that comes too late for that comes in loose."""
    assert theft.crew_style(7, 1) == ("tight", 0.0)
    drawn = [(theft.arrival(s), *theft.crew_style(s, 2)) for s in range(5000)]
    early = [style for arrive_s, style, _ in drawn if arrive_s == 0.0]
    for style, share in theft.CREW_STYLES.items():
        assert abs(early.count(style) / len(early) - share) < 0.035, style
    for arrive_s, style, second_s in drawn:
        if style == "split":
            assert theft.SPLIT_LATER_S[0] <= second_s - arrive_s <= theft.SPLIT_LATER_S[1], (arrive_s, second_s)
            assert second_s <= theft.ARRIVE_S[1]
        else:
            assert second_s == 0.0
    late = [style for arrive_s, style, _ in drawn if arrive_s > theft.ARRIVE_S[1] - theft.SPLIT_LATER_S[0]]
    assert late and "split" not in late


@lru_cache(maxsize=4)
def _styled(style: str = "") -> dict:
    """The thefts of the first hundred seeds, written on one park with a crew of three each, every crew taking one style
    or, given none, its own; each worker takes only the first stretch of his table, so a hundred stories write fast."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(theft, "CREW", {3: 1.0})
        patch.setattr(theft, "_spots", lambda rng, site, table, tables, reachable, entry:
                      theft._spots_of(site, table, reachable)[:1])
        if style:
            patch.setattr(theft, "CREW_STYLES", {style: 1.0})
        site = _site(THEFT_SEEDS[0])
        return {seed: theft.write(seed, site) for seed in range(100) if theft.has_theft(seed)}


def _lookout_depth(story, thief, depth) -> bool:
    """Whether a lookout stands depth inside the hole, or just inside it where the park has no free ground that deep."""
    inside = np.dot(np.array(thief.lookout) - story.hole, story.inward)
    return bool(depth[0] - 1e-9 <= inside <= depth[1] + 1e-9 or abs(inside - (theft.INSIDE_M + 1.0)) < 1e-6)


def _clear_stop(site: theft_site.Site, story, slot: int, side_m: float) -> bool:
    """Whether a follower stopping side_m to the side stands on free ground he can walk to straight from inside the
    hole."""
    stop = theft._stop(story, slot, side_m)
    return site.free(stop) and site.clear_line(story.hole + story.inward * 2.0, stop)


def test_a_tight_crew_is_written_as_before_and_the_style_moves_no_earlier_draw():
    """A tight crew's men stop STOP_SIDE_M further over each, nobody comes later and its lookout stands LOOKOUT_M inside
    the hole, and it is exactly the crew its seed writes when every crew is tight; a loose or split crew keeps the way
    in, the arrival and its first man exactly as a tight one would have them."""
    real, tight = _styled(), _styled("tight")
    for seed, story in real.items():
        if story.style == "tight":
            assert story == tight[seed], seed
            assert story.second_s == 0.0
            for n, thief in enumerate(story.thieves):
                assert (thief.side_m, thief.late_s) == (theft.STOP_SIDE_M * n, 0.0), seed
                if thief.role == "lookout":
                    assert _lookout_depth(story, thief, theft.LOOKOUT_M), seed
            continue
        same = tight[seed]
        assert (story.way, story.opening.index, story.cut_along, story.cut_width, story.hinge, story.open_deg,
                story.arrive_s) == (same.way, same.opening.index, same.cut_along, same.cut_width, same.hinge,
                                    same.open_deg, same.arrive_s)
        assert story.thieves[0] == same.thieves[0], seed
    assert {story.style for story in real.values()} == set(theft.CREW_STYLES)


def test_a_loose_crew_spreads_out_once_in():
    """Over many seeds most men of a loose crew stop LOOSE_SIDE_M further to the side than the man before, on clear
    ground; one is pulled in only where the ground half a metre further over is not clear, and never nearer the hole
    than a tight crew's man; its workers work anywhere in the park, far more often beyond NEAR_ROWS of the first man's
    row than a tight crew's; and its lookout stands LOOSE_LOOKOUT_M inside the hole."""
    loose, tight = _styled("loose"), _styled("tight")
    site = _site(THEFT_SEEDS[0])
    table_of = {spot.xy: k for k, spots in site.__dict__["_spots"].items() for spot in spots}

    def far_share(stories: dict) -> float:
        """The share of the workers after the first whose table lies beyond NEAR_ROWS of the first man's row."""
        far = []
        for story in stories.values():
            row = site.tables[table_of[story.thieves[0].spots[0].xy]].row
            far += [abs(site.tables[table_of[thief.spots[0].xy]].row - row) > theft.NEAR_ROWS
                    for thief in story.thieves[1:] if thief.role == "worker"]
        return float(np.mean(far))

    gaps, lookouts = [], 0
    for seed, story in loose.items():
        assert story.style == "loose"
        for n, (ahead, thief) in enumerate(zip(story.thieves, story.thieves[1:]), 1):
            gaps.append(thief.side_m - ahead.side_m)
            assert thief.side_m - ahead.side_m <= theft.LOOSE_SIDE_M[1] + 1e-9, seed
            assert thief.side_m >= theft.STOP_SIDE_M * n, seed
            if thief.side_m > theft.STOP_SIDE_M * n:
                assert _clear_stop(site, story, n, thief.side_m), (seed, n)
                if gaps[-1] < theft.LOOSE_SIDE_M[0]:
                    assert not _clear_stop(site, story, n, thief.side_m + 0.5), (seed, n)
            if thief.role == "lookout":
                lookouts += 1
                assert _lookout_depth(story, thief, theft.LOOSE_LOOKOUT_M), seed
    assert len(gaps) >= 40 and lookouts
    assert np.mean(np.array(gaps) >= theft.LOOSE_SIDE_M[0] - 1e-9) >= 0.7
    assert far_share(loose) >= far_share(tight) + 0.2


def test_a_split_crew_comes_in_two_groups():
    """A split crew's first man cuts the way in at the crew's arrival, while the second group is out of the scene; the
    second group is still away at the first group's arrival, stands at its places outside by its own, never cuts, and
    steps in within ENTRY_S of it."""
    story, site, men = _crew(SPLIT_SEED)
    assert story.style == "split" and story.first_group == 2 and len(men) == 3
    assert story.second_s - story.arrive_s >= theft.SPLIT_LATER_S[0]
    assert men[0].actor.at(story.arrive_s + 0.1).stage == "at the fence"
    later = men[2]
    assert later.thief.late_s >= story.second_s - story.arrive_s
    assert later.actor.at(story.arrive_s + 0.1).index == theft_moves.GONE
    due = story.arrive_s + later.thief.late_s
    assert later.actor.at(due - 0.1).stage == "approach"
    assert later.actor.at(due + 0.1).stage == "at the fence"
    stages = {later.actor.at(k * SIM_DT).stage for k in range(0, int((story.second_s + theft.ENTRY_S) / SIM_DT), 10)}
    assert "cutting the fence" not in stages
    frame = later.actor.at(story.second_s + theft.ENTRY_S)
    assert theft_site.inside(site.ring, np.array([frame.x, frame.y]))[0]


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
def test_a_crew_cuts_in_on_the_side_its_way_names():
    """A crew coming from the road always cuts a panel on the road side and one coming from the woods a panel against
    the woods, the hole as wide as the rules allow and clear of the posts, on many draws of one park and in every
    seed's story."""
    site = _site(THEFT_SEEDS[0])
    rng = np.random.default_rng(0)
    for way in ("road", "forest"):
        for _ in range(200):
            opening, along, width = theft._way_in(rng, site, way)
            assert getattr(opening, way), way
            assert theft.HOLE_WIDTH_M[0] <= width <= theft.HOLE_WIDTH_M[1]
            assert abs(along) + width / 2.0 + theft.POST_CLEAR_M <= opening.half_width + 1e-9
    for seed in THEFT_SEEDS:
        story = theft.write(seed, _site(seed))
        if story.way in ("road", "forest"):
            assert getattr(story.opening, story.way), (seed, story.way, story.opening.index)


@pytest.mark.time_budget(45)
def test_every_crew_is_in_within_entry_s_of_its_arrival_and_never_inside_a_table():
    """Played for a whole patrol with no drone, a crew at the fence from the start stands at its places from the first
    frame, a man who comes later is out of the scene until he walks up from at least OUTSIDE_M beyond the fence and a
    late crew's first man starts at the fence at the crew's arrival, every man steps inside within ENTRY_S of his
    group's arrival and never before, so by 236 s, never stands inside a table, nobody leaves, since no camera ever saw
    him, no two men ever walk into each other, and the crews come in all three ways, early and late, tight, loose and
    split."""
    ways, arrivals, styles = set(), set(), set()
    for seed in THEFT_SEEDS + [LOOSE_SEED, SPLIT_SEED]:
        story, site, men = _crew(seed)
        ways.add(story.way)
        arrivals.add(story.arrive_s > 0.0)
        styles.add(story.style)
        assert 1 <= len(men) <= 5
        tables = unary_union([Polygon(table.corners(-0.05)) for table in site.tables])
        if story.arrive_s > 0.0:
            first = men[0].actor
            assert first.at(story.arrive_s - 0.1).stage == "approach", seed
            assert first.at(story.arrive_s + 0.1).stage == "at the fence", seed
        for n, man in enumerate(men):
            if not story.walks_up(n):
                start = man.actor.at(0.0)
                assert ((start.x, start.y), start.stage) == (man.thief.start, "at the fence"), seed
            entered, shown, track = None, None, []
            for k in range(0, int(390 / SIM_DT), 10):
                frame = man.actor.at(k * SIM_DT)
                here = np.array([frame.x, frame.y])
                if shown is None:
                    if frame.index == theft_moves.GONE:
                        assert story.walks_up(n), (seed, k * SIM_DT)
                        continue
                    shown = k * SIM_DT
                    if story.walks_up(n):
                        assert np.dot(story.hole - here, story.inward) >= theft.OUTSIDE_M, seed
                assert frame.index != theft_moves.GONE, (seed, k * SIM_DT)
                if entered is None and theft_site.inside(site.ring, here)[0]:
                    entered = k * SIM_DT
                track.append(here)
            track = np.array(track)
            assert not contains_xy(tables, track[:, 0], track[:, 1]).any(), seed
            due = story.due_s(n)
            assert entered is not None and due < entered <= due + theft.ENTRY_S, (seed, n, entered)
            assert entered <= theft.ARRIVE_S[1] + theft.ENTRY_S, (seed, n, entered)
        for k in range(0, int(390 / SIM_DT), 5):
            spots = [man.actor.at(k * SIM_DT) for man in men]
            for i in range(len(spots)):
                for j in range(i + 1, len(spots)):
                    if theft_moves.GONE in (spots[i].index, spots[j].index):
                        continue
                    gap = math.hypot(spots[i].x - spots[j].x, spots[i].y - spots[j].y)
                    assert gap >= theft.SPACING_M, (seed, k * SIM_DT, i, j, gap)
    assert ways == {"road", "gate", "forest"} and arrivals == {False, True}
    assert styles == set(theft.CREW_STYLES)


def test_a_crew_of_four_and_a_crew_of_five_settle_in_and_stand_apart():
    """A four-man and a five-man seed have every man in within ENTRY_S of his group's arrival, never within SPACING_M
    of each other, and every follower's stop is on free ground inside the fence."""
    for seed, size in ((FOUR_MAN_SEED, 4), (FIVE_MAN_SEED, 5)):
        story, site, men = _crew(seed)
        assert len(men) == size
        for slot in range(1, size):
            stop = theft._stop_spot(site, story, slot)
            assert site.free(stop) and site.holds(*stop), (seed, slot)
        for n, man in enumerate(men):
            entered = next(k * SIM_DT for k in range(0, int(390 / SIM_DT), 10)
                           if theft_site.inside(site.ring, np.array([man.actor.at(k * SIM_DT).x,
                                                                     man.actor.at(k * SIM_DT).y]))[0])
            assert entered <= story.due_s(n) + theft.ENTRY_S, (seed, n, entered)
        assert theft._first_clash([theft._actor(story, site, n) for n in range(size)]) is None, seed


def _entered(actor: theft_moves.Actor, site: theft_site.Site, until: float) -> float:
    """When a man first stands inside the fence by patrol time until, or infinity if he never does."""
    for k in range(0, int(until / SIM_DT) + 1, 5):
        frame = actor.at(k * SIM_DT)
        if theft_site.inside(site.ring, np.array([frame.x, frame.y]))[0]:
            return k * SIM_DT
    return math.inf


def test_a_crew_at_the_slowest_or_fastest_pace_goes_through_in_turn_by_entry_s():
    """The entry timetable allows for each man's pace: a crew all at the slowest or all at the fastest pace steps
    through one after another, nobody before the first man has opened the way, and each is in within ENTRY_S of his
    group's arrival."""
    for seed in THEFT_SEEDS[:4]:
        site = _site(seed)
        for pace in theft.PACE:
            story = theft.write(seed, site)
            for thief in story.thieves:
                thief.pace = pace
            theft._schedule(np.random.default_rng(seed), story)
            due = [story.due_s(n) + theft.ENTRY_S for n in range(len(story.thieves))]
            entered = [_entered(theft._actor(story, site, n), site, due[n]) for n in range(len(story.thieves))]
            assert entered == sorted(entered) and all(e <= d for e, d in zip(entered, due)), (seed, pace, entered)


def test_a_follower_who_takes_the_pace_ahead_still_steps_in_by_entry_s():
    """A follower given the pace of the man ahead, so they keep apart on a later walk, still steps in within ENTRY_S
    of his group's arrival."""
    site = _site(PACE_COPY_SEED)
    drawn = [thief.pace for thief in theft.write(PACE_COPY_SEED, site).thieves]
    story, site, men = _crew(PACE_COPY_SEED)
    assert story.thieves[1].pace == story.thieves[0].pace != drawn[1]
    due = [story.due_s(n) + theft.ENTRY_S for n in range(len(men))]
    assert all(_entered(man.actor, site, d) <= d for man, d in zip(men, due))


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


def test_a_step_along_a_route_never_carries_the_body_past_its_end():
    """A block steered along a route travels only as far as the route still goes: a walk shorter than its own first
    step and its stop ends on the mark at the slowest, plain and fastest pace, never a step past it."""
    lib = theft_moves.library()
    mark = np.array([0.3, 0.0])
    for pace in (theft.PACE[0], 1.0, theft.PACE[1]):
        route = theft_moves.Route([(0.0, 0.0), mark])

        def script(route=route):
            """Start walking and stop at once."""
            yield theft_moves.Step("stand_to_walk", steer=route)
            yield theft_moves.Step("walk_to_stand", steer=route)

        actor = theft_moves.Actor(lib, (0.0, 0.0), theft_moves.heading_to((1.0, 0.0)), pace=pace, script=script())
        actor.context = {"hub": "stand"}
        actor.run_to(400)
        assert np.linalg.norm(actor.pos - mark) < 0.1, (pace, actor.pos)


def _hover(monkeypatch, seed: int, reaction: str, docked: bool = False, dock_z: float = 0.0, pictures: bool = True,
           near: tuple = (120.0, 390.0), away_m=0.0, seen_from: float = 0.0, weather: tuple = ("light", False),
           **first):
    """Over flat ground, keep the drone 200 m up, and between the near times fly it 20 m up and away_m to one side of
    the first man, or away_m(t) when it is a function of patrol time, its camera showing him twice a second from
    seen_from on when pictures is set; return his frames and the episode. dock_z stands the dock that high, as it is up
    the park's slope, and weather is the seed's wind strength and whether it is night."""
    monkeypatch.setattr(theft, "_ground", lambda env, ep, x, y: 0.0)
    story, site, men = _crew(seed, reaction, **first)
    ep = _episode(seed, story, site, men)
    ep.theft.weather = weather
    ep.dock_position = np.array([0.0, 0.0, dock_z])
    ep.phase = "docked" if docked else "flying"
    first_man = men[0].actor
    for step in range(int(390 / SIM_DT)):
        ep.step = step
        t = (step + 1) * SIM_DT
        frame = first_man.at(t)
        over = near[0] <= t < near[1]
        side = away_m(t) if callable(away_m) else away_m
        _Drone.pos = np.array([[frame.x + side, frame.y, 20.0] if over else [frame.x, frame.y, 200.0]])
        if step % camera.FRAME_STEPS == 0:
            ep.camera = {"view": _picture(ep, {0: 12} if over and pictures and t >= seen_from else {})}
        theft.advance(_Drone, ep)
    return [first_man.at(k * SIM_DT) for k in range(int(390 / SIM_DT))], ep, site, story


def _away_for(distance_m: float) -> float:
    """How far to one side a drone 20 m up stands to be distance_m from a standing man's ears on flat ground."""
    return math.sqrt(distance_m ** 2 - (20.0 - theft.EAR_M) ** 2)


def _crossing(frames, site) -> np.ndarray:
    """Where a man who left last crossed the fence on his way out."""
    out = [np.array([f.x, f.y]) for f in frames if f.index != theft_moves.GONE]
    k = next(k for k in range(len(out) - 1, 0, -1) if theft_site.inside(site.ring, out[k - 1])[0]
             and not theft_site.inside(site.ring, out[k])[0])
    return out[k]


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("reaction", ["work_on", "freeze", "hide", "run"])
def test_each_reaction_plays_out_once_he_hears_the_drone(monkeypatch, reaction):
    """Nobody reacts to a drone 200 m up; once it flies over him and its camera shows him, a man working on never
    reacts, a freezer goes back to work, a hidden man lies flat inside or gets up again, and a runner leaves through
    the hole the crew cut."""
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
        assert np.linalg.norm(_crossing(frames, site) - story.hole) < 1.5
    assert all(man.entered for man in ep.theft.men)


@pytest.mark.timeout(1200)
def test_a_zoom_from_far_away_that_shows_him_sets_nobody_off(monkeypatch):
    """A drone 150 m off, its zoom showing him twice a second for the rest of the patrol, sets nobody off: he is seen,
    and works on as if no drone were there."""
    frames, ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", away_m=150.0)
    assert ep.theft.men[0].seen_at is not None
    assert not any(f.stage.startswith("reaction") for f in frames)
    assert frames[-1].index != theft_moves.GONE


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("margin_m, reacts", [(-1.0, True), (1.0, False)])
def test_he_hears_the_drone_only_within_his_own_distance(monkeypatch, margin_m, reacts):
    """A drone hovering a metre inside a man's own hearing distance sets him off; one a metre beyond it, its camera
    showing him all the while, never does."""
    away = _away_for(25.0 + margin_m)
    frames, _ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "freeze", away_m=away, hear_m=25.0)
    assert any(f.stage.startswith("reaction") for f in frames) == reacts


def test_every_man_hears_from_his_own_distance_in_the_base_range():
    """Every man of every scene seed is written with a base hearing distance inside the range for the weather to
    scale."""
    for seed in THEFT_SEEDS:
        story = theft.write(seed, _site(seed))
        assert all(theft.HEAR_M[0] <= thief.hear_m <= theft.HEAR_M[1] for thief in story.thieves)


@pytest.mark.parametrize("strength, night, expected", [
    ("none", False, 30.0), ("light", False, 24.0), ("strong", False, 15.6),
    ("none", True, 36.0), ("light", True, 28.8), ("strong", True, 18.72)])
def test_wind_and_night_scale_how_far_a_man_hears(strength, night, expected):
    """Still air lets the drone carry 1.25 times as far, strong wind masks it to 0.65 times, and the night's quiet adds
    a fifth, for a drone straight overhead where direction does not matter."""
    assert theft.hearing_m(24.0, strength, night, np.array([1.0, 0.0]), np.array([0.0, 0.0, -20.0])) == \
        pytest.approx(expected)


@pytest.mark.parametrize("to_man, expected", [
    ((30.0, 0.0, 0.0), 30.0), ((-30.0, 0.0, 0.0), 18.0), ((0.0, 30.0, 0.0), 24.0), ((30.0, 0.0, -18.4), 30.0)])
def test_a_man_downwind_of_the_drone_hears_it_further(to_man, expected):
    """With the wind blowing east, a man east of the drone hears it 1.25 times as far, one west of it 0.75 times, one
    to its side as in light air, whatever his height below it."""
    assert theft.hearing_m(24.0, "light", False, np.array([1.0, 0.0]), np.array(to_man)) == pytest.approx(expected)


def test_still_air_has_no_downwind():
    """In still air a man hears the same from every side, even if a wind vector were handed in."""
    reach = {theft.hearing_m(24.0, "none", False, np.array([1.0, 0.0]), np.array(v)) for v in
             ((30.0, 0.0, 0.0), (-30.0, 0.0, 0.0), (0.0, 30.0, 0.0))}
    assert reach == {30.0}


def test_nobody_hears_past_the_cap():
    """A man whose ears and weather would carry the drone past the cap hears it from the cap and no further."""
    assert theft.hearing_m(50.0, "none", True, np.array([1.0, 0.0]), np.array([30.0, 0.0, 0.0])) == theft.HEAR_CAP_M
    assert theft.hearing_m(theft.HEAR_M[1], "light", True, np.array([1.0, 0.0]),
                           np.array([30.0, 0.0, 0.0])) <= theft.HEAR_CAP_M


def test_the_sounds_carry_direction_is_the_seeds_steady_wind():
    """The direction the sound is carried is the unit ground direction of the environment's seeded steady wind, and
    zero in still air."""
    env = type("Env", (), {"_wind": type("Blowing", (), {"steady": np.array([0.0, -3.0, 0.0])})()})()
    assert wind.blowing_toward(env).tolist() == [0.0, -1.0]
    assert wind.blowing_toward(type("Env", (), {"_wind": None})()).tolist() == [0.0, 0.0]


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("weather, reacts", [(("none", True), True), (("strong", False), False)])
def test_a_man_who_hears_it_on_a_calm_night_misses_it_in_strong_wind(monkeypatch, weather, reacts):
    """A drone 35 m from a man whose own distance is 25 m sets him off on a still night, which carries it to 37.5 m,
    and never in strong wind, which masks it to 16 m."""
    away = _away_for(35.0)
    frames, _ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "freeze", away_m=away, hear_m=25.0,
                                        weather=weather)
    assert any(f.stage.startswith("reaction") for f in frames) == reacts


@pytest.mark.timeout(1200)
def test_the_theft_takes_the_seeds_weather():
    """A patrol's theft carries the seed's own wind strength and night, the ones the wind part reads."""
    seed = THEFT_SEEDS[2]
    task = build_benchmark_tasks(sim_dt=SIM_DT, seeds=[seed], family_id=FAMILY_ID)[0]
    with contextlib.redirect_stdout(io.StringIO()):
        env, _ = make_env_with_initial_obs(task)
    try:
        assert env._solar.theft.weather == (wind.strength(seed), wind.is_night(seed))
    finally:
        env.close()


@pytest.mark.timeout(1200)
def test_a_runner_who_hears_it_unseen_lies_flat_until_the_camera_sees_him(monkeypatch):
    """A runner who waits to be seen and hears the drone overhead before its camera has shown him freezes, drops flat
    and stays inside; once the camera shows him he gets up and runs out through the hole."""
    frames, ep, site, story = _hover(monkeypatch, THEFT_SEEDS[1], "run", seen_from=200.0)
    stages = [f.stage for f in frames]
    unseen = stages[int(120 / SIM_DT):int(200 / SIM_DT)]
    assert "reaction: freeze" in unseen and "reaction: hide" in unseen
    assert unseen.index("reaction: freeze") < unseen.index("reaction: hide")
    assert all(f.index != theft_moves.GONE for f in frames[:int(200 / SIM_DT)])
    assert 200.0 <= ep.theft.men[0].seen_at
    assert "reaction: run" in stages[int(200 / SIM_DT):]
    assert frames[-1].index == theft_moves.GONE
    assert np.linalg.norm(_crossing(frames, site) - story.hole) < 1.5


@pytest.mark.timeout(1200)
def test_a_runner_never_seen_gets_back_to_work_once_the_drone_has_gone(monkeypatch):
    """A runner who hears the drone for a minute and is never shown lies flat while it is near, gets up once it has
    gone, and never leaves: not at a run, and not with the coil his time to carry has come for."""
    frames, ep, site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", pictures=False, near=(120.0, 180.0),
                                      carry_after_s=60.0)
    stages = [f.stage for f in frames]
    assert "reaction: hide" in stages
    assert not stages[-1].startswith("reaction")
    assert all(f.index != theft_moves.GONE for f in frames)
    assert theft_site.inside(site.ring, np.array([frames[-1].x, frames[-1].y]))[0]
    assert ep.theft.men[0].seen_at is None


def test_a_share_of_the_runners_bolts_unseen():
    """Over many men, about UNSEEN_RUN_SHARE of them are drawn to flee unseen, the same for the same seed and man."""
    draws = [theft._bolts(seed, n) for seed in range(3000) for n in range(3)]
    assert abs(np.mean(draws) - theft.UNSEEN_RUN_SHARE) < 0.03
    assert theft._bolts(7, 1) == theft._bolts(7, 1)


def test_the_story_stores_who_bolts_unseen():
    """The story gives each man the draw of his own seed and place in the crew."""
    story = theft.write(THEFT_SEEDS[1], _site(THEFT_SEEDS[1]))
    expected = [theft._bolts(THEFT_SEEDS[1], n) for n in range(len(story.thieves))]
    assert [t.bolts_unseen for t in story.thieves] == expected


@pytest.mark.timeout(1200)
def test_a_runner_who_bolts_leaves_on_hearing_the_drone_before_any_picture(monkeypatch):
    """A runner who bolts and hears a drone no picture of his has shown runs out through the hole and is gone from
    the scene, never seen."""
    frames, ep, site, story = _hover(monkeypatch, THEFT_SEEDS[1], "run", pictures=False, near=(120.0, 180.0),
                                     bolts_unseen=True)
    stages = [f.stage for f in frames]
    assert "reaction: run" in stages and "reaction: hide" not in stages
    assert frames[-1].index == theft_moves.GONE
    assert np.linalg.norm(_crossing(frames, site) - story.hole) < 1.5
    assert ep.theft.men[0].seen_at is None


@pytest.mark.timeout(1200)
def test_a_runner_seen_while_he_freezes_runs_without_dropping_flat(monkeypatch):
    """A runner the camera first shows while he is still frozen runs from where he stands: he never drops flat first."""
    frames, _ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", seen_from=1e9)
    froze = next(k for k, f in enumerate(frames) if f.stage == "reaction: freeze") * SIM_DT
    frames, ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", seen_from=froze + 1.0)
    stages = [f.stage for f in frames]
    assert ep.theft.men[0].seen_at < froze + theft_moves.library().seconds("walk_freeze_resume.0")
    assert "reaction: hide" not in stages
    assert "reaction: run" in stages and frames[-1].index == theft_moves.GONE


@pytest.mark.timeout(1200)
def test_a_flat_runner_seen_from_beyond_his_hearing_stays_flat(monkeypatch):
    """A runner lying flat whom the camera first shows from 30 m, past his 25 m hearing distance, stays flat: he runs
    only once he hears the drone again."""
    away = _away_for(30.0)
    frames, ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", seen_from=150.0, hear_m=25.0,
                                      away_m=lambda t: 0.0 if t < 150.0 else away)
    stages = [f.stage for f in frames]
    assert 150.0 <= ep.theft.men[0].seen_at
    assert "reaction: hide" in stages and "reaction: run" not in stages
    assert frames[-1].index != theft_moves.GONE


@pytest.mark.timeout(1200)
@pytest.mark.parametrize("shown", [True, False])
def test_a_runner_with_no_way_back_up_lies_flat_until_seen(monkeypatch, shown):
    """A runner whose loop has no way back to it from lying flat, who hears the drone unseen, lies flat; once the
    camera shows him he runs out, and unseen he lies flat for good, long after the drone has gone."""
    monkeypatch.setattr(theft, "_BACK_TO", {})
    frames, ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", pictures=shown, near=(120.0, 260.0),
                                      seen_from=200.0)
    stages = [f.stage for f in frames]
    flat = stages[int(135 / SIM_DT):int(200 / SIM_DT)]
    assert flat and all(s == "reaction: hide" for s in flat)
    if shown:
        assert "reaction: run" in stages and frames[-1].index == theft_moves.GONE
    else:
        assert stages[-1] == "reaction: hide" and ep.theft.men[0].seen_at is None


def test_a_drone_below_the_dock_over_low_ground_still_sets_him_off(monkeypatch):
    """The park rises 39 m: a drone 20 m over a man on low ground can fly below its dock, and he still hears it."""
    frames, _ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", dock_z=39.0)
    assert any(f.stage.startswith("reaction") for f in frames)


def test_nobody_reacts_to_a_drone_still_in_its_dock(monkeypatch):
    """A drone that never takes off never sets anyone off, even when its pictures show him."""
    frames, _ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", docked=True)
    assert not any(f.stage.startswith("reaction") for f in frames)


def test_a_reaction_due_after_the_drone_has_gone_is_dropped(monkeypatch):
    """A man who hears the drone for one second, whose reaction is due only after it has flown out of earshot, never
    reacts."""
    frames, _ep, _site, _story = _hover(monkeypatch, THEFT_SEEDS[1], "run", near=(120.0, 121.0), react_delay_s=10.0)
    assert not any(f.stage.startswith("reaction") for f in frames)


def test_a_worker_carries_his_coil_out_only_once_the_camera_has_seen_him(monkeypatch):
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
    ep.step = int((story.arrive_s + theft.ENTRY_S) / SIM_DT)
    wide = int(theft.SEEN_PX)
    for shows, sees, expected in (({0: wide}, True, {0}), ({0: wide - 2}, True, set()), ({0: wide}, False, set())):
        ep.camera = {"view": _picture(ep, shows, sees)}
        assert theft._shown(ep, ep.theft) == expected
    assert theft._shown(ep, ep.theft) == set()


def test_a_man_away_comes_into_the_scene_once_his_wait_is_over():
    """A body kept away is out of the scene for its whole wait, then walks in, and is never taken for one who left."""
    lib = theft_moves.library()

    def script():
        """Wait away two seconds, then walk two strides."""
        yield theft_moves.Step("away", hold=2.0, stage="away")
        yield theft_moves.Step("stand_to_walk")
        yield theft_moves.Step("loop", cycles=2)

    actor = theft_moves.Actor(lib, (0.0, 0.0), 0.0, script=script())
    actor.context = {"hub": "stand"}
    actor.run_to(int(2 * lib.fps) + 1)
    assert not actor.gone
    assert all(f.index == theft_moves.GONE for f in actor.frames[:int(2 * lib.fps)])
    assert actor.frames[-1].index != theft_moves.GONE
    actor.run_to(int(10 * lib.fps))
    assert np.linalg.norm(actor.pos) > 1.0


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
    drop flat, and a man lying flat can get up again, walking or at a run."""
    lib = theft_moves.library()
    for kind in ("walk", "walk_wary", "sneak", "carry", "cut_fence", "cut_cable", "pull_cable", "look_around",
                 "crouch_watch"):
        takes = [name for name, block in lib.blocks.items() if block["loop"] and block["kind"] == kind]
        assert takes, kind
        for take in takes:
            follows = lib.kinds({"loop": take})
            for way in ("freeze", "hide"):
                assert any(pattern.format(kind) in follows for pattern in theft._REACT[way]), (take, way)
    assert {"prone_to_stand", "prone_to_run"} <= set(lib.kinds({"hub": "prone"}))


REACTION_SEED = 3                       # its first man walks along the fence and between tables on his way in
WALK_AHEAD_S = 6.0                      # a reaction tried this long before a walk ends still starts in the walk
FENCE_FREEZE_SEED = 2320772054          # where a thief was seen to freeze his way out through the fence and back
FENCE_FREEZE_AT_S = 152.0               # its first man walking towards the fence a few metres before his route turns


def _react_on_the_way(story, site, n: int, t: float, reaction: str) -> list:
    """Man n of a planned crew made to react his given way at patrol time t, as when he hears the drone then: his
    frames over the next twelve seconds."""
    actor = theft._actor(story, site, n)
    man = theft.Man(replace(story.thieves[n], reaction=reaction, bolts_unseen=False), actor,
                    np.random.default_rng([theft.THEFT_SEED_STREAM, 0, n]))
    actor.at(t)
    man.about = man.hears = True
    man.noticed_at = t
    actor.interrupt(lambda a: theft._react(a, man, SimpleNamespace(story=story, site=site), t))
    return [actor.at(t + k * SIM_DT) for k in range(int(12.0 / SIM_DT))]


def test_a_man_who_freezes_or_drops_flat_on_his_way_keeps_to_his_path():
    """A man who freezes or drops flat while walking keeps to the path he was on, its turns and its end, instead of
    carrying on straight: at every moment of his walks, nothing in the twelve seconds after stands outside the fence
    or inside a table."""
    story, site, _men = _crew(REACTION_SEED)
    tables = unary_union([Polygon(table.corners(-0.05)) for table in site.tables])
    dry = theft._actor(story, site, 0)
    tried = 0
    for t in np.arange(2.0, 120.0, 4.0):
        now, ahead = dry.at(t), dry.at(t + WALK_AHEAD_S)
        if now.stage not in ("arrival", "moving on") or ahead.stage != now.stage or not site.holds(now.x, now.y):
            continue
        for reaction in ("freeze", "hide"):
            frames = _react_on_the_way(story, site, 0, float(t), reaction)
            track = np.array([[f.x, f.y] for f in frames])
            assert all(site.holds(x, y) for x, y in track), (t, reaction)
            assert not contains_xy(tables, track[:, 0], track[:, 1]).any(), (t, reaction)
            tried += 1
    assert tried >= 6


def test_the_man_seen_freezing_through_the_fence_stays_inside():
    """On the seed where a thief was seen to freeze his way out through the fence and back in, the first man frozen as
    he walks towards the fence keeps to his path and stays inside the park."""
    story, site, _men = _crew(FENCE_FREEZE_SEED)
    frames = _react_on_the_way(story, site, 0, FENCE_FREEZE_AT_S, "freeze")
    assert "reaction: freeze" in {f.stage for f in frames}
    assert all(site.holds(f.x, f.y) for f in frames)


def test_an_actor_keeps_what_steers_the_block_it_plays():
    """The steer of the block being played is kept on the actor, so a reaction can keep to the same path; a block
    played with none clears it."""
    lib = theft_moves.library()
    route = theft_moves.Route([(0.0, 0.0), (20.0, 0.0)])

    def script():
        """Walk a few strides along the route, then stop with no steer."""
        yield theft_moves.Step("stand_to_walk", steer=route)
        yield theft_moves.Step("loop", cycles=2, steer=route)
        yield theft_moves.Step("walk_to_stand")

    actor = theft_moves.Actor(lib, (0.0, 0.0), theft_moves.heading_to((1.0, 0.0)), script=script())
    actor.context = {"hub": "stand"}
    actor.run_to(int(1.5 * lib.fps))
    assert actor.steer is route
    actor.run_to(int(20 * lib.fps))
    assert actor.steer is None


def test_threats_are_the_men_inside_the_fence():
    """At the start every man waits outside and is no threat; once in, each one's bodies are threats."""
    story, site, men = _crew(THEFT_SEEDS[0])
    ep = _episode(THEFT_SEEDS[0], story, site, men)
    for man in men:
        man.body = type("Body", (), {"bodies": [id(man)]})()
    ep.step = 0
    assert not theft.threat_bodies(ep) and all(not theft.inside(ep, n, 0) for n in range(len(men)))
    ep.step = int((story.arrive_s + theft.ENTRY_S) / SIM_DT)
    assert theft.threat_bodies(ep) == {id(man) for man in men}
    assert all(theft.inside(ep, n, ep.step) for n in range(len(men)))
    assert theft.bodies(ep) == {id(man): n for n, man in enumerate(men)}


@pytest.mark.time_budget(35)
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


@pytest.mark.time_budget(40)
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


@pytest.mark.time_budget(60)
@pytest.mark.timeout(1200)
def test_the_fence_is_cut_and_the_thief_posed_in_the_engine():
    """In a real patrol of a crew that comes later the men wait out of the scene, the panel or gate stands whole until
    the crew has reached it and gives way to its cut or opened pieces once the first man is through, and a picture of
    a thief poses him where he stands."""
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

        story = ep.theft.story
        assert story.arrive_s > 0.0
        picture(0.0, story.hole)
        assert all(man.posed is None for man in ep.theft.men)
        assert p.getBasePositionAndOrientation(panel, physicsClientId=env.CLIENT)[0][2] > -100.0
        picture(story.arrive_s, story.hole)
        assert p.getBasePositionAndOrientation(panel, physicsClientId=env.CLIENT)[0][2] > -100.0
        picture(story.arrive_s + theft.ENTRY_S, story.hole)
        assert p.getBasePositionAndOrientation(panel, physicsClientId=env.CLIENT)[0][2] < -100.0
        man = ep.theft.men[0]
        frame = man.actor.at(story.arrive_s + 60.0)
        picture(story.arrive_s + 60.0, np.array([frame.x, frame.y]))
        assert math.dist(man.posed[:2], (frame.x, frame.y)) < 1e-6
    finally:
        env.close()
