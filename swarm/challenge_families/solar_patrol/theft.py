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

"""Theft scenarios (task 19): the thieves one seed writes, what they do minute by minute, and who is a threat when.

One seed in five has a theft: one to three men on foot, already at the fence when the patrol starts. They come in one
of three ways, each where it happens on the real site: a hole cut on the side of the public road, the gate on that
road forced, or a hole cut where the woods reach the fence. The hole is really cut in the scene: a fence panel gives
way to the chain link either side of two slits, and the flap between them is pushed into the park and stays open; a
forced gate swings one leaf in. Everyone is inside before the drone's take-off is over.

Inside, the crew works for the whole patrol. A worker walks to a table anywhere in the park, deep rows as likely as
the edge, stoops under its edge and cuts the string cable, pulls it out along the row, and moves on along the rows;
late in the patrol he may shoulder the coil and carry it out the way he came. A lookout stays near the hole, and the
last man in sometimes turns back to widen the hole before he goes to work. Nobody leaves early unless he has heard the
drone close by, and a man inside the fence is a threat from the moment he steps in until he is out again.

Each thief hears the drone at his own distance, with his own delay, and reacts his own way at the next moment his
move allows: he works on, freezes until it passes (again and again if it hovers), throws himself flat and lies still
until it has been gone a while and then runs, or runs for the hole at once.

Every draw comes from the seed's own stream, the park as the seed shifts it and, once the patrol runs, where the
drone flies, so every validator sees the same theft for the same flight. Thieves are posed only for a picture that
can see them, and a picture always shows them where they are at the moment it is taken.
"""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from swarm.constants import SIM_DT
from swarm.core.maps.solar import builder

from . import intruders
from .episode import SolarEpisode
from .theft_moves import GONE, Actor, Lane, Route, Step, heading_to, library
from .theft_site import Opening, Site, Table, yaw_of
from .theft_site import inside as in_ring

THEFT_SEED_STREAM = 0x7E1F7             # the theft's own stream, so no other draw of the seed moves it
THEFT_SHARE = 0.2                       # decided: one seed in five has intruders
CREW = {1: 0.45, 2: 0.35, 3: 0.20}      # how many come; one alone is the most common on a real site
WAYS = {"road": 0.45, "gate": 0.25, "forest": 0.30}
LOOKOUT_SHARE = 0.4                     # of crews of two or more, one man keeps watch by the hole
HOLE_WORK_SHARE = 0.35                  # the last man in turns back to widen the hole before he goes to work
HOLE_WORK_S = (20.0, 90.0)
HOLE_DOCK_M = 30.0                      # no way in this near the dock: the take-off is never over the thieves
HOLE_WIDTH_M = (1.0, 1.3)              # narrower, a man's swinging arms catch the chain link either side
FLAP_OPEN_DEG = (100.0, 150.0)
FLAP_SPRING_DEG = 20.0                  # the flap springs out this far as the slits get longer
FLAP_PUSH_S = 0.7
GATE_OPEN_DEG = (55.0, 95.0)
FENCE_GAP_M = 1.05                      # a cutter's hips from the fence: hands just short of it, jaws in it
INSIDE_M = 3.0                          # the first stop inside
LEAD_M = 6.0                            # a man leaving comes at the gap straight for this long: a sprinter needs 3.4 m to
                                        # turn a right angle, and would cross the fence beside the gap
TURN_BACK_M = 6.5                       # how far in a man walks before he turns back to the hole
OUTSIDE_M = 14.0                        # how far out a man who leaves goes before he is out of the scene
ENTRY_S = 36.0                          # everyone steps in before the drone's 40 s take-off is over
MATE_GAP_S = (1.5, 4.0)
SPACING_M = 0.9                         # two men getting in never come nearer than this, hips to hips
SPACING_WAIT_S = 1.0                    # how much longer a follower waits each time he would come too near
STOP_SIDE_M = 1.2                       # each follower stops this much to one side of the man in before him
SPOT_GAP_M = 0.7                        # a worker's hips from the table's edge
SPOT_EDGE_M = 1.5                       # no spot this near a table's end
NEXT_SPOT_M = (7.0, 16.0)               # how far a worker moves on to the next stretch of cable
NEAR_ROWS = 2                           # a crew mate who stays near works within this many rows of the first man
NEAR_SHARE = 0.6
CUT_CABLE_S = (20.0, 60.0)
PULL_M = (5.0, 14.0)
CARRY_AFTER_S = (300.0, 560.0)          # when a worker shoulders the coil; past the patrol means he never does
PACE = (0.9, 1.1)
GAITS = {"walk": 0.5, "walk_wary": 0.3, "sneak": 0.2}
HEAR_M = (22.0, 32.0)                   # how near the drone is, through the air, when a thief reacts: 11 to 26 m across the
                                        # ground at the 20 m patrol height, near enough for its camera to see him first
EAR_M = 1.6                             # a standing man's ears above the ground
HEAR_DELAY_S = (0.3, 2.0)
QUIET_M = 10.0                          # the drone counts as gone once it is this much further than he hears
QUIET_S = (8.0, 30.0)                   # a hidden man waits this long after the drone has gone before he runs
REFREEZE_SHARE = 0.7                    # a frozen man who still hears the drone freezes again
REACTIONS = {"work_on": 0.2, "freeze": 0.4, "hide": 0.25, "run": 0.15}
AIRBORNE_M = 3.0                        # the drone is heard once it is this high above the dock
VIEW_MARGIN_M = 1.6                     # a body this near the edge of a picture is posed for it
AWAY_Z = -1000.0                        # where a body out of the scene is kept

# The blocks a loop may react with, best first, for each way of reacting. A loop with none of them carries on, and he
# reacts at the first seam that has one while the drone is still near: a man walking wary freezes once he stops.
_REACT = {
    "freeze": ("{}_freeze_resume",),
    "hide": ("{}_to_prone",),
    "run": ("{}_to_run", "{}_freeze_run"),
}
# How a man standing up again takes back what the drone interrupted: the blocks from standing into that loop.
_BACK_TO = {
    "walk": ("stand_to_walk",),
    "sneak": ("stand_to_crouch", "crouch_to_sneak"),
    "cut_cable": ("cut_cable_in",),
    "crouch_watch": ("stand_to_crouch", "crouch_watch_in"),
}


@dataclass
class Spot:
    """A stretch of cable a worker takes: where he stoops, the heading that faces the table, and where pulling the
    cable out along the row takes him."""

    xy: Tuple[float, float]
    face: float
    pull_to: Tuple[float, float]


@dataclass
class Thief:
    """One man of the crew as the seed writes him."""

    role: str                           # "worker" or "lookout"
    dress: Dict[str, Any]
    scale: float
    mirrored: bool
    pace: float
    start: Tuple[float, float]
    yaw: float
    gait: str
    hear_m: float
    hear_delay_s: float
    reaction: str
    quiet_s: float
    wait_s: float = 0.0                 # how long he waits outside before following the man ahead through
    spots: List[Spot] = field(default_factory=list)
    cut_s: List[float] = field(default_factory=list)
    carry_after_s: float = 1e9
    hole_work_s: float = 0.0
    lookout: Optional[Tuple[float, float]] = None
    watch: str = "look_around"          # a lookout stands and looks around, or crouches and watches


@dataclass
class Story:
    """One seed's theft: the way in, where the hole is cut or which gate leaf swings, and the crew."""

    way: str
    opening: Opening
    cut_along: float                    # the way through, metres along the opening from its middle
    cut_width: float
    hinge: int                          # which side of the hole, or which gate leaf, turns: -1 or +1
    open_deg: float
    thieves: List[Thief]
    cycles: int = 3                     # the first man's cutting cycles at the fence

    @property
    def hole(self) -> np.ndarray:
        """The middle of the way through, on the fence line."""
        return np.array(self.opening.centre) + np.array(self.opening.along) * self.cut_along

    @property
    def inward(self) -> np.ndarray:
        """The unit ground direction into the park through the opening."""
        return np.array(self.opening.inward)


def has_theft(seed: int) -> bool:
    """Whether a seed has intruders: one seed in five."""
    return bool(np.random.default_rng([THEFT_SEED_STREAM, int(seed)]).random() < THEFT_SHARE)


def _pick(rng: np.random.Generator, weights: Dict[Any, float]) -> Any:
    """One key drawn with its weight."""
    keys = list(weights)
    share = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=share / share.sum()))]


def write(seed: int, site: Site) -> Optional[Story]:
    """The theft a seed writes on its park, or None for the four seeds in five without one."""
    rng = np.random.default_rng([THEFT_SEED_STREAM, int(seed)])
    if rng.random() >= THEFT_SHARE:
        return None
    crew = int(_pick(rng, CREW))
    way = _pick(rng, WAYS)
    opening, along, width = _way_in(rng, site, way)
    hinge = int(rng.choice([-1, 1]))
    if opening.kind == "gate":
        # The crew walks through the middle of the leaf that swings in.
        along = hinge * opening.half_width / 2.0
    story = Story(way, opening, along, width, hinge,
                  float(rng.uniform(*(FLAP_OPEN_DEG if opening.kind == "panel" else GATE_OPEN_DEG))), [])
    hole, inward = story.hole, story.inward
    reachable = site.distances(hole + inward * INSIDE_M)[0]
    tables = [k for k in range(len(site.tables)) if _spots_of(site, k, reachable)]
    lookout = crew >= 2 and rng.random() < LOOKOUT_SHARE
    first_table = int(rng.choice(tables))
    cat = intruders.catalogue()
    for n in range(crew):
        role = "lookout" if lookout and n == crew - 1 else "worker"
        dress = intruders.outfit(rng)
        if (n == 0 or (n == crew - 1 and not lookout)) and "bolt_cutters" not in dress["garments"]:
            dress["garments"].append("bolt_cutters")
            dress["colours"].update({piece["file"]: piece["colours"][0] for piece in cat.pieces["bolt_cutters"]})
        side = 1.0 if n % 2 else -1.0
        at_fence = np.array(opening.centre) if opening.kind == "gate" and n == 0 else hole
        start = at_fence - inward * FENCE_GAP_M
        if n:
            start = start - inward * rng.uniform(2.0, 4.0) + np.array(opening.along) * side * rng.uniform(0.8, 1.8)
        thief = Thief(role=role, dress=dress, scale=float(cat.root_scale[cat.builds.index(dress["build"])]),
                      mirrored=bool(rng.random() < 0.5), pace=float(rng.uniform(*PACE)),
                      start=(float(start[0]), float(start[1])), yaw=heading_to(inward),
                      gait="sneak" if way == "forest" and rng.random() < 0.5 else _pick(rng, GAITS),
                      hear_m=float(rng.uniform(*HEAR_M)), hear_delay_s=float(rng.uniform(*HEAR_DELAY_S)),
                      reaction=_pick(rng, REACTIONS), quiet_s=float(rng.uniform(*QUIET_S)))
        if role == "lookout":
            thief.lookout = _lookout_spot(rng, site, hole, inward, reachable)
            thief.watch = "look_around" if rng.random() < 0.5 else "crouch_watch"
        else:
            table = first_table
            if n > 0:
                table = _near_table(rng, site, tables, first_table) if rng.random() < NEAR_SHARE else int(rng.choice(tables))
            thief.spots = _spots(rng, site, table, tables, reachable, hole + inward * INSIDE_M)
            thief.cut_s = [float(rng.uniform(*CUT_CABLE_S)) for _ in thief.spots]
            thief.carry_after_s = float(rng.uniform(*CARRY_AFTER_S))
            # Only the last man in: anyone still to come through would find him in the hole.
            if n == crew - 1 and rng.random() < HOLE_WORK_SHARE and _turn_back_clear(site, story):
                thief.hole_work_s = float(rng.uniform(*HOLE_WORK_S))
        story.thieves.append(thief)
    _schedule(rng, story)
    return story


def _way_in(rng: np.random.Generator, site: Site, way: str) -> Tuple[Opening, float, float]:
    """The opening a way in uses, where along it the hole is cut and how wide."""
    width = float(rng.uniform(*HOLE_WIDTH_M))
    if way == "gate":
        gates = [o for o in site.openings if o.kind == "gate" and _room(site, o, 0.0)]
        if gates:
            return gates[0], 0.0, 0.0
        way = "road"
    panels = [o for o in site.openings if o.kind == "panel" and _room(site, o, 0.0)]
    wanted = [o for o in panels if (o.forest if way == "forest" else not o.forest)] or panels
    opening = wanted[int(rng.integers(len(wanted)))]
    slack = max(0.0, opening.half_width - width / 2.0 - 0.15)
    return opening, float(rng.uniform(-slack, slack)), width


def _room(site: Site, opening: Opening, along: float) -> bool:
    """Whether a crew can use an opening: free ground just inside it, far enough from the dock."""
    hole = np.array(opening.centre) + np.array(opening.along) * along
    inward = np.array(opening.inward)
    if site.dock is not None and np.linalg.norm(hole - site.dock) < HOLE_DOCK_M:
        return False
    return site.clear_line(hole + inward * 1.5, hole + inward * LEAD_M)


def _turn_back_clear(site: Site, story: Story) -> bool:
    """Whether the last man in has room to walk in, turn and come back to the hole from inside."""
    hole, inward = story.hole, story.inward
    turn = hole + inward * TURN_BACK_M + np.array(story.opening.along) * 1.5
    return site.free(turn) and site.clear_line(hole + inward * INSIDE_M, turn) and \
        site.clear_line(turn, hole + inward * (FENCE_GAP_M + 1.0))


def _spots_of(site: Site, table: int, reachable: np.ndarray) -> List[Spot]:
    """Every stretch of a table's cable a worker can reach and pull out along the row, both sides, a metre apart."""
    cache = site.__dict__.setdefault("_spots", {})
    if table in cache:
        return cache[table]
    t: Table = site.tables[table]
    out = []
    c, u, v = np.array(t.centre), np.array(t.along), np.array(t.across)
    for side in (-1.0, 1.0):
        for s in np.arange(-t.half_length + SPOT_EDGE_M, t.half_length - SPOT_EDGE_M + 1e-9, 1.0):
            xy = c + u * s + v * side * (t.half_depth + SPOT_GAP_M)
            i, j = site.cell(xy)
            if not site.free(xy) or not np.isfinite(reachable[i * site.shape[1] + j]):
                continue
            for way in (-1.0, 1.0):
                if site.clear_line(xy, xy + u * way * PULL_M[0]):
                    end = xy + u * way * PULL_M[1]
                    out.append(Spot((float(xy[0]), float(xy[1])), heading_to(-v * side), (float(end[0]), float(end[1]))))
    cache[table] = out
    return out


def _spots(rng: np.random.Generator, site: Site, table: int, tables: List[int], reachable: np.ndarray,
           entry: np.ndarray) -> List[Spot]:
    """A worker's stretches of cable: the first on his table, each next one a short walk on along the rows from where
    his pull really ended, each pulled only as far as the line along the row stays clear. Every walk is at least
    NEXT_SPOT_M[0] long: starting and stopping alone carry a man 3.7 m, so a shorter one would overshoot its mark."""
    candidates = [s for k in tables for s in _spots_of(site, k, reachable)]
    here = [s for s in _spots_of(site, table, reachable) if np.linalg.norm(np.array(s.xy) - entry) >= NEXT_SPOT_M[0]]
    chosen = [_pulled(rng, site, (here or candidates)[int(rng.integers(len(here or candidates)))])]
    for _ in range(15):
        last = np.array(chosen[-1].pull_to)
        near = [s for s in candidates if NEXT_SPOT_M[0] <= np.linalg.norm(np.array(s.xy) - last) <= NEXT_SPOT_M[1]]
        if not near:
            break
        # A stretch he can walk to straight, along the same aisle, before one round the end of a row.
        near = [s for s in near if site.clear_line(last, s.xy)] or near
        chosen.append(_pulled(rng, site, near[int(rng.integers(len(near)))]))
    return chosen


def _pulled(rng: np.random.Generator, site: Site, spot: Spot) -> Spot:
    """A stretch with its pull drawn: as long as the seed says, short of anything in the way along the row."""
    start, end = np.array(spot.xy), np.array(spot.pull_to)
    axis = (end - start) / np.linalg.norm(end - start)
    length = float(rng.uniform(*PULL_M))
    while length > PULL_M[0] and not site.clear_line(start, start + axis * length):
        length -= 0.5
    pulled = start + axis * length
    return Spot(spot.xy, spot.face, (float(pulled[0]), float(pulled[1])))


def _near_table(rng: np.random.Generator, site: Site, tables: List[int], first: int) -> int:
    """A table in the first man's row or the ones next to it."""
    row = site.tables[first].row
    near = [k for k in tables if abs(site.tables[k].row - row) <= NEAR_ROWS]
    return int(rng.choice(near)) if near else first


def _lookout_spot(rng: np.random.Generator, site: Site, hole: np.ndarray, inward: np.ndarray,
                  reachable: np.ndarray) -> Tuple[float, float]:
    """Where a lookout stands: a few metres inside the hole, a little to one side, on free ground."""
    along = np.array([-inward[1], inward[0]])
    for _ in range(40):
        xy = hole + inward * rng.uniform(9.0, 13.0) + along * rng.uniform(-3.0, 3.0)
        i, j = site.cell(xy)
        if site.free(xy) and np.isfinite(reachable[i * site.shape[1] + j]):
            return float(xy[0]), float(xy[1])
    xy = hole + inward * (INSIDE_M + 1.0)
    return float(xy[0]), float(xy[1])


def _schedule(rng: np.random.Generator, story: Story) -> None:
    """How many cycles the first man cuts, and how long each man after him waits outside, so they step through one
    after another and all are in by ENTRY_S."""
    lib = library()
    cycle = lib.blocks["cut_fence.0"]["frames"] / lib.fps
    stand = {"hub": "stand"}
    lead_in = lib.seconds(lib.kinds(stand)["cut_fence_in"][0])
    budget = ENTRY_S - lead_in - 4.0 - (len(story.thieves) - 1) * MATE_GAP_S[1]
    story.cycles = int(min(rng.integers(3, 9), max(2, int(budget / cycle))))
    through = lead_in + story.cycles * cycle + 2.0
    getting_up = (lib.seconds(lib.kinds(stand)["stand_to_crouch"][0]) +
                  lib.seconds(lib.kinds({"hub": "crouch"})["crouch_to_stand"][0]) +
                  lib.seconds(lib.kinds(stand)["stand_to_walk"][0]))
    for thief in story.thieves[1:]:
        through += float(rng.uniform(*MATE_GAP_S))
        walk = max(0.0, float(np.linalg.norm(np.array(thief.start) - story.hole)) - 1.8) / 1.38
        thief.wait_s = max(0.5, through - getting_up - walk)


# ---------------------------------------------------------------------------------------------------- the scripts

def _walk_to(actor: Actor, site: Site, goal: Sequence[float], gait: str, stage: str,
             via: Optional[np.ndarray] = None) -> Iterator[Step]:
    """Walk from where the body is to a goal on the park's free ground, by way of a point first if given, and stop
    there standing."""
    start = actor.pos if via is None else via
    legs = site.route(start, goal)
    points = ([] if via is None else [via]) + (list(legs[1:]) if legs else [np.asarray(goal, dtype=float)])
    yield from _walk_route(actor, Route(points), gait, stage)


def _walk_route(actor: Actor, route: Any, gait: str, stage: str) -> Iterator[Step]:
    """Walk a route, starting from standing or from a walking loop, and stop on its end standing."""
    if actor.context == {"hub": "stand"}:
        if gait == "sneak":
            yield Step("stand_to_crouch", stage=stage)
            yield Step("crouch_to_sneak", steer=route, stage=stage)
        else:
            yield Step("stand_to_walk" if gait == "walk" else "stand_to_walk_wary", steer=route, stage=stage)
    loop = actor.lib.blocks[actor.context["loop"]]["kind"]
    stop = {"walk": "walk_to_stand", "walk_wary": "walk_wary_to_stand", "sneak": "sneak_to_crouch"}[loop]
    yield Step("loop", cycles=None, steer=route, then=stop, stage=stage)
    yield Step(stop, steer=route, stage=stage)
    if loop == "sneak":
        yield Step("crouch_to_stand", stage=stage)


def _leave(actor: Actor, story: Story, site: Site, first: str, stage: str) -> Iterator[Step]:
    """Get out the way the crew came in: a block that sets off, its travelling loop to the hole and through it, then out
    of the scene once well outside."""
    hole, inward = story.hole, story.inward
    lead = hole + inward * LEAD_M
    legs = site.route(actor.pos, lead) or [actor.pos, lead]
    route = Route(list(legs[1:]) + [hole - inward * OUTSIDE_M], reach=1.2, straight_end=True)
    yield Step(first, steer=route, stage=stage)
    cycle = max(actor.lib.reach(actor.context["loop"]) * actor.scale, 0.3)
    yield Step("loop", cycles=int(math.ceil(route.left(actor.pos) / cycle)) + 1, steer=route, stage=stage)
    yield Step("gone", stage="gone")


def _enter(actor: Actor, thief: Thief, story: Story, slot: int) -> Iterator[Step]:
    """Get in: the first man cuts the hole or the gate's lock and steps through; the others wait crouched, follow him
    through and stop a little to one side of each other."""
    hole, inward = story.hole, story.inward
    if slot == 0:
        yield Step("cut_fence_in", steer=heading_to(inward), stage="at the fence")
        yield Step("loop", cycles=story.cycles, stage="cutting the fence")
        yield Step("cut_fence_to_walk", steer=Lane(hole - inward * 2.0, hole + inward * INSIDE_M), stage="getting in")
        return
    yield Step("stand_to_crouch", stage="at the fence")
    yield Step("hold", hold=thief.wait_s, stage="at the fence")
    stop = hole + inward * INSIDE_M + np.array(story.opening.along) * STOP_SIDE_M * (1.0 if slot % 2 else -1.0)
    route = Route([hole - inward * 2.5, hole + inward * 1.0, stop], straight_end=True)
    if story.way == "forest" and thief.gait == "sneak":
        yield Step("crouch_to_sneak", steer=route, stage="getting in")
        yield Step("loop", cycles=None, steer=route, then="sneak_to_crouch", stage="getting in")
        yield Step("sneak_to_crouch", steer=route, stage="getting in")
        yield Step("crouch_to_stand", stage="getting in")
        return
    yield Step("crouch_to_stand", stage="at the fence")
    yield Step("stand_to_walk", steer=route, stage="getting in")
    yield Step("loop", cycles=None, steer=route, then="walk_to_stand", stage="getting in")
    yield Step("walk_to_stand", steer=route, stage="getting in")


def _worker(actor: Actor, thief: Thief, story: Story, site: Site, slot: int) -> Iterator[Step]:
    """A worker's whole patrol: in, perhaps back to widen the hole, then cutting and pulling cable along the rows until
    he shoulders a coil and carries it out."""
    yield from _enter(actor, thief, story, slot)
    if thief.hole_work_s > 0.0:
        hole, inward = story.hole, story.inward
        turn = hole + inward * TURN_BACK_M + np.array(story.opening.along) * 1.5
        yield from _walk_route(actor, Route([turn, hole + inward * FENCE_GAP_M], straight_end=True), "walk",
                               "getting in")
        cycle = actor.lib.blocks["cut_fence.0"]["frames"] / actor.lib.fps
        yield Step("cut_fence_in", steer=heading_to(-inward), stage="getting in")
        yield Step("loop", cycles=max(1, int(round(thief.hole_work_s / cycle))), stage="getting in")
        yield Step("cut_fence_to_stand", stage="getting in")
    # He works his stretches in order and, should he finish them all, goes round them again: he leaves only with a coil
    # once his time to carry has come.
    for k in itertools.count():
        n = k
        if k >= len(thief.spots):
            # Round again: the nearest of his stretches that is still a proper walk away.
            far = [i for i, sp in enumerate(thief.spots) if np.linalg.norm(np.array(sp.xy) - actor.pos) >= NEXT_SPOT_M[0]]
            n = min(far or range(len(thief.spots)), key=lambda i: np.linalg.norm(np.array(thief.spots[i].xy) - actor.pos))
        spot = thief.spots[n]
        # A sneak is for getting through the hole; inside he walks, warily if that is his way, and he moves on
        # between stretches at a plain walk unless he is the wary kind.
        gait = "walk" if thief.gait == "walk" or (k and thief.gait == "sneak") else "walk_wary"
        # The first man in is still in the hole: he walks straight on into the park before he turns for his table.
        clear = story.hole + story.inward * INSIDE_M if k == 0 and slot == 0 and thief.hole_work_s == 0.0 else None
        yield from _walk_to(actor, site, spot.xy, gait, "arrival" if k == 0 else "moving on", via=clear)
        cycle = actor.lib.blocks["cut_cable.0"]["frames"] / actor.lib.fps
        yield Step("cut_cable_in", steer=spot.face, stage="cutting cable")
        yield Step("loop", cycles=max(1, int(round(thief.cut_s[n] / cycle))), stage="cutting cable")
        yield Step("cut_cable_to_stand", stage="cutting cable")
        pull = Lane(spot.xy, spot.pull_to)
        away = np.array(spot.pull_to) - np.array(spot.xy)
        yield Step("pull_cable_in", steer=heading_to(-away), stage="pulling cable")
        per = max(0.2, actor.lib.reach(actor.context["loop"]) * actor.scale)
        yield Step("loop", cycles=max(1, int(pull.left(actor.pos) / per)), steer=pull, stage="pulling cable")
        if actor.time_s >= thief.carry_after_s:
            yield from _leave(actor, story, site, "pull_cable_to_carry", "carrying")
            return
        yield Step("pull_cable_to_stand", stage="pulling cable")


def _lookout(actor: Actor, thief: Thief, story: Story, site: Site, slot: int) -> Iterator[Step]:
    """A lookout's whole patrol: in behind the others, then watching from near the hole, standing or crouched."""
    yield from _enter(actor, thief, story, slot)
    yield from _walk_to(actor, site, thief.lookout, "walk", "lookout")
    if thief.watch == "look_around":
        yield Step("look_around_in", steer=heading_to(-story.inward), stage="lookout")
    else:
        yield Step("stand_to_crouch", stage="lookout")
        yield Step("crouch_watch_in", stage="lookout")
    yield Step("loop", cycles=100000, stage="lookout")


# ---------------------------------------------------------------------------------------------------- the patrol

@dataclass
class Man:
    """One thief in the running patrol: his story, his track, his body and what he has heard."""

    thief: Thief
    actor: Actor
    rng: np.random.Generator
    body: Optional[intruders.Intruder] = None
    posed: Optional[Tuple[float, float, float]] = None
    pose_key: Optional[tuple] = None    # the frame, place and yaw the body was last posed in
    entered: bool = False
    heard_at: Optional[float] = None
    reacting: str = ""
    quiet_since: Optional[float] = None


@dataclass
class Theft:
    """A seed's theft while the patrol runs."""

    story: Story
    site: Site
    men: List[Man]
    props: List[Any] = field(default_factory=list)
    heights: Dict[Tuple[int, int], float] = field(default_factory=dict)   # ground height per metre, read as needed


def cast(seed: int, story: Story, site: Site) -> List[Man]:
    """The crew of a story, each man with his track ready to be written from his script.

    A follower's wait outside is played out against the men ahead of him and lengthened until, all the way in and
    a while after, he never comes within SPACING_M of any of them: through a one-metre hole, men go one at a time.
    """
    lib = library()
    men: List[Man] = []
    for n, thief in enumerate(story.thieves):
        while True:
            actor = Actor(lib, thief.start, thief.yaw, scale=thief.scale, mirrored=thief.mirrored, pace=thief.pace)
            actor.context = {"hub": "stand"}
            actor.script = (_lookout(actor, thief, story, site, n) if thief.role == "lookout"
                            else _worker(actor, thief, story, site, n))
            if not men or thief.wait_s > ENTRY_S or _spaced(actor, [m.actor for m in men]):
                break
            thief.wait_s += SPACING_WAIT_S
        men.append(Man(thief, actor, np.random.default_rng([THEFT_SEED_STREAM, int(seed), n])))
    return men


def _spaced(actor: Actor, ahead: List[Actor]) -> bool:
    """Whether a man keeps SPACING_M from every man ahead of him from the start until well after they are all in."""
    fps = actor.lib.fps
    for k in range(0, int((ENTRY_S + 12.0) * fps), 3):
        mine = actor.at(k / fps)
        for other in ahead:
            theirs = other.at(k / fps)
            if math.hypot(mine.x - theirs.x, mine.y - theirs.y) < SPACING_M:
                return False
    return True


def reset(env: Any, ep: SolarEpisode) -> None:
    """Stage this seed's theft, if it has one: the story, the crew's bodies, the cut fence or forced gate."""
    ep.theft = None
    ep.outcome.threats = 0
    if not has_theft(ep.seed):
        return
    asset_dir = ep.park["world"]["asset_dir"]
    try:
        lib = library()
        intruders.catalogue()
        builder.solar_manifest(asset_dir)
    except FileNotFoundError:
        # An install from before the thieves' bodies and moves, or a park without its survey, stages no theft.
        return
    site = Site(ep.seed, asset_dir, dock=ep.dock_position[:2])
    story = write(ep.seed, site)
    if story is None:
        return
    men = cast(ep.seed, story, site)
    ep.theft = Theft(story, site, men)
    for man in men:
        frame = man.actor.at(0.0)
        ground = _ground(env, ep, frame.x, frame.y)
        man.body = intruders.Intruder(man.thief.dress, _pose_args(lib, frame, ground), env.CLIENT)
        man.posed = (frame.x, frame.y, ground + 0.9)
        man.pose_key = (frame.index, frame.mirrored, frame.x, frame.y, frame.yaw)
    ep.theft.props = [_opening_prop(env, ep, story, men[0].actor)]


def advance(env: Any, ep: SolarEpisode) -> None:
    """Write every thief's track to the step physics is about to run, count each man who steps in as a threat, and let
    each one inside hear the drone."""
    theft = ep.theft
    if theft is None:
        return
    t = (ep.step + 1) * SIM_DT
    drone = np.asarray(env.pos[0], dtype=float)
    airborne = ep.phase != "docked" and drone[2] - float(ep.dock_position[2]) > AIRBORNE_M
    for man in theft.men:
        frame = man.actor.at(t)
        if frame.index == GONE:
            continue
        here = np.array([frame.x, frame.y])
        if not man.entered and in_ring(theft.site.ring, here)[0]:
            man.entered = True
            ep.outcome.threats += 1
        if man.entered:
            # The ground is read once per metre he stands on; the park rises 39 m, so the dock's height will not do.
            spot = (int(round(frame.x)), int(round(frame.y)))
            if spot not in theft.heights:
                theft.heights[spot] = _ground(env, ep, float(spot[0]), float(spot[1]))
            ears = np.array([frame.x, frame.y, theft.heights[spot] + EAR_M])
            _hear(man, theft, t, float(np.linalg.norm(drone - ears)), airborne)


def show(env: Any, ep: SolarEpisode, view: Any) -> None:
    """Pose the thieves and the cut fence or gate for a picture about to be taken of a view, as they are at the moment
    it is taken: every thief it can see, and any whose body still stands where it could see it."""
    theft = ep.theft
    if theft is None:
        return
    t = view.step * SIM_DT
    lib = library()
    for prop in theft.props:
        prop.apply(t)
    for man in theft.men:
        frame = man.actor.at(t)
        if frame.index == GONE:
            if man.posed is not None:
                man.body.pose(*_pose_args(lib, frame, AWAY_Z))
                man.posed, man.pose_key = None, None
            continue
        key = (frame.index, frame.mirrored, frame.x, frame.y, frame.yaw)
        if key == man.pose_key:
            # Still (frozen, flat, holding a pose): the body already shows this frame, and every rewrite costs memory.
            continue
        ground = _ground(env, ep, frame.x, frame.y)
        now = (frame.x, frame.y, ground + 0.9)
        if not (_in_view(view, now) or (man.posed is not None and _in_view(view, man.posed))):
            continue
        man.body.pose(*_pose_args(lib, frame, ground))
        man.posed, man.pose_key = now, key


def people(ep: SolarEpisode, step: Optional[int] = None) -> List[Dict[str, Any]]:
    """Every thief in the scene at a step up to the current one: his body ids, where he stands, what he is doing, and
    whether he is a threat, which is a man inside the fence."""
    theft = ep.theft
    if theft is None:
        return []
    t = (ep.step if step is None else min(step, ep.step)) * SIM_DT
    out = []
    for man in theft.men:
        frame = man.actor.at(t)
        if frame.index == GONE:
            continue
        here = np.array([frame.x, frame.y])
        out.append({"bodies": list(man.body.bodies) if man.body else [], "xy": (frame.x, frame.y),
                    "stage": frame.stage, "threat": bool(in_ring(theft.site.ring, here)[0])})
    return out


def threat_bodies(ep: SolarEpisode, step: Optional[int] = None) -> set:
    """The body ids of every threat at a step: the parts of each man inside the fence."""
    return {uid for person in people(ep, step) if person["threat"] for uid in person["bodies"]}


def bodies(ep: SolarEpisode) -> Dict[int, int]:
    """Every drawn body of every thief, mapped to the thief's index in the crew."""
    if ep.theft is None:
        return {}
    return {int(uid): n for n, man in enumerate(ep.theft.men) if man.body for uid in man.body.bodies}


def inside(ep: SolarEpisode, thief: int, step: int) -> bool:
    """Whether a thief stood inside the fence at a control step up to the current one: the test of a threat."""
    if ep.theft is None or not 0 <= thief < len(ep.theft.men):
        return False
    frame = ep.theft.men[thief].actor.at(min(int(step), ep.step) * SIM_DT)
    return frame.index != GONE and bool(in_ring(ep.theft.site.ring, np.array([frame.x, frame.y]))[0])


# ---------------------------------------------------------------------------------------------------- hearing

def _hear(man: Man, theft: Theft, t: float, distance: float, airborne: bool) -> None:
    """Let a man inside hear the drone at his own distance and react after his own delay; a hidden man counts how long
    the drone has been gone."""
    thief = man.thief
    near = airborne and distance <= thief.hear_m
    if man.reacting == "hide":
        if near or distance <= thief.hear_m + QUIET_M:
            man.quiet_since = None
        elif man.quiet_since is None:
            man.quiet_since = t
        return
    if man.reacting:
        return
    if not near:
        if distance > thief.hear_m + QUIET_M:
            man.heard_at = None
        return
    if man.heard_at is not None:
        return
    man.heard_at = t
    if thief.reaction != "work_on":
        man.actor.interrupt(lambda actor: _react(actor, man, theft, t + thief.hear_delay_s))


def _react(actor: Actor, man: Man, theft: Theft, not_before: float) -> Optional[Iterator[Step]]:
    """The reaction his loop allows at this seam, or None to carry on and ask again at the next seam while his delay
    runs."""
    if actor.time_s < not_before:
        actor.interrupt(lambda a: _react(a, man, theft, not_before))
        return None
    kind = actor.lib.blocks[actor.context["loop"]]["kind"]
    options = actor.lib.kinds(actor.context)
    for pattern in _REACT[man.thief.reaction]:
        block = pattern.format(kind)
        if block in options:
            man.reacting = {"freeze": "freeze", "hide": "hide", "run": "run"}[man.thief.reaction]
            return _reaction(actor, man, theft, block, kind, actor.take_rest())
    if man.heard_at is not None:
        # Nothing in this loop for his way of reacting: he carries on and reacts at the next seam that allows it.
        actor.interrupt(lambda a: _react(a, man, theft, not_before))
    return None


def _reaction(actor: Actor, man: Man, theft: Theft, block: str, kind: str, rest: Iterator[Step]) -> Iterator[Step]:
    """Play one reaction block and what follows from it: back to what he was doing, flat until the drone has been gone a
    while, or out at a run."""
    if man.reacting == "freeze":
        yield Step(block, stage="reaction: freeze")
        man.reacting = ""
        # Still hearing it, he may freeze again at the next seam.
        if man.rng.random() < REFREEZE_SHARE:
            man.heard_at = None
        yield from rest
        return
    if man.reacting == "run":
        yield from _leave(actor, theft.story, theft.site, block, "reaction: run")
        return
    yield Step(block, stage="reaction: hide")
    man.quiet_since = None
    getting_up = actor.lib.kinds({"hub": "prone"}).get("prone_to_stand")
    if not getting_up or kind not in _BACK_TO:
        # With no way up but a sprint, he stays flat: he is still inside, and still a threat.
        yield Step("hold", hold=1e6, stage="reaction: hide")
        return
    while man.quiet_since is None or actor.time_s - man.quiet_since < man.thief.quiet_s:
        yield Step("hold", hold=1.0, stage="reaction: hide")
    yield Step("prone_to_stand", stage="reaction: hide")
    for entry in _BACK_TO[kind]:
        yield Step(entry, stage="reaction: hide")
    man.reacting, man.heard_at = "", None
    yield from rest


# ---------------------------------------------------------------------------------------------------- the scene

def _pose_args(lib: Any, frame: Any, ground: float) -> tuple:
    """The intruder pose for a frame standing on the ground: local rotations, the root height, place and yaw."""
    index = max(frame.index, 0)
    rot = lib.rotations[int(frame.mirrored)][index]
    root = lib.root[int(frame.mirrored)][index] * np.array([0.0, 1.0, 0.0])
    return rot, root, (frame.x, frame.y, ground), frame.yaw


def _ground(env: Any, ep: SolarEpisode, x: float, y: float) -> float:
    """Terrain height under a point: the first terrain body a ray straight down meets, past panels, fences and trees."""
    top = 1000.0
    for _ in range(16):
        hit = p.rayTest([x, y, top], [x, y, -1000.0], physicsClientId=env.CLIENT)[0]
        if hit[0] < 0:
            break
        if hit[0] in ep.terrain_uids:
            return float(hit[3][2])
        top = float(hit[3][2]) - 0.01
    return float(ep.dock_position[2])


def _in_view(view: Any, point: Sequence[float]) -> bool:
    """Whether a point, grown by a body's reach, falls inside a picture's field of view."""
    eye, forward, up = (np.asarray(v, dtype=float) for v in (view.eye, view.forward, view.up))
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= max(np.linalg.norm(right), 1e-9)
    up = np.cross(right, forward)
    rel = np.asarray(point, dtype=float) - eye
    depth = float(rel @ forward)
    if depth < -VIEW_MARGIN_M:
        return False
    half_v = math.tan(math.radians(view.vertical_fov_deg) / 2.0)
    half_h = half_v * view.width / view.height
    slack = VIEW_MARGIN_M * math.sqrt(1.0 + half_h * half_h + half_v * half_v)
    return abs(float(rel @ right)) <= depth * half_h + slack and abs(float(rel @ up)) <= depth * half_v + slack


def _opening_prop(env: Any, ep: SolarEpisode, story: Story, cutter: Actor) -> Any:
    """The cut fence panel or the forced gate, timed to the first man's cutting."""
    fps = cutter.lib.fps
    frames = [cutter.at(k / fps) for k in range(int(ENTRY_S * fps))]
    cut_from = next((k for k, f in enumerate(frames) if f.stage == "cutting the fence"), 0) / fps
    through = next((k for k, f in enumerate(frames) if f.stage == "getting in"), len(frames)) / fps
    if story.opening.kind == "gate":
        return GateLeaf(env, ep, story, through - FLAP_PUSH_S)
    return FenceHole(env, ep, story, cut_from, through - FLAP_PUSH_S)


class FenceHole:
    """A fence panel cut open: the panel gives way to the chain link either side of the hole and the flap between two
    slits, which springs out as the slits get longer and is pushed into the park as he steps through, where it stays."""

    def __init__(self, env: Any, ep: SolarEpisode, story: Story, cut_from: float, push_at: float):
        """Build the pieces out of sight, with the panel's own texture."""
        self.cli, self.story, self.cut_from, self.push_at = env.CLIENT, story, cut_from, push_at
        world = ep.park["world"]
        manifest = builder.solar_manifest(world["asset_dir"])
        self.place = manifest["placements"][story.opening.index]
        item = manifest["items"]["fence_panel"]
        self.panel = _body_at(self.cli, world["bodies"]["park"], self.place["position"])
        sx = float(self.place["scale"][0])
        half, middle = story.cut_width / 2.0 / sx, story.cut_along / sx
        lo, hi = max(-1.25, middle - half), min(1.25, middle + half)
        hinge = hi if story.hinge > 0 else lo
        texture = p.loadTexture(os.path.join(world["asset_dir"], item["folder"], item["texture"]), physicsClientId=self.cli)
        self.pieces = {}
        for name, (x0, x1, origin) in {"left": (-1.25, lo, 0.0), "right": (hi, 1.25, 0.0), "flap": (lo, hi, hinge)}.items():
            if x1 - x0 > 1e-3:
                self.pieces[name] = (_panel_piece(self.cli, x0, x1, origin, self.place["scale"], texture), origin)
        yaw = yaw_of(self.place["quaternion"])
        local_y = np.array([-math.sin(yaw), math.cos(yaw)])
        # Turning the flap about its hinge by +a moves its free edge, d along the panel from the hinge, by d sin(a)
        # along the panel's local y; the sign that sends it into the park follows from which side the park lies on.
        free_edge = (lo + hi) / 2.0 - hinge
        self.sense = float(np.sign(local_y @ story.inward) * np.sign(free_edge))
        self.state: Optional[tuple] = None

    def apply(self, t: float) -> None:
        """Set the fence for patrol time t: whole before cutting starts, then cut, the flap swinging in at the push."""
        if t < self.cut_from:
            state: tuple = ("whole",)
        elif t < self.push_at:
            state = ("cut", round(FLAP_SPRING_DEG * min(1.0, (t - self.cut_from) / max(self.push_at - self.cut_from, 1e-6)), 1))
        else:
            share = min(1.0, (t - self.push_at) / FLAP_PUSH_S)
            state = ("cut", round(FLAP_SPRING_DEG + (self.story.open_deg - FLAP_SPRING_DEG) * share, 1))
        if state == self.state:
            return
        self.state = state
        pos, quat = self.place["position"], self.place["quaternion"]
        whole = state[0] == "whole"
        p.resetBasePositionAndOrientation(self.panel, [pos[0], pos[1], pos[2] if whole else AWAY_Z], quat,
                                          physicsClientId=self.cli)
        yaw = yaw_of(quat)
        c, s = math.cos(yaw), math.sin(yaw)
        sx = float(self.place["scale"][0])
        for name, (uid, origin) in self.pieces.items():
            if whole:
                p.resetBasePositionAndOrientation(uid, [0.0, 0.0, AWAY_Z], [0, 0, 0, 1], physicsClientId=self.cli)
                continue
            turn = self.sense * math.radians(state[1]) if name == "flap" else 0.0
            orient = p.multiplyTransforms([0, 0, 0], quat, [0, 0, 0], p.getQuaternionFromEuler([0, 0, turn]))[1]
            p.resetBasePositionAndOrientation(uid, [pos[0] + c * origin * sx, pos[1] + s * origin * sx, pos[2]], orient,
                                              physicsClientId=self.cli)


class GateLeaf:
    """The forced gate: the gate gives way to its two leaves, one of which swings into the park as the crew goes
    through and stays open."""

    def __init__(self, env: Any, ep: SolarEpisode, story: Story, push_at: float):
        """Split the gate's mesh into its two leaves, each turning on its outer post."""
        self.cli, self.story, self.push_at = env.CLIENT, story, push_at
        world = ep.park["world"]
        manifest = builder.solar_manifest(world["asset_dir"])
        self.place = manifest["placements"][story.opening.index]
        item = manifest["items"]["gate"]
        self.gate = _body_at(self.cli, world["bodies"]["park"], self.place["position"])
        folder = os.path.join(world["asset_dir"], item["folder"])
        texture = p.loadTexture(os.path.join(folder, item["texture"]), physicsClientId=self.cli)
        verts, uvs, faces = _read_textured_obj(os.path.join(folder, item["obj"]))
        yaw = yaw_of(self.place["quaternion"])
        c, s = math.cos(yaw), math.sin(yaw)
        self.leaves = {}
        for side in (-1, 1):
            keep = np.sign(verts[faces][:, :, 0].mean(1)) == side
            used = np.unique(faces[keep])
            remap = np.full(len(verts), -1)
            remap[used] = np.arange(len(used))
            hinge = verts[used][np.argmax(np.abs(verts[used][:, 0]))][:2]
            local = verts[used] - np.array([hinge[0], hinge[1], 0.0])
            shape = p.createVisualShape(p.GEOM_MESH, vertices=local.tolist(), indices=remap[faces[keep]].ravel().tolist(),
                                        uvs=uvs[used].tolist(), flags=getattr(p, "VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY", 0),
                                        physicsClientId=self.cli)
            # A thin box along the leaf, from its hinge to the gate's middle, standing in for the gate's own mesh.
            span = -hinge
            box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[float(np.linalg.norm(span)) / 2.0, 0.03,
                                                                  float(verts[:, 2].max()) / 2.0],
                                         collisionFramePosition=[float(span[0]) / 2.0, float(span[1]) / 2.0,
                                                                 float(verts[:, 2].max()) / 2.0],
                                         collisionFrameOrientation=p.getQuaternionFromEuler(
                                             [0.0, 0.0, math.atan2(float(span[1]), float(span[0]))]),
                                         physicsClientId=self.cli)
            uid = int(p.createMultiBody(0, box, shape, [0.0, 0.0, AWAY_Z], physicsClientId=self.cli))
            p.changeVisualShape(uid, -1, textureUniqueId=texture, rgbaColor=[1, 1, 1, 1], physicsClientId=self.cli)
            # Turning by +a moves the leaf's middle, -hinge/2 from the hinge, along its perpendicular; keep the sense
            # that sends it into the park.
            mid = -hinge / 2.0
            push = np.array([-mid[1], mid[0]])
            sense = float(np.sign(np.array([c * push[0] - s * push[1], s * push[0] + c * push[1]]) @ story.inward))
            anchor = np.array([c * hinge[0] - s * hinge[1], s * hinge[0] + c * hinge[1]])
            self.leaves[side] = (uid, anchor, sense)
        self.state: Optional[float] = None

    def apply(self, t: float) -> None:
        """Set the gate for patrol time t: shut, then its leaf swinging in, then open."""
        share = round(min(1.0, max(0.0, (t - self.push_at) / FLAP_PUSH_S)), 3)
        if share == self.state:
            return
        self.state = share
        pos, quat = self.place["position"], self.place["quaternion"]
        shut = share <= 0.0
        p.resetBasePositionAndOrientation(self.gate, [pos[0], pos[1], pos[2] if shut else AWAY_Z], quat,
                                          physicsClientId=self.cli)
        for side, (uid, anchor, sense) in self.leaves.items():
            if shut:
                p.resetBasePositionAndOrientation(uid, [0.0, 0.0, AWAY_Z], [0, 0, 0, 1], physicsClientId=self.cli)
                continue
            turn = sense * math.radians(self.story.open_deg) * share if side == self.story.hinge else 0.0
            orient = p.multiplyTransforms([0, 0, 0], quat, [0, 0, 0], p.getQuaternionFromEuler([0, 0, turn]))[1]
            p.resetBasePositionAndOrientation(uid, [pos[0] + anchor[0], pos[1] + anchor[1], pos[2]], orient,
                                              physicsClientId=self.cli)


def _body_at(cli: int, bodies: Sequence[int], position: Sequence[float]) -> int:
    """The park body standing at a position."""
    target = np.asarray(position, dtype=float)
    for uid in bodies:
        if np.linalg.norm(np.array(p.getBasePositionAndOrientation(uid, physicsClientId=cli)[0]) - target) < 1e-3:
            return int(uid)
    raise ValueError(f"no park body at {list(target)}")


def _panel_piece(cli: int, x0: float, x1: float, origin: float, scale: Sequence[float], texture: int) -> int:
    """One upright rectangle of a fence panel, x0 to x1 along it in its own metres, turning about origin, with the
    panel's texture where that rectangle sat on it."""
    sx, _sy, sz = (float(v) for v in scale)
    verts = [[(x0 - origin) * sx, 0.0, 0.0], [(x1 - origin) * sx, 0.0, 0.0], [(x1 - origin) * sx, 0.0, 2.0 * sz],
             [(x0 - origin) * sx, 0.0, 2.0 * sz]]
    uvs = [[(x0 + 1.25) / 2.5, 0.0], [(x1 + 1.25) / 2.5, 0.0], [(x1 + 1.25) / 2.5, 1.0], [(x0 + 1.25) / 2.5, 1.0]]
    shape = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=[0, 1, 2, 0, 2, 3], uvs=uvs,
                                normals=[[0.0, -1.0, 0.0]] * 4,
                                flags=getattr(p, "VISUAL_SHAPE_DOUBLE_SIDED_MULTIBODY", 0), physicsClientId=cli)
    # The same thin box the whole panel had, so the cut fence stops a drone as the whole one did.
    box = p.createCollisionShape(p.GEOM_BOX, halfExtents=[(x1 - x0) / 2.0 * sx, 0.01, sz],
                                 collisionFramePosition=[((x0 + x1) / 2.0 - origin) * sx, 0.0, sz], physicsClientId=cli)
    uid = int(p.createMultiBody(0, box, shape, [0.0, 0.0, AWAY_Z], physicsClientId=cli))
    p.changeVisualShape(uid, -1, textureUniqueId=texture, rgbaColor=[1, 1, 1, 1], specularColor=[0.0] * 3,
                        physicsClientId=cli)
    return uid


def _read_textured_obj(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertices, one texture coordinate per vertex and the triangles of an OBJ that gives each vertex one uv."""
    verts, tex, faces, uv_of = [], [], [], {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                tex.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                corners = []
                for token in line.split()[1:4]:
                    parts = token.split("/")
                    corners.append(int(parts[0]) - 1)
                    if len(parts) > 1 and parts[1]:
                        uv_of[corners[-1]] = int(parts[1]) - 1
                faces.append(corners)
    uvs = np.zeros((len(verts), 2))
    for v, k in uv_of.items():
        uvs[v] = tex[k]
    return np.asarray(verts, dtype=float), uvs, np.asarray(faces, dtype=np.int64)
