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

"""People on the field: how many a seed holds, who they are, where they stay and how they hold themselves; and the
people passing outside it.

A seed holds nobody on the field when the mix says so (one seed in twenty), else 1 to 10 people by the decided shares,
never more than the field has room for: one person per AREA_PER_PERSON_M2 of field. They are the people of the worlds'
people package (the figures module): women and men of many builds with painted skin, eyes, hair and everyday clothes
fit for the air's temperature, each body scaled a little on top of its build. Any two of them differ in at least two
of the three things that tell people apart from above: the top's colour, the head (hair and its colour, or none) and
the colour of the trousers or the skirt.

Most stay where the seed puts them for the whole patrol: standing, or low (squatting, kneeling or sitting). Nobody
lies on the ground, and nobody is a statue: each keeps busy on his spot with habits of his own and may look up at the
drone or wave (the idlers module). About one in four walks a little inside a patch of his own and stops for the drone
(the walkers module); at night fewer do. A night seed draws the same number for each person's pose as it would by day, read against
the night's shares, so the rest of the seed's draws stay in their places and only someone who would walk by day may
crouch instead.

Each person's reference point lies on the field PERSON_EDGE_M inside its outline, at least DOCK_CLEAR_M from the
dock, where only the ground meets rays dropped round it, and CROWN_CLEAR_M clear of every tree crown that reaches the
field; a walker's whole patch does, and people's patches stay APART_M apart. The reference point, the place a report
must name, is the centre of the box a still body covers seen from above: between the feet of someone standing or
crouching; for a walker, the ground under his hips.

On a field that reaches into the trees (the site's field kind) the people who stay put take cover (the cover module):
some stand beside a tree, a few under one, where only a drone flying low and looking under the leaves sees them, and
the rest in the open, where the survey lets someone stand. Someone under a tree stays there only when the fairness rule
finds enough places a drone may fly that see him clearly; that is checked once everyone stands where the seed put them,
and when it fails he is moved to another tree, and at last to the open. Walkers always stroll in the open, and on an
open field everyone stands as above.

In the seeds the mix gives them, people also pass outside the field (the outsiders module). They come after everyone
on the field in the patrol's list, so a person's index is his place in it and the people on the field hold the first
ones. They are dressed the same way and apart from everyone before them, so only where they walk tells them apart; they
are never people to find.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from swarm.constants import SIM_DT

from ..solar_patrol.fixed_order import norm
from ..solar_patrol.flight_limit import _distance, _inside, _sides_of
from ..solar_patrol.theft_moves import heading_to, hips_heading
from . import cover, figures, idlers, mix, moves, outsiders, site, walkers
from .episode import FieldEpisode

PEOPLE_SEED_STREAM = 0x9E091E          # the people's own stream, so no other draw of the seed moves them
COUNT_SHARES = {1: 0.03, 2: 0.06, 3: 0.10, 4: 0.16, 5: 0.19, 6: 0.18, 7: 0.16, 8: 0.07, 9: 0.03, 10: 0.02}
MOST_PEOPLE = max(COUNT_SHARES)
AREA_PER_PERSON_M2 = 350.0             # room one person takes on a field, the reference pilot's included
POSE_SHARES = {"standing": 0.45, "crouching": 0.30, "walking": 0.25}
NIGHT_POSE_SHARES = {"standing": 0.45, "crouching": 0.40, "walking": 0.15}   # only a day's walker changes pose
LOW_SHARES = {"crouch": 0.45, "kneel": 0.35, "sit": 0.20}   # the base pose of someone low, of those the library holds
LIFE_STREAM = 6                        # each person's own stream for his habits, after the people's other ones
TOGETHER_STREAM = 7                    # who is with whom, and whose patch is wide
TOGETHER_SHARE = 0.30                  # how many people, after the first, are with someone drawn before them
GROUP_MOST = 4                         # the most people who stand together; two at most walk together
TOGETHER_M = (0.9, 1.5)                # how far from the first of a group another stands
LOW_TOGETHER_M = (1.3, 1.9)            # when one of the two squats, kneels or sits
MEMBER_APART_M = 0.8                   # the least between any two of one group
TOGETHER_TRIES = 32                    # draws of a spot beside someone before the newcomer stands alone
FACE_SWAY_RAD = math.radians(25.0)     # how far off the other someone in a group may face
STEP_DRAWS = 6                         # outfits tried for someone who walks with another, to match his stride
STEP_OFF = 0.02                        # a stride this near the other's is taken at once
STEP_SIZE_RANGE = (0.94, 1.06)         # he may be sized this far off his build to match the other's stride
SIZE_RANGE = (0.97, 1.03)              # each body scaled on top of its build
PARTS_APART = 2                        # of top, head and trousers, two people in a seed differ in at least this many
OUTFIT_TRIES = 200
PERSON_EDGE_M = 2.0                    # a report within its 2 m tolerance of anyone inside never names a point outside
DOCK_CLEAR_M = 8.0
APART_M = 3.0
CROWN_CLEAR_M = 0.5                    # off the edge of every crown circle that reaches the field
OPEN_M = 1.0                           # only the ground meets a ray this close to a person's spot
OPEN_RING_RAYS = 8
PLACE_TRIES = 4000
PLACE_ROUNDS = 4                       # fresh draws of everyone's spots before a walker is made to stand still
AWAY_Z = -500.0                        # where a body waits while it is being measured


@dataclass
class Person:
    """One person on the field or passing outside it: his pose, where his reference point stands, his body and its
    drawn ids."""

    index: int
    pose: str
    xy: Tuple[float, float]            # the reference point, world metres; a walker's home, a passer's first spot
    ground_z: float
    yaw: float
    body: Any = None
    inside: bool = True                # on the field, so a person to find; False for someone passing outside it
    height_m: float = 1.8              # how tall he stands
    top_m: Optional[float] = None      # how high he reaches as he holds himself still; None for someone who walks
    size: float = 1.0                  # his body's scale on top of its build
    walk: Optional[walkers.Walk] = None
    passer: Optional[outsiders.Pass] = None
    cover: str = cover.OPEN            # "open", or "beside" or "under" a tree on a wooded field
    hub: str = ""                      # the base pose someone who stays put holds or returns to
    stride: float = 1.0                # how far a move carries him for each metre it carries the standard build
    idle: Optional[idlers.Idle] = None  # what someone who stays put does on his spot
    beside: Optional[int] = None       # the person he stands or walks with, the first of his group; None for that one
    company: bool = False              # he is one of a group

    @property
    def head_m(self) -> float:
        """How high above the ground under him the top of his body is: lower than he stands when he crouches or
        kneels, his full height when he walks."""
        return self.height_m if self.top_m is None else self.top_m

    @property
    def patch_m(self) -> float:
        """How far from his reference point he may walk: his patch's radius, none for someone who stays put."""
        return 0.0 if self.walk is None else float(self.walk.patch)

    @property
    def lane(self) -> Optional[np.ndarray]:
        """The line a passer walks, world metres east and north; None for anyone on the field."""
        return None if self.passer is None else self.passer.lane


class NoRoom(RuntimeError):
    """The field has no room left for the seed's people: the seed's own world fails, not the machine."""


class NoLook(NoRoom):
    """No outfit drawn for a person looked different enough from everyone before him: the seed's own world fails."""


def wanted(seed: int) -> int:
    """How many people the seed asks for before the field's room caps it: none when the mix puts nobody on the field,
    else 1 to MOST_PEOPLE by the decided shares. A field cutter draws this first and then a field of min_area_m2 for
    it."""
    if not mix.present(seed).people:
        return 0
    return int(_pick(np.random.default_rng([PEOPLE_SEED_STREAM, int(seed), 0]), COUNT_SHARES))


def capacity(area: float) -> int:
    """The most people a field of an area in square metres has room for."""
    return min(MOST_PEOPLE, int(area // AREA_PER_PERSON_M2))


def min_area_m2(people_count: int) -> float:
    """The smallest field, in square metres, with room for a number of people."""
    return float(people_count) * AREA_PER_PERSON_M2


def count(seed: int, area: Optional[float] = None) -> int:
    """How many people the seed puts on a field of an area in square metres: what it asks for, capped by the room;
    uncapped when no area is given."""
    n = wanted(seed)
    return n if area is None else min(n, capacity(area))


def area_m2(outline: np.ndarray) -> float:
    """The area inside an outline, square metres, summed exactly so every machine gets the same bits."""
    points = [(float(x), float(y)) for x, y in np.asarray(outline, dtype=float)]
    return abs(math.fsum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1]))) / 2.0


def _pick(rng: np.random.Generator, weights: Dict[Any, float]) -> Any:
    """One key drawn with its weight."""
    keys = list(weights)
    share = np.array([weights[k] for k in keys], dtype=float)
    return keys[int(rng.choice(len(keys), p=share / share.sum()))]


def air_c(ep: FieldEpisode) -> Optional[float]:
    """The air's temperature on the seed's field, degrees Celsius, which people dress for; None where the site gives
    none."""
    return (ep.park or {}).get("air_c")


def outfit(rng: np.random.Generator, air: Optional[float] = None) -> Dict[str, Any]:
    """One person's build, skin, eyes, hair and clothes fit for the air's temperature, drawn from the seed's generator."""
    return figures.outfit(rng, air)


def look(dress: Dict[str, Any]) -> Tuple[str, str, str]:
    """What tells a person apart from above: the colour of his top, his head and the colour of his trousers or skirt."""
    return figures.look(dress)


def apart(a: Sequence[str], b: Sequence[str]) -> bool:
    """Whether two looks differ in at least PARTS_APART of their parts."""
    return sum(x != y for x, y in zip(a, b)) >= PARTS_APART


def dress_apart(rng: np.random.Generator, looks: Sequence[Tuple[str, str, str]], air: Optional[float] = None) -> Dict[str, Any]:
    """An outfit whose look is apart from every look given, drawn again from the seed's generator until it is."""
    for _ in range(OUTFIT_TRIES):
        dress = outfit(rng, air)
        if all(apart(look(dress), other) for other in looks):
            return dress
    raise NoLook(f"no outfit apart from {len(looks)} others in {OUTFIT_TRIES} draws")


def height_m(dress: Dict[str, Any], size: float) -> float:
    """How tall a person of a build and size stands, metres."""
    return figures.height_m(dress, size)


def pose_hub(rng: np.random.Generator, pose: str) -> Tuple[str, bool]:
    """The base pose someone who stays put holds, and whether it is mirrored: standing, or one of the low poses the
    motion library holds, by their shares."""
    hub = moves.STAND
    if pose != "standing":
        held = moves.low(moves.library())
        low = {name: share for name, share in LOW_SHARES.items() if name in held}
        hub = _pick(rng, low) if low else moves.STAND
    return hub, bool(rng.random() < 0.5)


def pose_frame(hub: str) -> int:
    """The library frame of a base pose."""
    return moves.hub_frame(moves.library(), hub)[0]


def pose_args(index: int, mirrored: bool, place: Tuple[float, float, float], yaw: float) -> tuple:
    """A body's pose for a library frame standing on a place: local rotations, root height, place and yaw."""
    lib = moves.library()
    rot = lib.rotations[int(mirrored)][index]
    root = lib.root[int(mirrored)][index] * np.array([0.0, 1.0, 0.0])
    return rot, root, place, yaw


def _open(cli: int, terrain_uids: frozenset, xy: np.ndarray, reach: float = 0.0) -> bool:
    """Whether only the ground meets rays dropped on a spot and on rings round it out to reach and OPEN_M beyond, about
    a metre apart, so nobody stands or walks into a wall or onto a stone."""
    points = [(float(xy[0]), float(xy[1]))]
    for ring in np.arange(OPEN_M, reach + OPEN_M + 1e-9, OPEN_M):
        rays = max(OPEN_RING_RAYS, int(math.ceil(2.0 * math.pi * ring / OPEN_M)))
        points += [(float(xy[0]) + ring * math.cos(2.0 * math.pi * k / rays),
                    float(xy[1]) + ring * math.sin(2.0 * math.pi * k / rays)) for k in range(rays)]
    hits = p.rayTestBatch([[x, y, site.RAY_REACH_M] for x, y in points], [[x, y, -site.RAY_REACH_M] for x, y in points],
                          physicsClientId=cli)
    return all(hit[0] in terrain_uids for hit in hits)


class Ground:
    """What a spot on one field is checked against: the outline, the dock, the crowns, the terrain the rays must meet
    and, on a wooded field, where the survey lets someone stand in the open; and the trees a spot under or beside one
    is drawn round, the crowns themselves unless given."""

    def __init__(self, cli: int, outline: np.ndarray, terrain_uids: frozenset, dock_xy: np.ndarray, crowns: np.ndarray,
                 open_at: Optional[Callable[[Any, Any], np.ndarray]] = None, trees: Optional[np.ndarray] = None):
        """Keep what every spot is checked against."""
        self.cli, self.outline, self.terrain_uids = cli, outline, terrain_uids
        self.dock_xy, self.crowns, self.open_at = dock_xy, crowns, open_at
        self.trees = crowns if trees is None else trees
        self.sides = _sides_of(outline)

    def fits(self, spot: np.ndarray, r: float, cover_kind: str, others: Sequence[Tuple[np.ndarray, float]]) -> bool:
        """Whether a disc of reach r round a spot keeps the rules for its cover: inside the outline by PERSON_EDGE_M,
        clear of the dock and of the others' discs by APART_M, clear of every crown but under a tree, in the open on
        ground the survey lets someone stand on when it says, and where only the ground meets rays dropped round it."""
        if not _inside(self.sides, spot) or _distance(self.outline, spot) < PERSON_EDGE_M + r:
            return False
        if np.hypot(*(spot - self.dock_xy)) < DOCK_CLEAR_M + r:
            return False
        if any(np.hypot(*(spot - other)) < APART_M + r + reach for other, reach in others):
            return False
        crowns = self.crowns
        if cover_kind != cover.UNDER and len(crowns) and np.any(np.hypot(*(crowns[:, :2] - spot).T) < crowns[:, 2] + CROWN_CLEAR_M + r):
            return False
        if cover_kind == cover.OPEN and self.open_at is not None and not self.open_at(spot[:1], spot[1:2])[0]:
            return False
        return _open(self.cli, self.terrain_uids, spot, r)


def _near(rng: np.random.Generator, ground: Ground, anchor: np.ndarray, gap: Tuple[float, float], r: float,
          others: Sequence[Tuple[np.ndarray, float]], group: Sequence[np.ndarray]) -> Optional[np.ndarray]:
    """A spot for someone who is with another: gap metres from the first of his group, at least MEMBER_APART_M from
    each of them, his disc keeping the open's rules against everyone else; None when TOGETHER_TRIES draws give none."""
    for _ in range(TOGETHER_TRIES):
        angle, away = float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(*gap))
        spot = anchor + away * np.array([math.cos(angle), math.sin(angle)])
        if any(norm(spot - member) < MEMBER_APART_M for member in group):
            continue
        if ground.fits(spot, r, cover.OPEN, others):
            return spot
    return None


def layout(ground: Ground, seed: int, wanted_count: int, reach: Optional[Sequence[float]] = None, attempt: int = 0,
           covers: Optional[Sequence[str]] = None, lead: Optional[Sequence[Optional[int]]] = None,
           gaps: Optional[Sequence[Tuple[float, float]]] = None,
           alone: Optional[Sequence[float]] = None) -> Tuple[List[np.ndarray], List[str], List[Optional[int]]]:
    """Where the seed's people stand and under what cover, drawn from the people's stream, widest discs first; the
    spots, the covers and who is still with whom come back in the people's order. Groups and covers are explained in
    company() and the cover module."""
    rng = np.random.default_rng([PEOPLE_SEED_STREAM, int(seed), 1, int(attempt)])
    given = reach if isinstance(reach, list) else None
    reach = [0.0] * wanted_count if reach is None else [float(r) for r in reach]
    kinds = [cover.OPEN] * wanted_count if covers is None else list(covers)
    lead = [None if first is None or not 0 <= first < wanted_count else first for first in (lead or [None] * wanted_count)]
    gaps = [TOGETHER_M] * wanted_count if gaps is None else list(gaps)
    low, high = ground.outline.min(axis=0), ground.outline.max(axis=0)
    placed: Dict[int, np.ndarray] = {}
    order = sorted((n for n in range(wanted_count) if lead[n] is None), key=lambda n: -reach[n])
    tries = 0
    for n in order + [m for m in range(wanted_count) if lead[m] is not None]:
        if n in placed:
            continue
        if lead[n] is not None and alone is not None:
            reach[n] = float(alone[n])
        r = reach[n]
        others = [(other, reach[m]) for m, other in placed.items()]
        if kinds[n] != cover.OPEN:
            for _ in range(cover.COVER_TRIES):
                spot = cover.candidate(rng, kinds[n], ground.trees)
                if spot is None:
                    break
                if ground.fits(spot, r, kinds[n], others):
                    placed[n] = spot
                    break
            if n not in placed:
                kinds[n] = cover.OPEN
        while n not in placed and tries < PLACE_TRIES:
            tries += 1
            spot = rng.uniform(low, high)
            if ground.fits(spot, r, cover.OPEN, others):
                placed[n] = spot
        if n not in placed:
            break
        # Anyone reached here who still names a leader found no spot beside him and stands alone.
        lead[n] = None
        group = [n]
        for m in (m for m in range(wanted_count) if lead[m] == n):
            apart = [(other, reach[q]) for q, other in placed.items() if q not in group]
            spot = _near(rng, ground, placed[n], gaps[m], reach[m], apart, [placed[q] for q in group])
            if spot is not None:
                placed[m], kinds[m] = spot, cover.OPEN
                group.append(m)
    if len(placed) < wanted_count:
        raise NoRoom(f"room for {len(placed)} of {wanted_count} people in seed {seed}")
    if given is not None:
        given[:] = reach
    return [placed[n] for n in range(wanted_count)], kinds, lead


def spots(cli: int, seed: int, outline: np.ndarray, terrain_uids: frozenset, dock_xy: np.ndarray,
          crowns: np.ndarray, wanted_count: int, reach: Optional[Sequence[float]] = None,
          attempt: int = 0) -> List[np.ndarray]:
    """Where the seed's people stand on an open field, drawn from its own stream: each person's disc of his reach round
    the spot (none for someone who stays put) inside the outline by PERSON_EDGE_M, clear of the dock, of every crown
    and of the others' discs by APART_M, on open ground. The widest discs are placed first; the spots come back in the
    people's order."""
    return layout(Ground(cli, outline, terrain_uids, dock_xy, crowns), seed, wanted_count, reach, attempt)[0]


def company(rng: np.random.Generator, poses: Sequence[str],
            covers: Optional[Sequence[str]] = None) -> Tuple[List[Optional[int]], List[float]]:
    """Who is with whom and how wide each walker's patch is, drawn in the people's order: for each person, the first
    of his group (None for that one and for anyone alone) and the radius of his patch (none for someone who stays
    put). A group is a walker with a walker, or people who stay put in the open, of fewer than GROUP_MOST."""
    n = len(poses)
    kinds = [cover.OPEN] * n if covers is None else list(covers)
    lead: List[Optional[int]] = [None] * n
    patches = [0.0] * n
    for k in range(n):
        walking = poses[k] == "walking"
        if walking:
            wide = bool(rng.random() < walkers.WIDE_SHARE)
            patches[k] = float(rng.uniform(*walkers.WIDE_PATCH_M)) if wide else walkers.PATCH_M
        if k == 0 or rng.random() >= TOGETHER_SHARE:
            continue
        if walking:
            free = [j for j in range(k) if poses[j] == "walking" and lead[j] is None and j not in lead]
        elif kinds[k] == cover.OPEN:
            free = [j for j in range(k) if poses[j] != "walking" and kinds[j] == cover.OPEN and lead[j] is None
                    and lead.count(j) < GROUP_MOST - 1]
        else:
            free = []
        if free:
            lead[k] = free[int(rng.integers(len(free)))]
    return lead, patches


def _gaps(poses: Sequence[str], lead: Sequence[Optional[int]]) -> List[Tuple[float, float]]:
    """How far from the first of his group each person keeps: a step for two who walk, closer for two who stand than
    when one of them is low."""
    gaps = []
    for pose, first in zip(poses, lead):
        if pose == "walking":
            gaps.append(walkers.BESIDE_M)
        else:
            both_stand = pose == "standing" and (first is None or poses[first] == "standing")
            gaps.append(TOGETHER_M if both_stand else LOW_TOGETHER_M)
    return gaps


def place(ground: Ground, seed: int, poses: Sequence[str], covers: Optional[Sequence[str]] = None,
          patches: Optional[Sequence[float]] = None, lead: Optional[Sequence[Optional[int]]] = None,
          ) -> Tuple[List[np.ndarray], List[str], List[str], List[float], List[Optional[int]]]:
    """Everyone's spot, pose, cover, patch and company, narrowing the walkers' patches until all fit; NoRoom once
    nobody walks and they still do not. patches gives each walker's radius (PATCH_M if not given), lead the first of
    each person's group."""
    poses = list(poses)
    kinds = [cover.OPEN] * len(poses) if covers is None else list(covers)
    lead = [None] * len(poses) if lead is None else list(lead)
    patches = [walkers.PATCH_M] * len(poses) if patches is None else [float(patch) for patch in patches]
    while True:
        own = [patches[n] if pose == "walking" else 0.0 for n, pose in enumerate(poses)]
        for attempt in range(PLACE_ROUNDS):
            # Two who walk together share the first one's patch; whoever ends up alone keeps his own.
            reach = [own[n] if lead[n] is None else own[lead[n]] for n in range(len(poses))]
            try:
                where, placed_covers, placed_lead = layout(ground, seed, len(poses), reach, attempt, kinds, lead,
                                                           _gaps(poses, lead), own)
                return where, poses, placed_covers, reach, placed_lead
            except NoRoom:
                continue
        walking = [n for n, pose in enumerate(poses) if pose == "walking"]
        wide = [n for n in walking if patches[n] > walkers.PATCH_M]
        if wide:
            patches[wide[-1]] = walkers.PATCH_M
            continue
        if not walking:
            raise NoRoom(f"room for fewer than {len(poses)} people in seed {seed}")
        last = walking[-1]
        poses[last] = "standing"
        lead = [None if n == last or first == last else first for n, first in enumerate(lead)]


def reset(env: Any, ep: FieldEpisode) -> None:
    """Put the seed's people on the field, each dressed apart from the others, sized, and posed on the ground at his
    spot or setting off on his stroll; then the people passing outside it."""
    n = count(ep.seed, area_m2(ep.fence))
    ep.people = []
    ep.outcome.threats = n
    looks = _on_field(env, ep, n)
    _passing(env, ep, looks)


def _on_field(env: Any, ep: FieldEpisode, n: int) -> List[Tuple[str, str, str]]:
    """Put n people on the field, under the cover a wooded field gives them; returns their looks."""
    looks: List[Tuple[str, str, str]] = []
    if not n:
        return looks
    cli = env.CLIENT
    rng = np.random.default_rng([PEOPLE_SEED_STREAM, int(ep.seed), 2])
    shares = NIGHT_POSE_SHARES if site.night(ep.seed) else POSE_SHARES
    poses = [_pick(rng, shares) for _ in range(n)]
    wooded = site.wooded(ep)
    covers = cover.draw(np.random.default_rng([PEOPLE_SEED_STREAM, int(ep.seed), 4]),
                        [pose != "walking" for pose in poses]) if wooded else None
    ground = Ground(cli, ep.fence, ep.terrain_uids, ep.dock_position[:2], site.passable_outlines(ep),
                    ep.park.get("open_at") if wooded else None, site.trees(ep) if wooded else None)
    lead, patches = company(np.random.default_rng([PEOPLE_SEED_STREAM, int(ep.seed), TOGETHER_STREAM]), poses, covers)
    where, poses, covers, patches, lead = place(ground, ep.seed, poses, covers, patches, lead)
    frames: Dict[int, Tuple[int, bool]] = {}
    for k, (spot, pose, kind) in enumerate(zip(where, poses, covers)):
        first = lead[k]
        with_him = [m for m in range(n) if lead[m] == k]
        if pose == "walking" and first is not None:
            dress, size = _in_step(rng, looks, air_c(ep), ep.people[first].walk.scale)
            looks.append(look(dress))
        else:
            dress = dress_apart(rng, looks, air_c(ep))
            looks.append(look(dress))
            size = float(rng.uniform(*SIZE_RANGE))
        if pose == "walking":
            leader = None if first is None else ep.people[first].walk
            ep.people.append(_walker(env, ep, k, dress, size, spot, patches[k], leader))
            ep.people[-1].beside, ep.people[-1].company = first, first is not None or bool(with_him)
            continue
        hub, mirrored = pose_hub(rng, pose)
        index = pose_frame(hub)
        yaw = float(rng.uniform(-math.pi, math.pi))
        if first is not None or with_him:
            # Someone in a group faces the one he is with, give or take.
            other = where[first] if first is not None else where[with_him[0]]
            yaw = facing(index, mirrored, other - spot) + float(rng.uniform(-FACE_SWAY_RAD, FACE_SWAY_RAD))
        ep.people.append(_stand(cli, ep, k, pose, dress, size, hub, mirrored, spot, yaw, first is not None or bool(with_him)))
        ep.people[-1].cover, ep.people[-1].beside = kind, first
        frames[k] = (index, mirrored)
    if cover.UNDER in covers:
        _fair(env, ep, ground, frames)
    return looks


def _fair(env: Any, ep: FieldEpisode, ground: Ground, frames: Dict[int, Tuple[int, bool]]) -> None:
    """Keep everyone under a tree only where the fairness rule finds enough places that see him clearly and few
    enough that the tree hides him, checked against where everyone stands: someone it fails is moved to another place
    under a tree, up to cover.UNDER_DRAWS times, and then to the open; the checks run again until a whole round moves
    nobody."""
    sight = cover.Sight(env, ep, ground.crowns)
    under = [k for k, person in enumerate(ep.people) if person.cover == cover.UNDER]
    rngs = {k: np.random.default_rng([PEOPLE_SEED_STREAM, int(ep.seed), 5, k]) for k in under}
    draws = dict.fromkeys(under, 0)
    moved = True
    while moved:
        moved = False
        for k in under:
            person = ep.people[k]
            while person.cover == cover.UNDER and not _seen(sight, ep, k):
                moved = True
                spot = None
                while spot is None and draws[k] < cover.UNDER_DRAWS:
                    draws[k] += 1
                    spot = _redraw(ground, ep, k, rngs[k], cover.UNDER)
                if spot is None:
                    person.cover = cover.OPEN
                    spot = _redraw(ground, ep, k, rngs[k], cover.OPEN)
                if spot is None:
                    raise NoRoom(f"no room in the open for person {k} of seed {ep.seed}")
                _set_down(env.CLIENT, ep, person, *frames[k], spot)


def _everyone(ep: FieldEpisode) -> np.ndarray:
    """Where everyone on the field stands at the patrol's clock, world metres east, north and up."""
    return np.array([position(ep, k) for k, person in enumerate(ep.people) if person.inside]).reshape(-1, 3)


def _seen(sight: cover.Sight, ep: FieldEpisode, k: int) -> bool:
    """Whether the fairness rule keeps one person under his tree where he stands at the patrol's clock: found from
    enough places,
    hidden from most."""
    person = ep.people[k]
    return sight.kept(position(ep, k), person.height_m, person.pose, list(person.body.bodies), _everyone(ep))


def _redraw(ground: Ground, ep: FieldEpisode, k: int, rng: np.random.Generator, kind: str) -> Optional[np.ndarray]:
    """A new spot for one person who stays put, keeping the rules of a cover against everyone else where they stand:
    up to cover.COVER_TRIES draws under a tree, PLACE_TRIES in the open; None when none fits."""
    others = [(np.asarray(person.xy, dtype=float), person.patch_m)
              for m, person in enumerate(ep.people) if m != k and person.inside]
    low, high = ground.outline.min(axis=0), ground.outline.max(axis=0)
    for _ in range(cover.COVER_TRIES if kind == cover.UNDER else PLACE_TRIES):
        spot = cover.candidate(rng, kind, ground.trees) if kind == cover.UNDER else rng.uniform(low, high)
        if spot is None:
            return None
        if ground.fits(spot, 0.0, kind, others):
            return spot
    return None


def _passing(env: Any, ep: FieldEpisode, looks: List[Tuple[str, str, str]]) -> None:
    """Put the seed's people passing outside the field after everyone on it, each dressed apart from everyone before
    him, sized, and on a lane of his own where the site lets someone stand, out from under the crowns it lists and
    along its paths, if it lists any; a passer with no lane that fits stays away, so a field hemmed in by forest or
    houses has nobody passing."""
    wanted_passers = outsiders.count(ep.seed)
    if not wanted_passers:
        return
    rng = np.random.default_rng([outsiders.OUTSIDERS_SEED_STREAM, int(ep.seed), 2])
    park = ep.park or {}
    ground = outsiders.ground_test(env.CLIENT, ep.terrain_uids, park.get("open_at"))
    paths, cover = park.get("paths"), site.passable_outlines(ep)
    planned: List[outsiders.Pass] = []
    for k in range(wanted_passers):
        dress = dress_apart(rng, looks, air_c(ep))
        looks.append(look(dress))
        size = float(rng.uniform(*SIZE_RANGE))
        passer = outsiders.plan(ep.seed, k, ep.fence, figures.stride(dress, size), ground, paths, planned, cover)
        if passer is None:
            continue
        planned.append(passer)
        outsiders.start(env, ep, passer, dress, size)
        first = passer.frames[0]
        ep.people.append(Person(index=len(ep.people), pose="passing", xy=(first.x, first.y),
                                ground_z=outsiders.ground(passer, first), yaw=first.yaw, body=passer.body, inside=False,
                                height_m=height_m(dress, size), size=size, passer=passer))


def _in_step(rng: np.random.Generator, looks: Sequence[Tuple[str, str, str]], air: Optional[float],
             scale: float) -> Tuple[Dict[str, Any], float]:
    """An outfit and a size for someone who walks with another, whose track he keeps: apart from every look given,
    and of a stride as near the other's as STEP_SIZE_RANGE allows, the nearest of up to STEP_DRAWS outfits, so his
    feet keep to the ground they are set on. A draw that finds no look apart ends the search with the nearest so
    far."""
    best: Optional[Tuple[float, Dict[str, Any], float]] = None
    for _ in range(STEP_DRAWS):
        try:
            dress = dress_apart(rng, looks, air)
        except NoLook:
            if best is None:
                raise
            break
        size = float(np.clip(scale / figures.stride(dress, 1.0), *STEP_SIZE_RANGE))
        off = abs(figures.stride(dress, size) / scale - 1.0)
        if best is None or off < best[0]:
            best = (off, dress, size)
        if off <= STEP_OFF:
            break
    return best[1], best[2]


def facing(index: int, mirrored: bool, toward: Sequence[float]) -> float:
    """The yaw at which a body in a library frame faces along a ground direction."""
    hips = hips_heading(moves.library().rotations[int(mirrored)][index][0])
    return heading_to(toward) - (0.0 if hips is None else hips)


def _walker(env: Any, ep: FieldEpisode, n: int, dress: Dict[str, Any], size: float, home: np.ndarray,
            patch: float = walkers.PATCH_M, leader: Optional[walkers.Walk] = None) -> Person:
    """A walker at home in his patch: his walk planned on his own stream, or kept beside the one he walks with, his
    body standing where it begins."""
    if leader is None:
        walk = walkers.plan(np.random.default_rng([PEOPLE_SEED_STREAM, int(ep.seed), 3, n]), home,
                            figures.stride(dress, size), patch)
    else:
        walk = walkers.plan_beside(leader, home)
    walkers.start(env, ep, walk, dress, size)
    return Person(index=n, pose="walking", xy=(float(home[0]), float(home[1])), ground_z=walkers.ground(walk, 0),
                  yaw=walk.yaw, body=walk.body, height_m=height_m(dress, size), size=size, walk=walk)


def _stand(cli: int, ep: FieldEpisode, n: int, pose: str, dress: Dict[str, Any], size: float, hub: str,
           mirrored: bool, spot: np.ndarray, yaw: float, with_others: bool = False) -> Person:
    """One person's body, posed in his base pose so his reference point lies on the spot and his lowest point on the
    highest ground under him, so no limb sinks into the slope; and his life on that spot."""
    index = pose_frame(hub)
    body = figures.Figure(figures.catalogue(), dress, pose_args(index, mirrored, (0.0, 0.0, AWAY_Z), yaw), cli, size=size,
                          textures=figures.pictures(ep))
    person = Person(index=n, pose=pose, xy=(0.0, 0.0), ground_z=0.0, yaw=yaw, body=body, height_m=height_m(dress, size),
                    size=size, hub=hub, stride=figures.stride(dress, size), company=with_others)
    _set_down(cli, ep, person, index, mirrored, spot)
    return person


def _set_down(cli: int, ep: FieldEpisode, person: Person, index: int, mirrored: bool, spot: np.ndarray) -> None:
    """Pose a still person's body on a spot, his reference point on it and his lowest point on the highest ground
    under him, and note where he stands."""
    yaw = person.yaw
    v = person.body.vertices(*pose_args(index, mirrored, (0.0, 0.0, 0.0), yaw))
    low, high = v[:, :2].min(axis=0), v[:, :2].max(axis=0)
    centre = (low + high) / 2.0
    corners = [(float(spot[0] + dx), float(spot[1] + dy)) for dx in (low[0] - centre[0], 0.0, high[0] - centre[0])
               for dy in (low[1] - centre[1], 0.0, high[1] - centre[1])]
    heights = [h for h in (site.ground_at(cli, ep.terrain_uids, x, y) for x, y in corners) if h is not None]
    ground = max(heights) if heights else float(ep.dock_position[2])
    origin = (float(spot[0] - centre[0]), float(spot[1] - centre[1]))
    person.body.pose(*pose_args(index, mirrored, (origin[0], origin[1], ground), yaw))
    person.xy, person.ground_z = (float(spot[0]), float(spot[1])), ground
    # Someone standing reaches his full height whatever he is doing; someone low, as high as his base pose.
    person.top_m = None if person.hub == moves.STAND else float(v[:, 2].max() - v[:, 2].min())
    if person.hub:
        life = np.random.default_rng([PEOPLE_SEED_STREAM, int(ep.seed), LIFE_STREAM, int(person.index)])
        person.idle = idlers.plan(life, person.hub, origin, yaw, mirrored, person.stride, person.company)
        person.idle.body, person.idle.ground_z = person.body, ground
        first = idlers.actor(person.idle).at(0.0)
        person.idle.pose_key = (first.index, first.mirrored, first.x, first.y, first.yaw)


def people(ep: FieldEpisode) -> List[Person]:
    """Everyone on the field this patrol, then everyone passing outside it."""
    return list(ep.people or ())


def walks(ep: FieldEpisode) -> List[walkers.Walk]:
    """The stroll of every walker on the field."""
    return [person.walk for person in people(ep) if person.walk is not None]


def passers(ep: FieldEpisode) -> List[outsiders.Pass]:
    """The walk of everyone passing outside the field."""
    return [person.passer for person in people(ep) if person.passer is not None]


def idles(ep: FieldEpisode) -> List[idlers.Idle]:
    """The life on the spot of everyone on the field who stays put."""
    return [person.idle for person in people(ep) if person.idle is not None]


def advance(env: Any, ep: FieldEpisode) -> None:
    """Move everyone to the step physics is about to run, and pose those a picture taken then can see: the walkers,
    the people who stay put and the passers."""
    walkers.advance(env, ep, walks(ep))
    idlers.advance(env, ep, idles(ep), walkers.shots(env, ep, ep.step + 1))
    outsiders.advance(env, ep, passers(ep))


def show(ep: FieldEpisode, view: Any) -> None:
    """Pose the people visible to the laser at the current step."""
    t = ep.step * SIM_DT
    shots = (view,)
    walkers.show(walks(ep), t, shots)
    idlers.show(idles(ep), t, shots)
    outsiders.show(passers(ep), t, shots)


def bodies(ep: FieldEpisode) -> Dict[int, int]:
    """Every drawn body of every person, on the field or passing outside it, mapped to the person's index."""
    return {int(uid): person.index for person in people(ep) if person.body for uid in person.body.bodies}


def position(ep: FieldEpisode, index: int, step: Optional[int] = None) -> np.ndarray:
    """A person's reference point and the ground under it at a step up to now, world metres east, north and up; where
    he is now when no step is given."""
    person = ep.people[index]
    if person.walk is not None:
        return walkers.position(ep, person.walk, step)
    if person.passer is not None:
        return outsiders.position(ep, person.passer, step)
    return np.array([person.xy[0], person.xy[1], person.ground_z], dtype=float)


def reference(ep: FieldEpisode, index: int, step: Optional[int] = None) -> np.ndarray:
    """A person's reference point at a step, metres east and north of the dock; where he is now when no step is
    given."""
    return position(ep, index, step)[:2] - np.asarray(ep.dock_position[:2], dtype=float)
