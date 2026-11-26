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

"""People who walk on the field, each inside a patch of his own, and stop for the drone.

A walker has a home, the centre of a disc that the people's placement keeps clear of the edge, the dock, the crowns
and everyone else; his reference point never leaves it. Most patches are PATCH_M wide: there he strolls in legs,
setting off and stopping, and pauses after each. Every such leg is as long as setting off and stopping carry him, and
its end is drawn from where he stands on the arc of points that far away which lie in the ring between inner_m and
MARK_M from home: so he never walks out of his patch, and from every end another leg is possible. Some patches are
wide (WIDE_PATCH_M): there he walks back and forth across a part of the field, up to WIDE_MARKS marks in a row without
stopping, each drawn over the patch but WIDE_EDGE_M inside its rim and a good way from the last. He walks one of the
gaits the motion library holds, never a run or a sneak, and when he pauses he does what people do standing, or
crouches as if to look at the ground. His start, gait, pace, first heading and every leg and pause come from his own
stream in the order he walks them.

Two may walk together: the second keeps the first's track a step to one side, in step with him, so they set off,
turn, pause and stop as one.

When the airborne drone comes within STOP_M of him across the ground he stops: on a wide patch at his next stride, on
a narrow one at the end of the short leg he is on. He looks up at it or waves, then stands about, and sets off again
only once it has been more than CALM_M away for CALM_S. Whatever he does standing leaves him exactly where he stood.
So the marks a stroller walks to are the seed's and when he walks depends on the flight, as an intruder's does in
Sentinel; someone crossing a wide patch is stopped where the drone finds him, so his later legs depend on the flight
too. A track is written only as far as the patrol's clock, and where its
walker stood at any step up to now can always be read back.

His body is posed, as Sentinel poses its intruders, only for a picture that can see it and only when his frame changed.
His reference point is the ground under his hips, which on an upright body is the middle of what he covers seen from
above.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

from swarm.constants import SIM_DT

from ..solar_patrol import camera, zoom
from ..solar_patrol.fixed_order import norm
from ..solar_patrol.theft import _in_view
from ..solar_patrol.theft_moves import STRETCH_RANGE, Actor, Route, Step
from . import figures, moves, site
from .episode import FieldEpisode
from .moves import library

PATCH_M = 3.0                          # a stroller's reference point never leaves this disc round his home
MARK_M = 2.5                           # every leg of his ends this near home, so his turns and stops stay in the patch
WIDE_PATCH_M = (5.0, 8.0)              # the disc of someone who walks back and forth across a part of the field
WIDE_SHARE = 0.5                       # how many walkers are given a wide patch, where the field has the room
WIDE_EDGE_M = 1.0                      # a wide patch's marks lie this far inside its rim, room for the turn after
WIDE_LEG_SHARE = 0.8                   # and at least this share of the marks' radius from where he stands
WIDE_MARKS = 3                         # the most marks he walks to in a row without stopping
MARK_TRIES = 64                        # draws of one mark before he simply walks straight across home
CORNER_COS = -0.7                      # between two marks walked in a row he turns no sharper than about 135 degrees
HALT_MOST_M = 6.0                      # caught by the drone, a wide walker stands within this of where it caught him
BESIDE_M = (0.85, 1.1)                 # how far apart two who walk together keep
LEG_SLACK_M = 0.1                      # a leg is this much longer than setting off and stopping carry him
INNER_SLACK_M = 0.3                    # no leg ends nearer home than a full leg can still be walked from
PACE = (0.8, 0.95)                     # how fast one walker plays his gait: as a share of the library's plain walk
FIRST_PAUSE_S = (0.0, 15.0)            # walkers set off at different times
PAUSE_S = (4.0, 20.0)
WIDE_PAUSE_S = (2.0, 10.0)             # someone crossing a wide patch pauses for less
CROUCH_SHARE = 0.3                     # pauses he crouches for, as if to look at the ground, and gets up
CROUCH_S = (2.0, 6.0)
STOP_M = 12.0                          # he stops when the airborne drone is this near across the ground
CALM_M = 18.0                          # and walks on once it has been further than this
CALM_S = 5.0                           # for this long
WAIT_S = 3.0                           # how long a stopped walker stands about before he looks again for the drone
GROUNDED = ("docked", "landed")        # the drone sits in its dock: nobody stops for it
WALK, STAND = "walk", "stand"
SET_OFF, STOP = f"stand_to_{WALK}", f"{WALK}_to_stand"   # the plain walk's own; another gait's are named after it


@dataclass
class Walk:
    """One walker's stroll and how it is going: what the seed drew for him and his own stream for the rest, then his
    track, his body and whether he is stopped for the drone."""

    home: Tuple[float, float]
    start: Tuple[float, float]
    rng: np.random.Generator           # his legs and pauses, drawn in the order he walks them
    first_pause_s: float
    yaw: float
    pace: float
    mirrored: bool
    scale: float                       # his build's stride times his size
    gait: str = WALK                   # his way of walking, one of those the motion library holds
    patch: float = PATCH_M             # the disc round his home his reference point never leaves
    life: Any = None                   # his own stream for what he does while the drone keeps him standing
    leader: Optional["Walk"] = None    # whom he walks with, keeping that walker's track a step to one side
    partner: Optional["Walk"] = None   # who walks with him
    offset: Tuple[float, float] = (0.0, 0.0)
    route: Any = None                  # the path of the leg he is on, which a stop for the drone keeps to
    actor: Any = None
    script: Any = None
    stopped: bool = False
    calm_since: Optional[float] = None
    body: Any = None
    cli: int = 0
    terrain_uids: frozenset = frozenset()
    dock_z: float = 0.0
    posed: Optional[Tuple[float, float, float]] = None
    pose_key: Optional[tuple] = None
    grounds: Dict[int, float] = field(default_factory=dict)


def set_off(gait: str) -> str:
    """The kind of move that sets off from standing into a gait."""
    return f"stand_to_{gait}"


def stop(gait: str) -> str:
    """The kind of move that stops out of a gait into standing."""
    return f"{gait}_to_stand"


def leg_m(scale: float, gait: str = WALK) -> float:
    """How far one leg carries a walker of a stride scale and a gait: setting off and stopping, and a little more."""
    lib = library()
    return (lib.reach(_take(set_off(gait), gait)) + lib.reach(_take(stop(gait), gait))) * scale + LEG_SLACK_M


def inner_m(scale: float, gait: str = WALK) -> float:
    """How near home a walker of a stride scale and a gait ends a leg at the nearest: from there a full leg still ends
    within MARK_M."""
    return leg_m(scale, gait) - MARK_M + INNER_SLACK_M


def _take(kind: str, gait: str = WALK) -> str:
    """The first take of a kind of block that starts from standing or from a gait's loop."""
    lib = library()
    for context in ({"hub": STAND}, {"loop": f"{gait}.0"}):
        takes = lib.kinds(context).get(kind)
        if takes:
            return takes[0]
    raise ValueError(f"the motion library has no {kind}")


def mark_m(patch: float) -> float:
    """How far from home the legs walked in a patch end at the furthest."""
    return MARK_M if patch <= PATCH_M else patch - WIDE_EDGE_M


def fits(scale: float, gait: str, patch: float) -> bool:
    """Whether a gait's leg can be walked inside a patch by a walker of a stride scale: in a stroller's, a full leg
    still ends within MARK_M from the nearest he ever stands to home; in a wide one, a leg is no longer than the
    marks' radius, and a walker the drone catches ends within HALT_MOST_M of where it caught him."""
    if patch <= PATCH_M:
        return inner_m(scale, gait) < MARK_M
    return leg_m(scale, gait) <= mark_m(patch) and halt_m(scale, gait) <= HALT_MOST_M


def halt_m(scale: float, gait: str) -> float:
    """The furthest a walker of a stride scale and a gait still goes once the drone has caught him walking: the
    longest of setting off or one stride of his loop at its longest stretch, and then his stop."""
    lib = library()
    stride = lib.reach(_take(set_off(gait), gait))
    loop = lib.blocks[_take(set_off(gait), gait)]["next"]["loop"]
    stride = max(stride, lib.reach(loop) * STRETCH_RANGE[1])
    return (stride + lib.reach(_take(stop(gait), gait))) * scale


def plan(rng: np.random.Generator, home: Sequence[float], scale: float, patch: float = PATCH_M) -> Walk:
    """A walker's walk from his own generator: his gait, where in his patch he starts, his first pause, heading, pace
    and mirroring; the generator goes with him for his legs and pauses, and one drawn from it for what he does while
    the drone keeps him standing."""
    home = np.asarray(home, dtype=float)
    # A gait whose leg would not fit the patch is not drawn; a wide patch no gait fits is walked as a stroller's.
    held = {gait: share for gait, share in moves.gaits(library()).items() if fits(scale, gait, patch)}
    if not held and patch > PATCH_M:
        patch = PATCH_M
        held = {gait: share for gait, share in moves.gaits(library()).items() if fits(scale, gait, patch)}
    keys = list(held) or [WALK]
    share = np.array([held.get(key, 1.0) for key in keys], dtype=float)
    gait = keys[int(rng.choice(len(keys), p=share / share.sum()))]
    angle = float(rng.uniform(-math.pi, math.pi))
    inner = max(inner_m(scale, gait), 0.0) if patch <= PATCH_M else 0.0
    radius = math.sqrt(float(rng.uniform(inner ** 2, mark_m(patch) ** 2)))
    start = home + radius * np.array([math.cos(angle), math.sin(angle)])
    walk = Walk(home=(float(home[0]), float(home[1])), start=(float(start[0]), float(start[1])), rng=rng,
                first_pause_s=float(rng.uniform(*FIRST_PAUSE_S)), yaw=float(rng.uniform(-math.pi, math.pi)),
                pace=float(rng.uniform(*PACE)), mirrored=bool(rng.random() < 0.5), scale=float(scale), gait=gait,
                patch=float(patch))
    walk.life = np.random.default_rng(int(rng.integers(0, 2 ** 62)))
    return walk


def plan_beside(leader: Walk, home: Sequence[float]) -> Walk:
    """The walk of someone who walks with another: the leader's own, from a home a step to one side of his."""
    offset = (float(home[0]) - leader.home[0], float(home[1]) - leader.home[1])
    walk = replace(leader, home=(float(home[0]), float(home[1])),
                   start=(leader.start[0] + offset[0], leader.start[1] + offset[1]), mirrored=not leader.mirrored,
                   leader=leader, partner=None, offset=offset, actor=None, script=None, body=None, grounds={}, route=None)
    leader.partner = walk
    return walk


def far_mark(rng: np.random.Generator, home: Sequence[float], here: Sequence[float], mark: float, least: float,
             came: Optional[Sequence[float]] = None) -> Optional[np.ndarray]:
    """Where a leg across a wide patch ends: a point drawn evenly over the disc mark metres round home, at least
    least metres from here. The first leg of a walk takes the point of that disc straight across home when
    MARK_TRIES draws give none. A leg that follows another without a stop (came is where the last one began) also
    turns no sharper than CORNER_COS allows, so the path never doubles back on itself, and is None when no draw
    fits: the walk then ends at here."""
    home, here = np.asarray(home, dtype=float), np.asarray(here, dtype=float)
    ahead = None
    if came is not None:
        ahead = here - np.asarray(came, dtype=float)
        ahead = ahead / max(norm(ahead), 1e-9)
    for _ in range(MARK_TRIES):
        angle = float(rng.uniform(-math.pi, math.pi))
        spot = home + mark * math.sqrt(float(rng.uniform(0.0, 1.0))) * np.array([math.cos(angle), math.sin(angle)])
        length = norm(spot - here)
        if length < least:
            continue
        if ahead is None or float(ahead[0] * (spot[0] - here[0]) + ahead[1] * (spot[1] - here[1])) >= CORNER_COS * length:
            return spot
    if ahead is not None:
        return None
    out = here - home
    d = norm(out)
    away = out / d if d > 1e-9 else np.array([1.0, 0.0])
    return home - mark * away


def next_mark(rng: np.random.Generator, home: Sequence[float], here: Sequence[float], leg: float,
              inner: float) -> np.ndarray:
    """Where the next leg ends: a point leg metres from here, drawn evenly over the arc of those between inner and
    MARK_M from home; straight back across home when no point of the arc lies there."""
    home, here = np.asarray(home, dtype=float), np.asarray(here, dtype=float)
    out = here - home
    d = norm(out)
    away = out / d if d > 1e-9 else np.array([1.0, 0.0])
    span = 2.0 * max(d, 1e-9) * leg
    hi = min(1.0, (MARK_M ** 2 - d * d - leg * leg) / span)
    lo = max(-1.0, (inner * inner - d * d - leg * leg) / span)
    phi = math.pi if lo > hi else float(rng.uniform(math.acos(hi), math.acos(lo))) * (1.0 if rng.random() < 0.5 else -1.0)
    c, s = math.cos(phi), math.sin(phi)
    return here + leg * np.array([away[0] * c - away[1] * s, away[0] * s + away[1] * c])


class Beside:
    """The track of someone who walks with another: the leader's own frames a step to one side. Walking he is in step
    with him, so his feet meet the ground the track carries him over; standing he is the leader's mirror image, so
    the two never do the same thing with the same hand."""

    def __init__(self, leader: Walk, offset: Sequence[float]):
        """Keep whose track it follows and how far to the side."""
        self.leader, self.offset = leader, (float(offset[0]), float(offset[1]))

    def _beside(self, frame: Any) -> Any:
        """A frame of the leader's track as the one beside him shows it."""
        mirrored = frame.mirrored if frame.stage == "walking" else not frame.mirrored
        return replace(frame, x=frame.x + self.offset[0], y=frame.y + self.offset[1], mirrored=mirrored)

    def at(self, t: float) -> Any:
        """The frame shown at patrol time t."""
        return self._beside(self.leader.actor.at(t))

    def frame(self, k: int) -> Any:
        """Frame k of the track, which the leader's must already hold."""
        return self._beside(self.leader.actor.frames[k])

    def run_to(self, count: int) -> None:
        """Have the leader's track written until it holds count frames."""
        self.leader.actor.run_to(count)


def actor(walk: Walk) -> Any:
    """A fresh track for a planned walk, standing at its start, nothing written yet; the walk keeps it and its
    script. Someone who walks with another gets the leader's track a step to one side."""
    if walk.leader is not None:
        walk.actor = Beside(walk.leader, walk.offset)
        return walk.actor
    lib = library()
    track = Actor(lib, walk.start, walk.yaw, scale=walk.scale, mirrored=walk.mirrored, pace=walk.pace)
    track.context = {"hub": STAND}
    # Standing still before the first block: the first frame of setting off is the standing pose it starts from.
    track.last = (int(lib.blocks[_take(set_off(walk.gait), walk.gait)]["first"]), "pause")
    walk.actor, walk.script = track, _stroll(track, walk)
    track.script = walk.script
    return track


def _stroll(track: Actor, walk: Walk) -> Iterator[Step]:
    """The walker's whole patrol: a first wait standing, then leg after leg."""
    yield from moves.standing(library(), walk.rng, walk.first_pause_s, walk.pace, "pause", track)
    yield from _legs(track, walk)


def _legs(track: Actor, walk: Walk) -> Iterator[Step]:
    """Leg after leg with a pause after each, standing about before a leg while he is stopped for the drone."""
    wide = walk.patch > PATCH_M
    leg, inner = leg_m(walk.scale, walk.gait), max(inner_m(walk.scale, walk.gait), 0.0)
    mark = mark_m(walk.patch)
    leave, arrive = set_off(walk.gait), stop(walk.gait)
    while True:
        yield from _stood(track, walk)
        points = [track.pos.copy()]
        if wide:
            for _ in range(int(walk.rng.integers(1, WIDE_MARKS + 1))):
                came = points[-2] if len(points) > 1 else None
                end = far_mark(walk.rng, walk.home, points[-1], mark, max(leg, WIDE_LEG_SHARE * mark), came)
                if end is None:
                    break
                points.append(end)
        else:
            points.append(next_mark(walk.rng, walk.home, track.pos, leg, inner))
        route = walk.route = Route(points)
        yield Step(leave, steer=route, stage="walking")
        if wide:
            # Whole strides at their own length, as many as fit before the stop, so no stride is stretched.
            lib = library()
            stride = lib.reach(track.context["loop"]) * walk.scale
            strides = int((route.left(track.pos) - lib.reach(_take(arrive, walk.gait)) * walk.scale) // stride)
            if strides > 0:
                yield Step("loop", cycles=strides, steer=route, stage="walking")
        else:
            yield Step("loop", cycles=None, steer=route, then=arrive, stage="walking")
        yield Step(arrive, steer=route, stage="walking")
        yield from _pause(track, walk)


def _pause(track: Actor, walk: Walk) -> Iterator[Step]:
    """The pause after a leg: squatting, busy with the ground, or standing for as long as the pause drawn, with one of
    the things people do standing where it fits in whole. He ends it standing exactly where the leg left him."""
    lib = library()
    crouch = moves.crouch(lib)
    if walk.rng.random() < CROUCH_SHARE and crouch:
        seconds = float(walk.rng.uniform(*CROUCH_S))
        yield from moves.anchored(track, (Step(kind, stage="pause", hold=seconds) for kind in crouch))
        return
    seconds = float(walk.rng.uniform(*(WIDE_PAUSE_S if walk.patch > PATCH_M else PAUSE_S)))
    habits = moves.habits(lib, moves.HABITS)
    if habits:
        habit = moves.pick(walk.rng, habits)
        length = moves.bout_seconds(lib, habit, 1, walk.pace)
        if length <= seconds:
            yield from moves.anchored(track, moves.bout(habit, 1, "pause"))
            seconds -= length
    yield from moves.standing(lib, walk.rng, seconds, walk.pace, "pause", track)


def _stood(track: Actor, walk: Walk) -> Iterator[Step]:
    """What a walker does while he is stopped for the drone: he looks up at it or waves, then stands about, never a
    statue, until it has gone, and is left standing exactly where he stopped. Drawn from his other stream, so the
    flight never changes the marks a stroller walks to."""
    lib = library()
    greeted = False
    while walk.stopped:
        greetings = {} if greeted else moves.habits(lib, moves.GREETINGS)
        greeted = True
        if greetings:
            yield from moves.anchored(track, moves.bout(moves.pick(walk.life, greetings), 1, "stopped"))
        else:
            yield from moves.standing(lib, walk.life, WAIT_S, walk.pace, "stopped", track)


def _halt(walk: Walk) -> Any:
    """What an actor is handed when the drone stops someone on a wide patch: at the point where his next stride can
    start, if the
    drone still holds him, he stops where he is and his legs begin again from there."""
    def chooser(track: Actor) -> Optional[Iterator[Step]]:
        """The stop and what follows, or nothing when he is not in his gait or is free again."""
        loop = (track.context or {}).get("loop")
        if not walk.stopped or loop is None or library().blocks[loop]["kind"] != walk.gait:
            return None

        def script() -> Iterator[Step]:
            """Out of his gait into standing along the path he was on, round a corner too, then his legs again."""
            yield Step(stop(walk.gait), steer=walk.route, stage="walking")
            yield from _legs(track, walk)

        return script()

    return chooser


def start(env: Any, ep: FieldEpisode, walk: Walk, dress: Dict[str, Any], size: float) -> None:
    """Put a planned walker on the field: his track, and his body dressed and standing where it begins."""
    track = actor(walk)
    walk.cli, walk.terrain_uids, walk.dock_z = env.CLIENT, ep.terrain_uids, float(ep.dock_position[2])
    frame = track.at(0.0)
    walk.body = figures.Figure(figures.catalogue(), dress, _pose_args(frame, ground(walk, 0)), env.CLIENT, size=size,
                               textures=figures.pictures(ep))
    # A body reads right in the thermal camera only once it has been rewritten: as first built, one side reads cold.
    walk.body.pose(*_pose_args(frame, ground(walk, 0)))
    walk.posed = (frame.x, frame.y, ground(walk, 0) + 0.9)
    walk.pose_key = (frame.index, frame.mirrored, frame.x, frame.y, frame.yaw)


def ground(walk: Walk, k: int) -> float:
    """The ground under a walker's hips in frame k of his track, read once per frame; the dock's height where no
    terrain lies under him."""
    if k not in walk.grounds:
        frame = walk.actor.frame(k) if walk.leader is not None else walk.actor.frames[k]
        height = site.ground_at(walk.cli, walk.terrain_uids, frame.x, frame.y)
        walk.grounds[k] = walk.dock_z if height is None else height
    return walk.grounds[k]


def _frame_number(t: float) -> int:
    """The 30 fps frame of a track shown at patrol time t."""
    return max(0, int(math.floor(t * library().fps + 1e-9)))


def position(ep: FieldEpisode, walk: Walk, step: Optional[int] = None) -> np.ndarray:
    """Where a walker's reference point stood at a step up to now, and the ground under it, world metres; now when no
    step is given."""
    t = (ep.step if step is None else min(int(step), ep.step)) * SIM_DT
    frame = walk.actor.at(t)
    return np.array([frame.x, frame.y, ground(walk, _frame_number(t))], dtype=float)


def _pose_args(frame: Any, ground_z: float) -> tuple:
    """The intruder pose for a frame of a track standing on the ground: local rotations, root height, place and yaw."""
    lib = library()
    rot = lib.rotations[int(frame.mirrored)][frame.index]
    root = lib.root[int(frame.mirrored)][frame.index] * np.array([0.0, 1.0, 0.0])
    return rot, root, (frame.x, frame.y, ground_z), frame.yaw


def advance(env: Any, ep: FieldEpisode, walks: Sequence[Walk]) -> None:
    """Write every walker's track to the current step, which the physics is about to run, stop or free each one by the
    drone's distance,
    and pose the ones a picture taken on that step can see."""
    if not walks:
        return
    step = ep.step + 1
    t = step * SIM_DT
    drone = np.asarray(env.pos[0], dtype=float)
    airborne = ep.phase not in GROUNDED
    for walk in walks:
        if walk.leader is not None:
            continue
        notice(walk, t, across_m(walk, walk.actor.at(t), drone), airborne)
    due = shots(env, ep, step)
    if due:
        show(walks, t, due)


def across_m(walk: Walk, frame: Any, drone: Sequence[float]) -> float:
    """How far across the ground the drone is from a walker in a frame of his track, or from the one who walks with
    him if that one is nearer: two who walk together stop as one."""
    across = math.hypot(float(drone[0]) - frame.x, float(drone[1]) - frame.y)
    if walk.partner is not None:
        across = min(across, math.hypot(float(drone[0]) - frame.x - walk.partner.offset[0],
                                        float(drone[1]) - frame.y - walk.partner.offset[1]))
    return across


def shots(env: Any, ep: FieldEpisode, step: int) -> list:
    """The views of the pictures taken on a control step: the wide or thermal frame and the zoom, those that are due."""
    return [shot for shot in (camera.upcoming(env, ep, step), zoom.upcoming(env, ep)) if shot is not None]


def notice(walk: Walk, t: float, across: float, airborne: bool) -> None:
    """Stop a walker once the airborne drone is within STOP_M across the ground, and free him once it has been further
    than CALM_M, or in its dock, for CALM_S; a stopped walker sets off on no other leg, and one on a wide patch stops
    at his next stride instead of walking his long leg out."""
    if airborne and across <= STOP_M:
        if not walk.stopped and walk.patch > PATCH_M and walk.actor is not None:
            walk.actor.interrupt(_halt(walk))
        walk.stopped, walk.calm_since = True, None
        return
    if not walk.stopped:
        return
    if airborne and across <= CALM_M:
        walk.calm_since = None
        return
    walk.calm_since = t if walk.calm_since is None else walk.calm_since
    if t - walk.calm_since >= CALM_S:
        walk.stopped, walk.calm_since = False, None


def show(walks: Sequence[Walk], t: float, shots: Sequence[Any]) -> None:
    """Pose every walker as he stands at time t whose new pose, or the one his body still shows, one of the pictures
    can see, and whose frame changed."""
    k = _frame_number(t)
    for walk in walks:
        frame = walk.actor.at(t)
        key = (frame.index, frame.mirrored, frame.x, frame.y, frame.yaw)
        if key == walk.pose_key:
            continue
        ground_z = ground(walk, k)
        now = (frame.x, frame.y, ground_z + 0.9)
        if not any(_in_view(shot, now) or (walk.posed is not None and _in_view(shot, walk.posed)) for shot in shots):
            continue
        walk.body.pose(*_pose_args(frame, ground_z))
        walk.posed, walk.pose_key = now, key


def track_xy(walk: Walk, seconds: float) -> np.ndarray:
    """East and north of a walker's reference point in every 30 fps frame of his first seconds, with no drone near."""
    track = walk.actor or actor(walk)
    count = _frame_number(seconds) + 1
    track.run_to(count)
    if walk.leader is not None:
        return np.array([[f.x, f.y] for f in (track.frame(k) for k in range(count))], dtype=float)
    return np.array([[f.x, f.y] for f in track.frames[:count]], dtype=float)
