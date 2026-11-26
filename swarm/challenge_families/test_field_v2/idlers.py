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

"""People who stay where they are: what they do on the spot, and how they greet the drone.

Someone who stays put is not a statue. Standing, he works through habits of his own drawn from the motion library
(the moves module): he shifts his weight, talks, looks at a phone, crosses his arms, stretches, each for a few cycles
with a moment just standing between two of them, which the library's small moves fill; someone who stands with others
mostly talks. Low, he squats, kneels or sits and keeps busy in that pose for the
whole patrol. His reference point never moves: only his body does, about where it was set down, and after every
thing he does standing it is back exactly there.

When the airborne drone comes within GREET_M across the ground, someone standing may look up at it or wave at the
next point where one of his moves ends and another can start, if it is still that near, and then goes back to his
habits; under a drone that stays
he does so again no sooner than GREET_AGAIN_S after he last began. A drone in its dock is greeted by nobody.

Every person has his own generator for his habits and pauses, drawn in the order he lives them, and his track is
written only as far as the patrol clock asks, so what the drone does changes nothing already played. A body is posed
only for a picture that sees it and only when its frame changed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from swarm.constants import SIM_DT

from ..solar_patrol.theft import _in_view
from ..solar_patrol.theft_moves import Actor, Step
from . import moves
from .episode import FieldEpisode

FIRST_PAUSE_S = (0.0, 6.0)             # people fall into their first habit at different times
PAUSE_S = (1.0, 6.0)                   # the quiet moment standing between two habits
CYCLES = (1, 3)                        # how many cycles of a habit's loop one bout lasts
LOW_CYCLES = 4                         # a low pose's loop is asked for this many cycles at a time, for ever
PACE = (0.85, 1.05)                    # how fast one person lives against the library
GREET_M = 15.0                         # the airborne drone this near across the ground may be looked up at or waved to
GREET_SHARE = 0.6                      # how many of those standing greet it at all
GREET_CYCLES = (1, 3)
GREET_AGAIN_S = 40.0                   # nobody greets the drone twice within this long
GROUNDED = ("docked", "landed")        # the drone sits in its dock: nobody greets it


@dataclass
class Idle:
    """One person who stays put: the base pose he holds or returns to, his own stream, his track, his body, and what
    of him is drawn now."""

    hub: str
    start: Tuple[float, float]         # where his body's own origin stands, world metres
    yaw: float
    mirrored: bool
    scale: float
    rng: np.random.Generator
    first_pause_s: float
    pace: float
    greets: bool
    company: bool = False              # he stands with others, so he mostly talks
    ground_z: float = 0.0
    actor: Any = None
    body: Any = None
    greeted_s: Optional[float] = None
    near: bool = False                 # set while the airborne drone is within GREET_M of him
    posed: Optional[Tuple[float, float, float]] = None
    pose_key: Optional[tuple] = None


def plan(rng: np.random.Generator, hub: str, start: Sequence[float], yaw: float, mirrored: bool, scale: float,
         company: bool = False) -> Idle:
    """One person's life on the spot from his own generator: when he first stirs, how fast he lives, whether he
    greets the drone; the generator goes with him for his habits and pauses."""
    return Idle(hub=hub, start=(float(start[0]), float(start[1])), yaw=float(yaw), mirrored=bool(mirrored), scale=float(scale),
                rng=rng, first_pause_s=float(rng.uniform(*FIRST_PAUSE_S)), pace=float(rng.uniform(*PACE)),
                greets=bool(rng.random() < GREET_SHARE), company=bool(company))


def actor(idle: Idle) -> Actor:
    """A fresh track for a planned life, holding its base pose at its start, nothing written yet."""
    lib = moves.library()
    track = Actor(lib, idle.start, idle.yaw, scale=idle.scale, mirrored=idle.mirrored, pace=idle.pace)
    track.context = {"hub": idle.hub}
    track.last = (moves.hub_frame(lib, idle.hub)[0], "still")
    track.script = _life(track, idle)
    idle.actor = track
    return track


def _life(track: Actor, idle: Idle) -> Iterator[Step]:
    """The whole patrol of someone who stays put: a first quiet moment, then habit after habit when he stands, or his
    low pose's loop for ever when he is low; someone the library has nothing for holds still."""
    lib = moves.library()
    if idle.hub != moves.STAND:
        yield Step("hold", hold=idle.first_pause_s, stage="still")
        lead_in = moves.low(lib).get(idle.hub)
        if not lead_in:
            return
        yield Step(lead_in, stage="low")
        while True:
            yield Step("loop", cycles=LOW_CYCLES, stage="low")
    yield from moves.standing(lib, idle.rng, idle.first_pause_s, idle.pace, "still", track)
    habits = moves.habits(lib, moves.COMPANY_HABITS if idle.company else moves.HABITS)
    if not habits and not moves.fidgets(lib):
        return
    while True:
        if habits:
            habit = moves.pick(idle.rng, habits)
            yield from moves.bout(habit, int(idle.rng.integers(CYCLES[0], CYCLES[1] + 1)), habit)
            _home(track, idle)
        yield from moves.standing(lib, idle.rng, float(idle.rng.uniform(*PAUSE_S)), idle.pace, "still", track)


def _home(track: Actor, idle: Idle) -> None:
    """Stand a track exactly where and as its person was set down: whatever he did, his body is back on his spot."""
    track.pos, track.yaw = np.asarray(idle.start, dtype=float), idle.yaw


def _greeting(idle: Idle) -> Any:
    """What an actor is handed when its person greets the drone: at the next point where a move of his habit ends and
    another can start,
    he leaves it, looks up or waves for a few cycles, and takes his habits up again."""
    def chooser(track: Actor) -> Optional[Iterator[Step]]:
        """The greeting, then the rest of his life, from where he is in his loop."""
        lib = moves.library()
        greetings = moves.habits(lib, moves.GREETINGS)
        loop = (track.context or {}).get("loop")
        if not greetings or loop is None or not idle.near:
            return None
        idle.greeted_s = track.time_s
        doing = lib.blocks[loop]["kind"]
        greeting = moves.pick(idle.rng, greetings)
        cycles = int(idle.rng.integers(GREET_CYCLES[0], GREET_CYCLES[1] + 1))

        def script() -> Iterator[Step]:
            """Out of the habit, the greeting, and back to his habits."""
            yield Step(f"{doing}_to_stand", stage=doing)
            _home(track, idle)
            yield from moves.bout(greeting, cycles, "greeting")
            _home(track, idle)
            yield from _life_after(track, idle)

        return script()

    return chooser


def _life_after(track: Actor, idle: Idle) -> Iterator[Step]:
    """His habits again after a greeting, with no first wait."""
    waited, idle.first_pause_s = idle.first_pause_s, 0.0
    life = _life(track, idle)
    try:
        first = next(life)
    except StopIteration:
        return
    finally:
        idle.first_pause_s = waited
    yield first
    yield from life


def notice(idle: Idle, t: float, across: float, airborne: bool) -> None:
    """Have someone standing greet the airborne drone once it is within GREET_M across the ground, if he is one who
    does and has not begun to within GREET_AGAIN_S. He greets at the next point where one of his moves ends and another
    can start, and only if the
    drone is still that near then."""
    idle.near = bool(airborne and across <= GREET_M)
    if not idle.greets or idle.hub != moves.STAND or not idle.near or idle.actor.pending is not None:
        return
    if idle.greeted_s is not None and t - idle.greeted_s < GREET_AGAIN_S:
        return
    idle.actor.interrupt(_greeting(idle))


def _frame_number(t: float) -> int:
    """The 30 fps frame of a track shown at patrol time t."""
    return max(0, int(math.floor(t * moves.library().fps + 1e-9)))


def pose_args(frame: Any, ground_z: float) -> tuple:
    """A body's pose for a frame of a track standing on the ground: local rotations, root height, place and yaw."""
    lib = moves.library()
    rot = lib.rotations[int(frame.mirrored)][frame.index]
    root = lib.root[int(frame.mirrored)][frame.index] * np.array([0.0, 1.0, 0.0])
    return rot, root, (frame.x, frame.y, ground_z), frame.yaw


def advance(env: Any, ep: FieldEpisode, idles: Sequence[Idle], shots: Sequence[Any]) -> None:
    """Write everyone's track to the step physics is about to run, let those standing notice the drone, and pose the
    ones a picture taken on that step can see."""
    if not idles:
        return
    t = (ep.step + 1) * SIM_DT
    drone = np.asarray(env.pos[0], dtype=float)
    airborne = ep.phase not in GROUNDED
    for idle in idles:
        frame = idle.actor.at(t)
        notice(idle, t, math.hypot(float(drone[0]) - frame.x, float(drone[1]) - frame.y), airborne)
    if shots:
        show(idles, t, shots)


def show(idles: Sequence[Idle], t: float, shots: Sequence[Any]) -> None:
    """Pose everyone as he is at time t whose body one of the pictures can see and whose frame changed."""
    for idle in idles:
        frame = idle.actor.at(t)
        key = (frame.index, frame.mirrored, frame.x, frame.y, frame.yaw)
        if key == idle.pose_key:
            continue
        now = (frame.x, frame.y, idle.ground_z + 0.9)
        if not any(_in_view(shot, now) for shot in shots):
            continue
        idle.body.pose(*pose_args(frame, idle.ground_z))
        idle.posed, idle.pose_key = now, key


def stages(idles: Sequence[Idle], t: float) -> List[str]:
    """What each person is doing at time t: the habit he is in, "greeting", "low" or "still"."""
    return [idle.actor.at(t).stage for idle in idles]
