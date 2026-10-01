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

"""Theft scenarios (task 19): the motion library played as one thief's continuous track.

The library (maps/custom/solar/motions in swarm-worlds, task 7) is short blocks that start and end on a few base poses,
so a block may follow another when its start is the other's end. A script names the kinds of block in order; the
actor resolves each kind against where the body is now, plays it frame by frame at the library's 30 fps, carries the
body by the block's own root travel, and turns it towards a route at most 120 degrees a second, the way a person
walking turns. The track is written ahead of the patrol clock only as far as it is asked, so a reaction to the drone
can take over at the next seam without anything already played changing.

A loop walked to a mark has its strides lengthened or shortened by the same small share, so the body stops on the
mark instead of up to a stride off; spread over every cycle still to walk, the share stays under a fifth of a stride.
Each thief also plays the library at a pace of its own, which changes how fast he moves but never how far a block
carries him.
"""

from __future__ import annotations

import itertools
import json
import math
import os
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import swarm_worlds

ASSET_DIR = os.path.join("custom", "solar", "motions")
STEER_RAD_S = math.radians(120.0)       # the sharpest turn a walking or running body makes
MOVING_M_S = 0.2                        # a loop slower than this is worked on the spot and is never steered
UPRIGHT = 0.6                           # below this share of the hips' forward axis in the ground plane they face up or down
STRETCH_RANGE = (0.8, 1.25)             # how far a stride walked to a mark may be shortened or lengthened
GONE = -1                               # frame index of a body taken out of the scene


class Library:
    """The baked library: joint rotations and root per frame, the blocks and how they chain, and the mirrored copy."""

    def __init__(self, folder: str):
        """Read the arrays and the manifest, and build the mirrored arrays the manifest's rule describes."""
        with np.load(os.path.join(folder, "motions.npz")) as data:
            rotations = data["rotations"].astype(np.float64)
            root = data["root"].astype(np.float64)
        with open(os.path.join(folder, "motions.json"), encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.fps = int(manifest["fps"])
        self.blocks: Dict[str, Dict[str, Any]] = manifest["blocks"]
        swapped = rotations * np.array([1.0, -1.0, -1.0, 1.0])
        for a, b in manifest["mirror"]["pairs"]:
            swapped[:, [a, b]] = swapped[:, [b, a]]
        self.rotations = (rotations, swapped)
        self.root = (root, root * np.array([-1.0, 1.0, 1.0]))
        self.by_start: Dict[Tuple[str, str], Dict[str, List[str]]] = {}
        for name in sorted(self.blocks):
            block = self.blocks[name]
            self.by_start.setdefault(_key(block["start"]), {}).setdefault(block["kind"], []).append(name)

    def kinds(self, context: Optional[Dict[str, str]]) -> Dict[str, List[str]]:
        """Every kind of block that may follow a block ending in context, with its takes."""
        return self.by_start.get(_key(context), {}) if context else {}

    def travel(self, name: str, mirrored: bool) -> Tuple[float, float, float]:
        """Where a block leaves the body relative to its first frame: sideways, forward and the turn, in its own frame."""
        t = self.blocks[name]["travel"]
        if not t:
            root = self.root[int(mirrored)][self.blocks[name]["first"] + self.blocks[name]["frames"] - 1]
            return float(root[0]), float(root[2]), 0.0
        sign = -1.0 if mirrored else 1.0
        return sign * float(t["x"]), float(t["z"]), sign * float(t["turn"])

    def reach(self, name: str) -> float:
        """How far one play of a block carries the body, metres, before any build's scale."""
        x, z, _turn = self.travel(name, False)
        return math.hypot(x, z)

    def seconds(self, name: str) -> float:
        """How long one play of a block takes."""
        return self.blocks[name]["frames"] / self.fps


def _key(context: Optional[Dict[str, str]]) -> Tuple[str, str]:
    """A block's start or end as a hashable pair."""
    if not context:
        return ("", "")
    return ("loop", context["loop"]) if "loop" in context else ("hub", context["hub"])


@lru_cache(maxsize=1)
def library() -> Library:
    """The shipped library; raises FileNotFoundError when the installed swarm-worlds predates it."""
    folder = os.path.join(swarm_worlds.maps_dir(), ASSET_DIR)
    if not os.path.isfile(os.path.join(folder, "motions.json")):
        raise FileNotFoundError(f"motion library missing: {folder}")
    return Library(folder)


def hips_heading(quat: np.ndarray) -> Optional[float]:
    """The heading the hips face within their block, from the root joint's xyzw rotation; None when the hips face up
    or down, as a body lying on the ground does."""
    x, y, z, w = quat
    fx, fz = 2 * (x * z + w * y), 1 - 2 * (x * x + y * y)
    return math.atan2(fx, fz) if math.hypot(fx, fz) >= UPRIGHT else None


def heading_to(delta: Sequence[float]) -> float:
    """The engine yaw of a body facing along a ground direction; a body at yaw 0 faces -y."""
    return math.atan2(float(delta[0]), -float(delta[1]))


def direction(yaw: float) -> np.ndarray:
    """The ground direction a body at an engine yaw faces."""
    return np.array([math.sin(yaw), -math.cos(yaw)])


class Route:
    """A path of straight legs a body keeps to: it aims at the point ahead metres further along the path than it is,
    so it tracks the legs themselves, through a gap's middle and round a corner close in, instead of cutting across
    to the next waypoint. Progress along the path only ever moves forward. A single point is simply walked to."""

    def __init__(self, points: Sequence[Sequence[float]], ahead: float = 1.5):
        """The path's points in order, world metres, the first one where the walk starts, and how far ahead it aims."""
        self.points = np.asarray(points, dtype=float).reshape(-1, 2)
        self.ahead = ahead
        legs = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(legs, axis=1)
        self.dirs = legs / np.maximum(self.lengths, 1e-9)[:, None]
        self.starts = np.concatenate([[0.0], np.cumsum(self.lengths)])
        self.leg = 0
        self.done = 0.0

    def _progress(self, pos: np.ndarray) -> float:
        """How far along the path the body is: its nearest point on this leg or the next few, never going back."""
        if len(self.lengths) == 0:
            return 0.0
        best = None
        for k in range(self.leg, min(self.leg + 3, len(self.lengths))):
            t = float(np.clip((pos - self.points[k]) @ self.dirs[k], 0.0, self.lengths[k]))
            gap = float(np.linalg.norm(self.points[k] + self.dirs[k] * t - pos))
            if best is None or gap < best[0] - 1e-9:
                best = (gap, k, self.starts[k] + t)
        _gap, self.leg, along = best
        self.done = max(self.done, along)
        return self.done

    def _point(self, s: float) -> np.ndarray:
        """The point s metres along the path, the end if past it."""
        if s >= self.starts[-1] or len(self.lengths) == 0:
            return self.points[-1]
        k = int(np.searchsorted(self.starts, s, side="right")) - 1
        return self.points[k] + self.dirs[k] * (s - self.starts[k])

    def heading(self, pos: np.ndarray) -> float:
        """The heading towards the point ahead metres further along the path."""
        if len(self.lengths) == 0:
            return heading_to(self.points[0] - pos)
        return heading_to(self._point(self._progress(pos) + self.ahead) - pos)

    def left(self, pos: np.ndarray) -> float:
        """Distance still to walk along the path."""
        if len(self.lengths) == 0:
            return float(np.linalg.norm(self.points[0] - pos))
        return float(self.starts[-1] - self._progress(pos))


class Lane:
    """A straight line a body keeps to, aiming a few metres ahead on it, so drifting off the line turns it back on;
    distance left is measured along the line to its end point."""

    def __init__(self, start: Sequence[float], end: Sequence[float], ahead: float = 3.0):
        """The line from start towards end, and how far ahead the body aims."""
        self.start, self.end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
        self.axis = (self.end - self.start) / max(float(np.linalg.norm(self.end - self.start)), 1e-9)
        self.ahead = ahead

    def heading(self, pos: np.ndarray) -> float:
        """The heading towards the point ahead on the line."""
        along = float((pos - self.start) @ self.axis)
        return heading_to(self.start + self.axis * (along + self.ahead) - pos)

    def left(self, pos: np.ndarray) -> float:
        """Distance still to go along the line."""
        return float((self.end - pos) @ self.axis)


@dataclass
class Step:
    """One entry of a script.

    kind names the block to play next, resolved against where the body is now; "loop" repeats the loop the body is in.
    A loop plays cycles times, or with cycles None until steer's distance left is used up but for the reach of then,
    the kind that follows, its strides stretched so the stop lands on the mark. hold keeps the last frame for that many
    seconds; gone takes the body out of the scene. steer is a fixed heading, a Route or a Lane, and stage labels what the
    thief is doing.
    """

    kind: str
    cycles: Optional[int] = 1
    steer: Any = None
    stage: str = ""
    hold: float = 0.0
    then: Optional[str] = None           # the kind that follows a loop walked to a mark, whose reach it stops short by
    budget: int = 400                    # most cycles a loop walked to a mark may take, should the mark never come


@dataclass
class Frame:
    """One 30 fps frame of a track: the library frame shown (GONE when out of the scene), whether it is mirrored,
    where the body stands, its yaw, and what the thief is doing."""

    index: int
    mirrored: bool
    x: float
    y: float
    yaw: float
    stage: str


@dataclass
class Actor:
    """One thief's track, written frame by frame from a script as far as the patrol clock asks.

    The script is a generator of Steps; it is resumed each time the actor needs its next step, so it can look at where
    the body is and what time it is before choosing. A loop is played one cycle at a time and a hold a second at a
    time, the rest going back to the front of the queue, so the track is never written more than a cycle ahead.
    interrupt() hands the actor a chooser, called at the next seam where the body is in a loop, which is where every
    reaction in the library starts; the chooser may take what was left to do with take_rest() and play it after.
    """

    lib: Library
    xy: Tuple[float, float]
    yaw: float
    scale: float = 1.0
    mirrored: bool = False
    pace: float = 1.0
    script: Optional[Iterator[Step]] = None
    frames: List[Frame] = field(default_factory=list)
    context: Optional[Dict[str, str]] = None
    pending: Optional[Callable[["Actor"], Optional[Iterator[Step]]]] = None

    def __post_init__(self) -> None:
        """Start standing still where the script begins."""
        self.pos = np.asarray(self.xy, dtype=float)
        self.last = (GONE, "")
        self._queue: List[Step] = []

    @property
    def time_s(self) -> float:
        """Seconds of track written so far."""
        return len(self.frames) / self.lib.fps

    @property
    def gone(self) -> bool:
        """True once the body has been taken out of the scene."""
        return bool(self.frames) and self.frames[-1].index == GONE

    def at(self, t: float) -> Frame:
        """The frame shown at patrol time t, writing the track that far first."""
        k = max(0, int(math.floor(t * self.lib.fps + 1e-9)))
        self.run_to(k + 1)
        return self.frames[k]

    def run_to(self, count: int) -> None:
        """Write frames until the track holds count of them."""
        while len(self.frames) < count:
            step = self._next()
            if step is None:
                self._emit_hold(count - len(self.frames))
                return
            self._play(step)

    def interrupt(self, chooser: Callable[["Actor"], Optional[Iterator[Step]]]) -> None:
        """Ask for another script at the next seam where the body is in a loop; chooser gets the actor there and returns
        the new script, or None to carry on."""
        self.pending = chooser

    def take_rest(self) -> Iterator[Step]:
        """Everything still to do, the queue then the script, handed over and cleared."""
        queue, script = self._queue, self.script
        self._queue, self.script = [], None
        return itertools.chain(queue, script if script is not None else ())

    def _next(self) -> Optional[Step]:
        """The next step: a waiting reaction at a loop seam first, then the queue, then the script."""
        if self.pending is not None and self.context and "loop" in self.context:
            chooser, self.pending = self.pending, None
            script = chooser(self)
            if script is not None:
                self.script, self._queue = script, []
        if self._queue:
            return self._queue.pop(0)
        if self.script is None:
            return None
        try:
            return next(self.script)
        except StopIteration:
            self.script = None
            return None

    def _emit_hold(self, count: int) -> None:
        """Repeat the last frame count times."""
        index, stage = self.last
        for _ in range(count):
            self.frames.append(Frame(index, self.mirrored, float(self.pos[0]), float(self.pos[1]), self.yaw, stage))

    def _resolve(self, kind: str) -> str:
        """The block of a kind that may follow where the body is now."""
        if kind == "loop":
            if not self.context or "loop" not in self.context:
                raise ValueError(f"no loop to repeat after {self.context}")
            return self.context["loop"]
        takes = self.lib.kinds(self.context).get(kind)
        if not takes:
            raise ValueError(f"no {kind} block follows {self.context}")
        return takes[len(self.frames) % len(takes)]

    def _play(self, step: Step) -> None:
        """Write the frames of one step."""
        if step.kind == "gone":
            self.last = (GONE, step.stage)
            self.frames.append(Frame(GONE, self.mirrored, float(self.pos[0]), float(self.pos[1]), self.yaw, step.stage))
            self.script, self._queue = None, []
            return
        if step.kind == "hold":
            self.last = (self.last[0], step.stage or self.last[1])
            now = min(step.hold, 1.0)
            self._emit_hold(max(1, int(round(now * self.lib.fps))))
            if step.hold - now > 1e-9:
                self._queue.insert(0, replace(step, hold=step.hold - now))
            return
        name = self._resolve(step.kind)
        block = self.lib.blocks[name]
        if not block["loop"]:
            self._cycle(name, step, 1.0)
            self.context = block["next"]
            return
        if step.cycles is not None:
            if step.cycles <= 0:
                return
            self._cycle(name, step, 1.0)
            if step.cycles > 1:
                self._queue.insert(0, replace(step, cycles=step.cycles - 1))
        else:
            # Every cycle still to walk takes the same share of what is left, so once the share is set it holds and the
            # last cycle ends on the mark.
            stop = self.lib.reach(self._peek_then(name, step)) * self.scale
            cycle = self.lib.reach(name) * self.scale
            remaining = step.steer.left(self.pos) - stop
            if remaining < STRETCH_RANGE[0] * cycle or step.budget <= 0:
                return
            n = max(1, int(round(remaining / cycle)))
            self._cycle(name, step, float(np.clip(remaining / (n * cycle), *STRETCH_RANGE)))
            self._queue.insert(0, replace(step, budget=step.budget - 1))
        self.context = block["next"]

    def _peek_then(self, name: str, step: Step) -> str:
        """The block a paced loop stops for, so its reach can be kept in hand."""
        takes = self.lib.kinds({"loop": name}).get(step.then or "")
        if not takes:
            raise ValueError(f"a loop played to a mark needs the kind that follows it, got {step.then}")
        return takes[0]

    def _cycle(self, name: str, step: Step, stretch: float) -> None:
        """One play of a block at the actor's pace, its travel stretched by a share: each frame shown, the body carried
        by the root travel between frames, and turned towards the steer at most STEER_RAD_S."""
        block = self.lib.blocks[name]
        first, count = block["first"], block["frames"]
        rot = self.lib.rotations[int(self.mirrored)][first:first + count]
        root = self.lib.root[int(self.mirrored)][first:first + count][:, [0, 2]]
        tx, tz, turn = self.lib.travel(name, self.mirrored)
        if block["loop"] and math.hypot(tx, tz) / (count / self.lib.fps) <= MOVING_M_S:
            # A loop worked on the spot ends each cycle where it began: its centimetre of drift would carry a man on
            # watch for minutes into the table beside him.
            tx, tz, turn = float(root[0, 0]), float(root[0, 1]), 0.0
        path = np.vstack([root, [[tx, tz]]])
        out = max(1, int(round(count / self.pace)))
        spots = np.arange(out + 1) * (count / out)
        track = np.stack([np.interp(spots, np.arange(count + 1), path[:, k]) for k in (0, 1)], 1)
        steer_step = STEER_RAD_S / self.lib.fps
        loop_facing = None
        if block["loop"] and math.hypot(tx, tz) / (count / self.lib.fps) > MOVING_M_S:
            loop_facing = math.atan2(tx, tz)
        for j in range(out):
            f = min(count - 1, int(round(spots[j])))
            self.last = (first + f, step.stage)
            self.frames.append(Frame(first + f, self.mirrored, float(self.pos[0]), float(self.pos[1]), self.yaw,
                                     step.stage))
            dx, dz = (track[j + 1] - track[j]) * self.scale * stretch
            facing = loop_facing if block["loop"] else hips_heading(rot[f][0])
            if step.steer is not None and facing is not None:
                want = step.steer.heading(self.pos) if hasattr(step.steer, "heading") else float(step.steer)
                self.yaw += float(np.clip(math.remainder(want - self.yaw - facing, math.tau), -steer_step, steer_step))
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            self.pos = self.pos + np.array([dx * c + dz * s, dx * s - dz * c])
        self.yaw = math.remainder(self.yaw + turn, math.tau)
