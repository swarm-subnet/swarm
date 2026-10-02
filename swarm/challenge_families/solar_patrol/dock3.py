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

"""DJI Dock 3 (task 3): the dock body with its two lids, and DJI's own landing pad, as bodies in the world.

Two static bodies from swarm-worlds: dock3.urdf (body and lids, a solid the aircraft can hit) and dock3_pad.urdf
(the pad, the only part where a landing counts). The lids swing about hinges fitted to DJI's closed and open sizes.
The dock stands on the concrete base DJI's installation manual asks for (700 x 700 mm, at least 100 mm), poured
level at the high side of the ground under it and sunk into the low side, so it sits upright on any slope.
Where the dock stands and when it opens belong to the dock part (task 9); this module builds it and moves the lids.
The dock's surfaces keep the engine's passive temperatures, which follow the air.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import pybullet as p
import swarm_worlds

DOCK_URDF = "dock3.urdf"
PAD_URDF = "dock3_pad.urdf"
CLOSED_SIZE_M = (0.640, 0.745, 0.770)    # DJI: outer sizes with the RTK module, the wind gauge and the base
OPEN_SIZE_M = (1.760, 0.745, 0.485)
LID_TRAVEL_S = 4.0                       # DJI's footage: the lids move for about four seconds
REST_HEIGHT_M = 0.417                    # the aircraft's body origin above the ground when it rests on the pad (DJI CAD)
LID_JOINTS = (0, 1)                      # left and right hinges
BASE_SIZE_M = 0.70                       # DJI: concrete base at least 700 x 700 mm
BASE_ABOVE_M = 0.10                      # its top stands this far above the highest ground under it
BASE_BURIED_M = 0.10                     # and it reaches this far below the lowest
BASE_RGBA = (0.71, 0.70, 0.67, 1.0)


@dataclass
class Dock:
    """One placed dock: its two bodies, where it stands, and how far its lids are open (0 closed, 1 open)."""

    uid: int
    pad_uid: int
    base_uid: int
    origin: np.ndarray
    yaw: float
    opening: float = 0.0


def spawn(env: Any, x: float, y: float, yaw: float = 0.0) -> Dock:
    """Pour a level concrete base on the ground at (x, y) and stand a closed dock on it, the aircraft's nose along
    the yaw."""
    cli = env.CLIENT
    low, high = ground_under(env, x, y, yaw)
    top = high + BASE_ABOVE_M
    half = [BASE_SIZE_M / 2.0, BASE_SIZE_M / 2.0, (top - low + BASE_BURIED_M) / 2.0]
    orn = p.getQuaternionFromEuler([0.0, 0.0, float(yaw)])
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cli)
    look = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=list(BASE_RGBA), physicsClientId=cli)
    base = int(p.createMultiBody(0, shape, look, [float(x), float(y), top - half[2]], orn, physicsClientId=cli))
    origin = [float(x), float(y), float(top)]
    robots = swarm_worlds.robots_dir()
    uid = int(p.loadURDF(os.path.join(robots, DOCK_URDF), origin, orn, useFixedBase=True, physicsClientId=cli))
    pad = int(p.loadURDF(os.path.join(robots, PAD_URDF), origin, orn, useFixedBase=True, physicsClientId=cli))
    dock = Dock(uid=uid, pad_uid=pad, base_uid=base, origin=np.array(origin), yaw=float(yaw))
    set_opening(env, dock, 0.0)
    return dock


def ground_under(env: Any, x: float, y: float, yaw: float = 0.0) -> Tuple[float, float]:
    """Lowest and highest ground under the concrete base's footprint, from nine rays straight down."""
    c, s = math.cos(yaw), math.sin(yaw)
    half = BASE_SIZE_M / 2.0
    points = [(x + c * u - s * v, y + s * u + c * v) for u in (-half, 0.0, half) for v in (-half, 0.0, half)]
    hits = p.rayTestBatch([[a, b, 1000.0] for a, b in points], [[a, b, -1000.0] for a, b in points],
                          physicsClientId=env.CLIENT)
    z = [h[3][2] if h[0] >= 0 else 0.0 for h in hits]
    return float(min(z)), float(max(z))


def rest_pose(dock: Dock) -> Tuple[np.ndarray, float]:
    """World position and yaw of the aircraft's body when it rests on the pad."""
    return dock.origin + np.array([0.0, 0.0, REST_HEIGHT_M]), dock.yaw


def set_opening(env: Any, dock: Dock, opening: float) -> None:
    """Swing both lids to a share of their travel, easing in and out like the real actuators."""
    dock.opening = float(np.clip(opening, 0.0, 1.0))
    eased = dock.opening * dock.opening * (3.0 - 2.0 * dock.opening)
    for joint in LID_JOINTS:
        p.resetJointState(dock.uid, joint, eased * math.pi / 2.0, physicsClientId=env.CLIENT)


def move_lids(env: Any, dock: Dock, open_: bool, dt: float) -> bool:
    """Drive the lids one step towards open or closed; True once they are all the way there."""
    target = 1.0 if open_ else 0.0
    step = dt / LID_TRAVEL_S
    opening = dock.opening + float(np.clip(target - dock.opening, -step, step))
    # Lids at rest hold their hinge angles between steps, so writing them again would change nothing.
    if opening != dock.opening:
        set_opening(env, dock, opening)
    return dock.opening == target
