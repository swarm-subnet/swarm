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

"""Dock take-off, return home and landing (task 9): where the dock stands, and the flights it makes by itself.

The dock owns the flight phase. While it flies the drone (take-off, the flight home, the landing) its setpoint
replaces the model's; once the drone is up, the model flies until it asks to come home.

Stand-in: the Dock 3 (task 3's model) at one fixed open spot. Take-off opens the lids, then climbs straight up to
the patrol height; return home climbs to it, flies straight back, and descends onto the pad.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pybullet as p

from . import dock3
from .contract import (
    FLIGHT_PHASES,
    MAX_CLIMB_MPS,
    MAX_DESCENT_MPS,
    MAX_HORIZONTAL_MPS,
    PATROL_HEIGHT_M,
    STATE_SLICES,
    Command,
    put,
)
from .episode import Setpoint, SolarEpisode

DOCK_XY = (60.0, 97.0)                 # an open patch between two rows, 18 m inside the fence
DRONE_REST_M = 0.0                     # dock_position is the aircraft's own resting point on the pad
ARRIVE_M = 0.5                         # close enough to a target height or to the point above the pad
LANDED_SPEED_MPS = 0.3                 # slower than this on the pad counts as landed
TOUCHDOWN_MPS = 0.4                    # the slowest descent, so the last centimetres still close
GAIN_PER_S = 1.0                       # speed asked for per metre still to go
CENTRED_M = 0.3                        # the descent pauses while the aircraft is further than this off the pad
BRAKE_MPS2 = 1.5                       # approach no faster than a stop at this deceleration allows, inside the setpoint's own 2 m/s2


def reset(env: Any, ep: SolarEpisode) -> None:
    """Stand the dock on the ground, set the drone down on its pad, and tell the environment the pad is a landing."""
    cli = env.CLIENT
    x, y = DOCK_XY
    model = dock3.spawn(env, x, y, yaw=0.0)
    pad, yaw = dock3.rest_pose(model)
    ep.dock_uid = model.uid
    ep.dock_position = pad
    ep.dock_yaw = yaw
    ep.phase = "docked"
    ep.dock = {"model": model}
    drone = int(env.DRONE_IDS[0])
    p.resetBasePositionAndOrientation(drone, (pad + [0.0, 0.0, DRONE_REST_M]).tolist(),
                                      p.getQuaternionFromEuler([0.0, 0.0, ep.dock_yaw]), physicsClientId=cli)
    p.resetBaseVelocity(drone, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=cli)
    # Only the pad takes a landing; the body and the lids are a solid the aircraft can hit.
    env._end_platform_uids = [model.pad_uid]
    env.task.start = tuple(float(v) for v in pad)
    env.task.goal = tuple(float(v) for v in pad)
    env.GOAL_POS = pad.copy()


def command(env: Any, ep: SolarEpisode, cmd: Command) -> None:
    """Act on the take-off, return home and cancel buttons, each only in the phase where the real dock takes it."""
    if cmd.take_off and ep.phase == "docked":
        ep.phase = "taking_off"
        ep.outcome.took_off = True
    elif cmd.return_home and ep.phase == "flying":
        ep.phase = "returning"
        ep.outcome.returned_by_model = True
    elif cmd.cancel_return and ep.phase in ("returning", "landing"):
        ep.phase = "flying"
        ep.outcome.returned_by_model = False


def _towards(delta: np.ndarray, limit: float) -> np.ndarray:
    """A velocity along delta, proportional to its length, capped at limit and slow enough to stop in time."""
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        return np.zeros_like(delta)
    return delta / distance * min(limit, GAIN_PER_S * distance, math.sqrt(2.0 * BRAKE_MPS2 * distance))


def autopilot(env: Any, ep: SolarEpisode) -> Optional[Setpoint]:
    """The dock's own setpoint in the phases it flies, or None while the model has the drone."""
    if ep.phase in ("docked", "landed"):
        return Setpoint(motors_on=False)
    if ep.phase == "flying":
        return None
    pos = np.asarray(env.pos[0], dtype=float)
    patrol_z = ep.dock_position[2] + PATROL_HEIGHT_M
    horizontal = _towards(ep.dock_position[:2] - pos[:2], MAX_HORIZONTAL_MPS)
    if ep.phase == "taking_off" and ep.dock["model"].opening < 1.0:
        return Setpoint(motors_on=False)
    if ep.phase == "taking_off":
        vz = float(np.clip(GAIN_PER_S * (patrol_z - pos[2]), -MAX_DESCENT_MPS, MAX_CLIMB_MPS))
        if abs(patrol_z - pos[2]) < ARRIVE_M:
            ep.phase = "flying"
        return Setpoint((float(horizontal[0]), float(horizontal[1]), vz))
    if ep.phase == "returning":
        vz = float(np.clip(GAIN_PER_S * (patrol_z - pos[2]), -MAX_DESCENT_MPS, MAX_CLIMB_MPS))
        if pos[2] < patrol_z - ARRIVE_M:
            return Setpoint((0.0, 0.0, vz))
        if math.hypot(*(ep.dock_position[:2] - pos[:2])) < ARRIVE_M:
            ep.phase = "landing"
        return Setpoint((float(horizontal[0]), float(horizontal[1]), vz))
    height = pos[2] - DRONE_REST_M - ep.dock_position[2]
    vz = -min(MAX_DESCENT_MPS, max(TOUCHDOWN_MPS, GAIN_PER_S * height))
    if math.hypot(*(ep.dock_position[:2] - pos[:2])) > CENTRED_M:
        vz = 0.0
    return Setpoint((float(horizontal[0]), float(horizontal[1]), vz))


def update(env: Any, ep: SolarEpisode) -> None:
    """Open the lids for the flight, shut them at rest, hold a resting aircraft still on the pad, and close the patrol
    once a landing drone rests on the pad."""
    model = ep.dock["model"]
    dock3.move_lids(env, model, ep.phase not in ("docked", "landed"), float(env.CTRL_TIMESTEP))
    if ep.phase == "docked" or (ep.phase == "taking_off" and model.opening < 1.0):
        _hold(env, ep)
    if ep.phase != "landing" or not env._platform_hit:
        return
    if float(np.linalg.norm(env.vel[0])) < LANDED_SPEED_MPS:
        ep.phase = "landed"
        ep.outcome.landed_in_dock = True
        ep.end("landed")


def _hold(env: Any, ep: SolarEpisode) -> None:
    """Keep the aircraft on its resting pose while its motors are off: the pad's V holds the real one still, where
    four small feet in the pad's V slowly creep at the patrol's physics step."""
    drone = int(env.DRONE_IDS[0])
    orn = p.getQuaternionFromEuler([0.0, 0.0, ep.dock_yaw])
    p.resetBasePositionAndOrientation(drone, ep.dock_position.tolist(), orn, physicsClientId=env.CLIENT)
    p.resetBaseVelocity(drone, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=env.CLIENT)


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The flight phase, as the dock reports it."""
    put(state, STATE_SLICES, "flight_phase", FLIGHT_PHASES.index(ep.phase))
