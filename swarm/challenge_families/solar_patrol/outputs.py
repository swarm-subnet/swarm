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

"""Model outputs (task 8): every value the model sends, checked and applied to the drone and its camera.

Every value is held to its contract bounds before it is read. The move and turn sticks are flown through the drone's
velocity controller in its body frame, the speed across the ground capped as a whole and the velocity asked for eased
in at a fixed acceleration. The gimbal moves towards the tilt asked for at its top rate, night vision works only
through the 7x lens, and the camera, zoom and report values are then handed to the parts that own them.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from . import camera, reports, zoom
from .contract import (
    ACTION_HIGH,
    ACTION_LOW,
    GIMBAL_TILT_RANGE_DEG,
    MAX_CLIMB_MPS,
    MAX_DESCENT_MPS,
    MAX_GIMBAL_RATE_DEG_S,
    MAX_HORIZONTAL_MPS,
    MAX_YAW_RATE_DEG_S,
    MAX_ZOOMS,
    Command,
)
from .episode import Setpoint, SolarEpisode

MAX_ACCEL_MPS2 = 2.0                   # a step change of velocity would tip the body past the tilt limit
NIGHT_VISION_LENS = 7                  # DJI: night vision with the infrared light only at 7x zoom or above


def reset(env: Any, ep: SolarEpisode) -> None:
    """The velocity the controller holds starts at rest, the gimbal level, and no zoom lens in use."""
    ep.outputs = {"velocity": np.zeros(3), "tilt_deg": 0.0, "lens": 0}


def clip(action: np.ndarray) -> np.ndarray:
    """The action with every value held inside its contract bounds."""
    return np.clip(action, ACTION_LOW, ACTION_HIGH).astype(np.float32)


def apply(env: Any, ep: SolarEpisode, command: Command) -> None:
    """Hand the camera, zoom and report values to their parts, with the gimbal and night vision in their limits."""
    low, high = GIMBAL_TILT_RANGE_DEG
    target = low + (command.gimbal_tilt + 1.0) / 2.0 * (high - low)
    reach = MAX_GIMBAL_RATE_DEG_S * env.CTRL_TIMESTEP
    ep.outputs["tilt_deg"] += float(np.clip(target - ep.outputs["tilt_deg"], -reach, reach))
    # The zoom part refuses a press once the patrol's zooms are spent, and the lens in use stays the last one taken.
    if command.zoom is not None and ep.outcome.zooms_used < MAX_ZOOMS:
        ep.outputs["lens"] = command.zoom.lens
    command = replace(
        command,
        gimbal_tilt=(ep.outputs["tilt_deg"] - low) / (high - low) * 2.0 - 1.0,
        night_vision=command.night_vision and ep.outputs["lens"] == NIGHT_VISION_LENS,
    )
    camera.request(env, ep, command)
    zoom.request(env, ep, command)
    if command.report is not None:
        reports.submit(env, ep, command.report)


def setpoint(env: Any, ep: SolarEpisode, command: Command) -> Setpoint:
    """The model's sticks as a world velocity and a turn rate, with forward along the drone's heading."""
    yaw = float(env.rpy[0, 2])
    # The cap is on the speed across the ground, so a diagonal stick flies no faster than a straight one.
    share = max(1.0, math.hypot(command.move_forward, command.move_right))
    forward = command.move_forward / share * MAX_HORIZONTAL_MPS
    right = command.move_right / share * MAX_HORIZONTAL_MPS
    up = command.move_up * (MAX_CLIMB_MPS if command.move_up >= 0.0 else MAX_DESCENT_MPS)
    c, s = math.cos(yaw), math.sin(yaw)
    velocity = (forward * c + right * s, forward * s - right * c, up)
    # A clockwise turn on the compass is a negative yaw rate in the simulator's frame.
    return Setpoint(velocity, -math.radians(command.turn * MAX_YAW_RATE_DEG_S))


def fly(env: Any, ep: SolarEpisode, target: Setpoint) -> np.ndarray:
    """Rotor speeds that move the held velocity one step towards the setpoint, or all zero with the motors off."""
    velocity = ep.outputs["velocity"]
    if not target.motors_on:
        velocity[:] = 0.0
        return np.zeros((env.NUM_DRONES, 4))
    reach = MAX_ACCEL_MPS2 * env.CTRL_TIMESTEP
    velocity += np.clip(np.asarray(target.velocity_mps, dtype=float) - velocity, -reach, reach)
    yaw = float(env.rpy[0, 2]) + target.yaw_rate_rad_s * env.CTRL_TIMESTEP
    rpm, _, _ = env.ctrl[0].computeControl(
        control_timestep=env.CTRL_TIMESTEP,
        cur_pos=env.pos[0],
        cur_quat=env.quat[0],
        cur_vel=env.vel[0],
        cur_ang_vel=env.ang_v[0],
        target_pos=env.pos[0],
        target_rpy=np.array([0.0, 0.0, yaw]),
        target_vel=velocity,
        target_rpy_rates=np.array([0.0, 0.0, target.yaw_rate_rad_s]),
    )
    return np.asarray(rpm, dtype=float).reshape(env.NUM_DRONES, 4)
