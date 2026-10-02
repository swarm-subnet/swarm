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

"""Laser point (task 12): the rangefinder in the camera, one distance at the crosshair, once a second.

The beam is read from the renderer, not the physics: intruders and tree crowns are drawn but have no collision, and
the laser must find what the camera shows. A coarse read finds the first surface on the beam; a tight read around it
gives the distance and the surface's slope, which sets DJI's shorter range for a slanted hit. Each normal reading
then carries DJI's range error, drawn from the seed so every validator sees the same one.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pybullet as p

from swarm.constants import SIM_DT

from . import airframe
from .contract import LASER_STATUSES, STATE_SLICES, put
from .episode import SolarEpisode
from .fixed_order import dot, norm

READING_HZ = 1.0
READING_STEPS = int(round(1.0 / (READING_HZ * SIM_DT)))
BLIND_ZONE_M = 1.0                      # DJI: blind zone
MAX_RANGE_M = 1800.0                    # DJI: 1 Hz, 20 % reflectivity target
OBLIQUE_RANGE_M = 600.0                 # DJI: oblique incidence range, 1:5
OBLIQUE_SLOPE = 1.0 / 5.0               # the beam meets the surface at this rise over run, or shallower
SEARCH_M = 2000.0                       # past the range, so a surface too far is told apart from open sky
COARSE_SIZE, COARSE_TAN, COARSE_NEAR_M = 2, 1e-4, 0.005
FINE_SIZE, FINE_SPACING_M = 4, 0.1      # neighbour pixels this far apart on the surface, for its slope
FLAGS = getattr(p, "ER_SWARM_RAYCAST", 0) | getattr(p, "ER_ALPHA_CUTOUT", 0)
ERROR_M, ERROR_SHARE = 0.2, 0.0015      # DJI: range accuracy +-(0.2 m + 0.15 % of the distance)
NOISE_SEED_STREAM = 0x1A5E              # the laser's own stream, so its error never moves another part's draws


def reset(env: Any, ep: SolarEpisode) -> None:
    """No reading taken yet: the aircraft's pose is only current from the first control step."""
    ep.laser = {"taken_step": None, "status": "no_signal", "range_m": 0.0, "point": np.zeros(3),
                "origin": np.zeros(3), "direction": np.zeros(3), "body": -1}


def update(env: Any, ep: SolarEpisode) -> None:
    """Take a reading on the first control step, then once a second."""
    taken = ep.laser["taken_step"]
    if taken is None or ep.step - taken >= READING_STEPS:
        ep.laser = with_error(read(env, float(ep.camera.get("tilt_deg", 0.0))), ep.seed, ep.step)
        ep.laser["taken_step"] = ep.step


def with_error(reading: dict, seed: int, step: int) -> dict:
    """A normal reading moved along its beam by DJI's range error, drawn from the seed and the step: normal, with the
    stated accuracy as two standard deviations and never past it."""
    if reading["status"] != "normal":
        return reading
    bound = ERROR_M + ERROR_SHARE * reading["range_m"]
    rng = np.random.default_rng([NOISE_SEED_STREAM, int(seed), int(step)])
    error = float(np.clip(rng.normal(0.0, bound / 2.0), -bound, bound))
    distance = reading["range_m"] + error
    return dict(reading, range_m=distance, point=reading["origin"] + reading["direction"] * distance, error_m=error)


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The last reading: its distance and the point it hit from the dock, both zero unless the status is normal."""
    normal = ep.laser["status"] == "normal"
    put(state, STATE_SLICES, "laser_range_m", ep.laser["range_m"] if normal else 0.0)
    put(state, STATE_SLICES, "laser_point_m", ep.laser["point"] - ep.dock_position if normal else (0.0, 0.0, 0.0))
    put(state, STATE_SLICES, "laser_status", LASER_STATUSES.index(ep.laser["status"]))


def read(env: Any, tilt_deg: float) -> dict:
    """One reading along the beam at this gimbal tilt: status, distance, world hit point and the body hit."""
    origin, direction = airframe.laser_pose(env, tilt_deg)
    direction = direction / norm(direction)
    reading = {"origin": origin, "direction": direction, "range_m": 0.0, "point": np.zeros(3), "body": -1}
    beam = _Beam(env, origin, direction)
    coarse = beam.render(COARSE_SIZE, COARSE_TAN, COARSE_NEAR_M, SEARCH_M)
    if coarse is None:
        return dict(reading, status="no_signal")
    distance, body = coarse.on_beam()
    if distance < BLIND_ZONE_M:
        return dict(reading, status="too_close", body=body)
    margin = 0.1 * distance + 0.5
    fine = beam.render(FINE_SIZE, FINE_SPACING_M / distance * FINE_SIZE / 2.0,
                       max(COARSE_NEAR_M, distance - margin), distance + margin)
    if fine is None:
        return dict(reading, status="no_signal")
    distance, body = fine.on_beam()
    reading.update(range_m=distance, point=origin + direction * distance, body=body)
    slope = fine.slope()
    limit = OBLIQUE_RANGE_M if slope is not None and slope <= OBLIQUE_SLOPE else MAX_RANGE_M
    return dict(reading, status="normal" if distance <= limit else "too_far")


class _Beam:
    """Tiny renders looking down the beam. Both engine renderers sample a pixel at its corner, so in an image of even
    size the pixel at row size / 2 - 1, column size / 2 lies exactly on the beam."""

    def __init__(self, env: Any, origin: np.ndarray, direction: np.ndarray):
        """Frame the beam: its origin, direction, and an up and right square to it."""
        self.env, self.origin, self.direction = env, origin, direction
        ref = np.array([1.0, 0.0, 0.0]) if abs(direction[2]) > 0.99 else np.array([0.0, 0.0, 1.0])
        up = ref - direction * dot(ref, direction)
        self.up = up / norm(up)
        self.right = np.cross(direction, self.up)

    def render(self, size: int, tan_half: float, near: float, far: float) -> Optional["_Shot"]:
        """A size x size depth and body-id render with this half-angle tangent; None when the beam hits nothing."""
        view = p.computeViewMatrix(self.origin.tolist(), (self.origin + self.direction).tolist(), self.up.tolist())
        projection = p.computeProjectionMatrixFOV(math.degrees(2.0 * math.atan(tan_half)), 1.0, near, far)
        _w, _h, _rgb, raw, seg = p.getCameraImage(size, size, view, projection, renderer=p.ER_TINY_RENDERER,
                                                  flags=FLAGS, physicsClientId=self.env.CLIENT)
        raw = np.asarray(raw, dtype=np.float64).reshape(size, size)
        body = np.where(raw >= 1.0, -1, np.asarray(seg, dtype=np.int64).reshape(size, size))
        shot = _Shot(self, size, tan_half, far * near / (far - (far - near) * raw), body)
        return None if shot.body[shot.row, shot.col] < 0 else shot


class _Shot:
    """One tiny render: the depth along the view axis and the body under each pixel."""

    def __init__(self, beam: _Beam, size: int, tan_half: float, depth: np.ndarray, body: np.ndarray):
        """Keep the render and where the beam falls in it."""
        self.beam, self.size, self.tan_half, self.depth, self.body = beam, size, tan_half, depth, body
        self.row, self.col = size // 2 - 1, size // 2

    def on_beam(self) -> tuple[float, int]:
        """Distance along the beam and the body it hit."""
        return float(self.depth[self.row, self.col]), int(self.body[self.row, self.col])

    def point(self, row: int, col: int) -> Optional[np.ndarray]:
        """The surface point under one pixel, from the laser, or None when the pixel shows another body."""
        if self.body[row, col] != self.body[self.row, self.col]:
            return None
        x = (2.0 * col / self.size - 1.0) * self.tan_half
        y = (1.0 - (2.0 * row + 2.0) / self.size) * self.tan_half
        b = self.beam
        return self.depth[row, col] * (b.direction + x * b.right + y * b.up)

    def slope(self) -> Optional[float]:
        """Rise over run of the beam against the surface it hit; None when the neighbouring pixels miss that body."""
        centre = self.point(self.row, self.col)
        across = [v for v in (self.point(self.row, self.col + 1), self.point(self.row, self.col - 1)) if v is not None]
        along = [v for v in (self.point(self.row - 1, self.col), self.point(self.row + 1, self.col)) if v is not None]
        if not across or not along:
            return None
        normal = np.cross(across[0] - centre, along[0] - centre)
        length = norm(normal)
        if length == 0.0:
            return None
        sin_angle = min(1.0, abs(dot(self.beam.direction, normal)) / length)
        return sin_angle / max(math.sqrt(1.0 - sin_angle * sin_angle), 1e-9)
