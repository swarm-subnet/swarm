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

"""Coverage cells (task 22): the park split into cells, and the share of them the camera saw closely enough.

Each seed lays a grid of 2 m cells over the park, inside the fence only, with its own offset and a small turn so the
cell edges are never the same twice. Every cell sits on the ground under its centre, read once with rays that pass
through the tables, the dock and anything else standing, down to the terrain.

A cell is searched when the colour or thermal frame the model was shown holds it and saw it closely enough:

- the frame can see: a colour frame in the dark with night mode off shows nothing;
- it was taken from 20 m or lower above the dock, and above the top of the dock, so a drone that never leaves marks
  nothing;
- the cell's centre falls inside the frame;
- a person in the cell would be at least PERSON_MIN_PX pixels wide: one pixel of the frame spans no more than
  PERSON_WIDTH_M / PERSON_MIN_PX of ground there, stretched by how far away and how slanted the ground is seen.

Zoom close-ups mark nothing: coverage comes from flying the park. The share of cells searched goes into the
outcome's coverage. Geometry only, no extra render.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
import pybullet as p
import shapely
from shapely.geometry import Polygon

from . import camera, dock3, sensor_noise
from .camera import View
from .contract import PATROL_HEIGHT_M
from .episode import SolarEpisode

CELL_M = 2.0
MAX_TURN_DEG = 10.0                    # the grid turns by up to this much either way each seed
MAX_HEIGHT_M = PATROL_HEIGHT_M         # seen from 20 m or lower, above the dock
DOCK_TOP_M = dock3.CLOSED_SIZE_M[2] - dock3.REST_HEIGHT_M   # the closed dock's top over the aircraft's resting point
PERSON_WIDTH_M = 0.5                   # shoulder width, what a person shows from above
PERSON_MIN_PX = 6.0                    # Johnson's criterion to recognise an object: 6 pixels across it
MAX_PIXEL_M = PERSON_WIDTH_M / PERSON_MIN_PX
GRID_SEED_STREAM = 0xC022              # the grid's own stream, so its draws never move another part's
RAY_REACH_M = 1000.0
MAX_HITS = 32                          # surfaces a ground ray may cross before the terrain


def grid(seed: int, fence: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """The centres of this seed's cells inside the fence, east and north in world metres, with the grid's turn in
    degrees and its offset in metres along its own axes."""
    rng = np.random.default_rng([GRID_SEED_STREAM, int(seed)])
    turn = float(rng.uniform(-MAX_TURN_DEG, MAX_TURN_DEG))
    offset = rng.uniform(0.0, CELL_M, size=2)
    c, s = math.cos(math.radians(turn)), math.sin(math.radians(turn))
    anchor = fence.min(axis=0)
    east, north = fence[:, 0] - anchor[0], fence[:, 1] - anchor[1]
    local = np.column_stack([east * c + north * s, north * c - east * s])       # the fence along the grid's axes
    low = np.floor((local.min(axis=0) - offset) / CELL_M) * CELL_M + offset + CELL_M / 2.0
    u, v = np.meshgrid(np.arange(low[0], local[:, 0].max(), CELL_M), np.arange(low[1], local[:, 1].max(), CELL_M))
    u, v = u.ravel(), v.ravel()
    centres = np.column_stack([anchor[0] + (u * c - v * s), anchor[1] + (u * s + v * c)])
    return centres[shapely.contains_xy(Polygon(fence), centres[:, 0], centres[:, 1])], turn, offset


def ground(cli: int, xy: np.ndarray, terrain_uids: frozenset) -> Tuple[np.ndarray, np.ndarray]:
    """The terrain point and its upward unit normal under each east, north point, NaN where no terrain lies below.

    Each ray reports every surface it crosses, so the terrain under a table, the dock or a person is found without
    moving anything.
    """
    points = np.full((len(xy), 3), np.nan)
    normals = np.full((len(xy), 3), np.nan)
    live = list(range(len(xy)))
    for number in range(MAX_HITS):
        if not live:
            break
        hits = _cast(cli, xy[live], number)
        for i, (uid, _link, _fraction, position, normal) in zip(live, hits):
            if uid in terrain_uids and (np.isnan(points[i, 2]) or position[2] > points[i, 2]):
                points[i], normals[i] = position, normal
        # Only a ray that met something at this hit number can meet anything further down.
        live = [i for i, hit in zip(live, hits) if hit[0] >= 0]
    normals *= np.where(normals[:, 2:] < 0.0, -1.0, 1.0)
    normals /= np.sqrt(_dot(normals, normals))[:, None]
    return points, normals


def _dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise dot products of 3-vectors, summed in a fixed order so every validator rounds them the same way."""
    return a[:, 0] * b[..., 0] + a[:, 1] * b[..., 1] + a[:, 2] * b[..., 2]


def _cast(cli: int, xy: np.ndarray, number: int) -> list:
    """Vertical rays through each east, north point, each reporting the given hit number along it."""
    batch = int(getattr(p, "MAX_RAY_INTERSECTION_BATCH_SIZE", 256))
    hits = []
    for start in range(0, len(xy), batch):
        rays = xy[start:start + batch]
        hits += p.rayTestBatch([[x, y, RAY_REACH_M] for x, y in rays], [[x, y, -RAY_REACH_M] for x, y in rays],
                               reportHitNumber=number, numThreads=0, physicsClientId=cli)
    return hits


def reset(env: Any, ep: SolarEpisode) -> None:
    """Lay this seed's grid over the park and put every cell on the ground under it."""
    centres, turn, offset = grid(ep.seed, ep.fence)
    points, normals = ground(env.CLIENT, centres, ep.terrain_uids)
    found = ~np.isnan(points[:, 2])
    ep.coverage = {"cells": points[found], "normals": normals[found], "searched": np.zeros(int(found.sum()), bool),
                   "turn_deg": turn, "offset_m": offset, "last_view": None}
    ep.outcome.coverage = 0.0


def update(env: Any, ep: SolarEpisode) -> None:
    """Mark the cells the frame the model was last shown saw closely enough, once per frame."""
    cov = ep.coverage
    shot = sensor_noise.shown_view(ep, "feed") or camera.view(ep)
    if shot is None or shot is cov["last_view"]:
        return
    cov["last_view"] = shot
    cov["searched"] |= seen_closely(shot, cov["cells"], cov["normals"], ep.dock_position)
    ep.outcome.coverage = float(cov["searched"].mean()) if cov["searched"].size else 0.0


def seen_closely(shot: View, cells: np.ndarray, normals: np.ndarray, dock_position: np.ndarray) -> np.ndarray:
    """Which cells a frame saw closely enough, as a mask over the cells."""
    height = shot.eye[2] - float(dock_position[2])
    if not shot.sees or not DOCK_TOP_M < height <= MAX_HEIGHT_M:
        return np.zeros(len(cells), bool)
    return pixel_ground_m(shot, cells, normals) <= MAX_PIXEL_M


def pixel_ground_m(shot: View, cells: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """The side of the patch of ground one pixel of the frame covers at each cell, infinite where the cell is out of
    the frame or its ground faces away from the camera."""
    forward = np.asarray(shot.forward, dtype=float)
    up = np.asarray(shot.up, dtype=float)
    right = np.cross(forward, up)
    offsets = cells - np.asarray(shot.eye, dtype=float)
    depth = _dot(offsets, forward)
    half_up = math.tan(math.radians(shot.vertical_fov_deg) / 2.0)
    half_across = half_up * shot.width / shot.height
    ahead = depth > camera.NEAR_M
    safe = np.where(ahead, depth, 1.0)
    in_frame = (ahead & (np.abs(_dot(offsets, right)) <= half_across * safe)
                & (np.abs(_dot(offsets, up)) <= half_up * safe))
    distance = np.sqrt(_dot(offsets, offsets))
    facing = -_dot(offsets, normals) / np.maximum(distance, 1e-9)
    # One pixel spans depth * pitch across the frame; seen off the axis and on slanted ground, its patch of ground
    # grows by cos(off-axis) / cos(incidence).
    pitch = 2.0 * half_up / shot.height
    area = (safe * pitch) ** 2 * (safe / np.maximum(distance, 1e-9)) / np.maximum(facing, 1e-9)
    return np.where(in_frame & (facing > 0.0), np.sqrt(area), np.inf)
