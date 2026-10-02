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

"""Drone state inputs (task 14): where the drone is and how it is doing every step, and the site map at the start.

Positions count from the dock, so the dock is at 0, 0, 0 wherever the seed stands it. The site map is the survey
of the park as built, read from the map's manifest, so a seed's own small shifts of each piece never reach it.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np
import pybullet as p

from swarm.core.maps.solar.builder import solar_manifest

from . import park
from .contract import HORIZON_S, MAX_BUILDINGS, MAX_FENCE_POINTS, MAX_TABLES, SITE_MAP_SLICES, STATE_SLICES, put
from .episode import SolarEpisode
from .fixed_order import norm, rows_dot

# DJI's M4TD figures: 47 min of hover on one battery, and the Dock 3 fast charge stops at 95 %.
HOVER_ENDURANCE_S = 47.0 * 60.0
BATTERY_START_PCT = 95.0
# A panel table is three pieces on one pose; its outline covers all three.
TABLE_PARTS = {"full_table_frame": ("full_table_frame", "full_table_glass", "full_table_racking"),
               "half_table_frame": ("half_table_frame", "half_table_glass", "half_table_racking")}
BUILDINGS = ("white_unit_north", "white_unit_south", "service_object")


def _footprint(low: np.ndarray, high: np.ndarray, place: dict[str, Any]) -> np.ndarray:
    """A placed piece seen from above: centre east, centre north, length, width, and the compass heading of its
    long side from 0 to 180 degrees."""
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    rotation = np.reshape(p.getMatrixFromQuaternion(place["quaternion"]), (3, 3))
    scaled = corners * place["scale"]
    flat = np.stack([rows_dot(scaled, rotation[0]), rows_dot(scaled, rotation[1])], 1)
    along = rotation[:2, 0] / norm(rotation[:2, 0])
    axes = (along, np.array([-along[1], along[0]]))
    spans = [rows_dot(flat, axis) for axis in axes]
    extents = [float(span.max() - span.min()) for span in spans]
    centre = np.asarray(place["position"][:2], dtype=float)
    for axis, span in zip(axes, spans):
        centre = centre + axis * (span.max() + span.min()) / 2.0
    long = int(extents[1] > extents[0])
    heading = math.degrees(math.atan2(axes[long][0], axes[long][1])) % 180.0
    return np.array([centre[0], centre[1], extents[long], extents[1 - long], heading])


@lru_cache(maxsize=2)
def survey(asset_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """The panel tables and the buildings where the map places them, world metres, five numbers each as the site
    map carries them."""
    manifest = solar_manifest(asset_dir)
    items = manifest["items"]
    outlines = {name: (items[name]["bounds_min"], items[name]["bounds_max"]) for name in BUILDINGS}
    for frame, parts in TABLE_PARTS.items():
        outlines[frame] = (np.min([items[part]["bounds_min"] for part in parts], axis=0),
                           np.max([items[part]["bounds_max"] for part in parts], axis=0))
    tables, buildings = [], []
    for place in manifest["placements"]:
        if place["item"] in outlines:
            rows = tables if place["item"] in TABLE_PARTS else buildings
            rows.append(_footprint(*outlines[place["item"]], place))
    return np.reshape(tables, (-1, 5)), np.reshape(buildings, (-1, 5))


def reset(env: Any, ep: SolarEpisode) -> None:
    """The patrol starts on the charge the dock left in the battery."""
    ep.drone_state = {"battery_pct": BATTERY_START_PCT}


def update(env: Any, ep: SolarEpisode) -> None:
    """After one control step: the battery drains at the hover rate while the rotors turn."""
    rpm = np.asarray(getattr(env, "last_clipped_action", np.zeros((1, 4))), dtype=float).reshape(-1, 4)[0]
    if np.any(rpm > 1.0):
        used = 100.0 * float(env.CTRL_TIMESTEP) / HOVER_ENDURANCE_S
        ep.drone_state["battery_pct"] = max(0.0, ep.drone_state["battery_pct"] - used)


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """Position and velocity from the dock, compass heading, height above take-off, time left and battery."""
    position = np.asarray(env.pos[0], dtype=float) - ep.dock_position
    heading = (90.0 - math.degrees(float(env.rpy[0, 2])) + 180.0) % 360.0 - 180.0
    put(state, STATE_SLICES, "position_m", position)
    put(state, STATE_SLICES, "heading_deg", heading)
    put(state, STATE_SLICES, "velocity_mps", np.asarray(env.vel[0], dtype=float))
    put(state, STATE_SLICES, "height_above_takeoff_m", position[2])
    put(state, STATE_SLICES, "time_left_s", max(0.0, HORIZON_S - ep.time_s))
    # The aircraft reports its charge in whole percent.
    put(state, STATE_SLICES, "battery_pct", math.floor(ep.drone_state["battery_pct"] + 0.5))


def _put_rows(site: np.ndarray, count: str, name: str, rows: np.ndarray, capacity: int) -> None:
    """Write rows into a zero-padded site map field, and how many there are into its count."""
    rows = rows[:capacity]
    padded = np.zeros((capacity, rows.shape[1]))
    padded[:len(rows)] = rows
    put(site, SITE_MAP_SLICES, count, len(rows))
    put(site, SITE_MAP_SLICES, name, padded.reshape(-1))


def site_map(env: Any, ep: SolarEpisode, site: np.ndarray) -> None:
    """The survey of the site in metres from the dock: the fence, the panel tables and the buildings."""
    origin = ep.dock_position[:2]
    tables, buildings = (rows.copy() for rows in survey(park.asset_dir(ep)))
    tables[:, :2] -= origin
    buildings[:, :2] -= origin
    _put_rows(site, "fence_count", "fence_xy", ep.fence - origin, MAX_FENCE_POINTS)
    _put_rows(site, "table_count", "tables", tables, MAX_TABLES)
    _put_rows(site, "building_count", "buildings", buildings, MAX_BUILDINGS)
