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

"""Grass that is drawn only where a picture can show it.

An area's grass is hundreds of thousands of small cards. Seen from the patrol's height they are specks that only
darken the ground photograph under them, and the thermal camera reads every card as a spot of cold sky; seen from low
over the ground, or through a long lens, they are what makes the ground look real. They are also most of what a
picture costs to draw.

So the seed's grass is stood in square patches of CELL_M, only within MARGIN_M of the field's box, and every patch is
hidden until a picture needs it. A colour picture shows a patch when its camera is at most HEIGHT_M over the patch's
top and at most REACH_M across the ground from its box. A lens longer than the wide one stretches both by its
magnification, and shows only the patches its frame can hold. The thermal camera shows none.

Which patches a picture shows follows from its own view alone, never from the pictures before it, so the same view
always gives the same picture. A reading that is no picture follows the same idea: the laser meets the patches the
wide lens would show from where the drone is, and a measure of what a place shows of someone is taken on bare ground,
at a seed's build and in flight alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
import pybullet as p

from swarm.core.maps.andorra.builder import Patch

from ..solar_patrol import camera
from ..solar_patrol.contract import RGB_SHAPE
from .episode import FieldEpisode

CELL_M = 12.0                          # a patch's side: small enough to show little, large enough to change seldom
MARGIN_M = 24.0                        # grass is stood this far past the field's box, REACH_M past its edge
HEIGHT_M = 8.0                         # the wide lens shows grass from this low over it; the patrol cruises higher
REACH_M = 24.0                         # and this far across the ground
SHOWN, HIDDEN = (1.0, 1.0, 1.0, 1.0), (1.0, 1.0, 1.0, 0.0)


@dataclass
class Lawn:
    """A seed's grass patches: their bodies, the boxes their cards fill, and which of them are drawn now."""

    bodies: np.ndarray
    low: np.ndarray
    high: np.ndarray
    shown: np.ndarray


def lay(cli: int, patches: Sequence[Patch]) -> Lawn:
    """Take the patches an area load stood and hide them all, as they stay until a picture needs one."""
    for patch in patches:
        p.changeVisualShape(patch.body, -1, rgbaColor=HIDDEN, physicsClientId=cli)
    return Lawn(bodies=np.array([patch.body for patch in patches], dtype=np.int64),
                low=np.array([patch.low for patch in patches], dtype=float).reshape(-1, 3),
                high=np.array([patch.high for patch in patches], dtype=float).reshape(-1, 3),
                shown=np.zeros(len(patches), dtype=bool))


def magnification(view: camera.View) -> float:
    """How many times longer a view's lens is than the wide camera's: 1 for the wide lens itself."""
    height, width = RGB_SHAPE[:2]
    wide = camera.vertical_fov_deg(camera.WIDE_DIAGONAL_FOV_DEG, width, height)
    return math.tan(math.radians(wide) / 2.0) / math.tan(math.radians(view.vertical_fov_deg) / 2.0)


def in_frame(lawn: Lawn, view: camera.View) -> np.ndarray:
    """Which patches a view's frame can hold: those whose bounding sphere reaches into the cone round the frame's
    corners."""
    eye = np.asarray(view.eye, dtype=float)
    centre = (lawn.low + lawn.high) / 2.0 - eye
    radius = np.linalg.norm(lawn.high - lawn.low, axis=1) / 2.0
    distance = np.linalg.norm(centre, axis=1)
    corner = math.atan(math.tan(math.radians(view.vertical_fov_deg) / 2.0) * math.hypot(view.width, view.height) / view.height)
    along = centre @ np.asarray(view.forward, dtype=float)
    off_axis = np.arccos(np.clip(along / np.maximum(distance, 1e-9), -1.0, 1.0))
    spread = np.arcsin(np.clip(radius / np.maximum(distance, 1e-9), 0.0, 1.0))
    return (distance <= radius) | (off_axis <= corner + spread)


def near(lawn: Lawn, eye: Sequence[float], times: float = 1.0) -> np.ndarray:
    """Which patches a point is low over and near: at most HEIGHT_M over a patch's top and REACH_M across the ground
    from its box, both stretched by a lens's magnification."""
    eye = np.asarray(eye, dtype=float)
    over = eye[2] - lawn.high[:, 2]
    across = np.linalg.norm(np.maximum(np.maximum(lawn.low[:, :2] - eye[:2], eye[:2] - lawn.high[:, :2]), 0.0), axis=1)
    return (over <= HEIGHT_M * times) & (across <= REACH_M * times)


def wanted(lawn: Lawn, view: camera.View) -> np.ndarray:
    """Which patches the picture of a view shows: none through the thermal camera; else those the camera is low over
    and near, by its lens, and for a lens longer than the wide one only those in its frame."""
    if view.feed == "thermal" or not len(lawn.bodies):
        return np.zeros(len(lawn.bodies), dtype=bool)
    times = magnification(view)
    close = near(lawn, view.eye, times)
    return close & in_frame(lawn, view) if times > 1.0 else close


def _lawn(ep: FieldEpisode) -> Optional[Lawn]:
    """The seed's grass, or None for a seed that stood none."""
    return ep.park.get("grass") if isinstance(ep.park, dict) else None


def _draw(env: Any, lawn: Lawn, want: np.ndarray) -> None:
    """Draw exactly the wanted patches, switching only those that change."""
    for n in np.flatnonzero(want != lawn.shown):
        p.changeVisualShape(int(lawn.bodies[n]), -1, rgbaColor=SHOWN if want[n] else HIDDEN, physicsClientId=env.CLIENT)
    lawn.shown = want


def show(env: Any, ep: FieldEpisode, view: Optional[camera.View]) -> None:
    """Draw exactly the patches the picture of a view shows; nothing for no view or a seed with no grass."""
    lawn = _lawn(ep)
    if view is not None and lawn is not None:
        _draw(env, lawn, wanted(lawn, view))


def show_near(env: Any, ep: FieldEpisode, eye: Sequence[float]) -> None:
    """Draw the patches the wide lens shows from a point, whatever the feed: what a reading taken there, not a
    picture, meets."""
    lawn = _lawn(ep)
    if lawn is not None:
        _draw(env, lawn, near(lawn, eye))


def hide(env: Any, ep: FieldEpisode) -> None:
    """Draw no patch: for a measure taken on bare ground, as a seed's build takes it."""
    lawn = _lawn(ep)
    if lawn is not None:
        _draw(env, lawn, np.zeros(len(lawn.bodies), dtype=bool))
