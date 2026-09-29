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

"""Zoom lenses (task 11): a box and a lens asked for, and the close view that arrives one decision later.

The M4TD carries two tele cameras beside the wide one: 3x (35 degree diagonal) and 7x (15 degree). A zoom is a
request, not a free look: the model draws a box on the frame it is looking at and picks a lens, as the dock's
zoom-on-a-box command takes them. The box centre becomes a ray from that frame's view, and the first thing the ray
meets past the aircraft is the point zoomed on. One decision later the lens is pointed at that point and one 640 x 480
frame is drawn, so what was boxed sits in the middle even if the aircraft moved or turned meanwhile. The box's size
picks nothing: there is no digital zoom, the lens alone sets the view.

At night the zoom frame takes night mode by the wide camera's rule. Through the 7x lens the model can switch on night
vision (the outputs part holds it to that lens): a black and white picture lit by the aircraft's infrared light, a
5.7 degree beam reaching 100 m. Stand-in until the engine draws both looks (task 27): night mode is the wide camera's
own stand-in, and night vision is the frame's brightness, lit inside the beam when the point zoomed on is in reach.

Each zoom frame keeps the view it was drawn from, so a report boxed on it is read against the same view.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
import pybullet as p

from . import airframe, camera
from .contract import DECISION_STEPS, MAX_ZOOMS, STATE_SLICES, ZOOM_SHAPE, Box, Command, ZoomRequest, put
from .episode import SolarEpisode

LENS_DIAGONAL_FOV_DEG = {3: 35.0, 7: 15.0}  # DJI: M4TD medium tele and tele cameras
NIGHT_VISION_BEAM_DEG = 5.7                 # DJI: M4TD infrared auxiliary light
NIGHT_VISION_RANGE_M = 100.0
NIGHT_VISION_GAIN = 4.0                     # stand-in: brightness inside the beam at night
NIGHT_VISION_OUTSIDE = 0.1                  # stand-in: share of the brightness left outside it
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float32)
# Zoom frames draw their night grain from their own indices, clear of the wide camera's captures.
GRAIN_OFFSET = 1 << 20


def reset(env: Any, ep: SolarEpisode) -> None:
    """No zoom taken yet, night mode off and night vision off."""
    ep.zoom = {"lens": 0, "arrived_s": 0.0, "pending": None, "pending_step": 0, "seen": None, "view": None,
               "night_mode": "off", "night_vision": False}


def request(env: Any, ep: SolarEpisode, command: Command) -> None:
    """Take the night switches, and queue a zoom with the view it was boxed on while the patrol has zooms left."""
    ep.zoom["night_mode"] = command.night_mode
    ep.zoom["night_vision"] = command.night_vision
    if command.zoom is None or ep.outcome.zooms_used >= MAX_ZOOMS:
        return
    ep.outcome.zooms_used += 1
    ep.zoom.update(pending=command.zoom, pending_step=ep.step, seen=camera.view(ep))


def update(env: Any, ep: SolarEpisode) -> None:
    """Draw a queued zoom once a full decision has passed since it was asked for."""
    asked = ep.zoom["pending"]
    if asked is None or ep.step - ep.zoom["pending_step"] < DECISION_STEPS:
        return
    shot, reach_m = aim(env, ep, asked)
    ep.frames.zoom = _draw(env, ep, shot, reach_m)
    ep.zoom.update(lens=asked.lens, arrived_s=ep.time_s, pending=None, seen=None, view=shot)


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """Night vision, the lens of the zoom view shown, its age, and the zooms left."""
    put(state, STATE_SLICES, "night_vision", float(ep.zoom["night_vision"]))
    put(state, STATE_SLICES, "zoom_lens", ep.zoom["lens"])
    put(state, STATE_SLICES, "zoom_age_s", ep.time_s - ep.zoom["arrived_s"] if ep.zoom["lens"] else 0.0)
    put(state, STATE_SLICES, "zooms_left", MAX_ZOOMS - ep.outcome.zooms_used)


def view(ep: SolarEpisode) -> Optional[camera.View]:
    """The view of the zoom frame the model is looking at, None before the first one arrives."""
    return ep.zoom["view"]


def box_ray(seen: camera.View, box: Box) -> np.ndarray:
    """The world direction from a frame's eye through the centre of a box on it, box shares counted from the top left."""
    forward, up = np.asarray(seen.forward, dtype=float), np.asarray(seen.up, dtype=float)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    half_v = math.tan(math.radians(seen.vertical_fov_deg) / 2.0)
    half_h = half_v * seen.width / seen.height
    ray = forward + (2.0 * box.cx - 1.0) * half_h * right + (1.0 - 2.0 * box.cy) * half_v * up
    return ray / np.linalg.norm(ray)


def aim(env: Any, ep: SolarEpisode, asked: ZoomRequest) -> tuple[camera.View, float]:
    """The asked lens pointed from where the camera is now at the point under the box, and how far away that point is.

    The point is where the box's ray first meets the world past the aircraft; a ray that meets nothing keeps its
    direction, and its point is out of any light's reach.
    """
    seen = ep.zoom["seen"]
    origin = np.asarray(seen.eye, dtype=float)
    ray = box_ray(seen, asked.box)
    point = _first_hit(env, origin, ray)
    eye, _forward, camera_up = airframe.camera_pose(env, ep.camera["tilt_deg"])
    if point is None:
        forward, reach_m = ray, math.inf
    else:
        forward = point - eye
        reach_m = float(np.linalg.norm(forward))
        forward /= reach_m
    # The zoom frame stays upright the way the frame it was boxed on was.
    up = np.asarray(seen.up, dtype=float) - np.dot(seen.up, forward) * forward
    if np.linalg.norm(up) < 1e-6:
        up = np.asarray(camera_up, dtype=float)
    up /= np.linalg.norm(up)
    height, width = ZOOM_SHAPE[:2]
    dark = camera.night(env)
    shot = camera.View(feed="colour", eye=_floats(eye), forward=_floats(forward), up=_floats(up), width=width,
                       height=height,
                       vertical_fov_deg=camera.vertical_fov_deg(LENS_DIAGONAL_FOV_DEG[asked.lens], width, height),
                       sees=not dark or _night_scene(ep, dark) or ep.zoom["night_vision"], step=ep.step)
    return shot, reach_m


def _first_hit(env: Any, origin: np.ndarray, ray: np.ndarray) -> Optional[np.ndarray]:
    """Where a ray first meets anything but the aircraft within the camera's range, or None."""
    aircraft = int(env.DRONE_IDS[0])
    end = origin + ray * camera.FAR_M
    start = origin
    for _ in range(8):
        uid, _link, _fraction, position, _normal = p.rayTest(start.tolist(), end.tolist(),
                                                              physicsClientId=env.CLIENT)[0]
        if uid < 0:
            return None
        if uid != aircraft:
            return np.asarray(position, dtype=float)
        start = np.asarray(position, dtype=float) + ray * 1e-3
    return None


def _night_scene(ep: SolarEpisode, dark: bool) -> bool:
    """True when the zoom frame is taken in night mode: on, or auto in the dark."""
    return ep.zoom["night_mode"] == "on" or (ep.zoom["night_mode"] == "auto" and dark)


def _draw(env: Any, ep: SolarEpisode, shot: camera.View, reach_m: float) -> np.ndarray:
    """One zoom frame through the camera's own draw, with night mode or night vision on top."""
    frame = camera.colour_frame(env, shot)
    dark = camera.night(env)
    if ep.zoom["night_vision"]:
        return _night_vision(frame, shot, reach_m, dark)
    if dark and _night_scene(ep, dark):
        return camera.night_scene_stand_in(frame, ep.seed, GRAIN_OFFSET + ep.outcome.zooms_used)
    return frame


def _night_vision(frame: np.ndarray, shot: camera.View, reach_m: float, dark: bool) -> np.ndarray:
    """Black and white; in the dark, bright inside the infrared beam while the point zoomed on is in its reach."""
    grey = frame @ LUMA
    if dark:
        half_v = math.tan(math.radians(shot.vertical_fov_deg) / 2.0)
        half_h = half_v * shot.width / shot.height
        ys = ((np.arange(shot.height, dtype=np.float32) + 0.5) / shot.height * 2.0 - 1.0) * half_v
        xs = ((np.arange(shot.width, dtype=np.float32) + 0.5) / shot.width * 2.0 - 1.0) * half_h
        in_beam = np.hypot(xs[None, :], ys[:, None]) <= math.tan(math.radians(NIGHT_VISION_BEAM_DEG / 2.0))
        lit = in_beam & (reach_m <= NIGHT_VISION_RANGE_M)
        grey = np.clip(grey * np.where(lit, NIGHT_VISION_GAIN, NIGHT_VISION_OUTSIDE).astype(np.float32), 0.0, 1.0)
    return np.repeat(grey[..., None], 3, axis=2).astype(np.float32)


def _floats(vector: np.ndarray) -> tuple[float, float, float]:
    """A world vector as plain floats."""
    return tuple(float(v) for v in vector)
