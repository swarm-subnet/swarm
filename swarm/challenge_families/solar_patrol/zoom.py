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

"""Zoom lenses (task 11): a box and a lens asked for, and the close view the link brings back.

The M4TD carries two tele cameras beside the wide one: 3x (35 degree diagonal) and 7x (15 degree). A zoom is a
request, not a free look: the model draws a box on the frame it is looking at and picks a lens, as the dock's
zoom-on-a-box command takes them. The box centre becomes a ray from that frame's view, and the first thing the ray
meets past the aircraft is the point zoomed on. When the request reaches the aircraft the lens is pointed at that point
and one 640 x 480 frame is drawn, so what was boxed sits in the middle even if the aircraft moved or turned meanwhile;
the link's delays (task 16) set when the model sees it. The box's size picks nothing: there is no digital zoom, the
lens alone sets the view.

At night the zoom frame takes night mode by the wide camera's rule, through its own lens, which gathers less light than
the wide one and so shows more grain. Through the 7x lens the model can switch on night vision (the outputs part holds
it to that lens): a black and white picture lit by the aircraft's infrared light, a 5.7 degree beam reaching 100 m,
drawn by the engine (task 27) along the zoom's view.

Each zoom frame keeps the view it was drawn from, so a report boxed on it is read against the same view.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Optional

import numpy as np
import pybullet as p

from . import airframe, camera, sensor_noise, theft
from .contract import MAX_ZOOMS, STATE_SLICES, ZOOM_SHAPE, Box, Command, ZoomRequest, put
from .episode import SolarEpisode
from .fixed_order import dot, norm

LENS_DIAGONAL_FOV_DEG = {3: 35.0, 7: 15.0}  # DJI: M4TD medium tele and tele cameras
NIGHT_VISION_BEAM_DEG = 5.7                 # DJI: M4TD infrared auxiliary light
NIGHT_VISION_RANGE_M = 100.0
# DJI publishes no power for the light, so a target at the end of its reach is lit as a full moon lights the ground.
NIGHT_VISION_INTENSITY_LUX_M2 = camera.FULL_MOON_ZENITH_LUX * NIGHT_VISION_RANGE_M ** 2
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
    # The box was drawn on the frame the model was shown, which the link delivers late.
    seen = sensor_noise.shown_view(ep, "feed") or camera.view(ep)
    ep.zoom.update(pending=command.zoom, pending_step=ep.step, seen=seen)


def update(env: Any, ep: SolarEpisode) -> None:
    """Draw a queued zoom on the control step its request reached the aircraft."""
    asked = ep.zoom["pending"]
    if asked is None:
        return
    shot, _reach_m = aim(env, ep, asked)
    ep.frames.zoom, objects = _draw(env, ep, shot, asked.lens)
    ep.zoom.update(lens=asked.lens, arrived_s=ep.time_s, pending=None, seen=None, view=replace(shot, objects=objects))


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """Night vision, the lens of the zoom view shown, its age, and the zooms left."""
    put(state, STATE_SLICES, "night_vision", float(ep.zoom["night_vision"]))
    put(state, STATE_SLICES, "zoom_lens", ep.zoom["lens"])
    put(state, STATE_SLICES, "zoom_age_s", ep.time_s - ep.zoom["arrived_s"] if ep.zoom["lens"] else 0.0)
    put(state, STATE_SLICES, "zooms_left", MAX_ZOOMS - ep.outcome.zooms_used)


def upcoming(env: Any, ep: SolarEpisode) -> Optional[camera.View]:
    """The view of the zoom this control step draws, from where the camera is now; None when none is queued."""
    asked = ep.zoom["pending"]
    return None if asked is None else aim(env, ep, asked)[0]


def view(ep: SolarEpisode) -> Optional[camera.View]:
    """The view of the zoom frame the model is looking at, None before the first one arrives."""
    return ep.zoom["view"]


def box_ray(seen: camera.View, box: Box) -> np.ndarray:
    """The world direction from a frame's eye through the centre of a box on it, box shares counted from the top left."""
    forward, up = np.asarray(seen.forward, dtype=float), np.asarray(seen.up, dtype=float)
    right = np.cross(forward, up)
    right /= norm(right)
    up = np.cross(right, forward)
    half_v = math.tan(math.radians(seen.vertical_fov_deg) / 2.0)
    half_h = half_v * seen.width / seen.height
    ray = forward + (2.0 * box.cx - 1.0) * half_h * right + (1.0 - 2.0 * box.cy) * half_v * up
    return ray / norm(ray)


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
        reach_m = norm(forward)
        forward /= reach_m
    # The zoom frame stays upright the way the frame it was boxed on was.
    up = np.asarray(seen.up, dtype=float) - dot(seen.up, forward) * forward
    if norm(up) < 1e-6:
        up = np.asarray(camera_up, dtype=float)
    up /= norm(up)
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


def _draw(env: Any, ep: SolarEpisode, shot: camera.View, lens: int) -> tuple[np.ndarray, np.ndarray]:
    """One zoom frame through the camera's own draw and the lens's own camera at night, or in night vision, and its
    object map."""
    theft.show(env, ep, shot)
    dark = camera.night(env)
    beam = (NIGHT_VISION_BEAM_DEG, NIGHT_VISION_RANGE_M, NIGHT_VISION_INTENSITY_LUX_M2) if ep.zoom["night_vision"] else None
    at_night = camera.night_camera(env, lens, shot, _night_scene(ep, dark), ep.seed, GRAIN_OFFSET + ep.outcome.zooms_used,
                                   beam=beam)
    frame, objects = camera.colour_frame(env, shot, at_night)
    # An engine without the near infrared still gives night vision in black and white.
    if beam and not camera.LOW_LIGHT:
        # Weighted in float32 one channel at a time, not by a BLAS product whose rounding follows the CPU.
        luma = frame[..., 0] * LUMA[0] + frame[..., 1] * LUMA[1] + frame[..., 2] * LUMA[2]
        frame = np.repeat(luma[..., None], 3, axis=2).astype(np.float32)
    return frame, objects


def _floats(vector: np.ndarray) -> tuple[float, float, float]:
    """A world vector as plain floats."""
    return tuple(float(v) for v in vector)
