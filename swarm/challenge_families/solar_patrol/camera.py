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

"""Colour camera and gimbal (task 10): the wide camera on its tilt-only gimbal, and which feed the model sees.

The camera sits on the M4TD's gimbal where DJI's CAD puts the wide lens and tilts from straight down to straight up;
it never pans, so the model turns the aircraft to look sideways. Twice a second it takes one frame of the feed the
model picked: colour at 640 x 480 through the 82 degree wide lens, or thermal at 640 x 512 through the 45 degree
thermal lens in the engine's White Hot mode (task 2). The other feed's image is left at zero, and a feed switched
between frames shows from the next one. Nothing hides the aircraft from its own camera, so above +70 degrees of tilt
its body fills the view as it does on the real one.

Frames are lit by the seed's light as the environment set it. At night the colour frame is what the real camera
gives: dark with night mode off, brighter and grainy with it on, and auto turns it on in the dark. The engine's
low-light camera draws it (task 27): the moon's real brightness, the lens's aperture and sensor, and the top ISO of the
mode set how bright the frame can get and how much grain the light it collects carries.

Each frame keeps the view it was taken from, so the zoom, report and coverage parts work from the image the model saw.
The view also carries the frame's object map from the same draw: which body each pixel shows, never shown to the model.
"""

from __future__ import annotations

import functools
import math
import operator
from dataclasses import dataclass, field, replace
from typing import Any, Optional, Tuple

import numpy as np
import pybullet as p

from swarm.constants import MOON_DIFFUSE_RANGE, SIM_DT
from swarm.core.daylight import sun_render_kwargs

from . import airframe, park, theft
from .contract import GIMBAL_TILT_RANGE_DEG, NIGHT_MODES, RGB_SHAPE, STATE_SLICES, THERMAL_SHAPE, Command, put
from .episode import SolarEpisode

# The ray caster is the only path with thermal and daylight; an engine without it draws colour on TinyRenderer and
# leaves the thermal image blank.
RAYCAST = hasattr(p, "ER_SWARM_RAYCAST")
THERMAL = RAYCAST and hasattr(p, "ER_SWARM_THERMAL")
RENDER_BACKEND = "raycast" if RAYCAST else "tiny"
# The ray caster's picture flags for a colour camera, the daylight model's own set, so a frame without it (at night,
# or before the park turns it on) keeps shadows, leaf cut-outs, filtered textures and clean edges.
PICTURE_FLAGS = functools.reduce(operator.or_, (getattr(p, name, 0) for name in (
    "ER_SWARM_SHADOW_MAP", "ER_SWARM_MOVER_SHADOW", "ER_EDGE_ANTIALIAS", "ER_ALPHA_CUTOUT", "ER_TEXTURE_FILTER",
    "ER_SPECULAR_GLINT", "ER_SWARM_LINEAR_LIGHT")), 0) if RAYCAST else 0

FRAME_HZ = 2.0
FRAME_STEPS = int(round(1.0 / (FRAME_HZ * SIM_DT)))
FEEDS = ("colour", "thermal")
WIDE_DIAGONAL_FOV_DEG = 82.0            # DJI: M4TD wide camera
THERMAL_DIAGONAL_FOV_DEG = 45.0         # DJI: M4TD thermal camera
NEAR_M = 0.004                          # under the airframe's 5 mm, so the aircraft is drawn when it is in view
FAR_M = 2000.0

# The engine's low-light camera and near infrared; an engine without them draws the plain moonlit frame.
LOW_LIGHT = RAYCAST and hasattr(p, "ER_SWARM_LOW_LIGHT") and hasattr(p, "ER_SWARM_NEAR_INFRARED")
NIGHT_GRAIN_STREAM = 0x4E16


@dataclass(frozen=True)
class Lens:
    """One of the M4TD's colour cameras in the dark: its aperture, its sensor's area and its top ISO by mode."""

    f_number: float
    sensor_mm2: float
    iso_normal: float
    iso_night: float


# DJI Matrice 4 series specs: wide 1/1.3 inch f/1.7, medium tele 1/1.3 inch f/2.8, tele 1/1.5 inch f/2.8, each 48 MP;
# ISO up to 25,600 in normal mode and 409,600 in night mode, 819,200 on the tele. Keyed by zoom, the wide lens as 1.
LENSES = {1: Lens(1.7, 70.3, 25600.0, 409600.0), 3: Lens(2.8, 70.3, 25600.0, 409600.0),
          7: Lens(2.8, 48.8, 25600.0, 819200.0)}
WIDE_LENS = 1
# The sensor: normal video exposes a frame of the 30 fps stream, and Night Scene slows the stream to 15 fps to expose
# twice as long (M4T footage over unlit farmland, 1/15 s in its middle setting); a back-lit CMOS through its colour
# filters turns about 4,000 photons of white light per square micrometre and lux second into electrons and saturates at
# about 4,000 per square micrometre at ISO 100; video reads the 48 MP array binned to 12 MP with 1.5 electrons of read
# noise at the high conversion gain of night ISOs; the stream's temporal noise reduction averages about four frames.
EXPOSURE_S = 1.0 / 30.0
NIGHT_SCENE_EXPOSURE_S = 1.0 / 15.0
LENS_TRANSMISSION = 0.9
ELECTRONS_PER_LUX_S_UM2 = 4000.0
FULL_WELL_E_PER_UM2 = 4000.0
BINNED_PIXELS = 12.0e6
READ_NOISE_E = 1.5
NOISE_REDUCTION_FRAMES = 4.0
# Night vision takes the IR-cut filter out, and silicon collects about half as much again past it.
NEAR_INFRARED_GAIN = 1.5
# Moonlight on level ground: 0.25 lux from a full moon at the zenith, dimmed by its phase angle as Allen gives it and by
# its height, plus the glow of the sky near a town. The renderer's moon runs from a crescent (120 degrees of phase
# angle) to full by its strength.
FULL_MOON_ZENITH_LUX = 0.25
CRESCENT_PHASE_ANGLE_DEG = 120.0
SKY_GLOW_LUX = 0.002


@dataclass(frozen=True)
class View:
    """Where a frame was taken from and through which lens, and whether it can show anything."""

    feed: str
    eye: Tuple[float, float, float]
    forward: Tuple[float, float, float]
    up: Tuple[float, float, float]
    width: int
    height: int
    vertical_fov_deg: float
    sees: bool                          # False for a colour frame in the dark with night mode off
    step: int
    # Per pixel, the body id plus (link index + 1) << 24, or -1 where nothing was hit; set once the frame is drawn.
    objects: Optional[np.ndarray] = field(default=None, compare=False, repr=False)

    def matrices(self) -> Tuple[tuple, tuple]:
        """The renderer's view and projection matrices of this frame."""
        target = [e + f for e, f in zip(self.eye, self.forward)]
        view = p.computeViewMatrix(list(self.eye), target, list(self.up))
        projection = p.computeProjectionMatrixFOV(self.vertical_fov_deg, self.width / self.height, NEAR_M, FAR_M)
        return view, projection


def vertical_fov_deg(diagonal_deg: float, width: int, height: int) -> float:
    """The vertical field of view of a lens sold by its diagonal one, on a width x height image."""
    half = math.tan(math.radians(diagonal_deg / 2.0)) * height / math.hypot(width, height)
    return 2.0 * math.degrees(math.atan(half))


def reset(env: Any, ep: SolarEpisode) -> None:
    """The gimbal level, the colour feed, night mode off, and no frame yet."""
    ep.camera = {"tilt_deg": 0.0, "thermal": False, "night_mode": "off", "captured_s": 0.0, "captures": 0,
                 "view": None}


def request(env: Any, ep: SolarEpisode, command: Command) -> None:
    """Take the tilt, feed and night mode the model asked for; the outputs part already holds the tilt rate."""
    low, high = GIMBAL_TILT_RANGE_DEG
    ep.camera["tilt_deg"] = low + (command.gimbal_tilt + 1.0) / 2.0 * (high - low)
    ep.camera["thermal"] = command.thermal
    ep.camera["night_mode"] = command.night_mode


def update(env: Any, ep: SolarEpisode) -> None:
    """Capture a new frame at the camera's own rate."""
    if ep.step % FRAME_STEPS == 0:
        capture(env, ep)


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The gimbal tilt, the feed of the frame shown, its age and the night mode; the first frame is taken here."""
    if ep.camera["view"] is None:
        capture(env, ep)
    put(state, STATE_SLICES, "gimbal_tilt_deg", ep.camera["tilt_deg"])
    put(state, STATE_SLICES, "camera_feed", float(FEEDS.index(ep.camera["view"].feed)))
    put(state, STATE_SLICES, "frame_age_s", ep.time_s - ep.camera["captured_s"])
    put(state, STATE_SLICES, "night_mode", NIGHT_MODES.index(ep.camera["night_mode"]))


def view(ep: SolarEpisode) -> View:
    """The view of the frame the model is looking at."""
    return ep.camera["view"]


def upcoming(env: Any, ep: SolarEpisode, step: int) -> Optional[View]:
    """The view the frame taken on that control step will have, from where the camera is now; None on a step
    between frames."""
    return aim(env, ep) if step % FRAME_STEPS == 0 else None


def aim(env: Any, ep: SolarEpisode) -> View:
    """The view a frame taken now has: its feed, lens and pose, and whether it can show anything."""
    cam = ep.camera
    feed = FEEDS[int(cam["thermal"])]
    height, width = (THERMAL_SHAPE if cam["thermal"] else RGB_SHAPE)[:2]
    diagonal = THERMAL_DIAGONAL_FOV_DEG if cam["thermal"] else WIDE_DIAGONAL_FOV_DEG
    eye, forward, up = airframe.camera_pose(env, cam["tilt_deg"])
    return View(feed=feed, eye=_floats(eye), forward=_floats(forward), up=_floats(up), width=int(width),
                height=int(height), vertical_fov_deg=vertical_fov_deg(diagonal, width, height),
                sees=cam["thermal"] or not night(env) or _night_scene(cam, env), step=ep.step)


def _night_scene(cam: dict, env: Any) -> bool:
    """Whether the colour camera brightens its frame: night mode on, or on auto in the dark."""
    return cam["night_mode"] == "on" or (cam["night_mode"] == "auto" and night(env))


def capture(env: Any, ep: SolarEpisode) -> None:
    """Take one frame of the feed asked for from where the camera is now, and blank the other feed."""
    cam = ep.camera
    shot = aim(env, ep)
    theft.show(env, ep, shot)
    if cam["thermal"]:
        ep.frames.thermal, objects = _thermal_frame(env, ep, shot)
        ep.frames.rgb = np.zeros(RGB_SHAPE, dtype=np.float32)
    else:
        ep.frames.rgb, objects = colour_frame(env, shot, night_camera(env, WIDE_LENS, shot, _night_scene(cam, env), ep.seed,
                                                                      cam["captures"]))
        ep.frames.thermal = np.zeros(THERMAL_SHAPE, dtype=np.float32)
    cam.update(view=replace(shot, objects=objects), captured_s=ep.time_s, captures=cam["captures"] + 1)


def moon_lux(sun: Any) -> float:
    """Light on level ground under a seed's moon, in lux: its phase read off the renderer's strength, then its height."""
    low, high = MOON_DIFFUSE_RANGE
    phase = min(1.0, max(0.0, (sun.diffuse - low) / (high - low)))
    angle = CRESCENT_PHASE_ANGLE_DEG * (1.0 - phase)
    dimming = 10.0 ** (-0.4 * (0.026 * angle + 4e-9 * angle ** 4))
    return FULL_MOON_ZENITH_LUX * dimming * max(0.0, sun.direction[2]) + SKY_GLOW_LUX


def night_camera(env: Any, lens: int, shot: View, night_scene: bool, seed: int, index: int,
                 beam: Optional[Tuple[float, float, float]] = None) -> Tuple[int, dict]:
    """The engine flags and arguments that make a colour frame at night what the M4TD's camera gives.

    The frame collects the electrons the moon's real light puts through this lens onto its sensor, the gain is capped
    at the mode's top ISO, and `index` moves the grain frame by frame. `beam` is night vision: the IR-cut filter out and
    the infrared light, `(angle_deg, range_m, intensity_lux_m2)`, along the view. Nothing by day, when night vision is
    only grey, or on an engine without the low-light camera.
    """
    if not LOW_LIGHT:
        return 0, {}
    if not night(env):
        return (p.ER_SWARM_NEAR_INFRARED if beam else 0), {}
    sun, optic = env._sun, LENSES[lens]
    moon_colour = 0.2126 * sun.color[0] + 0.7152 * sun.color[1] + 0.0722 * sun.color[2]
    lux_per_unit = moon_lux(sun) / (sun.ambient + sun.diffuse * moon_colour * max(0.0, sun.direction[2]))
    pixels = shot.width * shot.height
    pixel_um2 = optic.sensor_mm2 * 1e6 / pixels
    night_mode = bool(night_scene or beam)
    electrons = (lux_per_unit * LENS_TRANSMISSION / (4.0 * optic.f_number ** 2) * pixel_um2 * ELECTRONS_PER_LUX_S_UM2
                 * (NIGHT_SCENE_EXPOSURE_S if night_mode else EXPOSURE_S) * (NEAR_INFRARED_GAIN if beam else 1.0))
    iso = optic.iso_night if night_mode else optic.iso_normal
    kwargs = {
        "sensorPhotons": _digits(electrons * NOISE_REDUCTION_FRAMES),
        "sensorReadNoise": _digits(READ_NOISE_E * math.sqrt(BINNED_PIXELS / pixels * NOISE_REDUCTION_FRAMES)),
        "sensorGainCap": _digits(electrons * iso / (100.0 * FULL_WELL_E_PER_UM2 * pixel_um2)),
        "sensorSeed": (int(seed) * 1000003 + int(index)) & 0x7FFFFFFF ^ NIGHT_GRAIN_STREAM,
    }
    if not beam:
        return p.ER_SWARM_LOW_LIGHT, kwargs
    angle_deg, range_m, intensity = beam
    kwargs["spotLight"] = [*shot.eye, *shot.forward, angle_deg, range_m, _digits(intensity / lux_per_unit)]
    return p.ER_SWARM_LOW_LIGHT | p.ER_SWARM_NEAR_INFRARED, kwargs


def _digits(value: float) -> float:
    """A value held to six significant digits, so a last-bit difference in the maths library never reaches the engine."""
    return float(f"{value:.6g}")


def night(env: Any) -> bool:
    """True when the seed is lit by the moon."""
    sun = getattr(env, "_sun", None)
    return bool(sun is not None and sun.night)


def _floats(vector: np.ndarray) -> Tuple[float, float, float]:
    """A world vector as plain floats."""
    return tuple(float(v) for v in vector)


def _object_map(seg: Any, shot: View) -> np.ndarray:
    """The engine's segmentation buffer as one int32 body code per pixel, laid out like the frame."""
    return np.reshape(np.asarray(seg, dtype=np.int32), (shot.height, shot.width))


def colour_frame(env: Any, shot: View, camera_at_night: Tuple[int, dict] = (0, {})) -> Tuple[np.ndarray, np.ndarray]:
    """A colour frame in the seed's light, the way the environment lights its own colour frames, through the flags
    and arguments of `night_camera` when given, and its object map."""
    cli = env.CLIENT
    view_matrix, projection = shot.matrices()
    kwargs = sun_render_kwargs(env._sun) if env._sun is not None else {}
    # The shadow share and the sky are the fork's arguments; an engine without the ray caster predates both.
    if RAYCAST:
        kwargs["shadowLightCoeff"] = 0.0
        kwargs.update(env._daylight_kwargs(cli))
        kwargs.update(env._sky_kwargs())
    night_flags, night_kwargs = camera_at_night
    kwargs.update(night_kwargs)
    flags = env._render_flags | env._sky_flags | env._daylight_flags | PICTURE_FLAGS | night_flags
    _w, _h, rgb, _depth, seg = p.getCameraImage(
        shot.width, shot.height, view_matrix, projection, renderer=p.ER_TINY_RENDERER,
        shadow=1 if env._daylight_flags or PICTURE_FLAGS else 0, lightDirection=env._light_direction, flags=flags,
        physicsClientId=cli, **kwargs,
    )
    frame = np.reshape(np.asarray(rgb, dtype=np.uint8), (shot.height, shot.width, 4))[:, :, :3].astype(np.float32) / 255.0
    return frame, _object_map(seg, shot)


def _thermal_frame(env: Any, ep: SolarEpisode, shot: View) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """A White Hot frame from the engine's thermal mode and its object map; blank and no map on an engine without it."""
    if not THERMAL:
        return np.zeros(THERMAL_SHAPE, dtype=np.float32), None
    view_matrix, projection = shot.matrices()
    # The engine heats sunlit surfaces from the light's direction, so a moon is put below the horizon.
    light = [0.0, 0.0, -1.0] if night(env) else env._light_direction
    flags = p.ER_SWARM_RAYCAST | p.ER_SWARM_THERMAL | p.ER_ALPHA_CUTOUT
    _w, _h, image, _depth, seg = p.getCameraImage(
        shot.width, shot.height, view_matrix, projection, renderer=p.ER_TINY_RENDERER, lightDirection=light,
        flags=flags, airTemperature=park.air_c(ep), skyTemperature=park.sky_c(ep),
        thermalSeed=(int(ep.seed) * 1000003 + ep.camera["captures"]) & 0x7FFFFFFF, physicsClientId=env.CLIENT,
    )
    white_hot = np.reshape(np.asarray(image, dtype=np.uint8), (shot.height, shot.width, 4))[:, :, :1]
    return white_hot.astype(np.float32) / 255.0, _object_map(seg, shot)
