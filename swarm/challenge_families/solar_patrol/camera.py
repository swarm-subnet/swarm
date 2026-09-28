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

A captured frame goes into the episode's rgb or thermal image, whichever feed the model picked, and the other one
is left at zero. The thermal picture itself comes from the engine's thermal render mode (task 2).

Stand-in: the gimbal jumps to the tilt asked for, the feed switches at once, and every frame is blank.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from swarm.constants import SIM_DT

from .contract import GIMBAL_TILT_RANGE_DEG, NIGHT_MODES, STATE_SLICES, Command, put
from .episode import SolarEpisode

RENDER_BACKEND = "tiny"
FRAME_HZ = 2.0
FRAME_STEPS = int(round(1.0 / (FRAME_HZ * SIM_DT)))


def reset(env: Any, ep: SolarEpisode) -> None:
    """The gimbal level, the colour feed, night mode off, and a first frame at take-off time zero."""
    ep.camera = {"tilt_deg": 0.0, "thermal": False, "night_mode": "off", "captured_s": 0.0}


def request(env: Any, ep: SolarEpisode, command: Command) -> None:
    """Take the tilt, feed and night mode the model asked for."""
    low, high = GIMBAL_TILT_RANGE_DEG
    ep.camera["tilt_deg"] = low + (command.gimbal_tilt + 1.0) / 2.0 * (high - low)
    ep.camera["thermal"] = command.thermal
    ep.camera["night_mode"] = command.night_mode


def update(env: Any, ep: SolarEpisode) -> None:
    """Capture a new frame at the camera's own rate."""
    if ep.step % FRAME_STEPS == 0:
        ep.camera["captured_s"] = ep.time_s


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The gimbal tilt, the feed shown, the age of its frame and the night mode."""
    put(state, STATE_SLICES, "gimbal_tilt_deg", ep.camera["tilt_deg"])
    put(state, STATE_SLICES, "camera_feed", float(ep.camera["thermal"]))
    put(state, STATE_SLICES, "frame_age_s", ep.time_s - ep.camera["captured_s"])
    put(state, STATE_SLICES, "night_mode", NIGHT_MODES.index(ep.camera["night_mode"]))
