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

The view goes into the episode's zoom image. Night vision belongs here too: it works only through the 7x lens.

Stand-in: requests are counted against the cap and answered with a blank view at the next decision.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contract import DECISION_STEPS, MAX_ZOOMS, STATE_SLICES, Command, put
from .episode import SolarEpisode


def reset(env: Any, ep: SolarEpisode) -> None:
    """No zoom taken yet and night vision off."""
    ep.zoom = {"lens": 0, "arrived_s": 0.0, "pending": None, "pending_step": 0, "night_vision": False}


def request(env: Any, ep: SolarEpisode, command: Command) -> None:
    """Take the night vision switch, and queue a zoom while the patrol still has zooms left."""
    ep.zoom["night_vision"] = command.night_vision
    if command.zoom is None or ep.outcome.zooms_used >= MAX_ZOOMS:
        return
    ep.outcome.zooms_used += 1
    ep.zoom["pending"] = command.zoom
    ep.zoom["pending_step"] = ep.step


def update(env: Any, ep: SolarEpisode) -> None:
    """Deliver a queued zoom once a full decision has passed since it was asked for."""
    request = ep.zoom["pending"]
    if request is not None and ep.step - ep.zoom["pending_step"] >= DECISION_STEPS:
        ep.zoom.update(lens=request.lens, arrived_s=ep.time_s, pending=None)


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """Night vision, the lens of the zoom view shown, its age, and the zooms left."""
    put(state, STATE_SLICES, "night_vision", float(ep.zoom["night_vision"]))
    put(state, STATE_SLICES, "zoom_lens", ep.zoom["lens"])
    put(state, STATE_SLICES, "zoom_age_s", ep.time_s - ep.zoom["arrived_s"] if ep.zoom["lens"] else 0.0)
    put(state, STATE_SLICES, "zooms_left", MAX_ZOOMS - ep.outcome.zooms_used)
