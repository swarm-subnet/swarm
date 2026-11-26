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

"""Model outputs on the field: Sentinel's, every value checked and applied to the drone and its camera, with the
report handed to the field's own judge."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..solar_patrol import outputs as sentinel_outputs
from ..solar_patrol.contract import Command
from . import reports
from .contract import ACTION_HIGH, ACTION_LOW
from .episode import FieldEpisode

reset = sentinel_outputs.reset
setpoint = sentinel_outputs.setpoint
fly = sentinel_outputs.fly


def clip(action: np.ndarray) -> np.ndarray:
    """The action with every value held inside its contract bounds."""
    return np.clip(action, ACTION_LOW, ACTION_HIGH).astype(np.float32)


def apply(env: Any, ep: FieldEpisode, command: Command) -> None:
    """Sentinel's gimbal, camera and zoom handling, then the report, if one was pressed, judged by the field."""
    sentinel_outputs.apply(env, ep, replace(command, report=None))
    if command.report is not None:
        reports.submit(env, ep, command.report)
