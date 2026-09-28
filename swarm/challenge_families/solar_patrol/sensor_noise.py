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

"""Sensor noise and delay (task 16): the errors the real aircraft and its link carry, drawn from the seed.

The delay acts on the command before anything applies it; the errors act on the state after every part has written
its clean value, and on the images once they are captured.

Stand-in: no delay and no error.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .contract import Command
from .episode import SolarEpisode


def reset(env: Any, ep: SolarEpisode) -> None:
    """Nothing is drawn yet."""
    ep.sensor_noise = None


def delay(env: Any, ep: SolarEpisode, command: Command) -> Command:
    """The command the aircraft acts on this step, after the link's delay."""
    return command


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """Lay the sensors' errors over the clean state, in place."""
