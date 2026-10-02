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

"""The state one patrol keeps between steps: the facts every part reads, then one slot per part for its own."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from swarm.constants import SIM_DT

from .contract import Command, Frames, Outcome


@dataclass(frozen=True)
class Setpoint:
    """What the flight controller holds for one control step: a world velocity and a turn rate, or the motors off."""

    velocity_mps: tuple = (0.0, 0.0, 0.0)            # east, north, up
    yaw_rate_rad_s: float = 0.0                      # counter-clockwise seen from above, the simulator's own sense
    motors_on: bool = True


@dataclass
class SolarEpisode:
    """One patrol, from the dock opening to the end of the seed.

    The shared fields are the ones more than one part reads. Each part keeps whatever else it needs in its own
    slot, and no other part reaches into it.
    """

    seed: int
    step: int = 0                                    # control steps since the patrol started
    phase: str = "docked"                            # one of FLIGHT_PHASES, moved only by the dock part
    dock_position: np.ndarray = field(default_factory=lambda: np.zeros(3))  # world centre of the landing pad
    dock_yaw: float = 0.0
    dock_uid: int = -1
    fence: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))     # the fence line, world metres
    terrain_uids: frozenset = frozenset()
    command: Optional[Command] = None                # this decision's decoded action
    previous_action: Optional[np.ndarray] = None
    outcome: Outcome = field(default_factory=Outcome)
    frames: Frames = field(default_factory=Frames)
    view_step: int = -1                              # the decision step the view is shown at
    view: dict = field(default_factory=dict)

    park: Any = None
    dock: Any = None
    airframe: Any = None
    flight_limit: Any = None
    wind: Any = None
    theft: Any = None
    decoys: Any = None
    coverage: Any = None
    camera: Any = None
    zoom: Any = None
    laser: Any = None
    ground_distance: Any = None
    drone_state: Any = None
    sensor_noise: Any = None
    reports: Any = None
    score: Any = None
    outputs: Any = None

    @property
    def time_s(self) -> float:
        """Seconds of patrol flown so far."""
        return self.step * SIM_DT

    def end(self, reason: str) -> None:
        """Close the patrol for the first reason that fires; a later one never overwrites it."""
        if not self.outcome.end_reason:
            self.outcome.end_reason = reason
