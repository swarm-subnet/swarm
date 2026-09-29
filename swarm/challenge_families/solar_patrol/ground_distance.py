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

"""Ground distance (task 13): how far the ground is straight down, from the sensors under the drone.

The dock sends it as the downward obstacle distance, with whether downward sensing works (DJI hsi_info_push). By day
the downward cameras measure it and at night the infrared sensor does; V1 gives both the same range. The reading is
the clean distance to the first thing below, whatever it is; the sensor errors go on in sensor_noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pybullet as p

from .contract import DECISION_STEPS, STATE_SLICES, put
from .episode import SolarEpisode

MIN_RANGE_M = 0.5
MAX_RANGE_M = 16.0
OUT_OF_RANGE_M = 60.0                    # DJI's perception SDK reports 60000 mm when it detects nothing
READING_STEPS = DECISION_STEPS           # 10 Hz
SENSOR = np.array([0.0, 0.0, -0.062])    # centre of the belly in the drone frame, from the M4TD's collision boxes
_SELF_HITS = 8                           # the aircraft's own parts a ray may pass before it gives up


def reset(env: Any, ep: SolarEpisode) -> None:
    """Take the first reading where the drone stands, with downward sensing working."""
    ep.ground_distance = {"distance_m": _read(env), "working": True}


def update(env: Any, ep: SolarEpisode) -> None:
    """Take a new reading ten times a second and hold it in between."""
    if ep.ground_distance is not None and ep.step % READING_STEPS == 0:
        ep.ground_distance["distance_m"] = _read(env)


def observe(env: Any, ep: SolarEpisode, state: np.ndarray) -> None:
    """The last reading and whether downward sensing works."""
    reading = ep.ground_distance or {"distance_m": OUT_OF_RANGE_M, "working": True}
    put(state, STATE_SLICES, "ground_distance_m", reading["distance_m"])
    put(state, STATE_SLICES, "downward_sensing_ok", float(reading["working"]))


def _read(env: Any) -> float:
    """Distance from the sensor straight down to the first thing that is not the aircraft, held to the range."""
    cli, aircraft = env.CLIENT, int(env.DRONE_IDS[0])
    pos, orn = p.getBasePositionAndOrientation(aircraft, physicsClientId=cli)
    sensor = np.asarray(pos, dtype=float) + np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3) @ SENSOR
    start, end = sensor.tolist(), [sensor[0], sensor[1], sensor[2] - MAX_RANGE_M]
    for _ in range(_SELF_HITS):
        uid, _link, _fraction, hit, _normal = p.rayTest(start, end, physicsClientId=cli)[0]
        if uid < 0:
            return OUT_OF_RANGE_M
        if uid != aircraft:
            return max(MIN_RANGE_M, float(sensor[2] - hit[2]))
        start = [sensor[0], sensor[1], hit[2] - 1e-3]
    return OUT_OF_RANGE_M
