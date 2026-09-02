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

from __future__ import annotations

import math
import random
from typing import Tuple

from swarm.constants import SAR_SEARCH_RADIUS as SEARCH_RADIUS_M


def sample_search_centre(
    rng: random.Random,
    victim_centre_xy: Tuple[float, float],
    radius: float = SEARCH_RADIUS_M,
) -> Tuple[float, float]:
    u = rng.random()
    v = rng.random()
    r = radius * math.sqrt(u)
    theta = 2.0 * math.pi * v
    cx, cy = float(victim_centre_xy[0]), float(victim_centre_xy[1])
    return (cx + r * math.cos(theta), cy + r * math.sin(theta))
