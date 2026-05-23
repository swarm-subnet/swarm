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
