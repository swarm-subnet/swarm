# swarm/validator/task_gen.py
from __future__ import annotations
import random, math
from typing import Tuple
from swarm.protocol import MapTask

from swarm.constants import R_MIN, R_MAX, H_MIN, H_MAX  # constants for goal radius

def _goal(seed_rng: random.Random) -> Tuple[float, float, float]:
    ang  = seed_rng.uniform(0, 2*math.pi)
    r    = seed_rng.uniform(R_MIN, R_MAX)
    x, y = r*math.cos(ang), r*math.sin(ang)
    z    = seed_rng.uniform(H_MIN, H_MAX)
    return x, y, z

def random_task(sim_dt: float, horizon: float) -> MapTask:
    seed  = random.randrange(2**32)
    rng   = random.Random(seed)
    return MapTask(
        map_seed = seed,
        start    = (0.0, 0.0, 1.5),
        goal     = _goal(rng),
        sim_dt   = sim_dt,
        horizon  = horizon,
        version  = "1",
    )
