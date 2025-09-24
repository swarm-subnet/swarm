# swarm/validator/task_gen.py
from __future__ import annotations
import random, math
from typing import Tuple
from swarm.protocol import MapTask

from swarm.constants import (
    R_MIN,
    R_MAX,
    H_MIN,
    H_MAX,
    WORLD_RANGE,
    RANDOM_START,
    START_H_MIN,
    START_H_MAX,
)
from typing import Optional   

def _goal(seed_rng: random.Random) -> Tuple[float, float, float]:
    """Legacy goal generation from origin."""
    ang  = seed_rng.uniform(0, 2*math.pi)
    r    = seed_rng.uniform(R_MIN, R_MAX)
    x, y = r*math.cos(ang), r*math.sin(ang)
    z    = seed_rng.uniform(H_MIN, H_MAX)
    return x, y, z

def _random_start(seed_rng: random.Random) -> Tuple[float, float, float]:
    """Generate random start position within world bounds."""
    x = seed_rng.uniform(-WORLD_RANGE, WORLD_RANGE)
    y = seed_rng.uniform(-WORLD_RANGE, WORLD_RANGE)
    z = seed_rng.uniform(START_H_MIN, START_H_MAX)
    return x, y, z

def _goal_from_start(seed_rng: random.Random, start: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Generate goal at required distance from start within world bounds."""
    start_x, start_y, start_z = start
    
    for _ in range(100):
        angle = seed_rng.uniform(0, 2*math.pi)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        
        max_radius_x = float('inf')
        max_radius_y = float('inf')
        
        if abs(cos_a) > 1e-8:
            if cos_a > 0:
                max_radius_x = (WORLD_RANGE - start_x) / cos_a
            else:
                max_radius_x = (-WORLD_RANGE - start_x) / cos_a
        
        if abs(sin_a) > 1e-8:
            if sin_a > 0:
                max_radius_y = (WORLD_RANGE - start_y) / sin_a
            else:
                max_radius_y = (-WORLD_RANGE - start_y) / sin_a
        
        max_radius = min(max_radius_x, max_radius_y, R_MAX)
        
        if max_radius >= R_MIN:
            radius = seed_rng.uniform(R_MIN, min(max_radius * 0.999, R_MAX))
            x = start_x + radius * cos_a
            y = start_y + radius * sin_a
            z = seed_rng.uniform(H_MIN, H_MAX)
            
            if (-WORLD_RANGE <= x <= WORLD_RANGE and 
                -WORLD_RANGE <= y <= WORLD_RANGE):
                return x, y, z
    
    # Fallback: generate simple goal within constraints
    angle = seed_rng.uniform(0, 2*math.pi)
    radius = seed_rng.uniform(R_MIN, R_MAX)
    x = start_x + radius * math.cos(angle)
    y = start_y + radius * math.sin(angle)
    z = seed_rng.uniform(H_MIN, H_MAX)
    
    # Clamp to world bounds if needed
    x = max(-WORLD_RANGE, min(WORLD_RANGE, x))
    y = max(-WORLD_RANGE, min(WORLD_RANGE, y))
    
    return x, y, z

def random_task(sim_dt: float, horizon: float, seed: Optional[int] = None) -> MapTask:
    if seed is None:
        # If no seed is provided, generate a random one
        seed  = random.randrange(2**32)
    rng   = random.Random(seed)
    if RANDOM_START:
        start = _random_start(rng)
        goal = _goal_from_start(rng, start)
    else:
        start = (0.0, 0.0, 1.5)
        goal = _goal(rng)
    return MapTask(
        map_seed = seed,
        start    = start,
        goal     = goal,
        sim_dt   = sim_dt,
        horizon  = horizon,
        version  = "1",
    )
