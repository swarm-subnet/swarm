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
    START_PLATFORM,
    START_PLATFORM_SURFACE_Z,
    START_PLATFORM_TAKEOFF_BUFFER,
    START_PLATFORM_RANDOMIZE,
    START_PLATFORM_MIN_Z,
    START_PLATFORM_MAX_Z,
    CHALLENGE_TYPE_DISTRIBUTION,
    TYPE_1_N_OBSTACLES,
    TYPE_1_HEIGHT_SCALE,
    TYPE_1_SAFE_ZONE,
    TYPE_2_N_OBSTACLES,
    TYPE_2_HEIGHT_SCALE,
    TYPE_2_SAFE_ZONE,
    TYPE_3_N_OBSTACLES,
    TYPE_3_HEIGHT_SCALE,
    TYPE_3_SAFE_ZONE,
    TYPE_4_N_OBSTACLES,
    TYPE_4_HEIGHT_SCALE,
    TYPE_4_SAFE_ZONE,
    TYPE_5_N_OBSTACLES,
    TYPE_5_HEIGHT_SCALE,
    TYPE_5_SAFE_ZONE,
    TYPE_5_PLATFORM_SPEED,
    TYPE_5_ORBIT_RADIUS,
    TYPE_5_MOVEMENT_PATTERNS,
)
from typing import Optional   

def get_platform_height_for_seed(seed: int, start_pos: Tuple[float, float, float]) -> float:
    """Calculate platform height for a given seed and start position.
    
    This ensures env_builder uses the same random platform height as task generation.
    """
    if not START_PLATFORM or not START_PLATFORM_RANDOMIZE:
        return START_PLATFORM_SURFACE_Z
    
    # Use same random sequence as _random_start
    rng = random.Random(seed)
    # Skip the x, y generation to get to the platform height part
    rng.uniform(-WORLD_RANGE, WORLD_RANGE)  # x
    rng.uniform(-WORLD_RANGE, WORLD_RANGE)  # y
    # Now generate the same platform height
    return rng.uniform(START_PLATFORM_MIN_Z, START_PLATFORM_MAX_Z)

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
    if START_PLATFORM:
        if START_PLATFORM_RANDOMIZE:
            # Random platform height within specified range
            platform_z = seed_rng.uniform(START_PLATFORM_MIN_Z, START_PLATFORM_MAX_Z)
        else:
            # Fixed platform height
            platform_z = START_PLATFORM_SURFACE_Z
        z = platform_z + START_PLATFORM_TAKEOFF_BUFFER
    else:
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
        if START_PLATFORM:
            start_z = START_PLATFORM_SURFACE_Z + START_PLATFORM_TAKEOFF_BUFFER
        else:
            start_z = 1.5
        start = (0.0, 0.0, start_z)
        goal = _goal(rng)
    
    # Select challenge type AFTER generating start/goal to avoid affecting random sequence
    challenge_types = list(CHALLENGE_TYPE_DISTRIBUTION.keys())
    probabilities = list(CHALLENGE_TYPE_DISTRIBUTION.values())
    chosen_type = rng.choices(challenge_types, weights=probabilities, k=1)[0]
    
    return MapTask(
        map_seed = seed,
        start    = start,
        goal     = goal,
        sim_dt   = sim_dt,
        horizon  = horizon,
        challenge_type = chosen_type,
        version  = "1",
    )
