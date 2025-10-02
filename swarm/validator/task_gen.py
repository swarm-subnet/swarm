# swarm/validator/task_gen.py
from __future__ import annotations
import random, math
from typing import Tuple, Optional
from swarm.protocol import MapTask

from swarm.constants import (
    H_MIN,
    H_MAX,
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
    TYPE_1_R_MIN, TYPE_1_R_MAX, TYPE_1_WORLD_RANGE,
    TYPE_2_R_MIN, TYPE_2_R_MAX, TYPE_2_WORLD_RANGE,
    TYPE_3_R_MIN, TYPE_3_R_MAX, TYPE_3_WORLD_RANGE,
    TYPE_4_R_MIN, TYPE_4_R_MAX, TYPE_4_WORLD_RANGE,
    TYPE_5_R_MIN, TYPE_5_R_MAX, TYPE_5_WORLD_RANGE,
)   

def get_platform_height_for_seed(seed: int, start_pos: Tuple[float, float, float], world_range: float = 30) -> float:
    if not START_PLATFORM or not START_PLATFORM_RANDOMIZE:
        return START_PLATFORM_SURFACE_Z
    rng = random.Random(seed)
    rng.uniform(-world_range, world_range)
    rng.uniform(-world_range, world_range)
    return rng.uniform(START_PLATFORM_MIN_Z, START_PLATFORM_MAX_Z)

def _goal(seed_rng: random.Random, r_min: float, r_max: float) -> Tuple[float, float, float]:
    ang = seed_rng.uniform(0, 2*math.pi)
    r = seed_rng.uniform(r_min, r_max)
    x, y = r*math.cos(ang), r*math.sin(ang)
    z = seed_rng.uniform(H_MIN, H_MAX)
    return x, y, z

def _random_start(seed_rng: random.Random, world_range: float) -> Tuple[float, float, float]:
    x = seed_rng.uniform(-world_range, world_range)
    y = seed_rng.uniform(-world_range, world_range)
    if START_PLATFORM:
        if START_PLATFORM_RANDOMIZE:
            platform_z = seed_rng.uniform(START_PLATFORM_MIN_Z, START_PLATFORM_MAX_Z)
        else:
            platform_z = START_PLATFORM_SURFACE_Z
        z = platform_z + START_PLATFORM_TAKEOFF_BUFFER
    else:
        z = seed_rng.uniform(START_H_MIN, START_H_MAX)
    return x, y, z

def _goal_from_start(seed_rng: random.Random, start: Tuple[float, float, float], r_min: float, r_max: float, world_range: float) -> Tuple[float, float, float]:
    start_x, start_y, start_z = start
    for _ in range(100):
        angle = seed_rng.uniform(0, 2*math.pi)
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        max_radius_x = float('inf')
        max_radius_y = float('inf')
        if abs(cos_a) > 1e-8:
            if cos_a > 0:
                max_radius_x = (world_range - start_x) / cos_a
            else:
                max_radius_x = (-world_range - start_x) / cos_a
        if abs(sin_a) > 1e-8:
            if sin_a > 0:
                max_radius_y = (world_range - start_y) / sin_a
            else:
                max_radius_y = (-world_range - start_y) / sin_a
        max_radius = min(max_radius_x, max_radius_y, r_max)
        if max_radius >= r_min:
            radius = seed_rng.uniform(r_min, min(max_radius * 0.999, r_max))
            x = start_x + radius * cos_a
            y = start_y + radius * sin_a
            z = seed_rng.uniform(H_MIN, H_MAX)
            if (-world_range <= x <= world_range and -world_range <= y <= world_range):
                return x, y, z
    angle = seed_rng.uniform(0, 2*math.pi)
    radius = seed_rng.uniform(r_min, r_max)
    x = start_x + radius * math.cos(angle)
    y = start_y + radius * math.sin(angle)
    z = seed_rng.uniform(H_MIN, H_MAX)
    x = max(-world_range, min(world_range, x))
    y = max(-world_range, min(world_range, y))
    return x, y, z

def random_task(sim_dt: float, horizon: float, seed: Optional[int] = None) -> MapTask:
    if seed is None:
        seed = random.randrange(2**32)
    rng = random.Random(seed)
    
    challenge_types = list(CHALLENGE_TYPE_DISTRIBUTION.keys())
    probabilities = list(CHALLENGE_TYPE_DISTRIBUTION.values())
    chosen_type = rng.choices(challenge_types, weights=probabilities, k=1)[0]
    
    if chosen_type == 1:
        r_min, r_max, world_range = TYPE_1_R_MIN, TYPE_1_R_MAX, TYPE_1_WORLD_RANGE
    elif chosen_type == 2:
        r_min, r_max, world_range = TYPE_2_R_MIN, TYPE_2_R_MAX, TYPE_2_WORLD_RANGE
    elif chosen_type == 3:
        r_min, r_max, world_range = TYPE_3_R_MIN, TYPE_3_R_MAX, TYPE_3_WORLD_RANGE
    elif chosen_type == 4:
        r_min, r_max, world_range = TYPE_4_R_MIN, TYPE_4_R_MAX, TYPE_4_WORLD_RANGE
    elif chosen_type == 5:
        r_min, r_max, world_range = TYPE_5_R_MIN, TYPE_5_R_MAX, TYPE_5_WORLD_RANGE
    else:
        r_min, r_max, world_range = TYPE_1_R_MIN, TYPE_1_R_MAX, TYPE_1_WORLD_RANGE
    
    if RANDOM_START:
        start = _random_start(rng, world_range)
        goal = _goal_from_start(rng, start, r_min, r_max, world_range)
    else:
        if START_PLATFORM:
            start_z = START_PLATFORM_SURFACE_Z + START_PLATFORM_TAKEOFF_BUFFER
        else:
            start_z = 1.5
        start = (0.0, 0.0, start_z)
        goal = _goal(rng, r_min, r_max)
    
    return MapTask(
        map_seed=seed,
        start=start,
        goal=goal,
        sim_dt=sim_dt,
        horizon=horizon,
        challenge_type=chosen_type,
        version="1",
    )
