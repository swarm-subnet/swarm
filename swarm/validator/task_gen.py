# swarm/validator/task_gen.py
from __future__ import annotations
import random
import math
from typing import Tuple, Optional

from swarm.protocol import MapTask
from swarm.constants import (
    RANDOM_START,
    START_PLATFORM,
    START_PLATFORM_SURFACE_Z,
    START_PLATFORM_TAKEOFF_BUFFER,
    START_PLATFORM_RANDOMIZE,
    START_PLATFORM_MIN_Z,
    START_PLATFORM_MAX_Z,
    CHALLENGE_TYPE_DISTRIBUTION,
    TYPE_1_WORLD_RANGE, TYPE_1_R_MIN, TYPE_1_R_MAX, TYPE_1_H_MIN, TYPE_1_H_MAX,
    TYPE_1_START_H_MIN, TYPE_1_START_H_MAX,
    TYPE_2_WORLD_RANGE, TYPE_2_R_MIN, TYPE_2_R_MAX, TYPE_2_H_MIN, TYPE_2_H_MAX,
    TYPE_2_START_H_MIN, TYPE_2_START_H_MAX,
    TYPE_3_WORLD_RANGE, TYPE_3_R_MIN, TYPE_3_R_MAX, TYPE_3_H_MIN, TYPE_3_H_MAX,
    TYPE_3_START_H_MIN, TYPE_3_START_H_MAX,
    TYPE_4_WORLD_RANGE, TYPE_4_R_MIN, TYPE_4_R_MAX, TYPE_4_H_MIN, TYPE_4_H_MAX,
    TYPE_4_START_H_MIN, TYPE_4_START_H_MAX,
    TYPE_5_WORLD_RANGE, TYPE_5_R_MIN, TYPE_5_R_MAX, TYPE_5_H_MIN, TYPE_5_H_MAX,
    TYPE_5_START_H_MIN, TYPE_5_START_H_MAX,
)


TYPE_PARAMS = {
    1: {
        'world_range': TYPE_1_WORLD_RANGE,
        'r_min': TYPE_1_R_MIN, 'r_max': TYPE_1_R_MAX,
        'h_min': TYPE_1_H_MIN, 'h_max': TYPE_1_H_MAX,
        'start_h_min': TYPE_1_START_H_MIN, 'start_h_max': TYPE_1_START_H_MAX,
    },
    2: {
        'world_range': TYPE_2_WORLD_RANGE,
        'r_min': TYPE_2_R_MIN, 'r_max': TYPE_2_R_MAX,
        'h_min': TYPE_2_H_MIN, 'h_max': TYPE_2_H_MAX,
        'start_h_min': TYPE_2_START_H_MIN, 'start_h_max': TYPE_2_START_H_MAX,
    },
    3: {
        'world_range': TYPE_3_WORLD_RANGE,
        'r_min': TYPE_3_R_MIN, 'r_max': TYPE_3_R_MAX,
        'h_min': TYPE_3_H_MIN, 'h_max': TYPE_3_H_MAX,
        'start_h_min': TYPE_3_START_H_MIN, 'start_h_max': TYPE_3_START_H_MAX,
    },
    4: {
        'world_range': TYPE_4_WORLD_RANGE,
        'r_min': TYPE_4_R_MIN, 'r_max': TYPE_4_R_MAX,
        'h_min': TYPE_4_H_MIN, 'h_max': TYPE_4_H_MAX,
        'start_h_min': TYPE_4_START_H_MIN, 'start_h_max': TYPE_4_START_H_MAX,
    },
    5: {
        'world_range': TYPE_5_WORLD_RANGE,
        'r_min': TYPE_5_R_MIN, 'r_max': TYPE_5_R_MAX,
        'h_min': TYPE_5_H_MIN, 'h_max': TYPE_5_H_MAX,
        'start_h_min': TYPE_5_START_H_MIN, 'start_h_max': TYPE_5_START_H_MAX,
    },
}


def get_type_params(challenge_type: int) -> dict:
    return TYPE_PARAMS.get(challenge_type, TYPE_PARAMS[1])


def get_platform_height_for_seed(seed: int, challenge_type: int = 1) -> float:
    if not START_PLATFORM or not START_PLATFORM_RANDOMIZE:
        return START_PLATFORM_SURFACE_Z

    params = get_type_params(challenge_type)
    world_range = params['world_range']

    rng = random.Random(seed)
    rng.uniform(-world_range, world_range)
    rng.uniform(-world_range, world_range)
    return rng.uniform(START_PLATFORM_MIN_Z, START_PLATFORM_MAX_Z)


def _random_start(seed_rng: random.Random, params: dict) -> Tuple[float, float, float]:
    world_range = params['world_range']
    x = seed_rng.uniform(-world_range, world_range)
    y = seed_rng.uniform(-world_range, world_range)

    if START_PLATFORM:
        if START_PLATFORM_RANDOMIZE:
            platform_z = seed_rng.uniform(START_PLATFORM_MIN_Z, START_PLATFORM_MAX_Z)
        else:
            platform_z = START_PLATFORM_SURFACE_Z
        z = platform_z + START_PLATFORM_TAKEOFF_BUFFER
    else:
        z = seed_rng.uniform(params['start_h_min'], params['start_h_max'])

    return x, y, z


def _goal_from_start(
    seed_rng: random.Random,
    start: Tuple[float, float, float],
    params: dict
) -> Tuple[float, float, float]:
    start_x, start_y, _ = start
    world_range = params['world_range']
    r_min, r_max = params['r_min'], params['r_max']
    h_min, h_max = params['h_min'], params['h_max']

    for _ in range(100):
        angle = seed_rng.uniform(0, 2 * math.pi)
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
            z = seed_rng.uniform(h_min, h_max)

            if -world_range <= x <= world_range and -world_range <= y <= world_range:
                return x, y, z

    angle = seed_rng.uniform(0, 2 * math.pi)
    radius = seed_rng.uniform(r_min, r_max)
    x = start_x + radius * math.cos(angle)
    y = start_y + radius * math.sin(angle)
    z = seed_rng.uniform(h_min, h_max)

    x = max(-world_range, min(world_range, x))
    y = max(-world_range, min(world_range, y))

    return x, y, z


def _goal_from_origin(seed_rng: random.Random, params: dict) -> Tuple[float, float, float]:
    r_min, r_max = params['r_min'], params['r_max']
    h_min, h_max = params['h_min'], params['h_max']

    ang = seed_rng.uniform(0, 2 * math.pi)
    r = seed_rng.uniform(r_min, r_max)
    x, y = r * math.cos(ang), r * math.sin(ang)
    z = seed_rng.uniform(h_min, h_max)
    return x, y, z


def random_task(sim_dt: float, horizon: float, seed: Optional[int] = None) -> MapTask:
    if seed is None:
        seed = random.randrange(2**32)
    rng = random.Random(seed)

    challenge_types = list(CHALLENGE_TYPE_DISTRIBUTION.keys())
    probabilities = list(CHALLENGE_TYPE_DISTRIBUTION.values())
    type_rng = random.Random(seed + 999999)
    chosen_type = type_rng.choices(challenge_types, weights=probabilities, k=1)[0]

    params = get_type_params(chosen_type)

    if RANDOM_START:
        start = _random_start(rng, params)
        goal = _goal_from_start(rng, start, params)
    else:
        if START_PLATFORM:
            start_z = START_PLATFORM_SURFACE_Z + START_PLATFORM_TAKEOFF_BUFFER
        else:
            start_z = 1.5
        start = (0.0, 0.0, start_z)
        goal = _goal_from_origin(rng, params)

    return MapTask(
        map_seed=seed,
        start=start,
        goal=goal,
        sim_dt=sim_dt,
        horizon=horizon,
        challenge_type=chosen_type,
        version="1",
    )
