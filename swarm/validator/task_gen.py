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
    SEARCH_RADIUS_MIN, SEARCH_RADIUS_MAX,
    MOVING_PLATFORM_PROB, MOVING_PLATFORM_SEED_OFFSET,
    TYPE_1_WORLD_RANGE, TYPE_1_R_MIN, TYPE_1_R_MAX, TYPE_1_H_MIN, TYPE_1_H_MAX,
    TYPE_1_START_H_MIN, TYPE_1_START_H_MAX, TYPE_1_HORIZON,
    TYPE_2_WORLD_RANGE, TYPE_2_R_MIN, TYPE_2_R_MAX, TYPE_2_H_MIN, TYPE_2_H_MAX,
    TYPE_2_START_H_MIN, TYPE_2_START_H_MAX, TYPE_2_HORIZON,
    TYPE_3_R_MIN, TYPE_3_R_MAX, TYPE_3_H_MIN, TYPE_3_H_MAX,
    TYPE_3_START_H_MIN, TYPE_3_START_H_MAX, TYPE_3_HORIZON,
    TYPE_3_WORLD_RANGE_RATIO, TYPE_3_VILLAGE_RANGE,
)
from swarm.core.mountain_generator import get_terrain_z, get_global_scale, get_mountain_subtype


TYPE_PARAMS = {
    1: {
        'world_range': TYPE_1_WORLD_RANGE,
        'r_min': TYPE_1_R_MIN, 'r_max': TYPE_1_R_MAX,
        'h_min': TYPE_1_H_MIN, 'h_max': TYPE_1_H_MAX,
        'start_h_min': TYPE_1_START_H_MIN, 'start_h_max': TYPE_1_START_H_MAX,
        'horizon': TYPE_1_HORIZON,
    },
    2: {
        'world_range': TYPE_2_WORLD_RANGE,
        'r_min': TYPE_2_R_MIN, 'r_max': TYPE_2_R_MAX,
        'h_min': TYPE_2_H_MIN, 'h_max': TYPE_2_H_MAX,
        'start_h_min': TYPE_2_START_H_MIN, 'start_h_max': TYPE_2_START_H_MAX,
        'horizon': TYPE_2_HORIZON,
    },
}


def get_type_params(challenge_type: int) -> dict:
    return TYPE_PARAMS.get(challenge_type, TYPE_PARAMS[1])


def get_platform_height_for_seed(seed: int, challenge_type: int = 1) -> float:
    if challenge_type == 3:
        return 0.0
    if not START_PLATFORM or not START_PLATFORM_RANDOMIZE:
        return START_PLATFORM_SURFACE_Z

    params = get_type_params(challenge_type)
    world_range = params['world_range']

    rng = random.Random(seed)
    rng.uniform(-world_range, world_range)
    rng.uniform(-world_range, world_range)
    return rng.uniform(START_PLATFORM_MIN_Z, START_PLATFORM_MAX_Z)


def _get_type3_world_range(seed: int) -> float:
    subtype = get_mountain_subtype(seed)
    if subtype == 2:
        return TYPE_3_VILLAGE_RANGE
    gs = get_global_scale(seed)
    half = 250.0 * gs
    return half * TYPE_3_WORLD_RANGE_RATIO


def _get_type3_surface_z(x: float, y: float, seed: int) -> float:
    subtype = get_mountain_subtype(seed)
    if subtype == 2:
        return 0.0
    gs = get_global_scale(seed)
    return get_terrain_z(x, y, seed, gs)


def _random_start(seed_rng: random.Random, params: dict,
                  challenge_type: int = 1, seed: int = 0) -> Tuple[float, float, float]:
    world_range = params['world_range']
    x = seed_rng.uniform(-world_range, world_range)
    y = seed_rng.uniform(-world_range, world_range)

    if challenge_type == 3:
        seed_rng.uniform(0, 1)
        terrain_z = _get_type3_surface_z(x, y, seed)
        z = terrain_z + START_PLATFORM_TAKEOFF_BUFFER
    elif START_PLATFORM:
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
    params: dict,
    challenge_type: int = 1,
    seed: int = 0,
) -> Tuple[float, float, float]:
    start_x, start_y, start_z = start
    world_range = params['world_range']
    r_min, r_max = params['r_min'], params['r_max']
    h_min, h_max = params['h_min'], params['h_max']
    start_surface_z = (
        start_z - START_PLATFORM_TAKEOFF_BUFFER if challenge_type == 3 else start_z
    )

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

            if challenge_type == 3:
                seed_rng.uniform(0, 1)
                surface_z = _get_type3_surface_z(x, y, seed)
                z = surface_z
                dist_3d = math.sqrt((x - start_x)**2 + (y - start_y)**2 + (surface_z - start_surface_z)**2)
                if r_min <= dist_3d <= r_max and -world_range <= x <= world_range and -world_range <= y <= world_range:
                    d2 = math.hypot(x - start_x, y - start_y)
                    if d2 < r_min:
                        scale = r_min / max(d2, 1e-8)
                        x = start_x + (x - start_x) * scale
                        y = start_y + (y - start_y) * scale
                        x = max(-world_range, min(world_range, x))
                        y = max(-world_range, min(world_range, y))
                        d2_after = math.hypot(x - start_x, y - start_y)
                        if d2_after < r_min:
                            continue
                        z = _get_type3_surface_z(x, y, seed)
                    return x, y, z
            else:
                z = seed_rng.uniform(h_min, h_max)
                if -world_range <= x <= world_range and -world_range <= y <= world_range:
                    d2 = math.hypot(x - start_x, y - start_y)
                    if d2 < r_min:
                        scale = r_min / max(d2, 1e-8)
                        x = start_x + (x - start_x) * scale
                        y = start_y + (y - start_y) * scale
                        x = max(-world_range, min(world_range, x))
                        y = max(-world_range, min(world_range, y))
                        if math.hypot(x - start_x, y - start_y) < r_min:
                            continue
                    return x, y, z

    angle = seed_rng.uniform(0, 2 * math.pi)
    radius = seed_rng.uniform(r_min, r_max)
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    x = start_x + radius * cos_a
    y = start_y + radius * sin_a

    x = max(-world_range, min(world_range, x))
    y = max(-world_range, min(world_range, y))

    dist_2d = math.hypot(x - start_x, y - start_y)
    if dist_2d < r_min:
        if dist_2d <= 1e-8:
            dir_x, dir_y = cos_a, sin_a
        else:
            dir_x = (x - start_x) / dist_2d
            dir_y = (y - start_y) / dist_2d

        x = start_x + r_min * dir_x
        y = start_y + r_min * dir_y
        x = max(-world_range, min(world_range, x))
        y = max(-world_range, min(world_range, y))

        if math.hypot(x - start_x, y - start_y) < r_min:
            candidates = [
                (start_x + r_min, start_y),
                (start_x - r_min, start_y),
                (start_x, start_y + r_min),
                (start_x, start_y - r_min),
            ]
            for cx, cy in candidates:
                cx = max(-world_range, min(world_range, cx))
                cy = max(-world_range, min(world_range, cy))
                if math.hypot(cx - start_x, cy - start_y) >= r_min:
                    x, y = cx, cy
                    break

    if challenge_type == 3:
        seed_rng.uniform(0, 1)
        z = _get_type3_surface_z(x, y, seed)
    else:
        z = seed_rng.uniform(h_min, h_max)

    return x, y, z


def _goal_from_origin(seed_rng: random.Random, params: dict) -> Tuple[float, float, float]:
    r_min, r_max = params['r_min'], params['r_max']
    h_min, h_max = params['h_min'], params['h_max']

    ang = seed_rng.uniform(0, 2 * math.pi)
    r = seed_rng.uniform(r_min, r_max)
    x, y = r * math.cos(ang), r * math.sin(ang)
    z = seed_rng.uniform(h_min, h_max)
    return x, y, z


def random_task(sim_dt: float, seed: Optional[int] = None) -> MapTask:
    if seed is None:
        seed = random.randrange(2**32)
    rng = random.Random(seed)

    challenge_types = list(CHALLENGE_TYPE_DISTRIBUTION.keys())
    probabilities = list(CHALLENGE_TYPE_DISTRIBUTION.values())
    type_rng = random.Random(seed + 999999)
    chosen_type = type_rng.choices(challenge_types, weights=probabilities, k=1)[0]

    if chosen_type == 3:
        params = {
            'world_range': _get_type3_world_range(seed),
            'r_min': TYPE_3_R_MIN, 'r_max': TYPE_3_R_MAX,
            'h_min': TYPE_3_H_MIN, 'h_max': TYPE_3_H_MAX,
            'start_h_min': TYPE_3_START_H_MIN, 'start_h_max': TYPE_3_START_H_MAX,
            'horizon': TYPE_3_HORIZON,
        }
    else:
        params = get_type_params(chosen_type)

    search_rng = random.Random(seed + 888888)
    search_radius = search_rng.uniform(SEARCH_RADIUS_MIN, SEARCH_RADIUS_MAX)

    platform_rng = random.Random((seed + MOVING_PLATFORM_SEED_OFFSET) & 0xFFFFFFFF)
    moving_platform = platform_rng.random() < MOVING_PLATFORM_PROB.get(chosen_type, 0.0)

    if RANDOM_START:
        start = _random_start(rng, params, challenge_type=chosen_type, seed=seed)
        goal = _goal_from_start(rng, start, params, challenge_type=chosen_type, seed=seed)
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
        horizon=params['horizon'],
        challenge_type=chosen_type,
        search_radius=search_radius,
        moving_platform=moving_platform,
        version="1",
    )
