"""Progressive curriculum helpers for the stage-02 privileged expert.

This module is intentionally training-only. It lets us create easier custom
tasks than the validator benchmark so the privileged teacher can be developed
incrementally instead of one-shot against the full distribution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable

from swarm.constants import (
    TYPE_1_H_MAX,
    TYPE_1_H_MIN,
    TYPE_1_WORLD_RANGE,
    TYPE_2_H_MAX,
    TYPE_2_H_MIN,
    TYPE_2_WORLD_RANGE,
    TYPE_3_H_MAX,
    TYPE_3_H_MIN,
    TYPE_3_R_MAX,
    TYPE_3_VILLAGE_RANGE,
    TYPE_4_H_MAX,
    TYPE_4_H_MIN,
    TYPE_4_WORLD_RANGE_X,
    TYPE_4_WORLD_RANGE_Y,
    TYPE_6_H_MAX,
    TYPE_6_H_MIN,
    TYPE_6_WORLD_RANGE,
)
from swarm.core.mountain_generator import get_global_scale
from swarm.protocol import MapTask

from training_env import build_task


@dataclass(frozen=True)
class ExpertCurriculumStage:
    name: str
    challenge_type: int
    radius_min_m: float
    radius_max_m: float
    moving_platform: bool = False
    num_episodes: int = 8
    seed_start: int = 20000
    goal_z_min_m: float | None = None
    goal_z_max_m: float | None = None
    goal_z_relative_to_start: bool = False


PROGRESSIVE_STATIC_STAGES: tuple[ExpertCurriculumStage, ...] = (
    ExpertCurriculumStage(
        "open_r1_5_static",
        challenge_type=2,
        radius_min_m=1.0,
        radius_max_m=5.0,
        goal_z_min_m=-0.5,
        goal_z_max_m=0.5,
        goal_z_relative_to_start=True,
    ),
    ExpertCurriculumStage("open_r5_10_static", challenge_type=2, radius_min_m=5.0, radius_max_m=10.0, seed_start=20100),
    ExpertCurriculumStage("city_r5_10_static", challenge_type=1, radius_min_m=5.0, radius_max_m=10.0, seed_start=20200),
    ExpertCurriculumStage("forest_r5_10_static", challenge_type=6, radius_min_m=5.0, radius_max_m=10.0, seed_start=20300),
    ExpertCurriculumStage("warehouse_r5_10_static", challenge_type=5, radius_min_m=5.0, radius_max_m=10.0, seed_start=20400),
    ExpertCurriculumStage("village_r10_20_static", challenge_type=4, radius_min_m=10.0, radius_max_m=20.0, seed_start=20500),
    ExpertCurriculumStage("mountain_r10_20_static", challenge_type=3, radius_min_m=10.0, radius_max_m=20.0, seed_start=20600),
)


def _xy_bounds_for_type(challenge_type: int, seed: int) -> tuple[tuple[float, float], tuple[float, float]]:
    if challenge_type == 1:
        return (-TYPE_1_WORLD_RANGE, TYPE_1_WORLD_RANGE), (-TYPE_1_WORLD_RANGE, TYPE_1_WORLD_RANGE)
    if challenge_type == 2:
        return (-TYPE_2_WORLD_RANGE, TYPE_2_WORLD_RANGE), (-TYPE_2_WORLD_RANGE, TYPE_2_WORLD_RANGE)
    if challenge_type == 3:
        half = 250.0 * get_global_scale(seed) * 0.60
        return (-half, half), (-half, half)
    if challenge_type == 4:
        return (-TYPE_3_VILLAGE_RANGE, TYPE_3_VILLAGE_RANGE), (-TYPE_3_VILLAGE_RANGE, TYPE_3_VILLAGE_RANGE)
    if challenge_type == 5:
        return (-TYPE_4_WORLD_RANGE_X, TYPE_4_WORLD_RANGE_X), (-TYPE_4_WORLD_RANGE_Y, TYPE_4_WORLD_RANGE_Y)
    if challenge_type == 6:
        return (-TYPE_6_WORLD_RANGE, TYPE_6_WORLD_RANGE), (-TYPE_6_WORLD_RANGE, TYPE_6_WORLD_RANGE)
    raise ValueError(f"Unsupported challenge_type={challenge_type}")


def _z_bounds_for_type(challenge_type: int) -> tuple[float, float]:
    if challenge_type == 1:
        return TYPE_1_H_MIN, TYPE_1_H_MAX
    if challenge_type == 2:
        return TYPE_2_H_MIN, TYPE_2_H_MAX
    if challenge_type == 3:
        return TYPE_3_H_MIN, TYPE_3_H_MAX
    if challenge_type == 4:
        return TYPE_3_H_MIN, TYPE_3_H_MAX
    if challenge_type == 5:
        return TYPE_4_H_MIN, TYPE_4_H_MAX
    if challenge_type == 6:
        return TYPE_6_H_MIN, TYPE_6_H_MAX
    raise ValueError(f"Unsupported challenge_type={challenge_type}")


def _within_bounds(x: float, y: float, challenge_type: int, seed: int) -> bool:
    (x_min, x_max), (y_min, y_max) = _xy_bounds_for_type(challenge_type, seed)
    return x_min <= x <= x_max and y_min <= y <= y_max


def build_custom_radius_task(
    *,
    seed: int,
    challenge_type: int,
    radius_min_m: float,
    radius_max_m: float,
    moving_platform: bool = False,
    goal_z_min_m: float | None = None,
    goal_z_max_m: float | None = None,
    goal_z_relative_to_start: bool = False,
) -> MapTask:
    """Create a deterministic easier task for a given map seed and type.

    We preserve the benchmark world/map seed, but overwrite the geometric start
    -> goal distance band so stage 02 can learn simple landing behavior before
    moving to the full validator distribution.
    """

    if radius_min_m <= 0.0 or radius_max_m <= 0.0 or radius_min_m > radius_max_m:
        raise ValueError("radius_min_m/radius_max_m must define a positive interval")

    base = build_task(seed=seed, challenge_type=challenge_type, moving_platform=moving_platform)
    rng = random.Random(seed + 424242 + int(radius_min_m * 1000) + int(radius_max_m * 1000) + challenge_type * 97)
    start_x, start_y, start_z = (float(v) for v in base.start)
    z_min, z_max = _z_bounds_for_type(challenge_type)
    if goal_z_min_m is not None and goal_z_max_m is not None:
        if goal_z_relative_to_start:
            goal_z = start_z + rng.uniform(goal_z_min_m, goal_z_max_m)
        else:
            goal_z = rng.uniform(goal_z_min_m, goal_z_max_m)
    else:
        goal_z = float(base.goal[2])
    goal_z = float(min(max(goal_z, z_min), z_max))

    goal_x, goal_y = float(base.goal[0]), float(base.goal[1])
    for _attempt in range(256):
        radius = rng.uniform(radius_min_m, radius_max_m)
        angle = rng.uniform(0.0, 2.0 * math.pi)
        cand_x = start_x + radius * math.cos(angle)
        cand_y = start_y + radius * math.sin(angle)
        if _within_bounds(cand_x, cand_y, challenge_type, seed):
            goal_x, goal_y = cand_x, cand_y
            break

    return MapTask(
        map_seed=base.map_seed,
        start=base.start,
        goal=(goal_x, goal_y, goal_z),
        sim_dt=base.sim_dt,
        horizon=base.horizon,
        challenge_type=base.challenge_type,
        search_radius=base.search_radius,
        moving_platform=base.moving_platform,
        version=base.version,
    )


def iter_curriculum_stage_tasks(stage: ExpertCurriculumStage) -> Iterable[tuple[str, str, dict[str, Any]]]:
    """Yield build-compatible task rows for a custom stage."""

    for episode_idx in range(stage.num_episodes):
        task = build_custom_radius_task(
            seed=stage.seed_start + episode_idx,
            challenge_type=stage.challenge_type,
            radius_min_m=stage.radius_min_m,
            radius_max_m=stage.radius_max_m,
            moving_platform=stage.moving_platform,
            goal_z_min_m=stage.goal_z_min_m,
            goal_z_max_m=stage.goal_z_max_m,
            goal_z_relative_to_start=stage.goal_z_relative_to_start,
        )
        yield (
            stage.name,
            "curriculum",
            {
                "map_seed": int(task.map_seed),
                "start": list(task.start),
                "goal": list(task.goal),
                "sim_dt": float(task.sim_dt),
                "horizon": float(task.horizon),
                "challenge_type": int(task.challenge_type),
                "search_radius": float(task.search_radius),
                "moving_platform": bool(task.moving_platform),
                "version": str(task.version),
            },
        )


def curriculum_stage_payload(stage: ExpertCurriculumStage) -> dict[str, Any]:
    return {
        "name": stage.name,
        "challenge_type": stage.challenge_type,
        "radius_min_m": stage.radius_min_m,
        "radius_max_m": stage.radius_max_m,
        "moving_platform": stage.moving_platform,
        "num_episodes": stage.num_episodes,
        "seed_start": stage.seed_start,
        "goal_z_min_m": stage.goal_z_min_m,
        "goal_z_max_m": stage.goal_z_max_m,
        "goal_z_relative_to_start": stage.goal_z_relative_to_start,
    }
