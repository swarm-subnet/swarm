"""Builders for the Type 2 open-world benchmark map."""

from __future__ import annotations

import math
import random
from typing import Optional, Tuple

import pybullet as p

from swarm.constants import (
    MAX_ATTEMPTS_PER_OBS,
    TYPE_2_HEIGHT_SCALE,
    TYPE_2_N_OBSTACLES,
    TYPE_2_SAFE_ZONE,
    TYPE_2_WORLD_RANGE,
)


def _add_box(cli: int, pos, size, yaw) -> None:
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[s / 2 for s in size], physicsClientId=cli
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[s / 2 for s in size],
        rgbaColor=[0.2, 0.6, 0.8, 1.0],
        physicsClientId=cli,
    )
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    p.createMultiBody(
        0,
        col,
        vis,
        basePosition=pos,
        baseOrientation=quat,
        physicsClientId=cli,
    )


def build_open_world(
    cli: int,
    seed: int,
    start: Optional[Tuple[float, float, float]] = None,
    goal: Optional[Tuple[float, float, float]] = None,
) -> None:
    rng = random.Random(seed)
    sx = sy = gx = gy = None
    if start is not None:
        sx, sy, _ = start
    if goal is not None:
        gx, gy, _ = goal

    placed = 0
    placed_obstacles: list[tuple[float, float, float]] = []
    min_obstacle_distance = 0.6

    while placed < TYPE_2_N_OBSTACLES:
        for _ in range(MAX_ATTEMPTS_PER_OBS):
            kind = rng.choice(["wall", "pillar", "box"])
            x = rng.uniform(-TYPE_2_WORLD_RANGE, TYPE_2_WORLD_RANGE)
            y = rng.uniform(-TYPE_2_WORLD_RANGE, TYPE_2_WORLD_RANGE)
            yaw = rng.uniform(0, math.pi)

            if kind == "box":
                sx_len, sy_len, sz_len = (rng.uniform(1, 4) for _ in range(3))
                sz_len *= TYPE_2_HEIGHT_SCALE
                obj_r = math.hypot(sx_len / 2, sy_len / 2)
            elif kind == "wall":
                length = rng.uniform(5, 15)
                height = rng.uniform(2, 5) * TYPE_2_HEIGHT_SCALE
                sx_len, sy_len, sz_len = length, 0.3, height
                obj_r = length / 2.0
            else:
                r = rng.uniform(0.3, 1.0)
                h = rng.uniform(2, 7) * TYPE_2_HEIGHT_SCALE
                sx_len = sy_len = r * 2
                sz_len = h
                obj_r = r

            def _violates_zone(cx, cy):
                if cx is None:
                    return False
                required_clearance = obj_r + TYPE_2_SAFE_ZONE + 0.5
                return math.hypot(x - cx, y - cy) < required_clearance

            if _violates_zone(sx, sy) or _violates_zone(gx, gy):
                continue

            obstacle_collision = False
            for prev_x, prev_y, prev_r in placed_obstacles:
                distance = math.hypot(x - prev_x, y - prev_y)
                base_distance = obj_r + prev_r + min_obstacle_distance
                if obj_r > 2.0 or prev_r > 2.0:
                    base_distance += 0.5
                if distance < base_distance:
                    obstacle_collision = True
                    break
            if obstacle_collision:
                continue

            if kind == "box":
                _add_box(cli, [x, y, sz_len / 2], [sx_len, sy_len, sz_len], yaw)
            elif kind == "wall":
                col = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[sx_len / 2, sy_len / 2, sz_len / 2],
                    physicsClientId=cli,
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[sx_len / 2, sy_len / 2, sz_len / 2],
                    rgbaColor=[0.9, 0.8, 0.1, 1.0],
                    physicsClientId=cli,
                )
                quat = p.getQuaternionFromEuler([0, 0, yaw])
                p.createMultiBody(
                    0,
                    col,
                    vis,
                    basePosition=[x, y, sz_len / 2],
                    baseOrientation=quat,
                    physicsClientId=cli,
                )
            else:
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=obj_r,
                    height=sz_len,
                    physicsClientId=cli,
                )
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=obj_r,
                    length=sz_len,
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                    physicsClientId=cli,
                )
                p.createMultiBody(
                    0,
                    col,
                    vis,
                    basePosition=[x, y, sz_len / 2],
                    physicsClientId=cli,
                )

            placed_obstacles.append((x, y, obj_r))
            placed += 1
            break
        else:
            if placed < TYPE_2_N_OBSTACLES * 0.7:
                min_obstacle_distance = max(0.8, min_obstacle_distance - 0.1)
            break

