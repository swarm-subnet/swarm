# swarm/core/env_builder.py
"""
Procedurally build the random world and (optionally) add a *visual‑only*
marker that shows the goal position.

Key changes
-----------
• Introduced SAFE_ZONE_RADIUS (2 m) around both the spawn and the goal.
• Obstacles are now rejected if *any part* of them could intrude into a
  safe zone, considering their own footprint/half‑extent.
• `build_world()` now accepts the drone's *start* position in addition
  to the goal.

The marker itself has **no collision shape** (baseCollisionShapeIndex = ‑1);
it is only visual.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Optional, Tuple, List

import pybullet as p

from swarm.validator.task_gen import get_platform_height_for_seed
from swarm.constants import (
    BENCHMARK_VERSION,
    MAP_CACHE_ENABLED,
    MAP_CACHE_SAVE_ON_BUILD,
    LANDING_PLATFORM_RADIUS,
    PLATFORM,
    MAX_ATTEMPTS_PER_OBS,
    START_PLATFORM,
    START_PLATFORM_RADIUS,
    START_PLATFORM_HEIGHT,
    START_PLATFORM_SURFACE_Z,
    START_PLATFORM_TAKEOFF_BUFFER,
    START_PLATFORM_RANDOMIZE,
    TYPE_1_SAFE_ZONE, TYPE_1_WORLD_RANGE,
    TYPE_2_N_OBSTACLES, TYPE_2_HEIGHT_SCALE, TYPE_2_SAFE_ZONE, TYPE_2_WORLD_RANGE,
    TYPE_3_SAFE_ZONE,
    GOAL_COLOR_PALETTE,
)
from swarm.core.city_generator import build_city as build_city_map
from swarm.core.mountain_generator import build_mountains, get_mountain_subtype


STATE_DIR = Path(__file__).parent.parent.parent / "state"
MAP_CACHE_DIR = STATE_DIR / "map_cache"


# --------------------------------------------------------------------------
# Raycast helper — find real collision surface height
# --------------------------------------------------------------------------
def _raycast_surface_z(cli: int, x: float, y: float) -> float:
    result = p.rayTest(
        rayFromPosition=[x, y, 500.0],
        rayToPosition=[x, y, -100.0],
        physicsClientId=cli,
    )
    if result and result[0][0] != -1:
        return result[0][3][2]
    return 0.0


# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Texture loader (cache per client)
# --------------------------------------------------------------------------
_TAO_TEX_ID: dict[int, int] = {}


def _get_tao_tex(cli: int) -> int:
    """Load swarm/assets/tao.png exactly once per PyBullet client."""
    if cli not in _TAO_TEX_ID:
        tex_path = Path(__file__).parent.parent / "assets" / "tao.png"
        _TAO_TEX_ID[cli] = p.loadTexture(str(tex_path))
    return _TAO_TEX_ID[cli]


def _normalize_xy(point: Optional[Tuple[float, float, float]]) -> Optional[Tuple[float, float]]:
    if point is None:
        return None
    return (round(float(point[0]), 6), round(float(point[1]), 6))


def _static_world_cache_file(
    seed: int,
    challenge_type: int,
    start: Optional[Tuple[float, float, float]],
    goal: Optional[Tuple[float, float, float]],
) -> Path:
    payload = {
        "benchmark_version": BENCHMARK_VERSION,
        "seed": int(seed),
        "challenge_type": int(challenge_type),
        "start_xy": _normalize_xy(start),
        "goal_xy": _normalize_xy(goal),
    }
    key_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    key_hash = hashlib.sha256(key_json.encode("utf-8")).hexdigest()
    return MAP_CACHE_DIR / BENCHMARK_VERSION / f"type{challenge_type}" / f"{key_hash}.bullet"


def _static_world_cache_meta_file(
    seed: int,
    challenge_type: int,
    start: Optional[Tuple[float, float, float]],
    goal: Optional[Tuple[float, float, float]],
) -> Path:
    return _static_world_cache_file(seed, challenge_type, start, goal).with_suffix(".json")


def _build_static_world_cache_meta(
    cli: int,
    *,
    start: Optional[Tuple[float, float, float]],
    goal: Optional[Tuple[float, float, float]],
    challenge_type: int,
    base_body_count: int = 0,
) -> dict:
    total_bodies = int(p.getNumBodies(physicsClientId=cli))
    map_body_count = max(0, total_bodies - int(base_body_count))
    meta = {
        "body_count": total_bodies,
        "map_body_count": map_body_count,
    }

    if challenge_type == 3:
        if start is not None:
            meta["start_surface_z"] = float(
                _raycast_surface_z(cli, float(start[0]), float(start[1]))
            )
        if goal is not None:
            meta["goal_surface_z"] = float(
                _raycast_surface_z(cli, float(goal[0]), float(goal[1]))
            )

    return meta


def _write_static_world_cache_meta(meta_file: Path, meta: dict) -> None:
    tmp_meta_file = meta_file.with_suffix(".json.tmp")
    tmp_meta_file.unlink(missing_ok=True)
    tmp_meta_file.write_text(
        json.dumps(meta, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    tmp_meta_file.replace(meta_file)


def _read_static_world_cache_meta(meta_file: Path) -> Optional[dict]:
    if not meta_file.exists():
        return None
    try:
        return json.loads(meta_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def _invalidate_static_world_cache(cache_file: Path, meta_file: Path) -> None:
    cache_file.unlink(missing_ok=True)
    meta_file.unlink(missing_ok=True)


def get_static_world_cache_path(
    seed: int,
    challenge_type: int,
    start: Optional[Tuple[float, float, float]],
    goal: Optional[Tuple[float, float, float]],
) -> Path:
    return _static_world_cache_file(seed, challenge_type, start, goal)


def _build_static_world(
    seed: int,
    cli: int,
    *,
    start: Optional[Tuple[float, float, float]],
    goal: Optional[Tuple[float, float, float]],
    challenge_type: int,
) -> None:
    rng = random.Random(seed)

    if challenge_type == 1:
        n_obstacles = 0
        safe_zone = TYPE_1_SAFE_ZONE
        world_range = TYPE_1_WORLD_RANGE
    elif challenge_type == 3:
        n_obstacles = 0
        safe_zone = TYPE_3_SAFE_ZONE
        world_range = 0
    else:
        n_obstacles = TYPE_2_N_OBSTACLES
        height_scale = TYPE_2_HEIGHT_SCALE
        safe_zone = TYPE_2_SAFE_ZONE
        world_range = TYPE_2_WORLD_RANGE

    if start is not None:
        sx, sy, _ = start
    else:
        sx = sy = None
    if goal is not None:
        gx, gy, _ = goal
    else:
        gx = gy = None

    placed = 0
    placed_obstacles = []
    min_obstacle_distance = 0.6

    while placed < n_obstacles:
        for _ in range(MAX_ATTEMPTS_PER_OBS):
            kind = rng.choice(["wall", "pillar", "box"])
            x = rng.uniform(-world_range, world_range)
            y = rng.uniform(-world_range, world_range)
            yaw = rng.uniform(0, math.pi)

            if kind == "box":
                sx_len, sy_len, sz_len = (rng.uniform(1, 4) for _ in range(3))
                sz_len *= height_scale
                obj_r = math.hypot(sx_len / 2, sy_len / 2)

            elif kind == "wall":
                length = rng.uniform(5, 15)
                height = rng.uniform(2, 5) * height_scale
                sx_len, sy_len, sz_len = length, 0.3, height
                obj_r = length / 2.0

            else:
                r = rng.uniform(0.3, 1.0)
                h = rng.uniform(2, 7) * height_scale
                sx_len = sy_len = r * 2
                sz_len = h
                obj_r = r

            def _violates_start(cx, cy):
                if cx is None:
                    return False
                required_clearance = obj_r + safe_zone + 0.5
                return math.hypot(x - cx, y - cy) < required_clearance

            def _violates_goal(cx, cy):
                if cx is None:
                    return False
                required_clearance = obj_r + safe_zone + 0.5
                return math.hypot(x - cx, y - cy) < required_clearance

            if _violates_start(sx, sy) or _violates_goal(gx, gy):
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
            if placed < n_obstacles * 0.7:
                min_obstacle_distance = max(0.8, min_obstacle_distance - 0.1)
            break

    if challenge_type == 1:
        safe_zones = []
        if sx is not None and sy is not None:
            safe_zones.append((sx, sy))
        if gx is not None and gy is not None:
            safe_zones.append((gx, gy))
        build_city_map(cli, seed, safe_zones, safe_zone)

    elif challenge_type == 3:
        safe_zones = []
        if sx is not None and sy is not None:
            safe_zones.append((sx, sy))
        if gx is not None and gy is not None:
            safe_zones.append((gx, gy))
        build_mountains(cli, seed, safe_zones, safe_zone)


def prebuild_static_world_cache(
    seed: int,
    challenge_type: int,
    *,
    start: Optional[Tuple[float, float, float]],
    goal: Optional[Tuple[float, float, float]],
) -> Path:
    cache_file = _static_world_cache_file(seed, challenge_type, start, goal)
    if challenge_type == 2:
        return cache_file
    meta_file = _static_world_cache_meta_file(seed, challenge_type, start, goal)
    if not MAP_CACHE_ENABLED:
        return cache_file
    if cache_file.exists() and meta_file.exists():
        return cache_file
    if cache_file.exists() or meta_file.exists():
        _invalidate_static_world_cache(cache_file, meta_file)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(".tmp")
    tmp_file.unlink(missing_ok=True)

    cli = p.connect(p.DIRECT)
    try:
        base_body_count = p.getNumBodies(physicsClientId=cli)
        _build_static_world(
            seed=seed,
            cli=cli,
            start=start,
            goal=goal,
            challenge_type=challenge_type,
        )
        p.saveBullet(str(tmp_file), physicsClientId=cli)
        meta = _build_static_world_cache_meta(
            cli,
            start=start,
            goal=goal,
            challenge_type=challenge_type,
            base_body_count=base_body_count,
        )
    except Exception:
        tmp_file.unlink(missing_ok=True)
        _invalidate_static_world_cache(cache_file, meta_file)
        raise
    finally:
        p.disconnect(cli)

    tmp_file.replace(cache_file)
    try:
        _write_static_world_cache_meta(meta_file, meta)
    except Exception:
        _invalidate_static_world_cache(cache_file, meta_file)
    return cache_file


def _save_static_world_cache_from_client(
    seed: int,
    cli: int,
    *,
    start: Optional[Tuple[float, float, float]],
    goal: Optional[Tuple[float, float, float]],
    challenge_type: int,
    base_body_count: int = 0,
) -> None:
    if challenge_type == 2:
        return

    cache_file = _static_world_cache_file(seed, challenge_type, start, goal)
    meta_file = _static_world_cache_meta_file(seed, challenge_type, start, goal)
    if cache_file.exists() and meta_file.exists():
        return
    if cache_file.exists() or meta_file.exists():
        _invalidate_static_world_cache(cache_file, meta_file)

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_file = cache_file.with_suffix(".tmp")
    tmp_file.unlink(missing_ok=True)

    try:
        p.saveBullet(str(tmp_file), physicsClientId=cli)
        tmp_file.replace(cache_file)
        meta = _build_static_world_cache_meta(
            cli,
            start=start,
            goal=goal,
            challenge_type=challenge_type,
            base_body_count=base_body_count,
        )
        _write_static_world_cache_meta(meta_file, meta)
    except Exception:
        tmp_file.unlink(missing_ok=True)
        _invalidate_static_world_cache(cache_file, meta_file)


def _try_load_static_world_cache(
    seed: int,
    cli: int,
    *,
    start: Optional[Tuple[float, float, float]],
    goal: Optional[Tuple[float, float, float]],
    challenge_type: int,
) -> bool:
    if challenge_type == 2:
        return False

    cache_file = _static_world_cache_file(seed, challenge_type, start, goal)
    meta_file = _static_world_cache_meta_file(seed, challenge_type, start, goal)
    if not cache_file.exists() or not meta_file.exists():
        _invalidate_static_world_cache(cache_file, meta_file)
        return False

    expected_meta = _read_static_world_cache_meta(meta_file)
    if not isinstance(expected_meta, dict):
        _invalidate_static_world_cache(cache_file, meta_file)
        return False

    before_bodies = p.getNumBodies(physicsClientId=cli)

    try:
        p.loadBullet(str(cache_file), physicsClientId=cli)
        after_bodies = p.getNumBodies(physicsClientId=cli)

        actual_meta = _build_static_world_cache_meta(
            cli,
            start=start,
            goal=goal,
            challenge_type=challenge_type,
            base_body_count=before_bodies,
        )

        loaded_map_bodies = max(0, int(after_bodies - before_bodies))
        expected_map_bodies = int(
            expected_meta.get(
                "map_body_count",
                expected_meta.get("body_count", -2),
            )
        )

        if challenge_type in (1, 3) and loaded_map_bodies <= 0:
            _invalidate_static_world_cache(cache_file, meta_file)
            return False

        if int(actual_meta.get("map_body_count", -1)) != expected_map_bodies:
            _invalidate_static_world_cache(cache_file, meta_file)
            return False

        if challenge_type == 3:
            tolerance = 1.0

            expected_start_surface = expected_meta.get("start_surface_z")
            actual_start_surface = actual_meta.get("start_surface_z")
            if expected_start_surface is None or actual_start_surface is None:
                _invalidate_static_world_cache(cache_file, meta_file)
                return False
            if abs(float(actual_start_surface) - float(expected_start_surface)) > tolerance:
                _invalidate_static_world_cache(cache_file, meta_file)
                return False

            expected_goal_surface = expected_meta.get("goal_surface_z")
            actual_goal_surface = actual_meta.get("goal_surface_z")
            if expected_goal_surface is None or actual_goal_surface is None:
                _invalidate_static_world_cache(cache_file, meta_file)
                return False
            if abs(float(actual_goal_surface) - float(expected_goal_surface)) > tolerance:
                _invalidate_static_world_cache(cache_file, meta_file)
                return False

        return True
    except Exception:
        _invalidate_static_world_cache(cache_file, meta_file)
        return False


# --------------------------------------------------------------------------
# Main world builder
# --------------------------------------------------------------------------
def build_world(
    seed: int,
    cli: int,
    *,
    start: Optional[Tuple[float, float, float]] = None,
    goal: Optional[Tuple[float, float, float]] = None,
    challenge_type: int = 1,
) -> Tuple[List[int], List[int], Optional[float], Optional[float]]:
    """
    Create procedural obstacles (with safe‑zone constraints) and—if *goal*
    is provided—place a visual TAO badge at that position.

    Parameters
    ----------
    seed   : int      • PRNG seed so miners and validator share the same map
    cli    : int      • PyBullet client id
    start  : (x,y,z)  • drone take‑off location (obstacles keep clear)
    goal   : (x,y,z)  • desired target (obstacles keep clear; visual marker)
    """
    rng = random.Random(seed)

    if start is not None:
        sx, sy, sz = start
    else:
        sx = sy = sz = None

    if goal is not None:
        gx, gy, gz = goal
    else:
        gx = gy = gz = None

    cache_loaded = False
    static_world_body_base = p.getNumBodies(physicsClientId=cli)
    if MAP_CACHE_ENABLED:
        cache_loaded = _try_load_static_world_cache(
            seed=seed,
            cli=cli,
            start=start,
            goal=goal,
            challenge_type=challenge_type,
        )

    if not cache_loaded:
        _build_static_world(
            seed=seed,
            cli=cli,
            start=start,
            goal=goal,
            challenge_type=challenge_type,
        )
        if MAP_CACHE_ENABLED and MAP_CACHE_SAVE_ON_BUILD:
            _save_static_world_cache_from_client(
                seed=seed,
                cli=cli,
                start=start,
                goal=goal,
                challenge_type=challenge_type,
                base_body_count=static_world_body_base,
            )

    start_platform_surface_z = None
    goal_platform_surface_z = None

    start_platform_uids: List[int] = []
    end_platform_uids: List[int] = []

    # ------------------------------------------------------------------
    # Optional solid start platform
    # ------------------------------------------------------------------
    if START_PLATFORM and sx is not None and sy is not None and sz is not None:
        platform_radius = START_PLATFORM_RADIUS
        platform_height = START_PLATFORM_HEIGHT

        # Calculate platform surface height (random or fixed)
        if START_PLATFORM_RANDOMIZE:
            if challenge_type == 3:
                inferred_surface = sz - START_PLATFORM_TAKEOFF_BUFFER
                terrain_surface = _raycast_surface_z(cli, sx, sy)
                if abs(float(inferred_surface) - float(terrain_surface)) <= 0.5:
                    surface_z = float(terrain_surface)
                else:
                    surface_z = float(inferred_surface)
            else:
                surface_z = get_platform_height_for_seed(seed, challenge_type)
        else:
            surface_z = START_PLATFORM_SURFACE_Z

        start_platform_surface_z = surface_z

        base_position = [sx, sy, surface_z - platform_height / 2 + 0.05]

        start_platform_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=platform_radius,
            height=platform_height,
            physicsClientId=cli,
        )

        start_platform_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=platform_radius,
            length=platform_height,
            rgbaColor=[1.0, 0.0, 0.0, 1.0],
            specularColor=[1.0, 0.3, 0.3],
            physicsClientId=cli,
        )

        start_platform_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=start_platform_collision,
            baseVisualShapeIndex=start_platform_visual,
            basePosition=base_position,
            physicsClientId=cli,
        )
        start_platform_uids.append(start_platform_uid)

        p.changeDynamics(
            bodyUniqueId=start_platform_uid,
            linkIndex=-1,
            restitution=0.0,
            lateralFriction=2.5,
            spinningFriction=1.2,
            rollingFriction=0.6,
            physicsClientId=cli,
        )

        flat_surface_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=platform_radius * 0.9,
            height=0.001,
            physicsClientId=cli,
        )

        flat_surface_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=flat_surface_collision,
            baseVisualShapeIndex=-1,
            basePosition=[sx, sy, surface_z],
            physicsClientId=cli,
        )
        start_platform_uids.append(flat_surface_uid)

        p.changeDynamics(
            bodyUniqueId=flat_surface_uid,
            linkIndex=-1,
            restitution=0.0,
            lateralFriction=3.0,
            spinningFriction=2.0,
            rollingFriction=1.0,
            physicsClientId=cli,
        )

        start_surface_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=platform_radius * 0.9,
            length=0.002,
            rgbaColor=[1.0, 0.0, 0.0, 1.0],
            specularColor=[1.0, 0.3, 0.3],
            physicsClientId=cli,
        )

        start_visual_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=start_surface_visual,
            basePosition=[sx, sy, surface_z + 0.001],
            physicsClientId=cli,
        )
        start_platform_uids.append(start_visual_uid)

    # ------------------------------------------------------------------
    # Physical landing platform with visual goal marker
    # ------------------------------------------------------------------
    if goal is not None:
        gx, gy, gz = goal

        if challenge_type == 3:
            subtype = get_mountain_subtype(seed)
            if subtype == 2:
                surface_z = 0.0
            else:
                terrain_surface = _raycast_surface_z(cli, gx, gy)
                if abs(float(terrain_surface) - float(gz)) <= 0.5:
                    surface_z = float(terrain_surface)
                else:
                    surface_z = float(gz)
        else:
            surface_z = gz

        goal_platform_surface_z = surface_z

        # Platform mode: solid if PLATFORM else visual-only
        if PLATFORM:
            goal_color = rng.choice(GOAL_COLOR_PALETTE)

            # 1) Physical landing platform - SOLID AND PRECISE -----------
            platform_radius = LANDING_PLATFORM_RADIUS  # Consistent radius
            platform_height = 0.2         # Thicker for better physics stability

            # Create FLAT CIRCULAR platform - very short cylinder (like a coin)
            platform_collision = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=platform_radius,
                height=platform_height,
                physicsClientId=cli,
            )

            # Create visual shape for the platform
            platform_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=platform_radius,
                length=platform_height,
                rgbaColor=goal_color,
                specularColor=[goal_color[0] * 0.6 + 0.4, goal_color[1] * 0.6 + 0.4, goal_color[2] * 0.6 + 0.4],
                physicsClientId=cli,
            )

            goal_platform_z = surface_z - platform_height / 2 + 0.05
            platform_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=platform_collision,
                baseVisualShapeIndex=platform_visual,
                basePosition=[gx, gy, goal_platform_z],
                physicsClientId=cli
            )
            end_platform_uids.append(platform_uid)

            p.changeDynamics(
                bodyUniqueId=platform_uid,
                linkIndex=-1,
                restitution=0.0,
                lateralFriction=2.0,
                spinningFriction=1.0,
                rollingFriction=0.5,
                physicsClientId=cli
            )

            # 3)landing zone ---------------
            # Create multiple layers for depth and glow effect
            surface_radius = platform_radius * 0.8  # Slightly smaller than platform
            surface_height = 0.008                  # Slightly thicker for better visibility

            bright_goal_color = [min(1.0, goal_color[0] * 1.25), min(1.0, goal_color[1] * 1.25), min(1.0, goal_color[2] * 1.25), 1.0]

            # Main landing surface with glow effect
            surface_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=surface_radius,
                length=surface_height,
                rgbaColor=bright_goal_color,
                specularColor=[bright_goal_color[0] * 0.8, bright_goal_color[1] * 0.8, bright_goal_color[2] * 0.8],
                physicsClientId=cli,
            )

            # Position main green surface on top of platform
            surface_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=surface_visual,
                basePosition=[gx, gy, surface_z + surface_height / 2 + 0.001],
                physicsClientId=cli,
            )
            end_platform_uids.append(surface_uid)

            flat_landing_collision = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=surface_radius,
                height=0.001,
                physicsClientId=cli,
            )

            flat_landing_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=flat_landing_collision,
                baseVisualShapeIndex=-1,
                basePosition=[gx, gy, surface_z + surface_height + 0.002],
                physicsClientId=cli
            )
            end_platform_uids.append(flat_landing_uid)

            p.changeDynamics(
                bodyUniqueId=flat_landing_uid,
                linkIndex=-1,
                restitution=0.0,
                lateralFriction=3.0,
                spinningFriction=2.0,
                rollingFriction=1.0,
                physicsClientId=cli
            )

            # TAO logo as MASSIVE CIRCULAR badge covering the ENTIRE surface
            # Make it BIG and OBVIOUS - covering all the area
            tao_logo_radius = surface_radius * 1.06  # Cover all of circle
            badge_height = 0.005       # Thicker for visibility

            # Create LARGE circular background first
            tao_background_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=tao_logo_radius,
                length=badge_height,
                rgbaColor=bright_goal_color,
                physicsClientId=cli,
            )

            # Position the white background
            tao_background_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=tao_background_visual,
                basePosition=[gx, gy, surface_z + surface_height + badge_height + 0.008],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=cli,
            )
            end_platform_uids.append(tao_background_uid)

            tao_logo_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=tao_logo_radius * 0.95,
                length=badge_height * 0.5,
                rgbaColor=bright_goal_color,
                physicsClientId=cli,
            )

            tao_logo_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=tao_logo_visual,
                basePosition=[gx, gy, surface_z + surface_height + badge_height + 0.011],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=cli,
            )
            end_platform_uids.append(tao_logo_uid)

            p.changeVisualShape(
                tao_logo_uid,
                -1,
                textureUniqueId=_get_tao_tex(cli),
                flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
                physicsClientId=cli,
            )

            # 4) glowing guidance beacon ----------------------
            pole_h = 0.5              # Taller, more elegant
            pole_radius = 0.012        # Sleeker profile

            # Main beacon pole with gradient effect
            pole_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=pole_radius,
                length=pole_h,
                rgbaColor=[1.0, 0.2, 0.1, 0.9],  # Bright glowing red-orange
                specularColor=[1.0, 0.8, 0.2],   # Golden specular highlight
                physicsClientId=cli,
            )

            # Add beacon top cap for elegant finish
            cap_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=pole_radius * 2,
                rgbaColor=[1.0, 0.3, 0.0, 1.0],  # Bright orange cap
                specularColor=[1.0, 1.0, 0.4],   # Bright golden specular
                physicsClientId=cli,
            )

            # Position main beacon pole
            pole_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=pole_visual,
                basePosition=[gx, gy, surface_z + pole_h / 2 + 0.008],
                physicsClientId=cli,
            )
            end_platform_uids.append(pole_uid)

            cap_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=cap_visual,
                basePosition=[gx, gy, surface_z + pole_h + 0.015],
                physicsClientId=cli,
            )
            end_platform_uids.append(cap_uid)

            return (end_platform_uids, start_platform_uids, start_platform_surface_z, goal_platform_surface_z)

    return (end_platform_uids, start_platform_uids, start_platform_surface_z, goal_platform_surface_z)
