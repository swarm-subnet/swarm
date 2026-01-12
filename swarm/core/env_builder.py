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

import math
import random
from pathlib import Path
from typing import Optional, Tuple, List

import pybullet as p

from swarm.constants import (
    LANDING_PLATFORM_RADIUS,
    PLATFORM,
    SAFE_ZONE_RADIUS,
    MAX_ATTEMPTS_PER_OBS,
    START_PLATFORM,
    START_PLATFORM_RADIUS,
    START_PLATFORM_HEIGHT,
    START_PLATFORM_SURFACE_Z,
    START_PLATFORM_TAKEOFF_BUFFER,
    START_PLATFORM_RANDOMIZE,
    TYPE_1_N_OBSTACLES, TYPE_1_HEIGHT_SCALE, TYPE_1_SAFE_ZONE, TYPE_1_WORLD_RANGE,
    TYPE_2_N_OBSTACLES, TYPE_2_HEIGHT_SCALE, TYPE_2_SAFE_ZONE, TYPE_2_WORLD_RANGE,
    TYPE_3_N_OBSTACLES, TYPE_3_HEIGHT_SCALE, TYPE_3_SAFE_ZONE, TYPE_3_WORLD_RANGE,
    TYPE_4_N_OBSTACLES, TYPE_4_HEIGHT_SCALE, TYPE_4_SAFE_ZONE, TYPE_4_WORLD_RANGE,
    TYPE_5_N_OBSTACLES, TYPE_5_HEIGHT_SCALE, TYPE_5_SAFE_ZONE, TYPE_5_WORLD_RANGE,
    TYPE_5_RADIUS_MAX,
    GOAL_COLOR_PALETTE,
    DISTANT_SCENERY_ENABLED,
    DISTANT_SCENERY_MIN_RANGE,
    DISTANT_SCENERY_MAX_RANGE,
    DISTANT_SCENERY_COUNT,
)

# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _add_box(cli: int, pos, size, yaw) -> None:
    # Create collision and visual for colored box
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[s / 2 for s in size], physicsClientId=cli
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[s / 2 for s in size],
        rgbaColor=[0.2, 0.6, 0.8, 1.0],  # cyan-ish color for boxes
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

def _add_distant_scenery(cli: int, rng) -> None:
    """Add visual-only distant buildings and mountains outside playable world."""
    if not DISTANT_SCENERY_ENABLED:
        return
    
    scenery_colors = [
        [0.3, 0.3, 0.35, 1.0],
        [0.25, 0.28, 0.32, 1.0],
        [0.35, 0.32, 0.30, 1.0],
        [0.22, 0.25, 0.30, 1.0],
        [0.2, 0.5, 0.65, 1.0],
        [0.7, 0.6, 0.1, 1.0],
        [0.6, 0.2, 0.2, 1.0],
        [0.15, 0.45, 0.6, 1.0],
    ]
    
    sector_size = (2 * math.pi) / DISTANT_SCENERY_COUNT
    for i in range(DISTANT_SCENERY_COUNT):
        base_angle = i * sector_size
        angle = base_angle + rng.uniform(0.1, sector_size - 0.1)
        distance = rng.uniform(DISTANT_SCENERY_MIN_RANGE, DISTANT_SCENERY_MAX_RANGE)
        x = distance * math.cos(angle)
        y = distance * math.sin(angle)
        
        obj_type = rng.choice(["building", "building", "mountain"])
        color = rng.choice(scenery_colors)
        
        if obj_type == "building":
            width = rng.uniform(5, 12)
            depth = rng.uniform(5, 12)
            height = rng.uniform(20, 50)
            
            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[width / 2, depth / 2, height / 2],
                rgbaColor=color,
                physicsClientId=cli,
            )
            
            p.createMultiBody(
                0,
                -1,
                vis,
                basePosition=[x, y, height / 2],
                physicsClientId=cli,
            )
        else:
            base_radius = rng.uniform(6, 14)
            height = rng.uniform(25, 55)
            
            vis = p.createVisualShape(
                p.GEOM_CYLINDER,
                radius=base_radius,
                length=height,
                rgbaColor=color,
                physicsClientId=cli,
            )
            
            p.createMultiBody(
                0,
                -1,
                vis,
                basePosition=[x, y, height / 2],
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
) -> Tuple[List[int], List[int]]:
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

    if challenge_type == 1:
        n_obstacles = TYPE_1_N_OBSTACLES
        height_scale = TYPE_1_HEIGHT_SCALE
        safe_zone = TYPE_1_SAFE_ZONE
        world_range = TYPE_1_WORLD_RANGE
    elif challenge_type == 2:
        n_obstacles = TYPE_2_N_OBSTACLES
        height_scale = TYPE_2_HEIGHT_SCALE
        safe_zone = TYPE_2_SAFE_ZONE
        world_range = TYPE_2_WORLD_RANGE
    elif challenge_type == 3:
        n_obstacles = TYPE_3_N_OBSTACLES
        height_scale = TYPE_3_HEIGHT_SCALE
        safe_zone = TYPE_3_SAFE_ZONE
        world_range = TYPE_3_WORLD_RANGE
    elif challenge_type == 4:
        n_obstacles = TYPE_4_N_OBSTACLES
        height_scale = TYPE_4_HEIGHT_SCALE
        safe_zone = TYPE_4_SAFE_ZONE
        world_range = TYPE_4_WORLD_RANGE
    elif challenge_type == 5:
        n_obstacles = TYPE_5_N_OBSTACLES
        height_scale = TYPE_5_HEIGHT_SCALE
        safe_zone = TYPE_5_SAFE_ZONE
        world_range = TYPE_5_WORLD_RANGE
    else:
        n_obstacles = TYPE_1_N_OBSTACLES
        height_scale = TYPE_1_HEIGHT_SCALE
        safe_zone = TYPE_1_SAFE_ZONE
        world_range = TYPE_1_WORLD_RANGE

    if start is not None:
        sx, sy, sz = start
    else:
        sx = sy = sz = None
    gx, gy = (goal[0], goal[1]) if goal is not None else (None, None)

    # ------------------------------------------------------------------
    # Random obstacles with safe‑zone rejection
    # ------------------------------------------------------------------
    placed = 0
    placed_obstacles = []  # Track all placed obstacles: [(x, y, radius), ...]
    MIN_OBSTACLE_DISTANCE = 0.6  # Reduced minimum distance between obstacles
    
    while placed < n_obstacles:
        for _ in range(MAX_ATTEMPTS_PER_OBS):
            kind = rng.choice(["wall", "pillar", "box"])
            x = rng.uniform(-world_range, world_range)
            y = rng.uniform(-world_range, world_range)
            yaw = rng.uniform(0, math.pi)

            # — determine random size & bounding radius ---------------
            if kind == "box":
                sx_len, sy_len, sz_len = (rng.uniform(1, 4) for _ in range(3))
                sz_len *= height_scale
                # 2‑D footprint radius (half diagonal of rectangle)
                obj_r = math.hypot(sx_len / 2, sy_len / 2)

            elif kind == "wall":
                length = rng.uniform(5, 15)
                height = rng.uniform(2, 5) * height_scale
                sx_len, sy_len, sz_len = length, 0.3, height
                obj_r = length / 2.0

            else:  # pillar
                r = rng.uniform(0.3, 1.0)
                h = rng.uniform(2, 7) * height_scale
                sx_len = sy_len = r * 2
                sz_len = h
                obj_r = r

            # — safe‑zone checks (improved) ---------------------------
            def _violates_start(cx, cy):
                if cx is None:
                    return False
                required_clearance = obj_r + safe_zone + 0.5
                return math.hypot(x - cx, y - cy) < required_clearance

            def _violates_goal(cx, cy):
                if cx is None:
                    return False
                required_clearance = obj_r + safe_zone + 0.5
                if challenge_type == 5:
                    required_clearance += TYPE_5_RADIUS_MAX + LANDING_PLATFORM_RADIUS
                return math.hypot(x - cx, y - cy) < required_clearance

            if _violates_start(sx, sy) or _violates_goal(gx, gy):
                continue  # too close to start/goal – try another location

            # — obstacle overlap prevention (improved) ----------------
            # Check distance to all previously placed obstacles
            obstacle_collision = False
            for prev_x, prev_y, prev_r in placed_obstacles:
                distance = math.hypot(x - prev_x, y - prev_y)
                # Dynamic required distance based on obstacle sizes
                base_distance = obj_r + prev_r + MIN_OBSTACLE_DISTANCE
                # Add extra margin for large obstacles to prevent visual overlap
                if obj_r > 2.0 or prev_r > 2.0:  # Large obstacles
                    base_distance += 0.5  # Extra spacing for large obstacles
                
                if distance < base_distance:
                    obstacle_collision = True
                    break
            
            if obstacle_collision:
                continue  # too close to existing obstacle – try another location 
            # ----------------------------------------------------------
            # Passed all tests → create the obstacle
            # ----------------------------------------------------------
            if kind == "box":
                _add_box(cli, [x, y, sz_len / 2], [sx_len, sy_len, sz_len], yaw)

            elif kind == "wall":
                # Walls colored yellow
                col = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[sx_len / 2, sy_len / 2, sz_len / 2],
                    physicsClientId=cli,
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[sx_len / 2, sy_len / 2, sz_len / 2],
                    rgbaColor=[0.9, 0.8, 0.1, 1.0],  # yellow for walls
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

            else:  # pillar
                # Pillar collision and visual (red)
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
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],  # red pillars
                    physicsClientId=cli,
                )
                p.createMultiBody(
                    0,
                    col,
                    vis,
                    basePosition=[x, y, sz_len / 2],
                    physicsClientId=cli,
                )

            # ✅ CRITICAL FIX: Add the obstacle to placed_obstacles list to prevent overlapping
            placed_obstacles.append((x, y, obj_r))
            placed += 1
            break  # obstacle placed – move to next one
        else:
            # Unable to place this obstacle after many attempts
            # Try with reduced requirements for the remaining obstacles
            if placed < n_obstacles * 0.7:  # If we've placed less than 70% of obstacles
                # Reduce minimum distance temporarily for dense worlds
                MIN_OBSTACLE_DISTANCE = max(0.8, MIN_OBSTACLE_DISTANCE - 0.1)
            break

    # ------------------------------------------------------------------
    # World building report
    # ------------------------------------------------------------------
    if placed < n_obstacles:
        if placed < n_obstacles * 0.8:
            pass
    
    if challenge_type != 4:
        _add_distant_scenery(cli, rng)

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
            from swarm.validator.task_gen import get_platform_height_for_seed
            surface_z = get_platform_height_for_seed(seed, challenge_type)
        else:
            surface_z = START_PLATFORM_SURFACE_Z

        base_position = [sx, sy, surface_z - platform_height / 1.7]

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
            
            # Create the physical landing platform - POSITIONED CORRECTLY
            platform_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=platform_collision,
                baseVisualShapeIndex=platform_visual,
                basePosition=[gx, gy, gz - platform_height / 2],
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
                basePosition=[gx, gy, gz + surface_height / 2 + 0.001],
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
                basePosition=[gx, gy, gz + surface_height + 0.002],
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
                basePosition=[gx, gy, gz + surface_height + badge_height + 0.008],
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
                basePosition=[gx, gy, gz + surface_height + badge_height + 0.011],
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
                basePosition=[gx, gy, gz + pole_h / 2 + 0.008],
                physicsClientId=cli,
            )
            end_platform_uids.append(pole_uid)
            
            cap_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=cap_visual,
                basePosition=[gx, gy, gz + pole_h + 0.015],
                physicsClientId=cli,
            )
            end_platform_uids.append(cap_uid)
            
            return (end_platform_uids, start_platform_uids)
        
        else:
            # Visual-only markers (legacy mode for easier challenges)
            # 1) outer halo ------------------------------------------------
            halo_thick = 0.02
            halo = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.45,
                length=halo_thick,
                rgbaColor=[0.15, 0.8, 0.15, 1.0],
                specularColor=[0.3, 0.3, 0.3],
                physicsClientId=cli,
            )
            p.createMultiBody(
                0, -1, halo, [gx, gy, gz - halo_thick / 2], physicsClientId=cli
            )

            # 2) TAO badge -------------------------------------------------
            badge_size = 0.50
            half = badge_size / 2
            badge_offset = 0.001

            vertices = [
                [-half, -half, 0.0],
                [ half, -half, 0.0],
                [ half,  half, 0.0],
                [-half,  half, 0.0],
            ]
            indices = [0, 1, 2, 0, 2, 3]
            uvs = [[0, 0], [1, 0], [1, 1], [0, 1]]

            vis = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                vertices=vertices,
                indices=indices,
                uvs=uvs,
                physicsClientId=cli,
            )

            uid = p.createMultiBody(
                0,
                -1,
                vis,
                [gx, gy, gz + badge_offset],
                [0, 0, 0, 1],
                physicsClientId=cli,
            )
            p.changeVisualShape(
                uid,
                -1,
                textureUniqueId=_get_tao_tex(cli),
                flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
                physicsClientId=cli,
            )

            # 3) red pole --------------------------------------------------
            pole_h = 0.30
            pole_vis = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.012,
                length=pole_h,
                rgbaColor=[0.9, 0.1, 0.1, 1.0],
                specularColor=[0.4, 0.4, 0.4],
                physicsClientId=cli,
            )
            pole_uid_legacy = p.createMultiBody(
                0,
                -1,
                pole_vis,
                [gx, gy, gz + pole_h / 2 + 0.001],
                physicsClientId=cli,
            )
            end_platform_uids.append(pole_uid_legacy)
            
    return (end_platform_uids, start_platform_uids)
