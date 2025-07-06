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
from typing import Optional, Tuple

import pybullet as p

from swarm.constants import WORLD_RANGE, HEIGHT_SCALE, N_OBSTACLES, LANDING_PLATFORM_RADIUS

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
SAFE_ZONE_RADIUS = 2.0         # keep at least 2 m of clearance
MAX_ATTEMPTS_PER_OBS = 100     # retry limit when placing each obstacle

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
) -> None:
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

    sx, sy = (start[0], start[1]) if start is not None else (None, None)
    gx, gy = (goal[0], goal[1]) if goal is not None else (None, None)

    # ------------------------------------------------------------------
    # Random obstacles with safe‑zone rejection
    # ------------------------------------------------------------------
    placed = 0
    placed_obstacles = []  # Track all placed obstacles: [(x, y, radius), ...]
    MIN_OBSTACLE_DISTANCE = 1.2  # Increased minimum distance between obstacles for better separation
    
    while placed < N_OBSTACLES:
        for _ in range(MAX_ATTEMPTS_PER_OBS):
            kind = rng.choice(["wall", "pillar", "box"])
            x = rng.uniform(-WORLD_RANGE, WORLD_RANGE)
            y = rng.uniform(-WORLD_RANGE, WORLD_RANGE)
            yaw = rng.uniform(0, math.pi)

            # — determine random size & bounding radius ---------------
            if kind == "box":
                sx_len, sy_len, sz_len = (rng.uniform(1, 4) for _ in range(3))
                sz_len *= HEIGHT_SCALE
                # 2‑D footprint radius (half diagonal of rectangle)
                obj_r = math.hypot(sx_len / 2, sy_len / 2)

            elif kind == "wall":
                length = rng.uniform(5, 15)
                height = rng.uniform(2, 5) * HEIGHT_SCALE
                sx_len, sy_len, sz_len = length, 0.3, height
                obj_r = length / 2.0

            else:  # pillar
                r = rng.uniform(0.3, 1.0)
                h = rng.uniform(2, 7) * HEIGHT_SCALE
                sx_len = sy_len = r * 2
                sz_len = h
                obj_r = r

            # — safe‑zone checks (improved) ---------------------------
            def _violates(cx, cy):
                if cx is None:
                    return False
                # More conservative safe zone calculation
                required_clearance = obj_r + SAFE_ZONE_RADIUS + 0.5  # Extra 0.5m margin
                return math.hypot(x - cx, y - cy) < required_clearance

            if _violates(sx, sy) or _violates(gx, gy):
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
            if placed < N_OBSTACLES * 0.7:  # If we've placed less than 70% of obstacles
                # Reduce minimum distance temporarily for dense worlds
                MIN_OBSTACLE_DISTANCE = max(0.8, MIN_OBSTACLE_DISTANCE - 0.1)
                print(f"⚠️ Reducing obstacle spacing to {MIN_OBSTACLE_DISTANCE:.1f}m to fit more obstacles")
            break

    # ------------------------------------------------------------------
    # World building report
    # ------------------------------------------------------------------
    if placed < N_OBSTACLES:
        print(f"🌍 World Building: Successfully placed {placed}/{N_OBSTACLES} obstacles")
        if placed < N_OBSTACLES * 0.8:
            print(f"⚠️ World may be sparse due to space constraints or overlap prevention")
    else:
        print(f"✅ World Building: Successfully placed all {N_OBSTACLES} obstacles with proper spacing")

    # ------------------------------------------------------------------
    # Physical landing platform with visual goal marker
    # ------------------------------------------------------------------
    if goal is not None:
        gx, gy, gz = goal

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
            rgbaColor=[0.15, 0.35, 0.8, 1.0],  # blue platform
            specularColor=[0.8, 0.8, 0.9],     # High reflectivity for metallic look
            physicsClientId=cli,
        )
        
        # Create the physical landing platform - POSITIONED CORRECTLY
        platform_uid = p.createMultiBody(
            baseMass=0,  # Static platform (infinite mass)
            baseCollisionShapeIndex=platform_collision,
            baseVisualShapeIndex=platform_visual,
            basePosition=[gx, gy, gz - platform_height / 2],  # Bottom at gz level
            physicsClientId=cli
        )
        
        # Set platform material properties for MAXIMUM stability
        p.changeDynamics(
            bodyUniqueId=platform_uid,
            linkIndex=-1,
            restitution=0.0,      # NO bounce whatsoever
            lateralFriction=2.0,  # VERY high friction to prevent sliding
            spinningFriction=1.0, # High spinning friction
            rollingFriction=0.5,  # High rolling friction
            physicsClientId=cli
        )

        # 3)landing zone ---------------
        # Create multiple layers for depth and glow effect
        surface_radius = platform_radius * 0.8  # Slightly smaller than platform
        surface_height = 0.008                  # Slightly thicker for better visibility
        
        # Main green landing surface with glow effect
        surface_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=surface_radius,
            length=surface_height,
            rgbaColor=[0.3, 0.9, 0.4, 0.9],  # Bright glowing green with slight transparency
            specularColor=[0.6, 1.0, 0.6],   # Green specular highlight
            physicsClientId=cli,
        )
        
        # Position main green surface on top of platform
        surface_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # No collision for surface
            baseVisualShapeIndex=surface_visual,
            basePosition=[gx, gy, gz + surface_height / 2 + 0.001],  # On platform top
            physicsClientId=cli,
        )
        
        # Add SOLID FLAT landing surface for stable drone landing
        # This invisible collision surface ensures drone lands on completely flat surface
        flat_landing_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=surface_radius,  # Same size as green circle
            height=0.001,           # Paper-thin but solid
            physicsClientId=cli,
        )
        
        flat_landing_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=flat_landing_collision,
            baseVisualShapeIndex=-1,  # Invisible
            basePosition=[gx, gy, gz + surface_height + 0.002],  # Exactly on green surface
            physicsClientId=cli
        )
        
        # Set maximum friction for this landing surface
        p.changeDynamics(
            bodyUniqueId=flat_landing_uid,
            linkIndex=-1,
            restitution=0.0,      # No bounce at all
            lateralFriction=3.0,  # MAXIMUM friction to prevent sliding
            spinningFriction=2.0,
            rollingFriction=1.0,
            physicsClientId=cli
        )

        # TAO logo as MASSIVE CIRCULAR badge covering the ENTIRE green surface  
        # Make it BIG and OBVIOUS - covering all the green area
        tao_logo_radius = surface_radius * 1.06  # Cover all of green circle
        badge_height = 0.005       # Thicker for visibility
        
        # Create LARGE white circular background first
        tao_background_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=tao_logo_radius,
            length=badge_height,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],  # Pure white opaque background
            physicsClientId=cli,
        )

        # Position the white background
        tao_background_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # No collision
            baseVisualShapeIndex=tao_background_visual,
            basePosition=[gx, gy, gz + surface_height + badge_height + 0.008],  # Higher for visibility
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=cli,
        )
        
        # Create MASSIVE circular TAO logo with texture on top
        tao_logo_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=tao_logo_radius * 0.95,  # Slightly smaller for border effect
            length=badge_height * 0.5,      # Thinner texture layer
            rgbaColor=[1.0, 1.0, 1.0, 1.0],  # White for texture
            physicsClientId=cli,
        )

        # Position the TAO logo texture on top of background
        tao_logo_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # No collision
            baseVisualShapeIndex=tao_logo_visual,
            basePosition=[gx, gy, gz + surface_height + badge_height + 0.011],  # On top of background
            baseOrientation=[0, 0, 0, 1],
            physicsClientId=cli,
        )
        
        # Apply TAO texture to the MASSIVE logo
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
            baseCollisionShapeIndex=-1,  # No collision for pole
            baseVisualShapeIndex=pole_visual,
            basePosition=[gx, gy, gz + pole_h / 2 + 0.008],  # Above platform
            physicsClientId=cli,
        )
        
        # Position beacon cap on top
        cap_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,  # No collision for cap
            baseVisualShapeIndex=cap_visual,
            basePosition=[gx, gy, gz + pole_h + 0.015],  # Top of pole
            physicsClientId=cli,
        )
