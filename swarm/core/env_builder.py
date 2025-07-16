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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pybullet as p

# Absolute path to the birds assets directory for reliable loading
BIRD_ASSETS_DIR = (Path(__file__).parent.parent / "assets" / "birds").resolve()

from swarm.constants import WORLD_RANGE, HEIGHT_SCALE, N_OBSTACLES, LANDING_PLATFORM_RADIUS, PLATFORM, ENABLE_BIRDS, N_BIRDS, BIRD_SIZE, ENABLE_WIND, WIND_SPEED_MIN, WIND_SPEED_MAX, WIND_DIRECTION_CHANGE_INTERVAL, ENABLE_MOVING_PLATFORM, PLATFORM_MOTION_TYPE, PLATFORM_MOTION_SPEED, PLATFORM_MOTION_RADIUS, PLATFORM_PATH_LENGTH, PLATFORM_SAFE_MARGIN

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
SAFE_ZONE_RADIUS = 2.0         # keep at least 2 m of clearance
MAX_ATTEMPTS_PER_OBS = 100     # retry limit when placing each obstacle

# Avian Simulation Parameters - Enhanced Realistic Flight Dynamics
BIRD_HEIGHT_MIN = 4.0          # Minimum flight altitude above ground (meters)
BIRD_HEIGHT_MAX = 12.0         # Maximum flight altitude above ground (meters)
BIRD_SPEED_MIN = 2.5           # Minimum flight velocity (m/s)
BIRD_SPEED_MAX = 5.0           # Maximum flight velocity (m/s)
BIRD_INDIVIDUAL_RATIO = 0.4    # Proportion of solitary avian entities
BIRD_FLOCK_RATIO = 0.6         # Proportion of grouped avian entities

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
        try:
            if tex_path.exists():
                _TAO_TEX_ID[cli] = p.loadTexture(str(tex_path))
            else:
                print(f"Warning: TAO texture not found at {tex_path}")
                _TAO_TEX_ID[cli] = -1
        except Exception as e:
            print(f"Error loading TAO texture: {e}")
            _TAO_TEX_ID[cli] = -1
    return _TAO_TEX_ID[cli]

# --------------------------------------------------------------------------
# Moving platform system
# --------------------------------------------------------------------------
@dataclass(slots=True)
class _MovingPlatform:
    cli: int
    platform_uids: List[int]
    motion_type: str
    speed: float
    radius: float
    path_length: float
    center: np.ndarray
    rng: random.Random
    
    # Runtime state variables
    elapsed: float = 0.0
    pos: np.ndarray = None
    direction: int = 1
    phase_offset: float = 0.0
    speed_variation: float = 1.0
    
    def __post_init__(self):
        self.elapsed = 0.0
        self.pos = self.center.copy()
        self.direction = 1
        self.phase_offset = self.rng.uniform(0, 2 * math.pi)
        self.speed_variation = self.rng.uniform(0.8, 1.2)
    
    def _calculate_component_position(self, component_index: int) -> List[float]:
        if component_index == 0:
            return [self.pos[0], self.pos[1], self.pos[2] - 0.06]
        elif component_index == 1:
            return [self.pos[0], self.pos[1], self.pos[2] + 0.001]
        elif component_index == 2:
            return [self.pos[0], self.pos[1], self.pos[2] + 0.002]
        elif component_index == 3:
            return [self.pos[0], self.pos[1], self.pos[2] + 0.01]
        elif component_index == 4:
            return [self.pos[0], self.pos[1], self.pos[2] + 0.015]
        elif component_index == 5:
            return [self.pos[0], self.pos[1], self.pos[2] + 0.25]
        elif component_index == 6:
            return [self.pos[0], self.pos[1], self.pos[2] + 0.5]
        else:
            return [self.pos[0], self.pos[1], self.pos[2]]

    def step(self, dt: float) -> None:
        try:
            old_pos = self.pos.copy()
            self.elapsed += dt
            effective_speed = self.speed * self.speed_variation
            if self.motion_type == "circular":
                angle = self.phase_offset + self.elapsed * (effective_speed / self.radius)
                self.pos[0] = self.center[0] + self.radius * np.cos(angle)
                self.pos[1] = self.center[1] + self.radius * np.sin(angle)
                self.pos[2] = self.center[2]
            else:
                travel = (self.elapsed * effective_speed) % (2 * self.path_length)
                if travel > self.path_length:
                    travel = 2 * self.path_length - travel
                self.pos[0] = self.center[0] + travel - self.path_length / 2
                self.pos[1] = self.center[1]
                self.pos[2] = self.center[2]
            # Restore visual movement for all platform components
            for i, uid in enumerate(self.platform_uids):
                pos = self._calculate_component_position(i)
                p.resetBasePositionAndOrientation(uid, pos, [0, 0, 0, 1], physicsClientId=self.cli)
        except Exception:
            self.pos = old_pos

# --------------------------------------------------------------------------
# Main world builder
# --------------------------------------------------------------------------
def build_world(
    seed: int,
    cli: int,
    *,
    start: Optional[Tuple[float, float, float]] = None,
    goal: Optional[Tuple[float, float, float]] = None,
) -> Tuple[List[int], List, List[int]]:
    """
    Create procedural obstacles (with safe‑zone constraints) and—if *goal*
    is provided—place a visual TAO badge at that position.

    Parameters
    ----------
    seed   : int      • PRNG seed so miners and validator share the same map
    cli    : int      • PyBullet client id
    start  : (x,y,z)  • drone take‑off location (obstacles keep clear)
    goal   : (x,y,z)  • desired target (obstacles keep clear; visual marker)

    Returns
    -------
    Tuple[List[int], List, List[int]] • (bird IDs, obstacles, platform UIDs)
    """
    rng = random.Random(seed)

    sx, sy = (start[0], start[1]) if start is not None else (None, None)
    gx, gy = (goal[0], goal[1]) if goal is not None else (None, None)

    # ------------------------------------------------------------------
    # Random obstacles with safe‑zone rejection
    # ------------------------------------------------------------------
    placed = 0
    placed_obstacles = []  # Track all placed obstacles: [(x, y, radius), ...]
    MIN_OBSTACLE_DISTANCE = 0.6  # Reduced minimum distance between obstacles
    
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

            # — moving platform path avoidance ------------------------
            if ENABLE_MOVING_PLATFORM and gx is not None and gy is not None:
                # Calculate motion envelope based on platform type
                if PLATFORM_MOTION_TYPE == "circular":
                    motion_range = PLATFORM_MOTION_RADIUS
                else:  # linear
                    motion_range = PLATFORM_PATH_LENGTH * 0.5
                
                # Total reserved area around platform path
                reserved_clearance = (LANDING_PLATFORM_RADIUS + 
                                    SAFE_ZONE_RADIUS + 
                                    motion_range + 
                                    PLATFORM_SAFE_MARGIN + 
                                    obj_r)
                
                # Check if obstacle would intersect with platform motion path
                if math.hypot(x - gx, y - gy) < reserved_clearance:
                    continue  # too close to platform motion path – try another location

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
            break

    # ------------------------------------------------------------------
    # World building report
    # ------------------------------------------------------------------
    if placed < N_OBSTACLES:
        if placed < N_OBSTACLES * 0.8:
            pass  

    # ------------------------------------------------------------------
    # Physical landing platform with visual goal marker
    # ------------------------------------------------------------------
    platform_uids = []  # Initialize at function level
    
    if goal is not None:
        gx, gy, gz = goal

        # Platform mode: solid if PLATFORM else visual-only
        if PLATFORM:
            # Physical landing platform base
            platform_radius = LANDING_PLATFORM_RADIUS * 0.9
            platform_height = 0.12
            
            # Create circular platform collision shape
            platform_collision = p.createCollisionShape(
                shapeType=p.GEOM_CYLINDER,
                radius=platform_radius,
                height=platform_height,
                physicsClientId=cli,
            )
            
            # Create platform visual shape
            platform_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=platform_radius,
                length=platform_height,
                rgbaColor=[0.15, 0.35, 0.8, 1.0],
                specularColor=[0.8, 0.8, 0.9],
                physicsClientId=cli,
            )
            
            # Create the physical platform body
            platform_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=platform_collision,
                baseVisualShapeIndex=platform_visual,
                basePosition=[gx, gy, gz - platform_height / 2],
                physicsClientId=cli
            )
            
            # Set platform material properties
            p.changeDynamics(
                bodyUniqueId=platform_uid,
                linkIndex=-1,
                restitution=0.0,
                lateralFriction=1.5,
                spinningFriction=0.5,
                rollingFriction=0.3,
                physicsClientId=cli
            )
            

            # Landing zone visual surface
            surface_radius = platform_radius * 0.8
            surface_height = 0.006
            
            # Green landing surface visual
            surface_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=surface_radius,
                length=surface_height,
                rgbaColor=[0.3, 0.9, 0.4, 0.9],
                specularColor=[0.6, 1.0, 0.6],
                physicsClientId=cli,
            )
            
            # Position green surface on platform top
            surface_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=surface_visual,
                basePosition=[gx, gy, gz + surface_height / 2 + 0.001],
                physicsClientId=cli,
            )
            
            # Thin flat landing collision surface
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
            
            # Set landing surface friction
            p.changeDynamics(
                bodyUniqueId=flat_landing_uid,
                linkIndex=-1,
                restitution=0.0,
                lateralFriction=1.0,
                spinningFriction=0.5,
                rollingFriction=0.3,
                physicsClientId=cli
            )

            # TAO logo badge
            tao_logo_radius = surface_radius * 1.06
            badge_height = 0.004
            
            # White circular background
            tao_background_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=tao_logo_radius,
                length=badge_height,
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                physicsClientId=cli,
            )

            # Position background
            tao_background_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=tao_background_visual,
                basePosition=[gx, gy, gz + surface_height + badge_height + 0.008],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=cli,
            )
            
            # Create circular mesh for TAO logo
            tao_logo_radius_inner = tao_logo_radius * 0.98
            num_segments = 32
            vertices = [[0.0, 0.0, 0.0]]
            indices = []
            uvs = [[0.5, 0.5]]
            
            # Generate circle vertices
            for i in range(num_segments):
                angle = 2 * math.pi * i / num_segments
                x = tao_logo_radius_inner * math.cos(angle)
                y = tao_logo_radius_inner * math.sin(angle)
                vertices.append([x, y, 0.0])
                uvs.append([0.5 + 0.5 * math.cos(angle), 0.5 + 0.5 * math.sin(angle)])
            
            # Create triangular faces
            for i in range(num_segments):
                next_i = (i + 1) % num_segments
                indices.extend([0, i + 1, next_i + 1])
            
            tao_logo_visual = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                vertices=vertices,
                indices=indices,
                uvs=uvs,
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                physicsClientId=cli,
            )

            # Position TAO logo
            tao_logo_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=tao_logo_visual,
                basePosition=[gx, gy, gz + surface_height + badge_height + 0.011],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=cli,
            )
            
            # Apply TAO texture
            tao_tex = _get_tao_tex(cli)
            if tao_tex != -1:
                p.changeVisualShape(
                    tao_logo_uid,
                    -1,
                    textureUniqueId=tao_tex,
                    flags=p.VISUAL_SHAPE_DOUBLE_SIDED,
                    physicsClientId=cli,
                )

            # Guidance beacon pole
            pole_h = 0.5
            pole_radius = 0.012
            
            # Main beacon pole
            pole_visual = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=pole_radius,
                length=pole_h,
                rgbaColor=[1.0, 0.2, 0.1, 0.9],
                specularColor=[1.0, 0.8, 0.2],
                physicsClientId=cli,
            )
            
            # Beacon top cap
            cap_visual = p.createVisualShape(
                shapeType=p.GEOM_SPHERE,
                radius=pole_radius * 2,
                rgbaColor=[1.0, 0.3, 0.0, 1.0],
                specularColor=[1.0, 1.0, 0.4],
                physicsClientId=cli,
            )
            
            # Position beacon pole
            pole_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=pole_visual,
                basePosition=[gx, gy, gz + pole_h / 2 + 0.008],
                physicsClientId=cli,
            )
            
            # Position beacon cap
            cap_uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=cap_visual,
                basePosition=[gx, gy, gz + pole_h + 0.015],
                physicsClientId=cli,
            )
            
            # Collect all platform component UIDs for moving platform system
            platform_uids = [
                platform_uid, surface_uid, flat_landing_uid, 
                tao_background_uid, tao_logo_uid, pole_uid, cap_uid
            ]
        
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
            p.createMultiBody(
                0,
                -1,
                pole_vis,
                [gx, gy, gz + pole_h / 2 + 0.001],
                physicsClientId=cli,
            )

    # Spawn intelligent bird distribution based on flight distance
    bird_ids = _spawn_birds(cli, rng, placed_obstacles, start, goal)
    return bird_ids, placed_obstacles, platform_uids


def _calculate_smart_height_distribution() -> dict:
    """
    Calculate smart height distribution for birds with more randomness.
    
    Height zones with increased randomness:
    - Low altitude: 4-6m (20% of birds)
    - Mid altitude: 6-9m (30% of birds) 
    - High altitude: 9-11m (30% of birds)
    - Very high altitude: 11-12m (20% of birds)
    
    Returns dict with height ranges and ratios.
    """
    return {
        'low_altitude': {
            'min': BIRD_HEIGHT_MIN,
            'max': BIRD_HEIGHT_MIN + 2.0,
            'ratio': 0.20
        },
        'mid_altitude': {
            'min': BIRD_HEIGHT_MIN + 2.0,
            'max': BIRD_HEIGHT_MIN + 5.0,
            'ratio': 0.30
        },
        'high_altitude': {
            'min': BIRD_HEIGHT_MIN + 5.0,
            'max': BIRD_HEIGHT_MAX - 1.0,
            'ratio': 0.30
        },
        'very_high_altitude': {
            'min': BIRD_HEIGHT_MAX - 1.0,
            'max': BIRD_HEIGHT_MAX,
            'ratio': 0.20
        }
    }

def _calculate_smart_bird_distribution(start_pos: Optional[Tuple] = None, 
                                     goal_pos: Optional[Tuple] = None) -> dict:
    """
    Calculate intelligent bird distribution ratios based on flight distance.
    
    Distance-based distribution:
    - 5m: 20% flight path birds (further increased)
    - 8m: 25% flight path birds (further increased)
    - 12m: 30% flight path birds (further increased)
    - 16m: 35% flight path birds (further increased)
    - 20m: 40% flight path birds (further increased)
    - 25m: 45% flight path birds (further increased)
    - 30m+: 50% flight path birds (further increased)
    
    Returns dict with zone ratios: {'flight_path': 0.30, 'perimeter': 0.20, ...}
    """
    if not start_pos or not goal_pos:
        # Default distribution when no start/goal positions
        return {
            'flight_path': 0.30,  # Increased from 0.25
            'perimeter': 0.20,    # Reduced from 0.25
            'high_altitude': 0.25, # Same
            'random': 0.25        # Same
        }
    
    # Calculate flight distance
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    flight_distance = math.hypot(dx, dy)
    
    # Smart distribution based on distance - FURTHER INCREASED PERCENTAGES
    if flight_distance <= 5.0:
        flight_path_ratio = 0.20  # Increased from 0.15
    elif flight_distance <= 8.0:
        flight_path_ratio = 0.25  # Increased from 0.20
    elif flight_distance <= 12.0:
        flight_path_ratio = 0.30  # Increased from 0.25
    elif flight_distance <= 16.0:
        flight_path_ratio = 0.35  # Increased from 0.30
    elif flight_distance <= 20.0:
        flight_path_ratio = 0.40  # Increased from 0.35
    elif flight_distance <= 25.0:
        flight_path_ratio = 0.45  # Increased from 0.40
    else:
        flight_path_ratio = 0.50  # Increased from 0.45
    
    # Adjust other zones proportionally
    remaining_ratio = 1.0 - flight_path_ratio
    perimeter_ratio = remaining_ratio * 0.30    # Reduced from 0.35
    high_altitude_ratio = remaining_ratio * 0.35  # Increased from 0.30
    random_ratio = remaining_ratio * 0.35       # Same
    
    return {
        'flight_path': flight_path_ratio,
        'perimeter': perimeter_ratio,
        'high_altitude': high_altitude_ratio,
        'random': random_ratio
    }

def _find_safe_bird_position(rng: random.Random, placed_obstacles: List, 
                            start_pos: Optional[Tuple] = None, 
                            goal_pos: Optional[Tuple] = None,
                            zone_type: str = "random", height_zone: str = "mid_altitude") -> List[float]:
    """
    Find safe bird position with strategic zone distribution.
    
    Zone types:
    - 'flight_path': Birds along the drone's flight corridor
    - 'perimeter': Birds around map boundaries
    - 'high_altitude': Birds at maximum heights
    - 'random': Birds scattered throughout the map
    
    Returns [x, y, z] position or fallback position if no safe spot found.
    """
    
    for _ in range(100):  # Maximum attempts to find safe position
        if zone_type == "flight_path" and start_pos and goal_pos:
            # Birds along flight corridor - intelligent distribution with random heights
            # Better distribution along the entire flight path with more randomness
            t = rng.uniform(0.05, 0.95)  # Full flight path coverage with edge avoidance
            path_x = start_pos[0] + t * (goal_pos[0] - start_pos[0])
            path_y = start_pos[1] + t * (goal_pos[1] - start_pos[1])
            
            # Wider offset perpendicular to flight path for better coverage
            perpendicular_offset = rng.uniform(-15, 15)  # Increased range for more spread
            parallel_offset = rng.uniform(-10, 10)  # Increased range for more spread
            
            # Calculate perpendicular direction
            dx = goal_pos[0] - start_pos[0]
            dy = goal_pos[1] - start_pos[1]
            length = math.hypot(dx, dy)
            if length > 0:
                perp_x = -dy / length * perpendicular_offset
                perp_y = dx / length * perpendicular_offset
                par_x = dx / length * parallel_offset
                par_y = dy / length * parallel_offset
            else:
                perp_x = perp_y = par_x = par_y = 0
            
            x = path_x + perp_x + par_x
            y = path_y + perp_y + par_y
            
            # More random height distribution for flight path birds
            # Birds can be at any height with equal probability
            z = rng.uniform(BIRD_HEIGHT_MIN, BIRD_HEIGHT_MAX)
            
        elif zone_type == "perimeter":
            # Birds around map perimeter - better distribution along all sides
            side = rng.choice(['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest'])
            if side == 'north':
                x = rng.uniform(-WORLD_RANGE + 8, WORLD_RANGE - 8)
                y = rng.uniform(WORLD_RANGE - 15, WORLD_RANGE - 8)
            elif side == 'south':
                x = rng.uniform(-WORLD_RANGE + 8, WORLD_RANGE - 8)
                y = rng.uniform(-WORLD_RANGE + 8, -WORLD_RANGE + 15)
            elif side == 'east':
                x = rng.uniform(WORLD_RANGE - 15, WORLD_RANGE - 8)
                y = rng.uniform(-WORLD_RANGE + 8, WORLD_RANGE - 8)
            elif side == 'west':
                x = rng.uniform(-WORLD_RANGE + 8, -WORLD_RANGE + 15)
                y = rng.uniform(-WORLD_RANGE + 8, WORLD_RANGE - 8)
            elif side == 'northeast':
                x = rng.uniform(WORLD_RANGE - 20, WORLD_RANGE - 10)
                y = rng.uniform(WORLD_RANGE - 20, WORLD_RANGE - 10)
            elif side == 'northwest':
                x = rng.uniform(-WORLD_RANGE + 10, -WORLD_RANGE + 20)
                y = rng.uniform(WORLD_RANGE - 20, WORLD_RANGE - 10)
            elif side == 'southeast':
                x = rng.uniform(WORLD_RANGE - 20, WORLD_RANGE - 10)
                y = rng.uniform(-WORLD_RANGE + 10, -WORLD_RANGE + 20)
            else:  # southwest
                x = rng.uniform(-WORLD_RANGE + 10, -WORLD_RANGE + 20)
                y = rng.uniform(-WORLD_RANGE + 10, -WORLD_RANGE + 20)
            # Use smart height distribution for perimeter birds
            height_dist = _calculate_smart_height_distribution()
            height_range = height_dist['low_altitude']  # Perimeter birds prefer lower altitudes
            z = rng.uniform(height_range['min'], height_range['max'])
            
        elif zone_type == "high_altitude":
            # High-flying birds - better distribution across the entire map
            # Use different patterns for high altitude birds
            pattern = rng.choice(['center', 'corners', 'edges', 'random'])
            
            if pattern == 'center':
                # Birds in the center area
                x = rng.uniform(-WORLD_RANGE + 15, WORLD_RANGE - 15)
                y = rng.uniform(-WORLD_RANGE + 15, WORLD_RANGE - 15)
            elif pattern == 'corners':
                # Birds in corner areas
                corner = rng.choice(['nw', 'ne', 'sw', 'se'])
                if corner == 'nw':
                    x = rng.uniform(-WORLD_RANGE + 5, -WORLD_RANGE + 20)
                    y = rng.uniform(WORLD_RANGE - 20, WORLD_RANGE - 5)
                elif corner == 'ne':
                    x = rng.uniform(WORLD_RANGE - 20, WORLD_RANGE - 5)
                    y = rng.uniform(WORLD_RANGE - 20, WORLD_RANGE - 5)
                elif corner == 'sw':
                    x = rng.uniform(-WORLD_RANGE + 5, -WORLD_RANGE + 20)
                    y = rng.uniform(-WORLD_RANGE + 5, -WORLD_RANGE + 20)
                else:  # se
                    x = rng.uniform(WORLD_RANGE - 20, WORLD_RANGE - 5)
                    y = rng.uniform(-WORLD_RANGE + 5, -WORLD_RANGE + 20)
            elif pattern == 'edges':
                # Birds along edges
                edge = rng.choice(['north', 'south', 'east', 'west'])
                if edge == 'north':
                    x = rng.uniform(-WORLD_RANGE + 10, WORLD_RANGE - 10)
                    y = rng.uniform(WORLD_RANGE - 15, WORLD_RANGE - 8)
                elif edge == 'south':
                    x = rng.uniform(-WORLD_RANGE + 10, WORLD_RANGE - 10)
                    y = rng.uniform(-WORLD_RANGE + 8, -WORLD_RANGE + 15)
                elif edge == 'east':
                    x = rng.uniform(WORLD_RANGE - 15, WORLD_RANGE - 8)
                    y = rng.uniform(-WORLD_RANGE + 10, WORLD_RANGE - 10)
                else:  # west
                    x = rng.uniform(-WORLD_RANGE + 8, -WORLD_RANGE + 15)
                    y = rng.uniform(-WORLD_RANGE + 10, WORLD_RANGE - 10)
            else:  # random
                x = rng.uniform(-WORLD_RANGE + 8, WORLD_RANGE - 8)
                y = rng.uniform(-WORLD_RANGE + 8, WORLD_RANGE - 8)
            # Use smart height distribution for high altitude birds
            height_dist = _calculate_smart_height_distribution()
            height_range = height_dist['very_high_altitude']  # High altitude birds prefer maximum heights
            z = rng.uniform(height_range['min'], height_range['max'])
            
        else:  # random distribution
            # Birds scattered throughout map - better coverage with grid-like distribution
            # Use grid-based random selection for more even distribution
            grid_size = 6  # Divide map into 6x6 grid
            grid_x = rng.randint(0, grid_size - 1)
            grid_y = rng.randint(0, grid_size - 1)
            
            # Calculate position within the selected grid cell
            cell_width = (2 * WORLD_RANGE - 20) / grid_size
            x = -WORLD_RANGE + 10 + grid_x * cell_width + rng.uniform(0, cell_width)
            y = -WORLD_RANGE + 10 + grid_y * cell_width + rng.uniform(0, cell_width)
            # Use smart height distribution for random birds
            height_dist = _calculate_smart_height_distribution()
            # Random birds can be at any height, weighted by distribution
            height_zones = list(height_dist.keys())
            weights = [height_dist[zone]['ratio'] for zone in height_zones]
            chosen_zone = rng.choices(height_zones, weights=weights)[0]
            height_range = height_dist[chosen_zone]
            z = rng.uniform(height_range['min'], height_range['max'])
        
        safe = True
        
        # Check obstacles with minimal safety margin
        for obs_x, obs_y, obs_r in placed_obstacles:
            distance = math.hypot(x - obs_x, y - obs_y)
            if distance < (obs_r + 2.5):  # Small safety margin
                safe = False
                break
        
        # Minimal safe zones - only 1m protection around start and goal
        if safe and start_pos:
            start_distance = math.hypot(x - start_pos[0], y - start_pos[1])
            if start_distance < 1.0:  # 1m safety zone around start position
                safe = False
        
        if safe and goal_pos:
            goal_distance = math.hypot(x - goal_pos[0], y - goal_pos[1])
            if goal_distance < 1.0:  # 1m safety zone around goal position
                safe = False
        
        if safe:
            return [x, y, z]
    
    # Fallback: random safe position when no suitable location found
    return [
        rng.uniform(-WORLD_RANGE + 15, WORLD_RANGE - 15),
        rng.uniform(-WORLD_RANGE + 15, WORLD_RANGE - 15),
        rng.uniform(BIRD_HEIGHT_MIN + 2, BIRD_HEIGHT_MAX - 2)
    ]

def _spawn_birds(cli: int, rng: random.Random, placed_obstacles: List,
                start_pos: Optional[Tuple] = None, 
                goal_pos: Optional[Tuple] = None) -> List[int]:
    """
    Spawn birds with intelligent distribution based on flight distance.
    
    Uses smart ratios that adapt to the distance between start and goal:
    - Short flights (≤10m): 18% flight path birds
    - Medium flights (10-20m): 25% flight path birds  
    - Long flights (20-30m): 30% flight path birds
    - Very long flights (>30m): 40% flight path birds
    """
    if not ENABLE_BIRDS:
        return []
    
    bird_ids = []
    p.setAdditionalSearchPath(str(BIRD_ASSETS_DIR))
    
    # Calculate intelligent distribution based on flight distance
    distribution = _calculate_smart_bird_distribution(start_pos, goal_pos)
    
    # Create zones with smart distribution ratios
    zones = [
        ("flight_path", int(N_BIRDS * distribution['flight_path'])),
        ("perimeter", int(N_BIRDS * distribution['perimeter'])),
        ("high_altitude", int(N_BIRDS * distribution['high_altitude'])),
        ("random", int(N_BIRDS * distribution['random']))
    ]
    
    # Ensure we spawn exactly N_BIRDS by adjusting the last zone if needed
    total_assigned = sum(count for _, count in zones)
    if total_assigned < N_BIRDS:
        zones[-1] = (zones[-1][0], zones[-1][1] + (N_BIRDS - total_assigned))
    
    # Get height distribution for intelligent placement
    height_dist = _calculate_smart_height_distribution()
    
    for zone_type, bird_count in zones:
        for _ in range(bird_count):
            # Determine appropriate height zone for this bird
            if zone_type == "flight_path":
                height_zone = rng.choices(
                    ['low_altitude', 'mid_altitude', 'high_altitude'],
                    weights=[0.3, 0.5, 0.2]
                )[0]
            elif zone_type == "perimeter":
                height_zone = "low_altitude"  # Perimeter birds stay low
            elif zone_type == "high_altitude":
                height_zone = "very_high_altitude"  # High altitude birds go very high
            else:  # random
                height_zone = rng.choices(
                    list(height_dist.keys()),
                    weights=[height_dist[zone]['ratio'] for zone in height_dist.keys()]
                )[0]
            
            pos = _find_safe_bird_position(rng, placed_obstacles, start_pos, goal_pos, zone_type, height_zone)
            bird_id = p.loadURDF(str(BIRD_ASSETS_DIR / "bird.urdf"), basePosition=pos, physicsClientId=cli)
            bird_ids.append(bird_id)
    
    return bird_ids

class BirdSystem:
    """
    Advanced bird simulation system for drone environment.
    
    Features:
    - Multiple flight patterns (circular, linear, patrol)
    - Realistic physics-based movement
    - Intelligent obstacle avoidance
    - Personality-driven behavior variations
    - Dynamic energy management system
    
    Designed for realistic drone simulation environments.
    """
    
    def __init__(self, cli: int, bird_ids: List[int], obstacles: List, rng: random.Random, 
                 start_pos: Optional[Tuple] = None, goal_pos: Optional[Tuple] = None):
        """
        Initialize the bird simulation system with behavioral parameters.
        
        Args:
            cli: PyBullet physics client identifier
            bird_ids: List of bird entity IDs in physics simulation
            obstacles: List of obstacle coordinates and dimensions
            rng: Seeded random number generator for deterministic behavior
            start_pos: Drone starting coordinates (for safe zone calculation)
            goal_pos: Drone target coordinates (for safe zone calculation)
        """
        self.cli = cli
        self.bird_ids = bird_ids
        self.obstacles = obstacles
        self.rng = rng
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.birds = []
        
        # Initialize behavioral parameters for each bird entity
        for bird_id in bird_ids:
            # Assign flight pattern: circular, linear, or patrol
            behavior = rng.choice(['circle', 'straight', 'patrol'])
            speed = rng.uniform(BIRD_SPEED_MIN, BIRD_SPEED_MAX)
            
            # Core bird entity data structure
            bird_data = {
                'id': bird_id,
                'behavior': behavior,
                'speed': speed,
                'timer': 0.0,
                'duration': rng.uniform(15, 30),  # Behavior duration in seconds
                'energy': rng.uniform(0.6, 1.0),  # Energy level (0.0 to 1.0)
                'personality': rng.choice(['aggressive', 'cautious', 'curious', 'calm'])
            }
            
            # Initialize pattern-specific movement parameters
            if behavior == 'circle':
                # Circular flight pattern configuration
                pos, _ = p.getBasePositionAndOrientation(bird_id, physicsClientId=cli)
                bird_data.update({
                    'center': np.array(pos) + np.array([rng.uniform(-8, 8), rng.uniform(-8, 8), 0]),
                    'radius': rng.uniform(4, 10),
                    'angle': 0.0
                })
            elif behavior == 'straight':
                # Linear flight pattern configuration
                bird_data.update({
                    'velocity': np.array([rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-0.2, 0.2)]),
                    'direction_change_timer': 0.0
                })
            elif behavior == 'patrol':
                # Patrol flight pattern configuration
                bird_data.update({
                    'waypoints': [
                        np.array([rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5),
                                rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5)]),
                        np.array([rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5),
                                rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5)]),
                        np.array([rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5),
                                rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5)])
                    ],
                    'current_waypoint': 0,
                    'waypoint_progress': 0.0
                })
            
            self.birds.append(bird_data)
    
    def update(self, dt: float):
        """
        Update bird positions and behavioral states for current simulation frame.
        
        Args:
            dt: Time delta for physics integration
        """
        for bird in self.birds:
            # Skip birds that have collided with drone
            if bird.get('collision_hit', False):
                continue
            
            # Retrieve current spatial position
            pos, _ = p.getBasePositionAndOrientation(bird['id'], physicsClientId=self.cli)
            pos = np.array(pos)
            
            # Compute movement vector based on behavioral pattern
            direction = self._calculate_movement(bird, pos, dt)
            
            # Apply collision avoidance forces
            avoidance = self._calculate_obstacle_avoidance(bird, pos)
            
            # Apply safe zone repulsion forces
            safe_zone_force = self._calculate_safe_zone_avoidance(bird, pos)
            
            # Apply boundary constraint forces
            boundary_force = self._calculate_boundary_force(bird, pos)
            
            # Synthesize total movement force vector
            total_force = direction * 0.7 + avoidance * 0.2 + safe_zone_force * 0.05 + boundary_force * 0.05
            
            # Apply velocity limiting for realistic movement
            max_speed = 1.2  # Maximum velocity in meters per second
            force_magnitude = np.linalg.norm(total_force)
            if force_magnitude > max_speed:
                total_force = total_force * max_speed / force_magnitude
            
            # Integrate position update
            new_pos = pos + total_force * dt
            
            # Constrain altitude to valid flight range
            new_pos[2] = max(BIRD_HEIGHT_MIN, min(BIRD_HEIGHT_MAX, new_pos[2]))
            
            # Update bird entity position in physics simulation
            p.resetBasePositionAndOrientation(
                bird['id'], 
                new_pos, 
                [0, 0, 0, 1], 
                physicsClientId=self.cli
            )
            # Update behavioral timers and energy consumption
            bird['timer'] += dt
            bird['energy'] = max(0.0, bird['energy'] - 0.002)  # Energy decay rate
            
            # Trigger behavioral state transition when timer expires
            if bird['timer'] > bird['duration']:
                self._switch_behavior(bird, new_pos)
    
    def _calculate_movement(self, bird: dict, pos: np.ndarray, dt: float) -> np.ndarray:
        """
        Calculate movement direction vector based on bird behavioral pattern.
        
        Args:
            bird: Bird entity data dictionary
            pos: Current spatial position
            dt: Time delta for integration
            
        Returns:
            Movement direction vector in 3D space
        """
        if bird['behavior'] == 'circle':
            return self._circle_movement(bird, pos, dt)
        elif bird['behavior'] == 'straight':
            return self._straight_movement(bird, pos, dt)
        elif bird['behavior'] == 'patrol':
            return self._patrol_movement(bird, pos, dt)
        else:
            return np.array([0, 0, 0])
    
    def _circle_movement(self, bird: dict, pos: np.ndarray, dt: float) -> np.ndarray:
        """Calculate circular flight pattern movement vector."""
        center = bird['center']
        radius = bird['radius']
        
        # Compute angular velocity based on personality traits
        base_speed = bird['speed'] * 0.15  # Base angular velocity multiplier
        if bird['personality'] == 'aggressive':
            base_speed *= 1.5  # Aggressive personality speed multiplier
        elif bird['personality'] == 'calm':
            base_speed *= 0.8  # Calm personality speed multiplier
        
        # Update angular position
        bird['angle'] += base_speed * dt
        
        # Calculate target position on circular trajectory
        target = center + np.array([
            np.cos(bird['angle']) * radius,
            np.sin(bird['angle']) * radius,
            pos[2]  # Maintain current altitude
        ])
        
        # Return direction vector toward target position
        return (target - pos) * 0.7  # Movement strength coefficient
    
    def _straight_movement(self, bird: dict, pos: np.ndarray, dt: float) -> np.ndarray:
        """Calculate linear flight pattern with periodic direction changes."""
        # Update velocity vector at regular intervals
        if bird['timer'] % 2.0 < dt:  # Direction change interval
            bird['velocity'] = np.array([
                self.rng.uniform(-1.5, 1.5),  # X-axis velocity range
                self.rng.uniform(-1.5, 1.5),  # Y-axis velocity range
                self.rng.uniform(-0.3, 0.3)   # Z-axis velocity range
            ])
        
        # Apply personality-based velocity modifications
        velocity = bird['velocity'].copy()
        if bird['personality'] == 'cautious':
            velocity *= 0.7  # Cautious personality velocity reduction
        elif bird['personality'] == 'aggressive':
            velocity *= 1.6  # Aggressive personality velocity increase
        
        return velocity * bird['speed']
    
    def _patrol_movement(self, bird: dict, pos: np.ndarray, dt: float) -> np.ndarray:
        """Calculate waypoint-based patrol movement pattern."""
        waypoints = bird['waypoints']
        current_idx = bird['current_waypoint']
        progress = bird['waypoint_progress']
        
        # Retrieve current and subsequent waypoint coordinates
        current_waypoint = waypoints[current_idx]
        next_idx = (current_idx + 1) % len(waypoints)
        next_waypoint = waypoints[next_idx]
        
        # Update progress toward next waypoint
        progress += bird['speed'] * dt * 0.03  # Progress rate multiplier
        if progress >= 1.0:
            # Transition to next waypoint in sequence
            bird['current_waypoint'] = next_idx
            progress = 0.0
        
        bird['waypoint_progress'] = progress
        
        # Interpolate target position between waypoints
        target_2d = current_waypoint + (next_waypoint - current_waypoint) * progress
        target = np.array([target_2d[0], target_2d[1], pos[2]])
        
        # Return direction vector toward interpolated target
        return (target - pos) * 0.5  # Movement strength coefficient
    
    def _calculate_obstacle_avoidance(self, bird: dict, pos: np.ndarray) -> np.ndarray:
        """Calculate collision avoidance forces from environmental obstacles."""
        avoidance = np.zeros(3)
        
        # Identify nearest obstacle for collision prevention
        closest_distance = float('inf')
        closest_obstacle = None
        
        for obs_x, obs_y, obs_r in self.obstacles[:10]:  # Evaluate first 10 obstacles
            distance = math.hypot(pos[0] - obs_x, pos[1] - obs_y)
            if distance < closest_distance:
                closest_distance = distance
                closest_obstacle = (obs_x, obs_y, obs_r)
        
        # Apply repulsion force when approaching obstacle threshold
        if closest_obstacle and closest_distance < closest_obstacle[2] + 3.0:
            obs_x, obs_y, obs_r = closest_obstacle
            
            # Compute escape direction vector
            escape_x = pos[0] - obs_x
            escape_y = pos[1] - obs_y
            escape_length = math.hypot(escape_x, escape_y)
            
            if escape_length > 0.1:
                # Normalize escape vector and apply repulsion force
                escape_x /= escape_length
                escape_y /= escape_length
                
                # Adjust avoidance strength based on personality traits
                strength = 5.0  # Base avoidance strength
                if bird['personality'] == 'cautious':
                    strength = 8.0  # Enhanced avoidance for cautious birds
                elif bird['personality'] == 'aggressive':
                    strength = 3.0  # Reduced avoidance for aggressive birds
                
                avoidance[0] = escape_x * strength
                avoidance[1] = escape_y * strength
        
        return avoidance
    
    def _calculate_safe_zone_avoidance(self, bird: dict, pos: np.ndarray) -> np.ndarray:
        """Calculate repulsion forces from drone start and goal positions."""
        safe_zone_force = np.zeros(3)
        safe_radius = 4.0  # Safe zone radius in meters
        
        # Apply repulsion from drone starting position
        if self.start_pos:
            start_distance = math.hypot(pos[0] - self.start_pos[0], pos[1] - self.start_pos[1])
            if start_distance < safe_radius:
                repel_force = (safe_radius - start_distance) / safe_radius
                safe_zone_force[0] = (pos[0] - self.start_pos[0]) / max(start_distance, 0.1) * repel_force
                safe_zone_force[1] = (pos[1] - self.start_pos[1]) / max(start_distance, 0.1) * repel_force
        
        # Apply repulsion from drone goal position
        if self.goal_pos:
            goal_distance = math.hypot(pos[0] - self.goal_pos[0], pos[1] - self.goal_pos[1])
            if goal_distance < safe_radius:
                repel_force = (safe_radius - goal_distance) / safe_radius
                safe_zone_force[0] += (pos[0] - self.goal_pos[0]) / max(goal_distance, 0.1) * repel_force
                safe_zone_force[1] += (pos[1] - self.goal_pos[1]) / max(goal_distance, 0.1) * repel_force
        
        return safe_zone_force
    
    def _calculate_boundary_force(self, bird: dict, pos: np.ndarray) -> np.ndarray:
        """Calculate boundary constraint forces to maintain birds within simulation area."""
        boundary_force = np.zeros(3)
        margin = 5.0  # Boundary margin in meters
        
        # Apply horizontal boundary constraints
        if pos[0] > WORLD_RANGE - margin:
            boundary_force[0] = -3.0  # Repel from positive X boundary
        elif pos[0] < -WORLD_RANGE + margin:
            boundary_force[0] = 3.0   # Repel from negative X boundary
        
        if pos[1] > WORLD_RANGE - margin:
            boundary_force[1] = -3.0  # Repel from positive Y boundary
        elif pos[1] < -WORLD_RANGE + margin:
            boundary_force[1] = 3.0   # Repel from negative Y boundary
        
        # Apply vertical boundary constraints
        if pos[2] < BIRD_HEIGHT_MIN + 1:
            boundary_force[2] = 2.0   # Repel from lower altitude boundary
        elif pos[2] > BIRD_HEIGHT_MAX - 1:
            boundary_force[2] = -2.0  # Repel from upper altitude boundary
        
        return boundary_force
    
    def _switch_behavior(self, bird: dict, pos: np.ndarray):
        """Execute behavioral state transition based on energy levels and personality."""
        old_behavior = bird['behavior']
        
        # Select new behavioral pattern based on energy state
        if bird['energy'] < 0.3:
            # Low energy state: prefer energy-efficient patterns
            new_behavior = self.rng.choice(['circle', 'patrol'])
        elif bird['energy'] > 0.7:
            # High energy state: prefer active movement patterns
            new_behavior = self.rng.choice(['straight', 'patrol', 'circle'])
        else:
            # Normal energy state: all patterns available
            new_behavior = self.rng.choice(['circle', 'straight', 'patrol'])
        
        # Apply personality-based behavioral preferences
        if bird['personality'] == 'aggressive' and new_behavior == 'straight':
            new_behavior = 'straight'  # Maintain aggressive linear movement
        elif bird['personality'] == 'cautious' and new_behavior == 'circle':
            new_behavior = 'circle'    # Maintain cautious circular patterns
        
        # Update behavioral state parameters
        bird['behavior'] = new_behavior
        bird['timer'] = 0.0
        bird['duration'] = self.rng.uniform(20, 40)  # New behavior duration in seconds
        
        # Initialize pattern-specific parameters for new behavior
        if new_behavior == 'circle':
            bird['center'] = np.array(pos) + np.array([self.rng.uniform(-8, 8), self.rng.uniform(-8, 8), 0])
            bird['radius'] = self.rng.uniform(4, 10)
            bird['angle'] = 0.0
        elif new_behavior == 'straight':
            bird['velocity'] = np.array([self.rng.uniform(-1, 1), self.rng.uniform(-1, 1), self.rng.uniform(-0.2, 0.2)])
        elif new_behavior == 'patrol':
            bird['waypoints'] = [
                np.array([self.rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5),
                         self.rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5)]),
                np.array([self.rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5),
                         self.rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5)]),
                np.array([self.rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5),
                         self.rng.uniform(-WORLD_RANGE + 5, WORLD_RANGE - 5)])
            ]
            bird['current_waypoint'] = 0
            bird['waypoint_progress'] = 0.0
        
        # Energy recovery after behavioral transition
        bird['energy'] = min(1.0, bird['energy'] + 0.2)
    
    def handle_bird_collision(self, bird_id: int):
        """
        Process collision event between bird and drone.
        
        Collision response:
        1. Apply realistic physics (mass, gravity, rotation)
        2. Mark bird as inactive in simulation
        3. Remove from active behavioral processing
        
        Args:
            bird_id: Physics entity identifier of collided bird
        """
        try:
            # Apply realistic mass for gravity simulation
            p.changeDynamics(
                bird_id, 
                -1, 
                mass=0.3,  # Realistic bird mass in kilograms
                physicsClientId=self.cli
            )
            
            # Apply falling motion with rotation
            p.resetBaseVelocity(
                bird_id,
                linearVelocity=[0, 0, -5],  # Downward velocity in m/s
                angularVelocity=[1, 1, 1],  # Rotational velocity in rad/s
                physicsClientId=self.cli
            )
            
            # Mark bird entity as collision-inactive
            for bird in self.birds:
                if bird['id'] == bird_id:
                    bird['collision_hit'] = True
                    break
                    
        except Exception:
            pass  # Gracefully handle physics simulation errors


class WindSystem:
    """
    Advanced atmospheric wind simulation system for drone environment.
    
    Features:
    - Dynamic wind direction changes
    - Realistic wind speed variations
    - Turbulence simulation
    - Height-based wind effects
    - Deterministic behavior with seed
    
    Designed for realistic atmospheric conditions in drone simulation.
    """
    
    def __init__(self, rng: random.Random):
        """
        Initialize the wind simulation system with atmospheric parameters.
        
        Args:
            rng: Seeded random number generator for deterministic wind behavior
        """
        self.rng = rng
        self.timer = 0.0
        
        # Initialize wind parameters
        self.wind_speed = rng.uniform(WIND_SPEED_MIN, WIND_SPEED_MAX)
        self.wind_direction = rng.uniform(0, 2 * math.pi)  # Random initial direction
        self.wind_vector = np.array([
            self.wind_speed * math.cos(self.wind_direction),
            self.wind_speed * math.sin(self.wind_direction),
            0.0  # Minimal vertical wind component
        ])
        
        # Turbulence parameters
        self.turbulence_timer = 0.0
        self.turbulence_vector = np.zeros(3)
        
        # Wind change parameters
        self.direction_change_timer = 0.0
        self.target_direction = self.wind_direction
        self.direction_transition_speed = 0.1  # Radians per second
        
    def update(self, dt: float):
        """
        Update wind conditions for current simulation frame.
        
        Args:
            dt: Time delta for physics integration
        """
        self.timer += dt
        self.direction_change_timer += dt
        self.turbulence_timer += dt
        
        # Update wind direction changes
        if self.direction_change_timer >= WIND_DIRECTION_CHANGE_INTERVAL:
            self._change_wind_direction()
            self.direction_change_timer = 0.0
        
        # Smoothly transition to target direction
        direction_diff = self.target_direction - self.wind_direction
        if abs(direction_diff) > 0.01:
            # Normalize angle difference
            if direction_diff > math.pi:
                direction_diff -= 2 * math.pi
            elif direction_diff < -math.pi:
                direction_diff += 2 * math.pi
            
            # Apply smooth transition
            self.wind_direction += direction_diff * self.direction_transition_speed * dt
            self.wind_direction = self.wind_direction % (2 * math.pi)
        
        # Update turbulence
        if self.turbulence_timer >= 0.5:  # Update turbulence every 0.5 seconds
            self._update_turbulence()
            self.turbulence_timer = 0.0
        
        # Update main wind vector
        self.wind_vector = np.array([
            self.wind_speed * math.cos(self.wind_direction),
            self.wind_speed * math.sin(self.wind_direction),
            0.0
        ])
    
    def get_wind_force(self, position: np.ndarray) -> np.ndarray:
        """
        Calculate physically accurate wind force at specific position.
        
        Uses the aerodynamic drag equation: F = 0.5 × ρ × A × v² × Cd
        where:
        - ρ (rho) = air density (1.225 kg/m³)
        - A = drone cross-sectional area (0.05 m²)
        - v = wind velocity magnitude
        - Cd = drag coefficient (~1.0 for drone)
        
        Args:
            position: 3D position in simulation space
            
        Returns:
            Wind force vector in 3D space (Newtons)
        """
        # Atmospheric parameters for realistic wind force calculation
        rho = 1.225  # Air density at sea level (kg/m³)
        A = 0.02     # Reduced drone cross-sectional area (m²) - more realistic for small drone
        Cd = 0.3     # Reduced drag coefficient for streamlined quadrotor
        
        # Calculate stable wind force (reduced turbulence effects)
        wind_velocity = self.wind_vector.copy()  # Use only base wind, minimal turbulence
        wind_speed = np.linalg.norm(wind_velocity)
        
        # Apply simplified and stable wind force calculation
        if wind_speed > 0.01:  # Avoid division by zero
            # Use simplified force calculation for stability
            force_magnitude = 0.5 * rho * A * wind_speed**2 * Cd
            
            # Apply additional scaling factor for micro-drone sensitivity
            force_magnitude *= 0.3  # Further reduce force for stability
            
            # Apply force in wind direction
            wind_force = (wind_velocity / wind_speed) * force_magnitude
        else:
            wind_force = np.zeros(3)
        
        # Apply height-based wind effects (linear scaling with altitude)
        # Force increases linearly with height: 0.01N at 1m, 0.02N at 2m, etc.
        height_factor = position[2] * 1.0  # Reduced scale factor for gentler wind forces
        wind_force *= height_factor
        
        # Apply maximum force limit for stability
        max_force = 0.1  # Maximum wind force in Newtons
        force_magnitude = np.linalg.norm(wind_force)
        if force_magnitude > max_force:
            wind_force = (wind_force / force_magnitude) * max_force
        
        return wind_force
    
    def _change_wind_direction(self):
        """Generate new random wind direction."""
        # Random direction change (can be any direction)
        angle_change = self.rng.uniform(-math.pi/2, math.pi/2)  # ±90 degrees
        self.target_direction = (self.wind_direction + angle_change) % (2 * math.pi)
        
        # Random speed variation
        speed_change = self.rng.uniform(-0.5, 0.5)
        self.wind_speed = max(WIND_SPEED_MIN, min(WIND_SPEED_MAX, self.wind_speed + speed_change))
    
    def _update_turbulence(self):
        """Update turbulence vector for stable atmospheric effects."""
        # Generate minimal turbulence for stability
        self.turbulence_vector = np.array([
            self.rng.uniform(-0.1, 0.1),  
            self.rng.uniform(-0.1, 0.1), 
            self.rng.uniform(-0.05, 0.05)  
        ]) * self.wind_speed * 0.02  
    
    def get_wind_info(self) -> dict:
        """
        Get current wind information for debugging or display.
        
        Returns:
            Dictionary containing wind parameters
        """
        return {
            'speed': self.wind_speed,
            'direction_degrees': math.degrees(self.wind_direction),
            'vector': self.wind_vector.tolist(),
            'turbulence': self.turbulence_vector.tolist()
        }