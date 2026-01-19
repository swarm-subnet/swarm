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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, List

import pybullet as p

from swarm.constants import (
    LANDING_PLATFORM_RADIUS,
    PLATFORM,
    MAX_ATTEMPTS_PER_OBS,
    START_PLATFORM,
    START_PLATFORM_RADIUS,
    START_PLATFORM_HEIGHT,
    START_PLATFORM_SURFACE_Z,
    START_PLATFORM_RANDOMIZE,
    TYPE_1_N_BUILDINGS, TYPE_1_HEIGHT_SCALE, TYPE_1_SAFE_ZONE, TYPE_1_WORLD_RANGE,
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


@dataclass
class RoadSegment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    width: float
    road_type: str

    @property
    def length(self) -> float:
        return math.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1])

    @property
    def angle(self) -> float:
        return math.atan2(self.end[1] - self.start[1], self.end[0] - self.start[0])

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.start[0] + self.end[0]) / 2, (self.start[1] + self.end[1]) / 2)


@dataclass
class Intersection:
    center: Tuple[float, float]
    radius: float
    connected_roads: List[int] = field(default_factory=list)


# =============================================================================
# CITY GENERATION PARAMETERS (Type 1 - City Navigation)
# =============================================================================

# Road network generation
CITY_NUM_PRIMARY_ROADS_MIN = 3
CITY_NUM_PRIMARY_ROADS_MAX = 5
CITY_PRIMARY_ROAD_WIDTH = 4.5
CITY_SECONDARY_ROAD_WIDTH = 3.5
CITY_ROAD_CURVE_PROBABILITY = 0.3
CITY_SECONDARY_BRANCH_INTERVAL_MIN = 8.0
CITY_SECONDARY_BRANCH_INTERVAL_MAX = 15.0

# Road marking parameters
CITY_CENTER_LINE_WIDTH = 0.10
CITY_CENTER_LINE_DASH_LENGTH = 2.0
CITY_CENTER_LINE_GAP_LENGTH = 2.0
CITY_EDGE_LINE_WIDTH = 0.12
CITY_STOP_LINE_WIDTH = 0.40

# Building parameters
CITY_BUILDING_SETBACK = 2.5
CITY_BUILDING_MIN_DIST = 1.0
CITY_DOWNTOWN_RADIUS = 15.0
CITY_MIDTOWN_RADIUS = 25.0
CITY_BUILDING_MIN_HEIGHT = 4.0
CITY_BUILDING_MAX_HEIGHT = 18.0
CITY_BUILDING_MIN_FOOTPRINT = 1.5
CITY_BUILDING_MAX_FOOTPRINT = 3.5

# Tree parameters
CITY_TREE_SPACING = 6.0
CITY_TREE_SIDEWALK_OFFSET = 2.0
CITY_TREE_MIN_DIST = 3.0
CITY_TREE_INTERSECTION_CLEARANCE = 4.0
CITY_TREE_HEIGHT_MIN = 2.0
CITY_TREE_HEIGHT_MAX = 4.0
CITY_TREE_RADIUS = 0.25

# Streetlight parameters
CITY_LIGHT_SPACING = 15.0
CITY_LIGHT_MIN_DIST = 8.0
CITY_LIGHT_TREE_CLEARANCE = 2.5
CITY_STREETLIGHT_HEIGHT = 3.5
CITY_STREETLIGHT_RADIUS = 0.04

# Road generation parameters
CITY_SECONDARY_ROAD_SPACING = 18.0
CITY_INTERSECTION_MERGE_DISTANCE = 6.0
CITY_YELLOW_LINE_MARGIN = 2.0

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


def _add_rooftop_details(cli: int, x: float, y: float, z: float, w: float, d: float, rng) -> None:
    """Add professional rooftop mechanicals: HVAC, water towers, and elevator penthouses."""
    METAL_SPECULAR = [0.8, 0.8, 0.8]
    HVAC_COLOR = [0.6, 0.6, 0.65, 1.0]
    WATER_TOWER_WOOD = [0.45, 0.35, 0.25, 1.0]
    WATER_TOWER_METAL = [0.35, 0.35, 0.40, 1.0]
    PENTHOUSE_COLOR = [0.75, 0.75, 0.75, 1.0]

    # 1. Elevator Penthouse / Stairwell Access
    ph_w, ph_d, ph_h = w * 0.35, d * 0.35, 1.1
    ph_x = x + rng.uniform(-w*0.1, w*0.1)
    ph_y = y + rng.uniform(-d*0.1, d*0.1)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[ph_w/2, ph_d/2, ph_h/2],
                                rgbaColor=PENTHOUSE_COLOR, specularColor=[0.2, 0.2, 0.2], physicsClientId=cli)
    p.createMultiBody(0, -1, vis, basePosition=[ph_x, ph_y, z + ph_h/2], physicsClientId=cli)

    # 2. HVAC / AC Units
    num_hvac = rng.randint(2, 4)
    for _ in range(num_hvac):
        hw, hd, hh = 0.5, 0.5, 0.45
        hx = x + rng.uniform(-w*0.38, w*0.38)
        hy = y + rng.uniform(-d*0.38, d*0.38)

        # Avoid overlapping with penthouse
        if abs(hx - ph_x) < (hw+ph_w)/2 + 0.1 and abs(hy - ph_y) < (hd+ph_d)/2 + 0.1:
            continue

        # HVAC Base/Frame
        base_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hw/2 + 0.05, hd/2 + 0.05, 0.02],
                                        rgbaColor=[0.2, 0.2, 0.2, 1.0], physicsClientId=cli)
        p.createMultiBody(0, -1, base_vis, basePosition=[hx, hy, z + 0.02], physicsClientId=cli)

        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hw/2, hd/2, hh/2],
                                    rgbaColor=HVAC_COLOR, specularColor=METAL_SPECULAR, physicsClientId=cli)
        p.createMultiBody(0, -1, vis, basePosition=[hx, hy, z + 0.02 + hh/2], physicsClientId=cli)

        # Fan Grill and Hub
        fan_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.18, length=0.02,
                                       rgbaColor=[0.1, 0.1, 0.1, 1.0], physicsClientId=cli)
        p.createMultiBody(0, -1, fan_vis, basePosition=[hx, hy, z + 0.02 + hh + 0.01], physicsClientId=cli)
        hub_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[0.4, 0.4, 0.4, 1.0], physicsClientId=cli)
        p.createMultiBody(0, -1, hub_vis, basePosition=[hx, hy, z + 0.02 + hh + 0.02], physicsClientId=cli)

    # 3. Water Tower (Classic Urban Look)
    if rng.random() < 0.5:
        tw, th = 1.0, 1.4
        tx = x + rng.uniform(-w*0.3, w*0.3)
        ty = y + rng.uniform(-d*0.3, d*0.3)

        # Skip if overlapping with penthouse
        if abs(tx - ph_x) < (tw+ph_w)/2 + 0.2 and abs(ty - ph_y) < (tw+ph_d)/2 + 0.2:
            pass
        else:
            # Steel Leg Structure
            leg_h = 0.7
            leg_r = 0.04
            for lx, ly in [(-0.35, -0.35), (0.35, -0.35), (-0.35, 0.35), (0.35, 0.35)]:
                l_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=leg_r, length=leg_h,
                                             rgbaColor=WATER_TOWER_METAL, specularColor=METAL_SPECULAR, physicsClientId=cli)
                p.createMultiBody(0, -1, l_vis, basePosition=[tx+lx, ty+ly, z + leg_h/2], physicsClientId=cli)

            # Wooden Tank
            t_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=tw/2, length=th,
                                         rgbaColor=WATER_TOWER_WOOD, physicsClientId=cli)
            p.createMultiBody(0, -1, t_vis, basePosition=[tx, ty, z + leg_h + th/2], physicsClientId=cli)

            # Conical Cap
            c_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=tw/2 + 0.05, length=0.2,
                                         rgbaColor=[0.35, 0.3, 0.2, 1.0], physicsClientId=cli)
            p.createMultiBody(0, -1, c_vis, basePosition=[tx, ty, z + leg_h + th + 0.1], physicsClientId=cli)


def _add_city_building(cli: int, pos: List[float], size: List[float], rng) -> None:
    width, depth, height = size
    x, y, z_base = pos[0], pos[1], pos[2]

    GLASS_COLORS = [
        [0.22, 0.35, 0.45, 0.95], # Deep architectural blue
        [0.25, 0.38, 0.42, 0.94], # Muted teal
        [0.45, 0.48, 0.52, 0.92], # Modern grey-glass
        [0.18, 0.28, 0.38, 0.94], # Midnight blue
        [0.35, 0.42, 0.45, 0.94], # Steel glass
    ]
    CONCRETE_COLORS = [
        [0.85, 0.84, 0.82, 1.0], # Light stone
        [0.72, 0.70, 0.68, 1.0], # Weathered concrete
        [0.55, 0.53, 0.50, 1.0], # Darker industrial
        [0.68, 0.65, 0.62, 1.0], # Warm concrete
    ]
    BRICK_COLORS = [
        [0.55, 0.28, 0.22, 1.0], # Burnt sienna
        [0.48, 0.32, 0.25, 1.0], # Brown brick
        [0.62, 0.35, 0.25, 1.0], # Reddish clay
        [0.42, 0.25, 0.18, 1.0], # Old industrial
        [0.38, 0.22, 0.18, 1.0], # Deep charcoal brick
    ]
    WINDOW_COLOR = [0.25, 0.45, 0.65, 0.90]
    ROOF_DARK = [0.18, 0.18, 0.20, 1.0]
    DOOR_COLOR = [0.20, 0.12, 0.08, 1.0]
    METAL_SPECULAR = [0.7, 0.7, 0.7]
    AWNING_COLORS = [
        [0.70, 0.15, 0.15, 1.0],
        [0.15, 0.50, 0.15, 1.0],
        [0.15, 0.15, 0.60, 1.0],
        [0.60, 0.40, 0.10, 1.0],
    ]

    building_type = rng.choice(["skyscraper", "office", "residential", "stepped", "modern_tower"])

    if building_type == "skyscraper" and height > 8:
        glass = rng.choice(GLASS_COLORS)
        lobby_h = min(1.5, height * 0.10)
        lobby_w, lobby_d = width * 1.08, depth * 1.08

        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[lobby_w/2, lobby_d/2, lobby_h/2], physicsClientId=cli)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[lobby_w/2, lobby_d/2, lobby_h/2],
                                  rgbaColor=CONCRETE_COLORS[0], specularColor=[0.5, 0.5, 0.5], physicsClientId=cli)
        p.createMultiBody(0, col, vis, basePosition=[x, y, z_base + lobby_h/2], physicsClientId=cli)

        main_h = height * 0.80
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[width/2, depth/2, main_h/2], physicsClientId=cli)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2, depth/2, main_h/2],
                                  rgbaColor=glass, specularColor=[0.8, 0.8, 0.8], physicsClientId=cli)
        p.createMultiBody(0, col, vis, basePosition=[x, y, z_base + lobby_h + main_h/2], physicsClientId=cli)

        floor_h = 2.0
        num_floors = int(main_h / floor_h)
        for i in range(num_floors + 1):
            fz = z_base + lobby_h + i * floor_h
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2 + 0.02, depth/2 + 0.02, 0.03],
                                      rgbaColor=CONCRETE_COLORS[2], physicsClientId=cli)
            p.createMultiBody(0, -1, vis, basePosition=[x, y, fz], physicsClientId=cli)

        top_h = height * 0.08
        top_w, top_d = width * 0.7, depth * 0.7
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[top_w/2, top_d/2, top_h/2],
                                  rgbaColor=[glass[0]*0.85, glass[1]*0.85, glass[2]*0.85, 1.0], physicsClientId=cli)
        p.createMultiBody(0, -1, vis, basePosition=[x, y, z_base + lobby_h + main_h + top_h/2], physicsClientId=cli)

        roof_z = z_base + lobby_h + main_h + top_h
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[top_w/2 + 0.03, top_d/2 + 0.03, 0.05],
                                  rgbaColor=ROOF_DARK, physicsClientId=cli)
        p.createMultiBody(0, -1, vis, basePosition=[x, y, roof_z + 0.05], physicsClientId=cli)

        _add_rooftop_details(cli, x, y, roof_z + 0.1, top_w, top_d, rng)

        antenna_h = min(1.0, height * 0.05)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.03, length=antenna_h,
                                  rgbaColor=[0.50, 0.50, 0.55, 1.0], physicsClientId=cli)
        p.createMultiBody(0, -1, vis, basePosition=[x, y, roof_z + 0.1 + antenna_h/2], physicsClientId=cli)

    elif building_type == "stepped" and height > 6:
        glass = rng.choice(GLASS_COLORS)
        num_steps = rng.randint(3, 5)
        step_h = height / num_steps
        z = z_base
        for i in range(num_steps):
            scale = 1 - i * 0.12
            sw, sd = width * scale, depth * scale
            color = glass if i % 2 == 0 else CONCRETE_COLORS[1]
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sw/2, sd/2, step_h/2], physicsClientId=cli)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sw/2, sd/2, step_h/2],
                                      rgbaColor=color, specularColor=[0.5, 0.5, 0.5], physicsClientId=cli)
            p.createMultiBody(0, col, vis, basePosition=[x, y, z + step_h/2], physicsClientId=cli)
            if i < num_steps - 1:
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sw/2 + 0.03, sd/2 + 0.03, 0.04],
                                          rgbaColor=CONCRETE_COLORS[2], physicsClientId=cli)
                p.createMultiBody(0, -1, vis, basePosition=[x, y, z + step_h], physicsClientId=cli)
            z += step_h
        _add_rooftop_details(cli, x, y, z, width * (1 - (num_steps-1) * 0.12), depth * (1 - (num_steps-1) * 0.12), rng)

    elif building_type == "office":
        glass = rng.choice(GLASS_COLORS)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[width/2, depth/2, height/2], physicsClientId=cli)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2, depth/2, height/2],
                                  rgbaColor=glass, specularColor=[0.7, 0.7, 0.7], physicsClientId=cli)
        p.createMultiBody(0, col, vis, basePosition=[x, y, z_base + height/2], physicsClientId=cli)

        floor_h = 1.5
        num_floors = int(height / floor_h)
        for i in range(num_floors + 1):
            fz = z_base + i * floor_h
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2 + 0.02, depth/2 + 0.02, 0.03],
                                      rgbaColor=CONCRETE_COLORS[2], physicsClientId=cli)
            p.createMultiBody(0, -1, vis, basePosition=[x, y, fz], physicsClientId=cli)

        roof_h = 0.1
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2 + 0.05, depth/2 + 0.05, roof_h/2],
                                  rgbaColor=ROOF_DARK, physicsClientId=cli)
        p.createMultiBody(0, -1, vis, basePosition=[x, y, z_base + height + roof_h/2], physicsClientId=cli)

        _add_rooftop_details(cli, x, y, z_base + height + roof_h, width, depth, rng)

    elif building_type == "modern_tower" and height > 5:
        glass = rng.choice(GLASS_COLORS)
        accent = rng.choice([[0.85, 0.45, 0.12, 1.0], [0.12, 0.55, 0.45, 1.0], [0.55, 0.12, 0.35, 1.0]])

        base_h = min(2.0, height * 0.15)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[width/2 * 1.1, depth/2 * 1.1, base_h/2], physicsClientId=cli)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2 * 1.1, depth/2 * 1.1, base_h/2],
                                  rgbaColor=CONCRETE_COLORS[0], physicsClientId=cli)
        p.createMultiBody(0, col, vis, basePosition=[x, y, z_base + base_h/2], physicsClientId=cli)

        main_h = height * 0.75
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[width/2, depth/2, main_h/2], physicsClientId=cli)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2, depth/2, main_h/2],
                                  rgbaColor=glass, specularColor=[0.9, 0.9, 0.9], physicsClientId=cli)
        p.createMultiBody(0, col, vis, basePosition=[x, y, z_base + base_h + main_h/2], physicsClientId=cli)

        stripe_h = 0.15
        num_stripes = 4
        for i in range(num_stripes):
            sz = z_base + base_h + main_h * (i + 1) / (num_stripes + 1)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2 + 0.03, depth/2 + 0.03, stripe_h/2],
                                      rgbaColor=accent, physicsClientId=cli)
            p.createMultiBody(0, -1, vis, basePosition=[x, y, sz], physicsClientId=cli)

        crown_h = height * 0.08
        crown_w = width * 0.6
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[crown_w/2, crown_w/2, crown_h/2],
                                  rgbaColor=accent, physicsClientId=cli)
        p.createMultiBody(0, -1, vis, basePosition=[x, y, z_base + base_h + main_h + crown_h/2], physicsClientId=cli)

        spire_h = height * 0.12
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=spire_h,
                                  rgbaColor=[0.7, 0.7, 0.75, 1.0], physicsClientId=cli)
        p.createMultiBody(0, -1, vis, basePosition=[x, y, z_base + base_h + main_h + crown_h + spire_h/2], physicsClientId=cli)

        roof_z = z_base + base_h + main_h
        # Add details on the main roof shoulders if the crown is smaller
        if crown_w < width * 0.8:
            _add_rooftop_details(cli, x, y, roof_z + 0.05, width, depth, rng)

    else:
        brick = rng.choice(BRICK_COLORS)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[width/2, depth/2, height/2], physicsClientId=cli)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2, depth/2, height/2],
                                  rgbaColor=brick, specularColor=[0.1, 0.1, 0.1], physicsClientId=cli)
        p.createMultiBody(0, col, vis, basePosition=[x, y, z_base + height/2], physicsClientId=cli)

        floor_h = 1.2
        num_floors = int(height / floor_h)
        win_w, win_h = 0.25, 0.35
        for floor in range(num_floors):
            fz = z_base + floor * floor_h + floor_h * 0.6
            for side, (dx, dy, rot) in enumerate([
                (width/2 + 0.01, 0, 0), (-width/2 - 0.01, 0, 0),
                (0, depth/2 + 0.01, math.pi/2), (0, -depth/2 - 0.01, math.pi/2)
            ]):
                face_len = depth if side < 2 else width
                num_win = max(2, int(face_len / 0.8))
                spacing = face_len / (num_win + 1)
                for w in range(num_win):
                    if side < 2:
                        wy = y + (w - num_win/2 + 0.5) * spacing
                        wx = x + dx
                    else:
                        wx = x + (w - num_win/2 + 0.5) * spacing
                        wy = y + dy
                    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.01, win_w/2, win_h/2],
                                              rgbaColor=WINDOW_COLOR, physicsClientId=cli)
                    p.createMultiBody(0, -1, vis, basePosition=[wx, wy, fz],
                                      baseOrientation=p.getQuaternionFromEuler([0, 0, rot]), physicsClientId=cli)

        roof_h = 0.12
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[width/2 + 0.06, depth/2 + 0.06, roof_h/2],
                                  rgbaColor=ROOF_DARK, physicsClientId=cli)
        p.createMultiBody(0, -1, vis, basePosition=[x, y, z_base + height + roof_h/2], physicsClientId=cli)

        _add_rooftop_details(cli, x, y, z_base + height + roof_h, width, depth, rng)

    if building_type not in ["stepped"] and rng.random() < 0.7:
        door_w, door_h = 0.4, 0.7
        door_side = rng.choice([(width/2 + 0.02, 0), (-width/2 - 0.02, 0), (0, depth/2 + 0.02), (0, -depth/2 - 0.02)])
        door_x = x + door_side[0]
        door_y = y + door_side[1]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.02, door_w/2, door_h/2],
                                  rgbaColor=DOOR_COLOR, physicsClientId=cli)
        rot = math.pi/2 if door_side[1] != 0 else 0
        p.createMultiBody(0, -1, vis, basePosition=[door_x, door_y, z_base + door_h/2],
                          baseOrientation=p.getQuaternionFromEuler([0, 0, rot]), physicsClientId=cli)

        if rng.random() < 0.5:
            awning_color = rng.choice(AWNING_COLORS)
            aw_w, aw_d, aw_h = door_w * 1.5, 0.4, 0.08
            aw_z = door_h + 0.1
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[aw_d/2, aw_w/2, aw_h/2],
                                      rgbaColor=awning_color, physicsClientId=cli)
            aw_x = door_x + (0.2 if door_side[0] > 0 else -0.2 if door_side[0] < 0 else 0)
            aw_y = door_y + (0.2 if door_side[1] > 0 else -0.2 if door_side[1] < 0 else 0)
            p.createMultiBody(0, -1, vis, basePosition=[aw_x, aw_y, aw_z],
                              baseOrientation=p.getQuaternionFromEuler([0, 0, rot]), physicsClientId=cli)


def _add_city_tree(cli: int, pos: List[float], height: float, radius: float, rng) -> None:
    tree_type = rng.choice(["deciduous", "conifer", "palm"])
    TRUNK_COLOR = [0.38, 0.28, 0.18, 1.0]
    GRATE_COLOR = [0.25, 0.25, 0.25, 1.0]
    z_base = pos[2]

    # Tree Grate at base
    grate_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.6, 0.6, 0.005], rgbaColor=GRATE_COLOR, physicsClientId=cli)
    p.createMultiBody(0, -1, grate_vis, basePosition=[pos[0], pos[1], z_base + 0.001], physicsClientId=cli)

    foliage_colors = [
        [0.15, 0.45, 0.15, 1.0],
        [0.20, 0.50, 0.20, 1.0],
        [0.10, 0.40, 0.10, 1.0],
    ]
    foliage_color = rng.choice(foliage_colors)

    if tree_type == "deciduous":
        # Tapered trunk (two segments)
        trunk_h1 = height * 0.25
        trunk_h2 = height * 0.15
        t_vis1 = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=trunk_h1, rgbaColor=TRUNK_COLOR, physicsClientId=cli)
        t_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=trunk_h1 + trunk_h2, physicsClientId=cli)
        p.createMultiBody(0, t_col, t_vis1, basePosition=[pos[0], pos[1], z_base + trunk_h1/2], physicsClientId=cli)

        t_vis2 = p.createVisualShape(p.GEOM_CYLINDER, radius=radius*0.7, length=trunk_h2, rgbaColor=TRUNK_COLOR, physicsClientId=cli)
        p.createMultiBody(0, -1, t_vis2, basePosition=[pos[0], pos[1], z_base + trunk_h1 + trunk_h2/2], physicsClientId=cli)

        # Foliage clusters
        f_radius = height * 0.40
        trunk_total = trunk_h1 + trunk_h2
        clusters = [
            (0, 0, z_base + trunk_total + f_radius*0.7, 1.0),
            (f_radius*0.5, 0, z_base + trunk_total + f_radius*0.4, 0.8),
            (-f_radius*0.4, f_radius*0.3, z_base + trunk_total + f_radius*0.5, 0.75),
            (f_radius*0.3, -f_radius*0.4, z_base + trunk_total + f_radius*0.3, 0.8),
            (-f_radius*0.5, -f_radius*0.2, z_base + trunk_total + f_radius*0.6, 0.7),
            (0, f_radius*0.5, z_base + trunk_total + f_radius*0.4, 0.75),
            (0, 0, z_base + trunk_total + f_radius*1.1, 0.65), # Top cap
        ]
        for dx, dy, dz, r_mult in clusters:
            # Vary color slightly per cluster for depth
            c_var = rng.uniform(0.95, 1.05)
            c = [foliage_color[0]*c_var, foliage_color[1]*c_var, foliage_color[2]*c_var, 1.0]
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=f_radius * r_mult, rgbaColor=c, physicsClientId=cli)
            p.createMultiBody(0, -1, vis, basePosition=[pos[0]+dx, pos[1]+dy, dz], physicsClientId=cli)

    elif tree_type == "conifer":
        trunk_h = height * 0.15
        t_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius * 0.8, length=trunk_h, rgbaColor=TRUNK_COLOR, physicsClientId=cli)
        t_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius * 0.8, height=trunk_h, physicsClientId=cli)
        p.createMultiBody(0, t_col, t_vis, basePosition=[pos[0], pos[1], z_base + trunk_h/2], physicsClientId=cli)

        levels = 7
        for i in range(levels):
            layer_h = (height - trunk_h) / levels
            layer_r = radius * 4.5 * (1.0 - (i/levels)**0.8) # Non-linear taper
            layer_z = z_base + trunk_h + i * layer_h * 0.75 + layer_h/2
            # Darker at bottom, lighter at top
            c_mult = 0.8 + (i/levels) * 0.3
            color = [foliage_color[0]*c_mult, foliage_color[1]*c_mult, foliage_color[2]*c_mult, 1.0]
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=layer_r, length=layer_h * 1.2, rgbaColor=color, physicsClientId=cli)
            p.createMultiBody(0, -1, vis, basePosition=[pos[0], pos[1], layer_z], physicsClientId=cli)

    else:  # Palm
        trunk_h = height * 0.9
        # Slightly tapered/curved trunk look using two segments
        t_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius * 0.5, length=trunk_h, rgbaColor=[0.45, 0.38, 0.3, 1.0], physicsClientId=cli)
        t_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius * 0.5, height=trunk_h, physicsClientId=cli)
        p.createMultiBody(0, t_col, t_vis, basePosition=[pos[0], pos[1], z_base + trunk_h/2], physicsClientId=cli)

        # Dry frond skirt
        skirt_color = [0.4, 0.35, 0.25, 1.0]
        for i in range(8):
            angle = (i / 8) * 2 * math.pi
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.3, 0.1, 0.01], rgbaColor=skirt_color, physicsClientId=cli)
            quat = p.getQuaternionFromEuler([1.2, 0, angle])
            p.createMultiBody(0, -1, vis, basePosition=[pos[0], pos[1], z_base + trunk_h - 0.2], baseOrientation=quat, physicsClientId=cli)

        palm_green = [0.12, 0.45, 0.12, 1.0]
        num_fronds = 12
        for i in range(num_fronds):
            angle = (i / num_fronds) * 2 * math.pi
            f_len = height * 0.5
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[f_len/2, 0.15, 0.01], rgbaColor=palm_green, physicsClientId=cli)
            # Tilt and rotate
            quat = p.getQuaternionFromEuler([0.4 + rng.uniform(-0.1, 0.1), 0, angle])
            fx = pos[0] + math.cos(angle) * f_len/2.5
            fy = pos[1] + math.sin(angle) * f_len/2.5
            p.createMultiBody(0, -1, vis, basePosition=[fx, fy, z_base + trunk_h], baseOrientation=quat, physicsClientId=cli)


def _add_city_streetlight(cli: int, pos: List[float], height: float, radius: float, arm_dir: List[int], rng) -> None:
    LAMP_POST = [0.22, 0.22, 0.25, 1.0]
    LAMP_LIGHT = [1.0, 0.98, 0.85, 1.0] # Warmer LED look
    HOUSING_COLOR = [0.30, 0.30, 0.32, 1.0]
    GLASS_COLOR = [0.8, 0.9, 1.0, 0.4] # Semi-transparent glass
    METAL_SPECULAR = [0.8, 0.8, 0.9]
    z_base = pos[2]

    # Base of the pole (reinforced footing)
    base_h = 0.6
    base_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius * 1.8, length=base_h,
                                    rgbaColor=LAMP_POST, specularColor=[0.4, 0.4, 0.4], physicsClientId=cli)
    p.createMultiBody(0, -1, base_vis, basePosition=[pos[0], pos[1], z_base + base_h/2], physicsClientId=cli)

    # Main pole (tapered look with two segments)
    pole_h1 = height * 0.7
    vis1 = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=pole_h1,
                                rgbaColor=LAMP_POST, specularColor=METAL_SPECULAR, physicsClientId=cli)
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height, physicsClientId=cli)
    p.createMultiBody(0, col, vis1, basePosition=[pos[0], pos[1], z_base + pole_h1/2], physicsClientId=cli)

    pole_h2 = height * 0.3
    vis2 = p.createVisualShape(p.GEOM_CYLINDER, radius=radius * 0.7, length=pole_h2,
                                rgbaColor=LAMP_POST, specularColor=METAL_SPECULAR, physicsClientId=cli)
    p.createMultiBody(0, -1, vis2, basePosition=[pos[0], pos[1], z_base + pole_h1 + pole_h2/2], physicsClientId=cli)

    # Mounting bracket
    bracket_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[radius*1.2, radius*1.2, 0.1],
                                      rgbaColor=HOUSING_COLOR, specularColor=METAL_SPECULAR, physicsClientId=cli)
    p.createMultiBody(0, -1, bracket_vis, basePosition=[pos[0], pos[1], z_base + height - 0.1], physicsClientId=cli)

    arm_len = 1.0
    arm_r = radius * 0.5
    arm_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=arm_r, length=arm_len,
                                   rgbaColor=LAMP_POST, specularColor=METAL_SPECULAR, physicsClientId=cli)

    arm_x = pos[0] + arm_dir[0] * arm_len/2
    arm_y = pos[1] + arm_dir[1] * arm_len/2

    if arm_dir[0] != 0:
        quat = p.getQuaternionFromEuler([0, 1.57, 0])
    else:
        quat = p.getQuaternionFromEuler([1.57, 0, 0])
    p.createMultiBody(0, -1, arm_vis, basePosition=[arm_x, arm_y, z_base + height - 0.1], baseOrientation=quat, physicsClientId=cli)

    # Modern housing (rectangular box)
    housing_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.22, 0.10, 0.04],
                                       rgbaColor=HOUSING_COLOR, specularColor=METAL_SPECULAR, physicsClientId=cli)
    housing_x = pos[0] + arm_dir[0] * arm_len
    housing_y = pos[1] + arm_dir[1] * arm_len

    # Orientation for the housing box
    h_quat = p.getQuaternionFromEuler([0, 0, 0]) if arm_dir[0] != 0 else p.getQuaternionFromEuler([0, 0, 1.57])
    p.createMultiBody(0, -1, housing_vis, basePosition=[housing_x, housing_y, z_base + height - 0.14], baseOrientation=h_quat, physicsClientId=cli)

    # Glass lens
    lens_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.18, 0.08, 0.01],
                                    rgbaColor=GLASS_COLOR, specularColor=[1.0, 1.0, 1.0], physicsClientId=cli)
    p.createMultiBody(0, -1, lens_vis, basePosition=[housing_x, housing_y, z_base + height - 0.18], baseOrientation=h_quat, physicsClientId=cli)

    # Light source (bulb)
    bulb_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.06, rgbaColor=LAMP_LIGHT, physicsClientId=cli)
    p.createMultiBody(0, -1, bulb_vis, basePosition=[housing_x, housing_y, z_base + height - 0.17], physicsClientId=cli)

    # Lamp glow
    glow_color = [LAMP_LIGHT[0], LAMP_LIGHT[1], LAMP_LIGHT[2], 0.15]
    glow_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.4, rgbaColor=glow_color, physicsClientId=cli)
    # Position glow slightly below the lens
    p.createMultiBody(0, -1, glow_vis, basePosition=[housing_x, housing_y, z_base + height - 0.3], physicsClientId=cli)


def _add_traffic_light(cli: int, pos: List[float], facing: str, rng) -> None:
    POLE_COLOR = [0.25, 0.25, 0.28, 1.0]
    HOUSING_COLOR = [0.18, 0.18, 0.20, 1.0]
    METAL_SPECULAR = [0.7, 0.7, 0.8]
    RED = [0.85, 0.15, 0.10, 1.0]
    YELLOW = [0.90, 0.80, 0.15, 1.0]
    GREEN = [0.15, 0.75, 0.20, 1.0]
    z_base = pos[2]

    # Pole
    pole_h = 3.2
    pole_r = 0.06
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=pole_r, length=pole_h,
                               rgbaColor=POLE_COLOR, specularColor=METAL_SPECULAR, physicsClientId=cli)
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=pole_r, height=pole_h, physicsClientId=cli)
    p.createMultiBody(0, col, vis, basePosition=[pos[0], pos[1], z_base + pole_h/2], physicsClientId=cli)

    # Base control box
    box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.12, 0.12, 0.4],
                                   rgbaColor=POLE_COLOR, specularColor=METAL_SPECULAR, physicsClientId=cli)
    p.createMultiBody(0, -1, box_vis, basePosition=[pos[0], pos[1], z_base + 0.4], physicsClientId=cli)

    # Concrete footing
    footing_h = 0.2
    footing_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.18, 0.18, footing_h/2],
                                       rgbaColor=[0.7, 0.7, 0.7, 1.0], specularColor=[0.2, 0.2, 0.2], physicsClientId=cli)
    p.createMultiBody(0, -1, footing_vis, basePosition=[pos[0], pos[1], z_base + footing_h/2], physicsClientId=cli)

    arm_len = 1.4
    arm_r = 0.04
    if facing in ['x', '-x']:
        arm_quat = p.getQuaternionFromEuler([0, 1.57, 0])
        arm_dx = (1 if facing == 'x' else -1) * arm_len/2
        arm_dy = 0
    else:
        arm_quat = p.getQuaternionFromEuler([1.57, 0, 0])
        arm_dx = 0
        arm_dy = (1 if facing == 'y' else -1) * arm_len/2

    arm_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=arm_r, length=arm_len, rgbaColor=POLE_COLOR, physicsClientId=cli)
    p.createMultiBody(0, -1, arm_vis, basePosition=[pos[0] + arm_dx, pos[1] + arm_dy, z_base + pole_h - 0.1], baseOrientation=arm_quat, physicsClientId=cli)

    light_x = pos[0] + (1 if facing == 'x' else -1 if facing == '-x' else 0) * arm_len
    light_y = pos[1] + (1 if facing == 'y' else -1 if facing == '-y' else 0) * arm_len
    light_z = z_base + pole_h - 0.1

    # Housing
    box_h, box_w, box_d = 0.5, 0.18, 0.15
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[box_d/2, box_w/2, box_h/2], rgbaColor=HOUSING_COLOR, physicsClientId=cli)
    p.createMultiBody(0, -1, vis, basePosition=[light_x, light_y, light_z - box_h/2], physicsClientId=cli)

    light_r = 0.05
    active = rng.choice(['red', 'yellow', 'green'])
    for i, (color, name) in enumerate([(RED, 'red'), (YELLOW, 'yellow'), (GREEN, 'green')]):
        lz = light_z - 0.1 - i * 0.15
        brightness = 1.0 if name == active else 0.2
        actual_color = [color[0] * brightness, color[1] * brightness, color[2] * brightness, 1.0]

        vis = p.createVisualShape(p.GEOM_SPHERE, radius=light_r, rgbaColor=actual_color, physicsClientId=cli)
        p.createMultiBody(0, -1, vis, basePosition=[light_x, light_y, lz], physicsClientId=cli)

        # Simple visor for each light
        visor_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.01], rgbaColor=HOUSING_COLOR, physicsClientId=cli)
        v_quat = p.getQuaternionFromEuler([0, 0, 0]) if facing in ['x', '-x'] else p.getQuaternionFromEuler([0, 0, 1.57])
        p.createMultiBody(0, -1, visor_vis, basePosition=[light_x, light_y, lz + 0.06], baseOrientation=v_quat, physicsClientId=cli)


def _line_intersection(p1: Tuple[float, float], p2: Tuple[float, float],
                       p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-10:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t <= 1 and 0 <= u <= 1:
        ix = x1 + t * (x2 - x1)
        iy = y1 + t * (y2 - y1)
        return (ix, iy)
    return None


def _generate_primary_roads(rng, world_range: float) -> List[RoadSegment]:
    roads = []

    # Main N-S avenue (slight random offset from center)
    ns_offset = rng.uniform(-2, 2)
    roads.append(RoadSegment(
        (ns_offset, -world_range + 1),
        (ns_offset, world_range - 1),
        CITY_PRIMARY_ROAD_WIDTH, "primary"
    ))

    # Main E-W avenue (slight random offset from center)
    ew_offset = rng.uniform(-2, 2)
    roads.append(RoadSegment(
        (-world_range + 1, ew_offset),
        (world_range - 1, ew_offset),
        CITY_PRIMARY_ROAD_WIDTH, "primary"
    ))

    return roads


def _generate_secondary_roads(rng, primary_roads: List[RoadSegment],
                              world_range: float) -> List[RoadSegment]:
    secondary_roads = []

    # Parallel N-S roads (skip center where primary road exists)
    for x in range(-int(world_range) + 8, int(world_range) - 7, int(CITY_SECONDARY_ROAD_SPACING)):
        x_pos = x + rng.uniform(-1, 1)
        if abs(x_pos) < 16:
            continue

        secondary_roads.append(RoadSegment(
            (x_pos, -world_range + 2),
            (x_pos, world_range - 2),
            CITY_SECONDARY_ROAD_WIDTH, "secondary"
        ))

    # Parallel E-W roads (skip center where primary road exists)
    for y in range(-int(world_range) + 8, int(world_range) - 7, int(CITY_SECONDARY_ROAD_SPACING)):
        y_pos = y + rng.uniform(-1, 1)
        if abs(y_pos) < 16:
            continue

        secondary_roads.append(RoadSegment(
            (-world_range + 2, y_pos),
            (world_range - 2, y_pos),
            CITY_SECONDARY_ROAD_WIDTH, "secondary"
        ))

    return secondary_roads


def _find_intersections(roads: List[RoadSegment]) -> List[Intersection]:
    intersections = []
    intersection_points = []

    for i, road1 in enumerate(roads):
        for j, road2 in enumerate(roads):
            if i >= j:
                continue

            point = _line_intersection(road1.start, road1.end, road2.start, road2.end)
            if point is None:
                continue

            merged = False
            for k, existing in enumerate(intersection_points):
                if math.hypot(point[0] - existing[0], point[1] - existing[1]) < CITY_INTERSECTION_MERGE_DISTANCE:
                    new_center = ((existing[0] + point[0]) / 2, (existing[1] + point[1]) / 2)
                    intersection_points[k] = new_center
                    intersections[k].center = new_center
                    if i not in intersections[k].connected_roads:
                        intersections[k].connected_roads.append(i)
                    if j not in intersections[k].connected_roads:
                        intersections[k].connected_roads.append(j)
                    merged = True
                    break

            if not merged:
                max_width = max(road1.width, road2.width)
                radius = max_width * 0.6
                intersection_points.append(point)
                intersections.append(Intersection(point, radius, [i, j]))

    for i, road in enumerate(roads):
        for endpoint in [road.start, road.end]:
            for inter in intersections:
                if math.hypot(endpoint[0] - inter.center[0],
                            endpoint[1] - inter.center[1]) < inter.radius + 1.0:
                    if i not in inter.connected_roads:
                        inter.connected_roads.append(i)
                    break

    return intersections


def _render_road_segment(cli: int, road: RoadSegment, intersections: List[Intersection], rng) -> None:
    ROAD_ASPHALT = [0.15, 0.15, 0.17, 1.0]
    ROAD_SURFACE_Z = 0.08
    ROAD_THICKNESS = 0.10
    YELLOW_CENTER = [1.0, 0.85, 0.0, 1.0]
    WHITE_EDGE = [0.95, 0.95, 0.95, 1.0]

    length = road.length
    angle = road.angle
    width = road.width
    center = road.center
    quat = p.getQuaternionFromEuler([0, 0, angle])

    # Render full road surface (no segmentation)
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[length / 2, width / 2, ROAD_THICKNESS / 2],
        rgbaColor=ROAD_ASPHALT,
        specularColor=[0.08, 0.08, 0.08],
        physicsClientId=cli
    )
    p.createMultiBody(
        0, -1, vis,
        basePosition=[center[0], center[1], ROAD_SURFACE_Z - ROAD_THICKNESS / 2],
        baseOrientation=quat,
        physicsClientId=cli
    )

    # Calculate yellow line segments that stop before intersections
    stripe_z = ROAD_SURFACE_Z + 0.04
    yellow_width = 0.08  # Reduced from 0.12 for more realistic look
    yellow_height = 0.012

    # Find intersections along this road
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Road extends from start to end
    road_start = road.start
    road_end = road.end

    # Collect intersection points along this road
    intersection_distances = []
    for inter in intersections:
        # Project intersection center onto road line
        dx = inter.center[0] - road_start[0]
        dy = inter.center[1] - road_start[1]
        proj_dist = dx * cos_a + dy * sin_a

        # Check if intersection is actually on this road (not just near endpoints)
        perp_dist = abs(-dx * sin_a + dy * cos_a)
        if perp_dist < inter.radius + width / 2 and CITY_YELLOW_LINE_MARGIN < proj_dist < (length - CITY_YELLOW_LINE_MARGIN):
            # Use a smaller gap radius for cleaner look
            gap_radius = inter.radius + 0.8
            intersection_distances.append((proj_dist, gap_radius))

    # Sort by distance along road
    intersection_distances.sort(key=lambda x: x[0])

    # Create line segments between intersections
    segments = []
    current_start = 0.3  # Small offset from road start

    for inter_dist, gap_radius in intersection_distances:
        segment_end = inter_dist - gap_radius
        if segment_end > current_start + 1.0:  # Minimum visible segment
            segments.append((current_start, segment_end))
        current_start = inter_dist + gap_radius

    # Add final segment to end of road
    road_end_offset = length - 0.3  # Small offset from road end
    if road_end_offset > current_start + 1.0:
        segments.append((current_start, road_end_offset))

    # If no segments created (e.g., no intersections), draw full line with small margins
    if not segments:
        segments = [(0.3, length - 0.3)]

    # Render each yellow line segment
    for seg_start, seg_end in segments:
        seg_length = seg_end - seg_start
        seg_center_dist = (seg_start + seg_end) / 2.0

        # Calculate segment center position
        seg_center_x = road_start[0] + seg_center_dist * cos_a
        seg_center_y = road_start[1] + seg_center_dist * sin_a

        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[seg_length / 2, yellow_width, yellow_height],
            rgbaColor=YELLOW_CENTER,
            physicsClientId=cli
        )
        p.createMultiBody(
            0, -1, vis,
            basePosition=[seg_center_x, seg_center_y, stripe_z],
            baseOrientation=quat,
            physicsClientId=cli
        )

    # White edge lines (also continuous)
    # Position them at the edge between sidewalk and road
    edge_offset = width / 2 - 0.10
    edge_line_width = 0.08
    for side in [-1, 1]:
        offset_x = side * edge_offset * (-math.sin(angle))
        offset_y = side * edge_offset * math.cos(angle)

        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length / 2, edge_line_width / 2, 0.003],
            rgbaColor=WHITE_EDGE,
            physicsClientId=cli
        )
        p.createMultiBody(
            0, -1, vis,
            basePosition=[center[0] + offset_x, center[1] + offset_y, stripe_z],
            baseOrientation=quat,
            physicsClientId=cli
        )


def _render_intersection(cli: int, inter: Intersection, roads: List[RoadSegment], rng) -> None:
    ROAD_ASPHALT = [0.15, 0.15, 0.17, 1.0]
    ROAD_SURFACE_Z = 0.08

    vis = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=inter.radius + 0.5,
        length=0.02,
        rgbaColor=ROAD_ASPHALT,
        specularColor=[0.08, 0.08, 0.08],
        physicsClientId=cli
    )
    p.createMultiBody(
        0, -1, vis,
        basePosition=[inter.center[0], inter.center[1], ROAD_SURFACE_Z + 0.01],
        physicsClientId=cli
    )


def _generate_procedural_city(
    cli: int,
    rng,
    world_range: float,
    safe_zones: List[Tuple[float, float]],
    safe_zone_radius: float
) -> None:
    GRASS = [0.28, 0.55, 0.25, 1.0]
    SIDEWALK_COLOR = [0.75, 0.75, 0.75, 1.0]
    SIDEWALK_Z = 0.12

    grass_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[world_range + 5, world_range + 5, 0.02],
        rgbaColor=GRASS, physicsClientId=cli
    )
    p.createMultiBody(0, -1, grass_vis, basePosition=[0, 0, 0.02], physicsClientId=cli)

    primary_roads = _generate_primary_roads(rng, world_range)
    secondary_roads = _generate_secondary_roads(rng, primary_roads, world_range)

    all_roads = primary_roads + secondary_roads

    intersections = _find_intersections(all_roads)

    for road in all_roads:
        _render_road_segment(cli, road, intersections, rng)

    for inter in intersections:
        _render_intersection(cli, inter, all_roads, rng)

    for road in all_roads:
        angle = road.angle
        sidewalk_width = 0.8
        sidewalk_offset = road.width / 2 + sidewalk_width / 2 + 0.1
        quat = p.getQuaternionFromEuler([0, 0, angle])

        for side in [-1, 1]:
            offset_x = side * sidewalk_offset * (-math.sin(angle))
            offset_y = side * sidewalk_offset * math.cos(angle)

            vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[road.length / 2, sidewalk_width / 2, 0.04],
                rgbaColor=SIDEWALK_COLOR,
                specularColor=[0.15, 0.15, 0.15],
                physicsClientId=cli
            )
            p.createMultiBody(
                0, -1, vis,
                basePosition=[
                    road.center[0] + offset_x,
                    road.center[1] + offset_y,
                    SIDEWALK_Z - 0.02
                ],
                baseOrientation=quat,
                physicsClientId=cli
            )

            # Add white edge line at the edge between sidewalk and road
            edge_offset = road.width / 2 + 0.05
            edge_offset_x = side * edge_offset * (-math.sin(angle))
            edge_offset_y = side * edge_offset * math.cos(angle)

            WHITE_LINE = [0.95, 0.95, 0.95, 1.0]
            edge_vis = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[road.length / 2, 0.03, 0.001],
                rgbaColor=WHITE_LINE,
                physicsClientId=cli
            )
            p.createMultiBody(
                0, -1, edge_vis,
                basePosition=[
                    road.center[0] + edge_offset_x,
                    road.center[1] + edge_offset_y,
                    SIDEWALK_Z + 0.001
                ],
                baseOrientation=quat,
                physicsClientId=cli
            )

    placed_buildings = []

    for _ in range(TYPE_1_N_BUILDINGS):
        for attempt in range(200):
            bld_x = rng.uniform(-world_range + 3, world_range - 3)
            bld_y = rng.uniform(-world_range + 3, world_range - 3)

            dist_from_center = math.hypot(bld_x, bld_y)
            if dist_from_center < CITY_DOWNTOWN_RADIUS:
                h_range = (14, CITY_BUILDING_MAX_HEIGHT + 4)
                w_range = (2.5, 4.0)
            elif dist_from_center < CITY_MIDTOWN_RADIUS:
                h_range = (8, 16)
                w_range = (2.0, 3.5)
            else:
                h_range = (CITY_BUILDING_MIN_HEIGHT, 12)
                w_range = (1.8, 3.0)

            width = rng.uniform(*w_range)
            depth = width * rng.uniform(0.8, 1.0)
            bld_radius = max(width, depth) / 2 + 0.3

            # Smart safe zone: allow buildings close to platforms but no overlap
            # Required clearance = platform_radius + building_radius + small_gap
            platform_clearance = LANDING_PLATFORM_RADIUS + bld_radius + 0.3
            too_close_safe = any(
                math.hypot(bld_x - sx, bld_y - sy) < platform_clearance
                for sx, sy in safe_zones
            )
            if too_close_safe:
                continue

            on_road = False
            for road in all_roads:
                dx = bld_x - road.center[0]
                dy = bld_y - road.center[1]
                along = dx * math.cos(road.angle) + dy * math.sin(road.angle)
                perp = abs(-dx * math.sin(road.angle) + dy * math.cos(road.angle))

                min_dist = road.width / 2 + bld_radius + 0.5
                if abs(along) < road.length / 2 + bld_radius and perp < min_dist:
                    on_road = True
                    break

            if on_road:
                continue

            collision = any(
                math.hypot(bld_x - px, bld_y - py) < bld_radius + pr + 0.3
                for px, py, pr in placed_buildings
            )

            if not collision:
                height = rng.uniform(*h_range)
                _add_city_building(cli, [bld_x, bld_y, 0.12], [width, depth, height], rng)
                placed_buildings.append((bld_x, bld_y, bld_radius))
                break

    placed_trees = []
    for road in all_roads:
        angle = road.angle
        tree_offset = road.width / 2 + CITY_TREE_SIDEWALK_OFFSET + 0.5

        num_trees = int(road.length / CITY_TREE_SPACING)
        for i in range(num_trees):
            t = (i + 0.5) / num_trees
            base_x = road.start[0] + t * (road.end[0] - road.start[0])
            base_y = road.start[1] + t * (road.end[1] - road.start[1])

            near_intersection = any(
                math.hypot(base_x - inter.center[0], base_y - inter.center[1]) <
                inter.radius + CITY_TREE_INTERSECTION_CLEARANCE
                for inter in intersections
            )
            if near_intersection:
                continue

            for side in [-1, 1]:
                if rng.random() > 0.85:
                    continue

                tx = base_x + side * tree_offset * (-math.sin(angle))
                ty = base_y + side * tree_offset * math.cos(angle)

                # Check if tree is on ANY road
                on_any_road = False
                for check_road in all_roads:
                    dx = tx - check_road.center[0]
                    dy = ty - check_road.center[1]
                    along = dx * math.cos(check_road.angle) + dy * math.sin(check_road.angle)
                    perp = abs(-dx * math.sin(check_road.angle) + dy * math.cos(check_road.angle))
                    if abs(along) < check_road.length / 2 + 0.5 and perp < check_road.width / 2 + 0.3:
                        on_any_road = True
                        break

                if on_any_road:
                    continue

                # Smart safe zone: tree can be close to platform but not overlap
                tree_platform_clearance = LANDING_PLATFORM_RADIUS + CITY_TREE_RADIUS + 0.3
                too_close_safe = any(
                    math.hypot(tx - sx, ty - sy) < tree_platform_clearance
                    for sx, sy in safe_zones
                )
                if too_close_safe:
                    continue

                too_close_tree = any(
                    math.hypot(tx - px, ty - py) < CITY_TREE_MIN_DIST
                    for px, py in placed_trees
                )
                if too_close_tree:
                    continue

                too_close_building = any(
                    math.hypot(tx - bx, ty - by) < CITY_TREE_RADIUS + br + 0.5
                    for bx, by, br in placed_buildings
                )
                if too_close_building:
                    continue

                h = rng.uniform(CITY_TREE_HEIGHT_MIN, CITY_TREE_HEIGHT_MAX)
                _add_city_tree(cli, [tx, ty, 0.12], h, CITY_TREE_RADIUS, rng)
                placed_trees.append((tx, ty))

    placed_lights = []
    for road in all_roads:
        angle = road.angle
        light_offset = road.width / 2 + 1.2

        num_lights = int(road.length / CITY_LIGHT_SPACING)
        for i in range(num_lights):
            t = (i + 0.5) / max(1, num_lights)
            base_x = road.start[0] + t * (road.end[0] - road.start[0])
            base_y = road.start[1] + t * (road.end[1] - road.start[1])

            near_intersection = any(
                math.hypot(base_x - inter.center[0], base_y - inter.center[1]) < inter.radius + 3.0
                for inter in intersections
            )
            if near_intersection:
                continue

            side = 1 if i % 2 == 0 else -1
            lx = base_x + side * light_offset * (-math.sin(angle))
            ly = base_y + side * light_offset * math.cos(angle)

            # Smart safe zone: streetlight can be close to platform but not overlap
            light_platform_clearance = LANDING_PLATFORM_RADIUS + CITY_STREETLIGHT_RADIUS + 0.3
            too_close_safe = any(
                math.hypot(lx - sx, ly - sy) < light_platform_clearance
                for sx, sy in safe_zones
            )
            if too_close_safe:
                continue

            too_close_tree = any(
                math.hypot(lx - px, ly - py) < CITY_LIGHT_TREE_CLEARANCE
                for px, py in placed_trees
            )
            too_close_light = any(
                math.hypot(lx - px, ly - py) < CITY_LIGHT_MIN_DIST
                for px, py in placed_lights
            )
            if too_close_tree or too_close_light:
                continue

            arm_dir = [-side * (-math.sin(angle)), -side * math.cos(angle)]
            arm_dir_int = [int(round(arm_dir[0])), int(round(arm_dir[1]))]
            if arm_dir_int == [0, 0]:
                arm_dir_int = [0, -side]

            _add_city_streetlight(cli, [lx, ly, 0.12], CITY_STREETLIGHT_HEIGHT,
                                 CITY_STREETLIGHT_RADIUS, arm_dir_int, rng)
            placed_lights.append((lx, ly))

    for inter in intersections:
        if rng.random() < 0.4:
            continue

        corner_offset = 2.5
        corners = [
            (inter.center[0] + corner_offset, inter.center[1] + corner_offset, 'x'),
            (inter.center[0] - corner_offset, inter.center[1] + corner_offset, '-x'),
            (inter.center[0] + corner_offset, inter.center[1] - corner_offset, 'x'),
            (inter.center[0] - corner_offset, inter.center[1] - corner_offset, '-x'),
        ]

        for cx, cy, facing in corners:
            # Smart safe zone: traffic light needs more clearance (pole + arm)
            traffic_light_clearance = LANDING_PLATFORM_RADIUS + 1.5
            too_close_safe = any(
                math.hypot(cx - sx, cy - sy) < traffic_light_clearance
                for sx, sy in safe_zones
            )
            if too_close_safe:
                continue

            if rng.random() < 0.35:
                _add_traffic_light(cli, [cx, cy, 0.12], facing, rng)
                break


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
        n_obstacles = 0
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
        n_obstacles = TYPE_2_N_OBSTACLES
        height_scale = TYPE_2_HEIGHT_SCALE
        safe_zone = TYPE_2_SAFE_ZONE
        world_range = TYPE_2_WORLD_RANGE

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

    if challenge_type == 1:
        safe_zones = []
        if sx is not None and sy is not None:
            safe_zones.append((sx, sy))
        if gx is not None and gy is not None:
            safe_zones.append((gx, gy))
        _generate_procedural_city(cli, rng, world_range, safe_zones, safe_zone)
    elif challenge_type != 4:
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
