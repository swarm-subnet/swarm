"""Start / goal platform geometry.

Platform bodies are tagged OBSTACLE_OTHER (never a SUPPORT_* category) so the victim
spawn / clearance logic keeps the victim off the pad. Callers must register the returned
uids in the family's protected_body_uids set so they are excluded from the fatal-collision
check, the obstacle cull, and the safety-clearance metric.
"""
from __future__ import annotations

from pathlib import Path

import pybullet as p

from swarm import constants as C
from .body_tagger import BodyTagger
from .sar_types import BodyCategory

_TAO_TEX: dict[int, int] = {}


def _tao_texture(cli: int) -> int:
    if cli not in _TAO_TEX:
        tex_path = Path(__file__).resolve().parent.parent.parent / "assets" / "tao.png"
        _TAO_TEX[cli] = p.loadTexture(str(tex_path))
    return _TAO_TEX[cli]


def surface_z_at(cli: int, x: float, y: float, top: float = 80.0, bottom: float = -10.0) -> float:
    """Top-down raycast for the surface Z under (x, y); 0.0 on a miss."""
    hit = p.rayTest([x, y, top], [x, y, bottom], physicsClientId=cli)
    if hit and hit[0][0] != -1:
        return float(hit[0][3][2])
    return 0.0


def build_start_platform(tagger: BodyTagger, cli: int, sx: float, sy: float,
                         surface_z: float, challenge_type: int):
    """Spawn the solid red start platform seated on `surface_z`.

    Returns (uids, top_z) where top_z is the platform's top face (the drone takes off
    START_PLATFORM_TAKEOFF_BUFFER above it).
    """
    r = C.START_PLATFORM_RADIUS
    h = C.START_PLATFORM_HEIGHT
    # village (type 4) sits the pad on top of the surface; other maps sink it slightly
    base_z = surface_z + h / 2 + 0.03 if challenge_type == 4 else surface_z - h / 2 + 0.05
    top_z = base_z + h / 2
    uids = []

    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h, physicsClientId=cli)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h,
                              rgbaColor=[1.0, 0.0, 0.0, 1.0], specularColor=[1.0, 0.3, 0.3],
                              physicsClientId=cli)
    body = tagger.create_body(BodyCategory.OBSTACLE_OTHER, baseMass=0,
                              baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                              basePosition=[sx, sy, base_z])
    p.changeDynamics(body, -1, restitution=0.0, lateralFriction=2.5,
                     spinningFriction=1.2, rollingFriction=0.6, physicsClientId=cli)
    uids.append(body)

    # thin high-friction landing disc flush with the top face
    flat_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r * 0.9, height=0.001, physicsClientId=cli)
    flat = tagger.create_body(BodyCategory.OBSTACLE_OTHER, baseMass=0,
                              baseCollisionShapeIndex=flat_col, baseVisualShapeIndex=-1,
                              basePosition=[sx, sy, top_z])
    p.changeDynamics(flat, -1, restitution=0.0, lateralFriction=3.0,
                     spinningFriction=2.0, rollingFriction=1.0, physicsClientId=cli)
    uids.append(flat)

    # bright visual ring marking the takeoff surface
    ring_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r * 0.9, length=0.002,
                                   rgbaColor=[1.0, 0.0, 0.0, 1.0], specularColor=[1.0, 0.3, 0.3],
                                   physicsClientId=cli)
    ring = tagger.create_body(BodyCategory.OBSTACLE_OTHER, baseMass=0,
                              baseCollisionShapeIndex=-1, baseVisualShapeIndex=ring_vis,
                              basePosition=[sx, sy, top_z + 0.001])
    uids.append(ring)

    return uids, top_z


def build_goal_platform(tagger: BodyTagger, cli: int, gx: float, gy: float,
                        surface_z: float, challenge_type: int, rng):
    """Spawn the colored goal platform (landing pad + bright top, TAO badge, marker pole).

    Returns (uids, top_z) where top_z is the landing surface the drone touches down on.
    """
    goal_color = rng.choice(C.GOAL_COLOR_PALETTE)
    r = C.LANDING_PLATFORM_RADIUS
    h = 0.2
    uids = []

    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h, physicsClientId=cli)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=goal_color,
                              specularColor=[goal_color[0] * 0.6 + 0.4,
                                             goal_color[1] * 0.6 + 0.4,
                                             goal_color[2] * 0.6 + 0.4],
                              physicsClientId=cli)
    base_z = surface_z + h / 2 + 0.03 if challenge_type == 4 else surface_z - h / 2 + 0.05
    body = tagger.create_body(BodyCategory.OBSTACLE_OTHER, baseMass=0,
                              baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
                              basePosition=[gx, gy, base_z])
    p.changeDynamics(body, -1, restitution=0.0, lateralFriction=2.0,
                     spinningFriction=1.0, rollingFriction=0.5, physicsClientId=cli)
    uids.append(body)

    top_z = base_z + h / 2
    surface_radius = r * 0.8
    surface_height = 0.008
    bright = [min(1.0, goal_color[0] * 1.25), min(1.0, goal_color[1] * 1.25),
              min(1.0, goal_color[2] * 1.25), 1.0]

    surface_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=surface_radius, length=surface_height,
                                      rgbaColor=bright,
                                      specularColor=[bright[0] * 0.8, bright[1] * 0.8, bright[2] * 0.8],
                                      physicsClientId=cli)
    uids.append(tagger.create_body(BodyCategory.OBSTACLE_OTHER, baseMass=0,
                                   baseCollisionShapeIndex=-1, baseVisualShapeIndex=surface_vis,
                                   basePosition=[gx, gy, top_z + surface_height / 2 + 0.001]))

    flat_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=surface_radius, height=0.001, physicsClientId=cli)
    flat = tagger.create_body(BodyCategory.OBSTACLE_OTHER, baseMass=0,
                              baseCollisionShapeIndex=flat_col, baseVisualShapeIndex=-1,
                              basePosition=[gx, gy, top_z + surface_height + 0.002])
    p.changeDynamics(flat, -1, restitution=0.0, lateralFriction=3.0,
                     spinningFriction=2.0, rollingFriction=1.0, physicsClientId=cli)
    uids.append(flat)

    tao_r = surface_radius * 1.06
    badge_h = 0.005
    bg_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=tao_r, length=badge_h,
                                 rgbaColor=bright, physicsClientId=cli)
    uids.append(tagger.create_body(BodyCategory.OBSTACLE_OTHER, baseMass=0,
                                   baseCollisionShapeIndex=-1, baseVisualShapeIndex=bg_vis,
                                   basePosition=[gx, gy, top_z + surface_height + badge_h + 0.008],
                                   baseOrientation=[0, 0, 0, 1]))
    logo_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=tao_r * 0.95, length=badge_h * 0.5,
                                   rgbaColor=bright, physicsClientId=cli)
    logo = tagger.create_body(BodyCategory.OBSTACLE_OTHER, baseMass=0,
                              baseCollisionShapeIndex=-1, baseVisualShapeIndex=logo_vis,
                              basePosition=[gx, gy, top_z + surface_height + badge_h + 0.011],
                              baseOrientation=[0, 0, 0, 1])
    p.changeVisualShape(logo, -1, textureUniqueId=_tao_texture(cli),
                        flags=p.VISUAL_SHAPE_DOUBLE_SIDED, physicsClientId=cli)
    uids.append(logo)

    pole_h = 0.5
    pole_r = 0.012
    pole_vis = p.createVisualShape(p.GEOM_CYLINDER, radius=pole_r, length=pole_h,
                                   rgbaColor=[1.0, 0.2, 0.1, 0.9], specularColor=[1.0, 0.8, 0.2],
                                   physicsClientId=cli)
    uids.append(tagger.create_body(BodyCategory.OBSTACLE_OTHER, baseMass=0,
                                   baseCollisionShapeIndex=-1, baseVisualShapeIndex=pole_vis,
                                   basePosition=[gx, gy, top_z + pole_h / 2 + 0.008]))
    cap_vis = p.createVisualShape(p.GEOM_SPHERE, radius=pole_r * 2,
                                  rgbaColor=[1.0, 0.3, 0.0, 1.0], specularColor=[1.0, 1.0, 0.4],
                                  physicsClientId=cli)
    uids.append(tagger.create_body(BodyCategory.OBSTACLE_OTHER, baseMass=0,
                                   baseCollisionShapeIndex=-1, baseVisualShapeIndex=cap_vis,
                                   basePosition=[gx, gy, top_z + pole_h + 0.015]))

    return uids, top_z
