# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Start / goal platform geometry.

Platform bodies are tagged OBSTACLE_OTHER (never a SUPPORT_* category) so the victim
spawn / clearance logic keeps the victim off the pad. Callers must register the returned
uids in the family's protected_body_uids set so they are excluded from the fatal-collision
check, the obstacle cull, and the safety-clearance metric.
"""
from __future__ import annotations

import math
from pathlib import Path

import pybullet as p

from swarm import constants as C

from .body_tagger import BodyTagger
from .sar_types import BodyCategory
from .surface_resolver import resolve_surface

_TAO_TEX: dict[int, int] = {}

# Launch volume probed for a warehouse start pad, measured from the resolved floor.
PAD_PROBE_RADIUS = C.START_PLATFORM_RADIUS + 0.05
PAD_PROBE_BOTTOM = 0.05
PAD_PROBE_TOP = 0.45
# Decide on a real margin; branching on exact tangency can differ between Bullet builds.
PAD_CLEARANCE_EPS = 0.02
_PAD_LEGACY_RINGS = ((2.0, 8), (4.0, 8), (6.0, 8), (8.0, 8))
_PAD_EXTRA_RINGS = ((1.0, 12), (1.5, 12), (3.0, 12), (5.0, 12), (10.0, 12), (12.0, 12))
# The victim is placed near the original start, so a distant pad turns a solvable seed
# into an infeasible one; past this the placement fails instead.
PAD_MAX_RELOCATION = 12.0
_PAD_SWEEP_STEP = 1.0
_PAD_PARK = (0.0, 0.0, 1.0e4)


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


def surface_z_max_at(cli: int, x: float, y: float, radius: float,
                     top: float = 500.0, bottom: float = -100.0) -> float:
    """Highest surface Z inside the disc of ``radius`` around (x, y), so a pad
    placed at the result cannot be pierced by terrain anywhere under it."""
    steps = [-radius + i * (2.0 * radius / 8.0) for i in range(9)]
    best = surface_z_at(cli, x, y, top, bottom)
    for dx in steps:
        for dy in steps:
            if dx * dx + dy * dy > radius * radius:
                continue
            z = surface_z_at(cli, x + dx, y + dy, top, bottom)
            if z > best:
                best = z
    return best


def blocking_top(cli: int, x: float, y: float, exclude, pad_r: float = 1.2,
                 min_height: float = 1.0):
    """Top Z of the tallest floor-standing structure overlapping a pad footprint
    at (x, y), or None when the spot is clear. Ignores map-wide slabs and
    overhead bodies (roofs, trusses) that never block a ground pad."""
    top = None
    for idx in range(p.getNumBodies(physicsClientId=cli)):
        uid = p.getBodyUniqueId(idx, physicsClientId=cli)
        if uid in exclude:
            continue
        mn, mx = p.getAABB(uid, physicsClientId=cli)
        if (mx[0] - mn[0]) > 60.0 or (mx[1] - mn[1]) > 60.0:
            continue
        if (mx[2] - mn[2]) <= min_height or mn[2] > 2.0:
            continue
        if mn[0] - pad_r <= x <= mx[0] + pad_r and mn[1] - pad_r <= y <= mx[1] + pad_r:
            top = mx[2] if top is None else max(top, mx[2])
    return top


def clear_pad_spot(cli: int, sx: float, sy: float, exclude):
    """Nudge a start-pad position out of any structure; the original spot wins
    when already clear."""
    if blocking_top(cli, sx, sy, exclude) is None:
        return sx, sy
    for r in (2.0, 4.0, 6.0, 8.0):
        for k in range(8):
            a = k * math.pi / 4.0
            nx, ny = sx + r * math.cos(a), sy + r * math.sin(a)
            if blocking_top(cli, nx, ny, exclude) is None:
                return nx, ny
    return sx, sy


def _pad_candidates(sx: float, sy: float, legacy_xy, bounds, keep_out):
    """Pad positions to try, nearest-first: the legacy pick, the original spot, the
    legacy rings, wider rings, then a lattice sweep within the relocation limit."""
    bx, by = bounds
    seen: set = set()

    def _emit(x, y):
        if abs(x) > bx or abs(y) > by:
            return None
        if math.hypot(x - sx, y - sy) > PAD_MAX_RELOCATION:
            return None
        if keep_out is not None and math.hypot(x - keep_out[0], y - keep_out[1]) < keep_out[2]:
            return None
        key = (round(x, 4), round(y, 4))
        if key in seen:
            return None
        seen.add(key)
        return (x, y)

    ordered = []
    if legacy_xy is not None:
        ordered.append((float(legacy_xy[0]), float(legacy_xy[1])))
    ordered.append((sx, sy))
    for r, n in _PAD_LEGACY_RINGS + _PAD_EXTRA_RINGS:
        for k in range(n):
            a = k * 2.0 * math.pi / n
            ordered.append((sx + r * math.cos(a), sy + r * math.sin(a)))

    sweep = []
    steps = int(PAD_MAX_RELOCATION / _PAD_SWEEP_STEP)
    for i in range(-steps, steps + 1):
        for j in range(-steps, steps + 1):
            x, y = sx + i * _PAD_SWEEP_STEP, sy + j * _PAD_SWEEP_STEP
            sweep.append(((x - sx) ** 2 + (y - sy) ** 2, x, y))
    sweep.sort()
    ordered.extend((x, y) for _d, x, y in sweep)

    for x, y in ordered:
        spot = _emit(x, y)
        if spot is not None:
            yield spot


def _launch_volume_blocked(cli: int, probe: int, x: float, y: float,
                           surface_z: float, exclude) -> bool:
    """True when the volume the drone will occupy is within PAD_CLEARANCE_EPS of a solid."""
    mid = surface_z + (PAD_PROBE_BOTTOM + PAD_PROBE_TOP) / 2.0
    p.resetBasePositionAndOrientation(probe, [x, y, mid], [0.0, 0.0, 0.0, 1.0],
                                      physicsClientId=cli)
    mn, mx = p.getAABB(probe, physicsClientId=cli)
    e = PAD_CLEARANCE_EPS
    near = p.getOverlappingObjects([mn[0] - e, mn[1] - e, mn[2] - e],
                                   [mx[0] + e, mx[1] + e, mx[2] + e],
                                   physicsClientId=cli) or ()
    for uid in sorted({int(item[0]) for item in near}):
        if uid == probe or uid in exclude:
            continue
        if p.getClosestPoints(probe, uid, distance=e, physicsClientId=cli):
            return True
    return False


def resolve_warehouse_pad_spot(cli: int, sx: float, sy: float, *, body_tags,
                               accepted_categories, exclude, bounds, legacy_xy=None,
                               keep_out=None):
    """Nearest start-pad spot with a real floor under it and an empty launch volume.

    Returns (x, y, surface_z), or None when no candidate qualifies. An AABB test cannot
    see a warehouse pallet stack because every body in it is 0.30 m tall, so the volume
    is put to the collision engine instead. ``keep_out`` is an (x, y, radius) disc the
    pad must stay outside, so relocating never lands on the victim.
    """
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=PAD_PROBE_RADIUS,
                                 height=PAD_PROBE_TOP - PAD_PROBE_BOTTOM,
                                 physicsClientId=cli)
    probe = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                              basePosition=list(_PAD_PARK), physicsClientId=cli)
    try:
        for x, y in _pad_candidates(sx, sy, legacy_xy, bounds, keep_out):
            # Park the probe first: left over a candidate it blocks the surface ray.
            p.resetBasePositionAndOrientation(probe, list(_PAD_PARK), [0.0, 0.0, 0.0, 1.0],
                                              physicsClientId=cli)
            hit = resolve_surface(cli, x, y, body_tags, accepted_categories)
            if hit is None:
                continue
            surface_z = float(hit.surface_z)
            skip = set(exclude) | {int(hit.support_uid)}
            if not _launch_volume_blocked(cli, probe, x, y, surface_z, skip):
                return x, y, surface_z
    finally:
        p.removeBody(probe, physicsClientId=cli)
    return None


def build_start_platform(tagger: BodyTagger, cli: int, sx: float, sy: float,
                         surface_z: float, challenge_type: int,
                         radius: float = None, height: float = None):
    """Spawn the solid red start platform seated on `surface_z`.

    Returns (uids, top_z) where top_z is the platform's top face (the drone takes off
    START_PLATFORM_TAKEOFF_BUFFER above it). `radius`/`height` override the defaults for
    families that fly a different-sized drone (e.g. interceptor); other callers keep them.
    """
    r = C.START_PLATFORM_RADIUS if radius is None else float(radius)
    h = C.START_PLATFORM_HEIGHT if height is None else float(height)
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
