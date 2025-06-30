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

The marker itself has **no collision shape** (baseCollisionShapeIndex = ‑1);
it is only visual.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, Tuple

import pybullet as p

from swarm.constants import WORLD_RANGE, HEIGHT_SCALE, N_OBSTACLES

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------
SAFE_ZONE_RADIUS = 2.0         # keep at least 2 m of clearance
MAX_ATTEMPTS_PER_OBS = 100     # retry limit when placing each obstacle

# --------------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------------
def _add_box(cli: int, pos, size, yaw) -> None:
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[s / 2 for s in size], physicsClientId=cli
    )
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    p.createMultiBody(
        0, col, basePosition=pos, baseOrientation=quat, physicsClientId=cli
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

            # — safe‑zone checks --------------------------------------
            def _violates(cx, cy):
                if cx is None:
                    return False
                return math.hypot(x - cx, y - cy) < (obj_r + SAFE_ZONE_RADIUS)

            if _violates(sx, sy) or _violates(gx, gy):
                continue  # too close – try another location

            # ----------------------------------------------------------
            # Passed all tests → create the obstacle
            # ----------------------------------------------------------
            if kind == "box":
                _add_box(cli, [x, y, sz_len / 2], [sx_len, sy_len, sz_len], yaw)

            elif kind == "wall":
                _add_box(cli, [x, y, sz_len / 2], [sx_len, sy_len, sz_len], yaw)

            else:  # pillar
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=obj_r, height=sz_len, physicsClientId=cli
                )
                p.createMultiBody(
                    0, col, basePosition=[x, y, sz_len / 2], physicsClientId=cli
                )

            placed += 1
            break  # obstacle placed – move to next one
        else:
            # Unable to place this obstacle after many attempts; skip it
            break

    # ------------------------------------------------------------------
    # Visual‑only goal marker (unchanged)
    # ------------------------------------------------------------------
    if goal is not None:
        gx, gy, gz = goal

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
