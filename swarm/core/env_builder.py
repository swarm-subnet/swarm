# swarm/validator/env_builder.py
"""
Procedurally build the random world and (optionally) add a *visual‑only*
marker that shows the goal position.

The marker has **no collision shape** (baseCollisionShapeIndex = -1) so it
cannot interfere with the drone, but it gives pilots and observers a clear
visual cue of the objective.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, Tuple

from swarm.constants import WORLD_RANGE, HEIGHT_SCALE, N_OBSTACLES
import pybullet as p


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
    """
    Load swarm/assets/tao.png exactly once per PyBullet client
    and return its textureUniqueId.
    """
    if cli not in _TAO_TEX_ID:
        tex_path = Path(__file__).parent.parent / "assets" / "tao.png"
        _TAO_TEX_ID[cli] = p.loadTexture(str(tex_path))
    return _TAO_TEX_ID[cli]


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def build_world(
    seed: int,
    cli: int,
    goal: Optional[Tuple[float, float, float]] = None,
) -> None:
    """
    Build the procedural obstacles and, if ``goal`` is given, place a visual
    marker at that location.

    Parameters
    ----------
    seed
        PRNG seed so strategy and validator see the same world.
    cli
        PyBullet client id.
    goal
        Optional (x, y, z) of the designated target.  When provided, a
        textured TAO badge is rendered at that position.
    """
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Random obstacles (unchanged)
    # ------------------------------------------------------------------
    for _ in range(N_OBSTACLES):
        kind = rng.choice(["wall", "pillar", "box"])
        x, y = rng.uniform(-WORLD_RANGE, WORLD_RANGE), rng.uniform(
            -WORLD_RANGE, WORLD_RANGE
        )
        if math.hypot(x, y) < 2.0:
            continue  # keep take‑off zone clear

        yaw = rng.uniform(0, math.pi)

        if kind == "box":
            sx, sy, sz = (rng.uniform(1, 4) for _ in range(3))
            sz *= HEIGHT_SCALE
            _add_box(cli, pos=[x, y, sz / 2], size=[sx, sy, sz], yaw=yaw)

        elif kind == "wall":
            length = rng.uniform(5, 15)
            height = rng.uniform(2, 5) * HEIGHT_SCALE
            _add_box(
                cli,
                pos=[x, y, height / 2],
                size=[length, 0.3, height],
                yaw=yaw,
            )

        else:  # pillar
            r = rng.uniform(0.3, 1)
            h = rng.uniform(2, 7) * HEIGHT_SCALE
            col = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=r, height=h, physicsClientId=cli
            )
            p.createMultiBody(
                0, col, basePosition=[x, y, h / 2], physicsClientId=cli
            )

    # ------------------------------------------------------------------
    # Visual‑only goal marker: textured TAO badge
    # ------------------------------------------------------------------
    if goal is not None:
        gx, gy, gz = goal

        # ————————————————————————————————————————————————
        # 1) Optional outer green halo (flat disc)
        # ————————————————————————————————————————————————
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
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=halo,
            basePosition=[gx, gy, gz - halo_thick / 2],
            physicsClientId=cli,
        )

        # ────────────────────────────────────────────────────────────────
        # 2) TAO badge – textured quad built in code
        # ────────────────────────────────────────────────────────────────
        badge_size   = 0.50          # 0.5 m × 0.5 m in the X-Y plane
        half         = badge_size/2  # convenience
        badge_offset = 0.001         # raises the quad so its top face is at z = gz

        # 4 vertices, arranged CCW so the front face points towards +Z
        vertices = [
            [-half, -half, 0.0],   # 0 : bottom-left  (u,v) = (0,0)
            [ half, -half, 0.0],   # 1 : bottom-right (u,v) = (1,0)
            [ half,  half, 0.0],   # 2 : top-right    (u,v) = (1,1)
            [-half,  half, 0.0],   # 3 : top-left     (u,v) = (0,1)
        ]

        # two triangles → 6 indices
        indices = [0, 1, 2,   0, 2, 3]

        # per-vertex UVs (same order as vertices above)
        uvs = [
            [0.0, 0.0],  # bottom-left  texel
            [1.0, 0.0],  # bottom-right texel
            [1.0, 1.0],  # top-right    texel
            [0.0, 1.0],  # top-left     texel
        ]

        # Build the visual shape from raw arrays
        vis = p.createVisualShape(
                shapeType=p.GEOM_MESH,
                vertices=vertices,
                indices=indices,
                uvs=uvs,
                # normals are optional; Bullet will compute flat normals for you
                physicsClientId=cli,
        )

        # Spawn the (visual-only) multibody
        uid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,     # no collisions
                baseVisualShapeIndex=vis,
                basePosition=[gx, gy, gz + badge_offset],
                baseOrientation=[0, 0, 0, 1],   # identity quaternion
                physicsClientId=cli,
        )

        # Apply the PNG as a texture
        p.changeVisualShape(
                uid, -1,
                textureUniqueId=_get_tao_tex(cli),
                flags=p.VISUAL_SHAPE_DOUBLE_SIDED,   # render front & back
                physicsClientId=cli,
        )

        # ————————————————————————————————————————————————
        # 3) Optional red pole for extra visibility
        # ————————————————————————————————————————————————
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
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=pole_vis,
            basePosition=[gx, gy, gz + pole_h / 2 + 0.001],
            physicsClientId=cli,
        )
