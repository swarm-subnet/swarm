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
from typing import Optional, Tuple
from swarm.constants import WORLD_RANGE
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


# 100 % height (kept from original – change here if you want flatter scenery)
HEIGHT_SCALE = 0.2


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def build_world(
    seed: int,
    cli: int,
    goal: Optional[Tuple[float, float, float]] = None,   # NEW
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
        Optional (x, y, z) of the designated target.  When provided a *flat
        black disk* is rendered at that position.
    """
    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Random obstacles (unchanged from the original version)
    # ------------------------------------------------------------------
    for _ in range(120):
        kind = rng.choice(["wall", "pillar", "box"])
        x, y = rng.uniform(-WORLD_RANGE, WORLD_RANGE), rng.uniform(
            -WORLD_RANGE, WORLD_RANGE
        )
        if math.hypot(x, y) < 2.0:       # keep take‑off zone clear
            continue

        yaw = rng.uniform(0, math.pi)

        # ------------------------------ box ---------------------------
        if kind == "box":
            sx, sy, sz = (rng.uniform(1, 4) for _ in range(3))
            sz *= HEIGHT_SCALE
            _add_box(
                cli,
                pos=[x, y, sz / 2],
                size=[sx, sy, sz],
                yaw=yaw,
            )

        # ------------------------------ wall --------------------------
        elif kind == "wall":
            length = rng.uniform(5, 15)
            height = rng.uniform(2, 5) * HEIGHT_SCALE
            _add_box(
                cli,
                pos=[x, y, height / 2],
                size=[length, 0.3, height],
                yaw=yaw,
            )

        # ----------------------------- pillar -------------------------
        else:
            r = rng.uniform(0.3, 0.6)
            h = rng.uniform(2, 7) * HEIGHT_SCALE
            col = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=r, height=h, physicsClientId=cli
            )
            p.createMultiBody(
                0, col, basePosition=[x, y, h / 2], physicsClientId=cli
            )

        # ------------------------------------------------------------------
    # Visual-only goal marker  (revamped "bull-seye + flag" style)
    # ------------------------------------------------------------------
    if goal is not None:
        gx, gy, gz = goal
        thickness = 0.02           # still paper-thin

        # --- outer ring (bright green) --------------------------------
        outer_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.45,           # a touch larger than before
            length=thickness,
            rgbaColor=[0.15, 0.8, 0.15, 1],   # vibrant green
            specularColor=[0.3, 0.3, 0.3],
            physicsClientId=cli,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=outer_id,
            basePosition=[gx, gy, gz - thickness / 2],   # centre on z
            physicsClientId=cli,
        )

        # --- inner disk (white) ---------------------------------------
        inner_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.22,           # half the outer radius → bull-seye
            length=thickness,
            rgbaColor=[1, 1, 1, 1],   # crisp white
            specularColor=[0.6, 0.6, 0.6],
            physicsClientId=cli,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=inner_id,
            basePosition=[gx, gy, gz - thickness / 2],
            physicsClientId=cli,
        )

        # --- vertical flag-pole (helps spotting the goal from afar) ----
        pole_height = 0.5          # 50 cm pole
        pole_id = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.015,
            length=pole_height,
            rgbaColor=[0.9, 0.1, 0.1, 1],   # red pole
            specularColor=[0.4, 0.4, 0.4],
            physicsClientId=cli,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=pole_id,
            basePosition=[gx, gy, gz + pole_height / 2],    # stand on the disk
            physicsClientId=cli,
        )
