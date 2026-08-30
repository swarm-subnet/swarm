"""Builder for the office indoor map (Tello interceptor family).

Loads the baked office digital twin: 7 mesh groups whose lighting is
pre-baked into texture atlases. The family layers seeded per-episode
appearance on top (body tints and render light), so identical rendering
across validators comes from the shared seed, not a fixed look.

The map is fully static and deterministic: no RNG is consumed, every body
is spawned at mass 0 from committed assets.
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import pybullet as p

from swarm.constants import (
    OFFICE_SCALE_JITTER_MAX,
    OFFICE_SCALE_JITTER_MIN,
    OFFICE_SCALE_SEED_OFFSET,
)

_PACKAGE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_OFFICE_ASSET_DIR = os.path.join(_PACKAGE_DIR, "assets", "maps", "custom", "office")

OFFICE_X_RANGE: Tuple[float, float] = (0.0, 18.0)
OFFICE_Y_RANGE: Tuple[float, float] = (0.0, 7.6)
OFFICE_CEILING_M: float = 3.0

OFFICE_LIGHT_DIRECTION: List[float] = [0.0, 0.0, 1.0]
OFFICE_LIGHT_AMBIENT: float = 1.0
OFFICE_LIGHT_DIFFUSE: float = 0.0
OFFICE_LIGHT_SPECULAR: float = 0.0

_SOLID_GROUPS: Tuple[str, ...] = ("floor", "shell", "west", "mid", "east")
_VISUAL_ONLY_GROUPS: Tuple[str, ...] = ("led", "backdrop")

_WINDOW_PLUG_CENTER: Tuple[float, float, float] = (17.99, 3.8, 1.65)
_WINDOW_PLUG_HALF_EXTENTS: Tuple[float, float, float] = (0.02, 1.26, 0.76)

def _asset(name: str) -> str:
    path = os.path.join(_OFFICE_ASSET_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"office map asset missing: {path}")
    return path


def _visual_shape(cli: int, obj_path: str, scale) -> int:
    return p.createVisualShape(p.GEOM_MESH, fileName=obj_path,
                               meshScale=list(scale), physicsClientId=cli)


def _collision_shape(cli: int, obj_path: str, scale) -> int:
    flags = p.GEOM_FORCE_CONCAVE_TRIMESH if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH") else 0
    return p.createCollisionShape(
        p.GEOM_MESH, fileName=obj_path, flags=flags, meshScale=list(scale),
        physicsClientId=cli
    )


def office_scale(seed: int) -> Tuple[float, float, float]:
    """This episode's room proportions. A real site is never the exact size of the
    drawing, and the axes stretch independently so a metric grid fitted to one
    floorplan does not line up with the next."""
    rng = random.Random((int(seed) ^ OFFICE_SCALE_SEED_OFFSET) & 0xFFFFFFFF)
    def axis() -> float:
        return 1.0 + rng.choice((-1.0, 1.0)) * rng.uniform(
            OFFICE_SCALE_JITTER_MIN, OFFICE_SCALE_JITTER_MAX)
    return (axis(), axis(), axis())


def build_office_map(seed: int = 0, cli: int = 0) -> dict:
    """Build the office map inside an existing PyBullet world.

    Parameters
    ----------
    seed : int
        Accepted for signature parity with the other map builders; the
        office is static so the seed is never consumed.
    cli : int
        PyBullet physics client id.

    Returns
    -------
    dict
        Body ids per group, the window-plug body id, and the flyable bounds.
    """
    bodies: Dict[str, int] = {}
    sx, sy, sz = office_scale(seed)
    scale = (sx, sy, sz)

    for group in _SOLID_GROUPS:
        obj_path = _asset(f"office_{group}.obj")
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=_collision_shape(cli, obj_path, scale),
            baseVisualShapeIndex=_visual_shape(cli, obj_path, scale),
            physicsClientId=cli,
        )
        p.changeVisualShape(bid, -1, rgbaColor=[1, 1, 1, 1], physicsClientId=cli)
        bodies[group] = bid

    for group in _VISUAL_ONLY_GROUPS:
        obj_path = _asset(f"office_{group}.obj")
        bid = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=_visual_shape(cli, obj_path, scale),
            physicsClientId=cli,
        )
        p.changeVisualShape(bid, -1, rgbaColor=[1, 1, 1, 1], physicsClientId=cli)
        bodies[group] = bid

    plug_col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[h * a for h, a in zip(_WINDOW_PLUG_HALF_EXTENTS, scale)],
        physicsClientId=cli
    )
    window_plug = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=plug_col,
        basePosition=[c * a for c, a in zip(_WINDOW_PLUG_CENTER, scale)],
        physicsClientId=cli,
    )

    return {
        "bodies": bodies,
        "window_plug": window_plug,
        "x_range": (OFFICE_X_RANGE[0] * sx, OFFICE_X_RANGE[1] * sx),
        "y_range": (OFFICE_Y_RANGE[0] * sy, OFFICE_Y_RANGE[1] * sy),
        "ceiling_m": OFFICE_CEILING_M * sz,
        "scale": scale,
        "light": {
            "lightDirection": OFFICE_LIGHT_DIRECTION,
            "lightAmbientCoeff": OFFICE_LIGHT_AMBIENT,
            "lightDiffuseCoeff": OFFICE_LIGHT_DIFFUSE,
            "lightSpecularCoeff": OFFICE_LIGHT_SPECULAR,
        },
    }
