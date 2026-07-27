"""Builder for the office indoor map (Tello interceptor family).

Loads the baked office digital twin: 7 mesh groups whose lighting is
pre-baked into texture atlases, so the map renders identically on every
validator with plain ambient light and zero run-time light computation.

The map is fully static and deterministic: no RNG is consumed, every body
is spawned at mass 0 from committed assets.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import pybullet as p

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


def _visual_shape(cli: int, obj_path: str) -> int:
    return p.createVisualShape(p.GEOM_MESH, fileName=obj_path, physicsClientId=cli)


def _collision_shape(cli: int, obj_path: str) -> int:
    flags = p.GEOM_FORCE_CONCAVE_TRIMESH if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH") else 0
    return p.createCollisionShape(
        p.GEOM_MESH, fileName=obj_path, flags=flags, physicsClientId=cli
    )


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

    for group in _SOLID_GROUPS:
        obj_path = _asset(f"office_{group}.obj")
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=_collision_shape(cli, obj_path),
            baseVisualShapeIndex=_visual_shape(cli, obj_path),
            physicsClientId=cli,
        )
        p.changeVisualShape(bid, -1, rgbaColor=[1, 1, 1, 1], physicsClientId=cli)
        bodies[group] = bid

    for group in _VISUAL_ONLY_GROUPS:
        obj_path = _asset(f"office_{group}.obj")
        bid = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=_visual_shape(cli, obj_path),
            physicsClientId=cli,
        )
        p.changeVisualShape(bid, -1, rgbaColor=[1, 1, 1, 1], physicsClientId=cli)
        bodies[group] = bid

    plug_col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=list(_WINDOW_PLUG_HALF_EXTENTS), physicsClientId=cli
    )
    window_plug = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=plug_col,
        basePosition=list(_WINDOW_PLUG_CENTER),
        physicsClientId=cli,
    )

    return {
        "bodies": bodies,
        "window_plug": window_plug,
        "x_range": OFFICE_X_RANGE,
        "y_range": OFFICE_Y_RANGE,
        "ceiling_m": OFFICE_CEILING_M,
        "light": {
            "lightDirection": OFFICE_LIGHT_DIRECTION,
            "lightAmbientCoeff": OFFICE_LIGHT_AMBIENT,
            "lightDiffuseCoeff": OFFICE_LIGHT_DIFFUSE,
            "lightSpecularCoeff": OFFICE_LIGHT_SPECULAR,
        },
    }
