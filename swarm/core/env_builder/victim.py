from __future__ import annotations

import math
import os
import random
from pathlib import Path
from typing import Set, Tuple

import pybullet as p

from .body_tagger import BodyTagger
from .mesh_loader import (
    iter_prebaked_parts,
    make_aabb_collision_shape,
    prebaked_union_bounds,
    spawn_split_material_mesh,
)
from .sar_types import BodyCategory


VictimAttrs = Tuple[list, Tuple[Tuple[float, float, float], Tuple[float, float, float]], Tuple[float, float, float]]


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_SPLIT_DIR = (
    _REPO_ROOT
    / "swarm"
    / "assets"
    / "maps"
    / "custom"
    / "people"
    / "open_mannequin_raw"
    / "split"
)

_Y_UP_TO_Z_UP = (math.pi / 2.0, 0.0, 0.0)
_GROUND_EPS = 0.005


def accepted_categories_for(challenge_type: int) -> Set[BodyCategory]:
    if challenge_type == 5:
        return {BodyCategory.SUPPORT_FLOOR}
    if challenge_type in (1, 4):
        return {BodyCategory.SUPPORT_TERRAIN, BodyCategory.SUPPORT_ROOFTOP}
    return {
        BodyCategory.SUPPORT_TERRAIN,
        BodyCategory.SUPPORT_SLOPE,
        BodyCategory.SUPPORT_WALKWAY,
    }


def _compose_orientation(yaw_rad: float):
    yaw_quat = p.getQuaternionFromEuler([0.0, 0.0, yaw_rad])
    rot_quat = p.getQuaternionFromEuler(list(_Y_UP_TO_Z_UP))
    return _quat_mul(yaw_quat, rot_quat)


def _quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def _rotated_bounds(
    raw_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
    quat,
    base_position,
):
    (mn_x, mn_y, mn_z), (mx_x, mx_y, mx_z) = raw_bounds
    corners = [
        (x, y, z)
        for x in (mn_x, mx_x)
        for y in (mn_y, mx_y)
        for z in (mn_z, mx_z)
    ]
    rot_matrix = p.getMatrixFromQuaternion(list(quat))
    rmin = [float("inf")] * 3
    rmax = [float("-inf")] * 3
    for c in corners:
        wx = rot_matrix[0] * c[0] + rot_matrix[1] * c[1] + rot_matrix[2] * c[2]
        wy = rot_matrix[3] * c[0] + rot_matrix[4] * c[1] + rot_matrix[5] * c[2]
        wz = rot_matrix[6] * c[0] + rot_matrix[7] * c[1] + rot_matrix[8] * c[2]
        rmin[0] = min(rmin[0], wx); rmin[1] = min(rmin[1], wy); rmin[2] = min(rmin[2], wz)
        rmax[0] = max(rmax[0], wx); rmax[1] = max(rmax[1], wy); rmax[2] = max(rmax[2], wz)
    return (
        (rmin[0] + base_position[0], rmin[1] + base_position[1], rmin[2] + base_position[2]),
        (rmax[0] + base_position[0], rmax[1] + base_position[1], rmax[2] + base_position[2]),
    )


def spawn_victim(
    cli: int,
    *,
    surface_x: float,
    surface_y: float,
    surface_z: float,
    rng: random.Random,
    tagger: BodyTagger,
    split_dir: str | os.PathLike | None = None,
    double_sided: bool = True,
) -> VictimAttrs:
    split_path = Path(split_dir) if split_dir else _DEFAULT_SPLIT_DIR

    if not iter_prebaked_parts(split_path):
        raise FileNotFoundError(f"no prebaked mannequin parts in {split_path}")

    raw_min, raw_max = prebaked_union_bounds(split_path)
    half_extents = (
        0.5 * (raw_max[0] - raw_min[0]),
        0.5 * (raw_max[1] - raw_min[1]),
        0.5 * (raw_max[2] - raw_min[2]),
    )
    centroid = (
        0.5 * (raw_max[0] + raw_min[0]),
        0.5 * (raw_max[1] + raw_min[1]),
        0.5 * (raw_max[2] + raw_min[2]),
    )

    collision_id = make_aabb_collision_shape(
        cli,
        half_extents=half_extents,
        frame_position=centroid,
    )

    yaw = rng.uniform(0.0, 2.0 * math.pi)
    quat = _compose_orientation(yaw)

    rot_min, rot_max = _rotated_bounds(
        (raw_min, raw_max), quat, base_position=(0.0, 0.0, 0.0),
    )
    base_z = surface_z - rot_min[2] + _GROUND_EPS
    base_position = (float(surface_x), float(surface_y), float(base_z))

    uids = spawn_split_material_mesh(
        cli,
        prebaked_dir=split_path,
        base_position=base_position,
        base_orientation=quat,
        collision_id=collision_id,
        double_sided=double_sided,
    )
    tagger.tag_body_group(BodyCategory.VICTIM, uids)

    union_min, union_max = _rotated_bounds(
        (raw_min, raw_max), quat, base_position=base_position,
    )
    centre = (
        0.5 * (union_min[0] + union_max[0]),
        0.5 * (union_min[1] + union_max[1]),
        0.5 * (union_min[2] + union_max[2]),
    )
    return uids, (union_min, union_max), centre
