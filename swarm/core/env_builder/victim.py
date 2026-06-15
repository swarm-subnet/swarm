from __future__ import annotations

import json
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
_FLAT_SLOPE_RAD = math.radians(5.0)
_STAND_MAX_TILT = math.radians(12.0)
_LIE_MAX_TILT = math.radians(40.0)
_MAX_TERRAIN_PENETRATION = 0.35

_PEOPLE_DIR = (
    _REPO_ROOT
    / "swarm"
    / "assets"
    / "maps"
    / "custom"
    / "people"
    / "lost_person_characters"
)
_MANIFEST_PATH = _PEOPLE_DIR / "manifest.json"
_VICTIM_SELECT_SALT = 0x56494354
_CHALLENGE_TYPE_TO_MAP = {
    1: "city",
    2: "open",
    3: "mountain",
    4: "village",
    5: "warehouse",
    6: "forest",
}

_manifest_characters: list | None = None


def _load_characters() -> list:
    global _manifest_characters
    if _manifest_characters is None:
        try:
            data = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []
        _manifest_characters = list(data.get("characters", []))
    return _manifest_characters


def select_victim_split_dir(seed: int, challenge_type: int, slope_deg: float = 0.0) -> Path | None:
    map_name = _CHALLENGE_TYPE_TO_MAP.get(int(challenge_type))
    if map_name is None:
        return None
    candidates = [c for c in _load_characters() if map_name in c.get("maps", [])]
    eligible = [c for c in candidates if c.get("max_slope_deg", 30.0) >= slope_deg]
    if not eligible:
        eligible = candidates
    if not eligible:
        return None
    eligible.sort(key=lambda c: c["index"])
    rng = random.Random((int(seed) & 0xFFFFFFFF) ^ _VICTIM_SELECT_SALT)
    chosen = eligible[rng.randrange(len(eligible))]
    split_dir = (_PEOPLE_DIR / chosen["model"]).parent / "split"
    if not iter_prebaked_parts(split_dir):
        return None
    return split_dir


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


def _sample_terrain_slope(cli, cx, cy, surface_z, radius):
    r = max(float(radius), 0.15)
    offsets = (
        (0.0, 0.0),
        (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r),
        (r, r), (r, -r), (-r, r), (-r, -r),
    )
    z_hi = surface_z + 20.0
    z_lo = surface_z - 20.0
    points = []
    for dx, dy in offsets:
        x = cx + dx
        y = cy + dy
        hit = p.rayTest([x, y, z_hi], [x, y, z_lo], physicsClientId=cli)
        if hit and hit[0][0] >= 0:
            points.append((x, y, float(hit[0][3][2])))
    if len(points) < 3:
        return (0.0, 0.0, 1.0), [pt[2] for pt in points]
    kept = [pt for pt in points if abs(pt[2] - surface_z) <= 2.0]
    if len(kept) < 3:
        return (0.0, 0.0, 1.0), [float(surface_z)]
    return _fit_plane_normal(kept), [pt[2] for pt in kept]


def _solve3x3(matrix, rhs):
    (a, b, c), (d, e, f), (g, h, i) = matrix
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) < 1e-12:
        return None
    r0, r1, r2 = rhs
    da = r0 * (e * i - f * h) - b * (r1 * i - f * r2) + c * (r1 * h - e * r2)
    db = a * (r1 * i - f * r2) - r0 * (d * i - f * g) + c * (d * r2 - r1 * g)
    dc = a * (e * r2 - r1 * h) - b * (d * r2 - r1 * g) + r0 * (d * h - e * g)
    return da / det, db / det, dc / det


def _fit_plane_normal(points):
    sx = sy = sz = sxx = sxy = syy = sxz = syz = 0.0
    for x, y, z in points:
        sx += x
        sy += y
        sz += z
        sxx += x * x
        sxy += x * y
        syy += y * y
        sxz += x * z
        syz += y * z
    solution = _solve3x3(
        ((sxx, sxy, sx), (sxy, syy, sy), (sx, sy, float(len(points)))),
        (sxz, syz, sz),
    )
    if solution is None:
        return (0.0, 0.0, 1.0)
    a, b, _c = solution
    nx, ny, nz = -a, -b, 1.0
    mag = math.sqrt(nx * nx + ny * ny + nz * nz)
    if mag < 1e-9:
        return (0.0, 0.0, 1.0)
    return (nx / mag, ny / mag, nz / mag)


def _percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[int(q * (len(ordered) - 1))]


def _quat_axis_angle(axis, angle):
    half = angle * 0.5
    s = math.sin(half)
    return (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(half))


def _tilt_to_slope(base_quat, normal, lying):
    nz = max(-1.0, min(1.0, normal[2]))
    full_angle = math.acos(nz)
    if full_angle < 1e-4:
        return base_quat
    axis_x = -normal[1]
    axis_y = normal[0]
    mag = math.hypot(axis_x, axis_y)
    if mag < 1e-9:
        return base_quat
    cap = _LIE_MAX_TILT if lying else _STAND_MAX_TILT
    angle = min(full_angle, cap)
    tilt = _quat_axis_angle((axis_x / mag, axis_y / mag, 0.0), angle)
    return _quat_mul(tilt, base_quat)


def terrain_slope_deg(cli, x, y, surface_z, radius=0.4):
    normal, _ = _sample_terrain_slope(cli, float(x), float(y), float(surface_z), radius)
    return math.degrees(math.acos(max(-1.0, min(1.0, normal[2]))))


def _fall_line_yaw(raw_bounds, normal):
    up0 = _rotated_bounds(raw_bounds, _compose_orientation(0.0), (0.0, 0.0, 0.0))
    axis0 = 0.0 if (up0[1][0] - up0[0][0]) >= (up0[1][1] - up0[0][1]) else math.pi / 2.0
    return math.atan2(-normal[1], -normal[0]) - axis0


def _footprint_seat_z(cli, cx, cy, fx, fy, surface_z, q, normal):
    fx = max(fx, 0.15)
    fy = max(fy, 0.15)
    z_hi = surface_z + 20.0
    z_lo = surface_z - 20.0
    samples = []
    for ix in range(5):
        for iy in range(5):
            dx = (ix / 2.0 - 1.0) * fx
            dy = (iy / 2.0 - 1.0) * fy
            hit = p.rayTest([cx + dx, cy + dy, z_hi], [cx + dx, cy + dy, z_lo], physicsClientId=cli)
            if hit and hit[0][0] >= 0 and abs(hit[0][3][2] - surface_z) <= 2.0:
                samples.append((dx, dy, hit[0][3][2]))
    if not samples:
        return float(surface_z)
    seat = min(_percentile([s[2] for s in samples], q), float(surface_z))
    nz = normal[2] if abs(normal[2]) > 1e-6 else 1.0
    max_pen = 0.0
    for dx, dy, z in samples:
        underside = seat - (normal[0] * dx + normal[1] * dy) / nz
        max_pen = max(max_pen, z - underside)
    if max_pen > _MAX_TERRAIN_PENETRATION:
        seat += max_pen - _MAX_TERRAIN_PENETRATION
    return seat


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
    base_quat = _compose_orientation(yaw)
    upright_min, upright_max = _rotated_bounds(
        (raw_min, raw_max), base_quat, base_position=(0.0, 0.0, 0.0),
    )

    footprint_radius = 0.5 * max(
        upright_max[0] - upright_min[0], upright_max[1] - upright_min[1],
    )
    normal, _ = _sample_terrain_slope(
        cli, float(surface_x), float(surface_y), float(surface_z), footprint_radius,
    )
    slope_angle = math.acos(max(-1.0, min(1.0, normal[2])))

    if slope_angle < _FLAT_SLOPE_RAD:
        quat = base_quat
        rot_min = upright_min
        seat_z = float(surface_z)
    else:
        vert_extent = upright_max[2] - upright_min[2]
        flat = vert_extent < 0.6
        if flat:
            oriented = _compose_orientation(_fall_line_yaw((raw_min, raw_max), normal))
            quat = _tilt_to_slope(oriented, normal, True)
        else:
            quat = _tilt_to_slope(base_quat, normal, False)
        rot_min, rot_max = _rotated_bounds(
            (raw_min, raw_max), quat, base_position=(0.0, 0.0, 0.0),
        )
        seat_z = _footprint_seat_z(
            cli, float(surface_x), float(surface_y),
            0.5 * (rot_max[0] - rot_min[0]), 0.5 * (rot_max[1] - rot_min[1]),
            float(surface_z), 0.10 if flat else 0.15, normal,
        )

    base_z = seat_z - rot_min[2] + _GROUND_EPS
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
