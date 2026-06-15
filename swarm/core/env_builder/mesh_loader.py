from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pybullet as p


Vec3 = Tuple[float, float, float]


_PREBAKED_OPEN_FILE = open


def obj_bounds(obj_path: str | Path) -> Tuple[Vec3, Vec3]:
    mn = [float("inf")] * 3
    mx = [float("-inf")] * 3
    with _PREBAKED_OPEN_FILE(str(obj_path), "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            mn[0] = min(mn[0], x); mn[1] = min(mn[1], y); mn[2] = min(mn[2], z)
            mx[0] = max(mx[0], x); mx[1] = max(mx[1], y); mx[2] = max(mx[2], z)
    if not (mn[0] < mx[0]):
        raise ValueError(f"OBJ has no vertices: {obj_path}")
    return tuple(mn), tuple(mx)  # type: ignore[return-value]


def iter_prebaked_parts(prebaked_dir: str | Path) -> List[str]:
    d = Path(prebaked_dir)
    return [str(p) for p in sorted(d.glob("*.obj"))]


def prebaked_union_bounds(prebaked_dir: str | Path) -> Tuple[Vec3, Vec3]:
    parts = iter_prebaked_parts(prebaked_dir)
    if not parts:
        raise FileNotFoundError(f"no prebaked parts in {prebaked_dir}")
    union_min = [float("inf")] * 3
    union_max = [float("-inf")] * 3
    for part in parts:
        mn, mx = obj_bounds(part)
        for i in range(3):
            if mn[i] < union_min[i]:
                union_min[i] = mn[i]
            if mx[i] > union_max[i]:
                union_max[i] = mx[i]
    return tuple(union_min), tuple(union_max)  # type: ignore[return-value]


def make_aabb_collision_shape(
    cli: int,
    *,
    half_extents: Vec3,
    frame_position: Vec3,
) -> int:
    return p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=list(half_extents),
        collisionFramePosition=list(frame_position),
        physicsClientId=cli,
    )


def _srgb(c: float) -> float:
    c = 0.0 if c < 0.0 else (1.0 if c > 1.0 else c)
    return 1.055 * (c ** (1.0 / 2.4)) - 0.055 if c > 0.0031308 else 12.92 * c


def _part_rgba(obj_path: str | Path) -> List[float]:
    mtl_path = Path(obj_path).with_suffix(".mtl")
    if not mtl_path.exists():
        return [1.0, 1.0, 1.0, 1.0]
    kd = None
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            low = line.strip().lower()
            if low.startswith("map_kd"):
                return [1.0, 1.0, 1.0, 1.0]
            if low.startswith("kd ") and kd is None:
                vals = line.split()
                if len(vals) >= 4:
                    kd = [float(vals[1]), float(vals[2]), float(vals[3])]
    if kd is None:
        return [1.0, 1.0, 1.0, 1.0]
    return [_srgb(kd[0]), _srgb(kd[1]), _srgb(kd[2]), 1.0]


def spawn_split_material_mesh(
    cli: int,
    *,
    prebaked_dir: str | Path,
    base_position: Vec3,
    base_orientation,
    collision_id: int,
    double_sided: bool = True,
) -> List[int]:
    parts = iter_prebaked_parts(prebaked_dir)
    if not parts:
        raise ValueError(f"no prebaked parts in {prebaked_dir}")

    flags = (
        p.VISUAL_SHAPE_DOUBLE_SIDED
        if double_sided and hasattr(p, "VISUAL_SHAPE_DOUBLE_SIDED")
        else 0
    )

    uids: list[int] = []
    for index, part_path in enumerate(parts):
        rgba = _part_rgba(part_path)
        if flags:
            visual_id = p.createVisualShape(
                p.GEOM_MESH,
                fileName=part_path,
                meshScale=[1.0, 1.0, 1.0],
                rgbaColor=rgba,
                flags=flags,
                physicsClientId=cli,
            )
        else:
            visual_id = p.createVisualShape(
                p.GEOM_MESH,
                fileName=part_path,
                meshScale=[1.0, 1.0, 1.0],
                rgbaColor=rgba,
                physicsClientId=cli,
            )
        uid = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_id if index == 0 else -1,
            baseVisualShapeIndex=visual_id,
            basePosition=list(base_position),
            baseOrientation=list(base_orientation),
            physicsClientId=cli,
        )
        uids.append(int(uid))
    return uids
