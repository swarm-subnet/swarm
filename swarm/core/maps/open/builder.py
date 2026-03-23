"""Builders for the Type 2 open-world benchmark map."""

from __future__ import annotations

import math
import os
import random
from typing import Optional, Tuple

import pybullet as p

from swarm.constants import (
    MAX_ATTEMPTS_PER_OBS,
    TYPE_2_HEIGHT_SCALE,
    TYPE_2_N_OBSTACLES,
    TYPE_2_SAFE_ZONE,
    TYPE_2_WORLD_RANGE,
)

_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))), "assets")
_TERRAIN_CACHE_DIR = os.path.join(_ASSETS_DIR, "open_terrain")
_TERRAIN_TEXTURE_RES = 512
_TERRAIN_SIZE = 80.0
_TERRAIN_RES = 110
_TERRAIN_AMPLITUDE = 1.5
_TERRAIN_FREQUENCY = 0.05
_TERRAIN_OCTAVES = 6
_TERRAIN_FLAT_CENTER = 10.0
_TERRAIN_MESH_VERSION = 2

_CLI_TEX_CACHE: dict = {}


# ---------------------------------------------------------------------------
# Fractal noise
# ---------------------------------------------------------------------------
def _hash_noise(x: float, y: float, seed: int) -> float:
    ix, iy = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - ix, y - iy
    fx = fx * fx * (3.0 - 2.0 * fx)
    fy = fy * fy * (3.0 - 2.0 * fy)
    a = ((ix * 127 + iy * 311 + seed * 8349) * 43758) & 0x7FFFFFFF
    b = (((ix + 1) * 127 + iy * 311 + seed * 8349) * 43758) & 0x7FFFFFFF
    c = ((ix * 127 + (iy + 1) * 311 + seed * 8349) * 43758) & 0x7FFFFFFF
    d = (((ix + 1) * 127 + (iy + 1) * 311 + seed * 8349) * 43758) & 0x7FFFFFFF
    va, vb = (a % 10000) / 10000.0, (b % 10000) / 10000.0
    vc, vd = (c % 10000) / 10000.0, (d % 10000) / 10000.0
    return va + (vb - va) * fx + (vc - va) * fy + (va - vb - vc + vd) * fx * fy


def _fbm(x: float, y: float, seed: int, octaves: int = 5, gain: float = 0.5) -> float:
    value = 0.0
    amp = 1.0
    freq = 1.0
    for i in range(octaves):
        value += amp * _hash_noise(x * freq, y * freq, seed + i * 1337)
        amp *= gain
        freq *= 2.0
    return value


_FBM_MAX = sum(0.5 ** i for i in range(_TERRAIN_OCTAVES))


def _terrain_z(x: float, y: float, seed: int) -> float:
    h = _fbm(x * _TERRAIN_FREQUENCY, y * _TERRAIN_FREQUENCY, seed,
             octaves=_TERRAIN_OCTAVES, gain=0.5)
    h = (h / _FBM_MAX - 0.5) * 2.0 * _TERRAIN_AMPLITUDE
    dist = math.sqrt(x * x + y * y)
    if dist < _TERRAIN_FLAT_CENTER:
        h *= (dist / _TERRAIN_FLAT_CENTER) ** 2
    return h


# ---------------------------------------------------------------------------
# Terrain mesh generation
# ---------------------------------------------------------------------------
def _terrain_obj_path(seed: int) -> str:
    os.makedirs(_TERRAIN_CACHE_DIR, exist_ok=True)
    return os.path.join(_TERRAIN_CACHE_DIR, f"open_terrain_v{_TERRAIN_MESH_VERSION}_s{seed}.obj")


def _generate_terrain_obj(seed: int) -> str:
    path = _terrain_obj_path(seed)
    if os.path.exists(path):
        return path

    step = _TERRAIN_SIZE / _TERRAIN_RES
    with open(path, "w") as f:
        for i in range(_TERRAIN_RES + 1):
            for j in range(_TERRAIN_RES + 1):
                x = -_TERRAIN_SIZE / 2 + j * step
                y = -_TERRAIN_SIZE / 2 + i * step
                z = _terrain_z(x, y, seed)
                u = j / _TERRAIN_RES
                v = i / _TERRAIN_RES
                f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
                f.write(f"vt {u:.5f} {v:.5f}\n")
        for i in range(_TERRAIN_RES):
            for j in range(_TERRAIN_RES):
                v0 = i * (_TERRAIN_RES + 1) + j + 1
                v1 = v0 + 1
                v2 = v0 + (_TERRAIN_RES + 1)
                v3 = v2 + 1
                f.write(f"f {v0}/{v0} {v1}/{v1} {v3}/{v3}\n")
                f.write(f"f {v0}/{v0} {v3}/{v3} {v2}/{v2}\n")
    return path


# ---------------------------------------------------------------------------
# Grass texture generation
# ---------------------------------------------------------------------------
def _texture_path() -> str:
    os.makedirs(_TERRAIN_CACHE_DIR, exist_ok=True)
    return os.path.join(_TERRAIN_CACHE_DIR, "open_grass.bmp")


def _clamp_u8(v: float) -> int:
    return max(0, min(255, int(round(v))))


def _generate_grass_texture() -> str:
    path = _texture_path()
    if os.path.exists(path):
        return path

    w = h = _TERRAIN_TEXTURE_RES
    rng = random.Random(9421)
    data = bytearray(w * h * 3)

    for y_px in range(h):
        for x_px in range(w):
            idx = (y_px * w + x_px) * 3
            nx, ny = x_px / w * 8.0, y_px / h * 8.0
            base = _fbm(nx, ny, 9421, octaves=4, gain=0.45)
            detail = _fbm(nx * 3, ny * 3, 9521, octaves=3, gain=0.4)
            t = base * 0.7 + detail * 0.3
            r = 65 + t * 45 + (detail - 0.5) * 20
            g = 110 + t * 55 + (detail - 0.5) * 25
            b = 45 + t * 30 + (detail - 0.5) * 15
            data[idx + 0] = _clamp_u8(r)
            data[idx + 1] = _clamp_u8(g)
            data[idx + 2] = _clamp_u8(b)

    for _ in range(20):
        cx, cy = rng.uniform(0, w - 1), rng.uniform(0, h - 1)
        radius = rng.uniform(15, 45)
        dirt_r = rng.uniform(105, 130)
        dirt_g = rng.uniform(95, 115)
        dirt_b = rng.uniform(70, 90)
        min_x = max(0, int(cx - radius - 1))
        max_x = min(w - 1, int(cx + radius + 1))
        min_y = max(0, int(cy - radius - 1))
        max_y = min(h - 1, int(cy + radius + 1))
        inv_r2 = 1.0 / max(1.0, radius * radius)
        for py in range(min_y, max_y + 1):
            dy = py - cy
            for px in range(min_x, max_x + 1):
                dx = px - cx
                d2 = (dx * dx + dy * dy) * inv_r2
                if d2 >= 1.0:
                    continue
                blend = ((1.0 - d2) ** 2.5) * rng.uniform(0.15, 0.45)
                i = (py * w + px) * 3
                data[i + 0] = _clamp_u8(data[i + 0] * (1 - blend) + dirt_r * blend)
                data[i + 1] = _clamp_u8(data[i + 1] * (1 - blend) + dirt_g * blend)
                data[i + 2] = _clamp_u8(data[i + 2] * (1 - blend) + dirt_b * blend)

    _write_bmp24(path, w, h, data)
    return path


def _write_bmp24(path: str, width: int, height: int, rgb_data: bytearray) -> None:
    row_stride = width * 3
    row_pad = (4 - (row_stride % 4)) % 4
    image_size = (row_stride + row_pad) * height
    file_size = 54 + image_size
    with open(path, "wb") as f:
        f.write(b"BM")
        f.write(file_size.to_bytes(4, "little"))
        f.write((0).to_bytes(4, "little"))
        f.write((54).to_bytes(4, "little"))
        f.write((40).to_bytes(4, "little"))
        f.write(width.to_bytes(4, "little", signed=True))
        f.write(height.to_bytes(4, "little", signed=True))
        f.write((1).to_bytes(2, "little"))
        f.write((24).to_bytes(2, "little"))
        f.write((0).to_bytes(4, "little"))
        f.write(image_size.to_bytes(4, "little"))
        f.write((2835).to_bytes(4, "little"))
        f.write((2835).to_bytes(4, "little"))
        f.write((0).to_bytes(4, "little"))
        f.write((0).to_bytes(4, "little"))
        pad = b"\x00" * row_pad
        for y in range(height - 1, -1, -1):
            row = y * row_stride
            for x in range(width):
                idx = row + x * 3
                f.write(bytes((rgb_data[idx + 2], rgb_data[idx + 1], rgb_data[idx])))
            if row_pad:
                f.write(pad)


def _load_texture(cli: int) -> Optional[int]:
    if cli in _CLI_TEX_CACHE:
        return _CLI_TEX_CACHE[cli]
    tex_path = _generate_grass_texture()
    try:
        tex_id = p.loadTexture(tex_path, physicsClientId=cli)
    except Exception:
        tex_id = None
    _CLI_TEX_CACHE[cli] = tex_id
    return tex_id


# ---------------------------------------------------------------------------
# Terrain spawning
# ---------------------------------------------------------------------------
def _spawn_terrain(cli: int, seed: int) -> None:
    obj_path = _generate_terrain_obj(seed)
    kwargs = {}
    if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH"):
        kwargs["flags"] = p.GEOM_FORCE_CONCAVE_TRIMESH
    col = p.createCollisionShape(
        p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1],
        physicsClientId=cli, **kwargs,
    )
    vis = p.createVisualShape(
        p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1],
        rgbaColor=[1.0, 1.0, 1.0, 1.0], specularColor=[0.02, 0.02, 0.02],
        physicsClientId=cli,
    )
    body = p.createMultiBody(
        baseMass=0.0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
        basePosition=[0, 0, 0], physicsClientId=cli,
    )
    tex_id = _load_texture(cli)
    if tex_id is not None:
        p.changeVisualShape(
            body, -1, textureUniqueId=tex_id,
            rgbaColor=[1, 1, 1, 1], specularColor=[0.02, 0.02, 0.02],
            physicsClientId=cli,
        )


# ---------------------------------------------------------------------------
# Obstacles (kept for compatibility, currently TYPE_2_N_OBSTACLES = 0)
# ---------------------------------------------------------------------------
def _add_box(cli: int, pos, size, yaw) -> None:
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[s / 2 for s in size], physicsClientId=cli
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[s / 2 for s in size],
        rgbaColor=[0.2, 0.6, 0.8, 1.0],
        physicsClientId=cli,
    )
    quat = p.getQuaternionFromEuler([0, 0, yaw])
    p.createMultiBody(
        0,
        col,
        vis,
        basePosition=pos,
        baseOrientation=quat,
        physicsClientId=cli,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_open_world(
    cli: int,
    seed: int,
    start: Optional[Tuple[float, float, float]] = None,
    goal: Optional[Tuple[float, float, float]] = None,
) -> None:
    _spawn_terrain(cli, seed)

    rng = random.Random(seed)
    sx = sy = gx = gy = None
    if start is not None:
        sx, sy, _ = start
    if goal is not None:
        gx, gy, _ = goal

    placed = 0
    placed_obstacles: list[tuple[float, float, float]] = []
    min_obstacle_distance = 0.6

    while placed < TYPE_2_N_OBSTACLES:
        for _ in range(MAX_ATTEMPTS_PER_OBS):
            kind = rng.choice(["wall", "pillar", "box"])
            x = rng.uniform(-TYPE_2_WORLD_RANGE, TYPE_2_WORLD_RANGE)
            y = rng.uniform(-TYPE_2_WORLD_RANGE, TYPE_2_WORLD_RANGE)
            yaw = rng.uniform(0, math.pi)

            if kind == "box":
                sx_len, sy_len, sz_len = (rng.uniform(1, 4) for _ in range(3))
                sz_len *= TYPE_2_HEIGHT_SCALE
                obj_r = math.hypot(sx_len / 2, sy_len / 2)
            elif kind == "wall":
                length = rng.uniform(5, 15)
                height = rng.uniform(2, 5) * TYPE_2_HEIGHT_SCALE
                sx_len, sy_len, sz_len = length, 0.3, height
                obj_r = length / 2.0
            else:
                r = rng.uniform(0.3, 1.0)
                h = rng.uniform(2, 7) * TYPE_2_HEIGHT_SCALE
                sx_len = sy_len = r * 2
                sz_len = h
                obj_r = r

            def _violates_zone(cx, cy):
                if cx is None:
                    return False
                required_clearance = obj_r + TYPE_2_SAFE_ZONE + 0.5
                return math.hypot(x - cx, y - cy) < required_clearance

            if _violates_zone(sx, sy) or _violates_zone(gx, gy):
                continue

            obstacle_collision = False
            for prev_x, prev_y, prev_r in placed_obstacles:
                distance = math.hypot(x - prev_x, y - prev_y)
                base_distance = obj_r + prev_r + min_obstacle_distance
                if obj_r > 2.0 or prev_r > 2.0:
                    base_distance += 0.5
                if distance < base_distance:
                    obstacle_collision = True
                    break
            if obstacle_collision:
                continue

            if kind == "box":
                _add_box(cli, [x, y, sz_len / 2], [sx_len, sy_len, sz_len], yaw)
            elif kind == "wall":
                col = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=[sx_len / 2, sy_len / 2, sz_len / 2],
                    physicsClientId=cli,
                )
                vis = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=[sx_len / 2, sy_len / 2, sz_len / 2],
                    rgbaColor=[0.9, 0.8, 0.1, 1.0],
                    physicsClientId=cli,
                )
                quat = p.getQuaternionFromEuler([0, 0, yaw])
                p.createMultiBody(
                    0,
                    col,
                    vis,
                    basePosition=[x, y, sz_len / 2],
                    baseOrientation=quat,
                    physicsClientId=cli,
                )
            else:
                col = p.createCollisionShape(
                    p.GEOM_CYLINDER,
                    radius=obj_r,
                    height=sz_len,
                    physicsClientId=cli,
                )
                vis = p.createVisualShape(
                    p.GEOM_CYLINDER,
                    radius=obj_r,
                    length=sz_len,
                    rgbaColor=[0.8, 0.2, 0.2, 1.0],
                    physicsClientId=cli,
                )
                p.createMultiBody(
                    0,
                    col,
                    vis,
                    basePosition=[x, y, sz_len / 2],
                    physicsClientId=cli,
                )

            placed_obstacles.append((x, y, obj_r))
            placed += 1
            break
        else:
            if placed < TYPE_2_N_OBSTACLES * 0.7:
                min_obstacle_distance = max(0.8, min_obstacle_distance - 0.1)
            break
