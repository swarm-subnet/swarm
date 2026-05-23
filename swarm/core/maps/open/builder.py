"""Builders for the Type 2 open-world benchmark map."""

from __future__ import annotations

import math
import os
import random
from collections import OrderedDict
from typing import Optional, Tuple

import pybullet as p

from swarm.constants import (
    MAX_ATTEMPTS_PER_OBS,
    TYPE_2_HEIGHT_SCALE,
    TYPE_2_N_OBSTACLES,
    TYPE_2_SAFE_ZONE,
    TYPE_2_WORLD_RANGE,
)

_PACKAGE_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_ASSETS_DIR = os.path.join(_PACKAGE_DIR, "assets")
_STATE_DIR = os.path.join(_PACKAGE_DIR, "state")
_TERRAIN_CACHE_DIR = os.path.join(_STATE_DIR, "open_terrain")
_OPEN_PROP_CACHE_DIR = os.path.join(_STATE_DIR, "open_prop_visuals")
_OPEN_MANNEQUIN_BLENDER_PATH = os.path.join(
    _ASSETS_DIR,
    "maps",
    "custom",
    "people",
    "open_mannequin",
    "person_visual.obj",
)
_OPEN_MANNEQUIN_MAKEHUMAN_PATH = os.path.join(
    _ASSETS_DIR,
    "maps",
    "custom",
    "people",
    "open_mannequin_raw",
    "mannequin_a_raw.obj",
)
_TERRAIN_TEXTURE_RES = 512
_TERRAIN_SIZE = 80.0
_TERRAIN_RES = 110
_TERRAIN_AMPLITUDE = 1.5
_TERRAIN_FREQUENCY = 0.05
_TERRAIN_OCTAVES = 6
_TERRAIN_FLAT_CENTER = 10.0
_TERRAIN_MESH_VERSION = 2
_TERRAIN_BASE_RGBA = [0.72, 0.90, 0.62, 1.0]
_TERRAIN_SPECULAR = [0.0, 0.0, 0.0]
_MANNEQUIN_RGBA = [0.82, 0.82, 0.82, 1.0]
_MANNEQUIN_SPECULAR = [0.03, 0.03, 0.03]
_MAKEHUMAN_MANNEQUIN_RGBA = [1.0, 1.0, 1.0, 1.0]
_MANNEQUIN_GROUND_EPS = 0.005


_CLI_TEX_CACHE: dict = {}
_OBJ_BOUNDS_CACHE: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
_OBJ_MATERIAL_PARTS_CACHE: dict[str, tuple[dict, ...]] = {}


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
        if isinstance(tex_id, int) and tex_id < 0:
            tex_id = None
    except Exception:
        tex_id = None
    _CLI_TEX_CACHE[cli] = tex_id
    return tex_id


def _obj_bounds(obj_path: str) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    cached = _OBJ_BOUNDS_CACHE.get(obj_path)
    if cached is not None:
        return cached

    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)

    if min_x == float("inf"):
        raise ValueError(f"OBJ file has no vertices: {obj_path}")

    bounds = ((min_x, min_y, min_z), (max_x, max_y, max_z))
    _OBJ_BOUNDS_CACHE[obj_path] = bounds
    return bounds


def _safe_token(value: str) -> str:
    token = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return token or "default"


def _obj_mtl_path(obj_path: str) -> Optional[str]:
    obj_dir = os.path.dirname(obj_path)
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.lower().startswith("mtllib "):
                rel = line.split(None, 1)[1].strip()
                mtl_path = os.path.normpath(os.path.join(obj_dir, rel))
                if os.path.exists(mtl_path):
                    return mtl_path
    fallback = os.path.splitext(obj_path)[0] + ".mtl"
    return fallback if os.path.exists(fallback) else None


def _parse_mtl_materials(mtl_path: Optional[str]) -> dict[str, dict]:
    if not mtl_path or not os.path.exists(mtl_path):
        return {}
    materials: dict[str, dict] = {}
    current: Optional[str] = None
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("newmtl "):
                current = line.split(None, 1)[1].strip()
                materials.setdefault(current, {"mtl_lines": [raw]})
                continue
            if current is None:
                continue
            materials[current].setdefault("mtl_lines", []).append(raw)
            if line.lower().startswith("kd "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                materials[current]["rgba"] = [
                    max(0.0, min(1.0, float(parts[1]))),
                    max(0.0, min(1.0, float(parts[2]))),
                    max(0.0, min(1.0, float(parts[3]))),
                    1.0,
                ]
                continue
            if line.lower().startswith("map_kd "):
                rel = line.split(None, 1)[1].strip()
                tex_path = os.path.normpath(os.path.join(os.path.dirname(mtl_path), rel))
                if os.path.exists(tex_path):
                    materials[current]["texture_path"] = tex_path
                    materials[current]["texture_ref"] = rel
    return materials


def _obj_material_parts(obj_path: str) -> tuple[dict, ...]:
    cached = _OBJ_MATERIAL_PARTS_CACHE.get(obj_path)
    if cached is not None:
        return cached

    mtl_path = _obj_mtl_path(obj_path)
    material_info = _parse_mtl_materials(mtl_path)
    vertices: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, ...]] = []
    normals: list[tuple[float, float, float]] = []
    faces_by_mat: "OrderedDict[str, list[list[tuple[int, Optional[int], Optional[int]]]]]" = OrderedDict()
    current_mat = "__default__"
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if line.startswith("vt "):
                parts = line.split()
                texcoords.append(tuple(float(v) for v in parts[1:]))
                continue
            if line.startswith("vn "):
                parts = line.split()
                normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if line.lower().startswith("usemtl "):
                current_mat = line.split(None, 1)[1].strip()
                faces_by_mat.setdefault(current_mat, [])
                continue
            if line.startswith("f "):
                corners: list[tuple[int, Optional[int], Optional[int]]] = []
                for token in line.split()[1:]:
                    chunks = token.split("/")
                    vi = int(chunks[0])
                    if vi < 0:
                        vi = len(vertices) + 1 + vi
                    vi -= 1
                    vti = None
                    if len(chunks) >= 2 and chunks[1]:
                        vti = int(chunks[1])
                        if vti < 0:
                            vti = len(texcoords) + 1 + vti
                        vti -= 1
                    vni = None
                    if len(chunks) >= 3 and chunks[2]:
                        vni = int(chunks[2])
                        if vni < 0:
                            vni = len(normals) + 1 + vni
                        vni -= 1
                    corners.append((vi, vti, vni))
                if len(corners) < 3:
                    continue
                tris = faces_by_mat.setdefault(current_mat, [])
                for i in range(1, len(corners) - 1):
                    tris.append([corners[0], corners[i], corners[i + 1]])

    if not faces_by_mat:
        parts = (
            {
                "material": "__default__",
                "obj_path": obj_path,
                "rgba": [1.0, 1.0, 1.0, 1.0],
            },
        )
        _OBJ_MATERIAL_PARTS_CACHE[obj_path] = parts
        return parts

    cache_dir = os.path.join(
        _OPEN_PROP_CACHE_DIR,
        os.path.splitext(os.path.basename(obj_path))[0],
    )
    os.makedirs(cache_dir, exist_ok=True)

    parts_out: list[dict] = []
    for mat_name, tris in faces_by_mat.items():
        if not tris:
            continue
        safe_mat = _safe_token(mat_name)
        part_obj_path = os.path.join(cache_dir, f"{safe_mat}.obj")
        info = material_info.get(mat_name, {})
        used_v = sorted({corner[0] for tri in tris for corner in tri})
        used_vt = sorted({corner[1] for tri in tris for corner in tri if corner[1] is not None})
        used_vn = sorted({corner[2] for tri in tris for corner in tri if corner[2] is not None})
        remap_v = {old_i: new_i + 1 for new_i, old_i in enumerate(used_v)}
        remap_vt = {old_i: new_i + 1 for new_i, old_i in enumerate(used_vt)}
        remap_vn = {old_i: new_i + 1 for new_i, old_i in enumerate(used_vn)}

        part_mtl_path = os.path.join(cache_dir, f"{safe_mat}.mtl")
        texture_path = info.get("texture_path")
        if texture_path and info.get("texture_ref"):
            texture_ref = os.path.relpath(str(texture_path), cache_dir).replace("\\", "/")
        else:
            texture_ref = ""
        with open(part_mtl_path, "w", encoding="utf-8") as mtl_out:
            if info.get("mtl_lines"):
                for raw in info["mtl_lines"]:
                    stripped = raw.strip()
                    if stripped.lower().startswith("map_kd ") and texture_ref:
                        indent = raw[: len(raw) - len(raw.lstrip())]
                        mtl_out.write(f"{indent}map_Kd {texture_ref}\n")
                    else:
                        mtl_out.write(raw)
            else:
                mtl_out.write(f"newmtl {mat_name}\n")
                rgba = list(info.get("rgba", [1.0, 1.0, 1.0, 1.0]))
                mtl_out.write(f"Kd {rgba[0]:.6f} {rgba[1]:.6f} {rgba[2]:.6f}\n")
                if texture_ref:
                    mtl_out.write(f"map_Kd {texture_ref}\n")

        with open(part_obj_path, "w", encoding="utf-8") as out:
            out.write(f"# split from {os.path.basename(obj_path)} material {mat_name}\n")
            out.write(f"mtllib {os.path.basename(part_mtl_path)}\n")
            out.write(f"usemtl {mat_name}\n")
            for old_i in used_v:
                vx, vy, vz = vertices[old_i]
                out.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for old_i in used_vt:
                tc = texcoords[old_i]
                if len(tc) >= 3:
                    out.write(f"vt {tc[0]:.6f} {tc[1]:.6f} {tc[2]:.6f}\n")
                else:
                    out.write(f"vt {tc[0]:.6f} {tc[1]:.6f}\n")
            for old_i in used_vn:
                nx, ny, nz = normals[old_i]
                out.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
            for tri in tris:
                tokens = []
                for vi, vti, vni in tri:
                    rv = remap_v[vi]
                    rvt = remap_vt.get(vti) if vti is not None else None
                    rvn = remap_vn.get(vni) if vni is not None else None
                    if rvt is not None and rvn is not None:
                        tokens.append(f"{rv}/{rvt}/{rvn}")
                    elif rvt is not None:
                        tokens.append(f"{rv}/{rvt}")
                    elif rvn is not None:
                        tokens.append(f"{rv}//{rvn}")
                    else:
                        tokens.append(f"{rv}")
                out.write("f " + " ".join(tokens) + "\n")

        rgba = list(info.get("rgba", [1.0, 1.0, 1.0, 1.0]))
        parts_out.append(
            {
                "material": mat_name,
                "obj_path": part_obj_path,
                "rgba": rgba,
            }
        )

    if not parts_out:
        parts_out.append(
            {
                "material": "__default__",
                "obj_path": obj_path,
                "rgba": [1.0, 1.0, 1.0, 1.0],
            }
        )

    cached = tuple(parts_out)
    _OBJ_MATERIAL_PARTS_CACHE[obj_path] = cached
    return cached


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
        rgbaColor=_TERRAIN_BASE_RGBA,
        specularColor=_TERRAIN_SPECULAR,
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
            rgbaColor=_TERRAIN_BASE_RGBA,
            specularColor=_TERRAIN_SPECULAR,
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
    sar_mode: bool = True,
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
