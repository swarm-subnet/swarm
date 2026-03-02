"""
Shared utilities for the warehouse map system.
MeshKitLoader handles OBJ model loading, caching, and spawning with Y-up → Z-up correction.
"""

import json
import math
import os

import pybullet as p

from .constants import MESH_UP_FIX_RPY, UNIFORM_SPECULAR_COLOR


# ---------------------------------------------------------------------------
# MeshKitLoader
# ---------------------------------------------------------------------------
class MeshKitLoader:
    """Loads OBJ models from a Kenney-style asset kit.

    Handles vertex parsing, bounding-box caching, shape caching,
    and spawning into the PyBullet world with Y-up → Z-up correction.
    """

    def __init__(self, obj_dir, texture_path, cli=0):
        self.obj_dir = obj_dir
        self.texture_path = texture_path
        self.cli = cli
        self.asset_path_cache = {}
        self.vertices_cache = {}
        self.bounds_cache = {}
        self.spawn_basis_cache = {}
        self.yaw_cache = {}
        self.visual_shape_cache = {}
        self.collision_shape_cache = {}
        self.texture_id = None
        self.up_fix_quat = p.getQuaternionFromEuler(MESH_UP_FIX_RPY)

    def _asset_path(self, model_name):
        cached = self.asset_path_cache.get(model_name)
        if cached is not None:
            return cached
        path = os.path.join(self.obj_dir, model_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}")
        self.asset_path_cache[model_name] = path
        return path

    def _scale_xyz(self, scale):
        if isinstance(scale, (tuple, list)):
            if len(scale) != 3:
                raise ValueError("scale tuple/list must have exactly 3 components")
            return (float(scale[0]), float(scale[1]), float(scale[2]))
        s = float(scale)
        return (s, s, s)

    def _parse_vertices(self, model_name):
        if model_name in self.vertices_cache:
            return self.vertices_cache[model_name]
        vertices = []
        path = self._asset_path(model_name)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    _, xs, ys, zs = line.split()[:4]
                    x = float(xs)
                    y = float(ys)
                    z = float(zs)
                    vertices.append((x, -z, y))
        if not vertices:
            raise ValueError(f"No vertices found in {path}")
        self.vertices_cache[model_name] = vertices
        return vertices

    def _bounds(self, model_name, scale):
        sxyz = self._scale_xyz(scale)
        key = (model_name, sxyz)
        if key in self.bounds_cache:
            return self.bounds_cache[key]
        verts = self._parse_vertices(model_name)
        transformed = [(v[0] * sxyz[0], v[1] * sxyz[1], v[2] * sxyz[2]) for v in verts]
        min_v = [min(v[i] for v in transformed) for i in range(3)]
        max_v = [max(v[i] for v in transformed) for i in range(3)]
        self.bounds_cache[key] = (min_v, max_v)
        return min_v, max_v

    def model_size(self, model_name, scale):
        min_v, max_v = self._bounds(model_name, scale)
        return (max_v[0] - min_v[0], max_v[1] - min_v[1], max_v[2] - min_v[2])

    def _shape_ids(self, model_name, scale, with_collision, double_sided=False):
        sxyz = self._scale_xyz(scale)
        visual_key = (model_name, sxyz, bool(double_sided))
        visual_id = self.visual_shape_cache.get(visual_key)

        mesh_path = self._asset_path(model_name).replace("\\", "/")
        if visual_id is None:
            visual_kwargs = {}
            if double_sided and hasattr(p, "VISUAL_SHAPE_DOUBLE_SIDED"):
                visual_kwargs["flags"] = p.VISUAL_SHAPE_DOUBLE_SIDED
            visual_id = p.createVisualShape(
                p.GEOM_MESH,
                fileName=mesh_path,
                meshScale=[sxyz[0], sxyz[1], sxyz[2]],
                rgbaColor=[1.0, 1.0, 1.0, 1.0],
                visualFrameOrientation=self.up_fix_quat,
                physicsClientId=self.cli,
                **visual_kwargs,
            )
            self.visual_shape_cache[visual_key] = visual_id

        if with_collision:
            collision_key = (model_name, sxyz)
            collision_id = self.collision_shape_cache.get(collision_key)
            if collision_id is None:
                kwargs = {}
                if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH"):
                    kwargs["flags"] = p.GEOM_FORCE_CONCAVE_TRIMESH
                collision_id = p.createCollisionShape(
                    p.GEOM_MESH,
                    fileName=mesh_path,
                    meshScale=[sxyz[0], sxyz[1], sxyz[2]],
                    collisionFrameOrientation=self.up_fix_quat,
                    physicsClientId=self.cli,
                    **kwargs,
                )
                self.collision_shape_cache[collision_key] = collision_id
        else:
            collision_id = -1

        return visual_id, collision_id

    def _ensure_texture(self):
        if self.texture_id is None and self.texture_path:
            self.texture_id = p.loadTexture(
                self.texture_path.replace("\\", "/"),
                physicsClientId=self.cli,
            )
        return self.texture_id

    def model_min_z(self, model_name, scale):
        min_v, _ = self._bounds(model_name, scale)
        return min_v[2]

    def _spawn_basis(self, model_name, scale):
        sxyz = self._scale_xyz(scale)
        key = (model_name, sxyz)
        cached = self.spawn_basis_cache.get(key)
        if cached is not None:
            return cached
        min_v, max_v = self._bounds(model_name, sxyz)
        min_z = min_v[2]
        center_x = (min_v[0] + max_v[0]) * 0.5
        center_y = (min_v[1] + max_v[1]) * 0.5
        cached = (min_z, center_x, center_y)
        self.spawn_basis_cache[key] = cached
        return cached

    def _yaw_components(self, yaw_deg):
        yaw_key = float(yaw_deg)
        cached = self.yaw_cache.get(yaw_key)
        if cached is not None:
            return cached
        yaw_rad = math.radians(yaw_key)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        yaw_quat = p.getQuaternionFromEuler((0.0, 0.0, yaw_rad))
        cached = (yaw_rad, cos_y, sin_y, yaw_quat)
        self.yaw_cache[yaw_key] = cached
        return cached

    def spawn(
        self,
        model_name,
        x,
        y,
        yaw_deg,
        floor_z,
        scale,
        extra_z=0.0,
        with_collision=True,
        use_texture=True,
        rgba=(1.0, 1.0, 1.0, 1.0),
        double_sided=False,
    ):
        min_z, center_x, center_y = self._spawn_basis(model_name, scale)
        yaw_rad, cos_y, sin_y, yaw_quat = self._yaw_components(yaw_deg)

        offset_x = center_x * cos_y - center_y * sin_y
        offset_y = center_x * sin_y + center_y * cos_y

        spawn_x = x - offset_x
        spawn_y = y - offset_y
        spawn_z = floor_z - min_z + extra_z

        visual_id, collision_id = self._shape_ids(model_name, scale, with_collision, double_sided=double_sided)
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=[spawn_x, spawn_y, spawn_z],
            baseOrientation=yaw_quat,
            useMaximalCoordinates=True,
            physicsClientId=self.cli,
        )
        tex_id = self._ensure_texture() if use_texture else -1
        p.changeVisualShape(
            body_id,
            -1,
            rgbaColor=list(rgba),
            textureUniqueId=tex_id if tex_id is not None else -1,
            specularColor=list(UNIFORM_SPECULAR_COLOR),
            physicsClientId=self.cli,
        )
        return body_id


# ---------------------------------------------------------------------------
# Path utilities
# ---------------------------------------------------------------------------
def first_existing_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def normalize_mtl_texture_paths(obj_dir):
    try:
        names = os.listdir(obj_dir)
    except OSError:
        return
    mtl_names = [name for name in names if name.lower().endswith(".mtl")]
    if not mtl_names:
        return

    stamp_path = os.path.join(obj_dir, ".mtl_normalize_stamp_v1.json")
    snapshot = []
    for name in sorted(mtl_names):
        mtl_path = os.path.join(obj_dir, name)
        try:
            snapshot.append([name, int(os.path.getmtime(mtl_path)), int(os.path.getsize(mtl_path))])
        except OSError:
            snapshot.append([name, -1, -1])

    try:
        with open(stamp_path, "r", encoding="utf-8", errors="ignore") as f:
            stamp = json.load(f) or {}
        if stamp.get("snapshot") == snapshot:
            return
    except (OSError, Exception):
        pass

    for name in mtl_names:
        mtl_path = os.path.join(obj_dir, name)
        try:
            with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except OSError:
            continue

        changed = False
        out = []
        for line in lines:
            stripped = line.strip()
            if stripped.lower().startswith("map_kd "):
                tex_path = stripped.split(None, 1)[1].strip().replace("\\", "/")
                if tex_path.lower().endswith("colormap.png") and tex_path != "Textures/colormap.png":
                    line = "map_Kd Textures/colormap.png\n"
                    changed = True
            out.append(line)

        if not changed:
            continue
        try:
            with open(mtl_path, "w", encoding="utf-8", errors="ignore") as f:
                f.writelines(out)
        except OSError:
            continue

    try:
        with open(stamp_path, "w", encoding="utf-8", errors="ignore") as f:
            json.dump({"snapshot": snapshot}, f, ensure_ascii=True, separators=(",", ":"))
    except OSError:
        pass
