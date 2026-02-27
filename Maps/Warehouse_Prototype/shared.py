"""
Shared utilities for the warehouse map system.

Extracted from warehouse.py, factory_map.py, and office.py to eliminate
code duplication. Contains CameraController, MeshKitLoader, and common
asset helpers used across all three map modules.
"""

import math
import os
import json

import pybullet as p


                                                                              
                  
                                                                              
                                               
MESH_UP_FIX_RPY = (math.pi / 2.0, 0.0, 0.0)
UNIFORM_SPECULAR_COLOR = (0.0, 0.0, 0.0)


                                                                              
                   
                                                                              
class CameraController:
    """First-person-style camera for PyBullet debug visualizer.

    Each map module can pass its own default starting position/orientation.
    """

    def __init__(
        self,
        x=0.0,
        y=0.0,
        z=10.0,
        yaw=0.0,
        pitch=-45.0,
        dist=0.1,
        speed=0.125,
        mouse_sensitivity=0.5,
    ):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.pitch = pitch
        self.dist = dist
        self.speed = speed
        self.mouse_sensitivity = mouse_sensitivity
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        self.lmb_held = False
        self.update_camera()

    def update(self, keys=None):
        if keys is None:
            keys = p.getKeyboardEvents()
        mouse = p.getMouseEvents()
        dx = 0
        dy = 0

        for e in mouse:
            if e[0] == 1:
                if self.lmb_held:
                    dx = e[1] - self.last_mouse_x
                    dy = e[2] - self.last_mouse_y
                self.last_mouse_x = e[1]
                self.last_mouse_y = e[2]
            if e[0] == 2 and e[3] == 0:
                self.lmb_held = (e[4] == 3 or e[4] == 1)
                if self.lmb_held:
                    self.last_mouse_x = e[1]
                    self.last_mouse_y = e[2]

        if self.lmb_held and (dx or dy):
            self.yaw -= dx * self.mouse_sensitivity
            self.pitch -= dy * self.mouse_sensitivity
            self.pitch = max(-89, min(89, self.pitch))

        move_speed = self.speed * (3.0 if keys.get(p.B3G_SHIFT, 0) else 1.0)
        rad_yaw = math.radians(self.yaw)
        f_x, f_y = -math.sin(rad_yaw), math.cos(rad_yaw)
        r_x, r_y = math.cos(rad_yaw), math.sin(rad_yaw)

        fwd = (1 if keys.get(ord("w"), 0) else 0) - (1 if keys.get(ord("s"), 0) else 0)
        right = (1 if keys.get(ord("d"), 0) else 0) - (1 if keys.get(ord("a"), 0) else 0)
        up = (1 if keys.get(ord("e"), 0) else 0) - (1 if keys.get(ord("q"), 0) else 0)

        self.x += (f_x * fwd + r_x * right) * move_speed
        self.y += (f_y * fwd + r_y * right) * move_speed
        self.z += up * move_speed
        self.update_camera()

    def update_camera(self):
        p.resetDebugVisualizerCamera(self.dist, self.yaw, self.pitch, [self.x, self.y, self.z])


                                                                              
                 
                                                                              
class MeshKitLoader:
    """Loads OBJ models from a Kenney-style asset kit.

    Handles vertex parsing, bounding-box caching, shape caching,
    and spawning into the PyBullet world with Y-up → Z-up correction.
    """

    def __init__(self, obj_dir, texture_path):
        self.obj_dir = obj_dir
        self.texture_path = texture_path
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
        return (
            max_v[0] - min_v[0],
            max_v[1] - min_v[1],
            max_v[2] - min_v[2],
        )

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
                    **kwargs,
                )
                self.collision_shape_cache[collision_key] = collision_id
        else:
            collision_id = -1

        return visual_id, collision_id

    def _ensure_texture(self):
        if self.texture_id is None:
            self.texture_id = p.loadTexture(self.texture_path.replace("\\", "/"))
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
        )
        if use_texture:
            p.changeVisualShape(
                body_id,
                -1,
                rgbaColor=list(rgba),
                textureUniqueId=self._ensure_texture(),
                specularColor=list(UNIFORM_SPECULAR_COLOR),
            )
        else:
            p.changeVisualShape(
                body_id,
                -1,
                rgbaColor=list(rgba),
                textureUniqueId=-1,
                specularColor=list(UNIFORM_SPECULAR_COLOR),
            )
        return body_id


                                                                              
               
                                                                              
def first_existing_path(candidates):
    """Return the first path from *candidates* that exists on disk, or None."""
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def normalize_mtl_texture_paths(obj_dir):
    """Normalize broken absolute map_Kd paths to local kit texture path."""
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
    except OSError:
        pass
    except Exception:
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
