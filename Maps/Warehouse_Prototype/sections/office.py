import argparse
import math
import os
import random
import sys
import tempfile
import time
from collections import OrderedDict

import pybullet as p
import pybullet_data

try:
    from shared import (
        MESH_UP_FIX_RPY,
        CameraController,
    )
except ModuleNotFoundError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from shared import (
        MESH_UP_FIX_RPY,
        CameraController,
    )
try:
    from PIL import Image, ImageDraw, ImageOps, ImageChops
except Exception:
    Image = None
    ImageDraw = None
    ImageOps = None
    ImageChops = None


                                                                              
        
                                                                              
FLOOR_SIZE = 12.0
UNIFORM_SCALE = 2.0
ROOM_CENTER = (0.0, 0.0)
SHOW_SHELF_LABELS = False
SCREEN_BRANDING_ENABLED = True
SCREEN_BRANDING_LABEL = "SWARM"
SCREEN_BRANDING_TICKER = "$SWARM"
SCREEN_BRANDING_TAG = "BETA"
SCREEN_TEXTURE_ROTATE_DEG = 180

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
ASSET_PATH = os.path.join(
    PROJECT_ROOT,
    "assets",
    "kenney",
    "kenney_furniture-kit",
    "Models",
    "OBJ format",
)
TEMP_URDF_DIR = os.path.join(tempfile.gettempdir(), "swarm_warehouse_office_urdfs")
SCREEN_LOGO_CANDIDATES = (
    os.path.join(PROJECT_ROOT, "assets", "other_sources", "swarm_brand", "Swarm.png"),
    os.path.join(PROJECT_ROOT, "assets", "other_sources", "swarm_brand", "Swarm_2.png"),
    os.path.join(PROJECT_ROOT, "swarm", "swarm", "assets", "Swarm.png"),
    os.path.join(PROJECT_ROOT, "swarm", "swarm", "assets", "Swarm_2.png"),
    os.path.join(PROJECT_ROOT, "swarm_backup_v1", "swarm", "assets", "Swarm_2.png"),
    os.path.join(PROJECT_ROOT, "swarm_backup_v1", "swarm", "assets", "Swarm.png"),
)
SCREEN_TAO_CANDIDATES = (
    os.path.join(PROJECT_ROOT, "swarm", "swarm", "assets", "tao.png"),
    os.path.join(PROJECT_ROOT, "swarm_backup_v1", "swarm", "assets", "tao.png"),
)


ASSETS = {
    "floor_tile": "floorFull.obj",
    "wall": "wall.obj",
    "wall_door": "wallDoorway.obj",
    "wall_corner": "wallCorner.obj",
    "doorway": "doorwayOpen.obj",
    "meeting_table": "tableCrossCloth.obj",
    "meeting_chair": "chairDesk.obj",
    "entry_coat_rack": "coatRackStanding.obj",
    "entry_plant": "pottedPlant.obj",
    "desk": "desk.obj",
    "desk_corner": "deskCorner.obj",
    "desk_chair": "chairDesk.obj",
    "monitor": "computerScreen.obj",
    "keyboard": "computerKeyboard.obj",
    "mouse": "computerMouse.obj",
    "bookcase_open": "bookcaseOpen.obj",
    "bookcase_open_low": "bookcaseOpenLow.obj",
    "bookcase_closed": "bookcaseClosed.obj",
    "bookcase_closed_doors": "bookcaseClosedDoors.obj",
    "bookcase_wide": "bookcaseClosedWide.obj",
    "books": "books.obj",
    "box": "cardboardBoxClosed.obj",
    "box_open": "cardboardBoxOpen.obj",
    "fridge": "kitchenFridgeSmall.obj",
    "fridge_tall": "kitchenFridge.obj",
    "fridge_large": "kitchenFridgeLarge.obj",
    "cabinet": "kitchenCabinet.obj",
    "cabinet_tv_doors": "cabinetTelevisionDoors.obj",
    "coffee_machine": "kitchenCoffeeMachine.obj",
    "trashcan": "trashcan.obj",
    "side_table": "sideTable.obj",
    "plant_small": "plantSmall1.obj",
    "plant_small_2": "plantSmall2.obj",
    "plant_small_3": "plantSmall3.obj",
    "table_coffee": "tableCoffee.obj",
    "table_coffee_square": "tableCoffeeSquare.obj",
    "lamp_table_round": "lampRoundTable.obj",
    "lamp_table_square": "lampSquareTable.obj",
    "lamp_floor": "lampRoundFloor.obj",
}

WALL_SLOTS = ("north", "east", "south", "west")
WALL_ROLES = ("entry", "workstations", "files", "services")
ENABLE_PERIMETER_WALL_MESHES = True
ENABLE_PERIMETER_WALL_CORNERS = True
ENTRY_WALL_OPENING_MODE = "door_segment"                           
ENTRY_WALL_OPENING_ALONG = 0.0
                                                                              
                                                       
                                                                        
                                          
PERIMETER_WALL_ALONG_SCALE = 1.0
                                                            
PERIMETER_WALL_CORNER_OUTWARD_EPS = 0.0
                                                                    
                                                                  
PERIMETER_WALL_CORNER_JOIN_GAP_M = 0.0
                                                                                  
OFFICE_WALL_FORCE_FLAT_COLOR = True
OFFICE_WALL_FLAT_RGBA = (0.72, 0.74, 0.78, 1.0)
OFFICE_FLOOR_FORCE_FLAT_COLOR = True
OFFICE_FLOOR_FLAT_RGBA = (0.64, 0.66, 0.70, 1.0)

                                                                              
                                                                            
ASSET_FRONT_DEG = {
    "chairDesk.obj": 90.0,
    "desk.obj": 90.0,
    "deskCorner.obj": 90.0,
    "computerScreen.obj": 90.0,
    "computerKeyboard.obj": 90.0,
    "computerMouse.obj": 90.0,
    "bookcaseOpen.obj": 90.0,
    "bookcaseClosed.obj": 90.0,
    "bookcaseClosedDoors.obj": 90.0,
    "bookcaseClosedWide.obj": 90.0,
    "kitchenFridgeSmall.obj": 90.0,
    "kitchenFridge.obj": 90.0,
    "kitchenFridgeLarge.obj": 90.0,
    "kitchenCabinet.obj": 90.0,
    "cabinetTelevisionDoors.obj": 90.0,
    "kitchenCoffeeMachine.obj": 90.0,
    "trashcan.obj": 90.0,
    "pottedPlant.obj": 90.0,
    "plantSmall1.obj": 90.0,
    "plantSmall2.obj": 90.0,
    "plantSmall3.obj": 90.0,
    "coatRackStanding.obj": 90.0,
    "sideTable.obj": 90.0,
    "tableCoffee.obj": 90.0,
    "tableCoffeeSquare.obj": 90.0,
    "lampRoundTable.obj": 90.0,
    "lampSquareTable.obj": 90.0,
    "lampRoundFloor.obj": 90.0,
    "cardboardBoxClosed.obj": 90.0,
    "cardboardBoxOpen.obj": 90.0,
    "bookcaseOpenLow.obj": 90.0,
    "books.obj": 90.0,
}


                                                                              
              
                                                                              
class AssetLoader:
    def __init__(self, asset_dir, temp_dir, uniform_scale):
        self.asset_dir = asset_dir
        self.temp_dir = temp_dir
        self.uniform_scale = uniform_scale
        self.bounds_cache = {}
        self.shelf_levels_cache = {}
        self.material_parts_cache = {}
        self.front_angle_cache = {}
        self.urdf_cache = {}
        self.texture_id_cache = {}
        self.brand_screen_texture_id = None
        os.makedirs(self.temp_dir, exist_ok=True)

    @staticmethod
    def _normalize_scale(scale):
        if isinstance(scale, (tuple, list)):
            if len(scale) != 3:
                raise ValueError("Scale tuple must have 3 values (sx, sy, sz).")
            return (float(scale[0]), float(scale[1]), float(scale[2]))
        s = float(scale)
        return (s, s, s)

    def _asset_path(self, filename):
        return os.path.join(self.asset_dir, filename)

    def _parse_obj_vertices(self, filename):
        path = self._asset_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Asset not found: {path}")

        verts = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        if not verts:
            raise ValueError(f"No vertices found in {path}")
        return verts

    @staticmethod
    def _rotate_y_up_to_z_up(v):
        x, y, z = v
        return (x, -z, y)

    def _mesh_bounds(self, filename, scale):
        scale_xyz = self._normalize_scale(scale)
        key = (filename, scale_xyz)
        if key in self.bounds_cache:
            return self.bounds_cache[key]

        verts = self._parse_obj_vertices(filename)
        transformed = []
        for v in verts:
            rx, ry, rz = self._rotate_y_up_to_z_up(v)
            transformed.append((rx * scale_xyz[0], ry * scale_xyz[1], rz * scale_xyz[2]))

        min_v = [min(v[i] for v in transformed) for i in range(3)]
        max_v = [max(v[i] for v in transformed) for i in range(3)]
        self.bounds_cache[key] = (min_v, max_v)
        return min_v, max_v

    def _parse_obj_vertices_faces(self, filename):
        path = self._asset_path(filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Asset not found: {path}")

        verts = []
        faces = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    parts = line.split()
                    if len(parts) >= 4:
                        verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                elif line.startswith("f "):
                    tokens = line.split()[1:]
                    idx = []
                    for token in tokens:
                        v_idx = token.split("/")[0]
                        if not v_idx:
                            continue
                        i = int(v_idx)
                        if i < 0:
                            i = len(verts) + 1 + i
                        idx.append(i - 1)
                    if len(idx) >= 3:
                        for k in range(1, len(idx) - 1):
                            faces.append((idx[0], idx[k], idx[k + 1]))
        return verts, faces

    def shelf_surface_levels(self, filename, scale=None):
        if scale is None:
            scale = self.uniform_scale
        key = (filename, scale)
        if key in self.shelf_levels_cache:
            return self.shelf_levels_cache[key]

        verts, faces = self._parse_obj_vertices_faces(filename)
        transformed = []
        for v in verts:
            rx, ry, rz = self._rotate_y_up_to_z_up(v)
            transformed.append((rx * scale, ry * scale, rz * scale))

        min_v, max_v = self._mesh_bounds(filename, scale)
        z_min = min_v[2]
        z_max = max_v[2]

        samples = []
        for i0, i1, i2 in faces:
            p0 = transformed[i0]
            p1 = transformed[i1]
            p2 = transformed[i2]
            ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
            vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
            nx = uy * vz - uz * vy
            ny = uz * vx - ux * vz
            nz = ux * vy - uy * vx
            nlen = math.sqrt(nx * nx + ny * ny + nz * nz)
            if nlen < 1e-8:
                continue
            area = 0.5 * nlen
            nz_unit = nz / nlen
            if area < 1e-4 or abs(nz_unit) < 0.92:
                continue
            z = (p0[2] + p1[2] + p2[2]) / 3.0
            samples.append((z, area))

        if not samples:
            self.shelf_levels_cache[key] = []
            return []

        samples.sort(key=lambda t: t[0])
        z_cluster_tol = 0.03
        clusters = []
        for z, area in samples:
            if not clusters or abs(z - clusters[-1][0]) > z_cluster_tol:
                clusters.append([z, area])
            else:
                prev_z, prev_area = clusters[-1]
                new_area = prev_area + area
                new_z = (prev_z * prev_area + z * area) / new_area
                clusters[-1] = [new_z, new_area]

        max_area = max(a for _, a in clusters)
        area_threshold = max_area * 0.22
        peaks = [z for z, a in clusters if a >= area_threshold]
        peaks.sort()

                                                                            
        pair_tol = 0.085
        grouped = []
        for z in peaks:
            if not grouped or abs(z - grouped[-1][-1]) > pair_tol:
                grouped.append([z])
            else:
                grouped[-1].append(z)
        board_tops = [max(g) for g in grouped]

                                                                             
        row_levels = []
        for z in board_tops:
            if z <= z_min + 0.12:
                continue
            if z >= z_max - 0.10:
                continue
            row_levels.append(z)

        self.shelf_levels_cache[key] = row_levels
        return row_levels

    def _create_urdf(self, filename, scale):
        key = (filename, scale)
        if key in self.urdf_cache:
            return self.urdf_cache[key]

        mesh_path = self._asset_path(filename).replace("\\", "/")
        name = os.path.splitext(filename)[0]
        urdf_path = os.path.join(self.temp_dir, f"{name}_s{str(scale).replace('.', '_')}.urdf")

        roll, pitch, yaw = MESH_UP_FIX_RPY
        urdf = f"""<?xml version="1.0" ?>
<robot name="{name}">
  <link name="baseLink">
    <contact>
      <lateral_friction value="1.0"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <origin rpy="{roll} {pitch} {yaw}" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </visual>
    <collision>
      <origin rpy="{roll} {pitch} {yaw}" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_path}" scale="{scale} {scale} {scale}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
        with open(urdf_path, "w", encoding="utf-8") as f:
            f.write(urdf)
        self.urdf_cache[key] = urdf_path
        return urdf_path

    def _create_urdf_for_mesh(self, mesh_path, mesh_tag, scale, rgba):
        scale_xyz = self._normalize_scale(scale)
        mesh_key = mesh_path.replace("\\", "/")
        rgba_key = tuple(round(v, 6) for v in rgba)
        key = (mesh_key, scale_xyz, rgba_key)
        if key in self.urdf_cache:
            return self.urdf_cache[key]

        safe_tag = mesh_tag.replace(" ", "_")
        scale_tag = "_".join(str(v).replace(".", "_").replace("-", "m") for v in scale_xyz)
        urdf_path = os.path.join(
            self.temp_dir,
            f"{safe_tag}_s{scale_tag}_{rgba_key[0]}_{rgba_key[1]}_{rgba_key[2]}.urdf",
        )

        roll, pitch, yaw = MESH_UP_FIX_RPY
        urdf = f"""<?xml version="1.0" ?>
<robot name="{safe_tag}">
  <link name="baseLink">
    <contact>
      <lateral_friction value="1.0"/>
    </contact>
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.0"/>
      <inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
    </inertial>
    <visual>
      <origin rpy="{roll} {pitch} {yaw}" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_key}" scale="{scale_xyz[0]} {scale_xyz[1]} {scale_xyz[2]}"/>
      </geometry>
      <material name="mat">
        <color rgba="{rgba[0]} {rgba[1]} {rgba[2]} 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="{roll} {pitch} {yaw}" xyz="0 0 0"/>
      <geometry>
        <mesh filename="{mesh_key}" scale="{scale_xyz[0]} {scale_xyz[1]} {scale_xyz[2]}"/>
      </geometry>
    </collision>
  </link>
</robot>
"""
        with open(urdf_path, "w", encoding="utf-8") as f:
            f.write(urdf)
        self.urdf_cache[key] = urdf_path
        return urdf_path

    def _sanitize_name(self, text):
        out = []
        for ch in text:
            if ch.isalnum() or ch in ("_", "-"):
                out.append(ch)
            else:
                out.append("_")
        return "".join(out)

    def _retile_obj_uv_single_image(self, obj_path):
                                                                                       
        if not os.path.exists(obj_path):
            return

        v_lines = []
        vn_lines = []
        face_tokens = []

        with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    v_lines.append(line.strip())
                elif line.startswith("vn "):
                    vn_lines.append(line.strip())
                elif line.startswith("f "):
                    toks = line.strip().split()[1:]
                    face_tokens.append(toks)

        if not v_lines or not face_tokens:
            return

        verts = []
        for line in v_lines:
            _, xs, ys, zs = line.split()[:4]
            verts.append((float(xs), float(ys), float(zs)))

        used_vidx = set()
        parsed_faces = []
        for toks in face_tokens:
            parsed = []
            for tok in toks:
                parts = tok.split("/")
                vi = int(parts[0])
                vti = int(parts[1]) if len(parts) > 1 and parts[1] else None
                vni = int(parts[2]) if len(parts) > 2 and parts[2] else None
                parsed.append((vi, vti, vni))
                used_vidx.add(vi)
            parsed_faces.append(parsed)

        if not used_vidx:
            return

        used = [verts[i - 1] for i in sorted(used_vidx)]
        mins = [min(v[i] for v in used) for i in range(3)]
        maxs = [max(v[i] for v in used) for i in range(3)]
        spans = [maxs[i] - mins[i] for i in range(3)]
        axes = sorted(range(3), key=lambda a: spans[a], reverse=True)[:2]
        ax_u, ax_v = axes[0], axes[1]
        span_u = spans[ax_u] if spans[ax_u] > 1e-8 else 1.0
        span_v = spans[ax_v] if spans[ax_v] > 1e-8 else 1.0

        vt_index_by_vi = {}
        vt_lines = []
        for vi in sorted(used_vidx):
            vx, vy, vz = verts[vi - 1]
            coords = (vx, vy, vz)
            u = (coords[ax_u] - mins[ax_u]) / span_u
            v = (coords[ax_v] - mins[ax_v]) / span_v
            v = 1.0 - v
            vt_index_by_vi[vi] = len(vt_lines) + 1
            vt_lines.append(f"vt {u:.6f} {v:.6f}")

        out_lines = []
        out_lines.append("# UV retiled for single-image monitor screen\n")
        for line in v_lines:
            out_lines.append(line + "\n")
        for line in vt_lines:
            out_lines.append(line + "\n")
        for line in vn_lines:
            out_lines.append(line + "\n")

        for face in parsed_faces:
            tokens = []
            for vi, _vti, vni in face:
                new_vti = vt_index_by_vi[vi]
                if vni is None:
                    tokens.append(f"{vi}/{new_vti}")
                else:
                    tokens.append(f"{vi}/{new_vti}/{vni}")
            out_lines.append("f " + " ".join(tokens) + "\n")

        with open(obj_path, "w", encoding="utf-8") as f:
            f.writelines(out_lines)

    def _read_material_usage(self, filename):
        path = self._asset_path(filename)
        mtllib = None
        v_lines = []
        vt_lines = []
        vn_lines = []
        faces_by_mat = OrderedDict()
        current_mat = "__default__"

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("mtllib "):
                    mtllib = line.split(None, 1)[1].strip()
                elif line.startswith("v "):
                    v_lines.append(line)
                elif line.startswith("vt "):
                    vt_lines.append(line)
                elif line.startswith("vn "):
                    vn_lines.append(line)
                elif line.startswith("usemtl "):
                    current_mat = line.split(None, 1)[1].strip()
                    if current_mat not in faces_by_mat:
                        faces_by_mat[current_mat] = []
                elif line.startswith("f "):
                    if current_mat not in faces_by_mat:
                        faces_by_mat[current_mat] = []
                    faces_by_mat[current_mat].append(line)

        return mtllib, v_lines, vt_lines, vn_lines, faces_by_mat

    def _read_mtl_colors(self, mtllib_name):
        colors = {}
        if not mtllib_name:
            return colors

        mtl_path = os.path.join(self.asset_dir, mtllib_name)
        if not os.path.exists(mtl_path):
            return colors

        current = None
        with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("newmtl "):
                    current = line.split(None, 1)[1].strip()
                elif current and line.startswith("Kd "):
                    parts = line.split()
                    if len(parts) >= 4:
                        colors[current] = (
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        )
        return colors

    def _material_parts(self, filename):
        if filename in self.material_parts_cache:
            return self.material_parts_cache[filename]

        mtllib, v_lines, vt_lines, vn_lines, faces_by_mat = self._read_material_usage(filename)
        mtl_colors = self._read_mtl_colors(mtllib)

        if not faces_by_mat:
                                                                
            mesh = self._asset_path(filename).replace("\\", "/")
            parts = [(mesh, (0.8, 0.8, 0.8), "__default__")]
            self.material_parts_cache[filename] = parts
            return parts

                                                                
        if len(faces_by_mat) == 1:
            mat = next(iter(faces_by_mat.keys()))
            mesh = self._asset_path(filename).replace("\\", "/")
            color = mtl_colors.get(mat, (0.8, 0.8, 0.8))
            parts = [(mesh, color, mat)]
            self.material_parts_cache[filename] = parts
            return parts

        src_stem = os.path.splitext(os.path.basename(filename))[0]
        split_dir = os.path.join(self.temp_dir, "_split_obj", src_stem)
        os.makedirs(split_dir, exist_ok=True)

        parts = []
        for mat, face_lines in faces_by_mat.items():
            if not face_lines:
                continue
            safe_mat = self._sanitize_name(mat)
            part_obj_path = os.path.join(split_dir, f"{safe_mat}.obj")

            if not os.path.exists(part_obj_path):
                with open(part_obj_path, "w", encoding="utf-8") as out:
                    out.write(f"# split from {filename} material {mat}\n")
                    out.write(f"g {safe_mat}\n")
                    for line in v_lines:
                        out.write(line)
                    for line in vt_lines:
                        out.write(line)
                    for line in vn_lines:
                        out.write(line)
                    for line in face_lines:
                        out.write(line)
            if filename == ASSETS["monitor"] and mat == "metal":
                self._retile_obj_uv_single_image(part_obj_path)

            color = mtl_colors.get(mat, (0.8, 0.8, 0.8))
            parts.append((part_obj_path.replace("\\", "/"), color, mat))

        if not parts:
            mesh = self._asset_path(filename).replace("\\", "/")
            parts = [(mesh, (0.8, 0.8, 0.8), "__default__")]

        self.material_parts_cache[filename] = parts
        return parts

    def _foreground_bbox_against_corner_bg(self, pil_image):
        rgb = pil_image.convert("RGB")
        corner_rgb = rgb.getpixel((0, 0))
        bg = Image.new("RGB", rgb.size, corner_rgb)
        diff = ImageChops.difference(rgb, bg)
        bbox = diff.getbbox()
        if bbox is None:
            return (0, 0, rgb.size[0], rgb.size[1])
        return bbox

    def _pick_best_logo_path(self, candidates):
        best_path = None
        best_bbox = None
        best_score = -1
        for path in candidates:
            if not os.path.exists(path):
                continue
            try:
                logo = Image.open(path).convert("RGBA")
            except Exception:
                continue
            w, h = logo.size
            l, t, r, b = self._foreground_bbox_against_corner_bg(logo)
            margin = min(l, t, w - r, h - b)
            score = max(0, margin)
            if score > best_score:
                best_score = score
                best_path = path
                best_bbox = (l, t, r, b)
        return best_path, best_bbox

    def _ensure_brand_screen_texture(self):
        if Image is None or ImageDraw is None or ImageOps is None or ImageChops is None:
            return None
        tex_path = os.path.join(self.temp_dir, "screen_swarm_logo.png")

        w, h = 512, 320
        resampling = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

        logo_path, logo_bbox = self._pick_best_logo_path(SCREEN_LOGO_CANDIDATES)
        if logo_path is not None:
            logo = Image.open(logo_path).convert("RGBA")
            if logo_bbox is not None:
                l, t, r, b = logo_bbox
                pad = 4
                l = max(0, l - pad)
                t = max(0, t - pad)
                r = min(logo.size[0], r + pad)
                b = min(logo.size[1], b + pad)
                logo = logo.crop((l, t, r, b))
                                                                   
            img = Image.new("RGBA", (w, h), (8, 8, 8, 255))
            safe_pad = 12
            fitted = ImageOps.contain(logo, (w - safe_pad * 2, h - safe_pad * 2), method=resampling)
            px = (w - fitted.width) // 2
            py = (h - fitted.height) // 2
            img.alpha_composite(fitted, (px, py))
        else:
            img = Image.new("RGBA", (w, h), (224, 238, 245, 255))
            draw = ImageDraw.Draw(img)
            draw.text((int(w * 0.20), int(h * 0.38)), SCREEN_BRANDING_LABEL, fill=(25, 94, 122, 255))

        if SCREEN_TEXTURE_ROTATE_DEG % 360 != 0:
            img = img.rotate(SCREEN_TEXTURE_ROTATE_DEG, expand=False)
        img.convert("RGB").save(tex_path)
        return tex_path

    def _load_texture_cached(self, tex_path):
        key = tex_path.replace("\\", "/")
        if key in self.texture_id_cache:
            return self.texture_id_cache[key]
        tid = p.loadTexture(key)
        self.texture_id_cache[key] = tid
        return tid

    def _brand_texture_for_monitor(self):
        if self.brand_screen_texture_id is not None:
            return self.brand_screen_texture_id
        tex_path = self._ensure_brand_screen_texture()
        if tex_path is None:
            return None
        self.brand_screen_texture_id = self._load_texture_cached(tex_path)
        return self.brand_screen_texture_id

    def spawn(self, filename, x, y, yaw_deg, floor_z, extra_z=0.0, scale=None):
        if scale is None:
            scale = self.uniform_scale
        scale_xyz = self._normalize_scale(scale)

        min_v, max_v = self._mesh_bounds(filename, scale_xyz)
        min_z = min_v[2]
        spawn_z = floor_z - min_z + extra_z

                                                                           
        center_x = (min_v[0] + max_v[0]) * 0.5
        center_y = (min_v[1] + max_v[1]) * 0.5
        yaw_rad = math.radians(yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        offset_x = center_x * cos_y - center_y * sin_y
        offset_y = center_x * sin_y + center_y * cos_y
        spawn_x = x - offset_x
        spawn_y = y - offset_y

        quat = p.getQuaternionFromEuler([0, 0, yaw_rad])
        wall_assets = {ASSETS["wall"], ASSETS["wall_door"], ASSETS["wall_corner"]}
        if OFFICE_WALL_FORCE_FLAT_COLOR and filename in wall_assets:
            mesh_path = self._asset_path(filename).replace("\\", "/")
            mesh_tag_base = self._sanitize_name(os.path.splitext(filename)[0])
            urdf_path = self._create_urdf_for_mesh(
                mesh_path=mesh_path,
                mesh_tag=f"{mesh_tag_base}_flat",
                scale=scale_xyz,
                rgba=OFFICE_WALL_FLAT_RGBA[:3],
            )
            return p.loadURDF(
                urdf_path,
                [spawn_x, spawn_y, spawn_z],
                quat,
                useFixedBase=True,
            )
        if OFFICE_FLOOR_FORCE_FLAT_COLOR and filename == ASSETS["floor_tile"]:
            mesh_path = self._asset_path(filename).replace("\\", "/")
            mesh_tag_base = self._sanitize_name(os.path.splitext(filename)[0])
            urdf_path = self._create_urdf_for_mesh(
                mesh_path=mesh_path,
                mesh_tag=f"{mesh_tag_base}_flat",
                scale=scale_xyz,
                rgba=OFFICE_FLOOR_FLAT_RGBA[:3],
            )
            return p.loadURDF(
                urdf_path,
                [spawn_x, spawn_y, spawn_z],
                quat,
                useFixedBase=True,
            )

        parts = self._material_parts(filename)

        first_body = None
        mesh_tag_base = self._sanitize_name(os.path.splitext(filename)[0])
        for idx, (mesh_path, color, mat_name) in enumerate(parts):
            urdf_path = self._create_urdf_for_mesh(
                mesh_path=mesh_path,
                mesh_tag=f"{mesh_tag_base}_part_{idx}",
                scale=scale_xyz,
                rgba=color,
            )
            body_id = p.loadURDF(
                urdf_path,
                [spawn_x, spawn_y, spawn_z],
                quat,
                useFixedBase=True,
            )
            if first_body is None:
                first_body = body_id
            if SCREEN_BRANDING_ENABLED and filename == ASSETS["monitor"] and mat_name == "metal":
                tex_id = self._brand_texture_for_monitor()
                if tex_id is not None:
                    p.changeVisualShape(body_id, -1, rgbaColor=[1, 1, 1, 1], textureUniqueId=tex_id)
        return first_body

    def back_offset(self, filename, yaw_deg, inward_normal, scale=None):
        if scale is None:
            scale = self.uniform_scale
        scale_xyz = self._normalize_scale(scale)

        verts = self._parse_obj_vertices(filename)
        transformed = []
        for v in verts:
            rx, ry, rz = self._rotate_y_up_to_z_up(v)
            transformed.append((rx * scale_xyz[0], ry * scale_xyz[1], rz * scale_xyz[2]))

        min_v = [min(v[i] for v in transformed) for i in range(3)]
        max_v = [max(v[i] for v in transformed) for i in range(3)]
        center_x = (min_v[0] + max_v[0]) * 0.5
        center_y = (min_v[1] + max_v[1]) * 0.5

        yaw_rad = math.radians(yaw_deg)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)

        nx, ny = inward_normal
        outward_x, outward_y = -nx, -ny
        back = -1e9
        for vx, vy, _ in transformed:
            lx = vx - center_x
            ly = vy - center_y
            wx = lx * cos_y - ly * sin_y
            wy = lx * sin_y + ly * cos_y
            proj = wx * outward_x + wy * outward_y
            if proj > back:
                back = proj
        return back

    def top_height_from_floor(self, filename, scale=None):
        if scale is None:
            scale = self.uniform_scale
        min_v, max_v = self._mesh_bounds(filename, scale)
        return max_v[2] - min_v[2]

    def model_size(self, filename, scale=None):
        if scale is None:
            scale = self.uniform_scale
        min_v, max_v = self._mesh_bounds(filename, scale)
        return (
            max_v[0] - min_v[0],
            max_v[1] - min_v[1],
            max_v[2] - min_v[2],
        )

    def _local_front_angle_rad(self, filename, scale=None):
        if filename in ASSET_FRONT_DEG:
            return math.radians(ASSET_FRONT_DEG[filename])

        if scale is None:
            scale = self.uniform_scale
        key = (filename, scale)
        if key in self.front_angle_cache:
            return self.front_angle_cache[key]

        verts = self._parse_obj_vertices(filename)
        transformed = []
        for v in verts:
            rx, ry, rz = self._rotate_y_up_to_z_up(v)
            transformed.append((rx * scale, ry * scale, rz * scale))

        cx = sum(v[0] for v in transformed) / len(transformed)
        cy = sum(v[1] for v in transformed) / len(transformed)
        zmin = min(v[2] for v in transformed)
        zmax = max(v[2] for v in transformed)
        zcut = zmin + (zmax - zmin) * 0.88
        top = [v for v in transformed if v[2] >= zcut]

        if len(top) < 3:
                                                                              
            angle = math.radians(90.0)
            self.front_angle_cache[key] = angle
            return angle

        tx = sum(v[0] for v in top) / len(top)
        ty = sum(v[1] for v in top) / len(top)
        back_angle = math.atan2(ty - cy, tx - cx)
        front_angle = back_angle + math.pi
        self.front_angle_cache[key] = front_angle
        return front_angle

    def yaw_to_face_point(self, filename, x, y, tx, ty, scale=None):
        local_front = self._local_front_angle_rad(filename, scale=scale)
        desired = math.atan2(ty - y, tx - x)
        yaw_rad = desired - local_front
        return math.degrees(yaw_rad)


                                                                              
                                               
                                                                              
def build_floor(loader):
                                                             
    tile_model = ASSETS["floor_tile"]
    tile_x, tile_y, _ = loader.model_size(tile_model)

    nx = round(FLOOR_SIZE / tile_x)
    ny = round(FLOOR_SIZE / tile_y)
    if abs(nx * tile_x - FLOOR_SIZE) > 1e-6 or abs(ny * tile_y - FLOOR_SIZE) > 1e-6:
        raise ValueError(
            f"Uniform scale {UNIFORM_SCALE} does not tile 12x12 exactly with {tile_model} "
            f"(tile {tile_x:.3f}x{tile_y:.3f})."
        )

    start_x = -FLOOR_SIZE / 2.0 + tile_x / 2.0
    start_y = -FLOOR_SIZE / 2.0 + tile_y / 2.0
    for ix in range(nx):
        for iy in range(ny):
            x = start_x + ix * tile_x
            y = start_y + iy * tile_y
            loader.spawn(tile_model, x, y, yaw_deg=0, floor_z=0.0)

    return loader.top_height_from_floor(tile_model)


def slot_config(slot):
    if slot == "north":
        return {
            "normal": (0.0, -1.0),
            "tangent": (1.0, 0.0),
            "wall_yaw": 0.0,
        }
    if slot == "south":
        return {
            "normal": (0.0, 1.0),
            "tangent": (1.0, 0.0),
            "wall_yaw": 0.0,
        }
    if slot == "east":
        return {
            "normal": (-1.0, 0.0),
            "tangent": (0.0, 1.0),
            "wall_yaw": 90.0,
        }
    if slot == "west":
        return {
            "normal": (1.0, 0.0),
            "tangent": (0.0, 1.0),
            "wall_yaw": 90.0,
        }
    raise ValueError(f"Unknown slot: {slot}")


def snap_cardinal(yaw_deg):
    return (round(yaw_deg / 90.0) * 90.0) % 360.0


def snap_octant(yaw_deg):
    return (round(yaw_deg / 45.0) * 45.0) % 360.0


def inward_facing_yaw(loader, model_name, x, y, diagonal=False):
                                                                                
    raw_yaw = loader.yaw_to_face_point(model_name, x, y, ROOM_CENTER[0], ROOM_CENTER[1])
    return snap_octant(raw_yaw) if diagonal else snap_cardinal(raw_yaw)


def slot_inward_wall_yaw(loader, model_name, slot):
                                                               
                                                                      
                                                                                 
    _ = loader
    _ = model_name
    return float(wall_face_yaw(slot))


def wall_face_yaw(slot):
                                                 
    return {
        "north": 180.0,
        "south": 0.0,
        "east": 90.0,
        "west": 270.0,
    }[slot]


def wall_tangent_yaw(slot):
                                     
    return {
        "north": 0.0,
        "south": 0.0,
        "east": 90.0,
        "west": 90.0,
    }[slot]


def desk_lr_along_offsets(slot, separation):
                                                                                
    if slot in ("north", "west"):
        return -separation, separation
    return separation, -separation


def slot_xy(slot, along, inward):
    cfg = slot_config(slot)
    nx, ny = cfg["normal"]
    tx, ty = cfg["tangent"]
    edge = FLOOR_SIZE / 2.0
    cx0, cy0 = ROOM_CENTER

    if slot == "north":
        bx, by = cx0, cy0 + edge
    elif slot == "south":
        bx, by = cx0, cy0 - edge
    elif slot == "east":
        bx, by = cx0 + edge, cy0
    else:
        bx, by = cx0 - edge, cy0

    return bx + tx * along + nx * inward, by + ty * along + ny * inward


def corner_points():
    cx0, cy0 = ROOM_CENTER
    return [
        (cx0 - FLOOR_SIZE / 2.0 + 1.05, cy0 + FLOOR_SIZE / 2.0 - 1.05),
        (cx0 + FLOOR_SIZE / 2.0 - 1.05, cy0 + FLOOR_SIZE / 2.0 - 1.05),
        (cx0 - FLOOR_SIZE / 2.0 + 1.05, cy0 - FLOOR_SIZE / 2.0 + 1.05),
        (cx0 + FLOOR_SIZE / 2.0 - 1.05, cy0 - FLOOR_SIZE / 2.0 + 1.05),
    ]


def nearest_corner_index(x, y, corners):
    best_i = 0
    best_d2 = None
    for i, (cx, cy) in enumerate(corners):
        d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def workstation_l_corner_index(slot):
                                                                  
    return {
        "north": 1,              
        "south": 3,              
        "east": 1,               
        "west": 0,               
    }[slot]


def adjacent_slot_for_l(slot):
    return {
        "north": "east",
        "south": "east",
        "east": "north",
        "west": "north",
    }[slot]


def along_sign_for_corner(slot, corner_idx):
                                                                              
    sign_map = {
        "north": {0: -1.0, 1: 1.0},
        "south": {2: -1.0, 3: 1.0},
        "east": {3: -1.0, 1: 1.0},
        "west": {2: -1.0, 0: 1.0},
    }
    return sign_map[slot].get(corner_idx, 1.0)


def workstation_right_is_positive_along(slot):
                                                                                
    return slot in ("north", "west")


def _corner_trim_from_model(loader):
    if not ENABLE_PERIMETER_WALL_CORNERS:
        return 0.0
    corner_model = ASSETS.get("wall_corner", "")
    if not corner_model:
        return 0.0
    corner_path = loader._asset_path(corner_model)
    if not os.path.exists(corner_path):
        return 0.0

    wall_x, wall_y, _ = loader.model_size(ASSETS["wall"])
    wall_thickness = min(float(wall_x), float(wall_y))
    min_v, max_v = loader._mesh_bounds(corner_model, UNIFORM_SCALE)

                                                 
    anchor_local_x = float(min_v[0]) + (wall_thickness * 0.5)
    anchor_local_y = float(max_v[1]) - (wall_thickness * 0.5)

    inward_x = max(0.0, float(max_v[0]) - anchor_local_x)
    inward_y = max(0.0, anchor_local_y - float(min_v[1]))
    return max(inward_x, inward_y) + max(0.0, float(PERIMETER_WALL_CORNER_JOIN_GAP_M))


def _wall_segment_plan(loader):
    wall_len, _, _ = loader.model_size(ASSETS["wall"])
    if wall_len <= 1e-6:
        raise ValueError("Invalid wall length from wall.obj")

    if not ENABLE_PERIMETER_WALL_CORNERS:
        nseg = round(FLOOR_SIZE / wall_len)
        if abs(nseg * wall_len - FLOOR_SIZE) > 1e-6:
            raise ValueError(f"Wall model does not tile {FLOOR_SIZE:.2f}m exactly at scale {UNIFORM_SCALE}.")
        start = -FLOOR_SIZE / 2.0 + wall_len / 2.0
        return [(start + i * wall_len, float(PERIMETER_WALL_ALONG_SCALE)) for i in range(int(nseg))]

    trim = _corner_trim_from_model(loader)
    inner_span = FLOOR_SIZE - (2.0 * trim)
    if inner_span <= 0.2:
        raise ValueError("Corner trim too large for office wall span.")

    nseg = max(1, round(inner_span / wall_len))
    seg_len = inner_span / float(nseg)
    along_scale = (seg_len / wall_len) * float(PERIMETER_WALL_ALONG_SCALE)
    start = -inner_span / 2.0 + seg_len / 2.0
    return [(start + i * seg_len, along_scale) for i in range(int(nseg))]


def spawn_walls(loader, floor_top_z):
    seg_plan = _wall_segment_plan(loader)

    for slot in WALL_SLOTS:
        wall_yaw = slot_inward_wall_yaw(loader, ASSETS["wall"], slot)
        for along, along_scale in seg_plan:
            x, y = slot_xy(slot, along, inward=0.0)
            loader.spawn(
                ASSETS["wall"],
                x,
                y,
                yaw_deg=wall_yaw,
                floor_z=floor_top_z,
                scale=(UNIFORM_SCALE * along_scale, UNIFORM_SCALE, UNIFORM_SCALE),
            )
    spawn_wall_corners(loader, floor_top_z)


def spawn_wall_corners(loader, floor_top_z):
    if not ENABLE_PERIMETER_WALL_CORNERS:
        return
    corner_model = ASSETS.get("wall_corner", "")
    if not corner_model:
        return
    corner_path = loader._asset_path(corner_model)
    if not os.path.exists(corner_path):
        return

    min_v, max_v = loader._mesh_bounds(corner_model, UNIFORM_SCALE)
    cx_local = float((min_v[0] + max_v[0]) * 0.5)
    cy_local = float((min_v[1] + max_v[1]) * 0.5)
    wall_x, wall_y, _ = loader.model_size(ASSETS["wall"])
    wall_thickness = min(float(wall_x), float(wall_y))

                
                                                                                
                                                                         
    anchor_local_x = float(min_v[0]) + (wall_thickness * 0.5)
    anchor_local_y = float(max_v[1]) - (wall_thickness * 0.5)
    anchor_off_x = anchor_local_x - cx_local
    anchor_off_y = anchor_local_y - cy_local

    edge = (FLOOR_SIZE * 0.5) + float(PERIMETER_WALL_CORNER_OUTWARD_EPS)
    cx0, cy0 = ROOM_CENTER
                                                                   
                                                                                   
    corner_specs = (
        ("nw", -1.0, 1.0, 0.0),
        ("ne", 1.0, 1.0, 270.0),
        ("sw", -1.0, -1.0, 90.0),
        ("se", 1.0, -1.0, 180.0),
    )
    for _name, sx, sy, yaw in corner_specs:
        anchor_x = cx0 + (sx * edge)
        anchor_y = cy0 + (sy * edge)
        yaw_rad = math.radians(yaw)
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        rox = anchor_off_x * c - anchor_off_y * s
        roy = anchor_off_x * s + anchor_off_y * c
        x = anchor_x - rox
        y = anchor_y - roy
        loader.spawn(corner_model, x, y, yaw_deg=yaw, floor_z=floor_top_z)


def spawn_walls_with_entry(loader, floor_top_z, entry_slot, door_along=0.0, open_mode=ENTRY_WALL_OPENING_MODE):
    if entry_slot not in WALL_SLOTS:
        raise ValueError(f"Unknown entry slot: {entry_slot}")
    seg_plan = _wall_segment_plan(loader)
    seg_along = [a for a, _s in seg_plan]
    if not seg_along:
        return

    door_idx = min(range(len(seg_along)), key=lambda i: abs(seg_along[i] - float(door_along)))
    use_gap = str(open_mode).lower() == "gap"
    slot_yaw_cache = {}
    for slot in WALL_SLOTS:
        slot_yaw_cache[(slot, ASSETS["wall"])] = slot_inward_wall_yaw(loader, ASSETS["wall"], slot)
        slot_yaw_cache[(slot, ASSETS["wall_door"])] = slot_inward_wall_yaw(loader, ASSETS["wall_door"], slot)
        for i, (along, along_scale) in enumerate(seg_plan):
            if slot == entry_slot and i == door_idx:
                if use_gap:
                    continue
                model = ASSETS["wall_door"]
            else:
                model = ASSETS["wall"]
            x, y = slot_xy(slot, along, inward=0.0)
            wall_yaw = slot_yaw_cache[(slot, model)]
            loader.spawn(
                model,
                x,
                y,
                yaw_deg=wall_yaw,
                floor_z=floor_top_z,
                scale=(UNIFORM_SCALE * along_scale, UNIFORM_SCALE, UNIFORM_SCALE),
            )
    spawn_wall_corners(loader, floor_top_z)


def place_entry_wall(loader, floor_top_z, slot, seed, corners=None, spawn_doorway=True):
    if corners is None:
        corners = corner_points()
    rng = random.Random(seed + 5100 + WALL_SLOTS.index(slot))
    face_yaw = wall_face_yaw(slot)

                                                    
    if spawn_doorway:
        door_inward = 0.58
        dx, dy = slot_xy(slot, 0.0, inward=door_inward)
        loader.spawn(ASSETS["doorway"], dx, dy, yaw_deg=face_yaw, floor_z=floor_top_z)

                                                      
    mirror = -1.0 if rng.random() < 0.5 else 1.0
    coat_along = -2.55 * mirror
    plant_along = 2.55 * mirror
    cx, cy = slot_xy(slot, coat_along, inward=1.00)
    px, py = slot_xy(slot, plant_along, inward=1.02)
    loader.spawn(ASSETS["entry_coat_rack"], cx, cy, yaw_deg=face_yaw, floor_z=floor_top_z)
    loader.spawn(ASSETS["entry_plant"], px, py, yaw_deg=face_yaw, floor_z=floor_top_z)

                                                                            
    forbidden_by_corner = {}
    coat_corner = nearest_corner_index(cx, cy, corners)
    plant_corner = nearest_corner_index(px, py, corners)
    forbidden_by_corner.setdefault(coat_corner, set()).add("tall_accent")
    forbidden_by_corner.setdefault(plant_corner, set()).add("entry_plant")
    return forbidden_by_corner


def place_workstations_wall(loader, floor_top_z, slot, seed):
    rng = random.Random(seed + 1700 + WALL_SLOTS.index(slot))
    desk_w, desk_d, _ = loader.model_size(ASSETS["desk"])
    _, chair_d, _ = loader.model_size(ASSETS["desk_chair"])

                                                                                 
    right_corner_model = ASSETS["desk_corner"] if rng.random() < 0.65 else ASSETS["desk"]
    if workstation_right_is_positive_along(slot):
        row_models = [ASSETS["desk"], ASSETS["desk"], ASSETS["desk"], ASSETS["desk"], right_corner_model]
    else:
        row_models = [right_corner_model, ASSETS["desk"], ASSETS["desk"], ASSETS["desk"], ASSETS["desk"]]

    widths = [loader.model_size(m)[0] for m in row_models]
    gap = 0.14
    max_span = FLOOR_SIZE - 1.9
    row_span = sum(widths) + gap * (len(widths) - 1)
    if row_span > max_span:
        gap = max(0.08, (max_span - sum(widths)) / (len(widths) - 1))
        row_span = sum(widths) + gap * (len(widths) - 1)

    centers = []
    cursor = -row_span / 2.0
    for w in widths:
        centers.append(cursor + w / 2.0)
        cursor += w + gap

    base_desk_inward = 1.00
    base_chair_offset = (desk_d / 2.0) + (chair_d / 2.0) + 0.12
    base_monitor_offset = -0.16
    base_keyboard_offset = 0.12

    desk_yaw = wall_face_yaw(slot)
    inward_normal = slot_config(slot)["normal"]
    back_regular = loader.back_offset(ASSETS["desk"], desk_yaw, inward_normal, scale=UNIFORM_SCALE)
    target_back_inward = base_desk_inward - back_regular

    unified_chair_inward = base_desk_inward + base_chair_offset
    unified_monitor_inward = base_desk_inward + base_monitor_offset
    unified_keyboard_inward = base_desk_inward + base_keyboard_offset

    def spawn_station(st_slot, along, desk_model, chair_along_nudge=0.0):
        key_along_offset, mouse_along_offset = desk_lr_along_offsets(st_slot, separation=0.28)
        desk_yaw = wall_face_yaw(st_slot)
        chair_yaw = (desk_yaw + 180.0) % 360.0
        desk_h = loader.model_size(desk_model)[2]
        inward_normal = slot_config(st_slot)["normal"]
        back_model = loader.back_offset(desk_model, desk_yaw, inward_normal, scale=UNIFORM_SCALE)

                                                                      
        desk_inward = target_back_inward + back_model
        chair_inward = unified_chair_inward
        monitor_inward = unified_monitor_inward
        keyboard_inward = unified_keyboard_inward
        logical_along = along

        dx, dy = slot_xy(st_slot, along, inward=desk_inward)
        loader.spawn(desk_model, dx, dy, yaw_deg=desk_yaw, floor_z=floor_top_z, scale=UNIFORM_SCALE)

        cx, cy = slot_xy(st_slot, logical_along + chair_along_nudge, inward=chair_inward)
        loader.spawn(ASSETS["desk_chair"], cx, cy, yaw_deg=chair_yaw, floor_z=floor_top_z)

        mx, my = slot_xy(st_slot, logical_along, inward=monitor_inward)
        loader.spawn(ASSETS["monitor"], mx, my, yaw_deg=desk_yaw, floor_z=floor_top_z, extra_z=desk_h + 0.01)

        kx, ky = slot_xy(st_slot, logical_along + key_along_offset, inward=keyboard_inward)
        loader.spawn(ASSETS["keyboard"], kx, ky, yaw_deg=desk_yaw, floor_z=floor_top_z, extra_z=desk_h + 0.01)

        msx, msy = slot_xy(st_slot, logical_along + mouse_along_offset, inward=keyboard_inward)
        loader.spawn(ASSETS["mouse"], msx, msy, yaw_deg=desk_yaw, floor_z=floor_top_z, extra_z=desk_h + 0.01)

    for i, (model, along) in enumerate(zip(row_models, centers)):
        chair_along_nudge = 0.0
        if model == ASSETS["desk_corner"]:
                                                                                                             
            if i == 0:
                chair_along_nudge = +0.16
            elif i == len(row_models) - 1:
                chair_along_nudge = -0.16
        spawn_station(slot, along, model, chair_along_nudge=chair_along_nudge)

    return None


def place_files_wall(loader, floor_top_z, slot, seed):
    rng = random.Random(seed + 3100 + WALL_SLOTS.index(slot))
    face_yaw = wall_face_yaw(slot)
    tangent_yaw = wall_tangent_yaw(slot)

                                                      
    shelf_variants = [
        [
            ASSETS["bookcase_open"],
            ASSETS["bookcase_open"],
            ASSETS["bookcase_closed"],
            ASSETS["bookcase_wide"],
        ],
        [
            ASSETS["bookcase_open"],
            ASSETS["bookcase_closed"],
            ASSETS["bookcase_open"],
            ASSETS["bookcase_open_low"],
            ASSETS["bookcase_open"],
        ],
        [
            ASSETS["bookcase_open"],
            ASSETS["bookcase_wide"],
            ASSETS["bookcase_open_low"],
            ASSETS["bookcase_open"],
            ASSETS["bookcase_closed"],
        ],
    ]
    shelves = rng.choice(shelf_variants)
    shelf_gap = 0.08
    shelf_inward = 0.72

    widths = [loader.model_size(m)[0] for m in shelves]
    total_span = sum(widths) + shelf_gap * (len(shelves) - 1)
    start = -total_span / 2.0

    centers = []
    cursor = start
    for w in widths:
        centers.append(cursor + w / 2.0)
        cursor += w + shelf_gap

                                                     
    intervals = []
    for c, w in zip(centers, widths):
        intervals.append((c - w / 2.0, c + w / 2.0))
    for i in range(len(intervals) - 1):
        if intervals[i][1] > intervals[i + 1][0]:
            raise ValueError("Files wall layout overlap detected.")

    shelf_profiles = {
        ASSETS["bookcase_open_low"]: {"max_per_row": 1, "book_inward": shelf_inward + 0.00},
        ASSETS["bookcase_open"]: {"max_per_row": 1, "book_inward": shelf_inward + 0.00},
        ASSETS["bookcase_closed"]: {"max_per_row": 1, "book_inward": shelf_inward + 0.00},
        ASSETS["bookcase_wide"]: {"max_per_row": 3, "book_inward": shelf_inward + 0.00},
    }

    for model, along in zip(shelves, centers):
        x, y = slot_xy(slot, along, inward=shelf_inward)
        loader.spawn(model, x, y, yaw_deg=face_yaw, floor_z=floor_top_z)

        profile = shelf_profiles.get(
            model,
            {
                "max_per_row": 1,
                "book_inward": shelf_inward + 0.00,
            },
        )
        shelf_h = loader.model_size(model)[2]
        book_inward = profile["book_inward"]

        if SHOW_SHELF_LABELS:
            label = os.path.splitext(os.path.basename(model))[0]
            p.addUserDebugText(
                text=label,
                textPosition=[x, y, floor_top_z + shelf_h + 0.16],
                textColorRGB=[0.1, 0.1, 0.1],
                textSize=1.0,
                lifeTime=0,
            )

                                                       
        row_levels = list(loader.shelf_surface_levels(model))
        if model == ASSETS["bookcase_open_low"] and row_levels:
                                                    
            row_levels = [row_levels[0]]
        if not row_levels:
            row_levels = [max(0.20, shelf_h * 0.30)]

        for li, level in enumerate(row_levels):
            max_per_row = profile["max_per_row"]
            count = 1 if max_per_row == 1 else rng.randint(1, max_per_row)
            if count == 1:
                row_offsets = [rng.choice([-0.14, 0.0, 0.14])]
            elif count == 2:
                row_offsets = [-0.20, 0.20]
            else:
                row_offsets = [-0.30, 0.0, 0.30]

            for oi, offset in enumerate(row_offsets):
                bx, by = slot_xy(slot, along + offset, inward=book_inward)
                flip = ((li + oi) % 2 == 1)
                byaw = face_yaw if not flip else (face_yaw + 180.0) % 360.0
                loader.spawn(
                    ASSETS["books"],
                    bx,
                    by,
                    yaw_deg=byaw,
                    floor_z=floor_top_z,
                    extra_z=level + 0.005,
                )

                                                                             
    left_box_along = -total_span / 2.0 - loader.model_size(ASSETS["box"])[0] / 2.0 - 0.20
    right_box_along = total_span / 2.0 + loader.model_size(ASSETS["box_open"])[0] / 2.0 + 0.20
    bx1, by1 = slot_xy(slot, left_box_along, inward=1.08)
    bx2, by2 = slot_xy(slot, right_box_along, inward=1.08)
    loader.spawn(ASSETS["box"], bx1, by1, yaw_deg=tangent_yaw, floor_z=floor_top_z)
    loader.spawn(ASSETS["box_open"], bx2, by2, yaw_deg=tangent_yaw, floor_z=floor_top_z)


def place_services_wall(loader, floor_top_z, slot, seed, corners=None):
    rng = random.Random(seed + 7300 + WALL_SLOTS.index(slot))
    cab_h = loader.model_size(ASSETS["cabinet"])[2]
    face_yaw = wall_face_yaw(slot)
    tangent_yaw = wall_tangent_yaw(slot)
    if corners is None:
        corners = corner_points()

    fridge_model = rng.choices(
        [ASSETS["fridge"], ASSETS["fridge_tall"], ASSETS["fridge_large"]],
        weights=[0.55, 0.35, 0.10],
        k=1,
    )[0]
    storage_model = rng.choice([ASSETS["bookcase_closed_doors"], ASSETS["cabinet_tv_doors"]])
    mirror = -1.0 if rng.random() < 0.5 else 1.0

                                                                                    
    fridge_along = -1.90 * mirror
    cabinet_along = 0.00 * mirror
    storage_along = 1.90 * mirror
    trash_along = 2.95 * mirror
    plant_along = -3.05 * mirror

    fx, fy = slot_xy(slot, fridge_along, inward=0.94)
    loader.spawn(fridge_model, fx, fy, yaw_deg=face_yaw, floor_z=floor_top_z)

    cx, cy = slot_xy(slot, cabinet_along, inward=0.93)
    loader.spawn(ASSETS["cabinet"], cx, cy, yaw_deg=face_yaw, floor_z=floor_top_z)

    kx, ky = slot_xy(slot, cabinet_along, inward=0.88)
    loader.spawn(ASSETS["coffee_machine"], kx, ky, yaw_deg=face_yaw, floor_z=floor_top_z, extra_z=cab_h + 0.01)

    sx, sy = slot_xy(slot, storage_along, inward=0.92)
    loader.spawn(storage_model, sx, sy, yaw_deg=face_yaw, floor_z=floor_top_z)

    tx, ty = slot_xy(slot, trash_along, inward=1.06)
    loader.spawn(ASSETS["trashcan"], tx, ty, yaw_deg=tangent_yaw, floor_z=floor_top_z)

    px, py = slot_xy(slot, plant_along, inward=1.02)
    loader.spawn(ASSETS["entry_plant"], px, py, yaw_deg=face_yaw, floor_z=floor_top_z)

    forbidden_by_corner = {}
    trash_corner = nearest_corner_index(tx, ty, corners)
    plant_corner = nearest_corner_index(px, py, corners)
    forbidden_by_corner.setdefault(trash_corner, set()).add("trashcan")
    forbidden_by_corner.setdefault(plant_corner, set()).add("entry_plant")
    return forbidden_by_corner


def place_corner_decor(loader, floor_top_z, seed, forbidden_styles_by_corner=None, blocked_corner_indices=None):
    rng = random.Random(seed + 9007)
    if forbidden_styles_by_corner is None:
        forbidden_styles_by_corner = {}
    if blocked_corner_indices is None:
        blocked_corner_indices = set()

    corners = corner_points()
    active_corner_indices = [i for i in range(len(corners)) if i not in blocked_corner_indices]
    if len(active_corner_indices) > 3:
        rng.shuffle(active_corner_indices)
        active_corner_indices = active_corner_indices[:3]

                                   
                                                                          
    tall_asset = rng.choice([ASSETS["lamp_floor"], ASSETS["entry_coat_rack"]])
    styles = ["entry_plant", "trashcan", "tall_accent"]

    assigned = None
    for _ in range(64):
        trial = list(styles)
        rng.shuffle(trial)
        ok = True
        for ci, st in zip(active_corner_indices, trial):
            blocked = forbidden_styles_by_corner.get(ci, set())
            if st in blocked:
                ok = False
                break
        if ok:
            assigned = trial
            break
    if assigned is None:
        assigned = list(styles)
        rng.shuffle(assigned)

    for ci, style in zip(active_corner_indices, assigned):
        x, y = corners[ci]
        if style == "entry_plant":
            yaw = snap_octant(loader.yaw_to_face_point(ASSETS["entry_plant"], x, y, ROOM_CENTER[0], ROOM_CENTER[1]))
            loader.spawn(ASSETS["entry_plant"], x, y, yaw_deg=yaw, floor_z=floor_top_z)
            continue

        if style == "trashcan":
            yaw = snap_octant(loader.yaw_to_face_point(ASSETS["trashcan"], x, y, ROOM_CENTER[0], ROOM_CENTER[1]))
            loader.spawn(ASSETS["trashcan"], x, y, yaw_deg=yaw, floor_z=floor_top_z)
            continue

        yaw = snap_octant(loader.yaw_to_face_point(tall_asset, x, y, ROOM_CENTER[0], ROOM_CENTER[1]))
        loader.spawn(tall_asset, x, y, yaw_deg=yaw, floor_z=floor_top_z)


def place_wall_contents(loader, floor_top_z, role_by_slot, seed):
    for slot in WALL_SLOTS:
        role = role_by_slot[slot]
        if role == "entry":
            place_entry_wall(loader, floor_top_z, slot, seed)
        elif role == "workstations":
            place_workstations_wall(loader, floor_top_z, slot, seed)
        elif role == "files":
            place_files_wall(loader, floor_top_z, slot, seed)
        elif role == "services":
            place_services_wall(loader, floor_top_z, slot, seed)


def build_center_meeting(loader, floor_top_z, seed):
    cx, cy = ROOM_CENTER
    rng = random.Random(seed + 5200)
    table_model = ASSETS["meeting_table"]
    chair_model = ASSETS["meeting_chair"]

    table_x, table_y, _ = loader.model_size(table_model)
    chair_x, chair_y, _ = loader.model_size(chair_model)

                                                                  
    horizontal = rng.random() < 0.5
    table_yaw = 0.0 if horizontal else 90.0
    len_axis = (1.0, 0.0) if horizontal else (0.0, 1.0)
    side_axis = (0.0, 1.0) if horizontal else (1.0, 0.0)

    overlap = 0.14
    separation = table_x - overlap
    t1x = cx - len_axis[0] * separation / 2.0
    t1y = cy - len_axis[1] * separation / 2.0
    t2x = cx + len_axis[0] * separation / 2.0
    t2y = cy + len_axis[1] * separation / 2.0
    loader.spawn(table_model, t1x, t1y, yaw_deg=table_yaw, floor_z=floor_top_z)
    loader.spawn(table_model, t2x, t2y, yaw_deg=table_yaw, floor_z=floor_top_z)

    table_length = table_x * 2.0 - overlap
    row_offset = (table_y / 2.0) + (chair_y / 2.0) + 0.08
    end_margin = max(chair_x * 0.60, 0.40)
    along_slots = [
        -(table_length / 2.0) + end_margin,
        0.0,
        (table_length / 2.0) - end_margin,
    ]

    for along in along_slots:
        tx = cx + len_axis[0] * along
        ty = cy + len_axis[1] * along

        x_top = tx + side_axis[0] * row_offset
        y_top = ty + side_axis[1] * row_offset
        yaw_top = loader.yaw_to_face_point(chair_model, x_top, y_top, tx, ty)
        loader.spawn(chair_model, x_top, y_top, yaw_deg=snap_cardinal(yaw_top), floor_z=floor_top_z)

        x_bottom = tx - side_axis[0] * row_offset
        y_bottom = ty - side_axis[1] * row_offset
        yaw_bottom = loader.yaw_to_face_point(chair_model, x_bottom, y_bottom, tx, ty)
        loader.spawn(chair_model, x_bottom, y_bottom, yaw_deg=snap_cardinal(yaw_bottom), floor_z=floor_top_z)

                                                                
    head_offset = (table_length / 2.0) + (chair_y / 2.0) + 0.10
    x_left = cx - len_axis[0] * head_offset
    y_left = cy - len_axis[1] * head_offset
    x_right = cx + len_axis[0] * head_offset
    y_right = cy + len_axis[1] * head_offset
    yaw_left = loader.yaw_to_face_point(chair_model, x_left, y_left, cx, cy)
    yaw_right = loader.yaw_to_face_point(chair_model, x_right, y_right, cx, cy)
    loader.spawn(chair_model, x_left, y_left, yaw_deg=snap_cardinal(yaw_left), floor_z=floor_top_z)
    loader.spawn(chair_model, x_right, y_right, yaw_deg=snap_cardinal(yaw_right), floor_z=floor_top_z)


def layout_metrics(loader):
    tile_model = ASSETS["floor_tile"]
    desk_model = ASSETS["desk"]
    chair_model = ASSETS["desk_chair"]

    tile_x, tile_y, tile_z = loader.model_size(tile_model)
    desk_x, desk_y, desk_z = loader.model_size(desk_model)
    chair_x, chair_y, chair_z = loader.model_size(chair_model)
    desk_count = 5
    desk_gap = 0.20
    row_span = desk_count * desk_x + (desk_count - 1) * desk_gap

    return {
        "tile": (tile_x, tile_y, tile_z),
        "desk": (desk_x, desk_y, desk_z),
        "chair": (chair_x, chair_y, chair_z),
        "desk_row": (desk_count, desk_gap, row_span),
    }


def wall_role_map(seed):
    rng = random.Random(int(seed) + 1201)
    slots = list(WALL_SLOTS)
    roles = list(WALL_ROLES)
    rng.shuffle(slots)
    rng.shuffle(roles)
    return {slot: role for slot, role in zip(slots, roles)}


                                                                              
                 
                                                                              
def setup_simulation(use_gui):
    if p.isConnected():
        p.disconnect()
    p.connect(p.GUI if use_gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)


def build_map(seed):
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)

    loader = AssetLoader(ASSET_PATH, TEMP_URDF_DIR, UNIFORM_SCALE)
    floor_top_z = build_floor(loader)
    role_by_slot = wall_role_map(seed)
    entry_slot = next((s for s in WALL_SLOTS if role_by_slot.get(s) == "entry"), WALL_SLOTS[0])
    if ENABLE_PERIMETER_WALL_MESHES:
        spawn_walls_with_entry(
            loader,
            floor_top_z,
            entry_slot=entry_slot,
            door_along=ENTRY_WALL_OPENING_ALONG,
            open_mode=ENTRY_WALL_OPENING_MODE,
        )
    corners = corner_points()
    forbidden_styles_by_corner = {}
    blocked_corner_indices = set()
    spawn_entry_doorway = (not ENABLE_PERIMETER_WALL_MESHES) or (ENTRY_WALL_OPENING_MODE == "gap")
    for slot in WALL_SLOTS:
        role = role_by_slot[slot]
        if role == "entry":
            blocked = place_entry_wall(
                loader,
                floor_top_z,
                slot,
                seed,
                corners=corners,
                spawn_doorway=spawn_entry_doorway,
            )
            for k, vals in blocked.items():
                forbidden_styles_by_corner.setdefault(k, set()).update(vals)
        elif role == "workstations":
            workstation_corner = place_workstations_wall(loader, floor_top_z, slot, seed)
            if workstation_corner is not None:
                blocked_corner_indices.add(workstation_corner)
        elif role == "files":
            place_files_wall(loader, floor_top_z, slot, seed)
        elif role == "services":
            blocked = place_services_wall(loader, floor_top_z, slot, seed, corners=corners)
            for k, vals in blocked.items():
                forbidden_styles_by_corner.setdefault(k, set()).update(vals)
    place_corner_decor(
        loader,
        floor_top_z,
        seed,
        forbidden_styles_by_corner=forbidden_styles_by_corner,
        blocked_corner_indices=blocked_corner_indices,
    )
    build_center_meeting(loader, floor_top_z, seed)
    return role_by_slot


def run(use_gui=True, seed=None):
    if not os.path.isdir(ASSET_PATH):
        raise FileNotFoundError(f"Kenney furniture-kit path not found: {ASSET_PATH}")

    setup_simulation(use_gui=use_gui)
    active_seed = int(time.time()) if seed is None else int(seed)
    loader_for_metrics = AssetLoader(ASSET_PATH, TEMP_URDF_DIR, UNIFORM_SCALE)
    metrics = layout_metrics(loader_for_metrics)
    role_by_slot = build_map(active_seed)

    if not use_gui:
        for _ in range(5):
            p.stepSimulation()
        p.disconnect()
        return

    print("\n==================================================")
    walls_label = "WITH WALL MESHES" if ENABLE_PERIMETER_WALL_MESHES else "NO WALL MESHES"
    print(f"OFFICE MAP (12x12m) - 4 WALLS + CORNERS + CENTER ({walls_label})")
    print("Seed:", active_seed)
    print("Uniform scale:", UNIFORM_SCALE)
    print("Floor model: floorFull.obj tiled to 12x12")
    print("Models: entry wall (open door) + workstation wall + files wall + services wall + corner decor + center meeting")
    print(
        "Desk (WxDxH): "
        f"{metrics['desk'][0]:.3f} x {metrics['desk'][1]:.3f} x {metrics['desk'][2]:.3f} m"
    )
    print(
        "Chair (WxDxH): "
        f"{metrics['chair'][0]:.3f} x {metrics['chair'][1]:.3f} x {metrics['chair'][2]:.3f} m"
    )
    print(
        "Desk row: "
        f"{metrics['desk_row'][0]} desks, gap {metrics['desk_row'][1]:.2f} m, span {metrics['desk_row'][2]:.3f} m"
    )
    active = [f"{slot}:{role}" for slot, role in role_by_slot.items()]
    print("Active walls:", ", ".join(active))
    print("Controls:")
    print("  LMB + Drag : Rotate Camera")
    print("  WASD       : Move Camera")
    print("  Q / E      : Up/Down")
    print("  Shift      : Faster move")
    print("  1          : Toggle Wireframe")
    print("  2          : Toggle Shadows")
    print("  R          : Rebuild same phase")
    print("  ESC        : Quit")
    print("==================================================")

    cam = CameraController(z=10, pitch=-45, speed=0.125)
    wireframe_enabled = False
    shadows_enabled = True
    wireframe_pressed = False
    shadows_pressed = False
    r_pressed = False

    while True:
        keys = p.getKeyboardEvents()

        if keys.get(27, 0) & p.KEY_WAS_TRIGGERED:
            break

        cam.update(keys)

        if keys.get(ord("1"), 0) == 1:
            if not wireframe_pressed:
                wireframe_pressed = True
                wireframe_enabled = not wireframe_enabled
                p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1 if wireframe_enabled else 0)
        else:
            wireframe_pressed = False

        if keys.get(ord("2"), 0) == 1:
            if not shadows_pressed:
                shadows_pressed = True
                shadows_enabled = not shadows_enabled
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if shadows_enabled else 0)
        else:
            shadows_pressed = False

        if keys.get(ord("r"), 0) == 1:
            if not r_pressed:
                r_pressed = True
                active_seed = int(time.time())
                role_by_slot = build_map(active_seed)
                print(f"\nRegenerated seed: {active_seed}")
                for slot in WALL_SLOTS:
                    print(f"  {slot:5s} -> {role_by_slot[slot]}")
        else:
            r_pressed = False

        p.stepSimulation()
        time.sleep(1.0 / 60.0)

    p.disconnect()


def main():
    parser = argparse.ArgumentParser(description="12x12 office map using Kenney furniture-kit.")
    parser.add_argument("--seed", type=int, default=None, help="Layout seed (wall role shuffle).")
    parser.add_argument("--headless", action="store_true", help="Run in DIRECT mode without GUI.")
    _ = parser.parse_args()
    run(use_gui=not _.headless, seed=_.seed)


if __name__ == "__main__":
    main()
