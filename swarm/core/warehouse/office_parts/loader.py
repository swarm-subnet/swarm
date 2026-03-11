from ._shared import *


class AssetLoader:
    def __init__(self, asset_dir, temp_dir, uniform_scale, cli=0):
        self.asset_dir = asset_dir
        self.temp_dir = temp_dir
        self.uniform_scale = uniform_scale
        self.cli = cli
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
                        verts.append(
                            (float(parts[1]), float(parts[2]), float(parts[3]))
                        )
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
            transformed.append(
                (rx * scale_xyz[0], ry * scale_xyz[1], rz * scale_xyz[2])
            )
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
                        verts.append(
                            (float(parts[1]), float(parts[2]), float(parts[3]))
                        )
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
        urdf_path = os.path.join(
            self.temp_dir, f"{name}_s{str(scale).replace('.', '_')}.urdf"
        )
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
        scale_tag = "_".join(
            str(v).replace(".", "_").replace("-", "m") for v in scale_xyz
        )
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
        out_lines = ["# UV retiled for single-image monitor screen\n"]
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
        mtllib, v_lines, vt_lines, vn_lines, faces_by_mat = self._read_material_usage(
            filename
        )
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
            left, top, right, bottom = self._foreground_bbox_against_corner_bg(logo)
            margin = min(left, top, w - right, h - bottom)
            score = max(0, margin)
            if score > best_score:
                best_score = score
                best_path = path
                best_bbox = (left, top, right, bottom)
        return best_path, best_bbox

    def _ensure_brand_screen_texture(self):
        if Image is None or ImageDraw is None or ImageOps is None or ImageChops is None:
            return None
        tex_path = os.path.join(self.temp_dir, "screen_swarm_logo.png")
        w, h = 512, 320
        resampling = (
            Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
        )
        logo_path, logo_bbox = self._pick_best_logo_path(SCREEN_LOGO_CANDIDATES)
        if logo_path is not None:
            logo = Image.open(logo_path).convert("RGBA")
            if logo_bbox is not None:
                left, top, right, bottom = logo_bbox
                pad = 4
                left = max(0, left - pad)
                top = max(0, top - pad)
                right = min(logo.size[0], right + pad)
                bottom = min(logo.size[1], bottom + pad)
                logo = logo.crop((left, top, right, bottom))
            img = Image.new("RGBA", (w, h), (8, 8, 8, 255))
            safe_pad = 12
            fitted = ImageOps.contain(
                logo, (w - safe_pad * 2, h - safe_pad * 2), method=resampling
            )
            px = (w - fitted.width) // 2
            py = (h - fitted.height) // 2
            img.alpha_composite(fitted, (px, py))
        else:
            img = Image.new("RGBA", (w, h), (224, 238, 245, 255))
            draw = ImageDraw.Draw(img)
            draw.text(
                (int(w * 0.20), int(h * 0.38)),
                SCREEN_BRANDING_LABEL,
                fill=(25, 94, 122, 255),
            )
        if SCREEN_TEXTURE_ROTATE_DEG % 360 != 0:
            img = img.rotate(SCREEN_TEXTURE_ROTATE_DEG, expand=False)
        img.convert("RGB").save(tex_path)
        return tex_path

    def _load_texture_cached(self, tex_path):
        key = tex_path.replace("\\", "/")
        if key in self.texture_id_cache:
            return self.texture_id_cache[key]
        tid = p.loadTexture(key, physicsClientId=self.cli)
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
                physicsClientId=self.cli,
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
                physicsClientId=self.cli,
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
                physicsClientId=self.cli,
            )
            if first_body is None:
                first_body = body_id
            if (
                SCREEN_BRANDING_ENABLED
                and filename == ASSETS["monitor"]
                and mat_name == "metal"
            ):
                tex_id = self._brand_texture_for_monitor()
                if tex_id is not None:
                    p.changeVisualShape(
                        body_id,
                        -1,
                        rgbaColor=[1, 1, 1, 1],
                        textureUniqueId=tex_id,
                        physicsClientId=self.cli,
                    )
        return first_body

    def back_offset(self, filename, yaw_deg, inward_normal, scale=None):
        if scale is None:
            scale = self.uniform_scale
        scale_xyz = self._normalize_scale(scale)
        verts = self._parse_obj_vertices(filename)
        transformed = []
        for v in verts:
            rx, ry, rz = self._rotate_y_up_to_z_up(v)
            transformed.append(
                (rx * scale_xyz[0], ry * scale_xyz[1], rz * scale_xyz[2])
            )
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
