"""OBJ parsing and footprint geometry helpers for forest generation."""

import hashlib
import tempfile

from ._shared import *


# ---------------------------------------------------------------------------
# SECTION 4: OBJ geometry parsing (cached, client-independent)
# ---------------------------------------------------------------------------
def _obj_bounds(path: str) -> Tuple[float, float, float, float, float, float]:
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            min_x, min_y, min_z = min(min_x, x), min(min_y, y), min(min_z, z)
            max_x, max_y, max_z = max(max_x, x), max(max_y, y), max(max_z, z)
    if min_x == float("inf"):
        raise RuntimeError(f"No vertices found in OBJ: {path}")
    return min_x, min_y, min_z, max_x, max_y, max_z


def _obj_bounds_cached(path: str) -> Tuple[float, float, float, float, float, float]:
    cached = _OBJ_BOUNDS_CACHE.get(path)
    if cached is None:
        cached = _obj_bounds(path)
        _OBJ_BOUNDS_CACHE[path] = cached
    return cached


def _obj_planar_radius_cached(path: str) -> float:
    cached = _OBJ_PLANAR_RADIUS_CACHE.get(path)
    if cached is not None:
        return cached
    min_x, _, min_z, max_x, _, max_z = _obj_bounds_cached(path)
    cached = max((max_x - min_x) * 0.5, (max_z - min_z) * 0.5)
    _OBJ_PLANAR_RADIUS_CACHE[path] = cached
    return cached


def _compute_vertex_normals(
    verts: List[List[float]], indices: List[int]
) -> List[List[float]]:
    normals = [[0.0, 0.0, 0.0] for _ in verts]
    for i in range(0, len(indices), 3):
        ia, ib, ic = indices[i], indices[i + 1], indices[i + 2]
        ax, ay, az = verts[ia]
        bx, by, bz = verts[ib]
        cx, cy, cz = verts[ic]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx = (uy * vz) - (uz * vy)
        ny = (uz * vx) - (ux * vz)
        nz = (ux * vy) - (uy * vx)
        for idx in (ia, ib, ic):
            normals[idx][0] += nx
            normals[idx][1] += ny
            normals[idx][2] += nz
    for n in normals:
        length = math.sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2])
        if length > 1e-9:
            n[0] /= length
            n[1] /= length
            n[2] /= length
        else:
            n[0], n[1], n[2] = 0.0, 1.0, 0.0
    return normals


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


def _parse_mtl_diffuse_colors(mtl_path: Optional[str]) -> Dict[str, List[float]]:
    if not mtl_path or not os.path.exists(mtl_path):
        return {}
    out: Dict[str, List[float]] = {}
    current: Optional[str] = None
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("newmtl "):
                current = line.split(None, 1)[1].strip()
                continue
            if current and line.lower().startswith("kd "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    r = max(0.0, min(1.0, float(parts[1])))
                    g = max(0.0, min(1.0, float(parts[2])))
                    b = max(0.0, min(1.0, float(parts[3])))
                except ValueError:
                    continue
                r = min(1.0, (r ** MTL_COLOR_GAMMA) * MTL_COLOR_MULTIPLIER)
                g = min(1.0, (g ** MTL_COLOR_GAMMA) * MTL_COLOR_MULTIPLIER)
                b = min(1.0, (b ** MTL_COLOR_GAMMA) * MTL_COLOR_MULTIPLIER)
                out[current] = [r, g, b, 1.0]
    return out


def _parse_obj_material_meshes(
    obj_path: str,
) -> Dict[str, Tuple[List[List[float]], List[int], List[List[float]]]]:
    cached = _OBJ_MATERIAL_MESH_CACHE.get(obj_path)
    if cached is not None:
        return cached

    pkl_path = obj_path + ".meshcache.pkl"
    try:
        if (
            os.path.exists(pkl_path)
            and os.path.getmtime(pkl_path) >= os.path.getmtime(obj_path)
        ):
            with open(pkl_path, "rb") as pf:
                cached = pickle.load(pf)
                _OBJ_MATERIAL_MESH_CACHE[obj_path] = cached
                return cached
    except Exception:
        pass

    vertices: List[Tuple[float, float, float]] = []
    faces_by_mat: Dict[str, List[List[int]]] = {}
    current_mat = "__default__"
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append(
                        (float(parts[1]), float(parts[2]), float(parts[3]))
                    )
                continue
            if line.lower().startswith("usemtl "):
                split = line.split(None, 1)
                current_mat = split[1].strip() if len(split) > 1 else "__default__"
                faces_by_mat.setdefault(current_mat, [])
                continue
            if line.startswith("f "):
                tokens = line.split()[1:]
                poly: List[int] = []
                for token in tokens:
                    v_tok = token.split("/")[0]
                    if not v_tok:
                        continue
                    idx = int(v_tok)
                    if idx < 0:
                        idx = len(vertices) + 1 + idx
                    poly.append(idx - 1)
                if len(poly) >= 3:
                    faces_by_mat.setdefault(current_mat, []).append(poly)

    mesh_by_mat: Dict[str, Tuple[List[List[float]], List[int], List[List[float]]]] = {}
    for mat_name, polys in faces_by_mat.items():
        remap: Dict[int, int] = {}
        out_vertices: List[List[float]] = []
        out_indices: List[int] = []
        for poly in polys:
            for i in range(1, len(poly) - 1):
                tri = (poly[0], poly[i], poly[i + 1])
                for src_idx in tri:
                    mapped = remap.get(src_idx)
                    if mapped is None:
                        mapped = len(out_vertices)
                        remap[src_idx] = mapped
                        vx, vy, vz = vertices[src_idx]
                        out_vertices.append([vx, vy, vz])
                    out_indices.append(mapped)
        if out_indices:
            out_normals = _compute_vertex_normals(out_vertices, out_indices)
            mesh_by_mat[mat_name] = (out_vertices, out_indices, out_normals)

    _OBJ_MATERIAL_MESH_CACHE[obj_path] = mesh_by_mat
    try:
        with open(pkl_path, "wb") as pf:
            pickle.dump(mesh_by_mat, pf, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return mesh_by_mat


def _material_visual_obj_paths(obj_path: str) -> Dict[str, str]:
    material_meshes = _parse_obj_material_meshes(obj_path)
    if not material_meshes:
        return {}

    stat = os.stat(obj_path)
    cache_root = os.path.join(tempfile.gettempdir(), "swarm_forest_visual_cache_v1")
    source_key = hashlib.sha256(
        f"{obj_path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()[:16]
    obj_stem = os.path.splitext(os.path.basename(obj_path))[0]
    target_dir = os.path.join(cache_root, source_key)
    os.makedirs(target_dir, exist_ok=True)

    out: Dict[str, str] = {}
    for mat_name, (verts, indices, normals) in material_meshes.items():
        token = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in mat_name
        ) or "default"
        out_path = os.path.join(target_dir, f"{obj_stem}__{token}.obj")
        if not os.path.exists(out_path):
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write("# Generated by Swarm forest visualizer GPU mesh splitter\n")
                f.write(f"o {obj_stem}_{token}\n")
                for vx, vy, vz in verts:
                    f.write(f"v {vx:.9f} {vy:.9f} {vz:.9f}\n")
                if len(normals) != len(verts):
                    normals = _compute_vertex_normals(verts, indices)
                for nx, ny, nz in normals:
                    f.write(f"vn {nx:.9f} {ny:.9f} {nz:.9f}\n")
                f.write("s off\n")
                for i in range(0, len(indices), 3):
                    ia = indices[i] + 1
                    ib = indices[i + 1] + 1
                    ic = indices[i + 2] + 1
                    f.write(f"f {ia}//{ia} {ib}//{ib} {ic}//{ic}\n")
            os.replace(tmp_path, out_path)
        out[mat_name] = out_path
    return out


def _parse_obj_flat_mesh(
    obj_path: str,
) -> Tuple[List[List[float]], List[int], List[List[float]]]:
    cached = _OBJ_FLAT_MESH_CACHE.get(obj_path)
    if cached is not None:
        return cached

    vertices: List[List[float]] = []
    indices: List[int] = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append(
                        [float(parts[1]), float(parts[2]), float(parts[3])]
                    )
                continue
            if line.startswith("f "):
                tokens = line.split()[1:]
                poly: List[int] = []
                for token in tokens:
                    v_tok = token.split("/")[0]
                    if not v_tok:
                        continue
                    idx = int(v_tok)
                    if idx < 0:
                        idx = len(vertices) + 1 + idx
                    poly.append(idx - 1)
                if len(poly) >= 3:
                    for i in range(1, len(poly) - 1):
                        indices.extend([poly[0], poly[i], poly[i + 1]])

    normals = (
        _compute_vertex_normals(vertices, indices)
        if indices
        else [[0.0, 1.0, 0.0] for _ in vertices]
    )
    out = (vertices, indices, normals)
    _OBJ_FLAT_MESH_CACHE[obj_path] = out
    return out


# ---------------------------------------------------------------------------
# SECTION 5: Rect geometry helpers
# ---------------------------------------------------------------------------
def _rect_from_points_xy(
    points_xy: List[Tuple[float, float]],
) -> Tuple[float, float, float, float]:
    min_x = min(px for px, _ in points_xy)
    max_x = max(px for px, _ in points_xy)
    min_y = min(py for _, py in points_xy)
    max_y = max(py for _, py in points_xy)
    return min_x, max_x, min_y, max_y


def _expand_rect_to_min_size(
    rect: Tuple[float, float, float, float],
    *,
    min_w: float,
    min_h: float,
) -> Tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = rect
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half_w = 0.5 * max(max_x - min_x, min_w)
    half_h = 0.5 * max(max_y - min_y, min_h)
    return cx - half_w, cx + half_w, cy - half_h, cy + half_h


def _scale_rect(
    rect: Tuple[float, float, float, float], scale: float
) -> Tuple[float, float, float, float]:
    return rect[0] * scale, rect[1] * scale, rect[2] * scale, rect[3] * scale


def _shift_rect(
    rect: Tuple[float, float, float, float], dx: float, dy: float
) -> Tuple[float, float, float, float]:
    return rect[0] + dx, rect[1] + dx, rect[2] + dy, rect[3] + dy


def _circle_bounds_rect(
    x: float, y: float, radius: float
) -> Tuple[float, float, float, float]:
    return x - radius, x + radius, y - radius, y + radius


def _shrink_rect_from_center(
    rect: Tuple[float, float, float, float], factor: float
) -> Tuple[float, float, float, float]:
    if factor >= 0.999:
        return rect
    factor = max(0.05, float(factor))
    min_x, max_x, min_y, max_y = rect
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half_w = 0.5 * (max_x - min_x) * factor
    half_h = 0.5 * (max_y - min_y) * factor
    return cx - half_w, cx + half_w, cy - half_h, cy + half_h


def _rect_overlap(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> bool:
    return not (a[1] <= b[0] or a[0] >= b[1] or a[3] <= b[2] or a[2] >= b[3])


def _tree_rect_template_unit(obj_path: str) -> Optional[Tuple[tuple, tuple]]:
    cached = _TREE_RECT_TEMPLATE_CACHE.get(obj_path)
    if cached is not None:
        return cached

    verts, _indices, _normals = _parse_obj_flat_mesh(obj_path)
    if not verts:
        return None

    min_x, min_y, min_z, max_x, _max_y, max_z = _obj_bounds_cached(obj_path)
    cx = (min_x + max_x) * 0.5
    cz = (min_z + max_z) * 0.5
    z0 = max(0.0, -min_y)

    quat = p.getQuaternionFromEuler([1.5708, 0.0, 0.0])
    rot = p.getMatrixFromQuaternion(quat)
    r00, r01, r02 = rot[0], rot[1], rot[2]
    r10, r11, r12 = rot[3], rot[4], rot[5]
    r20, r21, r22 = rot[6], rot[7], rot[8]
    px, py, pz = (-cx, cz, z0)

    world_verts: List[Tuple[float, float, float]] = []
    for vx, vy, vz in verts:
        wx = px + r00 * vx + r01 * vy + r02 * vz
        wy = py + r10 * vx + r11 * vy + r12 * vz
        wz = pz + r20 * vx + r21 * vy + r22 * vz
        world_verts.append((wx, wy, wz))

    if not world_verts:
        return None

    min_wz = min(v[2] for v in world_verts)
    max_wz = max(v[2] for v in world_verts)
    h_w = max(1e-6, max_wz - min_wz)
    eps = max(0.005, h_w * 0.0025)
    contact_points = [(vx, vy) for vx, vy, vz in world_verts if vz <= (min_wz + eps)]
    if len(contact_points) < 6:
        eps = max(eps, h_w * 0.005)
        contact_points = [
            (vx, vy) for vx, vy, vz in world_verts if vz <= (min_wz + eps)
        ]
    if len(contact_points) < 3:
        near = sorted(world_verts, key=lambda t: t[2])[:12]
        contact_points = [(vx, vy) for vx, vy, _ in near]
    if len(contact_points) < 3:
        return None

    base_rect = _rect_from_points_xy(contact_points)
    span_rect = _rect_from_points_xy([(vx, vy) for vx, vy, _ in world_verts])
    out = (base_rect, span_rect)
    _TREE_RECT_TEMPLATE_CACHE[obj_path] = out
    return out


def _tree_dual_rects_for_scale(
    obj_path: str, total_scale: float
) -> Optional[Tuple[tuple, tuple]]:
    tpl = _tree_rect_template_unit(obj_path)
    if tpl is None:
        return None
    base_rect_u, span_rect_u = tpl
    base_rect = _scale_rect(base_rect_u, total_scale)
    span_rect = _scale_rect(span_rect_u, total_scale)
    base_rect = _expand_rect_to_min_size(base_rect, min_w=0.35, min_h=0.35)
    return base_rect, span_rect


def _tree_base_rects_from_instances(
    tree_instances: List[Tuple[float, float, str, str, float, float]],
) -> List[Tuple[float, float, float, float]]:
    rects: List[Tuple[float, float, float, float]] = []
    for x, y, category, obj_name, total_scale, _radius in tree_instances:
        obj_path = os.path.join(FOREST_ASSET_DIR, category, obj_name)
        dual = _tree_dual_rects_for_scale(obj_path, total_scale)
        if dual is None:
            continue
        rects.append(_shift_rect(dual[0], x, y))
    return rects


def _tree_span_rects_from_instances(
    tree_instances: List[Tuple[float, float, str, str, float, float]],
) -> List[Tuple[float, float, float, float]]:
    rects: List[Tuple[float, float, float, float]] = []
    for x, y, category, obj_name, total_scale, _radius in tree_instances:
        obj_path = os.path.join(FOREST_ASSET_DIR, category, obj_name)
        dual = _tree_dual_rects_for_scale(obj_path, total_scale)
        if dual is None:
            continue
        rects.append(_shift_rect(dual[1], x, y))
    return rects


def _protected_tree_span_rects_from_instances(
    tree_instances: List[Tuple[float, float, str, str, float, float]],
) -> List[Tuple[float, float, float, float]]:
    rects: List[Tuple[float, float, float, float]] = []
    for x, y, category, obj_name, total_scale, _radius in tree_instances:
        if obj_name not in LOW_CANOPY_PROTECTED_TREE_NAMES:
            continue
        obj_path = os.path.join(FOREST_ASSET_DIR, category, obj_name)
        dual = _tree_dual_rects_for_scale(obj_path, total_scale)
        if dual is None:
            continue
        rects.append(_shift_rect(dual[1], x, y))
    return rects


__all__ = [name for name in globals() if not name.startswith("__")]
