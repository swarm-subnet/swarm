"""Surrounding hills mesh generation for forest maps."""

from ._shared import *
from .ground import _ground_texture_id


# ---------------------------------------------------------------------------
# SECTION 13: Hills ring (surrounding terrain)
# ---------------------------------------------------------------------------
def _load_obj_triangles(obj_path: str) -> Tuple[list, list]:
    cached = _HILL_OBJ_TRI_CACHE.get(obj_path)
    if cached is not None:
        return cached
    verts: list = []
    tris: list = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            if raw.startswith("v "):
                parts = raw.split()
                if len(parts) >= 4:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if not raw.startswith("f "):
                continue
            parts = raw.strip().split()[1:]
            idxs: list = []
            for tok in parts:
                vtok = tok.split("/")[0]
                if not vtok:
                    continue
                vi = int(vtok)
                if vi < 0:
                    vi = len(verts) + vi + 1
                idxs.append(vi - 1)
            if len(idxs) < 3:
                continue
            for i in range(1, len(idxs) - 1):
                tris.append((idxs[0], idxs[i], idxs[i + 1]))
    result = (verts, tris)
    _HILL_OBJ_TRI_CACHE[obj_path] = result
    return result


def _hill_obj_candidates() -> List[str]:
    if not os.path.isdir(MOUNTAIN_ASSET_DIR):
        return []
    return sorted(
        f for f in os.listdir(MOUNTAIN_ASSET_DIR)
        if f.lower().endswith(".obj") and "mountain_peak" not in f.lower()
    )


def _merged_hills_obj_path() -> str:
    return os.path.join(
        HILLS_MESH_CACHE_DIR,
        f"forest_hills_v{HILLS_MESH_VERSION}.obj",
    )


def _ensure_merged_hills_obj() -> Optional[str]:
    hill_candidates = _hill_obj_candidates()
    if not hill_candidates:
        return None

    out_path = _merged_hills_obj_path()
    if os.path.exists(out_path):
        return out_path

    os.makedirs(HILLS_MESH_CACHE_DIR, exist_ok=True)
    rng = random.Random(99173 + 97)

    instances: list = []

    for i in range(6):
        angle = (2.0 * math.pi / 6.0) * i
        r = 165.0 + rng.uniform(-5.0, 5.0)
        x, y = r * math.cos(angle), r * math.sin(angle)
        s = round(rng.uniform(10.0, 16.0) * 2.0) / 2.0
        yaw_deg = rng.uniform(0.0, 360.0)
        instances.append((rng.choice(hill_candidates), s, x, y, yaw_deg))

    for i in range(9):
        angle = (2.0 * math.pi / 9.0) * i + (math.pi / 9.0)
        r = 320.0 + rng.uniform(-15.0, 15.0)
        x, y = r * math.cos(angle), r * math.sin(angle)
        s = round(rng.uniform(16.5, 21.0) * 2.0) / 2.0
        yaw_deg = rng.uniform(0.0, 360.0)
        instances.append((rng.choice(hill_candidates), s, x, y, yaw_deg))

    transformed_instances: list = []
    min_x_all = float("inf")
    min_y_all = float("inf")
    max_x_all = float("-inf")
    max_y_all = float("-inf")

    for hill_file, s, tx, ty, yaw_deg in instances:
        hill_path = os.path.join(MOUNTAIN_ASSET_DIR, hill_file)
        src_v, src_tris = _load_obj_triangles(hill_path)
        quat = p.getQuaternionFromEuler([1.5708, 0.0, math.radians(yaw_deg)])
        m = p.getMatrixFromQuaternion(quat)
        rot = (
            (m[0], m[1], m[2]),
            (m[3], m[4], m[5]),
            (m[6], m[7], m[8]),
        )
        transformed: list = []
        min_z = float("inf")
        for vx, vy, vz in src_v:
            sx, sy, sz = vx * s, vy * s, vz * s
            rx = rot[0][0] * sx + rot[0][1] * sy + rot[0][2] * sz
            ry = rot[1][0] * sx + rot[1][1] * sy + rot[1][2] * sz
            rz = rot[2][0] * sx + rot[2][1] * sy + rot[2][2] * sz
            wx, wy, wz = rx + tx, ry + ty, rz
            transformed.append((wx, wy, wz))
            if wz < min_z:
                min_z = wz
        z_corr = -min_z - 0.02
        corrected: list = []
        for wx, wy, wz in transformed:
            wz2 = wz + z_corr
            corrected.append((wx, wy, wz2))
            min_x_all = min(min_x_all, wx)
            min_y_all = min(min_y_all, wy)
            max_x_all = max(max_x_all, wx)
            max_y_all = max(max_y_all, wy)
        transformed_instances.append((corrected, src_tris))

    if not math.isfinite(min_x_all):
        far_half = HILLS_WORLD_HALF_SIZE_M
    else:
        extent = max(abs(min_x_all), abs(max_x_all), abs(min_y_all), abs(max_y_all))
        far_half = max((GROUND_SIZE_M * 0.5) + 4.0, extent + 4.0)

    merged_v: list = []
    merged_vt: list = []
    merged_f: list = []

    def add_vertex(wx: float, wy: float, wz: float) -> int:
        merged_v.append((wx, wy, wz))
        u = (wx + far_half) / (2.0 * far_half)
        v = (wy + far_half) / (2.0 * far_half)
        merged_vt.append((u, v))
        return len(merged_v)

    i1 = add_vertex(-far_half, -far_half, -0.1)
    i2 = add_vertex(far_half, -far_half, -0.1)
    i3 = add_vertex(far_half, far_half, -0.1)
    i4 = add_vertex(-far_half, far_half, -0.1)
    merged_f.append((i1, i2, i3))
    merged_f.append((i1, i3, i4))

    for corrected_vertices, src_tris in transformed_instances:
        base_idx = len(merged_v)
        for wx, wy, wz in corrected_vertices:
            add_vertex(wx, wy, wz)
        for a, b, c in src_tris:
            merged_f.append((base_idx + a + 1, base_idx + b + 1, base_idx + c + 1))

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Auto-generated merged hills + far ground\n")
        f.write("o MergedHillsTerrain\n")
        for x, y, z in merged_v:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in merged_vt:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        for a, b, c in merged_f:
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    return out_path


def _spawn_hills(
    cli: int, rgba: Optional[List[float]] = None, apply_texture: bool = True,
) -> None:
    merged_obj = _ensure_merged_hills_obj()
    if merged_obj is None:
        return
    if rgba is None:
        rgba = GROUND_RGBA
    terrain_vis = p.createVisualShape(
        p.GEOM_MESH,
        fileName=merged_obj,
        meshScale=[1.0, 1.0, 1.0],
        rgbaColor=rgba,
        specularColor=[0.0, 0.0, 0.0],
        physicsClientId=cli,
    )
    terrain_col = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=merged_obj,
        meshScale=[1.0, 1.0, 1.0],
        physicsClientId=cli,
    )
    terrain_body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=terrain_col,
        baseVisualShapeIndex=terrain_vis,
        basePosition=[0.0, 0.0, 0.0],
        physicsClientId=cli,
    )
    if apply_texture:
        tex_id = _ground_texture_id(cli)
        if tex_id is not None:
            p.changeVisualShape(
                terrain_body, -1, textureUniqueId=tex_id, physicsClientId=cli
            )


__all__ = [name for name in globals() if not name.startswith("__")]
