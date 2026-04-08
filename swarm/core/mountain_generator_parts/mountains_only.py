from ._shared import *
from .terrain import _cache, _ShapeCache, _spawn_terrain


@dataclass
class _Placed:
    x: float
    y: float
    radius: float


def _too_close(
    x: float, y: float, radius: float, placed: List[_Placed], max_overlap: float = 0.60
) -> bool:
    for p0 in placed:
        min_dist = (radius + p0.radius) * (1.0 - max_overlap)
        dx = x - p0.x
        dy = y - p0.y
        if dx * dx + dy * dy < min_dist * min_dist:
            return True
    return False


def _estimate_radius_peak(gs: float, scale_var: float) -> float:
    return 95.0 * gs * scale_var * 0.06


def _estimate_radius_hill(gs: float, scale: float) -> float:
    return scale * gs * 1.8


def _sample_point_square(rng: random.Random, half: float) -> Tuple[float, float]:
    return rng.uniform(-half, half), rng.uniform(-half, half)


def _sample_point_circle(rng: random.Random, radius: float) -> Tuple[float, float]:
    a = rng.uniform(0, 2 * math.pi)
    r = math.sqrt(rng.uniform(0, 1)) * radius
    return r * math.cos(a), r * math.sin(a)


def _hill_objs() -> List[str]:
    cands = []
    for fn in os.listdir(MOUNTAIN_DIR):
        if fn.lower().endswith(".obj") and "peak" not in fn.lower():
            cands.append(os.path.join(MOUNTAIN_DIR, fn))
    cands.sort()
    return cands


def _spawn_mesh_snapped(
    cli: int,
    cache: _ShapeCache,
    path: str,
    x: float,
    y: float,
    rot_deg: float,
    scale_vec: list,
    rgba: list = None,
    sink_z: float = 0.0,
    tex_id: Optional[int] = None,
    get_z: Optional[Callable] = None,
) -> Tuple[int, float]:
    if rgba is None:
        rgba = SNOW
    vis, col = cache.get(cli, path, scale_vec, rgba)
    orn = p.getQuaternionFromEuler([1.5708, 0, math.radians(rot_deg)])
    bid = p.createMultiBody(0, col, vis, [x, y, 0], orn, physicsClientId=cli)
    mn, mx = p.getAABB(bid, physicsClientId=cli)
    terrain_z = get_z(x, y) if get_z else 0.0
    p.resetBasePositionAndOrientation(
        bid, [x, y, terrain_z - mn[2] - sink_z], orn, physicsClientId=cli
    )
    if tex_id is not None:
        try:
            p.changeVisualShape(bid, -1, textureUniqueId=tex_id, physicsClientId=cli)
        except Exception:
            pass
    mn2, mx2 = p.getAABB(bid, physicsClientId=cli)
    return bid, mx2[2] - mn2[2]


# ---------------------------------------------------------------------------
# SECTION 6: Mountains Only generator
# ---------------------------------------------------------------------------
def _build_mountains_only(
    cli: int,
    seed: int,
    gs: float,
    safe_zones: List[Tuple[float, float]],
    safe_zone_radius: float,
) -> Tuple[Callable, List]:
    rng = random.Random(seed)
    map_size = 500.0 * gs
    half = map_size / 2.0
    center_radius = 150.0 * gs
    outer_min = half + 5 * gs
    outer_max = (1500.0 * gs) / 2 - 50 * gs

    hills = _hill_objs()
    if not hills:

        def flat_z(x, y):
            return 0.0

        return flat_z, []

    tex_id = None
    if os.path.exists(PEAK_TEX):
        tex_id = p.loadTexture(PEAK_TEX, physicsClientId=cli)

    obj_dir = str(_terrain_mesh_cache_dir())
    get_z = _spawn_terrain(cli, seed, obj_dir, gs)

    total = rng.randint(70, 90)
    edge_n = rng.randint(6, 12)
    center_n = rng.randint(10, 16)
    remaining = max(0, total - edge_n - center_n)
    large_n = min(rng.randint(8, 14), remaining)
    remaining2 = remaining - large_n
    med_n = min(rng.randint(max(8, remaining2 // 2), remaining2), remaining2)
    small_n = max(0, remaining2 - med_n)

    placed: List[_Placed] = []
    peak_heights: List[Tuple[float, float, float]] = []

    def spawn_peak(x, y, is_edge=False, scale_var=None):
        if not os.path.exists(PEAK_OBJ):
            return
        if scale_var is None:
            scale_var = rng.uniform(0.85, 1.25)
        base = [95.0 * gs, 95.0 * gs, 145.0 * gs]
        sv = [round(v * scale_var, 2) for v in base]
        sink = (18.0 if is_edge else 14.0) * gs
        _, h = _spawn_mesh_snapped(
            cli,
            _cache,
            PEAK_OBJ,
            x,
            y,
            rot_deg=rng.uniform(0, 360),
            scale_vec=sv,
            rgba=SNOW,
            sink_z=sink,
            tex_id=tex_id,
            get_z=get_z,
        )
        peak_heights.append((x, y, h))

    def spawn_hill(x, y, scale):
        path = rng.choice(hills)
        s = round(scale * 2) / 2
        sz = round(s * 0.55 * 2) / 2
        _spawn_mesh_snapped(
            cli,
            _cache,
            path,
            x,
            y,
            rot_deg=rng.uniform(0, 360),
            scale_vec=[s, s, sz],
            rgba=SNOW,
            get_z=get_z,
        )

    # Edge walls
    for _ in range(edge_n):
        for _try in range(400):
            side = rng.choice(["N", "S", "E", "W"])
            t = rng.uniform(-half, half)
            pad = rng.uniform(10.0, 30.0) * gs
            if side == "N":
                x, y = t, half + pad
            elif side == "S":
                x, y = t, -half - pad
            elif side == "E":
                x, y = half + pad, t
            else:
                x, y = -half - pad, t
            edge_radius = _estimate_radius_peak(gs, rng.uniform(0.85, 1.25))
            if not _too_close(x, y, edge_radius, placed, max_overlap=0.40):
                placed.append(_Placed(x, y, edge_radius))
                break

    # Center hills
    for _ in range(center_n):
        for _try in range(500):
            x, y = _sample_point_square(rng, half)
            s = rng.uniform(10.0, 18.0) * gs
            r = _estimate_radius_hill(gs, s)
            if not _too_close(x, y, r, placed, max_overlap=0.60):
                placed.append(_Placed(x, y, r))
                break

    # Large peaks
    large_points = []
    for _ in range(large_n):
        for _try in range(800):
            if rng.random() < 0.70:
                x, y = _sample_point_circle(rng, center_radius)
            else:
                x, y = _sample_point_square(rng, half)
            var = rng.uniform(0.85, 1.25)
            r = _estimate_radius_peak(gs, var)
            if not _too_close(x, y, r, placed, max_overlap=0.45):
                placed.append(_Placed(x, y, r))
                large_points.append((x, y, var))
                break

    # Medium hills
    med_points = []
    for _ in range(med_n):
        for _try in range(800):
            x, y = _sample_point_square(rng, half)
            s = rng.uniform(14.0, 22.0) * gs
            r = _estimate_radius_hill(gs, s)
            if not _too_close(x, y, r, placed, max_overlap=0.60):
                placed.append(_Placed(x, y, r))
                med_points.append((x, y, s))
                break

    # Small hills
    small_points = []
    for _ in range(small_n):
        for _try in range(800):
            x, y = _sample_point_square(rng, half)
            s = rng.uniform(7.0, 13.0) * gs
            r = _estimate_radius_hill(gs, s)
            if not _too_close(x, y, r, placed):
                placed.append(_Placed(x, y, r))
                small_points.append((x, y, s))
                break

    # --- Spawn ---
    edge_set = set()
    for pl in placed:
        if abs(pl.x) > half or abs(pl.y) > half:
            edge_set.add((pl.x, pl.y))
            spawn_peak(pl.x, pl.y, is_edge=True)

    large_set = {(x, y) for x, y, _ in large_points}
    med_set = {(x, y) for x, y, _ in med_points}
    small_set = {(x, y) for x, y, _ in small_points}

    for pl in placed:
        key = (pl.x, pl.y)
        if key in edge_set or key in large_set or key in med_set or key in small_set:
            continue
        if rng.random() < 0.30:
            spawn_peak(pl.x, pl.y, is_edge=False)
        else:
            spawn_hill(pl.x, pl.y, rng.uniform(10.0, 18.0) * gs)

    for x, y, var in large_points:
        spawn_peak(x, y, is_edge=False, scale_var=var)

    for x, y, s in med_points:
        spawn_hill(x, y, s)
    for x, y, s in small_points:
        spawn_hill(x, y, s)

    # Outer hills
    target_n = rng.randint(45, 65)
    for _ in range(target_n):
        for _try in range(300):
            x = rng.uniform(-outer_max, outer_max)
            y = rng.uniform(-outer_max, outer_max)
            if abs(x) < outer_min and abs(y) < outer_min:
                continue
            s = rng.uniform(10.0, 22.0) * gs
            r = _estimate_radius_hill(gs, s)
            if not _too_close(x, y, r, placed, max_overlap=0.50):
                placed.append(_Placed(x, y, r))
                path = rng.choice(hills)
                s_q = round(s * 2) / 2
                _spawn_mesh_snapped(
                    cli,
                    _cache,
                    path,
                    x,
                    y,
                    rot_deg=rng.uniform(0, 360),
                    scale_vec=[s_q, s_q, s_q],
                    rgba=SNOW,
                    get_z=get_z,
                )
                break

    return get_z, peak_heights


# ---------------------------------------------------------------------------
# SECTION 7: Village road generation (self-contained, supports map_size param)
