"""
OBJ mesh mountain generator for Type 3 challenge maps.
Ports PR #72 mountain scripts (load_mountains_only + load_ski_village) into a single V4-compatible module.

Subtypes:
    Mountains Only (75%) — procedural snow terrain with scattered peaks/hills
    Ski Village    (25%) — flat village with road grid, buildings, and mountain rings
"""

import math
import os
import random
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import pybullet as p

from swarm.constants import (
    TYPE_3_SCALE_MIN, TYPE_3_SCALE_MAX, TYPE_3_SCALE_SEED_OFFSET,
    MOUNTAIN_SUBTYPE_DISTRIBUTION,
)

# ---------------------------------------------------------------------------
# SECTION 1: Constants & asset paths
# ---------------------------------------------------------------------------
ASSETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets", "maps")
)
CUSTOM_DIR = os.path.join(ASSETS_DIR, "custom")
MOUNTAIN_DIR = os.path.join(CUSTOM_DIR, "mountains")
BUILDING_DIR = os.path.join(CUSTOM_DIR, "buildings")
KENNEY_DIR = os.path.join(ASSETS_DIR, "kenney")
ROAD_ASSET_DIR = os.path.join(KENNEY_DIR, "kenney_roads", "Models", "OBJ format")
SUBURBAN_DIR = os.path.join(KENNEY_DIR, "kenney_suburban")
HOLIDAY_DIR = os.path.join(KENNEY_DIR, "holiday")
CAR_ASSET_DIR = os.path.join(KENNEY_DIR, "kenney_car-kit", "Models", "OBJ format")

PEAK_OBJ = os.path.join(MOUNTAIN_DIR, "mountain_peak.obj")
PEAK_TEX = os.path.join(MOUNTAIN_DIR, "mountain_peak.png")
LANTERN_PATH = os.path.join(HOLIDAY_DIR, "lantern.obj")
LANTERN_ROOF_PATH = os.path.join(BUILDING_DIR, "SnowRoofs", "lantern_roof.obj")

SNOW = [0.98, 0.98, 1.0, 1]
ROAD_COLOR = [0.35, 0.35, 0.38, 1]

TERRAIN_RESOLUTION = 193
TERRAIN_N_OCTAVES = 4

VILLAGE_SIZE = 100.0
ROAD_WIDTH = 6.0
HOUSE_SCALE = 2.0
HOUSE_GAP = 0.3

CAR_ASSETS = [
    "sedan.obj", "hatchback-sports.obj", "taxi.obj",
    "police.obj", "suv.obj", "van.obj", "truck.obj",
]

ROAD_ASSETS = {
    "intersection": "road-crossroad.obj",
    "corner": "road-bend.obj",
    "t_junction": "road-intersection.obj",
    "straight_v": "road-straight.obj",
    "straight_h": "road-straight.obj",
    "crossing": "road-crossing.obj",
    "roundabout": "road-roundabout.obj",
    "dead_end": "road-end-round.obj",
}

HOUSE_SPECS = [
    ("building-type-a.obj", 6.50, 5.14),
    ("building-type-b.obj", 9.14, 5.70),
    ("building-type-c.obj", 6.44, 5.14),
    ("building-type-d.obj", 8.79, 5.14),
    ("building-type-e.obj", 6.50, 5.14),
    ("building-type-f.obj", 7.14, 7.03),
    ("building-type-g.obj", 7.25, 5.89),
    ("building-type-h.obj", 6.50, 4.58),
    ("building-type-i.obj", 6.44, 5.14),
    ("building-type-j.obj", 6.85, 4.58),
    ("building-type-k.obj", 4.61, 5.10),
    ("building-type-l.obj", 5.20, 5.12),
    ("building-type-m.obj", 7.14, 7.14),
    ("building-type-n.obj", 8.93, 6.89),
    ("building-type-o.obj", 6.35, 5.14),
    ("building-type-p.obj", 6.20, 4.95),
    ("building-type-q.obj", 6.20, 4.43),
    ("building-type-r.obj", 5.16, 5.12),
    ("building-type-s.obj", 7.03, 5.44),
    ("building-type-t.obj", 6.60, 7.05),
    ("building-type-u.obj", 7.14, 5.44),
]


# ---------------------------------------------------------------------------
# SECTION 2: Pure-math terrain height (no PyBullet — importable by task_gen)
# ---------------------------------------------------------------------------
def get_global_scale(seed: int) -> float:
    rng = random.Random(seed + TYPE_3_SCALE_SEED_OFFSET)
    return rng.uniform(TYPE_3_SCALE_MIN, TYPE_3_SCALE_MAX)


def _make_noise_params(seed: int, gs: float) -> List[dict]:
    rng = random.Random(seed + 9999)
    params = []
    for i in range(TERRAIN_N_OCTAVES):
        params.append({
            "amp": 1.0 / (1.6 ** i),
            "fx": rng.uniform(0.008, 0.022) / gs * (1.4 ** i),
            "fy": rng.uniform(0.008, 0.022) / gs * (1.4 ** i),
            "px": rng.uniform(0, 2 * math.pi),
            "py": rng.uniform(0, 2 * math.pi),
        })
    return params


def _sample_terrain_height(x: float, y: float, noise_params: List[dict], amplitude: float) -> float:
    h = 0.0
    for o in noise_params:
        h += o["amp"] * math.sin(x * o["fx"] + o["px"]) * math.cos(y * o["fy"] + o["py"])
    return h * amplitude


def get_terrain_z(x: float, y: float, seed: int, gs: float) -> float:
    noise = _make_noise_params(seed, gs)
    amp_rng = random.Random(seed + 12345)
    amp = amp_rng.uniform(2.0 * gs, 5.0 * gs)

    mesh_size = 1500.0 * gs
    res = TERRAIN_RESOLUTION
    half = mesh_size / 2.0
    step = mesh_size / (res - 1)

    gx = (x + half) / step
    gy = (y + half) / step
    gx = max(0.0, min(res - 1.001, gx))
    gy = max(0.0, min(res - 1.001, gy))

    c0, r0 = int(gx), int(gy)
    c1, r1 = min(c0 + 1, res - 1), min(r0 + 1, res - 1)
    tx, ty = gx - c0, gy - r0

    x0 = -half + c0 * step
    x1 = -half + c1 * step
    y0 = -half + r0 * step
    y1 = -half + r1 * step

    h00 = round(_sample_terrain_height(x0, y0, noise, amp), 4)
    h10 = round(_sample_terrain_height(x1, y0, noise, amp), 4)
    h01 = round(_sample_terrain_height(x0, y1, noise, amp), 4)
    h11 = round(_sample_terrain_height(x1, y1, noise, amp), 4)

    return h00 * (1 - tx) * (1 - ty) + h10 * tx * (1 - ty) + h01 * (1 - tx) * ty + h11 * tx * ty


# ---------------------------------------------------------------------------
# SECTION 3: Shape cache
# ---------------------------------------------------------------------------
class _ShapeCache:
    def __init__(self):
        self._cache: Dict = {}

    def clear(self):
        self._cache = {}

    def get(self, cli: int, path: str, scale_vec: list, rgba: Optional[list] = None):
        key = (cli, path, tuple(scale_vec), tuple(rgba) if rgba else None)
        if key in self._cache:
            return self._cache[key]
        kw = {"fileName": path, "meshScale": scale_vec}
        if rgba:
            kw["rgbaColor"] = rgba
            kw["specularColor"] = [0, 0, 0]
        vis = p.createVisualShape(p.GEOM_MESH, physicsClientId=cli, **kw)
        col = p.createCollisionShape(p.GEOM_MESH, fileName=path, meshScale=scale_vec, physicsClientId=cli)
        self._cache[key] = (vis, col)
        return vis, col


_cache = _ShapeCache()


# ---------------------------------------------------------------------------
# SECTION 4: Terrain OBJ generation & spawning
# ---------------------------------------------------------------------------
def _generate_terrain_obj(filepath: str, size: float, res: int,
                          noise_params: List[dict], amplitude: float) -> Tuple[Dict, float]:
    half = size / 2.0
    step = size / (res - 1)
    heights: Dict[Tuple[int, int], float] = {}
    with open(filepath, "w") as f:
        f.write(f"# Procedural snow terrain (amp={amplitude:.1f}m)\n")
        for r in range(res):
            for c in range(res):
                x = -half + c * step
                y = -half + r * step
                z = _sample_terrain_height(x, y, noise_params, amplitude)
                heights[(r, c)] = z
                f.write(f"v {x:.4f} {y:.4f} {z:.4f}\n")
        for r in range(res - 1):
            for c in range(res - 1):
                v00 = r * res + c + 1
                v10 = v00 + 1
                v01 = v00 + res
                v11 = v01 + 1
                f.write(f"f {v00} {v10} {v11}\n")
                f.write(f"f {v00} {v11} {v01}\n")
    return heights, step


def _terrain_z_at(x: float, y: float, heights: Dict, res: int,
                  step: float, half: float) -> float:
    gx = (x + half) / step
    gy = (y + half) / step
    gx = max(0.0, min(res - 1.001, gx))
    gy = max(0.0, min(res - 1.001, gy))
    c0, r0 = int(gx), int(gy)
    c1, r1 = min(c0 + 1, res - 1), min(r0 + 1, res - 1)
    tx, ty = gx - c0, gy - r0
    h00 = heights.get((r0, c0), 0.0)
    h10 = heights.get((r0, c1), 0.0)
    h01 = heights.get((r1, c0), 0.0)
    h11 = heights.get((r1, c1), 0.0)
    return h00 * (1 - tx) * (1 - ty) + h10 * tx * (1 - ty) + h01 * (1 - tx) * ty + h11 * tx * ty


def _spawn_terrain(cli: int, seed: int, obj_dir: str, gs: float):
    noise_params = _make_noise_params(seed, gs)
    amp_rng = random.Random(seed + 12345)
    amplitude = amp_rng.uniform(2.0 * gs, 5.0 * gs)

    mesh_size = 1500.0 * gs
    os.makedirs(obj_dir, exist_ok=True)
    obj_path = os.path.join(obj_dir, f"terrain_seed_{seed}.obj")
    heights, step = _generate_terrain_obj(obj_path, mesh_size, TERRAIN_RESOLUTION, noise_params, amplitude)

    vi = p.createVisualShape(
        p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1],
        rgbaColor=SNOW, specularColor=[0, 0, 0], physicsClientId=cli,
    )
    ci = p.createCollisionShape(
        p.GEOM_MESH, fileName=obj_path, meshScale=[1, 1, 1],
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=cli,
    )
    p.createMultiBody(0, ci, vi, [0, 0, 0], physicsClientId=cli)

    min_h = min(heights.values())
    gv = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[2000 * gs, 2000 * gs, 0.1],
        rgbaColor=SNOW, specularColor=[0, 0, 0], physicsClientId=cli,
    )
    p.createMultiBody(0, -1, gv, [0, 0, min_h - 2.0], physicsClientId=cli)

    res = TERRAIN_RESOLUTION
    half = mesh_size / 2.0

    def get_z(x: float, y: float) -> float:
        return _terrain_z_at(x, y, heights, res, step, half)

    return get_z


# ---------------------------------------------------------------------------
# SECTION 5: Placement helpers
# ---------------------------------------------------------------------------
@dataclass
class _Placed:
    x: float
    y: float
    radius: float


def _too_close(x: float, y: float, radius: float, placed: List[_Placed],
               max_overlap: float = 0.60) -> bool:
    for p0 in placed:
        min_dist = (radius + p0.radius) * (1.0 - max_overlap)
        dx = x - p0.x
        dy = y - p0.y
        if dx * dx + dy * dy < min_dist * min_dist:
            return True
    return False


def _in_safe_zone(x: float, y: float, safe_zones: List[Tuple[float, float]],
                  safe_zone_radius: float) -> bool:
    for sx, sy in safe_zones:
        if math.hypot(x - sx, y - sy) < safe_zone_radius:
            return True
    return False


def _body_intersects_safe_zone(cli: int, body_id: int,
                               safe_zones: Optional[List[Tuple[float, float]]],
                               safe_zone_radius: float) -> bool:
    if body_id is None or not safe_zones:
        return False
    mn, mx = p.getAABB(body_id, physicsClientId=cli)
    x1, y1 = mn[0], mn[1]
    x2, y2 = mx[0], mx[1]
    rr = safe_zone_radius * safe_zone_radius
    for sx, sy in safe_zones:
        nx = max(x1, min(sx, x2))
        ny = max(y1, min(sy, y2))
        dx = sx - nx
        dy = sy - ny
        if dx * dx + dy * dy < rr:
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


def _spawn_mesh_snapped(cli: int, cache: _ShapeCache, path: str,
                        x: float, y: float, rot_deg: float, scale_vec: list,
                        rgba: list = None, sink_z: float = 0.0,
                        tex_id: Optional[int] = None,
                        get_z: Optional[Callable] = None) -> Tuple[int, float]:
    if rgba is None:
        rgba = SNOW
    vis, col = cache.get(cli, path, scale_vec, rgba)
    orn = p.getQuaternionFromEuler([1.5708, 0, math.radians(rot_deg)])
    bid = p.createMultiBody(0, col, vis, [x, y, 0], orn, physicsClientId=cli)
    mn, mx = p.getAABB(bid, physicsClientId=cli)
    terrain_z = get_z(x, y) if get_z else 0.0
    p.resetBasePositionAndOrientation(bid, [x, y, terrain_z - mn[2] - sink_z], orn, physicsClientId=cli)
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
def _build_mountains_only(cli: int, seed: int, gs: float,
                          safe_zones: List[Tuple[float, float]],
                          safe_zone_radius: float) -> Tuple[Callable, List]:
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

    obj_dir = os.path.join(tempfile.gettempdir(), "swarm_terrain_meshes")
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

    def _violates_safe_zone(x: float, y: float, radius: float) -> bool:
        if not safe_zones:
            return False
        required = safe_zone_radius + radius
        required_sq = required * required
        for sx, sy in safe_zones:
            dx = x - sx
            dy = y - sy
            if dx * dx + dy * dy < required_sq:
                return True
        return False

    def spawn_peak(x, y, is_edge=False, scale_var=None):
        if not os.path.exists(PEAK_OBJ):
            return
        if scale_var is None:
            scale_var = rng.uniform(0.85, 1.25)
        base = [95.0 * gs, 95.0 * gs, 145.0 * gs]
        sv = [round(v * scale_var, 2) for v in base]
        sink = (18.0 if is_edge else 14.0) * gs
        bid, h = _spawn_mesh_snapped(
            cli, _cache, PEAK_OBJ, x, y,
            rot_deg=rng.uniform(0, 360), scale_vec=sv, rgba=SNOW,
            sink_z=sink, tex_id=tex_id, get_z=get_z,
        )
        if _body_intersects_safe_zone(cli, bid, safe_zones, safe_zone_radius):
            p.removeBody(bid, physicsClientId=cli)
            return
        peak_heights.append((x, y, h))

    def spawn_hill(x, y, scale):
        path = rng.choice(hills)
        s = round(scale * 2) / 2
        sz = round(s * 0.55 * 2) / 2
        bid, _ = _spawn_mesh_snapped(
            cli, _cache, path, x, y,
            rot_deg=rng.uniform(0, 360), scale_vec=[s, s, sz],
            rgba=SNOW, get_z=get_z,
        )
        if _body_intersects_safe_zone(cli, bid, safe_zones, safe_zone_radius):
            p.removeBody(bid, physicsClientId=cli)

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
            if _violates_safe_zone(x, y, edge_radius):
                continue
            if not _too_close(x, y, edge_radius, placed, max_overlap=0.40):
                placed.append(_Placed(x, y, edge_radius))
                break

    # Center hills
    for _ in range(center_n):
        for _try in range(500):
            x, y = _sample_point_square(rng, half)
            s = rng.uniform(10.0, 18.0) * gs
            r = _estimate_radius_hill(gs, s)
            if _violates_safe_zone(x, y, r):
                continue
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
            if _violates_safe_zone(x, y, r):
                continue
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
            if _violates_safe_zone(x, y, r):
                continue
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
            if _violates_safe_zone(x, y, r):
                continue
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
            if _violates_safe_zone(x, y, r):
                continue
            if not _too_close(x, y, r, placed, max_overlap=0.50):
                placed.append(_Placed(x, y, r))
                path = rng.choice(hills)
                s_q = round(s * 2) / 2
                bid, _ = _spawn_mesh_snapped(
                    cli, _cache, path, x, y,
                    rot_deg=rng.uniform(0, 360), scale_vec=[s_q, s_q, s_q],
                    rgba=SNOW, get_z=get_z,
                )
                if _body_intersects_safe_zone(cli, bid, safe_zones, safe_zone_radius):
                    p.removeBody(bid, physicsClientId=cli)
                break

    return get_z, peak_heights


# ---------------------------------------------------------------------------
# SECTION 7: Village road generation (self-contained, supports map_size param)
# ---------------------------------------------------------------------------
@dataclass
class _Rect:
    x: float
    y: float
    w: float
    h: float


@dataclass
class _RoadTile:
    x: float
    y: float
    type: str
    rotation: int = 0


@dataclass
class _Block:
    id: int
    rect: _Rect
    too_small: bool


class _SeededRNG:
    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def range(self, min_val, max_val):
        return self._rng.uniform(min_val, max_val)

    def rand_int(self, min_val, max_val):
        return self._rng.randint(min_val, max_val)

    def next_float(self):
        return self._rng.random()

    def choice(self, seq):
        if not seq:
            return None
        return self._rng.choice(seq)


def _village_road_positions(rng: _SeededRNG, min_spacing: int, target_area: int,
                            tile_size: float, map_size: float) -> List[float]:
    min_side = math.sqrt(target_area)
    needed_gap = max(min_spacing, min_side)
    raw_min_step = needed_gap + tile_size
    min_step = math.ceil(raw_min_step / tile_size) * tile_size
    max_step = math.ceil((min_step * 1.5) / tile_size) * tile_size
    num_tiles = round(map_size / tile_size)
    if num_tiles < 1:
        num_tiles = 1
    eff_size = num_tiles * tile_size

    raw = [0.0]
    cur = 0.0
    safety = 0
    while cur < eff_size - tile_size and safety < 100:
        safety += 1
        jump = round(rng.range(min_step, max_step) / tile_size) * tile_size
        nxt = cur + jump
        if nxt <= eff_size - tile_size:
            raw.append(nxt)
            cur = nxt
        else:
            break
    if raw[-1] != eff_size - tile_size:
        raw.append(eff_size - tile_size)

    valid = [0.0]
    for i in range(1, len(raw) - 1):
        gap = raw[i] - valid[-1] - tile_size
        if gap >= needed_gap:
            valid.append(raw[i])
    end = eff_size - tile_size
    if end - valid[-1] - tile_size < needed_gap and len(valid) > 1:
        valid.pop()
    valid.append(end)
    return valid


def _village_road_tiles(v_pos: List[float], h_pos: List[float],
                        rng: _SeededRNG, tile_size: float,
                        map_size: float) -> Tuple[List[_RoadTile], list, list, list, list]:
    num_tiles = round(map_size / tile_size)
    eff_size = num_tiles * tile_size
    max_block_size = 18

    tiles = []
    occupied = set()
    v_set, h_set = set(v_pos), set(h_pos)
    h_list = sorted(h_pos)
    v_list = sorted(v_pos)

    removed_h, added_h = [], []
    for y in h_list:
        if y == h_list[0] or y == h_list[-1]:
            continue
        idx = h_list.index(y)
        y_prev, y_next = h_list[idx - 1], h_list[idx + 1]
        for i in range(len(v_list) - 1):
            x1, x2 = v_list[i], v_list[i + 1]
            is_wide = (x2 - x1) > max_block_size
            if rng.next_float() < 0.2:
                if not is_wide:
                    removed_h.append((y, x1, x2))
                else:
                    margin = 2 * tile_size
                    shifts = []
                    if (y - y_prev) - margin >= tile_size:
                        shifts.append(-tile_size)
                    if (y_next - y) - margin >= tile_size:
                        shifts.append(tile_size)
                    if shifts:
                        shift = rng.choice(shifts)
                        removed_h.append((y, x1, x2))
                        added_h.append((y + shift, x1, x2))

    removed_v, added_v = [], []
    for x in v_list:
        if x == v_list[0] or x == v_list[-1]:
            continue
        idx = v_list.index(x)
        x_prev, x_next = v_list[idx - 1], v_list[idx + 1]
        for j in range(len(h_list) - 1):
            y1, y2 = h_list[j], h_list[j + 1]
            is_tall = (y2 - y1) > max_block_size
            if rng.next_float() < 0.2:
                if not is_tall:
                    removed_v.append((x, y1, y2))
                else:
                    margin = 2 * tile_size
                    shifts = []
                    if (x - x_prev) - margin >= tile_size:
                        shifts.append(-tile_size)
                    if (x_next - x) - margin >= tile_size:
                        shifts.append(tile_size)
                    if shifts:
                        shift = rng.choice(shifts)
                        removed_v.append((x, y1, y2))
                        added_v.append((x + shift, y1, y2))

    rm_h_set = set(removed_h)
    rm_v_set = set(removed_v)
    add_h_set = set(added_h)
    add_v_set = set(added_v)

    def _seg_overlaps(seg_coord, seg_a, seg_b, check_set):
        for c, a, b in check_set:
            if c == seg_coord and seg_a < b and seg_b > a:
                return True
        return False

    def is_v_road(x, y1, y2):
        if _seg_overlaps(x, y1, y2, add_v_set):
            return True
        if x in v_set:
            if _seg_overlaps(x, y1, y2, rm_v_set):
                return False
            return True
        return False

    def is_h_road(y, x1, x2):
        if _seg_overlaps(y, x1, x2, add_h_set):
            return True
        if y in h_set:
            if _seg_overlaps(y, x1, x2, rm_h_set):
                return False
            return True
        return False

    all_v = sorted(set(v_pos) | {seg[0] for seg in added_v})
    all_h = sorted(set(h_pos) | {seg[0] for seg in added_h})

    for v in all_v:
        for j in range(len(all_h) - 1):
            y1, y2 = all_h[j], all_h[j + 1]
            for y_step in range(int(y1), int(y2 + tile_size), int(tile_size)):
                fy = float(y_step)
                if fy < 0 or fy >= eff_size:
                    continue
                if (v, fy) in occupied:
                    continue
                if is_v_road(v, y1, y2):
                    occupied.add((v, fy))
                    tiles.append(_RoadTile(v, fy, "straight_v", 0))

    for h in all_h:
        for i in range(len(all_v) - 1):
            x1, x2 = all_v[i], all_v[i + 1]
            for x_step in range(int(x1), int(x2 + tile_size), int(tile_size)):
                fx = float(x_step)
                if fx < 0 or fx >= eff_size:
                    continue
                if (fx, h) in occupied:
                    continue
                if is_h_road(h, x1, x2):
                    occupied.add((fx, h))
                    tiles.append(_RoadTile(fx, h, "straight_h", 0))

    for v in all_v:
        for h in all_h:
            if (v, h) in occupied:
                for t in tiles:
                    if t.x == v and t.y == h:
                        t.type = "intersection"
                        t.rotation = 0
                        break

    return tiles, removed_h, removed_v, added_h, added_v


def _village_extract_blocks(v_pos, h_pos, min_area, removed_h, removed_v,
                            added_h, added_v, tile_size) -> List[_Block]:
    h_rm_set = set(removed_h)
    v_rm_set = set(removed_v)
    n_cols = len(v_pos) - 1
    n_rows = len(h_pos) - 1
    if n_cols <= 0 or n_rows <= 0:
        return []

    parent = {}
    for i in range(n_cols):
        for j in range(n_rows):
            parent[(i, j)] = (i, j)

    def find(cell):
        if parent[cell] != cell:
            parent[cell] = find(parent[cell])
        return parent[cell]

    def union(c1, c2):
        r1, r2 = find(c1), find(c2)
        if r1 != r2:
            parent[r1] = r2

    for i in range(n_cols - 1):
        for j in range(n_rows):
            x = v_pos[i + 1]
            y1, y2 = h_pos[j], h_pos[j + 1]
            if (x, y1, y2) in v_rm_set:
                union((i, j), (i + 1, j))

    for i in range(n_cols):
        for j in range(n_rows - 1):
            y = h_pos[j + 1]
            x1, x2 = v_pos[i], v_pos[i + 1]
            if (y, x1, x2) in h_rm_set:
                union((i, j), (i, j + 1))

    groups: Dict = {}
    for i in range(n_cols):
        for j in range(n_rows):
            root = find((i, j))
            groups.setdefault(root, []).append((i, j))

    blocks = []
    bid = 0
    for cells in groups.values():
        cell_set = set(cells)
        min_i = min(c[0] for c in cells)
        max_i = max(c[0] for c in cells)
        min_j = min(c[1] for c in cells)
        max_j = max(c[1] for c in cells)
        bbox_count = (max_i - min_i + 1) * (max_j - min_j + 1)

        if bbox_count == len(cells):
            x1 = v_pos[min_i] + tile_size
            x2 = v_pos[max_i + 1]
            y1 = h_pos[min_j] + tile_size
            y2 = h_pos[max_j + 1]
            w, h = x2 - x1, y2 - y1
            if w > 1 and h > 1:
                blocks.append(_Block(bid, _Rect(x1, y1, w, h), w * h < min_area * 0.95))
                bid += 1
        else:
            covered = set()
            for j in range(min_j, max_j + 1):
                row_cells = sorted([c[0] for c in cells if c[1] == j])
                if not row_cells:
                    continue
                run_start = row_cells[0]
                for idx in range(len(row_cells)):
                    ci = row_cells[idx]
                    next_ci = row_cells[idx + 1] if idx + 1 < len(row_cells) else None
                    is_end = (next_ci is None or next_ci != ci + 1
                              or (v_pos[ci + 1], h_pos[j], h_pos[j + 1]) not in v_rm_set)
                    if is_end:
                        rect_key = (run_start, j, ci, j)
                        if rect_key not in covered:
                            ry1 = j
                            ry2 = j
                            while ry2 + 1 <= max_j:
                                next_row = ry2 + 1
                                all_present = True
                                for ri in range(run_start, ci + 1):
                                    if (ri, next_row) not in cell_set:
                                        all_present = False
                                        break
                                if not all_present:
                                    break
                                mid_y = h_pos[next_row]
                                all_h_removed = True
                                for ri in range(run_start, ci + 1):
                                    sx1, sx2 = v_pos[ri], v_pos[ri + 1]
                                    if (mid_y, sx1, sx2) not in h_rm_set:
                                        all_h_removed = False
                                        break
                                if not all_h_removed:
                                    break
                                ry2 = next_row

                            already = False
                            for c_key in covered:
                                if (c_key[0] <= run_start and c_key[2] >= ci
                                        and c_key[1] <= ry1 and c_key[3] >= ry2):
                                    already = True
                                    break
                            if not already:
                                covered.add((run_start, ry1, ci, ry2))
                                x1 = v_pos[run_start] + tile_size
                                x2 = v_pos[ci + 1]
                                y1 = h_pos[ry1] + tile_size
                                y2 = h_pos[ry2 + 1]
                                w, h = x2 - x1, y2 - y1
                                if w > 1 and h > 1:
                                    blocks.append(_Block(bid, _Rect(x1, y1, w, h),
                                                         w * h < min_area * 0.95))
                                    bid += 1

                        if next_ci is not None and next_ci == ci + 1:
                            pass
                        run_start = next_ci if next_ci is not None else run_start

    final = []
    queue = list(blocks)
    while queue:
        block = queue.pop(0)
        split = False
        for y_seg, sx1, sx2 in added_h:
            if block.rect.y < y_seg and (y_seg + tile_size) < (block.rect.y + block.rect.h):
                if sx1 < block.rect.x + block.rect.w and sx2 > block.rect.x:
                    h1 = y_seg - block.rect.y
                    b1 = _Block(bid, _Rect(block.rect.x, block.rect.y, block.rect.w, h1), False)
                    bid += 1
                    y2_new = y_seg + tile_size
                    h2 = (block.rect.y + block.rect.h) - y2_new
                    b2 = _Block(bid, _Rect(block.rect.x, y2_new, block.rect.w, h2), False)
                    bid += 1
                    queue.extend([b1, b2])
                    split = True
                    break
        if split:
            continue
        for x_seg, sy1, sy2 in added_v:
            if block.rect.x < x_seg and (x_seg + tile_size) < (block.rect.x + block.rect.w):
                if sy1 < block.rect.y + block.rect.h and sy2 > block.rect.y:
                    w1 = x_seg - block.rect.x
                    b1 = _Block(bid, _Rect(block.rect.x, block.rect.y, w1, block.rect.h), False)
                    bid += 1
                    x2_new = x_seg + tile_size
                    w2 = (block.rect.x + block.rect.w) - x2_new
                    b2 = _Block(bid, _Rect(x2_new, block.rect.y, w2, block.rect.h), False)
                    bid += 1
                    queue.extend([b1, b2])
                    split = True
                    break
        if not split:
            area = block.rect.w * block.rect.h
            block.too_small = area < min_area * 0.95
            final.append(block)

    return final


# ---------------------------------------------------------------------------
# SECTION 8: Village spawning helpers
# ---------------------------------------------------------------------------
def _spawn_village_asset(cli: int, cache: _ShapeCache, path: str,
                         x: float, y: float, z: float, rot_deg: float,
                         scale, rgba: Optional[list] = None) -> Optional[int]:
    if not os.path.exists(path):
        return None
    if isinstance(scale, (list, tuple)):
        scale_vec = list(scale)
    else:
        scale_vec = [scale, scale, scale]
    vis, col = cache.get(cli, path, scale_vec, rgba)
    orn = p.getQuaternionFromEuler([1.5708, 0, math.radians(rot_deg)])
    bid = p.createMultiBody(0, col, vis, [x, y, z], orn, physicsClientId=cli)
    return bid


def _spawn_village_roads(cli: int, cache: _ShapeCache,
                         tiles: List[_RoadTile], offset: float,
                         safe_zones: Optional[List[Tuple[float, float]]] = None,
                         safe_zone_radius: float = 0.0):
    for tile in tiles:
        if tile.type == "roundabout_arm":
            continue
        asset_name = ROAD_ASSETS.get(tile.type)
        if not asset_name:
            continue
        path = os.path.join(ROAD_ASSET_DIR, asset_name)
        if not os.path.exists(path):
            continue
        scale_vec = [ROAD_WIDTH, ROAD_WIDTH, ROAD_WIDTH]
        vis, col = cache.get(cli, path, scale_vec, ROAD_COLOR)
        rot = math.radians(tile.rotation) + math.radians(90)
        orn = p.getQuaternionFromEuler([1.5708, 0, rot])
        fx = tile.x - offset + ROAD_WIDTH / 2
        fy = tile.y - offset + ROAD_WIDTH / 2
        p.createMultiBody(0, col, vis, [fx, fy, 0.08], orn, physicsClientId=cli)


def _spawn_village_buildings(cli: int, cache: _ShapeCache,
                             blocks: List[_Block], offset: float,
                             rng: random.Random,
                             roundabout_centers: List[Tuple[float, float]],
                             tiles: List[_RoadTile] = None,
                             safe_zones: Optional[List[Tuple[float, float]]] = None,
                             safe_zone_radius: float = 0.0):
    row_depth = 3.0
    corner_reserve = row_depth + 0.5

    road_rects = []
    if tiles:
        margin = 0.3
        for tile in tiles:
            if tile.type == "roundabout_arm":
                continue
            rx = tile.x - offset - margin
            ry = tile.y - offset - margin
            rw = ROAD_WIDTH + margin * 2
            road_rects.append((rx, ry, rx + rw, ry + rw))

    def get_house(target_depth):
        candidates = []
        for fn, raw_w, raw_d in HOUSE_SPECS:
            scaled_d = raw_d * HOUSE_SCALE / 5.0
            candidates.append((fn, raw_w * HOUSE_SCALE / 5.0, scaled_d))
        fitting = [c for c in candidates if c[2] <= target_depth + 0.5]
        return rng.choice(fitting) if fitting else rng.choice(candidates)

    placed_aabbs = []

    def spawn_house_at(x, y, rotation, filename):
        for ra_x, ra_y in roundabout_centers:
            if math.hypot(x - ra_x, y - ra_y) < 10.0:
                return
        hw, hd = 1.5, 1.5
        for fn, raw_w, raw_d in HOUSE_SPECS:
            if fn == filename:
                hw = raw_w * HOUSE_SCALE / 5.0 / 2
                hd = raw_d * HOUSE_SCALE / 5.0 / 2
                break
        if safe_zones:
            safe_clearance = safe_zone_radius + math.hypot(hw, hd)
            if _in_safe_zone(x, y, safe_zones, safe_clearance):
                return
        if rotation in (0, 180):
            bx1, by1, bx2, by2 = x - hw, y - hd, x + hw, y + hd
        else:
            bx1, by1, bx2, by2 = x - hd, y - hw, x + hd, y + hw
        if road_rects:
            for rx1, ry1, rx2, ry2 in road_rects:
                if not (bx1 >= rx2 or bx2 <= rx1 or by1 >= ry2 or by2 <= ry1):
                    return
        for px1, py1, px2, py2 in placed_aabbs:
            if not (bx1 >= px2 or bx2 <= px1 or by1 >= py2 or by2 <= py1):
                return
        placed_aabbs.append((bx1, by1, bx2, by2))
        winter_path = os.path.join(BUILDING_DIR, filename)
        if os.path.exists(winter_path):
            final_rot = rotation + 180 if rotation % 180 == 0 else rotation
            house_id = _spawn_village_asset(cli, cache, winter_path, x, y, 0, final_rot, HOUSE_SCALE)
            if _body_intersects_safe_zone(cli, house_id, safe_zones, safe_zone_radius):
                p.removeBody(house_id, physicsClientId=cli)
                return
            roof_name = filename.replace(".obj", "_roof.obj")
            roof_path = os.path.join(BUILDING_DIR, "SnowRoofs", roof_name)
            if os.path.exists(roof_path):
                roof_id = _spawn_village_asset(cli, cache, roof_path, x, y, 0.01, final_rot, HOUSE_SCALE)
                if _body_intersects_safe_zone(cli, roof_id, safe_zones, safe_zone_radius):
                    p.removeBody(roof_id, physicsClientId=cli)
        else:
            sub_path = os.path.join(SUBURBAN_DIR, filename)
            final_rot = rotation + 180 if rotation % 180 == 0 else rotation
            house_id = _spawn_village_asset(cli, cache, sub_path, x, y, 0, final_rot, HOUSE_SCALE)
            if _body_intersects_safe_zone(cli, house_id, safe_zones, safe_zone_radius):
                p.removeBody(house_id, physicsClientId=cli)

    def fill_row(start, end, fixed_pos, rotation, is_vertical=False):
        available = end - start
        if available < 2.0:
            return
        houses = []
        total_w = 0.0
        for _ in range(50):
            fn, w, d = get_house(row_depth)
            if total_w + w + (HOUSE_GAP if houses else 0) > available:
                break
            houses.append((fn, w))
            total_w += w + (HOUSE_GAP if houses else 0)
        if not houses:
            return
        used = sum(h[1] for h in houses) + HOUSE_GAP * (len(houses) - 1)
        cur = start + (available - used) / 2
        for fn, sz in houses:
            center = cur + sz / 2
            if is_vertical:
                spawn_house_at(fixed_pos, center, rotation, fn)
            else:
                spawn_house_at(center, fixed_pos, rotation, fn)
            cur += sz + HOUSE_GAP

    for block in blocks:
        bx, by, bw, bh = block.rect.x, block.rect.y, block.rect.w, block.rect.h
        if bw < 5 or bh < 5:
            continue
        road_margin = 1.5
        left = (bx - offset) + road_margin
        right = (bx - offset) + bw - road_margin
        bottom = (by - offset) + road_margin
        top = (by - offset) + bh - road_margin

        if bw < 10 or bh < 10:
            if bw >= bh:
                fill_row(left, right, (bottom + top) / 2, 0, False)
            else:
                fill_row(bottom, top, (left + right) / 2, 90, True)
        else:
            fill_row(left, right, bottom + row_depth / 2, 0, False)
            fill_row(left, right, top - row_depth / 2, 180, False)
            col_b = bottom + corner_reserve
            col_t = top - corner_reserve
            fill_row(col_b, col_t, left + row_depth / 2, 90, True)
            fill_row(col_b, col_t, right - row_depth / 2, 270, True)


def _spawn_village_lanterns(cli: int, cache: _ShapeCache,
                            tiles: List[_RoadTile], offset: float,
                            rng: random.Random,
                            safe_zones: Optional[List[Tuple[float, float]]] = None,
                            safe_zone_radius: float = 0.0):
    half_tile = ROAD_WIDTH / 2
    lamp_offset_val = half_tile - 0.3
    lantern_scale = [1.05, 1.62, 1.05]
    lantern_z = 0.15
    idx = 0
    for tile in tiles:
        if tile.type == "intersection" or tile.type == "roundabout_arm":
            continue
        idx += 1
        if idx % 4 != 0:
            continue
        cx = tile.x - offset + half_tile
        cy = tile.y - offset + half_tile
        if tile.type == "straight_v":
            for lx, rot in [(cx - lamp_offset_val, 180), (cx + lamp_offset_val, 0)]:
                if safe_zones and _in_safe_zone(lx, cy, safe_zones, safe_zone_radius):
                    continue
                lantern_id = _spawn_village_asset(cli, cache, LANTERN_PATH, lx, cy, lantern_z, rot, lantern_scale)
                if _body_intersects_safe_zone(cli, lantern_id, safe_zones, safe_zone_radius):
                    p.removeBody(lantern_id, physicsClientId=cli)
                    continue
                if os.path.exists(LANTERN_ROOF_PATH):
                    roof_id = _spawn_village_asset(cli, cache, LANTERN_ROOF_PATH, lx, cy, lantern_z + 0.01, rot, lantern_scale)
                    if _body_intersects_safe_zone(cli, roof_id, safe_zones, safe_zone_radius):
                        p.removeBody(roof_id, physicsClientId=cli)
        elif tile.type == "straight_h":
            for ly, rot in [(cy - lamp_offset_val, 270), (cy + lamp_offset_val, 90)]:
                if safe_zones and _in_safe_zone(cx, ly, safe_zones, safe_zone_radius):
                    continue
                lantern_id = _spawn_village_asset(cli, cache, LANTERN_PATH, cx, ly, lantern_z, rot, lantern_scale)
                if _body_intersects_safe_zone(cli, lantern_id, safe_zones, safe_zone_radius):
                    p.removeBody(lantern_id, physicsClientId=cli)
                    continue
                if os.path.exists(LANTERN_ROOF_PATH):
                    roof_id = _spawn_village_asset(cli, cache, LANTERN_ROOF_PATH, cx, ly, lantern_z + 0.01, rot, lantern_scale)
                    if _body_intersects_safe_zone(cli, roof_id, safe_zones, safe_zone_radius):
                        p.removeBody(roof_id, physicsClientId=cli)


def _spawn_village_cars(cli: int, cache: _ShapeCache,
                        tiles: List[_RoadTile], offset: float,
                        rng: random.Random,
                        safe_zones: Optional[List[Tuple[float, float]]] = None,
                        safe_zone_radius: float = 0.0):
    lane_off = ROAD_WIDTH * 0.18
    for tile in tiles:
        if tile.type not in ("straight_v", "straight_h"):
            continue
        if rng.random() >= 0.35:
            continue
        cx = tile.x - offset + ROAD_WIDTH / 2
        cy = tile.y - offset + ROAD_WIDTH / 2
        car_file = rng.choice(CAR_ASSETS)
        car_path = os.path.join(CAR_ASSET_DIR, car_file)
        direction = rng.choice([1, -1])
        if tile.type == "straight_v":
            car_x = cx + lane_off * direction
            car_rot = 180 if direction == 1 else 0
            if safe_zones and _in_safe_zone(car_x, cy, safe_zones, safe_zone_radius):
                continue
            car_id = _spawn_village_asset(cli, cache, car_path, car_x, cy, 0.1, car_rot, 0.58)
            if _body_intersects_safe_zone(cli, car_id, safe_zones, safe_zone_radius):
                p.removeBody(car_id, physicsClientId=cli)
        else:
            car_y = cy + lane_off * (-direction)
            car_rot = 90 if direction == 1 else 270
            if safe_zones and _in_safe_zone(cx, car_y, safe_zones, safe_zone_radius):
                continue
            car_id = _spawn_village_asset(cli, cache, car_path, cx, car_y, 0.1, car_rot, 0.58)
            if _body_intersects_safe_zone(cli, car_id, safe_zones, safe_zone_radius):
                p.removeBody(car_id, physicsClientId=cli)


def _spawn_village_mountain_rings(cli: int, cache: _ShapeCache,
                                  rng: random.Random):
    hills = _hill_objs()
    if not hills:
        return

    tex_id = None
    if os.path.exists(PEAK_TEX):
        tex_id = p.loadTexture(PEAK_TEX, physicsClientId=cli)

    def snap_to_ground(bid, x, y, orn, sink=0.0):
        mn, _ = p.getAABB(bid, physicsClientId=cli)
        z_corr = -mn[2] - sink
        p.resetBasePositionAndOrientation(bid, [x, y, z_corr], orn, physicsClientId=cli)

    # Inner ring: 6 hills
    for i in range(6):
        angle = (2 * math.pi / 6) * i
        r = 165.0 + rng.uniform(-5, 5)
        x, y = r * math.cos(angle), r * math.sin(angle)
        s = round(rng.uniform(10.0, 16.0) * 2) / 2
        hill_path = rng.choice(hills)
        vis, col = cache.get(cli, hill_path, [s, s, s], SNOW)
        orn = p.getQuaternionFromEuler([1.5708, 0, math.radians(rng.uniform(0, 360))])
        bid = p.createMultiBody(0, col, vis, [x, y, 0.0], orn, physicsClientId=cli)
        snap_to_ground(bid, x, y, orn)

    # Middle ring: 9 hills
    for i in range(9):
        angle = (2 * math.pi / 9) * i + (math.pi / 9)
        r = 320.0 + rng.uniform(-15, 15)
        x, y = r * math.cos(angle), r * math.sin(angle)
        s = round(rng.uniform(16.5, 21.0) * 2) / 2
        hill_path = rng.choice(hills)
        vis, col = cache.get(cli, hill_path, [s, s, s], SNOW)
        orn = p.getQuaternionFromEuler([1.5708, 0, math.radians(rng.uniform(0, 360))])
        bid = p.createMultiBody(0, col, vis, [x, y, 0.0], orn, physicsClientId=cli)
        snap_to_ground(bid, x, y, orn)

    # Outer ring: 10 peaks
    if os.path.exists(PEAK_OBJ):
        for i in range(10):
            angle = (2 * math.pi / 10) * i
            r = 550.0 + rng.uniform(-20, 20)
            x, y = r * math.cos(angle), r * math.sin(angle)
            base_scale = [126.4, 126.4, 158.4]
            s_var = round(rng.uniform(0.9, 1.2), 1)
            final_scale = [round(v * s_var, 2) for v in base_scale]
            vis, col = cache.get(cli, PEAK_OBJ, final_scale, SNOW)
            orn = p.getQuaternionFromEuler([1.5708, 0, math.radians(rng.uniform(0, 360))])
            bid = p.createMultiBody(0, col, vis, [x, y, 0.0], orn, physicsClientId=cli)
            snap_to_ground(bid, x, y, orn, sink=10.0)
            if tex_id is not None:
                try:
                    p.changeVisualShape(bid, -1, textureUniqueId=tex_id, physicsClientId=cli)
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# SECTION 9: Ski Village generator
# ---------------------------------------------------------------------------
def _build_ski_village(cli: int, seed: int, gs: float,
                       safe_zones: List[Tuple[float, float]],
                       safe_zone_radius: float) -> Tuple[Callable, List]:
    rng = random.Random(seed)

    ground_size = VILLAGE_SIZE * 20
    ground_half = ground_size / 2
    gv = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[ground_half, ground_half, 0.1],
        rgbaColor=SNOW, specularColor=[0, 0, 0], physicsClientId=cli,
    )
    gc = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[ground_half, ground_half, 0.1],
        physicsClientId=cli,
    )
    p.createMultiBody(0, gc, gv, [0, 0, 0.0], physicsClientId=cli)

    village_rng = _SeededRNG(seed)
    v_pos = _village_road_positions(village_rng, 8, 100, ROAD_WIDTH, VILLAGE_SIZE)
    h_pos = _village_road_positions(village_rng, 8, 100, ROAD_WIDTH, VILLAGE_SIZE)
    tiles, rm_h, rm_v, add_h, add_v = _village_road_tiles(
        v_pos, h_pos, village_rng, ROAD_WIDTH, VILLAGE_SIZE
    )
    blocks = _village_extract_blocks(v_pos, h_pos, 50, rm_h, rm_v, add_h, add_v, ROAD_WIDTH)

    offset = VILLAGE_SIZE / 2
    roundabout_centers = []
    for tile in tiles:
        if tile.type == "roundabout":
            ra_cx = tile.x - offset + ROAD_WIDTH / 2
            ra_cy = tile.y - offset + ROAD_WIDTH / 2
            roundabout_centers.append((ra_cx, ra_cy))

    _spawn_village_roads(
        cli,
        _cache,
        tiles,
        offset,
        safe_zones=safe_zones,
        safe_zone_radius=safe_zone_radius,
    )
    _spawn_village_buildings(
        cli,
        _cache,
        blocks,
        offset,
        rng,
        roundabout_centers,
        tiles,
        safe_zones=safe_zones,
        safe_zone_radius=safe_zone_radius,
    )
    _spawn_village_lanterns(
        cli,
        _cache,
        tiles,
        offset,
        rng,
        safe_zones=safe_zones,
        safe_zone_radius=safe_zone_radius,
    )
    _spawn_village_cars(
        cli,
        _cache,
        tiles,
        offset,
        rng,
        safe_zones=safe_zones,
        safe_zone_radius=safe_zone_radius,
    )
    _spawn_village_mountain_rings(cli, _cache, rng)

    def get_z_flat(x: float, y: float) -> float:
        return 0.0

    return get_z_flat, []


# ---------------------------------------------------------------------------
# SECTION 10: Public API
# ---------------------------------------------------------------------------
def get_mountain_subtype(seed: int) -> int:
    subtype_rng = random.Random(seed + 666666)
    subtypes = list(MOUNTAIN_SUBTYPE_DISTRIBUTION.keys())
    weights = list(MOUNTAIN_SUBTYPE_DISTRIBUTION.values())
    return subtype_rng.choices(subtypes, weights=weights, k=1)[0]


def build_mountains(cli: int, seed: int,
                    safe_zones: List[Tuple[float, float]],
                    safe_zone_radius: float) -> Tuple[Callable, List, float]:
    _cache.clear()

    gs = get_global_scale(seed)
    chosen = get_mountain_subtype(seed)

    if chosen == 1:
        get_z, peaks = _build_mountains_only(cli, seed, gs, safe_zones, safe_zone_radius)
    else:
        get_z, peaks = _build_ski_village(cli, seed, gs, safe_zones, safe_zone_radius)

    return get_z, peaks, gs
