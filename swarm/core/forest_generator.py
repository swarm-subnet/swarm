"""
OBJ mesh forest generator for Type 6 challenge maps.
Ports PR #72 forest_map logic into a V4-compatible module.

Ground: 100×100 m flat playable area (96×96 m after edge margin).
Mode:   Normal (mode_id=1) — green foliage, all asset categories.
Difficulty: Normal (difficulty_id=2) — 170 target trees + bushes/rocks/logs.
"""

import math
import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import pybullet as p

# ---------------------------------------------------------------------------
# SECTION 1: Constants & asset paths
# ---------------------------------------------------------------------------
ASSETS_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, "assets", "maps"
    )
)
FOREST_ASSET_DIR = os.path.join(ASSETS_DIR, "forest", "quaternius_ultimate_nature")
MOUNTAIN_ASSET_DIR = os.path.join(ASSETS_DIR, "custom", "mountains")

GROUND_SIZE_M = 100.0
GROUND_HALF_THICKNESS_M = 0.05
GROUND_RGBA = [0.72, 0.75, 0.72, 1.0]
GROUND_RGBA_AUTUMN = [0.58, 0.52, 0.38, 1.0]
GROUND_RGBA_DEAD = [0.46, 0.42, 0.35, 1.0]
GROUND_RGBA_SNOW = [0.98, 0.98, 1.0, 1.0]

MAP_MODE_CONFIG = {
    1: {"name": "Normal", "primary_category": "normal", "use_misc_willow": True},
    2: {"name": "Autumn", "primary_category": "autumn", "use_misc_willow": False},
    3: {"name": "Snow", "primary_category": "snow", "use_misc_willow": False},
    4: {"name": "No Leaves", "primary_category": "dead", "use_misc_willow": False},
}

GROUND_TEXTURE_RES = 384
GROUND_TEXTURE_SEED = 13731
GROUND_TEXTURE_DIR = os.path.join(ASSETS_DIR, "forest", "textures")
GROUND_TEXTURE_PATH = os.path.join(GROUND_TEXTURE_DIR, "forest_ground.bmp")

HILLS_MESH_CACHE_DIR = os.path.join(ASSETS_DIR, "forest", "terrain_cache")
HILLS_MESH_VERSION = 12
HILLS_WORLD_HALF_SIZE_M = 520.0

FOREST_EDGE_MARGIN_M = 2.0
PREVIEW_UNIFORM_SCALE = 2.0
FAST_BUILD_MODE = True
FAST_SCALE_STEP = 0.10
TREE_BATCH_MAX_LINKS = 127

# Tree scale (effective scale = PREVIEW_UNIFORM_SCALE * TREE_SCALE = 4.275)
TREE_SCALE_MIN = 2.1375
TREE_SCALE_MAX = 2.1375
SHRUB_SCALE_MIN = 1.05
SHRUB_SCALE_MAX = 1.05
ROCK_STUMP_SCALE_MIN = 1.25
ROCK_STUMP_SCALE_MAX = 1.25
LOG_SCALE_MIN = 1.25
LOG_SCALE_MAX = 1.25
GROUND_COVER_SCALE_MIN = 1.125
GROUND_COVER_SCALE_MAX = 1.125

# Density and difficulty tuning
DENSITY_MULTIPLIER = 1.10
DIFFICULTY_CONFIG = {
    1: {
        "name": "Easy",
        "tree_count": 130, "tree_clearance_m": 0.14,
        "bush_count": 32, "bush_clearance_m": 0.58,
        "rock_stump_count": 20, "rock_stump_clearance_m": 0.64,
        "log_count": 10, "log_clearance_m": 0.42,
        "ground_cover_count": 20, "ground_cover_clearance_m": 0.20,
    },
    2: {
        "name": "Normal",
        "tree_count": 170, "tree_clearance_m": 0.10,
        "bush_count": 48, "bush_clearance_m": 0.48,
        "rock_stump_count": 34, "rock_stump_clearance_m": 0.54,
        "log_count": 16, "log_clearance_m": 0.36,
        "ground_cover_count": 34, "ground_cover_clearance_m": 0.16,
    },
    3: {
        "name": "Hard",
        "tree_count": 210, "tree_clearance_m": 0.06,
        "bush_count": 66, "bush_clearance_m": 0.38,
        "rock_stump_count": 50, "rock_stump_clearance_m": 0.46,
        "log_count": 24, "log_clearance_m": 0.30,
        "ground_cover_count": 52, "ground_cover_clearance_m": 0.12,
    },
}

CLASS_DENSITY_MULTIPLIER = {
    "trees": 1.184865, "bushes": 0.6567321796875, "logs": 1.12,
    "rocks": 0.91854, "stumps": 0.91854,
    "plants": 1.0206, "cactus": 1.0206,
}
TREE_DIFFICULTY_MULTIPLIER = {1: 1.3115, 2: 1.3975, 3: 1.10}
DIFFICULTY_DENSITY_MULTIPLIER = {1: 1.00, 2: 1.00, 3: 1.20}

# MTL color processing
MTL_COLOR_GAMMA = 0.66
MTL_COLOR_MULTIPLIER = 1.62

# Tree family and spacing
BIRCH_TREE_PREFIX = "BirchTree_"
COMMON_TREE_PREFIX = "CommonTree_"
PALM_TREE_PREFIX = "PalmTree_"
PINE_TREE_PREFIX = "PineTree_"
WILLOW_TREE_PREFIX = "Willow_"
TREE_FAMILY_PREFIXES = (
    BIRCH_TREE_PREFIX, COMMON_TREE_PREFIX, PALM_TREE_PREFIX,
    PINE_TREE_PREFIX, WILLOW_TREE_PREFIX,
)
BIRCH_TREE_MAX_RATIO = 0.18
BUSH_BERRIES_PREFIX = "BushBerries_"
BUSH_BERRIES_WEIGHT = 0.85
SNOW_DEAD_TREE_WEIGHT = 0.28
TREE_FAMILY_REPEAT_PENALTY = 0.22
TREE_FAMILY_NEARBY_PENALTY = 0.40
TREE_FAMILY_CLUSTER_RULES = {
    PINE_TREE_PREFIX: {
        1: {"radius_m": 11.0, "max_neighbors": 0},
        2: {"radius_m": 10.0, "max_neighbors": 0},
        3: {"radius_m": 8.0, "max_neighbors": 1},
    },
    WILLOW_TREE_PREFIX: {
        1: {"radius_m": 9.5, "max_neighbors": 0},
        2: {"radius_m": 8.5, "max_neighbors": 0},
        3: {"radius_m": 7.0, "max_neighbors": 1},
    },
}

TREE_SPACING_RADIUS_MULTIPLIER_DEFAULT = 0.32
TREE_SPACING_RADIUS_MIN_M = 0.30
TREE_TREE_CLEARANCE_FACTOR = 0.30
TREE_SPACING_RADIUS_MULTIPLIER_DEFAULT_BY_DIFFICULTY = {1: 0.54, 2: 0.48, 3: 0.32}
TREE_TREE_CLEARANCE_FACTOR_BY_DIFFICULTY = {1: 1.00, 2: 0.90, 3: 0.30}
TREE_CANOPY_OVERLAP_SCALE_BY_DIFFICULTY = {1: 0.90, 2: 0.79, 3: 0.64}
TREE_SPACING_RADIUS_BY_PREFIX_BY_DIFFICULTY = {
    1: {"PalmTree_": 0.30, "Willow_": 0.42, "BirchTree_": 0.44, "PineTree_": 1.00, "CommonTree_": 0.52},
    2: {"PalmTree_": 0.26, "Willow_": 0.36, "BirchTree_": 0.40, "PineTree_": 1.00, "CommonTree_": 0.46},
    3: {"PalmTree_": 0.12, "Willow_": 0.20, "BirchTree_": 0.24, "PineTree_": 1.00, "CommonTree_": 0.34},
}
LOW_CANOPY_PROTECTED_TREE_NAMES = {
    "PineTree_1.obj", "PineTree_2.obj", "PineTree_3.obj",
    "PineTree_Autumn_1.obj", "PineTree_Autumn_2.obj", "PineTree_Autumn_3.obj",
    "PineTree_Snow_1.obj", "PineTree_Snow_2.obj", "PineTree_Snow_3.obj",
}

TREE_OCCUPANCY_RADIUS_MULTIPLIER_DEFAULT = 0.70
TREE_OCCUPANCY_RADIUS_MIN_M = 0.45
TREE_OCCUPANCY_RADIUS_BY_PREFIX = {
    "PalmTree_": 0.95, "Willow_": 0.84, "BirchTree_": 0.76,
    "PineTree_": 1.00, "CommonTree_": 0.82,
}
TREE_LOCAL_CLUSTER_MAX_BY_DIFFICULTY = {1: 1, 2: 1, 3: 2}
TREE_LOCAL_CLUSTER_SEARCH_BY_DIFFICULTY_M = {1: 3.6, 2: 3.2, 3: 2.8}

BUSH_DISTRIBUTION_CELL_SIZE_M = 8.0
BUSH_DISTRIBUTION_MAX_PER_CELL = 1

SMALL_ASSET_EDGE_MARGIN_M = 6.0
SMALL_ASSET_CENTER_REGION_RATIO = 0.68
SMALL_ASSET_CENTER_BIAS = 0.70
SMALL_ASSET_TREE_OCCUPANCY_SCALE = {
    "logs": 0.60, "bushes": 0.58, "rocks": 0.55,
    "stumps": 0.53, "plants": 0.48, "cactus": 0.48,
}
SMALL_ASSET_TREE_OCCUPANCY_CAP_M = {
    "logs": 2.40, "bushes": 1.90, "rocks": 1.70,
    "stumps": 1.70, "plants": 1.30, "cactus": 1.30,
}
SMALL_ASSET_TREE_OCCUPANCY_MIN_M = 0.40

ROCK_STUMP_MODEL_WEIGHT_BONUS = {
    "Rock_Moss_4.obj": 3.0, "Rock_Moss_6.obj": 3.0,
    "Rock_Moss_7.obj": 3.0, "TreeStump_Moss.obj": 3.0,
}
ROCK_STUMP_MODEL_SCALE_FACTOR: Dict[str, float] = {}
GROUND_COVER_MODEL_WEIGHT_BONUS = {
    "Plant_1.obj": 0.45, "Plant_4.obj": 0.70, "Plant_5.obj": 4.0,
    "Cactus_1.obj": 2.4, "Cactus_2.obj": 2.4, "Cactus_3.obj": 2.4,
    "Cactus_4.obj": 2.4, "Cactus_5.obj": 2.4,
    "CactusFlower_1.obj": 2.4, "CactusFlowers_2.obj": 2.4,
    "CactusFlowers_3.obj": 2.4, "CactusFlowers_4.obj": 2.4,
    "CactusFlowers_5.obj": 2.4,
}
GROUND_COVER_MODEL_SCALE_FACTOR: Dict[str, float] = {}
NORMAL_ONLY_GROUND_COVER_ALLOWLIST = {
    "Corn_1.obj", "Corn_2.obj", "Lilypad.obj",
    "Cactus_1.obj", "Cactus_2.obj", "Cactus_3.obj",
    "Cactus_4.obj", "Cactus_5.obj",
    "CactusFlower_1.obj", "CactusFlowers_2.obj",
    "CactusFlowers_3.obj", "CactusFlowers_4.obj", "CactusFlowers_5.obj",
}
NORMAL_AUTUMN_DEAD_GROUND_COVER_ALLOWLIST = {"Wheat.obj"}
AUTUMN_GROUND_COVER_ALLOWLIST = {"Plant_2.obj", "Plant_5.obj", "Wheat.obj"}
DEAD_GROUND_COVER_ALLOWLIST = {"Plant_2.obj"}
NORMAL_ONLY_SINGLE_SPAWN_GROUND_COVER: set = set()
ROCK_STUMP_TOTAL_MULTIPLIER = 1.02
ROCK_STUMP_PRIMARY_RATIO = 0.78
GROUND_COVER_PLANT_PRIMARY_RATIO = 0.75
LOG_CLEARANCE_MULTIPLIER = 0.90
STUMP_CLEARANCE_MULTIPLIER = 0.90

TRUNK_COUNT_BOUNDS_BY_DIFFICULTY = {
    1: {"logs_max": 7, "stumps_min": 6, "stumps_max": 7},
    2: {"logs_max": 8, "stumps_min": 7, "stumps_max": 8},
    3: {"logs_min": 5, "logs_max": 5, "stumps_min": 5, "stumps_max": 5},
}
TRUNK_OCCUPANCY_MULTIPLIER_BY_DIFFICULTY = {3: {"logs": 0.65, "stumps": 0.82}}
TRUNK_CLEARANCE_MULTIPLIER_BY_DIFFICULTY = {3: {"logs": 0.72, "stumps": 0.82}}
TRUNK_FALLBACK_FILL_BY_DIFFICULTY = {3: {"logs": True, "stumps": True}}

GROUND_COVER_MODE_WEIGHT_BONUS = {
    1: {"Plant_2.obj": 0.25, "Wheat.obj": 0.25},
}
ROCK_STUMP_MODE_WEIGHT_BONUS = {1: {"TreeStump.obj": 0.25}}

# Scoring mode: Normal (1) and difficulty Normal (2)
SCORING_MODE_ID = 1
SCORING_DIFFICULTY_ID = 2

# ---------------------------------------------------------------------------
# SECTION 2: Module-level geometry caches (pure data, client-independent)
# ---------------------------------------------------------------------------
_OBJ_BOUNDS_CACHE: Dict[str, tuple] = {}
_OBJ_PLANAR_RADIUS_CACHE: Dict[str, float] = {}
_OBJ_MATERIAL_MESH_CACHE: Dict[str, dict] = {}
_OBJ_FLAT_MESH_CACHE: Dict[str, tuple] = {}
_TREE_RECT_TEMPLATE_CACHE: Dict[str, Optional[tuple]] = {}
_CLASS_ASSET_CACHE: Dict[str, dict] = {}
_HILL_OBJ_TRI_CACHE: Dict[str, tuple] = {}

# Per-client caches for PyBullet shape IDs (invalidated on each build)
_CLI_COL_CACHE: Dict[int, Dict[tuple, int]] = {}
_CLI_VIS_CACHE: Dict[int, Dict[tuple, list]] = {}
_CLI_TEX_CACHE: Dict[int, Optional[int]] = {}


# ---------------------------------------------------------------------------
# SECTION 3: Spatial grid for placement collision
# ---------------------------------------------------------------------------
class _SpatialGrid:
    __slots__ = ('cells', 'cell_size', 'max_radius')

    def __init__(self, cell_size: float = 4.0):
        self.cells: dict = {}
        self.cell_size = cell_size
        self.max_radius = 0.0

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, x: float, y: float, radius: float) -> None:
        if radius > self.max_radius:
            self.max_radius = radius
        self.cells.setdefault(self._key(x, y), []).append((x, y, radius))

    def has_conflict(self, x: float, y: float, radius: float, clearance: float) -> bool:
        if not self.cells:
            return False
        search_dist = radius + self.max_radius + clearance
        search_cells = int(search_dist / self.cell_size) + 1
        cx, cy = self._key(x, y)
        for dx in range(-search_cells, search_cells + 1):
            for dy in range(-search_cells, search_cells + 1):
                cell = self.cells.get((cx + dx, cy + dy))
                if cell is None:
                    continue
                for ox, oy, orad in cell:
                    ddx = x - ox
                    ddy = y - oy
                    min_dist = radius + orad + clearance
                    if (ddx * ddx + ddy * ddy) < (min_dist * min_dist):
                        return True
        return False

    def count_neighbors(self, x: float, y: float, search_radius: float) -> int:
        if not self.cells:
            return 0
        search_cells = int(search_radius / self.cell_size) + 1
        cx, cy = self._key(x, y)
        search_r2 = search_radius * search_radius
        count = 0
        for dx in range(-search_cells, search_cells + 1):
            for dy in range(-search_cells, search_cells + 1):
                cell = self.cells.get((cx + dx, cy + dy))
                if cell is None:
                    continue
                for ox, oy, _ in cell:
                    ddx = x - ox
                    ddy = y - oy
                    if (ddx * ddx + ddy * ddy) <= search_r2:
                        count += 1
        return count


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


# ---------------------------------------------------------------------------
# SECTION 6: Ground texture generation
# ---------------------------------------------------------------------------
def _clamp_u8(v: float) -> int:
    return max(0, min(255, int(round(v))))


def _hash_noise_01(x: int, y: int, seed: int) -> float:
    n = (x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)
    n = (n << 13) ^ n
    return (
        1.0
        - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF)
        / 1073741824.0
    )


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


def _ensure_ground_texture() -> str:
    if os.path.exists(GROUND_TEXTURE_PATH):
        return GROUND_TEXTURE_PATH
    os.makedirs(os.path.dirname(GROUND_TEXTURE_PATH), exist_ok=True)
    w = h = GROUND_TEXTURE_RES
    data = bytearray(w * h * 3)
    seed = GROUND_TEXTURE_SEED
    rng = random.Random(seed)
    grass_a = (96.0, 130.0, 88.0)
    grass_b = (109.0, 146.0, 95.0)
    for y in range(h):
        for x in range(w):
            idx = (y * w + x) * 3
            macro = 0.5 + 0.5 * math.sin((x * 0.022) + (y * 0.017))
            noise = _hash_noise_01(x, y, seed) * 0.5 + 0.5
            t = (0.7 * macro) + (0.3 * noise)
            r = grass_a[0] * (1.0 - t) + grass_b[0] * t
            g = grass_a[1] * (1.0 - t) + grass_b[1] * t
            b = grass_a[2] * (1.0 - t) + grass_b[2] * t
            grain = (_hash_noise_01(x * 3, y * 3, seed + 19) * 0.5 + 0.5) - 0.5
            data[idx + 0] = _clamp_u8(r + grain * 14.0)
            data[idx + 1] = _clamp_u8(g + grain * 18.0)
            data[idx + 2] = _clamp_u8(b + grain * 12.0)
    for _ in range(56):
        cx = rng.uniform(0.0, w - 1.0)
        cy = rng.uniform(0.0, h - 1.0)
        radius = rng.uniform(16.0, 52.0)
        dirt = (
            rng.uniform(103.0, 126.0),
            rng.uniform(93.0, 111.0),
            rng.uniform(74.0, 93.0),
        )
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
                edge = 1.0 - d2
                blend = (edge * edge) * rng.uniform(0.28, 0.58)
                idx = (py * w + px) * 3
                data[idx + 0] = _clamp_u8(
                    data[idx + 0] * (1.0 - blend) + dirt[0] * blend
                )
                data[idx + 1] = _clamp_u8(
                    data[idx + 1] * (1.0 - blend) + dirt[1] * blend
                )
                data[idx + 2] = _clamp_u8(
                    data[idx + 2] * (1.0 - blend) + dirt[2] * blend
                )
    _write_bmp24(GROUND_TEXTURE_PATH, w, h, data)
    return GROUND_TEXTURE_PATH


def _ground_texture_id(cli: int) -> Optional[int]:
    if cli in _CLI_TEX_CACHE:
        return _CLI_TEX_CACHE[cli]
    tex_path = _ensure_ground_texture()
    try:
        tex_id = p.loadTexture(tex_path, physicsClientId=cli)
    except Exception:
        tex_id = None
    _CLI_TEX_CACHE[cli] = tex_id
    return tex_id


# ---------------------------------------------------------------------------
# SECTION 7: Ground spawning
# ---------------------------------------------------------------------------
def _ground_rgba_for_mode(mode_id: int) -> List[float]:
    cat = MAP_MODE_CONFIG[_clamp_mode_id(mode_id)]["primary_category"]
    if cat == "autumn":
        return GROUND_RGBA_AUTUMN
    if cat == "dead":
        return GROUND_RGBA_DEAD
    if cat == "snow":
        return GROUND_RGBA_SNOW
    return GROUND_RGBA


def _spawn_ground(cli: int, mode_id: int = 1) -> None:
    rgba = _ground_rgba_for_mode(mode_id)
    half = GROUND_SIZE_M * 0.5
    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half, half, GROUND_HALF_THICKNESS_M],
        physicsClientId=cli,
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half, half, GROUND_HALF_THICKNESS_M],
        rgbaColor=rgba,
        specularColor=[0.0, 0.0, 0.0],
        physicsClientId=cli,
    )
    body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 0.0, -GROUND_HALF_THICKNESS_M],
        physicsClientId=cli,
    )
    if mode_id == 1:
        tex_id = _ground_texture_id(cli)
        if tex_id is not None:
            p.changeVisualShape(body, -1, textureUniqueId=tex_id, physicsClientId=cli)


# ---------------------------------------------------------------------------
# SECTION 8: Asset resolution
# ---------------------------------------------------------------------------
def _list_obj_names(category: str) -> List[str]:
    cat_dir = os.path.join(FOREST_ASSET_DIR, category)
    if not os.path.isdir(cat_dir):
        return []
    return sorted(
        name for name in os.listdir(cat_dir) if name.lower().endswith(".obj")
    )


def _clamp_mode_id(mode_id: int) -> int:
    return max(1, min(4, int(mode_id)))


def _resolve_assets_for_class(mode_id: int) -> Dict[str, List[Tuple[str, str]]]:
    mode_id = _clamp_mode_id(mode_id)
    cache_key = f"mode_{mode_id}"
    cached = _CLASS_ASSET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mode_cfg = MAP_MODE_CONFIG[mode_id]
    primary_cat = mode_cfg["primary_category"]
    is_dry = primary_cat in ("autumn", "dead", "dead_snow")
    is_snow = primary_cat in ("snow", "dead_snow")

    primary_objs = _list_obj_names(primary_cat)
    normal_objs = _list_obj_names("normal")
    misc_objs = _list_obj_names("misc")

    if is_snow:
        primary_mode_objs = [n for n in primary_objs if "Snow" in n]
    else:
        primary_mode_objs = [n for n in primary_objs if "Snow" not in n]
    normal_nosnow = [n for n in normal_objs if "Snow" not in n]

    trees = [
        (primary_cat, name) for name in primary_mode_objs
        if ("Tree" in name or name.startswith("Willow_"))
        and "Stump" not in name and "Bush" not in name
    ]
    if primary_cat == "snow":
        dead_snow_objs = _list_obj_names("dead_snow")
        trees += [
            ("dead_snow", n) for n in dead_snow_objs
            if "Tree" in n and "Stump" not in n and "Bush" not in n
        ]
    if mode_cfg["use_misc_willow"]:
        trees += [("misc", name) for name in misc_objs if "Willow_" in name]

    bushes = [(primary_cat, n) for n in primary_mode_objs if "Bush" in n]
    rocks = [(primary_cat, n) for n in primary_mode_objs if n.startswith("Rock")]
    stumps = [(primary_cat, n) for n in primary_mode_objs if "TreeStump" in n]
    logs = [(primary_cat, n) for n in primary_mode_objs if "WoodLog" in n]

    if primary_cat == "snow":
        snow_stumps = [("normal", n) for n in normal_objs if n == "TreeStump_Snow.obj"]
        stumps += snow_stumps

    if is_snow:
        plants: List[Tuple[str, str]] = []
        cactus: List[Tuple[str, str]] = []
    else:
        plants = [
            ("misc", n) for n in misc_objs
            if n.startswith("Plant_") or n in {"Flowers.obj"}
        ]
        if mode_id == 1:
            plants += [("misc", n) for n in misc_objs if n in NORMAL_ONLY_GROUND_COVER_ALLOWLIST]
        if mode_id in (1, 2, 4):
            plants += [("misc", n) for n in misc_objs if n in NORMAL_AUTUMN_DEAD_GROUND_COVER_ALLOWLIST]
        cactus = [
            ("misc", n) for n in misc_objs
            if n.startswith("Cactus_") or n.startswith("CactusFlower_")
            or n.startswith("CactusFlowers_")
        ]

    if primary_cat == "autumn":
        plants = [(c, n) for c, n in plants if n != "Flowers.obj"]

    if is_dry:
        bushes = []
        if primary_cat == "autumn":
            plants = [(c, n) for c, n in plants if n in AUTUMN_GROUND_COVER_ALLOWLIST]
        elif primary_cat == "dead":
            plants = [("misc", n) for n in misc_objs if n in DEAD_GROUND_COVER_ALLOWLIST]
        else:
            plants = []
        logs = [(c, n) for c, n in logs if "Moss" not in n]
        rocks = [(c, n) for c, n in rocks if "Moss" not in n]
        stumps = [(c, n) for c, n in stumps if "Moss" not in n]
        cactus = []

    if not is_snow:
        logs = [(c, n) for c, n in logs if "Snow" not in n]

    if not trees:
        trees = [("normal", "CommonTree_1.obj")]
    if not bushes and not is_dry:
        bushes = [("normal", n) for n in normal_nosnow if "Bush" in n]
        if primary_cat == "autumn":
            bushes = [(c, n) for c, n in bushes if not n.startswith(BUSH_BERRIES_PREFIX)]
        if not bushes:
            bushes = [("normal", "Bush_1.obj")]
    if not rocks:
        fallback = [("normal", n) for n in normal_nosnow if n.startswith("Rock")]
        if is_dry:
            fallback = [(c, n) for c, n in fallback if "Moss" not in n]
        rocks = fallback if fallback else [("normal", "Rock_1.obj")]
    if not stumps:
        fallback = [("normal", n) for n in normal_nosnow if "TreeStump" in n]
        if is_dry:
            fallback = [(c, n) for c, n in fallback if "Moss" not in n]
        stumps = fallback if fallback else [("normal", "TreeStump.obj")]
    if not logs:
        normal_logs = [("normal", n) for n in normal_objs if "WoodLog" in n]
        if is_snow:
            snow_logs = [(c, n) for c, n in normal_logs if "Snow" in n]
            logs = snow_logs if snow_logs else normal_logs
        else:
            no_snow = [(c, n) for c, n in normal_logs if "Snow" not in n]
            logs = no_snow if no_snow else normal_logs
        if is_dry:
            logs = [(c, n) for c, n in logs if "Moss" not in n and "Snow" not in n]
        if not logs:
            logs = [("normal", "WoodLog.obj")]

    resolved = {
        "trees": trees,
        "bushes": bushes,
        "logs": logs,
        "rocks": rocks,
        "stumps": stumps,
        "plants": plants,
        "cactus": cactus,
        "rocks_stumps": rocks + stumps,
        "ground_cover": plants + cactus,
    }
    _CLASS_ASSET_CACHE[cache_key] = resolved
    return resolved


# ---------------------------------------------------------------------------
# SECTION 9: Tree family selection helpers
# ---------------------------------------------------------------------------
def _map_half_extent() -> float:
    return GROUND_SIZE_M * 0.5


def _tree_spacing_radius(
    obj_name: str, canopy_radius: float, difficulty_id: int
) -> float:
    mul = TREE_SPACING_RADIUS_MULTIPLIER_DEFAULT_BY_DIFFICULTY.get(
        difficulty_id, TREE_SPACING_RADIUS_MULTIPLIER_DEFAULT
    )
    by_prefix = TREE_SPACING_RADIUS_BY_PREFIX_BY_DIFFICULTY.get(difficulty_id, {})
    for prefix, ratio in by_prefix.items():
        if obj_name.startswith(prefix):
            mul = ratio
            break
    overlap_scale = TREE_CANOPY_OVERLAP_SCALE_BY_DIFFICULTY.get(difficulty_id, 1.0)
    return max(TREE_SPACING_RADIUS_MIN_M, canopy_radius * mul * overlap_scale)


def _tree_occupancy_radius(obj_name: str, canopy_radius: float) -> float:
    mul = TREE_OCCUPANCY_RADIUS_MULTIPLIER_DEFAULT
    for prefix, ratio in TREE_OCCUPANCY_RADIUS_BY_PREFIX.items():
        if obj_name.startswith(prefix):
            mul = ratio
            break
    return max(TREE_OCCUPANCY_RADIUS_MIN_M, canopy_radius * mul)


def _tree_family_prefix(obj_name: str) -> str:
    for prefix in TREE_FAMILY_PREFIXES:
        if obj_name.startswith(prefix):
            return prefix
    return obj_name.split("_", 1)[0]


def _build_tree_family_assets(
    assets: List[Tuple[str, str]],
) -> Dict[str, dict]:
    families: Dict[str, dict] = {}
    for category, obj_name in assets:
        family = _tree_family_prefix(obj_name)
        info = families.setdefault(
            family, {"total_weight": 0.0, "weighted_assets": []}
        )
        weight = SNOW_DEAD_TREE_WEIGHT if category == "dead_snow" else 1.0
        if weight <= 0.0:
            continue
        info["total_weight"] += weight
        info["weighted_assets"].append((info["total_weight"], category, obj_name))
    return families


def _pick_weighted_tree_from_family(
    rng: random.Random, family_info: dict
) -> Tuple[str, str]:
    weighted = family_info.get("weighted_assets", [])
    total = float(family_info.get("total_weight", 0.0))
    if not weighted or total <= 0.0:
        raise RuntimeError("tree family has no weighted assets")
    r = rng.uniform(0.0, total)
    for cum_w, category, obj_name in weighted:
        if r <= cum_w:
            return category, obj_name
    return weighted[-1][1], weighted[-1][2]


def _count_tree_family_neighbors(
    placed_families: List[Tuple[float, float, str]],
    *, x: float, y: float, family: str, radius_m: float,
) -> int:
    radius_sq = radius_m * radius_m
    count = 0
    for ox, oy, other_family in placed_families:
        if other_family != family:
            continue
        dx, dy = x - ox, y - oy
        if (dx * dx + dy * dy) <= radius_sq:
            count += 1
    return count


def _rank_tree_families_for_point(
    rng: random.Random,
    *, x: float, y: float, difficulty_id: int,
    family_assets: Dict[str, dict],
    family_counts: Dict[str, int],
    placed_families: List[Tuple[float, float, str]],
    max_birch_count: int, birch_placed: int,
) -> List[str]:
    ranked: List[Tuple[float, str]] = []
    for family, info in family_assets.items():
        if family == BIRCH_TREE_PREFIX and birch_placed >= max_birch_count:
            continue
        base_weight = float(info.get("total_weight", 0.0))
        if base_weight <= 0.0:
            continue
        repeat_penalty = 1.0 / (
            1.0 + family_counts.get(family, 0) * TREE_FAMILY_REPEAT_PENALTY
        )
        nearby_count = _count_tree_family_neighbors(
            placed_families, x=x, y=y, family=family,
            radius_m=max(
                6.0,
                TREE_LOCAL_CLUSTER_SEARCH_BY_DIFFICULTY_M.get(difficulty_id, 3.0)
                * 1.8,
            ),
        )
        score = (
            base_weight
            * repeat_penalty
            / (1.0 + nearby_count * TREE_FAMILY_NEARBY_PENALTY)
        )
        cluster_rule = TREE_FAMILY_CLUSTER_RULES.get(family, {}).get(difficulty_id)
        if cluster_rule is not None:
            same_neighbors = _count_tree_family_neighbors(
                placed_families, x=x, y=y, family=family,
                radius_m=cluster_rule["radius_m"],
            )
            if same_neighbors > cluster_rule["max_neighbors"]:
                continue
            score /= 1.0 + same_neighbors * 0.75
        score *= rng.uniform(0.85, 1.15)
        ranked.append((score, family))
    ranked.sort(reverse=True)
    return [family for _score, family in ranked if _score > 0.0]


# ---------------------------------------------------------------------------
# SECTION 10: Instance picking (placement logic)
# ---------------------------------------------------------------------------
def _tree_candidate_points(
    rng: random.Random, count: int
) -> List[Tuple[float, float]]:
    half = _map_half_extent() - FOREST_EDGE_MARGIN_M
    width = half * 2.0
    cells_side = max(8, int(math.ceil(math.sqrt(count * 1.18))))
    cell = width / cells_side
    jitter = cell * 0.33
    points: List[Tuple[float, float]] = []
    for gy in range(cells_side):
        row_offset = 0.5 * cell if (gy % 2) else 0.0
        for gx in range(cells_side):
            x = -half + (gx + 0.5) * cell + row_offset
            if x > half:
                x -= width
            y = -half + (gy + 0.5) * cell
            x += rng.uniform(-jitter, jitter)
            y += rng.uniform(-jitter, jitter)
            x = max(-half, min(half, x))
            y = max(-half, min(half, y))
            points.append((x, y))
    rng.shuffle(points)
    for _ in range(count * 3):
        points.append((rng.uniform(-half, half), rng.uniform(-half, half)))
    return points


def _small_asset_half_extent() -> float:
    return _map_half_extent() - max(FOREST_EDGE_MARGIN_M, SMALL_ASSET_EDGE_MARGIN_M)


def _extend_small_asset_candidates(
    rng: random.Random,
    candidates: List[Tuple[float, float]],
    *, count: int, half: float,
) -> None:
    inner_half = half * SMALL_ASSET_CENTER_REGION_RATIO
    for _ in range(count):
        if rng.random() < SMALL_ASSET_CENTER_BIAS:
            x = rng.uniform(-inner_half, inner_half)
            y = rng.uniform(-inner_half, inner_half)
        else:
            x = rng.uniform(-half, half)
            y = rng.uniform(-half, half)
        candidates.append((x, y))


def _scaled_occupied_instances(
    instances: List[Tuple[float, float, str, str, float, float]],
    *, radius_scale: float, min_radius: float = 0.0,
    max_radius: Optional[float] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    scaled: List[Tuple[float, float, str, str, float, float]] = []
    for x, y, category, obj_name, total_scale, radius in instances:
        out_r = max(min_radius, radius * radius_scale)
        if max_radius is not None:
            out_r = min(out_r, max_radius)
        scaled.append((x, y, category, obj_name, total_scale, out_r))
    return scaled


def _pick_tree_instances(
    rng: random.Random, *, count: int,
    assets: List[Tuple[str, str]], clearance_m: float, difficulty_id: int,
) -> List[Tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    placed: List[Tuple[float, float, str, str, float, float]] = []
    placed_families: List[Tuple[float, float, str]] = []
    grid = _SpatialGrid()
    placed_base_rects: List[Tuple[float, float, float, float]] = []
    placed_span_rects: List[Tuple[float, float, float, float]] = []
    candidates = _tree_candidate_points(rng, count)
    half = _map_half_extent() - FOREST_EDGE_MARGIN_M
    family_assets = _build_tree_family_assets(assets)
    family_counts: Dict[str, int] = {f: 0 for f in family_assets}
    non_birch = [f for f in family_assets if f != BIRCH_TREE_PREFIX]
    max_birch = int(math.floor(count * BIRCH_TREE_MAX_RATIO)) if non_birch else count
    birch_placed = 0
    cluster_max = TREE_LOCAL_CLUSTER_MAX_BY_DIFFICULTY.get(difficulty_id, 1)
    cluster_search_m = TREE_LOCAL_CLUSTER_SEARCH_BY_DIFFICULTY_M.get(difficulty_id, 3.0)
    canopy_overlap_scale = TREE_CANOPY_OVERLAP_SCALE_BY_DIFFICULTY.get(
        difficulty_id, 1.0
    )

    for relax in (1.0, 0.85, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            ranked = _rank_tree_families_for_point(
                rng, x=x, y=y, difficulty_id=difficulty_id,
                family_assets=family_assets, family_counts=family_counts,
                placed_families=placed_families,
                max_birch_count=max_birch, birch_placed=birch_placed,
            )
            if not ranked:
                continue

            for family in ranked:
                info = family_assets.get(family)
                if not info:
                    continue
                category, obj_name = _pick_weighted_tree_from_family(rng, info)
                obj_path = os.path.join(FOREST_ASSET_DIR, category, obj_name)
                if not os.path.exists(obj_path):
                    continue

                size_mul = rng.uniform(TREE_SCALE_MIN, TREE_SCALE_MAX)
                total_scale = PREVIEW_UNIFORM_SCALE * size_mul
                if FAST_BUILD_MODE and FAST_SCALE_STEP > 0.0:
                    total_scale = max(
                        0.01,
                        round(total_scale / FAST_SCALE_STEP) * FAST_SCALE_STEP,
                    )
                canopy_r = _obj_planar_radius_cached(obj_path) * total_scale
                spacing_r = _tree_spacing_radius(obj_name, canopy_r, difficulty_id)
                occupancy_r = _tree_occupancy_radius(obj_name, canopy_r)
                rects = _tree_dual_rects_for_scale(obj_path, total_scale)
                if rects is not None:
                    base_local, span_local = rects
                    base_world = _shift_rect(base_local, x, y)
                    span_world = _shift_rect(span_local, x, y)
                    if (
                        base_world[0] < -half or base_world[1] > half
                        or base_world[2] < -half or base_world[3] > half
                    ):
                        continue
                    if (
                        difficulty_id in (2, 3)
                        and obj_name in LOW_CANOPY_PROTECTED_TREE_NAMES
                    ):
                        span_collision = span_world
                    else:
                        span_collision = _shrink_rect_from_center(
                            span_world,
                            max(0.05, canopy_overlap_scale * relax),
                        )
                    blocked = False
                    for i in range(len(placed_base_rects)):
                        if _rect_overlap(base_world, placed_base_rects[i]):
                            blocked = True
                            break
                        if _rect_overlap(span_collision, placed_span_rects[i]):
                            blocked = True
                            break
                    if blocked:
                        continue
                else:
                    if (
                        (x - occupancy_r) < -half or (x + occupancy_r) > half
                        or (y - occupancy_r) < -half or (y + occupancy_r) > half
                    ):
                        continue
                    clearance_factor = TREE_TREE_CLEARANCE_FACTOR_BY_DIFFICULTY.get(
                        difficulty_id, TREE_TREE_CLEARANCE_FACTOR,
                    )
                    if grid.has_conflict(
                        x, y, spacing_r, clearance_m * relax * clearance_factor
                    ):
                        continue

                local_neighbors = grid.count_neighbors(
                    x, y, max(cluster_search_m, spacing_r * 2.2),
                )
                if local_neighbors > (cluster_max + (1 if relax < 0.9 else 0)):
                    continue

                placed.append((x, y, category, obj_name, total_scale, occupancy_r))
                placed_families.append((x, y, family))
                family_counts[family] = family_counts.get(family, 0) + 1
                grid.insert(x, y, spacing_r)
                if rects is not None:
                    placed_base_rects.append(base_world)
                    placed_span_rects.append(span_collision)
                if obj_name.startswith(BIRCH_TREE_PREFIX):
                    birch_placed += 1
                break

        if len(placed) >= count:
            break
        rng.shuffle(candidates)

    return placed


def _pick_shrub_instances(
    rng: random.Random, *, count: int,
    assets: List[Tuple[str, str]], clearance_m: float,
    tree_instances: List[Tuple[float, float, str, str, float, float]],
    tree_base_rects: Optional[List[tuple]] = None,
    protected_tree_span_rects: Optional[List[tuple]] = None,
    occupied_instances: Optional[List[Tuple[float, float, str, str, float, float]]] = None,
    tree_occupancy_scale: float = 1.0,
    tree_occupancy_cap_m: Optional[float] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    half = _small_asset_half_extent()
    weighted: List[Tuple[float, str, str]] = []
    total_w = 0.0
    for cat, name in assets:
        w = BUSH_BERRIES_WEIGHT if name.startswith(BUSH_BERRIES_PREFIX) else 1.0
        if w <= 0.0:
            continue
        total_w += w
        weighted.append((total_w, cat, name))
    if not weighted:
        return []

    candidates: List[Tuple[float, float]] = list(
        _tree_candidate_points(rng, max(count * 2, 120))
    )
    for tx, ty, _, _, _, tr in tree_instances:
        for _ in range(2):
            ang = rng.uniform(0.0, math.tau)
            dist = tr + rng.uniform(1.1, 3.2)
            nx = tx + math.cos(ang) * dist
            ny = ty + math.sin(ang) * dist
            if -half <= nx <= half and -half <= ny <= half:
                candidates.append((nx, ny))
    _extend_small_asset_candidates(rng, candidates, count=count * 8, half=half)
    rng.shuffle(candidates)

    occupied: List[Tuple[float, float, float]] = []
    for ox, oy, _, _, _, orad in tree_instances:
        tr = max(SMALL_ASSET_TREE_OCCUPANCY_MIN_M, orad * tree_occupancy_scale)
        if tree_occupancy_cap_m is not None:
            tr = min(tr, tree_occupancy_cap_m)
        occupied.append((ox, oy, tr))
    if occupied_instances:
        occupied.extend((ox, oy, orad) for ox, oy, _, _, _, orad in occupied_instances)
    grid = _SpatialGrid()
    for ox, oy, orad in occupied:
        grid.insert(ox, oy, orad)
    per_cell: Dict[Tuple[int, int], int] = {}
    placed: List[Tuple[float, float, str, str, float, float]] = []

    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            cell_key = (
                int((x + half) // BUSH_DISTRIBUTION_CELL_SIZE_M),
                int((y + half) // BUSH_DISTRIBUTION_CELL_SIZE_M),
            )
            if per_cell.get(cell_key, 0) >= BUSH_DISTRIBUTION_MAX_PER_CELL:
                continue
            r = rng.uniform(0.0, total_w)
            cat = name = ""
            for cum, c, n in weighted:
                if r <= cum:
                    cat, name = c, n
                    break
            else:
                cat, name = weighted[-1][1], weighted[-1][2]
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            size_mul = rng.uniform(SHRUB_SCALE_MIN, SHRUB_SCALE_MAX)
            total_scale = PREVIEW_UNIFORM_SCALE * size_mul
            radius = _obj_planar_radius_cached(obj_path) * total_scale
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if grid.has_conflict(x, y, radius, clearance_m * relax):
                continue
            placed.append((x, y, cat, name, total_scale, radius))
            grid.insert(x, y, radius)
            per_cell[cell_key] = per_cell.get(cell_key, 0) + 1
        if len(placed) >= count:
            break
        rng.shuffle(candidates)
    return placed


def _pick_rock_stump_instances(
    rng: random.Random, *, count: int,
    assets: List[Tuple[str, str]], mode_id: int, clearance_m: float,
    occupied_instances: List[Tuple[float, float, str, str, float, float]],
    tree_base_rects: Optional[List[tuple]] = None,
    protected_tree_span_rects: Optional[List[tuple]] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    weighted: List[Tuple[float, str, str]] = []
    total_w = 0.0
    mode_wb = ROCK_STUMP_MODE_WEIGHT_BONUS.get(mode_id, {})
    for cat, name in assets:
        w = ROCK_STUMP_MODEL_WEIGHT_BONUS.get(name, 1.0) * mode_wb.get(name, 1.0)
        if w <= 0.0:
            continue
        total_w += w
        weighted.append((total_w, cat, name))
    if not weighted:
        return []

    half = _small_asset_half_extent()
    candidates: List[Tuple[float, float]] = list(
        _tree_candidate_points(rng, max(count * 2, 120))
    )
    _extend_small_asset_candidates(rng, candidates, count=count * 10, half=half)
    rng.shuffle(candidates)

    occupied: List[Tuple[float, float, float]] = [
        (ox, oy, orad) for ox, oy, _, _, _, orad in occupied_instances
    ]
    grid = _SpatialGrid()
    for ox, oy, orad in occupied:
        grid.insert(ox, oy, orad)
    placed: List[Tuple[float, float, str, str, float, float]] = []

    priority_names = [
        n
        for n in ROCK_STUMP_MODEL_WEIGHT_BONUS
        if any(obj_name == n for _, obj_name in assets)
    ]
    for target_name in priority_names:
        if len(placed) >= count:
            break
        target = [(c, n) for c, n in assets if n == target_name]
        if not target:
            continue
        for _ in range(140):
            x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            cat, name = rng.choice(target)
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            sm = rng.uniform(ROCK_STUMP_SCALE_MIN, ROCK_STUMP_SCALE_MAX)
            sm *= ROCK_STUMP_MODEL_SCALE_FACTOR.get(name, 1.0)
            ts = PREVIEW_UNIFORM_SCALE * sm
            radius = _obj_planar_radius_cached(obj_path) * ts
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if grid.has_conflict(x, y, radius, clearance_m * 0.35):
                continue
            placed.append((x, y, cat, name, ts, radius))
            grid.insert(x, y, radius)
            break

    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            r = rng.uniform(0.0, total_w)
            cat = name = ""
            for cum, c, n in weighted:
                if r <= cum:
                    cat, name = c, n
                    break
            else:
                cat, name = weighted[-1][1], weighted[-1][2]
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            sm = rng.uniform(ROCK_STUMP_SCALE_MIN, ROCK_STUMP_SCALE_MAX)
            sm *= ROCK_STUMP_MODEL_SCALE_FACTOR.get(name, 1.0)
            ts = PREVIEW_UNIFORM_SCALE * sm
            radius = _obj_planar_radius_cached(obj_path) * ts
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if grid.has_conflict(x, y, radius, clearance_m * relax):
                continue
            placed.append((x, y, cat, name, ts, radius))
            grid.insert(x, y, radius)
        if len(placed) >= count:
            break
        rng.shuffle(candidates)
    return placed


def _pick_log_instances(
    rng: random.Random, *, count: int,
    assets: List[Tuple[str, str]], clearance_m: float,
    occupied_instances: List[Tuple[float, float, str, str, float, float]],
    tree_base_rects: Optional[List[tuple]] = None,
    protected_tree_span_rects: Optional[List[tuple]] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    half = _small_asset_half_extent()
    candidates: List[Tuple[float, float]] = list(
        _tree_candidate_points(rng, max(count * 3, 120))
    )
    _extend_small_asset_candidates(rng, candidates, count=count * 12, half=half)
    rng.shuffle(candidates)

    occupied = [(ox, oy, orad) for ox, oy, _, _, _, orad in occupied_instances]
    grid = _SpatialGrid()
    for ox, oy, orad in occupied:
        grid.insert(ox, oy, orad)
    placed: List[Tuple[float, float, str, str, float, float]] = []

    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            cat, name = rng.choice(assets)
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            sm = rng.uniform(LOG_SCALE_MIN, LOG_SCALE_MAX)
            ts = PREVIEW_UNIFORM_SCALE * sm
            radius = _obj_planar_radius_cached(obj_path) * ts
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if grid.has_conflict(x, y, radius, clearance_m * relax):
                continue
            placed.append((x, y, cat, name, ts, radius))
            grid.insert(x, y, radius)
        if len(placed) >= count:
            break
        rng.shuffle(candidates)
    return placed


def _pick_ground_cover_instances(
    rng: random.Random, *, count: int,
    assets: List[Tuple[str, str]], mode_id: int, clearance_m: float,
    occupied_instances: List[Tuple[float, float, str, str, float, float]],
    tree_base_rects: Optional[List[tuple]] = None,
    tree_span_rects: Optional[List[tuple]] = None,
    protected_tree_span_rects: Optional[List[tuple]] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    weighted: List[Tuple[float, str, str]] = []
    total_w = 0.0
    mode_wb = GROUND_COVER_MODE_WEIGHT_BONUS.get(mode_id, {})
    for cat, name in assets:
        w = GROUND_COVER_MODEL_WEIGHT_BONUS.get(name, 1.0) * mode_wb.get(name, 1.0)
        if w <= 0.0:
            continue
        total_w += w
        weighted.append((total_w, cat, name))
    if not weighted:
        return []

    half = _small_asset_half_extent()
    candidates: List[Tuple[float, float]] = list(
        _tree_candidate_points(rng, max(count * 2, 80))
    )
    _extend_small_asset_candidates(rng, candidates, count=count * 12, half=half)
    rng.shuffle(candidates)

    occupied: List[Tuple[float, float, float]] = [
        (ox, oy, orad) for ox, oy, _, _, _, orad in occupied_instances
    ]
    placed: List[Tuple[float, float, str, str, float, float]] = []
    placed_by_name: Dict[str, int] = {}

    priority_names = [
        n for n, w in GROUND_COVER_MODEL_WEIGHT_BONUS.items()
        if w > 1.0 and any(obj_name == n for _, obj_name in assets)
    ]
    for target_name in priority_names:
        if len(placed) >= count:
            break
        target = [(c, n) for c, n in assets if n == target_name]
        if not target:
            continue
        for _ in range(180):
            x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            cat, name = rng.choice(target)
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            sm = rng.uniform(GROUND_COVER_SCALE_MIN, GROUND_COVER_SCALE_MAX)
            sm *= GROUND_COVER_MODEL_SCALE_FACTOR.get(name, 1.0)
            ts = PREVIEW_UNIFORM_SCALE * sm
            radius = _obj_planar_radius_cached(obj_path) * ts
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            if name in NORMAL_ONLY_SINGLE_SPAWN_GROUND_COVER and placed_by_name.get(name, 0) >= 1:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if name.startswith("Corn_") and tree_span_rects and any(
                _rect_overlap(cr, r) for r in tree_span_rects
            ):
                continue
            keep = True
            for ox, oy, orad in occupied:
                dx, dy = x - ox, y - oy
                min_dist = radius + orad + clearance_m * 0.65
                if (dx * dx + dy * dy) < (min_dist * min_dist):
                    keep = False
                    break
            if not keep:
                continue
            placed.append((x, y, cat, name, ts, radius))
            occupied.append((x, y, radius))
            placed_by_name[name] = placed_by_name.get(name, 0) + 1
            break

    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            r = rng.uniform(0.0, total_w)
            cat = name = ""
            for cum, c, n in weighted:
                if r <= cum:
                    cat, name = c, n
                    break
            else:
                cat, name = weighted[-1][1], weighted[-1][2]
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            sm = rng.uniform(GROUND_COVER_SCALE_MIN, GROUND_COVER_SCALE_MAX)
            sm *= GROUND_COVER_MODEL_SCALE_FACTOR.get(name, 1.0)
            ts = PREVIEW_UNIFORM_SCALE * sm
            radius = _obj_planar_radius_cached(obj_path) * ts
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            if name in NORMAL_ONLY_SINGLE_SPAWN_GROUND_COVER and placed_by_name.get(name, 0) >= 1:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if name.startswith("Corn_") and tree_span_rects and any(
                _rect_overlap(cr, r) for r in tree_span_rects
            ):
                continue
            keep = True
            for ox, oy, orad in occupied:
                dx, dy = x - ox, y - oy
                min_dist = radius + orad + clearance_m * relax
                if (dx * dx + dy * dy) < (min_dist * min_dist):
                    keep = False
                    break
            if not keep:
                continue
            placed.append((x, y, cat, name, ts, radius))
            occupied.append((x, y, radius))
            placed_by_name[name] = placed_by_name.get(name, 0) + 1
        if len(placed) >= count:
            break
        rng.shuffle(candidates)
    return placed


def _split_asset_count(
    total: int,
    primary: List[Tuple[str, str]],
    secondary: List[Tuple[str, str]],
    *, primary_ratio: Optional[float] = None,
    secondary_cap: Optional[int] = None,
) -> Tuple[int, int]:
    if total <= 0 or (not primary and not secondary):
        return 0, 0
    if not secondary:
        return total, 0
    if not primary:
        sec = total if secondary_cap is None else min(total, max(0, secondary_cap))
        return 0, sec
    if primary_ratio is None:
        n = max(1, len(primary) + len(secondary))
        primary_ratio = len(primary) / n
    pri = max(0, min(total, int(round(total * primary_ratio))))
    sec = total - pri
    if secondary_cap is not None:
        sec = min(sec, max(0, secondary_cap))
        pri = total - sec
    return pri, sec


# ---------------------------------------------------------------------------
# SECTION 11: PyBullet spawning (with physicsClientId)
# ---------------------------------------------------------------------------
def _collision_shape_for_obj(cli: int, obj_path: str, scale: float) -> int:
    cli_cache = _CLI_COL_CACHE.setdefault(cli, {})
    key = (obj_path, round(scale, 4))
    cached = cli_cache.get(key)
    if cached is not None:
        return cached
    flags = p.GEOM_FORCE_CONCAVE_TRIMESH if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH") else 0
    shape = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale],
        flags=flags,
        physicsClientId=cli,
    )
    if shape < 0:
        raise RuntimeError(f"Failed to create collision shape for {obj_path}")
    cli_cache[key] = shape
    return shape


def _spawn_colored_obj(
    cli: int, *, obj_path: str, scale: float, double_sided_flags: int,
) -> List[int]:
    cli_cache = _CLI_VIS_CACHE.setdefault(cli, {})
    cache_key = (obj_path, round(scale, 4), int(double_sided_flags))
    vis_ids = cli_cache.get(cache_key)
    if vis_ids is None:
        material_meshes = _parse_obj_material_meshes(obj_path)
        mtl_colors = _parse_mtl_diffuse_colors(_obj_mtl_path(obj_path))
        default_rgba = [0.7, 0.7, 0.7, 1.0]
        vis_ids = []
        if material_meshes:
            for mat_name, (verts, indices, normals) in material_meshes.items():
                rgba = mtl_colors.get(mat_name, default_rgba)
                kwargs = {
                    "vertices": verts, "indices": indices, "normals": normals,
                    "meshScale": [scale, scale, scale],
                    "rgbaColor": rgba,
                    "specularColor": [0.0, 0.0, 0.0],
                }
                if double_sided_flags:
                    kwargs["flags"] = double_sided_flags
                vis = p.createVisualShape(
                    p.GEOM_MESH, physicsClientId=cli, **kwargs
                )
                if vis >= 0:
                    vis_ids.append(vis)
        if not vis_ids:
            kwargs = {
                "fileName": obj_path,
                "meshScale": [scale, scale, scale],
                "specularColor": [0.0, 0.0, 0.0],
            }
            if double_sided_flags:
                kwargs["flags"] = double_sided_flags
            vis = p.createVisualShape(
                p.GEOM_MESH, physicsClientId=cli, **kwargs
            )
            if vis >= 0:
                vis_ids = [vis]
        cli_cache[cache_key] = vis_ids
    return list(vis_ids)


def _spawn_asset_instance(
    cli: int, *, category: str, obj_name: str,
    x: float, y: float, yaw_deg: float,
    scale: float, flags: int, enable_collision: bool = True,
) -> bool:
    obj_path = os.path.join(FOREST_ASSET_DIR, category, obj_name)
    if not os.path.exists(obj_path):
        return False

    effective_scale = scale
    if FAST_BUILD_MODE and FAST_SCALE_STEP > 0.0:
        effective_scale = max(
            0.01, round(scale / FAST_SCALE_STEP) * FAST_SCALE_STEP
        )

    min_x, min_y, min_z, max_x, _, max_z = _obj_bounds_cached(obj_path)
    cx = (min_x + max_x) * 0.5
    cz = (min_z + max_z) * 0.5
    z0 = max(0.0, -min_y)
    yaw_rad = math.radians(yaw_deg)
    spawn_pos = [
        x - cx * effective_scale,
        y + cz * effective_scale,
        z0 * effective_scale,
    ]
    spawn_quat = p.getQuaternionFromEuler([1.5708, 0.0, yaw_rad])
    vis_shapes = _spawn_colored_obj(
        cli, obj_path=obj_path, scale=effective_scale,
        double_sided_flags=flags,
    )
    if not vis_shapes:
        return False

    col_shape = (
        _collision_shape_for_obj(cli, obj_path, effective_scale)
        if enable_collision
        else -1
    )
    extra_vis = vis_shapes[1:]
    if not extra_vis:
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shapes[0],
            basePosition=spawn_pos,
            baseOrientation=spawn_quat,
            physicsClientId=cli,
        )
        return True

    n_links = len(extra_vis)
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shapes[0],
        basePosition=spawn_pos,
        baseOrientation=spawn_quat,
        linkMasses=[0.0] * n_links,
        linkCollisionShapeIndices=[-1] * n_links,
        linkVisualShapeIndices=extra_vis,
        linkPositions=[[0.0, 0.0, 0.0]] * n_links,
        linkOrientations=[[0.0, 0.0, 0.0, 1.0]] * n_links,
        linkInertialFramePositions=[[0.0, 0.0, 0.0]] * n_links,
        linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * n_links,
        linkParentIndices=[0] * n_links,
        linkJointTypes=[p.JOINT_FIXED] * n_links,
        linkJointAxis=[[0.0, 0.0, 1.0]] * n_links,
        physicsClientId=cli,
    )
    return True


def _spawn_instances_as_single_multibody(
    cli: int, *,
    instances: List[Tuple[float, float, str, str, float, float]],
    rng: random.Random, flags: int, class_name: str,
    enable_collision: bool, fixed_yaw_deg: Optional[float] = None,
) -> int:
    if not instances:
        return 0

    link_masses: List[float] = []
    link_col: List[int] = []
    link_vis: List[int] = []
    link_pos: List[List[float]] = []
    link_orn: List[list] = []
    link_ifp: List[List[float]] = []
    link_ifo: List[List[float]] = []
    link_parent: List[int] = []
    link_jtype: List[int] = []
    link_jaxis: List[List[float]] = []
    placed_count = 0

    def _flush() -> None:
        nonlocal link_masses, link_col, link_vis, link_pos, link_orn
        nonlocal link_ifp, link_ifo, link_parent, link_jtype, link_jaxis
        if not link_vis:
            return
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_col,
            linkVisualShapeIndices=link_vis,
            linkPositions=link_pos,
            linkOrientations=link_orn,
            linkInertialFramePositions=link_ifp,
            linkInertialFrameOrientations=link_ifo,
            linkParentIndices=link_parent,
            linkJointTypes=link_jtype,
            linkJointAxis=link_jaxis,
            physicsClientId=cli,
        )
        link_masses = []
        link_col = []
        link_vis = []
        link_pos = []
        link_orn = []
        link_ifp = []
        link_ifo = []
        link_parent = []
        link_jtype = []
        link_jaxis = []

    for x, y, category, obj_name, scale, _ in instances:
        obj_path = os.path.join(FOREST_ASSET_DIR, category, obj_name)
        if not os.path.exists(obj_path):
            continue

        effective_scale = scale
        if FAST_BUILD_MODE and FAST_SCALE_STEP > 0.0:
            effective_scale = max(
                0.01, round(scale / FAST_SCALE_STEP) * FAST_SCALE_STEP
            )

        min_x, min_y, min_z, max_x, _, max_z = _obj_bounds_cached(obj_path)
        cx = (min_x + max_x) * 0.5
        cz = (min_z + max_z) * 0.5
        z0 = max(0.0, -min_y)

        yaw_deg = (
            fixed_yaw_deg if fixed_yaw_deg is not None else rng.uniform(-180.0, 180.0)
        )
        yaw_rad = math.radians(yaw_deg)
        spawn_pos = [
            x - cx * effective_scale,
            y + cz * effective_scale,
            z0 * effective_scale,
        ]
        spawn_quat = list(p.getQuaternionFromEuler([1.5708, 0.0, yaw_rad]))
        vis_shapes = _spawn_colored_obj(
            cli, obj_path=obj_path, scale=effective_scale, double_sided_flags=flags,
        )
        if not vis_shapes:
            continue

        if len(vis_shapes) > TREE_BATCH_MAX_LINKS:
            if _spawn_asset_instance(
                cli, category=category, obj_name=obj_name,
                x=x, y=y, yaw_deg=yaw_deg,
                scale=scale, flags=flags, enable_collision=enable_collision,
            ):
                placed_count += 1
            continue

        if link_vis and (len(link_vis) + len(vis_shapes) > TREE_BATCH_MAX_LINKS):
            _flush()

        col_shape = (
            _collision_shape_for_obj(cli, obj_path, effective_scale)
            if enable_collision
            else -1
        )
        for idx, vs in enumerate(vis_shapes):
            link_masses.append(0.0)
            link_col.append(col_shape if idx == 0 else -1)
            link_vis.append(vs)
            link_pos.append(spawn_pos)
            link_orn.append(spawn_quat)
            link_ifp.append([0.0, 0.0, 0.0])
            link_ifo.append([0.0, 0.0, 0.0, 1.0])
            link_parent.append(0)
            link_jtype.append(p.JOINT_FIXED)
            link_jaxis.append([0.0, 0.0, 1.0])

        placed_count += 1

    _flush()
    return placed_count


# ---------------------------------------------------------------------------
# SECTION 12: Main forest builder
# ---------------------------------------------------------------------------
def _spawn_forest_assets(
    cli: int, seed: int, mode_id: int = SCORING_MODE_ID,
    difficulty_id: int = SCORING_DIFFICULTY_ID,
) -> None:
    diff_cfg = DIFFICULTY_CONFIG[difficulty_id]
    flags = (
        p.VISUAL_SHAPE_DOUBLE_SIDED
        if hasattr(p, "VISUAL_SHAPE_DOUBLE_SIDED")
        else 0
    )
    rng = random.Random(seed)
    assets = _resolve_assets_for_class(mode_id)

    def _scaled_count(cls: str, base: int) -> int:
        mul = CLASS_DENSITY_MULTIPLIER.get(cls, 1.0)
        mul *= DIFFICULTY_DENSITY_MULTIPLIER.get(difficulty_id, 1.0)
        if cls == "trees":
            mul *= TREE_DIFFICULTY_MULTIPLIER.get(difficulty_id, 1.0)
        return max(0, int(round(base * DENSITY_MULTIPLIER * mul)))

    trees_count = _scaled_count("trees", diff_cfg["tree_count"])
    logs_count = _scaled_count("logs", diff_cfg["log_count"])
    bushes_count = _scaled_count("bushes", diff_cfg["bush_count"])

    rocks_assets = assets.get("rocks", [])
    stump_assets = assets.get("stumps", [])
    plants_assets = assets.get("plants", [])
    cactus_assets = assets.get("cactus", [])

    rock_stump_total = max(
        0,
        int(
            round(
                _scaled_count("rocks", diff_cfg["rock_stump_count"])
                * ROCK_STUMP_TOTAL_MULTIPLIER
            )
        ),
    )
    rocks_count, stumps_count = _split_asset_count(
        rock_stump_total, rocks_assets, stump_assets,
        primary_ratio=ROCK_STUMP_PRIMARY_RATIO,
    )
    plants_count, cactus_count = _split_asset_count(
        _scaled_count("plants", diff_cfg["ground_cover_count"]),
        plants_assets, cactus_assets,
        primary_ratio=GROUND_COVER_PLANT_PRIMARY_RATIO,
    )

    trunk_bounds = TRUNK_COUNT_BOUNDS_BY_DIFFICULTY.get(difficulty_id, {})
    logs_max = trunk_bounds.get("logs_max")
    if logs_max is not None:
        logs_count = min(logs_count, logs_max)
    stumps_min = trunk_bounds.get("stumps_min")
    if stumps_min is not None and stumps_count < stumps_min:
        shift = min(stumps_min - stumps_count, rocks_count)
        stumps_count += shift
        rocks_count -= shift
    stumps_max = trunk_bounds.get("stumps_max")
    if stumps_max is not None and stumps_count > stumps_max:
        shift = stumps_count - stumps_max
        stumps_count -= shift
        rocks_count += shift

    tree_instances = _pick_tree_instances(
        rng, count=trees_count, assets=list(assets.get("trees", [])),
        clearance_m=diff_cfg["tree_clearance_m"], difficulty_id=difficulty_id,
    )

    log_tree_occ = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["logs"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["logs"],
    )
    tree_base_rects = _tree_base_rects_from_instances(tree_instances)
    tree_span_rects = _tree_span_rects_from_instances(tree_instances)
    prot_span_rects = _protected_tree_span_rects_from_instances(tree_instances)

    log_instances = _pick_log_instances(
        rng, count=logs_count, assets=assets.get("logs", []),
        clearance_m=diff_cfg["log_clearance_m"] * LOG_CLEARANCE_MULTIPLIER,
        occupied_instances=log_tree_occ,
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=prot_span_rects,
    )
    bush_occ_scale = SMALL_ASSET_TREE_OCCUPANCY_SCALE["bushes"]
    bush_occ_cap = SMALL_ASSET_TREE_OCCUPANCY_CAP_M["bushes"]
    bush_instances = _pick_shrub_instances(
        rng, count=bushes_count, assets=assets.get("bushes", []),
        clearance_m=diff_cfg["bush_clearance_m"],
        tree_instances=tree_instances,
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=prot_span_rects,
        occupied_instances=log_instances,
        tree_occupancy_scale=bush_occ_scale,
        tree_occupancy_cap_m=bush_occ_cap,
    )
    rock_tree_occ = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["rocks"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["rocks"],
    )
    rock_instances = _pick_rock_stump_instances(
        rng, count=rocks_count, assets=rocks_assets,
        mode_id=mode_id, clearance_m=diff_cfg["rock_stump_clearance_m"],
        occupied_instances=rock_tree_occ + bush_instances + log_instances,
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=prot_span_rects,
    )
    stump_tree_occ = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["stumps"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["stumps"],
    )
    stump_instances = _pick_rock_stump_instances(
        rng, count=stumps_count, assets=stump_assets,
        mode_id=mode_id,
        clearance_m=diff_cfg["rock_stump_clearance_m"] * STUMP_CLEARANCE_MULTIPLIER,
        occupied_instances=(
            stump_tree_occ + bush_instances + log_instances + rock_instances
        ),
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=prot_span_rects,
    )
    plant_tree_occ = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["plants"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["plants"],
    )
    plant_instances = _pick_ground_cover_instances(
        rng, count=plants_count, assets=plants_assets,
        mode_id=mode_id, clearance_m=diff_cfg["ground_cover_clearance_m"],
        occupied_instances=(
            plant_tree_occ + bush_instances + log_instances + rock_instances
            + stump_instances
        ),
        tree_base_rects=tree_base_rects,
        tree_span_rects=tree_span_rects,
        protected_tree_span_rects=prot_span_rects,
    )
    cactus_tree_occ = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["cactus"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["cactus"],
    )
    cactus_instances = _pick_ground_cover_instances(
        rng, count=cactus_count, assets=cactus_assets,
        mode_id=mode_id, clearance_m=diff_cfg["ground_cover_clearance_m"],
        occupied_instances=(
            cactus_tree_occ + bush_instances + log_instances + rock_instances
            + stump_instances + plant_instances
        ),
        tree_base_rects=tree_base_rects,
        tree_span_rects=tree_span_rects,
        protected_tree_span_rects=prot_span_rects,
    )

    tree_yaw = 0.0
    _spawn_instances_as_single_multibody(
        cli, instances=tree_instances, rng=rng, flags=flags,
        class_name="trees", enable_collision=True, fixed_yaw_deg=tree_yaw,
    )
    for cls, inst_list in [
        ("bushes", bush_instances),
        ("rocks", rock_instances),
        ("stumps", stump_instances),
        ("logs", log_instances),
        ("plants", plant_instances),
        ("cactus", cactus_instances),
    ]:
        _spawn_instances_as_single_multibody(
            cli, instances=inst_list, rng=rng, flags=flags,
            class_name=cls, enable_collision=True,
        )


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


def get_forest_subtype(seed: int) -> Tuple[int, int]:
    from swarm.constants import FOREST_MODE_DISTRIBUTION, FOREST_DIFFICULTY_DISTRIBUTION
    mode_rng = random.Random(seed + 777777)
    modes = list(FOREST_MODE_DISTRIBUTION.keys())
    mode_weights = list(FOREST_MODE_DISTRIBUTION.values())
    mode_id = mode_rng.choices(modes, weights=mode_weights, k=1)[0]

    diff_rng = random.Random(seed + 888888)
    diffs = list(FOREST_DIFFICULTY_DISTRIBUTION.keys())
    diff_weights = list(FOREST_DIFFICULTY_DISTRIBUTION.values())
    difficulty_id = diff_rng.choices(diffs, weights=diff_weights, k=1)[0]

    return mode_id, difficulty_id


def build_forest(
    cli: int, seed: int, safe_zones: list, safe_zone_radius: float,
    hills_enabled: bool = False,
    forced_mode: Optional[int] = None,
    forced_difficulty: Optional[int] = None,
) -> None:
    """Build a Type 6 forest map on the given PyBullet client.

    Mode and difficulty are deterministic per seed via get_forest_subtype().
    Use forced_mode / forced_difficulty to override.
    """
    sub_mode, sub_diff = get_forest_subtype(seed)
    mode_id = _clamp_mode_id(forced_mode) if forced_mode is not None else sub_mode
    difficulty_id = max(1, min(3, int(forced_difficulty))) if forced_difficulty is not None else sub_diff

    _CLI_COL_CACHE[cli] = {}
    _CLI_VIS_CACHE[cli] = {}
    _CLI_TEX_CACHE.pop(cli, None)

    _spawn_ground(cli, mode_id)
    if hills_enabled:
        rgba = _ground_rgba_for_mode(mode_id)
        _spawn_hills(cli, rgba=rgba, apply_texture=(mode_id == 1))
    _spawn_forest_assets(cli, seed, mode_id=mode_id, difficulty_id=difficulty_id)
