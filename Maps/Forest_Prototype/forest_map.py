import argparse
import math
import os
import pickle
import random
import sys
import time

import pybullet as p
import pybullet_data


GROUND_SIZE_M = 100.0
GROUND_HALF_THICKNESS_M = 0.05
GROUND_RGBA = [0.72, 0.75, 0.72, 1.0]
GROUND_RGBA_AUTUMN = [0.58, 0.52, 0.38, 1.0]
GROUND_RGBA_DEAD = [0.46, 0.42, 0.35, 1.0]
GROUND_RGBA_SNOW = [0.98, 0.98, 1.0, 1.0]
GROUND_TEXTURE_RES = 384
GROUND_TEXTURE_SEED = 13731
GROUND_TEXTURE_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "custom",
    "textures",
    "forest_ground.bmp",
)
GROUND_TEXTURE_PATH_AUTUMN = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "custom",
    "textures",
    "forest_ground_autumn.bmp",
)
GROUND_TEXTURE_PATH_DEAD = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "custom",
    "textures",
    "forest_ground_dead_v2.bmp",
)
HILLS_MESH_CACHE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "custom",
    "terrain_cache",
)
HILLS_MESH_VERSION = 12
HILLS_WORLD_HALF_SIZE_M_NON_SNOW = 520.0
HILLS_WORLD_HALF_SIZE_M_SNOW = 900.0
HILLS_CENTER_FLAT_RADIUS_M = 78.0
HILLS_CENTER_BLEND_BAND_M = 130.0
# UV tiling: texture repeats every N world-meters (seamless tiling).
HILLS_TEXTURE_TILE_M = 6.0

ASSET_BASE_DIR = os.path.join(
    os.path.dirname(__file__),
    "..",
    "assets",
    "quaternius_ultimate_nature",
)
CUSTOM_ASSET_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "assets",
        "custom",
    )
)
MOUNTAIN_ASSET_DIR = os.path.join(CUSTOM_ASSET_DIR, "mountains")
PREVIEW_UNIFORM_SCALE = 2.0
MAP_NAME = "Forest"
MAP_MODE_CONFIG = {
    1: {"name": "Normal", "primary_category": "normal", "use_misc_willow": True},
    2: {"name": "Autumn", "primary_category": "autumn", "use_misc_willow": False},
    3: {"name": "Snow", "primary_category": "snow", "use_misc_willow": False},
    4: {"name": "No Leaves", "primary_category": "dead", "use_misc_willow": False},
}
HILLS_RING_DEFAULT = False
DIFFICULTY_CONFIG = {
    1: {
        "name": "Easy",
        "tree_count": 130,
        "tree_clearance_m": 0.14,
        "bush_count": 32,
        "bush_clearance_m": 0.58,
        "rock_stump_count": 20,
        "rock_stump_clearance_m": 0.64,
        "log_count": 10,
        "log_clearance_m": 0.42,
        "ground_cover_count": 20,
        "ground_cover_clearance_m": 0.20,
    },
    2: {
        "name": "Normal",
        "tree_count": 170,
        "tree_clearance_m": 0.10,
        "bush_count": 48,
        "bush_clearance_m": 0.48,
        "rock_stump_count": 34,
        "rock_stump_clearance_m": 0.54,
        "log_count": 16,
        "log_clearance_m": 0.36,
        "ground_cover_count": 34,
        "ground_cover_clearance_m": 0.16,
    },
    3: {
        "name": "Hard",
        "tree_count": 210,
        "tree_clearance_m": 0.06,
        "bush_count": 66,
        "bush_clearance_m": 0.38,
        "rock_stump_count": 50,
        "rock_stump_clearance_m": 0.46,
        "log_count": 24,
        "log_clearance_m": 0.30,
        "ground_cover_count": 52,
        "ground_cover_clearance_m": 0.12,
    },
}
FOREST_EDGE_MARGIN_M = 2.0
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
DENSITY_MULTIPLIER = 1.10
CLASS_ENABLE = {
    "trees": True,
    "bushes": True,
    "logs": True,
    "rocks": True,
    "stumps": True,
    "plants": True,
    "cactus": True,
}
CLASS_DENSITY_MULTIPLIER = {
    "trees": 1.184865,
    "bushes": 0.6567321796875,
    "logs": 1.12,
    "rocks": 0.91854,
    "stumps": 0.91854,
    "plants": 1.0206,
    "cactus": 1.0206,
}
TREE_DIFFICULTY_MULTIPLIER = {
    1: 1.3115,
    2: 1.3975,
    3: 1.10,
}
DIFFICULTY_DENSITY_MULTIPLIER = {
    1: 1.00,
    2: 1.00,
    3: 1.20,
}
DISABLE_MODEL_MATERIALS_FOR_DEBUG = False
DEBUG_MODEL_RGBA = [0.62, 0.62, 0.62, 1.0]
DISABLE_TREE_COLLISIONS_FOR_DEBUG = False
TREE_RECT_DEBUG_DRAW = False
GROUND_ONLY_MODE = False
SINGLE_TREE_TEST_MODE = False
MTL_COLOR_GAMMA = 0.66
MTL_COLOR_MULTIPLIER = 1.62
FAST_BUILD_MODE = True
FAST_SCALE_STEP = 0.10
TREE_BATCH_INSTANCE_MODE = True
TREE_BATCH_MAX_LINKS = 127
TREE_LAYOUT_MODE = "row"
USE_DECIMATED_TREE_MODELS = False
DECIMATED_TREE_SUBDIR = "decimated_75"
DECIMATE_CLASS_ALLOWLIST = {"trees", "rocks", "stumps"}
DECIMATE_EXCLUDE_TREE_PREFIXES = ("BirchTree_",)
ROCKS_ONLY_IN_ROCKS_SECTION = False
BIRCH_TREE_PREFIX = "BirchTree_"
COMMON_TREE_PREFIX = "CommonTree_"
PALM_TREE_PREFIX = "PalmTree_"
PINE_TREE_PREFIX = "PineTree_"
WILLOW_TREE_PREFIX = "Willow_"
TREE_FAMILY_PREFIXES = (
    BIRCH_TREE_PREFIX,
    COMMON_TREE_PREFIX,
    PALM_TREE_PREFIX,
    PINE_TREE_PREFIX,
    WILLOW_TREE_PREFIX,
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
ROCK_STUMP_MODEL_WEIGHT_BONUS = {
    "Rock_Moss_4.obj": 3.0,
    "Rock_Moss_6.obj": 3.0,
    "Rock_Moss_7.obj": 3.0,
    "TreeStump_Moss.obj": 3.0,
}
ROCK_STUMP_MODEL_SCALE_FACTOR = {}
GROUND_COVER_MODEL_WEIGHT_BONUS = {
    "Plant_1.obj": 0.45,
    "Plant_4.obj": 0.70,
    "Plant_5.obj": 4.0,
    "Cactus_1.obj": 2.4,
    "Cactus_2.obj": 2.4,
    "Cactus_3.obj": 2.4,
    "Cactus_4.obj": 2.4,
    "Cactus_5.obj": 2.4,
    "CactusFlower_1.obj": 2.4,
    "CactusFlowers_2.obj": 2.4,
    "CactusFlowers_3.obj": 2.4,
    "CactusFlowers_4.obj": 2.4,
    "CactusFlowers_5.obj": 2.4,
}
GROUND_COVER_MODEL_SCALE_FACTOR = {}
AUTUMN_GROUND_COVER_ALLOWLIST = {"Plant_2.obj", "Plant_5.obj", "Wheat.obj"}
DEAD_GROUND_COVER_ALLOWLIST = {"Plant_2.obj"}
NORMAL_ONLY_GROUND_COVER_ALLOWLIST = {
    "Corn_1.obj",
    "Corn_2.obj",
    "Lilypad.obj",
    "Cactus_1.obj",
    "Cactus_2.obj",
    "Cactus_3.obj",
    "Cactus_4.obj",
    "Cactus_5.obj",
    "CactusFlower_1.obj",
    "CactusFlowers_2.obj",
    "CactusFlowers_3.obj",
    "CactusFlowers_4.obj",
    "CactusFlowers_5.obj",
}
NORMAL_AUTUMN_DEAD_GROUND_COVER_ALLOWLIST = {"Wheat.obj"}
NORMAL_ONLY_SINGLE_SPAWN_GROUND_COVER = set()
ROCK_STUMP_TOTAL_MULTIPLIER = 1.02
ROCK_STUMP_PRIMARY_RATIO = 0.78
GROUND_COVER_PLANT_PRIMARY_RATIO = 0.75
LOG_CLEARANCE_MULTIPLIER = 0.90
STUMP_CLEARANCE_MULTIPLIER = 0.90
TRUNK_COUNT_BOUNDS_BY_DIFFICULTY = {
    1: {
        "logs_max": 7,
        "stumps_min": 6,
        "stumps_max": 7,
    },
    2: {
        "logs_max": 8,
        "stumps_min": 7,
        "stumps_max": 8,
    },
    3: {
        "logs_min": 5,
        "logs_max": 5,
        "stumps_min": 5,
        "stumps_max": 5,
    },
}
TRUNK_OCCUPANCY_MULTIPLIER_BY_DIFFICULTY = {
    3: {
        "logs": 0.65,
        "stumps": 0.82,
    },
}
TRUNK_CLEARANCE_MULTIPLIER_BY_DIFFICULTY = {
    3: {
        "logs": 0.72,
        "stumps": 0.82,
    },
}
TRUNK_FALLBACK_FILL_BY_DIFFICULTY = {
    3: {
        "logs": True,
        "stumps": True,
    },
}
GROUND_COVER_MODE_WEIGHT_BONUS = {
    1: {"Plant_2.obj": 0.25, "Wheat.obj": 0.25},
    2: {"Plant_2.obj": 3.0, "Plant_5.obj": 1.3},
    4: {"Plant_2.obj": 6.0},
}
ROCK_STUMP_MODE_WEIGHT_BONUS = {
    1: {"TreeStump.obj": 0.25},
}
MODE_CLASS_COUNT_MULTIPLIER = {
    4: {
        "logs": 3.0,
    },
}
MODE_ROCK_STUMP_TOTAL_MULTIPLIER = {
    4: 1.35,
}
MODE_ROCK_STUMP_PRIMARY_RATIO = {
    4: 0.68,
}
MODE_SMALL_ASSET_TREE_OCCUPANCY_MULTIPLIER = {
    4: {
        "logs": 0.45,
        "stumps": 0.68,
    },
}
MODE_SMALL_ASSET_CLEARANCE_MULTIPLIER = {
    4: {
        "logs": 0.72,
        "stumps": 0.82,
    },
}
TREE_SPACING_RADIUS_MULTIPLIER_DEFAULT = 0.32
TREE_SPACING_RADIUS_MIN_M = 0.30
TREE_TREE_CLEARANCE_FACTOR = 0.30
TREE_SPACING_RADIUS_MULTIPLIER_DEFAULT_BY_DIFFICULTY = {
    1: 0.54,
    2: 0.48,
    3: 0.32,
}
TREE_TREE_CLEARANCE_FACTOR_BY_DIFFICULTY = {
    1: 1.00,
    2: 0.90,
    3: 0.30,
}
TREE_CANOPY_OVERLAP_SCALE_BY_DIFFICULTY = {
    1: 0.90,
    2: 0.79,
    3: 0.64,
}
TREE_SPACING_RADIUS_BY_PREFIX_BY_DIFFICULTY = {
    1: {
        "PalmTree_": 0.30,
        "Willow_": 0.42,
        "BirchTree_": 0.44,
        "PineTree_": 1.00,
        "CommonTree_": 0.52,
    },
    2: {
        "PalmTree_": 0.26,
        "Willow_": 0.36,
        "BirchTree_": 0.40,
        "PineTree_": 1.00,
        "CommonTree_": 0.46,
    },
    3: {
        "PalmTree_": 0.12,
        "Willow_": 0.20,
        "BirchTree_": 0.24,
        "PineTree_": 1.00,
        "CommonTree_": 0.34,
    },
}
LOW_CANOPY_PROTECTED_TREE_NAMES = {
    "PineTree_1.obj",
    "PineTree_2.obj",
    "PineTree_3.obj",
    "PineTree_Autumn_1.obj",
    "PineTree_Autumn_2.obj",
    "PineTree_Autumn_3.obj",
    "PineTree_Snow_1.obj",
    "PineTree_Snow_2.obj",
    "PineTree_Snow_3.obj",
}
TREE_OCCUPANCY_RADIUS_MULTIPLIER_DEFAULT = 0.70
TREE_OCCUPANCY_RADIUS_MIN_M = 0.45
TREE_OCCUPANCY_RADIUS_BY_PREFIX = {
    "PalmTree_": 0.95,
    "Willow_": 0.84,
    "BirchTree_": 0.76,
    "PineTree_": 1.00,
    "CommonTree_": 0.82,
}
TREE_LOCAL_CLUSTER_MAX_BY_DIFFICULTY = {
    1: 1,
    2: 1,
    3: 2,
}
TREE_LOCAL_CLUSTER_SEARCH_BY_DIFFICULTY_M = {
    1: 3.6,
    2: 3.2,
    3: 2.8,
}
BUSH_DISTRIBUTION_CELL_SIZE_M = 8.0
BUSH_DISTRIBUTION_MAX_PER_CELL = 1
SMALL_ASSET_EDGE_MARGIN_M = 6.0
SMALL_ASSET_CENTER_REGION_RATIO = 0.68
SMALL_ASSET_CENTER_BIAS = 0.70
SMALL_ASSET_TREE_OCCUPANCY_SCALE = {
    "logs": 0.60,
    "bushes": 0.58,
    "rocks": 0.55,
    "stumps": 0.53,
    "plants": 0.48,
    "cactus": 0.48,
}
SMALL_ASSET_TREE_OCCUPANCY_CAP_M = {
    "logs": 2.40,
    "bushes": 1.90,
    "rocks": 1.70,
    "stumps": 1.70,
    "plants": 1.30,
    "cactus": 1.30,
}
SMALL_ASSET_TREE_OCCUPANCY_MIN_M = 0.40

LAST_MAP_SPAWNED = []
_OBJ_BOUNDS_CACHE = {}
_OBJ_PLANAR_RADIUS_CACHE = {}
_OBJ_COLLISION_SHAPE_CACHE = {}
_OBJ_VISUAL_SHAPE_CACHE = {}
_OBJ_MATERIAL_MESH_CACHE = {}
_OBJ_FLAT_MESH_CACHE = {}
_CLASS_ASSET_CACHE = {}
_TREE_RECT_TEMPLATE_CACHE = {}
_HILL_OBJ_TRI_CACHE = {}
_GROUND_TEXTURE_ID = None
_GROUND_TEXTURE_ID_AUTUMN = None
_GROUND_TEXTURE_ID_DEAD = None
_HILLS_TEXTURE_ID = None
_HILLS_TEXTURE_ID_AUTUMN = None
_HILLS_TEXTURE_ID_DEAD = None
_LAST_BUILT_SCENE = None
_LAST_BUILT_RESULT = None


class _SpatialGrid:
    """Fast spatial lookup grid for placement collision checks. Avoids O(n^2)."""
    __slots__ = ('cells', 'cell_size', 'max_radius')

    def __init__(self, cell_size=4.0):
        self.cells = {}
        self.cell_size = cell_size
        self.max_radius = 0.0

    def _key(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, x, y, radius):
        if radius > self.max_radius:
            self.max_radius = radius
        self.cells.setdefault(self._key(x, y), []).append((x, y, radius))

    def has_conflict(self, x, y, radius, clearance):
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

    def count_neighbors(self, x, y, search_radius):
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


class CameraController:
    def __init__(
        self,
        *,
        x: float = 0.0,
        y: float = -12.0,
        z: float = 5.0,
        yaw: float = 0.0,
        pitch: float = -18.0,
        dist: float = 0.1,
        speed: float = 0.31875,
        mouse_sensitivity: float = 0.5,
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

        for event in mouse:
            if event[0] == 1:
                if self.lmb_held:
                    dx = event[1] - self.last_mouse_x
                    dy = event[2] - self.last_mouse_y
                self.last_mouse_x = event[1]
                self.last_mouse_y = event[2]
            if event[0] == 2 and event[3] == 0:
                self.lmb_held = event[4] == 3 or event[4] == 1
                if self.lmb_held:
                    self.last_mouse_x = event[1]
                    self.last_mouse_y = event[2]

        if self.lmb_held and (dx or dy):
            self.yaw -= dx * self.mouse_sensitivity
            self.pitch -= dy * self.mouse_sensitivity
            self.pitch = max(-89.0, min(89.0, self.pitch))

        move_speed = self.speed * (3.0 if keys.get(p.B3G_SHIFT, 0) else 1.0)
        rad_yaw = math.radians(self.yaw)
        f_x = -math.sin(rad_yaw)
        f_y = math.cos(rad_yaw)
        r_x = math.cos(rad_yaw)
        r_y = math.sin(rad_yaw)

        fwd = (1 if keys.get(ord("w"), 0) else 0) - (1 if keys.get(ord("s"), 0) else 0)
        right = (1 if keys.get(ord("d"), 0) else 0) - (1 if keys.get(ord("a"), 0) else 0)
        up = (1 if keys.get(ord("e"), 0) else 0) - (1 if keys.get(ord("q"), 0) else 0)

        self.x += (f_x * fwd + r_x * right) * move_speed
        self.y += (f_y * fwd + r_y * right) * move_speed
        self.z += up * move_speed
        self.update_camera()

    def update_camera(self):
        p.resetDebugVisualizerCamera(self.dist, self.yaw, self.pitch, [self.x, self.y, self.z])


def _clamp_u8(v: float) -> int:
    return max(0, min(255, int(round(v))))


def _hash_noise_01(x: int, y: int, seed: int) -> float:
    n = (x * 73856093) ^ (y * 19349663) ^ (seed * 83492791)
    n = (n << 13) ^ n
    return 1.0 - ((n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF) / 1073741824.0


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
                r = rgb_data[idx + 0]
                g = rgb_data[idx + 1]
                b = rgb_data[idx + 2]
                f.write(bytes((b, g, r)))
            if row_pad:
                f.write(pad)


def _ensure_ground_texture(*, variant: str = "normal") -> str:
    if variant == "autumn":
        texture_path = GROUND_TEXTURE_PATH_AUTUMN
    elif variant == "dead":
        texture_path = GROUND_TEXTURE_PATH_DEAD
    else:
        texture_path = GROUND_TEXTURE_PATH
    if os.path.exists(texture_path):
        return texture_path

    os.makedirs(os.path.dirname(texture_path), exist_ok=True)
    w = h = GROUND_TEXTURE_RES
    data = bytearray(w * h * 3)
    if variant == "autumn":
        seed = GROUND_TEXTURE_SEED + 20011
    elif variant == "dead":
        seed = GROUND_TEXTURE_SEED + 40027
    else:
        seed = GROUND_TEXTURE_SEED
    rng = random.Random(seed)

    if variant == "autumn":
        grass_a = (112.0, 124.0, 82.0)
        grass_b = (134.0, 122.0, 74.0)
        patch_count = 88
        dirt_r = (118.0, 145.0)
        dirt_g = (95.0, 118.0)
        dirt_b = (62.0, 84.0)
        blend_r = (0.40, 0.72)
    elif variant == "dead":
        grass_a = (88.0, 92.0, 76.0)
        grass_b = (106.0, 100.0, 80.0)
        patch_count = 94
        dirt_r = (98.0, 122.0)
        dirt_g = (78.0, 96.0)
        dirt_b = (58.0, 76.0)
        blend_r = (0.46, 0.80)
    else:
        grass_a = (96.0, 130.0, 88.0)
        grass_b = (109.0, 146.0, 95.0)
        patch_count = 56
        dirt_r = (103.0, 126.0)
        dirt_g = (93.0, 111.0)
        dirt_b = (74.0, 93.0)
        blend_r = (0.28, 0.58)

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

    # Paint soft dirt/mud patches to avoid flat color and add route variety.
    for _ in range(patch_count):
        cx = rng.uniform(0.0, w - 1.0)
        cy = rng.uniform(0.0, h - 1.0)
        radius = rng.uniform(16.0, 52.0)
        dirt = (
            rng.uniform(*dirt_r),
            rng.uniform(*dirt_g),
            rng.uniform(*dirt_b),
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
                blend = (edge * edge) * rng.uniform(*blend_r)
                idx = (py * w + px) * 3
                data[idx + 0] = _clamp_u8(data[idx + 0] * (1.0 - blend) + dirt[0] * blend)
                data[idx + 1] = _clamp_u8(data[idx + 1] * (1.0 - blend) + dirt[1] * blend)
                data[idx + 2] = _clamp_u8(data[idx + 2] * (1.0 - blend) + dirt[2] * blend)

    _write_bmp24(texture_path, w, h, data)
    return texture_path


def _ground_texture_id(*, mode_id: int) -> int | None:
    global _GROUND_TEXTURE_ID, _GROUND_TEXTURE_ID_AUTUMN, _GROUND_TEXTURE_ID_DEAD
    mode_cfg = MAP_MODE_CONFIG[_clamp_mode_id(mode_id)]
    primary_cat = mode_cfg["primary_category"]
    if primary_cat == "snow":
        return None
    if primary_cat == "autumn":
        if _GROUND_TEXTURE_ID_AUTUMN is not None:
            return _GROUND_TEXTURE_ID_AUTUMN
        variant = "autumn"
    elif primary_cat == "dead":
        if _GROUND_TEXTURE_ID_DEAD is not None:
            return _GROUND_TEXTURE_ID_DEAD
        variant = "dead"
    else:
        if _GROUND_TEXTURE_ID is not None:
            return _GROUND_TEXTURE_ID
        variant = "normal"

    tex_path = _ensure_ground_texture(variant=variant)
    try:
        tex_id = p.loadTexture(tex_path)
    except Exception:
        tex_id = None
    if variant == "autumn":
        _GROUND_TEXTURE_ID_AUTUMN = tex_id
        return _GROUND_TEXTURE_ID_AUTUMN
    if variant == "dead":
        _GROUND_TEXTURE_ID_DEAD = tex_id
        return _GROUND_TEXTURE_ID_DEAD
    _GROUND_TEXTURE_ID = tex_id
    return _GROUND_TEXTURE_ID


def _hills_texture_id(*, mode_id: int) -> int | None:
    return _ground_texture_id(mode_id=mode_id)


def _ground_rgba_for_mode(*, mode_id: int) -> list[float]:
    mode_cfg = MAP_MODE_CONFIG[_clamp_mode_id(mode_id)]
    primary_cat = mode_cfg["primary_category"]
    if primary_cat == "autumn":
        return GROUND_RGBA_AUTUMN
    if primary_cat == "dead":
        return GROUND_RGBA_DEAD
    if primary_cat == "snow":
        return GROUND_RGBA_SNOW
    return GROUND_RGBA


def _smoothstep01(x: float) -> float:
    t = max(0.0, min(1.0, x))
    return t * t * (3.0 - 2.0 * t)


def _spawn_ground(*, mode_id: int) -> None:
    mode_cfg = MAP_MODE_CONFIG[_clamp_mode_id(mode_id)]
    is_snow_mode = mode_cfg["primary_category"] == "snow"
    if is_snow_mode:
        return
    ground_rgba = _ground_rgba_for_mode(mode_id=mode_id)

    half = GROUND_SIZE_M * 0.5
    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half, half, GROUND_HALF_THICKNESS_M],
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half, half, GROUND_HALF_THICKNESS_M],
        rgbaColor=ground_rgba,
        specularColor=[0.0, 0.0, 0.0],
    )
    body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 0.0, -GROUND_HALF_THICKNESS_M],
    )
    tex_id = _ground_texture_id(mode_id=mode_id)
    if tex_id is not None:
        p.changeVisualShape(body, -1, textureUniqueId=tex_id)


def _spawn_snow_base_ground() -> None:
    half = GROUND_SIZE_M * 0.5
    col = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[half, half, GROUND_HALF_THICKNESS_M],
    )
    vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[half, half, GROUND_HALF_THICKNESS_M],
        rgbaColor=GROUND_RGBA_SNOW,
        specularColor=[0.0, 0.0, 0.0],
    )
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[0.0, 0.0, -GROUND_HALF_THICKNESS_M],
    )


def _load_obj_triangles_cached(obj_path: str) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    cached = _HILL_OBJ_TRI_CACHE.get(obj_path)
    if cached is not None:
        return cached

    verts: list[tuple[float, float, float]] = []
    tris: list[tuple[int, int, int]] = []
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
            idxs: list[int] = []
            for ptoken in parts:
                vtok = ptoken.split("/")[0]
                if not vtok:
                    continue
                vi = int(vtok)
                if vi < 0:
                    vi = len(verts) + vi + 1
                idxs.append(vi - 1)
            if len(idxs) < 3:
                continue
            # Fan triangulation for quads/ngons.
            for i in range(1, len(idxs) - 1):
                tris.append((idxs[0], idxs[i], idxs[i + 1]))

    cached = (verts, tris)
    _HILL_OBJ_TRI_CACHE[obj_path] = cached
    return cached


def _merged_hills_obj_path(*, mode_id: int) -> str:
    return os.path.join(
        HILLS_MESH_CACHE_DIR,
        f"merged_hills_mode{_clamp_mode_id(mode_id)}_v{HILLS_MESH_VERSION}.obj",
    )


def _ensure_merged_hills_obj(*, mode_id: int) -> str | None:
    mode_id = _clamp_mode_id(mode_id)
    primary_cat = MAP_MODE_CONFIG[mode_id]["primary_category"]
    if not os.path.isdir(MOUNTAIN_ASSET_DIR):
        return None
    hill_candidates = sorted(
        [
            f
            for f in os.listdir(MOUNTAIN_ASSET_DIR)
            if f.lower().endswith(".obj") and "mountain_peak" not in f.lower()
        ]
    )
    if not hill_candidates:
        return None

    out_path = _merged_hills_obj_path(mode_id=mode_id)
    if os.path.exists(out_path):
        return out_path

    os.makedirs(HILLS_MESH_CACHE_DIR, exist_ok=True)

    # Keep layout stable per mode so we don't generate infinite cache files.
    rng = random.Random(99173 + mode_id * 97)

    instances: list[tuple[str, float, float, float, float]] = []

    radius_inner = 165.0
    for i in range(6):
        angle = (2.0 * math.pi / 6.0) * i
        r = radius_inner + rng.uniform(-5.0, 5.0)
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

    transformed_instances: list[tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]] = []
    min_x_all = float("inf")
    min_y_all = float("inf")
    max_x_all = float("-inf")
    max_y_all = float("-inf")

    for hill_file, s, tx, ty, yaw_deg in instances:
        hill_path = os.path.join(MOUNTAIN_ASSET_DIR, hill_file)
        src_v, src_tris = _load_obj_triangles_cached(hill_path)
        quat = p.getQuaternionFromEuler([1.5708, 0.0, math.radians(yaw_deg)])
        m = p.getMatrixFromQuaternion(quat)
        rot = (
            (m[0], m[1], m[2]),
            (m[3], m[4], m[5]),
            (m[6], m[7], m[8]),
        )

        transformed: list[tuple[float, float, float]] = []
        min_z = float("inf")
        for vx, vy, vz in src_v:
            sx = vx * s
            sy = vy * s
            sz = vz * s
            rx = rot[0][0] * sx + rot[0][1] * sy + rot[0][2] * sz
            ry = rot[1][0] * sx + rot[1][1] * sy + rot[1][2] * sz
            rz = rot[2][0] * sx + rot[2][1] * sy + rot[2][2] * sz
            wx = rx + tx
            wy = ry + ty
            wz = rz
            transformed.append((wx, wy, wz))
            if wz < min_z:
                min_z = wz

        z_corr = (0.0 - min_z) - 0.02
        corrected: list[tuple[float, float, float]] = []
        for wx, wy, wz in transformed:
            wz2 = wz + z_corr
            corrected.append((wx, wy, wz2))
            if wx < min_x_all:
                min_x_all = wx
            if wy < min_y_all:
                min_y_all = wy
            if wx > max_x_all:
                max_x_all = wx
            if wy > max_y_all:
                max_y_all = wy

        transformed_instances.append((corrected, src_tris))

    if primary_cat == "snow":
        far_half = HILLS_WORLD_HALF_SIZE_M_SNOW
    else:
        # Tight envelope for non-snow: keep almost no wasted terrain behind the furthest hills.
        if not math.isfinite(min_x_all):
            far_half = HILLS_WORLD_HALF_SIZE_M_NON_SNOW
        else:
            extent_x = max(abs(min_x_all), abs(max_x_all))
            extent_y = max(abs(min_y_all), abs(max_y_all))
            far_half = max((GROUND_SIZE_M * 0.5) + 4.0, max(extent_x, extent_y) + 4.0)

    merged_v: list[tuple[float, float, float]] = []
    merged_vt: list[tuple[float, float]] = []
    merged_f: list[tuple[int, int, int]] = []

    def add_vertex(wx: float, wy: float, wz: float) -> int:
        merged_v.append((wx, wy, wz))
        # Fallback UV behavior: normalize across merged terrain bounds.
        u = (wx + far_half) / (2.0 * far_half)
        v = (wy + far_half) / (2.0 * far_half)
        merged_vt.append((u, v))
        return len(merged_v)

    # One base sheet in same mesh (visual join).
    i1 = add_vertex(-far_half, -far_half, -0.1)
    i2 = add_vertex( far_half, -far_half, -0.1)
    i3 = add_vertex( far_half,  far_half, -0.1)
    i4 = add_vertex(-far_half,  far_half, -0.1)
    merged_f.append((i1, i2, i3))
    merged_f.append((i1, i3, i4))

    for corrected_vertices, src_tris in transformed_instances:
        base_idx = len(merged_v)
        for wx, wy, wz in corrected_vertices:
            add_vertex(wx, wy, wz)
        for a, b, c in src_tris:
            merged_f.append((base_idx + a + 1, base_idx + b + 1, base_idx + c + 1))

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Auto-generated merged hills + far ground (single mesh)\n")
        f.write("o MergedHillsTerrain\n")
        for x, y, z in merged_v:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for u, v in merged_vt:
            f.write(f"vt {u:.6f} {v:.6f}\n")
        for a, b, c in merged_f:
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    return out_path


def _spawn_mode_hills(*, mode_id: int, seed: int, include_peaks: bool = False) -> None:
    mode_id = _clamp_mode_id(mode_id)
    ring_rgba = _ground_rgba_for_mode(mode_id=mode_id)
    ring_tex_id = _hills_texture_id(mode_id=mode_id)
    merged_obj = _ensure_merged_hills_obj(mode_id=mode_id)
    if merged_obj is None:
        return

    terrain_vis = p.createVisualShape(
        p.GEOM_MESH,
        fileName=merged_obj,
        meshScale=[1.0, 1.0, 1.0],
        rgbaColor=ring_rgba,
        specularColor=[0.0, 0.0, 0.0],
    )
    terrain_col = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=merged_obj,
        meshScale=[1.0, 1.0, 1.0],
    )
    terrain_body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=terrain_col,
        baseVisualShapeIndex=terrain_vis,
        basePosition=[0.0, 0.0, 0.0],
    )
    if ring_tex_id is not None:
        p.changeVisualShape(terrain_body, -1, textureUniqueId=ring_tex_id)

    if not include_peaks or not os.path.isdir(MOUNTAIN_ASSET_DIR):
        return

    peak_obj = os.path.join(MOUNTAIN_ASSET_DIR, "mountain_peak.obj")
    peak_tex = os.path.join(MOUNTAIN_ASSET_DIR, "mountain_peak.png")
    rng = random.Random(seed + 99173)
    shape_cache = {}
    peak_tex_id = p.loadTexture(peak_tex) if os.path.exists(peak_tex) else -1

    # OUTER RING (10 peaks) - snow only when requested.
    if include_peaks and os.path.exists(peak_obj):
        for i in range(10):
            angle = (2.0 * math.pi / 10.0) * i
            r = 550.0 + rng.uniform(-20.0, 20.0)
            x, y = r * math.cos(angle), r * math.sin(angle)
            scale_vals = [126.4, 126.4, 158.4]
            s_var = round(rng.uniform(0.9, 1.2), 1)
            final_scale = [round(v * s_var, 2) for v in scale_vals]
            cache_key = (peak_obj, tuple(final_scale), tuple(ring_rgba))
            if cache_key in shape_cache:
                vis_id, col_id = shape_cache[cache_key]
            else:
                vis_id = p.createVisualShape(
                    p.GEOM_MESH,
                    fileName=peak_obj,
                    meshScale=final_scale,
                    rgbaColor=ring_rgba,
                    specularColor=[0.0, 0.0, 0.0],
                )
                col_id = p.createCollisionShape(
                    p.GEOM_MESH,
                    fileName=peak_obj,
                    meshScale=final_scale,
                )
                shape_cache[cache_key] = (vis_id, col_id)
            orn = p.getQuaternionFromEuler([1.5708, 0.0, math.radians(rng.uniform(0.0, 360.0))])
            peak_body = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=col_id,
                baseVisualShapeIndex=vis_id,
                basePosition=[x, y, 0.0],
                baseOrientation=orn,
            )
            min_aabb, _ = p.getAABB(peak_body)
            min_z = min_aabb[2]
            z_correction = (0.0 - min_z) - 10.0
            p.resetBasePositionAndOrientation(peak_body, [x, y, z_correction], orn)
            if peak_tex_id >= 0:
                p.changeVisualShape(peak_body, -1, textureUniqueId=peak_tex_id)


def _obj_bounds(path: str):
    min_x = min_y = min_z = float("inf")
    max_x = max_y = max_z = float("-inf")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
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
        raise RuntimeError(f"No vertices found in OBJ: {path}")
    return min_x, min_y, min_z, max_x, max_y, max_z


def _obj_path_for_spawn(category: str, obj_name: str, *, asset_class: str | None = None) -> str:
    default_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
    if not USE_DECIMATED_TREE_MODELS:
        return default_path
    if asset_class not in DECIMATE_CLASS_ALLOWLIST:
        return default_path
    if asset_class == "trees" and obj_name.startswith(DECIMATE_EXCLUDE_TREE_PREFIXES):
        return default_path

    decimated_path = os.path.join(ASSET_BASE_DIR, category, DECIMATED_TREE_SUBDIR, obj_name)
    return decimated_path if os.path.exists(decimated_path) else default_path


def _obj_bounds_cached(path: str):
    cached = _OBJ_BOUNDS_CACHE.get(path)
    if cached is None:
        cached = _obj_bounds(path)
        _OBJ_BOUNDS_CACHE[path] = cached
    return cached


def _clamp_difficulty_id(difficulty_id: int) -> int:
    return max(1, min(3, int(difficulty_id)))


def _clamp_mode_id(mode_id: int) -> int:
    return max(1, min(4, int(mode_id)))


def _list_obj_names(category: str) -> list[str]:
    cat_dir = os.path.join(ASSET_BASE_DIR, category)
    if not os.path.isdir(cat_dir):
        return []
    return sorted([name for name in os.listdir(cat_dir) if name.lower().endswith(".obj")])


def _resolve_assets_for_class(mode_id: int) -> dict[str, list[tuple[str, str]]]:
    mode_id = _clamp_mode_id(mode_id)
    cache_key = f"mode_{mode_id}"
    cached = _CLASS_ASSET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mode_cfg = MAP_MODE_CONFIG[mode_id]
    primary_cat = mode_cfg["primary_category"]
    is_dry_profile = primary_cat in ("autumn", "dead", "dead_snow")
    primary_objs = _list_obj_names(primary_cat)
    dead_snow_objs = _list_obj_names("dead_snow")
    normal_objs = _list_obj_names("normal")
    misc_objs = _list_obj_names("misc")
    is_snow_profile = primary_cat in ("snow", "dead_snow")
    if is_snow_profile:
        primary_mode_objs = [name for name in primary_objs if "Snow" in name]
    else:
        primary_mode_objs = [name for name in primary_objs if "Snow" not in name]
    normal_nosnow = [name for name in normal_objs if "Snow" not in name]

    trees = [
        (primary_cat, name)
        for name in primary_mode_objs
        if (("Tree" in name or name.startswith("Willow_")) and "Stump" not in name and "Bush" not in name)
    ]
    if primary_cat == "snow":
        dead_snow_trees = [
            ("dead_snow", name)
            for name in dead_snow_objs
            if ("Tree" in name and "Stump" not in name and "Bush" not in name)
        ]
        trees += dead_snow_trees
    if mode_cfg["use_misc_willow"]:
        trees += [("misc", name) for name in misc_objs if "Willow_" in name]

    bushes = [(primary_cat, name) for name in primary_mode_objs if "Bush" in name]
    rocks = [(primary_cat, name) for name in primary_mode_objs if name.startswith("Rock")]
    stumps = [(primary_cat, name) for name in primary_mode_objs if "TreeStump" in name]
    logs = [(primary_cat, name) for name in primary_mode_objs if "WoodLog" in name]
    if primary_cat == "snow":
        snow_stumps = [("normal", name) for name in normal_objs if name == "TreeStump_Snow.obj"]
        stumps += snow_stumps
    if is_snow_profile:
        plants = []
        cactus = []
    else:
        plants = [
            ("misc", name)
            for name in misc_objs
            if (
                name.startswith("Plant_")
                or name in {"Flowers.obj"}
            )
        ]
        if mode_id == 1:
            plants += [("misc", name) for name in misc_objs if name in NORMAL_ONLY_GROUND_COVER_ALLOWLIST]
        if mode_id in (1, 2, 4):
            plants += [
                ("misc", name)
                for name in misc_objs
                if name in NORMAL_AUTUMN_DEAD_GROUND_COVER_ALLOWLIST
            ]
        cactus = [
            ("misc", name)
            for name in misc_objs
            if (
                name.startswith("Cactus_")
                or name.startswith("CactusFlower_")
                or name.startswith("CactusFlowers_")
            )
        ]
    if primary_cat == "autumn":
        bushes = [(cat, name) for cat, name in bushes if not name.startswith(BUSH_BERRIES_PREFIX)]
        plants = [(cat, name) for cat, name in plants if name != "Flowers.obj"]
    if is_dry_profile:
        # Autumn / no-leaves: remove green clutter and moss-covered props.
        bushes = []
        if primary_cat == "autumn":
            plants = [(cat, name) for cat, name in plants if name in AUTUMN_GROUND_COVER_ALLOWLIST]
        elif primary_cat == "dead":
            plants = [("misc", name) for name in misc_objs if name in DEAD_GROUND_COVER_ALLOWLIST]
        else:
            plants = []
        logs = [(cat, name) for cat, name in logs if "Moss" not in name]
        rocks = [(cat, name) for cat, name in rocks if "Moss" not in name]
        stumps = [(cat, name) for cat, name in stumps if "Moss" not in name]
        cactus = []
    if primary_cat not in ("snow", "dead_snow"):
        logs = [(cat, name) for cat, name in logs if "Snow" not in name]

    # Fallback guarantees for sparse sets / modes without full class coverage.
    if not trees:
        trees = [("normal", "CommonTree_1.obj")]
    if not bushes and not is_dry_profile:
        bushes = [("normal", name) for name in normal_nosnow if "Bush" in name]
        if primary_cat == "autumn":
            bushes = [(cat, name) for cat, name in bushes if not name.startswith(BUSH_BERRIES_PREFIX)]
        if not bushes:
            bushes = [("normal", "Bush_1.obj")]
    if not rocks:
        fallback_rocks = [("normal", name) for name in normal_nosnow if name.startswith("Rock")]
        if is_dry_profile:
            fallback_rocks = [(cat, name) for cat, name in fallback_rocks if "Moss" not in name]
        rocks = fallback_rocks
        if not rocks:
            rocks = [("normal", "Rock_1.obj")]
    if not stumps:
        fallback_stumps = [("normal", name) for name in normal_nosnow if "TreeStump" in name]
        if is_dry_profile:
            fallback_stumps = [(cat, name) for cat, name in fallback_stumps if "Moss" not in name]
        stumps = fallback_stumps
        if not stumps:
            stumps = [("normal", "TreeStump.obj")]
    if not logs:
        normal_logs = [("normal", name) for name in normal_objs if "WoodLog" in name]
        if primary_cat in ("snow", "dead_snow"):
            snow_logs = [(cat, name) for cat, name in normal_logs if "Snow" in name]
            logs = snow_logs if snow_logs else normal_logs
        else:
            no_snow_logs = [(cat, name) for cat, name in normal_logs if "Snow" not in name]
            logs = no_snow_logs if no_snow_logs else normal_logs
        if is_dry_profile:
            logs = [(cat, name) for cat, name in logs if "Moss" not in name and "Snow" not in name]
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


def _map_half_extent() -> float:
    return GROUND_SIZE_M * 0.5


def _is_gui_connected() -> bool:
    try:
        info = p.getConnectionInfo()
        return int(info.get("connectionMethod", -1)) == p.GUI
    except Exception:
        return False


def _draw_rect_xy(
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    *,
    color: list[float],
    width: float = 1.1,
    z: float = 0.04,
) -> None:
    p1 = [min_x, min_y, z]
    p2 = [max_x, min_y, z]
    p3 = [max_x, max_y, z]
    p4 = [min_x, max_y, z]
    p.addUserDebugLine(p1, p2, color, lineWidth=width, lifeTime=0)
    p.addUserDebugLine(p2, p3, color, lineWidth=width, lifeTime=0)
    p.addUserDebugLine(p3, p4, color, lineWidth=width, lifeTime=0)
    p.addUserDebugLine(p4, p1, color, lineWidth=width, lifeTime=0)


def _rect_from_points_xy(points_xy: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    min_x = min(px for px, _ in points_xy)
    max_x = max(px for px, _ in points_xy)
    min_y = min(py for _, py in points_xy)
    max_y = max(py for _, py in points_xy)
    return min_x, max_x, min_y, max_y


def _expand_rect_to_min_size(
    rect: tuple[float, float, float, float],
    *,
    min_w: float,
    min_h: float,
) -> tuple[float, float, float, float]:
    min_x, max_x, min_y, max_y = rect
    w = max_x - min_x
    h = max_y - min_y
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half_w = 0.5 * max(w, min_w)
    half_h = 0.5 * max(h, min_h)
    return cx - half_w, cx + half_w, cy - half_h, cy + half_h


def _scale_rect(rect: tuple[float, float, float, float], scale: float) -> tuple[float, float, float, float]:
    return rect[0] * scale, rect[1] * scale, rect[2] * scale, rect[3] * scale


def _shift_rect(rect: tuple[float, float, float, float], dx: float, dy: float) -> tuple[float, float, float, float]:
    return rect[0] + dx, rect[1] + dx, rect[2] + dy, rect[3] + dy


def _circle_bounds_rect(x: float, y: float, radius: float) -> tuple[float, float, float, float]:
    return x - radius, x + radius, y - radius, y + radius


def _shrink_rect_from_center(
    rect: tuple[float, float, float, float],
    factor: float,
) -> tuple[float, float, float, float]:
    if factor >= 0.999:
        return rect
    factor = max(0.05, float(factor))
    min_x, max_x, min_y, max_y = rect
    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half_w = 0.5 * (max_x - min_x) * factor
    half_h = 0.5 * (max_y - min_y) * factor
    return cx - half_w, cx + half_w, cy - half_h, cy + half_h


def _rect_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    if a[1] <= b[0] or a[0] >= b[1]:
        return False
    if a[3] <= b[2] or a[2] >= b[3]:
        return False
    return True


def _tree_rect_template_unit(obj_path: str) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None:
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
    r00, r01, r02, r10, r11, r12, r20, r21, r22 = rot
    px, py, pz = (-cx, cz, z0)

    world_verts: list[tuple[float, float, float]] = []
    for vx, vy, vz in verts:
        wx = px + (r00 * vx) + (r01 * vy) + (r02 * vz)
        wy = py + (r10 * vx) + (r11 * vy) + (r12 * vz)
        wz = pz + (r20 * vx) + (r21 * vy) + (r22 * vz)
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
        contact_points = [(vx, vy) for vx, vy, vz in world_verts if vz <= (min_wz + eps)]
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
    obj_path: str,
    total_scale: float,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None:
    tpl = _tree_rect_template_unit(obj_path)
    if tpl is None:
        return None
    base_rect_u, span_rect_u = tpl
    base_rect = _scale_rect(base_rect_u, total_scale)
    span_rect = _scale_rect(span_rect_u, total_scale)
    base_rect = _expand_rect_to_min_size(base_rect, min_w=0.35, min_h=0.35)
    return base_rect, span_rect


def _tree_base_rects_from_instances(
    tree_instances: list[tuple[float, float, str, str, float, float]],
) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    for x, y, category, obj_name, total_scale, _radius in tree_instances:
        obj_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
        dual_rects = _tree_dual_rects_for_scale(obj_path, total_scale)
        if dual_rects is None:
            continue
        base_local, _span_local = dual_rects
        rects.append(_shift_rect(base_local, x, y))
    return rects


def _tree_span_rects_from_instances(
    tree_instances: list[tuple[float, float, str, str, float, float]],
) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    for x, y, category, obj_name, total_scale, _radius in tree_instances:
        obj_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
        dual_rects = _tree_dual_rects_for_scale(obj_path, total_scale)
        if dual_rects is None:
            continue
        _base_local, span_local = dual_rects
        rects.append(_shift_rect(span_local, x, y))
    return rects


def _protected_tree_span_rects_from_instances(
    tree_instances: list[tuple[float, float, str, str, float, float]],
) -> list[tuple[float, float, float, float]]:
    rects: list[tuple[float, float, float, float]] = []
    for x, y, category, obj_name, total_scale, _radius in tree_instances:
        if obj_name not in LOW_CANOPY_PROTECTED_TREE_NAMES:
            continue
        obj_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
        dual_rects = _tree_dual_rects_for_scale(obj_path, total_scale)
        if dual_rects is None:
            continue
        _base_local, span_local = dual_rects
        rects.append(_shift_rect(span_local, x, y))
    return rects


def _obj_planar_radius_cached(path: str) -> float:
    cached = _OBJ_PLANAR_RADIUS_CACHE.get(path)
    if cached is not None:
        return cached
    min_x, _, min_z, max_x, _, max_z = _obj_bounds_cached(path)
    cached = max((max_x - min_x) * 0.5, (max_z - min_z) * 0.5)
    _OBJ_PLANAR_RADIUS_CACHE[path] = cached
    return cached


def _tree_spacing_radius(obj_name: str, canopy_radius: float, difficulty_id: int) -> float:
    did = _clamp_difficulty_id(difficulty_id)
    mul = TREE_SPACING_RADIUS_MULTIPLIER_DEFAULT_BY_DIFFICULTY.get(did, TREE_SPACING_RADIUS_MULTIPLIER_DEFAULT)
    by_prefix = TREE_SPACING_RADIUS_BY_PREFIX_BY_DIFFICULTY.get(did, {})
    for prefix, ratio in by_prefix.items():
        if obj_name.startswith(prefix):
            mul = ratio
            break
    overlap_scale = TREE_CANOPY_OVERLAP_SCALE_BY_DIFFICULTY.get(did, 1.0)
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
    assets: list[tuple[str, str]],
) -> dict[str, dict[str, object]]:
    families: dict[str, dict[str, object]] = {}
    for category, obj_name in assets:
        family = _tree_family_prefix(obj_name)
        family_info = families.setdefault(
            family,
            {
                "total_weight": 0.0,
                "weighted_assets": [],
            },
        )
        weight = SNOW_DEAD_TREE_WEIGHT if category == "dead_snow" else 1.0
        if weight <= 0.0:
            continue
        family_info["total_weight"] += weight
        family_info["weighted_assets"].append((family_info["total_weight"], category, obj_name))
    return families


def _pick_weighted_tree_from_family(
    rng: random.Random,
    family_info: dict[str, object],
) -> tuple[str, str]:
    weighted_assets = family_info.get("weighted_assets", [])
    total_weight = float(family_info.get("total_weight", 0.0))
    if not weighted_assets or total_weight <= 0.0:
        raise RuntimeError("tree family has no weighted assets")
    r = rng.uniform(0.0, total_weight)
    for cum_w, category, obj_name in weighted_assets:
        if r <= cum_w:
            return category, obj_name
    return weighted_assets[-1][1], weighted_assets[-1][2]


def _count_tree_family_neighbors(
    placed_families: list[tuple[float, float, str]],
    *,
    x: float,
    y: float,
    family: str,
    radius_m: float,
) -> int:
    radius_sq = radius_m * radius_m
    count = 0
    for ox, oy, other_family in placed_families:
        if other_family != family:
            continue
        dx = x - ox
        dy = y - oy
        if (dx * dx + dy * dy) <= radius_sq:
            count += 1
    return count


def _rank_tree_families_for_point(
    rng: random.Random,
    *,
    x: float,
    y: float,
    difficulty_id: int,
    family_assets: dict[str, dict[str, object]],
    family_counts: dict[str, int],
    placed_families: list[tuple[float, float, str]],
    max_birch_count: int,
    birch_placed: int,
) -> list[str]:
    ranked: list[tuple[float, str]] = []
    did = _clamp_difficulty_id(difficulty_id)
    for family, family_info in family_assets.items():
        if family == BIRCH_TREE_PREFIX and birch_placed >= max_birch_count:
            continue
        base_weight = float(family_info.get("total_weight", 0.0))
        if base_weight <= 0.0:
            continue
        repeat_penalty = 1.0 / (1.0 + (family_counts.get(family, 0) * TREE_FAMILY_REPEAT_PENALTY))
        nearby_count = _count_tree_family_neighbors(
            placed_families,
            x=x,
            y=y,
            family=family,
            radius_m=max(6.0, TREE_LOCAL_CLUSTER_SEARCH_BY_DIFFICULTY_M.get(did, 3.0) * 1.8),
        )
        score = base_weight * repeat_penalty / (1.0 + (nearby_count * TREE_FAMILY_NEARBY_PENALTY))
        cluster_rule = TREE_FAMILY_CLUSTER_RULES.get(family, {}).get(did)
        if cluster_rule is not None:
            same_family_neighbors = _count_tree_family_neighbors(
                placed_families,
                x=x,
                y=y,
                family=family,
                radius_m=cluster_rule["radius_m"],
            )
            if same_family_neighbors > cluster_rule["max_neighbors"]:
                continue
            score /= (1.0 + (same_family_neighbors * 0.75))
        score *= rng.uniform(0.85, 1.15)
        ranked.append((score, family))
    ranked.sort(reverse=True)
    return [family for _score, family in ranked if _score > 0.0]


def _collision_shape_for_obj(obj_path: str, scale: float) -> int:
    key = (obj_path, round(scale, 4))
    cached = _OBJ_COLLISION_SHAPE_CACHE.get(key)
    if cached is not None:
        return cached

    flags = 0
    if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH"):
        flags = p.GEOM_FORCE_CONCAVE_TRIMESH
    shape = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale],
        flags=flags,
    )
    if shape < 0:
        raise RuntimeError(f"Failed to create collision shape for {obj_path}")
    _OBJ_COLLISION_SHAPE_CACHE[key] = shape
    return shape


def _compute_vertex_normals(verts: list[list[float]], indices: list[int]) -> list[list[float]]:
    normals = [[0.0, 0.0, 0.0] for _ in verts]
    for i in range(0, len(indices), 3):
        ia = indices[i]
        ib = indices[i + 1]
        ic = indices[i + 2]
        ax, ay, az = verts[ia]
        bx, by, bz = verts[ib]
        cx, cy, cz = verts[ic]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz - az
        nx = (uy * vz) - (uz * vy)
        ny = (uz * vx) - (ux * vz)
        nz = (ux * vy) - (uy * vx)
        normals[ia][0] += nx
        normals[ia][1] += ny
        normals[ia][2] += nz
        normals[ib][0] += nx
        normals[ib][1] += ny
        normals[ib][2] += nz
        normals[ic][0] += nx
        normals[ic][1] += ny
        normals[ic][2] += nz

    for n in normals:
        length = math.sqrt((n[0] * n[0]) + (n[1] * n[1]) + (n[2] * n[2]))
        if length > 1e-9:
            n[0] /= length
            n[1] /= length
            n[2] /= length
        else:
            n[0], n[1], n[2] = 0.0, 1.0, 0.0
    return normals


def _obj_mtl_path(obj_path: str) -> str | None:
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


def _parse_mtl_diffuse_colors(mtl_path: str | None) -> dict[str, list[float]]:
    if not mtl_path or not os.path.exists(mtl_path):
        return {}
    out: dict[str, list[float]] = {}
    current: str | None = None
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


def _parse_obj_material_meshes(obj_path: str) -> dict[str, tuple[list[list[float]], list[int], list[list[float]]]]:
    cached = _OBJ_MATERIAL_MESH_CACHE.get(obj_path)
    if cached is not None:
        return cached

    # Try binary pickle cache (much faster than text parsing)
    pkl_path = obj_path + ".meshcache.pkl"
    try:
        if os.path.exists(pkl_path) and os.path.getmtime(pkl_path) >= os.path.getmtime(obj_path):
            with open(pkl_path, "rb") as pf:
                cached = pickle.load(pf)
                _OBJ_MATERIAL_MESH_CACHE[obj_path] = cached
                return cached
    except Exception:
        pass

    vertices: list[tuple[float, float, float]] = []
    faces_by_mat: dict[str, list[list[int]]] = {}
    current_mat = "__default__"

    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                continue
            if line.lower().startswith("usemtl "):
                split = line.split(None, 1)
                current_mat = split[1].strip() if len(split) > 1 else "__default__"
                faces_by_mat.setdefault(current_mat, [])
                continue
            if line.startswith("f "):
                tokens = line.split()[1:]
                poly: list[int] = []
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

    mesh_by_mat: dict[str, tuple[list[list[float]], list[int], list[list[float]]]] = {}
    for mat_name, polys in faces_by_mat.items():
        remap: dict[int, int] = {}
        out_vertices: list[list[float]] = []
        out_indices: list[int] = []

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
    # Save binary cache for faster loading next time
    try:
        with open(pkl_path, "wb") as pf:
            pickle.dump(mesh_by_mat, pf, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:
        pass
    return mesh_by_mat


def _parse_obj_flat_mesh(obj_path: str) -> tuple[list[list[float]], list[int], list[list[float]]]:
    cached = _OBJ_FLAT_MESH_CACHE.get(obj_path)
    if cached is not None:
        return cached

    vertices: list[list[float]] = []
    indices: list[int] = []
    with open(obj_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                continue
            if line.startswith("f "):
                tokens = line.split()[1:]
                poly: list[int] = []
                for token in tokens:
                    v_tok = token.split("/")[0]
                    if not v_tok:
                        continue
                    idx = int(v_tok)
                    if idx < 0:
                        idx = len(vertices) + 1 + idx
                    poly.append(idx - 1)
                if len(poly) < 3:
                    continue
                for i in range(1, len(poly) - 1):
                    indices.extend([poly[0], poly[i], poly[i + 1]])

    normals = _compute_vertex_normals(vertices, indices) if indices else [[0.0, 1.0, 0.0] for _ in vertices]
    out = (vertices, indices, normals)
    _OBJ_FLAT_MESH_CACHE[obj_path] = out
    return out


def _tree_candidate_points(rng: random.Random, count: int) -> list[tuple[float, float]]:
    half = _map_half_extent() - FOREST_EDGE_MARGIN_M
    width = half * 2.0
    cells_side = max(8, int(math.ceil(math.sqrt(count * 1.18))))
    cell = width / cells_side
    jitter = cell * 0.33
    points: list[tuple[float, float]] = []

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
    candidates: list[tuple[float, float]],
    *,
    count: int,
    half: float,
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
    instances: list[tuple[float, float, str, str, float, float]],
    *,
    radius_scale: float,
    min_radius: float = 0.0,
    max_radius: float | None = None,
) -> list[tuple[float, float, str, str, float, float]]:
    scaled: list[tuple[float, float, str, str, float, float]] = []
    for x, y, category, obj_name, total_scale, radius in instances:
        out_radius = max(min_radius, radius * radius_scale)
        if max_radius is not None:
            out_radius = min(out_radius, max_radius)
        scaled.append((x, y, category, obj_name, total_scale, out_radius))
    return scaled


def _pick_tree_instances(
    rng: random.Random,
    *,
    count: int,
    assets: list[tuple[str, str]],
    clearance_m: float,
    difficulty_id: int,
) -> list[tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    placed: list[tuple[float, float, str, str, float, float]] = []
    placed_families: list[tuple[float, float, str]] = []
    grid = _SpatialGrid()
    placed_base_rects: list[tuple[float, float, float, float]] = []
    placed_span_rects: list[tuple[float, float, float, float]] = []
    debug_tree_rects: list[tuple[tuple[float, float, float, float], tuple[float, float, float, float]]] = []
    candidates = _tree_candidate_points(rng, count)
    half = _map_half_extent() - FOREST_EDGE_MARGIN_M
    family_assets = _build_tree_family_assets(assets)
    family_counts = {family: 0 for family in family_assets}
    non_birch_families = [family for family in family_assets if family != BIRCH_TREE_PREFIX]
    max_birch_count = int(math.floor(count * BIRCH_TREE_MAX_RATIO)) if non_birch_families else count
    birch_placed = 0
    difficulty_id = _clamp_difficulty_id(difficulty_id)
    cluster_max = TREE_LOCAL_CLUSTER_MAX_BY_DIFFICULTY.get(difficulty_id, 1)
    cluster_search_m = TREE_LOCAL_CLUSTER_SEARCH_BY_DIFFICULTY_M.get(difficulty_id, 3.0)
    canopy_overlap_scale = TREE_CANOPY_OVERLAP_SCALE_BY_DIFFICULTY.get(difficulty_id, 1.0)

    for relax in (1.0, 0.85, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            ranked_families = _rank_tree_families_for_point(
                rng,
                x=x,
                y=y,
                difficulty_id=difficulty_id,
                family_assets=family_assets,
                family_counts=family_counts,
                placed_families=placed_families,
                max_birch_count=max_birch_count,
                birch_placed=birch_placed,
            )
            if not ranked_families:
                continue

            for family in ranked_families:
                family_info = family_assets.get(family)
                if not family_info:
                    continue
                category, obj_name = _pick_weighted_tree_from_family(rng, family_info)
                obj_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
                if not os.path.exists(obj_path):
                    continue

                size_mul = rng.uniform(TREE_SCALE_MIN, TREE_SCALE_MAX)
                total_scale = PREVIEW_UNIFORM_SCALE * size_mul
                if FAST_BUILD_MODE and FAST_SCALE_STEP > 0.0:
                    total_scale = max(0.01, round(total_scale / FAST_SCALE_STEP) * FAST_SCALE_STEP)
                canopy_radius = _obj_planar_radius_cached(obj_path) * total_scale
                spacing_radius = _tree_spacing_radius(obj_name, canopy_radius, difficulty_id)
                occupancy_radius = _tree_occupancy_radius(obj_name, canopy_radius)
                rects = _tree_dual_rects_for_scale(obj_path, total_scale)
                if rects is not None:
                    base_local, span_local = rects
                    base_world = _shift_rect(base_local, x, y)
                    span_world = _shift_rect(span_local, x, y)
                    if (
                        base_world[0] < -half
                        or base_world[1] > half
                        or base_world[2] < -half
                        or base_world[3] > half
                    ):
                        continue
                    if difficulty_id in (2, 3) and obj_name in LOW_CANOPY_PROTECTED_TREE_NAMES:
                        span_collision_world = span_world
                    else:
                        span_collision_world = _shrink_rect_from_center(
                            span_world,
                            max(0.05, canopy_overlap_scale * relax),
                        )
                    blocked = False
                    for i in range(len(placed_base_rects)):
                        if _rect_overlap(base_world, placed_base_rects[i]):
                            blocked = True
                            break
                        if _rect_overlap(span_collision_world, placed_span_rects[i]):
                            blocked = True
                            break
                    if blocked:
                        continue
                else:
                    if (
                        (x - occupancy_radius) < -half
                        or (x + occupancy_radius) > half
                        or (y - occupancy_radius) < -half
                        or (y + occupancy_radius) > half
                    ):
                        continue
                    clearance_factor = TREE_TREE_CLEARANCE_FACTOR_BY_DIFFICULTY.get(
                        difficulty_id,
                        TREE_TREE_CLEARANCE_FACTOR,
                    )
                    if grid.has_conflict(x, y, spacing_radius, clearance_m * relax * clearance_factor):
                        continue

                local_neighbors = grid.count_neighbors(
                    x,
                    y,
                    max(cluster_search_m, spacing_radius * 2.2),
                )
                if local_neighbors > (cluster_max + (1 if relax < 0.9 else 0)):
                    continue

                placed.append((x, y, category, obj_name, total_scale, occupancy_radius))
                placed_families.append((x, y, family))
                family_counts[family] = family_counts.get(family, 0) + 1
                grid.insert(x, y, spacing_radius)
                if rects is not None:
                    placed_base_rects.append(base_world)
                    placed_span_rects.append(span_collision_world)
                    debug_tree_rects.append((base_world, span_collision_world))
                if obj_name.startswith(BIRCH_TREE_PREFIX):
                    birch_placed += 1
                break

        if len(placed) >= count:
            break
        rng.shuffle(candidates)

    if TREE_RECT_DEBUG_DRAW and _is_gui_connected():
        for base_world, span_world in debug_tree_rects:
            _draw_rect_xy(*base_world, color=[0.95, 0.88, 0.20], width=1.4, z=0.03)
            _draw_rect_xy(*span_world, color=[0.95, 0.30, 0.95], width=1.0, z=0.05)

    return placed


def _pick_shrub_instances(
    rng: random.Random,
    *,
    count: int,
    assets: list[tuple[str, str]],
    clearance_m: float,
    tree_instances: list[tuple[float, float, str, str, float, float]],
    tree_base_rects: list[tuple[float, float, float, float]] | None = None,
    protected_tree_span_rects: list[tuple[float, float, float, float]] | None = None,
    occupied_instances: list[tuple[float, float, str, str, float, float]] | None = None,
    tree_occupancy_scale: float = 1.0,
    tree_occupancy_cap_m: float | None = None,
) -> list[tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    half = _small_asset_half_extent()
    weighted_assets: list[tuple[float, str, str]] = []
    total_weight = 0.0
    for category, obj_name in assets:
        weight = BUSH_BERRIES_WEIGHT if obj_name.startswith(BUSH_BERRIES_PREFIX) else 1.0
        if weight <= 0.0:
            continue
        total_weight += weight
        weighted_assets.append((total_weight, category, obj_name))
    if not weighted_assets:
        return []

    def _pick_weighted_bush() -> tuple[str, str]:
        r = rng.uniform(0.0, total_weight)
        for cum_w, category, obj_name in weighted_assets:
            if r <= cum_w:
                return category, obj_name
        return weighted_assets[-1][1], weighted_assets[-1][2]

    def _bush_cell_key(x: float, y: float) -> tuple[int, int]:
        cell = BUSH_DISTRIBUTION_CELL_SIZE_M
        return (int((x + half) // cell), int((y + half) // cell))

    candidates: list[tuple[float, float]] = []
    candidates.extend(_tree_candidate_points(rng, max(count * 2, 120)))

    for tx, ty, _, _, _, tr in tree_instances:
        for _ in range(2):
            ang = rng.uniform(0.0, math.tau)
            dist = tr + rng.uniform(1.1, 3.2)
            x = tx + math.cos(ang) * dist
            y = ty + math.sin(ang) * dist
            if -half <= x <= half and -half <= y <= half:
                candidates.append((x, y))

    _extend_small_asset_candidates(rng, candidates, count=count * 8, half=half)
    rng.shuffle(candidates)

    occupied: list[tuple[float, float, float]] = []
    for x, y, _, _, _, r in tree_instances:
        tree_r = max(SMALL_ASSET_TREE_OCCUPANCY_MIN_M, r * tree_occupancy_scale)
        if tree_occupancy_cap_m is not None:
            tree_r = min(tree_r, tree_occupancy_cap_m)
        occupied.append((x, y, tree_r))
    if occupied_instances:
        occupied.extend([(x, y, r) for x, y, _, _, _, r in occupied_instances])
    grid = _SpatialGrid()
    for _ox, _oy, _orad in occupied:
        grid.insert(_ox, _oy, _orad)
    per_cell_counts: dict[tuple[int, int], int] = {}
    placed: list[tuple[float, float, str, str, float, float]] = []
    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            cell_key = _bush_cell_key(x, y)
            if per_cell_counts.get(cell_key, 0) >= BUSH_DISTRIBUTION_MAX_PER_CELL:
                continue
            category, obj_name = _pick_weighted_bush()
            obj_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
            if not os.path.exists(obj_path):
                continue

            size_mul = rng.uniform(SHRUB_SCALE_MIN, SHRUB_SCALE_MAX)
            total_scale = PREVIEW_UNIFORM_SCALE * size_mul
            radius = _obj_planar_radius_cached(obj_path) * total_scale

            if (
                (x - radius) < -half
                or (x + radius) > half
                or (y - radius) < -half
                or (y + radius) > half
            ):
                continue

            candidate_rect = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(candidate_rect, rect) for rect in tree_base_rects):
                continue
            if protected_tree_span_rects and any(_rect_overlap(candidate_rect, rect) for rect in protected_tree_span_rects):
                continue

            if grid.has_conflict(x, y, radius, clearance_m * relax):
                continue

            placed.append((x, y, category, obj_name, total_scale, radius))
            grid.insert(x, y, radius)
            per_cell_counts[cell_key] = per_cell_counts.get(cell_key, 0) + 1

        if len(placed) >= count:
            break
        rng.shuffle(candidates)

    return placed


def _pick_rock_stump_instances(
    rng: random.Random,
    *,
    count: int,
    assets: list[tuple[str, str]],
    mode_id: int,
    clearance_m: float,
    occupied_instances: list[tuple[float, float, str, str, float, float]],
    tree_base_rects: list[tuple[float, float, float, float]] | None = None,
    protected_tree_span_rects: list[tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    weighted_assets: list[tuple[float, str, str]] = []
    total_weight = 0.0
    mode_weight_bonus = ROCK_STUMP_MODE_WEIGHT_BONUS.get(_clamp_mode_id(mode_id), {})
    for category, obj_name in assets:
        weight = ROCK_STUMP_MODEL_WEIGHT_BONUS.get(obj_name, 1.0)
        weight *= mode_weight_bonus.get(obj_name, 1.0)
        if weight <= 0.0:
            continue
        total_weight += weight
        weighted_assets.append((total_weight, category, obj_name))
    if not weighted_assets:
        return []

    def _pick_weighted_rock_stump() -> tuple[str, str]:
        r = rng.uniform(0.0, total_weight)
        for cum_w, category, obj_name in weighted_assets:
            if r <= cum_w:
                return category, obj_name
        return weighted_assets[-1][1], weighted_assets[-1][2]

    half = _small_asset_half_extent()
    candidates: list[tuple[float, float]] = []
    candidates.extend(_tree_candidate_points(rng, max(count * 2, 120)))

    _extend_small_asset_candidates(rng, candidates, count=count * 10, half=half)
    rng.shuffle(candidates)

    occupied: list[tuple[float, float, float]] = [(x, y, r) for x, y, _, _, _, r in occupied_instances]
    grid = _SpatialGrid()
    for _ox, _oy, _orad in occupied:
        grid.insert(_ox, _oy, _orad)
    placed: list[tuple[float, float, str, str, float, float]] = []

    priority_names = [n for n in ROCK_STUMP_MODEL_WEIGHT_BONUS if any(obj_name == n for _, obj_name in assets)]
    for target_name in priority_names:
        if len(placed) >= count:
            break
        target_assets = [(cat, name) for cat, name in assets if name == target_name]
        if not target_assets:
            continue
        placed_one = False
        for _ in range(140):
            x = rng.uniform(-half, half)
            y = rng.uniform(-half, half)
            category, obj_name = rng.choice(target_assets)
            obj_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
            if not os.path.exists(obj_path):
                continue
            size_mul = rng.uniform(ROCK_STUMP_SCALE_MIN, ROCK_STUMP_SCALE_MAX)
            size_mul *= ROCK_STUMP_MODEL_SCALE_FACTOR.get(obj_name, 1.0)
            total_scale = PREVIEW_UNIFORM_SCALE * size_mul
            radius = _obj_planar_radius_cached(obj_path) * total_scale
            if (
                (x - radius) < -half
                or (x + radius) > half
                or (y - radius) < -half
                or (y + radius) > half
            ):
                continue
            candidate_rect = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(candidate_rect, rect) for rect in tree_base_rects):
                continue
            if protected_tree_span_rects and any(_rect_overlap(candidate_rect, rect) for rect in protected_tree_span_rects):
                continue
            if grid.has_conflict(x, y, radius, clearance_m * 0.35):
                continue
            placed.append((x, y, category, obj_name, total_scale, radius))
            grid.insert(x, y, radius)
            placed_one = True
            break
        if not placed_one:
            continue

    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            category, obj_name = _pick_weighted_rock_stump()
            obj_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
            if not os.path.exists(obj_path):
                continue

            size_mul = rng.uniform(ROCK_STUMP_SCALE_MIN, ROCK_STUMP_SCALE_MAX)
            size_mul *= ROCK_STUMP_MODEL_SCALE_FACTOR.get(obj_name, 1.0)
            total_scale = PREVIEW_UNIFORM_SCALE * size_mul
            radius = _obj_planar_radius_cached(obj_path) * total_scale

            if (
                (x - radius) < -half
                or (x + radius) > half
                or (y - radius) < -half
                or (y + radius) > half
            ):
                continue

            candidate_rect = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(candidate_rect, rect) for rect in tree_base_rects):
                continue
            if protected_tree_span_rects and any(_rect_overlap(candidate_rect, rect) for rect in protected_tree_span_rects):
                continue

            if grid.has_conflict(x, y, radius, clearance_m * relax):
                continue

            placed.append((x, y, category, obj_name, total_scale, radius))
            grid.insert(x, y, radius)

        if len(placed) >= count:
            break
        rng.shuffle(candidates)

    return placed


def _pick_log_instances(
    rng: random.Random,
    *,
    count: int,
    assets: list[tuple[str, str]],
    clearance_m: float,
    occupied_instances: list[tuple[float, float, str, str, float, float]],
    tree_base_rects: list[tuple[float, float, float, float]] | None = None,
    protected_tree_span_rects: list[tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    half = _small_asset_half_extent()
    candidates: list[tuple[float, float]] = []
    candidates.extend(_tree_candidate_points(rng, max(count * 3, 120)))

    _extend_small_asset_candidates(rng, candidates, count=count * 12, half=half)
    rng.shuffle(candidates)

    occupied: list[tuple[float, float, float]] = [(x, y, r) for x, y, _, _, _, r in occupied_instances]
    grid = _SpatialGrid()
    for _ox, _oy, _orad in occupied:
        grid.insert(_ox, _oy, _orad)
    placed: list[tuple[float, float, str, str, float, float]] = []
    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            category, obj_name = rng.choice(assets)
            obj_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
            if not os.path.exists(obj_path):
                continue

            size_mul = rng.uniform(LOG_SCALE_MIN, LOG_SCALE_MAX)
            total_scale = PREVIEW_UNIFORM_SCALE * size_mul
            radius = _obj_planar_radius_cached(obj_path) * total_scale

            if (
                (x - radius) < -half
                or (x + radius) > half
                or (y - radius) < -half
                or (y + radius) > half
            ):
                continue

            candidate_rect = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(candidate_rect, rect) for rect in tree_base_rects):
                continue
            if protected_tree_span_rects and any(_rect_overlap(candidate_rect, rect) for rect in protected_tree_span_rects):
                continue

            if grid.has_conflict(x, y, radius, clearance_m * relax):
                continue

            placed.append((x, y, category, obj_name, total_scale, radius))
            grid.insert(x, y, radius)

        if len(placed) >= count:
            break
        rng.shuffle(candidates)

    return placed


def _pick_ground_cover_instances(
    rng: random.Random,
    *,
    count: int,
    assets: list[tuple[str, str]],
    mode_id: int,
    clearance_m: float,
    occupied_instances: list[tuple[float, float, str, str, float, float]],
    tree_base_rects: list[tuple[float, float, float, float]] | None = None,
    tree_span_rects: list[tuple[float, float, float, float]] | None = None,
    protected_tree_span_rects: list[tuple[float, float, float, float]] | None = None,
) -> list[tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    weighted_assets: list[tuple[float, str, str]] = []
    total_weight = 0.0
    mode_weight_bonus = GROUND_COVER_MODE_WEIGHT_BONUS.get(_clamp_mode_id(mode_id), {})
    for category, obj_name in assets:
        weight = GROUND_COVER_MODEL_WEIGHT_BONUS.get(obj_name, 1.0)
        weight *= mode_weight_bonus.get(obj_name, 1.0)
        if weight <= 0.0:
            continue
        total_weight += weight
        weighted_assets.append((total_weight, category, obj_name))
    if not weighted_assets:
        return []

    def _pick_weighted_ground_cover() -> tuple[str, str]:
        r = rng.uniform(0.0, total_weight)
        for cum_w, category, obj_name in weighted_assets:
            if r <= cum_w:
                return category, obj_name
        return weighted_assets[-1][1], weighted_assets[-1][2]

    half = _small_asset_half_extent()
    candidates: list[tuple[float, float]] = []
    candidates.extend(_tree_candidate_points(rng, max(count * 2, 80)))
    _extend_small_asset_candidates(rng, candidates, count=count * 12, half=half)
    rng.shuffle(candidates)

    occupied: list[tuple[float, float, float]] = [(x, y, r) for x, y, _, _, _, r in occupied_instances]
    placed: list[tuple[float, float, str, str, float, float]] = []
    placed_by_name: dict[str, int] = {}

    priority_names = [
        n
        for n, w in GROUND_COVER_MODEL_WEIGHT_BONUS.items()
        if w > 1.0 and any(obj_name == n for _, obj_name in assets)
    ]
    for target_name in priority_names:
        if len(placed) >= count:
            break
        target_assets = [(cat, name) for cat, name in assets if name == target_name]
        if not target_assets:
            continue
        for _ in range(180):
            x = rng.uniform(-half, half)
            y = rng.uniform(-half, half)
            category, obj_name = rng.choice(target_assets)
            obj_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
            if not os.path.exists(obj_path):
                continue
            size_mul = rng.uniform(GROUND_COVER_SCALE_MIN, GROUND_COVER_SCALE_MAX)
            size_mul *= GROUND_COVER_MODEL_SCALE_FACTOR.get(obj_name, 1.0)
            total_scale = PREVIEW_UNIFORM_SCALE * size_mul
            radius = _obj_planar_radius_cached(obj_path) * total_scale
            if (
                (x - radius) < -half
                or (x + radius) > half
                or (y - radius) < -half
                or (y + radius) > half
            ):
                continue
            if obj_name in NORMAL_ONLY_SINGLE_SPAWN_GROUND_COVER and placed_by_name.get(obj_name, 0) >= 1:
                continue
            candidate_rect = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(candidate_rect, rect) for rect in tree_base_rects):
                continue
            if protected_tree_span_rects and any(_rect_overlap(candidate_rect, rect) for rect in protected_tree_span_rects):
                continue
            if obj_name.startswith("Corn_") and tree_span_rects and any(
                _rect_overlap(candidate_rect, rect) for rect in tree_span_rects
            ):
                continue
            keep = True
            for ox, oy, orad in occupied:
                dx = x - ox
                dy = y - oy
                min_dist = radius + orad + (clearance_m * 0.65)
                if (dx * dx + dy * dy) < (min_dist * min_dist):
                    keep = False
                    break
            if not keep:
                continue
            placed.append((x, y, category, obj_name, total_scale, radius))
            occupied.append((x, y, radius))
            placed_by_name[obj_name] = placed_by_name.get(obj_name, 0) + 1
            break

    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            category, obj_name = _pick_weighted_ground_cover()
            obj_path = os.path.join(ASSET_BASE_DIR, category, obj_name)
            if not os.path.exists(obj_path):
                continue

            size_mul = rng.uniform(GROUND_COVER_SCALE_MIN, GROUND_COVER_SCALE_MAX)
            size_mul *= GROUND_COVER_MODEL_SCALE_FACTOR.get(obj_name, 1.0)
            total_scale = PREVIEW_UNIFORM_SCALE * size_mul
            radius = _obj_planar_radius_cached(obj_path) * total_scale

            if (
                (x - radius) < -half
                or (x + radius) > half
                or (y - radius) < -half
                or (y + radius) > half
            ):
                continue
            if obj_name in NORMAL_ONLY_SINGLE_SPAWN_GROUND_COVER and placed_by_name.get(obj_name, 0) >= 1:
                continue

            candidate_rect = _circle_bounds_rect(x, y, radius)
            if tree_base_rects and any(_rect_overlap(candidate_rect, rect) for rect in tree_base_rects):
                continue
            if protected_tree_span_rects and any(_rect_overlap(candidate_rect, rect) for rect in protected_tree_span_rects):
                continue
            if obj_name.startswith("Corn_") and tree_span_rects and any(
                _rect_overlap(candidate_rect, rect) for rect in tree_span_rects
            ):
                continue

            keep = True
            for ox, oy, orad in occupied:
                dx = x - ox
                dy = y - oy
                min_dist = radius + orad + (clearance_m * relax)
                if (dx * dx + dy * dy) < (min_dist * min_dist):
                    keep = False
                    break
            if not keep:
                continue

            placed.append((x, y, category, obj_name, total_scale, radius))
            occupied.append((x, y, radius))
            placed_by_name[obj_name] = placed_by_name.get(obj_name, 0) + 1

        if len(placed) >= count:
            break
        rng.shuffle(candidates)

    return placed


def _split_asset_count(
    total_count: int,
    primary_assets: list[tuple[str, str]],
    secondary_assets: list[tuple[str, str]],
    *,
    primary_ratio: float | None = None,
    secondary_cap: int | None = None,
) -> tuple[int, int]:
    if total_count <= 0:
        return 0, 0
    if not primary_assets and not secondary_assets:
        return 0, 0
    if not secondary_assets:
        return total_count, 0
    if not primary_assets:
        secondary = total_count if secondary_cap is None else min(total_count, max(0, secondary_cap))
        return 0, secondary

    if primary_ratio is None:
        total_assets = max(1, len(primary_assets) + len(secondary_assets))
        primary_ratio = len(primary_assets) / total_assets

    primary = int(round(total_count * primary_ratio))
    primary = max(0, min(total_count, primary))
    secondary = total_count - primary
    if secondary_cap is not None:
        secondary = min(secondary, max(0, secondary_cap))
        primary = total_count - secondary
    return primary, secondary


def _spawn_colored_obj(
    *,
    obj_path: str,
    scale: float,
    double_sided_flags: int,
) -> list[int]:
    mode_key = "flat" if DISABLE_MODEL_MATERIALS_FOR_DEBUG else "mtl"
    cache_key = (obj_path, round(scale, 4), int(double_sided_flags), mode_key)
    vis_ids = _OBJ_VISUAL_SHAPE_CACHE.get(cache_key)
    if vis_ids is None:
        if DISABLE_MODEL_MATERIALS_FOR_DEBUG:
            verts, indices, normals = _parse_obj_flat_mesh(obj_path)
            kwargs = {
                "vertices": verts,
                "indices": indices,
                "normals": normals,
                "meshScale": [scale, scale, scale],
                "rgbaColor": DEBUG_MODEL_RGBA,
                "specularColor": [0.0, 0.0, 0.0],
            }
            if double_sided_flags:
                kwargs["flags"] = double_sided_flags
            vis = p.createVisualShape(p.GEOM_MESH, **kwargs)
            vis_ids = [vis] if vis >= 0 else []
        else:
            material_meshes = _parse_obj_material_meshes(obj_path)
            mtl_colors = _parse_mtl_diffuse_colors(_obj_mtl_path(obj_path))
            default_rgba = [0.7, 0.7, 0.7, 1.0]
            vis_ids = []
            if material_meshes:
                for mat_name, (verts, indices, normals) in material_meshes.items():
                    rgba = mtl_colors.get(mat_name, default_rgba)
                    kwargs = {
                        "vertices": verts,
                        "indices": indices,
                        "normals": normals,
                        "meshScale": [scale, scale, scale],
                        "rgbaColor": rgba,
                        "specularColor": [0.0, 0.0, 0.0],
                    }
                    if double_sided_flags:
                        kwargs["flags"] = double_sided_flags
                    vis = p.createVisualShape(p.GEOM_MESH, **kwargs)
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
                vis = p.createVisualShape(p.GEOM_MESH, **kwargs)
                if vis >= 0:
                    vis_ids = [vis]
        _OBJ_VISUAL_SHAPE_CACHE[cache_key] = vis_ids

    return list(vis_ids)


def _spawn_asset_instance(
    *,
    category: str,
    obj_name: str,
    x: float,
    y: float,
    yaw_deg: float,
    scale: float,
    flags: int,
    enable_collision: bool = True,
    asset_class: str | None = None,
) -> bool:
    obj_path = _obj_path_for_spawn(category, obj_name, asset_class=asset_class)
    if not os.path.exists(obj_path):
        return False

    effective_scale = scale
    if FAST_BUILD_MODE and FAST_SCALE_STEP > 0.0:
        effective_scale = max(0.01, round(scale / FAST_SCALE_STEP) * FAST_SCALE_STEP)

    min_x, min_y, min_z, max_x, _, max_z = _obj_bounds_cached(obj_path)
    cx = (min_x + max_x) * 0.5
    cz = (min_z + max_z) * 0.5
    z0 = max(0.0, -min_y)
    yaw_rad = math.radians(yaw_deg)
    spawn_pos = [x - cx * effective_scale, y + cz * effective_scale, z0 * effective_scale]

    spawn_quat = p.getQuaternionFromEuler([1.5708, 0.0, yaw_rad])
    vis_shapes = _spawn_colored_obj(
        obj_path=obj_path,
        scale=effective_scale,
        double_sided_flags=flags,
    )
    if not vis_shapes:
        return False

    col_shape = _collision_shape_for_obj(obj_path, effective_scale) if enable_collision else -1
    extra_vis = vis_shapes[1:]
    if not extra_vis:
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shapes[0],
            basePosition=spawn_pos,
            baseOrientation=spawn_quat,
            useMaximalCoordinates=True,
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
        useMaximalCoordinates=True,
    )
    return True


def _spawn_instances_as_single_multibody(
    *,
    instances: list[tuple[float, float, str, str, float, float]],
    rng: random.Random,
    flags: int,
    class_name: str,
    enable_collision: bool,
    fixed_yaw_deg: float | None = None,
) -> int:
    if not instances:
        return 0

    linkMasses: list[float] = []
    linkCollisionShapeIndices: list[int] = []
    linkVisualShapeIndices: list[int] = []
    linkPositions: list[list[float]] = []
    linkOrientations: list[list[float]] = []
    linkInertialFramePositions: list[list[float]] = []
    linkInertialFrameOrientations: list[list[float]] = []
    linkParentIndices: list[int] = []
    linkJointTypes: list[int] = []
    linkJointAxis: list[list[float]] = []
    placed_count = 0

    def _flush_batch() -> None:
        nonlocal linkMasses, linkCollisionShapeIndices, linkVisualShapeIndices
        nonlocal linkPositions, linkOrientations, linkInertialFramePositions
        nonlocal linkInertialFrameOrientations, linkParentIndices, linkJointTypes, linkJointAxis
        if not linkVisualShapeIndices:
            return
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            linkMasses=linkMasses,
            linkCollisionShapeIndices=linkCollisionShapeIndices,
            linkVisualShapeIndices=linkVisualShapeIndices,
            linkPositions=linkPositions,
            linkOrientations=linkOrientations,
            linkInertialFramePositions=linkInertialFramePositions,
            linkInertialFrameOrientations=linkInertialFrameOrientations,
            linkParentIndices=linkParentIndices,
            linkJointTypes=linkJointTypes,
            linkJointAxis=linkJointAxis,
            useMaximalCoordinates=True,
        )
        linkMasses = []
        linkCollisionShapeIndices = []
        linkVisualShapeIndices = []
        linkPositions = []
        linkOrientations = []
        linkInertialFramePositions = []
        linkInertialFrameOrientations = []
        linkParentIndices = []
        linkJointTypes = []
        linkJointAxis = []

    for x, y, category, obj_name, scale, _ in instances:
        obj_path = _obj_path_for_spawn(category, obj_name, asset_class=class_name)
        if not os.path.exists(obj_path):
            continue

        effective_scale = scale
        if FAST_BUILD_MODE and FAST_SCALE_STEP > 0.0:
            effective_scale = max(0.01, round(scale / FAST_SCALE_STEP) * FAST_SCALE_STEP)

        min_x, min_y, min_z, max_x, _, max_z = _obj_bounds_cached(obj_path)
        cx = (min_x + max_x) * 0.5
        cz = (min_z + max_z) * 0.5
        z0 = max(0.0, -min_y)

        yaw_deg = fixed_yaw_deg if fixed_yaw_deg is not None else rng.uniform(-180.0, 180.0)
        yaw_rad = math.radians(yaw_deg)
        spawn_pos = [x - cx * effective_scale, y + cz * effective_scale, z0 * effective_scale]
        spawn_quat = p.getQuaternionFromEuler([1.5708, 0.0, yaw_rad])
        vis_shapes = _spawn_colored_obj(
            obj_path=obj_path,
            scale=effective_scale,
            double_sided_flags=flags,
        )
        if not vis_shapes:
            continue

        if len(vis_shapes) > TREE_BATCH_MAX_LINKS:
            # Fallback to standalone spawn if batching can't cover this case.
            if _spawn_asset_instance(
                category=category,
                obj_name=obj_name,
                x=x,
                y=y,
                yaw_deg=yaw_deg,
                scale=scale,
                flags=flags,
                enable_collision=enable_collision,
                asset_class=class_name,
            ):
                LAST_MAP_SPAWNED.append((class_name, category, obj_name))
                placed_count += 1
            continue

        if linkVisualShapeIndices and (len(linkVisualShapeIndices) + len(vis_shapes) > TREE_BATCH_MAX_LINKS):
            _flush_batch()

        col_shape = _collision_shape_for_obj(obj_path, effective_scale) if enable_collision else -1
        for idx, vis_shape in enumerate(vis_shapes):
            linkMasses.append(0.0)
            linkCollisionShapeIndices.append(col_shape if idx == 0 else -1)
            linkVisualShapeIndices.append(vis_shape)
            linkPositions.append(spawn_pos)
            linkOrientations.append(spawn_quat)
            linkInertialFramePositions.append([0.0, 0.0, 0.0])
            linkInertialFrameOrientations.append([0.0, 0.0, 0.0, 1.0])
            linkParentIndices.append(0)
            linkJointTypes.append(p.JOINT_FIXED)
            linkJointAxis.append([0.0, 0.0, 1.0])

        LAST_MAP_SPAWNED.append((class_name, category, obj_name))
        placed_count += 1

    _flush_batch()
    return placed_count


def _spawn_forest_assets(*, mode_id: int, difficulty_id: int, seed: int) -> dict[str, object]:
    global LAST_MAP_SPAWNED
    LAST_MAP_SPAWNED = []

    mode_id = _clamp_mode_id(mode_id)
    difficulty_id = _clamp_difficulty_id(difficulty_id)
    mode_cfg = MAP_MODE_CONFIG[mode_id]
    diff_cfg = DIFFICULTY_CONFIG[difficulty_id]
    mode_class_count_mul = MODE_CLASS_COUNT_MULTIPLIER.get(mode_id, {})
    mode_rock_stump_total_mul = MODE_ROCK_STUMP_TOTAL_MULTIPLIER.get(mode_id, 1.0)
    mode_rock_stump_primary_ratio = MODE_ROCK_STUMP_PRIMARY_RATIO.get(mode_id)
    mode_tree_occupancy_mul = MODE_SMALL_ASSET_TREE_OCCUPANCY_MULTIPLIER.get(mode_id, {})
    mode_clearance_mul = MODE_SMALL_ASSET_CLEARANCE_MULTIPLIER.get(mode_id, {})
    diff_trunk_occupancy_mul = TRUNK_OCCUPANCY_MULTIPLIER_BY_DIFFICULTY.get(difficulty_id, {})
    diff_trunk_clearance_mul = TRUNK_CLEARANCE_MULTIPLIER_BY_DIFFICULTY.get(difficulty_id, {})
    diff_trunk_fallback_fill = TRUNK_FALLBACK_FILL_BY_DIFFICULTY.get(difficulty_id, {})
    flags = p.VISUAL_SHAPE_DOUBLE_SIDED if hasattr(p, "VISUAL_SHAPE_DOUBLE_SIDED") else 0
    rng = random.Random(seed)
    assets = _resolve_assets_for_class(mode_id)

    if SINGLE_TREE_TEST_MODE:
        trees = assets.get("trees", [])
        if trees:
            category, obj_name = trees[0]
            _spawn_asset_instance(
                category=category,
                obj_name=obj_name,
                x=0.0,
                y=0.0,
                yaw_deg=0.0,
                scale=PREVIEW_UNIFORM_SCALE,
                flags=flags,
                enable_collision=not DISABLE_TREE_COLLISIONS_FOR_DEBUG,
                asset_class="trees",
            )
            LAST_MAP_SPAWNED.append(("trees", category, obj_name))
            return {
                "map_name": MAP_NAME,
                "mode_name": mode_cfg["name"],
                "difficulty_name": diff_cfg["name"],
                "seed": seed,
                "placed_count": 1,
                "class_counts": {
                    "trees": 1,
                    "bushes": 0,
                    "logs": 0,
                    "rocks": 0,
                    "stumps": 0,
                    "plants": 0,
                    "cactus": 0,
                },
            }
        return {
            "map_name": MAP_NAME,
            "mode_name": mode_cfg["name"],
            "difficulty_name": diff_cfg["name"],
            "seed": seed,
            "placed_count": 0,
            "class_counts": {
                "trees": 0,
                "bushes": 0,
                "logs": 0,
                "rocks": 0,
                "stumps": 0,
                "plants": 0,
                "cactus": 0,
            },
        }

    if GROUND_ONLY_MODE:
        by_class = {
            "trees": 0,
            "bushes": 0,
            "logs": 0,
            "rocks": 0,
            "stumps": 0,
            "plants": 0,
            "cactus": 0,
        }
        return {
            "map_name": MAP_NAME,
            "mode_name": mode_cfg["name"],
            "difficulty_name": diff_cfg["name"],
            "seed": seed,
            "placed_count": 0,
            "class_counts": by_class,
        }

    def _scaled_count(class_name: str, base_count: int) -> int:
        if not CLASS_ENABLE.get(class_name, True):
            return 0
        class_mul = CLASS_DENSITY_MULTIPLIER.get(class_name, 1.0)
        class_mul *= mode_class_count_mul.get(class_name, 1.0)
        class_mul *= DIFFICULTY_DENSITY_MULTIPLIER.get(difficulty_id, 1.0)
        if class_name == "trees":
            class_mul *= TREE_DIFFICULTY_MULTIPLIER.get(difficulty_id, 1.0)
        return max(0, int(round(base_count * DENSITY_MULTIPLIER * class_mul)))

    spawn_sections = {
        "trees": {"count": _scaled_count("trees", diff_cfg["tree_count"])},
        "logs": {"count": _scaled_count("logs", diff_cfg["log_count"])},
        "bushes": {"count": _scaled_count("bushes", diff_cfg["bush_count"])},
        "rocks": {"count": 0},
        "stumps": {"count": 0},
        "plants": {"count": 0},
        "cactus": {"count": 0},
    }
    rocks_assets = assets.get("rocks", [])
    stump_assets = assets.get("stumps", [])
    plants_assets = assets.get("plants", [])
    cactus_assets = assets.get("cactus", [])
    rock_stump_total_count = max(
        0,
        int(round(
            _scaled_count("rocks", diff_cfg["rock_stump_count"])
            * ROCK_STUMP_TOTAL_MULTIPLIER
            * mode_rock_stump_total_mul
        )),
    )
    rocks_count, stumps_count = _split_asset_count(
        rock_stump_total_count,
        rocks_assets,
        stump_assets,
        primary_ratio=(
            mode_rock_stump_primary_ratio
            if mode_rock_stump_primary_ratio is not None
            else ROCK_STUMP_PRIMARY_RATIO
        ),
    )
    plants_count, cactus_count = _split_asset_count(
        _scaled_count("plants", diff_cfg["ground_cover_count"]),
        plants_assets,
        cactus_assets,
        primary_ratio=GROUND_COVER_PLANT_PRIMARY_RATIO,
    )
    spawn_sections["rocks"]["count"] = rocks_count
    spawn_sections["stumps"]["count"] = stumps_count
    spawn_sections["plants"]["count"] = plants_count
    spawn_sections["cactus"]["count"] = cactus_count
    trunk_bounds = TRUNK_COUNT_BOUNDS_BY_DIFFICULTY.get(difficulty_id, {})
    logs_min = trunk_bounds.get("logs_min")
    if logs_min is not None:
        spawn_sections["logs"]["count"] = max(spawn_sections["logs"]["count"], logs_min)
    logs_max = trunk_bounds.get("logs_max")
    if logs_max is not None:
        spawn_sections["logs"]["count"] = min(spawn_sections["logs"]["count"], logs_max)
    stumps_min = trunk_bounds.get("stumps_min")
    if stumps_min is not None and spawn_sections["stumps"]["count"] < stumps_min:
        shift = min(
            stumps_min - spawn_sections["stumps"]["count"],
            spawn_sections["rocks"]["count"],
        )
        spawn_sections["stumps"]["count"] += shift
        spawn_sections["rocks"]["count"] -= shift
    stumps_max = trunk_bounds.get("stumps_max")
    if stumps_max is not None and spawn_sections["stumps"]["count"] > stumps_max:
        shift = spawn_sections["stumps"]["count"] - stumps_max
        spawn_sections["stumps"]["count"] -= shift
        spawn_sections["rocks"]["count"] += shift
    trees_count = spawn_sections["trees"]["count"]
    logs_count = spawn_sections["logs"]["count"]
    bushes_count = spawn_sections["bushes"]["count"]
    rocks_count = spawn_sections["rocks"]["count"]
    stumps_count = spawn_sections["stumps"]["count"]
    plants_count = spawn_sections["plants"]["count"]
    cactus_count = spawn_sections["cactus"]["count"]
    tree_assets = list(assets.get("trees", []))
    if ROCKS_ONLY_IN_ROCKS_SECTION:
        rocks_assets = [(cat, name) for cat, name in rocks_assets if name.startswith("Rock")]

    tree_eval_fixed_yaw = 0.0 if (TREE_RECT_DEBUG_DRAW or TREE_LAYOUT_MODE.strip().lower() == "row") else None

    tree_instances = _pick_tree_instances(
        rng,
        count=trees_count,
        assets=tree_assets,
        clearance_m=diff_cfg["tree_clearance_m"],
        difficulty_id=difficulty_id,
    )
    log_tree_occupied = _scaled_occupied_instances(
        tree_instances,
        radius_scale=(
            SMALL_ASSET_TREE_OCCUPANCY_SCALE["logs"]
            * mode_tree_occupancy_mul.get("logs", 1.0)
            * diff_trunk_occupancy_mul.get("logs", 1.0)
        ),
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["logs"],
    )
    bush_tree_occupied_scale = SMALL_ASSET_TREE_OCCUPANCY_SCALE["bushes"]
    bush_tree_occupied_cap = SMALL_ASSET_TREE_OCCUPANCY_CAP_M["bushes"]
    rock_tree_occupied = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["rocks"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["rocks"],
    )
    stump_tree_occupied = _scaled_occupied_instances(
        tree_instances,
        radius_scale=(
            SMALL_ASSET_TREE_OCCUPANCY_SCALE["stumps"]
            * mode_tree_occupancy_mul.get("stumps", 1.0)
            * diff_trunk_occupancy_mul.get("stumps", 1.0)
        ),
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["stumps"],
    )
    plants_tree_occupied = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["plants"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["plants"],
    )
    cactus_tree_occupied = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["cactus"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["cactus"],
    )
    tree_base_rects = _tree_base_rects_from_instances(tree_instances)
    tree_span_rects = _tree_span_rects_from_instances(tree_instances)
    protected_tree_span_rects = _protected_tree_span_rects_from_instances(tree_instances)
    log_instances = _pick_log_instances(
        rng,
        count=logs_count,
        assets=assets.get("logs", []),
        clearance_m=(
            diff_cfg["log_clearance_m"]
            * LOG_CLEARANCE_MULTIPLIER
            * diff_trunk_clearance_mul.get("logs", 1.0)
            * mode_clearance_mul.get("logs", 1.0)
        ),
        occupied_instances=log_tree_occupied,
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=protected_tree_span_rects,
    )
    if diff_trunk_fallback_fill.get("logs") and len(log_instances) < logs_count:
        extra_log_instances = _pick_log_instances(
            rng,
            count=(logs_count - len(log_instances)),
            assets=assets.get("logs", []),
            clearance_m=0.02,
            occupied_instances=log_instances,
            tree_base_rects=tree_base_rects,
            protected_tree_span_rects=None,
        )
        log_instances.extend(extra_log_instances)
    bush_instances = _pick_shrub_instances(
        rng,
        count=bushes_count,
        assets=assets.get("bushes", []),
        clearance_m=diff_cfg["bush_clearance_m"],
        tree_instances=tree_instances,
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=protected_tree_span_rects,
        occupied_instances=log_instances,
        tree_occupancy_scale=bush_tree_occupied_scale,
        tree_occupancy_cap_m=bush_tree_occupied_cap,
    )
    rock_instances = _pick_rock_stump_instances(
        rng,
        count=rocks_count,
        assets=rocks_assets,
        mode_id=mode_id,
        clearance_m=diff_cfg["rock_stump_clearance_m"],
        occupied_instances=(rock_tree_occupied + bush_instances + log_instances),
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=protected_tree_span_rects,
    )
    stump_instances = _pick_rock_stump_instances(
        rng,
        count=stumps_count,
        assets=stump_assets,
        mode_id=mode_id,
        clearance_m=(
            diff_cfg["rock_stump_clearance_m"]
            * STUMP_CLEARANCE_MULTIPLIER
            * diff_trunk_clearance_mul.get("stumps", 1.0)
            * mode_clearance_mul.get("stumps", 1.0)
        ),
        occupied_instances=(stump_tree_occupied + bush_instances + log_instances + rock_instances),
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=protected_tree_span_rects,
    )
    if diff_trunk_fallback_fill.get("stumps") and len(stump_instances) < stumps_count:
        fallback_stump_tree_occupied = _scaled_occupied_instances(
            tree_instances,
            radius_scale=max(
                SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
                SMALL_ASSET_TREE_OCCUPANCY_SCALE["stumps"] * 0.35,
            ),
            min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
            max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["stumps"] * 0.70,
        )
        extra_stump_instances = _pick_rock_stump_instances(
            rng,
            count=(stumps_count - len(stump_instances)),
            assets=stump_assets,
            mode_id=mode_id,
            clearance_m=max(0.08, diff_cfg["rock_stump_clearance_m"] * 0.55),
            occupied_instances=(fallback_stump_tree_occupied + bush_instances + log_instances + rock_instances + stump_instances),
            tree_base_rects=tree_base_rects,
            protected_tree_span_rects=None,
        )
        stump_instances.extend(extra_stump_instances)
    plant_instances = _pick_ground_cover_instances(
        rng,
        count=plants_count,
        assets=plants_assets,
        mode_id=mode_id,
        clearance_m=diff_cfg["ground_cover_clearance_m"],
        occupied_instances=(plants_tree_occupied + bush_instances + log_instances + rock_instances + stump_instances),
        tree_base_rects=tree_base_rects,
        tree_span_rects=tree_span_rects,
        protected_tree_span_rects=protected_tree_span_rects,
    )
    cactus_instances = _pick_ground_cover_instances(
        rng,
        count=cactus_count,
        assets=cactus_assets,
        mode_id=mode_id,
        clearance_m=diff_cfg["ground_cover_clearance_m"],
        occupied_instances=(cactus_tree_occupied + bush_instances + log_instances + rock_instances + stump_instances + plant_instances),
        tree_base_rects=tree_base_rects,
        tree_span_rects=tree_span_rects,
        protected_tree_span_rects=protected_tree_span_rects,
    )

    if TREE_BATCH_INSTANCE_MODE and tree_instances:
        trees_placed = _spawn_instances_as_single_multibody(
            instances=tree_instances,
            rng=rng,
            flags=flags,
            class_name="trees",
            enable_collision=not DISABLE_TREE_COLLISIONS_FOR_DEBUG,
            fixed_yaw_deg=tree_eval_fixed_yaw,
        )
    else:
        trees_placed = 0
        for x, y, category, obj_name, total_scale, _ in tree_instances:
            yaw_deg = tree_eval_fixed_yaw if tree_eval_fixed_yaw is not None else rng.uniform(-180.0, 180.0)
            if _spawn_asset_instance(
                category=category,
                obj_name=obj_name,
                x=x,
                y=y,
                yaw_deg=yaw_deg,
                scale=total_scale,
                flags=flags,
                enable_collision=not DISABLE_TREE_COLLISIONS_FOR_DEBUG,
                asset_class="trees",
            ):
                LAST_MAP_SPAWNED.append(("trees", category, obj_name))
                trees_placed += 1

    bushes_placed = _spawn_instances_as_single_multibody(
        instances=bush_instances,
        rng=rng,
        flags=flags,
        class_name="bushes",
        enable_collision=True,
    )

    rocks_placed = _spawn_instances_as_single_multibody(
        instances=rock_instances,
        rng=rng,
        flags=flags,
        class_name="rocks",
        enable_collision=True,
    )
    stumps_placed = _spawn_instances_as_single_multibody(
        instances=stump_instances,
        rng=rng,
        flags=flags,
        class_name="stumps",
        enable_collision=True,
    )

    logs_placed = _spawn_instances_as_single_multibody(
        instances=log_instances,
        rng=rng,
        flags=flags,
        class_name="logs",
        enable_collision=True,
    )

    plants_placed = _spawn_instances_as_single_multibody(
        instances=plant_instances,
        rng=rng,
        flags=flags,
        class_name="plants",
        enable_collision=True,
    )
    cactus_placed = _spawn_instances_as_single_multibody(
        instances=cactus_instances,
        rng=rng,
        flags=flags,
        class_name="cactus",
        enable_collision=True,
    )

    by_class = {
        "trees": trees_placed,
        "bushes": bushes_placed,
        "logs": logs_placed,
        "rocks": rocks_placed,
        "stumps": stumps_placed,
        "plants": plants_placed,
        "cactus": cactus_placed,
    }

    return {
        "map_name": MAP_NAME,
        "mode_name": mode_cfg["name"],
        "difficulty_name": diff_cfg["name"],
        "seed": seed,
        "placed_count": sum(by_class.values()),
        "class_counts": by_class,
    }


def _apply_visualizer_defaults() -> None:
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    if hasattr(p, "COV_ENABLE_PLANAR_REFLECTION"):
        p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 0)
    _disable_debug_previews()


def _apply_preview_safety() -> None:
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    _disable_debug_previews()


def _disable_debug_previews() -> None:
    if hasattr(p, "COV_ENABLE_TINY_RENDERER"):
        p.configureDebugVisualizer(p.COV_ENABLE_TINY_RENDERER, 0)
    if hasattr(p, "COV_ENABLE_SINGLE_STEP_RENDERING"):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 0)
    if hasattr(p, "COV_ENABLE_RGB_BUFFER_PREVIEW"):
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    if hasattr(p, "COV_ENABLE_DEPTH_BUFFER_PREVIEW"):
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    if hasattr(p, "COV_ENABLE_SEGMENTATION_MARK_PREVIEW"):
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)


def build_scene(*, mode_id: int, difficulty_id: int, seed: int, hills_enabled: bool = HILLS_RING_DEFAULT) -> dict[str, object]:
    global _OBJ_COLLISION_SHAPE_CACHE, _OBJ_VISUAL_SHAPE_CACHE, _GROUND_TEXTURE_ID, _GROUND_TEXTURE_ID_AUTUMN, _GROUND_TEXTURE_ID_DEAD
    global _HILLS_TEXTURE_ID, _HILLS_TEXTURE_ID_AUTUMN, _HILLS_TEXTURE_ID_DEAD
    global _LAST_BUILT_SCENE, _LAST_BUILT_RESULT

    # Skip the rebuild if this exact scene is already loaded.
    hills_enabled = bool(hills_enabled)
    cache_key = (mode_id, difficulty_id, seed, hills_enabled)
    if _LAST_BUILT_SCENE == cache_key and _LAST_BUILT_RESULT is not None:
        return _LAST_BUILT_RESULT

    p.resetSimulation()
    # IDs of collision/visual shapes and textures are not stable across reset.
    _OBJ_COLLISION_SHAPE_CACHE = {}
    _OBJ_VISUAL_SHAPE_CACHE = {}
    _GROUND_TEXTURE_ID = None
    _GROUND_TEXTURE_ID_AUTUMN = None
    _GROUND_TEXTURE_ID_DEAD = None
    _HILLS_TEXTURE_ID = None
    _HILLS_TEXTURE_ID_AUTUMN = None
    _HILLS_TEXTURE_ID_DEAD = None
    _apply_visualizer_defaults()
    p.setGravity(0, 0, -9.81)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    clamped_mode = _clamp_mode_id(mode_id)
    is_snow_mode = MAP_MODE_CONFIG[clamped_mode]["primary_category"] == "snow"
    if hills_enabled:
        # Hills mode owns the base terrain; don't spawn center 100x100 slab.
        _spawn_mode_hills(
            mode_id=clamped_mode,
            seed=seed,
            include_peaks=is_snow_mode,
        )
    elif is_snow_mode:
        _spawn_snow_base_ground()
    else:
        _spawn_ground(mode_id=clamped_mode)
    result = _spawn_forest_assets(
        mode_id=clamped_mode,
        difficulty_id=difficulty_id,
        seed=seed,
    )
    if isinstance(result, dict):
        result = dict(result)
        result["hills_enabled"] = hills_enabled
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    _LAST_BUILT_SCENE = cache_key
    _LAST_BUILT_RESULT = result
    return result


def _setup(use_gui: bool, use_opengl2: bool = False) -> None:
    global _GROUND_TEXTURE_ID, _GROUND_TEXTURE_ID_AUTUMN, _GROUND_TEXTURE_ID_DEAD, _OBJ_COLLISION_SHAPE_CACHE
    global _HILLS_TEXTURE_ID, _HILLS_TEXTURE_ID_AUTUMN, _HILLS_TEXTURE_ID_DEAD
    if p.isConnected():
        p.disconnect()
    if use_gui:
        if use_opengl2:
            p.connect(p.GUI, options="--opengl2")
        else:
            p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)
    _GROUND_TEXTURE_ID = None
    _GROUND_TEXTURE_ID_AUTUMN = None
    _GROUND_TEXTURE_ID_DEAD = None
    _HILLS_TEXTURE_ID = None
    _HILLS_TEXTURE_ID_AUTUMN = None
    _HILLS_TEXTURE_ID_DEAD = None
    _OBJ_COLLISION_SHAPE_CACHE = {}
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    _apply_visualizer_defaults()


def run(
    *,
    use_gui: bool = True,
    mode_id: int = 1,
    difficulty_id: int = 2,
    seed: int = 0,
    hills_enabled: bool = HILLS_RING_DEFAULT,
    use_opengl2: bool = False,
) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    _setup(use_gui=use_gui, use_opengl2=use_opengl2)

    if use_gui:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        _apply_visualizer_defaults()

    active_mode = _clamp_mode_id(mode_id)
    active_difficulty = _clamp_difficulty_id(difficulty_id)
    active_seed = max(0, int(seed))
    active_hills = bool(hills_enabled)

    regen_btn = None
    mode_slider = None
    difficulty_slider = None
    seed_slider = None
    hills_slider = None
    last_regen_value = 0.0
    if use_gui:
        regen_btn = p.addUserDebugParameter("[ REGENERATE ]", 1, 0, 0)
        mode_slider = p.addUserDebugParameter("Mode (1=Normal 2=Autumn 3=Snow 4=NoLeaves)", 1, 4, active_mode)
        difficulty_slider = p.addUserDebugParameter("Difficulty (1=Easy 2=Normal 3=Hard)", 1, 3, active_difficulty)
        seed_slider = p.addUserDebugParameter("Seed (0=random)", 0, 9999, active_seed)
        hills_slider = p.addUserDebugParameter("Hills Ring (0=OFF 1=ON)", 0, 1, 1 if active_hills else 0)
        active_mode = _clamp_mode_id(int(round(p.readUserDebugParameter(mode_slider))))
        active_difficulty = _clamp_difficulty_id(int(round(p.readUserDebugParameter(difficulty_slider))))
        active_seed = max(0, int(round(p.readUserDebugParameter(seed_slider))))
        active_hills = p.readUserDebugParameter(hills_slider) >= 0.5
        last_regen_value = p.readUserDebugParameter(regen_btn)

    init_seed = int(time.time() * 1000) % 1_000_000 if active_seed == 0 else active_seed
    t0 = time.perf_counter()
    scene_info = build_scene(
        mode_id=active_mode,
        difficulty_id=active_difficulty,
        seed=init_seed,
        hills_enabled=active_hills,
    )
    build_seconds = time.perf_counter() - t0

    def _print_terminal_summary(*, title: str, seconds: float, info: dict[str, object]) -> None:
        class_counts = info.get("class_counts", {}) if isinstance(info, dict) else {}
        mode_name = info.get("mode_name", "-") if isinstance(info, dict) else "-"
        difficulty_name = info.get("difficulty_name", "-") if isinstance(info, dict) else "-"
        seed_val = info.get("seed", "-") if isinstance(info, dict) else "-"
        total_val = info.get("placed_count", 0) if isinstance(info, dict) else 0

        icon = {
            "trees": "🌳",
            "bushes": "🌿",
            "logs": "🪵",
            "rocks": "🪨",
            "stumps": "🪵",
            "plants": "🍂",
            "cactus": "🌵",
        }
        label = {
            "trees": "Trees",
            "bushes": "Bushes",
            "logs": "Logs",
            "rocks": "Rocks",
            "stumps": "Stumps",
            "plants": "Plants",
            "cactus": "Cactus",
        }
        ordered_keys = ("trees", "bushes", "logs", "rocks", "stumps", "plants", "cactus")

        print("\n" + "═" * 64)
        print(f"🗺️  FOREST MAP | {title}")
        print("═" * 64)
        print(f"🎛️  Mode: {mode_name} | Difficulty: {difficulty_name}")
        print(f"🏞️  Hills Ring: {'ON' if bool(info.get('hills_enabled', HILLS_RING_DEFAULT)) else 'OFF'}")
        print(f"🎲 Seed: {seed_val} | ⏱️ Load: {seconds:.2f}s")
        print(f"📦 Spawned Total: {total_val}")
        print("📊 Spawn Breakdown:")
        if class_counts:
            for key in ordered_keys:
                if key in class_counts:
                    print(f"  {icon[key]} {label[key]:12s} -> {class_counts[key]}")
        else:
            print("  (no assets spawned)")
        print("═" * 64)

    _print_terminal_summary(title="READY", seconds=build_seconds, info=scene_info)

    if not use_gui:
        for _ in range(4):
            p.stepSimulation()
        p.disconnect()
        return

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    print("⌨️  Controls: R Rebuild | ESC Quit | 1 Wireframe | 2 Shadows")
    print("🎥 Camera: LMB Rotate | WASD Move | Q/E Up-Down | Shift Faster")

    cam = CameraController()
    wireframe_enabled = False
    shadows_enabled = False
    r_pressed = False
    wire_pressed = False
    shadows_pressed = False

    while p.isConnected():
        _apply_preview_safety()
        keys = p.getKeyboardEvents()
        cam.update(keys)

        if keys.get(27, 0) & p.KEY_WAS_TRIGGERED:
            break

        if use_gui and mode_slider is not None and difficulty_slider is not None and seed_slider is not None and hills_slider is not None and regen_btn is not None:
            regen_value = p.readUserDebugParameter(regen_btn)
            regenerate_by_button = regen_value > last_regen_value
            if regenerate_by_button:
                last_regen_value = regen_value
                active_mode = _clamp_mode_id(int(round(p.readUserDebugParameter(mode_slider))))
                active_difficulty = _clamp_difficulty_id(int(round(p.readUserDebugParameter(difficulty_slider))))
                active_seed = max(0, int(round(p.readUserDebugParameter(seed_slider))))
                active_hills = p.readUserDebugParameter(hills_slider) >= 0.5
                rebuild_seed = int(time.time() * 1000) % 1_000_000 if active_seed == 0 else active_seed
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                t1 = time.perf_counter()
                scene_info = build_scene(
                    mode_id=active_mode,
                    difficulty_id=active_difficulty,
                    seed=rebuild_seed,
                    hills_enabled=active_hills,
                )
                dt = time.perf_counter() - t1
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                _print_terminal_summary(title="REGENERATE", seconds=dt, info=scene_info)

        if keys.get(ord("r"), 0) == 1:
            if not r_pressed:
                r_pressed = True
                if use_gui and mode_slider is not None and difficulty_slider is not None and seed_slider is not None and hills_slider is not None:
                    active_mode = _clamp_mode_id(int(round(p.readUserDebugParameter(mode_slider))))
                    active_difficulty = _clamp_difficulty_id(int(round(p.readUserDebugParameter(difficulty_slider))))
                    active_seed = max(0, int(round(p.readUserDebugParameter(seed_slider))))
                    active_hills = p.readUserDebugParameter(hills_slider) >= 0.5
                rebuild_seed = int(time.time() * 1000) % 1_000_000 if active_seed == 0 else active_seed
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
                t1 = time.perf_counter()
                scene_info = build_scene(
                    mode_id=active_mode,
                    difficulty_id=active_difficulty,
                    seed=rebuild_seed,
                    hills_enabled=active_hills,
                )
                dt = time.perf_counter() - t1
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
                _print_terminal_summary(title="REGENERATE", seconds=dt, info=scene_info)
        else:
            r_pressed = False

        if keys.get(ord("1"), 0) == 1:
            if not wire_pressed:
                wire_pressed = True
                wireframe_enabled = not wireframe_enabled
                p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1 if wireframe_enabled else 0)
        else:
            wire_pressed = False

        if keys.get(ord("2"), 0) == 1:
            if not shadows_pressed:
                shadows_pressed = True
                shadows_enabled = not shadows_enabled
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if shadows_enabled else 0)
        else:
            shadows_pressed = False

        p.stepSimulation()
        time.sleep(1.0 / 60.0)

    if p.isConnected():
        p.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Forest map runner.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument("--opengl2", action="store_true", help="Use legacy OpenGL2 path in GUI (helps some color/render issues).")
    parser.add_argument("--mode", type=int, default=1, help="Mode (1=Normal, 2=Autumn, 3=Snow, 4=NoLeaves).")
    parser.add_argument("--difficulty", type=int, default=2, help="Difficulty (1-3).")
    parser.add_argument("--seed", type=int, default=0, help="Seed (0 uses time-based random seed).")
    parser.add_argument(
        "--hills",
        action=argparse.BooleanOptionalAction,
        default=HILLS_RING_DEFAULT,
        help="Toggle outer hills ring (mountain peaks stay snow-only).",
    )
    args = parser.parse_args()
    run(
        use_gui=not args.headless,
        mode_id=args.mode,
        difficulty_id=args.difficulty,
        seed=args.seed,
        hills_enabled=bool(args.hills),
        use_opengl2=args.opengl2,
    )


if __name__ == "__main__":
    main()
