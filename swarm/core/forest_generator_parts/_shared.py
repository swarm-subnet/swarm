"""Shared constants, caches, and imports for forest generation."""

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


def _reset_client_caches(cli: int) -> None:
    _CLI_COL_CACHE[cli] = {}
    _CLI_VIS_CACHE[cli] = {}
    _CLI_TEX_CACHE.pop(cli, None)


__all__ = [name for name in globals() if not name.startswith("__")]
