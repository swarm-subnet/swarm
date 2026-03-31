"""
Embedded factory conveyor belt network for Type 5 warehouse maps.
Generates a seeded conveyor path with belt segments, supports, drones,
workers, cartons, and an optional barrier fence ring.
"""

import math
import os
import random

import pybullet as p

from ..constants import (
    EMBEDDED_FACTORY_SEED_OFFSET,
    ENABLE_EMBEDDED_FACTORY_MAP,
    ENABLE_FACTORY_BARRIER_RING,
    FACTORY_BARRIER_DOUBLE_SIDED,
    FACTORY_BARRIER_FLAT_RGBA,
    FACTORY_BARRIER_INSET_M,
    FACTORY_BARRIER_MODEL_PATH,
    FACTORY_BARRIER_SCALE_XYZ,
    FACTORY_BARRIER_SEGMENT_GAP_M,
    FACTORY_BARRIER_WITH_COLLISION,
    OTHER_SOURCES_DIR,
    SHOW_AREA_LAYOUT_MARKERS,
    UNIFORM_SPECULAR_COLOR,
)
from ..helpers import (
    _obj_collision_proxy_path,
    _spawn_collision_only_with_anchor,
    _spawn_mesh_with_anchor,
    model_bounds_xyz,
)
from ..shared import MeshKitLoader, first_existing_path

# ---------------------------------------------------------------------------
# Factory geometry constants
# ---------------------------------------------------------------------------
FACTORY_SIZE_X = 40.0
FACTORY_SIZE_Y = 20.0
FLOOR_HALF_Z = 0.03
CONVEYOR_SCALE = 0.95
CONVEYOR_ELEVATION_M = 0.62
EDGE_MARGIN_M = 3.0
ROW_MARGIN_M = 1.4

# ---------------------------------------------------------------------------
# Conveyor support models
# ---------------------------------------------------------------------------
SUPPORT_MODEL_CANDIDATES = (
    "structure-medium.obj",
    "structure-short.obj",
    "structure-high.obj",
    "structure-tall.obj",
)
SUPPORT_SPACING_M = 1.9

# ---------------------------------------------------------------------------
# Belt models
# ---------------------------------------------------------------------------
NETWORK_BELT_MODEL = "conveyor-stripe-sides.obj"
NETWORK_BELT_MODEL_CANDIDATES = ("conveyor-stripe-sides.obj",)
BELT_COMPAT_LEN_RATIO_MIN = 0.96
BELT_COMPAT_LEN_RATIO_MAX = 1.04
BELT_COMPAT_WID_RATIO_MIN = 0.96
BELT_COMPAT_WID_RATIO_MAX = 1.04
ALLOW_MIXED_SECTION_BELTS = False

# ---------------------------------------------------------------------------
# End caps / section decor
# ---------------------------------------------------------------------------
END_CAP_MODEL_CANDIDATES = (
    "cover-stripe-top.obj",
    "cover-top.obj",
    "cover-stripe.obj",
    "cover.obj",
)
END_CAP_OUTWARD_OFFSET_CELLS = 0.55
SECTION_DECOR_ENABLE = True
SECTION_SPLIT_RATIO_MIN = 0.62
SECTION_SPLIT_RATIO_MAX = 0.72
SECTION_SPLIT_MARGIN_CELLS = 18
SECTION_DIVIDER_MODEL_CANDIDATES = (
    "cover-stripe-hopper.obj",
    "cover-stripe-top.obj",
    "cover-top.obj",
)

# ---------------------------------------------------------------------------
# Assembly drones / workers / cartons
# ---------------------------------------------------------------------------
ASSEMBLY_DRONE_MODEL_CANDIDATES = (
    "scanner-low.obj",
    "scanner-high.obj",
    "robot-arm-a.obj",
)
ASSEMBLY_DRONE_INTERVAL_MIN = 12
ASSEMBLY_DRONE_INTERVAL_MAX = 18
ASSEMBLY_DRONE_Z_OFFSET_M = 0.0
ASSEMBLY_WORKER_MODEL_CANDIDATES = (
    "robot-arm-b-angled.obj",
    "robot-arm-b.obj",
)
ASSEMBLY_WORKER_SIDE_OFFSET_CELLS_MIN = 0.95
ASSEMBLY_WORKER_SIDE_OFFSET_CELLS_MAX = 1.45
ASSEMBLY_WORKER_MIN_CLEARANCE_CELLS = 1.10
ASSEMBLY_WORKER_MIN_SEPARATION_CELLS = 2.20
ASSEMBLY_WORKER_INTERVAL_CELLS = 11
ASSEMBLY_WORKER_MODEL_YAW_FIX_BY_MODEL = {
    "robot-arm-b-angled.obj": 90.0,
    "robot-arm-b.obj": 0.0,
}
ASSEMBLY_WORKER_MODEL_YAW_FIX_DEFAULT_DEG = 0.0
PACKOUT_BOX_MODEL_CANDIDATES = ("box-small.obj",)
PACKOUT_BOX_INTERVAL_MIN = 7
PACKOUT_BOX_INTERVAL_MAX = 12
PACKOUT_BOX_Z_OFFSET_M = 0.01
PACKOUT_BOX_SIDE_OFFSET_CELLS = 0.0
PACKOUT_BOX_SCALE_XY_MULT = 1.10
PACKOUT_BOX_SCALE_Z_MULT = 1.00

# ---------------------------------------------------------------------------
# Swarm drone URDF
# ---------------------------------------------------------------------------
_SWARM_DRONE_DIR = os.path.join(OTHER_SOURCES_DIR, "swarm_drone")
SWARM_DRONE_URDF_CANDIDATES = (os.path.join(_SWARM_DRONE_DIR, "swarm_drone.urdf"),)
SWARM_DRONE_GLOBAL_SCALE = 3.5
SWARM_DRONE_SCALE_MULT = 1.38

# ---------------------------------------------------------------------------
# Path generation tuning
# ---------------------------------------------------------------------------
PATH_MIN_SEG_CELLS = 2
PATH_MAX_SEG_CELLS = 8
PATH_MIN_TURNS = 10
PATH_MIN_CELLS = 95
PATH_BUILD_ATTEMPTS = 320
PATH_MIN_SPAN_X_RATIO = 0.70
PATH_MIN_SPAN_Y_RATIO = 0.65
PATH_MAX_CELLS_HARD = 220
END_TARGET_OFFSET_CELLS = 2
PATH_HALF_MIN_RATIO = 0.30
PATH_QUADRANT_MIN_RATIO = 0.06
PATH_OCCUPANCY_BINS_X = 6
PATH_OCCUPANCY_BINS_Y = 3
PATH_FALLBACK_EMPTY_BINS_MAX = 3
LANE_EMPTY_GAP_CHOICES = (2, 2, 3, 3, 4)
TOP_PHASE_EDGE_INSET_MAX_CELLS = 0
VERTICAL_PHASE_EXTRA_SKIP_CHANCE = 0.30

FLOOR_RGBA = (0.64, 0.67, 0.75, 1.0)

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_SWARM_URDF_SEARCH_PATHS = {}
_SWARM_URDF_BOTTOM_Z_OFFSET_CACHE = {}
_FACTORY_BARRIER_LOADER_CACHE = {}


# ---------------------------------------------------------------------------

__all__ = [name for name in globals() if not name.startswith("__")]
