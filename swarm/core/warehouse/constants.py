"""
Warehouse map constants for Type 4 challenge maps.
All tuning parameters, feature flags, model names, and asset paths.
"""

import math
import os

# ---------------------------------------------------------------------------
# Asset paths (relative to this file → swarm/assets/maps/)
# ---------------------------------------------------------------------------
ASSETS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "assets", "maps")
)
KENNEY_DIR = os.path.join(ASSETS_DIR, "kenney")
CUSTOM_DIR = os.path.join(ASSETS_DIR, "custom")
OTHER_SOURCES_DIR = os.path.join(ASSETS_DIR, "other_sources")

CONVEYOR_KIT_OBJ_DIR = os.path.join(KENNEY_DIR, "kenney_conveyor-kit", "Models", "OBJ format")
CONVEYOR_KIT_TEXTURE = os.path.join(CONVEYOR_KIT_OBJ_DIR, "Textures", "colormap.png")
FURNITURE_KIT_OBJ_DIR = os.path.join(KENNEY_DIR, "kenney_furniture-kit", "Models", "OBJ format")
FURNITURE_KIT_TEXTURE = os.path.join(FURNITURE_KIT_OBJ_DIR, "Textures", "colormap.png")

VEHICLE_DIR = os.path.join(OTHER_SOURCES_DIR, "vehicles")
LOADING_KIT_DIR = os.path.join(OTHER_SOURCES_DIR, "loading_kit")
CRANE_DIR = os.path.join(OTHER_SOURCES_DIR, "overhead_crane")
FENCE_DIR = os.path.join(OTHER_SOURCES_DIR, "factory_fence_new")
WAREHOUSE_SHELL_DIR = os.path.join(CUSTOM_DIR, "warehouse_shell")

# ---------------------------------------------------------------------------
# Warehouse geometry
# ---------------------------------------------------------------------------
WAREHOUSE_BASE_SIZE_X = 104.0
WAREHOUSE_BASE_SIZE_Y = 72.0
WAREHOUSE_SHELL_SHRINK_RATIO = 0.925
WAREHOUSE_SIZE_X = WAREHOUSE_BASE_SIZE_X * WAREHOUSE_SHELL_SHRINK_RATIO
WAREHOUSE_SIZE_Y = WAREHOUSE_BASE_SIZE_Y * WAREHOUSE_SHELL_SHRINK_RATIO
HALF_X = WAREHOUSE_SIZE_X * 0.5
HALF_Y = WAREHOUSE_SIZE_Y * 0.5
UNIFORM_SCALE = 4.0

# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
WALL_TIERS = 1
CURVED_ROOF_RISE = 3.2
ENABLE_CORNER_COLUMNS = False

WALL_UNIFORM_COLOR = (0.60, 0.64, 0.72, 1.0)
ROOF_UNIFORM_COLOR = (0.66, 0.69, 0.77, 1.0)
FLOOR_UNIFORM_COLOR = (0.66, 0.69, 0.77, 1.0)
DOCK_INWARD_NUDGE = 0.00
FLOOR_INNER_MARGIN_TILES = 1
FLOOR_SPAWN_SAFETY_MARGIN_M = 0.00
ENABLE_ROOF_TRUSS_SYSTEM = True
TRUSS_UNIFORM_COLOR = (0.30, 0.33, 0.40, 1.0)
TRUSS_WITH_COLLISION = True
SHOW_AREA_LAYOUT_MARKERS = False

# ---------------------------------------------------------------------------
# Shell mesh files
# ---------------------------------------------------------------------------
WAREHOUSE_SHELL_FILES = {
    "roof": "roof_curved_104x72.obj",
    "fillers": "roof_fillers_104x72.obj",
    "truss": "roof_truss_104x72.obj",
}

# ---------------------------------------------------------------------------
# Conveyor kit model names
# ---------------------------------------------------------------------------
CONVEYOR_ASSETS = {
    "floor": "top-large.obj",
    "wall": "structure-wall.obj",
    "wall_window": "structure-window.obj",
    "wall_window_wide": "structure-window-wide.obj",
    "wall_corner": "structure-corner-inner.obj",
    "column": "structure-tall.obj",
    "dock_frame": "structure-doorway-wide.obj",
    "dock_door_closed": "door-wide-closed.obj",
    "dock_door_half": "door-wide-half.obj",
    "dock_door_open": "door-wide-open.obj",
    "personnel_frame": "structure-doorway.obj",
    "personnel_door": "door.obj",
}

ENABLE_PERSONNEL_FLOOR_LANE = True
PERSONNEL_FLOOR_LANE_MODEL_CANDIDATES = ("floor.obj", "floor-large.obj")
PERSONNEL_FLOOR_LANE_Z_OFFSET = 0.004
PERSONNEL_FLOOR_LANE_EDGE_TOLERANCE_TILES = 6.0

# ---------------------------------------------------------------------------
# Area layout zones
# ---------------------------------------------------------------------------
AREA_LAYOUT_BLOCKS = (
    {"name": "OFFICE", "size_m": (11.0, 11.0), "corner": "sw", "rgba": (0.96, 0.66, 0.83, 0.72)},
    {"name": "LOADING", "size_m": (22.5, 42.5), "corner": "se", "rgba": (0.99, 0.73, 0.45, 0.72)},
    {"name": "FORKLIFT_PARK", "size_m": (14.0, 5.0), "corner": "ne", "rgba": (0.84, 0.92, 0.66, 0.72)},
    {"name": "MACHINING_CELL", "size_m": (12.0, 8.0), "corner": "nw", "rgba": (0.98, 0.90, 0.55, 0.72)},
    {"name": "STORAGE", "size_m": (22.5, 45.0), "corner": "nw", "rgba": (0.53, 0.91, 0.67, 0.72)},
    {"name": "FACTORY", "size_m": (22.5, 50.0), "corner": "ne", "rgba": (0.58, 0.77, 0.99, 0.72)},
)

AREA_LAYOUT_EDGE_MARGIN = 0.0
AREA_LAYOUT_TILE_HALF_Z = 0.01
AREA_LAYOUT_MIN_GAP = 0.8
AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR = 0.5
PERSONNEL_DOOR_CLEAR_DEPTH = 6.0
PERSONNEL_DOOR_CLEAR_EXTRA_ALONG = 1.5

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------
ENABLE_EMBEDDED_OFFICE_MAP = True
EMBEDDED_OFFICE_SEED_OFFSET = 313
ENABLE_EMBEDDED_FACTORY_MAP = True
EMBEDDED_FACTORY_SEED_OFFSET = 1701
ENABLE_LOADING_TRUCKS = True
ENABLE_LOADING_STAGING = True
ENABLE_STORAGE_RACK_LAYOUT = True
ENABLE_FORKLIFT_PARKING = True
ENABLE_LOADING_OPERATION_FORKLIFTS = True
ENABLE_WORKER_CREW = True
ENABLE_OVERHEAD_CRANES = True
ENABLE_MACHINING_CELL_LAYOUT = False
ENABLE_FACTORY_BARRIER_RING = True

# ---------------------------------------------------------------------------
# Loading trucks
# ---------------------------------------------------------------------------
LOADING_TRUCK_MODELS = ("oppen door truck.obj",)
LOADING_TRUCK_SCALE_UNIFORM = 0.82
LOADING_TRUCK_SCALE_XYZ = (
    LOADING_TRUCK_SCALE_UNIFORM,
    LOADING_TRUCK_SCALE_UNIFORM,
    LOADING_TRUCK_SCALE_UNIFORM,
)
LOADING_TRUCK_WALL_GAP = 0.10
LOADING_TRUCK_EXTRA_GAP_HALF = 2.4
LOADING_TRUCK_EXTRA_GAP_CLOSED = 2.8
LOADING_INTER_GATE_WALL_STEPS = 0.5
LOADING_DOOR_CENTER_STEP_FRACTION = 0.5

# ---------------------------------------------------------------------------
# Loading staging
# ---------------------------------------------------------------------------
LOADING_STAGING_MODELS = {
    "pallet": "loading_pallet.obj",
    "box": "loading_box.obj",
    "barrel": "loading_barrel.obj",
}
LOADING_STAGING_SCALES = {
    "pallet": (1.00, 1.00, 1.00),
    "box": (0.48, 0.48, 0.48),
    "barrel": (0.75, 0.75, 0.75),
}
LOADING_STAGING_EDGE_MARGIN_M = 1.0
LOADING_STAGING_MAX_DEPTH_M = 10.0
LOADING_STAGING_TRUCK_TAIL_CLEARANCE_M = 0.55
LOADING_STAGING_PROP_GAP_M = 0.35
LOADING_STAGING_SUPPORT_BACK_BIAS = 0.82
LOADING_STAGING_GOODS_BACK_EDGE_PAD_M = 0.06
LOADING_BARREL_MAX_STACK_LAYERS = 2
LOADING_CONTAINER_STACK_ENABLED = True
LOADING_CONTAINER_MODEL_NAME = "loading_container.obj"
LOADING_CONTAINER_SCALE_XYZ = (0.84, 0.84, 0.84)
LOADING_CONTAINER_STACK_VERTICAL_GAP_M = 0.05
LOADING_SECTION_MIN_SPAN_M = 8.0
LOADING_BUNDLES_PER_TRUCK_MIN = 4
LOADING_BUNDLES_PER_TRUCK_MAX = 6
LOADING_LOADED_PALLET_STACK_MIN_LAYERS = 1
LOADING_LOADED_PALLET_STACK_MAX_LAYERS = 3
LOADING_EMPTY_PALLET_STACK_COUNT = 7
LOADING_EMPTY_PALLET_STACK_MIN_LAYERS = 5
LOADING_EMPTY_PALLET_STACK_MAX_LAYERS = 10

# ---------------------------------------------------------------------------
# Storage racks
# ---------------------------------------------------------------------------
STORAGE_RACK_MODEL_NAME = "storage_rack_empty.obj"
STORAGE_RACK_SCALE_UNIFORM = 1.00
STORAGE_RACK_EDGE_MARGIN_M = 0.35
STORAGE_RACK_ROW_GAP_M = 1.4
STORAGE_RACK_SLOT_GAP_M = 0.45
STORAGE_RACK_MAIN_AISLE_M = 2.6
STORAGE_RACK_MAX_COUNT = 0
STORAGE_RACK_RGBA = (0.64, 0.67, 0.72, 1.0)
STORAGE_RACK_BARREL_RACK_PROBABILITY = 0.196
STORAGE_RACK_BARREL_LAYER2_PROBABILITY = 0.154
STORAGE_RACK_TARGET_ROW_COUNT = 4
STORAGE_RACK_PALLET_LEVELS = 3
STORAGE_RACK_PALLETS_PER_LEVEL = 2
STORAGE_RACK_LEVEL_MIN_CLEAR_M = 0.10
STORAGE_RACK_LEVELS_RATIO = (0.00, 0.47, 0.69)
STORAGE_RACK_LEVEL_CONTACT_SNAP_M = 0.015
STORAGE_RACK_PALLET_INSET_X_RATIO = 0.23
STORAGE_RACK_PALLET_INSET_Y_RATIO = 0.0
STORAGE_RACK_BOX_PROBABILITY = 0.92
STORAGE_RACK_BOX_LAYER2_PROBABILITY = 0.42
STORAGE_RACK_CENTER_AISLE_TARGET_M = 2.0
STORAGE_RACK_ENABLE_ENDCAP_ROWS = True
STORAGE_RACK_NO_TOP_LEVEL_PROBABILITY = 0.25
STORAGE_RACK_LAYOUT_FIXED_SEED = None
STORAGE_RACK_CARGO_FIXED_SEED = 0
STORAGE_RACK_LEVEL_DENSITY = (1.00, 0.82, 0.62)
STORAGE_RACK_GLOBAL_YAW_OFFSET_DEG = 0.0
STORAGE_RACK_FORCE_ALONG_AXIS = "auto"
STORAGE_RACK_GROUP_ROTATE_DEG = 0.0
STORAGE_RACK_LAYOUT_SHIFT_MAX_M = 1.2

# ---------------------------------------------------------------------------
# Forklifts
# ---------------------------------------------------------------------------
FORKLIFT_MODEL_NAME = "forklift.obj"
FORKLIFT_TEXTURE_NAME = ""
FORKLIFT_SCALE_UNIFORM = 0.78
FORKLIFT_PARK_SLOT_COUNT = 4
FORKLIFT_PARK_SPAWN_MIN = 3
FORKLIFT_PARK_SPAWN_MAX = 4
FORKLIFT_PARK_GAP_M = 1.0
FORKLIFT_AREA_PREFERENCE = ("FORKLIFT_PARK",)
FORKLIFT_WALL_BACK_CLEARANCE = 0.35
FORKLIFT_PARK_YAW_EXTRA_DEG = 180.0
ENABLE_FORKLIFT_PARK_SLOT_LINES = True
FORKLIFT_PARK_LINE_RGBA = (0.84, 0.76, 0.20, 1.0)
FORKLIFT_PARK_LINE_WIDTH_M = 0.07
FORKLIFT_PARK_LINE_HEIGHT_M = 0.008
FORKLIFT_PARK_LINE_CENTER_Z = 0.033
FORKLIFT_PARK_SLOT_ALONG_PAD_M = 0.45
FORKLIFT_PARK_SLOT_CROSS_PAD_M = 0.55
LOADING_OPERATION_FORKLIFT_TARGET_COUNT = 3
LOADING_OPERATION_FORKLIFT_TRUCK_OFFSET_M = 2.0
LOADING_OPERATION_FORKLIFT_EMPTY_OFFSET_M = 1.2
LOADING_OPERATION_TRUCK_KEEPOUT_ALONG_PAD_M = 1.10
LOADING_OPERATION_TRUCK_KEEPOUT_CROSS_PAD_M = 1.35

# ---------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------
WORKER_MODEL_CANDIDATES = ("worker.obj", "Worker.obj")
WORKER_TARGET_COUNT = 6
WORKER_TARGET_HEIGHT_M = 1.72
WORKER_MIN_SPACING_M = 1.7
WORKER_COLOR_GAIN = 1.12

# ---------------------------------------------------------------------------
# Overhead cranes
# ---------------------------------------------------------------------------
OVERHEAD_CRANE_MODEL_CANDIDATES = ("Crane.obj", "crane.obj")
OVERHEAD_CRANE_SCALE_UNIFORM = 0.20
OVERHEAD_CRANE_TARGET_BY_ZONE = (("FACTORY", 2), ("LOADING", 1))
OVERHEAD_CRANE_ZONE_EDGE_MARGIN_M = 1.4
OVERHEAD_CRANE_MIN_SPACING_M = 7.5
OVERHEAD_CRANE_ATTACH_CLEARANCE_M = 0.0
OVERHEAD_CRANE_WITH_COLLISION = True
OVERHEAD_CRANE_COLOR_GAIN = 1.02
OVERHEAD_CRANE_YAW_EXTRA_DEG = 90.0
OVERHEAD_CRANE_TRUSS_TOUCH_EXTRA_M = 0.01

# ---------------------------------------------------------------------------
# Machining cell
# ---------------------------------------------------------------------------
MACHINING_CELL_AREA_NAME = "MACHINING_CELL"
MACHINING_MILL_MODEL_NAME = "mill.obj"
MACHINING_MILL_SCALE_UNIFORM = 0.45
MACHINING_LATHE_MODEL_NAME = "lathe.obj"
MACHINING_LATHE_SCALE_UNIFORM = 0.42
MACHINING_HEAVY_EXTRA_YAW_DEG = 180.0
MACHINING_SLOT_TYPES = ("LATHE", "DRILL", "MILL", "LATHE", "DRILL", "MILL")
MACHINING_AISLE_WIDTH = 1.6
MACHINING_EDGE_MARGIN = 0.9
MACHINING_PENDING_SLOT_SIZE = (1.8, 1.3, 0.04)
MACHINING_PENDING_RGBA = (0.94, 0.78, 0.38, 0.62)
MACHINING_TABLE_SIZE = (2.2, 0.9, 0.9)
MACHINING_TABLE_RGBA = (0.56, 0.36, 0.20, 1.0)
MACHINING_SHOW_PENDING_MARKERS = True
MACHINING_FORCE_SIMPLE_VISUALS = False
MACHINING_SIMPLE_MILL_RGBA = (0.30, 0.70, 0.40, 1.0)
MACHINING_SIMPLE_LATHE_RGBA = (0.62, 0.68, 0.64, 1.0)
MACHINING_USE_NATIVE_MTL_VISUALS = False
MACHINING_VISUAL_DOUBLE_SIDED = True
MACHINING_USE_PART_TEXTURES = False
MACHINING_FORCE_REFRESH_MTL_PROXY = False

# ---------------------------------------------------------------------------
# Factory barrier fence
# ---------------------------------------------------------------------------
FACTORY_BARRIER_MODEL_PATH = os.path.join(FENCE_DIR, "fence.obj")
FACTORY_BARRIER_SCALE_UNIFORM = 1.30
FACTORY_BARRIER_SCALE_XYZ = (
    FACTORY_BARRIER_SCALE_UNIFORM,
    FACTORY_BARRIER_SCALE_UNIFORM,
    FACTORY_BARRIER_SCALE_UNIFORM,
)
FACTORY_BARRIER_INSET_M = 0.30
FACTORY_BARRIER_SEGMENT_GAP_M = 0.18
FACTORY_BARRIER_WITH_COLLISION = True
FACTORY_BARRIER_DOUBLE_SIDED = True
FACTORY_BARRIER_FLAT_RGBA = (0.78, 0.80, 0.84, 1.0)

# ---------------------------------------------------------------------------
# Wall slot names
# ---------------------------------------------------------------------------
WALL_SLOTS = ("north", "east", "south", "west")
LOADING_SLOTS = WALL_SLOTS

# ---------------------------------------------------------------------------
# Mesh rendering
# ---------------------------------------------------------------------------
MESH_UP_FIX_RPY = (math.pi / 2.0, 0.0, 0.0)
UNIFORM_SPECULAR_COLOR = (0.0, 0.0, 0.0)
