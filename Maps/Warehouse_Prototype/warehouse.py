import argparse
import importlib.util
import itertools
import json
import math
import os
import random
import shutil
import sys
import time

import pybullet as p
import pybullet_data

from shared import (
    UNIFORM_SPECULAR_COLOR,
    CameraController,
    MeshKitLoader,
    first_existing_path,
    normalize_mtl_texture_paths,
)


                                                                              
        
                                                                              
WAREHOUSE_BASE_SIZE_X = 104.0
WAREHOUSE_BASE_SIZE_Y = 72.0
WAREHOUSE_SHELL_SHRINK_RATIO = 0.925                
WAREHOUSE_SIZE_X = WAREHOUSE_BASE_SIZE_X * WAREHOUSE_SHELL_SHRINK_RATIO
WAREHOUSE_SIZE_Y = WAREHOUSE_BASE_SIZE_Y * WAREHOUSE_SHELL_SHRINK_RATIO
UNIFORM_SCALE = 4.0
WALL_TIERS = 1
CURVED_ROOF_RISE = 3.2
ENABLE_CORNER_COLUMNS = False

WALL_UNIFORM_COLOR = (0.60, 0.64, 0.72, 1.0)
ROOF_UNIFORM_COLOR = (0.66, 0.69, 0.77, 1.0)
FLOOR_UNIFORM_COLOR = (0.66, 0.69, 0.77, 1.0)
SHADOWS_DEFAULT = False
TURBO_BUILD_MODE_DEFAULT = True
DOCK_INWARD_NUDGE = 0.00
FLOOR_INNER_MARGIN_TILES = 1
FLOOR_SPAWN_SAFETY_MARGIN_M = 0.00
ENABLE_ROOF_TRUSS_SYSTEM = True
TRUSS_UNIFORM_COLOR = (0.30, 0.33, 0.40, 1.0)
TRUSS_WITH_COLLISION = True
SHOW_AREA_LAYOUT_MARKERS = False
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WAREHOUSE_SHELL_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "..", "assets", "custom", "warehouse_shell"),
)
WAREHOUSE_SHELL_FILES = {
    "roof": "roof_curved_104x72.obj",
    "fillers": "roof_fillers_104x72.obj",
    "truss": "roof_truss_104x72.obj",
}

CONVEYOR_KIT_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "..", "assets", "kenney", "kenney_conveyor-kit"),
)
RGS_TRUCK_ASSET_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "..", "assets", "other_sources", "vehicles"),
)
FORKLIFT_ASSET_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "..", "assets", "other_sources", "vehicles"),
)
LOADING_STAGING_ASSET_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "..", "assets", "other_sources", "loading_kit"),
)


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
PERSONNEL_FLOOR_LANE_MODEL_CANDIDATES = (
    "floor.obj",
    "floor-large.obj",
)
PERSONNEL_FLOOR_LANE_Z_OFFSET = 0.004
PERSONNEL_FLOOR_LANE_EDGE_TOLERANCE_TILES = 6.0


AREA_LAYOUT_BLOCKS = (
    {"name": "OFFICE", "size_m": (11.0, 11.0), "corner": "sw", "rgba": (0.96, 0.66, 0.83, 0.72)},
    {
        "name": "LOADING",
        "size_m": (22.5, 42.5),                                                 
        "corner": "se",
        "rgba": (0.99, 0.73, 0.45, 0.72),
    },
    {
        "name": "FORKLIFT_PARK",
        "size_m": (14.0, 5.0),                                                   
        "corner": "ne",
        "rgba": (0.84, 0.92, 0.66, 0.72),
    },
    {
        "name": "MACHINING_CELL",
        "size_m": (12.0, 8.0),                                              
        "corner": "nw",
        "rgba": (0.98, 0.90, 0.55, 0.72),
    },
    {"name": "STORAGE", "size_m": (22.5, 45.0), "corner": "nw", "rgba": (0.53, 0.91, 0.67, 0.72)},
    {"name": "FACTORY", "size_m": (22.5, 50.0), "corner": "ne", "rgba": (0.58, 0.77, 0.99, 0.72)},
)
AREA_LAYOUT_EDGE_MARGIN = 0.0
AREA_LAYOUT_TILE_HALF_Z = 0.01
AREA_LAYOUT_MIN_GAP = 0.8
AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR = 0.5
PERSONNEL_DOOR_CLEAR_DEPTH = 6.0
PERSONNEL_DOOR_CLEAR_EXTRA_ALONG = 1.5
ENABLE_EMBEDDED_OFFICE_MAP = True
EMBEDDED_OFFICE_SEED_OFFSET = 313
ENABLE_EMBEDDED_FACTORY_MAP = True
EMBEDDED_FACTORY_SEED_OFFSET = 1701
ENABLE_LOADING_TRUCKS = True
LOADING_TRUCK_MODELS = (
    "oppen door truck.obj",
)
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
ENABLE_LOADING_STAGING = True
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
ENABLE_STORAGE_RACK_LAYOUT = True
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
ENABLE_FORKLIFT_PARKING = True
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
ENABLE_LOADING_OPERATION_FORKLIFTS = True
LOADING_OPERATION_FORKLIFT_TARGET_COUNT = 3
LOADING_OPERATION_FORKLIFT_TRUCK_OFFSET_M = 2.0
LOADING_OPERATION_FORKLIFT_EMPTY_OFFSET_M = 1.2
LOADING_OPERATION_TRUCK_KEEPOUT_ALONG_PAD_M = 1.10
LOADING_OPERATION_TRUCK_KEEPOUT_CROSS_PAD_M = 1.35
ENABLE_WORKER_CREW = True
WORKER_TARGET_COUNT = 6
WORKER_ASSET_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "..", "assets", "workers"),
    os.path.join(SCRIPT_DIR, "..", "assets", "other_sources", "loading_kit"),
    os.path.join(SCRIPT_DIR, "..", "assets", "other_sources", "vehicles"),
)
WORKER_MODEL_CANDIDATES = (
    "worker.obj",
    "Worker.obj",
)
WORKER_TARGET_HEIGHT_M = 1.72
WORKER_MIN_SPACING_M = 1.7
WORKER_COLOR_GAIN = 1.12
ENABLE_OVERHEAD_CRANES = True
OVERHEAD_CRANE_ASSET_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "..", "assets", "other_sources", "overhead_crane"),
)
OVERHEAD_CRANE_MODEL_CANDIDATES = (
    "Crane.obj",
    "crane.obj",
)
OVERHEAD_CRANE_SCALE_UNIFORM = 0.20
OVERHEAD_CRANE_TARGET_BY_ZONE = (
    ("FACTORY", 2),
    ("LOADING", 1),
)
OVERHEAD_CRANE_ZONE_EDGE_MARGIN_M = 1.4
OVERHEAD_CRANE_MIN_SPACING_M = 7.5
OVERHEAD_CRANE_ATTACH_CLEARANCE_M = 0.0
OVERHEAD_CRANE_WITH_COLLISION = True
OVERHEAD_CRANE_COLOR_GAIN = 1.02
OVERHEAD_CRANE_YAW_EXTRA_DEG = 90.0
OVERHEAD_CRANE_TRUSS_TOUCH_EXTRA_M = 0.01
ENABLE_MACHINING_CELL_LAYOUT = False
MACHINING_CELL_AREA_NAME = "MACHINING_CELL"
MACHINING_MILL_MODEL_NAME = "mill .obj"
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

WALL_SLOTS = ("north", "east", "south", "west")
LOADING_SLOTS = WALL_SLOTS

ENABLE_FACTORY_BARRIER_RING = True
FACTORY_BARRIER_MODEL_CANDIDATES = (
    os.path.join(SCRIPT_DIR, "..", "assets", "other_sources", "factory_fence_new", "fence.obj"),
)
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


                                                                              
               
                                                                              


def _resolve_optional_model(candidates, model_names):
    for root in candidates:
        if not root or not os.path.exists(root):
            continue
        root_abs = os.path.abspath(root)
        for model_name in model_names:
            pth = os.path.join(root_abs, model_name)
            if os.path.exists(pth):
                return root_abs, model_name
        try:
            entries = os.listdir(root_abs)
        except OSError:
            continue
        lower_map = {name.lower(): name for name in entries}
        for model_name in model_names:
            hit = lower_map.get(str(model_name).lower())
            if hit and hit.lower().endswith(".obj"):
                return root_abs, hit
    return "", ""


def _resolve_kit_paths():
    conveyor_root = first_existing_path(CONVEYOR_KIT_CANDIDATES)
    truck_obj_root = first_existing_path(RGS_TRUCK_ASSET_CANDIDATES)
    forklift_obj_root = first_existing_path(FORKLIFT_ASSET_CANDIDATES)
    loading_staging_obj_root = first_existing_path(LOADING_STAGING_ASSET_CANDIDATES)

    if conveyor_root is None:
        raise FileNotFoundError(
            "kenney_conveyor-kit not found. Expected one of: "
            + ", ".join(CONVEYOR_KIT_CANDIDATES)
        )
    if ENABLE_LOADING_TRUCKS and truck_obj_root is None:
        raise FileNotFoundError(
            "Loading truck OBJ assets not found. Expected one of: "
            + ", ".join(RGS_TRUCK_ASSET_CANDIDATES)
        )
    needs_industrial_obj = (
        ENABLE_FORKLIFT_PARKING
        or ENABLE_MACHINING_CELL_LAYOUT
        or ENABLE_LOADING_OPERATION_FORKLIFTS
    )
    if needs_industrial_obj and forklift_obj_root is None:
        raise FileNotFoundError(
            "Industrial OBJ assets not found. Expected one of: "
            + ", ".join(FORKLIFT_ASSET_CANDIDATES)
        )

    conveyor_obj = os.path.join(conveyor_root, "Models", "OBJ format")
    conveyor_tex = os.path.join(conveyor_obj, "Textures", "colormap.png")
    truck_obj = truck_obj_root if truck_obj_root else ""
    forklift_obj = forklift_obj_root if forklift_obj_root else ""
    forklift_tex = (
        os.path.join(forklift_obj, FORKLIFT_TEXTURE_NAME)
        if forklift_obj and FORKLIFT_TEXTURE_NAME
        else ""
    )

    if not os.path.exists(conveyor_obj):
        raise FileNotFoundError(f"Missing OBJ folder: {conveyor_obj}")
    if not os.path.exists(conveyor_tex):
        raise FileNotFoundError(f"Missing conveyor texture: {conveyor_tex}")
    conveyor_obj_key = os.path.abspath(conveyor_obj)
    if conveyor_obj_key not in _NORMALIZED_MTL_DIRS:
        normalize_mtl_texture_paths(conveyor_obj)
        _NORMALIZED_MTL_DIRS.add(conveyor_obj_key)
    if ENABLE_LOADING_TRUCKS:
        if not os.path.exists(truck_obj):
            raise FileNotFoundError(f"Missing truck OBJ folder: {truck_obj}")
        for model_name in LOADING_TRUCK_MODELS:
            mp = os.path.join(truck_obj, model_name)
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Missing truck model: {mp}")
    if ENABLE_FORKLIFT_PARKING:
        if not os.path.exists(forklift_obj):
            raise FileNotFoundError(f"Missing forklift OBJ folder: {forklift_obj}")
        fp = os.path.join(forklift_obj, FORKLIFT_MODEL_NAME)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing forklift model: {fp}")
        if forklift_tex and not os.path.exists(forklift_tex):
            raise FileNotFoundError(f"Missing forklift texture: {forklift_tex}")
    needs_loading_staging_obj = ENABLE_LOADING_STAGING or ENABLE_STORAGE_RACK_LAYOUT
    if needs_loading_staging_obj and loading_staging_obj_root:
        for model_name in LOADING_STAGING_MODELS.values():
            mp = os.path.join(loading_staging_obj_root, model_name)
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Missing loading staging model: {mp}")
    if ENABLE_STORAGE_RACK_LAYOUT:
        if not loading_staging_obj_root:
            raise FileNotFoundError(
                "Storage rack assets folder not found. Expected one of: "
                + ", ".join(LOADING_STAGING_ASSET_CANDIDATES)
            )
        rack_mp = os.path.join(loading_staging_obj_root, STORAGE_RACK_MODEL_NAME)
        if not os.path.exists(rack_mp):
            raise FileNotFoundError(f"Missing storage rack model: {rack_mp}")

    return {
        "conveyor_root": conveyor_root,
        "conveyor_obj": conveyor_obj,
        "conveyor_tex": conveyor_tex,
        "truck_obj": truck_obj,
        "forklift_obj": forklift_obj,
        "forklift_tex": forklift_tex,
        "loading_staging_obj": loading_staging_obj_root or "",
    }




def _resolve_shell_mesh_paths():
    for base in WAREHOUSE_SHELL_CANDIDATES:
        root = os.path.abspath(base)
        roof = os.path.join(root, WAREHOUSE_SHELL_FILES["roof"])
        fillers = os.path.join(root, WAREHOUSE_SHELL_FILES["fillers"])
        truss = os.path.join(root, WAREHOUSE_SHELL_FILES["truss"])
        if os.path.exists(roof) and os.path.exists(fillers) and os.path.exists(truss):
            meta_path = os.path.join(root, "metadata.json")
            config = {}
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        config = (json.load(f) or {}).get("config", {}) or {}
                except Exception:
                    config = {}
            return {
                "root": root,
                "roof": roof,
                "fillers": fillers,
                "truss": truss,
                "config": config,
            }

    missing_root = os.path.abspath(WAREHOUSE_SHELL_CANDIDATES[0])
    raise FileNotFoundError(
        "Missing baked warehouse shell meshes. Run: "
        "python tools/procedural_warehouse/bake_warehouse_shell.py "
        f"(expected in {missing_root})"
    )




                                                                              
                 
                                                                              
_TEXTURE_CACHE = {}
_OBJ_MTL_SPLIT_CACHE = {}
_OBJ_COLLISION_PROXY_CACHE = {}
_OBJ_MTL_VISUAL_PROXY_CACHE = {}
_OBJ_DOUBLE_SIDED_PROXY_CACHE = {}
_OFFICE_MODULE_CACHE = None
_FACTORY_MODULE_CACHE = None
_LOADING_TRUCK_ALONG_EXTENT_CACHE = {}
_NORMALIZED_MTL_DIRS = set()
_MESH_VISUAL_SHAPE_CACHE = {}
_MESH_COLLISION_SHAPE_CACHE = {}
_RESOLVED_MESH_PATH_CACHE = {}
_ORIENTED_XY_SIZE_CACHE = {}
_MODEL_BOUNDS_CACHE = {}
_STORAGE_RACK_SUPPORT_LEVELS_CACHE = {}
_FACTORY_BARRIER_LOADER_CACHE = {}


def _loader_runtime_key(loader):
    cached = getattr(loader, "_swarm_runtime_key", None)
    if cached:
        return cached
    obj_dir = str(getattr(loader, "obj_dir", "") or "").strip()
    if obj_dir:
        key = os.path.abspath(obj_dir).replace("\\", "/")
    else:
        key = f"loader:{id(loader)}"
    try:
        setattr(loader, "_swarm_runtime_key", key)
    except Exception:
        pass
    return key


def _clear_loader_spawn_caches(loader):
    if loader is None:
        return
    if hasattr(loader, "visual_shape_cache"):
        loader.visual_shape_cache.clear()
    if hasattr(loader, "collision_shape_cache"):
        loader.collision_shape_cache.clear()
    if hasattr(loader, "texture_id"):
        loader.texture_id = None


def _load_texture_cached(texture_path):
    key = texture_path.replace("\\", "/")
    if key in _TEXTURE_CACHE:
        return _TEXTURE_CACHE[key]
    tid = p.loadTexture(key)
    _TEXTURE_CACHE[key] = tid
    return tid


def _resolve_mesh_path(loader, model_name):
    model_key = str(model_name)
    cache_key = (_loader_runtime_key(loader), model_key)
    cached = _RESOLVED_MESH_PATH_CACHE.get(cache_key)
    if cached is not None:
        return cached

    has_sep = ("/" in model_key) or ("\\" in model_key)
    if os.path.isabs(model_key) or has_sep:
        abs_path = os.path.abspath(model_key)
        if os.path.exists(abs_path):
            resolved = abs_path.replace("\\", "/")
            _RESOLVED_MESH_PATH_CACHE[cache_key] = resolved
            return resolved

    resolved = loader._asset_path(model_key).replace("\\", "/")
    _RESOLVED_MESH_PATH_CACHE[cache_key] = resolved
    return resolved


def _spawn_generated_mesh(
    mesh_path,
    texture_path,
    with_collision=True,
    use_texture=True,
    rgba=(1.0, 1.0, 1.0, 1.0),
    double_sided=False,
    base_position=(0.0, 0.0, 0.0),
    mesh_scale_xyz=(1.0, 1.0, 1.0),
):
    mesh_key = mesh_path.replace("\\", "/")
    msx, msy, msz = (float(mesh_scale_xyz[0]), float(mesh_scale_xyz[1]), float(mesh_scale_xyz[2]))
    create_visual_kwargs = {}
    if double_sided and hasattr(p, "VISUAL_SHAPE_DOUBLE_SIDED"):
        create_visual_kwargs["flags"] = p.VISUAL_SHAPE_DOUBLE_SIDED
    vid = p.createVisualShape(
        p.GEOM_MESH,
        fileName=mesh_key,
        meshScale=[msx, msy, msz],
        rgbaColor=list(rgba),
        **create_visual_kwargs,
    )
    if with_collision:
        kwargs = {}
        if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH"):
            kwargs["flags"] = p.GEOM_FORCE_CONCAVE_TRIMESH
        cid = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=mesh_key,
            meshScale=[msx, msy, msz],
            **kwargs,
        )
    else:
        cid = -1

    body = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=cid,
        baseVisualShapeIndex=vid,
        basePosition=list(base_position),
        useMaximalCoordinates=True,
    )
    visual_kwargs = {
        "rgbaColor": list(rgba),
        "specularColor": list(UNIFORM_SPECULAR_COLOR),
    }

    if use_texture:
        p.changeVisualShape(
            body,
            -1,
            textureUniqueId=_load_texture_cached(texture_path),
            **visual_kwargs,
        )
    else:
        p.changeVisualShape(
            body,
            -1,
            textureUniqueId=-1,
            **visual_kwargs,
        )
    return body


def _estimate_loading_truck_along_extent_m(loading_slot):
    """Estimate truck footprint extent along loading wall axis (meters)."""
    cache_key = (str(loading_slot), tuple(float(v) for v in LOADING_TRUCK_SCALE_XYZ))
    if cache_key in _LOADING_TRUCK_ALONG_EXTENT_CACHE:
        return _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key]

    truck_root = first_existing_path(RGS_TRUCK_ASSET_CANDIDATES)
    if truck_root is None:
        _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key] = 0.0
        return 0.0

    model_name = LOADING_TRUCK_MODELS[0] if LOADING_TRUCK_MODELS else ""
    model_path = os.path.join(truck_root, model_name) if model_name else ""
    if not model_path or not os.path.exists(model_path):
        _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key] = 0.0
        return 0.0

    sx_scale, sy_scale, _sz_scale = LOADING_TRUCK_SCALE_XYZ
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    found = False
    with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            px = x * sx_scale
            py = (-z) * sy_scale
            min_x = min(min_x, px)
            max_x = max(max_x, px)
            min_y = min(min_y, py)
            max_y = max(max_y, py)
            found = True
    if not found:
        _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key] = 0.0
        return 0.0

    size_x = max_x - min_x
    size_y = max_y - min_y
    yaw = math.radians(dock_inward_yaw_for_slot(loading_slot))
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    ex = (c * size_x) + (s * size_y)
    ey = (s * size_x) + (c * size_y)
    along_extent = ex if loading_slot in ("north", "south") else ey
    along_extent = max(0.0, float(along_extent))
    _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key] = along_extent
    return along_extent


def _spawn_mesh_with_anchor(
    loader,
    model_name,
    world_anchor_xyz,
    yaw_deg,
    mesh_scale_xyz,
    local_anchor_xyz,
    with_collision=True,
    use_texture=True,
    texture_path_override="",
    rgba=(1.0, 1.0, 1.0, 1.0),
    double_sided=False,
    frame_quat_override=None,
):
    mesh_path = _resolve_mesh_path(loader, model_name)
    sx, sy, sz = mesh_scale_xyz
    ax, ay, az = local_anchor_xyz
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    wx, wy, wz = world_anchor_xyz

    anchor_off_x = ax * cos_y - ay * sin_y
    anchor_off_y = ax * sin_y + ay * cos_y
    base_pos = [wx - anchor_off_x, wy - anchor_off_y, wz - az]
    yaw_quat = p.getQuaternionFromEuler((0.0, 0.0, yaw_rad))

    frame_quat = frame_quat_override if frame_quat_override is not None else loader.up_fix_quat
    frame_quat_key = tuple(round(float(v), 8) for v in frame_quat) if frame_quat is not None else None
    scale_key = (round(float(sx), 8), round(float(sy), 8), round(float(sz), 8))
    rgba_key = tuple(round(float(v), 6) for v in rgba)
    visual_key = (mesh_path, scale_key, rgba_key, bool(double_sided), frame_quat_key)

    visual_id = _MESH_VISUAL_SHAPE_CACHE.get(visual_key)
    if visual_id is None:
        create_visual_kwargs = {}
        if double_sided and hasattr(p, "VISUAL_SHAPE_DOUBLE_SIDED"):
            create_visual_kwargs["flags"] = p.VISUAL_SHAPE_DOUBLE_SIDED
        visual_id = p.createVisualShape(
            p.GEOM_MESH,
            fileName=mesh_path,
            meshScale=[sx, sy, sz],
            rgbaColor=list(rgba),
            visualFrameOrientation=frame_quat,
            **create_visual_kwargs,
        )
        _MESH_VISUAL_SHAPE_CACHE[visual_key] = visual_id
    if with_collision:
        collision_key = (mesh_path, scale_key, frame_quat_key)
        collision_id = _MESH_COLLISION_SHAPE_CACHE.get(collision_key)
        if collision_id is None:
            kwargs = {}
            if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH"):
                kwargs["flags"] = p.GEOM_FORCE_CONCAVE_TRIMESH
            collision_id = p.createCollisionShape(
                p.GEOM_MESH,
                fileName=mesh_path,
                meshScale=[sx, sy, sz],
                collisionFrameOrientation=frame_quat,
                **kwargs,
            )
            _MESH_COLLISION_SHAPE_CACHE[collision_key] = collision_id
    else:
        collision_id = -1

    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=base_pos,
        baseOrientation=yaw_quat,
        useMaximalCoordinates=True,
    )
    visual_kwargs = {
        "rgbaColor": list(rgba),
        "specularColor": list(UNIFORM_SPECULAR_COLOR),
    }

    if use_texture:
        tex_id = (
            _load_texture_cached(texture_path_override)
            if texture_path_override
            else loader._ensure_texture()
        )
        p.changeVisualShape(
            body_id,
            -1,
            textureUniqueId=tex_id,
            **visual_kwargs,
        )
    else:
        p.changeVisualShape(
            body_id,
            -1,
            textureUniqueId=-1,
            **visual_kwargs,
        )
    return body_id


def _spawn_native_mtl_visual_with_anchor(
    loader,
    model_name,
    world_anchor_xyz,
    yaw_deg,
    mesh_scale_xyz,
    local_anchor_xyz,
    model_path_override="",
    collision_model_path_override="",
    with_collision=True,
    double_sided=False,
):
    if model_path_override and os.path.exists(model_path_override):
        mesh_path = os.path.abspath(model_path_override).replace("\\", "/")
    else:
        mesh_path = _resolve_mesh_path(loader, model_name)

    sx, sy, sz = mesh_scale_xyz
    ax, ay, az = local_anchor_xyz
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    wx, wy, wz = world_anchor_xyz

    anchor_off_x = ax * cos_y - ay * sin_y
    anchor_off_y = ax * sin_y + ay * cos_y
    base_pos = [wx - anchor_off_x, wy - anchor_off_y, wz - az]
    yaw_quat = p.getQuaternionFromEuler((0.0, 0.0, yaw_rad))

    create_visual_kwargs = {}
    if double_sided and hasattr(p, "VISUAL_SHAPE_DOUBLE_SIDED"):
        create_visual_kwargs["flags"] = p.VISUAL_SHAPE_DOUBLE_SIDED
    visual_id = p.createVisualShape(
        p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[sx, sy, sz],
        rgbaColor=[1.0, 1.0, 1.0, 1.0],
        visualFrameOrientation=loader.up_fix_quat,
        **create_visual_kwargs,
    )
    if with_collision:
        collision_mesh_path = (
            os.path.abspath(collision_model_path_override).replace("\\", "/")
            if collision_model_path_override and os.path.exists(collision_model_path_override)
            else mesh_path
        )
        kwargs = {}
        if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH"):
            kwargs["flags"] = p.GEOM_FORCE_CONCAVE_TRIMESH
        collision_id = p.createCollisionShape(
            p.GEOM_MESH,
            fileName=collision_mesh_path,
            meshScale=[sx, sy, sz],
            collisionFrameOrientation=loader.up_fix_quat,
            **kwargs,
        )
    else:
        collision_id = -1
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        basePosition=base_pos,
        baseOrientation=yaw_quat,
        useMaximalCoordinates=True,
    )
    return body_id


def _spawn_collision_only_with_anchor(
    loader,
    model_name,
    world_anchor_xyz,
    yaw_deg,
    mesh_scale_xyz,
    local_anchor_xyz,
    model_path_override="",
    frame_quat_override=None,
):
    if model_path_override and os.path.exists(model_path_override):
        mesh_path = os.path.abspath(model_path_override).replace("\\", "/")
    else:
        mesh_path = _resolve_mesh_path(loader, model_name)

    sx, sy, sz = mesh_scale_xyz
    ax, ay, az = local_anchor_xyz
    yaw_rad = math.radians(yaw_deg)
    cos_y = math.cos(yaw_rad)
    sin_y = math.sin(yaw_rad)
    wx, wy, wz = world_anchor_xyz

    anchor_off_x = ax * cos_y - ay * sin_y
    anchor_off_y = ax * sin_y + ay * cos_y
    base_pos = [wx - anchor_off_x, wy - anchor_off_y, wz - az]
    yaw_quat = p.getQuaternionFromEuler((0.0, 0.0, yaw_rad))

    kwargs = {}
    if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH"):
        kwargs["flags"] = p.GEOM_FORCE_CONCAVE_TRIMESH
    collision_id = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[sx, sy, sz],
        collisionFrameOrientation=frame_quat_override if frame_quat_override is not None else loader.up_fix_quat,
        **kwargs,
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=-1,
        basePosition=base_pos,
        baseOrientation=yaw_quat,
        useMaximalCoordinates=True,
    )
    p.changeVisualShape(
        body_id,
        -1,
        rgbaColor=[1.0, 1.0, 1.0, 0.0],
        textureUniqueId=-1,
        specularColor=list(UNIFORM_SPECULAR_COLOR),
    )
    return body_id


def _spawn_box_primitive(center_xyz, size_xyz, rgba, with_collision=True):
    hx = float(size_xyz[0]) * 0.5
    hy = float(size_xyz[1]) * 0.5
    hz = float(size_xyz[2]) * 0.5
    vid = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[hx, hy, hz],
        rgbaColor=list(rgba),
    )
    if with_collision:
        cid = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[hx, hy, hz],
        )
    else:
        cid = -1
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=cid,
        baseVisualShapeIndex=vid,
        basePosition=list(center_xyz),
        useMaximalCoordinates=True,
    )
    p.changeVisualShape(
        body_id,
        -1,
        rgbaColor=list(rgba),
        textureUniqueId=-1,
        specularColor=list(UNIFORM_SPECULAR_COLOR),
    )
    return body_id


def _safe_token_name(name):
    out = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    token = "".join(out).strip("._")
    return token or "mat"


def _purge_generated_model_artifacts(model_path):
    cache_key = os.path.abspath(model_path)
    _OBJ_MTL_SPLIT_CACHE.pop(cache_key, None)
    _OBJ_COLLISION_PROXY_CACHE.pop(cache_key, None)
    _OBJ_MTL_VISUAL_PROXY_CACHE.pop(cache_key, None)
    _OBJ_DOUBLE_SIDED_PROXY_CACHE.pop(cache_key, None)
    _TEXTURE_CACHE.clear()

    split_root = os.path.join(
        os.path.dirname(model_path),
        "_split_by_mtl",
        _safe_token_name(os.path.splitext(os.path.basename(model_path))[0]),
    )
    if os.path.isdir(split_root):
        shutil.rmtree(split_root, ignore_errors=True)

    double_sided_root = os.path.join(
        os.path.dirname(model_path),
        "_double_sided",
    )
    if os.path.isdir(double_sided_root):
        shutil.rmtree(double_sided_root, ignore_errors=True)


def _obj_double_sided_proxy_path(model_path):
    cache_key = os.path.abspath(model_path)
    cached = _OBJ_DOUBLE_SIDED_PROXY_CACHE.get(cache_key)
    if cached and os.path.exists(cached):
        return cached

    try:
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        _OBJ_DOUBLE_SIDED_PROXY_CACHE[cache_key] = model_path
        return model_path

    out_lines = []
    reversed_face_count = 0
    for raw in lines:
        out_lines.append(raw)
        stripped = raw.lstrip()
        if not stripped.startswith("f "):
            continue
        tokens = stripped.split()[1:]
        if len(tokens) < 3:
            continue
        indent = raw[: len(raw) - len(stripped)]
        out_lines.append(f"{indent}f " + " ".join(reversed(tokens)) + "\n")
        reversed_face_count += 1

    if reversed_face_count == 0:
        _OBJ_DOUBLE_SIDED_PROXY_CACHE[cache_key] = model_path
        return model_path

    out_root = os.path.join(os.path.dirname(model_path), "_double_sided")
    os.makedirs(out_root, exist_ok=True)
    out_name = f"{os.path.splitext(os.path.basename(model_path))[0]}__double.obj"
    out_path = os.path.join(out_root, out_name)
    with open(out_path, "w", encoding="utf-8") as o:
        o.writelines(out_lines)

    _OBJ_DOUBLE_SIDED_PROXY_CACHE[cache_key] = out_path
    return out_path


def _obj_collision_proxy_path(model_path):
    cache_key = os.path.abspath(model_path)
    if cache_key in _OBJ_COLLISION_PROXY_CACHE:
        return _OBJ_COLLISION_PROXY_CACHE[cache_key]

    try:
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        _OBJ_COLLISION_PROXY_CACHE[cache_key] = model_path
        return model_path

    sanitized = []
    for raw in lines:
        stripped = raw.lstrip()
        if stripped.startswith("mtllib ") or stripped.startswith("usemtl "):
            continue
        sanitized.append(raw)

    if not sanitized:
        _OBJ_COLLISION_PROXY_CACHE[cache_key] = model_path
        return model_path

    split_root = os.path.join(
        os.path.dirname(model_path),
        "_split_by_mtl",
        _safe_token_name(os.path.splitext(os.path.basename(model_path))[0]),
    )
    os.makedirs(split_root, exist_ok=True)
    out_path = os.path.join(split_root, "__collision_proxy.obj")
    with open(out_path, "w", encoding="utf-8") as o:
        o.write("# collision-only proxy: stripped mtllib/usemtl\n")
        o.writelines(sanitized)

    _OBJ_COLLISION_PROXY_CACHE[cache_key] = out_path
    return out_path


def _resolve_mtl_texture_path(mtl_path, tex_ref):
    if not tex_ref:
        return ""
    ref = str(tex_ref).strip().strip("\"'").replace("\\", "/")
    if not ref:
        return ""

    mtl_dir = os.path.dirname(os.path.abspath(mtl_path))
    base_name = os.path.basename(ref)
    candidates = []

    if os.path.isabs(ref):
        candidates.append(ref)
    else:
        candidates.append(os.path.join(mtl_dir, ref))
    if base_name:
        candidates.append(os.path.join(mtl_dir, base_name))

    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c).replace("\\", "/")
    return ""


def _obj_mtl_visual_proxy_path(model_path):
    cache_key = os.path.abspath(model_path)
    if (not MACHINING_FORCE_REFRESH_MTL_PROXY) and cache_key in _OBJ_MTL_VISUAL_PROXY_CACHE:
        return _OBJ_MTL_VISUAL_PROXY_CACHE[cache_key]

    try:
        with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
            obj_lines = f.readlines()
    except Exception:
        _OBJ_MTL_VISUAL_PROXY_CACHE[cache_key] = model_path
        return model_path

    mtllib_name = None
    for raw in obj_lines:
        stripped = raw.strip()
        if stripped.startswith("mtllib "):
            mtllib_name = stripped.split(maxsplit=1)[1].strip()
            break

    if not mtllib_name:
        _OBJ_MTL_VISUAL_PROXY_CACHE[cache_key] = model_path
        return model_path

    mtl_path = os.path.join(os.path.dirname(model_path), mtllib_name)
    if not os.path.exists(mtl_path):
        _OBJ_MTL_VISUAL_PROXY_CACHE[cache_key] = model_path
        return model_path

    split_root = os.path.join(
        os.path.dirname(model_path),
        "_split_by_mtl",
        _safe_token_name(os.path.splitext(os.path.basename(model_path))[0]),
    )
    os.makedirs(split_root, exist_ok=True)
    out_obj_path = os.path.join(split_root, "__visual_proxy.obj")
    out_mtl_name = "__visual_proxy.mtl"
    out_mtl_path = os.path.join(split_root, out_mtl_name)

    rewritten_mtl = []
    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            stripped = raw.strip()
            if stripped.lower().startswith("map_kd "):
                tex_ref = stripped.split(maxsplit=1)[1].strip() if len(stripped.split(maxsplit=1)) >= 2 else ""
                resolved_tex = _resolve_mtl_texture_path(mtl_path, tex_ref)
                if resolved_tex and os.path.exists(resolved_tex):
                    tex_name = os.path.basename(resolved_tex)
                    dst_tex = os.path.join(split_root, tex_name)
                    if not os.path.exists(dst_tex):
                        shutil.copy2(resolved_tex, dst_tex)
                    indent = raw[: len(raw) - len(raw.lstrip())]
                    rewritten_mtl.append(f"{indent}map_Kd {tex_name}\n")
                    continue
            rewritten_mtl.append(raw)

    with open(out_mtl_path, "w", encoding="utf-8") as f:
        f.writelines(rewritten_mtl)

    out_obj_lines = []
    for raw in obj_lines:
        stripped = raw.strip()
        if stripped.startswith("mtllib "):
            indent = raw[: len(raw) - len(raw.lstrip())]
            out_obj_lines.append(f"{indent}mtllib {out_mtl_name}\n")
        else:
            out_obj_lines.append(raw)
    with open(out_obj_path, "w", encoding="utf-8") as f:
        f.writelines(out_obj_lines)

    _OBJ_MTL_VISUAL_PROXY_CACHE[cache_key] = out_obj_path
    return out_obj_path


def _parse_mtl_colors(mtl_path):
    colors = {}
    texture_by_material = {}
    ka_colors = {}
    map_kd = {}
    alpha = {}
    current = None
    if not os.path.exists(mtl_path):
        return colors, texture_by_material

    def _color_from_texture_ref(tex_ref):
        key = os.path.basename(str(tex_ref).replace("\\", "/")).strip().lower()
        palette = {
            "trak-k3-kmx-left-side-view-zoom.jpg": (0.66, 0.78, 0.72),                     
            "trak-k3-kmx-front-view-zoom.jpg": (0.66, 0.78, 0.72),                         
            "staal.jpg": (0.60, 0.61, 0.63),                                    
            "perfo plaat.jpg": (0.44, 0.45, 0.47),                                         
            "donker staal.jpeg": (0.16, 0.16, 0.18),                                 
            "tnt124control-00000248.jpg": (0.24, 0.26, 0.30),                           
            "img0.jpg": (0.92, 0.92, 0.92),                                                   
        }
        return palette.get(key)

    with open(mtl_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            head = parts[0]
            if head == "newmtl" and len(parts) >= 2:
                current = " ".join(parts[1:])
                continue
            if current is None:
                continue
            if head == "Kd" and len(parts) >= 4:
                try:
                    r = float(parts[1])
                    g = float(parts[2])
                    b = float(parts[3])
                    colors[current] = [r, g, b, alpha.get(current, 1.0)]
                except Exception:
                    pass
            elif head == "Ka" and len(parts) >= 4:
                try:
                    r = float(parts[1])
                    g = float(parts[2])
                    b = float(parts[3])
                    ka_colors[current] = [r, g, b]
                except Exception:
                    pass
            elif head.lower() == "map_kd":
                tex = line.split(maxsplit=1)[1].strip() if len(line.split(maxsplit=1)) >= 2 else ""
                if tex:
                    map_kd[current] = tex
            elif head == "d" and len(parts) >= 2:
                try:
                    a = float(parts[1])
                    alpha[current] = a
                    if current in colors:
                        colors[current][3] = a
                except Exception:
                    pass
            elif head == "Tr" and len(parts) >= 2:
                try:
                    a = 1.0 - float(parts[1])
                    alpha[current] = a
                    if current in colors:
                        colors[current][3] = a
                except Exception:
                    pass

    for mat, tex_ref in map_kd.items():
        tex_path = _resolve_mtl_texture_path(mtl_path, tex_ref)
        if tex_path:
            texture_by_material[mat] = tex_path

    for mat, tex_ref in map_kd.items():
        if mat in colors:
            continue
        rgb = _color_from_texture_ref(tex_ref)
        if rgb is None:
            continue
        r, g, b = rgb
        colors[mat] = [r, g, b, alpha.get(mat, 1.0)]
    for mat, ka in ka_colors.items():
        if mat in colors:
            continue
        r, g, b = ka
        colors[mat] = [r, g, b, alpha.get(mat, 1.0)]
    for mat in map_kd.keys():
        if mat in colors:
            continue
        colors[mat] = [0.62, 0.64, 0.66, alpha.get(mat, 1.0)]

    for mat in list(colors.keys()):
        key = str(mat).strip().lower()
        if key == "slang":
            colors[mat] = [0.10, 0.11, 0.12, alpha.get(mat, colors[mat][3])]

    return colors, texture_by_material


def _obj_material_parts(model_path):
    cache_key = os.path.abspath(model_path)
    if cache_key in _OBJ_MTL_SPLIT_CACHE:
        return _OBJ_MTL_SPLIT_CACHE[cache_key]

    split_root = os.path.join(
        os.path.dirname(model_path),
        "_split_by_mtl",
        _safe_token_name(os.path.splitext(os.path.basename(model_path))[0]),
    )
    os.makedirs(split_root, exist_ok=True)

    model_sig = {"size": -1, "mtime_ns": -1}
    try:
        st = os.stat(model_path)
        model_sig = {
            "size": int(st.st_size),
            "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
        }
    except Exception:
        pass

    manifest_path = os.path.join(split_root, "_parts_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f) or {}
            sig = manifest.get("model_sig", {}) or {}
            sig_size = int(sig.get("size", -1))
            sig_mtime = int(sig.get("mtime_ns", -1))
            if sig_size == model_sig["size"] and sig_mtime == model_sig["mtime_ns"]:
                cached_parts = []
                valid_manifest = True
                for item in manifest.get("parts", []) or []:
                    rel_or_abs = str(item.get("path", "")).strip()
                    if not rel_or_abs:
                        valid_manifest = False
                        break
                    part_path = (
                        rel_or_abs
                        if os.path.isabs(rel_or_abs)
                        else os.path.join(split_root, rel_or_abs)
                    )
                    part_path = os.path.abspath(part_path)
                    if not os.path.exists(part_path):
                        valid_manifest = False
                        break
                    rgba = item.get("rgba", [0.72, 0.72, 0.72, 1.0])
                    if not isinstance(rgba, (list, tuple)) or len(rgba) < 3:
                        rgba = [0.72, 0.72, 0.72, 1.0]
                    rgba = [
                        float(rgba[0]),
                        float(rgba[1]),
                        float(rgba[2]),
                        float(rgba[3]) if len(rgba) >= 4 else 1.0,
                    ]
                    cached_parts.append(
                        {
                            "path": part_path,
                            "material": str(item.get("material", "default")),
                            "rgba": rgba,
                            "texture_path": str(item.get("texture_path", "")),
                        }
                    )
                if valid_manifest:
                    _OBJ_MTL_SPLIT_CACHE[cache_key] = cached_parts
                    return cached_parts
        except Exception:
            pass

    with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    vertices = []
    texcoords = []
    normals = []
    mtllib_name = None
    for raw in lines:
        if raw.startswith("v "):
            parts = raw.split()
            if len(parts) >= 4:
                try:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except Exception:
                    pass
        elif raw.startswith("vt "):
            parts = raw.split()
            if len(parts) >= 3:
                try:
                    if len(parts) >= 4:
                        texcoords.append((float(parts[1]), float(parts[2]), float(parts[3])))
                    else:
                        texcoords.append((float(parts[1]), float(parts[2])))
                except Exception:
                    pass
        elif raw.startswith("vn "):
            parts = raw.split()
            if len(parts) >= 4:
                try:
                    normals.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except Exception:
                    pass
        elif raw.startswith("mtllib ") and mtllib_name is None:
            mtllib_name = raw.split(maxsplit=1)[1].strip()

    if not vertices:
        _OBJ_MTL_SPLIT_CACHE[cache_key] = []
        return []

    def _resolve_obj_index(token, count):
        if not token:
            return None
        try:
            idx = int(token)
        except Exception:
            return None
        if idx > 0:
            idx0 = idx - 1
        elif idx < 0:
            idx0 = count + idx
        else:
            return None
        if 0 <= idx0 < count:
            return idx0
        return None

    material_faces = {}
    current_mtl = "default"
    for raw in lines:
        if raw.startswith("usemtl "):
            current_mtl = raw.split(maxsplit=1)[1].strip() or "default"
            continue
        if not raw.startswith("f "):
            continue
        toks = raw.split()[1:]
        corners = []
        for t in toks:
            chunks = t.split("/")
            vi = _resolve_obj_index(chunks[0] if len(chunks) >= 1 else "", len(vertices))
            if vi is None:
                continue
            vti = _resolve_obj_index(chunks[1] if len(chunks) >= 2 else "", len(texcoords))
            vni = _resolve_obj_index(chunks[2] if len(chunks) >= 3 else "", len(normals))
            corners.append((vi, vti, vni))
        if len(corners) < 3:
            continue
        tris = material_faces.setdefault(current_mtl, [])
        for i in range(1, len(corners) - 1):
            tris.append((corners[0], corners[i], corners[i + 1]))

    mtl_colors = {}
    mtl_textures = {}
    if mtllib_name:
        mtl_path = os.path.join(os.path.dirname(model_path), mtllib_name)
        mtl_colors, mtl_textures = _parse_mtl_colors(mtl_path)

    out_parts = []
    for mtl_name, tris in material_faces.items():
        if not tris:
            continue
        unique_vi = {c[0] for tri in tris for c in tri if c[0] is not None}
        if len(unique_vi) < 3:
            continue
        total_area = 0.0
        for tri in tris:
            try:
                (v0, _t0, _n0), (v1, _t1, _n1), (v2, _t2, _n2) = tri
                p0 = vertices[v0]
                p1 = vertices[v1]
                p2 = vertices[v2]
                ux, uy, uz = (p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2])
                vx, vy, vz = (p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2])
                cx = uy * vz - uz * vy
                cy = uz * vx - ux * vz
                cz = ux * vy - uy * vx
                total_area += 0.5 * math.sqrt(cx * cx + cy * cy + cz * cz)
            except Exception:
                continue
        if total_area <= 1e-8:
            continue
        used_v = sorted({c[0] for tri in tris for c in tri if c[0] is not None})
        used_vt = sorted({c[1] for tri in tris for c in tri if c[1] is not None})
        used_vn = sorted({c[2] for tri in tris for c in tri if c[2] is not None})
        remap_v = {old_i: (new_i + 1) for new_i, old_i in enumerate(used_v)}
        remap_vt = {old_i: (new_i + 1) for new_i, old_i in enumerate(used_vt)}
        remap_vn = {old_i: (new_i + 1) for new_i, old_i in enumerate(used_vn)}
        out_name = f"{_safe_token_name(mtl_name)}.obj"
        out_path = os.path.join(split_root, out_name)
        with open(out_path, "w", encoding="utf-8") as o:
            o.write(f"# material split: {mtl_name}\n")
            for old_i in used_v:
                vx, vy, vz = vertices[old_i]
                o.write(f"v {vx:.6f} {vy:.6f} {vz:.6f}\n")
            for old_i in used_vt:
                tc = texcoords[old_i]
                if len(tc) >= 3:
                    o.write(f"vt {tc[0]:.6f} {tc[1]:.6f} {tc[2]:.6f}\n")
                else:
                    o.write(f"vt {tc[0]:.6f} {tc[1]:.6f}\n")
            for old_i in used_vn:
                nx, ny, nz = normals[old_i]
                o.write(f"vn {nx:.6f} {ny:.6f} {nz:.6f}\n")
            for tri in tris:
                tokens = []
                for vi, vti, vni in tri:
                    rv = remap_v[vi]
                    rvt = remap_vt.get(vti) if vti is not None else None
                    rvn = remap_vn.get(vni) if vni is not None else None
                    if rvt is not None and rvn is not None:
                        tokens.append(f"{rv}/{rvt}/{rvn}")
                    elif rvt is not None:
                        tokens.append(f"{rv}/{rvt}")
                    elif rvn is not None:
                        tokens.append(f"{rv}//{rvn}")
                    else:
                        tokens.append(f"{rv}")
                o.write("f " + " ".join(tokens) + "\n")
        rgba = mtl_colors.get(mtl_name, [0.72, 0.72, 0.72, 1.0])
        out_parts.append(
            {
                "path": out_path,
                "material": mtl_name,
                "rgba": rgba,
                "texture_path": mtl_textures.get(mtl_name, ""),
            }
        )

    try:
        manifest_parts = []
        for part in out_parts:
            part_path_abs = os.path.abspath(str(part.get("path", "")))
            part_rel = os.path.relpath(part_path_abs, split_root).replace("\\", "/")
            manifest_parts.append(
                {
                    "path": part_rel,
                    "material": str(part.get("material", "default")),
                    "rgba": [float(v) for v in list(part.get("rgba", [0.72, 0.72, 0.72, 1.0]))[:4]],
                    "texture_path": str(part.get("texture_path", "")),
                }
            )
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "version": 1,
                    "model_sig": model_sig,
                    "parts": manifest_parts,
                },
                mf,
                ensure_ascii=True,
                indent=0,
            )
    except Exception:
        pass

    _OBJ_MTL_SPLIT_CACHE[cache_key] = out_parts
    return out_parts


def slot_point(slot, along, inward):
    hx = WAREHOUSE_SIZE_X * 0.5
    hy = WAREHOUSE_SIZE_Y * 0.5
    if slot == "north":
        return along, hy - inward
    if slot == "south":
        return along, -hy + inward
    if slot == "east":
        return hx - inward, along
    if slot == "west":
        return -hx + inward, along
    raise ValueError(f"Unknown slot: {slot}")


def wall_yaw_for_slot(slot):
    return dock_inward_yaw_for_slot(slot)


def dock_inward_yaw_for_slot(slot):
    if slot == "north":
        return 180.0
    if slot == "south":
        return 0.0
    if slot == "east":
        return 90.0
    if slot == "west":
        return 270.0
    raise ValueError(f"Unknown slot: {slot}")


def _truck_extra_gap_for_gate_state(gate_model_name):
    name = os.path.basename(str(gate_model_name)).lower()
    if "closed" in name:
        return LOADING_TRUCK_EXTRA_GAP_CLOSED
    if "half" in name:
        return LOADING_TRUCK_EXTRA_GAP_HALF
    return 0.0


def tiled_centers(total_size, tile_size):
    if tile_size <= 1e-9:
        raise ValueError(f"Invalid tile size: {tile_size}")
    n = max(1, int(math.floor(float(total_size) / float(tile_size))))
    covered = n * float(tile_size)
    start = -covered * 0.5 + float(tile_size) * 0.5
    return [start + i * tile_size for i in range(n)]


def oriented_xy_size(loader, model_name, scale, yaw_deg):
    if isinstance(scale, (tuple, list)):
        scale_key = (float(scale[0]), float(scale[1]), float(scale[2]))
    else:
        s = float(scale)
        scale_key = (s, s, s)
    yaw_key = round(float(yaw_deg) % 360.0, 6)
    cache_key = (_loader_runtime_key(loader), str(model_name), scale_key, yaw_key)
    cached = _ORIENTED_XY_SIZE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    sx, sy, _ = loader.model_size(model_name, scale_key)
    yaw = math.radians(float(yaw_key))
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    ex = (c * sx) + (s * sy)
    ey = (s * sx) + (c * sy)
    out = (float(ex), float(ey))
    _ORIENTED_XY_SIZE_CACHE[cache_key] = out
    return out


def model_bounds_xyz(loader, model_name, scale_xyz):
    sx, sy, sz = float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])
    scale_key = (sx, sy, sz)
    cache_key = (_loader_runtime_key(loader), str(model_name), scale_key)
    cached = _MODEL_BOUNDS_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if hasattr(loader, "_bounds"):
        min_v, max_v = loader._bounds(model_name, scale_key)
    else:
        verts = loader._parse_vertices(model_name)
        transformed = [(v[0] * sx, v[1] * sy, v[2] * sz) for v in verts]
        min_v = [min(v[i] for v in transformed) for i in range(3)]
        max_v = [max(v[i] for v in transformed) for i in range(3)]

    out = (
        [float(min_v[0]), float(min_v[1]), float(min_v[2])],
        [float(max_v[0]), float(max_v[1]), float(max_v[2])],
    )
    _MODEL_BOUNDS_CACHE[cache_key] = out
    return out


def mirrored_window_indices(segment_count):
    if segment_count <= 5:
        return set()
    if segment_count >= 12:
        picks = [2, segment_count // 2, segment_count - 3]
    elif segment_count >= 8:
        picks = [2, segment_count - 3]
    else:
        picks = [segment_count // 2]
    return {i for i in picks if 0 < i < (segment_count - 1)}


def mirrored_wide_window_starts(segment_count, span_steps, seed_key):
    if span_steps <= 1 or segment_count < (span_steps + 6):
        return []

    rng = random.Random(seed_key + segment_count * 97 + span_steps * 13)
    shift = rng.choice((-1, 0, 1))
    starts = []

    if segment_count >= 20:
        left = (segment_count // 4) - (span_steps // 2) + shift
        left = max(1, min(left, segment_count - (2 * span_steps) - 2))
        right = segment_count - span_steps - left
        if right - left >= span_steps + 1:
            starts = [left, right]
        else:
            starts = [max(1, min(segment_count // 2 - (span_steps // 2), segment_count - span_steps - 1))]
    elif segment_count >= 12:
        center = segment_count // 2 - (span_steps // 2)
        starts = [max(1, min(center, segment_count - span_steps - 1))]

    out = []
    for s in starts:
        s = max(1, min(s, segment_count - span_steps - 1))
        if s not in out:
            out.append(s)
    return sorted(out)


def _indices_blocked_by_doors(along_values, door_centers, door_span):
    blocked = set()
    if not along_values or not door_centers:
        return blocked
    if len(along_values) >= 2:
        step = abs(float(along_values[1]) - float(along_values[0]))
    else:
        step = max(1e-6, float(door_span))
    seg_half = step * 0.5
    door_half = float(door_span) * 0.5
    for idx, along in enumerate(along_values):
        seg_lo = float(along) - seg_half
        seg_hi = float(along) + seg_half
        for c in door_centers:
            door_lo = float(c) - door_half
            door_hi = float(c) + door_half
            if (seg_hi > (door_lo + 1e-6)) and (seg_lo < (door_hi - 1e-6)):
                blocked.add(idx)
                break
    return blocked


def _merge_spans_1d(spans, eps=1e-6):
    if not spans:
        return []
    ordered = sorted((float(lo), float(hi)) for lo, hi in spans if float(hi) > float(lo) + eps)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1] + eps:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _subtract_spans_1d(base_spans, cut_spans, eps=1e-6):
    if not base_spans:
        return []
    base_merged = _merge_spans_1d(base_spans, eps=eps)
    cut_merged = _merge_spans_1d(cut_spans, eps=eps)
    if not cut_merged:
        return base_merged

    out = []
    for blo, bhi in base_merged:
        segments = [(blo, bhi)]
        for clo, chi in cut_merged:
            next_segments = []
            for slo, shi in segments:
                if chi <= slo + eps or clo >= shi - eps:
                    next_segments.append((slo, shi))
                    continue
                if clo > slo + eps:
                    next_segments.append((slo, min(shi, clo)))
                if chi < shi - eps:
                    next_segments.append((max(slo, chi), shi))
            segments = next_segments
            if not segments:
                break
        for slo, shi in segments:
            if shi > slo + eps:
                out.append((slo, shi))
    return _merge_spans_1d(out, eps=eps)


def _filter_mirrored_single_windows(candidate_indices, blocked_indices, segment_count):
    out = set()
    for i in sorted(candidate_indices):
        j = segment_count - 1 - i
        if i > j:
            continue
        if i == j:
            if i not in blocked_indices:
                out.add(i)
            continue
        if i not in blocked_indices and j not in blocked_indices:
            out.add(i)
            out.add(j)
    return out


def _span_is_clear(start_idx, span_steps, blocked_indices):
    for k in range(span_steps):
        if (start_idx + k) in blocked_indices:
            return False
    return True


def _filter_mirrored_wide_windows(candidate_starts, span_steps, blocked_indices, segment_count):
    if span_steps <= 1:
        return sorted(candidate_starts)
    min_start = 1
    max_start = segment_count - span_steps - 1
    out = set()
    for s in sorted(candidate_starts):
        s = max(min_start, min(s, max_start))
        m = segment_count - span_steps - s
        m = max(min_start, min(m, max_start))
        if s > m:
            continue
        if s == m:
            if _span_is_clear(s, span_steps, blocked_indices):
                out.add(s)
            continue
        if _span_is_clear(s, span_steps, blocked_indices) and _span_is_clear(m, span_steps, blocked_indices):
            out.add(s)
            out.add(m)
    return sorted(out)


def build_floor(loader):
    tile_x, tile_y, _ = loader.model_size(CONVEYOR_ASSETS["floor"], UNIFORM_SCALE)
    margin_x = tile_x * FLOOR_INNER_MARGIN_TILES
    margin_y = tile_y * FLOOR_INNER_MARGIN_TILES
    floor_size_x = WAREHOUSE_SIZE_X - (2.0 * margin_x)
    floor_size_y = WAREHOUSE_SIZE_Y - (2.0 * margin_y)
    if floor_size_x <= 0.0 or floor_size_y <= 0.0:
        raise ValueError("Floor interior margin is too large for warehouse footprint.")

    floor_half_z = 0.03
    cid = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[floor_size_x * 0.5, floor_size_y * 0.5, floor_half_z],
    )
    vid = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[floor_size_x * 0.5, floor_size_y * 0.5, floor_half_z],
        rgbaColor=list(FLOOR_UNIFORM_COLOR),
    )
    floor_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=cid,
        baseVisualShapeIndex=vid,
        basePosition=[0.0, 0.0, -floor_half_z],
        useMaximalCoordinates=True,
    )
    p.changeVisualShape(
        floor_id,
        -1,
        rgbaColor=list(FLOOR_UNIFORM_COLOR),
        textureUniqueId=-1,
        specularColor=list(UNIFORM_SPECULAR_COLOR),
    )
    return 0.0


def _floor_spawn_half_extents(loader, safety_margin_m=FLOOR_SPAWN_SAFETY_MARGIN_M):
    tile_x, tile_y, _ = loader.model_size(CONVEYOR_ASSETS["floor"], UNIFORM_SCALE)
    margin_x = tile_x * FLOOR_INNER_MARGIN_TILES
    margin_y = tile_y * FLOOR_INNER_MARGIN_TILES
    floor_half_x = (WAREHOUSE_SIZE_X - (2.0 * margin_x)) * 0.5
    floor_half_y = (WAREHOUSE_SIZE_Y - (2.0 * margin_y)) * 0.5
    safe_half_x = max(0.5, floor_half_x - float(safety_margin_m))
    safe_half_y = max(0.5, floor_half_y - float(safety_margin_m))
    return safe_half_x, safe_half_y


def _first_existing_model_name(loader, candidates):
    for model_name in candidates:
        if os.path.exists(os.path.join(loader.obj_dir, model_name)):
            return model_name
    return None


def build_personnel_floor_lane(loader, floor_top_z, wall_info):
    if not ENABLE_PERSONNEL_FLOOR_LANE:
        return {"personnel_floor_lane_enabled": False}

    lane_model = _first_existing_model_name(loader, PERSONNEL_FLOOR_LANE_MODEL_CANDIDATES)
    if lane_model is None:
        return {
            "personnel_floor_lane_enabled": False,
            "personnel_floor_lane_reason": "No alternate floor model found.",
        }

    personnel_side = wall_info.get("personnel_side")
    if personnel_side not in WALL_SLOTS:
        return {
            "personnel_floor_lane_enabled": False,
            "personnel_floor_lane_reason": "Personnel door side is missing.",
        }

    personnel_along = float(wall_info.get("personnel_along", 0.0))
    wall_thickness = float(wall_info.get("wall_thickness", 0.0))

    start_x, start_y = slot_point(
        personnel_side,
        personnel_along,
        inward=(wall_thickness * 0.5) + DOCK_INWARD_NUDGE,
    )

    if personnel_side == "east":
        dir_x, dir_y = -1.0, 0.0
    elif personnel_side == "west":
        dir_x, dir_y = 1.0, 0.0
    elif personnel_side == "north":
        dir_x, dir_y = 0.0, -1.0
    else:
        dir_x, dir_y = 0.0, 1.0

    move_along_x = abs(dir_x) > 0.5
    yaw_deg = 0.0 if move_along_x else 90.0
    ex, ey = oriented_xy_size(loader, lane_model, UNIFORM_SCALE, yaw_deg)
    step_len = ex if move_along_x else ey
    lane_wid = ey if move_along_x else ex
    if step_len <= 1e-6:
        return {
            "personnel_floor_lane_enabled": False,
            "personnel_floor_lane_reason": "Invalid lane tile size.",
        }

    half_step = step_len * 0.5
    half_wid = lane_wid * 0.5
    edge_tol = step_len * PERSONNEL_FLOOR_LANE_EDGE_TOLERANCE_TILES
    floor_half_x, floor_half_y = _floor_spawn_half_extents(loader)
    face_x = min((WAREHOUSE_SIZE_X * 0.5) - (wall_thickness * 0.5), floor_half_x)
    face_y = min((WAREHOUSE_SIZE_Y * 0.5) - (wall_thickness * 0.5), floor_half_y)

    if move_along_x:
        y = max(-face_y + half_wid, min(face_y - half_wid, start_y))
        x0 = start_x + (dir_x * half_step)
        target_face_x = (-face_x if dir_x < 0.0 else face_x)
        x_target = target_face_x - (dir_x * half_step) - (dir_x * edge_tol)
        run_len = abs(x_target - x0)
    else:
        x = max(-face_x + half_wid, min(face_x - half_wid, start_x))
        y0 = start_y + (dir_y * half_step)
        target_face_y = (-face_y if dir_y < 0.0 else face_y)
        y_target = target_face_y - (dir_y * half_step) - (dir_y * edge_tol)
        run_len = abs(y_target - y0)

    tile_count = max(1, int(math.floor(run_len / step_len)) + 1)
    spawned = 0
    eps = 1e-6
    for i in range(tile_count + 2):
        if move_along_x:
            cx = x0 + (dir_x * step_len * i)
            if (dir_x < 0.0 and cx < (x_target - eps)) or (dir_x > 0.0 and cx > (x_target + eps)):
                break
            cy = y
        else:
            cx = x
            cy = y0 + (dir_y * step_len * i)
            if (dir_y < 0.0 and cy < (y_target - eps)) or (dir_y > 0.0 and cy > (y_target + eps)):
                break

        loader.spawn(
            lane_model,
            x=cx,
            y=cy,
            yaw_deg=yaw_deg,
            floor_z=floor_top_z,
            scale=UNIFORM_SCALE,
            extra_z=PERSONNEL_FLOOR_LANE_Z_OFFSET,
            with_collision=False,
            use_texture=True,
        )
        spawned += 1

    return {
        "personnel_floor_lane_enabled": spawned > 0,
        "personnel_floor_lane_model": lane_model,
        "personnel_floor_lane_tiles": spawned,
    }


def _loading_marker_xy_size(size_pair, loading_side):
    depth = float(size_pair[0])
    span = float(size_pair[1])
    if loading_side in ("north", "south"):
        return span, depth
    return depth, span


def _rect_bounds(cx, cy, sx, sy):
    return (cx - sx * 0.5, cx + sx * 0.5, cy - sy * 0.5, cy + sy * 0.5)


def _candidate_rect_bounds(candidate):
    cached = candidate.get("_rect_bounds")
    if cached is None:
        cached = _rect_bounds(candidate["cx"], candidate["cy"], candidate["sx"], candidate["sy"])
        candidate["_rect_bounds"] = cached
    return cached


def _rects_overlap(a, b, gap):
    a_bounds = a.get("_rect_bounds")
    if a_bounds is None:
        a_bounds = _candidate_rect_bounds(a)
    b_bounds = b.get("_rect_bounds")
    if b_bounds is None:
        b_bounds = _candidate_rect_bounds(b)
    a_min_x, a_max_x, a_min_y, a_max_y = a_bounds
    b_min_x, b_max_x, b_min_y, b_max_y = b_bounds
    return not (
        (a_max_x + gap) <= b_min_x
        or (b_max_x + gap) <= a_min_x
        or (a_max_y + gap) <= b_min_y
        or (b_max_y + gap) <= a_min_y
    )


def _size_fits_half_span(sx, sy, half_x, half_y, margin):
    max_sx = (2.0 * (half_x - margin))
    max_sy = (2.0 * (half_y - margin))
    return float(sx) <= max_sx + 1e-6 and float(sy) <= max_sy + 1e-6


def _sample_random_center(rng, sx, sy, floor_half_x, floor_half_y, margin):
    min_x = -floor_half_x + margin + (sx * 0.5)
    max_x = floor_half_x - margin - (sx * 0.5)
    min_y = -floor_half_y + margin + (sy * 0.5)
    max_y = floor_half_y - margin - (sy * 0.5)
    if max_x < min_x or max_y < min_y:
        return 0.0, 0.0
    return rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)


def _wall_along_limits(wall, sx, sy, half_x, half_y, margin):
    if wall in ("north", "south"):
        return (
            -half_x + margin + (sx * 0.5),
            half_x - margin - (sx * 0.5),
        )
    return (
        -half_y + margin + (sy * 0.5),
        half_y - margin - (sy * 0.5),
    )


def _wall_attached_center(wall, along, sx, sy, half_x, half_y, margin):
    if wall == "north":
        return along, half_y - margin - (sy * 0.5)
    if wall == "south":
        return along, -half_y + margin + (sy * 0.5)
    if wall == "east":
        return half_x - margin - (sx * 0.5), along
    if wall == "west":
        return -half_x + margin + (sx * 0.5), along
    raise ValueError(f"Unknown wall: {wall}")


def _orient_dims_long_side_on_wall(wall, sx, sy):
    if wall in ("north", "south"):
        return (max(sx, sy), min(sx, sy))
    return (min(sx, sy), max(sx, sy))


def build_area_layout_markers(loader, floor_top_z, wall_info, seed):
    rng = random.Random(int(seed) + 991)
    floor_half_x, floor_half_y = _floor_spawn_half_extents(loader)

    area_defs = {a["name"]: a for a in AREA_LAYOUT_BLOCKS}
    placed = []

    wall_thickness = max(0.0, float(wall_info.get("wall_thickness", 0.0)))
    attach_inset = wall_thickness * float(AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR)
    attach_half_x = min(
        floor_half_x,
        max(1.0, (WAREHOUSE_SIZE_X * 0.5) - attach_inset),
    )
    attach_half_y = min(
        floor_half_y,
        max(1.0, (WAREHOUSE_SIZE_Y * 0.5) - attach_inset),
    )
    personnel_clear_rect = None
    personnel_side = wall_info.get("personnel_side")
    personnel_along = float(wall_info.get("personnel_along", 0.0))
    personnel_span = max(0.0, float(wall_info.get("personnel_door_span", 0.0)))
    if personnel_side in WALL_SLOTS and personnel_span > 0.0:
        if personnel_side in ("north", "south"):
            clear_sx = personnel_span + (2.0 * PERSONNEL_DOOR_CLEAR_EXTRA_ALONG)
            clear_sy = PERSONNEL_DOOR_CLEAR_DEPTH
        else:
            clear_sx = PERSONNEL_DOOR_CLEAR_DEPTH
            clear_sy = personnel_span + (2.0 * PERSONNEL_DOOR_CLEAR_EXTRA_ALONG)
        clear_cx, clear_cy = _wall_attached_center(
            personnel_side,
            personnel_along,
            clear_sx,
            clear_sy,
            attach_half_x,
            attach_half_y,
            AREA_LAYOUT_EDGE_MARGIN,
        )
        personnel_clear_rect = {
            "name": "_PERSONNEL_CLEAR",
            "sx": clear_sx,
            "sy": clear_sy,
            "cx": clear_cx,
            "cy": clear_cy,
            "rgba": (0.0, 0.0, 0.0, 0.0),
        }
    critical_zone_keepout_rect = None
    if personnel_clear_rect is not None:
        critical_zone_clear_depth = max(PERSONNEL_DOOR_CLEAR_DEPTH + 3.5, 9.5)
        critical_zone_extra_along = PERSONNEL_DOOR_CLEAR_EXTRA_ALONG + 2.2
        if personnel_side in ("north", "south"):
            kz_sx = personnel_span + (2.0 * critical_zone_extra_along)
            kz_sy = critical_zone_clear_depth
        else:
            kz_sx = critical_zone_clear_depth
            kz_sy = personnel_span + (2.0 * critical_zone_extra_along)
        kz_cx, kz_cy = _wall_attached_center(
            personnel_side,
            personnel_along,
            kz_sx,
            kz_sy,
            attach_half_x,
            attach_half_y,
            AREA_LAYOUT_EDGE_MARGIN,
        )
        critical_zone_keepout_rect = {
            "name": "_PERSONNEL_CRITICAL_KEEPOUT",
            "sx": kz_sx,
            "sy": kz_sy,
            "cx": kz_cx,
            "cy": kz_cy,
            "rgba": (0.0, 0.0, 0.0, 0.0),
        }
    office_passage_keepout_rect = None
    if personnel_clear_rect is not None:
        office_clear_depth = max(PERSONNEL_DOOR_CLEAR_DEPTH + 5.0, 11.0)
        office_extra_along = PERSONNEL_DOOR_CLEAR_EXTRA_ALONG + 2.8
        if personnel_side in ("north", "south"):
            ok_sx = personnel_span + (2.0 * office_extra_along)
            ok_sy = office_clear_depth
        else:
            ok_sx = office_clear_depth
            ok_sy = personnel_span + (2.0 * office_extra_along)
        ok_cx, ok_cy = _wall_attached_center(
            personnel_side,
            personnel_along,
            ok_sx,
            ok_sy,
            attach_half_x,
            attach_half_y,
            AREA_LAYOUT_EDGE_MARGIN,
        )
        office_passage_keepout_rect = {
            "name": "_OFFICE_DOORWAY_PASSAGE_KEEPOUT",
            "sx": ok_sx,
            "sy": ok_sy,
            "cx": ok_cx,
            "cy": ok_cy,
            "rgba": (0.0, 0.0, 0.0, 0.0),
        }

    critical_door_blocking_zones = {"LOADING", "STORAGE", "FACTORY"}
    major_zone_fixed_short_side_m = {
        zone_name: float(min(area_defs[zone_name]["size_m"]))
        for zone_name in ("LOADING", "STORAGE", "FACTORY")
        if zone_name in area_defs
    }
    opposite_personnel_side = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }.get(personnel_side)
    major_zones = {"LOADING", "STORAGE", "FACTORY"}
    transverse_major_zone = str(wall_info.get("transverse_major_zone", "LOADING")).upper()
    if transverse_major_zone not in major_zones:
        transverse_major_zone = "LOADING"
    longitudinal_major_zones = set(major_zones) - {transverse_major_zone}
    utility_longitudinal_zones = {"STORAGE", "FACTORY"} & longitudinal_major_zones
    if personnel_side in ("east", "west"):
        longitudinal_side_walls = ("north", "south")
    elif personnel_side in ("north", "south"):
        longitudinal_side_walls = ("east", "west")
    else:
        longitudinal_side_walls = tuple(WALL_SLOTS)
    longitudinal_side_walls = tuple(w for w in longitudinal_side_walls if w in WALL_SLOTS)
    transverse_end_only_strip = None
    if opposite_personnel_side in WALL_SLOTS:
        strip_depth = max(22.5, min(26.0, max(major_zone_fixed_short_side_m.values(), default=22.5)))
        if opposite_personnel_side == "north":
            strip_sx = max(1.0, attach_half_x * 2.0)
            strip_sy = strip_depth
            strip_cx = 0.0
            strip_cy = attach_half_y - (strip_sy * 0.5)
        elif opposite_personnel_side == "south":
            strip_sx = max(1.0, attach_half_x * 2.0)
            strip_sy = strip_depth
            strip_cx = 0.0
            strip_cy = -attach_half_y + (strip_sy * 0.5)
        elif opposite_personnel_side == "east":
            strip_sx = strip_depth
            strip_sy = max(1.0, attach_half_y * 2.0)
            strip_cx = attach_half_x - (strip_sx * 0.5)
            strip_cy = 0.0
        else:
            strip_sx = strip_depth
            strip_sy = max(1.0, attach_half_y * 2.0)
            strip_cx = -attach_half_x + (strip_sx * 0.5)
            strip_cy = 0.0
        transverse_end_only_strip = {
            "name": "_TRANSVERSE_END_ONLY_STRIP",
            "sx": strip_sx,
            "sy": strip_sy,
            "cx": strip_cx,
            "cy": strip_cy,
            "rgba": (0.0, 0.0, 0.0, 0.0),
        }

    def _scaled_dims_for_zone(name, base_sx, base_sy, shrink):
        sx0 = float(base_sx)
        sy0 = float(base_sy)
        k = max(0.01, float(shrink))
        zone_key = str(name).upper()
        fixed_short = major_zone_fixed_short_side_m.get(zone_key)
        if fixed_short is None:
            return sx0 * k, sy0 * k
        long_base = max(sx0, sy0)
        long_try = max(fixed_short, long_base * k)
        if sx0 <= sy0:
            return fixed_short, long_try
        return long_try, fixed_short

    def _candidate_attached_wall(candidate):
        cached = candidate.get("_attached_wall")
        if cached is not None:
            return cached
        attached_wall = _attached_wall_from_area_bounds(
            float(candidate.get("sx", 0.0)),
            float(candidate.get("sy", 0.0)),
            float(candidate.get("cx", 0.0)),
            float(candidate.get("cy", 0.0)),
        )
        candidate["_attached_wall"] = attached_wall
        return attached_wall

    def _make_candidate(name, sx, sy, cx, cy, color):
        sx_f = float(sx)
        sy_f = float(sy)
        cx_f = float(cx)
        cy_f = float(cy)
        fits_attach_span = _size_fits_half_span(
            sx_f,
            sy_f,
            attach_half_x,
            attach_half_y,
            AREA_LAYOUT_EDGE_MARGIN,
        )
        cand = {
            "name": name,
            "sx": sx_f,
            "sy": sy_f,
            "cx": cx_f,
            "cy": cy_f,
            "rgba": color,
            "_fits_attach_span": fits_attach_span,
        }
        cand["_rect_bounds"] = _rect_bounds(cx_f, cy_f, sx_f, sy_f)
        cand["_attached_wall"] = _attached_wall_from_area_bounds(sx_f, sy_f, cx_f, cy_f)
        return cand

    def _is_far_from_personnel_door_on_same_wall(candidate):
        if personnel_side not in WALL_SLOTS:
            return True
        attached_wall = _candidate_attached_wall(candidate)
        if attached_wall != personnel_side:
            return True
        if attached_wall in ("north", "south"):
            cand_along = float(candidate.get("cx", 0.0))
            cand_span = float(candidate.get("sx", 0.0))
        else:
            cand_along = float(candidate.get("cy", 0.0))
            cand_span = float(candidate.get("sy", 0.0))
        door_half = (personnel_span * 0.5) + PERSONNEL_DOOR_CLEAR_EXTRA_ALONG + 1.0
        zone_half = cand_span * 0.5
        min_center_distance = door_half + zone_half
        return abs(cand_along - personnel_along) >= (min_center_distance - 1e-6)

    def _opposite_wall_end_targets(wall, sx, sy):
        lo, hi = _wall_along_limits(
            wall,
            float(sx),
            float(sy),
            attach_half_x,
            attach_half_y,
            AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return None
        return float(lo), float(hi)

    def _is_at_preferred_opposite_end(candidate):
        if opposite_personnel_side not in WALL_SLOTS:
            return True
        attached_wall = _candidate_attached_wall(candidate)
        if attached_wall != opposite_personnel_side:
            return False
        if attached_wall in ("north", "south"):
            cand_along = float(candidate.get("cx", 0.0))
            cand_span = float(candidate.get("sx", 0.0))
        else:
            cand_along = float(candidate.get("cy", 0.0))
            cand_span = float(candidate.get("sy", 0.0))
        end_targets = _opposite_wall_end_targets(
            attached_wall,
            float(candidate.get("sx", 0.0)),
            float(candidate.get("sy", 0.0)),
        )
        if end_targets is None:
            return False
        target_lo, target_hi = end_targets
        end_tol = max(0.45, min(2.6, cand_span * 0.12))
        return abs(cand_along - target_lo) <= end_tol or abs(cand_along - target_hi) <= end_tol

    def _is_at_wall_end(candidate, end_tol_factor=0.16):
        attached_wall = _candidate_attached_wall(candidate)
        if attached_wall in ("north", "south"):
            cand_along = float(candidate.get("cx", 0.0))
            cand_span = float(candidate.get("sx", 0.0))
        else:
            cand_along = float(candidate.get("cy", 0.0))
            cand_span = float(candidate.get("sy", 0.0))
        lo, hi = _wall_along_limits(
            attached_wall,
            float(candidate.get("sx", 0.0)),
            float(candidate.get("sy", 0.0)),
            attach_half_x,
            attach_half_y,
            AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return False
        end_tol = max(0.9, min(4.2, cand_span * float(end_tol_factor)))
        return abs(cand_along - lo) <= end_tol or abs(cand_along - hi) <= end_tol

    def _can_place_static(candidate, gap):
        fits_attach_span = candidate.get("_fits_attach_span")
        if fits_attach_span is None:
            fits_attach_span = _size_fits_half_span(
                candidate["sx"],
                candidate["sy"],
                attach_half_x,
                attach_half_y,
                AREA_LAYOUT_EDGE_MARGIN,
            )
            candidate["_fits_attach_span"] = fits_attach_span
        if not fits_attach_span:
            return False
        name = str(candidate.get("name", ""))
        if name not in critical_door_blocking_zones:
            return True
        attached_wall = _candidate_attached_wall(candidate)
        if transverse_end_only_strip is not None and name != transverse_major_zone:
            if _rects_overlap(candidate, transverse_end_only_strip, 0.0):
                return False
        if name == transverse_major_zone:
            if opposite_personnel_side in WALL_SLOTS and attached_wall != opposite_personnel_side:
                return False
        elif name in longitudinal_major_zones and longitudinal_side_walls:
            if attached_wall not in longitudinal_side_walls:
                return False
        if critical_zone_keepout_rect is not None and _rects_overlap(candidate, critical_zone_keepout_rect, 0.0):
            if attached_wall in (personnel_side, opposite_personnel_side):
                return False
        if attached_wall in (personnel_side, opposite_personnel_side):
            if not _is_far_from_personnel_door_on_same_wall(candidate):
                return False
        return True

    def _can_place(candidate, gap):
        fits_attach_span = candidate.get("_fits_attach_span")
        if fits_attach_span is None:
            fits_attach_span = _size_fits_half_span(
                candidate["sx"],
                candidate["sy"],
                attach_half_x,
                attach_half_y,
                AREA_LAYOUT_EDGE_MARGIN,
            )
            candidate["_fits_attach_span"] = fits_attach_span
        if not fits_attach_span:
            return False
        cand_bounds = candidate.get("_rect_bounds")
        if cand_bounds is None:
            cand_bounds = _candidate_rect_bounds(candidate)
        cand_min_x, cand_max_x, cand_min_y, cand_max_y = cand_bounds
        for prev in placed:
            prev_bounds = prev.get("_rect_bounds")
            if prev_bounds is None:
                prev_bounds = _candidate_rect_bounds(prev)
            prev_min_x, prev_max_x, prev_min_y, prev_max_y = prev_bounds
            if not (
                (cand_max_x + gap) <= prev_min_x
                or (prev_max_x + gap) <= cand_min_x
                or (cand_max_y + gap) <= prev_min_y
                or (prev_max_y + gap) <= cand_min_y
            ):
                return False
        name = str(candidate.get("name", ""))
        if name in critical_door_blocking_zones:
            attached_wall = _candidate_attached_wall(candidate)
            if transverse_end_only_strip is not None and name != transverse_major_zone:
                if _rects_overlap(candidate, transverse_end_only_strip, 0.0):
                    return False
            if name == transverse_major_zone:
                if opposite_personnel_side in WALL_SLOTS and attached_wall != opposite_personnel_side:
                    return False
            elif name in longitudinal_major_zones and longitudinal_side_walls:
                if attached_wall not in longitudinal_side_walls:
                    return False
                for prev in placed:
                    prev_name = str(prev.get("name", ""))
                    if prev_name not in longitudinal_major_zones or prev_name == name:
                        continue
                    prev_wall = _candidate_attached_wall(prev)
                    if prev_wall == attached_wall:
                        return False
            if critical_zone_keepout_rect is not None and _rects_overlap(candidate, critical_zone_keepout_rect, 0.0):
                if attached_wall in (personnel_side, opposite_personnel_side):
                    return False
            if attached_wall in (personnel_side, opposite_personnel_side):
                if not _is_far_from_personnel_door_on_same_wall(candidate):
                    return False
        return True

    def _place_on_wall(
        name,
        sx,
        sy,
        wall,
        color,
        along_pref=None,
        tries=600,
        gap=AREA_LAYOUT_MIN_GAP,
        validator=None,
        deterministic_first=True,
    ):
        sx, sy = float(sx), float(sy)
        sx, sy = _orient_dims_long_side_on_wall(wall, sx, sy)
        if not _size_fits_half_span(sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN):
            return None
        lo, hi = _wall_along_limits(wall, sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN)
        if hi < lo:
            return None
        anchors = [
            lo,
            hi,
            0.5 * (lo + hi),
            max(lo, min(hi, 0.0)),
        ]
        if along_pref is not None:
            anchors.append(max(lo, min(hi, along_pref)))
        def _try_along(along_value):
            cx, cy = _wall_attached_center(
                wall,
                along_value,
                sx,
                sy,
                attach_half_x,
                attach_half_y,
                AREA_LAYOUT_EDGE_MARGIN,
            )
            cand = _make_candidate(name, sx, sy, cx, cy, color)
            if not _can_place(cand, gap):
                return None
            if validator is not None and not validator(cand):
                return None
            return cand

        if deterministic_first:
            for along in anchors:
                cand = _try_along(along)
                if cand is not None:
                    return cand
            for _ in range(max(0, int(tries))):
                cand = _try_along(rng.uniform(lo, hi))
                if cand is not None:
                    return cand
        else:
            for _ in range(max(0, int(tries))):
                cand = _try_along(rng.uniform(lo, hi))
                if cand is not None:
                    return cand
            for along in anchors:
                cand = _try_along(along)
                if cand is not None:
                    return cand
        return None

    def _place_anywhere(name, sx, sy, color, tries=1200, gap=AREA_LAYOUT_MIN_GAP, validator=None):
        sx, sy = float(sx), float(sy)
        if not _size_fits_half_span(sx, sy, floor_half_x, floor_half_y, AREA_LAYOUT_EDGE_MARGIN):
            return None
        for _ in range(max(1, tries)):
            cx, cy = _sample_random_center(
                rng,
                sx,
                sy,
                floor_half_x,
                floor_half_y,
                AREA_LAYOUT_EDGE_MARGIN,
            )
            cand = _make_candidate(name, sx, sy, cx, cy, color)
            if _can_place(cand, gap):
                if validator is not None and not validator(cand):
                    continue
                return cand
        return None

    def _force_place_zone(name, base_sx, base_sy, color, preferred_walls=None, along_pref=None, validator=None):
        if preferred_walls is None:
            preferred_walls = list(WALL_SLOTS)
        sx = float(base_sx)
        sy = float(base_sy)
        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for wall in preferred_walls:
                picked = _place_on_wall(
                    name=name,
                    sx=sx,
                    sy=sy,
                    wall=wall,
                    color=color,
                    along_pref=along_pref,
                    tries=320,
                    gap=gap,
                    validator=validator,
                )
                if picked is not None:
                    return picked
            picked = _place_anywhere(
                name=name,
                sx=sx,
                sy=sy,
                color=color,
                tries=600,
                gap=gap,
                validator=validator,
            )
            if picked is not None:
                return picked
        raise ValueError(
            f"Unable to place zone '{name}' at fixed size {sx:.2f} x {sy:.2f} m "
            f"without overlap/out-of-bounds."
        )

    loading_def = area_defs["LOADING"]
    loading_side = wall_info.get("loading_side", "north")
    door_centers = wall_info.get("door_centers", [])
    frame_yaw_for_span = dock_inward_yaw_for_slot(loading_side)
    frame_ex, frame_ey = oriented_xy_size(
        loader,
        CONVEYOR_ASSETS["dock_frame"],
        UNIFORM_SCALE,
        frame_yaw_for_span,
    )
    door_span = frame_ex if loading_side in ("north", "south") else frame_ey
    load_sx, load_sy = _loading_marker_xy_size(loading_def["size_m"], loading_side)
    along_pref = (sum(door_centers) / float(len(door_centers))) if door_centers else 0.0

    def _loading_zone_covers_doors(candidate):
        if not door_centers:
            return True
        if loading_side in ("north", "south"):
            along_center = float(candidate["cx"])
            along_half = float(candidate["sx"]) * 0.5
        else:
            along_center = float(candidate["cy"])
            along_half = float(candidate["sy"]) * 0.5
        door_half = max(0.0, float(door_span) * 0.5)
        min_along = along_center - along_half + door_half
        max_along = along_center + along_half - door_half
        return all((min_along - 1e-6) <= c <= (max_along + 1e-6) for c in door_centers)

    fit_sx, fit_sy = _orient_dims_long_side_on_wall(loading_side, load_sx, load_sy)
    lo, hi = _wall_along_limits(
        loading_side,
        fit_sx,
        fit_sy,
        attach_half_x,
        attach_half_y,
        AREA_LAYOUT_EDGE_MARGIN,
    )
    loading_candidate = None
    if hi >= lo:
        corner_pref = str(wall_info.get("loading_corner_side", "")).lower()
        if corner_pref == "left":
            corner_alongs = [lo, hi]
        elif corner_pref == "right":
            corner_alongs = [hi, lo]
        else:
            center_from_doors = 0.5 * (min(door_centers) + max(door_centers)) if door_centers else 0.0
            corner_alongs = [lo, hi] if center_from_doors <= 0.0 else [hi, lo]

        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for along in corner_alongs:
                cx, cy = _wall_attached_center(
                    loading_side,
                    along,
                    fit_sx,
                    fit_sy,
                    attach_half_x,
                    attach_half_y,
                    AREA_LAYOUT_EDGE_MARGIN,
                )
                cand = _make_candidate("LOADING", fit_sx, fit_sy, cx, cy, loading_def["rgba"])
                if _can_place(cand, gap) and _loading_zone_covers_doors(cand):
                    loading_candidate = cand
                    break
            if loading_candidate is not None:
                break

        if loading_candidate is None:
            preferred_end = corner_alongs[0] if corner_alongs else hi
            span = max(0.0, hi - lo)
            sweep_steps = max(36, int(span / 0.35))
            sweep_alongs = []
            for i in range(sweep_steps + 1):
                t = float(i) / float(sweep_steps) if sweep_steps > 0 else 0.0
                sweep_alongs.append(float(lo + ((hi - lo) * t)))
            sweep_alongs = list(
                sorted(
                    sweep_alongs,
                    key=lambda a: abs(float(a) - float(preferred_end)),
                )
            )
            for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
                for along in sweep_alongs:
                    cx, cy = _wall_attached_center(
                        loading_side,
                        along,
                        fit_sx,
                        fit_sy,
                        attach_half_x,
                        attach_half_y,
                        AREA_LAYOUT_EDGE_MARGIN,
                    )
                    cand = _make_candidate("LOADING", fit_sx, fit_sy, cx, cy, loading_def["rgba"])
                    if _can_place(cand, gap) and _loading_zone_covers_doors(cand):
                        loading_candidate = cand
                        break
                if loading_candidate is not None:
                    break

    if loading_candidate is None:
        raise ValueError(
            "Unable to place LOADING in a corner while covering all 3 dock doors."
        )
    if loading_candidate is not None:
        placed.append(loading_candidate)

    office_def = area_defs["OFFICE"]
    fit_sx, fit_sy = float(office_def["size_m"][0]), float(office_def["size_m"][1])
    office_wall_priority = [personnel_side] if personnel_side in WALL_SLOTS else list(WALL_SLOTS)

    def _office_along_for_wall(wall_name):
        if wall_name in ("north", "south"):
            if personnel_side == "east":
                return float(max(0.0, attach_half_x - (fit_sx * 0.5)))
            if personnel_side == "west":
                return float(-max(0.0, attach_half_x - (fit_sx * 0.5)))
            return float(personnel_along if personnel_side in ("north", "south") else 0.0)
        if personnel_side == "north":
            return float(max(0.0, attach_half_y - (fit_sy * 0.5)))
        if personnel_side == "south":
            return float(-max(0.0, attach_half_y - (fit_sy * 0.5)))
        return float(personnel_along if personnel_side in ("east", "west") else 0.0)

    def _office_pref_alongs_for_wall(wall_name):
        along_center = _office_along_for_wall(wall_name)
        if wall_name in ("north", "south"):
            door_along_span = float(personnel_span if personnel_side in ("north", "south") else 0.0)
        else:
            door_along_span = float(personnel_span if personnel_side in ("east", "west") else 0.0)
        office_along_span = fit_sx if wall_name in ("north", "south") else fit_sy
        door_clear_offset = (
            (door_along_span * 0.5)
            + (office_along_span * 0.5)
            + float(PERSONNEL_DOOR_CLEAR_EXTRA_ALONG)
            + 0.9
        )
        if wall_name == personnel_side:
            return [
                along_center,
                along_center - door_clear_offset,
                along_center + door_clear_offset,
                0.0,
            ]
        return [
            along_center - door_clear_offset,
            along_center + door_clear_offset,
            along_center,
            0.0,
        ]

    def _office_sweep_alongs_for_wall(wall_name):
        sx_o, sy_o = _orient_dims_long_side_on_wall(wall_name, fit_sx, fit_sy)
        lo, hi = _wall_along_limits(
            wall_name,
            sx_o,
            sy_o,
            attach_half_x,
            attach_half_y,
            AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return []
        span = max(0.0, hi - lo)
        steps = max(12, int(span / 0.9))
        out = []
        if steps <= 0:
            return [float(lo)]
        for i in range(steps + 1):
            t = float(i) / float(steps)
            out.append(float(lo + (span * t)))
        return out

    def _office_matches_personnel_side(candidate):
        if personnel_side not in WALL_SLOTS:
            return True
        attached_wall = _candidate_attached_wall(candidate)
        if attached_wall != personnel_side:
            return False
        if not _is_far_from_personnel_door_on_same_wall(candidate):
            return False
        if office_passage_keepout_rect is not None and _rects_overlap(candidate, office_passage_keepout_rect, 0.0):
            return False
        return True

    office_candidates = []
    office_seen = set()

    def _push_office_candidate(cand):
        if cand is None:
            return
        key = (
            round(float(cand["cx"]), 4),
            round(float(cand["cy"]), 4),
            round(float(cand["sx"]), 4),
            round(float(cand["sy"]), 4),
        )
        if key in office_seen:
            return
        office_seen.add(key)
        office_candidates.append(cand)

    for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
        for wall in office_wall_priority:
            along_candidates = list(_office_pref_alongs_for_wall(wall)) + list(_office_sweep_alongs_for_wall(wall))
            for along_pref in along_candidates:
                _push_office_candidate(
                    _place_on_wall(
                        name="OFFICE",
                        sx=fit_sx,
                        sy=fit_sy,
                        wall=wall,
                        color=office_def["rgba"],
                        along_pref=along_pref,
                        tries=220,
                        gap=gap,
                        validator=_office_matches_personnel_side,
                        deterministic_first=True,
                    )
                )
    for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
        for wall in office_wall_priority:
            _push_office_candidate(
                _place_on_wall(
                    name="OFFICE",
                    sx=fit_sx,
                    sy=fit_sy,
                    wall=wall,
                    color=office_def["rgba"],
                    along_pref=None,
                    tries=320,
                    gap=gap,
                    validator=_office_matches_personnel_side,
                    deterministic_first=False,
                )
            )
    _push_office_candidate(
        _place_anywhere(
            name="OFFICE",
            sx=fit_sx,
            sy=fit_sy,
            color=office_def["rgba"],
            tries=600,
            gap=0.0,
            validator=_office_matches_personnel_side,
        )
    )
    if not office_candidates:
        raise ValueError(
            "Unable to place OFFICE near personnel door side without overlap/out-of-bounds."
        )

    utility_zones = ["FACTORY", "STORAGE"]
    optional_utility_zones = []
    if ENABLE_FORKLIFT_PARKING:
        optional_utility_zones.append("FORKLIFT_PARK")
    if ENABLE_MACHINING_CELL_LAYOUT:
        optional_utility_zones.append("MACHINING_CELL")
    utility_zones = list(
        sorted(
            utility_zones,
            key=lambda zn: float(area_defs[zn]["size_m"][0]) * float(area_defs[zn]["size_m"][1]),
            reverse=True,
        )
    )
    optional_utility_zones = list(
        sorted(
            optional_utility_zones,
            key=lambda zn: float(area_defs[zn]["size_m"][0]) * float(area_defs[zn]["size_m"][1]),
            reverse=True,
        )
    )

    def _utility_wall_priority():
        walls = list(WALL_SLOTS)
        rng.shuffle(walls)
        walls.sort(
            key=lambda w: (
                1 if w == loading_side else 0,
                1 if w == personnel_side else 0,
            )
        )
        return walls

    def _utility_candidate_pool(name, gap):
        area = area_defs[name]
        base_sx = float(area["size_m"][0])
        base_sy = float(area["size_m"][1])
        zone_key = str(name).upper()
        color = area["rgba"]
        seen = set()
        out = []
        max_candidates = 56 if zone_key in utility_longitudinal_zones else 48

        wall_order = list(_utility_wall_priority())
        if zone_key == transverse_major_zone and opposite_personnel_side in WALL_SLOTS:
            wall_order = [opposite_personnel_side]
        elif zone_key in utility_longitudinal_zones and longitudinal_side_walls:
            wall_order = list(longitudinal_side_walls)

        for wall in wall_order:
            sx, sy = _orient_dims_long_side_on_wall(wall, base_sx, base_sy)
            if not _size_fits_half_span(sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN):
                continue
            lo, hi = _wall_along_limits(wall, sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN)
            if hi < lo:
                continue
            span = max(0.0, hi - lo)
            if zone_key in utility_longitudinal_zones:
                end_band = max(1.6, min(span, 8.5))
                lo_band_hi = min(hi, lo + end_band)
                hi_band_lo = max(lo, hi - end_band)
                alongs = [lo, hi, lo_band_hi, hi_band_lo]
                sweep_steps = max(5, int(end_band / 1.1))
                for i in range(sweep_steps + 1):
                    t = float(i) / float(sweep_steps) if sweep_steps > 0 else 0.0
                    alongs.append(float(lo + ((lo_band_hi - lo) * t)))
                    alongs.append(float(hi_band_lo + ((hi - hi_band_lo) * t)))
                for _ in range(14):
                    alongs.append(rng.uniform(lo, lo_band_hi))
                    alongs.append(rng.uniform(hi_band_lo, hi))
            else:
                if zone_key == transverse_major_zone:
                    alongs = [
                        0.5 * (lo + hi),
                        max(lo, min(hi, 0.0)),
                        lo,
                        hi,
                    ]
                else:
                    alongs = [
                        lo,
                        hi,
                        0.5 * (lo + hi),
                        max(lo, min(hi, 0.0)),
                    ]
                sweep_steps = max(6, int(span / 2.5))
                for i in range(sweep_steps + 1):
                    t = float(i) / float(sweep_steps) if sweep_steps > 0 else 0.0
                    alongs.append(float(lo + ((hi - lo) * t)))
                for _ in range(20):
                    alongs.append(rng.uniform(lo, hi))
            for along in alongs:
                cx, cy = _wall_attached_center(
                    wall,
                    along,
                    sx,
                    sy,
                    attach_half_x,
                    attach_half_y,
                    AREA_LAYOUT_EDGE_MARGIN,
                )
                cand = _make_candidate(name, sx, sy, cx, cy, color)
                key = (round(cx, 4), round(cy, 4), round(sx, 4), round(sy, 4))
                if key in seen:
                    continue
                if not _can_place_static(cand, gap):
                    continue
                seen.add(key)
                out.append(cand)
                if len(out) >= max_candidates:
                    return out

        if zone_key in ("STORAGE", "FACTORY"):
            return out

        if _size_fits_half_span(base_sx, base_sy, floor_half_x, floor_half_y, AREA_LAYOUT_EDGE_MARGIN):
            center_cand = _make_candidate(name, base_sx, base_sy, 0.0, 0.0, color)
            key = (
                round(center_cand["cx"], 4),
                round(center_cand["cy"], 4),
                round(center_cand["sx"], 4),
                round(center_cand["sy"], 4),
            )
            if key not in seen and _can_place_static(center_cand, gap):
                seen.add(key)
                out.append(center_cand)
            for _ in range(40):
                cx, cy = _sample_random_center(
                    rng,
                    base_sx,
                    base_sy,
                    floor_half_x,
                    floor_half_y,
                    AREA_LAYOUT_EDGE_MARGIN,
                )
                cand = _make_candidate(name, base_sx, base_sy, cx, cy, color)
                key = (round(cx, 4), round(cy, 4), round(base_sx, 4), round(base_sy, 4))
                if key in seen:
                    continue
                if not _can_place_static(cand, gap):
                    continue
                seen.add(key)
                out.append(cand)
                if len(out) >= max_candidates:
                    break
        return out

    utility_pool_cache = {}
    utility_orders = list(itertools.permutations(utility_zones))
    rng.shuffle(utility_orders)
    base_order = tuple(utility_zones)
    if base_order in utility_orders:
        utility_orders.remove(base_order)
    utility_orders = [base_order] + utility_orders

    def _place_all_utilities_for_current_state():
        def _pool(name, gap):
            key = (name, float(gap))
            if key not in utility_pool_cache:
                utility_pool_cache[key] = _utility_candidate_pool(name, gap)
            return utility_pool_cache[key]

        def _dfs_place(order, idx, gap):
            if idx >= len(order):
                return True
            name = order[idx]
            for cand in _pool(name, gap):
                if not _can_place(cand, gap):
                    continue
                placed.append(cand)
                if _dfs_place(order, idx + 1, gap):
                    return True
                placed.pop()
            return False

        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for order in utility_orders:
                snapshot = len(placed)
                if _dfs_place(order, 0, gap):
                    return True
                del placed[snapshot:]
        return False

    office_and_utilities_placed = False
    for office_candidate in office_candidates:
        snapshot = len(placed)
        placed.append(office_candidate)
        if _place_all_utilities_for_current_state():
            office_and_utilities_placed = True
            break
        del placed[snapshot:]

    if not office_and_utilities_placed:
        extra_candidates = []
        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for wall in office_wall_priority:
                for _ in range(6):
                    cand = _place_on_wall(
                        name="OFFICE",
                        sx=fit_sx,
                        sy=fit_sy,
                        wall=wall,
                        color=office_def["rgba"],
                        along_pref=None,
                        tries=280,
                        gap=gap,
                        validator=_office_matches_personnel_side,
                        deterministic_first=False,
                    )
                    if cand is not None:
                        extra_candidates.append(cand)
        for cand in extra_candidates:
            snapshot = len(placed)
            placed.append(cand)
            if _place_all_utilities_for_current_state():
                office_and_utilities_placed = True
                break
            del placed[snapshot:]

    if not office_and_utilities_placed:
        salvage_office_candidates = list(office_candidates)
        for cand in extra_candidates:
            salvage_office_candidates.append(cand)

        for strict_storage_shape in (True, False):
            for office_candidate in salvage_office_candidates:
                snapshot = len(placed)
                placed.append(office_candidate)
                salvage_ok = True
                for name in utility_zones:
                    if any(a.get("name") == name for a in placed):
                        continue
                    area = area_defs[name]
                    base_sx = float(area["size_m"][0])
                    base_sy = float(area["size_m"][1])
                    picked = None
                    if str(name).upper() == "STORAGE":
                        if strict_storage_shape:
                            shrink_steps = (1.0, 0.97, 0.94, 0.90, 0.86, 0.82)
                        else:
                            shrink_steps = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45)
                    else:
                        shrink_steps = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30)
                    for shrink in shrink_steps:
                        sx_try, sy_try = _scaled_dims_for_zone(name, base_sx, base_sy, shrink)
                        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
                            wall_candidates = list(_utility_wall_priority())
                            zone_key = str(name).upper()
                            if zone_key == transverse_major_zone and opposite_personnel_side in WALL_SLOTS:
                                wall_candidates = [opposite_personnel_side]
                            elif zone_key in utility_longitudinal_zones and longitudinal_side_walls:
                                wall_candidates = list(longitudinal_side_walls)
                            for wall in wall_candidates:
                                picked = _place_on_wall(
                                    name=name,
                                    sx=sx_try,
                                    sy=sy_try,
                                    wall=wall,
                                    color=area["rgba"],
                                    along_pref=None,
                                    tries=320,
                                    gap=gap,
                                    deterministic_first=False,
                                )
                                if picked is not None:
                                    break
                            if picked is not None:
                                break
                            if zone_key not in utility_longitudinal_zones and zone_key != transverse_major_zone:
                                picked = _place_anywhere(
                                    name=name,
                                    sx=sx_try,
                                    sy=sy_try,
                                    color=area["rgba"],
                                    tries=600,
                                    gap=gap,
                                )
                                if picked is not None:
                                    break
                        if picked is not None:
                            break
                    if picked is None:
                        salvage_ok = False
                        break
                    placed.append(picked)
                if salvage_ok:
                    office_and_utilities_placed = True
                    break
                del placed[snapshot:]
            if office_and_utilities_placed:
                break

    if not office_and_utilities_placed:
        raise ValueError(
            "Unable to place OFFICE + required utility zones while keeping OFFICE near personnel door side."
        )

    for name in optional_utility_zones:
        if any(a.get("name") == name for a in placed):
            continue
        area = area_defs[name]
        base_sx = float(area["size_m"][0])
        base_sy = float(area["size_m"][1])
        require_wall_attachment = (name == "FORKLIFT_PARK")
        is_forklift_park = (str(name).upper() == "FORKLIFT_PARK")
        office_ref = next((a for a in placed if a.get("name") == "OFFICE"), None)
        walls = list(WALL_SLOTS)
        rng.shuffle(walls)
        walls.sort(key=lambda w: 1 if w == loading_side else 0)
        if require_wall_attachment:
            office_wall = None
            if office_ref is not None:
                office_wall = _candidate_attached_wall(office_ref)
            preferred_walls = []
            for w in (office_wall, personnel_side):
                if w in WALL_SLOTS and w not in preferred_walls:
                    preferred_walls.append(w)
            anchor_wall = office_wall if office_wall in WALL_SLOTS else personnel_side
            side_order = {
                "north": ["north", "east", "west", "south"],
                "south": ["south", "east", "west", "north"],
                "east": ["east", "north", "south", "west"],
                "west": ["west", "north", "south", "east"],
            }
            if anchor_wall in side_order:
                ordered = list(side_order[anchor_wall])
                if personnel_side in WALL_SLOTS and personnel_side in ordered:
                    ordered.remove(personnel_side)
                    ordered.insert(1, personnel_side)
                walls = ordered
            elif preferred_walls:
                walls = preferred_walls + [w for w in walls if w not in preferred_walls]

        def _optional_along_pref_for_wall(wall_name):
            if is_forklift_park and wall_name == personnel_side:
                return None
            if office_ref is not None:
                if wall_name in ("north", "south"):
                    return float(office_ref["cx"])
                return float(office_ref["cy"])
            if wall_name == personnel_side:
                return float(personnel_along)
            return None

        office_keepout_gap = 2.0 if require_wall_attachment else 0.0

        def _optional_zone_validator(candidate):
            if require_wall_attachment and office_ref is not None:
                if _rects_overlap(candidate, office_ref, office_keepout_gap):
                    return False
            if is_forklift_park:
                if office_passage_keepout_rect is not None and _rects_overlap(candidate, office_passage_keepout_rect, 0.0):
                    return False
                attached_wall = _candidate_attached_wall(candidate)
                if attached_wall == personnel_side and (not _is_far_from_personnel_door_on_same_wall(candidate)):
                    return False
            return True

        picked = None
        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for wall in walls:
                cand = _place_on_wall(
                    name=name,
                    sx=base_sx,
                    sy=base_sy,
                    wall=wall,
                    color=area["rgba"],
                    along_pref=_optional_along_pref_for_wall(wall),
                    tries=260,
                    gap=gap,
                    validator=_optional_zone_validator,
                    deterministic_first=False,
                )
                if cand is not None:
                    picked = cand
                    break
            if picked is not None:
                break
            if not require_wall_attachment:
                cand = _place_anywhere(
                    name=name,
                    sx=base_sx,
                    sy=base_sy,
                    color=area["rgba"],
                    tries=500,
                    gap=gap,
                    validator=_optional_zone_validator,
                )
                if cand is not None:
                    picked = cand
                    break
        if picked is None and require_wall_attachment:
            long_base = max(base_sx, base_sy)
            short_base = min(base_sx, base_sy)
            for shrink in (0.90, 0.80, 0.72, 0.65, 0.58, 0.50, 0.42):
                long_try = max(6.0, long_base * shrink)
                short_try = max(4.8, short_base)
                for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
                    for wall in walls:
                        if wall in ("north", "south"):
                            sx_try, sy_try = long_try, short_try
                        else:
                            sx_try, sy_try = short_try, long_try
                        cand = _place_on_wall(
                            name=name,
                            sx=sx_try,
                            sy=sy_try,
                            wall=wall,
                            color=area["rgba"],
                            along_pref=_optional_along_pref_for_wall(wall),
                            tries=320,
                            gap=gap,
                            validator=_optional_zone_validator,
                            deterministic_first=False,
                        )
                        if cand is not None:
                            picked = cand
                            break
                    if picked is not None:
                        break
                if picked is not None:
                    break
        if picked is not None:
            placed.append(picked)

    required_names = ["LOADING", "OFFICE"] + utility_zones
    existing_names = {a["name"] for a in placed}
    for name in required_names:
        if name in existing_names:
            continue
        area = area_defs[name]
        if name == "LOADING":
            fallback_walls = [loading_side]
        elif name == transverse_major_zone and opposite_personnel_side in WALL_SLOTS:
            fallback_walls = [opposite_personnel_side]
        elif name in utility_longitudinal_zones and longitudinal_side_walls:
            fallback_walls = list(longitudinal_side_walls)
        else:
            fallback_walls = list(WALL_SLOTS)
        fallback = _force_place_zone(
            name=name,
            base_sx=area["size_m"][0],
            base_sy=area["size_m"][1],
            color=area["rgba"],
            preferred_walls=fallback_walls,
            along_pref=along_pref if name == "LOADING" else None,
        )
        placed.append(fallback)
        existing_names.add(name)

    def _try_center_zone_if_isolated_on_wall(zone_name):
        zone_idx = next((i for i, a in enumerate(placed) if str(a.get("name", "")) == str(zone_name)), None)
        if zone_idx is None:
            return False

        current = placed[zone_idx]
        attached_wall = _candidate_attached_wall(current)
        if attached_wall not in WALL_SLOTS:
            return False

        wall_zone_names = []
        for i, area in enumerate(placed):
            area_name = str(area.get("name", ""))
            if area_name.startswith("_"):
                continue
            if area_name not in area_defs:
                continue
            if _candidate_attached_wall(area) != attached_wall:
                continue
            wall_zone_names.append((i, area_name))
        if len(wall_zone_names) != 1 or wall_zone_names[0][0] != zone_idx:
            return False

        sx = float(current.get("sx", 0.0))
        sy = float(current.get("sy", 0.0))
        lo, hi = _wall_along_limits(
            attached_wall,
            sx,
            sy,
            attach_half_x,
            attach_half_y,
            AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return False

        along_center = 0.5 * (lo + hi)
        along_zero = max(lo, min(hi, 0.0))
        along_targets = [along_center]
        if abs(along_zero - along_center) > 1e-6:
            along_targets.append(along_zero)

        color = current.get("rgba", (0.7, 0.7, 0.7, 0.7))
        placed.pop(zone_idx)
        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for along in along_targets:
                cx, cy = _wall_attached_center(
                    attached_wall,
                    along,
                    sx,
                    sy,
                    attach_half_x,
                    attach_half_y,
                    AREA_LAYOUT_EDGE_MARGIN,
                )
                cand = _make_candidate(str(zone_name), sx, sy, cx, cy, color)
                if not _can_place(cand, gap):
                    continue
                placed.append(cand)
                return True
        placed.insert(zone_idx, current)
        return False

    for zone_name in ("STORAGE", "FACTORY", "LOADING", "FORKLIFT_PARK", "MACHINING_CELL"):
        _try_center_zone_if_isolated_on_wall(zone_name)

    def _try_relocate_longitudinal_zone_to_corner(zone_name):
        zone_idx = next((i for i, a in enumerate(placed) if str(a.get("name", "")) == str(zone_name)), None)
        if zone_idx is None:
            return False
        current = placed[zone_idx]
        current_wall = _candidate_attached_wall(current)
        if current_wall not in longitudinal_side_walls:
            return False

        placed.pop(zone_idx)
        base_sx = float(current.get("sx", 0.0))
        base_sy = float(current.get("sy", 0.0))
        color = current.get("rgba", (0.7, 0.7, 0.7, 0.7))
        wall_order = [current_wall] + [w for w in longitudinal_side_walls if w != current_wall]
        if str(zone_name).upper() == "FACTORY":
            shrink_steps = (1.0, 0.95, 0.90, 0.85, 0.80)
        else:
            shrink_steps = (1.0, 0.95, 0.90, 0.85)

        for shrink in shrink_steps:
            sx_try, sy_try = _scaled_dims_for_zone(zone_name, base_sx, base_sy, shrink)
            for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
                for wall in wall_order:
                    sx_on_wall, sy_on_wall = _orient_dims_long_side_on_wall(wall, sx_try, sy_try)
                    lo, hi = _wall_along_limits(
                        wall,
                        sx_on_wall,
                        sy_on_wall,
                        attach_half_x,
                        attach_half_y,
                        AREA_LAYOUT_EDGE_MARGIN,
                    )
                if hi < lo:
                    continue
                span = max(0.0, hi - lo)
                end_band = max(1.2, min(span, 6.5))
                if wall in ("north", "south"):
                    current_along = float(current.get("cx", 0.0))
                else:
                    current_along = float(current.get("cy", 0.0))
                if abs(current_along - lo) <= abs(current_along - hi):
                    end_first, end_second = lo, hi
                else:
                    end_first, end_second = hi, lo
                along_prefs = (
                    end_first,
                    end_second,
                    end_first + (0.5 * end_band) if end_first <= end_second else end_first - (0.5 * end_band),
                    end_second - (0.5 * end_band) if end_first <= end_second else end_second + (0.5 * end_band),
                )
                for along_pref in along_prefs:
                    cand = _place_on_wall(
                        name=str(zone_name),
                        sx=sx_try,
                        sy=sy_try,
                        wall=wall,
                        color=color,
                        along_pref=along_pref,
                        tries=0,
                        gap=gap,
                        deterministic_first=True,
                    )
                    if cand is None:
                        continue
                    if not _is_at_wall_end(cand):
                        continue
                    placed.append(cand)
                    return True

        placed.insert(zone_idx, current)
        return False

    def _snap_longitudinal_zone_to_exact_end(zone_name):
        zone_idx = next((i for i, a in enumerate(placed) if str(a.get("name", "")) == str(zone_name)), None)
        if zone_idx is None:
            return False
        current = placed[zone_idx]
        current_wall = _candidate_attached_wall(current)
        if current_wall not in longitudinal_side_walls:
            return False

        sx = float(current.get("sx", 0.0))
        sy = float(current.get("sy", 0.0))
        lo, hi = _wall_along_limits(
            current_wall,
            sx,
            sy,
            attach_half_x,
            attach_half_y,
            AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return False
        if current_wall in ("north", "south"):
            current_along = float(current.get("cx", 0.0))
        else:
            current_along = float(current.get("cy", 0.0))
        end_targets = [lo, hi]
        end_targets.sort(key=lambda v: abs(float(v) - current_along))

        placed.pop(zone_idx)
        color = current.get("rgba", (0.7, 0.7, 0.7, 0.7))
        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for along in end_targets:
                cx, cy = _wall_attached_center(
                    current_wall,
                    along,
                    sx,
                    sy,
                    attach_half_x,
                    attach_half_y,
                    AREA_LAYOUT_EDGE_MARGIN,
                )
                cand = _make_candidate(str(zone_name), sx, sy, cx, cy, color)
                if not _can_place(cand, gap):
                    continue
                placed.append(cand)
                return True
        placed.insert(zone_idx, current)
        return False

                                                                               
                                        

    layout = {
        area["name"]: {
            "sx": float(area["sx"]),
            "sy": float(area["sy"]),
            "cx": float(area["cx"]),
            "cy": float(area["cy"]),
        }
        for area in placed
    }

    if SHOW_AREA_LAYOUT_MARKERS:
        z = floor_top_z + AREA_LAYOUT_TILE_HALF_Z + 0.005
        for area in placed:
            name = area["name"]
            sx = area["sx"]
            sy = area["sy"]
            cx = area["cx"]
            cy = area["cy"]
            if name in ("LOADING", "STORAGE", "FACTORY"):
                hx = sx * 0.5
                hy = sy * 0.5
                corners = (
                    (cx - hx, cy - hy, z + 0.02),
                    (cx + hx, cy - hy, z + 0.02),
                    (cx + hx, cy + hy, z + 0.02),
                    (cx - hx, cy + hy, z + 0.02),
                )
                outline_colors = {
                    "LOADING": (0.98, 0.82, 0.16),
                    "STORAGE": (0.36, 0.88, 0.62),
                    "FACTORY": (0.42, 0.72, 0.98),
                }
                outline_rgb = outline_colors.get(name, (0.9, 0.9, 0.9))
                for i in range(4):
                    p0 = corners[i]
                    p1 = corners[(i + 1) % 4]
                    p.addUserDebugLine(
                        lineFromXYZ=p0,
                        lineToXYZ=p1,
                        lineColorRGB=outline_rgb,
                        lineWidth=3.0,
                        lifeTime=0.0,
                    )
                p.addUserDebugText(
                    text=f"{name} | {sx:.0f} x {sy:.0f} m",
                    textPosition=[cx, cy, z + 0.05],
                    textColorRGB=[0.06, 0.07, 0.10],
                    textSize=1.2,
                    lifeTime=0.0,
                )
                continue
            if ENABLE_FACTORY_BARRIER_RING and name == "FACTORY":
                continue
            if name == "FORKLIFT_PARK" and ENABLE_FORKLIFT_PARK_SLOT_LINES:
                continue
            if name == "OFFICE" and ENABLE_EMBEDDED_OFFICE_MAP:
                continue
            vid = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[sx * 0.5, sy * 0.5, AREA_LAYOUT_TILE_HALF_Z],
                rgbaColor=list(area["rgba"]),
            )
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vid,
                basePosition=[cx, cy, z],
                useMaximalCoordinates=True,
            )
            p.changeVisualShape(
                body_id,
                -1,
                rgbaColor=list(area["rgba"]),
                textureUniqueId=-1,
                specularColor=list(UNIFORM_SPECULAR_COLOR),
            )
            p.addUserDebugText(
                text=f"{name} | {sx:.0f} x {sy:.0f} m",
                textPosition=[cx, cy, z + 0.05],
                textColorRGB=[0.06, 0.07, 0.10],
                textSize=1.2,
                lifeTime=0.0,
            )

    return layout


def _load_local_office_module():
    global _OFFICE_MODULE_CACHE
    if _OFFICE_MODULE_CACHE is not None:
        return _OFFICE_MODULE_CACHE

    office_path = ""
    for candidate in (
        os.path.join(SCRIPT_DIR, "sections", "office.py"),
        os.path.join(SCRIPT_DIR, "office.py"),
    ):
        if os.path.exists(candidate):
            office_path = candidate
            break
    if not os.path.exists(office_path):
        return None
    spec = importlib.util.spec_from_file_location("embedded_office_map", office_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _OFFICE_MODULE_CACHE = module
    return module


def _embedded_office_role_map(office_mod, seed, office_center_xy, entry_target_xy=None):
    cx, cy = office_center_xy
    blocked_slots = {
        "east" if cx >= 0.0 else "west",
        "north" if cy >= 0.0 else "south",
    }
    slots = list(office_mod.WALL_SLOTS)
    roles = list(office_mod.WALL_ROLES)

    allowed_entry_slots = [s for s in slots if s not in blocked_slots]
    if not allowed_entry_slots:
        return office_mod.wall_role_map(int(seed) + EMBEDDED_OFFICE_SEED_OFFSET)

    if entry_target_xy is not None:
        to_target_x = float(entry_target_xy[0]) - float(cx)
        to_target_y = float(entry_target_xy[1]) - float(cy)
    else:
        to_target_x = -float(cx)
        to_target_y = -float(cy)
    if abs(to_target_x) >= abs(to_target_y):
        preferred_entry_slot = "east" if to_target_x >= 0.0 else "west"
    else:
        preferred_entry_slot = "north" if to_target_y >= 0.0 else "south"
    if preferred_entry_slot in allowed_entry_slots:
        entry_slot = preferred_entry_slot
    else:
        slot_dirs = {
            "east": (1.0, 0.0),
            "west": (-1.0, 0.0),
            "north": (0.0, 1.0),
            "south": (0.0, -1.0),
        }
        mag = max(1e-6, math.hypot(to_target_x, to_target_y))
        tx = to_target_x / mag
        ty = to_target_y / mag
        entry_slot = max(
            allowed_entry_slots,
            key=lambda s: (slot_dirs.get(str(s), (0.0, 0.0))[0] * tx)
            + (slot_dirs.get(str(s), (0.0, 0.0))[1] * ty),
        )

    non_entry_roles = [r for r in roles if r != "entry"]
    rng = random.Random(int(seed) + EMBEDDED_OFFICE_SEED_OFFSET + 701)
    rng.shuffle(non_entry_roles)
    remaining_slots = [s for s in slots if s != entry_slot]
    rng.shuffle(remaining_slots)

    role_by_slot = {entry_slot: "entry"}
    for slot, role in zip(remaining_slots, non_entry_roles):
        role_by_slot[slot] = role
    return role_by_slot


def build_embedded_office_map(floor_top_z, area_layout, seed, wall_info=None):
    if not ENABLE_EMBEDDED_OFFICE_MAP:
        return {"office_map_embedded": False}

    office_area = (area_layout or {}).get("OFFICE")
    if not office_area:
        return {"office_map_embedded": False}

    office_mod = _load_local_office_module()
    if office_mod is None:
        return {"office_map_embedded": False}
    if not os.path.isdir(office_mod.ASSET_PATH):
        return {
            "office_map_embedded": False,
            "office_map_reason": f"Missing furniture assets: {office_mod.ASSET_PATH}",
        }

    old_floor_size = office_mod.FLOOR_SIZE
    old_room_center = office_mod.ROOM_CENTER

    try:
        office_mod.FLOOR_SIZE = min(float(office_area["sx"]), float(office_area["sy"]))
        office_mod.ROOM_CENTER = (float(office_area["cx"]), float(office_area["cy"]))

        loader = office_mod.AssetLoader(
            office_mod.ASSET_PATH,
            office_mod.TEMP_URDF_DIR,
            office_mod.UNIFORM_SCALE,
        )
        entry_target_xy = None
        personnel_side = str((wall_info or {}).get("personnel_side", "")).strip().lower()
        if personnel_side in WALL_SLOTS:
            personnel_along = float((wall_info or {}).get("personnel_along", 0.0))
            wall_thickness = float((wall_info or {}).get("wall_thickness", 0.0))
            px, py = slot_point(
                personnel_side,
                personnel_along,
                inward=(wall_thickness * 0.5) + DOCK_INWARD_NUDGE,
            )
            if personnel_side == "east":
                dir_x, dir_y = -1.0, 0.0
            elif personnel_side == "west":
                dir_x, dir_y = 1.0, 0.0
            elif personnel_side == "north":
                dir_x, dir_y = 0.0, -1.0
            else:
                dir_x, dir_y = 0.0, 1.0
            corridor_nudge_m = 1.2
            entry_target_xy = (float(px) + (dir_x * corridor_nudge_m), float(py) + (dir_y * corridor_nudge_m))

        role_by_slot = _embedded_office_role_map(
            office_mod,
            seed=int(seed),
            office_center_xy=office_mod.ROOM_CENTER,
            entry_target_xy=entry_target_xy,
        )
        entry_slot = next(
            (s for s in office_mod.WALL_SLOTS if role_by_slot.get(s) == "entry"),
            office_mod.WALL_SLOTS[0],
        )
        office_walls_enabled = bool(getattr(office_mod, "ENABLE_PERIMETER_WALL_MESHES", False))
        if office_walls_enabled and hasattr(office_mod, "spawn_walls_with_entry"):
            entry_door_along = float(getattr(office_mod, "ENTRY_WALL_OPENING_ALONG", 0.0))
            office_mod.spawn_walls_with_entry(
                loader,
                floor_top_z,
                entry_slot=entry_slot,
                door_along=entry_door_along,
                open_mode=str(getattr(office_mod, "ENTRY_WALL_OPENING_MODE", "door_segment")),
            )
        spawn_entry_doorway = (not office_walls_enabled) or (
            str(getattr(office_mod, "ENTRY_WALL_OPENING_MODE", "door_segment")).lower() == "gap"
        )
        corners = office_mod.corner_points()
        forbidden_styles_by_corner = {}
        blocked_corner_indices = set()

        for slot in office_mod.WALL_SLOTS:
            role = role_by_slot[slot]
            if role == "entry":
                blocked = office_mod.place_entry_wall(
                    loader,
                    floor_top_z,
                    slot,
                    int(seed),
                    corners=corners,
                    spawn_doorway=spawn_entry_doorway,
                )
                for k, vals in blocked.items():
                    forbidden_styles_by_corner.setdefault(k, set()).update(vals)
            elif role == "workstations":
                workstation_corner = office_mod.place_workstations_wall(loader, floor_top_z, slot, int(seed))
                if workstation_corner is not None:
                    blocked_corner_indices.add(workstation_corner)
            elif role == "files":
                office_mod.place_files_wall(loader, floor_top_z, slot, int(seed))
            elif role == "services":
                blocked = office_mod.place_services_wall(loader, floor_top_z, slot, int(seed), corners=corners)
                for k, vals in blocked.items():
                    forbidden_styles_by_corner.setdefault(k, set()).update(vals)

        office_mod.place_corner_decor(
            loader,
            floor_top_z,
            int(seed),
            forbidden_styles_by_corner=forbidden_styles_by_corner,
            blocked_corner_indices=blocked_corner_indices,
        )
        office_mod.build_center_meeting(loader, floor_top_z, int(seed))

        return {
            "office_map_embedded": True,
            "office_map_center_xy": office_mod.ROOM_CENTER,
            "office_map_size_m": office_mod.FLOOR_SIZE,
            "office_map_roles": role_by_slot,
            "office_walls_enabled": office_walls_enabled,
            "office_entry_slot": entry_slot,
            "office_entry_door_along": float(entry_door_along if office_walls_enabled else 0.0),
        }
    except Exception as exc:
        return {"office_map_embedded": False, "office_map_reason": str(exc)}
    finally:
        office_mod.FLOOR_SIZE = old_floor_size
        office_mod.ROOM_CENTER = old_room_center


def _load_local_factory_module():
    global _FACTORY_MODULE_CACHE
    if _FACTORY_MODULE_CACHE is not None:
        return _FACTORY_MODULE_CACHE

    factory_path = ""
    for candidate in (
        os.path.join(SCRIPT_DIR, "sections", "factory_map.py"),
        os.path.join(SCRIPT_DIR, "factory_map.py"),
    ):
        if os.path.exists(candidate):
            factory_path = candidate
            break
    if not os.path.exists(factory_path):
        return None
    spec = importlib.util.spec_from_file_location("embedded_factory_map", factory_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _FACTORY_MODULE_CACHE = module
    return module


def _resolve_factory_barrier_model():
    for path in FACTORY_BARRIER_MODEL_CANDIDATES:
        if path and os.path.exists(path):
            return os.path.abspath(path)
    return ""


def _get_factory_barrier_loader(model_dir, texture_path):
    key = (os.path.abspath(str(model_dir)), str(texture_path or ""))
    cached = _FACTORY_BARRIER_LOADER_CACHE.get(key)
    if cached is not None:
        return cached
    loader = MeshKitLoader(obj_dir=key[0], texture_path=key[1])
    _FACTORY_BARRIER_LOADER_CACHE[key] = loader
    return loader


def _segment_centers_1d(min_v, max_v, segment_len, gap):
    span = max_v - min_v
    if segment_len <= 1e-6 or span <= 1e-6:
        return []
    if span <= segment_len:
        return [0.5 * (min_v + max_v)]
    step = segment_len + max(0.0, gap)
    count = max(1, int((span + gap) // step))
    used = (count * segment_len) + ((count - 1) * max(0.0, gap))
    if used > span and count > 1:
        count -= 1
        used = (count * segment_len) + ((count - 1) * max(0.0, gap))
    if count <= 0:
        return [0.5 * (min_v + max_v)]
    slack = max(0.0, span - used)
    start = min_v + (slack * 0.5) + (segment_len * 0.5)
    return [start + (i * step) for i in range(count)]


def build_factory_barrier_ring(conveyor_loader, floor_top_z, factory_area, network, factory_mod):
    if not ENABLE_FACTORY_BARRIER_RING:
        return {"factory_barrier_enabled": False}
    if not factory_area:
        return {"factory_barrier_enabled": False, "factory_barrier_reason": "FACTORY area missing."}

    barrier_model_path = _resolve_factory_barrier_model()
    if not barrier_model_path:
        return {
            "factory_barrier_enabled": False,
            "factory_barrier_reason": "Barrier model not found.",
        }

    model_dir = os.path.dirname(barrier_model_path)
    model_name = os.path.basename(barrier_model_path)
    barrier_loader = _get_factory_barrier_loader(
        model_dir=model_dir,
        texture_path=conveyor_loader.texture_path,
    )

    min_v, max_v = model_bounds_xyz(barrier_loader, model_name, FACTORY_BARRIER_SCALE_XYZ)
    seg_len = max(0.0, max_v[0] - min_v[0])
    seg_dep = max(0.0, max_v[1] - min_v[1])
    if seg_len <= 1e-6 or seg_dep <= 1e-6:
        return {
            "factory_barrier_enabled": False,
            "factory_barrier_reason": "Barrier model has invalid bounds.",
        }

    anchor_x = (min_v[0] + max_v[0]) * 0.5
    anchor_y = (min_v[1] + max_v[1]) * 0.5
    anchor_z = min_v[2]

    cx = float(factory_area["cx"])
    cy = float(factory_area["cy"])
    sx = float(factory_area["sx"])
    sy = float(factory_area["sy"])
    half_x = sx * 0.5
    half_y = sy * 0.5
    inset = max(0.0, float(FACTORY_BARRIER_INSET_M))

    x_min = cx - half_x + inset
    x_max = cx + half_x - inset
    y_min = cy - half_y + inset
    y_max = cy + half_y - inset
    if x_max <= x_min or y_max <= y_min:
        return {
            "factory_barrier_enabled": False,
            "factory_barrier_reason": "FACTORY area too small for barrier inset.",
        }

    north_y = y_max - (seg_dep * 0.5)
    south_y = y_min + (seg_dep * 0.5)
    east_x = x_max - (seg_dep * 0.5)
    west_x = x_min + (seg_dep * 0.5)

    segments = []
    for x in _segment_centers_1d(x_min, x_max, seg_len, FACTORY_BARRIER_SEGMENT_GAP_M):
        segments.append({"side": "north", "x": x, "y": north_y, "yaw": 0.0})
        segments.append({"side": "south", "x": x, "y": south_y, "yaw": 0.0})
    for y in _segment_centers_1d(y_min, y_max, seg_len, FACTORY_BARRIER_SEGMENT_GAP_M):
        segments.append({"side": "east", "x": east_x, "y": y, "yaw": 90.0})
        segments.append({"side": "west", "x": west_x, "y": y, "yaw": 90.0})
    if not segments:
        return {
            "factory_barrier_enabled": False,
            "factory_barrier_reason": "No barrier segments could be generated.",
        }

    start_pos = (cx, cy)
    start_cell = network.get("start_cell")
    module_size = network.get("module_size_xyz_m", (0.0, 0.0, 0.0))
    cell_size = float(module_size[0]) if module_size else 0.0
    if (
        isinstance(start_cell, (tuple, list))
        and len(start_cell) >= 2
        and cell_size > 1e-6
    ):
        edge_margin = float(getattr(factory_mod, "EDGE_MARGIN_M", 3.0))
        row_margin = float(getattr(factory_mod, "ROW_MARGIN_M", 1.4))
        local_x_min = -half_x + edge_margin
        local_y_min = -half_y + row_margin
        start_pos = (
            cx + local_x_min + (float(start_cell[0]) + 0.5) * cell_size,
            cy + local_y_min + (float(start_cell[1]) + 0.5) * cell_size,
        )

    gap_idx = min(
        range(len(segments)),
        key=lambda i: (segments[i]["x"] - start_pos[0]) ** 2 + (segments[i]["y"] - start_pos[1]) ** 2,
    )
    entry_side = segments[gap_idx]["side"]
    segments_to_spawn = [s for i, s in enumerate(segments) if i != gap_idx]

    collision_proxy = _obj_collision_proxy_path(barrier_model_path)
    spawned = 0
    for seg in segments_to_spawn:
        world_anchor = (seg["x"], seg["y"], float(floor_top_z))
        yaw_deg = float(seg["yaw"])
        if FACTORY_BARRIER_WITH_COLLISION:
            _spawn_collision_only_with_anchor(
                loader=barrier_loader,
                model_name=model_name,
                world_anchor_xyz=world_anchor,
                yaw_deg=yaw_deg,
                mesh_scale_xyz=FACTORY_BARRIER_SCALE_XYZ,
                local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                model_path_override=collision_proxy,
            )

        _spawn_mesh_with_anchor(
            loader=barrier_loader,
            model_name=model_name,
            world_anchor_xyz=world_anchor,
            yaw_deg=yaw_deg,
            mesh_scale_xyz=FACTORY_BARRIER_SCALE_XYZ,
            local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
            with_collision=False,
            use_texture=False,
            rgba=FACTORY_BARRIER_FLAT_RGBA,
            double_sided=FACTORY_BARRIER_DOUBLE_SIDED,
        )
        spawned += 1

    return {
        "factory_barrier_enabled": spawned > 0,
        "factory_barrier_model": model_name,
        "factory_barrier_scale_xyz": FACTORY_BARRIER_SCALE_XYZ,
        "factory_barrier_count": int(spawned),
        "factory_barrier_entry_side": entry_side,
        "factory_barrier_entry_xy": (float(start_pos[0]), float(start_pos[1])),
    }


def build_embedded_factory_map(conveyor_loader, floor_top_z, area_layout, seed):
    if not ENABLE_EMBEDDED_FACTORY_MAP:
        return {"factory_map_embedded": False}

    factory_area = (area_layout or {}).get("FACTORY")
    if not factory_area:
        return {"factory_map_embedded": False, "factory_map_reason": "FACTORY area not present in layout."}

    factory_mod = _load_local_factory_module()
    if factory_mod is None:
        return {"factory_map_embedded": False, "factory_map_reason": "factory_map.py not found."}

    size_xy = (float(factory_area["sx"]), float(factory_area["sy"]))
    center_xy = (float(factory_area["cx"]), float(factory_area["cy"]))
    try:
        network = factory_mod.build_single_belt_network(
            conveyor_loader,
            seed=int(seed) + EMBEDDED_FACTORY_SEED_OFFSET,
            center_xy=center_xy,
            size_xy=size_xy,
            floor_z=float(floor_top_z),
        )
        barrier_info = build_factory_barrier_ring(
            conveyor_loader=conveyor_loader,
            floor_top_z=float(floor_top_z),
            factory_area=factory_area,
            network=network,
            factory_mod=factory_mod,
        )
        if SHOW_AREA_LAYOUT_MARKERS:
            p.addUserDebugText(
                text=f"EMBEDDED FACTORY | {size_xy[0]:.0f} x {size_xy[1]:.0f} m",
                textPosition=[center_xy[0], center_xy[1], float(floor_top_z) + 0.03],
                textColorRGB=[0.08, 0.10, 0.12],
                textSize=1.2,
                lifeTime=0.0,
            )
        return {
            "factory_map_embedded": True,
            "factory_area_center_xy": center_xy,
            "factory_area_size_m": size_xy,
            "factory_network": network,
            **barrier_info,
        }
    except Exception as exc:
        return {"factory_map_embedded": False, "factory_map_reason": str(exc)}


def _shell_mesh_scale_xy(shell_meshes):
    cfg = shell_meshes.get("config", {}) or {}
    base_x = float(cfg.get("warehouse_size_x", WAREHOUSE_BASE_SIZE_X))
    base_y = float(cfg.get("warehouse_size_y", WAREHOUSE_BASE_SIZE_Y))
    sx = (float(WAREHOUSE_SIZE_X) / base_x) if abs(base_x) > 1e-9 else 1.0
    sy = (float(WAREHOUSE_SIZE_Y) / base_y) if abs(base_y) > 1e-9 else 1.0
    return sx, sy


def build_curved_roof(loader, roof_base_z, shell_meshes):
    shell_sx, shell_sy = _shell_mesh_scale_xy(shell_meshes)
    _spawn_generated_mesh(
        shell_meshes["roof"],
        loader.texture_path,
        with_collision=True,
        use_texture=False,
        rgba=ROOF_UNIFORM_COLOR,
        double_sided=True,
        base_position=(0.0, 0.0, roof_base_z),
        mesh_scale_xyz=(shell_sx, shell_sy, 1.0),
    )
    _spawn_generated_mesh(
        shell_meshes["fillers"],
        loader.texture_path,
        with_collision=True,
        use_texture=False,
        rgba=WALL_UNIFORM_COLOR,
        double_sided=True,
        base_position=(0.0, 0.0, roof_base_z),
        mesh_scale_xyz=(shell_sx, shell_sy, 1.0),
    )


def build_roof_truss_system(floor_top_z, roof_base_z, shell_meshes):
    _ = floor_top_z
    if not ENABLE_ROOF_TRUSS_SYSTEM:
        return {
            "roof_ribs": 0,
            "roof_rib_segments": 0,
            "truss_frames": 0,
            "truss_members": 0,
            "interior_columns": 0,
        }

    shell_sx, shell_sy = _shell_mesh_scale_xy(shell_meshes)
    _spawn_generated_mesh(
        shell_meshes["truss"],
        texture_path="",
        with_collision=TRUSS_WITH_COLLISION,
        use_texture=False,
        rgba=TRUSS_UNIFORM_COLOR,
        double_sided=True,
        base_position=(0.0, 0.0, roof_base_z),
        mesh_scale_xyz=(shell_sx, shell_sy, 1.0),
    )

    cfg = shell_meshes.get("config", {})
    rib_count = max(1, int(cfg.get("truss_rib_count", 5)))
    node_count = max(3, int(cfg.get("truss_node_count", 9)))
    panel_count = node_count - 1
    top_profile_segments = max(int(cfg.get("truss_top_profile_segments", 96)), node_count * 8)
    top_segments = rib_count * top_profile_segments
    lower_segments = rib_count * panel_count
    web_segments = rib_count * max(0, panel_count - 2)
    total_segments = top_segments + lower_segments + web_segments
    return {
        "roof_ribs": rib_count,
        "roof_rib_segments": total_segments,
        "truss_frames": rib_count,
        "truss_members": total_segments,
        "interior_columns": 0,
    }


def build_columns(loader, floor_top_z):
    hx = WAREHOUSE_SIZE_X * 0.5
    hy = WAREHOUSE_SIZE_Y * 0.5
    column_size = loader.model_size(CONVEYOR_ASSETS["column"], UNIFORM_SCALE)
    inset = max(column_size[0], column_size[1]) * 0.55
    corners = (
        (-hx + inset, -hy + inset),
        (hx - inset, -hy + inset),
        (-hx + inset, hy - inset),
        (hx - inset, hy - inset),
    )
    for x, y in corners:
        loader.spawn(
            CONVEYOR_ASSETS["column"],
            x=x,
            y=y,
            yaw_deg=0.0,
            floor_z=floor_top_z,
            scale=UNIFORM_SCALE,
            with_collision=True,
        )


def build_walls(conveyor_loader, floor_top_z, seed):
    rng = random.Random(seed)

    wall_model = CONVEYOR_ASSETS["wall"]
    window_model = CONVEYOR_ASSETS["wall_window"]
    window_wide_model = CONVEYOR_ASSETS["wall_window_wide"]
    corner_model = CONVEYOR_ASSETS["wall_corner"]
    dock_frame_model = CONVEYOR_ASSETS["dock_frame"]
    dock_door_models = (
        CONVEYOR_ASSETS["dock_door_closed"],
        CONVEYOR_ASSETS["dock_door_half"],
        CONVEYOR_ASSETS["dock_door_open"],
    )
    personnel_frame_model = CONVEYOR_ASSETS["personnel_frame"]
    personnel_door_model = CONVEYOR_ASSETS["personnel_door"]

    wall_size = conveyor_loader.model_size(wall_model, UNIFORM_SCALE)
    wall_h = wall_size[2]
    corner_size = conveyor_loader.model_size(corner_model, UNIFORM_SCALE)
    floor_spawn_half_x, floor_spawn_half_y = _floor_spawn_half_extents(conveyor_loader)

    personnel_slot = rng.choice(("east", "west"))
    personnel_wall_yaw = wall_yaw_for_slot(personnel_slot)
    personnel_slot_extent = WAREHOUSE_SIZE_X if personnel_slot in ("north", "south") else WAREHOUSE_SIZE_Y
    pers_ex, pers_ey = oriented_xy_size(conveyor_loader, wall_model, UNIFORM_SCALE, personnel_wall_yaw)
    personnel_wall_step = round(pers_ex if personnel_slot in ("north", "south") else pers_ey, 6)
    personnel_wall_thickness = round(pers_ey if personnel_slot in ("north", "south") else pers_ex, 6)
    pers_frame_ex, pers_frame_ey = oriented_xy_size(
        conveyor_loader, personnel_frame_model, UNIFORM_SCALE, personnel_wall_yaw
    )
    personnel_door_span = round(
        pers_frame_ex if personnel_slot in ("north", "south") else pers_frame_ey,
        6,
    )
    personnel_along_values = tiled_centers(personnel_slot_extent, personnel_wall_step)
    personnel_along = 0.0
    personnel_half_span = max(0.0, personnel_door_span * 0.5)
    personnel_segment_half = max(0.0, personnel_wall_step * 0.5)
    personnel_frame_lo = personnel_along - personnel_half_span
    personnel_frame_hi = personnel_along + personnel_half_span
    personnel_block_indices = [
        i
        for i, c in enumerate(personnel_along_values)
        if ((c + personnel_segment_half) > (personnel_frame_lo + 1e-6))
        and ((c - personnel_segment_half) < (personnel_frame_hi - 1e-6))
    ]
    if not personnel_block_indices:
        personnel_block_indices = [
            min(
                range(len(personnel_along_values)),
                key=lambda i: abs(personnel_along_values[i] - personnel_along),
            )
        ]
    personnel_segment_idx = personnel_block_indices[0]

    loading_candidates = [slot for slot in LOADING_SLOTS if slot != personnel_slot]
    if not loading_candidates:
        loading_candidates = list(WALL_SLOTS)
    if not loading_candidates:
        raise ValueError("No available wall slot for loading docks.")
    opposite_personnel_slot = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }.get(personnel_slot)
    transverse_major_zone = str(rng.choice(("LOADING", "STORAGE", "FACTORY")))
    if personnel_slot in ("east", "west"):
        longitudinal_side_walls = ("north", "south")
    elif personnel_slot in ("north", "south"):
        longitudinal_side_walls = ("east", "west")
    else:
        longitudinal_side_walls = tuple(WALL_SLOTS)
    longitudinal_loading_candidates = [s for s in loading_candidates if s in longitudinal_side_walls]
    if transverse_major_zone == "LOADING":
        preferred_loading_slots = (
            [opposite_personnel_slot] if opposite_personnel_slot in loading_candidates else []
        )
    else:
        preferred_loading_slots = list(longitudinal_loading_candidates)
    loading_def = next((a for a in AREA_LAYOUT_BLOCKS if a.get("name") == "LOADING"), None)
    if loading_def is not None:
        corner_capable_slots = []
        for cand_slot in loading_candidates:
            cand_yaw = wall_yaw_for_slot(cand_slot)
            c_ex, c_ey = oriented_xy_size(conveyor_loader, wall_model, UNIFORM_SCALE, cand_yaw)
            cand_wall_thickness = round(c_ey if cand_slot in ("north", "south") else c_ex, 6)
            cand_attach_inset = cand_wall_thickness * float(AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR)
            cand_attach_half_x = min(
                floor_spawn_half_x,
                max(1.0, (WAREHOUSE_SIZE_X * 0.5) - cand_attach_inset),
            )
            cand_attach_half_y = min(
                floor_spawn_half_y,
                max(1.0, (WAREHOUSE_SIZE_Y * 0.5) - cand_attach_inset),
            )
            cand_sx, cand_sy = _loading_marker_xy_size(loading_def["size_m"], cand_slot)
            cand_sx, cand_sy = _orient_dims_long_side_on_wall(cand_slot, float(cand_sx), float(cand_sy))
            cand_lo, cand_hi = _wall_along_limits(
                cand_slot,
                cand_sx,
                cand_sy,
                cand_attach_half_x,
                cand_attach_half_y,
                AREA_LAYOUT_EDGE_MARGIN,
            )
            if cand_hi > cand_lo + 1e-6:
                corner_capable_slots.append(cand_slot)
        if corner_capable_slots:
            for pref_slot in preferred_loading_slots:
                if pref_slot in loading_candidates and pref_slot not in corner_capable_slots:
                    corner_capable_slots.append(pref_slot)
            loading_candidates = list(dict.fromkeys(corner_capable_slots))

    preferred_pool = [s for s in preferred_loading_slots if s in loading_candidates]
    if preferred_pool:
        loading_slot = str(rng.choice(preferred_pool))
    else:
        loading_slot = str(rng.choice(loading_candidates))
    opposite_slot = {
        "north": "south",
        "south": "north",
        "east": "west",
        "west": "east",
    }[loading_slot]

    loading_wall_yaw = wall_yaw_for_slot(loading_slot)
    loading_slot_extent = WAREHOUSE_SIZE_X if loading_slot in ("north", "south") else WAREHOUSE_SIZE_Y
    load_ex, load_ey = oriented_xy_size(conveyor_loader, wall_model, UNIFORM_SCALE, loading_wall_yaw)
    loading_wall_step = round(load_ex if loading_slot in ("north", "south") else load_ey, 6)
    loading_wall_thickness = round(load_ey if loading_slot in ("north", "south") else load_ex, 6)
    loading_wall_centers = tiled_centers(loading_slot_extent, loading_wall_step)

    frame_yaw_for_span = dock_inward_yaw_for_slot(loading_slot)
    frame_ex, frame_ey = oriented_xy_size(conveyor_loader, dock_frame_model, UNIFORM_SCALE, frame_yaw_for_span)
    door_span = round(frame_ex if loading_slot in ("north", "south") else frame_ey, 6)

    if loading_wall_step <= 0.0:
        raise ValueError("Invalid wall step size for loading side.")
    if not loading_wall_centers:
        raise ValueError("Loading wall has no usable wall segments.")
    loading_covered_span = float(len(loading_wall_centers)) * float(loading_wall_step)

    door_center_step = loading_wall_step * float(LOADING_DOOR_CENTER_STEP_FRACTION)
    if door_center_step <= 0.0:
        raise ValueError("Invalid loading dock center step.")
    n_loading = int(round(loading_covered_span / door_center_step))
    along_start = -loading_covered_span * 0.5
    boundary_centers = [along_start + i * door_center_step for i in range(n_loading + 1)]
    max_center = (loading_covered_span * 0.5) - (door_span * 0.5)
    if loading_slot in ("north", "south"):
        zone_attach_half_along = min(
            floor_spawn_half_x,
            (WAREHOUSE_SIZE_X * 0.5) - (
                loading_wall_thickness * float(AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR)
            ),
        )
    else:
        zone_attach_half_along = min(
            floor_spawn_half_y,
            (WAREHOUSE_SIZE_Y * 0.5) - (
                loading_wall_thickness * float(AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR)
            ),
        )
    zone_cover_limit = max(0.0, zone_attach_half_along - (door_span * 0.5))
    max_center = min(max_center, zone_cover_limit)
    truck_along_extent = _estimate_loading_truck_along_extent_m(loading_slot)
    if truck_along_extent > 0.0:
        interior_half_span = (loading_covered_span * 0.5) - (loading_wall_thickness * 0.5)
        truck_half_along = truck_along_extent * 0.5
        truck_side_clearance = max(0.05, LOADING_TRUCK_WALL_GAP)
        truck_center_limit = interior_half_span - truck_half_along - truck_side_clearance
        if truck_center_limit > 0.0:
            max_center = min(max_center, truck_center_limit)
    safe_centers = [c for c in boundary_centers if abs(c) <= max_center + 1e-6]
    if len(safe_centers) < 3:
        raise ValueError("Loading wall does not have enough span for 3 dock gates.")

    door_center_spacing = door_span + (LOADING_INTER_GATE_WALL_STEPS * loading_wall_step)
    min_idx_gap = max(1, int(math.ceil(door_center_spacing / door_center_step)))
    n_safe = len(safe_centers)
    max_supported_gap = max(1, (n_safe - 1) // 2)
    min_idx_gap = min(min_idx_gap, max_supported_gap)
    if min_idx_gap <= 0:
        raise ValueError("Unable to place 3 dock gates with current wall span.")
    loading_def = next((a for a in AREA_LAYOUT_BLOCKS if a.get("name") == "LOADING"), None)
    loading_span_along = float(loading_def["size_m"][1]) if loading_def is not None else float(loading_slot_extent)
    loading_span_along = max(0.0, min(float(loading_slot_extent), loading_span_along))
    max_door_spread_for_loading = max(0.0, loading_span_along - float(door_span))
    if loading_def is not None:
        loading_zone_sx, loading_zone_sy = _loading_marker_xy_size(loading_def["size_m"], loading_slot)
    else:
        loading_zone_sx, loading_zone_sy = (
            loading_span_along if loading_slot in ("north", "south") else float(door_span) * 3.0,
            float(door_span) * 3.0 if loading_slot in ("north", "south") else loading_span_along,
        )
    loading_zone_sx, loading_zone_sy = _orient_dims_long_side_on_wall(
        loading_slot,
        float(loading_zone_sx),
        float(loading_zone_sy),
    )
    zone_attach_half_x = min(
        floor_spawn_half_x,
        (WAREHOUSE_SIZE_X * 0.5) - (
            loading_wall_thickness * float(AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR)
        ),
    )
    zone_attach_half_y = min(
        floor_spawn_half_y,
        (WAREHOUSE_SIZE_Y * 0.5) - (
            loading_wall_thickness * float(AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR)
        ),
    )
    zone_along_lo, zone_along_hi = _wall_along_limits(
        loading_slot,
        loading_zone_sx,
        loading_zone_sy,
        zone_attach_half_x,
        zone_attach_half_y,
        AREA_LAYOUT_EDGE_MARGIN,
    )
    loading_along_half = (
        (loading_zone_sx * 0.5) if loading_slot in ("north", "south") else (loading_zone_sy * 0.5)
    )
    center_pad = max(0.0, loading_along_half - (float(door_span) * 0.5))

    left_corner_center = zone_along_lo
    right_corner_center = zone_along_hi

    triplets = []
    for i in range(0, n_safe - (2 * min_idx_gap)):
        j = i + min_idx_gap
        k = j + min_idx_gap
        spread = safe_centers[k] - safe_centers[i]
        if spread > max_door_spread_for_loading + 1e-6:
            continue
        door_min = safe_centers[i]
        door_max = safe_centers[k]
        fits_left_corner = (
            door_min >= (left_corner_center - center_pad - 1e-6)
            and door_max <= (left_corner_center + center_pad + 1e-6)
        )
        fits_right_corner = (
            door_min >= (right_corner_center - center_pad - 1e-6)
            and door_max <= (right_corner_center + center_pad + 1e-6)
        )
        if fits_left_corner or fits_right_corner:
            triplets.append((i, j, k, fits_left_corner, fits_right_corner))
    if not triplets:
        raise ValueError(
            "Unable to place 3 dock gates that fit inside current LOADING span "
            f"({loading_span_along:.1f}m) while keeping LOADING at a corner."
        )

    preferred_loading_corner = None
    if loading_slot in ("north", "south"):
        if personnel_slot == "west":
            preferred_loading_corner = "right" if transverse_major_zone == "LOADING" else "left"
        elif personnel_slot == "east":
            preferred_loading_corner = "left" if transverse_major_zone == "LOADING" else "right"
    elif loading_slot in ("east", "west"):
        if personnel_slot == "south":
            preferred_loading_corner = "right" if transverse_major_zone == "LOADING" else "left"
        elif personnel_slot == "north":
            preferred_loading_corner = "left" if transverse_major_zone == "LOADING" else "right"
    if preferred_loading_corner not in ("left", "right"):
        preferred_loading_corner = "left"

    loading_corner_side = preferred_loading_corner
    corner_triplets = [t for t in triplets if (t[3] if loading_corner_side == "left" else t[4])]
    if not corner_triplets:
        loading_corner_side = "right" if loading_corner_side == "left" else "left"
        corner_triplets = [t for t in triplets if (t[3] if loading_corner_side == "left" else t[4])]

    def _triplet_key(tri):
        i, j, k = tri[0], tri[1], tri[2]
        mean_idx = (i + j + k) / 3.0
        if loading_corner_side == "left":
            return (mean_idx,)
        return (-mean_idx,)

    ranked = sorted(corner_triplets, key=_triplet_key)
    top_n = min(6, len(ranked))
    picked = ranked[rng.randrange(top_n)]
    door_centers = sorted([safe_centers[picked[0]], safe_centers[picked[1]], safe_centers[picked[2]]])
    gate_states = list(dock_door_models)
    rng.shuffle(gate_states)

    axis_layout_cache = {}

    for slot in WALL_SLOTS:
        wall_yaw = wall_yaw_for_slot(slot)
        slot_extent = WAREHOUSE_SIZE_X if slot in ("north", "south") else WAREHOUSE_SIZE_Y
        ex, ey = oriented_xy_size(conveyor_loader, wall_model, UNIFORM_SCALE, wall_yaw)
        wall_step = round(ex if slot in ("north", "south") else ey, 6)
        wall_thickness = round(ey if slot in ("north", "south") else ex, 6)
        along_values = tiled_centers(slot_extent, wall_step)
        straight_along = along_values
        segment_count = len(straight_along)
        axis_key = "x" if slot in ("north", "south") else "y"

        blocked_indices = set()
        loading_door_spans = []
        if slot == loading_slot:
            blocked_indices = _indices_blocked_by_doors(straight_along, door_centers, door_span)
            door_half = float(door_span) * 0.5
            loading_door_spans = _merge_spans_1d(
                [(float(c) - door_half, float(c) + door_half) for c in door_centers]
            )
        if slot == personnel_slot:
            blocked_indices.update(personnel_block_indices)

        wide_ex, wide_ey = oriented_xy_size(conveyor_loader, window_wide_model, UNIFORM_SCALE, wall_yaw)
        wide_step = round(wide_ex if slot in ("north", "south") else wide_ey, 6)
        wide_span_steps = max(1, int(round(wide_step / wall_step)))

        pair_seed_key = int(seed) * 17 + (0 if axis_key == "x" else 1) * 191
        if axis_key not in axis_layout_cache:
            base_wide_starts = mirrored_wide_window_starts(
                segment_count,
                wide_span_steps,
                pair_seed_key,
            )
            base_wide_covered = set()
            for s in base_wide_starts:
                base_wide_covered.update(range(s, s + wide_span_steps))
            base_window_indices = mirrored_window_indices(segment_count) - base_wide_covered
            axis_layout_cache[axis_key] = {
                "wide_starts": list(base_wide_starts),
                "window_indices": set(base_window_indices),
            }

        wide_starts = list(axis_layout_cache[axis_key]["wide_starts"])
        window_indices = set(axis_layout_cache[axis_key]["window_indices"])

        if slot in (loading_slot, opposite_slot):
            wide_starts = _filter_mirrored_wide_windows(
                candidate_starts=wide_starts,
                span_steps=wide_span_steps,
                blocked_indices=blocked_indices,
                segment_count=segment_count,
            )

        wide_start_set = set(wide_starts)
        wide_covered = set()
        for s in wide_starts:
            wide_covered.update(range(s, s + wide_span_steps))

        window_indices = window_indices - wide_covered
        window_indices = _filter_mirrored_single_windows(
            candidate_indices=window_indices,
            blocked_indices=blocked_indices,
            segment_count=segment_count,
        )

        corner_along_span = round(corner_size[0] if slot in ("north", "south") else corner_size[1], 6)
        slot_half = float(slot_extent) * 0.5
        corner_inner_lo = -slot_half + float(corner_along_span)
        corner_inner_hi = slot_half - float(corner_along_span)
        half_step = wall_step * 0.5

        reserved_left = 0
        while reserved_left < segment_count:
            seg_hi = float(straight_along[reserved_left]) + half_step
            if seg_hi > (corner_inner_lo + 1e-6):
                break
            reserved_left += 1

        reserved_right = segment_count
        while reserved_right > 0:
            seg_lo = float(straight_along[reserved_right - 1]) - half_step
            if seg_lo < (corner_inner_hi - 1e-6):
                break
            reserved_right -= 1

        wide_start_set = {
            s
            for s in wide_start_set
            if (s >= reserved_left and (s + wide_span_steps) <= reserved_right)
        }
        window_indices = {i for i in window_indices if reserved_left <= i < reserved_right}
        if slot == personnel_slot:
            wide_start_set = {
                s
                for s in wide_start_set
                if not (s <= personnel_segment_idx < (s + wide_span_steps))
            }
            for p_idx in personnel_block_indices:
                window_indices.discard(p_idx)
        if slot == loading_slot:
            wide_start_set = set()
            window_indices = set()

        if slot == opposite_slot and not wide_start_set and not window_indices:
            fallback = {
                i
                for i in mirrored_window_indices(segment_count)
                if reserved_left <= i < reserved_right and i not in blocked_indices
            }
            if not fallback:
                fallback = {i for i in range(reserved_left, reserved_right) if i not in blocked_indices}
            window_indices = _filter_mirrored_single_windows(
                candidate_indices=fallback,
                blocked_indices=blocked_indices,
                segment_count=segment_count,
            )
            if not window_indices and fallback:
                center_i = min(sorted(fallback), key=lambda i: abs(i - (segment_count // 2)))
                window_indices = {center_i}

        for tier in range(WALL_TIERS):
            tier_base_z = floor_top_z + tier * wall_h
            idx = 0
            while idx < segment_count:
                if idx < reserved_left or idx >= reserved_right:
                    idx += 1
                    continue

                along = straight_along[idx]

                if slot == personnel_slot and tier == 0:
                    if idx == personnel_segment_idx:
                        x, y = slot_point(slot, personnel_along, inward=wall_thickness * 0.5)
                        conveyor_loader.spawn(
                            personnel_frame_model,
                            x=x,
                            y=y,
                            yaw_deg=wall_yaw,
                            floor_z=tier_base_z,
                            scale=UNIFORM_SCALE,
                            with_collision=True,
                        )
                    if idx in personnel_block_indices:
                        idx += 1
                        continue

                if slot == loading_slot and tier == 0:
                    if idx in blocked_indices:
                        idx += 1
                        continue

                if (
                    tier == WALL_TIERS - 1
                    and idx in wide_start_set
                    and idx + wide_span_steps <= segment_count
                ):
                    span_along = straight_along[idx : idx + wide_span_steps]
                    along_center = sum(span_along) / float(len(span_along))
                    x, y = slot_point(slot, along_center, inward=wall_thickness * 0.5)
                    conveyor_loader.spawn(
                        window_wide_model,
                        x=x,
                        y=y,
                        yaw_deg=wall_yaw,
                        floor_z=tier_base_z,
                        scale=UNIFORM_SCALE,
                        with_collision=True,
                    )
                    idx += wide_span_steps
                    continue

                use_small_window = (tier == WALL_TIERS - 1) and (idx in window_indices)
                model = window_model if use_small_window else wall_model
                x, y = slot_point(slot, along, inward=wall_thickness * 0.5)
                conveyor_loader.spawn(
                    model,
                    x=x,
                    y=y,
                    yaw_deg=wall_yaw,
                    floor_z=tier_base_z,
                    scale=UNIFORM_SCALE,
                    with_collision=True,
                )
                idx += 1

            if slot == personnel_slot and tier == 0 and personnel_block_indices:
                blocked_spans = []
                for p_idx in sorted(set(personnel_block_indices)):
                    seg_center = straight_along[p_idx]
                    blocked_spans.append((seg_center - (wall_step * 0.5), seg_center + (wall_step * 0.5)))
                blocked_spans.sort(key=lambda s: s[0])
                merged = []
                for lo, hi in blocked_spans:
                    if not merged or lo > merged[-1][1] + 1e-6:
                        merged.append([lo, hi])
                    else:
                        merged[-1][1] = max(merged[-1][1], hi)

                for lo, hi in merged:
                    left_lo = lo
                    left_hi = min(hi, personnel_frame_lo)
                    right_lo = max(lo, personnel_frame_hi)
                    right_hi = hi
                    for fill_lo, fill_hi in ((left_lo, left_hi), (right_lo, right_hi)):
                        fill_span = float(fill_hi) - float(fill_lo)
                        if fill_span <= 1e-4:
                            continue
                        fill_center = 0.5 * (float(fill_lo) + float(fill_hi))
                        fill_scale_along = max(0.05, fill_span / wall_step)
                        x, y = slot_point(slot, fill_center, inward=wall_thickness * 0.5)
                        conveyor_loader.spawn(
                            wall_model,
                            x=x,
                            y=y,
                            yaw_deg=wall_yaw,
                            floor_z=tier_base_z,
                            scale=(UNIFORM_SCALE * fill_scale_along, UNIFORM_SCALE, UNIFORM_SCALE),
                            with_collision=True,
                        )

            if slot == loading_slot and tier == 0 and blocked_indices:
                blocked_spans = [
                    (
                        float(straight_along[i]) - (wall_step * 0.5),
                        float(straight_along[i]) + (wall_step * 0.5),
                    )
                    for i in sorted(blocked_indices)
                    if reserved_left <= i < reserved_right
                ]
                fill_spans = _subtract_spans_1d(blocked_spans, loading_door_spans)
                for lo, hi in fill_spans:
                    fill_span = float(hi) - float(lo)
                    if fill_span <= 1e-4:
                        continue
                    along_center = 0.5 * (float(lo) + float(hi))
                    along_scale = max(0.05, fill_span / wall_step)
                    x, y = slot_point(slot, along_center, inward=wall_thickness * 0.5)
                    conveyor_loader.spawn(
                        wall_model,
                        x=x,
                        y=y,
                        yaw_deg=wall_yaw,
                        floor_z=tier_base_z,
                        scale=(UNIFORM_SCALE * along_scale, UNIFORM_SCALE, UNIFORM_SCALE),
                        with_collision=True,
                    )

    corner_half_x = corner_size[0] * 0.5
    corner_half_y = corner_size[1] * 0.5
    corner_defs = (
        (WAREHOUSE_SIZE_X * 0.5 - corner_half_x, WAREHOUSE_SIZE_Y * 0.5 - corner_half_y, 90.0),               
        (-WAREHOUSE_SIZE_X * 0.5 + corner_half_x, WAREHOUSE_SIZE_Y * 0.5 - corner_half_y, 180.0),             
        (-WAREHOUSE_SIZE_X * 0.5 + corner_half_x, -WAREHOUSE_SIZE_Y * 0.5 + corner_half_y, 270.0),            
        (WAREHOUSE_SIZE_X * 0.5 - corner_half_x, -WAREHOUSE_SIZE_Y * 0.5 + corner_half_y, 0.0),               
    )
    for tier in range(WALL_TIERS):
        tier_base_z = floor_top_z + tier * wall_h
        for x, y, yaw in corner_defs:
            conveyor_loader.spawn(
                corner_model,
                x=x,
                y=y,
                yaw_deg=yaw,
                floor_z=tier_base_z,
                scale=UNIFORM_SCALE,
                with_collision=True,
            )


    frame_min_v, frame_max_v = conveyor_loader._bounds(dock_frame_model, UNIFORM_SCALE)
    frame_anchor_x = (frame_min_v[0] + frame_max_v[0]) * 0.5
    frame_anchor_y = (frame_min_v[1] + frame_max_v[1]) * 0.5
    door_yaw = dock_inward_yaw_for_slot(loading_slot)
    frame_yaw = door_yaw

    for i, along in enumerate(door_centers):
        x, y = slot_point(
            loading_slot,
            along,
            inward=(loading_wall_thickness * 0.5) + DOCK_INWARD_NUDGE,
        )
        _spawn_mesh_with_anchor(
            loader=conveyor_loader,
            model_name=dock_frame_model,
            world_anchor_xyz=(x, y, floor_top_z),
            yaw_deg=frame_yaw,
            mesh_scale_xyz=(UNIFORM_SCALE, UNIFORM_SCALE, UNIFORM_SCALE),
            local_anchor_xyz=(frame_anchor_x, frame_anchor_y, 0.0),
            with_collision=True,
            use_texture=True,
            double_sided=False,
        )

        gate_model_name = gate_states[i]
        min_v, max_v = conveyor_loader._bounds(gate_model_name, UNIFORM_SCALE)
        cx = (min_v[0] + max_v[0]) * 0.5
        cy = max_v[1]
        gate_yaw = door_yaw
        _spawn_mesh_with_anchor(
            loader=conveyor_loader,
            model_name=gate_model_name,
            world_anchor_xyz=(x, y, floor_top_z),
            yaw_deg=gate_yaw,
            mesh_scale_xyz=(UNIFORM_SCALE, UNIFORM_SCALE, UNIFORM_SCALE),
            local_anchor_xyz=(cx, cy, 0.0),
            with_collision=True,
            use_texture=True,
            double_sided=True,
        )

    personnel_door_yaw = dock_inward_yaw_for_slot(personnel_slot)
    personnel_min_v, personnel_max_v = conveyor_loader._bounds(personnel_door_model, UNIFORM_SCALE)
    personnel_anchor_x = (personnel_min_v[0] + personnel_max_v[0]) * 0.5
    personnel_anchor_y = personnel_max_v[1]
    px, py = slot_point(
        personnel_slot,
        personnel_along,
        inward=(personnel_wall_thickness * 0.5) + DOCK_INWARD_NUDGE,
    )
    _spawn_mesh_with_anchor(
        loader=conveyor_loader,
        model_name=personnel_door_model,
        world_anchor_xyz=(px, py, floor_top_z),
        yaw_deg=personnel_door_yaw,
        mesh_scale_xyz=(UNIFORM_SCALE, UNIFORM_SCALE, UNIFORM_SCALE),
        local_anchor_xyz=(personnel_anchor_x, personnel_anchor_y, 0.0),
        with_collision=True,
        use_texture=True,
        double_sided=True,
    )

    return {
        "loading_side": loading_slot,
        "transverse_major_zone": transverse_major_zone,
        "personnel_side": personnel_slot,
        "personnel_along": personnel_along,
        "personnel_door_span": personnel_door_span,
        "door_centers": door_centers,
        "door_states": gate_states,
        "loading_corner_side": loading_corner_side,
        "wall_height": wall_h * WALL_TIERS,
        "wall_thickness": loading_wall_thickness,
        "floor_spawn_half_x": floor_spawn_half_x,
        "floor_spawn_half_y": floor_spawn_half_y,
        "roof_eave_z": floor_top_z + wall_h * WALL_TIERS,
    }


def build_loading_trucks(truck_loader, floor_top_z, wall_info):
    if not ENABLE_LOADING_TRUCKS:
        return {"truck_scale_xyz": LOADING_TRUCK_SCALE_XYZ, "loading_trucks": []}

    loading_side = wall_info.get("loading_side", "north")
    door_centers = list(wall_info.get("door_centers", []))
    door_states = list(wall_info.get("door_states", []))
    if not door_centers:
        return {"truck_scale_xyz": LOADING_TRUCK_SCALE_XYZ, "loading_trucks": []}

    outward_yaw = dock_inward_yaw_for_slot(loading_side)
    wall_thickness = float(wall_info.get("wall_thickness", 0.0))
    floor_spawn_half_x = float(wall_info.get("floor_spawn_half_x", (WAREHOUSE_SIZE_X * 0.5)))
    floor_spawn_half_y = float(wall_info.get("floor_spawn_half_y", (WAREHOUSE_SIZE_Y * 0.5)))

    trucks = []
    for i, along in enumerate(door_centers):
        model_name = LOADING_TRUCK_MODELS[i % len(LOADING_TRUCK_MODELS)]
        min_v, max_v = model_bounds_xyz(truck_loader, model_name, LOADING_TRUCK_SCALE_XYZ)
        sx = max_v[0] - min_v[0]
        sy = max_v[1] - min_v[1]
        sz = max_v[2] - min_v[2]
        yaw = math.radians(outward_yaw)
        c = abs(math.cos(yaw))
        s = abs(math.sin(yaw))
        ex = (c * sx) + (s * sy)
        ey = (s * sx) + (c * sy)
        depth_to_wall = ey if loading_side in ("north", "south") else ex
        inner_wall_face_inward = wall_thickness * 0.5
        wall_clearance = max(0.05, LOADING_TRUCK_WALL_GAP)
        gate_model_name = door_states[i] if i < len(door_states) else ""
        gate_extra_gap = _truck_extra_gap_for_gate_state(gate_model_name)
        inward = inner_wall_face_inward + (depth_to_wall * 0.5) + wall_clearance + gate_extra_gap
        x, y = slot_point(loading_side, along, inward=inward)
        x_min = -floor_spawn_half_x + (ex * 0.5)
        x_max = floor_spawn_half_x - (ex * 0.5)
        y_min = -floor_spawn_half_y + (ey * 0.5)
        y_max = floor_spawn_half_y - (ey * 0.5)
        if x_min > x_max or y_min > y_max:
            continue
        x = max(x_min, min(x_max, x))
        y = max(y_min, min(y_max, y))
        anchor_x = (min_v[0] + max_v[0]) * 0.5
        anchor_y = (min_v[1] + max_v[1]) * 0.5
        anchor_z = min_v[2]

        _spawn_mesh_with_anchor(
            loader=truck_loader,
            model_name=model_name,
            world_anchor_xyz=(x, y, floor_top_z),
            yaw_deg=outward_yaw,
            mesh_scale_xyz=LOADING_TRUCK_SCALE_XYZ,
            local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
            with_collision=True,
            use_texture=False,
            rgba=(1.0, 1.0, 1.0, 0.0),
            double_sided=False,
        )

        model_path = truck_loader._asset_path(model_name)
        material_parts = _obj_material_parts(model_path)
        if material_parts:
            for part in material_parts:
                _spawn_mesh_with_anchor(
                    loader=truck_loader,
                    model_name=part["path"],
                    world_anchor_xyz=(x, y, floor_top_z),
                    yaw_deg=outward_yaw,
                    mesh_scale_xyz=LOADING_TRUCK_SCALE_XYZ,
                    local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                    with_collision=False,
                    use_texture=False,
                    rgba=part["rgba"],
                    double_sided=False,
                )
        else:
            _spawn_mesh_with_anchor(
                loader=truck_loader,
                model_name=model_name,
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=outward_yaw,
                mesh_scale_xyz=LOADING_TRUCK_SCALE_XYZ,
                local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                with_collision=False,
                use_texture=False,
                rgba=(0.85, 0.85, 0.85, 1.0),
                double_sided=False,
            )
        if loading_side in ("north", "south"):
            along_out = float(x)
        else:
            along_out = float(y)
        if loading_side == "north":
            inward_out = (WAREHOUSE_SIZE_Y * 0.5) - float(y)
        elif loading_side == "south":
            inward_out = float(y) + (WAREHOUSE_SIZE_Y * 0.5)
        elif loading_side == "east":
            inward_out = (WAREHOUSE_SIZE_X * 0.5) - float(x)
        else:
            inward_out = float(x) + (WAREHOUSE_SIZE_X * 0.5)
        trucks.append(
            {
                "model": model_name,
                "size_xyz_m": (sx, sy, sz),
                "footprint_xy_m": (ex, ey),
                "yaw_deg": outward_yaw,
                "x": x,
                "y": y,
                "along": along_out,
                "inward": inward_out,
                "gate_state": gate_model_name,
            }
        )

    return {
        "truck_scale_xyz": LOADING_TRUCK_SCALE_XYZ,
        "loading_trucks": trucks,
    }


def _spawn_obj_with_mtl_parts(
    loader,
    model_name,
    world_anchor_xyz,
    yaw_deg,
    mesh_scale_xyz,
    local_anchor_xyz,
    with_collision=True,
    fallback_rgba=(0.78, 0.78, 0.78, 1.0),
    rgba_gain=1.0,
):
    def _gain_rgba(c):
        r = max(0.0, min(1.0, float(c[0]) * float(rgba_gain)))
        g = max(0.0, min(1.0, float(c[1]) * float(rgba_gain)))
        b = max(0.0, min(1.0, float(c[2]) * float(rgba_gain)))
        a = float(c[3]) if len(c) >= 4 else 1.0
        return (r, g, b, a)

    if with_collision:
        _spawn_mesh_with_anchor(
            loader=loader,
            model_name=model_name,
            world_anchor_xyz=world_anchor_xyz,
            yaw_deg=yaw_deg,
            mesh_scale_xyz=mesh_scale_xyz,
            local_anchor_xyz=local_anchor_xyz,
            with_collision=True,
            use_texture=False,
            rgba=(1.0, 1.0, 1.0, 0.0),
            double_sided=False,
        )

    model_path = loader._asset_path(model_name)
    material_parts = _obj_material_parts(model_path)
    if material_parts:
        for part in material_parts:
            part_rgba = _gain_rgba(part["rgba"])
            _spawn_mesh_with_anchor(
                loader=loader,
                model_name=part["path"],
                world_anchor_xyz=world_anchor_xyz,
                yaw_deg=yaw_deg,
                mesh_scale_xyz=mesh_scale_xyz,
                local_anchor_xyz=local_anchor_xyz,
                with_collision=False,
                use_texture=False,
                rgba=part_rgba,
                double_sided=False,
            )
        return len(material_parts)

    fallback_rgba_adj = _gain_rgba(fallback_rgba)
    _spawn_mesh_with_anchor(
        loader=loader,
        model_name=model_name,
        world_anchor_xyz=world_anchor_xyz,
        yaw_deg=yaw_deg,
        mesh_scale_xyz=mesh_scale_xyz,
        local_anchor_xyz=local_anchor_xyz,
        with_collision=False,
        use_texture=False,
        rgba=fallback_rgba_adj,
        double_sided=False,
    )
    return 0


def _truss_rib_x_positions(shell_meshes):
    cfg = (shell_meshes or {}).get("config", {}) or {}
    rib_count = max(1, int(cfg.get("truss_rib_count", 5)))
    base_x = float(cfg.get("warehouse_size_x", WAREHOUSE_BASE_SIZE_X))
    end_margin_base = max(0.0, float(cfg.get("truss_end_margin_x", 6.0)))
    shell_sx = (float(WAREHOUSE_SIZE_X) / base_x) if abs(base_x) > 1e-9 else 1.0
    end_margin = end_margin_base * shell_sx

    half_x = float(WAREHOUSE_SIZE_X) * 0.5
    lo = -half_x + end_margin
    hi = half_x - end_margin
    if hi <= lo:
        lo = -half_x * 0.8
        hi = half_x * 0.8
    if rib_count <= 1:
        return [0.0]
    step = (hi - lo) / float(rib_count - 1)
    return [float(lo + (i * step)) for i in range(rib_count)]


def build_overhead_cranes(crane_loader, crane_model_name, floor_top_z, roof_base_z, area_layout, shell_meshes, seed=0):
    _ = seed
    if not ENABLE_OVERHEAD_CRANES:
        return {"overhead_cranes_enabled": False}
    if crane_loader is None or not crane_model_name:
        return {
            "overhead_cranes_enabled": False,
            "overhead_cranes_reason": "Crane model not found.",
        }

    zones = area_layout or {}
    targets = []
    for zone_name, count in OVERHEAD_CRANE_TARGET_BY_ZONE:
        if zone_name in zones:
            targets.append((str(zone_name), max(0, int(count))))
    if not targets:
        return {
            "overhead_cranes_enabled": False,
            "overhead_cranes_reason": "Target zones missing in layout.",
        }

    s = float(OVERHEAD_CRANE_SCALE_UNIFORM)
    scale_xyz = (s, s, s)
    min_v, max_v = model_bounds_xyz(crane_loader, crane_model_name, scale_xyz)
    crane_height = max(0.1, float(max_v[2] - min_v[2]))
    anchor_x = (float(min_v[0]) + float(max_v[0])) * 0.5
    anchor_y = (float(min_v[1]) + float(max_v[1])) * 0.5
    anchor_z = float(max_v[2])                        

    rib_xs = list(_truss_rib_x_positions(shell_meshes))
    if not rib_xs:
        rib_xs = [0.0]

    cranes = []
    min_spacing = max(0.0, float(OVERHEAD_CRANE_MIN_SPACING_M))
    edge_margin = max(0.0, float(OVERHEAD_CRANE_ZONE_EDGE_MARGIN_M))

    def _zone_span_candidates(area, count):
        sx = float(area["sx"])
        sy = float(area["sy"])
        cx = float(area["cx"])
        cy = float(area["cy"])
        hx = max(0.1, (sx * 0.5) - edge_margin)
        hy = max(0.1, (sy * 0.5) - edge_margin)

        if count <= 1:
            return [(cx, cy, 90.0 if sy > sx else 0.0)]
        if count == 2:
            fracs = (-0.34, 0.34)
        elif count == 3:
            fracs = (-0.38, 0.0, 0.38)
        else:
            fracs = [(-0.40 + (0.80 * float(i) / float(max(1, count - 1)))) for i in range(count)]

        out = []
        if sx >= sy:
            side_jitter = min(1.3, hy * 0.24)
            for i, f in enumerate(fracs):
                y_off = side_jitter if (i % 2) else -side_jitter
                out.append((cx + (f * hx), cy + y_off, 0.0))
        else:
            side_jitter = min(1.3, hx * 0.24)
            for i, f in enumerate(fracs):
                x_off = side_jitter if (i % 2) else -side_jitter
                out.append((cx + x_off, cy + (f * hy), 90.0))
        return out

    def _far_enough(x, y):
        for prev in cranes:
            if math.hypot(float(x) - float(prev["x"]), float(y) - float(prev["y"])) < (min_spacing - 1e-6):
                return False
        return True

    for zone_name, count in targets:
        if count <= 0:
            continue
        area = zones.get(zone_name)
        if not area:
            continue
        sx = float(area["sx"])
        sy = float(area["sy"])
        cx = float(area["cx"])
        cy = float(area["cy"])
        min_x = cx - (sx * 0.5) + edge_margin
        max_x = cx + (sx * 0.5) - edge_margin
        min_y = cy - (sy * 0.5) + edge_margin
        max_y = cy + (sy * 0.5) - edge_margin
        preferred = _zone_span_candidates(area, count)

        for pref_x, pref_y, yaw_deg in preferred:
            y = max(min_y, min(max_y, float(pref_y)))
            rib_choices = [x for x in rib_xs if (min_x - 1e-6) <= x <= (max_x + 1e-6)]
            if rib_choices:
                rib_choices.sort(key=lambda x: abs(float(x) - float(pref_x)))
                x_candidates = rib_choices
            else:
                x_candidates = [max(min_x, min(max_x, float(pref_x)))]

            picked = None
            for x in x_candidates:
                if not _far_enough(x, y):
                    continue
                picked = (float(x), float(y), float(yaw_deg))
                break
            if picked is None and x_candidates:
                x = float(x_candidates[0])
                if _far_enough(x, y):
                    picked = (x, float(y), float(yaw_deg))
            if picked is None:
                continue

            x, y, yaw_deg = picked
            support_end_z = float(roof_base_z) + float(OVERHEAD_CRANE_TRUSS_TOUCH_EXTRA_M)
            anchor_world_z = float(support_end_z) - float(OVERHEAD_CRANE_ATTACH_CLEARANCE_M)
            if (anchor_world_z - crane_height) < (float(floor_top_z) + 1.2):
                anchor_world_z = float(floor_top_z) + 1.2 + crane_height

            yaw_deg = float(OVERHEAD_CRANE_YAW_EXTRA_DEG) % 360.0

            _spawn_obj_with_mtl_parts(
                loader=crane_loader,
                model_name=crane_model_name,
                world_anchor_xyz=(x, y, anchor_world_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=scale_xyz,
                local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                with_collision=OVERHEAD_CRANE_WITH_COLLISION,
                fallback_rgba=(0.80, 0.72, 0.20, 1.0),
                rgba_gain=OVERHEAD_CRANE_COLOR_GAIN,
            )
            cranes.append(
                {
                    "zone": zone_name,
                    "x": x,
                    "y": y,
                    "z_top_anchor": float(anchor_world_z),
                    "z_hook_bottom": float(anchor_world_z - crane_height),
                    "yaw_deg": float(yaw_deg),
                    "scale_uniform": s,
                }
            )

    return {
        "overhead_cranes_enabled": len(cranes) > 0,
        "overhead_crane_model": crane_model_name,
        "overhead_crane_scale_uniform": s,
        "overhead_crane_count": len(cranes),
        "overhead_cranes": cranes,
    }


def build_loading_staging(loading_loader, floor_top_z, area_layout, wall_info, seed=0):
    if not ENABLE_LOADING_STAGING:
        return {"loading_staging_enabled": False}
    if loading_loader is None:
        return {
            "loading_staging_enabled": False,
            "loading_staging_reason": (
                "Loading staging assets loader unavailable. Expected one of: "
                + ", ".join(LOADING_STAGING_ASSET_CANDIDATES)
            ),
        }

    loading_area = (area_layout or {}).get("LOADING")
    if not loading_area:
        return {
            "loading_staging_enabled": False,
            "loading_staging_reason": "LOADING area not found in area layout.",
        }

    rng = random.Random(int(seed) + 45019)

    def _build_spec(model_name, scale_xyz):
        min_v, max_v = model_bounds_xyz(loading_loader, model_name, scale_xyz)
        return {
            "model_name": model_name,
            "scale_xyz": scale_xyz,
            "min_v": min_v,
            "max_v": max_v,
            "size_xyz": (
                max_v[0] - min_v[0],
                max_v[1] - min_v[1],
                max_v[2] - min_v[2],
            ),
            "anchor_xyz": (
                (min_v[0] + max_v[0]) * 0.5,
                (min_v[1] + max_v[1]) * 0.5,
                min_v[2],
            ),
        }

    specs = {}
    try:
        for key in ("pallet", "box", "barrel"):
            model_name = str(LOADING_STAGING_MODELS[key])
            scale_xyz = tuple(float(v) for v in LOADING_STAGING_SCALES[key])
            specs[key] = _build_spec(model_name, scale_xyz)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "loading_staging_enabled": False,
            "loading_staging_reason": f"Failed to prepare loading staging assets: {exc}",
        }

    container_spec = None
    container_reason = ""
    if LOADING_CONTAINER_STACK_ENABLED:
        try:
            container_spec = _build_spec(
                str(LOADING_CONTAINER_MODEL_NAME),
                tuple(float(v) for v in LOADING_CONTAINER_SCALE_XYZ),
            )
        except (FileNotFoundError, ValueError):
            container_reason = (
                "Missing container model. Add: "
                f"{LOADING_CONTAINER_MODEL_NAME} + "
                f"{os.path.splitext(LOADING_CONTAINER_MODEL_NAME)[0]}.mtl"
            )

    area_cx = float(loading_area["cx"])
    area_cy = float(loading_area["cy"])
    area_sx = float(loading_area["sx"])
    area_sy = float(loading_area["sy"])
    x_min = area_cx - (area_sx * 0.5)
    x_max = area_cx + (area_sx * 0.5)
    y_min = area_cy - (area_sy * 0.5)
    y_max = area_cy + (area_sy * 0.5)

    loading_side = str(wall_info.get("loading_side", "north")).lower()
    if loading_side not in WALL_SLOTS:
        loading_side = "north"

    if loading_side in ("north", "south"):
        along_axis = "x"
        along_min = x_min
        along_max = x_max
        dock_edge = y_max if loading_side == "north" else y_min
        interior_edge = y_min if loading_side == "north" else y_max
    else:
        along_axis = "y"
        along_min = y_min
        along_max = y_max
        dock_edge = x_max if loading_side == "east" else x_min
        interior_edge = x_min if loading_side == "east" else x_max

    cross_to_dock_sign = 1.0 if dock_edge >= interior_edge else -1.0
    cross_depth_total = abs(dock_edge - interior_edge)
    area_along_span = max(0.0, along_max - along_min)
    if area_along_span <= 0.1 or cross_depth_total <= 0.1:
        return {
            "loading_staging_enabled": False,
            "loading_staging_reason": "LOADING area has invalid dimensions.",
        }

    truck_depth = 0.0
    for truck in wall_info.get("loading_trucks", []):
        fx, fy = truck.get("footprint_xy_m", (0.0, 0.0))
        if loading_side in ("north", "south"):
            truck_depth = max(truck_depth, float(fy))
        else:
            truck_depth = max(truck_depth, float(fx))
    dock_clearance = max(6.0, truck_depth + float(LOADING_STAGING_TRUCK_TAIL_CLEARANCE_M))

    edge_margin = max(0.4, float(LOADING_STAGING_EDGE_MARGIN_M))
    usable_depth = cross_depth_total - dock_clearance
    if usable_depth <= (edge_margin + 0.2):
        return {
            "loading_staging_enabled": False,
            "loading_staging_reason": "Not enough depth in LOADING zone after dock-truck clearance.",
        }
    s_min = edge_margin
    s_max = min(usable_depth, edge_margin + float(LOADING_STAGING_MAX_DEPTH_M))
    if s_max <= (s_min + 0.2):
        return {
            "loading_staging_enabled": False,
            "loading_staging_reason": "Staging strip collapsed by current LOADING dimensions.",
        }

    def _xy_from_along_s(along, s_from_interior):
        cross = interior_edge + (cross_to_dock_sign * s_from_interior)
        if along_axis == "x":
            return along, cross
        return cross, along

    def _oriented_xy(spec, yaw_deg):
        sx, sy, _sz = spec["size_xyz"]
        yaw = math.radians(yaw_deg)
        c = abs(math.cos(yaw))
        s = abs(math.sin(yaw))
        ex = (c * sx) + (s * sy)
        ey = (s * sx) + (c * sy)
        along_extent = ex if along_axis == "x" else ey
        cross_extent = ey if along_axis == "x" else ex
        return ex, ey, along_extent, cross_extent

    def _range_len(rng_pair):
        return max(0.0, float(rng_pair[1]) - float(rng_pair[0]))

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    def _valid_range(rng_pair, min_len=0.5):
        return _range_len(rng_pair) >= float(min_len)

    yaw_along = 0.0 if along_axis == "x" else 90.0
    pallet_spec = specs["pallet"]
    box_spec = specs["box"]
    barrel_spec = specs["barrel"]
    p_ex, p_ey, p_along, p_cross = _oriented_xy(pallet_spec, yaw_along)
    b_ex, b_ey, _b_along, _b_cross = _oriented_xy(box_spec, yaw_along)

    along_start = along_min + edge_margin
    along_end = along_max - edge_margin
    if along_end <= along_start:
        return {
            "loading_staging_enabled": False,
            "loading_staging_reason": "Loading along-span collapsed after edge margins.",
        }

    door_centers = [float(v) for v in wall_info.get("door_centers", [])]
    if door_centers:
        dmin = min(door_centers)
        dmax = max(door_centers)
    else:
        dmin = dmax = 0.5 * (along_start + along_end)
    door_pad = 3.6
    truck_min = _clamp(dmin - door_pad, along_start, along_end)
    truck_max = _clamp(dmax + door_pad, along_start, along_end)
    truck_center = 0.5 * (truck_min + truck_max)
    truck_width = max(float(LOADING_SECTION_MIN_SPAN_M), truck_max - truck_min)
    truck_half = truck_width * 0.5
    truck_min = _clamp(truck_center - truck_half, along_start, along_end)
    truck_max = _clamp(truck_center + truck_half, along_start, along_end)
    if (truck_max - truck_min) < LOADING_SECTION_MIN_SPAN_M:
        if abs(truck_min - along_start) <= 1e-6:
            truck_max = min(along_end, truck_min + LOADING_SECTION_MIN_SPAN_M)
        elif abs(truck_max - along_end) <= 1e-6:
            truck_min = max(along_start, truck_max - LOADING_SECTION_MIN_SPAN_M)

    seg_gap = 1.0
    left_range = (along_start, truck_min - seg_gap)
    right_range = (truck_max + seg_gap, along_end)
    left_len = _range_len(left_range)
    right_len = _range_len(right_range)
    zone_center = 0.5 * (along_start + along_end)
    gate_center = 0.5 * (dmin + dmax)
    gate_bias = (gate_center - zone_center) / max(1.0, (along_end - along_start))

    if left_len >= LOADING_SECTION_MIN_SPAN_M and right_len >= LOADING_SECTION_MIN_SPAN_M and abs(gate_bias) <= 0.12:
        gate_position = "center"
    else:
        gate_position = "min" if gate_bias < 0.0 else "max"

    goods_range = None
    container_range = None

    def _split_outer_range(base_range, near_at_start):
        base_len = _range_len(base_range)
        if base_len <= 0.5:
            return None, None
        min_span = max(4.5, float(LOADING_SECTION_MIN_SPAN_M) * 0.6)
        if base_len < (min_span * 2.0 + seg_gap):
            mid = 0.5 * (base_range[0] + base_range[1])
            if near_at_start:
                g = (base_range[0], mid - (seg_gap * 0.5))
                c = (mid + (seg_gap * 0.5), base_range[1])
            else:
                c = (base_range[0], mid - (seg_gap * 0.5))
                g = (mid + (seg_gap * 0.5), base_range[1])
            return g, c

        goods_len = _clamp(base_len * 0.58, min_span, base_len - min_span - seg_gap)
        if near_at_start:
            g = (base_range[0], base_range[0] + goods_len)
            c = (g[1] + seg_gap, base_range[1])
        else:
            g = (base_range[1] - goods_len, base_range[1])
            c = (base_range[0], g[0] - seg_gap)
        return g, c

    if gate_position == "center" and _valid_range(left_range) and _valid_range(right_range):
        office_area = (area_layout or {}).get("OFFICE")
        office_along = float(office_area["cx"] if along_axis == "x" else office_area["cy"]) if office_area else zone_center
        left_mid = 0.5 * (left_range[0] + left_range[1])
        right_mid = 0.5 * (right_range[0] + right_range[1])
        if abs(left_mid - office_along) >= abs(right_mid - office_along):
            container_range = left_range
            goods_range = right_range
        else:
            container_range = right_range
            goods_range = left_range
    elif gate_position == "min":
        base = right_range if _range_len(right_range) >= _range_len(left_range) else left_range
        near_start = abs(base[0] - (truck_max + seg_gap)) <= 1e-6
        goods_range, container_range = _split_outer_range(base, near_at_start=near_start)
    else:
        base = left_range if _range_len(left_range) >= _range_len(right_range) else right_range
        near_start = abs(base[0] - (truck_max + seg_gap)) <= 1e-6
        goods_range, container_range = _split_outer_range(base, near_at_start=near_start)

    if not _valid_range(goods_range, min_len=max(4.0, p_along + 1.0)):
        fallback_ranges = sorted([left_range, right_range], key=lambda r: _range_len(r), reverse=True)
        goods_range = fallback_ranges[0] if fallback_ranges else (along_start, along_end)
        container_range = fallback_ranges[1] if len(fallback_ranges) > 1 else None

    if not _valid_range(container_range, min_len=4.0):
        container_range = None

    if _valid_range(left_range, min_len=4.0) or _valid_range(right_range, min_len=4.0):
        if _range_len(left_range) >= _range_len(right_range):
            container_range = left_range if _valid_range(left_range, min_len=4.0) else right_range
        else:
            container_range = right_range if _valid_range(right_range, min_len=4.0) else left_range
    truck_mid = 0.5 * (truck_min + truck_max)

    box_count = 0
    barrel_count = 0
    pallet_count = 0
    container_count = 0
    spawned_items = []
    container_entries = []
    empty_stack_entries = []
    empty_stack_groups = []

    def _spawn_prop(spec, x, y, z_anchor, yaw_deg, with_collision):
        _spawn_obj_with_mtl_parts(
            loader=loading_loader,
            model_name=spec["model_name"],
            world_anchor_xyz=(x, y, z_anchor),
            yaw_deg=yaw_deg,
            mesh_scale_xyz=spec["scale_xyz"],
            local_anchor_xyz=spec["anchor_xyz"],
            with_collision=with_collision,
            fallback_rgba=(0.74, 0.74, 0.74, 1.0),
        )

    def _spawn_container(x, y, z_anchor, yaw_deg, with_collision=True, body_rgba=None):
        if container_spec is None:
            return

        if with_collision:
            _spawn_collision_only_with_anchor(
                loader=loading_loader,
                model_name=container_spec["model_name"],
                world_anchor_xyz=(x, y, z_anchor),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=container_spec["scale_xyz"],
                local_anchor_xyz=container_spec["anchor_xyz"],
            )

        model_path = loading_loader._asset_path(container_spec["model_name"])
        material_parts = _obj_material_parts(model_path)
        if material_parts:
            for part in material_parts:
                part_rgba = tuple(float(v) for v in part.get("rgba", (0.70, 0.70, 0.70, 1.0)))
                mtl_name = str(part.get("material", "")).lower()
                if body_rgba is not None:
                    if "container" in mtl_name:
                        part_rgba = body_rgba
                    elif "metal" in mtl_name or "bar" in mtl_name:
                        part_rgba = (0.80, 0.80, 0.82, 1.0)
                else:
                    if "container" in mtl_name:
                        part_rgba = (0.63, 0.17, 0.16, 1.0)
                visual_mesh_path = _obj_double_sided_proxy_path(part["path"])
                _spawn_mesh_with_anchor(
                    loader=loading_loader,
                    model_name=visual_mesh_path,
                    world_anchor_xyz=(x, y, z_anchor),
                    yaw_deg=yaw_deg,
                    mesh_scale_xyz=container_spec["scale_xyz"],
                    local_anchor_xyz=container_spec["anchor_xyz"],
                    with_collision=False,
                    use_texture=False,
                    rgba=part_rgba,
                    double_sided=False,
                )
            return

        visual_model_path = _obj_double_sided_proxy_path(model_path)
        _spawn_mesh_with_anchor(
            loader=loading_loader,
            model_name=visual_model_path,
            world_anchor_xyz=(x, y, z_anchor),
            yaw_deg=yaw_deg,
            mesh_scale_xyz=container_spec["scale_xyz"],
            local_anchor_xyz=container_spec["anchor_xyz"],
            with_collision=False,
            use_texture=False,
            rgba=body_rgba if body_rgba is not None else (0.64, 0.20, 0.18, 1.0),
            double_sided=False,
        )

    def _spawn_loaded_pallet_with_boxes(px, py, yaw_deg, stack_layers=1):
        nonlocal pallet_count, box_count
        stack_layers = int(max(1, stack_layers))
        pallet_h = float(pallet_spec["size_xyz"][2])
        box_h = float(box_spec["size_xyz"][2])
        pallet_over_cargo_gap = 0.01
        yaw_rad = math.radians(yaw_deg)
        ux = (math.cos(yaw_rad), math.sin(yaw_rad))
        uy = (-math.sin(yaw_rad), math.cos(yaw_rad))
        pallet_along = p_along
        pallet_cross = p_cross

        desired_gap = 0.03
        edge_margin = 0.01
        candidates = []
        for test_yaw in (yaw_deg % 360.0, (yaw_deg + 90.0) % 360.0):
            _bx, _by, box_along, box_cross = _oriented_xy(box_spec, test_yaw)
            max_gap_a = pallet_along - (2.0 * box_along) - (2.0 * edge_margin)
            max_gap_c = pallet_cross - (2.0 * box_cross) - (2.0 * edge_margin)
            if max_gap_a >= 0.0 and max_gap_c >= 0.0:
                usable_gap = min(desired_gap, max_gap_a, max_gap_c)
                candidates.append((usable_gap, test_yaw, box_along, box_cross))

        if candidates:
            candidates.sort(key=lambda t: t[0], reverse=True)
            gap_used, box_yaw, box_along, box_cross = candidates[0]
            off_a = (box_along * 0.5) + (gap_used * 0.5)
            off_c = (box_cross * 0.5) + (gap_used * 0.5)
            off_a_max = max(0.0, (pallet_along * 0.5) - (box_along * 0.5) - edge_margin)
            off_c_max = max(0.0, (pallet_cross * 0.5) - (box_cross * 0.5) - edge_margin)
            off_a = min(off_a, off_a_max)
            off_c = min(off_c, off_c_max)
            slots = (
                (-off_a, -off_c),
                (off_a, -off_c),
                (-off_a, off_c),
                (off_a, off_c),
            )
        else:
            box_yaw = yaw_deg % 360.0
            _bx, _by, box_along, _box_cross = _oriented_xy(box_spec, box_yaw)
            off_a_max = max(0.0, (pallet_along * 0.5) - (box_along * 0.5) - edge_margin)
            off_a = min(off_a_max, max(0.10, 0.5 * box_along))
            slots = (
                (-off_a, 0.0),
                (off_a, 0.0),
            )

        next_pallet_z = floor_top_z
        for tier_idx in range(stack_layers):
            pallet_z = next_pallet_z
            _spawn_prop(pallet_spec, px, py, pallet_z, yaw_deg, with_collision=True)
            pallet_count += 1
            spawned_items.append(
                {
                    "type": "pallet",
                    "x": px,
                    "y": py,
                    "z": pallet_z,
                    "yaw_deg": yaw_deg,
                    "cargo": "box",
                    "stack_layer": tier_idx,
                }
            )

            box_z = pallet_z + pallet_h
            for ox_local, oy_local in slots:
                bx = px + (ux[0] * ox_local) + (uy[0] * oy_local)
                by = py + (ux[1] * ox_local) + (uy[1] * oy_local)
                _spawn_prop(box_spec, bx, by, box_z, box_yaw, with_collision=False)
                box_count += 1
                spawned_items.append(
                    {
                        "type": "box",
                        "x": bx,
                        "y": by,
                        "z": box_z,
                        "yaw_deg": box_yaw,
                        "stack_layer": tier_idx,
                    }
                )

            cargo_top_z = box_z + box_h
            next_pallet_z = cargo_top_z + pallet_over_cargo_gap

    def _spawn_barrel_pallet(px, py, yaw_deg, stack_layers=1):
        nonlocal pallet_count, barrel_count
        stack_layers = min(int(LOADING_BARREL_MAX_STACK_LAYERS), int(max(1, stack_layers)))
        pallet_h = float(pallet_spec["size_xyz"][2])
        barrel_h = float(barrel_spec["size_xyz"][2])
        pallet_over_cargo_gap = 0.01
        yaw_rad = math.radians(yaw_deg)
        ux = (math.cos(yaw_rad), math.sin(yaw_rad))
        uy = (-math.sin(yaw_rad), math.cos(yaw_rad))
        barrel_yaw_base = yaw_deg
        _rx, _ry, barrel_along, barrel_cross = _oriented_xy(barrel_spec, barrel_yaw_base)
        desired_gap = 0.04
        need_off_a = 0.5 * (barrel_along + desired_gap)
        need_off_c = 0.5 * (barrel_cross + desired_gap)
        max_off_a = max(0.0, 0.5 * (p_along - barrel_along) - 0.01)
        max_off_c = max(0.0, 0.5 * (p_cross - barrel_cross) - 0.01)
        if max_off_a >= need_off_a and max_off_c >= need_off_c:
            off_a = min(max_off_a, need_off_a)
            off_c = min(max_off_c, need_off_c)
            barrel_slots = (
                (-off_a, -off_c),
                (off_a, -off_c),
                (-off_a, off_c),
                (off_a, off_c),
            )
        else:
            off_a = min(max_off_a, max(0.10, 0.5 * barrel_along))
            barrel_slots = ((-off_a, 0.0), (off_a, 0.0))

        next_pallet_z = floor_top_z
        for tier_idx in range(stack_layers):
            pallet_z = next_pallet_z
            _spawn_prop(pallet_spec, px, py, pallet_z, yaw_deg, with_collision=True)
            pallet_count += 1
            spawned_items.append(
                {
                    "type": "pallet",
                    "x": px,
                    "y": py,
                    "z": pallet_z,
                    "yaw_deg": yaw_deg,
                    "cargo": "barrel",
                    "stack_layer": tier_idx,
                }
            )

            barrel_z = pallet_z + pallet_h
            for k, (ox_local, oy_local) in enumerate(barrel_slots):
                bx = px + (ux[0] * ox_local) + (uy[0] * oy_local)
                by = py + (ux[1] * ox_local) + (uy[1] * oy_local)
                barrel_yaw = (barrel_yaw_base + (90.0 if (k % 2 == 1) else 0.0)) % 360.0
                _spawn_prop(barrel_spec, bx, by, barrel_z, barrel_yaw, with_collision=False)
                barrel_count += 1
                spawned_items.append(
                    {
                        "type": "barrel",
                        "x": bx,
                        "y": by,
                        "z": barrel_z,
                        "yaw_deg": barrel_yaw,
                        "stack_layer": tier_idx,
                    }
                )

            cargo_top_z = barrel_z + barrel_h
            next_pallet_z = cargo_top_z + pallet_over_cargo_gap

    goods_layout_range = (
        max(along_start, truck_min + 0.6),
        min(along_end, truck_max - 0.6),
    )
    if not _valid_range(goods_layout_range, min_len=max(5.0, (2.0 * p_along) + 0.6)):
        goods_layout_range = goods_range

    empty_stack_count = max(1, int(LOADING_EMPTY_PALLET_STACK_COUNT))
    empty_stack_min_layers = max(1, int(LOADING_EMPTY_PALLET_STACK_MIN_LAYERS))
    empty_stack_max_layers = max(empty_stack_min_layers, int(LOADING_EMPTY_PALLET_STACK_MAX_LAYERS))
    along_half = p_along * 0.5
    side_gap = 0.8
    left_near_gate = (truck_min - side_gap) - along_half
    left_far_end = along_start + along_half
    right_near_gate = (truck_max + side_gap) + along_half
    right_far_end = along_end - along_half

    left_len = max(0.0, left_near_gate - left_far_end)
    right_len = max(0.0, right_far_end - right_near_gate)
    left_range = (min(left_near_gate, left_far_end), max(left_near_gate, left_far_end))
    right_range = (min(right_near_gate, right_far_end), max(right_near_gate, right_far_end))
    left_valid = _valid_range(left_range, min_len=max(2.2, p_along * 1.4))
    right_valid = _valid_range(right_range, min_len=max(2.2, p_along * 1.4))

    left_raw = max(0.0, truck_min - along_start)
    right_raw = max(0.0, along_end - truck_max)
    container_min_span_for_one = 1.8
    if container_spec is not None:
        c_size_x = float(container_spec["size_xyz"][0])
        c_size_y = float(container_spec["size_xyz"][1])
        min_need = None
        for cand_yaw in ((yaw_along + 90.0) % 360.0, yaw_along % 360.0):
            c = abs(math.cos(math.radians(cand_yaw)))
            s = abs(math.sin(math.radians(cand_yaw)))
            ex = (c * c_size_x) + (s * c_size_y)
            ey = (s * c_size_x) + (c * c_size_y)
            along_need = ex if along_axis == "x" else ey
            if min_need is None or along_need < min_need:
                min_need = along_need
        if min_need is not None:
            container_min_span_for_one = max(1.8, float(min_need) + 0.05)

    left_can_container = _range_len(left_range) >= container_min_span_for_one
    right_can_container = _range_len(right_range) >= container_min_span_for_one
    container_min_span_for_three = max(container_min_span_for_one, 3.0 * container_min_span_for_one)
    left_can_three = _range_len(left_range) >= container_min_span_for_three
    right_can_three = _range_len(right_range) >= container_min_span_for_three

    if right_can_three and left_can_three:
        base_container_on_right = _range_len(right_range) >= _range_len(left_range)
    elif right_can_three:
        base_container_on_right = True
    elif left_can_three:
        base_container_on_right = False
    elif right_can_container and left_can_container:
        base_container_on_right = _range_len(right_range) >= _range_len(left_range)
    elif right_can_container:
        base_container_on_right = True
    elif left_can_container:
        base_container_on_right = False
    else:
        base_container_on_right = right_raw >= left_raw

    preferred_side_right = not base_container_on_right
    preferred_can_three = right_can_three if preferred_side_right else left_can_three
    other_can_three = left_can_three if preferred_side_right else right_can_three
    preferred_can_one = right_can_container if preferred_side_right else left_can_container
    other_can_one = left_can_container if preferred_side_right else right_can_container
    if preferred_can_three:
        container_on_right = preferred_side_right
    elif other_can_three:
        container_on_right = not preferred_side_right
    elif preferred_can_one:
        container_on_right = preferred_side_right
    elif other_can_one:
        container_on_right = not preferred_side_right
    else:
        container_on_right = _range_len(right_range) >= _range_len(left_range)
    empty_on_right = container_on_right

    same_span = right_range if container_on_right else left_range
    same_lo = float(same_span[0])
    same_hi = float(same_span[1])
    same_len = max(0.0, same_hi - same_lo)
    gap_along_pref = max(0.55, container_min_span_for_one * 0.16)
    cluster_span_pref = max(
        6.0,
        (3.0 * container_min_span_for_one) + (2.0 * gap_along_pref),
    )
    container_span_use = min(same_len, cluster_span_pref)
    if container_span_use < container_min_span_for_one:
        container_span_use = min(container_min_span_for_one, same_len)

    if container_on_right:
        container_lo = same_lo
        container_hi = min(same_hi, container_lo + container_span_use)
        empty_lo = container_lo
        empty_hi = container_hi
        container_start_along = container_hi
        container_end_along = container_lo
        container_dir = -1.0
        container_gate_edge_along = container_lo
        empty_start_along = empty_lo
        empty_end_along = empty_hi
        empty_dir = 1.0
    else:
        container_hi = same_hi
        container_lo = max(same_lo, container_hi - container_span_use)
        empty_lo = container_lo
        empty_hi = container_hi
        container_start_along = container_lo
        container_end_along = container_hi
        container_dir = 1.0
        container_gate_edge_along = container_hi
        empty_start_along = empty_hi
        empty_end_along = empty_lo
        empty_dir = -1.0

    container_candidate_range = (container_lo, container_hi)

    if abs(empty_end_along - empty_start_along) < 0.2:
        tight_side_gap = 0.25
        if empty_on_right:
            empty_start_along = (truck_max + tight_side_gap) + along_half
            empty_end_along = along_end - along_half
            empty_dir = 1.0
        else:
            empty_start_along = (truck_min - tight_side_gap) - along_half
            empty_end_along = along_start + along_half
            empty_dir = -1.0

        if abs(empty_end_along - empty_start_along) < 0.2:
            if empty_on_right:
                anchor = _clamp(
                    truck_max + along_half + 0.10,
                    along_start + along_half,
                    along_end - along_half,
                )
            else:
                anchor = _clamp(
                    truck_min - along_half - 0.10,
                    along_start + along_half,
                    along_end - along_half,
                )
            empty_start_along = anchor
            empty_end_along = anchor

    container_range = container_candidate_range if _valid_range(container_candidate_range, min_len=1.8) else None

    _container_cross_reserve = 0.0
    if container_spec is not None and container_range is not None:
        c_sx = float(container_spec["size_xyz"][0])
        c_sy = float(container_spec["size_xyz"][1])
        for _cand_yaw in ((yaw_along + 90.0) % 360.0, yaw_along % 360.0):
            _c = abs(math.cos(math.radians(_cand_yaw)))
            _s = abs(math.sin(math.radians(_cand_yaw)))
            _ex = (_c * c_sx) + (_s * c_sy)
            _ey = (_s * c_sx) + (_c * c_sy)
            _cross = _ey if along_axis == "x" else _ex
            if _cross <= (cross_depth_total + 1e-6):
                _container_cross_reserve = max(_container_cross_reserve, _cross)

    empty_container_gap = max(1.30, p_cross * 0.70)
    if _container_cross_reserve > 0.0:
        empty_cross_front = cross_depth_total - _container_cross_reserve - empty_container_gap - (p_cross * 0.5)
    else:
        empty_cross_front = cross_depth_total - (p_cross * 0.5) - 0.12
    empty_cross_front = _clamp(
        empty_cross_front,
        s_min + (p_cross * 0.5),
        cross_depth_total - (p_cross * 0.5) - 0.02,
    )

    goods_s_min = max(0.0, min(s_min, float(LOADING_STAGING_GOODS_BACK_EDGE_PAD_M)))
    goods_s_max = s_max
    goods_cross_span = max(0.0, goods_s_max - goods_s_min)
    cross_span = goods_cross_span
    row_gap = max(0.45, float(LOADING_STAGING_PROP_GAP_M))
    col_gap_default = max(0.60, float(LOADING_STAGING_PROP_GAP_M) + 0.20)
    max_rows_fit = int((goods_cross_span + row_gap) // (p_cross + row_gap))
    if max_rows_fit < 1:
        return {
            "loading_staging_enabled": False,
            "loading_staging_reason": "Not enough room for loaded pallets in goods section.",
        }

    truck_alongs = sorted(float(t.get("along", 0.0)) for t in wall_info.get("loading_trucks", []))
    if not truck_alongs:
        truck_alongs = [float(v) for v in door_centers] if door_centers else [0.5 * (goods_layout_range[0] + goods_layout_range[1])]
    truck_alongs = [a for a in truck_alongs if goods_layout_range[0] <= a <= goods_layout_range[1]]
    if not truck_alongs:
        truck_alongs = [0.5 * (goods_layout_range[0] + goods_layout_range[1])]

    truck_lanes = []
    for i, a in enumerate(truck_alongs):
        lo = goods_layout_range[0] if i == 0 else 0.5 * (truck_alongs[i - 1] + a)
        hi = goods_layout_range[1] if i == (len(truck_alongs) - 1) else 0.5 * (a + truck_alongs[i + 1])
        lane_margin = 0.06
        lane_lo = lo + lane_margin
        lane_hi = hi - lane_margin
        if lane_hi > lane_lo:
            truck_lanes.append((i, a, lane_lo, lane_hi))

    loaded_stack_min = max(1, int(LOADING_LOADED_PALLET_STACK_MIN_LAYERS))
    loaded_stack_max = max(loaded_stack_min, int(LOADING_LOADED_PALLET_STACK_MAX_LAYERS))
    truck_row_centers_used = []

    for truck_idx, truck_along, lane_lo, lane_hi in truck_lanes:
        lane_len = max(0.0, lane_hi - lane_lo)
        cols_fit = int((lane_len + col_gap_default) // (p_along + col_gap_default))
        if cols_fit < 1:
            continue

        bundles_min = min(int(LOADING_BUNDLES_PER_TRUCK_MIN), int(LOADING_BUNDLES_PER_TRUCK_MAX))
        bundles_max = max(int(LOADING_BUNDLES_PER_TRUCK_MIN), int(LOADING_BUNDLES_PER_TRUCK_MAX))
        bundles_min = max(4, bundles_min)
        bundles_max = max(bundles_min, min(6, bundles_max))
        target_bundles = rng.randint(bundles_min, bundles_max)

        use_two_rows = max_rows_fit >= 2
        max_bundle_capacity = cols_fit * (2 if use_two_rows else 1)
        bundle_count = max(1, min(target_bundles, max_bundle_capacity))

        row_bundle_counts = [bundle_count]
        if use_two_rows and bundle_count >= 2:
            row0_min = max(1, bundle_count - cols_fit)
            row0_max = min(cols_fit, bundle_count - 1)
            if row0_min <= row0_max:
                row0_count = rng.randint(row0_min, row0_max)
                row_bundle_counts = [row0_count, bundle_count - row0_count]

        row0_back_pad = 0.0
        row0_center_s = goods_s_min + (p_cross * 0.5) + row0_back_pad
        row0_center_s = _clamp(
            row0_center_s,
            goods_s_min + (p_cross * 0.5),
            goods_s_max - (p_cross * 0.5),
        )
        row_centers_s = [row0_center_s]
        if len(row_bundle_counts) >= 2:
            row_step = p_cross + row_gap
            row1_min = row0_center_s + max(0.35, p_cross * 0.65)
            row1_max = goods_s_max - (p_cross * 0.5) - max(
                1.20,
                float(LOADING_STAGING_TRUCK_TAIL_CLEARANCE_M) + 0.80,
            )
            if row1_max > row1_min:
                row1_center_s = _clamp(row0_center_s + row_step, row1_min, row1_max)
                row_centers_s.append(row1_center_s)
            else:
                row_bundle_counts = [bundle_count]

        bundle_count = sum(row_bundle_counts)
        stack_layers_per_bundle = [rng.randint(loaded_stack_min, loaded_stack_max) for _ in range(bundle_count)]
        row_cargo_modes = []
        mixed_barrel_indices = set()
        if len(row_bundle_counts) >= 2:
            row_cargo_modes = ["box", "barrel"] + [
                ("barrel" if rng.random() < 0.45 else "box")
                for _ in range(max(0, len(row_bundle_counts) - 2))
            ]
            rng.shuffle(row_cargo_modes)
        elif row_bundle_counts:
            mixed_count = int(row_bundle_counts[0])
            barrel_target = min(
                mixed_count,
                max(1, int(round(float(mixed_count) * 0.35))),
            )
            if barrel_target > 0:
                mixed_barrel_indices = set(rng.sample(range(mixed_count), barrel_target))
            row_cargo_modes = ["mixed"]

        bundle_cursor = 0
        for row_idx, row_count in enumerate(row_bundle_counts):
            if row_count <= 0:
                continue
            row_cargo = row_cargo_modes[row_idx] if row_idx < len(row_cargo_modes) else "box"

            if row_count <= 1:
                col_gap_use = 0.0
                total_cols_len = p_along
            else:
                fit_gap = (lane_len - (row_count * p_along)) / float(row_count - 1)
                col_gap_use = max(0.0, min(col_gap_default, fit_gap))
                total_cols_len = (row_count * p_along) + ((row_count - 1) * col_gap_use)

            first_col_center = lane_lo + ((lane_len - total_cols_len) * 0.5) + (p_along * 0.5)
            s_center = row_centers_s[min(row_idx, len(row_centers_s) - 1)]
            truck_row_centers_used.append(float(s_center))

            for col_idx in range(row_count):
                along = first_col_center + (col_idx * (p_along + col_gap_use))
                px, py = _xy_from_along_s(along, s_center)
                layers = (
                    stack_layers_per_bundle[bundle_cursor]
                    if bundle_cursor < len(stack_layers_per_bundle)
                    else 1
                )
                spawn_barrel = False
                if row_cargo == "barrel":
                    spawn_barrel = True
                elif row_cargo == "mixed":
                    spawn_barrel = col_idx in mixed_barrel_indices
                bundle_cursor += 1
                if spawn_barrel:
                    _spawn_barrel_pallet(px, py, yaw_along, stack_layers=layers)
                else:
                    _spawn_loaded_pallet_with_boxes(px, py, yaw_along, stack_layers=layers)

    side_spans = []
    container_side_name = None
    if container_range is not None:
        container_mid = 0.5 * (float(container_range[0]) + float(container_range[1]))
        container_side_name = "left" if container_mid <= truck_mid else "right"
    for side_name, span in (("left", left_range), ("right", right_range)):
        if container_side_name is not None and side_name == container_side_name:
            continue
        if _valid_range(span, min_len=max(2.6, p_along * 1.4)):
            side_spans.append((side_name, span))

    if side_spans:
        min_truck_s = min(truck_row_centers_used) if truck_row_centers_used else (usable_depth - (p_cross * 0.5))
        support_rows_s = []
        support_row_step = p_cross + max(0.60, row_gap)
        support_s_min = s_min + (p_cross * 0.5) + 0.20
        max_support_s = min_truck_s - (p_cross + 0.65)
        max_support_s = min(max_support_s, empty_cross_front - (p_cross + 0.65))
        if max_support_s >= (support_s_min - 1e-6):
            back_bias = max(0.0, min(1.0, float(LOADING_STAGING_SUPPORT_BACK_BIAS)))
            s_anchor = support_s_min + (max_support_s - support_s_min) * back_bias
            s_val = s_anchor
            while s_val >= (support_s_min - 1e-6):
                support_rows_s.append(float(s_val))
                if len(support_rows_s) >= 3:
                    break
                s_val -= support_row_step

        if support_rows_s:
            support_min_layers = max(1, loaded_stack_min)
            support_max_layers = max(support_min_layers, min(2, loaded_stack_max))
            for side_name, span in side_spans:
                lo = float(span[0]) + 0.06
                hi = float(span[1]) - 0.06
                span_len = max(0.0, hi - lo)
                cols_fit = int((span_len + col_gap_default) // (p_along + col_gap_default))
                if cols_fit < 1:
                    continue

                cols_use = max(1, min(cols_fit, 5))
                if cols_use <= 1:
                    col_gap_use = 0.0
                    total_cols_len = p_along
                else:
                    fit_gap = (span_len - (cols_use * p_along)) / float(cols_use - 1)
                    col_gap_use = max(0.0, min(col_gap_default, fit_gap))
                    total_cols_len = (cols_use * p_along) + ((cols_use - 1) * col_gap_use)
                first_col_center = lo + ((span_len - total_cols_len) * 0.5) + (p_along * 0.5)

                side_pref_barrel = side_name == "left"
                for ridx, s_center in enumerate(support_rows_s):
                    for cidx in range(cols_use):
                        along = first_col_center + (cidx * (p_along + col_gap_use))
                        px, py = _xy_from_along_s(along, s_center)
                        layers = rng.randint(support_min_layers, support_max_layers)
                        use_barrel = (ridx % 2 == 0) if side_pref_barrel else (ridx % 2 == 1)
                        if use_barrel:
                            _spawn_barrel_pallet(px, py, yaw_along, stack_layers=layers)
                        else:
                            _spawn_loaded_pallet_with_boxes(px, py, yaw_along, stack_layers=layers)

    container_target_center_along = None
    empty_slot_positions = []
    span_abs = abs(empty_end_along - empty_start_along)
    lo_along = min(empty_start_along, empty_end_along)
    hi_along = max(empty_start_along, empty_end_along)

    slot_step_need = max(0.80, p_along + 0.10)
    max_cols_fit = max(1, int(span_abs // slot_step_need) + 1)
    if max_cols_fit >= 3:
        rows_use = 3
    elif max_cols_fit >= 2:
        rows_use = 2
    else:
        rows_use = 1
    max_total_fit = max_cols_fit * rows_use
    fit_stack_count = max(1, min(empty_stack_count, max_total_fit))

    row_gap_extra = max(0.55, row_gap + 0.20)
    row_step = p_cross + row_gap_extra
    row_cross_values = [empty_cross_front]
    for _ in range(rows_use - 1):
        next_cross = row_cross_values[-1] - row_step
        if next_cross >= (s_min + (p_cross * 0.5)):
            row_cross_values.append(next_cross)
    if not row_cross_values:
        row_cross_values = [empty_cross_front]
    rows_use = len(row_cross_values)

    row_counts = []
    base_per_row = fit_stack_count // rows_use
    extra = fit_stack_count % rows_use
    for ridx in range(rows_use):
        row_counts.append(base_per_row + (1 if ridx < extra else 0))

    span_center = 0.5 * (empty_start_along + empty_end_along)
    span_len = abs(empty_end_along - empty_start_along)
    dir_sign = 1.0 if empty_end_along >= empty_start_along else -1.0
    step_pref = max(0.85, p_along + 0.12)
    for ridx, row_count in enumerate(row_counts):
        if row_count <= 0:
            continue
        if row_count <= 1:
            row_alongs = [span_center]
        else:
            step_max = span_len / float(row_count - 1)
            step_use = min(step_pref, max(0.20, step_max))
            cluster_half = 0.5 * step_use * float(max(0, row_count - 1))
            row_alongs = [span_center + (dir_sign * ((float(i) * step_use) - cluster_half)) for i in range(row_count)]
        row_cross = row_cross_values[min(ridx, len(row_cross_values) - 1)]
        for along in row_alongs:
            empty_slot_positions.append((along, row_cross))

    empty_cross_values = [float(cross_s) for _along, cross_s in empty_slot_positions]
    for slot_idx, (along, cross_s) in enumerate(empty_slot_positions):
        along = _clamp(along, lo_along, hi_along)
        empty_x, empty_y = _xy_from_along_s(along, cross_s)
        layer_count = rng.randint(empty_stack_min_layers, empty_stack_max_layers)
        empty_stack_groups.append(
            {
                "x": empty_x,
                "y": empty_y,
                "slot": slot_idx,
                "layers": layer_count,
            }
        )
        for level in range(layer_count):
            z_anchor = floor_top_z + (level * (pallet_spec["size_xyz"][2] + 0.002))
            _spawn_prop(
                pallet_spec,
                empty_x,
                empty_y,
                z_anchor,
                yaw_along,
                with_collision=True,
            )
            empty_stack_entries.append(
                {
                    "x": empty_x,
                    "y": empty_y,
                    "z": z_anchor,
                    "slot": slot_idx,
                    "row": 0,
                    "level": level,
                }
            )
            spawned_items.append(
                {
                    "type": "empty_pallet",
                    "x": empty_x,
                    "y": empty_y,
                    "z": z_anchor,
                    "slot": slot_idx,
                    "row": 0,
                    "level": level,
                }
            )

    def _place_container_stack_in_range(target_range):
        nonlocal container_count, container_reason
        if container_spec is None or target_range is None:
            return False
        have_along = _range_len(target_range)
        if have_along <= 0.1:
            return False

        yaw_candidates = ((yaw_along + 90.0) % 360.0, yaw_along % 360.0)
        c_size_x = float(container_spec["size_xyz"][0])
        c_size_y = float(container_spec["size_xyz"][1])
        c_size_z = float(container_spec["size_xyz"][2])

        def _container_oriented_xy(yaw_deg):
            c = abs(math.cos(math.radians(yaw_deg)))
            s = abs(math.sin(math.radians(yaw_deg)))
            ex = (c * c_size_x) + (s * c_size_y)
            ey = (s * c_size_x) + (c * c_size_y)
            along_extent = ex if along_axis == "x" else ey
            cross_extent = ey if along_axis == "x" else ex
            return along_extent, cross_extent

        layout_options = (
            (3, 2),             
            (2, 1),
            (2, 0),
            (1, 0),
        )
        best = None
        for cand_yaw in yaw_candidates:
            c_along, c_cross = _container_oriented_xy(cand_yaw)
            if c_cross > (cross_depth_total + 1e-6):
                continue
            for base_count, upper_count in layout_options:
                if upper_count >= base_count:
                    continue
                need_along = base_count * c_along
                if need_along <= (have_along + 1e-6):
                    total_target = base_count + upper_count
                    score = (total_target, -c_along)
                    cand = (score, cand_yaw, c_along, c_cross, base_count, upper_count, total_target)
                    if best is None or cand[0] > best[0]:
                        best = cand
                    break

        if best is None:
            return False

        _score, container_yaw, c_along, c_cross, base_count, upper_count, total_target = best
        range_lo = float(target_range[0])
        range_hi = float(target_range[1])
        center_lo = range_lo + (c_along * 0.5)
        center_hi = range_hi - (c_along * 0.5)
        if center_hi < center_lo:
            return False

        gap_pref = max(0.55, c_along * 0.16)
        if base_count <= 1:
            gap_along = 0.0
        else:
            section_span = max(0.0, range_hi - range_lo)
            gap_max = max(0.0, (section_span - (base_count * c_along)) / float(base_count - 1))
            gap_along = min(gap_pref, gap_max)
        step = c_along + gap_along

        dir_sign = 1.0 if container_dir >= 0.0 else -1.0
        if dir_sign > 0.0:
            max_first = center_hi - ((base_count - 1) * step)
            first_center = _clamp(max_first, center_lo, center_hi)
        else:
            min_first = center_lo + ((base_count - 1) * step)
            first_center = _clamp(min_first, center_lo, center_hi)

        target_cross_edge = float(cross_depth_total)
        row_cross = target_cross_edge - (c_cross * 0.5)
        outer_face_cross = row_cross + (c_cross * 0.5)
        cross_shift = target_cross_edge - outer_face_cross
        if abs(cross_shift) > 1e-9:
            row_cross += cross_shift

        base_alongs = [first_center + (dir_sign * i * step) for i in range(base_count)]
        if base_alongs:
            edge_target = float(container_gate_edge_along)
            outer_center = max(base_alongs) if dir_sign > 0.0 else min(base_alongs)
            outer_face = outer_center + (dir_sign * (c_along * 0.5))
            along_shift = edge_target - outer_face
            if abs(along_shift) > 1e-9:
                base_alongs = [a + along_shift for a in base_alongs]

        before = int(container_count)
        for along in base_alongs:
            cx, cy = _xy_from_along_s(along, row_cross)
            _spawn_container(cx, cy, floor_top_z, container_yaw, with_collision=True, body_rgba=None)
            container_count += 1
            container_entries.append({"x": cx, "y": cy, "level": 0})

        if upper_count >= 1 and len(base_alongs) >= 2:
            container_h = float(c_size_z)
            top_z = floor_top_z + container_h + float(LOADING_CONTAINER_STACK_VERTICAL_GAP_M)
            if upper_count >= 2 and len(base_alongs) >= 3:
                upper_alongs = [
                    0.5 * (base_alongs[0] + base_alongs[1]),
                    0.5 * (base_alongs[1] + base_alongs[2]),
                ]
            else:
                upper_alongs = [0.5 * (base_alongs[0] + base_alongs[-1])]
            upper_alongs = upper_alongs[:upper_count]
            for along in upper_alongs:
                cx, cy = _xy_from_along_s(along, row_cross)
                _spawn_container(cx, cy, top_z, container_yaw, with_collision=True, body_rgba=None)
                container_count += 1
                container_entries.append({"x": cx, "y": cy, "level": 1})

        placed_now = int(container_count) - before
        if placed_now < total_target and not container_reason:
            container_reason = (
                f"Container stack fallback: placed {placed_now}/{total_target} in LOADING section."
            )
        return placed_now > 0

    placed_container = False
    if container_spec is not None and container_range is not None:
        placed_container = _place_container_stack_in_range(container_range)
        if not placed_container and not container_reason:
            container_reason = "Container section too small for preferred 3+2 stack."
    if container_spec is not None and container_count <= 0:
        full_fallback_range = (
            along_start + 0.2,
            along_end - 0.2,
        )
        if _valid_range(full_fallback_range, min_len=0.6):
            if _place_container_stack_in_range(full_fallback_range):
                if not container_reason:
                    container_reason = "Container placed using full LOADING fallback range."
        elif not container_reason:
            container_reason = "Unable to place any container in LOADING zone."
    return {
        "loading_staging_enabled": True,
        "loading_staging_area": "LOADING",
        "loading_section_gate_position": gate_position,
        "loading_section_truck_range": (truck_min, truck_max),
        "loading_section_goods_range": goods_layout_range,
        "loading_section_container_range": container_range,
        "loading_staging_pallet_count": pallet_count,
        "loading_staging_box_count": box_count,
        "loading_staging_barrel_count": barrel_count,
        "loading_empty_pallet_stack_count": len(empty_stack_groups),
        "loading_empty_pallet_total_count": len(empty_stack_entries),
        "loading_container_count": container_count,
        "loading_container_entries": container_entries,
        "loading_container_reason": container_reason,
        "loading_pallet_size_xy_m": (round(p_ex, 2), round(p_ey, 2)),
        "loading_box_size_xy_m": (round(b_ex, 2), round(b_ey, 2)),
        "loading_staging_items": spawned_items,
    }


def build_storage_racks(storage_loader, floor_top_z, area_layout, wall_info, seed=0):
    if not ENABLE_STORAGE_RACK_LAYOUT:
        return {"storage_rack_enabled": False}
    if storage_loader is None:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "Storage loader unavailable.",
        }

    storage_area = (area_layout or {}).get("STORAGE")
    if not storage_area:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "STORAGE area not found in area layout.",
        }

    rack_model = str(STORAGE_RACK_MODEL_NAME)
    rack_scale = (
        float(STORAGE_RACK_SCALE_UNIFORM),
        float(STORAGE_RACK_SCALE_UNIFORM),
        float(STORAGE_RACK_SCALE_UNIFORM),
    )
    pallet_model = str(LOADING_STAGING_MODELS["pallet"])
    box_model = str(LOADING_STAGING_MODELS["box"])
    barrel_model = str(LOADING_STAGING_MODELS["barrel"])
    pallet_scale = tuple(float(v) for v in LOADING_STAGING_SCALES["pallet"])
    box_scale = tuple(float(v) for v in LOADING_STAGING_SCALES["box"])
    barrel_scale = tuple(float(v) for v in LOADING_STAGING_SCALES["barrel"])

    try:
        rack_min_v, rack_max_v = model_bounds_xyz(storage_loader, rack_model, rack_scale)
        pallet_min_v, pallet_max_v = model_bounds_xyz(storage_loader, pallet_model, pallet_scale)
        box_min_v, box_max_v = model_bounds_xyz(storage_loader, box_model, box_scale)
        barrel_min_v, barrel_max_v = model_bounds_xyz(storage_loader, barrel_model, barrel_scale)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": f"Failed to prepare storage rack assets: {exc}",
        }

    rack_size_x = float(rack_max_v[0] - rack_min_v[0])
    rack_size_y = float(rack_max_v[1] - rack_min_v[1])
    rack_size_z = float(rack_max_v[2] - rack_min_v[2])
    rack_anchor_x = float((rack_min_v[0] + rack_max_v[0]) * 0.5)
    rack_anchor_y = float((rack_min_v[1] + rack_max_v[1]) * 0.5)
    rack_anchor_z = float(rack_min_v[2])

    pallet_size_x = float(pallet_max_v[0] - pallet_min_v[0])
    pallet_size_y = float(pallet_max_v[1] - pallet_min_v[1])
    pallet_size_z = float(pallet_max_v[2] - pallet_min_v[2])
    pallet_anchor_x = float((pallet_min_v[0] + pallet_max_v[0]) * 0.5)
    pallet_anchor_y = float((pallet_min_v[1] + pallet_max_v[1]) * 0.5)
    pallet_anchor_z = float(pallet_min_v[2])

    box_size_x = float(box_max_v[0] - box_min_v[0])
    box_size_y = float(box_max_v[1] - box_min_v[1])
    box_size_z = float(box_max_v[2] - box_min_v[2])
    box_anchor_x = float((box_min_v[0] + box_max_v[0]) * 0.5)
    box_anchor_y = float((box_min_v[1] + box_max_v[1]) * 0.5)
    box_anchor_z = float(box_min_v[2])

    barrel_size_x = float(barrel_max_v[0] - barrel_min_v[0])
    barrel_size_y = float(barrel_max_v[1] - barrel_min_v[1])
    barrel_size_z = float(barrel_max_v[2] - barrel_min_v[2])
    barrel_anchor_x = float((barrel_min_v[0] + barrel_max_v[0]) * 0.5)
    barrel_anchor_y = float((barrel_min_v[1] + barrel_max_v[1]) * 0.5)
    barrel_anchor_z = float(barrel_min_v[2])

    area_cx = float(storage_area["cx"])
    area_cy = float(storage_area["cy"])
    area_sx = float(storage_area["sx"])
    area_sy = float(storage_area["sy"])
    x_min = area_cx - (area_sx * 0.5)
    x_max = area_cx + (area_sx * 0.5)
    y_min = area_cy - (area_sy * 0.5)
    y_max = area_cy + (area_sy * 0.5)

    floor_half_x = float(wall_info.get("floor_spawn_half_x", 0.0))
    floor_half_y = float(wall_info.get("floor_spawn_half_y", 0.0))
    if floor_half_x <= 0.0 or floor_half_y <= 0.0:
        floor_half_x, floor_half_y = _floor_spawn_half_extents(storage_loader)

    storage_layout_seed = (
        int(STORAGE_RACK_LAYOUT_FIXED_SEED)
        if STORAGE_RACK_LAYOUT_FIXED_SEED is not None
        else int(seed)
    )
    rng = random.Random(int(storage_layout_seed) + 73129)
    pallets_per_level = max(1, int(STORAGE_RACK_PALLETS_PER_LEVEL))
    top_level_drop_prob = max(0.0, min(0.95, float(STORAGE_RACK_NO_TOP_LEVEL_PROBABILITY)))
    level_min_clear_m = float(STORAGE_RACK_LEVEL_MIN_CLEAR_M)
    rack_z_limit = float(floor_top_z) + float(rack_size_z)

    oriented_xy_local_cache = {}
    barrel_layout_profile_cache = {}
    box_layout_profile_cache = {}

    def _yaw_key(yaw_deg):
        return round(float(yaw_deg) % 360.0, 6)

    def _oriented_xy_cached(model_name, scale_xyz, yaw_deg):
        key = (str(model_name), tuple(float(v) for v in scale_xyz), _yaw_key(yaw_deg))
        cached = oriented_xy_local_cache.get(key)
        if cached is not None:
            return cached
        out = oriented_xy_size(storage_loader, model_name, scale_xyz, key[2])
        oriented_xy_local_cache[key] = out
        return out

    def _barrel_layout_profile_for_slot_yaw(slot_yaw):
        key = _yaw_key(slot_yaw)
        cached = barrel_layout_profile_cache.get(key)
        if cached is not None:
            return cached

        barrel_edge_margin = 0.01
        target_gap = 0.04
        barrel_layout_candidates = []
        for swap_axes in (False, True):
            if swap_axes:
                barrel_local_x = barrel_size_y
                barrel_local_y = barrel_size_x
                barrel_yaw = (key + 90.0) % 360.0
            else:
                barrel_local_x = barrel_size_x
                barrel_local_y = barrel_size_y
                barrel_yaw = key
            max_gap_x = pallet_size_x - (2.0 * barrel_local_x) - (2.0 * barrel_edge_margin)
            max_gap_y = pallet_size_y - (2.0 * barrel_local_y) - (2.0 * barrel_edge_margin)
            if max_gap_x < -1e-6 or max_gap_y < -1e-6:
                continue
            use_gap = max(0.0, min(target_gap, max_gap_x, max_gap_y))
            barrel_layout_candidates.append((use_gap, barrel_yaw, barrel_local_x, barrel_local_y))

        if barrel_layout_candidates:
            barrel_layout_candidates.sort(key=lambda t: float(t[0]), reverse=True)
            use_gap, barrel_yaw, barrel_local_x, barrel_local_y = barrel_layout_candidates[0]
            off_x = (barrel_local_x * 0.5) + (use_gap * 0.5)
            off_y = (barrel_local_y * 0.5) + (use_gap * 0.5)
            off_x_max = max(0.0, (pallet_size_x * 0.5) - (barrel_local_x * 0.5) - barrel_edge_margin)
            off_y_max = max(0.0, (pallet_size_y * 0.5) - (barrel_local_y * 0.5) - barrel_edge_margin)
            off_x = min(off_x, off_x_max)
            off_y = min(off_y, off_y_max)
            layer1_slots = [(-off_x, -off_y), (off_x, -off_y), (-off_x, off_y), (off_x, off_y)]
        else:
            barrel_yaw = key
            barrel_local_x = barrel_size_x
            off_x_max = max(0.0, (pallet_size_x * 0.5) - (barrel_local_x * 0.5) - barrel_edge_margin)
            off_x = min(off_x_max, max(0.08, barrel_local_x * 0.5))
            layer1_slots = [(-off_x, 0.0), (off_x, 0.0)]

        bex, bey = _oriented_xy_cached(barrel_model, barrel_scale, barrel_yaw)
        out = {
            "barrel_yaw": float(barrel_yaw),
            "layer1_slots": tuple((float(x), float(y)) for x, y in layer1_slots),
            "bex": float(bex),
            "bey": float(bey),
        }
        barrel_layout_profile_cache[key] = out
        return out

    def _box_layout_profile_for_slot_yaw(slot_yaw):
        key = _yaw_key(slot_yaw)
        cached = box_layout_profile_cache.get(key)
        if cached is not None:
            return cached

        box_edge_margin = 0.01
        target_gap = 0.03
        layout_candidates = []
        for swap_axes in (False, True):
            if swap_axes:
                box_local_x = box_size_y
                box_local_y = box_size_x
                box_yaw = (key + 90.0) % 360.0
            else:
                box_local_x = box_size_x
                box_local_y = box_size_y
                box_yaw = key
            max_gap_x = pallet_size_x - (2.0 * box_local_x) - (2.0 * box_edge_margin)
            max_gap_y = pallet_size_y - (2.0 * box_local_y) - (2.0 * box_edge_margin)
            if max_gap_x < -1e-6 or max_gap_y < -1e-6:
                continue
            use_gap = max(0.0, min(target_gap, max_gap_x, max_gap_y))
            layout_candidates.append((use_gap, box_yaw, box_local_x, box_local_y))

        if layout_candidates:
            layout_candidates.sort(key=lambda t: float(t[0]), reverse=True)
            use_gap, box_yaw, box_local_x, box_local_y = layout_candidates[0]
            off_x = (box_local_x * 0.5) + (use_gap * 0.5)
            off_y = (box_local_y * 0.5) + (use_gap * 0.5)
            off_x_max = max(0.0, (pallet_size_x * 0.5) - (box_local_x * 0.5) - box_edge_margin)
            off_y_max = max(0.0, (pallet_size_y * 0.5) - (box_local_y * 0.5) - box_edge_margin)
            off_x = min(off_x, off_x_max)
            off_y = min(off_y, off_y_max)
            layer1_slots = [(-off_x, -off_y), (off_x, -off_y), (-off_x, off_y), (off_x, off_y)]
        else:
            box_yaw = key
            box_local_x = box_size_x
            off_x_max = max(0.0, (pallet_size_x * 0.5) - (box_local_x * 0.5) - box_edge_margin)
            off_x = min(off_x_max, max(0.08, box_local_x * 0.5))
            layer1_slots = [(-off_x, 0.0), (off_x, 0.0)]

        bex, bey = _oriented_xy_cached(box_model, box_scale, box_yaw)
        out = {
            "box_yaw": float(box_yaw),
            "layer1_slots": tuple((float(x), float(y)) for x, y in layer1_slots),
            "bex": float(bex),
            "bey": float(bey),
        }
        box_layout_profile_cache[key] = out
        return out

    def _packed_centers(lo, hi, size, gap):
        span = float(hi) - float(lo)
        if span < (float(size) - 1e-6):
            return []
        step = float(size) + max(0.0, float(gap))
        count = max(1, int(math.floor((span + max(0.0, float(gap))) / step)))
        used = (count * float(size)) + ((count - 1) * max(0.0, float(gap)))
        slack = max(0.0, span - used)
        start = float(lo) + (slack * 0.5) + (float(size) * 0.5)
        return [start + (i * step) for i in range(count)]

    edge_margin = max(0.35, float(STORAGE_RACK_EDGE_MARGIN_M))
    row_gap = max(1.4, float(STORAGE_RACK_ROW_GAP_M))
    slot_gap = max(0.25, float(STORAGE_RACK_SLOT_GAP_M))
    main_aisle = max(0.0, float(STORAGE_RACK_MAIN_AISLE_M))

    target_row_count = max(1, int(STORAGE_RACK_TARGET_ROW_COUNT))
    forced_axis_cfg = str(STORAGE_RACK_FORCE_ALONG_AXIS).strip().lower()
    if forced_axis_cfg in ("x", "y"):
        primary_along_axis = forced_axis_cfg
        axis_order = (forced_axis_cfg,)
    else:
        primary_along_axis = "x" if area_sx >= area_sy else "y"
        axis_order = ("x", "y") if primary_along_axis == "x" else ("y", "x")
    plan_candidates = []
    for preferred_along_axis in axis_order:
        yaw_candidates = []
        for rack_yaw in (0.0, 90.0):
            rack_ex, rack_ey = _oriented_xy_cached(rack_model, rack_scale, rack_yaw)
            along_size = float(rack_ex if preferred_along_axis == "x" else rack_ey)
            cross_size = float(rack_ey if preferred_along_axis == "x" else rack_ex)
            yaw_candidates.append(
                {
                    "yaw_deg": float(rack_yaw),
                    "rack_ex": float(rack_ex),
                    "rack_ey": float(rack_ey),
                    "along_size": along_size,
                    "cross_size": cross_size,
                    "is_long_along": along_size >= cross_size,
                }
            )

        yaw_candidates = sorted(
            yaw_candidates,
            key=lambda c: (
                1 if c["is_long_along"] else 0,
                float(c["along_size"]),
                -float(c["cross_size"]),
            ),
            reverse=True,
        )

        for cand in yaw_candidates:
            rack_yaw = float(cand["yaw_deg"])
            along_size = float(cand["along_size"])
            cross_size = float(cand["cross_size"])
            if preferred_along_axis == "x":
                along_lo = max(x_min + edge_margin + (along_size * 0.5), -floor_half_x + (along_size * 0.5))
                along_hi = min(x_max - edge_margin - (along_size * 0.5), floor_half_x - (along_size * 0.5))
                cross_lo = max(y_min + edge_margin + (cross_size * 0.5), -floor_half_y + (cross_size * 0.5))
                cross_hi = min(y_max - edge_margin - (cross_size * 0.5), floor_half_y - (cross_size * 0.5))
            else:
                along_lo = max(y_min + edge_margin + (along_size * 0.5), -floor_half_y + (along_size * 0.5))
                along_hi = min(y_max - edge_margin - (along_size * 0.5), floor_half_y - (along_size * 0.5))
                cross_lo = max(x_min + edge_margin + (cross_size * 0.5), -floor_half_x + (cross_size * 0.5))
                cross_hi = min(x_max - edge_margin - (cross_size * 0.5), floor_half_x - (cross_size * 0.5))
            if along_hi <= along_lo or cross_hi <= cross_lo:
                continue

            cross_span = float(cross_hi - cross_lo)
            max_rows_fit = max(1, int(math.floor((cross_span + row_gap) / max(1e-6, (cross_size + row_gap)))))
            rows_use = min(target_row_count, max_rows_fit)
            if rows_use <= 1:
                cross_centers = [0.5 * (cross_lo + cross_hi)]
            else:
                start = cross_lo + (cross_size * 0.5)
                end = cross_hi - (cross_size * 0.5)
                if end <= start:
                    continue
                step = (end - start) / float(rows_use - 1)
                cross_centers = [start + (i * step) for i in range(rows_use)]

            along_ranges = [(along_lo, along_hi)]
            slots = []
            along_centers_by_bank = {}
            for bank_idx, (alo, ahi) in enumerate(along_ranges):
                along_centers = _packed_centers(alo, ahi, along_size, slot_gap)
                if not along_centers:
                    continue
                along_centers_by_bank[int(bank_idx)] = [float(v) for v in along_centers]
                for row_idx, cross_v in enumerate(cross_centers):
                    for col_idx, along_v in enumerate(along_centers):
                        if preferred_along_axis == "x":
                            sx = float(along_v)
                            sy = float(cross_v)
                        else:
                            sx = float(cross_v)
                            sy = float(along_v)
                        slots.append(
                            {
                                "x": sx,
                                "y": sy,
                                "row": int(row_idx),
                                "col": int(col_idx),
                                "bank": int(bank_idx),
                                "along": float(along_v),
                            }
                        )
            if not slots:
                continue

            max_cols = max((len(v) for v in along_centers_by_bank.values()), default=0)
            plan_candidates.append(
                {
                "yaw_deg": rack_yaw,
                "along_axis": preferred_along_axis,
                "slots": slots,
                "rack_ex": float(cand["rack_ex"]),
                "rack_ey": float(cand["rack_ey"]),
                "along_size": float(along_size),
                "cross_size": float(cross_size),
                "along_lo": float(along_lo),
                "along_hi": float(along_hi),
                "cross_lo": float(cross_lo),
                "cross_hi": float(cross_hi),
                "along_centers_by_bank": along_centers_by_bank,
                "cross_rows": [float(v) for v in cross_centers],
                "is_long_along": bool(cand["is_long_along"]),
                "slot_count": int(len(slots)),
                "row_count": int(len(cross_centers)),
                "max_cols": int(max_cols),
                }
            )

    best_plan = None
    if plan_candidates:
        def _plan_score(plan):
            return (
                1 if bool(plan.get("is_long_along", False)) else 0,
                1 if str(plan.get("along_axis", "")) == str(primary_along_axis) else 0,
                int(plan.get("row_count", 0)),
                int(plan.get("max_cols", 0)),
                int(plan.get("slot_count", 0)),
            )

        best_plan = max(plan_candidates, key=_plan_score)

    if best_plan is None or not best_plan["slots"]:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "No valid storage rack grid fits STORAGE zone on floor.",
        }

    def _even_pick_indices(total_n, target_n):
        total_n = int(total_n)
        target_n = int(target_n)
        if target_n >= total_n:
            return list(range(total_n))
        if target_n <= 1:
            return [total_n // 2]
        step = float(total_n - 1) / float(target_n - 1)
        idxs = []
        for i in range(target_n):
            idx = int(round(float(i) * step))
            if idxs and idx <= idxs[-1]:
                idx = idxs[-1] + 1
            remaining = (target_n - 1) - i
            max_allowed = (total_n - 1) - remaining
            if idx > max_allowed:
                idx = max_allowed
            idxs.append(idx)
        return idxs

    main_rack_yaw = float(best_plan["yaw_deg"])
    along_axis = str(best_plan.get("along_axis", "x")).strip().lower()

    yaw_pref_candidates = []
    for cand_yaw in (0.0, 90.0):
        cand_ex, cand_ey = _oriented_xy_cached(rack_model, rack_scale, cand_yaw)
        cand_along = float(cand_ex if along_axis == "x" else cand_ey)
        cand_cross = float(cand_ey if along_axis == "x" else cand_ex)
        yaw_pref_candidates.append(
            (
                float(cand_along - cand_cross),
                float(cand_along),
                -float(cand_cross),
                -abs(float(cand_yaw) - float(main_rack_yaw)),
                float(cand_yaw),
            )
        )
    if yaw_pref_candidates:
        yaw_pref_candidates.sort(reverse=True)
        main_rack_yaw = float(yaw_pref_candidates[0][4])

    rack_yaw_offset = float(STORAGE_RACK_GLOBAL_YAW_OFFSET_DEG) % 360.0
    main_rack_yaw = (main_rack_yaw + rack_yaw_offset) % 360.0
    main_rack_ex, main_rack_ey = _oriented_xy_cached(rack_model, rack_scale, main_rack_yaw)
    along_size = float(main_rack_ex if along_axis == "x" else main_rack_ey)
    cross_size = float(main_rack_ey if along_axis == "x" else main_rack_ex)
    if along_axis == "x":
        along_lo = max(x_min + edge_margin + (along_size * 0.5), -floor_half_x + (along_size * 0.5))
        along_hi = min(x_max - edge_margin - (along_size * 0.5), floor_half_x - (along_size * 0.5))
        cross_lo = max(y_min + edge_margin + (cross_size * 0.5), -floor_half_y + (cross_size * 0.5))
        cross_hi = min(y_max - edge_margin - (cross_size * 0.5), floor_half_y - (cross_size * 0.5))
    else:
        along_lo = max(y_min + edge_margin + (along_size * 0.5), -floor_half_y + (along_size * 0.5))
        along_hi = min(y_max - edge_margin - (along_size * 0.5), floor_half_y - (along_size * 0.5))
        cross_lo = max(x_min + edge_margin + (cross_size * 0.5), -floor_half_x + (cross_size * 0.5))
        cross_hi = min(x_max - edge_margin - (cross_size * 0.5), floor_half_x - (cross_size * 0.5))

    target_rows = max(1, int(STORAGE_RACK_TARGET_ROW_COUNT))
    center_aisle_target = max(0.0, float(STORAGE_RACK_CENTER_AISLE_TARGET_M))
    max_racks_cfg = int(STORAGE_RACK_MAX_COUNT)

    endcap_enabled = bool(STORAGE_RACK_ENABLE_ENDCAP_ROWS)
    endcap_rack_yaw = (main_rack_yaw + 90.0) % 360.0
    endcap_rack_ex, endcap_rack_ey = _oriented_xy_cached(rack_model, rack_scale, endcap_rack_yaw)
    endcap_along_size = float(endcap_rack_ex if along_axis == "x" else endcap_rack_ey)
    endcap_cross_size = float(endcap_rack_ey if along_axis == "x" else endcap_rack_ex)

    along_min_bound = along_lo - (along_size * 0.5)
    along_max_bound = along_hi + (along_size * 0.5)
    cross_min_bound = cross_lo - (cross_size * 0.5)
    cross_max_bound = cross_hi + (cross_size * 0.5)

    center_along_bound = 0.5 * (along_min_bound + along_max_bound)
    half_aisle = center_aisle_target * 0.5
    left_bound_lo = along_min_bound
    left_bound_hi = center_along_bound - half_aisle
    right_bound_lo = center_along_bound + half_aisle
    right_bound_hi = along_max_bound

    endcap_main_gap = 0.0
    min_side_span = along_size
    if endcap_enabled:
        min_side_span = endcap_along_size + endcap_main_gap + along_size
    side_span_left = max(0.0, left_bound_hi - left_bound_lo)
    side_span_right = max(0.0, right_bound_hi - right_bound_lo)
    if side_span_left < min_side_span or side_span_right < min_side_span:
        total_span = max(0.0, along_max_bound - along_min_bound)
        max_half_aisle = max(0.0, (total_span - (2.0 * min_side_span)) * 0.5)
        half_aisle = min(half_aisle, max_half_aisle)
        center_along_bound = 0.5 * (along_min_bound + along_max_bound)
        left_bound_lo = along_min_bound
        left_bound_hi = center_along_bound - half_aisle
        right_bound_lo = center_along_bound + half_aisle
        right_bound_hi = along_max_bound

    left_h_main_lo = left_bound_lo
    left_h_main_hi = left_bound_hi
    right_h_main_lo = right_bound_lo
    right_h_main_hi = right_bound_hi
    left_endcap_center = None
    right_endcap_center = None
    if endcap_enabled:
        if (left_bound_hi - left_bound_lo) >= (endcap_along_size - 1e-6):
            left_endcap_center = left_bound_lo + (endcap_along_size * 0.5)
            left_h_main_lo = left_endcap_center + (endcap_along_size * 0.5) + endcap_main_gap
        if (right_bound_hi - right_bound_lo) >= (endcap_along_size - 1e-6):
            right_endcap_center = right_bound_hi - (endcap_along_size * 0.5)
            right_h_main_hi = right_endcap_center - (endcap_along_size * 0.5) - endcap_main_gap

    left_main_lo = left_h_main_lo
    left_main_hi = left_h_main_hi
    right_main_lo = right_h_main_lo
    right_main_hi = right_h_main_hi
    left_centers = _packed_centers(left_main_lo, left_main_hi, along_size, slot_gap) if left_main_hi >= left_main_lo else []
    right_centers = (
        _packed_centers(right_main_lo, right_main_hi, along_size, slot_gap) if right_main_hi >= right_main_lo else []
    )
    if left_centers:
        left_first_target = left_h_main_lo + (along_size * 0.5)
        left_shift = float(left_first_target) - float(left_centers[0])
        left_centers = [float(v) + left_shift for v in left_centers]
    if right_centers:
        right_last_target = right_h_main_hi - (along_size * 0.5)
        right_shift = float(right_last_target) - float(right_centers[-1])
        right_centers = [float(v) + right_shift for v in right_centers]

    if not left_centers or not right_centers:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "Unable to create left/right rack banks with center aisle.",
        }

    row_cross_size = float(cross_size)
    cross_rows_lo = cross_min_bound + (row_cross_size * 0.5)
    cross_rows_hi = cross_max_bound - (row_cross_size * 0.5)
    if cross_rows_hi < cross_rows_lo:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "No valid storage rows fit cross-axis span.",
        }

    cross_span_centers = max(0.0, float(cross_rows_hi) - float(cross_rows_lo))
    max_rows_fit = max(1, int(math.floor(cross_span_centers / max(1e-6, row_cross_size))) + 1)
    rows_use = max(1, min(int(target_rows), int(max_rows_fit)))
    if rows_use <= 1:
        cross_centers = [0.5 * (cross_rows_lo + cross_rows_hi)]
    else:
        step = cross_span_centers / float(rows_use - 1)
        cross_centers = [float(cross_rows_lo) + (i * step) for i in range(rows_use)]

    def _to_world_xy(along_v, cross_v):
        if along_axis == "x":
            return float(along_v), float(cross_v)
        return float(cross_v), float(along_v)

    selected = []
    for row_idx, cross_v in enumerate(cross_centers):
        for bank_idx in (0, 1):
            row_slots = []
            if bank_idx == 0:
                for along_v in left_centers:
                    row_slots.append(
                        {
                            "along": float(along_v),
                            "yaw_deg": float(main_rack_yaw),
                            "kind": "main",
                        }
                    )
            else:
                for along_v in right_centers:
                    row_slots.append(
                        {
                            "along": float(along_v),
                            "yaw_deg": float(main_rack_yaw),
                            "kind": "main",
                        }
                    )

            row_slots = sorted(row_slots, key=lambda s: float(s["along"]))
            for col_idx, slot in enumerate(row_slots):
                sx, sy = _to_world_xy(float(slot["along"]), float(cross_v))
                selected.append(
                    {
                        "x": sx,
                        "y": sy,
                        "row": int(row_idx),
                        "col": int(col_idx),
                        "bank": int(bank_idx),
                        "along": float(slot["along"]),
                        "yaw_deg": float(slot["yaw_deg"]),
                        "kind": str(slot["kind"]),
                    }
                )

    if endcap_enabled and (left_endcap_center is not None) and (right_endcap_center is not None):
        endcap_cross_lo = cross_min_bound + (endcap_cross_size * 0.5)
        endcap_cross_hi = cross_max_bound - (endcap_cross_size * 0.5)
        endcap_slot_gap = 0.0
        endcap_cross_centers = (
            _packed_centers(endcap_cross_lo, endcap_cross_hi, endcap_cross_size, endcap_slot_gap)
            if endcap_cross_hi >= endcap_cross_lo
            else []
        )
        if len(endcap_cross_centers) > 1:
            ec_start = float(endcap_cross_lo)
            ec_end = float(endcap_cross_hi)
            ec_step = (ec_end - ec_start) / float(len(endcap_cross_centers) - 1)
            endcap_cross_centers = [ec_start + (i * ec_step) for i in range(len(endcap_cross_centers))]
        for ec_idx, cross_v in enumerate(endcap_cross_centers):
            for bank_idx, along_v in ((0, left_endcap_center), (1, right_endcap_center)):
                sx, sy = _to_world_xy(float(along_v), float(cross_v))
                selected.append(
                    {
                        "x": sx,
                        "y": sy,
                        "row": int(target_rows + ec_idx),
                        "col": 0,
                        "bank": int(bank_idx),
                        "along": float(along_v),
                        "yaw_deg": float(endcap_rack_yaw),
                        "kind": "endcap",
                    }
                )

    group_rotate_deg = float(STORAGE_RACK_GROUP_ROTATE_DEG) % 360.0
    if selected and abs(group_rotate_deg) > 1e-6:
        rot_rad = math.radians(group_rotate_deg)
        cos_r = math.cos(rot_rad)
        sin_r = math.sin(rot_rad)
        cx = float(area_cx)
        cy = float(area_cy)
        for slot in selected:
            dx = float(slot["x"]) - cx
            dy = float(slot["y"]) - cy
            slot["x"] = cx + (dx * cos_r) - (dy * sin_r)
            slot["y"] = cy + (dx * sin_r) + (dy * cos_r)
            slot["yaw_deg"] = (float(slot.get("yaw_deg", 0.0)) + group_rotate_deg) % 360.0

    if max_racks_cfg > 0 and len(selected) > max_racks_cfg:
        selected = selected[: max_racks_cfg]

    storage_endcap_slot_count = sum(1 for s in selected if str(s.get("kind", "")).lower() == "endcap")

    selected_rows = {}
    for slot in selected:
        rk = (int(slot.get("row", 0)), int(slot.get("bank", 0)))
        selected_rows.setdefault(rk, []).append(slot)
    for rk in list(selected_rows.keys()):
        selected_rows[rk] = sorted(selected_rows[rk], key=lambda s: float(s.get("along", 0.0)))

    barrel_slot_keys = set()
    barrel_prob = max(0.0, min(0.95, float(STORAGE_RACK_BARREL_RACK_PROBABILITY)))
    barrel_phase = rng.randint(0, 2)
    for row_key in sorted(selected_rows.keys(), key=lambda rk: (int(rk[1]), int(rk[0]))):
        row_slots = selected_rows.get(row_key, [])
        if not row_slots:
            continue
        row_idx, bank_idx = int(row_key[0]), int(row_key[1])
        row_pref_barrel = ((row_idx + bank_idx + barrel_phase) % 3) == 0
        row_prob = barrel_prob * (1.45 if row_pref_barrel else 0.35)
        row_prob = max(0.0, min(0.90, row_prob))
        row_has_barrel = False
        for slot in row_slots:
            key = (int(slot.get("row", 0)), int(slot.get("bank", 0)), int(slot.get("col", 0)))
            if rng.random() < row_prob:
                barrel_slot_keys.add(key)
                row_has_barrel = True
        if row_pref_barrel and not row_has_barrel and row_slots and rng.random() < 0.78:
            mid_slot = row_slots[len(row_slots) // 2]
            mid_key = (
                int(mid_slot.get("row", 0)),
                int(mid_slot.get("bank", 0)),
                int(mid_slot.get("col", 0)),
            )
            barrel_slot_keys.add(mid_key)

    if not barrel_slot_keys and selected:
        mid_slot = selected[len(selected) // 2]
        barrel_slot_keys.add(
            (
                int(mid_slot.get("row", 0)),
                int(mid_slot.get("bank", 0)),
                int(mid_slot.get("col", 0)),
            )
        )

    rack_yaw = float(main_rack_yaw)

    def _cluster_level_area(area_map, merge_eps=0.03):
        points = sorted((float(z), float(a)) for z, a in area_map.items() if float(a) > 1e-8)
        if not points:
            return []
        clusters = []
        for z, area in points:
            if not clusters or abs(z - clusters[-1]["z_avg"]) > float(merge_eps):
                clusters.append({"z_avg": z, "area": area, "z_area_sum": z * area})
            else:
                cl = clusters[-1]
                cl["area"] += area
                cl["z_area_sum"] += z * area
                cl["z_avg"] = cl["z_area_sum"] / max(1e-8, cl["area"])
        return [{"z": float(cl["z_avg"]), "area": float(cl["area"])} for cl in clusters]

    def _rack_support_surface_levels_m():
        model_path = os.path.join(str(storage_loader.obj_dir), str(rack_model))
        if not os.path.exists(model_path):
            return []
        try:
            st = os.stat(model_path)
            model_sig = (
                int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
                int(st.st_size),
            )
        except OSError:
            model_sig = (-1, -1)
        cache_key = (
            os.path.abspath(model_path).replace("\\", "/"),
            tuple(round(float(v), 8) for v in rack_scale),
            model_sig,
        )
        cached = _STORAGE_RACK_SUPPORT_LEVELS_CACHE.get(cache_key)
        if cached is not None:
            return [float(v) for v in cached]

        raw_verts = []
        face_tokens = []
        try:
            with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("v "):
                        _, xs, ys, zs = line.split()[:4]
                        raw_verts.append((float(xs), float(ys), float(zs)))
                    elif line.startswith("f "):
                        toks = line.strip().split()[1:]
                        if len(toks) >= 3:
                            face_tokens.append(toks)
        except OSError:
            return []

        if not raw_verts or not face_tokens:
            return []

        sx, sy, sz = float(rack_scale[0]), float(rack_scale[1]), float(rack_scale[2])
        verts = [(x * sx, (-z) * sy, y * sz) for x, y, z in raw_verts]
        min_model_z = min(v[2] for v in verts)

        top_area = {}
        bottom_area = {}
        vcount = len(verts)
        for fpoly in face_tokens:
            idx = []
            for tok in fpoly:
                vtxt = tok.split("/")[0]
                if not vtxt:
                    continue
                vi = int(vtxt)
                if vi < 0:
                    vi = vcount + 1 + vi
                vi = vi - 1
                if vi < 0 or vi >= vcount:
                    idx = []
                    break
                idx.append(vi)
            if len(idx) < 3:
                continue

            v0 = verts[idx[0]]
            for k in range(1, len(idx) - 1):
                v1 = verts[idx[k]]
                v2 = verts[idx[k + 1]]
                ax = float(v1[0] - v0[0])
                ay = float(v1[1] - v0[1])
                az = float(v1[2] - v0[2])
                bx = float(v2[0] - v0[0])
                by = float(v2[1] - v0[1])
                bz = float(v2[2] - v0[2])
                nx = (ay * bz) - (az * by)
                ny = (az * bx) - (ax * bz)
                nz = (ax * by) - (ay * bx)
                nlen = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
                if nlen <= 1e-9:
                    continue
                nz_u = nz / nlen
                if abs(nz_u) < 0.92:
                    continue
                tri_area = 0.5 * nlen
                z_centroid = (float(v0[2]) + float(v1[2]) + float(v2[2])) / 3.0
                z_key = round(z_centroid, 3)
                if nz_u > 0.0:
                    top_area[z_key] = float(top_area.get(z_key, 0.0)) + tri_area
                else:
                    bottom_area[z_key] = float(bottom_area.get(z_key, 0.0)) + tri_area

        if not top_area or not bottom_area:
            return []

        top_levels = _cluster_level_area(top_area, merge_eps=0.025)
        bottom_levels = _cluster_level_area(bottom_area, merge_eps=0.025)
        if not top_levels or not bottom_levels:
            return []

        max_top_area = max(float(v["area"]) for v in top_levels)
        max_bottom_area = max(float(v["area"]) for v in bottom_levels)
        top_levels = [v for v in top_levels if float(v["area"]) >= max(0.03, max_top_area * 0.12)]
        bottom_levels = [v for v in bottom_levels if float(v["area"]) >= max(0.03, max_bottom_area * 0.12)]
        if not top_levels or not bottom_levels:
            return []

        support_rel_levels = []
        for top in sorted(top_levels, key=lambda d: float(d["z"])):
            top_z = float(top["z"])
            best_delta = None
            for bottom in bottom_levels:
                bot_z = float(bottom["z"])
                if bot_z >= top_z:
                    continue
                dz = top_z - bot_z
                if dz < 0.04 or dz > 0.35:
                    continue
                if best_delta is None or dz < best_delta:
                    best_delta = dz
            if best_delta is None:
                continue
            rel = float(top_z - min_model_z)
            if rel <= 0.06:
                continue
            if rel >= (rack_size_z + 0.05):
                continue
            support_rel_levels.append(rel)

        if not support_rel_levels:
            return []
        support_rel_levels = sorted(float(round(v, 4)) for v in support_rel_levels)
        dedup = []
        for z in support_rel_levels:
            if not dedup or (z - dedup[-1]) > 0.18:
                dedup.append(z)
            elif z > dedup[-1]:
                dedup[-1] = z
        _STORAGE_RACK_SUPPORT_LEVELS_CACHE[cache_key] = tuple(float(v) for v in dedup)
        return [float(v) for v in dedup]

    level_count = max(1, int(STORAGE_RACK_PALLET_LEVELS))
    level_ratios = list(STORAGE_RACK_LEVELS_RATIO[:level_count])
    if len(level_ratios) < level_count:
        level_ratios.extend([0.52] * (level_count - len(level_ratios)))

    contact_lift_m = min(0.008, max(0.0, float(STORAGE_RACK_LEVEL_CONTACT_SNAP_M) * 0.2))
    support_levels = _rack_support_surface_levels_m()
    level_zs = [0.0]

    if level_count > 1 and support_levels:
        next_support_idx = 0
        for li in range(1, level_count):
            desired = float(level_ratios[li]) * rack_size_z
            chosen = None
            chosen_idx = None
            for si in range(next_support_idx, len(support_levels)):
                cand = float(support_levels[si]) + contact_lift_m
                if cand <= (level_zs[-1] + 0.02):
                    continue
                if cand >= (desired - 0.06):
                    chosen = cand
                    chosen_idx = si
                    break
            if chosen is None:
                for si in range(next_support_idx, len(support_levels)):
                    cand = float(support_levels[si]) + contact_lift_m
                    if cand > (level_zs[-1] + 0.02):
                        chosen = cand
                        chosen_idx = si
                        break
            if chosen is None:
                break
            level_zs.append(float(chosen))
            next_support_idx = int(chosen_idx) + 1

    fallback_max_level_z = rack_size_z - max(0.02, pallet_size_z * 0.15)
    while len(level_zs) < level_count:
        li = len(level_zs)
        ratio = float(level_ratios[li]) if li < len(level_ratios) else min(0.95, 0.50 + (0.18 * li))
        z_rel = max(0.0, (ratio * rack_size_z) + contact_lift_m)
        if level_zs:
            z_rel = max(z_rel, level_zs[-1] + pallet_size_z + level_min_clear_m)
        z_rel = min(z_rel, fallback_max_level_z)
        if level_zs and z_rel <= level_zs[-1] + 0.03:
            break
        level_zs.append(float(z_rel))

    density_profile = [float(v) for v in STORAGE_RACK_LEVEL_DENSITY]
    if not density_profile:
        density_profile = [1.0]
    level_density_by_idx = []
    for i in range(len(level_zs)):
        if i < len(density_profile):
            d = density_profile[i]
        else:
            d = density_profile[-1]
        level_density_by_idx.append(max(0.25, min(1.0, float(d))))

    def _level_slot_count(slot_total, level_density):
        if slot_total <= 1:
            return 1
        target = int(round(float(slot_total) * float(level_density)))
        target = max(1, min(int(slot_total), target))
        low = target if float(level_density) >= 0.95 else max(1, target - 1)
        high = max(low, target)
        return rng.randint(low, high)

    if pallets_per_level <= 1:
        pallet_local_x_offsets = [0.0]
    else:
        off = rack_size_x * float(STORAGE_RACK_PALLET_INSET_X_RATIO)
        pallet_local_x_offsets = [-off, off]
    pallet_local_x_offsets_active = pallet_local_x_offsets[:pallets_per_level]
    pallet_local_y = rack_size_y * float(STORAGE_RACK_PALLET_INSET_Y_RATIO)

    rack_entries = []
    pallet_count = 0
    box_count = 0
    barrel_count = 0
    rack_no_top_level_count = 0
    top_level_idx = 2 if len(level_zs) >= 3 else -1
    box_entries = []
    for slot in selected:
        rx = float(slot["x"])
        ry = float(slot["y"])
        disable_top_level = False
        if top_level_idx >= 0 and rng.random() < top_level_drop_prob:
            disable_top_level = True
            rack_no_top_level_count += 1
        slot_yaw = float(slot.get("yaw_deg", rack_yaw))
        slot_yaw_rad = math.radians(slot_yaw)
        cos_y = math.cos(slot_yaw_rad)
        sin_y = math.sin(slot_yaw_rad)
        slot_rack_ex, slot_rack_ey = _oriented_xy_cached(rack_model, rack_scale, slot_yaw)
        _spawn_mesh_with_anchor(
            loader=storage_loader,
            model_name=rack_model,
            world_anchor_xyz=(rx, ry, floor_top_z),
            yaw_deg=slot_yaw,
            mesh_scale_xyz=rack_scale,
            local_anchor_xyz=(rack_anchor_x, rack_anchor_y, rack_anchor_z),
            with_collision=True,
            use_texture=False,
            rgba=STORAGE_RACK_RGBA,
            double_sided=False,
        )
        rack_entries.append(
            {
                "x": rx,
                "y": ry,
                "yaw_deg": slot_yaw,
                "footprint_xy_m": (float(slot_rack_ex), float(slot_rack_ey)),
                "size_xyz_m": (rack_size_x, rack_size_y, rack_size_z),
            }
        )

        for level_idx, z_rel in enumerate(level_zs):
            if disable_top_level and level_idx == top_level_idx:
                continue
            pallet_bottom_z = float(floor_top_z) + float(z_rel)
            level_density = float(level_density_by_idx[min(level_idx, len(level_density_by_idx) - 1)])
            cargo_spawn_prob = max(0.0, min(1.0, float(STORAGE_RACK_BOX_PROBABILITY) * level_density))
            barrel_layer2_prob = max(
                0.0,
                min(1.0, float(STORAGE_RACK_BARREL_LAYER2_PROBABILITY) * level_density),
            )
            box_layer2_prob = max(
                0.0,
                min(1.0, float(STORAGE_RACK_BOX_LAYER2_PROBABILITY) * level_density),
            )
            p_ex, p_ey = _oriented_xy_cached(pallet_model, pallet_scale, slot_yaw)
            for local_x in pallet_local_x_offsets_active:
                dx = (local_x * cos_y) - (pallet_local_y * sin_y)
                dy = (local_x * sin_y) + (pallet_local_y * cos_y)
                px = rx + dx
                py = ry + dy

                if (
                    (px - (p_ex * 0.5)) < (-floor_half_x - 1e-6)
                    or (px + (p_ex * 0.5)) > (floor_half_x + 1e-6)
                    or (py - (p_ey * 0.5)) < (-floor_half_y - 1e-6)
                    or (py + (p_ey * 0.5)) > (floor_half_y + 1e-6)
                ):
                    continue

                _spawn_obj_with_mtl_parts(
                    loader=storage_loader,
                    model_name=pallet_model,
                    world_anchor_xyz=(px, py, pallet_bottom_z),
                    yaw_deg=slot_yaw,
                    mesh_scale_xyz=pallet_scale,
                    local_anchor_xyz=(pallet_anchor_x, pallet_anchor_y, pallet_anchor_z),
                    with_collision=True,
                )
                pallet_count += 1

                if rng.random() > cargo_spawn_prob:
                    continue

                slot_key = (int(slot.get("row", 0)), int(slot.get("bank", 0)), int(slot.get("col", 0)))
                use_barrel_cargo = slot_key in barrel_slot_keys

                if use_barrel_cargo:
                    barrel_base_z = pallet_bottom_z + pallet_size_z + 0.01
                    barrel_profile = _barrel_layout_profile_for_slot_yaw(slot_yaw)
                    barrel_yaw = float(barrel_profile["barrel_yaw"])
                    layer1_slots = list(barrel_profile["layer1_slots"])
                    rng.shuffle(layer1_slots)
                    layer1_count = _level_slot_count(len(layer1_slots), level_density)
                    layer1_placed = []
                    bex = float(barrel_profile["bex"])
                    bey = float(barrel_profile["bey"])
                    for bx_local, by_local in layer1_slots[:layer1_count]:
                        bdx = (bx_local * cos_y) - (by_local * sin_y)
                        bdy = (bx_local * sin_y) + (by_local * cos_y)
                        bx = px + bdx
                        by = py + bdy
                        if (
                            (bx - (bex * 0.5)) < (-floor_half_x - 1e-6)
                            or (bx + (bex * 0.5)) > (floor_half_x + 1e-6)
                            or (by - (bey * 0.5)) < (-floor_half_y - 1e-6)
                            or (by + (bey * 0.5)) > (floor_half_y + 1e-6)
                        ):
                            continue

                        overlap_hit = False
                        for prior in layer1_placed:
                            dx = abs(float(bx) - float(prior["x"]))
                            dy = abs(float(by) - float(prior["y"]))
                            lim_x = 0.5 * (float(bex) + float(prior["ex"])) - 0.002
                            lim_y = 0.5 * (float(bey) + float(prior["ey"])) - 0.002
                            if dx < lim_x and dy < lim_y:
                                overlap_hit = True
                                break
                        if overlap_hit:
                            continue

                        _spawn_obj_with_mtl_parts(
                            loader=storage_loader,
                            model_name=barrel_model,
                            world_anchor_xyz=(bx, by, barrel_base_z),
                            yaw_deg=barrel_yaw,
                            mesh_scale_xyz=barrel_scale,
                            local_anchor_xyz=(barrel_anchor_x, barrel_anchor_y, barrel_anchor_z),
                            with_collision=False,
                        )
                        barrel_count += 1
                        box_entries.append(
                            {
                                "x": bx,
                                "y": by,
                                "z": barrel_base_z,
                                "yaw_deg": barrel_yaw,
                                "footprint_xy_m": (bex, bey),
                                "size_z_m": barrel_size_z,
                                "kind": "barrel",
                            }
                        )
                        layer1_placed.append({"x": bx, "y": by, "ex": bex, "ey": bey})

                    top_barrel_bottom = barrel_base_z + barrel_size_z + 0.01
                    max_barrel_bottom = rack_z_limit - barrel_size_z - 0.06
                    if (
                        layer1_placed
                        and top_barrel_bottom <= max_barrel_bottom
                        and rng.random() < barrel_layer2_prob
                    ):
                        top_seed = rng.choice(layer1_placed)
                        bx = float(top_seed["x"])
                        by = float(top_seed["y"])
                        _spawn_obj_with_mtl_parts(
                            loader=storage_loader,
                            model_name=barrel_model,
                            world_anchor_xyz=(bx, by, top_barrel_bottom),
                            yaw_deg=barrel_yaw,
                            mesh_scale_xyz=barrel_scale,
                            local_anchor_xyz=(barrel_anchor_x, barrel_anchor_y, barrel_anchor_z),
                            with_collision=False,
                        )
                        barrel_count += 1
                        box_entries.append(
                            {
                                "x": bx,
                                "y": by,
                                "z": top_barrel_bottom,
                                "yaw_deg": barrel_yaw,
                                "footprint_xy_m": (bex, bey),
                                "size_z_m": barrel_size_z,
                                "kind": "barrel",
                            }
                        )
                    continue

                box_base_z = pallet_bottom_z + pallet_size_z + 0.01
                box_profile = _box_layout_profile_for_slot_yaw(slot_yaw)
                box_yaw = float(box_profile["box_yaw"])
                layer1_slots = list(box_profile["layer1_slots"])
                rng.shuffle(layer1_slots)
                layer1_count = _level_slot_count(len(layer1_slots), level_density)
                layer1_placed = []
                bex = float(box_profile["bex"])
                bey = float(box_profile["bey"])
                for bx_local, by_local in layer1_slots[:layer1_count]:
                    bdx = (bx_local * cos_y) - (by_local * sin_y)
                    bdy = (bx_local * sin_y) + (by_local * cos_y)
                    bx = px + bdx
                    by = py + bdy
                    if (
                        (bx - (bex * 0.5)) < (-floor_half_x - 1e-6)
                        or (bx + (bex * 0.5)) > (floor_half_x + 1e-6)
                        or (by - (bey * 0.5)) < (-floor_half_y - 1e-6)
                        or (by + (bey * 0.5)) > (floor_half_y + 1e-6)
                    ):
                        continue

                    overlap_hit = False
                    for prior in layer1_placed:
                        dx = abs(float(bx) - float(prior["x"]))
                        dy = abs(float(by) - float(prior["y"]))
                        lim_x = 0.5 * (float(bex) + float(prior["ex"])) - 0.002
                        lim_y = 0.5 * (float(bey) + float(prior["ey"])) - 0.002
                        if dx < lim_x and dy < lim_y:
                            overlap_hit = True
                            break
                    if overlap_hit:
                        continue

                    _spawn_obj_with_mtl_parts(
                        loader=storage_loader,
                        model_name=box_model,
                        world_anchor_xyz=(bx, by, box_base_z),
                        yaw_deg=box_yaw,
                        mesh_scale_xyz=box_scale,
                        local_anchor_xyz=(box_anchor_x, box_anchor_y, box_anchor_z),
                        with_collision=False,
                    )
                    box_count += 1
                    box_entries.append(
                        {
                            "x": bx,
                            "y": by,
                            "z": box_base_z,
                            "yaw_deg": box_yaw,
                            "footprint_xy_m": (bex, bey),
                            "size_z_m": box_size_z,
                            "kind": "box",
                        }
                    )
                    layer1_placed.append({"x": bx, "y": by, "ex": bex, "ey": bey})

                top_box_bottom = box_base_z + box_size_z + 0.01
                max_box_bottom = rack_z_limit - box_size_z - 0.06
                if (
                    layer1_placed
                    and top_box_bottom <= max_box_bottom
                    and rng.random() < box_layer2_prob
                ):
                    top_seed = rng.choice(layer1_placed)
                    bx = float(top_seed["x"])
                    by = float(top_seed["y"])
                    _spawn_obj_with_mtl_parts(
                        loader=storage_loader,
                        model_name=box_model,
                        world_anchor_xyz=(bx, by, top_box_bottom),
                        yaw_deg=box_yaw,
                        mesh_scale_xyz=box_scale,
                        local_anchor_xyz=(box_anchor_x, box_anchor_y, box_anchor_z),
                        with_collision=False,
                    )
                    box_count += 1
                    box_entries.append(
                        {
                            "x": bx,
                            "y": by,
                            "z": top_box_bottom,
                            "yaw_deg": box_yaw,
                            "footprint_xy_m": (bex, bey),
                            "size_z_m": box_size_z,
                            "kind": "box",
                        }
                    )

    if not rack_entries:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "Storage rack grid generated but no valid rack spawn points remained.",
        }

    return {
        "storage_rack_enabled": True,
        "storage_rack_model": rack_model,
        "storage_rack_scale_xyz": rack_scale,
        "storage_rack_count": len(rack_entries),
        "storage_rack_pallet_count": int(pallet_count),
        "storage_rack_box_count": int(box_count),
        "storage_rack_barrel_count": int(barrel_count),
        "storage_rack_no_top_level_count": int(rack_no_top_level_count),
        "storage_rack_endcap_count": int(storage_endcap_slot_count),
        "storage_rack_box_entries": box_entries,
        "storage_rack_level_zs_m": [float(v) for v in level_zs],
        "storage_racks": rack_entries,
    }


def _attached_wall_from_area_bounds(area_sx, area_sy, area_cx, area_cy):
    half_x = WAREHOUSE_SIZE_X * 0.5
    half_y = WAREHOUSE_SIZE_Y * 0.5
    dist_to_wall = {
        "north": abs(half_y - (area_cy + area_sy * 0.5)),
        "south": abs((area_cy - area_sy * 0.5) + half_y),
        "east": abs(half_x - (area_cx + area_sx * 0.5)),
        "west": abs((area_cx - area_sx * 0.5) + half_x),
    }
    min_dist = min(dist_to_wall.values())
    near = [w for w, d in dist_to_wall.items() if abs(d - min_dist) <= 1e-6]
    if len(near) == 1:
        return near[0]

    if area_sx > area_sy:
        preferred = [w for w in near if w in ("north", "south")]
    elif area_sy > area_sx:
        preferred = [w for w in near if w in ("east", "west")]
    else:
        preferred = []
    if preferred:
        return preferred[0]
    return near[0]


def _forklift_yaw_back_to_wall(attached_wall):
    if attached_wall == "north":
        return 180.0
    if attached_wall == "south":
        return 0.0
    if attached_wall == "east":
        return 270.0
    if attached_wall == "west":
        return 90.0
    return 0.0


def build_loading_operation_forklifts(forklift_loader, floor_top_z, area_layout, wall_info, seed=0):
    if not ENABLE_LOADING_OPERATION_FORKLIFTS:
        return {"loading_operation_forklift_count": 0, "loading_operation_forklifts": []}
    if forklift_loader is None:
        return {
            "loading_operation_forklift_count": 0,
            "loading_operation_forklifts": [],
            "loading_operation_forklift_reason": "Industrial OBJ loader unavailable.",
        }

    loading_area = (area_layout or {}).get("LOADING")
    if not loading_area:
        return {
            "loading_operation_forklift_count": 0,
            "loading_operation_forklifts": [],
            "loading_operation_forklift_reason": "LOADING area not found in layout.",
        }

    loading_side = str(wall_info.get("loading_side", "north")).lower()
    if loading_side not in WALL_SLOTS:
        loading_side = "north"

    model_name = FORKLIFT_MODEL_NAME
    scale_xyz = (FORKLIFT_SCALE_UNIFORM, FORKLIFT_SCALE_UNIFORM, FORKLIFT_SCALE_UNIFORM)
    min_v, max_v = model_bounds_xyz(forklift_loader, model_name, scale_xyz)
    size_x = max_v[0] - min_v[0]
    size_y = max_v[1] - min_v[1]
    size_z = max_v[2] - min_v[2]
    anchor_x = (min_v[0] + max_v[0]) * 0.5
    anchor_y = (min_v[1] + max_v[1]) * 0.5
    anchor_z = min_v[2]

    yaw_deg = _forklift_yaw_back_to_wall(loading_side)
    yaw = math.radians(yaw_deg)
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    ex = (c * size_x) + (s * size_y)
    ey = (s * size_x) + (c * size_y)

    area_cx = float(loading_area["cx"])
    area_cy = float(loading_area["cy"])
    area_sx = float(loading_area["sx"])
    area_sy = float(loading_area["sy"])
    x_min = area_cx - (area_sx * 0.5)
    x_max = area_cx + (area_sx * 0.5)
    y_min = area_cy - (area_sy * 0.5)
    y_max = area_cy + (area_sy * 0.5)

    if loading_side in ("north", "south"):
        along_axis = "x"
        along_min = x_min
        along_max = x_max
        dock_edge = y_max if loading_side == "north" else y_min
        interior_edge = y_min if loading_side == "north" else y_max
        along_size = ex
        cross_size = ey
    else:
        along_axis = "y"
        along_min = y_min
        along_max = y_max
        dock_edge = x_max if loading_side == "east" else x_min
        interior_edge = x_min if loading_side == "east" else x_max
        along_size = ey
        cross_size = ex

    cross_to_dock_sign = 1.0 if dock_edge >= interior_edge else -1.0
    cross_depth_total = abs(dock_edge - interior_edge)

    def _xy_from_along_s(along, s_from_interior):
        cross = interior_edge + (cross_to_dock_sign * s_from_interior)
        if along_axis == "x":
            return along, cross
        return cross, along

    def _along_s_from_xy(x, y):
        along = float(x) if along_axis == "x" else float(y)
        cross = float(y) if along_axis == "x" else float(x)
        s_from_interior = (cross - interior_edge) * cross_to_dock_sign
        return along, s_from_interior

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    along_margin = (along_size * 0.5) + 0.30
    cross_margin = (cross_size * 0.5) + 0.30
    along_lo = along_min + along_margin
    along_hi = along_max - along_margin
    s_lo = cross_margin
    s_hi = cross_depth_total - cross_margin
    if along_hi <= along_lo or s_hi <= s_lo:
        return {
            "loading_operation_forklift_count": 0,
            "loading_operation_forklifts": [],
            "loading_operation_forklift_reason": "LOADING area too tight for operational forklifts.",
        }

    staging_items = wall_info.get("loading_staging_items", [])
    loaded_pallet_items = [it for it in staging_items if str(it.get("type", "")).lower() == "pallet"]
    empty_items = [it for it in staging_items if str(it.get("type", "")).lower() == "empty_pallet"]
    goods_along_s = []
    for it in loaded_pallet_items:
        if "x" in it and "y" in it:
            goods_along_s.append(_along_s_from_xy(it["x"], it["y"]))
    empty_xy_unique = {}
    for it in empty_items:
        if "x" not in it or "y" not in it:
            continue
        key = (round(float(it["x"]), 3), round(float(it["y"]), 3))
        if key not in empty_xy_unique:
            empty_xy_unique[key] = (float(it["x"]), float(it["y"]))
    empty_along_s = [_along_s_from_xy(x, y) for x, y in empty_xy_unique.values()]

    trucks = wall_info.get("loading_trucks", [])
    truck_alongs = sorted(float(t.get("along", 0.0)) for t in trucks)
    truck_alongs = [a for a in truck_alongs if along_lo <= a <= along_hi]
    if not truck_alongs:
        door_centers = [float(v) for v in wall_info.get("door_centers", [])]
        truck_alongs = [a for a in door_centers if along_lo <= a <= along_hi]
    if not truck_alongs:
        truck_alongs = [0.5 * (along_lo + along_hi)]

    truck_s_vals = []
    for t in trucks:
        if "x" in t and "y" in t:
            _ta, ts = _along_s_from_xy(t["x"], t["y"])
            truck_s_vals.append(ts)

    goods_alongs = [a for a, _s in goods_along_s]
    goods_s_vals = [s for _a, s in goods_along_s]
    empty_alongs = [a for a, _s in empty_along_s]
    empty_s_vals = [s for _a, s in empty_along_s]
    forklift_radius = 0.5 * math.hypot(ex, ey)
    forklift_half_along = along_size * 0.5
    forklift_half_cross = cross_size * 0.5

    hard_obstacle_discs = []
    soft_obstacle_discs = []
    truck_keepout_rects = []
    truck_keepout_pad_along = max(0.30, float(LOADING_OPERATION_TRUCK_KEEPOUT_ALONG_PAD_M))
    truck_keepout_pad_cross = max(0.45, float(LOADING_OPERATION_TRUCK_KEEPOUT_CROSS_PAD_M))
    for t in trucks:
        tx = t.get("x")
        ty = t.get("y")
        if tx is None or ty is None:
            t_along = t.get("along")
            t_inward = t.get("inward")
            if t_along is not None and t_inward is not None:
                try:
                    tx, ty = slot_point(loading_side, float(t_along), inward=float(t_inward))
                except Exception:
                    tx, ty = None, None
        if tx is None or ty is None:
            continue
        tfx, tfy = t.get("footprint_xy_m", (0.0, 0.0))
        tfx = max(0.0, float(tfx))
        tfy = max(0.0, float(tfy))
        t_along_c, t_s_c = _along_s_from_xy(float(tx), float(ty))
        if loading_side in ("north", "south"):
            truck_half_along = max(0.8, 0.5 * tfx)
            truck_half_cross = max(1.0, 0.5 * tfy)
        else:
            truck_half_along = max(0.8, 0.5 * tfy)
            truck_half_cross = max(1.0, 0.5 * tfx)
        truck_keepout_rects.append(
            (
                float(t_along_c),
                float(t_s_c),
                truck_half_along + truck_keepout_pad_along,
                truck_half_cross + truck_keepout_pad_cross,
            )
        )
        tr = max(1.40, (0.5 * min(tfx, tfy)) + 0.45)
        hard_obstacle_discs.append((float(tx), float(ty), tr + 0.25))

    for it in loaded_pallet_items:
        px = it.get("x")
        py = it.get("y")
        if px is None or py is None:
            continue
        soft_obstacle_discs.append((float(px), float(py), 1.45))

    for x, y in empty_xy_unique.values():
        soft_obstacle_discs.append((float(x), float(y), 1.25))

    for ce in wall_info.get("loading_container_entries", []):
        cx = ce.get("x")
        cy = ce.get("y")
        if cx is None or cy is None:
            continue
        hard_obstacle_discs.append((float(cx), float(cy), 4.80))

    for fk in wall_info.get("forklifts", []):
        fx = fk.get("x")
        fy = fk.get("y")
        if fx is None or fy is None:
            continue
        hard_obstacle_discs.append((float(fx), float(fy), forklift_radius + 0.15))

    zone_mid = 0.5 * (along_lo + along_hi)
    truck_s_ref = sum(truck_s_vals) / float(len(truck_s_vals)) if truck_s_vals else (s_hi - 0.8)
    goods_s_ref = sum(goods_s_vals) / float(len(goods_s_vals)) if goods_s_vals else (s_lo + (0.46 * (s_hi - s_lo)))
    if goods_s_ref > truck_s_ref:
        goods_s_ref, truck_s_ref = truck_s_ref, goods_s_ref

    truck_offset = max(1.0, float(LOADING_OPERATION_FORKLIFT_TRUCK_OFFSET_M))
    goods_offset = max(1.0, float(LOADING_OPERATION_FORKLIFT_EMPTY_OFFSET_M))
    truck_oper_s = _clamp(truck_s_ref - truck_offset, s_lo, s_hi)
    goods_oper_s = _clamp(goods_s_ref + goods_offset, s_lo, s_hi)
    if truck_oper_s <= goods_oper_s:
        corridor_s_mid = _clamp(0.5 * (truck_s_ref + goods_s_ref), s_lo, s_hi)
    else:
        corridor_s_mid = _clamp(0.5 * (truck_oper_s + goods_oper_s), s_lo, s_hi)

    target_count = max(1, int(LOADING_OPERATION_FORKLIFT_TARGET_COUNT))
    rng = random.Random(int(seed) + 17233)
    corridor_band_half = max(0.85, (s_hi - s_lo) * 0.18)
    corridor_s_lo = _clamp(corridor_s_mid - corridor_band_half, s_lo, s_hi)
    corridor_s_hi = _clamp(corridor_s_mid + corridor_band_half, s_lo, s_hi)
    if corridor_s_hi <= (corridor_s_lo + 0.15):
        corridor_s_lo = s_lo
        corridor_s_hi = s_hi

    along_seeds = list(truck_alongs)
    if len(along_seeds) >= target_count:
        along_seeds = rng.sample(along_seeds, target_count)
    else:
        while len(along_seeds) < target_count:
            along_seeds.append(rng.uniform(along_lo, along_hi))

    target_points = []
    for idx in range(target_count):
        a_seed = float(along_seeds[idx]) + rng.uniform(-2.2, 2.2)
        s_seed = rng.uniform(corridor_s_lo, corridor_s_hi)
        target_points.append(
            (
                f"corridor_random_{idx}",
                _clamp(a_seed, along_lo, along_hi),
                _clamp(s_seed, s_lo, s_hi),
            )
        )
    rng.shuffle(target_points)

    forklifts = []
    occupied_xy = []
    min_center_dist = max(2.2, forklift_radius * 1.30)
    hard_obstacle_clearance = max(0.22, forklift_radius * 0.22)
    soft_obstacle_clearance = 0.0

    def _try_place(along_base, s_base):
        ds_candidates = [0.0, -0.45, 0.45, -0.90, 0.90, -1.40, 1.40, -2.00, 2.00]
        da_candidates = [0.0, -1.6, 1.6, -3.2, 3.2, -4.8, 4.8, -6.4, 6.4]
        rng.shuffle(ds_candidates)
        rng.shuffle(da_candidates)
        for ds in ds_candidates:
            for da in da_candidates:
                along = _clamp(along_base + da, along_lo, along_hi)
                s_from_interior = _clamp(s_base + ds, s_lo, s_hi)
                x, y = _xy_from_along_s(along, s_from_interior)
                ok = True
                for ox, oy in occupied_xy:
                    if ((x - ox) ** 2 + (y - oy) ** 2) < (min_center_dist ** 2):
                        ok = False
                        break
                if not ok:
                    continue
                for ta, ts, half_a, half_s in truck_keepout_rects:
                    if (
                        abs(along - ta) <= (half_a + forklift_half_along)
                        and abs(s_from_interior - ts) <= (half_s + forklift_half_cross)
                    ):
                        ok = False
                        break
                if not ok:
                    continue
                for ox, oy, orad in hard_obstacle_discs:
                    lim = orad + forklift_radius + hard_obstacle_clearance
                    if ((x - ox) ** 2 + (y - oy) ** 2) < (lim ** 2):
                        ok = False
                        break
                if not ok:
                    continue
                for ox, oy, orad in soft_obstacle_discs:
                    lim = orad + forklift_radius + soft_obstacle_clearance
                    if ((x - ox) ** 2 + (y - oy) ** 2) < (lim ** 2):
                        ok = False
                        break
                if ok:
                    return along, s_from_interior, x, y
        return None

    for idx, (tag, along_raw, s_raw) in enumerate(target_points):
        along_raw += rng.uniform(-0.80, 0.80)
        s_raw += rng.uniform(-0.65, 0.65)
        picked = _try_place(along_raw, s_raw)
        if picked is None:
            picked = _try_place(along_raw, _clamp(corridor_s_mid - 0.85, s_lo, s_hi))
        if picked is None:
            picked = _try_place(along_raw, _clamp(corridor_s_mid + 0.85, s_lo, s_hi))
        if picked is None:
            continue
        along, s_from_interior, x, y = picked
        _spawn_obj_with_mtl_parts(
            loader=forklift_loader,
            model_name=model_name,
            world_anchor_xyz=(x, y, floor_top_z),
            yaw_deg=yaw_deg,
            mesh_scale_xyz=scale_xyz,
            local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
            with_collision=True,
            fallback_rgba=(0.86, 0.86, 0.86, 1.0),
        )
        occupied_xy.append((x, y))
        forklifts.append(
            {
                "model": model_name,
                "x": x,
                "y": y,
                "along": along,
                "s_from_interior": s_from_interior,
                "yaw_deg": yaw_deg,
                "loading_side": loading_side,
                "role": tag,
                "index": idx,
                "footprint_xy_m": (ex, ey),
                "size_xyz_m": (size_x, size_y, size_z),
            }
        )

    target_count = max(1, int(LOADING_OPERATION_FORKLIFT_TARGET_COUNT))
    if len(forklifts) < target_count:
        open_s_lo = _clamp(min(goods_oper_s, corridor_s_mid), s_lo, s_hi)
        open_s_hi = _clamp(max(truck_oper_s, corridor_s_mid), s_lo, s_hi)
        if open_s_hi <= open_s_lo:
            open_s_lo, open_s_hi = s_lo, s_hi
        for _ in range(220):
            if len(forklifts) >= target_count:
                break
            rand_along = rng.uniform(along_lo, along_hi)
            rand_s = rng.uniform(open_s_lo, open_s_hi)
            picked = _try_place(rand_along, rand_s)
            if picked is None:
                continue
            along, s_from_interior, x, y = picked
            _spawn_obj_with_mtl_parts(
                loader=forklift_loader,
                model_name=model_name,
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=scale_xyz,
                local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                with_collision=True,
                fallback_rgba=(0.86, 0.86, 0.86, 1.0),
            )
            occupied_xy.append((x, y))
            forklifts.append(
                {
                    "model": model_name,
                    "x": x,
                    "y": y,
                    "along": along,
                    "s_from_interior": s_from_interior,
                    "yaw_deg": yaw_deg,
                    "loading_side": loading_side,
                    "role": "fill_open_space",
                    "index": len(forklifts),
                    "footprint_xy_m": (ex, ey),
                    "size_xyz_m": (size_x, size_y, size_z),
                }
            )

    return {
        "loading_operation_forklift_count": len(forklifts),
        "loading_operation_forklifts": forklifts,
    }


def build_worker_crew(worker_loader, worker_model_name, floor_top_z, area_layout, wall_info, seed=0):
    if not ENABLE_WORKER_CREW:
        return {"worker_count": 0, "workers": []}
    if worker_loader is None or not worker_model_name:
        return {
            "worker_count": 0,
            "workers": [],
            "worker_reason": "Worker model not found in configured asset paths.",
        }

    try:
        raw_min_v, raw_max_v = model_bounds_xyz(worker_loader, worker_model_name, (1.0, 1.0, 1.0))
    except (FileNotFoundError, ValueError) as exc:
        return {
            "worker_count": 0,
            "workers": [],
            "worker_reason": f"Failed to load worker model: {exc}",
        }

    raw_height = max(1e-6, float(raw_max_v[2] - raw_min_v[2]))
    scale_uniform = max(0.25, min(3.0, float(WORKER_TARGET_HEIGHT_M) / raw_height))
    scale_xyz = (scale_uniform, scale_uniform, scale_uniform)
    min_v, max_v = model_bounds_xyz(worker_loader, worker_model_name, scale_xyz)
    size_x = float(max_v[0] - min_v[0])
    size_y = float(max_v[1] - min_v[1])
    size_z = float(max_v[2] - min_v[2])
    anchor_x = float((min_v[0] + max_v[0]) * 0.5)
    anchor_y = float((min_v[1] + max_v[1]) * 0.5)
    anchor_z = float(min_v[2])

    spacing_min = max(float(WORKER_MIN_SPACING_M), 0.6 * max(size_x, size_y))
    worker_radius = 0.5 * math.hypot(size_x, size_y)

    obstacle_discs = []

    def _add_obstacle_xy(x, y, radius):
        if x is None or y is None:
            return
        obstacle_discs.append((float(x), float(y), float(max(0.2, radius))))

    for t in wall_info.get("loading_trucks", []):
        fx, fy = t.get("footprint_xy_m", (0.0, 0.0))
        _add_obstacle_xy(t.get("x"), t.get("y"), 0.5 * math.hypot(float(fx), float(fy)) + 0.2)
    for fk in wall_info.get("forklifts", []):
        ffx, ffy = fk.get("footprint_xy_m", (size_x, size_y))
        _add_obstacle_xy(fk.get("x"), fk.get("y"), 0.5 * math.hypot(float(ffx), float(ffy)) + 0.2)
    for fk in wall_info.get("loading_operation_forklifts", []):
        ffx, ffy = fk.get("footprint_xy_m", (size_x, size_y))
        _add_obstacle_xy(fk.get("x"), fk.get("y"), 0.5 * math.hypot(float(ffx), float(ffy)) + 0.2)
    for sr in wall_info.get("storage_racks", []):
        rfx, rfy = sr.get("footprint_xy_m", (0.0, 0.0))
        _add_obstacle_xy(sr.get("x"), sr.get("y"), 0.5 * math.hypot(float(rfx), float(rfy)) + 0.25)
    for it in wall_info.get("loading_staging_items", []):
        itype = str(it.get("type", "")).lower()
        if itype in ("pallet", "empty_pallet", "barrel", "box"):
            _add_obstacle_xy(it.get("x"), it.get("y"), 0.85 if "pallet" in itype else 0.55)
    for ce in wall_info.get("loading_container_entries", []):
        _add_obstacle_xy(ce.get("x"), ce.get("y"), 2.2)

    def _zone_bounds(name):
        area = (area_layout or {}).get(name)
        if not area:
            return None
        cx = float(area["cx"])
        cy = float(area["cy"])
        sx = float(area["sx"])
        sy = float(area["sy"])
        return (cx - (sx * 0.5), cx + (sx * 0.5), cy - (sy * 0.5), cy + (sy * 0.5))

    zones_quota = (
        ("LOADING", 3),
        ("STORAGE", 3),
    )
    rng = random.Random(int(seed) + 19441)
    picked_workers = []
    picked_xy = []
    candidate_pool = []

    for zone_name, zone_quota in zones_quota:
        bounds = _zone_bounds(zone_name)
        if bounds is None:
            continue
        x0, x1, y0, y1 = bounds
        margin = max(0.7, worker_radius + 0.30)
        if (x1 - x0) <= (2.0 * margin) or (y1 - y0) <= (2.0 * margin):
            continue
        base_offsets = (
            (-0.30, -0.20),
            (0.30, -0.18),
            (-0.18, 0.20),
            (0.20, 0.22),
            (0.0, 0.0),
        )
        for ox, oy in base_offsets:
            bx = 0.5 * (x0 + x1) + ox * (x1 - x0)
            by = 0.5 * (y0 + y1) + oy * (y1 - y0)
            candidate_pool.append(
                {
                    "zone": zone_name,
                    "quota_weight": zone_quota,
                    "x": bx + rng.uniform(-0.40, 0.40),
                    "y": by + rng.uniform(-0.40, 0.40),
                }
            )
        grid_cols = 5 if zone_name == "STORAGE" else 4
        grid_rows = 3
        gx0 = x0 + margin
        gx1 = x1 - margin
        gy0 = y0 + margin
        gy1 = y1 - margin
        for gy in range(grid_rows):
            for gx in range(grid_cols):
                tx = (gx + 0.5) / float(grid_cols)
                ty = (gy + 0.5) / float(grid_rows)
                base_x = gx0 + (tx * max(0.0, gx1 - gx0))
                base_y = gy0 + (ty * max(0.0, gy1 - gy0))
                candidate_pool.append(
                    {
                        "zone": zone_name,
                        "quota_weight": zone_quota,
                        "x": base_x + rng.uniform(-0.32, 0.32),
                        "y": base_y + rng.uniform(-0.32, 0.32),
                    }
                )
        for _ in range(18):
            candidate_pool.append(
                {
                    "zone": zone_name,
                    "quota_weight": zone_quota,
                    "x": rng.uniform(x0 + margin, x1 - margin),
                    "y": rng.uniform(y0 + margin, y1 - margin),
                }
            )

    target_count = max(1, int(WORKER_TARGET_COUNT))
    quota_left = {name: int(q) for name, q in zones_quota}
    rng.shuffle(candidate_pool)

    def _valid_xy(x, y, obstacle_pad=0.25, spacing_rule=None):
        spacing = float(spacing_min if spacing_rule is None else spacing_rule)
        for px, py in picked_xy:
            if ((x - px) ** 2 + (y - py) ** 2) < (spacing ** 2):
                return False
        for ox, oy, orad in obstacle_discs:
            lim = orad + worker_radius + float(obstacle_pad)
            if ((x - ox) ** 2 + (y - oy) ** 2) < (lim ** 2):
                return False
        return True

    attempt_settings = (
        (0.25, spacing_min),
        (0.14, max(1.0, spacing_min * 0.88)),
        (0.06, max(0.85, spacing_min * 0.78)),
    )

    for obstacle_pad, spacing_rule in attempt_settings:
        for cand in candidate_pool:
            if len(picked_workers) >= target_count:
                break
            zname = cand["zone"]
            if quota_left.get(zname, 0) <= 0:
                continue
            x = float(cand["x"])
            y = float(cand["y"])
            if not _valid_xy(x, y, obstacle_pad=obstacle_pad, spacing_rule=spacing_rule):
                continue
            yaw_deg = rng.uniform(0.0, 360.0)
            picked_workers.append({"x": x, "y": y, "yaw_deg": yaw_deg, "zone": zname})
            picked_xy.append((x, y))
            quota_left[zname] = max(0, quota_left.get(zname, 0) - 1)
        if len(picked_workers) >= target_count:
            break

    if len(picked_workers) < target_count:
        for obstacle_pad, spacing_rule in attempt_settings:
            for cand in candidate_pool:
                if len(picked_workers) >= target_count:
                    break
                x = float(cand["x"])
                y = float(cand["y"])
                if not _valid_xy(x, y, obstacle_pad=obstacle_pad, spacing_rule=spacing_rule):
                    continue
                yaw_deg = rng.uniform(0.0, 360.0)
                picked_workers.append({"x": x, "y": y, "yaw_deg": yaw_deg, "zone": cand["zone"]})
                picked_xy.append((x, y))
            if len(picked_workers) >= target_count:
                break

    workers = []
    for i, item in enumerate(picked_workers):
        x = float(item["x"])
        y = float(item["y"])
        yaw_deg = float(item["yaw_deg"])
        _spawn_obj_with_mtl_parts(
            loader=worker_loader,
            model_name=worker_model_name,
            world_anchor_xyz=(x, y, floor_top_z),
            yaw_deg=yaw_deg,
            mesh_scale_xyz=scale_xyz,
            local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
            with_collision=False,
            fallback_rgba=(0.65, 0.65, 0.68, 1.0),
            rgba_gain=WORKER_COLOR_GAIN,
        )
        workers.append(
            {
                "index": i,
                "model": worker_model_name,
                "x": x,
                "y": y,
                "yaw_deg": yaw_deg,
                "zone": item.get("zone", ""),
                "size_xyz_m": (size_x, size_y, size_z),
                "scale_uniform": scale_uniform,
            }
        )

    return {
        "worker_count": len(workers),
        "worker_model": worker_model_name,
        "worker_scale_uniform": scale_uniform,
        "workers": workers,
    }


def build_forklift_parking(forklift_loader, floor_top_z, area_layout, seed=0):
    if not ENABLE_FORKLIFT_PARKING:
        return {"forklift_scale": FORKLIFT_SCALE_UNIFORM, "forklifts": []}
    if forklift_loader is None:
        return {"forklift_scale": FORKLIFT_SCALE_UNIFORM, "forklifts": []}

    area_name = None
    area = None
    for name in FORKLIFT_AREA_PREFERENCE:
        if name in (area_layout or {}):
            area_name = name
            area = area_layout[name]
            break
    if area is None:
        return {
            "forklift_scale": FORKLIFT_SCALE_UNIFORM,
            "forklifts": [],
            "forklift_reason": "No suitable area marker found for forklift parking.",
        }

    model_name = FORKLIFT_MODEL_NAME
    scale_xyz = (FORKLIFT_SCALE_UNIFORM, FORKLIFT_SCALE_UNIFORM, FORKLIFT_SCALE_UNIFORM)
    min_v, max_v = model_bounds_xyz(forklift_loader, model_name, scale_xyz)
    size_x = max_v[0] - min_v[0]
    size_y = max_v[1] - min_v[1]
    size_z = max_v[2] - min_v[2]
    anchor_x = (min_v[0] + max_v[0]) * 0.5
    anchor_y = (min_v[1] + max_v[1]) * 0.5
    anchor_z = min_v[2]

    area_sx = float(area["sx"])
    area_sy = float(area["sy"])
    area_cx = float(area["cx"])
    area_cy = float(area["cy"])
    attached_wall = _attached_wall_from_area_bounds(area_sx, area_sy, area_cx, area_cy)
    row_axis = "x" if attached_wall in ("north", "south") else "y"
    yaw_deg = (_forklift_yaw_back_to_wall(attached_wall) + float(FORKLIFT_PARK_YAW_EXTRA_DEG)) % 360.0

    yaw = math.radians(yaw_deg)
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    ex = (c * size_x) + (s * size_y)
    ey = (s * size_x) + (c * size_y)
    along_size = ex if row_axis == "x" else ey
    cross_size = ey if row_axis == "x" else ex

    rng = random.Random(int(seed) + 17003)
    target_slots = max(1, int(FORKLIFT_PARK_SLOT_COUNT))
    margin = 0.7
    area_along = area_sx if row_axis == "x" else area_sy
    area_cross = area_sy if row_axis == "x" else area_sx
    max_along = area_along - (2.0 * margin)
    interior_margin = margin
    max_cross_vehicle = area_cross - (float(FORKLIFT_WALL_BACK_CLEARANCE) + interior_margin)
    if max_cross_vehicle <= (cross_size + 1e-6):
        return {
            "forklift_scale": FORKLIFT_SCALE_UNIFORM,
            "forklifts": [],
            "forklift_reason": f"Area {area_name} too narrow for forklift width.",
        }

    gap = max(0.2, float(FORKLIFT_PARK_GAP_M))
    slot_count = target_slots
    row_span = 0.0
    while slot_count >= 1:
        gap = max(0.2, float(FORKLIFT_PARK_GAP_M))
        row_span = (slot_count * along_size) + ((slot_count - 1) * gap)
        if row_span > max_along and slot_count > 1:
            fit_gap = (max_along - (slot_count * along_size)) / float(slot_count - 1)
            gap = max(0.2, fit_gap)
            row_span = (slot_count * along_size) + ((slot_count - 1) * gap)
        if row_span <= (max_along + 1e-6):
            break
        slot_count -= 1
    if row_span > (max_along + 1e-6):
        return {
            "forklift_scale": FORKLIFT_SCALE_UNIFORM,
            "forklifts": [],
            "forklift_reason": f"Area {area_name} too short for forklift parking slots.",
        }

    if row_axis == "x":
        wall_sign = 1.0 if attached_wall == "north" else -1.0
        wall_edge = area_cy + (wall_sign * area_sy * 0.5)
        center_from_wall = float(FORKLIFT_WALL_BACK_CLEARANCE) + (cross_size * 0.5)
        row_center_cross = wall_edge - (wall_sign * center_from_wall)
        cross_offset = row_center_cross - area_cy
    else:
        wall_sign = 1.0 if attached_wall == "east" else -1.0
        wall_edge = area_cx + (wall_sign * area_sx * 0.5)
        center_from_wall = float(FORKLIFT_WALL_BACK_CLEARANCE) + (cross_size * 0.5)
        row_center_cross = wall_edge - (wall_sign * center_from_wall)
        cross_offset = row_center_cross - area_cx

    spawn_min = max(0, int(FORKLIFT_PARK_SPAWN_MIN))
    spawn_max = max(0, int(FORKLIFT_PARK_SPAWN_MAX))
    spawn_min = min(slot_count, spawn_min)
    spawn_max = min(slot_count, spawn_max)
    if spawn_max < spawn_min:
        spawn_min = spawn_max
    spawn_count = rng.randint(spawn_min, spawn_max) if slot_count > 0 else 0
    occupied_indices = sorted(rng.sample(range(slot_count), spawn_count)) if spawn_count > 0 else []
    occupied_set = set(occupied_indices)

    forklifts = []
    row_start = -0.5 * row_span + 0.5 * along_size
    step = along_size + gap
    slot_centers = []
    for i in range(slot_count):
        along = row_start + i * step
        if row_axis == "x":
            sx = area_cx + along
            sy = area_cy + cross_offset
        else:
            sx = area_cx + cross_offset
            sy = area_cy + along
        slot_centers.append((sx, sy, along))

    if ENABLE_FORKLIFT_PARK_SLOT_LINES and slot_centers:
        line_w = max(0.03, float(FORKLIFT_PARK_LINE_WIDTH_M))
        line_h = max(0.002, float(FORKLIFT_PARK_LINE_HEIGHT_M))
        line_z = floor_top_z + float(FORKLIFT_PARK_LINE_CENTER_Z)
        along_extra_max = max(0.0, max_along - row_span)
        cross_extra_max = max(0.0, area_cross - cross_size)
        slot_along = along_size + min(float(FORKLIFT_PARK_SLOT_ALONG_PAD_M), along_extra_max)
        slot_cross = cross_size + min(float(FORKLIFT_PARK_SLOT_CROSS_PAD_M), cross_extra_max)
        slot_cross = min(slot_cross, max(0.1, area_cross - line_w))
        if slot_count >= 1:
            join_cross = slot_cross + line_w
            if row_axis == "x":
                wall_line_y = wall_edge - (wall_sign * (line_w * 0.5))
                divider_center_y = wall_line_y - (wall_sign * (slot_cross * 0.5))
                x_min = slot_centers[0][0] - (slot_along * 0.5)
                x_max = slot_centers[-1][0] + (slot_along * 0.5)
                row_len = max(0.05, x_max - x_min)
                _spawn_box_primitive(
                    center_xyz=((x_min + x_max) * 0.5, wall_line_y, line_z),
                    size_xyz=(row_len, line_w, line_h),
                    rgba=FORKLIFT_PARK_LINE_RGBA,
                    with_collision=False,
                )
                boundary_x = [x_min] + [
                    0.5 * (slot_centers[i][0] + slot_centers[i + 1][0])
                    for i in range(slot_count - 1)
                ] + [x_max]
                for bx in boundary_x:
                    _spawn_box_primitive(
                        center_xyz=(bx, divider_center_y, line_z),
                        size_xyz=(line_w, join_cross, line_h),
                        rgba=FORKLIFT_PARK_LINE_RGBA,
                        with_collision=False,
                    )
            else:
                wall_line_x = wall_edge - (wall_sign * (line_w * 0.5))
                divider_center_x = wall_line_x - (wall_sign * (slot_cross * 0.5))
                y_min = slot_centers[0][1] - (slot_along * 0.5)
                y_max = slot_centers[-1][1] + (slot_along * 0.5)
                row_len = max(0.05, y_max - y_min)
                _spawn_box_primitive(
                    center_xyz=(wall_line_x, (y_min + y_max) * 0.5, line_z),
                    size_xyz=(line_w, row_len, line_h),
                    rgba=FORKLIFT_PARK_LINE_RGBA,
                    with_collision=False,
                )
                boundary_y = [y_min] + [
                    0.5 * (slot_centers[i][1] + slot_centers[i + 1][1])
                    for i in range(slot_count - 1)
                ] + [y_max]
                for by in boundary_y:
                    _spawn_box_primitive(
                        center_xyz=(divider_center_x, by, line_z),
                        size_xyz=(join_cross, line_w, line_h),
                        rgba=FORKLIFT_PARK_LINE_RGBA,
                        with_collision=False,
                    )

    model_path = forklift_loader._asset_path(model_name)
    material_parts = _obj_material_parts(model_path)
    for i, (x, y, along) in enumerate(slot_centers):
        if i not in occupied_set:
            continue

        _spawn_mesh_with_anchor(
            loader=forklift_loader,
            model_name=model_name,
            world_anchor_xyz=(x, y, floor_top_z),
            yaw_deg=yaw_deg,
            mesh_scale_xyz=scale_xyz,
            local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
            with_collision=True,
            use_texture=False,
            rgba=(1.0, 1.0, 1.0, 0.0),
            double_sided=False,
        )
        if material_parts:
            for part in material_parts:
                _spawn_mesh_with_anchor(
                    loader=forklift_loader,
                    model_name=part["path"],
                    world_anchor_xyz=(x, y, floor_top_z),
                    yaw_deg=yaw_deg,
                    mesh_scale_xyz=scale_xyz,
                    local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                    with_collision=False,
                    use_texture=False,
                    rgba=part["rgba"],
                    double_sided=False,
                )
        else:
            _spawn_mesh_with_anchor(
                loader=forklift_loader,
                model_name=model_name,
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=scale_xyz,
                local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                with_collision=False,
                use_texture=False,
                rgba=(0.85, 0.85, 0.85, 1.0),
                double_sided=False,
            )
        forklifts.append(
            {
                "model": model_name,
                "size_xyz_m": (size_x, size_y, size_z),
                "footprint_xy_m": (ex, ey),
                "yaw_deg": yaw_deg,
                "x": x,
                "y": y,
                "attached_wall": attached_wall,
                "row_axis": row_axis,
                "row_index": i,
            }
        )

    return {
        "forklift_scale": FORKLIFT_SCALE_UNIFORM,
        "forklift_area": area_name,
        "forklift_wall": attached_wall,
        "forklift_slot_count": slot_count,
        "forklift_spawned_count": len(forklifts),
        "forklift_occupied_slots": occupied_indices,
        "forklift_slot_centers": [
            {"row_index": i, "x": x, "y": y}
            for i, (x, y, _along) in enumerate(slot_centers)
        ],
        "forklifts": forklifts,
    }


def build_machining_cell_layout(industry_loader, floor_top_z, area_layout):
    if not ENABLE_MACHINING_CELL_LAYOUT:
        return {"machining_mills": [], "machining_lathes": [], "machining_pending_slots": []}
    if industry_loader is None:
        return {
            "machining_mills": [],
            "machining_lathes": [],
            "machining_pending_slots": [],
            "machining_reason": "Industrial OBJ loader unavailable.",
        }

    area = (area_layout or {}).get(MACHINING_CELL_AREA_NAME)
    if not area:
        return {
            "machining_mills": [],
            "machining_lathes": [],
            "machining_pending_slots": [],
            "machining_reason": f"Area {MACHINING_CELL_AREA_NAME} not present in layout.",
        }

    area_sx = float(area["sx"])
    area_sy = float(area["sy"])
    area_cx = float(area["cx"])
    area_cy = float(area["cy"])
    along_axis = "x" if area_sx >= area_sy else "y"
    along_len = area_sx if along_axis == "x" else area_sy
    cross_len = area_sy if along_axis == "x" else area_sx

    along_margin = min(MACHINING_EDGE_MARGIN, max(0.45, along_len * 0.15))
    along_room = max(0.6, along_len - (2.0 * along_margin))
    col_offsets = (-along_room * 0.5, 0.0, along_room * 0.5)

    slot_types = list(MACHINING_SLOT_TYPES)
    if len(slot_types) < 6:
        slot_types.extend(["MILL"] * (6 - len(slot_types)))
    slot_types = slot_types[:6]

    machine_library = {
        "MILL": {
            "model_name": MACHINING_MILL_MODEL_NAME,
            "scale_uniform": MACHINING_MILL_SCALE_UNIFORM,
            "simple_rgba": MACHINING_SIMPLE_MILL_RGBA,
        },
        "LATHE": {
            "model_name": MACHINING_LATHE_MODEL_NAME,
            "scale_uniform": MACHINING_LATHE_SCALE_UNIFORM,
            "simple_rgba": MACHINING_SIMPLE_LATHE_RGBA,
        },
    }
    active_specs = {}
    missing_machine_models = []
    for slot_type in sorted(set(slot_types)):
        cfg = machine_library.get(slot_type)
        if cfg is None:
            continue
        model_name = cfg["model_name"]
        model_path = industry_loader._asset_path(model_name)
        if not os.path.exists(model_path):
            missing_machine_models.append(f"{slot_type}:{model_path}")
            continue
        if MACHINING_FORCE_REFRESH_MTL_PROXY:
            _purge_generated_model_artifacts(model_path)
        collision_path = _obj_collision_proxy_path(model_path)
        visual_path = _obj_mtl_visual_proxy_path(model_path)
        s = float(cfg["scale_uniform"])
        scale_xyz = (s, s, s)
        min_v, max_v = model_bounds_xyz(industry_loader, model_name, scale_xyz)
        active_specs[slot_type] = {
            "slot_type": slot_type,
            "model_name": model_name,
            "model_path": model_path,
            "collision_path": collision_path,
            "visual_path": visual_path,
            "simple_rgba": cfg.get("simple_rgba", (0.62, 0.64, 0.66, 1.0)),
            "scale_uniform": s,
            "scale_xyz": scale_xyz,
            "size_x": max_v[0] - min_v[0],
            "size_y": max_v[1] - min_v[1],
            "size_z": max_v[2] - min_v[2],
            "anchor_x": (min_v[0] + max_v[0]) * 0.5,
            "anchor_y": (min_v[1] + max_v[1]) * 0.5,
            "anchor_z": min_v[2],
            "material_parts": []
            if (MACHINING_FORCE_SIMPLE_VISUALS or MACHINING_USE_NATIVE_MTL_VISUALS)
            else _obj_material_parts(model_path),
        }

    if not active_specs:
        reason = "No machining machine models found."
        if missing_machine_models:
            reason += " Missing: " + "; ".join(missing_machine_models)
        return {
            "machining_mills": [],
            "machining_lathes": [],
            "machining_pending_slots": [],
            "machining_reason": reason,
        }

    machine_depth = max(spec["size_y"] for spec in active_specs.values())
    cross_half_limit = (cross_len * 0.5) - MACHINING_EDGE_MARGIN - (machine_depth * 0.5)
    target_row_offset = (MACHINING_AISLE_WIDTH * 0.5) + (machine_depth * 0.5) + 0.15
    row_offset = min(cross_half_limit, target_row_offset)
    if row_offset <= 0.25:
        return {
            "machining_mills": [],
            "machining_lathes": [],
            "machining_pending_slots": [],
            "machining_reason": f"Area {MACHINING_CELL_AREA_NAME} too narrow for machining rows.",
        }

    def _slot_xy(along_off, row_sign):
        if along_axis == "x":
            return (area_cx + along_off, area_cy + (row_sign * row_offset))
        return (area_cx + (row_sign * row_offset), area_cy + along_off)

    def _yaw_to_aisle(row_sign):
        if along_axis == "x":
            return 0.0 if row_sign < 0 else 180.0
        return 90.0 if row_sign < 0 else 270.0

    def _spawn_machine_instance(spec, x, y, yaw_deg):
        if MACHINING_FORCE_SIMPLE_VISUALS:
            _spawn_mesh_with_anchor(
                loader=industry_loader,
                model_name=spec.get("collision_path", spec["model_name"]),
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=spec["scale_xyz"],
                local_anchor_xyz=(spec["anchor_x"], spec["anchor_y"], spec["anchor_z"]),
                with_collision=True,
                use_texture=False,
                rgba=spec.get("simple_rgba", (0.62, 0.64, 0.66, 1.0)),
                double_sided=MACHINING_VISUAL_DOUBLE_SIDED,
            )
        elif MACHINING_USE_NATIVE_MTL_VISUALS:
            _spawn_native_mtl_visual_with_anchor(
                loader=industry_loader,
                model_name=spec["model_name"],
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=spec["scale_xyz"],
                local_anchor_xyz=(spec["anchor_x"], spec["anchor_y"], spec["anchor_z"]),
                model_path_override=spec.get("visual_path", spec.get("model_path", "")),
                collision_model_path_override=spec.get("collision_path", ""),
                with_collision=True,
                double_sided=MACHINING_VISUAL_DOUBLE_SIDED,
            )
        else:
            _spawn_collision_only_with_anchor(
                loader=industry_loader,
                model_name=spec["model_name"],
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=spec["scale_xyz"],
                local_anchor_xyz=(spec["anchor_x"], spec["anchor_y"], spec["anchor_z"]),
                model_path_override=spec.get("collision_path", ""),
            )
        if (not MACHINING_FORCE_SIMPLE_VISUALS) and (not MACHINING_USE_NATIVE_MTL_VISUALS) and spec["material_parts"]:
            for part in spec["material_parts"]:
                part_tex = part.get("texture_path", "")
                use_part_tex = bool(
                    MACHINING_USE_PART_TEXTURES and part_tex and os.path.exists(part_tex)
                )
                part_rgba = [1.0, 1.0, 1.0, part["rgba"][3]] if use_part_tex else part["rgba"]
                _spawn_mesh_with_anchor(
                    loader=industry_loader,
                    model_name=part["path"],
                    world_anchor_xyz=(x, y, floor_top_z),
                    yaw_deg=yaw_deg,
                    mesh_scale_xyz=spec["scale_xyz"],
                    local_anchor_xyz=(spec["anchor_x"], spec["anchor_y"], spec["anchor_z"]),
                    with_collision=False,
                    use_texture=use_part_tex,
                    texture_path_override=part_tex,
                    rgba=part_rgba,
                    double_sided=MACHINING_VISUAL_DOUBLE_SIDED,
                )
        elif (not MACHINING_FORCE_SIMPLE_VISUALS) and (not MACHINING_USE_NATIVE_MTL_VISUALS):
            _spawn_mesh_with_anchor(
                loader=industry_loader,
                model_name=spec["model_name"],
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=spec["scale_xyz"],
                local_anchor_xyz=(spec["anchor_x"], spec["anchor_y"], spec["anchor_z"]),
                with_collision=False,
                use_texture=False,
                rgba=(0.62, 0.64, 0.66, 1.0),
                double_sided=MACHINING_VISUAL_DOUBLE_SIDED,
            )

    mills = []
    lathes = []
    pending_slots = []
    slot_index = 0
    for row_sign in (-1.0, 1.0):
        for along_off in col_offsets:
            slot_type = slot_types[slot_index]
            slot_index += 1
            x, y = _slot_xy(along_off, row_sign)
            yaw_deg = _yaw_to_aisle(row_sign)
            if slot_type in ("MILL", "LATHE"):
                yaw_deg = (yaw_deg + MACHINING_HEAVY_EXTRA_YAW_DEG) % 360.0
            spec = active_specs.get(slot_type)
            if spec is not None:
                _spawn_machine_instance(spec, x, y, yaw_deg)
                payload = {
                    "model": spec["model_name"],
                    "slot_type": slot_type,
                    "size_xyz_m": (spec["size_x"], spec["size_y"], spec["size_z"]),
                    "x": x,
                    "y": y,
                    "yaw_deg": yaw_deg,
                }
                if slot_type == "MILL":
                    mills.append(payload)
                elif slot_type == "LATHE":
                    lathes.append(payload)
            else:
                pending_slots.append({"slot_type": slot_type, "x": x, "y": y, "yaw_deg": yaw_deg})
                if MACHINING_SHOW_PENDING_MARKERS:
                    px, py, pz = MACHINING_PENDING_SLOT_SIZE
                    _spawn_box_primitive(
                        center_xyz=(x, y, floor_top_z + (pz * 0.5) + 0.002),
                        size_xyz=(px, py, pz),
                        rgba=MACHINING_PENDING_RGBA,
                        with_collision=False,
                    )
                    if SHOW_AREA_LAYOUT_MARKERS:
                        p.addUserDebugText(
                            text=f"{slot_type}_SLOT",
                            textPosition=[x, y, floor_top_z + pz + 0.05],
                            textColorRGB=[0.10, 0.10, 0.10],
                            textSize=1.0,
                            lifeTime=0.0,
                        )

    table_len, table_wid, table_h = MACHINING_TABLE_SIZE
    table_along = -0.5 * along_room + max(0.9, table_len * 0.5 + 0.2)
    if along_axis == "x":
        tx = area_cx + table_along
        ty = area_cy
    else:
        tx = area_cx
        ty = area_cy + table_along
    _spawn_box_primitive(
        center_xyz=(tx, ty, floor_top_z + (table_h * 0.5)),
        size_xyz=(table_len, table_wid, table_h),
        rgba=MACHINING_TABLE_RGBA,
        with_collision=True,
    )

    return {
        "machining_area": MACHINING_CELL_AREA_NAME,
        "machining_scale": MACHINING_MILL_SCALE_UNIFORM,
        "machining_scales": {
            "MILL": MACHINING_MILL_SCALE_UNIFORM,
            "LATHE": MACHINING_LATHE_SCALE_UNIFORM,
        },
        "machining_mills": mills,
        "machining_lathes": lathes,
        "machining_pending_slots": pending_slots,
        "machining_table": {
            "size_xyz_m": MACHINING_TABLE_SIZE,
            "x": tx,
            "y": ty,
        },
        "machining_missing_models": missing_machine_models,
    }


def _create_runtime_context():
    kit_paths = _resolve_kit_paths()
    runtime_context = {
        "kit_paths": kit_paths,
        "shell_meshes": _resolve_shell_mesh_paths(),
        "worker_model_name": "",
        "crane_model_name": "",
    }

    runtime_context["conveyor_loader"] = MeshKitLoader(
        obj_dir=kit_paths["conveyor_obj"],
        texture_path=kit_paths["conveyor_tex"],
    )

    runtime_context["truck_loader"] = (
        MeshKitLoader(
            obj_dir=kit_paths["truck_obj"],
            texture_path=kit_paths["conveyor_tex"],
        )
        if ENABLE_LOADING_TRUCKS
        else None
    )

    runtime_context["industry_loader"] = (
        MeshKitLoader(
            obj_dir=kit_paths["forklift_obj"],
            texture_path=kit_paths["forklift_tex"],
        )
        if (ENABLE_FORKLIFT_PARKING or ENABLE_MACHINING_CELL_LAYOUT or ENABLE_LOADING_OPERATION_FORKLIFTS)
        else None
    )

    runtime_context["loading_staging_loader"] = (
        MeshKitLoader(
            obj_dir=kit_paths["loading_staging_obj"],
            texture_path=kit_paths["conveyor_tex"],
        )
        if ((ENABLE_LOADING_STAGING or ENABLE_STORAGE_RACK_LAYOUT) and kit_paths.get("loading_staging_obj"))
        else None
    )

    worker_loader = None
    if ENABLE_WORKER_CREW:
        worker_obj_dir, worker_model_name = _resolve_optional_model(
            WORKER_ASSET_CANDIDATES,
            WORKER_MODEL_CANDIDATES,
        )
        if worker_obj_dir and worker_model_name:
            worker_loader = MeshKitLoader(
                obj_dir=worker_obj_dir,
                texture_path="",
            )
            runtime_context["worker_model_name"] = worker_model_name
    runtime_context["worker_loader"] = worker_loader

    crane_loader = None
    if ENABLE_OVERHEAD_CRANES:
        crane_obj_dir, crane_model_name = _resolve_optional_model(
            OVERHEAD_CRANE_ASSET_CANDIDATES,
            OVERHEAD_CRANE_MODEL_CANDIDATES,
        )
        if crane_obj_dir and crane_model_name:
            crane_loader = MeshKitLoader(
                obj_dir=crane_obj_dir,
                texture_path="",
            )
            runtime_context["crane_model_name"] = crane_model_name
    runtime_context["crane_loader"] = crane_loader

    return runtime_context


def _reset_runtime_context_for_build(runtime_context):
    for key in (
        "conveyor_loader",
        "truck_loader",
        "industry_loader",
        "loading_staging_loader",
        "worker_loader",
        "crane_loader",
    ):
        _clear_loader_spawn_caches(runtime_context.get(key))


def build_map(seed, runtime_context=None):
    if runtime_context is None:
        runtime_context = _create_runtime_context()

    p.resetSimulation()
    _MESH_VISUAL_SHAPE_CACHE.clear()
    _MESH_COLLISION_SHAPE_CACHE.clear()
    _RESOLVED_MESH_PATH_CACHE.clear()
    _reset_runtime_context_for_build(runtime_context)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    if hasattr(p, "COV_ENABLE_RGB_BUFFER_PREVIEW"):
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    if hasattr(p, "COV_ENABLE_DEPTH_BUFFER_PREVIEW"):
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    if hasattr(p, "COV_ENABLE_SEGMENTATION_MARK_PREVIEW"):
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

    conveyor_loader = runtime_context["conveyor_loader"]
    truck_loader = runtime_context.get("truck_loader")
    industry_loader = runtime_context.get("industry_loader")
    loading_staging_loader = runtime_context.get("loading_staging_loader")
    worker_loader = runtime_context.get("worker_loader")
    crane_loader = runtime_context.get("crane_loader")
    worker_model_name = runtime_context.get("worker_model_name", "")
    crane_model_name = runtime_context.get("crane_model_name", "")
    shell_meshes = runtime_context["shell_meshes"]

    build_stage_timings = {}

    def _stage(name, fn):
        t0 = time.perf_counter()
        out = fn()
        build_stage_timings[name] = time.perf_counter() - t0
        return out

    floor_top_z = _stage("floor", lambda: build_floor(conveyor_loader))
    wall_info = _stage("walls", lambda: build_walls(conveyor_loader, floor_top_z, seed))
    wall_info.update(
        _stage(
            "personnel_lane",
            lambda: build_personnel_floor_lane(conveyor_loader, floor_top_z, wall_info),
        )
    )
    truck_info = _stage(
        "loading_trucks",
        lambda: build_loading_trucks(truck_loader, floor_top_z, wall_info) if truck_loader is not None else {},
    )
    wall_info.update(truck_info)
    area_layout = _stage(
        "area_layout",
        lambda: build_area_layout_markers(conveyor_loader, floor_top_z, wall_info, seed=seed),
    )
    wall_info["area_layout"] = area_layout
    wall_info.update(
        _stage(
            "loading_staging",
            lambda: build_loading_staging(
                loading_staging_loader,
                floor_top_z,
                area_layout,
                wall_info,
                seed=seed,
            ),
        )
    )
    wall_info.update(
        _stage(
            "storage_racks",
            lambda: build_storage_racks(
                loading_staging_loader,
                floor_top_z,
                area_layout,
                wall_info,
                seed=seed,
            ),
        )
    )
    loading_operation_forklift_info = _stage(
        "loading_operation_forklifts",
        lambda: (
            build_loading_operation_forklifts(industry_loader, floor_top_z, area_layout, wall_info, seed=seed)
            if industry_loader is not None
            else {}
        ),
    )
    wall_info.update(loading_operation_forklift_info)
    wall_info.update(
        _stage(
            "office",
            lambda: build_embedded_office_map(floor_top_z, area_layout, seed=seed, wall_info=wall_info),
        )
    )
    wall_info.update(
        _stage(
            "factory",
            lambda: build_embedded_factory_map(conveyor_loader, floor_top_z, area_layout, seed=seed),
        )
    )
    forklift_info = _stage(
        "forklift_parking",
        lambda: (
            build_forklift_parking(industry_loader, floor_top_z, area_layout, seed=seed)
            if industry_loader is not None
            else {}
        ),
    )
    wall_info.update(forklift_info)
    machining_info = _stage(
        "machining",
        lambda: (
            build_machining_cell_layout(industry_loader, floor_top_z, area_layout)
            if industry_loader is not None
            else {"machining_mills": [], "machining_lathes": [], "machining_pending_slots": []}
        ),
    )
    wall_info.update(machining_info)
    wall_info.update(
        _stage(
            "workers",
            lambda: build_worker_crew(
                worker_loader,
                worker_model_name,
                floor_top_z,
                area_layout,
                wall_info,
                seed=seed,
            ),
        )
    )
    roof_base_z = wall_info["roof_eave_z"]
    if ENABLE_CORNER_COLUMNS:
        _stage("columns", lambda: build_columns(conveyor_loader, floor_top_z))
    _stage(
        "roof_shell",
        lambda: build_curved_roof(conveyor_loader, roof_base_z=roof_base_z, shell_meshes=shell_meshes),
    )
    support_info = _stage(
        "roof_truss",
        lambda: build_roof_truss_system(
            floor_top_z=floor_top_z,
            roof_base_z=roof_base_z,
            shell_meshes=shell_meshes,
        ),
    )
    wall_info.update(support_info)
    wall_info.update(
        _stage(
            "overhead_cranes",
            lambda: build_overhead_cranes(
                crane_loader=crane_loader,
                crane_model_name=crane_model_name,
                floor_top_z=floor_top_z,
                roof_base_z=roof_base_z,
                area_layout=area_layout,
                shell_meshes=shell_meshes,
                seed=seed,
            ),
        )
    )
    wall_info["build_stage_timings_s"] = {
        str(name): float(value) for name, value in build_stage_timings.items()
    }
    wall_info["build_stage_total_s"] = float(sum(build_stage_timings.values()))
    return wall_info


                                                                              
          
                                                                              
def setup_simulation(use_gui):
    if p.isConnected():
        p.disconnect()
    p.connect(p.GUI if use_gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.setRealTimeSimulation(0)
    p.configureDebugVisualizer(p.COV_ENABLE_KEYBOARD_SHORTCUTS, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if SHADOWS_DEFAULT else 0)
    _disable_debug_previews()


def _disable_debug_previews():
    if hasattr(p, "COV_ENABLE_RGB_BUFFER_PREVIEW"):
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
    if hasattr(p, "COV_ENABLE_DEPTH_BUFFER_PREVIEW"):
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
    if hasattr(p, "COV_ENABLE_SEGMENTATION_MARK_PREVIEW"):
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)


def _build_map_with_turbo(seed, use_gui, turbo_build, runtime_context=None):
    _ = turbo_build
    if use_gui:
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    try:
        return build_map(seed, runtime_context=runtime_context)
    finally:
        if use_gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        _disable_debug_previews()


def run(use_gui=True, seed=0, turbo_build=TURBO_BUILD_MODE_DEFAULT):
    setup_simulation(use_gui=use_gui)
    _ = turbo_build                                           
    turbo_build_enabled = True
    runtime_context = _create_runtime_context()
    active_seed = int(seed or 0)
    gen_start = time.perf_counter()
    info = _build_map_with_turbo(
        active_seed,
        use_gui=use_gui,
        turbo_build=turbo_build_enabled,
        runtime_context=runtime_context,
    )
    gen_seconds = time.perf_counter() - gen_start

    def _print_map_summary(summary_info, summary_seed, elapsed_s, show_controls):
        line = "=" * 62
        print("\n" + line)
        print("🏭 Warehouse Map Ready")
        print(line)

        print("📊 Build")
        print(f"  Seed: {summary_seed}")
        print(f"  Time: {elapsed_s:.2f}s | Size: {WAREHOUSE_SIZE_X:.1f}m x {WAREHOUSE_SIZE_Y:.1f}m | Turbo: ON")
        print(
            "  Roof: "
            f"ribs {summary_info.get('roof_ribs', summary_info.get('truss_frames', 0))} | "
            f"segments {summary_info.get('roof_rib_segments', summary_info.get('truss_members', 0))}"
        )
        stage_timings = summary_info.get("build_stage_timings_s", {}) or {}
        if stage_timings:
            ordered = sorted(
                ((str(k), float(v)) for k, v in stage_timings.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )
            top_chunks = [f"{name} {seconds:.2f}s" for name, seconds in ordered[:4]]
            if top_chunks:
                print("  Stages: " + " | ".join(top_chunks))
        print()

        door_states = list(summary_info.get("door_states", []))
        door_states_short = [
            str(state).replace(".obj", "").replace("door-wide-", "")
            for state in door_states
        ]
        print("🗺️ Layout")
        if door_states_short:
            print(f"  Doors: {', '.join(door_states_short)}")

        area_layout = summary_info.get("area_layout", {})
        if area_layout:
            zones = ", ".join(area_layout.keys())
            print(f"  Zones: {zones}")
        else:
            print("  Zones: n/a")
        if summary_info.get("personnel_floor_lane_enabled", False):
            print(f"  Personnel lane: tiles {summary_info.get('personnel_floor_lane_tiles', 0)}")
        print()

        print("📦 Assets")
        if summary_info.get("loading_staging_enabled", False):
            print(
                "  Staging: "
                f"pallets {summary_info.get('loading_staging_pallet_count', 0)} | "
                f"boxes {summary_info.get('loading_staging_box_count', 0)} | "
                f"barrels {summary_info.get('loading_staging_barrel_count', 0)} | "
                f"containers {summary_info.get('loading_container_count', 0)} | "
                f"empty stack {summary_info.get('loading_empty_pallet_stack_count', 0)}"
            )
        elif summary_info.get("loading_staging_reason"):
            print(f"  Staging: OFF ({summary_info.get('loading_staging_reason')})")
        if summary_info.get("storage_rack_enabled", False):
            print(
                "  Racks: "
                f"{summary_info.get('storage_rack_count', 0)} | "
                f"pallets {summary_info.get('storage_rack_pallet_count', 0)} | "
                f"boxes {summary_info.get('storage_rack_box_count', 0)} | "
                f"barrels {summary_info.get('storage_rack_barrel_count', 0)} | "
                f"endcaps {summary_info.get('storage_rack_endcap_count', 0)} | "
                f"no-top {summary_info.get('storage_rack_no_top_level_count', 0)}"
            )
        elif summary_info.get("storage_rack_reason"):
            print(f"  Racks: OFF ({summary_info.get('storage_rack_reason')})")
        print()

        ops_bits = []
        if "forklift_slot_count" in summary_info:
            ops_bits.append(
                "parking "
                f"{summary_info.get('forklift_spawned_count', 0)}/{summary_info.get('forklift_slot_count', 0)}"
            )
        if "loading_operation_forklift_count" in summary_info:
            ops_bits.append(f"loading {summary_info.get('loading_operation_forklift_count', 0)}")
        if "worker_count" in summary_info:
            ops_bits.append(f"workers {summary_info.get('worker_count', 0)}")
        if summary_info.get("overhead_cranes_enabled", False):
            ops_bits.append(f"cranes {summary_info.get('overhead_crane_count', 0)}")
        if ops_bits:
            print("⚙️ Operations")
            print("  " + " | ".join(ops_bits))
            print()

        if summary_info.get("factory_map_embedded", False):
            fsz = summary_info.get("factory_area_size_m", (0.0, 0.0))
            fnet = summary_info.get("factory_network", {})
            print("🏭 Factory")
            print(
                f"  Embedded: {fsz[0]:.0f}m x {fsz[1]:.0f}m | "
                f"cells {fnet.get('path_cells', 0)} | corners {fnet.get('corner_count', 0)} | "
                f"line {fnet.get('line_length_m', 0.0):.1f}m"
            )
            if fnet.get("section_split_index", -1) >= 0:
                print(
                    f"  Belts: {fnet.get('assembly_belt_model', 'n/a')} -> "
                    f"{fnet.get('packout_belt_model', 'n/a')} | split idx {fnet.get('section_split_index', -1)}"
                )
        elif summary_info.get("factory_map_reason"):
            print("🏭 Factory")
            print(f"  Embedded: OFF ({summary_info.get('factory_map_reason')})")

        if show_controls:
            print(line)
            print("🎮 Controls")
            print("  Camera: LMB Drag Rotate | WASD Move | Q/E Up-Down | Shift Faster")
            print("  Render: 1 Wireframe | 2 Shadows")
            print("  Session: R Rebuild (next seed) | ESC Quit")
            print(line)
        else:
            print(line)

    if not use_gui:
        for _ in range(5):
            p.stepSimulation()
        print(
            f"Headless ready | Seed {active_seed} | {gen_seconds:.2f}s"
        )
        p.disconnect()
        return

    _print_map_summary(info, active_seed, gen_seconds, show_controls=True)

    cam = CameraController(z=18.0, pitch=-42.0, speed=0.1875)
    wireframe_enabled = False
    shadows_enabled = SHADOWS_DEFAULT
    wireframe_pressed = False
    shadows_pressed = False
    r_pressed = False

    while True:
        keys = p.getKeyboardEvents()
        _disable_debug_previews()

        if keys.get(27, 0) & p.KEY_WAS_TRIGGERED:
            break

        cam.update(keys)

        if keys.get(ord("1"), 0) == 1:
            if not wireframe_pressed:
                wireframe_pressed = True
                wireframe_enabled = not wireframe_enabled
                p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1 if wireframe_enabled else 0)
        else:
            wireframe_pressed = False

        if keys.get(ord("2"), 0) == 1:
            if not shadows_pressed:
                shadows_pressed = True
                shadows_enabled = not shadows_enabled
                p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1 if shadows_enabled else 0)
        else:
            shadows_pressed = False

        if keys.get(ord("r"), 0) == 1:
            if not r_pressed:
                r_pressed = True
                active_seed += 1
                regen_start = time.perf_counter()
                info = _build_map_with_turbo(
                    active_seed,
                    use_gui=use_gui,
                    turbo_build=turbo_build_enabled,
                    runtime_context=runtime_context,
                )
                regen_seconds = time.perf_counter() - regen_start
                stage_timings = info.get("build_stage_timings_s", {}) or {}
                slowest = ""
                if stage_timings:
                    name, seconds = max(
                        ((str(k), float(v)) for k, v in stage_timings.items()),
                        key=lambda kv: kv[1],
                    )
                    slowest = f" | slowest {name} {seconds:.2f}s"
                print(
                    f"\n🔁 Rebuilt | Seed {active_seed} | {regen_seconds:.2f}s{slowest}"
                )
        else:
            r_pressed = False

        p.stepSimulation()
        time.sleep(1.0 / 60.0)

    p.disconnect()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description="Kenney-based closed warehouse shell.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for layout choices.")
    parser.add_argument("--headless", action="store_true", help="Run without GUI.")
    parser.add_argument(
        "--turbo",
        dest="turbo_build",
        action="store_true",
        default=TURBO_BUILD_MODE_DEFAULT,
        help="Enable turbo build mode (hide rendering while generating/rebuilding).",
    )
    parser.add_argument(
        "--no-turbo",
        dest="turbo_build",
        action="store_false",
        help="Disable turbo build mode.",
    )
    args = parser.parse_args()
    run(use_gui=not args.headless, seed=args.seed, turbo_build=args.turbo_build)

if __name__ == "__main__":
    main()
