"""
Type 5 warehouse map for Swarm subnet challenge environments.

Entry point: ``build_warehouse_map(seed, cli, start, goal)``
"""

import json
import os
import time

from .constants import (
    ASSETS_DIR,
    CONVEYOR_KIT_OBJ_DIR,
    CONVEYOR_KIT_TEXTURE,
    CRANE_DIR,
    ENABLE_CORNER_COLUMNS,
    ENABLE_FORKLIFT_PARKING,
    ENABLE_LOADING_OPERATION_FORKLIFTS,
    ENABLE_LOADING_STAGING,
    ENABLE_LOADING_TRUCKS,
    ENABLE_MACHINING_CELL_LAYOUT,
    ENABLE_OVERHEAD_CRANES,
    ENABLE_STORAGE_RACK_LAYOUT,
    ENABLE_WORKER_CREW,
    FORKLIFT_MODEL_NAME,
    FORKLIFT_TEXTURE_NAME,
    LOADING_KIT_DIR,
    LOADING_STAGING_MODELS,
    LOADING_TRUCK_MODELS,
    OVERHEAD_CRANE_MODEL_CANDIDATES,
    STORAGE_RACK_MODEL_NAME,
    VEHICLE_DIR,
    WAREHOUSE_SHELL_DIR,
    WAREHOUSE_SHELL_FILES,
    WORKER_MODEL_CANDIDATES,
)
from .factory import build_embedded_factory
from .helpers_parts.geometry import clear_build_caches
from .layout import build_area_layout_markers
from .loading import build_loading_staging, build_loading_trucks, build_overhead_cranes
from .office import build_embedded_office
from .operations import (
    build_forklift_parking,
    build_loading_operation_forklifts,
    build_machining_cell_layout,
    build_worker_crew,
)
from .shared import (
    MeshKitLoader,
    normalize_mtl_texture_paths,
)
from .shared import (
    first_existing_path as first_existing_path,
)
from .storage import build_storage_racks
from .structure import (
    build_columns,
    build_curved_roof,
    build_floor,
    build_personnel_floor_lane,
    build_roof_truss_system,
    build_walls,
)

# ---------------------------------------------------------------------------
# Asset resolution helpers
# ---------------------------------------------------------------------------
_NORMALIZED_MTL_DIRS = set()


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
    conveyor_obj = CONVEYOR_KIT_OBJ_DIR
    conveyor_tex = CONVEYOR_KIT_TEXTURE
    truck_obj = VEHICLE_DIR
    forklift_obj = VEHICLE_DIR
    forklift_tex = (
        os.path.join(forklift_obj, FORKLIFT_TEXTURE_NAME)
        if FORKLIFT_TEXTURE_NAME
        else ""
    )
    loading_staging_obj = LOADING_KIT_DIR

    if not os.path.exists(conveyor_obj):
        raise FileNotFoundError(f"Missing conveyor OBJ folder: {conveyor_obj}")
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

    needs_industrial_obj = (
        ENABLE_FORKLIFT_PARKING
        or ENABLE_MACHINING_CELL_LAYOUT
        or ENABLE_LOADING_OPERATION_FORKLIFTS
    )
    if needs_industrial_obj and not os.path.exists(forklift_obj):
        raise FileNotFoundError(f"Missing industrial OBJ folder: {forklift_obj}")

    if ENABLE_FORKLIFT_PARKING:
        fp = os.path.join(forklift_obj, FORKLIFT_MODEL_NAME)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing forklift model: {fp}")
        if forklift_tex and not os.path.exists(forklift_tex):
            raise FileNotFoundError(f"Missing forklift texture: {forklift_tex}")

    needs_loading_staging_obj = ENABLE_LOADING_STAGING or ENABLE_STORAGE_RACK_LAYOUT
    if needs_loading_staging_obj and os.path.exists(loading_staging_obj):
        for model_name in LOADING_STAGING_MODELS.values():
            mp = os.path.join(loading_staging_obj, model_name)
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Missing loading staging model: {mp}")

    if ENABLE_STORAGE_RACK_LAYOUT:
        if not os.path.exists(loading_staging_obj):
            raise FileNotFoundError(
                f"Missing loading staging folder: {loading_staging_obj}"
            )
        rack_mp = os.path.join(loading_staging_obj, STORAGE_RACK_MODEL_NAME)
        if not os.path.exists(rack_mp):
            raise FileNotFoundError(f"Missing storage rack model: {rack_mp}")

    return {
        "conveyor_obj": conveyor_obj,
        "conveyor_tex": conveyor_tex,
        "truck_obj": truck_obj,
        "forklift_obj": forklift_obj,
        "forklift_tex": forklift_tex,
        "loading_staging_obj": loading_staging_obj,
    }


def _resolve_shell_mesh_paths():
    root = os.path.abspath(WAREHOUSE_SHELL_DIR)
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
    raise FileNotFoundError(
        f"Missing baked warehouse shell meshes in {root}. "
        f"Expected: {WAREHOUSE_SHELL_FILES['roof']}, "
        f"{WAREHOUSE_SHELL_FILES['fillers']}, "
        f"{WAREHOUSE_SHELL_FILES['truss']}"
    )


# ---------------------------------------------------------------------------
# Loader cache helpers
# ---------------------------------------------------------------------------
_WORKER_ASSET_CANDIDATES = (
    os.path.join(ASSETS_DIR, os.pardir, "workers"),
    LOADING_KIT_DIR,
    VEHICLE_DIR,
)
_OVERHEAD_CRANE_ASSET_CANDIDATES = (CRANE_DIR,)


def _clear_loader_spawn_caches(loader):
    if loader is None:
        return
    if hasattr(loader, "visual_shape_cache"):
        loader.visual_shape_cache.clear()
    if hasattr(loader, "collision_shape_cache"):
        loader.collision_shape_cache.clear()
    if hasattr(loader, "texture_id"):
        loader.texture_id = None


# ---------------------------------------------------------------------------
# Runtime context
# ---------------------------------------------------------------------------
def _create_runtime_context(cli=0):
    kit_paths = _resolve_kit_paths()
    shell_meshes = _resolve_shell_mesh_paths()

    ctx = {
        "kit_paths": kit_paths,
        "shell_meshes": shell_meshes,
        "worker_model_name": "",
        "crane_model_name": "",
    }

    ctx["conveyor_loader"] = MeshKitLoader(
        obj_dir=kit_paths["conveyor_obj"],
        texture_path=kit_paths["conveyor_tex"],
        cli=cli,
    )

    ctx["truck_loader"] = (
        MeshKitLoader(
            obj_dir=kit_paths["truck_obj"],
            texture_path=kit_paths["conveyor_tex"],
            cli=cli,
        )
        if ENABLE_LOADING_TRUCKS
        else None
    )

    needs_industry = (
        ENABLE_FORKLIFT_PARKING
        or ENABLE_MACHINING_CELL_LAYOUT
        or ENABLE_LOADING_OPERATION_FORKLIFTS
    )
    ctx["industry_loader"] = (
        MeshKitLoader(
            obj_dir=kit_paths["forklift_obj"],
            texture_path=kit_paths["forklift_tex"],
            cli=cli,
        )
        if needs_industry
        else None
    )

    has_staging_dir = kit_paths.get("loading_staging_obj") and os.path.exists(
        kit_paths["loading_staging_obj"]
    )
    ctx["loading_staging_loader"] = (
        MeshKitLoader(
            obj_dir=kit_paths["loading_staging_obj"],
            texture_path=kit_paths["conveyor_tex"],
            cli=cli,
        )
        if (ENABLE_LOADING_STAGING or ENABLE_STORAGE_RACK_LAYOUT) and has_staging_dir
        else None
    )

    worker_loader = None
    if ENABLE_WORKER_CREW:
        worker_obj_dir, worker_model_name = _resolve_optional_model(
            _WORKER_ASSET_CANDIDATES,
            WORKER_MODEL_CANDIDATES,
        )
        if worker_obj_dir and worker_model_name:
            worker_loader = MeshKitLoader(
                obj_dir=worker_obj_dir, texture_path="", cli=cli
            )
            ctx["worker_model_name"] = worker_model_name
    ctx["worker_loader"] = worker_loader

    crane_loader = None
    if ENABLE_OVERHEAD_CRANES:
        crane_obj_dir, crane_model_name = _resolve_optional_model(
            _OVERHEAD_CRANE_ASSET_CANDIDATES,
            OVERHEAD_CRANE_MODEL_CANDIDATES,
        )
        if crane_obj_dir and crane_model_name:
            crane_loader = MeshKitLoader(
                obj_dir=crane_obj_dir, texture_path="", cli=cli
            )
            ctx["crane_model_name"] = crane_model_name
    ctx["crane_loader"] = crane_loader

    return ctx


def _reset_runtime_context(ctx):
    for key in (
        "conveyor_loader",
        "truck_loader",
        "industry_loader",
        "loading_staging_loader",
        "worker_loader",
        "crane_loader",
    ):
        _clear_loader_spawn_caches(ctx.get(key))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_warehouse_map(seed, cli=0, start=None, goal=None):
    """Build a full Type 5 warehouse map inside an existing PyBullet world.

    Parameters
    ----------
    seed : int
        Deterministic generation seed.
    cli : int
        PyBullet physics client id.
    start, goal : optional
        Reserved for future use (spawn/goal markers).

    Returns
    -------
    dict
        Consolidated build info (wall geometry, area layout, asset counts, timing, etc.).
    """
    clear_build_caches()
    ctx = _create_runtime_context(cli=cli)
    _reset_runtime_context(ctx)

    conveyor_loader = ctx["conveyor_loader"]
    truck_loader = ctx.get("truck_loader")
    industry_loader = ctx.get("industry_loader")
    loading_staging_loader = ctx.get("loading_staging_loader")
    worker_loader = ctx.get("worker_loader")
    crane_loader = ctx.get("crane_loader")
    worker_model_name = ctx.get("worker_model_name", "")
    crane_model_name = ctx.get("crane_model_name", "")
    shell_meshes = ctx["shell_meshes"]

    build_stage_timings = {}

    def _stage(name, fn):
        t0 = time.perf_counter()
        out = fn()
        build_stage_timings[name] = time.perf_counter() - t0
        return out

    # 1. Floor
    floor_top_z = _stage(
        "floor",
        lambda: build_floor(conveyor_loader, cli=cli),
    )

    # 2. Walls
    wall_info = _stage(
        "walls",
        lambda: build_walls(conveyor_loader, floor_top_z, seed, cli=cli),
    )

    # 3. Personnel floor lane
    wall_info.update(
        _stage(
            "personnel_lane",
            lambda: build_personnel_floor_lane(
                conveyor_loader, floor_top_z, wall_info, cli=cli
            ),
        )
    )

    # 4. Loading trucks
    truck_info = _stage(
        "loading_trucks",
        lambda: (
            build_loading_trucks(truck_loader, floor_top_z, wall_info, cli=cli)
            if truck_loader is not None
            else {}
        ),
    )
    wall_info.update(truck_info)

    # 5. Area layout zones
    area_layout = _stage(
        "area_layout",
        lambda: build_area_layout_markers(
            conveyor_loader, floor_top_z, wall_info, seed=seed, cli=cli
        ),
    )
    wall_info["area_layout"] = area_layout

    # 6. Loading staging
    wall_info.update(
        _stage(
            "loading_staging",
            lambda: build_loading_staging(
                loading_staging_loader,
                floor_top_z,
                area_layout,
                wall_info,
                seed=seed,
                cli=cli,
            ),
        )
    )

    # 7. Storage racks
    wall_info.update(
        _stage(
            "storage_racks",
            lambda: build_storage_racks(
                loading_staging_loader,
                floor_top_z,
                area_layout,
                wall_info,
                seed=seed,
                cli=cli,
            ),
        )
    )

    # 8. Loading operation forklifts
    loading_op_info = _stage(
        "loading_operation_forklifts",
        lambda: (
            build_loading_operation_forklifts(
                industry_loader, floor_top_z, area_layout, wall_info, seed=seed, cli=cli
            )
            if industry_loader is not None
            else {}
        ),
    )
    wall_info.update(loading_op_info)

    # 9. Embedded office
    wall_info.update(
        _stage(
            "office",
            lambda: build_embedded_office(
                floor_top_z, area_layout, wall_info, cli=cli, seed=seed
            ),
        )
    )

    # 10. Embedded factory
    wall_info.update(
        _stage(
            "factory",
            lambda: build_embedded_factory(
                conveyor_loader, floor_top_z, area_layout, seed=seed, cli=cli
            ),
        )
    )

    # 11. Forklift parking
    forklift_info = _stage(
        "forklift_parking",
        lambda: (
            build_forklift_parking(
                industry_loader, floor_top_z, area_layout, seed=seed, cli=cli
            )
            if industry_loader is not None
            else {}
        ),
    )
    wall_info.update(forklift_info)

    # 12. Machining cell
    machining_info = _stage(
        "machining",
        lambda: (
            build_machining_cell_layout(
                industry_loader, floor_top_z, area_layout, cli=cli
            )
            if industry_loader is not None
            else {
                "machining_mills": [],
                "machining_lathes": [],
                "machining_pending_slots": [],
            }
        ),
    )
    wall_info.update(machining_info)

    # 13. Worker crew
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
                cli=cli,
            ),
        )
    )

    # 14. Columns (optional)
    roof_base_z = wall_info["roof_eave_z"]
    if ENABLE_CORNER_COLUMNS:
        _stage("columns", lambda: build_columns(conveyor_loader, floor_top_z, cli=cli))

    # 15. Curved roof shell
    _stage(
        "roof_shell",
        lambda: build_curved_roof(
            conveyor_loader, roof_base_z=roof_base_z, shell_meshes=shell_meshes, cli=cli
        ),
    )

    # 16. Roof truss system
    support_info = _stage(
        "roof_truss",
        lambda: build_roof_truss_system(
            floor_top_z=floor_top_z,
            roof_base_z=roof_base_z,
            shell_meshes=shell_meshes,
            cli=cli,
        ),
    )
    wall_info.update(support_info)

    # 17. Overhead cranes
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
                cli=cli,
            ),
        )
    )

    # 18. Timing metadata
    wall_info["build_stage_timings_s"] = {
        str(name): float(value) for name, value in build_stage_timings.items()
    }
    wall_info["build_stage_total_s"] = float(sum(build_stage_timings.values()))

    return wall_info
