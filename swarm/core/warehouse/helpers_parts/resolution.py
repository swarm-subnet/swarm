from ._shared import *


def _resolve_optional_model(directory, model_names):
    if not directory or not os.path.exists(directory):
        return "", ""
    root_abs = os.path.abspath(directory)
    for model_name in model_names:
        pth = os.path.join(root_abs, model_name)
        if os.path.exists(pth):
            return root_abs, model_name
    try:
        entries = os.listdir(root_abs)
    except OSError:
        return "", ""
    lower_map = {name.lower(): name for name in entries}
    for model_name in model_names:
        hit = lower_map.get(str(model_name).lower())
        if hit and hit.lower().endswith(".obj"):
            return root_abs, hit
    return "", ""


def _resolve_kit_paths():
    conveyor_obj = CONVEYOR_KIT_OBJ_DIR
    conveyor_tex = CONVEYOR_KIT_TEXTURE

    if not os.path.exists(conveyor_obj):
        raise FileNotFoundError(f"Missing conveyor OBJ folder: {conveyor_obj}")
    if not os.path.exists(conveyor_tex):
        raise FileNotFoundError(f"Missing conveyor texture: {conveyor_tex}")

    conveyor_obj_key = os.path.abspath(conveyor_obj)
    if conveyor_obj_key not in _NORMALIZED_MTL_DIRS:
        normalize_mtl_texture_paths(conveyor_obj)
        _NORMALIZED_MTL_DIRS.add(conveyor_obj_key)

    truck_obj = VEHICLE_DIR
    forklift_obj = VEHICLE_DIR
    loading_staging_obj = LOADING_KIT_DIR
    forklift_tex = (
        os.path.join(forklift_obj, FORKLIFT_TEXTURE_NAME)
        if FORKLIFT_TEXTURE_NAME
        else ""
    )

    if ENABLE_LOADING_TRUCKS:
        if not os.path.exists(truck_obj):
            raise FileNotFoundError(f"Missing truck OBJ folder: {truck_obj}")
        for model_name in LOADING_TRUCK_MODELS:
            mp = os.path.join(truck_obj, model_name)
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Missing truck model: {mp}")

    needs_industrial = (
        ENABLE_FORKLIFT_PARKING
        or ENABLE_MACHINING_CELL_LAYOUT
        or ENABLE_LOADING_OPERATION_FORKLIFTS
    )
    if needs_industrial and not os.path.exists(forklift_obj):
        raise FileNotFoundError(f"Missing industrial OBJ folder: {forklift_obj}")
    if ENABLE_FORKLIFT_PARKING:
        fp = os.path.join(forklift_obj, FORKLIFT_MODEL_NAME)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing forklift model: {fp}")

    needs_loading = ENABLE_LOADING_STAGING or ENABLE_STORAGE_RACK_LAYOUT
    if needs_loading and os.path.exists(loading_staging_obj):
        for model_name in LOADING_STAGING_MODELS.values():
            mp = os.path.join(loading_staging_obj, model_name)
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Missing loading staging model: {mp}")

    if ENABLE_STORAGE_RACK_LAYOUT:
        if not os.path.exists(loading_staging_obj):
            raise FileNotFoundError(
                f"Missing loading kit folder: {loading_staging_obj}"
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
        "loading_staging_obj": loading_staging_obj
        if os.path.exists(loading_staging_obj)
        else "",
    }


def _resolve_shell_mesh_paths():
    root = os.path.abspath(WAREHOUSE_SHELL_DIR)
    roof = os.path.join(root, WAREHOUSE_SHELL_FILES["roof"])
    fillers = os.path.join(root, WAREHOUSE_SHELL_FILES["fillers"])
    truss = os.path.join(root, WAREHOUSE_SHELL_FILES["truss"])

    if not (os.path.exists(roof) and os.path.exists(fillers) and os.path.exists(truss)):
        raise FileNotFoundError(
            f"Missing baked warehouse shell meshes in {root}. "
            f"Expected: {WAREHOUSE_SHELL_FILES}"
        )

    config = {}
    meta_path = os.path.join(root, "metadata.json")
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


# ---------------------------------------------------------------------------
# Loader utilities
# ---------------------------------------------------------------------------
