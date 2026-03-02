"""
Warehouse spawn helpers, geometry utilities, OBJ material processing, and caches.
"""

import json
import math
import os
import random
import shutil

import pybullet as p

from .constants import (
    ASSETS_DIR,
    CONVEYOR_KIT_OBJ_DIR,
    CONVEYOR_KIT_TEXTURE,
    CONVEYOR_ASSETS,
    CRANE_DIR,
    ENABLE_FORKLIFT_PARKING,
    ENABLE_LOADING_OPERATION_FORKLIFTS,
    ENABLE_LOADING_STAGING,
    ENABLE_LOADING_TRUCKS,
    ENABLE_MACHINING_CELL_LAYOUT,
    ENABLE_OVERHEAD_CRANES,
    ENABLE_STORAGE_RACK_LAYOUT,
    ENABLE_WORKER_CREW,
    ENABLE_FACTORY_BARRIER_RING,
    FENCE_DIR,
    FLOOR_INNER_MARGIN_TILES,
    FLOOR_SPAWN_SAFETY_MARGIN_M,
    FORKLIFT_MODEL_NAME,
    FORKLIFT_TEXTURE_NAME,
    FURNITURE_KIT_OBJ_DIR,
    FURNITURE_KIT_TEXTURE,
    HALF_X,
    HALF_Y,
    LOADING_KIT_DIR,
    LOADING_STAGING_MODELS,
    LOADING_TRUCK_EXTRA_GAP_CLOSED,
    LOADING_TRUCK_EXTRA_GAP_HALF,
    LOADING_TRUCK_MODELS,
    LOADING_TRUCK_SCALE_XYZ,
    MACHINING_FORCE_REFRESH_MTL_PROXY,
    MESH_UP_FIX_RPY,
    OVERHEAD_CRANE_MODEL_CANDIDATES,
    STORAGE_RACK_MODEL_NAME,
    UNIFORM_SCALE,
    UNIFORM_SPECULAR_COLOR,
    VEHICLE_DIR,
    WAREHOUSE_BASE_SIZE_X,
    WAREHOUSE_BASE_SIZE_Y,
    WAREHOUSE_SHELL_DIR,
    WAREHOUSE_SHELL_FILES,
    WAREHOUSE_SIZE_X,
    WAREHOUSE_SIZE_Y,
    WORKER_MODEL_CANDIDATES,
)
from .shared import MeshKitLoader, first_existing_path, normalize_mtl_texture_paths


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_TEXTURE_CACHE = {}
_OBJ_MTL_SPLIT_CACHE = {}
_OBJ_COLLISION_PROXY_CACHE = {}
_OBJ_MTL_VISUAL_PROXY_CACHE = {}
_OBJ_DOUBLE_SIDED_PROXY_CACHE = {}
_LOADING_TRUCK_ALONG_EXTENT_CACHE = {}
_NORMALIZED_MTL_DIRS = set()
_MESH_VISUAL_SHAPE_CACHE = {}
_MESH_COLLISION_SHAPE_CACHE = {}
_RESOLVED_MESH_PATH_CACHE = {}
_ORIENTED_XY_SIZE_CACHE = {}
_MODEL_BOUNDS_CACHE = {}
_STORAGE_RACK_SUPPORT_LEVELS_CACHE = {}


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
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
            raise FileNotFoundError(f"Missing loading kit folder: {loading_staging_obj}")
        rack_mp = os.path.join(loading_staging_obj, STORAGE_RACK_MODEL_NAME)
        if not os.path.exists(rack_mp):
            raise FileNotFoundError(f"Missing storage rack model: {rack_mp}")

    return {
        "conveyor_obj": conveyor_obj,
        "conveyor_tex": conveyor_tex,
        "truck_obj": truck_obj,
        "forklift_obj": forklift_obj,
        "forklift_tex": forklift_tex,
        "loading_staging_obj": loading_staging_obj if os.path.exists(loading_staging_obj) else "",
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

    return {"root": root, "roof": roof, "fillers": fillers, "truss": truss, "config": config}


# ---------------------------------------------------------------------------
# Loader utilities
# ---------------------------------------------------------------------------
def _loader_runtime_key(loader):
    cached = getattr(loader, "_swarm_runtime_key", None)
    if cached:
        return cached
    obj_dir = str(getattr(loader, "obj_dir", "") or "").strip()
    key = os.path.abspath(obj_dir).replace("\\", "/") if obj_dir else f"loader:{id(loader)}"
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


# ---------------------------------------------------------------------------
# Texture cache (keyed by cli + path)
# ---------------------------------------------------------------------------
def _load_texture_cached(texture_path, cli):
    key = (cli, texture_path.replace("\\", "/"))
    if key in _TEXTURE_CACHE:
        return _TEXTURE_CACHE[key]
    tid = p.loadTexture(key[1], physicsClientId=cli)
    _TEXTURE_CACHE[key] = tid
    return tid


# ---------------------------------------------------------------------------
# Spawn helpers (all take cli parameter)
# ---------------------------------------------------------------------------
def _spawn_generated_mesh(
    mesh_path,
    texture_path,
    cli,
    with_collision=True,
    use_texture=True,
    rgba=(1.0, 1.0, 1.0, 1.0),
    double_sided=False,
    base_position=(0.0, 0.0, 0.0),
    mesh_scale_xyz=(1.0, 1.0, 1.0),
):
    mesh_key = mesh_path.replace("\\", "/")
    msx, msy, msz = float(mesh_scale_xyz[0]), float(mesh_scale_xyz[1]), float(mesh_scale_xyz[2])

    create_visual_kwargs = {}
    if double_sided and hasattr(p, "VISUAL_SHAPE_DOUBLE_SIDED"):
        create_visual_kwargs["flags"] = p.VISUAL_SHAPE_DOUBLE_SIDED
    vid = p.createVisualShape(
        p.GEOM_MESH,
        fileName=mesh_key,
        meshScale=[msx, msy, msz],
        rgbaColor=list(rgba),
        physicsClientId=cli,
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
            physicsClientId=cli,
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
        physicsClientId=cli,
    )
    visual_kwargs = {"rgbaColor": list(rgba), "specularColor": list(UNIFORM_SPECULAR_COLOR)}
    if use_texture:
        p.changeVisualShape(
            body, -1, textureUniqueId=_load_texture_cached(texture_path, cli),
            physicsClientId=cli, **visual_kwargs,
        )
    else:
        p.changeVisualShape(body, -1, textureUniqueId=-1, physicsClientId=cli, **visual_kwargs)
    return body


def _spawn_mesh_with_anchor(
    loader,
    model_name,
    world_anchor_xyz,
    yaw_deg,
    mesh_scale_xyz,
    local_anchor_xyz,
    cli,
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
    visual_key = (cli, mesh_path, scale_key, rgba_key, bool(double_sided), frame_quat_key)

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
            physicsClientId=cli,
            **create_visual_kwargs,
        )
        _MESH_VISUAL_SHAPE_CACHE[visual_key] = visual_id

    if with_collision:
        collision_key = (cli, mesh_path, scale_key, frame_quat_key)
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
                physicsClientId=cli,
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
        physicsClientId=cli,
    )
    visual_kwargs = {"rgbaColor": list(rgba), "specularColor": list(UNIFORM_SPECULAR_COLOR)}
    if use_texture:
        tex_id = (
            _load_texture_cached(texture_path_override, cli)
            if texture_path_override
            else loader._ensure_texture()
        )
        p.changeVisualShape(body_id, -1, textureUniqueId=tex_id, physicsClientId=cli, **visual_kwargs)
    else:
        p.changeVisualShape(body_id, -1, textureUniqueId=-1, physicsClientId=cli, **visual_kwargs)
    return body_id


def _spawn_native_mtl_visual_with_anchor(
    loader,
    model_name,
    world_anchor_xyz,
    yaw_deg,
    mesh_scale_xyz,
    local_anchor_xyz,
    cli,
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
        physicsClientId=cli,
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
            physicsClientId=cli,
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
        physicsClientId=cli,
    )
    return body_id


def _spawn_collision_only_with_anchor(
    loader,
    model_name,
    world_anchor_xyz,
    yaw_deg,
    mesh_scale_xyz,
    local_anchor_xyz,
    cli,
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
    frame = frame_quat_override if frame_quat_override is not None else loader.up_fix_quat
    collision_id = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=[sx, sy, sz],
        collisionFrameOrientation=frame,
        physicsClientId=cli,
        **kwargs,
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=-1,
        basePosition=base_pos,
        baseOrientation=yaw_quat,
        useMaximalCoordinates=True,
        physicsClientId=cli,
    )
    p.changeVisualShape(
        body_id, -1,
        rgbaColor=[1.0, 1.0, 1.0, 0.0],
        textureUniqueId=-1,
        specularColor=list(UNIFORM_SPECULAR_COLOR),
        physicsClientId=cli,
    )
    return body_id


def _spawn_box_primitive(center_xyz, size_xyz, rgba, cli, with_collision=True):
    hx = float(size_xyz[0]) * 0.5
    hy = float(size_xyz[1]) * 0.5
    hz = float(size_xyz[2]) * 0.5
    vid = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=list(rgba), physicsClientId=cli,
    )
    cid = (
        p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=cli)
        if with_collision
        else -1
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=cid,
        baseVisualShapeIndex=vid,
        basePosition=list(center_xyz),
        useMaximalCoordinates=True,
        physicsClientId=cli,
    )
    p.changeVisualShape(
        body_id, -1,
        rgbaColor=list(rgba),
        textureUniqueId=-1,
        specularColor=list(UNIFORM_SPECULAR_COLOR),
        physicsClientId=cli,
    )
    return body_id


# ---------------------------------------------------------------------------
# OBJ processing (file-only operations, no cli needed)
# ---------------------------------------------------------------------------
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
    double_sided_root = os.path.join(os.path.dirname(model_path), "_double_sided")
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
                    colors[current] = [float(parts[1]), float(parts[2]), float(parts[3]), alpha.get(current, 1.0)]
                except Exception:
                    pass
            elif head == "Ka" and len(parts) >= 4:
                try:
                    ka_colors[current] = [float(parts[1]), float(parts[2]), float(parts[3])]
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
        colors[mat] = [rgb[0], rgb[1], rgb[2], alpha.get(mat, 1.0)]
    for mat, ka in ka_colors.items():
        if mat in colors:
            continue
        colors[mat] = [ka[0], ka[1], ka[2], alpha.get(mat, 1.0)]
    for mat in map_kd.keys():
        if mat in colors:
            continue
        colors[mat] = [0.62, 0.64, 0.66, alpha.get(mat, 1.0)]
    for mat in list(colors.keys()):
        if str(mat).strip().lower() == "slang":
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
            if int(sig.get("size", -1)) == model_sig["size"] and int(sig.get("mtime_ns", -1)) == model_sig["mtime_ns"]:
                cached_parts = []
                valid_manifest = True
                for item in manifest.get("parts", []) or []:
                    rel_or_abs = str(item.get("path", "")).strip()
                    if not rel_or_abs:
                        valid_manifest = False
                        break
                    part_path = (
                        rel_or_abs if os.path.isabs(rel_or_abs)
                        else os.path.join(split_root, rel_or_abs)
                    )
                    part_path = os.path.abspath(part_path)
                    if not os.path.exists(part_path):
                        valid_manifest = False
                        break
                    rgba = item.get("rgba", [0.72, 0.72, 0.72, 1.0])
                    if not isinstance(rgba, (list, tuple)) or len(rgba) < 3:
                        rgba = [0.72, 0.72, 0.72, 1.0]
                    rgba = [float(rgba[0]), float(rgba[1]), float(rgba[2]),
                            float(rgba[3]) if len(rgba) >= 4 else 1.0]
                    cached_parts.append({
                        "path": part_path,
                        "material": str(item.get("material", "default")),
                        "rgba": rgba,
                        "texture_path": str(item.get("texture_path", "")),
                    })
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
        return idx0 if 0 <= idx0 < count else None

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
        out_parts.append({
            "path": out_path,
            "material": mtl_name,
            "rgba": rgba,
            "texture_path": mtl_textures.get(mtl_name, ""),
        })

    try:
        manifest_parts = []
        for part in out_parts:
            part_path_abs = os.path.abspath(str(part.get("path", "")))
            part_rel = os.path.relpath(part_path_abs, split_root).replace("\\", "/")
            manifest_parts.append({
                "path": part_rel,
                "material": str(part.get("material", "default")),
                "rgba": [float(v) for v in list(part.get("rgba", [0.72, 0.72, 0.72, 1.0]))[:4]],
                "texture_path": str(part.get("texture_path", "")),
            })
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump({"version": 1, "model_sig": model_sig, "parts": manifest_parts},
                      mf, ensure_ascii=True, indent=0)
    except Exception:
        pass

    _OBJ_MTL_SPLIT_CACHE[cache_key] = out_parts
    return out_parts


# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------
def slot_point(slot, along, inward):
    if slot == "north":
        return along, HALF_Y - inward
    if slot == "south":
        return along, -HALF_Y + inward
    if slot == "east":
        return HALF_X - inward, along
    if slot == "west":
        return -HALF_X + inward, along
    raise ValueError(f"Unknown slot: {slot}")


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


def wall_yaw_for_slot(slot):
    return dock_inward_yaw_for_slot(slot)


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


def _first_existing_model_name(loader, candidates):
    for model_name in candidates:
        if os.path.exists(os.path.join(loader.obj_dir, model_name)):
            return model_name
    return None


def _shell_mesh_scale_xy(shell_meshes):
    cfg = shell_meshes.get("config", {}) or {}
    base_x = float(cfg.get("warehouse_size_x", WAREHOUSE_BASE_SIZE_X))
    base_y = float(cfg.get("warehouse_size_y", WAREHOUSE_BASE_SIZE_Y))
    sx = (float(WAREHOUSE_SIZE_X) / base_x) if abs(base_x) > 1e-9 else 1.0
    sy = (float(WAREHOUSE_SIZE_Y) / base_y) if abs(base_y) > 1e-9 else 1.0
    return sx, sy


def _floor_spawn_half_extents(loader, safety_margin_m=FLOOR_SPAWN_SAFETY_MARGIN_M):
    tile_x, tile_y, _ = loader.model_size(CONVEYOR_ASSETS["floor"], UNIFORM_SCALE)
    margin_x = tile_x * FLOOR_INNER_MARGIN_TILES
    margin_y = tile_y * FLOOR_INNER_MARGIN_TILES
    floor_half_x = (WAREHOUSE_SIZE_X - (2.0 * margin_x)) * 0.5
    floor_half_y = (WAREHOUSE_SIZE_Y - (2.0 * margin_y)) * 0.5
    safe_half_x = max(0.5, floor_half_x - float(safety_margin_m))
    safe_half_y = max(0.5, floor_half_y - float(safety_margin_m))
    return safe_half_x, safe_half_y


def _truck_extra_gap_for_gate_state(gate_model_name):
    name = os.path.basename(str(gate_model_name)).lower()
    if "closed" in name:
        return LOADING_TRUCK_EXTRA_GAP_CLOSED
    if "half" in name:
        return LOADING_TRUCK_EXTRA_GAP_HALF
    return 0.0


def _estimate_loading_truck_along_extent_m(loading_slot):
    cache_key = (str(loading_slot), tuple(float(v) for v in LOADING_TRUCK_SCALE_XYZ))
    if cache_key in _LOADING_TRUCK_ALONG_EXTENT_CACHE:
        return _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key]

    if not os.path.exists(VEHICLE_DIR):
        _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key] = 0.0
        return 0.0
    model_name = LOADING_TRUCK_MODELS[0] if LOADING_TRUCK_MODELS else ""
    model_path = os.path.join(VEHICLE_DIR, model_name) if model_name else ""
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


# ---------------------------------------------------------------------------
# Area-layout geometry (shared by structure.py + layout.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Window utilities (used by structure.py wall building)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Cache clearing for build reset
# ---------------------------------------------------------------------------
def clear_build_caches():
    _MESH_VISUAL_SHAPE_CACHE.clear()
    _MESH_COLLISION_SHAPE_CACHE.clear()
    _RESOLVED_MESH_PATH_CACHE.clear()
    _TEXTURE_CACHE.clear()
