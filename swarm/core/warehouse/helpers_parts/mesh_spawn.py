from ._shared import *


def _loader_runtime_key(loader):
    cached = getattr(loader, "_swarm_runtime_key", None)
    if cached:
        return cached
    obj_dir = str(getattr(loader, "obj_dir", "") or "").strip()
    key = (
        os.path.abspath(obj_dir).replace("\\", "/")
        if obj_dir
        else f"loader:{id(loader)}"
    )
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
    msx, msy, msz = (
        float(mesh_scale_xyz[0]),
        float(mesh_scale_xyz[1]),
        float(mesh_scale_xyz[2]),
    )

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
    visual_kwargs = {
        "rgbaColor": list(rgba),
        "specularColor": list(UNIFORM_SPECULAR_COLOR),
    }
    if use_texture:
        p.changeVisualShape(
            body,
            -1,
            textureUniqueId=_load_texture_cached(texture_path, cli),
            physicsClientId=cli,
            **visual_kwargs,
        )
    else:
        p.changeVisualShape(
            body, -1, textureUniqueId=-1, physicsClientId=cli, **visual_kwargs
        )
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

    frame_quat = (
        frame_quat_override if frame_quat_override is not None else loader.up_fix_quat
    )
    frame_quat_key = (
        tuple(round(float(v), 8) for v in frame_quat)
        if frame_quat is not None
        else None
    )
    scale_key = (round(float(sx), 8), round(float(sy), 8), round(float(sz), 8))
    rgba_key = tuple(round(float(v), 6) for v in rgba)
    visual_key = (
        cli,
        mesh_path,
        scale_key,
        rgba_key,
        bool(double_sided),
        frame_quat_key,
    )

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
    visual_kwargs = {
        "rgbaColor": list(rgba),
        "specularColor": list(UNIFORM_SPECULAR_COLOR),
    }
    if use_texture:
        tex_id = (
            _load_texture_cached(texture_path_override, cli)
            if texture_path_override
            else loader._ensure_texture()
        )
        p.changeVisualShape(
            body_id, -1, textureUniqueId=tex_id, physicsClientId=cli, **visual_kwargs
        )
    else:
        p.changeVisualShape(
            body_id, -1, textureUniqueId=-1, physicsClientId=cli, **visual_kwargs
        )
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
            if collision_model_path_override
            and os.path.exists(collision_model_path_override)
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
    frame = (
        frame_quat_override if frame_quat_override is not None else loader.up_fix_quat
    )
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
        body_id,
        -1,
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
        p.GEOM_BOX,
        halfExtents=[hx, hy, hz],
        rgbaColor=list(rgba),
        physicsClientId=cli,
    )
    cid = (
        p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=cli
        )
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
        body_id,
        -1,
        rgbaColor=list(rgba),
        textureUniqueId=-1,
        specularColor=list(UNIFORM_SPECULAR_COLOR),
        physicsClientId=cli,
    )
    return body_id


# ---------------------------------------------------------------------------
# OBJ processing (file-only operations, no cli needed)
# ---------------------------------------------------------------------------
