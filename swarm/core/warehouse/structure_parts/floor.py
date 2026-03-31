from ._shared import *


def build_floor(loader, cli):
    tile_x, tile_y, _ = loader.model_size(CONVEYOR_ASSETS["floor"], UNIFORM_SCALE)
    margin_x = tile_x * 1
    margin_y = tile_y * 1
    floor_size_x = WAREHOUSE_SIZE_X - (2.0 * margin_x)
    floor_size_y = WAREHOUSE_SIZE_Y - (2.0 * margin_y)
    if floor_size_x <= 0.0 or floor_size_y <= 0.0:
        raise ValueError("Floor interior margin is too large for warehouse footprint.")

    floor_half_z = 0.03
    cid = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[floor_size_x * 0.5, floor_size_y * 0.5, floor_half_z],
        physicsClientId=cli,
    )
    vid = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[floor_size_x * 0.5, floor_size_y * 0.5, floor_half_z],
        rgbaColor=list(FLOOR_UNIFORM_COLOR),
        physicsClientId=cli,
    )
    floor_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=cid,
        baseVisualShapeIndex=vid,
        basePosition=[0.0, 0.0, -floor_half_z + 0.02],
        useMaximalCoordinates=True,
        physicsClientId=cli,
    )
    p.changeVisualShape(
        floor_id,
        -1,
        rgbaColor=list(FLOOR_UNIFORM_COLOR),
        textureUniqueId=-1,
        specularColor=list(UNIFORM_SPECULAR_COLOR),
        physicsClientId=cli,
    )
    return 0.02


def build_personnel_floor_lane(loader, floor_top_z, wall_info, cli):
    if not ENABLE_PERSONNEL_FLOOR_LANE:
        return {"personnel_floor_lane_enabled": False}

    lane_model = _first_existing_model_name(
        loader, PERSONNEL_FLOOR_LANE_MODEL_CANDIDATES
    )
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

    from ..constants import PERSONNEL_FLOOR_LANE_EDGE_TOLERANCE_TILES

    half_step = step_len * 0.5
    half_wid = lane_wid * 0.5
    edge_tol = step_len * PERSONNEL_FLOOR_LANE_EDGE_TOLERANCE_TILES
    floor_half_x, floor_half_y = _floor_spawn_half_extents(loader)
    face_x = min((WAREHOUSE_SIZE_X * 0.5) - (wall_thickness * 0.5), floor_half_x)
    face_y = min((WAREHOUSE_SIZE_Y * 0.5) - (wall_thickness * 0.5), floor_half_y)

    if move_along_x:
        y = max(-face_y + half_wid, min(face_y - half_wid, start_y))
        x0 = start_x + (dir_x * half_step)
        target_face_x = -face_x if dir_x < 0.0 else face_x
        x_target = target_face_x - (dir_x * half_step) - (dir_x * edge_tol)
        run_len = abs(x_target - x0)
    else:
        x = max(-face_x + half_wid, min(face_x - half_wid, start_x))
        y0 = start_y + (dir_y * half_step)
        target_face_y = -face_y if dir_y < 0.0 else face_y
        y_target = target_face_y - (dir_y * half_step) - (dir_y * edge_tol)
        run_len = abs(y_target - y0)

    tile_count = max(1, int(math.floor(run_len / step_len)) + 1)
    spawned = 0
    eps = 1e-6
    for i in range(tile_count + 2):
        if move_along_x:
            cx = x0 + (dir_x * step_len * i)
            if (dir_x < 0.0 and cx < (x_target - eps)) or (
                dir_x > 0.0 and cx > (x_target + eps)
            ):
                break
            cy = y
        else:
            cx = x
            cy = y0 + (dir_y * step_len * i)
            if (dir_y < 0.0 and cy < (y_target - eps)) or (
                dir_y > 0.0 and cy > (y_target + eps)
            ):
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
