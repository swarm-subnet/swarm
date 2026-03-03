"""
Warehouse structural build stages: floor, walls, personnel floor lane,
curved roof, roof truss system, and corner columns.
"""

import math
import os
import random

import pybullet as p

from .constants import (
    AREA_LAYOUT_BLOCKS,
    AREA_LAYOUT_EDGE_MARGIN,
    AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR,
    CONVEYOR_ASSETS,
    CURVED_ROOF_RISE,
    DOCK_INWARD_NUDGE,
    ENABLE_PERSONNEL_FLOOR_LANE,
    ENABLE_ROOF_TRUSS_SYSTEM,
    FLOOR_UNIFORM_COLOR,
    LOADING_DOOR_CENTER_STEP_FRACTION,
    LOADING_INTER_GATE_WALL_STEPS,
    LOADING_SLOTS,
    LOADING_TRUCK_WALL_GAP,
    PERSONNEL_FLOOR_LANE_MODEL_CANDIDATES,
    PERSONNEL_FLOOR_LANE_Z_OFFSET,
    ROOF_UNIFORM_COLOR,
    TRUSS_UNIFORM_COLOR,
    TRUSS_WITH_COLLISION,
    UNIFORM_SCALE,
    UNIFORM_SPECULAR_COLOR,
    WALL_SLOTS,
    WALL_TIERS,
    WALL_UNIFORM_COLOR,
    WAREHOUSE_SIZE_X,
    WAREHOUSE_SIZE_Y,
)
from .helpers import (
    _estimate_loading_truck_along_extent_m,
    _filter_mirrored_single_windows,
    _filter_mirrored_wide_windows,
    _first_existing_model_name,
    _floor_spawn_half_extents,
    _indices_blocked_by_doors,
    _loading_marker_xy_size,
    _merge_spans_1d,
    _orient_dims_long_side_on_wall,
    _shell_mesh_scale_xy,
    _spawn_generated_mesh,
    _spawn_mesh_with_anchor,
    _subtract_spans_1d,
    _wall_along_limits,
    dock_inward_yaw_for_slot,
    mirrored_wide_window_starts,
    mirrored_window_indices,
    oriented_xy_size,
    slot_point,
    tiled_centers,
    wall_yaw_for_slot,
)


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
        floor_id, -1,
        rgbaColor=list(FLOOR_UNIFORM_COLOR),
        textureUniqueId=-1,
        specularColor=list(UNIFORM_SPECULAR_COLOR),
        physicsClientId=cli,
    )
    return 0.02


def build_personnel_floor_lane(loader, floor_top_z, wall_info, cli):
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

    from .constants import PERSONNEL_FLOOR_LANE_EDGE_TOLERANCE_TILES

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


def build_curved_roof(loader, roof_base_z, shell_meshes, cli):
    shell_sx, shell_sy = _shell_mesh_scale_xy(shell_meshes)
    _spawn_generated_mesh(
        shell_meshes["roof"],
        loader.texture_path,
        cli,
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
        cli,
        with_collision=True,
        use_texture=False,
        rgba=WALL_UNIFORM_COLOR,
        double_sided=True,
        base_position=(0.0, 0.0, roof_base_z),
        mesh_scale_xyz=(shell_sx, shell_sy, 1.0),
    )


def build_roof_truss_system(floor_top_z, roof_base_z, shell_meshes, cli):
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
        cli=cli,
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


def build_columns(loader, floor_top_z, cli):
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


def build_walls(conveyor_loader, floor_top_z, seed, cli):
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
        pers_frame_ex if personnel_slot in ("north", "south") else pers_frame_ey, 6,
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
        "north": "south", "south": "north",
        "east": "west", "west": "east",
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
                cand_slot, cand_sx, cand_sy,
                cand_attach_half_x, cand_attach_half_y,
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
        "north": "south", "south": "north",
        "east": "west", "west": "east",
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
        loading_slot, float(loading_zone_sx), float(loading_zone_sy),
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
        loading_slot, loading_zone_sx, loading_zone_sy,
        zone_attach_half_x, zone_attach_half_y,
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
                segment_count, wide_span_steps, pair_seed_key,
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
            s for s in wide_start_set
            if (s >= reserved_left and (s + wide_span_steps) <= reserved_right)
        }
        window_indices = {i for i in window_indices if reserved_left <= i < reserved_right}
        if slot == personnel_slot:
            wide_start_set = {
                s for s in wide_start_set
                if not (s <= personnel_segment_idx < (s + wide_span_steps))
            }
            for p_idx in personnel_block_indices:
                window_indices.discard(p_idx)
        if slot == loading_slot:
            wide_start_set = set()
            window_indices = set()

        if slot == opposite_slot and not wide_start_set and not window_indices:
            fallback = {
                i for i in mirrored_window_indices(segment_count)
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
                            x=x, y=y, yaw_deg=wall_yaw,
                            floor_z=tier_base_z, scale=UNIFORM_SCALE,
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
                    span_along = straight_along[idx: idx + wide_span_steps]
                    along_center = sum(span_along) / float(len(span_along))
                    x, y = slot_point(slot, along_center, inward=wall_thickness * 0.5)
                    conveyor_loader.spawn(
                        window_wide_model,
                        x=x, y=y, yaw_deg=wall_yaw,
                        floor_z=tier_base_z, scale=UNIFORM_SCALE,
                        with_collision=True,
                    )
                    idx += wide_span_steps
                    continue

                use_small_window = (tier == WALL_TIERS - 1) and (idx in window_indices)
                model = window_model if use_small_window else wall_model
                x, y = slot_point(slot, along, inward=wall_thickness * 0.5)
                conveyor_loader.spawn(
                    model,
                    x=x, y=y, yaw_deg=wall_yaw,
                    floor_z=tier_base_z, scale=UNIFORM_SCALE,
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
                            x=x, y=y, yaw_deg=wall_yaw,
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
                        x=x, y=y, yaw_deg=wall_yaw,
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
                x=x, y=y, yaw_deg=yaw,
                floor_z=tier_base_z, scale=UNIFORM_SCALE,
                with_collision=True,
            )

    frame_min_v, frame_max_v = conveyor_loader._bounds(dock_frame_model, UNIFORM_SCALE)
    frame_anchor_x = (frame_min_v[0] + frame_max_v[0]) * 0.5
    frame_anchor_y = (frame_min_v[1] + frame_max_v[1]) * 0.5
    door_yaw = dock_inward_yaw_for_slot(loading_slot)
    frame_yaw = door_yaw

    for i, along in enumerate(door_centers):
        x, y = slot_point(
            loading_slot, along,
            inward=(loading_wall_thickness * 0.5) + DOCK_INWARD_NUDGE,
        )
        _spawn_mesh_with_anchor(
            loader=conveyor_loader,
            model_name=dock_frame_model,
            world_anchor_xyz=(x, y, floor_top_z),
            yaw_deg=frame_yaw,
            mesh_scale_xyz=(UNIFORM_SCALE, UNIFORM_SCALE, UNIFORM_SCALE),
            local_anchor_xyz=(frame_anchor_x, frame_anchor_y, 0.0),
            cli=cli,
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
            cli=cli,
            with_collision=True,
            use_texture=True,
            double_sided=True,
        )

    personnel_door_yaw = dock_inward_yaw_for_slot(personnel_slot)
    personnel_min_v, personnel_max_v = conveyor_loader._bounds(personnel_door_model, UNIFORM_SCALE)
    personnel_anchor_x = (personnel_min_v[0] + personnel_max_v[0]) * 0.5
    personnel_anchor_y = personnel_max_v[1]
    px, py = slot_point(
        personnel_slot, personnel_along,
        inward=(personnel_wall_thickness * 0.5) + DOCK_INWARD_NUDGE,
    )
    _spawn_mesh_with_anchor(
        loader=conveyor_loader,
        model_name=personnel_door_model,
        world_anchor_xyz=(px, py, floor_top_z),
        yaw_deg=personnel_door_yaw,
        mesh_scale_xyz=(UNIFORM_SCALE, UNIFORM_SCALE, UNIFORM_SCALE),
        local_anchor_xyz=(personnel_anchor_x, personnel_anchor_y, 0.0),
        cli=cli,
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
