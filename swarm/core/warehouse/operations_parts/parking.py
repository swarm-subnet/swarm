from ._shared import *
from .support import _forklift_yaw_back_to_wall


def build_forklift_parking(forklift_loader, floor_top_z, area_layout, cli, seed=0):
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
    yaw_deg = (
        _forklift_yaw_back_to_wall(attached_wall) + float(FORKLIFT_PARK_YAW_EXTRA_DEG)
    ) % 360.0

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
    max_cross_vehicle = area_cross - (
        float(FORKLIFT_WALL_BACK_CLEARANCE) + interior_margin
    )
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
    occupied_indices = (
        sorted(rng.sample(range(slot_count), spawn_count)) if spawn_count > 0 else []
    )
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
        slot_along = along_size + min(
            float(FORKLIFT_PARK_SLOT_ALONG_PAD_M), along_extra_max
        )
        slot_cross = cross_size + min(
            float(FORKLIFT_PARK_SLOT_CROSS_PAD_M), cross_extra_max
        )
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
                    cli=cli,
                    with_collision=False,
                )
                boundary_x = (
                    [x_min]
                    + [
                        0.5 * (slot_centers[i][0] + slot_centers[i + 1][0])
                        for i in range(slot_count - 1)
                    ]
                    + [x_max]
                )
                for bx in boundary_x:
                    _spawn_box_primitive(
                        center_xyz=(bx, divider_center_y, line_z),
                        size_xyz=(line_w, join_cross, line_h),
                        rgba=FORKLIFT_PARK_LINE_RGBA,
                        cli=cli,
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
                    cli=cli,
                    with_collision=False,
                )
                boundary_y = (
                    [y_min]
                    + [
                        0.5 * (slot_centers[i][1] + slot_centers[i + 1][1])
                        for i in range(slot_count - 1)
                    ]
                    + [y_max]
                )
                for by in boundary_y:
                    _spawn_box_primitive(
                        center_xyz=(divider_center_x, by, line_z),
                        size_xyz=(join_cross, line_w, line_h),
                        rgba=FORKLIFT_PARK_LINE_RGBA,
                        cli=cli,
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
            cli=cli,
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
                    cli=cli,
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
                cli=cli,
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
