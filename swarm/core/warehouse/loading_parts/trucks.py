from ._shared import *


def build_loading_trucks(truck_loader, floor_top_z, wall_info, cli):
    if not ENABLE_LOADING_TRUCKS:
        return {"truck_scale_xyz": LOADING_TRUCK_SCALE_XYZ, "loading_trucks": []}

    loading_side = wall_info.get("loading_side", "north")
    door_centers = list(wall_info.get("door_centers", []))
    door_states = list(wall_info.get("door_states", []))
    if not door_centers:
        return {"truck_scale_xyz": LOADING_TRUCK_SCALE_XYZ, "loading_trucks": []}

    outward_yaw = dock_inward_yaw_for_slot(loading_side)
    wall_thickness = float(wall_info.get("wall_thickness", 0.0))
    floor_spawn_half_x = float(
        wall_info.get("floor_spawn_half_x", (WAREHOUSE_SIZE_X * 0.5))
    )
    floor_spawn_half_y = float(
        wall_info.get("floor_spawn_half_y", (WAREHOUSE_SIZE_Y * 0.5))
    )

    trucks = []
    for i, along in enumerate(door_centers):
        model_name = LOADING_TRUCK_MODELS[i % len(LOADING_TRUCK_MODELS)]
        min_v, max_v = model_bounds_xyz(
            truck_loader, model_name, LOADING_TRUCK_SCALE_XYZ
        )
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
        inward = (
            inner_wall_face_inward
            + (depth_to_wall * 0.5)
            + wall_clearance
            + gate_extra_gap
        )
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
            cli=cli,
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
                    cli=cli,
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
                cli=cli,
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
