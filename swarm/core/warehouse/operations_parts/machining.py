from ._shared import *
from .support import _purge_generated_model_artifacts


def build_machining_cell_layout(industry_loader, floor_top_z, area_layout, cli):
    if not ENABLE_MACHINING_CELL_LAYOUT:
        return {
            "machining_mills": [],
            "machining_lathes": [],
            "machining_pending_slots": [],
        }
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
                cli=cli,
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
                cli=cli,
                model_path_override=spec.get("visual_path", spec.get("model_path", "")),
                collision_model_path_override=spec.get("collision_path", ""),
                with_collision=True,
                double_sided=MACHINING_VISUAL_DOUBLE_SIDED,
            )
        else:
            _spawn_mesh_with_anchor(
                loader=industry_loader,
                model_name=spec.get("collision_path", spec["model_name"]),
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=spec["scale_xyz"],
                local_anchor_xyz=(spec["anchor_x"], spec["anchor_y"], spec["anchor_z"]),
                cli=cli,
                with_collision=True,
                use_texture=False,
                rgba=spec.get("simple_rgba", (0.62, 0.64, 0.66, 1.0)),
                double_sided=MACHINING_VISUAL_DOUBLE_SIDED,
            )
        if (
            (not MACHINING_FORCE_SIMPLE_VISUALS)
            and (not MACHINING_USE_NATIVE_MTL_VISUALS)
            and spec["material_parts"]
        ):
            for part in spec["material_parts"]:
                part_tex = part.get("texture_path", "")
                use_part_tex = bool(
                    MACHINING_USE_PART_TEXTURES
                    and part_tex
                    and os.path.exists(part_tex)
                )
                part_rgba = (
                    [1.0, 1.0, 1.0, part["rgba"][3]] if use_part_tex else part["rgba"]
                )
                _spawn_mesh_with_anchor(
                    loader=industry_loader,
                    model_name=part["path"],
                    world_anchor_xyz=(x, y, floor_top_z),
                    yaw_deg=yaw_deg,
                    mesh_scale_xyz=spec["scale_xyz"],
                    local_anchor_xyz=(
                        spec["anchor_x"],
                        spec["anchor_y"],
                        spec["anchor_z"],
                    ),
                    cli=cli,
                    with_collision=False,
                    use_texture=use_part_tex,
                    texture_path_override=part_tex,
                    rgba=part_rgba,
                    double_sided=MACHINING_VISUAL_DOUBLE_SIDED,
                )
        elif (not MACHINING_FORCE_SIMPLE_VISUALS) and (
            not MACHINING_USE_NATIVE_MTL_VISUALS
        ):
            _spawn_mesh_with_anchor(
                loader=industry_loader,
                model_name=spec["model_name"],
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=spec["scale_xyz"],
                local_anchor_xyz=(spec["anchor_x"], spec["anchor_y"], spec["anchor_z"]),
                cli=cli,
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
                pending_slots.append(
                    {"slot_type": slot_type, "x": x, "y": y, "yaw_deg": yaw_deg}
                )
                if MACHINING_SHOW_PENDING_MARKERS:
                    px, py, pz = MACHINING_PENDING_SLOT_SIZE
                    _spawn_box_primitive(
                        center_xyz=(x, y, floor_top_z + (pz * 0.5) + 0.002),
                        size_xyz=(px, py, pz),
                        rgba=MACHINING_PENDING_RGBA,
                        cli=cli,
                        with_collision=False,
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
        cli=cli,
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
