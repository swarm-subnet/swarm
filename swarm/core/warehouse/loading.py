"""
Loading zone builders: trucks, overhead cranes, and staging props.
"""

import math
import os
import random


from .constants import (
    ENABLE_LOADING_TRUCKS,
    ENABLE_OVERHEAD_CRANES,
    ENABLE_LOADING_STAGING,
    LOADING_TRUCK_MODELS,
    LOADING_TRUCK_SCALE_XYZ,
    LOADING_TRUCK_WALL_GAP,
    OVERHEAD_CRANE_TARGET_BY_ZONE,
    OVERHEAD_CRANE_SCALE_UNIFORM,
    OVERHEAD_CRANE_MIN_SPACING_M,
    OVERHEAD_CRANE_ZONE_EDGE_MARGIN_M,
    OVERHEAD_CRANE_TRUSS_TOUCH_EXTRA_M,
    OVERHEAD_CRANE_ATTACH_CLEARANCE_M,
    OVERHEAD_CRANE_WITH_COLLISION,
    OVERHEAD_CRANE_COLOR_GAIN,
    OVERHEAD_CRANE_YAW_EXTRA_DEG,
    WAREHOUSE_SIZE_X,
    WAREHOUSE_SIZE_Y,
    WAREHOUSE_BASE_SIZE_X,
    WALL_SLOTS,
    LOADING_KIT_DIR,
    LOADING_STAGING_MODELS,
    LOADING_STAGING_SCALES,
    LOADING_CONTAINER_STACK_ENABLED,
    LOADING_CONTAINER_MODEL_NAME,
    LOADING_CONTAINER_SCALE_XYZ,
    LOADING_STAGING_TRUCK_TAIL_CLEARANCE_M,
    LOADING_STAGING_EDGE_MARGIN_M,
    LOADING_STAGING_MAX_DEPTH_M,
    LOADING_SECTION_MIN_SPAN_M,
    LOADING_BARREL_MAX_STACK_LAYERS,
    LOADING_EMPTY_PALLET_STACK_COUNT,
    LOADING_EMPTY_PALLET_STACK_MIN_LAYERS,
    LOADING_EMPTY_PALLET_STACK_MAX_LAYERS,
    LOADING_LOADED_PALLET_STACK_MIN_LAYERS,
    LOADING_LOADED_PALLET_STACK_MAX_LAYERS,
    LOADING_BUNDLES_PER_TRUCK_MIN,
    LOADING_BUNDLES_PER_TRUCK_MAX,
    LOADING_STAGING_PROP_GAP_M,
    LOADING_STAGING_GOODS_BACK_EDGE_PAD_M,
    LOADING_STAGING_SUPPORT_BACK_BIAS,
    LOADING_CONTAINER_STACK_VERTICAL_GAP_M,
)
from .helpers import (
    _spawn_mesh_with_anchor,
    _spawn_collision_only_with_anchor,
    _obj_material_parts,
    _obj_double_sided_proxy_path,
    slot_point,
    dock_inward_yaw_for_slot,
    model_bounds_xyz,
    _truck_extra_gap_for_gate_state,
)


def _spawn_obj_with_mtl_parts(
    loader,
    model_name,
    world_anchor_xyz,
    yaw_deg,
    mesh_scale_xyz,
    local_anchor_xyz,
    cli,
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
            cli=cli,
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
                cli=cli,
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
        cli=cli,
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


def build_overhead_cranes(
    crane_loader,
    crane_model_name,
    floor_top_z,
    roof_base_z,
    area_layout,
    shell_meshes,
    cli,
    seed=0,
):
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
            fracs = [
                (-0.40 + (0.80 * float(i) / float(max(1, count - 1))))
                for i in range(count)
            ]

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
            if math.hypot(float(x) - float(prev["x"]), float(y) - float(prev["y"])) < (
                min_spacing - 1e-6
            ):
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
            support_end_z = float(roof_base_z) + float(
                OVERHEAD_CRANE_TRUSS_TOUCH_EXTRA_M
            )
            anchor_world_z = float(support_end_z) - float(
                OVERHEAD_CRANE_ATTACH_CLEARANCE_M
            )
            if (anchor_world_z - crane_height) < (float(floor_top_z) + 1.2):
                anchor_world_z = float(floor_top_z) + 1.2 + crane_height

            yaw_deg = (float(yaw_deg) + float(OVERHEAD_CRANE_YAW_EXTRA_DEG)) % 360.0

            _spawn_obj_with_mtl_parts(
                loader=crane_loader,
                model_name=crane_model_name,
                world_anchor_xyz=(x, y, anchor_world_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=scale_xyz,
                local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                cli=cli,
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


def build_loading_staging(
    loading_loader, floor_top_z, area_layout, wall_info, cli, seed=0
):
    if not ENABLE_LOADING_STAGING:
        return {"loading_staging_enabled": False}
    if loading_loader is None:
        return {
            "loading_staging_enabled": False,
            "loading_staging_reason": (
                "Loading staging assets loader unavailable. Expected: "
                + LOADING_KIT_DIR
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
    dock_clearance = max(
        6.0, truck_depth + float(LOADING_STAGING_TRUCK_TAIL_CLEARANCE_M)
    )

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

    if (
        left_len >= LOADING_SECTION_MIN_SPAN_M
        and right_len >= LOADING_SECTION_MIN_SPAN_M
        and abs(gate_bias) <= 0.12
    ):
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

    if (
        gate_position == "center"
        and _valid_range(left_range)
        and _valid_range(right_range)
    ):
        office_area = (area_layout or {}).get("OFFICE")
        office_along = (
            float(office_area["cx"] if along_axis == "x" else office_area["cy"])
            if office_area
            else zone_center
        )
        left_mid = 0.5 * (left_range[0] + left_range[1])
        right_mid = 0.5 * (right_range[0] + right_range[1])
        if abs(left_mid - office_along) >= abs(right_mid - office_along):
            container_range = left_range
            goods_range = right_range
        else:
            container_range = right_range
            goods_range = left_range
    elif gate_position == "min":
        base = (
            right_range
            if _range_len(right_range) >= _range_len(left_range)
            else left_range
        )
        near_start = abs(base[0] - (truck_max + seg_gap)) <= 1e-6
        goods_range, container_range = _split_outer_range(
            base, near_at_start=near_start
        )
    else:
        base = (
            left_range
            if _range_len(left_range) >= _range_len(right_range)
            else right_range
        )
        near_start = abs(base[0] - (truck_max + seg_gap)) <= 1e-6
        goods_range, container_range = _split_outer_range(
            base, near_at_start=near_start
        )

    if not _valid_range(goods_range, min_len=max(4.0, p_along + 1.0)):
        fallback_ranges = sorted(
            [left_range, right_range], key=lambda r: _range_len(r), reverse=True
        )
        goods_range = (
            fallback_ranges[0] if fallback_ranges else (along_start, along_end)
        )
        container_range = fallback_ranges[1] if len(fallback_ranges) > 1 else None

    if not _valid_range(container_range, min_len=4.0):
        container_range = None

    if _valid_range(left_range, min_len=4.0) or _valid_range(right_range, min_len=4.0):
        if _range_len(left_range) >= _range_len(right_range):
            container_range = (
                left_range if _valid_range(left_range, min_len=4.0) else right_range
            )
        else:
            container_range = (
                right_range if _valid_range(right_range, min_len=4.0) else left_range
            )
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
            cli=cli,
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
                cli=cli,
            )

        model_path = loading_loader._asset_path(container_spec["model_name"])
        material_parts = _obj_material_parts(model_path)
        if material_parts:
            for part in material_parts:
                part_rgba = tuple(
                    float(v) for v in part.get("rgba", (0.70, 0.70, 0.70, 1.0))
                )
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
                    cli=cli,
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
            cli=cli,
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
        pallet_along_local = p_along
        pallet_cross_local = p_cross

        desired_gap = 0.03
        edge_margin_box = 0.01
        candidates = []
        for test_yaw in (yaw_deg % 360.0, (yaw_deg + 90.0) % 360.0):
            _bx, _by, box_along, box_cross = _oriented_xy(box_spec, test_yaw)
            max_gap_a = pallet_along_local - (2.0 * box_along) - (2.0 * edge_margin_box)
            max_gap_c = pallet_cross_local - (2.0 * box_cross) - (2.0 * edge_margin_box)
            if max_gap_a >= 0.0 and max_gap_c >= 0.0:
                usable_gap = min(desired_gap, max_gap_a, max_gap_c)
                candidates.append((usable_gap, test_yaw, box_along, box_cross))

        if candidates:
            candidates.sort(key=lambda t: t[0], reverse=True)
            gap_used, box_yaw, box_along, box_cross = candidates[0]
            off_a = (box_along * 0.5) + (gap_used * 0.5)
            off_c = (box_cross * 0.5) + (gap_used * 0.5)
            off_a_max = max(
                0.0, (pallet_along_local * 0.5) - (box_along * 0.5) - edge_margin_box
            )
            off_c_max = max(
                0.0, (pallet_cross_local * 0.5) - (box_cross * 0.5) - edge_margin_box
            )
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
            off_a_max = max(
                0.0, (pallet_along_local * 0.5) - (box_along * 0.5) - edge_margin_box
            )
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
        stack_layers = min(
            int(LOADING_BARREL_MAX_STACK_LAYERS), int(max(1, stack_layers))
        )
        pallet_h = float(pallet_spec["size_xyz"][2])
        barrel_h = float(barrel_spec["size_xyz"][2])
        pallet_over_cargo_gap = 0.01
        yaw_rad = math.radians(yaw_deg)
        ux = (math.cos(yaw_rad), math.sin(yaw_rad))
        uy = (-math.sin(yaw_rad), math.cos(yaw_rad))
        barrel_yaw_base = yaw_deg
        _rx, _ry, barrel_along, barrel_cross = _oriented_xy(
            barrel_spec, barrel_yaw_base
        )
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
                _spawn_prop(
                    barrel_spec, bx, by, barrel_z, barrel_yaw, with_collision=False
                )
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
    empty_stack_max_layers = max(
        empty_stack_min_layers, int(LOADING_EMPTY_PALLET_STACK_MAX_LAYERS)
    )
    along_half = p_along * 0.5
    side_gap = 0.8
    left_near_gate = (truck_min - side_gap) - along_half
    left_far_end = along_start + along_half
    right_near_gate = (truck_max + side_gap) + along_half
    right_far_end = along_end - along_half

    left_len = max(0.0, left_near_gate - left_far_end)
    right_len = max(0.0, right_far_end - right_near_gate)
    left_range = (min(left_near_gate, left_far_end), max(left_near_gate, left_far_end))
    right_range = (
        min(right_near_gate, right_far_end),
        max(right_near_gate, right_far_end),
    )

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
    container_min_span_for_three = max(
        container_min_span_for_one, 3.0 * container_min_span_for_one
    )
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
    preferred_can_one = (
        right_can_container if preferred_side_right else left_can_container
    )
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
        container_dir = -1.0
        container_gate_edge_along = container_lo
        empty_start_along = empty_lo
        empty_end_along = empty_hi
    else:
        container_hi = same_hi
        container_lo = max(same_lo, container_hi - container_span_use)
        empty_lo = container_lo
        empty_hi = container_hi
        container_dir = 1.0
        container_gate_edge_along = container_hi
        empty_start_along = empty_hi
        empty_end_along = empty_lo

    container_candidate_range = (container_lo, container_hi)

    if abs(empty_end_along - empty_start_along) < 0.2:
        tight_side_gap = 0.25
        if empty_on_right:
            empty_start_along = (truck_max + tight_side_gap) + along_half
            empty_end_along = along_end - along_half
        else:
            empty_start_along = (truck_min - tight_side_gap) - along_half
            empty_end_along = along_start + along_half

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

    container_range = (
        container_candidate_range
        if _valid_range(container_candidate_range, min_len=1.8)
        else None
    )

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
        empty_cross_front = (
            cross_depth_total
            - _container_cross_reserve
            - empty_container_gap
            - (p_cross * 0.5)
        )
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
    row_gap = max(0.45, float(LOADING_STAGING_PROP_GAP_M))
    col_gap_default = max(0.60, float(LOADING_STAGING_PROP_GAP_M) + 0.20)
    max_rows_fit = int((goods_cross_span + row_gap) // (p_cross + row_gap))
    if max_rows_fit < 1:
        return {
            "loading_staging_enabled": False,
            "loading_staging_reason": "Not enough room for loaded pallets in goods section.",
        }

    truck_alongs = sorted(
        float(t.get("along", 0.0)) for t in wall_info.get("loading_trucks", [])
    )
    if not truck_alongs:
        truck_alongs = (
            [float(v) for v in door_centers]
            if door_centers
            else [0.5 * (goods_layout_range[0] + goods_layout_range[1])]
        )
    truck_alongs = [
        a for a in truck_alongs if goods_layout_range[0] <= a <= goods_layout_range[1]
    ]
    if not truck_alongs:
        truck_alongs = [0.5 * (goods_layout_range[0] + goods_layout_range[1])]

    truck_lanes = []
    for i, a in enumerate(truck_alongs):
        lo = goods_layout_range[0] if i == 0 else 0.5 * (truck_alongs[i - 1] + a)
        hi = (
            goods_layout_range[1]
            if i == (len(truck_alongs) - 1)
            else 0.5 * (a + truck_alongs[i + 1])
        )
        lane_margin = 0.06
        lane_lo = lo + lane_margin
        lane_hi = hi - lane_margin
        if lane_hi > lane_lo:
            truck_lanes.append((i, a, lane_lo, lane_hi))

    loaded_stack_min = max(1, int(LOADING_LOADED_PALLET_STACK_MIN_LAYERS))
    loaded_stack_max = max(
        loaded_stack_min, int(LOADING_LOADED_PALLET_STACK_MAX_LAYERS)
    )
    truck_row_centers_used = []

    for truck_idx, truck_along, lane_lo, lane_hi in truck_lanes:
        lane_len = max(0.0, lane_hi - lane_lo)
        cols_fit = int((lane_len + col_gap_default) // (p_along + col_gap_default))
        if cols_fit < 1:
            continue

        bundles_min = min(
            int(LOADING_BUNDLES_PER_TRUCK_MIN), int(LOADING_BUNDLES_PER_TRUCK_MAX)
        )
        bundles_max = max(
            int(LOADING_BUNDLES_PER_TRUCK_MIN), int(LOADING_BUNDLES_PER_TRUCK_MAX)
        )
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
            row1_max = (
                goods_s_max
                - (p_cross * 0.5)
                - max(
                    1.20,
                    float(LOADING_STAGING_TRUCK_TAIL_CLEARANCE_M) + 0.80,
                )
            )
            if row1_max > row1_min:
                row1_center_s = _clamp(row0_center_s + row_step, row1_min, row1_max)
                row_centers_s.append(row1_center_s)
            else:
                row_bundle_counts = [bundle_count]

        bundle_count = sum(row_bundle_counts)
        stack_layers_per_bundle = [
            rng.randint(loaded_stack_min, loaded_stack_max) for _ in range(bundle_count)
        ]
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
                mixed_barrel_indices = set(
                    rng.sample(range(mixed_count), barrel_target)
                )
            row_cargo_modes = ["mixed"]

        bundle_cursor = 0
        for row_idx, row_count in enumerate(row_bundle_counts):
            if row_count <= 0:
                continue
            row_cargo = (
                row_cargo_modes[row_idx] if row_idx < len(row_cargo_modes) else "box"
            )

            if row_count <= 1:
                col_gap_use = 0.0
                total_cols_len = p_along
            else:
                fit_gap = (lane_len - (row_count * p_along)) / float(row_count - 1)
                col_gap_use = max(0.0, min(col_gap_default, fit_gap))
                total_cols_len = (row_count * p_along) + ((row_count - 1) * col_gap_use)

            first_col_center = (
                lane_lo + ((lane_len - total_cols_len) * 0.5) + (p_along * 0.5)
            )
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
                    _spawn_loaded_pallet_with_boxes(
                        px, py, yaw_along, stack_layers=layers
                    )

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
        min_truck_s = (
            min(truck_row_centers_used)
            if truck_row_centers_used
            else (usable_depth - (p_cross * 0.5))
        )
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
                cols_fit = int(
                    (span_len + col_gap_default) // (p_along + col_gap_default)
                )
                if cols_fit < 1:
                    continue

                cols_use = max(1, min(cols_fit, 5))
                if cols_use <= 1:
                    col_gap_use = 0.0
                    total_cols_len = p_along
                else:
                    fit_gap = (span_len - (cols_use * p_along)) / float(cols_use - 1)
                    col_gap_use = max(0.0, min(col_gap_default, fit_gap))
                    total_cols_len = (cols_use * p_along) + (
                        (cols_use - 1) * col_gap_use
                    )
                first_col_center = (
                    lo + ((span_len - total_cols_len) * 0.5) + (p_along * 0.5)
                )

                side_pref_barrel = side_name == "left"
                for ridx, s_center in enumerate(support_rows_s):
                    for cidx in range(cols_use):
                        along = first_col_center + (cidx * (p_along + col_gap_use))
                        px, py = _xy_from_along_s(along, s_center)
                        layers = rng.randint(support_min_layers, support_max_layers)
                        use_barrel = (
                            (ridx % 2 == 0) if side_pref_barrel else (ridx % 2 == 1)
                        )
                        if use_barrel:
                            _spawn_barrel_pallet(px, py, yaw_along, stack_layers=layers)
                        else:
                            _spawn_loaded_pallet_with_boxes(
                                px, py, yaw_along, stack_layers=layers
                            )

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
            row_alongs = [
                span_center + (dir_sign * ((float(i) * step_use) - cluster_half))
                for i in range(row_count)
            ]
        row_cross = row_cross_values[min(ridx, len(row_cross_values) - 1)]
        for along in row_alongs:
            empty_slot_positions.append((along, row_cross))

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
                    cand = (
                        score,
                        cand_yaw,
                        c_along,
                        c_cross,
                        base_count,
                        upper_count,
                        total_target,
                    )
                    if best is None or cand[0] > best[0]:
                        best = cand
                    break

        if best is None:
            return False

        (
            _score,
            container_yaw,
            c_along,
            c_cross,
            base_count,
            upper_count,
            total_target,
        ) = best
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
            gap_max = max(
                0.0, (section_span - (base_count * c_along)) / float(base_count - 1)
            )
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
            _spawn_container(
                cx, cy, floor_top_z, container_yaw, with_collision=True, body_rgba=None
            )
            container_count += 1
            container_entries.append({"x": cx, "y": cy, "level": 0})

        if upper_count >= 1 and len(base_alongs) >= 2:
            container_h = float(c_size_z)
            top_z = (
                floor_top_z
                + container_h
                + float(LOADING_CONTAINER_STACK_VERTICAL_GAP_M)
            )
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
                _spawn_container(
                    cx, cy, top_z, container_yaw, with_collision=True, body_rgba=None
                )
                container_count += 1
                container_entries.append({"x": cx, "y": cy, "level": 1})

        placed_now = int(container_count) - before
        if placed_now < total_target and not container_reason:
            container_reason = f"Container stack fallback: placed {placed_now}/{total_target} in LOADING section."
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
                    container_reason = (
                        "Container placed using full LOADING fallback range."
                    )
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
