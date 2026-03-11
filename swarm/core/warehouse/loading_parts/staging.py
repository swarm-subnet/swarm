from ._shared import *
from .staging_helpers import (
    make_build_spec_helper,
    make_staging_cargo_helpers,
    make_staging_layout_helpers,
    make_staging_spawn_helpers,
)
from .visuals import _spawn_obj_with_mtl_parts


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

    _build_spec = make_build_spec_helper(loading_loader)

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

    _layout_helpers = make_staging_layout_helpers(
        along_axis=along_axis,
        interior_edge=interior_edge,
        cross_to_dock_sign=cross_to_dock_sign,
        seg_gap=1.0,
    )
    _xy_from_along_s = _layout_helpers.xy_from_along_s
    _oriented_xy = _layout_helpers.oriented_xy
    _range_len = _layout_helpers.range_len
    _clamp = _layout_helpers.clamp
    _valid_range = _layout_helpers.valid_range

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
    _layout_helpers = make_staging_layout_helpers(
        along_axis=along_axis,
        interior_edge=interior_edge,
        cross_to_dock_sign=cross_to_dock_sign,
        seg_gap=seg_gap,
    )
    _xy_from_along_s = _layout_helpers.xy_from_along_s
    _oriented_xy = _layout_helpers.oriented_xy
    _range_len = _layout_helpers.range_len
    _clamp = _layout_helpers.clamp
    _valid_range = _layout_helpers.valid_range
    _split_outer_range = _layout_helpers.split_outer_range
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

    counts = {"box": 0, "barrel": 0, "pallet": 0, "container": 0}
    spawned_items = []
    container_entries = []
    empty_stack_entries = []
    empty_stack_groups = []
    container_state = {"spec": container_spec, "reason": container_reason}

    _spawn_helpers = make_staging_spawn_helpers(
        loading_loader=loading_loader,
        cli=cli,
        container_spec=container_spec,
    )
    _spawn_prop = _spawn_helpers.spawn_prop
    _spawn_container = _spawn_helpers.spawn_container

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

    _cargo_helpers = make_staging_cargo_helpers(
        floor_top_z=floor_top_z,
        pallet_spec=pallet_spec,
        box_spec=box_spec,
        barrel_spec=barrel_spec,
        p_along=p_along,
        p_cross=p_cross,
        _oriented_xy=_oriented_xy,
        _spawn_prop=_spawn_prop,
        _spawn_container=_spawn_container,
        spawned_items=spawned_items,
        container_entries=container_entries,
        counts=counts,
        container_state=container_state,
        along_axis=along_axis,
        cross_depth_total=cross_depth_total,
        yaw_along=yaw_along,
        container_dir=container_dir,
        container_gate_edge_along=container_gate_edge_along,
        _xy_from_along_s=_xy_from_along_s,
        _range_len=_range_len,
        _clamp=_clamp,
    )
    _spawn_loaded_pallet_with_boxes = _cargo_helpers.spawn_loaded_pallet_with_boxes
    _spawn_barrel_pallet = _cargo_helpers.spawn_barrel_pallet
    _place_container_stack_in_range = _cargo_helpers.place_container_stack_in_range

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

    placed_container = False
    if container_spec is not None and container_range is not None:
        placed_container = _place_container_stack_in_range(container_range)
        if not placed_container and not container_reason:
            container_reason = "Container section too small for preferred 3+2 stack."
    if container_spec is not None and counts["container"] <= 0:
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
        "loading_staging_pallet_count": counts["pallet"],
        "loading_staging_box_count": counts["box"],
        "loading_staging_barrel_count": counts["barrel"],
        "loading_empty_pallet_stack_count": len(empty_stack_groups),
        "loading_empty_pallet_total_count": len(empty_stack_entries),
        "loading_container_count": counts["container"],
        "loading_container_entries": container_entries,
        "loading_container_reason": container_state["reason"],
        "loading_pallet_size_xy_m": (round(p_ex, 2), round(p_ey, 2)),
        "loading_box_size_xy_m": (round(b_ex, 2), round(b_ey, 2)),
        "loading_staging_items": spawned_items,
    }
