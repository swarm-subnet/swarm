"""
Storage rack builder: build_storage_racks.
"""

import math
import os
import random

from .constants import (
    ENABLE_STORAGE_RACK_LAYOUT,
    LOADING_STAGING_MODELS,
    LOADING_STAGING_SCALES,
    STORAGE_RACK_BARREL_LAYER2_PROBABILITY,
    STORAGE_RACK_BARREL_RACK_PROBABILITY,
    STORAGE_RACK_BOX_LAYER2_PROBABILITY,
    STORAGE_RACK_BOX_PROBABILITY,
    STORAGE_RACK_CENTER_AISLE_TARGET_M,
    STORAGE_RACK_EDGE_MARGIN_M,
    STORAGE_RACK_ENABLE_ENDCAP_ROWS,
    STORAGE_RACK_FORCE_ALONG_AXIS,
    STORAGE_RACK_GLOBAL_YAW_OFFSET_DEG,
    STORAGE_RACK_GROUP_ROTATE_DEG,
    STORAGE_RACK_LAYOUT_FIXED_SEED,
    STORAGE_RACK_LEVEL_CONTACT_SNAP_M,
    STORAGE_RACK_LEVEL_DENSITY,
    STORAGE_RACK_LEVEL_MIN_CLEAR_M,
    STORAGE_RACK_LEVELS_RATIO,
    STORAGE_RACK_MAX_COUNT,
    STORAGE_RACK_MODEL_NAME,
    STORAGE_RACK_NO_TOP_LEVEL_PROBABILITY,
    STORAGE_RACK_PALLET_INSET_X_RATIO,
    STORAGE_RACK_PALLET_INSET_Y_RATIO,
    STORAGE_RACK_PALLET_LEVELS,
    STORAGE_RACK_PALLETS_PER_LEVEL,
    STORAGE_RACK_RGBA,
    STORAGE_RACK_ROW_GAP_M,
    STORAGE_RACK_SCALE_UNIFORM,
    STORAGE_RACK_SLOT_GAP_M,
    STORAGE_RACK_TARGET_ROW_COUNT,
)
from .helpers import (
    _floor_spawn_half_extents,
    _spawn_mesh_with_anchor,
    model_bounds_xyz,
)
from .loading import _spawn_obj_with_mtl_parts
from .storage_parts.helpers import (
    append_storage_endcaps,
    make_storage_layout_helpers,
    make_storage_support_helpers,
    pick_barrel_slot_keys,
    rotate_selected_slots,
    storage_plan_score,
)


def build_storage_racks(
    storage_loader, floor_top_z, area_layout, wall_info, cli, seed=0
):
    if not ENABLE_STORAGE_RACK_LAYOUT:
        return {"storage_rack_enabled": False}
    if storage_loader is None:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "Storage loader unavailable.",
        }
    storage_area = (area_layout or {}).get("STORAGE")
    if not storage_area:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "STORAGE area not found in area layout.",
        }
    rack_model = str(STORAGE_RACK_MODEL_NAME)
    rack_scale = (
        float(STORAGE_RACK_SCALE_UNIFORM),
        float(STORAGE_RACK_SCALE_UNIFORM),
        float(STORAGE_RACK_SCALE_UNIFORM),
    )
    pallet_model = str(LOADING_STAGING_MODELS["pallet"])
    box_model = str(LOADING_STAGING_MODELS["box"])
    barrel_model = str(LOADING_STAGING_MODELS["barrel"])
    pallet_scale = tuple(float(v) for v in LOADING_STAGING_SCALES["pallet"])
    box_scale = tuple(float(v) for v in LOADING_STAGING_SCALES["box"])
    barrel_scale = tuple(float(v) for v in LOADING_STAGING_SCALES["barrel"])
    try:
        rack_min_v, rack_max_v = model_bounds_xyz(
            storage_loader, rack_model, rack_scale
        )
        pallet_min_v, pallet_max_v = model_bounds_xyz(
            storage_loader, pallet_model, pallet_scale
        )
        box_min_v, box_max_v = model_bounds_xyz(storage_loader, box_model, box_scale)
        barrel_min_v, barrel_max_v = model_bounds_xyz(
            storage_loader, barrel_model, barrel_scale
        )
    except (FileNotFoundError, ValueError) as exc:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": f"Failed to prepare storage rack assets: {exc}",
        }
    rack_size_x = float(rack_max_v[0] - rack_min_v[0])
    rack_size_y = float(rack_max_v[1] - rack_min_v[1])
    rack_size_z = float(rack_max_v[2] - rack_min_v[2])
    rack_anchor_x = float((rack_min_v[0] + rack_max_v[0]) * 0.5)
    rack_anchor_y = float((rack_min_v[1] + rack_max_v[1]) * 0.5)
    rack_anchor_z = float(rack_min_v[2])
    pallet_size_x = float(pallet_max_v[0] - pallet_min_v[0])
    pallet_size_y = float(pallet_max_v[1] - pallet_min_v[1])
    pallet_size_z = float(pallet_max_v[2] - pallet_min_v[2])
    pallet_anchor_x = float((pallet_min_v[0] + pallet_max_v[0]) * 0.5)
    pallet_anchor_y = float((pallet_min_v[1] + pallet_max_v[1]) * 0.5)
    pallet_anchor_z = float(pallet_min_v[2])
    box_size_x = float(box_max_v[0] - box_min_v[0])
    box_size_y = float(box_max_v[1] - box_min_v[1])
    box_size_z = float(box_max_v[2] - box_min_v[2])
    box_anchor_x = float((box_min_v[0] + box_max_v[0]) * 0.5)
    box_anchor_y = float((box_min_v[1] + box_max_v[1]) * 0.5)
    box_anchor_z = float(box_min_v[2])
    barrel_size_x = float(barrel_max_v[0] - barrel_min_v[0])
    barrel_size_y = float(barrel_max_v[1] - barrel_min_v[1])
    barrel_size_z = float(barrel_max_v[2] - barrel_min_v[2])
    barrel_anchor_x = float((barrel_min_v[0] + barrel_max_v[0]) * 0.5)
    barrel_anchor_y = float((barrel_min_v[1] + barrel_max_v[1]) * 0.5)
    barrel_anchor_z = float(barrel_min_v[2])
    area_cx = float(storage_area["cx"])
    area_cy = float(storage_area["cy"])
    area_sx = float(storage_area["sx"])
    area_sy = float(storage_area["sy"])
    x_min = area_cx - (area_sx * 0.5)
    x_max = area_cx + (area_sx * 0.5)
    y_min = area_cy - (area_sy * 0.5)
    y_max = area_cy + (area_sy * 0.5)
    floor_half_x = float(wall_info.get("floor_spawn_half_x", 0.0))
    floor_half_y = float(wall_info.get("floor_spawn_half_y", 0.0))
    if floor_half_x <= 0.0 or floor_half_y <= 0.0:
        floor_half_x, floor_half_y = _floor_spawn_half_extents(storage_loader)
    storage_layout_seed = (
        int(STORAGE_RACK_LAYOUT_FIXED_SEED)
        if STORAGE_RACK_LAYOUT_FIXED_SEED is not None
        else int(seed)
    )
    rng = random.Random(int(storage_layout_seed) + 73129)
    pallets_per_level = max(1, int(STORAGE_RACK_PALLETS_PER_LEVEL))
    top_level_drop_prob = max(
        0.0, min(0.95, float(STORAGE_RACK_NO_TOP_LEVEL_PROBABILITY))
    )
    level_min_clear_m = float(STORAGE_RACK_LEVEL_MIN_CLEAR_M)
    rack_z_limit = float(floor_top_z) + float(rack_size_z)

    oriented_xy_local_cache = {}
    barrel_layout_profile_cache = {}
    box_layout_profile_cache = {}

    _layout_helpers = make_storage_layout_helpers(
        storage_loader=storage_loader,
        rack_model=rack_model,
        rack_scale=rack_scale,
        pallet_model=pallet_model,
        pallet_scale=pallet_scale,
        pallet_size_x=pallet_size_x,
        pallet_size_y=pallet_size_y,
        box_model=box_model,
        box_scale=box_scale,
        box_size_x=box_size_x,
        box_size_y=box_size_y,
        barrel_model=barrel_model,
        barrel_scale=barrel_scale,
        barrel_size_x=barrel_size_x,
        barrel_size_y=barrel_size_y,
        oriented_xy_local_cache=oriented_xy_local_cache,
        barrel_layout_profile_cache=barrel_layout_profile_cache,
        box_layout_profile_cache=box_layout_profile_cache,
    )
    _yaw_key = _layout_helpers.yaw_key
    _oriented_xy_cached = _layout_helpers.oriented_xy_cached
    _barrel_layout_profile_for_slot_yaw = (
        _layout_helpers.barrel_layout_profile_for_slot_yaw
    )
    _box_layout_profile_for_slot_yaw = _layout_helpers.box_layout_profile_for_slot_yaw
    _packed_centers = _layout_helpers.packed_centers

    edge_margin = max(0.35, float(STORAGE_RACK_EDGE_MARGIN_M))
    row_gap = max(1.4, float(STORAGE_RACK_ROW_GAP_M))
    slot_gap = max(0.25, float(STORAGE_RACK_SLOT_GAP_M))

    target_row_count = max(1, int(STORAGE_RACK_TARGET_ROW_COUNT))
    forced_axis_cfg = str(STORAGE_RACK_FORCE_ALONG_AXIS).strip().lower()
    if forced_axis_cfg in ("x", "y"):
        primary_along_axis = forced_axis_cfg
        axis_order = (forced_axis_cfg,)
    else:
        primary_along_axis = "x" if area_sx >= area_sy else "y"
        axis_order = ("x", "y") if primary_along_axis == "x" else ("y", "x")
    plan_candidates = []
    for preferred_along_axis in axis_order:
        yaw_candidates = []
        for rack_yaw in (0.0, 90.0):
            rack_ex, rack_ey = _oriented_xy_cached(rack_model, rack_scale, rack_yaw)
            along_size = float(rack_ex if preferred_along_axis == "x" else rack_ey)
            cross_size = float(rack_ey if preferred_along_axis == "x" else rack_ex)
            yaw_candidates.append(
                {
                    "yaw_deg": float(rack_yaw),
                    "rack_ex": float(rack_ex),
                    "rack_ey": float(rack_ey),
                    "along_size": along_size,
                    "cross_size": cross_size,
                    "is_long_along": along_size >= cross_size,
                }
            )

        yaw_candidates = sorted(
            yaw_candidates,
            key=lambda c: (
                1 if c["is_long_along"] else 0,
                float(c["along_size"]),
                -float(c["cross_size"]),
            ),
            reverse=True,
        )

        for cand in yaw_candidates:
            rack_yaw = float(cand["yaw_deg"])
            along_size = float(cand["along_size"])
            cross_size = float(cand["cross_size"])
            if preferred_along_axis == "x":
                along_lo = max(
                    x_min + edge_margin + (along_size * 0.5),
                    -floor_half_x + (along_size * 0.5),
                )
                along_hi = min(
                    x_max - edge_margin - (along_size * 0.5),
                    floor_half_x - (along_size * 0.5),
                )
                cross_lo = max(
                    y_min + edge_margin + (cross_size * 0.5),
                    -floor_half_y + (cross_size * 0.5),
                )
                cross_hi = min(
                    y_max - edge_margin - (cross_size * 0.5),
                    floor_half_y - (cross_size * 0.5),
                )
            else:
                along_lo = max(
                    y_min + edge_margin + (along_size * 0.5),
                    -floor_half_y + (along_size * 0.5),
                )
                along_hi = min(
                    y_max - edge_margin - (along_size * 0.5),
                    floor_half_y - (along_size * 0.5),
                )
                cross_lo = max(
                    x_min + edge_margin + (cross_size * 0.5),
                    -floor_half_x + (cross_size * 0.5),
                )
                cross_hi = min(
                    x_max - edge_margin - (cross_size * 0.5),
                    floor_half_x - (cross_size * 0.5),
                )
            if along_hi <= along_lo or cross_hi <= cross_lo:
                continue

            cross_span = float(cross_hi - cross_lo)
            max_rows_fit = max(
                1,
                int(
                    math.floor(
                        (cross_span + row_gap) / max(1e-6, (cross_size + row_gap))
                    )
                ),
            )
            rows_use = min(target_row_count, max_rows_fit)
            if rows_use <= 1:
                cross_centers = [0.5 * (cross_lo + cross_hi)]
            else:
                start = cross_lo + (cross_size * 0.5)
                end = cross_hi - (cross_size * 0.5)
                if end <= start:
                    continue
                step = (end - start) / float(rows_use - 1)
                cross_centers = [start + (i * step) for i in range(rows_use)]

            along_ranges = [(along_lo, along_hi)]
            slots = []
            along_centers_by_bank = {}
            for bank_idx, (alo, ahi) in enumerate(along_ranges):
                along_centers = _packed_centers(alo, ahi, along_size, slot_gap)
                if not along_centers:
                    continue
                along_centers_by_bank[int(bank_idx)] = [float(v) for v in along_centers]
                for row_idx, cross_v in enumerate(cross_centers):
                    for col_idx, along_v in enumerate(along_centers):
                        if preferred_along_axis == "x":
                            sx = float(along_v)
                            sy = float(cross_v)
                        else:
                            sx = float(cross_v)
                            sy = float(along_v)
                        slots.append(
                            {
                                "x": sx,
                                "y": sy,
                                "row": int(row_idx),
                                "col": int(col_idx),
                                "bank": int(bank_idx),
                                "along": float(along_v),
                            }
                        )
            if not slots:
                continue

            max_cols = max((len(v) for v in along_centers_by_bank.values()), default=0)
            plan_candidates.append(
                {
                    "yaw_deg": rack_yaw,
                    "along_axis": preferred_along_axis,
                    "slots": slots,
                    "rack_ex": float(cand["rack_ex"]),
                    "rack_ey": float(cand["rack_ey"]),
                    "along_size": float(along_size),
                    "cross_size": float(cross_size),
                    "along_lo": float(along_lo),
                    "along_hi": float(along_hi),
                    "cross_lo": float(cross_lo),
                    "cross_hi": float(cross_hi),
                    "along_centers_by_bank": along_centers_by_bank,
                    "cross_rows": [float(v) for v in cross_centers],
                    "is_long_along": bool(cand["is_long_along"]),
                    "slot_count": int(len(slots)),
                    "row_count": int(len(cross_centers)),
                    "max_cols": int(max_cols),
                }
            )

    best_plan = None
    if plan_candidates:

        best_plan = max(
            plan_candidates,
            key=lambda plan: storage_plan_score(plan, primary_along_axis),
        )

    if best_plan is None or not best_plan["slots"]:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "No valid storage rack grid fits STORAGE zone on floor.",
        }

    main_rack_yaw = float(best_plan["yaw_deg"])
    along_axis = str(best_plan.get("along_axis", "x")).strip().lower()

    yaw_pref_candidates = []
    for cand_yaw in (0.0, 90.0):
        cand_ex, cand_ey = _oriented_xy_cached(rack_model, rack_scale, cand_yaw)
        cand_along = float(cand_ex if along_axis == "x" else cand_ey)
        cand_cross = float(cand_ey if along_axis == "x" else cand_ex)
        yaw_pref_candidates.append(
            (
                float(cand_along - cand_cross),
                float(cand_along),
                -float(cand_cross),
                -abs(float(cand_yaw) - float(main_rack_yaw)),
                float(cand_yaw),
            )
        )
    if yaw_pref_candidates:
        yaw_pref_candidates.sort(reverse=True)
        main_rack_yaw = float(yaw_pref_candidates[0][4])

    rack_yaw_offset = float(STORAGE_RACK_GLOBAL_YAW_OFFSET_DEG) % 360.0
    main_rack_yaw = (main_rack_yaw + rack_yaw_offset) % 360.0
    main_rack_ex, main_rack_ey = _oriented_xy_cached(
        rack_model, rack_scale, main_rack_yaw
    )
    along_size = float(main_rack_ex if along_axis == "x" else main_rack_ey)
    cross_size = float(main_rack_ey if along_axis == "x" else main_rack_ex)
    if along_axis == "x":
        along_lo = max(
            x_min + edge_margin + (along_size * 0.5), -floor_half_x + (along_size * 0.5)
        )
        along_hi = min(
            x_max - edge_margin - (along_size * 0.5), floor_half_x - (along_size * 0.5)
        )
        cross_lo = max(
            y_min + edge_margin + (cross_size * 0.5), -floor_half_y + (cross_size * 0.5)
        )
        cross_hi = min(
            y_max - edge_margin - (cross_size * 0.5), floor_half_y - (cross_size * 0.5)
        )
    else:
        along_lo = max(
            y_min + edge_margin + (along_size * 0.5), -floor_half_y + (along_size * 0.5)
        )
        along_hi = min(
            y_max - edge_margin - (along_size * 0.5), floor_half_y - (along_size * 0.5)
        )
        cross_lo = max(
            x_min + edge_margin + (cross_size * 0.5), -floor_half_x + (cross_size * 0.5)
        )
        cross_hi = min(
            x_max - edge_margin - (cross_size * 0.5), floor_half_x - (cross_size * 0.5)
        )

    target_rows = max(1, int(STORAGE_RACK_TARGET_ROW_COUNT))
    center_aisle_target = max(0.0, float(STORAGE_RACK_CENTER_AISLE_TARGET_M))
    max_racks_cfg = int(STORAGE_RACK_MAX_COUNT)

    endcap_enabled = bool(STORAGE_RACK_ENABLE_ENDCAP_ROWS)
    endcap_rack_yaw = (main_rack_yaw + 90.0) % 360.0
    endcap_rack_ex, endcap_rack_ey = _oriented_xy_cached(
        rack_model, rack_scale, endcap_rack_yaw
    )
    endcap_along_size = float(endcap_rack_ex if along_axis == "x" else endcap_rack_ey)
    endcap_cross_size = float(endcap_rack_ey if along_axis == "x" else endcap_rack_ex)

    along_min_bound = along_lo - (along_size * 0.5)
    along_max_bound = along_hi + (along_size * 0.5)
    cross_min_bound = cross_lo - (cross_size * 0.5)
    cross_max_bound = cross_hi + (cross_size * 0.5)

    center_along_bound = 0.5 * (along_min_bound + along_max_bound)
    half_aisle = center_aisle_target * 0.5
    left_bound_lo = along_min_bound
    left_bound_hi = center_along_bound - half_aisle
    right_bound_lo = center_along_bound + half_aisle
    right_bound_hi = along_max_bound

    endcap_main_gap = 0.0
    min_side_span = along_size
    if endcap_enabled:
        min_side_span = endcap_along_size + endcap_main_gap + along_size
    side_span_left = max(0.0, left_bound_hi - left_bound_lo)
    side_span_right = max(0.0, right_bound_hi - right_bound_lo)
    if side_span_left < min_side_span or side_span_right < min_side_span:
        total_span = max(0.0, along_max_bound - along_min_bound)
        max_half_aisle = max(0.0, (total_span - (2.0 * min_side_span)) * 0.5)
        half_aisle = min(half_aisle, max_half_aisle)
        center_along_bound = 0.5 * (along_min_bound + along_max_bound)
        left_bound_lo = along_min_bound
        left_bound_hi = center_along_bound - half_aisle
        right_bound_lo = center_along_bound + half_aisle
        right_bound_hi = along_max_bound

    left_h_main_lo = left_bound_lo
    left_h_main_hi = left_bound_hi
    right_h_main_lo = right_bound_lo
    right_h_main_hi = right_bound_hi
    left_endcap_center = None
    right_endcap_center = None
    if endcap_enabled:
        if (left_bound_hi - left_bound_lo) >= (endcap_along_size - 1e-6):
            left_endcap_center = left_bound_lo + (endcap_along_size * 0.5)
            left_h_main_lo = (
                left_endcap_center + (endcap_along_size * 0.5) + endcap_main_gap
            )
        if (right_bound_hi - right_bound_lo) >= (endcap_along_size - 1e-6):
            right_endcap_center = right_bound_hi - (endcap_along_size * 0.5)
            right_h_main_hi = (
                right_endcap_center - (endcap_along_size * 0.5) - endcap_main_gap
            )

    left_main_lo = left_h_main_lo
    left_main_hi = left_h_main_hi
    right_main_lo = right_h_main_lo
    right_main_hi = right_h_main_hi
    left_centers = (
        _packed_centers(left_main_lo, left_main_hi, along_size, slot_gap)
        if left_main_hi >= left_main_lo
        else []
    )
    right_centers = (
        _packed_centers(right_main_lo, right_main_hi, along_size, slot_gap)
        if right_main_hi >= right_main_lo
        else []
    )
    if left_centers:
        left_first_target = left_h_main_lo + (along_size * 0.5)
        left_shift = float(left_first_target) - float(left_centers[0])
        left_centers = [float(v) + left_shift for v in left_centers]
    if right_centers:
        right_last_target = right_h_main_hi - (along_size * 0.5)
        right_shift = float(right_last_target) - float(right_centers[-1])
        right_centers = [float(v) + right_shift for v in right_centers]

    if not left_centers or not right_centers:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "Unable to create left/right rack banks with center aisle.",
        }

    row_cross_size = float(cross_size)
    cross_rows_lo = cross_min_bound + (row_cross_size * 0.5)
    cross_rows_hi = cross_max_bound - (row_cross_size * 0.5)
    if cross_rows_hi < cross_rows_lo:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "No valid storage rows fit cross-axis span.",
        }

    cross_span_centers = max(0.0, float(cross_rows_hi) - float(cross_rows_lo))
    max_rows_fit = max(
        1, int(math.floor(cross_span_centers / max(1e-6, row_cross_size))) + 1
    )
    rows_use = max(1, min(int(target_rows), int(max_rows_fit)))
    if rows_use <= 1:
        cross_centers = [0.5 * (cross_rows_lo + cross_rows_hi)]
    else:
        step = cross_span_centers / float(rows_use - 1)
        cross_centers = [float(cross_rows_lo) + (i * step) for i in range(rows_use)]

    _support_helpers = make_storage_support_helpers(
        along_axis=along_axis,
        storage_loader=storage_loader,
        rack_model=rack_model,
        rack_scale=rack_scale,
        rack_size_z=rack_size_z,
        rng=rng,
    )
    _to_world_xy = _support_helpers.to_world_xy

    selected = []
    for row_idx, cross_v in enumerate(cross_centers):
        for bank_idx in (0, 1):
            row_slots = []
            if bank_idx == 0:
                for along_v in left_centers:
                    row_slots.append(
                        {
                            "along": float(along_v),
                            "yaw_deg": float(main_rack_yaw),
                            "kind": "main",
                        }
                    )
            else:
                for along_v in right_centers:
                    row_slots.append(
                        {
                            "along": float(along_v),
                            "yaw_deg": float(main_rack_yaw),
                            "kind": "main",
                        }
                    )

            row_slots = sorted(row_slots, key=lambda s: float(s["along"]))
            for col_idx, slot in enumerate(row_slots):
                sx, sy = _to_world_xy(float(slot["along"]), float(cross_v))
                selected.append(
                    {
                        "x": sx,
                        "y": sy,
                        "row": int(row_idx),
                        "col": int(col_idx),
                        "bank": int(bank_idx),
                        "along": float(slot["along"]),
                        "yaw_deg": float(slot["yaw_deg"]),
                        "kind": str(slot["kind"]),
                    }
                )

    append_storage_endcaps(
        selected,
        endcap_enabled=endcap_enabled,
        left_endcap_center=left_endcap_center,
        right_endcap_center=right_endcap_center,
        cross_min_bound=cross_min_bound,
        cross_max_bound=cross_max_bound,
        endcap_cross_size=endcap_cross_size,
        endcap_rack_yaw=endcap_rack_yaw,
        target_rows=target_rows,
        along_axis=along_axis,
        _packed_centers=_packed_centers,
    )

    group_rotate_deg = float(STORAGE_RACK_GROUP_ROTATE_DEG) % 360.0
    rotate_selected_slots(
        selected,
        group_rotate_deg=group_rotate_deg,
        area_cx=area_cx,
        area_cy=area_cy,
    )

    if max_racks_cfg > 0 and len(selected) > max_racks_cfg:
        selected = selected[:max_racks_cfg]

    storage_endcap_slot_count = sum(
        1 for s in selected if str(s.get("kind", "")).lower() == "endcap"
    )

    selected_rows = {}
    for slot in selected:
        rk = (int(slot.get("row", 0)), int(slot.get("bank", 0)))
        selected_rows.setdefault(rk, []).append(slot)
    for rk in list(selected_rows.keys()):
        selected_rows[rk] = sorted(
            selected_rows[rk], key=lambda s: float(s.get("along", 0.0))
        )

    barrel_prob = max(0.0, min(0.95, float(STORAGE_RACK_BARREL_RACK_PROBABILITY)))
    barrel_slot_keys = pick_barrel_slot_keys(
        selected_rows,
        selected,
        barrel_prob=barrel_prob,
        rng=rng,
    )

    rack_yaw = float(main_rack_yaw)

    _cluster_level_area = _support_helpers.cluster_level_area
    _rack_support_surface_levels_m = _support_helpers.rack_support_surface_levels_m

    level_count = max(1, int(STORAGE_RACK_PALLET_LEVELS))
    level_ratios = list(STORAGE_RACK_LEVELS_RATIO[:level_count])
    if len(level_ratios) < level_count:
        level_ratios.extend([0.52] * (level_count - len(level_ratios)))

    contact_lift_m = min(
        0.008, max(0.0, float(STORAGE_RACK_LEVEL_CONTACT_SNAP_M) * 0.2)
    )
    support_levels = _rack_support_surface_levels_m()
    level_zs = [0.0]

    if level_count > 1 and support_levels:
        next_support_idx = 0
        for li in range(1, level_count):
            desired = float(level_ratios[li]) * rack_size_z
            chosen = None
            chosen_idx = None
            for si in range(next_support_idx, len(support_levels)):
                cand = float(support_levels[si]) + contact_lift_m
                if cand <= (level_zs[-1] + 0.02):
                    continue
                if cand >= (desired - 0.06):
                    chosen = cand
                    chosen_idx = si
                    break
            if chosen is None:
                for si in range(next_support_idx, len(support_levels)):
                    cand = float(support_levels[si]) + contact_lift_m
                    if cand > (level_zs[-1] + 0.02):
                        chosen = cand
                        chosen_idx = si
                        break
            if chosen is None:
                break
            level_zs.append(float(chosen))
            next_support_idx = int(chosen_idx) + 1

    fallback_max_level_z = rack_size_z - max(0.02, pallet_size_z * 0.15)
    while len(level_zs) < level_count:
        li = len(level_zs)
        ratio = (
            float(level_ratios[li])
            if li < len(level_ratios)
            else min(0.95, 0.50 + (0.18 * li))
        )
        z_rel = max(0.0, (ratio * rack_size_z) + contact_lift_m)
        if level_zs:
            z_rel = max(z_rel, level_zs[-1] + pallet_size_z + level_min_clear_m)
        z_rel = min(z_rel, fallback_max_level_z)
        if level_zs and z_rel <= level_zs[-1] + 0.03:
            break
        level_zs.append(float(z_rel))

    density_profile = [float(v) for v in STORAGE_RACK_LEVEL_DENSITY]
    if not density_profile:
        density_profile = [1.0]
    level_density_by_idx = []
    for i in range(len(level_zs)):
        if i < len(density_profile):
            d = density_profile[i]
        else:
            d = density_profile[-1]
        level_density_by_idx.append(max(0.25, min(1.0, float(d))))

    _level_slot_count = _support_helpers.level_slot_count

    if pallets_per_level <= 1:
        pallet_local_x_offsets = [0.0]
    else:
        off = rack_size_x * float(STORAGE_RACK_PALLET_INSET_X_RATIO)
        pallet_local_x_offsets = [-off, off]
    pallet_local_x_offsets_active = pallet_local_x_offsets[:pallets_per_level]
    pallet_local_y = rack_size_y * float(STORAGE_RACK_PALLET_INSET_Y_RATIO)

    rack_entries = []
    pallet_count = 0
    box_count = 0
    barrel_count = 0
    rack_no_top_level_count = 0
    top_level_idx = 2 if len(level_zs) >= 3 else -1
    box_entries = []
    for slot in selected:
        rx = float(slot["x"])
        ry = float(slot["y"])
        disable_top_level = False
        if top_level_idx >= 0 and rng.random() < top_level_drop_prob:
            disable_top_level = True
            rack_no_top_level_count += 1
        slot_yaw = float(slot.get("yaw_deg", rack_yaw))
        slot_yaw_rad = math.radians(slot_yaw)
        cos_y = math.cos(slot_yaw_rad)
        sin_y = math.sin(slot_yaw_rad)
        slot_rack_ex, slot_rack_ey = _oriented_xy_cached(
            rack_model, rack_scale, slot_yaw
        )
        _spawn_mesh_with_anchor(
            loader=storage_loader,
            model_name=rack_model,
            world_anchor_xyz=(rx, ry, floor_top_z),
            yaw_deg=slot_yaw,
            mesh_scale_xyz=rack_scale,
            local_anchor_xyz=(rack_anchor_x, rack_anchor_y, rack_anchor_z),
            cli=cli,
            with_collision=True,
            use_texture=False,
            rgba=STORAGE_RACK_RGBA,
            double_sided=False,
        )
        rack_entries.append(
            {
                "x": rx,
                "y": ry,
                "yaw_deg": slot_yaw,
                "footprint_xy_m": (float(slot_rack_ex), float(slot_rack_ey)),
                "size_xyz_m": (rack_size_x, rack_size_y, rack_size_z),
            }
        )

        for level_idx, z_rel in enumerate(level_zs):
            if disable_top_level and level_idx == top_level_idx:
                continue
            pallet_bottom_z = float(floor_top_z) + float(z_rel)
            level_density = float(
                level_density_by_idx[min(level_idx, len(level_density_by_idx) - 1)]
            )
            cargo_spawn_prob = max(
                0.0, min(1.0, float(STORAGE_RACK_BOX_PROBABILITY) * level_density)
            )
            barrel_layer2_prob = max(
                0.0,
                min(1.0, float(STORAGE_RACK_BARREL_LAYER2_PROBABILITY) * level_density),
            )
            box_layer2_prob = max(
                0.0,
                min(1.0, float(STORAGE_RACK_BOX_LAYER2_PROBABILITY) * level_density),
            )
            p_ex, p_ey = _oriented_xy_cached(pallet_model, pallet_scale, slot_yaw)
            for local_x in pallet_local_x_offsets_active:
                dx = (local_x * cos_y) - (pallet_local_y * sin_y)
                dy = (local_x * sin_y) + (pallet_local_y * cos_y)
                px = rx + dx
                py = ry + dy

                if (
                    (px - (p_ex * 0.5)) < (-floor_half_x - 1e-6)
                    or (px + (p_ex * 0.5)) > (floor_half_x + 1e-6)
                    or (py - (p_ey * 0.5)) < (-floor_half_y - 1e-6)
                    or (py + (p_ey * 0.5)) > (floor_half_y + 1e-6)
                ):
                    continue

                _spawn_obj_with_mtl_parts(
                    loader=storage_loader,
                    model_name=pallet_model,
                    world_anchor_xyz=(px, py, pallet_bottom_z),
                    yaw_deg=slot_yaw,
                    mesh_scale_xyz=pallet_scale,
                    local_anchor_xyz=(
                        pallet_anchor_x,
                        pallet_anchor_y,
                        pallet_anchor_z,
                    ),
                    cli=cli,
                    with_collision=True,
                )
                pallet_count += 1

                if rng.random() > cargo_spawn_prob:
                    continue

                slot_key = (
                    int(slot.get("row", 0)),
                    int(slot.get("bank", 0)),
                    int(slot.get("col", 0)),
                )
                use_barrel_cargo = slot_key in barrel_slot_keys

                if use_barrel_cargo:
                    barrel_base_z = pallet_bottom_z + pallet_size_z + 0.01
                    barrel_profile = _barrel_layout_profile_for_slot_yaw(slot_yaw)
                    barrel_yaw = float(barrel_profile["barrel_yaw"])
                    layer1_slots = list(barrel_profile["layer1_slots"])
                    rng.shuffle(layer1_slots)
                    layer1_count = _level_slot_count(len(layer1_slots), level_density)
                    layer1_placed = []
                    bex = float(barrel_profile["bex"])
                    bey = float(barrel_profile["bey"])
                    for bx_local, by_local in layer1_slots[:layer1_count]:
                        bdx = (bx_local * cos_y) - (by_local * sin_y)
                        bdy = (bx_local * sin_y) + (by_local * cos_y)
                        bx = px + bdx
                        by = py + bdy
                        if (
                            (bx - (bex * 0.5)) < (-floor_half_x - 1e-6)
                            or (bx + (bex * 0.5)) > (floor_half_x + 1e-6)
                            or (by - (bey * 0.5)) < (-floor_half_y - 1e-6)
                            or (by + (bey * 0.5)) > (floor_half_y + 1e-6)
                        ):
                            continue

                        overlap_hit = False
                        for prior in layer1_placed:
                            ddx = abs(float(bx) - float(prior["x"]))
                            ddy = abs(float(by) - float(prior["y"]))
                            lim_x = 0.5 * (float(bex) + float(prior["ex"])) - 0.002
                            lim_y = 0.5 * (float(bey) + float(prior["ey"])) - 0.002
                            if ddx < lim_x and ddy < lim_y:
                                overlap_hit = True
                                break
                        if overlap_hit:
                            continue

                        _spawn_obj_with_mtl_parts(
                            loader=storage_loader,
                            model_name=barrel_model,
                            world_anchor_xyz=(bx, by, barrel_base_z),
                            yaw_deg=barrel_yaw,
                            mesh_scale_xyz=barrel_scale,
                            local_anchor_xyz=(
                                barrel_anchor_x,
                                barrel_anchor_y,
                                barrel_anchor_z,
                            ),
                            cli=cli,
                            with_collision=False,
                        )
                        barrel_count += 1
                        box_entries.append(
                            {
                                "x": bx,
                                "y": by,
                                "z": barrel_base_z,
                                "yaw_deg": barrel_yaw,
                                "footprint_xy_m": (bex, bey),
                                "size_z_m": barrel_size_z,
                                "kind": "barrel",
                            }
                        )
                        layer1_placed.append({"x": bx, "y": by, "ex": bex, "ey": bey})

                    top_barrel_bottom = barrel_base_z + barrel_size_z + 0.01
                    max_barrel_bottom = rack_z_limit - barrel_size_z - 0.06
                    if (
                        layer1_placed
                        and top_barrel_bottom <= max_barrel_bottom
                        and rng.random() < barrel_layer2_prob
                    ):
                        top_seed = rng.choice(layer1_placed)
                        bx = float(top_seed["x"])
                        by = float(top_seed["y"])
                        _spawn_obj_with_mtl_parts(
                            loader=storage_loader,
                            model_name=barrel_model,
                            world_anchor_xyz=(bx, by, top_barrel_bottom),
                            yaw_deg=barrel_yaw,
                            mesh_scale_xyz=barrel_scale,
                            local_anchor_xyz=(
                                barrel_anchor_x,
                                barrel_anchor_y,
                                barrel_anchor_z,
                            ),
                            cli=cli,
                            with_collision=False,
                        )
                        barrel_count += 1
                        box_entries.append(
                            {
                                "x": bx,
                                "y": by,
                                "z": top_barrel_bottom,
                                "yaw_deg": barrel_yaw,
                                "footprint_xy_m": (bex, bey),
                                "size_z_m": barrel_size_z,
                                "kind": "barrel",
                            }
                        )
                    continue

                box_base_z = pallet_bottom_z + pallet_size_z + 0.01
                box_profile = _box_layout_profile_for_slot_yaw(slot_yaw)
                box_yaw = float(box_profile["box_yaw"])
                layer1_slots = list(box_profile["layer1_slots"])
                rng.shuffle(layer1_slots)
                layer1_count = _level_slot_count(len(layer1_slots), level_density)
                layer1_placed = []
                bex = float(box_profile["bex"])
                bey = float(box_profile["bey"])
                for bx_local, by_local in layer1_slots[:layer1_count]:
                    bdx = (bx_local * cos_y) - (by_local * sin_y)
                    bdy = (bx_local * sin_y) + (by_local * cos_y)
                    bx = px + bdx
                    by = py + bdy
                    if (
                        (bx - (bex * 0.5)) < (-floor_half_x - 1e-6)
                        or (bx + (bex * 0.5)) > (floor_half_x + 1e-6)
                        or (by - (bey * 0.5)) < (-floor_half_y - 1e-6)
                        or (by + (bey * 0.5)) > (floor_half_y + 1e-6)
                    ):
                        continue

                    overlap_hit = False
                    for prior in layer1_placed:
                        ddx = abs(float(bx) - float(prior["x"]))
                        ddy = abs(float(by) - float(prior["y"]))
                        lim_x = 0.5 * (float(bex) + float(prior["ex"])) - 0.002
                        lim_y = 0.5 * (float(bey) + float(prior["ey"])) - 0.002
                        if ddx < lim_x and ddy < lim_y:
                            overlap_hit = True
                            break
                    if overlap_hit:
                        continue

                    _spawn_obj_with_mtl_parts(
                        loader=storage_loader,
                        model_name=box_model,
                        world_anchor_xyz=(bx, by, box_base_z),
                        yaw_deg=box_yaw,
                        mesh_scale_xyz=box_scale,
                        local_anchor_xyz=(box_anchor_x, box_anchor_y, box_anchor_z),
                        cli=cli,
                        with_collision=False,
                    )
                    box_count += 1
                    box_entries.append(
                        {
                            "x": bx,
                            "y": by,
                            "z": box_base_z,
                            "yaw_deg": box_yaw,
                            "footprint_xy_m": (bex, bey),
                            "size_z_m": box_size_z,
                            "kind": "box",
                        }
                    )
                    layer1_placed.append({"x": bx, "y": by, "ex": bex, "ey": bey})

                top_box_bottom = box_base_z + box_size_z + 0.01
                max_box_bottom = rack_z_limit - box_size_z - 0.06
                if (
                    layer1_placed
                    and top_box_bottom <= max_box_bottom
                    and rng.random() < box_layer2_prob
                ):
                    top_seed = rng.choice(layer1_placed)
                    bx = float(top_seed["x"])
                    by = float(top_seed["y"])
                    _spawn_obj_with_mtl_parts(
                        loader=storage_loader,
                        model_name=box_model,
                        world_anchor_xyz=(bx, by, top_box_bottom),
                        yaw_deg=box_yaw,
                        mesh_scale_xyz=box_scale,
                        local_anchor_xyz=(box_anchor_x, box_anchor_y, box_anchor_z),
                        cli=cli,
                        with_collision=False,
                    )
                    box_count += 1
                    box_entries.append(
                        {
                            "x": bx,
                            "y": by,
                            "z": top_box_bottom,
                            "yaw_deg": box_yaw,
                            "footprint_xy_m": (bex, bey),
                            "size_z_m": box_size_z,
                            "kind": "box",
                        }
                    )

    if not rack_entries:
        return {
            "storage_rack_enabled": False,
            "storage_rack_reason": "Storage rack grid generated but no valid rack spawn points remained.",
        }

    return {
        "storage_rack_enabled": True,
        "storage_rack_model": rack_model,
        "storage_rack_scale_xyz": rack_scale,
        "storage_rack_count": len(rack_entries),
        "storage_rack_pallet_count": int(pallet_count),
        "storage_rack_box_count": int(box_count),
        "storage_rack_barrel_count": int(barrel_count),
        "storage_rack_no_top_level_count": int(rack_no_top_level_count),
        "storage_rack_endcap_count": int(storage_endcap_slot_count),
        "storage_rack_box_entries": box_entries,
        "storage_rack_level_zs_m": [float(v) for v in level_zs],
        "storage_racks": rack_entries,
    }
