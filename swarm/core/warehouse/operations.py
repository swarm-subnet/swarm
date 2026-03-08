"""
Operational builders: forklifts, worker crew, parking, machining cell.
"""

import math
import os
import random
import shutil


from .constants import (
    WALL_SLOTS,
    ENABLE_LOADING_OPERATION_FORKLIFTS,
    FORKLIFT_MODEL_NAME,
    FORKLIFT_SCALE_UNIFORM,
    LOADING_OPERATION_TRUCK_KEEPOUT_ALONG_PAD_M,
    LOADING_OPERATION_TRUCK_KEEPOUT_CROSS_PAD_M,
    LOADING_OPERATION_FORKLIFT_TRUCK_OFFSET_M,
    LOADING_OPERATION_FORKLIFT_EMPTY_OFFSET_M,
    LOADING_OPERATION_FORKLIFT_TARGET_COUNT,
    ENABLE_WORKER_CREW,
    WORKER_TARGET_HEIGHT_M,
    WORKER_MIN_SPACING_M,
    WORKER_TARGET_COUNT,
    WORKER_COLOR_GAIN,
    ENABLE_FORKLIFT_PARKING,
    FORKLIFT_AREA_PREFERENCE,
    FORKLIFT_PARK_YAW_EXTRA_DEG,
    FORKLIFT_WALL_BACK_CLEARANCE,
    FORKLIFT_PARK_SLOT_COUNT,
    FORKLIFT_PARK_GAP_M,
    FORKLIFT_PARK_SPAWN_MIN,
    FORKLIFT_PARK_SPAWN_MAX,
    ENABLE_FORKLIFT_PARK_SLOT_LINES,
    FORKLIFT_PARK_LINE_WIDTH_M,
    FORKLIFT_PARK_LINE_HEIGHT_M,
    FORKLIFT_PARK_LINE_CENTER_Z,
    FORKLIFT_PARK_LINE_RGBA,
    FORKLIFT_PARK_SLOT_ALONG_PAD_M,
    FORKLIFT_PARK_SLOT_CROSS_PAD_M,
    ENABLE_MACHINING_CELL_LAYOUT,
    MACHINING_CELL_AREA_NAME,
    MACHINING_EDGE_MARGIN,
    MACHINING_SLOT_TYPES,
    MACHINING_MILL_MODEL_NAME,
    MACHINING_MILL_SCALE_UNIFORM,
    MACHINING_LATHE_MODEL_NAME,
    MACHINING_LATHE_SCALE_UNIFORM,
    MACHINING_SIMPLE_MILL_RGBA,
    MACHINING_SIMPLE_LATHE_RGBA,
    MACHINING_AISLE_WIDTH,
    MACHINING_FORCE_SIMPLE_VISUALS,
    MACHINING_USE_NATIVE_MTL_VISUALS,
    MACHINING_VISUAL_DOUBLE_SIDED,
    MACHINING_FORCE_REFRESH_MTL_PROXY,
    MACHINING_USE_PART_TEXTURES,
    MACHINING_HEAVY_EXTRA_YAW_DEG,
    MACHINING_SHOW_PENDING_MARKERS,
    MACHINING_PENDING_SLOT_SIZE,
    MACHINING_PENDING_RGBA,
    MACHINING_TABLE_SIZE,
    MACHINING_TABLE_RGBA,
)
from .helpers import (
    _spawn_mesh_with_anchor,
    _spawn_native_mtl_visual_with_anchor,
    _spawn_collision_only_with_anchor,
    _spawn_box_primitive,
    _obj_material_parts,
    _obj_collision_proxy_path,
    _obj_mtl_visual_proxy_path,
    _attached_wall_from_area_bounds,
    slot_point,
    model_bounds_xyz,
    _OBJ_MTL_SPLIT_CACHE,
    _OBJ_COLLISION_PROXY_CACHE,
    _OBJ_MTL_VISUAL_PROXY_CACHE,
    _OBJ_DOUBLE_SIDED_PROXY_CACHE,
    _TEXTURE_CACHE,
    _safe_token_name,
)
from .loading import _spawn_obj_with_mtl_parts


def _forklift_yaw_back_to_wall(attached_wall):
    if attached_wall == "north":
        return 180.0
    if attached_wall == "south":
        return 0.0
    if attached_wall == "east":
        return 270.0
    if attached_wall == "west":
        return 90.0
    return 0.0


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

    double_sided_root = os.path.join(
        os.path.dirname(model_path),
        "_double_sided",
    )
    if os.path.isdir(double_sided_root):
        shutil.rmtree(double_sided_root, ignore_errors=True)


def build_loading_operation_forklifts(
    forklift_loader, floor_top_z, area_layout, wall_info, cli, seed=0
):
    if not ENABLE_LOADING_OPERATION_FORKLIFTS:
        return {
            "loading_operation_forklift_count": 0,
            "loading_operation_forklifts": [],
        }
    if forklift_loader is None:
        return {
            "loading_operation_forklift_count": 0,
            "loading_operation_forklifts": [],
            "loading_operation_forklift_reason": "Industrial OBJ loader unavailable.",
        }

    loading_area = (area_layout or {}).get("LOADING")
    if not loading_area:
        return {
            "loading_operation_forklift_count": 0,
            "loading_operation_forklifts": [],
            "loading_operation_forklift_reason": "LOADING area not found in layout.",
        }

    loading_side = str(wall_info.get("loading_side", "north")).lower()
    if loading_side not in WALL_SLOTS:
        loading_side = "north"

    model_name = FORKLIFT_MODEL_NAME
    scale_xyz = (FORKLIFT_SCALE_UNIFORM, FORKLIFT_SCALE_UNIFORM, FORKLIFT_SCALE_UNIFORM)
    min_v, max_v = model_bounds_xyz(forklift_loader, model_name, scale_xyz)
    size_x = max_v[0] - min_v[0]
    size_y = max_v[1] - min_v[1]
    size_z = max_v[2] - min_v[2]
    anchor_x = (min_v[0] + max_v[0]) * 0.5
    anchor_y = (min_v[1] + max_v[1]) * 0.5
    anchor_z = min_v[2]

    yaw_deg = _forklift_yaw_back_to_wall(loading_side)
    yaw = math.radians(yaw_deg)
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    ex = (c * size_x) + (s * size_y)
    ey = (s * size_x) + (c * size_y)

    area_cx = float(loading_area["cx"])
    area_cy = float(loading_area["cy"])
    area_sx = float(loading_area["sx"])
    area_sy = float(loading_area["sy"])
    x_min = area_cx - (area_sx * 0.5)
    x_max = area_cx + (area_sx * 0.5)
    y_min = area_cy - (area_sy * 0.5)
    y_max = area_cy + (area_sy * 0.5)

    if loading_side in ("north", "south"):
        along_axis = "x"
        along_min = x_min
        along_max = x_max
        dock_edge = y_max if loading_side == "north" else y_min
        interior_edge = y_min if loading_side == "north" else y_max
        along_size = ex
        cross_size = ey
    else:
        along_axis = "y"
        along_min = y_min
        along_max = y_max
        dock_edge = x_max if loading_side == "east" else x_min
        interior_edge = x_min if loading_side == "east" else x_max
        along_size = ey
        cross_size = ex

    cross_to_dock_sign = 1.0 if dock_edge >= interior_edge else -1.0
    cross_depth_total = abs(dock_edge - interior_edge)

    def _xy_from_along_s(along, s_from_interior):
        cross = interior_edge + (cross_to_dock_sign * s_from_interior)
        if along_axis == "x":
            return along, cross
        return cross, along

    def _along_s_from_xy(x, y):
        along = float(x) if along_axis == "x" else float(y)
        cross = float(y) if along_axis == "x" else float(x)
        s_from_interior = (cross - interior_edge) * cross_to_dock_sign
        return along, s_from_interior

    def _clamp(v, lo, hi):
        return max(lo, min(hi, v))

    along_margin = (along_size * 0.5) + 0.30
    cross_margin = (cross_size * 0.5) + 0.30
    along_lo = along_min + along_margin
    along_hi = along_max - along_margin
    s_lo = cross_margin
    s_hi = cross_depth_total - cross_margin
    if along_hi <= along_lo or s_hi <= s_lo:
        return {
            "loading_operation_forklift_count": 0,
            "loading_operation_forklifts": [],
            "loading_operation_forklift_reason": "LOADING area too tight for operational forklifts.",
        }

    staging_items = wall_info.get("loading_staging_items", [])
    loaded_pallet_items = [
        it for it in staging_items if str(it.get("type", "")).lower() == "pallet"
    ]
    empty_items = [
        it for it in staging_items if str(it.get("type", "")).lower() == "empty_pallet"
    ]
    goods_along_s = []
    for it in loaded_pallet_items:
        if "x" in it and "y" in it:
            goods_along_s.append(_along_s_from_xy(it["x"], it["y"]))
    empty_xy_unique = {}
    for it in empty_items:
        if "x" not in it or "y" not in it:
            continue
        key = (round(float(it["x"]), 3), round(float(it["y"]), 3))
        if key not in empty_xy_unique:
            empty_xy_unique[key] = (float(it["x"]), float(it["y"]))

    trucks = wall_info.get("loading_trucks", [])
    truck_alongs = sorted(float(t.get("along", 0.0)) for t in trucks)
    truck_alongs = [a for a in truck_alongs if along_lo <= a <= along_hi]
    if not truck_alongs:
        door_centers = [float(v) for v in wall_info.get("door_centers", [])]
        truck_alongs = [a for a in door_centers if along_lo <= a <= along_hi]
    if not truck_alongs:
        truck_alongs = [0.5 * (along_lo + along_hi)]

    truck_s_vals = []
    for t in trucks:
        if "x" in t and "y" in t:
            _ta, ts = _along_s_from_xy(t["x"], t["y"])
            truck_s_vals.append(ts)

    goods_s_vals = [s for _a, s in goods_along_s]
    forklift_radius = 0.5 * math.hypot(ex, ey)
    forklift_half_along = along_size * 0.5
    forklift_half_cross = cross_size * 0.5

    hard_obstacle_discs = []
    soft_obstacle_discs = []
    truck_keepout_rects = []
    truck_keepout_pad_along = max(
        0.30, float(LOADING_OPERATION_TRUCK_KEEPOUT_ALONG_PAD_M)
    )
    truck_keepout_pad_cross = max(
        0.45, float(LOADING_OPERATION_TRUCK_KEEPOUT_CROSS_PAD_M)
    )
    for t in trucks:
        tx = t.get("x")
        ty = t.get("y")
        if tx is None or ty is None:
            t_along = t.get("along")
            t_inward = t.get("inward")
            if t_along is not None and t_inward is not None:
                try:
                    tx, ty = slot_point(
                        loading_side, float(t_along), inward=float(t_inward)
                    )
                except Exception:
                    tx, ty = None, None
        if tx is None or ty is None:
            continue
        tfx, tfy = t.get("footprint_xy_m", (0.0, 0.0))
        tfx = max(0.0, float(tfx))
        tfy = max(0.0, float(tfy))
        t_along_c, t_s_c = _along_s_from_xy(float(tx), float(ty))
        if loading_side in ("north", "south"):
            truck_half_along = max(0.8, 0.5 * tfx)
            truck_half_cross = max(1.0, 0.5 * tfy)
        else:
            truck_half_along = max(0.8, 0.5 * tfy)
            truck_half_cross = max(1.0, 0.5 * tfx)
        truck_keepout_rects.append(
            (
                float(t_along_c),
                float(t_s_c),
                truck_half_along + truck_keepout_pad_along,
                truck_half_cross + truck_keepout_pad_cross,
            )
        )
        tr = max(1.40, (0.5 * min(tfx, tfy)) + 0.45)
        hard_obstacle_discs.append((float(tx), float(ty), tr + 0.25))

    for it in loaded_pallet_items:
        px = it.get("x")
        py = it.get("y")
        if px is None or py is None:
            continue
        soft_obstacle_discs.append((float(px), float(py), 1.45))

    for x, y in empty_xy_unique.values():
        soft_obstacle_discs.append((float(x), float(y), 1.25))

    for ce in wall_info.get("loading_container_entries", []):
        cx = ce.get("x")
        cy = ce.get("y")
        if cx is None or cy is None:
            continue
        hard_obstacle_discs.append((float(cx), float(cy), 4.80))

    for fk in wall_info.get("forklifts", []):
        fx = fk.get("x")
        fy = fk.get("y")
        if fx is None or fy is None:
            continue
        hard_obstacle_discs.append((float(fx), float(fy), forklift_radius + 0.15))

    truck_s_ref = (
        sum(truck_s_vals) / float(len(truck_s_vals)) if truck_s_vals else (s_hi - 0.8)
    )
    goods_s_ref = (
        sum(goods_s_vals) / float(len(goods_s_vals))
        if goods_s_vals
        else (s_lo + (0.46 * (s_hi - s_lo)))
    )
    if goods_s_ref > truck_s_ref:
        goods_s_ref, truck_s_ref = truck_s_ref, goods_s_ref

    truck_offset = max(1.0, float(LOADING_OPERATION_FORKLIFT_TRUCK_OFFSET_M))
    goods_offset = max(1.0, float(LOADING_OPERATION_FORKLIFT_EMPTY_OFFSET_M))
    truck_oper_s = _clamp(truck_s_ref - truck_offset, s_lo, s_hi)
    goods_oper_s = _clamp(goods_s_ref + goods_offset, s_lo, s_hi)
    if truck_oper_s <= goods_oper_s:
        corridor_s_mid = _clamp(0.5 * (truck_s_ref + goods_s_ref), s_lo, s_hi)
    else:
        corridor_s_mid = _clamp(0.5 * (truck_oper_s + goods_oper_s), s_lo, s_hi)

    target_count = max(1, int(LOADING_OPERATION_FORKLIFT_TARGET_COUNT))
    rng = random.Random(int(seed) + 17233)
    corridor_band_half = max(0.85, (s_hi - s_lo) * 0.18)
    corridor_s_lo = _clamp(corridor_s_mid - corridor_band_half, s_lo, s_hi)
    corridor_s_hi = _clamp(corridor_s_mid + corridor_band_half, s_lo, s_hi)
    if corridor_s_hi <= (corridor_s_lo + 0.15):
        corridor_s_lo = s_lo
        corridor_s_hi = s_hi

    along_seeds = list(truck_alongs)
    if len(along_seeds) >= target_count:
        along_seeds = rng.sample(along_seeds, target_count)
    else:
        while len(along_seeds) < target_count:
            along_seeds.append(rng.uniform(along_lo, along_hi))

    target_points = []
    for idx in range(target_count):
        a_seed = float(along_seeds[idx]) + rng.uniform(-2.2, 2.2)
        s_seed = rng.uniform(corridor_s_lo, corridor_s_hi)
        target_points.append(
            (
                f"corridor_random_{idx}",
                _clamp(a_seed, along_lo, along_hi),
                _clamp(s_seed, s_lo, s_hi),
            )
        )
    rng.shuffle(target_points)

    forklifts = []
    occupied_xy = []
    min_center_dist = max(2.2, forklift_radius * 1.30)
    hard_obstacle_clearance = max(0.22, forklift_radius * 0.22)
    soft_obstacle_clearance = 0.0

    def _try_place(along_base, s_base):
        ds_candidates = [0.0, -0.45, 0.45, -0.90, 0.90, -1.40, 1.40, -2.00, 2.00]
        da_candidates = [0.0, -1.6, 1.6, -3.2, 3.2, -4.8, 4.8, -6.4, 6.4]
        rng.shuffle(ds_candidates)
        rng.shuffle(da_candidates)
        for ds in ds_candidates:
            for da in da_candidates:
                along = _clamp(along_base + da, along_lo, along_hi)
                s_from_interior = _clamp(s_base + ds, s_lo, s_hi)
                x, y = _xy_from_along_s(along, s_from_interior)
                ok = True
                for ox, oy in occupied_xy:
                    if ((x - ox) ** 2 + (y - oy) ** 2) < (min_center_dist**2):
                        ok = False
                        break
                if not ok:
                    continue
                for ta, ts, half_a, half_s in truck_keepout_rects:
                    if abs(along - ta) <= (half_a + forklift_half_along) and abs(
                        s_from_interior - ts
                    ) <= (half_s + forklift_half_cross):
                        ok = False
                        break
                if not ok:
                    continue
                for ox, oy, orad in hard_obstacle_discs:
                    lim = orad + forklift_radius + hard_obstacle_clearance
                    if ((x - ox) ** 2 + (y - oy) ** 2) < (lim**2):
                        ok = False
                        break
                if not ok:
                    continue
                for ox, oy, orad in soft_obstacle_discs:
                    lim = orad + forklift_radius + soft_obstacle_clearance
                    if ((x - ox) ** 2 + (y - oy) ** 2) < (lim**2):
                        ok = False
                        break
                if ok:
                    return along, s_from_interior, x, y
        return None

    for idx, (tag, along_raw, s_raw) in enumerate(target_points):
        along_raw += rng.uniform(-0.80, 0.80)
        s_raw += rng.uniform(-0.65, 0.65)
        picked = _try_place(along_raw, s_raw)
        if picked is None:
            picked = _try_place(along_raw, _clamp(corridor_s_mid - 0.85, s_lo, s_hi))
        if picked is None:
            picked = _try_place(along_raw, _clamp(corridor_s_mid + 0.85, s_lo, s_hi))
        if picked is None:
            continue
        along, s_from_interior, x, y = picked
        _spawn_obj_with_mtl_parts(
            loader=forklift_loader,
            model_name=model_name,
            world_anchor_xyz=(x, y, floor_top_z),
            yaw_deg=yaw_deg,
            mesh_scale_xyz=scale_xyz,
            local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
            cli=cli,
            with_collision=True,
            fallback_rgba=(0.86, 0.86, 0.86, 1.0),
        )
        occupied_xy.append((x, y))
        forklifts.append(
            {
                "model": model_name,
                "x": x,
                "y": y,
                "along": along,
                "s_from_interior": s_from_interior,
                "yaw_deg": yaw_deg,
                "loading_side": loading_side,
                "role": tag,
                "index": idx,
                "footprint_xy_m": (ex, ey),
                "size_xyz_m": (size_x, size_y, size_z),
            }
        )

    target_count = max(1, int(LOADING_OPERATION_FORKLIFT_TARGET_COUNT))
    if len(forklifts) < target_count:
        open_s_lo = _clamp(min(goods_oper_s, corridor_s_mid), s_lo, s_hi)
        open_s_hi = _clamp(max(truck_oper_s, corridor_s_mid), s_lo, s_hi)
        if open_s_hi <= open_s_lo:
            open_s_lo, open_s_hi = s_lo, s_hi
        for _ in range(220):
            if len(forklifts) >= target_count:
                break
            rand_along = rng.uniform(along_lo, along_hi)
            rand_s = rng.uniform(open_s_lo, open_s_hi)
            picked = _try_place(rand_along, rand_s)
            if picked is None:
                continue
            along, s_from_interior, x, y = picked
            _spawn_obj_with_mtl_parts(
                loader=forklift_loader,
                model_name=model_name,
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=scale_xyz,
                local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                cli=cli,
                with_collision=True,
                fallback_rgba=(0.86, 0.86, 0.86, 1.0),
            )
            occupied_xy.append((x, y))
            forklifts.append(
                {
                    "model": model_name,
                    "x": x,
                    "y": y,
                    "along": along,
                    "s_from_interior": s_from_interior,
                    "yaw_deg": yaw_deg,
                    "loading_side": loading_side,
                    "role": "fill_open_space",
                    "index": len(forklifts),
                    "footprint_xy_m": (ex, ey),
                    "size_xyz_m": (size_x, size_y, size_z),
                }
            )

    return {
        "loading_operation_forklift_count": len(forklifts),
        "loading_operation_forklifts": forklifts,
    }


def build_worker_crew(
    worker_loader, worker_model_name, floor_top_z, area_layout, wall_info, cli, seed=0
):
    if not ENABLE_WORKER_CREW:
        return {"worker_count": 0, "workers": []}
    if worker_loader is None or not worker_model_name:
        return {
            "worker_count": 0,
            "workers": [],
            "worker_reason": "Worker model not found in configured asset paths.",
        }

    try:
        raw_min_v, raw_max_v = model_bounds_xyz(
            worker_loader, worker_model_name, (1.0, 1.0, 1.0)
        )
    except (FileNotFoundError, ValueError) as exc:
        return {
            "worker_count": 0,
            "workers": [],
            "worker_reason": f"Failed to load worker model: {exc}",
        }

    raw_height = max(1e-6, float(raw_max_v[2] - raw_min_v[2]))
    scale_uniform = max(0.25, min(3.0, float(WORKER_TARGET_HEIGHT_M) / raw_height))
    scale_xyz = (scale_uniform, scale_uniform, scale_uniform)
    min_v, max_v = model_bounds_xyz(worker_loader, worker_model_name, scale_xyz)
    size_x = float(max_v[0] - min_v[0])
    size_y = float(max_v[1] - min_v[1])
    size_z = float(max_v[2] - min_v[2])
    anchor_x = float((min_v[0] + max_v[0]) * 0.5)
    anchor_y = float((min_v[1] + max_v[1]) * 0.5)
    anchor_z = float(min_v[2])

    spacing_min = max(float(WORKER_MIN_SPACING_M), 0.6 * max(size_x, size_y))
    worker_radius = 0.5 * math.hypot(size_x, size_y)

    obstacle_discs = []

    def _add_obstacle_xy(x, y, radius):
        if x is None or y is None:
            return
        obstacle_discs.append((float(x), float(y), float(max(0.2, radius))))

    for t in wall_info.get("loading_trucks", []):
        fx, fy = t.get("footprint_xy_m", (0.0, 0.0))
        _add_obstacle_xy(
            t.get("x"), t.get("y"), 0.5 * math.hypot(float(fx), float(fy)) + 0.2
        )
    for fk in wall_info.get("forklifts", []):
        ffx, ffy = fk.get("footprint_xy_m", (size_x, size_y))
        _add_obstacle_xy(
            fk.get("x"), fk.get("y"), 0.5 * math.hypot(float(ffx), float(ffy)) + 0.2
        )
    for fk in wall_info.get("loading_operation_forklifts", []):
        ffx, ffy = fk.get("footprint_xy_m", (size_x, size_y))
        _add_obstacle_xy(
            fk.get("x"), fk.get("y"), 0.5 * math.hypot(float(ffx), float(ffy)) + 0.2
        )
    for sr in wall_info.get("storage_racks", []):
        rfx, rfy = sr.get("footprint_xy_m", (0.0, 0.0))
        _add_obstacle_xy(
            sr.get("x"), sr.get("y"), 0.5 * math.hypot(float(rfx), float(rfy)) + 0.25
        )
    for it in wall_info.get("loading_staging_items", []):
        itype = str(it.get("type", "")).lower()
        if itype in ("pallet", "empty_pallet", "barrel", "box"):
            _add_obstacle_xy(
                it.get("x"), it.get("y"), 0.85 if "pallet" in itype else 0.55
            )
    for ce in wall_info.get("loading_container_entries", []):
        _add_obstacle_xy(ce.get("x"), ce.get("y"), 2.2)

    def _zone_bounds(name):
        area = (area_layout or {}).get(name)
        if not area:
            return None
        cx = float(area["cx"])
        cy = float(area["cy"])
        sx = float(area["sx"])
        sy = float(area["sy"])
        return (cx - (sx * 0.5), cx + (sx * 0.5), cy - (sy * 0.5), cy + (sy * 0.5))

    zones_quota = (
        ("LOADING", 3),
        ("STORAGE", 3),
    )
    rng = random.Random(int(seed) + 19441)
    picked_workers = []
    picked_xy = []
    candidate_pool = []

    for zone_name, zone_quota in zones_quota:
        bounds = _zone_bounds(zone_name)
        if bounds is None:
            continue
        x0, x1, y0, y1 = bounds
        margin = max(0.7, worker_radius + 0.30)
        if (x1 - x0) <= (2.0 * margin) or (y1 - y0) <= (2.0 * margin):
            continue
        base_offsets = (
            (-0.30, -0.20),
            (0.30, -0.18),
            (-0.18, 0.20),
            (0.20, 0.22),
            (0.0, 0.0),
        )
        for ox, oy in base_offsets:
            bx = 0.5 * (x0 + x1) + ox * (x1 - x0)
            by = 0.5 * (y0 + y1) + oy * (y1 - y0)
            candidate_pool.append(
                {
                    "zone": zone_name,
                    "quota_weight": zone_quota,
                    "x": bx + rng.uniform(-0.40, 0.40),
                    "y": by + rng.uniform(-0.40, 0.40),
                }
            )
        grid_cols = 5 if zone_name == "STORAGE" else 4
        grid_rows = 3
        gx0 = x0 + margin
        gx1 = x1 - margin
        gy0 = y0 + margin
        gy1 = y1 - margin
        for gy in range(grid_rows):
            for gx in range(grid_cols):
                tx = (gx + 0.5) / float(grid_cols)
                ty = (gy + 0.5) / float(grid_rows)
                base_x = gx0 + (tx * max(0.0, gx1 - gx0))
                base_y = gy0 + (ty * max(0.0, gy1 - gy0))
                candidate_pool.append(
                    {
                        "zone": zone_name,
                        "quota_weight": zone_quota,
                        "x": base_x + rng.uniform(-0.32, 0.32),
                        "y": base_y + rng.uniform(-0.32, 0.32),
                    }
                )
        for _ in range(18):
            candidate_pool.append(
                {
                    "zone": zone_name,
                    "quota_weight": zone_quota,
                    "x": rng.uniform(x0 + margin, x1 - margin),
                    "y": rng.uniform(y0 + margin, y1 - margin),
                }
            )

    target_count = max(1, int(WORKER_TARGET_COUNT))
    quota_left = {name: int(q) for name, q in zones_quota}
    rng.shuffle(candidate_pool)

    def _valid_xy(x, y, obstacle_pad=0.25, spacing_rule=None):
        spacing = float(spacing_min if spacing_rule is None else spacing_rule)
        for px, py in picked_xy:
            if ((x - px) ** 2 + (y - py) ** 2) < (spacing**2):
                return False
        for ox, oy, orad in obstacle_discs:
            lim = orad + worker_radius + float(obstacle_pad)
            if ((x - ox) ** 2 + (y - oy) ** 2) < (lim**2):
                return False
        return True

    attempt_settings = (
        (0.25, spacing_min),
        (0.14, max(1.0, spacing_min * 0.88)),
        (0.06, max(0.85, spacing_min * 0.78)),
    )

    for obstacle_pad, spacing_rule in attempt_settings:
        for cand in candidate_pool:
            if len(picked_workers) >= target_count:
                break
            zname = cand["zone"]
            if quota_left.get(zname, 0) <= 0:
                continue
            x = float(cand["x"])
            y = float(cand["y"])
            if not _valid_xy(
                x, y, obstacle_pad=obstacle_pad, spacing_rule=spacing_rule
            ):
                continue
            yaw_deg = rng.uniform(0.0, 360.0)
            picked_workers.append({"x": x, "y": y, "yaw_deg": yaw_deg, "zone": zname})
            picked_xy.append((x, y))
            quota_left[zname] = max(0, quota_left.get(zname, 0) - 1)
        if len(picked_workers) >= target_count:
            break

    if len(picked_workers) < target_count:
        for obstacle_pad, spacing_rule in attempt_settings:
            for cand in candidate_pool:
                if len(picked_workers) >= target_count:
                    break
                x = float(cand["x"])
                y = float(cand["y"])
                if not _valid_xy(
                    x, y, obstacle_pad=obstacle_pad, spacing_rule=spacing_rule
                ):
                    continue
                yaw_deg = rng.uniform(0.0, 360.0)
                picked_workers.append(
                    {"x": x, "y": y, "yaw_deg": yaw_deg, "zone": cand["zone"]}
                )
                picked_xy.append((x, y))
            if len(picked_workers) >= target_count:
                break

    workers = []
    for i, item in enumerate(picked_workers):
        x = float(item["x"])
        y = float(item["y"])
        yaw_deg = float(item["yaw_deg"])
        _spawn_obj_with_mtl_parts(
            loader=worker_loader,
            model_name=worker_model_name,
            world_anchor_xyz=(x, y, floor_top_z),
            yaw_deg=yaw_deg,
            mesh_scale_xyz=scale_xyz,
            local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
            cli=cli,
            with_collision=False,
            fallback_rgba=(0.65, 0.65, 0.68, 1.0),
            rgba_gain=WORKER_COLOR_GAIN,
        )
        workers.append(
            {
                "index": i,
                "model": worker_model_name,
                "x": x,
                "y": y,
                "yaw_deg": yaw_deg,
                "zone": item.get("zone", ""),
                "size_xyz_m": (size_x, size_y, size_z),
                "scale_uniform": scale_uniform,
            }
        )

    return {
        "worker_count": len(workers),
        "worker_model": worker_model_name,
        "worker_scale_uniform": scale_uniform,
        "workers": workers,
    }


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
            _spawn_collision_only_with_anchor(
                loader=industry_loader,
                model_name=spec["model_name"],
                world_anchor_xyz=(x, y, floor_top_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=spec["scale_xyz"],
                local_anchor_xyz=(spec["anchor_x"], spec["anchor_y"], spec["anchor_z"]),
                cli=cli,
                model_path_override=spec.get("collision_path", ""),
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
