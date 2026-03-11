from ._shared import *
from .support import _forklift_yaw_back_to_wall


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
