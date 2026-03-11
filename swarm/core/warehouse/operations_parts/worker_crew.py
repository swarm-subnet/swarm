from ._shared import *


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
