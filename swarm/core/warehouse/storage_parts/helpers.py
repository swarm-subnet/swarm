from types import SimpleNamespace

from ._shared import *


def make_storage_layout_helpers(
    *,
    storage_loader,
    rack_model,
    rack_scale,
    pallet_model,
    pallet_scale,
    pallet_size_x,
    pallet_size_y,
    box_model,
    box_scale,
    box_size_x,
    box_size_y,
    barrel_model,
    barrel_scale,
    barrel_size_x,
    barrel_size_y,
    oriented_xy_local_cache,
    barrel_layout_profile_cache,
    box_layout_profile_cache,
):
    def _yaw_key(yaw_deg):
        return round(float(yaw_deg) % 360.0, 6)

    def _oriented_xy_cached(model_name, scale_xyz, yaw_deg):
        key = (str(model_name), tuple(float(v) for v in scale_xyz), _yaw_key(yaw_deg))
        cached = oriented_xy_local_cache.get(key)
        if cached is not None:
            return cached
        out = oriented_xy_size(storage_loader, model_name, scale_xyz, key[2])
        oriented_xy_local_cache[key] = out
        return out

    def _barrel_layout_profile_for_slot_yaw(slot_yaw):
        key = _yaw_key(slot_yaw)
        cached = barrel_layout_profile_cache.get(key)
        if cached is not None:
            return cached

        barrel_edge_margin = 0.01
        target_gap = 0.04
        barrel_layout_candidates = []
        for swap_axes in (False, True):
            if swap_axes:
                barrel_local_x = barrel_size_y
                barrel_local_y = barrel_size_x
                barrel_yaw = (key + 90.0) % 360.0
            else:
                barrel_local_x = barrel_size_x
                barrel_local_y = barrel_size_y
                barrel_yaw = key
            max_gap_x = (
                pallet_size_x - (2.0 * barrel_local_x) - (2.0 * barrel_edge_margin)
            )
            max_gap_y = (
                pallet_size_y - (2.0 * barrel_local_y) - (2.0 * barrel_edge_margin)
            )
            if max_gap_x < -1e-6 or max_gap_y < -1e-6:
                continue
            use_gap = max(0.0, min(target_gap, max_gap_x, max_gap_y))
            barrel_layout_candidates.append(
                (use_gap, barrel_yaw, barrel_local_x, barrel_local_y)
            )

        if barrel_layout_candidates:
            barrel_layout_candidates.sort(key=lambda t: float(t[0]), reverse=True)
            (
                use_gap,
                barrel_yaw,
                barrel_local_x,
                barrel_local_y,
            ) = barrel_layout_candidates[0]
            off_x = (barrel_local_x * 0.5) + (use_gap * 0.5)
            off_y = (barrel_local_y * 0.5) + (use_gap * 0.5)
            off_x_max = max(
                0.0, (pallet_size_x * 0.5) - (barrel_local_x * 0.5) - barrel_edge_margin
            )
            off_y_max = max(
                0.0, (pallet_size_y * 0.5) - (barrel_local_y * 0.5) - barrel_edge_margin
            )
            off_x = min(off_x, off_x_max)
            off_y = min(off_y, off_y_max)
            layer1_slots = [
                (-off_x, -off_y),
                (off_x, -off_y),
                (-off_x, off_y),
                (off_x, off_y),
            ]
        else:
            barrel_yaw = key
            barrel_local_x = barrel_size_x
            off_x_max = max(
                0.0, (pallet_size_x * 0.5) - (barrel_local_x * 0.5) - barrel_edge_margin
            )
            off_x = min(off_x_max, max(0.08, barrel_local_x * 0.5))
            layer1_slots = [(-off_x, 0.0), (off_x, 0.0)]

        bex, bey = _oriented_xy_cached(barrel_model, barrel_scale, barrel_yaw)
        out = {
            "barrel_yaw": float(barrel_yaw),
            "layer1_slots": tuple((float(x), float(y)) for x, y in layer1_slots),
            "bex": float(bex),
            "bey": float(bey),
        }
        barrel_layout_profile_cache[key] = out
        return out

    def _box_layout_profile_for_slot_yaw(slot_yaw):
        key = _yaw_key(slot_yaw)
        cached = box_layout_profile_cache.get(key)
        if cached is not None:
            return cached

        box_edge_margin = 0.01
        target_gap = 0.03
        layout_candidates = []
        for swap_axes in (False, True):
            if swap_axes:
                box_local_x = box_size_y
                box_local_y = box_size_x
                box_yaw = (key + 90.0) % 360.0
            else:
                box_local_x = box_size_x
                box_local_y = box_size_y
                box_yaw = key
            max_gap_x = pallet_size_x - (2.0 * box_local_x) - (2.0 * box_edge_margin)
            max_gap_y = pallet_size_y - (2.0 * box_local_y) - (2.0 * box_edge_margin)
            if max_gap_x < -1e-6 or max_gap_y < -1e-6:
                continue
            use_gap = max(0.0, min(target_gap, max_gap_x, max_gap_y))
            layout_candidates.append((use_gap, box_yaw, box_local_x, box_local_y))

        if layout_candidates:
            layout_candidates.sort(key=lambda t: float(t[0]), reverse=True)
            use_gap, box_yaw, box_local_x, box_local_y = layout_candidates[0]
            off_x = (box_local_x * 0.5) + (use_gap * 0.5)
            off_y = (box_local_y * 0.5) + (use_gap * 0.5)
            off_x_max = max(
                0.0, (pallet_size_x * 0.5) - (box_local_x * 0.5) - box_edge_margin
            )
            off_y_max = max(
                0.0, (pallet_size_y * 0.5) - (box_local_y * 0.5) - box_edge_margin
            )
            off_x = min(off_x, off_x_max)
            off_y = min(off_y, off_y_max)
            layer1_slots = [
                (-off_x, -off_y),
                (off_x, -off_y),
                (-off_x, off_y),
                (off_x, off_y),
            ]
        else:
            box_yaw = key
            box_local_x = box_size_x
            off_x_max = max(
                0.0, (pallet_size_x * 0.5) - (box_local_x * 0.5) - box_edge_margin
            )
            off_x = min(off_x_max, max(0.08, box_local_x * 0.5))
            layer1_slots = [(-off_x, 0.0), (off_x, 0.0)]

        bex, bey = _oriented_xy_cached(box_model, box_scale, box_yaw)
        out = {
            "box_yaw": float(box_yaw),
            "layer1_slots": tuple((float(x), float(y)) for x, y in layer1_slots),
            "bex": float(bex),
            "bey": float(bey),
        }
        box_layout_profile_cache[key] = out
        return out

    def _packed_centers(lo, hi, size, gap):
        span = float(hi) - float(lo)
        if span < (float(size) - 1e-6):
            return []
        step = float(size) + max(0.0, float(gap))
        count = max(1, int(math.floor((span + max(0.0, float(gap))) / step)))
        used = (count * float(size)) + ((count - 1) * max(0.0, float(gap)))
        slack = max(0.0, span - used)
        start = float(lo) + (slack * 0.5) + (float(size) * 0.5)
        return [start + (i * step) for i in range(count)]

    return SimpleNamespace(
        yaw_key=_yaw_key,
        oriented_xy_cached=_oriented_xy_cached,
        barrel_layout_profile_for_slot_yaw=_barrel_layout_profile_for_slot_yaw,
        box_layout_profile_for_slot_yaw=_box_layout_profile_for_slot_yaw,
        packed_centers=_packed_centers,
    )


def make_storage_support_helpers(
    *,
    along_axis,
    storage_loader,
    rack_model,
    rack_scale,
    rack_size_z,
    rng,
):
    def _to_world_xy(along_v, cross_v):
        if along_axis == "x":
            return float(along_v), float(cross_v)
        return float(cross_v), float(along_v)

    def _cluster_level_area(area_map, merge_eps=0.03):
        points = sorted(
            (float(z), float(a)) for z, a in area_map.items() if float(a) > 1e-8
        )
        if not points:
            return []
        clusters = []
        for z, area in points:
            if not clusters or abs(z - clusters[-1]["z_avg"]) > float(merge_eps):
                clusters.append({"z_avg": z, "area": area, "z_area_sum": z * area})
            else:
                cl = clusters[-1]
                cl["area"] += area
                cl["z_area_sum"] += z * area
                cl["z_avg"] = cl["z_area_sum"] / max(1e-8, cl["area"])
        return [{"z": float(cl["z_avg"]), "area": float(cl["area"])} for cl in clusters]

    def _rack_support_surface_levels_m():
        model_path = os.path.join(str(storage_loader.obj_dir), str(rack_model))
        if not os.path.exists(model_path):
            return []
        try:
            st = os.stat(model_path)
            model_sig = (
                int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1_000_000_000))),
                int(st.st_size),
            )
        except OSError:
            model_sig = (-1, -1)
        cache_key = (
            os.path.abspath(model_path).replace("\\", "/"),
            tuple(round(float(v), 8) for v in rack_scale),
            model_sig,
        )
        cached = _STORAGE_RACK_SUPPORT_LEVELS_CACHE.get(cache_key)
        if cached is not None:
            return [float(v) for v in cached]

        raw_verts = []
        face_tokens = []
        try:
            with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if line.startswith("v "):
                        _, xs, ys, zs = line.split()[:4]
                        raw_verts.append((float(xs), float(ys), float(zs)))
                    elif line.startswith("f "):
                        toks = line.strip().split()[1:]
                        if len(toks) >= 3:
                            face_tokens.append(toks)
        except OSError:
            return []

        if not raw_verts or not face_tokens:
            return []

        sx, sy, sz = float(rack_scale[0]), float(rack_scale[1]), float(rack_scale[2])
        verts = [(x * sx, (-z) * sy, y * sz) for x, y, z in raw_verts]
        min_model_z = min(v[2] for v in verts)

        top_area = {}
        bottom_area = {}
        vcount = len(verts)
        for fpoly in face_tokens:
            idx = []
            for tok in fpoly:
                vtxt = tok.split("/")[0]
                if not vtxt:
                    continue
                vi = int(vtxt)
                if vi < 0:
                    vi = vcount + 1 + vi
                vi = vi - 1
                if vi < 0 or vi >= vcount:
                    idx = []
                    break
                idx.append(vi)
            if len(idx) < 3:
                continue

            v0 = verts[idx[0]]
            for k in range(1, len(idx) - 1):
                v1 = verts[idx[k]]
                v2 = verts[idx[k + 1]]
                ax = float(v1[0] - v0[0])
                ay = float(v1[1] - v0[1])
                az = float(v1[2] - v0[2])
                bx = float(v2[0] - v0[0])
                by = float(v2[1] - v0[1])
                bz = float(v2[2] - v0[2])
                nx = (ay * bz) - (az * by)
                ny = (az * bx) - (ax * bz)
                nz = (ax * by) - (ay * bx)
                nlen = math.sqrt((nx * nx) + (ny * ny) + (nz * nz))
                if nlen <= 1e-9:
                    continue
                nz_u = nz / nlen
                if abs(nz_u) < 0.92:
                    continue
                tri_area = 0.5 * nlen
                z_centroid = (float(v0[2]) + float(v1[2]) + float(v2[2])) / 3.0
                z_key = round(z_centroid, 3)
                if nz_u > 0.0:
                    top_area[z_key] = float(top_area.get(z_key, 0.0)) + tri_area
                else:
                    bottom_area[z_key] = float(bottom_area.get(z_key, 0.0)) + tri_area

        if not top_area or not bottom_area:
            return []

        top_levels = _cluster_level_area(top_area, merge_eps=0.025)
        bottom_levels = _cluster_level_area(bottom_area, merge_eps=0.025)
        if not top_levels or not bottom_levels:
            return []

        max_top_area = max(float(v["area"]) for v in top_levels)
        max_bottom_area = max(float(v["area"]) for v in bottom_levels)
        top_levels = [
            v for v in top_levels if float(v["area"]) >= max(0.03, max_top_area * 0.12)
        ]
        bottom_levels = [
            v
            for v in bottom_levels
            if float(v["area"]) >= max(0.03, max_bottom_area * 0.12)
        ]
        if not top_levels or not bottom_levels:
            return []

        support_rel_levels = []
        for top in sorted(top_levels, key=lambda d: float(d["z"])):
            top_z = float(top["z"])
            best_delta = None
            for bottom in bottom_levels:
                bot_z = float(bottom["z"])
                if bot_z >= top_z:
                    continue
                dz = top_z - bot_z
                if dz < 0.04 or dz > 0.35:
                    continue
                if best_delta is None or dz < best_delta:
                    best_delta = dz
            if best_delta is None:
                continue
            rel = float(top_z - min_model_z)
            if rel <= 0.06:
                continue
            if rel >= (rack_size_z + 0.05):
                continue
            support_rel_levels.append(rel)

        if not support_rel_levels:
            return []
        support_rel_levels = sorted(float(round(v, 4)) for v in support_rel_levels)
        dedup = []
        for z in support_rel_levels:
            if not dedup or (z - dedup[-1]) > 0.18:
                dedup.append(z)
            elif z > dedup[-1]:
                dedup[-1] = z
        _STORAGE_RACK_SUPPORT_LEVELS_CACHE[cache_key] = tuple(float(v) for v in dedup)
        return [float(v) for v in dedup]

    def _level_slot_count(slot_total, level_density):
        if slot_total <= 1:
            return 1
        target = int(round(float(slot_total) * float(level_density)))
        target = max(1, min(int(slot_total), target))
        low = target if float(level_density) >= 0.95 else max(1, target - 1)
        high = max(low, target)
        return rng.randint(low, high)

    return SimpleNamespace(
        to_world_xy=_to_world_xy,
        cluster_level_area=_cluster_level_area,
        rack_support_surface_levels_m=_rack_support_surface_levels_m,
        level_slot_count=_level_slot_count,
    )


def storage_plan_score(plan, primary_along_axis):
    return (
        1 if bool(plan.get("is_long_along", False)) else 0,
        1 if str(plan.get("along_axis", "")) == str(primary_along_axis) else 0,
        int(plan.get("row_count", 0)),
        int(plan.get("max_cols", 0)),
        int(plan.get("slot_count", 0)),
    )


def append_storage_endcaps(
    selected,
    *,
    endcap_enabled,
    left_endcap_center,
    right_endcap_center,
    cross_min_bound,
    cross_max_bound,
    endcap_cross_size,
    endcap_rack_yaw,
    target_rows,
    along_axis,
    _packed_centers,
):
    if not (
        endcap_enabled
        and (left_endcap_center is not None)
        and (right_endcap_center is not None)
    ):
        return

    def _to_world_xy(along_v, cross_v):
        if along_axis == "x":
            return float(along_v), float(cross_v)
        return float(cross_v), float(along_v)

    endcap_cross_lo = cross_min_bound + (endcap_cross_size * 0.5)
    endcap_cross_hi = cross_max_bound - (endcap_cross_size * 0.5)
    endcap_slot_gap = 0.0
    endcap_cross_centers = (
        _packed_centers(
            endcap_cross_lo, endcap_cross_hi, endcap_cross_size, endcap_slot_gap
        )
        if endcap_cross_hi >= endcap_cross_lo
        else []
    )
    if len(endcap_cross_centers) > 1:
        ec_start = float(endcap_cross_lo)
        ec_end = float(endcap_cross_hi)
        ec_step = (ec_end - ec_start) / float(len(endcap_cross_centers) - 1)
        endcap_cross_centers = [
            ec_start + (i * ec_step) for i in range(len(endcap_cross_centers))
        ]
    for ec_idx, cross_v in enumerate(endcap_cross_centers):
        for bank_idx, along_v in (
            (0, left_endcap_center),
            (1, right_endcap_center),
        ):
            sx, sy = _to_world_xy(float(along_v), float(cross_v))
            selected.append(
                {
                    "x": sx,
                    "y": sy,
                    "row": int(target_rows + ec_idx),
                    "col": 0,
                    "bank": int(bank_idx),
                    "along": float(along_v),
                    "yaw_deg": float(endcap_rack_yaw),
                    "kind": "endcap",
                }
            )


def rotate_selected_slots(selected, *, group_rotate_deg, area_cx, area_cy):
    if not selected or abs(group_rotate_deg) <= 1e-6:
        return
    rot_rad = math.radians(group_rotate_deg)
    cos_r = math.cos(rot_rad)
    sin_r = math.sin(rot_rad)
    cx = float(area_cx)
    cy = float(area_cy)
    for slot in selected:
        dx = float(slot["x"]) - cx
        dy = float(slot["y"]) - cy
        slot["x"] = cx + (dx * cos_r) - (dy * sin_r)
        slot["y"] = cy + (dx * sin_r) + (dy * cos_r)
        slot["yaw_deg"] = (
            float(slot.get("yaw_deg", 0.0)) + group_rotate_deg
        ) % 360.0


def pick_barrel_slot_keys(selected_rows, selected, *, barrel_prob, rng):
    barrel_slot_keys = set()
    barrel_phase = rng.randint(0, 2)
    for row_key in sorted(
        selected_rows.keys(), key=lambda rk: (int(rk[1]), int(rk[0]))
    ):
        row_slots = selected_rows.get(row_key, [])
        if not row_slots:
            continue
        row_idx, bank_idx = int(row_key[0]), int(row_key[1])
        row_pref_barrel = ((row_idx + bank_idx + barrel_phase) % 3) == 0
        row_prob = barrel_prob * (1.45 if row_pref_barrel else 0.35)
        row_prob = max(0.0, min(0.90, row_prob))
        row_has_barrel = False
        for slot in row_slots:
            key = (
                int(slot.get("row", 0)),
                int(slot.get("bank", 0)),
                int(slot.get("col", 0)),
            )
            if rng.random() < row_prob:
                barrel_slot_keys.add(key)
                row_has_barrel = True
        if row_pref_barrel and not row_has_barrel and row_slots and rng.random() < 0.78:
            mid_slot = row_slots[len(row_slots) // 2]
            mid_key = (
                int(mid_slot.get("row", 0)),
                int(mid_slot.get("bank", 0)),
                int(mid_slot.get("col", 0)),
            )
            barrel_slot_keys.add(mid_key)

    if not barrel_slot_keys and selected:
        mid_slot = selected[len(selected) // 2]
        barrel_slot_keys.add(
            (
                int(mid_slot.get("row", 0)),
                int(mid_slot.get("bank", 0)),
                int(mid_slot.get("col", 0)),
            )
        )
    return barrel_slot_keys
