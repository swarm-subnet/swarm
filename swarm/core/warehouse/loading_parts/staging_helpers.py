from types import SimpleNamespace

from ._shared import *
from .visuals import _spawn_obj_with_mtl_parts


def make_build_spec_helper(loading_loader):
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

    return _build_spec


def make_staging_layout_helpers(*, along_axis, interior_edge, cross_to_dock_sign, seg_gap):
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

    return SimpleNamespace(
        xy_from_along_s=_xy_from_along_s,
        oriented_xy=_oriented_xy,
        range_len=_range_len,
        clamp=_clamp,
        valid_range=_valid_range,
        split_outer_range=_split_outer_range,
    )


def make_staging_spawn_helpers(*, loading_loader, cli, container_spec):
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
            _spawn_mesh_with_anchor(
                loader=loading_loader,
                model_name=container_spec["model_name"],
                world_anchor_xyz=(x, y, z_anchor),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=container_spec["scale_xyz"],
                local_anchor_xyz=container_spec["anchor_xyz"],
                cli=cli,
                with_collision=True,
                use_texture=False,
                rgba=(0.70, 0.70, 0.70, 1.0),
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

    return SimpleNamespace(
        spawn_prop=_spawn_prop,
        spawn_container=_spawn_container,
    )


def make_staging_cargo_helpers(
    *,
    floor_top_z,
    pallet_spec,
    box_spec,
    barrel_spec,
    p_along,
    p_cross,
    _oriented_xy,
    _spawn_prop,
    _spawn_container,
    spawned_items,
    container_entries,
    counts,
    container_state,
    along_axis,
    cross_depth_total,
    yaw_along,
    container_dir,
    container_gate_edge_along,
    _xy_from_along_s,
    _range_len,
    _clamp,
):
    def _spawn_loaded_pallet_with_boxes(px, py, yaw_deg, stack_layers=1):
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
            counts["pallet"] += 1
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
                counts["box"] += 1
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
            counts["pallet"] += 1
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
                counts["barrel"] += 1
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

    def _place_container_stack_in_range(target_range):
        if container_state["spec"] is None or target_range is None:
            return False
        have_along = _range_len(target_range)
        if have_along <= 0.1:
            return False

        yaw_candidates = ((yaw_along + 90.0) % 360.0, yaw_along % 360.0)
        c_size_x = float(container_state["spec"]["size_xyz"][0])
        c_size_y = float(container_state["spec"]["size_xyz"][1])
        c_size_z = float(container_state["spec"]["size_xyz"][2])

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

        before = int(counts["container"])
        for along in base_alongs:
            cx, cy = _xy_from_along_s(along, row_cross)
            _spawn_container(
                cx, cy, floor_top_z, container_yaw, with_collision=True, body_rgba=None
            )
            counts["container"] += 1
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
                counts["container"] += 1
                container_entries.append({"x": cx, "y": cy, "level": 1})

        placed_now = int(counts["container"]) - before
        if placed_now < total_target and not container_state["reason"]:
            container_state["reason"] = (
                f"Container stack fallback: placed {placed_now}/{total_target} in LOADING section."
            )
        return placed_now > 0

    return SimpleNamespace(
        spawn_loaded_pallet_with_boxes=_spawn_loaded_pallet_with_boxes,
        spawn_barrel_pallet=_spawn_barrel_pallet,
        place_container_stack_in_range=_place_container_stack_in_range,
    )
