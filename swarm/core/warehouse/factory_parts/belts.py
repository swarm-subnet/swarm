from ._shared import *
from .pathing import (
    _dir_to_yaw,
    _generate_step1_path_seeded,
    _spawn_swarm_drone_urdf,
    list_existing_models,
    pick_end_cap_model,
    pick_first_existing_model,
    pick_section_belt_models,
    pick_support_model,
    resolve_swarm_drone_urdf,
    select_support_for_target_height,
    support_scale_for_top_alignment,
)


def build_single_belt_network(
    loader, seed, cli, center_xy=(0.0, 0.0), size_xy=None, floor_z=0.0
):
    belt_rng = random.Random(int(seed) + 1701)
    (
        base_belt_model,
        base_belt_size,
        compatible_belts,
        assembly_belt_model,
        packout_belt_model,
    ) = pick_section_belt_models(loader, belt_rng)
    if size_xy is None:
        size_x = FACTORY_SIZE_X
        size_y = FACTORY_SIZE_Y
    else:
        size_x = float(size_xy[0])
        size_y = float(size_xy[1])
    center_x = float(center_xy[0])
    center_y = float(center_xy[1])
    base_floor_z = float(floor_z)

    half_x = size_x * 0.5
    half_y = size_y * 0.5
    x_min = -half_x + EDGE_MARGIN_M
    x_max = half_x - EDGE_MARGIN_M
    y_min = -half_y + ROW_MARGIN_M
    y_max = half_y - ROW_MARGIN_M

    cell = max(0.2, base_belt_size[0])
    cols = max(8, int((x_max - x_min) // cell))
    rows = max(6, int((y_max - y_min) // cell))

    path_cells, start_corner, end_corner, end_goal = _generate_step1_path_seeded(
        cols, rows, seed=seed
    )

    def _to_world(cell_xy):
        cx, cy = cell_xy
        return (
            center_x + x_min + (cx + 0.5) * cell,
            center_y + y_min + (cy + 0.5) * cell,
        )

    base_support = select_support_for_target_height(loader, CONVEYOR_ELEVATION_M)
    if base_support is not None:
        support_model, support_scale = base_support
    else:
        support_model = pick_support_model(loader)
        support_scale = (
            support_scale_for_top_alignment(loader, support_model)
            if support_model is not None
            else None
        )
    support_choice_cache = {}

    def _support_for_height(target_top_z):
        key = round(float(target_top_z), 2)
        if key in support_choice_cache:
            return support_choice_cache[key]
        chosen = select_support_for_target_height(loader, target_top_z)
        if chosen is None:
            chosen = (support_model, support_scale)
        support_choice_cache[key] = chosen
        return chosen

    end_cap_model = pick_end_cap_model(loader)
    section_divider_model = pick_first_existing_model(
        loader, SECTION_DIVIDER_MODEL_CANDIDATES
    )
    drone_model = pick_first_existing_model(loader, ASSEMBLY_DRONE_MODEL_CANDIDATES)
    worker_model = pick_first_existing_model(loader, ASSEMBLY_WORKER_MODEL_CANDIDATES)
    swarm_drone_urdf = resolve_swarm_drone_urdf()
    assembly_model_label = (
        "swarm_drone.urdf" if swarm_drone_urdf is not None else (drone_model or "none")
    )
    worker_model_label = worker_model or "none"
    box_models = list_existing_models(loader, PACKOUT_BOX_MODEL_CANDIDATES)
    decor_rng = random.Random(int(seed) + 4049)
    split_idx = None
    if SECTION_DECOR_ENABLE and len(path_cells) >= (
        SECTION_SPLIT_MARGIN_CELLS * 2 + 12
    ):
        split_ratio = decor_rng.uniform(
            SECTION_SPLIT_RATIO_MIN, SECTION_SPLIT_RATIO_MAX
        )
        split_idx = int(round((len(path_cells) - 1) * split_ratio))
        split_idx = max(
            SECTION_SPLIT_MARGIN_CELLS,
            min(len(path_cells) - 1 - SECTION_SPLIT_MARGIN_CELLS, split_idx),
        )
    if split_idx is None:
        packout_belt_model = assembly_belt_model

    assembly_belt_size = loader.model_size(assembly_belt_model, CONVEYOR_SCALE)
    packout_belt_size = loader.model_size(packout_belt_model, CONVEYOR_SCALE)

    drones_spawned = 0
    workers_spawned = 0
    cartons_spawned = 0

    corner_indices = set()
    for i in range(1, len(path_cells) - 1):
        p0 = path_cells[i - 1]
        p1 = path_cells[i]
        p2 = path_cells[i + 1]
        in_dir = (p1[0] - p0[0], p1[1] - p0[1])
        out_dir = (p2[0] - p1[0], p2[1] - p1[1])
        in_dir = (max(-1, min(1, in_dir[0])), max(-1, min(1, in_dir[1])))
        out_dir = (max(-1, min(1, out_dir[0])), max(-1, min(1, out_dir[1])))
        if in_dir != out_dir:
            corner_indices.add(i)
    corners = len(corner_indices)
    support_every_cells = max(1, int(round(SUPPORT_SPACING_M / cell)))

    def _z_at(_idx):
        return CONVEYOR_ELEVATION_M

    cell_world = []
    cell_yaw = []
    cell_z = []
    cell_belt_height = []
    worker_points = []

    for i, cell_xy in enumerate(path_cells):
        wx, wy = _to_world(cell_xy)
        if i < len(path_cells) - 1:
            nx, ny = path_cells[i + 1]
            dx = nx - cell_xy[0]
            dy = ny - cell_xy[1]
            out_dir = (max(-1, min(1, dx)), max(-1, min(1, dy)))
        else:
            px, py = path_cells[i - 1]
            dx = cell_xy[0] - px
            dy = cell_xy[1] - py
            out_dir = (max(-1, min(1, dx)), max(-1, min(1, dy)))

        yaw = _dir_to_yaw(out_dir)
        z_here = _z_at(i)
        use_packout_belt = split_idx is not None and i >= split_idx
        belt_model = packout_belt_model if use_packout_belt else assembly_belt_model
        belt_h = packout_belt_size[2] if use_packout_belt else assembly_belt_size[2]
        loader.spawn(
            belt_model,
            x=wx,
            y=wy,
            yaw_deg=yaw,
            floor_z=base_floor_z,
            scale=CONVEYOR_SCALE,
            extra_z=z_here,
        )
        cell_world.append((wx, wy))
        cell_yaw.append(yaw)
        cell_z.append(z_here)
        cell_belt_height.append(belt_h)

        if support_model is not None and support_scale is not None:
            if (i % support_every_cells) == 0 or i in corner_indices:
                use_model, use_scale = _support_for_height(z_here)
                if use_model is not None and use_scale is not None:
                    loader.spawn(
                        use_model,
                        x=wx,
                        y=wy,
                        yaw_deg=yaw,
                        floor_z=base_floor_z,
                        scale=use_scale,
                        extra_z=0.0,
                    )

    # -- inner helpers for decor placement --
    def _is_straight_index(i):
        if i <= 0 or i >= (len(path_cells) - 1):
            return False
        if i in corner_indices:
            return False
        a, b, c = path_cells[i - 1], path_cells[i], path_cells[i + 1]
        return (b[0] - a[0], b[1] - a[1]) == (c[0] - b[0], c[1] - b[1])

    def _nearest_straight_index(start_i, lo, hi, max_radius=6):
        lo = max(1, lo)
        hi = min(len(path_cells) - 2, hi)
        if lo > hi:
            return None
        if _is_straight_index(start_i):
            return start_i
        for r in range(1, max_radius + 1):
            left = start_i - r
            right = start_i + r
            if left >= lo and _is_straight_index(left):
                return left
            if right <= hi and _is_straight_index(right):
                return right
        return None

    def _worker_slot_clear(xw, yw, anchor_idx):
        min_clear = cell * ASSEMBLY_WORKER_MIN_CLEARANCE_CELLS
        for j, (px, py) in enumerate(cell_world):
            if j == anchor_idx:
                continue
            if math.hypot(xw - px, yw - py) < min_clear:
                return False
        min_sep = cell * ASSEMBLY_WORKER_MIN_SEPARATION_CELLS
        for px, py in worker_points:
            if math.hypot(xw - px, yw - py) < min_sep:
                return False
        return True

    # -- section divider + assembly drones + workers + packout boxes --
    if SECTION_DECOR_ENABLE and split_idx is not None:
        divider_idx = _nearest_straight_index(
            split_idx,
            SECTION_SPLIT_MARGIN_CELLS,
            len(path_cells) - 1 - SECTION_SPLIT_MARGIN_CELLS,
            max_radius=24,
        )
        if divider_idx is None:
            divider_idx = split_idx

        if section_divider_model is not None and 0 <= divider_idx < len(path_cells):
            swx, swy = cell_world[divider_idx]
            loader.spawn(
                section_divider_model,
                x=swx,
                y=swy,
                yaw_deg=cell_yaw[divider_idx],
                floor_z=base_floor_z,
                scale=CONVEYOR_SCALE,
                extra_z=cell_z[divider_idx],
            )

        if (swarm_drone_urdf is not None or drone_model is not None) and split_idx > 10:
            interval = decor_rng.randint(
                ASSEMBLY_DRONE_INTERVAL_MIN, ASSEMBLY_DRONE_INTERVAL_MAX
            )
            i = max(6, interval // 2)
            while i < (split_idx - 4):
                idx_guess = max(4, min(split_idx - 4, i + decor_rng.randint(-2, 2)))
                idx = _nearest_straight_index(idx_guess, 4, split_idx - 4)
                if idx is None:
                    i += interval
                    continue
                dwx, dwy = cell_world[idx]
                drone_z = (
                    cell_z[idx] + cell_belt_height[idx] + ASSEMBLY_DRONE_Z_OFFSET_M
                )
                if swarm_drone_urdf is not None:
                    belt_top_z = (
                        cell_z[idx] + cell_belt_height[idx] + ASSEMBLY_DRONE_Z_OFFSET_M
                    )
                    _spawn_swarm_drone_urdf(
                        swarm_drone_urdf,
                        x=dwx,
                        y=dwy,
                        z=base_floor_z + drone_z,
                        yaw_deg=cell_yaw[idx],
                        global_scale=SWARM_DRONE_GLOBAL_SCALE * SWARM_DRONE_SCALE_MULT,
                        cli=cli,
                        target_bottom_z=base_floor_z + belt_top_z,
                    )
                elif drone_model is not None:
                    loader.spawn(
                        drone_model,
                        x=dwx,
                        y=dwy,
                        yaw_deg=cell_yaw[idx],
                        floor_z=base_floor_z,
                        scale=CONVEYOR_SCALE,
                        extra_z=drone_z,
                    )
                drones_spawned += 1
                i += interval

        if worker_model is not None and split_idx is not None and split_idx > 10:
            i = 5
            worker_anchor_used = set()
            worker_yaw_fix = ASSEMBLY_WORKER_MODEL_YAW_FIX_BY_MODEL.get(
                worker_model,
                ASSEMBLY_WORKER_MODEL_YAW_FIX_DEFAULT_DEG,
            )
            while i < (split_idx - 4):
                idx_guess = max(4, min(split_idx - 4, i + decor_rng.randint(-1, 1)))
                idx = _nearest_straight_index(idx_guess, 4, split_idx - 4)
                if idx is None or idx in worker_anchor_used:
                    i += ASSEMBLY_WORKER_INTERVAL_CELLS
                    continue
                worker_anchor_used.add(idx)
                dwx, dwy = cell_world[idx]
                yaw_rad = math.radians(cell_yaw[idx])
                side_x = -math.sin(yaw_rad)
                side_y = math.cos(yaw_rad)
                preferred_sign = decor_rng.choice((-1.0, 1.0))
                signs = (preferred_sign, -preferred_sign)
                placed_worker = False
                for sign in signs:
                    for _ in range(4):
                        off_cells = decor_rng.uniform(
                            ASSEMBLY_WORKER_SIDE_OFFSET_CELLS_MIN,
                            ASSEMBLY_WORKER_SIDE_OFFSET_CELLS_MAX,
                        )
                        rx = dwx + side_x * (cell * off_cells * sign)
                        ry = dwy + side_y * (cell * off_cells * sign)
                        if not _worker_slot_clear(rx, ry, idx):
                            continue
                        to_belt_x = dwx - rx
                        to_belt_y = dwy - ry
                        aim_yaw = math.degrees(math.atan2(to_belt_y, to_belt_x))
                        face_yaw = (aim_yaw + worker_yaw_fix) % 360.0
                        loader.spawn(
                            worker_model,
                            x=rx,
                            y=ry,
                            yaw_deg=face_yaw,
                            floor_z=base_floor_z,
                            scale=CONVEYOR_SCALE,
                            extra_z=0.0,
                        )
                        worker_points.append((rx, ry))
                        workers_spawned += 1
                        placed_worker = True
                        break
                    if placed_worker:
                        break
                i += ASSEMBLY_WORKER_INTERVAL_CELLS

        if box_models and split_idx is not None and split_idx < (len(path_cells) - 10):
            interval = decor_rng.randint(
                PACKOUT_BOX_INTERVAL_MIN, PACKOUT_BOX_INTERVAL_MAX
            )
            i = split_idx + 5
            while i < (len(path_cells) - 4):
                idx_guess = max(
                    split_idx + 4,
                    min(len(path_cells) - 4, i + decor_rng.randint(-2, 2)),
                )
                idx = _nearest_straight_index(
                    idx_guess, split_idx + 4, len(path_cells) - 4
                )
                if idx is None:
                    i += interval
                    continue
                bwx, bwy = cell_world[idx]
                yaw = cell_yaw[idx]
                yaw_rad = math.radians(yaw)
                side_x = -math.sin(yaw_rad)
                side_y = math.cos(yaw_rad)
                side_off = (
                    cell * PACKOUT_BOX_SIDE_OFFSET_CELLS * decor_rng.choice((-1.0, 1.0))
                )
                model_name = box_models[decor_rng.randint(0, len(box_models) - 1)]
                loader.spawn(
                    model_name,
                    x=bwx + (side_x * side_off),
                    y=bwy + (side_y * side_off),
                    yaw_deg=yaw,
                    floor_z=base_floor_z,
                    scale=(
                        CONVEYOR_SCALE * PACKOUT_BOX_SCALE_XY_MULT,
                        CONVEYOR_SCALE * PACKOUT_BOX_SCALE_XY_MULT,
                        CONVEYOR_SCALE * PACKOUT_BOX_SCALE_Z_MULT,
                    ),
                    extra_z=cell_z[idx]
                    + cell_belt_height[idx]
                    + PACKOUT_BOX_Z_OFFSET_M,
                )
                cartons_spawned += 1
                i += interval

    # -- end caps --
    if end_cap_model is not None and len(path_cells) >= 2:
        start_cell_v = path_cells[0]
        next_cell = path_cells[1]
        end_prev = path_cells[-2]
        end_cell_v = path_cells[-1]
        start_dir = (
            max(-1, min(1, next_cell[0] - start_cell_v[0])),
            max(-1, min(1, next_cell[1] - start_cell_v[1])),
        )
        end_dir = (
            max(-1, min(1, end_cell_v[0] - end_prev[0])),
            max(-1, min(1, end_cell_v[1] - end_prev[1])),
        )
        start_wx, start_wy = _to_world(start_cell_v)
        end_wx, end_wy = _to_world(end_cell_v)
        start_z = _z_at(0)
        end_z = _z_at(len(path_cells) - 1)
        off = cell * END_CAP_OUTWARD_OFFSET_CELLS
        loader.spawn(
            end_cap_model,
            x=start_wx - (start_dir[0] * off),
            y=start_wy - (start_dir[1] * off),
            yaw_deg=_dir_to_yaw(start_dir),
            floor_z=base_floor_z,
            scale=CONVEYOR_SCALE,
            extra_z=start_z,
        )
        loader.spawn(
            end_cap_model,
            x=end_wx + (end_dir[0] * off),
            y=end_wy + (end_dir[1] * off),
            yaw_deg=_dir_to_yaw(end_dir),
            floor_z=base_floor_z,
            scale=CONVEYOR_SCALE,
            extra_z=end_z,
        )

    line_len = max(0.0, (len(path_cells) - 1) * cell)
    if assembly_belt_model == packout_belt_model:
        belt_label = assembly_belt_model[:-4]
    else:
        belt_label = f"{assembly_belt_model[:-4]} -> {packout_belt_model[:-4]}"

    return {
        "model": belt_label,
        "module_size_xyz_m": base_belt_size,
        "base_cell_model": base_belt_model,
        "assembly_belt_model": assembly_belt_model,
        "packout_belt_model": packout_belt_model,
        "assembly_module_size_xyz_m": assembly_belt_size,
        "packout_module_size_xyz_m": packout_belt_size,
        "belt_model_pool_size": len(compatible_belts),
        "path_cells": len(path_cells),
        "corner_count": corners,
        "line_length_m": line_len,
        "support_model": support_model or "none",
        "support_scale": float(support_scale) if support_scale is not None else 0.0,
        "end_cap_model": end_cap_model or "none",
        "start_corner_cell": start_corner,
        "end_corner_cell": end_corner,
        "end_goal_cell": end_goal,
        "start_cell": path_cells[0] if path_cells else start_corner,
        "end_cell": path_cells[-1] if path_cells else end_goal,
        "section_split_index": int(split_idx) if split_idx is not None else -1,
        "section_divider_model": section_divider_model or "none",
        "assembly_model": assembly_model_label,
        "assembly_count": int(drones_spawned),
        "assembly_worker_models": worker_model_label,
        "assembly_worker_count": int(workers_spawned),
        "packout_count": int(cartons_spawned),
    }


# ---------------------------------------------------------------------------
