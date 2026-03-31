from ._shared import *


def resolve_swarm_drone_urdf():
    return first_existing_path(SWARM_DRONE_URDF_CANDIDATES)


def _spawn_box(center_xyz, size_xyz, rgba, cli, with_collision=True):
    hx = float(size_xyz[0]) * 0.5
    hy = float(size_xyz[1]) * 0.5
    hz = float(size_xyz[2]) * 0.5
    vid = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[hx, hy, hz],
        rgbaColor=list(rgba),
        physicsClientId=cli,
    )
    cid = (
        p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=cli
        )
        if with_collision
        else -1
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=cid,
        baseVisualShapeIndex=vid,
        basePosition=list(center_xyz),
        useMaximalCoordinates=True,
        physicsClientId=cli,
    )
    p.changeVisualShape(
        body_id,
        -1,
        textureUniqueId=-1,
        specularColor=list(UNIFORM_SPECULAR_COLOR),
        physicsClientId=cli,
    )


def _spawn_swarm_drone_urdf(
    urdf_path, x, y, z, yaw_deg, global_scale, cli, target_bottom_z=None
):
    urdf_abs = os.path.abspath(urdf_path)
    urdf_dir = os.path.dirname(urdf_abs)
    cache_key = (cli, urdf_dir)
    if cache_key not in _SWARM_URDF_SEARCH_PATHS:
        p.setAdditionalSearchPath(urdf_dir, physicsClientId=cli)
        _SWARM_URDF_SEARCH_PATHS[cache_key] = True
    z_spawn = float(z)
    offset_key = (cli, urdf_abs.replace("\\", "/"), round(float(global_scale), 6))
    cached_bottom_offset = (
        _SWARM_URDF_BOTTOM_Z_OFFSET_CACHE.get(offset_key)
        if target_bottom_z is not None
        else None
    )
    if cached_bottom_offset is not None:
        z_spawn += float(cached_bottom_offset)
    body_id = p.loadURDF(
        urdf_abs.replace("\\", "/"),
        basePosition=[float(x), float(y), z_spawn],
        baseOrientation=p.getQuaternionFromEuler(
            (0.0, 0.0, math.radians(float(yaw_deg)))
        ),
        useFixedBase=True,
        globalScaling=float(global_scale),
        physicsClientId=cli,
    )
    if target_bottom_z is not None and cached_bottom_offset is None:
        aabb = p.getAABB(body_id, physicsClientId=cli)
        min_z = float(aabb[0][2])
        dz = float(target_bottom_z) - min_z
        if abs(dz) > 1e-6:
            pos, orn = p.getBasePositionAndOrientation(body_id, physicsClientId=cli)
            p.resetBasePositionAndOrientation(
                body_id,
                [float(pos[0]), float(pos[1]), float(pos[2]) + dz],
                orn,
                physicsClientId=cli,
            )
        _SWARM_URDF_BOTTOM_Z_OFFSET_CACHE[offset_key] = float(dz)
    p.changeVisualShape(
        body_id,
        -1,
        specularColor=list(UNIFORM_SPECULAR_COLOR),
        physicsClientId=cli,
    )
    joint_count = p.getNumJoints(body_id, physicsClientId=cli)
    for j in range(joint_count):
        p.changeVisualShape(
            body_id,
            j,
            specularColor=list(UNIFORM_SPECULAR_COLOR),
            physicsClientId=cli,
        )


def _dir_to_yaw(direction):
    if direction == (1, 0):
        return 0.0
    if direction == (0, 1):
        return 90.0
    if direction == (-1, 0):
        return 180.0
    if direction == (0, -1):
        return 270.0
    return 0.0


def _inside_grid(cell, cols, rows):
    cx, cy = cell
    return 1 <= cx <= (cols - 2) and 1 <= cy <= (rows - 2)


def _expand_waypoints_to_cells(waypoints, cols, rows):
    path = []
    visited = set()
    for i in range(len(waypoints) - 1):
        x0, y0 = waypoints[i]
        x1, y1 = waypoints[i + 1]
        dx = 0 if x1 == x0 else (1 if x1 > x0 else -1)
        dy = 0 if y1 == y0 else (1 if y1 > y0 else -1)
        if dx != 0 and dy != 0:
            raise ValueError("Waypoints must be axis-aligned")
        cx, cy = x0, y0
        if i == 0:
            if not _inside_grid((cx, cy), cols, rows):
                raise ValueError("Waypoint out of grid")
            path.append((cx, cy))
            visited.add((cx, cy))
        while (cx, cy) != (x1, y1):
            cx += dx
            cy += dy
            if not _inside_grid((cx, cy), cols, rows):
                raise ValueError("Expanded cell out of grid")
            if (cx, cy) in visited:
                raise ValueError("Path self-intersection detected")
            path.append((cx, cy))
            visited.add((cx, cy))
    return path


# ---------------------------------------------------------------------------
# Two-phase seeded path generator
# ---------------------------------------------------------------------------
def _generate_step1_path_seeded(cols, rows, seed):
    base_seed = int(seed) + 1187
    corners = [(1, 1), (cols - 2, 1), (cols - 2, rows - 2), (1, rows - 2)]
    start_idx = int(seed) % 4

    def _fallback_path():
        start_corner = corners[start_idx]
        end_corner = corners[(start_idx + 2) % 4]
        end_goal = (
            max(
                1,
                min(
                    cols - 2,
                    end_corner[0]
                    + (
                        END_TARGET_OFFSET_CELLS
                        if end_corner[0] <= 1
                        else -END_TARGET_OFFSET_CELLS
                    ),
                ),
            ),
            max(
                1,
                min(
                    rows - 2,
                    end_corner[1]
                    + (
                        END_TARGET_OFFSET_CELLS
                        if end_corner[1] <= 1
                        else -END_TARGET_OFFSET_CELLS
                    ),
                ),
            ),
        )
        fallback = _expand_waypoints_to_cells(
            [start_corner, (end_goal[0], start_corner[1]), end_goal],
            cols,
            rows,
        )
        return fallback

    if cols < 8 or rows < 6:
        path = _fallback_path()
        start_corner = path[0]
        end_goal = path[-1]
        end_corner = corners[(start_idx + 2) % 4]
        return path, start_corner, end_corner, end_goal

    def _build_candidate(rng):
        def _lane_step():
            gap = rng.choice(LANE_EMPTY_GAP_CHOICES)
            return gap + 1

        def _build_two_phase(gen_cols, gen_rows, local_start_on_left, split_ratio):
            gx_left = 1
            gx_right = gen_cols - 2
            gy_bottom = 1
            gy_top = gen_rows - 2

            split_y = gy_bottom + int((gy_top - gy_bottom) * split_ratio)
            split_y += rng.randint(-1, 1)
            split_y = max(gy_bottom + 4, min(gy_top - 4, split_y))

            top_lanes = [gy_top]
            y_cursor = gy_top
            while True:
                ny = y_cursor - _lane_step()
                if ny <= split_y:
                    break
                top_lanes.append(ny)
                y_cursor = ny
            if len(top_lanes) < 2:
                top_lanes = [gy_top, max(gy_bottom + 2, gy_top - 2)]

            y_v_top = max(gy_bottom + 1, split_y - 1)
            current_x = gx_left if local_start_on_left else gx_right
            current_y = top_lanes[0]
            out_path = [(current_x, current_y)]

            def _append_line_to(tx, ty):
                nonlocal current_x, current_y, out_path
                sx = 0 if tx == current_x else (1 if tx > current_x else -1)
                sy = 0 if ty == current_y else (1 if ty > current_y else -1)
                while current_x != tx:
                    current_x += sx
                    out_path.append((current_x, current_y))
                while current_y != ty:
                    current_y += sy
                    out_path.append((current_x, current_y))

            top_span = max(1, gx_right - gx_left)
            max_inset = min(
                TOP_PHASE_EDGE_INSET_MAX_CELLS, max(0, (top_span - 10) // 8)
            )
            go_right = local_start_on_left
            for i, y_lane in enumerate(top_lanes):
                _append_line_to(current_x, y_lane)
                edge_x = gx_right if go_right else gx_left
                inset = rng.randint(0, max_inset) if max_inset > 0 else 0
                target_x = (edge_x - inset) if go_right else (edge_x + inset)
                _append_line_to(target_x, y_lane)
                if i < len(top_lanes) - 1:
                    _append_line_to(target_x, top_lanes[i + 1])
                    go_right = not go_right

            _append_line_to(current_x, y_v_top)

            if current_x == gx_left:
                x_lanes = [gx_left]
                x_cursor = gx_left
                while True:
                    step = _lane_step() + (
                        1 if rng.random() < VERTICAL_PHASE_EXTRA_SKIP_CHANCE else 0
                    )
                    nx = x_cursor + step
                    if nx > gx_right:
                        break
                    x_lanes.append(nx)
                    x_cursor = nx
            else:
                x_lanes = [gx_right]
                x_cursor = gx_right
                while True:
                    step = _lane_step() + (
                        1 if rng.random() < VERTICAL_PHASE_EXTRA_SKIP_CHANCE else 0
                    )
                    nx = x_cursor - step
                    if nx < gx_left:
                        break
                    x_lanes.append(nx)
                    x_cursor = nx

            if len(x_lanes) >= 2 and (len(x_lanes) % 2) == 0:
                x_lanes = x_lanes[:-1]

            go_down = True
            for i, x_lane in enumerate(x_lanes):
                _append_line_to(x_lane, current_y)
                target_y = gy_bottom if go_down else y_v_top
                _append_line_to(x_lane, target_y)
                if i < len(x_lanes) - 1:
                    _append_line_to(x_lanes[i + 1], target_y)
                    go_down = not go_down

            return out_path

        split_mode = rng.choices(
            ("h_dominant", "balanced", "v_dominant"),
            weights=(2, 7, 2),
            k=1,
        )[0]
        if split_mode == "h_dominant":
            split_ratio = rng.uniform(0.36, 0.46)
        elif split_mode == "v_dominant":
            split_ratio = rng.uniform(0.54, 0.64)
        else:
            split_ratio = rng.uniform(0.42, 0.60)

        start_on_left = (start_idx % 2) == 0
        aspect = float(cols) / float(max(1, rows))
        if aspect >= 1.25:
            transpose_prob = 0.20
        elif aspect <= 0.80:
            transpose_prob = 0.80
        else:
            transpose_prob = 0.50
        use_transposed_layout = rng.random() < transpose_prob
        if not use_transposed_layout:
            return _build_two_phase(cols, rows, start_on_left, split_ratio)
        path_swapped = _build_two_phase(rows, cols, start_on_left, split_ratio)
        return [(y, x) for (x, y) in path_swapped]

    def _path_metrics(path):
        if not path:
            return {
                "ok": False,
                "cells": 0,
                "corners": 0,
                "span_x_ratio": 0.0,
                "span_y_ratio": 0.0,
            }
        if len(path) != len(set(path)):
            return {
                "ok": False,
                "unique": False,
                "cells": len(path),
                "corners": 0,
                "span_x_ratio": 0.0,
                "span_y_ratio": 0.0,
                "min_seg_len": 0,
            }

        def _min_segment_len_edges():
            if len(path) < 2:
                return 0
            seg_lens = []
            prev_dx = path[1][0] - path[0][0]
            prev_dy = path[1][1] - path[0][1]
            seg_edges = 1
            for i in range(2, len(path)):
                cur_dx = path[i][0] - path[i - 1][0]
                cur_dy = path[i][1] - path[i - 1][1]
                if (cur_dx, cur_dy) == (prev_dx, prev_dy):
                    seg_edges += 1
                else:
                    seg_lens.append(seg_edges)
                    seg_edges = 1
                    prev_dx, prev_dy = cur_dx, cur_dy
            seg_lens.append(seg_edges)
            return min(seg_lens) if seg_lens else 0

        corners_n = 0
        for i in range(1, len(path) - 1):
            a, b, c = path[i - 1], path[i], path[i + 1]
            if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
                corners_n += 1
        xs = [c[0] for c in path]
        ys = [c[1] for c in path]
        span_x = (max(xs) - min(xs) + 1) if xs else 0
        span_y = (max(ys) - min(ys) + 1) if ys else 0
        usable_x = max(1, cols - 2)
        usable_y = max(1, rows - 2)
        span_x_ratio = float(span_x) / float(usable_x)
        span_y_ratio = float(span_y) / float(usable_y)
        min_seg_len = _min_segment_len_edges()
        mid_x = (1 + (cols - 2)) * 0.5
        mid_y = (1 + (rows - 2)) * 0.5
        left_count = sum(1 for x in xs if x <= mid_x)
        right_count = len(path) - left_count
        bottom_count = sum(1 for y in ys if y <= mid_y)
        top_count = len(path) - bottom_count
        left_ratio = left_count / float(len(path))
        right_ratio = right_count / float(len(path))
        bottom_ratio = bottom_count / float(len(path))
        top_ratio = top_count / float(len(path))
        q_lb = q_rb = q_lt = q_rt = 0
        for x, y in path:
            if x <= mid_x and y <= mid_y:
                q_lb += 1
            elif x > mid_x and y <= mid_y:
                q_rb += 1
            elif x <= mid_x and y > mid_y:
                q_lt += 1
            else:
                q_rt += 1
        min_quad_ratio = min(q_lb, q_rb, q_lt, q_rt) / float(len(path))
        bins_x = max(1, int(PATH_OCCUPANCY_BINS_X))
        bins_y = max(1, int(PATH_OCCUPANCY_BINS_Y))
        occ = [[0 for _ in range(bins_y)] for _ in range(bins_x)]
        for x, y in path:
            nx = (float(x - 1) / float(max(1, usable_x))) if usable_x > 0 else 0.0
            ny = (float(y - 1) / float(max(1, usable_y))) if usable_y > 0 else 0.0
            bx = max(0, min(bins_x - 1, int(nx * bins_x)))
            by = max(0, min(bins_y - 1, int(ny * bins_y)))
            occ[bx][by] += 1
        empty_bins = sum(
            1 for bx in range(bins_x) for by in range(bins_y) if occ[bx][by] <= 0
        )
        min_bin_fill = min(occ[bx][by] for bx in range(bins_x) for by in range(bins_y))

        min_cells_target = min(PATH_MAX_CELLS_HARD, max(48, PATH_MIN_CELLS))
        min_turns_target = max(6, PATH_MIN_TURNS)
        ok = (
            len(path) >= min_cells_target
            and len(path) <= PATH_MAX_CELLS_HARD
            and corners_n >= min_turns_target
            and span_x_ratio >= PATH_MIN_SPAN_X_RATIO
            and span_y_ratio >= PATH_MIN_SPAN_Y_RATIO
            and min_seg_len >= PATH_MIN_SEG_CELLS
            and left_ratio >= PATH_HALF_MIN_RATIO
            and right_ratio >= PATH_HALF_MIN_RATIO
            and bottom_ratio >= PATH_HALF_MIN_RATIO
            and top_ratio >= PATH_HALF_MIN_RATIO
            and min_quad_ratio >= PATH_QUADRANT_MIN_RATIO
        )
        return {
            "ok": ok,
            "unique": True,
            "cells": len(path),
            "corners": corners_n,
            "span_x_ratio": span_x_ratio,
            "span_y_ratio": span_y_ratio,
            "min_seg_len": min_seg_len,
            "half_balance": min(left_ratio, right_ratio, bottom_ratio, top_ratio),
            "min_quad_ratio": min_quad_ratio,
            "empty_bins": empty_bins,
            "min_bin_fill": min_bin_fill,
        }

    best_path = None
    best_score = -1.0
    best_path_limited_empty = None
    best_score_limited_empty = -1.0
    best_ok_path = None
    best_ok_score = -1.0
    for attempt in range(max(1, PATH_BUILD_ATTEMPTS)):
        rng = random.Random(base_seed + (attempt * 9973))
        candidate = _build_candidate(rng)
        metrics = _path_metrics(candidate)
        if metrics["ok"]:
            ok_score = (
                metrics["cells"] * 1.0
                + metrics["corners"] * 3.0
                + metrics["span_x_ratio"] * 60.0
                + metrics["span_y_ratio"] * 60.0
                + metrics["min_seg_len"] * 25.0
                + metrics["half_balance"] * 85.0
                + metrics["min_quad_ratio"] * 150.0
                - metrics["empty_bins"] * 120.0
                + metrics["min_bin_fill"] * 2.0
            )
            if ok_score > best_ok_score:
                best_ok_score = ok_score
                best_ok_path = candidate
            if metrics["empty_bins"] <= 0 and metrics["min_bin_fill"] >= 4:
                break
        if metrics.get("unique", False):
            score = (
                metrics["cells"] * 1.0
                + metrics["corners"] * 3.0
                + metrics["span_x_ratio"] * 60.0
                + metrics["span_y_ratio"] * 60.0
                + metrics["min_seg_len"] * 25.0
                + metrics["half_balance"] * 80.0
                + metrics["min_quad_ratio"] * 140.0
                - metrics["empty_bins"] * 90.0
                + metrics["min_bin_fill"] * 1.5
            )
            if score > best_score:
                best_score = score
                best_path = candidate
            if (
                metrics["empty_bins"] <= PATH_FALLBACK_EMPTY_BINS_MAX
                and score > best_score_limited_empty
            ):
                best_score_limited_empty = score
                best_path_limited_empty = candidate
    if best_ok_path is not None:
        path = best_ok_path
    else:
        path = (
            best_path_limited_empty
            if best_path_limited_empty is not None
            else (best_path if best_path is not None else _fallback_path())
        )

    start_cell = path[0]
    end_goal = path[-1]

    def _dist(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    start_corner = min(corners, key=lambda c: _dist(c, start_cell))
    end_corner = min(corners, key=lambda c: _dist(c, end_goal))
    if start_corner == end_corner:
        end_corner = corners[(corners.index(start_corner) + 2) % 4]

    return path, start_corner, end_corner, end_goal


# ---------------------------------------------------------------------------
# Model selection helpers
# ---------------------------------------------------------------------------
def pick_support_model(loader):
    for model_name in SUPPORT_MODEL_CANDIDATES:
        if os.path.exists(os.path.join(loader.obj_dir, model_name)):
            return model_name
    return None


def pick_end_cap_model(loader):
    for model_name in END_CAP_MODEL_CANDIDATES:
        if os.path.exists(os.path.join(loader.obj_dir, model_name)):
            return model_name
    return None


def pick_first_existing_model(loader, candidates):
    for model_name in candidates:
        if os.path.exists(os.path.join(loader.obj_dir, model_name)):
            return model_name
    return None


def list_existing_models(loader, candidates):
    out = []
    for model_name in candidates:
        if os.path.exists(os.path.join(loader.obj_dir, model_name)):
            out.append(model_name)
    return out


def pick_section_belt_models(loader, rng):
    available = list_existing_models(loader, NETWORK_BELT_MODEL_CANDIDATES)
    if not available:
        raise FileNotFoundError(
            "No conveyor models found. Expected one of: "
            + ", ".join(NETWORK_BELT_MODEL_CANDIDATES)
        )
    if NETWORK_BELT_MODEL in available:
        base_model = NETWORK_BELT_MODEL
    else:
        base_model = available[0]
    base_size = loader.model_size(base_model, CONVEYOR_SCALE)
    compatible = []
    for model_name in available:
        sx, sy, _ = loader.model_size(model_name, CONVEYOR_SCALE)
        len_ratio = sx / max(1e-6, base_size[0])
        wid_ratio = sy / max(1e-6, base_size[1])
        if (
            BELT_COMPAT_LEN_RATIO_MIN <= len_ratio <= BELT_COMPAT_LEN_RATIO_MAX
            and BELT_COMPAT_WID_RATIO_MIN <= wid_ratio <= BELT_COMPAT_WID_RATIO_MAX
        ):
            compatible.append(model_name)
    if not compatible:
        compatible = [base_model]
    assembly_model = compatible[rng.randrange(len(compatible))]
    packout_model = assembly_model
    if ALLOW_MIXED_SECTION_BELTS and len(compatible) > 1:
        alternatives = [m for m in compatible if m != assembly_model]
        if alternatives:
            packout_model = alternatives[rng.randrange(len(alternatives))]
    return base_model, base_size, compatible, assembly_model, packout_model


def support_scale_for_top_alignment(loader, support_model):
    _, _, raw_h = loader.model_size(support_model, 1.0)
    if raw_h <= 1e-6:
        return None
    return max(0.05, CONVEYOR_ELEVATION_M / raw_h)


def select_support_for_target_height(loader, target_top_z):
    candidates = (
        "structure-medium.obj",
        "structure-high.obj",
        "structure-tall.obj",
        "structure-short.obj",
    )
    best = None
    best_score = float("inf")
    for model_name in candidates:
        full = os.path.join(loader.obj_dir, model_name)
        if not os.path.exists(full):
            continue
        _, _, raw_h = loader.model_size(model_name, 1.0)
        if raw_h <= 1e-6:
            continue
        scale = max(0.05, float(target_top_z) / raw_h)
        penalty = 0.0
        if scale < 0.45:
            penalty += (0.45 - scale) * 3.5
        if scale > 1.45:
            penalty += (scale - 1.45) * 3.5
        score = abs(scale - 1.0) + penalty
        if score < best_score:
            best_score = score
            best = (model_name, scale)
    return best


# ---------------------------------------------------------------------------
