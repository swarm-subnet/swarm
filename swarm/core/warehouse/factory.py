"""
Embedded factory conveyor belt network for Type 5 warehouse maps.
Generates a seeded conveyor path with belt segments, supports, drones,
workers, cartons, and an optional barrier fence ring.
"""

import math
import os
import random

import pybullet as p

from .constants import (
    ENABLE_EMBEDDED_FACTORY_MAP,
    ENABLE_FACTORY_BARRIER_RING,
    EMBEDDED_FACTORY_SEED_OFFSET,
    FACTORY_BARRIER_DOUBLE_SIDED,
    FACTORY_BARRIER_FLAT_RGBA,
    FACTORY_BARRIER_INSET_M,
    FACTORY_BARRIER_MODEL_PATH,
    FACTORY_BARRIER_SCALE_XYZ,
    FACTORY_BARRIER_SEGMENT_GAP_M,
    FACTORY_BARRIER_WITH_COLLISION,
    OTHER_SOURCES_DIR,
    SHOW_AREA_LAYOUT_MARKERS,
    UNIFORM_SPECULAR_COLOR,
)
from .helpers import (
    _obj_collision_proxy_path,
    _spawn_collision_only_with_anchor,
    _spawn_mesh_with_anchor,
    model_bounds_xyz,
)
from .shared import MeshKitLoader, first_existing_path

# ---------------------------------------------------------------------------
# Factory geometry constants
# ---------------------------------------------------------------------------
FACTORY_SIZE_X = 40.0
FACTORY_SIZE_Y = 20.0
FLOOR_HALF_Z = 0.03
CONVEYOR_SCALE = 0.95
CONVEYOR_ELEVATION_M = 0.62
EDGE_MARGIN_M = 3.0
ROW_MARGIN_M = 1.4

# ---------------------------------------------------------------------------
# Conveyor support models
# ---------------------------------------------------------------------------
SUPPORT_MODEL_CANDIDATES = (
    "structure-medium.obj",
    "structure-short.obj",
    "structure-high.obj",
    "structure-tall.obj",
)
SUPPORT_SPACING_M = 1.9

# ---------------------------------------------------------------------------
# Belt models
# ---------------------------------------------------------------------------
NETWORK_BELT_MODEL = "conveyor-stripe-sides.obj"
NETWORK_BELT_MODEL_CANDIDATES = ("conveyor-stripe-sides.obj",)
BELT_COMPAT_LEN_RATIO_MIN = 0.96
BELT_COMPAT_LEN_RATIO_MAX = 1.04
BELT_COMPAT_WID_RATIO_MIN = 0.96
BELT_COMPAT_WID_RATIO_MAX = 1.04
ALLOW_MIXED_SECTION_BELTS = False

# ---------------------------------------------------------------------------
# End caps / section decor
# ---------------------------------------------------------------------------
END_CAP_MODEL_CANDIDATES = (
    "cover-stripe-top.obj",
    "cover-top.obj",
    "cover-stripe.obj",
    "cover.obj",
)
END_CAP_OUTWARD_OFFSET_CELLS = 0.55
SECTION_DECOR_ENABLE = True
SECTION_SPLIT_RATIO_MIN = 0.62
SECTION_SPLIT_RATIO_MAX = 0.72
SECTION_SPLIT_MARGIN_CELLS = 18
SECTION_DIVIDER_MODEL_CANDIDATES = (
    "cover-stripe-hopper.obj",
    "cover-stripe-top.obj",
    "cover-top.obj",
)

# ---------------------------------------------------------------------------
# Assembly drones / workers / cartons
# ---------------------------------------------------------------------------
ASSEMBLY_DRONE_MODEL_CANDIDATES = (
    "scanner-low.obj",
    "scanner-high.obj",
    "robot-arm-a.obj",
)
ASSEMBLY_DRONE_INTERVAL_MIN = 12
ASSEMBLY_DRONE_INTERVAL_MAX = 18
ASSEMBLY_DRONE_Z_OFFSET_M = 0.0
ASSEMBLY_WORKER_MODEL_CANDIDATES = (
    "robot-arm-b-angled.obj",
    "robot-arm-b.obj",
)
ASSEMBLY_WORKER_SIDE_OFFSET_CELLS_MIN = 0.95
ASSEMBLY_WORKER_SIDE_OFFSET_CELLS_MAX = 1.45
ASSEMBLY_WORKER_MIN_CLEARANCE_CELLS = 1.10
ASSEMBLY_WORKER_MIN_SEPARATION_CELLS = 2.20
ASSEMBLY_WORKER_INTERVAL_CELLS = 11
ASSEMBLY_WORKER_MODEL_YAW_FIX_BY_MODEL = {
    "robot-arm-b-angled.obj": 90.0,
    "robot-arm-b.obj": 0.0,
}
ASSEMBLY_WORKER_MODEL_YAW_FIX_DEFAULT_DEG = 0.0
PACKOUT_BOX_MODEL_CANDIDATES = ("box-small.obj",)
PACKOUT_BOX_INTERVAL_MIN = 7
PACKOUT_BOX_INTERVAL_MAX = 12
PACKOUT_BOX_Z_OFFSET_M = 0.01
PACKOUT_BOX_SIDE_OFFSET_CELLS = 0.0
PACKOUT_BOX_SCALE_XY_MULT = 1.10
PACKOUT_BOX_SCALE_Z_MULT = 1.00

# ---------------------------------------------------------------------------
# Swarm drone URDF
# ---------------------------------------------------------------------------
_SWARM_DRONE_DIR = os.path.join(OTHER_SOURCES_DIR, "swarm_drone")
SWARM_DRONE_URDF_CANDIDATES = (os.path.join(_SWARM_DRONE_DIR, "swarm_drone.urdf"),)
SWARM_DRONE_GLOBAL_SCALE = 3.5
SWARM_DRONE_SCALE_MULT = 1.38

# ---------------------------------------------------------------------------
# Path generation tuning
# ---------------------------------------------------------------------------
PATH_MIN_SEG_CELLS = 2
PATH_MAX_SEG_CELLS = 8
PATH_MIN_TURNS = 10
PATH_MIN_CELLS = 95
PATH_BUILD_ATTEMPTS = 320
PATH_MIN_SPAN_X_RATIO = 0.70
PATH_MIN_SPAN_Y_RATIO = 0.65
PATH_MAX_CELLS_HARD = 220
END_TARGET_OFFSET_CELLS = 2
PATH_HALF_MIN_RATIO = 0.30
PATH_QUADRANT_MIN_RATIO = 0.06
PATH_OCCUPANCY_BINS_X = 6
PATH_OCCUPANCY_BINS_Y = 3
PATH_FALLBACK_EMPTY_BINS_MAX = 3
LANE_EMPTY_GAP_CHOICES = (2, 2, 3, 3, 4)
TOP_PHASE_EDGE_INSET_MAX_CELLS = 0
VERTICAL_PHASE_EXTRA_SKIP_CHANCE = 0.30

FLOOR_RGBA = (0.64, 0.67, 0.75, 1.0)

# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------
_SWARM_URDF_SEARCH_PATHS = {}
_SWARM_URDF_BOTTOM_Z_OFFSET_CACHE = {}
_FACTORY_BARRIER_LOADER_CACHE = {}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
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
# Main belt network builder
# ---------------------------------------------------------------------------
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
# Factory barrier fence ring
# ---------------------------------------------------------------------------
def _segment_centers_1d(min_v, max_v, segment_len, gap):
    span = max_v - min_v
    if segment_len <= 1e-6 or span <= 1e-6:
        return []
    if span <= segment_len:
        return [0.5 * (min_v + max_v)]
    step = segment_len + max(0.0, gap)
    count = max(1, int((span + gap) // step))
    used = (count * segment_len) + ((count - 1) * max(0.0, gap))
    if used > span and count > 1:
        count -= 1
        used = (count * segment_len) + ((count - 1) * max(0.0, gap))
    if count <= 0:
        return [0.5 * (min_v + max_v)]
    slack = max(0.0, span - used)
    start = min_v + (slack * 0.5) + (segment_len * 0.5)
    return [start + (i * step) for i in range(count)]


def _resolve_factory_barrier_model():
    if FACTORY_BARRIER_MODEL_PATH and os.path.exists(FACTORY_BARRIER_MODEL_PATH):
        return os.path.abspath(FACTORY_BARRIER_MODEL_PATH)
    return ""


def _get_factory_barrier_loader(model_dir, texture_path, cli):
    key = (cli, os.path.abspath(str(model_dir)), str(texture_path or ""))
    cached = _FACTORY_BARRIER_LOADER_CACHE.get(key)
    if cached is not None:
        return cached
    loader = MeshKitLoader(obj_dir=key[1], texture_path=key[2], cli=cli)
    _FACTORY_BARRIER_LOADER_CACHE[key] = loader
    return loader


def build_factory_barrier_ring(
    conveyor_loader, floor_top_z, factory_area, network, cli
):
    if not ENABLE_FACTORY_BARRIER_RING:
        return {"factory_barrier_enabled": False}
    if not factory_area:
        return {
            "factory_barrier_enabled": False,
            "factory_barrier_reason": "FACTORY area missing.",
        }

    barrier_model_path = _resolve_factory_barrier_model()
    if not barrier_model_path:
        return {
            "factory_barrier_enabled": False,
            "factory_barrier_reason": "Barrier model not found.",
        }

    model_dir = os.path.dirname(barrier_model_path)
    model_name = os.path.basename(barrier_model_path)
    barrier_loader = _get_factory_barrier_loader(
        model_dir=model_dir,
        texture_path=conveyor_loader.texture_path,
        cli=cli,
    )

    min_v, max_v = model_bounds_xyz(
        barrier_loader, model_name, FACTORY_BARRIER_SCALE_XYZ
    )
    seg_len = max(0.0, max_v[0] - min_v[0])
    seg_dep = max(0.0, max_v[1] - min_v[1])
    if seg_len <= 1e-6 or seg_dep <= 1e-6:
        return {
            "factory_barrier_enabled": False,
            "factory_barrier_reason": "Barrier model has invalid bounds.",
        }

    anchor_x = (min_v[0] + max_v[0]) * 0.5
    anchor_y = (min_v[1] + max_v[1]) * 0.5
    anchor_z = min_v[2]

    cx = float(factory_area["cx"])
    cy = float(factory_area["cy"])
    sx = float(factory_area["sx"])
    sy = float(factory_area["sy"])
    half_x = sx * 0.5
    half_y = sy * 0.5
    inset = max(0.0, float(FACTORY_BARRIER_INSET_M))

    x_min = cx - half_x + inset
    x_max = cx + half_x - inset
    y_min = cy - half_y + inset
    y_max = cy + half_y - inset
    if x_max <= x_min or y_max <= y_min:
        return {
            "factory_barrier_enabled": False,
            "factory_barrier_reason": "FACTORY area too small for barrier inset.",
        }

    north_y = y_max - (seg_dep * 0.5)
    south_y = y_min + (seg_dep * 0.5)
    east_x = x_max - (seg_dep * 0.5)
    west_x = x_min + (seg_dep * 0.5)

    segments = []
    for x in _segment_centers_1d(x_min, x_max, seg_len, FACTORY_BARRIER_SEGMENT_GAP_M):
        segments.append({"side": "north", "x": x, "y": north_y, "yaw": 0.0})
        segments.append({"side": "south", "x": x, "y": south_y, "yaw": 0.0})
    for y in _segment_centers_1d(y_min, y_max, seg_len, FACTORY_BARRIER_SEGMENT_GAP_M):
        segments.append({"side": "east", "x": east_x, "y": y, "yaw": 90.0})
        segments.append({"side": "west", "x": west_x, "y": y, "yaw": 90.0})
    if not segments:
        return {
            "factory_barrier_enabled": False,
            "factory_barrier_reason": "No barrier segments could be generated.",
        }

    start_pos = (cx, cy)
    start_cell = network.get("start_cell")
    module_size = network.get("module_size_xyz_m", (0.0, 0.0, 0.0))
    cell_size = float(module_size[0]) if module_size else 0.0
    if (
        isinstance(start_cell, (tuple, list))
        and len(start_cell) >= 2
        and cell_size > 1e-6
    ):
        edge_margin = EDGE_MARGIN_M
        row_margin = ROW_MARGIN_M
        local_x_min = -half_x + edge_margin
        local_y_min = -half_y + row_margin
        start_pos = (
            cx + local_x_min + (float(start_cell[0]) + 0.5) * cell_size,
            cy + local_y_min + (float(start_cell[1]) + 0.5) * cell_size,
        )

    gap_idx = min(
        range(len(segments)),
        key=lambda i: (segments[i]["x"] - start_pos[0]) ** 2
        + (segments[i]["y"] - start_pos[1]) ** 2,
    )
    entry_side = segments[gap_idx]["side"]
    segments_to_spawn = [s for i, s in enumerate(segments) if i != gap_idx]

    collision_proxy = _obj_collision_proxy_path(barrier_model_path)
    spawned = 0
    for seg in segments_to_spawn:
        world_anchor = (seg["x"], seg["y"], float(floor_top_z))
        yaw_deg = float(seg["yaw"])
        if FACTORY_BARRIER_WITH_COLLISION:
            _spawn_collision_only_with_anchor(
                loader=barrier_loader,
                model_name=model_name,
                world_anchor_xyz=world_anchor,
                yaw_deg=yaw_deg,
                mesh_scale_xyz=FACTORY_BARRIER_SCALE_XYZ,
                local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                cli=cli,
                model_path_override=collision_proxy,
            )
        _spawn_mesh_with_anchor(
            loader=barrier_loader,
            model_name=model_name,
            world_anchor_xyz=world_anchor,
            yaw_deg=yaw_deg,
            mesh_scale_xyz=FACTORY_BARRIER_SCALE_XYZ,
            local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
            cli=cli,
            with_collision=False,
            use_texture=False,
            rgba=FACTORY_BARRIER_FLAT_RGBA,
            double_sided=FACTORY_BARRIER_DOUBLE_SIDED,
        )
        spawned += 1

    return {
        "factory_barrier_enabled": spawned > 0,
        "factory_barrier_model": model_name,
        "factory_barrier_scale_xyz": FACTORY_BARRIER_SCALE_XYZ,
        "factory_barrier_count": int(spawned),
        "factory_barrier_entry_side": entry_side,
        "factory_barrier_entry_xy": (float(start_pos[0]), float(start_pos[1])),
    }


# ---------------------------------------------------------------------------
# Entry point — called from __init__.py
# ---------------------------------------------------------------------------
def build_embedded_factory(conveyor_loader, floor_top_z, area_layout, seed, cli=0):
    if not ENABLE_EMBEDDED_FACTORY_MAP:
        return {"factory_map_embedded": False}

    factory_area = (area_layout or {}).get("FACTORY")
    if not factory_area:
        return {
            "factory_map_embedded": False,
            "factory_map_reason": "FACTORY area not present in layout.",
        }

    size_xy = (float(factory_area["sx"]), float(factory_area["sy"]))
    center_xy = (float(factory_area["cx"]), float(factory_area["cy"]))
    try:
        network = build_single_belt_network(
            conveyor_loader,
            seed=int(seed) + EMBEDDED_FACTORY_SEED_OFFSET,
            cli=cli,
            center_xy=center_xy,
            size_xy=size_xy,
            floor_z=float(floor_top_z),
        )
        barrier_info = build_factory_barrier_ring(
            conveyor_loader=conveyor_loader,
            floor_top_z=float(floor_top_z),
            factory_area=factory_area,
            network=network,
            cli=cli,
        )
        if SHOW_AREA_LAYOUT_MARKERS:
            p.addUserDebugText(
                text=f"EMBEDDED FACTORY | {size_xy[0]:.0f} x {size_xy[1]:.0f} m",
                textPosition=[center_xy[0], center_xy[1], float(floor_top_z) + 0.03],
                textColorRGB=[0.08, 0.10, 0.12],
                textSize=1.2,
                lifeTime=0.0,
                physicsClientId=cli,
            )
        return {
            "factory_map_embedded": True,
            "factory_area_center_xy": center_xy,
            "factory_area_size_m": size_xy,
            "factory_network": network,
            **barrier_info,
        }
    except Exception as exc:
        return {"factory_map_embedded": False, "factory_map_reason": str(exc)}
