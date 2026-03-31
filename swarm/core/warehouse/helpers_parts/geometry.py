from ._shared import *
from .mesh_spawn import _loader_runtime_key


def slot_point(slot, along, inward):
    if slot == "north":
        return along, HALF_Y - inward
    if slot == "south":
        return along, -HALF_Y + inward
    if slot == "east":
        return HALF_X - inward, along
    if slot == "west":
        return -HALF_X + inward, along
    raise ValueError(f"Unknown slot: {slot}")


def dock_inward_yaw_for_slot(slot):
    if slot == "north":
        return 180.0
    if slot == "south":
        return 0.0
    if slot == "east":
        return 90.0
    if slot == "west":
        return 270.0
    raise ValueError(f"Unknown slot: {slot}")


def wall_yaw_for_slot(slot):
    return dock_inward_yaw_for_slot(slot)


def tiled_centers(total_size, tile_size):
    if tile_size <= 1e-9:
        raise ValueError(f"Invalid tile size: {tile_size}")
    n = max(1, int(math.floor(float(total_size) / float(tile_size))))
    covered = n * float(tile_size)
    start = -covered * 0.5 + float(tile_size) * 0.5
    return [start + i * tile_size for i in range(n)]


def oriented_xy_size(loader, model_name, scale, yaw_deg):
    if isinstance(scale, (tuple, list)):
        scale_key = (float(scale[0]), float(scale[1]), float(scale[2]))
    else:
        s = float(scale)
        scale_key = (s, s, s)
    yaw_key = round(float(yaw_deg) % 360.0, 6)
    cache_key = (_loader_runtime_key(loader), str(model_name), scale_key, yaw_key)
    cached = _ORIENTED_XY_SIZE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    sx, sy, _ = loader.model_size(model_name, scale_key)
    yaw = math.radians(float(yaw_key))
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    ex = (c * sx) + (s * sy)
    ey = (s * sx) + (c * sy)
    out = (float(ex), float(ey))
    _ORIENTED_XY_SIZE_CACHE[cache_key] = out
    return out


def model_bounds_xyz(loader, model_name, scale_xyz):
    sx, sy, sz = float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])
    scale_key = (sx, sy, sz)
    cache_key = (_loader_runtime_key(loader), str(model_name), scale_key)
    cached = _MODEL_BOUNDS_CACHE.get(cache_key)
    if cached is not None:
        return cached
    if hasattr(loader, "_bounds"):
        min_v, max_v = loader._bounds(model_name, scale_key)
    else:
        verts = loader._parse_vertices(model_name)
        transformed = [(v[0] * sx, v[1] * sy, v[2] * sz) for v in verts]
        min_v = [min(v[i] for v in transformed) for i in range(3)]
        max_v = [max(v[i] for v in transformed) for i in range(3)]
    out = (
        [float(min_v[0]), float(min_v[1]), float(min_v[2])],
        [float(max_v[0]), float(max_v[1]), float(max_v[2])],
    )
    _MODEL_BOUNDS_CACHE[cache_key] = out
    return out


def _first_existing_model_name(loader, candidates):
    for model_name in candidates:
        if os.path.exists(os.path.join(loader.obj_dir, model_name)):
            return model_name
    return None


def _shell_mesh_scale_xy(shell_meshes):
    cfg = shell_meshes.get("config", {}) or {}
    base_x = float(cfg.get("warehouse_size_x", WAREHOUSE_BASE_SIZE_X))
    base_y = float(cfg.get("warehouse_size_y", WAREHOUSE_BASE_SIZE_Y))
    sx = (float(WAREHOUSE_SIZE_X) / base_x) if abs(base_x) > 1e-9 else 1.0
    sy = (float(WAREHOUSE_SIZE_Y) / base_y) if abs(base_y) > 1e-9 else 1.0
    return sx, sy


def _floor_spawn_half_extents(loader, safety_margin_m=FLOOR_SPAWN_SAFETY_MARGIN_M):
    tile_x, tile_y, _ = loader.model_size(CONVEYOR_ASSETS["floor"], UNIFORM_SCALE)
    margin_x = tile_x * FLOOR_INNER_MARGIN_TILES
    margin_y = tile_y * FLOOR_INNER_MARGIN_TILES
    floor_half_x = (WAREHOUSE_SIZE_X - (2.0 * margin_x)) * 0.5
    floor_half_y = (WAREHOUSE_SIZE_Y - (2.0 * margin_y)) * 0.5
    safe_half_x = max(0.5, floor_half_x - float(safety_margin_m))
    safe_half_y = max(0.5, floor_half_y - float(safety_margin_m))
    return safe_half_x, safe_half_y


def _truck_extra_gap_for_gate_state(gate_model_name):
    name = os.path.basename(str(gate_model_name)).lower()
    if "closed" in name:
        return LOADING_TRUCK_EXTRA_GAP_CLOSED
    if "half" in name:
        return LOADING_TRUCK_EXTRA_GAP_HALF
    return 0.0


def _estimate_loading_truck_along_extent_m(loading_slot):
    cache_key = (str(loading_slot), tuple(float(v) for v in LOADING_TRUCK_SCALE_XYZ))
    if cache_key in _LOADING_TRUCK_ALONG_EXTENT_CACHE:
        return _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key]

    if not os.path.exists(VEHICLE_DIR):
        _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key] = 0.0
        return 0.0
    model_name = LOADING_TRUCK_MODELS[0] if LOADING_TRUCK_MODELS else ""
    model_path = os.path.join(VEHICLE_DIR, model_name) if model_name else ""
    if not model_path or not os.path.exists(model_path):
        _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key] = 0.0
        return 0.0

    sx_scale, sy_scale, _sz_scale = LOADING_TRUCK_SCALE_XYZ
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")
    found = False
    with open(model_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            x = float(parts[1])
            z = float(parts[3])
            px = x * sx_scale
            py = (-z) * sy_scale
            min_x = min(min_x, px)
            max_x = max(max_x, px)
            min_y = min(min_y, py)
            max_y = max(max_y, py)
            found = True

    if not found:
        _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key] = 0.0
        return 0.0

    size_x = max_x - min_x
    size_y = max_y - min_y
    yaw = math.radians(dock_inward_yaw_for_slot(loading_slot))
    c = abs(math.cos(yaw))
    s = abs(math.sin(yaw))
    ex = (c * size_x) + (s * size_y)
    ey = (s * size_x) + (c * size_y)
    along_extent = ex if loading_slot in ("north", "south") else ey
    along_extent = max(0.0, float(along_extent))
    _LOADING_TRUCK_ALONG_EXTENT_CACHE[cache_key] = along_extent
    return along_extent


# ---------------------------------------------------------------------------
# Area-layout geometry (shared by structure.py + layout.py)
# ---------------------------------------------------------------------------
def _loading_marker_xy_size(size_pair, loading_side):
    depth = float(size_pair[0])
    span = float(size_pair[1])
    if loading_side in ("north", "south"):
        return span, depth
    return depth, span


def _rect_bounds(cx, cy, sx, sy):
    return (cx - sx * 0.5, cx + sx * 0.5, cy - sy * 0.5, cy + sy * 0.5)


def _candidate_rect_bounds(candidate):
    cached = candidate.get("_rect_bounds")
    if cached is None:
        cached = _rect_bounds(
            candidate["cx"], candidate["cy"], candidate["sx"], candidate["sy"]
        )
        candidate["_rect_bounds"] = cached
    return cached


def _rects_overlap(a, b, gap):
    a_bounds = a.get("_rect_bounds")
    if a_bounds is None:
        a_bounds = _candidate_rect_bounds(a)
    b_bounds = b.get("_rect_bounds")
    if b_bounds is None:
        b_bounds = _candidate_rect_bounds(b)
    a_min_x, a_max_x, a_min_y, a_max_y = a_bounds
    b_min_x, b_max_x, b_min_y, b_max_y = b_bounds
    return not (
        (a_max_x + gap) <= b_min_x
        or (b_max_x + gap) <= a_min_x
        or (a_max_y + gap) <= b_min_y
        or (b_max_y + gap) <= a_min_y
    )


def _size_fits_half_span(sx, sy, half_x, half_y, margin):
    max_sx = 2.0 * (half_x - margin)
    max_sy = 2.0 * (half_y - margin)
    return float(sx) <= max_sx + 1e-6 and float(sy) <= max_sy + 1e-6


def _sample_random_center(rng, sx, sy, floor_half_x, floor_half_y, margin):
    min_x = -floor_half_x + margin + (sx * 0.5)
    max_x = floor_half_x - margin - (sx * 0.5)
    min_y = -floor_half_y + margin + (sy * 0.5)
    max_y = floor_half_y - margin - (sy * 0.5)
    if max_x < min_x or max_y < min_y:
        return 0.0, 0.0
    return rng.uniform(min_x, max_x), rng.uniform(min_y, max_y)


def _wall_along_limits(wall, sx, sy, half_x, half_y, margin):
    if wall in ("north", "south"):
        return (
            -half_x + margin + (sx * 0.5),
            half_x - margin - (sx * 0.5),
        )
    return (
        -half_y + margin + (sy * 0.5),
        half_y - margin - (sy * 0.5),
    )


def _wall_attached_center(wall, along, sx, sy, half_x, half_y, margin):
    if wall == "north":
        return along, half_y - margin - (sy * 0.5)
    if wall == "south":
        return along, -half_y + margin + (sy * 0.5)
    if wall == "east":
        return half_x - margin - (sx * 0.5), along
    if wall == "west":
        return -half_x + margin + (sx * 0.5), along
    raise ValueError(f"Unknown wall: {wall}")


def _orient_dims_long_side_on_wall(wall, sx, sy):
    if wall in ("north", "south"):
        return (max(sx, sy), min(sx, sy))
    return (min(sx, sy), max(sx, sy))


def _attached_wall_from_area_bounds(area_sx, area_sy, area_cx, area_cy):
    half_x = WAREHOUSE_SIZE_X * 0.5
    half_y = WAREHOUSE_SIZE_Y * 0.5
    dist_to_wall = {
        "north": abs(half_y - (area_cy + area_sy * 0.5)),
        "south": abs((area_cy - area_sy * 0.5) + half_y),
        "east": abs(half_x - (area_cx + area_sx * 0.5)),
        "west": abs((area_cx - area_sx * 0.5) + half_x),
    }
    min_dist = min(dist_to_wall.values())
    near = [w for w, d in dist_to_wall.items() if abs(d - min_dist) <= 1e-6]
    if len(near) == 1:
        return near[0]
    if area_sx > area_sy:
        preferred = [w for w in near if w in ("north", "south")]
    elif area_sy > area_sx:
        preferred = [w for w in near if w in ("east", "west")]
    else:
        preferred = []
    if preferred:
        return preferred[0]
    return near[0]


# ---------------------------------------------------------------------------
# Window utilities (used by structure.py wall building)
# ---------------------------------------------------------------------------
def mirrored_window_indices(segment_count):
    if segment_count <= 5:
        return set()
    if segment_count >= 12:
        picks = [2, segment_count // 2, segment_count - 3]
    elif segment_count >= 8:
        picks = [2, segment_count - 3]
    else:
        picks = [segment_count // 2]
    return {i for i in picks if 0 < i < (segment_count - 1)}


def mirrored_wide_window_starts(segment_count, span_steps, seed_key):
    if span_steps <= 1 or segment_count < (span_steps + 6):
        return []
    rng = random.Random(seed_key + segment_count * 97 + span_steps * 13)
    shift = rng.choice((-1, 0, 1))
    starts = []
    if segment_count >= 20:
        left = (segment_count // 4) - (span_steps // 2) + shift
        left = max(1, min(left, segment_count - (2 * span_steps) - 2))
        right = segment_count - span_steps - left
        if right - left >= span_steps + 1:
            starts = [left, right]
        else:
            starts = [
                max(
                    1,
                    min(
                        segment_count // 2 - (span_steps // 2),
                        segment_count - span_steps - 1,
                    ),
                )
            ]
    elif segment_count >= 12:
        center = segment_count // 2 - (span_steps // 2)
        starts = [max(1, min(center, segment_count - span_steps - 1))]
    out = []
    for s in starts:
        s = max(1, min(s, segment_count - span_steps - 1))
        if s not in out:
            out.append(s)
    return sorted(out)


def _indices_blocked_by_doors(along_values, door_centers, door_span):
    blocked = set()
    if not along_values or not door_centers:
        return blocked
    if len(along_values) >= 2:
        step = abs(float(along_values[1]) - float(along_values[0]))
    else:
        step = max(1e-6, float(door_span))
    seg_half = step * 0.5
    door_half = float(door_span) * 0.5
    for idx, along in enumerate(along_values):
        seg_lo = float(along) - seg_half
        seg_hi = float(along) + seg_half
        for c in door_centers:
            door_lo = float(c) - door_half
            door_hi = float(c) + door_half
            if (seg_hi > (door_lo + 1e-6)) and (seg_lo < (door_hi - 1e-6)):
                blocked.add(idx)
                break
    return blocked


def _merge_spans_1d(spans, eps=1e-6):
    if not spans:
        return []
    ordered = sorted(
        (float(lo), float(hi)) for lo, hi in spans if float(hi) > float(lo) + eps
    )
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for lo, hi in ordered[1:]:
        if lo <= merged[-1][1] + eps:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def _subtract_spans_1d(base_spans, cut_spans, eps=1e-6):
    if not base_spans:
        return []
    base_merged = _merge_spans_1d(base_spans, eps=eps)
    cut_merged = _merge_spans_1d(cut_spans, eps=eps)
    if not cut_merged:
        return base_merged
    out = []
    for blo, bhi in base_merged:
        segments = [(blo, bhi)]
        for clo, chi in cut_merged:
            next_segments = []
            for slo, shi in segments:
                if chi <= slo + eps or clo >= shi - eps:
                    next_segments.append((slo, shi))
                    continue
                if clo > slo + eps:
                    next_segments.append((slo, min(shi, clo)))
                if chi < shi - eps:
                    next_segments.append((max(slo, chi), shi))
            segments = next_segments
            if not segments:
                break
        for slo, shi in segments:
            if shi > slo + eps:
                out.append((slo, shi))
    return _merge_spans_1d(out, eps=eps)


def _filter_mirrored_single_windows(candidate_indices, blocked_indices, segment_count):
    out = set()
    for i in sorted(candidate_indices):
        j = segment_count - 1 - i
        if i > j:
            continue
        if i == j:
            if i not in blocked_indices:
                out.add(i)
            continue
        if i not in blocked_indices and j not in blocked_indices:
            out.add(i)
            out.add(j)
    return out


def _span_is_clear(start_idx, span_steps, blocked_indices):
    for k in range(span_steps):
        if (start_idx + k) in blocked_indices:
            return False
    return True


def _filter_mirrored_wide_windows(
    candidate_starts, span_steps, blocked_indices, segment_count
):
    if span_steps <= 1:
        return sorted(candidate_starts)
    min_start = 1
    max_start = segment_count - span_steps - 1
    out = set()
    for s in sorted(candidate_starts):
        s = max(min_start, min(s, max_start))
        m = segment_count - span_steps - s
        m = max(min_start, min(m, max_start))
        if s > m:
            continue
        if s == m:
            if _span_is_clear(s, span_steps, blocked_indices):
                out.add(s)
            continue
        if _span_is_clear(s, span_steps, blocked_indices) and _span_is_clear(
            m, span_steps, blocked_indices
        ):
            out.add(s)
            out.add(m)
    return sorted(out)


# ---------------------------------------------------------------------------
# Cache clearing for build reset
# ---------------------------------------------------------------------------
def clear_build_caches():
    _MESH_VISUAL_SHAPE_CACHE.clear()
    _MESH_COLLISION_SHAPE_CACHE.clear()
    _RESOLVED_MESH_PATH_CACHE.clear()
    _TEXTURE_CACHE.clear()
