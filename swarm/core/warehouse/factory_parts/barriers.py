from ._shared import *


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

    spawned = 0
    for seg in segments_to_spawn:
        world_anchor = (seg["x"], seg["y"], float(floor_top_z))
        yaw_deg = float(seg["yaw"])
        _spawn_mesh_with_anchor(
            loader=barrier_loader,
            model_name=model_name,
            world_anchor_xyz=world_anchor,
            yaw_deg=yaw_deg,
            mesh_scale_xyz=FACTORY_BARRIER_SCALE_XYZ,
            local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
            cli=cli,
            with_collision=FACTORY_BARRIER_WITH_COLLISION,
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
