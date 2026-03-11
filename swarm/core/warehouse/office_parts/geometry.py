from ._shared import *


def build_floor(loader):
    tile_model = ASSETS["floor_tile"]
    tile_x, tile_y, _ = loader.model_size(tile_model)
    nx = round(FLOOR_SIZE / tile_x)
    ny = round(FLOOR_SIZE / tile_y)
    if abs(nx * tile_x - FLOOR_SIZE) > 1e-6 or abs(ny * tile_y - FLOOR_SIZE) > 1e-6:
        raise ValueError(
            f"Uniform scale {UNIFORM_SCALE} does not tile {FLOOR_SIZE}x{FLOOR_SIZE} exactly "
            f"with {tile_model} (tile {tile_x:.3f}x{tile_y:.3f})."
        )
    start_x = -FLOOR_SIZE / 2.0 + tile_x / 2.0
    start_y = -FLOOR_SIZE / 2.0 + tile_y / 2.0
    for ix in range(nx):
        for iy in range(ny):
            x = start_x + ix * tile_x
            y = start_y + iy * tile_y
            loader.spawn(tile_model, x, y, yaw_deg=0, floor_z=0.0)
    return loader.top_height_from_floor(tile_model)


def slot_config(slot):
    if slot == "north":
        return {"normal": (0.0, -1.0), "tangent": (1.0, 0.0), "wall_yaw": 0.0}
    if slot == "south":
        return {"normal": (0.0, 1.0), "tangent": (1.0, 0.0), "wall_yaw": 0.0}
    if slot == "east":
        return {"normal": (-1.0, 0.0), "tangent": (0.0, 1.0), "wall_yaw": 90.0}
    if slot == "west":
        return {"normal": (1.0, 0.0), "tangent": (0.0, 1.0), "wall_yaw": 90.0}
    raise ValueError(f"Unknown slot: {slot}")


def snap_cardinal(yaw_deg):
    return (round(yaw_deg / 90.0) * 90.0) % 360.0


def snap_octant(yaw_deg):
    return (round(yaw_deg / 45.0) * 45.0) % 360.0


def inward_facing_yaw(loader, model_name, x, y, diagonal=False):
    raw_yaw = loader.yaw_to_face_point(model_name, x, y, ROOM_CENTER[0], ROOM_CENTER[1])
    return snap_octant(raw_yaw) if diagonal else snap_cardinal(raw_yaw)


def slot_inward_wall_yaw(loader, model_name, slot):
    _ = loader
    _ = model_name
    return float(wall_face_yaw(slot))


def wall_face_yaw(slot):
    return {"north": 180.0, "south": 0.0, "east": 90.0, "west": 270.0}[slot]


def wall_tangent_yaw(slot):
    return {"north": 0.0, "south": 0.0, "east": 90.0, "west": 90.0}[slot]


def desk_lr_along_offsets(slot, separation):
    if slot in ("north", "west"):
        return -separation, separation
    return separation, -separation


def slot_xy(slot, along, inward):
    cfg = slot_config(slot)
    nx, ny = cfg["normal"]
    tx, ty = cfg["tangent"]
    edge = FLOOR_SIZE / 2.0
    cx0, cy0 = ROOM_CENTER
    if slot == "north":
        bx, by = cx0, cy0 + edge
    elif slot == "south":
        bx, by = cx0, cy0 - edge
    elif slot == "east":
        bx, by = cx0 + edge, cy0
    else:
        bx, by = cx0 - edge, cy0
    return bx + tx * along + nx * inward, by + ty * along + ny * inward


def corner_points():
    cx0, cy0 = ROOM_CENTER
    return [
        (cx0 - FLOOR_SIZE / 2.0 + 1.05, cy0 + FLOOR_SIZE / 2.0 - 1.05),
        (cx0 + FLOOR_SIZE / 2.0 - 1.05, cy0 + FLOOR_SIZE / 2.0 - 1.05),
        (cx0 - FLOOR_SIZE / 2.0 + 1.05, cy0 - FLOOR_SIZE / 2.0 + 1.05),
        (cx0 + FLOOR_SIZE / 2.0 - 1.05, cy0 - FLOOR_SIZE / 2.0 + 1.05),
    ]


def nearest_corner_index(x, y, corners):
    best_i = 0
    best_d2 = None
    for i, (cx, cy) in enumerate(corners):
        d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def workstation_l_corner_index(slot):
    return {"north": 1, "south": 3, "east": 1, "west": 0}[slot]


def adjacent_slot_for_l(slot):
    return {"north": "east", "south": "east", "east": "north", "west": "north"}[slot]


def along_sign_for_corner(slot, corner_idx):
    sign_map = {
        "north": {0: -1.0, 1: 1.0},
        "south": {2: -1.0, 3: 1.0},
        "east": {3: -1.0, 1: 1.0},
        "west": {2: -1.0, 0: 1.0},
    }
    return sign_map[slot].get(corner_idx, 1.0)


def workstation_right_is_positive_along(slot):
    return slot in ("north", "west")


def _corner_trim_from_model(loader):
    if not ENABLE_PERIMETER_WALL_CORNERS:
        return 0.0
    corner_model = ASSETS.get("wall_corner", "")
    if not corner_model:
        return 0.0
    corner_path = loader._asset_path(corner_model)
    if not os.path.exists(corner_path):
        return 0.0
    wall_x, wall_y, _ = loader.model_size(ASSETS["wall"])
    wall_thickness = min(float(wall_x), float(wall_y))
    min_v, max_v = loader._mesh_bounds(corner_model, UNIFORM_SCALE)
    anchor_local_x = float(min_v[0]) + (wall_thickness * 0.5)
    anchor_local_y = float(max_v[1]) - (wall_thickness * 0.5)
    inward_x = max(0.0, float(max_v[0]) - anchor_local_x)
    inward_y = max(0.0, anchor_local_y - float(min_v[1]))
    return max(inward_x, inward_y) + max(0.0, float(PERIMETER_WALL_CORNER_JOIN_GAP_M))


def _wall_segment_plan(loader):
    wall_len, _, _ = loader.model_size(ASSETS["wall"])
    if wall_len <= 1e-6:
        raise ValueError("Invalid wall length from wall.obj")
    if not ENABLE_PERIMETER_WALL_CORNERS:
        nseg = round(FLOOR_SIZE / wall_len)
        if abs(nseg * wall_len - FLOOR_SIZE) > 1e-6:
            raise ValueError(
                f"Wall model does not tile {FLOOR_SIZE:.2f}m exactly at scale {UNIFORM_SCALE}."
            )
        start = -FLOOR_SIZE / 2.0 + wall_len / 2.0
        return [
            (start + i * wall_len, float(PERIMETER_WALL_ALONG_SCALE))
            for i in range(int(nseg))
        ]
    trim = _corner_trim_from_model(loader)
    inner_span = FLOOR_SIZE - (2.0 * trim)
    if inner_span <= 0.2:
        raise ValueError("Corner trim too large for office wall span.")
    nseg = max(1, round(inner_span / wall_len))
    seg_len = inner_span / float(nseg)
    along_scale = (seg_len / wall_len) * float(PERIMETER_WALL_ALONG_SCALE)
    start = -inner_span / 2.0 + seg_len / 2.0
    return [(start + i * seg_len, along_scale) for i in range(int(nseg))]


def spawn_walls(loader, floor_top_z):
    seg_plan = _wall_segment_plan(loader)
    for slot in WALL_SLOTS:
        wall_yaw = slot_inward_wall_yaw(loader, ASSETS["wall"], slot)
        for along, along_scale in seg_plan:
            x, y = slot_xy(slot, along, inward=0.0)
            loader.spawn(
                ASSETS["wall"],
                x,
                y,
                yaw_deg=wall_yaw,
                floor_z=floor_top_z,
                scale=(UNIFORM_SCALE * along_scale, UNIFORM_SCALE, UNIFORM_SCALE),
            )
    spawn_wall_corners(loader, floor_top_z)


def spawn_wall_corners(loader, floor_top_z):
    if not ENABLE_PERIMETER_WALL_CORNERS:
        return
    corner_model = ASSETS.get("wall_corner", "")
    if not corner_model:
        return
    corner_path = loader._asset_path(corner_model)
    if not os.path.exists(corner_path):
        return
    min_v, max_v = loader._mesh_bounds(corner_model, UNIFORM_SCALE)
    cx_local = float((min_v[0] + max_v[0]) * 0.5)
    cy_local = float((min_v[1] + max_v[1]) * 0.5)
    wall_x, wall_y, _ = loader.model_size(ASSETS["wall"])
    wall_thickness = min(float(wall_x), float(wall_y))
    anchor_local_x = float(min_v[0]) + (wall_thickness * 0.5)
    anchor_local_y = float(max_v[1]) - (wall_thickness * 0.5)
    anchor_off_x = anchor_local_x - cx_local
    anchor_off_y = anchor_local_y - cy_local
    edge = (FLOOR_SIZE * 0.5) + float(PERIMETER_WALL_CORNER_OUTWARD_EPS)
    cx0, cy0 = ROOM_CENTER
    corner_specs = (
        ("nw", -1.0, 1.0, 0.0),
        ("ne", 1.0, 1.0, 270.0),
        ("sw", -1.0, -1.0, 90.0),
        ("se", 1.0, -1.0, 180.0),
    )
    for _name, sx, sy, yaw in corner_specs:
        anchor_x = cx0 + (sx * edge)
        anchor_y = cy0 + (sy * edge)
        yaw_rad = math.radians(yaw)
        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        rox = anchor_off_x * c - anchor_off_y * s
        roy = anchor_off_x * s + anchor_off_y * c
        x = anchor_x - rox
        y = anchor_y - roy
        loader.spawn(corner_model, x, y, yaw_deg=yaw, floor_z=floor_top_z)


def spawn_walls_with_entry(
    loader, floor_top_z, entry_slot, door_along=0.0, open_mode=ENTRY_WALL_OPENING_MODE
):
    if entry_slot not in WALL_SLOTS:
        raise ValueError(f"Unknown entry slot: {entry_slot}")
    seg_plan = _wall_segment_plan(loader)
    seg_along = [a for a, _s in seg_plan]
    if not seg_along:
        return
    door_idx = min(
        range(len(seg_along)), key=lambda i: abs(seg_along[i] - float(door_along))
    )
    use_gap = str(open_mode).lower() == "gap"
    slot_yaw_cache = {}
    for slot in WALL_SLOTS:
        slot_yaw_cache[(slot, ASSETS["wall"])] = slot_inward_wall_yaw(
            loader, ASSETS["wall"], slot
        )
        slot_yaw_cache[(slot, ASSETS["wall_door"])] = slot_inward_wall_yaw(
            loader, ASSETS["wall_door"], slot
        )
        for i, (along, along_scale) in enumerate(seg_plan):
            if slot == entry_slot and i == door_idx:
                if use_gap:
                    continue
                model = ASSETS["wall_door"]
            else:
                model = ASSETS["wall"]
            x, y = slot_xy(slot, along, inward=0.0)
            wall_yaw = slot_yaw_cache[(slot, model)]
            loader.spawn(
                model,
                x,
                y,
                yaw_deg=wall_yaw,
                floor_z=floor_top_z,
                scale=(UNIFORM_SCALE * along_scale, UNIFORM_SCALE, UNIFORM_SCALE),
            )
    spawn_wall_corners(loader, floor_top_z)
