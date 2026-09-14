# The MIT License (MIT)
# Copyright © 2026 Swarm

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Wall-slot geometry for the office room: local frames, corner points and wall spawning."""

from ._shared import *


def slot_config(slot):
    """Inward normal, tangent and wall yaw of one of the four perimeter slots."""
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
    """Round a yaw to the nearest quarter turn, wrapped into [0, 360)."""
    return (round(yaw_deg / 90.0) * 90.0) % 360.0


def snap_octant(yaw_deg):
    """Round a yaw to the nearest 45 degrees, wrapped into [0, 360)."""
    return (round(yaw_deg / 45.0) * 45.0) % 360.0


def wall_face_yaw(slot):
    """Yaw that turns a wall model's front toward the room interior."""
    return {"north": 180.0, "south": 0.0, "east": 90.0, "west": 270.0}[slot]


def wall_tangent_yaw(slot):
    """Yaw of the axis a wall runs along: 0 on the north and south sides, 90 on east and west."""
    return {"north": 0.0, "south": 0.0, "east": 90.0, "west": 90.0}[slot]


def desk_lr_along_offsets(slot, separation):
    """Left and right desk positions along the wall, ordered for a worker seated facing it."""
    if slot in ("north", "west"):
        return -separation, separation
    return separation, -separation


def slot_xy(slot, along, inward):
    """World x, y of a point `along` a wall and `inward` from it, measured off that wall's midpoint."""
    cfg = slot_config(slot)
    nx, ny = cfg["normal"]
    tx, ty = cfg["tangent"]
    edge = FLOOR_SIZE[0] / 2.0
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
    """The four room corners, each inset 1.05 m from the walls."""
    cx0, cy0 = ROOM_CENTER
    return [
        (cx0 - FLOOR_SIZE[0] / 2.0 + 1.05, cy0 + FLOOR_SIZE[0] / 2.0 - 1.05),
        (cx0 + FLOOR_SIZE[0] / 2.0 - 1.05, cy0 + FLOOR_SIZE[0] / 2.0 - 1.05),
        (cx0 - FLOOR_SIZE[0] / 2.0 + 1.05, cy0 - FLOOR_SIZE[0] / 2.0 + 1.05),
        (cx0 + FLOOR_SIZE[0] / 2.0 - 1.05, cy0 - FLOOR_SIZE[0] / 2.0 + 1.05),
    ]


def nearest_corner_index(x, y, corners):
    """Position in `corners` of the point closest to (x, y) by squared distance."""
    best_i = 0
    best_d2 = None
    for i, (cx, cy) in enumerate(corners):
        d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_i = i
    return best_i


def workstation_l_corner_index(slot):
    """Which room corner an L-shaped workstation on this wall wraps into."""
    return {"north": 1, "south": 3, "east": 1, "west": 0}[slot]


def adjacent_slot_for_l(slot):
    """The perpendicular wall the short arm of an L workstation runs against."""
    return {"north": "east", "south": "east", "east": "north", "west": "north"}[slot]


def along_sign_for_corner(slot, corner_idx):
    """Direction to walk the wall to reach that corner, +1 for corners not on this wall."""
    sign_map = {
        "north": {0: -1.0, 1: 1.0},
        "south": {2: -1.0, 3: 1.0},
        "east": {3: -1.0, 1: 1.0},
        "west": {2: -1.0, 0: 1.0},
    }
    return sign_map[slot].get(corner_idx, 1.0)


def workstation_right_is_positive_along(slot):
    """True where a seated worker's right hand points up the wall's tangent."""
    return slot in ("north", "west")


def _corner_trim_from_model(loader):
    """Metres a corner piece eats off each wall run, join gap included; 0 when corners are disabled."""
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
    """Centre offset and length scale of each segment tiling one side, corner trim taken off first."""
    wall_len, _, _ = loader.model_size(ASSETS["wall"])
    if wall_len <= 1e-6:
        raise ValueError("Invalid wall length from wall.obj")
    if not ENABLE_PERIMETER_WALL_CORNERS:
        nseg = round(FLOOR_SIZE[0] / wall_len)
        if abs(nseg * wall_len - FLOOR_SIZE[0]) > 1e-6:
            raise ValueError(
                f"Wall model does not tile {FLOOR_SIZE[0]:.2f}m exactly at scale {UNIFORM_SCALE}."
            )
        start = -FLOOR_SIZE[0] / 2.0 + wall_len / 2.0
        return [
            (start + i * wall_len, float(PERIMETER_WALL_ALONG_SCALE))
            for i in range(int(nseg))
        ]
    trim = _corner_trim_from_model(loader)
    inner_span = FLOOR_SIZE[0] - (2.0 * trim)
    if inner_span <= 0.2:
        raise ValueError("Corner trim too large for office wall span.")
    nseg = max(1, round(inner_span / wall_len))
    seg_len = inner_span / float(nseg)
    along_scale = (seg_len / wall_len) * float(PERIMETER_WALL_ALONG_SCALE)
    start = -inner_span / 2.0 + seg_len / 2.0
    return [(start + i * seg_len, along_scale) for i in range(int(nseg))]


def spawn_wall_corners(loader, floor_top_z):
    """Place the four corner pieces so each one's wall-centreline anchor lands on the room corner."""
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
    edge = (FLOOR_SIZE[0] * 0.5) + float(PERIMETER_WALL_CORNER_OUTWARD_EPS)
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
    """Tile all four sides, with a door model or an open gap at the segment nearest `door_along`."""
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
        slot_yaw_cache[(slot, ASSETS["wall"])] = float(wall_face_yaw(slot))
        slot_yaw_cache[(slot, ASSETS["wall_door"])] = float(wall_face_yaw(slot))
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
