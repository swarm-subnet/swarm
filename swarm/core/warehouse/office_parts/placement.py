from ._shared import *
from .geometry import (
    build_floor,
    corner_points,
    desk_lr_along_offsets,
    nearest_corner_index,
    slot_config,
    slot_xy,
    snap_cardinal,
    snap_octant,
    spawn_walls_with_entry,
    wall_face_yaw,
    wall_tangent_yaw,
    workstation_right_is_positive_along,
)


def place_entry_wall(loader, floor_top_z, slot, seed, corners=None, spawn_doorway=True):
    if corners is None:
        corners = corner_points()
    rng = random.Random(seed + 5100 + WALL_SLOTS.index(slot))
    face_yaw = wall_face_yaw(slot)
    if spawn_doorway:
        door_inward = 0.58
        dx, dy = slot_xy(slot, 0.0, inward=door_inward)
        loader.spawn(ASSETS["doorway"], dx, dy, yaw_deg=face_yaw, floor_z=floor_top_z)
    mirror = -1.0 if rng.random() < 0.5 else 1.0
    coat_along = -2.55 * mirror
    plant_along = 2.55 * mirror
    cx, cy = slot_xy(slot, coat_along, inward=1.00)
    px, py = slot_xy(slot, plant_along, inward=1.02)
    loader.spawn(
        ASSETS["entry_coat_rack"], cx, cy, yaw_deg=face_yaw, floor_z=floor_top_z
    )
    loader.spawn(ASSETS["entry_plant"], px, py, yaw_deg=face_yaw, floor_z=floor_top_z)
    forbidden_by_corner = {}
    coat_corner = nearest_corner_index(cx, cy, corners)
    plant_corner = nearest_corner_index(px, py, corners)
    forbidden_by_corner.setdefault(coat_corner, set()).add("tall_accent")
    forbidden_by_corner.setdefault(plant_corner, set()).add("entry_plant")
    return forbidden_by_corner


def place_workstations_wall(loader, floor_top_z, slot, seed):
    rng = random.Random(seed + 1700 + WALL_SLOTS.index(slot))
    desk_w, desk_d, _ = loader.model_size(ASSETS["desk"])
    _, chair_d, _ = loader.model_size(ASSETS["desk_chair"])
    right_corner_model = (
        ASSETS["desk_corner"] if rng.random() < 0.65 else ASSETS["desk"]
    )
    if workstation_right_is_positive_along(slot):
        row_models = [
            ASSETS["desk"],
            ASSETS["desk"],
            ASSETS["desk"],
            ASSETS["desk"],
            right_corner_model,
        ]
    else:
        row_models = [
            right_corner_model,
            ASSETS["desk"],
            ASSETS["desk"],
            ASSETS["desk"],
            ASSETS["desk"],
        ]
    widths = [loader.model_size(m)[0] for m in row_models]
    gap = 0.14
    max_span = FLOOR_SIZE - 1.9
    row_span = sum(widths) + gap * (len(widths) - 1)
    if row_span > max_span:
        gap = max(0.08, (max_span - sum(widths)) / (len(widths) - 1))
        row_span = sum(widths) + gap * (len(widths) - 1)
    centers = []
    cursor = -row_span / 2.0
    for w in widths:
        centers.append(cursor + w / 2.0)
        cursor += w + gap
    base_desk_inward = 1.00
    base_chair_offset = (desk_d / 2.0) + (chair_d / 2.0) + 0.12
    base_monitor_offset = -0.16
    base_keyboard_offset = 0.12
    desk_yaw = wall_face_yaw(slot)
    inward_normal = slot_config(slot)["normal"]
    back_regular = loader.back_offset(
        ASSETS["desk"], desk_yaw, inward_normal, scale=UNIFORM_SCALE
    )
    target_back_inward = base_desk_inward - back_regular
    unified_chair_inward = base_desk_inward + base_chair_offset
    unified_monitor_inward = base_desk_inward + base_monitor_offset
    unified_keyboard_inward = base_desk_inward + base_keyboard_offset

    def spawn_station(st_slot, along, desk_model, chair_along_nudge=0.0):
        key_along_offset, mouse_along_offset = desk_lr_along_offsets(
            st_slot, separation=0.28
        )
        st_desk_yaw = wall_face_yaw(st_slot)
        st_chair_yaw = (st_desk_yaw + 180.0) % 360.0
        desk_h = loader.model_size(desk_model)[2]
        st_inward_normal = slot_config(st_slot)["normal"]
        back_model = loader.back_offset(
            desk_model, st_desk_yaw, st_inward_normal, scale=UNIFORM_SCALE
        )
        desk_inward = target_back_inward + back_model
        chair_inward = unified_chair_inward
        monitor_inward = unified_monitor_inward
        keyboard_inward = unified_keyboard_inward
        logical_along = along
        dx, dy = slot_xy(st_slot, along, inward=desk_inward)
        loader.spawn(
            desk_model,
            dx,
            dy,
            yaw_deg=st_desk_yaw,
            floor_z=floor_top_z,
            scale=UNIFORM_SCALE,
        )
        cx, cy = slot_xy(
            st_slot, logical_along + chair_along_nudge, inward=chair_inward
        )
        loader.spawn(
            ASSETS["desk_chair"], cx, cy, yaw_deg=st_chair_yaw, floor_z=floor_top_z
        )
        mx, my = slot_xy(st_slot, logical_along, inward=monitor_inward)
        loader.spawn(
            ASSETS["monitor"],
            mx,
            my,
            yaw_deg=st_desk_yaw,
            floor_z=floor_top_z,
            extra_z=desk_h + 0.01,
        )
        kx, ky = slot_xy(
            st_slot, logical_along + key_along_offset, inward=keyboard_inward
        )
        loader.spawn(
            ASSETS["keyboard"],
            kx,
            ky,
            yaw_deg=st_desk_yaw,
            floor_z=floor_top_z,
            extra_z=desk_h + 0.01,
        )
        msx, msy = slot_xy(
            st_slot, logical_along + mouse_along_offset, inward=keyboard_inward
        )
        loader.spawn(
            ASSETS["mouse"],
            msx,
            msy,
            yaw_deg=st_desk_yaw,
            floor_z=floor_top_z,
            extra_z=desk_h + 0.01,
        )

    for i, (model, along) in enumerate(zip(row_models, centers)):
        chair_along_nudge = 0.0
        if model == ASSETS["desk_corner"]:
            if i == 0:
                chair_along_nudge = +0.16
            elif i == len(row_models) - 1:
                chair_along_nudge = -0.16
        spawn_station(slot, along, model, chair_along_nudge=chair_along_nudge)
    return None


def place_files_wall(loader, floor_top_z, slot, seed):
    rng = random.Random(seed + 3100 + WALL_SLOTS.index(slot))
    face_yaw = wall_face_yaw(slot)
    tangent_yaw = wall_tangent_yaw(slot)
    shelf_variants = [
        [
            ASSETS["bookcase_open"],
            ASSETS["bookcase_open"],
            ASSETS["bookcase_closed"],
            ASSETS["bookcase_wide"],
        ],
        [
            ASSETS["bookcase_open"],
            ASSETS["bookcase_closed"],
            ASSETS["bookcase_open"],
            ASSETS["bookcase_open_low"],
            ASSETS["bookcase_open"],
        ],
        [
            ASSETS["bookcase_open"],
            ASSETS["bookcase_wide"],
            ASSETS["bookcase_open_low"],
            ASSETS["bookcase_open"],
            ASSETS["bookcase_closed"],
        ],
    ]
    shelves = rng.choice(shelf_variants)
    shelf_gap = 0.08
    shelf_inward = 0.72
    widths = [loader.model_size(m)[0] for m in shelves]
    total_span = sum(widths) + shelf_gap * (len(shelves) - 1)
    start = -total_span / 2.0
    centers = []
    cursor = start
    for w in widths:
        centers.append(cursor + w / 2.0)
        cursor += w + shelf_gap
    shelf_profiles = {
        ASSETS["bookcase_open_low"]: {
            "max_per_row": 1,
            "book_inward": shelf_inward + 0.00,
        },
        ASSETS["bookcase_open"]: {"max_per_row": 1, "book_inward": shelf_inward + 0.00},
        ASSETS["bookcase_closed"]: {
            "max_per_row": 1,
            "book_inward": shelf_inward + 0.00,
        },
        ASSETS["bookcase_wide"]: {"max_per_row": 3, "book_inward": shelf_inward + 0.00},
    }
    for model, along in zip(shelves, centers):
        x, y = slot_xy(slot, along, inward=shelf_inward)
        loader.spawn(model, x, y, yaw_deg=face_yaw, floor_z=floor_top_z)
        profile = shelf_profiles.get(
            model, {"max_per_row": 1, "book_inward": shelf_inward + 0.00}
        )
        shelf_h = loader.model_size(model)[2]
        book_inward = profile["book_inward"]
        row_levels = list(loader.shelf_surface_levels(model))
        if model == ASSETS["bookcase_open_low"] and row_levels:
            row_levels = [row_levels[0]]
        if not row_levels:
            row_levels = [max(0.20, shelf_h * 0.30)]
        for li, level in enumerate(row_levels):
            max_per_row = profile["max_per_row"]
            count = 1 if max_per_row == 1 else rng.randint(1, max_per_row)
            if count == 1:
                row_offsets = [rng.choice([-0.14, 0.0, 0.14])]
            elif count == 2:
                row_offsets = [-0.20, 0.20]
            else:
                row_offsets = [-0.30, 0.0, 0.30]
            for oi, offset in enumerate(row_offsets):
                bx, by = slot_xy(slot, along + offset, inward=book_inward)
                flip = (li + oi) % 2 == 1
                byaw = face_yaw if not flip else (face_yaw + 180.0) % 360.0
                loader.spawn(
                    ASSETS["books"],
                    bx,
                    by,
                    yaw_deg=byaw,
                    floor_z=floor_top_z,
                    extra_z=level + 0.005,
                )
    left_box_along = (
        -total_span / 2.0 - loader.model_size(ASSETS["box"])[0] / 2.0 - 0.20
    )
    right_box_along = (
        total_span / 2.0 + loader.model_size(ASSETS["box_open"])[0] / 2.0 + 0.20
    )
    bx1, by1 = slot_xy(slot, left_box_along, inward=1.08)
    bx2, by2 = slot_xy(slot, right_box_along, inward=1.08)
    loader.spawn(ASSETS["box"], bx1, by1, yaw_deg=tangent_yaw, floor_z=floor_top_z)
    loader.spawn(ASSETS["box_open"], bx2, by2, yaw_deg=tangent_yaw, floor_z=floor_top_z)


def place_services_wall(loader, floor_top_z, slot, seed, corners=None):
    rng = random.Random(seed + 7300 + WALL_SLOTS.index(slot))
    cab_h = loader.model_size(ASSETS["cabinet"])[2]
    face_yaw = wall_face_yaw(slot)
    tangent_yaw = wall_tangent_yaw(slot)
    if corners is None:
        corners = corner_points()
    fridge_model = rng.choices(
        [ASSETS["fridge"], ASSETS["fridge_tall"], ASSETS["fridge_large"]],
        weights=[0.55, 0.35, 0.10],
        k=1,
    )[0]
    storage_model = rng.choice(
        [ASSETS["bookcase_closed_doors"], ASSETS["cabinet_tv_doors"]]
    )
    mirror = -1.0 if rng.random() < 0.5 else 1.0
    fridge_along = -1.90 * mirror
    cabinet_along = 0.00 * mirror
    storage_along = 1.90 * mirror
    trash_along = 2.95 * mirror
    plant_along = -3.05 * mirror
    fx, fy = slot_xy(slot, fridge_along, inward=0.94)
    loader.spawn(fridge_model, fx, fy, yaw_deg=face_yaw, floor_z=floor_top_z)
    cx, cy = slot_xy(slot, cabinet_along, inward=0.93)
    loader.spawn(ASSETS["cabinet"], cx, cy, yaw_deg=face_yaw, floor_z=floor_top_z)
    kx, ky = slot_xy(slot, cabinet_along, inward=0.88)
    loader.spawn(
        ASSETS["coffee_machine"],
        kx,
        ky,
        yaw_deg=face_yaw,
        floor_z=floor_top_z,
        extra_z=cab_h + 0.01,
    )
    sx, sy = slot_xy(slot, storage_along, inward=0.92)
    loader.spawn(storage_model, sx, sy, yaw_deg=face_yaw, floor_z=floor_top_z)
    tx, ty = slot_xy(slot, trash_along, inward=1.06)
    loader.spawn(ASSETS["trashcan"], tx, ty, yaw_deg=tangent_yaw, floor_z=floor_top_z)
    px, py = slot_xy(slot, plant_along, inward=1.02)
    loader.spawn(ASSETS["entry_plant"], px, py, yaw_deg=face_yaw, floor_z=floor_top_z)
    forbidden_by_corner = {}
    trash_corner = nearest_corner_index(tx, ty, corners)
    plant_corner = nearest_corner_index(px, py, corners)
    forbidden_by_corner.setdefault(trash_corner, set()).add("trashcan")
    forbidden_by_corner.setdefault(plant_corner, set()).add("entry_plant")
    return forbidden_by_corner


def place_corner_decor(
    loader,
    floor_top_z,
    seed,
    forbidden_styles_by_corner=None,
    blocked_corner_indices=None,
):
    rng = random.Random(seed + 9007)
    if forbidden_styles_by_corner is None:
        forbidden_styles_by_corner = {}
    if blocked_corner_indices is None:
        blocked_corner_indices = set()
    corners = corner_points()
    active_corner_indices = [
        i for i in range(len(corners)) if i not in blocked_corner_indices
    ]
    if len(active_corner_indices) > 3:
        rng.shuffle(active_corner_indices)
        active_corner_indices = active_corner_indices[:3]
    tall_asset = rng.choice([ASSETS["lamp_floor"], ASSETS["entry_coat_rack"]])
    styles = ["entry_plant", "trashcan", "tall_accent"]
    assigned = None
    for _ in range(64):
        trial = list(styles)
        rng.shuffle(trial)
        ok = True
        for ci, st in zip(active_corner_indices, trial):
            blocked = forbidden_styles_by_corner.get(ci, set())
            if st in blocked:
                ok = False
                break
        if ok:
            assigned = trial
            break
    if assigned is None:
        assigned = list(styles)
        rng.shuffle(assigned)
    for ci, style in zip(active_corner_indices, assigned):
        x, y = corners[ci]
        if style == "entry_plant":
            yaw = snap_octant(
                loader.yaw_to_face_point(
                    ASSETS["entry_plant"], x, y, ROOM_CENTER[0], ROOM_CENTER[1]
                )
            )
            loader.spawn(ASSETS["entry_plant"], x, y, yaw_deg=yaw, floor_z=floor_top_z)
            continue
        if style == "trashcan":
            yaw = snap_octant(
                loader.yaw_to_face_point(
                    ASSETS["trashcan"], x, y, ROOM_CENTER[0], ROOM_CENTER[1]
                )
            )
            loader.spawn(ASSETS["trashcan"], x, y, yaw_deg=yaw, floor_z=floor_top_z)
            continue
        yaw = snap_octant(
            loader.yaw_to_face_point(tall_asset, x, y, ROOM_CENTER[0], ROOM_CENTER[1])
        )
        loader.spawn(tall_asset, x, y, yaw_deg=yaw, floor_z=floor_top_z)


def build_center_meeting(loader, floor_top_z, seed):
    cx, cy = ROOM_CENTER
    rng = random.Random(seed + 5200)
    table_model = ASSETS["meeting_table"]
    chair_model = ASSETS["meeting_chair"]
    table_x, table_y, _ = loader.model_size(table_model)
    chair_x, chair_y, _ = loader.model_size(chair_model)
    horizontal = rng.random() < 0.5
    table_yaw = 0.0 if horizontal else 90.0
    len_axis = (1.0, 0.0) if horizontal else (0.0, 1.0)
    side_axis = (0.0, 1.0) if horizontal else (1.0, 0.0)
    overlap = 0.14
    separation = table_x - overlap
    t1x = cx - len_axis[0] * separation / 2.0
    t1y = cy - len_axis[1] * separation / 2.0
    t2x = cx + len_axis[0] * separation / 2.0
    t2y = cy + len_axis[1] * separation / 2.0
    loader.spawn(table_model, t1x, t1y, yaw_deg=table_yaw, floor_z=floor_top_z)
    loader.spawn(table_model, t2x, t2y, yaw_deg=table_yaw, floor_z=floor_top_z)
    table_length = table_x * 2.0 - overlap
    row_offset = (table_y / 2.0) + (chair_y / 2.0) + 0.08
    end_margin = max(chair_x * 0.60, 0.40)
    along_slots = [
        -(table_length / 2.0) + end_margin,
        0.0,
        (table_length / 2.0) - end_margin,
    ]
    for along in along_slots:
        tx = cx + len_axis[0] * along
        ty = cy + len_axis[1] * along
        x_top = tx + side_axis[0] * row_offset
        y_top = ty + side_axis[1] * row_offset
        yaw_top = loader.yaw_to_face_point(chair_model, x_top, y_top, tx, ty)
        loader.spawn(
            chair_model,
            x_top,
            y_top,
            yaw_deg=snap_cardinal(yaw_top),
            floor_z=floor_top_z,
        )
        x_bottom = tx - side_axis[0] * row_offset
        y_bottom = ty - side_axis[1] * row_offset
        yaw_bottom = loader.yaw_to_face_point(chair_model, x_bottom, y_bottom, tx, ty)
        loader.spawn(
            chair_model,
            x_bottom,
            y_bottom,
            yaw_deg=snap_cardinal(yaw_bottom),
            floor_z=floor_top_z,
        )
    head_offset = (table_length / 2.0) + (chair_y / 2.0) + 0.10
    x_left = cx - len_axis[0] * head_offset
    y_left = cy - len_axis[1] * head_offset
    x_right = cx + len_axis[0] * head_offset
    y_right = cy + len_axis[1] * head_offset
    yaw_left = loader.yaw_to_face_point(chair_model, x_left, y_left, cx, cy)
    yaw_right = loader.yaw_to_face_point(chair_model, x_right, y_right, cx, cy)
    loader.spawn(
        chair_model,
        x_left,
        y_left,
        yaw_deg=snap_cardinal(yaw_left),
        floor_z=floor_top_z,
    )
    loader.spawn(
        chair_model,
        x_right,
        y_right,
        yaw_deg=snap_cardinal(yaw_right),
        floor_z=floor_top_z,
    )

