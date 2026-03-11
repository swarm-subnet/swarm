from ._shared import *
from .geometry import corner_points, spawn_walls_with_entry
from .loader import AssetLoader
from .placement import (
    build_center_meeting,
    place_corner_decor,
    place_entry_wall,
    place_files_wall,
    place_services_wall,
    place_workstations_wall,
)


def wall_role_map(seed):
    rng = random.Random(int(seed) + 1201)
    slots = list(WALL_SLOTS)
    roles = list(WALL_ROLES)
    rng.shuffle(slots)
    rng.shuffle(roles)
    return {slot: role for slot, role in zip(slots, roles)}


def _embedded_office_role_map(seed, office_center_xy, entry_target_xy=None):
    cx, cy = office_center_xy
    blocked_slots = {
        "east" if cx >= 0.0 else "west",
        "north" if cy >= 0.0 else "south",
    }
    slots = list(WALL_SLOTS)
    roles = list(WALL_ROLES)
    allowed_entry_slots = [s for s in slots if s not in blocked_slots]
    if not allowed_entry_slots:
        return wall_role_map(int(seed) + EMBEDDED_OFFICE_SEED_OFFSET)
    if entry_target_xy is not None:
        to_target_x = float(entry_target_xy[0]) - float(cx)
        to_target_y = float(entry_target_xy[1]) - float(cy)
    else:
        to_target_x = -float(cx)
        to_target_y = -float(cy)
    if abs(to_target_x) >= abs(to_target_y):
        preferred_entry_slot = "east" if to_target_x >= 0.0 else "west"
    else:
        preferred_entry_slot = "north" if to_target_y >= 0.0 else "south"
    if preferred_entry_slot in allowed_entry_slots:
        entry_slot = preferred_entry_slot
    else:
        slot_dirs = {
            "east": (1.0, 0.0),
            "west": (-1.0, 0.0),
            "north": (0.0, 1.0),
            "south": (0.0, -1.0),
        }
        mag = max(1e-6, math.hypot(to_target_x, to_target_y))
        tx = to_target_x / mag
        ty = to_target_y / mag
        entry_slot = max(
            allowed_entry_slots,
            key=lambda s: (slot_dirs.get(str(s), (0.0, 0.0))[0] * tx)
            + (slot_dirs.get(str(s), (0.0, 0.0))[1] * ty),
        )
    non_entry_roles = [r for r in roles if r != "entry"]
    rng = random.Random(int(seed) + EMBEDDED_OFFICE_SEED_OFFSET + 701)
    rng.shuffle(non_entry_roles)
    remaining_slots = [s for s in slots if s != entry_slot]
    rng.shuffle(remaining_slots)
    role_by_slot = {entry_slot: "entry"}
    for slot, role in zip(remaining_slots, non_entry_roles):
        role_by_slot[slot] = role
    return role_by_slot


def build_embedded_office(floor_top_z, area_layout, wall_info, cli=0, seed=0):
    global FLOOR_SIZE, ROOM_CENTER
    if not ENABLE_EMBEDDED_OFFICE_MAP:
        return {"office_map_embedded": False}
    office_area = (area_layout or {}).get("OFFICE")
    if not office_area:
        return {"office_map_embedded": False}
    if not os.path.isdir(ASSET_PATH):
        return {
            "office_map_embedded": False,
            "office_map_reason": f"Missing furniture assets: {ASSET_PATH}",
        }
    old_floor_size = FLOOR_SIZE
    old_room_center = ROOM_CENTER
    try:
        FLOOR_SIZE = min(float(office_area["sx"]), float(office_area["sy"]))
        ROOM_CENTER = (float(office_area["cx"]), float(office_area["cy"]))
        loader = AssetLoader(ASSET_PATH, TEMP_URDF_DIR, UNIFORM_SCALE, cli=cli)
        entry_target_xy = None
        personnel_side = (
            str((wall_info or {}).get("personnel_side", "")).strip().lower()
        )
        if personnel_side in WALL_SLOTS:
            personnel_along = float((wall_info or {}).get("personnel_along", 0.0))
            wall_thickness = float((wall_info or {}).get("wall_thickness", 0.0))
            px, py = slot_point(
                personnel_side,
                personnel_along,
                inward=(wall_thickness * 0.5) + DOCK_INWARD_NUDGE,
            )
            if personnel_side == "east":
                dir_x, dir_y = -1.0, 0.0
            elif personnel_side == "west":
                dir_x, dir_y = 1.0, 0.0
            elif personnel_side == "north":
                dir_x, dir_y = 0.0, -1.0
            else:
                dir_x, dir_y = 0.0, 1.0
            corridor_nudge_m = 1.2
            entry_target_xy = (
                float(px) + (dir_x * corridor_nudge_m),
                float(py) + (dir_y * corridor_nudge_m),
            )
        role_by_slot = _embedded_office_role_map(
            seed=int(seed),
            office_center_xy=ROOM_CENTER,
            entry_target_xy=entry_target_xy,
        )
        entry_slot = next(
            (s for s in WALL_SLOTS if role_by_slot.get(s) == "entry"),
            WALL_SLOTS[0],
        )
        office_walls_enabled = bool(ENABLE_PERIMETER_WALL_MESHES)
        entry_door_along = 0.0
        if office_walls_enabled:
            entry_door_along = float(ENTRY_WALL_OPENING_ALONG)
            spawn_walls_with_entry(
                loader,
                floor_top_z,
                entry_slot=entry_slot,
                door_along=entry_door_along,
                open_mode=str(ENTRY_WALL_OPENING_MODE),
            )
        spawn_entry_doorway = (not office_walls_enabled) or (
            str(ENTRY_WALL_OPENING_MODE).lower() == "gap"
        )
        corners = corner_points()
        forbidden_styles_by_corner = {}
        blocked_corner_indices = set()
        for slot in WALL_SLOTS:
            role = role_by_slot[slot]
            if role == "entry":
                blocked = place_entry_wall(
                    loader,
                    floor_top_z,
                    slot,
                    int(seed),
                    corners=corners,
                    spawn_doorway=spawn_entry_doorway,
                )
                for k, vals in blocked.items():
                    forbidden_styles_by_corner.setdefault(k, set()).update(vals)
            elif role == "workstations":
                workstation_corner = place_workstations_wall(
                    loader, floor_top_z, slot, int(seed)
                )
                if workstation_corner is not None:
                    blocked_corner_indices.add(workstation_corner)
            elif role == "files":
                place_files_wall(loader, floor_top_z, slot, int(seed))
            elif role == "services":
                blocked = place_services_wall(
                    loader, floor_top_z, slot, int(seed), corners=corners
                )
                for k, vals in blocked.items():
                    forbidden_styles_by_corner.setdefault(k, set()).update(vals)
        place_corner_decor(
            loader,
            floor_top_z,
            int(seed),
            forbidden_styles_by_corner=forbidden_styles_by_corner,
            blocked_corner_indices=blocked_corner_indices,
        )
        build_center_meeting(loader, floor_top_z, int(seed))
        return {
            "office_map_embedded": True,
            "office_map_center_xy": ROOM_CENTER,
            "office_map_size_m": FLOOR_SIZE,
            "office_map_roles": role_by_slot,
            "office_walls_enabled": office_walls_enabled,
            "office_entry_slot": entry_slot,
            "office_entry_door_along": float(entry_door_along),
        }
    except Exception as exc:
        return {"office_map_embedded": False, "office_map_reason": str(exc)}
    finally:
        FLOOR_SIZE = old_floor_size
        ROOM_CENTER = old_room_center
