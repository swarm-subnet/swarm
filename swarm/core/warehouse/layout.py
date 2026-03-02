"""
Warehouse area layout: DFS zone placement algorithm.
Places LOADING, OFFICE, FACTORY, STORAGE, FORKLIFT_PARK, MACHINING_CELL
zones with wall attachment, door clearance, and corner constraints.
"""

import itertools
import random

import pybullet as p

from .constants import (
    AREA_LAYOUT_BLOCKS,
    AREA_LAYOUT_EDGE_MARGIN,
    AREA_LAYOUT_MIN_GAP,
    AREA_LAYOUT_TILE_HALF_Z,
    AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR,
    CONVEYOR_ASSETS,
    ENABLE_FACTORY_BARRIER_RING,
    ENABLE_FORKLIFT_PARKING,
    ENABLE_FORKLIFT_PARK_SLOT_LINES,
    ENABLE_MACHINING_CELL_LAYOUT,
    ENABLE_EMBEDDED_OFFICE_MAP,
    PERSONNEL_DOOR_CLEAR_DEPTH,
    PERSONNEL_DOOR_CLEAR_EXTRA_ALONG,
    SHOW_AREA_LAYOUT_MARKERS,
    UNIFORM_SCALE,
    UNIFORM_SPECULAR_COLOR,
    WALL_SLOTS,
    WAREHOUSE_SIZE_X,
    WAREHOUSE_SIZE_Y,
)
from .helpers import (
    _attached_wall_from_area_bounds,
    _candidate_rect_bounds,
    _floor_spawn_half_extents,
    _loading_marker_xy_size,
    _orient_dims_long_side_on_wall,
    _rect_bounds,
    _rects_overlap,
    _sample_random_center,
    _size_fits_half_span,
    _wall_along_limits,
    _wall_attached_center,
    dock_inward_yaw_for_slot,
    oriented_xy_size,
)


def build_area_layout_markers(loader, floor_top_z, wall_info, seed, cli):
    rng = random.Random(int(seed) + 991)
    floor_half_x, floor_half_y = _floor_spawn_half_extents(loader)

    area_defs = {a["name"]: a for a in AREA_LAYOUT_BLOCKS}
    placed = []

    wall_thickness = max(0.0, float(wall_info.get("wall_thickness", 0.0)))
    attach_inset = wall_thickness * float(AREA_LAYOUT_WALL_ATTACH_THICKNESS_FACTOR)
    attach_half_x = min(
        floor_half_x,
        max(1.0, (WAREHOUSE_SIZE_X * 0.5) - attach_inset),
    )
    attach_half_y = min(
        floor_half_y,
        max(1.0, (WAREHOUSE_SIZE_Y * 0.5) - attach_inset),
    )
    personnel_clear_rect = None
    personnel_side = wall_info.get("personnel_side")
    personnel_along = float(wall_info.get("personnel_along", 0.0))
    personnel_span = max(0.0, float(wall_info.get("personnel_door_span", 0.0)))
    if personnel_side in WALL_SLOTS and personnel_span > 0.0:
        if personnel_side in ("north", "south"):
            clear_sx = personnel_span + (2.0 * PERSONNEL_DOOR_CLEAR_EXTRA_ALONG)
            clear_sy = PERSONNEL_DOOR_CLEAR_DEPTH
        else:
            clear_sx = PERSONNEL_DOOR_CLEAR_DEPTH
            clear_sy = personnel_span + (2.0 * PERSONNEL_DOOR_CLEAR_EXTRA_ALONG)
        clear_cx, clear_cy = _wall_attached_center(
            personnel_side, personnel_along, clear_sx, clear_sy,
            attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        )
        personnel_clear_rect = {
            "name": "_PERSONNEL_CLEAR",
            "sx": clear_sx, "sy": clear_sy,
            "cx": clear_cx, "cy": clear_cy,
            "rgba": (0.0, 0.0, 0.0, 0.0),
        }
    critical_zone_keepout_rect = None
    if personnel_clear_rect is not None:
        critical_zone_clear_depth = max(PERSONNEL_DOOR_CLEAR_DEPTH + 3.5, 9.5)
        critical_zone_extra_along = PERSONNEL_DOOR_CLEAR_EXTRA_ALONG + 2.2
        if personnel_side in ("north", "south"):
            kz_sx = personnel_span + (2.0 * critical_zone_extra_along)
            kz_sy = critical_zone_clear_depth
        else:
            kz_sx = critical_zone_clear_depth
            kz_sy = personnel_span + (2.0 * critical_zone_extra_along)
        kz_cx, kz_cy = _wall_attached_center(
            personnel_side, personnel_along, kz_sx, kz_sy,
            attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        )
        critical_zone_keepout_rect = {
            "name": "_PERSONNEL_CRITICAL_KEEPOUT",
            "sx": kz_sx, "sy": kz_sy,
            "cx": kz_cx, "cy": kz_cy,
            "rgba": (0.0, 0.0, 0.0, 0.0),
        }
    office_passage_keepout_rect = None
    if personnel_clear_rect is not None:
        office_clear_depth = max(PERSONNEL_DOOR_CLEAR_DEPTH + 5.0, 11.0)
        office_extra_along = PERSONNEL_DOOR_CLEAR_EXTRA_ALONG + 2.8
        if personnel_side in ("north", "south"):
            ok_sx = personnel_span + (2.0 * office_extra_along)
            ok_sy = office_clear_depth
        else:
            ok_sx = office_clear_depth
            ok_sy = personnel_span + (2.0 * office_extra_along)
        ok_cx, ok_cy = _wall_attached_center(
            personnel_side, personnel_along, ok_sx, ok_sy,
            attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        )
        office_passage_keepout_rect = {
            "name": "_OFFICE_DOORWAY_PASSAGE_KEEPOUT",
            "sx": ok_sx, "sy": ok_sy,
            "cx": ok_cx, "cy": ok_cy,
            "rgba": (0.0, 0.0, 0.0, 0.0),
        }

    critical_door_blocking_zones = {"LOADING", "STORAGE", "FACTORY"}
    major_zone_fixed_short_side_m = {
        zone_name: float(min(area_defs[zone_name]["size_m"]))
        for zone_name in ("LOADING", "STORAGE", "FACTORY")
        if zone_name in area_defs
    }
    opposite_personnel_side = {
        "north": "south", "south": "north",
        "east": "west", "west": "east",
    }.get(personnel_side)
    major_zones = {"LOADING", "STORAGE", "FACTORY"}
    transverse_major_zone = str(wall_info.get("transverse_major_zone", "LOADING")).upper()
    if transverse_major_zone not in major_zones:
        transverse_major_zone = "LOADING"
    longitudinal_major_zones = set(major_zones) - {transverse_major_zone}
    utility_longitudinal_zones = {"STORAGE", "FACTORY"} & longitudinal_major_zones
    if personnel_side in ("east", "west"):
        longitudinal_side_walls = ("north", "south")
    elif personnel_side in ("north", "south"):
        longitudinal_side_walls = ("east", "west")
    else:
        longitudinal_side_walls = tuple(WALL_SLOTS)
    longitudinal_side_walls = tuple(w for w in longitudinal_side_walls if w in WALL_SLOTS)
    transverse_end_only_strip = None
    if opposite_personnel_side in WALL_SLOTS:
        strip_depth = max(22.5, min(26.0, max(major_zone_fixed_short_side_m.values(), default=22.5)))
        if opposite_personnel_side == "north":
            strip_sx = max(1.0, attach_half_x * 2.0)
            strip_sy = strip_depth
            strip_cx = 0.0
            strip_cy = attach_half_y - (strip_sy * 0.5)
        elif opposite_personnel_side == "south":
            strip_sx = max(1.0, attach_half_x * 2.0)
            strip_sy = strip_depth
            strip_cx = 0.0
            strip_cy = -attach_half_y + (strip_sy * 0.5)
        elif opposite_personnel_side == "east":
            strip_sx = strip_depth
            strip_sy = max(1.0, attach_half_y * 2.0)
            strip_cx = attach_half_x - (strip_sx * 0.5)
            strip_cy = 0.0
        else:
            strip_sx = strip_depth
            strip_sy = max(1.0, attach_half_y * 2.0)
            strip_cx = -attach_half_x + (strip_sx * 0.5)
            strip_cy = 0.0
        transverse_end_only_strip = {
            "name": "_TRANSVERSE_END_ONLY_STRIP",
            "sx": strip_sx, "sy": strip_sy,
            "cx": strip_cx, "cy": strip_cy,
            "rgba": (0.0, 0.0, 0.0, 0.0),
        }

    def _scaled_dims_for_zone(name, base_sx, base_sy, shrink):
        sx0 = float(base_sx)
        sy0 = float(base_sy)
        k = max(0.01, float(shrink))
        zone_key = str(name).upper()
        fixed_short = major_zone_fixed_short_side_m.get(zone_key)
        if fixed_short is None:
            return sx0 * k, sy0 * k
        long_base = max(sx0, sy0)
        long_try = max(fixed_short, long_base * k)
        if sx0 <= sy0:
            return fixed_short, long_try
        return long_try, fixed_short

    def _candidate_attached_wall(candidate):
        cached = candidate.get("_attached_wall")
        if cached is not None:
            return cached
        attached_wall = _attached_wall_from_area_bounds(
            float(candidate.get("sx", 0.0)),
            float(candidate.get("sy", 0.0)),
            float(candidate.get("cx", 0.0)),
            float(candidate.get("cy", 0.0)),
        )
        candidate["_attached_wall"] = attached_wall
        return attached_wall

    def _make_candidate(name, sx, sy, cx, cy, color):
        sx_f = float(sx)
        sy_f = float(sy)
        cx_f = float(cx)
        cy_f = float(cy)
        fits_attach_span = _size_fits_half_span(
            sx_f, sy_f, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        )
        cand = {
            "name": name,
            "sx": sx_f, "sy": sy_f,
            "cx": cx_f, "cy": cy_f,
            "rgba": color,
            "_fits_attach_span": fits_attach_span,
        }
        cand["_rect_bounds"] = _rect_bounds(cx_f, cy_f, sx_f, sy_f)
        cand["_attached_wall"] = _attached_wall_from_area_bounds(sx_f, sy_f, cx_f, cy_f)
        return cand

    def _is_far_from_personnel_door_on_same_wall(candidate):
        if personnel_side not in WALL_SLOTS:
            return True
        attached_wall = _candidate_attached_wall(candidate)
        if attached_wall != personnel_side:
            return True
        if attached_wall in ("north", "south"):
            cand_along = float(candidate.get("cx", 0.0))
            cand_span = float(candidate.get("sx", 0.0))
        else:
            cand_along = float(candidate.get("cy", 0.0))
            cand_span = float(candidate.get("sy", 0.0))
        door_half = (personnel_span * 0.5) + PERSONNEL_DOOR_CLEAR_EXTRA_ALONG + 1.0
        zone_half = cand_span * 0.5
        min_center_distance = door_half + zone_half
        return abs(cand_along - personnel_along) >= (min_center_distance - 1e-6)

    def _opposite_wall_end_targets(wall, sx, sy):
        lo, hi = _wall_along_limits(
            wall, float(sx), float(sy),
            attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return None
        return float(lo), float(hi)

    def _is_at_preferred_opposite_end(candidate):
        if opposite_personnel_side not in WALL_SLOTS:
            return True
        attached_wall = _candidate_attached_wall(candidate)
        if attached_wall != opposite_personnel_side:
            return False
        if attached_wall in ("north", "south"):
            cand_along = float(candidate.get("cx", 0.0))
            cand_span = float(candidate.get("sx", 0.0))
        else:
            cand_along = float(candidate.get("cy", 0.0))
            cand_span = float(candidate.get("sy", 0.0))
        end_targets = _opposite_wall_end_targets(
            attached_wall,
            float(candidate.get("sx", 0.0)),
            float(candidate.get("sy", 0.0)),
        )
        if end_targets is None:
            return False
        target_lo, target_hi = end_targets
        end_tol = max(0.45, min(2.6, cand_span * 0.12))
        return abs(cand_along - target_lo) <= end_tol or abs(cand_along - target_hi) <= end_tol

    def _is_at_wall_end(candidate, end_tol_factor=0.16):
        attached_wall = _candidate_attached_wall(candidate)
        if attached_wall in ("north", "south"):
            cand_along = float(candidate.get("cx", 0.0))
            cand_span = float(candidate.get("sx", 0.0))
        else:
            cand_along = float(candidate.get("cy", 0.0))
            cand_span = float(candidate.get("sy", 0.0))
        lo, hi = _wall_along_limits(
            attached_wall,
            float(candidate.get("sx", 0.0)),
            float(candidate.get("sy", 0.0)),
            attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return False
        end_tol = max(0.9, min(4.2, cand_span * float(end_tol_factor)))
        return abs(cand_along - lo) <= end_tol or abs(cand_along - hi) <= end_tol

    def _can_place_static(candidate, gap):
        fits_attach_span = candidate.get("_fits_attach_span")
        if fits_attach_span is None:
            fits_attach_span = _size_fits_half_span(
                candidate["sx"], candidate["sy"],
                attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
            )
            candidate["_fits_attach_span"] = fits_attach_span
        if not fits_attach_span:
            return False
        name = str(candidate.get("name", ""))
        if name not in critical_door_blocking_zones:
            return True
        attached_wall = _candidate_attached_wall(candidate)
        if transverse_end_only_strip is not None and name != transverse_major_zone:
            if _rects_overlap(candidate, transverse_end_only_strip, 0.0):
                return False
        if name == transverse_major_zone:
            if opposite_personnel_side in WALL_SLOTS and attached_wall != opposite_personnel_side:
                return False
        elif name in longitudinal_major_zones and longitudinal_side_walls:
            if attached_wall not in longitudinal_side_walls:
                return False
        if critical_zone_keepout_rect is not None and _rects_overlap(candidate, critical_zone_keepout_rect, 0.0):
            if attached_wall in (personnel_side, opposite_personnel_side):
                return False
        if attached_wall in (personnel_side, opposite_personnel_side):
            if not _is_far_from_personnel_door_on_same_wall(candidate):
                return False
        return True

    def _can_place(candidate, gap):
        fits_attach_span = candidate.get("_fits_attach_span")
        if fits_attach_span is None:
            fits_attach_span = _size_fits_half_span(
                candidate["sx"], candidate["sy"],
                attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
            )
            candidate["_fits_attach_span"] = fits_attach_span
        if not fits_attach_span:
            return False
        cand_bounds = candidate.get("_rect_bounds")
        if cand_bounds is None:
            cand_bounds = _candidate_rect_bounds(candidate)
        cand_min_x, cand_max_x, cand_min_y, cand_max_y = cand_bounds
        for prev in placed:
            prev_bounds = prev.get("_rect_bounds")
            if prev_bounds is None:
                prev_bounds = _candidate_rect_bounds(prev)
            prev_min_x, prev_max_x, prev_min_y, prev_max_y = prev_bounds
            if not (
                (cand_max_x + gap) <= prev_min_x
                or (prev_max_x + gap) <= cand_min_x
                or (cand_max_y + gap) <= prev_min_y
                or (prev_max_y + gap) <= cand_min_y
            ):
                return False
        name = str(candidate.get("name", ""))
        if name in critical_door_blocking_zones:
            attached_wall = _candidate_attached_wall(candidate)
            if transverse_end_only_strip is not None and name != transverse_major_zone:
                if _rects_overlap(candidate, transverse_end_only_strip, 0.0):
                    return False
            if name == transverse_major_zone:
                if opposite_personnel_side in WALL_SLOTS and attached_wall != opposite_personnel_side:
                    return False
            elif name in longitudinal_major_zones and longitudinal_side_walls:
                if attached_wall not in longitudinal_side_walls:
                    return False
                for prev in placed:
                    prev_name = str(prev.get("name", ""))
                    if prev_name not in longitudinal_major_zones or prev_name == name:
                        continue
                    prev_wall = _candidate_attached_wall(prev)
                    if prev_wall == attached_wall:
                        return False
            if critical_zone_keepout_rect is not None and _rects_overlap(candidate, critical_zone_keepout_rect, 0.0):
                if attached_wall in (personnel_side, opposite_personnel_side):
                    return False
            if attached_wall in (personnel_side, opposite_personnel_side):
                if not _is_far_from_personnel_door_on_same_wall(candidate):
                    return False
        return True

    def _place_on_wall(
        name, sx, sy, wall, color,
        along_pref=None, tries=600, gap=AREA_LAYOUT_MIN_GAP,
        validator=None, deterministic_first=True,
    ):
        sx, sy = float(sx), float(sy)
        sx, sy = _orient_dims_long_side_on_wall(wall, sx, sy)
        if not _size_fits_half_span(sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN):
            return None
        lo, hi = _wall_along_limits(wall, sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN)
        if hi < lo:
            return None
        anchors = [lo, hi, 0.5 * (lo + hi), max(lo, min(hi, 0.0))]
        if along_pref is not None:
            anchors.append(max(lo, min(hi, along_pref)))

        def _try_along(along_value):
            cx, cy = _wall_attached_center(
                wall, along_value, sx, sy,
                attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
            )
            cand = _make_candidate(name, sx, sy, cx, cy, color)
            if not _can_place(cand, gap):
                return None
            if validator is not None and not validator(cand):
                return None
            return cand

        if deterministic_first:
            for along in anchors:
                cand = _try_along(along)
                if cand is not None:
                    return cand
            for _ in range(max(0, int(tries))):
                cand = _try_along(rng.uniform(lo, hi))
                if cand is not None:
                    return cand
        else:
            for _ in range(max(0, int(tries))):
                cand = _try_along(rng.uniform(lo, hi))
                if cand is not None:
                    return cand
            for along in anchors:
                cand = _try_along(along)
                if cand is not None:
                    return cand
        return None

    def _place_anywhere(name, sx, sy, color, tries=1200, gap=AREA_LAYOUT_MIN_GAP, validator=None):
        sx, sy = float(sx), float(sy)
        if not _size_fits_half_span(sx, sy, floor_half_x, floor_half_y, AREA_LAYOUT_EDGE_MARGIN):
            return None
        for _ in range(max(1, tries)):
            cx, cy = _sample_random_center(
                rng, sx, sy, floor_half_x, floor_half_y, AREA_LAYOUT_EDGE_MARGIN,
            )
            cand = _make_candidate(name, sx, sy, cx, cy, color)
            if _can_place(cand, gap):
                if validator is not None and not validator(cand):
                    continue
                return cand
        return None

    def _force_place_zone(name, base_sx, base_sy, color, preferred_walls=None, along_pref=None, validator=None):
        if preferred_walls is None:
            preferred_walls = list(WALL_SLOTS)
        sx = float(base_sx)
        sy = float(base_sy)
        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for wall in preferred_walls:
                picked = _place_on_wall(
                    name=name, sx=sx, sy=sy, wall=wall, color=color,
                    along_pref=along_pref, tries=320, gap=gap, validator=validator,
                )
                if picked is not None:
                    return picked
            picked = _place_anywhere(
                name=name, sx=sx, sy=sy, color=color, tries=600, gap=gap, validator=validator,
            )
            if picked is not None:
                return picked
        raise ValueError(
            f"Unable to place zone '{name}' at fixed size {sx:.2f} x {sy:.2f} m "
            f"without overlap/out-of-bounds."
        )

    # --- LOADING zone placement ---
    loading_def = area_defs["LOADING"]
    loading_side = wall_info.get("loading_side", "north")
    door_centers = wall_info.get("door_centers", [])
    frame_yaw_for_span = dock_inward_yaw_for_slot(loading_side)
    frame_ex, frame_ey = oriented_xy_size(
        loader, CONVEYOR_ASSETS["dock_frame"], UNIFORM_SCALE, frame_yaw_for_span,
    )
    door_span = frame_ex if loading_side in ("north", "south") else frame_ey
    load_sx, load_sy = _loading_marker_xy_size(loading_def["size_m"], loading_side)
    along_pref = (sum(door_centers) / float(len(door_centers))) if door_centers else 0.0

    def _loading_zone_covers_doors(candidate):
        if not door_centers:
            return True
        if loading_side in ("north", "south"):
            along_center = float(candidate["cx"])
            along_half = float(candidate["sx"]) * 0.5
        else:
            along_center = float(candidate["cy"])
            along_half = float(candidate["sy"]) * 0.5
        door_half = max(0.0, float(door_span) * 0.5)
        min_along = along_center - along_half + door_half
        max_along = along_center + along_half - door_half
        return all((min_along - 1e-6) <= c <= (max_along + 1e-6) for c in door_centers)

    fit_sx, fit_sy = _orient_dims_long_side_on_wall(loading_side, load_sx, load_sy)
    lo, hi = _wall_along_limits(
        loading_side, fit_sx, fit_sy,
        attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
    )
    loading_candidate = None
    if hi >= lo:
        corner_pref = str(wall_info.get("loading_corner_side", "")).lower()
        if corner_pref == "left":
            corner_alongs = [lo, hi]
        elif corner_pref == "right":
            corner_alongs = [hi, lo]
        else:
            center_from_doors = 0.5 * (min(door_centers) + max(door_centers)) if door_centers else 0.0
            corner_alongs = [lo, hi] if center_from_doors <= 0.0 else [hi, lo]

        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for along in corner_alongs:
                cx, cy = _wall_attached_center(
                    loading_side, along, fit_sx, fit_sy,
                    attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
                )
                cand = _make_candidate("LOADING", fit_sx, fit_sy, cx, cy, loading_def["rgba"])
                if _can_place(cand, gap) and _loading_zone_covers_doors(cand):
                    loading_candidate = cand
                    break
            if loading_candidate is not None:
                break

        if loading_candidate is None:
            preferred_end = corner_alongs[0] if corner_alongs else hi
            span = max(0.0, hi - lo)
            sweep_steps = max(36, int(span / 0.35))
            sweep_alongs = []
            for i in range(sweep_steps + 1):
                t = float(i) / float(sweep_steps) if sweep_steps > 0 else 0.0
                sweep_alongs.append(float(lo + ((hi - lo) * t)))
            sweep_alongs = list(
                sorted(sweep_alongs, key=lambda a: abs(float(a) - float(preferred_end)))
            )
            for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
                for along in sweep_alongs:
                    cx, cy = _wall_attached_center(
                        loading_side, along, fit_sx, fit_sy,
                        attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
                    )
                    cand = _make_candidate("LOADING", fit_sx, fit_sy, cx, cy, loading_def["rgba"])
                    if _can_place(cand, gap) and _loading_zone_covers_doors(cand):
                        loading_candidate = cand
                        break
                if loading_candidate is not None:
                    break

    if loading_candidate is None:
        raise ValueError("Unable to place LOADING in a corner while covering all 3 dock doors.")
    if loading_candidate is not None:
        placed.append(loading_candidate)

    # --- OFFICE zone placement ---
    office_def = area_defs["OFFICE"]
    fit_sx, fit_sy = float(office_def["size_m"][0]), float(office_def["size_m"][1])
    office_wall_priority = [personnel_side] if personnel_side in WALL_SLOTS else list(WALL_SLOTS)

    def _office_along_for_wall(wall_name):
        if wall_name in ("north", "south"):
            if personnel_side == "east":
                return float(max(0.0, attach_half_x - (fit_sx * 0.5)))
            if personnel_side == "west":
                return float(-max(0.0, attach_half_x - (fit_sx * 0.5)))
            return float(personnel_along if personnel_side in ("north", "south") else 0.0)
        if personnel_side == "north":
            return float(max(0.0, attach_half_y - (fit_sy * 0.5)))
        if personnel_side == "south":
            return float(-max(0.0, attach_half_y - (fit_sy * 0.5)))
        return float(personnel_along if personnel_side in ("east", "west") else 0.0)

    def _office_pref_alongs_for_wall(wall_name):
        along_center = _office_along_for_wall(wall_name)
        if wall_name in ("north", "south"):
            door_along_span = float(personnel_span if personnel_side in ("north", "south") else 0.0)
        else:
            door_along_span = float(personnel_span if personnel_side in ("east", "west") else 0.0)
        office_along_span = fit_sx if wall_name in ("north", "south") else fit_sy
        door_clear_offset = (
            (door_along_span * 0.5)
            + (office_along_span * 0.5)
            + float(PERSONNEL_DOOR_CLEAR_EXTRA_ALONG)
            + 0.9
        )
        if wall_name == personnel_side:
            return [along_center, along_center - door_clear_offset, along_center + door_clear_offset, 0.0]
        return [along_center - door_clear_offset, along_center + door_clear_offset, along_center, 0.0]

    def _office_sweep_alongs_for_wall(wall_name):
        sx_o, sy_o = _orient_dims_long_side_on_wall(wall_name, fit_sx, fit_sy)
        lo, hi = _wall_along_limits(
            wall_name, sx_o, sy_o, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return []
        span = max(0.0, hi - lo)
        steps = max(12, int(span / 0.9))
        out = []
        if steps <= 0:
            return [float(lo)]
        for i in range(steps + 1):
            t = float(i) / float(steps)
            out.append(float(lo + (span * t)))
        return out

    def _office_matches_personnel_side(candidate):
        if personnel_side not in WALL_SLOTS:
            return True
        attached_wall = _candidate_attached_wall(candidate)
        if attached_wall != personnel_side:
            return False
        if not _is_far_from_personnel_door_on_same_wall(candidate):
            return False
        if office_passage_keepout_rect is not None and _rects_overlap(candidate, office_passage_keepout_rect, 0.0):
            return False
        return True

    office_candidates = []
    office_seen = set()

    def _push_office_candidate(cand):
        if cand is None:
            return
        key = (
            round(float(cand["cx"]), 4), round(float(cand["cy"]), 4),
            round(float(cand["sx"]), 4), round(float(cand["sy"]), 4),
        )
        if key in office_seen:
            return
        office_seen.add(key)
        office_candidates.append(cand)

    for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
        for wall in office_wall_priority:
            along_candidates = list(_office_pref_alongs_for_wall(wall)) + list(_office_sweep_alongs_for_wall(wall))
            for ap in along_candidates:
                _push_office_candidate(
                    _place_on_wall(
                        name="OFFICE", sx=fit_sx, sy=fit_sy, wall=wall,
                        color=office_def["rgba"], along_pref=ap, tries=220,
                        gap=gap, validator=_office_matches_personnel_side,
                        deterministic_first=True,
                    )
                )
    for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
        for wall in office_wall_priority:
            _push_office_candidate(
                _place_on_wall(
                    name="OFFICE", sx=fit_sx, sy=fit_sy, wall=wall,
                    color=office_def["rgba"], along_pref=None, tries=320,
                    gap=gap, validator=_office_matches_personnel_side,
                    deterministic_first=False,
                )
            )
    _push_office_candidate(
        _place_anywhere(
            name="OFFICE", sx=fit_sx, sy=fit_sy, color=office_def["rgba"],
            tries=600, gap=0.0, validator=_office_matches_personnel_side,
        )
    )
    if not office_candidates:
        raise ValueError("Unable to place OFFICE near personnel door side without overlap/out-of-bounds.")

    # --- Utility zone placement (FACTORY, STORAGE, etc.) ---
    utility_zones = ["FACTORY", "STORAGE"]
    optional_utility_zones = []
    if ENABLE_FORKLIFT_PARKING:
        optional_utility_zones.append("FORKLIFT_PARK")
    if ENABLE_MACHINING_CELL_LAYOUT:
        optional_utility_zones.append("MACHINING_CELL")
    utility_zones = list(sorted(
        utility_zones,
        key=lambda zn: float(area_defs[zn]["size_m"][0]) * float(area_defs[zn]["size_m"][1]),
        reverse=True,
    ))
    optional_utility_zones = list(sorted(
        optional_utility_zones,
        key=lambda zn: float(area_defs[zn]["size_m"][0]) * float(area_defs[zn]["size_m"][1]),
        reverse=True,
    ))

    def _utility_wall_priority():
        walls = list(WALL_SLOTS)
        rng.shuffle(walls)
        walls.sort(key=lambda w: (1 if w == loading_side else 0, 1 if w == personnel_side else 0))
        return walls

    def _utility_candidate_pool(name, gap):
        area = area_defs[name]
        base_sx = float(area["size_m"][0])
        base_sy = float(area["size_m"][1])
        zone_key = str(name).upper()
        color = area["rgba"]
        seen = set()
        out = []
        max_candidates = 56 if zone_key in utility_longitudinal_zones else 48

        wall_order = list(_utility_wall_priority())
        if zone_key == transverse_major_zone and opposite_personnel_side in WALL_SLOTS:
            wall_order = [opposite_personnel_side]
        elif zone_key in utility_longitudinal_zones and longitudinal_side_walls:
            wall_order = list(longitudinal_side_walls)

        for wall in wall_order:
            sx, sy = _orient_dims_long_side_on_wall(wall, base_sx, base_sy)
            if not _size_fits_half_span(sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN):
                continue
            lo, hi = _wall_along_limits(wall, sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN)
            if hi < lo:
                continue
            span = max(0.0, hi - lo)
            if zone_key in utility_longitudinal_zones:
                end_band = max(1.6, min(span, 8.5))
                lo_band_hi = min(hi, lo + end_band)
                hi_band_lo = max(lo, hi - end_band)
                alongs = [lo, hi, lo_band_hi, hi_band_lo]
                sweep_steps = max(5, int(end_band / 1.1))
                for i in range(sweep_steps + 1):
                    t = float(i) / float(sweep_steps) if sweep_steps > 0 else 0.0
                    alongs.append(float(lo + ((lo_band_hi - lo) * t)))
                    alongs.append(float(hi_band_lo + ((hi - hi_band_lo) * t)))
                for _ in range(14):
                    alongs.append(rng.uniform(lo, lo_band_hi))
                    alongs.append(rng.uniform(hi_band_lo, hi))
            else:
                if zone_key == transverse_major_zone:
                    alongs = [0.5 * (lo + hi), max(lo, min(hi, 0.0)), lo, hi]
                else:
                    alongs = [lo, hi, 0.5 * (lo + hi), max(lo, min(hi, 0.0))]
                sweep_steps = max(6, int(span / 2.5))
                for i in range(sweep_steps + 1):
                    t = float(i) / float(sweep_steps) if sweep_steps > 0 else 0.0
                    alongs.append(float(lo + ((hi - lo) * t)))
                for _ in range(20):
                    alongs.append(rng.uniform(lo, hi))
            for along in alongs:
                cx, cy = _wall_attached_center(
                    wall, along, sx, sy,
                    attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
                )
                cand = _make_candidate(name, sx, sy, cx, cy, color)
                key = (round(cx, 4), round(cy, 4), round(sx, 4), round(sy, 4))
                if key in seen:
                    continue
                if not _can_place_static(cand, gap):
                    continue
                seen.add(key)
                out.append(cand)
                if len(out) >= max_candidates:
                    return out

        if zone_key in ("STORAGE", "FACTORY"):
            return out

        if _size_fits_half_span(base_sx, base_sy, floor_half_x, floor_half_y, AREA_LAYOUT_EDGE_MARGIN):
            center_cand = _make_candidate(name, base_sx, base_sy, 0.0, 0.0, color)
            key = (
                round(center_cand["cx"], 4), round(center_cand["cy"], 4),
                round(center_cand["sx"], 4), round(center_cand["sy"], 4),
            )
            if key not in seen and _can_place_static(center_cand, gap):
                seen.add(key)
                out.append(center_cand)
            for _ in range(40):
                cx, cy = _sample_random_center(
                    rng, base_sx, base_sy, floor_half_x, floor_half_y, AREA_LAYOUT_EDGE_MARGIN,
                )
                cand = _make_candidate(name, base_sx, base_sy, cx, cy, color)
                key = (round(cx, 4), round(cy, 4), round(base_sx, 4), round(base_sy, 4))
                if key in seen:
                    continue
                if not _can_place_static(cand, gap):
                    continue
                seen.add(key)
                out.append(cand)
                if len(out) >= max_candidates:
                    break
        return out

    utility_pool_cache = {}
    utility_orders = list(itertools.permutations(utility_zones))
    rng.shuffle(utility_orders)
    base_order = tuple(utility_zones)
    if base_order in utility_orders:
        utility_orders.remove(base_order)
    utility_orders = [base_order] + utility_orders

    def _place_all_utilities_for_current_state():
        def _pool(name, gap):
            key = (name, float(gap))
            if key not in utility_pool_cache:
                utility_pool_cache[key] = _utility_candidate_pool(name, gap)
            return utility_pool_cache[key]

        def _dfs_place(order, idx, gap):
            if idx >= len(order):
                return True
            name = order[idx]
            for cand in _pool(name, gap):
                if not _can_place(cand, gap):
                    continue
                placed.append(cand)
                if _dfs_place(order, idx + 1, gap):
                    return True
                placed.pop()
            return False

        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for order in utility_orders:
                snapshot = len(placed)
                if _dfs_place(order, 0, gap):
                    return True
                del placed[snapshot:]
        return False

    office_and_utilities_placed = False
    for office_candidate in office_candidates:
        snapshot = len(placed)
        placed.append(office_candidate)
        if _place_all_utilities_for_current_state():
            office_and_utilities_placed = True
            break
        del placed[snapshot:]

    if not office_and_utilities_placed:
        extra_candidates = []
        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for wall in office_wall_priority:
                for _ in range(6):
                    cand = _place_on_wall(
                        name="OFFICE", sx=fit_sx, sy=fit_sy, wall=wall,
                        color=office_def["rgba"], along_pref=None, tries=280,
                        gap=gap, validator=_office_matches_personnel_side,
                        deterministic_first=False,
                    )
                    if cand is not None:
                        extra_candidates.append(cand)
        for cand in extra_candidates:
            snapshot = len(placed)
            placed.append(cand)
            if _place_all_utilities_for_current_state():
                office_and_utilities_placed = True
                break
            del placed[snapshot:]

    if not office_and_utilities_placed:
        salvage_office_candidates = list(office_candidates)
        for cand in extra_candidates:
            salvage_office_candidates.append(cand)

        for strict_storage_shape in (True, False):
            for office_candidate in salvage_office_candidates:
                snapshot = len(placed)
                placed.append(office_candidate)
                salvage_ok = True
                for name in utility_zones:
                    if any(a.get("name") == name for a in placed):
                        continue
                    area = area_defs[name]
                    base_sx = float(area["size_m"][0])
                    base_sy = float(area["size_m"][1])
                    picked = None
                    if str(name).upper() == "STORAGE":
                        if strict_storage_shape:
                            shrink_steps = (1.0, 0.97, 0.94, 0.90, 0.86, 0.82)
                        else:
                            shrink_steps = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45)
                    else:
                        shrink_steps = (1.0, 0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30)
                    for shrink in shrink_steps:
                        sx_try, sy_try = _scaled_dims_for_zone(name, base_sx, base_sy, shrink)
                        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
                            wall_candidates = list(_utility_wall_priority())
                            zone_key = str(name).upper()
                            if zone_key == transverse_major_zone and opposite_personnel_side in WALL_SLOTS:
                                wall_candidates = [opposite_personnel_side]
                            elif zone_key in utility_longitudinal_zones and longitudinal_side_walls:
                                wall_candidates = list(longitudinal_side_walls)
                            for wall in wall_candidates:
                                picked = _place_on_wall(
                                    name=name, sx=sx_try, sy=sy_try, wall=wall,
                                    color=area["rgba"], along_pref=None, tries=320,
                                    gap=gap, deterministic_first=False,
                                )
                                if picked is not None:
                                    break
                            if picked is not None:
                                break
                            if zone_key not in utility_longitudinal_zones and zone_key != transverse_major_zone:
                                picked = _place_anywhere(
                                    name=name, sx=sx_try, sy=sy_try,
                                    color=area["rgba"], tries=600, gap=gap,
                                )
                                if picked is not None:
                                    break
                        if picked is not None:
                            break
                    if picked is None:
                        salvage_ok = False
                        break
                    placed.append(picked)
                if salvage_ok:
                    office_and_utilities_placed = True
                    break
                del placed[snapshot:]
            if office_and_utilities_placed:
                break

    if not office_and_utilities_placed:
        raise ValueError(
            "Unable to place OFFICE + required utility zones while keeping OFFICE near personnel door side."
        )

    # --- Optional utility zones ---
    for name in optional_utility_zones:
        if any(a.get("name") == name for a in placed):
            continue
        area = area_defs[name]
        base_sx = float(area["size_m"][0])
        base_sy = float(area["size_m"][1])
        require_wall_attachment = (name == "FORKLIFT_PARK")
        is_forklift_park = (str(name).upper() == "FORKLIFT_PARK")
        office_ref = next((a for a in placed if a.get("name") == "OFFICE"), None)
        walls = list(WALL_SLOTS)
        rng.shuffle(walls)
        walls.sort(key=lambda w: 1 if w == loading_side else 0)
        if require_wall_attachment:
            office_wall = None
            if office_ref is not None:
                office_wall = _candidate_attached_wall(office_ref)
            preferred_walls = []
            for w in (office_wall, personnel_side):
                if w in WALL_SLOTS and w not in preferred_walls:
                    preferred_walls.append(w)
            anchor_wall = office_wall if office_wall in WALL_SLOTS else personnel_side
            side_order = {
                "north": ["north", "east", "west", "south"],
                "south": ["south", "east", "west", "north"],
                "east": ["east", "north", "south", "west"],
                "west": ["west", "north", "south", "east"],
            }
            if anchor_wall in side_order:
                ordered = list(side_order[anchor_wall])
                if personnel_side in WALL_SLOTS and personnel_side in ordered:
                    ordered.remove(personnel_side)
                    ordered.insert(1, personnel_side)
                walls = ordered
            elif preferred_walls:
                walls = preferred_walls + [w for w in walls if w not in preferred_walls]

        def _optional_along_pref_for_wall(wall_name):
            if is_forklift_park and wall_name == personnel_side:
                return None
            if office_ref is not None:
                if wall_name in ("north", "south"):
                    return float(office_ref["cx"])
                return float(office_ref["cy"])
            if wall_name == personnel_side:
                return float(personnel_along)
            return None

        office_keepout_gap = 2.0 if require_wall_attachment else 0.0

        def _optional_zone_validator(candidate):
            if require_wall_attachment and office_ref is not None:
                if _rects_overlap(candidate, office_ref, office_keepout_gap):
                    return False
            if is_forklift_park:
                if office_passage_keepout_rect is not None and _rects_overlap(candidate, office_passage_keepout_rect, 0.0):
                    return False
                attached_wall = _candidate_attached_wall(candidate)
                if attached_wall == personnel_side and (not _is_far_from_personnel_door_on_same_wall(candidate)):
                    return False
            return True

        picked = None
        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for wall in walls:
                cand = _place_on_wall(
                    name=name, sx=base_sx, sy=base_sy, wall=wall,
                    color=area["rgba"],
                    along_pref=_optional_along_pref_for_wall(wall),
                    tries=260, gap=gap, validator=_optional_zone_validator,
                    deterministic_first=False,
                )
                if cand is not None:
                    picked = cand
                    break
            if picked is not None:
                break
            if not require_wall_attachment:
                cand = _place_anywhere(
                    name=name, sx=base_sx, sy=base_sy,
                    color=area["rgba"], tries=500, gap=gap,
                    validator=_optional_zone_validator,
                )
                if cand is not None:
                    picked = cand
                    break
        if picked is None and require_wall_attachment:
            long_base = max(base_sx, base_sy)
            short_base = min(base_sx, base_sy)
            for shrink in (0.90, 0.80, 0.72, 0.65, 0.58, 0.50, 0.42):
                long_try = max(6.0, long_base * shrink)
                short_try = max(4.8, short_base)
                for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
                    for wall in walls:
                        if wall in ("north", "south"):
                            sx_try, sy_try = long_try, short_try
                        else:
                            sx_try, sy_try = short_try, long_try
                        cand = _place_on_wall(
                            name=name, sx=sx_try, sy=sy_try, wall=wall,
                            color=area["rgba"],
                            along_pref=_optional_along_pref_for_wall(wall),
                            tries=320, gap=gap, validator=_optional_zone_validator,
                            deterministic_first=False,
                        )
                        if cand is not None:
                            picked = cand
                            break
                    if picked is not None:
                        break
                if picked is not None:
                    break
        if picked is not None:
            placed.append(picked)

    # --- Fallback placement for required zones ---
    required_names = ["LOADING", "OFFICE"] + utility_zones
    existing_names = {a["name"] for a in placed}
    for name in required_names:
        if name in existing_names:
            continue
        area = area_defs[name]
        if name == "LOADING":
            fallback_walls = [loading_side]
        elif name == transverse_major_zone and opposite_personnel_side in WALL_SLOTS:
            fallback_walls = [opposite_personnel_side]
        elif name in utility_longitudinal_zones and longitudinal_side_walls:
            fallback_walls = list(longitudinal_side_walls)
        else:
            fallback_walls = list(WALL_SLOTS)
        fallback = _force_place_zone(
            name=name,
            base_sx=area["size_m"][0],
            base_sy=area["size_m"][1],
            color=area["rgba"],
            preferred_walls=fallback_walls,
            along_pref=along_pref if name == "LOADING" else None,
        )
        placed.append(fallback)
        existing_names.add(name)

    # --- Post-processing: centering isolated zones ---
    def _try_center_zone_if_isolated_on_wall(zone_name):
        zone_idx = next((i for i, a in enumerate(placed) if str(a.get("name", "")) == str(zone_name)), None)
        if zone_idx is None:
            return False
        current = placed[zone_idx]
        attached_wall = _candidate_attached_wall(current)
        if attached_wall not in WALL_SLOTS:
            return False
        wall_zone_names = []
        for i, area in enumerate(placed):
            area_name = str(area.get("name", ""))
            if area_name.startswith("_"):
                continue
            if area_name not in area_defs:
                continue
            if _candidate_attached_wall(area) != attached_wall:
                continue
            wall_zone_names.append((i, area_name))
        if len(wall_zone_names) != 1 or wall_zone_names[0][0] != zone_idx:
            return False
        sx = float(current.get("sx", 0.0))
        sy = float(current.get("sy", 0.0))
        lo, hi = _wall_along_limits(
            attached_wall, sx, sy,
            attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return False
        along_center = 0.5 * (lo + hi)
        along_zero = max(lo, min(hi, 0.0))
        along_targets = [along_center]
        if abs(along_zero - along_center) > 1e-6:
            along_targets.append(along_zero)
        color = current.get("rgba", (0.7, 0.7, 0.7, 0.7))
        placed.pop(zone_idx)
        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for along in along_targets:
                cx, cy = _wall_attached_center(
                    attached_wall, along, sx, sy,
                    attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
                )
                cand = _make_candidate(str(zone_name), sx, sy, cx, cy, color)
                if not _can_place(cand, gap):
                    continue
                placed.append(cand)
                return True
        placed.insert(zone_idx, current)
        return False

    for zone_name in ("STORAGE", "FACTORY", "LOADING", "FORKLIFT_PARK", "MACHINING_CELL"):
        _try_center_zone_if_isolated_on_wall(zone_name)

    # --- Post-processing: corner snapping for longitudinal zones ---
    def _try_relocate_longitudinal_zone_to_corner(zone_name):
        zone_idx = next((i for i, a in enumerate(placed) if str(a.get("name", "")) == str(zone_name)), None)
        if zone_idx is None:
            return False
        current = placed[zone_idx]
        current_wall = _candidate_attached_wall(current)
        if current_wall not in longitudinal_side_walls:
            return False
        placed.pop(zone_idx)
        base_sx = float(current.get("sx", 0.0))
        base_sy = float(current.get("sy", 0.0))
        color = current.get("rgba", (0.7, 0.7, 0.7, 0.7))
        wall_order = [current_wall] + [w for w in longitudinal_side_walls if w != current_wall]
        if str(zone_name).upper() == "FACTORY":
            shrink_steps = (1.0, 0.95, 0.90, 0.85, 0.80)
        else:
            shrink_steps = (1.0, 0.95, 0.90, 0.85)

        for shrink in shrink_steps:
            sx_try, sy_try = _scaled_dims_for_zone(zone_name, base_sx, base_sy, shrink)
            for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
                for wall in wall_order:
                    sx_on_wall, sy_on_wall = _orient_dims_long_side_on_wall(wall, sx_try, sy_try)
                    lo, hi = _wall_along_limits(
                        wall, sx_on_wall, sy_on_wall,
                        attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
                    )
                if hi < lo:
                    continue
                span = max(0.0, hi - lo)
                end_band = max(1.2, min(span, 6.5))
                if wall in ("north", "south"):
                    current_along = float(current.get("cx", 0.0))
                else:
                    current_along = float(current.get("cy", 0.0))
                if abs(current_along - lo) <= abs(current_along - hi):
                    end_first, end_second = lo, hi
                else:
                    end_first, end_second = hi, lo
                along_prefs = (
                    end_first,
                    end_second,
                    end_first + (0.5 * end_band) if end_first <= end_second else end_first - (0.5 * end_band),
                    end_second - (0.5 * end_band) if end_first <= end_second else end_second + (0.5 * end_band),
                )
                for along_pref in along_prefs:
                    cand = _place_on_wall(
                        name=str(zone_name), sx=sx_try, sy=sy_try,
                        wall=wall, color=color, along_pref=along_pref,
                        tries=0, gap=gap, deterministic_first=True,
                    )
                    if cand is None:
                        continue
                    if not _is_at_wall_end(cand):
                        continue
                    placed.append(cand)
                    return True

        placed.insert(zone_idx, current)
        return False

    def _snap_longitudinal_zone_to_exact_end(zone_name):
        zone_idx = next((i for i, a in enumerate(placed) if str(a.get("name", "")) == str(zone_name)), None)
        if zone_idx is None:
            return False
        current = placed[zone_idx]
        current_wall = _candidate_attached_wall(current)
        if current_wall not in longitudinal_side_walls:
            return False
        sx = float(current.get("sx", 0.0))
        sy = float(current.get("sy", 0.0))
        lo, hi = _wall_along_limits(
            current_wall, sx, sy,
            attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return False
        if current_wall in ("north", "south"):
            current_along = float(current.get("cx", 0.0))
        else:
            current_along = float(current.get("cy", 0.0))
        end_targets = [lo, hi]
        end_targets.sort(key=lambda v: abs(float(v) - current_along))
        placed.pop(zone_idx)
        color = current.get("rgba", (0.7, 0.7, 0.7, 0.7))
        for gap in (AREA_LAYOUT_MIN_GAP, 0.0):
            for along in end_targets:
                cx, cy = _wall_attached_center(
                    current_wall, along, sx, sy,
                    attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
                )
                cand = _make_candidate(str(zone_name), sx, sy, cx, cy, color)
                if not _can_place(cand, gap):
                    continue
                placed.append(cand)
                return True
        placed.insert(zone_idx, current)
        return False

    # --- Build result layout dict ---
    layout = {
        area["name"]: {
            "sx": float(area["sx"]),
            "sy": float(area["sy"]),
            "cx": float(area["cx"]),
            "cy": float(area["cy"]),
        }
        for area in placed
    }

    if SHOW_AREA_LAYOUT_MARKERS:
        z = floor_top_z + AREA_LAYOUT_TILE_HALF_Z + 0.005
        for area in placed:
            name = area["name"]
            sx = area["sx"]
            sy = area["sy"]
            cx = area["cx"]
            cy = area["cy"]
            if name in ("LOADING", "STORAGE", "FACTORY"):
                continue
            if ENABLE_FACTORY_BARRIER_RING and name == "FACTORY":
                continue
            if name == "FORKLIFT_PARK" and ENABLE_FORKLIFT_PARK_SLOT_LINES:
                continue
            if name == "OFFICE" and ENABLE_EMBEDDED_OFFICE_MAP:
                continue
            vid = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[sx * 0.5, sy * 0.5, AREA_LAYOUT_TILE_HALF_Z],
                rgbaColor=list(area["rgba"]),
                physicsClientId=cli,
            )
            body_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=vid,
                basePosition=[cx, cy, z],
                useMaximalCoordinates=True,
                physicsClientId=cli,
            )
            p.changeVisualShape(
                body_id, -1,
                rgbaColor=list(area["rgba"]),
                textureUniqueId=-1,
                specularColor=list(UNIFORM_SPECULAR_COLOR),
                physicsClientId=cli,
            )

    return layout
