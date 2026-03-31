from types import SimpleNamespace

from ..constants import (
    AREA_LAYOUT_EDGE_MARGIN,
    AREA_LAYOUT_MIN_GAP,
    PERSONNEL_DOOR_CLEAR_EXTRA_ALONG,
    WALL_SLOTS,
)
from ..helpers import (
    _attached_wall_from_area_bounds,
    _rect_bounds,
    _rects_overlap,
    _sample_random_center,
    _size_fits_half_span,
    _wall_along_limits,
    _wall_attached_center,
)


def make_core_layout_helpers(
    *,
    area_defs,
    attach_half_x,
    attach_half_y,
    major_zone_fixed_short_side_m,
    placed,
    personnel_side,
    personnel_span,
    personnel_along,
    opposite_personnel_side,
    critical_zone_keepout_rect,
    critical_door_blocking_zones,
    rng,
):
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
            along_val = float(candidate.get("cx", 0.0))
        else:
            along_val = float(candidate.get("cy", 0.0))
        targets = _opposite_wall_end_targets(
            attached_wall,
            float(candidate.get("sx", 0.0)),
            float(candidate.get("sy", 0.0)),
        )
        if targets is None:
            return False
        lo, hi = targets
        if abs(hi - lo) <= 1e-6:
            return True
        target_along = hi if personnel_along <= 0.0 else lo
        tol = max(
            0.6,
            0.10
            * (
                float(candidate.get("sx", 0.0))
                if attached_wall in ("north", "south")
                else float(candidate.get("sy", 0.0))
            ),
        )
        return abs(along_val - target_along) <= tol

    def _is_at_wall_end(candidate, end_tol_factor=0.16):
        attached_wall = _candidate_attached_wall(candidate)
        if attached_wall not in WALL_SLOTS:
            return False
        targets = _opposite_wall_end_targets(
            attached_wall,
            float(candidate.get("sx", 0.0)),
            float(candidate.get("sy", 0.0)),
        )
        if targets is None:
            return False
        lo, hi = targets
        if attached_wall in ("north", "south"):
            along_val = float(candidate.get("cx", 0.0))
            span = float(candidate.get("sx", 0.0))
        else:
            along_val = float(candidate.get("cy", 0.0))
            span = float(candidate.get("sy", 0.0))
        tol = max(0.5, end_tol_factor * span)
        return abs(along_val - lo) <= tol or abs(along_val - hi) <= tol

    def _can_place_static(candidate, gap):
        if not bool(candidate.get("_fits_attach_span", True)):
            return False
        for other in placed:
            if _rects_overlap(candidate, other, gap):
                return False
        return True

    def _can_place(candidate, gap):
        if not _can_place_static(candidate, gap):
            return False
        if not _is_far_from_personnel_door_on_same_wall(candidate):
            return False
        return True

    def _place_on_wall(
        *,
        name,
        sx,
        sy,
        wall,
        color,
        along_pref=None,
        tries=220,
        gap=AREA_LAYOUT_MIN_GAP,
        validator=None,
        deterministic_first=True,
    ):
        if wall not in WALL_SLOTS:
            return None
        sx = float(sx)
        sy = float(sy)
        if not _size_fits_half_span(
            sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        ):
            return None
        lo, hi = _wall_along_limits(
            wall, sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
        )
        if hi < lo:
            return None

        def _candidate_for_along(along):
            cx, cy = _wall_attached_center(
                wall, along, sx, sy, attach_half_x, attach_half_y, AREA_LAYOUT_EDGE_MARGIN,
            )
            cand = _make_candidate(name, sx, sy, cx, cy, color)
            if validator is not None and not validator(cand):
                return None
            if _can_place(cand, gap):
                return cand
            return None

        along_values = []
        seen = set()

        def _push_along(value):
            if value is None:
                return
            along = max(float(lo), min(float(hi), float(value)))
            key = round(along, 6)
            if key in seen:
                return
            seen.add(key)
            along_values.append(along)

        _push_along(along_pref)
        center = 0.5 * (float(lo) + float(hi))
        if deterministic_first:
            for value in (center, 0.0, lo, hi):
                _push_along(value)

        for _ in range(max(0, int(tries))):
            _push_along(rng.uniform(float(lo), float(hi)))

        if not deterministic_first:
            for value in (center, 0.0, lo, hi):
                _push_along(value)

        for along in along_values:
            cand = _candidate_for_along(along)
            if cand is not None:
                return cand
        return None

    def _place_anywhere(
        name, sx, sy, color, tries=1200, gap=AREA_LAYOUT_MIN_GAP, validator=None
    ):
        for _ in range(int(tries)):
            cx, cy = _sample_random_center(
                rng, float(sx), float(sy), attach_half_x, attach_half_y,
                AREA_LAYOUT_EDGE_MARGIN,
            )
            cand = _make_candidate(name, sx, sy, cx, cy, color)
            if validator is not None and not validator(cand):
                continue
            if _can_place(cand, gap):
                return cand
        return None

    def _force_place_zone(
        name,
        base_sx,
        base_sy,
        color,
        preferred_walls=None,
        along_pref=None,
        validator=None,
    ):
        preferred_walls = tuple(preferred_walls or ())
        preferred_alongs = ()
        if along_pref is None:
            preferred_alongs = ()
        elif isinstance(along_pref, (list, tuple)):
            preferred_alongs = tuple(along_pref)
        else:
            preferred_alongs = (along_pref,)
        shrink_sequence = (1.0, 0.94, 0.88, 0.82, 0.76, 0.70)
        for shrink in shrink_sequence:
            sx, sy = _scaled_dims_for_zone(name, base_sx, base_sy, shrink)
            for wall_name in preferred_walls:
                for along in preferred_alongs or (None,):
                    cand = _place_on_wall(
                        name=name,
                        sx=sx,
                        sy=sy,
                        color=color,
                        wall=wall_name,
                        along_pref=along,
                        validator=validator,
                    )
                    if cand is not None:
                        return cand
            cand = _place_anywhere(name, sx, sy, color, validator=validator)
            if cand is not None:
                return cand
        return None

    def _loading_zone_covers_doors(candidate):
        if critical_zone_keepout_rect is None:
            return False
        if str(candidate.get("name", "")).upper() not in critical_door_blocking_zones:
            return False
        return _rects_overlap(candidate, critical_zone_keepout_rect, 0.0)

    return SimpleNamespace(
        scaled_dims_for_zone=_scaled_dims_for_zone,
        candidate_attached_wall=_candidate_attached_wall,
        make_candidate=_make_candidate,
        is_far_from_personnel_door_on_same_wall=_is_far_from_personnel_door_on_same_wall,
        opposite_wall_end_targets=_opposite_wall_end_targets,
        is_at_preferred_opposite_end=_is_at_preferred_opposite_end,
        is_at_wall_end=_is_at_wall_end,
        can_place_static=_can_place_static,
        can_place=_can_place,
        place_on_wall=_place_on_wall,
        place_anywhere=_place_anywhere,
        force_place_zone=_force_place_zone,
        loading_zone_covers_doors=_loading_zone_covers_doors,
    )
