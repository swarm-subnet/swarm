from ._shared import *
from .visuals import _spawn_obj_with_mtl_parts, _truss_rib_x_positions


def build_overhead_cranes(
    crane_loader,
    crane_model_name,
    floor_top_z,
    roof_base_z,
    area_layout,
    shell_meshes,
    cli,
    seed=0,
):
    _ = seed
    if not ENABLE_OVERHEAD_CRANES:
        return {"overhead_cranes_enabled": False}
    if crane_loader is None or not crane_model_name:
        return {
            "overhead_cranes_enabled": False,
            "overhead_cranes_reason": "Crane model not found.",
        }

    zones = area_layout or {}
    targets = []
    for zone_name, count in OVERHEAD_CRANE_TARGET_BY_ZONE:
        if zone_name in zones:
            targets.append((str(zone_name), max(0, int(count))))
    if not targets:
        return {
            "overhead_cranes_enabled": False,
            "overhead_cranes_reason": "Target zones missing in layout.",
        }

    s = float(OVERHEAD_CRANE_SCALE_UNIFORM)
    scale_xyz = (s, s, s)
    min_v, max_v = model_bounds_xyz(crane_loader, crane_model_name, scale_xyz)
    crane_height = max(0.1, float(max_v[2] - min_v[2]))
    anchor_x = (float(min_v[0]) + float(max_v[0])) * 0.5
    anchor_y = (float(min_v[1]) + float(max_v[1])) * 0.5
    anchor_z = float(max_v[2])

    rib_xs = list(_truss_rib_x_positions(shell_meshes))
    if not rib_xs:
        rib_xs = [0.0]

    cranes = []
    min_spacing = max(0.0, float(OVERHEAD_CRANE_MIN_SPACING_M))
    edge_margin = max(0.0, float(OVERHEAD_CRANE_ZONE_EDGE_MARGIN_M))

    def _zone_span_candidates(area, count):
        sx = float(area["sx"])
        sy = float(area["sy"])
        cx = float(area["cx"])
        cy = float(area["cy"])
        hx = max(0.1, (sx * 0.5) - edge_margin)
        hy = max(0.1, (sy * 0.5) - edge_margin)

        if count <= 1:
            return [(cx, cy, 90.0 if sy > sx else 0.0)]
        if count == 2:
            fracs = (-0.34, 0.34)
        elif count == 3:
            fracs = (-0.38, 0.0, 0.38)
        else:
            fracs = [
                (-0.40 + (0.80 * float(i) / float(max(1, count - 1))))
                for i in range(count)
            ]

        out = []
        if sx >= sy:
            side_jitter = min(1.3, hy * 0.24)
            for i, f in enumerate(fracs):
                y_off = side_jitter if (i % 2) else -side_jitter
                out.append((cx + (f * hx), cy + y_off, 0.0))
        else:
            side_jitter = min(1.3, hx * 0.24)
            for i, f in enumerate(fracs):
                x_off = side_jitter if (i % 2) else -side_jitter
                out.append((cx + x_off, cy + (f * hy), 90.0))
        return out

    def _far_enough(x, y):
        for prev in cranes:
            if math.hypot(float(x) - float(prev["x"]), float(y) - float(prev["y"])) < (
                min_spacing - 1e-6
            ):
                return False
        return True

    for zone_name, count in targets:
        if count <= 0:
            continue
        area = zones.get(zone_name)
        if not area:
            continue
        sx = float(area["sx"])
        sy = float(area["sy"])
        cx = float(area["cx"])
        cy = float(area["cy"])
        min_x = cx - (sx * 0.5) + edge_margin
        max_x = cx + (sx * 0.5) - edge_margin
        min_y = cy - (sy * 0.5) + edge_margin
        max_y = cy + (sy * 0.5) - edge_margin
        preferred = _zone_span_candidates(area, count)

        for pref_x, pref_y, yaw_deg in preferred:
            y = max(min_y, min(max_y, float(pref_y)))
            rib_choices = [x for x in rib_xs if (min_x - 1e-6) <= x <= (max_x + 1e-6)]
            if rib_choices:
                rib_choices.sort(key=lambda x: abs(float(x) - float(pref_x)))
                x_candidates = rib_choices
            else:
                x_candidates = [max(min_x, min(max_x, float(pref_x)))]

            picked = None
            for x in x_candidates:
                if not _far_enough(x, y):
                    continue
                picked = (float(x), float(y), float(yaw_deg))
                break
            if picked is None and x_candidates:
                x = float(x_candidates[0])
                if _far_enough(x, y):
                    picked = (x, float(y), float(yaw_deg))
            if picked is None:
                continue

            x, y, yaw_deg = picked
            support_end_z = float(roof_base_z) + float(
                OVERHEAD_CRANE_TRUSS_TOUCH_EXTRA_M
            )
            anchor_world_z = float(support_end_z) - float(
                OVERHEAD_CRANE_ATTACH_CLEARANCE_M
            )
            if (anchor_world_z - crane_height) < (float(floor_top_z) + 1.2):
                anchor_world_z = float(floor_top_z) + 1.2 + crane_height

            yaw_deg = (float(yaw_deg) + float(OVERHEAD_CRANE_YAW_EXTRA_DEG)) % 360.0

            _spawn_obj_with_mtl_parts(
                loader=crane_loader,
                model_name=crane_model_name,
                world_anchor_xyz=(x, y, anchor_world_z),
                yaw_deg=yaw_deg,
                mesh_scale_xyz=scale_xyz,
                local_anchor_xyz=(anchor_x, anchor_y, anchor_z),
                cli=cli,
                with_collision=OVERHEAD_CRANE_WITH_COLLISION,
                fallback_rgba=(0.80, 0.72, 0.20, 1.0),
                rgba_gain=OVERHEAD_CRANE_COLOR_GAIN,
            )
            cranes.append(
                {
                    "zone": zone_name,
                    "x": x,
                    "y": y,
                    "z_top_anchor": float(anchor_world_z),
                    "z_hook_bottom": float(anchor_world_z - crane_height),
                    "yaw_deg": float(yaw_deg),
                    "scale_uniform": s,
                }
            )

    return {
        "overhead_cranes_enabled": len(cranes) > 0,
        "overhead_crane_model": crane_model_name,
        "overhead_crane_scale_uniform": s,
        "overhead_crane_count": len(cranes),
        "overhead_cranes": cranes,
    }
