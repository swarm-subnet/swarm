from __future__ import annotations

import math
import math as _math
import random
from typing import Optional, Tuple

import pybullet as p

from swarm import constants as C
from .platform import build_goal_platform, build_start_platform


def _find_flat_platform_spot(
    cli: int,
    x: float,
    y: float,
    radius: float,
    *,
    nudge_distance: float = 2.0,
    nudge_attempts: int = 8,
    clearance: float = 0.15,
    orbit_radius: float = 0.0,
    orbit_ring_samples: int = 16,
) -> tuple:
    """Find the best (x, y, surface_z) for a platform on mountain terrain.

    Samples terrain height at platform edges. If the terrain slopes into the
    platform, tries nudging in the direction away from the highest edge.
    Returns (best_x, best_y, safe_surface_z) where safe_surface_z is the max
    terrain height around the platform plus clearance.

    When orbit_radius > 0 the helper also samples a swept disk out to
    orbit_radius + 0.3 (avoidance margin) + radius (landing pad), so a moving
    platform's full path stays above the terrain at every angle.
    """

    def _sample(cx, cy):
        center_z = _raycast_surface_z(cli, cx, cy)
        edge_zs = [center_z]
        for angle_deg in range(0, 360, 45):
            rad = _math.radians(angle_deg)
            ex = cx + radius * _math.cos(rad)
            ey = cy + radius * _math.sin(rad)
            ez = _raycast_surface_z(cli, ex, ey)
            edge_zs.append(ez)
        max_z = max(edge_zs)
        slope = max_z - center_z
        return center_z, max_z, slope, edge_zs

    best_x, best_y = x, y
    center_z, max_z, slope, edge_zs = _sample(x, y)
    best_slope = slope
    best_max_z = max_z

    if slope > 0.3:
        for attempt in range(nudge_attempts):
            angle = attempt * (360.0 / nudge_attempts)
            rad = _math.radians(angle)
            nx = x + nudge_distance * _math.cos(rad)
            ny = y + nudge_distance * _math.sin(rad)
            nc_z, nm_z, ns, _ = _sample(nx, ny)
            if ns < best_slope:
                best_slope = ns
                best_x, best_y = nx, ny
                best_max_z = nm_z

        if best_slope > 0.3 and nudge_distance < 4.0:
            for attempt in range(nudge_attempts):
                angle = attempt * (360.0 / nudge_attempts)
                rad = _math.radians(angle)
                nx = x + nudge_distance * 2.0 * _math.cos(rad)
                ny = y + nudge_distance * 2.0 * _math.sin(rad)
                nc_z, nm_z, ns, _ = _sample(nx, ny)
                if ns < best_slope:
                    best_slope = ns
                    best_x, best_y = nx, ny
                    best_max_z = nm_z

    _, final_max_z, _, _ = _sample(best_x, best_y)

    if orbit_radius > 0:
        outer = orbit_radius + 0.3 + radius
        for r_factor, n in (
            (1.00, orbit_ring_samples),
            (0.66, max(8, orbit_ring_samples // 2)),
            (0.33, max(6, orbit_ring_samples // 4)),
        ):
            ring = outer * r_factor
            for k in range(n):
                ang = 2 * _math.pi * k / n
                ex = best_x + ring * _math.cos(ang)
                ey = best_y + ring * _math.sin(ang)
                ez = _raycast_surface_z(cli, ex, ey)
                if ez > final_max_z:
                    final_max_z = ez

    safe_z = final_max_z + clearance
    return best_x, best_y, safe_z


def _raycast_surface_z(cli: int, x: float, y: float) -> float:
    result = p.rayTest(
        rayFromPosition=[x, y, 500.0],
        rayToPosition=[x, y, -100.0],
        physicsClientId=cli,
    )
    if result and result[0][0] != -1:
        return result[0][3][2]
    return 0.0


def _find_clear_platform_position(
    cli: int,
    candidate_x: float,
    candidate_y: float,
    candidate_z: float,
    rng: random.Random,
    body_count_before: int,
    world_range_x: float,
    world_range_y: float,
    h_min: float,
    h_max: float,
    avoid_pos: Optional[Tuple[float, float, float]] = None,
    min_distance: float = 0.0,
    required_distance_min: Optional[float] = None,
    required_distance_max: Optional[float] = None,
    preferred_distance: Optional[float] = None,
    distance_mode: str = "xyz",
    allow_candidate_fallback: bool = True,
    min_obstacle_height: float = 0.0,
) -> Tuple[float, float, float]:
    clearance = C.TYPE_4_PLATFORM_CLEARANCE
    platform_r = C.START_PLATFORM_RADIUS
    check_r = platform_r + clearance
    wx, wy = world_range_x, world_range_y

    skip_bodies: Optional[set] = None
    if min_obstacle_height > 0:
        skip_bodies = set()
        for bid in range(body_count_before, p.getNumBodies(physicsClientId=cli)):
            mn, mx = p.getAABB(bid, physicsClientId=cli)
            if (mx[2] - mn[2]) <= min_obstacle_height:
                skip_bodies.add(bid)

    def _distance(
        x1: float,
        y1: float,
        z1: float,
        x2: float,
        y2: float,
        z2: float,
    ) -> float:
        dx = x1 - x2
        dy = y1 - y2
        if distance_mode == "xy":
            return math.hypot(dx, dy)
        dz = z1 - z2
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _overlaps(x: float, y: float, z: float) -> bool:
        probe_col = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=check_r,
            physicsClientId=cli,
        )
        probe_uid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=probe_col,
            basePosition=[x, y, z],
            physicsClientId=cli,
        )
        overlapping = False
        for body_id in range(body_count_before, p.getNumBodies(physicsClientId=cli)):
            if body_id == probe_uid:
                continue
            if skip_bodies is not None and body_id in skip_bodies:
                continue
            contacts = p.getClosestPoints(
                probe_uid,
                body_id,
                distance=0.0,
                physicsClientId=cli,
            )
            if contacts:
                overlapping = True
                break
        p.removeBody(probe_uid, physicsClientId=cli)
        return overlapping

    def _too_close(x: float, y: float, z: float) -> bool:
        if avoid_pos is None or min_distance <= 0:
            return False
        return _distance(x, y, z, avoid_pos[0], avoid_pos[1], avoid_pos[2]) < min_distance

    def _within_distance_bounds(x: float, y: float, z: float) -> bool:
        if avoid_pos is None:
            return True
        dist = _distance(x, y, z, avoid_pos[0], avoid_pos[1], avoid_pos[2])
        if required_distance_min is not None and dist < required_distance_min:
            return False
        if required_distance_max is not None and dist > required_distance_max:
            return False
        return True

    def _candidate_is_valid(x: float, y: float, z: float) -> bool:
        if not (-wx <= x <= wx and -wy <= y <= wy):
            return False
        if _overlaps(x, y, z):
            return False
        if _too_close(x, y, z):
            return False
        return _within_distance_bounds(x, y, z)

    def _sample_bounded_candidate() -> Optional[Tuple[float, float, float]]:
        if avoid_pos is None:
            return None
        if required_distance_min is None and required_distance_max is None:
            return None

        min_bound = 0.0 if required_distance_min is None else float(required_distance_min)
        if required_distance_max is None:
            return None
        max_bound = float(required_distance_max)

        def _target_xy_radius(z: float, min_xy: float, max_xy: float) -> float:
            if preferred_distance is None:
                return min_xy + 0.5 * (max_xy - min_xy)
            target = float(preferred_distance)
            if distance_mode != "xy":
                dz = z - avoid_pos[2]
                target_sq = target * target - dz * dz
                target = min_xy if target_sq <= 0.0 else math.sqrt(target_sq)
            return max(min_xy, min(max_xy, target))

        for _ in range(C.TYPE_4_PLATFORM_MAX_ATTEMPTS):
            z = rng.uniform(h_min, h_max)
            dz = 0.0 if distance_mode == "xy" else (z - avoid_pos[2])
            max_xy_sq = max_bound * max_bound - dz * dz
            if max_xy_sq < 0.0:
                continue
            min_xy_sq = max(0.0, min_bound * min_bound - dz * dz)
            max_xy = math.sqrt(max_xy_sq)
            min_xy = math.sqrt(min_xy_sq)
            if max_xy < min_xy:
                continue

            angle = rng.uniform(0.0, 2.0 * math.pi)
            target_xy = _target_xy_radius(z, min_xy, max_xy)
            radius = rng.triangular(min_xy, max_xy, target_xy)
            x = avoid_pos[0] + radius * math.cos(angle)
            y = avoid_pos[1] + radius * math.sin(angle)
            if _candidate_is_valid(x, y, z):
                return x, y, z

        # Deterministic sweep as a last attempt before failing.
        z_candidates = [candidate_z, max(h_min, min(h_max, avoid_pos[2]))]
        if h_min <= h_max:
            z_candidates.extend([h_min, h_max, (h_min + h_max) / 2.0])

        seen_z = set()
        unique_z_candidates = []
        for z in z_candidates:
            z_key = round(float(z), 6)
            if z_key in seen_z:
                continue
            seen_z.add(z_key)
            unique_z_candidates.append(float(z))

        radius_steps = 16
        angle_steps = 72
        for z in unique_z_candidates:
            dz = 0.0 if distance_mode == "xy" else (z - avoid_pos[2])
            max_xy_sq = max_bound * max_bound - dz * dz
            if max_xy_sq < 0.0:
                continue
            min_xy_sq = max(0.0, min_bound * min_bound - dz * dz)
            max_xy = math.sqrt(max_xy_sq)
            min_xy = math.sqrt(min_xy_sq)
            if max_xy < min_xy:
                continue
            target_xy = _target_xy_radius(z, min_xy, max_xy)
            radius_candidates = []
            for radius_idx in range(radius_steps):
                frac = 0.0 if radius_steps == 1 else radius_idx / (radius_steps - 1)
                radius = min_xy + frac * (max_xy - min_xy)
                radius_candidates.append(radius)
            radius_candidates.sort(key=lambda radius: (abs(radius - target_xy), radius))
            for radius in radius_candidates:
                for angle_idx in range(angle_steps):
                    angle = (2.0 * math.pi * angle_idx) / angle_steps
                    x = avoid_pos[0] + radius * math.cos(angle)
                    y = avoid_pos[1] + radius * math.sin(angle)
                    if _candidate_is_valid(x, y, z):
                        return x, y, z

        return None

    if _candidate_is_valid(candidate_x, candidate_y, candidate_z):
        return candidate_x, candidate_y, candidate_z

    bounded_candidate = _sample_bounded_candidate()
    if bounded_candidate is not None:
        return bounded_candidate

    for _ in range(C.TYPE_4_PLATFORM_MAX_ATTEMPTS):
        x = rng.uniform(-wx, wx)
        y = rng.uniform(-wy, wy)
        z = rng.uniform(h_min, h_max)
        if _candidate_is_valid(x, y, z):
            return x, y, z

    if not allow_candidate_fallback:
        raise RuntimeError("unable to find collision-free platform position within distance bounds")

    return candidate_x, candidate_y, candidate_z


def _goal_distance_bounds(challenge_type: int) -> Optional[Tuple[float, float, str]]:
    if challenge_type == 1:
        return (C.TYPE_1_R_MIN, C.TYPE_1_R_MAX, "xy")
    if challenge_type == 4:
        return (C.TYPE_3_R_MIN, C.TYPE_3_R_MAX, "xy")
    if challenge_type == 5:
        return (C.TYPE_4_R_MIN, C.TYPE_4_R_MAX, "xy")
    if challenge_type == 6:
        return (C.TYPE_6_R_MIN, C.TYPE_6_R_MAX, "xy")
    return None


def _distance_between_points(a, b, *, mode: str) -> float:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    if mode == "xy":
        return math.hypot(dx, dy)
    dz = float(b[2]) - float(a[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def build_autopilot_world(tagger, cli, seed, start, goal, challenge_type,
                          moving_platform, static_world_body_base):
    """Place and build the autopilot start + goal platforms on an already-built world.

    The static world must already exist; ``static_world_body_base`` is the body count
    captured before it was built (so the placement scan only sees world obstacles).
    Returns (end_platform_uids, start_platform_uids, start_surface_z, goal_surface_z,
    adjusted_start, adjusted_goal).
    """
    rng = random.Random(seed)

    sx = sy = sz = None
    if start is not None:
        sx, sy, sz = start
    gx = gy = gz = None
    if goal is not None:
        gx, gy, gz = goal

    start_surface_z = None
    goal_surface_z = None
    adjusted_start = None
    adjusted_goal = None
    start_platform_uids: list = []
    end_platform_uids: list = []

    collision_scan_types = {
        1: (C.TYPE_1_WORLD_RANGE, C.TYPE_1_WORLD_RANGE, C.TYPE_1_H_MIN, C.TYPE_1_H_MAX),
        4: (C.TYPE_3_VILLAGE_RANGE, C.TYPE_3_VILLAGE_RANGE, 0.0, 0.0),
        5: (C.TYPE_4_WORLD_RANGE_X, C.TYPE_4_WORLD_RANGE_Y, C.TYPE_4_H_MIN, C.TYPE_4_H_MAX),
        6: (C.TYPE_6_WORLD_RANGE, C.TYPE_6_WORLD_RANGE, C.TYPE_6_H_MIN, C.TYPE_6_H_MAX),
    }
    _VILLAGE_MIN_OBSTACLE_HEIGHT = 1.0

    if challenge_type in collision_scan_types and sx is not None and sy is not None and sz is not None:
        _wx, _wy, _hmin, _hmax = collision_scan_types[challenge_type]
        placement_rng = random.Random(seed + 777777)
        obstacle_height_filter = _VILLAGE_MIN_OBSTACLE_HEIGHT if challenge_type == 4 else 0.0

        start_surface = sz - C.START_PLATFORM_TAKEOFF_BUFFER
        new_sx, new_sy, new_s_surface = _find_clear_platform_position(
            cli, sx, sy, start_surface, placement_rng, static_world_body_base,
            world_range_x=_wx, world_range_y=_wy, h_min=_hmin, h_max=_hmax,
            min_obstacle_height=obstacle_height_filter,
        )
        sx, sy = new_sx, new_sy
        if challenge_type == 4:
            new_s_surface = float(_raycast_surface_z(cli, sx, sy))
            for bid in range(static_world_body_base, p.getNumBodies(physicsClientId=cli)):
                mn, mx = p.getAABB(bid, physicsClientId=cli)
                if mn[0] <= sx <= mx[0] and mn[1] <= sy <= mx[1] and mx[2] > new_s_surface:
                    new_s_surface = mx[2]
            sz = new_s_surface + C.START_PLATFORM_HEIGHT + 0.15
        else:
            sz = new_s_surface + C.START_PLATFORM_TAKEOFF_BUFFER
        start_surface_z = new_s_surface
        adjusted_start = (sx, sy, sz)

        if gx is not None and gy is not None and gz is not None:
            goal_bounds = _goal_distance_bounds(challenge_type)
            required_distance_min = required_distance_max = None
            distance_mode = "xyz"
            if goal_bounds is not None:
                required_distance_min, required_distance_max, distance_mode = goal_bounds
            preferred_distance = _distance_between_points(
                (float(start[0]), float(start[1]), float(start[2])),
                (float(goal[0]), float(goal[1]), float(goal[2])),
                mode=distance_mode,
            )
            new_gx, new_gy, new_gz = _find_clear_platform_position(
                cli, gx, gy, gz, placement_rng, static_world_body_base,
                world_range_x=_wx, world_range_y=_wy, h_min=_hmin, h_max=_hmax,
                avoid_pos=(sx, sy, sz), min_distance=required_distance_min or 0.0,
                required_distance_min=required_distance_min,
                required_distance_max=required_distance_max,
                preferred_distance=preferred_distance, distance_mode=distance_mode,
                allow_candidate_fallback=challenge_type == 4,
                min_obstacle_height=obstacle_height_filter,
            )
            if challenge_type == 4:
                new_gz = float(_raycast_surface_z(cli, new_gx, new_gy))
                for bid in range(static_world_body_base, p.getNumBodies(physicsClientId=cli)):
                    mn, mx = p.getAABB(bid, physicsClientId=cli)
                    if mn[0] <= new_gx <= mx[0] and mn[1] <= new_gy <= mx[1] and mx[2] > new_gz:
                        new_gz = mx[2]
            gx, gy, gz = new_gx, new_gy, new_gz
            adjusted_goal = (gx, gy, gz)

    if sx is not None and sy is not None and sz is not None:
        if challenge_type in (2, 3):
            sx, sy, surface_z = _find_flat_platform_spot(cli, sx, sy, C.START_PLATFORM_RADIUS)
            adjusted_start = (sx, sy, surface_z + C.START_PLATFORM_TAKEOFF_BUFFER)
        elif challenge_type in (1, 4, 5, 6) and start_surface_z is not None:
            surface_z = start_surface_z
        else:
            surface_z = _raycast_surface_z(cli, sx, sy)

        start_platform_uids, start_top_z = build_start_platform(
            tagger, cli, sx, sy, surface_z, challenge_type,
        )
        start_surface_z = start_top_z if challenge_type == 4 else surface_z

    if gx is not None and gy is not None and gz is not None:
        if challenge_type in (2, 3):
            orbit_r = C.PLATFORM_RADIUS_MAX if moving_platform else 0.0
            gx, gy, surface_z = _find_flat_platform_spot(
                cli, gx, gy, C.START_PLATFORM_RADIUS, orbit_radius=orbit_r,
            )
            adjusted_goal = (gx, gy, surface_z)
        else:
            surface_z = gz
            if challenge_type == 4 and moving_platform:
                outer = C.PLATFORM_RADIUS_MAX + 0.3 + C.LANDING_PLATFORM_RADIUS
                max_top = surface_z
                exclude_uids = set(start_platform_uids)
                n_bodies = p.getNumBodies(physicsClientId=cli)
                for body_idx in range(static_world_body_base, n_bodies):
                    uid = p.getBodyUniqueId(body_idx, physicsClientId=cli)
                    if uid in exclude_uids:
                        continue
                    mn, mx = p.getAABB(uid, physicsClientId=cli)
                    if (mx[0] - mn[0]) > 50.0 or (mx[1] - mn[1]) > 50.0:
                        continue
                    cx = max(mn[0], min(gx, mx[0]))
                    cy = max(mn[1], min(gy, mx[1]))
                    if (gx - cx) ** 2 + (gy - cy) ** 2 <= outer * outer and mx[2] > max_top:
                        max_top = mx[2]
                surface_z = max(surface_z, max_top + 0.20)
                adjusted_goal = (gx, gy, surface_z)

        end_platform_uids, goal_top_z = build_goal_platform(
            tagger, cli, gx, gy, surface_z, challenge_type, rng,
        )
        goal_surface_z = goal_top_z

    if challenge_type == 4:
        all_plat = set(start_platform_uids) | set(end_platform_uids)
        to_remove = set()
        for plat_uid in list(all_plat):
            for i in range(p.getNumBodies(physicsClientId=cli)):
                bid = p.getBodyUniqueId(i, physicsClientId=cli)
                if bid in all_plat or bid < static_world_body_base:
                    continue
                mn, mx = p.getAABB(bid, physicsClientId=cli)
                if (mx[2] - mn[2]) > 2.0:
                    continue
                contacts = p.getClosestPoints(plat_uid, bid, distance=0.0, physicsClientId=cli)
                if contacts and min(c[8] for c in contacts) < -0.03:
                    to_remove.add(bid)
        for bid in to_remove:
            p.removeBody(bid, physicsClientId=cli)
            tagger.body_tags.pop(bid, None)

    return (
        end_platform_uids,
        start_platform_uids,
        start_surface_z,
        goal_surface_z,
        adjusted_start,
        adjusted_goal,
    )
