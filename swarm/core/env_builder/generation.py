from __future__ import annotations

from . import _shared as shared


def _raycast_surface_z(cli: int, x: float, y: float) -> float:
    result = shared.p.rayTest(
        rayFromPosition=[x, y, 500.0],
        rayToPosition=[x, y, -100.0],
        physicsClientId=cli,
    )
    if result and result[0][0] != -1:
        return result[0][3][2]
    return 0.0


def _add_box(cli: int, pos, size, yaw) -> None:
    col = shared.p.createCollisionShape(
        shared.p.GEOM_BOX, halfExtents=[s / 2 for s in size], physicsClientId=cli
    )
    vis = shared.p.createVisualShape(
        shared.p.GEOM_BOX,
        halfExtents=[s / 2 for s in size],
        rgbaColor=[0.2, 0.6, 0.8, 1.0],
        physicsClientId=cli,
    )
    quat = shared.p.getQuaternionFromEuler([0, 0, yaw])
    shared.p.createMultiBody(
        0,
        col,
        vis,
        basePosition=pos,
        baseOrientation=quat,
        physicsClientId=cli,
    )


def _get_tao_tex(cli: int) -> int:
    if cli not in shared._tao_tex_id:
        tex_path = (
            shared.Path(__file__).resolve().parent.parent.parent
            / "assets"
            / "tao.png"
        )
        shared._tao_tex_id[cli] = shared.p.loadTexture(str(tex_path))
    return shared._tao_tex_id[cli]


def _build_static_world(
    seed: int,
    cli: int,
    *,
    start: shared.Optional[shared.Tuple[float, float, float]],
    goal: shared.Optional[shared.Tuple[float, float, float]],
    challenge_type: int,
) -> None:
    if challenge_type == 1:
        shared.build_city_map(cli, seed, [], 0.0)
    elif challenge_type == 2:
        shared.build_open_world(cli, seed, start=start, goal=goal)
    elif challenge_type == 3:
        shared.build_mountain_map(cli, seed, [], 0.0)
    elif challenge_type == 4:
        shared.build_village_map(cli, seed, [], 0.0)
    elif challenge_type == 5:
        shared.build_warehouse_map(seed=seed, cli=cli, start=start, goal=goal)
    elif challenge_type == 6:
        safe_zones = []
        if start is not None:
            safe_zones.append(start)
        if goal is not None:
            safe_zones.append(goal)
        # Forest maps need a much larger reserved XY clearing than other maps.
        # A small radius removes the trunk base but can still leave canopy or
        # adjacent tree meshes close enough to destabilize the first step.
        forest_safe_zone_radius = max(
            8.0,
            shared.SAFE_ZONE_RADIUS + max(
                shared.START_PLATFORM_RADIUS,
                shared.LANDING_PLATFORM_RADIUS,
            ),
        )
        shared.build_forest_map(cli, seed, safe_zones, forest_safe_zone_radius)


def _find_clear_platform_position(
    cli: int,
    candidate_x: float,
    candidate_y: float,
    candidate_z: float,
    rng: shared.random.Random,
    body_count_before: int,
    world_range_x: float,
    world_range_y: float,
    h_min: float,
    h_max: float,
    avoid_pos: shared.Optional[shared.Tuple[float, float, float]] = None,
    min_distance: float = 0.0,
    required_distance_min: shared.Optional[float] = None,
    required_distance_max: shared.Optional[float] = None,
    preferred_distance: shared.Optional[float] = None,
    distance_mode: str = "xyz",
    allow_candidate_fallback: bool = True,
) -> shared.Tuple[float, float, float]:
    clearance = shared.TYPE_4_PLATFORM_CLEARANCE
    platform_r = shared.START_PLATFORM_RADIUS
    check_r = platform_r + clearance
    wx, wy = world_range_x, world_range_y

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
            return shared.math.hypot(dx, dy)
        dz = z1 - z2
        return shared.math.sqrt(dx * dx + dy * dy + dz * dz)

    def _overlaps(x: float, y: float, z: float) -> bool:
        probe_col = shared.p.createCollisionShape(
            shared.p.GEOM_SPHERE,
            radius=check_r,
            physicsClientId=cli,
        )
        probe_uid = shared.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=probe_col,
            basePosition=[x, y, z],
            physicsClientId=cli,
        )
        overlapping = False
        for body_id in range(body_count_before, shared.p.getNumBodies(physicsClientId=cli)):
            if body_id == probe_uid:
                continue
            contacts = shared.p.getClosestPoints(
                probe_uid,
                body_id,
                distance=0.0,
                physicsClientId=cli,
            )
            if contacts:
                overlapping = True
                break
        shared.p.removeBody(probe_uid, physicsClientId=cli)
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

    def _sample_bounded_candidate() -> shared.Optional[shared.Tuple[float, float, float]]:
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
                target = min_xy if target_sq <= 0.0 else shared.math.sqrt(target_sq)
            return max(min_xy, min(max_xy, target))

        for _ in range(shared.TYPE_4_PLATFORM_MAX_ATTEMPTS):
            z = rng.uniform(h_min, h_max)
            dz = 0.0 if distance_mode == "xy" else (z - avoid_pos[2])
            max_xy_sq = max_bound * max_bound - dz * dz
            if max_xy_sq < 0.0:
                continue
            min_xy_sq = max(0.0, min_bound * min_bound - dz * dz)
            max_xy = shared.math.sqrt(max_xy_sq)
            min_xy = shared.math.sqrt(min_xy_sq)
            if max_xy < min_xy:
                continue

            angle = rng.uniform(0.0, 2.0 * shared.math.pi)
            target_xy = _target_xy_radius(z, min_xy, max_xy)
            radius = rng.triangular(min_xy, max_xy, target_xy)
            x = avoid_pos[0] + radius * shared.math.cos(angle)
            y = avoid_pos[1] + radius * shared.math.sin(angle)
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
            max_xy = shared.math.sqrt(max_xy_sq)
            min_xy = shared.math.sqrt(min_xy_sq)
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
                    angle = (2.0 * shared.math.pi * angle_idx) / angle_steps
                    x = avoid_pos[0] + radius * shared.math.cos(angle)
                    y = avoid_pos[1] + radius * shared.math.sin(angle)
                    if _candidate_is_valid(x, y, z):
                        return x, y, z

        return None

    if _candidate_is_valid(candidate_x, candidate_y, candidate_z):
        return candidate_x, candidate_y, candidate_z

    bounded_candidate = _sample_bounded_candidate()
    if bounded_candidate is not None:
        return bounded_candidate

    for _ in range(shared.TYPE_4_PLATFORM_MAX_ATTEMPTS):
        x = rng.uniform(-wx, wx)
        y = rng.uniform(-wy, wy)
        z = rng.uniform(h_min, h_max)
        if _candidate_is_valid(x, y, z):
            return x, y, z

    if not allow_candidate_fallback:
        raise RuntimeError("unable to find collision-free platform position within distance bounds")

    return candidate_x, candidate_y, candidate_z
