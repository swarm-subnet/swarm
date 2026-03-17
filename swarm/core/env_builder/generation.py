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
) -> shared.Tuple[float, float, float]:
    clearance = shared.TYPE_4_PLATFORM_CLEARANCE
    platform_r = shared.START_PLATFORM_RADIUS
    check_r = platform_r + clearance
    wx, wy = world_range_x, world_range_y

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
        dx = x - avoid_pos[0]
        dy = y - avoid_pos[1]
        dz = z - avoid_pos[2]
        return shared.math.sqrt(dx * dx + dy * dy + dz * dz) < min_distance

    if not _overlaps(candidate_x, candidate_y, candidate_z) and not _too_close(
        candidate_x, candidate_y, candidate_z
    ):
        return candidate_x, candidate_y, candidate_z

    for _ in range(shared.TYPE_4_PLATFORM_MAX_ATTEMPTS):
        x = rng.uniform(-wx, wx)
        y = rng.uniform(-wy, wy)
        z = rng.uniform(h_min, h_max)
        if not _overlaps(x, y, z) and not _too_close(x, y, z):
            return x, y, z

    return candidate_x, candidate_y, candidate_z
