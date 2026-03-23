from __future__ import annotations

from . import _shared as shared
from .generation import (
    _build_static_world,
    _find_clear_platform_position,
    _find_flat_platform_spot,
    _get_tao_tex,
    _raycast_surface_z,
)


def _goal_distance_bounds(
    challenge_type: int,
) -> shared.Optional[shared.Tuple[float, float, str]]:
    if challenge_type == 1:
        return (shared.TYPE_1_R_MIN, shared.TYPE_1_R_MAX, "xy")
    if challenge_type == 5:
        return (shared.TYPE_4_R_MIN, shared.TYPE_4_R_MAX, "xy")
    if challenge_type == 6:
        return (shared.TYPE_6_R_MIN, shared.TYPE_6_R_MAX, "xy")
    return None


def _distance_between_points(
    a: shared.Tuple[float, float, float],
    b: shared.Tuple[float, float, float],
    *,
    mode: str,
) -> float:
    dx = float(b[0]) - float(a[0])
    dy = float(b[1]) - float(a[1])
    if mode == "xy":
        return shared.math.hypot(dx, dy)
    dz = float(b[2]) - float(a[2])
    return shared.math.sqrt(dx * dx + dy * dy + dz * dz)


def build_world(
    seed: int,
    cli: int,
    *,
    start: shared.Optional[shared.Tuple[float, float, float]] = None,
    goal: shared.Optional[shared.Tuple[float, float, float]] = None,
    challenge_type: int = 1,
) -> shared.Tuple[
    shared.List[int],
    shared.List[int],
    shared.Optional[float],
    shared.Optional[float],
    shared.Optional[shared.Tuple[float, float, float]],
    shared.Optional[shared.Tuple[float, float, float]],
]:
    rng = shared.random.Random(seed)

    if start is not None:
        sx, sy, sz = start
    else:
        sx = sy = sz = None

    if goal is not None:
        gx, gy, gz = goal
    else:
        gx = gy = gz = None

    static_world_body_base = shared.p.getNumBodies(physicsClientId=cli)
    _build_static_world(
        seed=seed,
        cli=cli,
        start=start,
        goal=goal,
        challenge_type=challenge_type,
    )

    start_platform_surface_z = None
    goal_platform_surface_z = None
    adjusted_start = None
    adjusted_goal = None
    start_platform_uids: shared.List[int] = []
    end_platform_uids: shared.List[int] = []

    collision_scan_types = {
        1: (shared.TYPE_1_WORLD_RANGE, shared.TYPE_1_WORLD_RANGE, shared.TYPE_1_H_MIN, shared.TYPE_1_H_MAX),
        4: (shared.TYPE_3_VILLAGE_RANGE, shared.TYPE_3_VILLAGE_RANGE, 0.0, 0.0),
        5: (shared.TYPE_4_WORLD_RANGE_X, shared.TYPE_4_WORLD_RANGE_Y, shared.TYPE_4_H_MIN, shared.TYPE_4_H_MAX),
        6: (shared.TYPE_6_WORLD_RANGE, shared.TYPE_6_WORLD_RANGE, shared.TYPE_6_H_MIN, shared.TYPE_6_H_MAX),
    }

    _VILLAGE_MIN_OBSTACLE_HEIGHT = 1.0

    if challenge_type in collision_scan_types and sx is not None and sy is not None and sz is not None:
        _wx, _wy, _hmin, _hmax = collision_scan_types[challenge_type]
        placement_rng = shared.random.Random(seed + 777777)

        obstacle_height_filter = _VILLAGE_MIN_OBSTACLE_HEIGHT if challenge_type == 4 else 0.0
        start_surface = sz - shared.START_PLATFORM_TAKEOFF_BUFFER
        new_sx, new_sy, new_s_surface = _find_clear_platform_position(
            cli,
            sx,
            sy,
            start_surface,
            placement_rng,
            static_world_body_base,
            world_range_x=_wx,
            world_range_y=_wy,
            h_min=_hmin,
            h_max=_hmax,
            min_obstacle_height=obstacle_height_filter,
        )
        sx, sy = new_sx, new_sy
        if challenge_type == 4:
            new_s_surface = float(_raycast_surface_z(cli, sx, sy))
            for bid in range(static_world_body_base, shared.p.getNumBodies(physicsClientId=cli)):
                mn, mx = shared.p.getAABB(bid, physicsClientId=cli)
                if mn[0] <= sx <= mx[0] and mn[1] <= sy <= mx[1] and mx[2] > new_s_surface:
                    new_s_surface = mx[2]
        if challenge_type == 4:
            sz = new_s_surface + shared.START_PLATFORM_HEIGHT + 0.15
        else:
            sz = new_s_surface + shared.START_PLATFORM_TAKEOFF_BUFFER
        start_platform_surface_z = new_s_surface
        adjusted_start = (sx, sy, sz)

        if gx is not None and gy is not None and gz is not None:
            goal_bounds = _goal_distance_bounds(challenge_type)
            required_distance_min = None
            required_distance_max = None
            distance_mode = "xyz"
            if goal_bounds is not None:
                required_distance_min, required_distance_max, distance_mode = goal_bounds
            preferred_distance = _distance_between_points(
                (float(start[0]), float(start[1]), float(start[2])),
                (float(goal[0]), float(goal[1]), float(goal[2])),
                mode=distance_mode,
            )

            new_gx, new_gy, new_gz = _find_clear_platform_position(
                cli,
                gx,
                gy,
                gz,
                placement_rng,
                static_world_body_base,
                world_range_x=_wx,
                world_range_y=_wy,
                h_min=_hmin,
                h_max=_hmax,
                avoid_pos=(sx, sy, sz),
                min_distance=required_distance_min or 0.0,
                required_distance_min=required_distance_min,
                required_distance_max=required_distance_max,
                preferred_distance=preferred_distance,
                distance_mode=distance_mode,
                allow_candidate_fallback=challenge_type == 4,
                min_obstacle_height=obstacle_height_filter,
            )
            if challenge_type == 4:
                new_gz = float(_raycast_surface_z(cli, new_gx, new_gy))
                for bid in range(static_world_body_base, shared.p.getNumBodies(physicsClientId=cli)):
                    mn, mx = shared.p.getAABB(bid, physicsClientId=cli)
                    if mn[0] <= new_gx <= mx[0] and mn[1] <= new_gy <= mx[1] and mx[2] > new_gz:
                        new_gz = mx[2]
            gx, gy, gz = new_gx, new_gy, new_gz
            adjusted_goal = (gx, gy, gz)

    if shared.START_PLATFORM and sx is not None and sy is not None and sz is not None:
        platform_radius = shared.START_PLATFORM_RADIUS
        platform_height = shared.START_PLATFORM_HEIGHT

        if challenge_type in (2, 3):
            sx, sy, surface_z = _find_flat_platform_spot(
                cli, sx, sy, platform_radius,
            )
        elif challenge_type in (1, 4, 5, 6) and start_platform_surface_z is not None:
            surface_z = start_platform_surface_z
        elif shared.START_PLATFORM_RANDOMIZE:
            surface_z = shared.get_platform_height_for_seed(seed, challenge_type)
        else:
            surface_z = shared.START_PLATFORM_SURFACE_Z

        start_platform_surface_z = surface_z
        if challenge_type == 4:
            base_position = [sx, sy, surface_z + platform_height / 2 + 0.03]
            start_platform_surface_z = base_position[2] + platform_height / 2
        else:
            base_position = [sx, sy, surface_z - platform_height / 2 + 0.05]

        start_platform_collision = shared.p.createCollisionShape(
            shapeType=shared.p.GEOM_CYLINDER,
            radius=platform_radius,
            height=platform_height,
            physicsClientId=cli,
        )
        start_platform_visual = shared.p.createVisualShape(
            shapeType=shared.p.GEOM_CYLINDER,
            radius=platform_radius,
            length=platform_height,
            rgbaColor=[1.0, 0.0, 0.0, 1.0],
            specularColor=[1.0, 0.3, 0.3],
            physicsClientId=cli,
        )
        start_platform_uid = shared.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=start_platform_collision,
            baseVisualShapeIndex=start_platform_visual,
            basePosition=base_position,
            physicsClientId=cli,
        )
        start_platform_uids.append(start_platform_uid)
        shared.p.changeDynamics(
            bodyUniqueId=start_platform_uid,
            linkIndex=-1,
            restitution=0.0,
            lateralFriction=2.5,
            spinningFriction=1.2,
            rollingFriction=0.6,
            physicsClientId=cli,
        )

        start_plat_top_z = base_position[2] + platform_height / 2
        flat_surface_collision = shared.p.createCollisionShape(
            shapeType=shared.p.GEOM_CYLINDER,
            radius=platform_radius * 0.9,
            height=0.001,
            physicsClientId=cli,
        )
        flat_surface_uid = shared.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=flat_surface_collision,
            baseVisualShapeIndex=-1,
            basePosition=[sx, sy, start_plat_top_z],
            physicsClientId=cli,
        )
        start_platform_uids.append(flat_surface_uid)
        shared.p.changeDynamics(
            bodyUniqueId=flat_surface_uid,
            linkIndex=-1,
            restitution=0.0,
            lateralFriction=3.0,
            spinningFriction=2.0,
            rollingFriction=1.0,
            physicsClientId=cli,
        )
        start_surface_visual = shared.p.createVisualShape(
            shapeType=shared.p.GEOM_CYLINDER,
            radius=platform_radius * 0.9,
            length=0.002,
            rgbaColor=[1.0, 0.0, 0.0, 1.0],
            specularColor=[1.0, 0.3, 0.3],
            physicsClientId=cli,
        )
        start_visual_uid = shared.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=start_surface_visual,
            basePosition=[sx, sy, start_plat_top_z + 0.001],
            physicsClientId=cli,
        )
        start_platform_uids.append(start_visual_uid)

    if goal is not None:
        if adjusted_goal is None:
            gx, gy, gz = goal

        if challenge_type in (2, 3):
            gx, gy, surface_z = _find_flat_platform_spot(
                cli, gx, gy, shared.START_PLATFORM_RADIUS,
            )
        else:
            surface_z = gz

        goal_platform_surface_z = surface_z

        if shared.PLATFORM:
            goal_color = rng.choice(shared.GOAL_COLOR_PALETTE)
            platform_radius = shared.LANDING_PLATFORM_RADIUS
            platform_height = 0.2
            platform_collision = shared.p.createCollisionShape(
                shapeType=shared.p.GEOM_CYLINDER,
                radius=platform_radius,
                height=platform_height,
                physicsClientId=cli,
            )
            platform_visual = shared.p.createVisualShape(
                shapeType=shared.p.GEOM_CYLINDER,
                radius=platform_radius,
                length=platform_height,
                rgbaColor=goal_color,
                specularColor=[
                    goal_color[0] * 0.6 + 0.4,
                    goal_color[1] * 0.6 + 0.4,
                    goal_color[2] * 0.6 + 0.4,
                ],
                physicsClientId=cli,
            )
            if challenge_type == 4:
                goal_platform_z = surface_z + platform_height / 2 + 0.03
            else:
                goal_platform_z = surface_z - platform_height / 2 + 0.05
            platform_uid = shared.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=platform_collision,
                baseVisualShapeIndex=platform_visual,
                basePosition=[gx, gy, goal_platform_z],
                physicsClientId=cli,
            )
            end_platform_uids.append(platform_uid)
            shared.p.changeDynamics(
                bodyUniqueId=platform_uid,
                linkIndex=-1,
                restitution=0.0,
                lateralFriction=2.0,
                spinningFriction=1.0,
                rollingFriction=0.5,
                physicsClientId=cli,
            )

            goal_plat_top_z = goal_platform_z + platform_height / 2
            surface_radius = platform_radius * 0.8
            surface_height = 0.008
            bright_goal_color = [
                min(1.0, goal_color[0] * 1.25),
                min(1.0, goal_color[1] * 1.25),
                min(1.0, goal_color[2] * 1.25),
                1.0,
            ]
            surface_visual = shared.p.createVisualShape(
                shapeType=shared.p.GEOM_CYLINDER,
                radius=surface_radius,
                length=surface_height,
                rgbaColor=bright_goal_color,
                specularColor=[
                    bright_goal_color[0] * 0.8,
                    bright_goal_color[1] * 0.8,
                    bright_goal_color[2] * 0.8,
                ],
                physicsClientId=cli,
            )
            surface_uid = shared.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=surface_visual,
                basePosition=[gx, gy, goal_plat_top_z + surface_height / 2 + 0.001],
                physicsClientId=cli,
            )
            end_platform_uids.append(surface_uid)

            flat_landing_collision = shared.p.createCollisionShape(
                shapeType=shared.p.GEOM_CYLINDER,
                radius=surface_radius,
                height=0.001,
                physicsClientId=cli,
            )
            flat_landing_uid = shared.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=flat_landing_collision,
                baseVisualShapeIndex=-1,
                basePosition=[gx, gy, goal_plat_top_z + surface_height + 0.002],
                physicsClientId=cli,
            )
            end_platform_uids.append(flat_landing_uid)
            shared.p.changeDynamics(
                bodyUniqueId=flat_landing_uid,
                linkIndex=-1,
                restitution=0.0,
                lateralFriction=3.0,
                spinningFriction=2.0,
                rollingFriction=1.0,
                physicsClientId=cli,
            )

            tao_logo_radius = surface_radius * 1.06
            badge_height = 0.005
            tao_background_visual = shared.p.createVisualShape(
                shapeType=shared.p.GEOM_CYLINDER,
                radius=tao_logo_radius,
                length=badge_height,
                rgbaColor=bright_goal_color,
                physicsClientId=cli,
            )
            tao_background_uid = shared.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=tao_background_visual,
                basePosition=[gx, gy, goal_plat_top_z + surface_height + badge_height + 0.008],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=cli,
            )
            end_platform_uids.append(tao_background_uid)
            tao_logo_visual = shared.p.createVisualShape(
                shapeType=shared.p.GEOM_CYLINDER,
                radius=tao_logo_radius * 0.95,
                length=badge_height * 0.5,
                rgbaColor=bright_goal_color,
                physicsClientId=cli,
            )
            tao_logo_uid = shared.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=tao_logo_visual,
                basePosition=[gx, gy, goal_plat_top_z + surface_height + badge_height + 0.011],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=cli,
            )
            end_platform_uids.append(tao_logo_uid)
            shared.p.changeVisualShape(
                tao_logo_uid,
                -1,
                textureUniqueId=_get_tao_tex(cli),
                flags=shared.p.VISUAL_SHAPE_DOUBLE_SIDED,
                physicsClientId=cli,
            )

            pole_h = 0.5
            pole_radius = 0.012
            pole_visual = shared.p.createVisualShape(
                shapeType=shared.p.GEOM_CYLINDER,
                radius=pole_radius,
                length=pole_h,
                rgbaColor=[1.0, 0.2, 0.1, 0.9],
                specularColor=[1.0, 0.8, 0.2],
                physicsClientId=cli,
            )
            cap_visual = shared.p.createVisualShape(
                shapeType=shared.p.GEOM_SPHERE,
                radius=pole_radius * 2,
                rgbaColor=[1.0, 0.3, 0.0, 1.0],
                specularColor=[1.0, 1.0, 0.4],
                physicsClientId=cli,
            )
            pole_uid = shared.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=pole_visual,
                basePosition=[gx, gy, goal_plat_top_z + pole_h / 2 + 0.008],
                physicsClientId=cli,
            )
            end_platform_uids.append(pole_uid)
            cap_uid = shared.p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=cap_visual,
                basePosition=[gx, gy, goal_plat_top_z + pole_h + 0.015],
                physicsClientId=cli,
            )
            end_platform_uids.append(cap_uid)

    if challenge_type == 4:
        all_plat = set(start_platform_uids) | set(end_platform_uids)
        to_remove = set()
        for plat_uid in list(all_plat):
            for i in range(shared.p.getNumBodies(physicsClientId=cli)):
                bid = shared.p.getBodyUniqueId(i, physicsClientId=cli)
                if bid in all_plat or bid < static_world_body_base:
                    continue
                mn, mx = shared.p.getAABB(bid, physicsClientId=cli)
                if (mx[2] - mn[2]) > 2.0:
                    continue
                contacts = shared.p.getClosestPoints(plat_uid, bid, distance=0.0, physicsClientId=cli)
                if contacts and min(c[8] for c in contacts) < -0.03:
                    to_remove.add(bid)
        for bid in to_remove:
            shared.p.removeBody(bid, physicsClientId=cli)

    return (
        end_platform_uids,
        start_platform_uids,
        start_platform_surface_z,
        goal_platform_surface_z,
        adjusted_start,
        adjusted_goal,
    )
