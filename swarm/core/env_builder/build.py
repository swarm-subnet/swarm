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
    if challenge_type == 4:
        return (shared.TYPE_3_R_MIN, shared.TYPE_3_R_MAX, "xy")
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


def _human_assets_root() -> shared.Path:
    return (
        shared.Path(__file__).resolve().parent.parent.parent
        / "assets"
        / "maps"
        / "custom"
        / "humans"
    )


def _load_human_target_spec(rng: shared.random.Random) -> dict:
    root = _human_assets_root()
    manifest_path = root / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = shared.json.load(f)
    models = manifest["models"]
    model_keys = sorted(models)
    model_key = rng.choice(model_keys)
    spec = dict(models[model_key])
    spec["name"] = model_key
    spec["root"] = root
    spec["obj_path"] = root / spec["path"]
    spec["mtl_path"] = root / spec.get("material", spec["path"].replace(".obj", ".mtl"))
    return spec


def _read_mtl_materials(mtl_path: shared.Path) -> dict[str, dict]:
    def _linear_to_srgb(value: float) -> float:
        return max(0.0, min(1.0, float(value))) ** (1.0 / 2.2)

    def _finish_material(
        materials: dict[str, dict],
        name: str,
        diffuse: list[float],
        alpha: float,
        texture_path: shared.Optional[shared.Path],
    ) -> None:
        materials[name] = {
            "rgba": [
                _linear_to_srgb(diffuse[0]),
                _linear_to_srgb(diffuse[1]),
                _linear_to_srgb(diffuse[2]),
                max(0.05, alpha),
            ],
            "texture_path": texture_path,
        }

    materials: dict[str, dict] = {}
    current = None
    alpha = 1.0
    diffuse = [1.0, 1.0, 1.0]
    texture_path: shared.Optional[shared.Path] = None
    for raw in mtl_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if parts[0] == "newmtl":
            if current is not None:
                _finish_material(materials, current, diffuse, alpha, texture_path)
            current = parts[1]
            alpha = 1.0
            diffuse = [1.0, 1.0, 1.0]
            texture_path = None
        elif parts[0] == "Kd" and len(parts) >= 4:
            diffuse = [float(parts[1]), float(parts[2]), float(parts[3])]
        elif parts[0] == "d" and len(parts) >= 2:
            alpha = float(parts[1])
        elif parts[0] == "map_Kd" and len(parts) >= 2:
            candidate = shared.Path(parts[-1])
            if not candidate.exists():
                candidate = mtl_path.parent / candidate.name
            texture_path = candidate if candidate.exists() else None
    if current is not None:
        _finish_material(materials, current, diffuse, alpha, texture_path)
    return materials


def _human_material_parts(
    obj_path: shared.Path,
    mtl_path: shared.Path,
) -> list[tuple[shared.Path, list[float], shared.Optional[shared.Path]]]:
    materials = _read_mtl_materials(mtl_path)
    cache_dir = shared.STATE_DIR / "humans" / obj_path.parent.name / "material_parts"
    cache_dir.mkdir(parents=True, exist_ok=True)

    geometry: list[str] = []
    faces_by_material: dict[str, list[str]] = {}
    current_material = "default"
    for raw in obj_path.read_text(encoding="utf-8").splitlines():
        if raw.startswith(("v ", "vt ", "vn ", "vp ")):
            geometry.append(raw)
        elif raw.startswith("usemtl "):
            current_material = raw.split(maxsplit=1)[1].strip()
            faces_by_material.setdefault(current_material, [])
        elif raw.startswith("f "):
            faces_by_material.setdefault(current_material, []).append(raw)

    parts: list[tuple[shared.Path, list[float], shared.Optional[shared.Path]]] = []
    for material, faces in faces_by_material.items():
        if not faces:
            continue
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in material)
        part_path = cache_dir / f"{safe_name}.obj"
        with part_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("# Auto-generated material split for PyBullet visual colors\n")
            f.write(f"o {obj_path.parent.name}_{safe_name}\n")
            for line in geometry:
                f.write(line + "\n")
            f.write(f"usemtl {material}\n")
            f.write("s 1\n")
            for face in faces:
                f.write(face + "\n")
        material_info = materials.get(material, {})
        parts.append((
            part_path,
            material_info.get("rgba", [1.0, 1.0, 1.0, 1.0]),
            material_info.get("texture_path"),
        ))
    return parts


def _spawn_human_goal_target(
    *,
    cli: int,
    x: float,
    y: float,
    surface_z: float,
    rng: shared.random.Random,
) -> tuple[shared.List[int], float]:
    spec = _load_human_target_spec(rng)
    height = float(spec.get("height_m", 1.8))
    scale = float(spec.get("scale", 1.0))
    center_z = float(surface_z) + height * 0.5

    uids: shared.List[int] = []
    yaw = rng.uniform(-shared.math.pi, shared.math.pi)
    body_orientation = shared.p.getQuaternionFromEuler([0.0, 0.0, yaw])
    mesh_orientation = shared.p.getQuaternionFromEuler([shared.math.pi * 0.5, 0.0, 0.0])
    mesh_base_z = float(surface_z) + 0.006
    for part_path, rgba, texture_path in _human_material_parts(spec["obj_path"], spec["mtl_path"]):
        visual_id = shared.p.createVisualShape(
            shapeType=shared.p.GEOM_MESH,
            fileName=str(part_path),
            meshScale=[scale, scale, scale],
            rgbaColor=rgba,
            visualFrameOrientation=mesh_orientation,
            physicsClientId=cli,
        )
        collision_id = shared.p.createCollisionShape(
            shapeType=shared.p.GEOM_MESH,
            fileName=str(part_path),
            meshScale=[scale, scale, scale],
            collisionFrameOrientation=mesh_orientation,
            physicsClientId=cli,
        )
        uid = shared.p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=[float(x), float(y), mesh_base_z],
            baseOrientation=body_orientation,
            physicsClientId=cli,
        )
        shared.p.changeDynamics(
            bodyUniqueId=uid,
            linkIndex=-1,
            restitution=0.0,
            lateralFriction=2.0,
            spinningFriction=1.0,
            rollingFriction=0.5,
            physicsClientId=cli,
        )
        visual_kwargs = {
            "rgbaColor": rgba,
            "flags": shared.p.VISUAL_SHAPE_DOUBLE_SIDED,
            "physicsClientId": cli,
        }
        if texture_path is not None:
            visual_kwargs["textureUniqueId"] = shared.p.loadTexture(
                str(texture_path),
                physicsClientId=cli,
            )
        shared.p.changeVisualShape(uid, -1, **visual_kwargs)
        uids.append(uid)
    return uids, center_z


def _person_ground_surface_z(
    cli: int,
    x: float,
    y: float,
    fallback_surface_z: float,
    challenge_type: int,
) -> float:
    if challenge_type in (5, 6):
        return 0.0
    if challenge_type in (1, 2, 4, 6):
        ray_z = float(_raycast_surface_z(cli, x, y))
        if shared.math.isfinite(ray_z):
            return ray_z
    return float(fallback_surface_z)


def build_world(
    seed: int,
    cli: int,
    *,
    start: shared.Optional[shared.Tuple[float, float, float]] = None,
    goal: shared.Optional[shared.Tuple[float, float, float]] = None,
    challenge_type: int = 1,
    moving_platform: bool = False,
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
            adjusted_start = (sx, sy, surface_z + shared.START_PLATFORM_TAKEOFF_BUFFER)
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
            orbit_r = (
                shared.PLATFORM_RADIUS_MAX
                if (challenge_type in (2, 3) and moving_platform)
                else 0.0
            )
            gx, gy, surface_z = _find_flat_platform_spot(
                cli, gx, gy, shared.START_PLATFORM_RADIUS,
                orbit_radius=orbit_r,
            )
            adjusted_goal = (gx, gy, surface_z)
        else:
            surface_z = gz
            if challenge_type == 4 and moving_platform:
                outer = (
                    shared.PLATFORM_RADIUS_MAX
                    + 0.3
                    + shared.LANDING_PLATFORM_RADIUS
                )
                max_top = surface_z
                exclude_uids = set(start_platform_uids)
                n_bodies = shared.p.getNumBodies(physicsClientId=cli)
                for body_idx in range(static_world_body_base, n_bodies):
                    uid = shared.p.getBodyUniqueId(body_idx, physicsClientId=cli)
                    if uid in exclude_uids:
                        continue
                    mn, mx = shared.p.getAABB(uid, physicsClientId=cli)
                    if (mx[0] - mn[0]) > 50.0 or (mx[1] - mn[1]) > 50.0:
                        continue
                    cx = max(mn[0], min(gx, mx[0]))
                    cy = max(mn[1], min(gy, mx[1]))
                    if (gx - cx) ** 2 + (gy - cy) ** 2 <= outer * outer:
                        if mx[2] > max_top:
                            max_top = mx[2]
                surface_z = max(surface_z, max_top + 0.20)
                adjusted_goal = (gx, gy, surface_z)

        goal_platform_surface_z = surface_z

        if shared.PLATFORM:
            surface_z = _person_ground_surface_z(
                cli,
                gx,
                gy,
                surface_z,
                challenge_type,
            )
            human_uids, human_goal_z = _spawn_human_goal_target(
                cli=cli,
                x=gx,
                y=gy,
                surface_z=surface_z,
                rng=rng,
            )
            end_platform_uids.extend(human_uids)
            goal_platform_surface_z = human_goal_z
            adjusted_goal = (gx, gy, human_goal_z)

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
