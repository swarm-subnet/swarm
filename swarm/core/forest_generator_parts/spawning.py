"""PyBullet spawning and full forest asset build orchestration."""

from ._shared import *
from .assets import *
from .geometry import *
from .placement import *


# ---------------------------------------------------------------------------
# SECTION 11: PyBullet spawning (with physicsClientId)
# ---------------------------------------------------------------------------
def _collision_shape_for_obj(cli: int, obj_path: str, scale: float) -> int:
    cli_cache = _CLI_COL_CACHE.setdefault(cli, {})
    key = (obj_path, round(scale, 4))
    cached = cli_cache.get(key)
    if cached is not None:
        return cached
    flags = p.GEOM_FORCE_CONCAVE_TRIMESH if hasattr(p, "GEOM_FORCE_CONCAVE_TRIMESH") else 0
    shape = p.createCollisionShape(
        p.GEOM_MESH,
        fileName=obj_path,
        meshScale=[scale, scale, scale],
        flags=flags,
        physicsClientId=cli,
    )
    if shape < 0:
        raise RuntimeError(f"Failed to create collision shape for {obj_path}")
    cli_cache[key] = shape
    return shape


def _tree_trunk_collision_shape(
    cli: int, *, obj_path: str, scale: float, occupancy_radius: float,
) -> tuple[int, float]:
    cli_cache = _CLI_COL_CACHE.setdefault(cli, {})
    min_x, min_y, min_z, max_x, max_y, max_z = _obj_bounds_cached(obj_path)
    total_height = max(1.0, (max_y - min_y) * scale)
    trunk_height = max(2.0, min(total_height * 0.55, total_height))
    trunk_radius = max(0.18, min(0.55, occupancy_radius * 0.18))
    key = (
        "tree_trunk",
        round(trunk_radius, 3),
        round(trunk_height, 3),
    )
    cached = cli_cache.get(key)
    if cached is None:
        cached = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=trunk_radius,
            height=trunk_height,
            physicsClientId=cli,
        )
        cli_cache[key] = cached
    return cached, trunk_height


def _spawn_colored_obj(
    cli: int, *, obj_path: str, scale: float, double_sided_flags: int,
) -> List[int]:
    cli_cache = _CLI_VIS_CACHE.setdefault(cli, {})
    use_file_visuals_only = os.environ.get("SWARM_FOREST_FILE_VISUALS_ONLY", "0") == "1"
    cache_key = (
        obj_path,
        round(scale, 4),
        int(double_sided_flags),
        int(use_file_visuals_only),
    )
    vis_ids = cli_cache.get(cache_key)
    if vis_ids is None:
        material_meshes = _parse_obj_material_meshes(obj_path)
        mtl_colors = _parse_mtl_diffuse_colors(_obj_mtl_path(obj_path))
        default_rgba = [0.7, 0.7, 0.7, 1.0]
        vis_ids = []
        if material_meshes:
            if use_file_visuals_only:
                for mat_name, split_obj_path in _material_visual_obj_paths(obj_path).items():
                    rgba = mtl_colors.get(mat_name, default_rgba)
                    kwargs = {
                        "fileName": split_obj_path,
                        "meshScale": [scale, scale, scale],
                        "rgbaColor": rgba,
                        "specularColor": [0.0, 0.0, 0.0],
                    }
                    if double_sided_flags:
                        kwargs["flags"] = double_sided_flags
                    vis = p.createVisualShape(
                        p.GEOM_MESH, physicsClientId=cli, **kwargs
                    )
                    if vis >= 0:
                        vis_ids.append(vis)
            else:
                for mat_name, (verts, indices, normals) in material_meshes.items():
                    rgba = mtl_colors.get(mat_name, default_rgba)
                    kwargs = {
                        "vertices": verts, "indices": indices, "normals": normals,
                        "meshScale": [scale, scale, scale],
                        "rgbaColor": rgba,
                        "specularColor": [0.0, 0.0, 0.0],
                    }
                    if double_sided_flags:
                        kwargs["flags"] = double_sided_flags
                    vis = p.createVisualShape(
                        p.GEOM_MESH, physicsClientId=cli, **kwargs
                    )
                    if vis >= 0:
                        vis_ids.append(vis)
        if not vis_ids:
            kwargs = {
                "fileName": obj_path,
                "meshScale": [scale, scale, scale],
                "specularColor": [0.0, 0.0, 0.0],
            }
            if double_sided_flags:
                kwargs["flags"] = double_sided_flags
            vis = p.createVisualShape(
                p.GEOM_MESH, physicsClientId=cli, **kwargs
            )
            if vis >= 0:
                vis_ids = [vis]
        cli_cache[cache_key] = vis_ids
    return list(vis_ids)


def _spawn_asset_instance(
    cli: int, *, category: str, obj_name: str,
    x: float, y: float, yaw_deg: float,
    scale: float, flags: int, enable_collision: bool = True,
) -> bool:
    obj_path = os.path.join(FOREST_ASSET_DIR, category, obj_name)
    if not os.path.exists(obj_path):
        return False

    effective_scale = scale
    if FAST_BUILD_MODE and FAST_SCALE_STEP > 0.0:
        effective_scale = max(
            0.01, round(scale / FAST_SCALE_STEP) * FAST_SCALE_STEP
        )

    min_x, min_y, min_z, max_x, _, max_z = _obj_bounds_cached(obj_path)
    cx = (min_x + max_x) * 0.5
    cz = (min_z + max_z) * 0.5
    z0 = max(0.0, -min_y)
    yaw_rad = math.radians(yaw_deg)
    spawn_pos = [
        x - cx * effective_scale,
        y + cz * effective_scale,
        z0 * effective_scale,
    ]
    spawn_quat = p.getQuaternionFromEuler([1.5708, 0.0, yaw_rad])
    vis_shapes = _spawn_colored_obj(
        cli, obj_path=obj_path, scale=effective_scale,
        double_sided_flags=flags,
    )
    if not vis_shapes:
        return False

    col_shape = (
        _collision_shape_for_obj(cli, obj_path, effective_scale)
        if enable_collision
        else -1
    )
    extra_vis = vis_shapes[1:]
    if not extra_vis:
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shapes[0],
            basePosition=spawn_pos,
            baseOrientation=spawn_quat,
            physicsClientId=cli,
        )
        return True

    n_links = len(extra_vis)
    p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shapes[0],
        basePosition=spawn_pos,
        baseOrientation=spawn_quat,
        linkMasses=[0.0] * n_links,
        linkCollisionShapeIndices=[-1] * n_links,
        linkVisualShapeIndices=extra_vis,
        linkPositions=[[0.0, 0.0, 0.0]] * n_links,
        linkOrientations=[[0.0, 0.0, 0.0, 1.0]] * n_links,
        linkInertialFramePositions=[[0.0, 0.0, 0.0]] * n_links,
        linkInertialFrameOrientations=[[0.0, 0.0, 0.0, 1.0]] * n_links,
        linkParentIndices=[0] * n_links,
        linkJointTypes=[p.JOINT_FIXED] * n_links,
        linkJointAxis=[[0.0, 0.0, 1.0]] * n_links,
        physicsClientId=cli,
    )
    return True


def _spawn_instances_as_single_multibody(
    cli: int, *,
    instances: List[Tuple[float, float, str, str, float, float]],
    rng: random.Random, flags: int, class_name: str,
    enable_collision: bool, fixed_yaw_deg: Optional[float] = None,
) -> int:
    if not instances:
        return 0

    link_masses: List[float] = []
    link_col: List[int] = []
    link_vis: List[int] = []
    link_pos: List[List[float]] = []
    link_orn: List[list] = []
    link_ifp: List[List[float]] = []
    link_ifo: List[List[float]] = []
    link_parent: List[int] = []
    link_jtype: List[int] = []
    link_jaxis: List[List[float]] = []
    placed_count = 0

    def _flush() -> None:
        nonlocal link_masses, link_col, link_vis, link_pos, link_orn
        nonlocal link_ifp, link_ifo, link_parent, link_jtype, link_jaxis
        if not link_vis:
            return
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=-1,
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_col,
            linkVisualShapeIndices=link_vis,
            linkPositions=link_pos,
            linkOrientations=link_orn,
            linkInertialFramePositions=link_ifp,
            linkInertialFrameOrientations=link_ifo,
            linkParentIndices=link_parent,
            linkJointTypes=link_jtype,
            linkJointAxis=link_jaxis,
            physicsClientId=cli,
        )
        link_masses = []
        link_col = []
        link_vis = []
        link_pos = []
        link_orn = []
        link_ifp = []
        link_ifo = []
        link_parent = []
        link_jtype = []
        link_jaxis = []

    for x, y, category, obj_name, scale, _ in instances:
        obj_path = os.path.join(FOREST_ASSET_DIR, category, obj_name)
        if not os.path.exists(obj_path):
            continue

        effective_scale = scale
        if FAST_BUILD_MODE and FAST_SCALE_STEP > 0.0:
            effective_scale = max(
                0.01, round(scale / FAST_SCALE_STEP) * FAST_SCALE_STEP
            )

        min_x, min_y, min_z, max_x, _, max_z = _obj_bounds_cached(obj_path)
        cx = (min_x + max_x) * 0.5
        cz = (min_z + max_z) * 0.5
        z0 = max(0.0, -min_y)

        yaw_deg = (
            fixed_yaw_deg if fixed_yaw_deg is not None else rng.uniform(-180.0, 180.0)
        )
        yaw_rad = math.radians(yaw_deg)
        spawn_pos = [
            x - cx * effective_scale,
            y + cz * effective_scale,
            z0 * effective_scale,
        ]
        spawn_quat = list(p.getQuaternionFromEuler([1.5708, 0.0, yaw_rad]))
        vis_shapes = _spawn_colored_obj(
            cli, obj_path=obj_path, scale=effective_scale, double_sided_flags=flags,
        )
        if not vis_shapes:
            continue

        if len(vis_shapes) > TREE_BATCH_MAX_LINKS:
            if _spawn_asset_instance(
                cli, category=category, obj_name=obj_name,
                x=x, y=y, yaw_deg=yaw_deg,
                scale=scale, flags=flags, enable_collision=enable_collision,
            ):
                placed_count += 1
            continue

        if link_vis and (len(link_vis) + len(vis_shapes) > TREE_BATCH_MAX_LINKS):
            _flush()

        col_shape = (
            _collision_shape_for_obj(cli, obj_path, effective_scale)
            if enable_collision
            else -1
        )
        for idx, vs in enumerate(vis_shapes):
            link_masses.append(0.0)
            link_col.append(col_shape if idx == 0 else -1)
            link_vis.append(vs)
            link_pos.append(spawn_pos)
            link_orn.append(spawn_quat)
            link_ifp.append([0.0, 0.0, 0.0])
            link_ifo.append([0.0, 0.0, 0.0, 1.0])
            link_parent.append(0)
            link_jtype.append(p.JOINT_FIXED)
            link_jaxis.append([0.0, 0.0, 1.0])

        placed_count += 1

    _flush()
    return placed_count


def _spawn_tree_trunk_colliders(
    cli: int, *, instances: List[Tuple[float, float, str, str, float, float]],
) -> int:
    invis_vis = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.001, 0.001, 0.001],
        rgbaColor=[0, 0, 0, 0],
        physicsClientId=cli,
    )
    placed_count = 0
    for x, y, category, obj_name, scale, occupancy_radius in instances:
        obj_path = os.path.join(FOREST_ASSET_DIR, category, obj_name)
        if not os.path.exists(obj_path):
            continue
        effective_scale = scale
        if FAST_BUILD_MODE and FAST_SCALE_STEP > 0.0:
            effective_scale = max(
                0.01, round(scale / FAST_SCALE_STEP) * FAST_SCALE_STEP
            )
        col_shape, trunk_height = _tree_trunk_collision_shape(
            cli,
            obj_path=obj_path,
            scale=effective_scale,
            occupancy_radius=occupancy_radius,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=invis_vis,
            basePosition=[x, y, trunk_height * 0.5],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=cli,
        )
        placed_count += 1
    return placed_count


def _spawn_tree_visuals_individually(
    cli: int, *, instances: List[Tuple[float, float, str, str, float, float]], yaw_deg: float, flags: int,
) -> int:
    placed_count = 0
    for x, y, category, obj_name, scale, _ in instances:
        if _spawn_asset_instance(
            cli,
            category=category,
            obj_name=obj_name,
            x=x,
            y=y,
            yaw_deg=yaw_deg,
            scale=scale,
            flags=flags,
            enable_collision=False,
        ):
            placed_count += 1
    return placed_count


# ---------------------------------------------------------------------------
# SECTION 12: Main forest builder
# ---------------------------------------------------------------------------
def _spawn_forest_assets(
    cli: int, seed: int, mode_id: int = SCORING_MODE_ID,
    difficulty_id: int = SCORING_DIFFICULTY_ID,
    safe_zones: Optional[List[Tuple[float, float, float]]] = None,
    safe_zone_radius: float = 0.0,
) -> None:
    diff_cfg = DIFFICULTY_CONFIG[difficulty_id]
    flags = (
        p.VISUAL_SHAPE_DOUBLE_SIDED
        if hasattr(p, "VISUAL_SHAPE_DOUBLE_SIDED")
        else 0
    )
    rng = random.Random(seed)
    assets = _resolve_assets_for_class(mode_id)
    safe_zone_circles = _normalize_safe_zone_circles(safe_zones, safe_zone_radius)
    safe_zone_rects = _safe_zone_rects(safe_zone_circles)

    def _scaled_count(cls: str, base: int) -> int:
        mul = CLASS_DENSITY_MULTIPLIER.get(cls, 1.0)
        mul *= DIFFICULTY_DENSITY_MULTIPLIER.get(difficulty_id, 1.0)
        if cls == "trees":
            mul *= TREE_DIFFICULTY_MULTIPLIER.get(difficulty_id, 1.0)
        return max(0, int(round(base * DENSITY_MULTIPLIER * mul)))

    trees_count = _scaled_count("trees", diff_cfg["tree_count"])
    logs_count = _scaled_count("logs", diff_cfg["log_count"])
    bushes_count = _scaled_count("bushes", diff_cfg["bush_count"])

    rocks_assets = assets.get("rocks", [])
    stump_assets = assets.get("stumps", [])
    plants_assets = assets.get("plants", [])
    cactus_assets = assets.get("cactus", [])

    rock_stump_total = max(
        0,
        int(
            round(
                _scaled_count("rocks", diff_cfg["rock_stump_count"])
                * ROCK_STUMP_TOTAL_MULTIPLIER
            )
        ),
    )
    rocks_count, stumps_count = _split_asset_count(
        rock_stump_total, rocks_assets, stump_assets,
        primary_ratio=ROCK_STUMP_PRIMARY_RATIO,
    )
    plants_count, cactus_count = _split_asset_count(
        _scaled_count("plants", diff_cfg["ground_cover_count"]),
        plants_assets, cactus_assets,
        primary_ratio=GROUND_COVER_PLANT_PRIMARY_RATIO,
    )

    trunk_bounds = TRUNK_COUNT_BOUNDS_BY_DIFFICULTY.get(difficulty_id, {})
    logs_max = trunk_bounds.get("logs_max")
    if logs_max is not None:
        logs_count = min(logs_count, logs_max)
    stumps_min = trunk_bounds.get("stumps_min")
    if stumps_min is not None and stumps_count < stumps_min:
        shift = min(stumps_min - stumps_count, rocks_count)
        stumps_count += shift
        rocks_count -= shift
    stumps_max = trunk_bounds.get("stumps_max")
    if stumps_max is not None and stumps_count > stumps_max:
        shift = stumps_count - stumps_max
        stumps_count -= shift
        rocks_count += shift

    tree_instances = _pick_tree_instances(
        rng, count=trees_count, assets=list(assets.get("trees", [])),
        clearance_m=diff_cfg["tree_clearance_m"], difficulty_id=difficulty_id,
        safe_zone_circles=safe_zone_circles,
        safe_zone_rects=safe_zone_rects,
    )

    log_tree_occ = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["logs"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["logs"],
    )
    tree_base_rects = _tree_base_rects_from_instances(tree_instances)
    tree_span_rects = _tree_span_rects_from_instances(tree_instances)
    prot_span_rects = _protected_tree_span_rects_from_instances(tree_instances)

    log_instances = _pick_log_instances(
        rng, count=logs_count, assets=assets.get("logs", []),
        clearance_m=diff_cfg["log_clearance_m"] * LOG_CLEARANCE_MULTIPLIER,
        occupied_instances=log_tree_occ,
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=prot_span_rects,
    )
    bush_occ_scale = SMALL_ASSET_TREE_OCCUPANCY_SCALE["bushes"]
    bush_occ_cap = SMALL_ASSET_TREE_OCCUPANCY_CAP_M["bushes"]
    bush_instances = _pick_shrub_instances(
        rng, count=bushes_count, assets=assets.get("bushes", []),
        clearance_m=diff_cfg["bush_clearance_m"],
        tree_instances=tree_instances,
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=prot_span_rects,
        occupied_instances=log_instances,
        tree_occupancy_scale=bush_occ_scale,
        tree_occupancy_cap_m=bush_occ_cap,
        safe_zone_circles=safe_zone_circles,
        safe_zone_rects=safe_zone_rects,
    )
    rock_tree_occ = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["rocks"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["rocks"],
    )
    rock_instances = _pick_rock_stump_instances(
        rng, count=rocks_count, assets=rocks_assets,
        mode_id=mode_id, clearance_m=diff_cfg["rock_stump_clearance_m"],
        occupied_instances=rock_tree_occ + bush_instances + log_instances,
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=prot_span_rects,
        safe_zone_circles=safe_zone_circles,
        safe_zone_rects=safe_zone_rects,
    )
    stump_tree_occ = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["stumps"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["stumps"],
    )
    stump_instances = _pick_rock_stump_instances(
        rng, count=stumps_count, assets=stump_assets,
        mode_id=mode_id,
        clearance_m=diff_cfg["rock_stump_clearance_m"] * STUMP_CLEARANCE_MULTIPLIER,
        occupied_instances=(
            stump_tree_occ + bush_instances + log_instances + rock_instances
        ),
        tree_base_rects=tree_base_rects,
        protected_tree_span_rects=prot_span_rects,
        safe_zone_circles=safe_zone_circles,
        safe_zone_rects=safe_zone_rects,
    )
    plant_tree_occ = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["plants"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["plants"],
    )
    plant_instances = _pick_ground_cover_instances(
        rng, count=plants_count, assets=plants_assets,
        mode_id=mode_id, clearance_m=diff_cfg["ground_cover_clearance_m"],
        occupied_instances=(
            plant_tree_occ + bush_instances + log_instances + rock_instances
            + stump_instances
        ),
        tree_base_rects=tree_base_rects,
        tree_span_rects=tree_span_rects,
        protected_tree_span_rects=prot_span_rects,
        safe_zone_circles=safe_zone_circles,
        safe_zone_rects=safe_zone_rects,
    )
    cactus_tree_occ = _scaled_occupied_instances(
        tree_instances,
        radius_scale=SMALL_ASSET_TREE_OCCUPANCY_SCALE["cactus"],
        min_radius=SMALL_ASSET_TREE_OCCUPANCY_MIN_M,
        max_radius=SMALL_ASSET_TREE_OCCUPANCY_CAP_M["cactus"],
    )
    cactus_instances = _pick_ground_cover_instances(
        rng, count=cactus_count, assets=cactus_assets,
        mode_id=mode_id, clearance_m=diff_cfg["ground_cover_clearance_m"],
        occupied_instances=(
            cactus_tree_occ + bush_instances + log_instances + rock_instances
            + stump_instances + plant_instances
        ),
        tree_base_rects=tree_base_rects,
        tree_span_rects=tree_span_rects,
        protected_tree_span_rects=prot_span_rects,
        safe_zone_circles=safe_zone_circles,
        safe_zone_rects=safe_zone_rects,
    )

    tree_yaw = 0.0
    _spawn_tree_visuals_individually(
        cli, instances=tree_instances, yaw_deg=tree_yaw, flags=flags,
    )
    _spawn_tree_trunk_colliders(cli, instances=tree_instances)
    for cls, inst_list in [
        ("bushes", bush_instances),
        ("rocks", rock_instances),
        ("stumps", stump_instances),
        ("logs", log_instances),
        ("plants", plant_instances),
        ("cactus", cactus_instances),
    ]:
        _spawn_instances_as_single_multibody(
            cli, instances=inst_list, rng=rng, flags=flags,
            class_name=cls, enable_collision=True,
        )



__all__ = [name for name in globals() if not name.startswith("__")]
