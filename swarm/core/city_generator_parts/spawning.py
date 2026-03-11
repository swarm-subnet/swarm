from ._shared import *
from .generation import Block, Building, Rect, RoadTile, SeededRNG, generate_city


def ceil_half(x):
    return math.ceil(x * 2) / 2


RAW_DATA = [
    ("house", "kenney_suburban/building-type-a.obj", 6.50, 5.14),
    ("house", "kenney_suburban/building-type-b.obj", 9.14, 5.70),
    ("house", "kenney_suburban/building-type-c.obj", 6.44, 5.14),
    ("house", "kenney_suburban/building-type-d.obj", 8.79, 5.14),
    ("house", "kenney_suburban/building-type-e.obj", 6.50, 5.14),
    ("house", "kenney_suburban/building-type-f.obj", 7.14, 7.03),
    ("house", "kenney_suburban/building-type-g.obj", 7.25, 5.89),
    ("house", "kenney_suburban/building-type-h.obj", 6.50, 4.58),
    ("house", "kenney_suburban/building-type-i.obj", 6.44, 5.14),
    ("house", "kenney_suburban/building-type-j.obj", 6.85, 4.58),
    ("house", "kenney_suburban/building-type-k.obj", 4.61, 5.10),
    ("house", "kenney_suburban/building-type-l.obj", 5.20, 5.12),
    ("house", "kenney_suburban/building-type-m.obj", 7.14, 7.14),
    ("house", "kenney_suburban/building-type-n.obj", 8.93, 6.89),
    ("house", "kenney_suburban/building-type-o.obj", 6.35, 5.14),
    ("house", "kenney_suburban/building-type-p.obj", 6.20, 4.95),
    ("house", "kenney_suburban/building-type-q.obj", 6.20, 4.43),
    ("house", "kenney_suburban/building-type-r.obj", 5.16, 5.12),
    ("house", "kenney_suburban/building-type-s.obj", 7.03, 5.44),
    ("house", "kenney_suburban/building-type-t.obj", 6.60, 7.05),
    ("house", "kenney_suburban/building-type-u.obj", 7.14, 5.44),
    ("apt", "kenney_commercial/building-a.obj", 4.42, 4.70),
    ("apt", "kenney_commercial/building-b.obj", 4.85, 4.70),
    ("apt", "kenney_commercial/building-c.obj", 4.42, 5.45),
    ("apt", "kenney_commercial/building-d.obj", 4.20, 4.50),
    ("apt", "kenney_commercial/building-e.obj", 8.20, 5.04),
    ("apt", "kenney_commercial/building-f.obj", 4.20, 5.15),
    ("apt", "kenney_commercial/building-g.obj", 4.85, 4.61),
    ("apt", "kenney_commercial/building-h.obj", 4.42, 5.04),
    ("apt", "kenney_commercial/building-i.obj", 6.20, 6.51),
    ("apt", "kenney_commercial/building-j.obj", 10.42, 6.70),
    ("apt", "kenney_commercial/building-k.obj", 10.42, 4.71),
    ("apt", "kenney_commercial/building-l.obj", 6.85, 7.01),
    ("apt", "kenney_commercial/building-m.obj", 6.20, 6.21),
    ("apt", "kenney_commercial/building-n.obj", 11.60, 9.10),
    ("tower", "kenney_commercial/building-skyscraper-a.obj", 6.80, 6.80),
    ("tower", "kenney_commercial/building-skyscraper-b.obj", 6.80, 6.80),
    ("tower", "kenney_commercial/building-skyscraper-c.obj", 6.40, 6.94),
    ("tower", "kenney_commercial/building-skyscraper-d.obj", 6.40, 6.94),
    ("tower", "kenney_commercial/building-skyscraper-e.obj", 6.48, 6.21),
]

ASSET_SPECS: Dict[str, list] = {"house": [], "apt": [], "tower": []}


def _init_specs():
    for cat, path, raw_w, raw_d in RAW_DATA:
        margin = 1.10
        final_w = ceil_half(raw_w * margin)
        final_d = ceil_half(raw_d * margin)
        ASSET_SPECS[cat].append({
            "w": final_w,
            "d": final_d,
            "path": _resolve_asset_path(path),
            "area": final_w * final_d,
            "raw_w": raw_w,
            "raw_d": raw_d,
        })


def configure_templates():
    new_templates = {"house": [], "apt": [], "tower": []}
    for cat in ASSET_SPECS:
        seen_dims = set()
        for spec in ASSET_SPECS[cat]:
            dims = (spec["w"], spec["d"])
            if dims not in seen_dims:
                new_templates[cat].append({"w": spec["w"], "d": spec["d"]})
                seen_dims.add(dims)
    global TEMPLATES
    TEMPLATES = new_templates


# ---------------------------------------------------------------------------
# SECTION 4: Zone-based building selection
# ---------------------------------------------------------------------------
def get_building_zone(x, y, map_size=200):
    center = map_size / 2
    distance = math.sqrt((x - center) ** 2 + (y - center) ** 2)
    max_dist = math.sqrt(2) * (map_size / 2)
    ratio = distance / max_dist
    if ratio > 0.6:
        return "outer"
    elif ratio > 0.3:
        return "middle"
    return "center"


def get_zone_building_type(zone, city_type, rng):
    if city_type == 1:
        return "house"
    r = rng.next_float()
    if zone == "outer":
        if city_type == 2:
            if r < 0.80:
                return "house"
            elif r < 0.95:
                return "apt"
            return "tower"
        else:
            if r < 0.50:
                return "house"
            elif r < 0.85:
                return "apt"
            return "tower"
    elif zone == "middle":
        if city_type == 2:
            if r < 0.30:
                return "house"
            elif r < 0.80:
                return "apt"
            return "tower"
        else:
            if r < 0.15:
                return "house"
            elif r < 0.55:
                return "apt"
            return "tower"
    else:
        if city_type == 2:
            if r < 0.05:
                return "house"
            elif r < 0.40:
                return "apt"
            return "tower"
        else:
            if r < 0.02:
                return "house"
            elif r < 0.25:
                return "apt"
            return "tower"


def get_asset_for_zone(category, lot_w, lot_d, city_type, block_style, rng):
    if category == "tower":
        skyscrapers = [
            s for s in ASSET_SPECS.get("tower", [])
            if "skyscraper" in s["path"].lower()
        ]
        if not skyscrapers:
            return None
        for spec in skyscrapers:
            if abs(spec["w"] - lot_w) < 0.5 and abs(spec["d"] - lot_d) < 0.5:
                return spec["path"]
            if abs(spec["d"] - lot_w) < 0.5 and abs(spec["w"] - lot_d) < 0.5:
                return spec["path"]
        target_area = lot_w * lot_d
        closest = min(skyscrapers, key=lambda x: abs(x["area"] - target_area))
        return closest["path"]

    effective_category = category
    folder_filter = None
    if city_type == 1:
        effective_category = "house"
        folder_filter = "kenney_suburban"
    elif city_type == 3:
        if category == "house":
            effective_category = "apt"
        folder_filter = "kenney_commercial"
    else:
        if block_style == "suburban":
            effective_category = "house"
            folder_filter = "kenney_suburban"
        elif block_style == "commercial":
            if category == "house":
                effective_category = "apt"
            folder_filter = "kenney_commercial"

    candidates = ASSET_SPECS.get(effective_category, [])
    if not candidates:
        return None
    if folder_filter:
        filtered = [s for s in candidates if folder_filter in s["path"]]
        if filtered:
            candidates = filtered

    matches = []
    for spec in candidates:
        if abs(spec["w"] - lot_w) < 0.1 and abs(spec["d"] - lot_d) < 0.1:
            matches.append(spec)
        elif abs(spec["d"] - lot_w) < 0.1 and abs(spec["w"] - lot_d) < 0.1:
            matches.append(spec)
    if matches:
        return rng.choice(matches)["path"]
    target_area = lot_w * lot_d
    closest = min(candidates, key=lambda x: abs(x["area"] - target_area))
    return closest["path"]


# ---------------------------------------------------------------------------
# SECTION 5: Shape cache & OBJ spawning
# ---------------------------------------------------------------------------
_shape_cache: Dict[tuple, Tuple[int, int]] = {}


def _resolve_asset_path(rel_path):
    rel_norm = rel_path.replace("\\", "/")
    if rel_norm.startswith("obj_converted/"):
        return os.path.join(OTHER_SOURCES_DIR, rel_path)
    return os.path.join(KENNEY_DIR, rel_path)


def _get_asset_path(name, rng):
    variants = ASSET_MAP.get(name, [])
    if not variants and name in ["straight_v", "straight_h"]:
        variants = ASSET_MAP["straight"]
    if variants:
        chosen = rng.choice(variants)
        if chosen is None:
            return None
        return _resolve_asset_path(chosen)
    return None


def spawn_asset_exact(cli, path, x, y, z, rotation_deg, scale_vec, rgba=None):
    if not os.path.exists(path):
        return None
    cache_key = (cli, path, tuple(scale_vec), tuple(rgba) if rgba else None)
    if cache_key in _shape_cache:
        vis_id, col_id = _shape_cache[cache_key]
    else:
        vis_id = p.createVisualShape(
            p.GEOM_MESH, fileName=path, meshScale=scale_vec,
            rgbaColor=rgba, physicsClientId=cli,
        )
        col_id = p.createCollisionShape(
            p.GEOM_MESH, fileName=path, meshScale=scale_vec,
            physicsClientId=cli,
        )
        _shape_cache[cache_key] = (vis_id, col_id)
    base_x_rot = 1.5708
    rad_z = math.radians(rotation_deg)
    orn = p.getQuaternionFromEuler([base_x_rot, 0, rad_z])
    body_id = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id, basePosition=[x, y, z],
        baseOrientation=orn, physicsClientId=cli,
    )
    return body_id


def spawn_asset_with_random_color(cli, path, x, y, z, rotation_deg, scale_vec, rng):
    if "kenney_commercial" not in path:
        return spawn_asset_exact(cli, path, x, y, z, rotation_deg, scale_vec)
    color_choice = rng.next_float()
    if color_choice < 0.5:
        return spawn_asset_exact(cli, path, x, y, z, rotation_deg, scale_vec)
    elif color_choice < 0.75:
        mtl_suffix = "-green"
    else:
        mtl_suffix = "-orange"
    try:
        with open(path, "r") as f:
            obj_content = f.read()

        def replace_mtl(match):
            original = match.group(1)
            base_name = original.replace(".mtl", "")
            return f"mtllib {base_name}{mtl_suffix}.mtl"

        modified_content = re.sub(r"mtllib\s+(\S+\.mtl)", replace_mtl, obj_content)
        temp_dir = os.path.dirname(path)
        temp_path = os.path.join(temp_dir, f"_temp_colored{mtl_suffix}.obj")
        with open(temp_path, "w") as f:
            f.write(modified_content)
        result = spawn_asset_exact(cli, temp_path, x, y, z, rotation_deg, scale_vec)
        try:
            os.remove(temp_path)
        except OSError:
            pass
        return result
    except Exception:
        return spawn_asset_exact(cli, path, x, y, z, rotation_deg, scale_vec)


# ---------------------------------------------------------------------------
# SECTION 6: Environment element spawners
# ---------------------------------------------------------------------------
def _in_safe_zone(x, y, safe_zones, safe_zone_radius):
    if not safe_zones:
        return False
    for sx, sy in safe_zones:
        if math.hypot(x - sx, y - sy) < safe_zone_radius:
            return True
    return False


def _rect_intersects_safe_zone(cx, cy, half_w, half_h, safe_zones, safe_zone_radius):
    if not safe_zones:
        return False
    x1 = cx - half_w
    x2 = cx + half_w
    y1 = cy - half_h
    y2 = cy + half_h
    for sx, sy in safe_zones:
        nx = max(x1, min(sx, x2))
        ny = max(y1, min(sy, y2))
        dx = sx - nx
        dy = sy - ny
        if dx * dx + dy * dy < safe_zone_radius * safe_zone_radius:
            return True
    return False


def _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
    if body_id is None or not safe_zones:
        return False
    mn, mx = p.getAABB(body_id, physicsClientId=cli)
    x1, y1 = mn[0], mn[1]
    x2, y2 = mx[0], mx[1]
    for sx, sy in safe_zones:
        nx = max(x1, min(sx, x2))
        ny = max(y1, min(sy, y2))
        dx = sx - nx
        dy = sy - ny
        if dx * dx + dy * dy < safe_zone_radius * safe_zone_radius:
            return True
    return False


def _spawn_grass(cli, blocks, offset):
    for block in blocks:
        bx = block.rect.x - offset
        by = block.rect.y - offset
        half_w = block.rect.w / 2
        half_h = block.rect.h / 2
        vis_id = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half_w, half_h, 0.05],
            rgbaColor=COLORS["grass"], physicsClientId=cli,
        )
        p.createMultiBody(
            baseMass=0, baseVisualShapeIndex=vis_id,
            basePosition=[bx + half_w, by + half_h, 0.01],
            physicsClientId=cli,
        )


def _spawn_roads(cli, tiles, tile_size, rng, offset,
                 safe_zones=None, safe_zone_radius=0.0):
    half_tile = tile_size / 2
    scale_mod = 2.0
    visual_scale = 1.0
    for tile in tiles:
        if tile.type == "roundabout_arm":
            continue
        path = _get_asset_path(tile.type, rng)
        if not path:
            continue
        cx = tile.x + half_tile - offset
        cy = tile.y + half_tile - offset
        road_scale_vec = [
            scale_mod * SCALE_FACTOR * visual_scale,
            SCALE_FACTOR,
            scale_mod * SCALE_FACTOR * visual_scale,
        ]
        spawn_asset_exact(
            cli, path, cx, cy, 0, tile.rotation + 90, road_scale_vec,
        )


def _spawn_buildings(cli, buildings, blocks, city_type, tile_size,
                     rng, safe_zones, safe_zone_radius, difficulty, offset):
    block_styles = {}
    block_types = {}
    for block in blocks:
        block_id = (block.rect.x, block.rect.y)
        block_cx = block.rect.x + block.rect.w / 2
        block_cy = block.rect.y + block.rect.h / 2
        zone = get_building_zone(block_cx, block_cy)
        if difficulty == 3:
            block_types[block_id] = "tower"
            block_styles[block_id] = "commercial"
        elif city_type == 1:
            block_types[block_id] = "house"
            block_styles[block_id] = "suburban"
        elif city_type in [2, 3]:
            building_type = get_zone_building_type(zone, city_type, rng)
            block_types[block_id] = building_type
            block_styles[block_id] = "suburban" if building_type == "house" else "commercial"

    platform_r = LANDING_PLATFORM_RADIUS
    for b in buildings:
        rect = b.rect
        if difficulty == 3 and b.type != "tower":
            continue
        cx = rect.x + rect.w / 2 - offset
        cy = rect.y + rect.h / 2 - offset
        bld_radius = max(rect.w, rect.h) / 2
        platform_clearance = platform_r + bld_radius + 0.3
        too_close = any(
            math.hypot(cx - sz[0], cy - sz[1]) < platform_clearance
            for sz in safe_zones
        )
        if too_close:
            continue
        half_w = rect.w / 2
        half_h = rect.h / 2
        if _rect_intersects_safe_zone(cx, cy, half_w, half_h, safe_zones, safe_zone_radius):
            continue
        block_style = "suburban"
        effective_type = b.type
        cx_temp = rect.x + rect.w / 2
        cy_temp = rect.y + rect.h / 2
        for block in blocks:
            bx, by, bw, bh = block.rect.x, block.rect.y, block.rect.w, block.rect.h
            if bx <= cx_temp <= bx + bw and by <= cy_temp <= by + bh:
                block_id = (block.rect.x, block.rect.y)
                block_style = block_styles.get(block_id, "suburban")
                if block_id in block_types:
                    effective_type = block_types[block_id]
                break
        path = get_asset_for_zone(effective_type, rect.w, rect.h, city_type, block_style, rng)
        if not path:
            if difficulty == 3:
                continue
            path = _get_asset_path(b.type, rng)
        if not path:
            continue
        final_scale = SCALE_FACTOR
        body_id = spawn_asset_with_random_color(
            cli, path, cx, cy, 0, b.facing + 180,
            [final_scale, final_scale, final_scale], rng,
        )
        if _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
            p.removeBody(body_id, physicsClientId=cli)


def _spawn_streetlights(cli, tiles, tile_size, rng, offset,
                       safe_zones=None, safe_zone_radius=0.0):
    half_tile = tile_size / 2
    lamp_offset = half_tile - 0.425
    road_idx = 0
    lamp_freq = 2
    for tile in tiles:
        if tile.type == "intersection":
            continue
        road_idx += 1
        if road_idx % lamp_freq != 0:
            continue
        cx = tile.x + half_tile - offset
        cy = tile.y + half_tile - offset
        scale_mod = 0.8
        s_val = SCALE_FACTOR * scale_mod
        if tile.type == "straight_v":
            path = _get_asset_path("streetlight", rng)
            if path:
                left_x = cx - lamp_offset
                right_x = cx + lamp_offset
                if not _in_safe_zone(left_x, cy, safe_zones, safe_zone_radius):
                    body_id = spawn_asset_exact(cli, path, left_x, cy, 0, 180, [s_val, s_val, s_val])
                    if _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
                        p.removeBody(body_id, physicsClientId=cli)
                if not _in_safe_zone(right_x, cy, safe_zones, safe_zone_radius):
                    body_id = spawn_asset_exact(cli, path, right_x, cy, 0, 0, [s_val, s_val, s_val])
                    if _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
                        p.removeBody(body_id, physicsClientId=cli)
        elif tile.type == "straight_h":
            path = _get_asset_path("streetlight", rng)
            if path:
                down_y = cy - lamp_offset
                up_y = cy + lamp_offset
                if not _in_safe_zone(cx, down_y, safe_zones, safe_zone_radius):
                    body_id = spawn_asset_exact(cli, path, cx, down_y, 0, 270, [s_val, s_val, s_val])
                    if _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
                        p.removeBody(body_id, physicsClientId=cli)
                if not _in_safe_zone(cx, up_y, safe_zones, safe_zone_radius):
                    body_id = spawn_asset_exact(cli, path, cx, up_y, 0, 90, [s_val, s_val, s_val])
                    if _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
                        p.removeBody(body_id, physicsClientId=cli)


def _spawn_traffic_lights(cli, tiles, tile_size, rng, offset,
                         safe_zones=None, safe_zone_radius=0.0):
    half_tile = tile_size / 2
    tl_offset = half_tile - 0.425
    scale_mod = 0.8
    s_val = SCALE_FACTOR * scale_mod
    for tile in tiles:
        if tile.type != "crossing":
            continue
        cx = tile.x + half_tile - offset
        cy = tile.y + half_tile - offset
        path = _get_asset_path("traffic_light", rng)
        if not path:
            continue
        if tile.rotation == 0:
            right_x = cx + tl_offset
            left_x = cx - tl_offset
            if not _in_safe_zone(right_x, cy, safe_zones, safe_zone_radius):
                body_id = spawn_asset_exact(cli, path, right_x, cy, 0, 270, [s_val, s_val, s_val])
                if _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
                    p.removeBody(body_id, physicsClientId=cli)
            if not _in_safe_zone(left_x, cy, safe_zones, safe_zone_radius):
                body_id = spawn_asset_exact(cli, path, left_x, cy, 0, 90, [s_val, s_val, s_val])
                if _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
                    p.removeBody(body_id, physicsClientId=cli)
        elif tile.rotation == 90:
            down_y = cy - tl_offset
            up_y = cy + tl_offset
            if not _in_safe_zone(cx, down_y, safe_zones, safe_zone_radius):
                body_id = spawn_asset_exact(cli, path, cx, down_y, 0, 0, [s_val, s_val, s_val])
                if _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
                    p.removeBody(body_id, physicsClientId=cli)
            if not _in_safe_zone(cx, up_y, safe_zones, safe_zone_radius):
                body_id = spawn_asset_exact(cli, path, cx, up_y, 0, 180, [s_val, s_val, s_val])
                if _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
                    p.removeBody(body_id, physicsClientId=cli)


def _spawn_cars(cli, tiles, tile_size, rng, offset,
                safe_zones=None, safe_zone_radius=0.0):
    lane_offset_factor = 0.18
    scale_mod = 0.25
    s_val = SCALE_FACTOR * scale_mod
    half_tile = tile_size / 2
    spawned_tiles = set()
    for tile in tiles:
        if tile.type not in ["straight_v", "straight_h"]:
            continue
        tile_key = (tile.x, tile.y)
        if tile_key in spawned_tiles:
            continue
        if rng.next_float() >= 0.35:
            continue
        spawned_tiles.add(tile_key)
        car_type = rng.choice(CAR_TYPES)
        path = _get_asset_path(car_type, rng)
        if not path:
            continue
        direction = rng.choice([1, -1])
        tile_offset = tile_size * lane_offset_factor
        cx = tile.x + half_tile - offset
        cy = tile.y + half_tile - offset
        car_x, car_y = cx, cy
        car_rot = 0
        if tile.type == "straight_v":
            if direction == 1:
                car_x = cx + tile_offset
                car_rot = 180
            else:
                car_x = cx - tile_offset
                car_rot = 0
        elif tile.type == "straight_h":
            if direction == 1:
                car_y = cy - tile_offset
                car_rot = 90
            else:
                car_y = cy + tile_offset
                car_rot = 270
        if _in_safe_zone(car_x, car_y, safe_zones, safe_zone_radius):
            continue
        body_id = spawn_asset_exact(cli, path, car_x, car_y, 0, car_rot, [s_val, s_val, s_val])
        if _body_intersects_safe_zone(cli, body_id, safe_zones, safe_zone_radius):
            p.removeBody(body_id, physicsClientId=cli)


def _spawn_trees(cli, buildings, rng, offset):
    for b in buildings:
        if b.type != "park":
            continue
        rect = b.rect
        cx = rect.x + rect.w / 2 - offset
        cy = rect.y + rect.h / 2 - offset
        area = rect.w * rect.h
        s_val = SCALE_FACTOR
        if area < 100:
            path = _get_asset_path("tree", rng)
            if path:
                spawn_asset_exact(
                    cli, path, cx, cy, 0,
                    rng.range(0, 360), [s_val, s_val, s_val],
                )
        else:
            num_trees = int(area / 50)
            for _ in range(num_trees):
                tx = rng.range(rect.x + 2, rect.x + rect.w - 2) - offset
                ty = rng.range(rect.y + 2, rect.y + rect.h - 2) - offset
                path = _get_asset_path("tree", rng)
                if path:
                    spawn_asset_exact(
                        cli, path, tx, ty, 0,
                        rng.range(0, 360), [s_val, s_val, s_val],
                    )


# ---------------------------------------------------------------------------
# SECTION 7: Public API
# ---------------------------------------------------------------------------
def _pick_city_variant(rng):
    r = rng.next_float()
    cumulative = 0.0
    for variant_id, prob in CITY_VARIANT_DISTRIBUTION.items():
        cumulative += prob
        if r < cumulative:
            if variant_id == 4:
                return (3, 3)
            return (variant_id, 1)
    return (3, 1)


def build_city(cli: int, seed: int, safe_zones: list, safe_zone_radius: float) -> None:
    global _shape_cache
    _shape_cache = {}

    if not ASSET_SPECS["house"]:
        _init_specs()

    rng = SeededRNG(seed + 111111)
    variant = _pick_city_variant(rng)
    city_type, difficulty = variant

    configure_templates()
    city_data = generate_city(
        seed=seed, city_type=city_type, difficulty=difficulty,
    )

    offset = MAP_SIZE / 2
    _spawn_grass(cli, city_data["blocks"], offset)
    _spawn_roads(
        cli,
        city_data["roads"],
        TILE_SIZE,
        rng,
        offset,
        safe_zones=safe_zones,
        safe_zone_radius=safe_zone_radius,
    )
    _spawn_buildings(
        cli, city_data["buildings"], city_data["blocks"],
        city_type, TILE_SIZE, rng, safe_zones, safe_zone_radius,
        difficulty, offset,
    )
    _spawn_streetlights(
        cli,
        city_data["roads"],
        TILE_SIZE,
        rng,
        offset,
        safe_zones=safe_zones,
        safe_zone_radius=safe_zone_radius,
    )
    _spawn_traffic_lights(
        cli,
        city_data["roads"],
        TILE_SIZE,
        rng,
        offset,
        safe_zones=safe_zones,
        safe_zone_radius=safe_zone_radius,
    )
    _spawn_cars(
        cli,
        city_data["roads"],
        TILE_SIZE,
        rng,
        offset,
        safe_zones=safe_zones,
        safe_zone_radius=safe_zone_radius,
    )
    _spawn_trees(cli, city_data["buildings"], rng, offset)
