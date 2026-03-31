"""Asset resolution and tree-family heuristics for forest generation."""

from ._shared import *


# ---------------------------------------------------------------------------
# SECTION 8: Asset resolution
# ---------------------------------------------------------------------------
def _list_obj_names(category: str) -> List[str]:
    cat_dir = os.path.join(FOREST_ASSET_DIR, category)
    if not os.path.isdir(cat_dir):
        return []
    return sorted(
        name for name in os.listdir(cat_dir) if name.lower().endswith(".obj")
    )


def _clamp_mode_id(mode_id: int) -> int:
    return max(1, min(4, int(mode_id)))


def _resolve_assets_for_class(mode_id: int) -> Dict[str, List[Tuple[str, str]]]:
    mode_id = _clamp_mode_id(mode_id)
    cache_key = f"mode_{mode_id}"
    cached = _CLASS_ASSET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    mode_cfg = MAP_MODE_CONFIG[mode_id]
    primary_cat = mode_cfg["primary_category"]
    is_dry = primary_cat in ("autumn", "dead", "dead_snow")
    is_snow = primary_cat in ("snow", "dead_snow")

    primary_objs = _list_obj_names(primary_cat)
    normal_objs = _list_obj_names("normal")
    misc_objs = _list_obj_names("misc")

    if is_snow:
        primary_mode_objs = [n for n in primary_objs if "Snow" in n]
    else:
        primary_mode_objs = [n for n in primary_objs if "Snow" not in n]
    normal_nosnow = [n for n in normal_objs if "Snow" not in n]

    trees = [
        (primary_cat, name) for name in primary_mode_objs
        if ("Tree" in name or name.startswith("Willow_"))
        and "Stump" not in name and "Bush" not in name
    ]
    if primary_cat == "snow":
        dead_snow_objs = _list_obj_names("dead_snow")
        trees += [
            ("dead_snow", n) for n in dead_snow_objs
            if "Tree" in n and "Stump" not in n and "Bush" not in n
        ]
    if mode_cfg["use_misc_willow"]:
        trees += [("misc", name) for name in misc_objs if "Willow_" in name]

    bushes = [(primary_cat, n) for n in primary_mode_objs if "Bush" in n]
    rocks = [(primary_cat, n) for n in primary_mode_objs if n.startswith("Rock")]
    stumps = [(primary_cat, n) for n in primary_mode_objs if "TreeStump" in n]
    logs = [(primary_cat, n) for n in primary_mode_objs if "WoodLog" in n]

    if primary_cat == "snow":
        snow_stumps = [("normal", n) for n in normal_objs if n == "TreeStump_Snow.obj"]
        stumps += snow_stumps

    if is_snow:
        plants: List[Tuple[str, str]] = []
        cactus: List[Tuple[str, str]] = []
    else:
        plants = [
            ("misc", n) for n in misc_objs
            if n.startswith("Plant_") or n in {"Flowers.obj"}
        ]
        if mode_id == 1:
            plants += [("misc", n) for n in misc_objs if n in NORMAL_ONLY_GROUND_COVER_ALLOWLIST]
        if mode_id in (1, 2, 4):
            plants += [("misc", n) for n in misc_objs if n in NORMAL_AUTUMN_DEAD_GROUND_COVER_ALLOWLIST]
        cactus = [
            ("misc", n) for n in misc_objs
            if n.startswith("Cactus_") or n.startswith("CactusFlower_")
            or n.startswith("CactusFlowers_")
        ]

    if primary_cat == "autumn":
        plants = [(c, n) for c, n in plants if n != "Flowers.obj"]

    if is_dry:
        bushes = []
        if primary_cat == "autumn":
            plants = [(c, n) for c, n in plants if n in AUTUMN_GROUND_COVER_ALLOWLIST]
        elif primary_cat == "dead":
            plants = [("misc", n) for n in misc_objs if n in DEAD_GROUND_COVER_ALLOWLIST]
        else:
            plants = []
        logs = [(c, n) for c, n in logs if "Moss" not in n]
        rocks = [(c, n) for c, n in rocks if "Moss" not in n]
        stumps = [(c, n) for c, n in stumps if "Moss" not in n]
        cactus = []

    if not is_snow:
        logs = [(c, n) for c, n in logs if "Snow" not in n]

    if not trees:
        trees = [("normal", "CommonTree_1.obj")]
    if not bushes and not is_dry:
        bushes = [("normal", n) for n in normal_nosnow if "Bush" in n]
        if primary_cat == "autumn":
            bushes = [(c, n) for c, n in bushes if not n.startswith(BUSH_BERRIES_PREFIX)]
        if not bushes:
            bushes = [("normal", "Bush_1.obj")]
    if not rocks:
        fallback = [("normal", n) for n in normal_nosnow if n.startswith("Rock")]
        if is_dry:
            fallback = [(c, n) for c, n in fallback if "Moss" not in n]
        rocks = fallback if fallback else [("normal", "Rock_1.obj")]
    if not stumps:
        fallback = [("normal", n) for n in normal_nosnow if "TreeStump" in n]
        if is_dry:
            fallback = [(c, n) for c, n in fallback if "Moss" not in n]
        stumps = fallback if fallback else [("normal", "TreeStump.obj")]
    if not logs:
        normal_logs = [("normal", n) for n in normal_objs if "WoodLog" in n]
        if is_snow:
            snow_logs = [(c, n) for c, n in normal_logs if "Snow" in n]
            logs = snow_logs if snow_logs else normal_logs
        else:
            no_snow = [(c, n) for c, n in normal_logs if "Snow" not in n]
            logs = no_snow if no_snow else normal_logs
        if is_dry:
            logs = [(c, n) for c, n in logs if "Moss" not in n and "Snow" not in n]
        if not logs:
            logs = [("normal", "WoodLog.obj")]

    resolved = {
        "trees": trees,
        "bushes": bushes,
        "logs": logs,
        "rocks": rocks,
        "stumps": stumps,
        "plants": plants,
        "cactus": cactus,
        "rocks_stumps": rocks + stumps,
        "ground_cover": plants + cactus,
    }
    _CLASS_ASSET_CACHE[cache_key] = resolved
    return resolved


# ---------------------------------------------------------------------------
# SECTION 9: Tree family selection helpers
# ---------------------------------------------------------------------------
def _map_half_extent() -> float:
    return GROUND_SIZE_M * 0.5


def _tree_spacing_radius(
    obj_name: str, canopy_radius: float, difficulty_id: int
) -> float:
    mul = TREE_SPACING_RADIUS_MULTIPLIER_DEFAULT_BY_DIFFICULTY.get(
        difficulty_id, TREE_SPACING_RADIUS_MULTIPLIER_DEFAULT
    )
    by_prefix = TREE_SPACING_RADIUS_BY_PREFIX_BY_DIFFICULTY.get(difficulty_id, {})
    for prefix, ratio in by_prefix.items():
        if obj_name.startswith(prefix):
            mul = ratio
            break
    overlap_scale = TREE_CANOPY_OVERLAP_SCALE_BY_DIFFICULTY.get(difficulty_id, 1.0)
    return max(TREE_SPACING_RADIUS_MIN_M, canopy_radius * mul * overlap_scale)


def _tree_occupancy_radius(obj_name: str, canopy_radius: float) -> float:
    mul = TREE_OCCUPANCY_RADIUS_MULTIPLIER_DEFAULT
    for prefix, ratio in TREE_OCCUPANCY_RADIUS_BY_PREFIX.items():
        if obj_name.startswith(prefix):
            mul = ratio
            break
    return max(TREE_OCCUPANCY_RADIUS_MIN_M, canopy_radius * mul)


def _tree_family_prefix(obj_name: str) -> str:
    for prefix in TREE_FAMILY_PREFIXES:
        if obj_name.startswith(prefix):
            return prefix
    return obj_name.split("_", 1)[0]


def _build_tree_family_assets(
    assets: List[Tuple[str, str]],
) -> Dict[str, dict]:
    families: Dict[str, dict] = {}
    for category, obj_name in assets:
        family = _tree_family_prefix(obj_name)
        info = families.setdefault(
            family, {"total_weight": 0.0, "weighted_assets": []}
        )
        weight = SNOW_DEAD_TREE_WEIGHT if category == "dead_snow" else 1.0
        if weight <= 0.0:
            continue
        info["total_weight"] += weight
        info["weighted_assets"].append((info["total_weight"], category, obj_name))
    return families


def _pick_weighted_tree_from_family(
    rng: random.Random, family_info: dict
) -> Tuple[str, str]:
    weighted = family_info.get("weighted_assets", [])
    total = float(family_info.get("total_weight", 0.0))
    if not weighted or total <= 0.0:
        raise RuntimeError("tree family has no weighted assets")
    r = rng.uniform(0.0, total)
    for cum_w, category, obj_name in weighted:
        if r <= cum_w:
            return category, obj_name
    return weighted[-1][1], weighted[-1][2]


def _count_tree_family_neighbors(
    placed_families: List[Tuple[float, float, str]],
    *, x: float, y: float, family: str, radius_m: float,
) -> int:
    radius_sq = radius_m * radius_m
    count = 0
    for ox, oy, other_family in placed_families:
        if other_family != family:
            continue
        dx, dy = x - ox, y - oy
        if (dx * dx + dy * dy) <= radius_sq:
            count += 1
    return count


def _rank_tree_families_for_point(
    rng: random.Random,
    *, x: float, y: float, difficulty_id: int,
    family_assets: Dict[str, dict],
    family_counts: Dict[str, int],
    placed_families: List[Tuple[float, float, str]],
    max_birch_count: int, birch_placed: int,
) -> List[str]:
    ranked: List[Tuple[float, str]] = []
    for family, info in family_assets.items():
        if family == BIRCH_TREE_PREFIX and birch_placed >= max_birch_count:
            continue
        base_weight = float(info.get("total_weight", 0.0))
        if base_weight <= 0.0:
            continue
        repeat_penalty = 1.0 / (
            1.0 + family_counts.get(family, 0) * TREE_FAMILY_REPEAT_PENALTY
        )
        nearby_count = _count_tree_family_neighbors(
            placed_families, x=x, y=y, family=family,
            radius_m=max(
                6.0,
                TREE_LOCAL_CLUSTER_SEARCH_BY_DIFFICULTY_M.get(difficulty_id, 3.0)
                * 1.8,
            ),
        )
        score = (
            base_weight
            * repeat_penalty
            / (1.0 + nearby_count * TREE_FAMILY_NEARBY_PENALTY)
        )
        cluster_rule = TREE_FAMILY_CLUSTER_RULES.get(family, {}).get(difficulty_id)
        if cluster_rule is not None:
            same_neighbors = _count_tree_family_neighbors(
                placed_families, x=x, y=y, family=family,
                radius_m=cluster_rule["radius_m"],
            )
            if same_neighbors > cluster_rule["max_neighbors"]:
                continue
            score /= 1.0 + same_neighbors * 0.75
        score *= rng.uniform(0.85, 1.15)
        ranked.append((score, family))
    ranked.sort(reverse=True)
    return [family for _score, family in ranked if _score > 0.0]


__all__ = [name for name in globals() if not name.startswith("__")]
