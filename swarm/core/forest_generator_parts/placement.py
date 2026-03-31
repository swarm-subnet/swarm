"""Placement and instance selection for forest generation."""

from ._shared import *
from .assets import *
from .geometry import *


# ---------------------------------------------------------------------------
# SECTION 3: Spatial grid for placement collision
# ---------------------------------------------------------------------------
class _SpatialGrid:
    __slots__ = ('cells', 'cell_size', 'max_radius')

    def __init__(self, cell_size: float = 4.0):
        self.cells: dict = {}
        self.cell_size = cell_size
        self.max_radius = 0.0

    def _key(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x // self.cell_size), int(y // self.cell_size))

    def insert(self, x: float, y: float, radius: float) -> None:
        if radius > self.max_radius:
            self.max_radius = radius
        self.cells.setdefault(self._key(x, y), []).append((x, y, radius))

    def has_conflict(self, x: float, y: float, radius: float, clearance: float) -> bool:
        if not self.cells:
            return False
        search_dist = radius + self.max_radius + clearance
        search_cells = int(search_dist / self.cell_size) + 1
        cx, cy = self._key(x, y)
        for dx in range(-search_cells, search_cells + 1):
            for dy in range(-search_cells, search_cells + 1):
                cell = self.cells.get((cx + dx, cy + dy))
                if cell is None:
                    continue
                for ox, oy, orad in cell:
                    ddx = x - ox
                    ddy = y - oy
                    min_dist = radius + orad + clearance
                    if (ddx * ddx + ddy * ddy) < (min_dist * min_dist):
                        return True
        return False

    def count_neighbors(self, x: float, y: float, search_radius: float) -> int:
        if not self.cells:
            return 0
        search_cells = int(search_radius / self.cell_size) + 1
        cx, cy = self._key(x, y)
        search_r2 = search_radius * search_radius
        count = 0
        for dx in range(-search_cells, search_cells + 1):
            for dy in range(-search_cells, search_cells + 1):
                cell = self.cells.get((cx + dx, cy + dy))
                if cell is None:
                    continue
                for ox, oy, _ in cell:
                    ddx = x - ox
                    ddy = y - oy
                    if (ddx * ddx + ddy * ddy) <= search_r2:
                        count += 1
        return count

# ---------------------------------------------------------------------------
# SECTION 10: Instance picking (placement logic)
# ---------------------------------------------------------------------------
def _tree_candidate_points(
    rng: random.Random, count: int
) -> List[Tuple[float, float]]:
    half = _map_half_extent() - FOREST_EDGE_MARGIN_M
    width = half * 2.0
    cells_side = max(8, int(math.ceil(math.sqrt(count * 1.18))))
    cell = width / cells_side
    jitter = cell * 0.33
    points: List[Tuple[float, float]] = []
    for gy in range(cells_side):
        row_offset = 0.5 * cell if (gy % 2) else 0.0
        for gx in range(cells_side):
            x = -half + (gx + 0.5) * cell + row_offset
            if x > half:
                x -= width
            y = -half + (gy + 0.5) * cell
            x += rng.uniform(-jitter, jitter)
            y += rng.uniform(-jitter, jitter)
            x = max(-half, min(half, x))
            y = max(-half, min(half, y))
            points.append((x, y))
    rng.shuffle(points)
    for _ in range(count * 3):
        points.append((rng.uniform(-half, half), rng.uniform(-half, half)))
    return points


def _small_asset_half_extent() -> float:
    return _map_half_extent() - max(FOREST_EDGE_MARGIN_M, SMALL_ASSET_EDGE_MARGIN_M)


def _normalize_safe_zone_circles(
    safe_zones: Optional[List[Tuple[float, float, float]]],
    safe_zone_radius: float,
) -> List[Tuple[float, float, float]]:
    if not safe_zones or safe_zone_radius <= 0.0:
        return []

    circles: List[Tuple[float, float, float]] = []
    for zone in safe_zones:
        if zone is None or len(zone) < 2:
            continue
        circles.append((float(zone[0]), float(zone[1]), float(safe_zone_radius)))
    return circles


def _safe_zone_rects(
    safe_zone_circles: List[Tuple[float, float, float]]
) -> List[Tuple[float, float, float, float]]:
    return [_circle_bounds_rect(x, y, radius) for x, y, radius in safe_zone_circles]


def _circle_conflicts_safe_zones(
    x: float,
    y: float,
    radius: float,
    safe_zone_circles: List[Tuple[float, float, float]],
) -> bool:
    for zx, zy, zradius in safe_zone_circles:
        dx = x - zx
        dy = y - zy
        min_dist = radius + zradius
        if (dx * dx + dy * dy) < (min_dist * min_dist):
            return True
    return False


def _rect_conflicts_safe_zones(
    rect: Tuple[float, float, float, float],
    safe_zone_rects: List[Tuple[float, float, float, float]],
) -> bool:
    return any(_rect_overlap(rect, safe_rect) for safe_rect in safe_zone_rects)


def _extend_small_asset_candidates(
    rng: random.Random,
    candidates: List[Tuple[float, float]],
    *, count: int, half: float,
) -> None:
    inner_half = half * SMALL_ASSET_CENTER_REGION_RATIO
    for _ in range(count):
        if rng.random() < SMALL_ASSET_CENTER_BIAS:
            x = rng.uniform(-inner_half, inner_half)
            y = rng.uniform(-inner_half, inner_half)
        else:
            x = rng.uniform(-half, half)
            y = rng.uniform(-half, half)
        candidates.append((x, y))


def _scaled_occupied_instances(
    instances: List[Tuple[float, float, str, str, float, float]],
    *, radius_scale: float, min_radius: float = 0.0,
    max_radius: Optional[float] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    scaled: List[Tuple[float, float, str, str, float, float]] = []
    for x, y, category, obj_name, total_scale, radius in instances:
        out_r = max(min_radius, radius * radius_scale)
        if max_radius is not None:
            out_r = min(out_r, max_radius)
        scaled.append((x, y, category, obj_name, total_scale, out_r))
    return scaled


def _pick_tree_instances(
    rng: random.Random, *, count: int,
    assets: List[Tuple[str, str]], clearance_m: float, difficulty_id: int,
    safe_zone_circles: Optional[List[Tuple[float, float, float]]] = None,
    safe_zone_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    placed: List[Tuple[float, float, str, str, float, float]] = []
    placed_families: List[Tuple[float, float, str]] = []
    grid = _SpatialGrid()
    placed_base_rects: List[Tuple[float, float, float, float]] = []
    placed_span_rects: List[Tuple[float, float, float, float]] = []
    candidates = _tree_candidate_points(rng, count)
    half = _map_half_extent() - FOREST_EDGE_MARGIN_M
    family_assets = _build_tree_family_assets(assets)
    family_counts: Dict[str, int] = {f: 0 for f in family_assets}
    non_birch = [f for f in family_assets if f != BIRCH_TREE_PREFIX]
    max_birch = int(math.floor(count * BIRCH_TREE_MAX_RATIO)) if non_birch else count
    birch_placed = 0
    cluster_max = TREE_LOCAL_CLUSTER_MAX_BY_DIFFICULTY.get(difficulty_id, 1)
    cluster_search_m = TREE_LOCAL_CLUSTER_SEARCH_BY_DIFFICULTY_M.get(difficulty_id, 3.0)
    canopy_overlap_scale = TREE_CANOPY_OVERLAP_SCALE_BY_DIFFICULTY.get(
        difficulty_id, 1.0
    )
    safe_zone_circles = safe_zone_circles or []
    safe_zone_rects = safe_zone_rects or []

    for zx, zy, zradius in safe_zone_circles:
        grid.insert(zx, zy, zradius)

    for relax in (1.0, 0.85, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            ranked = _rank_tree_families_for_point(
                rng, x=x, y=y, difficulty_id=difficulty_id,
                family_assets=family_assets, family_counts=family_counts,
                placed_families=placed_families,
                max_birch_count=max_birch, birch_placed=birch_placed,
            )
            if not ranked:
                continue

            for family in ranked:
                info = family_assets.get(family)
                if not info:
                    continue
                category, obj_name = _pick_weighted_tree_from_family(rng, info)
                obj_path = os.path.join(FOREST_ASSET_DIR, category, obj_name)
                if not os.path.exists(obj_path):
                    continue

                size_mul = rng.uniform(TREE_SCALE_MIN, TREE_SCALE_MAX)
                total_scale = PREVIEW_UNIFORM_SCALE * size_mul
                if FAST_BUILD_MODE and FAST_SCALE_STEP > 0.0:
                    total_scale = max(
                        0.01,
                        round(total_scale / FAST_SCALE_STEP) * FAST_SCALE_STEP,
                    )
                canopy_r = _obj_planar_radius_cached(obj_path) * total_scale
                spacing_r = _tree_spacing_radius(obj_name, canopy_r, difficulty_id)
                occupancy_r = _tree_occupancy_radius(obj_name, canopy_r)
                rects = _tree_dual_rects_for_scale(obj_path, total_scale)
                if rects is not None:
                    base_local, span_local = rects
                    base_world = _shift_rect(base_local, x, y)
                    span_world = _shift_rect(span_local, x, y)
                    if (
                        base_world[0] < -half or base_world[1] > half
                        or base_world[2] < -half or base_world[3] > half
                    ):
                        continue
                    if safe_zone_rects and (
                        _rect_conflicts_safe_zones(base_world, safe_zone_rects)
                        or _rect_conflicts_safe_zones(span_world, safe_zone_rects)
                    ):
                        continue
                    if (
                        difficulty_id in (2, 3)
                        and obj_name in LOW_CANOPY_PROTECTED_TREE_NAMES
                    ):
                        span_collision = span_world
                    else:
                        span_collision = _shrink_rect_from_center(
                            span_world,
                            max(0.05, canopy_overlap_scale * relax),
                        )
                    blocked = False
                    for i in range(len(placed_base_rects)):
                        if _rect_overlap(base_world, placed_base_rects[i]):
                            blocked = True
                            break
                        if _rect_overlap(span_collision, placed_span_rects[i]):
                            blocked = True
                            break
                    if blocked:
                        continue
                else:
                    if (
                        (x - occupancy_r) < -half or (x + occupancy_r) > half
                        or (y - occupancy_r) < -half or (y + occupancy_r) > half
                    ):
                        continue
                    if _circle_conflicts_safe_zones(
                        x,
                        y,
                        max(occupancy_r, spacing_r),
                        safe_zone_circles,
                    ):
                        continue
                    clearance_factor = TREE_TREE_CLEARANCE_FACTOR_BY_DIFFICULTY.get(
                        difficulty_id, TREE_TREE_CLEARANCE_FACTOR,
                    )
                    if grid.has_conflict(
                        x, y, spacing_r, clearance_m * relax * clearance_factor
                    ):
                        continue

                local_neighbors = grid.count_neighbors(
                    x, y, max(cluster_search_m, spacing_r * 2.2),
                )
                if local_neighbors > (cluster_max + (1 if relax < 0.9 else 0)):
                    continue

                placed.append((x, y, category, obj_name, total_scale, occupancy_r))
                placed_families.append((x, y, family))
                family_counts[family] = family_counts.get(family, 0) + 1
                grid.insert(x, y, spacing_r)
                if rects is not None:
                    placed_base_rects.append(base_world)
                    placed_span_rects.append(span_collision)
                if obj_name.startswith(BIRCH_TREE_PREFIX):
                    birch_placed += 1
                break

        if len(placed) >= count:
            break
        rng.shuffle(candidates)

    return placed


def _pick_shrub_instances(
    rng: random.Random, *, count: int,
    assets: List[Tuple[str, str]], clearance_m: float,
    tree_instances: List[Tuple[float, float, str, str, float, float]],
    tree_base_rects: Optional[List[tuple]] = None,
    protected_tree_span_rects: Optional[List[tuple]] = None,
    occupied_instances: Optional[List[Tuple[float, float, str, str, float, float]]] = None,
    tree_occupancy_scale: float = 1.0,
    tree_occupancy_cap_m: Optional[float] = None,
    safe_zone_circles: Optional[List[Tuple[float, float, float]]] = None,
    safe_zone_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    half = _small_asset_half_extent()
    weighted: List[Tuple[float, str, str]] = []
    total_w = 0.0
    for cat, name in assets:
        w = BUSH_BERRIES_WEIGHT if name.startswith(BUSH_BERRIES_PREFIX) else 1.0
        if w <= 0.0:
            continue
        total_w += w
        weighted.append((total_w, cat, name))
    if not weighted:
        return []

    candidates: List[Tuple[float, float]] = list(
        _tree_candidate_points(rng, max(count * 2, 120))
    )
    for tx, ty, _, _, _, tr in tree_instances:
        for _ in range(2):
            ang = rng.uniform(0.0, math.tau)
            dist = tr + rng.uniform(1.1, 3.2)
            nx = tx + math.cos(ang) * dist
            ny = ty + math.sin(ang) * dist
            if -half <= nx <= half and -half <= ny <= half:
                candidates.append((nx, ny))
    _extend_small_asset_candidates(rng, candidates, count=count * 8, half=half)
    rng.shuffle(candidates)

    occupied: List[Tuple[float, float, float]] = []
    for ox, oy, _, _, _, orad in tree_instances:
        tr = max(SMALL_ASSET_TREE_OCCUPANCY_MIN_M, orad * tree_occupancy_scale)
        if tree_occupancy_cap_m is not None:
            tr = min(tr, tree_occupancy_cap_m)
        occupied.append((ox, oy, tr))
    if occupied_instances:
        occupied.extend((ox, oy, orad) for ox, oy, _, _, _, orad in occupied_instances)
    if safe_zone_circles:
        occupied.extend(safe_zone_circles)
    grid = _SpatialGrid()
    for ox, oy, orad in occupied:
        grid.insert(ox, oy, orad)
    per_cell: Dict[Tuple[int, int], int] = {}
    placed: List[Tuple[float, float, str, str, float, float]] = []

    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            cell_key = (
                int((x + half) // BUSH_DISTRIBUTION_CELL_SIZE_M),
                int((y + half) // BUSH_DISTRIBUTION_CELL_SIZE_M),
            )
            if per_cell.get(cell_key, 0) >= BUSH_DISTRIBUTION_MAX_PER_CELL:
                continue
            r = rng.uniform(0.0, total_w)
            cat = name = ""
            for cum, c, n in weighted:
                if r <= cum:
                    cat, name = c, n
                    break
            else:
                cat, name = weighted[-1][1], weighted[-1][2]
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            size_mul = rng.uniform(SHRUB_SCALE_MIN, SHRUB_SCALE_MAX)
            total_scale = PREVIEW_UNIFORM_SCALE * size_mul
            radius = _obj_planar_radius_cached(obj_path) * total_scale
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if safe_zone_rects and _rect_conflicts_safe_zones(cr, safe_zone_rects):
                continue
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if grid.has_conflict(x, y, radius, clearance_m * relax):
                continue
            placed.append((x, y, cat, name, total_scale, radius))
            grid.insert(x, y, radius)
            per_cell[cell_key] = per_cell.get(cell_key, 0) + 1
        if len(placed) >= count:
            break
        rng.shuffle(candidates)
    return placed


def _pick_rock_stump_instances(
    rng: random.Random, *, count: int,
    assets: List[Tuple[str, str]], mode_id: int, clearance_m: float,
    occupied_instances: List[Tuple[float, float, str, str, float, float]],
    tree_base_rects: Optional[List[tuple]] = None,
    protected_tree_span_rects: Optional[List[tuple]] = None,
    safe_zone_circles: Optional[List[Tuple[float, float, float]]] = None,
    safe_zone_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    weighted: List[Tuple[float, str, str]] = []
    total_w = 0.0
    mode_wb = ROCK_STUMP_MODE_WEIGHT_BONUS.get(mode_id, {})
    for cat, name in assets:
        w = ROCK_STUMP_MODEL_WEIGHT_BONUS.get(name, 1.0) * mode_wb.get(name, 1.0)
        if w <= 0.0:
            continue
        total_w += w
        weighted.append((total_w, cat, name))
    if not weighted:
        return []

    half = _small_asset_half_extent()
    candidates: List[Tuple[float, float]] = list(
        _tree_candidate_points(rng, max(count * 2, 120))
    )
    _extend_small_asset_candidates(rng, candidates, count=count * 10, half=half)
    rng.shuffle(candidates)

    occupied: List[Tuple[float, float, float]] = [
        (ox, oy, orad) for ox, oy, _, _, _, orad in occupied_instances
    ]
    if safe_zone_circles:
        occupied.extend(safe_zone_circles)
    grid = _SpatialGrid()
    for ox, oy, orad in occupied:
        grid.insert(ox, oy, orad)
    placed: List[Tuple[float, float, str, str, float, float]] = []

    priority_names = [
        n
        for n in ROCK_STUMP_MODEL_WEIGHT_BONUS
        if any(obj_name == n for _, obj_name in assets)
    ]
    for target_name in priority_names:
        if len(placed) >= count:
            break
        target = [(c, n) for c, n in assets if n == target_name]
        if not target:
            continue
        for _ in range(140):
            x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            cat, name = rng.choice(target)
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            sm = rng.uniform(ROCK_STUMP_SCALE_MIN, ROCK_STUMP_SCALE_MAX)
            sm *= ROCK_STUMP_MODEL_SCALE_FACTOR.get(name, 1.0)
            ts = PREVIEW_UNIFORM_SCALE * sm
            radius = _obj_planar_radius_cached(obj_path) * ts
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if safe_zone_rects and _rect_conflicts_safe_zones(cr, safe_zone_rects):
                continue
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if grid.has_conflict(x, y, radius, clearance_m * 0.35):
                continue
            placed.append((x, y, cat, name, ts, radius))
            grid.insert(x, y, radius)
            break

    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            r = rng.uniform(0.0, total_w)
            cat = name = ""
            for cum, c, n in weighted:
                if r <= cum:
                    cat, name = c, n
                    break
            else:
                cat, name = weighted[-1][1], weighted[-1][2]
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            sm = rng.uniform(ROCK_STUMP_SCALE_MIN, ROCK_STUMP_SCALE_MAX)
            sm *= ROCK_STUMP_MODEL_SCALE_FACTOR.get(name, 1.0)
            ts = PREVIEW_UNIFORM_SCALE * sm
            radius = _obj_planar_radius_cached(obj_path) * ts
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if safe_zone_rects and _rect_conflicts_safe_zones(cr, safe_zone_rects):
                continue
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if grid.has_conflict(x, y, radius, clearance_m * relax):
                continue
            placed.append((x, y, cat, name, ts, radius))
            grid.insert(x, y, radius)
        if len(placed) >= count:
            break
        rng.shuffle(candidates)
    return placed


def _pick_log_instances(
    rng: random.Random, *, count: int,
    assets: List[Tuple[str, str]], clearance_m: float,
    occupied_instances: List[Tuple[float, float, str, str, float, float]],
    tree_base_rects: Optional[List[tuple]] = None,
    protected_tree_span_rects: Optional[List[tuple]] = None,
    safe_zone_circles: Optional[List[Tuple[float, float, float]]] = None,
    safe_zone_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    half = _small_asset_half_extent()
    candidates: List[Tuple[float, float]] = list(
        _tree_candidate_points(rng, max(count * 3, 120))
    )
    _extend_small_asset_candidates(rng, candidates, count=count * 12, half=half)
    rng.shuffle(candidates)

    occupied = [(ox, oy, orad) for ox, oy, _, _, _, orad in occupied_instances]
    if safe_zone_circles:
        occupied.extend(safe_zone_circles)
    grid = _SpatialGrid()
    for ox, oy, orad in occupied:
        grid.insert(ox, oy, orad)
    placed: List[Tuple[float, float, str, str, float, float]] = []

    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            cat, name = rng.choice(assets)
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            sm = rng.uniform(LOG_SCALE_MIN, LOG_SCALE_MAX)
            ts = PREVIEW_UNIFORM_SCALE * sm
            radius = _obj_planar_radius_cached(obj_path) * ts
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if safe_zone_rects and _rect_conflicts_safe_zones(cr, safe_zone_rects):
                continue
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if grid.has_conflict(x, y, radius, clearance_m * relax):
                continue
            placed.append((x, y, cat, name, ts, radius))
            grid.insert(x, y, radius)
        if len(placed) >= count:
            break
        rng.shuffle(candidates)
    return placed


def _pick_ground_cover_instances(
    rng: random.Random, *, count: int,
    assets: List[Tuple[str, str]], mode_id: int, clearance_m: float,
    occupied_instances: List[Tuple[float, float, str, str, float, float]],
    tree_base_rects: Optional[List[tuple]] = None,
    tree_span_rects: Optional[List[tuple]] = None,
    protected_tree_span_rects: Optional[List[tuple]] = None,
    safe_zone_circles: Optional[List[Tuple[float, float, float]]] = None,
    safe_zone_rects: Optional[List[Tuple[float, float, float, float]]] = None,
) -> List[Tuple[float, float, str, str, float, float]]:
    if not assets or count <= 0:
        return []

    weighted: List[Tuple[float, str, str]] = []
    total_w = 0.0
    mode_wb = GROUND_COVER_MODE_WEIGHT_BONUS.get(mode_id, {})
    for cat, name in assets:
        w = GROUND_COVER_MODEL_WEIGHT_BONUS.get(name, 1.0) * mode_wb.get(name, 1.0)
        if w <= 0.0:
            continue
        total_w += w
        weighted.append((total_w, cat, name))
    if not weighted:
        return []

    half = _small_asset_half_extent()
    candidates: List[Tuple[float, float]] = list(
        _tree_candidate_points(rng, max(count * 2, 80))
    )
    _extend_small_asset_candidates(rng, candidates, count=count * 12, half=half)
    rng.shuffle(candidates)

    occupied: List[Tuple[float, float, float]] = [
        (ox, oy, orad) for ox, oy, _, _, _, orad in occupied_instances
    ]
    if safe_zone_circles:
        occupied.extend(safe_zone_circles)
    placed: List[Tuple[float, float, str, str, float, float]] = []
    placed_by_name: Dict[str, int] = {}

    priority_names = [
        n for n, w in GROUND_COVER_MODEL_WEIGHT_BONUS.items()
        if w > 1.0 and any(obj_name == n for _, obj_name in assets)
    ]
    for target_name in priority_names:
        if len(placed) >= count:
            break
        target = [(c, n) for c, n in assets if n == target_name]
        if not target:
            continue
        for _ in range(180):
            x, y = rng.uniform(-half, half), rng.uniform(-half, half)
            cat, name = rng.choice(target)
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            sm = rng.uniform(GROUND_COVER_SCALE_MIN, GROUND_COVER_SCALE_MAX)
            sm *= GROUND_COVER_MODEL_SCALE_FACTOR.get(name, 1.0)
            ts = PREVIEW_UNIFORM_SCALE * sm
            radius = _obj_planar_radius_cached(obj_path) * ts
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            if name in NORMAL_ONLY_SINGLE_SPAWN_GROUND_COVER and placed_by_name.get(name, 0) >= 1:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if safe_zone_rects and _rect_conflicts_safe_zones(cr, safe_zone_rects):
                continue
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if name.startswith("Corn_") and tree_span_rects and any(
                _rect_overlap(cr, r) for r in tree_span_rects
            ):
                continue
            keep = True
            for ox, oy, orad in occupied:
                dx, dy = x - ox, y - oy
                min_dist = radius + orad + clearance_m * 0.65
                if (dx * dx + dy * dy) < (min_dist * min_dist):
                    keep = False
                    break
            if not keep:
                continue
            placed.append((x, y, cat, name, ts, radius))
            occupied.append((x, y, radius))
            placed_by_name[name] = placed_by_name.get(name, 0) + 1
            break

    for relax in (1.0, 0.86, 0.72):
        for x, y in candidates:
            if len(placed) >= count:
                break
            r = rng.uniform(0.0, total_w)
            cat = name = ""
            for cum, c, n in weighted:
                if r <= cum:
                    cat, name = c, n
                    break
            else:
                cat, name = weighted[-1][1], weighted[-1][2]
            obj_path = os.path.join(FOREST_ASSET_DIR, cat, name)
            if not os.path.exists(obj_path):
                continue
            sm = rng.uniform(GROUND_COVER_SCALE_MIN, GROUND_COVER_SCALE_MAX)
            sm *= GROUND_COVER_MODEL_SCALE_FACTOR.get(name, 1.0)
            ts = PREVIEW_UNIFORM_SCALE * sm
            radius = _obj_planar_radius_cached(obj_path) * ts
            if (x - radius) < -half or (x + radius) > half or (y - radius) < -half or (y + radius) > half:
                continue
            if name in NORMAL_ONLY_SINGLE_SPAWN_GROUND_COVER and placed_by_name.get(name, 0) >= 1:
                continue
            cr = _circle_bounds_rect(x, y, radius)
            if safe_zone_rects and _rect_conflicts_safe_zones(cr, safe_zone_rects):
                continue
            if tree_base_rects and any(_rect_overlap(cr, r) for r in tree_base_rects):
                continue
            if protected_tree_span_rects and any(
                _rect_overlap(cr, r) for r in protected_tree_span_rects
            ):
                continue
            if name.startswith("Corn_") and tree_span_rects and any(
                _rect_overlap(cr, r) for r in tree_span_rects
            ):
                continue
            keep = True
            for ox, oy, orad in occupied:
                dx, dy = x - ox, y - oy
                min_dist = radius + orad + clearance_m * relax
                if (dx * dx + dy * dy) < (min_dist * min_dist):
                    keep = False
                    break
            if not keep:
                continue
            placed.append((x, y, cat, name, ts, radius))
            occupied.append((x, y, radius))
            placed_by_name[name] = placed_by_name.get(name, 0) + 1
        if len(placed) >= count:
            break
        rng.shuffle(candidates)
    return placed


def _split_asset_count(
    total: int,
    primary: List[Tuple[str, str]],
    secondary: List[Tuple[str, str]],
    *, primary_ratio: Optional[float] = None,
    secondary_cap: Optional[int] = None,
) -> Tuple[int, int]:
    if total <= 0 or (not primary and not secondary):
        return 0, 0
    if not secondary:
        return total, 0
    if not primary:
        sec = total if secondary_cap is None else min(total, max(0, secondary_cap))
        return 0, sec
    if primary_ratio is None:
        n = max(1, len(primary) + len(secondary))
        primary_ratio = len(primary) / n
    pri = max(0, min(total, int(round(total * primary_ratio))))
    sec = total - pri
    if secondary_cap is not None:
        sec = min(sec, max(0, secondary_cap))
        pri = total - sec
    return pri, sec


__all__ = [name for name in globals() if not name.startswith("__")]
